// Copyright 2024 The XLS Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "xls/passes/lut_conversion_pass.h"

#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/interpreter/ir_interpreter.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/ternary.h"
#include "xls/ir/topo_sort.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/passes/dataflow_dominator_analysis.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/stateless_query_engine.h"
#include "xls/passes/ternary_query_engine.h"
#include "xls/passes/union_query_engine.h"

namespace xls {

namespace {

bool IsTriviallyDerived(Node* node, Node* ancestor) {
  static constexpr auto is_trivial_array_index = [](Node* node) {
    if (!node->Is<ArrayIndex>()) {
      return false;
    }
    return absl::c_all_of(node->As<ArrayIndex>()->indices(),
                          [](Node* index) { return index->Is<Literal>(); });
  };
  while (node != ancestor && (node->OpIn({Op::kTupleIndex, Op::kBitSlice}) ||
                              is_trivial_array_index(node))) {
    node = node->operand(0);
  }
  if (node->Is<Literal>()) {
    return true;
  }
  if (node != ancestor && node->Is<Concat>()) {
    return absl::c_all_of(node->operands(), [&](Node* operand) {
      return IsTriviallyDerived(operand, ancestor);
    });
  }
  return node == ancestor;
}

absl::StatusOr<bool> MaybeMergeLutIntoSelect(
    Select* select, const QueryEngine& query_engine, int64_t opt_level,
    const DataflowDominatorAnalysis& dataflow_dominator_analysis) {
  Node* selector = select->selector();
  absl::Span<Node* const> dominators =
      dataflow_dominator_analysis.GetDominatorsOfNode(selector);

  // We favor the dominator with the fewest unknown bits, since this minimizes
  // the risk that we need too many additional pipeline flops on the path to
  // the resulting LUT; we break ties by preferring the most distant
  // dominator, which should usually reduce both area & delay as much as
  // possible.
  //
  // NOTE: The dominators are topologically sorted, so we iterate from most to
  // least distant.
  VLOG(4) << "Looking for earlier selector: " << selector->ToString();
  int64_t original_bits_needed = Bits::MinBitCountUnsigned(
      select->default_value().has_value() ? select->cases().size()
                                          : select->cases().size() - 1);
  Node* best_dominator = nullptr;
  std::optional<SharedLeafTypeTree<TernaryVector>> best_dominator_ternary;
  for (Node* dominator : dominators) {
    std::optional<SharedLeafTypeTree<TernaryVector>> dominator_ternary =
        query_engine.GetTernary(dominator);
    int64_t unknown_bits =
        dominator_ternary.has_value()
            ? absl::c_accumulate(
                  dominator_ternary->elements(), 0,
                  [](int64_t sum, const TernaryVector& ternary) {
                    return sum + absl::c_count_if(ternary, [](TernaryValue v) {
                             return ternary_ops::IsUnknown(v);
                           });
                  })
            : dominator->GetType()->GetFlatBitCount();

    // For now, only consider dominators that will not result in a wider
    // selector than the original. If the selector is "trivially" derived from
    // the dominator (via tuple index, bit slice, array index with literal
    // indices, and concat), we only consider it if the selector will be
    // *strictly* narrower.
    if (unknown_bits <= original_bits_needed &&
        (!IsTriviallyDerived(selector, dominator) ||
         unknown_bits < original_bits_needed)) {
      best_dominator = dominator;
      if (dominator_ternary.has_value()) {
        best_dominator_ternary = std::move(dominator_ternary);
      }
      VLOG(3) << "Found earlier selector with " << unknown_bits
              << " unknown bits (original: " << selector->BitCountOrDie()
              << " bits, " << original_bits_needed
              << " needed): " << dominator->ToString();
      break;
    }
  }
  if (best_dominator == nullptr || best_dominator == selector) {
    // There's no better alternative; this selector is already optimal.
    return false;
  }
  VLOG(2) << "Merging a lookup table into a select: " << select->ToString();

  if (!best_dominator_ternary.has_value()) {
    best_dominator_ternary =
        LeafTypeTree<TernaryVector>::CreateFromFunction(
            best_dominator->GetType(),
            [](Type* leaf_type,
               absl::Span<const int64_t>) -> absl::StatusOr<TernaryVector> {
              return TernaryVector(leaf_type->GetFlatBitCount(),
                                   TernaryValue::kUnknown);
            })
            .value()
            .AsShared();
  }
  XLS_ASSIGN_OR_RETURN(
      std::vector<Value> dominator_values,
      ternary_ops::AllValues(best_dominator_ternary->AsView()));
  XLS_RET_CHECK(!dominator_values.empty());

  // Populate an interpreter with all known values that feed into the
  // selector.
  IrInterpreter base_interpreter;
  std::vector<Node*> to_visit({selector});
  absl::flat_hash_set<Node*> visited;
  while (!to_visit.empty()) {
    Node* n = to_visit.back();
    to_visit.pop_back();
    if (visited.contains(n) || base_interpreter.IsVisited(n)) {
      continue;
    }
    if (std::optional<Value> known_value = query_engine.KnownValue(n);
        known_value.has_value()) {
      XLS_RETURN_IF_ERROR(base_interpreter.SetValueResult(n, *known_value));
      base_interpreter.MarkVisited(n);
    } else {
      absl::c_copy(n->operands(), std::back_inserter(to_visit));
      visited.insert(n);
    }
  }

  std::vector<Node*> new_cases;
  new_cases.reserve(dominator_values.size());
  for (const Value& dominator_value : dominator_values) {
    // Invoke an interpreter using known values & this `dominator_value` to
    // compute the value of the selector.
    IrInterpreter interpreter = base_interpreter;
    if (interpreter.IsVisited(best_dominator)) {
      // It seems the dominator is actually fully-known!
      XLS_RET_CHECK_EQ(interpreter.ResolveAsValue(best_dominator),
                       dominator_value);
    } else {
      XLS_RETURN_IF_ERROR(
          interpreter.SetValueResult(best_dominator, dominator_value));
      interpreter.MarkVisited(best_dominator);
    }
    XLS_RETURN_IF_ERROR(selector->Accept(&interpreter));

    Value selector_value = interpreter.ResolveAsValue(selector);
    XLS_RET_CHECK(selector_value.IsBits());
    new_cases.push_back(
        bits_ops::ULessThan(selector_value.bits(), select->cases().size())
            ? select->get_case(
                  static_cast<int64_t>(*selector_value.bits().ToUint64()))
            : *select->default_value());
  }

  if (new_cases.size() == 1) {
    // We've proven that only one case is ever selected; just use that
    // directly.
    XLS_RETURN_IF_ERROR(select->ReplaceUsesWith(new_cases.front()));
    return true;
  }

  // Assemble the new selector out of the unknown bits of the dominator.
  LeafTypeTree<Bits> unknown_positions_ltt =
      leaf_type_tree::Map<Bits, TernaryVector>(
          best_dominator_ternary->AsView(),
          [&](const TernaryVector& ternary) -> Bits {
            return bits_ops::Not(ternary_ops::ToKnownBits(ternary));
          });
  XLS_ASSIGN_OR_RETURN(
      Node * new_selector,
      GatherBits(best_dominator, unknown_positions_ltt.AsView()));

  XLS_RETURN_IF_ERROR(
      select
          ->ReplaceUsesWithNew<Select>(new_selector, new_cases,
                                       /*default_value=*/std::nullopt)
          .status());
  return true;
}

absl::StatusOr<bool> SimplifyNode(
    Node* node, const QueryEngine& query_engine, int64_t opt_level,
    const DataflowDominatorAnalysis& dataflow_dominator_analysis) {
  if (node->Is<Select>()) {
    XLS_ASSIGN_OR_RETURN(
        bool changed_select_incorporating_lut,
        MaybeMergeLutIntoSelect(node->As<Select>(), query_engine, opt_level,
                                dataflow_dominator_analysis));
    if (changed_select_incorporating_lut) {
      return true;
    }
  }

  return false;
}

}  // namespace

absl::StatusOr<bool> LutConversionPass::RunOnFunctionBaseInternal(
    FunctionBase* func, const OptimizationPassOptions& options,
    PassResults* results) const {
  if (!options.narrowing_enabled()) {
    return false;
  }

  std::vector<std::unique_ptr<QueryEngine>> query_engines;
  query_engines.push_back(std::make_unique<StatelessQueryEngine>());
  query_engines.push_back(std::make_unique<TernaryQueryEngine>());

  UnionQueryEngine query_engine(std::move(query_engines));
  XLS_RETURN_IF_ERROR(query_engine.Populate(func).status());

  XLS_ASSIGN_OR_RETURN(DataflowDominatorAnalysis dataflow_dominator_analysis,
                       DataflowDominatorAnalysis::Run(func));

  bool changed = false;
  // By running in reverse topological order, the analyses will stay valid for
  // all nodes we're considering through the full pass.
  for (Node* node : ReverseTopoSort(func)) {
    if (node->IsDead()) {
      continue;
    }
    XLS_ASSIGN_OR_RETURN(bool node_changed,
                         SimplifyNode(node, query_engine, options.opt_level,
                                      dataflow_dominator_analysis));
    changed = changed || node_changed;
  }
  return changed;
}

REGISTER_OPT_PASS(LutConversionPass);

}  // namespace xls
