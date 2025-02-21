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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/fixed_array.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/math_util.h"
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
#include "xls/ir/value.h"
#include "xls/passes/dataflow_graph_analysis.h"
#include "xls/passes/lazy_ternary_query_engine.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/stateless_query_engine.h"
#include "xls/passes/ternary_query_engine.h"
#include "xls/passes/union_query_engine.h"

namespace xls {

namespace {

bool IsTriviallyDerived(Node* node, absl::flat_hash_set<Node*> ancestors) {
  static constexpr auto is_trivial_array_index = [](Node* node) {
    if (!node->Is<ArrayIndex>()) {
      return false;
    }
    return absl::c_all_of(node->As<ArrayIndex>()->indices(),
                          [](Node* index) { return index->Is<Literal>(); });
  };
  while (!ancestors.contains(node) &&
         (node->OpIn({Op::kTupleIndex, Op::kBitSlice}) ||
          is_trivial_array_index(node))) {
    node = node->operand(0);
  }
  if (ancestors.contains(node)) {
    return true;
  }
  if (node->Is<Literal>()) {
    return true;
  }
  if (node->Is<Concat>()) {
    return absl::c_all_of(node->operands(), [&](Node* operand) {
      return IsTriviallyDerived(operand, ancestors);
    });
  }
  return false;
}

int64_t CaseCount(Select* select) {
  if (select->default_value().has_value()) {
    return select->cases().size() + 1;
  }
  return select->cases().size();
}

Node* GetCase(Select* select, const Bits& selector) {
  if (bits_ops::UGreaterThanOrEqual(selector, select->cases().size())) {
    CHECK(select->default_value().has_value());
    return *select->default_value();
  }
  absl::StatusOr<uint64_t> selector_value = selector.ToUint64();
  CHECK_OK(selector_value.status());
  return select->get_case(static_cast<int64_t>(*selector_value));
}

absl::StatusOr<bool> MaybeMergeLutIntoSelects(
    Node* selector, const QueryEngine& query_engine, int64_t opt_level,
    std::optional<DataflowGraphAnalysis>& dataflow_graph_analysis) {
  int64_t max_case_count = 0;
  std::vector<Select*> candidate_selects;
  candidate_selects.reserve(1);
  for (Node* user : selector->users()) {
    if (user->Is<Select>() && user->As<Select>()->selector() == selector) {
      candidate_selects.push_back(user->As<Select>());
      max_case_count = std::max(max_case_count, CaseCount(user->As<Select>()));
    }
  }
  if (candidate_selects.empty()) {
    return false;
  }
  CHECK(!candidate_selects.empty());

  // Find the minimum set of unknown bits that fully determine the value of the
  // selector; we can treat the selector as defined by a LUT, then merge it into
  // the select(s) it controls by reordering cases.
  int64_t unknown_bits = 0;
  int64_t max_bits_needed = Bits::MinBitCountUnsigned(max_case_count - 1);
  if (max_bits_needed <= 1) {
    // We can't narrow the controlled selects any further unless the selector is
    // actually constant... which should be handled by other (cheaper) passes.
    return false;
  }

  // Initialize the graph analysis if not done already.
  if (!dataflow_graph_analysis.has_value()) {
    dataflow_graph_analysis.emplace(selector->function_base(), &query_engine);
  }

  VLOG(3) << "Finding min cut for " << selector->GetName() << " ("
          << max_bits_needed << " bits needed)" << " controlling "
          << candidate_selects.size() << " select(s)";
  XLS_ASSIGN_OR_RETURN(
      std::vector<Node*> min_cut,
      dataflow_graph_analysis->GetMinCutFor(
          selector, /*max_unknown_bits=*/max_bits_needed, &unknown_bits));
  if (min_cut.empty()) {
    // There's no better alternative; this selector is already optimal.
    return false;
  }
  VLOG(3) << "Found " << unknown_bits << "-bit min cut for "
          << selector->GetName() << ": "
          << absl::StrJoin(min_cut, ", ", [](std::string* out, Node* node) {
               absl::StrAppend(out, node->GetName());
             });

  // Remove all candidate selects that wouldn't benefit from this transform.
  const bool selector_is_trivial = IsTriviallyDerived(
      selector, absl::flat_hash_set<Node*>(min_cut.begin(), min_cut.end()));
  std::erase_if(candidate_selects, [&](Select* select) {
    int64_t bits_needed = Bits::MinBitCountUnsigned(CaseCount(select) - 1);
    if (unknown_bits < bits_needed) {
      // This transform will narrow this select.
      return false;
    }
    if (unknown_bits == bits_needed && !selector_is_trivial) {
      // This transform will keep this select approximately the same width, but
      // should save delay through the selector.
      return false;
    }
    // Without a way to tell whether this transform is still beneficial, we
    // can't confidently use this optimization.
    //
    // TODO(epastor): Use delay & area estimators to check for net benefit.
    return true;
  });
  if (candidate_selects.empty()) {
    return false;
  }

  std::vector<SharedLeafTypeTree<TernaryVector>> cut_ternaries;
  cut_ternaries.reserve(min_cut.size());
  for (size_t i = 0; i < min_cut.size(); ++i) {
    Node* cut_node = min_cut[i];
    std::optional<SharedLeafTypeTree<TernaryVector>> ternary =
        query_engine.GetTernary(cut_node);
    VLOG(4) << "Ternary for cut node " << cut_node->GetName() << ": "
            << ternary->ToString(
                   [](TernarySpan span) { return ToString(span); });
    cut_ternaries.push_back(*std::move(ternary));
  }

  VLOG(2) << "Merging a " << unknown_bits
          << "-bit lookup table into its controlled selects: "
          << absl::StrJoin(candidate_selects, ", ",
                           [](std::string* out, Select* select) {
                             absl::StrAppend(out, select->GetName());
                           });
  if (VLOG_IS_ON(3)) {
    for (Select* candidate : candidate_selects) {
      VLOG(3) << "- " << candidate->ToString();
    }
  }

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

  std::vector<std::vector<Value>> cut_values(min_cut.size());
  for (size_t i = 0; i < min_cut.size(); ++i) {
    XLS_ASSIGN_OR_RETURN(cut_values[i],
                         ternary_ops::AllValues(cut_ternaries[i].AsView()));
    XLS_RET_CHECK(!cut_values[i].empty());
  }

  int64_t new_case_count = 1;
  std::vector<int64_t> values_radix;
  values_radix.reserve(cut_values.size());
  for (const std::vector<Value>& cut_value : cut_values) {
    new_case_count *= cut_value.size();
    values_radix.push_back(cut_value.size());
  }

  std::vector<Bits> new_case_sequence;
  new_case_sequence.reserve(new_case_count);
  absl::Status status = absl::OkStatus();
  MixedRadixIterate(
      values_radix, [&](const std::vector<int64_t>& value_indices) {
        // Invoke an interpreter using known values & these values on the
        // min-cut to compute the value of the selector.
        IrInterpreter interpreter = base_interpreter;
        for (size_t i = 0; i < value_indices.size(); ++i) {
          Node* cut_node = min_cut[i];
          int64_t value_index = value_indices[i];
          const Value& cut_value = cut_values[i][value_index];
          if (interpreter.IsVisited(cut_node)) {
            // It seems this cut node is actually fully-known!
            if (const Value& resolved_value =
                    interpreter.ResolveAsValue(cut_node);
                resolved_value != cut_value) {
              status.Update(absl::InternalError(absl::StrFormat(
                  "Cut node %s has different value in interpreter (%s) than "
                  "expected (%s)",
                  cut_node->ToString(), resolved_value.ToString(),
                  cut_value.ToString())));
              return true;
            }
          } else {
            status.Update(interpreter.SetValueResult(cut_node, cut_value));
            if (!status.ok()) {
              return true;
            }
            interpreter.MarkVisited(cut_node);
          }
        }
        status.Update(selector->Accept(&interpreter));
        if (!status.ok()) {
          return true;
        }

        Value selector_value = interpreter.ResolveAsValue(selector);
        CHECK(selector_value.IsBits());
        new_case_sequence.push_back(std::move(selector_value).bits());
        return false;
      });
  XLS_RETURN_IF_ERROR(status);
  XLS_RET_CHECK_EQ(new_case_sequence.size(), new_case_count);

  if (absl::c_all_of(new_case_sequence, [&](const Bits& index) {
        return index == new_case_sequence.front();
      })) {
    // We've proven that only one case is ever selected; just use that
    // directly.
    for (Select* select : candidate_selects) {
      XLS_RETURN_IF_ERROR(
          select->ReplaceUsesWith(GetCase(select, new_case_sequence.front())));
    }
    return true;
  }

  // Assemble the new selector out of the unknown bits of the min-cut nodes.
  std::vector<Node*> selector_pieces;
  selector_pieces.reserve(min_cut.size());
  for (size_t i = 0; i < min_cut.size(); ++i) {
    LeafTypeTree<Bits> unknown_positions_ltt =
        leaf_type_tree::Map<Bits, TernaryVector>(
            cut_ternaries[i].AsView(),
            [&](const TernaryVector& ternary) -> Bits {
              return bits_ops::Not(ternary_ops::ToKnownBits(ternary));
            });
    XLS_ASSIGN_OR_RETURN(
        Node * new_selector_piece,
        GatherBits(min_cut[i], unknown_positions_ltt.AsView()));
    selector_pieces.push_back(new_selector_piece);
  }

  Node* new_selector;
  XLS_RET_CHECK(!selector_pieces.empty());
  if (selector_pieces.size() == 1) {
    new_selector = selector_pieces.front();
  } else {
    // Concat assumes big-endian order.
    absl::c_reverse(selector_pieces);
    XLS_ASSIGN_OR_RETURN(new_selector,
                         selector->function_base()->MakeNode<Concat>(
                             selector->loc(), selector_pieces));
  }

  for (Select* select : candidate_selects) {
    absl::FixedArray<Node*> new_cases(new_case_count);
    for (int64_t i = 0; i < new_case_count; ++i) {
      new_cases[i] = GetCase(select, new_case_sequence[i]);
    }
    XLS_ASSIGN_OR_RETURN(
        Node * new_select,
        select->ReplaceUsesWithNew<Select>(new_selector, new_cases,
                                           /*default_value=*/std::nullopt));
    VLOG(3) << "Replaced " << select->GetName()
            << " with: " << new_select->ToString();
  }
  return true;
}

absl::StatusOr<bool> SimplifyNode(
    Node* node, const QueryEngine& query_engine, int64_t opt_level,
    std::optional<DataflowGraphAnalysis>& dataflow_graph_analysis) {
  XLS_ASSIGN_OR_RETURN(bool changed_select_incorporating_lut,
                       MaybeMergeLutIntoSelects(node, query_engine, opt_level,
                                                dataflow_graph_analysis));
  if (changed_select_incorporating_lut) {
    return true;
  }

  return false;
}

}  // namespace

absl::StatusOr<bool> LutConversionPass::RunOnFunctionBaseInternal(
    FunctionBase* func, const OptimizationPassOptions& options,
    PassResults* results, OptimizationContext* context) const {
  if (!options.narrowing_enabled()) {
    return false;
  }

  auto query_engine = UnionQueryEngine::Of(
      StatelessQueryEngine(),
      GetSharedQueryEngine<LazyTernaryQueryEngine>(context, func));
  XLS_RETURN_IF_ERROR(query_engine.Populate(func).status());

  std::optional<DataflowGraphAnalysis> dataflow_graph_analysis;

  bool changed = false;
  // By running in reverse topological order, the analyses will stay valid for
  // all nodes we're considering through the full pass.
  for (Node* node : ReverseTopoSort(func)) {
    if (node->IsDead()) {
      continue;
    }
    XLS_ASSIGN_OR_RETURN(bool changed_at_node,
                         SimplifyNode(node, query_engine, options.opt_level,
                                      dataflow_graph_analysis));
    changed = changed || changed_at_node;
  }
  return changed;
}

REGISTER_OPT_PASS(LutConversionPass);

}  // namespace xls
