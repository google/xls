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
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/fixed_array.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
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
#include "xls/ir/value.h"
#include "xls/passes/dataflow_graph_analysis.h"
#include "xls/passes/lazy_ternary_query_engine.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/stateless_query_engine.h"
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

absl::StatusOr<std::optional<LutConversionCandidate>> GetLutCandidate(
    Node* node, const QueryEngine& query_engine,
    std::optional<DataflowGraphAnalysis>& dataflow_graph_analysis) {
  int64_t max_case_count = 0;
  std::vector<Node*> candidate_users;
  candidate_users.reserve(node->users().size());
  for (Node* user : node->users()) {
    if (user->Is<Select>() && user->As<Select>()->selector() == node) {
      candidate_users.push_back(user);
      max_case_count = std::max(max_case_count, CaseCount(user->As<Select>()));
    }
  }
  if (candidate_users.empty()) {
    return std::nullopt;
  }

  // Find the minimum set of unknown bits that fully determine the value of the
  // selector; we can treat the selector as defined by a LUT, then merge it into
  // the select(s) it controls by reordering cases.
  int64_t unknown_bits = 0;
  int64_t max_bits_needed = Bits::MinBitCountUnsigned(max_case_count - 1);
  if (max_bits_needed <= 1) {
    // We can't narrow the controlled selects any further unless the selector is
    // actually constant... which should be handled by other (cheaper) passes.
    return std::nullopt;
  }

  // Initialize the graph analysis if not done already.
  if (!dataflow_graph_analysis.has_value()) {
    XLS_ASSIGN_OR_RETURN(
        DataflowGraphAnalysis dataflow_analysis,
        DataflowGraphAnalysis::Create(node->function_base(), &query_engine));
    dataflow_graph_analysis.emplace(std::move(dataflow_analysis));
  }

  VLOG(3) << "Finding min cut for " << node->GetName() << " ("
          << max_bits_needed << " bits needed)" << " controlling "
          << candidate_users.size() << " select(s)";
  XLS_ASSIGN_OR_RETURN(
      std::vector<Node*> min_cut,
      dataflow_graph_analysis->GetMinCutFor(
          node, /*max_unknown_bits=*/max_bits_needed, &unknown_bits));
  if (min_cut.empty()) {
    // There's no better alternative; this selector is already optimal.
    return std::nullopt;
  }
  VLOG(3) << "Found " << unknown_bits << "-bit min cut for " << node->GetName()
          << ": "
          << absl::StrJoin(min_cut, ", ", [](std::string* out, Node* node) {
               absl::StrAppend(out, node->GetName());
             });

  // Remove all candidate selects that wouldn't benefit from this transform.
  const bool selector_is_trivial = IsTriviallyDerived(
      node, absl::flat_hash_set<Node*>(min_cut.begin(), min_cut.end()));
  std::erase_if(candidate_users, [&](Node* select) {
    int64_t bits_needed =
        Bits::MinBitCountUnsigned(CaseCount(select->As<Select>()) - 1);
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
  if (candidate_users.empty()) {
    return std::nullopt;
  }

  return LutConversionCandidate{
      .node = node,
      .min_cut = std::move(min_cut),
      .candidate_users = std::move(candidate_users),
      .min_cut_unknown_bits = unknown_bits,
  };
}

absl::StatusOr<bool> MergeLutIntoSelects(
    const LutConversionCandidate& candidate, const QueryEngine& query_engine) {
  std::vector<SharedLeafTypeTree<TernaryVector>> cut_ternaries;
  cut_ternaries.reserve(candidate.min_cut.size());
  for (size_t i = 0; i < candidate.min_cut.size(); ++i) {
    Node* cut_node = candidate.min_cut[i];
    std::optional<SharedLeafTypeTree<TernaryVector>> ternary =
        query_engine.GetTernary(cut_node);
    VLOG(4) << "Ternary for cut node " << cut_node->GetName() << ": "
            << ternary->ToString(
                   [](TernarySpan span) { return ToString(span); });
    cut_ternaries.push_back(*std::move(ternary));
  }

  VLOG(2) << "Merging a " << candidate.min_cut_unknown_bits
          << "-bit lookup table into its controlled selects: "
          << absl::StrJoin(candidate.candidate_users, ", ",
                           [](std::string* out, Node* user) {
                             absl::StrAppend(out, user->GetName());
                           });
  if (VLOG_IS_ON(3)) {
    for (Node* user : candidate.candidate_users) {
      VLOG(3) << "- " << user->ToString();
    }
  }

  XLS_ASSIGN_OR_RETURN(std::vector<Bits> new_case_sequence,
                       LutConversionPass::ComputeLutSelectCases(
                           candidate, cut_ternaries, query_engine));

  if (absl::c_all_of(new_case_sequence, [&](const Bits& index) {
        return index == new_case_sequence.front();
      })) {
    // We've proven that only one case is ever selected; just use that
    // directly.
    for (Node* user : candidate.candidate_users) {
      XLS_RETURN_IF_ERROR(user->ReplaceUsesWith(
          GetCase(user->As<Select>(), new_case_sequence.front())));
    }
    return true;
  }

  // Assemble the new selector out of the unknown bits of the min-cut nodes.
  std::vector<Node*> selector_pieces;
  selector_pieces.reserve(candidate.min_cut.size());
  for (size_t i = 0; i < candidate.min_cut.size(); ++i) {
    LeafTypeTree<Bits> unknown_positions_ltt =
        leaf_type_tree::Map<Bits, TernaryVector>(
            cut_ternaries[i].AsView(),
            [&](const TernaryVector& ternary) -> Bits {
              return bits_ops::Not(ternary_ops::ToKnownBits(ternary));
            });
    XLS_ASSIGN_OR_RETURN(
        Node * new_selector_piece,
        GatherBits(candidate.min_cut[i], unknown_positions_ltt.AsView()));
    selector_pieces.push_back(new_selector_piece);
  }

  Node* new_selector;
  XLS_RET_CHECK(!selector_pieces.empty());
  if (selector_pieces.size() == 1) {
    new_selector = selector_pieces.front();
  } else {
    // `ComputeLutSelectCases` uses earlier min_cut nodes' values as
    // lower-ordered bits when determining case values, so we need to reverse
    // the order of selector pieces to match the concat's ordering with
    // `new_case_sequence`.
    absl::c_reverse(selector_pieces);
    XLS_ASSIGN_OR_RETURN(new_selector,
                         candidate.node->function_base()->MakeNode<Concat>(
                             candidate.node->loc(), selector_pieces));
  }

  for (Node* user : candidate.candidate_users) {
    absl::FixedArray<Node*> new_cases(new_case_sequence.size());
    for (size_t i = 0; i < new_case_sequence.size(); ++i) {
      new_cases[i] = GetCase(user->As<Select>(), new_case_sequence[i]);
    }
    XLS_ASSIGN_OR_RETURN(
        Node * new_select,
        user->ReplaceUsesWithNew<Select>(new_selector, new_cases,
                                         /*default_value=*/std::nullopt));
    VLOG(3) << "Replaced " << user->GetName()
            << " with: " << new_select->ToString();
  }
  return true;
}

}  // namespace

absl::StatusOr<std::vector<LutConversionCandidate>>
LutConversionPass::ComputeLutConversionCandidates(
    FunctionBase* func, const QueryEngine& query_engine,
    OptimizationContext& context) {
  std::optional<DataflowGraphAnalysis> dataflow_graph_analysis;

  // By running in reverse topological order, the analyses will stay valid for
  // all nodes we're considering through the full pass.
  XLS_ASSIGN_OR_RETURN(std::vector<Node*> reverse_topo_sort_nodes,
                       context.ReverseTopoSort(func));

  std::vector<LutConversionCandidate> candidates;
  for (Node* node : reverse_topo_sort_nodes) {
    if (node->IsDead()) {
      continue;
    }
    XLS_ASSIGN_OR_RETURN(
        std::optional<LutConversionCandidate> candidate,
        GetLutCandidate(node, query_engine, dataflow_graph_analysis));
    if (candidate.has_value()) {
      candidates.push_back(std::move(candidate.value()));
    }
  }

  return candidates;
}

absl::StatusOr<std::vector<Bits>> LutConversionPass::ComputeLutSelectCases(
    const LutConversionCandidate& candidate,
    std::vector<SharedLeafTypeTree<TernaryVector>>& cut_ternaries,
    const QueryEngine& query_engine) {
  // Populate an interpreter with all known values that feed into the
  // node.
  IrInterpreter base_interpreter;
  std::vector<Node*> to_visit({candidate.node});
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

  std::vector<std::vector<Value>> cut_values(candidate.min_cut.size());
  for (size_t i = 0; i < candidate.min_cut.size(); ++i) {
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
        // min-cut to compute the value of the node.
        IrInterpreter interpreter = base_interpreter;
        for (size_t i = 0; i < value_indices.size(); ++i) {
          Node* cut_node = candidate.min_cut[i];
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
        status.Update(candidate.node->Accept(&interpreter));
        if (!status.ok()) {
          return true;
        }

        Value selector_value = interpreter.ResolveAsValue(candidate.node);
        CHECK(selector_value.IsBits());
        new_case_sequence.push_back(std::move(selector_value).bits());
        return false;
      });
  XLS_RETURN_IF_ERROR(status);
  XLS_RET_CHECK_EQ(new_case_sequence.size(), new_case_count);
  return new_case_sequence;
}

RedundancyGuard LutConversionPass::GetRedundancyGuard(
    const OptimizationPassOptions& options,
    OptimizationContext& context) const {
  return RedundancyGuard::CanSkip(absl::StrFormat("O%d", options.opt_level));
}

absl::StatusOr<bool> LutConversionPass::RunOnFunctionBaseInternal(
    FunctionBase* func, const OptimizationPassOptions& options,
    PassResults* results, OptimizationContext& context) const {
  if (!options.narrowing_enabled()) {
    return false;
  }

  auto query_engine = UnionQueryEngine::Of(
      StatelessQueryEngine(),
      GetSharedQueryEngine<LazyTernaryQueryEngine>(context, func));
  XLS_RETURN_IF_ERROR(query_engine.Populate(func).status());

  std::optional<DataflowGraphAnalysis> dataflow_graph_analysis;

  XLS_ASSIGN_OR_RETURN(
      std::vector<LutConversionCandidate> lut_conversion_candidates,
      LutConversionPass::ComputeLutConversionCandidates(func, query_engine,
                                                        context));

  bool changed = false;

  for (const LutConversionCandidate& candidate : lut_conversion_candidates) {
    XLS_ASSIGN_OR_RETURN(bool changed_at_node,
                         MergeLutIntoSelects(candidate, query_engine));
    changed = changed || changed_at_node;
  }

  return changed;
}

}  // namespace xls
