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
#include <queue>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
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
#include "xls/estimators/area_model/area_estimator.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/interpreter/ir_interpreter.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/source_location.h"
#include "xls/ir/ternary.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/dataflow_graph_analysis.h"
#include "xls/passes/lazy_ternary_query_engine.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/stateless_query_engine.h"
#include "xls/passes/union_query_engine.h"

namespace xls {

namespace {

constexpr int64_t kDefaultMaxUnknownBits = 10;
constexpr double kAreaTolerance = 1e-2;

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

absl::StatusOr<Node*> GetLutIndex(
    Node* node, absl::Span<Node* const> cut,
    absl::Span<const std::optional<SharedLeafTypeTree<TernaryVector>>>
        cut_ternaries) {
  XLS_RET_CHECK_EQ(cut.size(), cut_ternaries.size());
  std::vector<Node*> index_pieces;
  index_pieces.reserve(cut.size());
  for (size_t i = 0; i < cut.size(); ++i) {
    Node* index_piece;
    if (cut_ternaries[i].has_value()) {
      LeafTypeTree<Bits> unknown_positions_ltt =
          leaf_type_tree::Map<Bits, TernaryVector>(
              cut_ternaries[i]->AsView(),
              [&](const TernaryVector& ternary) -> Bits {
                return bits_ops::Not(ternary_ops::ToKnownBits(ternary));
              });
      XLS_ASSIGN_OR_RETURN(index_piece,
                           GatherBits(cut[i], unknown_positions_ltt.AsView()));
    } else {
      XLS_ASSIGN_OR_RETURN(index_piece, GatherBits(cut[i], std::nullopt));
    }
    index_pieces.push_back(index_piece);
  }

  Node* index;
  XLS_RET_CHECK(!index_pieces.empty());
  if (index_pieces.size() == 1) {
    index = index_pieces.front();
  } else {
    // Concat assumes big-endian order.
    absl::c_reverse(index_pieces);
    XLS_ASSIGN_OR_RETURN(index, node->function_base()->MakeNode<Concat>(
                                    node->loc(), index_pieces));
  }
  return index;
}

absl::StatusOr<std::vector<Value>> GetLutCases(
    Node* node, absl::Span<Node* const> cut, const QueryEngine& query_engine) {
  // Populate an interpreter with all known values that feed into the
  // selector.
  IrInterpreter base_interpreter;
  std::vector<Node*> to_visit({node});
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

  std::vector<std::vector<Value>> cut_values(cut.size());
  for (size_t i = 0; i < cut.size(); ++i) {
    Node* cut_node = cut[i];
    std::optional<SharedLeafTypeTree<TernaryVector>> ternary =
        query_engine.GetTernary(cut_node);
    VLOG(4) << "Ternary for cut node " << cut_node->GetName() << ": "
            << ternary->ToString(
                   [](TernarySpan span) { return ToString(span); });
    XLS_ASSIGN_OR_RETURN(cut_values[i],
                         ternary_ops::AllValues(ternary->AsView()));
    XLS_RET_CHECK(!cut_values[i].empty());
  }

  int64_t new_case_count = 1;
  std::vector<int64_t> values_radix;
  values_radix.reserve(cut_values.size());
  for (const std::vector<Value>& cut_value : cut_values) {
    new_case_count *= cut_value.size();
    values_radix.push_back(cut_value.size());
  }

  std::vector<Value> lut_values;
  lut_values.reserve(new_case_count);
  absl::Status status = absl::OkStatus();
  MixedRadixIterate(
      values_radix, [&](const std::vector<int64_t>& value_indices) {
        // Invoke an interpreter using known values & these values on the
        // min-cut to compute the value of the selector.
        IrInterpreter interpreter = base_interpreter;
        for (size_t i = 0; i < value_indices.size(); ++i) {
          Node* cut_node = cut[i];
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
        status.Update(node->Accept(&interpreter));
        if (!status.ok()) {
          return true;
        }

        Value node_value = interpreter.ResolveAsValue(node);
        lut_values.push_back(interpreter.ResolveAsValue(node));
        return false;
      });
  XLS_RETURN_IF_ERROR(status);
  XLS_RET_CHECK_EQ(lut_values.size(), new_case_count);
  return lut_values;
}

// Compute the area we would be replacing if the given node were replaced,
// counting all nodes that would become dead if the node were removed (unless
// they're in the min-cut set).
absl::StatusOr<double> ComputeReplaceableArea(
    Node* node, absl::flat_hash_set<Node*> min_cut,
    AreaEstimator& area_estimator) {
  double saved_area = 0.0;
  absl::flat_hash_set<Node*> visited = {};
  std::queue<Node*> to_visit({node});
  while (!to_visit.empty()) {
    Node* replaced = to_visit.back();
    to_visit.pop();
    if (min_cut.contains(replaced)) {
      continue;
    }
    auto [_, inserted] = visited.insert(replaced);
    if (!inserted) {
      continue;
    }
    XLS_ASSIGN_OR_RETURN(
        double replaced_area,
        area_estimator.GetOperationAreaInSquareMicrons(replaced));
    saved_area += replaced_area;
    for (Node* operand : replaced->operands()) {
      if (!node->function_base()->HasImplicitUse(operand) &&
          absl::c_all_of(operand->users(), [&](Node* operand_user) {
            return visited.contains(operand_user);
          })) {
        to_visit.push(operand);
      }
    }
  }
  return saved_area;
}

// Compute the potential delay that would be replaced if we replaced the given
// node; this amounts to the critical-path delay from the min-cut set to the
// output of the given node.
absl::StatusOr<int64_t> ComputeSavedDelay(
    Node* node, absl::flat_hash_set<Node*> min_cut,
    DataflowGraphAnalysis& dataflow_graph_analysis,
    DelayEstimator& delay_estimator) {
  int64_t saved_delay = 0;
  absl::flat_hash_map<Node*, int64_t> delay_to_node;
  std::vector<Node*> to_visit({node});
  while (!to_visit.empty()) {
    Node* n = to_visit.back();
    to_visit.pop_back();
    XLS_RET_CHECK(!delay_to_node.contains(n))
        << "Node visited twice: " << n->ToString();
    int64_t critical_path_delay = 0;
    for (Node* user : n->users()) {
      if (delay_to_node.contains(user)) {
        critical_path_delay =
            std::max(critical_path_delay, delay_to_node.at(user));
      }
    }
    if (min_cut.contains(n)) {
      saved_delay = std::max(saved_delay, critical_path_delay);
      continue;
    }
    XLS_ASSIGN_OR_RETURN(const int64_t n_delay,
                         delay_estimator.GetOperationDelayInPs(n));
    delay_to_node.emplace(n, critical_path_delay + n_delay);
    if (min_cut.contains(n)) {
      continue;
    }
    absl::flat_hash_set<Node*> unique_operands(n->operands().begin(),
                                               n->operands().end());
    for (Node* operand : unique_operands) {
      if (dataflow_graph_analysis.GetUnknownBitsThrough(node, operand)
              .value_or(0) <= 0) {
        // This operand doesn't participate in the flow of unknown bits to
        // `node`.
        continue;
      }
      if (absl::c_all_of(operand->users(), [&](Node* operand_user) {
            if (dataflow_graph_analysis.GetUnknownBitsThrough(node, operand)
                    .value_or(0) <= 0) {
              // This user doesn't participate in the flow of unknown bits to
              // `node`; we can ignore it.
              return true;
            }
            return delay_to_node.contains(operand_user);
          })) {
        to_visit.push_back(operand);
      }
    }
  }
  return saved_delay;
}

absl::StatusOr<bool> MaybeMergeLutIntoSelects(
    Node* selector, const QueryEngine& query_engine, int64_t opt_level,
    std::optional<DataflowGraphAnalysis>& dataflow_graph_analysis,
    DelayEstimator* delay_estimator, AreaEstimator* area_estimator) {
  FunctionBase* f = selector->function_base();

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

  // Initialize the graph analysis if not done already.
  if (!dataflow_graph_analysis.has_value()) {
    dataflow_graph_analysis.emplace(selector->function_base(), &query_engine);
  }

  // Find the minimum set of unknown bits that fully determine the value of the
  // selector; we can treat the selector as defined by a LUT, then merge it into
  // the select(s) it controls by reordering cases.
  int64_t unknown_bits = 0;
  XLS_ASSIGN_OR_RETURN(
      std::vector<Node*> min_cut,
      dataflow_graph_analysis->GetMinCutFor(
          selector, /*max_unknown_bits=*/
          std::max(Bits::MinBitCountUnsigned(max_case_count - 1),
                   kDefaultMaxUnknownBits),
          &unknown_bits));
  if (min_cut.empty()) {
    // There's no better alternative; this selector is already optimal.
    return false;
  }
  VLOG(3) << "Found " << unknown_bits << "-bit min cut for "
          << selector->GetName() << ": "
          << absl::StrJoin(min_cut, ", ", [](std::string* out, Node* node) {
               absl::StrAppend(out, node->GetName());
             });
  absl::flat_hash_set<Node*> min_cut_set(min_cut.begin(), min_cut.end());

  // Remove all candidate selects that wouldn't benefit from this transform.
  double replaceable_area = 0.0;
  int64_t saved_delay = 0;
  const bool selector_is_trivial = IsTriviallyDerived(selector, min_cut_set);
  if (!selector_is_trivial && area_estimator != nullptr) {
    XLS_ASSIGN_OR_RETURN(
        replaceable_area,
        ComputeReplaceableArea(selector, min_cut_set, *area_estimator));
  }
  if (!selector_is_trivial && delay_estimator != nullptr) {
    XLS_ASSIGN_OR_RETURN(
        saved_delay,
        ComputeSavedDelay(selector, min_cut_set, *dataflow_graph_analysis,
                          *delay_estimator));
  }

  std::vector<Node*> placeholders;
  absl::Cleanup cleanup_placeholders = [&] {
    for (auto it = placeholders.rbegin(); it != placeholders.rend(); ++it) {
      CHECK_OK(f->RemoveNode(*it));
    }
  };

  XLS_ASSIGN_OR_RETURN(
      Node * lut_index_placeholder,
      f->MakeNode<Literal>(SourceInfo(), Value(UBits(0, unknown_bits))));
  placeholders.push_back(lut_index_placeholder);
  XLS_ASSIGN_OR_RETURN(
      Node * lut_index,
      f->MakeNode<UnOp>(SourceInfo(), lut_index_placeholder, Op::kIdentity));

  absl::flat_hash_map<Select*, Select*> new_selects;
  new_selects.reserve(candidate_selects.size());

  struct NotImproved : public std::monostate {};
  struct Improved : public std::monostate {};
  struct DeltaPPA {
    double net_area;
    int64_t net_delay;
  };
  absl::flat_hash_map<Select*, std::variant<NotImproved, Improved, DeltaPPA>>
      delta_ppa;
  delta_ppa.reserve(candidate_selects.size());

  for (Select* select : candidate_selects) {
    XLS_ASSIGN_OR_RETURN(
        Node * case_placeholder,
        f->MakeNode<Literal>(SourceInfo(), ZeroOfType(select->GetType())));
    placeholders.push_back(case_placeholder);
    std::vector<Node*> cases;
    int64_t case_count = int64_t{1} << unknown_bits;
    cases.reserve(case_count);
    for (int64_t i = 0; i < case_count; ++i) {
      XLS_ASSIGN_OR_RETURN(
          Node * case_node,
          f->MakeNode<UnOp>(SourceInfo(), case_placeholder, Op::kIdentity));
      cases.push_back(case_node);
    }
    XLS_ASSIGN_OR_RETURN(Select * new_select,
                         f->MakeNode<Select>(select->loc(), lut_index, cases,
                                             /*default_value=*/std::nullopt));
    new_selects.insert({select, new_select});

    if (area_estimator == nullptr || delay_estimator == nullptr) {
      int64_t bits_needed = Bits::MinBitCountUnsigned(CaseCount(select) - 1);
      if (unknown_bits < bits_needed) {
        // This transform will narrow this select.
        VLOG(3) << "Narrowing select " << select->GetName() << " from "
                << bits_needed << " bits to " << unknown_bits << " bits";
        delta_ppa.insert({select, Improved()});
      } else if (unknown_bits == bits_needed && !selector_is_trivial) {
        // This transform will keep this select approximately the same width,
        // but should save delay through the selector.
        VLOG(3) << "Skipping unnecessary selector transforms for "
                << select->GetName();
        delta_ppa.insert({select, Improved()});
      } else {
        // Without a way to tell whether this transform is still beneficial, we
        // can't confidently use this optimization.
        delta_ppa.insert({select, NotImproved()});
      }
      continue;
    }

    XLS_ASSIGN_OR_RETURN(
        double new_area,
        area_estimator->GetOperationAreaInSquareMicrons(new_select));
    XLS_ASSIGN_OR_RETURN(
        double old_area,
        area_estimator->GetOperationAreaInSquareMicrons(select));
    double net_area = new_area - old_area;

    XLS_ASSIGN_OR_RETURN(int64_t new_delay,
                         delay_estimator->GetOperationDelayInPs(new_select));
    XLS_ASSIGN_OR_RETURN(int64_t old_delay,
                         delay_estimator->GetOperationDelayInPs(select));
    int64_t net_delay = new_delay - old_delay;

    delta_ppa.insert(
        {select, DeltaPPA{.net_area = net_area, .net_delay = net_delay}});
  }

  enum class MergeStrategy {
    kMergeAll,
    kMergeSome,
    kNone,
  };
  MergeStrategy merge_strategy = MergeStrategy::kMergeSome;
  if (area_estimator != nullptr && delay_estimator != nullptr &&
      !selector_is_trivial) {
    double full_replacement_net_area = -replaceable_area;
    int64_t full_replacement_net_delay = -saved_delay;
    double local_replacement_net_area = 0.0;
    int64_t local_replacement_net_delay = 0;

    full_replacement_net_area = -replaceable_area;
    full_replacement_net_delay = -saved_delay;
    for (const auto& [select, change] : delta_ppa) {
      CHECK(std::holds_alternative<DeltaPPA>(change));
      const DeltaPPA& delta = std::get<DeltaPPA>(change);

      full_replacement_net_area += delta.net_area;
      full_replacement_net_delay += delta.net_delay;

      if (delta.net_area < 0 && delta.net_delay < 0) {
        local_replacement_net_area += delta.net_area;
        local_replacement_net_delay += delta.net_delay;
      }
    }

    auto is_beneficial = [](double net_area, int64_t net_delay) {
      return (net_area <= kAreaTolerance) && (net_delay <= 0) &&
             (net_area < 0.0 || net_delay < 0);
    };
    const bool local_replacement_benefits =
        is_beneficial(local_replacement_net_area, local_replacement_net_delay);
    const bool full_replacement_benefits =
        is_beneficial(full_replacement_net_area, full_replacement_net_delay);

    if (!local_replacement_benefits && !full_replacement_benefits) {
      merge_strategy = MergeStrategy::kNone;
    } else if (full_replacement_net_area <= local_replacement_net_area &&
               full_replacement_net_delay <= local_replacement_net_delay) {
      merge_strategy = MergeStrategy::kMergeAll;
    } else {
      merge_strategy = MergeStrategy::kMergeSome;
    }
  }

  auto remove_new_select = [&](Select* select) -> absl::Status {
    Select* new_select = new_selects.at(select);
    std::vector<Node*> cases(new_select->cases().begin(),
                             new_select->cases().end());
    XLS_RETURN_IF_ERROR(f->RemoveNode(new_select));
    for (Node* case_node : cases) {
      XLS_RETURN_IF_ERROR(f->RemoveNode(case_node));
    }
    return absl::OkStatus();
  };

  if (merge_strategy == MergeStrategy::kNone) {
    for (Select* select : candidate_selects) {
      XLS_RETURN_IF_ERROR(remove_new_select(select));
    }
    XLS_RETURN_IF_ERROR(f->RemoveNode(lut_index));
    return false;
  }

  if (merge_strategy == MergeStrategy::kMergeSome) {
    std::erase_if(candidate_selects, [&](Select* select) {
      if (std::holds_alternative<DeltaPPA>(delta_ppa.at(select))) {
        const DeltaPPA& delta = std::get<DeltaPPA>(delta_ppa.at(select));
        if (delta.net_area >= 0 || delta.net_delay >= 0) {
          CHECK_OK(remove_new_select(select));
          return true;
        }
      }
      if (std::holds_alternative<NotImproved>(delta_ppa.at(select))) {
        CHECK_OK(remove_new_select(select));
        return true;
      }
      return false;
    });
    if (candidate_selects.empty()) {
      CHECK_OK(f->RemoveNode(lut_index));
      return false;
    }
  }

  VLOG(2) << "Merging a " << unknown_bits << "-bit lookup table into "
          << candidate_selects.size() << " controlled select(s): "
          << absl::StrJoin(candidate_selects, ", ",
                           [](std::string* out, Select* select) {
                             return select->GetName();
                           });
  if (VLOG_IS_ON(3)) {
    for (Select* candidate : candidate_selects) {
      VLOG(3) << "- " << candidate->ToString();
    }
  }

  std::vector<std::optional<SharedLeafTypeTree<TernaryVector>>> cut_ternaries;
  cut_ternaries.reserve(min_cut.size());
  for (size_t i = 0; i < min_cut.size(); ++i) {
    Node* cut_node = min_cut[i];
    std::optional<SharedLeafTypeTree<TernaryVector>> ternary =
        query_engine.GetTernary(cut_node);
    VLOG(4) << "Ternary for cut node " << cut_node->GetName() << ": "
            << (ternary.has_value() ? "none"
                                    : ternary->ToString([](TernarySpan span) {
                                        return ToString(span);
                                      }));
    cut_ternaries.push_back(*std::move(ternary));
  }

  XLS_ASSIGN_OR_RETURN(std::vector<Value> new_case_sequence,
                       GetLutCases(selector, min_cut, query_engine));
  XLS_RET_CHECK(new_case_sequence.front().IsBits());
  if (absl::c_all_of(new_case_sequence, [&](const Value& index) {
        return index == new_case_sequence.front();
      })) {
    // We've proven that only one case is ever selected; just use that
    // directly.
    for (Select* select : candidate_selects) {
      XLS_RETURN_IF_ERROR(remove_new_select(select));
      XLS_RETURN_IF_ERROR(select->ReplaceUsesWith(
          GetCase(select, new_case_sequence.front().bits())));
    }
    CHECK_OK(f->RemoveNode(lut_index));
    return true;
  }

  XLS_ASSIGN_OR_RETURN(Node * new_selector,
                       GetLutIndex(selector, min_cut, cut_ternaries));
  XLS_RETURN_IF_ERROR(lut_index->ReplaceUsesWith(new_selector));
  CHECK_OK(f->RemoveNode(lut_index));
  for (Select* select : candidate_selects) {
    Select* new_select = new_selects.at(select);
    for (int64_t i = 0; i < new_case_sequence.size(); ++i) {
      XLS_RET_CHECK(new_case_sequence[i].IsBits());
      placeholders.push_back(new_select->get_case(i));
      XLS_RETURN_IF_ERROR(new_select->ReplaceOperandNumber(
          Select::kCasesStart + i,
          GetCase(select, new_case_sequence[i].bits())));
    }
    std::string select_name(select->GetNameView());
    select->SetName("");
    new_select->SetNameDirectly(select_name);
    XLS_RETURN_IF_ERROR(select->ReplaceUsesWith(new_select));
  }
  return true;
}

absl::StatusOr<bool> SimplifyNode(
    Node* node, const QueryEngine& query_engine, int64_t opt_level,
    std::optional<DataflowGraphAnalysis>& dataflow_graph_analysis,
    DelayEstimator* delay_estimator, AreaEstimator* area_estimator) {
  XLS_ASSIGN_OR_RETURN(
      bool changed_select_incorporating_lut,
      MaybeMergeLutIntoSelects(node, query_engine, opt_level,
                               dataflow_graph_analysis, delay_estimator,
                               area_estimator));
  if (changed_select_incorporating_lut) {
    return true;
  }

  return false;
}

}  // namespace

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

  bool changed = false;
  // By running in reverse topological order, the analyses will stay valid for
  // all nodes we're considering through the full pass.
  for (Node* node : context.ReverseTopoSort(func)) {
    if (node->IsDead()) {
      continue;
    }
    XLS_ASSIGN_OR_RETURN(
        bool changed_at_node,
        SimplifyNode(node, query_engine, options.opt_level,
                     dataflow_graph_analysis, options.delay_estimator,
                     options.area_estimator));
    changed = changed || changed_at_node;
  }
  return changed;
}

REGISTER_OPT_PASS(LutConversionPass);

}  // namespace xls
