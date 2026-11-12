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

#include "xls/passes/next_value_optimization_pass.h"

#include <compare>
#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/proc.h"
#include "xls/ir/state_element.h"
#include "xls/ir/value.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/stateless_query_engine.h"

namespace xls {

namespace {

struct PredicatedValue {
  Node* value;
  std::optional<Node*> predicate;

  friend bool operator==(const PredicatedValue& lhs,
                         const PredicatedValue& rhs) = default;

  friend std::strong_ordering operator<=>(const PredicatedValue& lhs,
                                          const PredicatedValue& rhs) {
    auto pred_id = [](std::optional<Node*> p) -> std::optional<int64_t> {
      return p.has_value() ? std::optional<int64_t>((*p)->id()) : std::nullopt;
    };
    return std::tuple(lhs.value->id(), pred_id(lhs.predicate)) <=>
           std::tuple(rhs.value->id(), pred_id(rhs.predicate));
  }
};

struct StateElementInfo {
  std::vector<PredicatedValue> next_values;
  Value initial_value;

  static StateElementInfo Create(Proc* p, StateElement* st) {
    StateElementInfo si;
    si.initial_value = st->initial_value();
    si.next_values.reserve(p->next_values(st).size());
    for (Next* n : p->next_values(st)) {
      si.next_values.push_back(
          PredicatedValue{.value = n->value(), .predicate = n->predicate()});
    }
    absl::c_sort(si.next_values);
    return si;
  }

  friend bool operator==(const StateElementInfo& lhs,
                         const StateElementInfo& rhs) = default;
};

absl::Status RemoveNextValue(Proc* proc, Next* next,
                             absl::flat_hash_map<Next*, int64_t>& split_depth) {
  XLS_RETURN_IF_ERROR(
      next->ReplaceUsesWithNew<Literal>(Value::Tuple({})).status());
  split_depth.erase(next);
  XLS_RETURN_IF_ERROR(proc->RemoveNode(next))
      << "Unable to clean up next value: " << next->ToString();
  return absl::OkStatus();
}

absl::StatusOr<std::optional<std::vector<Next*>>> RemoveConstantPredicate(
    Proc* proc, Next* next, absl::flat_hash_map<Next*, int64_t>& split_depth,
    const QueryEngine& query_engine) {
  if (!next->predicate().has_value()) {
    return std::nullopt;
  }
  std::optional<Bits> constant_predicate =
      query_engine.KnownValueAsBits(*next->predicate());
  if (!constant_predicate.has_value()) {
    return std::nullopt;
  }

  if (constant_predicate->IsZero()) {
    VLOG(2) << "Identified node as dead due to zero predicate; removing: "
            << *next;
    XLS_RETURN_IF_ERROR(RemoveNextValue(proc, next, split_depth));
    return std::vector<Next*>();
  }
  VLOG(2) << "Identified node as always live; removing predicate: " << *next;
  XLS_ASSIGN_OR_RETURN(Next * new_next, next->ReplaceUsesWithNew<Next>(
                                            next->state_element(),
                                            /*value=*/next->value(),
                                            /*predicate=*/std::nullopt,
                                            /*label=*/next->label()));
  if (auto it = split_depth.find(next); it != split_depth.end()) {
    split_depth[new_next] = it->second;
  }
  XLS_RETURN_IF_ERROR(RemoveNextValue(proc, next, split_depth));
  return std::vector<Next*>({new_next});
}

absl::StatusOr<std::optional<std::vector<Next*>>> SplitSelect(
    Proc* proc, Next* next, const OptimizationPassOptions& options,
    absl::flat_hash_map<Next*, int64_t>& split_depth,
    const absl::flat_hash_map<StateElement*, StateElementInfo>&
        state_element_info,
    const QueryEngine& qe) {
  GenericSelect selected_value;
  if (next->value()->Is<Select>()) {
    selected_value = next->value()->As<Select>();
    if (!options.split_next_value_selects ||
        selected_value.cases().size() > *options.split_next_value_selects) {
      VLOG(4) << "Not splitting " << selected_value
              << " because it is larger than the split limit of "
              << options.split_next_value_selects.value_or(0);
      return std::nullopt;
    }
  } else if (next->value()->Is<PrioritySelect>()) {
    selected_value = next->value()->As<PrioritySelect>();
  } else if (next->value()->Is<OneHotSelect>()) {
    selected_value = next->value()->As<OneHotSelect>();
    if (!qe.ExactlyOneBitTrue(selected_value.selector())) {
      VLOG(4) << "Not splitting " << selected_value
              << " because its selector is not known to be one-hot";
      return std::nullopt;
    }
  } else {
    return std::nullopt;
  }

  int64_t depth = 1;
  if (auto it = split_depth.find(next); it != split_depth.end()) {
    depth = it->second + 1;
  }

  VLOG(2) << "Splitting " << *next << " at depth " << depth << " into "
          << (selected_value.cases().size() +
              (selected_value.default_value() ? 1 : 0))
          << " cases.";

  // When splitting a non-synthesizable state element whose initial value and
  // pre-pass transitions are identical to a synthesizable state element, both
  // hold equal values on every activation, so pass-through reads of that
  // synthesizable state element can be rewritten to matching reads of the
  // non-synthesizable state element.
  auto resolve_case_val = [&](Node* val) -> Node* {
    if (!next->state_element()->non_synthesizable() || !val->Is<StateRead>()) {
      return val;
    }
    StateRead* sr = val->As<StateRead>();
    if (sr->state_element()->non_synthesizable() ||
        state_element_info.at(next->state_element()) !=
            state_element_info.at(sr->state_element())) {
      return val;
    }
    for (StateRead* non_synth_sr :
         proc->GetStateReadsByStateElement(next->state_element())) {
      if (non_synth_sr->predicate() == sr->predicate()) {
        return non_synth_sr;
      }
    }
    return val;
  };

  std::vector<Next*> new_next_values;
  new_next_values.reserve(selected_value.cases().size() +
                          (selected_value.default_value() ? 1 : 0));
  for (int64_t i = 0; i < selected_value.cases().size(); ++i) {
    XLS_ASSIGN_OR_RETURN(Node * predicate,
                         selected_value.MakePredicateForCase(i));
    if (next->predicate().has_value()) {
      XLS_ASSIGN_OR_RETURN(
          predicate,
          proc->MakeNode<NaryOp>(
              selected_value.selector()->loc(),
              std::vector<Node*>{*next->predicate(), predicate}, Op::kAnd));
    }
    std::string name = NodeNameFormat("%s_case_%d", next, i);
    XLS_ASSIGN_OR_RETURN(
        Next * new_next,
        proc->MakeNodeWithName<Next>(
            next->loc(), next->state_element(),
            /*value=*/resolve_case_val(selected_value.cases()[i]), predicate,
            /*label=*/next->label(), name));
    new_next_values.push_back(new_next);
    split_depth[new_next] = depth;
  }

  if (selected_value.default_value()) {
    XLS_ASSIGN_OR_RETURN(Node * predicate,
                         selected_value.MakePredicateForDefault());
    if (next->predicate().has_value()) {
      XLS_ASSIGN_OR_RETURN(
          predicate,
          proc->MakeNode<NaryOp>(
              selected_value.selector()->loc(),
              std::vector<Node*>{*next->predicate(), predicate}, Op::kAnd));
    }

    std::string name = NodeNameConcat(next, "_default_case");
    XLS_ASSIGN_OR_RETURN(
        Next * new_next,
        proc->MakeNodeWithName<Next>(
            next->loc(), next->state_element(),
            /*value=*/resolve_case_val(*selected_value.default_value()),
            predicate, next->label(), name));
    new_next_values.push_back(new_next);
    split_depth[new_next] = depth;
  }

  XLS_RETURN_IF_ERROR(RemoveNextValue(proc, next, split_depth));
  return new_next_values;
}

}  // namespace

RedundancyGuard NextValueOptimizationPass::GetRedundancyGuard(
    const OptimizationPassOptions& options,
    OptimizationContext& context) const {
  std::vector<std::string> config_parts;
  config_parts.reserve(3);
  config_parts.push_back(absl::StrFormat("O%d", options.opt_level));
  config_parts.push_back(absl::StrFormat("max_depth=%d", max_split_depth_));
  if (options.split_next_value_selects.has_value() &&
      *options.split_next_value_selects > 0) {
    config_parts.push_back(
        absl::StrFormat("split_selects=%d", *options.split_next_value_selects));
  }
  return RedundancyGuard::CanSkip(absl::StrJoin(config_parts, ","));
}

absl::StatusOr<bool> NextValueOptimizationPass::RunOnProcInternal(
    Proc* proc, const OptimizationPassOptions& options, PassResults* results,
    OptimizationContext& context) const {
  bool changed = false;

  StatelessQueryEngine query_engine;

  absl::flat_hash_map<StateElement*, StateElementInfo> state_element_info;
  state_element_info.reserve(proc->StateElements().size());
  std::deque<Next*> worklist;
  for (StateElement* state_element : proc->StateElements()) {
    state_element_info.emplace(state_element,
                               StateElementInfo::Create(proc, state_element));
    for (Next* next : proc->next_values(state_element)) {
      worklist.push_back(next);
    }
  }
  if (VLOG_IS_ON(2)) {
    VLOG(2) << "Initial worklist:";
    for (Next* next : worklist) {
      VLOG(2) << "  " << *next;
    }
  }
  absl::flat_hash_map<Next*, int64_t> split_depth;
  while (!worklist.empty()) {
    Next* next = worklist.front();
    worklist.pop_front();

    XLS_ASSIGN_OR_RETURN(
        std::optional<std::vector<Next*>> literal_predicate_next_values,
        RemoveConstantPredicate(proc, next, split_depth, query_engine));
    if (literal_predicate_next_values.has_value()) {
      changed = true;
      worklist.insert(worklist.end(), literal_predicate_next_values->begin(),
                      literal_predicate_next_values->end());
      continue;
    }

    if (auto it = split_depth.find(next);
        options.splits_enabled() && max_split_depth_ > 0 &&
        (it == split_depth.end() || it->second < max_split_depth_)) {
      XLS_ASSIGN_OR_RETURN(
          std::optional<std::vector<Next*>> split_select_next_values,
          SplitSelect(proc, next, options, split_depth, state_element_info,
                      query_engine));
      if (split_select_next_values.has_value()) {
        changed = true;
        worklist.insert(worklist.end(), split_select_next_values->begin(),
                        split_select_next_values->end());
        continue;
      }
    }
  }

  XLS_RETURN_IF_ERROR(proc->RebuildSideTables());
  return changed;
}

}  // namespace xls
