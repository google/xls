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

#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "cppitertools/sorted.hpp"
#include "xls/common/status/ret_check.h"
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

// either a next value in the proc or a next value and the 1 non-synth next that
// always sets the exact same value.
struct IdenticalNexts {
  Next* main;
  std::optional<Next*> non_synth;

  template <typename H>
  friend H AbslHashValue(H h, const IdenticalNexts& nexts) {
    return H::combine(std::move(h), nexts.main, nexts.non_synth);
  }

  friend bool operator==(const IdenticalNexts& lhs, const IdenticalNexts& rhs) {
    return lhs.main == rhs.main && lhs.non_synth == rhs.non_synth;
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const IdenticalNexts& nexts) {
    absl::Format(&sink, "{ main: %v", *nexts.main);
    if (nexts.non_synth.has_value()) {
      absl::Format(&sink, ", non_synth: %v", **nexts.non_synth);
    }
    absl::Format(&sink, " }");
  }

  friend std::ostream& operator<<(std::ostream& oss,
                                  const IdenticalNexts& nexts) {
    return oss << absl::StreamFormat("%v", nexts);
  }
};

absl::Status RemoveNextValue(
    Proc* proc, const IdenticalNexts& next,
    absl::flat_hash_map<IdenticalNexts, int64_t>& split_depth) {
  XLS_RETURN_IF_ERROR(
      next.main->ReplaceUsesWithNew<Literal>(Value::Tuple({})).status());
  if (next.non_synth) {
    XLS_RETURN_IF_ERROR((*next.non_synth)
                            ->ReplaceUsesWithNew<Literal>(Value::Tuple({}))
                            .status());
  }
  if (auto it = split_depth.find(next); it != split_depth.end()) {
    split_depth.erase(it);
  }
  XLS_RETURN_IF_ERROR(proc->RemoveNode(next.main))
      << "Unable to clean up main next value for " << next;
  if (next.non_synth) {
    XLS_RETURN_IF_ERROR(proc->RemoveNode(*next.non_synth))
        << "Unable to clean up non-synth next value for " << next;
  }
  return absl::OkStatus();
}

absl::StatusOr<std::optional<std::vector<IdenticalNexts>>>
RemoveConstantPredicate(
    Proc* proc, const IdenticalNexts next,
    absl::flat_hash_map<IdenticalNexts, int64_t>& split_depth,
    const QueryEngine& query_engine) {
  if (!next.main->predicate().has_value()) {
    return std::nullopt;
  }
  std::optional<Bits> constant_predicate =
      query_engine.KnownValueAsBits(*next.main->predicate());
  if (!constant_predicate.has_value()) {
    return std::nullopt;
  }

  if (constant_predicate->IsZero()) {
    VLOG(2) << "Identified node as dead due to zero predicate; removing: "
            << next;
    XLS_RETURN_IF_ERROR(RemoveNextValue(proc, next, split_depth));
    return std::vector<IdenticalNexts>();
  }
  VLOG(2) << "Identified node as always live; removing predicate: " << next;
  IdenticalNexts new_next;
  XLS_ASSIGN_OR_RETURN(new_next.main,
                       next.main->ReplaceUsesWithNew<Next>(
                           /*state_read=*/next.main->state_read(),
                           /*value=*/next.main->value(),
                           /*predicate=*/std::nullopt,
                           /*label=*/std::nullopt));
  if (next.non_synth) {
    XLS_ASSIGN_OR_RETURN(new_next.non_synth,
                         (*next.non_synth)
                             ->ReplaceUsesWithNew<Next>(
                                 /*state_read=*/(*next.non_synth)->state_read(),
                                 /*value=*/(*next.non_synth)->value(),
                                 /*predicate=*/std::nullopt,
                                 /*label=*/std::nullopt));
  }
  if (split_depth.contains(next)) {
    split_depth[new_next] = split_depth[next];
  }
  XLS_RETURN_IF_ERROR(RemoveNextValue(proc, next, split_depth));
  return std::vector<IdenticalNexts>({new_next});
}

absl::StatusOr<std::optional<std::vector<IdenticalNexts>>> SplitSelect(
    Proc* proc, const IdenticalNexts& next,
    const OptimizationPassOptions& options,
    absl::flat_hash_map<IdenticalNexts, int64_t>& split_depth,
    const QueryEngine& qe) {
  GenericSelect selected_value;
  if (next.main->value()->Is<Select>()) {
    selected_value = next.main->value()->As<Select>();
    if (!options.split_next_value_selects ||
        selected_value.cases().size() > *options.split_next_value_selects) {
      VLOG(4) << "Not splitting " << selected_value
              << " because it is larger than the split limit of "
              << options.split_next_value_selects.value_or(0);
      return std::nullopt;
    }
  } else if (next.main->value()->Is<PrioritySelect>()) {
    selected_value = next.main->value()->As<PrioritySelect>();
  } else if (next.main->value()->Is<OneHotSelect>()) {
    selected_value = next.main->value()->As<OneHotSelect>();
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

  VLOG(2) << "Splitting " << next << " at depth " << depth << " into "
          << (selected_value.cases().size() +
              (selected_value.default_value() ? 1 : 0))
          << " cases.";

  std::vector<IdenticalNexts> new_next_values;
  for (int64_t i = 0; i < selected_value.cases().size(); ++i) {
    IdenticalNexts new_next;
    XLS_ASSIGN_OR_RETURN(Node * predicate,
                         selected_value.MakePredicateForCase(i));
    if (next.main->predicate()) {
      XLS_ASSIGN_OR_RETURN(
          predicate, proc->MakeNode<NaryOp>(
                         selected_value.selector()->loc(),
                         std::vector<Node*>{*next.main->predicate(), predicate},
                         Op::kAnd));
    }
    std::string name;
    if (next.main->HasAssignedName()) {
      name = absl::StrCat(next.main->GetName(), "_case_", i);
    }
    XLS_ASSIGN_OR_RETURN(
        new_next.main,
        proc->MakeNodeWithName<Next>(next.main->loc(),
                                     /*state_read=*/next.main->state_read(),
                                     /*value=*/selected_value.cases()[i],
                                     predicate, /*label=*/std::nullopt, name));

    if (next.non_synth) {
      std::string non_synth_name;
      if ((*next.non_synth)->HasAssignedName()) {
        non_synth_name =
            absl::StrCat((*next.non_synth)->GetName(), "_case_", i);
      }
      // Change main pass-through updates to pass-through on the non-synth one
      // too.
      Node* case_val = selected_value.cases()[i] == next.main->state_read()
                           ? (*next.non_synth)->state_read()
                           : selected_value.cases()[i];
      XLS_ASSIGN_OR_RETURN(new_next.non_synth,
                           proc->MakeNodeWithName<Next>(
                               (*next.non_synth)->loc(),
                               /*state_read=*/(*next.non_synth)->state_read(),
                               /*value=*/case_val, predicate,
                               /*label=*/std::nullopt, non_synth_name));
    }
    new_next_values.push_back(new_next);
    split_depth[new_next] = depth;
  }

  if (selected_value.default_value()) {
    XLS_ASSIGN_OR_RETURN(Node * predicate,
                         selected_value.MakePredicateForDefault());
    if (next.main->predicate().has_value()) {
      XLS_ASSIGN_OR_RETURN(
          predicate, proc->MakeNode<NaryOp>(
                         selected_value.selector()->loc(),
                         std::vector<Node*>{*next.main->predicate(), predicate},
                         Op::kAnd));
    }

    IdenticalNexts new_next;
    std::string name;
    if (next.main->HasAssignedName()) {
      name = absl::StrCat(next.main->GetNameView(), "_default_case");
    }
    XLS_ASSIGN_OR_RETURN(
        new_next.main,
        proc->MakeNodeWithName<Next>(next.main->loc(),
                                     /*state_read=*/next.main->state_read(),
                                     /*value=*/*selected_value.default_value(),
                                     predicate, /*label=*/std::nullopt, name));
    if (next.non_synth) {
      std::string non_synth_name;
      if ((*next.non_synth)->HasAssignedName()) {
        non_synth_name =
            absl::StrCat((*next.non_synth)->GetNameView(), "_default_case");
      }
      Node* value = *selected_value.default_value() == next.main->state_read()
                        ? (*next.non_synth)->state_read()
                        : *selected_value.default_value();
      XLS_ASSIGN_OR_RETURN(
          new_next.non_synth,
          proc->MakeNodeWithName<Next>(
              (*next.non_synth)->loc(),
              /*state_read=*/(*next.non_synth)->state_read(), value, predicate,
              /*label=*/std::nullopt, non_synth_name));
    }
    new_next_values.push_back(new_next);
    split_depth[new_next] = depth;
  }

  XLS_RETURN_IF_ERROR(RemoveNextValue(proc, next, split_depth));
  return new_next_values;
}

struct StateElementInfo {
  // sorted by id of next values
  std::vector<Node*> values;
  // sorted by id of next values
  std::vector<std::optional<Node*>> predicates;
  Value initial_value;

  static StateElementInfo Create(Proc* p, StateElement* st) {
    StateElementInfo si;
    si.initial_value = st->initial_value();
    StateRead* rd = p->GetStateRead(st);
    for (Next* n : iter::sorted(p->next_values(rd), [](Node* l, Node* r) {
           return l->id() < r->id();
         })) {
      si.values.push_back(n->value());
      si.predicates.push_back(n->predicate());
    }
    absl::c_sort(si.values, [](Node* l, Node* r) { return l->id() < r->id(); });
    absl::c_sort(si.predicates,
                 [](std::optional<Node*> l, std::optional<Node*> r) {
                   if (!l && r) {
                     return true;
                   } else if (!r) {
                     return false;
                   }
                   return (*l)->id() < (*r)->id();
                 });
    return si;
  }

  template <typename H>
  friend H AbslHashValue(H h, const StateElementInfo& info) {
    return H::combine(std::move(h), info.values, info.predicates,
                      info.initial_value);
  }

  friend bool operator==(const StateElementInfo& lhs,
                         const StateElementInfo& rhs) {
    return lhs.values == rhs.values && lhs.predicates == rhs.predicates &&
           lhs.initial_value == rhs.initial_value;
  }
};

struct StateElementPair {
  StateElement* main;
  std::optional<StateElement*> non_synth;
};

}  // namespace

absl::StatusOr<bool> NextValueOptimizationPass::RunOnProcInternal(
    Proc* proc, const OptimizationPassOptions& options, PassResults* results,
    OptimizationContext& context) const {
  bool changed = false;

  StatelessQueryEngine query_engine;

  // Find the non-synth next values generated by non_synth_separation_pass.cc
  // and group them with the original (identified by having an identical
  // update).
  //
  // We might have multiple state elements that look identical and we need to
  // pair them up so one non-synth goes with each synth.
  absl::flat_hash_map<StateElementInfo, std::deque<StateElement*>> found_ses;
  std::vector<StateElementInfo> sei_order;
  std::vector<StateElementPair> synth_nonsynth_pairs;
  synth_nonsynth_pairs.reserve(proc->StateElements().size());

  for (StateElement* state_element : proc->StateElements()) {
    auto info = StateElementInfo::Create(proc, state_element);
    auto it = found_ses.find(info);
    if (it != found_ses.end() && !it->second.empty()) {
      if (state_element->non_synthesizable() ==
          it->second.front()->non_synthesizable()) {
        // Must be 2 identical state elements.
        it->second.push_front(state_element);
        continue;
      }
      synth_nonsynth_pairs.push_back(
          state_element->non_synthesizable()
              ? StateElementPair{.main = it->second.front(),
                                 .non_synth = state_element}
              : StateElementPair{.main = state_element,
                                 .non_synth = it->second.front()});
      it->second.pop_front();
    } else {
      found_ses[info].push_front(state_element);
      sei_order.push_back(info);
    }
  }
  // Include the rest.
  for (const StateElementInfo& info : sei_order) {
    const auto& state_elements = found_ses[info];
    for (StateElement* state_element : state_elements) {
      synth_nonsynth_pairs.push_back({.main = state_element});
    }
  }
  if (VLOG_IS_ON(2)) {
    int64_t synth_cnt = 0;
    int64_t non_synth_cnt = 0;
    for (const auto& [synth, non_synth] : synth_nonsynth_pairs) {
      if (non_synth) {
        VLOG(2) << "Element '" << synth->ToString() << "' has non-synth '"
                << (*non_synth)->ToString() << "'.";
        non_synth_cnt++;
      } else {
        VLOG(2) << "Element '" << synth->ToString() << "' no non-synth";
        synth_cnt++;
      }
    }
    VLOG(2) << "Found " << synth_cnt
            << " state elements without a non-synth and " << non_synth_cnt
            << " state elements with a non-synth.";
  }
  std::deque<IdenticalNexts> worklist;
  for (const auto& [elem, non_synth] : synth_nonsynth_pairs) {
    StateRead* read = proc->GetStateRead(elem);
    if (!non_synth) {
      for (Next* next : proc->next_values(read)) {
        worklist.push_back({.main = next});
      }
    } else {
      absl::flat_hash_map<absl::Span<Node* const>, Next*> nonsynth_nexts;
      for (Next* next : proc->next_values(proc->GetStateRead(*non_synth))) {
        nonsynth_nexts[next->operands().subspan(1)] = next;
      }
      for (Next* next : proc->next_values(read)) {
        auto it = nonsynth_nexts.find(next->operands().subspan(1));
        XLS_RET_CHECK(it != nonsynth_nexts.end())
            << "Unable to find corresponding non-synth next for " << next;
        worklist.push_back({.main = next, .non_synth = it->second});
      }
    }
  }
  if (VLOG_IS_ON(2)) {
    VLOG(2) << "Initial worklist:";
    for (const auto& elem : worklist) {
      VLOG(2) << "  " << elem;
    }
  }
  absl::flat_hash_map<IdenticalNexts, int64_t> split_depth;
  while (!worklist.empty()) {
    IdenticalNexts next = worklist.front();
    worklist.pop_front();

    XLS_ASSIGN_OR_RETURN(
        std::optional<std::vector<IdenticalNexts>>
            literal_predicate_next_values,
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
          std::optional<std::vector<IdenticalNexts>> split_select_next_values,
          SplitSelect(proc, next, options, split_depth, query_engine));
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
