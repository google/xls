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

#include "xls/scheduling/schedule_util.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/proc.h"
#include "xls/passes/node_dependency_analysis.h"

namespace xls {

// TODO(allight): This currently treats any state element that drives a
// different state element as unconditionally live. Ideally we'd want to figure
// out if the transitive closure of nodes extended into the future activations
// is ever held live by a normal operation and only consider a state element
// live if it is. This is more complicated to do however and so for now we will
// simply limit things to one activation.
absl::StatusOr<absl::flat_hash_set<Node*>> GetDeadAfterSynthesisNodes(
    FunctionBase* f) {
  NodeForwardDependencyAnalysis nda;
  XLS_RETURN_IF_ERROR(nda.Attach(f).status());
  absl::flat_hash_set<Node*> live_after_synthesis;
  live_after_synthesis.reserve(f->node_count());
  auto mark_live = [&](Node* node) {
    if (VLOG_IS_ON(2)) {
      if (!live_after_synthesis.contains(node)) {
        VLOG(2) << "Marking live " << node << " makes live: ["
                << absl::StrJoin(nda.NodesDependedOnBy(node), ", ") << "]";
      } else {
        VLOG(2) << node << " already live";
      }
    }
    live_after_synthesis.insert(node);
    const absl::flat_hash_set<Node*>& depending_on =
        nda.NodesDependedOnBy(node);
    live_after_synthesis.insert(depending_on.begin(), depending_on.end());
  };
  for (Node* node : f->nodes()) {
    if (f->HasImplicitUse(node) ||
        (OpIsSideEffecting(node->op()) &&
         // Asserts, covers, and traces are never synthesized. Next and state
         // read are only synthesized if their results are used by synthesized
         // things so we do a second pass to determine this
         !node->OpIn({Op::kAssert, Op::kCover, Op::kTrace, Op::kNext,
                      Op::kStateRead}))) {
      mark_live(node);
      VLOG(2) << "  reason: "
              << (f->HasImplicitUse(node) ? "implicit use" : "side effect");
    } else if (node->Is<Next>()) {
      // Next's do explicitly keep live any state-reads that aren't their own
      // though.
      for (Node* n : nda.NodesDependedOnBy(node)) {
        if (n->Is<StateRead>() && n->As<StateRead>()->state_element() !=
                                      node->As<Next>()->state_element()) {
          mark_live(n);
          VLOG(2) << "  reason: in dependencies of other states next: " << node;
        }
      }
    }
  }
  // Figure out which states are live.
  if (f->IsProc()) {
    Proc* proc = f->AsProcOrDie();
    for (StateElement* state_element : proc->StateElements()) {
      VLOG(2) << "Considering state element: " << state_element->name();
      bool state_is_live = false;
      for (StateRead* read : proc->GetStateReadsByStateElement(state_element)) {
        if (live_after_synthesis.contains(read)) {
          state_is_live = true;
          break;
        }
      }
      if (state_is_live) {
        for (Next* next : proc->next_values(state_element)) {
          mark_live(next);
        }
      }
    }
  }
  absl::flat_hash_set<Node*> dead_after_synthesis;
  dead_after_synthesis.reserve(f->node_count() - live_after_synthesis.size());
  for (Node* node : f->nodes()) {
    if (!live_after_synthesis.contains(node)) {
      dead_after_synthesis.insert(node);
    }
  }
  return dead_after_synthesis;
}

std::optional<int> GetLabelMatchScore(std::string_view pattern,
                                      const std::optional<std::string>& label) {
  // Wildcards always match any target with the lowest specificity score.
  if (pattern == "*") {
    return 0;
  }

  // Unlabeled operation check, unlabeled operations are represented by
  // nullopt or empty strings.
  if (!label.has_value() || label->empty()) {
    return (pattern == "_") ? std::optional<int>(1) : std::nullopt;
  }

  // Labeled operation exact match check.
  return (pattern == *label) ? std::optional<int>(2) : std::nullopt;
}

std::optional<int> GetArcMatchScore(
    const std::pair<std::string, std::string>& pattern_pair,
    const std::optional<std::string>& label_w,
    const std::optional<std::string>& label_r) {
  std::optional<int> score_w = GetLabelMatchScore(pattern_pair.first, label_w);
  std::optional<int> score_r = GetLabelMatchScore(pattern_pair.second, label_r);
  if (score_w && score_r) {
    return *score_w + *score_r;
  }
  return std::nullopt;
}

ResolvedThroughput GetResolvedThroughputLimit(
    const std::optional<std::string>& write_label,
    const std::optional<std::string>& read_label,
    const absl::flat_hash_map<std::pair<std::string, std::string>, int64_t>&
        arc_worst_case_throughput,
    std::optional<int64_t> default_arc_worst_case_throughput,
    std::optional<int64_t> initiation_interval,
    absl::flat_hash_set<std::pair<std::string, std::string>>* viable_patterns) {
  // Find best match using specificity scoring.
  std::optional<std::pair<std::string, std::string>> highest_priority_pattern =
      std::nullopt;
  int64_t highest_priority_pattern_throughput = 0;
  int64_t highest_pattern_score = -1;

  for (const auto& [pattern, throughput] : arc_worst_case_throughput) {
    std::optional<int> pattern_score =
        GetArcMatchScore(pattern, write_label, read_label);
    if (pattern_score.has_value()) {
      if (viable_patterns != nullptr) {
        viable_patterns->insert(pattern);
      }
      if (*pattern_score > highest_pattern_score) {
        highest_pattern_score = *pattern_score;
        highest_priority_pattern = pattern;
        highest_priority_pattern_throughput = throughput;
      }
    }
  }

  // Fallback and clamping logic (treating <= 0 as "not enforced")
  std::optional<int64_t> throughput_limit = std::nullopt;

  if (highest_priority_pattern.has_value() &&
      highest_priority_pattern_throughput > 0) {
    // 1. Best matching specific pattern from arc_worst_case_throughput.
    throughput_limit = highest_priority_pattern_throughput;
  } else if (default_arc_worst_case_throughput.value_or(0) > 0) {
    // 2. Fall back to default_arc_worst_case_throughput.
    throughput_limit = default_arc_worst_case_throughput;
  } else if (initiation_interval.has_value() && *initiation_interval > 0) {
    // 3. Fall back to global worst_case_throughput.
    throughput_limit = *initiation_interval;
  }

  // Enforce initiation_interval as strict upper bound (clamping), if
  // globally enabled.
  if (throughput_limit.has_value() && initiation_interval.value_or(0) > 0) {
    throughput_limit = std::min(*throughput_limit, *initiation_interval);
  }

  return ResolvedThroughput{
      .limit = throughput_limit,
      .matched_pattern = highest_priority_pattern,
  };
}

}  // namespace xls
