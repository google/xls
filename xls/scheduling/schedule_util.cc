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
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
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
                                      std::optional<std::string_view> label) {
  // Wildcards always match any target with the lowest specificity score.
  if (pattern == "*") {
    return 0;
  }

  // Unlabeled operation check. Unlabeled operations are represented by
  // nullopt or empty strings.
  if (!label.has_value() || label->empty()) {
    return (pattern == "_") ? std::optional<int>(1) : std::nullopt;
  }

  // Labeled operation exact match check.
  return (pattern == *label) ? std::optional<int>(2) : std::nullopt;
}

std::optional<int> GetArcMatchScore(
    const std::pair<std::string, std::string>& pattern_pair,
    std::optional<std::string_view> label_w,
    std::optional<std::string_view> label_r) {
  std::optional<int> score_w = GetLabelMatchScore(pattern_pair.first, label_w);
  std::optional<int> score_r = GetLabelMatchScore(pattern_pair.second, label_r);
  if (score_w && score_r) {
    return *score_w + *score_r;
  }
  return std::nullopt;
}

ResolvedThroughput GetResolvedThroughputLimit(
    std::optional<std::string_view> write_label,
    std::optional<std::string_view> read_label,
    const absl::flat_hash_map<std::pair<std::string, std::string>, int64_t>&
        arc_worst_case_throughput,
    std::optional<int64_t> default_arc_worst_case_throughput,
    std::optional<int64_t> worst_case_throughput) {
  int64_t highest_pattern_score = -1;
  std::optional<std::pair<std::string, std::string>> highest_priority_pattern =
      std::nullopt;
  int64_t highest_priority_pattern_throughput = 0;

  // Find best match using specificity scoring.
  for (const auto& [pattern, throughput] : arc_worst_case_throughput) {
    std::optional<int> pattern_score =
        GetArcMatchScore(pattern, write_label, read_label);
    if (pattern_score.has_value()) {
      if (*pattern_score > highest_pattern_score) {
        highest_pattern_score = *pattern_score;
        highest_priority_pattern = pattern;
        highest_priority_pattern_throughput = throughput;
      }
    }
  }

  // Fallback and clamping logic (treating <= 0 as "not enforced")
  std::optional<int64_t> throughput_limit = std::nullopt;

  if (highest_priority_pattern.has_value()) {
    // 1. Best matching specific pattern from arc_worst_case_throughput.
    if (highest_priority_pattern_throughput > 0) {
      throughput_limit = highest_priority_pattern_throughput;
    } else {
      throughput_limit = std::nullopt;
    }
  } else if (default_arc_worst_case_throughput.value_or(0) > 0) {
    // 2. Fall back to default_arc_worst_case_throughput.
    throughput_limit = default_arc_worst_case_throughput;
  } else if (worst_case_throughput.has_value() && *worst_case_throughput > 0) {
    // 3. Fall back to global worst_case_throughput.
    throughput_limit = *worst_case_throughput;
  }

  // Enforce worst_case_throughput as strict upper bound (clamping), if
  // globally enabled.
  if (throughput_limit.has_value() && worst_case_throughput.value_or(0) > 0) {
    throughput_limit = std::min(*throughput_limit, *worst_case_throughput);
  }

  return ResolvedThroughput{
      .limit = throughput_limit,
      .matched_pattern = highest_priority_pattern,
  };
}

std::vector<FeedbackArc> GetFeedbackArcsProc(const Proc* proc) {
  std::vector<FeedbackArc> arcs;
  for (Next* next : proc->next_values()) {
    for (StateRead* read :
         proc->GetStateReadsByStateElement(next->state_element())) {
      arcs.push_back(FeedbackArc{
          .write_node = next,
          .read_node = read,
      });
    }
  }
  return arcs;
}

std::vector<FeedbackArc> GetFeedbackArcsPackage(const Package* package) {
  std::vector<FeedbackArc> arcs;
  for (const auto& proc : package->procs()) {
    std::vector<FeedbackArc> proc_arcs = GetFeedbackArcsProc(proc.get());
    arcs.insert(arcs.end(), proc_arcs.begin(), proc_arcs.end());
  }
  return arcs;
}

int GetSpecificityScore(const std::pair<std::string, std::string>& pattern) {
  auto get_component_score = [](std::string_view p) -> int {
    if (p == "*") {
      return 0;
    }
    if (p == "_") {
      return 1;
    }
    return 2;  // Exact label
  };
  return get_component_score(pattern.first) +
         get_component_score(pattern.second);
}

std::optional<std::string> GetPatternIntersectionComponent(
    std::string_view p1, std::string_view p2) {
  if (p1 == p2) {
    return std::string(p1);
  }
  if (p1 == "*") {
    return std::string(p2);
  }
  if (p2 == "*") {
    return std::string(p1);
  }
  return std::nullopt;  // Disjoint
}

std::optional<std::pair<std::string, std::string>> GetPatternIntersection(
    const std::pair<std::string, std::string>& pattern_a,
    const std::pair<std::string, std::string>& pattern_b) {
  auto w_int =
      GetPatternIntersectionComponent(pattern_a.first, pattern_b.first);
  auto r_int =
      GetPatternIntersectionComponent(pattern_a.second, pattern_b.second);
  if (w_int.has_value() && r_int.has_value()) {
    return std::make_pair(*w_int, *r_int);
  }
  return std::nullopt;
}

bool HasPatternIntersection(const std::pair<std::string, std::string>& pattern,
                            const std::pair<std::string, std::string>& target) {
  return GetPatternIntersectionComponent(pattern.first, target.first)
             .has_value() &&
         GetPatternIntersectionComponent(pattern.second, target.second)
             .has_value();
}

absl::Status CheckAmbiguousArcWorstCaseThroughput(
    const absl::flat_hash_map<std::pair<std::string, std::string>, int64_t>&
        throughput_map) {
  // A single rule (or empty configuration) can never conflict with itself.
  if (throughput_map.size() < 2) {
    return absl::OkStatus();
  }

  std::vector<std::pair<std::pair<std::string, std::string>, int64_t>> rules(
      throughput_map.begin(), throughput_map.end());
  rules.reserve(throughput_map.size());

  for (int i = 0; i < rules.size(); ++i) {
    for (int j = i + 1; j < rules.size(); ++j) {
      const auto& pattern_a = rules[i].first;
      const auto& pattern_b = rules[j].first;
      int64_t throughput_a = rules[i].second;
      int64_t throughput_b = rules[j].second;

      // 1. Harmless tie if they have the same throughput value.
      if (throughput_a == throughput_b) {
        continue;
      }

      // 2. Harmless if they have different specificity scores (higher score
      // wins).
      int score_a = GetSpecificityScore(pattern_a);
      int score_b = GetSpecificityScore(pattern_b);
      if (score_a != score_b) {
        continue;
      }

      // 3. Check if they overlap (intersect). If they don't, they are disjoint.
      auto intersection = GetPatternIntersection(pattern_a, pattern_b);
      if (!intersection.has_value()) {
        continue;
      }

      // 4. Overlap detected. Check if there's a strictly more specific rule
      // in the map that overrides the overlap.
      bool tie_is_overridden = false;
      for (const auto& [pattern_c, throughput_c] : throughput_map) {
        if (GetSpecificityScore(pattern_c) > score_a &&
            HasPatternIntersection(pattern_c, *intersection)) {
          tie_is_overridden = true;
          break;
        }
      }

      // 5. Unresolvable tie found.
      if (!tie_is_overridden) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Ambiguous throughput configuration: patterns \"%s,%s\" and "
            "\"%s,%s\" can both match target \"%s,%s\" with the same "
            "specificity score %d but have different throughputs (%d vs %d).",
            pattern_a.first, pattern_a.second, pattern_b.first,
            pattern_b.second, intersection->first, intersection->second,
            score_a, throughput_a, throughput_b));
      }
    }
  }

  return absl::OkStatus();
}

}  // namespace xls
