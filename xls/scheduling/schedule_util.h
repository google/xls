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

#ifndef XLS_SCHEDULING_SCHEDULE_UTIL_H_
#define XLS_SCHEDULING_SCHEDULE_UTIL_H_

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"

namespace xls {

class Package;
class SchedulingOptions;
class Proc;
class Next;
class StateRead;

// Returns the set of nodes in `f` that will have no use when the entity is
// synthesized; that is, the only effect of these nodes is to compute a value
// that is only used pre-synthesis (e.g., asserts, covers, & traces).
absl::StatusOr<absl::flat_hash_set<Node*>> GetDeadAfterSynthesisNodes(
    FunctionBase* f);

// Returns the specificity score for a single pattern against a label.
// Specificity represents how "precise" a matching pattern is:
// - Exact Match (2 points): Matches the specific label of a read or a write
//   (e.g. "my_label").
// - Unlabeled Match (1 point): Represented by "_", matches all unlabeled state
//   operations.
// - Wildcard Match (0 points): Represented by "*", matches any state
//   operation, labeled or unlabeled.
std::optional<int> GetLabelMatchScore(std::string_view pattern,
                                      std::optional<std::string_view> label);

// Returns a score for a pattern pair (write, read) against a feedback arc's
// labels (write, read). The total score is the sum of the scores of its
// components; the higher the score, the more specific the pattern.
// Total Score = (Write Score) + (Read Score)
// exact label match = 2, unlabeled(_) = 1, wildcard(*) = 0, the maximum
// possible score is 4 (exact-exact) and the minimum is 0 (wildcard-wildcard).
// Returns std::nullopt if the pattern pair does not match the labels.
std::optional<int> GetArcMatchScore(
    const std::pair<std::string, std::string>& pattern_pair,
    std::optional<std::string_view> label_w,
    std::optional<std::string_view> label_r);

struct ResolvedThroughput {
  std::optional<int64_t> limit;
  std::optional<std::pair<std::string, std::string>> matched_pattern;
};

// Resolves the final throughput limit for a state feedback arc using the
// configured scheduling options and proc properties (worst-case throughput).
// Performs specificity matching on write and read labels, applies fallback
// and clamping rules, and returns the resolved limit and the winning pattern,
// if any. Assumes that `arc_worst_case_throughput` has no ambiguous cases; if
// multiple patterns match with the same specificity score, one is chosen
// arbitrarily based on map iteration order.
ResolvedThroughput GetResolvedThroughputLimit(
    std::optional<std::string_view> write_label,
    std::optional<std::string_view> read_label,
    const absl::flat_hash_map<std::pair<std::string, std::string>, int64_t>&
        arc_worst_case_throughput,
    std::optional<int64_t> default_arc_worst_case_throughput,
    std::optional<int64_t> worst_case_throughput);

struct FeedbackArc {
  Next* write_node;
  StateRead* read_node;
};

// Collects all feedback arcs (pairs of Next and StateRead accessing the same
// state element) in the given proc.
std::vector<FeedbackArc> GetFeedbackArcsProc(const Proc* proc);

// Collects all feedback arcs (pairs of Next and StateRead accessing the same
// state element) across all procs in the package.
std::vector<FeedbackArc> GetFeedbackArcsPackage(const Package* package);

}  // namespace xls

#endif  // XLS_SCHEDULING_SCHEDULE_UTIL_H_
