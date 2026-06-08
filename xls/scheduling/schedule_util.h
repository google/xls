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
#include "absl/status/status.h"
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

// Returns the static specificity score for a matching pattern pair:
// - Exact Label Match (e.g., "my_label") = 2 points.
// - Unlabeled Match ("_")                = 1 point.
// - Wildcard Match ("*")                 = 0 points.
// The total score is the sum of the write and read components. Range: [0, 4].
int GetSpecificityScore(const std::pair<std::string, std::string>& pattern);

// Returns the overlapping intersection of two pattern components:
// - If they are identical (e.g. both are "L_W"), they intersect perfectly.
// - If one component is a wildcard ("*"), it matches everything, so the
//   intersection is the other component.
// - If they are different and neither is "*", they are disjoint (nullopt).
std::optional<std::string> GetPatternIntersectionComponent(std::string_view p1,
                                                           std::string_view p2);

// Returns the overlapping intersection between two pattern pairs.
// Two pattern pairs intersect if both their write and read components
// intersect. Returns std::nullopt if the patterns are completely disjoint.
std::optional<std::pair<std::string, std::string>> GetPatternIntersection(
    const std::pair<std::string, std::string>& pattern_a,
    const std::pair<std::string, std::string>& pattern_b);

// Returns true if two patterns have a non-empty intersection (overlap).
bool HasPatternIntersection(const std::pair<std::string, std::string>& pattern,
                            const std::pair<std::string, std::string>& target);

// Validates the entire throughput override map for unresolvable conflicts.
// Returns InvalidArgumentError if two rules can match the same label pair with
// the same highest specificity score but specify different throughput bounds.
absl::Status CheckAmbiguousArcWorstCaseThroughput(
    const absl::flat_hash_map<std::pair<std::string, std::string>, int64_t>&
        throughput_map);

// Verify that every pattern specified in options.arc_worst_case_throughput()
// matches at least one feedback arc in the package. Returns
// InvalidArgumentError if any pattern is unused (to catch typos).
absl::Status VerifyArcWorstCaseThroughputViability(
    const Package* package, const SchedulingOptions& options);

}  // namespace xls

#endif  // XLS_SCHEDULING_SCHEDULE_UTIL_H_
