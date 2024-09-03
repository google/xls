// Copyright 2023 The XLS Authors
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

#ifndef XLS_FDO_DELAY_MANAGER_H_
#define XLS_FDO_DELAY_MANAGER_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/random/bit_gen_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/scheduling/scheduling_options.h"

namespace xls {

struct PathInfo {
  PathInfo(int64_t delay, Node *source, Node *target)
      : delay(delay), source(source), target(target) {}

  int64_t delay;
  Node *source;
  Node *target;

  bool operator==(const PathInfo &other) const {
    return delay == other.delay && source == other.source &&
           target == other.target;
  }
};

struct PathExtractOptions {
  // Record the pipeline schedule.
  const ScheduleCycleMap *cycle_map = nullptr;

  // Ensure all paths not having Param as sources.
  bool exclude_param_source = true;

  // Ensure all paths having more than one node.
  bool exclude_single_node_path = false;

  // Ensure all paths are combinational. cycle_map must be provided to work.
  bool combinational_only = true;

  // Ensure all paths having primary input (or pipeline register) as sources.
  bool input_source_only = false;

  // Ensure all paths having primary output (or pipeline register) as targets.
  bool output_target_only = false;

  // Ensure all paths having unique targets.
  bool unique_target_only = true;
};

// This class manages the delay estimations of all pairs of nodes in a function
// or proc. It allows users to update the delay between a certain pair of nodes,
// re-calculate the critical delay of all pairs of nodes, extract paths longer
// than a threshold, extract top-N longest paths, etc.
class DelayManager {
 public:
  explicit DelayManager(FunctionBase *function,
                        const DelayEstimator &delay_estimator);

  absl::StatusOr<int64_t> GetNodeDelay(Node *node) const;

  absl::StatusOr<int64_t> GetCriticalPathDelay(Node *from, Node *to) const;

  absl::Status SetCriticalPathDelay(Node *from, Node *to, int64_t delay,
                                    bool if_shorter = true,
                                    bool if_exist = true);

  // Find and return the full critical path (a chain of nodes) between "from"
  // and "to".
  absl::StatusOr<std::vector<Node *>> GetFullCriticalPath(Node *from,
                                                          Node *to) const;

  // Recalculate the delays of all pairs of nodes.
  //
  // Implementation note: With the delay of some paths updated, the delay of
  // related paths can also be recalculated. For instance, if we have updated
  // the critical path delay of A-B, and we have a path A-B-C, then the critical
  // path delay of A-C can be recalculated by adding the critical path delay of
  // A-B and the delay of C. The recalculation is done by propagating the
  // updated delays in a topological order once and then in a reversed
  // topological order once. Note that this method is not optimal - it cannot
  // find the best combination of partial paths as Floyd–Warshall but it's
  // complexity is in O(n^2).
  void PropagateDelays();

  // Get all the paths whose delay is longer than the given delay threshold.
  absl::flat_hash_map<Node *, std::vector<Node *>> GetPathsOverDelayThreshold(
      int64_t delay_threshold) const;

  // Get the top-N highest score or longest delay paths (if scores are on-par).
  // The score will be calculated through the given "score" function. The score
  // is always the higher the better. The paths evaluated as true by the given
  // "except" function are excepted.
  absl::StatusOr<std::vector<PathInfo>> GetTopNPaths(
      int64_t number_paths, const PathExtractOptions &options,
      absl::FunctionRef<bool(Node *, Node *)> except = GetFalse,
      absl::FunctionRef<float(Node *, Node *)> score = GetZeroScore) const;

  // Same as GetTopNPaths method, but randomly choose number_paths with the
  // given stochastic_ratio. "ratio" should always > 0.0 and <= 1.0.
  absl::StatusOr<std::vector<PathInfo>> GetTopNPathsStochastically(
      int64_t number_paths, float stochastic_ratio,
      const PathExtractOptions &options, absl::BitGenRef bit_gen,
      absl::FunctionRef<bool(Node *, Node *)> except = GetFalse,
      absl::FunctionRef<float(Node *, Node *)> score = GetZeroScore) const;

  // Get the critical path and its delay.
  absl::StatusOr<PathInfo> GetLongestPath(
      const PathExtractOptions &options,
      absl::FunctionRef<bool(Node *, Node *)> except = GetFalse) const;

 private:
  static float GetZeroScore(Node *from, Node *to) { return 0.0; }
  static bool GetFalse(Node *from, Node *to) { return false; }

  FunctionBase *function_;

  // A mapping from a node to its index in the function.
  absl::flat_hash_map<Node *, int64_t> node_to_index_;

  // A mapping from a node index to the corresponding node.
  std::vector<Node *> index_to_node_;

  // An all-to-all delay mapping. The self-to-self delay of a node is defined as
  // the delay of itself. If there is no valid path exists, the delay is defined
  // as -1. Both the source and target node delays are counted.
  std::vector<std::vector<int64_t>> indices_to_delay_;

  // A all-to-all mapping from the source and target node indices to the
  // critical operand of the target node.
  std::vector<std::vector<Node *>> indices_to_critical_operand_;

  // Name of the delay estimator.
  const std::string name_;
};

}  // namespace xls

#endif  // XLS_FDO_DELAY_MANAGER_H_
