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

#include "xls/fdo/delay_manager.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <tuple>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/random/bit_gen_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/node_iterator.h"
#include "xls/ir/op.h"
#include "xls/scheduling/scheduling_options.h"

namespace xls {

DelayManager::DelayManager(FunctionBase *function,
                           const DelayEstimator &delay_estimator)
    : function_(function),
      index_to_node_(function_->node_count()),
      indices_to_delay_(function_->node_count(),
                        std::vector<int64_t>(function_->node_count(), -1)),
      indices_to_critical_operand_(
          function_->node_count(),
          std::vector<Node *>(function_->node_count(), nullptr)),
      name_(delay_estimator.name()) {
  // Get the mapping between function node and their index. Also, estimate the
  // delay of each node.
  node_to_index_.reserve(function_->node_count());
  int32_t index = 0;
  for (Node *node : function_->nodes()) {
    node_to_index_[node] = index;
    index_to_node_[index] = node;
    absl::StatusOr<int64_t> maybe_delay =
        delay_estimator.GetOperationDelayInPs(node);
    CHECK_OK(maybe_delay.status());
    indices_to_delay_[index][index] = maybe_delay.value();
    index++;
  }
  PropagateDelays();
}

absl::StatusOr<int64_t> DelayManager::GetNodeDelay(Node *node) const {
  if (node->function_base() != function_) {
    return absl::InvalidArgumentError("invalid node");
  }
  int64_t node_index = node_to_index_.at(node);
  return indices_to_delay_[node_index][node_index];
}

absl::StatusOr<int64_t> DelayManager::GetCriticalPathDelay(Node *from,
                                                           Node *to) const {
  if (from->function_base() != function_ || to->function_base() != function_) {
    return absl::InvalidArgumentError("invalid path");
  }
  int64_t from_index = node_to_index_.at(from);
  int64_t to_index = node_to_index_.at(to);
  return indices_to_delay_[from_index][to_index];
}

absl::Status DelayManager::SetCriticalPathDelay(Node *from, Node *to,
                                                int64_t delay, bool if_shorter,
                                                bool if_exist) {
  if (from->function_base() != function_ || to->function_base() != function_) {
    return absl::InvalidArgumentError("invalid path");
  }
  int64_t from_index = node_to_index_.at(from);
  int64_t to_index = node_to_index_.at(to);
  int64_t &current_delay = indices_to_delay_[from_index][to_index];
  if (!if_shorter || current_delay > delay) {
    if (!if_exist || current_delay != -1) {
      current_delay = delay;
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<Node *>> DelayManager::GetFullCriticalPath(
    Node *from, Node *to) const {
  int64_t from_index = node_to_index_.at(from);
  int64_t to_index = node_to_index_.at(to);
  std::vector<Node *> critical_path;

  Node *critical_operand = indices_to_critical_operand_[from_index][to_index];
  critical_path.push_back(to);
  while (critical_operand != nullptr && critical_operand != from) {
    critical_path.push_back(critical_operand);
    int64_t critical_operand_index = node_to_index_.at(critical_operand);
    critical_operand =
        indices_to_critical_operand_[from_index][critical_operand_index];
  }
  XLS_RET_CHECK(critical_operand == from);
  critical_path.push_back(from);
  std::reverse(critical_path.begin(), critical_path.end());
  return critical_path;
}

void DelayManager::PropagateDelays() {
  // Traverse the function in a reversed topological order.
  for (Node *node : ReverseTopoSort(function_)) {
    int64_t node_index = node_to_index_[node];
    int64_t node_delay = indices_to_delay_[node_index][node_index];
    std::vector<int64_t> new_delays(function_->node_count(), -1);

    // Compute the critical-path distance from `a` to `node` for all nodes `a`
    // from the delays of `a` to each operand of `node`.
    for (Node *user : node->users()) {
      int64_t user_index = node_to_index_[user];
      for (int64_t i = 0; i < function_->node_count(); ++i) {
        int64_t from_user_delay = indices_to_delay_[user_index][i];
        if (from_user_delay != -1) {
          // Always pick the critical path.
          if (new_delays[i] < from_user_delay + node_delay) {
            new_delays[i] = from_user_delay + node_delay;
          }
        }
      }
    }

    // Update the original delay if the newly calculated delay is smaller.
    for (int64_t i = 0; i < function_->node_count(); ++i) {
      if (new_delays[i] != -1) {
        int64_t &current_delay = indices_to_delay_[node_index][i];
        if (current_delay >= new_delays[i] || current_delay == -1) {
          current_delay = new_delays[i];
        }
      }
    }
  }

  // Traverse the function in a topological order.
  for (Node *node : TopoSort(function_)) {
    int64_t node_index = node_to_index_[node];
    int64_t node_delay = indices_to_delay_[node_index][node_index];
    std::vector<int64_t> new_delays(function_->node_count(), -1);
    std::vector<Node *> new_critical_operands(function_->node_count(), nullptr);

    // Compute the critical-path distance from `a` to `node` for all nodes `a`
    // from the delays of `a` to each operand of `node`.
    for (Node *operand : node->operands()) {
      int64_t operand_index = node_to_index_[operand];
      for (int64_t i = 0; i < function_->node_count(); ++i) {
        int64_t to_operand_delay = indices_to_delay_[i][operand_index];
        if (to_operand_delay != -1) {
          // Always pick the critical path.
          if (new_delays[i] < to_operand_delay + node_delay) {
            new_delays[i] = to_operand_delay + node_delay;
            new_critical_operands[i] = operand;
          }
        }
      }
    }

    // Update the original delay if the newly calculated delay is smaller.
    for (int64_t i = 0; i < function_->node_count(); ++i) {
      if (new_delays[i] != -1) {
        int64_t &current_delay = indices_to_delay_[i][node_index];
        if (current_delay >= new_delays[i] || current_delay == -1) {
          current_delay = new_delays[i];
          indices_to_critical_operand_[i][node_index] =
              new_critical_operands[i];
        }
      }
    }
  }
}

absl::flat_hash_map<Node *, std::vector<Node *>>
DelayManager::GetPathsOverDelayThreshold(int64_t delay_threshold) const {
  absl::flat_hash_map<Node *, std::vector<Node *>> paths;
  if (delay_threshold < 0) {
    return paths;
  }
  for (int64_t i = 0; i < function_->node_count(); ++i) {
    Node *from = index_to_node_[i];
    for (int64_t j = 0; j < function_->node_count(); ++j) {
      Node *to = index_to_node_[j];
      if (indices_to_delay_[i][j] > delay_threshold) {
        paths[from].push_back(to);
      }
    }
  }
  return paths;
}

absl::StatusOr<std::vector<PathInfo>> DelayManager::GetTopNPaths(
    int64_t number_paths, const PathExtractOptions &options,
    absl::FunctionRef<bool(Node *, Node *)> except,
    absl::FunctionRef<float(Node *, Node *)> score) const {
  // To extract combinational paths, the cycle_map must be provided.
  if (options.combinational_only) {
    XLS_RET_CHECK(options.cycle_map);
  }

  // Traverse all nodes in the function and construct a worklist with score of
  // each path.
  std::vector<std::tuple<float, int64_t, Node *, Node *>> worklist;
  for (int64_t i = 0; i < function_->node_count(); ++i) {
    Node *from = index_to_node_[i];
    for (int64_t j = 0; j < function_->node_count(); ++j) {
      Node *to = index_to_node_[j];
      int64_t delay = indices_to_delay_[i][j];

      if (delay < 0) {
        continue;
      }
      if (options.exclude_single_node_path && from == to) {
        continue;
      }
      if (options.exclude_param_source && from->Is<Param>()) {
        continue;
      }

      if (options.combinational_only) {
        const ScheduleCycleMap &cycle_map = *options.cycle_map;

        // Because we only collect combinational paths, we always skip a path
        // that is crossing different pipeline stages.
        if (cycle_map.at(from) != cycle_map.at(to)) {
          continue;
        }
        // If the source has operands and all operands are scheduled in the same
        // clock cycle with the source, indicating the source is an internal
        // node, skip it if applicable.
        if (options.input_source_only && !from->operands().empty() &&
            std::all_of(from->operands().begin(), from->operands().end(),
                        [&](Node *operand) {
                          return cycle_map.at(operand) == cycle_map.at(from) &&
                                 !operand->Is<Param>();
                        })) {
          continue;
        }
        // If the target has users and all users are scheduled in the same clock
        // cycle with the target, indicating the target is an internal node,
        // skip it if applicable.
        if (options.output_target_only && !to->users().empty() &&
            std::all_of(to->users().begin(), to->users().end(),
                        [&](Node *user) {
                          return cycle_map.at(user) == cycle_map.at(to);
                        })) {
          continue;
        }
      } else {
        if (options.input_source_only && !from->operands().empty()) {
          continue;
        }
        if (options.output_target_only && !to->users().empty()) {
          continue;
        }
      }

      worklist.emplace_back(score(from, to), delay, from, to);
    }
  }

  std::sort(worklist.begin(), worklist.end(), [](const auto &a, const auto &b) {
    return std::get<0>(a) > std::get<0>(b) ||
           (std::get<0>(a) == std::get<0>(b) &&
            std::get<1>(a) > std::get<1>(b));
  });

  absl::flat_hash_set<Node *> targets;
  std::vector<PathInfo> paths;
  int64_t counter = 0;
  for (auto [path_score, delay, source, target] : worklist) {
    if (options.unique_target_only && targets.contains(target)) {
      continue;
    }
    targets.emplace(target);
    if (except(source, target)) {
      continue;
    }
    paths.emplace_back(delay, source, target);
    counter++;
    if (counter >= number_paths) {
      break;
    }
  }
  return paths;
}

absl::StatusOr<std::vector<PathInfo>> DelayManager::GetTopNPathsStochastically(
    int64_t number_paths, float stochastic_ratio,
    const PathExtractOptions &options, absl::BitGenRef bit_gen,
    absl::FunctionRef<bool(Node *, Node *)> except,
    absl::FunctionRef<float(Node *, Node *)> score) const {
  int64_t number_candidate_paths =
      static_cast<int64_t>(static_cast<float>(number_paths) / stochastic_ratio);
  XLS_ASSIGN_OR_RETURN(
      std::vector<PathInfo> candidate_paths,
      GetTopNPaths(number_candidate_paths, options, except, score));
  if (candidate_paths.size() <= number_paths) {
    return candidate_paths;
  }

  // Randomly sample number_paths.
  std::vector<PathInfo> paths;
  std::sample(candidate_paths.begin(), candidate_paths.end(),
              std::back_inserter(paths), number_paths, bit_gen);
  return paths;
}

absl::StatusOr<PathInfo> DelayManager::GetLongestPath(
    const PathExtractOptions &options,
    absl::FunctionRef<bool(Node *, Node *)> except) const {
  XLS_ASSIGN_OR_RETURN(std::vector<PathInfo> critical_paths,
                       GetTopNPaths(1, options, except));

  if (critical_paths.empty()) {
    return PathInfo(0, nullptr, nullptr);
  }
  return critical_paths.front();
}

}  // namespace xls
