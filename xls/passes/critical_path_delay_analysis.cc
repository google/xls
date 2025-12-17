// Copyright 2025 The XLS Authors
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

#include "xls/passes/critical_path_delay_analysis.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/ir/node.h"
#include "xls/passes/lazy_dag_cache.h"
#include "xls/passes/lazy_node_data.h"

namespace xls {

CriticalPathDelayAnalysis::CriticalPathDelayAnalysis(
    const DelayEstimator* estimator)
    : LazyNodeData<int64_t>(DagCacheInvalidateDirection::kInvalidatesUsers),
      delay_estimator_(estimator) {
  CHECK(delay_estimator_ != nullptr);
}

std::vector<Node*> CriticalPathDelayAnalysis::NodesAtEndOfCriticalPath(
    FunctionBase* f) const {
  int64_t max_delay = -1;
  absl::flat_hash_set<Node*> terminal_nodes;
  for (Node* node : f->nodes()) {
    if (!node->users().empty()) {
      continue;
    }
    terminal_nodes.insert(node);
    int64_t delay = *GetInfo(node);
    if (max_delay < delay) {
      max_delay = delay;
    }
  }
  std::vector<Node*> max_delay_nodes;
  for (Node* node : terminal_nodes) {
    if (*GetInfo(node) == max_delay) {
      max_delay_nodes.push_back(node);
    }
  }
  return max_delay_nodes;
}

int64_t CriticalPathDelayAnalysis::ComputeInfo(
    Node* node, absl::Span<const int64_t* const> operand_infos) const {
  int64_t max_operand_arrival_time = 0;
  for (const int64_t* op_info : operand_infos) {
    if (op_info == nullptr) {
      continue;
    }
    max_operand_arrival_time = std::max(max_operand_arrival_time, *op_info);
  }
  absl::StatusOr<int64_t> delay = delay_estimator_->GetOperationDelayInPs(node);

  // If estimator returns error or negative delay, treat delay as 0.
  int64_t node_delay = 0;
  if (delay.ok() && *delay > 0) {
    node_delay = *delay;
  }

  int64_t node_arrival_time = node_delay + max_operand_arrival_time;
  return node_arrival_time;
}

absl::Status CriticalPathDelayAnalysis::MergeWithGiven(
    int64_t& info, const int64_t& given) const {
  // Arrival time is lower-bounded by givens.
  info = std::max(info, given);
  return absl::OkStatus();
}

}  // namespace xls
