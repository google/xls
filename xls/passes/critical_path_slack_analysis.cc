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

#include "xls/passes/critical_path_slack_analysis.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xls/ir/node.h"
#include "xls/passes/critical_path_delay_analysis.h"
#include "xls/passes/lazy_node_data.h"

namespace xls {

int64_t CriticalPathSlackAnalysis::SlackFromCriticalPath(Node* node) const {
  return *GetInfo(node);
}

int64_t CriticalPathSlackAnalysis::ComputeInfo(
    Node* node, absl::Span<const int64_t* const> user_infos) const {
  std::vector<Node*> ends_of_critical_path =
      critical_path_delay_analysis_->NodesAtEndOfCriticalPath(
          node->function_base());
  if (ends_of_critical_path.empty()) {
    return 0;
  }

  int64_t critical_path_delay =
      *critical_path_delay_analysis_->GetInfo(ends_of_critical_path[0]);
  int64_t node_delay = *critical_path_delay_analysis_->GetInfo(node);
  if (node->users().empty()) {
    return std::max((int64_t)0, critical_path_delay - node_delay);
  }

  int64_t min_slack = std::numeric_limits<int64_t>::max();
  for (Node* user : node->users()) {
    int64_t max_other_operand_delay = node_delay;
    for (Node* operand : user->operands()) {
      max_other_operand_delay =
          std::max(max_other_operand_delay,
                   *critical_path_delay_analysis_->GetInfo(operand));
    }

    // A node's slack w.r.t a user is the user's slack plus how much less this
    // node's delay is than the largest delay of the user's other operands.
    min_slack = std::min(min_slack,
                         *GetInfo(user) + max_other_operand_delay - node_delay);
  }

  return min_slack;
}

absl::Status CriticalPathSlackAnalysis::MergeWithGiven(
    int64_t& info, const int64_t& given) const {
  info = std::min(info, given);
  return absl::OkStatus();
}

void CriticalPathSlackAnalysis::NodeAdded(Node* node) {
  LazyNodeData<int64_t>::NodeAdded(node);
  ClearCache();
}
void CriticalPathSlackAnalysis::NodeDeleted(Node* node) {
  LazyNodeData<int64_t>::NodeDeleted(node);
  ClearCache();
}

void CriticalPathSlackAnalysis::OperandChanged(
    Node* node, Node* old_operand, absl::Span<const int64_t> operand_nos) {
  LazyNodeData<int64_t>::OperandChanged(node, old_operand, operand_nos);
  ClearCache();
}

void CriticalPathSlackAnalysis::OperandRemoved(Node* node, Node* old_operand) {
  LazyNodeData<int64_t>::OperandRemoved(node, old_operand);
  ClearCache();
}

void CriticalPathSlackAnalysis::OperandAdded(Node* node) {
  LazyNodeData<int64_t>::OperandAdded(node);
  ClearCache();
}

}  // namespace xls
