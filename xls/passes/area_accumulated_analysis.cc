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

#include "xls/passes/area_accumulated_analysis.h"

#include <algorithm>
#include <cstdint>
#include <queue>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xls/ir/node.h"
#include "xls/passes/lazy_node_data.h"

namespace xls {

double AreaAccumulatedAnalysis::GetAreaThroughToNode(
    Node* accumulated_to_node) const {
  auto it = area_through_to_node_.find(accumulated_to_node);
  if (it != area_through_to_node_.end()) {
    return it->second;
  }

  absl::flat_hash_set<Node*> visited;
  std::queue<Node*> worklist;
  visited.insert(accumulated_to_node);
  worklist.push(accumulated_to_node);
  double total_area = 0.0;
  while (!worklist.empty()) {
    Node* next = worklist.front();
    worklist.pop();
    double area = *GetInfo(next);
    total_area += area;
    for (Node* operand : next->operands()) {
      if (visited.contains(operand)) {
        continue;
      }
      visited.insert(operand);
      worklist.push(operand);
    }
  }

  area_through_to_node_[accumulated_to_node] = total_area;
  return total_area;
}

double AreaAccumulatedAnalysis::ComputeInfo(
    Node* node, absl::Span<const double* const> operand_infos) const {
  auto area_status = area_estimator_->GetOperationAreaInSquareMicrons(node);
  CHECK_OK(area_status.status());
  return *area_status;
}

absl::Status AreaAccumulatedAnalysis::MergeWithGiven(
    double& info, const double& given) const {
  info = std::max(info, given);
  return absl::OkStatus();
}

void AreaAccumulatedAnalysis::NodeDeleted(Node* node) {
  LazyNodeData<double>::NodeDeleted(node);
  area_through_to_node_.erase(node);
}

void AreaAccumulatedAnalysis::ForgetAccumulatedAreaDependingOn(Node* node) {
  absl::flat_hash_set<Node*> visited;
  std::queue<Node*> worklist;
  visited.insert(node);
  worklist.push(node);
  while (!worklist.empty()) {
    Node* next = worklist.front();
    worklist.pop();
    area_through_to_node_.erase(next);
    for (auto user : next->users()) {
      if (visited.contains(user)) {
        continue;
      }
      visited.insert(user);
      worklist.push(user);
    }
  }
}

void AreaAccumulatedAnalysis::OperandChanged(
    Node* node, Node* old_operand, absl::Span<const int64_t> operand_nos) {
  LazyNodeData<double>::OperandChanged(node, old_operand, operand_nos);
  ForgetAccumulatedAreaDependingOn(node);
}

void AreaAccumulatedAnalysis::OperandRemoved(Node* node, Node* old_operand) {
  LazyNodeData<double>::OperandRemoved(node, old_operand);
  ForgetAccumulatedAreaDependingOn(node);
}

void AreaAccumulatedAnalysis::OperandAdded(Node* node) {
  LazyNodeData<double>::OperandAdded(node);
  ForgetAccumulatedAreaDependingOn(node);
}

}  // namespace xls
