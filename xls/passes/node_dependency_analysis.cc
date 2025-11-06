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

#include "xls/passes/node_dependency_analysis.h"

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xls/ir/node.h"

namespace xls {

absl::flat_hash_set<Node*> NodeForwardDependencyAnalysis::ComputeInfo(
    Node* to,
    absl::Span<const absl::flat_hash_set<Node*>* const> operand_infos) const {
  int max_size = 0;
  for (const absl::flat_hash_set<Node*>* operand_info : operand_infos) {
    max_size += operand_info->size();
  }
  absl::flat_hash_set<Node*> can_reach_to{to};
  can_reach_to.reserve(max_size + 1);
  for (const absl::flat_hash_set<Node*>* operand_info : operand_infos) {
    can_reach_to.insert(operand_info->begin(), operand_info->end());
  }
  return can_reach_to;
}

absl::Status NodeForwardDependencyAnalysis::MergeWithGiven(
    absl::flat_hash_set<Node*>& info,
    const absl::flat_hash_set<Node*>& given) const {
  info.insert(given.begin(), given.end());
  return absl::OkStatus();
}

absl::flat_hash_set<Node*> NodeBackwardDependencyAnalysis::ComputeInfo(
    Node* from,
    absl::Span<const absl::flat_hash_set<Node*>* const> user_infos) const {
  int max_size = 0;
  for (const absl::flat_hash_set<Node*>* user_info : user_infos) {
    max_size += user_info->size();
  }
  absl::flat_hash_set<Node*> can_reach_from{from};
  can_reach_from.reserve(max_size + 1);
  for (const absl::flat_hash_set<Node*>* user_info : user_infos) {
    can_reach_from.insert(user_info->begin(), user_info->end());
  }
  return can_reach_from;
}

absl::Status NodeBackwardDependencyAnalysis::MergeWithGiven(
    absl::flat_hash_set<Node*>& info,
    const absl::flat_hash_set<Node*>& given) const {
  info.insert(given.begin(), given.end());
  return absl::OkStatus();
}

}  // namespace xls
