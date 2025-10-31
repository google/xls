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

#include "xls/passes/reachability_analysis.h"

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xls/ir/node.h"

namespace xls {

bool ReachabilityAnalysis::IsReachableFrom(Node* end, Node* start) const {
  return GetInfo(end)->contains(start);
}

absl::flat_hash_set<Node*> ReachabilityAnalysis::ComputeInfo(
    Node* end,
    absl::Span<const absl::flat_hash_set<Node*>* const> operand_infos) const {
  int max_size = 0;
  for (const absl::flat_hash_set<Node*>* operand_info : operand_infos) {
    max_size += operand_info->size();
  }
  absl::flat_hash_set<Node*> can_reach_end{end};
  can_reach_end.reserve(max_size + 1);
  for (const absl::flat_hash_set<Node*>* operand_info : operand_infos) {
    can_reach_end.insert(operand_info->begin(), operand_info->end());
  }
  return can_reach_end;
}

absl::Status ReachabilityAnalysis::MergeWithGiven(
    absl::flat_hash_set<Node*>& info,
    const absl::flat_hash_set<Node*>& given) const {
  info.insert(given.begin(), given.end());
  return absl::OkStatus();
}

}  // namespace xls
