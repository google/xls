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

#ifndef XLS_PASSES_REACHABILITY_ANALYSIS_H_
#define XLS_PASSES_REACHABILITY_ANALYSIS_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xls/ir/node.h"
#include "xls/passes/lazy_node_data.h"

namespace xls {

class ReachabilityAnalysis : public LazyNodeData<absl::flat_hash_set<Node*>> {
 public:
  // Returns if `end` can be reached by traversing users beginning at `start`.
  bool IsReachableFrom(Node* end, Node* start) const;

  const absl::flat_hash_set<Node*>& NodesThatCanReach(Node* end) const {
    return *GetInfo(end);
  }

 protected:
  // Computes the set of nodes reach-able from `start` by traversing users.
  absl::flat_hash_set<Node*> ComputeInfo(
      Node* end,
      absl::Span<const absl::flat_hash_set<Node*>* const> operand_infos)
      const override;

  absl::Status MergeWithGiven(
      absl::flat_hash_set<Node*>& info,
      const absl::flat_hash_set<Node*>& given) const override;
};

}  // namespace xls

#endif  // XLS_PASSES_REACHABILITY_ANALYSIS_H_
