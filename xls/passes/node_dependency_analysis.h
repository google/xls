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

#ifndef XLS_PASSES_NODE_DEPENDENCY_ANALYSIS_H_
#define XLS_PASSES_NODE_DEPENDENCY_ANALYSIS_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xls/ir/node.h"
#include "xls/passes/lazy_node_data.h"

namespace xls {

class NodeForwardDependencyAnalysis
    : public LazyNodeData<absl::flat_hash_set<Node*>> {
 public:
  bool IsDependent(Node* from, Node* to) const {
    return GetInfo(to)->contains(from);
  }

  absl::flat_hash_set<Node*> NodesDependedOnBy(Node* to) const {
    return *GetInfo(to);
  }

 protected:
  absl::flat_hash_set<Node*> ComputeInfo(
      Node* to,
      absl::Span<const absl::flat_hash_set<Node*>* const> operand_infos)
      const override;

  absl::Status MergeWithGiven(
      absl::flat_hash_set<Node*>& info,
      const absl::flat_hash_set<Node*>& given) const override;
};

class NodeBackwardDependencyAnalysis
    : public LazyNodeData<absl::flat_hash_set<Node*>> {
 public:
  bool IsDependent(Node* from, Node* to) const {
    return GetInfo(from)->contains(to);
  }

  absl::flat_hash_set<Node*> NodesDependingOn(Node* from) const {
    return *GetInfo(from);
  }

 protected:
  absl::flat_hash_set<Node*> ComputeInfo(
      Node* from,
      absl::Span<const absl::flat_hash_set<Node*>* const> user_infos)
      const override;

  absl::Status MergeWithGiven(
      absl::flat_hash_set<Node*>& info,
      const absl::flat_hash_set<Node*>& given) const override;

  // Propagate from users to operands
  absl::Span<Node* const> GetInputs(Node* const& node) const override {
    return node->users();
  }
  absl::Span<Node* const> GetUsers(Node* const& node) const override {
    return node->operands();
  }
};

}  // namespace xls

#endif  // XLS_PASSES_NODE_DEPENDENCY_ANALYSIS_H_
