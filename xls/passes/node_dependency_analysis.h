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

#include <cstdint>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/inline_bitmap.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"

namespace xls {

class NodeDependencyAnalysis;
class DependencyBitmap {
 public:
  DependencyBitmap(const DependencyBitmap&) = default;
  DependencyBitmap(DependencyBitmap&&) = default;
  // Deleted because bitmap_ is const reference.
  DependencyBitmap& operator=(const DependencyBitmap&) = delete;
  DependencyBitmap& operator=(DependencyBitmap&&) = delete;

  const InlineBitmap& bitmap() const { return bitmap_; }
  const absl::flat_hash_map<Node*, int64_t>& node_indices() const {
    return node_indices_;
  }
  absl::StatusOr<bool> IsDependent(Node* n) const {
    if (!node_indices_.contains(n)) {
      return absl::InvalidArgumentError("node is from a different function!");
    }
    return bitmap_.Get(node_indices_.at(n));
  }

 private:
  DependencyBitmap(const InlineBitmap& bitmap ABSL_ATTRIBUTE_LIFETIME_BOUND,
                   const absl::flat_hash_map<Node*, int64_t>& node_indices
                       ABSL_ATTRIBUTE_LIFETIME_BOUND)
      : bitmap_(bitmap), node_indices_(node_indices) {}
  const InlineBitmap& bitmap_;
  const absl::flat_hash_map<Node*, int64_t>& node_indices_;
  friend class NodeDependencyAnalysis;
};

// Analysis which lets us check whether different nodes are connected or over a
// horizon from each other.
class NodeDependencyAnalysis {
 public:
  NodeDependencyAnalysis(NodeDependencyAnalysis&&) = default;
  NodeDependencyAnalysis(const NodeDependencyAnalysis&) = default;
  NodeDependencyAnalysis& operator=(NodeDependencyAnalysis&&) = default;
  NodeDependencyAnalysis& operator=(const NodeDependencyAnalysis&) = default;

  // Analyze the forward dependents of the given nodes. That is find the nodes
  // which are fed by a given nodes.
  //
  // Optionally provide the set of nodes we will care about which will cause
  // this to only calculate the dependents for those given nodes.
  static NodeDependencyAnalysis ForwardDependents(
      FunctionBase* fb, absl::Span<Node* const> nodes = {});

  // Analyze the backwards dependents of the given nodes. That is find the nodes
  // which feed any a given node.
  //
  // Optionally provide the set of nodes we will care about which will cause
  // this to only calculate the dependents for those given nodes.
  static NodeDependencyAnalysis BackwardDependents(
      FunctionBase* fb, absl::Span<Node* const> nodes = {});

  // Returns if this is a forwards-dependency relationship. That is if
  // 'IsDependent(X, Y)' implies that a change in X could cause a change in Y.
  bool IsForward() const { return is_forward_; }

  // Returns if the dependents are analyzed for this node. If this returns false
  // other calls will return error.
  bool IsAnalyzed(Node* node) const { return dependents_.contains(node); }

  // Get the bitmap for Node->GetId() -> bool for dependents of 'node'. Return
  // is owned by the NodeDependencyAnalysis object.
  absl::StatusOr<DependencyBitmap> GetDependents(Node* node) const;

  // Return if 'to' is a dependent of 'from'
  absl::StatusOr<bool> IsDependent(Node* from, Node* to) const {
    XLS_ASSIGN_OR_RETURN(auto bitmap, GetDependents(from));
    return bitmap.IsDependent(to);
  }
  const absl::flat_hash_map<Node*, int64_t>& node_indices() const {
    return node_indices_;
  }

 private:
  NodeDependencyAnalysis(bool is_forwards,
                         absl::flat_hash_map<Node*, InlineBitmap> dependents,
                         absl::flat_hash_map<Node*, int64_t> node_ids)
      : is_forward_(is_forwards),
        dependents_(std::move(dependents)),
        node_indices_(std::move(node_ids)) {}

  bool is_forward_;
  absl::flat_hash_map<Node*, InlineBitmap> dependents_;
  absl::flat_hash_map<Node*, int64_t> node_indices_;
};

}  // namespace xls

#endif  // XLS_PASSES_NODE_DEPENDENCY_ANALYSIS_H_
