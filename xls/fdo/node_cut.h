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

#ifndef XLS_FDO_NODE_CUT_H_
#define XLS_FDO_NODE_CUT_H_

#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "xls/common/logging/logging.h"
#include "xls/ir/node.h"
#include "xls/ir/op.h"
#include "xls/scheduling/scheduling_options.h"

namespace xls {

class NodeCut;

// Mappings from a node to a cut or a vector of cuts rooted at it. Due to the
// instability of pointers, these maps will not guarantee a deterministic order
// of item traversing. As a result, one should not reply on item traversing of
// these maps for any purpose that demands deterministicity. Our cut-formation
// algorithm only performs lookups in this map that come from traversing the
// graph in deterministic ways from the graph's structure, so the fact the keys
// are unordered is irrelevant.
using NodeCutMap = absl::flat_hash_map<Node *, NodeCut>;
using NodeCutsMap = absl::flat_hash_map<Node *, std::vector<NodeCut>>;

// A cut C of node n is a set of nodes, called leaves, such that:
//    1. Each path from any primary input (PI) of the DAG (Directed Acyclic
//       Graph) to n passes through a leaf.
//    2. For each leaf, there is a path from an input to n passing through the
//       leaf and not through any other leaf.
// Node n is called the root of C. Naturally, A trivial cut of node n is the cut
// {n} composed of the node itself. A non-trivial cut is said to "cover" all the
// nodes found on the paths from the leaves to the root, including the root but
// excluding the leaves.
//
// Essentially, the concept of "leaf" should be corresponding to "value" in
// compilers. However, in the context of our IR, a "value" is equal to a "node"
// because every node defines either one or zero value. Therefore, we use "node"
// to represent leaves.
class NodeCut {
 public:
  explicit NodeCut(Node *root, const absl::flat_hash_set<Node *> &leaves = {})
      : root_(root), leaves_(leaves) {
    CHECK(root);
  }

  // Get the trivial cut of root.
  static NodeCut GetTrivialCut(Node *root) { return NodeCut(root, {root}); }

  // Merge the lhs and rhs cut at root.
  static NodeCut GetMergedCut(Node *root, const NodeCut &lhs,
                              const NodeCut &rhs);

  bool IsTrivial() const {
    return leaves_.size() == 1 && leaves_.contains(root_);
  }

  bool Includes(const NodeCut &other) const;

  // Get the node cone, which contains all nodes covered by the cut, including
  // the root, but excluding the leaves.
  absl::flat_hash_set<Node *> GetNodeCone() const;

  bool operator==(const NodeCut &other) const {
    return root_ == other.root_ && leaves_ == other.leaves();
  }
  bool operator!=(const NodeCut &other) const { return !(*this == other); }

  Node *root() const { return root_; }
  const absl::flat_hash_set<Node *> &leaves() const { return leaves_; }

 private:
  Node *root_;
  absl::flat_hash_set<Node *> leaves_;
};

template <typename H>
H AbslHashValue(H state, const NodeCut &cut) {
  return H::combine(std::move(state), cut.root(), cut.leaves());
}

// Enumerate the maximum cut of every node in the given schedule. Maximum cut is
// defined as a cut with all leaves being a primary inputs (PI) or pipeline
// register.
absl::StatusOr<NodeCutMap> EnumerateMaxCutInSchedule(
    FunctionBase *f, int64_t pipeline_length,
    const ScheduleCycleMap &cycle_map);

// Cuts enumeration is used to construct a cuts map. Note that the enumeration
// will be applied to each pipeline cycle separately. As a result, there will be
// no cut crossing different pipeline cycles. All cuts will have primary input
// (PI) or pipeline register as leaves if input_leaves_only is set true.
// TODO(hanchenye): 2023-08-14 We should add an optional argument to constrain
// the maximum number of leaves.
absl::StatusOr<NodeCutsMap> EnumerateCutsInSchedule(
    FunctionBase *f, int64_t pipeline_length, const ScheduleCycleMap &cycle_map,
    bool input_leaves_only = true);

}  // namespace xls

#endif  // XLS_FDO_NODE_CUT_H_
