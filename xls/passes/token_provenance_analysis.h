// Copyright 2022 The XLS Authors
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

#ifndef XLS_PASSES_TOKEN_PROVENANCE_ANALYSIS_H_
#define XLS_PASSES_TOKEN_PROVENANCE_ANALYSIS_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"

namespace xls {

using TokenProvenance = absl::flat_hash_map<
    Node*, std::unique_ptr<SharedLeafTypeTree<absl::flat_hash_set<Node*>>>>;

// Compute, for each token-type in the given `FunctionBase*`, what
// side-effecting node(s) contributed to that token. If a leaf type in one of
// the `LeafTypeTree`s is not a token, the corresponding `Node*` will be
// `nullptr`.
absl::StatusOr<TokenProvenance> TokenProvenanceAnalysis(FunctionBase* f);

std::string ToString(const TokenProvenance& provenance);

// A map from side-effecting nodes (and AfterAll) to the set of side-effecting
// nodes (/ AfterAll) that their token inputs immediately came from. Note that
// this skips over intermediate movement of tokens through tuples, `identity`,
// or selects.
using TokenDAG = absl::flat_hash_map<Node*, absl::flat_hash_set<Node*>>;

// Compute the immediate preceding side-effecting nodes (including proc token
// param and `after_all`s) for each side-effecting node (and after_all). Note
// that this skips over intermediate movement of token through tuples or
// `identity`, and that the proc token param does not appear as a key in the
// result map.
absl::StatusOr<TokenDAG> ComputeTokenDAG(FunctionBase* f);

struct NodeAndPredecessors {
  Node* node;

  using PredecessorSet = absl::btree_set<Node*, Node::NodeIdLessThan>;
  PredecessorSet predecessors;
  template <typename Sink>
  friend void AbslStringify(Sink& sink, const NodeAndPredecessors& p) {
    absl::Format(&sink, "%v: {%s}", *p.node,
                 absl::StrJoin(p.predecessors, ", "));
  }
};

// Returns a predecessor-list representation of the token graph connecting
// side-effecting operations in the given `proc`. The returns nodes will be in a
// topological sort.
absl::StatusOr<std::vector<NodeAndPredecessors>> ComputeTopoSortedTokenDAG(
    FunctionBase* f);

}  // namespace xls

#endif  // XLS_PASSES_TOKEN_PROVENANCE_ANALYSIS_H_
