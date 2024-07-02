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

#ifndef XLS_SCHEDULING_MUTUAL_EXCLUSION_PASS_H_
#define XLS_SCHEDULING_MUTUAL_EXCLUSION_PASS_H_

#include <cstdint>
#include <optional>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/scheduling/scheduling_pass.h"

namespace xls {

// This stores a mapping from nodes in a FunctionBase to 1-bit nodes that are
// the "predicate" of that node. The idea is that it should always be sound to
// replace a node `N` that has predicate `P` with `gate(P, N)` (where `gate` is
// extended from its usual semantics to also gate the effect of side-effectful
// operations).
//
// It also contains a relation on the set of predicate nodes that describes
// whether any given pair of predicates is known to be mutually exclusive; i.e.:
// for a pair `(A, B)` whether `A NAND B` is known to be valid. If `A NAND B` is
// known not to be valid, that information is tracked too.
class Predicates {
 public:
  // Set the predicate of the given node to the given predicate node.
  void SetPredicate(Node* node, Node* pred);

  // Get the predicate of the given node.
  std::optional<Node*> GetPredicate(Node* node) const;

  // Get all nodes predicated by a given predicate node.
  absl::flat_hash_set<Node*> GetNodesPredicatedBy(Node* node) const;

  // Assert that the two given predicates are mutually exclusive.
  absl::Status MarkMutuallyExclusive(Node* pred_a, Node* pred_b);

  // Assert that the two given predicates are not mutually exclusive.
  absl::Status MarkNotMutuallyExclusive(Node* pred_a, Node* pred_b);

  // Query whether the two given predicates are known to be mutually exclusive
  // (`true`), known to not be mutually exclusive (`false`), or nothing is known
  // about them (`std::nullopt`).
  //
  // For all `P` and `Q`,
  // `QueryMutuallyExclusive(P, Q) == QueryMutuallyExclusive(Q, P)`.
  std::optional<bool> QueryMutuallyExclusive(Node* pred_a, Node* pred_b) const;

  // Returns all neighbors of the given predicate in the mutual exclusion graph.
  // The return value of `MutualExclusionNeighbors(P)` should be all `Q` such
  // that `QueryMutuallyExclusive(P, Q).has_value()`.
  absl::flat_hash_map<Node*, bool> MutualExclusionNeighbors(Node* pred) const;

  // Update the metadata contained within the `Predicates` to respect the
  // replacement of a node by another node.
  void ReplaceNode(Node* original, Node* replacement);

 private:
  absl::flat_hash_map<Node*, Node*> predicated_by_;
  absl::flat_hash_map<Node*, absl::flat_hash_set<Node*>> predicate_of_;

  // The `bool` represents knowledge about the mutual exclusion of two nodes;
  // if it is true then the two nodes are mutually exclusive, if it is false
  // then they are known to not be mutually exclusive.
  //
  // Invariant: if mutual_exclusion_.at(a).contains(b) then
  // mutual_exclusion.at(b).contains(a), i.e.: this is a symmetric relation;
  // this fact is used to make garbage collection efficient.
  //
  // Invariant: all nodes must have Bits type and bitwidth 1
  absl::flat_hash_map<Node*, absl::flat_hash_map<Node*, bool>>
      mutual_exclusion_;
};

// Add a predicate to a node. If the node does not already have a predicate,
// this will simply set the predicate of the node to the given predicate and
// then return the given predicate. Otherwise, this will replace the predicate
// of the node with AND of the given predicate and the existing predicate,
// returning this new predicate.
absl::StatusOr<Node*> AddPredicate(Predicates* p, Node* node, Node* pred);

// Add predicates to all nodes postdominated by a select, where the predicate is
// of the form `selector == case_number`. This function goes to great lengths
// to not create messy/redundant predicates that would need to be cleaned up by
// another pass.
absl::Status AddSelectPredicates(Predicates* p, FunctionBase* f);

// Use an SMT solver to populate the given `Predicates*` with information about
// whether nodes are used in a mutually exclusive way.
absl::Status ComputeMutualExclusion(Predicates* p, FunctionBase* f,
                                    int64_t z3_rlimit);

// Pass which merges together nodes that are determined to be mutually exclusive
// via SMT solver analysis.
class MutualExclusionPass : public SchedulingOptimizationFunctionBasePass {
 public:
  MutualExclusionPass()
      : SchedulingOptimizationFunctionBasePass(
            "mutual_exclusion",
            "Merge mutually exclusively used nodes using SMT solver") {}
  ~MutualExclusionPass() override = default;

 protected:
  absl::StatusOr<bool> RunOnFunctionBaseInternal(
      FunctionBase* f, SchedulingUnit* unit,
      const SchedulingPassOptions& options,
      SchedulingPassResults* results) const override;
};

}  // namespace xls

#endif  // XLS_SCHEDULING_MUTUAL_EXCLUSION_PASS_H_
