// Copyright 2020 The XLS Authors
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

#ifndef XLS_PASSES_BDD_FUNCTION_H_
#define XLS_PASSES_BDD_FUNCTION_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/data_structures/binary_decision_diagram.h"
#include "xls/ir/function_base.h"
#include "xls/ir/op.h"
#include "xls/ir/value.h"

namespace xls {

using BddNodeVector = std::vector<BddNodeIndex>;
using NodeMap = absl::flat_hash_map<const Node*, BddNodeVector>;

// A class which represents an XLS function using a binary decision diagram
// (BDD). The BDD is constructed by an abstract evaluation of the operations in
// the function using compositions of the And/Or/Not functions of the BDD. Only
// a subset of the function's nodes are evaluated with the BDD (defined by
// BddFunction::IsExpressedInBdd). For example, arithmetic operations are not
// evaluated as they generally produce very large BDDs. Non-bits types are
// skipped as well.
//
// For each bits-typed XLS Node, BddFunction holds a BddNodeVector which is a
// vector of BDD nodes corresponding to the expression for each bit in the XLS
// Node output.
class BddFunction {
 public:
  // The default limit on the number of paths from a BDD node to the BDD
  // terminal nodes 0 and 1. If a BDD node associated with
  // a particular bit in the function ({Node*, bit index} pair) exceeds this
  // value the bit's representation in the BDD is replaced with a new BDD
  // variable. This provides a mechanism for limiting the growth of the BDD.
  static constexpr int64_t kDefaultPathLimit = 1024;

  // Construct a BDD representing the given function/proc.
  // `node_filter` is an optional function which filters the nodes to be
  // evaluated. If this function returns false for a node then the node will not
  // be evaluated using BDDs. The node's bits will be new variables in the BDD
  // for which no information is known. If `node_filter` returns true, the node
  // still might *not* be evaluated because some kinds of nodes are never
  // evaluated for various reasons including computation expense.
  static absl::StatusOr<std::unique_ptr<BddFunction>> Run(
      FunctionBase* f, int64_t path_limit = 0,
      std::optional<std::function<bool(const Node*)>> node_filter =
          std::nullopt);

  // Returns the underlying BDD.
  const BinaryDecisionDiagram& bdd() const { return bdd_; }
  BinaryDecisionDiagram& bdd() { return bdd_; }

  FunctionBase* function_base() const { return func_base_; }

  // Returns the node associated with the given bit.
  BddNodeIndex GetBddNode(Node* node, int64_t bit_index) const {
    std::optional<BddNodeIndex> bdd_node = TryGetBddNode(node, bit_index);
    CHECK(bdd_node.has_value());
    return *std::move(bdd_node);
  }
  std::optional<BddNodeIndex> TryGetBddNode(Node* node,
                                            int64_t bit_index) const {
    CHECK(node->GetType()->IsBits());
    auto it = node_map_.find(node);
    if (it == node_map_.end()) {
      return std::nullopt;
    }
    return it->second.at(bit_index);
  }

  // Evaluates the function using the BDD with the given argument values.
  // Operations such as arithmetic operations which are not expressed in the BDD
  // are evaluated using the IR interpreter. This method is for testing purposes
  // only for verifying that the BDD is properly constructed. Prerequisite: the
  // FunctionBase used to build the BddFunction must be a function not a proc.
  absl::StatusOr<Value> Evaluate(absl::Span<const Value> args) const;

 private:
  explicit BddFunction(FunctionBase* f) : func_base_(f) {}

  FunctionBase* func_base_;
  BinaryDecisionDiagram bdd_;

  // A map from XLS Node to vector of BDD nodes representing the XLS Node's
  // expression.
  NodeMap node_map_;

  // Set containing the Nodes which have exceeded the maximum number of paths
  // from the XLS node's BDD node to the terminal nodes 0 and 1 in the
  // BDD. These are the XLS Nodes for which it was determined the precisely
  // computing the expression for the node using the BDD was too expensive.
  absl::flat_hash_set<Node*> saturated_expressions_;
};

// Returns true if the given node is very cheap to evaluate using a
// BDD. Typically single-bit and logical operations are considered cheap as well
// as "free" operations like bitslice and concat.
bool IsCheapForBdds(const Node* node);

}  // namespace xls

#endif  // XLS_PASSES_BDD_FUNCTION_H_
