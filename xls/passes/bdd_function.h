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

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xls/common/logging/logging.h"
#include "xls/data_structures/binary_decision_diagram.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/function.h"
#include "xls/ir/op.h"

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
  // Construct a BDD representing the given function/proc. 'minterm_limit' is an
  // upper bound on the number of minterms in an expression. If a BDD node
  // associated with a particular bit in the function ({Node*, bit index} pair)
  // exceeds this value the bit's representation in the BDD is replaced with a
  // new BDD variable. If a node's op is in 'do_not_evaluate_ops', its
  // bits are modeled as BDD variables. Otherwise, bits are represented as BDD
  // nodes whose values are determined by the values of other BDD nodes.
  static absl::StatusOr<std::unique_ptr<BddFunction>> Run(
      FunctionBase* f, int64_t minterm_limit = 0,
      absl::Span<const Op> do_not_evaluate_ops = {});

  // Returns the underlying BDD.
  const BinaryDecisionDiagram& bdd() const { return bdd_; }
  BinaryDecisionDiagram& bdd() { return bdd_; }

  // Returns the node associated with the given bit.
  BddNodeIndex GetBddNode(Node* node, int64_t bit_index) const {
    XLS_CHECK(node->GetType()->IsBits());
    return node_map_.at(node).at(bit_index);
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

  // Map containing the Nodes whose expressions exceeded the maximum number of
  // minterms.
  absl::flat_hash_set<Node*> saturated_expressions_;
};

}  // namespace xls

#endif  // XLS_PASSES_BDD_FUNCTION_H_
