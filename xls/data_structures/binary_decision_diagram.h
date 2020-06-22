// Copyright 2020 Google LLC
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

#ifndef XLS_DATA_STRUCTURES_BINARY_DECISION_DIAGRAM_H_
#define XLS_DATA_STRUCTURES_BINARY_DECISION_DIAGRAM_H_

#include "absl/container/flat_hash_map.h"
#include "xls/common/integral_types.h"
#include "xls/common/status/statusor.h"
#include "xls/common/strong_int.h"

namespace xls {

// An efficient implementation of binary decision diagrams (BDD). This
// implementation allows an arbitrary number of expressions over a set of
// variables to be represented in a single BDD.
//
// Based on:
//   K.S. Brace, R.L. Rudell, and R.E. Bryant,
//   "Efficient Implementation of a BDD package"
//   https://ieeexplore.ieee.org/document/114826

// For efficiency variables and nodes are referred to by indices into vector
// data members in the BDD.
DEFINE_STRONG_INT_TYPE(BddVariable, int32);
DEFINE_STRONG_INT_TYPE(BddNodeIndex, int32);

// A node in the BDD. The node is associated with a single variable and has
// children corresponding to when the variable is true (high) and when it is
// false (low).
struct BddNode {
  BddNode() : variable(0), high(0), low(0), minterm_count(0) {}
  BddNode(BddVariable v, BddNodeIndex h, BddNodeIndex l, int32 m)
      : variable(v), high(h), low(l), minterm_count(m) {}

  BddVariable variable;
  BddNodeIndex high;
  BddNodeIndex low;

  // Number of minterms in the expression. A minterm is a term in a
  // sum-of-products form of the boolean function, or equivalently a path to the
  // leaf node '1' from the BDD node. Saturates at INT32_MAX.
  int32 minterm_count;
};

class BinaryDecisionDiagram {
 public:
  // Creates an empty BDD. Initialize the BDD contains only the nodes
  // corresponding to zero and one.
  BinaryDecisionDiagram();

  // Adds a new variable to the BDD and returns the node corresponding the
  // variable's value.
  BddNodeIndex NewVariable();

  // Returns the inverse of the given expression.
  BddNodeIndex Not(BddNodeIndex expr);

  // Returns the OR/AND of the given expressions.
  BddNodeIndex And(BddNodeIndex a, BddNodeIndex b);
  BddNodeIndex Or(BddNodeIndex a, BddNodeIndex b);

  // Returns the leaf node corresponding to zero or one.
  BddNodeIndex zero() const { return BddNodeIndex(0); }
  BddNodeIndex one() const { return BddNodeIndex(1); }

  // Evaluates the given expression with the given variable values. The keys in
  // the map are the *node* indices of the respective variable (value returned
  // by NewVariable). An error is returned if the map is missing a variable
  // required to evaluate the expression.
  xabsl::StatusOr<bool> Evaluate(
      BddNodeIndex expr,
      const absl::flat_hash_map<BddNodeIndex, bool>& variable_values) const;

  // Returns the BDD node with the given index.
  const BddNode& GetNode(BddNodeIndex node_index) const {
    return nodes_.at(node_index.value());
  }

  // Returns the number of nodes in the graph.
  int64 size() const { return nodes_.size(); }

  // Returns the number of variables in the graph.
  int64 variable_count() const { return next_var_.value(); }

  // Returns the number of minterms in the given expression.
  int64 minterm_count(BddNodeIndex expr) const {
    return GetNode(expr).minterm_count;
  }

  // Returns the given expression in disjunctive normal form (sum of products).
  // The expression is not minimal. 'minterm_limit' is the maximum number of
  // minterms to emit before truncating the output.
  std::string ToStringDnf(BddNodeIndex expr, int64 minterm_limit = 0) const;

  // Returns true if the given node is the "base" node for its respective
  // variable. The expression of a base node is exactly equal to the value of
  // the variable.
  bool IsVariableBaseNode(BddNodeIndex expr) const {
    return GetNode(expr).high == one() && GetNode(expr).low == zero();
  }

 private:
  // Helper for constructing a DNF string respresentation.
  void ToStringDnfHelper(BddNodeIndex expr, int64* minterms_to_emit,
                         std::vector<std::string>* terms,
                         std::string* str) const;

  // Get the node corresponding to the given variable with the given low/high
  // children. Creates it if it does not exist.
  BddNodeIndex GetOrCreateNode(BddVariable var, BddNodeIndex high,
                               BddNodeIndex low);

  // Returns the node equal to given expression with the given variable
  // set to the given value.
  BddNodeIndex Restrict(BddNodeIndex expr, BddVariable var, bool value);

  // Returns the node corresponding to the given if-then-else expression.
  BddNodeIndex IfThenElse(BddNodeIndex cond, BddNodeIndex if_true,
                          BddNodeIndex if_false);

  // Returns the node corresponding to the value of the given variable.
  BddNodeIndex GetVariableBaseNode(BddVariable variable) const {
    return node_map_.at({variable, one(), zero()});
  }

  // The numeric id to use for the next created variable. Increments with each
  // call to NewVariable which
  BddVariable next_var_ = BddVariable(0);

  // The vector of all the nodes in the BDD.
  std::vector<BddNode> nodes_;

  // A map from BDD node content (variable id, high child, low child) to the
  // index of the respective node. This map is used to ensure that no duplicate
  // nodes are created.
  using NodeKey = std::tuple<BddVariable, BddNodeIndex, BddNodeIndex>;
  absl::flat_hash_map<NodeKey, BddNodeIndex> node_map_;

  // A map from if-then-else expression to the node corresponding to that
  // expression. The key elements are (condition, if-true, if-false). This map
  // enables fast lookup for expressions.
  using IteKey = std::tuple<BddNodeIndex, BddNodeIndex, BddNodeIndex>;
  absl::flat_hash_map<IteKey, BddNodeIndex> ite_map_;
};

}  // namespace xls

#endif  // XLS_DATA_STRUCTURES_BINARY_DECISION_DIAGRAM_H_
