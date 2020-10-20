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

#include "xls/passes/reassociation_pass.h"

#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/logging/logging.h"
#include "xls/common/math_util.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/node_iterator.h"
#include "xls/ir/node_util.h"
#include "xls/ir/op.h"

namespace xls {

// Walks an expression tree of operations with the given op and bit
// width. 'node' is the node currently being visited.  The leaves of the
// expression tree are added to 'leaves', and the interior nodes of the tree are
// added to 'interior_nodes'. Returns the height of the tree.
//
// For example, if called at the root of the following expression:
//
//   a    b   c    d
//    \  /     \  /
//   add.2    add.3
//       \    /
//       add.1
//
// Upon return the passed in vectors would be:
//
//   leaves = {a, b, c, d}
//   interior_nodes = {add.1, add.2, add.3}
//
// And the function would return 2, the depth of the tree (not counting the
// leaves).
int64 GatherExpressionLeaves(Op op, int64 bit_count, Node* node,
                             std::vector<Node*>* leaves,
                             std::vector<Node*>* interior_nodes) {
  if (node->op() != op || node->BitCountOrDie() != bit_count ||
      node->operand(0)->BitCountOrDie() != bit_count ||
      node->operand(1)->BitCountOrDie() != bit_count) {
    // 'node' does not match the other nodes of the tree and is thus a leaf.
    leaves->push_back(node);
    return 0;
  }
  // 'node' is an interior node in the tree of identical operations. Traverse
  // into the operands if the operand has a single use, otherwise the operand is
  // a leaf.
  interior_nodes->push_back(node);
  int64 max_depth = 1;
  for (Node* operand : node->operands()) {
    if (operand->users().size() == 1) {
      max_depth =
          std::max(max_depth, GatherExpressionLeaves(op, bit_count, operand,
                                                     leaves, interior_nodes) +
                                  1);
    } else {
      leaves->push_back(operand);
    }
  }
  return max_depth;
}

absl::StatusOr<bool> ReassociationPass::RunOnFunctionBase(
    FunctionBase* f, const PassOptions& options, PassResults* results) const {
  XLS_VLOG(2) << "Running reassociation on function " << f->name();
  XLS_VLOG(3) << "Before:";
  XLS_VLOG_LINES(3, f->DumpIr());

  bool changed = false;

  // Keep track of which nodes we've already considered for reassociation so we
  // don't revisit subexpressions multiple times.
  absl::flat_hash_set<Node*> visited_nodes;

  // Traverse the nodes in reverse order because we construct expressions for
  // reassociation starting from the roots.
  for (Node* node : ReverseTopoSort(f)) {
    if (visited_nodes.contains(node)) {
      continue;
    }

    // Only reassociate arithmetic operations. Logical operations can
    // theoretically be reassociated, but they are better handled by collapsing
    // into a single operation as logical operations are n-ary.
    if (node->op() != Op::kAdd && node->op() != Op::kUMul &&
        node->op() != Op::kSMul) {
      continue;
    }
    std::vector<Node*> leaves;
    std::vector<Node*> interior_nodes;
    int64 expression_depth = GatherExpressionLeaves(
        node->op(), node->BitCountOrDie(), node, &leaves, &interior_nodes);

    // Interior nodes in the expression will be reassociated so add them
    // to the set of associated nodes. This will prevent future
    visited_nodes.insert(interior_nodes.begin(), interior_nodes.end());

    // We want to reassociate under two conditions:
    //
    // (1) Reduce the height (delay) of the expression. An
    //     example is reassociating the following:
    //
    //              c   d
    //               \ /
    //            b   +
    //             \ /
    //          a    +
    //           \ /
    //            +
    //
    //     into:
    //
    //        a   b   c   d
    //         \ /     \ /
    //          +       +
    //            \   /
    //              +
    //
    // (2) The expression includes more than one literal. This enables some
    //     folding and canonicalization by putting the literals on the right.
    //     For example, reassociating the following:
    //
    //        a  C_0  b  C_1
    //         \ /     \ /
    //          +       +
    //            \   /
    //              +
    //
    //     into:
    //
    //        a   b  C_0  C_1
    //         \ /     \ /
    //          +       +
    //            \   /
    //              +
    //
    //     Then C_0 + C_1 can be folded.

    // First, separate the leaves of the expression into literals and
    // non-literals ('inputs').
    std::vector<Node*> literals;
    std::vector<Node*> inputs;
    for (Node* leaf : leaves) {
      if (leaf->Is<Literal>()) {
        literals.push_back(leaf);
      } else {
        inputs.push_back(leaf);
      }
    }

    // We only want to transform for one of the two cases above.
    if (interior_nodes.size() <= 1 ||
        (expression_depth == CeilOfLog2(leaves.size()) &&
         literals.size() <= 1)) {
      continue;
    }

    XLS_VLOG(4) << "Reassociated expression rooted at: "
                << inputs[0]->GetName();
    XLS_VLOG(4) << "  operations to reassociate:  "
                << absl::StrJoin(interior_nodes, ", ", NodeFormatter);
    XLS_VLOG(4) << "  leaves:  " << absl::StrJoin(leaves, ", ", NodeFormatter);

    // Create a clone of 'node' for construcing a reassociated expression.
    auto new_node = [&](Node* lhs, Node* rhs) -> absl::StatusOr<Node*> {
      return node->Clone({lhs, rhs});
    };

    if (literals.size() == 1) {
      // Only one literal in the expression. Just add it to the other inputs. It
      // will appear on the far right of the tree.
      inputs.push_back(literals.front());
    } else if (literals.size() > 1) {
      // More than one literal appears in the expression. Compute the result of
      // the literals separately so it will be folded, then append the result to
      // the other inputs.
      XLS_ASSIGN_OR_RETURN(Node * literal_expr,
                           new_node(literals[0], literals[1]));
      for (int64 i = 2; i < literals.size(); ++i) {
        XLS_ASSIGN_OR_RETURN(literal_expr, new_node(literals[i], literal_expr));
      }
      inputs.push_back(literal_expr);
    }

    // Reassociate the expressions into a balanced tree. First, reduce the
    // number of inputs to a power of two. Then build a balanced tree.
    if (!IsPowerOfTwo(inputs.size())) {
      // Number of operations to apply to the inputs to reduce operand count to
      // a power of two. These ops will be the ragged top layer of the
      // expression tree.
      int64 op_count = inputs.size() - (1ULL << FloorOfLog2(inputs.size()));
      std::vector<Node*> next_inputs;
      for (int64 i = 0; i < op_count; ++i) {
        XLS_ASSIGN_OR_RETURN(Node * new_op,
                             new_node(inputs[2 * i], inputs[2 * i + 1]));
        next_inputs.push_back(new_op);
      }
      for (int64 i = op_count * 2; i < inputs.size(); ++i) {
        next_inputs.push_back(inputs[i]);
      }
      inputs = std::move(next_inputs);
    }

    XLS_RET_CHECK(IsPowerOfTwo(inputs.size()));
    while (inputs.size() != 1) {
      // Inputs for the next layer in the tree. Will contain half as many
      // elements as 'inputs'.
      std::vector<Node*> next_inputs;
      for (int64 i = 0; i < inputs.size() / 2; ++i) {
        XLS_ASSIGN_OR_RETURN(Node * new_op,
                             new_node(inputs[2 * i], inputs[2 * i + 1]));
        next_inputs.push_back(new_op);
      }
      inputs = std::move(next_inputs);
    }
    XLS_RETURN_IF_ERROR(node->ReplaceUsesWith(inputs[0]).status());
    changed = true;
  }
  return changed;

  XLS_VLOG(3) << "After:";
  XLS_VLOG_LINES(3, f->DumpIr());
}

}  // namespace xls
