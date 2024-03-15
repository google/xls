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

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/logging/logging.h"
#include "xls/common/math_util.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/node_util.h"
#include "xls/ir/op.h"
#include "xls/ir/topo_sort.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"

namespace xls {

namespace {

// Returns true if the node and its operands are all the same type.
bool NodeAndOperandsSameType(Node* n) {
  return std::all_of(n->operands().begin(), n->operands().end(),
                     [n](Node* o) { return o->GetType() == n->GetType(); });
}

// Returns true if the node is an addition, the operands of the node are all the
// same type of extension, and all are extended by at least one bit.
bool IsFullWidthAddition(Node* n) {
  if (n->op() != Op::kAdd) {
    return false;
  }
  Op first_operand_op = n->operand(0)->op();
  if (first_operand_op != Op::kZeroExt && first_operand_op != Op::kSignExt) {
    return false;
  }
  return std::all_of(n->operands().begin(), n->operands().end(),
                     [first_operand_op](Node* o) {
                       return o->op() == first_operand_op &&
                              o->GetType()->GetFlatBitCount() >
                                  o->operand(0)->GetType()->GetFlatBitCount();
                     });
}

// Walks an expression tree of adds and substracts and gathers the leafs of the
// expression. Each leaf includes a boolean value indicating whether the leaf id
// negated (as in the right side of a subtract). For example, if called at the
// root of the following expression:
//
//         c     d
//          \   /
//   a    b  sub    e
//    \  /     \   /
//    add       add
//       \     /
//         sub
//
// Upon return leaves might contain the following:
//
//   leaves = {{false, a}, {false, b}, {true, c}, {false, d}, {true, e}}
//
// The expression is equivalently: a + b + (-c) + d + (-e)
struct AddSubLeaf {
  bool negated;
  Node* node;
};
absl::Status GatherAddsAndSubtracts(Node* node, bool negated,
                                    std::vector<AddSubLeaf>* leaves,
                                    std::vector<Node*>* interior_nodes) {
  if ((node->op() != Op::kAdd && node->op() != Op::kSub) ||
      !NodeAndOperandsSameType(node)) {
    // 'node' is not an add or subtract or has differently typed operads.
    leaves->push_back(AddSubLeaf{negated, node});
    return absl::OkStatus();
  }

  interior_nodes->push_back(node);

  XLS_RET_CHECK_EQ(node->operand_count(), 2);
  for (int64_t operand_no = 0; operand_no < 2; ++operand_no) {
    Node* operand = node->operand(operand_no);
    // Subtraction negates it's second operand (operand number 1).
    bool negated_next =
        (operand_no == 1 && node->op() == Op::kSub) ? !negated : negated;
    XLS_RETURN_IF_ERROR(
        GatherAddsAndSubtracts(operand, negated_next, leaves, interior_nodes));
  }
  return absl::OkStatus();
}

// Returns an expression equal to the sum of the given nodes. The expression is
// a right-leaning tree of adds. That is, given:
//
//   {a, b, c, d}
//
// Will return:
//
//   (a + (b + (c + d)))
//
// TODO(meheff): 2021-01-27 Use n-ary adds when they are supported.
absl::StatusOr<Node*> CreateSum(absl::Span<Node* const> nodes) {
  XLS_RET_CHECK(!nodes.empty());
  if (nodes.size() == 1) {
    return nodes.front();
  }
  Node* lhs = nodes.front();
  XLS_ASSIGN_OR_RETURN(Node * rhs, CreateSum(nodes.subspan(1)));
  return lhs->function_base()->MakeNode<BinOp>(lhs->loc(), lhs, rhs, Op::kAdd);
}

// Attempts to simplify expressions containing adds and subtracts using
// reassociation. The optimizations transforms an arbitrary expression of adds
// and subtracts into a single subtract operation where the LHS and RHS are
// sums. Literals always appear in terms on the LHS because they can be negated
// at compile time. For example, the following:
//
//   ((a + b) - C0) - ((c + (C1 - d)) + e)
//
// Might be transformed into:
//
//  (a + b + -C0 + -C1 + d) - (c + e)
//
// The advantage of this optimization is that add expressions of maximal size
// are created enabling beneficial reassociation for delay minimization when
// Reassociate() is run. Also, literals are grouped together across add and
// subtract expressions.
absl::StatusOr<bool> ReassociateSubtracts(FunctionBase* f) {
  XLS_VLOG(4) << "Reassociating subtracts";
  bool changed = false;
  // Keep track of which nodes we've already considered for reassociation so we
  // don't revisit subexpressions multiple times.
  absl::flat_hash_set<Node*> visited_nodes;

  // Traverse the nodes in reverse order because we construct expressions for
  // reassociation starting from the roots.
  for (Node* node : ReverseTopoSort(f)) {
    XLS_VLOG(4) << "Considering node: " << node->GetName();

    if (visited_nodes.contains(node)) {
      XLS_VLOG(4) << "  Already visited.";
      continue;
    }

    if (node->op() != Op::kAdd && node->op() != Op::kSub) {
      XLS_VLOG(4) << "  Is not add or subtract.";
      continue;
    }

    std::vector<AddSubLeaf> leaves;
    std::vector<Node*> interior_nodes;
    XLS_RETURN_IF_ERROR(GatherAddsAndSubtracts(node, /*negated=*/false, &leaves,
                                               &interior_nodes));

    // Count the number of subtraction operations in the tree, and mark any
    // interior nodes as visited so they are not traverse in later iterations.
    int64_t subtract_count = 0;
    for (Node* interior_node : interior_nodes) {
      visited_nodes.insert(interior_node);
      if (interior_node->op() == Op::kSub) {
        ++subtract_count;
      }
    }

    if (XLS_VLOG_IS_ON(4)) {
      std::vector<std::string> terms;
      for (const AddSubLeaf& leaf : leaves) {
        if (leaf.negated) {
          terms.push_back(absl::StrCat("-", leaf.node->GetName()));
        } else {
          terms.push_back(leaf.node->GetName());
        }
      }
      XLS_VLOG(4) << "Found expression: " << absl::StrJoin(terms, " + ");
      XLS_VLOG(4) << "subtract count: " << subtract_count;
    }

    if (subtract_count == 0) {
      XLS_VLOG(4) << "No subtracts, nothing to do.";
      continue;
    }

    bool negated_constant = false;
    std::vector<Node*> nonnegated_nodes;
    std::vector<Node*> negated_nodes;
    for (const AddSubLeaf& leaf : leaves) {
      if (leaf.negated) {
        if (leaf.node->Is<Literal>()) {
          // Negated literal term. Negate the literal making it a nonnegated
          // term.
          const Bits& value = leaf.node->As<Literal>()->value().bits();
          XLS_ASSIGN_OR_RETURN(
              Node * new_literal,
              f->MakeNode<Literal>(leaf.node->loc(),
                                   Value(bits_ops::Negate(value))));
          nonnegated_nodes.push_back(new_literal);
          negated_constant = true;
        } else {
          // Negated non-literal term.
          negated_nodes.push_back(leaf.node);
        }
      } else {
        // Non-negated node.
        nonnegated_nodes.push_back(leaf.node);
      }
    }

    if (subtract_count == 1 && !negated_constant) {
      // The expression tree had a single subtract and no literal was found that
      // could be negated. Nothing to do here as we would transform into an
      // expression with a single subtraction anyway.
      XLS_VLOG(4) << "Only a single subtract and no negated literals found,"
                     "continuing.";
      continue;
    }

    Node* replacement;
    if (negated_nodes.empty()) {
      // There are no negated nodes in the expression. This can only occur if
      // some of the originally negated terms were literals that were negated in
      // the above loop.
      XLS_RET_CHECK(negated_constant);
      XLS_ASSIGN_OR_RETURN(replacement, CreateSum(nonnegated_nodes));
      XLS_VLOG(4) << "All nodes non-negated. Replacing with chain of adds.";
    } else {
      // Create a subtraction with the LHS being the sum of 'nonnegated_nodes'
      // and the RHS a sum of 'negated_nodes'.
      XLS_RET_CHECK(!nonnegated_nodes.empty());
      XLS_ASSIGN_OR_RETURN(Node * lhs, CreateSum(nonnegated_nodes));
      XLS_ASSIGN_OR_RETURN(Node * rhs, CreateSum(negated_nodes));
      XLS_ASSIGN_OR_RETURN(replacement,
                           f->MakeNode<BinOp>(node->loc(), lhs, rhs, Op::kSub));
      XLS_VLOG(4) << "Expression includes negated and non-negated terms. "
                     "Replacing with single subtraction.";
    }
    XLS_RETURN_IF_ERROR(node->ReplaceUsesWith(replacement));

    changed = true;
  }

  XLS_VLOG(4) << "Finished reassociating  subtracts, changed = " << changed;

  return changed;
}

// Walks an expression tree of full-width additions using the given
// `extension_op`. 'node' is the node currently being visited.  The leaves of
// the expression tree are added to 'leaves', and the interior nodes of the tree
// are added to 'interior_nodes'. Returns the height of the tree.
//
// For example, if called at the root of the following expression:
//
//     a           b           c           d
//     |           |           |           |
// zero_ext.6  zero_ext.7  zero_ext.8  zero_ext.9
//        \     /                 \     /
//         add.4                   add.5
//           |                       |
//       zero_ext.2              zero_ext.3
//              \                 /
//               \               /
//                \             /
//                 \           /
//                  \         /
//                   \       /
//                    \     /
//                     add.1
//
// Upon return the passed in vectors would be:
//
//   leaves = {a, b, c, d}
//   interior_nodes = {add.1, add.2, add.3}
//
// And the function would return 2, the depth of the tree (not counting the
// leaves or the extensions).
int64_t GatherFullWidthAdditionLeaves(Op extension_op, Node* node,
                                      std::vector<Node*>* leaves,
                                      std::vector<Node*>* interior_nodes) {
  if (!IsFullWidthAddition(node) || node->operand(0)->op() != extension_op) {
    // 'node' is not a full-width addition of the same type and is therefore a
    // leaf.
    leaves->push_back(node);
    return 0;
  }
  // 'node' could be an interior node in the tree of full-width additions.
  // Traverse into the operands if the operand & its extension are both
  // single-use; otherwise the operand is a leaf.
  interior_nodes->push_back(node);
  int64_t max_depth = 1;
  for (Node* extension : node->operands()) {
    Node* operand = extension->operand(0);
    // TODO(meheff): 2021-01-27 Consider handling cases with more than one user.
    if (HasSingleUse(extension) && HasSingleUse(operand)) {
      max_depth = std::max(
          max_depth, 1 + GatherFullWidthAdditionLeaves(extension_op, operand,
                                                       leaves, interior_nodes));
    } else {
      leaves->push_back(operand);
    }
  }
  return max_depth;
}

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
int64_t GatherExpressionLeaves(Op op, Node* node, std::vector<Node*>* leaves,
                               std::vector<Node*>* interior_nodes) {
  if (node->op() != op || !NodeAndOperandsSameType(node)) {
    // 'node' does not match the other nodes of the tree and is thus a leaf.
    leaves->push_back(node);
    return 0;
  }
  // 'node' is an interior node in the tree of identical operations. Traverse
  // into the operands if the operand has a single use, otherwise the operand is
  // a leaf.
  interior_nodes->push_back(node);
  int64_t max_depth = 1;
  for (Node* operand : node->operands()) {
    // TODO(meheff): 2021-01-27 Consider handling cases with more than one user.
    if (HasSingleUse(operand)) {
      max_depth = std::max(
          max_depth,
          1 + GatherExpressionLeaves(op, operand, leaves, interior_nodes));
    } else {
      leaves->push_back(operand);
    }
  }
  return max_depth;
}

// Reassociate associative and commutative operations to minimize delay and
// maximize opportunity for constant folding.
absl::StatusOr<bool> Reassociate(FunctionBase* f) {
  bool changed = false;
  // Keep track of which nodes we've already considered for reassociation so we
  // don't revisit subexpressions multiple times.
  absl::flat_hash_set<Node*> visited_nodes;

  // Traverse the nodes in reverse order because we construct expressions for
  // reassociation starting from the roots.
  for (Node* node : ReverseTopoSort(f)) {
    auto [it, inserted] = visited_nodes.insert(node);
    if (!inserted) {
      // Already visited; skip this node.
      continue;
    }

    // Only reassociate arithmetic operations. Logical operations can
    // theoretically be reassociated, but they are better handled by collapsing
    // into a single operation as logical operations are n-ary.
    if (node->op() != Op::kAdd && node->op() != Op::kUMul &&
        node->op() != Op::kSMul) {
      continue;
    }
    const bool is_full_width_addition = IsFullWidthAddition(node);

    std::vector<Node*> leaves;
    std::vector<Node*> interior_nodes;
    int64_t expression_depth;
    if (is_full_width_addition) {
      // Full-width addition; we need to treat the addition & operand extensions
      // as a single operation.
      expression_depth = GatherFullWidthAdditionLeaves(
          node->operand(0)->op(), node, &leaves, &interior_nodes);

    } else {
      expression_depth =
          GatherExpressionLeaves(node->op(), node, &leaves, &interior_nodes);
    }

    // Interior nodes in the expression will be reassociated so add them
    // to the set of associated nodes. This keeps us from revisiting the same
    // nodes later in the same pass.
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

    XLS_VLOG(4) << "Reassociated expression rooted at: " << node->GetName();
    XLS_VLOG(4) << "  is_full_width_addition: " << is_full_width_addition;

    auto node_joiner = [](std::string* s, Node* node) {
      absl::StrAppendFormat(s, "%s (%s)", node->GetName(),
                            node->GetType()->ToString());
    };
    XLS_VLOG(4) << "  operations to reassociate:  "
                << absl::StrJoin(interior_nodes, ", ", node_joiner);
    XLS_VLOG(4) << "  leaves:  " << absl::StrJoin(leaves, ", ", node_joiner);
    XLS_VLOG(4) << "  literals leaves:  "
                << absl::StrJoin(literals, ", ", node_joiner);

    auto new_node = [&](Node* lhs, Node* rhs) -> absl::StatusOr<Node*> {
      if (is_full_width_addition) {
        // Widths may vary between the tree & the original, so we need to
        // construct new nodes rather than clones. We know that the final result
        // is big enough for all values (since is_full_width_addition ensures no
        // overflow) so we stop bit-extension there. This also prevents issues
        // where the number of bits depends on the number of inputs and could
        // end up as more than the expected number of bits.
        int64_t addition_width =
            std::min(std::max(lhs->BitCountOrDie(), rhs->BitCountOrDie()) + 1,
                     node->BitCountOrDie());
        XLS_ASSIGN_OR_RETURN(
            Node * extended_lhs,
            node->function_base()->MakeNode<ExtendOp>(
                node->loc(), lhs, /*new_bit_count=*/addition_width,
                node->operand(0)->op()));
        XLS_ASSIGN_OR_RETURN(
            Node * extended_rhs,
            node->function_base()->MakeNode<ExtendOp>(
                node->loc(), rhs, /*new_bit_count=*/addition_width,
                node->operand(0)->op()));
        XLS_VLOG(4) << absl::StreamFormat(
            "Creating new add of type %s: %s (%s) + %s (%s) ",
            extended_lhs->GetType()->ToString(), lhs->GetName(),
            lhs->GetType()->ToString(), rhs->GetName(),
            rhs->GetType()->ToString());
        return node->function_base()->MakeNode<BinOp>(node->loc(), extended_lhs,
                                                      extended_rhs, Op::kAdd);
      }

      // Create a clone of 'node' for constructing a reassociated expression.
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
      for (int64_t i = 2; i < literals.size(); ++i) {
        XLS_ASSIGN_OR_RETURN(literal_expr, new_node(literals[i], literal_expr));
      }
      inputs.push_back(literal_expr);
    }

    XLS_VLOG(4) << "  inputs before balancing:  "
                << absl::StrJoin(inputs, ", ", node_joiner);

    // Reassociate the expressions into a balanced tree. First, reduce the
    // number of inputs to a power of two. Then build a balanced tree.
    if (!IsPowerOfTwo(inputs.size())) {
      // Number of operations to apply to the inputs to reduce operand count to
      // a power of two. These ops will be the ragged top layer of the
      // expression tree.
      int64_t op_count = inputs.size() - (1ULL << FloorOfLog2(inputs.size()));
      std::vector<Node*> next_inputs;
      for (int64_t i = 0; i < op_count; ++i) {
        XLS_ASSIGN_OR_RETURN(Node * new_op,
                             new_node(inputs[2 * i], inputs[2 * i + 1]));
        next_inputs.push_back(new_op);
      }
      for (int64_t i = op_count * 2; i < inputs.size(); ++i) {
        next_inputs.push_back(inputs[i]);
      }
      inputs = std::move(next_inputs);
    }

    XLS_VLOG(4) << "  inputs after balancing:  "
                << absl::StrJoin(inputs, ", ", node_joiner);

    XLS_RET_CHECK(IsPowerOfTwo(inputs.size()));
    while (inputs.size() != 1) {
      // Inputs for the next layer in the tree. Will contain half as many
      // elements as 'inputs'.
      std::vector<Node*> next_inputs;
      for (int64_t i = 0; i < inputs.size() / 2; ++i) {
        XLS_ASSIGN_OR_RETURN(Node * new_op,
                             new_node(inputs[2 * i], inputs[2 * i + 1]));
        next_inputs.push_back(new_op);
      }
      inputs = std::move(next_inputs);
    }
    Node* new_root = inputs[0];
    if (is_full_width_addition &&
        new_root->BitCountOrDie() < node->BitCountOrDie()) {
      XLS_ASSIGN_OR_RETURN(
          new_root,
          node->function_base()->MakeNode<ExtendOp>(
              node->loc(), new_root, /*new_bit_count=*/node->BitCountOrDie(),
              node->operand(0)->op()));
    }
    XLS_RETURN_IF_ERROR(node->ReplaceUsesWith(new_root))
        << "Root: " << node->ToStringWithOperandTypes() << " replaced with "
        << new_root->ToStringWithOperandTypes();
    changed = true;
  }

  return changed;
}

}  // namespace

absl::StatusOr<bool> ReassociationPass::RunOnFunctionBaseInternal(
    FunctionBase* f, const OptimizationPassOptions& options,
    PassResults* results) const {
  XLS_ASSIGN_OR_RETURN(bool reassoc_subtracts_changed, ReassociateSubtracts(f));
  XLS_ASSIGN_OR_RETURN(bool reassoc_changed, Reassociate(f));
  return reassoc_subtracts_changed || reassoc_changed;
}

REGISTER_OPT_PASS(ReassociationPass);

}  // namespace xls
