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

#include "xls/passes/canonicalization_pass.h"

#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/status/statusor.h"
#include "xls/ir/node_iterator.h"

namespace xls {

namespace {

// For the given comparison Op, returns the op op_commuted for which the
// following identity holds:
//   op(x, y) == op_commuted(y, x)
Op CompareOpCommuted(Op op) {
  switch (op) {
    case OP_EQ:
    case OP_NE:
      return op;
    case OP_SGE:
      return OP_SLE;
    case OP_UGE:
      return OP_ULE;
    case OP_SGT:
      return OP_SLT;
    case OP_UGT:
      return OP_ULT;
    case OP_SLE:
      return OP_SGE;
    case OP_ULE:
      return OP_UGE;
    case OP_SLT:
      return OP_SGT;
    case OP_ULT:
      return OP_UGT;
    default:
      XLS_LOG(FATAL) << "Op is not comparison: " << OpToString(op);
  }
}

}  // namespace

// CanonicalizeNodes performs simple canonicalization of expressions,
// such as moving a literal in an associative expression to the right.
// Being able to rely on the shape of such nodes greatly simplifies
// the implementation of transformation passes, as only one pattern needs
// to be matched, instead of two.
xabsl::StatusOr<bool> CanonicalizeNodes(Node* n, Function* f) {
  // Always move kLiteral to right for commutative operators.
  Op op = n->op();
  if (OpIsCommutative(op) && n->operand_count() == 2) {
    if (n->operand(0)->Is<Literal>() && !n->operand(1)->Is<Literal>()) {
      XLS_VLOG(2) << "Replaced 'op(literal, x) with op(x, literal)";
      n->SwapOperands(0, 1);
      return true;
    }
  }

  // Move kLiterals to the right for comparison operators.
  if (OpIsCompare(n->op()) && n->operand(0)->Is<Literal>() &&
      !n->operand(1)->Is<Literal>()) {
    Op commuted_op = CompareOpCommuted(n->op());
    XLS_VLOG(2) << absl::StreamFormat(
        "Replaced %s(literal, x) with %s(x, literal)", OpToString(n->op()),
        OpToString(commuted_op));
    XLS_RETURN_IF_ERROR(n->ReplaceUsesWithNew<CompareOp>(
                             n->operand(1), n->operand(0), commuted_op)
                            .status());
    return true;
  }

  // Replace (x - lit) with x + (-literal)
  if (n->op() == OP_SUB && n->operand(1)->Is<Literal>()) {
    XLS_ASSIGN_OR_RETURN(Node * neg_rhs,
                         f->MakeNode<UnOp>(n->loc(), n->operand(1), OP_NEG));
    XLS_RETURN_IF_ERROR(
        n->ReplaceUsesWithNew<BinOp>(n->operand(0), neg_rhs, OP_ADD)
            .status());
    XLS_VLOG(2) << "Replaced 'sub(lhs, rhs)' with 'add(lhs, neg(rhs))'";
    return true;
  }

  if (n->Is<ExtendOp>() &&
      n->BitCountOrDie() == n->operand(0)->BitCountOrDie()) {
    return n->ReplaceUsesWith(n->operand(0));
  }

  // Replace zero-extend operations with concat with zero.
  if (n->op() == OP_ZERO_EXT) {
    const int64 operand_bit_count = n->operand(0)->BitCountOrDie();
    const int64 bit_count = n->BitCountOrDie();
    // The optimization above should have caught any degenerate non-extending
    // zero-extend ops.
    XLS_RET_CHECK_GT(bit_count, operand_bit_count);
    XLS_ASSIGN_OR_RETURN(
        Node * zero,
        f->MakeNode<Literal>(n->loc(),
                             Value(UBits(0, bit_count - operand_bit_count))));
    std::vector<Node*> concat_operands = {zero, n->operand(0)};
    XLS_RETURN_IF_ERROR(
        n->ReplaceUsesWithNew<Concat>(concat_operands).status());
    return true;
  }

  return false;
}

xabsl::StatusOr<bool> CanonicalizationPass::RunOnFunction(
    Function* func, const PassOptions& options, PassResults* results) const {
  bool changed = false;
  for (Node* node : TopoSort(func)) {
    XLS_ASSIGN_OR_RETURN(bool node_changed, CanonicalizeNodes(node, func));
    changed = changed | node_changed;
  }
  return changed;
}

}  // namespace xls
