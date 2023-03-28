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

#include "xls/passes/canonicalization_pass.h"

#include "absl/status/statusor.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/node_util.h"

namespace xls {

namespace {

// For the given comparison Op, returns the op op_commuted for which the
// following identity holds:
//   op(x, y) == op_commuted(y, x)
Op CompareOpCommuted(Op op) {
  switch (op) {
    case Op::kEq:
    case Op::kNe:
      return op;
    case Op::kSGe:
      return Op::kSLe;
    case Op::kUGe:
      return Op::kULe;
    case Op::kSGt:
      return Op::kSLt;
    case Op::kUGt:
      return Op::kULt;
    case Op::kSLe:
      return Op::kSGe;
    case Op::kULe:
      return Op::kUGe;
    case Op::kSLt:
      return Op::kSGt;
    case Op::kULt:
      return Op::kUGt;
    default:
      XLS_LOG(FATAL) << "Op is not comparison: " << OpToString(op);
  }
}

// Returns true if 'm' and 'n' are both literals whose bits values are
// sequential unsigned values (m + 1 = n).
bool AreSequentialLiterals(Node* m, Node* n) {
  if (!m->Is<Literal>() || !m->As<Literal>()->value().IsBits() ||
      !n->Is<Literal>() || !n->As<Literal>()->value().IsBits()) {
    return false;
  }
  const Bits& m_bits = m->As<Literal>()->value().bits();
  const Bits& n_bits = n->As<Literal>()->value().bits();
  // Zero extend before adding one to avoid overflow.
  return bits_ops::UEqual(
      bits_ops::Add(bits_ops::ZeroExtend(m_bits, m_bits.bit_count() + 1),
                    UBits(1, m_bits.bit_count() + 1)),
      n_bits);
}

// Returns true if 'm' and 'n' are both literals whose bits values are equal.
bool AreEqualLiterals(Node* m, Node* n) {
  if (!m->Is<Literal>() || !m->As<Literal>()->value().IsBits() ||
      !n->Is<Literal>() || !n->As<Literal>()->value().IsBits()) {
    return false;
  }
  const Bits& m_bits = m->As<Literal>()->value().bits();
  const Bits& n_bits = n->As<Literal>()->value().bits();
  return bits_ops::UEqual(m_bits, n_bits);
}

// Change clamps to high or low values to a canonical form:
//
//  (i)    x > K - 1 ? K : x
//  (ii)   x < K     ? x : K      =>   x > K ? K : x
//  (iii)  x < K + 1 ? x : K
//
//  (iv)   x < K + 1 ? K : x
//  (v)    x > K     ? x : K      =>   x < K ? K : x
//  (vi)   x > K - 1 ? x : K
//
// We only have to consider forms where the literal in the comparison is on the
// rhs and strict comparison operations because of other canonicalizations.
absl::StatusOr<bool> MaybeCanonicalizeClamp(Node* n) {
  if (!n->GetType()->IsBits() || !n->Is<Select>()) {
    return false;
  }
  Select* select = n->As<Select>();
  if (select->cases().size() != 2 || !OpIsCompare(select->selector()->op())) {
    return false;
  }

  // Assign a, b, c, and d matching the following pattern.
  //
  //   a cmp b ? c : d
  Op cmp = select->selector()->op();
  Node* a = select->selector()->operand(0);
  Node* b = select->selector()->operand(1);
  Node* c = select->get_case(1);
  Node* d = select->get_case(0);

  Literal* k = nullptr;
  bool is_clamp_low = false;
  bool is_clamp_high = false;
  if (cmp == Op::kUGt && a == d && AreSequentialLiterals(b, c)) {
    //         a cmp b   ? c : d
    //  (i)    x > K - 1 ? K : x   =>   x > K ? K : x
    is_clamp_high = true;
    k = c->As<Literal>();
  } else if (cmp == Op::kULt && a == c && AreEqualLiterals(b, d)) {
    //         a cmp b   ? c : d
    //  (ii)   x < K     ? x : K   =>   x > K ? K : x
    is_clamp_high = true;
    k = d->As<Literal>();
  } else if (cmp == Op::kULt && a == c && AreSequentialLiterals(d, b)) {
    //         a cmp b   ? c : d
    //  (iii)  x < K + 1 ? x : K   =>   x > K ? K : x
    is_clamp_high = true;
    k = d->As<Literal>();
  } else if (cmp == Op::kULt && a == d && AreSequentialLiterals(c, b)) {
    //         a cmp b   ? c : d
    //  (iv)   x < K + 1 ? K : x   =>   x < K ? K : x
    is_clamp_low = true;
    k = c->As<Literal>();
  } else if (cmp == Op::kUGt && a == c && AreEqualLiterals(b, d)) {
    //         a cmp b   ? c : d
    //  (v)    x > K     ? x : K   =>   x < K ? K : x
    is_clamp_low = true;
    k = d->As<Literal>();
  } else if (cmp == Op::kUGt && a == c && AreSequentialLiterals(b, d)) {
    //         a cmp b   ? c : d
    //  (vi)   x > K - 1 ? x : K   =>   x < K ? K : x
    is_clamp_low = true;
    k = d->As<Literal>();
  }
  if (is_clamp_high || is_clamp_low) {
    // Create an expression:
    //
    //   Sel(UGt(x, k), cases=[x, k])  // is_clamp_high
    //   Sel(ULt(x, k), cases=[x, k])  // is_clamp_low
    //
    // Node 'a' is 'x' in the comments.
    XLS_ASSIGN_OR_RETURN(
        Node * cmp, n->function_base()->MakeNode<CompareOp>(
                        n->loc(), a, k, is_clamp_high ? Op::kUGt : Op::kULt));
    XLS_RETURN_IF_ERROR(
        n->ReplaceUsesWithNew<Select>(cmp, /*cases=*/std::vector<Node*>({a, k}),
                                      /*default_value=*/std::nullopt)
            .status());
    return true;
  }

  return false;
}

}  // namespace

// CanonicalizeNodes performs simple canonicalization of expressions,
// such as moving a literal in an associative expression to the right.
// Being able to rely on the shape of such nodes greatly simplifies
// the implementation of transformation passes, as only one pattern needs
// to be matched, instead of two.
static absl::StatusOr<bool> CanonicalizeNode(Node* n) {
  FunctionBase* f = n->function_base();

  // Always move kLiteral to right for commutative operators.
  Op op = n->op();
  if (OpIsCommutative(op) && n->operand_count() == 2) {
    if (n->operand(0)->Is<Literal>() && !n->operand(1)->Is<Literal>()) {
      XLS_VLOG(2) << "Replaced 'op(literal, x) with op(x, literal)";
      XLS_ASSIGN_OR_RETURN(Node * replacement,
                           n->Clone({n->operand(1), n->operand(0)}));
      XLS_RETURN_IF_ERROR(n->ReplaceUsesWith(replacement));
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

  // Canonicalize comparison against literals to the strict form (not "or
  // equals" form). Literal operand should be on the right according to the
  // above canonicalization.
  // TODO(meheff): 2020-01-22 Handle the signed variants.
  if (OpIsCompare(n->op()) && n->operand(1)->Is<Literal>() &&
      n->operand(1)->GetType()->IsBits()) {
    const Bits& literal = n->operand(1)->As<Literal>()->value().bits();
    if (n->op() == Op::kUGe && !literal.IsZero()) {
      XLS_VLOG(2) << "Replaced Uge(x, K) with Ugt(x, K - 1)";
      Bits k_minus_one = bits_ops::Sub(literal, UBits(1, literal.bit_count()));
      XLS_ASSIGN_OR_RETURN(
          Literal * new_literal,
          n->function_base()->MakeNode<Literal>(n->loc(), Value(k_minus_one)));
      XLS_RETURN_IF_ERROR(
          n->ReplaceUsesWithNew<CompareOp>(n->operand(0), new_literal, Op::kUGt)
              .status());
      return true;
    }
    if (n->op() == Op::kULe && !literal.IsAllOnes()) {
      XLS_VLOG(2) << "Replaced ULe(x, literal) with Ult(x, literal + 1)";
      Bits k_plus_one = bits_ops::Add(literal, UBits(1, literal.bit_count()));
      XLS_ASSIGN_OR_RETURN(
          Literal * new_literal,
          n->function_base()->MakeNode<Literal>(n->loc(), Value(k_plus_one)));
      XLS_RETURN_IF_ERROR(
          n->ReplaceUsesWithNew<CompareOp>(n->operand(0), new_literal, Op::kULt)
              .status());
      return true;
    }
  }

  XLS_ASSIGN_OR_RETURN(bool min_max_changed, MaybeCanonicalizeClamp(n));
  if (min_max_changed) {
    return true;
  }

  // Replace (x - literal) with x + (-literal)
  if (n->op() == Op::kSub && n->operand(1)->Is<Literal>()) {
    XLS_ASSIGN_OR_RETURN(
        Node * neg_rhs,
        f->MakeNode<Literal>(
            n->loc(), Value(bits_ops::Negate(
                          n->operand(1)->As<Literal>()->value().bits()))));
    XLS_RETURN_IF_ERROR(
        n->ReplaceUsesWithNew<BinOp>(n->operand(0), neg_rhs, Op::kAdd)
            .status());
    XLS_VLOG(2) << "Replaced 'sub(lhs, rhs)' with 'add(lhs, -rhs)'";
    return true;
  }

  if (n->Is<ExtendOp>() &&
      n->BitCountOrDie() == n->operand(0)->BitCountOrDie()) {
    XLS_RETURN_IF_ERROR(n->ReplaceUsesWith(n->operand(0)));
    return true;
  }

  // Replace zero-extend operations with concat with zero.
  if (n->op() == Op::kZeroExt) {
    const int64_t operand_bit_count = n->operand(0)->BitCountOrDie();
    const int64_t bit_count = n->BitCountOrDie();
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

  // For a two-way select with an inverted selector, the NOT can be
  // removed and the cases interchanged.
  //
  //   p = ...
  //   not_p = not(p)
  //   sel = sel(not_p, cases=[a, b])
  //
  // becomes:
  //
  //   sel = sel(p, case=[b, a])
  if (IsBinarySelect(n) &&
      n->As<Select>()->selector()->op() == Op::kNot) {
    Select* sel = n->As<Select>();
    XLS_RETURN_IF_ERROR(
        sel->ReplaceUsesWithNew<Select>(
               sel->selector()->operand(0),
               /*cases=*/std::vector<Node*>{sel->get_case(1), sel->get_case(0)},
               /*default_value=*/std::nullopt)
            .status());
    return true;
  }

  // Replace a select with a default which only handles a single value of the
  // selector with a select without a default.
  if (n->Is<Select>()) {
    Select* sel = n->As<Select>();
    if (sel->default_value().has_value() &&
        (sel->cases().size() + 1) ==
            (1ULL << sel->selector()->BitCountOrDie())) {
      std::vector<Node*> new_cases(sel->cases().begin(), sel->cases().end());
      new_cases.push_back(*sel->default_value());
      XLS_RETURN_IF_ERROR(
          n->ReplaceUsesWithNew<Select>(sel->selector(), new_cases,
                                           /*default_value=*/std::nullopt)
              .status());
      return true;
    }
  }

  return false;
}

absl::StatusOr<bool> CanonicalizationPass::RunOnFunctionBaseInternal(
    FunctionBase* func, const PassOptions& options,
    PassResults* results) const {
  return TransformNodesToFixedPoint(func, CanonicalizeNode);
}

}  // namespace xls
