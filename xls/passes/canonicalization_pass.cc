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

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/value.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/stateless_query_engine.h"

namespace xls {

namespace {

// Returns the Bits values of 'm' and 'n' if they are both constants whose bits
// values are sequential unsigned values (m + 1 = n).
std::optional<std::pair<Bits, Bits>> AreSequentialConstants(
    Node* m, Node* n, QueryEngine& query_engine) {
  std::optional<Bits> m_bits = query_engine.KnownValueAsBits(m);
  std::optional<Bits> n_bits = query_engine.KnownValueAsBits(n);
  if (!m_bits.has_value() || !n_bits.has_value()) {
    return std::nullopt;
  }
  // Zero extend before adding one to avoid overflow.
  if (bits_ops::UEqual(bits_ops::Increment(bits_ops::ZeroExtend(
                           *m_bits, m_bits->bit_count() + 1)),
                       *n_bits)) {
    return std::make_pair(std::move(*m_bits), std::move(*n_bits));
  }
  return std::nullopt;
}

// Returns the Bits value of 'm' and 'n' if they are both constants whose bits
// values are equal.
std::optional<Bits> AreEqualConstants(Node* m, Node* n,
                                      QueryEngine& query_engine) {
  std::optional<Bits> m_bits = query_engine.KnownValueAsBits(m);
  std::optional<Bits> n_bits = query_engine.KnownValueAsBits(n);
  if (!m_bits.has_value() || !n_bits.has_value() ||
      !bits_ops::UEqual(*m_bits, *n_bits)) {
    return std::nullopt;
  }
  return *m_bits;
}

absl::StatusOr<Literal*> AsLiteralOrMakeLiteral(Node* candidate,
                                                const Bits& bits) {
  if (candidate->Is<Literal>() && candidate->As<Literal>()->value().IsBits() &&
      candidate->As<Literal>()->value().bits() == bits) {
    return candidate->As<Literal>();
  }
  return candidate->function_base()->MakeNode<Literal>(candidate->loc(),
                                                       Value(bits));
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
absl::StatusOr<bool> MaybeCanonicalizeClamp(Node* n,
                                            QueryEngine& query_engine) {
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
  if (!a->GetType()->IsBits() || !b->GetType()->IsBits() ||
      !c->GetType()->IsBits() || !d->GetType()->IsBits()) {
    return false;
  }

  Literal* k = nullptr;
  bool is_clamp_low = false;
  bool is_clamp_high = false;
  std::optional<Bits> b_d_equal = AreEqualConstants(b, d, query_engine);
  std::optional<std::pair<Bits, Bits>> b_c_sequential =
      AreSequentialConstants(b, c, query_engine);
  std::optional<std::pair<Bits, Bits>> b_d_sequential =
      AreSequentialConstants(b, d, query_engine);
  std::optional<std::pair<Bits, Bits>> c_b_sequential =
      AreSequentialConstants(c, b, query_engine);
  std::optional<std::pair<Bits, Bits>> d_b_sequential =
      AreSequentialConstants(d, b, query_engine);
  if (cmp == Op::kUGt && a == d && b_c_sequential.has_value()) {
    //         a cmp b   ? c : d
    //  (i)    x > K - 1 ? K : x   =>   x > K ? K : x
    is_clamp_high = true;
    XLS_ASSIGN_OR_RETURN(k, AsLiteralOrMakeLiteral(c, b_c_sequential->second));
  } else if (cmp == Op::kULt && a == c && b_d_equal.has_value()) {
    //         a cmp b   ? c : d
    //  (ii)   x < K     ? x : K   =>   x > K ? K : x
    is_clamp_high = true;
    XLS_ASSIGN_OR_RETURN(k, AsLiteralOrMakeLiteral(d, *b_d_equal));
  } else if (cmp == Op::kULt && a == c && d_b_sequential.has_value()) {
    //         a cmp b   ? c : d
    //  (iii)  x < K + 1 ? x : K   =>   x > K ? K : x
    is_clamp_high = true;
    XLS_ASSIGN_OR_RETURN(k, AsLiteralOrMakeLiteral(d, d_b_sequential->first));
  } else if (cmp == Op::kULt && a == d && c_b_sequential.has_value()) {
    //         a cmp b   ? c : d
    //  (iv)   x < K + 1 ? K : x   =>   x < K ? K : x
    is_clamp_low = true;
    XLS_ASSIGN_OR_RETURN(k, AsLiteralOrMakeLiteral(c, c_b_sequential->first));
  } else if (cmp == Op::kUGt && a == c && b_d_equal.has_value()) {
    //         a cmp b   ? c : d
    //  (v)    x > K     ? x : K   =>   x < K ? K : x
    is_clamp_low = true;
    XLS_ASSIGN_OR_RETURN(k, AsLiteralOrMakeLiteral(d, *b_d_equal));
  } else if (cmp == Op::kUGt && a == c && b_d_sequential.has_value()) {
    //         a cmp b   ? c : d
    //  (vi)   x > K - 1 ? x : K   =>   x < K ? K : x
    is_clamp_low = true;
    XLS_ASSIGN_OR_RETURN(k, AsLiteralOrMakeLiteral(d, b_d_sequential->second));
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

// CanonicalizeNodes performs simple canonicalization of expressions,
// such as moving a literal in an associative expression to the right.
// Being able to rely on the shape of such nodes greatly simplifies
// the implementation of transformation passes, as only one pattern needs
// to be matched, instead of two.
absl::StatusOr<bool> CanonicalizeNode(Node* n) {
  FunctionBase* f = n->function_base();
  StatelessQueryEngine query_engine;

  // Always move constants to right for commutative operators.
  Op op = n->op();
  if (OpIsCommutative(op) && n->operand_count() == 2) {
    if (query_engine.IsFullyKnown(n->operand(0)) &&
        !query_engine.IsFullyKnown(n->operand(1))) {
      VLOG(2) << "Replaced 'op(constant, x) with op(x, constant)";
      XLS_ASSIGN_OR_RETURN(Node * replacement,
                           n->Clone({n->operand(1), n->operand(0)}));
      XLS_RETURN_IF_ERROR(n->ReplaceUsesWith(replacement));
      return true;
    }
  }

  // Move kLiterals to the right for comparison operators.
  if (OpIsCompare(n->op()) && query_engine.IsFullyKnown(n->operand(0)) &&
      !query_engine.IsFullyKnown(n->operand(1))) {
    XLS_ASSIGN_OR_RETURN(Op commuted_op, ReverseComparisonOp(n->op()));
    VLOG(2) << absl::StreamFormat("Replaced %s(literal, x) with %s(x, literal)",
                                  OpToString(n->op()), OpToString(commuted_op));
    XLS_RETURN_IF_ERROR(n->ReplaceUsesWithNew<CompareOp>(
                             n->operand(1), n->operand(0), commuted_op)
                            .status());
    return true;
  }

  // Canonicalize comparison against literals to the strict form (not "or
  // equals" form). Literal operand should be on the right according to the
  // above canonicalization.
  // TODO(meheff): 2020-01-22 Handle the signed variants.
  if (OpIsCompare(n->op()) && n->operand(1)->GetType()->IsBits() &&
      query_engine.IsFullyKnown(n->operand(1))) {
    Bits constant = *query_engine.KnownValueAsBits(n->operand(1));
    if (n->op() == Op::kUGe && !constant.IsZero()) {
      VLOG(2) << "Replaced Uge(x, K) with Ugt(x, K - 1)";
      Bits k_minus_one = bits_ops::Decrement(constant);
      XLS_ASSIGN_OR_RETURN(
          Literal * new_literal,
          n->function_base()->MakeNode<Literal>(n->loc(), Value(k_minus_one)));
      XLS_RETURN_IF_ERROR(
          n->ReplaceUsesWithNew<CompareOp>(n->operand(0), new_literal, Op::kUGt)
              .status());
      return true;
    }
    if (n->op() == Op::kULe && !constant.IsAllOnes()) {
      VLOG(2) << "Replaced ULe(x, literal) with Ult(x, literal + 1)";
      Bits k_plus_one = bits_ops::Increment(constant);
      XLS_ASSIGN_OR_RETURN(
          Literal * new_literal,
          n->function_base()->MakeNode<Literal>(n->loc(), Value(k_plus_one)));
      XLS_RETURN_IF_ERROR(
          n->ReplaceUsesWithNew<CompareOp>(n->operand(0), new_literal, Op::kULt)
              .status());
      return true;
    }
  }

  XLS_ASSIGN_OR_RETURN(bool min_max_changed,
                       MaybeCanonicalizeClamp(n, query_engine));
  if (min_max_changed) {
    return true;
  }

  // Replace (x - literal) with x + (-literal)
  if (n->op() == Op::kSub && query_engine.IsFullyKnown(n->operand(1))) {
    XLS_ASSIGN_OR_RETURN(
        Node * neg_rhs,
        f->MakeNode<Literal>(
            n->loc(), Value(bits_ops::Negate(
                          *query_engine.KnownValueAsBits(n->operand(1))))));
    XLS_RETURN_IF_ERROR(
        n->ReplaceUsesWithNew<BinOp>(n->operand(0), neg_rhs, Op::kAdd)
            .status());
    VLOG(2) << "Replaced 'sub(lhs, rhs)' with 'add(lhs, -rhs)'";
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

  // Remove useless bitwise reductions of a single bit.
  if (n->Is<BitwiseReductionOp>() && n->operand(0)->BitCountOrDie() == 1) {
    XLS_RETURN_IF_ERROR(n->ReplaceUsesWith(n->operand(0)));
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
  if (IsBinarySelectTwoCases(n) &&
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
    int64_t selector_bit_count = sel->selector()->BitCountOrDie();
    if (selector_bit_count < 63 &&  // don't consider 63+ bit selectors,
                                    // otherwise 1ULL << selector_bit_count will
                                    // overflow (it's too many cases anyways)
        sel->default_value().has_value() &&
        (sel->cases().size() + 1) == (1ULL << selector_bit_count)) {
      std::vector<Node*> new_cases(sel->cases().begin(), sel->cases().end());
      new_cases.push_back(*sel->default_value());
      XLS_RETURN_IF_ERROR(
          n->ReplaceUsesWithNew<Select>(sel->selector(), new_cases,
                                        /*default_value=*/std::nullopt)
              .status());
      return true;
    }
  }

  // Operations with an optional predicate can have their predicate removed if
  // the predicate is always true, or be deleted if the predicate is always
  // false.
  if (n->Is<Next>() && n->As<Next>()->predicate().has_value()) {
    Node* predicate = *n->As<Next>()->predicate();
    if (!predicate->GetType()->IsBits()) {
      return false;
    }
    std::optional<Bits> known_predicate =
        query_engine.KnownValueAsBits(predicate);
    if (known_predicate.has_value() && known_predicate->IsAllOnes()) {
      Next* next = n->As<Next>();
      XLS_RETURN_IF_ERROR(next->RemovePredicate());
      return true;
    } else if (known_predicate.has_value() && known_predicate->IsZero()) {
      XLS_RETURN_IF_ERROR(
          n->ReplaceUsesWithNew<Literal>(Value::Tuple({})).status());
      XLS_RETURN_IF_ERROR(n->function_base()->RemoveNode(n));
      return true;
    }
  }
  if (n->Is<StateRead>() && n->As<StateRead>()->predicate().has_value()) {
    Node* predicate = *n->As<StateRead>()->predicate();
    if (!predicate->GetType()->IsBits()) {
      return false;
    }
    std::optional<Bits> known_predicate =
        query_engine.KnownValueAsBits(predicate);
    if (known_predicate.has_value() && known_predicate->IsAllOnes()) {
      StateRead* state_read = n->As<StateRead>();
      XLS_RETURN_IF_ERROR(state_read->RemovePredicate());
      return true;
    }
    // If the predicate is always false (i.e., the state is never read), it
    // still might end up being augmented by state legalization, so we can't
    // safely remove it here.
  }

  return false;
}

}  // namespace

absl::StatusOr<bool> CanonicalizationPass::RunOnFunctionBaseInternal(
    FunctionBase* func, const OptimizationPassOptions& options,
    PassResults* results, OptimizationContext& context) const {
  return TransformNodesToFixedPoint(func, CanonicalizeNode);
}

}  // namespace xls
