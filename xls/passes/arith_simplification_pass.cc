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

#include "xls/passes/arith_simplification_pass.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/inline_bitmap.h"
#include "xls/ir/big_int.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"

namespace xls {
namespace {

// For the given comparison Op, returns the op op_inverse for which the
// following identity holds:
//   op(x, y) == !op_inverse(x, y)
Op CompareOpInverse(Op op) {
  switch (op) {
    case Op::kEq:
      return Op::kNe;
    case Op::kNe:
      return Op::kEq;
    case Op::kSGe:
      return Op::kSLt;
    case Op::kUGe:
      return Op::kULt;
    case Op::kSGt:
      return Op::kSLe;
    case Op::kUGt:
      return Op::kULe;
    case Op::kSLe:
      return Op::kSGt;
    case Op::kULe:
      return Op::kUGt;
    case Op::kSLt:
      return Op::kSGe;
    case Op::kULt:
      return Op::kUGe;
    default:
      XLS_LOG(FATAL) << "Op is not comparison: " << OpToString(op);
  }
}

// Returns true if `node` is a comparison between bits-typed values.
bool IsBitsCompare(Node* node) {
  return OpIsCompare(node->op()) && node->operand(0)->GetType()->IsBits();
}

// Matches the given node to the following expressions:
//
//  Select(UGt(x, LIMIT), cases=[x, LIMIT])
//  Select(UGt(x, LIMIT), cases=[x], default=LIMIT)
//
// Where LIMIT is a literal. Returns a ClampExpr containing 'x' and 'LIMIT'
// values.
struct ClampExpr {
  Node* node;
  Bits upper_limit;
};
std::optional<ClampExpr> MatchClampUpperLimit(Node* n) {
  if (!n->Is<Select>()) {
    return std::nullopt;
  }
  Select* select = n->As<Select>();
  Node* cmp = select->selector();
  int total_cases =
      select->cases().size() + (select->default_value().has_value() ? 1 : 0);
  if (total_cases != 2) {
    return std::nullopt;
  }
  Node* consequent = select->get_case(0);
  Node* alternative;
  if (select->default_value().has_value()) {
    alternative = *select->default_value();
  } else {
    alternative = select->get_case(1);
  }
  if (select->selector()->op() == Op::kUGt && alternative->Is<Literal>() &&
      cmp->operand(1) == alternative && cmp->operand(0) == consequent) {
    return ClampExpr{cmp->operand(0),
                     alternative->As<Literal>()->value().bits()};
  }
  return std::nullopt;
}

// The values required to emulate division by a specific divisor.
struct UnsignedDivisionConstants {
  // dividend and divisor must be < 2^N
  const int64_t N;

  // multplies the dividend
  const BigInt m;

  // right shift amount before the multiply
  const int64_t pre_shift;

  // right shift amount after the multiply
  const int64_t post_shift;
};

// Returns (m, post_shift, l). Used in optimizing division by
// a constant.
//
// Args:
//   N: the smallest value that satisfies: (divisor < 2^N) and (numerator < 2^N)
//   divisor: the divisor, a compile-time constant.
//   precision: bits of precision needed. Frequently this is N, and
//    sometimes an optimization makes this smaller.
//
// With respect to the paper: prec = precision, sh_post = post_shift
std::tuple<BigInt, int64_t, int64_t> ChooseMultiplier(int64_t N, BigInt divisor,
                                                      int64_t precision) {
  const int64_t l = BigInt::CeilingLog2(divisor);
  int64_t post_shift = l;

  // Possible optimization: the expression below can also be computed as:
  //      ((2^l - divisor) * 2^N)
  // floor(---------------------) + 2^N
  //      (        divisor      )
  BigInt m_low = BigInt::Exp2(N + l) / divisor;

  // Possible optimization: the expression below can also be computed as:
  //      (2^(N+l-p) + 2^N * (2^l - divisor))
  // floor(---------------------------------) + 2^N
  //      (        divisor                  )

  const BigInt two_n_l = BigInt::Exp2(N + l);
  const BigInt two_n_l_p = BigInt::Exp2(N + l - precision);
  BigInt m_high = (two_n_l + two_n_l_p) / divisor;

  const BigInt two = BigInt::Exp2(1);

  while ((m_low / two) < (m_high / two) && (post_shift > 0)) {
    m_low = m_low / two;
    m_high = m_high / two;
    post_shift = post_shift - 1;
  }

  return std::make_tuple(m_high, post_shift, l);
}

UnsignedDivisionConstants ComputeUnsignedDivisionConstants(int64_t num_bits,
                                                           BigInt divisor) {
  CHECK(!BigInt::IsPowerOfTwo(divisor))
      << "divide by power of two isn't handled by UnsignedDivision; other code "
         "handles that case.";

  auto [m, post_shift, ignored] = ChooseMultiplier(num_bits, divisor, num_bits);

  int64_t pre_shift;
  if (m >= BigInt::Exp2(num_bits) && BigInt::IsEven(divisor)) {
    auto [divisor_odd_factor, exponent] = BigInt::FactorizePowerOfTwo(divisor);
    pre_shift = exponent;
    std::tie(m, post_shift, ignored) =
        ChooseMultiplier(num_bits, divisor_odd_factor, num_bits - exponent);
  } else {
    pre_shift = 0;
  }

  return UnsignedDivisionConstants{num_bits, m, pre_shift, post_shift};
}

// Narrows or extends n.
//
// Extends when resize_to > n.BitCount. If n_is_signed, sign extension is
// performed, otherwise zero extension.
//
// Narrows (discards most significant bits) when resize_to < n.BitCount.
//
// Is identity function when resize_to == n.BitCount.
//
// Assumes resize_to >= 0.
absl::StatusOr<Node*> NarrowOrExtend(Node* n, bool n_is_signed,
                                     int64_t resize_to) {
  CHECK_GE(resize_to, 0);

  if (n->BitCountOrDie() < resize_to) {
    return n->function_base()->MakeNode<ExtendOp>(
        n->loc(), n,
        /*new_bit_count=*/resize_to, n_is_signed ? Op::kSignExt : Op::kZeroExt);
  }

  if (n->BitCountOrDie() > resize_to) {
    return n->function_base()->MakeNode<BitSlice>(n->loc(), n,
                                                  /*start=*/0,
                                                  /*width=*/resize_to);
  }

  return n;
}

struct BinaryOpWithLiteral {
  Node* operand;
  Literal* literal;
  bool literal_on_lhs;
  Op op;
};

// Matches the given node as binary operation performed with a literal on the
// lhs or rhs. If match succeeds, returns BinaryOpWithLiteral metadata.
std::optional<BinaryOpWithLiteral> MatchBinaryOpWithLiteral(Node* node) {
  if (node->operand_count() != 2) {
    return std::nullopt;
  }
  if (node->operand(0)->Is<Literal>()) {
    return BinaryOpWithLiteral{.operand = node->operand(1),
                               .literal = node->operand(0)->As<Literal>(),
                               .literal_on_lhs = true,
                               .op = node->op()};
  }
  if (node->operand(1)->Is<Literal>()) {
    return BinaryOpWithLiteral{.operand = node->operand(0),
                               .literal = node->operand(1)->As<Literal>(),
                               .literal_on_lhs = false,
                               .op = node->op()};
  }
  return std::nullopt;
}

// Match a pattern where the result of an injective operation is compared for
// equality/inequality against a constant. An example might be:
//
//   X + C_0 == C_1
//
// Where C_0 and C_1 are constants. In this case the operation `X + C_0` is
// injective, that is, the result of `X + C_0` strictly determines the value of
// `X`. In this case you can simplify the above to:
//
//   X == C_1 - C_0
//
// Operations handled are `kAdd` and `kSub`.
absl::StatusOr<bool> MatchComparisonOfInjectiveOp(Node* node) {
  if (!node->GetType()->IsBits()) {
    return false;
  }
  std::optional<BinaryOpWithLiteral> compare = MatchBinaryOpWithLiteral(node);
  if (!compare.has_value()) {
    return false;
  }
  if (compare->op != Op::kEq && compare->op != Op::kNe) {
    return false;
  }
  std::optional<BinaryOpWithLiteral> binary_op =
      MatchBinaryOpWithLiteral(compare->operand);
  if (!binary_op.has_value()) {
    return false;
  }
  // TODO(meheff): 2024/01/29 Add support for full-width (non-truncating) kUMul.
  if (binary_op->op != Op::kAdd && binary_op->op != Op::kSub) {
    return false;
  }
  Bits solution;
  if (binary_op->op == Op::kAdd) {
    // (X + C_0) cmp C_1  => x cmp C_1 - C_0
    solution = bits_ops::Sub(compare->literal->value().bits(),
                             binary_op->literal->value().bits());
  } else {
    XLS_RET_CHECK_EQ(binary_op->op, Op::kSub);
    if (binary_op->literal_on_lhs) {
      // (C_0 - X) cmp C_1  => x cmp C_0 - C_1
      solution = bits_ops::Sub(binary_op->literal->value().bits(),
                               compare->literal->value().bits());
    } else {
      // (X - C_0) cmp C_1  => x cmp C_0 + C_1
      solution = bits_ops::Add(compare->literal->value().bits(),
                               binary_op->literal->value().bits());
    }
  }
  XLS_ASSIGN_OR_RETURN(
      Literal * new_literal,
      node->function_base()->MakeNode<Literal>(node->loc(), Value(solution)));

  XLS_VLOG(2) << "FOUND: compairson of injective operation.";
  XLS_RETURN_IF_ERROR(node->ReplaceUsesWithNew<CompareOp>(
                              binary_op->operand, new_literal, compare->op)
                          .status());
  return true;
}

// Matches unsigned integer division by a constant; replaces with a multiply and
// shift(s).
//
// Returns 'true' if the IR was modified (uses of node was replaced with a
// different expression).
//
// In accordance with XLS semantics
// (https://google.github.io/xls/ir_semantics/), quotient is rounded towards 0.
//
// Note: the source for the algorithms used to optimize division by constant is
// "Division by Invariant Integers using Multiplication"
// https://gmplib.org/~tege/divcnst-pldi94.pdf
absl::StatusOr<bool> MatchUnsignedDivide(Node* original_div_op) {
  if (original_div_op->op() == Op::kUDiv &&
      original_div_op->operand(1)->Is<Literal>()) {
    const Bits& rhs =
        original_div_op->operand(1)->As<Literal>()->value().bits();
    if (!rhs.IsPowerOfTwo()  // power of two is handled elsewhere.
        && !rhs.IsZero()     // div by 0 is handled elsewhere
    ) {
      FunctionBase* fb = original_div_op->function_base();
      const SourceInfo loc = original_div_op->loc();

      Node* dividend = original_div_op->operand(0);

      UnsignedDivisionConstants division_constants =
          ComputeUnsignedDivisionConstants(dividend->BitCountOrDie(),
                                           BigInt::MakeUnsigned(rhs));

      // In the paper, there is a case for m >= 2^N.
      // Comparing the "large m" case to the m < 2^N case (AKA "small m case"),
      // the "large m" case has:
      //  * 2 more subs
      //  * 1 more add
      //  * a narrower multiply (one operand is 1 bit narrower - details below)
      //
      // I believe the large m case only exists because the paper assumes that
      // the computer's widest multiplier is N bits. Obviously we can synthesize
      // a wider multiplier. I'm not sure what's more optimal (in terms of
      // latency, area, power):
      // * have a separate large m case
      // * fold the large m case into the small m case
      //
      // Careful reading reveals that in the large m case, m is just 1 bit wider
      // than in the small m case. That means that the penalty for reusing the
      // small m algorithm for the large m case is small: one multiplier operand
      // is 1 bit wider. I also verified this empirically for all divisors in
      // [1, 2^24].
      //
      // It's less work to reuse the small m case, so that's what we do.
      //
      // If you want to add the large m case, the condition is:
      // if (division_constants.m >= Exp2<uint64_t>(division_constants.N))
      // the expression to implement is:
      // t1 = mulhi(m-Exp2<uint64_t>(N), numerator)
      // return SRL(t1 + SRL(numerator-t1,1), post_shift-1)

      // The operations below implement the small m case. The expression is:
      // SRL(mulhi(m, SRL(numerator, pre_shift)), post_shift)

      // The dividend after pre_shift.
      Node* shift_dividend;
      if (division_constants.pre_shift > 0) {
        // SRL(dividend, pre_shift)
        //
        // We use BitSlice instead of SRL, because BitSlice produces a
        // narrower result (and the extra bits returned by SRL are zeros).
        XLS_ASSIGN_OR_RETURN(
            shift_dividend,
            fb->MakeNode<BitSlice>(
                loc, dividend,
                /*start=*/division_constants.pre_shift,
                /*width=*/
                dividend->BitCountOrDie() - division_constants.pre_shift));
      } else {
        shift_dividend = dividend;
      }

      // Mul(m, ...)
      XLS_ASSIGN_OR_RETURN(
          Node * multiplicand_literal,
          fb->MakeNode<Literal>(loc,
                                Value(division_constants.m.ToUnsignedBits())));
      XLS_ASSIGN_OR_RETURN(
          Node * multiply,
          fb->MakeNode<ArithOp>(loc, multiplicand_literal, shift_dividend,
                                multiplicand_literal->BitCountOrDie() +
                                    shift_dividend->BitCountOrDie(),
                                Op::kUMul));

      // Notes on how this works:
      // m / 2^post_shift ~= 1/divisor
      // m is effectively a fixed point value with N fractional bits (and
      // frequently 0 but sometimes more than 0 integer bits).
      // We're only interested in the integer part of dividend/divisor, thus
      // we're only interested in the integer part of (dividend * m). So we
      // discard the fractional bits of (m * dividend), via BitSlice.
      XLS_ASSIGN_OR_RETURN(
          Node * integer_part_of_product,
          fb->MakeNode<BitSlice>(
              loc, multiply,
              /*start=*/division_constants.N,
              /*width=*/
              multiply->BitCountOrDie() - division_constants.N));

      // SRL(..., post_shift)
      XLS_ASSIGN_OR_RETURN(
          Node * shift_amount_literal,
          fb->MakeNode<Literal>(
              loc, Value(UBits(division_constants.post_shift,
                               Bits::MinBitCountUnsigned(
                                   division_constants.post_shift)))));
      XLS_ASSIGN_OR_RETURN(
          Node * post_shift,
          fb->MakeNode<BinOp>(loc, integer_part_of_product,
                              shift_amount_literal, Op::kShrl));

      XLS_ASSIGN_OR_RETURN(
          Node * resize_post_shift,
          NarrowOrExtend(post_shift, false, original_div_op->BitCountOrDie()));

      XLS_RETURN_IF_ERROR(original_div_op->ReplaceUsesWith(resize_post_shift));
      return true;
    }
  }

  return false;
}

// Matches signed integer division by a constant; replaces with a multiply and
// shift(s).
//
// Returns 'true' if the IR was modified (uses of node was replaced with a
// different expression).
//
// In accordance with XLS semantics
// (https://google.github.io/xls/ir_semantics/), quotient is rounded towards 0.
//
// Note: the source for the algorithms used to optimize divison by constant is
// "Division by Invariant Integers using Multiplication"
// https://gmplib.org/~tege/divcnst-pldi94.pdf
absl::StatusOr<bool> MatchSignedDivide(Node* original_div_op) {
  // TODO paper mentions overflow when n=-(2^(N-1)) and d=-1. Make sure I handle
  // that correctly. I'm not sure if Figure 5.2 does.
  if (original_div_op->op() == Op::kSDiv &&
      original_div_op->operand(1)->Is<Literal>()) {
    const Bits& rhs =
        original_div_op->operand(1)->As<Literal>()->value().bits();
    if (!rhs.IsZero()  // div by 0 is handled elsewhere
    ) {
      FunctionBase* fb = original_div_op->function_base();
      const SourceInfo loc = original_div_op->loc();

      Node* dividend = original_div_op->operand(0);

      const BigInt divisor = BigInt::MakeSigned(rhs);
      const BigInt magnitude_divisor = BigInt::Absolute(divisor);
      const bool divisor_negative = divisor < BigInt::Zero();

      const int64_t n = dividend->BitCountOrDie();
      auto [m, post_shift, clog2_divisor] =
          ChooseMultiplier(n, magnitude_divisor, n - 1);

      auto maybe_negate = [divisor_negative, &fb,
                           &loc](Node* quotient) -> absl::StatusOr<Node*> {
        if (divisor_negative) {
          return fb->MakeNode<UnOp>(loc, quotient, Op::kNeg);
        }

        return quotient;
      };

      if (magnitude_divisor == BigInt::One()) {
        XLS_ASSIGN_OR_RETURN(Node * negate, maybe_negate(dividend));
        XLS_RETURN_IF_ERROR(original_div_op->ReplaceUsesWith(negate));
        return true;
      }

      if (magnitude_divisor == BigInt::Exp2(clog2_divisor)) {
        // case: |d| = 2^l
        // q = SRA(n + SRL(SRA(n, l − 1), N − l), l)

        // SRA(n, l − 1)
        const int64_t first_sra_amount = clog2_divisor - 1;
        XLS_ASSIGN_OR_RETURN(
            Node * first_sra_literal,
            fb->MakeNode<Literal>(
                loc, Value(UBits(first_sra_amount, Bits::MinBitCountUnsigned(
                                                       first_sra_amount)))));
        XLS_ASSIGN_OR_RETURN(
            Node * first_sra,
            fb->MakeNode<BinOp>(loc, dividend, first_sra_literal, Op::kShra));

        // SRL(..., N − l)
        const int64_t shift_right_logical_amount = n - clog2_divisor;
        XLS_ASSIGN_OR_RETURN(
            Node * srl_literal,
            fb->MakeNode<Literal>(
                loc, Value(UBits(shift_right_logical_amount,
                                 Bits::MinBitCountUnsigned(
                                     shift_right_logical_amount)))));
        XLS_ASSIGN_OR_RETURN(
            Node * srl,
            fb->MakeNode<BinOp>(loc, first_sra, srl_literal, Op::kShrl));

        // n + SRL(...)
        XLS_ASSIGN_OR_RETURN(Node * add,
                             fb->MakeNode<BinOp>(loc, dividend, srl, Op::kAdd));

        // SRA(..., l)
        const int64_t second_sra_amount = clog2_divisor;
        XLS_ASSIGN_OR_RETURN(
            Node * second_sra_literal,
            fb->MakeNode<Literal>(
                loc, Value(UBits(second_sra_amount, Bits::MinBitCountUnsigned(
                                                        second_sra_amount)))));
        XLS_ASSIGN_OR_RETURN(
            Node * second_sra,
            fb->MakeNode<BinOp>(loc, add, second_sra_literal, Op::kShra));

        // if d<0 then negate q
        XLS_ASSIGN_OR_RETURN(Node * negate, maybe_negate(second_sra));
        XLS_RETURN_IF_ERROR(original_div_op->ReplaceUsesWith(negate));
        return true;
      }

      // In the paper, there is a case for m >= 2^N. However, we reuse the
      // same logic as for the "small m" case. See comment in
      // MatchUnsignedDivide for details.

      // q = SRA(MULSH(m, n), post_shift) - XSIGN(n)

      // MULSH(m, n)
      XLS_ASSIGN_OR_RETURN(Node * multiplicand_literal,
                           fb->MakeNode<Literal>(loc, Value(m.ToSignedBits())));
      XLS_ASSIGN_OR_RETURN(
          Node * multiply,
          fb->MakeNode<ArithOp>(
              loc, multiplicand_literal, dividend,
              multiplicand_literal->BitCountOrDie() + dividend->BitCountOrDie(),
              Op::kSMul));

      // m is a fixed point value with N fractional bits. So discard the N
      // least significant bits of m * dividend. See comment in
      // MatchUnsignedDivide for details.
      XLS_ASSIGN_OR_RETURN(
          Node * integer_part_of_product,
          fb->MakeNode<BitSlice>(loc, multiply,
                                 /*start=*/n,
                                 /*width=*/
                                 multiply->BitCountOrDie() - n));

      // SRA(..., post_shift)
      XLS_ASSIGN_OR_RETURN(
          Node * shift_right_literal,
          fb->MakeNode<Literal>(
              loc,
              Value(UBits(post_shift, Bits::MinBitCountUnsigned(post_shift)))));
      XLS_ASSIGN_OR_RETURN(Node * shift_right,
                           fb->MakeNode<BinOp>(loc, integer_part_of_product,
                                               shift_right_literal, Op::kShra));

      // XSIGN(n) = SRA(n, N − 1)
      const int64_t xsign_sra_amount = n - 1;
      XLS_ASSIGN_OR_RETURN(
          Node * xsign_sra_literal,
          fb->MakeNode<Literal>(
              loc, Value(UBits(xsign_sra_amount,
                               Bits::MinBitCountUnsigned(xsign_sra_amount)))));
      XLS_ASSIGN_OR_RETURN(
          Node * xsign,
          fb->MakeNode<BinOp>(loc, dividend, xsign_sra_literal, Op::kShra));

      // SRA(..., post_shift) - XSIGN(n)
      XLS_ASSIGN_OR_RETURN(
          Node * resize_shift_right,
          NarrowOrExtend(shift_right, true, dividend->BitCountOrDie()));
      XLS_ASSIGN_OR_RETURN(
          Node * subtract,
          fb->MakeNode<BinOp>(loc, resize_shift_right, xsign, Op::kSub));

      // if d<0 then negate q
      XLS_ASSIGN_OR_RETURN(Node * negate, maybe_negate(subtract));
      XLS_RETURN_IF_ERROR(original_div_op->ReplaceUsesWith(negate));
      return true;
    }
  }

  return false;
}

// MatchArithPatterns matches simple tree patterns to find opportunities
// for simplification, such as adding a zero, multiplying by 1, etc.
//
// Return 'true' if the IR was modified (uses of node was replaced with a
// different expression).
absl::StatusOr<bool> MatchArithPatterns(int64_t opt_level, Node* n) {
  // Pattern: Add/Sub/Or/Xor/Shift a value with 0 on the RHS.
  if ((n->op() == Op::kAdd || n->op() == Op::kSub || n->op() == Op::kShll ||
       n->op() == Op::kShrl || n->op() == Op::kShra) &&
      IsLiteralZero(n->operand(1))) {
    XLS_VLOG(2) << "FOUND: Useless operation of value with zero";
    XLS_RETURN_IF_ERROR(n->ReplaceUsesWith(n->operand(0)));
    return true;
  }

  // Returns true if all operands of 'node' are the same.
  auto all_operands_same = [](Node* node) {
    return std::all_of(node->operands().begin(), node->operands().end(),
                       [node](Node* op) { return op == node->operand(0); });
  };

  // Duplicate operands of AND and OR (and their inverting forms NAND and OR)
  // can be removed.
  //
  //   Op(x, y, y, x)  =>  Op(x, y)
  if (n->Is<NaryOp>() && (n->op() == Op::kAnd || n->op() == Op::kOr ||
                          n->op() == Op::kNand || n->op() == Op::kNor)) {
    std::vector<Node*> unique_operands;
    for (Node* operand : n->operands()) {
      // This is quadratic in the number of operands, but shouldn't cause
      // problems unless we have at least hundreds of thousands of operands
      // which seems unlikely.
      if (std::find(unique_operands.begin(), unique_operands.end(), operand) ==
          unique_operands.end()) {
        unique_operands.push_back(operand);
      }
    }
    if (unique_operands.size() != n->operand_count()) {
      XLS_VLOG(2) << "FOUND: remove duplicate operands in and/or/nand/nor";
      XLS_RETURN_IF_ERROR(
          n->ReplaceUsesWithNew<NaryOp>(unique_operands, n->op()).status());
      return true;
    }
  }

  // Single operand forms of non-inverting logical ops (AND, OR) can be
  // replaced with the operand.
  //
  //   Op(x)  =>  x
  if (n->Is<NaryOp>() && (n->op() == Op::kAnd || n->op() == Op::kOr) &&
      n->operand_count() == 1) {
    XLS_VLOG(2) << "FOUND: replace single operand or(x) with x";
    XLS_RETURN_IF_ERROR(n->ReplaceUsesWith(n->operand(0)));
    return true;
  }

  // Single operand forms of inverting logical ops (NAND, NOR) can be
  // replaced with the inverted operand.
  //
  //   Op(x)  =>  Not(x)
  if (n->Is<NaryOp>() && (n->op() == Op::kNor || n->op() == Op::kNand) &&
      n->operand_count() == 1) {
    XLS_VLOG(2) << "FOUND: replace single operand nand/nor(x) with not(x)";
    XLS_RETURN_IF_ERROR(
        n->ReplaceUsesWithNew<UnOp>(n->operand(0), Op::kNot).status());
    return true;
  }

  // All operands the same for XOR:
  //
  //   XOR(x, x, ...)  =>  x  // Odd number of operands.
  //   XOR(x, x, ...)  =>  0  // Even number of operands.
  if (n->op() == Op::kXor && all_operands_same(n)) {
    XLS_VLOG(2) << "FOUND: replace xor(x, x, ...) with 0 or 1";
    if (n->operand_count() % 2 == 0) {
      // Even number of operands. Replace with zero.
      XLS_RETURN_IF_ERROR(
          n->ReplaceUsesWithNew<Literal>(Value(UBits(0, n->BitCountOrDie())))
              .status());
    } else {
      // Odd number of operands. Replace with XOR operand.
      XLS_RETURN_IF_ERROR(n->ReplaceUsesWith(n->operand(0)));
    }
    return true;
  }

  // Replaces uses of n with a new node by eliminating operands for which the
  // "predicate" holds. If the predicate holds for all operands, the
  // NaryOpNullaryResult is used as a replacement.
  auto eliminate_operands_where =
      [n](std::function<bool(Node*)> predicate) -> absl::StatusOr<bool> {
    XLS_RET_CHECK(n->Is<NaryOp>());
    std::vector<Node*> new_operands;
    for (Node* operand : n->operands()) {
      if (!predicate(operand)) {
        new_operands.push_back(operand);
      }
    }
    if (new_operands.size() == n->operand_count()) {
      return false;
    }
    if (new_operands.empty()) {
      XLS_RETURN_IF_ERROR(
          n
              ->ReplaceUsesWithNew<Literal>(Value(DoLogicalOp(
                  n->op(), {LogicalOpIdentity(n->op(), n->BitCountOrDie())})))
              .status());
    } else {
      XLS_RETURN_IF_ERROR(
          n->ReplaceUsesWithNew<NaryOp>(new_operands, n->op()).status());
    }
    return true;
  };

  // Or(x, 0, y) => Or(x, y)
  // Xor(x, 0, y) => Xor(x, y)
  // Nor(x, 0, y) => Nor(x, y)
  if ((n->op() == Op::kOr || n->op() == Op::kXor || n->op() == Op::kNor)) {
    XLS_VLOG(2) << "FOUND: remove zero valued operands from or, nor, or, xor";
    XLS_ASSIGN_OR_RETURN(bool changed, eliminate_operands_where(IsLiteralZero));
    if (changed) {
      return true;
    }
  }

  // Or(x, -1, y) => -1
  if (n->op() == Op::kOr && AnyOperandWhere(n, IsLiteralAllOnes)) {
    XLS_VLOG(2) << "FOUND: replace or(..., 1, ...) with 1";
    XLS_RETURN_IF_ERROR(
        n->ReplaceUsesWithNew<Literal>(AllOnesOfType(n->GetType())).status());
    return true;
  }

  // Nor(x, -1, y) => 0
  if (n->op() == Op::kNor && AnyOperandWhere(n, IsLiteralAllOnes)) {
    XLS_VLOG(2) << "FOUND: replace nor(..., 1, ...) with 0";
    XLS_RETURN_IF_ERROR(
        n->ReplaceUsesWithNew<Literal>(ZeroOfType(n->GetType())).status());
    return true;
  }

  // And(x, -1, y) => And(x, y)
  // Nand(x, -1, y) => Nand(x, y)
  if (n->op() == Op::kAnd || n->op() == Op::kNand) {
    XLS_VLOG(2) << "FOUND: remove all-ones operands from and/nand";
    XLS_ASSIGN_OR_RETURN(bool changed,
                         eliminate_operands_where(IsLiteralAllOnes));
    if (changed) {
      return true;
    }
  }

  // And(x, 0) => 0
  if (n->op() == Op::kAnd && AnyOperandWhere(n, IsLiteralZero)) {
    XLS_VLOG(2) << "FOUND: replace and(..., 0, ...) with 0";
    XLS_RETURN_IF_ERROR(
        n->ReplaceUsesWithNew<Literal>(ZeroOfType(n->GetType())).status());
    return true;
  }

  // Nand(x, 0) => 1
  if (n->op() == Op::kNand && AnyOperandWhere(n, IsLiteralZero)) {
    XLS_VLOG(2) << "FOUND: replace nand(..., 0, ...) with 1";
    XLS_RETURN_IF_ERROR(
        n->ReplaceUsesWithNew<Literal>(AllOnesOfType(n->GetType())).status());
    return true;
  }

  auto has_inverted_operand = [&] {
    for (Node* operand : n->operands()) {
      if (operand->op() == Op::kNot &&
          std::find(n->operands().begin(), n->operands().end(),
                    operand->operand(0)) != n->operands().end()) {
        return true;
      }
    }
    return false;
  };

  // And(x, Not(x)) => 0
  // And(Not(x), x) => 0
  //
  // Note that this won't be found through the ternary query engine because
  // conservatively it determines `not(UNKNOWN) = UNKNOWN`.
  if (n->op() == Op::kAnd && has_inverted_operand()) {
    XLS_VLOG(2) << "FOUND: replace and(x, not(x)) with 0";
    XLS_RETURN_IF_ERROR(
        n->ReplaceUsesWithNew<Literal>(Value(UBits(0, n->BitCountOrDie())))
            .status());
    return true;
  }

  //   Or(X, Not(X), ...)  => 1...
  //
  // Note that this won't be found through the ternary query engine because
  // conservatively it determines `not(UNKNOWN) = UNKNOWN`.
  if (n->op() == Op::kOr && has_inverted_operand()) {
    XLS_VLOG(2) << "FOUND: replace Or(x, not(x)) with 1...";
    XLS_RETURN_IF_ERROR(
        n->ReplaceUsesWithNew<Literal>(Value(Bits::AllOnes(n->BitCountOrDie())))
            .status());
    return true;
  }

  //   Add(X, Not(X)) => 1...
  //   Add(Not(X), X) => 1...
  //
  // Note that this won't be found through the ternary query engine because
  // conservatively it determines `not(UNKNOWN) = UNKNOWN`.
  if (n->op() == Op::kAdd && n->operand_count() == 2 &&
      has_inverted_operand()) {
    XLS_VLOG(2) << "FOUND: replace Add(x, not(x)) with 1...";
    XLS_RETURN_IF_ERROR(
        n->ReplaceUsesWithNew<Literal>(Value(Bits::AllOnes(n->BitCountOrDie())))
            .status());
    return true;
  }

  // Xor(x, -1) => Not(x)
  if (n->op() == Op::kXor && n->operand_count() == 2 &&
      IsLiteralAllOnes(n->operand(1))) {
    XLS_VLOG(2) << "FOUND: Found xor with all ones";
    XLS_RETURN_IF_ERROR(
        n->ReplaceUsesWithNew<UnOp>(n->operand(0), Op::kNot).status());
    return true;
  }

  // Sub(X, X) => Literal(0)
  if (n->op() == Op::kSub && all_operands_same(n)) {
    XLS_VLOG(2) << "FOUND: Sub of value with itself";
    XLS_RETURN_IF_ERROR(
        n->ReplaceUsesWithNew<Literal>(Value(UBits(0, n->BitCountOrDie())))
            .status());
    return true;
  }

  auto is_same_opcode = [&](Node* other) { return n->op() == other->op(); };

  // Flatten nested associative nary ops into their root op:
  //
  //   Op(Op(x, y), z)  =>  Op(x, y, z)
  //
  // This operation should only be performed if the only user of the nested ops
  // is the outer op itself.
  if (OpIsAssociative(n->op()) && n->Is<NaryOp>() &&
      AnyOperandWhere(n, is_same_opcode)) {
    std::vector<Node*> new_operands;
    bool should_transform = false;
    for (Node* operand : n->operands()) {
      if (operand->op() == n->op() && HasSingleUse(operand)) {
        should_transform = true;
        for (Node* suboperand : operand->operands()) {
          new_operands.push_back(suboperand);
        }
      } else {
        new_operands.push_back(operand);
      }
    }
    if (should_transform) {
      XLS_VLOG(2) << "FOUND: flatten nested associative nary ops";
      XLS_RETURN_IF_ERROR(
          n->ReplaceUsesWithNew<NaryOp>(new_operands, n->op()).status());
      return true;
    }
  }

  // Fold the literal values presented to the nary op.
  //
  //   Op(C0, x, C1)  =>  Op(C2, x) where C2 == Op(C0, C1)
  if (OpIsCommutative(n->op()) && OpIsAssociative(n->op()) && n->Is<NaryOp>() &&
      AnyTwoOperandsWhere(n, IsLiteral)) {
    std::vector<Node*> new_operands;
    Bits bits = LogicalOpIdentity(n->op(), n->BitCountOrDie());
    for (Node* operand : n->operands()) {
      if (operand->Is<Literal>()) {
        bits = DoLogicalOp(n->op(),
                           {bits, operand->As<Literal>()->value().bits()});
      } else {
        new_operands.push_back(operand);
      }
    }
    XLS_ASSIGN_OR_RETURN(Node * literal, n->function_base()->MakeNode<Literal>(
                                             n->loc(), Value(bits)));
    new_operands.push_back(literal);
    XLS_VLOG(2)
        << "FOUND: fold literal operands of associative commutative operation";
    XLS_RETURN_IF_ERROR(
        n->ReplaceUsesWithNew<NaryOp>(new_operands, n->op()).status());
    return true;
  }

  // Pattern: UDiv/UMul/SMul by a positive power of two.
  if ((n->op() == Op::kSMul || n->op() == Op::kUMul || n->op() == Op::kUDiv) &&
      n->operand(1)->Is<Literal>()) {
    const Bits& rhs = n->operand(1)->As<Literal>()->value().bits();
    const bool is_signed = n->op() == Op::kSMul;
    if (rhs.IsPowerOfTwo() &&
        (!is_signed || (is_signed && bits_ops::SGreaterThan(rhs, 0)))) {
      XLS_VLOG(2) << "FOUND: Div/Mul by positive power of two";
      // Extend/trunc operand 0 (the non-literal operand) to the width of the
      // div/mul then shift by a constant amount.
      XLS_ASSIGN_OR_RETURN(
          Node * adjusted_lhs,
          NarrowOrExtend(n->operand(0), is_signed, n->BitCountOrDie()));
      XLS_ASSIGN_OR_RETURN(Node * shift_amount,
                           n->function_base()->MakeNode<Literal>(
                               n->loc(), Value(UBits(rhs.CountTrailingZeros(),
                                                     rhs.bit_count()))));
      Op shift_op;
      if (n->op() == Op::kSMul || n->op() == Op::kUMul) {
        // Multiply operation is replaced with shift left;
        shift_op = Op::kShll;
      } else {
        // Unsigned divide operation is replaced with shift right logical.
        shift_op = Op::kShrl;
      }
      XLS_VLOG(2) << "FOUND: div/mul of positive power of two";
      XLS_RETURN_IF_ERROR(
          n->ReplaceUsesWithNew<BinOp>(adjusted_lhs, shift_amount, shift_op)
              .status());
      return true;
    }
  }

  // Pattern: UMod/SMod by a literal.
  if (n->OpIn({Op::kUMod, Op::kSMod}) && n->operand(1)->Is<Literal>()) {
    const Bits& rhs = n->operand(1)->As<Literal>()->value().bits();
    const bool is_signed = n->op() == Op::kSMod;

    if (rhs.IsOne() || rhs.IsZero()) {
      XLS_VLOG(2) << "FOUND: UMod/SMod by one or zero";
      XLS_RETURN_IF_ERROR(
          n->ReplaceUsesWithNew<Literal>(ZeroOfType(n->GetType())).status());
      return true;
    }

    if (rhs.IsPowerOfTwo() &&
        (!is_signed || (is_signed && bits_ops::SGreaterThan(rhs, 0)))) {
      XLS_VLOG(2) << "FOUND: UMod/SMod by a positive power of two";
      // Truncate operand 0 to the relevant low bits, then zero-extend it to the
      // width of the mod.
      XLS_ASSIGN_OR_RETURN(Node * truncated_lhs,
                           NarrowOrExtend(n->operand(0), /*n_is_signed=*/false,
                                          rhs.CountTrailingZeros()));
      XLS_ASSIGN_OR_RETURN(Node * nonnegative_result,
                           NarrowOrExtend(truncated_lhs, /*n_is_signed=*/false,
                                          n->BitCountOrDie()));
      if (!is_signed) {
        XLS_RETURN_IF_ERROR(n->ReplaceUsesWith(nonnegative_result));
        return true;
      }

      // By XLS convention, the sign of the modulus matches the sign of the LHS,
      // so we should subtract the RHS from our result to get the negative
      // remainder iff the LHS is negative and the result would otherwise be
      // positive.
      XLS_ASSIGN_OR_RETURN(
          Node * truncated_zero,
          n->function_base()->MakeNode<Literal>(
              n->loc(), Value(UBits(0, truncated_lhs->BitCountOrDie()))));
      XLS_ASSIGN_OR_RETURN(
          Node * result_is_positive,
          n->function_base()->MakeNode<CompareOp>(n->loc(), truncated_lhs,
                                                  truncated_zero, Op::kNe));
      XLS_ASSIGN_OR_RETURN(
          Node * lhs_is_negative,
          n->function_base()->MakeNode<BitSlice>(
              n->loc(), n->operand(0),
              /*start=*/n->operand(0)->BitCountOrDie() - 1, /*width=*/1));
      XLS_ASSIGN_OR_RETURN(
          Node * negative_result,
          n->function_base()->MakeNode<BinOp>(n->loc(), nonnegative_result,
                                              n->operand(1), Op::kSub));
      XLS_ASSIGN_OR_RETURN(
          Node * use_negative_result,
          n->function_base()->MakeNode<NaryOp>(
              n->loc(),
              std::vector<Node*>({result_is_positive, lhs_is_negative}),
              Op::kAnd));
      XLS_RETURN_IF_ERROR(
          n->ReplaceUsesWithNew<Select>(
               use_negative_result,
               std::vector<Node*>({nonnegative_result, negative_result}),
               /*default_value=*/std::nullopt)
              .status());
      return true;
    }

    // Convert UMod/SMod by any other literal into a divide, multiply, and
    // subtract; we'll later simplify the divide into a multiply & shift(s).
    // (See MatchUnsignedDivide for how.)
    XLS_ASSIGN_OR_RETURN(Node * quotient,
                         n->function_base()->MakeNode<BinOp>(
                             n->loc(), n->operand(0), n->operand(1),
                             is_signed ? Op::kSDiv : Op::kUDiv));
    XLS_ASSIGN_OR_RETURN(Node * approximant,
                         n->function_base()->MakeNode<ArithOp>(
                             n->loc(), quotient, n->operand(1),
                             /*width=*/n->operand(0)->BitCountOrDie(),
                             is_signed ? Op::kSMul : Op::kUMul));
    XLS_RETURN_IF_ERROR(
        n->ReplaceUsesWithNew<BinOp>(n->operand(0), approximant, Op::kSub)
            .status());
    return true;
  }

  XLS_ASSIGN_OR_RETURN(bool udiv_matched, MatchUnsignedDivide(n));
  if (udiv_matched) {
    return true;
  }

  XLS_ASSIGN_OR_RETURN(bool sdiv_matched, MatchSignedDivide(n));
  if (sdiv_matched) {
    return true;
  }

  // Pattern: Mul by 0
  if ((n->op() == Op::kSMul || n->op() == Op::kUMul) &&
      IsLiteralZero(n->operand(1))) {
    XLS_VLOG(2) << "FOUND: Mul by 0";
    XLS_RETURN_IF_ERROR(
        n->ReplaceUsesWithNew<Literal>(Value(UBits(0, n->BitCountOrDie())))
            .status());
    return true;
  }

  // Pattern: Not(Not(x)) => x
  if (n->op() == Op::kNot && n->operand(0)->op() == Op::kNot) {
    XLS_VLOG(2) << "FOUND: replace not(not(x)) with x";
    XLS_RETURN_IF_ERROR(n->ReplaceUsesWith(n->operand(0)->operand(0)));
    return true;
  }

  // Logical shift by a constant can be replaced by a slice and concat.
  //    (val << lit) -> Concat(BitSlice(val, ...), UBits(0, ...))
  //    (val >> lit) -> Concat(UBits(0, ...), BitSlice(val, ...))
  // If the shift amount is greater than or equal to the bit width the
  // expression can be replaced with zero.
  //
  // This simplification is desirable as in the canonical lower-level IR a shift
  // implies a barrel shifter which is not necessary for a shift by a constant
  // amount.
  if ((n->op() == Op::kShll || n->op() == Op::kShrl) &&
      n->operand(1)->Is<Literal>()) {
    int64_t bit_count = n->BitCountOrDie();
    const Bits& shift_bits = n->operand(1)->As<Literal>()->value().bits();
    if (shift_bits.IsZero()) {
      // A shift by zero is a nop.
      XLS_RETURN_IF_ERROR(n->ReplaceUsesWith(n->operand(0)));
      return true;
    }
    if (bits_ops::UGreaterThanOrEqual(shift_bits, bit_count)) {
      // Replace with zero.
      XLS_RETURN_IF_ERROR(
          n->ReplaceUsesWithNew<Literal>(Value(UBits(0, bit_count))).status());
      return true;
    }
    XLS_ASSIGN_OR_RETURN(uint64_t shift_amount, shift_bits.ToUint64());
    XLS_ASSIGN_OR_RETURN(Node * zero,
                         n->function_base()->MakeNode<Literal>(
                             n->loc(), Value(UBits(0, shift_amount))));
    int64_t slice_start = (n->op() == Op::kShll) ? 0 : shift_amount;
    int64_t slice_width = bit_count - shift_amount;
    XLS_ASSIGN_OR_RETURN(
        Node * slice, n->function_base()->MakeNode<BitSlice>(
                          n->loc(), n->operand(0), slice_start, slice_width));
    auto concat_operands = (n->op() == Op::kShll)
                               ? std::vector<Node*>{slice, zero}
                               : std::vector<Node*>{zero, slice};
    XLS_VLOG(2) << "FOUND: logical shift by constant";
    XLS_RETURN_IF_ERROR(
        n->ReplaceUsesWithNew<Concat>(concat_operands).status());
    return true;
  }

  // SignExt(SignExt(x, w_0), w_1) => SignExt(x, w_1)
  if (n->op() == Op::kSignExt && n->operand(0)->op() == Op::kSignExt) {
    XLS_VLOG(2) << "FOUND: replace signext(signext(x)) with signext(x)";
    XLS_RETURN_IF_ERROR(
        n->ReplaceUsesWithNew<ExtendOp>(n->operand(0)->operand(0),
                                        n->BitCountOrDie(), Op::kSignExt)
            .status());
    return true;
  }

  // An arithmetic shift right by a constant can be replaced by sign-extended
  // slice of the to-shift operand.
  //
  //    (val >>> lit) -> sign_extend(BitSlice(val, ...))
  //
  // This simplification is desirable because in the canonical lower-level IR a
  // shift implies a barrel shifter which is not necessary for a shift by a
  // constant amount.
  if (n->op() == Op::kShra && n->operand(1)->Is<Literal>()) {
    const int64_t bit_count = n->BitCountOrDie();
    const Bits& shift_bits = n->operand(1)->As<Literal>()->value().bits();
    if (shift_bits.IsZero()) {
      // A shift by zero is a nop.
      XLS_RETURN_IF_ERROR(n->ReplaceUsesWith(n->operand(0)));
      return true;
    }
    int64_t slice_width;
    if (bits_ops::UGreaterThanOrEqual(shift_bits, bit_count)) {
      slice_width = 1;
    } else {
      XLS_ASSIGN_OR_RETURN(uint64_t shift_amount, shift_bits.ToUint64());
      slice_width = bit_count - shift_amount;
    }
    XLS_ASSIGN_OR_RETURN(Node * slice, n->function_base()->MakeNode<BitSlice>(
                                           n->loc(), n->operand(0),
                                           /*start=*/bit_count - slice_width,
                                           /*width=*/slice_width));
    XLS_VLOG(2) << "FOUND: replace ashr by constant with signext(slice(x))";
    XLS_RETURN_IF_ERROR(
        n->ReplaceUsesWithNew<ExtendOp>(slice, bit_count, Op::kSignExt)
            .status());
    return true;
  }

  // An arithmetic shift-right of a 1-bit value is a no-op.
  if (n->op() == Op::kShra && n->operand(0)->BitCountOrDie() == 1) {
    XLS_VLOG(2) << "FOUND: arithmetic shift-right of 1-bit value";
    XLS_RETURN_IF_ERROR(n->ReplaceUsesWith(n->operand(0)));
    return true;
  }

  // Any shift right of a constant 1 can be replaced by a test for equality with
  // zero (followed by a zero extension), since
  //
  //   (1 >> x) = 1 if x == 0,
  //              0 otherwise.
  //
  // ... as long as it's not an arithmetic shift-right of bits[1]:1.
  if ((n->op() == Op::kShra || n->op() == Op::kShrl) &&
      IsLiteralUnsignedOne(n->operand(0))) {
    // Make absolutely sure we're not dealing with an arithmetic shift-right of
    // bits[1]:1. (For correctness, that case must be handled first.)
    XLS_RET_CHECK(n->op() != Op::kShra || n->operand(0)->BitCountOrDie() > 1);

    XLS_VLOG(2) << "FOUND: shift right of constant 1";
    Node* x = n->operand(1);
    XLS_ASSIGN_OR_RETURN(Literal * zero,
                         n->function_base()->MakeNode<Literal>(
                             n->loc(), Value(UBits(0, x->BitCountOrDie()))));
    XLS_ASSIGN_OR_RETURN(Node * test, n->function_base()->MakeNode<CompareOp>(
                                          n->loc(), x, zero, Op::kEq));
    if (n->BitCountOrDie() == 1) {
      XLS_RETURN_IF_ERROR(n->ReplaceUsesWith(test));
    } else {
      XLS_RETURN_IF_ERROR(n->ReplaceUsesWithNew<ExtendOp>(
                               test, n->BitCountOrDie(), Op::kZeroExt)
                              .status());
    }
    return true;
  }

  // A 1-wide decode can be simplified to a test for equality with zero; any
  // other index results in zero output.
  if (n->op() == Op::kDecode && n->BitCountOrDie() == 1) {
    XLS_VLOG(2) << "FOUND: 1-wide decode";
    XLS_ASSIGN_OR_RETURN(
        Literal * zero,
        n->function_base()->MakeNode<Literal>(
            n->loc(), Value(UBits(0, n->operand(0)->BitCountOrDie()))));
    XLS_RETURN_IF_ERROR(
        n->ReplaceUsesWithNew<CompareOp>(n->operand(0), zero, Op::kEq)
            .status());
    return true;
  }

  // A 2-wide decode with a 1-bit operand can be simplified to:
  //   decode(operand:u1) => concat(operand, not(operand)).
  // 1-wide decodes should be handled above, and no wider decode is allowed for
  // a 1-bit operand.
  if (n->op() == Op::kDecode && n->operand(0)->BitCountOrDie() == 1 &&
      n->BitCountOrDie() == 2) {
    XLS_VLOG(2) << "FOUND: decode of 1-bit operand";
    XLS_ASSIGN_OR_RETURN(
        Node * not_operand,
        n->function_base()->MakeNode<UnOp>(n->loc(), n->operand(0), Op::kNot));
    XLS_RETURN_IF_ERROR(n->ReplaceUsesWithNew<Concat>(
                             std::vector<Node*>({n->operand(0), not_operand}))
                            .status());
    return true;
  }

  // Pattern: Double negative.
  //   -(-expr)
  if (n->op() == Op::kNeg && n->operand(0)->op() == Op::kNeg) {
    XLS_VLOG(2) << "FOUND: Double negative";
    XLS_RETURN_IF_ERROR(n->ReplaceUsesWith(n->operand(0)->operand(0)));
    return true;
  }

  // We can eliminate negations that are only used in signed comparisons to
  // literals or other negations, at minimal cost.
  //
  // NOTE: This is expected to reduce delay in other scenarios too, but we
  //       currently have no way to model the tradeoff between area & delay, so
  //       we only do this where we can fully eliminate one negation.
  auto is_removable_negate = [](Node* node) {
    return node->op() == Op::kNeg &&
           !node->function_base()->HasImplicitUse(node) &&
           absl::c_all_of(node->users(), [](Node* user) {
             return IsSignedCompare(user) &&
                    user->operand(0)->op() == Op::kNeg &&
                    (user->operand(1)->Is<Literal>() ||
                     user->operand(1)->op() == Op::kNeg);
           });
  };

  // Pattern: Signed comparison of negated operands
  //   eq(-lhs, -rhs)  =>  eq(lhs, rhs)
  //   ne(-lhs, -rhs)  =>  ne(lhs, rhs)
  //    (-lhs < -rhs)  =>  (lhs > rhs) XOR (lhs != MIN) XOR (rhs != MIN)
  //    (-lhs > -rhs)  =>  (lhs < rhs) XOR (lhs != MIN) XOR (rhs != MIN)
  //   (-lhs <= -rhs)  =>  (lhs >= rhs) XOR (lhs != MIN) XOR (rhs != MIN)
  //   (-lhs >= -rhs)  =>  (lhs <= rhs) XOR (lhs != MIN) XOR (rhs != MIN)
  if (IsBitsCompare(n) && IsSignedCompare(n) &&
      n->operand(0)->op() == Op::kNeg && n->operand(1)->op() == Op::kNeg &&
      (is_removable_negate(n->operand(0)) ||
       is_removable_negate(n->operand(1)))) {
    XLS_VLOG(2)
        << "FOUND: Signed comparison of negated operands with no other use";
    Node* lhs = n->operand(0)->operand(0);
    Node* rhs = n->operand(1)->operand(0);

    Node* equivalent;
    XLS_ASSIGN_OR_RETURN(Op reversed_op, ReverseComparisonOp(n->op()));
    XLS_ASSIGN_OR_RETURN(Node * reversed_cmp,
                         n->function_base()->MakeNode<CompareOp>(
                             n->loc(), lhs, rhs, reversed_op));
    if (n->op() == Op::kEq || n->op() == Op::kNe) {
      equivalent = reversed_cmp;
    } else {
      XLS_RET_CHECK_EQ(n->operand(0)->BitCountOrDie(),
                       n->operand(1)->BitCountOrDie());
      const int64_t bit_count = n->operand(0)->BitCountOrDie();
      InlineBitmap hi_bit(bit_count);
      hi_bit.Set(bit_count - 1);
      XLS_ASSIGN_OR_RETURN(
          Literal * min_value,
          n->function_base()->MakeNode<Literal>(
              n->loc(), Value(Bits::FromBitmap(std::move(hi_bit)))));
      XLS_ASSIGN_OR_RETURN(Node * lhs_not_min,
                           n->function_base()->MakeNode<CompareOp>(
                               n->loc(), lhs, min_value, Op::kNe));
      XLS_ASSIGN_OR_RETURN(Node * rhs_not_min,
                           n->function_base()->MakeNode<CompareOp>(
                               n->loc(), rhs, min_value, Op::kNe));
      XLS_ASSIGN_OR_RETURN(
          equivalent,
          n->function_base()->MakeNode<NaryOp>(
              n->loc(),
              std::vector<Node*>{reversed_cmp, lhs_not_min, rhs_not_min},
              Op::kXor));
    }

    XLS_RETURN_IF_ERROR(n->ReplaceUsesWith(equivalent));
    return true;
  }

  // Pattern: Signed comparison of negation to literal
  //   eq(-expr, K)  =>  eq(expr, -K)
  //   ne(-expr, K)  =>  ne(expr, -K)
  //    (-expr < K)  =>  (expr > -K) XOR (expr != MIN) XOR (K != MIN)
  //    (-expr > K)  =>  (expr < -K) XOR (expr != MIN) XOR (K != MIN)
  //   (-expr <= K)  =>  (expr >= -K) XOR (expr != MIN) XOR (K != MIN)
  //   (-expr >= K)  =>  (expr <= -K) XOR (expr != MIN) XOR (K != MIN)
  //
  // Canonicalization puts the literal on the right for comparisons.
  if (IsBitsCompare(n) && IsSignedCompare(n) &&
      n->operand(0)->op() == Op::kNeg && n->operand(1)->Is<Literal>() &&
      is_removable_negate(n->operand(0))) {
    XLS_VLOG(2) << "FOUND: Signed comparison of negation to literal";
    Node* expr = n->operand(0)->operand(0);
    Literal* k = n->operand(1)->As<Literal>();

    Bits neg_k_bits = bits_ops::Negate(k->value().bits());
    XLS_ASSIGN_OR_RETURN(Literal * neg_k, n->function_base()->MakeNode<Literal>(
                                              k->loc(), Value(neg_k_bits)));

    Node* equivalent;
    XLS_ASSIGN_OR_RETURN(Op reversed_op, ReverseComparisonOp(n->op()));
    XLS_ASSIGN_OR_RETURN(Node * reversed_cmp,
                         n->function_base()->MakeNode<CompareOp>(
                             n->loc(), expr, neg_k, reversed_op));
    if (n->op() == Op::kEq || n->op() == Op::kNe) {
      equivalent = reversed_cmp;
    } else {
      XLS_RET_CHECK_EQ(n->operand(0)->BitCountOrDie(), k->BitCountOrDie());
      const int64_t bit_count = n->operand(0)->BitCountOrDie();
      InlineBitmap hi_bit(bit_count);
      hi_bit.Set(bit_count - 1);
      Bits min_value_bits = Bits::FromBitmap(std::move(hi_bit));

      Node* x;  // x := (expr != MIN) XOR (K != MIN)...
                // but since k is known at compile time, we fold the answer in.
      XLS_ASSIGN_OR_RETURN(Literal * min_value,
                           n->function_base()->MakeNode<Literal>(
                               n->loc(), Value(min_value_bits)));
      if (k->value().bits() != min_value_bits) {
        XLS_ASSIGN_OR_RETURN(x, n->function_base()->MakeNode<CompareOp>(
                                    n->loc(), expr, min_value, Op::kEq));
      } else {
        XLS_ASSIGN_OR_RETURN(x, n->function_base()->MakeNode<CompareOp>(
                                    n->loc(), expr, min_value, Op::kNe));
      }
      XLS_ASSIGN_OR_RETURN(
          equivalent,
          n->function_base()->MakeNode<NaryOp>(
              n->loc(), std::vector<Node*>{reversed_cmp, x}, Op::kXor));
    }

    XLS_RETURN_IF_ERROR(n->ReplaceUsesWith(equivalent));
    return true;
  }

  // Patterns (where x is a bits[1] type):
  //   eq(x, 1) => x
  //   eq(x, 0) => not(x)
  //
  // Because eq is commutative, we can rely on the literal being on the right
  // because of canonicalization.
  if (n->op() == Op::kEq && n->operand(0)->GetType()->IsBits() &&
      n->operand(0)->BitCountOrDie() == 1 && n->operand(1)->Is<Literal>()) {
    if (IsLiteralZero(n->operand(1))) {
      XLS_RETURN_IF_ERROR(
          n->ReplaceUsesWithNew<UnOp>(n->operand(0), Op::kNot).status());
      return true;
    }
    XLS_VLOG(2) << "FOUND: eq comparison with bits[1]:0 or bits[1]:1";
    XLS_RET_CHECK(IsLiteralUnsignedOne(n->operand(1)));
    XLS_RETURN_IF_ERROR(n->ReplaceUsesWith(n->operand(0)));
    return true;
  }

  // Shift amounts from the front-end are often unnecessarily zero
  // extended. Strip the zero-extension (canonicalized to concat with zero):
  //
  //   a << {0, b} => a << b
  if (n->op() == Op::kShll || n->op() == Op::kShrl || n->op() == Op::kShra) {
    if (n->operand(1)->Is<Concat>()) {
      Concat* concat = n->operand(1)->As<Concat>();
      if (IsLiteralZero(concat->operand(0))) {
        Node* new_shift_amount;
        if (concat->operand_count() == 1) {
          new_shift_amount = concat->operand(0);
        } else if (concat->operand_count() == 2) {
          new_shift_amount = concat->operand(1);
        } else {
          XLS_ASSIGN_OR_RETURN(
              new_shift_amount,
              n->function_base()->MakeNode<Concat>(
                  concat->loc(), concat->operands().subspan(1)));
        }
        XLS_VLOG(2) << "FOUND: Removal of zext of shift amount";
        XLS_RETURN_IF_ERROR(n->ReplaceOperandNumber(1, new_shift_amount,
                                                    /*type_must_match=*/false));
        return true;
      }
    }
  }
  // This also applies to decode, though we may need to zero-extend the result
  // if the argument gets too small.
  //
  //   decode({0, b}) => zero_ext(decode(b))
  if (n->op() == Op::kDecode && n->operand(0)->Is<Concat>() &&
      IsLiteralZero(n->operand(0)->As<Concat>()->operand(0))) {
    Concat* concat = n->operand(0)->As<Concat>();
    Node* new_index;
    if (concat->operand_count() == 1) {
      new_index = concat->operand(0);
    } else if (concat->operand_count() == 2) {
      new_index = concat->operand(1);
    } else {
      XLS_ASSIGN_OR_RETURN(new_index,
                           n->function_base()->MakeNode<Concat>(
                               concat->loc(), concat->operands().subspan(1)));
    }
    XLS_VLOG(2) << "FOUND: Removal of zext of decode index";

    int64_t n_width = n->BitCountOrDie();
    int64_t operand_width = new_index->BitCountOrDie();

    int64_t decode_width = n_width;
    if (operand_width < 63) {
      // We can't decode to something wider than 2**(n_bit_count) bits...
      // so we decode to exactly 2**(n_bit_count) bits, and resize after.
      decode_width = std::min(int64_t{1} << operand_width, n_width);
    }
    XLS_ASSIGN_OR_RETURN(Node * decoded,
                         n->function_base()->MakeNode<Decode>(
                             n->loc(), new_index, decode_width));
    XLS_ASSIGN_OR_RETURN(
        decoded, NarrowOrExtend(decoded, /*n_is_signed=*/false, n_width));
    XLS_RETURN_IF_ERROR(n->ReplaceUsesWith(decoded));
    return true;
  }

  // Guards to prevent overshifting can be removed:
  //
  //   shift(x, clamp(amt, LIMIT))  =>  shift(x, amt) if LIMIT >= width(x)
  //
  // Where clamp(amt, LIMIT) is Sel(UGt(x, LIMIT), [x, LIMIT]).
  // This is legal because the following identity holds:
  //
  //   shift(x, K) = shift(x, width(x)) for all K >= width(x).
  //
  // This transformation can be performed for any value of LIMIT greater than
  // or equal to width(x).
  if (n->op() == Op::kShll || n->op() == Op::kShrl || n->op() == Op::kShra) {
    std::optional<ClampExpr> clamp_expr = MatchClampUpperLimit(n->operand(1));
    if (clamp_expr.has_value() &&
        bits_ops::UGreaterThanOrEqual(clamp_expr->upper_limit,
                                      n->BitCountOrDie())) {
      XLS_VLOG(2) << "FOUND: Removal of unnecessary shift guard";
      XLS_RETURN_IF_ERROR(n->ReplaceOperandNumber(1, clamp_expr->node));
      return true;
    }
  }
  // This also applies to decode(clamp(amt, LIMIT)), since decode(x) == 1 << x.
  if (n->op() == Op::kDecode) {
    std::optional<ClampExpr> clamp_expr = MatchClampUpperLimit(n->operand(0));
    if (clamp_expr.has_value() &&
        bits_ops::UGreaterThanOrEqual(clamp_expr->upper_limit,
                                      n->BitCountOrDie())) {
      XLS_VLOG(2) << "FOUND: Removal of unnecessary decode guard";
      XLS_RETURN_IF_ERROR(n->ReplaceOperandNumber(0, clamp_expr->node));
      return true;
    }
  }

  // If either x or y is zero width:
  //   [US]Mul(x, y) => 0
  //   [US]Mulp(x, y) => 0
  // This can arise due to narrowing of multiplies.
  if (n->op() == Op::kUMul || n->op() == Op::kSMul || n->op() == Op::kUMulp ||
      n->op() == Op::kSMulp) {
    for (Node* operand : n->operands()) {
      if (operand->BitCountOrDie() == 0) {
        XLS_VLOG(2) << "FOUND: replace " << OpToString(n->op())
                    << "(bits[0], ...) with 0";
        XLS_RETURN_IF_ERROR(
            n->ReplaceUsesWithNew<Literal>(ZeroOfType(n->GetType())).status());
        return true;
      }
    }
  }

  // eq(x, x) => 1
  // ne(x, x) => 0
  if (n->op() == Op::kEq && n->operand(0) == n->operand(1)) {
    XLS_RETURN_IF_ERROR(
        n->ReplaceUsesWithNew<Literal>(Value(UBits(1, 1))).status());
    return true;
  }
  if (n->op() == Op::kNe && n->operand(0) == n->operand(1)) {
    XLS_RETURN_IF_ERROR(
        n->ReplaceUsesWithNew<Literal>(Value(UBits(0, 1))).status());
    return true;
  }

  // If x and y are zero-width values:
  //    eq(x, y) => 1
  //    ne(x, y) => 0
  if (n->op() == Op::kEq && n->operand(0)->GetType()->GetFlatBitCount() == 0) {
    XLS_RETURN_IF_ERROR(
        n->ReplaceUsesWithNew<Literal>(Value(UBits(1, 1))).status());
    return true;
  }
  if (n->op() == Op::kNe && n->operand(0)->GetType()->GetFlatBitCount() == 0) {
    XLS_RETURN_IF_ERROR(
        n->ReplaceUsesWithNew<Literal>(Value(UBits(0, 1))).status());
    return true;
  }

  // Slt(x, 0) -> msb(x)
  // SGe(x, 0) -> not(msb(x))
  //
  // Canonicalization puts the literal on the right for comparisons.
  //
  if (NarrowingEnabled(opt_level) && IsBitsCompare(n) &&
      IsLiteralZero(n->operand(1))) {
    if (n->op() == Op::kSLt) {
      XLS_VLOG(2) << "FOUND: SLt(x, 0)";
      XLS_RETURN_IF_ERROR(n->ReplaceUsesWithNew<BitSlice>(
                               n->operand(0),
                               /*start=*/n->operand(0)->BitCountOrDie() - 1,
                               /*width=*/1)
                              .status());
      return true;
    }
    if (n->op() == Op::kSGe) {
      XLS_VLOG(2) << "FOUND: SGe(x, 0)";
      XLS_ASSIGN_OR_RETURN(
          Node * sign_bit,
          n->function_base()->MakeNode<BitSlice>(
              n->loc(), n->operand(0),
              /*start=*/n->operand(0)->BitCountOrDie() - 1, /*width=*/1));
      XLS_RETURN_IF_ERROR(
          n->ReplaceUsesWithNew<UnOp>(sign_bit, Op::kNot).status());
      return true;
    }
  }

  // Not(comparison_op(x, y)) => comparison_op_inverse(x, y)
  //
  // Only perform this if the only user of `comparison_op` is the `not` because
  // otherwise _both_ comparisons will remain in the graph which is a
  // de-optimization.
  if (n->op() == Op::kNot && OpIsCompare(n->operand(0)->op()) &&
      HasSingleUse(n->operand(0))) {
    XLS_VLOG(2) << "FOUND: Not(CompareOp(x, y))";
    XLS_RETURN_IF_ERROR(
        n->ReplaceUsesWithNew<CompareOp>(n->operand(0)->operand(0),
                                         n->operand(0)->operand(1),
                                         CompareOpInverse(n->operand(0)->op()))
            .status());
    return true;
  }

  // An unsigned comparison against a literal mask of LSBs (e.g., 0b0001111)
  // can be simplified:
  //
  //    x < 0b0001111  =>  or_reduce(msb_slice(x)) NOR
  //    and_reduce(lsb_slice(x)) x > 0b0001111  =>  or_reduce(msb_slice(x))
  //
  //   x <= 0b0001111  =>  nor_reduce(msb_slice(x))
  //   x >= 0b0001111  =>  or_reduce(msb_slice(x)) OR and_reduce(lsb_slice(x))
  int64_t leading_zeros, trailing_ones;
  if (NarrowingEnabled(opt_level) && IsBitsCompare(n) &&
      IsLiteralMask(n->operand(1), &leading_zeros, &trailing_ones)) {
    XLS_VLOG(2) << "Found comparison to literal mask; leading zeros: "
                << leading_zeros << " trailing ones: " << trailing_ones
                << " :: " << n;
    switch (n->op()) {
      case Op::kULt: {
        XLS_ASSIGN_OR_RETURN(
            Node * or_red,
            OrReduceLeading(n->operand(0), leading_zeros, n->loc()));
        XLS_ASSIGN_OR_RETURN(
            Node * and_trail,
            AndReduceTrailing(n->operand(0), trailing_ones, n->loc()));
        std::vector<Node*> args = {or_red, and_trail};
        XLS_RETURN_IF_ERROR(
            n->ReplaceUsesWithNew<NaryOp>(args, Op::kNor).status());
        return true;
      }
      case Op::kUGt: {
        XLS_ASSIGN_OR_RETURN(
            Node * or_red,
            OrReduceLeading(n->operand(0), leading_zeros, n->loc()));
        XLS_RETURN_IF_ERROR(n->ReplaceUsesWith(or_red));
        return true;
      }
      case Op::kULe: {
        XLS_ASSIGN_OR_RETURN(
            Node * nor_red,
            NorReduceLeading(n->operand(0), leading_zeros, n->loc()));
        XLS_RETURN_IF_ERROR(n->ReplaceUsesWith(nor_red));
        return true;
      }
      case Op::kUGe: {
        XLS_ASSIGN_OR_RETURN(
            Node * or_red,
            OrReduceLeading(n->operand(0), leading_zeros, n->loc()));
        XLS_ASSIGN_OR_RETURN(
            Node * and_trail,
            AndReduceTrailing(n->operand(0), trailing_ones, n->loc()));
        std::vector<Node*> args = {or_red, and_trail};
        XLS_RETURN_IF_ERROR(
            n->ReplaceUsesWithNew<NaryOp>(args, Op::kOr).status());
        return true;
      }
      default:
        break;
    }
  }

  // A shift-left of a power of two can be replaced by a decode operation and a
  // shift-left by a literal, since
  //   (2**K) << N == (1 << K) << N
  //               == (1 << N) << K
  //               == decode(N) << K.
  //
  // Another pass will simplify the shift-left by a literal to a concat & slice
  // (see above).
  if (n->op() == Op::kShll && n->operand(0)->Is<Literal>()) {
    Literal* lhs_literal = n->operand(0)->As<Literal>();
    const Bits& lhs = lhs_literal->value().bits();
    if (lhs.IsPowerOfTwo()) {
      XLS_VLOG(2) << "FOUND: shift left of power of two";
      int64_t k = lhs.CountTrailingZeros();
      int64_t n_bit_count = n->operand(1)->BitCountOrDie();
      int64_t decode_width = lhs.bit_count();
      if (n_bit_count < 63) {
        // We can't decode to something wider than 2**(n_bit_count) bits...
        // so we decode to exactly 2**(n_bit_count) bits, and resize after.
        decode_width = std::min(int64_t{1} << n_bit_count, decode_width);
      }
      XLS_ASSIGN_OR_RETURN(Node * decoded,
                           n->function_base()->MakeNode<Decode>(
                               n->loc(), n->operand(1), decode_width));
      XLS_ASSIGN_OR_RETURN(
          decoded,
          NarrowOrExtend(decoded, /*n_is_signed=*/false, lhs.bit_count()));
      if (k > 0) {
        XLS_ASSIGN_OR_RETURN(Literal * shift,
                             n->function_base()->MakeNode<Literal>(
                                 n->loc(), Value(UBits(k, 64))));
        XLS_ASSIGN_OR_RETURN(decoded, n->function_base()->MakeNode<BinOp>(
                                          n->loc(), decoded, shift, Op::kShll));
      }
      XLS_RETURN_IF_ERROR(n->ReplaceUsesWith(decoded));
      return true;
    }
  }

  XLS_ASSIGN_OR_RETURN(bool injective_matched, MatchComparisonOfInjectiveOp(n));
  if (injective_matched) {
    return true;
  }

  return false;
}

}  // namespace

absl::StatusOr<bool> ArithSimplificationPass::RunOnFunctionBaseInternal(
    FunctionBase* f, const OptimizationPassOptions& options,
    PassResults* results) const {
  return TransformNodesToFixedPoint(
      f, [this](Node* n) { return MatchArithPatterns(opt_level_, n); });
}

REGISTER_OPT_PASS(ArithSimplificationPass, pass_config::kOptLevel);

}  // namespace xls
