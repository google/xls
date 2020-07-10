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

#include "xls/passes/narrowing_pass.h"

#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/node_iterator.h"
#include "xls/ir/node_util.h"
#include "xls/ir/op.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/ternary_query_engine.h"

namespace xls {

namespace {

// Return the number of leading known zeros in the given nodes values.
int64 CountLeadingKnownZeros(Node* node, const QueryEngine& query_engine) {
  int64 leading_zeros = 0;
  for (int64 i = node->BitCountOrDie() - 1; i >= 0; --i) {
    if (!query_engine.IsZero(BitLocation{node, i})) {
      break;
    }
    ++leading_zeros;
  }
  return leading_zeros;
}

// Return the number of leading known ones in the given nodes values.
int64 CountLeadingKnownOnes(Node* node, const QueryEngine& query_engine) {
  int64 leading_ones = 0;
  for (int64 i = node->BitCountOrDie() - 1; i >= 0; --i) {
    if (!query_engine.IsOne(BitLocation{node, i})) {
      break;
    }
    ++leading_ones;
  }
  return leading_ones;
}

// Try to narrow the operands of comparison operations. Returns true if the
// given compare operation was narrowed.
xabsl::StatusOr<bool> MaybeNarrowCompare(CompareOp* compare,
                                         const QueryEngine& query_engine) {
  // Returns the number of consecutive leading/trailing bits that are known to
  // be equal between the LHS and RHS of the given compare operation.
  auto matched_leading_operand_bits = [&](CompareOp* c) -> int64 {
    int64 bit_count = c->operand(0)->BitCountOrDie();
    for (int64 i = 0; i < bit_count; ++i) {
      int64 bit_index = bit_count - i - 1;
      if (!query_engine.KnownEquals(BitLocation{c->operand(0), bit_index},
                                    BitLocation{c->operand(1), bit_index})) {
        return i;
      }
    }
    return bit_count;
  };
  auto matched_trailing_operand_bits = [&](CompareOp* c) -> int64 {
    int64 bit_count = c->operand(0)->BitCountOrDie();
    for (int64 i = 0; i < bit_count; ++i) {
      if (!query_engine.KnownEquals(BitLocation{c->operand(0), i},
                                    BitLocation{c->operand(1), i})) {
        return i;
      }
    }
    return bit_count;
  };

  // Narrow the operands of the compare to the given bit count. Replace the
  // given comparison operation with the new narrower compare operation.
  auto narrow_compare_operands = [](CompareOp* c, int64 start,
                                    int64 bit_count) -> absl::Status {
    XLS_ASSIGN_OR_RETURN(Node * narrowed_lhs,
                         c->function()->MakeNode<BitSlice>(
                             c->loc(), c->operand(0), start, bit_count));
    XLS_ASSIGN_OR_RETURN(Node * narrowed_rhs,
                         c->function()->MakeNode<BitSlice>(
                             c->loc(), c->operand(1), start, bit_count));
    return c->ReplaceUsesWithNew<CompareOp>(narrowed_lhs, narrowed_rhs, c->op())
        .status();
  };

  int64 operand_width = compare->operand(0)->BitCountOrDie();

  // Matched leading and trailing bits of operands for unsigned comparisons (and
  // Eq and Ne) can be stripped away. For example:
  //
  //  UGt(0110_0XXX_0011, 0110_0YYY_0011) == UGt(XXX, YYY)
  //
  // Skip this optimization if all bits match because the logic needs to be
  // special cased for this, and that case is handled via other optimization
  // passes.
  int64 matched_leading_bits = matched_leading_operand_bits(compare);
  int64 matched_trailing_bits = matched_trailing_operand_bits(compare);
  bool all_bits_match =
      matched_leading_bits == compare->operand(0)->BitCountOrDie();
  if ((IsUnsignedCompare(compare) || compare->op() == OP_EQ ||
       compare->op() == OP_NE) &&
      (matched_leading_bits > 0 || matched_trailing_bits > 0) &&
      !all_bits_match) {
    XLS_RETURN_IF_ERROR(narrow_compare_operands(
        compare, /*start=*/matched_trailing_bits,
        operand_width - matched_leading_bits - matched_trailing_bits));
    return true;
  }

  // All but one of the leading known zeros (ones) on both sides of an signed
  // compare can be sliced away except. The unsliced bit must remain as the sign
  // bit.
  int64 common_leading_ones_or_zeros =
      std::min(CountLeadingKnownZeros(compare->operand(0), query_engine),
               CountLeadingKnownZeros(compare->operand(1), query_engine));
  if (common_leading_ones_or_zeros == 0) {
    common_leading_ones_or_zeros =
        std::min(CountLeadingKnownOnes(compare->operand(0), query_engine),
                 CountLeadingKnownOnes(compare->operand(1), query_engine));
  }
  if (IsSignedCompare(compare) && common_leading_ones_or_zeros > 1) {
    XLS_RETURN_IF_ERROR(narrow_compare_operands(
        compare, /*start=*/0,
        /*bit_count=*/operand_width - common_leading_ones_or_zeros + 1));
    return true;
  }

  // If both operands of a signed compare are sign-extensions we can narrow
  // the compare to wider of the two operands *before* sign_extension.
  if (IsSignedCompare(compare) && compare->operand(0)->op() == OP_SIGN_EXT &&
      compare->operand(1)->op() == OP_SIGN_EXT) {
    int64 max_unextended_width =
        std::max(compare->operand(0)->operand(0)->BitCountOrDie(),
                 compare->operand(1)->operand(0)->BitCountOrDie());
    if (max_unextended_width < operand_width) {
      XLS_RETURN_IF_ERROR(narrow_compare_operands(
          compare, /*start=*/0, /*bit_count=*/max_unextended_width));
      return true;
    }
  }
  return false;
}

// Try to narrow the shift amount of a shift node.
xabsl::StatusOr<bool> MaybeNarrowShiftAmount(Node* shift,
                                             const QueryEngine& query_engine) {
  XLS_RET_CHECK(shift->op() == OP_SHLL || shift->op() == OP_SHRL ||
                shift->op() == OP_SHRA);
  int64 leading_zeros = CountLeadingKnownZeros(shift->operand(1), query_engine);
  if (leading_zeros == shift->operand(1)->BitCountOrDie()) {
    // Shift amount is zero. Replace with (slice of) input operand of shift.
    if (shift->BitCountOrDie() == shift->operand(0)->BitCountOrDie()) {
      XLS_RETURN_IF_ERROR(shift->ReplaceUsesWith(shift->operand(0)).status());
    } else {
      // Shift instruction is narrower than its input operand. Replace with
      // slice of input.
      XLS_RET_CHECK_LE(shift->BitCountOrDie(),
                       shift->operand(0)->BitCountOrDie());
      XLS_RETURN_IF_ERROR(
          shift
              ->ReplaceUsesWithNew<BitSlice>(shift->operand(0), /*start=*/0,
                                             /*width=*/shift->BitCountOrDie())
              .status());
    }
    return true;
  } else if (leading_zeros > 0) {
    // Prune the leading zeros from the shift amount.
    XLS_ASSIGN_OR_RETURN(
        Node * narrowed_shift_amount,
        shift->function()->MakeNode<BitSlice>(
            shift->loc(), shift->operand(1), /*start=*/0,
            /*width=*/shift->operand(1)->BitCountOrDie() - leading_zeros));
    XLS_RETURN_IF_ERROR(shift
                            ->ReplaceUsesWithNew<BinOp>(shift->operand(0),
                                                        narrowed_shift_amount,
                                                        shift->op())
                            .status());
    return true;
  }
  return false;
}

// Try to narrow the index value of an array index operation.
xabsl::StatusOr<bool> MaybeNarrowArrayIndex(ArrayIndex* array_index,
                                            const QueryEngine& query_engine) {
  Node* index = array_index->operand(1);
  // TODO(b/148457283): Unconditionally narrow the width of the index to the
  // minimum number of bits require to index the entire array.
  if (index->Is<Literal>()) {
    return false;
  }
  int64 index_width = index->BitCountOrDie();
  int64 leading_zeros = CountLeadingKnownZeros(index, query_engine);
  if (leading_zeros == index_width) {
    XLS_ASSIGN_OR_RETURN(Node * zero,
                         array_index->function()->MakeNode<Literal>(
                             array_index->loc(), Value(UBits(0, index_width))));
    XLS_RETURN_IF_ERROR(array_index->ReplaceOperandNumber(1, zero));
    return true;
  } else if (leading_zeros > 0) {
    XLS_ASSIGN_OR_RETURN(Node * narrowed_index,
                         array_index->function()->MakeNode<BitSlice>(
                             array_index->loc(), index, /*start=*/0,
                             /*width=*/index_width - leading_zeros));
    XLS_RETURN_IF_ERROR(array_index
                            ->ReplaceUsesWithNew<ArrayIndex>(
                                array_index->operand(0), narrowed_index)
                            .status());
    return true;
  }
  return false;
}

// Try to narrow an add with known bits.
xabsl::StatusOr<bool> MaybeNarrowAdd(Node* add,
                                     const QueryEngine& query_engine) {
  XLS_VLOG(3) << "Trying to narrow add: " << add->ToString();

  XLS_RET_CHECK_EQ(add->op(), OP_ADD);

  Node* lhs = add->operand(0);
  Node* rhs = add->operand(1);
  const int64 bit_count = add->BitCountOrDie();
  if (lhs->BitCountOrDie() != rhs->BitCountOrDie()) {
    return false;
  }

  int64 common_leading_zeros = std::min(
      CountLeadingKnownZeros(lhs, query_engine),
      CountLeadingKnownZeros(rhs, query_engine));

  if (common_leading_zeros > 1) {
    // Narrow the add removing all but one of the known-zero leading
    // bits. Example:
    //
    //    000XXX + 0000YY => { 00, 0XXX + 00YY }
    //
    if (common_leading_zeros == bit_count) {
      // All of the bits of both operands are zero. This case is handled
      // elsewhere by replacing the operands with literal zeros.
      return false;
    }
    int64 narrowed_bit_count = bit_count - common_leading_zeros + 1;
    XLS_ASSIGN_OR_RETURN(
        Node * narrowed_lhs,
        lhs->function()->MakeNode<BitSlice>(lhs->loc(), lhs, /*start=*/0,
                                            /*width=*/narrowed_bit_count));
    XLS_ASSIGN_OR_RETURN(
        Node * narrowed_rhs,
        rhs->function()->MakeNode<BitSlice>(rhs->loc(), rhs, /*start=*/0,
                                            /*width=*/narrowed_bit_count));
    XLS_ASSIGN_OR_RETURN(Node * narrowed_add,
                         add->function()->MakeNode<BinOp>(
                             add->loc(), narrowed_lhs, narrowed_rhs, OP_ADD));
    XLS_RETURN_IF_ERROR(
        add->ReplaceUsesWithNew<ExtendOp>(narrowed_add, bit_count, OP_ZERO_EXT)
            .status());
    return true;
  }

  return false;
}

// Try to narrow the operands and/or the result of a multiply.
xabsl::StatusOr<bool> MaybeNarrowMultiply(ArithOp* mul,
                                          const QueryEngine& query_engine) {
  XLS_VLOG(3) << "Trying to narrow multiply: " << mul->ToString();

  XLS_RET_CHECK(mul->op() == OP_SMUL || mul->op() == OP_UMUL);

  Node* lhs = mul->operand(0);
  Node* rhs = mul->operand(1);
  const int64 result_bit_count = mul->BitCountOrDie();
  const int64 lhs_bit_count = lhs->BitCountOrDie();
  const int64 rhs_bit_count = rhs->BitCountOrDie();
  XLS_VLOG(3) << absl::StreamFormat(
      "  result_bit_count = %d, lhs_bit_count = %d, rhs_bit_count = %d",
      result_bit_count, lhs_bit_count, rhs_bit_count);

  // Return the given node sign-extended (if 'mul' is OP_SMUL) or
  // zero-extended (if 'mul' is OP_UMUL) to the given bit count. If the node
  // is already of the given width, then the node is returned.
  auto maybe_extend = [&](Node* node,
                          int64 bit_count) -> xabsl::StatusOr<Node*> {
    XLS_RET_CHECK(node->BitCountOrDie() <= bit_count);
    if (node->BitCountOrDie() == bit_count) {
      return node;
    }
    return node->function()->MakeNode<ExtendOp>(
        node->loc(), node,
        /*new_bit_count=*/bit_count,
        /*op=*/mul->op() == OP_SMUL ? OP_SIGN_EXT : OP_ZERO_EXT);
  };

  // Return the given node narrowed to the given bit count. If the node
  // is already of the given width, then the node is returned.
  auto maybe_narrow = [&](Node* node,
                          int64 bit_count) -> xabsl::StatusOr<Node*> {
    XLS_RET_CHECK(node->BitCountOrDie() >= bit_count);
    if (node->BitCountOrDie() == bit_count) {
      return node;
    }
    return node->function()->MakeNode<BitSlice>(node->loc(), node, /*start=*/0,
                                                /*width=*/bit_count);
  };

  // The result can be unconditionally narrowed to the sum of the operand
  // widths, then zero/sign extended.
  if (result_bit_count > lhs_bit_count + rhs_bit_count) {
    XLS_VLOG(3) << "Result is wider than sum of operands. Narrowing multiply.";
    XLS_ASSIGN_OR_RETURN(
        Node * narrowed_mul,
        mul->function()->MakeNode<ArithOp>(
            mul->loc(), lhs, rhs,
            /*width=*/lhs_bit_count + rhs_bit_count, mul->op()));
    XLS_ASSIGN_OR_RETURN(Node * replacement,
                         maybe_extend(narrowed_mul, result_bit_count));
    return mul->ReplaceUsesWith(replacement);
  }

  // The operands can be unconditionally narrowed to the result width.
  if (lhs_bit_count > result_bit_count || rhs_bit_count > result_bit_count) {
    Node* narrowed_lhs = lhs;
    Node* narrowed_rhs = rhs;
    if (lhs_bit_count > result_bit_count) {
      XLS_ASSIGN_OR_RETURN(narrowed_lhs, maybe_narrow(lhs, result_bit_count));
    }
    if (rhs_bit_count > result_bit_count) {
      XLS_ASSIGN_OR_RETURN(narrowed_rhs, maybe_narrow(rhs, result_bit_count));
    }
    XLS_RETURN_IF_ERROR(
        mul->ReplaceUsesWithNew<ArithOp>(narrowed_lhs, narrowed_rhs,
                                         result_bit_count, mul->op())
            .status());
    return true;
  }

  // A multiply where the result and both operands are the same width is the
  // same operation whether it is signed or unsigned.
  bool is_sign_agnostic =
      result_bit_count == lhs_bit_count && result_bit_count == rhs_bit_count;

  // Zero-extended operands of unsigned multiplies can be narrowed.
  if (mul->op() == OP_UMUL || is_sign_agnostic) {
    bool operand_narrowed = false;
    auto maybe_narrow_operand = [&](Node* operand) -> xabsl::StatusOr<Node*> {
      int64 leading_zeros = CountLeadingKnownZeros(operand, query_engine);
      if (leading_zeros == 0) {
        return operand;
      }
      operand_narrowed = true;
      return mul->function()->MakeNode<BitSlice>(
          mul->loc(), operand, /*start=*/0,
          /*width=*/operand->BitCountOrDie() - leading_zeros);
    };
    XLS_ASSIGN_OR_RETURN(Node * operand0,
                         maybe_narrow_operand(mul->operand(0)));
    XLS_ASSIGN_OR_RETURN(Node * operand1,
                         maybe_narrow_operand(mul->operand(1)));
    if (operand_narrowed) {
      XLS_RETURN_IF_ERROR(mul->ReplaceUsesWithNew<ArithOp>(operand0, operand1,
                                                           result_bit_count,
                                                           OP_UMUL)
                              .status());
      return true;
    }
  }

  // Sign-extended operands of signed multiplies can be narrowed by replacing
  // the operand of the multiply with the value before sign-extension.
  if (mul->op() == OP_SMUL || is_sign_agnostic) {
    bool operand_narrowed = false;
    auto maybe_narrow_operand = [&](Node* operand) -> xabsl::StatusOr<Node*> {
      if (operand->op() == OP_SIGN_EXT) {
        // Operand is a sign-extended value. Just use the value before
        // sign-extension.
        operand_narrowed = true;
        return operand->operand(0);
      }
      if (CountLeadingKnownZeros(operand, query_engine) > 1) {
        // Operand has more than one leading zero, something like:
        //    operand = 0000XXXX
        // This is equivalent to:
        //    operand = signextend(0XXXX)
        // So we can replace the operand with 0XXXX.
        operand_narrowed = true;
        return maybe_narrow(
            operand, operand->BitCountOrDie() -
                         CountLeadingKnownZeros(operand, query_engine) + 1);
      }
      return operand;
    };
    XLS_ASSIGN_OR_RETURN(Node * operand0,
                         maybe_narrow_operand(mul->operand(0)));
    XLS_ASSIGN_OR_RETURN(Node * operand1,
                         maybe_narrow_operand(mul->operand(1)));
    if (operand_narrowed) {
      XLS_RETURN_IF_ERROR(mul->ReplaceUsesWithNew<ArithOp>(operand0, operand1,
                                                           result_bit_count,
                                                           OP_SMUL)
                              .status());
      return true;
    }
  }

  // TODO(meheff): If either lhs or rhs has trailing zeros, the multiply can be
  // narrowed and the result concatenated with trailing zeros.

  return false;
}

}  // namespace

xabsl::StatusOr<bool> NarrowingPass::RunOnFunction(Function* f,
                                                   const PassOptions& options,
                                                   PassResults* results) const {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<TernaryQueryEngine> query_engine,
                       TernaryQueryEngine::Run(f));

  bool modified = false;
  for (Node* node : TopoSort(f)) {
    // Narrow the shift-amount operand of shift operations if the shift-amount
    // has leading zeros.
    bool node_modified = false;
    switch (node->op()) {
      case OP_SHLL:
      case OP_SHRL:
      case OP_SHRA: {
        XLS_ASSIGN_OR_RETURN(node_modified,
                             MaybeNarrowShiftAmount(node, *query_engine));
        break;
      }
      case OP_ARRAY_INDEX: {
        XLS_ASSIGN_OR_RETURN(
            node_modified,
            MaybeNarrowArrayIndex(node->As<ArrayIndex>(), *query_engine));
        break;
      }
      case OP_SMUL:
      case OP_UMUL: {
        XLS_ASSIGN_OR_RETURN(
            node_modified,
            MaybeNarrowMultiply(node->As<ArithOp>(), *query_engine));
        break;
      }
      case OP_ULE:
      case OP_ULT:
      case OP_UGE:
      case OP_UGT:
      case OP_SLE:
      case OP_SLT:
      case OP_SGE:
      case OP_SGT:
      case OP_EQ:
      case OP_NE: {
        XLS_ASSIGN_OR_RETURN(
            node_modified,
            MaybeNarrowCompare(node->As<CompareOp>(), *query_engine));
        break;
      }
      case OP_ADD: {
        XLS_ASSIGN_OR_RETURN(node_modified,
                             MaybeNarrowAdd(node, *query_engine));
        break;
      }
      default:
        break;
    }
    modified |= node_modified;
  }
  return modified;
}

}  // namespace xls
