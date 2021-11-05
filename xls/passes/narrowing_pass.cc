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

#include "xls/passes/narrowing_pass.h"

#include "absl/status/statusor.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/node_iterator.h"
#include "xls/ir/node_util.h"
#include "xls/ir/op.h"
#include "xls/ir/ternary.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/range_query_engine.h"
#include "xls/passes/ternary_query_engine.h"
#include "xls/passes/union_query_engine.h"

namespace xls {

namespace {

// Return the number of leading known zeros in the given nodes values.
int64_t CountLeadingKnownZeros(Node* node, const QueryEngine& query_engine) {
  int64_t leading_zeros = 0;
  for (int64_t i = node->BitCountOrDie() - 1; i >= 0; --i) {
    if (!query_engine.IsZero(BitLocation{node, i})) {
      break;
    }
    ++leading_zeros;
  }
  return leading_zeros;
}

// Return the number of leading known ones in the given nodes values.
int64_t CountLeadingKnownOnes(Node* node, const QueryEngine& query_engine) {
  int64_t leading_ones = 0;
  for (int64_t i = node->BitCountOrDie() - 1; i >= 0; --i) {
    if (!query_engine.IsOne(BitLocation{node, i})) {
      break;
    }
    ++leading_ones;
  }
  return leading_ones;
}

// Try to narrow the operands of comparison operations. Returns true if the
// given compare operation was narrowed.
absl::StatusOr<bool> MaybeNarrowCompare(CompareOp* compare,
                                        const QueryEngine& query_engine) {
  // Returns the number of consecutive leading/trailing bits that are known to
  // be equal between the LHS and RHS of the given compare operation.
  auto matched_leading_operand_bits = [&](CompareOp* c) -> int64_t {
    int64_t bit_count = c->operand(0)->BitCountOrDie();
    for (int64_t i = 0; i < bit_count; ++i) {
      int64_t bit_index = bit_count - i - 1;
      if (!query_engine.KnownEquals(BitLocation{c->operand(0), bit_index},
                                    BitLocation{c->operand(1), bit_index})) {
        return i;
      }
    }
    return bit_count;
  };
  auto matched_trailing_operand_bits = [&](CompareOp* c) -> int64_t {
    int64_t bit_count = c->operand(0)->BitCountOrDie();
    for (int64_t i = 0; i < bit_count; ++i) {
      if (!query_engine.KnownEquals(BitLocation{c->operand(0), i},
                                    BitLocation{c->operand(1), i})) {
        return i;
      }
    }
    return bit_count;
  };

  // Narrow the operands of the compare to the given bit count. Replace the
  // given comparison operation with the new narrower compare operation.
  auto narrow_compare_operands = [](CompareOp* c, int64_t start,
                                    int64_t bit_count) -> absl::Status {
    XLS_ASSIGN_OR_RETURN(Node * narrowed_lhs,
                         c->function_base()->MakeNode<BitSlice>(
                             c->loc(), c->operand(0), start, bit_count));
    XLS_ASSIGN_OR_RETURN(Node * narrowed_rhs,
                         c->function_base()->MakeNode<BitSlice>(
                             c->loc(), c->operand(1), start, bit_count));
    return c->ReplaceUsesWithNew<CompareOp>(narrowed_lhs, narrowed_rhs, c->op())
        .status();
  };

  int64_t operand_width = compare->operand(0)->BitCountOrDie();

  // Matched leading and trailing bits of operands for unsigned comparisons (and
  // Eq and Ne) can be stripped away. For example:
  //
  //  UGt(0110_0XXX_0011, 0110_0YYY_0011) == UGt(XXX, YYY)
  //
  // Skip this optimization if all bits match because the logic needs to be
  // special cased for this, and that case is handled via other optimization
  // passes.
  int64_t matched_leading_bits = matched_leading_operand_bits(compare);
  int64_t matched_trailing_bits = matched_trailing_operand_bits(compare);
  bool all_bits_match =
      matched_leading_bits == compare->operand(0)->BitCountOrDie();
  if ((IsUnsignedCompare(compare) || compare->op() == Op::kEq ||
       compare->op() == Op::kNe) &&
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
  int64_t common_leading_ones_or_zeros =
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
  if (IsSignedCompare(compare) && compare->operand(0)->op() == Op::kSignExt &&
      compare->operand(1)->op() == Op::kSignExt) {
    int64_t max_unextended_width =
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
absl::StatusOr<bool> MaybeNarrowShiftAmount(Node* shift,
                                            const QueryEngine& query_engine) {
  XLS_RET_CHECK(shift->op() == Op::kShll || shift->op() == Op::kShrl ||
                shift->op() == Op::kShra);
  int64_t leading_zeros =
      CountLeadingKnownZeros(shift->operand(1), query_engine);
  if (leading_zeros == shift->operand(1)->BitCountOrDie()) {
    // Shift amount is zero. Replace with (slice of) input operand of shift.
    if (shift->BitCountOrDie() == shift->operand(0)->BitCountOrDie()) {
      XLS_RETURN_IF_ERROR(shift->ReplaceUsesWith(shift->operand(0)));
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
        shift->function_base()->MakeNode<BitSlice>(
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
absl::StatusOr<bool> MaybeNarrowArrayIndex(ArrayIndex* array_index,
                                           const QueryEngine& query_engine) {
  bool changed = false;
  std::vector<Node*> new_indices;
  for (int64_t i = 0; i < array_index->indices().size(); ++i) {
    Node* index = array_index->indices()[i];

    // TODO(b/148457283): Unconditionally narrow the width of the index to the
    // minimum number of bits require to index the entire array.
    if (index->Is<Literal>()) {
      continue;
    }

    int64_t index_width = index->BitCountOrDie();
    int64_t leading_zeros = CountLeadingKnownZeros(index, query_engine);
    if (leading_zeros == index_width) {
      XLS_ASSIGN_OR_RETURN(
          Node * zero, array_index->function_base()->MakeNode<Literal>(
                           array_index->loc(), Value(UBits(0, index_width))));
      new_indices.push_back(zero);
      changed = true;
      continue;
    }
    if (leading_zeros > 0) {
      XLS_ASSIGN_OR_RETURN(Node * narrowed_index,
                           array_index->function_base()->MakeNode<BitSlice>(
                               array_index->loc(), index, /*start=*/0,
                               /*width=*/index_width - leading_zeros));
      new_indices.push_back(narrowed_index);
      changed = true;
      continue;
    }
    new_indices.push_back(index);
  }
  if (changed) {
    XLS_RETURN_IF_ERROR(
        array_index
            ->ReplaceUsesWithNew<ArrayIndex>(array_index->array(), new_indices)
            .status());
    return true;
  }
  return false;
}

// Try to narrow an add with known bits.
absl::StatusOr<bool> MaybeNarrowAdd(Node* add,
                                    const QueryEngine& query_engine) {
  XLS_VLOG(3) << "Trying to narrow add: " << add->ToString();

  XLS_RET_CHECK_EQ(add->op(), Op::kAdd);

  Node* lhs = add->operand(0);
  Node* rhs = add->operand(1);
  const int64_t bit_count = add->BitCountOrDie();
  if (lhs->BitCountOrDie() != rhs->BitCountOrDie()) {
    return false;
  }

  int64_t common_leading_zeros =
      std::min(CountLeadingKnownZeros(lhs, query_engine),
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
    int64_t narrowed_bit_count = bit_count - common_leading_zeros + 1;
    XLS_ASSIGN_OR_RETURN(
        Node * narrowed_lhs,
        lhs->function_base()->MakeNode<BitSlice>(lhs->loc(), lhs, /*start=*/0,
                                                 /*width=*/narrowed_bit_count));
    XLS_ASSIGN_OR_RETURN(
        Node * narrowed_rhs,
        rhs->function_base()->MakeNode<BitSlice>(rhs->loc(), rhs, /*start=*/0,
                                                 /*width=*/narrowed_bit_count));
    XLS_ASSIGN_OR_RETURN(Node * narrowed_add,
                         add->function_base()->MakeNode<BinOp>(
                             add->loc(), narrowed_lhs, narrowed_rhs, Op::kAdd));
    XLS_RETURN_IF_ERROR(
        add->ReplaceUsesWithNew<ExtendOp>(narrowed_add, bit_count, Op::kZeroExt)
            .status());
    return true;
  }

  return false;
}

// Try to narrow the operands and/or the result of a multiply.
absl::StatusOr<bool> MaybeNarrowMultiply(ArithOp* mul,
                                         const QueryEngine& query_engine) {
  XLS_VLOG(3) << "Trying to narrow multiply: " << mul->ToString();

  XLS_RET_CHECK(mul->op() == Op::kSMul || mul->op() == Op::kUMul);

  Node* lhs = mul->operand(0);
  Node* rhs = mul->operand(1);
  const int64_t result_bit_count = mul->BitCountOrDie();
  const int64_t lhs_bit_count = lhs->BitCountOrDie();
  const int64_t rhs_bit_count = rhs->BitCountOrDie();
  XLS_VLOG(3) << absl::StreamFormat(
      "  result_bit_count = %d, lhs_bit_count = %d, rhs_bit_count = %d",
      result_bit_count, lhs_bit_count, rhs_bit_count);

  // Return the given node sign-extended (if 'mul' is Op::kSMul) or
  // zero-extended (if 'mul' is Op::kUMul) to the given bit count. If the node
  // is already of the given width, then the node is returned.
  auto maybe_extend = [&](Node* node,
                          int64_t bit_count) -> absl::StatusOr<Node*> {
    XLS_RET_CHECK(node->BitCountOrDie() <= bit_count);
    if (node->BitCountOrDie() == bit_count) {
      return node;
    }
    return node->function_base()->MakeNode<ExtendOp>(
        node->loc(), node,
        /*new_bit_count=*/bit_count,
        /*op=*/mul->op() == Op::kSMul ? Op::kSignExt : Op::kZeroExt);
  };

  // Return the given node narrowed to the given bit count. If the node
  // is already of the given width, then the node is returned.
  auto maybe_narrow = [&](Node* node,
                          int64_t bit_count) -> absl::StatusOr<Node*> {
    XLS_RET_CHECK(node->BitCountOrDie() >= bit_count);
    if (node->BitCountOrDie() == bit_count) {
      return node;
    }
    return node->function_base()->MakeNode<BitSlice>(node->loc(), node,
                                                     /*start=*/0,
                                                     /*width=*/bit_count);
  };

  // The result can be unconditionally narrowed to the sum of the operand
  // widths, then zero/sign extended.
  if (result_bit_count > lhs_bit_count + rhs_bit_count) {
    XLS_VLOG(3) << "Result is wider than sum of operands. Narrowing multiply.";
    XLS_ASSIGN_OR_RETURN(
        Node * narrowed_mul,
        mul->function_base()->MakeNode<ArithOp>(
            mul->loc(), lhs, rhs,
            /*width=*/lhs_bit_count + rhs_bit_count, mul->op()));
    XLS_ASSIGN_OR_RETURN(Node * replacement,
                         maybe_extend(narrowed_mul, result_bit_count));
    XLS_RETURN_IF_ERROR(mul->ReplaceUsesWith(replacement));
    return true;
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
  if (mul->op() == Op::kUMul || is_sign_agnostic) {
    bool operand_narrowed = false;
    auto maybe_narrow_operand = [&](Node* operand) -> absl::StatusOr<Node*> {
      int64_t leading_zeros = CountLeadingKnownZeros(operand, query_engine);
      if (leading_zeros == 0) {
        return operand;
      }
      operand_narrowed = true;
      return mul->function_base()->MakeNode<BitSlice>(
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
                                                           Op::kUMul)
                              .status());
      return true;
    }
  }

  // Sign-extended operands of signed multiplies can be narrowed by replacing
  // the operand of the multiply with the value before sign-extension.
  if (mul->op() == Op::kSMul || is_sign_agnostic) {
    bool operand_narrowed = false;
    auto maybe_narrow_operand = [&](Node* operand) -> absl::StatusOr<Node*> {
      if (operand->op() == Op::kSignExt) {
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
                                                           Op::kSMul)
                              .status());
      return true;
    }
  }

  // TODO(meheff): If either lhs or rhs has trailing zeros, the multiply can be
  // narrowed and the result concatenated with trailing zeros.

  return false;
}

}  // namespace

void RangeAnalysisLog(FunctionBase* f,
                      const TernaryQueryEngine& ternary_query_engine,
                      const RangeQueryEngine& range_query_engine) {
  int64_t bits_saved = 0;
  absl::flat_hash_map<Node*, absl::flat_hash_set<Node*>> parents_map;

  for (Node* node : f->nodes()) {
    for (Node* operand : node->operands()) {
      parents_map[operand].insert(node);
    }
  }

  for (Node* node : f->nodes()) {
    if (node->GetType()->IsBits()) {
      IntervalSet intervals =
          range_query_engine.GetIntervalSetTree(node).Get({});
      int64_t current_size = node->BitCountOrDie();
      Bits compressed_total(current_size + 1);
      for (const Interval& interval : intervals.Intervals()) {
        compressed_total = bits_ops::Add(compressed_total, interval.SizeBits());
      }
      int64_t compressed_size =
          bits_ops::DropLeadingZeroes(compressed_total).bit_count();
      int64_t hull_size =
          bits_ops::DropLeadingZeroes(intervals.ConvexHull().value().SizeBits())
              .bit_count();
      if (node->Is<Literal>() && (compressed_size < current_size)) {
        std::vector<std::string> parents;
        for (Node* parent : parents_map[node]) {
          parents.push_back(parent->ToString());
        }
        XLS_VLOG(3) << "narrowing_pass: " << OpToString(node->op())
                    << " shrinkable: "
                    << "current size = " << current_size << "; "
                    << "compressed size = " << compressed_size << "; "
                    << "convex hull size = " << hull_size << "; "
                    << "parents = [" << absl::StrJoin(parents, ", ") << "]\n";
      }

      bool inputs_all_maximal = true;
      for (Node* operand : node->operands()) {
        if (!operand->GetType()->IsBits()) {
          break;
        }
        IntervalSet intervals =
            range_query_engine.GetIntervalSetTree(operand).Get({});
        intervals.Normalize();
        if (!intervals.IsMaximal()) {
          inputs_all_maximal = false;
          break;
        }
      }
      if (!inputs_all_maximal &&
          range_query_engine.GetIntervalSetTree(node).Get({}).IsMaximal()) {
        XLS_VLOG(3) << "narrowing_pass: range analysis lost precision for "
                    << node << "\n";
      }
    }

    if (ternary_query_engine.IsTracked(node) &&
        range_query_engine.IsTracked(node)) {
      TernaryVector ternary_result = ternary_ops::FromKnownBits(
          ternary_query_engine.GetKnownBits(node),
          ternary_query_engine.GetKnownBitsValues(node));
      TernaryVector range_result = ternary_ops::FromKnownBits(
          range_query_engine.GetKnownBits(node),
          range_query_engine.GetKnownBitsValues(node));
      absl::optional<TernaryVector> difference =
          ternary_ops::Difference(range_result, ternary_result);
      XLS_CHECK(difference.has_value())
          << "Inconsistency detected in node: " << node->GetName();
      bits_saved += ternary_ops::NumberOfKnownBits(difference.value());
    }
  }

  if (bits_saved != 0) {
    XLS_VLOG(3) << "narrowing_pass: range analysis saved " << bits_saved
                << " bits in " << f << "\n";
  }
}

absl::StatusOr<bool> NarrowingPass::RunOnFunctionBaseInternal(
    FunctionBase* f, const PassOptions& options, PassResults* results) const {
  std::unique_ptr<TernaryQueryEngine> ternary_query_engine =
      std::make_unique<TernaryQueryEngine>();
  std::unique_ptr<RangeQueryEngine> range_query_engine =
      std::make_unique<RangeQueryEngine>();

  if (XLS_VLOG_IS_ON(3)) {
    RangeAnalysisLog(f, *ternary_query_engine, *range_query_engine);
  }

  std::vector<std::unique_ptr<QueryEngine>> engines;
  engines.push_back(std::move(ternary_query_engine));
  engines.push_back(std::move(range_query_engine));
  UnionQueryEngine query_engine(std::move(engines));
  XLS_RETURN_IF_ERROR(query_engine.Populate(f).status());

  bool modified = false;
  for (Node* node : TopoSort(f)) {
    // Narrow the shift-amount operand of shift operations if the shift-amount
    // has leading zeros.
    bool node_modified = false;
    switch (node->op()) {
      case Op::kShll:
      case Op::kShrl:
      case Op::kShra: {
        XLS_ASSIGN_OR_RETURN(node_modified,
                             MaybeNarrowShiftAmount(node, query_engine));
        break;
      }
      case Op::kArrayIndex: {
        XLS_ASSIGN_OR_RETURN(
            node_modified,
            MaybeNarrowArrayIndex(node->As<ArrayIndex>(), query_engine));
        break;
      }
      case Op::kSMul:
      case Op::kUMul: {
        XLS_ASSIGN_OR_RETURN(
            node_modified,
            MaybeNarrowMultiply(node->As<ArithOp>(), query_engine));
        break;
      }
      case Op::kULe:
      case Op::kULt:
      case Op::kUGe:
      case Op::kUGt:
      case Op::kSLe:
      case Op::kSLt:
      case Op::kSGe:
      case Op::kSGt:
      case Op::kEq:
      case Op::kNe: {
        XLS_ASSIGN_OR_RETURN(
            node_modified,
            MaybeNarrowCompare(node->As<CompareOp>(), query_engine));
        break;
      }
      case Op::kAdd: {
        XLS_ASSIGN_OR_RETURN(node_modified, MaybeNarrowAdd(node, query_engine));
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
