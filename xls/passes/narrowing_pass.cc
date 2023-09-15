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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xls/common/logging/logging.h"
#include "xls/common/logging/vlog_is_on.h"
#include "xls/common/math_util.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/function_base.h"
#include "xls/ir/interval.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/node_iterator.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/source_location.h"
#include "xls/ir/ternary.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_helpers.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/range_query_engine.h"
#include "xls/passes/ternary_query_engine.h"
#include "xls/passes/union_query_engine.h"

namespace xls {

namespace {

using AnalysisType = NarrowingPass::AnalysisType;

// Return the given node sign-extended (if is_signed is true) or zero-extended
// (if 'is_signed' is false) to the given bit count. If the node is already of
// the given width, then the node is returned.
absl::StatusOr<Node*> MaybeExtend(Node* node, int64_t bit_count,
                                  bool is_signed) {
  XLS_RET_CHECK(node->BitCountOrDie() <= bit_count);
  if (node->BitCountOrDie() == bit_count) {
    return node;
  }
  return node->function_base()->MakeNode<ExtendOp>(
      node->loc(), node, /*new_bit_count=*/bit_count,
      /*op=*/is_signed ? Op::kSignExt : Op::kZeroExt);
}

// Return the given node narrowed to the given bit count. If the node is already
// of the given width, then the node is returned.
absl::StatusOr<Node*> MaybeNarrow(Node* node, int64_t bit_count) {
  XLS_RET_CHECK(node->BitCountOrDie() >= bit_count);
  if (node->BitCountOrDie() == bit_count) {
    return node;
  }
  return node->function_base()->MakeNode<BitSlice>(node->loc(), node,
                                                   /*start=*/0,
                                                   /*width=*/bit_count);
}

class NarrowVisitor final : public DfsVisitorWithDefault {
 public:
  explicit NarrowVisitor(const QueryEngine& engine, AnalysisType analysis,
                         const OptimizationPassOptions& options)
      : query_engine_(engine), analysis_(analysis), options_(options) {}

  absl::StatusOr<bool> MaybeReplacePreciseWithLiteral(Node* node) {
    LeafTypeTree<IntervalSet> intervals = query_engine_.GetIntervals(node);
    for (Type* leaf_type : intervals.leaf_types()) {
      if (leaf_type->IsToken()) {
        XLS_RETURN_IF_ERROR(NoChange());
        return false;
      }
    }
    for (const IntervalSet& interval_set : intervals.elements()) {
      if (!interval_set.IsPrecise()) {
        XLS_RETURN_IF_ERROR(NoChange());
        return false;
      }
    }
    LeafTypeTree<Value> value_tree =
        LeafTypeTree<Value>::Zip<IntervalSet, Type*>(
            [](const IntervalSet& interval_set, Type* type) -> Value {
              return Value(interval_set.GetPreciseValue().value());
            },
            intervals,
            LeafTypeTree<Type*>(intervals.type(), intervals.leaf_types()));
    XLS_ASSIGN_OR_RETURN(Value value, LeafTypeTreeToValue(value_tree));
    XLS_RETURN_IF_ERROR(
        node->ReplaceUsesWithNew<Literal>(value).status());
    XLS_VLOG(3) << absl::StreamFormat(
        "Range analysis found precise value for %s, replacing with literal\n",
        node->GetName());
    XLS_RETURN_IF_ERROR(Change());
    return true;
  }

  bool changed() const { return changed_; }

  absl::Status DefaultHandler(Node* node) override { return NoChange(); }
  absl::Status HandleGate(Gate* gate) override {
    // We explicitly never want anything to occur here except being replaced
    // with a constant.
    return NoChange();
  }
  absl::Status HandleEq(CompareOp* eq) override {
    return MaybeNarrowCompare(eq);
  }
  absl::Status HandleNe(CompareOp* ne) override {
    return MaybeNarrowCompare(ne);
  }
  absl::Status HandleSGe(CompareOp* ge) override {
    return MaybeNarrowCompare(ge);
  }
  absl::Status HandleSGt(CompareOp* gt) override {
    return MaybeNarrowCompare(gt);
  }
  absl::Status HandleSLe(CompareOp* le) override {
    return MaybeNarrowCompare(le);
  }
  absl::Status HandleSLt(CompareOp* lt) override {
    return MaybeNarrowCompare(lt);
  }
  absl::Status HandleUGe(CompareOp* ge) override {
    return MaybeNarrowCompare(ge);
  }
  absl::Status HandleUGt(CompareOp* gt) override {
    return MaybeNarrowCompare(gt);
  }
  absl::Status HandleULe(CompareOp* le) override {
    return MaybeNarrowCompare(le);
  }
  absl::Status HandleULt(CompareOp* lt) override {
    return MaybeNarrowCompare(lt);
  }

  // Try to narrow the operands of comparison operations. Returns true if the
  // given compare operation was narrowed.
  absl::Status MaybeNarrowCompare(CompareOp* compare) {
    if (!compare->operand(0)->GetType()->IsBits()) {
      return NoChange();
    }
    // Returns the number of consecutive leading/trailing bits that are known to
    // be equal between the LHS and RHS of the given compare operation.
    auto matched_leading_operand_bits = [&](CompareOp* c) -> int64_t {
      int64_t bit_count = c->operand(0)->BitCountOrDie();
      for (int64_t i = 0; i < bit_count; ++i) {
        int64_t bit_index = bit_count - i - 1;
        if (!query_engine_.KnownEquals(
                TreeBitLocation(c->operand(0), bit_index),
                TreeBitLocation(c->operand(1), bit_index))) {
          return i;
        }
      }
      return bit_count;
    };
    auto matched_trailing_operand_bits = [&](CompareOp* c) -> int64_t {
      int64_t bit_count = c->operand(0)->BitCountOrDie();
      for (int64_t i = 0; i < bit_count; ++i) {
        if (!query_engine_.KnownEquals(TreeBitLocation(c->operand(0), i),
                                       TreeBitLocation(c->operand(1), i))) {
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
      XLS_VLOG(3) << absl::StreamFormat(
          "Narrowing operands of comparison %s to slice [%d:%d]", c->GetName(),
          start, start + bit_count);
      return c
          ->ReplaceUsesWithNew<CompareOp>(narrowed_lhs, narrowed_rhs, c->op())
          .status();
    };

    int64_t operand_width = compare->operand(0)->BitCountOrDie();

    // Matched leading and trailing bits of operands for unsigned comparisons
    // (and Eq and Ne) can be stripped away. For example:
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
      XLS_VLOG(3) << absl::StreamFormat(
          "Leading %d bits and trailing %d bits of comparison operation %s "
          "match",
          matched_leading_bits, matched_trailing_bits, compare->GetName());
      XLS_RETURN_IF_ERROR(narrow_compare_operands(
          compare, /*start=*/matched_trailing_bits,
          operand_width - matched_leading_bits - matched_trailing_bits));
      return Change();
    }

    // All but one of the leading known zeros (ones) on both sides of an signed
    // compare can be sliced away (except the unsliced bit must remain as the
    // sign bit).
    int64_t common_leading_ones_or_zeros =
        std::min(CountLeadingKnownZeros(compare->operand(0)),
                 CountLeadingKnownZeros(compare->operand(1)));
    if (common_leading_ones_or_zeros == 0) {
      common_leading_ones_or_zeros =
          std::min(CountLeadingKnownOnes(compare->operand(0)),
                   CountLeadingKnownOnes(compare->operand(1)));
    }
    if (IsSignedCompare(compare) && common_leading_ones_or_zeros > 1) {
      XLS_RETURN_IF_ERROR(narrow_compare_operands(
          compare, /*start=*/0,
          /*bit_count=*/operand_width - common_leading_ones_or_zeros + 1));
      return Change();
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
        return Change();
      }
    }
    return NoChange();
  }

  absl::Status HandleShll(BinOp* shll) override {
    return MaybeNarrowShiftAmount(shll);
  }
  absl::Status HandleShra(BinOp* shra) override {
    return MaybeNarrowShiftAmount(shra);
  }
  absl::Status HandleShrl(BinOp* shrl) override {
    return MaybeNarrowShiftAmount(shrl);
  }

  // Narrow the shift-amount operand of shift operations if the shift-amount
  // has leading zeros.
  absl::Status MaybeNarrowShiftAmount(Node* shift) {
    XLS_RET_CHECK(shift->op() == Op::kShll || shift->op() == Op::kShrl ||
                  shift->op() == Op::kShra);
    int64_t leading_zeros = CountLeadingKnownZeros(shift->operand(1));
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
      return Change();
    }
    if (leading_zeros > 0) {
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
      return Change();
    }
    return NoChange();
  }

  absl::Status HandleAdd(BinOp* add) override {
    XLS_VLOG(3) << "Trying to narrow add: " << add->ToString();

    XLS_RET_CHECK_EQ(add->op(), Op::kAdd);

    Node* lhs = add->operand(0);
    Node* rhs = add->operand(1);
    const int64_t bit_count = add->BitCountOrDie();
    if (lhs->BitCountOrDie() != rhs->BitCountOrDie()) {
      return NoChange();
    }

    int64_t common_leading_zeros =
        std::min(CountLeadingKnownZeros(lhs), CountLeadingKnownZeros(rhs));

    if (common_leading_zeros > 1) {
      // Narrow the add removing all but one of the known-zero leading
      // bits. Example:
      //
      //    000XXX + 0000YY => { 00, 0XXX + 00YY }
      //
      if (common_leading_zeros == bit_count) {
        // All of the bits of both operands are zero. This case is handled
        // elsewhere by replacing the operands with literal zeros.
        return NoChange();
      }
      int64_t narrowed_bit_count = bit_count - common_leading_zeros + 1;
      XLS_ASSIGN_OR_RETURN(Node * narrowed_lhs,
                           lhs->function_base()->MakeNode<BitSlice>(
                               lhs->loc(), lhs, /*start=*/0,
                               /*width=*/narrowed_bit_count));
      XLS_ASSIGN_OR_RETURN(Node * narrowed_rhs,
                           rhs->function_base()->MakeNode<BitSlice>(
                               rhs->loc(), rhs, /*start=*/0,
                               /*width=*/narrowed_bit_count));
      XLS_ASSIGN_OR_RETURN(
          Node * narrowed_add,
          add->function_base()->MakeNode<BinOp>(add->loc(), narrowed_lhs,
                                                narrowed_rhs, Op::kAdd));
      XLS_RETURN_IF_ERROR(add->ReplaceUsesWithNew<ExtendOp>(
                                 narrowed_add, bit_count, Op::kZeroExt)
                              .status());
      return Change();
    }

    return NoChange();
  }

  absl::Status HandleUMul(ArithOp* umul) override {
    return MaybeNarrowMultiply(umul);
  }
  absl::Status HandleSMul(ArithOp* smul) override {
    return MaybeNarrowMultiply(smul);
  }

  // Try to narrow the operands and/or the result of a multiply.
  absl::Status MaybeNarrowMultiply(ArithOp* mul) {
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

    // The result can be unconditionally narrowed to the sum of the operand
    // widths, then zero/sign extended.
    if (result_bit_count > lhs_bit_count + rhs_bit_count) {
      XLS_VLOG(3)
          << "Result is wider than sum of operands. Narrowing multiply.";
      XLS_ASSIGN_OR_RETURN(
          Node * narrowed_mul,
          mul->function_base()->MakeNode<ArithOp>(
              mul->loc(), lhs, rhs,
              /*width=*/lhs_bit_count + rhs_bit_count, mul->op()));
      XLS_ASSIGN_OR_RETURN(Node * replacement,
                           MaybeExtend(narrowed_mul, result_bit_count,
                                       /*is_signed=*/mul->op() == Op::kSMul));
      XLS_RETURN_IF_ERROR(mul->ReplaceUsesWith(replacement));
      return Change();
    }

    // The operands can be unconditionally narrowed to the result width.
    if (lhs_bit_count > result_bit_count || rhs_bit_count > result_bit_count) {
      Node* narrowed_lhs = lhs;
      Node* narrowed_rhs = rhs;
      if (lhs_bit_count > result_bit_count) {
        XLS_ASSIGN_OR_RETURN(narrowed_lhs, MaybeNarrow(lhs, result_bit_count));
      }
      if (rhs_bit_count > result_bit_count) {
        XLS_ASSIGN_OR_RETURN(narrowed_rhs, MaybeNarrow(rhs, result_bit_count));
      }
      XLS_RETURN_IF_ERROR(
          mul->ReplaceUsesWithNew<ArithOp>(narrowed_lhs, narrowed_rhs,
                                           result_bit_count, mul->op())
              .status());
      return Change();
    }

    // A multiply where the result and both operands are the same width is the
    // same operation whether it is signed or unsigned.
    bool is_sign_agnostic =
        result_bit_count == lhs_bit_count && result_bit_count == rhs_bit_count;

    // Zero-extended operands of unsigned multiplies can be narrowed.
    if (mul->op() == Op::kUMul || is_sign_agnostic) {
      XLS_ASSIGN_OR_RETURN(std::optional<Node*> operand0,
                           MaybeNarrowUnsignedOperand(mul->operand(0)));
      XLS_ASSIGN_OR_RETURN(std::optional<Node*> operand1,
                           MaybeNarrowUnsignedOperand(mul->operand(1)));
      if (operand0.has_value() || operand1.has_value()) {
        XLS_RETURN_IF_ERROR(
            mul->ReplaceUsesWithNew<ArithOp>(operand0.value_or(mul->operand(0)),
                                             operand1.value_or(mul->operand(1)),
                                             result_bit_count, Op::kUMul)
                .status());
        return Change();
      }
    }

    // Sign-extended operands of signed multiplies can be narrowed by replacing
    // the operand of the multiply with the value before sign-extension.
    if (mul->op() == Op::kSMul || is_sign_agnostic) {
      XLS_ASSIGN_OR_RETURN(std::optional<Node*> operand0,
                           MaybeNarrowSignedOperand(mul->operand(0)));
      XLS_ASSIGN_OR_RETURN(std::optional<Node*> operand1,
                           MaybeNarrowSignedOperand(mul->operand(1)));
      if (operand0.has_value() || operand1.has_value()) {
        XLS_RETURN_IF_ERROR(
            mul->ReplaceUsesWithNew<ArithOp>(operand0.value_or(mul->operand(0)),
                                             operand1.value_or(mul->operand(1)),
                                             result_bit_count, Op::kSMul)
                .status());
        return Change();
      }
    }

    // TODO(meheff): If either lhs or rhs has trailing zeros, the multiply can
    // be narrowed and the result concatenated with trailing zeros.

    return NoChange();
  }
  absl::Status HandleArrayIndex(ArrayIndex* array_index) override {
    bool changed = false;

    if (analysis_ == AnalysisType::kRange &&
        options_.convert_array_index_to_select.has_value()) {
      int64_t threshold = options_.convert_array_index_to_select.value();
      XLS_ASSIGN_OR_RETURN(bool subpass_changed, MaybeConvertArrayIndexToSelect(
                                                     array_index, threshold));
      if (subpass_changed) {
        return Change();
      }
    }

    std::vector<Node*> new_indices;
    for (int64_t i = 0; i < array_index->indices().size(); ++i) {
      Node* index = array_index->indices()[i];

      // Determine the array type that this index element is indexing into.
      ArrayType* array_type = array_index->array()->GetType()->AsArrayOrDie();
      for (int64_t j = 0; j < i; ++j) {
        array_type = array_type->element_type()->AsArrayOrDie();
      }

      // Compute the minimum number of bits required to index the entire array.
      int64_t array_size = array_type->AsArrayOrDie()->size();
      int64_t min_index_width =
          std::max(int64_t{1}, Bits::MinBitCountUnsigned(array_size - 1));

      if (index->Is<Literal>()) {
        const Bits& bits_index = index->As<Literal>()->value().bits();
        Bits new_bits_index = bits_index;
        if (bits_ops::UGreaterThanOrEqual(bits_index, array_size)) {
          // Index is out-of-bounds. Replace with a (potentially narrower) index
          // equal to the first out-of-bounds element.
          new_bits_index =
              UBits(array_size, Bits::MinBitCountUnsigned(array_size));
        } else if (bits_index.bit_count() > min_index_width) {
          // Index is in-bounds and is wider than necessary to index the entire
          // array. Replace with a literal which is perfectly sized (width) to
          // index the whole array.
          XLS_ASSIGN_OR_RETURN(int64_t int_index, bits_index.ToUint64());
          new_bits_index = UBits(int_index, min_index_width);
        }
        Node* new_index = index;
        if (bits_index != new_bits_index) {
          XLS_ASSIGN_OR_RETURN(new_index,
                               array_index->function_base()->MakeNode<Literal>(
                                   array_index->loc(), Value(new_bits_index)));
          changed = true;
        }
        new_indices.push_back(new_index);
        continue;
      }

      int64_t index_width = index->BitCountOrDie();
      int64_t leading_zeros = CountLeadingKnownZeros(index);
      if (leading_zeros == index_width) {
        XLS_ASSIGN_OR_RETURN(
            Node * zero,
            array_index->function_base()->MakeNode<Literal>(
                array_index->loc(), Value(UBits(0, min_index_width))));
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
      XLS_RETURN_IF_ERROR(array_index
                              ->ReplaceUsesWithNew<ArrayIndex>(
                                  array_index->array(), new_indices)
                              .status());
      return Change();
    }
    return NoChange();
  }

  absl::Status HandleSMulp(PartialProductOp* mul) override {
    return MaybeNarrowPartialMultiply(mul);
  }
  absl::Status HandleUMulp(PartialProductOp* mul) override {
    return MaybeNarrowPartialMultiply(mul);
  }

  // Try to narrow the operands and/or the result of a multiply.
  absl::Status MaybeNarrowPartialMultiply(PartialProductOp* mul) {
    XLS_VLOG(3) << "Trying to narrow multiply: " << mul->ToString();

    XLS_RET_CHECK(mul->op() == Op::kSMulp || mul->op() == Op::kUMulp);

    Node* lhs = mul->operand(0);
    Node* rhs = mul->operand(1);
    const int64_t result_bit_count = mul->width();
    const int64_t lhs_bit_count = lhs->BitCountOrDie();
    const int64_t rhs_bit_count = rhs->BitCountOrDie();
    XLS_VLOG(3) << absl::StreamFormat(
        "  result_bit_count = %d, lhs_bit_count = %d, rhs_bit_count = %d",
        result_bit_count, lhs_bit_count, rhs_bit_count);

    // If both elements of the mulp tuple are immediately added, the mulp result
    // can be unconditionally narrowed to the sum of the operand widths, the add
    // replaced with a narrowed add, and the result of the addition zero/sign
    // extended.
    if (std::optional<BinOp*> add_immediately_after =
            PartialMultiplyImmediateSum(mul);
        result_bit_count > lhs_bit_count + rhs_bit_count &&
        add_immediately_after.has_value()) {
      XLS_VLOG(3)
          << "Result is wider than sum of operands. Narrowing multiply.";
      int64_t narrowed_width = lhs_bit_count + rhs_bit_count;
      XLS_ASSIGN_OR_RETURN(Node * narrowed_mul,
                           mul->function_base()->MakeNode<PartialProductOp>(
                               mul->loc(), lhs, rhs,
                               /*width=*/narrowed_width, mul->op()));
      XLS_ASSIGN_OR_RETURN(Node * product0,
                           mul->function_base()->MakeNode<TupleIndex>(
                               mul->loc(), narrowed_mul, /*index=*/0));
      XLS_ASSIGN_OR_RETURN(Node * product1,
                           mul->function_base()->MakeNode<TupleIndex>(
                               mul->loc(), narrowed_mul, /*index=*/1));
      XLS_ASSIGN_OR_RETURN(Node * sum,
                           mul->function_base()->MakeNode<BinOp>(
                               mul->loc(), product0, product1, Op::kAdd));
      XLS_ASSIGN_OR_RETURN(Node * extended_sum,
                           MaybeExtend(sum, result_bit_count,
                                       /*is_signed=*/mul->op() == Op::kSMulp));
      XLS_RETURN_IF_ERROR(
          (*add_immediately_after)->ReplaceUsesWith(extended_sum));
      return Change();
    }

    // The operands can be unconditionally narrowed to the result width.
    if (lhs_bit_count > result_bit_count || rhs_bit_count > result_bit_count) {
      Node* narrowed_lhs = lhs;
      Node* narrowed_rhs = rhs;
      if (lhs_bit_count > result_bit_count) {
        XLS_ASSIGN_OR_RETURN(narrowed_lhs, MaybeNarrow(lhs, result_bit_count));
      }
      if (rhs_bit_count > result_bit_count) {
        XLS_ASSIGN_OR_RETURN(narrowed_rhs, MaybeNarrow(rhs, result_bit_count));
      }
      XLS_RETURN_IF_ERROR(
          mul->ReplaceUsesWithNew<PartialProductOp>(narrowed_lhs, narrowed_rhs,
                                                    result_bit_count, mul->op())
              .status());
      return Change();
    }

    // A multiply where the result and both operands are the same width is the
    // same operation whether it is signed or unsigned.
    bool is_sign_agnostic =
        result_bit_count == lhs_bit_count && result_bit_count == rhs_bit_count;

    // Zero-extended operands of unsigned multiplies can be narrowed.
    if (mul->op() == Op::kUMulp || is_sign_agnostic) {
      XLS_ASSIGN_OR_RETURN(std::optional<Node*> operand0,
                           MaybeNarrowUnsignedOperand(mul->operand(0)));
      XLS_ASSIGN_OR_RETURN(std::optional<Node*> operand1,
                           MaybeNarrowUnsignedOperand(mul->operand(1)));
      if (operand0.has_value() || operand1.has_value()) {
        XLS_RETURN_IF_ERROR(mul->ReplaceUsesWithNew<PartialProductOp>(
                                   operand0.value_or(mul->operand(0)),
                                   operand1.value_or(mul->operand(1)),
                                   result_bit_count, Op::kUMulp)
                                .status());
        return Change();
      }
    }

    // Sign-extended operands of signed multiplies can be narrowed by replacing
    // the operand of the multiply with the value before sign-extension.
    if (mul->op() == Op::kSMulp || is_sign_agnostic) {
      XLS_ASSIGN_OR_RETURN(std::optional<Node*> operand0,
                           MaybeNarrowSignedOperand(mul->operand(0)));
      XLS_ASSIGN_OR_RETURN(std::optional<Node*> operand1,
                           MaybeNarrowSignedOperand(mul->operand(1)));
      if (operand0.has_value() || operand1.has_value()) {
        XLS_RETURN_IF_ERROR(mul->ReplaceUsesWithNew<PartialProductOp>(
                                   operand0.value_or(mul->operand(0)),
                                   operand1.value_or(mul->operand(1)),
                                   result_bit_count, Op::kSMulp)
                                .status());
        return Change();
      }
    }

    // TODO(meheff): If either lhs or rhs has trailing zeros, the multiply can
    // be narrowed and the result concatenated with trailing zeros.

    return NoChange();
  }

 private:
  // Return the number of leading known zeros in the given nodes values.
  int64_t CountLeadingKnownZeros(Node* node) {
    XLS_CHECK(node->GetType()->IsBits());
    int64_t leading_zeros = 0;
    for (int64_t i = node->BitCountOrDie() - 1; i >= 0; --i) {
      if (!query_engine_.IsZero(TreeBitLocation(node, i))) {
        break;
      }
      ++leading_zeros;
    }
    return leading_zeros;
  }

  // Return the number of leading known ones in the given nodes values.
  int64_t CountLeadingKnownOnes(Node* node) {
    XLS_CHECK(node->GetType()->IsBits());
    int64_t leading_ones = 0;
    for (int64_t i = node->BitCountOrDie() - 1; i >= 0; --i) {
      if (!query_engine_.IsOne(TreeBitLocation(node, i))) {
        break;
      }
      ++leading_ones;
    }
    return leading_ones;
  }

  absl::StatusOr<std::optional<Node*>> MaybeNarrowUnsignedOperand(
      Node* operand) {
    int64_t leading_zeros = CountLeadingKnownZeros(operand);
    if (leading_zeros == 0) {
      return std::nullopt;
    }
    return operand->function_base()->MakeNode<BitSlice>(
        operand->loc(), operand, /*start=*/0,
        /*width=*/operand->BitCountOrDie() - leading_zeros);
  }

  absl::StatusOr<std::optional<Node*>> MaybeNarrowSignedOperand(Node* operand) {
    if (operand->op() == Op::kSignExt) {
      // Operand is a sign-extended value. Just use the value before
      // sign-extension.
      return operand->operand(0);
    }
    if (CountLeadingKnownZeros(operand) > 1) {
      // Operand has more than one leading zero, something like:
      //    operand = 0000XXXX
      // This is equivalent to:
      //    operand = signextend(0XXXX)
      // So we can replace the operand with 0XXXX.
      return MaybeNarrow(operand, operand->BitCountOrDie() -
                                      CountLeadingKnownZeros(operand) + 1);
    }
    return std::nullopt;
  }

  // If an ArrayIndex has a small set of possible indexes (based on range
  // analysis), replace it with a small select chain.
  absl::StatusOr<bool> MaybeConvertArrayIndexToSelect(ArrayIndex* array_index,
                                                      int64_t threshold) {
    if (array_index->indices().empty()) {
      return false;
    }

    // The dimension of the multidimensional array, truncated to the number of
    // indexes we are indexing on.
    // For example, if the array has shape [5, 7, 3, 2] and we index with [i, j]
    // then this will contain [5, 7].
    std::vector<int64_t> dimension;

    // The interval set that each index lives in.
    std::vector<IntervalSet> index_intervals;

    // Populate `dimension` and `index_intervals`.
    {
      ArrayType* array_type = array_index->array()->GetType()->AsArrayOrDie();
      for (Node* index : array_index->indices()) {
        dimension.push_back(array_type->size());
        index_intervals.push_back(query_engine_.GetIntervals(index).Get({}));
        absl::StatusOr<ArrayType*> array_type_status =
            array_type->element_type()->AsArray();
        array_type = array_type_status.ok() ? *array_type_status : nullptr;
      }
    }

    // Return early to avoid generating too much code if the index space is big.
    {
      int64_t index_space_size = 1;
      for (const IntervalSet& index_interval_set : index_intervals) {
        if (index_space_size > threshold) {
          return false;
        }
        if (std::optional<int64_t> size = index_interval_set.Size()) {
          index_space_size *= (*size);
        } else {
          return false;
        }
      }

      // This means that the indexes are literals; we shouldn't run this
      // optimization in this case because we generate ArrayIndexes with
      // literals as part of this optimization, so failing to skip this would
      // result in an infinite amount of code being generated.
      if (index_space_size == 1) {
        return false;
      }

      if (index_space_size > threshold) {
        return false;
      }
    }

    // This vector contains one element per case in the ultimate OneHotSelect.
    std::vector<Node*> cases;
    // This vector contains one one-bit Node per case in the OneHotSelect, which
    // will be concatenated together to form the selector.
    std::vector<Node*> conditions;

    // Reserve the right amount of space in `cases` and `conditions`.
    {
      int64_t overall_size = 1;
      for (int64_t i : dimension) {
        overall_size *= i;
      }

      cases.reserve(overall_size);
      conditions.reserve(overall_size);
    }

    // Helpful shorthands
    FunctionBase* f = array_index->function_base();
    SourceInfo loc = array_index->loc();

    // Given a possible index (possible based on range query engine data),
    // populate `cases` and `conditions` with the relevant nodes.
    auto handle_possible_index =
        [&](const std::vector<Bits>& indexes) -> absl::Status {
      // Turn the `indexes` into a vector of literal nodes.
      std::vector<Node*> index_literals;
      index_literals.reserve(indexes.size());
      for (int64_t i = 0; i < indexes.size(); ++i) {
        XLS_ASSIGN_OR_RETURN(Node * literal,
                             f->MakeNode<Literal>(loc, Value(indexes[i])));
        index_literals.push_back(literal);
      }

      // Index into the array using `index_literals`.
      // Hopefully these `ArrayIndex`es will be fused into their indexed
      // arrays by later passes.
      XLS_ASSIGN_OR_RETURN(
          Node * element,
          f->MakeNode<ArrayIndex>(loc, array_index->array(), index_literals));

      cases.push_back(element);

      // Create a vector of `index_expr == index_literal` nodes, where
      // `index_expr` is something in `array_index->indices()` and
      // `index_literal` is from `index_literals`.
      std::vector<Node*> equality_checks;
      equality_checks.reserve(indexes.size());
      for (int64_t i = 0; i < indexes.size(); ++i) {
        XLS_ASSIGN_OR_RETURN(
            Node * equality_check,
            f->MakeNode<CompareOp>(loc, array_index->indices()[i],
                                   index_literals[i], Op::kEq));
        equality_checks.push_back(equality_check);
      }

      // AND all those equality checks together to form one condition.
      XLS_ASSIGN_OR_RETURN(Node * condition,
                           f->MakeNode<NaryOp>(loc, equality_checks, Op::kAnd));
      conditions.push_back(condition);

      return absl::OkStatus();
    };

    // Iterate over all possible indexes.
    absl::Status failure = absl::OkStatus();
    MixedRadixIterate(dimension, [&](const std::vector<int64_t>& indexes) {
      // Skip indexes that are impossible by range analysis, and convert the
      // indexes to `Bits`.

      std::vector<Bits> indexes_bits;
      indexes_bits.reserve(indexes.size());

      for (int64_t i = 0; i < indexes.size(); ++i) {
        IntervalSet intervals = index_intervals[i];
        absl::StatusOr<Bits> index_bits =
            UBitsWithStatus(indexes[i], intervals.BitCount());
        if (!index_bits.ok()) {
          return false;
        }
        if (!intervals.Covers(*index_bits)) {
          return false;
        }
        indexes_bits.push_back(*index_bits);
      }

      // Build up `cases` and `conditions`.

      failure = handle_possible_index(indexes_bits);

      return !failure.ok();
    });

    if (!failure.ok()) {
      return failure;
    }

    // Finally, build the select chain.

    Node* rest_of_chain = nullptr;

    for (int64_t i = 0; i < conditions.size(); ++i) {
      if (rest_of_chain == nullptr) {
        rest_of_chain = cases[i];
        continue;
      }
      XLS_ASSIGN_OR_RETURN(
          rest_of_chain,
          f->MakeNode<Select>(loc, conditions[i],
                              std::vector<Node*>({rest_of_chain, cases[i]}),
                              /*default_value=*/std::nullopt));
    }

    if (rest_of_chain == nullptr) {
      return false;
    }

    XLS_RETURN_IF_ERROR(array_index->ReplaceUsesWith(rest_of_chain));

    return true;
  }

  // If it exists, returns the unique add immediately following a mulp.
  std::optional<BinOp*> PartialMultiplyImmediateSum(PartialProductOp* mul) {
    // Check for two tuple_index users, one for index=0 and another for index=1.
    if (mul->users().size() != 2) {
      return std::nullopt;
    }
    BinOp* seen_add[2] = {nullptr, nullptr};
    bool seen_index[2] = {false, false};
    for (Node* node : mul->users()) {
      if (!node->Is<TupleIndex>()) {
        return std::nullopt;
      }
      int64_t index = node->As<TupleIndex>()->index();
      XLS_CHECK_GE(index, 0);
      XLS_CHECK_LT(index, 2);
      if (seen_index[index]) {
        return std::nullopt;
      }
      seen_index[index] = true;
      if (node->users().size() != 1) {
        return std::nullopt;
      }
      Node* user = *node->users().begin();
      if (!user->Is<BinOp>() || user->As<BinOp>()->op() != Op::kAdd) {
        return std::nullopt;
      }
      seen_add[index] = user->As<BinOp>();
    }
    if (!seen_index[0] || !seen_index[1] || seen_add[0] != seen_add[1] ||
        seen_add[0] == nullptr || seen_add[1] == nullptr) {
      return std::nullopt;
    }
    return seen_add[0];
  }

  absl::Status NoChange() { return absl::OkStatus(); }
  absl::Status Change() {
    changed_ = true;
    return absl::OkStatus();
  }

  const QueryEngine& query_engine_;
  const AnalysisType analysis_;
  const OptimizationPassOptions& options_;
  bool changed_ = false;
};

}  // namespace

static void RangeAnalysisLog(FunctionBase* f,
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

      if (ternary_query_engine.IsTracked(node) &&
          range_query_engine.IsTracked(node)) {
        TernaryVector ternary_result =
            ternary_query_engine.GetTernary(node).Get({});
        TernaryVector range_result =
            range_query_engine.GetTernary(node).Get({});
        std::optional<TernaryVector> difference =
            ternary_ops::Difference(range_result, ternary_result);
        XLS_CHECK(difference.has_value())
            << "Inconsistency detected in node: " << node->GetName();
        bits_saved += ternary_ops::NumberOfKnownBits(difference.value());
      }
    }
  }

  if (bits_saved != 0) {
    XLS_VLOG(3) << "narrowing_pass: range analysis saved " << bits_saved
                << " bits in " << f << "\n";
  }
}

static absl::StatusOr<std::unique_ptr<QueryEngine>> GetQueryEngine(
    FunctionBase* f, AnalysisType analysis) {
  std::unique_ptr<QueryEngine> query_engine;
  if (analysis == AnalysisType::kRange) {
    auto ternary_query_engine = std::make_unique<TernaryQueryEngine>();
    auto range_query_engine = std::make_unique<RangeQueryEngine>();

    if (XLS_VLOG_IS_ON(3)) {
      RangeAnalysisLog(f, *ternary_query_engine, *range_query_engine);
    }

    std::vector<std::unique_ptr<QueryEngine>> engines;
    engines.push_back(std::move(ternary_query_engine));
    engines.push_back(std::move(range_query_engine));
    query_engine = std::make_unique<UnionQueryEngine>(std::move(engines));
  } else {
    query_engine = std::make_unique<TernaryQueryEngine>();
  }
  XLS_RETURN_IF_ERROR(query_engine->Populate(f).status());
  return std::move(query_engine);
}

absl::StatusOr<bool> NarrowingPass::RunOnFunctionBaseInternal(
    FunctionBase* f, const OptimizationPassOptions& options,
    PassResults* results) const {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<QueryEngine> query_engine,
                       GetQueryEngine(f, analysis_));

  NarrowVisitor narrower(*query_engine, analysis_, options);

  for (Node* node : TopoSort(f)) {
    // We specifically want gate ops to be eligible for being reduced to a
    // constant since there entire purpose is for preventing power consumption
    // and literals are basically free.
    if (OpIsSideEffecting(node->op()) && !node->Is<Gate>()) {
      continue;
    }
    if (!node->Is<Literal>() && !node->Is<Param>()) {
      XLS_ASSIGN_OR_RETURN(bool modified,
                           narrower.MaybeReplacePreciseWithLiteral(node));
      if (modified) {
        continue;
      }
    }
    XLS_RETURN_IF_ERROR(node->VisitSingleNode(&narrower));
  }
  return narrower.changed();
}

}  // namespace xls
