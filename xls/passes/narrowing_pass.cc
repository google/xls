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
#include <functional>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/logging/logging.h"
#include "xls/common/math_util.h"
#include "xls/common/module_initializer.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/inline_bitmap.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/function_base.h"
#include "xls/ir/interval.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/source_location.h"
#include "xls/ir/ternary.h"
#include "xls/ir/topo_sort.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/context_sensitive_range_query_engine.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/predicate_dominator_analysis.h"
#include "xls/passes/predicate_state.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/range_query_engine.h"
#include "xls/passes/ternary_query_engine.h"
#include "xls/passes/union_query_engine.h"

namespace xls {

namespace {

using AnalysisType = NarrowingPass::AnalysisType;

class SpecializedQueryEngines {
 public:
  SpecializedQueryEngines(AnalysisType type, PredicateDominatorAnalysis& pda,
                          const QueryEngine& base)
      : type_(type), base_(base), pda_(pda) {}
  const QueryEngine& ForSelect(PredicateState state) const {
    if (type_ != AnalysisType::kRangeWithContext) {
      return base_;
    }
    if (!engines_.contains(state)) {
      engines_.emplace(state, base_.SpecializeGivenPredicate({state}));
    }
    return *engines_.at(state);
  }
  const QueryEngine& ForNode(Node* node) const {
    if (type_ != AnalysisType::kRangeWithContext) {
      return base_;
    }
    return ForSelect(pda_.GetSingleNearestPredicate(node));
  }

 private:
  AnalysisType type_;
  const QueryEngine& base_;
  const PredicateDominatorAnalysis& pda_;
  mutable absl::flat_hash_map<PredicateState, std::unique_ptr<QueryEngine>>
      engines_;
};

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

absl::StatusOr<Node*> ExtractMostSignificantBits(Node* node,
                                                 int64_t bit_count) {
  XLS_RET_CHECK_GE(node->BitCountOrDie(), bit_count);
  if (node->BitCountOrDie() == bit_count) {
    return node;
  }
  return node->function_base()->MakeNode<BitSlice>(
      node->loc(), node, /*start=*/node->BitCountOrDie() - bit_count,
      /*width=*/bit_count);
}

class NarrowVisitor final : public DfsVisitorWithDefault {
 public:
  explicit NarrowVisitor(const SpecializedQueryEngines& engine,
                         AnalysisType analysis,
                         const OptimizationPassOptions& options,
                         bool splits_enabled)
      : specialized_query_engine_(engine),
        analysis_(analysis),
        options_(options),
        splits_enabled_(splits_enabled) {
    CHECK_NE(analysis_, NarrowingPass::AnalysisType::kRangeWithOptionalContext);
  }

  absl::Status MaybeReplacePreciseInputEdgeWithLiteral(Node* node) {
    if (analysis_ != AnalysisType::kRangeWithContext) {
      return NoChange();
    }
    const QueryEngine& node_qe = specialized_query_engine_.ForNode(node);
    bool changed = false;
    for (int64_t i = 0; i < node->operand_count(); ++i) {
      Node* to_replace = node->operand(i);
      if (to_replace->Is<Literal>()) {
        continue;
      }
      // TODO(allight): 2023-09-11: google/xls#1104 means we might have already
      // replaced the argument and can't easily retrieve what it was and so
      // won't be able to replace it with a literal. This is unfortunate and
      // should be remedied at some point.
      // if (!node_qe.IsTracked(to_replace)) {
      //   continue;
      // }
      XLS_ASSIGN_OR_RETURN(
          bool one_change,
          MaybeReplacePreciseWithLiteral(
              to_replace, node_qe,
              [&](const Value& value) -> absl::Status {
                XLS_ASSIGN_OR_RETURN(
                    Node * literal,
                    to_replace->function_base()->MakeNode<Literal>(
                        to_replace->loc(), value));
                return node->ReplaceOperandNumber(i, literal);
              },
              absl::StrFormat("in operand %d of %s", i, node->ToString())));
      changed = changed || one_change;
    }
    if (changed) {
      return Change();
    }
    return NoChange();
  }
  absl::StatusOr<bool> MaybeReplacePreciseWithLiteral(Node* node) {
    return MaybeReplacePreciseWithLiteral(
        node, specialized_query_engine_.ForNode(node),
        [&](const Value& value) {
          return node->ReplaceUsesWithNew<Literal>(value).status();
        },
        "in global usage");
  }
  absl::StatusOr<bool> MaybeReplacePreciseWithLiteral(
      Node* to_replace, const QueryEngine& query_engine,
      const std::function<absl::Status(const Value&)>& replace_with,
      std::string_view context) {
    LeafTypeTree<TernaryVector> ternary = query_engine.GetTernary(to_replace);
    for (Type* leaf_type : ternary.leaf_types()) {
      if (leaf_type->IsToken()) {
        XLS_RETURN_IF_ERROR(NoChange());
        return false;
      }
    }
    for (const TernaryVector& ternary_vector : ternary.elements()) {
      if (!ternary_ops::IsFullyKnown(ternary_vector)) {
        XLS_RETURN_IF_ERROR(NoChange());
        return false;
      }
    }
    LeafTypeTree<Value> value_ltt = leaf_type_tree::Map<Value, TernaryVector>(
        ternary.AsView(), [](const TernaryVector& ternary_vector) -> Value {
          return Value(ternary_ops::ToKnownBitsValues(ternary_vector));
        });
    XLS_ASSIGN_OR_RETURN(Value value, LeafTypeTreeToValue(value_ltt.AsView()));
    XLS_RETURN_IF_ERROR(replace_with(value));
    XLS_VLOG(3) << absl::StreamFormat(
        "Range analysis found precise value for %s == %s %s, replacing with "
        "literal\n",
        to_replace->GetName(), value.ToString(), context);
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

  absl::Status HandleSel(Select* sel) override {
    if (analysis_ != AnalysisType::kRangeWithContext) {
      return NoChange();
    }
    // Replace any input edge with the constant value if we can
    bool changed = false;
    for (int64_t i = 0; i < sel->cases().size(); ++i) {
      XLS_ASSIGN_OR_RETURN(
          bool one_change,
          MaybeReplaceSelectArmWithPreciseValue(PredicateState(sel, i)));
      changed = changed || one_change;
    }
    if (sel->default_value()) {
      XLS_ASSIGN_OR_RETURN(bool one_change,
                           MaybeReplaceSelectArmWithPreciseValue(PredicateState(
                               sel, PredicateState::kDefaultArm)));
      changed = changed || one_change;
    }
    if (changed) {
      return Change();
    }
    return NoChange();
  }

  absl::StatusOr<bool> MaybeReplaceSelectArmWithPreciseValue(
      PredicateState state) {
    CHECK(!state.IsBasePredicate());
    const QueryEngine& qe = specialized_query_engine_.ForSelect(state);
    Select* select = state.node()->As<Select>();
    Node* to_replace = state.IsDefaultArm()
                           ? *select->default_value()
                           : select->get_case(state.arm_index());
    int64_t arg_num = state.IsDefaultArm() ? select->operand_count() - 1
                                           : state.arm_index() + 1;
    return MaybeReplacePreciseWithLiteral(
        to_replace, qe,
        [&](const Value& value) -> absl::Status {
          XLS_ASSIGN_OR_RETURN(Node * literal,
                               to_replace->function_base()->MakeNode<Literal>(
                                   to_replace->loc(), value));
          return select->ReplaceOperandNumber(arg_num, literal);
        },
        absl::StrFormat(
            "as case %d in %s",
            state.IsDefaultArm() ? select->cases().size() : state.arm_index(),
            select->ToString()));
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
  absl::Status HandleNeg(UnOp* neg) override {
    // Narrows negate:
    //
    //  neg(0b00...00XXX) => signext(neg(0b0XXX))
    //  neg(0b00...0000X) => signext(0bX)
    Node* input = neg->operand(UnOp::kArgOperand);
    int64_t leading_zero = CountLeadingKnownZeros(input, neg);
    if (leading_zero == 0 || leading_zero == 1) {
      // Transform is - [00000??] -> signext[-[0??], width] so if there are no
      // leading zeros or only one we don't get any benefit.
      return NoChange();
    }
    int64_t unknown_segment = input->BitCountOrDie() - leading_zero;
    if (unknown_segment == 1) {
      // Bit-extend the Least significant (zero-th) bit.
      XLS_ASSIGN_OR_RETURN(
          Node * narrowed_input,
          neg->function_base()->MakeNode<BitSlice>(
              neg->loc(), neg->operand(UnOp::kArgOperand), 0, 1));
      XLS_RETURN_IF_ERROR(
          neg->ReplaceUsesWithNew<ExtendOp>(narrowed_input,
                                            neg->BitCountOrDie(), Op::kSignExt)
              .status());
      return Change();
    }
    // Slice then neg then extend the negated value.
    XLS_ASSIGN_OR_RETURN(Node * narrowed_input,
                         neg->function_base()->MakeNode<BitSlice>(
                             neg->loc(), neg->operand(UnOp::kArgOperand), 0,
                             unknown_segment + 1));
    XLS_ASSIGN_OR_RETURN(Node * narrowed_neg,
                         neg->function_base()->MakeNode<UnOp>(
                             neg->loc(), narrowed_input, Op::kNeg));
    XLS_RETURN_IF_ERROR(neg->ReplaceUsesWithNew<ExtendOp>(
                               narrowed_neg, neg->BitCountOrDie(), Op::kSignExt)
                            .status());
    return Change();
  }

  // Try to narrow the operands of comparison operations. Returns true if the
  // given compare operation was narrowed.
  absl::Status MaybeNarrowCompare(CompareOp* compare) {
    if (!compare->operand(0)->GetType()->IsBits()) {
      return NoChange();
    }
    const QueryEngine& query_engine =
        specialized_query_engine_.ForNode(compare);
    // Returns the number of consecutive leading/trailing bits that are known to
    // be equal between the LHS and RHS of the given compare operation.
    auto matched_leading_operand_bits = [&](CompareOp* c) -> int64_t {
      int64_t bit_count = c->operand(0)->BitCountOrDie();
      for (int64_t i = 0; i < bit_count; ++i) {
        int64_t bit_index = bit_count - i - 1;
        if (!query_engine.KnownEquals(
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
        if (!query_engine.KnownEquals(TreeBitLocation(c->operand(0), i),
                                      TreeBitLocation(c->operand(1), i))) {
          return i;
        }
      }
      return bit_count;
    };

    // Narrow the operands of the compare to the given bit count. Replace the
    // given comparison operation with the new narrower compare operation.
    // Optionally provide a new op, which if unspecified will be the same as the
    // original op.
    auto narrow_compare_operands =
        [](CompareOp* c, int64_t start, int64_t bit_count,
           std::optional<Op> new_op = std::nullopt) -> absl::Status {
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
          ->ReplaceUsesWithNew<CompareOp>(narrowed_lhs, narrowed_rhs,
                                          new_op.value_or(c->op()))
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
    if (all_bits_match) {
      return NoChange();
    }
    if (matched_leading_bits > 0 || matched_trailing_bits > 0) {
      // In most cases, we narrow to the same op.
      Op new_op = compare->op();
      // Signed comparisons can be narrowed to unsigned comparisons if operands
      // have known-equal MSBs.
      // If we're only trimming LSBs, keep the operation the same.
      if (matched_leading_bits > 0 && IsSignedCompare(compare) &&
          compare->op() != Op::kEq && compare->op() != Op::kNe) {
        XLS_ASSIGN_OR_RETURN(new_op, SignedCompareToUnsigned(compare->op()));
      }
      XLS_VLOG(3) << absl::StreamFormat(
          "Leading %d bits and trailing %d bits of comparison operation %s "
          "match",
          matched_leading_bits, matched_trailing_bits, compare->GetName());
      XLS_RETURN_IF_ERROR(narrow_compare_operands(
          compare, /*start=*/matched_trailing_bits,
          operand_width - matched_leading_bits - matched_trailing_bits,
          /*new_op=*/new_op));
      return Change();
    }

    // For a signed comparison, if both operands look like sign extensions
    // (either because they are sign extend op or because leading bits == MSB),
    // the leading bits can be sliced away (except the unsliced bit must remain
    // as the sign bit).

    // Helper to evaluate how many leading bits == MSB.
    auto bits_eq_to_msb = [&](Node* n) -> int64_t {
      int64_t bit_count = n->BitCountOrDie();
      int64_t msb_index = bit_count - 1;
      for (int64_t i = 1; i < bit_count; ++i) {
        int64_t bit_index = bit_count - i - 1;
        if (!query_engine.KnownEquals(TreeBitLocation(n, msb_index),
                                      TreeBitLocation(n, bit_index))) {
          return i - 1;
        }
      }
      return bit_count - 1;
    };
    // Ideally, bits_eq_to_msb would be sufficient, but query engines are not
    // good at evaluating ExtendOps. This is a similar helper specialized on
    // ExtendOps.
    auto sign_ext_bits = [](Node* n) -> int64_t {
      switch (n->op()) {
        case Op::kSignExt:
          return n->As<ExtendOp>()->new_bit_count() -
                 n->operand(0)->BitCountOrDie();
        case Op::kZeroExt:
          // subtract 1 because zero extend's extra bits do not match old msb.
          // Take max with 0 in case new_bit_count == old_bit_count.
          return std::max(int64_t{0}, n->As<ExtendOp>()->new_bit_count() -
                                          n->operand(0)->BitCountOrDie() - 1);
        default:
          return 0;
      }
    };
    int64_t op0_bits_eq_to_msb = std::max(bits_eq_to_msb(compare->operand(0)),
                                          sign_ext_bits(compare->operand(0)));
    int64_t op1_bits_eq_to_msb = std::max(bits_eq_to_msb(compare->operand(1)),
                                          sign_ext_bits(compare->operand(1)));
    int64_t extra_msbs = std::min(op0_bits_eq_to_msb, op1_bits_eq_to_msb);
    if (extra_msbs > 0) {
      XLS_RETURN_IF_ERROR(
          narrow_compare_operands(compare, /*start=*/0,
                                  /*bit_count=*/operand_width - extra_msbs));
      return Change();
    }

    return NoChange();
  }

  absl::Status HandleLiteral(Literal* literal) override {
    return MaybeNarrowLiteralArray(literal);
  }

  struct NonconstantSlice {
    int64_t start = -1;
    int64_t width = 0;

    bool empty() { return width == 0; }

    void clear() {
      start = -1;
      width = 0;
    }
  };
  using Slice = std::variant<Literal*, NonconstantSlice>;
  struct SlicedArray {
    std::vector<Slice> slices;
    Literal* array_literal;
  };
  absl::StatusOr<SlicedArray> NarrowLiteralArrayToSlices(
      Literal* array_literal) {
    absl::Span<const Value> elements = array_literal->value().elements();
    const int64_t bit_count = elements[0].bits().bit_count();
    TernaryVector constant_bits =
        ternary_ops::BitsToTernary(elements[0].bits());
    for (const Value& element : elements) {
      DCHECK(element.IsBits());
      ternary_ops::UpdateWithIntersection(constant_bits, element.bits());
    }

    int64_t known_bits = ternary_ops::NumberOfKnownBits(constant_bits);
    if (known_bits == bit_count) {
      XLS_ASSIGN_OR_RETURN(
          Literal * constant_literal,
          array_literal->function_base()->MakeNode<Literal>(
              array_literal->loc(),
              Value(ternary_ops::ToKnownBitsValues(constant_bits))));
      return SlicedArray{
          .slices = {constant_literal},
          .array_literal = nullptr,
      };
    }
    if (known_bits == 0) {
      return SlicedArray{
          .slices = {NonconstantSlice{.start = 0, .width = bit_count}},
          .array_literal = array_literal,
      };
    }

    if (!splits_enabled_) {
      // Cannot slice our array without creating multiple ops per ArrayIndex.
      return SlicedArray{
          .slices = {NonconstantSlice{.start = 0, .width = bit_count}},
          .array_literal = array_literal,
      };
    }

    std::vector<Slice> slices;
    absl::InlinedVector<bool, 64> current_known_slice;
    current_known_slice.reserve(ternary_ops::NumberOfKnownBits(constant_bits));
    NonconstantSlice current_unknown_slice;
    auto finish_known_slice = [&]() -> absl::Status {
      XLS_ASSIGN_OR_RETURN(Literal * constant_slice,
                           array_literal->function_base()->MakeNode<Literal>(
                               array_literal->loc(),
                               Value(Bits::FromBitmap(InlineBitmap::FromBits(
                                   current_known_slice)))));
      slices.push_back(constant_slice);
      current_known_slice.clear();
      return absl::OkStatus();
    };
    auto finish_unknown_slice = [&]() -> absl::Status {
      slices.push_back(current_unknown_slice);
      current_unknown_slice.clear();
      return absl::OkStatus();
    };
    for (int64_t i = 0; i < constant_bits.size(); ++i) {
      if (constant_bits[i] == TernaryValue::kUnknown) {
        // We're in an unknown slice. Record the preceding known slice, if any.
        if (!current_known_slice.empty()) {
          XLS_RETURN_IF_ERROR(finish_known_slice());
        }

        // Extend the current slice, or start a new one if we need to.
        if (current_unknown_slice.empty()) {
          current_unknown_slice = {.start = i, .width = 1};
        } else {
          current_unknown_slice.width++;
        }
      } else {
        // We're in a known slice. Record the preceding unknown slice, if any.
        if (!current_unknown_slice.empty()) {
          XLS_RETURN_IF_ERROR(finish_unknown_slice());
        }

        // Extend the current slice; if the vector was empty, this will be
        // interpreted to start a new one.
        current_known_slice.push_back(constant_bits[i] ==
                                      TernaryValue::kKnownOne);
        continue;
      }
    }
    if (!current_known_slice.empty()) {
      XLS_RETURN_IF_ERROR(finish_known_slice());
    }
    if (!current_unknown_slice.empty()) {
      XLS_RETURN_IF_ERROR(finish_unknown_slice());
    }
    DCHECK_GT(slices.size(), 1);

    std::vector<Value> narrowed_elements;
    narrowed_elements.reserve(array_literal->value().elements().size());
    for (const Value& element : elements) {
      const Bits& bits = element.bits();
      InlineBitmap narrowed_bits(bits.bit_count() - known_bits);
      int64_t narrowed_idx = 0;
      for (int64_t i = 0; i < bits.bit_count(); ++i) {
        if (ternary_ops::IsUnknown(constant_bits[i])) {
          narrowed_bits.Set(narrowed_idx++, bits.Get(i));
        }
      }
      narrowed_elements.push_back(
          Value(Bits::FromBitmap(std::move(narrowed_bits))));
    }
    XLS_ASSIGN_OR_RETURN(Value narrowed_array, Value::Array(narrowed_elements));
    XLS_ASSIGN_OR_RETURN(Literal * narrowed_array_literal,
                         array_literal->function_base()->MakeNode<Literal>(
                             array_literal->loc(), narrowed_array));
    return SlicedArray{
        .slices = std::move(slices),
        .array_literal = narrowed_array_literal,
    };
  }

  // If the literal is an array of Bits with some constant bits, narrow all
  // elements and (whenever we index into it) replace the known bits by
  // concatenation with an appropriate constant, using appropriate slicing.
  absl::Status MaybeNarrowLiteralArray(Literal* literal) {
    if (literal->IsDead()) {
      // Don't try to optimize a dead literal; among other problems, it may be
      // the leftovers from a previous run of this optimization.
      return NoChange();
    }
    if (!literal->GetType()->IsArray() ||
        !literal->GetType()->AsArrayOrDie()->element_type()->IsBits()) {
      return NoChange();
    }
    if (literal->value().elements().empty()) {
      return NoChange();
    }

    // If there are no ArrayIndex accesses for this literal that we can improve,
    // stop.
    if (absl::c_none_of(literal->users(), [literal](Node* user) {
          return user->Is<ArrayIndex>() &&
                 user->As<ArrayIndex>()->array() == literal &&
                 !user->As<ArrayIndex>()->indices().empty();
        })) {
      return NoChange();
    }

    XLS_ASSIGN_OR_RETURN(SlicedArray sliced_array,
                         NarrowLiteralArrayToSlices(literal));

    CHECK_GE(sliced_array.slices.size(), 1);
    if (sliced_array.slices.size() == 1) {
      if (std::holds_alternative<NonconstantSlice>(sliced_array.slices[0])) {
        // This array can't be narrowed.
        DCHECK_EQ(sliced_array.array_literal, literal);
        return NoChange();
      }

      // All ArrayIndex accesses can be replaced with a single literal.
      CHECK(std::holds_alternative<Literal*>(sliced_array.slices[0]));
      CHECK_EQ(sliced_array.array_literal, nullptr);
      Literal* constant_literal = std::get<Literal*>(sliced_array.slices[0]);
      for (Node* user : literal->users()) {
        if (!user->Is<ArrayIndex>()) {
          continue;
        }
        ArrayIndex* array_index = user->As<ArrayIndex>();
        if (array_index->indices().empty()) {
          continue;
        }
        XLS_RET_CHECK(array_index->array() == literal);

        // Replace the ArrayIndex with a direct reference to the common Literal.
        XLS_RETURN_IF_ERROR(array_index->ReplaceUsesWith(constant_literal));
      }
      return Change();
    }

    // We have more than one slice; at least one should be from the narrowed
    // array literal.
    CHECK_NE(sliced_array.array_literal, nullptr);

    for (Node* user : literal->users()) {
      if (!user->Is<ArrayIndex>()) {
        continue;
      }
      ArrayIndex* array_index = user->As<ArrayIndex>();
      if (array_index->indices().empty()) {
        continue;
      }
      XLS_RET_CHECK(array_index->array() == literal);

      // Index into the narrowed array; we'll slice this up & concat with
      // literals to reconstruct the full value.
      XLS_ASSIGN_OR_RETURN(ArrayIndex * new_array_index,
                           array_index->function_base()->MakeNode<ArrayIndex>(
                               array_index->loc(), sliced_array.array_literal,
                               array_index->indices()));

      std::vector<Node*> bit_slices;
      bit_slices.reserve(sliced_array.slices.size());

      int64_t nonconstant_bit_position = 0;
      for (const Slice& slice : sliced_array.slices) {
        if (std::holds_alternative<NonconstantSlice>(slice)) {
          const NonconstantSlice& nonconstant_slice =
              std::get<NonconstantSlice>(slice);
          XLS_ASSIGN_OR_RETURN(
              BitSlice * value_slice,
              array_index->function_base()->MakeNode<BitSlice>(
                  array_index->loc(), new_array_index, nonconstant_bit_position,
                  nonconstant_slice.width));
          nonconstant_bit_position += nonconstant_slice.width;
          bit_slices.push_back(value_slice);
        } else {
          CHECK(std::holds_alternative<Literal*>(slice));
          Literal* constant_slice = std::get<Literal*>(slice);
          DCHECK(constant_slice->value().IsBits());
          bit_slices.push_back(constant_slice);
        }
      }

      // Reverse the slice order for correct bit ordering.
      absl::c_reverse(bit_slices);
      XLS_ASSIGN_OR_RETURN(Node * array_index_value,
                           array_index->function_base()->MakeNode<Concat>(
                               array_index->loc(), bit_slices));
      XLS_RETURN_IF_ERROR(user->ReplaceUsesWith(array_index_value));
    }

    return Change();
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
    int64_t leading_zeros =
        CountLeadingKnownZeros(shift->operand(1), /*user=*/shift);
    if (leading_zeros == 0) {
      return NoChange();
    }

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

  absl::Status HandleDecode(Decode* decode) override {
    // Narrow the index operand of decode operations if the index has leading
    // zeros.
    Node* index = decode->operand(0);

    int64_t leading_zeros = CountLeadingKnownZeros(index, /*user=*/decode);
    if (leading_zeros == 0) {
      return NoChange();
    }

    if (leading_zeros == index->BitCountOrDie()) {
      // Index is zero; result is 1.
      XLS_RETURN_IF_ERROR(decode
                              ->ReplaceUsesWithNew<Literal>(
                                  Value(UBits(1, decode->BitCountOrDie())))
                              .status());
      return Change();
    }

    // Prune the leading zeros from the index.
    XLS_ASSIGN_OR_RETURN(Node * narrowed_index,
                         decode->function_base()->MakeNode<BitSlice>(
                             decode->loc(), index, /*start=*/0,
                             /*width=*/index->BitCountOrDie() - leading_zeros));

    // Decode doesn't automatically zero-extend past the last bit the input
    // could code for, so we limit the width to what it finds acceptable, then
    // zero-extend afterwards.
    int64_t result_width = decode->BitCountOrDie();
    int64_t new_decode_width = result_width;

    // Any index with >= 63 bits can't be full-decoded, so we don't need to
    // limit the width in this case.
    constexpr int64_t kMaxFullDecodeWidth = 63;
    if (narrowed_index->BitCountOrDie() < kMaxFullDecodeWidth) {
      new_decode_width = std::min(
          new_decode_width, int64_t{1} << narrowed_index->BitCountOrDie());
    }
    XLS_ASSIGN_OR_RETURN(Decode * narrowed_decode,
                         decode->function_base()->MakeNode<Decode>(
                             decode->loc(), narrowed_index, new_decode_width));
    if (new_decode_width == result_width) {
      XLS_RETURN_IF_ERROR(decode->ReplaceUsesWith(narrowed_decode));
    } else {
      XLS_RETURN_IF_ERROR(decode
                              ->ReplaceUsesWithNew<ExtendOp>(
                                  narrowed_decode, result_width, Op::kZeroExt)
                              .status());
    }
    return Change();
  }

  // TODO(allight): 2023-11-08: We could simplify this and add by recognizing
  // when the leading bits match on both sides and just doing a sign extend.
  // i.e. lhs[MSB] = lhs[MSB-1] = ... and rhs[MSB] = rhs[MSB-1] = ...,
  absl::Status HandleSub(BinOp* sub) override {
    XLS_VLOG(3) << "Trying to narrow sub: " << sub->ToString();
    XLS_RET_CHECK_EQ(sub->op(), Op::kSub);

    Node* lhs = sub->operand(0);
    Node* rhs = sub->operand(1);
    const int64_t bit_count = sub->BitCountOrDie();
    if (lhs->BitCountOrDie() != rhs->BitCountOrDie()) {
      return NoChange();
    }
    // Figure out how many known bits we have.
    int64_t leading_zeros = CountLeadingKnownZeros(sub, /*user=*/std::nullopt);
    int64_t leading_ones = CountLeadingKnownOnes(sub, /*user=*/std::nullopt);
    if (leading_zeros == 0 && leading_ones == 0) {
      return NoChange();
    }
    bool is_one = leading_ones != 0;
    int64_t known_leading = is_one ? leading_ones : leading_zeros;
    int64_t required_bits = bit_count - known_leading;
    if (is_one) {
      // XLS_ASSIGN_OR_RETURN(
      //     Node * all_ones,
      //     sub->function_base()->MakeNode<Literal>(
      //         sub->loc(), Value(Bits::AllOnes(known_leading))));
      // XLS_RETURN_IF_ERROR(sub->ReplaceUsesWithNew<Concat>(
      //                            absl::MakeConstSpan({all_ones, new_sub}))
      //                         .status());
      // TODO(allight) 2023-11-08: We should extend range analysis to catch
      // known-negative results. Until we do though this is dead code.
      return NoChange();
    }
    XLS_ASSIGN_OR_RETURN(Node * new_lhs, MaybeNarrow(lhs, required_bits));
    XLS_ASSIGN_OR_RETURN(Node * new_rhs, MaybeNarrow(rhs, required_bits));
    XLS_ASSIGN_OR_RETURN(Node * new_sub,
                         sub->function_base()->MakeNode<BinOp>(
                             sub->loc(), new_lhs, new_rhs, Op::kSub));
    XLS_RETURN_IF_ERROR(
        sub->ReplaceUsesWithNew<ExtendOp>(new_sub, bit_count, Op::kZeroExt)
            .status());
    return Change();
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
        std::min(CountLeadingKnownZeros(lhs, /*user=*/add),
                 CountLeadingKnownZeros(rhs, /*user=*/add));

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
      XLS_ASSIGN_OR_RETURN(
          std::optional<Node*> operand0,
          MaybeNarrowUnsignedOperand(mul->operand(0), /*user=*/mul));
      XLS_ASSIGN_OR_RETURN(
          std::optional<Node*> operand1,
          MaybeNarrowUnsignedOperand(mul->operand(1), /*user=*/mul));
      if (operand0.has_value() || operand1.has_value()) {
        XLS_RETURN_IF_ERROR(
            mul->ReplaceUsesWithNew<ArithOp>(operand0.value_or(mul->operand(0)),
                                             operand1.value_or(mul->operand(1)),
                                             result_bit_count, Op::kUMul)
                .status());
        return Change();
      }
    }

    // Sign-extended operands of signed multiplies can be narrowed by
    // replacing the operand of the multiply with the value before
    // sign-extension.
    if (mul->op() == Op::kSMul || is_sign_agnostic) {
      XLS_ASSIGN_OR_RETURN(
          std::optional<Node*> operand0,
          MaybeNarrowSignedOperand(mul->operand(0), /*user=*/mul));
      XLS_ASSIGN_OR_RETURN(
          std::optional<Node*> operand1,
          MaybeNarrowSignedOperand(mul->operand(1), /*user=*/mul));
      if (operand0.has_value() || operand1.has_value()) {
        XLS_RETURN_IF_ERROR(
            mul->ReplaceUsesWithNew<ArithOp>(operand0.value_or(mul->operand(0)),
                                             operand1.value_or(mul->operand(1)),
                                             result_bit_count, Op::kSMul)
                .status());
        return Change();
      }
    }

    int64_t left_trailing_zeros = CountTrailingKnownZeros(lhs, mul);
    int64_t right_trailing_zeros = CountTrailingKnownZeros(rhs, mul);
    if (left_trailing_zeros > 0 || right_trailing_zeros > 0) {
      int64_t removed_bits = left_trailing_zeros + right_trailing_zeros;
      XLS_ASSIGN_OR_RETURN(Node * new_left, ExtractMostSignificantBits(
                                                lhs, lhs->BitCountOrDie() -
                                                         left_trailing_zeros));
      XLS_ASSIGN_OR_RETURN(
          Node * new_right,
          ExtractMostSignificantBits(
              rhs, rhs->BitCountOrDie() - right_trailing_zeros));
      XLS_ASSIGN_OR_RETURN(
          Node * new_mul,
          mul->function_base()->MakeNodeWithName<ArithOp>(
              mul->loc(), new_left, new_right, mul->width() - removed_bits,
              mul->op(), mul->GetName() + "_NarrowedMult_"));
      XLS_ASSIGN_OR_RETURN(Node * zeros,
                           mul->function_base()->MakeNodeWithName<Literal>(
                               mul->loc(), Value(Bits(removed_bits)),
                               mul->GetName() + "_TrailingBits_"));
      XLS_RETURN_IF_ERROR(mul->ReplaceUsesWithNew<Concat>(
                                 absl::Span<Node* const>{new_mul, zeros})
                              .status());
      return Change();
    }

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

      // Compute the minimum number of bits required to index the entire
      // array.
      int64_t array_size = array_type->AsArrayOrDie()->size();
      int64_t min_index_width =
          std::max(int64_t{1}, Bits::MinBitCountUnsigned(array_size - 1));

      if (index->Is<Literal>()) {
        const Bits& bits_index = index->As<Literal>()->value().bits();
        Bits new_bits_index = bits_index;
        if (bits_ops::UGreaterThanOrEqual(bits_index, array_size)) {
          // Index is out-of-bounds. Replace with a (potentially narrower)
          // index equal to the first out-of-bounds element.
          new_bits_index =
              UBits(array_size, Bits::MinBitCountUnsigned(array_size));
        } else if (bits_index.bit_count() > min_index_width) {
          // Index is in-bounds and is wider than necessary to index the
          // entire array. Replace with a literal which is perfectly sized
          // (width) to index the whole array.
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
      int64_t leading_zeros =
          CountLeadingKnownZeros(index, /*user=*/array_index);
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

    // If both elements of the mulp tuple are immediately added, the mulp
    // result can be unconditionally narrowed to the sum of the operand
    // widths, the add replaced with a narrowed add, and the result of the
    // addition zero/sign extended.
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
      XLS_ASSIGN_OR_RETURN(
          std::optional<Node*> operand0,
          MaybeNarrowUnsignedOperand(mul->operand(0), /*user=*/mul));
      XLS_ASSIGN_OR_RETURN(
          std::optional<Node*> operand1,
          MaybeNarrowUnsignedOperand(mul->operand(1), /*user=*/mul));
      if (operand0.has_value() || operand1.has_value()) {
        XLS_RETURN_IF_ERROR(mul->ReplaceUsesWithNew<PartialProductOp>(
                                   operand0.value_or(mul->operand(0)),
                                   operand1.value_or(mul->operand(1)),
                                   result_bit_count, Op::kUMulp)
                                .status());
        return Change();
      }
    }

    // Sign-extended operands of signed multiplies can be narrowed by
    // replacing the operand of the multiply with the value before
    // sign-extension.
    if (mul->op() == Op::kSMulp || is_sign_agnostic) {
      XLS_ASSIGN_OR_RETURN(
          std::optional<Node*> operand0,
          MaybeNarrowSignedOperand(mul->operand(0), /*user=*/mul));
      XLS_ASSIGN_OR_RETURN(
          std::optional<Node*> operand1,
          MaybeNarrowSignedOperand(mul->operand(1), /*user=*/mul));
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
  // Return the number of trailing known zeros in the given nodes values when
  // used as an argument to 'user'. If user is std::nullopt the context isn't
  // used.
  int64_t CountTrailingKnownZeros(Node* node, std::optional<Node*> user) const {
    CHECK(node->GetType()->IsBits());
    CHECK(user != nullptr);
    int64_t trailing_zeros = 0;
    const QueryEngine& node_query_engine =
        specialized_query_engine_.ForNode(node);
    const std::optional<const QueryEngine*> user_query_engine =
        analysis_ == AnalysisType::kRangeWithContext && user.has_value()
            ? std::make_optional(&specialized_query_engine_.ForNode(*user))
            : std::nullopt;
    auto is_user_zero = [&](const TreeBitLocation& tbl) {
      if (user_query_engine) {
        return (*user_query_engine)->IsZero(tbl);
      }
      return false;
    };
    for (int64_t i = 0; i < node->BitCountOrDie(); ++i) {
      if (!node_query_engine.IsZero(TreeBitLocation(node, i)) &&
          !is_user_zero(TreeBitLocation(node, i))) {
        break;
      }
      ++trailing_zeros;
    }
    return trailing_zeros;
  }
  // Return the number of leading known zeros in the given nodes values when
  // used as an argument to 'user'. If user is std::nullopt the context isn't
  // used.
  int64_t CountLeadingKnownZeros(Node* node, std::optional<Node*> user) const {
    CHECK(node->GetType()->IsBits());
    CHECK(user != nullptr);
    int64_t leading_zeros = 0;
    const QueryEngine& node_query_engine =
        specialized_query_engine_.ForNode(node);
    const std::optional<const QueryEngine*> user_query_engine =
        analysis_ == AnalysisType::kRangeWithContext && user.has_value()
            ? std::make_optional(&specialized_query_engine_.ForNode(*user))
            : std::nullopt;
    auto is_user_zero = [&](const TreeBitLocation& tbl) {
      if (user_query_engine) {
        return (*user_query_engine)->IsZero(tbl);
      }
      return false;
    };
    for (int64_t i = node->BitCountOrDie() - 1; i >= 0; --i) {
      if (!node_query_engine.IsZero(TreeBitLocation(node, i)) &&
          !is_user_zero(TreeBitLocation(node, i))) {
        break;
      }
      ++leading_zeros;
    }
    return leading_zeros;
  }

  // Return the number of leading known ones in the given nodes values when
  // the nodes value is used as an argument to 'user'. 'user' must not be
  // null.
  int64_t CountLeadingKnownOnes(Node* node, std::optional<Node*> user) const {
    CHECK(node->GetType()->IsBits());
    CHECK(user != nullptr);
    int64_t leading_ones = 0;
    const QueryEngine& node_query_engine =
        specialized_query_engine_.ForNode(node);
    const std::optional<const QueryEngine*> user_query_engine =
        analysis_ == AnalysisType::kRangeWithContext && user.has_value()
            ? std::make_optional(&specialized_query_engine_.ForNode(*user))
            : std::nullopt;
    auto is_user_one = [&](const TreeBitLocation& tbl) {
      if (user_query_engine) {
        return (*user_query_engine)->IsOne(tbl);
      }
      return false;
    };
    for (int64_t i = node->BitCountOrDie() - 1; i >= 0; --i) {
      if (!node_query_engine.IsOne(TreeBitLocation(node, i)) &&
          !is_user_one(TreeBitLocation(node, i))) {
        break;
      }
      ++leading_ones;
    }
    return leading_ones;
  }

  absl::StatusOr<std::optional<Node*>> MaybeNarrowUnsignedOperand(Node* operand,
                                                                  Node* user) {
    int64_t leading_zeros = CountLeadingKnownZeros(operand, user);
    if (leading_zeros == 0) {
      return std::nullopt;
    }
    return operand->function_base()->MakeNode<BitSlice>(
        operand->loc(), operand, /*start=*/0,
        /*width=*/operand->BitCountOrDie() - leading_zeros);
  }

  absl::StatusOr<std::optional<Node*>> MaybeNarrowSignedOperand(Node* operand,
                                                                Node* user) {
    if (operand->op() == Op::kSignExt) {
      // Operand is a sign-extended value. Just use the value before
      // sign-extension.
      return operand->operand(0);
    }
    if (CountLeadingKnownZeros(operand, user) > 1) {
      // Operand has more than one leading zero, something like:
      //    operand = 0000XXXX
      // This is equivalent to:
      //    operand = signextend(0XXXX)
      // So we can replace the operand with 0XXXX.
      return MaybeNarrow(
          operand,
          operand->BitCountOrDie() - CountLeadingKnownZeros(operand, user) + 1);
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
    // For example, if the array has shape [5, 7, 3, 2] and we index with [i,
    // j] then this will contain [5, 7].
    std::vector<int64_t> dimension;

    // The interval set that each index lives in.
    std::vector<IntervalSet> index_intervals;

    // Populate `dimension` and `index_intervals`.
    auto get_intervals = [&](Node* value) {
      const QueryEngine& array_engine =
          specialized_query_engine_.ForNode(array_index);
      if (analysis_ != AnalysisType::kRangeWithContext) {
        return array_engine.GetIntervals(value).Get({});
      }
      const QueryEngine& value_engine =
          specialized_query_engine_.ForNode(value);
      return IntervalSet::Intersect(value_engine.GetIntervals(value).Get({}),
                                    array_engine.GetIntervals(value).Get({}));
    };
    {
      ArrayType* array_type = array_index->array()->GetType()->AsArrayOrDie();
      for (Node* index : array_index->indices()) {
        dimension.push_back(array_type->size());
        index_intervals.push_back(get_intervals(index));
        absl::StatusOr<ArrayType*> array_type_status =
            array_type->element_type()->AsArray();
        array_type = array_type_status.ok() ? *array_type_status : nullptr;
      }
    }

    // Return early to avoid generating too much code if the index space is
    // big.
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
    // This vector contains one one-bit Node per case in the OneHotSelect,
    // which will be concatenated together to form the selector.
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
    // Check for two tuple_index users, one for index=0 and another for
    // index=1.
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
      CHECK_GE(index, 0);
      CHECK_LT(index, 2);
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

  const SpecializedQueryEngines& specialized_query_engine_;
  const AnalysisType analysis_;
  const OptimizationPassOptions& options_;
  const bool splits_enabled_;
  bool changed_ = false;
};

}  // namespace

template <typename RangeEngine>
static void RangeAnalysisLog(FunctionBase* f,
                             const TernaryQueryEngine& ternary_query_engine,
                             const RangeEngine& range_query_engine) {
  int64_t bits_saved = 0;
  absl::flat_hash_map<Node*, absl::flat_hash_set<Node*>> parents_map;

  for (Node* node : f->nodes()) {
    for (Node* operand : node->operands()) {
      parents_map[operand].insert(node);
    }
  }

  for (Node* node : f->nodes()) {
    if (node->GetType()->IsBits()) {
      IntervalSet intervals = range_query_engine.GetIntervals(node).Get({});
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
            range_query_engine.GetIntervals(operand).Get({});
        intervals.Normalize();
        if (!intervals.IsMaximal()) {
          inputs_all_maximal = false;
          break;
        }
      }
      if (!inputs_all_maximal &&
          range_query_engine.GetIntervals(node).Get({}).IsMaximal()) {
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
        CHECK(difference.has_value())
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
  if (analysis == AnalysisType::kRangeWithContext) {
    auto ternary_query_engine = std::make_unique<TernaryQueryEngine>();
    auto range_query_engine =
        std::make_unique<ContextSensitiveRangeQueryEngine>();

    if (VLOG_IS_ON(3)) {
      RangeAnalysisLog(f, *ternary_query_engine, *range_query_engine);
    }

    std::vector<std::unique_ptr<QueryEngine>> engines;
    engines.push_back(std::move(ternary_query_engine));
    engines.push_back(std::move(range_query_engine));
    query_engine = std::make_unique<UnionQueryEngine>(std::move(engines));
  } else if (analysis == AnalysisType::kRange) {
    auto ternary_query_engine = std::make_unique<TernaryQueryEngine>();
    auto range_query_engine = std::make_unique<RangeQueryEngine>();

    if (VLOG_IS_ON(3)) {
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
                       GetQueryEngine(f, RealAnalysis(options)));

  PredicateDominatorAnalysis pda = PredicateDominatorAnalysis::Run(f);
  SpecializedQueryEngines sqe(RealAnalysis(options), pda, *query_engine);

  NarrowVisitor narrower(sqe, RealAnalysis(options), options,
                         SplitsEnabled(opt_level_));

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
    // Force input edges to be constant if possible.
    // We do this after handling the node itself with the expectation that those
    // have more powerful transforms.
    // TODO(allight): 2023-09-11: google/xls#1104 makes this a bit less
    // effective than it could be by hitting both this transform and the
    // transform the node-specific handler did.
    XLS_RETURN_IF_ERROR(narrower.MaybeReplacePreciseInputEdgeWithLiteral(node));
  }
  // LOG(ERROR) << "Unable to analyze " << narrower.err_cnt() << " times!";
  return narrower.changed();
}
AnalysisType NarrowingPass::RealAnalysis(
    const OptimizationPassOptions& options) const {
  if (analysis_ == AnalysisType::kRangeWithOptionalContext) {
    return options.use_context_narrowing_analysis
               ? AnalysisType::kRangeWithContext
               : AnalysisType::kRange;
  }
  return analysis_;
}

std::ostream& operator<<(std::ostream& os, NarrowingPass::AnalysisType a) {
  switch (a) {
    case NarrowingPass::AnalysisType::kTernary:
      return os << "Ternary";
    case NarrowingPass::AnalysisType::kRange:
      return os << "Range";
    case NarrowingPass::AnalysisType::kRangeWithContext:
      return os << "Context";
    case NarrowingPass::AnalysisType::kRangeWithOptionalContext:
      return os << "OptionalContext";
  }
}

XLS_REGISTER_MODULE_INITIALIZER(narrowing_pass, {
  CHECK_OK(RegisterOptimizationPass<NarrowingPass>("narrow"));
  CHECK_OK(RegisterOptimizationPass<NarrowingPass>(
      "narrow(Ternary)", NarrowingPass::AnalysisType::kTernary,
      pass_config::kOptLevel));
  CHECK_OK(RegisterOptimizationPass<NarrowingPass>(
      "narrow(Range)", NarrowingPass::AnalysisType::kRange,
      pass_config::kOptLevel));
  CHECK_OK(RegisterOptimizationPass<NarrowingPass>(
      "narrow(Context)", NarrowingPass::AnalysisType::kRangeWithContext,
      pass_config::kOptLevel));
  CHECK_OK(RegisterOptimizationPass<NarrowingPass>(
      "narrow(OptionalContext)",
      NarrowingPass::AnalysisType::kRangeWithOptionalContext,
      pass_config::kOptLevel));
});

}  // namespace xls
