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
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
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
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/source_location.h"
#include "xls/ir/ternary.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/aliasing_query_engine.h"
#include "xls/passes/bit_count_query_engine.h"
#include "xls/passes/context_sensitive_range_query_engine.h"
#include "xls/passes/lazy_ternary_query_engine.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/partial_info_query_engine.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/predicate_dominator_analysis.h"
#include "xls/passes/predicate_state.h"
#include "xls/passes/proc_state_range_query_engine.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/stateless_query_engine.h"
#include "xls/passes/union_query_engine.h"

namespace xls {

namespace {

using AnalysisType = NarrowingPass::AnalysisType;

class SpecializedQueryEngines {
 public:
  SpecializedQueryEngines(AnalysisType type, PredicateDominatorAnalysis& pda,
                          AliasingQueryEngine& base)
      : type_(type), base_(base), pda_(pda) {}
  const QueryEngine& ForSelect(PredicateState state) const {
    if (type_ != AnalysisType::kRangeWithContext ||
        !base_.IsPredicatePossible(state)) {
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

  absl::Status AddAlias(Node* new_node, Node* alias_target) const {
    return base_.AddAlias(new_node, alias_target);
  }

 private:
  AnalysisType type_;
  AliasingQueryEngine& base_;
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
  if (node->OpIn({Op::kSignExt, Op::kZeroExt}) &&
      node->operand(0)->BitCountOrDie() == bit_count) {
    // Don't bother doing a bitslice of extend just simplify it right here.
    // Makes testing easier.
    // TODO(allight): We could also simplify the bitslice-extend in general
    // here.
    return node->operand(0);
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
    if (to_replace->Is<Literal>()) {
      // Don't replace literals since they can only be replaced with themselves
      // which is not useful.
      return false;
    }
    // If the node has no users, replacing it with a literal is pointless.
    if (to_replace->users().empty() &&
        !to_replace->function_base()->HasImplicitUse(to_replace)) {
      return false;
    }
    std::optional<SharedLeafTypeTree<TernaryVector>> ternary =
        query_engine.GetTernary(to_replace);
    if (!ternary.has_value()) {
      return false;
    }
    for (Type* leaf_type : ternary->leaf_types()) {
      if (leaf_type->IsToken()) {
        XLS_RETURN_IF_ERROR(NoChange());
        return false;
      }
    }
    for (const TernaryVector& ternary_vector : ternary->elements()) {
      if (!ternary_ops::IsFullyKnown(ternary_vector)) {
        XLS_RETURN_IF_ERROR(NoChange());
        return false;
      }
    }
    LeafTypeTree<Value> value_ltt = leaf_type_tree::Map<Value, TernaryVector>(
        ternary->AsView(), [](const TernaryVector& ternary_vector) -> Value {
          return Value(ternary_ops::ToKnownBitsValues(ternary_vector));
        });
    XLS_ASSIGN_OR_RETURN(Value value, LeafTypeTreeToValue(value_ltt.AsView()));
    XLS_RETURN_IF_ERROR(replace_with(value));
    VLOG(3) << absl::StreamFormat(
        "Range analysis found precise value for %s == %s %s, replacing with "
        "literal\n",
        to_replace->GetName(), value.ToString(), context);
    XLS_RETURN_IF_ERROR(Change());
    return true;
  }

  bool changed() const { return changed_; }

  absl::Status DefaultHandler(Node* node) override { return NoChange(); }
  absl::Status HandleArrayUpdate(ArrayUpdate* update) override {
    if (!update->assumed_in_bounds() &&
        ArrayIndicesAssumedInBounds(
            update, update->indices(),
            update->array_to_update()->GetType()->AsArrayOrDie())) {
      update->SetAssumedInBounds(true);
      VLOG(3) << "analysis proves that " << update
              << " does not require bounds checks";
      return Change();
    }
    return NoChange();
  }
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
      XLS_ASSIGN_OR_RETURN(
          Node * replacement,
          neg->ReplaceUsesWithNew<ExtendOp>(
              narrowed_input, neg->BitCountOrDie(), Op::kSignExt));
      return Change(/*original=*/neg, replacement);
    }
    // Slice then neg then extend the negated value.
    XLS_ASSIGN_OR_RETURN(Node * narrowed_input,
                         neg->function_base()->MakeNode<BitSlice>(
                             neg->loc(), neg->operand(UnOp::kArgOperand), 0,
                             unknown_segment + 1));
    XLS_ASSIGN_OR_RETURN(Node * narrowed_neg,
                         neg->function_base()->MakeNode<UnOp>(
                             neg->loc(), narrowed_input, Op::kNeg));
    XLS_ASSIGN_OR_RETURN(Node * replacement,
                         neg->ReplaceUsesWithNew<ExtendOp>(
                             narrowed_neg, neg->BitCountOrDie(), Op::kSignExt));
    return Change(/*original=*/neg, replacement);
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
      VLOG(3) << absl::StreamFormat(
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
      VLOG(3) << absl::StreamFormat(
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

  struct NarrowedArray {
    Literal* literal;
    TernaryVector pattern;
  };
  absl::StatusOr<std::optional<NarrowedArray>> NarrowLiteralArray(
      Literal* array_literal) {
    absl::Span<const Value> elements = array_literal->value().elements();
    const int64_t bit_count = elements[0].bits().bit_count();
    TernaryVector array_pattern =
        ternary_ops::BitsToTernary(elements[0].bits());
    for (const Value& element : elements.subspan(1)) {
      ternary_ops::UpdateWithIntersection(array_pattern, element.bits());
    }

    int64_t known_bits = ternary_ops::NumberOfKnownBits(array_pattern);
    if (known_bits == bit_count) {
      return NarrowedArray{
          .literal = nullptr,
          .pattern = array_pattern,
      };
    }
    if (known_bits == 0 || !splits_enabled_) {
      // Even if we have any known bits, we can't slice our array without
      // creating multiple ops per ArrayIndex.
      return std::nullopt;
    }

    std::vector<Value> narrowed_elements;
    narrowed_elements.reserve(elements.size());
    for (const Value& element : elements) {
      const Bits& bits = element.bits();
      InlineBitmap narrowed_bits(bits.bit_count() - known_bits);
      int64_t narrowed_idx = 0;
      for (int64_t i = 0; i < bits.bit_count(); ++i) {
        if (ternary_ops::IsUnknown(array_pattern[i])) {
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
    return NarrowedArray{
        .literal = narrowed_array_literal,
        .pattern = array_pattern,
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

    XLS_ASSIGN_OR_RETURN(std::optional<NarrowedArray> sliced_array,
                         NarrowLiteralArray(literal));
    if (!sliced_array.has_value()) {
      // This array can't be narrowed.
      return NoChange();
    }
    if (sliced_array->literal == nullptr) {
      // All entries are identical, so all ArrayIndex accesses can be replaced
      // with a single literal following the pattern.
      XLS_RET_CHECK(ternary_ops::IsFullyKnown(sliced_array->pattern));
      XLS_ASSIGN_OR_RETURN(
          Literal * constant_literal,
          literal->function_base()->MakeNode<Literal>(
              literal->loc(),
              Value(ternary_ops::ToKnownBitsValues(sliced_array->pattern))));
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

    // We should have at least some unknown bits, which will need to come from
    // the narrowed array literal.
    CHECK(!ternary_ops::IsFullyKnown(sliced_array->pattern));
    CHECK_NE(sliced_array->literal, nullptr);

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
                               array_index->loc(), sliced_array->literal,
                               array_index->indices()));
      XLS_ASSIGN_OR_RETURN(Node * array_index_value,
                           FillPattern(sliced_array->pattern, new_array_index));
      XLS_RETURN_IF_ERROR(user->ReplaceUsesWith(array_index_value));
    }

    return Change();
  }

  // If leading bits are known of the shift-value we can remove any where the
  // unknown bits cannot overwrite them. We will also narrow shift_amnt and
  // split the value based on the calculated ranges of shift_amnt.
  absl::Status HandleShll(BinOp* shll) override {
    Node* shift_value = shll->operand(0);
    Node* shift_amnt = shll->operand(1);
    // Narrow the shift amount.
    const QueryEngine& shift_amnt_qe =
        QueryEngineForNodeAndUser(shift_amnt, shll);
    if (shift_amnt_qe.IsAllZeros(shift_amnt)) {
      // Shift amount is zero. Replace with (slice of) input operand of shift.
      XLS_RETURN_IF_ERROR(shll->ReplaceUsesWith(shift_value));
      return Change();
    }
    // NB Shifting all the bits off the end unconditionally would be caught by
    // earlier checks for constant results.

    const QueryEngine& value_qe = QueryEngineForNodeAndUser(shift_value, shll);
    int64_t leading_value_signs = CountLeadingKnownSignBits(shift_value, shll);
    bool is_sign_bit_known = value_qe.IsMsbKnown(shift_value);
    auto replace_with_narrowed = [&](Node* narrow_shift_val,
                                     Node* narrow_shift_amnt) -> absl::Status {
      if (narrow_shift_val == shift_value && narrow_shift_amnt == shift_amnt) {
        return NoChange();
      }

      XLS_ASSIGN_OR_RETURN(
          Node * narrow_shift,
          shift_value->function_base()->MakeNodeWithName<BinOp>(
              shll->loc(), narrow_shift_val, narrow_shift_amnt, Op::kShll,
              shll->HasAssignedName()
                  ? absl::StrFormat("%s_narrowed", shll->GetNameView())
                  : ""));
      if (shift_value == narrow_shift_val) {
        XLS_RETURN_IF_ERROR(shll->ReplaceUsesWith(narrow_shift));
        return Change(shll, narrow_shift);
      }
      if (is_sign_bit_known) {
        XLS_ASSIGN_OR_RETURN(
            Node * leading,
            shll->function_base()->MakeNodeWithName<Literal>(
                shll->loc(),
                Value(SBits(
                    value_qe.GetKnownMsb(shift_value) ? -1 : 0,
                    shll->BitCountOrDie() - narrow_shift->BitCountOrDie())),
                shll->HasAssignedName()
                    ? absl::StrFormat("%s_leading", shll->GetNameView())
                    : ""));
        XLS_ASSIGN_OR_RETURN(
            Node * res, shll->ReplaceUsesWithNew<Concat>(
                            absl::Span<Node* const>{leading, narrow_shift}));
        return Change(shll, res);
      }
      XLS_ASSIGN_OR_RETURN(
          Node * res, shll->ReplaceUsesWithNew<ExtendOp>(
                          narrow_shift, shll->BitCountOrDie(), Op::kSignExt));
      return Change(shll, res);
    };

    Node* narrowed_shift_amnt;
    int64_t leading_amnt_zeros = CountLeadingKnownZeros(shift_amnt, shll);
    if (leading_amnt_zeros > 0) {
      XLS_ASSIGN_OR_RETURN(
          narrowed_shift_amnt,
          shll->function_base()->MakeNode<BitSlice>(
              shll->loc(), shift_amnt, /*start=*/0,
              /*width=*/shift_amnt->BitCountOrDie() - leading_amnt_zeros));
    } else {
      narrowed_shift_amnt = shift_amnt;
    }

    // If op is like (SSSSSSSABCD) << (X) where X+4 < value-width we can narrow
    // cutting off some of the high shift-value bits.
    // If sign bit isn't known we always want to keep it around to allow
    // sign-extend
    int64_t unknown_bits =
        (shift_value->BitCountOrDie() - leading_value_signs) +
        (is_sign_bit_known ? 0 : 1);
    if (leading_value_signs == 0 ||
        unknown_bits >= shift_value->BitCountOrDie()) {
      return replace_with_narrowed(shift_value, narrowed_shift_amnt);
    }
    Bits max_shift_bits = shift_amnt_qe.MaxUnsignedValue(shift_amnt);
    if (!max_shift_bits.FitsInNBitsUnsigned(63)) {
      // Potentially the entire shift_value is shifted out.
      return replace_with_narrowed(shift_value, narrowed_shift_amnt);
    }
    XLS_ASSIGN_OR_RETURN(int64_t max_shift, max_shift_bits.ToUint64());
    if (max_shift >= shift_value->BitCountOrDie()) {
      return replace_with_narrowed(shift_value, narrowed_shift_amnt);
    }
    int64_t max_result_unknown_bits = max_shift + unknown_bits;
    XLS_RET_CHECK_GT(max_shift, 0);
    XLS_RET_CHECK_GT(max_result_unknown_bits, 0);
    if (max_result_unknown_bits >= shift_value->BitCountOrDie()) {
      // Some bit we don't know the value of is shifted to the leading bit
      // location.
      return replace_with_narrowed(shift_value, narrowed_shift_amnt);
    }
    // Narrow the actual shift.
    XLS_ASSIGN_OR_RETURN(
        Node * narrowed_shift_value,
        shll->function_base()->MakeNodeWithName<BitSlice>(
            shll->loc(), shift_value, 0, max_result_unknown_bits,
            shift_value->HasAssignedName()
                ? absl::StrFormat("%s_narrowed_for_shll",
                                  shift_value->GetNameView())
                : ""));

    return replace_with_narrowed(narrowed_shift_value, narrowed_shift_amnt);
  }

  absl::Status HandleShra(BinOp* shra) override {
    return MaybeNarrowShiftRight(shra);
  }
  absl::Status HandleShrl(BinOp* shrl) override {
    return MaybeNarrowShiftRight(shrl);
  }

  // Shift right can be narrowed in several complimentary ways.
  //
  // K=bit has a known value
  // X=bit has an unknown value
  // 1=bit is true
  // 0=bit is false
  //
  // Imagine we have some computation like
  //
  // ```
  // shift_value = KK..K_XXXX_KK...K;
  // shift_amnt = ...; // range is (a, b)
  // logic = shrl(shift_value, shift_amnt);
  // arith = shra(shift_value, shift_amnt);
  // ```
  //
  // We can cut off the known top bits if they are all the same value and its a
  // arith shift or if they are all 0s. We append back the 'K' value if we cut
  // it down smaller than the actual result.
  //
  // TODO(allight): Choosing to narrow with adding arithmetic to the
  // shift-amount is something we should consider. It likely will almost always
  // be beneficial but its not a very narrow-ing transform and there might be
  // some tricky edge cases.
  absl::Status MaybeNarrowShiftRight(Node* shift) {
    XLS_RET_CHECK(shift->OpIn({Op::kShrl, Op::kShra})) << shift;
    bool is_arithmetic = shift->op() == Op::kShra;

    Node* shift_value = shift->operand(0);
    Node* shift_amnt = shift->operand(1);
    const QueryEngine& shift_amnt_qe =
        QueryEngineForNodeAndUser(shift_amnt, shift);
    if (shift_amnt_qe.IsAllZeros(shift_amnt)) {
      // Shift amount is zero. Replace with (slice of) input operand of shift.
      XLS_RETURN_IF_ERROR(shift->ReplaceUsesWith(shift_value));
      return Change();
    }
    int64_t leading_bit_cnt =
        is_arithmetic ? CountLeadingKnownSignBits(shift_value, shift)
                      : CountLeadingKnownZeros(shift_value, shift);
    // How many bits have values which aren't provably the sign-bit/zero.
    int64_t values_bit_cnt = shift_value->BitCountOrDie() - leading_bit_cnt;
    XLS_RET_CHECK_GE(values_bit_cnt, 0);
    if (values_bit_cnt == 0) {
      // value is all the fill bit value.
      XLS_RETURN_IF_ERROR(shift->ReplaceUsesWith(shift_value));
      return Change();
    }
    // How many bits we need to keep in the shift_value to be able to expand it
    // out the original value. For arithmetic we need to keep a sign-bit-valued
    // bit.
    int64_t to_ext_bit_count =
        is_arithmetic ? values_bit_cnt + 1 : values_bit_cnt;

    IntervalSet amnt_range = shift_amnt_qe.GetIntervals(shift_amnt).Get({});
    if (!amnt_range.IsEmpty() && amnt_range.LowerBound()->FitsInUint64() &&
        amnt_range.LowerBound()->ToUint64().value() > values_bit_cnt) {
      // Shift all non-sign-bit values out.
      if (is_arithmetic) {
        XLS_ASSIGN_OR_RETURN(
            Node * sign_bit,
            shift->function_base()->MakeNodeWithName<BitSlice>(
                shift_value->loc(), shift_value,
                shift_value->BitCountOrDie() - 1, 1,
                shift_value->HasAssignedName()
                    ? absl::StrFormat("%s_sign_bit", shift_value->GetNameView())
                    : ""));
        XLS_RETURN_IF_ERROR(
            shift
                ->ReplaceUsesWithNew<ExtendOp>(sign_bit, shift->BitCountOrDie(),
                                               Op::kSignExt)
                .status());
      } else {
        // TODO(allight): I think this will be dead since result should be
        // constant anyway.
        XLS_RETURN_IF_ERROR(shift
                                ->ReplaceUsesWithNew<Literal>(
                                    Value(UBits(0, shift->BitCountOrDie())))
                                .status());
      }
      return Change();
    }
    Node* narrowed_shift_amnt;
    int64_t leading_amnt_zeros = CountLeadingKnownZeros(shift_amnt, shift);
    if (leading_amnt_zeros > 0) {
      XLS_ASSIGN_OR_RETURN(
          narrowed_shift_amnt,
          shift->function_base()->MakeNode<BitSlice>(
              shift->loc(), shift_amnt, /*start=*/0,
              /*width=*/shift_amnt->BitCountOrDie() - leading_amnt_zeros));
      XLS_RETURN_IF_ERROR(Change());
    } else {
      narrowed_shift_amnt = shift_amnt;
    }
    if (to_ext_bit_count == shift->BitCountOrDie()) {
      // All bits are important to result, can't narrow the shift_value.
      if (narrowed_shift_amnt != shift_amnt) {
        // Still narrowed the shift-amount some.
        XLS_ASSIGN_OR_RETURN(
            Node * replacment,
            shift->ReplaceUsesWithNew<BinOp>(shift_value, narrowed_shift_amnt,
                                             shift->op()));
        return Change(/*original=*/shift, replacment);
      } else {
        return NoChange();
      }
    }
    // We can narrow the shift value to only those bits that aren't the sign
    // bit.
    XLS_ASSIGN_OR_RETURN(
        Node * narrowed_shift_value,
        shift->function_base()->MakeNodeWithName<BitSlice>(
            shift->loc(), shift_value, /*start=*/0,
            /*width=*/to_ext_bit_count,
            shift_value->HasAssignedName()
                ? absl::StrFormat("%s_shift_bits", shift_value->GetNameView())
                : ""));
    XLS_ASSIGN_OR_RETURN(
        Node * narrowed_shift,
        shift->function_base()->MakeNodeWithName<BinOp>(
            shift->loc(), narrowed_shift_value, narrowed_shift_amnt,
            shift->op(),
            shift->HasAssignedName()
                ? absl::StrFormat("%s_narrowed", shift->GetNameView())
                : ""));
    XLS_ASSIGN_OR_RETURN(Node * replacment,
                         shift->ReplaceUsesWithNew<ExtendOp>(
                             narrowed_shift, shift->BitCountOrDie(),
                             is_arithmetic ? Op::kSignExt : Op::kZeroExt));
    return Change(/*original=*/shift, replacment);
  }

  // This is basically SHRL with a truncate. We can simplify a bit since OOB
  // requests are always 0s anyway so just get rid of all high-bit zeros. We can
  // futher get rid of any bits beyond max_start + width.
  //
  // We can also narrow the start value.
  //
  // Finally if the number of interesting bits is less then the slice size we
  // can slice to a smaller size and extend.
  absl::Status HandleDynamicBitSlice(DynamicBitSlice* slice) override {
    int64_t leading_start_zeros = CountLeadingKnownZeros(slice->start(), slice);
    int64_t leading_to_slice_zeros =
        CountLeadingKnownZeros(slice->to_slice(), slice);
    int64_t interesting_to_slice_bits =
        slice->to_slice()->BitCountOrDie() - leading_to_slice_zeros;

    // Check for known-value. Since other passes use less precise qes we can end
    // up here.
    const QueryEngine& slice_start_qe =
        QueryEngineForNodeAndUser(slice->start(), slice);
    if (auto known_slice_start =
            slice_start_qe.KnownValueAsBits(slice->start());
        known_slice_start.has_value()) {
      if (!known_slice_start->FitsInInt64Unsigned() ||
          *known_slice_start->UnsignedToInt64() >=
              slice->to_slice()->BitCountOrDie()) {
        // The dynamic slice is guaranteed to start after the end of the
        // `to_slice` value, so it will always be zero.
        XLS_ASSIGN_OR_RETURN(Node * replaced,
                             slice->ReplaceUsesWithNew<Literal>(
                                 Value(UBits(0, slice->BitCountOrDie()))));
        return Change(slice, replaced);
      }
      int64_t start = *known_slice_start->UnsignedToInt64();
      int64_t width =
          std::min(slice->width(), slice->to_slice()->BitCountOrDie() - start);
      Node* replaced;
      if (width == slice->width()) {
        XLS_ASSIGN_OR_RETURN(replaced, slice->ReplaceUsesWithNew<BitSlice>(
                                           slice->to_slice(), start, width));
      } else {
        XLS_ASSIGN_OR_RETURN(
            Node * static_slice,
            slice->function_base()->MakeNodeWithName<BitSlice>(
                slice->loc(), slice->to_slice(), start, width,
                slice->HasAssignedName()
                    ? absl::StrFormat("%s_slice", slice->GetNameView())
                    : ""));
        XLS_ASSIGN_OR_RETURN(
            replaced, slice->ReplaceUsesWithNew<ExtendOp>(
                          static_slice, slice->BitCountOrDie(), Op::kZeroExt));
      }
      return Change(slice, replaced);
    }
    Bits max_start = slice_start_qe.MaxUnsignedValue(slice->start());

    int64_t start_bit_width =
        1 + std::max(max_start.bit_count(),
                     Bits::MinBitCountUnsigned(slice->width()));
    Bits max_sliced_start_idx =
        bits_ops::Add(bits_ops::ZeroExtend(max_start, start_bit_width),
                      UBits(slice->width(), start_bit_width));
    if (max_sliced_start_idx.FitsInNBitsUnsigned(63)) {
      XLS_ASSIGN_OR_RETURN(int64_t max_sliced_val,
                           max_sliced_start_idx.ToUint64());
      interesting_to_slice_bits =
          std::min(interesting_to_slice_bits, max_sliced_val);
    }
    if (interesting_to_slice_bits >= slice->to_slice()->BitCountOrDie() &&
        leading_start_zeros == 0) {
      return NoChange();
    }
    Node* narrowed_start;
    Node* narrowed_to_slice;
    if (leading_start_zeros != 0) {
      XLS_ASSIGN_OR_RETURN(
          narrowed_start,
          slice->function_base()->MakeNodeWithName<BitSlice>(
              slice->start()->loc(), slice->start(), /*start=*/0,
              /*width=*/slice->start()->BitCountOrDie() - leading_start_zeros,
              slice->start()->HasAssignedName()
                  ? absl::StrFormat("%s_narrowed",
                                    slice->start()->GetNameView())
                  : ""));
    } else {
      narrowed_start = slice->start();
    }
    if (interesting_to_slice_bits == 0) {
      // We can just return a zero of the appropriate size.
      XLS_ASSIGN_OR_RETURN(Node * res, slice->ReplaceUsesWithNew<Literal>(
                                           ZeroOfType(slice->GetType())));
      return Change(slice, res);
    }
    if (interesting_to_slice_bits < slice->to_slice()->BitCountOrDie()) {
      XLS_ASSIGN_OR_RETURN(
          narrowed_to_slice,
          slice->function_base()->MakeNodeWithName<BitSlice>(
              slice->to_slice()->loc(), slice->to_slice(), /*start=*/0,
              /*width=*/interesting_to_slice_bits,
              slice->to_slice()->HasAssignedName()
                  ? absl::StrFormat("%s_narrowed",
                                    slice->to_slice()->GetNameView())
                  : ""));
    } else {
      narrowed_to_slice = slice->to_slice();
    }
    int64_t narrow_width =
        std::min(narrowed_to_slice->BitCountOrDie(), slice->width());
    if (narrow_width == slice->width()) {
      XLS_ASSIGN_OR_RETURN(
          Node * res, slice->ReplaceUsesWithNew<DynamicBitSlice>(
                          narrowed_to_slice, narrowed_start, slice->width()));
      return Change(slice, res);
    }
    // We can narrow to a smaller slice then extend.
    // TODO(allight): This dyn-slice is exactly identical to a SHRL. Should we
    // just emit that instead?
    // TODO(allight): We probably want to investigate doing this in
    // strength-reduction or somewhere else appropriate.
    XLS_ASSIGN_OR_RETURN(
        Node * dyn_slice,
        slice->function_base()->MakeNodeWithName<DynamicBitSlice>(
            slice->loc(), narrowed_to_slice, narrowed_start, narrow_width,
            slice->HasAssignedName()
                ? absl::StrFormat("%s_narrowed", slice->GetNameView())
                : ""));
    XLS_ASSIGN_OR_RETURN(Node * res,
                         slice->ReplaceUsesWithNew<ExtendOp>(
                             dyn_slice, slice->width(), Op::kZeroExt));
    return Change(slice, res);
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
      return Change(/*original=*/decode, /*replacement=*/narrowed_decode);
    }
    XLS_ASSIGN_OR_RETURN(Node * replacement,
                         decode->ReplaceUsesWithNew<ExtendOp>(
                             narrowed_decode, result_width, Op::kZeroExt));
    return Change(/*original=*/decode, /*replacement=*/replacement);
  }

  absl::Status HandleSub(BinOp* sub) override {
    VLOG(3) << "Trying to narrow sub: " << sub->ToString();
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
    auto narrow_known_sign = [&](int64_t known_leading,
                                 Op extend) -> absl::Status {
      XLS_RET_CHECK_GT(known_leading, 0);
      int64_t required_bits = bit_count - known_leading;
      XLS_ASSIGN_OR_RETURN(Node * new_lhs, MaybeNarrow(lhs, required_bits));
      XLS_ASSIGN_OR_RETURN(Node * new_rhs, MaybeNarrow(rhs, required_bits));
      XLS_ASSIGN_OR_RETURN(Node * new_sub,
                           sub->function_base()->MakeNode<BinOp>(
                               sub->loc(), new_lhs, new_rhs, Op::kSub));
      XLS_ASSIGN_OR_RETURN(
          Node * replacement,
          sub->ReplaceUsesWithNew<ExtendOp>(new_sub, bit_count, extend));
      return Change(/*original=*/sub, /*replacement=*/replacement);
    };
    if (leading_zeros != 0) {
      // Known positive result, just snip off leading bits.
      return narrow_known_sign(leading_zeros, Op::kZeroExt);
    }
    if (leading_ones > 1) {
      // known negative result, need to keep the sign bit.
      return narrow_known_sign(leading_ones - 1, Op::kSignExt);
    }
    // Perform actual sign-bit count analysis
    int64_t lhs_leading_sign_bits =
        CountLeadingKnownSignBits(lhs, /*user=*/sub);
    int64_t rhs_leading_sign_bits =
        CountLeadingKnownSignBits(rhs, /*user=*/sub);
    // If the arguments are potentially different signs overflowing into the
    // sign bit is possible so we need an extra bit of width.
    // TODO(allight): Other conditions can also stop the overflow related to the
    // actual ranges.
    auto lhs_qe = QueryEngineForNodeAndUser(lhs, sub);
    auto rhs_qe = QueryEngineForNodeAndUser(rhs, sub);
    int64_t overflow_bit =
        lhs_qe.IsMsbKnown(lhs) && rhs_qe.IsMsbKnown(rhs) &&
                lhs_qe.GetKnownMsb(lhs) == rhs_qe.GetKnownMsb(rhs)
            ? 0
            : 1;
    int64_t min_signed_size =
        1 + overflow_bit + sub->BitCountOrDie() -
        std::min(lhs_leading_sign_bits, rhs_leading_sign_bits);
    if (min_signed_size < sub->BitCountOrDie()) {
      if (min_signed_size == 0) {
        // Implies that the sub is actually unused somehow, can't do anything in
        // any case.
        return NoChange();
      }
      XLS_ASSIGN_OR_RETURN(
          Node * narrowed_lhs,
          lhs->function_base()->MakeNode<BitSlice>(lhs->loc(), lhs, /*start=*/0,
                                                   /*width=*/min_signed_size));
      XLS_ASSIGN_OR_RETURN(
          Node * narrowed_rhs,
          rhs->function_base()->MakeNode<BitSlice>(rhs->loc(), rhs, /*start=*/0,
                                                   /*width=*/min_signed_size));
      XLS_ASSIGN_OR_RETURN(
          Node * narrowed_sub,
          sub->function_base()->MakeNode<BinOp>(sub->loc(), narrowed_lhs,
                                                narrowed_rhs, Op::kSub));
      XLS_ASSIGN_OR_RETURN(Node * replacement,
                           sub->ReplaceUsesWithNew<ExtendOp>(
                               narrowed_sub, bit_count, Op::kSignExt));
      return Change(/*original=*/sub, /*replacement=*/replacement);
    }
    return NoChange();
  }
  absl::Status HandleAdd(BinOp* add) override {
    VLOG(3) << "Trying to narrow add: " << add->ToString();

    XLS_RET_CHECK_EQ(add->op(), Op::kAdd);

    Node* lhs = add->operand(0);
    Node* rhs = add->operand(1);
    const int64_t bit_count = add->BitCountOrDie();
    if (lhs->BitCountOrDie() != rhs->BitCountOrDie()) {
      return NoChange();
    }
    int64_t lhs_lead_zero = CountLeadingKnownZeros(lhs, /*user=*/add);
    int64_t rhs_lead_zero = CountLeadingKnownZeros(rhs, /*user=*/add);
    int64_t common_leading_zeros = std::min(lhs_lead_zero, rhs_lead_zero);

    int64_t lhs_lead_sign = CountLeadingKnownSignBits(lhs, /*user=*/add);
    int64_t rhs_lead_sign = CountLeadingKnownSignBits(rhs, /*user=*/add);
    int64_t common_leading_sign = std::min(lhs_lead_sign, rhs_lead_sign);

    auto make_narrow_add = [&](Node* lhs, Node* rhs, int64_t width,
                               Op extend) -> absl::Status {
      XLS_RET_CHECK_LT(width, add->BitCountOrDie());
      XLS_ASSIGN_OR_RETURN(Node * narrowed_lhs,
                           lhs->function_base()->MakeNode<BitSlice>(
                               lhs->loc(), lhs, /*start=*/0, /*width=*/width));
      XLS_ASSIGN_OR_RETURN(Node * narrowed_rhs,
                           rhs->function_base()->MakeNode<BitSlice>(
                               rhs->loc(), rhs, /*start=*/0, /*width=*/width));
      XLS_ASSIGN_OR_RETURN(
          Node * narrowed_add,
          add->function_base()->MakeNode<BinOp>(add->loc(), narrowed_lhs,
                                                narrowed_rhs, Op::kAdd));
      XLS_ASSIGN_OR_RETURN(
          Node * new_add,
          add->ReplaceUsesWithNew<ExtendOp>(narrowed_add, bit_count, extend));
      return Change(/*original=*/add, /*replacement=*/new_add);
    };
    if (common_leading_zeros > 1 &&
        common_leading_zeros >= common_leading_sign) {
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
      return make_narrow_add(lhs, rhs, narrowed_bit_count, Op::kZeroExt);
    }
    if (common_leading_sign > 1) {
      // Narrow the add removing all but one of the known-sign-bit leading
      // bits. Example:
      //
      //    SSSXXX + SSSSYY => { SS, SXXX + SSYY }
      //
      if (common_leading_sign == bit_count) {
        // Both values are either 0 or -1. Just narrow to 2 bits so all results
        // (0, -2, -1) can be represented.
        if (bit_count <= 2) {
          return NoChange();
        }
        return make_narrow_add(lhs, rhs, 2, Op::kSignExt);
      }
      // Need an extra bit to ensure that overflow won't happen in cases where
      // the sign of the arguments are known to be potentially the same.
      // If the operands have the same sign then overflow can reach the sign
      // bit. If they are definitely not the same sign then overflow into the
      // sign bit is not possible.
      auto lhs_qe = QueryEngineForNodeAndUser(lhs, add);
      auto rhs_qe = QueryEngineForNodeAndUser(rhs, add);
      int64_t overflow_bit =
          lhs_qe.IsMsbKnown(lhs) && rhs_qe.IsMsbKnown(rhs) &&
                  lhs_qe.GetKnownMsb(lhs) != rhs_qe.GetKnownMsb(rhs)
              ? 0
              : 1;
      int64_t narrowed_bit_count =
          bit_count - common_leading_sign + 1 + overflow_bit;
      XLS_RET_CHECK_LE(narrowed_bit_count, bit_count);
      // Due to needing an extra overflow bit sometimes we need either 2 or 1
      // leading bits to do a narrow.
      if (narrowed_bit_count != bit_count) {
        return make_narrow_add(lhs, rhs, narrowed_bit_count, Op::kSignExt);
      }
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
    VLOG(3) << "Trying to narrow multiply: " << mul->ToString();

    XLS_RET_CHECK(mul->op() == Op::kSMul || mul->op() == Op::kUMul);

    Node* lhs = mul->operand(0);
    Node* rhs = mul->operand(1);
    const int64_t result_bit_count = mul->BitCountOrDie();
    const int64_t lhs_bit_count = lhs->BitCountOrDie();
    const int64_t rhs_bit_count = rhs->BitCountOrDie();
    VLOG(3) << absl::StreamFormat(
        "  result_bit_count = %d, lhs_bit_count = %d, rhs_bit_count = %d",
        result_bit_count, lhs_bit_count, rhs_bit_count);

    // The result can be unconditionally narrowed to the sum of the operand
    // widths, then zero/sign extended.
    if (result_bit_count > lhs_bit_count + rhs_bit_count) {
      VLOG(3) << "Result is wider than sum of operands. Narrowing multiply.";
      XLS_ASSIGN_OR_RETURN(
          Node * narrowed_mul,
          mul->function_base()->MakeNode<ArithOp>(
              mul->loc(), lhs, rhs,
              /*width=*/lhs_bit_count + rhs_bit_count, mul->op()));
      XLS_ASSIGN_OR_RETURN(Node * replacement,
                           MaybeExtend(narrowed_mul, result_bit_count,
                                       /*is_signed=*/mul->op() == Op::kSMul));
      XLS_RETURN_IF_ERROR(mul->ReplaceUsesWith(replacement));
      return Change(/*original=*/mul, /*replacement=*/replacement);
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
      XLS_ASSIGN_OR_RETURN(
          Node * replacement,
          mul->ReplaceUsesWithNew<ArithOp>(narrowed_lhs, narrowed_rhs,
                                           result_bit_count, mul->op()));
      return Change(/*original=*/mul, /*replacement=*/replacement);
    }

    // A multiply where the result and both operands are the same width is the
    // same operation whether it is signed or unsigned.
    bool is_sign_agnostic =
        result_bit_count == lhs_bit_count && result_bit_count == rhs_bit_count;

    // Zero-extended operands of unsigned multiplies can be narrowed.
    bool can_narrow_unsigned = mul->op() == Op::kUMul || is_sign_agnostic;
    int64_t unsigned_narrowed_lhs_width =
        can_narrow_unsigned ? UnsignedNarrowedWidth(lhs, /*user=*/mul)
                            : lhs_bit_count;
    int64_t unsigned_narrowed_rhs_width =
        can_narrow_unsigned ? UnsignedNarrowedWidth(rhs, /*user=*/mul)
                            : rhs_bit_count;
    can_narrow_unsigned = unsigned_narrowed_lhs_width != lhs_bit_count ||
                          unsigned_narrowed_rhs_width != rhs_bit_count;
    absl::int128 unsigned_narrowed_complexity =
        absl::int128(unsigned_narrowed_lhs_width) *
        absl::int128(unsigned_narrowed_rhs_width);

    // Sign-extended operands of signed multiplies can be narrowed by
    // replacing the operand of the multiply with the value before
    // sign-extension.
    bool can_narrow_signed = mul->op() == Op::kSMul || is_sign_agnostic;
    int64_t signed_narrowed_lhs_width =
        can_narrow_signed ? SignedNarrowedWidth(lhs, /*user=*/mul)
                          : lhs_bit_count;
    int64_t signed_narrowed_rhs_width =
        can_narrow_signed ? SignedNarrowedWidth(rhs, /*user=*/mul)
                          : rhs_bit_count;
    can_narrow_signed = signed_narrowed_lhs_width != lhs_bit_count ||
                        signed_narrowed_rhs_width != rhs_bit_count;
    absl::int128 signed_narrowed_complexity =
        absl::int128(signed_narrowed_lhs_width) *
        absl::int128(signed_narrowed_rhs_width);

    if (can_narrow_unsigned &&
        unsigned_narrowed_complexity <= signed_narrowed_complexity) {
      XLS_ASSIGN_OR_RETURN(std::optional<Node*> narrowed_lhs,
                           MaybeNarrowUnsignedOperand(lhs, mul));
      XLS_ASSIGN_OR_RETURN(std::optional<Node*> narrowed_rhs,
                           MaybeNarrowUnsignedOperand(rhs, mul));
      XLS_RET_CHECK(narrowed_lhs.has_value() || narrowed_rhs.has_value());
      XLS_ASSIGN_OR_RETURN(
          Node * replacement,
          mul->ReplaceUsesWithNew<ArithOp>(narrowed_lhs.value_or(lhs),
                                           narrowed_rhs.value_or(rhs),
                                           result_bit_count, Op::kUMul));
      return Change(/*original=*/mul, /*replacement=*/replacement);
    }

    if (can_narrow_signed) {
      XLS_ASSIGN_OR_RETURN(std::optional<Node*> narrowed_lhs,
                           MaybeNarrowSignedOperand(lhs, mul));
      XLS_ASSIGN_OR_RETURN(std::optional<Node*> narrowed_rhs,
                           MaybeNarrowSignedOperand(rhs, mul));
      XLS_RET_CHECK(narrowed_lhs.has_value() || narrowed_rhs.has_value());
      XLS_ASSIGN_OR_RETURN(
          Node * replacement,
          mul->ReplaceUsesWithNew<ArithOp>(narrowed_lhs.value_or(lhs),
                                           narrowed_rhs.value_or(rhs),
                                           result_bit_count, Op::kSMul));
      return Change(/*original=*/mul, /*replacement=*/replacement);
    }

    int64_t left_trailing_zeros = CountTrailingKnownZeros(lhs, mul);
    int64_t right_trailing_zeros = CountTrailingKnownZeros(rhs, mul);
    int64_t known_zero_bits = left_trailing_zeros + right_trailing_zeros;
    if (known_zero_bits >= result_bit_count) {
      // All result bits are in the trailing 0s so the result is a constant 0.
      XLS_RETURN_IF_ERROR(
          mul->ReplaceUsesWithNew<Literal>(Value(UBits(0, result_bit_count)))
              .status());
      return Change();
    }
    if (known_zero_bits > 0) {
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
              mul->loc(), new_left, new_right, mul->width() - known_zero_bits,
              mul->op(), mul->GetName() + "_NarrowedMult_"));
      XLS_ASSIGN_OR_RETURN(Node * zeros,
                           mul->function_base()->MakeNodeWithName<Literal>(
                               mul->loc(), Value(Bits(known_zero_bits)),
                               mul->GetName() + "_TrailingBits_"));
      XLS_ASSIGN_OR_RETURN(Node * replacement,
                           mul->ReplaceUsesWithNew<Concat>(
                               absl::Span<Node* const>{new_mul, zeros}));
      return Change(/*original=*/mul, /*replacement=*/replacement);
    }

    return NoChange();
  }
  absl::Status HandleArrayIndex(ArrayIndex* array_index) override {
    bool changed = false;
    bool assumed_in_bounds = array_index->assumed_in_bounds();

    if (!assumed_in_bounds &&
        ArrayIndicesAssumedInBounds(
            array_index, array_index->indices(),
            array_index->array()->GetType()->AsArrayOrDie())) {
      VLOG(3) << "analysis proves that " << array_index
              << " does not require bounds checks";
      assumed_in_bounds = true;
    }
    if (analysis_ == AnalysisType::kRange &&
        options_.convert_array_index_to_select.has_value()) {
      int64_t threshold = options_.convert_array_index_to_select.value();
      XLS_ASSIGN_OR_RETURN(
          std::optional<Node*> chain_select,
          MaybeConvertArrayIndexToSelect(array_index, threshold));
      if (chain_select) {
        return Change(/*original=*/array_index, /*replacement=*/*chain_select);
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

      const QueryEngine& query_engine =
          specialized_query_engine_.ForNode(index);
      if (std::optional<Bits> bits_index = query_engine.KnownValueAsBits(index);
          bits_index.has_value()) {
        Bits new_bits_index = *bits_index;
        if (bits_ops::UGreaterThanOrEqual(*bits_index, array_size)) {
          // Index is out-of-bounds. Replace with a (potentially narrower)
          // index equal to the first out-of-bounds element.
          new_bits_index =
              UBits(array_size, Bits::MinBitCountUnsigned(array_size));
        } else if (bits_index->bit_count() > min_index_width) {
          // Index is in-bounds and is wider than necessary to index the
          // entire array. Replace with a literal which is perfectly sized
          // (width) to index the whole array.
          XLS_ASSIGN_OR_RETURN(int64_t int_index, bits_index->ToUint64());
          new_bits_index = UBits(int_index, min_index_width);
        }
        Node* new_index = index;
        if (*bits_index != new_bits_index) {
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
      XLS_ASSIGN_OR_RETURN(
          Node * new_idx,
          array_index->ReplaceUsesWithNew<ArrayIndex>(
              array_index->array(), new_indices, assumed_in_bounds));
      return Change(/*original=*/array_index, /*replacement=*/new_idx);
    }
    if (assumed_in_bounds != array_index->assumed_in_bounds()) {
      array_index->SetAssumedInBounds(assumed_in_bounds);
      // The pointer doesn't change so no need to add an alias.
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
    VLOG(3) << "Trying to narrow multiply: " << mul->ToString();

    XLS_RET_CHECK(mul->op() == Op::kSMulp || mul->op() == Op::kUMulp);

    Node* lhs = mul->operand(0);
    Node* rhs = mul->operand(1);
    const int64_t result_bit_count = mul->width();
    const int64_t lhs_bit_count = lhs->BitCountOrDie();
    const int64_t rhs_bit_count = rhs->BitCountOrDie();
    VLOG(3) << absl::StreamFormat(
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
      VLOG(3) << "Result is wider than sum of operands. Narrowing multiply.";
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
      return Change(/*original=*/*add_immediately_after, /*to*/ extended_sum);
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
      XLS_ASSIGN_OR_RETURN(
          Node * narrow_mul,
          mul->ReplaceUsesWithNew<PartialProductOp>(
              narrowed_lhs, narrowed_rhs, result_bit_count, mul->op()));
      return Change(/*original=*/mul, /*replacement=*/narrow_mul);
    }

    // A multiply where the result and both operands are the same width is the
    // same operation whether it is signed or unsigned.
    bool is_sign_agnostic =
        result_bit_count == lhs_bit_count && result_bit_count == rhs_bit_count;

    // Zero-extended operands of unsigned multiplies can be narrowed.
    bool can_narrow_unsigned = mul->op() == Op::kUMulp || is_sign_agnostic;
    int64_t unsigned_narrowed_lhs_width =
        can_narrow_unsigned ? UnsignedNarrowedWidth(lhs, /*user=*/mul)
                            : lhs_bit_count;
    int64_t unsigned_narrowed_rhs_width =
        can_narrow_unsigned ? UnsignedNarrowedWidth(rhs, /*user=*/mul)
                            : rhs_bit_count;
    can_narrow_unsigned = unsigned_narrowed_lhs_width != lhs_bit_count ||
                          unsigned_narrowed_rhs_width != rhs_bit_count;
    absl::int128 unsigned_narrowed_complexity =
        absl::int128(unsigned_narrowed_lhs_width) *
        absl::int128(unsigned_narrowed_rhs_width);

    // Sign-extended operands of signed multiplies can be narrowed by
    // replacing the operand of the multiply with the value before
    // sign-extension.
    bool can_narrow_signed = mul->op() == Op::kSMulp || is_sign_agnostic;
    int64_t signed_narrowed_lhs_width =
        can_narrow_signed ? SignedNarrowedWidth(lhs, /*user=*/mul)
                          : lhs_bit_count;
    int64_t signed_narrowed_rhs_width =
        can_narrow_signed ? SignedNarrowedWidth(rhs, /*user=*/mul)
                          : rhs_bit_count;
    can_narrow_signed = signed_narrowed_lhs_width != lhs_bit_count ||
                        signed_narrowed_rhs_width != rhs_bit_count;
    absl::int128 signed_narrowed_complexity =
        absl::int128(signed_narrowed_lhs_width) *
        absl::int128(signed_narrowed_rhs_width);

    if (can_narrow_unsigned &&
        unsigned_narrowed_complexity <= signed_narrowed_complexity) {
      XLS_ASSIGN_OR_RETURN(std::optional<Node*> narrowed_lhs,
                           MaybeNarrowUnsignedOperand(lhs, mul));
      XLS_ASSIGN_OR_RETURN(std::optional<Node*> narrowed_rhs,
                           MaybeNarrowUnsignedOperand(rhs, mul));
      XLS_RET_CHECK(narrowed_lhs.has_value() || narrowed_rhs.has_value());
      XLS_ASSIGN_OR_RETURN(
          Node * narrow_mul,
          mul->ReplaceUsesWithNew<PartialProductOp>(
              narrowed_lhs.value_or(lhs), narrowed_rhs.value_or(rhs),
              result_bit_count, Op::kUMulp));
      return Change(/*original=*/mul, /*replacement=*/narrow_mul);
    }

    if (can_narrow_signed) {
      XLS_ASSIGN_OR_RETURN(std::optional<Node*> narrowed_lhs,
                           MaybeNarrowSignedOperand(lhs, mul));
      XLS_ASSIGN_OR_RETURN(std::optional<Node*> narrowed_rhs,
                           MaybeNarrowSignedOperand(rhs, mul));
      XLS_RET_CHECK(narrowed_lhs.has_value() || narrowed_rhs.has_value());
      XLS_ASSIGN_OR_RETURN(
          Node * narrow_mul,
          mul->ReplaceUsesWithNew<PartialProductOp>(
              narrowed_lhs.value_or(lhs), narrowed_rhs.value_or(rhs),
              result_bit_count, Op::kSMulp));
      return Change(/*original=*/mul, /*replacement=*/narrow_mul);
    }

    // TODO(meheff): If either lhs or rhs has trailing zeros, the multiply can
    // be narrowed and the result concatenated with trailing zeros.

    return NoChange();
  }

 private:
  bool ArrayIndicesAssumedInBounds(Node* user, absl::Span<Node* const> indices,
                                   ArrayType* array_type) {
    Type* ty = array_type;
    for (Node* n : indices) {
      const QueryEngine& node_query_engine =
          specialized_query_engine_.ForNode(user);
      if (bits_ops::UGreaterThanOrEqual(node_query_engine.MaxUnsignedValue(n),
                                        ty->AsArrayOrDie()->size())) {
        return false;
      }
      ty = ty->AsArrayOrDie()->element_type();
    }
    return true;
  }

  // Gets a query engine which takes into account the node and its user.
  UnownedConstUnionQueryEngine QueryEngineForNodeAndUser(
      Node* node, std::optional<Node*> user) const {
    std::vector<const QueryEngine*> engines{
        &specialized_query_engine_.ForNode(node)};
    if (analysis_ == AnalysisType::kRangeWithContext && user.has_value()) {
      engines.push_back(&specialized_query_engine_.ForNode(*user));
    }
    return UnownedConstUnionQueryEngine(std::move(engines));
  }

  // Return the number of trailing known zeros in the given nodes values when
  // used as an argument to 'user'. If user is std::nullopt the context isn't
  // used.
  int64_t CountTrailingKnownZeros(Node* node, std::optional<Node*> user) const {
    CHECK(node->GetType()->IsBits());
    CHECK(user != nullptr);
    int64_t trailing_zeros = 0;
    auto qe = QueryEngineForNodeAndUser(node, user);
    for (int64_t i = 0; i < node->BitCountOrDie(); ++i) {
      if (!qe.IsZero(TreeBitLocation(node, i))) {
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
    return QueryEngineForNodeAndUser(node, user)
        .KnownLeadingZeros(node)
        .value_or(0);
  }

  // Return the number of leading known ones in the given nodes values when
  // the nodes value is used as an argument to 'user'. 'user' must not be
  // null.
  int64_t CountLeadingKnownOnes(Node* node, std::optional<Node*> user) const {
    CHECK(node->GetType()->IsBits());
    CHECK(user != nullptr);
    return QueryEngineForNodeAndUser(node, user)
        .KnownLeadingOnes(node)
        .value_or(0);
  }

  // Return the number of leading known sign-bit equal values in the given nodes
  // values when used as an argument to 'user'. If user is std::nullopt the
  // context isn't used.
  int64_t CountLeadingKnownSignBits(Node* node,
                                    std::optional<Node*> user) const {
    CHECK(node->GetType()->IsBits());
    CHECK(user != nullptr);
    return QueryEngineForNodeAndUser(node, user)
        .KnownLeadingSignBits(node)
        .value_or(1);
  }

  absl::StatusOr<IntervalSetTree> GetIntervals(Node* node) const {
    const QueryEngine& node_query_engine =
        specialized_query_engine_.ForNode(node);
    XLS_RET_CHECK(node_query_engine.IsTracked(node));
    return node_query_engine.GetIntervals(node);
  }

  int64_t UnsignedNarrowedWidth(Node* operand, Node* user) {
    return operand->BitCountOrDie() - CountLeadingKnownZeros(operand, user);
  }

  absl::StatusOr<std::optional<Node*>> MaybeNarrowUnsignedOperand(Node* operand,
                                                                  Node* user) {
    int64_t narrowed_width = UnsignedNarrowedWidth(operand, user);
    if (narrowed_width == operand->BitCountOrDie()) {
      return std::nullopt;
    }
    return MaybeNarrow(operand, narrowed_width);
  }

  int64_t SignedNarrowedWidth(Node* operand, Node* user) {
    int64_t leading_sign_bits = CountLeadingKnownSignBits(operand, user);
    if (leading_sign_bits > 1) {
      // Operand has more than one leading sign bit, something like:
      //    operand = 0000XXXX
      // or
      //    operand = 1111XXXX
      // This is equivalent to:
      //    operand = signextend(0XXXX)
      // or
      //    operand = signextend(1XXXX)
      // respectively, so we can drop the first (N-1) bits.
      return operand->BitCountOrDie() - leading_sign_bits + 1;
    }
    return operand->BitCountOrDie();
  }

  absl::StatusOr<std::optional<Node*>> MaybeNarrowSignedOperand(Node* operand,
                                                                Node* user) {
    int64_t narrowed_width = SignedNarrowedWidth(operand, user);
    if (narrowed_width == operand->BitCountOrDie()) {
      return std::nullopt;
    }
    return MaybeNarrow(operand, narrowed_width);
  }

  // If an ArrayIndex has a small set of possible indexes (based on range
  // analysis), replace it with a small select chain.
  absl::StatusOr<std::optional<Node*>> MaybeConvertArrayIndexToSelect(
      ArrayIndex* array_index, int64_t threshold) {
    if (array_index->indices().empty()) {
      return std::nullopt;
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
      return QueryEngineForNodeAndUser(value, array_index)
          .GetIntervals(value)
          .Get({});
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
          return std::nullopt;
        }
        if (std::optional<int64_t> size = index_interval_set.Size()) {
          index_space_size *= (*size);
        } else {
          return std::nullopt;
        }
      }

      // This means that the indexes are literals; we shouldn't run this
      // optimization in this case because we generate ArrayIndexes with
      // literals as part of this optimization, so failing to skip this would
      // result in an infinite amount of code being generated.
      if (index_space_size == 1) {
        return std::nullopt;
      }

      if (index_space_size > threshold) {
        return std::nullopt;
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
      return std::nullopt;
    }

    XLS_RETURN_IF_ERROR(array_index->ReplaceUsesWith(rest_of_chain));

    return rest_of_chain;
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
  // Record a change as having occurred rewritten node 'from' to 'to'. This is
  // not necessary when the 'to' node is a literal. If the 'to' node is only
  // used by a single other node noting it is not needed.
  absl::Status Change(Node* original, Node* replacement) {
    XLS_RETURN_IF_ERROR(
        specialized_query_engine_.AddAlias(replacement, original));
    changed_ = true;
    if (replacement->Is<ExtendOp>()) {
      VLOG(3) << "  Performed narrowing of " << original << " to "
              << replacement << " (unextended: " << replacement->operand(0)
              << ")";
    } else {
      VLOG(3) << "  Performed narrowing of " << original << " to "
              << replacement;
    }
    return absl::OkStatus();
  }
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

void AnalysisLog(FunctionBase* f, const QueryEngine& query_engine) {
  VLOG(3) << "narrowing pass: Preliminary analysis prediction for : "
          << f->name();
  absl::flat_hash_map<Node*, absl::flat_hash_set<Node*>> parents_map;

  for (Node* node : f->nodes()) {
    for (Node* operand : node->operands()) {
      parents_map[operand].insert(node);
    }
  }

  StatelessQueryEngine stateless_query_engine;
  for (Node* node : f->nodes()) {
    if (node->GetType()->IsBits()) {
      IntervalSet intervals = query_engine.GetIntervals(node).Get({});
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
      if (stateless_query_engine.IsFullyKnown(node) &&
          (compressed_size < current_size)) {
        std::vector<std::string> parents;
        for (Node* parent : parents_map[node]) {
          parents.push_back(parent->ToString());
        }
        VLOG(3) << "  narrowing_pass: " << OpToString(node->op())
                << " shrinkable: " << "current size = " << current_size << "; "
                << "compressed size = " << compressed_size << "; "
                << "convex hull size = " << hull_size << "; " << "parents = ["
                << absl::StrJoin(parents, ", ") << "]\n";
      }

      bool inputs_all_maximal = true;
      for (Node* operand : node->operands()) {
        if (!operand->GetType()->IsBits()) {
          break;
        }
        IntervalSet intervals = query_engine.GetIntervals(operand).Get({});
        intervals.Normalize();
        if (!intervals.IsMaximal()) {
          inputs_all_maximal = false;
          break;
        }
      }
      if (!inputs_all_maximal &&
          query_engine.GetIntervals(node).Get({}).IsMaximal()) {
        VLOG(3) << "  narrowing_pass: range analysis lost precision for "
                << node << "\n";
      }
    }
  }
}

absl::StatusOr<AliasingQueryEngine> GetQueryEngine(
    FunctionBase* f, AnalysisType analysis, OptimizationContext& context) {
  std::vector<std::unique_ptr<QueryEngine>> owned_engines;
  std::vector<QueryEngine*> unowned_engines;
  owned_engines.push_back(std::make_unique<StatelessQueryEngine>());
  unowned_engines.push_back(context.SharedQueryEngine<BitCountQueryEngine>(f));
  if (analysis == AnalysisType::kRangeWithContext) {
    if (ProcStateRangeQueryEngine::CanAnalyzeProcStateEvolution(f)) {
      // NB ProcStateRange already includes a ternary qe
      owned_engines.push_back(std::make_unique<ProcStateRangeQueryEngine>());
    }
    unowned_engines.push_back(
        context.SharedQueryEngine<PartialInfoQueryEngine>(f));
    owned_engines.push_back(
        std::make_unique<ContextSensitiveRangeQueryEngine>());
  } else if (analysis == AnalysisType::kRange) {
    if (ProcStateRangeQueryEngine::CanAnalyzeProcStateEvolution(f)) {
      // NB ProcStateRange already includes a ternary qe
      owned_engines.push_back(std::make_unique<ProcStateRangeQueryEngine>());
    }
    unowned_engines.push_back(
        context.SharedQueryEngine<PartialInfoQueryEngine>(f));
  } else {
    CHECK_EQ(analysis, AnalysisType::kTernary);
    unowned_engines.push_back(
        context.SharedQueryEngine<LazyTernaryQueryEngine>(f));
  }
  auto query_engine = std::make_unique<UnionQueryEngine>(
      std::move(owned_engines), std::move(unowned_engines));
  XLS_RETURN_IF_ERROR(query_engine->Populate(f).status());
  if (VLOG_IS_ON(3)) {
    AnalysisLog(f, *query_engine);
  }
  return AliasingQueryEngine(std::move(query_engine));
}

}  // namespace

absl::StatusOr<bool> NarrowingPass::RunOnFunctionBaseInternal(
    FunctionBase* f, const OptimizationPassOptions& options,
    PassResults* results, OptimizationContext& context) const {
  XLS_ASSIGN_OR_RETURN(AliasingQueryEngine query_engine,
                       GetQueryEngine(f, RealAnalysis(options), context));

  PredicateDominatorAnalysis pda = PredicateDominatorAnalysis::Run(f);
  SpecializedQueryEngines sqe(RealAnalysis(options), pda, query_engine);

  NarrowVisitor narrower(sqe, RealAnalysis(options), options,
                         options.splits_enabled());

  for (Node* node : context.TopoSort(f)) {
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
      "narrow(Ternary)", NarrowingPass::AnalysisType::kTernary));
  CHECK_OK(RegisterOptimizationPass<NarrowingPass>(
      "narrow(Range)", NarrowingPass::AnalysisType::kRange));
  CHECK_OK(RegisterOptimizationPass<NarrowingPass>(
      "narrow(Context)", NarrowingPass::AnalysisType::kRangeWithContext));
  CHECK_OK(RegisterOptimizationPass<NarrowingPass>(
      "narrow(OptionalContext)",
      NarrowingPass::AnalysisType::kRangeWithOptionalContext));
  RegisterOptimizationPassInfoFor(
      {"narrow", "narrow(Ternary)", "narrow(Range)", "narrow(Context)",
       "narrow(OptionalContext)"},
      "NarrowingPass");
});

}  // namespace xls
