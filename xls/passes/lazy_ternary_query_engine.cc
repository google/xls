// Copyright 2025 The XLS Authors
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

#include "xls/passes/lazy_ternary_query_engine.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "cppitertools/zip.hpp"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/interval.h"
#include "xls/ir/interval_ops.h"
#include "xls/ir/lsb_or_msb.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/ternary.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/dataflow_visitor.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/ternary_evaluator.h"

namespace xls {

namespace {

class TernaryVisitor : public DataflowVisitor<TernaryVector> {
 public:
  TernaryVisitor() = default;

  absl::Status InjectValue(Node* node, const TernaryTree* value) {
    if (value == nullptr) {
      XLS_ASSIGN_OR_RETURN(
          TernaryTree unknown,
          TernaryTree::CreateFromFunction(
              node->GetType(),
              [](Type* leaf_type) -> absl::StatusOr<TernaryVector> {
                return TernaryVector(leaf_type->GetFlatBitCount(),
                                     TernaryValue::kUnknown);
              }));
      return SetValue(node, std::move(unknown));
    }
    return SetValue(node, value->AsView());
  }

  absl::Status DefaultHandler(Node* node) override {
    XLS_ASSIGN_OR_RETURN(
        TernaryTree unknown,
        TernaryTree::CreateFromFunction(
            node->GetType(),
            [](Type* leaf_type) -> absl::StatusOr<TernaryVector> {
              return TernaryVector(leaf_type->GetFlatBitCount(),
                                   TernaryValue::kUnknown);
            }));
    return SetValue(node, std::move(unknown));
  }

  absl::Status HandleAdd(BinOp* add) override {
    Node* lhs = add->operand(0);
    Node* rhs = add->operand(1);
    return SetValue(
        add, TernaryTree::CreateSingleElementTree(
                 add->GetType(),
                 evaluator_.Add(GetValue(lhs).Get({}), GetValue(rhs).Get({}))));
  }
  absl::Status HandleAndReduce(BitwiseReductionOp* and_reduce) override {
    Node* input = and_reduce->operand(0);
    return SetValue(and_reduce,
                    TernaryTree::CreateSingleElementTree(
                        and_reduce->GetType(),
                        evaluator_.AndReduce(GetValue(input).Get({}))));
  }

  absl::Status HandleBitSlice(BitSlice* bit_slice) override {
    return SetValue(
        bit_slice,
        TernaryTree::CreateSingleElementTree(
            bit_slice->GetType(),
            evaluator_.BitSlice(GetValue(bit_slice->operand(0)).Get({}),
                                bit_slice->start(), bit_slice->width())));
  }
  absl::Status HandleBitSliceUpdate(BitSliceUpdate* update) override {
    return SetValue(update, TernaryTree::CreateSingleElementTree(
                                update->GetType(),
                                evaluator_.BitSliceUpdate(
                                    GetValue(update->to_update()).Get({}),
                                    GetValue(update->start()).Get({}),
                                    GetValue(update->update_value()).Get({}))));
  }
  absl::Status HandleConcat(Concat* concat) override {
    std::vector<TernarySpan> operands;
    for (Node* operand : concat->operands()) {
      operands.push_back(GetValue(operand).Get({}));
    }
    return SetValue(concat,
                    TernaryTree::CreateSingleElementTree(
                        concat->GetType(), evaluator_.Concat(operands)));
  }
  absl::Status HandleDecode(Decode* decode) override {
    return SetValue(decode,
                    TernaryTree::CreateSingleElementTree(
                        decode->GetType(),
                        evaluator_.Decode(GetValue(decode->operand(0)).Get({}),
                                          decode->width())));
  }
  absl::Status HandleDynamicBitSlice(
      DynamicBitSlice* dynamic_bit_slice) override {
    return SetValue(dynamic_bit_slice,
                    TernaryTree::CreateSingleElementTree(
                        dynamic_bit_slice->GetType(),
                        evaluator_.DynamicBitSlice(
                            GetValue(dynamic_bit_slice->operand(0)).Get({}),
                            GetValue(dynamic_bit_slice->start()).Get({}),
                            dynamic_bit_slice->width())));
  }
  absl::Status HandleEncode(Encode* encode) override {
    return SetValue(
        encode, TernaryTree::CreateSingleElementTree(
                    encode->GetType(),
                    evaluator_.Encode(GetValue(encode->operand(0)).Get({}))));
  }
  absl::Status HandleEq(CompareOp* eq) override {
    Node* lhs = eq->operand(0);
    Node* rhs = eq->operand(1);

    TernaryVector leaves_eq;
    leaves_eq.reserve(lhs->GetType()->leaf_count());
    for (const auto& [lhs_leaf, rhs_leaf] :
         iter::zip(GetValue(lhs).elements(), GetValue(rhs).elements())) {
      leaves_eq.push_back(evaluator_.Equals(lhs_leaf, rhs_leaf));
    }
    return SetValue(eq, TernaryTree::CreateSingleElementTree(
                            eq->GetType(),
                            TernaryVector({evaluator_.AndReduce(leaves_eq)})));
  }
  absl::Status HandleGate(Gate* gate) override {
    TernaryValue control = GetValue(gate->operand(0)).Get({}).front();
    TernaryTreeView input = GetValue(gate->operand(1)).AsView();
    return SetValue(gate, leaf_type_tree::Map<TernaryVector, TernaryVector>(
                              input, [&](TernaryVector input_leaf) {
                                return evaluator_.Gate(control, input_leaf);
                              }));
  }
  absl::Status HandleLiteral(Literal* literal) override {
    XLS_ASSIGN_OR_RETURN(
        LeafTypeTree<Value> value_tree,
        ValueToLeafTypeTree(literal->value(), literal->GetType()));
    return SetValue(literal,
                    leaf_type_tree::Map<TernaryVector, Value>(
                        value_tree.AsView(), [](Value value) {
                          if (value.IsToken()) {
                            return TernaryVector();
                          }
                          CHECK(value.IsBits());
                          return ternary_ops::BitsToTernary(value.bits());
                        }));
  }
  absl::Status HandleNaryAnd(NaryOp* and_op) override {
    std::vector<TernarySpan> operands;
    operands.reserve(and_op->operand_count());
    for (Node* operand : and_op->operands()) {
      operands.push_back(GetValue(operand).Get({}));
    }
    return SetValue(and_op,
                    TernaryTree::CreateSingleElementTree(
                        and_op->GetType(), evaluator_.BitwiseAnd(operands)));
  }
  absl::Status HandleNaryNand(NaryOp* nand_op) override {
    std::vector<TernarySpan> operands;
    operands.reserve(nand_op->operand_count());
    for (Node* operand : nand_op->operands()) {
      operands.push_back(GetValue(operand).Get({}));
    }
    return SetValue(
        nand_op, TernaryTree::CreateSingleElementTree(
                     nand_op->GetType(),
                     evaluator_.BitwiseNot(evaluator_.BitwiseAnd(operands))));
  }
  absl::Status HandleNaryNor(NaryOp* nor_op) override {
    std::vector<TernarySpan> operands;
    operands.reserve(nor_op->operand_count());
    for (Node* operand : nor_op->operands()) {
      operands.push_back(GetValue(operand).Get({}));
    }
    return SetValue(nor_op,
                    TernaryTree::CreateSingleElementTree(
                        nor_op->GetType(),
                        evaluator_.BitwiseNot(evaluator_.BitwiseOr(operands))));
  }
  absl::Status HandleNaryOr(NaryOp* or_op) override {
    std::vector<TernarySpan> operands;
    operands.reserve(or_op->operand_count());
    for (Node* operand : or_op->operands()) {
      operands.push_back(GetValue(operand).Get({}));
    }
    return SetValue(or_op,
                    TernaryTree::CreateSingleElementTree(
                        or_op->GetType(), evaluator_.BitwiseOr(operands)));
  }
  absl::Status HandleNaryXor(NaryOp* xor_op) override {
    std::vector<TernarySpan> operands;
    operands.reserve(xor_op->operand_count());
    for (Node* operand : xor_op->operands()) {
      operands.push_back(GetValue(operand).Get({}));
    }
    return SetValue(xor_op,
                    TernaryTree::CreateSingleElementTree(
                        xor_op->GetType(), evaluator_.BitwiseXor(operands)));
  }
  absl::Status HandleNe(CompareOp* ne) override {
    Node* lhs = ne->operand(0);
    Node* rhs = ne->operand(1);

    TernaryVector leaves_ne;
    leaves_ne.reserve(lhs->GetType()->leaf_count());
    for (const auto& [lhs_leaf, rhs_leaf] :
         iter::zip(GetValue(lhs).elements(), GetValue(rhs).elements())) {
      leaves_ne.push_back(
          evaluator_.Not(evaluator_.Equals(lhs_leaf, rhs_leaf)));
    }
    return SetValue(ne, TernaryTree::CreateSingleElementTree(
                            ne->GetType(),
                            TernaryVector({evaluator_.OrReduce(leaves_ne)})));
  }
  absl::Status HandleNeg(UnOp* neg) override {
    return SetValue(
        neg,
        TernaryTree::CreateSingleElementTree(
            neg->GetType(), evaluator_.Neg(GetValue(neg->operand(0)).Get({}))));
  }
  absl::Status HandleNot(UnOp* not_op) override {
    return SetValue(
        not_op,
        TernaryTree::CreateSingleElementTree(
            not_op->GetType(),
            evaluator_.BitwiseNot(GetValue(not_op->operand(0)).Get({}))));
  }
  absl::Status HandleOneHot(OneHot* one_hot) override {
    TernarySpan input = GetValue(one_hot->operand(0)).Get({});
    TernaryVector result;
    return SetValue(one_hot, TernaryTree::CreateSingleElementTree(
                                 one_hot->GetType(),
                                 one_hot->priority() == LsbOrMsb::kLsb
                                     ? evaluator_.OneHotLsbToMsb(input)
                                     : evaluator_.OneHotMsbToLsb(input)));
  }
  absl::Status HandleOneHotSel(OneHotSelect* ohs) override {
    TernarySpan selector = GetValue(ohs->selector()).Get({});
    const bool selector_can_be_zero =
        !query_engine_.AtLeastOneBitTrue(ohs->selector());

    std::vector<TernaryTreeView> cases;
    cases.reserve(ohs->cases().size());
    for (Node* case_node : ohs->cases()) {
      cases.push_back(GetValue(case_node));
    }

    XLS_ASSIGN_OR_RETURN(
        TernaryTree result,
        (leaf_type_tree::ZipIndex<TernaryVector, TernaryVector>(
            cases,
            [&](Type*, absl::Span<const TernaryVector* const> case_elements,
                absl::Span<const int64_t>) -> absl::StatusOr<TernaryVector> {
              std::vector<TernarySpan> case_spans;
              case_spans.reserve(case_elements.size());
              for (const TernaryVector* case_element : case_elements) {
                case_spans.push_back(*case_element);
              }
              return evaluator_.OneHotSelect(selector, case_spans,
                                             selector_can_be_zero);
            })));
    return SetValue(ohs, std::move(result));
  }
  absl::Status HandleOrReduce(BitwiseReductionOp* or_reduce) override {
    Node* input = or_reduce->operand(0);
    return SetValue(or_reduce,
                    TernaryTree::CreateSingleElementTree(
                        or_reduce->GetType(),
                        evaluator_.OrReduce(GetValue(input).Get({}))));
  }
  absl::Status HandleReverse(UnOp* reverse) override {
    TernaryVector result = GetValue(reverse->operand(0)).Get({});
    absl::c_reverse(result);
    return SetValue(reverse, TernaryTree::CreateSingleElementTree(
                                 reverse->GetType(), result));
  }
  absl::Status HandleSDiv(BinOp* div) override {
    Node* lhs = div->operand(0);
    Node* rhs = div->operand(1);
    return SetValue(
        div, TernaryTree::CreateSingleElementTree(
                 div->GetType(), evaluator_.SDiv(GetValue(lhs).Get({}),
                                                 GetValue(rhs).Get({}))));
  }
  absl::Status HandleSGe(CompareOp* ge) override {
    Node* lhs = ge->operand(0);
    Node* rhs = ge->operand(1);
    return SetValue(ge,
                    TernaryTree::CreateSingleElementTree(
                        ge->GetType(),
                        TernaryVector({evaluator_.Not(evaluator_.SLessThan(
                            GetValue(lhs).Get({}), GetValue(rhs).Get({})))})));
  }
  absl::Status HandleSGt(CompareOp* gt) override {
    Node* lhs = gt->operand(0);
    Node* rhs = gt->operand(1);
    return SetValue(
        gt, TernaryTree::CreateSingleElementTree(
                gt->GetType(),
                TernaryVector({evaluator_.SLessThan(GetValue(rhs).Get({}),
                                                    GetValue(lhs).Get({}))})));
  }
  absl::Status HandleSLe(CompareOp* le) override {
    Node* lhs = le->operand(0);
    Node* rhs = le->operand(1);
    return SetValue(le,
                    TernaryTree::CreateSingleElementTree(
                        le->GetType(),
                        TernaryVector({evaluator_.Not(evaluator_.SLessThan(
                            GetValue(rhs).Get({}), GetValue(lhs).Get({})))})));
  }
  absl::Status HandleSLt(CompareOp* lt) override {
    Node* lhs = lt->operand(0);
    Node* rhs = lt->operand(1);
    return SetValue(
        lt, TernaryTree::CreateSingleElementTree(
                lt->GetType(),
                TernaryVector({evaluator_.SLessThan(GetValue(lhs).Get({}),
                                                    GetValue(rhs).Get({}))})));
  }
  absl::Status HandleSMod(BinOp* mod) override {
    Node* lhs = mod->operand(0);
    Node* rhs = mod->operand(1);
    return SetValue(
        mod, TernaryTree::CreateSingleElementTree(
                 mod->GetType(), evaluator_.SMod(GetValue(lhs).Get({}),
                                                 GetValue(rhs).Get({}))));
  }
  absl::Status HandleSMul(ArithOp* mul) override {
    TernaryVector result = evaluator_.SMul(GetValue(mul->operand(0)).Get({}),
                                           GetValue(mul->operand(1)).Get({}));
    int64_t expected_width = mul->BitCountOrDie();
    if (result.size() > expected_width) {
      result = evaluator_.BitSlice(result, 0, expected_width);
    } else if (result.size() < expected_width) {
      result = evaluator_.SignExtend(result, expected_width);
    }
    return SetValue(
        mul, TernaryTree::CreateSingleElementTree(mul->GetType(), result));
  }
  absl::Status HandleShll(BinOp* shll) override {
    Node* input = shll->operand(0);
    Node* amount = shll->operand(1);
    return SetValue(shll, TernaryTree::CreateSingleElementTree(
                              shll->GetType(), evaluator_.ShiftLeftLogical(
                                                   GetValue(input).Get({}),
                                                   GetValue(amount).Get({}))));
  }
  absl::Status HandleShra(BinOp* shra) override {
    Node* input = shra->operand(0);
    Node* amount = shra->operand(1);
    return SetValue(shra, TernaryTree::CreateSingleElementTree(
                              shra->GetType(), evaluator_.ShiftRightArith(
                                                   GetValue(input).Get({}),
                                                   GetValue(amount).Get({}))));
  }
  absl::Status HandleShrl(BinOp* shrl) override {
    Node* input = shrl->operand(0);
    Node* amount = shrl->operand(1);
    return SetValue(shrl, TernaryTree::CreateSingleElementTree(
                              shrl->GetType(), evaluator_.ShiftRightLogical(
                                                   GetValue(input).Get({}),
                                                   GetValue(amount).Get({}))));
  }
  absl::Status HandleSignExtend(ExtendOp* sign_ext) override {
    return SetValue(sign_ext, TernaryTree::CreateSingleElementTree(
                                  sign_ext->GetType(),
                                  evaluator_.SignExtend(
                                      GetValue(sign_ext->operand(0)).Get({}),
                                      sign_ext->new_bit_count())));
  }
  absl::Status HandleSub(BinOp* sub) override {
    Node* lhs = sub->operand(0);
    Node* rhs = sub->operand(1);
    return SetValue(
        sub, TernaryTree::CreateSingleElementTree(
                 sub->GetType(),
                 evaluator_.Sub(GetValue(lhs).Get({}), GetValue(rhs).Get({}))));
  }
  absl::Status HandleUDiv(BinOp* div) override {
    Node* lhs = div->operand(0);
    Node* rhs = div->operand(1);
    return SetValue(
        div, TernaryTree::CreateSingleElementTree(
                 div->GetType(), evaluator_.UDiv(GetValue(lhs).Get({}),
                                                 GetValue(rhs).Get({}))));
  }
  absl::Status HandleUGe(CompareOp* ge) override {
    Node* lhs = ge->operand(0);
    Node* rhs = ge->operand(1);
    return SetValue(ge,
                    TernaryTree::CreateSingleElementTree(
                        ge->GetType(),
                        TernaryVector({evaluator_.UGreaterThanOrEqual(
                            GetValue(lhs).Get({}), GetValue(rhs).Get({}))})));
  }
  absl::Status HandleUGt(CompareOp* gt) override {
    Node* lhs = gt->operand(0);
    Node* rhs = gt->operand(1);
    return SetValue(gt,
                    TernaryTree::CreateSingleElementTree(
                        gt->GetType(),
                        TernaryVector({evaluator_.UGreaterThan(
                            GetValue(lhs).Get({}), GetValue(rhs).Get({}))})));
  }
  absl::Status HandleULe(CompareOp* le) override {
    Node* lhs = le->operand(0);
    Node* rhs = le->operand(1);
    return SetValue(le,
                    TernaryTree::CreateSingleElementTree(
                        le->GetType(),
                        TernaryVector({evaluator_.ULessThanOrEqual(
                            GetValue(lhs).Get({}), GetValue(rhs).Get({}))})));
  }
  absl::Status HandleULt(CompareOp* lt) override {
    Node* lhs = lt->operand(0);
    Node* rhs = lt->operand(1);
    return SetValue(
        lt, TernaryTree::CreateSingleElementTree(
                lt->GetType(),
                TernaryVector({evaluator_.ULessThan(GetValue(lhs).Get({}),
                                                    GetValue(rhs).Get({}))})));
  }
  absl::Status HandleUMod(BinOp* mod) override {
    Node* lhs = mod->operand(0);
    Node* rhs = mod->operand(1);
    return SetValue(
        mod, TernaryTree::CreateSingleElementTree(
                 mod->GetType(), evaluator_.UMod(GetValue(lhs).Get({}),
                                                 GetValue(rhs).Get({}))));
  }
  absl::Status HandleUMul(ArithOp* mul) override {
    TernaryVector result = evaluator_.UMul(GetValue(mul->operand(0)).Get({}),
                                           GetValue(mul->operand(1)).Get({}));
    int64_t expected_width = mul->BitCountOrDie();
    if (result.size() > expected_width) {
      result = evaluator_.BitSlice(result, 0, expected_width);
    } else if (result.size() < expected_width) {
      result = evaluator_.ZeroExtend(result, expected_width);
    }
    return SetValue(
        mul, TernaryTree::CreateSingleElementTree(mul->GetType(), result));
  }
  absl::Status HandleXorReduce(BitwiseReductionOp* xor_reduce) override {
    Node* input = xor_reduce->operand(0);
    return SetValue(xor_reduce,
                    TernaryTree::CreateSingleElementTree(
                        xor_reduce->GetType(),
                        evaluator_.XorReduce(GetValue(input).Get({}))));
  }
  absl::Status HandleZeroExtend(ExtendOp* zero_ext) override {
    return SetValue(zero_ext, TernaryTree::CreateSingleElementTree(
                                  zero_ext->GetType(),
                                  evaluator_.ZeroExtend(
                                      GetValue(zero_ext->operand(0)).Get({}),
                                      zero_ext->new_bit_count())));
  }

 protected:
  bool IndexMightBeEquivalent(const TernaryVector& index,
                              int64_t concrete_index, int64_t bound,
                              bool index_clamped) const override {
    CHECK_LT(concrete_index, bound);
    if (Bits::MinBitCountUnsigned(concrete_index) > index.size()) {
      // `index` is too narrow to represent `concrete_index`.
      return false;
    }
    Bits concrete_index_bits = UBits(concrete_index, index.size());
    if (index_clamped && concrete_index == bound - 1) {
      // We're at the end of the array; we'll end up clamped to this point if
      // `index` is greater than or equal to `concrete_index`.
      return interval_ops::CoversTernary(
          Interval::Closed(concrete_index_bits, Bits::AllOnes(index.size())),
          index);
    }
    return ternary_ops::IsCompatible(index, concrete_index_bits);
  }

  absl::StatusOr<TernaryVector> SelectElements(
      TernarySpan selector, absl::Span<const TernaryVector* const> data_sources,
      bool has_default_value) {
    std::optional<TernarySpan> default_value;
    if (has_default_value) {
      default_value = *data_sources.back();
      data_sources.remove_suffix(1);
    }

    std::vector<TernarySpan> case_spans;
    case_spans.reserve(data_sources.size());
    for (const TernaryVector* data_source : data_sources) {
      case_spans.push_back(*data_source);
    }

    return evaluator_.Select(selector, case_spans, default_value);
  }

  absl::StatusOr<TernaryVector> PrioritySelectElements(
      TernarySpan selector, absl::Span<const TernaryVector* const> data_sources,
      bool selector_can_be_zero) {
    TernarySpan default_value = *data_sources.back();
    data_sources.remove_suffix(1);

    std::vector<TernarySpan> case_spans;
    case_spans.reserve(data_sources.size());
    for (const TernaryVector* data_source : data_sources) {
      case_spans.push_back(*data_source);
    }

    return evaluator_.PrioritySelect(selector, case_spans, selector_can_be_zero,
                                     default_value);
  }

  absl::StatusOr<TernaryVector> JoinElements(
      Type* element_type, absl::Span<const TernaryVector* const> data_sources,
      absl::Span<const LeafTypeTreeView<TernaryVector>> control_sources,
      Node* node, absl::Span<const int64_t> index) override {
    if (node->Is<Select>()) {
      CHECK_EQ(control_sources.size(), 1);
      return SelectElements(/*selector=*/control_sources.front().Get({}),
                            data_sources,
                            /*has_default_value=*/
                            node->As<Select>()->default_value().has_value());
    }

    if (node->Is<PrioritySelect>()) {
      CHECK_EQ(control_sources.size(), 1);
      const bool selector_can_be_zero = !query_engine_.AtLeastOneBitTrue(
          node->As<PrioritySelect>()->selector());
      return PrioritySelectElements(
          /*selector=*/control_sources.front().Get({}), data_sources,
          selector_can_be_zero);
    }

    if (data_sources.empty()) {
      return TernaryVector(element_type->GetFlatBitCount(),
                           TernaryValue::kUnknown);
    }
    TernaryVector result = *data_sources.front();
    for (const TernaryVector* const data_source : data_sources.subspan(1)) {
      ternary_ops::UpdateWithIntersection(result, *data_source);
    }
    return result;
  }

 private:
  TernaryEvaluator evaluator_;
};

// Returns whether the operation will be computationally expensive to
// compute. The ternary query engine is intended to be fast so the analysis of
// these expensive operations is skipped with the effect being all bits are
// considered unknown. Operations and limits can be added as needed when
// pathological cases are encountered.
bool IsExpensiveToEvaluate(Node* node,
                           absl::Span<const TernaryTree* const> operand_infos) {
  // How many bits we allow to be unconstrained when we need to intersect
  // multiple possibilities.
  static constexpr int64_t kIndexBitLimit = 10;
  // How many bits of output we allow for complex evaluations.
  static constexpr int64_t kComplexEvaluationLimit = 256;
  // How many bits of data we are willing to keep track of for compound
  // data-types.
  static constexpr int64_t kCompoundDataTypeSizeLimit = 65536;
  // Shifts are quadratic in the width of the value being shifted, so wide
  // shifts are very slow to evaluate in the abstract evaluator.
  bool is_complex_evaluation = node->OpIn({
      Op::kShrl,
      Op::kShll,
      Op::kShra,
      Op::kBitSliceUpdate,
      Op::kDynamicBitSlice,
  });
  if (is_complex_evaluation) {
    return node->operand(0)->GetType()->GetFlatBitCount() >
           kComplexEvaluationLimit;
  }
  // Compound data types can get enormous. Put a limit on how much data we are
  // willing to carry around.
  if (!node->GetType()->IsBits() &&
      node->GetType()->GetFlatBitCount() > kCompoundDataTypeSizeLimit) {
    return true;
  }
  // Array index checks require (worst case) 2^|unknown bits| operations so
  // limit them to 1024 == 2**10 operations.
  bool needs_index_scan =
      node->OpIn({Op::kArrayIndex, Op::kArraySlice, Op::kArrayUpdate});
  if (needs_index_scan) {
    int64_t unknown_index_bits = 0;
    int64_t index_operand_start;
    int64_t index_operand_end;
    switch (node->op()) {
      default:
        LOG(FATAL) << "Unexpected node type: " << node->ToString();
      case Op::kArrayIndex: {
        index_operand_start = ArrayIndex::kIndexOperandStart;
        index_operand_end = ArrayIndex::kIndexOperandStart +
                            node->As<ArrayIndex>()->indices().size();
        break;
      }
      case Op::kArraySlice: {
        index_operand_start = ArraySlice::kStartOperand;
        index_operand_end = ArraySlice::kStartOperand + 1;
        break;
      }
      case Op::kArrayUpdate: {
        index_operand_start = ArrayUpdate::kIndexOperandStart;
        index_operand_end = ArrayUpdate::kIndexOperandStart +
                            node->As<ArrayUpdate>()->indices().size();
        break;
      }
    }
    for (int64_t operand = index_operand_start; operand < index_operand_end;
         ++operand) {
      if (operand_infos.at(operand) == nullptr) {
        unknown_index_bits +=
            node->operand(operand)->GetType()->GetFlatBitCount();
      } else {
        unknown_index_bits += absl::c_count(operand_infos.at(operand)->Get({}),
                                            TernaryValue::kUnknown);
      }
    }
    return unknown_index_bits >= kIndexBitLimit;
  }

  return false;
}

};  // namespace

TernaryTree LazyTernaryQueryEngine::ComputeInfo(
    Node* node, absl::Span<const TernaryTree* const> operand_infos) const {
  TernaryVisitor visitor;
  if (IsExpensiveToEvaluate(node, operand_infos)) {
    CHECK_OK(visitor.DefaultHandler(node));
  } else {
    for (const auto& [operand, operand_info] :
         iter::zip(node->operands(), operand_infos)) {
      CHECK_OK(visitor.InjectValue(operand, operand_info));
    }
    CHECK_OK(node->VisitSingleNode(&visitor));
  }
  absl::flat_hash_map<Node*, std::unique_ptr<SharedTernaryTree>> result =
      std::move(visitor).ToStoredValues();
  return std::move(*result.at(node)).ToOwned();
}

absl::Status LazyTernaryQueryEngine::MergeWithGiven(
    TernaryVector& info, const TernaryVector& given) const {
  return ternary_ops::UpdateWithUnion(info, given);
}

bool LazyTernaryQueryEngine::AtMostOneTrue(
    absl::Span<TreeBitLocation const> bits) const {
  int64_t maybe_one_count = 0;
  for (const TreeBitLocation& location : bits) {
    if (!IsKnown(location) || IsOne(location)) {
      maybe_one_count++;
    }
  }
  return maybe_one_count <= 1;
}

bool LazyTernaryQueryEngine::AtLeastOneTrue(
    absl::Span<TreeBitLocation const> bits) const {
  for (const TreeBitLocation& location : bits) {
    if (IsOne(location)) {
      return true;
    }
  }
  return false;
}

bool LazyTernaryQueryEngine::KnownEquals(const TreeBitLocation& a,
                                         const TreeBitLocation& b) const {
  return IsKnown(a) && IsKnown(b) && IsOne(a) == IsOne(b);
}

bool LazyTernaryQueryEngine::KnownNotEquals(const TreeBitLocation& a,
                                            const TreeBitLocation& b) const {
  return IsKnown(a) && IsKnown(b) && IsOne(a) != IsOne(b);
}

}  // namespace xls
