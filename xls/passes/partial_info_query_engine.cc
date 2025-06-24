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

#include "xls/passes/partial_info_query_engine.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "cppitertools/zip.hpp"
#include "xls/common/math_util.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/interval.h"
#include "xls/ir/interval_ops.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/lsb_or_msb.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/partial_information.h"
#include "xls/ir/partial_ops.h"
#include "xls/ir/ternary.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/dataflow_visitor.h"
#include "xls/passes/query_engine.h"

namespace xls {

namespace {

class PartialInfoVisitor : public DataflowVisitor<PartialInformation> {
 public:
  PartialInfoVisitor() = default;

  absl::Status InjectValue(Node* node,
                           const LeafTypeTree<PartialInformation>* value) {
    if (value == nullptr) {
      XLS_ASSIGN_OR_RETURN(
          LeafTypeTree<PartialInformation> unknown,
          LeafTypeTree<PartialInformation>::CreateFromFunction(
              node->GetType(),
              [](Type* leaf_type) -> absl::StatusOr<PartialInformation> {
                return PartialInformation::Unconstrained(
                    leaf_type->GetFlatBitCount());
              }));
      return SetValue(node, std::move(unknown));
    }
    return SetValue(node, value->AsView());
  }

  absl::Status DefaultHandler(Node* node) override {
    XLS_ASSIGN_OR_RETURN(
        LeafTypeTree<PartialInformation> unknown,
        LeafTypeTree<PartialInformation>::CreateFromFunction(
            node->GetType(),
            [](Type* leaf_type) -> absl::StatusOr<PartialInformation> {
              return PartialInformation::Unconstrained(
                  leaf_type->GetFlatBitCount());
            }));
    return SetValue(node, std::move(unknown));
  }

  absl::Status HandleAdd(BinOp* add) override {
    Node* lhs = add->operand(0);
    Node* rhs = add->operand(1);
    return SetValue(
        add, LeafTypeTree<PartialInformation>::CreateSingleElementTree(
                 add->GetType(), partial_ops::Add(GetValue(lhs).Get({}),
                                                  GetValue(rhs).Get({}))));
  }
  absl::Status HandleAndReduce(BitwiseReductionOp* and_reduce) override {
    Node* input = and_reduce->operand(0);
    return SetValue(and_reduce,
                    LeafTypeTree<PartialInformation>::CreateSingleElementTree(
                        and_reduce->GetType(),
                        partial_ops::AndReduce(GetValue(input).Get({}))));
  }

  absl::Status HandleBitSlice(BitSlice* bit_slice) override {
    return SetValue(
        bit_slice,
        LeafTypeTree<PartialInformation>::CreateSingleElementTree(
            bit_slice->GetType(),
            partial_ops::BitSlice(GetValue(bit_slice->operand(0)).Get({}),
                                  bit_slice->start(), bit_slice->width())));
  }
  absl::Status HandleBitSliceUpdate(BitSliceUpdate* update) override {
    return SetValue(
        update,
        LeafTypeTree<PartialInformation>::CreateSingleElementTree(
            update->GetType(), partial_ops::BitSliceUpdate(
                                   GetValue(update->to_update()).Get({}),
                                   GetValue(update->start()).Get({}),
                                   GetValue(update->update_value()).Get({}))));
  }
  absl::Status HandleConcat(Concat* concat) override {
    std::vector<PartialInformation> operands;
    for (Node* operand : concat->operands()) {
      operands.push_back(GetValue(operand).Get({}));
    }
    return SetValue(concat,
                    LeafTypeTree<PartialInformation>::CreateSingleElementTree(
                        concat->GetType(), partial_ops::Concat(operands)));
  }
  absl::Status HandleDecode(Decode* decode) override {
    return SetValue(
        decode, LeafTypeTree<PartialInformation>::CreateSingleElementTree(
                    decode->GetType(),
                    partial_ops::Decode(GetValue(decode->operand(0)).Get({}),
                                        decode->width())));
  }
  absl::Status HandleDynamicBitSlice(
      DynamicBitSlice* dynamic_bit_slice) override {
    return SetValue(dynamic_bit_slice,
                    LeafTypeTree<PartialInformation>::CreateSingleElementTree(
                        dynamic_bit_slice->GetType(),
                        partial_ops::DynamicBitSlice(
                            GetValue(dynamic_bit_slice->operand(0)).Get({}),
                            GetValue(dynamic_bit_slice->start()).Get({}),
                            dynamic_bit_slice->width())));
  }
  absl::Status HandleEncode(Encode* encode) override {
    return SetValue(
        encode, LeafTypeTree<PartialInformation>::CreateSingleElementTree(
                    encode->GetType(),
                    partial_ops::Encode(GetValue(encode->operand(0)).Get({}))));
  }
  absl::Status HandleEq(CompareOp* eq) override {
    Node* lhs = eq->operand(0);
    Node* rhs = eq->operand(1);

    PartialInformation result =
        PartialInformation(TernaryVector({TernaryValue::kKnownOne}));
    for (const auto& [lhs_leaf, rhs_leaf] :
         iter::zip(GetValue(lhs).elements(), GetValue(rhs).elements())) {
      result = partial_ops::And(result, partial_ops::Eq(lhs_leaf, rhs_leaf));
    }
    return SetValue(eq,
                    LeafTypeTree<PartialInformation>::CreateSingleElementTree(
                        eq->GetType(), result));
  }
  absl::Status HandleGate(Gate* gate) override {
    const PartialInformation& control = GetValue(gate->operand(0)).Get({});
    LeafTypeTreeView<PartialInformation> input =
        GetValue(gate->operand(1)).AsView();
    return SetValue(
        gate,
        leaf_type_tree::Map<PartialInformation, PartialInformation>(
            input,
            [&](const PartialInformation& input_leaf) -> PartialInformation {
              return partial_ops::Gate(control, input_leaf);
            }));
  }
  absl::Status HandleLiteral(Literal* literal) override {
    XLS_ASSIGN_OR_RETURN(
        LeafTypeTree<Value> value_tree,
        ValueToLeafTypeTree(literal->value(), literal->GetType()));
    return SetValue(
        literal,
        leaf_type_tree::Map<PartialInformation, Value>(
            value_tree.AsView(), [](Value value) -> PartialInformation {
              if (value.IsToken()) {
                return PartialInformation::Unconstrained(0);
              }
              CHECK(value.IsBits());
              return PartialInformation::Precise(value.bits());
            }));
  }
  absl::Status HandleNaryAnd(NaryOp* and_op) override {
    PartialInformation result = PartialInformation::Precise(
        Bits::AllOnes(and_op->GetType()->GetFlatBitCount()));
    for (Node* operand : and_op->operands()) {
      result.And(GetValue(operand).Get({}));
    }
    return SetValue(and_op,
                    LeafTypeTree<PartialInformation>::CreateSingleElementTree(
                        and_op->GetType(), result));
  }
  absl::Status HandleNaryNand(NaryOp* nand_op) override {
    PartialInformation result = PartialInformation::Precise(
        Bits::AllOnes(nand_op->GetType()->GetFlatBitCount()));
    for (Node* operand : nand_op->operands()) {
      result.And(GetValue(operand).Get({}));
    }
    return SetValue(nand_op,
                    LeafTypeTree<PartialInformation>::CreateSingleElementTree(
                        nand_op->GetType(), result.Not()));
  }
  absl::Status HandleNaryNor(NaryOp* nor_op) override {
    PartialInformation result =
        PartialInformation::Precise(Bits(nor_op->GetType()->GetFlatBitCount()));
    for (Node* operand : nor_op->operands()) {
      result.Or(GetValue(operand).Get({}));
    }
    return SetValue(nor_op,
                    LeafTypeTree<PartialInformation>::CreateSingleElementTree(
                        nor_op->GetType(), result.Not()));
  }
  absl::Status HandleNaryOr(NaryOp* or_op) override {
    PartialInformation result =
        PartialInformation::Precise(Bits(or_op->GetType()->GetFlatBitCount()));
    for (Node* operand : or_op->operands()) {
      result.Or(GetValue(operand).Get({}));
    }
    return SetValue(or_op,
                    LeafTypeTree<PartialInformation>::CreateSingleElementTree(
                        or_op->GetType(), result));
  }
  absl::Status HandleNaryXor(NaryOp* xor_op) override {
    PartialInformation result =
        PartialInformation::Precise(Bits(xor_op->GetType()->GetFlatBitCount()));
    for (Node* operand : xor_op->operands()) {
      result.Xor(GetValue(operand).Get({}));
    }
    return SetValue(xor_op,
                    LeafTypeTree<PartialInformation>::CreateSingleElementTree(
                        xor_op->GetType(), result));
  }
  absl::Status HandleNe(CompareOp* ne) override {
    Node* lhs = ne->operand(0);
    Node* rhs = ne->operand(1);

    PartialInformation result =
        PartialInformation(TernaryVector({TernaryValue::kKnownZero}));
    for (const auto& [lhs_leaf, rhs_leaf] :
         iter::zip(GetValue(lhs).elements(), GetValue(rhs).elements())) {
      result = partial_ops::Or(result, partial_ops::Ne(lhs_leaf, rhs_leaf));
      if (std::optional<Bits> precise_value = result.GetPreciseValue();
          precise_value.has_value() && precise_value->IsAllOnes()) {
        break;
      }
    }
    return SetValue(ne,
                    LeafTypeTree<PartialInformation>::CreateSingleElementTree(
                        ne->GetType(), result));
  }
  absl::Status HandleNeg(UnOp* neg) override {
    return SetValue(neg,
                    LeafTypeTree<PartialInformation>::CreateSingleElementTree(
                        neg->GetType(),
                        partial_ops::Neg(GetValue(neg->operand(0)).Get({}))));
  }
  absl::Status HandleNot(UnOp* not_op) override {
    return SetValue(
        not_op, LeafTypeTree<PartialInformation>::CreateSingleElementTree(
                    not_op->GetType(),
                    partial_ops::Not(GetValue(not_op->operand(0)).Get({}))));
  }
  absl::Status HandleOneHot(OneHot* one_hot) override {
    const PartialInformation& input = GetValue(one_hot->operand(0)).Get({});
    TernaryVector result;
    return SetValue(
        one_hot,
        LeafTypeTree<PartialInformation>::CreateSingleElementTree(
            one_hot->GetType(), one_hot->priority() == LsbOrMsb::kLsb
                                    ? partial_ops::OneHotLsbToMsb(input)
                                    : partial_ops::OneHotMsbToLsb(input)));
  }
  absl::Status HandleOneHotSel(OneHotSelect* ohs) override {
    const PartialInformation& selector = GetValue(ohs->selector()).Get({});
    const bool selector_can_be_zero =
        !query_engine_.AtLeastOneBitTrue(ohs->selector());

    std::vector<LeafTypeTreeView<PartialInformation>> cases;
    cases.reserve(ohs->cases().size());
    for (Node* case_node : ohs->cases()) {
      cases.push_back(GetValue(case_node));
    }

    XLS_ASSIGN_OR_RETURN(
        LeafTypeTree<PartialInformation> result,
        (leaf_type_tree::ZipIndex<PartialInformation, PartialInformation>(
            cases,
            [&](Type*,
                absl::Span<const PartialInformation* const> case_elements,
                absl::Span<const int64_t>)
                -> absl::StatusOr<PartialInformation> {
              std::vector<PartialInformation> case_spans;
              case_spans.reserve(case_elements.size());
              for (const PartialInformation* case_element : case_elements) {
                case_spans.push_back(*case_element);
              }
              return partial_ops::OneHotSelect(selector, case_spans,
                                               selector_can_be_zero);
            })));
    return SetValue(ohs, std::move(result));
  }
  absl::Status HandleOrReduce(BitwiseReductionOp* or_reduce) override {
    Node* input = or_reduce->operand(0);
    return SetValue(or_reduce,
                    LeafTypeTree<PartialInformation>::CreateSingleElementTree(
                        or_reduce->GetType(),
                        partial_ops::OrReduce(GetValue(input).Get({}))));
  }
  absl::Status HandleReverse(UnOp* reverse) override {
    PartialInformation result = GetValue(reverse->operand(0)).Get({});
    result.Reverse();
    return SetValue(reverse,
                    LeafTypeTree<PartialInformation>::CreateSingleElementTree(
                        reverse->GetType(), result));
  }
  absl::Status HandleSDiv(BinOp* div) override {
    Node* lhs = div->operand(0);
    Node* rhs = div->operand(1);
    return SetValue(
        div, LeafTypeTree<PartialInformation>::CreateSingleElementTree(
                 div->GetType(), partial_ops::SDiv(GetValue(lhs).Get({}),
                                                   GetValue(rhs).Get({}))));
  }
  absl::Status HandleSGe(CompareOp* ge) override {
    Node* lhs = ge->operand(0);
    Node* rhs = ge->operand(1);
    return SetValue(
        ge, LeafTypeTree<PartialInformation>::CreateSingleElementTree(
                ge->GetType(), partial_ops::SGe(GetValue(lhs).Get({}),
                                                GetValue(rhs).Get({}))));
  }
  absl::Status HandleSGt(CompareOp* gt) override {
    Node* lhs = gt->operand(0);
    Node* rhs = gt->operand(1);
    return SetValue(
        gt, LeafTypeTree<PartialInformation>::CreateSingleElementTree(
                gt->GetType(), partial_ops::SGt(GetValue(lhs).Get({}),
                                                GetValue(rhs).Get({}))));
  }
  absl::Status HandleSLe(CompareOp* le) override {
    Node* lhs = le->operand(0);
    Node* rhs = le->operand(1);
    return SetValue(
        le, LeafTypeTree<PartialInformation>::CreateSingleElementTree(
                le->GetType(), partial_ops::SLe(GetValue(lhs).Get({}),
                                                GetValue(rhs).Get({}))));
  }
  absl::Status HandleSLt(CompareOp* lt) override {
    Node* lhs = lt->operand(0);
    Node* rhs = lt->operand(1);
    return SetValue(
        lt, LeafTypeTree<PartialInformation>::CreateSingleElementTree(
                lt->GetType(), partial_ops::SLt(GetValue(lhs).Get({}),
                                                GetValue(rhs).Get({}))));
  }
  absl::Status HandleSMod(BinOp* mod) override {
    Node* lhs = mod->operand(0);
    Node* rhs = mod->operand(1);
    return SetValue(
        mod, LeafTypeTree<PartialInformation>::CreateSingleElementTree(
                 mod->GetType(), partial_ops::SMod(GetValue(lhs).Get({}),
                                                   GetValue(rhs).Get({}))));
  }
  absl::Status HandleSMul(ArithOp* mul) override {
    Node* lhs = mul->operand(0);
    Node* rhs = mul->operand(1);
    return SetValue(
        mul, LeafTypeTree<PartialInformation>::CreateSingleElementTree(
                 mul->GetType(),
                 partial_ops::SMul(GetValue(lhs).Get({}), GetValue(rhs).Get({}),
                                   mul->BitCountOrDie())));
  }
  absl::Status HandleShll(BinOp* shll) override {
    Node* input = shll->operand(0);
    Node* amount = shll->operand(1);
    return SetValue(
        shll,
        LeafTypeTree<PartialInformation>::CreateSingleElementTree(
            shll->GetType(), partial_ops::Shll(GetValue(input).Get({}),
                                               GetValue(amount).Get({}))));
  }
  absl::Status HandleShra(BinOp* shra) override {
    Node* input = shra->operand(0);
    Node* amount = shra->operand(1);
    return SetValue(
        shra,
        LeafTypeTree<PartialInformation>::CreateSingleElementTree(
            shra->GetType(), partial_ops::Shra(GetValue(input).Get({}),
                                               GetValue(amount).Get({}))));
  }
  absl::Status HandleShrl(BinOp* shrl) override {
    Node* input = shrl->operand(0);
    Node* amount = shrl->operand(1);
    return SetValue(
        shrl,
        LeafTypeTree<PartialInformation>::CreateSingleElementTree(
            shrl->GetType(), partial_ops::Shrl(GetValue(input).Get({}),
                                               GetValue(amount).Get({}))));
  }
  absl::Status HandleSignExtend(ExtendOp* sign_ext) override {
    return SetValue(
        sign_ext,
        LeafTypeTree<PartialInformation>::CreateSingleElementTree(
            sign_ext->GetType(),
            partial_ops::SignExtend(GetValue(sign_ext->operand(0)).Get({}),
                                    sign_ext->new_bit_count())));
  }
  absl::Status HandleSub(BinOp* sub) override {
    Node* lhs = sub->operand(0);
    Node* rhs = sub->operand(1);
    return SetValue(
        sub, LeafTypeTree<PartialInformation>::CreateSingleElementTree(
                 sub->GetType(), partial_ops::Sub(GetValue(lhs).Get({}),
                                                  GetValue(rhs).Get({}))));
  }
  absl::Status HandleUDiv(BinOp* div) override {
    Node* lhs = div->operand(0);
    Node* rhs = div->operand(1);
    return SetValue(
        div, LeafTypeTree<PartialInformation>::CreateSingleElementTree(
                 div->GetType(), partial_ops::UDiv(GetValue(lhs).Get({}),
                                                   GetValue(rhs).Get({}))));
  }
  absl::Status HandleUGe(CompareOp* ge) override {
    Node* lhs = ge->operand(0);
    Node* rhs = ge->operand(1);
    return SetValue(
        ge, LeafTypeTree<PartialInformation>::CreateSingleElementTree(
                ge->GetType(), partial_ops::UGe(GetValue(lhs).Get({}),
                                                GetValue(rhs).Get({}))));
  }
  absl::Status HandleUGt(CompareOp* gt) override {
    Node* lhs = gt->operand(0);
    Node* rhs = gt->operand(1);
    return SetValue(
        gt, LeafTypeTree<PartialInformation>::CreateSingleElementTree(
                gt->GetType(), partial_ops::UGt(GetValue(lhs).Get({}),
                                                GetValue(rhs).Get({}))));
  }
  absl::Status HandleULe(CompareOp* le) override {
    Node* lhs = le->operand(0);
    Node* rhs = le->operand(1);
    return SetValue(
        le, LeafTypeTree<PartialInformation>::CreateSingleElementTree(
                le->GetType(), partial_ops::ULe(GetValue(lhs).Get({}),
                                                GetValue(rhs).Get({}))));
  }
  absl::Status HandleULt(CompareOp* lt) override {
    Node* lhs = lt->operand(0);
    Node* rhs = lt->operand(1);
    return SetValue(
        lt, LeafTypeTree<PartialInformation>::CreateSingleElementTree(
                lt->GetType(), partial_ops::ULt(GetValue(lhs).Get({}),
                                                GetValue(rhs).Get({}))));
  }
  absl::Status HandleUMod(BinOp* mod) override {
    Node* lhs = mod->operand(0);
    Node* rhs = mod->operand(1);
    return SetValue(
        mod, LeafTypeTree<PartialInformation>::CreateSingleElementTree(
                 mod->GetType(), partial_ops::UMod(GetValue(lhs).Get({}),
                                                   GetValue(rhs).Get({}))));
  }
  absl::Status HandleUMul(ArithOp* mul) override {
    Node* lhs = mul->operand(0);
    Node* rhs = mul->operand(1);
    return SetValue(
        mul, LeafTypeTree<PartialInformation>::CreateSingleElementTree(
                 mul->GetType(),
                 partial_ops::UMul(GetValue(lhs).Get({}), GetValue(rhs).Get({}),
                                   mul->BitCountOrDie())));
  }
  absl::Status HandleXorReduce(BitwiseReductionOp* xor_reduce) override {
    Node* input = xor_reduce->operand(0);
    return SetValue(xor_reduce,
                    LeafTypeTree<PartialInformation>::CreateSingleElementTree(
                        xor_reduce->GetType(),
                        partial_ops::XorReduce(GetValue(input).Get({}))));
  }
  absl::Status HandleZeroExtend(ExtendOp* zero_ext) override {
    return SetValue(
        zero_ext,
        LeafTypeTree<PartialInformation>::CreateSingleElementTree(
            zero_ext->GetType(),
            partial_ops::ZeroExtend(GetValue(zero_ext->operand(0)).Get({}),
                                    zero_ext->new_bit_count())));
  }

 protected:
  bool IndexMightBeEquivalent(const PartialInformation& index,
                              int64_t concrete_index, int64_t bound,
                              bool index_clamped) const override {
    CHECK_LT(concrete_index, bound);
    if (Bits::MinBitCountUnsigned(concrete_index) > index.BitCount()) {
      // `index` is too narrow to represent `concrete_index`.
      return false;
    }
    Bits concrete_index_bits = UBits(concrete_index, index.BitCount());
    Interval concrete_interval = Interval::Precise(concrete_index_bits);
    if (index_clamped && concrete_index == bound - 1) {
      concrete_interval = Interval::Closed(concrete_index_bits,
                                           Bits::AllOnes(index.BitCount()));
    }
    bool intersects_ternary =
        !index.Ternary().has_value() ||
        interval_ops::CoversTernary(concrete_interval, *index.Ternary());
    bool intersects_range =
        !index.Range().has_value() ||
        !IntervalSet::Disjoint(IntervalSet::Of({concrete_interval}),
                               *index.Range());
    return intersects_ternary && intersects_range;
  }

  absl::StatusOr<PartialInformation> SelectElements(
      const PartialInformation& selector,
      absl::Span<const PartialInformation* const> data_sources,
      bool has_default_value) {
    const PartialInformation* default_value = nullptr;
    if (has_default_value) {
      default_value = data_sources.back();
      data_sources.remove_suffix(1);
    }
    absl::Span<const PartialInformation* const> cases = data_sources;

    PartialInformation result =
        PartialInformation::Impossible(cases.front()->BitCount());

    for (int64_t i = 0; i < cases.size(); ++i) {
      if (Bits::MinBitCountUnsigned(i) > selector.BitCount()) {
        continue;
      }
      if (selector.IsCompatibleWith(UBits(i, selector.BitCount()))) {
        result.MeetWith(*cases[i]);
      }
    }
    if (Bits::MinBitCountUnsigned(cases.size()) <= selector.BitCount() &&
        selector.IsCompatibleWith(
            Interval::Closed(UBits(cases.size(), selector.BitCount()),
                             Bits::AllOnes(selector.BitCount())))) {
      // The default value might be selected.
      CHECK_NE(default_value, nullptr);
      result.MeetWith(*default_value);
    }
    return result;
  }

  absl::StatusOr<PartialInformation> PrioritySelectElements(
      const PartialInformation& selector,
      absl::Span<const PartialInformation* const> data_sources,
      bool selector_can_be_zero) {
    PartialInformation default_value = *data_sources.back();
    absl::Span<const PartialInformation* const> cases =
        data_sources.subspan(0, data_sources.size() - 1);

    if (selector_can_be_zero &&
        !selector.IsCompatibleWith(Bits(selector.BitCount()))) {
      selector_can_be_zero = false;
    }

    PartialInformation result =
        selector_can_be_zero
            ? default_value
            : PartialInformation::Impossible(cases.front()->BitCount());

    TernaryVector selector_target(cases.size(), TernaryValue::kUnknown);
    for (int64_t i = 0; i < cases.size(); ++i) {
      selector_target[i] = TernaryValue::kKnownOne;
      if (selector.IsCompatibleWith(selector_target)) {
        result.MeetWith(*cases[i]);
      }
      selector_target[i] = TernaryValue::kKnownZero;
    }
    return result;
  }

  absl::StatusOr<PartialInformation> JoinElements(
      Type* element_type,
      absl::Span<const PartialInformation* const> data_sources,
      absl::Span<const LeafTypeTreeView<PartialInformation>> control_sources,
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
      return PartialInformation::Unconstrained(element_type->GetFlatBitCount());
    }
    PartialInformation result = *data_sources.front();
    for (const PartialInformation* const data_source :
         data_sources.subspan(1)) {
      result.MeetWith(*data_source);
    }
    return result;
  }
};

// Returns whether the operation will be computationally expensive to
// compute. The ternary query engine is intended to be fast so the analysis of
// these expensive operations is skipped with the effect being all bits are
// considered unknown. Operations and limits can be added as needed when
// pathological cases are encountered.
bool IsExpensiveToEvaluate(
    Node* node,
    absl::Span<const LeafTypeTree<PartialInformation>* const> operand_infos) {
  // How many bits we allow to be unconstrained when we need to intersect
  // multiple possibilities.
  static constexpr int64_t kIndexBitLimit = 10;
  static_assert(kIndexBitLimit < 63);
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
    int64_t index_possibilities = 1;
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
      if (operand_infos.at(operand) == nullptr ||
          operand_infos.at(operand)->Get({}).IsUnconstrained()) {
        int64_t unknown_bits =
            node->operand(operand)->GetType()->GetFlatBitCount();
        int64_t unrestricted_possibilities =
            unknown_bits < 63 ? (int64_t{1} << unknown_bits)
                              : std::numeric_limits<int64_t>::max();
        index_possibilities =
            SaturatingMul(index_possibilities, unrestricted_possibilities)
                .result;
        continue;
      }
      PartialInformation operand_info = operand_infos.at(operand)->Get({});
      int64_t ternary_possibilities = std::numeric_limits<int64_t>::max();
      int64_t range_possibilities = std::numeric_limits<int64_t>::max();
      if (operand_info.Ternary().has_value()) {
        int64_t unknown_bits =
            absl::c_count(*operand_info.Ternary(), TernaryValue::kUnknown);
        ternary_possibilities = unknown_bits < 63
                                    ? (int64_t{1} << unknown_bits)
                                    : std::numeric_limits<int64_t>::max();
      }
      if (operand_info.Range().has_value()) {
        range_possibilities = operand_info.Range()->Size().value_or(
            std::numeric_limits<int64_t>::max());
      }
      int64_t operand_possibilities =
          std::min(ternary_possibilities, range_possibilities);
      index_possibilities =
          SaturatingMul(index_possibilities, operand_possibilities).result;
    }
    return index_possibilities >= (int64_t{1} << kIndexBitLimit);
  }

  return false;
}

};  // namespace

LeafTypeTree<PartialInformation> PartialInfoQueryEngine::ComputeInfo(
    Node* node,
    absl::Span<const LeafTypeTree<PartialInformation>* const> operand_infos)
    const {
  PartialInfoVisitor visitor;
  if (IsExpensiveToEvaluate(node, operand_infos)) {
    CHECK_OK(visitor.DefaultHandler(node));
  } else {
    for (const auto& [operand, operand_info] :
         iter::zip(node->operands(), operand_infos)) {
      CHECK_OK(visitor.InjectValue(operand, operand_info));
    }
    CHECK_OK(node->VisitSingleNode(&visitor));
  }
  absl::flat_hash_map<Node*,
                      std::unique_ptr<SharedLeafTypeTree<PartialInformation>>>
      result = std::move(visitor).ToStoredValues();
  return std::move(*result.at(node)).ToOwned();
}

absl::Status PartialInfoQueryEngine::MergeWithGiven(
    PartialInformation& info, const PartialInformation& given) const {
  info.JoinWith(given);
  return absl::OkStatus();
}

std::optional<SharedTernaryTree> PartialInfoQueryEngine::GetTernary(
    Node* node) const {
  std::optional<SharedLeafTypeTree<PartialInformation>> info_tree =
      GetInfo(node);
  if (!info_tree.has_value() ||
      absl::c_all_of(info_tree->elements(), [](const PartialInformation& info) {
        return !info.Ternary().has_value();
      })) {
    return std::nullopt;
  }
  absl::InlinedVector<TernaryVector, 1> ternary_elements;
  ternary_elements.reserve(info_tree->elements().size());
  for (const PartialInformation& info : info_tree->elements()) {
    if (info.Ternary().has_value()) {
      ternary_elements.push_back(*info.Ternary());
    } else {
      ternary_elements.push_back(
          TernaryVector(info.BitCount(), TernaryValue::kUnknown));
    }
  }
  return TernaryTree::CreateFromVector(info_tree->type(),
                                       std::move(ternary_elements))
      .AsShared();
}

IntervalSetTree PartialInfoQueryEngine::GetIntervals(Node* node) const {
  std::optional<SharedLeafTypeTree<PartialInformation>> info_tree =
      GetInfo(node);
  if (!info_tree.has_value()) {
    absl::StatusOr<IntervalSetTree> result =
        IntervalSetTree::CreateFromFunction(
            node->GetType(),
            [](Type* leaf_type) -> absl::StatusOr<IntervalSet> {
              return IntervalSet::Maximal(leaf_type->GetFlatBitCount());
            });
    CHECK_OK(result);
    return *std::move(result);
  }
  absl::InlinedVector<IntervalSet, 1> range_elements;
  range_elements.reserve(info_tree->elements().size());
  for (const PartialInformation& info : info_tree->elements()) {
    if (info.Range().has_value()) {
      range_elements.push_back(*info.Range());
    } else {
      range_elements.push_back(IntervalSet::Maximal(info.BitCount()));
    }
  }
  return IntervalSetTree::CreateFromVector(info_tree->type(),
                                           std::move(range_elements));
}

bool PartialInfoQueryEngine::AtMostOneTrue(
    absl::Span<TreeBitLocation const> bits) const {
  int64_t maybe_one_count = 0;
  for (const TreeBitLocation& location : bits) {
    if (!IsKnown(location) || IsOne(location)) {
      maybe_one_count++;
    }
  }
  return maybe_one_count <= 1;
}

bool PartialInfoQueryEngine::AtLeastOneTrue(
    absl::Span<TreeBitLocation const> bits) const {
  for (const TreeBitLocation& location : bits) {
    if (IsOne(location)) {
      return true;
    }
  }
  return false;
}

bool PartialInfoQueryEngine::KnownEquals(const TreeBitLocation& a,
                                         const TreeBitLocation& b) const {
  return IsKnown(a) && IsKnown(b) && IsOne(a) == IsOne(b);
}

bool PartialInfoQueryEngine::KnownNotEquals(const TreeBitLocation& a,
                                            const TreeBitLocation& b) const {
  return IsKnown(a) && IsKnown(b) && IsOne(a) != IsOne(b);
}

bool PartialInfoQueryEngine::Covers(Node* node, const Bits& value) const {
  if (!node->GetType()->IsBits() ||
      node->BitCountOrDie() != value.bit_count()) {
    // The type doesn't match, so `node` can't possibly cover it.
    return false;
  }
  if (!IsTracked(node)) {
    return true;
  }
  std::optional<SharedLeafTypeTree<PartialInformation>> info_tree =
      GetInfo(node);
  if (!info_tree.has_value()) {
    return true;
  }
  return info_tree->Get({}).IsCompatibleWith(value);
}

}  // namespace xls
