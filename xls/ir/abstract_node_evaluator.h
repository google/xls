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

#ifndef XLS_IR_ABSTRACT_NODE_EVALUATOR_H_
#define XLS_IR_ABSTRACT_NODE_EVALUATOR_H_

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "cppitertools/zip.hpp"
#include "xls/common/math_util.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/lsb_or_msb.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"

namespace xls {

// An abstract evaluation visitor for XLS nodes.
//
// Works by calling the appropriate method on the AbstractEvaluatorT and storing
// the result. Must be populated by walking the IR in reverse-post-order.
//
// Additional operations can be implemented by overriding the required
// operations. Operations which return raw AbstractEvaluatorT::Vector values
// should call SetValue(node, value) before returning so predefined operations
// can see their values.
//
// To use this one should probably override HandleParam, some other source of
// values like RegisterRead or DefaultHandler to ensure they have some base
// values to start with.
//
// Once visited values can be obtained by calling GetValue or GetCompoundValue.
template <typename AbstractEvaluatorT>
class AbstractNodeEvaluator : public DfsVisitorWithDefault {
 public:
  using ElementT = typename AbstractEvaluatorT::Element;
  using LeafValueT = typename AbstractEvaluatorT::Vector;
  using LeafSpanT = typename AbstractEvaluatorT::Span;

  explicit AbstractNodeEvaluator(AbstractEvaluatorT& evaluator)
      : evaluator_(evaluator) {}
  absl::Status DefaultHandler(Node* node) override {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s is not supported by node evaluator", node->ToString()));
  }

  absl::Status HandleAdd(BinOp* add) override {
    XLS_ASSIGN_OR_RETURN(auto lhs, GetValue(add->operand(0)));
    XLS_ASSIGN_OR_RETURN(auto rhs, GetValue(add->operand(1)));
    return SetValue(add, evaluator_.Add(lhs, rhs));
  }

  absl::Status HandleAndReduce(BitwiseReductionOp* and_reduce) override {
    XLS_ASSIGN_OR_RETURN(auto args, GetValue(and_reduce->operand(0)));
    return SetValue(and_reduce, evaluator_.AndReduce(args));
  }

  absl::Status HandleArray(Array* array) override {
    XLS_ASSIGN_OR_RETURN(std::vector<LeafTypeTreeView<LeafValueT>> elements,
                         GetCompoundValueList(array->operands()),
                         _ << "from: " << array);
    XLS_ASSIGN_OR_RETURN(
        LeafTypeTree<LeafValueT> result,
        (leaf_type_tree::CreateArray<typename AbstractEvaluatorT::Vector>(
            array->GetType()->AsArrayOrDie(), elements)));
    return SetValue(array, std::move(result));
  }

  absl::Status HandleArrayConcat(ArrayConcat* array_concat) override {
    typename LeafTypeTree<LeafValueT>::DataContainerT leaves;
    leaves.reserve(array_concat->GetType()->AsArrayOrDie()->leaf_count());
    for (Node* o : array_concat->operands()) {
      XLS_ASSIGN_OR_RETURN(LeafTypeTreeView<LeafValueT> v, GetCompoundValue(o),
                           _ << "from: " << array_concat);
      absl::c_copy(v.elements(), std::back_inserter(leaves));
    }
    return SetValue(array_concat,
                    LeafTypeTree<LeafValueT>::CreateFromVector(
                        array_concat->GetType(), std::move(leaves)));
  }

  absl::Status HandleArrayIndex(ArrayIndex* array_index) override {
    XLS_ASSIGN_OR_RETURN(LeafTypeTreeView<LeafValueT> array_view,
                         GetCompoundValue(array_index->array()));
    SharedLeafTypeTree<LeafValueT> result = array_view.AsShared();
    for (Node* index : array_index->indices()) {
      XLS_ASSIGN_OR_RETURN(LeafSpanT index_value, GetValue(index));
      XLS_ASSIGN_OR_RETURN(ArrayType * array_type, result.type()->AsArray());
      const bool has_default =
          !IsPowerOfTwo(static_cast<uint64_t>(array_type->size())) ||
          index_value.size() != FloorOfLog2(array_type->size());

      if (array_type->element_type()->IsBits()) {
        std::vector<LeafSpanT> elements;
        elements.reserve(array_type->size());
        for (int64_t i = 0; i < array_type->size(); ++i) {
          elements.push_back(result.Get({i}));
        }
        std::optional<LeafSpanT> default_value;
        if (has_default) {
          default_value = std::move(elements.back());
          elements.pop_back();
        }
        return SetValue(array_index, evaluator_.Select(index_value, elements,
                                                       default_value));
      }

      std::vector<LeafTypeTreeView<LeafValueT>> elements;
      elements.reserve(array_type->size());
      for (int64_t i = 0; i < array_type->size(); ++i) {
        elements.push_back(result.AsView({i}));
      }
      XLS_ASSIGN_OR_RETURN(
          LeafTypeTree<LeafValueT> index_result,
          ZipPiecewise(array_type->element_type(), elements,
                       [&](absl::Span<LeafSpanT const> values) ->
                       typename AbstractEvaluatorT::Vector {
                         absl::Span<LeafSpanT const> cases = values;
                         if (has_default) {
                           cases.remove_suffix(1);
                         }
                         return evaluator_.Select(
                             index_value, cases,
                             has_default ? std::make_optional(values.back())
                                         : std::nullopt);
                       }));
      result = std::move(index_result).AsShared();
    }
    return SetValue(array_index, std::move(result));
  }

  absl::Status HandleArraySlice(ArraySlice* array_slice) override {
    XLS_ASSIGN_OR_RETURN(LeafSpanT start_value, GetValue(array_slice->start()));
    XLS_ASSIGN_OR_RETURN(LeafTypeTreeView<LeafValueT> array_view,
                         GetCompoundValue(array_slice->array()));

    XLS_ASSIGN_OR_RETURN(ArrayType * array_type, array_view.type()->AsArray());
    ArrayType* result_type = array_slice->package()->GetArrayType(
        array_slice->width(), array_type->element_type());

    std::vector<LeafTypeTree<LeafValueT>> slices;
    slices.reserve(array_type->size());
    for (int64_t start = 0; start < array_type->size(); ++start) {
      XLS_ASSIGN_OR_RETURN(
          LeafTypeTree<LeafValueT> slice,
          leaf_type_tree::SliceArray(result_type, array_view.AsView(), start));
    }

    const bool has_default =
        !IsPowerOfTwo(static_cast<uint64_t>(array_type->size())) ||
        start_value.size() != FloorOfLog2(array_type->size());
    XLS_ASSIGN_OR_RETURN(
        LeafTypeTree<LeafValueT> result,
        ZipPiecewise(array_type->element_type(), slices,
                     [&](absl::Span<LeafSpanT const> values) ->
                     typename AbstractEvaluatorT::Vector {
                       absl::Span<LeafSpanT const> cases = values;
                       if (has_default) {
                         cases.remove_suffix(1);
                       }
                       return evaluator_.Select(
                           start_value, cases,
                           has_default ? std::make_optional(values.back())
                                       : std::nullopt);
                     }));
    return SetValue(array_slice, std::move(result));
  }

  absl::Status HandleArrayUpdate(ArrayUpdate* update) override {
    XLS_ASSIGN_OR_RETURN(LeafTypeTreeView<LeafValueT> to_update,
                         GetCompoundValue(update->array_to_update()));
    XLS_ASSIGN_OR_RETURN(LeafTypeTreeView<LeafValueT> update_value,
                         GetCompoundValue(update->update_value()));
    XLS_ASSIGN_OR_RETURN(std::vector<LeafSpanT> indices,
                         GetValueList(update->indices()));

    LeafTypeTree<LeafValueT> updated = leaf_type_tree::Clone(to_update);
    XLS_RETURN_IF_ERROR(leaf_type_tree::ForEachSubArray<LeafValueT>(
        updated.AsMutableView(), indices.size(),
        [&](MutableLeafTypeTreeView<LeafValueT> entry,
            absl::Span<const int64_t> concrete_indices) -> absl::Status {
          LeafTypeTree<LeafValueT> updated_entry =
              leaf_type_tree::Zip<LeafValueT, LeafValueT, LeafValueT>(
                  entry.AsView(), update_value.AsView(),
                  [&](const LeafValueT& entry_element,
                      const LeafValueT& update_element) {
                    std::vector<ElementT> index_matches;
                    index_matches.reserve(indices.size());
                    for (const auto& [index, concrete_index] :
                         iter::zip(indices, concrete_indices)) {
                      if (Bits::MinBitCountUnsigned(concrete_index) <=
                          index.size()) {
                        index_matches.push_back(evaluator_.Equals(
                            index, evaluator_.BitsToVector(
                                       UBits(concrete_index, index.size()))));
                      } else {
                        index_matches.push_back(evaluator_.Zero());
                      }
                    }
                    LeafSpanT update_span = update_element;
                    return evaluator_.PrioritySelect(
                        /*selector=*/evaluator_.AndReduce(index_matches),
                        /*cases=*/
                        absl::MakeConstSpan(&update_span, 1),
                        /*selector_can_be_zero=*/true,
                        /*default_value=*/entry_element);
                  });
          return leaf_type_tree::ReplaceElements(entry, updated_entry.AsView());
        }));
    return SetValue(update, std::move(updated).AsShared());
  }

  absl::Status HandleBitSlice(BitSlice* bit_slice) override {
    XLS_ASSIGN_OR_RETURN(auto src, GetValue(bit_slice->operand(0)));
    return SetValue(bit_slice, evaluator_.BitSlice(src, bit_slice->start(),
                                                   bit_slice->width()));
  }
  absl::Status HandleBitSliceUpdate(BitSliceUpdate* update) override {
    XLS_ASSIGN_OR_RETURN(auto to_update, GetValue(update->to_update()));
    XLS_ASSIGN_OR_RETURN(auto update_value, GetValue(update->update_value()));
    XLS_ASSIGN_OR_RETURN(auto start, GetValue(update->start()));
    return SetValue(update,
                    evaluator_.BitSliceUpdate(to_update, start, update_value));
  }
  absl::Status HandleConcat(Concat* concat) override {
    XLS_ASSIGN_OR_RETURN(auto args, GetValueList(concat->operands()));
    return SetValue(concat, evaluator_.Concat(args));
  }
  absl::Status HandleDecode(Decode* decode) override {
    XLS_ASSIGN_OR_RETURN(auto input, GetValue(decode->operand(0)));
    return SetValue(decode, evaluator_.Decode(input, decode->width()));
  }
  absl::Status HandleDynamicBitSlice(
      DynamicBitSlice* dynamic_bit_slice) override {
    return SetValue(
        dynamic_bit_slice,
        evaluator_.DynamicBitSlice(*GetValue(dynamic_bit_slice->operand(0)),
                                   *GetValue(dynamic_bit_slice->start()),
                                   dynamic_bit_slice->width()));
  }
  absl::Status HandleEncode(Encode* encode) override {
    XLS_ASSIGN_OR_RETURN(auto input, GetValue(encode->operand(0)));
    return SetValue(encode, evaluator_.Encode(input));
  }

  absl::Status HandleEq(CompareOp* eq) override {
    XLS_ASSIGN_OR_RETURN(LeafTypeTreeView<LeafValueT> lhs,
                         GetCompoundValue(eq->operand(0)), _ << "from: " << eq);
    XLS_ASSIGN_OR_RETURN(LeafTypeTreeView<LeafValueT> rhs,
                         GetCompoundValue(eq->operand(1)), _ << "from: " << eq);
    XLS_RET_CHECK(lhs.type()->IsEqualTo(rhs.type())) << eq;
    std::vector<typename AbstractEvaluatorT::Element> values;
    values.reserve(lhs.elements().size());
    for (int64_t i = 0; i < lhs.elements().size(); ++i) {
      values.push_back(
          evaluator().Equals(lhs.elements().at(i), rhs.elements().at(i)));
    }
    if (values.size() == 1) {
      return SetValue(eq, std::move(values));
    }
    return SetValue(
        eq, typename AbstractEvaluatorT::Vector{evaluator().AndReduce(values)});
  }

  absl::Status HandleGate(Gate* gate) override {
    XLS_ASSIGN_OR_RETURN(auto cond, GetValue(gate->condition()));
    XLS_RET_CHECK_EQ(cond.size(), 1);
    if (gate->GetType()->IsBits()) {
      XLS_ASSIGN_OR_RETURN(auto data, GetValue(gate->data()));
      XLS_RET_CHECK_EQ(cond.size(), 1);
      return SetValue(gate, evaluator_.Gate(cond[0], data));
    }
    XLS_ASSIGN_OR_RETURN(
        LeafTypeTree<LeafValueT> result,
        ZipPiecewise(gate->GetType(), {gate->data()},
                     [&](absl::Span<typename AbstractEvaluatorT::Span const> v)
                         -> AbstractEvaluatorT::Vector {
                       return evaluator().Gate(cond[0], v[0]);
                     }));
    return SetValue(gate, std::move(result));
  }

  absl::Status HandleIdentity(UnOp* identity) override {
    XLS_ASSIGN_OR_RETURN(LeafTypeTreeView<LeafValueT> v,
                         GetCompoundValue(identity->operand(0)));
    auto elements = v.elements();
    return SetValue(identity,
                    LeafTypeTree<LeafValueT>::CreateFromVector(
                        identity->GetType(),
                        typename LeafTypeTree<LeafValueT>::DataContainerT(
                            elements.begin(), elements.end())));
  }
  absl::Status HandleLiteral(Literal* literal) override {
    XLS_ASSIGN_OR_RETURN(
        LeafTypeTree<Value> v_ltt,
        ValueToLeafTypeTree(literal->value(), literal->GetType()));
    XLS_RET_CHECK(absl::c_all_of(
        v_ltt.elements(),
        [](const Value& v) { return v.IsBits() || v.IsToken(); }))
        << literal << " has non-bits and non-token leaf.";
    LeafTypeTree<LeafValueT> result =
        leaf_type_tree::Map<typename AbstractEvaluatorT::Vector, Value>(
            v_ltt.AsView(), [&](const Value& value) {
              if (value.IsToken()) {
                return evaluator_.BitsToVector(Bits());
              }
              return evaluator_.BitsToVector(value.bits());
            });
    return SetValue(literal, std::move(result));
  }
  absl::Status HandleNaryAnd(NaryOp* and_op) override {
    XLS_ASSIGN_OR_RETURN(auto args, GetValueList(and_op->operands()));
    return SetValue(and_op, evaluator_.BitwiseAnd(args));
  }
  absl::Status HandleNaryNand(NaryOp* and_op) override {
    XLS_ASSIGN_OR_RETURN(auto args, GetValueList(and_op->operands()));
    return SetValue(and_op, evaluator_.BitwiseNot(evaluator_.BitwiseAnd(args)));
  }
  absl::Status HandleNaryNor(NaryOp* nor_op) override {
    XLS_ASSIGN_OR_RETURN(auto args, GetValueList(nor_op->operands()));
    return SetValue(nor_op, evaluator_.BitwiseNot(evaluator_.BitwiseOr(args)));
  }
  absl::Status HandleNaryOr(NaryOp* or_op) override {
    XLS_ASSIGN_OR_RETURN(auto args, GetValueList(or_op->operands()));
    return SetValue(or_op, evaluator_.BitwiseOr(args));
  }
  absl::Status HandleNaryXor(NaryOp* xor_op) override {
    XLS_ASSIGN_OR_RETURN(auto args, GetValueList(xor_op->operands()));
    return SetValue(xor_op, evaluator_.BitwiseXor(args));
  }
  absl::Status HandleNe(CompareOp* ne) override {
    XLS_ASSIGN_OR_RETURN(LeafTypeTreeView<LeafValueT> lhs,
                         GetCompoundValue(ne->operand(0)), _ << "from: " << ne);
    XLS_ASSIGN_OR_RETURN(LeafTypeTreeView<LeafValueT> rhs,
                         GetCompoundValue(ne->operand(1)), _ << "from: " << ne);
    XLS_RET_CHECK(lhs.type()->IsEqualTo(rhs.type())) << ne;
    std::vector<typename AbstractEvaluatorT::Element> values;
    values.reserve(lhs.elements().size());
    for (int64_t i = 0; i < lhs.elements().size(); ++i) {
      values.push_back(
          evaluator().Equals(lhs.elements().at(i), rhs.elements().at(i)));
    }
    if (values.size() == 1) {
      return SetValue(ne, typename AbstractEvaluatorT::Vector{
                              evaluator().Not(values.front())});
    }
    return SetValue(ne, typename AbstractEvaluatorT::Vector{evaluator().Not(
                            evaluator().AndReduce(values).front())});
  };
  absl::Status HandleNeg(UnOp* neg) override {
    XLS_ASSIGN_OR_RETURN(auto v, GetValue(neg->operand(0)));
    return SetValue(neg, evaluator_.Neg(v));
  }
  absl::Status HandleNot(UnOp* not_op) override {
    XLS_ASSIGN_OR_RETURN(auto v, GetValue(not_op->operand(0)));
    return SetValue(not_op, evaluator_.BitwiseNot(v));
  }
  absl::Status HandleOneHot(OneHot* one_hot) override {
    XLS_ASSIGN_OR_RETURN(auto v, GetValue(one_hot->operand(0)));
    return SetValue(one_hot, one_hot->priority() == LsbOrMsb::kLsb
                                 ? evaluator_.OneHotLsbToMsb(v)
                                 : evaluator_.OneHotMsbToLsb(v));
  }

  absl::Status HandleOneHotSel(OneHotSelect* sel) override {
    bool selector_can_be_zero = !sel->selector()->Is<OneHot>();
    XLS_ASSIGN_OR_RETURN(auto selector, GetValue(sel->selector()));
    if (sel->GetType()->IsBits()) {
      XLS_ASSIGN_OR_RETURN(auto args, GetValueList(sel->cases()));
      return SetValue(
          sel, evaluator_.OneHotSelect(selector, args, selector_can_be_zero));
    }
    XLS_ASSIGN_OR_RETURN(
        LeafTypeTree<LeafValueT> result,
        ZipPiecewise(
            sel->GetType(), sel->cases(),
            [&](absl::Span<typename AbstractEvaluatorT::Span const> values) ->
            typename AbstractEvaluatorT::Vector {
              return evaluator().OneHotSelect(selector, values,
                                              selector_can_be_zero);
            }));
    return SetValue(sel, std::move(result));
  };

  absl::Status HandlePrioritySel(PrioritySelect* sel) override {
    bool selector_can_be_zero = !sel->selector()->Is<OneHot>();
    XLS_ASSIGN_OR_RETURN(auto selector, GetValue(sel->selector()));
    if (sel->GetType()->IsBits()) {
      XLS_ASSIGN_OR_RETURN(auto args, GetValueList(sel->cases()));
      XLS_ASSIGN_OR_RETURN(auto default_value, GetValue(sel->default_value()));
      return SetValue(
          sel, evaluator_.PrioritySelect(selector, args, selector_can_be_zero,
                                         default_value));
    }
    XLS_ASSIGN_OR_RETURN(
        LeafTypeTree<LeafValueT> result,
        ZipPiecewise(
            sel->GetType(), sel->operands().subspan(1),
            [&](absl::Span<typename AbstractEvaluatorT::Span const> values) ->
            typename AbstractEvaluatorT::Vector {
              absl::Span<typename AbstractEvaluatorT::Span const> case_values =
                  values;
              auto default_value = case_values.back();
              case_values.remove_suffix(1);
              return evaluator().PrioritySelect(
                  selector, case_values, selector_can_be_zero, default_value);
            }));
    return SetValue(sel, std::move(result));
  }

  absl::Status HandleOrReduce(BitwiseReductionOp* or_reduce) override {
    XLS_ASSIGN_OR_RETURN(auto args, GetValue(or_reduce->operand(0)));
    return SetValue(or_reduce, evaluator_.OrReduce(args));
  }
  absl::Status HandleReverse(UnOp* reverse) override {
    XLS_ASSIGN_OR_RETURN(auto vec, GetOwnedValue(reverse->operand(0)));
    absl::c_reverse(vec);
    return SetValue(reverse, std::move(vec));
  }
  absl::Status HandleSDiv(BinOp* div) override {
    XLS_ASSIGN_OR_RETURN(auto lhs, GetValue(div->operand(0)));
    XLS_ASSIGN_OR_RETURN(auto rhs, GetValue(div->operand(1)));
    return SetValue(div, evaluator_.SDiv(lhs, rhs));
  }
  absl::Status HandleSGe(CompareOp* ge) override {
    XLS_ASSIGN_OR_RETURN(auto lhs, GetValue(ge->operand(0)));
    XLS_ASSIGN_OR_RETURN(auto rhs, GetValue(ge->operand(1)));
    using VectorT = AbstractEvaluatorT::Vector;
    return SetValue(ge,
                    VectorT({evaluator_.Not(evaluator_.SLessThan(lhs, rhs))}));
  }
  absl::Status HandleSGt(CompareOp* gt) override {
    XLS_ASSIGN_OR_RETURN(auto lhs, GetValue(gt->operand(0)));
    XLS_ASSIGN_OR_RETURN(auto rhs, GetValue(gt->operand(1)));
    using VectorT = AbstractEvaluatorT::Vector;
    // NB left and right swapped.
    return SetValue(gt, VectorT({evaluator_.SLessThan(rhs, lhs)}));
  }
  absl::Status HandleSLe(CompareOp* le) override {
    XLS_ASSIGN_OR_RETURN(auto lhs, GetValue(le->operand(0)));
    XLS_ASSIGN_OR_RETURN(auto rhs, GetValue(le->operand(1)));
    using VectorT = AbstractEvaluatorT::Vector;
    // NB left and right swapped.
    return SetValue(le,
                    VectorT({evaluator_.Not(evaluator_.SLessThan(rhs, lhs))}));
  }
  absl::Status HandleSLt(CompareOp* lt) override {
    XLS_ASSIGN_OR_RETURN(auto lhs, GetValue(lt->operand(0)));
    XLS_ASSIGN_OR_RETURN(auto rhs, GetValue(lt->operand(1)));
    using VectorT = AbstractEvaluatorT::Vector;
    return SetValue(lt, VectorT({evaluator_.SLessThan(lhs, rhs)}));
  }
  absl::Status HandleSMod(BinOp* mod) override {
    XLS_ASSIGN_OR_RETURN(auto lhs, GetValue(mod->operand(0)));
    XLS_ASSIGN_OR_RETURN(auto rhs, GetValue(mod->operand(1)));
    return SetValue(mod, evaluator_.SMod(lhs, rhs));
  }
  absl::Status HandleSMul(ArithOp* mul) override {
    XLS_ASSIGN_OR_RETURN(auto lhs, GetValue(mul->operand(0)));
    XLS_ASSIGN_OR_RETURN(auto rhs, GetValue(mul->operand(1)));
    auto result = evaluator_.SMul(lhs, rhs);
    int64_t expected_width = mul->BitCountOrDie();
    if (result.size() > expected_width) {
      result = evaluator_.BitSlice(result, 0, expected_width);
    } else if (result.size() < expected_width) {
      result = evaluator_.SignExtend(result, expected_width);
    }
    return SetValue(mul, std::move(result));
  };
  absl::Status HandleSel(Select* sel) override {
    XLS_ASSIGN_OR_RETURN(auto selector, GetValue(sel->selector()));
    bool has_default = sel->default_value().has_value();
    if (sel->GetType()->IsBits()) {
      XLS_ASSIGN_OR_RETURN(auto args, GetValueList(sel->cases()));
      std::optional<typename AbstractEvaluatorT::Span> default_value;
      if (sel->default_value()) {
        XLS_ASSIGN_OR_RETURN(default_value, GetValue(*sel->default_value()));
      }
      return SetValue(sel, evaluator_.Select(selector, args, default_value));
    }
    XLS_ASSIGN_OR_RETURN(
        LeafTypeTree<LeafValueT> result,
        ZipPiecewise(
            sel->GetType(),
            sel->operands().subspan(Select::kSelectorOperand + 1),
            [&](absl::Span<typename AbstractEvaluatorT::Span const> values) ->
            typename AbstractEvaluatorT::Vector {
              CHECK(!has_default || values.size() == sel->cases().size() + 1)
                  << "values size: " << values.size() << " sel: " << sel;
              return evaluator_.Select(
                  selector, values.subspan(0, sel->cases().size()),
                  has_default ? std::make_optional(values.back())
                              : std::nullopt);
            }));
    return SetValue(sel, std::move(result));
  }
  absl::Status HandleShll(BinOp* shll) override {
    XLS_ASSIGN_OR_RETURN(auto lhs, GetValue(shll->operand(0)));
    XLS_ASSIGN_OR_RETURN(auto rhs, GetValue(shll->operand(1)));
    return SetValue(shll, evaluator_.ShiftLeftLogical(lhs, rhs));
  }
  absl::Status HandleShra(BinOp* shra) override {
    XLS_ASSIGN_OR_RETURN(auto lhs, GetValue(shra->operand(0)));
    XLS_ASSIGN_OR_RETURN(auto rhs, GetValue(shra->operand(1)));
    return SetValue(shra, evaluator_.ShiftRightArith(lhs, rhs));
  }
  absl::Status HandleShrl(BinOp* shrl) override {
    XLS_ASSIGN_OR_RETURN(auto lhs, GetValue(shrl->operand(0)));
    XLS_ASSIGN_OR_RETURN(auto rhs, GetValue(shrl->operand(1)));
    return SetValue(shrl, evaluator_.ShiftRightLogical(lhs, rhs));
  }
  absl::Status HandleSignExtend(ExtendOp* sign_ext) override {
    XLS_ASSIGN_OR_RETURN(auto v, GetValue(sign_ext->operand(0)));
    return SetValue(sign_ext,
                    evaluator_.SignExtend(v, sign_ext->new_bit_count()));
  }
  absl::Status HandleSub(BinOp* sub) override {
    XLS_ASSIGN_OR_RETURN(auto lhs, GetValue(sub->operand(0)));
    XLS_ASSIGN_OR_RETURN(auto rhs, GetValue(sub->operand(1)));
    return SetValue(sub, evaluator_.Sub(lhs, rhs));
  }
  absl::Status HandleTuple(Tuple* tuple) override {
    std::vector<LeafTypeTreeView<LeafValueT>> views;
    views.reserve(tuple->size());
    for (auto* e : tuple->operands()) {
      XLS_ASSIGN_OR_RETURN(auto v, GetCompoundValue(e));
      views.push_back(v);
    }
    XLS_ASSIGN_OR_RETURN(
        LeafTypeTree<LeafValueT> result,
        leaf_type_tree::CreateTuple<typename AbstractEvaluatorT::Vector>(
            tuple->GetType()->AsTupleOrDie(), views));
    return SetValue(tuple, std::move(result));
  }

  absl::Status HandleTupleIndex(TupleIndex* index) override {
    XLS_ASSIGN_OR_RETURN(auto tup, GetCompoundValue(index->operand(0)));
    auto ltt = tup.AsView({index->index()});
    return SetValue(index,
                    LeafTypeTree<LeafValueT>(ltt.type(), ltt.elements()));
  }

  absl::Status HandleUDiv(BinOp* div) override {
    XLS_ASSIGN_OR_RETURN(auto lhs, GetValue(div->operand(0)));
    XLS_ASSIGN_OR_RETURN(auto rhs, GetValue(div->operand(1)));
    return SetValue(div, evaluator_.UDiv(lhs, rhs));
  };
  absl::Status HandleUGe(CompareOp* ge) override {
    XLS_ASSIGN_OR_RETURN(auto lhs, GetValue(ge->operand(0)));
    XLS_ASSIGN_OR_RETURN(auto rhs, GetValue(ge->operand(1)));
    using VectorT = AbstractEvaluatorT::Vector;
    return SetValue(ge,
                    VectorT({evaluator_.Not(evaluator_.ULessThan(lhs, rhs))}));
  }
  absl::Status HandleUGt(CompareOp* gt) override {
    XLS_ASSIGN_OR_RETURN(auto lhs, GetValue(gt->operand(0)));
    XLS_ASSIGN_OR_RETURN(auto rhs, GetValue(gt->operand(1)));
    using VectorT = AbstractEvaluatorT::Vector;
    // NB left and right swapped.
    return SetValue(gt, VectorT({evaluator_.ULessThan(rhs, lhs)}));
  }
  absl::Status HandleULe(CompareOp* le) override {
    XLS_ASSIGN_OR_RETURN(auto lhs, GetValue(le->operand(0)));
    XLS_ASSIGN_OR_RETURN(auto rhs, GetValue(le->operand(1)));
    using VectorT = AbstractEvaluatorT::Vector;
    // NB left and right swapped.
    return SetValue(le,
                    VectorT({evaluator_.Not(evaluator_.ULessThan(rhs, lhs))}));
  }
  absl::Status HandleULt(CompareOp* lt) override {
    XLS_ASSIGN_OR_RETURN(auto lhs, GetValue(lt->operand(0)));
    XLS_ASSIGN_OR_RETURN(auto rhs, GetValue(lt->operand(1)));
    using VectorT = AbstractEvaluatorT::Vector;
    return SetValue(lt, VectorT({evaluator_.ULessThan(lhs, rhs)}));
  }
  absl::Status HandleUMod(BinOp* mod) override {
    XLS_ASSIGN_OR_RETURN(auto lhs, GetValue(mod->operand(0)));
    XLS_ASSIGN_OR_RETURN(auto rhs, GetValue(mod->operand(1)));
    return SetValue(mod, evaluator_.UMod(lhs, rhs));
  }
  absl::Status HandleUMul(ArithOp* mul) override {
    XLS_ASSIGN_OR_RETURN(auto lhs, GetValue(mul->operand(0)));
    XLS_ASSIGN_OR_RETURN(auto rhs, GetValue(mul->operand(1)));
    auto result = evaluator_.UMul(lhs, rhs);
    int64_t expected_width = mul->BitCountOrDie();
    if (result.size() > expected_width) {
      result = evaluator_.BitSlice(result, 0, expected_width);
    } else if (result.size() < expected_width) {
      result = evaluator_.ZeroExtend(result, expected_width);
    }
    return SetValue(mul, std::move(result));
  }
  absl::Status HandleXorReduce(BitwiseReductionOp* xor_reduce) override {
    XLS_ASSIGN_OR_RETURN(auto args, GetValue(xor_reduce->operand(0)));
    return SetValue(xor_reduce, evaluator_.XorReduce(args));
  }
  absl::Status HandleZeroExtend(ExtendOp* zero_ext) override {
    XLS_ASSIGN_OR_RETURN(auto v, GetValue(zero_ext->operand(0)));
    return SetValue(zero_ext,
                    evaluator_.ZeroExtend(v, zero_ext->new_bit_count()));
  }

  // Get a view of the bits value associated with node n.
  //
  // Note that any call of SetCompoundValue or SetValue invalidates the view
  // returned by this function.
  absl::StatusOr<typename AbstractEvaluatorT::Span> GetValue(Node* n) const {
    XLS_RET_CHECK(n->GetType()->IsBits());
    XLS_ASSIGN_OR_RETURN(LeafTypeTreeView<LeafValueT> view,
                         GetCompoundValue(n));
    return view.Get({});
  }

  // Get an owned copy of the value of n.
  absl::StatusOr<typename AbstractEvaluatorT::Vector> GetOwnedValue(
      Node* n) const {
    XLS_ASSIGN_OR_RETURN(auto span, GetValue(n));
    return typename AbstractEvaluatorT::Vector(span.begin(), span.end());
  }

  // Gets the compound-value of the node n.
  //
  // Note that any call of SetCompoundValue or SetValue invalidates the view
  // returned by this function.
  absl::StatusOr<LeafTypeTreeView<LeafValueT>> GetCompoundValue(Node* n) const {
    XLS_RET_CHECK(values_.contains(n)) << n;
    return values_.at(n).AsView();
  }

  const absl::flat_hash_map<Node*, SharedLeafTypeTree<LeafValueT>>& values()
      const& {
    return values_;
  }

  absl::flat_hash_map<Node*, SharedLeafTypeTree<LeafValueT>>&& values() && {
    return std::move(values_);
  }

 protected:
  AbstractEvaluatorT& evaluator() { return evaluator_; }

  // Get the values of all bit-type nodes.
  absl::StatusOr<std::vector<typename AbstractEvaluatorT::Span>> GetValueList(
      absl::Span<Node* const> nodes) {
    std::vector<typename AbstractEvaluatorT::Span> args;
    args.reserve(nodes.size());
    for (Node* op : nodes) {
      XLS_ASSIGN_OR_RETURN(auto op_vec, GetValue(op));
      args.emplace_back(std::move(op_vec));
    }
    return args;
  }

  // Get the LTTs of all nodes.
  absl::StatusOr<std::vector<LeafTypeTreeView<LeafValueT>>>
  GetCompoundValueList(absl::Span<Node* const> nodes) {
    std::vector<LeafTypeTreeView<LeafValueT>> args;
    args.reserve(nodes.size());
    for (Node* op : nodes) {
      XLS_ASSIGN_OR_RETURN(auto op_vec, GetCompoundValue(op));
      args.emplace_back(std::move(op_vec));
    }
    return args;
  }

  // Set 'n' to the given bits value.
  //
  // This is intentionally a rvalue to avoid copying large bit vectors.
  //
  // This invalidates all Views returned by GetValue and GetCompoundValue.
  absl::Status SetValue(Node* n, typename AbstractEvaluatorT::Vector&& v) {
    XLS_RET_CHECK(n->GetType()->IsBits()) << n;
    return SetValue(
        n, LeafTypeTree<LeafValueT>::CreateFromVector(
               n->GetType(), typename LeafTypeTree<LeafValueT>::DataContainerT{
                                 std::move(v)}));
  }

  // Set 'n' to the given value.
  //
  // This is intentionally a rvalue to avoid copying large bit vectors.
  //
  // This invalidates all Views returned by GetValue and GetCompoundValue.
  absl::Status SetValue(Node* n, SharedLeafTypeTree<LeafValueT>&& value) {
    XLS_RET_CHECK(!values_.contains(n)) << n << " visited multiple times";
    values_.emplace(n, std::move(value));
    return absl::OkStatus();
  }
  absl::Status SetValue(Node* n, LeafTypeTree<LeafValueT>&& value) {
    XLS_RET_CHECK(!values_.contains(n)) << n << " visited multiple times";
    values_.emplace(n, std::move(value).AsShared());
    return absl::OkStatus();
  }

  // Call 'f' on all the corresponding components of 'cases'. Each element must
  // have type 'target'
  absl::StatusOr<LeafTypeTree<LeafValueT>> ZipPiecewise(
      Type* target, absl::Span<Node* const> cases,
      std::function<typename AbstractEvaluatorT::Vector(
          absl::Span<const typename AbstractEvaluatorT::Span>)>
          f) {
    XLS_ASSIGN_OR_RETURN(std::vector<LeafTypeTreeView<LeafValueT>> options,
                         GetCompoundValueList(cases));
    return ZipPiecewise(target, options, f);
  }
  absl::StatusOr<LeafTypeTree<LeafValueT>> ZipPiecewise(
      Type* target, absl::Span<const LeafTypeTree<LeafValueT>> cases,
      std::function<typename AbstractEvaluatorT::Vector(
          absl::Span<const typename AbstractEvaluatorT::Span>)>
          f) {
    std::vector<LeafTypeTreeView<LeafValueT>> options;
    options.reserve(cases.size());
    for (const LeafTypeTree<LeafValueT>& option : cases) {
      options.push_back(option.AsView());
    }
    return ZipPiecewise(target, options, f);
  }
  absl::StatusOr<LeafTypeTree<LeafValueT>> ZipPiecewise(
      Type* target, absl::Span<const LeafTypeTreeView<LeafValueT>> options,
      std::function<typename AbstractEvaluatorT::Vector(
          absl::Span<const typename AbstractEvaluatorT::Span>)>
          f) {
    std::vector<typename AbstractEvaluatorT::Span> spans;
    spans.reserve(options.size());
    return leaf_type_tree::ZipIndex<typename AbstractEvaluatorT::Vector,
                                    typename AbstractEvaluatorT::Vector>(
        options,
        [&](Type* et,
            absl::Span<const typename AbstractEvaluatorT::Vector* const>
                elements,
            absl::Span<const int64_t> index)
            -> absl::StatusOr<typename AbstractEvaluatorT::Vector> {
          spans.clear();
          spans.reserve(elements.size());
          absl::c_transform(elements, std::back_inserter(spans),
                            [](const typename AbstractEvaluatorT::Vector* v) ->
                            typename AbstractEvaluatorT::Span { return *v; });
          return f(spans);
        });
  }

 private:
  AbstractEvaluatorT& evaluator_;
  // Values of the components of components of compound values. nullptr values
  // represent values which are considered unconstrained. This uses unique_ptr
  // to ensure that the internal pointers do not move since we may want to hold
  // views to them.
  absl::flat_hash_map<Node*, SharedLeafTypeTree<LeafValueT>> values_;
};

// An abstract evaluator for XLS Nodes. The function takes an AbstractEvaluator
// and calls the appropriate method (e.g., AbstractEvaluator::BitSlice)
// depending upon the Op of the Node (e.g., Op::kBitSlice) using the given
// operand values. For unsupported operations the given function
// 'default_handler' is called to generate the return value.
template <typename AbstractEvaluatorT>
absl::StatusOr<typename AbstractEvaluatorT::Vector> AbstractEvaluate(
    Node* node, absl::Span<const typename AbstractEvaluatorT::Vector> operands,
    AbstractEvaluatorT* evaluator,
    std::function<typename AbstractEvaluatorT::Vector(Node*)> default_handler) {
  VLOG(3) << "Handling " << node->ToString();
  class CompatVisitor final : public AbstractNodeEvaluator<AbstractEvaluatorT> {
   public:
    CompatVisitor(AbstractEvaluatorT& eval,
                  std::function<typename AbstractEvaluatorT::Vector(Node*)>&
                      default_handler)
        : xls::AbstractNodeEvaluator<AbstractEvaluatorT>(eval),
          default_handler_(default_handler) {}
    absl::Status ForceSetValue(Node* n, typename AbstractEvaluatorT::Vector v) {
      if (this->values().contains(n)) {
        XLS_ASSIGN_OR_RETURN(auto existing, this->GetValue(n));
        XLS_RET_CHECK(absl::c_equal(existing, v))
            << "Node with identical operands has different values: " << n;
        return absl::OkStatus();
      }
      return SetValue(n, std::move(v));
    }
    absl::Status DefaultHandler(Node* node) override {
      return SetValue(node, default_handler_(node));
    }

   protected:
    using AbstractNodeEvaluator<AbstractEvaluatorT>::SetValue;

   private:
    std::function<typename AbstractEvaluatorT::Vector(Node*)>& default_handler_;
  };
  CompatVisitor v(*evaluator, default_handler);
  XLS_RET_CHECK_EQ(operands.size(), node->operand_count())
      << node << " has different operand count";
  for (int64_t i = 0; i < operands.size(); ++i) {
    XLS_RETURN_IF_ERROR(v.ForceSetValue(node->operand(i), operands[i]))
        << node << "@op" << i;
  }
  XLS_RETURN_IF_ERROR(node->VisitSingleNode(&v));
  return v.GetOwnedValue(node);
}

}  // namespace xls

#endif  // XLS_IR_ABSTRACT_NODE_EVALUATOR_H_
