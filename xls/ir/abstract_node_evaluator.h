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

#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/lsb_or_msb.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"

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
// Once visited values can be obtained by calling GetValue.
template <typename AbstractEvaluatorT>
class AbstractNodeEvaluator : public DfsVisitorWithDefault {
 public:
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
  absl::Status HandleEncode(Encode* encode) override {
    XLS_ASSIGN_OR_RETURN(auto input, GetValue(encode->operand(0)));
    return SetValue(encode, evaluator_.Encode(input));
  }
  absl::Status HandleEq(CompareOp* eq) override {
    XLS_ASSIGN_OR_RETURN(auto lhs, GetValue(eq->operand(0)));
    XLS_ASSIGN_OR_RETURN(auto rhs, GetValue(eq->operand(1)));
    using VectorT = AbstractEvaluatorT::Vector;
    return SetValue(eq, VectorT({evaluator_.Equals(lhs, rhs)}));
  }
  absl::Status HandleGate(Gate* gate) override {
    XLS_ASSIGN_OR_RETURN(auto cond, GetValue(gate->condition()));
    XLS_ASSIGN_OR_RETURN(auto data, GetValue(gate->data()));
    XLS_RET_CHECK_EQ(cond.size(), 1);
    return SetValue(gate, evaluator_.Gate(cond[0], data));
  }
  absl::Status HandleIdentity(UnOp* identity) override {
    XLS_ASSIGN_OR_RETURN(auto v, GetValue(identity->operand(0)));
    return SetValue(identity, v);
  }
  absl::Status HandleLiteral(Literal* literal) override {
    if (literal->value().IsBits()) {
      return SetValue(literal,
                      evaluator_.BitsToVector(literal->value().bits()));
    }
    return static_cast<AbstractNodeEvaluator<AbstractEvaluatorT>*>(this)
        ->DefaultHandler(literal);
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
    XLS_ASSIGN_OR_RETURN(auto lhs, GetValue(ne->operand(0)));
    XLS_ASSIGN_OR_RETURN(auto rhs, GetValue(ne->operand(1)));
    using VectorT = AbstractEvaluatorT::Vector;
    return SetValue(ne, VectorT({evaluator_.Not(evaluator_.Equals(lhs, rhs))}));
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
    XLS_ASSIGN_OR_RETURN(auto args, GetValueList(sel->cases()));
    XLS_ASSIGN_OR_RETURN(auto selector, GetValue(sel->selector()));
    return SetValue(
        sel, evaluator_.OneHotSelect(
                 selector, args,
                 /*selector_can_be_zero=*/!sel->selector()->Is<OneHot>()));
  };
  absl::Status HandlePrioritySel(PrioritySelect* sel) override {
    XLS_ASSIGN_OR_RETURN(auto args, GetValueList(sel->cases()));
    XLS_ASSIGN_OR_RETURN(auto selector, GetValue(sel->selector()));
    return SetValue(
        sel, evaluator_.PrioritySelect(
                 selector, args,
                 /*selector_can_be_zero=*/!sel->selector()->Is<OneHot>()));
  }
  absl::Status HandleOrReduce(BitwiseReductionOp* or_reduce) override {
    XLS_ASSIGN_OR_RETURN(auto args, GetValue(or_reduce->operand(0)));
    return SetValue(or_reduce, evaluator_.OrReduce(args));
  }
  absl::Status HandleReverse(UnOp* reverse) override {
    XLS_ASSIGN_OR_RETURN(auto v, GetValue(reverse->operand(0)));
    absl::c_reverse(v);
    return SetValue(reverse, v);
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
    return SetValue(mul, result);
  };
  absl::Status HandleSel(Select* sel) override {
    XLS_ASSIGN_OR_RETURN(auto args, GetValueList(sel->cases()));
    XLS_ASSIGN_OR_RETURN(auto selector, GetValue(sel->selector()));
    std::optional<typename AbstractEvaluatorT::Vector> default_value;
    if (sel->default_value()) {
      XLS_ASSIGN_OR_RETURN(default_value, GetValue(*sel->default_value()));
    }
    return SetValue(sel, evaluator_.Select(selector, args, default_value));
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
    return SetValue(sub, evaluator_.Add(lhs, evaluator_.Neg(rhs)));
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
    return SetValue(mul, result);
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

  const absl::flat_hash_map<Node*, typename AbstractEvaluatorT::Vector>&
  values() const {
    return values_;
  }

  absl::StatusOr<typename AbstractEvaluatorT::Vector> GetValue(Node* n) const {
    XLS_RET_CHECK(values_.contains(n)) << n;
    return values_.at(n);
  }

 protected:
  absl::StatusOr<std::vector<typename AbstractEvaluatorT::Vector>> GetValueList(
      absl::Span<Node* const> nodes) {
    std::vector<typename AbstractEvaluatorT::Vector> args;
    args.reserve(nodes.size());
    for (Node* op : nodes) {
      XLS_ASSIGN_OR_RETURN(auto op_vec, GetValue(op));
      args.emplace_back(std::move(op_vec));
    }
    return args;
  }

  absl::Status SetValue(Node* n, typename AbstractEvaluatorT::Vector v) {
    XLS_RET_CHECK(!values_.contains(n)) << n;
    values_[n] = std::move(v);
    return absl::OkStatus();
  }

 private:
  AbstractEvaluatorT& evaluator_;
  absl::flat_hash_map<Node*, typename AbstractEvaluatorT::Vector> values_;
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
  XLS_VLOG(3) << "Handling " << node->ToString();
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
      return SetValue(n, v);
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
  return v.GetValue(node);
}

}  // namespace xls

#endif  // XLS_IR_ABSTRACT_NODE_EVALUATOR_H_
