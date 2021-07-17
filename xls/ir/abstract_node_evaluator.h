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

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/optional.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"

namespace xls {

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
  using Vector = typename AbstractEvaluatorT::Vector;

  XLS_VLOG(3) << "Handling " << node->ToString();
  auto check_operand_count = [&](int64_t expected) {
    if (operands.size() != expected) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Expected %d operands, "
                          "got %d for node: %s",
                          expected, operands.size(), node->ToString()));
    }
    return absl::OkStatus();
  };

  // Important! Array and Tuples are "pass-through" - all ops will be passed to
  // the default_handler.
  // TODO(rspringer): Actually, the implementations in smtlib_emitter_main.cc
  // seem general enough. Make them "official"?

  switch (node->op()) {
    case Op::kAdd:
      XLS_RETURN_IF_ERROR(check_operand_count(2));
      return evaluator->Add(operands[0], operands[1]);
    case Op::kAnd:
      return evaluator->BitwiseAnd(operands);
    case Op::kAndReduce:
      return evaluator->AndReduce(operands[0]);
    case Op::kAfterAll:
      return default_handler(node);
    case Op::kArray:
      return default_handler(node);
    case Op::kArrayIndex:
      return default_handler(node);
    case Op::kArrayUpdate:
      return default_handler(node);
    case Op::kArrayConcat:
      return default_handler(node);
    case Op::kArraySlice:
      return default_handler(node);
    case Op::kAssert:
      return default_handler(node);
    case Op::kCover:
      return default_handler(node);
    case Op::kBitSlice: {
      XLS_RETURN_IF_ERROR(check_operand_count(1));
      BitSlice* bit_slice = node->As<BitSlice>();
      return evaluator->BitSlice(operands[0], bit_slice->start(),
                                 bit_slice->width());
    }
    case Op::kDynamicBitSlice:
      return default_handler(node);
    case Op::kBitSliceUpdate:
      XLS_RETURN_IF_ERROR(check_operand_count(3));
      return evaluator->BitSliceUpdate(operands[0], operands[1], operands[2]);
    case Op::kReceive:
      return default_handler(node);
    case Op::kSend:
      return default_handler(node);
    case Op::kConcat:
      return evaluator->Concat(operands);
    case Op::kCountedFor:
      return default_handler(node);
    case Op::kDecode: {
      XLS_RETURN_IF_ERROR(check_operand_count(1));
      Decode* decode = node->As<Decode>();
      return evaluator->Decode(operands[0], decode->width());
    }
    case Op::kDynamicCountedFor:
      return default_handler(node);
    case Op::kEncode:
      XLS_RETURN_IF_ERROR(check_operand_count(1));
      return evaluator->Encode(operands[0]);
    case Op::kEq:
      XLS_RETURN_IF_ERROR(check_operand_count(2));
      return Vector({evaluator->Equals(operands[0], operands[1])});
    case Op::kGate:
      return default_handler(node);
    case Op::kIdentity:
      XLS_RETURN_IF_ERROR(check_operand_count(1));
      return operands[0];
    case Op::kInvoke:
      return default_handler(node);
    case Op::kInputPort:
      return default_handler(node);
    case Op::kLiteral: {
      XLS_RETURN_IF_ERROR(check_operand_count(0));
      Literal* literal = node->As<Literal>();
      if (literal->value().IsBits()) {
        return evaluator->BitsToVector(literal->value().bits());
      }
      return default_handler(node);
    }
    case Op::kMap:
      return default_handler(node);
    case Op::kNand:
      return evaluator->BitwiseNot(evaluator->BitwiseAnd(operands));
    case Op::kNe:
      XLS_RETURN_IF_ERROR(check_operand_count(2));
      return Vector(
          {evaluator->Not(evaluator->Equals(operands[0], operands[1]))});
    case Op::kNeg:
      XLS_RETURN_IF_ERROR(check_operand_count(1));
      return evaluator->Neg(operands[0]);
    case Op::kNor:
      return evaluator->BitwiseNot(evaluator->BitwiseOr(operands));
    case Op::kNot:
      XLS_RETURN_IF_ERROR(check_operand_count(1));
      return evaluator->BitwiseNot(operands[0]);
    case Op::kOneHot: {
      XLS_RETURN_IF_ERROR(check_operand_count(1));
      OneHot* one_hot = node->As<OneHot>();
      return one_hot->priority() == LsbOrMsb::kLsb
                 ? evaluator->OneHotLsbToMsb(operands[0])
                 : evaluator->OneHotMsbToLsb(operands[0]);
    }
    case Op::kOneHotSel: {
      OneHotSelect* sel = node->As<OneHotSelect>();
      return evaluator->OneHotSelect(
          operands[0], operands.subspan(1), /*selector_can_be_zero=*/
          !sel->selector()->Is<OneHot>());
    }
    case Op::kOr:
      return evaluator->BitwiseOr(operands);
    case Op::kOrReduce:
      return evaluator->OrReduce(operands[0]);
    case Op::kOutputPort:
      return default_handler(node);
    case Op::kParam:
      return default_handler(node);
    case Op::kRegisterRead:
      return default_handler(node);
    case Op::kRegisterWrite:
      return default_handler(node);
    case Op::kReverse: {
      XLS_RETURN_IF_ERROR(check_operand_count(1));
      Vector result(operands[0].begin(), operands[0].end());
      std::reverse(result.begin(), result.end());
      return result;
    }
    case Op::kSDiv:
      XLS_RETURN_IF_ERROR(check_operand_count(2));
      return evaluator->SDiv(operands[0], operands[1]);
    case Op::kSMod:
      XLS_RETURN_IF_ERROR(check_operand_count(2));
      return evaluator->SMod(operands[0], operands[1]);
    case Op::kSGe:
      XLS_RETURN_IF_ERROR(check_operand_count(2));
      return Vector(
          {evaluator->Not(evaluator->SLessThan(operands[0], operands[1]))});
    case Op::kSGt:
      XLS_RETURN_IF_ERROR(check_operand_count(2));
      return Vector({evaluator->SLessThan(operands[1], operands[0])});
    case Op::kSLe:
      XLS_RETURN_IF_ERROR(check_operand_count(2));
      return Vector(
          {evaluator->Not(evaluator->SLessThan(operands[1], operands[0]))});
    case Op::kSLt:
      XLS_RETURN_IF_ERROR(check_operand_count(2));
      return Vector({evaluator->SLessThan(operands[0], operands[1])});
    case Op::kSel: {
      Select* sel = node->As<Select>();
      absl::Span<const Vector> cases =
          sel->default_value().has_value()
              ? operands.subspan(1, operands.size() - 2)
              : operands.subspan(1);
      absl::optional<const Vector> default_value =
          sel->default_value().has_value()
              ? absl::optional<const Vector>(operands.back())
              : absl::nullopt;
      return evaluator->Select(operands[0], cases, default_value);
    }
    case Op::kShll:
      XLS_RETURN_IF_ERROR(check_operand_count(2));
      return evaluator->ShiftLeftLogical(operands[0], operands[1]);
    case Op::kShra:
      XLS_RETURN_IF_ERROR(check_operand_count(2));
      return evaluator->ShiftRightArith(operands[0], operands[1]);
    case Op::kShrl:
      XLS_RETURN_IF_ERROR(check_operand_count(2));
      return evaluator->ShiftRightLogical(operands[0], operands[1]);
    case Op::kSignExt:
      XLS_RETURN_IF_ERROR(check_operand_count(1));
      return evaluator->SignExtend(operands[0], node->BitCountOrDie());
    case Op::kSMul: {
      XLS_RETURN_IF_ERROR(check_operand_count(2));
      Vector result = evaluator->SMul(operands[0], operands[1]);
      int64_t expected_width = node->BitCountOrDie();
      if (result.size() > expected_width) {
        result = evaluator->BitSlice(result, 0, expected_width);
      } else if (result.size() < expected_width) {
        result = evaluator->SignExtend(result, expected_width);
      }
      return result;
    }
    case Op::kSub:
      XLS_RETURN_IF_ERROR(check_operand_count(2));
      return evaluator->Add(operands[0], evaluator->Neg(operands[1]));
    case Op::kTuple:
      return default_handler(node);
    case Op::kTupleIndex:
      return default_handler(node);
    case Op::kUDiv:
      XLS_RETURN_IF_ERROR(check_operand_count(2));
      return evaluator->UDiv(operands[0], operands[1]);
    case Op::kUMod:
      XLS_RETURN_IF_ERROR(check_operand_count(2));
      return evaluator->UMod(operands[0], operands[1]);
    case Op::kUGe:
      XLS_RETURN_IF_ERROR(check_operand_count(2));
      return Vector(
          {evaluator->Not(evaluator->ULessThan(operands[0], operands[1]))});
    case Op::kUGt:
      XLS_RETURN_IF_ERROR(check_operand_count(2));
      return Vector({evaluator->ULessThan(operands[1], operands[0])});
    case Op::kULe:
      XLS_RETURN_IF_ERROR(check_operand_count(2));
      return Vector(
          {evaluator->Not(evaluator->ULessThan(operands[1], operands[0]))});
    case Op::kULt:
      XLS_RETURN_IF_ERROR(check_operand_count(2));
      return Vector({evaluator->ULessThan(operands[0], operands[1])});
    case Op::kUMul: {
      XLS_RETURN_IF_ERROR(check_operand_count(2));
      Vector result = evaluator->UMul(operands[0], operands[1]);
      int64_t expected_width = node->BitCountOrDie();
      if (result.size() > expected_width) {
        result = evaluator->BitSlice(result, 0, expected_width);
      } else if (result.size() < expected_width) {
        result = evaluator->ZeroExtend(result, expected_width);
      }
      return result;
    }
    case Op::kXor:
      return evaluator->BitwiseXor(operands);
    case Op::kXorReduce:
      return evaluator->XorReduce(operands[0]);
    case Op::kZeroExt:
      XLS_RETURN_IF_ERROR(check_operand_count(1));
      return evaluator->ZeroExtend(operands[0], node->BitCountOrDie());
  }
  XLS_LOG(FATAL) << "Invalid op: " << static_cast<int64_t>(node->op());
}

}  // namespace xls

#endif  // XLS_IR_ABSTRACT_NODE_EVALUATOR_H_
