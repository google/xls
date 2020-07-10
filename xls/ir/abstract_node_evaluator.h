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

#ifndef XLS_IR_ABSTRACT_NODE_EVALUATOR_H_
#define XLS_IR_ABSTRACT_NODE_EVALUATOR_H_

#include "absl/status/status.h"
#include "absl/types/optional.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/status/statusor.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"

namespace xls {

// An abstract evaluator for XLS Nodes. The function takes an AbstractEvaluator
// and calls the appropriate method (e.g., AbstractEvaluator::BitSlice)
// depending upon the Op of the Node (e.g., OP_BIT_SLICE) using the given
// operand values. For unsupported operations the given function
// 'default_handler' is called to generate the return value.
template <typename AbstractEvaluatorT>
xabsl::StatusOr<typename AbstractEvaluatorT::Vector> AbstractEvaluate(
    Node* node, absl::Span<const typename AbstractEvaluatorT::Vector> operands,
    AbstractEvaluatorT* evaluator,
    std::function<typename AbstractEvaluatorT::Vector(Node*)> default_handler) {
  using Vector = typename AbstractEvaluatorT::Vector;

  auto check_operand_count = [&](int64 expected) {
    if (operands.size() != expected) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Expected %d operands, "
                          "got %d for node: %s",
                          expected, operands.size(), node->ToString()));
    }
    return absl::OkStatus();
  };

  // TODO(meheff): Extend this to non-Bits-typed values.
  if (!node->GetType()->IsBits() ||
      !std::all_of(node->operands().begin(), node->operands().end(),
                   [](Node* o) { return o->GetType()->IsBits(); })) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Only Bits types supported for evaluation, node: %s",
                        node->ToString()));
  }

  switch (node->op()) {
    case OP_ADD:
      return default_handler(node);
    case OP_AND:
      return evaluator->BitwiseAnd(operands);
    case OP_AND_REDUCE:
      return evaluator->AndReduce(operands[0]);
    case OP_ARRAY:
      return default_handler(node);
    case OP_ARRAY_INDEX:
      return default_handler(node);
    case OP_ARRAY_UPDATE:
      return default_handler(node);
    case OP_BIT_SLICE: {
      XLS_RETURN_IF_ERROR(check_operand_count(1));
      BitSlice* bit_slice = node->As<BitSlice>();
      return evaluator->BitSlice(operands[0], bit_slice->start(),
                                 bit_slice->width());
    }
    case OP_DYNAMIC_BIT_SLICE: {
      return default_handler(node);
    }
    case OP_CONCAT:
      return evaluator->Concat(operands);
    case OP_COUNTED_FOR:
      return default_handler(node);
    case OP_DECODE: {
      XLS_RETURN_IF_ERROR(check_operand_count(1));
      Decode* decode = node->As<Decode>();
      return evaluator->Decode(operands[0], decode->width());
    }
    case OP_ENCODE:
      XLS_RETURN_IF_ERROR(check_operand_count(1));
      return evaluator->Encode(operands[0]);
    case OP_EQ:
      XLS_RETURN_IF_ERROR(check_operand_count(2));
      return Vector({evaluator->Equals(operands[0], operands[1])});
    case OP_IDENTITY:
      XLS_RETURN_IF_ERROR(check_operand_count(1));
      return operands[0];
    case OP_INVOKE:
      return default_handler(node);
    case OP_LITERAL: {
      XLS_RETURN_IF_ERROR(check_operand_count(0));
      Literal* literal = node->As<Literal>();
      return evaluator->BitsToVector(literal->value().bits());
    }
    case OP_MAP:
      return default_handler(node);
    case OP_NAND:
      return evaluator->BitwiseNot(evaluator->BitwiseAnd(operands));
    case OP_NE:
      XLS_RETURN_IF_ERROR(check_operand_count(2));
      return Vector(
          {evaluator->Not(evaluator->Equals(operands[0], operands[1]))});
    case OP_NEG:
      return default_handler(node);
    case OP_NOR:
      return evaluator->BitwiseNot(evaluator->BitwiseOr(operands));
    case OP_NOT:
      XLS_RETURN_IF_ERROR(check_operand_count(1));
      return evaluator->BitwiseNot(operands[0]);
    case OP_ONE_HOT: {
      XLS_RETURN_IF_ERROR(check_operand_count(1));
      OneHot* one_hot = node->As<OneHot>();
      return one_hot->priority() == LsbOrMsb::kLsb
                 ? evaluator->OneHotLsbToMsb(operands[0])
                 : evaluator->OneHotMsbToLsb(operands[0]);
    }
    case OP_ONE_HOT_SEL: {
      OneHotSelect* sel = node->As<OneHotSelect>();
      return evaluator->OneHotSelect(
          operands[0], operands.subspan(1), /*selector_can_be_zero=*/
          !sel->selector()->Is<OneHot>());
    }
    case OP_OR:
      return evaluator->BitwiseOr(operands);
    case OP_OR_REDUCE:
      return evaluator->OrReduce(operands[0]);
    case OP_PARAM:
      return default_handler(node);
    case OP_REVERSE: {
      XLS_RETURN_IF_ERROR(check_operand_count(1));
      Vector result(operands[0].begin(), operands[0].end());
      std::reverse(result.begin(), result.end());
      return result;
    }
    case OP_SDIV:
      return default_handler(node);
    case OP_SGE:
      return default_handler(node);
    case OP_SGT:
      return default_handler(node);
    case OP_SLE:
      return default_handler(node);
    case OP_SLT:
      return default_handler(node);
    case OP_SEL: {
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
    case OP_SHLL:
      XLS_RETURN_IF_ERROR(check_operand_count(2));
      return evaluator->ShiftLeftLogical(operands[0], operands[1]);
    case OP_SHRA:
      XLS_RETURN_IF_ERROR(check_operand_count(2));
      return evaluator->ShiftRightArith(operands[0], operands[1]);
    case OP_SHRL:
      XLS_RETURN_IF_ERROR(check_operand_count(2));
      return evaluator->ShiftRightLogical(operands[0], operands[1]);
    case OP_SIGN_EXT:
      XLS_RETURN_IF_ERROR(check_operand_count(1));
      return evaluator->SignExtend(operands[0], node->BitCountOrDie());
    case OP_SMUL:
      return default_handler(node);
    case OP_SUB:
      return default_handler(node);
    case OP_TUPLE:
      return default_handler(node);
    case OP_TUPLE_INDEX:
      return default_handler(node);
    case OP_UDIV:
      return default_handler(node);
    case OP_UGE:
      XLS_RETURN_IF_ERROR(check_operand_count(2));
      return Vector(
          {evaluator->Not(evaluator->ULessThan(operands[0], operands[1]))});
    case OP_UGT:
      XLS_RETURN_IF_ERROR(check_operand_count(2));
      return Vector({evaluator->ULessThan(operands[1], operands[0])});
    case OP_ULE:
      XLS_RETURN_IF_ERROR(check_operand_count(2));
      return Vector(
          {evaluator->Not(evaluator->ULessThan(operands[1], operands[0]))});
    case OP_ULT:
      XLS_RETURN_IF_ERROR(check_operand_count(2));
      return Vector({evaluator->ULessThan(operands[0], operands[1])});
    case OP_UMUL:
      return default_handler(node);
    case OP_XOR:
      return evaluator->BitwiseXor(operands);
    case OP_XOR_REDUCE:
      return evaluator->XorReduce(operands[0]);
    case OP_ZERO_EXT:
      XLS_RETURN_IF_ERROR(check_operand_count(1));
      return evaluator->ZeroExtend(operands[0], node->BitCountOrDie());
  }
}

}  // namespace xls

#endif  // XLS_IR_ABSTRACT_NODE_EVALUATOR_H_
