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

#include "xls/ir/verifier.h"

#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/logging/logging.h"
#include "xls/common/math_util.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"
#include "xls/ir/channel.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/function.h"
#include "xls/ir/node.h"
#include "xls/ir/node_iterator.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/type.h"

namespace xls {
namespace {

using ::absl::StrFormat;

// Visitor which verifies various properties of Nodes including the types of the
// operands and the type of the result.
class NodeChecker : public DfsVisitor {
 public:
  absl::Status HandleAdd(BinOp* add) override {
    XLS_RETURN_IF_ERROR(ExpectOperandCount(add, 2));
    return ExpectAllSameBitsType(add);
  }

  absl::Status HandleAndReduce(BitwiseReductionOp* and_reduce) override {
    XLS_RETURN_IF_ERROR(ExpectOperandCount(and_reduce, 1));
    XLS_RETURN_IF_ERROR(ExpectOperandHasBitsType(and_reduce, 0));
    return ExpectHasBitsType(and_reduce, 1);
  }

  absl::Status HandleOrReduce(BitwiseReductionOp* or_reduce) override {
    XLS_RETURN_IF_ERROR(ExpectOperandCount(or_reduce, 1));
    XLS_RETURN_IF_ERROR(ExpectOperandHasBitsType(or_reduce, 0));
    return ExpectHasBitsType(or_reduce, 1);
  }

  absl::Status HandleXorReduce(BitwiseReductionOp* xor_reduce) override {
    XLS_RETURN_IF_ERROR(ExpectOperandCount(xor_reduce, 1));
    XLS_RETURN_IF_ERROR(ExpectOperandHasBitsType(xor_reduce, 0));
    return ExpectHasBitsType(xor_reduce, 1);
  }

  absl::Status HandleAssert(Assert* assert_op) override {
    XLS_RETURN_IF_ERROR(ExpectOperandCount(assert_op, 2));
    XLS_RETURN_IF_ERROR(ExpectOperandHasTokenType(assert_op, 0));
    XLS_RETURN_IF_ERROR(ExpectOperandHasBitsType(assert_op, /*operand_no=*/1,
                                                 /*expected_bit_count=*/1));
    return ExpectHasTokenType(assert_op);
  }

  absl::Status HandleCover(Cover* cover) override {
    XLS_RETURN_IF_ERROR(ExpectOperandCount(cover, 2));
    XLS_RETURN_IF_ERROR(ExpectOperandHasTokenType(cover, /*operand_no=*/0));
    XLS_RETURN_IF_ERROR(ExpectOperandHasBitsType(cover, /*operand_no=*/1,
                                                 /*expected_bit_count=*/1));
    return ExpectHasTokenType(cover);
  }

  absl::Status HandleNaryAnd(NaryOp* and_op) override {
    XLS_RETURN_IF_ERROR(ExpectOperandCountGt(and_op, 0));
    return ExpectAllSameBitsType(and_op);
  }

  absl::Status HandleNaryNand(NaryOp* nand_op) override {
    XLS_RETURN_IF_ERROR(ExpectOperandCountGt(nand_op, 0));
    return ExpectAllSameBitsType(nand_op);
  }

  absl::Status HandleNaryNor(NaryOp* nor_op) override {
    XLS_RETURN_IF_ERROR(ExpectOperandCountGt(nor_op, 0));
    return ExpectAllSameBitsType(nor_op);
  }

  absl::Status HandleAfterAll(AfterAll* after_all) override {
    XLS_RETURN_IF_ERROR(ExpectHasTokenType(after_all));
    for (int64_t i = 0; i < after_all->operand_count(); ++i) {
      XLS_RETURN_IF_ERROR(ExpectOperandHasTokenType(after_all, i));
    }
    return absl::OkStatus();
  }

  absl::Status HandleReceive(Receive* receive) override {
    XLS_RETURN_IF_ERROR(ExpectOperandCountRange(receive, 1, 2));
    XLS_RETURN_IF_ERROR(ExpectOperandHasTokenType(receive, /*operand_no=*/0));
    if (receive->predicate().has_value()) {
      XLS_RETURN_IF_ERROR(
          ExpectOperandHasBitsType(receive, 1, /*expected_bit_count=*/1));
    }
    if (!receive->package()->HasChannelWithId(receive->channel_id())) {
      return absl::InternalError(
          StrFormat("%s refers to channel ID %d which does not exist",
                    receive->GetName(), receive->channel_id()));
    }
    XLS_ASSIGN_OR_RETURN(Channel * channel,
                         receive->package()->GetChannel(receive->channel_id()));
    Type* expected_type = receive->package()->GetTupleType(
        {receive->package()->GetTokenType(), channel->type()});
    if (receive->GetType() != expected_type) {
      return absl::InternalError(StrFormat(
          "Expected %s to have type %s, has type %s", receive->GetName(),
          expected_type->ToString(), receive->GetType()->ToString()));
    }
    if (!channel->CanReceive()) {
      return absl::InternalError(StrFormat(
          "Cannot receive over channel %s (%d), receive operation: %s",
          channel->name(), channel->id(), receive->GetName()));
    }
    return absl::OkStatus();
  }

  absl::Status HandleSend(Send* send) override {
    XLS_RETURN_IF_ERROR(ExpectHasTokenType(send));
    XLS_RETURN_IF_ERROR(ExpectOperandCountRange(send, 2, 3));
    XLS_RETURN_IF_ERROR(ExpectOperandHasTokenType(send, /*operand_no=*/0));
    if (send->predicate().has_value()) {
      XLS_RETURN_IF_ERROR(
          ExpectOperandHasBitsType(send, 2, /*expected_bit_count=*/1));
    }

    if (!send->package()->HasChannelWithId(send->channel_id())) {
      return absl::InternalError(
          StrFormat("%s refers to channel ID %d which does not exist",
                    send->GetName(), send->channel_id()));
    }
    XLS_ASSIGN_OR_RETURN(Channel * channel,
                         send->package()->GetChannel(send->channel_id()));
    if (!channel->CanSend()) {
      return absl::InternalError(
          StrFormat("Cannot send over channel %s (%d), send operation: %s",
                    channel->name(), channel->id(), send->GetName()));
    }
    XLS_RETURN_IF_ERROR(ExpectOperandHasType(send, 1, channel->type()));
    return absl::OkStatus();
  }

  absl::Status HandleArray(Array* array) override {
    XLS_RETURN_IF_ERROR(ExpectHasArrayType(array));
    ArrayType* array_type = array->GetType()->AsArrayOrDie();
    XLS_RETURN_IF_ERROR(ExpectOperandCount(array, array_type->size()));
    Type* element_type = array_type->element_type();
    for (int64_t i = 0; i < array->operand_count(); ++i) {
      XLS_RETURN_IF_ERROR(ExpectOperandHasType(array, i, element_type));
    }
    return absl::OkStatus();
  }

  absl::Status HandleBitSlice(BitSlice* bit_slice) override {
    XLS_RETURN_IF_ERROR(ExpectHasBitsType(bit_slice, bit_slice->width()));
    XLS_RETURN_IF_ERROR(ExpectOperandCount(bit_slice, 1));
    XLS_RETURN_IF_ERROR(ExpectOperandHasBitsType(bit_slice, 0));
    BitsType* operand_type = bit_slice->operand(0)->GetType()->AsBitsOrDie();
    if (bit_slice->start() < 0) {
      return absl::InternalError(
          StrFormat("Start index of bit slice must be non-negative: %s",
                    bit_slice->ToString()));
    }
    if (bit_slice->width() < 0) {
      return absl::InternalError(
          StrFormat("Width of bit slice must be non-negative: %s",
                    bit_slice->ToString()));
    }
    const int64_t bits_required = bit_slice->start() + bit_slice->width();
    if (operand_type->bit_count() < bits_required) {
      return absl::InternalError(
          StrFormat("Expected operand 0 of %s to have at least %d bits (start "
                    "%d + width %d), has only %d: %s",
                    bit_slice->GetName(), bits_required, bit_slice->start(),
                    bit_slice->width(), operand_type->bit_count(),
                    bit_slice->ToString()));
    }
    return absl::OkStatus();
  }

  absl::Status HandleDynamicBitSlice(
      DynamicBitSlice* dynamic_bit_slice) override {
    XLS_RETURN_IF_ERROR(
        ExpectHasBitsType(dynamic_bit_slice, dynamic_bit_slice->width()));
    XLS_RETURN_IF_ERROR(ExpectOperandCount(dynamic_bit_slice, 2));
    XLS_RETURN_IF_ERROR(ExpectOperandHasBitsType(dynamic_bit_slice, 0));
    XLS_RETURN_IF_ERROR(ExpectOperandHasBitsType(dynamic_bit_slice, 1));
    BitsType* operand_type =
        dynamic_bit_slice->operand(0)->GetType()->AsBitsOrDie();
    if (dynamic_bit_slice->width() < 0) {
      return absl::InternalError(
          StrFormat("Width of bit slice must be non-negative: %s",
                    dynamic_bit_slice->ToString()));
    }
    if (operand_type->bit_count() < dynamic_bit_slice->width()) {
      return absl::InternalError(
          StrFormat("Expected operand 0 of %s to have at least %d bits (width"
                    " %d), has only %d: %s",
                    dynamic_bit_slice->GetName(), dynamic_bit_slice->width(),
                    dynamic_bit_slice->width(), operand_type->bit_count(),
                    dynamic_bit_slice->ToString()));
    }
    return absl::OkStatus();
  }

  absl::Status HandleBitSliceUpdate(BitSliceUpdate* update) override {
    XLS_RETURN_IF_ERROR(ExpectOperandHasBitsType(update, 0));
    XLS_RETURN_IF_ERROR(ExpectOperandHasBitsType(update, 1));
    XLS_RETURN_IF_ERROR(ExpectOperandHasBitsType(update, 2));
    XLS_RETURN_IF_ERROR(
        ExpectHasBitsType(update, update->to_update()->BitCountOrDie()));
    return absl::OkStatus();
  }

  absl::Status HandleConcat(Concat* concat) override {
    // All operands should be bits types.
    int64_t total_bits = 0;
    for (int64_t i = 0; i < concat->operand_count(); ++i) {
      Type* operand_type = concat->operand(i)->GetType();
      XLS_RETURN_IF_ERROR(ExpectHasBitsType(concat->operand(i)));
      total_bits += operand_type->AsBitsOrDie()->bit_count();
    }
    return ExpectHasBitsType(concat, /*expected_bit_count=*/total_bits);
  }

  absl::Status HandleCountedFor(CountedFor* counted_for) override {
    XLS_RET_CHECK_GE(counted_for->trip_count(), 0);
    if (counted_for->operand_count() == 0) {
      return absl::InternalError(StrFormat(
          "Expected %s to have at least 1 operand", counted_for->GetName()));
    }

    Function* body = counted_for->body();

    // Verify function has signature
    //  body(i: bits[N], loop_carry_data: T, [inv_arg0, ..., inv_argN]) -> T
    //  where N is of sufficient size to store 0 .. stride * (trip_count - 1)
    //  where T is an arbitrary type
    //  where inv_argX each have arbitrary types
    int64_t invariant_args_count = counted_for->operand_count() - 1;

    // Verify number of parameters
    int64_t expected_param_count = 2 + invariant_args_count;
    int64_t actual_param_count = body->params().size();

    if (actual_param_count != expected_param_count) {
      return absl::InternalError(
          StrFormat("Function %s used as counted_for body should have "
                    "%d parameters, got %d instead; body type: %s; node: %s",
                    body->name(), expected_param_count, actual_param_count,
                    body->GetType()->ToString(), counted_for->ToString()));
    }

    // Verify i is of type bits with a sufficient width and at least 1 bit
    Type* i_type = body->param(0)->GetType();

    int64_t trip_count = counted_for->trip_count();
    int64_t max_i = counted_for->stride() * (trip_count - 1);
    int64_t min_i_bits =
        (trip_count <= 1) ? 1 : Bits::MinBitCountUnsigned(max_i);

    if (!i_type->IsBits() || i_type->AsBitsOrDie()->bit_count() < min_i_bits) {
      return absl::InternalError(
          StrFormat("Parameter 0 (%s) of function %s used as counted_for "
                    "body should have bits[N] type, where N >= %d, got %s "
                    "instead: %s",
                    body->param(0)->GetName(), body->name(), min_i_bits,
                    i_type->ToString(), counted_for->ToString()));
    }

    // Verify return type and loop_carry_data are of the correct type
    Type* data_type = counted_for->operand(0)->GetType();
    Type* body_ret_type = body->return_value()->GetType();
    Type* body_data_param_type = body->param(1)->GetType();

    if (data_type != body_ret_type) {
      return absl::InternalError(
          StrFormat("Return type of function %s used as counted_for "
                    "body should have %s type, got %s instead: %s",
                    body->name(), data_type->ToString(),
                    body_ret_type->ToString(), counted_for->ToString()));
    }

    if (data_type != body_data_param_type) {
      return absl::InternalError(StrFormat(
          "Parameter 1 (%s) of function %s used as counted_for "
          "body should have %s type, got %s instead: %s",
          body->param(1)->GetName(), body->name(), data_type->ToString(),
          body_data_param_type->ToString(), counted_for->ToString()));
    }

    // Verify invariant arg type matches corresponding function param type
    for (int64_t i = 0; i < invariant_args_count; ++i) {
      Type* inv_arg_type = counted_for->operand(i + 1)->GetType();
      Type* body_inv_param_type = body->param(i + 2)->GetType();

      if (inv_arg_type != body_inv_param_type) {
        return absl::InternalError(StrFormat(
            "Parameter %d (%s) of function %s used as counted_for "
            "body should have %s type from %s, got %s instead: %s",
            i + 2, body->param(i + 2)->GetName(), body->name(),
            inv_arg_type->ToString(), counted_for->operand(i + 1)->ToString(),
            body_inv_param_type->ToString(), counted_for->ToString()));
      }
    }

    return ExpectOperandHasType(counted_for, 0, counted_for->GetType());
  }

  absl::Status HandleDecode(Decode* decode) override {
    XLS_RETURN_IF_ERROR(ExpectOperandHasBitsType(decode, 0));
    XLS_RETURN_IF_ERROR(ExpectHasBitsType(decode, decode->width()));
    // The width of the decode output must be less than or equal to
    // 2**input_width.
    const int64_t operand_width = decode->operand(0)->BitCountOrDie();
    if (operand_width < 63 && (decode->width() > (1LL << operand_width))) {
      return absl::InternalError(
          StrFormat("Decode output width (%d) greater than 2**${operand width} "
                    "where operand width is %d",
                    decode->width(), operand_width));
    }
    return absl::OkStatus();
  }

  absl::Status HandleDynamicCountedFor(
      DynamicCountedFor* dynamic_counted_for) override {
    Function* body = dynamic_counted_for->body();
    // Verify function has signature
    //  body(i: bits[N], loop_carry_data: T, [inv_arg0, ..., inv_argN]) -> T
    //  where T is an arbitrary type
    //  where inv_argX each have arbitrary types
    int64_t invariant_args_count = dynamic_counted_for->operand_count() - 3;

    // Verify number of parameters
    int64_t expected_param_count = 2 + invariant_args_count;
    int64_t actual_param_count = body->params().size();
    if (actual_param_count != expected_param_count) {
      return absl::InternalError(
          StrFormat("Function %s used as dynamic_counted_for body should have "
                    "%d parameters, got %d instead: %s",
                    body->name(), expected_param_count, actual_param_count,
                    dynamic_counted_for->ToString()));
    }

    // Verify index is of type bits.
    Type* index_type = body->param(0)->GetType();
    if (!index_type->IsBits()) {
      return absl::InternalError(StrFormat(
          "Parameter 0 (%s) of function %s used as dynamic_counted_for "
          "body should have bits type.",
          body->param(0)->GetName(), body->name()));
    }

    // Verify return type and loop_carry_data are of the correct type
    Type* data_type = dynamic_counted_for->initial_value()->GetType();
    Type* body_ret_type = body->return_value()->GetType();
    Type* body_data_param_type = body->param(1)->GetType();

    if (data_type != body_ret_type) {
      return absl::InternalError(StrFormat(
          "Return type of function %s used as dynamic_counted_for "
          "body should have %s type, got %s instead: %s",
          body->name(), data_type->ToString(), body_ret_type->ToString(),
          dynamic_counted_for->ToString()));
    }

    if (data_type != body_data_param_type) {
      return absl::InternalError(StrFormat(
          "Parameter 1 (%s) of function %s used as dynamic_counted_for "
          "body should have %s type, got %s instead: %s",
          body->param(1)->GetName(), body->name(), data_type->ToString(),
          body_data_param_type->ToString(), dynamic_counted_for->ToString()));
    }

    // Verify invariant arg type matches corresponding function param type
    for (int64_t i = 0; i < invariant_args_count; ++i) {
      Type* inv_arg_type =
          dynamic_counted_for->invariant_args().at(i)->GetType();
      Type* body_inv_param_type = body->param(i + 2)->GetType();

      if (inv_arg_type != body_inv_param_type) {
        return absl::InternalError(StrFormat(
            "Parameter %d (%s) of function %s used as dynamic_counted_for "
            "body should have %s type from %s, got %s instead: %s",
            i + 2, body->param(i + 2)->GetName(), body->name(),
            inv_arg_type->ToString(),
            dynamic_counted_for->invariant_args().at(i)->ToString(),
            body_inv_param_type->ToString(), dynamic_counted_for->ToString()));
      }
    }

    // Verify that trip_count and stride are bit types of acceptable size.
    Type* trip_count_type = dynamic_counted_for->trip_count()->GetType();
    if (!trip_count_type->IsBits()) {
      return absl::InternalError(
          StrFormat("Operand 1 / trip_count of dynamic_counted_for "
                    "should have bits type."));
    }
    if (!(trip_count_type->AsBitsOrDie()->bit_count() <
          index_type->AsBitsOrDie()->bit_count())) {
      return absl::InternalError(
          StrFormat("Operand 1 / trip_count of dynamic_counted_for "
                    "should have < the number of bits of the function body "
                    "index parameter / function body Operand 0"));
    }

    Type* stride_type = dynamic_counted_for->stride()->GetType();
    if (!stride_type->IsBits()) {
      return absl::InternalError(
          StrFormat("Operand 2 / stride of dynamic_counted_for "
                    "should have bits type."));
    }
    // Trip count should have fewer bits than the index because index (and
    // stride) are treated as signed values while trip count is an unsigned
    // value. If trip count had the same number of bits as the index, it would
    // end up with 1 more bit than the index after adding a 0 sign bit.
    if (!(stride_type->AsBitsOrDie()->bit_count() <=
          index_type->AsBitsOrDie()->bit_count())) {
      return absl::InternalError(
          StrFormat("Operand 2 / stride of dynamic_counted_for "
                    "should have <= the number of bits of the function body "
                    "index parameter / function body Operand 0"));
    }

    return ExpectOperandHasType(dynamic_counted_for, 0,
                                dynamic_counted_for->GetType());
  }

  absl::Status HandleEncode(Encode* encode) override {
    XLS_RETURN_IF_ERROR(ExpectOperandHasBitsType(encode, 0));
    // Width of the encode output must be ceil(log_2(operand_width)). Subtract
    // one from the width to account for zero-based numbering.
    return ExpectHasBitsType(
        encode,
        Bits::MinBitCountUnsigned(encode->operand(0)->BitCountOrDie() - 1));
  }

  absl::Status HandleUDiv(BinOp* div) override {
    XLS_RETURN_IF_ERROR(ExpectOperandCount(div, 2));
    return ExpectAllSameBitsType(div);
  }

  absl::Status HandleSDiv(BinOp* div) override {
    XLS_RETURN_IF_ERROR(ExpectOperandCount(div, 2));
    return ExpectAllSameBitsType(div);
  }

  absl::Status HandleUMod(BinOp* mod) override {
    XLS_RETURN_IF_ERROR(ExpectOperandCount(mod, 2));
    return ExpectAllSameBitsType(mod);
  }

  absl::Status HandleSMod(BinOp* mod) override {
    XLS_RETURN_IF_ERROR(ExpectOperandCount(mod, 2));
    return ExpectAllSameBitsType(mod);
  }

  absl::Status HandleEq(CompareOp* eq) override {
    XLS_RETURN_IF_ERROR(ExpectOperandCount(eq, 2));
    XLS_RETURN_IF_ERROR(ExpectOperandsSameBitsType(eq));
    return ExpectHasBitsType(eq, /*expected_bit_count=*/1);
  }

  absl::Status HandleUGe(CompareOp* ge) override {
    XLS_RETURN_IF_ERROR(ExpectOperandCount(ge, 2));
    XLS_RETURN_IF_ERROR(ExpectOperandsSameBitsType(ge));
    return ExpectHasBitsType(ge, /*expected_bit_count=*/1);
  }

  absl::Status HandleUGt(CompareOp* gt) override {
    XLS_RETURN_IF_ERROR(ExpectOperandCount(gt, 2));
    XLS_RETURN_IF_ERROR(ExpectOperandsSameBitsType(gt));
    return ExpectHasBitsType(gt, /*expected_bit_count=*/1);
  }

  absl::Status HandleSGe(CompareOp* ge) override {
    XLS_RETURN_IF_ERROR(ExpectOperandCount(ge, 2));
    XLS_RETURN_IF_ERROR(ExpectOperandsSameBitsType(ge));
    return ExpectHasBitsType(ge, /*expected_bit_count=*/1);
  }

  absl::Status HandleSGt(CompareOp* gt) override {
    XLS_RETURN_IF_ERROR(ExpectOperandCount(gt, 2));
    XLS_RETURN_IF_ERROR(ExpectOperandsSameBitsType(gt));
    return ExpectHasBitsType(gt, /*expected_bit_count=*/1);
  }

  absl::Status HandleIdentity(UnOp* identity) override {
    XLS_RETURN_IF_ERROR(ExpectOperandCount(identity, 1));
    return ExpectAllSameType(identity);
  }

  absl::Status HandleArrayIndex(ArrayIndex* index) override {
    XLS_RETURN_IF_ERROR(ExpectOperandCountGt(index, 0));
    XLS_RETURN_IF_ERROR(VerifyMultidimensionalArrayIndex(
        index->indices(), index->array()->GetType(), index));
    XLS_ASSIGN_OR_RETURN(Type * indexed_type,
                         GetIndexedElementType(index->array()->GetType(),
                                               index->indices().size()));
    if (index->GetType() != indexed_type) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Expected array index operation %s to have type %s",
                          index->GetName(), indexed_type->ToString()));
    }
    return absl::OkStatus();
  }

  absl::Status HandleArraySlice(ArraySlice* slice) override {
    Node* array = slice->array();
    int64_t width = slice->width();
    XLS_RETURN_IF_ERROR(ExpectOperandHasArrayType(slice, 0));
    XLS_RETURN_IF_ERROR(ExpectOperandHasBitsType(slice, 1));
    XLS_RETURN_IF_ERROR(ExpectHasArrayType(
        slice, array->GetType()->AsArrayOrDie()->element_type(), width));
    if (array->GetType()->AsArrayOrDie()->size() == 0) {
      return absl::InvalidArgumentError(
          "Array slice cannot be applied to an empty array");
    }
    if (width <= 0) {
      return absl::InvalidArgumentError(
          "Array slice requires a positive width");
    }
    return absl::OkStatus();
  }

  absl::Status HandleArrayUpdate(ArrayUpdate* update) override {
    XLS_RETURN_IF_ERROR(ExpectOperandCountGt(update, 1));
    XLS_RETURN_IF_ERROR(
        ExpectSameType(update, update->GetType(), update->array_to_update(),
                       update->array_to_update()->GetType(),
                       "array update operation", "input array"));

    XLS_RETURN_IF_ERROR(VerifyMultidimensionalArrayIndex(
        update->indices(), update->array_to_update()->GetType(), update));
    XLS_ASSIGN_OR_RETURN(
        Type * indexed_type,
        GetIndexedElementType(update->array_to_update()->GetType(),
                              update->indices().size()));

    if (update->update_value()->GetType() != indexed_type) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Expected update value of array update operation %s "
                          "to have type %s, has type %s",
                          update->GetName(), indexed_type->ToString(),
                          update->update_value()->GetType()->ToString()));
    }
    return absl::OkStatus();
  }

  absl::Status HandleArrayConcat(ArrayConcat* array_concat) override {
    // Must have at least one operand
    XLS_RETURN_IF_ERROR(ExpectOperandCountGt(array_concat, 0));

    // Verify operands are all arrays and that their elements are
    // of the same type
    int64_t size = 0;
    Type* zeroth_element_type = nullptr;

    for (int64_t i = 0; i < array_concat->operand_count(); ++i) {
      Node* operand = array_concat->operand(i);

      XLS_RETURN_IF_ERROR(ExpectOperandHasArrayType(array_concat, i));

      ArrayType* operand_type = operand->GetType()->AsArrayOrDie();
      Type* element_type = operand_type->element_type();

      if (!zeroth_element_type) {
        zeroth_element_type = element_type;
      } else if (element_type != zeroth_element_type) {
        return absl::InternalError(StrFormat(
            "Element type of operand %d of %s (%s via %s) "
            "does not match element type of operand 0 (%s via %s): %s",
            i, array_concat->GetName(), element_type->ToString(),
            operand->GetName(), zeroth_element_type->ToString(),
            array_concat->operand(0)->GetName(), array_concat->ToString()));
      }

      size += operand_type->size();
    }

    // Verify return type is an array, with the expected type and size
    return ExpectHasArrayType(array_concat, zeroth_element_type, size);
  }

  absl::Status HandleInvoke(Invoke* invoke) override {
    // Verify the signature (inputs and output) of the invoked function matches
    // the Invoke node.
    Function* func = invoke->to_apply();
    if (invoke->operand_count() != func->params().size()) {
      std::string arg_types_str = absl::StrJoin(
          invoke->operands(), ", ", [](std::string* out, Node* node) {
            absl::StrAppend(out, node->GetType()->ToString());
          });
      return absl::InternalError(absl::StrFormat(
          "Expected invoke operand count (%d) to equal invoked function "
          "parameter count (%d); function name: %s; signature: %s; arg types: "
          "[%s]",
          invoke->operand_count(), func->params().size(), func->name(),
          func->GetType()->ToString(), arg_types_str));
    }
    for (int64_t i = 0; i < invoke->operand_count(); ++i) {
      XLS_RETURN_IF_ERROR(
          ExpectOperandHasType(invoke, i, func->param(i)->GetType()));
    }

    XLS_RETURN_IF_ERROR(
        ExpectSameType(invoke, invoke->GetType(), func->return_value(),
                       func->return_value()->GetType(), "invoke operation",
                       "invoked function return value"));

    return absl::OkStatus();
  }

  absl::Status HandleLiteral(Literal* literal) override {
    // Verify type matches underlying Value object.
    XLS_RETURN_IF_ERROR(ExpectOperandCount(literal, 0));
    return ExpectValueIsType(literal->value(), literal->GetType());
  }

  absl::Status HandleULe(CompareOp* le) override {
    XLS_RETURN_IF_ERROR(ExpectOperandCount(le, 2));
    XLS_RETURN_IF_ERROR(ExpectOperandsSameBitsType(le));
    return ExpectHasBitsType(le, /*expected_bit_count=*/1);
  }

  absl::Status HandleULt(CompareOp* lt) override {
    XLS_RETURN_IF_ERROR(ExpectOperandCount(lt, 2));
    XLS_RETURN_IF_ERROR(ExpectOperandsSameBitsType(lt));
    return ExpectHasBitsType(lt, /*expected_bit_count=*/1);
  }
  absl::Status HandleSLe(CompareOp* le) override {
    XLS_RETURN_IF_ERROR(ExpectOperandCount(le, 2));
    XLS_RETURN_IF_ERROR(ExpectOperandsSameBitsType(le));
    return ExpectHasBitsType(le, /*expected_bit_count=*/1);
  }

  absl::Status HandleSLt(CompareOp* lt) override {
    XLS_RETURN_IF_ERROR(ExpectOperandCount(lt, 2));
    XLS_RETURN_IF_ERROR(ExpectOperandsSameBitsType(lt));
    return ExpectHasBitsType(lt, /*expected_bit_count=*/1);
  }

  absl::Status HandleMap(Map* map) override {
    XLS_RETURN_IF_ERROR(ExpectOperandCount(map, 1));
    XLS_RETURN_IF_ERROR(ExpectHasArrayType(map));
    XLS_RETURN_IF_ERROR(ExpectHasArrayType(map->operand(0)));

    // Verify the signature of the applied function against the operand and
    // output element types.
    Type* output_element_type = map->GetType()->AsArrayOrDie()->element_type();
    XLS_RETURN_IF_ERROR(ExpectSameType(
        map, output_element_type, map->to_apply()->return_value(),
        map->to_apply()->return_value()->GetType(), "map output element",
        "applied function return type"));

    Type* operand_element_type =
        map->operand(0)->GetType()->AsArrayOrDie()->element_type();
    XLS_RET_CHECK_EQ(1, map->to_apply()->params().size());
    XLS_RETURN_IF_ERROR(ExpectSameType(
        map->operand(0), operand_element_type, map->to_apply()->params()[0],
        map->to_apply()->params()[0]->GetType(), "map operand element",
        "applied function input type"));

    return absl::OkStatus();
  }

  absl::Status HandleSMul(ArithOp* mul) override {
    XLS_RETURN_IF_ERROR(ExpectOperandCount(mul, 2));
    XLS_RETURN_IF_ERROR(ExpectOperandHasBitsType(mul, 0));
    XLS_RETURN_IF_ERROR(ExpectOperandHasBitsType(mul, 1));
    return ExpectHasBitsType(mul);
  }

  absl::Status HandleUMul(ArithOp* mul) override {
    XLS_RETURN_IF_ERROR(ExpectOperandCount(mul, 2));
    XLS_RETURN_IF_ERROR(ExpectOperandHasBitsType(mul, 0));
    XLS_RETURN_IF_ERROR(ExpectOperandHasBitsType(mul, 1));
    return ExpectHasBitsType(mul);
  }

  absl::Status HandleNe(CompareOp* ne) override {
    XLS_RETURN_IF_ERROR(ExpectOperandCount(ne, 2));
    XLS_RETURN_IF_ERROR(ExpectOperandsSameBitsType(ne));
    return ExpectHasBitsType(ne, /*expected_bit_count=*/1);
  }

  absl::Status HandleNeg(UnOp* neg) override {
    XLS_RETURN_IF_ERROR(ExpectOperandCount(neg, 1));
    return ExpectAllSameBitsType(neg);
  }

  absl::Status HandleNot(UnOp* not_op) override {
    XLS_RETURN_IF_ERROR(ExpectOperandCount(not_op, 1));
    return ExpectAllSameBitsType(not_op);
  }

  absl::Status HandleOneHot(OneHot* one_hot) override {
    XLS_RETURN_IF_ERROR(ExpectOperandCount(one_hot, 1));
    XLS_RETURN_IF_ERROR(ExpectOperandHasBitsType(one_hot, 0));
    int64_t operand_bit_count = one_hot->operand(0)->BitCountOrDie();
    // The output of one_hot should be one wider than the input to account for
    // the default value.
    return ExpectHasBitsType(one_hot, operand_bit_count + 1);
  }

  absl::Status HandleOneHotSel(OneHotSelect* sel) override {
    if (sel->operand_count() < 2) {
      return absl::InternalError(
          StrFormat("Expected %s to have at least 2 operands", sel->GetName()));
    }
    XLS_RETURN_IF_ERROR(ExpectOperandHasBitsType(sel, /*operand_no=*/0));
    int64_t selector_width = sel->selector()->BitCountOrDie();
    if (selector_width != sel->cases().size()) {
      return absl::InternalError(StrFormat("Selector has %d bits for %d cases",
                                           selector_width,
                                           sel->cases().size()));
    }
    return absl::OkStatus();
  }

  absl::Status HandleNaryOr(NaryOp* or_op) override {
    XLS_RETURN_IF_ERROR(ExpectOperandCountGt(or_op, 0));
    return ExpectAllSameBitsType(or_op);
  }

  absl::Status HandleNaryXor(NaryOp* xor_op) override {
    XLS_RETURN_IF_ERROR(ExpectOperandCountGt(xor_op, 0));
    return ExpectAllSameBitsType(xor_op);
  }

  absl::Status HandleParam(Param* param) override {
    return ExpectOperandCount(param, 0);
  }

  absl::Status HandleReverse(UnOp* reverse) override {
    XLS_RETURN_IF_ERROR(ExpectOperandCount(reverse, 1));
    return ExpectAllSameBitsType(reverse);
  }

  absl::Status HandleSel(Select* sel) override {
    if (sel->operand_count() < 2) {
      return absl::InternalError(
          StrFormat("Expected %s to have at least 2 operands", sel->GetName()));
    }

    XLS_RETURN_IF_ERROR(ExpectHasBitsType(sel->selector()));
    const int64_t selector_width = sel->selector()->BitCountOrDie();
    const int64_t minimum_selector_width =
        Bits::MinBitCountUnsigned(sel->cases().size() - 1);
    const bool power_of_2_cases = IsPowerOfTwo(sel->cases().size());
    if (selector_width < minimum_selector_width) {
      return absl::InternalError(StrFormat(
          "Selector must have at least %d bits to select amongst %d cases (has "
          "only %d bits)",
          minimum_selector_width, sel->cases().size(), selector_width));
    } else if (selector_width == minimum_selector_width && power_of_2_cases &&
               sel->default_value().has_value()) {
      return absl::InternalError(
          StrFormat("Select has useless default value: selector has %d bits "
                    "with %d cases",
                    selector_width, sel->cases().size()));
    } else if ((selector_width > minimum_selector_width ||
                (selector_width == minimum_selector_width &&
                 !power_of_2_cases)) &&
               !sel->default_value().has_value()) {
      return absl::InternalError(StrFormat(
          "Select has no default value: selector has %d bits with %d cases",
          selector_width, sel->cases().size()));
    }

    for (int64_t i = 0; i < sel->cases().size(); ++i) {
      Type* operand_type = sel->get_case(i)->GetType();
      if (operand_type != sel->GetType()) {
        return absl::InternalError(StrFormat(
            "Case %d (operand %d) type %s does not match node type: %s", i,
            i + 1, operand_type->ToString(), sel->ToString()));
      }
    }
    return absl::OkStatus();
  }

  absl::Status HandleShll(BinOp* shll) override { return HandleShiftOp(shll); }

  absl::Status HandleShra(BinOp* shra) override { return HandleShiftOp(shra); }

  absl::Status HandleShrl(BinOp* shrl) override { return HandleShiftOp(shrl); }

  absl::Status HandleSub(BinOp* sub) override {
    XLS_RETURN_IF_ERROR(ExpectOperandCount(sub, 2));
    return ExpectAllSameBitsType(sub);
  }

  absl::Status HandleTuple(Tuple* tuple) override {
    XLS_RETURN_IF_ERROR(ExpectHasTupleType(tuple));
    if (!tuple->GetType()->IsTuple()) {
      return absl::InternalError(
          StrFormat("Expected node to have tuple type: %s", tuple->ToString()));
    }
    TupleType* type = tuple->GetType()->AsTupleOrDie();
    if (type->size() != tuple->operand_count()) {
      return absl::InternalError(
          StrFormat("Type element count %d does not match operand count %d: %s",
                    type->size(), tuple->operand_count(), tuple->ToString()));
    }
    for (int64_t i = 0; i < tuple->operand_count(); ++i) {
      XLS_RETURN_IF_ERROR(
          ExpectOperandHasType(tuple, i, type->element_type(i)));
    }
    return absl::OkStatus();
  }

  absl::Status HandleTupleIndex(TupleIndex* index) override {
    XLS_RETURN_IF_ERROR(ExpectOperandCount(index, 1));
    XLS_RETURN_IF_ERROR(ExpectHasTupleType(index->operand(0)));
    TupleType* operand_type = index->operand(0)->GetType()->AsTupleOrDie();
    if ((index->index() < 0) || (index->index() >= operand_type->size())) {
      return absl::InternalError(
          StrFormat("Tuple index value %d out of bounds: %s", index->index(),
                    index->ToString()));
    }
    Type* element_type = operand_type->element_type(index->index());
    return ExpectSameType(index, index->GetType(), index->operand(0),
                          element_type, "tuple index operation",
                          "tuple operand element type");
  }

  absl::Status HandleSignExtend(ExtendOp* sign_ext) override {
    return HandleExtendOp(sign_ext);
  }
  absl::Status HandleZeroExtend(ExtendOp* zero_ext) override {
    return HandleExtendOp(zero_ext);
  }

  absl::Status HandleInputPort(InputPort* input_port) override {
    return absl::OkStatus();
  }

  absl::Status HandleOutputPort(OutputPort* output_port) override {
    return absl::OkStatus();
  }

  absl::Status HandleRegisterRead(RegisterRead* reg_read) override {
    return absl::OkStatus();
  }

  absl::Status HandleRegisterWrite(RegisterWrite* reg_write) override {
    return absl::OkStatus();
  }

  absl::Status HandleGate(Gate* gate) override {
    XLS_RETURN_IF_ERROR(ExpectOperandCount(gate, 2));
    XLS_RETURN_IF_ERROR(ExpectOperandHasBitsType(gate, /*operand_no=*/0,
                                                 /*expected_bit_count=*/1));

    return ExpectOperandHasType(gate, 1, gate->GetType());
  }

 private:
  absl::Status HandleShiftOp(Node* shift) {
    // A shift-amount operand can have arbitrary width, but the shifted operand
    // and the shift operation must be identical.
    XLS_RETURN_IF_ERROR(ExpectHasBitsType(shift));
    XLS_RETURN_IF_ERROR(
        ExpectSameType(shift->operand(0), shift->operand(0)->GetType(), shift,
                       shift->GetType(), "operand 0", "shift operation"));
    return ExpectOperandHasBitsType(shift, 1);
  }

  absl::Status HandleExtendOp(ExtendOp* ext) {
    XLS_RETURN_IF_ERROR(ExpectOperandCount(ext, 1));
    XLS_RETURN_IF_ERROR(ExpectOperandHasBitsType(ext, /*operand_no=*/0));
    int64_t operand_bit_count = ext->operand(0)->BitCountOrDie();
    int64_t new_bit_count = ext->new_bit_count();
    if (new_bit_count < operand_bit_count) {
      return absl::InternalError(StrFormat(
          "Extending operation %s is actually truncating from %d bits to %d "
          "bits.",
          ext->ToStringWithOperandTypes(), operand_bit_count, new_bit_count));
    }
    return ExpectHasBitsType(ext, new_bit_count);
  }

  // Verifies that the given node has the expected number of operands.
  absl::Status ExpectOperandCount(Node* node, int64_t expected) {
    if (node->operand_count() != expected) {
      return absl::InternalError(
          StrFormat("Expected %s to have %d operands, has %d", node->GetName(),
                    expected, node->operand_count()));
    }
    return absl::OkStatus();
  }

  // Verifies that the given node has a number of operands between the two
  // limits (inclusive).
  absl::Status ExpectOperandCountRange(Node* node, int64_t lower_limit,
                                       int64_t upper_limit) {
    if (node->operand_count() < lower_limit ||
        node->operand_count() > upper_limit) {
      return absl::InternalError(StrFormat(
          "Expected %s to have between %d and %d operands, has %d",
          node->GetName(), lower_limit, upper_limit, node->operand_count()));
    }
    return absl::OkStatus();
  }

  absl::Status ExpectOperandCountGt(Node* node, int64_t expected) {
    if (node->operand_count() <= expected) {
      return absl::InternalError(
          StrFormat("Expected %s to have > %d operands, has %d",
                    node->GetName(), expected, node->operand_count()));
    }
    return absl::OkStatus();
  }

  // Verifies that the given two types match. The argument desc_a (desc_b) is a
  // description of type_a (type_b) used in the error message. The arguments are
  // intentionally const char* rather than string_view because we want to avoid
  // *eagerly* constructing potentially expensive strings to include in the
  // error message.
  absl::Status ExpectSameType(Node* a_source, Type* type_a, Node* b_source,
                              Type* type_b, const char* desc_a,
                              const char* desc_b) const {
    if (type_a != type_b) {
      return absl::InternalError(StrFormat(
          "Type of %s (%s via %s) does not match type of %s (%s via %s)",
          desc_a, type_a->ToString(), a_source->GetName(), desc_b,
          type_b->ToString(), b_source->GetName()));
    }
    return absl::OkStatus();
  }

  // Verifies that a particular operand of the given node has the given type.
  absl::Status ExpectOperandHasType(Node* node, int64_t operand_no,
                                    Type* type) const {
    if (node->operand(operand_no)->GetType() != type) {
      return absl::InternalError(
          StrFormat("Expected operand %d of %s to have type %s, has type %s.",
                    operand_no, node->GetName(), type->ToString(),
                    node->operand(operand_no)->GetType()->ToString()));
    }
    return absl::OkStatus();
  }

  absl::Status ExpectHasTokenType(Node* node) const {
    if (!node->GetType()->IsToken()) {
      return absl::InternalError(
          StrFormat("Expected %s to have token type, has type %s",
                    node->GetName(), node->GetType()->ToString()));
    }
    return absl::OkStatus();
  }

  absl::Status ExpectHasArrayType(Node* node,
                                  Type* expected_element_type = nullptr,
                                  int64_t expected_size = -1) const {
    if (!node->GetType()->IsArray()) {
      return absl::InternalError(
          StrFormat("Expected %s to have Array type, has type %s",
                    node->GetName(), node->GetType()->ToString()));
    }

    Type* element_type = node->GetType()->AsArrayOrDie()->element_type();
    if (expected_element_type && element_type != expected_element_type) {
      return absl::InternalError(StrFormat(
          "Expected %s to have element type %s, has type %s", node->GetName(),
          expected_element_type->ToString(), element_type->ToString()));
    }

    int64_t size = node->GetType()->AsArrayOrDie()->size();
    if (expected_size >= 0 && size != expected_size) {
      return absl::InternalError(
          StrFormat("Expected %s to have size %d, has size %d", node->GetName(),
                    expected_size, size));
    }

    return absl::OkStatus();
  }

  absl::Status ExpectOperandHasArrayType(Node* node, int64_t operand_no,
                                         Type* expected_element_type = nullptr,
                                         int64_t expected_size = -1) const {
    Node* operand = node->operand(operand_no);

    if (!operand->GetType()->IsArray()) {
      return absl::InternalError(
          StrFormat("Expected operand %d of %s to have Array type, "
                    "has type %s: %s",
                    operand_no, node->GetName(), operand->GetType()->ToString(),
                    node->ToString()));
    }

    Type* element_type = operand->GetType()->AsArrayOrDie()->element_type();
    if (expected_element_type && element_type != expected_element_type) {
      return absl::InternalError(StrFormat(
          "Expected operand %d of %s to have "
          "element type %s, has type %s: %s",
          operand_no, node->GetName(), expected_element_type->ToString(),
          element_type->ToString(), node->ToString()));
    }

    int64_t size = operand->GetType()->AsArrayOrDie()->size();
    if (expected_size >= 0 && size != expected_size) {
      return absl::InternalError(StrFormat(
          "Expected operand %d of %s to have size %d, "
          "has size %d: %s",
          operand_no, node->GetName(), expected_size, size, node->ToString()));
    }

    return absl::OkStatus();
  }

  absl::Status ExpectOperandHasTupleType(Node* node, int64_t operand_no) {
    Node* operand = node->operand(operand_no);

    if (!operand->GetType()->IsTuple()) {
      return absl::InternalError(
          StrFormat("Expected operand %d of %s to have Tuple type, "
                    "has type %s: %s",
                    operand_no, node->GetName(), operand->GetType()->ToString(),
                    node->ToString()));
    }
    return absl::OkStatus();
  }

  absl::Status ExpectHasTupleType(Node* node) const {
    if (!node->GetType()->IsTuple()) {
      return absl::InternalError(
          StrFormat("Expected %s to have Tuple type, has type %s",
                    node->GetName(), node->GetType()->ToString()));
    }
    return absl::OkStatus();
  }

  absl::Status ExpectHasBitsType(Node* node,
                                 int64_t expected_bit_count = -1) const {
    if (!node->GetType()->IsBits()) {
      return absl::InternalError(
          StrFormat("Expected %s to have Bits type, has type %s",
                    node->GetName(), node->GetType()->ToString()));
    }
    if (expected_bit_count != -1 &&
        node->BitCountOrDie() != expected_bit_count) {
      return absl::InternalError(
          StrFormat("Expected node to have bit count %d: %s",
                    expected_bit_count, node->ToString()));
    }
    return absl::OkStatus();
  }

  absl::Status ExpectOperandHasBitsType(Node* node, int64_t operand_no,
                                        int64_t expected_bit_count = -1) const {
    Node* operand = node->operand(operand_no);
    if (!operand->GetType()->IsBits()) {
      return absl::InternalError(
          StrFormat("Expected operand %d of %s have Bits type, has type %s: %s",
                    operand_no, node->GetName(), node->GetType()->ToString(),
                    node->ToString()));
    }
    if (expected_bit_count != -1 &&
        operand->BitCountOrDie() != expected_bit_count) {
      return absl::InternalError(StrFormat(
          "Expected operand %d of %s to have bit count %d: %s", operand_no,
          node->GetName(), expected_bit_count, node->ToString()));
    }
    return absl::OkStatus();
  }

  absl::Status ExpectOperandHasTokenType(Node* node, int64_t operand_no) const {
    Node* operand = node->operand(operand_no);
    if (!operand->GetType()->IsToken()) {
      return absl::InternalError(StrFormat(
          "Expected operand %d of %s to have Token type, has type %s: %s",
          operand_no, node->GetName(), operand->GetType()->ToString(),
          node->ToString()));
    }
    return absl::OkStatus();
  }

  // Verifies all operands and the node itself are BitsType with the same bit
  // count.
  absl::Status ExpectAllSameBitsType(Node* node) const {
    XLS_RETURN_IF_ERROR(ExpectHasBitsType(node));
    return ExpectAllSameType(node);
  }

  // Verifies all operands and the node itself are the same type.
  absl::Status ExpectAllSameType(Node* node) const {
    for (int64_t i = 0; i < node->operand_count(); ++i) {
      XLS_RETURN_IF_ERROR(ExpectOperandHasType(node, i, node->GetType()));
    }
    return absl::OkStatus();
  }

  // Verifies all operands are BitsType with the same bit count.
  absl::Status ExpectOperandsSameBitsType(Node* node) const {
    if (node->operand_count() == 0) {
      return absl::OkStatus();
    }
    Type* type = node->operand(0)->GetType();
    for (int64_t i = 1; i < node->operand_count(); ++i) {
      XLS_RETURN_IF_ERROR(ExpectOperandHasType(node, i, type));
    }
    return absl::OkStatus();
  }

  // Verifies that the given Value has the given type. Walks the structures
  // recursively.
  absl::Status ExpectValueIsType(const Value& value, Type* type) {
    switch (value.kind()) {
      case ValueKind::kBits:
        XLS_RET_CHECK(type->IsBits());
        XLS_RET_CHECK_EQ(value.bits().bit_count(),
                         type->AsBitsOrDie()->bit_count());
        break;
      case ValueKind::kToken:
        XLS_RET_CHECK(type->IsToken());
        break;
      case ValueKind::kTuple: {
        XLS_RET_CHECK(type->IsTuple());
        TupleType* tuple_type = type->AsTupleOrDie();
        XLS_RET_CHECK_EQ(value.elements().size(), tuple_type->size());
        for (int64_t i = 0; i < tuple_type->size(); ++i) {
          XLS_RETURN_IF_ERROR(ExpectValueIsType(value.elements()[i],
                                                tuple_type->element_type(i)));
        }
        break;
      }
      case ValueKind::kArray: {
        XLS_RET_CHECK(type->IsArray());
        ArrayType* array_type = type->AsArrayOrDie();
        XLS_RET_CHECK_EQ(value.elements().size(), array_type->size());
        for (int64_t i = 0; i < array_type->size(); ++i) {
          XLS_RETURN_IF_ERROR(ExpectValueIsType(value.elements()[i],
                                                array_type->element_type()));
        }
        break;
      }
      default:
        return absl::InternalError("Invalid Value type.");
    }
    return absl::OkStatus();
  }

  // Verifies that the given index_type can be used as a multi-dimensional index
  // into type_to_index (as in multi-array index/update operations). index_type
  // should be a tuple of bits types.
  absl::Status VerifyMultidimensionalArrayIndex(absl::Span<Node* const> indices,
                                                Type* type_to_index,
                                                Node* node) {
    // All elements of the index must be bits type.
    for (int64_t i = 0; i < indices.size(); ++i) {
      Node* index = indices[i];
      if (!index->GetType()->IsBits()) {
        return absl::InvalidArgumentError(
            absl::StrFormat("All elements of index of node %s must be bits "
                            "type; index element %d type: %s",
                            node->GetName(), i, index->GetType()->ToString()));
      }
    }
    if (indices.size() > GetArrayDimensionCount(type_to_index)) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Index of node %s has more elements (%d) than the array dimensions "
          "(%d); array type: %s",
          node->GetName(), indices.size(),
          GetArrayDimensionCount(type_to_index), type_to_index->ToString()));
    }
    return absl::OkStatus();
  }
};

absl::Status VerifyNodeIdUnique(
    Node* node,
    absl::flat_hash_map<int64_t, absl::optional<SourceLocation>>* ids) {
  // TODO(meheff): param IDs currently collide with non-param IDs. All IDs
  // should be globally unique.
  if (!node->Is<Param>()) {
    if (!ids->insert({node->id(), node->loc()}).second) {
      const absl::optional<SourceLocation>& loc = ids->at(node->id());
      return absl::InternalError(absl::StrFormat(
          "ID %d is not unique; previously seen source location: %s",
          node->id(), loc.has_value() ? loc->ToString().c_str() : "<none>"));
    }
  }
  return absl::OkStatus();
}

// Verify common invariants to function-level constucts.
absl::Status VerifyFunctionBase(FunctionBase* function) {
  XLS_VLOG(2) << absl::StreamFormat("Verifying function %s:\n",
                                    function->name());
  XLS_VLOG_LINES(4, function->DumpIr());

  // Verify all types are owned by package.
  for (Node* node : function->nodes()) {
    XLS_RET_CHECK(node->package()->IsOwnedType(node->GetType()));
    XLS_RET_CHECK(node->package() == function->package());
  }

  // Verify ids are unique within the function.
  absl::flat_hash_map<int64_t, absl::optional<SourceLocation>> ids;
  ids.reserve(function->node_count());
  for (Node* node : function->nodes()) {
    XLS_RETURN_IF_ERROR(VerifyNodeIdUnique(node, &ids));
  }

  // Verify consistency of node::users() and node::operands().
  for (Node* node : function->nodes()) {
    XLS_RETURN_IF_ERROR(VerifyNode(node));
  }

  // Verify the set of parameter nodes is exactly Function::params(), and that
  // the parameter names are unique.
  absl::flat_hash_set<std::string> param_names;
  absl::flat_hash_set<Node*> param_set;
  for (Node* param : function->params()) {
    XLS_RET_CHECK(param_set.insert(param).second)
        << "Param appears more than once in Function::params()";
    XLS_RET_CHECK(param_names.insert(param->GetName()).second)
        << "Param name " << param->GetName()
        << " is duplicated in Function::params()";
  }
  int64_t param_node_count = 0;
  for (Node* node : function->nodes()) {
    if (node->Is<Param>()) {
      XLS_RET_CHECK(param_set.contains(node))
          << "Param " << node->GetName() << " is not in Function::params()";
      param_node_count++;
    }
  }
  XLS_RET_CHECK_EQ(param_set.size(), param_node_count)
      << "Number of param nodes not equal to Function::params() size for "
         "function "
      << function->name();

  return absl::OkStatus();
}

// Returns the channel used by the given send or receive node. Returns an error
// if the given node is not a send or receive.
absl::StatusOr<Channel*> GetSendOrReceiveChannel(Node* node) {
  if (node->Is<Send>()) {
    return node->package()->GetChannel(node->As<Send>()->channel_id());
  }
  if (node->Is<Receive>()) {
    return node->package()->GetChannel(node->As<Receive>()->channel_id());
  }
  return absl::InternalError(absl::StrFormat(
      "Node is not a send or receive node: %s", node->ToString()));
}

// Verify that all tokens in the given FunctionBase are connected. All tokens
// should flow from the source token to the sink token.
absl::Status VerifyTokenConnectivity(Node* source_token, Node* sink_token,
                                     FunctionBase* f) {
  absl::flat_hash_set<Node*> visited;
  std::deque<Node*> worklist;
  auto maybe_add_to_worklist = [&](Node* n) {
    if (visited.contains(n)) {
      return;
    }
    worklist.push_back(n);
    visited.insert(n);
  };

  // Verify connectivity to source param.
  absl::flat_hash_set<Node*> connected_to_source;
  maybe_add_to_worklist(source_token);
  while (!worklist.empty()) {
    Node* node = worklist.front();
    worklist.pop_front();
    connected_to_source.insert(node);
    if (TypeHasToken(node->GetType())) {
      for (Node* user : node->users()) {
        maybe_add_to_worklist(user);
      }
    }
  }

  // Verify connectivity to sink token.
  absl::flat_hash_set<Node*> connected_to_sink;
  visited.clear();
  maybe_add_to_worklist(sink_token);
  while (!worklist.empty()) {
    Node* node = worklist.front();
    worklist.pop_front();
    connected_to_sink.insert(node);
    for (Node* operand : node->operands()) {
      if (TypeHasToken(operand->GetType())) {
        maybe_add_to_worklist(operand);
      }
    }
  }

  for (Node* node : f->nodes()) {
    if (TypeHasToken(node->GetType())) {
      if (!connected_to_source.contains(node)) {
        return absl::InternalError(absl::StrFormat(
            "Token-typed nodes must be connected to the source token "
            "via a path of tokens: %s.",
            node->GetName()));
      }
      if (!connected_to_sink.contains(node)) {
        return absl::InternalError(absl::StrFormat(
            "Token-typed nodes must be connected to the sink token value "
            "via a path of tokens: %s.",
            node->GetName()));
      }
    }
  }

  if (!connected_to_source.contains(sink_token)) {
    return absl::InternalError(
        absl::StrFormat("The sink token must be connected to the token "
                        "parameter via a path of tokens: %s.",
                        sink_token->GetName()));
  }

  return absl::OkStatus();
}

// Verify various invariants about the channels owned by the given pacakge.
absl::Status VerifyChannels(Package* package) {
  // Verify unique ids.
  absl::flat_hash_map<int64_t, Channel*> channels_by_id;
  for (Channel* channel : package->channels()) {
    XLS_RET_CHECK(!channels_by_id.contains(channel->id()))
        << absl::StreamFormat("More than one channel has id %d: '%s' and '%s'",
                              channel->id(), channel->name(),
                              channels_by_id.at(channel->id())->name());
    channels_by_id[channel->id()] = channel;
  }

  // Verify unique names.
  absl::flat_hash_map<std::string, Channel*> channels_by_name;
  for (Channel* channel : package->channels()) {
    XLS_RET_CHECK(!channels_by_name.contains(channel->name()))
        << absl::StreamFormat(
               "More than one channel has name '%s'. IDs of channels: %d and "
               "%d",
               channel->name(), channel->id(),
               channels_by_name.at(channel->name())->id());
    channels_by_name[channel->name()] = channel;
  }

  // Verify each channel has the appropriate send/receive node.
  absl::flat_hash_map<Channel*, Node*> send_nodes;
  absl::flat_hash_map<Channel*, Node*> receive_nodes;
  for (auto& proc : package->procs()) {
    for (Node* node : proc->nodes()) {
      if (node->Is<Send>()) {
        XLS_ASSIGN_OR_RETURN(Channel * channel, GetSendOrReceiveChannel(node));
        XLS_RET_CHECK(!send_nodes.contains(channel)) << absl::StreamFormat(
            "Multiple send nodes associated with channel '%s': %s and %s (at "
            "least).",
            channel->name(), node->GetName(),
            send_nodes.at(channel)->GetName());
        send_nodes[channel] = node;
      }
      if (node->Is<Receive>()) {
        XLS_ASSIGN_OR_RETURN(Channel * channel, GetSendOrReceiveChannel(node));
        XLS_RET_CHECK(!receive_nodes.contains(channel)) << absl::StreamFormat(
            "Multiple receive nodes associated with channel '%s': %s and %s "
            "(at least).",
            channel->name(), node->GetName(),
            receive_nodes.at(channel)->GetName());
        receive_nodes[channel] = node;
      }
    }
  }

  // Verify that each channel has the appropriate number of send and receive
  // nodes (one or zero).
  for (Channel* channel : package->channels()) {
    if (channel->CanSend()) {
      XLS_RET_CHECK(send_nodes.contains(channel)) << absl::StreamFormat(
          "Channel '%s' (id %d) has no associated send node", channel->name(),
          channel->id());
    } else {
      XLS_RET_CHECK(!send_nodes.contains(channel)) << absl::StreamFormat(
          "Channel '%s' (id %d) cannot send but has a send node %s",
          channel->name(), channel->id(), send_nodes.at(channel)->GetName());
    }
    if (channel->CanReceive()) {
      XLS_RET_CHECK(receive_nodes.contains(channel)) << absl::StreamFormat(
          "Channel '%s' (id %d) has no associated receive node",
          channel->name(), channel->id());
    } else {
      XLS_RET_CHECK(!receive_nodes.contains(channel)) << absl::StreamFormat(
          "Channel '%s' (id %d) cannot receive but has a receive node %s",
          channel->name(), channel->id(), receive_nodes.at(channel)->GetName());
    }

    // Verify type-specific invariants of each channel.
    if (channel->kind() == ChannelKind::kSingleValue) {
      // Single-value channels cannot have initial values.
      XLS_RET_CHECK_EQ(channel->initial_values().size(), 0);
      // TODO(meheff): 2021/06/24 Single-value channels should not support
      // Send and Receive with predicates. Add check when such uses are removed.
    }
  }

  return absl::OkStatus();
}

}  // namespace

absl::Status VerifyPackage(Package* package) {
  XLS_VLOG(4) << absl::StreamFormat("Verifying package %s:\n", package->name());
  XLS_VLOG_LINES(4, package->DumpIr());

  for (auto& function : package->functions()) {
    XLS_RETURN_IF_ERROR(VerifyFunction(function.get()));
  }

  for (auto& proc : package->procs()) {
    XLS_RETURN_IF_ERROR(VerifyProc(proc.get()));
  }

  for (auto& block : package->blocks()) {
    XLS_RETURN_IF_ERROR(VerifyBlock(block.get()));
  }

  // Verify node IDs are unique within the package and uplinks point to this
  // package.
  absl::flat_hash_map<int64_t, absl::optional<SourceLocation>> ids;
  ids.reserve(package->GetNodeCount());
  for (FunctionBase* function : package->GetFunctionBases()) {
    XLS_RET_CHECK(function->package() == package);
    for (Node* node : function->nodes()) {
      XLS_RETURN_IF_ERROR(VerifyNodeIdUnique(node, &ids));
      XLS_RET_CHECK(node->package() == package);
    }
  }

  // Ensure that the package's "next ID" is not in the space of IDs currently
  // occupied by the package's nodes.
  int64_t max_id_seen = -1;
  for (const auto& item : ids) {
    max_id_seen = std::max(item.first, max_id_seen);
  }
  XLS_RET_CHECK_GT(package->next_node_id(), max_id_seen);

  // Verify function, proc, block names are unique among functions/procs/blocks.
  absl::flat_hash_set<FunctionBase*> function_bases;
  absl::flat_hash_set<std::string> function_names;
  absl::flat_hash_set<std::string> proc_names;
  absl::flat_hash_set<std::string> block_names;
  for (FunctionBase* function_base : package->GetFunctionBases()) {
    absl::flat_hash_set<std::string>* name_set;
    if (function_base->IsFunction()) {
      name_set = &function_names;
    } else if (function_base->IsProc()) {
      name_set = &proc_names;
    } else {
      XLS_RET_CHECK(function_base->IsBlock());
      name_set = &block_names;
    }
    XLS_RET_CHECK(!name_set->contains(function_base->name()))
        << "Function/proc/block with name " << function_base->name()
        << " is not unique within package " << package->name();
    name_set->insert(function_base->name());

    XLS_RET_CHECK(!function_bases.contains(function_base))
        << "Function or proc with name " << function_base->name()
        << " appears more than once in within package" << package->name();
    function_bases.insert(function_base);
  }

  XLS_RETURN_IF_ERROR(VerifyChannels(package));

  // TODO(meheff): Verify main entry point is one of the functions.
  // TODO(meheff): Verify functions called by any node are in the set of
  //   functions owned by the package.
  // TODO(meheff): Verify that there is no recursion.

  return absl::OkStatus();
}

absl::Status VerifyFunction(Function* function) {
  XLS_VLOG(4) << "Verifying function:\n";
  XLS_VLOG_LINES(4, function->DumpIr());

  XLS_RETURN_IF_ERROR(VerifyFunctionBase(function));

  for (Node* node : function->nodes()) {
    if (node->Is<Send>() || node->Is<Receive>()) {
      return absl::InternalError(absl::StrFormat(
          "Send and receive nodes can only be in procs, not functions (%s)",
          node->GetName()));
    }
  }

  return absl::OkStatus();
}

absl::Status VerifyProc(Proc* proc) {
  XLS_VLOG(4) << "Verifying proc:\n";
  XLS_VLOG_LINES(4, proc->DumpIr());

  XLS_RETURN_IF_ERROR(VerifyFunctionBase(proc));

  // A Proc should have two parameters: a token (parameter 0), and the recurent
  // state (parameter 1).
  XLS_RET_CHECK_EQ(proc->params().size(), 2) << absl::StreamFormat(
      "Proc %s does not have two parameters", proc->name());

  XLS_RET_CHECK_EQ(proc->param(0), proc->TokenParam());
  XLS_RET_CHECK_EQ(proc->param(0)->GetType(), proc->package()->GetTokenType())
      << absl::StreamFormat("Parameter 0 of a proc %s is not token type, is %s",
                            proc->name(),
                            proc->param(1)->GetType()->ToString());

  XLS_RET_CHECK_EQ(proc->param(1), proc->StateParam());
  XLS_RET_CHECK_EQ(proc->param(1)->GetType(), proc->StateType())
      << absl::StreamFormat(
             "Parameter 1 of a proc %s does not match state type %s, is %s",
             proc->name(), proc->StateType()->ToString(),
             proc->param(1)->GetType()->ToString());

  // Next token must be token type.
  XLS_RET_CHECK(proc->NextToken()->GetType()->IsToken());

  // Next state must be state type.
  XLS_RET_CHECK_EQ(proc->NextState()->GetType(), proc->StateType());

  // Verify that all send/receive nodes are connected to the token parameter and
  // the return value via paths of tokens.
  XLS_RETURN_IF_ERROR(
      VerifyTokenConnectivity(proc->TokenParam(), proc->NextToken(), proc));

  return absl::OkStatus();
}

absl::Status VerifyBlock(Block* block) {
  XLS_VLOG(4) << "Verifying block:\n";
  XLS_VLOG_LINES(4, block->DumpIr());

  XLS_RETURN_IF_ERROR(VerifyFunctionBase(block));

  // Verify the nodes returned by Block::Get*Port methods are consistent.
  absl::flat_hash_set<Node*> all_data_ports;
  for (const Block::Port& port : block->GetPorts()) {
    if (absl::holds_alternative<InputPort*>(port)) {
      all_data_ports.insert(absl::get<InputPort*>(port));
    } else if (absl::holds_alternative<OutputPort*>(port)) {
      all_data_ports.insert(absl::get<OutputPort*>(port));
    }
  }
  absl::flat_hash_set<Node*> input_data_ports(block->GetInputPorts().begin(),
                                              block->GetInputPorts().end());
  absl::flat_hash_set<Node*> output_data_ports(block->GetOutputPorts().begin(),
                                               block->GetOutputPorts().end());

  // All the pointers returned by the GetPort methods should be unique.
  XLS_RET_CHECK_EQ(block->GetInputPorts().size(), input_data_ports.size());
  XLS_RET_CHECK_EQ(block->GetOutputPorts().size(), output_data_ports.size());
  XLS_RET_CHECK_EQ(
      block->GetInputPorts().size() + block->GetOutputPorts().size(),
      all_data_ports.size());

  int64_t input_port_count = 0;
  int64_t output_port_count = 0;
  for (Node* node : block->nodes()) {
    if (node->Is<InputPort>()) {
      XLS_RET_CHECK(all_data_ports.contains(node)) << node->GetName();
      XLS_RET_CHECK(input_data_ports.contains(node)) << node->GetName();
      input_port_count++;
    } else if (node->Is<OutputPort>()) {
      XLS_RET_CHECK(all_data_ports.contains(node)) << node->GetName();
      XLS_RET_CHECK(output_data_ports.contains(node)) << node->GetName();
      output_port_count++;
    }
  }
  XLS_RET_CHECK_EQ(input_port_count, input_data_ports.size());
  XLS_RET_CHECK_EQ(output_port_count, output_data_ports.size());

  // Blocks should have no parameters.
  XLS_RET_CHECK(block->params().empty());

  // The block must have a clock port if it has any registers.
  if (!block->GetRegisters().empty() && !block->GetClockPort().has_value()) {
    return absl::InternalError(
        StrFormat("Block has registers but no clock port"));
  }

  // Verify all registers have exactly one read and write operation.
  absl::flat_hash_map<Register*, RegisterRead*> reg_reads;
  absl::flat_hash_map<Register*, RegisterWrite*> reg_writes;
  for (Node* node : block->nodes()) {
    if (node->Is<RegisterRead>()) {
      RegisterRead* reg_read = node->As<RegisterRead>();
      XLS_ASSIGN_OR_RETURN(Register * reg,
                           block->GetRegister(reg_read->register_name()));
      if (reg_reads.contains(reg)) {
        return absl::InternalError(
            StrFormat("Register %s has multiple reads", reg->name()));
      }
      XLS_RET_CHECK_EQ(reg->type(), node->GetType());
      reg_reads[reg] = reg_read;
    } else if (node->Is<RegisterWrite>()) {
      RegisterWrite* reg_write = node->As<RegisterWrite>();
      XLS_ASSIGN_OR_RETURN(Register * reg,
                           block->GetRegister(reg_write->register_name()));
      if (reg_writes.contains(reg)) {
        return absl::InternalError(
            StrFormat("Register %s has multiple writes", reg->name()));
      }
      XLS_RET_CHECK_EQ(reg->type(), reg_write->data()->GetType());
      if (reg_write->load_enable().has_value()) {
        XLS_RET_CHECK_EQ(reg_write->load_enable().value()->GetType(),
                         block->package()->GetBitsType(1));
      }
      reg_writes[reg] = reg_write;
    }
  }
  for (Register* reg : block->GetRegisters()) {
    if (!reg_reads.contains(reg)) {
      return absl::InternalError(
          StrFormat("Register %s has no read", reg->name()));
    }
    if (!reg_writes.contains(reg)) {
      return absl::InternalError(
          StrFormat("Register %s has no write", reg->name()));
    }
  }

  return absl::OkStatus();
}

absl::Status VerifyNode(Node* node) {
  XLS_VLOG(4) << "Verifying node: " << node->ToString();

  for (Node* operand : node->operands()) {
    XLS_RET_CHECK(operand->HasUser(node))
        << "Expected " << node->GetName() << " to be a user of "
        << operand->GetName();
    XLS_RET_CHECK(operand->function_base() == node->function_base())
        << StrFormat("Operand %s of node %s not in same function (%s vs %s).",
                     operand->GetName(), node->GetName(),
                     operand->function_base()->name(),
                     node->function_base()->name());
  }
  for (Node* user : node->users()) {
    XLS_RET_CHECK(absl::c_linear_search(user->operands(), node))
        << "Expected " << node->GetName() << " to be a operand of "
        << user->GetName();
  }

  NodeChecker node_checker;
  return node->VisitSingleNode(&node_checker);
}

}  // namespace xls
