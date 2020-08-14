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

#include "xls/ir/verifier.h"

#include "absl/strings/str_format.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/logging/logging.h"
#include "xls/common/math_util.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/dfs_visitor.h"

namespace xls {
namespace {

using ::absl::StrCat;
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

  absl::Status HandleArray(Array* array) override {
    XLS_RETURN_IF_ERROR(ExpectHasArrayType(array));
    ArrayType* array_type = array->GetType()->AsArrayOrDie();
    XLS_RETURN_IF_ERROR(ExpectOperandCount(array, array_type->size()));
    Type* element_type = array_type->element_type();
    for (int64 i = 0; i < array->operand_count(); ++i) {
      XLS_RETURN_IF_ERROR(ExpectSameType(
          array->operand(i), array->operand(i)->GetType(), array, element_type,
          StrCat("operand ", i), "array element type"));
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
    const int64 bits_required = bit_slice->start() + bit_slice->width();
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

  absl::Status HandleConcat(Concat* concat) override {
    // All operands should be bits types.
    int64 total_bits = 0;
    for (int64 i = 0; i < concat->operand_count(); ++i) {
      Type* operand_type = concat->operand(i)->GetType();
      XLS_RETURN_IF_ERROR(ExpectHasBitsType(concat));
      total_bits += operand_type->AsBitsOrDie()->bit_count();
    }
    return ExpectHasBitsType(concat, /*expected_bit_count=*/total_bits);
  }

  absl::Status HandleCountedFor(CountedFor* counted_for) override {
    // TODO(meheff): Verify signature of called function.

    XLS_RET_CHECK_GE(counted_for->trip_count(), 0);
    if (counted_for->operand_count() == 0) {
      return absl::InternalError(StrFormat(
          "Expected %s to have at least 1 operand", counted_for->GetName()));
    }
    return ExpectSameType(
        counted_for->operand(0), counted_for->operand(0)->GetType(),
        counted_for, counted_for->GetType(), "operand", counted_for->GetName());
  }

  absl::Status HandleDecode(Decode* decode) override {
    XLS_RETURN_IF_ERROR(ExpectOperandHasBitsType(decode, 0));
    XLS_RETURN_IF_ERROR(ExpectHasBitsType(decode, decode->width()));
    // The width of the decode output must be less than or equal to
    // 2**input_width.
    const int64 operand_width = decode->operand(0)->BitCountOrDie();
    if (operand_width < 63 && (decode->width() > (1LL << operand_width))) {
      return absl::InternalError(
          StrFormat("Decode output width (%d) greater than 2**${operand width} "
                    "where operand width is %d",
                    decode->width(), operand_width));
    }
    return absl::OkStatus();
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
    XLS_RETURN_IF_ERROR(ExpectOperandCount(index, 2));
    XLS_RETURN_IF_ERROR(ExpectHasArrayType(index->operand(0)));
    XLS_RETURN_IF_ERROR(ExpectOperandHasBitsType(index, 1));
    Type* element_type =
        index->operand(0)->GetType()->AsArrayOrDie()->element_type();
    return ExpectSameType(index, index->GetType(), index->operand(0),
                          element_type, "array index operation",
                          "array operand element type");
  }

  absl::Status HandleArrayUpdate(ArrayUpdate* update) override {
    XLS_RETURN_IF_ERROR(ExpectOperandCount(update, 3));
    XLS_RETURN_IF_ERROR(ExpectHasArrayType(update));
    XLS_RETURN_IF_ERROR(ExpectOperandHasBitsType(update, 1));
    XLS_RETURN_IF_ERROR(
        ExpectSameType(update, update->GetType(), update->operand(0),
                       update->operand(0)->GetType(), "array update operation",
                       "input array"));
    XLS_RETURN_IF_ERROR(ExpectOperandHasBitsType(update, 1));
    Type* element_type = update->GetType()->AsArrayOrDie()->element_type();
    return ExpectSameType(update, element_type, update->operand(2),
                          update->operand(2)->GetType(),
                          "array update operation elements", "update value");
  }

  absl::Status HandleInvoke(Invoke* invoke) override {
    // Verify the signature (inputs and output) of the invoked function matches
    // the Invoke node.
    Function* func = invoke->to_apply();
    for (int64 i = 0; i < invoke->operand_count(); ++i) {
      XLS_RETURN_IF_ERROR(ExpectSameType(
          invoke->operand(i), invoke->operand(i)->GetType(), func->params()[i],
          func->params()[i]->GetType(), StrFormat("invoke operand %d", i),
          StrFormat("invoked function argument %d", i)));
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
    int64 operand_bit_count = one_hot->operand(0)->BitCountOrDie();
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
    int64 selector_width = sel->selector()->BitCountOrDie();
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
    const int64 selector_width = sel->selector()->BitCountOrDie();
    const int64 minimum_selector_width =
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

    for (int64 i = 0; i < sel->cases().size(); ++i) {
      Type* operand_type = sel->cases()[i]->GetType();
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
    for (int64 i = 0; i < tuple->operand_count(); ++i) {
      XLS_RETURN_IF_ERROR(
          ExpectSameType(tuple->operand(i), tuple->operand(i)->GetType(), tuple,
                         type->element_type(i), StrFormat("operand %d", i),
                         StrFormat("tuple node %s", tuple->ToString())));
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
    int64 operand_bit_count = ext->operand(0)->BitCountOrDie();
    int64 new_bit_count = ext->new_bit_count();
    if (new_bit_count < operand_bit_count) {
      return absl::InternalError(StrFormat(
          "Extending operation %s is actually truncating from %d bits to %d "
          "bits.",
          ext->ToStringWithOperandTypes(), operand_bit_count, new_bit_count));
    }
    return ExpectHasBitsType(ext, new_bit_count);
  }

  // Verifies that the given node has the expected number of operands.
  absl::Status ExpectOperandCount(Node* node, int64 expected) {
    if (node->operand_count() != expected) {
      return absl::InternalError(
          StrFormat("Expected %s to have %d operands, has %d", node->GetName(),
                    expected, node->operand_count()));
    }
    return absl::OkStatus();
  }

  absl::Status ExpectOperandCountGt(Node* node, int64 expected) {
    if (node->operand_count() <= expected) {
      return absl::InternalError(
          StrFormat("Expected %s to have > %d operands, has %d",
                    node->GetName(), expected, node->operand_count()));
    }
    return absl::OkStatus();
  }

  // Verifies that the given two types match. The argument desc_a (desc_b) is a
  // description of type_a (type_b) used in the error message.
  absl::Status ExpectSameType(Node* a_source, Type* type_a, Node* b_source,
                              Type* type_b, absl::string_view desc_a,
                              absl::string_view desc_b) const {
    if (type_a != type_b) {
      return absl::InternalError(StrFormat(
          "Type of %s (%s via %s) does not match type of %s (%s via %s)",
          desc_a, type_a->ToString(), a_source->GetName(), desc_b,
          type_b->ToString(), b_source->GetName()));
    }
    return absl::OkStatus();
  }

  absl::Status ExpectHasArrayType(Node* node) const {
    if (!node->GetType()->IsArray()) {
      return absl::InternalError(
          StrFormat("Expected %s to have Array type, has type %s",
                    node->GetName(), node->GetType()->ToString()));
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
                                 int64 expected_bit_count = -1) const {
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

  absl::Status ExpectOperandHasBitsType(Node* node, int64 operand_no,
                                        int64 expected_bit_count = -1) const {
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

  // Verifies all operands and the node itself are BitsType with the same bit
  // count.
  absl::Status ExpectAllSameBitsType(Node* node) const {
    XLS_RETURN_IF_ERROR(ExpectHasBitsType(node));
    return ExpectAllSameType(node);
  }

  // Verifies all operands and the node itself are the same type.
  absl::Status ExpectAllSameType(Node* node) const {
    for (int64 i = 0; i < node->operand_count(); ++i) {
      XLS_RETURN_IF_ERROR(ExpectSameType(
          node->operand(i), node->operand(i)->GetType(), node, node->GetType(),
          StrFormat("operand %d", i), node->GetName()));
    }
    return absl::OkStatus();
  }

  // Verifies all operands are BitsType with the same bit count.
  absl::Status ExpectOperandsSameBitsType(Node* node) const {
    if (node->operand_count() == 0) {
      return absl::OkStatus();
    }
    Type* type = node->operand(0)->GetType();
    for (int64 i = 1; i < node->operand_count(); ++i) {
      XLS_RETURN_IF_ERROR(ExpectSameType(
          node->operand(i), node->operand(i)->GetType(), node->operand(0), type,
          StrFormat("operand %d", i), "operand 0"));
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
      case ValueKind::kTuple: {
        XLS_RET_CHECK(type->IsTuple());
        TupleType* tuple_type = type->AsTupleOrDie();
        XLS_RET_CHECK_EQ(value.elements().size(), tuple_type->size());
        for (int64 i = 0; i < tuple_type->size(); ++i) {
          XLS_RETURN_IF_ERROR(ExpectValueIsType(value.elements()[i],
                                                tuple_type->element_type(i)));
        }
        break;
      }
      case ValueKind::kArray: {
        XLS_RET_CHECK(type->IsArray());
        ArrayType* array_type = type->AsArrayOrDie();
        XLS_RET_CHECK_EQ(value.elements().size(), array_type->size());
        for (int64 i = 0; i < array_type->size(); ++i) {
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
};

absl::Status VerifyNodeIdUnique(
    Node* node,
    absl::flat_hash_map<int64, absl::optional<SourceLocation>>* ids) {
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

}  // namespace

absl::Status Verify(Package* package) {
  XLS_VLOG(2) << "Verifying package:\n";
  XLS_VLOG_LINES(2, package->DumpIr());

  for (auto& function : package->functions()) {
    XLS_RETURN_IF_ERROR(Verify(function.get()));
  }

  // Verify node IDs are unique within the package and uplinks point to this
  // package.
  absl::flat_hash_map<int64, absl::optional<SourceLocation>> ids;
  ids.reserve(package->GetNodeCount());
  for (auto& function : package->functions()) {
    XLS_RET_CHECK(function->package() == package);
    for (Node* node : function->nodes()) {
      XLS_RETURN_IF_ERROR(VerifyNodeIdUnique(node, &ids));
      XLS_RET_CHECK(node->package() == package);
    }
  }

  // Ensure that the package's "next ID" is not in the space of IDs currently
  // occupied by the package's nodes.
  int64 max_id_seen = -1;
  for (const auto& item : ids) {
    max_id_seen = std::max(item.first, max_id_seen);
  }
  XLS_RET_CHECK_GT(package->next_node_id(), max_id_seen);

  // Verify function names are unique within the package.
  absl::flat_hash_set<Function*> functions;
  absl::flat_hash_set<std::string> function_names;
  for (auto& function : package->functions()) {
    XLS_RET_CHECK(!function_names.contains(function->name()))
        << "Function with name " << function->name()
        << " is not unique within package " << package->name();
    function_names.insert(function->name());

    XLS_RET_CHECK(!functions.contains(function.get()))
        << "Function with name " << function->name()
        << " appears more than once in function list within package "
        << package->name();
    functions.insert(function.get());
  }

  // TODO(meheff): Verify main entry point is one of the functions.
  // TODO(meheff): Verify functions called by any node are in the set of
  //   functions owned by the package.
  // TODO(meheff): Verify that there is no recursion.

  return absl::OkStatus();
}

absl::Status Verify(Function* function) {
  XLS_VLOG(2) << "Verifying function:\n";
  XLS_VLOG_LINES(2, function->DumpIr());

  // Verify all types are owned by package.
  for (Node* node : function->nodes()) {
    XLS_RET_CHECK(node->package()->IsOwnedType(node->GetType()));
    XLS_RET_CHECK(node->package() == function->package());
  }

  // Verify ids are unique within the function.
  absl::flat_hash_map<int64, absl::optional<SourceLocation>> ids;
  ids.reserve(function->node_count());
  for (Node* node : function->nodes()) {
    XLS_RETURN_IF_ERROR(VerifyNodeIdUnique(node, &ids));
  }

  // Verify consistency of node::users() and node::operands().
  for (Node* node : function->nodes()) {
    XLS_RETURN_IF_ERROR(Verify(node));
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
  int64 param_node_count = 0;
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

absl::Status Verify(Node* node) {
  XLS_VLOG(2) << "Verifying node: " << node->ToString();

  for (Node* operand : node->operands()) {
    XLS_RET_CHECK(operand->HasUser(node))
        << "Expected " << node->GetName() << " to be a user of "
        << operand->GetName();
    XLS_RET_CHECK(operand->function() == node->function())
        << StrFormat("Operand %s of node %s not in same function (%s vs %s).",
                     operand->GetName(), node->GetName(),
                     operand->function()->name(), node->function()->name());
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
