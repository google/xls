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

#include "xls/ir/ir_interpreter.h"

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_interpreter_stats.h"
#include "xls/ir/keyword_args.h"
#include "xls/ir/node_iterator.h"
#include "xls/ir/package.h"

namespace xls {
namespace ir_interpreter {

// A visitor for traversing and evaluating a Function.
class InterpreterVisitor : public DfsVisitor {
 public:
  // Runs the visitor on the given function. 'args' are the argument values
  // indexed by parameter name.
  static xabsl::StatusOr<Value> Run(Function* function,
                                    absl::Span<const Value> args,
                                    InterpreterStats* stats) {
    if (args.size() != function->params().size()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Function %s wants %d arguments, got %d.", function->name(),
          function->params().size(), args.size()));
    }
    for (int64 argno = 0; argno < args.size(); ++argno) {
      Param* param = function->param(argno);
      const Value& value = args[argno];
      Type* param_type = param->GetType();
      Type* value_type = function->package()->GetTypeForValue(value);
      if (value_type != param_type) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Got argument %s for parameter %d which is not of type %s",
            value.ToString(), argno, param_type->ToString()));
      }
    }
    InterpreterVisitor visitor(args, stats);
    XLS_RETURN_IF_ERROR(function->return_value()->Accept(&visitor));
    return visitor.ResolveAsValue(function->return_value());
  }

  static xabsl::StatusOr<Value> EvaluateNodeWithLiteralOperands(Node* node) {
    InterpreterVisitor visitor({}, /*stats=*/nullptr);
    XLS_RETURN_IF_ERROR(node->Accept(&visitor));
    return visitor.ResolveAsValue(node);
  }

  static xabsl::StatusOr<Value> EvaluateNode(
      Node* node, absl::Span<const Value* const> operand_values) {
    XLS_RET_CHECK_EQ(node->operand_count(), operand_values.size());
    InterpreterVisitor visitor({}, /*stats=*/nullptr);
    for (int64 i = 0; i < operand_values.size(); ++i) {
      visitor.node_values_[node->operand(i)] = *operand_values[i];
    }
    XLS_RETURN_IF_ERROR(node->VisitSingleNode(&visitor));
    return visitor.ResolveAsValue(node);
  }

  absl::Status HandleAdd(BinOp* add) override {
    return SetBitsResult(add, bits_ops::Add(ResolveAsBits(add->operand(0)),
                                            ResolveAsBits(add->operand(1))));
  }

  absl::Status HandleAndReduce(BitwiseReductionOp* and_reduce) override {
    Bits operand = ResolveAsBits(and_reduce->operand(0));
    return SetBitsResult(and_reduce, bits_ops::AndReduce(operand));
  }

  absl::Status HandleOrReduce(BitwiseReductionOp* or_reduce) override {
    Bits operand = ResolveAsBits(or_reduce->operand(0));
    return SetBitsResult(or_reduce, bits_ops::OrReduce(operand));
  }

  absl::Status HandleXorReduce(BitwiseReductionOp* xor_reduce) override {
    Bits operand = ResolveAsBits(xor_reduce->operand(0));
    return SetBitsResult(xor_reduce, bits_ops::XorReduce(operand));
  }

  absl::Status HandleNaryAnd(NaryOp* and_op) override {
    Bits accum = ResolveAsBits(and_op->operand(0));
    for (Node* operand : and_op->operands().subspan(1)) {
      accum = bits_ops::And(accum, ResolveAsBits(operand));
    }
    return SetBitsResult(and_op, accum);
  }

  absl::Status HandleNaryNand(NaryOp* nand_op) override {
    std::vector<Bits> operands = ResolveAsBits(nand_op->operands());
    Bits accum = bits_ops::NaryNand(operands);
    return SetBitsResult(nand_op, accum);
  }

  absl::Status HandleNaryNor(NaryOp* nor_op) override {
    std::vector<Bits> operands = ResolveAsBits(nor_op->operands());
    Bits accum = bits_ops::NaryNor(operands);
    return SetBitsResult(nor_op, accum);
  }

  absl::Status HandleNaryOr(NaryOp* or_op) override {
    std::vector<Bits> operands = ResolveAsBits(or_op->operands());
    Bits accum = bits_ops::NaryOr(operands);
    return SetBitsResult(or_op, accum);
  }

  absl::Status HandleNaryXor(NaryOp* xor_op) override {
    std::vector<Bits> operands = ResolveAsBits(xor_op->operands());
    Bits accum = bits_ops::NaryXor(operands);
    return SetBitsResult(xor_op, accum);
  }

  absl::Status HandleArray(Array* array) override {
    std::vector<Value> operand_values;
    for (Node* operand : array->operands()) {
      operand_values.push_back(ResolveAsValue(operand));
    }
    XLS_ASSIGN_OR_RETURN(Value result, Value::Array(operand_values));
    return SetValueResult(array, result);
  }

  absl::Status HandleBitSlice(BitSlice* bit_slice) override {
    return SetBitsResult(bit_slice,
                         ResolveAsBits(bit_slice->operand(0))
                             .Slice(bit_slice->start(), bit_slice->width()));
  }

  absl::Status HandleDynamicBitSlice(
      DynamicBitSlice* dynamic_bit_slice) override {
    int64 start = ResolveAsBits(dynamic_bit_slice->operand(1)).ToInt64().value();
    return SetBitsResult(dynamic_bit_slice,
                         ResolveAsBits(dynamic_bit_slice->operand(0))
                         .Slice(start, dynamic_bit_slice->width()));
  }

  absl::Status HandleConcat(Concat* concat) override {
    std::vector<Bits> operand_values;
    for (Node* operand : concat->operands()) {
      operand_values.push_back(ResolveAsBits(operand));
    }
    return SetBitsResult(concat, bits_ops::Concat(operand_values));
  }

  absl::Status HandleCountedFor(CountedFor* counted_for) override {
    Function* body = counted_for->body();
    std::vector<Value> invariant_args;
    // Set the loop invariant args (from the second operand on up).
    for (int64 i = 1; i < counted_for->operand_count(); ++i) {
      // The n-th operand of the counted for actually feeds the (n+1)-th
      // parameter of the body as the first two are the induction variable and
      // the loop state.
      invariant_args.push_back(ResolveAsValue(counted_for->operand(i)));
    }
    Value loop_state = ResolveAsValue(counted_for->operand(0));
    BitsType* arg0_type = body->param(0)->GetType()->AsBitsOrDie();
    // For each iteration of counted_for, update the induction variable and loop
    // state arguments (params 0 and 1) and recursively call the interpreter
    // Run() on the body function -- the new accumulator value is the return
    // value of interpreting body.
    for (int64 i = 0, iv = 0; i < counted_for->trip_count();
         ++i, iv += counted_for->stride()) {
      std::vector<Value> args_for_body = {
          Value(UBits(iv, arg0_type->bit_count())), loop_state};
      for (const auto& value : invariant_args) {
        args_for_body.push_back(value);
      }
      XLS_ASSIGN_OR_RETURN(loop_state, Run(body, args_for_body, stats_));
    }
    return SetValueResult(counted_for, loop_state);
  }

  absl::Status HandleDecode(Decode* decode) override {
    XLS_ASSIGN_OR_RETURN(int64 input_value,
                         ResolveAsBits(decode->operand(0)).ToUint64());
    if (input_value < decode->BitCountOrDie()) {
      return SetBitsResult(decode,
                           Bits::PowerOfTwo(/*set_bit_index=*/input_value,
                                            decode->BitCountOrDie()));
    } else {
      return SetBitsResult(decode, Bits(decode->BitCountOrDie()));
    }
  }

  absl::Status HandleEncode(Encode* encode) override {
    const Bits& input = ResolveAsBits(encode->operand(0));
    Bits result(encode->BitCountOrDie());
    for (int64 i = 0; i < input.bit_count(); ++i) {
      if (input.Get(i)) {
        result = bits_ops::Or(result, UBits(i, encode->BitCountOrDie()));
      }
    }
    return SetBitsResult(encode, result);
  }

  absl::Status HandleUDiv(BinOp* div) override {
    return SetBitsResult(div, bits_ops::UDiv(ResolveAsBits(div->operand(0)),
                                             ResolveAsBits(div->operand(1))));
  }

  absl::Status HandleSDiv(BinOp* div) override {
    return SetBitsResult(div, bits_ops::SDiv(ResolveAsBits(div->operand(0)),
                                             ResolveAsBits(div->operand(1))));
  }

  absl::Status HandleEq(CompareOp* eq) override {
    XLS_RETURN_IF_ERROR(VerifyAllBitsTypes(eq));
    return SetUint64Result(
        eq, ResolveAsBits(eq->operand(0)) == ResolveAsBits(eq->operand(1)));
  }

  absl::Status HandleSGe(CompareOp* ge) override {
    return SetUint64Result(
        ge, bits_ops::SGreaterThanOrEqual(ResolveAsBits(ge->operand(0)),
                                          ResolveAsBits(ge->operand(1))));
  }

  absl::Status HandleSGt(CompareOp* gt) override {
    return SetUint64Result(
        gt, bits_ops::SGreaterThan(ResolveAsBits(gt->operand(0)),
                                   ResolveAsBits(gt->operand(1))));
  }

  absl::Status HandleUGe(CompareOp* ge) override {
    return SetUint64Result(
        ge, bits_ops::UGreaterThanOrEqual(ResolveAsBits(ge->operand(0)),
                                          ResolveAsBits(ge->operand(1))));
  }

  absl::Status HandleUGt(CompareOp* gt) override {
    return SetUint64Result(
        gt, bits_ops::UGreaterThan(ResolveAsBits(gt->operand(0)),
                                   ResolveAsBits(gt->operand(1))));
  }

  absl::Status HandleIdentity(UnOp* identity) override {
    return SetValueResult(identity, ResolveAsValue(identity->operand(0)));
  }

  absl::Status HandleArrayIndex(ArrayIndex* index) override {
    const Value& input_array = ResolveAsValue(index->operand(0));
    // Out-of-bounds accesses are clamped to the highest index.
    // TODO(meheff): Figure out what the right thing to do here is including
    // potentially making the behavior  an option.
    uint64 i =
        ResolveAsBoundedUint64(index->operand(1), input_array.size() - 1);
    return SetValueResult(index, input_array.elements().at(i));
  }

  absl::Status HandleArrayUpdate(ArrayUpdate* update) override {
    XLS_ASSIGN_OR_RETURN(std::vector<Value> array_elements,
                         ResolveAsValue(update->operand(0)).GetElements());
    uint64 index =
        ResolveAsBoundedUint64(update->operand(1), array_elements.size());
    const Value& update_value = ResolveAsValue(update->operand(2));
    // Out-of-bounds accesses have no effect.
    if (index < array_elements.size()) {
      array_elements[index] = update_value;
    }
    XLS_ASSIGN_OR_RETURN(Value result, Value::Array(array_elements));
    return SetValueResult(update, result);
  }

  absl::Status HandleInvoke(Invoke* invoke) override {
    Function* to_apply = invoke->to_apply();
    std::vector<Value> args;
    for (int64 i = 0; i < to_apply->params().size(); ++i) {
      args.push_back(ResolveAsValue(invoke->operand(i)));
    }
    XLS_ASSIGN_OR_RETURN(Value result, Run(to_apply, args, stats_));
    return SetValueResult(invoke, result);
  }

  absl::Status HandleLiteral(Literal* literal) override {
    return SetValueResult(literal, literal->value());
  }

  absl::Status HandleULe(CompareOp* le) override {
    return SetUint64Result(
        le, bits_ops::ULessThanOrEqual(ResolveAsBits(le->operand(0)),
                                       ResolveAsBits(le->operand(1))));
  }

  absl::Status HandleSLt(CompareOp* lt) override {
    return SetUint64Result(lt,
                           bits_ops::SLessThan(ResolveAsBits(lt->operand(0)),
                                               ResolveAsBits(lt->operand(1))));
  }

  absl::Status HandleSLe(CompareOp* le) override {
    return SetUint64Result(
        le, bits_ops::SLessThanOrEqual(ResolveAsBits(le->operand(0)),
                                       ResolveAsBits(le->operand(1))));
  }

  absl::Status HandleULt(CompareOp* lt) override {
    return SetUint64Result(lt,
                           bits_ops::ULessThan(ResolveAsBits(lt->operand(0)),
                                               ResolveAsBits(lt->operand(1))));
  }

  absl::Status HandleMap(Map* map) override {
    Function* to_apply = map->to_apply();
    std::vector<Value> results;
    for (const Value& operand_element :
         ResolveAsValue(map->operand(0)).elements()) {
      XLS_ASSIGN_OR_RETURN(Value result,
                           Run(to_apply, {operand_element}, stats_));
      results.push_back(result);
    }
    XLS_ASSIGN_OR_RETURN(Value result_array, Value::Array(results));
    return SetValueResult(map, result_array);
  }

  absl::Status HandleSMul(ArithOp* mul) override {
    const int64 mul_width = mul->BitCountOrDie();
    Bits result = bits_ops::SMul(ResolveAsBits(mul->operand(0)),
                                 ResolveAsBits(mul->operand(1)));
    if (result.bit_count() > mul_width) {
      return SetBitsResult(mul, result.Slice(0, mul_width));
    } else if (result.bit_count() < mul_width) {
      return SetBitsResult(mul, bits_ops::SignExtend(result, mul_width));
    }
    return SetBitsResult(mul, result);
  }

  absl::Status HandleUMul(ArithOp* mul) override {
    const int64 mul_width = mul->BitCountOrDie();
    Bits result = bits_ops::UMul(ResolveAsBits(mul->operand(0)),
                                 ResolveAsBits(mul->operand(1)));
    if (result.bit_count() > mul_width) {
      return SetBitsResult(mul, result.Slice(0, mul_width));
    } else if (result.bit_count() < mul_width) {
      return SetBitsResult(mul, bits_ops::ZeroExtend(result, mul_width));
    }
    return SetBitsResult(mul, result);
  }

  absl::Status HandleNe(CompareOp* ne) override {
    XLS_RETURN_IF_ERROR(VerifyAllBitsTypes(ne));
    return SetUint64Result(
        ne, ResolveAsBits(ne->operand(0)) != ResolveAsBits(ne->operand(1)));
  }

  absl::Status HandleNeg(UnOp* neg) override {
    return SetBitsResult(neg, bits_ops::Negate(ResolveAsBits(neg->operand(0))));
  }

  absl::Status HandleNot(UnOp* not_op) override {
    return SetBitsResult(not_op,
                         bits_ops::Not(ResolveAsBits(not_op->operand(0))));
  }

  absl::Status HandleOneHot(OneHot* one_hot) override {
    int64 output_width = one_hot->BitCountOrDie();
    const Bits& input = ResolveAsBits(one_hot->operand(0));
    const int64 input_width = input.bit_count();
    for (int64 i = 0; i < input.bit_count(); ++i) {
      int64 index =
          one_hot->priority() == LsbOrMsb::kLsb ? i : input_width - i - 1;
      if (ResolveAsBits(one_hot->operand(0)).Get(index)) {
        auto one = UBits(1, /*bit_count=*/output_width);
        return SetBitsResult(one_hot, bits_ops::ShiftLeftLogical(one, index));
      }
    }
    // No bits of the operand are set so assert the msb of the output indicating
    // the default value.
    return SetBitsResult(
        one_hot,
        bits_ops::ShiftLeftLogical(UBits(1, output_width), output_width - 1));
  }

  absl::Status HandleOneHotSel(OneHotSelect* sel) override {
    const Bits& selector = ResolveAsBits(sel->selector());
    std::vector<const Value*> activated_inputs;
    for (int64 i = 0; i < selector.bit_count(); ++i) {
      if (selector.Get(i)) {
        activated_inputs.push_back(&ResolveAsValue(sel->cases()[i]));
      }
    }
    XLS_ASSIGN_OR_RETURN(Value result,
                         DeepOr(sel->GetType(), activated_inputs));
    return SetValueResult(sel, result);
  }

  absl::Status HandleParam(Param* param) override {
    XLS_ASSIGN_OR_RETURN(int64 index, param->function()->GetParamIndex(param));
    if (index >= args_.size()) {
      return absl::InternalError(absl::StrFormat(
          "Parameter %s at index %d does not exist in args (of length %d)",
          param->ToString(), index, args_.size()));
    }
    return SetValueResult(param, args_[index]);
  }

  absl::Status HandleReverse(UnOp* reverse) override {
    return SetBitsResult(reverse,
                         bits_ops::Reverse(ResolveAsBits(reverse->operand(0))));
  }

  absl::Status HandleSel(Select* sel) override {
    Bits selector = ResolveAsBits(sel->selector());
    if (bits_ops::UGreaterThan(
            selector, UBits(sel->cases().size() - 1, selector.bit_count()))) {
      XLS_RET_CHECK(sel->default_value().has_value());
      return SetValueResult(sel, ResolveAsValue(*sel->default_value()));
    }
    XLS_ASSIGN_OR_RETURN(uint64 i, selector.ToUint64());
    return SetValueResult(sel, ResolveAsValue(sel->cases()[i]));
  }

  static int64 GetBitCountOrDie(Node* n) { return n->BitCountOrDie(); }

  absl::Status HandleShll(BinOp* shll) override {
    const Bits& input = ResolveAsBits(shll->operand(0));
    const int64 shift_amt =
        ResolveAsBoundedUint64(shll->operand(1), input.bit_count());
    return SetBitsResult(shll, bits_ops::ShiftLeftLogical(input, shift_amt));
  }

  absl::Status HandleShra(BinOp* shra) override {
    const Bits& input = ResolveAsBits(shra->operand(0));
    const int64 shift_amt =
        ResolveAsBoundedUint64(shra->operand(1), input.bit_count());
    return SetBitsResult(shra, bits_ops::ShiftRightArith(input, shift_amt));
  }

  absl::Status HandleShrl(BinOp* shrl) override {
    const Bits& input = ResolveAsBits(shrl->operand(0));
    const int64 shift_amt =
        ResolveAsBoundedUint64(shrl->operand(1), input.bit_count());
    return SetBitsResult(shrl, bits_ops::ShiftRightLogical(input, shift_amt));
  }

  absl::Status HandleSignExtend(ExtendOp* sign_ext) override {
    return SetBitsResult(
        sign_ext, bits_ops::SignExtend(ResolveAsBits(sign_ext->operand(0)),
                                       sign_ext->new_bit_count()));
  }

  absl::Status HandleSub(BinOp* sub) override {
    return SetBitsResult(sub, bits_ops::Sub(ResolveAsBits(sub->operand(0)),
                                            ResolveAsBits(sub->operand(1))));
  }

  absl::Status HandleTuple(Tuple* tuple) override {
    std::vector<Value> tuple_values;
    for (Node* operand : tuple->operands()) {
      tuple_values.push_back(ResolveAsValue(operand));
    }
    return SetValueResult(tuple, Value::Tuple(tuple_values));
  }

  absl::Status HandleTupleIndex(TupleIndex* index) override {
    int64 tuple_index = index->As<TupleIndex>()->index();
    return SetValueResult(
        index, ResolveAsValue(index->operand(0)).elements().at(tuple_index));
  }

  absl::Status HandleZeroExtend(ExtendOp* zero_ext) override {
    return SetBitsResult(
        zero_ext, bits_ops::ZeroExtend(ResolveAsBits(zero_ext->operand(0)),
                                       zero_ext->new_bit_count()));
  }

 private:
  InterpreterVisitor(absl::Span<const Value> args, InterpreterStats* stats)
      : stats_(stats), args_(args) {}

  // Verifies that the width of the given node and all of its operands are less
  // than or equal to 64 bits. Also returns an error if an operand or the node
  // are not Bits type.
  absl::Status VerifyFitsIn64Bits(Node* node) {
    XLS_RETURN_IF_ERROR(VerifyAllBitsTypes(node));
    for (Node* operand : node->operands()) {
      if (operand->BitCountOrDie() > 64) {
        return absl::UnimplementedError(
            absl::StrFormat("Interpreter does not support operation '%s' with "
                            "operands of more than 64-bits: %s",
                            OpToString(node->op()), node->ToString()));
      }
    }
    if (node->BitCountOrDie() > 64) {
      return absl::UnimplementedError(absl::StrFormat(
          "Interpreter does not support operation '%s' with more "
          "than 64-bits: %s",
          OpToString(node->op()), node->ToString()));
    }
    return absl::OkStatus();
  }

  // Returns an error if the given node or any of its operands are not Bits
  // types.
  absl::Status VerifyAllBitsTypes(Node* node) {
    for (Node* operand : node->operands()) {
      if (!operand->GetType()->IsBits()) {
        return absl::UnimplementedError(
            absl::StrFormat("Interpreter does not support operation '%s' with "
                            "non-bits type operand",
                            OpToString(node->op())));
      }
    }
    if (!node->GetType()->IsBits()) {
      return absl::UnimplementedError(
          absl::StrFormat("Interpreter does not support operation '%s' with "
                          "non-bits type operand",
                          OpToString(node->op())));
    }
    return absl::OkStatus();
  }

  // Returns the previously evaluated value of 'node' as a bits type. CHECK
  // fails if it is not bits.
  const Bits& ResolveAsBits(Node* node) { return node_values_.at(node).bits(); }

  std::vector<Bits> ResolveAsBits(absl::Span<Node* const> nodes) {
    std::vector<Bits> results;
    for (Node* node : nodes) {
      results.push_back(ResolveAsBits(node));
    }
    return results;
  }

  // Returns the previously evaluated value of 'node' as a uint64. If the value
  // is greater than 'upper_limit' then 'upper_limit' is returned.
  uint64 ResolveAsBoundedUint64(Node* node, uint64 upper_limit) {
    const Bits& bits = ResolveAsBits(node);
    if (Bits::MinBitCountUnsigned(upper_limit) <= bits.bit_count() &&
        bits_ops::UGreaterThan(bits, UBits(upper_limit, bits.bit_count()))) {
      return upper_limit;
    }
    // Necessarily the bits value fits in a uint64 so the value() call is safe.
    return bits.ToUint64().value();
  }

  // Returns the previously evaluated value of 'node' as a Value.
  const Value& ResolveAsValue(Node* node) { return node_values_.at(node); }

  // Sets the evaluated value for 'node' to the given uint64 value. Returns an
  // error if 'node' is not a bits type or the result does not fit in the type.
  absl::Status SetUint64Result(Node* node, uint64 result) {
    XLS_RET_CHECK(node->GetType()->IsBits());
    XLS_RET_CHECK_GE(node->BitCountOrDie(), Bits::MinBitCountUnsigned(result));
    return SetValueResult(node, Value(UBits(result, node->BitCountOrDie())));
  }

  // Sets the evaluated value for 'node' to the given bits value. Returns an
  // error if 'node' is not a bits type.
  absl::Status SetBitsResult(Node* node, const Bits& result) {
    XLS_RET_CHECK(node->GetType()->IsBits());
    XLS_RET_CHECK_EQ(node->BitCountOrDie(), result.bit_count());
    if (stats_ != nullptr) {
      stats_->NoteNodeBits(node->ToString(), result);
    }
    return SetValueResult(node, Value(result));
  }

  // Sets the evaluated value for 'node' to the given Value. 'value' must be
  // passed in by value (ha!) because a use case is passing in a previously
  // evaluated value and inserting a into flat_hash_map (done below) invalidates
  // all references to Values in the map.
  absl::Status SetValueResult(Node* node, Value result) {
    XLS_VLOG(4) << absl::StreamFormat("%s operands:", node->GetName());
    for (int64 i = 0; i < node->operand_count(); ++i) {
      XLS_VLOG(4) << absl::StreamFormat(
          "  operand %d (%s): %s", i, node->operand(i)->GetName(),
          ResolveAsValue(node->operand(i)).ToString());
    }
    XLS_VLOG(3) << absl::StreamFormat("Result of %s: %s", node->ToString(),
                                      result.ToString());
    XLS_RET_CHECK(!node_values_.contains(node));
    node_values_[node] = result;
    return absl::OkStatus();
  }

  // Performs a logical OR of the given inputs. If 'inputs' is a not a Bits type
  // (ie, tuple or array) the element a recursively traversed and the Bits-typed
  // leaves are OR-ed.
  xabsl::StatusOr<Value> DeepOr(Type* input_type,
                                absl::Span<const Value* const> inputs) {
    if (input_type->IsBits()) {
      Bits result(input_type->AsBitsOrDie()->bit_count());
      for (const Value* input : inputs) {
        result = bits_ops::Or(result, input->bits());
      }
      return Value(result);
    }

    auto input_elements = [&](int64 i) {
      std::vector<const Value*> values;
      for (int64 j = 0; j < inputs.size(); ++j) {
        values.push_back(&inputs[j]->elements()[i]);
      }
      return values;
    };

    if (input_type->IsArray()) {
      Type* element_type = input_type->AsArrayOrDie()->element_type();
      std::vector<Value> elements;
      for (int64 i = 0; i < input_type->AsArrayOrDie()->size(); ++i) {
        XLS_ASSIGN_OR_RETURN(Value element,
                             DeepOr(element_type, input_elements(i)));
        elements.push_back(element);
      }
      return Value::Array(elements);
    }

    XLS_RET_CHECK(input_type->IsTuple());
    std::vector<Value> elements;
    for (int64 i = 0; i < input_type->AsTupleOrDie()->size(); ++i) {
      XLS_ASSIGN_OR_RETURN(Value element,
                           DeepOr(input_type->AsTupleOrDie()->element_type(i),
                                  input_elements(i)));
      elements.push_back(element);
    }
    return Value::Tuple(elements);
  }

  // Statistics on interpreter execution. May be nullptr.
  InterpreterStats* stats_;

  // The arguments to the Function being evaluated indexed by parameter name.
  absl::Span<const Value> args_;

  // The evaluated values for the nodes in the Function.
  absl::flat_hash_map<Node*, Value> node_values_;
};

xabsl::StatusOr<Value> Run(Function* function, absl::Span<const Value> args,
                           InterpreterStats* stats) {
  XLS_VLOG(3) << "Function:";
  XLS_VLOG_LINES(3, function->DumpIr());
  XLS_ASSIGN_OR_RETURN(Value result,
                       InterpreterVisitor::Run(function, args, stats));
  XLS_VLOG(2) << "Result = " << result;
  return result;
}

xabsl::StatusOr<Value> RunKwargs(
    Function* function, const absl::flat_hash_map<std::string, Value>& args,
    InterpreterStats* stats) {
  XLS_VLOG(2) << "Interpreting function " << function->name()
              << " with arguments:";
  XLS_ASSIGN_OR_RETURN(std::vector<Value> positional_args,
                       KeywordArgsToPositional(*function, args));
  return Run(function, positional_args, stats);
}

xabsl::StatusOr<Value> EvaluateNodeWithLiteralOperands(Node* node) {
  XLS_RET_CHECK_GT(node->operand_count(), 0);
  XLS_RET_CHECK(std::all_of(node->operands().begin(), node->operands().end(),
                            [](Node* o) { return o->Is<Literal>(); }));
  return InterpreterVisitor::EvaluateNodeWithLiteralOperands(node);
}

xabsl::StatusOr<Value> EvaluateNode(
    Node* node, absl::Span<const Value* const> operand_values) {
  return InterpreterVisitor::EvaluateNode(node, operand_values);
}

}  // namespace ir_interpreter
}  // namespace xls
