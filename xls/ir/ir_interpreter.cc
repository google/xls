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

/* static */ xabsl::StatusOr<Value> IrInterpreter::Run(
    Function* function, absl::Span<const Value> args, InterpreterStats* stats) {
  XLS_VLOG(3) << "Interpreting function " << function->name();
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
  IrInterpreter visitor(args, stats);
  XLS_RETURN_IF_ERROR(function->return_value()->Accept(&visitor));
  Value result = visitor.ResolveAsValue(function->return_value());
  XLS_VLOG(2) << "Result = " << result;
  return std::move(result);
}

/* static */
xabsl::StatusOr<Value> IrInterpreter::RunKwargs(
    Function* function, const absl::flat_hash_map<std::string, Value>& args,
    InterpreterStats* stats) {
  XLS_VLOG(2) << "Interpreting function " << function->name()
              << " with arguments:";
  XLS_ASSIGN_OR_RETURN(std::vector<Value> positional_args,
                       KeywordArgsToPositional(*function, args));
  return Run(function, positional_args, stats);
}

/* static */ xabsl::StatusOr<Value>
IrInterpreter::EvaluateNodeWithLiteralOperands(Node* node) {
  XLS_RET_CHECK(std::all_of(node->operands().begin(), node->operands().end(),
                            [](Node* n) { return n->Is<Literal>(); }));
  IrInterpreter visitor({}, /*stats=*/nullptr);
  XLS_RETURN_IF_ERROR(node->Accept(&visitor));
  return visitor.ResolveAsValue(node);
}

/* static */ xabsl::StatusOr<Value> IrInterpreter::EvaluateNode(
    Node* node, absl::Span<const Value* const> operand_values) {
  XLS_RET_CHECK_EQ(node->operand_count(), operand_values.size());
  IrInterpreter visitor({}, /*stats=*/nullptr);
  for (int64 i = 0; i < operand_values.size(); ++i) {
    visitor.node_values_[node->operand(i)] = *operand_values[i];
  }
  XLS_RETURN_IF_ERROR(node->VisitSingleNode(&visitor));
  return visitor.ResolveAsValue(node);
}

absl::Status IrInterpreter::HandleAdd(BinOp* add) {
  return SetBitsResult(add, bits_ops::Add(ResolveAsBits(add->operand(0)),
                                          ResolveAsBits(add->operand(1))));
}

absl::Status IrInterpreter::HandleAndReduce(BitwiseReductionOp* and_reduce) {
  Bits operand = ResolveAsBits(and_reduce->operand(0));
  return SetBitsResult(and_reduce, bits_ops::AndReduce(operand));
}

absl::Status IrInterpreter::HandleOrReduce(BitwiseReductionOp* or_reduce) {
  Bits operand = ResolveAsBits(or_reduce->operand(0));
  return SetBitsResult(or_reduce, bits_ops::OrReduce(operand));
}

absl::Status IrInterpreter::HandleXorReduce(BitwiseReductionOp* xor_reduce) {
  Bits operand = ResolveAsBits(xor_reduce->operand(0));
  return SetBitsResult(xor_reduce, bits_ops::XorReduce(operand));
}

absl::Status IrInterpreter::HandleNaryAnd(NaryOp* and_op) {
  std::vector<Bits> operands = ResolveAsBitsVector(and_op->operands());
  Bits accum = bits_ops::NaryAnd(operands);
  return SetBitsResult(and_op, accum);
}

absl::Status IrInterpreter::HandleNaryNand(NaryOp* nand_op) {
  std::vector<Bits> operands = ResolveAsBitsVector(nand_op->operands());
  Bits accum = bits_ops::NaryNand(operands);
  return SetBitsResult(nand_op, accum);
}

absl::Status IrInterpreter::HandleNaryNor(NaryOp* nor_op) {
  std::vector<Bits> operands = ResolveAsBitsVector(nor_op->operands());
  Bits accum = bits_ops::NaryNor(operands);
  return SetBitsResult(nor_op, accum);
}

absl::Status IrInterpreter::HandleNaryOr(NaryOp* or_op) {
  std::vector<Bits> operands = ResolveAsBitsVector(or_op->operands());
  Bits accum = bits_ops::NaryOr(operands);
  return SetBitsResult(or_op, accum);
}

absl::Status IrInterpreter::HandleNaryXor(NaryOp* xor_op) {
  std::vector<Bits> operands = ResolveAsBitsVector(xor_op->operands());
  Bits accum = bits_ops::NaryXor(operands);
  return SetBitsResult(xor_op, accum);
}

absl::Status IrInterpreter::HandleAfterAll(AfterAll* after_all) {
  // AfterAll is only meaningful to the compiler and does not actually perform
  // any computation.
  return SetValueResult(after_all, Value::Token());
}

absl::Status IrInterpreter::HandleChannelReceive(ChannelReceive* receive) {
  return absl::UnimplementedError(
      "ChannelReceive not implemented in IrInterpreter");
}

absl::Status IrInterpreter::HandleChannelSend(ChannelSend* send) {
  return absl::UnimplementedError(
      "Channel send not implemented in IrInterpreter");
}

absl::Status IrInterpreter::HandleArray(Array* array) {
  std::vector<Value> operand_values;
  for (Node* operand : array->operands()) {
    operand_values.push_back(ResolveAsValue(operand));
  }
  XLS_ASSIGN_OR_RETURN(Value result, Value::Array(operand_values));
  return SetValueResult(array, result);
}

absl::Status IrInterpreter::HandleBitSlice(BitSlice* bit_slice) {
  return SetBitsResult(bit_slice,
                       ResolveAsBits(bit_slice->operand(0))
                           .Slice(bit_slice->start(), bit_slice->width()));
}

absl::Status IrInterpreter::HandleDynamicBitSlice(
    DynamicBitSlice* dynamic_bit_slice) {
  int64 operand_width = dynamic_bit_slice->operand(0)->BitCountOrDie();
  const Bits& start_bits = ResolveAsBits(dynamic_bit_slice->operand(1));
  if (bits_ops::UGreaterThanOrEqual(start_bits, operand_width)) {
    // Slice is entirely out-of-bounds. Return value should be all zero bits.
    return SetBitsResult(dynamic_bit_slice, Bits(dynamic_bit_slice->width()));
  }
  uint64 start = start_bits.ToUint64().value();
  const Bits& operand = ResolveAsBits(dynamic_bit_slice->operand(0));
  Bits shifted_value = bits_ops::ShiftRightLogical(operand, start);
  Bits truncated_value = shifted_value.Slice(0, dynamic_bit_slice->width());
  return SetBitsResult(dynamic_bit_slice, truncated_value);
}

absl::Status IrInterpreter::HandleConcat(Concat* concat) {
  std::vector<Bits> operand_values;
  for (Node* operand : concat->operands()) {
    operand_values.push_back(ResolveAsBits(operand));
  }
  return SetBitsResult(concat, bits_ops::Concat(operand_values));
}

absl::Status IrInterpreter::HandleCountedFor(CountedFor* counted_for) {
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

absl::Status IrInterpreter::HandleDecode(Decode* decode) {
  XLS_ASSIGN_OR_RETURN(int64 input_value,
                       ResolveAsBits(decode->operand(0)).ToUint64());
  if (input_value < decode->BitCountOrDie()) {
    return SetBitsResult(decode, Bits::PowerOfTwo(/*set_bit_index=*/input_value,
                                                  decode->BitCountOrDie()));
  } else {
    return SetBitsResult(decode, Bits(decode->BitCountOrDie()));
  }
}

absl::Status IrInterpreter::HandleEncode(Encode* encode) {
  const Bits& input = ResolveAsBits(encode->operand(0));
  Bits result(encode->BitCountOrDie());
  for (int64 i = 0; i < input.bit_count(); ++i) {
    if (input.Get(i)) {
      result = bits_ops::Or(result, UBits(i, encode->BitCountOrDie()));
    }
  }
  return SetBitsResult(encode, result);
}

absl::Status IrInterpreter::HandleUDiv(BinOp* div) {
  return SetBitsResult(div, bits_ops::UDiv(ResolveAsBits(div->operand(0)),
                                           ResolveAsBits(div->operand(1))));
}

absl::Status IrInterpreter::HandleSDiv(BinOp* div) {
  return SetBitsResult(div, bits_ops::SDiv(ResolveAsBits(div->operand(0)),
                                           ResolveAsBits(div->operand(1))));
}

absl::Status IrInterpreter::HandleEq(CompareOp* eq) {
  XLS_RETURN_IF_ERROR(VerifyAllBitsTypes(eq));
  return SetUint64Result(
      eq, ResolveAsBits(eq->operand(0)) == ResolveAsBits(eq->operand(1)));
}

absl::Status IrInterpreter::HandleSGe(CompareOp* ge) {
  return SetUint64Result(
      ge, bits_ops::SGreaterThanOrEqual(ResolveAsBits(ge->operand(0)),
                                        ResolveAsBits(ge->operand(1))));
}

absl::Status IrInterpreter::HandleSGt(CompareOp* gt) {
  return SetUint64Result(gt,
                         bits_ops::SGreaterThan(ResolveAsBits(gt->operand(0)),
                                                ResolveAsBits(gt->operand(1))));
}

absl::Status IrInterpreter::HandleUGe(CompareOp* ge) {
  return SetUint64Result(
      ge, bits_ops::UGreaterThanOrEqual(ResolveAsBits(ge->operand(0)),
                                        ResolveAsBits(ge->operand(1))));
}

absl::Status IrInterpreter::HandleUGt(CompareOp* gt) {
  return SetUint64Result(gt,
                         bits_ops::UGreaterThan(ResolveAsBits(gt->operand(0)),
                                                ResolveAsBits(gt->operand(1))));
}

absl::Status IrInterpreter::HandleIdentity(UnOp* identity) {
  return SetValueResult(identity, ResolveAsValue(identity->operand(0)));
}

absl::Status IrInterpreter::HandleArrayIndex(ArrayIndex* index) {
  const Value& input_array = ResolveAsValue(index->operand(0));
  // Out-of-bounds accesses are clamped to the highest index.
  // TODO(meheff): Figure out what the right thing to do here is including
  // potentially making the behavior  an option.
  uint64 i = ResolveAsBoundedUint64(index->operand(1), input_array.size() - 1);
  return SetValueResult(index, input_array.elements().at(i));
}

absl::Status IrInterpreter::HandleArrayUpdate(ArrayUpdate* update) {
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

absl::Status IrInterpreter::HandleArrayConcat(ArrayConcat* concat) {
  std::vector<Value> array_elements;

  for (Node* operand : concat->operands()) {
    const Value& operand_as_value = ResolveAsValue(operand);
    auto elements = operand_as_value.elements();

    array_elements.insert(array_elements.end(), elements.begin(),
                          elements.end());
  }

  XLS_ASSIGN_OR_RETURN(Value result, Value::Array(array_elements));
  return SetValueResult(concat, result);
}

absl::Status IrInterpreter::HandleInvoke(Invoke* invoke) {
  Function* to_apply = invoke->to_apply();
  std::vector<Value> args;
  for (int64 i = 0; i < to_apply->params().size(); ++i) {
    args.push_back(ResolveAsValue(invoke->operand(i)));
  }
  XLS_ASSIGN_OR_RETURN(Value result, Run(to_apply, args, stats_));
  return SetValueResult(invoke, result);
}

absl::Status IrInterpreter::HandleLiteral(Literal* literal) {
  return SetValueResult(literal, literal->value());
}

absl::Status IrInterpreter::HandleULe(CompareOp* le) {
  return SetUint64Result(
      le, bits_ops::ULessThanOrEqual(ResolveAsBits(le->operand(0)),
                                     ResolveAsBits(le->operand(1))));
}

absl::Status IrInterpreter::HandleSLt(CompareOp* lt) {
  return SetUint64Result(lt,
                         bits_ops::SLessThan(ResolveAsBits(lt->operand(0)),
                                             ResolveAsBits(lt->operand(1))));
}

absl::Status IrInterpreter::HandleSLe(CompareOp* le) {
  return SetUint64Result(
      le, bits_ops::SLessThanOrEqual(ResolveAsBits(le->operand(0)),
                                     ResolveAsBits(le->operand(1))));
}

absl::Status IrInterpreter::HandleULt(CompareOp* lt) {
  return SetUint64Result(lt,
                         bits_ops::ULessThan(ResolveAsBits(lt->operand(0)),
                                             ResolveAsBits(lt->operand(1))));
}

absl::Status IrInterpreter::HandleMap(Map* map) {
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

absl::Status IrInterpreter::HandleSMul(ArithOp* mul) {
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

absl::Status IrInterpreter::HandleUMul(ArithOp* mul) {
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

absl::Status IrInterpreter::HandleNe(CompareOp* ne) {
  XLS_RETURN_IF_ERROR(VerifyAllBitsTypes(ne));
  return SetUint64Result(
      ne, ResolveAsBits(ne->operand(0)) != ResolveAsBits(ne->operand(1)));
}

absl::Status IrInterpreter::HandleNeg(UnOp* neg) {
  return SetBitsResult(neg, bits_ops::Negate(ResolveAsBits(neg->operand(0))));
}

absl::Status IrInterpreter::HandleNot(UnOp* not_op) {
  return SetBitsResult(not_op,
                       bits_ops::Not(ResolveAsBits(not_op->operand(0))));
}

absl::Status IrInterpreter::HandleOneHot(OneHot* one_hot) {
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
  return SetBitsResult(one_hot, bits_ops::ShiftLeftLogical(
                                    UBits(1, output_width), output_width - 1));
}

absl::Status IrInterpreter::HandleOneHotSel(OneHotSelect* sel) {
  const Bits& selector = ResolveAsBits(sel->selector());
  std::vector<const Value*> activated_inputs;
  for (int64 i = 0; i < selector.bit_count(); ++i) {
    if (selector.Get(i)) {
      activated_inputs.push_back(&ResolveAsValue(sel->cases()[i]));
    }
  }
  XLS_ASSIGN_OR_RETURN(Value result, DeepOr(sel->GetType(), activated_inputs));
  return SetValueResult(sel, result);
}

absl::Status IrInterpreter::HandleParam(Param* param) {
  XLS_ASSIGN_OR_RETURN(int64 index, param->function()->GetParamIndex(param));
  if (index >= args_.size()) {
    return absl::InternalError(absl::StrFormat(
        "Parameter %s at index %d does not exist in args (of length %d)",
        param->ToString(), index, args_.size()));
  }
  return SetValueResult(param, args_[index]);
}

absl::Status IrInterpreter::HandleReverse(UnOp* reverse) {
  return SetBitsResult(reverse,
                       bits_ops::Reverse(ResolveAsBits(reverse->operand(0))));
}

absl::Status IrInterpreter::HandleSel(Select* sel) {
  Bits selector = ResolveAsBits(sel->selector());
  if (bits_ops::UGreaterThan(
          selector, UBits(sel->cases().size() - 1, selector.bit_count()))) {
    XLS_RET_CHECK(sel->default_value().has_value());
    return SetValueResult(sel, ResolveAsValue(*sel->default_value()));
  }
  XLS_ASSIGN_OR_RETURN(uint64 i, selector.ToUint64());
  return SetValueResult(sel, ResolveAsValue(sel->cases()[i]));
}

absl::Status IrInterpreter::HandleShll(BinOp* shll) {
  const Bits& input = ResolveAsBits(shll->operand(0));
  const int64 shift_amt =
      ResolveAsBoundedUint64(shll->operand(1), input.bit_count());
  return SetBitsResult(shll, bits_ops::ShiftLeftLogical(input, shift_amt));
}

absl::Status IrInterpreter::HandleShra(BinOp* shra) {
  const Bits& input = ResolveAsBits(shra->operand(0));
  const int64 shift_amt =
      ResolveAsBoundedUint64(shra->operand(1), input.bit_count());
  return SetBitsResult(shra, bits_ops::ShiftRightArith(input, shift_amt));
}

absl::Status IrInterpreter::HandleShrl(BinOp* shrl) {
  const Bits& input = ResolveAsBits(shrl->operand(0));
  const int64 shift_amt =
      ResolveAsBoundedUint64(shrl->operand(1), input.bit_count());
  return SetBitsResult(shrl, bits_ops::ShiftRightLogical(input, shift_amt));
}

absl::Status IrInterpreter::HandleSignExtend(ExtendOp* sign_ext) {
  return SetBitsResult(sign_ext,
                       bits_ops::SignExtend(ResolveAsBits(sign_ext->operand(0)),
                                            sign_ext->new_bit_count()));
}

absl::Status IrInterpreter::HandleSub(BinOp* sub) {
  return SetBitsResult(sub, bits_ops::Sub(ResolveAsBits(sub->operand(0)),
                                          ResolveAsBits(sub->operand(1))));
}

absl::Status IrInterpreter::HandleTuple(Tuple* tuple) {
  std::vector<Value> tuple_values;
  for (Node* operand : tuple->operands()) {
    tuple_values.push_back(ResolveAsValue(operand));
  }
  return SetValueResult(tuple, Value::Tuple(tuple_values));
}

absl::Status IrInterpreter::HandleTupleIndex(TupleIndex* index) {
  int64 tuple_index = index->As<TupleIndex>()->index();
  return SetValueResult(
      index, ResolveAsValue(index->operand(0)).elements().at(tuple_index));
}

absl::Status IrInterpreter::HandleZeroExtend(ExtendOp* zero_ext) {
  return SetBitsResult(zero_ext,
                       bits_ops::ZeroExtend(ResolveAsBits(zero_ext->operand(0)),
                                            zero_ext->new_bit_count()));
}

absl::Status IrInterpreter::VerifyAllBitsTypes(Node* node) {
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

const Bits& IrInterpreter::ResolveAsBits(Node* node) {
  return node_values_.at(node).bits();
}

std::vector<Bits> IrInterpreter::ResolveAsBitsVector(
    absl::Span<Node* const> nodes) {
  std::vector<Bits> results;
  for (Node* node : nodes) {
    results.push_back(ResolveAsBits(node));
  }
  return results;
}

uint64 IrInterpreter::ResolveAsBoundedUint64(Node* node, uint64 upper_limit) {
  const Bits& bits = ResolveAsBits(node);
  if (Bits::MinBitCountUnsigned(upper_limit) <= bits.bit_count() &&
      bits_ops::UGreaterThan(bits, UBits(upper_limit, bits.bit_count()))) {
    return upper_limit;
  }
  // Necessarily the bits value fits in a uint64 so the value() call is safe.
  return bits.ToUint64().value();
}

absl::Status IrInterpreter::SetUint64Result(Node* node, uint64 result) {
  XLS_RET_CHECK(node->GetType()->IsBits());
  XLS_RET_CHECK_GE(node->BitCountOrDie(), Bits::MinBitCountUnsigned(result));
  return SetValueResult(node, Value(UBits(result, node->BitCountOrDie())));
}

absl::Status IrInterpreter::SetBitsResult(Node* node, const Bits& result) {
  XLS_RET_CHECK(node->GetType()->IsBits());
  XLS_RET_CHECK_EQ(node->BitCountOrDie(), result.bit_count());
  if (stats_ != nullptr) {
    stats_->NoteNodeBits(node->ToString(), result);
  }
  return SetValueResult(node, Value(result));
}

absl::Status IrInterpreter::SetValueResult(Node* node, Value result) {
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

xabsl::StatusOr<Value> IrInterpreter::DeepOr(
    Type* input_type, absl::Span<const Value* const> inputs) {
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
    XLS_ASSIGN_OR_RETURN(
        Value element,
        DeepOr(input_type->AsTupleOrDie()->element_type(i), input_elements(i)));
    elements.push_back(element);
  }
  return Value::Tuple(elements);
}

}  // namespace xls
