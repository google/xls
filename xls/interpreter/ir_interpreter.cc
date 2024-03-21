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

#include "xls/interpreter/ir_interpreter.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/function_interpreter.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/events.h"
#include "xls/ir/function.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"

namespace xls {

namespace {

// Returns the given bits value as a uint64_t value. If the value exceeds
// upper_limit, then upper_limit is returned.
uint64_t BitsToBoundedUint64(const Bits& bits, uint64_t upper_limit) {
  if (Bits::MinBitCountUnsigned(upper_limit) <= bits.bit_count() &&
      bits_ops::UGreaterThan(bits, UBits(upper_limit, bits.bit_count()))) {
    return upper_limit;
  }
  // Necessarily the bits value fits in a uint64_t so the value() call is safe.
  return bits.ToUint64().value();
}

}  // namespace

absl::StatusOr<Value> InterpretNode(Node* node,
                                    absl::Span<const Value> operand_values) {
  // Gate nodes do not require side effects when interpreted.
  if (OpIsSideEffecting(node->op()) && node->op() != Op::kGate) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Cannot interpret side-effecting op %s in node %s "
                        "outside of an interpreter.",
                        OpToString(node->op()), node->ToString()));
  }

  XLS_RET_CHECK_EQ(node->operand_count(), operand_values.size());
  IrInterpreter visitor;
  for (int64_t i = 0; i < operand_values.size(); ++i) {
    // Operands may be duplicated so check to see if the operand value has
    // already been set.
    if (!visitor.HasResult(node->operand(i))) {
      XLS_RETURN_IF_ERROR(
          visitor.SetValueResult(node->operand(i), operand_values[i]));
    }
  }
  XLS_RETURN_IF_ERROR(node->VisitSingleNode(&visitor));
  return visitor.ResolveAsValue(node);
}

absl::Status IrInterpreter::AddInterpreterEvents(
    const InterpreterEvents& events) {
  for (const TraceMessage& trace_msg : events.trace_msgs) {
    GetInterpreterEvents().trace_msgs.push_back(trace_msg);
  }

  for (const std::string& assert_msg : events.assert_msgs) {
    GetInterpreterEvents().assert_msgs.push_back(assert_msg);
  }

  return absl::OkStatus();
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

absl::Status IrInterpreter::HandleMinDelay(MinDelay* min_delay) {
  // MinDelay is only meaningful to the compiler and does not actually perform
  // any computation.
  return SetValueResult(min_delay, Value::Token());
}

absl::Status IrInterpreter::HandleReceive(Receive* receive) {
  return absl::UnimplementedError("Receive not implemented in IrInterpreter");
}

absl::Status IrInterpreter::HandleRegisterRead(RegisterRead* reg_read) {
  return absl::UnimplementedError(
      "RegisterRead not implemented in IrInterpreter");
}

absl::Status IrInterpreter::HandleRegisterWrite(RegisterWrite* reg_write) {
  return absl::UnimplementedError(
      "RegisterWrite not implemented in IrInterpreter");
}

absl::Status IrInterpreter::HandleSend(Send* send) {
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

absl::Status IrInterpreter::HandleInputPort(InputPort* input_port) {
  return absl::UnimplementedError("InputPort not implemented in IrInterpreter");
}

absl::Status IrInterpreter::HandleOutputPort(OutputPort* output_port) {
  return absl::UnimplementedError(
      "OutputPort not implemented in IrInterpreter");
}

absl::Status IrInterpreter::HandleGate(Gate* gate) {
  const Bits& condition = ResolveAsBits(gate->condition());
  if (condition.IsOne()) {
    return SetValueResult(gate, ResolveAsValue(gate->data()));
  }
  return SetValueResult(gate, ZeroOfType(gate->GetType()));
}

absl::Status IrInterpreter::HandleBitSlice(BitSlice* bit_slice) {
  return SetBitsResult(bit_slice,
                       ResolveAsBits(bit_slice->operand(0))
                           .Slice(bit_slice->start(), bit_slice->width()));
}

absl::Status IrInterpreter::HandleBitSliceUpdate(BitSliceUpdate* update) {
  const Bits& to_update = ResolveAsBits(update->to_update());
  const Bits& start = ResolveAsBits(update->start());
  const Bits& update_value = ResolveAsBits(update->update_value());
  if (bits_ops::UGreaterThanOrEqual(start, to_update.bit_count())) {
    // Start index is entirely out-of-bounds. The return value is simply the
    // input data operand.
    return SetBitsResult(update, to_update);
  }

  // Safe to convert start to uint64_t because of the above check that start is
  // in-bounds.
  int64_t start_index = start.ToUint64().value();

  return SetBitsResult(
      update, bits_ops::BitSliceUpdate(to_update, start_index, update_value));
}

absl::Status IrInterpreter::HandleDynamicBitSlice(
    DynamicBitSlice* dynamic_bit_slice) {
  int64_t operand_width = dynamic_bit_slice->operand(0)->BitCountOrDie();
  const Bits& start_bits = ResolveAsBits(dynamic_bit_slice->operand(1));
  if (bits_ops::UGreaterThanOrEqual(start_bits, operand_width)) {
    // Slice is entirely out-of-bounds. Return value should be all zero bits.
    return SetBitsResult(dynamic_bit_slice, Bits(dynamic_bit_slice->width()));
  }
  uint64_t start = start_bits.ToUint64().value();
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
  for (int64_t i = 1; i < counted_for->operand_count(); ++i) {
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
  for (int64_t i = 0, iv = 0; i < counted_for->trip_count();
       ++i, iv += counted_for->stride()) {
    std::vector<Value> args_for_body = {
        Value(UBits(iv, arg0_type->bit_count())), loop_state};
    for (const auto& value : invariant_args) {
      args_for_body.push_back(value);
    }
    XLS_ASSIGN_OR_RETURN(InterpreterResult<Value> loop_result,
                         InterpretFunction(body, args_for_body));
    XLS_RETURN_IF_ERROR(AddInterpreterEvents(loop_result.events));
    loop_state = loop_result.value;
  }
  return SetValueResult(counted_for, loop_state);
}

absl::Status IrInterpreter::HandleDecode(Decode* decode) {
  Bits index_bits = ResolveAsBits(decode->operand(0));
  if (bits_ops::ULessThan(index_bits, decode->BitCountOrDie())) {
    XLS_ASSIGN_OR_RETURN(int64_t index, index_bits.ToUint64());
    return SetBitsResult(decode, Bits::PowerOfTwo(/*set_bit_index=*/index,
                                                  decode->BitCountOrDie()));
  }
  return SetBitsResult(decode, Bits(decode->BitCountOrDie()));
}

absl::Status IrInterpreter::HandleDynamicCountedFor(
    DynamicCountedFor* dynamic_counted_for) {
  Function* body = dynamic_counted_for->body();

  // Set the loop invariant args (from the fourth operand on up).
  std::vector<Value> invariant_args;
  for (int64_t i = 3; i < dynamic_counted_for->operand_count(); ++i) {
    invariant_args.push_back(ResolveAsValue(dynamic_counted_for->operand(i)));
  }

  // Grab initial accumulator value, trip count, and stride.
  Value loop_state = ResolveAsValue(dynamic_counted_for->operand(0));
  BitsType* index_type = body->param(0)->GetType()->AsBitsOrDie();
  const Bits& trip_count_unsigned =
      ResolveAsBits(dynamic_counted_for->operand(1));
  Bits trip_count = bits_ops::ZeroExtend(trip_count_unsigned,
                                         trip_count_unsigned.bit_count() + 1);
  const Bits& stride = ResolveAsBits(dynamic_counted_for->operand(2));

  // Setup index, extend stride for addition with index.
  Bits index_limit = bits_ops::SMul(trip_count, stride);
  Bits index(index_type->bit_count());
  XLS_RET_CHECK(index.IsZero());
  const Bits& extended_stride = bits_ops::SignExtend(stride, index.bit_count());

  // For each iteration of dynamic_counted_for, update the induction variable
  // and loop state arguments (params 0 and 1) and recursively call the
  // interpreter Run() on the body function -- the new accumulator value is the
  // return value of interpreting body.
  while (!bits_ops::SEqual(index, index_limit)) {
    std::vector<Value> args_for_body = {Value(index), loop_state};
    for (const auto& value : invariant_args) {
      args_for_body.push_back(value);
    }
    XLS_ASSIGN_OR_RETURN(InterpreterResult<Value> loop_result,
                         InterpretFunction(body, args_for_body));
    XLS_RETURN_IF_ERROR(AddInterpreterEvents(loop_result.events));
    loop_state = loop_result.value;
    index = bits_ops::Add(index, extended_stride);
  }

  return SetValueResult(dynamic_counted_for, loop_state);
}

absl::Status IrInterpreter::HandleEncode(Encode* encode) {
  const Bits& input = ResolveAsBits(encode->operand(0));
  Bits result(encode->BitCountOrDie());
  for (int64_t i = 0; i < input.bit_count(); ++i) {
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

absl::Status IrInterpreter::HandleUMod(BinOp* mod) {
  return SetBitsResult(mod, bits_ops::UMod(ResolveAsBits(mod->operand(0)),
                                           ResolveAsBits(mod->operand(1))));
}

absl::Status IrInterpreter::HandleSMod(BinOp* mod) {
  return SetBitsResult(mod, bits_ops::SMod(ResolveAsBits(mod->operand(0)),
                                           ResolveAsBits(mod->operand(1))));
}

absl::Status IrInterpreter::HandleEq(CompareOp* eq) {
  return SetUint64Result(eq,
                         static_cast<int64_t>(ResolveAsValue(eq->operand(0)) ==
                                              ResolveAsValue(eq->operand(1))));
}

absl::Status IrInterpreter::HandleSGe(CompareOp* ge) {
  return SetUint64Result(
      ge, static_cast<uint64_t>(bits_ops::SGreaterThanOrEqual(
              ResolveAsBits(ge->operand(0)), ResolveAsBits(ge->operand(1)))));
}

absl::Status IrInterpreter::HandleSGt(CompareOp* gt) {
  return SetUint64Result(
      gt, static_cast<uint64_t>(bits_ops::SGreaterThan(
              ResolveAsBits(gt->operand(0)), ResolveAsBits(gt->operand(1)))));
}

absl::Status IrInterpreter::HandleUGe(CompareOp* ge) {
  return SetUint64Result(
      ge, static_cast<uint64_t>(bits_ops::UGreaterThanOrEqual(
              ResolveAsBits(ge->operand(0)), ResolveAsBits(ge->operand(1)))));
}

absl::Status IrInterpreter::HandleUGt(CompareOp* gt) {
  return SetUint64Result(
      gt, static_cast<uint64_t>(bits_ops::UGreaterThan(
              ResolveAsBits(gt->operand(0)), ResolveAsBits(gt->operand(1)))));
}

absl::Status IrInterpreter::HandleIdentity(UnOp* identity) {
  return SetValueResult(identity, ResolveAsValue(identity->operand(0)));
}

// Recursive function for setting an element of a multidimensional array to a
// particular value. 'indices' is a multidimensional array index of type tuple
// of bits. 'value' is what to assign at the array element at the particular
// index. 'elements' is a vector of the outer-most elements of the array being
// indexed into.
static absl::Status SetArrayElement(absl::Span<const Bits> indices,
                                    const Value& value,
                                    std::vector<Value>* elements) {
  XLS_RET_CHECK(!indices.empty());
  uint64_t index = BitsToBoundedUint64(indices.front(), elements->size());
  if (index >= elements->size()) {
    // Out-of-bounds access it a no-op.
    return absl::OkStatus();
  }
  if (indices.size() == 1) {
    (*elements)[index] = value;
    return absl::OkStatus();
  }

  // Index has multiple element. Peel off the first index and recurse into that
  // element.
  const Value& selected_element = (*elements)[index];
  std::vector<Value> subelements(selected_element.elements().begin(),
                                 selected_element.elements().end());
  XLS_RETURN_IF_ERROR(SetArrayElement(indices.subspan(1), value, &subelements));
  // Reconstruct the affected element as an array and assign it to the indexed
  // slot.
  XLS_ASSIGN_OR_RETURN(Value array_element, Value::Array(subelements));
  (*elements)[index] = array_element;
  return absl::OkStatus();
}

absl::Status IrInterpreter::HandleArrayIndex(ArrayIndex* index) {
  const Value* array = &ResolveAsValue(index->array());
  for (Node* index_operand : index->indices()) {
    uint64_t idx =
        BitsToBoundedUint64(ResolveAsBits(index_operand), array->size() - 1);
    array = &array->element(idx);
  }
  return SetValueResult(index, *array);
}

absl::Status IrInterpreter::HandleArraySlice(ArraySlice* slice) {
  const Value& array = ResolveAsValue(slice->array());
  uint64_t start = ResolveAsBoundedUint64(slice->start(), array.size() - 1);
  std::vector<Value> sliced;
  sliced.reserve(slice->width());
  for (int64_t i = start; i < start + slice->width(); i++) {
    if (i >= array.elements().size()) {
      sliced.push_back(array.elements().back());
    } else {
      sliced.push_back(array.elements()[i]);
    }
  }
  XLS_ASSIGN_OR_RETURN(Value result, Value::Array(sliced));
  return SetValueResult(slice, result);
}

absl::Status IrInterpreter::HandleArrayUpdate(ArrayUpdate* update) {
  const Value& input_array = ResolveAsValue(update->array_to_update());
  const Value& update_value = ResolveAsValue(update->update_value());

  if (update->indices().empty()) {
    // Index is empty. The *entire* array is replaced with the update value.
    return SetValueResult(update, update_value);
  }

  XLS_ASSIGN_OR_RETURN(std::vector<Value> array_elements,
                       input_array.GetElements());
  std::vector<Bits> index_vector;
  for (Node* index_operand : update->indices()) {
    index_vector.push_back(ResolveAsBits(index_operand));
  }
  XLS_RETURN_IF_ERROR(
      SetArrayElement(index_vector, update_value, &array_elements));
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

absl::Status IrInterpreter::HandleAssert(Assert* assert_op) {
  VLOG(2) << "Checking assert " << assert_op->ToString();
  VLOG(2) << "Condition is " << ResolveAsBool(assert_op->condition());
  if (!ResolveAsBool(assert_op->condition())) {
    GetInterpreterEvents().assert_msgs.push_back(assert_op->message());
  }
  return SetValueResult(assert_op, Value::Token());
}

absl::Status IrInterpreter::HandleTrace(Trace* trace_op) {
  if (ResolveAsBool(trace_op->condition())) {
    absl::Span<Node* const> arg_nodes = trace_op->args();
    auto arg_node = arg_nodes.begin();

    // TODO(amfv): 2021-09-14 Remove the duplication with the PerformTraceFmt
    // printing code in dslx/builtins.cc by making a common utility function
    // that takes make_error as an argument.
    auto make_error = [trace_op](std::string msg) -> absl::Status {
      return absl::InternalError(absl::StrFormat(
          "%s for format %s in trace node %s", msg,
          StepsToXlsFormatString(trace_op->format()), trace_op->ToString()));
    };

    std::string trace_output;

    for (auto step : trace_op->format()) {
      if (std::holds_alternative<std::string>(step)) {
        absl::StrAppend(&trace_output, std::get<std::string>(step));
      }

      if (std::holds_alternative<FormatPreference>(step)) {
        if (arg_node == arg_nodes.end()) {
          return make_error("Not enough operands");
        }
        auto arg_format = std::get<FormatPreference>(step);
        absl::StrAppend(&trace_output,
                        ResolveAsValue(*arg_node).ToHumanString(arg_format));
        arg_node++;
      }
    }

    if (arg_node != arg_nodes.end()) {
      return make_error("Too many operands");
    }

    VLOG(3) << "Trace output: " << trace_output;

    GetInterpreterEvents().trace_msgs.push_back(TraceMessage{
        .message = trace_output,
        .verbosity = trace_op->verbosity(),
    });
  }
  return SetValueResult(trace_op, Value::Token());
}

absl::Status IrInterpreter::HandleCover(Cover* cover) {
  // TODO(rspringer): 2021-05-25: Implement.
  return absl::OkStatus();
}

absl::Status IrInterpreter::HandleInvoke(Invoke* invoke) {
  Function* to_apply = invoke->to_apply();
  std::vector<Value> args;
  for (int64_t i = 0; i < to_apply->params().size(); ++i) {
    args.push_back(ResolveAsValue(invoke->operand(i)));
  }
  XLS_ASSIGN_OR_RETURN(InterpreterResult<Value> result,
                       InterpretFunction(to_apply, args));
  XLS_RETURN_IF_ERROR(AddInterpreterEvents(result.events));
  return SetValueResult(invoke, result.value);
}

absl::Status IrInterpreter::HandleInstantiationInput(
    InstantiationInput* instantiation_input) {
  return absl::UnimplementedError(
      "InstantiationInput not implemented in IrInterpreter");
}

absl::Status IrInterpreter::HandleInstantiationOutput(
    InstantiationOutput* instantiation_output) {
  return absl::UnimplementedError(
      "InstantiationOutput not implemented in IrInterpreter");
}

absl::Status IrInterpreter::HandleLiteral(Literal* literal) {
  return SetValueResult(literal, literal->value());
}

absl::Status IrInterpreter::HandleULe(CompareOp* le) {
  return SetUint64Result(
      le, static_cast<uint64_t>(bits_ops::ULessThanOrEqual(
              ResolveAsBits(le->operand(0)), ResolveAsBits(le->operand(1)))));
}

absl::Status IrInterpreter::HandleSLt(CompareOp* lt) {
  return SetUint64Result(
      lt, static_cast<uint64_t>(bits_ops::SLessThan(
              ResolveAsBits(lt->operand(0)), ResolveAsBits(lt->operand(1)))));
}

absl::Status IrInterpreter::HandleSLe(CompareOp* le) {
  return SetUint64Result(
      le, static_cast<uint64_t>(bits_ops::SLessThanOrEqual(
              ResolveAsBits(le->operand(0)), ResolveAsBits(le->operand(1)))));
}

absl::Status IrInterpreter::HandleULt(CompareOp* lt) {
  return SetUint64Result(
      lt, static_cast<uint64_t>(bits_ops::ULessThan(
              ResolveAsBits(lt->operand(0)), ResolveAsBits(lt->operand(1)))));
}

absl::Status IrInterpreter::HandleMap(Map* map) {
  Function* to_apply = map->to_apply();
  std::vector<Value> results;
  for (const Value& operand_element :
       ResolveAsValue(map->operand(0)).elements()) {
    XLS_ASSIGN_OR_RETURN(InterpreterResult<Value> result,
                         InterpretFunction(to_apply, {operand_element}));
    XLS_RETURN_IF_ERROR(AddInterpreterEvents(result.events));
    results.push_back(result.value);
  }
  XLS_ASSIGN_OR_RETURN(Value result_array, Value::Array(results));
  return SetValueResult(map, result_array);
}

absl::Status IrInterpreter::HandleSMul(ArithOp* mul) {
  const int64_t mul_width = mul->BitCountOrDie();
  Bits result = bits_ops::SMul(ResolveAsBits(mul->operand(0)),
                               ResolveAsBits(mul->operand(1)));
  if (result.bit_count() > mul_width) {
    return SetBitsResult(mul, result.Slice(0, mul_width));
  }
  if (result.bit_count() < mul_width) {
    return SetBitsResult(mul, bits_ops::SignExtend(result, mul_width));
  }
  return SetBitsResult(mul, result);
}

absl::Status IrInterpreter::HandleUMul(ArithOp* mul) {
  const int64_t mul_width = mul->BitCountOrDie();
  Bits result = bits_ops::UMul(ResolveAsBits(mul->operand(0)),
                               ResolveAsBits(mul->operand(1)));
  if (result.bit_count() > mul_width) {
    return SetBitsResult(mul, result.Slice(0, mul_width));
  }
  if (result.bit_count() < mul_width) {
    return SetBitsResult(mul, bits_ops::ZeroExtend(result, mul_width));
  }
  return SetBitsResult(mul, result);
}

absl::Status IrInterpreter::HandleSMulp(PartialProductOp* mul) {
  const int64_t mul_width = mul->width();
  VLOG(1) << "mul_width = " << mul_width << "\n";
  Bits result = bits_ops::SMul(ResolveAsBits(mul->operand(0)),
                               ResolveAsBits(mul->operand(1)));

  Bits offset = MulpOffsetForSimulation(mul_width, /*shift_size=*/2);
  if (result.bit_count() > mul_width) {
    result = result.Slice(0, mul_width);
  } else if (result.bit_count() < mul_width) {
    result = bits_ops::SignExtend(result, mul_width);
  }
  // Return an unsigned result.
  return SetValueResult(
      mul, Value::Tuple({Value(offset), Value(bits_ops::Sub(result, offset))}));
}

absl::Status IrInterpreter::HandleUMulp(PartialProductOp* mul) {
  const int64_t mul_width = mul->width();
  VLOG(1) << "mul_width = " << mul_width << "\n";
  Bits result = bits_ops::UMul(ResolveAsBits(mul->operand(0)),
                               ResolveAsBits(mul->operand(1)));
  Bits offset = MulpOffsetForSimulation(mul_width, /*shift_size=*/2);
  if (result.bit_count() > mul_width) {
    result = result.Slice(0, mul_width);
  } else if (result.bit_count() < mul_width) {
    result = bits_ops::ZeroExtend(result, mul_width);
  }
  return SetValueResult(
      mul, Value::Tuple({Value(offset), Value(bits_ops::Sub(result, offset))}));
}

absl::Status IrInterpreter::HandleNe(CompareOp* ne) {
  return SetUint64Result(ne,
                         static_cast<int64_t>(ResolveAsValue(ne->operand(0)) !=
                                              ResolveAsValue(ne->operand(1))));
}

absl::Status IrInterpreter::HandleNeg(UnOp* neg) {
  return SetBitsResult(neg, bits_ops::Negate(ResolveAsBits(neg->operand(0))));
}

absl::Status IrInterpreter::HandleNot(UnOp* not_op) {
  return SetBitsResult(not_op,
                       bits_ops::Not(ResolveAsBits(not_op->operand(0))));
}

absl::Status IrInterpreter::HandleOneHot(OneHot* one_hot) {
  int64_t output_width = one_hot->BitCountOrDie();
  const Bits& input = ResolveAsBits(one_hot->operand(0));
  const int64_t input_width = input.bit_count();
  for (int64_t i = 0; i < input.bit_count(); ++i) {
    int64_t index =
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
  for (int64_t i = 0; i < selector.bit_count(); ++i) {
    if (selector.Get(i)) {
      activated_inputs.push_back(&ResolveAsValue(sel->get_case(i)));
    }
  }
  XLS_ASSIGN_OR_RETURN(Value result, DeepOr(sel->GetType(), activated_inputs));
  return SetValueResult(sel, result);
}

absl::Status IrInterpreter::HandlePrioritySel(PrioritySelect* sel) {
  const Bits& selector = ResolveAsBits(sel->selector());
  for (int64_t i = 0; i < selector.bit_count(); ++i) {
    if (selector.Get(i)) {
      return SetValueResult(sel, ResolveAsValue(sel->get_case(i)));
    }
  }
  return SetValueResult(sel, ZeroOfType(sel->GetType()));
}

absl::Status IrInterpreter::HandleParam(Param* param) {
  return absl::UnimplementedError("Param not implemented in IrInterpreter");
}

absl::Status IrInterpreter::HandleNext(Next* next) {
  return absl::UnimplementedError(
      "Next value not implemented in IrInterpreter");
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
  XLS_ASSIGN_OR_RETURN(uint64_t i, selector.ToUint64());
  return SetValueResult(sel, ResolveAsValue(sel->get_case(i)));
}

absl::Status IrInterpreter::HandleShll(BinOp* shll) {
  const Bits& input = ResolveAsBits(shll->operand(0));
  const int64_t shift_amt =
      ResolveAsBoundedUint64(shll->operand(1), input.bit_count());
  return SetBitsResult(shll, bits_ops::ShiftLeftLogical(input, shift_amt));
}

absl::Status IrInterpreter::HandleShra(BinOp* shra) {
  const Bits& input = ResolveAsBits(shra->operand(0));
  const int64_t shift_amt =
      ResolveAsBoundedUint64(shra->operand(1), input.bit_count());
  return SetBitsResult(shra, bits_ops::ShiftRightArith(input, shift_amt));
}

absl::Status IrInterpreter::HandleShrl(BinOp* shrl) {
  const Bits& input = ResolveAsBits(shrl->operand(0));
  const int64_t shift_amt =
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
  int64_t tuple_index = index->As<TupleIndex>()->index();
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
  return NodeValuesMap().at(node).bits();
}

bool IrInterpreter::ResolveAsBool(Node* node) {
  const Bits& bits = NodeValuesMap().at(node).bits();
  CHECK_EQ(bits.bit_count(), 1);
  return bits.IsAllOnes();
}

std::vector<Bits> IrInterpreter::ResolveAsBitsVector(
    absl::Span<Node* const> nodes) {
  std::vector<Bits> results;
  for (Node* node : nodes) {
    results.push_back(ResolveAsBits(node));
  }
  return results;
}

uint64_t IrInterpreter::ResolveAsBoundedUint64(Node* node,
                                               uint64_t upper_limit) {
  return BitsToBoundedUint64(ResolveAsBits(node), upper_limit);
}

absl::Status IrInterpreter::SetUint64Result(Node* node, uint64_t result) {
  XLS_RET_CHECK(node->GetType()->IsBits());
  XLS_RET_CHECK_GE(node->BitCountOrDie(), Bits::MinBitCountUnsigned(result));
  return SetValueResult(node, Value(UBits(result, node->BitCountOrDie())));
}

absl::Status IrInterpreter::SetBitsResult(Node* node, const Bits& result) {
  XLS_RET_CHECK(node->GetType()->IsBits());
  XLS_RET_CHECK_EQ(node->BitCountOrDie(), result.bit_count());
  return SetValueResult(node, Value(result));
}

absl::Status IrInterpreter::SetValueResult(Node* node, Value result) {
  if (VLOG_IS_ON(4) &&
      std::all_of(node->operands().begin(), node->operands().end(),
                  [this](Node* o) { return NodeValuesMap().contains(o); })) {
    VLOG(4) << absl::StreamFormat("%s operands:", node->GetName());
    for (int64_t i = 0; i < node->operand_count(); ++i) {
      VLOG(4) << absl::StreamFormat(
          "  operand %d (%s): %s", i, node->operand(i)->GetName(),
          ResolveAsValue(node->operand(i)).ToString());
    }
  }
  VLOG(3) << absl::StreamFormat("Result of %s: %s", node->ToString(),
                                result.ToString());

  XLS_RET_CHECK(!NodeValuesMap().contains(node));
  if (!ValueConformsToType(result, node->GetType())) {
    return absl::InternalError(absl::StrFormat(
        "Expected value %s to match type %s of node %s", result.ToString(),
        node->GetType()->ToString(), node->GetName()));
  }
  NodeValuesMap()[node] = std::move(result);
  return absl::OkStatus();
}

absl::StatusOr<Value> IrInterpreter::DeepOr(
    Type* input_type, absl::Span<const Value* const> inputs) {
  if (input_type->IsBits()) {
    Bits result(input_type->AsBitsOrDie()->bit_count());
    for (const Value* input : inputs) {
      result = bits_ops::Or(result, input->bits());
    }
    return Value(result);
  }

  auto input_elements = [&](int64_t i) {
    std::vector<const Value*> values;
    for (int64_t j = 0; j < inputs.size(); ++j) {
      values.push_back(&inputs[j]->elements()[i]);
    }
    return values;
  };

  if (input_type->IsArray()) {
    Type* element_type = input_type->AsArrayOrDie()->element_type();
    std::vector<Value> elements;
    for (int64_t i = 0; i < input_type->AsArrayOrDie()->size(); ++i) {
      XLS_ASSIGN_OR_RETURN(Value element,
                           DeepOr(element_type, input_elements(i)));
      elements.push_back(element);
    }
    return Value::Array(elements);
  }

  XLS_RET_CHECK(input_type->IsTuple());
  std::vector<Value> elements;
  for (int64_t i = 0; i < input_type->AsTupleOrDie()->size(); ++i) {
    XLS_ASSIGN_OR_RETURN(
        Value element,
        DeepOr(input_type->AsTupleOrDie()->element_type(i), input_elements(i)));
    elements.push_back(element);
  }
  return Value::Tuple(elements);
}

}  // namespace xls
