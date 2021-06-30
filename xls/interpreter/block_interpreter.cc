// Copyright 2021 The XLS Authors
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

#include "xls/interpreter/block_interpreter.h"

#include "absl/status/status.h"
#include "xls/ir/bits.h"

namespace xls {
namespace {

// An interpreter for XLS blocks.
class BlockInterpreter : public IrInterpreter {
 public:
  BlockInterpreter(const absl::flat_hash_map<std::string, Value>& inputs)
      : IrInterpreter(/*args=*/{}), inputs_(inputs) {}

  absl::Status HandleInputPort(InputPort* input_port) override {
    if (!inputs_.contains(input_port->GetName())) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Missing input for port '%s'", input_port->GetName()));
    }

    return SetValueResult(input_port, inputs_.at(input_port->GetName()));
  }

  absl::Status HandleOutputPort(OutputPort* output_port) override {
    // Output ports have empty tuple types.
    return SetValueResult(output_port, Value::Tuple({}));
  }

 private:
  absl::flat_hash_map<std::string, Value> inputs_;
};

}  // namespace

absl::StatusOr<absl::flat_hash_map<std::string, Value>>
InterpretCombinationalBlock(
    Block* block, const absl::flat_hash_map<std::string, Value>& inputs) {
  absl::flat_hash_set<std::string> input_port_names;
  for (InputPort* port : block->GetInputPorts()) {
    input_port_names.insert(port->GetName());
  }
  for (const auto& [name, value] : inputs) {
    if (!input_port_names.contains(name)) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Block has no input port '%s'", name));
    }
  }

  absl::flat_hash_map<std::string, Value> outputs;
  BlockInterpreter visitor(inputs);
  XLS_RETURN_IF_ERROR(block->Accept(&visitor));
  for (Node* port : block->GetOutputPorts()) {
    outputs[port->GetName()] = visitor.ResolveAsValue(port->operand(0));
  }
  return outputs;
}

absl::StatusOr<absl::flat_hash_map<std::string, uint64_t>>
InterpretCombinationalBlock(
    Block* block, const absl::flat_hash_map<std::string, uint64_t>& inputs) {
  absl::flat_hash_map<std::string, Value> input_values;
  // Convert uint64_t inputs to Value inputs and validate that each input port
  // can accepts the uint64_t value.
  for (InputPort* port : block->GetInputPorts()) {
    if (!inputs.contains(port->GetName())) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Missing input for port '%s'", port->GetName()));
    }
    uint64_t input = inputs.at(port->GetName());
    if (!port->GetType()->IsBits()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Block has non-Bits-typed input port '%s' of type: %s",
          port->GetName(), port->GetType()->ToString()));
    }
    if (Bits::MinBitCountUnsigned(input) >
        port->GetType()->AsBitsOrDie()->bit_count()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Input value %d for input port '%s' does not fit in type: %s", input,
          port->GetName(), port->GetType()->ToString()));
    }
    input_values[port->GetName()] =
        Value(UBits(input, port->GetType()->AsBitsOrDie()->bit_count()));
  }

  absl::flat_hash_map<std::string, uint64_t> outputs;
  absl::flat_hash_map<std::string, Value> output_values;
  XLS_ASSIGN_OR_RETURN(output_values,
                       InterpretCombinationalBlock(block, input_values));

  for (OutputPort* port : block->GetOutputPorts()) {
    Node* data = port->operand(0);
    if (!data->GetType()->IsBits()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Block has non-Bits-typed output port '%s' of type: %s",
          port->GetName(), data->GetType()->ToString()));
    }
    const Value& value_output = output_values.at(port->GetName());
    const Bits& bits_output = value_output.bits();
    if (!bits_output.FitsInUint64()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Output value '%s' does not fit in a uint64_t: %s",
                          port->GetName(), value_output.ToString()));
    }
    XLS_ASSIGN_OR_RETURN(outputs[port->GetName()], bits_output.ToUint64());
  }

  return outputs;
}

}  // namespace xls
