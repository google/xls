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
#include "xls/ir/value_helpers.h"

namespace xls {
namespace {

// An interpreter for XLS blocks.
class BlockInterpreter : public IrInterpreter {
 public:
  struct RunResult {
    absl::flat_hash_map<std::string, Value> outputs;
    absl::flat_hash_map<std::string, Value> reg_state;
  };

  // Runs a single cycle of a block with the given register values and input
  // values. Returns the value sent to the output port and the next register
  // state.
  static absl::StatusOr<RunResult> Run(
      const absl::flat_hash_map<std::string, Value>& inputs,
      const absl::flat_hash_map<std::string, Value>& reg_state, Block* block) {
    // Verify each input corresponds to an input port. The reverse check (each
    // input port has a corresponding value in `inputs`) is checked in
    // HandleInputPort.
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

    // Verify each register value corresponds to a register. The reverse check
    // (each register has a corresponding value in `reg_state`) is checked in
    // HandleRegisterRead.
    absl::flat_hash_set<std::string> reg_names;
    for (Register* reg : block->GetRegisters()) {
      reg_names.insert(reg->name());
    }
    for (const auto& [name, value] : reg_state) {
      if (!reg_names.contains(name)) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Block has no register '%s'", name));
      }
    }

    BlockInterpreter interpreter(inputs, reg_state);
    XLS_RETURN_IF_ERROR(block->Accept(&interpreter));

    RunResult result;
    for (Node* port : block->GetOutputPorts()) {
      result.outputs[port->GetName()] =
          interpreter.ResolveAsValue(port->operand(0));
    }
    result.reg_state = std::move(interpreter.next_reg_state_);

    return std::move(result);
  }

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

  absl::Status HandleRegisterRead(RegisterRead* reg_read) override {
    if (!reg_state_.contains(reg_read->GetRegister()->name())) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Missing value for register '%s'", reg_read->GetRegister()->name()));
    }
    return SetValueResult(reg_read,
                          reg_state_.at(reg_read->GetRegister()->name()));
  }

  absl::Status HandleRegisterWrite(RegisterWrite* reg_write) override {
    auto get_next_reg_state = [&]() -> Value {
      if (reg_write->reset().has_value()) {
        bool reset_signal = ResolveAsBool(reg_write->reset().value());
        const Reset& reset = reg_write->GetRegister()->reset().value();
        if ((reset_signal && !reset.active_low) ||
            (!reset_signal && reset.active_low)) {
          // Reset is activated. Next register state is the reset value.
          return reset.reset_value;
        }
      }
      if (reg_write->load_enable().has_value() &&
          !ResolveAsBool(reg_write->load_enable().value())) {
        // Load enable is not activated. Next register state is the previous
        // register value.
        return reg_state_.at(reg_write->GetRegister()->name());
      }

      // Next register state is the input data value.
      return ResolveAsValue(reg_write->data());
    };

    next_reg_state_[reg_write->GetRegister()->name()] = get_next_reg_state();
    XLS_VLOG(3) << absl::StreamFormat(
        "Next register value for register %s: %s",
        reg_write->GetRegister()->name(),
        next_reg_state_.at(reg_write->GetRegister()->name()).ToString());

    // Register writes have empty tuple types.
    return SetValueResult(reg_write, Value::Tuple({}));
  }

 private:
  BlockInterpreter(const absl::flat_hash_map<std::string, Value>& inputs,
                   const absl::flat_hash_map<std::string, Value>& reg_state)
      : IrInterpreter(/*args=*/{}), inputs_(inputs), reg_state_(reg_state) {}
  // Values fed to the input ports.
  const absl::flat_hash_map<std::string, Value> inputs_;

  // The state of the registers in this iteration.
  const absl::flat_hash_map<std::string, Value> reg_state_;

  // The next state for the registers.
  absl::flat_hash_map<std::string, Value> next_reg_state_;
};

}  // namespace

// Converts each uint64_t input to a Value and returns the resulting map. There
// must exist an input for each input port on the block. If the input uint64_t
// value does not fit in the respective type an error is returned.
static absl::StatusOr<absl::flat_hash_map<std::string, Value>>
ConvertInputsToValues(const absl::flat_hash_map<std::string, uint64_t> inputs,
                      Block* block) {
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
  return std::move(input_values);
}

// Converts each output Value to a uint64_t and returns the resulting map. There
// must exist an output for each output port on the block. If the Value
// does not fit into a uint64_t  an error is returned.
static absl::StatusOr<absl::flat_hash_map<std::string, uint64_t>>
ConvertOutputsToUint64(const absl::flat_hash_map<std::string, Value> outputs,
                       Block* block) {
  absl::flat_hash_map<std::string, uint64_t> output_uint64s;
  for (OutputPort* port : block->GetOutputPorts()) {
    Node* data = port->operand(0);
    if (!data->GetType()->IsBits()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Block has non-Bits-typed output port '%s' of type: %s",
          port->GetName(), data->GetType()->ToString()));
    }
    const Value& value_output = outputs.at(port->GetName());
    const Bits& bits_output = value_output.bits();
    if (!bits_output.FitsInUint64()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Output value '%s' does not fit in a uint64_t: %s",
                          port->GetName(), value_output.ToString()));
    }
    XLS_ASSIGN_OR_RETURN(output_uint64s[port->GetName()],
                         bits_output.ToUint64());
  }
  return std::move(output_uint64s);
}

absl::StatusOr<absl::flat_hash_map<std::string, Value>>
InterpretCombinationalBlock(
    Block* block, const absl::flat_hash_map<std::string, Value>& inputs) {
  std::vector<absl::flat_hash_map<std::string, Value>> outputs;
  XLS_ASSIGN_OR_RETURN(outputs, InterpretSequentialBlock(block, {inputs}));
  XLS_RET_CHECK_EQ(outputs.size(), 1);
  return std::move(outputs[0]);
}

absl::StatusOr<absl::flat_hash_map<std::string, uint64_t>>
InterpretCombinationalBlock(
    Block* block, const absl::flat_hash_map<std::string, uint64_t>& inputs) {
  std::vector<absl::flat_hash_map<std::string, uint64_t>> outputs;
  XLS_ASSIGN_OR_RETURN(outputs, InterpretSequentialBlock(block, {inputs}));
  XLS_RET_CHECK_EQ(outputs.size(), 1);
  return std::move(outputs[0]);
}

absl::StatusOr<std::vector<absl::flat_hash_map<std::string, Value>>>
InterpretSequentialBlock(
    Block* block,
    absl::Span<const absl::flat_hash_map<std::string, Value>> inputs) {
  // Initial register state is zero for all registers.
  absl::flat_hash_map<std::string, Value> reg_state;
  for (Register* reg : block->GetRegisters()) {
    reg_state[reg->name()] = ZeroOfType(reg->type());
  }

  std::vector<absl::flat_hash_map<std::string, Value>> outputs;
  for (const absl::flat_hash_map<std::string, Value>& input_set : inputs) {
    XLS_ASSIGN_OR_RETURN(BlockInterpreter::RunResult result,
                         BlockInterpreter::Run(input_set, reg_state, block));
    outputs.push_back(std::move(result.outputs));
    reg_state = std::move(result.reg_state);
  }
  return std::move(outputs);
}

absl::StatusOr<std::vector<absl::flat_hash_map<std::string, uint64_t>>>
InterpretSequentialBlock(
    Block* block,
    absl::Span<const absl::flat_hash_map<std::string, uint64_t>> inputs) {
  std::vector<absl::flat_hash_map<std::string, Value>> input_values;
  for (const absl::flat_hash_map<std::string, uint64_t>& input_set : inputs) {
    absl::flat_hash_map<std::string, Value> input_value_set;
    XLS_ASSIGN_OR_RETURN(input_value_set,
                         ConvertInputsToValues(input_set, block));
    input_values.push_back(std::move(input_value_set));
  }

  std::vector<absl::flat_hash_map<std::string, Value>> output_values;
  XLS_ASSIGN_OR_RETURN(output_values,
                       InterpretSequentialBlock(block, input_values));

  std::vector<absl::flat_hash_map<std::string, uint64_t>> outputs;
  for (const absl::flat_hash_map<std::string, Value>& output_value_set :
       output_values) {
    absl::flat_hash_map<std::string, uint64_t> output_set;
    XLS_ASSIGN_OR_RETURN(output_set,
                         ConvertOutputsToUint64(output_value_set, block));
    outputs.push_back(std::move(output_set));
  }

  return outputs;
}

}  // namespace xls
