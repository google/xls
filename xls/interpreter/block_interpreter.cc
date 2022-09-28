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
  BlockInterpreter(const absl::flat_hash_map<std::string, Value>& inputs,
                   const absl::flat_hash_map<std::string, Value>& reg_state)
      : IrInterpreter(/*args=*/{}), inputs_(inputs), reg_state_(reg_state) {}

  absl::Status HandleInputPort(InputPort* input_port) override {
    auto port_iter = inputs_.find(input_port->GetName());
    if (port_iter == inputs_.end()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Missing input for port '%s'", input_port->GetName()));
    }

    return SetValueResult(input_port, port_iter->second);
  }

  absl::Status HandleOutputPort(OutputPort* output_port) override {
    // Output ports have empty tuple types.
    return SetValueResult(output_port, Value::Tuple({}));
  }

  absl::Status HandleRegisterRead(RegisterRead* reg_read) override {
    auto reg_value_iter = reg_state_.find(reg_read->GetRegister()->name());
    if (reg_value_iter == reg_state_.end()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Missing value for register '%s'", reg_read->GetRegister()->name()));
    }
    return SetValueResult(reg_read, reg_value_iter->second);
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

  absl::flat_hash_map<std::string, Value>&& MoveRegState() {
    return std::move(next_reg_state_);
  }

 private:
  // Values fed to the input ports.
  const absl::flat_hash_map<std::string, Value> inputs_;

  // The state of the registers in this iteration.
  const absl::flat_hash_map<std::string, Value> reg_state_;

  // The next state for the registers.
  absl::flat_hash_map<std::string, Value> next_reg_state_;
};

}  // namespace

absl::StatusOr<BlockRunResult> BlockRun(
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

  BlockRunResult result;
  for (Node* port : block->GetOutputPorts()) {
    result.outputs[port->GetName()] =
        interpreter.ResolveAsValue(port->operand(0));
  }
  result.reg_state = std::move(interpreter.MoveRegState());

  return result;
}

// Convert a uint64_t to a Value suitable for node's type.
static absl::StatusOr<Value> ConvertInputUint64ToValue(uint64_t input,
                                                       const InputPort* port,
                                                       Block* block) {
  if (!port->GetType()->IsBits()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Block has non-Bits-typed input port '%s' of type: %s",
                        port->GetName(), port->GetType()->ToString()));
  }
  if (Bits::MinBitCountUnsigned(input) >
      port->GetType()->AsBitsOrDie()->bit_count()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Input value %d for input port '%s' does not fit in type: %s", input,
        port->GetName(), port->GetType()->ToString()));
  }

  return Value(UBits(input, port->GetType()->AsBitsOrDie()->bit_count()));
}

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
    auto port_iter = inputs.find(port->GetName());
    if (port_iter == inputs.end()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Missing input for port '%s'", port->GetName()));
    }
    uint64_t input = port_iter->second;
    XLS_ASSIGN_OR_RETURN(input_values[port->GetName()],
                         ConvertInputUint64ToValue(input, port, block));
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

// Converts each input Value to a uint64_t and returns the resulting map. There
// must exist an input for each input port on the block. If the Value
// does not fit into a uint64_t an error is returned.
static absl::StatusOr<absl::flat_hash_map<std::string, uint64_t>>
ConvertInputsToUint64(const absl::flat_hash_map<std::string, Value> inputs,
                      Block* block) {
  absl::flat_hash_map<std::string, uint64_t> input_uint64s;
  for (InputPort* port : block->GetInputPorts()) {
    if (!port->GetType()->IsBits()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Block has non-Bits-typed input port '%s' of type: %s",
          port->GetName(), port->GetType()->ToString()));
    }
    const Value& value_input = inputs.at(port->GetName());
    const Bits& bits_input = value_input.bits();
    if (!bits_input.FitsInUint64()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Input value '%s' does not fit in a uint64_t: %s",
                          port->GetName(), value_input.ToString()));
    }
    XLS_ASSIGN_OR_RETURN(input_uint64s[port->GetName()], bits_input.ToUint64());
  }
  return input_uint64s;
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
    XLS_ASSIGN_OR_RETURN(BlockRunResult result,
                         BlockRun(input_set, reg_state, block));
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

absl::Status ChannelSource::SetDataSequence(std::vector<Value> data) {
  // TODO(tedhong): 2022-03-15 - Add additional checks to ensure type of Value
  // elements in data are consistent and compatible with this channel's
  // data port type.
  data_sequence_ = std::move(data);

  return absl::OkStatus();
}

absl::Status ChannelSource::SetDataSequence(absl::Span<const uint64_t> data) {
  XLS_ASSIGN_OR_RETURN(const InputPort* port, block_->GetInputPort(data_name_));

  for (const uint64_t input : data) {
    XLS_ASSIGN_OR_RETURN(Value input_as_value,
                         ConvertInputUint64ToValue(input, port, block_));
    data_sequence_.push_back(input_as_value);
  }

  return absl::OkStatus();
}

absl::Status ChannelSource::SetBlockInputs(
    int64_t this_cycle, absl::flat_hash_map<std::string, Value>& inputs,
    std::minstd_rand& random_engine, std::optional<verilog::ResetProto> reset) {
  bool is_in_reset = false;

  // Don't send inputs when reset is asserted, because we don't typically care
  // about the behavior of the block when inputs are sent during reset.
  if (reset.has_value()) {
    if (inputs.contains(reset.value().name())) {
      Value value_when_reset_asserted =
          Value(UBits(reset.value().active_low() ? 0 : 1, 1));
      if (inputs.at(reset.value().name()) == value_when_reset_asserted) {
        is_in_reset = true;
      }
    }
  }

  if (!is_in_reset) {
    if (is_valid_) {
      // Continue to output valid and data, while waiting for the ready signal.
      XLS_CHECK_GE(current_index_, 0);
      XLS_CHECK_LT(current_index_, data_sequence_.size());

      inputs[data_name_] = data_sequence_.at(current_index_);
      inputs[valid_name_] = Value(UBits(1, 1));

      return absl::OkStatus();
    }

    if (HasMoreData()) {
      bool send_next_data = std::bernoulli_distribution(lambda_)(random_engine);
      if (send_next_data) {
        ++current_index_;

        XLS_CHECK_GE(current_index_, 0);
        XLS_CHECK_LT(current_index_, data_sequence_.size());

        inputs[data_name_] = data_sequence_.at(current_index_);
        inputs[valid_name_] = Value(UBits(1, 1));
        is_valid_ = true;

        return absl::OkStatus();
      }
    }
  }

  // If stalling, send a zero value with valid bit set to zero.
  XLS_ASSIGN_OR_RETURN(const InputPort* port, block_->GetInputPort(data_name_));
  inputs[data_name_] = ZeroOfType(port->GetType());
  inputs[valid_name_] = Value(UBits(0, 1));

  return absl::OkStatus();
}

absl::Status ChannelSource::GetBlockOutputs(
    int64_t this_cycle,
    const absl::flat_hash_map<std::string, Value>& outputs) {
  auto ready_iter = outputs.find(ready_name_);
  if (ready_iter == outputs.end()) {
    return absl::InternalError(absl::StrFormat(
        "Block %s Channel %s Port %s value not found in interpreter output",
        block_->name(), data_name_, ready_name_));
  }

  const bool ready = ready_iter->second.bits().IsAllOnes();
  if (is_valid_ && ready) {
    is_valid_ = false;
  }

  return absl::OkStatus();
}

absl::Status ChannelSink::SetBlockInputs(
    int64_t this_cycle, absl::flat_hash_map<std::string, Value>& inputs,
    std::minstd_rand& random_engine) {
  // Ready is independently random each cycle
  is_ready_ = std::bernoulli_distribution(lambda_)(random_engine);
  inputs[ready_name_] = is_ready_ ? Value(UBits(1, 1)) : Value(UBits(0, 1));

  return absl::OkStatus();
}

absl::Status ChannelSink::GetBlockOutputs(
    int64_t this_cycle,
    const absl::flat_hash_map<std::string, Value>& outputs) {
  auto valid_iter = outputs.find(valid_name_);
  if (valid_iter == outputs.end()) {
    return absl::InternalError(absl::StrFormat(
        "Block %s Channel %s Port %s value not found in interpreter output",
        block_->name(), data_name_, valid_name_));
  }

  // If ready and valid, grab data.
  const bool valid = valid_iter->second.bits().IsAllOnes();
  if (is_ready_ && valid) {
    Value data = outputs.at(data_name_);
    data_sequence_.push_back(data);
  }

  return absl::OkStatus();
}

absl::StatusOr<std::vector<uint64_t>> ChannelSink::GetOutputSequenceAsUint64()
    const {
  std::vector<uint64_t> ret;
  ret.reserve(data_sequence_.size());

  for (const Value& v : data_sequence_) {
    const Bits& bits_output = v.bits();
    XLS_ASSIGN_OR_RETURN(uint64_t v_as_int, bits_output.ToUint64());
    ret.push_back(v_as_int);
  }

  return ret;
}

absl::StatusOr<BlockIOResults> InterpretChannelizedSequentialBlock(
    Block* block, absl::Span<ChannelSource> channel_sources,
    absl::Span<ChannelSink> channel_sinks,
    absl::Span<const absl::flat_hash_map<std::string, Value>> inputs,
    std::optional<verilog::ResetProto> reset, int64_t seed) {
  std::minstd_rand random_engine;
  random_engine.seed(seed);

  // Initial register state is zero for all registers.
  absl::flat_hash_map<std::string, Value> reg_state;
  for (Register* reg : block->GetRegisters()) {
    reg_state[reg->name()] = ZeroOfType(reg->type());
  }

  int64_t max_cycle_count = inputs.size();

  BlockIOResults block_io_results;
  for (int64_t cycle = 0; cycle < max_cycle_count; ++cycle) {
    absl::flat_hash_map<std::string, Value> input_set = inputs.at(cycle);

    // Sources set data/valid
    for (ChannelSource& src : channel_sources) {
      XLS_RETURN_IF_ERROR(
          src.SetBlockInputs(cycle, input_set, random_engine, reset));
    }

    // Sinks set ready
    for (ChannelSink& sink : channel_sinks) {
      XLS_RETURN_IF_ERROR(sink.SetBlockInputs(cycle, input_set, random_engine));
    }

    if (XLS_VLOG_IS_ON(3)) {
      XLS_VLOG(3) << absl::StrFormat("Inputs Cycle %d", cycle);
      for (auto [name, val] : input_set) {
        XLS_VLOG(3) << absl::StrFormat("%s: %s", name, val.ToString());
      }
    }

    // Block results
    XLS_ASSIGN_OR_RETURN(BlockRunResult result,
                         BlockRun(input_set, reg_state, block));

    // Sources get ready
    for (ChannelSource& src : channel_sources) {
      XLS_RETURN_IF_ERROR(src.GetBlockOutputs(cycle, result.outputs));
    }

    // Sinks get data/valid
    for (ChannelSink& sink : channel_sinks) {
      XLS_RETURN_IF_ERROR(sink.GetBlockOutputs(cycle, result.outputs));
    }

    if (XLS_VLOG_IS_ON(3)) {
      XLS_VLOG(3) << absl::StrFormat("Outputs Cycle %d", cycle);
      for (auto [name, val] : result.outputs) {
        XLS_VLOG(3) << absl::StrFormat("%s: %s", name, val.ToString());
      }
    }

    reg_state = std::move(result.reg_state);

    block_io_results.inputs.push_back(std::move(input_set));
    block_io_results.outputs.push_back(std::move(result.outputs));
  }

  return block_io_results;
}

absl::StatusOr<BlockIOResultsAsUint64>
InterpretChannelizedSequentialBlockWithUint64(
    Block* block, absl::Span<ChannelSource> channel_sources,
    absl::Span<ChannelSink> channel_sinks,
    absl::Span<const absl::flat_hash_map<std::string, uint64_t>> inputs,
    std::optional<verilog::ResetProto> reset, int64_t seed) {
  std::vector<absl::flat_hash_map<std::string, Value>> input_values;
  for (const absl::flat_hash_map<std::string, uint64_t>& input_set : inputs) {
    absl::flat_hash_map<std::string, Value> input_value_set;

    for (const auto& [port_name, input_val] : input_set) {
      XLS_ASSIGN_OR_RETURN(const InputPort* port,
                           block->GetInputPort(port_name));
      XLS_ASSIGN_OR_RETURN(input_value_set[port_name],
                           ConvertInputUint64ToValue(input_val, port, block));
    }

    input_values.push_back(std::move(input_value_set));
  }

  XLS_ASSIGN_OR_RETURN(
      BlockIOResults block_io_result,
      InterpretChannelizedSequentialBlock(block, channel_sources, channel_sinks,
                                          input_values, reset, seed));

  BlockIOResultsAsUint64 block_io_result_as_uint64;

  for (const absl::flat_hash_map<std::string, Value>& output_value_set :
       block_io_result.outputs) {
    absl::flat_hash_map<std::string, uint64_t> output_set;
    XLS_ASSIGN_OR_RETURN(output_set,
                         ConvertOutputsToUint64(output_value_set, block));
    block_io_result_as_uint64.outputs.push_back(std::move(output_set));
  }

  for (const absl::flat_hash_map<std::string, Value>& input_value_set :
       block_io_result.inputs) {
    absl::flat_hash_map<std::string, uint64_t> input_set;
    XLS_ASSIGN_OR_RETURN(input_set,
                         ConvertInputsToUint64(input_value_set, block));
    block_io_result_as_uint64.inputs.push_back(std::move(input_set));
  }
  return block_io_result_as_uint64;
}

}  // namespace xls
