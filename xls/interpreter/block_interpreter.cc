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

#include <optional>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/block_evaluator.h"
#include "xls/interpreter/ir_interpreter.h"
#include "xls/ir/block.h"
#include "xls/ir/events.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/register.h"
#include "xls/ir/value.h"

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
    VLOG(3) << absl::StreamFormat(
        "Next register value for register %s: %s",
        reg_write->GetRegister()->name(),
        next_reg_state_.at(reg_write->GetRegister()->name()).ToString());

    // Register writes have empty tuple types.
    return SetValueResult(reg_write, Value::Tuple({}));
  }

  absl::flat_hash_map<std::string, Value>&& MoveRegState() {
    return std::move(next_reg_state_);
  }

  InterpreterEvents&& MoveInterpreterEvents() { return std::move(events_); }

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
    // Empty tuples don't have data
    if (value.GetFlatBitCount() == 0) {
      continue;
    }
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
  result.interpreter_events = interpreter.MoveInterpreterEvents();

  return result;
}

}  // namespace xls
