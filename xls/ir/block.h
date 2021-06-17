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

#ifndef XLS_IR_BLOCK_H_
#define XLS_IR_BLOCK_H_

#include "absl/strings/string_view.h"
#include "xls/ir/function_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"

namespace xls {

class Block;

// Data structure describing the reset behavior of a register.
struct Reset {
  Value reset_value;
  bool asynchronous;
  bool active_low;
};

// Data structure representing a RTL-level register. These constructs are
// contained in and owned by Blocks and lower to registers in Verilog.
class Register {
 public:
  Register(absl::string_view name, Type* type, absl::optional<Reset> reset,
           Block* block)
      : name_(name), type_(type), reset_(std::move(reset)), block_(block) {}

  const std::string& name() const { return name_; }
  Type* type() const { return type_; }
  const absl::optional<Reset>& reset() const { return reset_; }

  // Returns the block which owns the register.
  Block* block() const { return block_; }

 private:
  std::string name_;
  Type* type_;
  absl::optional<Reset> reset_;
  Block* block_;
};

// Abstraction representing a Verilog module used in code generation. Blocks are
// function-level constructs similar to functions and procs. Like functions and
// procs, blocks contain (and own) a data-flow graph of nodes. With a small
// number of exceptions (e.g., send/receive) blocks may contain arbitrary nodes
// including dead nodes. Blocks allow an arbitrary number of inputs and outputs
// represented with ports, and blocks may contain registers.
//
// Blocks contain a single token parameter and a single terminal token. These
// tokens are used for supporting token-requiring operations such as assert.
class Block : public FunctionBase {
 public:
  Block(absl::string_view name, Package* package)
      : FunctionBase(name, package) {}
  virtual ~Block() = default;

  // Abstraction describing the clock port.
  struct ClockPort {
    std::string name;
  };

  // Representation of a port of a block. Ports are the interface of the block
  // and represent ports on a Verilog module.
  using Port = absl::variant<InputPort*, OutputPort*, ClockPort*>;

  // Returns the ports in the block. The ports are returned in the order that
  // they will be emitted in the generated Verilog module. Input and output
  // ports may be arbitrarily ordered.
  absl::Span<const Port> GetPorts() const { return ports_; }

  // Returns the input/output ports of the block. Ports are ordered by the
  // position in the generated Verilog module.
  absl::Span<InputPort* const> GetInputPorts() const { return input_ports_; }
  absl::Span<OutputPort* const> GetOutputPorts() const { return output_ports_; }

  // Adds an input/output port to the block. These methods should be used to add
  // ports rather than FunctionBase::AddNode and FunctionBase::MakeNode (checked
  // later by the verifier).
  absl::StatusOr<InputPort*> AddInputPort(
      absl::string_view name, Type* type,
      absl::optional<SourceLocation> loc = absl::nullopt);
  absl::StatusOr<OutputPort*> AddOutputPort(
      absl::string_view name, Node* operand,
      absl::optional<SourceLocation> loc = absl::nullopt);

  // Add/get a clock port to the block. The clock is not represented as a value
  // within the IR (no input_port operation), but rather a Block::ClockPort
  // object is added to the list of ports as a place-holder for the clock which
  // records the port name and position.
  absl::Status AddClockPort(absl::string_view name);
  absl::optional<ClockPort> GetClockPort() const { return clock_port_; }

  absl::Status RemoveNode(Node* n) override;

  // Re-orders the ports of the block as determined by `port_order`. The order
  // of the ports in a block determines their order in the emitted verilog
  // module. `port_names` must include (exactly) the name of every port.
  absl::Status ReorderPorts(absl::Span<const std::string> port_names);

  // Returns all registers in the block in the order they were added.
  absl::Span<Register* const> GetRegisters() const { return register_vec_; }

  // Returns the register in the block with the given name. Returns an error if
  // no such register exists.
  absl::StatusOr<Register*> GetRegister(absl::string_view name) const;

  // Returns true iff this block contains a register with the given name.
  bool HasRegisterWithName(absl::string_view name) const {
    return registers_.contains(name);
  }

  // Adds a register to the block.
  absl::StatusOr<Register*> AddRegister(
      absl::string_view name, Type* type,
      absl::optional<Reset> reset = absl::nullopt);

  // Removes the given register from the block. If the register is not owned by
  // the block then an error is returned.
  absl::Status RemoveRegister(Register* reg);

  bool HasImplicitUse(Node* node) const override { return false; }

  std::string DumpIr(bool recursive = false) const override;

 private:
  static std::string PortName(const Port& port);

  // All ports in the block in the order they appear in the Verilog module.
  std::vector<Port> ports_;

  // Ports indexed by name.
  absl::flat_hash_map<std::string, Port> ports_by_name_;

  // All input/output ports in the order they appear in the Verilog module.
  std::vector<InputPort*> input_ports_;
  std::vector<OutputPort*> output_ports_;

  // Registers within this block. Indexed by register name. Stored as
  // std::unique_ptrs for pointer stability.
  absl::flat_hash_map<std::string, std::unique_ptr<Register>> registers_;

  // Vector of register pointers. Ordered by register creation time. Kept in
  // sync with the registers_ map. Enables easy, stable iteration over
  // registers. With this vector, deletion of a register is O(n) with the number
  // of registers. If this is a problem, a linked list might be used instead.
  std::vector<Register*> register_vec_;

  absl::optional<ClockPort> clock_port_;
};

}  // namespace xls

#endif  // XLS_IR_BLOCK_H_
