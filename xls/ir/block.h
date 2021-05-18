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
#include "xls/ir/package.h"
#include "xls/ir/type.h"

namespace xls {

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

  // Returns the ports in the block. The ports are returned in the order that
  // they will be emitted in the generated Verilog module. Input and output
  // ports may be arbitrarily ordered.
  absl::Span<Node* const> GetPorts() const { return ports_; }

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

  bool HasImplicitUse(Node* node) const override {
    return node->Is<OutputPort>();
  }

  std::string DumpIr(bool recursive = false) const override;

 private:
  // All ports in the block in the order they appear in the Verilog module.
  std::vector<Node*> ports_;

  // Ports indexed by name.
  absl::flat_hash_map<std::string, Node*> ports_by_name_;

  // All input/output ports in the order they appear in the Verilog module.
  std::vector<InputPort*> input_ports_;
  std::vector<OutputPort*> output_ports_;
};

}  // namespace xls

#endif  // XLS_IR_BLOCK_H_
