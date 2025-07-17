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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/ir/channel.h"
#include "xls/ir/function_base.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/name_uniquer.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/register.h"
#include "xls/ir/source_location.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

namespace xls {

enum class BlockProvenanceKind : int8_t { kProc, kFunction };

// Metadata used during codegen which describes where a block comes from.
struct BlockProvenance {
  // The name of the proc or function the block was lowered from.
  std::string name;
  // The kind of IR construct (proc or function) this block was lowered from.
  BlockProvenanceKind kind;

  bool IsFromProc() const { return kind == BlockProvenanceKind::kProc; }
  bool IsFromFunction() const { return kind == BlockProvenanceKind::kFunction; }

  bool operator==(const BlockProvenance& other) const {
    return name == other.name && kind == other.kind;
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const BlockProvenance& v) {
    absl::Format(&sink, "provenance(name=\"%s\", kind=\"%s\")", v.name,
                 v.kind == BlockProvenanceKind::kProc ? "proc" : "function");
  }

  template <typename H>
  friend H AbslHashValue(H h, const BlockProvenance& v) {
    return H::combine(std::move(h), v.name, v.kind);
  }
};

// Metadata which maps ports back to the channels the were derived from.
struct ChannelPortMetadata {
  std::string channel_name;
  Type* type;
  // The direction of the data/valid port (send or receive).
  ChannelDirection direction;
  ChannelKind channel_kind;
  FlopKind flop_kind;

  // Names of the ports for data/valid/ready signals for the channel. The value
  // is std::nullopt if no such port exists.
  std::optional<std::string> data_port;
  std::optional<std::string> valid_port;
  std::optional<std::string> ready_port;

  // For channels with no flow control, this is the number of cycles after reset
  // in which a value appears on the port (direction is kSend) or when a value
  // is read at the data port (direction is kReceive). Unused otherwise.
  std::optional<int64_t> stage;

  std::string ToString() const;
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
  Block(std::string_view name, Package* package)
      : FunctionBase(name, package), register_name_uniquer_("__") {}
  ~Block() override = default;

  // Abstraction describing the clock port.
  struct ClockPort {
    std::string name;
  };

  // Representation of a port of a block. Ports are the interface of the block
  // and represent ports on a Verilog module.
  using Port = std::variant<InputPort*, OutputPort*, ClockPort*>;

  // Returns the name of the port.
  static std::string PortName(const Port& port);

  // Returns the ports in the block. The ports are returned in the order that
  // they will be emitted in the generated Verilog module. Input and output
  // ports may be arbitrarily ordered.
  absl::Span<const Port> GetPorts() const { return ports_; }

  // Returns the input/output ports of the block. Ports are ordered by the
  // position in the generated Verilog module.
  absl::Span<InputPort* const> GetInputPorts() const { return input_ports_; }
  absl::Span<OutputPort* const> GetOutputPorts() const { return output_ports_; }

  // Return the input/output port of the given name.
  absl::StatusOr<PortNode*> GetPortNode(std::string_view name) const;

  // Returns a given input/output port by name.
  absl::StatusOr<InputPort*> GetInputPort(std::string_view name) const;
  bool HasInputPort(std::string_view name) const;
  absl::StatusOr<OutputPort*> GetOutputPort(std::string_view name) const;
  bool HasOutputPort(std::string_view name) const;

  // Adds an input/output port to the block. These methods should be used to add
  // ports rather than FunctionBase::AddNode and FunctionBase::MakeNode (checked
  // later by the verifier).
  absl::StatusOr<InputPort*> AddInputPort(std::string_view name, Type* type,
                                          const SourceInfo& loc = SourceInfo());
  absl::StatusOr<OutputPort*> AddOutputPort(
      std::string_view name, Node* operand,
      const SourceInfo& loc = SourceInfo());

  // Add/get a clock port to the block. The clock is not represented as a value
  // within the IR (no input_port operation), but rather a Block::ClockPort
  // object is added to the list of ports as a place-holder for the clock which
  // records the port name and position.
  absl::Status AddClockPort(std::string_view name);
  const std::optional<ClockPort>& GetClockPort() const { return clock_port_; }

  // Adds a new single-bit input port to the block and sets it as the reset port
  // with the given behavior. Returns the port.
  absl::StatusOr<InputPort*> AddResetPort(std::string_view port_name,
                                          ResetBehavior behavior);

  // Set an existing port in the block to be the reset port with the given
  // behavior. If the block already has a reset port an error is returned.
  absl::Status SetResetPort(InputPort* input_port, ResetBehavior behavior);

  // Override the reset behavior of the block. The block must already have a
  // reset port set (and necessarily an existing reset behavior defined).
  absl::Status OverrideResetBehavior(ResetBehavior behavior);

  // Returns the reset port or reset behavior of the block, or std::nullopt if
  // the block has no reset port. Either both reset port and reset behavior is
  // set, or neither is set.
  std::optional<InputPort*> GetResetPort() const { return reset_port_; }
  std::optional<ResetBehavior> GetResetBehavior() const {
    return reset_behavior_;
  };

  absl::Status RemoveNode(Node* n) override;

  // Re-orders the ports of the block as determined by `port_order`. The order
  // of the ports in a block determines their order in the emitted verilog
  // module. `port_names` must include (exactly) the name of every port.
  absl::Status ReorderPorts(absl::Span<const std::string> port_names);

  // Returns all registers in the block in the order they were added.
  absl::Span<Register* const> GetRegisters() const { return register_vec_; }

  // Returns the register in the block with the given name. Returns an error
  // if no such register exists.
  absl::StatusOr<Register*> GetRegister(std::string_view name) const;

  // Adds a register to the block.
  //
  // The requested_name is the name which will be used if possible. If the
  // name is already used a uniquified name will be used. Query the register
  // to get the actual name used by the register.
  absl::StatusOr<Register*> AddRegister(
      std::string_view requested_name, Type* type,
      std::optional<Value> reset_value = std::nullopt);

  // Adds a register with a reset value of zero.
  absl::StatusOr<Register*> AddRegisterWithZeroResetValue(
      std::string_view requested_name, Type* type);

  // Removes the given register from the block. If the register is not owned
  // by the block then an error is returned.
  absl::Status RemoveRegister(Register* reg);

  // Returns the unique register read or write operation associated with the
  // given register. Returns an error if the register is not owned by the
  // block or if no or more than one such read/write operation exists. A block
  // with a register without both a read and write operation is malformed but
  // may exist temporarily after the creation of the register and before
  // adding the read and write operations. A register must have only one
  // register read, but may have multiple register writes.
  absl::StatusOr<RegisterRead*> GetRegisterRead(Register* reg) const;
  absl::StatusOr<RegisterWrite*> GetUniqueRegisterWrite(Register* reg) const;

  // Returns the register write operations associated with given register. A
  // register may have multiple writes, but only one may activate in a given
  // cycle.
  absl::StatusOr<absl::Span<RegisterWrite* const>> GetRegisterWrites(
      Register* reg) const;

  // Add an instantiation of the given block `instantiated_block` to this
  // block. InstantiationInput and InstantiationOutput operations must be later
  // added to connect the instantiation to the data-flow graph.
  absl::StatusOr<BlockInstantiation*> AddBlockInstantiation(
      std::string_view instantiation_name, Block* instantiated_block);

  struct BlockInstantiationAndConnections {
    BlockInstantiation* instantiation;
    absl::flat_hash_map<std::string, InstantiationInput*> inputs;
    absl::flat_hash_map<std::string, InstantiationOutput*> outputs;
  };
  // Adds an instantiation of `instantiated_block` and creates the associated
  // InstantiationConnection nodes. Reset and clock are automatically connected,
  // if applicable. `inputs` includes the values to connect to the instantiation
  // inputs and should contain a value for every non-reset input port on
  // `instantiated_block`.
  absl::StatusOr<BlockInstantiationAndConnections>
  AddAndConnectBlockInstantiation(
      std::string_view instantiation_name, Block* instantiated_block,
      const absl::flat_hash_map<std::string, Node*>& inputs);

  struct DelayLineInstantiationAndConnections {
    DelayLineInstantiation* instantiation;
    InstantiationInput* data_input;
    InstantiationOutput* data_output;
    std::optional<InstantiationInput*> reset;
  };
  // Adds a delay line instantiation and creates the associated
  // InstantiationConnection nodes. Reset and clock are automatically connected,
  // if applicable. `input` is the data input to the delay line.
  absl::StatusOr<DelayLineInstantiationAndConnections>
  AddAndConnectDelayLineInstantiation(
      std::string_view instantiation_name, int64_t latency, Node* input,
      std::optional<std::string_view> channel = std::nullopt);

  // Add an instantiation of a FIFO to this block. InstantiationInput and
  // InstantiationOutput operations must be later added to connect the
  // instantiation to the data-flow graph.
  absl::StatusOr<FifoInstantiation*> AddFifoInstantiation(
      std::string_view name, FifoConfig fifo_config, Type* data_type,
      std::optional<std::string_view> channel = std::nullopt);
  absl::StatusOr<DelayLineInstantiation*> AddDelayLineInstantiation(
      std::string_view name, int64_t latency, Type* data_type,
      std::optional<std::string_view> channel = std::nullopt,
      std::optional<ResetBehavior> reset_behavior = std::nullopt);

  absl::StatusOr<Instantiation*> AddInstantiation(
      std::string_view name, std::unique_ptr<Instantiation> instantiation);

  // Removes the given instantiation from the block. InstantiationInput or
  // InstantiationOutput operations for this instantiation should be removed
  // prior to calling this method
  absl::Status RemoveInstantiation(Instantiation* instantiation);

  // Removes an instantiation and all of the associated InstantiationConnection
  // nodes. These nodes must have no users.
  absl::Status RemoveInstantiationAndConnections(Instantiation* instantiation);

  // Replaces all uses of old_isnt with new_inst and removes old_inst. Both
  // must be currently owned by this block.
  // If new_inst has different port names, an optional mapping of old to new
  // port names can be provided.
  absl::Status ReplaceInstantiationWith(
      Instantiation* old_inst, Instantiation* new_inst,
      absl::flat_hash_map<std::string, std::string> port_renaming = {});

  // Returns all instantiations owned by this block.
  absl::Span<Instantiation* const> GetInstantiations() const {
    return instantiation_vec_;
  }

  // Return the instantiation owned by this block with the given name or an
  // error if no such one exists.
  absl::StatusOr<Instantiation*> GetInstantiation(std::string_view name) const;

  // Returns the instantiation inputs/outputs associated with the given
  // instantiation.
  absl::Span<InstantiationInput* const> GetInstantiationInputs(
      Instantiation* instantiation) const;
  absl::Span<InstantiationOutput* const> GetInstantiationOutputs(
      Instantiation* instantiation) const;

  // Returns the instantiation input/output node associated with the port of the
  // given name.
  absl::StatusOr<InstantiationOutput*> GetInstantiationOutput(
      Instantiation* instantiation, std::string_view port_name) const;
  absl::StatusOr<InstantiationInput*> GetInstantiationInput(
      Instantiation* instantiation, std::string_view port_name) const;

  // Returns true if the given block-scoped construct (register or
  // instantiation) is owned by this block.
  bool IsOwned(Register* reg) const {
    return registers_.contains(reg->name()) &&
           registers_.at(reg->name()).get() == reg;
  }
  bool IsOwned(Instantiation* instantiation) const {
    return instantiations_.contains(instantiation->name()) &&
           instantiations_.at(instantiation->name()).get() == instantiation;
  }

  bool HasImplicitUse(Node* node) const override { return false; }

  // Creates a clone of the block with the new name 'new_name'.
  // reg_name_map is a map from old register names to new ones. If a register
  // name is not present it is an identity mapping.
  //
  // If a block is present in 'block_instantiation_map' the corresponding
  // block is used to provide an instantiation implementation. All
  // instantiated blocks must be present if target_package is not null and not
  // the existing block package.
  absl::StatusOr<Block*> Clone(
      std::string_view new_name, Package* target_package = nullptr,
      const absl::flat_hash_map<std::string, std::string>& reg_name_map = {},
      const absl::flat_hash_map<const Block*, Block*>& block_instantiation_map =
          {}) const;

  std::string DumpIr() const override;

  // Add metadata describing the mapping from ports to the channel they are
  // derived from.
  absl::Status AddChannelPortMetadata(ChannelPortMetadata metadata);
  absl::Status AddChannelPortMetadata(
      ChannelRef channel, ChannelDirection direction,
      std::optional<std::string> data_port,
      std::optional<std::string> valid_port,
      std::optional<std::string> ready_port,
      std::optional<int64_t> stage = std::nullopt);

  // Returns the port metadata for the channel with the given name or an error
  // if no such metadata exists.
  absl::StatusOr<ChannelPortMetadata> GetChannelPortMetadata(
      std::string_view channel_name, ChannelDirection direction) const;
  bool HasChannelPortMetadata(std::string_view channel_name,
                              ChannelDirection direction) const {
    return channel_port_metadata_.contains(
        std::pair<std::string, ChannelDirection>(channel_name, direction));
  }
  // Returns the port node associated with the ready/valid/data signal for the
  // given channel. Returns an error if no port metadata exists for the given
  // channel. Returns std::nullopt if port metadata exists for the channel but
  // no ready/valid/data port exists for the channel (for example, a data port
  // for a empty tuple typed channel).
  absl::StatusOr<std::optional<PortNode*>> GetReadyPortForChannel(
      std::string_view channel_name, ChannelDirection direction) const;
  absl::StatusOr<std::optional<PortNode*>> GetValidPortForChannel(
      std::string_view channel_name, ChannelDirection direction) const;
  absl::StatusOr<std::optional<PortNode*>> GetDataPortForChannel(
      std::string_view channel_name, ChannelDirection direction) const;

  // Returns the FifoInstantiation reresenting the channel with the given name.
  absl::StatusOr<Instantiation*> GetInstantiationForChannel(
      std::string_view channel_name) const;

  // Returns the instantiation input/output associated with teh ready/valid/data
  // signal for the given channel in the given direction.
  absl::StatusOr<std::optional<InstantiationConnection*>>
  GetReadyInstantiationConnectionForChannel(std::string_view channel_name,
                                            ChannelDirection direction) const;
  absl::StatusOr<std::optional<InstantiationConnection*>>
  GetValidInstantiationConnectionForChannel(std::string_view channel_name,
                                            ChannelDirection direction) const;
  absl::StatusOr<std::optional<InstantiationConnection*>>
  GetDataInstantiationConnectionForChannel(std::string_view channel_name,
                                           ChannelDirection direction) const;

  // Returns the names of and directions of channels which correspond to ports
  // on this block.
  std::vector<std::pair<std::string, ChannelDirection>>
  GetChannelsWithMappedPorts() const;

  void SetFunctionBaseProvenance(FunctionBase* fb) {
    BlockProvenanceKind kind;
    if (fb->IsFunction()) {
      kind = BlockProvenanceKind::kFunction;
    } else {
      CHECK(fb->IsProc());
      kind = BlockProvenanceKind::kProc;
    }
    SetProvenance(BlockProvenance{.name = fb->name(), .kind = kind});
  }

  void SetProvenance(std::optional<BlockProvenance> value) {
    provenance_ = value;
  }

  const std::optional<BlockProvenance>& GetProvenance() const {
    return provenance_;
  }

  const std::optional<verilog::ModuleSignatureProto>& GetSignature() const {
    return signature_;
  }
  void SetSignature(verilog::ModuleSignatureProto signature) {
    signature_ = std::move(signature);
  }

 private:
  // Sets the name of the given port node (InputPort or OutputPort) to the
  // given name. Unlike xls::Node::SetName which may name the node `name` with
  // an added suffix to ensure name uniqueness, SetNamePortExactly ensures the
  // given node is assigned the given name. This is useful for ports because
  // the port name is part of the interface of the generated Verilog module.
  // To avoid name collisions, if another *non-port* node already exists with
  // the name in the block, then that is node given an alternative name. If
  // another *port* node already exists with the name an error is returned.
  //
  // Fundamentally, port nodes have hard constraints on their name while
  // naming of interior nodes is best-effort. The problem this function solves
  // is if a node in the best-effort-naming category captures a name which is
  // required by a node in the hard-constraint-naming category.
  absl::Status SetPortNameExactly(std::string_view name, Node* port_node);

  Node* AddNodeInternal(std::unique_ptr<Node> node) override;

  // Returns the order of emission of block nodes in the text IR
  // (Block::DumpIR() output). The order is a topological sort with additional
  // constraints for readability such that logical pipeline stages tend to get
  // emitted together.
  std::vector<Node*> DumpOrder() const;

  absl::StatusOr<const ChannelPortMetadata*> GetChannelPortMetadataInternal(
      std::string_view channel_name, ChannelDirection direction) const;

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

  // Register read and write operations associated with each register.
  absl::flat_hash_map<Register*, std::vector<RegisterRead*>> register_reads_;
  absl::flat_hash_map<Register*, std::vector<RegisterWrite*>> register_writes_;

  // Vector of register pointers. Ordered by register creation time. Kept in
  // sync with the registers_ map. Enables easy, stable iteration over
  // registers. With this vector, deletion of a register is O(n) with the
  // number of registers. If this is a problem, a linked list might be used
  // instead.
  std::vector<Register*> register_vec_;

  // Instantiations owned by this block. Indexed by name. Stored as
  // std::unique_ptrs for pointer stability.
  absl::flat_hash_map<std::string, std::unique_ptr<Instantiation>>
      instantiations_;

  // Instiation input and output operations associated with instantiations in
  // this block.
  absl::flat_hash_map<Instantiation*, std::vector<InstantiationInput*>>
      instantiation_inputs_;
  absl::flat_hash_map<Instantiation*, std::vector<InstantiationOutput*>>
      instantiation_outputs_;
  std::vector<Instantiation*> instantiation_vec_;

  std::optional<ClockPort> clock_port_;
  std::optional<InputPort*> reset_port_;
  std::optional<ResetBehavior> reset_behavior_;
  NameUniquer register_name_uniquer_;

  // Map from channel name and direction to the data structure describing
  // which ports are associated with the channel (ready/valid/data ports).
  absl::flat_hash_map<std::pair<std::string, ChannelDirection>,
                      ChannelPortMetadata>
      channel_port_metadata_;

  std::optional<BlockProvenance> provenance_;
  std::optional<verilog::ModuleSignatureProto> signature_;
};

}  // namespace xls

#endif  // XLS_IR_BLOCK_H_
