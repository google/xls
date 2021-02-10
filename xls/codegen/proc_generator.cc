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

#include "xls/codegen/proc_generator.h"

#include "absl/status/status.h"
#include "xls/codegen/module_builder.h"
#include "xls/codegen/vast.h"
#include "xls/common/logging/log_lines.h"
#include "xls/ir/node_iterator.h"

namespace xls {
namespace verilog {
namespace {

// Returns true if the given node communicates over a channel (e.g., send).
bool IsChannelNode(Node* node) {
  return node->Is<Send>() || node->Is<Receive>() || node->Is<SendIf>() ||
         node->Is<ReceiveIf>();
}

// Returns the channel used by the given node which must be a
// send/sendif/receive/receiveif node.
absl::StatusOr<Channel*> GetChannelUsedByNode(Node* node) {
  int64 channel_id;
  if (node->Is<Send>()) {
    channel_id = node->As<Send>()->channel_id();
  } else if (node->Is<Receive>()) {
    channel_id = node->As<Receive>()->channel_id();
  } else if (node->Is<SendIf>()) {
    channel_id = node->As<SendIf>()->channel_id();
  } else if (node->Is<ReceiveIf>()) {
    channel_id = node->As<ReceiveIf>()->channel_id();
  } else {
    return absl::InvalidArgumentError(
        absl::StrFormat("No channel associated with node %s", node->GetName()));
  }
  return node->package()->GetChannel(channel_id);
}

// Simple record holding a channel and a node associated with the channel
// (Send(If) or Receive(If)).
struct ChannelNode {
  Channel* channel;
  Node* node;
};

// Returns all input ports of the proc. An input port is represented with a
// Receive node on a PortChannel.
absl::StatusOr<std::vector<ChannelNode>> GetInputPorts(Proc* proc) {
  std::vector<ChannelNode> ports;
  for (Node* node : proc->nodes()) {
    // Port channels should only have send/receive nodes, not
    // send_if/receive_if.
    if (node->Is<Receive>()) {
      XLS_ASSIGN_OR_RETURN(Channel * channel, GetChannelUsedByNode(node));
      if (channel->IsPort()) {
        ports.push_back(ChannelNode{channel, node});
      }
    }
  }
  std::sort(ports.begin(), ports.end(),
            [](const ChannelNode& a, const ChannelNode& b) {
              return a.channel->id() < b.channel->id();
            });
  return ports;
}

// Returns all output ports of the proc. An output port is represented with a
// Send node on a PortChannel.
absl::StatusOr<std::vector<ChannelNode>> GetOutputPorts(Proc* proc) {
  std::vector<ChannelNode> ports;
  for (Node* node : proc->nodes()) {
    // Port channels should only have send/receive nodes, not
    // send_if/receive_if.
    if (node->Is<Send>()) {
      XLS_ASSIGN_OR_RETURN(Channel * channel, GetChannelUsedByNode(node));
      if (channel->IsPort()) {
        ports.push_back(ChannelNode{channel, node});
      }
    }
  }
  std::sort(ports.begin(), ports.end(),
            [](const ChannelNode& a, const ChannelNode& b) {
              return a.channel->id() < b.channel->id();
            });
  return ports;
}

// Record holding a register channel and the associated send/receive nodes as
// well as the ModuleBuilder-level abstraction for a register.
struct RegisterInfo {
  RegisterChannel* channel = nullptr;
  Node* receive = nullptr;
  Node* send = nullptr;
  ModuleBuilder::Register mb_register;
};

// Returns the registers in the proc.
absl::StatusOr<std::vector<RegisterInfo>> GetRegisters(Proc* proc) {
  absl::flat_hash_map<Channel*, RegisterInfo> registers;
  for (Node* node : proc->nodes()) {
    if (node->Is<Send>() || node->Is<SendIf>()) {
      XLS_ASSIGN_OR_RETURN(Channel * channel, GetChannelUsedByNode(node));
      if (channel->IsRegister()) {
        registers[channel].channel = down_cast<RegisterChannel*>(channel);
        registers[channel].send = node;
      }
    }
    // Registers can only have Receive nodes not ReceiveIf.
    if (node->Is<Receive>()) {
      XLS_ASSIGN_OR_RETURN(Channel * channel, GetChannelUsedByNode(node));
      if (channel->IsRegister()) {
        registers[channel].channel = down_cast<RegisterChannel*>(channel);
        registers[channel].receive = node;
      }
    }
  }

  std::vector<RegisterInfo> register_vec;
  for (auto [channel, reg] : registers) {
    register_vec.push_back(reg);
  }
  std::sort(register_vec.begin(), register_vec.end(),
            [](const RegisterInfo& a, const RegisterInfo& b) {
              return a.channel->id() < b.channel->id();
            });

  // Verify that each channel has a send and receive node.
  for (const RegisterInfo& reg : register_vec) {
    if (reg.receive == nullptr) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Register channel %s has no receive node", reg.channel->name()));
    }
    if (reg.send == nullptr) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Register channel %s has no send node", reg.channel->name()));
    }
  }
  return register_vec;
}

// Not all nodes have direct representations in the Verilog. To handle these
// cases, use an absl::variant type which holds the various possible
// representations for a Node:
//
//  1) Expression* : node is represented directly by a Verilog expression. This
//     is the common case.
//
//  2) UnrepresentedSentinel : node has no representation in the Verilog. For
//     example, the node emits a token type.
//
//  3) ReceiveData : node is a receive node. The result of a receive node is
//     a token (UnrepresentedSentinel) and the received data (Expression*).
struct UnrepresentedSentinel {};
using ReceiveData = std::pair<UnrepresentedSentinel, Expression*>;
using NodeRepresentation =
    absl::variant<Expression*, UnrepresentedSentinel, ReceiveData>;

// Returns true if the given type is an unrepresented type in the Verilog
// (e.g., a token) or has as a unrepresented type as a subelement.
bool HasUnrepresentedType(Type* type) {
  if (type->IsToken() || type->GetFlatBitCount() == 0) {
    return true;
  }
  if (type->IsTuple()) {
    return std::any_of(type->AsTupleOrDie()->element_types().begin(),
                       type->AsTupleOrDie()->element_types().end(),
                       HasUnrepresentedType);
  }
  if (type->IsArray()) {
    return HasUnrepresentedType(type->AsArrayOrDie()->element_type());
  }
  return false;
}

// Return the Verilog representation for the given node which has at least one
// operand which is not represented by an Expression*.
absl::StatusOr<NodeRepresentation> CodegenNodeWithUnrepresentedOperands(
    Node* node, ModuleBuilder* mb,
    const absl::flat_hash_map<Node*, NodeRepresentation>& node_exprs) {
  if (node->Is<TupleIndex>() && node->operand(0)->Is<Receive>()) {
    // A tuple-index into a receive node.
    XLS_RET_CHECK(
        absl::holds_alternative<ReceiveData>(node_exprs.at(node->operand(0))));
    TupleIndex* tuple_index = node->As<TupleIndex>();
    const ReceiveData& receive_data =
        absl::get<ReceiveData>(node_exprs.at(node->operand(0)));
    if (tuple_index->index() == 0) {
      return receive_data.first;
    } else {
      XLS_RET_CHECK_EQ(tuple_index->index(), 1);
      return receive_data.second;
    }
  } else if (node->Is<xls::Assert>()) {
    // Asserts are statements, not expressions, and are emitted after all other
    // operations.
    return UnrepresentedSentinel();
  }
  // TODO(meheff): Tuples not from receive nodes which contains tokens will
  // return an error here. Implement a more general solution.
  return absl::UnimplementedError(
      absl::StrFormat("Unable to generate code for: %s", node->ToString()));
}

// Generates combinational logic for the given nodes which must be in
// topological sort order. The map node_exprs should contain the
// representations for any nodes which occur before the given nodes (e.g.,
// receive nodes or parameters). The Verilog representations (e.g.,
// Expression*) for each of the nodes is added to the map.
absl::Status GenerateCombinationalLogic(
    absl::Span<Node* const> nodes, ModuleBuilder* mb,
    absl::flat_hash_map<Node*, NodeRepresentation>* node_exprs) {
  for (Node* node : nodes) {
    XLS_VLOG(1) << "Generating expression for: " << node->GetName();
    if (HasUnrepresentedType(node->GetType())) {
      (*node_exprs)[node] = UnrepresentedSentinel();
      continue;
    }

    // Emit non-bits-typed literals as module-level constants because in
    // general these complicated types cannot be handled inline, and
    // constructing them in Verilog may require a sequence of assignments.
    if (node->Is<xls::Literal>() && !node->GetType()->IsBits()) {
      XLS_ASSIGN_OR_RETURN(
          (*node_exprs)[node],
          mb->DeclareModuleConstant(node->GetName(),
                                    node->As<xls::Literal>()->value()));
      continue;
    }

    // If any of the operands do not have an Expression* representation then
    // handle the node specially.
    if (std::any_of(
            node->operands().begin(), node->operands().end(), [&](Node* n) {
              return !absl::holds_alternative<Expression*>(node_exprs->at(n));
            })) {
      XLS_ASSIGN_OR_RETURN(
          (*node_exprs)[node],
          CodegenNodeWithUnrepresentedOperands(node, mb, *node_exprs));
      continue;
    }

    std::vector<Expression*> inputs;
    for (Node* operand : node->operands()) {
      inputs.push_back(absl::get<Expression*>(node_exprs->at(operand)));
    }

    // If the node has an assigned name then don't emit as an inline
    // expression. This ensures the name appears in the generated Verilog.
    if (node->HasAssignedName() || node->users().size() > 1 ||
        node->function_base()->HasImplicitUse(node) ||
        !mb->CanEmitAsInlineExpression(node)) {
      XLS_ASSIGN_OR_RETURN((*node_exprs)[node],
                           mb->EmitAsAssignment(node->GetName(), node, inputs));
    } else {
      XLS_ASSIGN_OR_RETURN((*node_exprs)[node],
                           mb->EmitAsInlineExpression(node, inputs));
    }
  }
  return absl::OkStatus();
}

// Verifies various preconditions which must hold for a proc at codegen time.
absl::Status VerifyProcForCodegen(Proc* proc) {
  // The only supported channels are PortChannels and RegisterChannels.
  for (Node* node : proc->nodes()) {
    if (IsChannelNode(node)) {
      XLS_ASSIGN_OR_RETURN(Channel * ch, GetChannelUsedByNode(node));
      if (!ch->IsPort() && !ch->IsRegister()) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Only register and port channel are supported in "
                            "codegen. Node %s communicates on channel %s",
                            node->GetName(), ch->ToString()));
      }
    }
  }

  // At this point in codegen, the state of the proc must be an empty tuple. Any
  // meaningful state must have been converted to a register via, for example,
  // StateRemovalPass.
  if (proc->StateType() != proc->package()->GetTupleType({})) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "The proc state must be an empty tuple for codegen, is type: %s",
        proc->StateType()->ToString()));
  }

  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<ModuleGeneratorResult> GenerateModule(
    Proc* proc, const GeneratorOptions& options) {
  XLS_VLOG(2) << "Generating combinational module for proc:";
  XLS_VLOG_LINES(2, proc->DumpIr());

  XLS_RETURN_IF_ERROR(VerifyProcForCodegen(proc));

  VerilogFile f;
  ModuleBuilder mb(proc->name(), &f,
                   /*use_system_verilog=*/options.use_system_verilog());

  XLS_ASSIGN_OR_RETURN(std::vector<ChannelNode> input_ports,
                       GetInputPorts(proc));
  XLS_ASSIGN_OR_RETURN(std::vector<ChannelNode> output_ports,
                       GetOutputPorts(proc));
  XLS_ASSIGN_OR_RETURN(std::vector<RegisterInfo> registers, GetRegisters(proc));

  // Build the module signature along the way.
  ModuleSignatureBuilder sig_builder(mb.module()->name());

  // Define clock and reset, if necessary.
  absl::optional<LogicRef*> clk;
  absl::optional<Reset> rst;
  if (!registers.empty()) {
    if (options.clock_name().empty()) {
      return absl::InvalidArgumentError(
          "Must specify clock name for proc containing registers.");
    }
  }
  if (!options.clock_name().empty()) {
    clk = mb.AddInputPort(options.clock_name(), /*bit_count=*/1);
    sig_builder.WithClock(options.clock_name());
  }
  if (options.reset().has_value()) {
    sig_builder.WithReset(options.reset()->name(),
                          options.reset()->asynchronous(),
                          options.reset()->active_low());
    rst = Reset{
        .signal = mb.AddInputPort(options.reset()->name(), /*bit_count=*/1)
                      ->AsLogicRefNOrDie<1>(),
        .asynchronous = options.reset()->asynchronous(),
        .active_low = options.reset()->active_low()};
  }

  // Map from Node* to the Verilog expression representing its value.
  absl::flat_hash_map<Node*, NodeRepresentation> node_exprs;

  // Add the input ports.
  for (const ChannelNode& cn : input_ports) {
    if (cn.node->Is<Receive>()) {
      std::vector<Expression*> ports;
      XLS_ASSIGN_OR_RETURN(
          Expression * port,
          mb.AddInputPort(cn.channel->name(), cn.channel->type()));
      sig_builder.AddDataInput(cn.channel->name(),
                               cn.channel->type()->GetFlatBitCount());
      node_exprs[cn.node] = ReceiveData{UnrepresentedSentinel(), port};
    }
  }

  // Declare the registers.
  for (RegisterInfo& reg : registers) {
    XLS_VLOG(1) << "Declaring register for channel: "
                << reg.channel->ToString();
    XLS_VLOG(1) << "  receive: " << reg.receive->GetName();
    XLS_VLOG(1) << "  send: " << reg.send->GetName();
    absl::optional<Expression*> reset_expr;
    if (reg.channel->reset_value().has_value()) {
      if (!rst.has_value()) {
        return absl::InvalidArgumentError(
            "Must specify a reset signal if registers have a reset value "
            "(RegisterChannel initial values are not empty)");
      }

      // If the value is a bits type it can be emitted inline. Otherwise emit
      // as a module constant.
      if (reg.channel->reset_value()->IsBits()) {
        reset_expr = f.Literal(reg.channel->reset_value()->bits());
      } else {
        XLS_ASSIGN_OR_RETURN(
            reset_expr,
            mb.DeclareModuleConstant(absl::StrCat(reg.channel->name(), "_init"),
                                     reg.channel->reset_value().value()));
      }
    }
    XLS_ASSIGN_OR_RETURN(
        reg.mb_register,
        mb.DeclareRegister(absl::StrCat(reg.channel->name()),
                           reg.channel->type(),
                           /*next=*/absl::nullopt, reset_expr));
    node_exprs[reg.receive] =
        ReceiveData{UnrepresentedSentinel(), reg.mb_register.ref};
  }

  // Generate list of nodes to emit as combinational logic.
  std::vector<Node*> nodes;
  for (Node* node : TopoSort(proc)) {
    if (node->Is<Param>() || IsChannelNode(node)) {
      continue;
    }
    nodes.push_back(node);
  }
  XLS_RETURN_IF_ERROR(GenerateCombinationalLogic(nodes, &mb, &node_exprs));

  // Assign the next values to the registers. These are the data operands of
  // the Receive node of the operand.
  for (RegisterInfo& reg : registers) {
    if (reg.send->Is<SendIf>()) {
      return absl::UnimplementedError(
          "SendIf to register channels not supported yet.");
    }

    reg.mb_register.next =
        absl::get<Expression*>(node_exprs.at(reg.send->As<Send>()->data()));
    XLS_RETURN_IF_ERROR(mb.AssignRegisters(clk.value(), {reg.mb_register},
                                           /*load_enable=*/nullptr, rst));
  }

  // Add assert statements in a separate always_comb block.
  for (Node* node : proc->nodes()) {
    if (node->Is<xls::Assert>()) {
      xls::Assert* asrt = node->As<xls::Assert>();
      Expression* condition =
          absl::get<Expression*>(node_exprs.at(asrt->condition()));
      XLS_RETURN_IF_ERROR(mb.EmitAssert(asrt, condition));
    }
  }

  // Add the output ports.
  for (const ChannelNode& cn : output_ports) {
    if (cn.node->Is<Send>()) {
      Send* send = cn.node->As<Send>();
      XLS_RETURN_IF_ERROR(mb.AddOutputPort(
          cn.channel->name(), cn.channel->type(),
          absl::get<Expression*>(node_exprs.at(send->data()))));
      sig_builder.AddDataOutput(cn.channel->name(),
                                cn.channel->type()->GetFlatBitCount());
    }
  }

  // Build the signature. Use "unknown" interface because the proc may have an
  // arbitrary interface/protocol.
  sig_builder.WithUnknownInterface();
  XLS_ASSIGN_OR_RETURN(ModuleSignature signature, sig_builder.Build());

  std::string text = f.Emit();

  XLS_VLOG(2) << "Verilog output:";
  XLS_VLOG_LINES(2, text);
  XLS_VLOG(2) << "Signature:";
  XLS_VLOG_LINES(2, signature.ToString());

  return ModuleGeneratorResult{text, signature};
}

}  // namespace verilog
}  // namespace xls
