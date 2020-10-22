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

#include "xls/codegen/combinational_generator.h"

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "xls/codegen/flattening.h"
#include "xls/codegen/module_builder.h"
#include "xls/codegen/node_expressions.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/node.h"
#include "xls/ir/node_iterator.h"

namespace xls {
namespace verilog {
namespace {

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
//  3) ReceiveData : node is a receive node and the vector holds the expressions
//     of the received data.
struct UnrepresentedSentinel {};
using ReceiveData = std::vector<Expression*>;
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
    // A tuple-index into a receive node. Only indexing into the data elements
    // (index 1 and up) is supported.
    XLS_RET_CHECK(
        absl::holds_alternative<ReceiveData>(node_exprs.at(node->operand(0))));
    TupleIndex* tuple_index = node->As<TupleIndex>();
    const ReceiveData& receive_data =
        absl::get<ReceiveData>(node_exprs.at(node->operand(0)));
    if (tuple_index->index() > 0) {
      return receive_data.at(tuple_index->index() - 1);
    }
  }
  return absl::UnimplementedError(
      absl::StrFormat("Unable to generated combinational module containing: %s",
                      node->ToString()));
}

// Generates combinational logic for the given nodes which must be in
// topological sort order. The map node_exprs should contain the representations
// for any nodes which occur before the given nodes (e.g., receive nodes or
// parameters). The Verilog representations (e.g., Expression*) for each of the
// nodes is added to the map.
absl::Status GenerateCombinationalLogic(
    absl::Span<Node* const> nodes, ModuleBuilder* mb,
    absl::flat_hash_map<Node*, NodeRepresentation>* node_exprs) {
  for (Node* node : nodes) {
    XLS_RET_CHECK(!node->Is<Param>());

    if (HasUnrepresentedType(node->GetType())) {
      (*node_exprs)[node] = UnrepresentedSentinel();
      continue;
    }

    // Emit non-bits-typed literals as module-level constants because in general
    // these complicated types cannot be handled inline, and constructing them
    // in Verilog may require a sequence of assignments.
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

}  // namespace

absl::StatusOr<ModuleGeneratorResult> GenerateCombinationalModule(
    Function* func, bool use_system_verilog) {
  XLS_VLOG(2) << "Generating combinational module for function:";
  XLS_VLOG_LINES(2, func->DumpIr());

  VerilogFile f;
  ModuleBuilder mb(func->name(), &f, /*use_system_verilog=*/use_system_verilog);

  // Build the module signature.
  ModuleSignatureBuilder sig_builder(mb.module()->name());
  for (Param* param : func->params()) {
    sig_builder.AddDataInput(param->name(),
                             param->GetType()->GetFlatBitCount());
  }
  const int64 output_width = func->return_value()->GetType()->GetFlatBitCount();
  // Don't use the assigned name if this is a parameter or there will be ports
  // with duplicate names.
  const char kOutputPortName[] = "out";
  sig_builder.AddDataOutput(kOutputPortName, output_width);
  sig_builder.WithFunctionType(func->GetType());
  sig_builder.WithCombinationalInterface();
  XLS_ASSIGN_OR_RETURN(ModuleSignature signature, sig_builder.Build());

  // Map from Node* to the Verilog expression representing its value.
  absl::flat_hash_map<Node*, NodeRepresentation> node_exprs;

  // Add parameters explicitly so the input ports are added in the order they
  // appear in the parameters of the function.
  for (Param* param : func->params()) {
    if (param->GetType()->GetFlatBitCount() == 0) {
      XLS_RET_CHECK_EQ(param->users().size(), 0);
      continue;
    }
    XLS_ASSIGN_OR_RETURN(
        node_exprs[param],
        mb.AddInputPort(param->As<Param>()->name(), param->GetType()));
  }

  // Generate list of nodes to emit as combinational logic.
  std::vector<Node*> nodes;
  for (Node* node : TopoSort(func)) {
    if (node->Is<Param>()) {
      // Parameters are added in the above loop.
      continue;
    }

    // Verilog has no zero-bit data types so elide such types. They should have
    // no uses.
    if (node->GetType()->GetFlatBitCount() == 0) {
      XLS_RET_CHECK_EQ(node->users().size(), 0);
      continue;
    }

    // Emit non-bits-typed literals as module-level constants because in general
    // these complicated types cannot be handled inline, and constructing them
    // in Verilog may require a sequence of assignments.
    if (node->Is<xls::Literal>() && !node->GetType()->IsBits()) {
      XLS_ASSIGN_OR_RETURN(
          node_exprs[node],
          mb.DeclareModuleConstant(node->GetName(),
                                   node->As<xls::Literal>()->value()));
      continue;
    }

    if (!node->Is<Param>()) {
      nodes.push_back(node);
    }
  }
  XLS_RETURN_IF_ERROR(GenerateCombinationalLogic(nodes, &mb, &node_exprs));

  // Skip adding an output port to the Verilog module if the output is
  // zero-width.
  if (output_width > 0) {
    XLS_RETURN_IF_ERROR(mb.AddOutputPort(
        kOutputPortName, func->return_value()->GetType(),
        absl::get<Expression*>(node_exprs.at(func->return_value()))));
  }
  std::string text = f.Emit();

  XLS_VLOG(2) << "Verilog output:";
  XLS_VLOG_LINES(2, text);

  return ModuleGeneratorResult{text, signature};
}

absl::StatusOr<ModuleGeneratorResult> GenerateCombinationalModuleFromProc(
    Proc* proc, bool use_system_verilog) {
  XLS_VLOG(2) << "Generating combinational module for proc:";
  XLS_VLOG_LINES(2, proc->DumpIr());

  VerilogFile f;
  ModuleBuilder mb(proc->name(), &f, /*use_system_verilog=*/use_system_verilog);

  // In a combinational module, the proc cannot have any state to avoid
  // combinational loops. That is, the loop state must be an empty tuple.
  if (proc->StateType() != proc->package()->GetTupleType({})) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Proc must have no state (state type is empty tuple) when lowering to "
        "a combinational module. Proc state is: %s",
        proc->StateType()->ToString()));
  }

  // Gather the send/receive nodes and their associated channels.
  struct ChannelNode {
    Node* node;
    Channel* channel;
  };
  std::vector<ChannelNode> channel_nodes;
  for (Node* node : proc->nodes()) {
    if (node->Is<Send>() || node->Is<Receive>()) {
      XLS_ASSIGN_OR_RETURN(Channel * channel, GetChannelUsedByNode(node));
      if (channel->data_elements().size() != 1) {
        return absl::UnimplementedError(
            absl::StrFormat("Only single data element channels supported: %s",
                            channel->ToString()));
      }
      if (!channel->metadata().has_module_port()) {
        return absl::UnimplementedError(absl::StrFormat(
            "Only module port channels supported: %s", channel->ToString()));
      }
      if (channel->metadata().module_port().flopped()) {
        return absl::UnimplementedError(absl::StrFormat(
            "Flopped module ports not supported in combinational generator: %s",
            channel->ToString()));
      }
      channel_nodes.push_back(ChannelNode{node, channel});
      continue;
    }
    if (node->Is<SendIf>() || node->Is<ReceiveIf>()) {
      return absl::UnimplementedError(
          "SendIf/ReceiveIf not yet implemented in combinational generator");
    }
  }

  // Sort the channels by port_order field. Currently inputs are ordered among
  // inputs, and same with outputs.
  // TODO(meheff): Allow arbitrary ordering of inputs and outputs.
  auto port_order_lt = [](const ChannelNode& a, const ChannelNode& b) {
    return a.channel->metadata().module_port().port_order() <
           b.channel->metadata().module_port().port_order();
  };
  std::sort(channel_nodes.begin(), channel_nodes.end(), port_order_lt);

  ModuleSignatureBuilder sig_builder(mb.module()->name());

  // Adds the port associated with the given Channel and Node to the signature.
  auto add_port_to_signature = [&](const ChannelNode& cn) -> absl::Status {
    std::string name = cn.channel->data_element(0).name;
    int64 width = cn.channel->data_element(0).type->GetFlatBitCount();
    if (cn.node->Is<Send>()) {
      sig_builder.AddDataOutput(name, width);
    } else {
      XLS_RET_CHECK(cn.node->Is<Receive>());
      sig_builder.AddDataInput(name, width);
    }
    return absl::OkStatus();
  };

  // Map from Node* to the Verilog expression representing its value.
  absl::flat_hash_map<Node*, NodeRepresentation> node_exprs;

  // Add the input ports.
  for (const ChannelNode& cn : channel_nodes) {
    XLS_RET_CHECK_EQ(cn.channel->data_elements().size(), 1);
    if (cn.node->Is<Receive>()) {
      XLS_ASSIGN_OR_RETURN(Expression * input_port,
                           mb.AddInputPort(cn.channel->data_element(0).name,
                                           cn.channel->data_element(0).type));
      node_exprs[cn.node] = ReceiveData({input_port});
      XLS_RETURN_IF_ERROR(add_port_to_signature(cn));
    }
  }

  // Generate list of nodes to emit as combinational logic.
  std::vector<Node*> nodes;
  for (Node* node : TopoSort(proc)) {
    if (node->Is<Param>() || node->Is<Send>() || node->Is<Receive>()) {
      continue;
    }
    nodes.push_back(node);
  }
  XLS_RETURN_IF_ERROR(GenerateCombinationalLogic(nodes, &mb, &node_exprs));

  // Add the output ports.
  for (const ChannelNode& cn : channel_nodes) {
    if (cn.node->Is<Send>()) {
      Send* send = cn.node->As<Send>();
      XLS_RETURN_IF_ERROR(mb.AddOutputPort(
          cn.channel->data_element(0).name, cn.channel->data_element(0).type,
          absl::get<Expression*>(node_exprs.at(send->data_operands()[0]))));
      XLS_RETURN_IF_ERROR(add_port_to_signature(cn));
    }
  }

  sig_builder.WithCombinationalInterface();
  XLS_ASSIGN_OR_RETURN(ModuleSignature signature, sig_builder.Build());

  std::string text = f.Emit();

  XLS_VLOG(2) << "Verilog output:";
  XLS_VLOG_LINES(2, text);

  return ModuleGeneratorResult{text, signature};
}

}  // namespace verilog
}  // namespace xls
