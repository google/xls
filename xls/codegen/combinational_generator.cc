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

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "xls/codegen/flattening.h"
#include "xls/codegen/function_to_proc.h"
#include "xls/codegen/module_builder.h"
#include "xls/codegen/node_expressions.h"
#include "xls/codegen/proc_generator.h"
#include "xls/codegen/vast.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/function.h"
#include "xls/ir/node.h"
#include "xls/ir/node_iterator.h"
#include "xls/ir/nodes.h"
#include "xls/ir/type.h"

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
  if (node->Is<TupleIndex>() &&
      (node->operand(0)->Is<Receive>() || node->operand(0)->Is<ReceiveIf>())) {
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
  XLS_ASSIGN_OR_RETURN(
      Proc * proc,
      FunctionToProc(func, absl::StrCat("__", func->name(), "_proc")));
  std::string module_name = SanitizeIdentifier(func->name());
  XLS_ASSIGN_OR_RETURN(
      ModuleGeneratorResult result,
      GenerateModule(proc, GeneratorOptions()
                               .module_name(module_name)
                               .use_system_verilog(use_system_verilog)));

  // Generate a signature for the module. ProcGenerate currently generates a
  // signature with "Unknown" interface type.
  // TODO(meheff): 2021/03/01 Remove this when proc builder can generate more
  // expressive signatures.
  ModuleSignatureBuilder sig_builder(module_name);
  for (Param* param : func->params()) {
    sig_builder.AddDataInput(param->name(),
                             param->GetType()->GetFlatBitCount());
  }
  sig_builder.AddDataOutput("out",
                            func->return_value()->GetType()->GetFlatBitCount());
  sig_builder.WithFunctionType(func->GetType());
  sig_builder.WithCombinationalInterface();
  XLS_ASSIGN_OR_RETURN(ModuleSignature signature, sig_builder.Build());

  return ModuleGeneratorResult{result.verilog_text, signature};
}

absl::StatusOr<ModuleGeneratorResult> GenerateCombinationalModuleFromProc(
    Proc* proc,
    const absl::flat_hash_map<const Channel*, ProcPortType>& channel_gen_types,
    bool use_system_verilog) {
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

  ModuleSignatureBuilder sig_builder(mb.module()->name());
  sig_builder.WithCombinationalInterface();

  // Gather the send/receive nodes and their associated channels.
  struct ChannelNode {
    Node* node;
    Channel* channel;
  };
  std::vector<ChannelNode> channel_nodes;
  for (Node* node : proc->nodes()) {
    if (node->Is<Send>() || node->Is<Receive>() || node->Is<SendIf>() ||
        node->Is<ReceiveIf>()) {
      XLS_ASSIGN_OR_RETURN(Channel * channel, GetChannelUsedByNode(node));
      XLS_CHECK(channel_gen_types.contains(channel));
      if (!channel->IsPort()) {
        return absl::UnimplementedError(absl::StrFormat(
            "Only port channels supported: %s", channel->ToString()));
      }
      if (channel->metadata().module_port().flopped()) {
        return absl::UnimplementedError(absl::StrFormat(
            "Flopped module ports not supported in combinational generator: %s",
            channel->ToString()));
      }
      channel_nodes.push_back(ChannelNode{node, channel});
      continue;
    }
  }

  xls::Type* single_bit_type = proc->package()->GetBitsType(1);

  // Generate list of nodes to emit as combinational logic.
  std::vector<Node*> nodes;
  for (Node* node : TopoSort(proc)) {
    if (node->Is<Param>() || node->Is<Send>() || node->Is<SendIf>() ||
        node->Is<Receive>() || node->Is<ReceiveIf>()) {
      continue;
    }
    nodes.push_back(node);
  }

  // Map from Node* to the Verilog expression representing its value.
  absl::flat_hash_map<Node*, NodeRepresentation> node_exprs;

  // Add the data input ports
  for (const ChannelNode& cn : channel_nodes) {
    if (cn.node->Is<Receive>() || cn.node->Is<ReceiveIf>()) {
      XLS_ASSIGN_OR_RETURN(
          Expression * input_port,
          mb.AddInputPort(cn.channel->name(), cn.channel->type()));

      node_exprs[cn.node] = ReceiveData({input_port});

      sig_builder.AddDataInput(cn.channel->name(),
                               cn.channel->type()->GetFlatBitCount());
    }
  }

  XLS_RETURN_IF_ERROR(GenerateCombinationalLogic(nodes, &mb, &node_exprs));

  // These expressions determine whether or not an operation "blocks"
  //  the proc from processing / generating IO transactions.
  // The form ~(!pred) is used because if predicate is 0 then the IO operation
  //  won't block. If the predicate is 1, then it can block depending on the
  //  the input ready/valids.
  // If there is no external ready/valid signal to block, then the predicate
  //  doesn't matter to whether or not processing blocks.
  std::vector<verilog::Expression*> sends_blocking;
  std::vector<verilog::Expression*> receives_blocking;

  // Add the rdy/vld inputs
  for (const ChannelNode& cn : channel_nodes) {
    XLS_CHECK(channel_gen_types.contains(cn.channel));
    const ProcPortType port_type = channel_gen_types.at(cn.channel);

    // Non ready/valid channels don't block
    if (port_type == ProcPortType::kSimple) continue;

    XLS_CHECK(port_type == ProcPortType::kReadyValid);

    // Get the predicate
    verilog::Expression* pred_expr = nullptr;

    if (cn.node->Is<ReceiveIf>() || cn.node->Is<SendIf>()) {
      Node* pred_node = nullptr;
      if (cn.node->Is<ReceiveIf>()) {
        ReceiveIf* recv = cn.node->As<ReceiveIf>();
        pred_node = recv->predicate();
      } else {
        SendIf* send = cn.node->As<SendIf>();
        pred_node = send->predicate();
      }
      pred_expr = absl::get<Expression*>(node_exprs.at(pred_node));
    }

    // Create any external blocking signals

    const bool is_receive = cn.node->Is<ReceiveIf>() || cn.node->Is<Receive>();
    XLS_CHECK(is_receive || cn.node->Is<SendIf>() || cn.node->Is<Send>());

    const char* ch_postfix = is_receive ? "_vld" : "_rdy";

    XLS_ASSIGN_OR_RETURN(
        Expression * external_not_blocking,
        mb.AddInputPort(cn.channel->name() + ch_postfix, single_bit_type));
    sig_builder.AddDataInput(cn.channel->name() + ch_postfix, 1);

    Expression* not_blocking = nullptr;

    if (pred_expr) {
      not_blocking =
          f.BitwiseOr(f.BitwiseNot(pred_expr), external_not_blocking);
    } else {
      not_blocking = external_not_blocking;
    }

    if (is_receive) {
      receives_blocking.push_back(not_blocking);
    } else {
      sends_blocking.push_back(not_blocking);
    }
  }

  verilog::LogicRef* all_active_outputs_ready = nullptr;

  if (!sends_blocking.empty()) {
    all_active_outputs_ready =
        mb.DeclareVariable("all_active_outputs_ready", single_bit_type);
    XLS_RETURN_IF_ERROR(mb.Assign(all_active_outputs_ready,
                                  f.AndReduce(f.Concat(sends_blocking)),
                                  single_bit_type));
  }

  verilog::LogicRef* all_active_inputs_valid = nullptr;

  if (!receives_blocking.empty()) {
    all_active_inputs_valid =
        mb.DeclareVariable("all_active_inputs_valid", single_bit_type);
    XLS_RETURN_IF_ERROR(mb.Assign(all_active_inputs_valid,
                                  f.AndReduce(f.Concat(receives_blocking)),
                                  single_bit_type));
  }

  // Add the outputs
  for (const ChannelNode& cn : channel_nodes) {
    // Data output if any kind of send
    if (cn.node->Is<Send>() || cn.node->Is<SendIf>()) {
      Expression* out_value = nullptr;

      if (cn.node->Is<Send>()) {
        Send* send = cn.node->As<Send>();
        out_value = absl::get<Expression*>(node_exprs.at(send->data()));
      } else {
        SendIf* sendif = cn.node->As<SendIf>();
        out_value = absl::get<Expression*>(node_exprs.at(sendif->data()));
      }

      XLS_RETURN_IF_ERROR(
          mb.AddOutputPort(cn.channel->name(), cn.channel->type(), out_value));
      sig_builder.AddDataOutput(cn.channel->name(),
                                cn.channel->type()->GetFlatBitCount());
    }

    XLS_CHECK(channel_gen_types.contains(cn.channel));
    const ProcPortType port_type = channel_gen_types.at(cn.channel);

    if (port_type != ProcPortType::kReadyValid) continue;

    // Get the predicate
    verilog::Expression* pred_expr = nullptr;

    if (cn.node->Is<ReceiveIf>() || cn.node->Is<SendIf>()) {
      Node* pred_node = nullptr;
      if (cn.node->Is<ReceiveIf>()) {
        ReceiveIf* recv = cn.node->As<ReceiveIf>();
        pred_node = recv->predicate();
      } else {
        SendIf* send = cn.node->As<SendIf>();
        pred_node = send->predicate();
      }
      pred_expr = absl::get<Expression*>(node_exprs.at(pred_node));
    }

    const bool is_receive = cn.node->Is<ReceiveIf>() || cn.node->Is<Receive>();
    XLS_CHECK(is_receive || cn.node->Is<SendIf>() || cn.node->Is<Send>());

    const char* ch_postfix = is_receive ? "_rdy" : "_vld";

    Expression* external_not_blocking =
        is_receive ? all_active_outputs_ready : all_active_inputs_valid;

    if (external_not_blocking == nullptr) {
      external_not_blocking = f.Literal(1, 1);
    }

    Expression* not_blocking = external_not_blocking;

    if (pred_expr != nullptr) {
      not_blocking = f.BitwiseAnd(pred_expr, external_not_blocking);
    }

    XLS_RETURN_IF_ERROR(mb.AddOutputPort(cn.channel->name() + ch_postfix,
                                         single_bit_type, not_blocking));
    sig_builder.AddDataOutput(cn.channel->name() + ch_postfix, 1);
  }

  XLS_ASSIGN_OR_RETURN(ModuleSignature signature, sig_builder.Build());

  std::string text = f.Emit();

  XLS_VLOG(2) << "Verilog output:";
  XLS_VLOG_LINES(2, text);

  return ModuleGeneratorResult{text, signature};
}

}  // namespace verilog
}  // namespace xls
