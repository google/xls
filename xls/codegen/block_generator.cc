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

#include "xls/codegen/block_generator.h"

#include "absl/status/status.h"
#include "xls/codegen/flattening.h"
#include "xls/codegen/module_builder.h"
#include "xls/codegen/vast.h"
#include "xls/common/logging/log_lines.h"
#include "xls/ir/node_iterator.h"
#include "xls/ir/node_util.h"

namespace xls {
namespace verilog {
namespace {

// Not all nodes have direct representations in the Verilog. To handle these
// cases, use an absl::variant type which holds one of the two possible
// representations for a Node:
//
//  1) Expression* : node is represented directly by a Verilog expression. This
//     is the common case.
//
//  2) UnrepresentedSentinel : node has no representation in the Verilog. For
//     example, the node emits a token type.
struct UnrepresentedSentinel {};
using NodeRepresentation = absl::variant<UnrepresentedSentinel, Expression*>;

// Returns true if the given type is representable in the Verilog.
bool IsRepresentable(Type* type) {
  return !TypeHasToken(type) && type->GetFlatBitCount() > 0;
}

// Return the Verilog representation for the given node which has at least one
// operand which is not represented by an Expression*.
absl::StatusOr<NodeRepresentation> CodegenNodeWithUnrepresentedOperands(
    Node* node, ModuleBuilder* mb,
    const absl::flat_hash_map<Node*, NodeRepresentation>& node_exprs) {
  if (node->Is<xls::Assert>() && node->Is<xls::Cover>()) {
    // Asserts are statements, not expressions, and are emitted after all other
    // operations.
    return UnrepresentedSentinel();
  } else if (node->Is<Tuple>()) {
    // A tuple may have unrepresentable inputs such as empty tuples.  Walk
    // through and gather non-zero-width inputs and flatten them.
    std::vector<Expression*> nonempty_elements;
    for (Node* operand : node->operands()) {
      if (!absl::holds_alternative<Expression*>(node_exprs.at(operand))) {
        if (operand->GetType()->GetFlatBitCount() != 0) {
          return absl::UnimplementedError(absl::StrFormat(
              "Unable to generate code for: %s", node->ToString()));
        }
        continue;
      }
      nonempty_elements.push_back(
          absl::get<Expression*>(node_exprs.at(operand)));
    }
    return FlattenTuple(nonempty_elements, node->GetType()->AsTupleOrDie(),
                        mb->file());
  }
  return absl::UnimplementedError(
      absl::StrFormat("Unable to generate code for: %s", node->ToString()));
}

// Generates logic for the given nodes which must be in
// topological sort order. The map node_exprs should contain the
// representations for any nodes which occur before the given nodes (e.g.,
// receive nodes or parameters). The Verilog representations (e.g.,
// Expression*) for each of the nodes is added to the map.
absl::Status GenerateLogic(
    Block* block, ModuleBuilder* mb,
    absl::flat_hash_map<Node*, NodeRepresentation>* node_exprs,
    absl::flat_hash_map<std::string, ModuleBuilder::Register>* registers) {
  for (Node* node : TopoSort(block)) {
    XLS_VLOG(1) << "Generating expression for: " << node->GetName();

    // Ports are handled elsewhere for deterministic insertion of ports.
    if (node->Is<InputPort>() || node->Is<OutputPort>()) {
      continue;
    }

    if (!IsRepresentable(node->GetType())) {
      (*node_exprs)[node] = UnrepresentedSentinel();
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

    if (node->Is<RegisterWrite>()) {
      RegisterWrite* reg_write = node->As<RegisterWrite>();
      registers->at(reg_write->register_name()).next =
          absl::get<Expression*>(node_exprs->at(reg_write->data()));
      if (reg_write->load_enable().has_value()) {
        registers->at(reg_write->register_name()).load_enable =
            absl::get<Expression*>(
                node_exprs->at(reg_write->load_enable().value()));
      }
      (*node_exprs)[node] = UnrepresentedSentinel();
      continue;
    }
    if (node->Is<RegisterRead>()) {
      RegisterRead* reg_read = node->As<RegisterRead>();
      (*node_exprs)[node] = registers->at(reg_read->register_name()).ref;
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

    if (std::any_of(
            node->operands().begin(), node->operands().end(), [&](Node* n) {
              return !absl::holds_alternative<Expression*>(node_exprs->at(n));
            })) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Unable to generate code for node %s, has unrepresentable operand",
          node->GetName()));
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

// Return a ResetProto representing the reset signal of the block. Requires that
// any register with a reset value have identical reset behavior
// (asynchronous/synchronous, and active high/low).
absl::StatusOr<absl::optional<ResetProto>> GetBlockResetProto(Block* block) {
  absl::optional<ResetProto> reset_proto;
  for (Node* node : block->nodes()) {
    if (node->Is<RegisterWrite>()) {
      RegisterWrite* reg_write = node->As<RegisterWrite>();
      if (!reg_write->reset().has_value()) {
        continue;
      }
      Node* reset_signal = reg_write->reset().value();
      XLS_ASSIGN_OR_RETURN(Register * reg,
                           block->GetRegister(reg_write->register_name()));
      XLS_RET_CHECK(reg->reset().has_value());
      if (reset_proto.has_value()) {
        if (reset_proto->name() != reset_signal->GetName()) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Block uses more than one reset signal: %s and %s",
              reset_proto->name(), reset_signal->GetName()));
        }
        if (reset_proto->asynchronous() != reg->reset()->asynchronous) {
          return absl::InvalidArgumentError(
              "Block has asynchronous and synchronous reset signals");
        }
        if (reset_proto->active_low() != reg->reset()->active_low) {
          return absl::InvalidArgumentError(
              "Block has active low and active high reset signals");
        }
      } else {
        reset_proto = ResetProto();
        reset_proto->set_name(reset_signal->GetName());
        reset_proto->set_asynchronous(reg->reset()->asynchronous);
        reset_proto->set_active_low(reg->reset()->active_low);
      }
    }
  }
  return reset_proto;
}

}  // namespace

absl::StatusOr<std::string> GenerateVerilog(Block* block,
                                            const CodegenOptions& options) {
  XLS_VLOG(2) << "Generating Verilog for block:";
  XLS_VLOG_LINES(2, block->DumpIr());

  VerilogFile f(options.use_system_verilog());
  absl::optional<std::string> clock_name;
  if (block->GetClockPort().has_value()) {
    clock_name = block->GetClockPort().value().name;
  } else if (!block->GetRegisters().empty()) {
    return absl::InvalidArgumentError("Block has registers but no clock port");
  }

  XLS_ASSIGN_OR_RETURN(absl::optional<ResetProto> reset_proto,
                       GetBlockResetProto(block));
  ModuleBuilder mb(block->name(), &f, options.use_system_verilog(), clock_name,
                   reset_proto);

  // Map from Node* to the Verilog expression representing its value.
  absl::flat_hash_map<Node*, NodeRepresentation> node_exprs;

  // Add input ports.
  for (const Block::Port& port : block->GetPorts()) {
    if (absl::holds_alternative<InputPort*>(port)) {
      InputPort* input_port = absl::get<InputPort*>(port);
      if (reset_proto.has_value() &&
          input_port->GetName() == reset_proto->name()) {
        // The reset signal is implicitly added by ModuleBuilder.
        node_exprs[input_port] = mb.reset().value().signal;
      } else {
        XLS_ASSIGN_OR_RETURN(
            Expression * port_expr,
            mb.AddInputPort(input_port->GetName(), input_port->GetType()));
        node_exprs[input_port] = port_expr;
      }
    }
  }

  // Declare the registers.
  const absl::optional<verilog::Reset>& reset = mb.reset();
  absl::flat_hash_map<std::string, ModuleBuilder::Register> registers;
  for (Register* reg : block->GetRegisters()) {
    XLS_VLOG(3) << "Declaring register " << reg->name();
    Expression* reset_expr = nullptr;
    if (reg->reset().has_value()) {
      XLS_RET_CHECK(reset.has_value());
      const Value& reset_value = reg->reset()->reset_value;

      // If the value is a bits type it can be emitted inline. Otherwise emit
      // as a module constant.
      if (reset_value.IsBits()) {
        reset_expr = f.Literal(reset_value.bits());
      } else {
        XLS_ASSIGN_OR_RETURN(
            reset_expr, mb.DeclareModuleConstant(
                            absl::StrCat(reg->name(), "_init"), reset_value));
      }
    }
    XLS_ASSIGN_OR_RETURN(
        registers[reg->name()],
        mb.DeclareRegister(absl::StrCat(reg->name()), reg->type(),
                           /*next=*/nullptr, reset_expr));
  }

  XLS_RETURN_IF_ERROR(GenerateLogic(block, &mb, &node_exprs, &registers));

  if (!registers.empty()) {
    std::vector<ModuleBuilder::Register> register_vec;
    for (Register* reg : block->GetRegisters()) {
      register_vec.push_back(registers.at(reg->name()));
    }
    XLS_RETURN_IF_ERROR(mb.AssignRegisters(register_vec));
  }

  // Emit all asserts together.
  for (Node* node : block->nodes()) {
    if (node->Is<xls::Assert>()) {
      xls::Assert* asrt = node->As<xls::Assert>();
      Expression* condition =
          absl::get<Expression*>(node_exprs.at(asrt->condition()));
      XLS_RETURN_IF_ERROR(
          mb.EmitAssert(asrt, condition, options.assert_format()));
    }
  }

  // Same for covers.
  for (Node* node : block->nodes()) {
    if (node->Is<xls::Cover>()) {
      xls::Cover* cover = node->As<xls::Cover>();
      Expression* condition =
          absl::get<Expression*>(node_exprs.at(cover->condition()));
      XLS_RETURN_IF_ERROR(mb.EmitCover(cover, condition));
    }
  }

  // Add the output ports.
  for (OutputPort* output_port : block->GetOutputPorts()) {
    XLS_RETURN_IF_ERROR(mb.AddOutputPort(
        output_port->GetName(), output_port->operand(0)->GetType(),
        absl::get<Expression*>(node_exprs.at(output_port->operand(0)))));
  }

  std::string text = f.Emit();

  XLS_VLOG(2) << "Verilog output:";
  XLS_VLOG_LINES(2, text);

  return text;
}

}  // namespace verilog
}  // namespace xls
