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

#include <deque>

#include "absl/status/status.h"
#include "xls/codegen/block_conversion.h"
#include "xls/codegen/flattening.h"
#include "xls/codegen/module_builder.h"
#include "xls/codegen/node_expressions.h"
#include "xls/codegen/vast.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/logging/logging.h"
#include "xls/ir/instantiation.h"
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
  if (node->Is<xls::Assert>() || node->Is<xls::Cover>()) {
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
      Register* reg = reg_write->GetRegister();
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

// A data structure representing a stage within a feed-forward pipeline.
struct Stage {
  // The register-reads at the beginning of the stage.
  std::vector<Node*> reg_reads;

  // The combinational logic within the stage in a topological sort order.
  std::vector<Node*> combinational_nodes;

  // The register-writes at the end of the stage.
  std::vector<Node*> reg_writes;

  // The registers written at the end of the pipeline stage.
  std::vector<Register*> registers;

  bool is_trivial = true;
};

// Partitions the nodes of the block into sequential sets of pipeline stages.
// Stages have the following properties:
//
//   (1) A node is in the same stage (or later) as its operands.
//
//   (2) If a RegisterWrite node for register R is in stage N the RegisterRead
//       node for R is in stage N+1.
//
// The stages are returned as a vector with stage N at index N of the
// vector.
//
// Returns an error if partitioning the nodes in this manner is impossible. This
// is the case if the block's dependency graph is cyclic (counting paths through
// registers), or if there exists two paths from a node A to a node B (again
// counting paths through registers) which have a different number of registers.
//
// TODO(meheff): 2021/08/27 Replace this pipeline reconstruction with tags on
// the pipeline registers which indicate the stage. These markers then are used
// to structure the Verilog.
absl::StatusOr<std::vector<Stage>> SplitBlockIntoStages(Block* block) {
  // Construct a graph as an edge list which indicates the minimum distance in
  // stages between nodes in the block.
  struct Edge {
    Node* node;
    int64_t distance;
  };
  absl::flat_hash_map<Node*, std::vector<Edge>> stage_graph;
  for (Node* node : block->nodes()) {
    // Create the node in the graph if it does not exist.
    stage_graph.insert({node, {}});

    for (Node* operand : node->operands()) {
      // A node must be in the same stage or later than its operand (distance
      // 0). This is condition (1) above.
      stage_graph[operand].push_back(Edge{node, 0});
    }

    if (node->Is<RegisterWrite>()) {
      RegisterWrite* reg_write = node->As<RegisterWrite>();
      XLS_ASSIGN_OR_RETURN(RegisterRead * reg_read,
                           block->GetRegisterRead(reg_write->GetRegister()));

      // A node register read must be must exactly one stage after the register
      // write. Express this as a distance of one from the RegisterWrite to the
      // RegisterRead, and a distance of -1 from the RegisterRead to the
      // RegisterWrite.
      stage_graph[reg_write].push_back(Edge{reg_read, 1});
      stage_graph[reg_read].push_back(Edge{reg_write, -1});
    }
  }

  // Starting at all nodes in stage 0, incrementally increment the stages of
  // nodes to satisfy the above conditions (1) and (2) as indicated in the stage
  // graph constructed above.
  std::deque<Node*> worklist;
  absl::flat_hash_map<Node*, int64_t> node_stage;
  for (Node* node : block->nodes()) {
    node_stage[node] = 0;
    worklist.push_back(node);
  }

  int64_t max_stage = 0;
  while (!worklist.empty()) {
    Node* source = worklist.front();
    worklist.pop_front();

    // The number of stages should never exceed the number of nodes. In this
    // case, there is an impossible-to-pipeline graph.
    XLS_RET_CHECK_LT(node_stage.at(source), block->node_count())
        << "Block is not a pipeline. May contain a (register) backedge or "
           "registers are not layered";

    for (const Edge& edge : stage_graph.at(source)) {
      Node* target = edge.node;
      // `min_stage` is the earliest stage `target` may appear in.
      int64_t min_stage = node_stage.at(source) + edge.distance;
      if (node_stage.at(target) < min_stage) {
        node_stage[target] = min_stage;
        max_stage = std::max(max_stage, min_stage);
        worklist.push_back(target);
      }
    }
  }

  // Move non-register nodes to the latest possible stage given their user
  // constraints. This is primarily to push nodes such as literals into the
  // stage of their user rather than all being placed in stage 0.
  for (Node* node : ReverseTopoSort(block)) {
    if (node->users().empty() || node->Is<RegisterRead>() ||
        node->Is<RegisterWrite>()) {
      continue;
    }
    int64_t stage = node_stage.at(*(node->users().begin()));
    for (Node* user : node->users()) {
      stage = std::min(stage, node_stage.at(user));
    }
    node_stage[node] = stage;
  }

  // Verify all minimum distances are satisfied in the stage assignment.
  for (Node* node : block->nodes()) {
    if (node->Is<RegisterWrite>()) {
      RegisterWrite* reg_write = node->As<RegisterWrite>();
      XLS_ASSIGN_OR_RETURN(RegisterRead * reg_read,
                           block->GetRegisterRead(reg_write->GetRegister()));
      XLS_RET_CHECK_EQ(node_stage[reg_write] + 1, node_stage[reg_read]);
      XLS_RET_CHECK_EQ(node_stage[reg_write], node_stage[reg_write->data()]);
    }
  }

  // Gather the nodes in a vector of stages.
  std::vector<Stage> stages(max_stage + 1);
  for (Node* node : TopoSort(block)) {
    if (node->Is<InputPort>() || node->Is<OutputPort>()) {
      continue;
    }

    Stage& stage = stages[node_stage[node]];
    if (node->Is<RegisterWrite>()) {
      stage.reg_writes.push_back(node->As<RegisterWrite>());
      stage.registers.push_back(node->As<RegisterWrite>()->GetRegister());
    } else if (node->Is<RegisterRead>()) {
      stage.reg_reads.push_back(node->As<RegisterRead>());
    } else {
      stage.combinational_nodes.push_back(node);
    }
  }
  return stages;
}

// Abstraction encapsulating the necessary information to generate
// (System)Verilog for an IR block.
class BlockGenerator {
 public:
  // Generates (System)Verilog from the given block into the given Verilog file
  // using the given options.
  static absl::Status Generate(Block* block, VerilogFile* file,
                               const CodegenOptions& options) {
    XLS_ASSIGN_OR_RETURN(absl::optional<ResetProto> reset_proto,
                         GetBlockResetProto(block));
    absl::optional<std::string> clock_name;
    if (block->GetClockPort().has_value()) {
      clock_name = block->GetClockPort()->name;
    } else if (!block->GetRegisters().empty()) {
      return absl::InvalidArgumentError(
          "Block has registers but no clock port");
    }

    BlockGenerator generator(block, options, clock_name, reset_proto, file);
    return generator.Emit();
  }

 private:
  BlockGenerator(Block* block, const CodegenOptions& options,
                 absl::optional<std::string> clock_name,
                 absl::optional<ResetProto> reset_proto, VerilogFile* file)
      : block_(block),
        options_(options),
        reset_proto_(reset_proto),
        file_(file),
        mb_(block->name(), file_, options.use_system_verilog(), clock_name,
            reset_proto) {}

  // Generates and returns the Verilog text for the underlying block.
  absl::Status Emit() {
    XLS_RETURN_IF_ERROR(EmitInputPorts());
    // TODO(meheff): 2021/11/04 Emit instantiations in pipeline stages if
    // possible.
    XLS_RETURN_IF_ERROR(DeclareInstantiationOutputs());
    if (options_.emit_as_pipeline()) {
      // Emits the block as a sequence of pipeline stages. First reconstruct the
      // stages and emit the stages one-by-one. Emitting as a pipeline is purely
      // cosmentic relative to the emit_as_pipeline=false option as the Verilog
      // generated each way is functionally identical.
      XLS_ASSIGN_OR_RETURN(std::vector<Stage> stages,
                           SplitBlockIntoStages(block_));
      for (int64_t stage_num = 0; stage_num < stages.size(); ++stage_num) {
        XLS_VLOG(2) << "Emitting stage: " << stage_num;
        const Stage& stage = stages.at(stage_num);
        mb_.NewDeclarationAndAssignmentSections();
        XLS_RETURN_IF_ERROR(EmitLogic(stage.reg_reads, stage_num));
        if (!stage.registers.empty() || !stage.combinational_nodes.empty()) {
          mb_.declaration_section()->Add<BlankLine>();
          mb_.declaration_section()->Add<Comment>(
              absl::StrFormat("===== Pipe stage %d:", stage_num));
          XLS_RETURN_IF_ERROR(EmitLogic(stage.combinational_nodes, stage_num));

          if (!stage.registers.empty()) {
            mb_.NewDeclarationAndAssignmentSections();
            mb_.declaration_section()->Add<BlankLine>();
            mb_.declaration_section()->Add<Comment>(
                absl::StrFormat("Registers for pipe stage %d:", stage_num));
            XLS_RETURN_IF_ERROR(DeclareRegisters(stage.registers));
            XLS_RETURN_IF_ERROR(EmitLogic(stage.reg_writes, stage_num));
            XLS_RETURN_IF_ERROR(AssignRegisters(stage.registers));
          }
        }
      }
    } else {
      XLS_RETURN_IF_ERROR(DeclareRegisters(block_->GetRegisters()));
      XLS_RETURN_IF_ERROR(EmitLogic(TopoSort(block_).AsVector()));
      XLS_RETURN_IF_ERROR(AssignRegisters(block_->GetRegisters()));
    }

    // Emit instantiations, asserts, coverpoints and traces separately at the
    // end of the Verilog module.
    XLS_RETURN_IF_ERROR(EmitInstantiations());
    XLS_RETURN_IF_ERROR(EmitAsserts());
    XLS_RETURN_IF_ERROR(EmitCovers());
    XLS_RETURN_IF_ERROR(EmitTraces());

    XLS_RETURN_IF_ERROR(EmitOutputPorts());

    return absl::OkStatus();
  }

  absl::Status EmitInputPorts() {
    for (const Block::Port& port : block_->GetPorts()) {
      if (absl::holds_alternative<InputPort*>(port)) {
        InputPort* input_port = absl::get<InputPort*>(port);
        if (reset_proto_.has_value() &&
            input_port->GetName() == reset_proto_->name()) {
          // The reset signal is implicitly added by ModuleBuilder.
          node_exprs_[input_port] = mb_.reset().value().signal;
        } else {
          XLS_ASSIGN_OR_RETURN(
              Expression * port_expr,
              mb_.AddInputPort(input_port->GetName(), input_port->GetType()));
          node_exprs_[input_port] = port_expr;
        }
      }
    }
    return absl::OkStatus();
  }

  // Emits the logic for the given nodes. This includes declaring and wires/regs
  // defined by the nodes.
  absl::Status EmitLogic(absl::Span<Node* const> nodes,
                         absl::optional<int64_t> stage = absl::nullopt) {
    for (Node* node : nodes) {
      XLS_VLOG(3) << "Emitting logic for: " << node->GetName();

      // Ports are handled elsewhere for proper ordering of ports.
      if (node->Is<InputPort>() || node->Is<OutputPort>()) {
        continue;
      }

      // Instantiation outputs are declared as expressions elsewhere.
      if (node->Is<InstantiationOutput>()) {
        continue;
      }

      // RegisterWrite and RegisterRead don't directly generate any VAST ast,
      // but some bookkeeping is necessary for associating an expression with
      // each register and setting the next and (optional) load-enable
      // expressions.
      if (node->Is<RegisterWrite>()) {
        RegisterWrite* reg_write = node->As<RegisterWrite>();
        mb_registers_.at(reg_write->GetRegister()).next =
            absl::get<Expression*>(node_exprs_.at(reg_write->data()));
        if (reg_write->load_enable().has_value()) {
          mb_registers_.at(reg_write->GetRegister()).load_enable =
              absl::get<Expression*>(
                  node_exprs_.at(reg_write->load_enable().value()));
        }
        node_exprs_[node] = UnrepresentedSentinel();
        continue;
      }
      if (node->Is<RegisterRead>()) {
        RegisterRead* reg_read = node->As<RegisterRead>();
        node_exprs_[node] = mb_registers_.at(reg_read->GetRegister()).ref;
        continue;
      }

      if (!IsRepresentable(node->GetType())) {
        node_exprs_[node] = UnrepresentedSentinel();
        continue;
      }

      // If any of the operands do not have an Expression* representation then
      // handle the node specially.
      if (std::any_of(
              node->operands().begin(), node->operands().end(), [&](Node* n) {
                return !absl::holds_alternative<Expression*>(node_exprs_.at(n));
              })) {
        XLS_ASSIGN_OR_RETURN(
            node_exprs_[node],
            CodegenNodeWithUnrepresentedOperands(node, &mb_, node_exprs_));
        continue;
      }

      // Emit non-bits-typed literals as module-level constants because in
      // general these complicated types cannot be handled inline, and
      // constructing them in Verilog may require a sequence of assignments.
      if (node->Is<xls::Literal>() && !node->GetType()->IsBits()) {
        XLS_ASSIGN_OR_RETURN(
            node_exprs_[node],
            mb_.DeclareModuleConstant(node->GetName(),
                                      node->As<xls::Literal>()->value()));
        continue;
      }

      if (std::any_of(
              node->operands().begin(), node->operands().end(), [&](Node* n) {
                return !absl::holds_alternative<Expression*>(node_exprs_.at(n));
              })) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Unable to generate code for node %s, has unrepresentable operand",
            node->GetName()));
      }

      std::vector<Expression*> inputs;
      for (Node* operand : node->operands()) {
        inputs.push_back(absl::get<Expression*>(node_exprs_.at(operand)));
      }

      // Gate operations are emitted specially as they may have a custom
      // user-specified format string.
      if (node->Is<Gate>()) {
        XLS_ASSIGN_OR_RETURN(node_exprs_[node],
                             mb_.EmitGate(node->As<Gate>(), inputs[0],
                                          inputs[1], options_.gate_format()));
        continue;
      }

      // If the node has an assigned name then don't emit as an inline
      // expression. This ensures the name appears in the generated Verilog.
      auto emit_as_assignment = [this](Node* n) {
        if (n->HasAssignedName() ||
            (n->users().size() > 1 &&
             !ShouldInlineExpressionIntoMultipleUses(n)) ||
            n->function_base()->HasImplicitUse(n) ||
            !mb_.CanEmitAsInlineExpression(n)) {
          return true;
        }
        // Emit operands of RegisterWrite's as assignments rather than inline
        // to avoid having logic in the always_ff blocks.
        for (Node* user : n->users()) {
          if (user->Is<RegisterWrite>()) {
            return true;
          }
        }
        return false;
      };

      if (emit_as_assignment(node)) {
        std::string name =
            stage.has_value() ? absl::StrCat(PipelineSignalName(node->GetName(),
                                                                stage.value()),
                                             "_comb")
                              : node->GetName();
        XLS_ASSIGN_OR_RETURN(node_exprs_[node],
                             mb_.EmitAsAssignment(name, node, inputs));
      } else {
        XLS_ASSIGN_OR_RETURN(node_exprs_[node],
                             mb_.EmitAsInlineExpression(node, inputs));
      }
    }
    return absl::OkStatus();
  }

  // Declares the given registers using `reg` declarations.
  absl::Status DeclareRegisters(absl::Span<Register* const> registers) {
    const absl::optional<verilog::Reset>& reset = mb_.reset();
    for (Register* reg : registers) {
      XLS_VLOG(3) << "Declaring register " << reg->name();
      Expression* reset_expr = nullptr;
      if (reg->reset().has_value()) {
        XLS_RET_CHECK(reset.has_value());
        const Value& reset_value = reg->reset()->reset_value;

        // If the value is a bits type it can be emitted inline. Otherwise emit
        // as a module constant.
        if (reset_value.IsBits()) {
          reset_expr = mb_.file()->Literal(reset_value.bits());
        } else {
          XLS_ASSIGN_OR_RETURN(
              reset_expr, mb_.DeclareModuleConstant(
                              absl::StrCat(reg->name(), "_init"), reset_value));
        }
      }
      XLS_ASSIGN_OR_RETURN(
          mb_registers_[reg],
          mb_.DeclareRegister(absl::StrCat(reg->name()), reg->type(),
                              /*next=*/nullptr, reset_expr));
    }
    return absl::OkStatus();
  }

  // Adds an always_ff (or Verilog equivalent) and use it to assign the next
  // cycle value for each of the given registers. Registers must have been
  // declared previously with DeclareRegisters.
  absl::Status AssignRegisters(absl::Span<Register* const> registers) {
    // Group registers with/without reset signal and call
    // ModuleBuilder::AssignRegisters for each.
    std::vector<ModuleBuilder::Register> registers_with_reset;
    std::vector<ModuleBuilder::Register> registers_without_reset;
    for (Register* reg : registers) {
      XLS_RET_CHECK(mb_registers_.contains(reg)) << absl::StreamFormat(
          "Register `%s` was not previously declared", reg->name());
      ModuleBuilder::Register mb_reg = mb_registers_.at(reg);
      if (reg->reset().has_value()) {
        registers_with_reset.push_back(mb_reg);
      } else {
        registers_without_reset.push_back(mb_reg);
      }
    }
    if (!registers_without_reset.empty()) {
      XLS_RETURN_IF_ERROR(mb_.AssignRegisters(registers_without_reset));
    }
    if (!registers_with_reset.empty()) {
      XLS_RETURN_IF_ERROR(mb_.AssignRegisters(registers_with_reset));
    }
    return absl::OkStatus();
  }

  absl::Status EmitOutputPorts() {
    // Iterate through GetPorts and pick out the output ports because GetPorts
    // contains the desired port ordering.
    for (const Block::Port& port : block_->GetPorts()) {
      if (absl::holds_alternative<OutputPort*>(port)) {
        OutputPort* output_port = absl::get<OutputPort*>(port);
        XLS_RETURN_IF_ERROR(mb_.AddOutputPort(
            output_port->GetName(), output_port->operand(0)->GetType(),
            absl::get<Expression*>(node_exprs_.at(output_port->operand(0)))));
        node_exprs_[output_port] = UnrepresentedSentinel();
      }
    }
    return absl::OkStatus();
  }

  // Declare a wire in the Verilog module for each output of each instantation
  // in the block. These declared outputs can then be used in downstream
  // expressions.
  absl::Status DeclareInstantiationOutputs() {
    for (xls::Instantiation* instantiation : block_->GetInstantiations()) {
      if (dynamic_cast<BlockInstantiation*>(instantiation) != nullptr) {
        for (InstantiationOutput* output :
             block_->GetInstantiationOutputs(instantiation)) {
          node_exprs_[output] =
              mb_.DeclareVariable(output->GetName(), output->GetType());
        }
      }
    }
    return absl::OkStatus();
  }

  // Emit each instantation in the block into the separate instantation module
  // section.
  absl::Status EmitInstantiations() {
    for (xls::Instantiation* instantiation : block_->GetInstantiations()) {
      if (xls::BlockInstantiation* block_instantiation =
              dynamic_cast<BlockInstantiation*>(instantiation)) {
        std::vector<Connection> connections;
        for (InstantiationInput* input :
             block_->GetInstantiationInputs(instantiation)) {
          XLS_RET_CHECK(absl::holds_alternative<Expression*>(
              node_exprs_.at(input->operand(0))));
          connections.push_back(Connection{
              input->port_name(),
              absl::get<Expression*>(node_exprs_.at(input->operand(0)))});
        }
        for (InstantiationOutput* output :
             block_->GetInstantiationOutputs(instantiation)) {
          XLS_RET_CHECK(
              absl::holds_alternative<Expression*>(node_exprs_.at(output)));
          connections.push_back(
              Connection{output->port_name(),
                         absl::get<Expression*>(node_exprs_.at(output))});
        }
        mb_.instantiation_section()->Add<Instantiation>(
            block_instantiation->instantiated_block()->name(),
            block_instantiation->name(),
            /*parameters=*/std::vector<Connection>(), connections);
      }
    }
    return absl::OkStatus();
  }

  absl::Status EmitAsserts() {
    for (Node* node : block_->nodes()) {
      if (node->Is<xls::Assert>()) {
        xls::Assert* asrt = node->As<xls::Assert>();
        Expression* condition =
            absl::get<Expression*>(node_exprs_.at(asrt->condition()));
        XLS_RETURN_IF_ERROR(
            mb_.EmitAssert(asrt, condition, options_.assert_format()));
      }
    }
    return absl::OkStatus();
  }

  absl::Status EmitCovers() {
    for (Node* node : block_->nodes()) {
      if (node->Is<xls::Cover>()) {
        xls::Cover* cover = node->As<xls::Cover>();
        Expression* condition =
            absl::get<Expression*>(node_exprs_.at(cover->condition()));
        XLS_RETURN_IF_ERROR(mb_.EmitCover(cover, condition));
      }
    }
    return absl::OkStatus();
  }

  absl::Status EmitTraces() {
    for (Node* node : block_->nodes()) {
      if (node->Is<xls::Trace>()) {
        xls::Trace* trace = node->As<xls::Trace>();
        Expression* condition =
            absl::get<Expression*>(node_exprs_.at(trace->condition()));

        std::vector<Expression*> trace_args;
        for (Node* arg : trace->args()) {
          trace_args.push_back(absl::get<Expression*>(node_exprs_.at(arg)));
        }
        XLS_RETURN_IF_ERROR(mb_.EmitTrace(trace, condition, trace_args));
      }
    }
    return absl::OkStatus();
  }

  Block* block_;
  const CodegenOptions& options_;
  absl::optional<ResetProto> reset_proto_;

  VerilogFile* file_;
  ModuleBuilder mb_;

  // Map from Node* to the Verilog expression representing its value.
  absl::flat_hash_map<Node*, NodeRepresentation> node_exprs_;

  // Map from xls::Register* to the ModuleBuilder register abstraction
  // representing the underlying Verilog register.
  absl::flat_hash_map<xls::Register*, ModuleBuilder::Register> mb_registers_;
};

// Recursive visitor of blocks in a DFS order. Edges are block instantiations.
// Visited blocks are collected into `post_order` in a DFS post-order.
absl::Status DfsVisitBlocks(Block* block, absl::flat_hash_set<Block*>& visited,
                            std::vector<Block*>& post_order) {
  if (!visited.insert(block).second) {
    // Block has already been visited.
    return absl::OkStatus();
  }
  for (xls::Instantiation* instantiation : block->GetInstantiations()) {
    xls::BlockInstantiation* block_instantiation =
        dynamic_cast<BlockInstantiation*>(instantiation);
    if (block_instantiation == nullptr) {
      return absl::UnimplementedError(absl::StrFormat(
          "Instantiations of kind `%s` are not supported in code generation",
          InstantiationKindToString(instantiation->kind())));
    }
    XLS_RETURN_IF_ERROR(DfsVisitBlocks(
        block_instantiation->instantiated_block(), visited, post_order));
  }
  post_order.push_back(block);
  return absl::OkStatus();
}

// Return the blocks instantiated by the given top-level block. This includes
// blocks transitively instantiated. In the returned vector, an instantiated
// block will always appear before the instantiating block (DFS post order).
absl::StatusOr<std::vector<Block*>> GatherInstantiatedBlocks(Block* top) {
  std::vector<Block*> blocks;
  absl::flat_hash_set<Block*> visited;
  XLS_RETURN_IF_ERROR(DfsVisitBlocks(top, visited, blocks));
  return blocks;
}

}  // namespace

absl::StatusOr<std::string> GenerateVerilog(Block* top,
                                            const CodegenOptions& options) {
  XLS_VLOG(2) << absl::StreamFormat(
      "Generating Verilog for packge with with top level block `%s`:",
      top->name());
  XLS_VLOG_LINES(2, top->DumpIr());

  XLS_ASSIGN_OR_RETURN(std::vector<Block*> blocks,
                       GatherInstantiatedBlocks(top));
  VerilogFile file(options.use_system_verilog());
  for (Block* block : blocks) {
    XLS_RETURN_IF_ERROR(BlockGenerator::Generate(block, &file, options));
    if (block != blocks.back()) {
      file.Add(file.Make<BlankLine>());
      file.Add(file.Make<BlankLine>());
    }
  }
  std::string text = file.Emit(nullptr);
  XLS_VLOG(2) << "Verilog output:";
  XLS_VLOG_LINES(2, text);

  return text;
}

}  // namespace verilog
}  // namespace xls
