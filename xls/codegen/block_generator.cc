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

#include <algorithm>
#include <array>
#include <cstdint>
#include <deque>
#include <initializer_list>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/random/bit_gen_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/codegen/block_conversion.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/flattening.h"
#include "xls/codegen/module_builder.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/codegen/node_expressions.h"
#include "xls/codegen/node_representation.h"
#include "xls/codegen/op_override.h"
#include "xls/codegen/vast/vast.h"
#include "xls/codegen/verilog_line_map.pb.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/block.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/register.h"
#include "xls/ir/source_location.h"
#include "xls/ir/topo_sort.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

namespace xls {
namespace verilog {
namespace {

template <typename T>
std::vector<T> Shuffle(absl::Span<const T> v, absl::BitGenRef rng) {
  std::vector<T> vector(v.begin(), v.end());
  absl::c_shuffle(vector, rng);
  return vector;
}

template <typename T>
absl::Span<const T> MaybeShuffle(absl::Span<const T> v, std::vector<T>& storage,
                                 std::optional<absl::BitGenRef> rng) {
  if (!rng.has_value()) {
    return v;
  }
  storage = Shuffle(v, *rng);
  return absl::MakeConstSpan(storage);
}

// Returns true if the given type is representable in the Verilog.
bool IsRepresentable(Type* type) { return type->GetFlatBitCount() > 0; }

// Return the Verilog representation for the given node which has at least one
// operand which is not represented by an Expression*.
absl::StatusOr<NodeRepresentation> CodegenNodeWithUnrepresentedOperands(
    Node* node, ModuleBuilder* mb,
    const absl::flat_hash_map<Node*, NodeRepresentation>& node_exprs,
    std::string_view name, bool emit_as_assignment) {
  if (node->Is<Tuple>()) {
    // A tuple may have unrepresentable inputs such as empty tuples.  Walk
    // through and gather non-zero-width inputs and flatten them.
    std::vector<Expression*> nonempty_elements;
    for (Node* operand : node->operands()) {
      if (!std::holds_alternative<Expression*>(node_exprs.at(operand))) {
        if (operand->GetType()->GetFlatBitCount() != 0) {
          return absl::UnimplementedError(absl::StrFormat(
              "Unable to generate code for: %s", node->ToString()));
        }
        continue;
      }
      nonempty_elements.push_back(
          std::get<Expression*>(node_exprs.at(operand)));
    }
    XLS_ASSIGN_OR_RETURN(
        Expression * expr,
        FlattenTuple(nonempty_elements, node->GetType()->AsTupleOrDie(),
                     mb->file(), node->loc()));
    if (emit_as_assignment) {
      LogicRef* ref = mb->DeclareVariable(name, node->GetType());
      XLS_RETURN_IF_ERROR(mb->Assign(ref, expr, node->GetType()));
      return ref;
    }
    return expr;
  }
  return absl::UnimplementedError(
      absl::StrFormat("Unable to generate code for: %s", node->ToString()));
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
absl::StatusOr<std::vector<Stage>> SplitBlockIntoStages(
    Block* block, std::optional<absl::BitGenRef> rng) {
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
        << absl::StreamFormat(
               "Node %v in stage %d! Block is not a pipeline. May contain a "
               "(register) backedge or registers are not layered\n%s",
               *source, node_stage.at(source), block->DumpIr());

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
  for (Node* node : TopoSort(block, rng)) {
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
  static absl::Status Generate(
      Block* block, VerilogFile* file, const CodegenOptions& options,
      const absl::flat_hash_map<InputPort*, std::string>& input_port_sv_types,
      const absl::flat_hash_map<OutputPort*, std::string>&
          output_port_sv_types) {
    const std::optional<ResetProto>& reset_proto = options.reset();
    if (reset_proto.has_value()) {
      VLOG(5) << absl::StreamFormat("Reset proto for %s: %s", block->name(),
                                    reset_proto->DebugString());
    } else {
      VLOG(5) << absl::StreamFormat("No reset proto for %s", block->name());
    }
    std::optional<std::string_view> clock_name;
    if (block->GetClockPort().has_value()) {
      clock_name = block->GetClockPort()->name;
    } else if (!block->GetRegisters().empty()) {
      return absl::InvalidArgumentError(
          "Block has registers but no clock port");
    }

    BlockGenerator generator(block, options, clock_name, reset_proto,
                             input_port_sv_types, output_port_sv_types, file);
    return generator.Emit();
  }

 private:
  BlockGenerator(
      Block* block, const CodegenOptions& options,
      std::optional<std::string_view> clock_name,
      const std::optional<ResetProto>& reset_proto,
      const absl::flat_hash_map<InputPort*, std::string>& input_port_sv_types,
      const absl::flat_hash_map<OutputPort*, std::string>& output_port_sv_types,
      VerilogFile* file)
      : block_(block),
        options_(options),
        reset_proto_(reset_proto),
        input_port_sv_types_(input_port_sv_types),
        output_port_sv_types_(output_port_sv_types),
        file_(file),
        mb_(block->name(), file_, options, clock_name, reset_proto) {
    if (!options.randomize_order_seed().empty()) {
      std::seed_seq seed_seq(options.randomize_order_seed().begin(),
                             options.randomize_order_seed().end());
      rng_.emplace(seed_seq);
    }
  }

  // Generates and returns the Verilog text for the underlying block.
  absl::Status Emit() {
    XLS_RETURN_IF_ERROR(EmitInputPorts());
    // TODO(meheff): 2021/11/04 Emit instantiations in pipeline stages if
    // possible.
    XLS_RETURN_IF_ERROR(DeclareInstantiationOutputs());
    if (options_.emit_as_pipeline()) {
      // Emits the block as a sequence of pipeline stages. First reconstruct the
      // stages and emit the stages one-by-one. Emitting as a pipeline is purely
      // cosmetic relative to the emit_as_pipeline=false option as the Verilog
      // generated each way is functionally identical.
      XLS_ASSIGN_OR_RETURN(std::vector<Stage> stages,
                           SplitBlockIntoStages(block_, rng_));
      for (int64_t stage_num = 0; stage_num < stages.size(); ++stage_num) {
        VLOG(2) << "Emitting stage: " << stage_num;
        const Stage& stage = stages.at(stage_num);
        mb_.NewDeclarationAndAssignmentSections();
        XLS_RETURN_IF_ERROR(EmitLogic(stage.reg_reads, stage_num));
        if (!stage.registers.empty() || !stage.combinational_nodes.empty()) {
          mb_.declaration_section()->Add<BlankLine>(SourceInfo());
          mb_.declaration_section()->Add<Comment>(
              SourceInfo(), absl::StrFormat("===== Pipe stage %d:", stage_num));
          XLS_RETURN_IF_ERROR(EmitLogic(stage.combinational_nodes, stage_num));

          if (!stage.registers.empty()) {
            mb_.NewDeclarationAndAssignmentSections();
            mb_.declaration_section()->Add<BlankLine>(SourceInfo());
            mb_.declaration_section()->Add<Comment>(
                SourceInfo(),
                absl::StrFormat("Registers for pipe stage %d:", stage_num));
            XLS_RETURN_IF_ERROR(DeclareRegisters(stage.registers));
            XLS_RETURN_IF_ERROR(EmitLogic(stage.reg_writes, stage_num));
            XLS_RETURN_IF_ERROR(AssignRegisters(stage.registers));
          }
        }
      }
    } else {
      std::vector<Register*> shuffled_storage;
      absl::Span<Register* const> registers =
          MaybeShuffle(block_->GetRegisters(), shuffled_storage, rng_);

      XLS_RETURN_IF_ERROR(DeclareRegisters(registers));
      XLS_RETURN_IF_ERROR(EmitLogic(TopoSort(block_, rng_)));
      XLS_RETURN_IF_ERROR(AssignRegisters(registers));
    }

    // Emit instantiations separately at the end of the Verilog module.
    XLS_RETURN_IF_ERROR(EmitInstantiations());

    XLS_RETURN_IF_ERROR(EmitOutputPorts());

    return absl::OkStatus();
  }

  std::optional<std::string_view> GetSvType(Node* n) const {
    if (n->Is<InputPort>() &&
        input_port_sv_types_.contains(n->As<InputPort>())) {
      return input_port_sv_types_.at(n->As<InputPort>());
    }
    if (n->Is<OutputPort>() &&
        output_port_sv_types_.contains(n->As<OutputPort>())) {
      return output_port_sv_types_.at(n->As<OutputPort>());
    }
    return std::nullopt;
  }

  absl::Status EmitInputPorts() {
    std::vector<Block::Port> shuffled_storage;
    for (const Block::Port& port :
         MaybeShuffle(block_->GetPorts(), shuffled_storage, rng_)) {
      if (std::holds_alternative<InputPort*>(port)) {
        InputPort* input_port = std::get<InputPort*>(port);
        if (reset_proto_.has_value() &&
            input_port->GetName() == reset_proto_->name()) {
          // The reset signal is implicitly added by ModuleBuilder.
          node_exprs_[input_port] = mb_.reset().value().signal;
        } else {
          XLS_ASSIGN_OR_RETURN(
              Expression * port_expr,
              mb_.AddInputPort(input_port->GetName(), input_port->GetType(),
                               GetSvType(input_port)));
          node_exprs_[input_port] = port_expr;
        }
      }
    }
    return absl::OkStatus();
  }

  // If the node has an assigned name then don't emit as an inline expression.
  // This ensures the name appears in the generated Verilog.
  bool ShouldEmitAsAssignment(Node* const n, int64_t inline_depth) {
    if (n->HasAssignedName() ||
        (n->users().size() > 1 && !ShouldInlineExpressionIntoMultipleUses(n)) ||
        n->function_base()->HasImplicitUse(n) ||
        !mb_.CanEmitAsInlineExpression(n) || options_.separate_lines() ||
        inline_depth > options_.max_inline_depth()) {
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
  }

  // Name of the node if it gets emitted as a separate assignment.
  std::string NodeAssignmentName(Node* const node,
                                 std::optional<int64_t> stage) {
    return stage.has_value() ? absl::StrCat(PipelineSignalName(node->GetName(),
                                                               stage.value()),
                                            "_comb")
                             : node->GetName();
  }

  // Emits the logic for the given nodes. This includes declaring and wires/regs
  // defined by the nodes.
  absl::Status EmitLogic(absl::Span<Node* const> nodes,
                         std::optional<int64_t> stage = std::nullopt) {
    absl::flat_hash_map<Node*, int64_t> node_depth;
    for (Node* node : nodes) {
      VLOG(3) << "Emitting logic for: " << node->GetName();

      // TODO(google/xls#653): support per-node overrides?
      std::optional<OpOverride*> op_override =
          options_.GetOpOverride(node->op());

      if (op_override.has_value()) {
        std::vector<NodeRepresentation> inputs;
        for (const Node* operand : node->operands()) {
          inputs.push_back(node_exprs_.at(operand));
        }

        XLS_ASSIGN_OR_RETURN(
            node_exprs_[node],
            (*op_override)
                ->Emit(node, NodeAssignmentName(node, stage), inputs, mb_));
        continue;
      }

      switch (node->op()) {
        // Ports are handled elsewhere for proper ordering of ports.
        case Op::kInputPort:
        case Op::kOutputPort:
        // Instantiation outputs are declared as expressions elsewhere.
        case Op::kInstantiationOutput:
          continue;

        // RegisterWrite and RegisterRead don't directly generate any VAST
        // ast, but some bookkeeping is necessary for associating an
        // expression with each register and setting the next and (optional)
        // load-enable expressions.
        case Op::kRegisterWrite: {
          RegisterWrite* reg_write = node->As<RegisterWrite>();
          mb_registers_.at(reg_write->GetRegister()).next =
              std::get<Expression*>(node_exprs_.at(reg_write->data()));
          if (reg_write->load_enable().has_value()) {
            mb_registers_.at(reg_write->GetRegister()).load_enable =
                std::get<Expression*>(
                    node_exprs_.at(reg_write->load_enable().value()));
          }
          node_exprs_[node] = UnrepresentedSentinel();
          continue;
        }
        case Op::kRegisterRead: {
          RegisterRead* reg_read = node->As<RegisterRead>();
          node_exprs_[node] = mb_registers_.at(reg_read->GetRegister()).ref;
          continue;
        }
        case Op::kAssert: {
          xls::Assert* asrt = node->As<xls::Assert>();
          NodeRepresentation condition = node_exprs_.at(asrt->condition());
          if (!std::holds_alternative<Expression*>(condition)) {
            return absl::InvalidArgumentError(
                absl::StrFormat("Unable to generate code for assert %s, has "
                                "condition that is not an expression",
                                asrt->GetName()));
          }
          XLS_ASSIGN_OR_RETURN(
              node_exprs_[node],
              mb_.EmitAssert(asrt, std::get<Expression*>(condition)));
          continue;
        }
        case Op::kCover: {
          xls::Cover* cover = node->As<xls::Cover>();
          NodeRepresentation condition = node_exprs_.at(cover->condition());
          if (!std::holds_alternative<Expression*>(condition)) {
            return absl::InvalidArgumentError(
                absl::StrFormat("Unable to generate code for cover %s, has "
                                "condition that is not an expression",
                                cover->GetName()));
          }
          XLS_ASSIGN_OR_RETURN(
              node_exprs_[node],
              mb_.EmitCover(cover, std::get<Expression*>(condition)));
          continue;
        }
        case Op::kTrace: {
          xls::Trace* trace = node->As<xls::Trace>();
          NodeRepresentation condition = node_exprs_.at(trace->condition());
          if (!std::holds_alternative<Expression*>(condition)) {
            return absl::InvalidArgumentError(
                absl::StrFormat("Unable to generate code for trace %s, has "
                                "condition that is not an expression",
                                trace->GetName()));
          }
          std::vector<Expression*> trace_args;
          for (Node* const arg : trace->args()) {
            const NodeRepresentation& arg_repr = node_exprs_.at(arg);
            if (!std::holds_alternative<Expression*>(arg_repr)) {
              return absl::InvalidArgumentError(
                  absl::StrFormat("Unable to generate code for trace %s, has "
                                  "arg that is not an expression",
                                  trace->GetName()));
            }
            trace_args.push_back(std::get<Expression*>(arg_repr));
          }
          XLS_ASSIGN_OR_RETURN(
              node_exprs_[node],
              mb_.EmitTrace(trace, std::get<Expression*>(condition),
                            trace_args));
          continue;
        }
        case Op::kLiteral: {
          if (!node->GetType()->IsBits() && IsRepresentable(node->GetType())) {
            CHECK_EQ(node->operands().size(), 0);
            XLS_ASSIGN_OR_RETURN(
                node_exprs_[node],
                mb_.DeclareModuleConstant(node->GetName(),
                                          node->As<xls::Literal>()->value()));
            continue;
          }
          break;
        }
        case Op::kGate: {
          CHECK_EQ(node->operands().size(), 2);
          const NodeRepresentation& data =
              node_exprs_.at(node->operands().at(0));
          const NodeRepresentation& condition =
              node_exprs_.at(node->operands().at(1));
          if (!std::holds_alternative<Expression*>(data)) {
            return absl::InvalidArgumentError(
                absl::StrFormat("Unable to generate code for gate node %s, has "
                                "unrepresentable data",
                                node->GetName()));
          }
          if (!std::holds_alternative<Expression*>(condition)) {
            return absl::InvalidArgumentError(
                absl::StrFormat("Unable to generate code for gate node %s, has "
                                "unrepresentable data",
                                node->GetName()));
          }
          XLS_ASSIGN_OR_RETURN(
              node_exprs_[node],
              mb_.EmitGate(node->As<Gate>(), std::get<Expression*>(data),
                           std::get<Expression*>(condition)));

          continue;
        }
        default:
          break;
      }

      if (!IsRepresentable(node->GetType())) {
        node_exprs_[node] = UnrepresentedSentinel();
        continue;
      }

      int64_t input_depth = 0;
      for (const Node* operand : node->operands()) {
        if (auto it = node_depth.find(operand); it != node_depth.end()) {
          input_depth = std::max(input_depth, it->second);
        }
      }
      int64_t inline_depth = input_depth + 1;

      // If any of the operands do not have an Expression* representation then
      // handle the node specially.
      if (std::any_of(
              node->operands().begin(), node->operands().end(), [&](Node* n) {
                return !std::holds_alternative<Expression*>(node_exprs_.at(n));
              })) {
        bool emit_as_assignment = ShouldEmitAsAssignment(node, inline_depth);
        XLS_ASSIGN_OR_RETURN(
            node_exprs_[node],
            CodegenNodeWithUnrepresentedOperands(
                node, &mb_, node_exprs_, NodeAssignmentName(node, stage),
                emit_as_assignment));
        if (!emit_as_assignment) {
          node_depth[node] = inline_depth;
        }
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
                return !std::holds_alternative<Expression*>(node_exprs_.at(n));
              })) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Unable to generate code for node %s, has unrepresentable operand",
            node->GetName()));
      }
      std::vector<Expression*> inputs;
      for (const Node* operand : node->operands()) {
        inputs.push_back(std::get<Expression*>(node_exprs_.at(operand)));
      }

      if (ShouldEmitAsAssignment(node, inline_depth)) {
        XLS_ASSIGN_OR_RETURN(
            node_exprs_[node],
            mb_.EmitAsAssignment(NodeAssignmentName(node, stage), node,
                                 inputs));
      } else {
        XLS_ASSIGN_OR_RETURN(node_exprs_[node],
                             mb_.EmitAsInlineExpression(node, inputs));
        node_depth[node] = inline_depth;
      }
    }
    return absl::OkStatus();
  }

  // Declares the given registers using `reg` declarations.
  absl::Status DeclareRegisters(absl::Span<Register* const> registers) {
    const std::optional<verilog::Reset>& reset = mb_.reset();
    for (Register* reg : registers) {
      VLOG(3) << "Declaring register " << reg->name();
      Expression* reset_expr = nullptr;
      if (reg->reset().has_value()) {
        XLS_RET_CHECK(reset.has_value());
        const Value& reset_value = reg->reset()->reset_value;

        // If the value is a bits type it can be emitted inline. Otherwise emit
        // as a module constant.
        if (reset_value.IsBits()) {
          reset_expr = mb_.file()->Literal(reset_value.bits(), SourceInfo());
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
    std::vector<Block::Port> shuffled_storage;
    for (const Block::Port& port :
         MaybeShuffle(block_->GetPorts(), shuffled_storage, rng_)) {
      if (std::holds_alternative<OutputPort*>(port)) {
        OutputPort* output_port = std::get<OutputPort*>(port);
        const NodeRepresentation& output_expr =
            node_exprs_.at(output_port->operand(0));
        XLS_RET_CHECK(std::holds_alternative<Expression*>(output_expr));
        XLS_RETURN_IF_ERROR(mb_.AddOutputPort(
            output_port->GetName(), output_port->operand(0)->GetType(),
            std::get<Expression*>(output_expr), GetSvType(output_port)));
        node_exprs_[output_port] = UnrepresentedSentinel();
      }
    }
    return absl::OkStatus();
  }

  // Declare a wire in the Verilog module for each output of each instantiation
  // in the block. These declared outputs can then be used in downstream
  // expressions.
  absl::Status DeclareInstantiationOutputs() {
    std::vector<xls::Instantiation*> shuffled_instantiations;
    for (xls::Instantiation* instantiation : MaybeShuffle(
             block_->GetInstantiations(), shuffled_instantiations, rng_)) {
      std::vector<InstantiationOutput*> shuffled_outputs;
      for (InstantiationOutput* output :
           MaybeShuffle(block_->GetInstantiationOutputs(instantiation),
                        shuffled_outputs, rng_)) {
        node_exprs_[output] =
            mb_.DeclareVariable(output->GetName(), output->GetType());
      }
    }
    return absl::OkStatus();
  }

  // Emit each instantiation in the block into the separate instantiation module
  // section.
  absl::Status EmitInstantiations() {
    // Since instantiations are emitted at the end, and not the pipeline stages
    // they are in, separate with headline to reduce confusion.
    if (!block_->GetInstantiations().empty()) {
      mb_.declaration_section()->Add<BlankLine>(SourceInfo());
      mb_.instantiation_section()->Add<Comment>(SourceInfo(),
                                                "===== Instantiations");
    }

    // Because we flatten arrays at module ports but otherwise use unpacked
    // arrays internally, we may need to make an expression that packs/unpacks
    // the port connection.
    // TODO(google/xls#320): This can be much simpler if we don't use unpacked
    // arrays.
    auto connection_expression =
        [this](const NodeRepresentation& expr,
               Type* type) -> absl::StatusOr<Expression*> {
      XLS_RET_CHECK(std::holds_alternative<Expression*>(expr));
      Expression* to_connect = std::get<Expression*>(expr);
      if (type->IsArray()) {
        to_connect =
            FlattenArray(to_connect->AsIndexableExpressionOrDie(),
                         type->AsArrayOrDie(), mb_.file(), SourceInfo());
      }
      return to_connect;
    };
    std::vector<xls::Instantiation*> shuffled_instantiations;
    for (xls::Instantiation* instantiation : MaybeShuffle(
             block_->GetInstantiations(), shuffled_instantiations, rng_)) {
      std::vector<Connection> connections;
      std::vector<InstantiationInput*> shuffled_inputs;
      for (InstantiationInput* input :
           MaybeShuffle(block_->GetInstantiationInputs(instantiation),
                        shuffled_inputs, rng_)) {
        XLS_RET_CHECK(input->operand_count() > 0);
        const NodeRepresentation& expr = node_exprs_.at(input->operand(0));
        XLS_RET_CHECK(std::holds_alternative<Expression*>(expr));
        XLS_ASSIGN_OR_RETURN(
            Expression * to_connect,
            connection_expression(expr, input->operand(0)->GetType()));
        connections.push_back(Connection{.port_name = input->port_name(),
                                         .expression = to_connect});
      }
      std::vector<InstantiationOutput*> shuffled_outputs;
      for (InstantiationOutput* output :
           MaybeShuffle(block_->GetInstantiationOutputs(instantiation),
                        shuffled_outputs, rng_)) {
        const NodeRepresentation& expr = node_exprs_.at(output);
        XLS_RET_CHECK(std::holds_alternative<Expression*>(expr));
        XLS_ASSIGN_OR_RETURN(Expression * to_connect,
                             connection_expression(expr, output->GetType()));
        connections.push_back(Connection{.port_name = output->port_name(),
                                         .expression = to_connect});
      }

      if (xls::BlockInstantiation* block_instantiation =
              dynamic_cast<BlockInstantiation*>(instantiation)) {
        std::optional<Block::ClockPort> port =
            block_instantiation->instantiated_block()->GetClockPort();
        if (port.has_value()) {
          if (mb_.clock() == nullptr) {
            return absl::InternalError(
                "The instantiated block requires a clock but the instantiating "
                "block has no clock.");
          }
          connections.push_back(Connection{port.value().name, mb_.clock()});
        }
        mb_.instantiation_section()->Add<Instantiation>(
            SourceInfo(), block_instantiation->instantiated_block()->name(),
            block_instantiation->name(),
            /*parameters=*/std::vector<Connection>(), connections);
      } else if (xls::ExternInstantiation* ffi_instantiation =
                     dynamic_cast<ExternInstantiation*>(instantiation)) {
        mb_.instantiation_section()->Add<TemplateInstantiation>(
            SourceInfo(), ffi_instantiation->name(),
            ffi_instantiation->function()
                ->ForeignFunctionData()
                ->code_template(),
            connections);
      } else if (xls::FifoInstantiation* fifo_instantiation =
                     dynamic_cast<FifoInstantiation*>(instantiation)) {
        std::vector<Connection> parameters;
        parameters.reserve(6);

        bool have_data = fifo_instantiation->data_type()->GetFlatBitCount() > 0;

        if (have_data) {
          parameters.push_back(Connection{
              .port_name = "Width",
              .expression = mb_.file()->Literal(
                  UBits(fifo_instantiation->data_type()->GetFlatBitCount(), 32),
                  SourceInfo(),
                  /*format=*/FormatPreference::kUnsignedDecimal)});
        }

        parameters.insert(
            parameters.end(),
            {
                Connection{
                    .port_name = "Depth",
                    .expression = mb_.file()->Literal(
                        UBits(fifo_instantiation->fifo_config().depth(), 32),
                        SourceInfo(),
                        /*format=*/FormatPreference::kUnsignedDecimal)},
                Connection{
                    .port_name = "EnableBypass",
                    .expression = mb_.file()->Literal(
                        UBits(
                            fifo_instantiation->fifo_config().bypass() ? 1 : 0,
                            1),
                        SourceInfo(),
                        /*format=*/FormatPreference::kUnsignedDecimal)},
                Connection{.port_name = "RegisterPushOutputs",
                           .expression = mb_.file()->Literal(
                               UBits(fifo_instantiation->fifo_config()
                                             .register_push_outputs()
                                         ? 1
                                         : 0,
                                     1),
                               SourceInfo(),
                               /*format=*/FormatPreference::kUnsignedDecimal)},
                Connection{.port_name = "RegisterPopOutputs",
                           .expression = mb_.file()->Literal(
                               UBits(fifo_instantiation->fifo_config()
                                             .register_pop_outputs()
                                         ? 1
                                         : 0,
                                     1),
                               SourceInfo(),
                               /*format=*/FormatPreference::kUnsignedDecimal)},
            });

        // Append clock to connections.
        connections.push_back(
            Connection{.port_name = "clk", .expression = mb_.clock()});
        // Sort clk and rst to top, then push, and finally pop ports.
        constexpr std::array<std::string_view, 8> kFifoPortPriority = {
            "clk",
            xls::FifoInstantiation::kResetPortName,
            xls::FifoInstantiation::kPushDataPortName,
            xls::FifoInstantiation::kPushValidPortName,
            xls::FifoInstantiation::kPopReadyPortName,
            xls::FifoInstantiation::kPushReadyPortName,
            xls::FifoInstantiation::kPopDataPortName,
            xls::FifoInstantiation::kPopValidPortName};
        absl::c_sort(connections, [&kFifoPortPriority](const Connection& a,
                                                       const Connection& b) {
          for (const std::string_view& port_name : kFifoPortPriority) {
            if (a.port_name == b.port_name) {
              // We don't want compare(x, x) == true.
              return false;
            }
            if (a.port_name == port_name) {
              return true;
            }
            if (b.port_name == port_name) {
              return false;
            }
          }
          return a.port_name < b.port_name;
        });

        std::string_view wrapper_name =
            have_data ? "xls_fifo_wrapper" : "xls_nodata_fifo_wrapper";

        mb_.instantiation_section()->Add<Instantiation>(
            SourceInfo(), wrapper_name, fifo_instantiation->name(),
            /*parameters=*/parameters, connections);
      } else {
        return absl::UnimplementedError(absl::StrFormat(
            "Instantiations of kind `%s` are not supported in code generation",
            InstantiationKindToString(instantiation->kind())));
      }
    }
    return absl::OkStatus();
  }

  Block* block_;
  const CodegenOptions& options_;
  std::optional<ResetProto> reset_proto_;
  const absl::flat_hash_map<InputPort*, std::string>& input_port_sv_types_;
  const absl::flat_hash_map<OutputPort*, std::string>& output_port_sv_types_;

  VerilogFile* file_;
  ModuleBuilder mb_;

  std::optional<std::mt19937_64> rng_;

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
    if (xls::BlockInstantiation* block_instantiation =
            dynamic_cast<BlockInstantiation*>(instantiation)) {
      XLS_RETURN_IF_ERROR(DfsVisitBlocks(
          block_instantiation->instantiated_block(), visited, post_order));
      continue;
    }
    if (instantiation->kind() == InstantiationKind::kExtern) {
      // An external block is a leaf from our perspective.
      continue;
    }
    if (instantiation->kind() == InstantiationKind::kFifo) {
      // A fifo is a leaf from our perspective.
      continue;
    }

    return absl::UnimplementedError(absl::StrFormat(
        "Instantiations of kind `%s` are not supported in code generation",
        InstantiationKindToString(instantiation->kind())));
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

absl::StatusOr<std::string> GenerateVerilog(
    Block* top, const CodegenOptions& options, VerilogLineMap* verilog_line_map,
    const absl::flat_hash_map<InputPort*, std::string>& input_port_sv_types,
    const absl::flat_hash_map<OutputPort*, std::string>& output_port_sv_types) {
  VLOG(2) << absl::StreamFormat(
      "Generating Verilog for packge with with top level block `%s`:",
      top->name());
  XLS_VLOG_LINES(2, top->DumpIr());

  XLS_ASSIGN_OR_RETURN(std::vector<Block*> blocks,
                       GatherInstantiatedBlocks(top));
  VerilogFile file(options.use_system_verilog() ? FileType::kSystemVerilog
                                                : FileType::kVerilog);
  for (Block* block : blocks) {
    XLS_RETURN_IF_ERROR(BlockGenerator::Generate(
        block, &file, options, input_port_sv_types, output_port_sv_types));
    if (block != blocks.back()) {
      file.Add(file.Make<BlankLine>(SourceInfo()));
      file.Add(file.Make<BlankLine>(SourceInfo()));
    }
  }

  LineInfo line_info;
  std::string text = file.Emit(&line_info);
  if (verilog_line_map != nullptr) {
    for (const VastNode* vast_node : line_info.nodes()) {
      std::optional<std::vector<LineSpan>> spans =
          line_info.LookupNode(vast_node);
      if (!spans.has_value()) {
        return absl::InternalError(
            "Unbalanced calls to LineInfo::{Start, End}");
      }
      for (const LineSpan& span : spans.value()) {
        SourceInfo info = vast_node->loc();
        for (const SourceLocation& loc : info.locations) {
          int64_t line = static_cast<int32_t>(loc.lineno());
          VerilogLineMapping* mapping = verilog_line_map->add_mapping();
          mapping->set_source_file(
              top->package()->GetFilename(loc.fileno()).value_or(""));
          mapping->mutable_source_span()->set_line_start(line);
          mapping->mutable_source_span()->set_line_end(line);
          mapping->set_verilog_file("");  // to be updated later on
          mapping->mutable_verilog_span()->set_line_start(span.StartLine());
          mapping->mutable_verilog_span()->set_line_end(span.EndLine());
        }
      }
    }
  }

  VLOG(2) << "Verilog output:";
  XLS_VLOG_LINES(2, text);

  return text;
}

}  // namespace verilog
}  // namespace xls
