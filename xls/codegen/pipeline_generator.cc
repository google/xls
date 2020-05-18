// Copyright 2020 Google LLC
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

#include "xls/codegen/pipeline_generator.h"

#include <algorithm>

#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xls/codegen/finite_state_machine.h"
#include "xls/codegen/flattening.h"
#include "xls/codegen/module_builder.h"
#include "xls/codegen/node_expressions.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/delay_model/delay_estimators.h"
#include "xls/ir/node_iterator.h"
#include "xls/ir/package.h"
#include "xls/scheduling/pipeline_schedule.h"

namespace xls {
namespace verilog {
namespace {

// Returns pipeline-stage prefixed signal name for the given node. For
// example: p3_foo.
std::string PipelineSignalName(Node* node, int64 stage) {
  return absl::StrFormat("p%d_%s", stage, SanitizeIdentifier(node->GetName()));
}

// Returns number of uses of the node's value in this stage. This includes the
// pipeline register if the node's value is used in a later stage.
int64 FanoutInStage(Node* node, const PipelineSchedule& schedule, int64 stage) {
  XLS_CHECK_EQ(schedule.cycle(node), stage);
  int64 fanout = 0;
  if (absl::c_any_of(node->users(),
                     [&](Node* n) { return schedule.cycle(n) > stage; })) {
    fanout = 1;
  }
  for (Node* user : node->users()) {
    if (schedule.cycle(user) == stage) {
      // Count each operand separately in case the node appears in multiple
      // operand slots.
      fanout += user->OperandInstanceCount(node);
    }
  }
  return fanout;
}

// Returns the users of the given node which are scheduled in the same stage.
std::vector<Node*> UsersInStage(Node* node, const PipelineSchedule& schedule) {
  std::vector<Node*> users;
  for (Node* user : node->users()) {
    if (schedule.cycle(user) == schedule.cycle(node)) {
      users.push_back(user);
    }
  }
  return users;
}

// Builds and returns a module signature for the given function, latency, and
// options.
xabsl::StatusOr<ModuleSignature> BuildSignature(
    absl::string_view module_name, Function* function, int64 latency,
    const PipelineOptions& options) {
  ModuleSignatureBuilder sig_builder(module_name);
  sig_builder.WithClock("clk");
  for (Param* param : function->params()) {
    sig_builder.AddDataInput(param->name(),
                             param->GetType()->GetFlatBitCount());
  }
  sig_builder.AddDataOutput(
      "out", function->return_value()->GetType()->GetFlatBitCount());
  sig_builder.WithFunctionType(function->GetType());
  sig_builder.WithPipelineInterface(
      /*latency=*/latency,
      /*initiation_interval=*/1,
      options.control().has_value()
          ? absl::optional<PipelineControl>(*options.control())
          : absl::nullopt);
  return sig_builder.Build();
}

}  // namespace

PipelineOptions& PipelineOptions::manual_control(absl::string_view input_name) {
  if (!pipeline_control_.has_value()) {
    pipeline_control_ = PipelineControl();
  }
  pipeline_control_->mutable_manual()->set_input_name(
      ToProtoString(input_name));
  return *this;
}

absl::optional<ManualPipelineControl> PipelineOptions::manual_control() const {
  if (!pipeline_control_.has_value() || !pipeline_control_->has_manual()) {
    return absl::nullopt;
  }
  return pipeline_control_->manual();
}

PipelineOptions& PipelineOptions::valid_control(
    absl::string_view input_name,
    absl::optional<absl::string_view> output_name) {
  if (!pipeline_control_.has_value()) {
    pipeline_control_ = PipelineControl();
  }
  ValidProto* valid = pipeline_control_->mutable_valid();
  valid->set_input_name(ToProtoString(input_name));
  if (output_name.has_value()) {
    valid->set_output_name(ToProtoString(*output_name));
  }
  return *this;
}

absl::optional<ValidProto> PipelineOptions::valid_control() const {
  if (!pipeline_control_.has_value() || !pipeline_control_->has_valid()) {
    return absl::nullopt;
  }
  return pipeline_control_->valid();
}

PipelineOptions& PipelineOptions::use_system_verilog(bool value) {
  use_system_verilog_ = value;
  return *this;
}

PipelineOptions& PipelineOptions::flop_inputs(bool value) {
  flop_inputs_ = value;
  return *this;
}

PipelineOptions& PipelineOptions::flop_outputs(bool value) {
  flop_outputs_ = value;
  return *this;
}

namespace {

// Adds pipeline registers to the module for the given stage. The registers to
// define are given as pairs of Node* and the expression to assign to the value
// corresponding to the node. The registers use the supplied clock and optional
// load enable (can be null). Returns references corresponding to the defined
// registers.
xabsl::StatusOr<std::vector<Expression*>> AddPipelineRegisters(
    absl::Span<const std::pair<Node*, Expression*>> assignments, int64 stage,
    LogicRef* clk, Expression* load_enable, ModuleBuilder* mb) {
  // Add always flop block for the registers.
  mb->NewDeclarationAndAssignmentSections();

  mb->declaration_section()->Add<BlankLine>();
  mb->declaration_section()->Add<Comment>(
      absl::StrFormat("Registers for pipe stage %d:", stage));

  std::vector<ModuleBuilder::Register> registers;
  std::vector<Expression*> register_refs;
  for (const auto& pair : assignments) {
    Node* node = pair.first;
    Expression* rhs = pair.second;
    if (node->GetType()->GetFlatBitCount() > 0) {
      XLS_ASSIGN_OR_RETURN(ModuleBuilder::Register reg,
                           mb->DeclareRegister(PipelineSignalName(node, stage),
                                               node->GetType(), rhs));
      registers.push_back(reg);
      register_refs.push_back(registers.back().ref);
    }
  }
  XLS_RETURN_IF_ERROR(mb->AssignRegisters(clk, registers, load_enable));

  return register_refs;
}

// Adds a "valid" signal register for the given stage using the given clock
// signal. Returns a reference to the defined register.
xabsl::StatusOr<LogicRef*> AddValidRegister(LogicRef* valid_load_enable,
                                            int64 stage, LogicRef* clk,
                                            ModuleBuilder* mb) {
  // Add always flop block for the valid signal. Add it separately from the
  // other pipeline register because it does not use a load_enable signal
  // like the other pipeline registers.
  mb->NewDeclarationAndAssignmentSections();

  mb->declaration_section()->Add<BlankLine>();
  XLS_ASSIGN_OR_RETURN(ModuleBuilder::Register valid_load_enable_register,
                       mb->DeclareRegister(absl::StrFormat("p%d_valid", stage),
                                           /*bit_count=*/1, valid_load_enable));
  XLS_RETURN_IF_ERROR(mb->AssignRegisters(clk, {valid_load_enable_register}));
  return valid_load_enable_register.ref;
}

}  // namespace

xabsl::StatusOr<ModuleGeneratorResult> ToPipelineModuleText(
    const PipelineSchedule& schedule, Function* func,
    const PipelineOptions& options) {
  XLS_VLOG(2) << "Generating pipelined module for function:";
  XLS_VLOG_LINES(2, func->DumpIr());
  XLS_VLOG_LINES(2, schedule.ToString());

  VerilogFile file;

  // TODO(meheff): Implement reset.
  if (options.reset().has_value()) {
    return absl::UnimplementedError(
        "Reset not supported for pipeline generator.");
  }

  // Create a module implementing the Function.
  ModuleBuilder mb(
      options.module_name().has_value() ? *options.module_name() : func->name(),
      &file,
      /*use_system_verilog=*/options.use_system_verilog());
  LogicRef* clk = mb.AddInputPort("clk", /*bit_count=*/1);
  LogicRef* valid_load_enable = nullptr;
  LogicRef* manual_load_enable = nullptr;
  if (options.control().has_value()) {
    if (options.control()->has_valid()) {
      const ValidProto& valid_proto = options.control()->valid();
      if (valid_proto.input_name().empty()) {
        return absl::InvalidArgumentError(
            "Must specify valid input signal name with valid pipeline register "
            "control");
      }
      valid_load_enable =
          mb.AddInputPort(valid_proto.input_name(), /*bit_count=*/1);
    } else if (options.control()->has_manual()) {
      const ManualPipelineControl& manual = options.control()->manual();
      if (manual.input_name().empty()) {
        return absl::InvalidArgumentError(
            "Must specify manual control signal name with manual pipeline "
            "register"
            "control");
      }
      manual_load_enable = mb.AddInputPort(manual.input_name(),
                                           /*bit_count=*/schedule.length() + 1);
    }
  }

  // Returns the load enable signal for the pipeline registers at the end of the
  // given stage.
  auto get_load_enable = [&](int64 stage) -> Expression* {
    if (valid_load_enable != nullptr) {
      // 'valid_load_enable' is updated to the latest flopped value in
      // each iteration of the loop.
      return valid_load_enable;
    }
    if (manual_load_enable != nullptr) {
      return file.Make<Index>(manual_load_enable, file.PlainLiteral(stage));
    }
    return nullptr;
  };

  // Map containing the VAST expression for each node. Values may be updated as
  // the pipeline is emitted. For example, a nodes value may be held in a
  // combinational expression, or a pipeline register depending upon the point
  // of the pipeline.
  absl::flat_hash_map<Node*, Expression*> node_expressions;

  // Create the input data ports.
  for (Param* param : func->params()) {
    if (param->GetType()->GetFlatBitCount() == 0) {
      XLS_RET_CHECK_EQ(param->users().size(), 0);
      continue;
    }
    XLS_ASSIGN_OR_RETURN(
        node_expressions[param],
        mb.AddInputPort(param->As<Param>()->name(), param->GetType()));
  }

  // Emit non-bits-typed literals separately as module-scoped constants.
  absl::flat_hash_set<Node*> module_constants;
  XLS_VLOG(4) << "Module constants:";
  for (Node* node : func->nodes()) {
    if (node->Is<xls::Literal>() && !node->GetType()->IsBits() &&
        node->GetType()->GetFlatBitCount() > 0) {
      XLS_VLOG(4) << "  " << node->GetName();
      XLS_ASSIGN_OR_RETURN(
          node_expressions[node],
          mb.DeclareModuleConstant(node->GetName(),
                                   node->As<xls::Literal>()->value()));
      module_constants.insert(node);
    }
  }
  // The set of nodes which are live out of the previous stage.
  std::vector<Node*> live_out_last_stage;

  int64 stage = 0;
  if (options.flop_inputs()) {
    XLS_VLOG(4) << "Flopping inputs.";
    std::vector<std::pair<Node*, Expression*>> assignments;
    for (Param* param : func->params()) {
      if (param->GetType()->GetFlatBitCount() != 0) {
        assignments.push_back({param, node_expressions[param]});
      }
    }
    if (!assignments.empty()) {
      mb.declaration_section()->Add<BlankLine>();
      mb.declaration_section()->Add<Comment>(
          absl::StrFormat("===== Pipe stage %d:", stage));
      XLS_ASSIGN_OR_RETURN(std::vector<Expression*> register_refs,
                           AddPipelineRegisters(assignments, stage, clk,
                                                get_load_enable(stage), &mb));
      for (int64 i = 0; i < assignments.size(); ++i) {
        node_expressions[assignments[i].first] = register_refs[i];
      }
      if (valid_load_enable != nullptr) {
        XLS_ASSIGN_OR_RETURN(
            valid_load_enable,
            AddValidRegister(valid_load_enable, stage, clk, &mb));
      }
      stage++;
    }
  }

  // Construct the stages defined by the schedule.
  for (int64 schedule_cycle = 0; schedule_cycle < schedule.length();
       ++schedule_cycle) {
    XLS_VLOG(4) << "Building pipeline stage " << stage;
    if (stage != 0) {
      mb.NewDeclarationAndAssignmentSections();
    }

    mb.declaration_section()->Add<BlankLine>();
    mb.declaration_section()->Add<Comment>(
        absl::StrFormat("===== Pipe stage %d:", stage));

    // Returns whether the given node is live out of this stage.
    auto is_live_out_of_stage = [&](Node* node) {
      if (module_constants.contains(node)) {
        return false;
      }
      if (node == func->return_value()) {
        return true;
      }
      for (Node* user : node->users()) {
        if (schedule.cycle(user) > schedule_cycle) {
          return true;
        }
      }
      return false;
    };

    // Identify nodes in this stage which must be named temporaries.
    // Conditions:
    //
    //   (0) Is not a module constant or parameter, AND one of the following
    //       is true:
    //
    //   (1) Is array-shaped, OR
    //
    //   (2) Has multiple in-stage uses and is not trivially inlinable (e.g.,
    //       unary negation), OR
    //
    //   (3) Has an in-stage use that needs a named reference, OR
    //
    //   (4) Is live out of the stage.
    absl::flat_hash_set<Node*> named_temps;
    for (Node* node : schedule.nodes_in_cycle(schedule_cycle)) {
      if (node->Is<Param>() || module_constants.contains(node)) {
        continue;
      }
      if (!mb.CanEmitAsInlineExpression(node, UsersInStage(node, schedule)) ||
          (FanoutInStage(node, schedule, schedule_cycle) > 1 &&
           !ShouldInlineExpressionIntoMultipleUses(node)) ||
          is_live_out_of_stage(node)) {
        named_temps.insert(node);
      }
    }

    // Emit expressions/assignments for every node in this stage.
    for (Node* node : schedule.nodes_in_cycle(schedule_cycle)) {
      if (node->Is<Param>() || module_constants.contains(node) ||
          node->GetType()->GetFlatBitCount() == 0) {
        continue;
      }

      std::vector<Expression*> inputs;
      for (Node* operand : node->operands()) {
        inputs.push_back(node_expressions.at(operand));
      }

      if (named_temps.contains(node)) {
        XLS_ASSIGN_OR_RETURN(
            node_expressions[node],
            mb.EmitAsAssignment(PipelineSignalName(node, stage) + "_comb", node,
                                inputs));
      } else {
        XLS_ASSIGN_OR_RETURN(node_expressions[node],
                             mb.EmitAsInlineExpression(node, inputs));
      }
    }

    // Generate the set of pipeline registers at the end of this stage. These
    // includes all live-out values from this stage including those nodes
    // scheduled in earlier stages.
    std::vector<Node*> live_out_nodes;
    for (Node* node : live_out_last_stage) {
      if (is_live_out_of_stage(node)) {
        live_out_nodes.push_back(node);
      }
    }
    for (Node* node : schedule.nodes_in_cycle(schedule_cycle)) {
      if (!module_constants.contains(node) && is_live_out_of_stage(node)) {
        live_out_nodes.push_back(node);
      }
    }

    if (!options.flop_outputs() && schedule_cycle == schedule.length() - 1) {
      // This is the last stage and the outputs are not flopped so nothing more
      // to do.
      break;
    }

    if (!live_out_nodes.empty() &&
        (options.flop_outputs() || schedule_cycle != schedule.length() - 1)) {
      // Add always flop block for the registers.
      mb.NewDeclarationAndAssignmentSections();

      std::vector<std::pair<Node*, Expression*>> assignments;
      for (Node* node : live_out_nodes) {
        if (node->GetType()->GetFlatBitCount() > 0) {
          assignments.push_back({node, node_expressions.at(node)});
        }
      }
      XLS_ASSIGN_OR_RETURN(std::vector<Expression*> register_refs,
                           AddPipelineRegisters(assignments, stage, clk,
                                                get_load_enable(stage), &mb));
      for (int64 i = 0; i < assignments.size(); ++i) {
        node_expressions[assignments[i].first] = register_refs[i];
      }
    }

    if (valid_load_enable != nullptr) {
      XLS_ASSIGN_OR_RETURN(
          valid_load_enable,
          AddValidRegister(valid_load_enable, stage, clk, &mb));
    }

    live_out_last_stage = std::move(live_out_nodes);
    stage++;
  }

  if (valid_load_enable != nullptr) {
    XLS_CHECK(options.control().has_value());
    if (!options.control()->valid().output_name().empty()) {
      XLS_RETURN_IF_ERROR(
          mb.AddOutputPort(options.control()->valid().output_name(),
                           /*bit_count=*/1, valid_load_enable));
    }
  }

  // Assign the output wire to the pipeline-registered output value.
  if (func->return_value()->GetType()->GetFlatBitCount() > 0) {
    XLS_RETURN_IF_ERROR(
        mb.AddOutputPort("out", func->return_value()->GetType(),
                         node_expressions.at(func->return_value())));
  }

  std::string text = file.Emit();
  XLS_VLOG(2) << "Verilog output:";
  XLS_VLOG_LINES(2, text);

  XLS_ASSIGN_OR_RETURN(
      ModuleSignature signature,
      BuildSignature(mb.module()->name(), func, /*latency=*/stage, options));
  return ModuleGeneratorResult{text, signature};
}

xabsl::StatusOr<ModuleGeneratorResult> ScheduleAndGeneratePipelinedModule(
    Package* package, int64 clock_period_ps,
    absl::optional<ResetProto> reset_proto,
    absl::optional<ValidProto> valid_proto,
    absl::optional<absl::string_view> module_name) {
  XLS_ASSIGN_OR_RETURN(Function * f, package->EntryFunction());
  XLS_ASSIGN_OR_RETURN(
      PipelineSchedule schedule,
      PipelineSchedule::Run(
          f, GetStandardDelayEstimator(),
          SchedulingOptions().clock_period_ps(clock_period_ps)));
  PipelineOptions options;
  if (reset_proto.has_value()) {
    options.reset(*reset_proto);
  }
  if (valid_proto.has_value()) {
    options.valid_control(valid_proto->input_name(),
                          valid_proto->output_name());
  }
  if (module_name.has_value()) {
    options.module_name();
  }
  return ToPipelineModuleText(schedule, f, options);
}

}  // namespace verilog
}  // namespace xls
