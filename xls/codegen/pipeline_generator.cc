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

#include "xls/codegen/pipeline_generator.h"

#include <algorithm>

#include "absl/algorithm/container.h"
#include "absl/status/statusor.h"
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
std::string PipelineSignalName(Node* node, int64_t stage) {
  return absl::StrFormat("p%d_%s", stage, SanitizeIdentifier(node->GetName()));
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

PipelineOptions& PipelineOptions::split_outputs(bool value) {
  split_outputs_ = value;
  return *this;
}

namespace {

// Class for constructing a pipeline. An abstraction containing the various
// inputs and options and temporary state used in the process.
class PipelineGenerator {
 public:
  PipelineGenerator(Function* func, const PipelineSchedule& schedule,
                    const PipelineOptions& options, VerilogFile* file)
      : func_(func), schedule_(schedule), options_(options), file_(file) {}

  absl::StatusOr<ModuleGeneratorResult> Run() {
    mb_ = absl::make_unique<ModuleBuilder>(
        options_.module_name().has_value() ? options_.module_name().value()
                                           : func_->name(),
        file_,
        /*use_system_verilog=*/options_.use_system_verilog(),
        /*clock_name=*/"clk", options_.reset());

    LogicRef* valid_load_enable = nullptr;
    LogicRef* manual_load_enable = nullptr;
    if (options_.control().has_value()) {
      if (options_.control()->has_valid()) {
        const ValidProto& valid_proto = options_.control()->valid();
        if (valid_proto.input_name().empty()) {
          return absl::InvalidArgumentError(
              "Must specify valid input signal name with valid pipeline "
              "register "
              "control");
        }
        valid_load_enable =
            mb_->AddInputPort(valid_proto.input_name(), /*bit_count=*/1);
      } else if (options_.control()->has_manual()) {
        const ManualPipelineControl& manual = options_.control()->manual();
        if (manual.input_name().empty()) {
          return absl::InvalidArgumentError(
              "Must specify manual control signal name with manual pipeline "
              "register"
              "control");
        }
        // Compute the number of pipeline registers. Unconditionally, there is
        // one register between each stage in the schedule, and optionally one
        // at the inputs and outputs.
        XLS_RET_CHECK_GT(schedule_.length(), 0);
        int64_t reg_count = schedule_.length() - 1;
        if (options_.flop_inputs()) {
          ++reg_count;
        }
        if (options_.flop_outputs()) {
          ++reg_count;
        }
        manual_load_enable = mb_->AddInputPort(manual.input_name(),
                                               /*bit_count=*/reg_count);
      }
    }

    // Returns the load enable signal for the pipeline registers at the end of
    // the given stage.
    auto get_load_enable = [&](int64_t stage) -> Expression* {
      if (valid_load_enable != nullptr) {
        // 'valid_load_enable' is updated to the latest flopped value in each
        // iteration of the loop. If the pipeline has a reset signal, OR in the
        // reset signal to enable pipeline flushing during reset.
        if (mb_->reset().has_value()) {
          return file_->LogicalOr(
              valid_load_enable,
              mb_->reset()->active_low
                  ? static_cast<Expression*>(
                        file_->LogicalNot(mb_->reset()->signal))
                  : static_cast<Expression*>(mb_->reset()->signal));
        }
        return valid_load_enable;
      }
      if (manual_load_enable != nullptr) {
        return file_->Make<Index>(manual_load_enable,
                                  file_->PlainLiteral(stage));
      }
      return nullptr;
    };

    // Map containing the VAST expression for each node. Values may be updated
    // as the pipeline is emitted. For example, a nodes value may be held in a
    // combinational expression, or a pipeline register depending upon the point
    // of the pipeline.
    absl::flat_hash_map<Node*, Expression*> node_expressions;

    // Create the input data ports.
    for (Param* param : func_->params()) {
      if (param->GetType()->GetFlatBitCount() == 0) {
        XLS_RET_CHECK_EQ(param->users().size(), 0);
        continue;
      }
      XLS_ASSIGN_OR_RETURN(
          node_expressions[param],
          mb_->AddInputPort(param->As<Param>()->name(), param->GetType()));
    }

    // Emit non-bits-typed literals separately as module-scoped constants.
    absl::flat_hash_set<Node*> module_constants;
    XLS_VLOG(4) << "Module constants:";
    for (Node* node : func_->nodes()) {
      if (node->Is<xls::Literal>() && !node->GetType()->IsBits() &&
          node->GetType()->GetFlatBitCount() > 0) {
        XLS_VLOG(4) << "  " << node->GetName();
        XLS_ASSIGN_OR_RETURN(
            node_expressions[node],
            mb_->DeclareModuleConstant(node->GetName(),
                                       node->As<xls::Literal>()->value()));
        module_constants.insert(node);
      }
    }
    // The set of nodes which are live out of the previous stage.
    std::vector<Node*> live_out_last_stage;

    int64_t stage = 0;
    if (options_.flop_inputs()) {
      XLS_VLOG(4) << "Flopping inputs.";
      std::vector<std::pair<Node*, Expression*>> assignments;
      for (Param* param : func_->params()) {
        if (param->GetType()->GetFlatBitCount() != 0) {
          assignments.push_back({param, node_expressions[param]});
        }
      }
      if (!assignments.empty()) {
        mb_->declaration_section()->Add<BlankLine>();
        mb_->declaration_section()->Add<Comment>(
            absl::StrFormat("===== Pipe stage %d:", stage));
        XLS_ASSIGN_OR_RETURN(
            std::vector<Expression*> register_refs,
            AddPipelineRegisters(assignments, stage, get_load_enable(stage)));
        for (int64_t i = 0; i < assignments.size(); ++i) {
          node_expressions[assignments[i].first] = register_refs[i];
        }
        if (valid_load_enable != nullptr) {
          XLS_ASSIGN_OR_RETURN(valid_load_enable,
                               AddValidRegister(valid_load_enable, stage));
        }
        stage++;
      }
    }

    // Construct the stages defined by the schedule.
    for (int64_t schedule_cycle = 0; schedule_cycle < schedule_.length();
         ++schedule_cycle) {
      XLS_VLOG(4) << "Building pipeline stage " << stage;
      if (stage != 0) {
        mb_->NewDeclarationAndAssignmentSections();
      }

      mb_->declaration_section()->Add<BlankLine>();
      mb_->declaration_section()->Add<Comment>(
          absl::StrFormat("===== Pipe stage %d:", stage));

      // Returns whether the given node is live out of this stage.
      auto is_live_out_of_stage = [&](Node* node) {
        if (module_constants.contains(node)) {
          return false;
        }
        if (node == func_->return_value()) {
          return true;
        }
        for (Node* user : node->users()) {
          if (schedule_.cycle(user) > schedule_cycle) {
            return true;
          }
        }
        return false;
      };

      // Identify nodes in this stage which should be named temporaries.
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
      //   (4) Is live out of the stage, OR
      //
      //   (5) Has an assigned name. This preserves names in the generated
      //       Verilog.
      absl::flat_hash_set<Node*> named_temps;
      for (Node* node : schedule_.nodes_in_cycle(schedule_cycle)) {
        if (node->Is<Param>() || module_constants.contains(node)) {
          continue;
        }
        if (!mb_->CanEmitAsInlineExpression(node, UsersInStage(node)) ||
            (FanoutInStage(node, schedule_cycle) > 1 &&
             !ShouldInlineExpressionIntoMultipleUses(node)) ||
            is_live_out_of_stage(node) || node->HasAssignedName()) {
          named_temps.insert(node);
        }
      }

      // Emit expressions/assignments for every node in this stage.
      for (Node* node : schedule_.nodes_in_cycle(schedule_cycle)) {
        if (node->Is<Param>() || module_constants.contains(node) ||
            node->GetType()->GetFlatBitCount() == 0) {
          continue;
        }

        std::vector<Expression*> inputs;
        for (Node* operand : node->operands()) {
          // Procs aren't yet plumbed through this generator, so just drop token
          // operands.
          if (operand->GetType()->IsToken()) {
            continue;
          }
          inputs.push_back(node_expressions.at(operand));
        }

        if (named_temps.contains(node)) {
          XLS_ASSIGN_OR_RETURN(
              node_expressions[node],
              mb_->EmitAsAssignment(PipelineSignalName(node, stage) + "_comb",
                                    node, inputs));
        } else {
          XLS_ASSIGN_OR_RETURN(node_expressions[node],
                               mb_->EmitAsInlineExpression(node, inputs));
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
      for (Node* node : schedule_.nodes_in_cycle(schedule_cycle)) {
        if (!module_constants.contains(node) && is_live_out_of_stage(node)) {
          live_out_nodes.push_back(node);
        }
      }

      if (!options_.flop_outputs() &&
          schedule_cycle == schedule_.length() - 1) {
        // This is the last stage and the outputs are not flopped so nothing
        // more to do.
        break;
      }

      if (!live_out_nodes.empty() &&
          (options_.flop_outputs() ||
           schedule_cycle != schedule_.length() - 1)) {
        // Add always flop block for the registers.
        mb_->NewDeclarationAndAssignmentSections();

        std::vector<std::pair<Node*, Expression*>> assignments;
        for (Node* node : live_out_nodes) {
          if (node->GetType()->GetFlatBitCount() > 0) {
            assignments.push_back({node, node_expressions.at(node)});
          }
        }
        XLS_ASSIGN_OR_RETURN(
            std::vector<Expression*> register_refs,
            AddPipelineRegisters(assignments, stage, get_load_enable(stage)));
        for (int64_t i = 0; i < assignments.size(); ++i) {
          node_expressions[assignments[i].first] = register_refs[i];
        }
      }

      if (valid_load_enable != nullptr) {
        XLS_ASSIGN_OR_RETURN(valid_load_enable,
                             AddValidRegister(valid_load_enable, stage));
      }

      // Emit Cover statements separately.
      for (Node* node : schedule_.nodes_in_cycle(schedule_cycle)) {
        if (node->Is<xls::Cover>()) {
          XLS_RETURN_IF_ERROR(mb_->EmitCover(
              node->As<xls::Cover>(), node_expressions.at(node->operand(1))));
        }
      }

      live_out_last_stage = std::move(live_out_nodes);
      stage++;
    }

    if (valid_load_enable != nullptr) {
      XLS_CHECK(options_.control().has_value());
      if (!options_.control()->valid().output_name().empty()) {
        XLS_RETURN_IF_ERROR(
            mb_->AddOutputPort(options_.control()->valid().output_name(),
                               /*bit_count=*/1, valid_load_enable));
      }
    }

    // Assign the output wire to the pipeline-registered output value.
    if (func_->return_value()->GetType()->GetFlatBitCount() > 0) {
      if (options_.split_outputs() &&
          func_->return_value()->GetType()->IsTuple()) {
        TupleType* output_type =
            func_->return_value()->GetType()->AsTupleOrDie();
        for (int64_t i = 0; i < output_type->size(); ++i) {
          int64_t start = GetFlatBitIndexOfElement(output_type, i);
          int64_t width = output_type->element_type(i)->GetFlatBitCount();
          XLS_RETURN_IF_ERROR(mb_->AddOutputPort(
              absl::StrFormat("out_%d", i), output_type->element_type(i),
              mb_->module()->file()->Slice(
                  node_expressions.at(func_->return_value())
                      ->AsIndexableExpressionOrDie(),
                  start + width - 1, start)));
        }
      } else {
        XLS_RETURN_IF_ERROR(
            mb_->AddOutputPort("out", func_->return_value()->GetType(),
                               node_expressions.at(func_->return_value())));
      }
    }

    std::string text = file_->Emit();
    XLS_ASSIGN_OR_RETURN(ModuleSignature signature,
                         BuildSignature(/*latency=*/stage));
    return ModuleGeneratorResult{text, signature};
  }

  // Builds and returns a module signature for the given latency.
  absl::StatusOr<ModuleSignature> BuildSignature(int64_t latency) {
    ModuleSignatureBuilder sig_builder(mb_->module()->name());
    sig_builder.WithClock("clk");
    for (Param* param : func_->params()) {
      sig_builder.AddDataInput(param->name(),
                               param->GetType()->GetFlatBitCount());
    }
    if (options_.split_outputs() &&
        func_->return_value()->GetType()->IsTuple()) {
      TupleType* output_type = func_->return_value()->GetType()->AsTupleOrDie();
      for (int64_t i = 0; i < output_type->size(); ++i) {
        sig_builder.AddDataOutput(
            absl::StrFormat("out_%d", i),
            output_type->element_type(i)->GetFlatBitCount());
      }
    } else {
      sig_builder.AddDataOutput(
          "out", func_->return_value()->GetType()->GetFlatBitCount());
    }
    sig_builder.WithFunctionType(func_->GetType());
    sig_builder.WithPipelineInterface(
        /*latency=*/latency,
        /*initiation_interval=*/1, options_.control());

    if (options_.reset().has_value()) {
      sig_builder.WithReset(options_.reset()->name(),
                            options_.reset()->asynchronous(),
                            options_.reset()->active_low());
    }
    return sig_builder.Build();
  }

  // Returns number of uses of the node's value in this stage. This includes the
  // pipeline register if the node's value is used in a later stage.
  int64_t FanoutInStage(Node* node, int64_t stage) {
    XLS_CHECK_EQ(schedule_.cycle(node), stage);
    int64_t fanout = 0;
    if (absl::c_any_of(node->users(),
                       [&](Node* n) { return schedule_.cycle(n) > stage; })) {
      fanout = 1;
    }
    for (Node* user : node->users()) {
      if (schedule_.cycle(user) == stage) {
        // Count each operand separately in case the node appears in multiple
        // operand slots.
        fanout += user->OperandInstanceCount(node);
      }
    }
    return fanout;
  }

  // Returns the users of the given node which are scheduled in the same stage.
  std::vector<Node*> UsersInStage(Node* node) {
    std::vector<Node*> users;
    for (Node* user : node->users()) {
      if (schedule_.cycle(user) == schedule_.cycle(node)) {
        users.push_back(user);
      }
    }
    return users;
  }

  // Adds pipeline registers to the module for the given stage. The registers to
  // define are given as pairs of Node* and the expression to assign to the
  // value corresponding to the node. The registers use the supplied clock and
  // optional load enable (can be null). Returns references corresponding to the
  // defined registers.
  absl::StatusOr<std::vector<Expression*>> AddPipelineRegisters(
      absl::Span<const std::pair<Node*, Expression*>> assignments,
      int64_t stage, Expression* load_enable) {
    // Add always flop block for the registers.
    mb_->NewDeclarationAndAssignmentSections();

    mb_->declaration_section()->Add<BlankLine>();
    mb_->declaration_section()->Add<Comment>(
        absl::StrFormat("Registers for pipe stage %d:", stage));

    std::vector<ModuleBuilder::Register> registers;
    std::vector<Expression*> register_refs;
    for (const auto& pair : assignments) {
      Node* node = pair.first;
      Expression* rhs = pair.second;
      if (node->GetType()->GetFlatBitCount() > 0) {
        if (options_.reset().has_value() &&
            options_.reset()->reset_data_path()) {
          return absl::UnimplementedError(
              "Reset of data path not supported for pipeline generator.");
        }
        XLS_ASSIGN_OR_RETURN(
            ModuleBuilder::Register reg,
            mb_->DeclareRegister(PipelineSignalName(node, stage),
                                 node->GetType(), rhs));
        reg.load_enable = load_enable;
        registers.push_back(reg);
        register_refs.push_back(registers.back().ref);
      }
    }
    XLS_RETURN_IF_ERROR(mb_->AssignRegisters(registers));

    return register_refs;
  }

  // Adds a "valid" signal register for the given stage using the given clock
  // signal. Returns a reference to the defined register.
  absl::StatusOr<LogicRef*> AddValidRegister(LogicRef* valid_load_enable,
                                             int64_t stage) {
    // Add always flop block for the valid signal. Add it separately from the
    // other pipeline register because it does not use a load_enable signal
    // like the other pipeline registers.
    mb_->NewDeclarationAndAssignmentSections();

    mb_->declaration_section()->Add<BlankLine>();

    Expression* reset_value =
        mb_->reset().has_value() ? file_->Literal(0, /*bit_count=*/1) : nullptr;

    XLS_ASSIGN_OR_RETURN(
        ModuleBuilder::Register valid_load_enable_register,
        mb_->DeclareRegister(absl::StrFormat("p%d_valid", stage),
                             /*bit_count=*/1, valid_load_enable, reset_value));
    XLS_RETURN_IF_ERROR(mb_->AssignRegisters({valid_load_enable_register}));
    return valid_load_enable_register.ref;
  }

 private:
  Function* func_;
  const PipelineSchedule& schedule_;
  const PipelineOptions& options_;
  VerilogFile* file_;

  std::unique_ptr<ModuleBuilder> mb_;
};

}  // namespace

absl::StatusOr<ModuleGeneratorResult> ToPipelineModuleText(
    const PipelineSchedule& schedule, Function* func,
    const PipelineOptions& options) {
  XLS_VLOG(2) << "Generating pipelined module for function:";
  XLS_VLOG_LINES(2, func->DumpIr());
  XLS_VLOG_LINES(2, schedule.ToString());

  VerilogFile file(options.use_system_verilog());
  PipelineGenerator generator(func, schedule, options, &file);
  XLS_ASSIGN_OR_RETURN(ModuleGeneratorResult result, generator.Run());

  XLS_VLOG(2) << "Signature:";
  XLS_VLOG_LINES(2, result.signature.ToString());
  XLS_VLOG(2) << "Verilog output:";
  XLS_VLOG_LINES(2, result.verilog_text);
  return result;
}

}  // namespace verilog
}  // namespace xls
