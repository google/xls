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

#include "xls/codegen/sequential_generator.h"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "xls/codegen/finite_state_machine.h"
#include "xls/codegen/module_builder.h"
#include "xls/codegen/module_signature.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/codegen/pipeline_generator.h"
#include "xls/codegen/vast.h"
#include "xls/common/integral_types.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/status/statusor.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/delay_model/delay_estimators.h"
#include "xls/ir/function.h"
#include "xls/ir/type.h"
#include "xls/passes/passes.h"
#include "xls/scheduling/pipeline_schedule.h"

namespace xls {
namespace verilog {

xabsl::StatusOr<SequentialModuleBuilder::StridedCounterReferences>
SequentialModuleBuilder::AddStaticStridedCounter(std::string name, int64 stride,
                                                 int64 value_limit_exclusive,
                                                 LogicRef* clk,
                                                 LogicRef* set_zero_arg,
                                                 LogicRef* increment_arg) {
  // Create references.
  StridedCounterReferences refs;
  refs.set_zero = set_zero_arg;
  refs.increment = increment_arg;

  // Check if specification is valid.
  if (value_limit_exclusive <= 0) {
    return absl::UnimplementedError(
        "Tried to generate static strided counter with non-positive "
        "value_limit_exlusive - not currently supported.");
  }
  if (stride <= 0) {
    return absl::UnimplementedError(
        "Tried to generate static strided counter with non-positive stride - "
        "not currently supported.");
  }

  // Pretty print verilog.
  module_builder_->declaration_section()->Add<BlankLine>();
  module_builder_->declaration_section()->Add<Comment>(
      "Declarations for counter " + name);
  module_builder_->assignment_section()->Add<BlankLine>();
  module_builder_->assignment_section()->Add<Comment>(
      "Assignments for counter " + name);

  // Determine counter value limit and the number of bits needed to represent
  // this number.
  int64 value_limit_exclusive_minus = value_limit_exclusive - 1;
  int64 max_inclusive_value =
      value_limit_exclusive_minus - ((value_limit_exclusive_minus) % stride);
  int64 num_counter_bits = Bits::MinBitCountUnsigned(max_inclusive_value);
  XLS_CHECK_GT(num_counter_bits, 0);

  // Create the counter.
  // Note: Need to "forward-declare" counter_wire so that we can compute
  // counter + stride before calling DeclareRegister, which gives us the counter
  // register.
  LogicRef* counter_wire =
      module_builder_->DeclareVariable(name + "_wire", num_counter_bits);
  Expression* counter_next =
      file_.Ternary(refs.set_zero, file_.PlainLiteral(0),
                    file_.Add(counter_wire, file_.PlainLiteral(stride)));
  XLS_ASSIGN_OR_RETURN(
      ModuleBuilder::Register counter_register,
      module_builder_->DeclareRegister(name, num_counter_bits, counter_next));
  AddContinuousAssignment(counter_wire, counter_register.ref);
  refs.value = counter_register.ref;

  // Add counter always-block.
  Expression* load_enable = file_.BitwiseOr(refs.increment, refs.set_zero);
  XLS_RETURN_IF_ERROR(
      module_builder_->AssignRegisters(clk, {counter_register}, load_enable));

  // Compare counter value to maximum value.
  refs.holds_max_inclusive_value = DeclareVariableAndAssign(
      name + "_holds_max_inclusive_value",
      file_.Equals(refs.value, file_.PlainLiteral(max_inclusive_value)),
      num_counter_bits);

  return refs;
}

// Generates the signature for the top-level module.
xabsl::StatusOr<std::unique_ptr<ModuleSignature>>
SequentialModuleBuilder::GenerateModuleSignature() {
  std::string module_name = sequential_options_.module_name().has_value()
                                ? sequential_options_.module_name().value()
                                : loop_->GetName() + "_sequential_module";
  ModuleSignatureBuilder sig_builder(SanitizeIdentifier(module_name));

  // Default Inputs.
  // We conservatively apply SanitizeIdentifier to all names, including
  // constants, so that signature names match verilog names.
  sig_builder.WithClock(SanitizeIdentifier("clk"));
  for (const Node* op_in : loop_->operands()) {
    sig_builder.AddDataInput(SanitizeIdentifier(op_in->GetName() + "_in"),
                             op_in->GetType()->GetFlatBitCount());
  }

  // Default Outputs.
  sig_builder.AddDataOutput(SanitizeIdentifier(loop_->GetName() + "_out"),
                            loop_->GetType()->GetFlatBitCount());

  // Reset.
  if (sequential_options_.reset().has_value()) {
    sig_builder.WithReset(
        SanitizeIdentifier(sequential_options_.reset()->name()),
        sequential_options_.reset()->asynchronous(),
        sequential_options_.reset()->active_low());
  }

  // TODO(jbaileyhandle): Add options for other interfaces.
  std::string ready_in_name = SanitizeIdentifier("ready_in");
  std::string valid_in_name = SanitizeIdentifier("valid_in");
  std::string ready_out_name = SanitizeIdentifier("ready_out");
  std::string valid_out_name = SanitizeIdentifier("valid_out");
  sig_builder.WithReadyValidInterface(ready_in_name, valid_in_name,
                                      ready_out_name, valid_out_name);

  // Build signature.
  std::unique_ptr<ModuleSignature> signature =
      absl::make_unique<ModuleSignature>();
  XLS_ASSIGN_OR_RETURN(*signature, sig_builder.Build());
  return std::move(signature);
}

// Generates a pipeline module that implements the loop's body.
xabsl::StatusOr<std::unique_ptr<ModuleGeneratorResult>>
SequentialModuleBuilder::GenerateLoopBodyPipeline(
    const SchedulingOptions& scheduling_options,
    const DelayEstimator& delay_estimator) {
  // Set pipeline options.
  PipelineOptions pipeline_options;
  pipeline_options.flop_inputs(false).flop_outputs(false).use_system_verilog(
      sequential_options_.use_system_verilog());
  if (sequential_options_.reset().has_value()) {
    pipeline_options.reset(sequential_options_.reset().value());
  }

  // Get schedule.
  Function* loop_body_function = loop_->body();
  XLS_ASSIGN_OR_RETURN(
      PipelineSchedule schedule,
      PipelineSchedule::Run(loop_body_function, delay_estimator,
                            scheduling_options));
  XLS_RETURN_IF_ERROR(schedule.Verify());

  std::unique_ptr<ModuleGeneratorResult> result =
      absl::make_unique<ModuleGeneratorResult>();
  XLS_ASSIGN_OR_RETURN(
      *result,
      ToPipelineModuleText(schedule, loop_body_function, pipeline_options));
  return std::move(result);
}

// Initializes the module builder according to the signature.
absl::Status SequentialModuleBuilder::InitializeModuleBuilder(
    const ModuleSignature& signature) {
  // Make builder.
  XLS_RET_CHECK(signature.proto().has_module_name());
  module_builder_ = absl::make_unique<ModuleBuilder>(
      signature.proto().module_name(), &file_,
      sequential_options_.use_system_verilog());

  auto add_input_port = [&](absl::string_view name, int64 num_bits) {
    return module_builder_->AddInputPort(SanitizeIdentifier(name), num_bits);
  };
  auto add_output_port = [&](absl::string_view name, int64 num_bits) {
    return module_builder_->module()->AddPort(
        Direction::kOutput, SanitizeIdentifier(name), num_bits);
  };

  // Clock.
  XLS_RET_CHECK(signature.proto().has_clock_name());
  port_references_.clk = add_input_port(signature.proto().clock_name(), 1);

  // Reset.
  if (signature.proto().has_reset()) {
    XLS_RET_CHECK(signature.proto().reset().has_name());
    port_references_.reset =
        add_input_port(signature.proto().reset().name(), 1);
  }

  // Ready-valid interface.
  if (signature.proto().has_ready_valid()) {
    const ReadyValidInterface& rv_interface = signature.proto().ready_valid();
    XLS_RET_CHECK(rv_interface.has_input_ready());
    XLS_RET_CHECK(rv_interface.has_input_valid());
    XLS_RET_CHECK(rv_interface.has_output_ready());
    XLS_RET_CHECK(rv_interface.has_output_valid());
    port_references_.ready_in = add_output_port(rv_interface.input_ready(), 1);
    port_references_.valid_in = add_input_port(rv_interface.input_valid(), 1);
    port_references_.ready_out = add_input_port(rv_interface.output_ready(), 1);
    port_references_.valid_out =
        add_output_port(rv_interface.output_valid(), 1);
  }

  // Data I/O.
  for (const PortProto& in_port : signature.data_inputs()) {
    XLS_RET_CHECK(in_port.has_direction());
    XLS_RET_CHECK_EQ(in_port.direction(), DIRECTION_INPUT);
    XLS_RET_CHECK(in_port.has_name());
    XLS_RET_CHECK(in_port.has_width());
    port_references_.data_in.push_back(
        add_input_port(in_port.name(), in_port.width()));
  }
  for (const PortProto& out_port : signature.data_outputs()) {
    XLS_RET_CHECK(out_port.has_direction());
    XLS_RET_CHECK_EQ(out_port.direction(), DIRECTION_OUTPUT);
    XLS_RET_CHECK(out_port.has_name());
    XLS_RET_CHECK(out_port.has_width());
    port_references_.data_out.push_back(
        add_output_port(out_port.name(), out_port.width()));
  }

  return absl::OkStatus();
}

xabsl::StatusOr<ModuleGeneratorResult> ToSequentialModuleText(Function* func) {
  return absl::UnimplementedError("Sequential generator not supported yet.");
}

}  // namespace verilog
}  // namespace xls
