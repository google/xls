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

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/combinational_generator.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/codegen/op_override_impls.h"
#include "xls/codegen/pipeline_generator.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/delay_model/delay_estimators.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/verifier.h"
#include "xls/passes/standard_pipeline.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/tools/scheduling_options_flags.h"

const char kUsage[] = R"(
Generates Verilog RTL from a given IR file. Writes a Verilog file and a module
signature describing the module interface to a specified location. Example
invocations:

Emit combinational module:
   codegen_main --generator=combinational --output_verilog_path=DIR IR_FILE

Emit a feed-forward pipelined module:
   codegen_main --generator=pipeline \
       --clock_period_ps=500 \
       --pipeline_stages=7 \
       IR_FILE
)";

// LINT.IfChange
ABSL_FLAG(
    std::string, output_verilog_path, "",
    "Specific output path for the Verilog generated. If not specified then "
    "Verilog is written to stdout.");
ABSL_FLAG(std::string, output_schedule_path, "",
          "Specific output path for the generated pipeline schedule. "
          "If not specified, then no schedule is output.");
ABSL_FLAG(std::string, output_block_ir_path, "",
          "Path to write the block-level IR.");
ABSL_FLAG(
    std::string, output_signature_path, "",
    "Specific output path for the module signature. If not specified then "
    "no module signature is generated.");
ABSL_FLAG(std::string, output_verilog_line_map_path, "",
          "Specific output path for Verilog line map. If not specified then "
          "Verilog line map is not generated.");
ABSL_FLAG(std::string, top, "",
          "Top entity of the package to generate the (System)Verilog code.");
ABSL_FLAG(std::string, generator, "pipeline",
          "The generator to use when emitting the device function. Valid "
          "values: pipeline, combinational.");
ABSL_FLAG(
    std::string, input_valid_signal, "",
    "If specified, the emitted module will use an external \"valid\" signal "
    "as the load enable for pipeline registers. The flag value is the "
    "name of the input port for this signal.");
ABSL_FLAG(
    std::string, output_valid_signal, "",
    "The name of the output port which holds the pipelined valid signal.");
ABSL_FLAG(
    std::string, manual_load_enable_signal, "",
    "If specified the load-enable of the pipeline registers of each stage is "
    "controlled via an input port of the indicated name. The width of the "
    "input port is equal to the number of pipeline stages. Bit N of the port "
    "is the load-enable signal for the pipeline registers after stage N.");
ABSL_FLAG(bool, flop_inputs, true,
          "If true, inputs of the the module are flopped into registers before "
          "use in generated pipelines. Only used with pipeline generator.");
ABSL_FLAG(bool, flop_outputs, true,
          "If true, the module outputs are flopped into registers before "
          "leaving module. Only used with pipeline generator.");
ABSL_FLAG(std::string, flop_inputs_kind, "flop",
          "Kind of inputs register to add.  "
          "Valid values: flop, skid, zerolatency");
ABSL_FLAG(std::string, flop_outputs_kind, "flop",
          "Kind of output register to add.  "
          "Valid values: flop, skid, zerolatency");
ABSL_FLAG(bool, flop_single_value_channels, true,
          "If false, flop_inputs() and flop_outputs() will not flop"
          "single value channels");
ABSL_FLAG(bool, add_idle_output, false,
          "If true, an additional idle signal tied to valids of input and "
          "flops is added to the block. This output signal is not registered, "
          "regardless of the setting of flop_outputs. "
          "use in generated pipelines. Only used with pipeline generator.");
ABSL_FLAG(std::string, module_name, "",
          "Explicit name to use for the generated module; if not provided the "
          "mangled IR function name is used");
// TODO(meheff): Rather than specify all reset (or codegen options in general)
// as a multitude of flags, these can be specified via a separate file (like a
// options proto).
ABSL_FLAG(std::string, reset, "",
          "Name of the reset signal. If empty, no reset signal is used.");
ABSL_FLAG(bool, reset_active_low, false,
          "Whether the reset signal is active low.");
ABSL_FLAG(bool, reset_asynchronous, false,
          "Whether the reset signal is asynchronous.");
ABSL_FLAG(bool, reset_data_path, false, "Whether to also reset the datapath.");
ABSL_FLAG(bool, use_system_verilog, true,
          "If true, emit SystemVerilog otherwise emit Verilog.");
ABSL_FLAG(bool, separate_lines, false,
          "If true, emit every subexpression on a separate line.");
ABSL_FLAG(std::string, gate_format, "", "Format string to use for gate! ops.");
ABSL_FLAG(std::string, assert_format, "",
          "Format string to use for assertions.");
ABSL_FLAG(std::string, smulp_format, "", "Format string to use for smulp.");
ABSL_FLAG(std::string, streaming_channel_data_suffix, "",
          "Suffix to append to data signals for streaming channels.");
ABSL_FLAG(std::string, streaming_channel_valid_suffix, "_vld",
          "Suffix to append to valid signals for streaming channels.");
ABSL_FLAG(std::string, streaming_channel_ready_suffix, "_rdy",
          "Suffix to append to ready signals for streaming channels.");
ABSL_FLAG(std::string, umulp_format, "", "Format string to use for smulp.");
// LINT.ThenChange(//xls/build_rules/xls_codegen_rules.bzl)

namespace xls {
namespace {

absl::StatusOr<FunctionBase*> FindEntry(Package* p) {
  std::string top_str = absl::GetFlag(FLAGS_top);

  if (!top_str.empty()) {
    XLS_RETURN_IF_ERROR(p->SetTopByName(top_str));
  }

  // Default to the top entity if nothing is specified.
  std::optional<FunctionBase*> top = p->GetTop();
  if (!top.has_value()) {
    return absl::InternalError(
        absl::StrFormat("Top entity not set for package: %s.", p->name()));
  }
  return top.value();
}

absl::StatusOr<verilog::CodegenOptions::IOKind> StrToIOKind(
    absl::string_view str) {
  if (str == "flop") {
    return verilog::CodegenOptions::IOKind::kFlop;
  }

  if (str == "skid") {
    return verilog::CodegenOptions::IOKind::kSkidBuffer;
  }

  if (str == "zerolatency") {
    return verilog::CodegenOptions::IOKind::kZeroLatencyBuffer;
  }

  return absl::InvalidArgumentError(
      absl::StrFormat("I/O flop kind does not support %s", str));
}

absl::StatusOr<verilog::CodegenOptions> GetCodegenOptions() {
  verilog::CodegenOptions options;

  if (absl::GetFlag(FLAGS_generator) == "pipeline") {
    options = verilog::BuildPipelineOptions();

    if (!absl::GetFlag(FLAGS_input_valid_signal).empty()) {
      std::optional<std::string> output_signal;
      if (!absl::GetFlag(FLAGS_output_valid_signal).empty()) {
        output_signal = absl::GetFlag(FLAGS_output_valid_signal);
      }
      options.valid_control(absl::GetFlag(FLAGS_input_valid_signal),
                            output_signal);
    } else if (!absl::GetFlag(FLAGS_manual_load_enable_signal).empty()) {
      options.manual_control(absl::GetFlag(FLAGS_manual_load_enable_signal));
    }
    options.flop_inputs(absl::GetFlag(FLAGS_flop_inputs));
    options.flop_outputs(absl::GetFlag(FLAGS_flop_outputs));

    std::string flop_inputs_kind_str = absl::GetFlag(FLAGS_flop_inputs_kind);
    XLS_ASSIGN_OR_RETURN(verilog::CodegenOptions::IOKind inputs_kind,
                         StrToIOKind(flop_inputs_kind_str));
    options.flop_inputs_kind(inputs_kind);

    std::string flop_outputs_kind_str = absl::GetFlag(FLAGS_flop_outputs_kind);
    XLS_ASSIGN_OR_RETURN(verilog::CodegenOptions::IOKind outputs_kind,
                         StrToIOKind(flop_outputs_kind_str));
    options.flop_outputs_kind(outputs_kind);

    options.flop_single_value_channels(
        absl::GetFlag(FLAGS_flop_single_value_channels));
    options.add_idle_output(absl::GetFlag(FLAGS_add_idle_output));

    if (!absl::GetFlag(FLAGS_reset).empty()) {
      options.reset(absl::GetFlag(FLAGS_reset),
                    absl::GetFlag(FLAGS_reset_asynchronous),
                    absl::GetFlag(FLAGS_reset_active_low),
                    absl::GetFlag(FLAGS_reset_data_path));
    }
  }

  if (!absl::GetFlag(FLAGS_module_name).empty()) {
    options.module_name(absl::GetFlag(FLAGS_module_name));
  }

  options.use_system_verilog(absl::GetFlag(FLAGS_use_system_verilog));

  options.separate_lines(absl::GetFlag(FLAGS_separate_lines));

  if (!absl::GetFlag(FLAGS_gate_format).empty()) {
    options.SetOpOverride(Op::kGate,
                          std::make_unique<verilog::OpOverrideGateAssignment>(
                              absl::GetFlag(FLAGS_gate_format)));
  }

  if (!absl::GetFlag(FLAGS_assert_format).empty()) {
    options.SetOpOverride(Op::kAssert,
                          std::make_unique<verilog::OpOverrideAssertion>(
                              absl::GetFlag(FLAGS_assert_format)));
  }

  if (!absl::GetFlag(FLAGS_smulp_format).empty()) {
    options.SetOpOverride(Op::kSMulp,
                          std::make_unique<verilog::OpOverrideInstantiation>(
                              absl::GetFlag(FLAGS_smulp_format)));
  }

  if (!absl::GetFlag(FLAGS_umulp_format).empty()) {
    options.SetOpOverride(Op::kUMulp,
                          std::make_unique<verilog::OpOverrideInstantiation>(
                              absl::GetFlag(FLAGS_umulp_format)));
  }

  options.streaming_channel_data_suffix(
      absl::GetFlag(FLAGS_streaming_channel_data_suffix));
  options.streaming_channel_valid_suffix(
      absl::GetFlag(FLAGS_streaming_channel_valid_suffix));
  options.streaming_channel_ready_suffix(
      absl::GetFlag(FLAGS_streaming_channel_ready_suffix));

  return options;
}

absl::StatusOr<PipelineSchedule> RunSchedulingPipeline(
    FunctionBase* main, const SchedulingOptions& scheduling_options,
    const DelayEstimator* delay_estimator) {
  absl::StatusOr<PipelineSchedule> schedule_status =
      PipelineSchedule::Run(main, *delay_estimator, scheduling_options);

  if (!schedule_status.ok()) {
    if (absl::IsResourceExhausted(schedule_status.status())) {
      // Resource exhausted error indicates that the schedule was
      // infeasible. Emit a meaningful error in this case.
      if (scheduling_options.pipeline_stages().has_value() &&
          scheduling_options.clock_period_ps().has_value()) {
        // TODO(meheff): Add link to documentation with more information and
        // guidance.
        XLS_LOG(QFATAL) << absl::StreamFormat(
            "Design cannot be scheduled in %d stages with a %dps clock.",
            scheduling_options.pipeline_stages().value(),
            scheduling_options.clock_period_ps().value());
      }
    }
  }

  return schedule_status;
}

absl::Status RealMain(absl::string_view ir_path, absl::string_view verilog_path,
                      absl::string_view signature_path,
                      absl::string_view schedule_path,
                      absl::string_view verilog_line_map_path,
                      absl::string_view output_block_ir_path) {
  if (ir_path == "-") {
    ir_path = "/dev/stdin";
  }

  XLS_ASSIGN_OR_RETURN(std::string ir_contents, GetFileContents(ir_path));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> p,
                       Parser::ParsePackage(ir_contents, ir_path));
  verilog::ModuleGeneratorResult result;

  XLS_RETURN_IF_ERROR(VerifyPackage(p.get(), /*codegen=*/true));

  XLS_ASSIGN_OR_RETURN(FunctionBase * main, FindEntry(p.get()));
  XLS_ASSIGN_OR_RETURN(verilog::CodegenOptions codegen_options,
                       GetCodegenOptions());

  if (absl::GetFlag(FLAGS_generator) == "pipeline") {
    XLS_QCHECK(absl::GetFlag(FLAGS_pipeline_stages) != 0 ||
               absl::GetFlag(FLAGS_clock_period_ps) != 0)
        << "Must specify --pipeline_stages or --clock_period_ps (or both).";

    XLS_ASSIGN_OR_RETURN(SchedulingOptions scheduling_options,
                         SetupSchedulingOptions(p.get()));
    XLS_ASSIGN_OR_RETURN(const DelayEstimator* delay_estimator,
                         SetupDelayEstimator());
    XLS_ASSIGN_OR_RETURN(
        PipelineSchedule schedule,
        RunSchedulingPipeline(main, scheduling_options, delay_estimator));

    XLS_ASSIGN_OR_RETURN(
        result, verilog::ToPipelineModuleText(schedule, main, codegen_options));

    if (!schedule_path.empty()) {
      XLS_RETURN_IF_ERROR(SetTextProtoFile(schedule_path, schedule.ToProto()));
    }
  } else if (absl::GetFlag(FLAGS_generator) == "combinational") {
    XLS_ASSIGN_OR_RETURN(
        result, verilog::GenerateCombinationalModule(main, codegen_options));
  } else {
    XLS_LOG(QFATAL) << absl::StreamFormat(
        "Invalid value for --generator: %s. Expected 'pipeline' or "
        "'combinational'",
        absl::GetFlag(FLAGS_generator));
  }

  if (!output_block_ir_path.empty()) {
    XLS_QCHECK_EQ(p->blocks().size(), 1)
        << "There should be exactly one block in the package after generating "
           "module text.";
    XLS_RETURN_IF_ERROR(SetFileContents(output_block_ir_path, p->DumpIr()));
  }

  if (!signature_path.empty()) {
    XLS_RETURN_IF_ERROR(
        SetTextProtoFile(signature_path, result.signature.proto()));
  }

  if (!verilog_path.empty()) {
    std::filesystem::path absolute = std::filesystem::absolute(verilog_path);
    for (int64_t i = 0; i < result.verilog_line_map.mapping_size(); ++i) {
      result.verilog_line_map.mutable_mapping(i)->set_verilog_file(absolute);
    }
  }

  if (!verilog_line_map_path.empty()) {
    XLS_RETURN_IF_ERROR(
        SetTextProtoFile(verilog_line_map_path, result.verilog_line_map));
  }

  if (verilog_path.empty()) {
    std::cout << result.verilog_text;
  } else {
    XLS_RETURN_IF_ERROR(SetFileContents(verilog_path, result.verilog_text));
  }
  return absl::OkStatus();
}

}  // namespace
}  // namespace xls

int main(int argc, char** argv) {
  std::vector<absl::string_view> positional_arguments =
      xls::InitXls(kUsage, argc, argv);

  if (positional_arguments.size() != 1) {
    XLS_LOG(QFATAL) << absl::StreamFormat("Expected invocation: %s IR_FILE",
                                          argv[0]);
  }
  absl::string_view ir_path = positional_arguments[0];
  XLS_QCHECK_OK(xls::RealMain(ir_path, absl::GetFlag(FLAGS_output_verilog_path),
                              absl::GetFlag(FLAGS_output_signature_path),
                              absl::GetFlag(FLAGS_output_schedule_path),
                              absl::GetFlag(FLAGS_output_verilog_line_map_path),
                              absl::GetFlag(FLAGS_output_block_ir_path)));

  return EXIT_SUCCESS;
}
