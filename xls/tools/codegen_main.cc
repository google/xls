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
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/combinational_generator.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/codegen/op_override_impls.h"
#include "xls/codegen/pipeline_generator.h"
#include "xls/codegen/ram_configuration.h"
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
#include "xls/tools/codegen_flags.h"
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

namespace xls {
namespace {

verilog::CodegenOptions::IOKind ToIOKind(IOKindProto p) {
  switch (p) {
    case IO_KIND_INVALID:
    case IO_KIND_FLOP:
      return verilog::CodegenOptions::IOKind::kFlop;
    case IO_KIND_SKID_BUFFER:
      return verilog::CodegenOptions::IOKind::kSkidBuffer;
    case IO_KIND_ZERO_LATENCY_BUFFER:
      return verilog::CodegenOptions::IOKind::kZeroLatencyBuffer;
    default:
      XLS_LOG(FATAL) << "Invalid IOKindProto value: " << static_cast<int>(p);
  }
}

absl::StatusOr<verilog::CodegenOptions> CodegenOptionsFromProto(
    const CodegenFlagsProto& p) {
  verilog::CodegenOptions options;

  if (p.generator() == GENERATOR_KIND_PIPELINE) {
    options = verilog::BuildPipelineOptions();

    if (!p.input_valid_signal().empty()) {
      std::optional<std::string> output_signal;
      if (!p.output_valid_signal().empty()) {
        output_signal = p.output_valid_signal();
      }
      options.valid_control(p.input_valid_signal(), output_signal);
    } else if (!p.manual_load_enable_signal().empty()) {
      options.manual_control(p.manual_load_enable_signal());
    }
    options.flop_inputs(p.flop_inputs());
    options.flop_outputs(p.flop_outputs());
    options.flop_inputs_kind(ToIOKind(p.flop_inputs_kind()));
    options.flop_outputs_kind(ToIOKind(p.flop_outputs_kind()));

    options.flop_single_value_channels(p.flop_single_value_channels());
    options.add_idle_output(p.add_idle_output());

    if (!p.reset().empty()) {
      options.reset(p.reset(), p.reset_asynchronous(), p.reset_active_low(),
                    p.reset_data_path());
    }
  }

  if (!p.module_name().empty()) {
    options.module_name(p.module_name());
  }

  options.use_system_verilog(p.use_system_verilog());
  options.separate_lines(p.separate_lines());

  if (!p.gate_format().empty()) {
    options.SetOpOverride(
        Op::kGate,
        std::make_unique<verilog::OpOverrideGateAssignment>(p.gate_format()));
  }

  if (!p.assert_format().empty()) {
    options.SetOpOverride(
        Op::kAssert,
        std::make_unique<verilog::OpOverrideAssertion>(p.assert_format()));
  }

  if (!p.smulp_format().empty()) {
    options.SetOpOverride(
        Op::kSMulp,
        std::make_unique<verilog::OpOverrideInstantiation>(p.smulp_format()));
  }

  if (!p.umulp_format().empty()) {
    options.SetOpOverride(
        Op::kUMulp,
        std::make_unique<verilog::OpOverrideInstantiation>(p.umulp_format()));
  }

  options.streaming_channel_data_suffix(p.streaming_channel_data_suffix());
  options.streaming_channel_valid_suffix(p.streaming_channel_valid_suffix());
  options.streaming_channel_ready_suffix(p.streaming_channel_ready_suffix());

  std::vector<std::unique_ptr<verilog::RamConfiguration>> ram_configurations;
  ram_configurations.reserve(p.ram_configurations_size());
  for (const std::string& config_text : p.ram_configurations()) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<verilog::RamConfiguration> config,
                         verilog::RamConfiguration::ParseString(config_text));
    ram_configurations.push_back(std::move(config));
  }
  options.ram_configurations(ram_configurations);

  options.gate_recvs(p.gate_recvs());
  options.array_index_bounds_checking(p.array_index_bounds_checking());

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

absl::Status RealMain(std::string_view ir_path) {
  XLS_ASSIGN_OR_RETURN(CodegenFlagsProto codegen_flags_proto,
                       CodegenFlagsFromAbslFlags());

  if (ir_path == "-") {
    ir_path = "/dev/stdin";
  }

  XLS_ASSIGN_OR_RETURN(std::string ir_contents, GetFileContents(ir_path));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> p,
                       Parser::ParsePackage(ir_contents, ir_path));
  verilog::ModuleGeneratorResult result;

  XLS_RETURN_IF_ERROR(VerifyPackage(p.get(), /*codegen=*/true));

  XLS_ASSIGN_OR_RETURN(verilog::CodegenOptions codegen_options,
                       CodegenOptionsFromProto(codegen_flags_proto));

  std::optional<std::string_view> maybe_top_str;
  if (!codegen_flags_proto.top().empty()) {
    maybe_top_str = codegen_flags_proto.top();
  }
  XLS_ASSIGN_OR_RETURN(FunctionBase * main, FindTop(p.get(), maybe_top_str));

  if (codegen_flags_proto.generator() == GENERATOR_KIND_PIPELINE) {
    XLS_QCHECK(absl::GetFlag(FLAGS_pipeline_stages) != 0 ||
               absl::GetFlag(FLAGS_clock_period_ps) != 0)
        << "Must specify --pipeline_stages or --clock_period_ps (or both).";

    XLS_ASSIGN_OR_RETURN(SchedulingOptions scheduling_options,
                         SetUpSchedulingOptions(p.get()));
    XLS_ASSIGN_OR_RETURN(const DelayEstimator* delay_estimator,
                         SetUpDelayEstimator());
    XLS_ASSIGN_OR_RETURN(
        PipelineSchedule schedule,
        RunSchedulingPipeline(main, scheduling_options, delay_estimator));

    XLS_ASSIGN_OR_RETURN(
        result, verilog::ToPipelineModuleText(schedule, main, codegen_options));

    if (!codegen_flags_proto.output_schedule_path().empty()) {
      XLS_RETURN_IF_ERROR(SetTextProtoFile(
          codegen_flags_proto.output_schedule_path(), schedule.ToProto()));
    }
  } else if (codegen_flags_proto.generator() == GENERATOR_KIND_COMBINATIONAL) {
    XLS_ASSIGN_OR_RETURN(
        result, verilog::GenerateCombinationalModule(main, codegen_options));
  } else {
    // Note: this should already be validated by CodegenFlagsFromAbslFlags().
    XLS_LOG(FATAL) << "Invalid generator kind: "
                   << static_cast<int>(codegen_flags_proto.generator());
  }

  if (!codegen_flags_proto.output_block_ir_path().empty()) {
    XLS_QCHECK_EQ(p->blocks().size(), 1)
        << "There should be exactly one block in the package after generating "
           "module text.";
    XLS_RETURN_IF_ERROR(SetFileContents(
        codegen_flags_proto.output_block_ir_path(), p->DumpIr()));
  }

  if (!codegen_flags_proto.output_signature_path().empty()) {
    XLS_RETURN_IF_ERROR(SetTextProtoFile(
        codegen_flags_proto.output_signature_path(), result.signature.proto()));
  }

  const std::string& verilog_path = codegen_flags_proto.output_verilog_path();
  if (!verilog_path.empty()) {
    std::filesystem::path absolute = std::filesystem::absolute(verilog_path);
    for (int64_t i = 0; i < result.verilog_line_map.mapping_size(); ++i) {
      result.verilog_line_map.mutable_mapping(i)->set_verilog_file(absolute);
    }
  }

  const std::string& verilog_line_map_path =
      codegen_flags_proto.output_verilog_line_map_path();
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
  std::vector<std::string_view> positional_arguments =
      xls::InitXls(kUsage, argc, argv);

  if (positional_arguments.size() != 1) {
    XLS_LOG(QFATAL) << absl::StreamFormat("Expected invocation: %s IR_FILE",
                                          argv[0]);
  }
  std::string_view ir_path = positional_arguments[0];
  XLS_QCHECK_OK(xls::RealMain(ir_path));

  return EXIT_SUCCESS;
}
