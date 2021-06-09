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
#include "xls/codegen/combinational_generator.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/codegen/pipeline_generator.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/delay_model/delay_estimators.h"
#include "xls/ir/ir_parser.h"
#include "xls/passes/standard_pipeline.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/scheduling/scheduling_pass.h"

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

ABSL_FLAG(int64_t, clock_period_ps, 0, "Target clock period, in picoseconds.");
ABSL_FLAG(int64_t, pipeline_stages, 0,
          "The number of stages in the generated pipeline.");
ABSL_FLAG(std::string, delay_model, "",
          "Delay model name to use from registry.");
ABSL_FLAG(
    std::string, output_verilog_path, "",
    "Specific output path for the Verilog generated. If not specified then "
    "Verilog is written to stdout.");
ABSL_FLAG(std::string, output_schedule_path, "",
          "Specific output path for the generated pipeline schedule. "
          "If not specified, then no schedule is output.");
ABSL_FLAG(
    std::string, output_signature_path, "",
    "Specific output path for the module signature. If not specified then "
    "no module signature is generated.");
ABSL_FLAG(std::string, entry, "", "Entry function for the package.");
ABSL_FLAG(std::string, generator, "pipeline",
          "The generator to use when emitting the device function. Valid "
          "values: pipeline, combinational.");
ABSL_FLAG(
    std::string, input_valid_signal, "",
    "If specified, the emitted module will use an external \"valid\" signal "
    "as the load enable for pipeline registers. The flag value is the "
    "name of the input port for this signal.");
ABSL_FLAG(std::string, output_valid_signal, "",
          "The name of the output port which holds the pipelined valid signal. "
          "Must be specified with --input_valid_signal.");
ABSL_FLAG(
    std::string, manual_load_enable_signal, "",
    "If specified the load-enable of the pipeline registers of each stage is "
    "controlled via an input port of the indicated name. The width of the "
    "input port is equal to the number of pipeline stages. Bit N of the port "
    "is the load-enable signal for the pipeline registers after stage N.");
ABSL_FLAG(bool, flop_inputs, true,
          "If true, inputs of the the module are flopped into registers before "
          "use in generated pipelines. Only used with pipline generator.");
ABSL_FLAG(bool, flop_outputs, true,
          "If true, the module outputs are flopped into registers before "
          "leaving module. Only used with pipline generator.");
ABSL_FLAG(std::string, module_name, "",
          "Explicit name to use for the generated module; if not provided the "
          "mangled IR function name is used");
ABSL_FLAG(int64_t, clock_margin_percent, 0,
          "The percentage of clock period to set aside as a margin to ensure "
          "timing is met. Effectively, this lowers the clock period by this "
          "percentage amount for the purposes of scheduling.");
// TODO(meheff): Rather than specify all reset (or codegen options in general)
// as a multitude of flags, these can be specified via a separate file (like a
// options proto).
ABSL_FLAG(std::string, reset, "",
          "Name of the reset signal. If empty, no reset signal is used.");
ABSL_FLAG(bool, reset_active_low, false,
          "Whether the reset signal is active low.");
ABSL_FLAG(bool, reset_asynchronous, false,
          "Whether the reset signal is asynchronous.");
ABSL_FLAG(bool, use_system_verilog, true,
          "If true, emit SystemVerilog otherwise emit Verilog.");

namespace xls {
namespace {

absl::Status RealMain(absl::string_view ir_path, absl::string_view verilog_path,
                      absl::string_view signature_path,
                      absl::string_view schedule_path) {
  XLS_ASSIGN_OR_RETURN(std::string ir_contents, GetFileContents(ir_path));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> p,
                       Parser::ParsePackage(ir_contents, ir_path));

  Function* main;
  if (absl::GetFlag(FLAGS_entry).empty()) {
    XLS_ASSIGN_OR_RETURN(main, p->EntryFunction());
  } else {
    XLS_ASSIGN_OR_RETURN(main, p->GetFunction(absl::GetFlag(FLAGS_entry)));
  }

  verilog::ModuleGeneratorResult result;
  if (absl::GetFlag(FLAGS_generator) == "pipeline") {
    XLS_QCHECK(absl::GetFlag(FLAGS_pipeline_stages) != 0 ||
               absl::GetFlag(FLAGS_clock_period_ps) != 0)
        << "Musts specify --pipeline_stages or --clock_period_ps (or both).";

    SchedulingPassOptions sched_options;
    if (!absl::GetFlag(FLAGS_entry).empty()) {
      sched_options.scheduling_options.entry(absl::GetFlag(FLAGS_entry));
    }
    if (absl::GetFlag(FLAGS_pipeline_stages) != 0) {
      sched_options.scheduling_options.pipeline_stages(
          absl::GetFlag(FLAGS_pipeline_stages));
    }
    if (absl::GetFlag(FLAGS_clock_period_ps) != 0) {
      sched_options.scheduling_options.clock_period_ps(
          absl::GetFlag(FLAGS_clock_period_ps));
    }
    if (absl::GetFlag(FLAGS_clock_margin_percent) != 0) {
      sched_options.scheduling_options.clock_margin_percent(
          absl::GetFlag(FLAGS_clock_margin_percent));
    }
    XLS_ASSIGN_OR_RETURN(sched_options.delay_estimator,
                         GetDelayEstimator(absl::GetFlag(FLAGS_delay_model)));
    std::unique_ptr<SchedulingCompoundPass> scheduling_pipeline =
        CreateStandardSchedulingPassPipeline();
    SchedulingPassResults results;
    SchedulingUnit scheduling_unit = {p.get(),
                                      /*schedule=*/absl::nullopt};
    absl::Status scheduling_status =
        scheduling_pipeline->Run(&scheduling_unit, sched_options, &results)
            .status();
    if (!scheduling_status.ok()) {
      if (absl::IsResourceExhausted(scheduling_status)) {
        // Resource exhausted error indicates that the schedule was
        // infeasible. Emit a meaningful error in this case.
        if (sched_options.scheduling_options.pipeline_stages().has_value() &&
            sched_options.scheduling_options.clock_period_ps().has_value()) {
          // TODO(meheff): Add link to documentation with more information and
          // guidance.
          XLS_LOG(QFATAL) << absl::StreamFormat(
              "Design cannot be scheduled in %d stages with a %dps clock.",
              sched_options.scheduling_options.pipeline_stages().value(),
              sched_options.scheduling_options.clock_period_ps().value());
        }
      }
      return scheduling_status;
    }
    XLS_RET_CHECK(scheduling_unit.schedule.has_value());

    verilog::PipelineOptions pipeline_options;
    if (!absl::GetFlag(FLAGS_module_name).empty()) {
      pipeline_options.module_name(absl::GetFlag(FLAGS_module_name));
    }
    pipeline_options.use_system_verilog(
        absl::GetFlag(FLAGS_use_system_verilog));
    if (!absl::GetFlag(FLAGS_input_valid_signal).empty()) {
      pipeline_options.valid_control(absl::GetFlag(FLAGS_input_valid_signal),
                                     absl::GetFlag(FLAGS_output_valid_signal));
    } else if (!absl::GetFlag(FLAGS_manual_load_enable_signal).empty()) {
      pipeline_options.manual_control(
          absl::GetFlag(FLAGS_manual_load_enable_signal));
    }
    pipeline_options.flop_inputs(absl::GetFlag(FLAGS_flop_inputs));
    pipeline_options.flop_outputs(absl::GetFlag(FLAGS_flop_outputs));

    if (!absl::GetFlag(FLAGS_reset).empty()) {
      verilog::ResetProto reset_proto;
      reset_proto.set_name(absl::GetFlag(FLAGS_reset));
      reset_proto.set_asynchronous(absl::GetFlag(FLAGS_reset_asynchronous));
      reset_proto.set_active_low(absl::GetFlag(FLAGS_reset_active_low));
      pipeline_options.reset(reset_proto);
    }

    XLS_ASSIGN_OR_RETURN(
        result, verilog::ToPipelineModuleText(*scheduling_unit.schedule, main,
                                              pipeline_options));
    if (!schedule_path.empty()) {
      XLS_RETURN_IF_ERROR(
          SetTextProtoFile(schedule_path, scheduling_unit.schedule->ToProto()));
    }
  } else if (absl::GetFlag(FLAGS_generator) == "combinational") {
    XLS_ASSIGN_OR_RETURN(result,
                         verilog::GenerateCombinationalModule(
                             main, absl::GetFlag(FLAGS_use_system_verilog)));
  } else {
    XLS_LOG(QFATAL) << absl::StreamFormat(
        "Invalid value for --generator: %s. Expected 'pipeline' or "
        "'combinational'",
        absl::GetFlag(FLAGS_generator));
  }
  if (!signature_path.empty()) {
    XLS_RETURN_IF_ERROR(
        SetTextProtoFile(signature_path, result.signature.proto()));
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
                              absl::GetFlag(FLAGS_output_schedule_path)));

  return EXIT_SUCCESS;
}
