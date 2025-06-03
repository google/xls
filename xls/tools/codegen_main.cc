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
#include <cstdint>
#include <filesystem>  // NOLINT
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xls/codegen/codegen_result.h"
#include "xls/codegen/module_signature.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dev_tools/tool_timeout.h"
#include "xls/ir/function_base.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/verifier.h"
#include "xls/scheduling/scheduling_options.h"
#include "xls/scheduling/scheduling_result.h"
#include "xls/tools/codegen.h"
#include "xls/tools/codegen_flags.h"
#include "xls/tools/codegen_flags.pb.h"
#include "xls/tools/scheduling_options_flags.h"
#include "xls/tools/scheduling_options_flags.pb.h"

static constexpr std::string_view kUsage = R"(
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

absl::Status RealMain(std::string_view ir_path) {
  auto timeout = StartTimeoutTimer();
  if (ir_path == "-") {
    ir_path = "/dev/stdin";
  }
  XLS_ASSIGN_OR_RETURN(std::string ir_contents, GetFileContents(ir_path));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> p,
                       Parser::ParsePackage(ir_contents, ir_path));

  XLS_ASSIGN_OR_RETURN(CodegenFlagsProto codegen_flags_proto,
                       GetCodegenFlags());
  if (!codegen_flags_proto.top().empty()) {
    XLS_RETURN_IF_ERROR(p->SetTopByName(codegen_flags_proto.top()));
  }

  XLS_RET_CHECK(p->GetTop().has_value())
      << "Package " << p->name() << " needs a top function/proc.";
  auto main = [&p]() -> FunctionBase* { return p->GetTop().value(); };

  XLS_ASSIGN_OR_RETURN(
      SchedulingOptionsFlagsProto scheduling_options_flags_proto,
      GetSchedulingOptionsFlagsProto());
  XLS_ASSIGN_OR_RETURN(
      bool delay_model_flag_passed,
      IsDelayModelSpecifiedViaFlag(scheduling_options_flags_proto));
  std::pair<SchedulingResult, verilog::CodegenResult> result;
  XLS_ASSIGN_OR_RETURN(
      result, ScheduleAndCodegen(p.get(), scheduling_options_flags_proto,
                                 codegen_flags_proto, delay_model_flag_passed));
  const SchedulingResult& scheduling_result = result.first;
  verilog::CodegenResult& codegen_result = result.second;

  if (!absl::GetFlag(FLAGS_output_schedule_ir_path).empty()) {
    XLS_RETURN_IF_ERROR(
        SetFileContents(absl::GetFlag(FLAGS_output_schedule_ir_path),
                        main()->package()->DumpIr()));
  }

  if (!absl::GetFlag(FLAGS_output_schedule_path).empty()) {
    XLS_RETURN_IF_ERROR(
        SetTextProtoFile(absl::GetFlag(FLAGS_output_schedule_path),
                         scheduling_result.schedules));
  }

  if (!absl::GetFlag(FLAGS_output_block_ir_path).empty()) {
    QCHECK_GE(p->blocks().size(), 1)
        << "There should be at least one block in the package after generating "
           "module text.";
    XLS_RETURN_IF_ERROR(SetFileContents(
        absl::GetFlag(FLAGS_output_block_ir_path), p->DumpIr()));
  }

  if (!absl::GetFlag(FLAGS_output_signature_path).empty()) {
    XLS_RETURN_IF_ERROR(
        SetTextProtoFile(absl::GetFlag(FLAGS_output_signature_path),
                         codegen_result.signature.proto()));
  }

  if (!absl::GetFlag(FLAGS_output_scheduling_pass_metrics_path).empty()) {
    XLS_RETURN_IF_ERROR(SetTextProtoFile(
        absl::GetFlag(FLAGS_output_scheduling_pass_metrics_path),
        scheduling_result.pass_pipeline_metrics));
  }

  if (!absl::GetFlag(FLAGS_output_codegen_pass_metrics_path).empty()) {
    XLS_RETURN_IF_ERROR(
        SetTextProtoFile(absl::GetFlag(FLAGS_output_codegen_pass_metrics_path),
                         codegen_result.pass_pipeline_metrics));
  }

  if (!absl::GetFlag(FLAGS_block_metrics_path).empty()) {
    XLS_RETURN_IF_ERROR(SetTextProtoFile(
        absl::GetFlag(FLAGS_block_metrics_path), codegen_result.block_metrics));
  }

  const std::string& verilog_path = absl::GetFlag(FLAGS_output_verilog_path);
  if (!verilog_path.empty()) {
    for (int64_t i = 0; i < codegen_result.verilog_line_map.mapping_size();
         ++i) {
      codegen_result.verilog_line_map.mutable_mapping(i)->set_verilog_file(
          verilog_path);
    }
  }

  const std::string& verilog_line_map_path =
      absl::GetFlag(FLAGS_output_verilog_line_map_path);
  if (!verilog_line_map_path.empty()) {
    XLS_RETURN_IF_ERROR(SetTextProtoFile(verilog_line_map_path,
                                         codegen_result.verilog_line_map));
  }

  if (verilog_path.empty()) {
    std::cout << codegen_result.verilog_text;
  } else {
    XLS_RETURN_IF_ERROR(
        SetFileContents(verilog_path, codegen_result.verilog_text));
  }
  return absl::OkStatus();
}

}  // namespace
}  // namespace xls

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_arguments =
      xls::InitXls(kUsage, argc, argv);

  if (positional_arguments.size() != 1) {
    LOG(QFATAL) << absl::StreamFormat("Expected invocation: %s IR_FILE",
                                      argv[0]);
  }
  std::string_view ir_path = positional_arguments[0];
  return xls::ExitStatus(xls::RealMain(ir_path));
}
