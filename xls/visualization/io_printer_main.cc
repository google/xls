// Copyright 2026 The XLS Authors
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

#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "cppitertools/enumerate.hpp"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/codegen_v_1_5/scheduling_pass.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function_base.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/passes/pass_base.h"

static constexpr std::string_view kUsage = R"(
Dump io-related scheduling result to stdout.

Explicitly show the pipeline stage.

This only includes IO nodes (send, receive, etc.).

Top must be a proc.

If --schedule_proto is not specified, the IR file must contain a
'scheduled_{proc,func}' for each proc/function.

Example invocation:
  io_printer_main \
       [--schedule_proto=...] \
       IR_FILE
)";

ABSL_FLAG(std::optional<std::string>, schedule_proto, std::nullopt,
          "Path to the schedule proto file to use.");

namespace xls {
namespace {

absl::Status ConvertSchedulePbToScheduledIr(
    const PackageScheduleProto& schedule_pb, Package* package) {
  std::vector<FunctionBase*> function_bases = package->GetFunctionBases();
  codegen::SchedulingPass scheduling_pass;
  PassResults pass_results;
  codegen::BlockConversionPassOptions pass_options;
  pass_options.package_schedule = schedule_pb;
  XLS_RETURN_IF_ERROR(
      scheduling_pass.Run(package, pass_options, &pass_results).status());
  return absl::OkStatus();
}

absl::Status RealMain(std::string_view ir_path,
                      std::optional<std::string> schedule_proto_path) {
  XLS_ASSIGN_OR_RETURN(std::string package_text, GetFileContents(ir_path));
  XLS_ASSIGN_OR_RETURN(auto package,
                       Parser::ParsePackage(package_text, ir_path));
  if (schedule_proto_path.has_value()) {
    XLS_ASSIGN_OR_RETURN(std::string schedule_proto_text,
                         GetFileContents(*schedule_proto_path));
    PackageScheduleProto schedule_pb;
    XLS_RETURN_IF_ERROR(ParseTextProtoFile(*schedule_proto_path, &schedule_pb));
    XLS_RETURN_IF_ERROR(
        ConvertSchedulePbToScheduledIr(schedule_pb, package.get()));
  }
  for (FunctionBase* fb : package->GetFunctionBases()) {
    if (!fb->IsProc() || !fb->IsScheduled()) {
      continue;
    }
    std::cout << "Function: " << fb->name() << "\n";
    for (auto [idx, stage] : iter::enumerate(fb->stages())) {
      std::cout << "  Stage " << idx << ":\n";
      for (Node* node : stage) {
        if (node->Is<Send>() || node->Is<Receive>()) {
          std::string blocking;
          if (node->Is<Receive>()) {
            blocking = node->As<Receive>()->is_blocking() ? " " : "!";
          }
          std::cout << "    " << node->As<ChannelNode>()->direction()
                    << blocking << " on "
                    << node->As<ChannelNode>()->channel_name() << "\n";
          //  << " op " << node->ToString() << "\n";
        }
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace
}  // namespace xls

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_arguments =
      xls::InitXls(kUsage, argc, argv);

  if (positional_arguments.size() != 1) {
    LOG(QFATAL) << "Expected invocation: " << argv[0] << " <ir_path>";
  }

  return xls::ExitStatus(xls::RealMain(positional_arguments[0],
                                       absl::GetFlag(FLAGS_schedule_proto)));
}
