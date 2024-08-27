// Copyright 2024 The XLS Authors
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
#include "absl/strings/str_format.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/dev_tools/remove_identifiers.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"

static constexpr std::string_view kUsage = R"(
Removes all identifiers and renumbers an IR file.

This can make some IR files which have undergone minimzation or other transforms
much more readable. It also enables one to share these minimized programs
without leaking information about the original context the code was used in
which may be confidential.
)";

ABSL_FLAG(std::optional<std::string>, force_package_name, "subrosa",
          "What to rename the package to.");
ABSL_FLAG(bool, strip_location_info, true, "remove all location information");
ABSL_FLAG(bool, strip_node_names, true,
          "Replace node names with identifiers based on their op and id");
ABSL_FLAG(bool, strip_chan_names, true,
          "Replace chan names with opaque identifiers.");
ABSL_FLAG(bool, strip_reg_names, true,
          "Replace register names with opaque identifiers.");
ABSL_FLAG(
    bool, strip_function_names, true,
    "Replace function/proc/block names with identifiers based on their type");

namespace xls {
namespace {

absl::Status RealMain(std::string_view ir_file) {
  XLS_ASSIGN_OR_RETURN(std::string ir, GetFileContents(ir_file));
  XLS_ASSIGN_OR_RETURN(auto package, Parser::ParsePackage(ir, ir_file));
  XLS_ASSIGN_OR_RETURN(
      auto result,
      StripPackage(
          package.get(),
          StripOptions{
              .new_package_name = absl::GetFlag(FLAGS_force_package_name)
                                      .value_or(package->name()),
              .strip_location_info = absl::GetFlag(FLAGS_strip_location_info),
              .strip_node_names = absl::GetFlag(FLAGS_strip_node_names),
              .strip_function_names = absl::GetFlag(FLAGS_strip_function_names),
              .strip_chan_names = absl::GetFlag(FLAGS_strip_chan_names),
              .strip_reg_names = absl::GetFlag(FLAGS_strip_reg_names)}));
  std::cout << result->DumpIr();
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

  std::string_view ir_file = positional_arguments[0] == "-"
                                 ? "/proc/self/fd/0"
                                 : positional_arguments[0];

  return xls::ExitStatus(xls::RealMain(ir_file));
}
