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

#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/types/span.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"

const char kUsage[] =
    "A bogus yosys binary used by tests. It regurgitates a precanned stdout "
    "which looks like yosys output and writes a json netlist file output.";

// Canned snippet of yosys output.
const char kYosysOutput[] = R"(
....

Top module:  \__input__fun

....

2.49. Printing statistics.

=== __input__fun ===

   Number of wires:                 11
   Number of wire bits:            578
   Number of public wires:          11
   Number of public wire bits:     578
   Number of memories:               0
   Number of memory bits:            0
   Number of processes:              0
   Number of cells:                224
     CCU2C                          32
     TRELLIS_FF                    192

2.50. Executing CHECK pass (checking for obvious problems).
checking module __input__fun..
found and reported 0 problems.

....
)";

namespace xls {
namespace synthesis {
namespace {

absl::StatusOr<std::string> GetJsonOutputPath(
    absl::Span<const std::string> args) {
  // Find the "-json PATH" substring in the arguments and extract the path. The
  // string could conceivably span arguments so just join all the arguments then
  // split them.
  std::string joined_args = absl::StrJoin(args, " ");
  std::vector<std::string> split_args = absl::StrSplit(joined_args, ' ');
  for (int64_t i = 0; i < split_args.size(); ++i) {
    if (split_args[i] == "-json") {
      return split_args[i + 1];
    }
  }
  return absl::InvalidArgumentError(
      "'-json FILE' substring not found in arguments.");
}

absl::Status RealMain(absl::Span<const std::string> args) {
  XLS_ASSIGN_OR_RETURN(std::string json_out_path, GetJsonOutputPath(args));
  XLS_ASSIGN_OR_RETURN(
      std::string runfile_path,
      GetXlsRunfilePath("xls/synthesis/yosys/testdata/netlist.json"));
  XLS_ASSIGN_OR_RETURN(std::string json, GetFileContents(runfile_path));
  XLS_RETURN_IF_ERROR(SetFileContents(json_out_path, json));
  std::cout << kYosysOutput;
  return absl::OkStatus();
}

}  // namespace
}  // namespace synthesis
}  // namespace xls

int main(int argc, char** argv) {
  // Call InitXls but don't pass it all the argv elements because we don't want
  // to do flag parsing because or we will have to define all the flags yosys
  // accepts.
  xls::InitXls(kUsage, 1, argv);
  std::vector<std::string> args(argc);
  for (int i = 0; i < argc; ++i) {
    args[i] = argv[i];
  }
  return xls::ExitStatus(xls::synthesis::RealMain(args));
}
