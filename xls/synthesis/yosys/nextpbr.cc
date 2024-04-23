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
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/types/span.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"

const char kUsage[] =
    "A bogus nextpnr binary used by tests. It consumes a json netlist, "
    "regurgitates a precanned stderr, and writes a place-and-route result "
    "file.";

namespace xls {
namespace synthesis {
namespace {

const char kConfigContents[] = R"(.device LFE5U-45F

.comment Part: LFE5U-45F-6CABGA381

.tile CIB_R10C3:PVT_COUNT2
unknown: F2B0
unknown: F3B0
unknown: F5B0
unknown: F11B0
unknown: F13B0

.tile CIB_R10C43:CIB_EBR
arc: S1_V02S0401 E1_H02W0401

.tile CIB_R10C45:CIB_EBR
arc: W1_H02W0401 V06S0203
)";

// Extracts a flag value argument from the given args. Only handles
// space-separated flags like "--flag VALUE".
absl::StatusOr<std::string> GetFlagValue(std::string_view flag,
                                         absl::Span<const std::string> args) {
  // The string could conceivably span arguments so just join all the arguments
  // then split them.
  std::string joined_args = absl::StrJoin(args, " ");
  std::vector<std::string> split_args = absl::StrSplit(joined_args, ' ');
  for (int64_t i = 0; i < split_args.size(); ++i) {
    if (split_args[i] == flag) {
      return split_args[i + 1];
    }
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("%s flag not found in arguments.", flag));
}

absl::Status RealMain(absl::Span<const std::string> args) {
  XLS_ASSIGN_OR_RETURN(std::string json_path, GetFlagValue("--json", args));
  XLS_ASSIGN_OR_RETURN(std::string json, GetFileContents(json_path));
  XLS_RET_CHECK(absl::StrContains(json, "\"modules\": {"));

  XLS_ASSIGN_OR_RETURN(std::string cfg_path, GetFlagValue("--textcfg", args));
  XLS_RETURN_IF_ERROR(SetFileContents(cfg_path, kConfigContents));

  XLS_ASSIGN_OR_RETURN(
      std::string runfile_path,
      GetXlsRunfilePath("xls/synthesis/yosys/testdata/nextpnr.out"));
  XLS_ASSIGN_OR_RETURN(std::string output, GetFileContents(runfile_path));
  // nextpnr prints output on stderr.
  std::cerr << output;
  return absl::OkStatus();
}

}  // namespace
}  // namespace synthesis
}  // namespace xls

int main(int argc, char** argv) {
  // Call InitXls but don't pass it all the argv elements because we don't want
  // to do flag parsing because or we will have to define all the flags nextpnr
  // accepts.
  xls::InitXls(kUsage, 1, argv);
  std::vector<std::string> args(argc);
  for (int i = 0; i < argc; ++i) {
    args[i] = argv[i];
  }
  return xls::ExitStatus(xls::synthesis::RealMain(args));
}
