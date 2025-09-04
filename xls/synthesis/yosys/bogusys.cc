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
#include <filesystem>
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

static constexpr std::string_view kUsage =
    "A bogus yosys binary used by tests. It regurgitates a precanned stdout "
    "which looks like yosys output and writes a json netlist file output.";

// Canned snippet of yosys output.
const char kYosysOutput[] = R"({
  "design": {
    "num_wires": 11,
    "num_wire_bits": 578,
    "num_public_wires": 11,
    "num_public_wire_bits": 578,
    "num_memories": 0,
    "num_memory_bits": 0,
    "num_processes": 0,
    "num_cells": 224,
    "num_cells_by_type": {
      "CCU2C": 32,
      "TRELLIS_FF": 192
    },
    "area": 1074.385620,
    "sequential_area": 37.324800
  }
})";

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
  std::filesystem::path json_stats_out_path =
      std::filesystem::path(json_out_path).parent_path() / "stats.json";
  XLS_RETURN_IF_ERROR(SetFileContents(json_stats_out_path, kYosysOutput));
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
