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

#include "xls/simulation/verilog_test_base.h"

#include <cstdlib>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/statusor.h"
#include "xls/simulation/update_golden_files.inc"
#include "xls/simulation/verilog_simulators.h"

ABSL_FLAG(std::string, xls_source_dir, "",
          "Absolute path to the root of XLS source directory to modify "
          "when --test_update_golden_files is given.");

namespace xls {
namespace verilog {

absl::Status VerilogTestBase::ValidateVerilog(
    absl::string_view text, absl::Span<const VerilogInclude> includes) {
  return GetDefaultVerilogSimulator().RunSyntaxChecking(text, includes);
}

void VerilogTestBase::ExpectVerilogEqual(
    absl::string_view expected, absl::string_view actual,
    absl::Span<const VerilogInclude> includes) {
  XLS_VLOG_LINES(1, absl::StrCat("Actual Verilog:\n", actual));
  XLS_VLOG_LINES(1, absl::StrCat("Expected Verilog:\n", expected));
  EXPECT_EQ(expected, actual);
  XLS_EXPECT_OK(ValidateVerilog(actual, includes));
}

void VerilogTestBase::ExpectVerilogEqualToGoldenFile(
    const std::filesystem::path& golden_file_path, absl::string_view text,
    absl::Span<const VerilogInclude> includes) {
  XLS_VLOG(1) << "Reading golden Verilog from: " << golden_file_path;
  if (absl::GetFlag(FLAGS_test_update_golden_files)) {
    XLS_CHECK(!absl::GetFlag(FLAGS_xls_source_dir).empty())
        << "Must specify --xls_source_dir with --test_update_golden_files.";
    // Strip the xls off the end of the xls_source_dir as the golden file path
    // already includes xls.
    std::filesystem::path abs_path =
        std::filesystem::path(absl::GetFlag(FLAGS_xls_source_dir))
            .remove_filename() /
        golden_file_path;
    XLS_CHECK_OK(SetFileContents(abs_path, text));
    XLS_LOG(INFO) << "Updated golden file: " << golden_file_path;
  } else {
    std::filesystem::path abs_path = GetXlsRunfilePath(golden_file_path);
    std::string golden = GetFileContents(abs_path).value();
    ExpectVerilogEqual(golden, text, includes);
  }
}

std::filesystem::path VerilogTestBase::GoldenFilePath(
    absl::string_view test_file_name,
    const std::filesystem::path& testdata_dir) {
  // We suffix the golden reference files with "txt" on top of the extension
  // just to indicate they're compiler byproduct comparison points and not
  // Verilog files that have been written by hand.
  std::string filename =
      absl::StrFormat("%s_%s.%s", test_file_name, TestBaseName(),
                      UseSystemVerilog() ? "svtxt" : "vtxt");
  return testdata_dir / filename;
}

}  // namespace verilog
}  // namespace xls
