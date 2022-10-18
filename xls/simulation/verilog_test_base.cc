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

#include "xls/simulation/verilog_test_base.h"

#include <cstdlib>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/golden_files.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/matchers.h"
#include "xls/simulation/verilog_simulators.h"

namespace xls {
namespace verilog {

absl::Status VerilogTestBase::ValidateVerilog(
    std::string_view text, absl::Span<const VerilogInclude> includes) {
  return GetDefaultVerilogSimulator().RunSyntaxChecking(text, GetFileType(),
                                                        includes);
}

void VerilogTestBase::ExpectVerilogEqual(
    std::string_view expected, std::string_view actual,
    absl::Span<const VerilogInclude> includes) {
  XLS_VLOG_LINES(1, absl::StrCat("Actual Verilog:\n", actual));
  XLS_VLOG_LINES(1, absl::StrCat("Expected Verilog:\n", expected));
  EXPECT_EQ(expected, actual);
  XLS_EXPECT_OK(ValidateVerilog(actual, includes));
}

void VerilogTestBase::ExpectVerilogEqualToGoldenFile(
    const std::filesystem::path& golden_file_path, std::string_view text,
    absl::Span<const VerilogInclude> includes, xabsl::SourceLocation loc) {
  ExpectEqualToGoldenFile(golden_file_path, text, loc);
  XLS_EXPECT_OK(ValidateVerilog(text, includes));
}

std::filesystem::path VerilogTestBase::GoldenFilePath(
    std::string_view test_file_name,
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
