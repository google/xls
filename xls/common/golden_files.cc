// Copyright 2021 The XLS Authors
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

#include "xls/common/golden_files.h"

#include <filesystem>  // NOLINT
#include <string>
#include <string_view>

#include "gtest/gtest.h"
#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/strip.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/source_location.h"
#include "xls/common/status/matchers.h"
#include "xls/common/update_golden_files_flag.inc"

ABSL_FLAG(std::string, xls_source_dir, "",
          "Absolute path to the root of XLS source directory to modify "
          "when --test_update_golden_files is given.");

namespace xls {

void ExpectEqualToGoldenFile(const std::filesystem::path& golden_file_path,
                             std::string_view text, xabsl::SourceLocation loc) {
  VLOG(1) << "Reading golden Verilog from: " << golden_file_path;
  if (absl::GetFlag(FLAGS_test_update_golden_files)) {
    CHECK(!absl::GetFlag(FLAGS_xls_source_dir).empty())
        << "Must specify --xls_source_dir with --test_update_golden_files.";
    // Strip the xls off the end of the xls_source_dir as the golden file path
    // already includes xls.
    std::string xls_source_dir = absl::GetFlag(FLAGS_xls_source_dir);
    xls_source_dir = std::string(absl::StripSuffix(xls_source_dir, "/"));
    std::filesystem::path xls_source_pardir =
        std::filesystem::path(xls_source_dir).remove_filename();
    std::filesystem::path abs_path = xls_source_pardir / golden_file_path;

    LOG(INFO) << "Updating golden file; abs_path: " << abs_path;
    CHECK_OK(SetFileContents(abs_path, text));
    LOG(INFO) << "Updated golden file: " << golden_file_path;
  } else {
    XLS_ASSERT_OK_AND_ASSIGN(std::filesystem::path abs_path,
                             GetXlsRunfilePath(golden_file_path));
    std::string golden = GetFileContents(abs_path).value();
    testing::ScopedTrace trace(loc.file_name(), loc.line(),
                               "ExpectEqualToGoldenFile failed");
    EXPECT_EQ(text, golden);
  }
}

}  // namespace xls
