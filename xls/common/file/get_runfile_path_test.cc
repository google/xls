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

#include "xls/common/file/get_runfile_path.h"

#include <filesystem>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/status/matchers.h"

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::HasSubstr;

TEST(GetRunfilePathTest, GetXlsRunfilePathReturnsTheRightPath) {
  XLS_ASSERT_OK_AND_ASSIGN(
      std::filesystem::path runfile_path,
      GetXlsRunfilePath("xls/common/file/get_runfile_path_test.cc"));
  absl::StatusOr<std::string> test_cc_file_contents =
      GetFileContents(runfile_path);

  EXPECT_THAT(
      test_cc_file_contents,
      IsOkAndHolds(HasSubstr("Some string that's only in this file. 1543234")));
}

}  // namespace
}  // namespace xls
