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

#include "xls/synthesis/yosys/yosys_util.h"

#include <filesystem>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"

namespace xls {
namespace synthesis {
namespace {

using status_testing::StatusIs;
using testing::HasSubstr;

TEST(YosysUtilTest, GetMaxFrequency) {
  XLS_ASSERT_OK_AND_ASSIGN(
      std::filesystem::path output_path,
      GetXlsRunfilePath("xls/synthesis/yosys/testdata/nextpnr.out"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string output, GetFileContents(output_path));
  XLS_ASSERT_OK_AND_ASSIGN(int64 freq_max, ParseNextpnrOutput(output));
  EXPECT_EQ(freq_max, 180280000);
}

TEST(YosysUtilTest, GetMaxFrequencyFailed) {
  EXPECT_THAT(
      ParseNextpnrOutput("total garbage"),
      StatusIs(
          absl::StatusCode::kNotFound,
          HasSubstr("Could not find maximum frequency in nextpnr output")));
}

}  // namespace
}  // namespace synthesis
}  // namespace xls
