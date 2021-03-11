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
using testing::UnorderedElementsAre;

TEST(YosysUtilTest, GetMaxFrequency) {
  XLS_ASSERT_OK_AND_ASSIGN(
      std::filesystem::path output_path,
      GetXlsRunfilePath("xls/synthesis/yosys/testdata/nextpnr.out"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string output, GetFileContents(output_path));
  XLS_ASSERT_OK_AND_ASSIGN(int64_t freq_max, ParseNextpnrOutput(output));
  EXPECT_EQ(freq_max, 180280000);
}

TEST(YosysUtilTest, GetMaxFrequencyFailed) {
  EXPECT_THAT(
      ParseNextpnrOutput("total garbage"),
      StatusIs(
          absl::StatusCode::kNotFound,
          HasSubstr("Could not find maximum frequency in nextpnr output")));
}

TEST(YosysUtilTest, ParseYosysOutput) {
  std::string input = R"(
2.1.1. Analyzing design hierarchy..
Top module:  \__input__fun
Used module:     \shll_1

2.49. Printing statistics.

=== __input__dummy ===

   Number of wires:                  0
   Number of wire bits:              0
   Number of public wires:           0
   Number of public wire bits:       0
   Number of memories:               0
   Number of memory bits:            0
   Number of processes:              0
   Number of cells:                  0
     CCU2C                           0
     TRELLIS_FF                      0

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
  )";
  XLS_ASSERT_OK_AND_ASSIGN(YosysSynthesisStatistics stats,
                           ParseYosysOutput(input));
  EXPECT_THAT(stats.cell_histogram,
              UnorderedElementsAre(std::pair(std::string("CCU2C"), 32),
                                   std::pair(std::string("TRELLIS_FF"), 192)));
}

TEST(YosysUtilTest, ParseYosysOutputTopModuleMissing) {
  std::string input = R"(
2.1.1. Analyzing design hierarchy..
Used module:     \shll_1

2.49. Printing statistics.

=== __input__dummy ===

   Number of wires:                  0
   Number of wire bits:              0
   Number of public wires:           0
   Number of public wire bits:       0
   Number of memories:               0
   Number of memory bits:            0
   Number of processes:              0
   Number of cells:                  0
     CCU2C                           0
     TRELLIS_FF                      0

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
  )";
  auto result = ParseYosysOutput(input);
  EXPECT_THAT(result,
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("ParseYosysOutput could not find the term "
                                 "\"Top module\" in the yosys output")));
}

TEST(YosysUtilTest, ParseYosysOutputNoStatsToParse) {
  std::string input = R"(
2.1.1. Analyzing design hierarchy..
Top module:  \__input__fun
Used module:     \shll_1

2.49. Printing statistics.

=== __input__fun ===

   Number of bugs in yosys_util:                 0

2.50. Executing CHECK pass (checking for obvious problems).
checking module __input__fun..
found and reported 0 problems.
  )";
  auto result = ParseYosysOutput(input);
  EXPECT_THAT(result,
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("ParseYosysOutput could not find the term "
                                 "\"Number of cells:\" in the yosys output")));
}

}  // namespace
}  // namespace synthesis
}  // namespace xls
