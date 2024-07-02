// Copyright 2020 Google LLC
//
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

#include "xls/synthesis/yosys/yosys_util.h"

#include <cstdint>
#include <filesystem>  // NOLINT
#include <string>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/status/matchers.h"

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

TEST(YosysUtilTest, ParseSTAOutput) {
  std::string input = R"(
op_clk period_min = 47.22 fmax = 21179.00
worst slack 22.95
tns 0.00
Startpoint: op0 (input port clocked by op_clk)
Endpoint: _13_ (falling edge-triggered flip-flop clocked by op_clk')
Path Group: op_clk
Path Type: min

Fanout     Cap    Slew   Delay    Time   Description
-----------------------------------------------------------------------------
                  0.00    0.00    0.00   clock op_clk (rise edge)
                          0.00    0.00   clock network delay (ideal)
                         15.38   15.38 ^ input external delay
                  0.00    0.00   15.38 ^ op0 (in)
     1    0.56                           op0 (net)
                  0.00    0.00   15.38 ^ _13_/D (DFFLQNx1_ASAP7_75t_R)
                                 15.38   data arrival time

                  0.00    0.00    0.00   clock op_clk' (fall edge)
                          0.00    0.00   clock network delay (ideal)
                          0.00    0.00   clock reconvergence pessimism
                                  0.00 v _13_/CLK (DFFLQNx1_ASAP7_75t_R)
                          6.20    6.20   library hold time
                                  6.20   data required time
-----------------------------------------------------------------------------
                                  6.20   data required time
                                -15.38   data arrival time
-----------------------------------------------------------------------------
                                  9.18   slack (MET)


Startpoint: _12_ (falling edge-triggered flip-flop clocked by op_clk')
Endpoint: out (output port clocked by op_clk)
Path Group: op_clk
Path Type: max

Fanout     Cap    Slew   Delay    Time   Description
-----------------------------------------------------------------------------
                  0.00    0.00    0.00   clock op_clk' (fall edge)
                          0.00    0.00   clock network delay (ideal)
                  0.00    0.00    0.00 v _12_/CLK (DFFLQNx1_ASAP7_75t_R)
                 12.53   33.79   33.79 v _12_/QN (DFFLQNx1_ASAP7_75t_R)
     1    0.68                           _7_ (net)
                 12.53    0.00   33.79 v _11_/A (INVx1_ASAP7_75t_R)
                  4.49    4.80   38.59 ^ _11_/Y (INVx1_ASAP7_75t_R)
     1    0.00                           _5_ (net)
                  4.49    0.00   38.59 ^ out (out)
                                 38.59   data arrival time

                  0.00   76.92   76.92   clock op_clk (rise edge)
                          0.00   76.92   clock network delay (ideal)
                          0.00   76.92   clock reconvergence pessimism
                        -15.38   61.54   output external delay
                                 61.54   data required time
-----------------------------------------------------------------------------
                                 61.54   data required time
                                -38.59   data arrival time
-----------------------------------------------------------------------------
                                 22.95   slack (MET)
  )";
  XLS_ASSERT_OK_AND_ASSIGN(STAStatistics sta_stats, ParseOpenSTAOutput(input));
  EXPECT_EQ((sta_stats.period_ps * 100), 4722);  // compare ints, not floats
  EXPECT_EQ(sta_stats.slack_ps, 22);
  EXPECT_EQ(sta_stats.max_frequency_hz, 21177466627);
}

}  // namespace
}  // namespace synthesis
}  // namespace xls
