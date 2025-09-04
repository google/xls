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
#include <filesystem>
#include <string>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/status/matchers.h"

namespace xls {
namespace synthesis {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::UnorderedElementsAre;

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
  std::string input = R"({
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
  XLS_ASSERT_OK_AND_ASSIGN(YosysSynthesisStatistics stats,
                           ParseYosysJsonOutput(input));
  EXPECT_THAT(stats.cell_histogram,
              UnorderedElementsAre(std::pair(std::string("CCU2C"), 32),
                                   std::pair(std::string("TRELLIS_FF"), 192)));
  EXPECT_THAT(stats.area, 1074.385620);
  EXPECT_THAT(stats.sequential_area, 37.324800);
}

TEST(YosysUtilTest, ParseYosysOutputWithoutAreaStats) {
  std::string input = R"({
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
    }
  }
})";
  XLS_ASSERT_OK_AND_ASSIGN(YosysSynthesisStatistics stats,
                           ParseYosysJsonOutput(input));
  EXPECT_THAT(stats.cell_histogram,
              UnorderedElementsAre(std::pair(std::string("CCU2C"), 32),
                                   std::pair(std::string("TRELLIS_FF"), 192)));
  EXPECT_THAT(stats.area, -1);
  EXPECT_THAT(stats.sequential_area, -1);
}

TEST(YosysUtilTest, ParseSTAOutput) {
  std::string input = R"(
op_clk period_min = 47.22 fmax = 21179.00
worst slack max 22.95
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
