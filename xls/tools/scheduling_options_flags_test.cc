// Copyright 2026 The XLS Authors
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

#include "xls/tools/scheduling_options_flags.h"

#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/flags/commandlineflag.h"
#include "absl/flags/reflection.h"
#include "xls/common/proto_test_utils.h"
#include "xls/common/status/matchers.h"
#include "xls/tools/scheduling_options_flags.pb.h"

namespace xls {
namespace {

using ::xls::proto_testing::EqualsProto;
using ::xls::proto_testing::Partially;

TEST(SchedulingOptionsFlagsTest,
     ArcThroughputFlagsParsedFromCommandLineWithSpaces) {
  absl::CommandLineFlag* flag =
      absl::FindCommandLineFlag("arc_worst_case_throughput");
  ASSERT_NE(flag, nullptr);

  std::string error;
  ASSERT_TRUE(flag->ParseFrom("L_W1:L_R1=1, L_W2:L_R2=2", &error)) << error;

  absl::CommandLineFlag* default_flag =
      absl::FindCommandLineFlag("default_arc_worst_case_throughput");
  ASSERT_NE(default_flag, nullptr);
  ASSERT_TRUE(default_flag->ParseFrom("3", &error)) << error;

  XLS_ASSERT_OK_AND_ASSIGN(SchedulingOptionsFlagsProto proto,
                           GetSchedulingOptionsFlagsProto());

  EXPECT_THAT(proto, Partially(EqualsProto(R"pb(
                default_arc_worst_case_throughput: 3
                arc_worst_case_throughput {
                  key: "L_W1"
                  value { read_to_throughput { key: "L_R1" value: 1 } }
                }
                arc_worst_case_throughput {
                  key: "L_W2"
                  value { read_to_throughput { key: "L_R2" value: 2 } }
                }
              )pb")));
}

TEST(SchedulingOptionsFlagsTest, ArcThroughputFlagsParseInvalidFormat) {
  absl::CommandLineFlag* flag =
      absl::FindCommandLineFlag("arc_worst_case_throughput");
  ASSERT_NE(flag, nullptr);

  std::string error;

  // Missing '=T' (Single Entry)
  EXPECT_FALSE(flag->ParseFrom("L_W1:L_R1", &error));
  EXPECT_THAT(error, testing::HasSubstr(
                         "Expected L_W:L_R=T format, but found: L_W1:L_R1"));

  // Missing '=T' (Multi Entry)
  EXPECT_FALSE(flag->ParseFrom("L_W1:L_R1=1,L_W2", &error));
  EXPECT_THAT(error,
              testing::HasSubstr("Expected L_W:L_R=T format, but found: L_W2"));

  // Missing ':L_R'
  EXPECT_FALSE(flag->ParseFrom("L_W1=1", &error));
  EXPECT_THAT(error, testing::HasSubstr(
                         "Expected L_W:L_R format for arc, but found: L_W1"));

  // Too many '='
  EXPECT_FALSE(flag->ParseFrom("L_W1:L_R1=1=2", &error));
  EXPECT_THAT(error,
              testing::HasSubstr(
                  "Expected L_W:L_R=T format, but found: L_W1:L_R1=1=2"));

  // Too many ':'
  EXPECT_FALSE(flag->ParseFrom("L_W1:L_R1:L_R2=1", &error));
  EXPECT_THAT(
      error, testing::HasSubstr(
                 "Expected L_W:L_R format for arc, but found: L_W1:L_R1:L_R2"));
}

TEST(SchedulingOptionsFlagsTest,
     ArcThroughputFlagsParseErrorInvalidThroughput) {
  absl::CommandLineFlag* flag =
      absl::FindCommandLineFlag("arc_worst_case_throughput");
  ASSERT_NE(flag, nullptr);

  std::string error;
  EXPECT_FALSE(flag->ParseFrom("L_W1:L_R1=not_an_int", &error));
  EXPECT_THAT(error,
              testing::HasSubstr(
                  "Invalid throughput value for arc L_W1:L_R1: not_an_int"));
}

}  // namespace
}  // namespace xls
