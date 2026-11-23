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
#include "ortools/math_opt/parameters.pb.h"
#include "xls/common/proto_test_utils.h"
#include "xls/common/status/matchers.h"
#include "xls/passes/optimization_pass.h"
#include "xls/scheduling/scheduling_options.h"
#include "xls/tools/scheduling_options_flags.pb.h"

namespace xls {
namespace {

using ::xls::proto_testing::EqualsProto;
using ::xls::proto_testing::Partially;

TEST(SchedulingOptionsFlagsTest, ProtoDefaultInstanceMatchesFlagDefaults) {
  absl::FlagSaver flag_saver;
  const SchedulingOptionsFlagsProto& defaults =
      SchedulingOptionsFlagsProto::default_instance();

  EXPECT_EQ(defaults.opt_level(), kMaxOptLevel);
  EXPECT_EQ(defaults.scheduling_strategy(),
            ProtoSchedulingStrategy::SCHEDULER_TYPE_SDC);
  EXPECT_EQ(defaults.find_bounds_strategy(),
            ProtoSchedulingStrategy::SCHEDULER_TYPE_UNSPECIFIED);
  EXPECT_TRUE(defaults.minimize_clock_on_failure());
  EXPECT_FALSE(defaults.recover_after_minimizing_clock());
  EXPECT_FALSE(defaults.minimize_worst_case_throughput());
  EXPECT_EQ(defaults.mutual_exclusion_z3_rlimit(), -1);
  EXPECT_EQ(defaults.default_next_value_z3_rlimit(), -1);
  EXPECT_FALSE(defaults.use_fdo());
  EXPECT_EQ(defaults.fdo_iteration_number(), 5);
  EXPECT_EQ(defaults.fdo_delay_driven_path_number(), 1);
  EXPECT_EQ(defaults.fdo_fanout_driven_path_number(), 0);
  EXPECT_FLOAT_EQ(defaults.fdo_refinement_stochastic_ratio(), 1.0f);
  EXPECT_EQ(defaults.fdo_path_evaluate_strategy(), "window");
  EXPECT_EQ(defaults.fdo_synthesizer_name(), "yosys");
  EXPECT_TRUE(defaults.merge_on_mutual_exclusion());
  EXPECT_TRUE(defaults.multi_proc());
  EXPECT_DOUBLE_EQ(defaults.sdc_solution_tolerance(),
                   kDefaultSdcSolutionTolerance);
  EXPECT_EQ(defaults.solver_type(),
            operations_research::math_opt::SOLVER_TYPE_GLOP);
  EXPECT_EQ(defaults.solver_kind(), SolverKind::SOLVER_KIND_Z3);
  EXPECT_TRUE(SchedulingFailureBehaviorProto::default_instance()
                  .explain_infeasibility());

  XLS_ASSERT_OK_AND_ASSIGN(SchedulingOptionsFlagsProto from_flags,
                           GetSchedulingOptionsFlagsProto());
  EXPECT_EQ(from_flags.opt_level(), defaults.opt_level());
  EXPECT_EQ(from_flags.scheduling_strategy(), defaults.scheduling_strategy());
  EXPECT_EQ(from_flags.find_bounds_strategy(), defaults.scheduling_strategy());
  EXPECT_EQ(from_flags.minimize_clock_on_failure(),
            defaults.minimize_clock_on_failure());
  EXPECT_EQ(from_flags.mutual_exclusion_z3_rlimit(),
            defaults.mutual_exclusion_z3_rlimit());
  EXPECT_EQ(from_flags.default_next_value_z3_rlimit(),
            defaults.default_next_value_z3_rlimit());
  EXPECT_EQ(from_flags.fdo_iteration_number(), defaults.fdo_iteration_number());
  EXPECT_EQ(from_flags.fdo_delay_driven_path_number(),
            defaults.fdo_delay_driven_path_number());
  EXPECT_FLOAT_EQ(from_flags.fdo_refinement_stochastic_ratio(),
                  defaults.fdo_refinement_stochastic_ratio());
  EXPECT_EQ(from_flags.fdo_path_evaluate_strategy(),
            defaults.fdo_path_evaluate_strategy());
  EXPECT_EQ(from_flags.fdo_synthesizer_name(), defaults.fdo_synthesizer_name());
  EXPECT_EQ(from_flags.merge_on_mutual_exclusion(),
            defaults.merge_on_mutual_exclusion());
  EXPECT_EQ(from_flags.multi_proc(), defaults.multi_proc());
  EXPECT_DOUBLE_EQ(from_flags.sdc_solution_tolerance(),
                   defaults.sdc_solution_tolerance());
  EXPECT_EQ(from_flags.solver_type(), defaults.solver_type());
  EXPECT_EQ(from_flags.solver_kind(), defaults.solver_kind());
  EXPECT_EQ(from_flags.failure_behavior().explain_infeasibility(),
            SchedulingFailureBehaviorProto::default_instance()
                .explain_infeasibility());
}

TEST(SchedulingOptionsFlagsTest,
     ArcThroughputFlagsParsedFromCommandLineWithSpaces) {
  absl::FlagSaver flag_saver;
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
