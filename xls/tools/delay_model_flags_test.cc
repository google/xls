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

#include "xls/tools/delay_model_flags.h"

#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/flags/declare.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/reflection.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xls/common/status/matchers.h"
#include "xls/estimators/delay_model/delay_estimator.h"

ABSL_DECLARE_FLAG(std::string, delay_model);

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::IsNull;

class DelayModelFlagsTest : public ::testing::Test {
 private:
  absl::FlagSaver flag_saver_;
};

TEST_F(DelayModelFlagsTest, UnspecifiedReturnsEmptyOrDefault) {
  EXPECT_FALSE(IsDelayModelSpecifiedOnCommandLine());
  EXPECT_EQ(GetDelayModelNameFromFlags(), "");
  EXPECT_EQ(GetDelayModelNameFromFlags("unit"), "unit");
  EXPECT_THAT(GetDelayEstimatorFromFlags(), IsOkAndHolds(IsNull()));
  EXPECT_THAT(
      GetRequiredDelayEstimatorFromFlags(),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("--delay_model")));

  XLS_ASSERT_OK_AND_ASSIGN(DelayEstimator * unit_estimator,
                           GetDelayEstimatorFromFlags("unit"));
  ASSERT_NE(unit_estimator, nullptr);
  EXPECT_EQ(unit_estimator->name(), "unit");

  XLS_ASSERT_OK_AND_ASSIGN(DelayEstimator * required_unit,
                           GetRequiredDelayEstimatorFromFlags("unit"));
  EXPECT_EQ(required_unit, unit_estimator);
}

TEST_F(DelayModelFlagsTest, ProgrammaticSetFlagOverridesDefault) {
  absl::SetFlag(&FLAGS_delay_model, "unit");

  EXPECT_FALSE(IsDelayModelSpecifiedOnCommandLine());
  EXPECT_EQ(GetDelayModelNameFromFlags(), "unit");
  EXPECT_EQ(GetDelayModelNameFromFlags("asap7"), "unit");

  XLS_ASSERT_OK_AND_ASSIGN(DelayEstimator * estimator,
                           GetDelayEstimatorFromFlags("asap7"));
  ASSERT_NE(estimator, nullptr);
  EXPECT_EQ(estimator->name(), "unit");
}

TEST_F(DelayModelFlagsTest, CommandLineFlagSetsSpecifiedAndOverridesDefault) {
  char arg0[] = "test";
  char arg1[] = "--delay_model=asap7";
  char* argv[] = {arg0, arg1};
  absl::ParseCommandLine(2, argv);

  EXPECT_TRUE(IsDelayModelSpecifiedOnCommandLine());
  EXPECT_EQ(GetDelayModelNameFromFlags(), "asap7");
  EXPECT_EQ(GetDelayModelNameFromFlags("unit"), "asap7");

  XLS_ASSERT_OK_AND_ASSIGN(DelayEstimator * estimator,
                           GetDelayEstimatorFromFlags("unit"));
  ASSERT_NE(estimator, nullptr);
  EXPECT_EQ(estimator->name(), "asap7");
}

TEST_F(DelayModelFlagsTest, ExplicitEmptyCommandLineFlagClearsDefault) {
  char arg0[] = "test";
  char arg1[] = "--delay_model=";
  char* argv[] = {arg0, arg1};
  absl::ParseCommandLine(2, argv);

  EXPECT_TRUE(IsDelayModelSpecifiedOnCommandLine());
  EXPECT_THAT(GetDelayModelNameFromFlags("unit"), Eq(""));
  EXPECT_THAT(GetDelayEstimatorFromFlags("unit"), IsOkAndHolds(IsNull()));
  EXPECT_THAT(
      GetRequiredDelayEstimatorFromFlags("unit"),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("--delay_model")));
}

TEST_F(DelayModelFlagsTest, UnknownModelReturnsNotFound) {
  absl::SetFlag(&FLAGS_delay_model, "no_such_delay_model");

  EXPECT_THAT(GetDelayEstimatorFromFlags(),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(GetRequiredDelayEstimatorFromFlags(),
              StatusIs(absl::StatusCode::kNotFound));
}

}  // namespace
}  // namespace xls
