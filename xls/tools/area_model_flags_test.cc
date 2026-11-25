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

#include "xls/tools/area_model_flags.h"

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
#include "xls/estimators/area_model/area_estimator.h"

ABSL_DECLARE_FLAG(std::string, area_model);

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::IsNull;

class AreaModelFlagsTest : public ::testing::Test {
 private:
  absl::FlagSaver flag_saver_;
};

TEST_F(AreaModelFlagsTest, UnspecifiedReturnsEmptyOrDefault) {
  EXPECT_FALSE(IsAreaModelSpecifiedOnCommandLine());
  EXPECT_EQ(GetAreaModelNameFromFlags(), "");
  EXPECT_EQ(GetAreaModelNameFromFlags("unit"), "unit");
  EXPECT_THAT(GetAreaEstimatorFromFlags(), IsOkAndHolds(IsNull()));
  EXPECT_THAT(
      GetRequiredAreaEstimatorFromFlags(),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("--area_model")));

  XLS_ASSERT_OK_AND_ASSIGN(AreaEstimator * unit_estimator,
                           GetAreaEstimatorFromFlags("unit"));
  ASSERT_NE(unit_estimator, nullptr);
  EXPECT_EQ(unit_estimator->name(), "unit");

  XLS_ASSERT_OK_AND_ASSIGN(AreaEstimator * required_unit,
                           GetRequiredAreaEstimatorFromFlags("unit"));
  EXPECT_EQ(required_unit, unit_estimator);
}

TEST_F(AreaModelFlagsTest, ProgrammaticSetFlagOverridesDefault) {
  absl::SetFlag(&FLAGS_area_model, "unit");

  EXPECT_FALSE(IsAreaModelSpecifiedOnCommandLine());
  EXPECT_EQ(GetAreaModelNameFromFlags(), "unit");
  EXPECT_EQ(GetAreaModelNameFromFlags("asap7"), "unit");

  XLS_ASSERT_OK_AND_ASSIGN(AreaEstimator * estimator,
                           GetAreaEstimatorFromFlags("asap7"));
  ASSERT_NE(estimator, nullptr);
  EXPECT_EQ(estimator->name(), "unit");
}

TEST_F(AreaModelFlagsTest, CommandLineFlagSetsSpecifiedAndOverridesDefault) {
  char arg0[] = "test";
  char arg1[] = "--area_model=asap7";
  char* argv[] = {arg0, arg1};
  absl::ParseCommandLine(2, argv);

  EXPECT_TRUE(IsAreaModelSpecifiedOnCommandLine());
  EXPECT_EQ(GetAreaModelNameFromFlags(), "asap7");
  EXPECT_EQ(GetAreaModelNameFromFlags("unit"), "asap7");

  XLS_ASSERT_OK_AND_ASSIGN(AreaEstimator * estimator,
                           GetAreaEstimatorFromFlags("unit"));
  ASSERT_NE(estimator, nullptr);
  EXPECT_EQ(estimator->name(), "asap7");
}

TEST_F(AreaModelFlagsTest, ExplicitEmptyCommandLineFlagClearsDefault) {
  char arg0[] = "test";
  char arg1[] = "--area_model=";
  char* argv[] = {arg0, arg1};
  absl::ParseCommandLine(2, argv);

  EXPECT_TRUE(IsAreaModelSpecifiedOnCommandLine());
  EXPECT_THAT(GetAreaModelNameFromFlags("unit"), Eq(""));
  EXPECT_THAT(GetAreaEstimatorFromFlags("unit"), IsOkAndHolds(IsNull()));
  EXPECT_THAT(
      GetRequiredAreaEstimatorFromFlags("unit"),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("--area_model")));
}

TEST_F(AreaModelFlagsTest, UnknownModelReturnsNotFound) {
  absl::SetFlag(&FLAGS_area_model, "no_such_area_model");

  EXPECT_THAT(GetAreaEstimatorFromFlags(),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(GetRequiredAreaEstimatorFromFlags(),
              StatusIs(absl::StatusCode::kNotFound));
}

}  // namespace
}  // namespace xls
