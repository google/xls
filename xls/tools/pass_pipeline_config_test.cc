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

#include "xls/tools/pass_pipeline_config.h"

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_pipeline.pb.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/pass_pipeline.pb.h"

namespace xls {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::IsNull;
using ::testing::Not;
using ::testing::NotNull;

absl::StatusOr<std::unique_ptr<Package>> MakeTestPackage() {
  auto pkg = std::make_unique<Package>("test_pkg");
  {
    FunctionBuilder fb("dead_fn", pkg.get());
    fb.Identity(fb.Param("x", pkg->GetBitsType(32)));
    XLS_RETURN_IF_ERROR(fb.Build().status());
  }
  {
    FunctionBuilder fb("main", pkg.get());
    fb.Add(fb.Param("x", pkg->GetBitsType(32)), fb.Literal(UBits(0, 32)));
    XLS_ASSIGN_OR_RETURN(Function * main, fb.Build());
    XLS_RETURN_IF_ERROR(pkg->SetTop(main));
  }
  return pkg;
}

OptimizationPipelineProto MakeCustomRegistry(std::string_view compound_name) {
  OptimizationPipelineProto proto;
  auto* compound = proto.add_compound_passes();
  compound->set_short_name(std::string(compound_name));
  compound->set_long_name("Custom test group");
  compound->add_passes("ident_remove");
  compound->add_passes("dce");
  proto.add_default_pipeline(std::string(compound_name));
  return proto;
}

TEST(PassPipelineConfigTest, MaybeCreateCustomOptimizationRegistryUnset) {
  XLS_ASSERT_OK_AND_ASSIGN(std::optional<OptimizationPassRegistry> registry,
                           MaybeCreateCustomOptimizationRegistry(std::nullopt));
  EXPECT_FALSE(registry.has_value());
}

TEST(PassPipelineConfigTest, MaybeCreateCustomOptimizationRegistryRegisters) {
  OptimizationPipelineProto proto = MakeCustomRegistry("my_custom_group");
  XLS_ASSERT_OK_AND_ASSIGN(std::optional<OptimizationPassRegistry> registry,
                           MaybeCreateCustomOptimizationRegistry(proto));
  ASSERT_TRUE(registry.has_value());
  EXPECT_THAT(GetAvailablePassesStr(proto),
              IsOkAndHolds(HasSubstr("my_custom_group")));
  EXPECT_THAT(GetAvailablePassesStr(std::nullopt),
              IsOkAndHolds(Not(HasSubstr("my_custom_group"))));
}

TEST(PassPipelineConfigTest, ParsePassPipelineFromStringAndEmpty) {
  XLS_ASSERT_OK_AND_ASSIGN(PassPipelineProto empty_proto,
                           ParsePassPipelineFromString(""));
  EXPECT_TRUE(empty_proto.has_top());
  EXPECT_TRUE(empty_proto.top().has_pipeline());
  EXPECT_THAT(empty_proto.top().pipeline().elements(), IsEmpty());

  XLS_ASSERT_OK_AND_ASSIGN(PassPipelineProto simp_proto,
                           ParsePassPipelineFromString("basic_simp dce"));
  EXPECT_EQ(simp_proto.top().pipeline().elements_size(), 2);

  EXPECT_THAT(ParsePassPipelineFromString("no_such_pass_xyz"),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(PassPipelineConfigTest, BuildNoneDefaultReturnsNullptrWhenUnset) {
  OptimizationPipelineConfig config;
  EXPECT_FALSE(config.HasExplicitPipeline());
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OptimizationCompoundPass> pipeline,
      BuildOptimizationPassPipeline(config, DefaultPassPipeline::kNone));
  EXPECT_THAT(pipeline, IsNull());
}

TEST(PassPipelineConfigTest, BuildNoneDefaultWithCustomRegistryRunsDefault) {
  OptimizationPipelineConfig config;
  config.custom_registry = MakeCustomRegistry("my_custom_group");
  EXPECT_TRUE(config.HasExplicitPipeline());
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OptimizationCompoundPass> pipeline,
      BuildOptimizationPassPipeline(config, DefaultPassPipeline::kNone));
  EXPECT_THAT(pipeline, NotNull());
}

TEST(PassPipelineConfigTest,
     BuildWithRegistryNoneReturnsNullptrIgnoringCustomRegistry) {
  OptimizationPipelineConfig config;
  config.custom_registry = MakeCustomRegistry("my_custom_group");
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OptimizationCompoundPass> pipeline,
      BuildOptimizationPassPipelineWithRegistry(
          config, GetOptimizationRegistry(), DefaultPassPipeline::kNone));
  EXPECT_THAT(pipeline, IsNull());
}

TEST(PassPipelineConfigTest, BuildExplicitPipelineRunsOnlySpecifiedPasses) {
  OptimizationPipelineConfig config;
  XLS_ASSERT_OK_AND_ASSIGN(config.pipeline,
                           ParsePassPipelineFromString("basic_simp dce"));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<OptimizationCompoundPass> pipeline,
                           BuildOptimizationPassPipeline(config));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> pkg, MakeTestPackage());
  PassResults results;
  OptimizationContext ctx;
  XLS_ASSERT_OK(pipeline->Run(pkg.get(), {}, &results, ctx).status());
  XLS_ASSERT_OK_AND_ASSIGN(Function * main, pkg->GetTopAsFunction());
  EXPECT_EQ(main->return_value()->op(), Op::kParam);
  // Dead function was untouched since `dfe` was not included.
  EXPECT_THAT(pkg->GetFunction("dead_fn").status(), IsOk());
}

TEST(PassPipelineConfigTest, BuildNoneDefaultWithEmptyPipelineIsNoop) {
  OptimizationPipelineConfig config;
  XLS_ASSERT_OK_AND_ASSIGN(config.pipeline, ParsePassPipelineFromString(""));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OptimizationCompoundPass> pipeline,
      BuildOptimizationPassPipeline(config, DefaultPassPipeline::kNone));
  ASSERT_THAT(pipeline, NotNull());
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> pkg, MakeTestPackage());
  PassResults results;
  OptimizationContext ctx;
  XLS_ASSERT_OK_AND_ASSIGN(bool changed,
                           pipeline->Run(pkg.get(), {}, &results, ctx));
  EXPECT_FALSE(changed);
  XLS_ASSERT_OK_AND_ASSIGN(Function * main, pkg->GetTopAsFunction());
  EXPECT_EQ(main->return_value()->op(), Op::kAdd);
  EXPECT_THAT(pkg->GetFunction("dead_fn").status(), IsOk());
}

TEST(PassPipelineConfigTest, BuildRejectsSkipPassesWithCustomPipeline) {
  OptimizationPipelineConfig config;
  XLS_ASSERT_OK_AND_ASSIGN(config.pipeline, ParsePassPipelineFromString("dce"));
  config.skip_passes = {"dfe"};
  EXPECT_THAT(BuildOptimizationPassPipeline(config),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(PassPipelineConfigTest, BuildDefaultWithSkipPassesAndDebug) {
  OptimizationPipelineConfig config;
  config.skip_passes = {"dfe"};
  config.debug_optimizations = true;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<OptimizationCompoundPass> pipeline,
                           BuildOptimizationPassPipeline(config));
  OptimizationPassOptions opts;
  ApplyOptimizationPipelineOptions(config, opts);
  EXPECT_EQ(opts.skip_passes, std::vector<std::string>({"dfe"}));

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> pkg, MakeTestPackage());
  PassResults results;
  OptimizationContext ctx;
  XLS_ASSERT_OK(pipeline->Run(pkg.get(), opts, &results, ctx).status());
  XLS_ASSERT_OK_AND_ASSIGN(Function * main, pkg->GetTopAsFunction());
  EXPECT_EQ(main->return_value()->op(), Op::kParam);
  // `dfe` was skipped, so `dead_fn` remains.
  EXPECT_THAT(pkg->GetFunction("dead_fn").status(), IsOk());
}

TEST(PassPipelineConfigTest, ApplyOptionsAndCheckPassesBisectLimit) {
  OptimizationPipelineConfig config;
  config.passes_bisect_limit = 5;
  config.passes_bisect_limit_is_error = false;

  OptimizationPassOptions opts;
  ApplyOptimizationPipelineOptions(config, opts);
  EXPECT_EQ(opts.bisect_limit, 5);
  EXPECT_THAT(CheckPassesBisectLimit(config, /*total_passes=*/5), IsOk());

  config.passes_bisect_limit_is_error = true;
  EXPECT_THAT(CheckPassesBisectLimit(config, /*total_passes=*/4), IsOk());
  EXPECT_THAT(CheckPassesBisectLimit(config, /*total_passes=*/5),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("passes bisect limit was reached")));
  EXPECT_THAT(CheckPassesBisectLimit(config, /*total_passes=*/10),
              StatusIs(absl::StatusCode::kInternal));

  config.passes_bisect_limit = std::nullopt;
  EXPECT_THAT(CheckPassesBisectLimit(config, /*total_passes=*/10), IsOk());
}

}  // namespace
}  // namespace xls
