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

#include "xls/passes/split_pipeline.h"

#include <memory>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/status/matchers.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_pipeline.pb.h"
#include "xls/passes/optimization_pass_registry.h"

namespace xls {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::NotNull;

std::vector<std::string> GetSubpassShortNames(
    const OptimizationCompoundPass& compound_pass) {
  std::vector<std::string> names;
  names.reserve(compound_pass.passes().size());
  for (const OptimizationPass* pass : compound_pass.passes()) {
    names.push_back(pass->short_name());
  }
  return names;
}

TEST(SplitPipelineTest, SplitMultiplePasses) {
  OptimizationPipelineProto pipeline_proto;
  XLS_ASSERT_OK(ParseTextProto(R"pb(
                                 compound_passes {
                                   short_name: "pre_split_1"
                                   long_name: "Pre Split 1"
                                   passes: "dce"
                                 }
                                 compound_passes {
                                   short_name: "pre_split_2"
                                   long_name: "Pre Split 2"
                                   passes: "dce"
                                 }
                                 compound_passes {
                                   short_name: "split_step"
                                   long_name: "Split Step"
                                   passes: "resource_sharing"
                                 }
                                 compound_passes {
                                   short_name: "post_split_1"
                                   long_name: "Post Split 1"
                                   passes: "dce"
                                 }
                                 compound_passes {
                                   short_name: "post_split_2"
                                   long_name: "Post Split 2"
                                   passes: "cse"
                                 }
                                 default_pipeline: "pre_split_1"
                                 default_pipeline: "pre_split_2"
                                 default_pipeline: "split_step"
                                 default_pipeline: "post_split_1"
                                 default_pipeline: "post_split_2"
                               )pb",
                               "", &pipeline_proto));

  OptimizationPassRegistry registry = GetOptimizationRegistry();
  const std::vector<std::string> split_passes = {"resource_sharing"};

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<SplitPipelineFactory> split_pipeline_factory,
      SplitPipelineFactory::Create(registry, split_passes, pipeline_proto));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OptimizationCompoundPass> pre_pipeline,
      split_pipeline_factory->GetPipelineBeforeSplit("resource_sharing"));
  ASSERT_THAT(pre_pipeline, NotNull());
  EXPECT_THAT(GetSubpassShortNames(*pre_pipeline),
              ElementsAre("pre_split_1", "pre_split_2"));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OptimizationCompoundPass> post_pipeline,
      split_pipeline_factory->GetPipelineAfterSplit("resource_sharing"));
  ASSERT_THAT(post_pipeline, NotNull());
  EXPECT_THAT(GetSubpassShortNames(*post_pipeline),
              ElementsAre("post_split_1", "post_split_2"));
}

TEST(SplitPipelineTest, NestedCompoundPassWithSingleSplitLeaf) {
  OptimizationPipelineProto pipeline_proto;
  XLS_ASSERT_OK(ParseTextProto(R"pb(
                                 compound_passes {
                                   short_name: "pre_split"
                                   long_name: "Pre Split"
                                   passes: "dce"
                                 }
                                 compound_passes {
                                   short_name: "inner_split_step"
                                   long_name: "Inner Split Step"
                                   passes: "resource_sharing"
                                 }
                                 compound_passes {
                                   short_name: "outer_split_step"
                                   long_name: "Outer Split Step"
                                   passes: "inner_split_step"
                                 }
                                 compound_passes {
                                   short_name: "post_split"
                                   long_name: "Post Split"
                                   passes: "dce"
                                 }
                                 default_pipeline: "pre_split"
                                 default_pipeline: "outer_split_step"
                                 default_pipeline: "post_split"
                               )pb",
                               "", &pipeline_proto));

  OptimizationPassRegistry registry = GetOptimizationRegistry();
  const std::vector<std::string> split_passes = {"resource_sharing"};

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<SplitPipelineFactory> split_pipeline_factory,
      SplitPipelineFactory::Create(registry, split_passes, pipeline_proto));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OptimizationCompoundPass> pre_pipeline,
      split_pipeline_factory->GetPipelineBeforeSplit("resource_sharing"));
  ASSERT_THAT(pre_pipeline, NotNull());
  EXPECT_THAT(GetSubpassShortNames(*pre_pipeline), ElementsAre("pre_split"));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OptimizationCompoundPass> post_pipeline,
      split_pipeline_factory->GetPipelineAfterSplit("resource_sharing"));
  ASSERT_THAT(post_pipeline, NotNull());
  EXPECT_THAT(GetSubpassShortNames(*post_pipeline), ElementsAre("post_split"));
}

TEST(SplitPipelineTest, MultipleSplitsInPipeline) {
  OptimizationPipelineProto pipeline_proto;
  XLS_ASSERT_OK(ParseTextProto(R"pb(
                                 compound_passes {
                                   short_name: "pre_split"
                                   long_name: "Pre Split"
                                   passes: "dce"
                                 }
                                 compound_passes {
                                   short_name: "first_split_step"
                                   long_name: "First Split Step"
                                   passes: "collapse_select_chains"
                                 }
                                 compound_passes {
                                   short_name: "mid_split"
                                   long_name: "Mid Split"
                                   passes: "cse"
                                 }
                                 compound_passes {
                                   short_name: "second_split_step"
                                   long_name: "Second Split Step"
                                   passes: "resource_sharing"
                                 }
                                 compound_passes {
                                   short_name: "post_split"
                                   long_name: "Post Split"
                                   passes: "dce"
                                 }
                                 default_pipeline: "pre_split"
                                 default_pipeline: "first_split_step"
                                 default_pipeline: "mid_split"
                                 default_pipeline: "second_split_step"
                                 default_pipeline: "post_split"
                               )pb",
                               "", &pipeline_proto));

  OptimizationPassRegistry registry = GetOptimizationRegistry();
  const std::vector<std::string> split_passes = {"collapse_select_chains",
                                                 "resource_sharing"};
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<SplitPipelineFactory> split_pipeline_factory,
      SplitPipelineFactory::Create(registry, split_passes, pipeline_proto));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OptimizationCompoundPass> before_first,
      split_pipeline_factory->GetPipelineBeforeSplit("collapse_select_chains"));
  ASSERT_THAT(before_first, NotNull());
  EXPECT_THAT(GetSubpassShortNames(*before_first), ElementsAre("pre_split"));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OptimizationCompoundPass> after_first,
      split_pipeline_factory->GetPipelineAfterSplit("collapse_select_chains"));
  ASSERT_THAT(after_first, NotNull());
  EXPECT_THAT(GetSubpassShortNames(*after_first),
              ElementsAre("mid_split", "second_split_step", "post_split"));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OptimizationCompoundPass> before_second,
      split_pipeline_factory->GetPipelineBeforeSplit("resource_sharing"));
  ASSERT_THAT(before_second, NotNull());
  EXPECT_THAT(GetSubpassShortNames(*before_second),
              ElementsAre("pre_split", "first_split_step", "mid_split"));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OptimizationCompoundPass> after_second,
      split_pipeline_factory->GetPipelineAfterSplit("resource_sharing"));
  ASSERT_THAT(after_second, NotNull());
  EXPECT_THAT(GetSubpassShortNames(*after_second), ElementsAre("post_split"));
}

TEST(SplitPipelineTest, SplitAsFirstPassReturnsNullBeforePipeline) {
  OptimizationPipelineProto pipeline_proto;
  XLS_ASSERT_OK(ParseTextProto(R"pb(
                                 compound_passes {
                                   short_name: "split_step"
                                   long_name: "Split Step"
                                   passes: "resource_sharing"
                                 }
                                 compound_passes {
                                   short_name: "post_split"
                                   long_name: "Post Split"
                                   passes: "dce"
                                 }
                                 default_pipeline: "split_step"
                                 default_pipeline: "post_split"
                               )pb",
                               "", &pipeline_proto));

  OptimizationPassRegistry registry = GetOptimizationRegistry();
  const std::vector<std::string> split_passes = {"resource_sharing"};

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<SplitPipelineFactory> split_pipeline_factory,
      SplitPipelineFactory::Create(registry, split_passes, pipeline_proto));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OptimizationCompoundPass> post_pipeline,
      split_pipeline_factory->GetPipelineBeforeSplit("resource_sharing"));
  EXPECT_THAT(post_pipeline, NotNull());
  EXPECT_THAT(GetSubpassShortNames(*post_pipeline), IsEmpty());
}

TEST(SplitPipelineTest, SplitAsLastPassReturnsNullAfterPipeline) {
  OptimizationPipelineProto pipeline_proto;
  XLS_ASSERT_OK(ParseTextProto(R"pb(
                                 compound_passes {
                                   short_name: "pre_split"
                                   long_name: "Pre Split"
                                   passes: "dce"
                                 }
                                 compound_passes {
                                   short_name: "split_step"
                                   long_name: "Split Step"
                                   passes: "resource_sharing"
                                 }
                                 default_pipeline: "pre_split"
                                 default_pipeline: "split_step"
                               )pb",
                               "", &pipeline_proto));

  OptimizationPassRegistry registry = GetOptimizationRegistry();
  const std::vector<std::string> split_passes = {"resource_sharing"};

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<SplitPipelineFactory> split_pipeline_factory,
      SplitPipelineFactory::Create(registry, split_passes, pipeline_proto));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OptimizationCompoundPass> post_pipeline,
      split_pipeline_factory->GetPipelineAfterSplit("resource_sharing"));
  EXPECT_THAT(post_pipeline, NotNull());
  EXPECT_THAT(GetSubpassShortNames(*post_pipeline), IsEmpty());
}

TEST(SplitPipelineTest, SplitPassWithOtherPassesInCompoundPassReturnsError) {
  OptimizationPipelineProto pipeline_proto;
  XLS_ASSERT_OK(ParseTextProto(R"pb(
                                 compound_passes {
                                   short_name: "mixed_step"
                                   long_name: "Mixed Step"
                                   passes: "resource_sharing"
                                   passes: "dce"
                                 }
                                 default_pipeline: "mixed_step"
                               )pb",
                               "", &pipeline_proto));

  OptimizationPassRegistry registry = GetOptimizationRegistry();
  const std::vector<std::string> split_passes = {"resource_sharing"};

  EXPECT_THAT(
      SplitPipelineFactory::Create(registry, split_passes, pipeline_proto),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("must be the only leaf pass in compound pass")));
}

TEST(SplitPipelineTest, DuplicateSplitPassInPipelineReturnsError) {
  OptimizationPipelineProto pipeline_proto;
  XLS_ASSERT_OK(ParseTextProto(R"pb(
                                 compound_passes {
                                   short_name: "split_step_1"
                                   long_name: "Split Step 1"
                                   passes: "resource_sharing"
                                 }
                                 compound_passes {
                                   short_name: "mid_step"
                                   long_name: "Mid Step"
                                   passes: "dce"
                                 }
                                 compound_passes {
                                   short_name: "split_step_2"
                                   long_name: "Split Step 2"
                                   passes: "resource_sharing"
                                 }
                                 default_pipeline: "split_step_1"
                                 default_pipeline: "mid_step"
                                 default_pipeline: "split_step_2"
                               )pb",
                               "", &pipeline_proto));

  OptimizationPassRegistry registry = GetOptimizationRegistry();
  const std::vector<std::string> split_passes = {"resource_sharing"};

  EXPECT_THAT(
      SplitPipelineFactory::Create(registry, split_passes, pipeline_proto),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Multiple instances of split pass")));
}

TEST(SplitPipelineTest, PassNotFoundInCompoundPassesReturnsError) {
  OptimizationPipelineProto pipeline_proto;
  XLS_ASSERT_OK(ParseTextProto(R"pb(
                                 default_pipeline: "dce"
                               )pb",
                               "", &pipeline_proto));

  OptimizationPassRegistry registry = GetOptimizationRegistry();
  const std::vector<std::string> split_passes = {"resource_sharing"};

  EXPECT_THAT(
      SplitPipelineFactory::Create(registry, split_passes, pipeline_proto),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("not found in compound passes")));
}

TEST(SplitPipelineTest, RepeatedInternalCompoundPassReturnsCorrectPipeline) {
  OptimizationPipelineProto pipeline_proto;
  XLS_ASSERT_OK(ParseTextProto(R"pb(
                                 compound_passes {
                                   short_name: "shared_inner"
                                   long_name: "Shared Inner"
                                   passes: "dce"
                                 }
                                 compound_passes {
                                   short_name: "branch_1"
                                   long_name: "Branch 1"
                                   passes: "shared_inner"
                                 }
                                 compound_passes {
                                   short_name: "branch_2"
                                   long_name: "Branch 2"
                                   passes: "shared_inner"
                                 }
                                 compound_passes {
                                   short_name: "pre_split"
                                   long_name: "Pre Split"
                                   passes: "branch_1"
                                   passes: "branch_2"
                                 }
                                 compound_passes {
                                   short_name: "split_step"
                                   long_name: "Split Step"
                                   passes: "resource_sharing"
                                 }
                                 compound_passes {
                                   short_name: "post_split"
                                   long_name: "Post Split"
                                   passes: "shared_inner"
                                 }
                                 default_pipeline: "pre_split"
                                 default_pipeline: "split_step"
                                 default_pipeline: "post_split"
                               )pb",
                               "", &pipeline_proto));

  OptimizationPassRegistry registry = GetOptimizationRegistry();
  const std::vector<std::string> split_passes = {"resource_sharing"};

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<SplitPipelineFactory> split_pipeline_factory,
      SplitPipelineFactory::Create(registry, split_passes, pipeline_proto));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OptimizationCompoundPass> pre_pipeline,
      split_pipeline_factory->GetPipelineBeforeSplit("resource_sharing"));
  ASSERT_THAT(pre_pipeline, NotNull());
  EXPECT_THAT(GetSubpassShortNames(*pre_pipeline), ElementsAre("pre_split"));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OptimizationCompoundPass> post_pipeline,
      split_pipeline_factory->GetPipelineAfterSplit("resource_sharing"));
  ASSERT_THAT(post_pipeline, NotNull());
  EXPECT_THAT(GetSubpassShortNames(*post_pipeline), ElementsAre("post_split"));
}

TEST(SplitPipelineTest, CycleInCompoundPassesReturnsError) {
  OptimizationPipelineProto pipeline_proto;
  XLS_ASSERT_OK(ParseTextProto(R"pb(
                                 compound_passes {
                                   short_name: "pass_a"
                                   long_name: "Pass A"
                                   passes: "pass_b"
                                 }
                                 compound_passes {
                                   short_name: "pass_b"
                                   long_name: "Pass B"
                                   passes: "pass_a"
                                 }
                                 default_pipeline: "pass_a"
                               )pb",
                               "", &pipeline_proto));

  OptimizationPassRegistry registry = GetOptimizationRegistry();
  const std::vector<std::string> split_passes = {"resource_sharing"};

  EXPECT_THAT(
      SplitPipelineFactory::Create(registry, split_passes, pipeline_proto),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Cycle detected involving pass")));
}

}  // namespace
}  // namespace xls
