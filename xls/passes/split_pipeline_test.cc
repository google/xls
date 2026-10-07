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

std::vector<std::string> GetSubpassShortNames(const OptimizationPass& pass) {
  if (!pass.IsCompound()) {
    return {std::string(pass.base_short_name())};
  }
  std::vector<std::string> names;
  const OptimizationCompoundPass* compound_pass =
      dynamic_cast<const OptimizationCompoundPass*>(&pass);
  if (compound_pass == nullptr) {
    return names;
  }
  for (const OptimizationPass* subpass : compound_pass->passes()) {
    std::vector<std::string> subpass_names = GetSubpassShortNames(*subpass);
    names.insert(names.end(), subpass_names.begin(), subpass_names.end());
  }
  return names;
}

TEST(SplitPipelineTest, SplitMultiplePasses) {
  OptimizationPipelineProto pipeline_proto;
  XLS_ASSERT_OK(ParseTextProto(R"pb(
                                 compound_passes {
                                   short_name: "pre_split_1"
                                   long_name: "Pre Split 1"
                                   passes: "cse"
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
                                   passes: "bitslice_simp"
                                 }
                                 compound_passes {
                                   short_name: "post_split_2"
                                   long_name: "Post Split 2"
                                   passes: "dce"
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
  EXPECT_THAT(GetSubpassShortNames(*pre_pipeline), ElementsAre("cse", "dce"));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OptimizationCompoundPass> post_pipeline,
      split_pipeline_factory->GetPipelineAfterSplit("resource_sharing"));
  ASSERT_THAT(post_pipeline, NotNull());
  EXPECT_THAT(GetSubpassShortNames(*post_pipeline),
              ElementsAre("bitslice_simp", "dce"));
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
                                   passes: "cse"
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
  EXPECT_THAT(GetSubpassShortNames(*pre_pipeline), ElementsAre("dce"));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OptimizationCompoundPass> post_pipeline,
      split_pipeline_factory->GetPipelineAfterSplit("resource_sharing"));
  ASSERT_THAT(post_pipeline, NotNull());
  EXPECT_THAT(GetSubpassShortNames(*post_pipeline), ElementsAre("cse"));
}

TEST(SplitPipelineTest, MultipleSplitsInPipeline) {
  OptimizationPipelineProto pipeline_proto;
  XLS_ASSERT_OK(ParseTextProto(R"pb(
                                 compound_passes {
                                   short_name: "pre_split"
                                   long_name: "Pre Split"
                                   passes: "bitslice_simp"
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
  EXPECT_THAT(GetSubpassShortNames(*before_first),
              ElementsAre("bitslice_simp"));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OptimizationCompoundPass> after_first,
      split_pipeline_factory->GetPipelineAfterSplit("collapse_select_chains"));
  ASSERT_THAT(after_first, NotNull());
  EXPECT_THAT(GetSubpassShortNames(*after_first),
              ElementsAre("cse", "resource_sharing", "dce"));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OptimizationCompoundPass> before_second,
      split_pipeline_factory->GetPipelineBeforeSplit("resource_sharing"));
  ASSERT_THAT(before_second, NotNull());
  EXPECT_THAT(GetSubpassShortNames(*before_second),
              ElementsAre("bitslice_simp", "collapse_select_chains", "cse"));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OptimizationCompoundPass> after_second,
      split_pipeline_factory->GetPipelineAfterSplit("resource_sharing"));
  ASSERT_THAT(after_second, NotNull());
  EXPECT_THAT(GetSubpassShortNames(*after_second), ElementsAre("dce"));
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

TEST(SplitPipelineTest, SplitPassWithinCompoundPass) {
  OptimizationPipelineProto pipeline_proto;
  XLS_ASSERT_OK(ParseTextProto(R"pb(
                                 compound_passes {
                                   short_name: "mixed_step"
                                   long_name: "Mixed Step"
                                   passes: "dce"
                                   passes: "resource_sharing"
                                   passes: "cse"
                                   passes: "bitslice_simp"
                                 }
                                 default_pipeline: "mixed_step"
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
  EXPECT_THAT(GetSubpassShortNames(*pre_pipeline), ElementsAre("dce"));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OptimizationCompoundPass> post_pipeline,
      split_pipeline_factory->GetPipelineAfterSplit("resource_sharing"));
  ASSERT_THAT(post_pipeline, NotNull());
  EXPECT_THAT(GetSubpassShortNames(*post_pipeline),
              ElementsAre("cse", "bitslice_simp"));
}

TEST(SplitPipelineTest, SplitPassInFixedpointCompoundPassReturnsError) {
  OptimizationPipelineProto pipeline_proto;
  XLS_ASSERT_OK(ParseTextProto(R"pb(
                                 compound_passes {
                                   short_name: "fixedpoint_step"
                                   long_name: "Fixedpoint Step"
                                   fixedpoint: true
                                   passes: "resource_sharing"
                                   passes: "dce"
                                 }
                                 default_pipeline: "fixedpoint_step"
                               )pb",
                               "", &pipeline_proto));

  OptimizationPassRegistry registry = GetOptimizationRegistry();
  const std::vector<std::string> split_passes = {"resource_sharing"};

  EXPECT_THAT(
      SplitPipelineFactory::Create(registry, split_passes, pipeline_proto),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Requested split passes not found as valid leaf passes")));
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

TEST(SplitPipelineTest, SplitPassNotFoundInPipelineReturnsError) {
  OptimizationPipelineProto pipeline_proto;
  XLS_ASSERT_OK(ParseTextProto(R"pb(
                                 default_pipeline: "dce"
                               )pb",
                               "", &pipeline_proto));

  OptimizationPassRegistry registry = GetOptimizationRegistry();
  const std::vector<std::string> split_passes = {"resource_sharing"};

  EXPECT_THAT(
      SplitPipelineFactory::Create(registry, split_passes, pipeline_proto),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Requested split passes not found as valid leaf passes")));
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
  EXPECT_THAT(GetSubpassShortNames(*pre_pipeline), ElementsAre("dce", "dce"));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OptimizationCompoundPass> post_pipeline,
      split_pipeline_factory->GetPipelineAfterSplit("resource_sharing"));
  ASSERT_THAT(post_pipeline, NotNull());
  EXPECT_THAT(GetSubpassShortNames(*post_pipeline), ElementsAre("dce"));
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
