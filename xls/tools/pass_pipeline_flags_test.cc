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

#include "xls/tools/pass_pipeline_flags.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/flags/declare.h"
#include "absl/flags/flag.h"
#include "absl/flags/reflection.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/temp_directory.h"
#include "xls/common/status/matchers.h"
#include "xls/passes/optimization_pass_pipeline.pb.h"
#include "xls/passes/pass_pipeline.pb.h"
#include "xls/tools/pass_pipeline_config.h"

ABSL_DECLARE_FLAG(std::vector<std::string>, skip_passes);
ABSL_DECLARE_FLAG(std::optional<std::string>, passes);
ABSL_DECLARE_FLAG(std::optional<std::string>, passes_proto);
ABSL_DECLARE_FLAG(std::optional<std::string>, passes_textproto);
ABSL_DECLARE_FLAG(std::optional<std::string>, pipeline_proto);
ABSL_DECLARE_FLAG(std::optional<std::string>, pipeline_textproto);
ABSL_DECLARE_FLAG(std::optional<int64_t>, passes_bisect_limit);
ABSL_DECLARE_FLAG(bool, passes_bisect_limit_is_error);
ABSL_DECLARE_FLAG(bool, debug_optimizations);

namespace xls {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;

class PassPipelineFlagsTest : public ::testing::Test {
 private:
  absl::FlagSaver flag_saver_;
};

TEST_F(PassPipelineFlagsTest, DefaultsReturnEmptyConfigAndUnset) {
  XLS_ASSERT_OK_AND_ASSIGN(PassPipelineFlagsResult result,
                           GetPassPipelineFlags());
  EXPECT_FALSE(result.any_flags_set);
  EXPECT_FALSE(result.config.HasExplicitPipeline());
  EXPECT_FALSE(result.config.custom_registry.has_value());
  EXPECT_FALSE(result.config.pipeline.has_value());
  EXPECT_THAT(result.config.skip_passes, IsEmpty());
  EXPECT_EQ(result.config.passes_bisect_limit, std::nullopt);
  EXPECT_FALSE(result.config.passes_bisect_limit_is_error);
  EXPECT_FALSE(result.config.debug_optimizations);
}

TEST_F(PassPipelineFlagsTest, ParsesPassListFlag) {
  absl::SetFlag(&FLAGS_passes, "basic_simp dce");
  XLS_ASSERT_OK_AND_ASSIGN(PassPipelineFlagsResult result,
                           GetPassPipelineFlags());
  EXPECT_TRUE(result.any_flags_set);
  ASSERT_TRUE(result.config.pipeline.has_value());
  EXPECT_EQ(result.config.pipeline->top().pipeline().elements_size(), 2);
}

TEST_F(PassPipelineFlagsTest, DistinguishesEmptyPassesFromUnset) {
  absl::SetFlag(&FLAGS_passes, "");
  XLS_ASSERT_OK_AND_ASSIGN(PassPipelineFlagsResult result,
                           GetPassPipelineFlags());
  EXPECT_TRUE(result.any_flags_set);
  ASSERT_TRUE(result.config.pipeline.has_value());
  EXPECT_THAT(result.config.pipeline->top().pipeline().elements(), IsEmpty());
}

TEST_F(PassPipelineFlagsTest, ResolvesPassesAgainstCustomPipelineTextproto) {
  XLS_ASSERT_OK_AND_ASSIGN(TempDirectory dir, TempDirectory::Create());
  OptimizationPipelineProto reg_proto;
  auto* compound = reg_proto.add_compound_passes();
  compound->set_short_name("my_group");
  compound->set_long_name("my group");
  compound->add_passes("dce");
  reg_proto.add_default_pipeline("my_group");
  std::filesystem::path textproto_path = dir.path() / "pipeline.txtpb";
  XLS_ASSERT_OK(SetTextProtoFile(textproto_path, reg_proto));

  absl::SetFlag(&FLAGS_pipeline_textproto, textproto_path.string());
  absl::SetFlag(&FLAGS_passes, "my_group");
  XLS_ASSERT_OK_AND_ASSIGN(PassPipelineFlagsResult result,
                           GetPassPipelineFlags());
  EXPECT_TRUE(result.any_flags_set);
  EXPECT_TRUE(result.config.custom_registry.has_value());
  EXPECT_TRUE(result.config.pipeline.has_value());
}

TEST_F(PassPipelineFlagsTest, LoadsBinaryPipelineAndPassesProtos) {
  XLS_ASSERT_OK_AND_ASSIGN(TempDirectory dir, TempDirectory::Create());

  OptimizationPipelineProto reg_proto;
  reg_proto.add_default_pipeline("dce");
  std::filesystem::path reg_binpb = dir.path() / "reg.binarypb";
  XLS_ASSERT_OK(SetFileContents(reg_binpb, reg_proto.SerializeAsString()));

  PassPipelineProto pass_proto;
  pass_proto.mutable_top()->set_pass_name("dce");
  std::filesystem::path pass_binpb = dir.path() / "pass.binarypb";
  XLS_ASSERT_OK(SetFileContents(pass_binpb, pass_proto.SerializeAsString()));

  absl::SetFlag(&FLAGS_pipeline_proto, reg_binpb.string());
  absl::SetFlag(&FLAGS_passes_proto, pass_binpb.string());
  XLS_ASSERT_OK_AND_ASSIGN(PassPipelineFlagsResult result,
                           GetPassPipelineFlags());
  EXPECT_TRUE(result.any_flags_set);
  ASSERT_TRUE(result.config.custom_registry.has_value());
  ASSERT_TRUE(result.config.pipeline.has_value());
  EXPECT_EQ(result.config.pipeline->top().pass_name(), "dce");
}

TEST_F(PassPipelineFlagsTest, LoadsPassesTextproto) {
  XLS_ASSERT_OK_AND_ASSIGN(TempDirectory dir, TempDirectory::Create());
  PassPipelineProto pass_proto;
  pass_proto.mutable_top()->set_pass_name("const_fold");
  std::filesystem::path pass_txtpb = dir.path() / "pass.txtpb";
  XLS_ASSERT_OK(SetTextProtoFile(pass_txtpb, pass_proto));

  absl::SetFlag(&FLAGS_passes_textproto, pass_txtpb.string());
  XLS_ASSERT_OK_AND_ASSIGN(PassPipelineFlagsResult result,
                           GetPassPipelineFlags());
  EXPECT_TRUE(result.any_flags_set);
  ASSERT_TRUE(result.config.pipeline.has_value());
  EXPECT_EQ(result.config.pipeline->top().pass_name(), "const_fold");
}

TEST_F(PassPipelineFlagsTest, RejectsConflictingPipelineRegistryFlags) {
  absl::SetFlag(&FLAGS_pipeline_proto, "a.binarypb");
  absl::SetFlag(&FLAGS_pipeline_textproto, "b.txtpb");
  EXPECT_THAT(GetPassPipelineFlags(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("--pipeline_proto and --pipeline_textproto")));
}

TEST_F(PassPipelineFlagsTest, RejectsPassesWithPassesProto) {
  absl::SetFlag(&FLAGS_passes, "dce");
  absl::SetFlag(&FLAGS_passes_proto, "a.binarypb");
  EXPECT_THAT(GetPassPipelineFlags(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(PassPipelineFlagsTest, RejectsPassesWithPassesTextproto) {
  absl::SetFlag(&FLAGS_passes, "dce");
  absl::SetFlag(&FLAGS_passes_textproto, "a.txtpb");
  EXPECT_THAT(GetPassPipelineFlags(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(PassPipelineFlagsTest, RejectsPassesProtoWithPassesTextproto) {
  absl::SetFlag(&FLAGS_passes_proto, "a.binarypb");
  absl::SetFlag(&FLAGS_passes_textproto, "b.txtpb");
  EXPECT_THAT(GetPassPipelineFlags(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(PassPipelineFlagsTest, RejectsSkipPassesWithExplicitPasses) {
  absl::SetFlag(&FLAGS_skip_passes, std::vector<std::string>{"dfe"});
  absl::SetFlag(&FLAGS_passes, "dce");
  EXPECT_THAT(
      GetPassPipelineFlags(),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("--skip_passes")));
}

TEST_F(PassPipelineFlagsTest, ReadsModifierFlags) {
  absl::SetFlag(&FLAGS_skip_passes, std::vector<std::string>{"dfe", "cse"});
  absl::SetFlag(&FLAGS_passes_bisect_limit, 42);
  absl::SetFlag(&FLAGS_passes_bisect_limit_is_error, true);
  absl::SetFlag(&FLAGS_debug_optimizations, true);

  XLS_ASSERT_OK_AND_ASSIGN(PassPipelineFlagsResult result,
                           GetPassPipelineFlags());
  EXPECT_TRUE(result.any_flags_set);
  EXPECT_THAT(result.config.skip_passes, ElementsAre("dfe", "cse"));
  EXPECT_EQ(result.config.passes_bisect_limit, 42);
  EXPECT_TRUE(result.config.passes_bisect_limit_is_error);
  EXPECT_TRUE(result.config.debug_optimizations);
}

}  // namespace
}  // namespace xls
