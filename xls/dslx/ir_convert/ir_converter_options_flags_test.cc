// Copyright 2025 The XLS Authors
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

#include "xls/dslx/ir_convert/ir_converter_options_flags.h"

#include <optional>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/cleanup/cleanup.h"
#include "absl/flags/commandlineflag.h"
#include "absl/flags/declare.h"
#include "absl/flags/flag.h"
#include "absl/flags/reflection.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/ir_convert/convert_options.h"
#include "xls/dslx/ir_convert/ir_converter_options_flags.pb.h"

ABSL_DECLARE_FLAG(std::optional<std::string>, enable_warnings);
ABSL_DECLARE_FLAG(std::optional<std::string>, disable_warnings);
ABSL_DECLARE_FLAG(bool, warnings_as_errors);
ABSL_DECLARE_FLAG(bool, force_implicit_token_calling_convention);

namespace xls {

TEST(IrConverterOptionsFlagsTest, WarningOptionsNoFlagSettings) {
  XLS_ASSERT_OK_AND_ASSIGN(IrConverterOptionsFlagsProto options,
                           GetIrConverterOptionsFlagsProto());
  EXPECT_TRUE(options.enable_warnings().empty());
  EXPECT_TRUE(options.disable_warnings().empty());
  EXPECT_TRUE(options.has_warnings_as_errors());
  EXPECT_EQ(options.warnings_as_errors(), true);
  EXPECT_TRUE(options.has_force_implicit_token_calling_convention());
  EXPECT_FALSE(options.force_implicit_token_calling_convention());
}

TEST(IrConverterOptionsFlagsTest,
     WarningOptionsBothEnableAndDisableSetsMultiEntry) {
  // Validate the default value we're going to reset this flag to.
  ASSERT_EQ(absl::GetFlag(FLAGS_warnings_as_errors), true);
  ASSERT_EQ(absl::GetFlag(FLAGS_enable_warnings), std::nullopt);
  ASSERT_EQ(absl::GetFlag(FLAGS_disable_warnings), std::nullopt);

  absl::SetFlag(&FLAGS_enable_warnings,
                "already_exhaustive_match,should_use_assert");
  absl::SetFlag(&FLAGS_disable_warnings, "constant_naming,member_naming");
  absl::SetFlag(&FLAGS_warnings_as_errors, false);
  absl::Cleanup reset_flags([] {
    absl::SetFlag(&FLAGS_enable_warnings, std::nullopt);
    absl::SetFlag(&FLAGS_disable_warnings, std::nullopt);
    absl::SetFlag(&FLAGS_warnings_as_errors, true);
  });
  XLS_ASSERT_OK_AND_ASSIGN(IrConverterOptionsFlagsProto options,
                           GetIrConverterOptionsFlagsProto());
  EXPECT_TRUE(options.has_warnings_as_errors());
  EXPECT_TRUE(options.has_enable_warnings());
  EXPECT_TRUE(options.has_disable_warnings());
  EXPECT_EQ(options.enable_warnings(),
            "already_exhaustive_match,should_use_assert");
  EXPECT_EQ(options.disable_warnings(), "constant_naming,member_naming");
  EXPECT_EQ(options.warnings_as_errors(), false);
}

TEST(IrConverterOptionsFlagsTest, ForceImplicitTokenFlagSet) {
  ASSERT_FALSE(absl::GetFlag(FLAGS_force_implicit_token_calling_convention));
  absl::SetFlag(&FLAGS_force_implicit_token_calling_convention, true);
  absl::Cleanup reset_flag([] {
    absl::SetFlag(&FLAGS_force_implicit_token_calling_convention, false);
  });

  XLS_ASSERT_OK_AND_ASSIGN(IrConverterOptionsFlagsProto options,
                           GetIrConverterOptionsFlagsProto());
  EXPECT_TRUE(options.has_force_implicit_token_calling_convention());
  EXPECT_TRUE(options.force_implicit_token_calling_convention());
}

TEST(IrConverterOptionsFlagsTest, LowerToProcScopedChannelsDefaultIsTrue) {
  absl::CommandLineFlag* flag =
      absl::FindCommandLineFlag("lower_to_proc_scoped_channels");
  ASSERT_NE(flag, nullptr);
  EXPECT_EQ(flag->DefaultValue(), "true");
}

TEST(IrConverterOptionsFlagsTest, ConfiguredValuesHelpUsesKeyColonValueFormat) {
  absl::CommandLineFlag* flag = absl::FindCommandLineFlag("configured_values");
  ASSERT_NE(flag, nullptr);
  EXPECT_THAT(flag->Help(), testing::HasSubstr("key:value"));
  EXPECT_THAT(flag->Help(), testing::Not(testing::HasSubstr("{\"")));
}

TEST(IrConverterOptionsFlagsTest,
     ProtoDefaultInstanceMatchesFlagsAndConvertOptions) {
  const IrConverterOptionsFlagsProto& defaults =
      IrConverterOptionsFlagsProto::default_instance();
  dslx::ConvertOptions convert_defaults;

  EXPECT_TRUE(defaults.verify());
  EXPECT_TRUE(defaults.warnings_as_errors());
  EXPECT_TRUE(defaults.lower_to_proc_scoped_channels());
  EXPECT_TRUE(defaults.emit_trace());
  EXPECT_TRUE(defaults.emit_cover());
  EXPECT_TRUE(defaults.emit_assert());
  EXPECT_FALSE(defaults.convert_tests());
  EXPECT_FALSE(defaults.type_inference_v2());
  EXPECT_FALSE(defaults.force_implicit_token_calling_convention());

  EXPECT_EQ(defaults.verify(), convert_defaults.verify_ir);
  EXPECT_EQ(defaults.warnings_as_errors(), convert_defaults.warnings_as_errors);
  EXPECT_EQ(defaults.lower_to_proc_scoped_channels(),
            convert_defaults.lower_to_proc_scoped_channels);
  EXPECT_EQ(defaults.emit_trace(), convert_defaults.emit_trace);
  EXPECT_EQ(defaults.emit_cover(), convert_defaults.emit_cover);
  EXPECT_EQ(defaults.emit_assert(), convert_defaults.emit_assert);
  EXPECT_EQ(defaults.convert_tests(), convert_defaults.convert_tests);
  EXPECT_EQ(defaults.type_inference_v2(), convert_defaults.type_inference_v2);
  EXPECT_EQ(defaults.force_implicit_token_calling_convention(),
            convert_defaults.force_implicit_token_calling_convention);

  XLS_ASSERT_OK_AND_ASSIGN(IrConverterOptionsFlagsProto from_flags,
                           GetIrConverterOptionsFlagsProto());
  EXPECT_EQ(from_flags.verify(), defaults.verify());
  EXPECT_EQ(from_flags.warnings_as_errors(), defaults.warnings_as_errors());
  EXPECT_EQ(from_flags.lower_to_proc_scoped_channels(),
            defaults.lower_to_proc_scoped_channels());
  EXPECT_EQ(from_flags.emit_trace(), defaults.emit_trace());
  EXPECT_EQ(from_flags.emit_cover(), defaults.emit_cover());
  EXPECT_EQ(from_flags.emit_assert(), defaults.emit_assert());
  EXPECT_EQ(from_flags.convert_tests(), defaults.convert_tests());
  EXPECT_EQ(from_flags.type_inference_v2(), defaults.type_inference_v2());
  EXPECT_EQ(from_flags.force_implicit_token_calling_convention(),
            defaults.force_implicit_token_calling_convention());
}

}  // namespace xls
