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

#include "xls/tools/codegen_flags_handler_registry.h"

#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/cleanup/cleanup.h"
#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xls/codegen/codegen_options.h"
#include "xls/common/status/matchers.h"
#include "xls/tools/codegen.h"
#include "xls/tools/codegen_flags.h"
#include "xls/tools/codegen_flags.pb.h"
#include "xls/tools/test_utils/test_codegen_flag_handler.h"
#include "xls/tools/test_utils/test_codegen_flags.pb.h"

namespace xls {
namespace {

using ::absl_testing::StatusIs;

TEST(CodegenFlagsHandlerRegistryTest, DefaultFlagValue) {
  CodegenFlagsProto proto;
  XLS_ASSERT_OK(CodegenFlagsHandlerRegistry::ParseFlags(proto));
  EXPECT_FALSE(proto.HasExtension(test_codegen_flag));

  verilog::CodegenOptions options;
  XLS_ASSERT_OK(CodegenFlagsHandlerRegistry::Process(proto, options));
  EXPECT_EQ(options.get_extension<TestCodegenOptionExtension>(), nullptr);
}

TEST(CodegenFlagsHandlerRegistryTest, CustomFlagValue) {
  std::string original = absl::GetFlag(FLAGS_test_codegen_flag);
  absl::SetFlag(&FLAGS_test_codegen_flag, "custom_value");
  absl::Cleanup cleanup = [&]() {
    absl::SetFlag(&FLAGS_test_codegen_flag, original);
  };

  CodegenFlagsProto proto;
  XLS_ASSERT_OK(CodegenFlagsHandlerRegistry::ParseFlags(proto));
  EXPECT_TRUE(proto.HasExtension(test_codegen_flag));
  EXPECT_EQ(proto.GetExtension(test_codegen_flag), "custom_value");

  verilog::CodegenOptions options;
  XLS_ASSERT_OK(CodegenFlagsHandlerRegistry::Process(proto, options));
  const auto* ext = options.get_extension<TestCodegenOptionExtension>();
  ASSERT_NE(ext, nullptr);
  EXPECT_EQ(ext->value, "custom_value");
}

TEST(CodegenFlagsHandlerRegistryTest, EndToEndViaCodegenFlagsAndOptions) {
  std::string original = absl::GetFlag(FLAGS_test_codegen_flag);
  absl::SetFlag(&FLAGS_test_codegen_flag, "end_to_end_value");
  absl::Cleanup cleanup = [&]() {
    absl::SetFlag(&FLAGS_test_codegen_flag, original);
  };

  XLS_ASSERT_OK_AND_ASSIGN(CodegenFlagsProto proto, GetCodegenFlags());
  XLS_ASSERT_OK_AND_ASSIGN(verilog::CodegenOptions options,
                           CodegenOptionsFromProto(proto));
  const auto* ext = options.get_extension<TestCodegenOptionExtension>();
  ASSERT_NE(ext, nullptr);
  EXPECT_EQ(ext->value, "end_to_end_value");
}

TEST(CodegenFlagsHandlerRegistryTest, ParseFlagsError) {
  std::string original = absl::GetFlag(FLAGS_test_codegen_flag);
  absl::SetFlag(&FLAGS_test_codegen_flag, "parse_error");
  absl::Cleanup cleanup = [&]() {
    absl::SetFlag(&FLAGS_test_codegen_flag, original);
  };

  CodegenFlagsProto proto;
  EXPECT_THAT(CodegenFlagsHandlerRegistry::ParseFlags(proto),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CodegenFlagsHandlerRegistryTest, ProcessError) {
  CodegenFlagsProto proto;
  proto.SetExtension(test_codegen_flag, "invalid");

  verilog::CodegenOptions options;
  EXPECT_THAT(CodegenFlagsHandlerRegistry::Process(proto, options),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace xls
