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

#include "xls/passes/optimization_pass_extension_flag.h"

#include <cstdint>
#include <optional>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/flags/flag.h"
#include "absl/flags/reflection.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xls/common/status/matchers.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/test_utils/test_opt_pass_ext_flags.pb.h"
#include "xls/tools/opt_flags.pb.h"
#include "xls/tools/opt_flags_handler_registry.h"

ABSL_FLAG(std::string, test_ext_str_flag, "default_str",
          "Test string extension flag.");
ABSL_FLAG(int64_t, test_ext_int_flag, 42, "Test int64_t extension flag.");
ABSL_FLAG(bool, test_unregistered_ext_flag, false,
          "Test unregistered extension flag.");

namespace xls {
namespace {

using ::absl_testing::StatusIs;
using ::testing::Eq;
using ::testing::Optional;

XLS_REGISTER_OPT_PASS_EXT_FLAG(FLAGS_test_ext_str_flag,
                               ext_flags_test::test_ext_str_flag);
XLS_REGISTER_OPT_PASS_EXT_FLAG(FLAGS_test_ext_int_flag,
                               ext_flags_test::test_ext_int_flag);

TEST(OptimizationPassExtensionFlagTest, UnconfiguredOptionsReturnsNullopt) {
  OptimizationPassOptions options;
  EXPECT_EQ(GetConfiguredExtensionFlag(options, FLAGS_test_ext_str_flag),
            std::nullopt);
  EXPECT_EQ(GetConfiguredExtensionFlag(options, FLAGS_test_ext_int_flag),
            std::nullopt);
}

TEST(OptimizationPassExtensionFlagTest, UnregisteredFlagReturnsNullopt) {
  OptFlagsProto proto;
  XLS_ASSERT_OK(OptFlagsHandlerRegistry::ParseFlags(proto));
  OptimizationPassOptions options;
  XLS_ASSERT_OK(OptFlagsHandlerRegistry::Process(proto, options));

  EXPECT_EQ(
      GetConfiguredExtensionFlag(options, FLAGS_test_unregistered_ext_flag),
      std::nullopt);
}

TEST(OptimizationPassExtensionFlagTest, DefaultFlagValues) {
  OptFlagsProto proto;
  XLS_ASSERT_OK(OptFlagsHandlerRegistry::ParseFlags(proto));
  EXPECT_TRUE(proto.HasExtension(ext_flags_test::test_ext_str_flag));
  EXPECT_EQ(proto.GetExtension(ext_flags_test::test_ext_str_flag),
            "default_str");
  EXPECT_TRUE(proto.HasExtension(ext_flags_test::test_ext_int_flag));
  EXPECT_EQ(proto.GetExtension(ext_flags_test::test_ext_int_flag), 42);

  OptimizationPassOptions options;
  XLS_ASSERT_OK(OptFlagsHandlerRegistry::Process(proto, options));
  EXPECT_THAT(GetConfiguredExtensionFlag(options, FLAGS_test_ext_str_flag),
              Optional(Eq("default_str")));
  EXPECT_THAT(GetConfiguredExtensionFlag(options, FLAGS_test_ext_int_flag),
              Optional(Eq(42)));
}

TEST(OptimizationPassExtensionFlagTest, CustomFlagValues) {
  absl::FlagSaver flag_saver;
  absl::SetFlag(&FLAGS_test_ext_str_flag, "custom_value");
  absl::SetFlag(&FLAGS_test_ext_int_flag, 1337);

  OptFlagsProto proto;
  XLS_ASSERT_OK(OptFlagsHandlerRegistry::ParseFlags(proto));
  EXPECT_EQ(proto.GetExtension(ext_flags_test::test_ext_str_flag),
            "custom_value");
  EXPECT_EQ(proto.GetExtension(ext_flags_test::test_ext_int_flag), 1337);

  OptimizationPassOptions options;
  XLS_ASSERT_OK(OptFlagsHandlerRegistry::Process(proto, options));
  EXPECT_THAT(GetConfiguredExtensionFlag(options, FLAGS_test_ext_str_flag),
              Optional(Eq("custom_value")));
  EXPECT_THAT(GetConfiguredExtensionFlag(options, FLAGS_test_ext_int_flag),
              Optional(Eq(1337)));
}

TEST(OptimizationPassExtensionFlagTest, ProtoExtensionSetDirectly) {
  OptFlagsProto proto;
  proto.SetExtension(ext_flags_test::test_ext_str_flag, "from_proto");
  proto.SetExtension(ext_flags_test::test_ext_int_flag, 999);

  OptimizationPassOptions options;
  XLS_ASSERT_OK(OptFlagsHandlerRegistry::Process(proto, options));
  EXPECT_THAT(GetConfiguredExtensionFlag(options, FLAGS_test_ext_str_flag),
              Optional(Eq("from_proto")));
  EXPECT_THAT(GetConfiguredExtensionFlag(options, FLAGS_test_ext_int_flag),
              Optional(Eq(999)));
}

TEST(OptimizationPassExtensionFlagTest, DuplicateProcessReturnsError) {
  OptFlagsProto proto;
  XLS_ASSERT_OK(OptFlagsHandlerRegistry::ParseFlags(proto));

  OptimizationPassOptions options;
  XLS_ASSERT_OK(OptFlagsHandlerRegistry::Process(proto, options));
  EXPECT_THAT(OptFlagsHandlerRegistry::Process(proto, options),
              StatusIs(absl::StatusCode::kInternal));
}

}  // namespace
}  // namespace xls
