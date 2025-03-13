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

#include <string>
#include <optional>
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/cleanup/cleanup.h"
#include "absl/flags/flag.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/warning_kind.h"

ABSL_DECLARE_FLAG(std::optional<std::string>, enable_warnings);
ABSL_DECLARE_FLAG(std::optional<std::string>, disable_warnings);
ABSL_DECLARE_FLAG(bool, warnings_as_errors);

namespace xls {

TEST(IrConverterOptionsFlagsTest, WarningOptionsNoFlagSettings) {
  XLS_ASSERT_OK_AND_ASSIGN(IrConverterOptionsFlagsProto options,
                           GetIrConverterOptionsFlagsProto());
  EXPECT_TRUE(options.enable_warnings().empty());
  EXPECT_TRUE(options.disable_warnings().empty());
  EXPECT_TRUE(options.has_warnings_as_errors());
  EXPECT_EQ(options.warnings_as_errors(), true);
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

}  // namespace xls
