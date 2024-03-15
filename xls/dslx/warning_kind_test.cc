// Copyright 2023 The XLS Authors
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

#include "xls/dslx/warning_kind.h"

#include <string_view>

#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"

namespace xls::dslx {
namespace {

TEST(WarningKindTest, WarningKindStringRoundTrip) {
  for (WarningKind kind : kAllWarningKinds) {
    XLS_ASSERT_OK_AND_ASSIGN(std::string_view s, WarningKindToString(kind));
    XLS_ASSERT_OK_AND_ASSIGN(WarningKind got, WarningKindFromString(s));
    EXPECT_EQ(got, kind);
  }
}

TEST(WarningKindTest, AllWarningsPresentInAllWarningSet) {
  for (const WarningKind kind : kAllWarningKinds) {
    // Check every warning kind is enabled in the all-warnings set.
    EXPECT_TRUE(WarningIsEnabled(kAllWarningsSet, kind));

    // After we disable it, it should not be enabled.
    EXPECT_FALSE(WarningIsEnabled(DisableWarning(kAllWarningsSet, kind), kind));
  }
}

TEST(WarningKindTest, InitializerListHasAllKinds) {
  for (WarningKindInt i = 0; i < kWarningKindCount; ++i) {
    EXPECT_EQ(kAllWarningKinds[i],
              WarningKind{static_cast<WarningKindInt>(1 << i)});
  }
  EXPECT_EQ(kWarningKindCount, kAllWarningKinds.size());
}

TEST(WarningKindTest, DefaultSetAnyMissing) {
  ASSERT_TRUE(WarningIsEnabled(kAllWarningsSet, WarningKind::kShouldUseAssert));

  // Currently "should use assert" warning is disabled for propagation delay.
  ASSERT_FALSE(
      WarningIsEnabled(kDefaultWarningsSet, WarningKind::kShouldUseAssert));
}

TEST(WarningKindTest, WarningKindSetFromString) {
  XLS_ASSERT_OK_AND_ASSIGN(WarningKindSet set,
                           WarningKindSetFromString("should_use_assert"));
  ASSERT_TRUE(WarningIsEnabled(set, WarningKind::kShouldUseAssert));
}

}  // namespace
}  // namespace xls::dslx
