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

#include <cstdint>
#include <memory>

#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/examples/configured_value_or_jit_wrapper.h"
#include "xls/ir/value.h"

namespace xls {
namespace examples {
namespace {

// Helper to get integer value from xls::Value
int64_t GetIntValue(const xls::Value& v) { return v.bits().ToInt64().value(); }

uint64_t GetUintValue(const xls::Value& v) {
  return v.bits().ToUint64().value();
}

// Enum for MyEnum
constexpr int64_t kMyEnumB = 1;
constexpr int64_t kMyEnumC = 2;

TEST(ConfiguredValueOrJitTest, TestOverrides) {
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ConfiguredValueOr> configured_value_or,
      ConfiguredValueOr::Create());
  XLS_ASSERT_OK_AND_ASSIGN(xls::Value result, configured_value_or->Run());

  ASSERT_TRUE(result.IsTuple());
  ASSERT_EQ(result.size(), 8);

  // Expected values from BUILD file overrides or defaults
  EXPECT_EQ(GetUintValue(result.element(0)), false);     // b_default
  EXPECT_EQ(GetUintValue(result.element(1)), 42);        // u32_default
  EXPECT_EQ(GetIntValue(result.element(2)), -100);       // s32_default
  EXPECT_EQ(GetUintValue(result.element(3)), kMyEnumC);  // enum_default

  EXPECT_EQ(GetUintValue(result.element(4)), true);      // b_override
  EXPECT_EQ(GetUintValue(result.element(5)), 123);       // u32_override
  EXPECT_EQ(GetIntValue(result.element(6)), -200);       // s32_override
  EXPECT_EQ(GetUintValue(result.element(7)), kMyEnumB);  // enum_override
}

}  // namespace
}  // namespace examples
}  // namespace xls
