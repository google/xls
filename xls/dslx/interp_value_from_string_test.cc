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

#include "xls/dslx/interp_value_from_string.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "xls/dslx/default_dslx_stdlib_path.h"
#include "xls/dslx/interp_value.h"

namespace xls::dslx {
namespace {

using ::absl_testing::IsOkAndHolds;

TEST(InterpValueFromStringTest, SimpleConstantBitLiterals) {
  EXPECT_THAT(InterpValueFromString("u32:42", kDefaultDslxStdlibPath),
              IsOkAndHolds(InterpValue::MakeU32(42)));
  EXPECT_THAT(InterpValueFromString("u8:64", kDefaultDslxStdlibPath),
              IsOkAndHolds(InterpValue::MakeU8(64)));
}

TEST(InterpValueFromStringTest, SimpleConstantTupleLiterals) {
  EXPECT_THAT(InterpValueFromString("()", kDefaultDslxStdlibPath),
              IsOkAndHolds(InterpValue::MakeTuple({})));
  const InterpValue want = InterpValue::MakeTuple({
      InterpValue::MakeU8(64),
      InterpValue::MakeU32(42),
  });
  EXPECT_THAT(InterpValueFromString("(u8:64, u32:42)", kDefaultDslxStdlibPath),
              IsOkAndHolds(want));
}

TEST(InterpValueFromStringTest, ConstexprUsingStdlib) {
  EXPECT_THAT(
      InterpValueFromString("std::popcount(u32:0x5)", kDefaultDslxStdlibPath),
      IsOkAndHolds(InterpValue::MakeU32(2)));
}

}  // namespace
}  // namespace xls::dslx
