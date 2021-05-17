// Copyright 2021 The XLS Authors
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

#include "xls/dslx/mangle.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"

namespace xls::dslx {
namespace {

using status_testing::IsOkAndHolds;

TEST(MangleTest, SimpleModuleFunction) {
  EXPECT_THAT(
      MangleDslxName("my_mod", "f", CallingConvention::kTypical, {}, nullptr),
      IsOkAndHolds("__my_mod__f"));
  EXPECT_THAT(MangleDslxName("my_mod", "f", CallingConvention::kImplicitToken,
                             {}, nullptr),
              IsOkAndHolds("__itok__my_mod__f"));
}

TEST(MangleTest, SingleFreeKey) {
  std::vector<std::pair<std::string, InterpValue>> bindings = {
      {"x", InterpValue::MakeU32(42)}};
  SymbolicBindings symbolic_bindings(bindings);
  EXPECT_THAT(MangleDslxName("my_mod", "p", CallingConvention::kTypical, {"x"},
                             &symbolic_bindings),
              IsOkAndHolds("__my_mod__p__42"));
}

TEST(MangleTest, TwoFreeKeys) {
  std::vector<std::pair<std::string, InterpValue>> bindings = {
      {"x", InterpValue::MakeU32(42)}, {"y", InterpValue::MakeU32(64)}};
  SymbolicBindings symbolic_bindings(bindings);
  EXPECT_THAT(MangleDslxName("my_mod", "p", CallingConvention::kTypical,
                             {"x", "y"}, &symbolic_bindings),
              IsOkAndHolds("__my_mod__p__42_64"));
}

}  // namespace
}  // namespace xls::dslx
