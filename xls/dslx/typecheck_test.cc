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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/parse_and_typecheck.h"

namespace xls::dslx {
namespace {

using status_testing::StatusIs;
using testing::HasSubstr;

TEST(TypecheckTest, ParametricWrongArgCount) {
  absl::string_view text = R"(
fn id<N: u32>(x: bits[N]) -> bits[N] { x }
fn f() -> u32 { id(u8:3, u8:4) }
)";
  ImportCache import_cache;
  EXPECT_THAT(
      ParseAndTypecheck(text, "fake.x", "fake", &import_cache),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Expected 1 parameter(s) but got 2 argument(s)")));
}

TEST(TypecheckTest, ParametricTooManyExplicitSupplied) {
  absl::string_view text = R"(
fn id<X: u32>(x: bits[X]) -> bits[X] { x }
fn main() -> u32 { id<u32:32, u32:64>(u32:5) }
)";
  ImportCache import_cache;
  EXPECT_THAT(
      ParseAndTypecheck(text, "fake.x", "fake", &import_cache),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Too many parametric values supplied; limit: 1 given: 2")));
}

}  // namespace
}  // namespace xls::dslx
