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
#include "absl/strings/match.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/interpreter.h"
#include "xls/dslx/parse_and_typecheck.h"

namespace xls::dslx {
namespace {

TEST(ConcolicTest, SingleBoolTest) {
  const std::string kProgram = R"(
fn top(a: u1) -> u1 {
  let result = if(a) {u1:1} else {u1:0};
  result
}

#![test]
fn top_test() {
  let a_test = u1:0;
  assert_eq(top(a_test), u1:0)
}
)";

  const std::string expected_test = R"(#![test]
fn top_test_0() {
  let in_0 = u1:1;

  let _ = assert_eq(top(in_0), u1:1);
  ()
})";

  auto import_data = ImportData::CreateForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "top.x", "top", &import_data));
  Interpreter interp(tm.module, /*typecheck=*/nullptr,
                     /*import_data=*/&import_data, /*trace_all=*/false,
                     /*run_concolic=*/true);

  absl::Status result = interp.RunTest("top_test");
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(expected_test, interp.GetConcolicTestCases());
}

TEST(ConcolicTest, LogicalAndTest) {
  const std::string kProgram = R"(
fn top(a: s8, b: s8, c: u1) -> s32 {
  let x = a;
  let y = b;
  let z = x + y;
  let d = a & b;

  let result = if((z == s8:9 && c == u1:1 && d == s8:10)) {s32:-1} else {s32:2};
  result
}

#![test]
fn top_test() {
  let a_test = s8:-5;
  let b_test = s8:4;
  let c_test = u1:1;
  assert_eq(top(a_test, b_test, c_test), s32:2)
}
)";

  const std::string expected_test = R"(#![test]
fn top_test_0() {
  let in_0 = s8:-1;
  let in_1 = s8:10;
  let in_2 = u1:1;

  let _ = assert_eq(top(in_0, in_1, in_2), s32:-1);
  ()
})";

  auto import_data = ImportData::CreateForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "top.x", "top", &import_data));
  Interpreter interp(tm.module, /*typecheck=*/nullptr,
                     /*import_data=*/&import_data, /*trace_all=*/false,
                     /*run_concolic=*/true);

  absl::Status result = interp.RunTest("top_test");
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(expected_test, interp.GetConcolicTestCases());
}

}  // namespace
}  // namespace xls::dslx
