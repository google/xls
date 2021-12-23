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
#include "xls/common/status/status_macros.h"
#include "xls/dslx/interpreter.h"
#include "xls/dslx/parse_and_typecheck.h"

namespace xls::dslx {
namespace {

// Runs the interpreter in concolic mode and returns the generated DSLX tests.
static absl::StatusOr<std::string> RunConcolic(std::string program) {
  auto import_data = ImportData::CreateForTest();
  XLS_ASSIGN_OR_RETURN(
      TypecheckedModule tm,
      ParseAndTypecheck(program, "top.x", "top", &import_data));
  Interpreter interp(tm.module, /*typecheck=*/nullptr,
                     /*import_data=*/&import_data, /*trace_all=*/false,
                     /*run_concolic=*/true);
  XLS_RETURN_IF_ERROR(interp.RunTest("top_test"));
  return interp.GetConcolicTestCases();
}

TEST(ConcolicTest, ParamCast) {
  const std::string kProgram = R"(
  fn dummy_cast(a: u16) -> s32 {
  let c = u4:4;
  let c_cast = c as u32;
  let a_cast = a as u32;
  let _ = if (a_cast + c_cast == u32:16) {u1:1} else {u1:0};
  s32:0
}

fn top(a: u16) -> s32 {
  let _ = dummy_cast(a);
  let a_cast = a as u10;
  let _ = if (a_cast == u10:233) {s32:1} else {s32:0};
  let _ = if (a * u16:3 == u16:21) {s32:1} else {s32:0};
  s32:0
}

#![test]
fn top_test() {
  let a_test = u16:0;
  assert_eq(top(a_test), s32:0)
}
)";
  const std::string expected_test = R"(
#![test]
fn top_test_0() {
  let in_0 = u16:12;

  let _ = assert_eq(top(in_0), s32:0);
  ()
}
#![test]
fn top_test_1() {
  let in_0 = u16:233;

  let _ = assert_eq(top(in_0), s32:0);
  ()
}
#![test]
fn top_test_2() {
  let in_0 = u16:7;

  let _ = assert_eq(top(in_0), s32:0);
  ()
})";
  XLS_ASSERT_OK_AND_ASSIGN(std::string concolic_result, RunConcolic(kProgram));
  EXPECT_EQ(expected_test, concolic_result);
}

TEST(ConcolicTest, BitSlice) {
  const std::string kProgram = R"(
  fn top(a: u8) -> s32 {
  let b = a;
  let _ = if (a[1:4] + u3:1 == u3:7) {u1:0} else {u1:1};
  let result = if (b[-1:]) {s32:0} else {s32:1};
  result
}

#![test]
fn top_test() {
  let a_test = u8:0;
  assert_eq(top(a_test), s32:1)
}
)";
  const std::string expected_test = R"(
#![test]
fn top_test_0() {
  let in_0 = u8:12;

  let _ = assert_eq(top(in_0), s32:1);
  ()
}
#![test]
fn top_test_1() {
  let in_0 = u8:128;

  let _ = assert_eq(top(in_0), s32:0);
  ()
})";
  XLS_ASSERT_OK_AND_ASSIGN(std::string concolic_result, RunConcolic(kProgram));
  EXPECT_EQ(expected_test, concolic_result);
}

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

  const std::string expected_test = R"(
#![test]
fn top_test_0() {
  let in_0 = u1:1;

  let _ = assert_eq(top(in_0), u1:1);
  ()
})";
  XLS_ASSERT_OK_AND_ASSIGN(std::string concolic_result, RunConcolic(kProgram));
  EXPECT_EQ(expected_test, concolic_result);
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

  const std::string expected_test = R"(
#![test]
fn top_test_0() {
  let in_0 = s8:-1;
  let in_1 = s8:10;
  let in_2 = u1:1;

  let _ = assert_eq(top(in_0, in_1, in_2), s32:-1);
  ()
})";

  XLS_ASSERT_OK_AND_ASSIGN(std::string concolic_result, RunConcolic(kProgram));
  EXPECT_EQ(expected_test, concolic_result);
}

}  // namespace
}  // namespace xls::dslx
