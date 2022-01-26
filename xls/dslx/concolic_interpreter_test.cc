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
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/interpreter.h"
#include "xls/dslx/parse_and_typecheck.h"

namespace xls::dslx {
namespace {

// Runs the interpreter in concolic mode and returns the generated DSLX tests.
static absl::StatusOr<std::string> RunConcolic(std::string program) {
  auto import_data = CreateImportDataForTest();
  XLS_ASSIGN_OR_RETURN(
      TypecheckedModule tm,
      ParseAndTypecheck(program, "top.x", "top", &import_data));
  Interpreter interp(tm.module, /*typecheck=*/nullptr,
                     /*import_data=*/&import_data, /*trace_all=*/false,
                     /*run_concolic=*/true);
  XLS_RETURN_IF_ERROR(interp.RunTest("top_test"));
  return interp.GetConcolicTestCases();
}

TEST(ConcolicTest, TernaryIfNode) {
  const std::string kProgram = R"(
fn my_fn(a: u8) -> u8 {
  let result = if (a == u8:10) {u8:1} else {u8:2};
  result
}
fn top(a: u8) -> s32 {
  let result = if (a > u8:4) {my_fn(a) + a + u8:2} else {my_fn(a)};
  let result = if (result == u8:24) {s32:0} else {s32:1};
  result
}

#![test]
fn top_test () {
  let a_test = u8:0;
  assert_eq(top(a_test), s32:1)
}
)";
  const std::string expected_test = R"(
#![test]
fn top_test_0() {
  let in_0 = u8:10;

  let _ = assert_eq(top(in_0), s32:1);
  ()
}
#![test]
fn top_test_1() {
  let in_0 = u8:128;

  let _ = assert_eq(top(in_0), s32:1);
  ()
}
#![test]
fn top_test_2() {
  let in_0 = u8:20;

  let _ = assert_eq(top(in_0), s32:0);
  ()
})";
  XLS_ASSERT_OK_AND_ASSIGN(std::string concolic_result, RunConcolic(kProgram));
  EXPECT_EQ(expected_test, concolic_result);
}

TEST(ConcolicTest, MatchTuple) {
  const std::string kProgram = R"(
fn top(a: u1, b: u1) -> s32 {
   let result_sign = match (a, b) {
    (true, _) => s32:0,
    (false, true) => s32:0,
    _ => s32:0,
  };
  result_sign
}

#![test]
fn top_test() {
  let a = u1:0;
  let b = u1:0;
  assert_eq(top(a, b), s32:0)
}
)";
  const std::string expected_test = R"(
#![test]
fn top_test_0() {
  let in_0 = u1:1;
  let in_1 = u1:0;

  let _ = assert_eq(top(in_0, in_1), s32:0);
  ()
}
#![test]
fn top_test_1() {
  let in_0 = u1:0;
  let in_1 = u1:1;

  let _ = assert_eq(top(in_0, in_1), s32:0);
  ()
}
#![test]
fn top_test_2() {
  let in_0 = u1:0;
  let in_1 = u1:0;

  let _ = assert_eq(top(in_0, in_1), s32:0);
  ()
})";
  XLS_ASSERT_OK_AND_ASSIGN(std::string concolic_result, RunConcolic(kProgram));
  EXPECT_EQ(expected_test, concolic_result);
}

TEST(ConcolicTest, MatchLiteral) {
  const std::string kProgram = R"(
fn top(a: u8) -> s32 {
  let b = u8:3;
  let a_match = match a {
    u8:0 | u8:1 => s32:0,
    u8:2 => s32:1,
    b => s32:2,
    y => y as s32 + s32:4,
    _ => s32:5,
  };

  let result = if (a_match == s32:12) {s32:0} else {s32:1};
  result
}

#![test]
fn top_test() {
  let a_test = u8:99;
  assert_eq(top(a_test), s32:1)
}
)";
  const std::string expected_test = R"(
#![test]
fn top_test_0() {
  let in_0 = u8:0;

  let _ = assert_eq(top(in_0), s32:1);
  ()
}
#![test]
fn top_test_1() {
  let in_0 = u8:1;

  let _ = assert_eq(top(in_0), s32:1);
  ()
}
#![test]
fn top_test_2() {
  let in_0 = u8:2;

  let _ = assert_eq(top(in_0), s32:1);
  ()
}
#![test]
fn top_test_3() {
  let in_0 = u8:3;

  let _ = assert_eq(top(in_0), s32:1);
  ()
}
#![test]
fn top_test_4() {
  let in_0 = u8:248;

  let _ = assert_eq(top(in_0), s32:1);
  ()
}
#![test]
fn top_test_5() {
  let in_0 = u8:8;

  let _ = assert_eq(top(in_0), s32:0);
  ()
})";
  XLS_ASSERT_OK_AND_ASSIGN(std::string concolic_result, RunConcolic(kProgram));
  EXPECT_EQ(expected_test, concolic_result);
}

TEST(ConcolicTest, UnaryOp) {
  const std::string kProgram = R"(
  fn top (a: s8) -> s32 {
  let neg = -a;
  let inv = !a;
  let _ = if (neg == s8:16) {s32: 0} else {s32: 1};
  let _ = if (inv == s8:0xf0) {s32: 0} else {s32: 1};
  s32: 1
}

#![test]
fn top_test() {
  let a_test = s8:0;
  assert_eq(top(a_test), s32:1)
}
)";
  const std::string expected_test = R"(
#![test]
fn top_test_0() {
  let in_0 = s8:-16;

  let _ = assert_eq(top(in_0), s32:1);
  ()
}
#![test]
fn top_test_1() {
  let in_0 = s8:15;

  let _ = assert_eq(top(in_0), s32:1);
  ()
})";
  XLS_ASSERT_OK_AND_ASSIGN(std::string concolic_result, RunConcolic(kProgram));
  EXPECT_EQ(expected_test, concolic_result);
}

TEST(ConcolicTest, FnInvocation) {
  const std::string kProgram = R"(
  fn my_abs(num: s16) -> s16 {
  let result = if (num >= s16:0) {num} else {num * s16:-1};
  result + s16:10
}

fn top(a: s16) -> s32 {
    let a_abs = my_abs(a);
    let result = if (my_abs(a) == s16:33) {s32:0} else {s32:-1};
    result
}

#![test]
fn top_test() {
  let a = s16:0;
  assert_eq(top(a), s32:-1)
}
)";
  const std::string expected_test = R"(
#![test]
fn top_test_0() {
  let in_0 = s16:-32768;

  let _ = assert_eq(top(in_0), s32:-1);
  ()
}
#![test]
fn top_test_1() {
  let in_0 = s16:-23;

  let _ = assert_eq(top(in_0), s32:0);
  ()
}
#![test]
fn top_test_2() {
  let in_0 = s16:0;

  let _ = assert_eq(top(in_0), s32:-1);
  ()
})";
  XLS_ASSERT_OK_AND_ASSIGN(std::string concolic_result, RunConcolic(kProgram));
  EXPECT_EQ(expected_test, concolic_result);
}

TEST(ConcolicTest, ForLoop) {
  const std::string kProgram = R"(
  fn top(a: u32[5]) -> u32 {

  let final_accum = for (i, accum): (u32, u32) in range(u32:0, u32:5) {
  let new_accum = a[i] * accum;
  new_accum
  }(u32:1);

 let _ = if (final_accum == u32:25) {s32:0} else {s32:1};
 final_accum
 }

#![test]
fn top_test() {
  let a = [u32:0, u32:0, u32:0, u32:0, u32:0];
  let b = u32:0;
  assert_eq(top(a), u32:0)
}
)";
  const std::string expected_test = R"(
#![test]
fn top_test_0() {
  let in_0 = [u32:25, u32:1, u32:1, u32:1, u32:1];

  let _ = assert_eq(top(in_0), u32:25);
  ()
})";
  XLS_ASSERT_OK_AND_ASSIGN(std::string concolic_result, RunConcolic(kProgram));
  EXPECT_EQ(expected_test, concolic_result);
}

TEST(ConcolicTest, TupleBinaryOp) {
  const std::string kProgram = R"(
  fn top(a: (u8, s8, u16, s16)) -> s32 {
  let t = (u8:1, s16:-1);
  let (t0, t1) = t;
  let _ = if (a[0] + t0 == u8:101) {s32:0} else {s32:1};
  let result = if (a[3] * t1  == s16:20 ) {s32:-1} else {s32:-2};
  result
}

#![test]
fn top_test() {
  let a_test = (u8:0, s8:0, u16:0, s16:0);
  assert_eq(top(a_test), s32:-2)
}
)";
  const std::string expected_test = R"(
#![test]
fn top_test_0() {
  let in_0 = (u8:100, s8:0, u16:0, s16:0);

  let _ = assert_eq(top(in_0), s32:-2);
  ()
}
#![test]
fn top_test_1() {
  let in_0 = (u8:0, s8:0, u16:0, s16:-20);

  let _ = assert_eq(top(in_0), s32:-1);
  ()
})";
  XLS_ASSERT_OK_AND_ASSIGN(std::string concolic_result, RunConcolic(kProgram));
  EXPECT_EQ(expected_test, concolic_result);
}

TEST(ConcolicTest, StrcutBinaryOp) {
  const std::string kProgram = R"(
  struct Point {
  x: u32,
  y: u32,
}

fn top(p0: Point) -> s32 {
  let p1 = Point {x: u32:1, y: u32:2};
  let result = if (p0.x * p0.y + p1.x == u32:16) {s32:1} else {s32:2};
  result
}

#![test]
fn top_test() {
  let p0 = Point {x: u32:0, y: u32:0 };
  assert_eq(top(p0), s32:2)
}
)";
  const std::string expected_test = R"(
#![test]
fn top_test_0() {
  let in_0 = Point {x: u32:15, y: u32:1};

  let _ = assert_eq(top(in_0), s32:1);
  ()
})";
  XLS_ASSERT_OK_AND_ASSIGN(std::string concolic_result, RunConcolic(kProgram));
  EXPECT_EQ(expected_test, concolic_result);
}

TEST(ConcolicTest, ArrayMult) {
  const std::string kProgram = R"(
  fn top(a: s32[2], b:s32) -> s32 {
  let x:s32[2] = [a[u8:0], a[u8:1]];
  let y = [s32:-4, s32:1, s32:8];
  let result = if (a[u8:0] * b * x[u8:1] + y[u8:0] == s32:4) {s32:1} else {s32:2};
  result
}

#![test]
fn top_test() {
 let a = [s32:0, s32:0];
 let b = s32:0;
 assert_eq(top(a, b), s32:2)
}
)";
  const std::string expected_test = R"(
#![test]
fn top_test_0() {
  let in_0 = [s32:8, s32:1];
  let in_1 = s32:1;

  let _ = assert_eq(top(in_0, in_1), s32:1);
  ()
})";
  XLS_ASSERT_OK_AND_ASSIGN(std::string concolic_result, RunConcolic(kProgram));
  EXPECT_EQ(expected_test, concolic_result);
}

TEST(ConcolicTest, StructEq) {
  const std::string kProgram = R"(
  struct Point {
  x: s16,
  y: u32,
 }
 fn top(p0: Point) -> s32 {
    let p1 = Point {x: s16:-100, y: u32:100};
    let result = if (p0 == p1) {s32:-1} else {s32:-2};
    result
}

 #![test]
 fn top_test() {
   let p0 = Point {x: s16:0, y: u32:0 };
   assert_eq(top(p0), s32:-2)
 })";
  const std::string expected_test = R"(
#![test]
fn top_test_0() {
  let in_0 = Point {x: s16:-100, y: u32:100};

  let _ = assert_eq(top(in_0), s32:-1);
  ()
})";
  XLS_ASSERT_OK_AND_ASSIGN(std::string concolic_result, RunConcolic(kProgram));
  EXPECT_EQ(expected_test, concolic_result);
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

TEST(ConcolicTest, Binop) {
  const std::string kProgram = R"(
  fn top(a: s8, b: s8) -> s32 {
  let _ = if (a + b == s8: -16) {s32:0} else {s32:1};
  let _ = if (a - b == s8: -16) {s32:0} else {s32:1};
  let _ = if (a * b == s8: -16) {s32:0} else {s32:1};
  let _ = if (a / b == s8: -16) {s32:0} else {s32:1};
  let _ = if (a as u8 ++ b as u8 == u16: 0xf00f) {s32:0} else {s32:1};
  let _ = if (a >> b as u8 == s8: -16) {s32:0} else {s32:1};
  let _ = if (a >> u8:2 == s8: -16) {s32:0} else {s32:1};
  let _ = if (a << u8:3 == s8: 16) {s32:0} else {s32:1};

  s32:1
}

#![test]
fn top_test () {
  let a_test = s8:0;
  let b_test = s8:1;
  assert_eq(top(a_test, b_test), s32:1)
})";
  const std::string expected_test = R"(
#![test]
fn top_test_0() {
  let in_0 = s8:-16;
  let in_1 = s8:0;

  let _ = assert_eq(top(in_0, in_1), s32:1);
  ()
}
#![test]
fn top_test_1() {
  let in_0 = s8:-16;
  let in_1 = s8:0;

  let _ = assert_eq(top(in_0, in_1), s32:1);
  ()
}
#![test]
fn top_test_2() {
  let in_0 = s8:-16;
  let in_1 = s8:1;

  let _ = assert_eq(top(in_0, in_1), s32:1);
  ()
}
#![test]
fn top_test_3() {
  let in_0 = s8:-16;
  let in_1 = s8:1;

  let _ = assert_eq(top(in_0, in_1), s32:1);
  ()
}
#![test]
fn top_test_4() {
  let in_0 = s8:-16;
  let in_1 = s8:15;

  let _ = assert_eq(top(in_0, in_1), s32:1);
  ()
}
#![test]
fn top_test_5() {
  let in_0 = s8:-16;
  let in_1 = s8:0;

  let _ = assert_eq(top(in_0, in_1), s32:1);
  ()
}
#![test]
fn top_test_6() {
  let in_0 = s8:-64;
  let in_1 = s8:0;

  let _ = assert_eq(top(in_0, in_1), s32:1);
  ()
}
#![test]
fn top_test_7() {
  let in_0 = s8:2;
  let in_1 = s8:0;

  let _ = assert_eq(top(in_0, in_1), s32:1);
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
