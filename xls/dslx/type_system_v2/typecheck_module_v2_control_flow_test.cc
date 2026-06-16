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

#include <string>
#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system/typecheck_test_utils.h"
#include "xls/dslx/type_system_v2/matchers.h"

// Tests for `if`, `for`, and `unroll_for` expressions.
namespace xls::dslx {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::AllOf;
using ::testing::HasSubstr;

TEST(TypecheckV2Test, IfType) {
  EXPECT_THAT("const X = if true { u32:1 } else { u32:0 };",
              TypecheckSucceeds(HasNodeWithType("X", "uN[32]")));
}

TEST(TypecheckV2Test, IfTestVariable) {
  EXPECT_THAT("const Y = true; const X = if Y { u32:1 } else { u32:0 };",
              TypecheckSucceeds(HasNodeWithType("X", "uN[32]")));
}

TEST(TypecheckV2Test, IfTestVariableNotVariable) {
  EXPECT_THAT("const Y = true; const X = if Y { Y } else { !Y };",
              TypecheckSucceeds(HasNodeWithType("X", "uN[1]")));
}

TEST(TypecheckV2Test, IfTestVariables) {
  EXPECT_THAT(R"(
const Y = true;
const Z = false;
const X = if (Y && Z) {u32:1} else { u32:2 };
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("Y", "uN[1]"),
                                      HasNodeWithType("Z", "uN[1]"),
                                      HasNodeWithType("X", "uN[32]"))));
}

TEST(TypecheckV2Test, IfTestBadVariable) {
  EXPECT_THAT("const Y = u32:1; const X = if Y { u32:1 } else { u32:0 };",
              TypecheckFails(HasSizeMismatch("u32", "bool")));
}

TEST(TypecheckV2Test, IfTestInt) {
  EXPECT_THAT("const X = if u32:1 { u32:1 } else { u32:0 };",
              TypecheckFails(HasSizeMismatch("u32", "bool")));
}

TEST(TypecheckV2Test, IfAlternativeWrongType) {
  EXPECT_THAT("const X = if true { u32:1 } else { u31:0 };",
              TypecheckFails(HasSizeMismatch("u31", "u32")));
}

TEST(TypecheckV2Test, IfElseIf) {
  EXPECT_THAT(R"(
const X = if false {
    u32:1
} else if true {
    u32:2
} else {
    u32:3
};)",
              TypecheckSucceeds(HasNodeWithType("X", "uN[32]")));
}

TEST(TypecheckV2Test, IfParametricVariable) {
  EXPECT_THAT(R"(
fn f<N:u32>(x: uN[N]) -> u32 { if true { N } else { N }}
const Y = f(u10:256);
)",
              TypecheckSucceeds(HasNodeWithType("Y", "uN[32]")));
}

TEST(TypecheckV2Test, IfParametricType) {
  EXPECT_THAT(R"(
fn f<N:u32>(x: uN[N]) -> uN[N] { if true { x } else { x }}
const Y = f(u10:256);
)",
              TypecheckSucceeds(HasNodeWithType("Y", "uN[10]")));
}

TEST(TypecheckV2Test, ConditionalWithConversionForUnification) {
  EXPECT_THAT(R"(
fn lsb<S: bool, N: u32>(x: xN[S][N]) -> u1 { x as u1 }
fn repro(x: u3) -> u2 {
    let (_dummy, upper) =
        if x == u3:0 { (u8:0, 0b00) } else { (u8:0, lsb(x) ++ u1:0) };
    upper
}
  )",
              TypecheckSucceeds(HasNodeWithType("upper", "uN[2]")));
}

TEST(TypecheckV2Test, ConstConditionalDisparateTypes) {
  EXPECT_THAT(R"(
fn f<N: u32>() -> uN[N] {
  const if (N == 1) {
    u1:1
  } else {
    zero!<uN[N]>()
  }
}
fn main() {
  let foo = f<u32:1>();
  let bar = f<u32:10>();
  let baz = f<u32:20>();
}
  )",
              TypecheckSucceeds(AllOf(HasNodeWithType("foo", "uN[1]"),
                                      HasNodeWithType("bar", "uN[10]"),
                                      HasNodeWithType("baz", "uN[20]"))));
}

TEST(TypecheckV2Test, ConstConditionalUnableToEvaluate) {
  EXPECT_THAT(
      R"(
fn broken_if(a: u32) -> bool {
  const if a == u32:1 {
    true
  } else {
    false
  }
}
  )",
      TypecheckFails(HasSubstr("Unable to evaluate const conditional")));
}

TEST(TypecheckV2Test, BinopWithConversionForUnification) {
  EXPECT_THAT(R"(
fn lsb<S: bool, N: u32>(x: xN[S][N]) -> u1 { x as u1 }
fn repro(x: u3) -> u2 {
  let upper = u2:1 + (lsb(x) ++ u1:0);
  upper
}
  )",
              TypecheckSucceeds(HasNodeWithType("upper", "uN[2]")));
}

TEST(TypecheckV2Test, ShiftByMoreThan64Bits) {
  // The point here is to prove that the validation does not rely on conversion
  // of the RHS to `int64_t`. In v1 it would do this and incorrectly allow this
  // example to pass.
  EXPECT_THAT(
      R"(
const X = sN[80]:0x8000_0000_0000_0000_0000 >> uN[80]:0x0aaa_bbbb_cccc_dddd_eeee;
)",
      TypecheckFails(
          HasSubstr("Shifting a 80-bit value (`sN[80]`) by a constexpr shift "
                    "of 0xaaa_bbbb_cccc_dddd_eeee exceeds its bit width")));
}

TEST(TypecheckV2Test, ConstAssertForcesTypeBool) {
  EXPECT_THAT(
      R"(
fn foo<N: u32>(x:uN[N]) -> uN[N] {
  const_assert!(x);
  x
}
fn main() {
  foo<u32:3>(7);
}
)",
      TypecheckFails(HasSizeMismatch("uN[3]", "uN[1]")));
}

TEST(TypecheckV2Test, TypeAliasWithUnspecifiedParametrics) {
  EXPECT_THAT(R"(
struct S<X: u32, Y: u32> {
  x: bits[X],
  y: bits[Y],
}
type Alias = S;
fn f() -> Alias<24, 32> { Alias {x: 3, y: 4 } }
)",
              TypecheckSucceeds(AllOf(
                  HasNodeWithType("f", "() -> S { x: uN[24], y: uN[32] }"))));
}

TEST(TypecheckV2Test, TypeAliasWithUnspecifiedParametricsAsParameter) {
  EXPECT_THAT(R"(
pub struct A<T: u32> { v: uN[T] }
type X = A;
fn f(a: X<32>) {}
)",
              TypecheckSucceeds(
                  AllOf(HasNodeWithType("f", "(A { v: uN[32] }) -> ()"))));
}

TEST(TypecheckV2Test, TypeAliasWithUnspecifiedParametricsChained) {
  EXPECT_THAT(R"(
pub struct A<T: u32> { v: uN[T] }
type X = A;
type Y = X;
type Z = Y<32>;
fn f(a: Z) {}
)",
              TypecheckSucceeds(
                  AllOf(HasNodeWithType("f", "(A { v: uN[32] }) -> ()"))));
}

TEST(TypecheckV2Test, SliceBeforeStart) {
  EXPECT_THAT("const X = (u6:0b011100)[-7:4];",
              TypecheckSucceeds(HasNodeWithType("X", "uN[4]")));
}

TEST(TypecheckV2Test, WidthSliceBeforeStartFails) {
  EXPECT_THAT("const X = (u6:0b011100)[-7+:u4];",
              TypecheckFails(HasSignednessMismatch("s4", "u6")));
}

TEST(TypecheckV2Test, ForRange) {
  EXPECT_THAT(
      R"(
const A = u32:0..u32:5;

fn foo() {
  let X = for (i, a) : (u32, s32) in A {
    a + (i as s32)
  } (0);
}
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("A", "uN[32][5]"),
                              HasNodeWithType("i", "uN[32]"),
                              HasNodeWithType("a", "sN[32]"),
                              HasNodeWithType("(i, a)", "(uN[32], sN[32])"),
                              HasNodeWithType("X", "sN[32]"))));
}

TEST(TypecheckV2Test, ForCompositeType) {
  EXPECT_THAT(
      R"(
fn foo(A : (s32, s16)[10]) {
  let a1 = u32:0;
  let b1 = s16:0;
  let c1 = u8:0;
  let X = for (i, (a, (b, c))) in A {
    (a1 + (i.0 as u32), (b1 + i.1, c1 + 1))
  } ((a1, (b1, c1)));
}
)",
      TypecheckSucceeds(
          AllOf(HasNodeWithType("A", "(sN[32], sN[16])[10]"),
                HasNodeWithType("i", "(sN[32], sN[16])"),
                HasNodeWithType("a", "uN[32]"), HasNodeWithType("b", "sN[16]"),
                HasNodeWithType("c", "uN[8]"),
                HasNodeWithType("X", "(uN[32], (sN[16], uN[8]))"))));
}

TEST(TypecheckV2Test, ForInvalidType) {
  EXPECT_THAT(
      R"(
fn foo() {
  let A = u32:10;
  let X = for (i, a) : (u32, s32) in A {
    a + (i as s32)
  } (0);
}
)",
      TypecheckFails(HasTypeMismatch("u32[>= u32:0]", "u32")));
}

TEST(TypecheckV2Test, ForInferenceFromBodyTypeFails) {
  EXPECT_THAT(
      R"(
fn foo() {
  let X = for (i, a) in u32:0..5 {
    a + (i as s32)
  } (0);
}
)",
      TypecheckFails(
          HasSubstr("Loop cannot have an implicit result type derived from "
                    "init expression `0`")));
}

TEST(TypecheckV2Test, ForInferenceFromLhsType) {
  EXPECT_THAT(
      R"(
fn foo() {
  let X:s32 = for (i, a) in u32:0..5 {
    a + (i as s32)
  } (0);
}
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("X", "sN[32]"),
                              HasNodeWithType("a", "sN[32]"))));
}

TEST(TypecheckV2Test, ForInferenceFromInitType) {
  EXPECT_THAT(
      R"(
fn foo() {
  let X = for (i, a) in 0..5 {
    a + i
  } (s32:0);
}
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("0..5", "uN[3][5]"),
                              HasNodeWithType("i", "uN[3]"),
                              HasNodeWithType("a", "sN[32]"),
                              HasNodeWithType("(i, a)", "(uN[3], sN[32])"),
                              HasNodeWithType("X", "sN[32]"))));
}

TEST(TypecheckV2Test, ForInferenceFromAnnotation) {
  EXPECT_THAT(
      R"(
fn foo() {
  let X = for (i, a) : (u32, u16) in 0..5 {
    a + (i as u16)
  } (123);
}
)",
      TypecheckSucceeds(AllOf(
          HasNodeWithType("0..5", "uN[32][5]"), HasNodeWithType("i", "uN[32]"),
          HasNodeWithType("a", "uN[16]"),
          HasNodeWithType("(i, a)", "(uN[32], uN[16])"),
          HasNodeWithType("123", "uN[16]"), HasNodeWithType("X", "uN[16]"))));
}

TEST(TypecheckV2Test, ForInferenceFromResult) {
  EXPECT_THAT(
      R"(
fn foo() -> s32 {
  for (i, a) in u32:0..5 {
    a + (i as s32)
  } (0)
}
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("u32:0..5", "uN[32][5]"),
                              HasNodeWithType("i", "uN[32]"),
                              HasNodeWithType("a", "sN[32]"),
                              HasNodeWithType("(i, a)", "(uN[32], sN[32])"))));
}

TEST(TypecheckV2Test, ForInvalidTypeAnnotation) {
  EXPECT_THAT(
      R"(
fn foo(A : s32[5]) {
  let X = for (i, a) : (u32) in A {
    a + i
  } (0);
}
)",
      TypecheckFails(HasSubstr(
          "For-loop annotated type should be a tuple containing a type for the "
          "iterable and a type for the accumulator.")));
}

TEST(TypecheckV2Test, IterativeDivMod) {
  // This is a copy of the function in std.x, which is of interest because it
  // takes an unacceptable amount of time to type check without caching.
  EXPECT_THAT(R"(
pub fn iterative_div_mod<N: u32, M: u32>(n: uN[N], d: uN[M]) -> (uN[N], uN[M]) {
    // Zero extend divisor by 1 bit.
    let divisor = d as uN[M + u32:1];

    for (i, (q, r)): (u32, (uN[N], uN[M])) in u32:0.. N {
        // Shift the next bit of n into r.
        let r = r ++ n[(N - u32:1 - i)+:u1];
        let (q, r) = if r >= divisor {
            (q as uN[N - u32:1] ++ u1:1, r - divisor)
        } else {
            (q as uN[N - u32:1] ++ u1:0, r)
        };
        // Remove the MSB of r; guaranteed to be 0 because r < d.
        (q, r[0:M as s32])
    }((uN[N]:0, uN[M]:0))
}

const X = iterative_div_mod(u32:20, u32:3);
)",
              TypecheckSucceeds(HasNodeWithType("X", "(uN[32], uN[32])")));
}

TEST(TypecheckV2Test, ConstFor) {
  EXPECT_THAT(
      R"(
fn foo() {
  let res = const for (i, acc) in u32:0..u32:5 {
    let acc = acc + i;
    acc * u32:2
  } (u32:0);
}
)",
      TypecheckSucceeds(
          AllOf(HasNodeWithType("acc", "uN[32]"),
                HasRepeatedNodeWithType("let acc = acc + i;", "uN[32]", 5))));
}

TEST(TypecheckV2Test, ConstForParametrized) {
  EXPECT_THAT(
      R"(
fn param_for<loops: u32>() -> u32 {
  const for (i, acc) in u32:0..loops {
    acc + i
  } (u32:0)
}

fn main() {
  let res = param_for<u32:6>();
}
  )",
      TypecheckSucceeds(HasNodeWithType("res", "uN[32]")));
}

TEST(TypecheckV2Test, ConstForUnableToEvaluate) {
  EXPECT_THAT(
      R"(
fn broken_for(a: u32) -> u32 {
  const for (i, acc) in u32:0..a {
    acc + i
  } (u32:0)
}
  )",
      TypecheckFails(HasSubstr("is not constexpr")));
}

TEST(TypecheckV2Test, UnrollFor) {
  // Verify that the loop is unrolled 5 times, and is also constant-folded.
  EXPECT_THAT(
      R"(
const A = u32:1;
fn foo() {
  let B = u32:2;
  const X = unroll_for! (i, a) in u32:0..u32:5 {
    let C = B + i;
    let D = A * a;
    C + D
  } (u32:0);
  let Y : u32[X] = [0, ...];
}
)",
      TypecheckSucceeds(AllOf(
          HasNodeWithType("X", "uN[32]"), HasNodeWithType("Y", "uN[32][20]"),
          HasRepeatedNodeWithType("let C = B + i;", "uN[32]", 5))));
}

TEST(TypecheckV2Test, UnrollForBodyNotUpdatingAcc) {
  EXPECT_THAT(
      R"(
const A = u32:1;
fn foo() {
  let B = u32:2;
  const X = unroll_for! (i, a) in u32:0..u32:5 {
    let C = B + i;
    let D = A * a;
  } (u32:0);
  let Y : u32[X] = [0, ...];
}
)",
      TypecheckFails(HasSubstr(
          "Loop has an accumulator but the body does not produce a value")));
}

TEST(TypecheckV2Test, UnrollForNested) {
  // Unrolled 0 + 1 + 2 + 3 + 4 = 10 times.
  EXPECT_THAT(
      R"(
fn foo() {
  let X = unroll_for! (i, a) in u32:0..5 {
    let Y = unroll_for! (j, b) in u32:0..i {
      let i = i + j;
      a + b + i
    } (u32:1);
    a + i + Y
  } (u32:0);
  let Z : u32[X] = [0, ...];
}
)",
      TypecheckSucceeds(AllOf(
          HasNodeWithType("X", "uN[32]"), HasNodeWithType("Z", "uN[32][567]"),
          HasRepeatedNodeWithType("let i = i + j;", "uN[32]", 10))));
}

TEST(TypecheckV2Test, UnrollForCompositeType) {
  EXPECT_THAT(
      R"(
type Tp = (s32, (u8, s16), u2);
struct St { x: u32, y: s64 }
const A : Tp[3] = [(1, (2, 3), 0), (1, (2, -3), 1), (4, (5, 6), 2)];
fn foo() {
  let X = unroll_for! ((i, (j, k), _), a) in A {
    St { x: a.x + (i as u32), y: a.y + (j as s64) + (k as s64)}
  } (St { x: u32:0, y: s64:0 } );
  let Y : u32[X.x + (X.y as u32)] = [0, ...];
}
)",
      TypecheckSucceeds(AllOf(
          HasNodeWithType("X", "St { x: uN[32], y: sN[64] }"),
          HasNodeWithType("Y", "uN[32][21]"),
          HasRepeatedNodeWithType(
              "St { x: a.x + (i as u32), y: a.y + (j as s64) + (k as s64) }",
              "St { x: uN[32], y: sN[64] }", 3))));
}

TEST(TypecheckV2Test, UnrollForNoReturnValue) {
  EXPECT_THAT(
      R"(
fn foo() {
  unroll_for! (i, _) in u32:0..5 {
    let _ = i;
  } (());
}
)",
      TypecheckSucceeds(HasRepeatedNodeWithType("let _ = i;", "uN[32]", 5)));
}

TEST(TypecheckV2Test, ProcWithFifoDepth) {
  EXPECT_THAT(R"(
proc Counter {
  c: chan<u32> out;
  max: u32;
  init { 0 }
  config(c: chan<u32> out, max: u32) {
    (c, max)
  }
  next(i: u32) {
    send(join(), c, i);
    if i == max { i } else { i + 1 }
  }
}

proc main {
  c: chan<u32> in;
  init { (join(), 0) }
  config() {
    let (p, c) = chan<u32, 11>("my_chan");
    spawn Counter(p, 50);
    (c,)
  }
  next(state: (token, u32)) {
    recv(state.0, c)
  }
}
)",
              TypecheckSucceeds(HasNodeWithType("11", "uN[32]")));
}

TEST(TypecheckV2Test, ProcWithInvalidTypedFifoDepth) {
  EXPECT_THAT(R"(
proc Counter {
  c: chan<u32> out;
  max: u32;
  init { 0 }
  config(c: chan<u32> out, max: u32) {
    (c, max)
  }
  next(i: u32) {
    send(join(), c, i);
    if i == max { i } else { i + 1 }
  }
}

proc main {
  c: chan<u32> in;
  init { (join(), 0) }
  config() {
    let (p, c) = chan<u32, u32[2]:[1, 2]>("my_chan");
    spawn Counter(p, 50);
    (c,)
  }
  next(state: (token, u32)) {
    recv(state.0, c)
  }
}
)",
              TypecheckFails(HasTypeMismatch("u32", "u32[2]")));
}

TEST(TypecheckV2Test, SendIfWithHiddenChannel) {
  EXPECT_THAT(
      R"(
proc Counter<T: u32> {
  c: chan<uN[T]> out;
  max: u32;
  init { u32:0 }
  config(c: chan<u32> out, max: u32) {
    (c, max)
  }

  next(i: u32) {
    let c = i;
    let z = send_if(join(), c, true, c);
    if i == max { i } else { i + u32:1 }
  }
}

proc main {
  c: chan<u32> in;
  init { (join(), u32:0) }
  config() {
    let (p, c) = chan<u32>("my_chan");
    spawn Counter<32>(p, u32:50);
    (c,)
  }
  next(state: (token, u32)) {
    recv(state.0, c);
  }
}
)",
      TypecheckFailsWithPayload(HasTypeMismatch("chan<Any>", "u32"),
                                AllOf(HasFilenameInSpan("builtin_stubs.x"),
                                      HasFilenameInSpan("fake.x"))));
}

TEST(TypecheckV2Test, ImportConstantUnifiesTypes) {
  constexpr std::string_view kImported = R"(
pub const SOME_CONSTANT = uN[32]:1;
)";
  constexpr std::string_view kProgram = R"(
import imported;

fn main() -> uN[32] {
  let var:u32 = imported::SOME_CONSTANT;
  var
})";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data).status());
  EXPECT_THAT(TypecheckV2(kProgram, "main", &import_data),
              IsOkAndHolds(HasTypeInfo(AllOf(
                  HasNodeWithType("var", "uN[32]"),
                  HasNodeWithType("imported::SOME_CONSTANT", "uN[32]")))));
}

TEST(TypecheckV2Test, ExplicitParametricExpressionMismatchingBindingTypeFails) {
  EXPECT_THAT(R"(
const X = u32:1;
fn foo<N: u32>(a: uN[N]) -> uN[N] { a }
const Y = foo<{X == 1}>(s4:3);
)",
              TypecheckFails(HasSizeMismatch("bool", "u32")));
}

TEST(TypecheckV2Test, IfTypeMismatch) {
  EXPECT_THAT("const X: u31 = if true { u32:1 } else { u32:0 };",
              TypecheckFails(HasSizeMismatch("u32", "u31")));
}

TEST(TypecheckV2Test, ElseIfMismatch) {
  EXPECT_THAT(R"(
const X = if false {
    u32:1
} else if true {
    u31:2
} else {
    u32:3
};)",
              TypecheckFails(HasSizeMismatch("u31", "u32")));
}

TEST(TypecheckV2Test, MatchArm) {
  EXPECT_THAT(R"(
const X = u32:1;
const Y = u32:2;
const Z = match X {
  u32:1 => X,
  _ => Y
};
)",
              TypecheckSucceeds(HasNodeWithType("Z", "uN[32]")));
}

TEST(TypecheckV2Test, MatchArmUnifiedWithTypeAliasInLaterArm) {
  EXPECT_THAT(R"(
const X = u32:1;
const Y = u32:2;
const Z = match X {
  u32:1 => u3:2,
  _ => {
    type InternalType = uN[Y + 1];
    InternalType:3
  }
};
)",
              TypecheckSucceeds(HasNodeWithType("Z", "uN[3]")));
}

TEST(TypecheckV2Test, MatchArmMismatch) {
  EXPECT_THAT(R"(
const X = u32:1;
const Y = u31:2;
const Z = match X {
  u32:1 => X,
  _ => Y
};
)",
              TypecheckFails(HasSizeMismatch("u31", "u32")));
}

TEST(TypecheckV2Test, MatchArmWithConversionForUnification) {
  // Based on https://github.com/google/xls/issues/2379.
  EXPECT_THAT(R"(
fn lsb<S: bool, N: u32>(x: xN[S][N]) -> u1 { x as u1 }
fn repro(x: u3) -> u2 {
    let (_dummy, upper) = match x {
        u3:0b000 | u3:0b001 => (u8:0, 0b00),
        u3:0b010 | u3:0b011 => (u8:0, lsb(x) ++ u1:0),
        _ => (u8:0, x[0 +: u2]),
    };
    upper
}
  )",
              TypecheckSucceeds(HasNodeWithType("upper", "uN[2]")));
}

TEST(TypecheckV2Test, MatchArmWithConversionForUnificationAndLetDep) {
  EXPECT_THAT(R"(
fn lsb<S: bool, N: u32>(x: xN[S][N]) -> u1 { x as u1 }
fn repro(x: u3) -> u2 {
    let (_dummy, upper) = match x {
        u3:0b000 | u3:0b001 => (u8:0, 0b00),
        u3:0b010 | u3:0b011 => {
          let a = s32:0;
          (u8:0, lsb(x[a:]) ++ u1:0)
        },
        _ => (u8:0, x[0 +: u2]),
    };
    upper
}
  )",
              TypecheckSucceeds(HasNodeWithType("upper", "uN[2]")));
}

TEST(TypecheckV2Test, MatchArmWithConversionForUnificationAndLetDepChain) {
  EXPECT_THAT(R"(
fn lsb<S: bool, N: u32>(x: xN[S][N]) -> u1 { x as u1 }
fn repro(x: u3) -> u2 {
    let (_dummy, upper) = match x {
        u3:0b000 | u3:0b001 => (u8:0, 0b00),
        u3:0b010 | u3:0b011 => {
          let a = s32:0;
          let b = a;
          (u8:0, lsb(x[b:]) ++ u1:0)
        },
        _ => (u8:0, x[0 +: u2]),
    };
    upper
}
  )",
              TypecheckSucceeds(HasNodeWithType("upper", "uN[2]")));
}

TEST(TypecheckV2Test, ConditionalWithConversionForUnificationAndConstForDep) {
  EXPECT_THAT(R"(
fn lsb<S: bool, N: u32>(x: xN[S][N]) -> u1 { x as u1 }
fn repro(x: u3) -> u2 {
    let (_dummy, upper) = if x > 3 {
      (u8:0, 0b00)
    } else {
      let c = const for (i, a) in s32:0..10 {
          let ignored_const = s32:2;
          a + i + ignored_const
      }(s32:0);
      let d = c;
      (u8:0, lsb(x[d:]) ++ u1:0)
    };
    upper
}
  )",
              TypecheckSucceeds(HasNodeWithType("upper", "uN[2]")));
}

TEST(TypecheckV2Test, ConstMatch) {
  EXPECT_THAT(R"(
fn main(a: u32, b: u32) -> u32 {
    const A = true;
    let retval = const match A {
        true => a,
        false => b
    };
    retval
}
  )",
              TypecheckSucceeds(HasNodeWithType("retval", "uN[32]")));
}

TEST(TypecheckV2Test, ConstMatchWithDifferentTypes) {
  EXPECT_THAT(R"(
fn match_test<N: u32>() -> uN[N] {
  const match N {
    u32:1 => u1:0,
        _ => zero!<uN[N]>()
  }
}

fn main() {
  let first = match_test<u32:1>();
  let second = match_test<u32:8>();
}
  )",
              TypecheckSucceeds(AllOf(HasNodeWithType("first", "uN[1]"),
                                      HasNodeWithType("second", "uN[8]"))));
}

TEST(TypecheckV2Test, ConstMatchWithParametricMatched) {
  EXPECT_THAT(R"(
fn match_test<cond: bool>() -> u32 {
    const A = u32:1;
    const B = u32:2;
    const match cond {
        true => A,
        _ => B
    }
}

fn main() {
    let retval = match_test<true>();
}
  )",
              TypecheckSucceeds(HasNodeWithType("retval", "uN[32]")));
}

TEST(TypecheckV2Test, ConstMatchUnableToEvaluate) {
  constexpr std::string_view kProgram = R"(
fn main(cond: u32) -> bool {
    const match cond {
        u32:0 => u32:0,
        _ => u32:1
    }
}
)";
  EXPECT_THAT(TypecheckV2(kProgram),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("No constexpr value found for node `cond`")));
}

TEST(TypecheckV2Test, MatchMismatch) {
  EXPECT_THAT(R"(
const X = u32:1;
const Y = u32:2;
const Z:u31 = match X {
  u32:1 => X,
  _ => Y
};
)",
              TypecheckFails(HasSizeMismatch("u32", "u31")));
}

TEST(TypecheckV2Test, MatchArmDuplicated) {
  EXPECT_THAT(R"(
const X = u32:1;
const Y = u32:2;
const Z = match X {
  u32:1 => X,
  u32:0 => X,
  u32:1 => Y,
  _ => Y
};
)",
              TypecheckFails(
                  HasSubstr("Exact-duplicate pattern match detected `u32:1`")));
}

TEST(TypecheckV2Test, MatchNonExhaustive) {
  EXPECT_THAT(R"(
fn f(x: u1) -> u32 {
    match x {
        true => u32:64,
    }
}
)",
              TypecheckFails(HasSubstr("Match patterns are not exhaustive")));
}

TEST(TypecheckV2Test, MatchAlreadyExhaustive) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
fn f(x: u4) -> u32 {
    match x {
        u4:0..u4:15 => u32:64,
        u4:15 => u32:32,
        u4:2 => u32:42,
    }
}
)"));
  ASSERT_THAT(result.tm.warnings.warnings().size(), 1);
  EXPECT_EQ(result.tm.warnings.warnings()[0].message,
            "Match is already exhaustive before this pattern");
}

TEST(TypecheckV2Test, PatternMatch) {
  EXPECT_THAT(R"(
fn f(t: (u8, u32)) -> u32 {
    match t {
        (42, y) => y,
        (_, y) => y + 1,
    }
}

fn main() {
    const VAL = f((42, 10));
    let res = uN[VAL]:0;

    let val2 = f((3, 10));
    let res2 = uN[val2]:0;
}
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("res", "uN[10]"),
                                      HasNodeWithType("res2", "uN[11]"),
                                      HasNodeWithType("y", "uN[32]"))));
}

TEST(TypecheckV2Test, PatternMatchToConstant) {
  EXPECT_THAT(R"(
const MY_FAVORITE_NUMBER = u8:42;

fn f(t: (u8, u32)) -> u32 {
    match t {
        (MY_FAVORITE_NUMBER, y) => y,
        (_, y) => y + 77,
    }
}

fn main() {
    const VAL = f((42, 10));
    let res = uN[VAL]:0;
}
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("res", "uN[10]"),
                                      HasNodeWithType("y", "uN[32]"))));
}

TEST(TypecheckV2Test, PatternMatchNested) {
  EXPECT_THAT(R"(
const MY_FAVORITE_NUMBER = u8:42;

fn f(t: (u8, (u16, u32))) -> u32 {
    match t {
        (MY_FAVORITE_NUMBER, (y, z)) => y as u32 + z,
        (_, (y, 42)) => y as u32,
        _ => 7,
    }
}

fn main() {
    const VAL = f((42, (10, 10))); // Returns 20
    let res = uN[VAL]:0;

    const VAL2 = f((40, (10, MY_FAVORITE_NUMBER as u32))); // Returns 10
    let res2 = uN[VAL2]:0;

    const VAL3 = f((40, (10, 0))); // Returns 7
    let res3 = uN[VAL3]:0;
}
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("res", "uN[20]"),
                                      HasNodeWithType("res2", "uN[10]"),
                                      HasNodeWithType("res3", "uN[7]"))));
}

TEST(TypecheckV2Test, PatternMatchWithRange) {
  EXPECT_THAT(
      R"(
fn f(x: u32) -> u32 {
    match x {
        1..3 => u32:1,
        _ => x,
    }
}

fn main() {
    let n = f(2);
    let res = uN[n]:0;

    let m = f(u32:6);
    let res2 = uN[m]:0;

}
)",
      TypecheckSucceeds(AllOf(
          HasNodeWithType("res", "uN[1]"), HasNodeWithType("res2", "uN[6]"),
          HasNodeWithType("1", "uN[32]"), HasNodeWithType("3", "uN[32]"),
          HasNodeWithType("1..3", "uN[32]"))));
}

TEST(TypecheckV2Test, PatternMatchWithInvalidRange) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
fn f(x: u32) -> u32 {
    match x {
        3..1 => u32:1,
        _ => x,
    }
}
)"));
  ASSERT_THAT(result.tm.warnings.warnings().size(), 1);
  EXPECT_EQ(result.tm.warnings.warnings()[0].message,
            "`3..1` from `u32:3` to `u32:1` is an empty range");
}

TEST(TypecheckV2Test, PatternMatchWithRangeInclusiveEnd) {
  XLS_EXPECT_OK(TypecheckV2(
      R"(
fn f(x: u32) -> u32 {
  match x {
    1..=3 => u32:1,
    _ => u32:2,
  }
}

const_assert!(f(0) == u32:2);
const_assert!(f(1) == u32:1);
const_assert!(f(2) == u32:1);
const_assert!(f(3) == u32:1);
const_assert!(f(4) == u32:2);
)"));
}

TEST(TypecheckV2Test, PatternMatchWithRangeInclusiveEndOneElement) {
  XLS_EXPECT_OK(TypecheckV2(
      R"(
fn f(x: u32) -> u32 {
  match x {
      1..=1 => u32:1,
      _ => u32:2,
  }
}

const_assert!(f(0) == u32:2);
const_assert!(f(1) == u32:1);
const_assert!(f(2) == u32:2);
)"));
}

TEST(TypecheckV2Test, PatternMatchWithRangeInclusiveEndMultipleMatches) {
  XLS_EXPECT_OK(TypecheckV2(
      R"(
fn f(x: u32) -> u32 {
  match x {
      1..=5 | 8..=9 => u32:1,
      _ => u32:2,
  }
}

const_assert!(f(5) == u32:1);
const_assert!(f(6) == u32:2);
const_assert!(f(8) == u32:1);
const_assert!(f(9) == u32:1);
const_assert!(f(10) == u32:2);
)"));
}

TEST(TypecheckV2Test, PatternMatchWithRangeInclusiveEndMixedType) {
  XLS_EXPECT_OK(TypecheckV2(
      R"(
fn f(x: u32) -> u32 {
  match x {
      1..=5 | 8..9 | 6 => u32:1,
      _ => u32:2,
  }
}

const_assert!(f(5) == u32:1);
const_assert!(f(6) == u32:1);
const_assert!(f(8) == u32:1);
const_assert!(f(9) == u32:2);
const_assert!(f(10) == u32:2);
)"));
}

TEST(TypecheckV2Test, PatternMatchWithConditional) {
  EXPECT_THAT(R"(
fn f(x: u2) -> u32 {
    match x {
        0..1 | 3 => 42,
        _ => 10,
    }
}

fn main() {
    let n = f(u2:3);
    let res = uN[n]:0;

    let m = f(u2:2);
    let res2 = uN[m]:0;

    let o = f(u2:0);
    let res3 = uN[o]:0;
}
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("res", "uN[42]"),
                                      HasNodeWithType("res2", "uN[10]"),
                                      HasNodeWithType("res3", "uN[42]"))));
}

TEST(TypecheckV2Test, PatternMatchWithParametric) {
  EXPECT_THAT(R"(
fn f<N: u8>(t: (u8, u32)) -> u32 {
    match t {
        (N, y) => y,
        (_, y) => y + 1,
    }
}

fn main() {
    const VAL = f<u8:2>((2, 10));
    let res = uN[VAL]:0;

    let val2 = f<u8:2>((3, 10));
    let res2 = uN[val2]:0;
}
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("res", "uN[10]"),
                                      HasNodeWithType("res2", "uN[11]"))));
}

TEST(TypecheckV2Test, PatternMismatch) {
  EXPECT_THAT(R"(fn f(t: (u8, u32)) -> u32 {
    match t {
        (u3:1, y) => y,
        (_, y) => y + 1,
    }
}
)",
              TypecheckFails(HasSizeMismatch("u3", "u8")));
}

TEST(TypecheckV2Test, PatternMatcherWrongType) {
  EXPECT_THAT(R"(fn f(t: (u8, u32)) -> u32 {
    match t {
        42 => 0,
        y => y + 1,
    }
}
)",
              TypecheckFails(HasTypeMismatch("(u8, u32)", "uN[6]")));
}

TEST(TypecheckV2Test, MatchWithNoArms) {
  EXPECT_THAT(R"(
const X = u32:0;
const Z = match X {};
)",
              TypecheckFails(HasSubstr("`match` expression has no arms")));
}

TEST(TypecheckV2Test, UnassignedReturnValueTypeMismatch) {
  EXPECT_THAT(
      R"(
fn ignored() -> u31 { u31:0 }

fn main(x: u32) -> u32 {
  ignored() + x;
  u32:1
}
)",
      TypecheckFails(HasSizeMismatch("u31", "u32")));
}

TEST(TypecheckV2Test, UnassignedReturnValueTypeMismatchParametric) {
  EXPECT_THAT(
      R"(
fn ignored<N:u32>() -> uN[N] { zero!<uN[N]>() }

fn main(x: u32) -> u32 {
  ignored<u32:31>() + x;
  u32:1
}
)",
      TypecheckFails(HasSizeMismatch("uN[31]", "u32")));
}

TEST(TypecheckV2Test, LetWithSizeMismatch) {
  EXPECT_THAT(
      R"(
fn f() -> u4 {
  let x = u32:5000;
  x
}
)",
      TypecheckFails(HasSizeMismatch("u32", "u4")));
}

TEST(TypecheckV2Test, ConstAssertMismatchFails) {
  EXPECT_THAT(
      R"(
const_assert!(4);
)",
      TypecheckFails(HasSizeMismatch("u3", "bool")));
}

TEST(TypecheckV2Test, SliceWithBoundTypeMismatchFails) {
  EXPECT_THAT(R"(
const X = s4:0;
const Y = s5:2;
const Z = 0b100111[X:Y];
)",
              TypecheckFails(HasSizeMismatch("s4", "s5")));
}

TEST(TypecheckV2Test, ValueTypeParametricMismatch) {
  EXPECT_THAT(
      R"(
  fn fake_decode<N: u32>(x: uN[N]) -> uN[N] { x }

  const Y = fake_decode<u32>(u32:1);)",
      TypecheckFails(HasSubstr("Expected parametric value, saw `u32`")));
}

TEST(TypecheckV2Test, ForTypeAnnotationInitMismatch) {
  EXPECT_THAT(
      R"(
fn foo(A : s32[5]) {
  let X = for (i, a) : (u32, s32) in A {
    a + i
  } (0);
}
)",
      TypecheckFails(HasSignednessMismatch("s32", "u32")));
}

TEST(TypecheckV2Test, ForTypeAnnotationIterableMismatch) {
  EXPECT_THAT(
      R"(
fn foo(A : s32[5]) {
  let X = for (i, a) : (u32, u32) in A {
    a + i
  } (0);
}
)",
      TypecheckFails(HasSignednessMismatch("u32", "s32")));
}

TEST(TypecheckV2Test, ForInitIterableTypeMismatch) {
  EXPECT_THAT(
      R"(
fn foo() {
  let X = for (i, a) in u32:0..u32:5 {
    a + i
  } (u16:0);
}
)",
      TypecheckFails(HasTypeMismatch("u32", "u16")));
}

TEST(TypecheckV2Test, ForBodyTypeMismatch) {
  EXPECT_THAT(
      R"(
fn foo() {
  let X = for (i, a) in 0..5 {
    let b : u64 = 0;
    b + i
  } (u32:0);
}
)",
      TypecheckFails(HasSizeMismatch("u32", "u64")));
}

TEST(TypecheckV2Test, EnumValueTypeMismatch) {
  EXPECT_THAT(
      R"(
enum MyEnum {
  A = u8:1,
  B = u16:2,
}
)",
      TypecheckFails(HasTypeMismatch("u8", "u16")));
}

TEST(TypecheckV2Test, EnumAnnotationTypeMismatch) {
  EXPECT_THAT(
      R"(
enum MyEnum : u8 {
  A = u16:1,
}
)",
      TypecheckFails(HasTypeMismatch("u8", "u16")));
}

TEST(TypecheckV2Test, SpawnWithTypeMismatchFails) {
  EXPECT_THAT(R"(
proc Counter {
  max: u32;
  init { () }
  config(max: u32) {
    (max,)
  }
  next(state: ()) { () }
}

proc main {
  init { () }
  config() {
    spawn Counter(u64:5);
    ()
  }
  next(state: ()) { () }
}
)",
              TypecheckFails(HasSizeMismatch("u32", "u64")));
}

TEST(TypecheckV2Test, ProcWithInitNextTypeMismatchFails) {
  EXPECT_THAT(
      R"(
proc main {
  init { u32:0 }
  config() { () }
  next(state: ()) { () }
}
)",
      TypecheckFails(HasTypeMismatch("u32", "()")));
}

TEST(TypecheckV2Test, ProcWithConfigReturnTypeMismatchFails) {
  EXPECT_THAT(
      R"(
proc main {
  max: u32;
  init { () }
  config() { (u64:5,) }
  next(state: ()) { () }
}
)",
      TypecheckFails(HasTypeMismatch("u32", "u64")));
}

TEST(TypecheckV2Test, ProcConfigMismatchingChannelTypes) {
  EXPECT_THAT(
      R"(
proc Proc {
  req: chan<()> in;
  resp: chan<()> out;
  data_in: chan<u32> in;
  data_out: chan<u32> out;
  config(data_in: chan<u32> in, data_out: chan<u32> out) {
    let (resp, req) = chan<()>("io");
    (data_in, data_out, req, resp)
  }
  init { () }
  next(state: ()) { () }
}

)",
      TypecheckFails(HasTypeMismatch("()", "u32")));
}

TEST(TypecheckV2Test, ProcConfigBranchedChannelTypeMismatch) {
  EXPECT_THAT(
      R"(
const A = u32:4;
proc Proc {
  input: chan<u32> in;
  output: chan<u32> out;
  config() {
    const if A == u32:5 {
      let (first_output, first_input) = chan<u32>("first");
      (first_input, first_output)
    } else {
      let (second_output, second_input) = chan<()>("second");
      (second_input, second_output)
    }
  }
  init { () }
  next(state: ()) { () }
}

)",
      TypecheckFails(HasTypeMismatch("()", "u32")));
}

TEST(TypecheckV2Test, ImportConstantSizeMismatch) {
  constexpr std::string_view kImported = R"(
pub const SOME_CONSTANT = u32:1;
)";
  constexpr std::string_view kProgram = R"(
import imported;

fn main() -> u1 {
  imported::SOME_CONSTANT
})";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data).status());
  EXPECT_THAT(TypecheckV2(kProgram, "main", &import_data),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSizeMismatch("u32", "u1")));
}

TEST(TypecheckV2Test, AssignImportedConstantTypeMismatch) {
  constexpr std::string_view kImported = R"(
pub const SOME_CONSTANT = s32:1;
)";
  constexpr std::string_view kProgram = R"(
import imported;

fn main() -> u1 {
  let var:u5 = imported::SOME_CONSTANT;
  uN[var]:0
})";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data).status());
  EXPECT_THAT(TypecheckV2(kProgram, "main", &import_data),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasTypeMismatch("s32", "u5")));
}

TEST(TypecheckV2Test, ImportConstantTypeMismatch) {
  constexpr std::string_view kImported = R"(
pub const SOME_CONSTANT = s32:1;
)";
  constexpr std::string_view kProgram = R"(
import imported;

fn main() -> u32 {
  imported::SOME_CONSTANT
})";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data).status());
  EXPECT_THAT(TypecheckV2(kProgram, "main", &import_data),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasTypeMismatch("s32", "u32")));
}

TEST(TypecheckV2Test, ImportConstantFromVfsWithSizeMismatch) {
  constexpr std::string_view kImported = R"(
pub const MY_CONSTANT: u30 = 42;
)";
  constexpr std::string_view kProgram = R"(
import imported;

fn f() -> u1 {
  imported::MY_CONSTANT
}
)";
  absl::flat_hash_map<std::filesystem::path, std::string> files = {
      {std::filesystem::path("/imported.x"), std::string(kImported)},
      {std::filesystem::path("/fake_main_path.x"), std::string(kProgram)},
  };
  auto vfs = std::make_unique<FakeFilesystem>(
      files, /*cwd=*/std::filesystem::path("/"));
  ImportData import_data = CreateImportDataForTest(std::move(vfs));
  EXPECT_THAT(TypecheckV2(kProgram, "fake_main_path", &import_data),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSizeMismatch("u30", "u1")));
}

TEST(TypecheckV2Test, UseConstantTypeMismatch) {
  constexpr std::string_view kImported = R"(
pub const SOME_CONSTANT = s32:1;
)";
  constexpr std::string_view kProgram = R"(#![feature(use_syntax)]
use imported::SOME_CONSTANT;

fn main() -> u32 {
  SOME_CONSTANT
})";
  absl::flat_hash_map<std::filesystem::path, std::string> files = {
      {std::filesystem::path("/imported.x"), std::string(kImported)},
      {std::filesystem::path("/fake_main_path.x"), std::string(kProgram)},
  };
  auto vfs = std::make_unique<FakeFilesystem>(
      files, /*cwd=*/std::filesystem::path("/"));
  ImportData import_data = CreateImportDataForTest(std::move(vfs));
  EXPECT_THAT(TypecheckV2(kProgram, "fake_main_path", &import_data),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasTypeMismatch("s32", "u32")));
}

TEST(TypecheckV2Test, MatchWithInvocationInExpression) {
  EXPECT_THAT(
      R"(
enum Tag : u3 {
    ZERO = 0,
    POSITIVE = 1,
}

fn tag(input: u32) -> Tag {
    match input {
        0 => Tag::ZERO,
        _ => Tag::POSITIVE,
    }
}

fn tuple_match(input: u32, sign: bool) -> u32 {
    let input_shifted = match (sign, tag(input)) {
        (true, Tag::ZERO) => input >> 1,
        (true, Tag::POSITIVE) => input >> 2,
        (false, _) => input,
    };
    input_shifted as u32
}
)",
      TypecheckSucceeds(
          HasNodeWithType("tuple_match", "(uN[32], uN[1]) -> uN[32]")));
}

}  // namespace
}  // namespace xls::dslx
