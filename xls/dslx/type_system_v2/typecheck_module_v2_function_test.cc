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

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system/typecheck_test_utils.h"
#include "xls/dslx/type_system_v2/matchers.h"
#include "xls/dslx/virtualizable_file_system.h"

// Tests for function definitions, calls, return types, and parametric
// functions.
namespace xls::dslx {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::AllOf;
using ::testing::HasSubstr;

TEST(TypecheckV2Test, ComparisonAsFunctionArgument) {
  EXPECT_THAT(R"(
fn foo(a: bool) -> bool { a }
const Y = foo(1 != 2);
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("1 != 2", "uN[1]"),
                                      HasNodeWithType("1", "uN[2]"),
                                      HasNodeWithType("2", "uN[2]"))));
}

TEST(TypecheckV2Test, FunctionCallReturningNothing) {
  EXPECT_THAT(
      R"(
fn foo() { () }
const Y = foo();
)",
      TypecheckSucceeds(AllOf(HasOneLineBlockWithType("()", "()"),
                              HasNodeWithType("const Y = foo();", "()"))));
}

TEST(TypecheckV2Test, FunctionCallReturningInteger) {
  EXPECT_THAT(
      R"(
fn foo() -> u32 { 3 }
const Y = foo();
)",
      TypecheckSucceeds(AllOf(HasOneLineBlockWithType("3", "uN[32]"),
                              HasNodeWithType("const Y = foo();", "uN[32]"))));
}

TEST(TypecheckV2Test, FunctionCallReturningBool) {
  EXPECT_THAT(
      R"(
fn foo() -> bool { true }
const Y = foo();
)",
      TypecheckSucceeds(AllOf(HasOneLineBlockWithType("true", "uN[1]"),
                              HasNodeWithType("const Y = foo();", "uN[1]"))));
}

TEST(TypecheckV2Test, FunctionCallReturningFunctionCall) {
  EXPECT_THAT(
      R"(
fn bar() -> s32 { 123 }
fn foo() -> s32 { bar() }
const Y = foo();
)",
      TypecheckSucceeds(AllOf(HasOneLineBlockWithType("123", "sN[32]"),
                              HasOneLineBlockWithType("bar()", "sN[32]"),
                              HasNodeWithType("const Y = foo();", "sN[32]"))));
}

TEST(TypecheckV2Test, ParametricFunctionInWrongContextWithoutInvocation) {
  EXPECT_THAT(
      R"(
fn foo<N: u32>() -> uN[N] { 3 }
const Y: u32 = foo;
)",
      TypecheckFails(HasSubstr("Expected type `uN[32]` but got `foo`, which is "
                               "a parametric function not being invoked")));
}

TEST(TypecheckV2Test, SumOfLiteralsAndParametricFunctionCall) {
  EXPECT_THAT(
      R"(
fn foo<N: u32>() -> uN[N] { 3 }
const Y = 1 + 2 + 3 + foo<32>();
)",
      TypecheckSucceeds(HasNodeWithType("Y", "uN[32]")));
}

TEST(TypecheckV2Test, ParametricFunctionWithParametricBinopParamType) {
  // The point here is to ensure that we solve for A and B by matching against
  // `uN[A][B][C]`, and the unsolvable `A * B` doesn't derail the solution of
  // any parametrics.
  EXPECT_THAT(
      R"(
fn foo<A: u32, B: u32, C: u32, D: u32>(
    _values: uN[A * B][D],
    _want: uN[A][B][C],
) -> uN[A][B][C][D] {
    zero!<uN[A][B][C][D]>()
}

const E = u32:12;
const F = u32:6;

fn main() {
    let values1 = zero!<uN[12][6]>();
    let result1 = foo(values1, zero!<uN[3][4][5]>());
    let values2 = zero!<uN[E][F]>();
    let result2 = foo(values2, zero!<uN[3][4][5]>());
}
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("result1", "uN[3][4][5][6]"),
                              HasNodeWithType("result2", "uN[3][4][5][6]"))));
}

TEST(TypecheckV2Test, FunctionReturningMismatchingIntegerAutoTypeFails) {
  EXPECT_THAT(R"(
fn foo() -> u4 { 65536 }
const Y = foo();
)",
              TypecheckFails(HasSizeMismatch("u17", "u4")));
}

TEST(TypecheckV2Test, FunctionReturningTooLargeExplicitTypeFails) {
  EXPECT_THAT(R"(
const X = u32:65536;
fn foo() -> u4 { X }
const Y = foo();
)",
              TypecheckFails(HasSizeMismatch("u32", "u4")));
}

TEST(TypecheckV2Test, FunctionReturningIntegerWithWrongSignednessFails) {
  EXPECT_THAT(R"(
const X = s32:65536;
fn foo() -> u32 { X }
const Y = foo();
)",
              TypecheckFails(HasSignednessMismatch("s32", "u32")));
}

TEST(TypecheckV2Test, FunctionCallReturningPassedInInteger) {
  EXPECT_THAT(
      R"(
fn foo(a: u32) -> u32 { a }
const Y = foo(4);
)",
      TypecheckSucceeds(AllOf(HasOneLineBlockWithType("a", "uN[32]"),
                              HasNodeWithType("const Y = foo(4);", "uN[32]"))));
}

TEST(TypecheckV2Test, FunctionCallReturningSumOfPassedInIntegers) {
  EXPECT_THAT(R"(
fn foo(a: u32, b: u32) -> u32 { a + b }
const Y = foo(4, 5);
)",
              TypecheckSucceeds(
                  AllOf(HasOneLineBlockWithType("a + b", "uN[32]"),
                        HasNodeWithType("const Y = foo(4, 5);", "uN[32]"))));
}

TEST(TypecheckV2Test, FunctionCallPassingInFunctionCalls) {
  EXPECT_THAT(
      R"(
fn foo(a: u32, b: u32) -> u32 { a + b }
const Y = foo(foo(3, 2), foo(4, 5));
)",
      TypecheckSucceeds(AllOf(
          HasOneLineBlockWithType("a + b", "uN[32]"),
          HasNodeWithType("const Y = foo(foo(3, 2), foo(4, 5));", "uN[32]"))));
}

TEST(TypecheckV2Test, FunctionCallPassingInSum) {
  EXPECT_THAT(R"(
const X: u32 = 4;
const Z: u32 = 5;
fn foo(a: u32) -> u32 { a }
const Y = foo(X + Z);
)",
              TypecheckSucceeds(
                  AllOf(HasOneLineBlockWithType("a", "uN[32]"),
                        HasNodeWithType("const Y = foo(X + Z);", "uN[32]"))));
}

TEST(TypecheckV2Test, FunctionCallWithUnusedResult) {
  EXPECT_THAT(R"(
fn foo(a: u32) -> u32 { a }

fn bar() -> u32 {
  foo(5);
  foo(6);
  7
}
  )",
              TypecheckSucceeds(AllOf(HasNodeWithType("foo(5)", "uN[32]"),
                                      HasNodeWithType("foo(6)", "uN[32]"))));
}

TEST(TypecheckV2Test, FunctionCallForwardingChannelParam) {
  EXPECT_THAT(R"(
fn foo(c: chan<u32> in) {}
fn bar(c: chan<u32> in) { foo(c); }
  )",
              TypecheckSucceeds(AllOf(HasNodeWithType("foo(c)", "()"))));
}

TEST(TypecheckV2Test, FunctionCallPassingInTooManyArgumentsFails) {
  EXPECT_THAT(R"(
fn foo(a: u4) -> u4 { a }
const Y:u32 = foo(1, 2);
)",
              TypecheckFails(HasSubstr("Expected 1 argument(s) but got 2.")));
}

TEST(TypecheckV2Test, FunctionCallPassingInTooFewArgumentsFails) {
  EXPECT_THAT(R"(
fn foo(a: u4, b: u4) -> u4 { a + b }
const Y:u32 = foo(1);
)",
              TypecheckFails(HasSubstr("Expected 2 argument(s) but got 1.")));
}

TEST(TypecheckV2Test, FunctionCallPassingInTooLargeAutoSizeFails) {
  EXPECT_THAT(R"(
fn foo(a: u4) -> u4 { a }
const Y = foo(32767);
)",
              TypecheckFails(HasSizeMismatch("u15", "u4")));
}

TEST(TypecheckV2Test, FunctionCallPassingInTooLargeExplicitIntegerSizeFails) {
  EXPECT_THAT(R"(
const X:u32 = 1;
fn foo(a: u4) -> u4 { a }
const Y = foo(X);
)",
              TypecheckFails(HasSizeMismatch("u32", "u4")));
}

TEST(TypecheckV2Test, FunctionCallPassingInWrongSignednessFails) {
  EXPECT_THAT(R"(
const X:u32 = 1;
fn foo(a: s32) -> s32 { a }
const Y = foo(X);
)",
              TypecheckFails(HasSignednessMismatch("u32", "s32")));
}

TEST(TypecheckV2Test, FunctionCallMismatchingLhsTypeFails) {
  EXPECT_THAT(R"(
fn foo(a: u4) -> u4 { a }
const Y:u32 = foo(1);
)",
              TypecheckFails(HasSizeMismatch("u4", "u32")));
}

TEST(TypecheckV2Test, FunctionCallToNonFunctionFails) {
  EXPECT_THAT(R"(
const X = u32:4;
const Y = X(1);
)",
              TypecheckFails(HasSubstr("callee `X` is not a function")));
}

TEST(TypecheckV2Test, ParametricFunctionCallWithTooManyParametricsFails) {
  EXPECT_THAT(R"(
fn foo<N: u32>() -> u32 { N }
const X = foo<3, 4>();
)",
              TypecheckFails(HasSubstr("Too many parametric values supplied")));
}

TEST(TypecheckV2Test, ParametricFunctionReturningIntegerParameter) {
  EXPECT_THAT(R"(
fn foo<N: u32>() -> u32 { N }
const X = foo<3>();
)",
              TypecheckSucceeds(
                  AllOf(HasNodeWithType("const X = foo<3>();", "uN[32]"))));
}

TEST(TypecheckV2Test, ParametricFunctionReturningIntegerOfParameterSize) {
  EXPECT_THAT(R"(
fn foo<N: u32>() -> uN[N] { 5 }
const X = foo<16>();
const Y = foo<17>();
)",
              TypecheckSucceeds(
                  AllOf(HasNodeWithType("const X = foo<16>();", "uN[16]"),
                        HasNodeWithType("const Y = foo<17>();", "uN[17]"))));
}

TEST(TypecheckV2Test, ParametricFunctionReturningIntegerOfNPlus1Size) {
  EXPECT_THAT(R"(
fn foo<N: u32>(a: uN[N + 1]) -> uN[N + 1] { a }
const X = foo<16>(1);
const Y = foo<17>(2);
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[17]"),
                                      HasNodeWithType("Y", "uN[18]"))));
}

TEST(TypecheckV2Test, ParametricFunctionReturningIntegerOfParameterSignedness) {
  EXPECT_THAT(R"(
fn foo<S: bool>() -> xN[S][32] { 5 }
const X = foo<false>();
const Y = foo<true>();
)",
              TypecheckSucceeds(
                  AllOf(HasNodeWithType("const X = foo<false>();", "uN[32]"),
                        HasNodeWithType("const Y = foo<true>();", "sN[32]"))));
}

TEST(TypecheckV2Test,
     ParametricFunctionReturningIntegerOfParameterSignednessAndSize) {
  EXPECT_THAT(R"(
fn foo<S: bool, N: u32>() -> xN[S][N] { 5 }
const X = foo<false, 10>();
const Y = foo<true, 11>();
)",
              TypecheckSucceeds(AllOf(
                  HasNodeWithType("const X = foo<false, 10>();", "uN[10]"),
                  HasNodeWithType("const Y = foo<true, 11>();", "sN[11]"))));
}

TEST(TypecheckV2Test, ParametricFunctionReturningIntegerOfCastedDifference) {
  EXPECT_THAT(R"(
fn f<A: s32, B: s32>(a: u32) -> uN[(B - A) as u32] {
   a as uN[(B - A) as u32]
}
const X = f<1, 3>(50);
const Y = f<1, 4>(50);
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[2]"),
                                      HasNodeWithType("Y", "uN[3]"))));
}

TEST(TypecheckV2Test, FunctionReturningIntegerOfSumOfInferredParametrics) {
  EXPECT_THAT(R"(
fn f<A: u32, B: u32>(a: uN[A], b: uN[B]) -> uN[A + B] {
   a as uN[A + B] + b as uN[A + B]
}
const X = f(u16:30, u8:40);
const Y = f(u32:30, u40:40);
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[24]"),
                                      HasNodeWithType("Y", "uN[72]"))));
}

TEST(TypecheckV2Test, ParametricFunctionTakingIntegerOfParameterizedSize) {
  EXPECT_THAT(R"(
fn foo<N: u32>(a: uN[N]) -> uN[N] { a }
const X = foo<10>(u10:5);
const Y = foo<11>(u11:5);
)",
              TypecheckSucceeds(AllOf(
                  HasNodeWithType("const X = foo<10>(u10:5);", "uN[10]"),
                  HasNodeWithType("const Y = foo<11>(u11:5);", "uN[11]"))));
}

TEST(TypecheckV2Test, ParametricFunctionWithNonInferrableParametric) {
  EXPECT_THAT(R"(
fn foo<M: u32, N: u32>(a: uN[M]) -> uN[M] { a }
const X = foo(u10:5);
)",
              TypecheckFails(HasSubstr(
                  "Could not infer parametric(s): N of function `foo`")));
}

TEST(TypecheckV2Test,
     ParametricFunctionTakingIntegerOfParameterizedSignedness) {
  EXPECT_THAT(R"(
fn foo<S: bool>(a: xN[S][32]) -> xN[S][32] { a }
const X = foo<false>(u32:5);
const Y = foo<true>(s32:5);
)",
              TypecheckSucceeds(AllOf(
                  HasNodeWithType("const X = foo<false>(u32:5);", "uN[32]"),
                  HasNodeWithType("const Y = foo<true>(s32:5);", "sN[32]"))));
}

TEST(TypecheckV2Test,
     ParametricFunctionTakingIntegerOfParameterizedSignednessAndSize) {
  EXPECT_THAT(
      R"(
fn foo<S: bool, N: u32>(a: xN[S][N]) -> xN[S][N] { a }
const X = foo<false, 10>(u10:5);
const Y = foo<true, 11>(s11:5);
)",
      TypecheckSucceeds(
          AllOf(HasNodeWithType("const X = foo<false, 10>(u10:5);", "uN[10]"),
                HasNodeWithType("const Y = foo<true, 11>(s11:5);", "sN[11]"))));
}

TEST(TypecheckV2Test,
     ParametricFunctionTakingIntegerOfDefaultParameterizedSize) {
  EXPECT_THAT(
      R"(
fn foo<N: u32 = {10}>(a: uN[N]) -> uN[N] { a }
const X = foo(u10:5);
)",
      TypecheckSucceeds(HasNodeWithType("const X = foo(u10:5);", "uN[10]")));
}

TEST(TypecheckV2Test,
     ParametricFunctionTakingIntegerOfOverriddenDefaultParameterizedSize) {
  EXPECT_THAT(R"(
fn foo<N: u32 = {10}>(a: uN[N]) -> uN[N] { a }
const X = foo<11>(u11:5);
)",
              TypecheckSucceeds(
                  HasNodeWithType("const X = foo<11>(u11:5);", "uN[11]")));
}

TEST(TypecheckV2Test,
     ParametricFunctionTakingIntegerWithDependentDefaultParametric) {
  EXPECT_THAT(R"(
fn foo<M: u32, N: u32 = {M + 1}>(a: uN[N]) -> uN[N] { a }
const X = foo<11>(u12:5);
)",
              TypecheckSucceeds(
                  HasNodeWithType("const X = foo<11>(u12:5);", "uN[12]")));
}

TEST(TypecheckV2Test,
     ParametricFunctionWithDefaultDependingOnInferredParametric) {
  EXPECT_THAT(
      R"(
fn foo<M: u32, N: u32 = {M + M}>(a: uN[M]) -> uN[M] { a }
const X = foo(u10:5);
)",
      TypecheckSucceeds(HasNodeWithType("const X = foo(u10:5);", "uN[10]")));
}

TEST(TypecheckV2Test,
     ParametricFunctionWithInferredThenDefaultThenInferredParametric) {
  EXPECT_THAT(
      R"(
fn foo<A: u32, B: u32 = {A + 1}, C: u32>(x: uN[A], y: uN[C][B]) -> uN[A] {
   x
}
const X = foo(u3:1, [u24:6, u24:7, u24:8, u24:9]);
)",
      TypecheckSucceeds(HasNodeWithType(
          "const X = foo(u3:1, [u24:6, u24:7, u24:8, u24:9]);", "uN[3]")));
}

TEST(TypecheckV2Test,
     ParametricFunctionTakingIntegerOfParameterizedSignednessAndSizeWithSum) {
  // The point here is to make sure that the uN[N] type annotation being
  // propagated onto a complex subtree in global scope is correctly dealt with.
  EXPECT_THAT(R"(
const X = u32:3;
const Y = u32:4;
fn foo<N: u32>(a: uN[N]) -> uN[N] { a }
const Z = foo<32>(X + Y + X + 50);
)",
              TypecheckSucceeds(HasNodeWithType(
                  "const Z = foo<32>(X + Y + X + 50);", "uN[32]")));
}

TEST(TypecheckV2Test,
     ParametricFunctionWithArgumentMismatchingParameterizedSizeFails) {
  EXPECT_THAT(R"(
fn foo<N: u32>(a: uN[N]) -> uN[N] { a }
const X = foo<10>(u11:5);
)",
              TypecheckFails(HasSizeMismatch("u11", "uN[10]")));
}

TEST(TypecheckV2Test,
     ParametricFunctionWithArgumentMismatchingParameterizedSignednessFails) {
  EXPECT_THAT(R"(
fn foo<S: bool>(a: xN[S][32]) -> xN[S][32] { a }
const X = foo<true>(u32:5);
)",
              TypecheckFails(HasSignednessMismatch("xN[1][32]", "u32")));
}

TEST(TypecheckV2Test, ParametricFunctionCallingAnotherParametricFunction) {
  EXPECT_THAT(R"(
fn bar<A: u32>(a: uN[A]) -> uN[A] { a + 1 }
fn foo<A: u32, B: u32>(a: uN[A]) -> uN[B] { bar<B>(2) }
const X = foo<24, 23>(4);
)",
              TypecheckSucceeds(
                  HasNodeWithType("const X = foo<24, 23>(4);", "uN[23]")));
}

TEST(TypecheckV2Test,
     ParametricFunctionCallingAnotherParametricFunctionMultiUse) {
  EXPECT_THAT(R"(
fn bar<A: u32>(a: uN[A]) -> uN[A] { a + 1 }
fn foo<A: u32>(a: uN[A]) -> uN[A] { bar<A>(a) }
const X = foo<24>(4);
const Y = foo<32>(5);
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[24]"),
                                      HasNodeWithType("Y", "uN[32]"))));
}

TEST(TypecheckV2Test, ParametricFunctionInvocationNesting) {
  EXPECT_THAT(R"(
fn foo<N: u32>(a: uN[N]) -> uN[N] { a + 1 }
const X = foo<24>(foo<24>(4) + foo<24>(5));
)",
              TypecheckSucceeds(HasNodeWithType(
                  "const X = foo<24>(foo<24>(4) + foo<24>(5));", "uN[24]")));
}

TEST(TypecheckV2Test,
     ParametricFunctionUsingGlobalConstantInParametricDefault) {
  EXPECT_THAT(R"(
const X = u32:3;
fn foo<M: u32, N: u32 = {M + X}>(a: uN[N]) -> uN[N] { a }
const Z = foo<12>(u15:1);
)",
              TypecheckSucceeds(
                  HasNodeWithType("const Z = foo<12>(u15:1);", "uN[15]")));
}

TEST(TypecheckV2Test,
     ParametricFunctionCallUsingGlobalConstantInParametricArgument) {
  EXPECT_THAT(
      R"(
fn foo<N: u32>(a: uN[N]) -> uN[N] { a }
const X = u32:3;
const Z = foo<X>(u3:1);
)",
      TypecheckSucceeds(HasNodeWithType("const Z = foo<X>(u3:1);", "uN[3]")));
}

TEST(TypecheckV2Test, ParametricFunctionCallFollowedByTypePropagation) {
  EXPECT_THAT(R"(
fn foo<N: u32>(a: uN[N]) -> uN[N] { a }
const Y = foo<15>(u15:1);
const Z = Y + 1;
)",
              TypecheckSucceeds(HasNodeWithType("const Z = Y + 1;", "uN[15]")));
}

TEST(TypecheckV2Test, UnaryOperatorInFunction) {
  EXPECT_THAT(
      R"(
fn foo(y: bool) -> bool {
  !y
}
)",
      TypecheckSucceeds(HasNodeWithType("!y", "uN[1]")));
}

TEST(TypecheckV2Test, UnaryOperatorOnInvalidTypeInFunction) {
  EXPECT_THAT(
      R"(
fn foo(y: (u32, u3)) -> (u32, u3) {
  !y
}

const F = foo((u32:5, u3:0));
)",
      TypecheckFails(HasSubstr(
          "Unary operations can only be applied to bits-typed operands.")));
}

TEST(TypecheckV2Test, LogicalBinopAsFnReturn) {
  EXPECT_THAT(R"(
fn foo(x: bool, y: bool) -> bool {
  x || y
}
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("x: bool", "uN[1]"),
                                      HasNodeWithType("y: bool", "uN[1]"),
                                      HasNodeWithType("x || y", "uN[1]"))));
}

TEST(TypecheckV2Test, LogicalBinopAsFnReturnWrongReturnType) {
  EXPECT_THAT(
      R"(
fn foo(x: bool, y: bool) -> u32 {
  x && y
}
)",
      TypecheckFails(HasSizeMismatch("bool", "u32")));
}

TEST(TypecheckV2Test, LogicalBinopAsFnReturnWrongLhsType) {
  EXPECT_THAT(R"(
fn foo(x: u32, y: bool) -> bool {
  x || y
}
)",
              TypecheckFails(HasSizeMismatch("u32", "bool")));
}

TEST(TypecheckV2Test, LogicalBinopAsFnReturnWrongParameterTypes) {
  EXPECT_THAT(R"(
fn foo(x: u32, y: u32) -> bool {
  x || y
}
)",
              TypecheckFails(HasSizeMismatch("u32", "bool")));
}

TEST(TypecheckV2Test, IfTestFnCall) {
  EXPECT_THAT(R"(
fn f() -> bool { true }
const X = if f() { u32:1 } else { u32:0 };
)",
              TypecheckSucceeds(HasNodeWithType("X", "uN[32]")));
}

TEST(TypecheckV2Test, IfTestBadFnCall) {
  EXPECT_THAT(R"(
fn f() -> u32 { u32:1 }
const X = if f() { u32:1 } else { u32:0 };
)",
              TypecheckFails(HasSizeMismatch("u32", "bool")));
}

TEST(TypecheckV2Test, FnReturnsIf) {
  EXPECT_THAT(R"(
fn f(x:u10) -> u32 { if x>u10:0 { u32:1 } else { u32:0 } }
const X = f(u10:1);
)",
              TypecheckSucceeds(HasNodeWithType("X", "uN[32]")));
}

TEST(TypecheckV2Test, CallFnWithIf) {
  EXPECT_THAT(R"(
fn f(x:u32) -> u32 { x }
const X = f(if true { u32:1 } else { u32:0 });
)",
              TypecheckSucceeds(HasNodeWithType("X", "uN[32]")));
}

TEST(TypecheckV2Test, ElseIfNotBool) {
  EXPECT_THAT(R"(const X = if false {
    u32:1
} else if u32:1 {
    u32:2
} else {
    u32:3
};)",
              TypecheckFails(HasSizeMismatch("u32", "bool")));
}

TEST(TypecheckV2Test, MatchArmFromFn) {
  EXPECT_THAT(R"(
fn f() -> u32 { u32:0 }
const X = u32:1;
const Y = u32:2;
const Z = match X {
  u32:1 => f(),
  _ => Y
};
)",
              TypecheckSucceeds(HasNodeWithType("Z", "uN[32]")));
}

TEST(TypecheckV2Test, MatchInFn) {
  EXPECT_THAT(R"(
fn f(a: u32) -> u32 {
  match a {
    u32:1 => a,
    _ => u32:0
  }
}
const Z = f(u32:1);
)",
              TypecheckSucceeds(HasNodeWithType("Z", "uN[32]")));
}

TEST(TypecheckV2Test, ImportedParametricFunctionWithConstantInSignature) {
  constexpr std::string_view kImported = R"(
pub struct Foo {
  value: u32
}

pub const C = Foo { value: 5 };

pub fn foo<N: u32>(a: uN[C.value]) -> uN[C.value] { a }
)";
  constexpr std::string_view kProgram = R"(
import imported;
const Y = imported::foo<1>(u5:2);
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data));
  EXPECT_THAT(TypecheckV2(kProgram, "main", &import_data),
              IsOkAndHolds(HasTypeInfo(HasNodeWithType("Y", "uN[5]"))));
}

TEST(TypecheckV2Test, RShiftAsFnReturn) {
  EXPECT_THAT(R"(
fn foo(x: u32, y: u2) -> u32 {
  x >> y
}
)",
              TypecheckSucceeds(HasNodeWithType("x >> y", "uN[32]")));
}

TEST(TypecheckV2Test, LShiftAsReturnFromParametricFn) {
  EXPECT_THAT(R"(
fn foo<N: u32>(x: u32) -> uN[N] {
  uN[N]:1 << x
}

const VAL = foo<u32:3>(u32:1);
)",
              TypecheckSucceeds(
                  HasNodeWithType("const VAL = foo<u32:3>(u32:1);", "uN[3]")));
}

TEST(TypecheckV2Test, LetInParametricFn) {
  EXPECT_THAT(R"(
fn f<N: u32>() -> uN[N] {
  const ZERO = uN[N]:0;
  ZERO
}

fn main() {
  let five_bits = f<5>();
  let four_bits = f<4>();
}
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("five_bits", "uN[5]"),
                                      HasNodeWithType("four_bits", "uN[4]"))));
}

TEST(TypecheckV2Test, ParametricConstantUsingParametricFunction) {
  EXPECT_THAT(R"(
struct S<N: u32> {}

fn f<N: u32>() -> u32 { N + 1 }

impl S<N> {
  const N_PLUS_1_VALUE = f<N>();
}

const X = uN[S<2>::N_PLUS_1_VALUE]:0;
const Y = uN[S<10>::N_PLUS_1_VALUE]:0;
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[3]"),
                                      HasNodeWithType("Y", "uN[11]"))));
}

TEST(TypecheckV2Test, ParametricFromFunctionUsedInConstantReference) {
  EXPECT_THAT(R"(
struct S<N: u32> {}

impl S<N> {
  const C = uN[N]:0;
}

fn f<N: u32>() -> uN[N] {
  S<N>::C
}

const C8 = f<8>();
const C9 = f<9>();
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("C8", "uN[8]"),
                                      HasNodeWithType("C9", "uN[9]"))));
}

TEST(TypecheckV2Test, TypeAliasInParametricFn) {
  EXPECT_THAT(R"(
fn f<T: u32>() -> uN[T] {
  type Ret = uN[T];
  Ret:0
}

fn main() {
  let x = f<8>();
  let y = f<15>();
}
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("x", "uN[8]"),
                                      HasNodeWithType("y", "uN[15]"))));
}

TEST(TypecheckV2Test, SliceOfNonBitsFails) {
  EXPECT_THAT(
      "const X = [u32:1, 2, 3][0:2];",
      TypecheckFails(HasSubstr("Value to slice is not of 'bits' type.")));
}

TEST(TypecheckV2Test, WidthSliceOfNonBitsFails) {
  EXPECT_THAT(
      "const X = [u32:1, u32:2, u32:3][0+:u2];",
      TypecheckFails(HasSubstr("Expected a bits-like type; got: `u32[3]`")));
}

TEST(TypecheckV2Test, UnrollForInParametricFunction) {
  EXPECT_THAT(
      R"(
fn factorial<N: u32>() -> u32 {
  unroll_for! (i, a) in 2..N + 1 {
    a * i
  } (1)
}

const X = factorial<3>();
const Y = factorial<4>();
const_assert!(X == 6);
const_assert!(Y == 24);
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[32]"),
                              HasNodeWithType("Y", "uN[32]"))));
}

TEST(TypecheckV2Test, ImportParametricFunctionWithDefaultExpression) {
  constexpr std::string_view kImported = R"(
pub fn some_function<N: u32, M: u32 = {N + 1}>() -> uN[M] { uN[M]:0 }
)";
  constexpr std::string_view kProgram = R"(
import imported;

fn main() -> u5 {
  let var = imported::some_function<4>();
  var
})";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data).status());
  EXPECT_THAT(TypecheckV2(kProgram, "main", &import_data),
              IsOkAndHolds(HasTypeInfo(HasNodeWithType("var", "uN[5]"))));
}

TEST(TypecheckV2Test, ImportParametricFunctionWithMultipleInvocations) {
  constexpr std::string_view kImported = R"(
pub fn add_one(x: u32) -> u32 { x + 1 }

pub fn some_function<N: u32, M: u32 = { add_one(N) }>() -> uN[M] { uN[M]:0 }

pub fn another_fn() -> u3 { some_function<2>() }

pub fn parametric_call<M: u32>() -> uN[M] { some_function<3, M>() }

)";
  constexpr std::string_view kInt = R"(
import imported;

pub fn mid() -> u32 { imported::some_function<31>() }

pub fn default_import<N: u32, M: u32 = { imported::add_one(N) }>() -> uN[M] { uN[M]:0 }

)";
  constexpr std::string_view kProgram = R"(
import imported;
import int;

const VAR = imported::some_function<4>();
const VAR2 = imported::some_function<3, 6>();
const VAR3 = imported::some_function<7>();
const VAR4 = imported::another_fn();
const VAR5 = imported::parametric_call<15>();
const VAR6 = int::mid();

fn main() -> u26 {
   int::default_import<25>()
}
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data).status());
  XLS_EXPECT_OK(TypecheckV2(kInt, "int", &import_data).status());
  EXPECT_THAT(
      TypecheckV2(kProgram, "main", &import_data),
      IsOkAndHolds(HasTypeInfo(AllOf(
          HasNodeWithType("VAR", "uN[5]"), HasNodeWithType("VAR2", "uN[6]"),
          HasNodeWithType("VAR4", "uN[3]"), HasNodeWithType("VAR5", "uN[15]"),
          HasNodeWithType("VAR6", "uN[32]"),
          HasNodeWithType("int::default_import<25>()", "uN[26]"),
          HasNodeWithType("VAR3", "uN[8]")))));
}

TEST(TypecheckV2Test, ImportParametricFunction) {
  constexpr std::string_view kImported = R"(
pub fn some_function<N: u32>() -> uN[N] { uN[N]:0 }
)";
  constexpr std::string_view kProgram = R"(
import imported;

fn main() -> u4 {
  imported::some_function<4>()
})";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK(TypecheckV2(kImported, "imported", &import_data).status());
  EXPECT_THAT(TypecheckV2(kProgram, "main", &import_data),
              IsOkAndHolds(HasTypeInfo(
                  HasNodeWithType("imported::some_function<4>()", "uN[4]"))));
}

TEST(TypecheckV2Test, ImportParametricFunctionInferredValue) {
  constexpr std::string_view kImported = R"(
pub fn some_function<N: u32 = {4}>() -> uN[N] { uN[N]:0 }
)";
  constexpr std::string_view kProgram = R"(
import imported;

fn main() -> u4 {
  imported::some_function()
})";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK(TypecheckV2(kImported, "imported", &import_data).status());
  EXPECT_THAT(TypecheckV2(kProgram, "main", &import_data),
              IsOkAndHolds(HasTypeInfo(
                  HasNodeWithType("imported::some_function()", "uN[4]"))));
}

TEST(TypecheckV2Test, ImportParametricFunctionSizeMismatch) {
  constexpr std::string_view kImported = R"(
pub fn some_function<N: u32>() -> uN[N] { uN[N]:0 }
)";
  constexpr std::string_view kProgram = R"(
import imported;

fn main() -> u8 {
  imported::some_function<4>()
})";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data).status());
  EXPECT_THAT(TypecheckV2(kProgram, "main", &import_data),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSizeMismatch("uN[4]", "u8")));
}

TEST(TypecheckV2Test, ImportFunction) {
  constexpr std::string_view kImported = R"(
pub fn some_function(x: u32) -> u32 { x }
)";
  constexpr std::string_view kProgram = R"(
import imported;

fn main() -> u32 {
  let var = imported::some_function(1);
  var
})";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data).status());
  EXPECT_THAT(TypecheckV2(kProgram, "main", &import_data),
              IsOkAndHolds(HasTypeInfo(HasNodeWithType("var", "uN[32]"))));
}

TEST(TypecheckV2Test, ImportNonExistingFunction) {
  constexpr std::string_view kImported = "";
  constexpr std::string_view kProgram = R"(
import imported;

fn main() -> u32 {
  let var = imported::some_function(1);
  var
})";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data).status());
  EXPECT_THAT(
      TypecheckV2(kProgram, "main", &import_data),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("doesn't exist")));
}

TEST(TypecheckV2Test, ImportNonPublicFunction) {
  constexpr std::string_view kImported = R"(
fn some_function(x: u32) -> u32 { x }
)";
  constexpr std::string_view kProgram = R"(
import imported;

fn main() -> u32 {
  let var = imported::some_function(1);
  var
})";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data).status());
  EXPECT_THAT(
      TypecheckV2(kProgram, "main", &import_data),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("not public")));
}

TEST(TypecheckV2Test, DISABLED_UseFunction) {
  constexpr std::string_view kImported = R"(
pub fn get_val() -> u5[3] {
  u5[3]:[1, 2, 3]
}
)";
  constexpr std::string_view kProgram = R"(#![feature(use_syntax)]
use imported::get_val;

fn main() -> u5 {
  get_val()[1]
})";
  absl::flat_hash_map<std::filesystem::path, std::string> files = {
      {std::filesystem::path("/imported.x"), std::string(kImported)},
      {std::filesystem::path("/fake_main_path.x"), std::string(kProgram)},
  };
  auto vfs = std::make_unique<FakeFilesystem>(
      files, /*cwd=*/std::filesystem::path("/"));
  ImportData import_data = CreateImportDataForTest(std::move(vfs));
  EXPECT_THAT(TypecheckV2(kProgram, "fake_main_path", &import_data),
              IsOkAndHolds(HasTypeInfo(HasNodeWithType("get_val()[1]", "u5"))));
}

TEST(TypecheckV2Test, ModuleWithParametricProcAliasCallingParametricFn) {
  XLS_EXPECT_OK(TypecheckV2(R"(fn bar<Y: u32>(i: uN[Y]) -> uN[Y] {
    i + i
}
proc Foo<N: u32> {
    c: chan<uN[N]> out;
    config(output_c: chan<uN[N]> out) {
        (output_c,)
    }
    init {
        uN[N]:1
    }
    next(i: uN[N]) {
        let result = bar<N>(i);
        let tok = send(join(), c, result);
        result + uN[N]:1
    }
}
proc Bar = Foo<16>;)"));
}

TEST(TypecheckV2Test, QuickcheckFn) {
  EXPECT_THAT(
      R"(
#[quickcheck]
fn f() -> bool { true }
)",
      TypecheckSucceeds(HasNodeWithType("f", "() -> uN[1]")));
}

TEST(TypecheckV2Test, QuickcheckFnBoolAlias) {
  EXPECT_THAT(
      R"(
type BoolAlias = bool;
#[quickcheck]
fn f() -> BoolAlias { true }
)",
      TypecheckSucceeds(HasNodeWithType("f", "() -> uN[1]")));
}

TEST(TypecheckV2Test, QuickcheckFnU1) {
  EXPECT_THAT(
      R"(
#[quickcheck]
fn f() -> u1 { u1:1 }
)",
      TypecheckSucceeds(HasNodeWithType("f", "() -> uN[1]")));
}

TEST(TypecheckV2Test, QuickcheckFnNotBool) {
  EXPECT_THAT(
      R"(
#[quickcheck]
fn f() -> u32 { u32:0 }
)",
      TypecheckFails(HasTypeMismatch("uN[1]", "uN[32]")));
}

TEST(TypecheckV2Test, QuickcheckFnAliasNotBool) {
  EXPECT_THAT(
      R"(
type IntAlias = u32;
#[quickcheck]
fn f() -> IntAlias { IntAlias:0 }
)",
      TypecheckFails(HasTypeMismatch("uN[1]", "uN[32]")));
}

TEST(TypecheckV2Test, ImportedEnumInFunction) {
  constexpr std::string_view kImported = R"(
pub enum MyEnum {
  A = s8:0,
  B = -128,
  C = 127,
}
)";
  constexpr std::string_view kProgram = R"(
import imported;

fn f(x: imported::MyEnum) -> imported::MyEnum {
  x
}
const_assert!(f(imported::MyEnum::A) as s8 == 0);
const_assert!(f(imported::MyEnum::B) as s8 == -128);
const_assert!(f(imported::MyEnum::C) as s8 == 127);
)";

  auto import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data));
  EXPECT_THAT(TypecheckV2(kProgram, "main", &import_data),
              IsOkAndHolds(HasTypeInfo(HasNodeWithType("x", "MyEnum"))));
}

TEST(TypecheckV2Test, ComparisonOfReturnValues) {
  EXPECT_THAT(R"(
fn foo(a: u32) -> u32 { a }
const Y = foo(1) > foo(2);
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("Y", "uN[1]"),
                                      HasNodeWithType("foo(1)", "uN[32]"),
                                      HasNodeWithType("foo(2)", "uN[32]"))));
}

TEST(TypecheckV2Test, TypeColonRefAsArgumentFails) {
  constexpr std::string_view kImported = R"(
pub type T = u32;
)";
  constexpr std::string_view kProgram = R"(
import imported;
fn f(a: u32) {}
fn g() {
  f(imported::T);
}
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data));
  EXPECT_THAT(
      TypecheckV2(kProgram, "main", &import_data),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Cannot pass a type as a function argument.")));
}

TEST(TypecheckV2Test, UnassignedReturnValueIgnored) {
  EXPECT_THAT(
      R"(
fn ignored() -> u32 { u32:0 }

fn main() -> u32 {
  ignored();
  u32:1
}
)",
      TypecheckSucceeds(HasNodeWithType("ignored()", "uN[32]")));
}

TEST(TypecheckV2Test, BasicLet) {
  EXPECT_THAT(
      R"(
fn f() -> u4 {
  let x = u4:1;
  x
}
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("let x = u4:1;", "uN[4]"),
                              HasNodeWithType("x", "uN[4]"))));
}

TEST(TypecheckV2Test, FuzzTestDomainsSuccess) {
  EXPECT_THAT(R"(
#[fuzz_test(domains=`u32:0..1`)]
fn f(x: u32) {}
)",
              TypecheckSucceeds(::testing::_));
}

TEST(TypecheckV2Test, FuzzTestBadRange) {
  EXPECT_THAT(R"(
#[fuzz_test(domains=`u32:0..u64:1`)]
fn f(x: u32) {}
)",
              TypecheckFails(HasSizeMismatch("u32", "u64")));
}

TEST(TypecheckV2Test, FuzzTestDomainNotSupported) {
  EXPECT_THAT(
      R"(
#[fuzz_test(domains=`u8:0`)]
fn f(x: u8) {}
)",
      TypecheckFails(HasSubstr("Unsupported fuzz test domain `u8:0`")));
}

TEST(TypecheckV2Test, FuzzTestConstRange) {
  EXPECT_THAT(R"(
const C = u32:0..1;
#[fuzz_test(domains=`C`)]
fn f(x: u32) {}
)",
              TypecheckSucceeds(::testing::_));
}

TEST(TypecheckV2Test, FuzzTestNoParameters) {
  EXPECT_THAT(
      R"(
#[fuzz_test]
fn f() {}
)",
      TypecheckFails(HasSubstr("Can only fuzz test functions with at least 1 "
                               "parameter; function `f` has 0")));
}

TEST(TypecheckV2Test, FuzzTestAttributeWithZeroArguments) {
  EXPECT_THAT(R"(
#[fuzz_test]
fn f(x: u32) {}
)",
              TypecheckSucceeds(::testing::_));
}

}  // namespace
}  // namespace xls::dslx
