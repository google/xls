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
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system/typecheck_test_utils.h"
#include "xls/dslx/type_system_v2/matchers.h"

// Tests for array and tuple literals, indexing, concatenation, and
// decomposition.
namespace xls::dslx {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::AllOf;
using ::testing::HasSubstr;

TEST(TypecheckV2Test, ConcatOfBitsLiterals) {
  EXPECT_THAT("const X = u8:3 ++ u1:1;", TopNodeHasType("uN[9]"));
}

TEST(TypecheckV2Test, ConcatWithSignedBitsLhsFails) {
  EXPECT_THAT(
      "const X = s8:3 ++ u1:1;",
      TypecheckFails(HasSubstr(
          "Concatenation requires operand types to both be unsigned bits")));
}

TEST(TypecheckV2Test, ConcatWithSignedBitsRhsFails) {
  EXPECT_THAT(
      "const X = u8:3 ++ s1:-1;",
      TypecheckFails(HasSubstr(
          "Concatenation requires operand types to both be unsigned bits")));
}

TEST(TypecheckV2Test, ConcatWithArrayAndNonArrayFails) {
  EXPECT_THAT(
      "const X = [u8:3] ++ u8:1;",
      TypecheckFails(HasSubstr(
          "Attempting to concatenate array/non-array values together")));
}

TEST(TypecheckV2Test, ConcatWithDifferentArrayElementTypesFails) {
  EXPECT_THAT("const X = [u8:3] ++ [u16:1];",
              TypecheckFails(HasTypeMismatch("uN[8]", "uN[16]")));
}

TEST(TypecheckV2Test, ConcatWithBitsAndStructFails) {
  EXPECT_THAT(
      R"(
struct Foo { x: u8 }
const X = Foo{ x: u8:1 } ++ u8:1;
)",
      TypecheckFails(HasSubstr("Concatenation requires operand types to be "
                               "either both-arrays or both-bits")));
}

TEST(TypecheckV2Test, ConcatOfArrayLiterals) {
  EXPECT_THAT("const X = [u32:1, 2, 3] ++ [u32:1, 2, 3];",
              TopNodeHasType("uN[32][6]"));
}

TEST(TypecheckV2Test, IndexOfArrayConcat) {
  EXPECT_THAT("const X = ([u32:1, 2, 3] ++ [u32:1, 2])[1];",
              TopNodeHasType("uN[32]"));
}

TEST(TypecheckV2Test, ConcatOfBitsConstants) {
  EXPECT_THAT(R"(
const A = u20:0;
const B = u30:0;
const X = A ++ B;
)",
              TypecheckSucceeds(HasNodeWithType("X", "uN[50]")));
}

TEST(TypecheckV2Test, ConcatOfArrayConstants) {
  EXPECT_THAT(R"(
const A = [u32:0, 1, 2];
const B = [u32:200, 300];
const X = A ++ B;
)",
              TypecheckSucceeds(HasNodeWithType("X", "uN[32][5]")));
}

TEST(TypecheckV2Test, ConcatOfBitsParametricFunctionArgs) {
  EXPECT_THAT(R"(
fn f<A: u32, B: u32>(a: uN[A], b: uN[B]) -> uN[A + B] {
  a ++ b
}
const X = f(u16:0, u32:0);
)",
              TypecheckSucceeds(HasNodeWithType("X", "uN[48]")));
}

TEST(TypecheckV2Test, ConcatOfArrayParametricFunctionArgs) {
  EXPECT_THAT(R"(
fn f<A: u32, B: u32>(a: u32[A], b: u32[B]) -> u32[A + B] {
  a ++ b
}
const X = f([u32:1, 2, 3], [u32:200]);
)",
              TypecheckSucceeds(HasNodeWithType("X", "uN[32][4]")));
}

TEST(TypecheckV2Test, SumOfConcatsOfBits) {
  EXPECT_THAT("const X = (u16:0 ++ u32:0) + u48:10;", TopNodeHasType("uN[48]"));
}

TEST(TypecheckV2Test, SignedArrayFromHexadecimalWithMSBSet) {
  EXPECT_THAT("const X: s8[2] = [1, 0xFF];",
              TypecheckSucceeds(AllOf(HasNodeWithType("1", "sN[8]"),
                                      HasNodeWithType("0xFF", "sN[8]"))));
}

TEST(TypecheckV2Test, ArrayWithArrayAnnotation) {
  EXPECT_THAT("const X = u32[2]:[1, 2];",
              TypecheckSucceeds(HasNodeWithType("X", "uN[32][2]")));
}

TEST(TypecheckV2Test, ArrayDeclarationMismatchingAnnotationFails) {
  EXPECT_THAT("const X: u16[2] = u32[2]:[1, 2];",
              TypecheckFails(HasSizeMismatch("u16", "u32")));
}

TEST(TypecheckV2Test, ArrayWithArrayAnnotationWithSignednessMismatchFails) {
  EXPECT_THAT("const X = u32[2]:[-1, 2];",
              TypecheckFails(HasSignednessMismatch("u32", "s1")));
}

TEST(TypecheckV2Test, ArrayWithArrayAnnotationWithSizeMismatchFails) {
  EXPECT_THAT("const X = u8[2]:[1, 65536];",
              TypecheckFails(HasSizeMismatch("u8", "u17")));
}

TEST(TypecheckV2Test, ArrayWithArrayAnnotationWithCountMismatchFails) {
  EXPECT_THAT("const X = u8[2]:[u8:1, 2, 3];",
              TypecheckFails(HasTypeMismatch("u8[2]", "u8[3]")));
}

TEST(TypecheckV2Test, AnnotatedEmptyArray) {
  EXPECT_THAT("const X = u8[0]:[];",
              TypecheckSucceeds(HasNodeWithType("X", "uN[8][0]")));
}

TEST(TypecheckV2Test, AnnotatedEmptyArrayMismatchFails) {
  EXPECT_THAT("const X = u8[1]:[];",
              TypecheckFails(HasTypeMismatch("u8[0]", "u8[1]")));
}

TEST(TypecheckV2Test, DecompositionOf2DArray) {
  // Note that this type of decomposition is reverse order in DSLX compared to
  // C++ or Java.
  EXPECT_THAT("fn f(a: u32[4][5]) { let x = a[0]; }",
              TypecheckSucceeds(HasNodeWithType("x", "uN[32][4]")));
}

TEST(TypecheckV2Test, OutOfBoundsDecompositionOf2DArrayFails) {
  // Note that this type of decomposition is reverse order in DSLX compared to
  // C++ or Java.
  EXPECT_THAT(
      "fn f(a: u32[5][4]) { let x = a[4]; }",
      TypecheckFails(HasSubstr("Index has a compile-time constant value 4 that "
                               "is out of bounds of the array type.")));
}

TEST(TypecheckV2Test, IndexWithU64IndexValue) {
  EXPECT_THAT("const X = [u32:1, u32:2][u64:0];",
              TypecheckSucceeds(HasNodeWithType("X", "uN[32]")));
}

TEST(TypecheckV2Test, AutoLiteralIndexBecomesU32) {
  EXPECT_THAT("const X = [u32:5, 6, 7][2];",
              TypecheckSucceeds(HasNodeWithType("2", "uN[32]")));
}

TEST(TypecheckV2Test, IndexWithSignedIndexTypeFails) {
  EXPECT_THAT("const X = [u32:1, u32:2][s32:0];",
              TypecheckFails(HasSignednessMismatch("s32", "u32")));
}

TEST(TypecheckV2Test, IndexWithNonBitsIndexTypeFails) {
  EXPECT_THAT("const X = [u32:1, u32:2][[u32:0]];",
              TypecheckFails(HasTypeMismatch("u32[1]", "u32")));
}

TEST(TypecheckV2Test, IndexOfConstantArray) {
  EXPECT_THAT(R"(
const X: s24[2] = [5, 4];
const Y = X[0];
)",
              TypecheckSucceeds(HasNodeWithType("Y", "sN[24]")));
}

TEST(TypecheckV2Test, IndexOfFunctionReturn) {
  EXPECT_THAT(R"(
fn foo() -> u24[3] { [1, 2, 3] }
const Y = foo()[1];
)",
              TypecheckSucceeds(HasNodeWithType("Y", "uN[24]")));
}

TEST(TypecheckV2Test, IndexOfParametricFunctionReturn) {
  EXPECT_THAT(R"(
fn foo<N: u32>() -> uN[N][3] { [1, 2, 3] }
const Y = foo<16>()[1];
)",
              TypecheckSucceeds(HasNodeWithType("Y", "uN[16]")));
}

TEST(TypecheckV2Test, IndexOfParametricFunctionReturnUsedForInference) {
  EXPECT_THAT(R"(
fn foo<N: u32>() -> uN[N][3] { [1, 2, 3] }
fn bar<N: u32>(a: uN[N]) -> uN[N] { a + 1 }
const Y = bar(foo<16>()[1]);
)",
              TypecheckSucceeds(HasNodeWithType("Y", "uN[16]")));
}

TEST(TypecheckV2Test, IndexOfParametricFunctionArgument) {
  EXPECT_THAT(R"(
fn foo<N: u32>(a: uN[N][3], i: u32) -> uN[N] { a[i] }
const Y = foo<16>([1, 2, 3], 1);
)",
              TypecheckSucceeds(HasNodeWithType("Y", "uN[16]")));
}

TEST(TypecheckV2Test, IndexOfStructFails) {
  EXPECT_THAT(R"(
struct S {
  x: u32
}
const Y = S{ x: 0 }[0];
)",
              TypecheckFails(HasSubstr("Value to index is not an array")));
}

TEST(TypecheckV2Test, IndexOfTupleFails) {
  EXPECT_THAT(R"(
const Y = (u32:1, u32:2)[0];
)",
              TypecheckFails(HasSubstr(
                  "Tuples should not be indexed with array-style syntax.")));
}

TEST(TypecheckV2Test, IndexOfBitsFails) {
  EXPECT_THAT(R"(
const Y = (bits[32]:1)[0];
)",
              TypecheckFails(HasSubstr("Bits-like value cannot be indexed")));
}

TEST(TypecheckV2Test, IndexWithConstexprOutOfRangeFails) {
  EXPECT_THAT(R"(
const X = u32:2;
const Y = [u32:1, u32:2][X];
)",
              TypecheckFails(HasSubstr("out of bounds of the array type")));
}

TEST(TypecheckV2Test, ArrayAnnotationWithTooLargeLiteralDimFails) {
  EXPECT_THAT(
      R"(
const Y = uN[u33:1]:1;
)",
      TypecheckFails(HasSizeMismatch("u33", "u32")));
}

TEST(TypecheckV2Test, ArrayAnnotationWithTooLargeConstantDimFails) {
  EXPECT_THAT(
      R"(
const X = u33:1;
const Y = uN[X]:1;
)",
      TypecheckFails(HasSizeMismatch("u33", "u32")));
}

TEST(TypecheckV2Test, ArrayAnnotationWithSignedLiteralDimFails) {
  EXPECT_THAT("const Y = uN[-1]:1;",
              TypecheckFails(HasSignednessMismatch("s1", "u32")));
}

TEST(TypecheckV2Test, ArrayAnnotationWithSignedConstantDimFails) {
  EXPECT_THAT(
      R"(
const X = s31:1;
const Y = uN[X]:1;
)",
      TypecheckFails(HasSizeMismatch("s31", "u32")));
}

TEST(TypecheckV2Test, ArrayOfTuples) {
  EXPECT_THAT(
      R"(
fn tuple_fn() -> u32 {
   let x = [(1, 2, 3), (3, 4, 5), (5, 6, 7), (7, 8, 9)];
   x[0].1
}
)",
      TypecheckSucceeds(HasNodeWithType("x", "(uN[3], uN[4], uN[4])[4]")));
}

TEST(TypecheckV2Test, ArrayOfTokensFails) {
  EXPECT_THAT(R"(
proc main {
  init {}
  config() {}
  next(st:()) {
    let x = [join()][u1:0];
    ()
  }
}
)",
              TypecheckFails(HasSubstr("tokens cannot be placed in arrays")));
}

TEST(TypecheckV2Test, ArrayWithTokenInTupleFails) {
  EXPECT_THAT(R"(
proc main {
  init {}
  config() {}
  next(st:()) {
    let x = [(join(),)];
    ()
  }
}
)",
              TypecheckFails(HasSubstr("tokens cannot be placed in arrays")));
}

TEST(TypecheckV2Test, NestedTuples) {
  EXPECT_THAT(
      R"(
const X = (((0, 1, 2, 3), 4, (5, 6, 7), 8), (9, (10, 11, 12)), 13);
)",
      TypecheckSucceeds(HasNodeWithType(
          "X",
          "(((uN[0], uN[1], uN[2], uN[2]), uN[3], (uN[3], uN[3], uN[3]), "
          "uN[4]), (uN[4], (uN[4], uN[4], uN[4])), uN[4])")));
}

TEST(TypecheckV2Test, NestedArraysAndTuples) {
  EXPECT_THAT(
      R"(
const X = (((0, 1, 2, 3), 4, [5, 6, 7], 8), (9, (10, [11, 12])), 13);
)",
      TypecheckSucceeds(
          HasNodeWithType("X",
                          "(((uN[0], uN[1], uN[2], uN[2]), uN[3], uN[3][3], "
                          "uN[4]), (uN[4], (uN[4], uN[4][2])), uN[4])")));
}

TEST(TypecheckV2Test, NestedArrays) {
  EXPECT_THAT(
      R"(
const X = [[[0, 1, 2], [2, 3, 4], [4, 5, 6]], [[6, 7, 8], [8, 9, 10], [10, 11, 12]]];
)",
      TypecheckSucceeds(HasNodeWithType("X", "uN[4][3][3][2]")));
}

TEST(TypecheckV2Test, TupleIndexOfConstantTuple) {
  EXPECT_THAT(R"(
const X = (s16:5, s8:4);
const Y = X.1;
)",
              TypecheckSucceeds(HasNodeWithType("Y", "sN[8]")));
}

TEST(TypecheckV2Test, TupleIndexOfFunctionReturn) {
  EXPECT_THAT(R"(
fn foo() -> (u8, u16, u4) { (1, 2, 3) }
const Y = foo().1;
)",
              TypecheckSucceeds(HasNodeWithType("Y", "uN[16]")));
}

TEST(TypecheckV2Test, TupleIndexOfParametricFunctionReturn) {
  EXPECT_THAT(R"(
fn foo<A: u32, B: u32, C: u32>() -> (uN[A], uN[B], uN[C]) { (1, 2, 3) }
const Y = foo<32, 33, 34>().1;
)",
              TypecheckSucceeds(HasNodeWithType("Y", "uN[33]")));
}

TEST(TypecheckV2Test, TupleIndexOfParametricFunctionReturnUsedForInference) {
  EXPECT_THAT(R"(
fn foo<A: u32, B: u32, C: u32>() -> (uN[A], uN[B], uN[C]) { (1, 2, 3) }
fn bar<N: u32>(a: uN[N]) -> uN[N] { a + 1 }
const Y = bar(foo<8, 64, 18>().2);
)",
              TypecheckSucceeds(HasNodeWithType("Y", "uN[18]")));
}

TEST(TypecheckV2Test, TupleIndexOfParametricFunctionArgument) {
  EXPECT_THAT(R"(
fn foo<N: u32>(a: (uN[N], uN[N])) -> uN[N] { a.1 }
const Y = foo<16>((1, 2));
)",
              TypecheckSucceeds(HasNodeWithType("Y", "uN[16]")));
}

TEST(TypecheckV2Test, TupleIndexOfStructFails) {
  EXPECT_THAT(R"(
struct S {
  x: u32
}
const Y = S{ x: 0 }.0;
)",
              TypecheckFails(
                  HasSubstr("Attempted to use tuple indexing on a non-tuple")));
}

TEST(TypecheckV2Test, TupleIndexOfStructMember) {
  EXPECT_THAT(R"(
struct S {
  x: (u16, u32)
}
const X = S{ x: (0, 1) }.x.0;
)",
              TypecheckSucceeds(HasNodeWithType("X", "uN[16]")));
}

TEST(TypecheckV2Test, TupleIndexOfArrayFails) {
  EXPECT_THAT(R"(
const Y = [u32:1, 2].0;
)",
              TypecheckFails(
                  HasSubstr("Attempted to use tuple indexing on a non-tuple")));
}

TEST(TypecheckV2Test, TupleIndexOutOfRangeFails) {
  EXPECT_THAT(
      R"(
const Y = (u32:1, s8:2).2;
)",
      TypecheckFails(HasSubstr("Out-of-bounds tuple index specified: 2")));
}

TEST(TypecheckV2Test, EllipsisArrayWithNoDeclaredSizeFails) {
  EXPECT_THAT(
      R"(
fn bar(x: u32[5]) -> u32 {
  x[4]
}

fn foo() -> u32 {
  let a = u32:1;
  let b = u32:2;
  let y = [a, b, a, ...];
  bar(y)
}

#[test]
fn foo_test() {
  assert_eq(foo(), u32:1);
}
)",
      TypecheckFails(HasSubstr(
          "Array has ellipsis (`...`) but does not have a type annotation.")));
}

TEST(TypecheckV2Test, AccessOfStructMemberArray) {
  EXPECT_THAT(
      R"(
struct S { x: u32[2] }
const X = S { x: [1, 2] }.x;
)",
      TypecheckSucceeds(HasNodeWithType("X", "uN[32][2]")));
}

TEST(TypecheckV2Test, AccessOfStructMemberArrayElement) {
  EXPECT_THAT(
      R"(
struct S { x: u32[2] }
const X = S { x: [1, 2] }.x[0];
)",
      TypecheckSucceeds(HasNodeWithType("X", "uN[32]")));
}

TEST(TypecheckV2Test, AccessOfParametricStructMemberArray) {
  EXPECT_THAT(
      R"(
struct S<M: u32, N: u32> { x: uN[M][N] }
const X = S { x: [u24:1, 2] }.x;
)",
      TypecheckSucceeds(HasNodeWithType("X", "uN[24][2]")));
}

TEST(TypecheckV2Test, AccessOfParametricStructMemberArrayElement) {
  EXPECT_THAT(
      R"(
struct S<M: u32, N: u32> { x: uN[M][N] }
const X = S { x: [u24:1, 2] }.x[1];
)",
      TypecheckSucceeds(HasNodeWithType("X", "uN[24]")));
}

TEST(TypecheckV2Test, AccessOfStructMemberInArray) {
  EXPECT_THAT(
      R"(
struct S { x: u24 }
const X = [S { x: 1 }, S { x: 2 }][0].x;
)",
      TypecheckSucceeds(HasNodeWithType("X", "uN[24]")));
}

TEST(TypecheckV2Test, ParametricStructWithParametricInferredFromArraySize) {
  EXPECT_THAT(
      R"(
struct S<N: u32> {
  x: u32[N]
}
const X = S { x: [1, 2, 3] };
)",
      TypecheckSucceeds(HasNodeWithType("X", "S { x: uN[32][3] }")));
}

TEST(TypecheckV2Test, ParametricStructWithParametricInferredFromArrayElement) {
  EXPECT_THAT(
      R"(
struct S<M: u32, N: u32> {
  x: uN[M][N]
}
const X = S { x: [u24:1, 2, 3] };
)",
      TypecheckSucceeds(HasNodeWithType("X", "S { x: uN[24][3] }")));
}

TEST(TypecheckV2Test, ParametricStructWithParametricInferredFromTupleElement) {
  EXPECT_THAT(
      R"(
struct S<M: u32, N: u32> {
  x: (uN[M], uN[N])
}
const X = S { x: (u24:1, u32:2) };
)",
      TypecheckSucceeds(HasNodeWithType("X", "S { x: (uN[24], uN[32]) }")));
}

TEST(TypecheckV2Test, ParametricStructWithTupleElementMismatchFails) {
  EXPECT_THAT(
      R"(
struct S<M: u32, N: u32> {
  x: (uN[M], uN[N])
}
const X = S<24, 32> { x: (u23:1, u32:2) };
)",
      TypecheckFails(HasSizeMismatch("u23", "uN[24]")));
}

TEST(TypecheckV2Test, FunctionCallReturningUnitTupleExplicitly) {
  EXPECT_THAT(
      R"(
fn foo() -> () { () }
const Y = foo();
)",
      TypecheckSucceeds(AllOf(HasOneLineBlockWithType("()", "()"),
                              HasNodeWithType("const Y = foo();", "()"))));
}

TEST(TypecheckV2Test, FunctionCallReturningArray) {
  EXPECT_THAT(R"(
fn foo() -> s8[3] { [1, 2, 3] }
const Y = foo();
)",
              TypecheckSucceeds(
                  AllOf(HasOneLineBlockWithType("[1, 2, 3]", "sN[8][3]"),
                        HasNodeWithType("const Y = foo();", "sN[8][3]"))));
}

TEST(TypecheckV2Test, FunctionCallReturningTuple) {
  EXPECT_THAT(
      R"(
fn foo() -> (s8, (u32, u24)) { (1, (2, 3)) }
const Y = foo();
)",
      TypecheckSucceeds(AllOf(
          HasOneLineBlockWithType("(1, (2, 3))", "(sN[8], (uN[32], uN[24]))"),
          HasNodeWithType("const Y = foo();", "(sN[8], (uN[32], uN[24]))"))));
}

TEST(TypecheckV2Test, FunctionReturningArrayForTupleFails) {
  EXPECT_THAT(
      R"(
fn foo() -> (u32, u32) { [u32:1, 2] }
)",
      TypecheckFails(HasTypeMismatch("(u32, u32)", "u32[2]")));
}

TEST(TypecheckV2Test, FunctionReturningArrayWithIntegerReturnTypeFails) {
  EXPECT_THAT(R"(
const X = [s32:1, s32:2, s32:3];
fn foo() -> s32 { X }
const Y = foo();
)",
              TypecheckFails(HasTypeMismatch("s32[3]", "s32")));
}

TEST(TypecheckV2Test, FunctionCallReturningPassedInTuple) {
  EXPECT_THAT(
      R"(
fn foo(a: (u32, s4)) -> (u32, s4) { a }
const Y = foo((4, -1));
)",
      TypecheckSucceeds(AllOf(
          HasOneLineBlockWithType("a", "(uN[32], sN[4])"),
          HasNodeWithType("const Y = foo((4, -1));", "(uN[32], sN[4])"))));
}

TEST(TypecheckV2Test, FunctionCallReturningPassedInArray) {
  EXPECT_THAT(R"(
fn foo(a: u32[2]) -> u32[2] { a }
const Y = foo([4, 5]);
)",
              TypecheckSucceeds(AllOf(
                  HasOneLineBlockWithType("a", "uN[32][2]"),
                  HasNodeWithType("const Y = foo([4, 5]);", "uN[32][2]"))));
}

TEST(TypecheckV2Test, FunctionCallPassingInArrayForIntegerFails) {
  EXPECT_THAT(R"(
fn foo(a: u4) -> u4 { a }
const Y = foo([u32:1, u32:2]);
)",
              TypecheckFails(HasTypeMismatch("u32[2]", "u4")));
}

TEST(TypecheckV2Test, ParametricFunctionTakingArrayOfParameterizedSize) {
  EXPECT_THAT(
      R"(
fn foo<N: u32>(a: u32[N]) -> u32[N] { a }
const X = foo<3>([5, 6, 7]);
const Y = foo<4>([8, 9, 10, 11]);
)",
      TypecheckSucceeds(AllOf(
          HasNodeWithType("const X = foo<3>([5, 6, 7]);", "uN[32][3]"),
          HasNodeWithType("const Y = foo<4>([8, 9, 10, 11]);", "uN[32][4]"),
          HasNodeWithType("5", "uN[32]"), HasNodeWithType("6", "uN[32]"),
          HasNodeWithType("7", "uN[32]"), HasNodeWithType("8", "uN[32]"),
          HasNodeWithType("9", "uN[32]"), HasNodeWithType("10", "uN[32]"),
          HasNodeWithType("11", "uN[32]"))));
}

TEST(TypecheckV2Test, ParametricFunctionInferenceOfAliasedArray) {
  EXPECT_THAT(R"(
struct Foo<N: u32> { a: uN[N] }
type Foo32 = Foo<32>;
fn f<N: u32>(a: Foo<32>[N]) -> Foo<32>[N] { a }
const X: Foo32[2] = [Foo32 { a: 1 }, Foo32 { a: 2 }];
const Y = f(X);
  )",
              TypecheckSucceeds(HasNodeWithType("Y", "Foo { a: uN[32] }[2]")));
}

TEST(TypecheckV2Test, ParametricFunctionWithArrayMismatchingParameterizedSize) {
  EXPECT_THAT(R"(
fn foo<N: u32>(a: u32[N]) -> u32[N] { a }
const X = foo<3>([u32:1, u32:2, u32:3, u32:4]);
)",
              TypecheckFails(HasTypeMismatch("u32[4]", "u32[3]")));
}

TEST(TypecheckV2Test, NestedParametricFunctionCallInArrayDim) {
  // This test is based on similar usage of `checked_cast(max(...))` in std.x.
  EXPECT_THAT(R"(
fn max<S: bool, N: u32>(x: xN[S][N], y: xN[S][N]) -> xN[S][N] {
  if x > y { x } else { y }
}

fn foo<A: u32, B: u32>() -> u32 {
  let foo = uN[checked_cast<u32>(max(s32:0, A as s32 - B as s32))]:0;
  foo as u32
}
const X = foo<10, 5>();
)",
              TypecheckSucceeds(HasNodeWithType("X", "uN[32]")));
}

TEST(TypecheckV2Test, CastTupleToU32) {
  EXPECT_THAT(R"(const X = (u32:1, u32:2);
const Y = X as u32;)",
              TypecheckFails(HasCastError("(uN[32], uN[32])", "uN[32]")));
}

TEST(TypecheckV2Test, CastBitsArray2xU16ToU32) {
  EXPECT_THAT(R"(const X = [u16:1, u16:2];
const Y = X as u32;)",
              TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[16][2]"),
                                      HasNodeWithType("Y", "uN[32]"))));
}

TEST(TypecheckV2Test, TestBitsArray2xU1ToU2) {
  EXPECT_THAT(R"(const X = u1[2]:[u1:1, u1:0];
const Y: u2 = X as u2;)",
              TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[1][2]"),
                                      HasNodeWithType("Y", "uN[2]"))));
}

TEST(TypecheckV2Test, TestBitsArray3xU1ToU4) {
  EXPECT_THAT(R"(const X = u1[3]:[u1:1, u1:0, u1:1];
const Y: u4 = X as u4;)",
              TypecheckFails(HasCastError("uN[1][3]", "uN[4]")));
}

TEST(TypecheckV2Test, MatchArmTupleType) {
  EXPECT_THAT(R"(
const X = u32:1;
const Y = u31:2;
const Z = match X {
  u32:1 => (X, Y),
  _ => (u32:0, Y)
};
)",
              TypecheckSucceeds(HasNodeWithType("Z", "(uN[32], uN[31])")));
}

TEST(TypecheckV2Test, PatternMatchWithRestOfTuple) {
  EXPECT_THAT(R"(
fn f(t: (u8, u32, u32, u32)) -> u32 {
    match t {
        (42, .., y) => y,
        (_, .., y) => y + 1,
    }
}
fn main() {
    const VAL = f((42, 0, 0, 10));
    let res = uN[VAL]:0;

    let val2 = f((3, 0, 0,10));
    let res2 = uN[val2]:0;
}
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("res", "uN[10]"),
                                      HasNodeWithType("res2", "uN[11]"),
                                      HasNodeWithType("y", "uN[32]"))));
}

TEST(TypecheckV2Test, PatternMatchWithRestOfTupleIsOne) {
  EXPECT_THAT(R"(
fn f(t: (u8, u32, u32)) -> u32 {
    match t {
        (42, .., y) => y,
        (_, .., y) => y + 1,
    }
}
fn main() {
    const VAL = f((42, 0, 10));
    let res = uN[VAL]:0;

    let val2 = f((3, 0, 10));
    let res2 = uN[val2]:0;
}
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("res", "uN[10]"),
                                      HasNodeWithType("res2", "uN[11]"),
                                      HasNodeWithType("y", "uN[32]"))));
}

TEST(TypecheckV2Test, PatternMatchWithRestOfTupleIsNone) {
  EXPECT_THAT(R"(
fn f(t: (u8, u32)) -> u32 {
    match t {
        (42, .., y) => y,
        (_, .., y) => y + 1,
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

TEST(TypecheckV2Test, PatternMatchWithRangeInTuple) {
  XLS_EXPECT_OK(TypecheckV2(R"(
fn f(x: (u32, u32)) -> u32 {
    match x {
        (1, 1..3) => u32:1,
        _ => u32:0,
    }
}

fn main() {
  const_assert!(f((1, 0)) == 0);
  const_assert!(f((1, 1)) == 1);
  const_assert!(f((1, 2)) == 1);
  const_assert!(f((0, 2)) == 0);
}
)"));
}

TEST(TypecheckV2Test, ZeroMacroArray) {
  EXPECT_THAT("const Y = zero!<u10[2]>();",
              TypecheckSucceeds(HasNodeWithType("Y", "uN[10][2]")));
}

TEST(TypecheckV2Test, ZeroMacroTuple) {
  EXPECT_THAT("const Y = zero!<(u10, u32)>();",
              TypecheckSucceeds(HasNodeWithType("Y", "(uN[10], uN[32])")));
}

TEST(TypecheckV2Test, ArrayOfTypeColonRefFails) {
  constexpr std::string_view kImported = R"(
pub type T = u32;
)";
  constexpr std::string_view kProgram = R"(
import imported;
const X = [imported::T];
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data));
  EXPECT_THAT(TypecheckV2(kProgram, "main", &import_data),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Array element cannot be a type reference.")));
}

TEST(TypecheckV2Test, AllOnesMacroArray) {
  EXPECT_THAT("const Y = all_ones!<u10[2]>();",
              TypecheckSucceeds(HasNodeWithType("Y", "uN[10][2]")));
}

TEST(TypecheckV2Test, AllOnesMacroTuple) {
  EXPECT_THAT("const Y = all_ones!<(u10, u32)>();",
              TypecheckSucceeds(HasNodeWithType("Y", "(uN[10], uN[32])")));
}

TEST(TypecheckV2Test, LetWithTupleConst) {
  EXPECT_THAT(R"(
const TUP = (u32:1, bits[4]:0);
fn f() -> bits[4] {
  let (x, y) = TUP;
  y
}
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("x", "uN[32]"),
                                      HasNodeWithType("y", "uN[4]"))));
}

TEST(TypecheckV2Test, LetWithTupleInParametricFn) {
  EXPECT_THAT(R"(
fn f<N: u32>(x: uN[N]) -> uN[N] {
  let (y, z) = (x + uN[N]:1, u32:3);
  y
}

fn main() {
  const C = f<16>(uN[16]:5);
  let z = f<4>(uN[4]:0);
}
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("C", "uN[16]"),
                                      HasNodeWithType("z", "uN[4]"))));
}

TEST(TypecheckV2Test, LetWithRestOfTupleInParametricFn) {
  EXPECT_THAT(R"(
fn f<N: u32, M: u32 = {N * 2}>(x: uN[N]) -> (uN[N], uN[M]) {
  let (y,.., z) = (x + uN[N]:1, u15:0, u6:7, uN[M]:3);
  (y, z)
}

fn main() {
  let (c, _) = f<16>(uN[16]:5);
  let (_, z) = f<4>(uN[4]:0);
}
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("c", "uN[16]"),
                                      HasNodeWithType("z", "uN[8]"))));
}

TEST(TypecheckV2Test, BadTupleAnnotation) {
  EXPECT_THAT(
      R"(
fn f() -> u32 {
  let (a, b, c): (u32, u32, s8) = (u32:1, u32:2, u32:3);
  a
}
)",
      TypecheckFails(HasSizeMismatch("u32", "s8")));
}

TEST(TypecheckV2Test, BadTupleType) {
  EXPECT_THAT(
      R"(
fn f() -> u32 {
  let (a, b, c): (u32, u32) = (u32:1, u32:2, u32:3);
  a
}
)",
      TypecheckFails(HasSubstr("Out-of-bounds tuple index specified")));
}

TEST(TypecheckV2Test, DuplicateRestOfTupleError) {
  EXPECT_THAT(R"(
 fn main() {
   let (x, .., ..) = (u32:7, u24:6, u18:5, u12:4, u8:3);
 }
 )",
              TypecheckFails(HasSubstr("can only be used once")));
}

TEST(TypecheckV2Test, TupleCountMismatch) {
  EXPECT_THAT(R"(
 fn main() {
   let (x, y) = (u32:7, u24:6, u18:5, u12:4, u8:3);
 }
 )",
              TypecheckFails(HasSubstr("a 5-element tuple to 2 values")));
}

TEST(TypecheckV2Test, RestOfTupleCountMismatch) {
  EXPECT_THAT(R"(
 fn main() {
   let (x, .., y, z) = (u32:7, u8:3);
 }
 )",
              TypecheckFails(HasSubstr("a 2-element tuple to 3 values")));
}

TEST(TypecheckV2Test, RestOfTupleCountMismatchNested) {
  EXPECT_THAT(R"(
fn main() {
  let (x, .., (y, .., z)) = (u32:7, u8:3, (u12:4,));
}
)",
              TypecheckFails(HasSubstr("a 1-element tuple to 2 values")));
}

TEST(TypecheckV2Test, TupleAssignsTypes) {
  EXPECT_THAT(R"(
fn main() {
  let (x, y): (u32, s8) = (u32:7, s8:3);
}
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("x", "uN[32]"),
                                      HasNodeWithType("y", "sN[8]"))));
}

TEST(TypecheckV2Test, RestOfTupleSkipsMiddle) {
  EXPECT_THAT(
      R"(
fn main() {
  let (x, .., y) = (u32:7, u12:4, s8:3);
  let (xx, yy): (u32, s8) = (x, y);
}
)",
      TypecheckSucceeds(AllOf(
          HasNodeWithType("x", "uN[32]"), HasNodeWithType("y", "sN[8]"),
          HasNodeWithType("xx", "uN[32]"), HasNodeWithType("yy", "sN[8]"))));
}

TEST(TypecheckV2Test, RestOfTupleSkipsNone) {
  EXPECT_THAT(
      R"(
fn main() {
  let (x, .., y) = (u32:7, s8:3);
  let (xx, yy): (u32, s8) = (x, y);
}
)",
      TypecheckSucceeds(AllOf(
          HasNodeWithType("x", "uN[32]"), HasNodeWithType("y", "sN[8]"),
          HasNodeWithType("xx", "uN[32]"), HasNodeWithType("yy", "sN[8]"))));
}

TEST(TypecheckV2Test, RestOfTuplekSkipsNoneWithThree) {
  EXPECT_THAT(
      R"(
fn main() {
  let (x, y, .., z) = (u32:7, u12:4, s8:3);
  let (xx, yy, zz): (u32, u12, s8) = (x, y, z);
}
)",
      TypecheckSucceeds(AllOf(
          HasNodeWithType("x", "uN[32]"), HasNodeWithType("y", "uN[12]"),
          HasNodeWithType("z", "sN[8]"), HasNodeWithType("xx", "uN[32]"),
          HasNodeWithType("yy", "uN[12]"), HasNodeWithType("zz", "sN[8]"))));
}

TEST(TypecheckV2Test, RestOfTupleSkipsEnd) {
  EXPECT_THAT(
      R"(
fn main() {
  let (x, y, ..) = (u32:7, s8:3, u12:4);
  let (xx, yy): (u32, s8) = (x, y);
}
)",
      TypecheckSucceeds(AllOf(
          HasNodeWithType("x", "uN[32]"), HasNodeWithType("y", "sN[8]"),
          HasNodeWithType("xx", "uN[32]"), HasNodeWithType("yy", "sN[8]"))));
}

TEST(TypecheckV2Test, RestOfTupleSkipsManyAtEnd) {
  EXPECT_THAT(
      R"(
fn main() {
  let (x, y, ..) = (u32:7, s8:3, u12:4, u32:0);
  let (xx, yy): (u32, s8) = (x, y);
}
)",
      TypecheckSucceeds(AllOf(
          HasNodeWithType("x", "uN[32]"), HasNodeWithType("y", "sN[8]"),
          HasNodeWithType("xx", "uN[32]"), HasNodeWithType("yy", "sN[8]"))));
}

TEST(TypecheckV2Test, RestOfTupleSkipsManyInMiddle) {
  EXPECT_THAT(
      R"(
fn main() {
  let (x, .., y) = (u32:7, u8:3, u12:4, s8:3);
  let (xx, yy): (u32, s8) = (x, y);
}
)",
      TypecheckSucceeds(AllOf(
          HasNodeWithType("x", "uN[32]"), HasNodeWithType("y", "sN[8]"),
          HasNodeWithType("xx", "uN[32]"), HasNodeWithType("yy", "sN[8]"))));
}

TEST(TypecheckV2Test, RestOfTupleSkipsBeginning) {
  EXPECT_THAT(
      R"(
fn main() {
  let (.., x, y) = (u12:7, u8:3, u32:4, s8:3);
  let (xx, yy): (u32, s8) = (x, y);
}
)",
      TypecheckSucceeds(AllOf(
          HasNodeWithType("x", "uN[32]"), HasNodeWithType("y", "sN[8]"),
          HasNodeWithType("xx", "uN[32]"), HasNodeWithType("yy", "sN[8]"))));
}

TEST(TypecheckV2Test, RestOfTupleSkipsManyAtBeginning) {
  EXPECT_THAT(
      R"(
fn main() {
  let (.., x) = (u8:3, u12:4, u32:7);
  let xx: u32 = x;
}
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("x", "uN[32]"),
                              HasNodeWithType("xx", "uN[32]"))));
}

TEST(TypecheckV2Test, RestOfTupleNested) {
  EXPECT_THAT(
      R"(
fn main() {
  let (x, .., (.., y)) = (u32:7, u8:3, u18:5, (u12:4, u11:5, s8:3));
}
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("x", "uN[32]"),
                              HasNodeWithType("y", "sN[8]"))));
}

TEST(TypecheckV2Test, RestOfTupleNestedSingleton) {
  EXPECT_THAT(
      R"(
fn main() {
  let (x, .., (y,)) = (u32:7, u8:3, (s8:3,));
  let (xx, yy): (u32, s8) = (x, y);
}
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("x", "uN[32]"),
                              HasNodeWithType("y", "sN[8]"))));
}

TEST(TypecheckV2Test, RestOfTupleIsLikeWildcard) {
  EXPECT_THAT(R"(
fn main() {
  let (x, .., (.., y)) = (u32:7, u18:5, (u12:4, s8:3));
}
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("x", "uN[32]"),
                                      HasNodeWithType("y", "sN[8]"))));
}

TEST(TypecheckV2Test, RestOfTupleDeeplyNested) {
  EXPECT_THAT(
      R"(
fn main() {
  let (x, y, .., ((.., z), .., d)) = (u32:7, u8:1,
                            ((u32:3, u64:4, uN[128]:5), u12:4, s8:3));
  }
)",
      TypecheckSucceeds(AllOf(
          HasNodeWithType("x", "uN[32]"), HasNodeWithType("y", "uN[8]"),
          HasNodeWithType("z", "uN[128]"), HasNodeWithType("d", "sN[8]"))));
}

TEST(TypecheckV2Test, RestOfTupleDeeplyNestedNonConstants) {
  EXPECT_THAT(
      R"(
fn main() {
  // Initial values
  let (xi, yi, zi): (u32, u8, uN[128]) = (u32:7, u8:1, uN[128]:5);
  let (x, y, .., ((.., z), .., d)) = (xi, yi,
                            ((u32:3, u64:4, zi), u12:4, s8:3));
  }
)",
      TypecheckSucceeds(AllOf(
          HasNodeWithType("x", "uN[32]"), HasNodeWithType("y", "uN[8]"),
          HasNodeWithType("z", "uN[128]"), HasNodeWithType("d", "sN[8]"))));
}

TEST(TypecheckV2Test, ArraySizeMismatchConst) {
  // Previously this was crashing in the const evaluator, because
  // it didn't have the type of the u2:1
  // Now, it knows what the type of the u2:1 is, and it fails
  // properly with a size mismatch (since array sizes must be u32).
  EXPECT_THAT(R"(
fn identity<N: u32>(x: uN[N]) -> uN[N] { x }
const X:uN[identity(u2:1)][4] = [1,2,1,2];
)",
              TypecheckFails(HasSizeMismatch("u32", "uN[2]")));
}

TEST(TypecheckV2Test, ArraySizeMismatchLet) {
  EXPECT_THAT(R"(
fn identity<N: u32>(x: uN[N]) -> uN[N] { x }
fn foo() {
  let X:uN[identity(u2:1)][4] = [1,2,1,2];
}
)",
              TypecheckFails(HasSizeMismatch("u32", "uN[2]")));
}

TEST(TypecheckV2Test, CastToXbitsBasedBoolArray) {
  EXPECT_THAT(R"(
const ARRAY_SIZE = u32:44;
type MyXn = xN[bool:0x0][1];  // equivalent to a bool

fn main() -> bool[44] {
  let x: u44 = 0;
  // Equivalent to casting bits to corresponding bool array.
  x as MyXn[ARRAY_SIZE]
}

fn f() {
  let n = main();
}
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("x", "uN[44]"),
                                      HasNodeWithType("n", "uN[1][44]"))));
}

TEST(TypecheckV2Test, SliceWithOneNegativeAndOnePositiveIndex) {
  EXPECT_THAT(R"(
const X = u6:0b100111;
const Y = X[-2:6];
)",
              TypecheckSucceeds(HasNodeWithType("Y", "uN[2]")));
}

TEST(TypecheckV2Test, RangeExprArraySizeType) {
  // Range expr is valid as long as the size can fit u32.
  EXPECT_THAT(R"(const X = u64:0x100000000..u64:0x100000005;)",
              TypecheckSucceeds(HasNodeWithType("X", "uN[64][5]")));
}

TEST(TypecheckV2Test, RangeExprArraySizeTypeSigned) {
  EXPECT_THAT(R"(const X = s8:-128..s8:127;)",
              TypecheckSucceeds(HasNodeWithType("X", "sN[8][255]")));
}

TEST(TypecheckV2Test, RangeExprArraySizeTypeTooLarge) {
  EXPECT_THAT(R"(const X = u64:0..u64:0x100000000;)",
              TypecheckFails(HasSubstr(
                  "Range expr `u64:0..u64:0x100000000` has size 4294967296")));
}

TEST(TypecheckV2Test, RangeExprArraySizeTypeTooLargeInclusive) {
  EXPECT_THAT(R"(const X = u32:0..=u32:0xFFFFFFFF;)",
              TypecheckFails(HasSubstr(
                  "Range expr `u32:0..=u32:0xFFFFFFFF` has size 4294967296")));
}

TEST(TypecheckV2Test, RangeExprArraySizeTypeTooLargeSigned) {
  EXPECT_THAT(R"(const X = s64:0x8000000000000000..s64:0xFFFFFFFFFFFFFFFF;)",
              TypecheckFails(HasSubstr(
                  "Range expr `s64:0x8000000000000000..s64:0xFFFFFFFFFFFFFFFF` "
                  "has size 9223372036854775807")));
}

TEST(TypecheckV2Test, DISABLED_AttributeOfArrayFormatBitsLikeType) {
  XLS_EXPECT_OK(TypecheckV2(R"(
const_assert!(uN[8]::MAX == 255);
const_assert!(sN[8]::MAX = 127);
const_assert!(xN[true][8]::MAX = 127);
)"));
}

TEST(TypecheckV2Test, ForArray) {
  EXPECT_THAT(
      R"(
fn foo(A : uN[32][5]) {
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

TEST(TypecheckV2Test, ForTupleTypeMismatch) {
  EXPECT_THAT(
      R"(
fn foo() {
  let X = for (i, a, _) in u32:0..u32:5 {
    a + i
  } (0);
}
)",
      TypecheckFails(
          HasSubstr("For-loop iterator and accumulator name tuple "
                    "must contain 2 top-level elements; got: `(i, a, _)`")));
}

TEST(TypecheckV2Test, UnrollForTupleTypeMismatch) {
  EXPECT_THAT(
      R"(
fn foo() {
  unroll_for! (i) in u32:0..5 {} (u32:0);
}
)",
      TypecheckFails(
          HasSubstr("For-loop iterator and accumulator name tuple must contain "
                    "2 top-level elements; got: `(i)`")));
}

TEST(TypecheckV2Test, ProcWithChannelArray) {
  EXPECT_THAT(
      R"(
proc P {
  c: chan<u32>[4] in;
  init { () }
  config(c: chan<u32>[4] in) {
    (c,)
  }
  next(state: ()) {
    unroll_for! (i, ()): (u32, ()) in u32:0..u32:4 {
      let (_, v) = recv(token(), c[i]);
      trace_fmt!("v: {}", v);
    }(());
  }
}
)",
      TypecheckSucceeds(
          HasRepeatedNodeWithType(R"(trace_fmt!("v: {}", v))", "token", 4)));
}

TEST(TypecheckV2Test, ProcWithChannelArrayOutOfBoundsFails) {
  EXPECT_THAT(
      R"(
proc P {
  c: chan<u32>[4] in;
  init { () }
  config(c: chan<u32>[4] in) {
    (c,)
  }
  next(state: ()) {
    unroll_for! (i, ()): (u32, ()) in u32:0..u32:5 {
      let (_, v) = recv(token(), c[i]);
      trace_fmt!("v: {}", v);
    }(());
  }
}
)",
      TypecheckFails(HasSubstr("Index has a compile-time constant value 4 that "
                               "is out of bounds of the array type.")));
}

TEST(TypecheckV2Test, ProcWith2DChannelArray) {
  EXPECT_THAT(
      R"(
proc P {
  c: chan<u32>[4][5] in;
  init { () }
  config(c: chan<u32>[4][5] in) {
    (c,)
  }
  next(state: ()) {
    unroll_for! (i, ()): (u32, ()) in u32:0..u32:5 {
      unroll_for! (j, ()): (u32, ()) in u32:0..u32:4 {
        let (_, v) = recv(token(), c[i][j]);
        trace_fmt!("v: {}", v);
      }(());
    }(());
  }
}
)",
      TypecheckSucceeds(
          HasRepeatedNodeWithType(R"(trace_fmt!("v: {}", v))", "token", 20)));
}

TEST(TypecheckV2Test, ProcWith2DChannelArrayIndexOutOfBoundsFails) {
  EXPECT_THAT(
      R"(
proc P {
  c: chan<u32>[4][5] in;
  init { () }
  config(c: chan<u32>[4][5] in) {
    (c,)
  }
  next(state: ()) {
    unroll_for! (i, ()): (u32, ()) in u32:0..u32:5 {
      unroll_for! (j, ()): (u32, ()) in u32:0..u32:5 {
        let (_, v) = recv(token(), c[i][j]);
        trace_fmt!("v: {}", v);
      }(());
    }(());
  }
}
)",
      TypecheckFails(HasSubstr("Index has a compile-time constant value 4 that "
                               "is out of bounds of the array type.")));
}

TEST(TypecheckV2Test, ProcWithChannelArrayExpr) {
  EXPECT_THAT(
      R"(
const A = u32:1;
proc Proc {
  inputs: chan<s32>[A + 1] in;
  outputs: chan<s32>[A + 1] out;
  config() {
    let (a, b) = chan<s32>[1 + A]("c");
    (b, a)
  }
  init { () }
  next(state: ()) { () }
}

)",
      TypecheckSucceeds(
          AllOf(HasNodeWithType("a", "chan(sN[32], dir=out)[2]"),
                HasNodeWithType("b", "chan(sN[32], dir=in)[2]"))));
}

TEST(TypecheckV2Test, ProcConfigRequireTuple) {
  EXPECT_THAT(
      R"(
proc Proc {
  input: chan<()> in;
  config(input: chan<()> in) {
    (input)
  }
  init { () }
  next(state: ()) { () }
}

)",
      TypecheckFails(HasTypeMismatch("(chan<()> in,)", "chan<()> in")));
}

TEST(TypecheckV2Test, ImportConstantArray) {
  constexpr std::string_view kImported = R"(
const SIZE = u32:5;
pub const ARRAY = u32[SIZE]:[1, 2, 3, 4, 5];
)";
  constexpr std::string_view kProgram = R"(
import imported;

fn main() -> u32[5] {
  let var = imported::ARRAY;
  var
})";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data).status());
  EXPECT_THAT(
      TypecheckV2(kProgram, "main", &import_data),
      IsOkAndHolds(HasTypeInfo(HasNodeWithType("main", "() -> uN[32][5]"))));
}

TEST(TypecheckV2Test, ImportTypeAliasAsArraySize) {
  constexpr std::string_view kImported = R"(
pub type T = u32;
)";
  constexpr std::string_view kProgram = R"(
import imported;

type T = imported::T;

fn main() {
  let a: s32[T:2] = [0, ...];
}
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data));
  XLS_EXPECT_OK(TypecheckV2(kProgram, "main", &import_data));
}

TEST(TypecheckV2Test, ImportArrayTypeSizeWithImportedInvocation) {
  constexpr std::string_view kImported1 = R"(
pub fn f(a:u32) -> u32 {
  a
}
)";
  constexpr std::string_view kImported2 = R"(
import imported1;
pub const B = u32:16;
pub type b = bits[imported1::f(B)];
)";
  constexpr std::string_view kProgram = R"(
import imported2;
pub fn g() -> imported2::b {
 imported2::b:1
}
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported1, "imported1", &import_data));
  XLS_EXPECT_OK(TypecheckV2(kImported2, "imported2", &import_data));
  XLS_EXPECT_OK(TypecheckV2(kProgram, "main", &import_data));
}

TEST(TypecheckV2Test, ImportArrayTypeSizeWithImportedConst) {
  constexpr std::string_view kImported1 = R"(
pub type T = u32;
pub const A = u32:2;
)";
  constexpr std::string_view kImported2 = R"(
import imported1;
pub fn f(i:imported1::T) -> u32[imported1::A + u32:1] {
  u32[3]:[0,0,0]
}
)";
  constexpr std::string_view kProgram = R"(
import imported1;
import imported2;
fn g(i:imported1::T) -> u32[imported1::A + u32:1] {
  imported2::f(i)
}
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported1, "imported1", &import_data));
  XLS_EXPECT_OK(TypecheckV2(kImported2, "imported2", &import_data));
  XLS_EXPECT_OK(TypecheckV2(kProgram, "main", &import_data));
}

TEST(TypecheckV2Test, ImportedArrayOfEnum) {
  constexpr std::string_view kImported = R"(
pub enum MyEnum : u2 {
    A = 0,
    B = 1,
    C = 2,
}

pub const ENUMS = MyEnum[3]:[MyEnum::A, MyEnum::B, MyEnum::C];
)";
  constexpr std::string_view kProgram = R"(
import imported;

fn main(i: u2) -> imported::MyEnum {
    imported::ENUMS[i]
}

const_assert!(main(u2:0) == imported::MyEnum::A);
const_assert!(main(u2:1) == imported::MyEnum::B);
const_assert!(main(u2:2) == imported::MyEnum::C);
)";

  auto import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data));
  EXPECT_THAT(TypecheckV2(kProgram, "main", &import_data),
              IsOkAndHolds(HasTypeInfo(
                  HasNodeWithType("imported::ENUMS[i]", "MyEnum"))));
}

TEST(TypecheckV2Test, ParametricArrayOfImportedType) {
  constexpr std::string_view kImported = R"(
pub type stuff_t = u32;
)";
  constexpr std::string_view kProgram = R"(
import imported;

type stuff_t = imported::stuff_t;
fn do_stuff<N: u32>(stuff: stuff_t[N]) -> stuff_t[N] { stuff }
const X = do_stuff([u32:1, 2, 3]);
)";

  auto import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data));
  EXPECT_THAT(TypecheckV2(kProgram, "main", &import_data),
              IsOkAndHolds(HasTypeInfo(HasNodeWithType("X", "uN[32][3]"))));
}

TEST(TypecheckV2Test, RangeAsArgumentArraySizeMismatch) {
  EXPECT_THAT(
      R"(
pub fn pass_back(input: u32[5]) -> u32[5] {
    input
}

fn test() {
    pass_back(0..4);
}

)",
      TypecheckFails(
          HasTypeMismatch("u32[5]", "u3[(4 as s32 - 0 as s32) as u32]")));
}

TEST(TypecheckV2Test, ConcatWithInvocationAsArgument) {
  EXPECT_THAT(R"(
fn add(vx: u6, vy: u32) -> u32 { vx as u32 + vy }

fn val() -> u3 { 5 }

fn test() {
    add(val() ++ 5, 5);
}
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("test", "() -> ()"),
                                      HasNodeWithType("val() ++ 5", "uN[6]"))));
}

TEST(TypecheckV2Test, ConcatWithInvocationAsArgumentInferredParametric) {
  EXPECT_THAT(R"(
fn add<N: u32>(vx: uN[N], vy: u32) -> u32 { vx as u32 + vy }

fn val<N: u32>() -> uN[N] { 1 }

fn test() {
    add(val<3>() ++ val<3>(), 5);
}
)",
              TypecheckSucceeds(
                  AllOf(HasNodeWithType("test", "() -> ()"),
                        HasNodeWithType("val<3>() ++ val<3>()", "uN[6]"))));
}

TEST(TypecheckV2Test, ParametricArraySizeInRange) {
  EXPECT_THAT(
      R"(
fn sum_elements<N: u32>(elements: u32[N]) -> u32 {
    let result: u32 = for (i, accum) in u32:0..array_size(elements) {
        accum + elements[i]
    }(u32:0);
    result
}

pub fn sum_elements_2(elements: u32[2]) -> u32 {
    sum_elements(elements)
}
)",
      TypecheckSucceeds(HasNodeWithType("sum_elements(elements)", "uN[32]")));
}

TEST(TypecheckV2Test, FuzzTestDomainArrayMismatch) {
  EXPECT_THAT(
      R"(
#[fuzz_test(domains=`[u32:0, 16384]`)]
fn f(x: u8) {}
)",
      TypecheckFails(HasSubstr("Fuzz test domain bit count (32) does not "
                               "match parameter bit count (8)")));
}

TEST(TypecheckV2Test, FuzzTestDomainsEmptyTupleAlwaysMatches) {
  EXPECT_THAT(R"(
#[fuzz_test(domains=`()`)]
fn f(x: u8) {}
)",
              TypecheckSucceeds(::testing::_));
}

TEST(TypecheckV2Test, FuzzTestEmptyTupleDomainAlwaysMatchesMultipleParams) {
  EXPECT_THAT(R"(
#[fuzz_test(domains=`u32:0..1, ()`)]
fn f(x: u32, y: u8) {}
)",
              TypecheckSucceeds(::testing::_));
}

TEST(TypecheckV2Test, FuzzTestTupleDomainSuccess) {
  EXPECT_THAT(R"(
const D = (u32:0..1, u8:0..2);
#[fuzz_test(domains=`D`)]
fn f(x: (u32, u8)) {}
)",
              TypecheckSucceeds(::testing::_));
}

TEST(TypecheckV2Test, FuzzTestTupleDomainWithAliasSuccess) {
  EXPECT_THAT(R"(
type my_tuple = (u32, u8);
const D = (u32:0..1, u8:0..2);
#[fuzz_test(domains=`D`)]
fn f(x: my_tuple) {}
)",
              TypecheckSucceeds(::testing::_));
}

TEST(TypecheckV2Test, FuzzTestTupleDomainNestedSuccess) {
  EXPECT_THAT(R"(
const D = ((u32:0..1, u8:0..2), u16:0..3);
#[fuzz_test(domains=`D`)]
fn f(x: ((u32, u8), u16)) {}
)",
              TypecheckSucceeds(::testing::_));
}

TEST(TypecheckV2Test, FuzzTestTupleDomainSizeMismatch) {
  EXPECT_THAT(R"(
const D = (u32:0..1, u8:0..2, u16:0..3);
#[fuzz_test(domains=`D`)]
fn f(x: (u32, u8)) {}
)",
              TypecheckFails(HasSubstr(
                  "Fuzz test domain tuple size (3) does "
                  "not match parameter 'x: (u32, u8)' tuple size (2)")));
}

TEST(TypecheckV2Test, FuzzTestTupleDomainTypeMismatch) {
  EXPECT_THAT(
      R"(
const D = (u32:0..1, u16:0..2);
#[fuzz_test(domains=`D`)]
fn f(x: (u32, u8)) {}
)",
      TypecheckFails(HasSubstr("is not compatible with parameter")));
}

TEST(TypecheckV2Test, FuzzTestTupleDomainNotATuple) {
  EXPECT_THAT(R"(
#[fuzz_test(domains=`u32:0..1`)]
fn f(x: (u32, u8)) {}
)",
              TypecheckFails(HasSubstr("is not compatible with parameter")));
}

TEST(TypecheckV2Test, FuzzTestTupleDomainParamNotATuple) {
  EXPECT_THAT(R"(
const D = (u32:0..1, u32:0..2);
#[fuzz_test(domains=`D`)]
fn f(x: u32) {}
)",
              TypecheckFails(HasSubstr("Fuzz test domain implies a tuple type, "
                                       "but parameter is not a tuple")));
}

TEST(TypecheckV2Test, FuzzTestTupleDomainDirectSuccess) {
  EXPECT_THAT(R"(
#[fuzz_test(domains=`(u32:0..1, u8:0..2)`)]
fn f(x: (u32, u8)) {}
)",
              TypecheckSucceeds(::testing::_));
}

TEST(TypecheckV2Test, FuzzTestTupleDomainMismatch) {
  EXPECT_THAT(
      R"(
#[fuzz_test(domains=`u32:0..1, u8:0..2`)]
fn f(x: (u32, u8)) {}
)",
      TypecheckFails(HasSubstr("fuzz_test attribute has 2 domain arguments, "
                               "but function `f` has 1 parameter")));
}

TEST(TypecheckV2Test, RangeAsArgument) {
  EXPECT_THAT(
      R"(
pub fn pass_back(input: s32[4]) -> s32[4] {
    input
}

fn test() {
    pass_back(-4..0);
}

)",
      TypecheckSucceeds(
          AllOf(HasNodeWithType("test", "() -> ()"),
                HasNodeWithType("pass_back(-4..0)", "sN[32][4]"))));
}

TEST(TypecheckV2Test, RangeExpr) {
  EXPECT_THAT(
      R"(
const X = u32:1..u32:4;
const X1 = 1..u32:4;
const X2 = u32:1..4;
const Y:s32[5] = s32:0..s32:5;
const Z:s17[1] = -1..0;
)",
      TypecheckSucceeds(AllOf(
          HasNodeWithType("X", "uN[32][3]"), HasNodeWithType("X1", "uN[32][3]"),
          HasNodeWithType("X2", "uN[32][3]"), HasNodeWithType("Y", "sN[32][5]"),
          HasNodeWithType("Z", "sN[17][1]"))));
}

TEST(TypecheckV2Test, RangeExprConstExpr) {
  EXPECT_THAT(R"(
fn foo() -> s16 {
  s16:4
}
const A = s16:6;
const X = foo()..(A * 2);
)",
              TypecheckSucceeds(HasNodeWithType("X", "sN[16][8]")));
}

TEST(TypecheckV2Test, RangeExprEmptyRange) {
  EXPECT_THAT(R"(
const A = s8:4;
const X = A..s8:4;
const_assert!(X == []);
)",
              TypecheckSucceeds(HasNodeWithType("X", "sN[8]")));
}

TEST(TypecheckV2Test, RangeExprEmptyRangeSignedMax) {
  EXPECT_THAT(R"(
const A = s32::MAX;
const X = A..A;
const_assert!(X == []);
)",
              TypecheckSucceeds(HasNodeWithType("X", "sN[32]")));
}

TEST(TypecheckV2Test, RangeExprEmptyRangeUnsignedMax) {
  EXPECT_THAT(R"(
const A = u32::MAX;
const X = A..A;
const_assert!(X == []);
)",
              TypecheckSucceeds(HasNodeWithType("X", "uN[32]")));
}

TEST(TypecheckV2Test, RangeExprSignednessMismatch) {
  EXPECT_THAT(R"(const X = u32:1..s32:2;)",
              TypecheckFails(HasSignednessMismatch("s32", "u32")));
}

TEST(TypecheckV2Test, RangeExprSizeMismatch) {
  EXPECT_THAT(
      R"(const X:u32[4] = u32:1..u32:3;)",
      TypecheckFails(HasTypeMismatch("3 as s32 - u32:1 as s32", "u32[4]")));
}

TEST(TypecheckV2Test, RangeExprTypeAnnotationMismatch) {
  EXPECT_THAT(R"(const X:u32[4] = 0..u16:4;)",
              TypecheckFails(HasSizeMismatch("u16", "u32")));
}

TEST(TypecheckV2Test, RangeAsArgumentTypeMismatch) {
  EXPECT_THAT(
      R"(
pub fn pass_back(input: u32[4]) -> u32[4] {
    input
}

fn test() {
    pass_back(-4..0);
}

)",
      TypecheckFails(HasSignednessMismatch("u32", "s3")));
}

TEST(TypecheckV2Test, LetSimpleTupleMismatch) {
  EXPECT_THAT(R"(
fn f() -> bits[3] {
  let (x, y) = (u32:1, bits[4]:3);
  y
}
)",
              TypecheckFails(HasSizeMismatch("bits[4]", "bits[3]")));
}

TEST(TypecheckV2Test, LetSimpleTuple) {
  EXPECT_THAT(R"(
fn f() -> bits[4] {
  let (x, y) = (u32:1, bits[4]:3);
  y
}
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("x", "uN[32]"),
                                      HasNodeWithType("y", "uN[4]"))));
}

TEST(TypecheckV2Test, ImplMethodCalledOnIntFails) {
  EXPECT_THAT(R"(
const X = u32:1;
const Y = uN[X.area()]:0;
)",
              TypecheckFails(HasSubstr(
                  "Cannot invoke method `area` on non-struct type `u32`")));
}

TEST(TypecheckV2Test, RangeExprInclusiveEndUnsigned) {
  EXPECT_THAT(
      R"(
const X = u4:0..=u4:15;
const Y = u4:0..=15;
const_assert!(X == u4[16]:[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
const_assert!(X == Y);
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[4][16]"),
                              HasNodeWithType("Y", "uN[4][16]"))));
}

TEST(TypecheckV2Test, RangeExprInclusiveEndSigned) {
  EXPECT_THAT(
      R"(
const X = s4:-8..=s4:7;
const Y = s4:-8..=7;
const_assert!(X == s4[16]:[-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]);
const_assert!(X == Y);
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("X", "sN[4][16]"),
                              HasNodeWithType("Y", "sN[4][16]"))));
}

TEST(TypecheckV2Test, RangeExprInclusiveEndOneElement) {
  EXPECT_THAT(
      R"(
const X = s32:2147483647..=s32:2147483647;
const_assert!(X == s32[1]:[2147483647]);
)",
      TypecheckSucceeds(HasNodeWithType("X", "sN[32][1]")));
}

TEST(TypecheckV2Test, RangeExprInclusiveEndConstant) {
  EXPECT_THAT(
      R"(
const A = u32:1;
const X = u32:1..=A;
const_assert!(X == u32[1]:[1]);
)",
      TypecheckSucceeds(HasNodeWithType("X", "uN[32][1]")));
}

TEST(TypecheckV2Test, RangeExprIntMaxInvalid) {
  EXPECT_THAT(
      R"(
const X = u4:0..=u4:16;
)",
      TypecheckFails(
          HasSubstr("Value '16' does not fit in the bitwidth of a uN[4]")));
}

TEST(TypecheckV2Test, RangeExprIntMaxInvalidSigned) {
  EXPECT_THAT(
      R"(
const X = s4:0..=s4:8;
)",
      TypecheckFails(
          HasSubstr("Value '8' does not fit in the bitwidth of a sN[4]")));
}

TEST(TypecheckV2Test, RangeExprTypeAnnotationConstExpr) {
  EXPECT_THAT(R"(
fn foo() -> s16 {
  s16:4
}
const A = s16:6;
const X : s16[foo() as u32 + 4] = foo()..(A * 2);
)",
              TypecheckSucceeds(HasNodeWithType("X", "sN[16][8]")));
}

TEST(TypecheckV2Test, RangeExprNonConstExpr) {
  EXPECT_THAT(R"(
fn foo(a : u32) {
  let A = u32:1..a;
}
)",
              TypecheckFails(HasSubstr("is not constexpr")));
}

TEST(TypecheckV2Test, RangeExprCheckInvalidTypePair) {
  EXPECT_THAT(R"(
type Pair = (u32, u32);
const A : Pair = (1, 2);
const B : Pair = (3, 4);
const X = A..B;
)",
              TypecheckFails(HasSubstr("Cannot cast from type")));
}

TEST(TypecheckV2Test, RangeExprInvalidTypeFunc) {
  EXPECT_THAT(R"(
fn foo() {}
fn bar() {}
const X = foo..bar;
)",
              TypecheckFails(HasSubstr("Cannot cast from type")));
}

TEST(TypecheckV2Test, RangeExprNegativeRange) {
  EXPECT_THAT(R"(
const A = s8:4;
const X = A..s8:3;
)",
              TypecheckFails(HasSubstr("Range expr `A..s8:3` start value 4 "
                                       "is greater than end value 3")));
}

TEST(TypecheckV2Test, FuzzTestArrayDomainSuccess) {
  EXPECT_THAT(R"(
#[fuzz_test(domains=`[u32:0..5, u32:1..6]`)]
fn f(x: u32[2]) {}
)",
              TypecheckSucceeds(::testing::_));
}

TEST(TypecheckV2Test, FuzzTestArrayDomainSizeMismatch) {
  EXPECT_THAT(R"(
#[fuzz_test(domains=`[u32:0..5, u32:1..6, u32:2..7]`)]
fn f(x: u32[2]) {}
)",
              TypecheckFails(
                  HasSubstr("Fuzz test domain array size (3) does "
                            "not match parameter 'x: u32[2]' array size (2)")));
}

TEST(TypecheckV2Test, FuzzTestArrayDomainElementTypeMismatch) {
  EXPECT_THAT(R"(
#[fuzz_test(domains=`[u8:0..5, u8:1..6]`)]
fn f(x: u32[2]) {}
)",
              TypecheckFails(HasSubstr("is not compatible with parameter")));
}

TEST(TypecheckV2Test, FuzzTestArrayDomainIsARangeFailure) {
  EXPECT_THAT(R"(
#[fuzz_test(domains=`u32:0..5`)]
fn f(x: u32[5]) {}
)",
              TypecheckFails(
                  HasSubstr("Expected array of domains for array parameter")));
}

TEST(TypecheckV2Test, FuzzTestArrayDomainIsAScalarFailure) {
  EXPECT_THAT(R"(
#[fuzz_test(domains=`u32:42`)]
fn f(x: u32[5]) {}
)",
              TypecheckFails(
                  HasSubstr("Expected array of domains for array parameter")));
}

}  // namespace
}  // namespace xls::dslx
