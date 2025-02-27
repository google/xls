// Copyright 2024 The XLS Authors
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
#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/type_system/typecheck_test_utils.h"
#include "xls/dslx/type_system_v2/type_system_test_utils.h"

namespace xls::dslx {
namespace {

using ::absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::HasSubstr;

// Verifies that a type info string contains the given node string and type
// string combo.
MATCHER_P2(HasNodeWithType, node, type, "") {
  return ExplainMatchResult(
      HasSubstr(absl::Substitute("node: `$0`, type: $1", node, type)), arg,
      result_listener);
}

// Verifies the type produced by `TypecheckV2`, for the topmost node only, in a
// simple AST (typically a one-liner). The `arg` is the DSLX code and `expected`
// is the type string.
MATCHER_P(TopNodeHasType, expected, "") {
  absl::StatusOr<TypecheckResult> result = TypecheckV2(arg);
  if (!result.ok()) {
    *result_listener << "Failed to typecheck: `" << arg
                     << "`; status: " << result.status();
    return false;
  }
  absl::StatusOr<std::string> type_info_string = TypeInfoToString(result->tm);
  if (!type_info_string.ok()) {
    *result_listener << "Failed to convert type info to string; status: "
                     << type_info_string.status();
    return false;
  }
  bool matched = ExplainMatchResult(HasNodeWithType(arg, expected),
                                    *type_info_string, result_listener);
  if (!matched) {
    *result_listener << "Type info: " << *type_info_string;
  }
  return matched;
}

// Verifies that the `TypecheckV2` output contains a one-line statement block
// with the given type.
MATCHER_P2(HasOneLineBlockWithType, expected_line, expected_type, "") {
  bool matched = ExplainMatchResult(
      HasSubstr(absl::Substitute("node: `{\n    $0\n}`, type: $1",
                                 expected_line, expected_type)),
      arg, result_listener);
  if (!matched) {
    *result_listener << "Type info: " << arg;
  }
  return matched;
}

// Verifies that `TypecheckV2` fails for the given DSLX code, using `matcher`
// for the error string. The `arg` is the DSLX code.
MATCHER_P(TypecheckFails, matcher, "") {
  return ExplainMatchResult(
      StatusIs(absl::StatusCode::kInvalidArgument, matcher), TypecheckV2(arg),
      result_listener);
}

// Verifies that `TypecheckV2` succeeds for the given DSLX code and the
// resulting type info string satisfies the given `matcher`.
MATCHER_P(TypecheckSucceeds, matcher, "") {
  absl::StatusOr<TypecheckResult> result = TypecheckV2(arg);
  if (!result.ok()) {
    *result_listener << "Failed to typecheck: `" << arg
                     << "`; status: " << result.status();
    return false;
  }
  absl::StatusOr<std::string> type_info_string = TypeInfoToString(result->tm);
  if (!type_info_string.ok()) {
    *result_listener << "Failed to convert type info to string; status: "
                     << type_info_string.status();
    return false;
  }
  bool matched =
      ExplainMatchResult(matcher, *type_info_string, result_listener);
  if (!matched) {
    *result_listener << "Type info: " << *type_info_string;
  }
  return matched;
}

TEST(TypecheckV2Test, GlobalIntegerConstantWithNoTypeAnnotations) {
  EXPECT_THAT("const X = 3;",
              TypecheckFails(AllOf(HasSubstr("A variable or constant cannot be "
                                             "defined with an implicit type."),
                                   HasSubstr("const X = 3;"))));
}

TEST(TypecheckV2Test, GlobalIntegerConstantWithTypeAnnotationOnLiteral) {
  EXPECT_THAT("const X = u32:3;", TopNodeHasType("uN[32]"));
}

TEST(TypecheckV2Test,
     GlobalIntegerConstantWithTooSmallAnnotationOnLiteralFails) {
  EXPECT_THAT("const X = u4:65536;",
              TypecheckFails(HasSubstr(
                  "Value '65536' does not fit in the bitwidth of a uN[4]")));
}

TEST(TypecheckV2Test, GlobalIntegerConstantWithTypeAnnotationOnName) {
  EXPECT_THAT("const X: s24 = 3;", TopNodeHasType("sN[24]"));
}

TEST(TypecheckV2Test, GlobalIntegerConstantWithSameTypeAnnotationOnBothSides) {
  EXPECT_THAT("const X: s24 = s24:3;", TopNodeHasType("sN[24]"));
}

TEST(TypecheckV2Test, GlobalIntegerConstantWithSignednessConflictFails) {
  EXPECT_THAT("const X: s24 = u24:3;",
              TypecheckFails(HasSignednessMismatch("u24", "s24")));
}

TEST(TypecheckV2Test, GlobalIntegerConstantEqualsAnotherConstant) {
  EXPECT_THAT(
      R"(
const X = u2:3;
const Y = X;
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("const X = u2:3;", "uN[2]"),
                              HasNodeWithType("const Y = X;", "uN[2]"))));
}

TEST(TypecheckV2Test, GlobalIntegerConstantEqualsSumOfLiterals) {
  EXPECT_THAT("const X = u32:3 + 1 + 5;", TopNodeHasType("uN[32]"));
}

TEST(TypecheckV2Test, GlobalIntegerConstantWithAutoLiteralCoercedToSigned) {
  // Here the auto type of the `3` would be `u2`, but the context should promote
  // it and make it signed.
  EXPECT_THAT("const X = s32:2 + 3;", TopNodeHasType("sN[32]"));
}

TEST(TypecheckV2Test, GlobalIntegerConstantWithNegativeAutoLiteral) {
  EXPECT_THAT("const X = s32:2 + -3;", TopNodeHasType("sN[32]"));
}

TEST(TypecheckV2Test,
     GlobalIntegerConstantUnsignedWithNegativeAutoLiteralFails) {
  EXPECT_THAT("const X = u32:2 + -3;",
              TypecheckFails(HasSignednessMismatch("sN[3]", "u32")));
}

TEST(TypecheckV2Test, GlobalIntegerConstantEqualsSumOfConstantsAndLiterals) {
  EXPECT_THAT(R"(
const X = u32:3;
const Y: u32 = 4;
const Z = X + 1 + Y + 2;
)",
              TypecheckSucceeds(AllOf(
                  HasNodeWithType("const X = u32:3;", "uN[32]"),
                  HasNodeWithType("const Y: u32 = 4;", "uN[32]"),
                  HasNodeWithType("const Z = X + 1 + Y + 2;", "uN[32]"))));
}

TEST(TypecheckV2Test, GlobalIntegerConstantEqualsSumOfAscendingAutoSizes) {
  EXPECT_THAT(R"(
const X = u32:1;
const Z = 1 + 2 + 3 + 4 + X;
)",
              TypecheckSucceeds(AllOf(
                  HasNodeWithType("const X = u32:1;", "uN[32]"),
                  HasNodeWithType("const Z = 1 + 2 + 3 + 4 + X;", "uN[32]"))));
}

TEST(TypecheckV2Test, GlobalIntegerConstantWithCoercionOfAutoToSigned) {
  EXPECT_THAT(R"(
const Z = 1 + s2:1;
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("1", "sN[2]"),
                                      HasNodeWithType("Z", "sN[2]"))));
}

TEST(TypecheckV2Test, GlobalIntegerConstantWithCoercionOfByAnotherAuto) {
  // Promote [auto u2, auto s2] to auto s3 due to the extra bit needed to fit
  // `2` in a signed value. The `s3:0` term then just avoids the prohibition on
  // a constant defined with a completely implicit type.
  EXPECT_THAT(R"(
const Z = 2 + -1 + s3:0;
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("2", "sN[3]"),
                                      HasNodeWithType("Z", "sN[3]"))));
}

TEST(TypecheckV2Test,
     GlobalIntegerConstantWithImpossibleAutoByAutoPromotionFails) {
  // Promote [auto u2, auto s2] to auto s3 due to the extra bit needed to fit
  // `2` in a signed value, and verify that it breaks the limit imposed by an
  // explicit s2.
  EXPECT_THAT("const Z = 2 + -1 + s2:0;",
              TypecheckFails(HasSizeMismatch("s2", "sN[3]")));
}

TEST(TypecheckV2Test, ImpossibleCoercionOfAutoToSignedFails) {
  EXPECT_THAT("const Z = 3 + s2:1;",
              TypecheckFails(HasSizeMismatch("s2", "sN[3]")));
}

TEST(TypecheckV2Test, GlobalIntegerConstantEqualsSumOfConstantAndTupleFails) {
  EXPECT_THAT(R"(
const X = u32:3;
const Y = (u32:1, u32:2);
const Z = X + Y;
)",
              TypecheckFails(HasSubstr("type mismatch")));
}

TEST(TypecheckV2Test, GlobalIntegerConstantWithTupleAnnotationOnLiteral) {
  EXPECT_THAT("const X = (u32, u32):3;",
              TypecheckFails(HasSubstr(
                  "Non-bits type used to define a numeric literal.")));
}

TEST(TypecheckV2Test,
     GlobalIntegerConstantEqualsDifferenceOfConstantsAndLiterals) {
  EXPECT_THAT(R"(
const X = u32:3;
const Y: u32 = 4;
const Z = X - 1 - Y - 2;
)",
              TypecheckSucceeds(AllOf(
                  HasNodeWithType("const X = u32:3;", "uN[32]"),
                  HasNodeWithType("const Y: u32 = 4;", "uN[32]"),
                  HasNodeWithType("const Z = X - 1 - Y - 2;", "uN[32]"))));
}

TEST(TypecheckV2Test,
     GlobalIntegerConstantEqualsProductOfConstantsAndLiterals) {
  EXPECT_THAT(R"(
const X = u32:3;
const Y: u32 = 4;
const Z = X * 1 * Y * 2;
)",
              TypecheckSucceeds(AllOf(
                  HasNodeWithType("const X = u32:3;", "uN[32]"),
                  HasNodeWithType("const Y: u32 = 4;", "uN[32]"),
                  HasNodeWithType("const Z = X * 1 * Y * 2;", "uN[32]"))));
}

TEST(TypecheckV2Test,
     GlobalIntegerConstantEqualsQuotientOfConstantsAndLiterals) {
  EXPECT_THAT(R"(
const X = u32:3;
const Y: u32 = 4;
const Z = X / 1 / Y / 2;
)",
              TypecheckSucceeds(AllOf(
                  HasNodeWithType("const X = u32:3;", "uN[32]"),
                  HasNodeWithType("const Y: u32 = 4;", "uN[32]"),
                  HasNodeWithType("const Z = X / 1 / Y / 2;", "uN[32]"))));
}

TEST(TypecheckV2Test,
     GlobalIntegerConstantEqualsBitwiseAndOfConstantsAndLiterals) {
  EXPECT_THAT(R"(
const X = u32:3;
const Y: u32 = 4;
const Z = X & 1 & Y & 2;
)",
              TypecheckSucceeds(AllOf(
                  HasNodeWithType("const X = u32:3;", "uN[32]"),
                  HasNodeWithType("const Y: u32 = 4;", "uN[32]"),
                  HasNodeWithType("const Z = X & 1 & Y & 2;", "uN[32]"))));
}

TEST(TypecheckV2Test,
     GlobalIntegerConstantEqualsBitwiseOrOfConstantsAndLiterals) {
  EXPECT_THAT(R"(
const X = u32:3;
const Y: u32 = 4;
const Z = X | 1 | Y | 2;
)",
              TypecheckSucceeds(AllOf(
                  HasNodeWithType("const X = u32:3;", "uN[32]"),
                  HasNodeWithType("const Y: u32 = 4;", "uN[32]"),
                  HasNodeWithType("const Z = X | 1 | Y | 2;", "uN[32]"))));
}

TEST(TypecheckV2Test,
     GlobalIntegerConstantEqualsBitwiseXorOfConstantsAndLiterals) {
  EXPECT_THAT(R"(
const X = u32:3;
const Y: u32 = 4;
const Z = X ^ 1 ^ Y ^ 2;
)",
              TypecheckSucceeds(AllOf(
                  HasNodeWithType("const X = u32:3;", "uN[32]"),
                  HasNodeWithType("const Y: u32 = 4;", "uN[32]"),
                  HasNodeWithType("const Z = X ^ 1 ^ Y ^ 2;", "uN[32]"))));
}

TEST(TypecheckV2Test, GlobalIntegerConstantEqualsModOfConstantsAndLiterals) {
  EXPECT_THAT(R"(
const X = u32:3;
const Y: u32 = 4;
const Z = X % 1 % Y % 2;
)",
              TypecheckSucceeds(AllOf(
                  HasNodeWithType("const X = u32:3;", "uN[32]"),
                  HasNodeWithType("const Y: u32 = 4;", "uN[32]"),
                  HasNodeWithType("const Z = X % 1 % Y % 2;", "uN[32]"))));
}

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

TEST(TypecheckV2Test, ConcatOfBitsAsImplicitParametric) {
  EXPECT_THAT(R"(
fn f<A: u32>(a: uN[A]) -> uN[A] { a }
const X = f(u16:0 ++ u32:0);
)",
              TypecheckSucceeds(HasNodeWithType("X", "uN[48]")));
}

TEST(TypecheckV2Test, ConcatOfArrayAsImplicitParametric) {
  EXPECT_THAT(R"(
fn f<A: u32>(a: u16[A]) -> u16[A] { a }
const X = f([u16:0, 1, 2] ++ [u16:20]);
)",
              TypecheckSucceeds(HasNodeWithType("X", "uN[16][4]")));
}

TEST(TypecheckV2Test, SumOfConcatsOfBits) {
  EXPECT_THAT("const X = (u16:0 ++ u32:0) + u48:10;", TopNodeHasType("uN[48]"));
}

TEST(TypecheckV2Test, GlobalBoolConstantEqualsComparisonOfUntypedLiterals) {
  EXPECT_THAT("const Z = 4 > 1;", TopNodeHasType("uN[1]"));
}

TEST(TypecheckV2Test, GlobalBoolConstantEqualsComparisonOfTypedLiterals) {
  EXPECT_THAT("const Z = u32:4 < u32:1;", TopNodeHasType("uN[1]"));
}

TEST(TypecheckV2Test, GlobalBoolConstantEqualsComparisonOfLiteralsWithOneType) {
  EXPECT_THAT("const Z = 4 < s32:1;", TopNodeHasType("uN[1]"));
}

TEST(TypecheckV2Test, GlobalBoolConstantEqualsComparisonOfVariables) {
  EXPECT_THAT(R"(
const X = u32:3;
const Y = u32:4;
const Z = Y >= X;
)",
              TypecheckSucceeds(HasNodeWithType("Z", "uN[1]")));
}

TEST(TypecheckV2Test, GlobalBoolConstantEqualsComparisonOfExprs) {
  EXPECT_THAT(R"(
const X = s24:3;
const Y = s24:4;
const Z = (Y + X * 2) == (1 - Y);
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("(Y + X * 2)", "sN[24]"),
                                      HasNodeWithType("(1 - Y)", "sN[24]"),
                                      HasNodeWithType("Z", "uN[1]"))));
}

TEST(TypecheckV2Test, ComparisonAsFunctionArgument) {
  EXPECT_THAT(R"(
fn foo(a: bool) -> bool { a }
const Y = foo(1 != 2);
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("1 != 2", "uN[1]"),
                                      HasNodeWithType("1", "uN[2]"),
                                      HasNodeWithType("2", "uN[2]"))));
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

TEST(TypecheckV2Test, ComparisonAsParametricArgument) {
  EXPECT_THAT(R"(
fn foo<S: bool>(a: xN[S][32]) -> xN[S][32] { a }
const Y = foo<{2 > 1}>(s32:5);
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("Y", "sN[32]"),
                                      HasNodeWithType("2", "uN[2]"),
                                      HasNodeWithType("1", "uN[2]"))));
}

TEST(TypecheckV2Test, ComparisonAsParametricArgumentWithConflictFails) {
  EXPECT_THAT(R"(
fn foo<S: bool>(a: xN[S][32]) -> xN[S][32] { a }
const Y = foo<{2 > 1}>(u32:5);
)",
              TypecheckFails(HasSignednessMismatch("xN[1][32]", "u32")));
}

TEST(TypecheckV2Test, ComparisonAndSumAsParametricArguments) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
const X = u32:1;
fn foo<S: bool, N: u32>(a: xN[S][N]) -> xN[S][N] { a }
const Y = foo<{X == 1}, {X + 3}>(s4:3);
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(type_info_string, HasSubstr("node: `Y`, type: sN[4]"));
}

TEST(TypecheckV2Test, ComparisonAndSumParametricArgumentsWithConflictFails) {
  EXPECT_THAT(R"(
const X = u32:1;
fn foo<S: bool, N: u32>(a: xN[S][N]) -> xN[S][N] { a }
const Y = foo<{X == 1}, {X + 4}>(s4:3);
)",
              TypecheckFails(HasSizeMismatch("xN[1][5]", "s4")));
}

TEST(TypecheckV2Test, ExplicitParametricExpressionMismatchingBindingTypeFails) {
  EXPECT_THAT(R"(
const X = u32:1;
fn foo<N: u32>(a: uN[N]) -> uN[N] { a }
const Y = foo<{X == 1}>(s4:3);
)",
              TypecheckFails(HasSizeMismatch("bool", "u32")));
}

TEST(TypecheckV2Test,
     GlobalBoolConstantEqualsComparisonOfConflictingTypedLiteralsFails) {
  EXPECT_THAT("const Z = u32:4 >= s32:1;",
              TypecheckFails(HasSignednessMismatch("s32", "u32")));
}

TEST(TypecheckV2Test,
     GlobalIntegerConstantEqualsAnotherConstantWithAnnotationOnName) {
  EXPECT_THAT(
      R"(
const X: u32 = 3;
const Y = X;
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("const X: u32 = 3;", "uN[32]"),
                              HasNodeWithType("const Y = X;", "uN[32]"))));
}

TEST(TypecheckV2Test, GlobalIntegerConstantRefWithSignednessConflictFails) {
  EXPECT_THAT(R"(
const X:u32 = 3;
const Y:s32 = X;
)",
              TypecheckFails(HasSignednessMismatch("uN[32]", "s32")));
}

TEST(TypecheckV2Test, GlobalIntegerConstantWithTwoLevelsOfReferences) {
  EXPECT_THAT(
      R"(
const X: s20 = 3;
const Y = X;
const Z = Y;
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("const X: s20 = 3;", "sN[20]"),
                              HasNodeWithType("const Y = X;", "sN[20]"),
                              HasNodeWithType("const Z = Y;", "sN[20]"))));
}

TEST(TypecheckV2Test, GlobalBoolConstantWithNoTypeAnnotations) {
  EXPECT_THAT("const X = true;", TopNodeHasType("uN[1]"));
}

TEST(TypecheckV2Test, GlobalBoolConstantWithTypeAnnotationOnLiteral) {
  EXPECT_THAT("const X = bool:true;", TopNodeHasType("uN[1]"));
}

TEST(TypecheckV2Test, GlobalBoolConstantWithTypeAnnotationOnName) {
  EXPECT_THAT("const X: bool = false;", TopNodeHasType("uN[1]"));
}

TEST(TypecheckV2Test, GlobalBoolConstantWithSameTypeAnnotationOnBothSides) {
  EXPECT_THAT("const X: bool = bool:false;", TopNodeHasType("uN[1]"));
}

TEST(TypecheckV2Test, GlobalBoolConstantAssignedToIntegerFails) {
  EXPECT_THAT("const X: bool = 50;",
              TypecheckFails(HasSizeMismatch("uN[6]", "bool")));
}

TEST(TypecheckV2Test, GlobalBoolConstantWithSignednessConflictFails) {
  EXPECT_THAT("const X: s2 = bool:false;",
              TypecheckFails(HasSizeMismatch("bool", "s2")));
}

TEST(TypecheckV2Test, GlobalBoolConstantWithBitCountConflictFails) {
  // We don't allow this with bool literals, even though it fits.
  EXPECT_THAT("const X: u2 = true;",
              TypecheckFails(HasSizeMismatch("bool", "u2")));
}

TEST(TypecheckV2Test, GlobalBoolConstantEqualsAnotherConstant) {
  EXPECT_THAT(
      R"(
const X = true;
const Y = X;
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("const X = true;", "uN[1]"),
                              HasNodeWithType("const Y = X;", "uN[1]"))));
}

TEST(TypecheckV2Test, GlobalIntegerConstantEqualsBoolConstantFails) {
  EXPECT_THAT(R"(
const X = true;
const Y: u32 = X;
)",
              TypecheckFails(HasSizeMismatch("bool", "u32")));
}

TEST(TypecheckV2Test, GlobalIntegerConstantEqualsSumOfBoolsFails) {
  EXPECT_THAT(R"(
const X = true;
const Y = true;
const Z: u32 = X + Y;
)",
              TypecheckFails(HasSizeMismatch("bool", "u32")));
}

TEST(TypecheckV2Test, GlobalBoolConstantEqualsIntegerConstantFails) {
  EXPECT_THAT(R"(
const X = u32:4;
const Y: bool = X;
)",
              TypecheckFails(HasSizeMismatch("u32", "bool")));
}

TEST(TypecheckV2Test, GlobalTupleConstantWithAnnotatedIntegerLiterals) {
  EXPECT_THAT("const X = (u32:1, u32:2);", TopNodeHasType("(uN[32], uN[32])"));
}

TEST(TypecheckV2Test, GlobalTupleConstantAnnotatedWithBareIntegerLiterals) {
  EXPECT_THAT("const X: (u32, u32) = (1, 2);",
              TypecheckSucceeds(AllOf(HasNodeWithType("X", "(uN[32], uN[32])"),
                                      HasNodeWithType("1", "uN[32]"),
                                      HasNodeWithType("2", "uN[32]"))));
}

TEST(TypecheckV2Test, GlobalTupleConstantWithIntegerAnnotationFails) {
  EXPECT_THAT("const X: u32 = (1, 2);",
              TypecheckFails(HasTypeMismatch("(uN[1], uN[2])", "u32")));
}

TEST(TypecheckV2Test, GlobalTupleConstantWithNestedTuple) {
  EXPECT_THAT(
      "const X: (u32, (s24, u32)) = (1, (-3, 2));",
      TypecheckSucceeds(AllOf(HasNodeWithType("X", "(uN[32], (sN[24], uN[32])"),
                              HasNodeWithType("1", "uN[32]"),
                              HasNodeWithType("(-3, 2)", "(sN[24], uN[32])"),
                              HasNodeWithType("-3", "sN[24]"),
                              HasNodeWithType("2", "uN[32]"))));
}

TEST(TypecheckV2Test, GlobalTupleConstantWithNestedTupleAndTypeViolationFails) {
  EXPECT_THAT("const X: (u32, (u24, u32)) = (1, (-3, 2));",
              TypecheckFails(HasSignednessMismatch("sN[3]", "u24")));
}

TEST(TypecheckV2Test, GlobalTupleConstantWithNestedTupleAndTypeConflict) {
  EXPECT_THAT("const X: (u32, (u24, u32)) = (1, (s24:3, 2));",
              TypecheckFails(HasSignednessMismatch("s24", "u24")));
}

TEST(TypecheckV2Test, GlobalTupleConstantReferencingIntegerConstant) {
  EXPECT_THAT(R"(
const X: u32 = 3;
const Y = (X, s24:-1);
)",
              TypecheckSucceeds(HasNodeWithType("const Y = (X, s24:-1);",
                                                "(uN[32], sN[24])")));
}

TEST(TypecheckV2Test, GlobalTupleConstantReferencingTupleConstant) {
  EXPECT_THAT(R"(
const X = (u32:3, s24:-1);
const Y = (X, u32:4);
)",
              TypecheckSucceeds(HasNodeWithType("const Y = (X, u32:4);",
                                                "((uN[32], sN[24]), uN[32])")));
}

TEST(TypecheckV2Test, GlobalTupleConstantWithArrays) {
  EXPECT_THAT("const X = ([u32:1, u32:2], [u32:3, u32:4, u32:5]);",
              TopNodeHasType("(uN[32][2], uN[32][3])"));
}

TEST(TypecheckV2Test, GlobalArrayConstantWithAnnotatedIntegerLiterals) {
  EXPECT_THAT("const X = [u32:1, u32:2];",
              TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[32][2]"),
                                      HasNodeWithType("u32:1", "uN[32]"),
                                      HasNodeWithType("u32:2", "uN[32]"))));
}

TEST(TypecheckV2Test, ArrayWithArrayAnnotation) {
  EXPECT_THAT("const X = u32[2]:[1, 2];",
              TypecheckSucceeds(HasNodeWithType("X", "uN[32][2]")));
}

TEST(TypecheckV2Test, ArrayWithArrayAnnotationWithSignednessMismatchFails) {
  EXPECT_THAT("const X = u32[2]:[-1, 2];",
              TypecheckFails(HasSignednessMismatch("u32", "sN[3]")));
}

TEST(TypecheckV2Test, ArrayWithArrayAnnotationWithSizeMismatchFails) {
  EXPECT_THAT("const X = u8[2]:[1, 65536];",
              TypecheckFails(HasSizeMismatch("u8", "uN[17]")));
}

TEST(TypecheckV2Test, ArrayWithArrayAnnotationWithCountMismatchFails) {
  EXPECT_THAT("const X = u8[2]:[u8:1, 2, 3];",
              TypecheckFails(HasTypeMismatch("u8[2]", "uN[8][3]")));
}

TEST(TypecheckV2Test, AnnotatedEmptyArray) {
  EXPECT_THAT("const X = u8[0]:[];",
              TypecheckSucceeds(HasNodeWithType("X", "uN[8][0]")));
}

TEST(TypecheckV2Test, AnnotatedEmptyArrayMismatchFails) {
  EXPECT_THAT("const X = u8[1]:[];",
              TypecheckFails(HasTypeMismatch("u8[0]", "u8[1]")));
}

TEST(TypecheckV2Test, GlobalConstantEqualsIndexOfTemporaryArray) {
  EXPECT_THAT("const X = [u32:1, u32:2][0];",
              TypecheckSucceeds(HasNodeWithType("X", "uN[32]")));
}

TEST(TypecheckV2Test, GlobalConstantEqualsIndexOfTemporaryArrayUsingBinop) {
  EXPECT_THAT(R"(
const A = u32:1;
const X = [u32:1, u32:2][0 + A];
)",
              TypecheckSucceeds(HasNodeWithType("X", "uN[32]")));
}

TEST(TypecheckV2Test, IndexWithTooLargeIndexTypeFails) {
  EXPECT_THAT("const X = [u32:1, u32:2][u64:0];",
              TypecheckFails(HasSizeMismatch("u64", "u32")));
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
              TypecheckFails(HasSignednessMismatch("sN[2]", "u32")));
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

TEST(TypecheckV2Test, XnAnnotationWithNonBoolLiteralSignednessFails) {
  EXPECT_THAT("const Y = xN[2][32]:1;",
              TypecheckFails(HasSizeMismatch("bool", "uN[2]")));
}

TEST(TypecheckV2Test, XnAnnotationWithNonBoolConstantSignednessFails) {
  EXPECT_THAT(R"(
const X = u32:2;
const Y = xN[X][32]:1;
)",
              TypecheckFails(HasSizeMismatch("bool", "u32")));
}

TEST(TypecheckV2Test, GlobalConstantEqualsIndexOfTemporaryTuple) {
  EXPECT_THAT("const X = (u24:1, u32:2).0;",
              TypecheckSucceeds(HasNodeWithType("X", "uN[24]")));
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

TEST(TypecheckV2Test, GlobalArrayConstantAnnotatedWithBareIntegerLiterals) {
  EXPECT_THAT("const X: u32[2] = [1, 3];",
              TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[32][2]"),
                                      HasNodeWithType("1", "uN[32]"),
                                      HasNodeWithType("3", "uN[32]"))));
}

TEST(TypecheckV2Test, GlobalArrayConstantWithTuples) {
  EXPECT_THAT("const X = [(u32:1, u32:2), (u32:3, u32:4)];",
              TopNodeHasType("(uN[32], uN[32])[2]"));
}

TEST(TypecheckV2Test, GlobalArrayConstantAnnotatedWithTooSmallSizeFails) {
  EXPECT_THAT("const X: u32[2] = [1, 2, 3];",
              TypecheckFails(HasTypeMismatch("uN[32][3]", "u32[2]")));
}

TEST(TypecheckV2Test, GlobalArrayConstantWithIntegerAnnotationFails) {
  EXPECT_THAT("const X: u32 = [1, 2];",
              TypecheckFails(HasTypeMismatch("uN[2][2]", "u32")));
}

TEST(TypecheckV2Test, GlobalArrayConstantWithTypeViolation) {
  EXPECT_THAT("const X: u32[2] = [-3, -2];",
              TypecheckFails(HasSignednessMismatch("sN[3]", "u32")));
}

TEST(TypecheckV2Test, GlobalArrayConstantWithTypeConflict) {
  EXPECT_THAT("const X: u32[2] = [s24:1, s24:2];",
              TypecheckFails(HasSizeMismatch("s24", "u32")));
}

TEST(TypecheckV2Test, GlobalArrayConstantReferencingIntegerConstant) {
  EXPECT_THAT(
      R"(
const X: u32 = 3;
const Y = [X, X];
)",
      TypecheckSucceeds(HasNodeWithType("const Y = [X, X];", "uN[32][2]")));
}

TEST(TypecheckV2Test, GlobalArrayConstantReferencingArrayConstant) {
  EXPECT_THAT(
      R"(
const X = [u32:3, u32:4];
const Y = [X, X];
)",
      TypecheckSucceeds(HasNodeWithType("const Y = [X, X];", "uN[32][2][2]")));
}

TEST(TypecheckV2Test, GlobalArrayConstantCombiningArrayAndIntegerFails) {
  EXPECT_THAT("const X = [u32:3, [u32:4, u32:5]];",
              TypecheckFails(HasTypeMismatch("uN[32][2]", "u32")));
}

TEST(TypecheckV2Test, GlobalArrayConstantWithEllipsis) {
  EXPECT_THAT("const X: u32[5] = [3, 4, ...];",
              TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[32][5]"),
                                      HasNodeWithType("3", "uN[32]"),
                                      HasNodeWithType("4", "uN[32]"))));
}

TEST(TypecheckV2Test, GlobalArrayConstantWithNestedEllipsis) {
  EXPECT_THAT("const X: u32[5][2] = [[5, ...], ...];",
              TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[32][5][2]"),
                                      HasNodeWithType("5", "uN[32]"))));
}

TEST(TypecheckV2Test, GlobalArrayConstantWithEllipsisAndTooSmallSizeFails) {
  EXPECT_THAT("const X: u32[2] = [3, 4, 5, ...];",
              TypecheckFails(HasSubstr("Annotated array size is too small for "
                                       "explicit element count")));
}

TEST(TypecheckV2Test, GlobalArrayConstantWithEllipsisAndNoElementsFails) {
  EXPECT_THAT("const X: u32[2] = [...];",
              TypecheckFails(HasSubstr("Array cannot have an ellipsis (`...`) "
                                       "without an element to repeat")));
}

TEST(TypecheckV2Test, GlobalArrayConstantEmptyWithoutAnnotationFails) {
  EXPECT_THAT(
      "const X = [];",
      TypecheckFails(HasSubstr(
          "A variable or constant cannot be defined with an implicit type.")));
}

TEST(TypecheckV2Test, GlobalArrayConstantEmptyWithAnnotation) {
  EXPECT_THAT("const X: u32[0] = [];", TopNodeHasType("uN[32][0]"));
}

TEST(TypecheckV2Test, GlobalEmptyStructConstant) {
  EXPECT_THAT(R"(
struct S {}
const X = S { };
)",
              TypecheckSucceeds(HasNodeWithType("X", "S {}")));
}

TEST(TypecheckV2Test, GlobalStructConstantWithIntegerMember) {
  EXPECT_THAT(
      R"(
struct S { field: u32 }
const X = S { field: 5 };
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("5", "uN[32]"),
                              HasNodeWithType("X", "S { field: uN[32] }"))));
}

TEST(TypecheckV2Test, GlobalStructConstantWithMultipleMembers) {
  EXPECT_THAT(
      R"(
struct S {
  foo: u32,
  bar: sN[4][3],
  baz: (s24, bool)
}
const X = S { baz: (1, false), foo: 5, bar: [1, 2, 3] };
)",
      TypecheckSucceeds(AllOf(
          HasNodeWithType("(1, false)", "(sN[24], uN[1])"),
          HasNodeWithType("5", "uN[32]"),
          HasNodeWithType("[1, 2, 3]", "sN[4][3]"),
          HasNodeWithType(
              "X", "S { foo: uN[32], bar: sN[4][3], baz: (sN[24], uN[1]) }"))));
}

TEST(TypecheckV2Test,
     GlobalStructConstantWithIntegerMemberSignednessMismatchFails) {
  EXPECT_THAT(R"(
struct S { field: u32 }
const X = S { field: s32:5 };
)",
              TypecheckFails(HasSignednessMismatch("s32", "u32")));
}

TEST(TypecheckV2Test, GlobalStructConstantWithIntegerMemberSizeMismatchFails) {
  EXPECT_THAT(R"(
struct S { field: u32 }
const X = S { field: u64:5 };
)",
              TypecheckFails(HasSizeMismatch("u64", "u32")));
}

TEST(TypecheckV2Test, GlobalStructConstantWithIntegerMemberTypeMismatchFails) {
  EXPECT_THAT(R"(
struct S { field: u32 }
const X = S { field: [u32:1, u32:2] };
)",
              TypecheckFails(HasTypeMismatch("uN[32][2]", "u32")));
}

TEST(TypecheckV2Test, GlobalStructConstantWithMemberDuplicatedFails) {
  EXPECT_THAT(R"(
struct S { x: u32 }
const X = S { x: u32:1, x: u32:2 };
)",
              TypecheckFails(HasSubstr(
                  "Duplicate value seen for `x` in this `S` struct instance")));
}

TEST(TypecheckV2Test, GlobalStructConstantWithMemberMissingFails) {
  EXPECT_THAT(
      R"(
struct S { x: u32 }
const X = S {};
)",
      TypecheckFails(
          HasSubstr("Instance of struct `S` is missing member(s): `x`")));
}

TEST(TypecheckV2Test, GlobalStructConstantWithExtraneousMemberFails) {
  EXPECT_THAT(
      R"(
struct S { x: u32 }
const X = S { x: 1, y: u32:2};
)",
      TypecheckFails(HasSubstr("Struct `S` has no member `y`, but it was "
                               "provided by this instance")));
}

TEST(TypecheckV2Test, GlobalStructInstancePropagation) {
  EXPECT_THAT(
      R"(
struct S { field: u32 }
const X = S { field: 5 };
const Y = X;
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("5", "uN[32]"),
                              HasNodeWithType("Y", "S { field: uN[32] }"))));
}

TEST(TypecheckV2Test, GlobalStructInstanceContainingStructInstance) {
  EXPECT_THAT(R"(
struct S { field: u32 }
struct T { s: S }
const X = T { s: S { field: 5 } };
)",
              TypecheckSucceeds(
                  AllOf(HasNodeWithType("X", "T { s: S { field: uN[32] } }"),
                        HasNodeWithType("5", "uN[32]"))));
}

TEST(TypecheckV2Test, GlobalStructInstanceContainingArray) {
  EXPECT_THAT(
      R"(
struct S { arr: u24[3] }
const X = S { arr: [10, 11, 12] };
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("X", "S { arr: uN[24][3] }"),
                              HasNodeWithType("10", "uN[24]"),
                              HasNodeWithType("11", "uN[24]"),
                              HasNodeWithType("12", "uN[24]"))));
}

TEST(TypecheckV2Test,
     GlobalStructConstantEqualsWrongKindOfStructInstanceFails) {
  EXPECT_THAT(
      R"(
struct S { x: u32 }
struct T { x: u32 }
const X: S = T { x: 1 };
)",
      TypecheckFails(HasTypeMismatch("S", "T")));
}

TEST(TypecheckV2Test,
     GlobalStructConstantEqualsWrongKindOfStructConstantFails) {
  EXPECT_THAT(
      R"(
struct S { x: u32 }
struct T { x: u32 }
const X = S { x: 1 };
const Y: T = X;
)",
      TypecheckFails(HasTypeMismatch("T", "S")));
}

TEST(TypecheckV2Test, GlobalStructConstantEqualsNonStructFails) {
  EXPECT_THAT(
      R"(
struct S { x: u32 }
const X: S = u32:1;
)",
      TypecheckFails(HasTypeMismatch("S", "u32")));
}

TEST(TypecheckV2Test, AccessOfStructMember) {
  EXPECT_THAT(
      R"(
struct S { x: u32 }
const X = S { x: u32:5 }.x;
)",
      TypecheckSucceeds(HasNodeWithType("X", "uN[32]")));
}

TEST(TypecheckV2Test, AccessOfNonexistentStructMemberFails) {
  EXPECT_THAT(
      R"(
struct S { x: u32 }
const X = S { x: u32:5 }.y;
)",
      TypecheckFails(HasSubstr("No member `y` in struct `S`")));
}

TEST(TypecheckV2Test, AccessOfMemberOfNonStructFails) {
  EXPECT_THAT(
      R"(
const X = (u32:1).y;
)",
      TypecheckFails(
          HasSubstr("Invalid access of member `y` of non-struct type: `u32`")));
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

TEST(TypecheckV2Test, SumOfStructMembers) {
  EXPECT_THAT(
      R"(
struct S {
  x: s16,
  y: s16
}
const X = S { x: -1, y: -2 };
const Y = X.x + X.y;
)",
      TypecheckSucceeds(HasNodeWithType("Y", "sN[16]")));
}

TEST(TypecheckV2Test, AccessOfStructMemberFromFunctionReturnValue) {
  EXPECT_THAT(
      R"(
struct S { x: u32 }
fn f(a: u32) -> S { S { x: a } }
const X = f(2).x;

)",
      TypecheckSucceeds(HasNodeWithType("X", "uN[32]")));
}

TEST(TypecheckV2Test, AccessOfStructMemberUsedForParametricInference) {
  EXPECT_THAT(
      R"(
struct S<N: u32> { x: uN[N] }
fn f<N: u32>(a: uN[N]) -> uN[N] { a }
const X = f(S { x: u24:1 }.x);
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

TEST(TypecheckV2Test, GlobalParametricStructConstant) {
  EXPECT_THAT(
      R"(
struct S<N: u32> {
  x: uN[N]
}
const X = S<u32:24>{ x: u24:5 };
)",
      TypecheckSucceeds(HasNodeWithType("X", "S { x: uN[24] }")));
}

TEST(TypecheckV2Test, GlobalParametricStructConstantWithInferredParametric) {
  EXPECT_THAT(
      R"(
struct S<N: u32> {
  x: uN[N]
}
const X = S{ x: u24:5 };
)",
      TypecheckSucceeds(HasNodeWithType("X", "S { x: uN[24] }")));
}

TEST(TypecheckV2Test, ParametricStructWithTooManyParametricsFails) {
  EXPECT_THAT(R"(
struct S<N: u32> {}
const X = S<16, 8>{};
)",
              TypecheckFails(HasSubstr("Too many parametric values supplied")));
}

TEST(TypecheckV2Test,
     ParametricStructWithInsufficientExplicitParametricsFails) {
  // In this case, N is inferrable, but we choose not to infer it.
  EXPECT_THAT(
      R"(
struct S<M: u32, N: u32> {
  x: uN[M],
  y: uN[N]
}
const X = S<32>{x: u32:4, y: u32:5};
)",
      TypecheckFails(HasSubstr("No parametric value provided for `N` in `S`")));
}

TEST(TypecheckV2Test,
     ParametricStructWithOneExplicitAndOneDefaultedParametric) {
  EXPECT_THAT(
      R"(
struct S<M: u32, N: u32 = {M * 2}> {
  x: uN[M],
  y: uN[N]
}
const X = S<16>{x: u16:4, y: u32:5};
)",
      TypecheckSucceeds(HasNodeWithType("X", "S { x: uN[16], y: uN[32] }")));
}

TEST(TypecheckV2Test, GlobalParametricStructConstantWithNominalMismatchFails) {
  EXPECT_THAT(
      R"(
struct S<N: u32> {}
const X: S<24> = S<23> {};
)",
      TypecheckFails(
          AllOf(HasSubstr("Value mismatch for parametric `N` of struct `S`"),
                HasSubstr("u32:23 vs. u32:24"))));
}

TEST(TypecheckV2Test,
     GlobalParametricStructConstantWithInferredAndDefaultParametrics) {
  EXPECT_THAT(
      R"(
struct S<N: u32, S: bool = {N < 25}> {
  x: uN[N],
  y: xN[S][N]
}
const X = S{ x: u24:5, y: 6 };
)",
      TypecheckSucceeds(HasNodeWithType("X", "S { x: uN[24], y: sN[24] }")));
}

TEST(TypecheckV2Test, ParametricStructAsFunctionArgument) {
  EXPECT_THAT(
      R"(
struct S<N: u32> {
  x: uN[N]
}
fn foo(a: S<24>) -> S<24> { a }
const X = foo(S<24> { x: u24:5 });
)",
      TypecheckSucceeds(HasNodeWithType("X", "S { x: uN[24] }")));
}

TEST(TypecheckV2Test, ParametricStructAsFunctionArgumentExplicitMismatchFails) {
  EXPECT_THAT(
      R"(
struct S<N: u32> {
  x: uN[N]
}
fn foo(a: S<24>) -> S<24> { a }
const X = foo(S<25> { x: u25:5 });
)",
      TypecheckFails(
          AllOf(HasSubstr("Value mismatch for parametric `N` of struct `S`"),
                HasSubstr("u32:24 vs. u32:25"))));
}

TEST(TypecheckV2Test,
     ParametricStructAsFunctionArgumentWithImplicitParametric) {
  EXPECT_THAT(
      R"(
struct S<N: u32> {
  x: uN[N]
}
fn foo(a: S<24>) -> S<24> { a }
const X = foo(S { x: u24:5 });
)",
      TypecheckSucceeds(HasNodeWithType("X", "S { x: uN[24] }")));
}

TEST(TypecheckV2Test,
     ParametricStructAsFunctionArgumentWithImplicitParametricMismatchFails) {
  EXPECT_THAT(
      R"(
struct S<N: u32> {
  x: uN[N]
}
fn foo(a: S<24>) -> S<24> { a }
const X = foo(S { x: u25:5 });
)",
      TypecheckFails(HasSubstr("Value mismatch for parametric `N` of struct "
                               "`S`: u32:24 vs. u32:25")));
}

TEST(TypecheckV2Test, ParametricStructAsFunctionReturnValue) {
  EXPECT_THAT(
      R"(
struct S<N: u32> {
  x: uN[N]
}
fn foo(x: u24) -> S<24> { S { x } }
const X = foo(u24:5);
)",
      TypecheckSucceeds(
          AllOf(HasNodeWithType("X", "S { x: uN[24] }"),
                HasNodeWithType("S { x: x }", "S { x: uN[24] }"))));
}

TEST(TypecheckV2Test, ParametricStructFormalReturnValueWithTooManyParametrics) {
  EXPECT_THAT(
      R"(
struct S<N: u32> {}
fn foo() -> S<24, 25> { S {} }
)",
      TypecheckFails(HasSubstr("Too many parametric values supplied")));
}

TEST(TypecheckV2Test,
     ParametricStructFormalReturnValueWithWrongTypeParametric) {
  EXPECT_THAT(
      R"(
struct S<N: u32> {}
fn foo() -> S<u64:24> { S {} }
)",
      TypecheckFails(HasSizeMismatch("u64", "u32")));
}

TEST(TypecheckV2Test, ParametricStructFormalArgumentWithTooManyParametrics) {
  EXPECT_THAT(
      R"(
struct S<N: u32> {}
fn foo(a: S<24, 25>) {}
)",
      TypecheckFails(HasSubstr("Too many parametric values supplied")));
}

TEST(TypecheckV2Test, ParametricStructFormalArgumentWithWrongTypeParametric) {
  EXPECT_THAT(
      R"(
struct S<N: u32> {}
fn foo(a: S<u64:24>) {}
)",
      TypecheckFails(HasSizeMismatch("u64", "u32")));
}

TEST(TypecheckV2Test,
     ParametricStructAsFunctionReturnValueWithExplicitismatchFails) {
  EXPECT_THAT(
      R"(
struct S<N: u32> {
  x: uN[N]
}
fn foo(x: u24) -> S<24> { S<25> { x } }
const X = foo(u24:5);
)",
      TypecheckFails(
          AllOf(HasSubstr("Value mismatch for parametric `N` of struct `S`"),
                HasSubstr("u32:25 vs. u32:24"))));
}

TEST(TypecheckV2Test,
     ParametricStructAsFunctionReturnValueWithImplicitMismatchFails) {
  EXPECT_THAT(
      R"(
struct S<N: u32> {
  x: uN[N]
}
fn foo() -> S<24> { S { x: u25:5 } }
const X = foo();
)",
      TypecheckFails(HasSizeMismatch("u25", "uN[24]")));
}

TEST(TypecheckV2Test,
     ParametricStructAsParametricFunctionArgumentWithImplicitParametric) {
  EXPECT_THAT(
      R"(
struct S<N: u32> {
  x: uN[N]
}
fn foo<N: u32>(a: S<N>) -> S<N> { a }
const X = foo(S { x: u24:5 });
)",
      TypecheckSucceeds(HasNodeWithType("X", "S { x: uN[24] }")));
}

TEST(TypecheckV2Test,
     ParametricFunctionWithImplicitParametricStructReturnExpr) {
  EXPECT_THAT(
      R"(
struct S<N: u32> {
  x: uN[N]
}
fn foo<N: u32>(x: uN[N]) -> S<N> { S { x } }
const X = foo(u24:5);
)",
      TypecheckSucceeds(HasNodeWithType("X", "S { x: uN[24] }")));
}

// See https://github.com/google/xls/issues/1615
TEST(TypecheckV2Test, ParametricStructWithWrongOrderParametricValues) {
  EXPECT_THAT(
      R"(
struct StructFoo<A: u32, B: u32> {
  x: uN[A],
  y: uN[B],
}

fn wrong_order<A: u32, B: u32>(x:uN[A], y:uN[B]) -> StructFoo<B, A> {
  StructFoo<B, A>{x, y}
}

fn test() -> StructFoo<32, 33> {
  wrong_order<32, 33>(2, 3)
}
)",
      TypecheckFails(HasSubstr("Value mismatch for parametric `A` of struct "
                               "`StructFoo`: u32:33 vs. u32:32")));
}

TEST(TypecheckV2Test, ParametricStructWithCorrectReverseOrderParametricValues) {
  EXPECT_THAT(
      R"(
struct StructFoo<A: u32, B: u32> {
  x: uN[A],
  y: uN[B],
}

fn wrong_order<A: u32, B: u32>(x:uN[A], y:uN[B]) -> StructFoo<B, A> {
  StructFoo<B, A>{x:y, y:x}
}

fn test() -> StructFoo<33, 32> {
  wrong_order<32, 33>(2, 3)
}
)",
      TypecheckSucceeds(HasNodeWithType("wrong_order<32, 33>(2, 3)",
                                        "StructFoo { x: uN[33], y: uN[32] }")));
}

TEST(TypecheckV2Test, GlobalParametricStructArrayConstant) {
  EXPECT_THAT(
      R"(
struct S<N: u32> {
  x: uN[N]
}
const X = [ S<u32:24>{ x: u24:5 }, S<u32:24>{ x: u24:6 } ];
)",
      TypecheckSucceeds(HasNodeWithType("X", "S { x: uN[24] }[2]")));
}

TEST(TypecheckV2Test,
     GlobalParametricStructArrayConstantWithImplicitParametric) {
  EXPECT_THAT(
      R"(
struct S<N: u32> {
  x: uN[N]
}
const X = [ S{ x: u24:5 }, S{ x: u24:6 } ];
)",
      TypecheckSucceeds(HasNodeWithType("X", "S { x: uN[24] }[2]")));
}

TEST(TypecheckV2Test, GlobalNonUnifiableParametricStructArrayConstantFails) {
  EXPECT_THAT(
      R"(
struct S<N: u32> {
  x: uN[N]
}
const X = [ S<u32:24>{ x: u24:5 }, S<u32:25>{ x: u25:6 } ];
)",
      TypecheckFails(
          AllOf(HasSubstr("Value mismatch for parametric `N` of struct `S`"),
                HasSubstr("u32:25 vs. u32:24"))));
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

TEST(TypecheckV2Test, ParametricStructWithConstantDimension) {
  EXPECT_THAT(
      R"(
const N = u32:4;
struct S<M: u32> {
  x: uN[M][N]
}
const X = S { x: [u24:1, u24:2, u24:3, u24:4] };
)",
      TypecheckSucceeds(HasNodeWithType("X", "S { x: uN[24][4] }")));
}

// Various samples of actual-argument compatibility with an `xN` field within a
// struct via a struct instantiation expression (based on original
// `typecheck_module_test`).
TEST(TypecheckV2Test, StructInstantiateParametricXnField) {
  EXPECT_THAT(
      R"(
struct XnWrapper<S: bool, N: u32> {
  field: xN[S][N]
}
fn f() -> XnWrapper<false, u32:8> { XnWrapper<false, u32:8> { field: u8:0 } }
fn g() -> XnWrapper<true, u32:8> { XnWrapper<true, u32:8> { field: s8:1 } }
fn h() -> XnWrapper<false, u32:8> { XnWrapper<false, u32:8> { field: xN[false][8]:2 } }
fn i() -> XnWrapper<true, u32:8> { XnWrapper<true, u32:8> { field: xN[true][8]:3 } }
)",
      TypecheckSucceeds(AllOf(
          HasNodeWithType("XnWrapper<false, u32:8> { field: u8:0 }",
                          "XnWrapper { field: uN[8] }"),
          HasNodeWithType("XnWrapper<true, u32:8> { field: s8:1 }",
                          "XnWrapper { field: sN[8] }"),
          HasNodeWithType("XnWrapper<false, u32:8> { field: xN[false][8]:2 }",
                          "XnWrapper { field: uN[8] }"),
          HasNodeWithType("XnWrapper<true, u32:8> { field: xN[true][8]:3 }",
                          "XnWrapper { field: sN[8] }"))));
}

TEST(TypecheckV2Test, StructFunctionArgument) {
  EXPECT_THAT(R"(
struct S { field: u32 }
fn f(s: S) {}
fn g() {
  f(S { field: 2 })
}
)",
              TypecheckSucceeds(AllOf(
                  HasNodeWithType("2", "uN[32]"),
                  HasNodeWithType("S { field: 2 }", "S { field: uN[32] }"))));
}

TEST(TypecheckV2Test, StructFunctionReturnValue) {
  EXPECT_THAT(R"(
struct S { field: u32 }
fn f(value: u32) -> S {
  S { field: value }
}
const X = f(2);
)",
              TypecheckSucceeds(AllOf(
                  HasNodeWithType("const X = f(2);", "S { field: uN[32] }"))));
}

TEST(TypecheckV2Test, InstantiationOfNonStruct) {
  EXPECT_THAT(
      "const X = u32 { foo: 1 };",
      TypecheckFails(HasSubstr(
          "Attempted to instantiate non-struct type `u32` as a struct.")));
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

TEST(TypecheckV2Test, FunctionCallReturningUnitTupleExplicitly) {
  EXPECT_THAT(
      R"(
fn foo() -> () { () }
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
      TypecheckFails(HasTypeMismatch("(u32, u32)", "uN[32][2]")));
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

TEST(TypecheckV2Test, SumOfLiteralsAndParametricFunctionCall) {
  EXPECT_THAT(
      R"(
fn foo<N: u32>() -> uN[N] { 3 }
const Y = 1 + 2 + 3 + foo<32>();
)",
      TypecheckSucceeds(HasNodeWithType("Y", "uN[32]")));
}

TEST(TypecheckV2Test, FunctionReturningMismatchingIntegerAutoTypeFails) {
  EXPECT_THAT(R"(
fn foo() -> u4 { 65536 }
const Y = foo();
)",
              TypecheckFails(HasSizeMismatch("uN[17]", "u4")));
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

TEST(TypecheckV2Test, FunctionReturningArrayWithIntegerReturnTypeFails) {
  EXPECT_THAT(R"(
const X = [s32:1, s32:2, s32:3];
fn foo() -> s32 { X }
const Y = foo();
)",
              TypecheckFails(HasTypeMismatch("sN[32][3]", "s32")));
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
              TypecheckFails(HasSizeMismatch("uN[15]", "u4")));
}

TEST(TypecheckV2Test, FunctionCallPassingInTooLargeExplicitIntegerSizeFails) {
  EXPECT_THAT(R"(
const X:u32 = 1;
fn foo(a: u4) -> u4 { a }
const Y = foo(X);
)",
              TypecheckFails(HasSizeMismatch("uN[32]", "u4")));
}

TEST(TypecheckV2Test, FunctionCallPassingInWrongSignednessFails) {
  EXPECT_THAT(R"(
const X:u32 = 1;
fn foo(a: s32) -> s32 { a }
const Y = foo(X);
)",
              TypecheckFails(HasSignednessMismatch("uN[32]", "s32")));
}

TEST(TypecheckV2Test, FunctionCallPassingInArrayForIntegerFails) {
  EXPECT_THAT(R"(
fn foo(a: u4) -> u4 { a }
const Y = foo([u32:1, u32:2]);
)",
              TypecheckFails(HasTypeMismatch("uN[32][2]", "u4")));
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

TEST(TypecheckV2Test,
     ParametricFunctionTakingIntegerOfImplicitParameterizedSize) {
  EXPECT_THAT(R"(
fn foo<N: u32>(a: uN[N]) -> uN[N] { a }
const X = foo(u10:5);
const Y = foo(u11:5);
)",
              TypecheckSucceeds(
                  AllOf(HasNodeWithType("const X = foo(u10:5);", "uN[10]"),
                        HasNodeWithType("const Y = foo(u11:5);", "uN[11]"))));
}

TEST(TypecheckV2Test, ParametricFunctionWithNonInferrableParametric) {
  EXPECT_THAT(R"(
fn foo<M: u32, N: u32>(a: uN[M]) -> uN[M] { a }
const X = foo(u10:5);
)",
              TypecheckFails(HasSubstr("Could not infer parametric(s): N")));
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
     ParametricFunctionTakingIntegerOfImplicitParameterizedSignedness) {
  EXPECT_THAT(R"(
fn foo<S: bool>(a: xN[S][32]) -> xN[S][32] { a }
const X = foo(u32:5);
const Y = foo(s32:5);
)",
              TypecheckSucceeds(
                  AllOf(HasNodeWithType("const X = foo(u32:5);", "uN[32]"),
                        HasNodeWithType("const Y = foo(s32:5);", "sN[32]"))));
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
     ParametricFunctionTakingIntegerOfImplicitParameterizedSignednessAndSize) {
  EXPECT_THAT(R"(
fn foo<S: bool, N: u32>(a: xN[S][N]) -> xN[S][N] { a }
const X = foo(u10:5);
const Y = foo(s11:5);
)",
              TypecheckSucceeds(
                  AllOf(HasNodeWithType("const X = foo(u10:5);", "uN[10]"),
                        HasNodeWithType("const Y = foo(s11:5);", "sN[11]"))));
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

TEST(TypecheckV2Test, ParametricFunctionWithDefaultImplicitlyOverriddenFails) {
  // In a case like this, the "overridden" value for `N` must be explicit (v1
  // agrees).
  EXPECT_THAT(R"(
fn foo<M: u32, N: u32 = {M + 1}>(a: uN[N]) -> uN[N] { a }
const X = foo<11>(u20:5);
)",
              TypecheckFails(HasSizeMismatch("u20", "uN[12]")));
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
     ParametricFunctionTakingIntegerOfImplicitSignednessAndSizeWithSum) {
  EXPECT_THAT(R"(
const X = u32:3;
const Y = u32:4;
fn foo<N: u32>(a: uN[N]) -> uN[N] { a }
const Z = foo(X + Y + X + 50);
)",
              TypecheckSucceeds(
                  HasNodeWithType("const Z = foo(X + Y + X + 50);", "uN[32]")));
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

TEST(TypecheckV2Test, ParametricFunctionTakingArrayOfImplicitSize) {
  EXPECT_THAT(
      R"(
fn foo<N: u32>(a: u32[N]) -> u32[N] { a }
const X = foo([1, 2, 3]);
const Y = foo([4, 5, 6, 7]);
)",
      TypecheckSucceeds(
          AllOf(HasNodeWithType("const X = foo([1, 2, 3]);", "uN[32][3]"),
                HasNodeWithType("const Y = foo([4, 5, 6, 7]);", "uN[32][4]"),
                HasNodeWithType("1", "uN[32]"), HasNodeWithType("2", "uN[32]"),
                HasNodeWithType("3", "uN[32]"), HasNodeWithType("4", "uN[32]"),
                HasNodeWithType("5", "uN[32]"), HasNodeWithType("6", "uN[32]"),
                HasNodeWithType("7", "uN[32]"))));
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

TEST(TypecheckV2Test, ParametricFunctionWithArrayMismatchingParameterizedSize) {
  EXPECT_THAT(R"(
fn foo<N: u32>(a: u32[N]) -> u32[N] { a }
const X = foo<3>([u32:1, u32:2, u32:3, u32:4]);
)",
              TypecheckFails(HasTypeMismatch("uN[32][4]", "u32[3]")));
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

TEST(TypecheckV2Test, ParametricFunctionImplicitParameterPropagation) {
  EXPECT_THAT(R"(
fn bar<A: u32, B: u32>(a: uN[A], b: uN[B]) -> uN[A] { a + 1 }
fn foo<A: u32, B: u32>(a: uN[A], b: uN[B]) -> uN[B] { bar(b, a) }
const X = foo(u23:4, u17:5);
)",
              TypecheckSucceeds(
                  HasNodeWithType("const X = foo(u23:4, u17:5);", "uN[17]")));
}

TEST(TypecheckV2Test, ParametricFunctionImplicitParameterExplicitPropagation) {
  EXPECT_THAT(R"(
fn bar<A: u32, B: u32>(a: uN[A], b: uN[B]) -> uN[A] { a + 1 }
fn foo<A: u32, B: u32>(a: uN[A], b: uN[B]) -> uN[B] { bar<B, A>(b, a) }
const X = foo(u23:4, u17:5);
)",
              TypecheckSucceeds(
                  HasNodeWithType("const X = foo(u23:4, u17:5);", "uN[17]")));
}

TEST(TypecheckV2Test, ParametricFunctionInvocationNesting) {
  EXPECT_THAT(R"(
fn foo<N: u32>(a: uN[N]) -> uN[N] { a + 1 }
const X = foo<24>(foo<24>(4) + foo<24>(5));
)",
              TypecheckSucceeds(HasNodeWithType(
                  "const X = foo<24>(foo<24>(4) + foo<24>(5));", "uN[24]")));
}

TEST(TypecheckV2Test, ParametricFunctionImplicitInvocationNesting) {
  EXPECT_THAT(R"(
fn foo<N: u32>(a: uN[N]) -> uN[N] { a + 1 }
const X = foo(foo(u24:4) + foo(u24:5));
)",
              TypecheckSucceeds(HasNodeWithType(
                  "const X = foo(foo(u24:4) + foo(u24:5));", "uN[24]")));
}

TEST(TypecheckV2Test,
     ParametricFunctionImplicitInvocationNestingWithExplicitOuter) {
  EXPECT_THAT(R"(
fn foo<N: u32>(a: uN[N]) -> uN[N] { a + 1 }
const X = foo<24>(foo(u24:4 + foo(u24:6)) + foo(u24:5));
)",
              TypecheckSucceeds(HasNodeWithType(
                  "const X = foo<24>(foo(u24:4 + foo(u24:6)) + foo(u24:5));",
                  "uN[24]")));
}

TEST(TypecheckV2Test,
     ParametricFunctionImplicitInvocationNestingWithExplicitInner) {
  EXPECT_THAT(R"(
fn foo<N: u32>(a: uN[N]) -> uN[N] { a + 1 }
const X = foo(foo<24>(4) + foo<24>(5));
)",
              TypecheckSucceeds(HasNodeWithType(
                  "const X = foo(foo<24>(4) + foo<24>(5));", "uN[24]")));
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

TEST(TypecheckV2Test,
     ParametricFunctionCallUsingGlobalConstantInImplicitParametricArgument) {
  EXPECT_THAT(R"(
fn foo<N: u32>(a: uN[N]) -> uN[N] { a }
const X = u3:1;
const Z = foo(X);
)",
              TypecheckSucceeds(HasNodeWithType("const Z = foo(X);", "uN[3]")));
}

TEST(TypecheckV2Test, ParametricFunctionCallFollowedByTypePropagation) {
  EXPECT_THAT(R"(
fn foo<N: u32>(a: uN[N]) -> uN[N] { a }
const Y = foo<15>(u15:1);
const Z = Y + 1;
)",
              TypecheckSucceeds(HasNodeWithType("const Z = Y + 1;", "uN[15]")));
}

TEST(TypecheckV2Test,
     ParametricFunctionCallWithImplicitParameterFollowedByTypePropagation) {
  EXPECT_THAT(R"(
fn foo<N: u32>(a: uN[N]) -> uN[N] { a }
const Y = foo(u15:1);
const Z = Y + 1;
)",
              TypecheckSucceeds(HasNodeWithType("const Z = Y + 1;", "uN[15]")));
}

TEST(TypecheckV2Test, GlobalConstantUsingParametricFunction) {
  EXPECT_THAT(R"(
fn foo<N: u32>(a: uN[N]) -> uN[N] { a }
const X = foo<32>(u32:3);
const Y = foo<X>(u3:1);
)",
              TypecheckSucceeds(
                  AllOf(HasNodeWithType("const X = foo<32>(u32:3);", "uN[32]"),
                        HasNodeWithType("const Y = foo<X>(u3:1);", "uN[3]"))));
}

TEST(TypecheckV2Test, GlobalConstantUsingAndUsedByParametricFunction) {
  EXPECT_THAT(R"(
fn foo<N: u32>(a: uN[N]) -> uN[N] { a }
const X = foo<32>(u32:3);
const Y = foo<X>(u3:1);
fn bar<N: u32>(a: uN[N]) -> uN[N] { a + Y + foo<3>(Y) }
const Z = bar<X>(u3:1 + Y);
)",
              TypecheckSucceeds(AllOf(
                  HasNodeWithType("const X = foo<32>(u32:3);", "uN[32]"),
                  HasNodeWithType("const Y = foo<X>(u3:1);", "uN[3]"),
                  HasNodeWithType("const Z = bar<X>(u3:1 + Y);", "uN[3]"))));
}

TEST(TypecheckV2Test, InvertUnaryOperator) {
  EXPECT_THAT(
      R"(
const X = false;
const Y = !X;
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("const X = false;", "uN[1]"),
                              HasNodeWithType("const Y = !X;", "uN[1]"))));
}

TEST(TypecheckV2Test, NegateUnaryOperator) {
  EXPECT_THAT(
      R"(
const X = u32:5;
const Y = -X;
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("const X = u32:5;", "uN[32]"),
                              HasNodeWithType("const Y = -X;", "uN[32]"))));
}

TEST(TypecheckV2Test, UnaryOperatorWithExplicitType) {
  EXPECT_THAT(
      R"(
const X = false;
const Y:bool = !X;
)",
      TypecheckSucceeds(
          AllOf(HasNodeWithType("const X = false;", "uN[1]"),
                HasNodeWithType("const Y: bool = !X;", "uN[1]"))));
}

TEST(TypecheckV2Test, UnaryOperatorOnInvalidType) {
  EXPECT_THAT(
      R"(
const X = (u32:1, u5:2);
const Y = -X;
)",
      TypecheckFails(HasSubstr(
          "Unary operations can only be applied to bits-typed operands.")));
}

TEST(TypecheckV2Test, UnaryOperatorWithWrongType) {
  EXPECT_THAT(
      R"(
const X = false;
const Y:u32 = !X;
)",
      TypecheckFails(HasSizeMismatch("bool", "u32")));
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

TEST(TypecheckV2Test, GlobalConstantEqualsLogicalAndOfLiterals) {
  EXPECT_THAT("const X = true && false;", TopNodeHasType("uN[1]"));
}

TEST(TypecheckV2Test, GlobalConstantLogicalAndWithWrongRhs) {
  EXPECT_THAT(R"(
const X = true;
const Y: u32 = 4;
const Z = X && Y;
)",
              TypecheckFails(HasSizeMismatch("uN[32]", "bool")));
}

TEST(TypecheckV2Test, GlobalConstantEqualsAndOfBooleanConstants) {
  EXPECT_THAT(R"(
const X = true;
const Y: bool = false;
const Z = X && Y;
)",
              TypecheckSucceeds(
                  AllOf(HasNodeWithType("const X = true;", "uN[1]"),
                        HasNodeWithType("const Y: bool = false;", "uN[1]"),
                        HasNodeWithType("const Z = X && Y;", "uN[1]"))));
}

TEST(TypecheckV2Test, GlobalConstantEqualsLogicalOrOfLiterals) {
  EXPECT_THAT("const X = bool:true || bool:false;", TopNodeHasType("uN[1]"));
}

TEST(TypecheckV2Test, GlobalConstantWithLogicalBinopOnWrongType) {
  EXPECT_THAT("const X = u32:5 || u32:6;",
              TypecheckFails(HasSubstr(
                  "Logical binary operations can only be applied to boolean")));
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

TEST(TypecheckV2Test, CastU32ToU32) {
  EXPECT_THAT(R"(const X = u32:1;
const Y = X as u32;)",
              TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[32]"),
                                      HasNodeWithType("Y", "uN[32]"))));
}

TEST(TypecheckV2Test, CastXPlus1AsU32) {
  EXPECT_THAT(R"(const X = u32:1;
const Y = (X + 1) as u32;)",
              TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[32]"),
                                      HasNodeWithType("Y", "uN[32]"))));
}

TEST(TypecheckV2Test, CastU32ToU16) {
  EXPECT_THAT(R"(const X = u32:1;
const Y = X as u16;)",
              TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[32]"),
                                      HasNodeWithType("Y", "uN[16]"))));
}

TEST(TypecheckV2Test, CastU16ToU32) {
  EXPECT_THAT(R"(const X = u16:1;
const Y = X as u32;)",
              TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[16]"),
                                      HasNodeWithType("Y", "uN[32]"))));
}

TEST(TypecheckV2Test, CastS16ToU32) {
  EXPECT_THAT(R"(const X = s16:1;
const Y = X as u32;)",
              TypecheckSucceeds(AllOf(HasNodeWithType("X", "sN[16]"),
                                      HasNodeWithType("Y", "uN[32]"))));
}

TEST(TypecheckV2Test, CastU16ToS32) {
  EXPECT_THAT(R"(const X = u16:1;
const Y = X as s32;)",
              TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[16]"),
                                      HasNodeWithType("Y", "sN[32]"))));
}

TEST(TypecheckV2Test, CastParametricBitCountToU32) {
  EXPECT_THAT(R"(fn f<N:u32>(x: uN[N]) -> u32 { x as u32 }
const X = f(u10:256);)",
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

TEST(TypecheckV2Test, ParametricCastWrapper) {
  EXPECT_THAT(R"(fn do_cast
  <WANT_S:bool, WANT_BITS:u32, GOT_S:bool, GOT_BITS:u32>
  (x: xN[GOT_S][GOT_BITS]) -> xN[WANT_S][WANT_BITS] {
    x as xN[WANT_S][WANT_BITS]
}
const X: s4 = do_cast<true, 4>(u32:64);)",
              TypecheckSucceeds(HasNodeWithType("X", "sN[4]")));
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

TEST(TypecheckV2Test, IfType) {
  EXPECT_THAT("const X = if true { u32:1 } else { u32:0 };",
              TypecheckSucceeds(HasNodeWithType("X", "uN[32]")));
}

TEST(TypecheckV2Test, IfTypeMismatch) {
  EXPECT_THAT("const X: u31 = if true { u32:1 } else { u32:0 };",
              TypecheckFails(HasSizeMismatch("u32", "u31")));
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
  EXPECT_THAT(R"(
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
              TypecheckSucceeds(AllOf(HasNodeWithType("res", "uN[1]"),
                                      HasNodeWithType("res2", "uN[6]"),
                                      HasNodeWithType("1..3", "uN[32]"))));
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
        (u3:42, y) => y,
        (_, y) => y + 1,
    }
}
)",
              TypecheckFails(HasSizeMismatch("uN[3]", "u8")));
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

TEST(TypecheckV2Test, ZeroMacroNumber) {
  EXPECT_THAT("const Y = zero!<u10>();",
              TypecheckSucceeds(HasNodeWithType("Y", "uN[10]")));
}

TEST(TypecheckV2Test, ZeroMacroArray) {
  EXPECT_THAT("const Y = zero!<u10[2]>();",
              TypecheckSucceeds(HasNodeWithType("Y", "uN[10][2]")));
}

TEST(TypecheckV2Test, DISABLED_ZeroMacroEnum) {
  // Type inference v2 cannot handle enums yet.
  EXPECT_THAT(R"(
enum E: u2 { ZERO=0, ONE=1, TWO=2}
const Y = zero!<E>();)",
              TypecheckSucceeds(HasNodeWithType("Y", "E")));
}

TEST(TypecheckV2Test, ZeroMacroTuple) {
  EXPECT_THAT("const Y = zero!<(u10, u32)>();",
              TypecheckSucceeds(HasNodeWithType("Y", "(uN[10], uN[32])")));
}

TEST(TypecheckV2Test, ZeroMacroEmptyStruct) {
  EXPECT_THAT(R"(
struct S { }
const Y = zero!<S>();
)",
              TypecheckSucceeds(HasNodeWithType("Y", "S {}")));
}

TEST(TypecheckV2Test, ZeroMacroStruct) {
  EXPECT_THAT(
      R"(
struct S { a: u32, b: u32, }
const Y = zero!<S>();
)",
      TypecheckSucceeds(HasNodeWithType("Y", "S { a: uN[32], b: uN[32] }")));
}

TEST(TypecheckV2Test, ZeroMacroParametricStruct) {
  EXPECT_THAT(
      R"(
struct S<A: u32, B: u32> { a: uN[A], b: uN[B], }
const Y = zero!<S<16, 64>>();
)",
      TypecheckSucceeds(HasNodeWithType("Y", "S { a: uN[16], b: uN[64] }")));
}

TEST(TypecheckV2Test, ZeroMacroParametricStructInFn) {
  EXPECT_THAT(
      R"(
struct S<A: u32, B: u32> { a: uN[A], b: uN[B], }
fn f<N:u32, M: u32={N*4}>()-> S<N, M> { zero!<S<N, M>>() }
const Y = f<16>();
)",
      TypecheckSucceeds(HasNodeWithType("Y", "S { a: uN[16], b: uN[64] }")));
}

TEST(TypecheckV2Test, ZeroMacroFromParametric) {
  EXPECT_THAT(R"(
fn f<N:u32>() -> uN[N] { zero!<uN[N]>() }
const Y = f<10>();
)",
              TypecheckSucceeds(HasNodeWithType("Y", "uN[10]")));
}

TEST(TypecheckV2Test, ZeroMacroExprError) {
  EXPECT_THAT(R"(
const X = u32:10;
const Y = zero!<X>();
)",
              TypecheckFails(HasSubstr("in `zero!<X>()`")));
}

TEST(TypecheckV2Test, ZeroMacroImplConstError) {
  EXPECT_THAT(R"(
struct S{}
impl S { const X = u32:10; }
const Y = zero!<S::X>();
)",
              TypecheckFails(HasSubstr("in `zero!<S::X>()`")));
}

// We don't support imports in the type system yet.
TEST(TypecheckV2Test, DISABLED_ZeroMacroImportedType) {
  EXPECT_THAT(R"(
import imported;
const Y = zero!<imported::X>();
)",
              TypecheckSucceeds(HasNodeWithType("Y", "uN[10]")));
}

TEST(TypecheckV2Test, GlobalConstantEqualsLShiftOfLiterals) {
  EXPECT_THAT("const X = u5:5 << 4;", TopNodeHasType("uN[5]"));
}

TEST(TypecheckV2Test, GlobalConstantEqualsLShiftOfLiteralsSameType) {
  EXPECT_THAT("const X = u32:1 << 4;", TopNodeHasType("uN[32]"));
}

TEST(TypecheckV2Test,
     GlobalConstantEqualsLShiftOfLiteralsRhsDifferentTypeAllSpecified) {
  EXPECT_THAT("const X: u5 = u5:1 << 4;", TopNodeHasType("uN[5]"));
}

TEST(TypecheckV2Test, GlobalConstantEqualsLShiftOfLiteralsSizeTooSmall) {
  EXPECT_THAT("const X = u2:3 << 4;",
              TypecheckFails(HasSubstr(
                  "Shift amount is larger than shift value bit width of 2.")));
}

TEST(TypecheckV2Test, GlobalConstantEqualsRShiftOfLiteralsSizeTooSmall) {
  EXPECT_THAT("const X = u1:1 >> 4;",
              TypecheckFails(HasSubstr(
                  "Shift amount is larger than shift value bit width of 1.")));
}

TEST(TypecheckV2Test, GlobalConstantEqualsLShiftOfLiteralsMismatchedType) {
  EXPECT_THAT("const X: u16 = u32:1 << 4;",
              TypecheckFails(HasSizeMismatch("u32", "u16")));
}

TEST(TypecheckV2Test, GlobalConstantEqualsLShiftOfNonBitsType) {
  EXPECT_THAT(
      "const X = (u32:1, u5:1) << 4;",
      TypecheckFails(HasSubstr("can only be applied to bits-typed operands")));
}

TEST(TypecheckV2Test, GlobalConstantEqualsRShiftOfNonBitsAmount) {
  EXPECT_THAT(
      "const X = u32:1 >> (u32:4, u4:1);",
      TypecheckFails(HasSubstr("can only be applied to bits-typed operands")));
}

TEST(TypecheckV2Test, GlobalConstantLShiftOfConstants) {
  EXPECT_THAT(R"(
const X = u4:1;
const Y = u3:2;
const Z = X << Y;
)",
              TypecheckSucceeds(HasNodeWithType("const Z = X << Y;", "uN[4]")));
}

TEST(TypecheckV2Test, GlobalConstantLShiftWithNegativeRhs) {
  EXPECT_THAT(R"(
const X = u4:1;
const Y = s3:-3;
const Z = X << Y;
)",
              TypecheckFails(HasSubstr("Shift amount must be unsigned")));
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

TEST(TypecheckV2Test, AllOnesMacroNumber) {
  EXPECT_THAT("const Y = all_ones!<u10>();",
              TypecheckSucceeds(HasNodeWithType("Y", "uN[10]")));
}

TEST(TypecheckV2Test, AllOnesMacroArray) {
  EXPECT_THAT("const Y = all_ones!<u10[2]>();",
              TypecheckSucceeds(HasNodeWithType("Y", "uN[10][2]")));
}

TEST(TypecheckV2Test, DISABLED_AllOnesMacroEnum) {
  // Type inference v2 cannot handle enums yet.
  EXPECT_THAT(R"(
enum E: u2 { ZERO=0, ONE=1, TWO=2}
const Y = all_ones!<E>();)",
              TypecheckSucceeds(HasNodeWithType("Y", "E")));
}

TEST(TypecheckV2Test, AllOnesMacroTuple) {
  EXPECT_THAT("const Y = all_ones!<(u10, u32)>();",
              TypecheckSucceeds(HasNodeWithType("Y", "(uN[10], uN[32])")));
}

TEST(TypecheckV2Test, AllOnesMacroEmptyStruct) {
  EXPECT_THAT(R"(
struct S { }
const Y = all_ones!<S>();
)",
              TypecheckSucceeds(HasNodeWithType("Y", "S {}")));
}

TEST(TypecheckV2Test, AllOnesMacroStruct) {
  EXPECT_THAT(
      R"(
struct S { a: u32, b: u32, }
const Y = all_ones!<S>();
)",
      TypecheckSucceeds(HasNodeWithType("Y", "S { a: uN[32], b: uN[32] }")));
}

TEST(TypecheckV2Test, AllOnesMacroParametricStruct) {
  EXPECT_THAT(
      R"(
struct S<A: u32, B: u32> { a: uN[A], b: uN[B], }
const Y = all_ones!<S<16, 64>>();
)",
      TypecheckSucceeds(HasNodeWithType("Y", "S { a: uN[16], b: uN[64] }")));
}

TEST(TypecheckV2Test, AllOnesMacroFromParametric) {
  EXPECT_THAT(R"(
fn f<N:u32>() -> uN[N] { all_ones!<uN[N]>() }
const Y = f<10>();
)",
              TypecheckSucceeds(HasNodeWithType("Y", "uN[10]")));
}

TEST(TypecheckV2Test, AllOnesMacroExprError) {
  EXPECT_THAT(R"(
const X = u32:10;
const Y = all_ones!<X>();
)",
              TypecheckFails(HasSubstr("in `all_ones!<X>()`")));
}

TEST(TypecheckV2Test, AllOnesMacroImplConstError) {
  EXPECT_THAT(R"(
struct S{}
impl S { const X = u32:10; }
const Y = all_ones!<S::X>();
)",
              TypecheckFails(HasSubstr("in `all_ones!<S::X>()`")));
}

// We don't support imports in the type system yet.
TEST(TypecheckV2Test, DISABLED_AllOnesMacroImportedType) {
  EXPECT_THAT(R"(
import imported;
const Y = all_ones!<imported::X>();
)",
              TypecheckSucceeds(HasNodeWithType("Y", "uN[10]")));
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

TEST(TypecheckV2Test, UnassignedReturnValueIgnoredParametric) {
  EXPECT_THAT(
      R"(
fn ignored<N:u32>() -> uN[N] { zero!<uN[N]>() }

fn main() -> u32 {
  ignored<u32:31>();
  u32:1
}
)",
      TypecheckSucceeds(HasNodeWithType("ignored<u32:31>()", "uN[31]")));
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

// The impl tests below are generally derived from impl_typecheck_test.cc, with
// some additions.

TEST(TypecheckV2Test, StaticConstantOnImpl) {
  EXPECT_THAT(R"(
struct Point {}
impl Point {
  const NUM_DIMS = u32:2;
}
const X = Point::NUM_DIMS;
)",
              TypecheckSucceeds(HasNodeWithType("X", "uN[32]")));
}

TEST(TypecheckV2Test, MissingFunctionOnImplFails) {
  EXPECT_THAT(
      R"(
struct Point {}
impl Point {
  const NUM_DIMS = u32:2;
}
const X = Point::num_dims();
)",
      TypecheckFails(HasSubstr(
          "Name 'num_dims' is not defined by the impl for struct 'Point'")));
}

TEST(TypecheckV2Test, ImplWithMissingConstantFails) {
  constexpr std::string_view kProgram = R"(
struct Point { x: u32, y: u32 }

impl Point {
    const NUM_DIMS = u32:2;
}

fn point_dims() -> u32 {
    Point::DIMENSIONS
}
)";
  EXPECT_THAT(Typecheck(kProgram),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Name 'DIMENSIONS' is not defined by the impl "
                                 "for struct 'Point'")));
}

TEST(TypecheckV2Test, MissingImplOnStructFails) {
  EXPECT_THAT(R"(
struct Point {}
const X = Point::num_dims();
)",
              TypecheckFails(
                  HasSubstr("Struct 'Point' has no impl defining 'num_dims'")));
}

TEST(TypecheckV2Test, ImplWithConstCalledAsFuncFails) {
  EXPECT_THAT(R"(
struct Point {}
impl Point {
    const num_dims = u32:4;
}
const X = Point::num_dims();
)",
              TypecheckFails(HasSubstr(
                  "Invocation callee `Point::num_dims` is not a function")));
}

TEST(TypecheckV2Test, StaticImplFunction) {
  EXPECT_THAT(R"(
struct Point {}
impl Point {
    fn num_dims() -> u32 { 2 }
}
const X = Point::num_dims();
)",
              TypecheckSucceeds(HasNodeWithType("X", "uN[32]")));
}

TEST(TypecheckV2Test, StaticImplFunctionWithWrongArgumentTypeFails) {
  EXPECT_THAT(R"(
struct Point {}
impl Point {
    fn foo(a: u32) -> u32 { a }
}
const X = Point::foo(u24:5);
)",
              TypecheckFails(HasSizeMismatch("u24", "u32")));
}

TEST(TypecheckV2Test, StaticImplFunctionCallWithMissingArgumentFails) {
  EXPECT_THAT(R"(
struct Point {}
impl Point {
    fn foo(a: u32) -> u32 { a }
}
const X = Point::foo();
)",
              TypecheckFails(HasSubstr("Expected 1 argument(s) but got 0")));
}

TEST(TypecheckV2Test, StaticImplFunctionCallWithExtraArgumentFails) {
  EXPECT_THAT(R"(
struct Point {}
impl Point {
    fn foo(a: u32) -> u32 { a }
}
const X = Point::foo(u32:1, u32:2);
)",
              TypecheckFails(HasSubstr("Expected 1 argument(s) but got 2")));
}

TEST(TypecheckV2Test, StaticImplFunctioUsingConst) {
  EXPECT_THAT(R"(
struct Point {}
impl Point {
    const DIMS = u32:2;

    fn num_dims() -> u32 {
        DIMS
    }
}

const X = uN[Point::num_dims()]:0;
)",
              TypecheckSucceeds(HasNodeWithType("X", "uN[2]")));
}

TEST(TypecheckV2Test, StaticConstUsingImplFunction) {
  EXPECT_THAT(R"(
struct Point {}
impl Point {
    fn num_dims() -> u32 { 2 }
    const DIMS = num_dims();
}

const X = uN[Point::DIMS]:0;
)",
              TypecheckSucceeds(HasNodeWithType("X", "uN[2]")));
}

TEST(TypecheckV2Test, ImplConstantUsedForParametricFunctionInference) {
  EXPECT_THAT(R"(
struct Foo {}
impl Foo {
  const X = u32:2;
}
fn f<N: u32>(a: uN[N]) -> uN[N] { a }
const Y = f(Foo::X);
)",
              TypecheckSucceeds(HasNodeWithType("Y", "uN[32]")));
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

TEST(TypecheckV2Test, ParametricLet) {
  EXPECT_THAT(R"(
fn f<N: u32>() -> uN[N] {
  let x = uN[N]:0;
  x
}

const X = f<4>();
const Y = f<16>();
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[4]"),
                                      HasNodeWithType("Y", "uN[16]"))));
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

TEST(TypecheckV2Test, LetValAsType) {
  EXPECT_THAT(R"(
fn main() -> u7 {
  let (x, .., (y,)) = (u32:7, u8:3, (s8:3,));
  uN[x]:0
}
)",
              TypecheckSucceeds(HasNodeWithType("x", "uN[32]")));
}

TEST(TypecheckV2Test, LetConstAsType) {
  EXPECT_THAT(R"(
fn main() -> u7 {
  let (x, .., (y,)) = (u32:7, u8:3, (s8:3,));
  const N = x;
  uN[N]:0
}
)",
              TypecheckSucceeds(HasNodeWithType("N", "uN[32]")));
}

TEST(TypecheckV2Test, LetConstWarnsOnBadName) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
fn main() {
  const bad_name_const = u32:5;
}
)"));
  ASSERT_THAT(result.tm.warnings.warnings().size(), 1);
  EXPECT_EQ(result.tm.warnings.warnings()[0].message,
            "Standard style is SCREAMING_SNAKE_CASE for constant identifiers; "
            "got: `bad_name_const`");
}

TEST(TypecheckV2Test, ImplFunctionUsingStructMembers) {
  EXPECT_THAT(R"(
struct Point { x: u32, y: u32 }

impl Point {
    fn area(self) -> u32 {
        self.x * self.y
    }
}

const P = Point{x: u32:4, y:u32:2};
const Y = P.area();
const Z = uN[Y]:0;
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("Y", "uN[32]"),
                                      HasNodeWithType("Z", "uN[8]"))));
}

TEST(TypecheckV2Test, ImplFunctionReturnsSelf) {
  EXPECT_THAT(R"(
struct Point { x: u32, y: u32 }

impl Point {
    fn unit() -> Self { Point { x: u32:1, y: u32:1 } }
}

const P = Point::unit();
const X = uN[P.x]:0;
)",
              TypecheckSucceeds(
                  AllOf(HasNodeWithType("P", "Point { x: uN[32], y: uN[32] }"),
                        HasNodeWithType("X", "uN[1]"))));
}

TEST(TypecheckV2Test, ImplsForDifferentStructs) {
  EXPECT_THAT(R"(
struct Point { x: u32, y: u32 }

struct Line { a: Point, b: Point }

impl Point {
    fn area(self) -> u32 {
        self.x * self.y
    }
}

impl Line {
    fn height(self) -> u32 {
        self.b.y - self.a.y
    }
}

const P = Point{x: u32:4, y:u32:2};
const A = P.area(); // 8
const L = Line{a: P, b: Point{x: u32:4, y: u32:4}};
const H = L.height(); // 2
const Z = uN[A + H]:0;
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("A", "uN[32]"),
                                      HasNodeWithType("H", "uN[32]"),
                                      HasNodeWithType("Z", "uN[10]"))));
}

TEST(TypecheckV2Test, ImplFunctionUsingStructMembersIndirect) {
  EXPECT_THAT(R"(
struct Point { x: u32, y: u32 }

impl Point {
    fn area(self) -> u32 {
        self.x * self.y
    }
}

const P = Point{x: u32:4, y:u32:2};
const Y = P;
const W = Y;
const X = uN[W.area()]:0;
)",
              TypecheckSucceeds(HasNodeWithType("X", "uN[8]")));
}

TEST(TypecheckV2Test, InstanceMethodCalledStaticallyWithNoParamsFails) {
  EXPECT_THAT(R"(
struct Point { x: u32, y: u32 }

impl Point {
    fn area(self) -> u32 {
        self.x * self.y
    }
}

const P = Point::area();
)",
              TypecheckFails(HasSubstr("Expected 1 argument(s) but got 0")));
}

TEST(TypecheckV2Test, ImplFunctionCalledOnSelf) {
  EXPECT_THAT(R"(
struct Rect { width: u32, height: u32 }

impl Rect {
    const BORDER = u32:2;
    fn compute_width(self) -> u32 { self.width + BORDER * 2 }
    fn compute_height(self) -> u32 { self.height + BORDER * 2 }

    fn area(self) -> u32 {
        self.compute_width() * self.compute_height()
    }
}

const R = Rect { width: 2, height: 4 };
const Z = uN[R.area()]:0;
)",
              TypecheckSucceeds(HasNodeWithType("Z", "uN[48]")));
}

TEST(TypecheckV2Test, InstanceMethodNotDefined) {
  EXPECT_THAT(
      R"(
struct Point { x: u32, y: u32 }

impl Point { }

const P = Point { x: u32:1, y: u32:4 };
const Z = uN[P.area()]:0;
)",
      TypecheckFails(HasSubstr(
          "Name 'area' is not defined by the impl for struct 'Point'")));
}

TEST(TypecheckV2Test, ImplMethodCalledOnIntFails) {
  EXPECT_THAT(R"(
const X = u32:1;
const Y = uN[X.area()]:0;
)",
              TypecheckFails(HasSubstr(
                  "Cannot invoke method `area` on non-struct type `uN[32]`")));
}

TEST(TypecheckV2Test, ImplFunctionUsingStructMembersAndArg) {
  EXPECT_THAT(R"(
struct Point { x: u32, y: u32 }

impl Point {
    fn area(self, a: u32, b: u32) -> u32 {
        self.x * self.y * a * b
    }
}

const P = Point{x: 4, y:2};
const Y = P.area(2, 1);
const Z = uN[Y]:0;
)",
              TypecheckSucceeds(HasNodeWithType("Z", "uN[16]")));
}

TEST(TypecheckV2Test, ImplFunctionUsingStructMembersExplicitSelfType) {
  EXPECT_THAT(R"(
struct Point { x: u32, y: u32 }

impl Point {
    fn area(self: Self) -> u32 {
        self.x * self.y
    }
}

const P = Point{x: u32:4, y:u32:2};
const A = P.area();
const Z = uN[A]:0;
)",
              TypecheckSucceeds(HasNodeWithType("Z", "uN[8]")));
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

TEST(TypecheckV2Test, ConstAssertTrue) {
  EXPECT_THAT(
      R"(
const_assert!(true);
)",
      TypecheckSucceeds(HasNodeWithType("true", "uN[1]")));
}

TEST(TypecheckV2Test, ConstAssertFalse) {
  EXPECT_THAT(
      R"(
const_assert!(false);
)",
      TypecheckSucceeds(HasNodeWithType("false", "uN[1]")));
}

TEST(TypecheckV2Test, ConstAssertConstExpr) {
  EXPECT_THAT(
      R"(
const_assert!(1 > 0);
)",
      TypecheckSucceeds(HasNodeWithType("1 > 0", "uN[1]")));
}

TEST(TypecheckV2Test, ConstAssertMismatch) {
  EXPECT_THAT(
      R"(
const_assert!(4);
)",
      TypecheckFails(HasSizeMismatch("uN[3]", "bool")));
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

TEST(TypecheckV2Test, ParametricImplConstant) {
  EXPECT_THAT(R"(
struct S<N: u32> {}

impl S<N> {
  const N_VALUE = N;
}

const X = uN[S<1>::N_VALUE]:0;
const Y = uN[S<2>::N_VALUE]:0;
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[1]"),
                                      HasNodeWithType("Y", "uN[2]"))));
}

TEST(TypecheckV2Test, SumOfImplConstantsFromDifferentParameterizations) {
  EXPECT_THAT(R"(
struct S<N: u32> {}

impl S<N> {
  const N_VALUE = N;
}

const X = uN[S<2>::N_VALUE + S<1>::N_VALUE]:0;
)",
              TypecheckSucceeds(HasNodeWithType("X", "uN[3]")));
}

TEST(TypecheckV2Test,
     ImplConstantWithDifferentExpressionsOfOneParameterization) {
  EXPECT_THAT(R"(
struct S<N: u32> {}

impl S<N> {
  const N_VALUE = N;
}

const A = u32:2;
const X = uN[S<2>::N_VALUE + S<A>::N_VALUE]:0;
)",
              TypecheckSucceeds(HasNodeWithType("X", "uN[4]")));
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

TEST(TypecheckV2Test, ParametricConstantUsingConstantFromSameImpl) {
  EXPECT_THAT(R"(
struct S<N: u32> {}

fn f<N: u32>() -> u32 { N + 1 }

impl S<N> {
  const N_PLUS_1_VALUE = f<N>();
  const N_PLUS_2_VALUE = f<N_PLUS_1_VALUE>();
}

const X = uN[S<2>::N_PLUS_2_VALUE]:0;
const Y = uN[S<10>::N_PLUS_2_VALUE]:0;
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[4]"),
                                      HasNodeWithType("Y", "uN[12]"))));
}

TEST(TypecheckV2Test, StaticParametricImplFnUsingConstant) {
  EXPECT_THAT(R"(
struct S<N: u32> {}

impl S<N> {
  const N_PLUS_1_VALUE = N + 1;
  fn foo(a: u32) -> u32 { N_PLUS_1_VALUE + N + a }
}

// Note that we would likely need constexpr evaluator to be aware of impl
// TypeInfos in order to use these return values in a mandatory constexpr
// context.
const X = S<3>::foo(10);
const Y = S<4>::foo(10);
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[32]"),
                                      HasNodeWithType("Y", "uN[32]"))));
}

TEST(TypecheckV2Test, StaticImplFnUsingParametricForInterface) {
  EXPECT_THAT(R"(
struct S<N: u32> {}

impl S<N> {
  fn foo(a: uN[N]) -> uN[N] { a }
}

const X = S<16>::foo(10);
const Y = S<32>::foo(11);
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[16]"),
                                      HasNodeWithType("Y", "uN[32]"))));
}

TEST(TypecheckV2Test, ParametricConstantUsingConstantFromOtherImpl) {
  EXPECT_THAT(R"(
// Note that the entities in here are sensitive to positioning due to
// https://github.com/google/xls/issues/1911

struct S<N: u32> {}

fn f<N: u32>() -> u32 { N + 1 }

impl S<N> {
  const N_PLUS_1_VALUE = f<N>();
}

struct T<N: u32> {}

impl T<N> {
  const N_PLUS_2_VALUE = f<{S<N>::N_PLUS_1_VALUE}>();
}

const X = uN[T<2>::N_PLUS_2_VALUE]:0;
const Y = uN[T<10>::N_PLUS_2_VALUE]:0;
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[4]"),
                                      HasNodeWithType("Y", "uN[12]"))));
}

TEST(TypecheckV2Test, ParametricBasedImplConstantType) {
  EXPECT_THAT(R"(
struct S<N: u32> {}

impl S<N> {
  const C = uN[N]:0;
}

const C2 = S<2>::C;
const C3 = S<3>::C;
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("C2", "uN[2]"),
                                      HasNodeWithType("C3", "uN[3]"))));
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

TEST(TypecheckV2Test, ImplConstantUsingParametricDefault) {
  EXPECT_THAT(R"(
struct S<A: u32, B: u32 = {A * 2}> {}

impl S<A, B> {
  const C = B;
}

const X = uN[S<2>::C]:0;
const Y = uN[S<3>::C]:0;
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[4]"),
                                      HasNodeWithType("Y", "uN[6]"))));
}

TEST(TypecheckV2Test, ParametricImplConstantUsedWithMissingParametrics) {
  EXPECT_THAT(
      R"(
struct S<A: u32, B: u32 = {A * 2}> {}

impl S<A, B> {
  const C = B;
}

const X = S::C;
)",
      TypecheckFails(HasSubstr("Use of `S` with missing parametric(s): A")));
}

TEST(TypecheckV2Test, ParametricImplConstantUsedWithTooManyParametrics) {
  EXPECT_THAT(
      R"(
struct S<A: u32> {}

impl S<A> {
  const C = A;
}

const X = S<1, 2>::C;
)",
      TypecheckFails(
          HasSubstr("Too many parametric values supplied; limit: 1 given: 2")));
}

TEST(TypecheckV2Test, InstanceMethodReturningStaticParametricType) {
  EXPECT_THAT(
      R"(
struct S<A: u32> {}

impl S<A> {
  fn foo(self) -> uN[A] { uN[A]:0 }
}

const X = S<16>{}.foo();
const Y = S<32>{}.foo();
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[16]"),
                              HasNodeWithType("Y", "uN[32]"))));
}

TEST(TypecheckV2Test, InstanceMethodReturningParametricConstType) {
  EXPECT_THAT(
      R"(
struct S<A: u32> {}

impl S<A> {
  const C = A;
  fn foo(self) -> uN[C] { uN[C]:0 }
}

const X = S<16>{}.foo();
const Y = S<32>{}.foo();
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[16]"),
                              HasNodeWithType("Y", "uN[32]"))));
}

TEST(TypecheckV2Test, InstanceMethodTakingStaticParametricType) {
  EXPECT_THAT(
      R"(
struct S<A: u32> {}

impl S<A> {
  fn foo(self, a: uN[A]) -> uN[A] { a + 1 }
}

const X = S<16>{}.foo(100);
const Y = S<32>{}.foo(200);
)",
      TypecheckSucceeds(AllOf(
          HasNodeWithType("X", "uN[16]"), HasNodeWithType("Y", "uN[32]"),
          HasNodeWithType("100", "uN[16]"), HasNodeWithType("200", "uN[32]"))));
}

TEST(TypecheckV2Test, InstanceMethodTakingStaticConstType) {
  EXPECT_THAT(
      R"(
struct S<A: u32> {}

impl S<A> {
  const C = A + 1;
  fn foo(self, a: uN[C]) -> uN[C] { a + 1 }
}

const X = S<16>{}.foo(100);
const Y = S<32>{}.foo(200);
)",
      TypecheckSucceeds(AllOf(
          HasNodeWithType("X", "uN[17]"), HasNodeWithType("Y", "uN[33]"),
          HasNodeWithType("100", "uN[17]"), HasNodeWithType("200", "uN[33]"))));
}

TEST(TypecheckV2Test, ParametricInstanceMethod) {
  EXPECT_THAT(
      R"(
struct S {}

impl S {
  fn foo<N: u32>(self, a: uN[N]) -> uN[N] { a + 1 }
}

const X = S{}.foo(u16:100);
const Y = S{}.foo(u32:200);
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[16]"),
                              HasNodeWithType("Y", "uN[32]"))));
}

TEST(TypecheckV2Test, ParametricInstanceMethodWithParametricSignedness) {
  EXPECT_THAT(
      R"(
struct S {}

impl S {
  fn foo<S: bool, N: u32>(self, a: xN[S][N]) -> xN[S][N] { a + 1 }
}

const X = S{}.foo(u16:100);
const Y = S{}.foo(s32:200);
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[16]"),
                              HasNodeWithType("Y", "sN[32]"))));
}

TEST(TypecheckV2Test, ParametricInstanceMethodOfParametricStruct) {
  EXPECT_THAT(
      R"(
struct S<M: u32> {}

impl S<M> {
  fn add<N: u32>(self) -> u32 { M + N }
}

const X = S<4>{}.add<3>();
const Y = S<5>{}.add<6>();
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[32]"),
                              HasNodeWithType("Y", "uN[32]"))));
}

TEST(TypecheckV2Test, ParametricInstanceMethodUsingParametricConstant) {
  EXPECT_THAT(
      R"(
struct S<M: u32> {}

impl S<M> {
  const M_VALUE = M;
  fn add<N: u32>(self) -> u32 { M_VALUE + N }
}

const X = S<4>{}.add<3>();
const Y = S<5>{}.add<6>();
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[32]"),
                              HasNodeWithType("Y", "uN[32]"))));
}

TEST(TypecheckV2Test, ParametricInstanceMethodUsingStaticParametricAsDefault) {
  EXPECT_THAT(
      R"(
struct S<M: u32> {}

impl S<M> {
  fn add<N: u32, P: u32 = {M + N}>(self) -> uN[P] { uN[P]:0 }
}

const X = S<4>{}.add<3>();
const Y = S<5>{}.add<6>();
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[7]"),
                              HasNodeWithType("Y", "uN[11]"))));
}

TEST(TypecheckV2Test, ParametricsFromStructAndMethodInType) {
  EXPECT_THAT(
      R"(
struct S<M: u32> {
  a: uN[M]
}

impl S<M> {
  fn replicate<N: u32>(self) -> uN[M][N] { [self.a, ...] }
}

const X = S{a: u16:5}.replicate<3>();
const Y = S{a: u32:6}.replicate<4>();
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[16][3]"),
                              HasNodeWithType("Y", "uN[32][4]"))));
}

TEST(TypecheckV2Test, ParametricsFromStructAndMethodBothInferred) {
  EXPECT_THAT(
      R"(
struct Data<M: u32> {
  a: uN[M]
}

impl Data<M> {
  fn combine<S: bool, N: u32>(self, b: xN[S][N]) -> (uN[M], xN[S][N]) {
    (self.a, b)
  }
}

const X = Data{a: 5}.combine(-42);
const Y = Data{a: 120}.combine(256);
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("X", "(uN[3], sN[7])"),
                              HasNodeWithType("Y", "(uN[7], uN[9])"))));
}

TEST(TypecheckV2Test, StackedParametricInstanceMethodCalls) {
  EXPECT_THAT(
      R"(
struct Data<M: u32> {
  a: uN[M]
}

impl Data<M> {
  fn bar<N: u32>(self, b: uN[N]) -> uN[N] { b }

  fn foo<N: u32>(self, b: uN[N]) -> (uN[M], uN[N], u16) {
    (self.a, b, self.bar(b as u16))
  }
}

const X = Data{a: 5}.foo(42);
const Y = Data{a: 120}.foo(256);
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("X", "(uN[3], uN[6], uN[16])"),
                              HasNodeWithType("Y", "(uN[7], uN[9], uN[16])"))));
}

TEST(TypecheckV2BuiltinTest, AndReduce) {
  EXPECT_THAT("const Y = and_reduce(u6:3);",
              TypecheckSucceeds(HasNodeWithType("Y", "uN[1]")));
}

TEST(TypecheckV2BuiltinTest, AssertLt) {
  EXPECT_THAT(
      R"(
fn foo(x: u10) -> u10 {
  assert_lt(x, 25);
  x
}

const X = foo(10);
)",
      TypecheckSucceeds(HasNodeWithType("25", "uN[10]")));
}

TEST(TypecheckV2BuiltinTest, AssertWithArray) {
  EXPECT_THAT(
      R"(
fn foo(x:u32) -> u32 {
  assert!(x>32, [1,2,3]);
  x
}

const X = foo(10);
)",
      TypecheckSucceeds(HasNodeWithType("[1, 2, 3]", "uN[8][3]")));
}

TEST(TypecheckV2BuiltinTest, DISABLED_Assert) {
  // TiV2 can't deal with strings yet.
  EXPECT_THAT(
      R"(
fn foo(x:u32) -> u32 {
  assert!(x>32, "Failed");
  x
}

const X = foo(10);
)",
      TypecheckSucceeds(HasNodeWithType("X", "uN[32]")));
}

TEST(TypecheckV2BuiltinTest, BitSliceUpdate) {
  EXPECT_THAT(R"(const Y = bit_slice_update(u32:10, u33:11, u34:12);)",
              TypecheckSucceeds(HasNodeWithType("Y", "uN[32]")));
}

TEST(TypecheckV2BuiltinTest, BitSliceUpdateError) {
  EXPECT_THAT(R"(const Y: u64 = bit_slice_update(u32:10, u33:11, u34:12);)",
              TypecheckFails(HasSizeMismatch("uN[32]", "u64")));
}

TEST(TypecheckV2BuiltinTest, Clz) {
  EXPECT_THAT(R"(const Y = clz(u8:3);)",
              TypecheckSucceeds(HasNodeWithType("Y", "uN[8]")));
}

TEST(TypecheckV2BuiltinTest, Ctz) {
  EXPECT_THAT(R"(const Y = ctz(u8:3);)",
              TypecheckSucceeds(HasNodeWithType("Y", "uN[8]")));
}

TEST(TypecheckV2BuiltinTest, DISABLED_OneHot) {
  // This doesn't work yet. It gives an error in GenerateTypeInfo,
  // probably because it needs the parametric environment in context of the main
  // module at the invocation site at the same time it needs the function
  // signature from the builtins module.
  EXPECT_THAT(R"(const Y = one_hot(u32:2, true);)",
              TypecheckSucceeds(HasNodeWithType("Y", "uN[33]")));
}

TEST(TypecheckV2BuiltinTest, MyOneHot) {
  EXPECT_THAT(R"(
fn my_one_hot<N: u32, M:u32={N+1}>(x: uN[N], y: u1) -> uN[M] {
  zero!<uN[M]>()
}

const Y = my_one_hot(u32:2, true);
)",
              TypecheckSucceeds(HasNodeWithType("Y", "uN[33]")));
}

TEST(TypecheckV2BuiltinTest, OneHotSel) {
  EXPECT_THAT(R"(const Y = one_hot_sel(2, [s10:1, s10:2]);)",
              TypecheckSucceeds(HasNodeWithType("Y", "sN[10]")));
}

TEST(TypecheckV2BuiltinTest, OrReduce) {
  EXPECT_THAT(R"(const Y = or_reduce(u6:3);)",
              TypecheckSucceeds(HasNodeWithType("Y", "uN[1]")));
}

TEST(TypecheckV2BuiltinTest, OrReduceError) {
  EXPECT_THAT(R"(const Y: u32 = or_reduce(u6:3);)",
              TypecheckFails(HasSizeMismatch("u1", "u32")));
}

TEST(TypecheckV2BuiltinTest, PrioritySel) {
  EXPECT_THAT(R"(const Y = priority_sel(2, [s10:1, s10:2], s10:3);)",
              TypecheckSucceeds(HasNodeWithType("Y", "sN[10]")));
}

TEST(TypecheckV2BuiltinTest, RevWithParametric) {
  EXPECT_THAT(R"(const Y = rev<u32:8>(u8:3);)",
              TypecheckSucceeds(HasNodeWithType("Y", "uN[8]")));
}

TEST(TypecheckV2BuiltinTest, RevWithParametricMismatch) {
  EXPECT_THAT(R"(const Y = rev<u32:8>(u6:3);)",
              TypecheckFails(HasSizeMismatch("uN[8]", "u6")));
}

TEST(TypecheckV2BuiltinTest, RevMismatch) {
  EXPECT_THAT(R"(const Y:u32 = rev(u6:3);)",
              TypecheckFails(HasSizeMismatch("u32", "uN[6]")));
}

TEST(TypecheckV2BuiltinTest, RevWithoutParametric) {
  EXPECT_THAT(R"(
const X = rev(u32:3);
const Y = rev(u8:3);
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[32]"),
                                      HasNodeWithType("Y", "uN[8]"))));
}

TEST(TypecheckV2BuiltinTest, RevWithArithmetic) {
  EXPECT_THAT(R"(
const Y = rev(u8:3);
const Z = rev(Y) + 1;
)",
              TypecheckSucceeds(HasNodeWithType("rev(Y) + 1", "uN[8]")));
}

TEST(TypecheckV2BuiltinTest, RevIndex) {
  EXPECT_THAT(R"(
const X:uN[32][4] = [1,2,3,4];
const Y = X[rev(u32:0)];
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[32][4]"),
                                      HasNodeWithType("Y", "uN[32]"))));
}

TEST(TypecheckV2BuiltinTest, RevArraySizeMismatch) {
  EXPECT_THAT(R"(const X:uN[rev(u2:1)][4] = [1,2,1,2];)",
              TypecheckFails(HasSizeMismatch("u32", "uN[2]")));
}

TEST(TypecheckV2BuiltinTest, RevArraySizeOK) {
  EXPECT_THAT(R"(
  // Should reverse to u32:2
const X:uN[rev(u32:0b1000000000000000000000000000000)][4] = [1,2,1,2];
)",
              TypecheckSucceeds(HasNodeWithType("X", "uN[2][4]")));
}

TEST(TypecheckV2BuiltinTest, RevTwiceArraySizeOK) {
  EXPECT_THAT(R"(
const X:uN[rev(rev(u32:2))][4] = [1,2,1,2];
)",
              TypecheckSucceeds(HasNodeWithType("X", "uN[2][4]")));
}

TEST(TypecheckV2BuiltinTest, SignEx) {
  EXPECT_THAT(R"(
const X = signex(s16:10, s16:0);
const Y = signex(u16:10, s32:0);
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("X", "sN[16]"),
                                      HasNodeWithType("Y", "sN[32]"))));
}

TEST(TypecheckV2BuiltinTest, Smulp) {
  EXPECT_THAT(R"(const Y = smulp(s16:10, s16:20);)",
              TypecheckSucceeds(HasNodeWithType("Y", "(sN[16], sN[16])")));
}

TEST(TypecheckV2BuiltinTest, Umulp) {
  EXPECT_THAT(R"(const Y = umulp(u16:10, u16:20);)",
              TypecheckSucceeds(HasNodeWithType("Y", "(uN[16], uN[16])")));
}

TEST(TypecheckV2Test, TypeAliasSelfReference) {
  EXPECT_THAT(
      "type T=uN[T::A as u2];",
      TypecheckFails(HasSubstr("Cannot find a definition for name: \"T\"")));
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

TEST(TypecheckV2Test, TypeAlias) {
  EXPECT_THAT(R"(
type MyTypeAlias = (u32, u8);
fn id(x: MyTypeAlias) -> MyTypeAlias { x }
fn f() -> MyTypeAlias { id((42, 127)) }
)",
              TypecheckSucceeds(AllOf(
                  HasNodeWithType("x", "(uN[32], uN[8])"),
                  HasNodeWithType("id", "((uN[32], uN[8])) -> (uN[32], uN[8])"),
                  HasNodeWithType("f", "() -> (uN[32], uN[8])"))));
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

TEST(TypecheckV2Test, TypeAliasOnStructInParametricFn) {
  EXPECT_THAT(R"(
struct S<X: u32> {
  x: bits[X],
}

fn f<T: u32>() -> uN[T] {
  type MyS = S<T>;
  MyS { x: 1 }.x
}

fn main() {
  let x = f<8>();
  let y = f<15>();
}
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("x", "uN[8]"),
                                      HasNodeWithType("y", "uN[15]"))));
}

TEST(TypecheckV2Test, TypeAliasInGlobalConstant) {
  EXPECT_THAT(
      R"(
type MyTypeAlias = (u32, u8);
const MY_TUPLE : MyTypeAlias = (u32:42, u8:127);
)",
      TypecheckSucceeds(HasNodeWithType("MY_TUPLE", "(uN[32], uN[8])")));
}

TEST(TypecheckV2Test, TypeAliasInLet) {
  EXPECT_THAT(
      R"(
type MyTypeAlias = (u32, u8);
fn f() -> u8 {
  let some_val: MyTypeAlias = (5, 10);
  some_val.1
}
)",
      TypecheckSucceeds(HasNodeWithType("some_val", "(uN[32], uN[8])")));
}

TEST(TypecheckV2Test, TypeAliasCircularReference) {
  EXPECT_THAT(R"(
type MyTypeAlias = AnotherAlias;
type AnotherAlias = MyTypeAlias;

fn id(x: AnotherAlias) -> AnotherAlias { x }
fn f() -> AnotherAlias { id((42, 127)) }
)",
              TypecheckFails(HasSubstr(
                  "Cannot find a definition for name: \"AnotherAlias\"")));
}

TEST(TypecheckV2Test, TypeAliasMultipleLevels) {
  EXPECT_THAT(R"(
type MyTypeAlias = (u32, u8);
type AnotherAlias = MyTypeAlias;

fn id(x: AnotherAlias) -> AnotherAlias { x }
fn f() -> AnotherAlias { id((42, 127)) }
)",
              TypecheckSucceeds(AllOf(
                  HasNodeWithType("x", "(uN[32], uN[8])"),
                  HasNodeWithType("id", "((uN[32], uN[8])) -> (uN[32], uN[8])"),
                  HasNodeWithType("f", "() -> (uN[32], uN[8])"))));
}

// ColonRefs not fully supported yet.
TEST(TypecheckV2Test, DISABLED_ColonRefTypeAlias) {
  EXPECT_THAT(
      R"(
type MyU8 = u8;
fn f() -> u8 { MyU8::MAX }
fn g() -> u8 { MyU8::ZERO }
fn h() -> u8 { MyU8::MIN }
)",
      TypecheckSucceeds(AllOf(
          HasNodeWithType("MyU8", "u8"), HasNodeWithType("f", "() -> u8"),
          HasNodeWithType("g", "() -> u8"), HasNodeWithType("MyU8", "u8"))));
}

TEST(TypecheckV2Test, TypeAliasOfStructWithBoundParametrics) {
  EXPECT_THAT(R"(
struct S<X: u32, Y: u32> {
  x: bits[X],
  y: bits[Y],
}
type MyS = S<3, 4>;
fn f() -> MyS { MyS {x: 3, y: 4 } }
)",
              TypecheckSucceeds(
                  AllOf(HasNodeWithType("f", "() -> S { x: uN[3], y: uN[4] }"),
                        HasNodeWithType("MyS", "S { x: uN[3], y: uN[4] }"))));
}

TEST(TypecheckV2Test, ParametricValuesDefinedMultipleTimesInTypeAlias) {
  EXPECT_THAT(R"(
struct S<X: u32, Y: u32 = {X * 2}> {
  x: bits[X],
  y: bits[Y],
}
type MyS = S<3>;
type MySDouble = MyS<4>;
fn f() -> uN[4] {
  let x = MySDouble { x: 3, y: 4 };
  x.y
}
)",
              TypecheckFails(HasSubstr("Parametric values defined multiple "
                                       "times for annotation: `MyS<4>`")));
}

TEST(TypecheckV2Test, ElementInTypeAliasOfStructWithBoundParametrics) {
  EXPECT_THAT(R"(
struct S<X: u32, Y: u32> {
  x: bits[X],
  y: bits[Y],
}
type MyS = S<3, 4>;
fn f() -> uN[3] {
  let x = MyS { x: 1, y: 1 };
  x.x
}
)",
              TypecheckSucceeds(
                  AllOf(HasNodeWithType("f", "() -> uN[3]"),
                        HasNodeWithType("MyS", "S { x: uN[3], y: uN[4] }"),
                        HasNodeWithType("x", "S { x: uN[3], y: uN[4] }"))));
}

TEST(TypecheckV2Test, SliceOfBitsLiteral) {
  EXPECT_THAT("const X = 0b100111[0:2];", TopNodeHasType("uN[2]"));
}

TEST(TypecheckV2Test, SliceOfBitsConstant) {
  EXPECT_THAT(R"(
const X = u6:0b100111;
const Y = X[0:2];
)",
              TypecheckSucceeds(HasNodeWithType("Y", "uN[2]")));
}

TEST(TypecheckV2Test, SliceWithOneNegativeAndOnePositiveIndex) {
  EXPECT_THAT(R"(
const X = u6:0b100111;
const Y = X[-2:6];
)",
              TypecheckSucceeds(HasNodeWithType("Y", "uN[2]")));
}

TEST(TypecheckV2Test, SliceWithBothNegativeIndices) {
  EXPECT_THAT(R"(
const X = u6:0b100111;
const Y = X[-4:-2];
)",
              TypecheckSucceeds(HasNodeWithType("Y", "uN[2]")));
}

TEST(TypecheckV2Test, SliceWithPositiveStartAndNoEnd) {
  EXPECT_THAT(R"(
const X = u6:0b100111;
const Y = X[2:];
)",
              TypecheckSucceeds(HasNodeWithType("Y", "uN[4]")));
}

TEST(TypecheckV2Test, SliceWithNoStartAndPositiveEnd) {
  EXPECT_THAT(R"(
const X = u6:0b100111;
const Y = X[:4];
)",
              TypecheckSucceeds(HasNodeWithType("Y", "uN[4]")));
}

TEST(TypecheckV2Test, SliceWithNegativeStartAndNoEnd) {
  EXPECT_THAT(R"(
const X = u6:0b100111;
const Y = X[-3:];
)",
              TypecheckSucceeds(HasNodeWithType("Y", "uN[3]")));
}

TEST(TypecheckV2Test, SliceWithNoStartAndNegativeEnd) {
  EXPECT_THAT(R"(
const X = u6:0b100111;
const Y = X[:-2];
)",
              TypecheckSucceeds(HasNodeWithType("Y", "uN[4]")));
}

TEST(TypecheckV2Test, SliceByConstants) {
  EXPECT_THAT(R"(
const X = s32:0;
const Y = s32:2;
const Z = 0b100111[X:Y];
)",
              TypecheckSucceeds(HasNodeWithType("Z", "uN[2]")));
}

TEST(TypecheckV2Test, SliceByParametrics) {
  EXPECT_THAT(R"(
fn f<A: s32, B: s32>(value: u32) -> uN[(B - A) as u32] { value[A:B] }
const X = f<1, 3>(0b100111);
const Y = f<1, 4>(0b100111);
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[2]"),
                                      HasNodeWithType("Y", "uN[3]"))));
}

TEST(TypecheckV2Test, SliceOfNonBitsFails) {
  EXPECT_THAT(
      "const X = [u32:1, 2, 3][0:2];",
      TypecheckFails(HasSubstr("Value to slice is not of 'bits' type.")));
}

TEST(TypecheckV2Test, SliceOfSignedBitsFails) {
  EXPECT_THAT("const X = (s6:0b011100)[0:4];",
              TypecheckFails(HasSubstr("Bit slice LHS must be unsigned.")));
}

TEST(TypecheckV2Test, SliceBeforeStartFails) {
  EXPECT_THAT("const X = (u6:0b011100)[-7:4];",
              TypecheckFails(
                  HasSubstr("Slice range out of bounds for array of size 6")));
}

TEST(TypecheckV2Test, SliceAfterEndFails) {
  EXPECT_THAT("const X = (u6:0b011100)[0:7];",
              TypecheckFails(
                  HasSubstr("Slice range out of bounds for array of size 6")));
}

TEST(TypecheckV2Test, WidthSliceOfBits) {
  EXPECT_THAT("const X = 0b100111[2+:u3];", TopNodeHasType("uN[3]"));
}

TEST(TypecheckV2Test, WidthSliceOfBitsWithNegativeStart) {
  EXPECT_THAT("const X = 0b100111[-5+:u3];", TopNodeHasType("uN[3]"));
}

TEST(TypecheckV2Test, WidthSliceWithNonBitsWidthAnnotationFails) {
  EXPECT_THAT("const X = 0b100111[0+:u2[2]];",
              TypecheckFails(HasSubstr(
                  "A bits type is required for a width-based slice")));
}

TEST(TypecheckV2Test, WidthSliceOfNonBitsFails) {
  EXPECT_THAT(
      "const X = [u32:1, u32:2, u32:3][0+:u32];",
      TypecheckFails(HasSubstr("Value to slice is not of 'bits' type.")));
}

TEST(TypecheckV2Test, WidthSliceOfSignedBitsFails) {
  EXPECT_THAT("const X = (s6:0b011100)[0+:u4];",
              TypecheckFails(HasSubstr("Bit slice LHS must be unsigned.")));
}

TEST(TypecheckV2Test, WidthSliceBeforeStartFails) {
  EXPECT_THAT("const X = (u6:0b011100)[-7+:u4];",
              TypecheckFails(
                  HasSubstr("Slice range out of bounds for array of size 6")));
}

TEST(TypecheckV2Test, WidthSliceAfterEndFails) {
  EXPECT_THAT("const X = (u6:0b011100)[3+:u4];",
              TypecheckFails(
                  HasSubstr("Slice range out of bounds for array of size 6")));
}

TEST(TypecheckV2Test, WidthSliceByParametrics) {
  EXPECT_THAT(R"(
fn f<A: s32, B: u32>(value: u32) -> uN[B] { value[A+:uN[B]] }
const X = f<2, 3>(0b100111);
const Y = f<1, 4>(0b100111);
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[3]"),
                                      HasNodeWithType("Y", "uN[4]"))));
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
              TypecheckFails(HasSubstr("was not constexpr")));
}

TEST(TypecheckV2Test, RangeExprSignednessMismatch) {
  EXPECT_THAT(R"(const X = u32:1..s32:2;)",
              TypecheckFails(HasSignednessMismatch("s32", "u32")));
}

TEST(TypecheckV2Test, RangeExprSizeMismatch) {
  EXPECT_THAT(
      R"(const X:u32[4] = u32:1..u32:3;)",
      TypecheckFails(HasTypeMismatch("u32:3 as s32 - u32:1 as s32", "u32[4]")));
}

TEST(TypecheckV2Test, RangeExprTypeAnnotationMismatch) {
  EXPECT_THAT(R"(const X:u32[4] = 0..u16:4;)",
              TypecheckFails(HasSizeMismatch("u16", "uN[0]")));
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
  EXPECT_THAT(
      R"(const X = u64:0..u64:0x100000000;)",
      TypecheckFails(HasSubstr("has size `4294967296` larger than u32")));
}

TEST(TypecheckV2Test, RangeExprArraySizeTypeTooLargeSigned) {
  EXPECT_THAT(R"(const X = s64:0x8000000000000000..s64:0xFFFFFFFFFFFFFFFF;)",
              TypecheckFails(
                  HasSubstr("has size `9223372036854775807` larger than u32")));
}

TEST(TypecheckV2Test, RangeExprNegativeRange) {
  // Currently if end is less than start, the range expr results in a zero-sized
  // array. This case may be regarded as a compile error in the future.
  EXPECT_THAT(R"(
const A = s8:4;
const X = A..s8:3;
)",
              TypecheckFails(HasSubstr("Range expr `A..s8:3` start value `4` "
                                       "is larger than end value `3`")));
}

}  // namespace
}  // namespace xls::dslx
