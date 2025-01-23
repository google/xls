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
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/type_system/typecheck_test_utils.h"
#include "xls/dslx/type_system_v2/type_system_test_utils.h"
#include "re2/re2.h"

namespace xls::dslx {
namespace {

using ::absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::ContainsRegex;
using ::testing::HasSubstr;

absl::StatusOr<TypecheckResult> TypecheckV2(std::string_view program) {
  return Typecheck(absl::StrCat("#![feature(type_inference_v2)]\n\n", program));
}

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

// Verifies that a failed `TypecheckV2` status message indicates a type mismatch
// between the given two types in string format.
MATCHER_P2(HasTypeMismatch, type1, type2, "") {
  return ExplainMatchResult(ContainsRegex(absl::Substitute(
                                "type mismatch.*$0.* vs. $1",
                                RE2::QuoteMeta(type1), RE2::QuoteMeta(type2))),
                            arg, result_listener);
}

// Verifies that a failed `TypecheckV2` status message indicates a size mismatch
// between the given two types in string format.
MATCHER_P2(HasSizeMismatch, type1, type2, "") {
  return ExplainMatchResult(ContainsRegex(absl::Substitute(
                                "size mismatch.*$0.* vs. $1",
                                RE2::QuoteMeta(type1), RE2::QuoteMeta(type2))),
                            arg, result_listener);
}

// Verifies that a failed `TypecheckV2` status message indicates a signedness
// mismatch between the given two types in string format.
MATCHER_P2(HasSignednessMismatch, type1, type2, "") {
  return ExplainMatchResult(ContainsRegex(absl::Substitute(
                                "signed vs. unsigned mismatch.*$0.* vs. $1",
                                RE2::QuoteMeta(type1), RE2::QuoteMeta(type2))),
                            arg, result_listener);
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
              TypecheckFails(HasSignednessMismatch("uN[32]", "sN[32]")));
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
              TypecheckFails(HasSizeMismatch("sN[4]", "sN[5]")));
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
                HasSubstr("u32:25 vs. u32:24"))));
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
      TypecheckFails(HasSizeMismatch("u25", "uN[24]")));
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
  EXPECT_THAT(R"(
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
              TypecheckFails(HasSizeMismatch("uN[32]", "uN[33]")));
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
TEST(TypecheckTest, StructInstantiateParametricXnField) {
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
              TypecheckFails(HasSizeMismatch("uN[20]", "uN[12]")));
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
              TypecheckFails(HasSizeMismatch("uN[11]", "uN[10]")));
}

TEST(TypecheckV2Test,
     ParametricFunctionWithArgumentMismatchingParameterizedSignednessFails) {
  EXPECT_THAT(R"(
fn foo<S: bool>(a: xN[S][32]) -> xN[S][32] { a }
const X = foo<true>(u32:5);
)",
              TypecheckFails(HasSignednessMismatch("uN[32]", "sN[32]")));
}

TEST(TypecheckV2Test, ParametricFunctionWithArrayMismatchingParameterizedSize) {
  EXPECT_THAT(R"(
fn foo<N: u32>(a: u32[N]) -> u32[N] { a }
const X = foo<3>([u32:1, u32:2, u32:3, u32:4]);
)",
              TypecheckFails(HasTypeMismatch("uN[32][4]", "uN[32][3]")));
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

}  // namespace
}  // namespace xls::dslx
