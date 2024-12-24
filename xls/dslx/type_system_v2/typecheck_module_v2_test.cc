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
  return Typecheck(absl::StrCat("#![type_inference_version = 2]\n\n", program));
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
  bool matched = ExplainMatchResult(
      HasSubstr(absl::Substitute("node: `$0`, type: $1", arg, expected)),
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
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
const X = u2:3;
const Y = X;
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(type_info_string,
              AllOf(HasSubstr("node: `const X = u2:3;`, type: uN[2]"),
                    HasSubstr("node: `const Y = X;`, type: uN[2]")));
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
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
const X = u32:3;
const Y: u32 = 4;
const Z = X + 1 + Y + 2;
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(
      type_info_string,
      AllOf(HasSubstr("node: `const X = u32:3;`, type: uN[32]"),
            HasSubstr("node: `const Y: u32 = 4;`, type: uN[32]"),
            HasSubstr("node: `const Z = X + 1 + Y + 2;`, type: uN[32]")));
}

TEST(TypecheckV2Test, GlobalIntegerConstantEqualsSumOfAscendingAutoSizes) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
const X = u32:1;
const Z = 1 + 2 + 3 + 4 + X;
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(
      type_info_string,
      AllOf(HasSubstr("node: `const X = u32:1;`, type: uN[32]"),
            HasSubstr("node: `const Z = 1 + 2 + 3 + 4 + X;`, type: uN[32]")));
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
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
const X = u32:3;
const Y: u32 = 4;
const Z = X - 1 - Y - 2;
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(
      type_info_string,
      AllOf(HasSubstr("node: `const X = u32:3;`, type: uN[32]"),
            HasSubstr("node: `const Y: u32 = 4;`, type: uN[32]"),
            HasSubstr("node: `const Z = X - 1 - Y - 2;`, type: uN[32]")));
}

TEST(TypecheckV2Test,
     GlobalIntegerConstantEqualsProductOfConstantsAndLiterals) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
const X = u32:3;
const Y: u32 = 4;
const Z = X * 1 * Y * 2;
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(
      type_info_string,
      AllOf(HasSubstr("node: `const X = u32:3;`, type: uN[32]"),
            HasSubstr("node: `const Y: u32 = 4;`, type: uN[32]"),
            HasSubstr("node: `const Z = X * 1 * Y * 2;`, type: uN[32]")));
}

TEST(TypecheckV2Test,
     GlobalIntegerConstantEqualsQuotientOfConstantsAndLiterals) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
const X = u32:3;
const Y: u32 = 4;
const Z = X / 1 / Y / 2;
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(
      type_info_string,
      AllOf(HasSubstr("node: `const X = u32:3;`, type: uN[32]"),
            HasSubstr("node: `const Y: u32 = 4;`, type: uN[32]"),
            HasSubstr("node: `const Z = X / 1 / Y / 2;`, type: uN[32]")));
}

TEST(TypecheckV2Test,
     GlobalIntegerConstantEqualsBitwiseAndOfConstantsAndLiterals) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
const X = u32:3;
const Y: u32 = 4;
const Z = X & 1 & Y & 2;
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(
      type_info_string,
      AllOf(HasSubstr("node: `const X = u32:3;`, type: uN[32]"),
            HasSubstr("node: `const Y: u32 = 4;`, type: uN[32]"),
            HasSubstr("node: `const Z = X & 1 & Y & 2;`, type: uN[32]")));
}

TEST(TypecheckV2Test,
     GlobalIntegerConstantEqualsBitwiseOrOfConstantsAndLiterals) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
const X = u32:3;
const Y: u32 = 4;
const Z = X | 1 | Y | 2;
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(
      type_info_string,
      AllOf(HasSubstr("node: `const X = u32:3;`, type: uN[32]"),
            HasSubstr("node: `const Y: u32 = 4;`, type: uN[32]"),
            HasSubstr("node: `const Z = X | 1 | Y | 2;`, type: uN[32]")));
}

TEST(TypecheckV2Test,
     GlobalIntegerConstantEqualsBitwiseXorOfConstantsAndLiterals) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
const X = u32:3;
const Y: u32 = 4;
const Z = X ^ 1 ^ Y ^ 2;
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(
      type_info_string,
      AllOf(HasSubstr("node: `const X = u32:3;`, type: uN[32]"),
            HasSubstr("node: `const Y: u32 = 4;`, type: uN[32]"),
            HasSubstr("node: `const Z = X ^ 1 ^ Y ^ 2;`, type: uN[32]")));
}

TEST(TypecheckV2Test, GlobalIntegerConstantEqualsModOfConstantsAndLiterals) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
const X = u32:3;
const Y: u32 = 4;
const Z = X % 1 % Y % 2;
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(
      type_info_string,
      AllOf(HasSubstr("node: `const X = u32:3;`, type: uN[32]"),
            HasSubstr("node: `const Y: u32 = 4;`, type: uN[32]"),
            HasSubstr("node: `const Z = X % 1 % Y % 2;`, type: uN[32]")));
}

TEST(TypecheckV2Test,
     GlobalIntegerConstantEqualsAnotherConstantWithAnnotationOnName) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
const X: u32 = 3;
const Y = X;
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(type_info_string,
              AllOf(HasSubstr("node: `const X: u32 = 3;`, type: uN[32]"),
                    HasSubstr("node: `const Y = X;`, type: uN[32]")));
}

TEST(TypecheckV2Test, GlobalIntegerConstantRefWithSignednessConflictFails) {
  EXPECT_THAT(R"(
const X:u32 = 3;
const Y:s32 = X;
)",
              TypecheckFails(HasSignednessMismatch("uN[32]", "s32")));
}

TEST(TypecheckV2Test, GlobalIntegerConstantWithTwoLevelsOfReferences) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
const X: s20 = 3;
const Y = X;
const Z = Y;
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(type_info_string,
              AllOf(HasSubstr("node: `const X: s20 = 3;`, type: sN[20]"),
                    HasSubstr("node: `const Y = X;`, type: sN[20]"),
                    HasSubstr("node: `const Z = Y;`, type: sN[20]")));
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
              TypecheckFails(HasSignednessMismatch("bool", "s2")));
}

TEST(TypecheckV2Test, GlobalBoolConstantWithBitCountConflictFails) {
  // We don't allow this with bool literals, even though it fits.
  EXPECT_THAT("const X: u2 = true;",
              TypecheckFails(HasSizeMismatch("bool", "u2")));
}

TEST(TypecheckV2Test, GlobalBoolConstantEqualsAnotherConstant) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
const X = true;
const Y = X;
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(type_info_string,
              AllOf(HasSubstr("node: `const X = true;`, type: uN[1]"),
                    HasSubstr("node: `const Y = X;`, type: uN[1]")));
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
              TopNodeHasType("(uN[32], uN[32])"));
}

TEST(TypecheckV2Test, GlobalTupleConstantWithIntegerAnnotationFails) {
  EXPECT_THAT("const X: u32 = (1, 2);",
              TypecheckFails(HasTypeMismatch("(uN[1], uN[2])", "u32")));
}

TEST(TypecheckV2Test, GlobalTupleConstantWithNestedTuple) {
  EXPECT_THAT("const X: (u32, (s24, u32)) = (1, (-3, 2));",
              TopNodeHasType("(uN[32], (sN[24], uN[32])"));
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
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
const X: u32 = 3;
const Y = (X, s24:-1);
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(
      type_info_string,
      HasSubstr("node: `const Y = (X, s24:-1);`, type: (uN[32], sN[24])"));
}

TEST(TypecheckV2Test, GlobalTupleConstantReferencingTupleConstant) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
const X = (u32:3, s24:-1);
const Y = (X, u32:4);
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(
      type_info_string,
      HasSubstr(
          "node: `const Y = (X, u32:4);`, type: ((uN[32], sN[24]), uN[32])"));
}

TEST(TypecheckV2Test, GlobalTupleConstantWithArrays) {
  EXPECT_THAT("const X = ([u32:1, u32:2], [u32:3, u32:4, u32:5]);",
              TopNodeHasType("(uN[32][2], uN[32][3])"));
}

TEST(TypecheckV2Test, GlobalArrayConstantWithAnnotatedIntegerLiterals) {
  EXPECT_THAT("const X = [u32:1, u32:2];", TopNodeHasType("uN[32][2]"));
}

TEST(TypecheckV2Test, GlobalArrayConstantAnnotatedWithBareIntegerLiterals) {
  EXPECT_THAT("const X: u32[2] = [1, 2];", TopNodeHasType("uN[32][2]"));
}

TEST(TypecheckV2Test, GlobalArrayConstantWithTuples) {
  EXPECT_THAT("const X = [(u32:1, u32:2), (u32:3, u32:4)];",
              TopNodeHasType("(uN[32], uN[32])[2]"));
}

TEST(TypecheckV2Test, GlobalArrayConstantAnnotatedWithTooSmallSizeFails) {
  EXPECT_THAT("const X: u32[2] = [1, 2, 3];",
              TypecheckFails(HasTypeMismatch("uN[2][3]", "u32[2]")));
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
              TypecheckFails(HasSignednessMismatch("sN[24]", "u32")));
}

TEST(TypecheckV2Test, GlobalArrayConstantReferencingIntegerConstant) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
const X: u32 = 3;
const Y = [X, X];
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(type_info_string,
              HasSubstr("node: `const Y = [X, X];`, type: uN[32][2]"));
}

TEST(TypecheckV2Test, GlobalArrayConstantReferencingArrayConstant) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
const X = [u32:3, u32:4];
const Y = [X, X];
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(type_info_string,
              HasSubstr("node: `const Y = [X, X];`, type: uN[32][2][2]"));
}

TEST(TypecheckV2Test, GlobalArrayConstantCombiningArrayAndIntegerFails) {
  EXPECT_THAT("const X = [u32:3, [u32:4, u32:5]];",
              TypecheckFails(HasTypeMismatch("uN[32][2]", "u32")));
}

TEST(TypecheckV2Test, GlobalArrayConstantWithEllipsis) {
  EXPECT_THAT("const X: u32[5] = [3, 4, ...];", TopNodeHasType("uN[32][5]"));
}

TEST(TypecheckV2Test, GlobalArrayConstantWithNestedEllipsis) {
  EXPECT_THAT("const X: u32[5][2] = [[5, ...], ...];",
              TopNodeHasType("uN[32][5][2]"));
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

TEST(TypecheckV2Test, FunctionCallReturningNothing) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
fn foo() { () }
const Y = foo();
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(type_info_string,
              AllOf(HasOneLineBlockWithType("()", "()"),
                    HasSubstr("node: `const Y = foo();`, type: ()")));
}

TEST(TypecheckV2Test, FunctionCallReturningUnitTupleExplicitly) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
fn foo() -> () { () }
const Y = foo();
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(type_info_string,
              AllOf(HasOneLineBlockWithType("()", "()"),
                    HasSubstr("node: `const Y = foo();`, type: ()")));
}

TEST(TypecheckV2Test, FunctionCallReturningInteger) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
fn foo() -> u32 { 3 }
const Y = foo();
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(type_info_string,
              AllOf(HasOneLineBlockWithType("3", "uN[32]"),
                    HasSubstr("node: `const Y = foo();`, type: uN[32]")));
}

TEST(TypecheckV2Test, FunctionCallReturningBool) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
fn foo() -> bool { true }
const Y = foo();
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(type_info_string,
              AllOf(HasOneLineBlockWithType("true", "uN[1]"),
                    HasSubstr("node: `const Y = foo();`, type: uN[1]")));
}

TEST(TypecheckV2Test, FunctionCallReturningArray) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
fn foo() -> s8[3] { [1, 2, 3] }
const Y = foo();
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(type_info_string,
              AllOf(HasOneLineBlockWithType("[1, 2, 3]", "sN[8][3]"),
                    HasSubstr("node: `const Y = foo();`, type: sN[8][3]")));
}

TEST(TypecheckV2Test, FunctionCallReturningTuple) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
fn foo() -> (s8, (u32, u24)) { (1, (2, 3)) }
const Y = foo();
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(
      type_info_string,
      AllOf(HasOneLineBlockWithType("(1, (2, 3))", "(sN[8], (uN[32], uN[24]))"),
            HasSubstr(
                "node: `const Y = foo();`, type: (sN[8], (uN[32], uN[24]))")));
}

TEST(TypecheckV2Test, FunctionCallReturningFunctionCall) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
fn bar() -> s32 { 123 }
fn foo() -> s32 { bar() }
const Y = foo();
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(type_info_string,
              AllOf(HasOneLineBlockWithType("123", "sN[32]"),
                    HasOneLineBlockWithType("bar()", "sN[32]"),
                    HasSubstr("node: `const Y = foo();`, type: sN[32]")));
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
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
fn foo(a: u32) -> u32 { a }
const Y = foo(4);
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(type_info_string,
              AllOf(HasOneLineBlockWithType("a", "uN[32]"),
                    HasSubstr("node: `const Y = foo(4);`, type: uN[32]")));
}

TEST(TypecheckV2Test, FunctionCallReturningPassedInTuple) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
fn foo(a: (u32, s4)) -> (u32, s4) { a }
const Y = foo((4, -1));
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(
      type_info_string,
      AllOf(
          HasOneLineBlockWithType("a", "(uN[32], sN[4])"),
          HasSubstr("node: `const Y = foo((4, -1));`, type: (uN[32], sN[4])")));
}

TEST(TypecheckV2Test, FunctionCallReturningPassedInArray) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
fn foo(a: u32[2]) -> u32[2] { a }
const Y = foo([4, 5]);
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(
      type_info_string,
      AllOf(HasOneLineBlockWithType("a", "uN[32][2]"),
            HasSubstr("node: `const Y = foo([4, 5]);`, type: uN[32][2]")));
}

TEST(TypecheckV2Test, FunctionCallReturningSumOfPassedInIntegers) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
fn foo(a: u32, b: u32) -> u32 { a + b }
const Y = foo(4, 5);
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(type_info_string,
              AllOf(HasOneLineBlockWithType("a + b", "uN[32]"),
                    HasSubstr("node: `const Y = foo(4, 5);`, type: uN[32]")));
}

TEST(TypecheckV2Test, FunctionCallPassingInFunctionCalls) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
fn foo(a: u32, b: u32) -> u32 { a + b }
const Y = foo(foo(3, 2), foo(4, 5));
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(
      type_info_string,
      AllOf(HasOneLineBlockWithType("a + b", "uN[32]"),
            HasSubstr(
                "node: `const Y = foo(foo(3, 2), foo(4, 5));`, type: uN[32]")));
}

TEST(TypecheckV2Test, FunctionCallPassingInSum) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
const X: u32 = 4;
const Z: u32 = 5;
fn foo(a: u32) -> u32 { a }
const Y = foo(X + Z);
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(type_info_string,
              AllOf(HasOneLineBlockWithType("a", "uN[32]"),
                    HasSubstr("node: `const Y = foo(X + Z);`, type: uN[32]")));
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
fn foo(a: u4) -> u32 { a }
const Y = foo(32767);
)",
              TypecheckFails(HasSizeMismatch("uN[15]", "u4")));
}

TEST(TypecheckV2Test, FunctionCallPassingInTooLargeExplicitIntegerSizeFails) {
  EXPECT_THAT(R"(
const X:u32 = 1;
fn foo(a: u4) -> u32 { a }
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
fn foo(a: u4) -> u32 { a }
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

TEST(TypecheckV2Test, ParametricFunctionReturningIntegerParameter) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
fn foo<N: u32>() -> u32 { N }
const X = foo<3>();
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(type_info_string,
              AllOf(HasSubstr("node: `const X = foo<3>();`, type: uN[32]")));
}

TEST(TypecheckV2Test, ParametricFunctionReturningIntegerOfParameterSize) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
fn foo<N: u32>() -> uN[N] { 5 }
const X = foo<16>();
const Y = foo<17>();
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(type_info_string,
              AllOf(HasSubstr("node: `const X = foo<16>();`, type: uN[16]"),
                    HasSubstr("node: `const Y = foo<17>();`, type: uN[17]")));
}

TEST(TypecheckV2Test, ParametricFunctionReturningIntegerOfParameterSignedness) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
fn foo<S: bool>() -> xN[S][32] { 5 }
const X = foo<false>();
const Y = foo<true>();
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(type_info_string,
              AllOf(HasSubstr("node: `const X = foo<false>();`, type: uN[32]"),
                    HasSubstr("node: `const Y = foo<true>();`, type: sN[32]")));
}

TEST(TypecheckV2Test,
     ParametricFunctionReturningIntegerOfParameterSignednessAndSize) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
fn foo<S: bool, N: u32>() -> xN[S][N] { 5 }
const X = foo<false, 10>();
const Y = foo<true, 11>();
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(
      type_info_string,
      AllOf(HasSubstr("node: `const X = foo<false, 10>();`, type: uN[10]"),
            HasSubstr("node: `const Y = foo<true, 11>();`, type: sN[11]")));
}

TEST(TypecheckV2Test, ParametricFunctionTakingIntegerOfParameterizedSize) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
fn foo<N: u32>(a: uN[N]) -> uN[N] { a }
const X = foo<10>(u10:5);
const Y = foo<11>(u11:5);
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(
      type_info_string,
      AllOf(HasSubstr("node: `const X = foo<10>(u10:5);`, type: uN[10]"),
            HasSubstr("node: `const Y = foo<11>(u11:5);`, type: uN[11]")));
}

TEST(TypecheckV2Test,
     ParametricFunctionTakingIntegerOfParameterizedSignedness) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
fn foo<S: bool>(a: xN[S][32]) -> xN[S][32] { a }
const X = foo<false>(u32:5);
const Y = foo<true>(s32:5);
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(
      type_info_string,
      AllOf(HasSubstr("node: `const X = foo<false>(u32:5);`, type: uN[32]"),
            HasSubstr("node: `const Y = foo<true>(s32:5);`, type: sN[32]")));
}

TEST(TypecheckV2Test,
     ParametricFunctionTakingIntegerOfParameterizedSignednessAndSize) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
fn foo<S: bool, N: u32>(a: xN[S][N]) -> xN[S][N] { a }
const X = foo<false, 10>(u10:5);
const Y = foo<true, 11>(s11:5);
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(
      type_info_string,
      AllOf(
          HasSubstr("node: `const X = foo<false, 10>(u10:5);`, type: uN[10]"),
          HasSubstr("node: `const Y = foo<true, 11>(s11:5);`, type: sN[11]")));
}

TEST(TypecheckV2Test,
     ParametricFunctionTakingIntegerOfDefaultParameterizedSize) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
fn foo<N: u32 = {10}>(a: uN[N]) -> uN[N] { a }
const X = foo(u10:5);
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(type_info_string,
              HasSubstr("node: `const X = foo(u10:5);`, type: uN[10]"));
}

TEST(TypecheckV2Test,
     ParametricFunctionTakingIntegerOfOverriddenDefaultParameterizedSize) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
fn foo<N: u32 = {10}>(a: uN[N]) -> uN[N] { a }
const X = foo<11>(u11:5);
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(type_info_string,
              HasSubstr("node: `const X = foo<11>(u11:5);`, type: uN[11]"));
}

TEST(TypecheckV2Test,
     ParametricFunctionTakingIntegerWithDependentDefaultParametric) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
fn foo<M: u32, N: u32 = {M + 1}>(a: uN[N]) -> uN[N] { a }
const X = foo<11>(u12:5);
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(type_info_string,
              HasSubstr("node: `const X = foo<11>(u12:5);`, type: uN[12]"));
}

TEST(TypecheckV2Test,
     ParametricFunctionTakingIntegerWithOverriddenDependentDefaultParametric) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
fn foo<M: u32, N: u32 = {M + 1}>(a: uN[N]) -> uN[N] { a }
const X = foo<11, 20>(u20:5);
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(type_info_string,
              HasSubstr("node: `const X = foo<11, 20>(u20:5);`, type: uN[20]"));
}

TEST(TypecheckV2Test,
     ParametricFunctionTakingIntegerOfParameterizedSignednessAndSizeWithSum) {
  // The point here is to make sure that the uN[N] type annotation being
  // propagated onto a complex subtree in global scope is correctly dealt with.
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
const X = u32:3;
const Y = u32:4;
fn foo<N: u32>(a: uN[N]) -> uN[N] { a }
const Z = foo<32>(X + Y + X + 50);
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(
      type_info_string,
      HasSubstr("node: `const Z = foo<32>(X + Y + X + 50);`, type: uN[32]"));
}

TEST(TypecheckV2Test, ParametricFunctionTakingArrayOfParameterizedSize) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
fn foo<N: u32>(a: u32[N]) -> u32[N] { a }
const X = foo<3>([1, 2, 3]);
const Y = foo<4>([1, 2, 3, 4]);
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(
      type_info_string,
      AllOf(HasSubstr("node: `const X = foo<3>([1, 2, 3]);`, type: uN[32][3]"),
            HasSubstr(
                "node: `const Y = foo<4>([1, 2, 3, 4]);`, type: uN[32][4]")));
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
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
fn bar<A: u32>(a: uN[A]) -> uN[A] { a + 1 }
fn foo<A: u32, B: u32>(a: uN[A]) -> uN[B] { bar<B>(2) }
const X = foo<24, 23>(4);
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(type_info_string,
              HasSubstr("node: `const X = foo<24, 23>(4);`, type: uN[23]"));
}

TEST(TypecheckV2Test, ParametricFunctionInvocationNesting) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
fn foo<N: u32>(a: uN[N]) -> uN[N] { a + 1 }
const X = foo<24>(foo<24>(4) + foo<24>(5));
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(
      type_info_string,
      HasSubstr(
          "node: `const X = foo<24>(foo<24>(4) + foo<24>(5));`, type: uN[24]"));
}

TEST(TypecheckV2Test,
     ParametricFunctionUsingGlobalConstantInParametricDefault) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
const X = u32:3;
fn foo<M: u32, N: u32 = {M + X}>(a: uN[N]) -> uN[N] { a }
const Z = foo<12>(u15:1);
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(type_info_string,
              HasSubstr("node: `const Z = foo<12>(u15:1);`, type: uN[15]"));
}

TEST(TypecheckV2Test,
     ParametricFunctionCallUsingGlobalConstantInParametricArgument) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
fn foo<N: u32>(a: uN[N]) -> uN[N] { a }
const X = u32:3;
const Z = foo<X>(u3:1);
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(type_info_string,
              HasSubstr("node: `const Z = foo<X>(u3:1);`, type: uN[3]"));
}

TEST(TypecheckV2Test, ParametricFunctionCallFollowedByTypePropagation) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
fn foo<N: u32>(a: uN[N]) -> uN[N] { a }
const Y = foo<15>(u15:1);
const Z = Y + 1;
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(type_info_string,
              HasSubstr("node: `const Z = Y + 1;`, type: uN[15]"));
}

TEST(TypecheckV2Test, GlobalConstantUsingParametricFunction) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
fn foo<N: u32>(a: uN[N]) -> uN[N] { a }
const X = foo<32>(u32:3);
const Y = foo<X>(u3:1);
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(
      type_info_string,
      AllOf(HasSubstr("node: `const X = foo<32>(u32:3);`, type: uN[32]"),
            HasSubstr("node: `const Y = foo<X>(u3:1);`, type: uN[3]")));
}

TEST(TypecheckV2Test, GlobalConstantUsingAndUsedByParametricFunction) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
fn foo<N: u32>(a: uN[N]) -> uN[N] { a }
const X = foo<32>(u32:3);
const Y = foo<X>(u3:1);
fn bar<N: u32>(a: uN[N]) -> uN[N] { a + Y + foo<3>(Y) }
const Z = bar<X>(u3:1 + Y);
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(
      type_info_string,
      AllOf(HasSubstr("node: `const X = foo<32>(u32:3);`, type: uN[32]"),
            HasSubstr("node: `const Y = foo<X>(u3:1);`, type: uN[3]"),
            HasSubstr("node: `const Z = bar<X>(u3:1 + Y);`, type: uN[3]")));
}

}  // namespace
}  // namespace xls::dslx
