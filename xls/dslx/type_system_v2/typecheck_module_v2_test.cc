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
              TypecheckFails(
                  ContainsRegex("signed vs. unsigned mismatch.*u24.*vs. s24")));
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
              TypecheckFails(ContainsRegex(
                  R"(signed vs. unsigned mismatch.*sN\[3\].*vs. u32)")));
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
              TypecheckFails(ContainsRegex(
                  R"(signed vs. unsigned mismatch.*uN\[32\].*vs. s32)")));
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
  EXPECT_THAT(
      "const X: bool = 50;",
      TypecheckFails(ContainsRegex(R"(size mismatch.*uN\[6\].*vs. bool)")));
}

TEST(TypecheckV2Test, GlobalBoolConstantWithSignednessConflictFails) {
  EXPECT_THAT("const X: s2 = bool:false;",
              TypecheckFails(
                  ContainsRegex("signed vs. unsigned mismatch.*bool.*vs. s2")));
}

TEST(TypecheckV2Test, GlobalBoolConstantWithBitCountConflictFails) {
  // We don't allow this with bool literals, even though it fits.
  EXPECT_THAT("const X: u2 = true;",
              TypecheckFails(ContainsRegex("size mismatch.*bool.*vs. u2")));
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
              TypecheckFails(ContainsRegex(R"(size mismatch.*bool.*vs. u32)")));
}

TEST(TypecheckV2Test, GlobalIntegerConstantEqualsSumOfBoolsFails) {
  EXPECT_THAT(R"(
const X = true;
const Y = true;
const Z: u32 = X + Y;
)",
              TypecheckFails(ContainsRegex(R"(size mismatch.*bool.*vs. u32)")));
}

TEST(TypecheckV2Test, GlobalBoolConstantEqualsIntegerConstantFails) {
  EXPECT_THAT(R"(
const X = u32:4;
const Y: bool = X;
)",
              TypecheckFails(ContainsRegex(R"(size mismatch.*u32.*vs. bool)")));
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
              TypecheckFails(ContainsRegex(
                  R"(type mismatch.*\(uN\[1\], uN\[2\]\).* vs. u32)")));
}

TEST(TypecheckV2Test, GlobalTupleConstantWithNestedTuple) {
  EXPECT_THAT("const X: (u32, (s24, u32)) = (1, (-3, 2));",
              TopNodeHasType("(uN[32], (sN[24], uN[32])"));
}

TEST(TypecheckV2Test, GlobalTupleConstantWithNestedTupleAndTypeViolationFails) {
  EXPECT_THAT("const X: (u32, (u24, u32)) = (1, (-3, 2));",
              TypecheckFails(ContainsRegex(
                  R"(signed vs. unsigned mismatch: sN\[3\] .*vs. u24)")));
}

TEST(TypecheckV2Test, GlobalTupleConstantWithNestedTupleAndTypeConflict) {
  EXPECT_THAT("const X: (u32, (u24, u32)) = (1, (s24:3, 2));",
              TypecheckFails(ContainsRegex(
                  "signed vs. unsigned mismatch: s24 .*vs. u24")));
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
              TypecheckFails(ContainsRegex(
                  R"(type mismatch.*uN\[2\]\[3\].* vs. u32\[2\])")));
}

TEST(TypecheckV2Test, GlobalArrayConstantWithIntegerAnnotationFails) {
  EXPECT_THAT("const X: u32 = [1, 2];",
              TypecheckFails(
                  ContainsRegex(R"(type mismatch.*uN\[2\]\[2\].* vs. u32)")));
}

TEST(TypecheckV2Test, GlobalArrayConstantWithTypeViolation) {
  EXPECT_THAT("const X: u32[2] = [-3, -2];",
              TypecheckFails(ContainsRegex(
                  R"(signed vs. unsigned mismatch: sN\[3\] .*vs. u32)")));
}

TEST(TypecheckV2Test, GlobalArrayConstantWithTypeConflict) {
  EXPECT_THAT("const X: u32[2] = [s24:1, s24:2];",
              TypecheckFails(ContainsRegex(
                  R"(signed vs. unsigned mismatch: sN\[24\] .*vs. u32)")));
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
              TypecheckFails(
                  ContainsRegex(R"(type mismatch.*uN\[32\]\[2\].* vs. u32)")));
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

}  // namespace
}  // namespace xls::dslx
