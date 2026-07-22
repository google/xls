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

// Tests for constant/let declarations.
namespace xls::dslx {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::AllOf;
using ::testing::HasSubstr;

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

TEST(TypecheckV2Test, GlobalSignedIntegerConstantFromHexadecimalWithMSBSet) {
  EXPECT_THAT("const X: s32 = 0x80000000;", TopNodeHasType("sN[32]"));
}

TEST(TypecheckV2Test, GlobalSignedIntegerConstantFromBinaryWithMSBSet) {
  EXPECT_THAT("const X: s32 = 0b10000000000000000000000000000000;",
              TopNodeHasType("sN[32]"));
}

TEST(TypecheckV2Test, GlobalSignedIntegerConstantFromDecimalOverflow) {
  EXPECT_THAT("const X: s32 = 2147483648;",
              TypecheckFails(HasSizeMismatch("s33", "s32")));
}

TEST(TypecheckV2Test, CharLiteralImplicitSizeTest) {
  EXPECT_THAT("const X: u32 = 'a';",
              TypecheckFails(HasSizeMismatch("u8", "u32")));
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
              TypecheckFails(HasSignednessMismatch("s3", "u32")));
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
              TypecheckFails(HasSizeMismatch("s2", "s3")));
}

TEST(TypecheckV2Test, ImpossibleCoercionOfAutoToSignedFails) {
  EXPECT_THAT("const Z = 3 + s2:1;",
              TypecheckFails(HasSizeMismatch("s2", "s3")));
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

TEST(TypecheckV2Test, XnAnnotationWithMissingSignednessFails) {
  EXPECT_THAT(
      "const X = xN:3;",
      TypecheckFails(HasSubstr("`xN` requires a specified signedness.")));
}

TEST(TypecheckV2Test, XnAnnotationWithMissingBitCountFails) {
  EXPECT_THAT(
      "const X = xN[false]:3;",
      TypecheckFails(HasSubstr("`xN` requires a specified bit count.")));
}

TEST(TypecheckV2Test, XnAnnotationWithBitCountInSignednessPositionFails) {
  EXPECT_THAT("const X = xN[32]:3;",
              TypecheckFails(HasTypeMismatch("u6", "bool")));
}

TEST(TypecheckV2Test, UnAnnotationWithMissingBitCountFails) {
  EXPECT_THAT(
      "const X = uN:3;",
      TypecheckFails(HasSubstr("`uN` requires a specified bit count.")));
}

TEST(TypecheckV2Test, SnAnnotationWithMissingBitCountFails) {
  EXPECT_THAT(
      "const X = sN:3;",
      TypecheckFails(HasSubstr("`sN` requires a specified bit count.")));
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
              TypecheckFails(HasSignednessMismatch("u32", "s32")));
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
              TypecheckFails(HasSizeMismatch("u6", "bool")));
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
              TypecheckFails(HasSignednessMismatch("s3", "u24")));
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

TEST(TypecheckV2Test, GlobalConstantEqualsIndexOfTemporaryTuple) {
  EXPECT_THAT("const X = (u24:1, u32:2).0;",
              TypecheckSucceeds(HasNodeWithType("X", "uN[24]")));
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

TEST(TypecheckV2Test, GlobalStructConstantEntirelySplatted) {
  EXPECT_THAT(
      R"(
struct S { field: u32, field2: u16[2] }
const X = S { field: 5, field2: [1, 2] };
const Y = S { ..X };
)",
      TypecheckSucceeds(AllOf(
          HasNodeWithType("Y", "S { field: uN[32], field2: uN[16][2] }"))));
}

TEST(TypecheckV2Test, GlobalStructConstantWithFirstMemberSplatted) {
  EXPECT_THAT(
      R"(
struct S { field: u32, field2: u16[2] }
const X = S { field: 5, field2: [1, 2] };
const Y = S { field2: [2, 3], ..X };
)",
      TypecheckSucceeds(AllOf(
          HasNodeWithType("Y", "S { field: uN[32], field2: uN[16][2] }"))));
}

TEST(TypecheckV2Test, GlobalStructConstantWithSecondMemberSplatted) {
  EXPECT_THAT(
      R"(
struct S { field: u32, field2: u16[2] }
const X = S { field: 5, field2: [1, 2] };
const Y = S { field: 1, ..X };
)",
      TypecheckSucceeds(AllOf(
          HasNodeWithType("Y", "S { field: uN[32], field2: uN[16][2] }"))));
}

TEST(TypecheckV2Test, GlobalStructConstantWithNothingSplatted) {
  EXPECT_THAT(
      R"(
struct S { field: u32, field2: u16[2] }
const X = S { field: 5, field2: [1, 2] };
const Y = S { field2: [2, 3], field: 1, ..X };
)",
      TypecheckSucceeds(AllOf(
          HasNodeWithType("Y", "S { field: uN[32], field2: uN[16][2] }"))));
}

TEST(TypecheckV2Test, GlobalStructConstantWithImplicitParametricsSplatted) {
  EXPECT_THAT(
      R"(
struct S<N: u32> { field: uN[N] }
const X = S { field: u5: 6 };
const Y = S { ..X };
)",
      TypecheckSucceeds(HasNodeWithType("Y", "S { field: uN[5] }")));
}

TEST(TypecheckV2Test, GlobalStructConstantWithExplicitParametricsSplatted) {
  EXPECT_THAT(
      R"(
struct S<N: u32> { field: uN[N] }
const X = S { field: u5: 6 };
const Y = S<5> { ..X };
)",
      TypecheckSucceeds(HasNodeWithType("Y", "S { field: uN[5] }")));
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

TEST(TypecheckV2Test, ShiftWithSignedLiteralAmountFails) {
  EXPECT_THAT("const X = u32:1 << s32:1;",
              TypecheckFails(HasSubstr("Shift amount must be unsigned")));
}
TEST(TypecheckV2Test, GlobalArrayConstantAnnotatedWithTooSmallSizeFails) {
  EXPECT_THAT("const X: u32[2] = [1, 2, 3];",
              TypecheckFails(HasTypeMismatch("u32[3]", "u32[2]")));
}

TEST(TypecheckV2Test, GlobalArrayConstantCombiningArrayAndIntegerFails) {
  EXPECT_THAT("const X = [u32:3, [u32:4, u32:5]];",
              TypecheckFails(HasTypeMismatch("u32[2]", "u32")));
}

TEST(TypecheckV2Test, GlobalArrayConstantEmptyWithoutAnnotationFails) {
  EXPECT_THAT(
      "const X = [];",
      TypecheckFails(HasSubstr(
          "A variable or constant cannot be defined with an implicit type.")));
}

TEST(TypecheckV2Test, GlobalArrayConstantWithEllipsisAndNoElementsFails) {
  EXPECT_THAT("const X: u32[2] = [...];",
              TypecheckFails(HasSubstr("Array cannot have an ellipsis (`...`) "
                                       "without an element to repeat")));
}

TEST(TypecheckV2Test, GlobalArrayConstantWithEllipsisAndTooSmallSizeFails) {
  EXPECT_THAT("const X: u32[2] = [3, 4, 5, ...];",
              TypecheckFails(HasSubstr("Annotated array size is too small for "
                                       "explicit element count")));
}

TEST(TypecheckV2Test, GlobalArrayConstantWithIntegerAnnotationFails) {
  EXPECT_THAT("const X: u32 = [1, 2];",
              TypecheckFails(HasTypeMismatch("u2[2]", "u32")));
}

TEST(TypecheckV2Test, GlobalArrayConstantWithTypeConflict) {
  EXPECT_THAT("const X: u32[2] = [s24:1, s24:2];",
              TypecheckFails(HasSizeMismatch("s24", "u32")));
}

TEST(TypecheckV2Test, GlobalArrayConstantWithTypeViolation) {
  EXPECT_THAT("const X: u32[2] = [-3, -2];",
              TypecheckFails(HasSignednessMismatch("s3", "u32")));
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

TEST(TypecheckV2Test, GlobalConstantEqualsLogicalAndOfLiterals) {
  EXPECT_THAT("const X = true && false;", TopNodeHasType("uN[1]"));
}

TEST(TypecheckV2Test, GlobalConstantEqualsLogicalOrOfLiterals) {
  EXPECT_THAT("const X = bool:true || bool:false;", TopNodeHasType("uN[1]"));
}

TEST(TypecheckV2Test, GlobalConstantEqualsLShiftOfLiterals) {
  EXPECT_THAT("const X = u5:5 << 4;", TopNodeHasType("uN[5]"));
}

TEST(TypecheckV2Test, GlobalConstantEqualsLShiftOfLiteralsMismatchedType) {
  EXPECT_THAT("const X: u16 = u32:1 << 4;",
              TypecheckFails(HasSizeMismatch("u32", "u16")));
}

TEST(TypecheckV2Test,
     GlobalConstantEqualsLShiftOfLiteralsRhsDifferentTypeAllSpecified) {
  EXPECT_THAT("const X: u5 = u5:1 << 4;", TopNodeHasType("uN[5]"));
}

TEST(TypecheckV2Test, GlobalConstantEqualsLShiftOfLiteralsSameType) {
  EXPECT_THAT("const X = u32:1 << 4;", TopNodeHasType("uN[32]"));
}

TEST(TypecheckV2Test, GlobalConstantEqualsLShiftOfLiteralsSizeTooSmall) {
  EXPECT_THAT(
      "const X = u2:3 << 4;",
      TypecheckFails(HasSubstr("Shifting a 2-bit value (`uN[2]`) by a "
                               "constexpr shift of 4 exceeds its bit width.")));
}

TEST(TypecheckV2Test, GlobalConstantEqualsLShiftOfNonBitsType) {
  EXPECT_THAT(
      "const X = (u32:1, u5:1) << 4;",
      TypecheckFails(HasSubstr("can only be applied to bits-typed operands")));
}

TEST(TypecheckV2Test, GlobalConstantEqualsRShiftOfLiteralsSizeTooSmall) {
  EXPECT_THAT(
      "const X = u1:1 >> 4;",
      TypecheckFails(HasSubstr("Shifting a 1-bit value (`uN[1]`) by a "
                               "constexpr shift of 4 exceeds its bit width.")));
}

TEST(TypecheckV2Test, GlobalConstantEqualsRShiftOfNonBitsAmount) {
  EXPECT_THAT(
      "const X = u32:1 >> (u32:4, u4:1);",
      TypecheckFails(HasSubstr("can only be applied to bits-typed operands")));
}

TEST(TypecheckV2Test, GlobalConstantLogicalAndWithWrongRhs) {
  EXPECT_THAT(R"(
const X = true;
const Y: u32 = 4;
const Z = X && Y;
)",
              TypecheckFails(HasSizeMismatch("u32", "bool")));
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

TEST(TypecheckV2Test, GlobalConstantOvershiftByNamedConstant) {
  // Overshifting by a non-literal is allowed.
  EXPECT_THAT(R"(
const AMOUNT = u32:4;
const X = u2:3 << AMOUNT;
)",
              TypecheckSucceeds(HasNodeWithType("X", "uN[2]")));
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

TEST(TypecheckV2Test, GlobalConstantWithLogicalBinopOnWrongType) {
  EXPECT_THAT("const X = u32:5 || u32:6;",
              TypecheckFails(HasSubstr(
                  "Logical binary operations can only be applied to boolean")));
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

TEST(TypecheckV2Test, GlobalStructConstantEqualsNonStructFails) {
  EXPECT_THAT(
      R"(
struct S { x: u32 }
const X: S = u32:1;
)",
      TypecheckFails(HasTypeMismatch("S", "u32")));
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

TEST(TypecheckV2Test, GlobalStructConstantWithExtraneousMemberFails) {
  EXPECT_THAT(
      R"(
struct S { x: u32 }
const X = S { x: 1, y: u32:2};
)",
      TypecheckFails(HasSubstr("Struct `S` has no member `y`, but it was "
                               "provided by this instance")));
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
              TypecheckFails(HasTypeMismatch("u32[2]", "u32")));
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

TEST(TypecheckV2Test, GlobalStructConstantWithWrongParametricsSplattedFails) {
  EXPECT_THAT(
      R"(
struct S<N: u32> { field: uN[N] }
const X = S { field: u5: 6 };
const Y = S<10> { ..X };
)",
      TypecheckFails(HasSubstr("Value mismatch for parametric `N` of struct "
                               "`S`: u32:5 vs. u32:10")));
}

TEST(TypecheckV2Test, GlobalStructConstantWithWrongStructSplattedFails) {
  EXPECT_THAT(
      R"(
struct S { field: u32, field2: u16[2] }
struct T { field: u32, field2: u16[2] }
const X = S { field: 5, field2: [1, 2] };
const Y = T { field: 1, ..X };
)",
      TypecheckFails(HasTypeMismatch("S", "T")));
}

TEST(TypecheckV2Test, LetConstWarnsOnBadName) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
const bad_name_const = u32:5;
)"));
  ASSERT_THAT(result.tm.warnings.warnings().size(), 1);
  EXPECT_EQ(result.tm.warnings.warnings()[0].message,
            "Standard style is SCREAMING_SNAKE_CASE for constant identifiers; "
            "got: `bad_name_const`");
}

TEST(TypecheckV2Test, TypeAliasInGlobalConstant) {
  EXPECT_THAT(
      R"(
type MyTypeAlias = (u32, u8);
const MY_TUPLE : MyTypeAlias = (u32:42, u8:127);
)",
      TypecheckSucceeds(HasNodeWithType("MY_TUPLE", "(uN[32], uN[8])")));
}

TEST(TypecheckV2Test, ImportConstant) {
  constexpr std::string_view kImported = R"(
pub const SOME_CONSTANT = u32:1;
)";
  constexpr std::string_view kProgram = R"(
import imported;

fn main() -> u1 {
  let var = imported::SOME_CONSTANT;
  uN[var]:0
})";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data).status());
  EXPECT_THAT(
      TypecheckV2(kProgram, "main", &import_data),
      IsOkAndHolds(HasTypeInfo(HasNodeWithType("main", "() -> uN[1]"))));
}

TEST(TypecheckV2Test, ImportConstantLiteralSizedType) {
  constexpr std::string_view kImported = R"(
pub const SOME_CONSTANT = uN[4]:1;
)";
  constexpr std::string_view kProgram = R"(
import imported;

fn main() -> u4 {
  let var = imported::SOME_CONSTANT;
  var
})";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data).status());
  EXPECT_THAT(
      TypecheckV2(kProgram, "main", &import_data),
      IsOkAndHolds(HasTypeInfo(HasNodeWithType("main", "() -> uN[4]"))));
}

TEST(TypecheckV2Test, ImportConstantVarSizedType) {
  constexpr std::string_view kImported = R"(
const SIZE = u32:5;
pub const SOME_CONSTANT = uN[SIZE]:1;
)";
  constexpr std::string_view kProgram = R"(
import imported;

fn main() -> u5 {
  let var = imported::SOME_CONSTANT;
  var
})";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data).status());
  EXPECT_THAT(
      TypecheckV2(kProgram, "main", &import_data),
      IsOkAndHolds(HasTypeInfo(HasNodeWithType("main", "() -> uN[5]"))));
}

TEST(TypecheckV2Test, ImportNonExistingConstant) {
  constexpr std::string_view kImported = R"(
pub const SOME_CONSTANT = s32:1;
)";
  constexpr std::string_view kProgram = R"(
import imported;

fn main() -> u32 {
  imported::SOME_OTHER_CONSTANT
})";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data).status());
  EXPECT_THAT(
      TypecheckV2(kProgram, "main", &import_data),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("doesn't exist")));
}

TEST(TypecheckV2Test, UseConstant) {
  constexpr std::string_view kImported = R"(
pub const SOME_CONSTANT = s32:1;
)";
  constexpr std::string_view kProgram = R"(#![feature(use_syntax)]
use imported::SOME_CONSTANT;

fn main() -> s32 {
  let var = SOME_CONSTANT;
  var
})";
  absl::flat_hash_map<std::filesystem::path, std::string> files = {
      {std::filesystem::path("/imported.x"), std::string(kImported)},
      {std::filesystem::path("/fake_main_path.x"), std::string(kProgram)},
  };
  auto vfs = std::make_unique<FakeFilesystem>(
      files, /*cwd=*/std::filesystem::path("/"));
  ImportData import_data = CreateImportDataForTest(std::move(vfs));
  EXPECT_THAT(TypecheckV2(kProgram, "main", &import_data),
              IsOkAndHolds(HasTypeInfo(HasNodeWithType("var", "sN[32]"))));
}

TEST(TypecheckV2Test, UseNonPublicConstant) {
  constexpr std::string_view kImported = R"(
const SOME_CONSTANT = s32:1;
)";
  constexpr std::string_view kProgram = R"(#![feature(use_syntax)]
use imported::SOME_OTHER_CONSTANT;

fn main() -> u32 {
  SOME_OTHER_CONSTANT
})";
  absl::flat_hash_map<std::filesystem::path, std::string> files = {
      {std::filesystem::path("/imported.x"), std::string(kImported)},
      {std::filesystem::path("/fake_main_path.x"), std::string(kProgram)},
  };
  auto vfs = std::make_unique<FakeFilesystem>(
      files, /*cwd=*/std::filesystem::path("/"));
  ImportData import_data = CreateImportDataForTest(std::move(vfs));
  EXPECT_THAT(TypecheckV2(kProgram, "main", &import_data),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("Could not find member")));
}

TEST(TypecheckV2Test, UseNonExistingConstant) {
  constexpr std::string_view kImported = R"(
pub const SOME_CONSTANT = s32:1;
)";
  constexpr std::string_view kProgram = R"(#![feature(use_syntax)]
use imported::SOME_OTHER_CONSTANT;

fn main() -> u32 {
  SOME_OTHER_CONSTANT
})";
  absl::flat_hash_map<std::filesystem::path, std::string> files = {
      {std::filesystem::path("/imported.x"), std::string(kImported)},
      {std::filesystem::path("/fake_main_path.x"), std::string(kProgram)},
  };
  auto vfs = std::make_unique<FakeFilesystem>(
      files, /*cwd=*/std::filesystem::path("/"));
  ImportData import_data = CreateImportDataForTest(std::move(vfs));
  EXPECT_THAT(TypecheckV2(kProgram, "main", &import_data),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("Could not find member")));
}

TEST(TypecheckV2Test, UseMultipleConstants) {
  constexpr std::string_view kImported = R"(
pub const FIVE = u3:5;
pub const FOUR = u3:4;
)";
  constexpr std::string_view kProgram = R"(#![feature(use_syntax)]
use imported::{FOUR, FIVE};

fn main() -> u3 {
  FIVE - FOUR
})";
  absl::flat_hash_map<std::filesystem::path, std::string> files = {
      {std::filesystem::path("/imported.x"), std::string(kImported)},
      {std::filesystem::path("/fake_main_path.x"), std::string(kProgram)},
  };
  auto vfs = std::make_unique<FakeFilesystem>(
      files, /*cwd=*/std::filesystem::path("/"));
  ImportData import_data = CreateImportDataForTest(std::move(vfs));
  EXPECT_THAT(
      TypecheckV2(kProgram, "fake_main_path", &import_data),
      IsOkAndHolds(HasTypeInfo(HasNodeWithType("main", "() -> uN[3]"))));
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

TEST(TypecheckV2Test, ChainOfImports) {
  constexpr std::string_view kFirstImport = R"(
pub const SOME_CONSTANT = u32:1;
)";
  constexpr std::string_view kSecondImport = R"(
import first_import;

pub fn get_const() -> u32 {
  first_import::SOME_CONSTANT
}
)";
  constexpr std::string_view kProgram = R"(
import second_import;

fn main() -> u1 {
  uN[second_import::get_const()]:0
})";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(
      TypecheckV2(kFirstImport, "first_import", &import_data).status());
  XLS_EXPECT_OK(
      TypecheckV2(kSecondImport, "second_import", &import_data).status());
  EXPECT_THAT(
      TypecheckV2(kProgram, "main", &import_data),
      IsOkAndHolds(HasTypeInfo(HasNodeWithType("main", "() -> uN[1]"))));
}

TEST(TypecheckV2Test, TypeAliasSelfReference) {
  EXPECT_THAT(
      "type T=uN[T::A as u2];",
      TypecheckFails(HasSubstr("Cannot find a definition for name: \"T\"")));
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

TEST(TypecheckV2Test, ColonRefTypeAlias) {
  XLS_EXPECT_OK(TypecheckV2(
      R"(
type MyU8 = u8;
fn f() -> u8 { MyU8::MAX }
fn g() -> u8 { MyU8::ZERO }
fn h() -> u8 { MyU8::MIN }
const_assert!(f() == u8::MAX);
const_assert!(g() == u8::ZERO);
const_assert!(h() == u8::MIN);
)"));
}

TEST(TypecheckV2Test, LetDestructuringWithMemberTypePushdown) {
  EXPECT_THAT(R"(
fn main() {
  let a = u32:1;
  let (x, (y, z)) = if a == 0 {
    (3, (8, u32:5))
  } else {
    (u32:1, (u16:100, u32:5))
  };
}
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("3", "uN[32]"),
                                      HasNodeWithType("8", "uN[16]"))));
}

TEST(TypecheckV2Test, LetDestructuringWithInvocationFeedingPushdown) {
  EXPECT_THAT(R"(
fn foo<N: u32>() -> (uN[N], (u16, uN[N])) {
  (u32:1, (u16:100, u32:5))
}

fn main() {
  let a = u32:1;
  let (x, (y, z)) = if a == 0 {
    (3, (8, u32:5))
  } else {
    foo<32>()
  };
}
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("3", "uN[32]"),
                                      HasNodeWithType("8", "uN[16]"))));
}

TEST(TypecheckV2Test, LetDestructuringWithMemberTypemismatch) {
  EXPECT_THAT(R"(
fn main() {
  let a = u32:1;
  let (x, (y, z)) = if a == 0 {
    (u8:3, (8, u32:5))
  } else {
    (u32:1, (u16:100, u32:5))
  };
}
)",
              TypecheckFails(HasTypeMismatch("u32", "u8")));
}

TEST(TypecheckV2Test, LetDestructuringWithMemberTypeImplicitmismatch) {
  EXPECT_THAT(R"(
fn main() {
  let a = u32:1;
  let (x, (y, z)) = if a == 0 {
    (5000, (8, u32:5))
  } else {
    (u8:1, (u16:100, u32:5))
  };
}
)",
              TypecheckFails(HasTypeMismatch("u13", "u8")));
}

TEST(TypecheckV2Test, LetArrayWithMemberTypePushdown) {
  EXPECT_THAT(
      R"(
fn main() {
  let a = u32:1;
  let arr = if a == 0 {
    [[3, 4], [8, 9]]
  } else {
    [[u32:1, 2], [u32:100, 200]]
  };
}
)",
      TypecheckSucceeds(
          AllOf(HasNodeWithType("3", "uN[32]"), HasNodeWithType("4", "uN[32]"),
                HasNodeWithType("8", "uN[32]"), HasNodeWithType("9", "uN[32]"),
                HasNodeWithType("arr", "uN[32][2][2]"))));
}

TEST(TypecheckV2Test, LetArrayWithMemberTypePushdownMismatch) {
  EXPECT_THAT(
      R"(
fn main() {
  let a = u32:1;
  let arr = if a == 0 {
    [[u16:3, 4], [8, 9]]
  } else {
    [[u32:1, 2], [u32:100, 200]]
  };
}
)",
      TypecheckFails(HasTypeMismatch("u32", "u16")));
}

TEST(TypecheckV2Test, LetArrayWithMemberTypePushdownImplicitMismatch) {
  EXPECT_THAT(
      R"(
fn main() {
  let a = u32:1;
  let arr = if a == 0 {
    [[5000, 4], [8, 9]]
  } else {
    [[u8:1, 2], [u8:100, 200]]
  };
}
)",
      TypecheckFails(HasTypeMismatch("u13", "u8")));
}

}  // namespace
}  // namespace xls::dslx
