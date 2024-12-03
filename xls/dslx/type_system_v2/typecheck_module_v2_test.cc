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

TEST(TypecheckV2Test, GlobalIntegerConstantWithNoTypeAnnotations) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
const X = 3;
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(type_info_string, HasSubstr("node: `const X = 3;`, type: uN[2]"));
}

TEST(TypecheckV2Test,
     GlobalIntegerConstantWithNegativeValueAndNoTypeAnnotations) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
const X = -3;
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(type_info_string,
              HasSubstr("node: `const X = -3;`, type: sN[3]"));
}

TEST(TypecheckV2Test, GlobalIntegerConstantWithTypeAnnotationOnLiteral) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
const X = u32:3;
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(type_info_string,
              HasSubstr("node: `const X = u32:3;`, type: uN[32]"));
}

TEST(TypecheckV2Test,
     GlobalIntegerConstantWithTooSmallAnnotationOnLiteralFails) {
  EXPECT_THAT(
      TypecheckV2(R"(
const X = u4:65536;
)"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Value '65536' does not fit in the bitwidth of a uN[4]")));
}

TEST(TypecheckV2Test, GlobalIntegerConstantWithTypeAnnotationOnName) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
const X: s24 = 3;
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(type_info_string,
              HasSubstr("node: `const X: s24 = 3;`, type: sN[24]"));
}

TEST(TypecheckV2Test, GlobalIntegerConstantWithSameTypeAnnotationOnBothSides) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
const X: s24 = s24:3;
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(type_info_string,
              HasSubstr("node: `const X: s24 = s24:3;`, type: sN[24]"));
}

TEST(TypecheckV2Test, GlobalIntegerConstantWithSignednessConflictFails) {
  EXPECT_THAT(
      TypecheckV2(R"(
const X: s24 = u24:3;
)"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               ContainsRegex("signed vs. unsigned mismatch.*u24.*vs. s24")));
}

TEST(TypecheckV2Test, GlobalIntegerConstantEqualsAnotherConstant) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
const X = 3;
const Y = X;
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(type_info_string,
              AllOf(HasSubstr("node: `const X = 3;`, type: uN[2]"),
                    HasSubstr("node: `const Y = X;`, type: uN[2]")));
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
  EXPECT_THAT(
      TypecheckV2(R"(
const X:u32 = 3;
const Y:s32 = X;
)"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          ContainsRegex(R"(signed vs. unsigned mismatch.*uN\[32\].*vs. s32)")));
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
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
const X = true;
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(type_info_string,
              HasSubstr("node: `const X = true;`, type: uN[1]"));
}

TEST(TypecheckV2Test, GlobalBoolConstantWithTypeAnnotationOnLiteral) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
const X = bool:true;
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(type_info_string,
              HasSubstr("node: `const X = bool:true;`, type: uN[1]"));
}

TEST(TypecheckV2Test, GlobalBoolConstantWithTypeAnnotationOnName) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
const X: bool = false;
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(type_info_string,
              HasSubstr("node: `const X: bool = false;`, type: uN[1]"));
}

TEST(TypecheckV2Test, GlobalBoolConstantWithSameTypeAnnotationOnBothSides) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
const X: bool = bool:false;
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(type_info_string,
              HasSubstr("node: `const X: bool = bool:false;`, type: uN[1]"));
}

TEST(TypecheckV2Test, GlobalBoolConstantAssignedToIntegerFails) {
  EXPECT_THAT(TypecheckV2(R"(
const X: bool = 50;
)"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       ContainsRegex(R"(size mismatch.*uN\[6\].*vs. bool)")));
}

TEST(TypecheckV2Test, GlobalBoolConstantWithSignednessConflictFails) {
  EXPECT_THAT(
      TypecheckV2(R"(
const X: s2 = bool:false;
)"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               ContainsRegex("signed vs. unsigned mismatch.*bool.*vs. s2")));
}

TEST(TypecheckV2Test, GlobalBoolConstantWithBitCountConflictFails) {
  // We don't allow this with bool literals, even though it fits.
  EXPECT_THAT(TypecheckV2(R"(
const X: u2 = true;
)"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       ContainsRegex("size mismatch.*bool.*vs. u2")));
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

TEST(TypecheckV2Test, GlobalBoolConstantEqualsIntegerConstantFails) {
  EXPECT_THAT(TypecheckV2(R"(
const X = true;
const Y: u32 = X;
)"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       ContainsRegex(R"(size mismatch.*bool.*vs. u32)")));
}

TEST(TypecheckV2Test, GlobalIntegerConstantEqualsBoolConstantFails) {
  EXPECT_THAT(TypecheckV2(R"(
const X = u32:4;
const Y: bool = X;
)"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       ContainsRegex(R"(size mismatch.*u32.*vs. bool)")));
}

}  // namespace
}  // namespace xls::dslx
