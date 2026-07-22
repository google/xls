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

// Tests for bit-slicing.
// TODO: davidplass - Move non-bit-slicing tests elsewhere.
namespace xls::dslx {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::AllOf;
using ::testing::HasSubstr;

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

TEST(TypecheckV2Test, ParametricCastWrapper) {
  EXPECT_THAT(R"(fn do_cast
  <WANT_S:bool, WANT_BITS:u32, GOT_S:bool, GOT_BITS:u32>
  (x: xN[GOT_S][GOT_BITS]) -> xN[WANT_S][WANT_BITS] {
    x as xN[WANT_S][WANT_BITS]
}
const X: s4 = do_cast<true, 4>(u32:64);)",
              TypecheckSucceeds(HasNodeWithType("X", "sN[4]")));
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

TEST(TypecheckV2Test, SliceWithNonstandardBoundType) {
  EXPECT_THAT(R"(
const X = s4:0;
const Y = s4:2;
const Z = 0b100111[X:Y];
)",
              TypecheckSucceeds(HasNodeWithType("Z", "uN[2]")));
}

TEST(TypecheckV2Test, SliceWithBinopInBound) {
  EXPECT_THAT(R"(
const A = s32:4;
const B = s32:2;
const X = 0b100111[A - B:4];
)",
              TypecheckSucceeds(HasNodeWithType("X", "uN[2]")));
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

TEST(TypecheckV2Test, SliceOfSignedBitsFails) {
  EXPECT_THAT("const X = (s6:0b011100)[0:4];",
              TypecheckFails(HasSubstr("Bit slice LHS must be unsigned.")));
}

TEST(TypecheckV2Test, SliceAfterEnd) {
  EXPECT_THAT("const X = (u6:0b011100)[0:7];",
              TypecheckSucceeds(HasNodeWithType("X", "uN[6]")));
}

TEST(TypecheckV2Test, WidthSliceOfBits) {
  EXPECT_THAT("const X = 0b100111[2+:u3];", TopNodeHasType("uN[3]"));
}

TEST(TypecheckV2Test, WidthSliceOfBitsWithSmallerThanU32Start) {
  EXPECT_THAT("const X = 0b100111[u2:2+:u3];", TopNodeHasType("uN[3]"));
}

TEST(TypecheckV2Test, WidthSliceOfBitsWithNegativeStartFails) {
  EXPECT_THAT("const X = 0b100111[-5+:u3];",
              TypecheckFails(HasSignednessMismatch("s4", "u6")));
}

TEST(TypecheckV2Test, WidthSliceOfBitsWithSignedStartFails) {
  EXPECT_THAT("const X = 0b100111[s32:2+:u3];",
              TypecheckFails(HasSignednessMismatch("s32", "u6")));
}

TEST(TypecheckV2Test, WidthSliceWithNonBitsWidthAnnotationFails) {
  EXPECT_THAT("const X = 0b100111[0+:u2[2]];",
              TypecheckFails(HasSubstr(
                  "A bits type is required for a width-based slice")));
}

TEST(TypecheckV2Test, WidthSliceOfSignedBitsFails) {
  EXPECT_THAT("const X = (s6:0b011100)[0+:u4];",
              TypecheckFails(HasSubstr("Bit slice LHS must be unsigned.")));
}

TEST(TypecheckV2Test, WidthSliceAfterEndFails) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result,
                           TypecheckV2("const X = (u6:0b011100)[3+:u4];"));
  ASSERT_THAT(result.tm.warnings.warnings().size(), 1);
  EXPECT_EQ(result.tm.warnings.warnings()[0].message,
            "Slice range out of bounds for array of size 6");
}

TEST(TypecheckV2Test, WidthSliceByParametrics) {
  EXPECT_THAT(R"(
fn f<A: u32, B: u32>(value: u32) -> uN[B] { value[A+:uN[B]] }
const X = f<2, 3>(0b100111);
const Y = f<1, 4>(0b100111);
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[3]"),
                                      HasNodeWithType("Y", "uN[4]"))));
}

TEST(TypecheckV2Test, WidthSliceStartOverflow) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
const A : u64 = 0;
const B = A[0x7FFFFFFFFFFFFFFF+:u32];
)"));
  ASSERT_THAT(result.tm.warnings.warnings().size(), 1);
  EXPECT_EQ(result.tm.warnings.warnings()[0].message,
            "Slice range out of bounds for array of size 64");
}

TEST(TypecheckV2Test, WithSliceSetsTypeOnStart) {
  EXPECT_THAT(R"(
fn f(x: u32, y: u32) -> u8 {
  x[3+:u8]+x[y+:u8]
}
)",
              TypecheckSucceeds(HasNodeWithType("3", "uN[32]")));
}

TEST(TypecheckV2Test, EnumCastToInt) {
  EXPECT_THAT(
      R"(
enum MyEnum : u8 {
  A = 1,
  B = 2,
}
const x = MyEnum::A as u8;
const y = MyEnum::B as u4;
const_assert!(x == u8:1);
const_assert!(y == u4:2);
)",
      TypecheckSucceeds(
          AllOf(HasNodeWithType("x", "uN[8]"), HasNodeWithType("y", "uN[4]"))));
}

TEST(TypecheckV2Test, IntCastToEnum) {
  EXPECT_THAT(
      R"(
enum MyEnum : u8 {
  A = 1,
  B = 2,
}
const x = u8:1 as MyEnum;
const y = u8:2 as MyEnum;
const_assert!(x == MyEnum::A);
const_assert!(y == MyEnum::B);
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("x", "MyEnum"),
                              HasNodeWithType("y", "MyEnum"))));
}

TEST(TypecheckV2Test, EnumBinop) {
  EXPECT_THAT(
      R"(
enum MyEnum : u8 {
  A = 1,
  B = 2,
}
const C = MyEnum::A + MyEnum::B;
)",
      TypecheckFails(HasSubstr(
          "Binary operations can only be applied to bits-typed operands")));
}

TEST(TypecheckV2Test, ImportConstantAndCast) {
  constexpr std::string_view kImported = R"(
pub const TO_CAST = s32:17;

pub type T = uN[TO_CAST as u32];
)";
  constexpr std::string_view kProgram = R"(
import imported;

const ARR = imported::T[2]:[0, 0];
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data));
  EXPECT_THAT(TypecheckV2(kProgram, "main", &import_data),
              IsOkAndHolds(HasTypeInfo(HasNodeWithType("ARR", "uN[17][2]"))));
}

TEST(TypecheckV2Test, ValueColonRefAsSliceWidthFails) {
  constexpr std::string_view kImported = R"(
pub const A = u20:1;
)";
  constexpr std::string_view kProgram = R"(
import imported;

fn f(x: u32) -> u20 {
  x[0+:imported::A]
}
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data));
  EXPECT_THAT(TypecheckV2(kProgram, "main", &import_data),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected a type, got `imported::A`")));
}

}  // namespace
}  // namespace xls::dslx
