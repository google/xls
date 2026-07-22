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

namespace xls::dslx {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::AllOf;
using ::testing::HasSubstr;

TEST(TypecheckV2Test, ZeroMacroEnum) {
  EXPECT_THAT(R"(
enum E: u2 { ZERO=0, ONE=1, TWO=2}
const Y = zero!<E>();)",
              TypecheckSucceeds(HasNodeWithType("Y", "E")));
}

TEST(TypecheckV2Test, DISABLED_AllOnesMacroEnum) {
  // Type inference v2 cannot handle enums yet.
  EXPECT_THAT(R"(
enum E: u2 { ZERO=0, ONE=1, TWO=2}
const Y = all_ones!<E>();)",
              TypecheckSucceeds(HasNodeWithType("Y", "E")));
}

TEST(TypecheckV2Test, EnumBool) {
  EXPECT_THAT(
      R"(
enum MyEnum {
  A = false,
  B = true,
}
fn f(x: MyEnum) -> MyEnum {
  let y = x;
  y
}
const_assert!(MyEnum::A as u1 == u1:0);
const_assert!(MyEnum::B as u1 == u1:1);
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("MyEnum", "typeof(MyEnum)"),
                              HasNodeWithType("x", "MyEnum"),
                              HasNodeWithType("y", "MyEnum"))));
}

TEST(TypecheckV2Test, EnumInt) {
  EXPECT_THAT(
      R"(
enum MyEnum {
  A = s8:0,
  B = -128,
  C = 127,
}
fn f(x: MyEnum) -> MyEnum {
  x
}
const_assert!(MyEnum::A as s8 == 0);
const_assert!(MyEnum::B as s8 == -128);
const_assert!(MyEnum::C as s8 == 127);
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("MyEnum", "typeof(MyEnum)"))));
}

TEST(TypecheckV2Test, EnumWithAnnotation) {
  EXPECT_THAT(
      R"(
enum MyEnum : u9 {
  A = 256,
}
fn f(x: MyEnum) -> MyEnum {
  x
}
const_assert!(MyEnum::A as u9 == 256);
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("MyEnum", "typeof(MyEnum)"),
                              HasNodeWithType("x", "MyEnum"))));
}

TEST(TypecheckV2Test, EnumMixedConstLiterals) {
  EXPECT_THAT(
      R"(
const X = u8:42;
const Y = u8:10;
enum MyEnum {
  A = 64,
  B = X,
  C = Y + Y,
}
fn f(x: MyEnum) -> MyEnum {
  x
}
const_assert!(MyEnum::A as u8 == 64);
const_assert!(MyEnum::B as u8 == 42);
const_assert!(MyEnum::C as u8 == 20);
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("MyEnum", "typeof(MyEnum)"))));
}

TEST(TypecheckV2Test, EnumOutOfRange) {
  EXPECT_THAT(
      R"(
enum MyEnum : u8 {
  A = 0,
  B = 256,
}
)",
      TypecheckFails(HasSizeMismatch("u9", "u8")));
}

TEST(TypecheckV2Test, EnumInvalidType) {
  EXPECT_THAT(
      R"(
enum MyEnum : u8 {
  A = 0,
  B = "A",
}
)",
      TypecheckFails(HasTypeMismatch("u8[u32:1]", "u8")));
}

TEST(TypecheckV2Test, EnumNoTypeOrValue) {
  EXPECT_THAT(
      R"(
enum MyEnum {
}
)",
      TypecheckFails(HasSubstr("has no type annotation and no value")));
}

TEST(TypecheckV2Test, EnumTypeAlias) {
  EXPECT_THAT(
      R"(
enum MyEnum : u8 {
  A = 1,
  B = 2,
}
type Alias1 = MyEnum;
type Alias2 = Alias1;
fn f(x : Alias2) -> Alias1 {
  x
}
const_assert!(Alias1::A as u8 == 1);
const_assert!(Alias2::A as u8 == 1);
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("MyEnum", "typeof(MyEnum)"),
                              HasNodeWithType("Alias1", "MyEnum"),
                              HasNodeWithType("Alias2", "MyEnum"),
                              HasNodeWithType("x", "MyEnum"))));
}

TEST(TypecheckV2Test, EnumValue) {
  EXPECT_THAT(
      R"(
enum MyEnum : u8 {
  A = 1,
  B = 2,
}
const x = MyEnum::B;
const_assert!(x as u8 == 2);
)",
      TypecheckSucceeds(HasNodeWithType("x", "MyEnum")));
}

TEST(TypecheckV2Test, EnumValueTypeAlias) {
  EXPECT_THAT(
      R"(
enum MyEnum : u8 {
  A = 1,
  B = 2,
}
type Alias1 = MyEnum;
type Alias2 = Alias1;
type Alias3 = Alias2;
const x = Alias2::A;
const y = Alias3::B;
const_assert!(x as u8 == 1);
const_assert!(y as u8 == 2);
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("x", "MyEnum"),
                              HasNodeWithType("y", "MyEnum"))));
}

TEST(TypecheckV2Test, EnumInvalidNameRef) {
  EXPECT_THAT(
      R"(
enum MyEnum : u8 {
  A = 1,
  B = 2,
}
const x = MyEnum::C;
)",
      TypecheckFails(HasSubstr("name `C` in `MyEnum::C` is undefined")));
}

TEST(TypecheckV2Test, EnumDuplicatedMember) {
  EXPECT_THAT(
      R"(
enum MyEnum : u8 {
  A = 1,
  A = 2,
}
)",
      TypecheckFails(HasSubstr("duplicated name `A`")));
}

TEST(TypecheckV2Test, EnumSelfReference) {
  EXPECT_THAT(
      R"(
enum MyEnum : u8 {
  A = MyEnum::A,
}
)",
      TypecheckFails(
          HasSubstr("Cannot find a definition for name: \"MyEnum\"")));
}

TEST(TypecheckV2Test, EnumOtherReference) {
  EXPECT_THAT(
      R"(
enum MyEnum1 : u8 {
  A = 10,
}
const B : u8 = 20;
enum MyEnum2 : u8 {
  A = B,
  B = MyEnum1::A as u8,
}
const x = MyEnum2::A;
const y = MyEnum2::B;
const_assert!(x as u8 == 20);
const_assert!(y as u8 == 10);
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("x", "MyEnum2"),
                              HasNodeWithType("y", "MyEnum2"))));
}

TEST(TypecheckV2Test, EnumParametric) {
  XLS_EXPECT_OK(TypecheckV2(
      R"(
enum MyEnum : u8 {
  A = 1,
  B = 2,
}

fn f<E: MyEnum>() -> bool { E == MyEnum::A }
const_assert!(f<MyEnum::A>());
const_assert!(!f<MyEnum::B>());
)"));
}

TEST(TypecheckV2Test, ImportedEnum) {
  constexpr std::string_view kImported = R"(
pub enum MyEnum {
  A = s8:0,
  B = -128,
  C = 127,
}
)";
  constexpr std::string_view kProgram = R"(
import imported;

const MY_ENUM_A = imported::MyEnum::A;

const_assert!(imported::MyEnum::A as s8 == 0);
const_assert!(imported::MyEnum::B as s8 == -128);
const_assert!(imported::MyEnum::C as s8 == 127);
)";

  auto import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data));
  EXPECT_THAT(
      TypecheckV2(kProgram, "main", &import_data),
      IsOkAndHolds(HasTypeInfo(HasNodeWithType("MY_ENUM_A", "MyEnum"))));
}

TEST(TypecheckV2Test, ImportedEnumAsType) {
  constexpr std::string_view kImported = R"(
pub enum MyEnum {
  A = s8:0,
  B = -128,
  C = 127,
}

pub type E = MyEnum;
)";
  constexpr std::string_view kProgram = R"(
import imported;

type A = imported::E;

fn f(x: A) -> A {
  x
}
)";

  auto import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data));
  EXPECT_THAT(
      TypecheckV2(kProgram, "main", &import_data),
      IsOkAndHolds(HasTypeInfo(AllOf(HasNodeWithType("A", "typeof(MyEnum)"),
                                     HasNodeWithType("x", "MyEnum")))));
}

TEST(TypecheckV2Test, ImportedEnumAsTypeTwoLevel) {
  constexpr std::string_view kFirst = R"(
pub enum MyEnum {
  A = s8:0,
  B = -128,
  C = 127,
}
)";

  constexpr std::string_view kSecond = R"(
import first;

pub type E = first::MyEnum;
)";
  constexpr std::string_view kProgram = R"(
import second;

type A = second::E;

fn f(x: A) -> A {
  x
}
)";

  auto import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kFirst, "first", &import_data));
  XLS_EXPECT_OK(TypecheckV2(kSecond, "second", &import_data));
  EXPECT_THAT(
      TypecheckV2(kProgram, "main", &import_data),
      IsOkAndHolds(HasTypeInfo(AllOf(HasNodeWithType("A", "typeof(MyEnum)"),
                                     HasNodeWithType("x", "MyEnum")))));
}

TEST(TypecheckV2Test, ZeroMacroImportedEnum) {
  constexpr std::string_view kImported = R"(
pub enum E: u2 { ZERO=0, ONE=1, TWO=2}
)";
  constexpr std::string_view kProgram = R"(
import imported;

const Y = zero!<imported::E>();
)";

  auto import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data));
  EXPECT_THAT(TypecheckV2(kProgram, "main", &import_data),
              IsOkAndHolds(HasTypeInfo(HasNodeWithType("Y", "E"))));
}

TEST(TypecheckV2Test, ImportEnumValue) {
  constexpr std::string_view kImported = R"(
pub const MY_CONST = u32:5;

fn foo<N: u32>(a: uN[N]) -> uN[N] { a / 2 }

pub enum ImportEnum : u16 {
  SINGLE_MY_CONST = MY_CONST as u16,
  DOUBLE_MY_CONST = foo(MY_CONST) as u16 * u16:2,
  TRIPLE_MY_CONST = (MY_CONST * u32:3) as u16,
}
)";
  constexpr std::string_view kProgram = R"(
import imported;

type ImportedEnum = imported::ImportEnum;

fn main(x: u32) -> u32 {
  imported::ImportEnum::TRIPLE_MY_CONST as u32 +
      (ImportedEnum::DOUBLE_MY_CONST as u32) +
      (imported::ImportEnum::SINGLE_MY_CONST as u32)
})";

  auto import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data));
  EXPECT_THAT(
      TypecheckV2(kProgram, "main", &import_data),
      IsOkAndHolds(HasTypeInfo(
          AllOf(HasNodeWithType("ImportedEnum::DOUBLE_MY_CONST", "ImportEnum"),
                HasNodeWithType("imported::ImportEnum::SINGLE_MY_CONST",
                                "ImportEnum")))));
}

TEST(TypecheckV2Test, InvalidDoubleColonRef) {
  EXPECT_THAT(R"(
enum Foo: u1 {
  BAR = 1,
}

fn enum_invalid_access() -> u1 {
  Foo::foo::BAR
}
)",
              TypecheckFails(HasSubstr("name `foo` in `Foo` is undefined")));
}

TEST(TypecheckV2Test, ValueColonRefAsTypeAnnotationFails) {
  constexpr std::string_view kImported = R"(
pub enum A: u1 {
  A1 = 1,
}
)";
  constexpr std::string_view kProgram = R"(
import imported;

fn f() -> u32 {
  imported::A::A1:3
}
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data));
  EXPECT_THAT(TypecheckV2(kProgram, "main", &import_data),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected a type, got `imported::A::A1`")));
}

TEST(TypecheckV2Test, EnumInvalidImplicitCastToInt) {
  EXPECT_THAT(
      R"(
enum MyEnum : u8 {
  A = 1,
  B = 2,
}
const x : u8 = MyEnum::A;
)",
      TypecheckFails(HasTypeMismatch("MyEnum", "u8")));
}

TEST(TypecheckV2Test, EnumInvalidImplicitCastFromInt) {
  EXPECT_THAT(
      R"(
enum MyEnum : u8 {
  A = 1,
  B = 2,
}
const x : MyEnum = u8:1;
)",
      TypecheckFails(HasTypeMismatch("u8", "MyEnum")));
}

}  // namespace
}  // namespace xls::dslx
