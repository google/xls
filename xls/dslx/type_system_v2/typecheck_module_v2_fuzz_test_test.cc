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

// Tests for fuzz tests and fuzz domains.
namespace xls::dslx {
namespace {

using ::absl_testing::IsOk;
using ::testing::HasSubstr;

TEST(TypecheckV2FuzzTestTest, FuzzTestStructDomainSuccess) {
  EXPECT_THAT(R"(
struct S { x: u32, y: u8 }
#[fuzz_test(domains=`S { x: u32:0..10, y: u8:0..5 }`)]
fn f(s: S) {}
)",
              TypecheckSucceeds(::testing::_));
}

TEST(TypecheckV2FuzzTestTest, FuzzTestStructDomainOmittedFieldsOk) {
  EXPECT_THAT(
      R"(
struct S { x: u32, y: u8 }
#[fuzz_test(domains=`S { x: u32:0..10 }`)]
fn f(s: S) {}
)",
      TypecheckSucceeds(::testing::_));
}

TEST(TypecheckV2FuzzTestTest, FuzzTestStructDomainInvalidField) {
  EXPECT_THAT(R"(
struct S { x: u32 }
#[fuzz_test(domains=`S { x: u32:0..10, z: u8:0..5 }`)]
fn f(s: S) {}
)",
              TypecheckFails(HasSubstr("Struct `S` has no member `z`")));
}

TEST(TypecheckV2FuzzTestTest, FuzzTestStructDomainFieldTypeMismatch) {
  EXPECT_THAT(R"(
struct S { x: u32 }
#[fuzz_test(domains=`S { x: u8:0..5 }`)]
fn f(s: S) {}
)",
              TypecheckFails(HasSubstr(
                  "bit count (8) does not match parameter bit count (32)")));
}

TEST(TypecheckV2FuzzTestTest, FuzzTestStructDomainNestedSuccess) {
  EXPECT_THAT(R"(
struct Inner { y: u32 }
struct Outer { x: Inner }
#[fuzz_test(domains=`Outer { x: Inner { y: u32:0..10 } }`)]
fn f(s: Outer) {}
)",
              TypecheckSucceeds(::testing::_));
}

TEST(TypecheckV2FuzzTestTest, FuzzTestStructDomainNestedInvalid) {
  EXPECT_THAT(R"(
struct Inner { y: u32 }
struct Outer { x: Inner }
#[fuzz_test(domains=`Outer { x: Inner { y: u8:0..10 } }`)]
fn f(s: Outer) {}
)",
              TypecheckFails(HasSubstr(
                  "bit count (8) does not match parameter bit count (32)")));
}

TEST(TypecheckV2FuzzTestTest, FuzzTestStructDomainEmpty) {
  EXPECT_THAT(R"(
struct S { x: u32, y: u8 }
#[fuzz_test(domains=`S {}`)]
fn f(s: S) {}
)",
              TypecheckSucceeds(::testing::_));
}

TEST(TypecheckV2FuzzTestTest, FuzzTestDerivedStructDomainSuccess) {
  EXPECT_THAT(R"(
#[fuzz_domain("MyStructDomain")]
struct MyStruct {
    x: u32,
}

fn create_f_domain() -> MyStructDomain {
   MyStructDomain {
     x: u32:0..10,
   }
}

#[fuzz_test(domains=`create_f_domain()`)]
fn f(s: MyStruct) {}
)",
              TypecheckSucceeds(::testing::_));
}

TEST(TypecheckV2FuzzTestTest, FuzzTestDerivedStructDomainTwoFieldsSuccess) {
  EXPECT_THAT(R"(
#[fuzz_domain("MyStructDomain")]
struct MyStruct {
    x: u32,
    y: u8,
}

fn create_x_domain() -> MyStructDomain {
   MyStructDomain {
     x: u32:0..10,
   }
}

fn create_y_domain() -> MyStructDomain {
   MyStructDomain {
     y: u8:0..10,
   }
}

#[fuzz_test(domains=`create_x_domain()`)]
fn x_test(s: MyStruct) {}

#[fuzz_test(domains=`create_y_domain()`)]
fn y_test(s: MyStruct) {}
)",
              TypecheckSucceeds(::testing::_));
}

TEST(TypecheckV2FuzzTestTest, FuzzTestDerivedStructDomainFieldMismatch) {
  EXPECT_THAT(R"(
#[fuzz_domain("MyStructDomain")]
struct MyStruct {
    x: u32,
}

fn create_f_domain() -> MyStructDomain {
   MyStructDomain {
     y: u32:0..10,
   }
}

#[fuzz_test(domains=`create_f_domain()`)]
fn f(s: MyStruct) {}
)",
              TypecheckFails(HasSubstr("has no member `y`")));
}

TEST(TypecheckV2FuzzTestTest, FuzzTestDerivedStructDomainFieldMissingOk) {
  EXPECT_THAT(
      R"(
#[fuzz_domain("MyStructDomain")]
struct MyStruct {
    x: u32,
    y: u8,
}

fn create_f_domain() -> MyStructDomain {
   MyStructDomain {
     x: u32:0..10,
   }
}

#[fuzz_test(domains=`create_f_domain()`)]
fn f(s: MyStruct) {}
)",
      TypecheckSucceeds(::testing::_));
}

TEST(TypecheckV2FuzzTestTest, FuzzTestDerivedStructDomainExtraField) {
  EXPECT_THAT(R"(
#[fuzz_domain("MyStructDomain")]
struct MyStruct {
    x: u32,
}

fn create_f_domain() -> MyStructDomain {
   MyStructDomain {
     x: u32:0..10,
     y: u32:0..10,
   }
}

#[fuzz_test(domains=`create_f_domain()`)]
fn f(s: MyStruct) {}
)",
              TypecheckFails(HasSubstr("has no member `y`")));
}

TEST(TypecheckV2FuzzTestTest, FuzzTestDerivedStructDomainNestedSuccess) {
  EXPECT_THAT(R"(
#[fuzz_domain("InnerDomain")]
struct Inner {
    y: u32,
}

#[fuzz_domain("OuterDomain")]
struct Outer {
    x: Inner,
}

fn create_f_domain() -> OuterDomain {
   OuterDomain {
     x: InnerDomain {
       y: u32:0..10,
     },
   }
}

#[fuzz_test(domains=`create_f_domain()`)]
fn f(s: Outer) {}
)",
              TypecheckSucceeds(::testing::_));
}

TEST(TypecheckV2FuzzTestTest,
     FuzzTestDerivedStructDomainNestedMissingFieldSuccess) {
  EXPECT_THAT(R"(
#[fuzz_domain("InnerDomain")]
struct Inner {
    y: u32,
}

#[fuzz_domain("OuterDomain")]
struct Outer {
    x: Inner,
}

fn create_f_domain() -> OuterDomain {
   OuterDomain {
     x: InnerDomain { },
   }
}

#[fuzz_test(domains=`create_f_domain()`)]
fn f(s: Outer) {}
)",
              TypecheckSucceeds(::testing::_));
}

TEST(TypecheckV2FuzzTestTest, FuzzTestDerivedStructDomainValueMismatch) {
  EXPECT_THAT(R"(
#[fuzz_domain("MyStructDomain")]
struct MyStruct {
    x: u8,
}

fn create_f_domain() -> MyStructDomain {
   MyStructDomain {
     x: u32:0..1000,
   }
}

#[fuzz_test(domains=`create_f_domain()`)]
fn f(s: MyStruct) {}
)",
              TypecheckFails(HasSubstr(
                  "bit count (32) does not match parameter bit count (8)")));
}

TEST(TypecheckV2FuzzTestTest,
     FuzzTestDerivedStructDomainStructVsRangeMismatch) {
  EXPECT_THAT(
      R"(
#[fuzz_domain("InnerDomain")]
struct Inner {
    y: u32,
}

#[fuzz_domain("OuterDomain")]
struct Outer {
    x: Inner,
}

fn create_f_domain() -> OuterDomain {
   OuterDomain {
     x: u32:0..10,
   }
}

#[fuzz_test(domains=`create_f_domain()`)]
fn f(s: Outer) {}
)",
      TypecheckFails(HasSubstr("expects a struct")));
}

TEST(TypecheckV2FuzzTestTest,
     FuzzTestDerivedStructDomainRangeVsStructMismatch) {
  EXPECT_THAT(R"(
#[fuzz_domain("InnerDomain")]
struct Inner {
    y: u32,
}

#[fuzz_domain("OuterDomain")]
struct Outer {
    x: u32,
}

fn create_f_domain() -> OuterDomain {
   OuterDomain {
     x: InnerDomain { y: u32:0..10 },
   }
}

#[fuzz_test(domains=`create_f_domain()`)]
fn f(s: Outer) {}
)",
              TypecheckFails(HasSubstr("Expected range or array domain")));
}

TEST(TypecheckV2FuzzTestTest, FuzzTestSignedRangeOverflow) {
  EXPECT_THAT(
      R"(
#[fuzz_test(domains=`s32:-200..200`)]
fn f(x: s8) {}
)",
      TypecheckFails(
          HasSubstr("bit count (32) does not match parameter bit count (8)")));
}

TEST(TypecheckV2FuzzTestTest, FuzzTestTupleDomainParamNonTuple) {
  EXPECT_THAT(
      R"(
#[fuzz_test(domains=`(u32:0..10, u32:0..10)`)]
fn f(x: u32) {}
)",
      TypecheckFails(HasSubstr("implies a tuple or array type")));
}

TEST(TypecheckV2FuzzTestTest, FuzzTestTupleDomainSizeMismatch) {
  EXPECT_THAT(
      R"(
#[fuzz_test(domains=`(u32:0..10, u32:0..10, u32:0..10)`)]
fn f(x: (u32, u32)) {}
)",
      TypecheckFails(HasSubstr("tuple size (3) does not match parameter")));
}

TEST(TypecheckV2FuzzTestTest, FuzzTestStructDomainParamNonStruct) {
  EXPECT_THAT(R"(
#[fuzz_domain("MyStructDomain")]
struct MyStruct {
    x: u32,
}
fn create_f_domain() -> MyStructDomain {
   MyStructDomain {
     x: u32:0..10,
   }
}
#[fuzz_test(domains=`create_f_domain()`)]
fn f(x: (u32, u32)) {}
)",
              TypecheckFails(HasSubstr("implies a struct type")));
}

TEST(TypecheckV2FuzzTestTest, FuzzTestUnsupportedDomainType) {
  EXPECT_THAT(
      R"(
const C = u32:42;
#[fuzz_test(domains=`C`)]
fn f(x: u8) {}
)",
      TypecheckFails(HasSubstr("Unsupported fuzz test domain `C` of type")));
}

TEST(TypecheckV2FuzzTestTest, FuzzTestImportedRangeDomainTypeMismatch) {
  constexpr std::string_view kImported = R"(
    pub const DOMAIN = u32:0..10;
  )";
  constexpr std::string_view kProgram = R"(
    import imported;
    #[fuzz_test(domains=`imported::DOMAIN`)]
    fn f(x: u8) {}
  )";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data));
  EXPECT_THAT(
      TypecheckV2(kProgram, "main", &import_data),
      absl_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("bit count (32) does not match parameter bit count (8)")));
}

TEST(TypecheckV2FuzzTestTest, FuzzTestImportedTupleDomainParamNonTuple) {
  constexpr std::string_view kImported = R"(
    pub const DOMAIN = (u32:0..10, u32:0..10);
  )";
  constexpr std::string_view kProgram = R"(
    import imported;
    #[fuzz_test(domains=`imported::DOMAIN`)]
    fn f(x: u8) {}
  )";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data));
  EXPECT_THAT(
      TypecheckV2(kProgram, "main", &import_data),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             HasSubstr("implies a tuple or array type, but "
                                       "parameter `x: u8` is of type")));
}

TEST(TypecheckV2FuzzTestTest, FuzzTestImportedTupleDomainSizeMismatch) {
  constexpr std::string_view kImported = R"(
    pub const DOMAIN = (u32:0..10, u32:0..10);
  )";
  constexpr std::string_view kProgram = R"(
    import imported;
    #[fuzz_test(domains=`imported::DOMAIN`)]
    fn f(x: (u8, u8, u8)) {}
  )";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data));
  EXPECT_THAT(TypecheckV2(kProgram, "main", &import_data),
              absl_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  HasSubstr("tuple size (2) does not match parameter")));
}

TEST(TypecheckV2FuzzTestTest, FuzzTestImportedStructDomainParamNonStruct) {
  constexpr std::string_view kImported = R"(
    #[fuzz_domain("SDomain")]
    pub struct S { x: u32 }
    pub const DOMAIN = SDomain { x: u32:0..10 };
  )";
  constexpr std::string_view kProgram = R"(
    import imported;
    #[fuzz_test(domains=`imported::DOMAIN`)]
    fn f(x: u8) {}
  )";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data));
  EXPECT_THAT(
      TypecheckV2(kProgram, "main", &import_data),
      absl_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "implies a struct type, but parameter `x: u8` is of type")));
}

TEST(TypecheckV2FuzzTestTest, FuzzTestImportedStructDomainSizeMismatch) {
  constexpr std::string_view kImported = R"(
    #[fuzz_domain("SDomain")]
    pub struct S { x: u32 }
    pub const DOMAIN = SDomain { x: u32:0..10 };
  )";
  constexpr std::string_view kProgram = R"(
    import imported;
    struct LocalStruct { x: u8, y: u8 }
    #[fuzz_test(domains=`imported::DOMAIN`)]
    fn f(x: LocalStruct) {}
  )";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data));
  EXPECT_THAT(TypecheckV2(kProgram, "main", &import_data),
              absl_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  HasSubstr("struct size (1) does not match parameter")));
}

TEST(TypecheckV2FuzzTestTest, FuzzTestSignedRangeNegativeForUnsignedParam) {
  EXPECT_THAT(
      R"(
#[fuzz_test(domains=`s32:-10..10`)]
fn f(x: u8) {}
)",
      TypecheckFails(
          HasSubstr("bit count (32) does not match parameter bit count (8)")));
}

TEST(TypecheckV2FuzzTestTest, FuzzTestSignedRangePositiveForUnsignedParam) {
  EXPECT_THAT(
      R"(
#[fuzz_test(domains=`s32:0..10`)]
fn f(x: u8) {}
)",
      TypecheckFails(
          HasSubstr("bit count (32) does not match parameter bit count (8)")));
}

TEST(TypecheckV2FuzzTestTest, FuzzTestUnsignedRangeOverflowForSignedParam) {
  EXPECT_THAT(
      R"(
#[fuzz_test(domains=`u32:0..200`)]
fn f(x: s8) {}
)",
      TypecheckFails(
          HasSubstr("bit count (32) does not match parameter bit count (8)")));
}

TEST(TypecheckV2FuzzTestTest, FuzzTestUnsignedRangeForSignedParam) {
  EXPECT_THAT(
      R"(
#[fuzz_test(domains=`u32:0..10`)]
fn f(x: s8) {}
)",
      TypecheckFails(
          HasSubstr("bit count (32) does not match parameter bit count (8)")));
}

TEST(TypecheckV2FuzzTestTest, FuzzTestEmptyDomainFunctionDoesNotCrash) {
  EXPECT_THAT(
      R"(
#[fuzz_domain("SDomain")]
struct S { x: u8 }

fn my_domain() -> SDomain {}

#[fuzz_test(domains=`my_domain()`)]
fn f(x: S) {}
)",
      TypecheckFails(HasSubstr("type mismatch: SDomain vs. ()")));
}

TEST(TypecheckV2FuzzTestTest, FuzzTestDomainLocalVariableCaught) {
  EXPECT_THAT(
      R"(
#[fuzz_domain("SDomain")]
struct S { x: u8 }

fn my_domain() -> SDomain {
  let tmp = SDomain { x: u32:1000..1001 };
  tmp
}

#[fuzz_test(domains=`my_domain()`)]
fn f(x: S) {}
)",
      TypecheckFails(
          HasSubstr("bit count (32) does not match parameter bit count (8)")));
}

TEST(TypecheckV2FuzzTestTest, FuzzTestDomainNestedFunctionCaught) {
  EXPECT_THAT(
      R"(
#[fuzz_domain("SDomain")]
struct S { x: u8 }

fn helper() -> SDomain {
  SDomain { x: u32:1000..1001 }
}

fn my_domain() -> SDomain {
  helper()
}

#[fuzz_test(domains=`my_domain()`)]
fn f(x: S) {}
)",
      TypecheckFails(
          HasSubstr("bit count (32) does not match parameter bit count (8)")));
}

TEST(TypecheckV2FuzzTestTest, FuzzTestDomainImportedCaught) {
  constexpr std::string_view kImported = R"(
    #[fuzz_domain("SDomain")]
    pub struct S { x: u8 }
    pub const DOMAIN = SDomain { x: u32:1000..1001 };
  )";
  constexpr std::string_view kProgram = R"(
    import imported;
    #[fuzz_test(domains=`imported::DOMAIN`)]
    fn f(x: imported::S) {}
  )";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data));
  EXPECT_THAT(
      TypecheckV2(kProgram, "main", &import_data),
      absl_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("bit count (32) does not match parameter bit count (8)")));
}

TEST(TypecheckV2FuzzTestTest, FuzzTestDomainArrayMismatch) {
  EXPECT_THAT(
      R"(
#[fuzz_test(domains=`[u32:0, 16384]`)]
fn f(x: u8) {}
)",
      TypecheckFails(HasSubstr("Fuzz test domain bit count (32) does not match "
                               "parameter bit count (8)")));
}

TEST(TypecheckV2FuzzTestTest, FuzzTestDomainsEmptyTupleAlwaysMatches) {
  EXPECT_THAT(R"(
#[fuzz_test(domains=`()`)]
fn f(x: u8) {}
)",
              TypecheckSucceeds(::testing::_));
}

TEST(TypecheckV2FuzzTestTest,
     FuzzTestEmptyTupleDomainAlwaysMatchesMultipleParams) {
  EXPECT_THAT(R"(
#[fuzz_test(domains=`u32:0..1, ()`)]
fn f(x: u32, y: u8) {}
)",
              TypecheckSucceeds(::testing::_));
}

TEST(TypecheckV2FuzzTestTest, FuzzTestTupleDomainSuccess) {
  EXPECT_THAT(R"(
const D = (u32:0..1, u8:0..2);
#[fuzz_test(domains=`D`)]
fn f(x: (u32, u8)) {}
)",
              TypecheckSucceeds(::testing::_));
}

TEST(TypecheckV2FuzzTestTest, FuzzTestTupleDomainWithAliasSuccess) {
  EXPECT_THAT(R"(
type my_tuple = (u32, u8);
const D = (u32:0..1, u8:0..2);
#[fuzz_test(domains=`D`)]
fn f(x: my_tuple) {}
)",
              TypecheckSucceeds(::testing::_));
}

TEST(TypecheckV2FuzzTestTest, FuzzTestTupleDomainNestedSuccess) {
  EXPECT_THAT(R"(
const D = ((u32:0..1, u8:0..2), u16:0..3);
#[fuzz_test(domains=`D`)]
fn f(x: ((u32, u8), u16)) {}
)",
              TypecheckSucceeds(::testing::_));
}

TEST(TypecheckV2FuzzTestTest, FuzzTestConstTupleDomainSizeMismatch) {
  EXPECT_THAT(
      R"(
const D = (u32:0..1, u8:0..2, u16:0..3);
#[fuzz_test(domains=`D`)]
fn f(x: (u32, u8)) {}
)",
      TypecheckFails(HasSubstr(
          "Fuzz test domain tuple size (3) does not match parameter")));
}

TEST(TypecheckV2FuzzTestTest, FuzzTestTupleDomainTypeMismatch) {
  EXPECT_THAT(
      R"(
const D = (u32:0..1, u16:0..2);
#[fuzz_test(domains=`D`)]
fn f(x: (u32, u8)) {}
)",
      TypecheckFails(
          HasSubstr("bit count (16) does not match parameter bit count (8)")));
}

TEST(TypecheckV2FuzzTestTest, FuzzTestTupleDomainNotATuple) {
  EXPECT_THAT(R"(
#[fuzz_test(domains=`u32:0..1`)]
fn f(x: (u32, u8)) {}
)",
              TypecheckFails(HasSubstr("expects a tuple")));
}

TEST(TypecheckV2FuzzTestTest, FuzzTestTupleDomainParamNotATuple) {
  EXPECT_THAT(R"(
const D = (u32:0..1, u32:0..2);
#[fuzz_test(domains=`D`)]
fn f(x: u32) {}
)",
              TypecheckFails(HasSubstr("implies a tuple or array type")));
}

TEST(TypecheckV2FuzzTestTest, FuzzTestTupleDomainDirectSuccess) {
  EXPECT_THAT(R"(
#[fuzz_test(domains=`(u32:0..1, u8:0..2)`)]
fn f(x: (u32, u8)) {}
)",
              TypecheckSucceeds(::testing::_));
}

TEST(TypecheckV2FuzzTestTest, FuzzTestTupleDomainMismatch) {
  EXPECT_THAT(
      R"(
#[fuzz_test(domains=`u32:0..1, u8:0..2`)]
fn f(x: (u32, u8)) {}
)",
      TypecheckFails(HasSubstr("fuzz_test attribute has 2 domain arguments, "
                               "but function `f` has 1 parameter")));
}

TEST(TypecheckV2FuzzTestTest, FuzzTestDomainRangeBitSizeMismatch) {
  EXPECT_THAT(
      R"(
#[fuzz_test(domains=`u32:0..16384, u32:0..u32:16284`)]
fn f(x: u8, y:u32) {}
)",
      TypecheckFails(
          HasSubstr("bit count (32) does not match parameter bit count (8)")));
}

TEST(TypecheckV2FuzzTestTest, FuzzTestDomainBitSizeMismatchAlias) {
  EXPECT_THAT(
      R"(
type u8_alias = u8;
#[fuzz_test(domains=`u32:0..16384`)]
fn f(x: u8_alias) {}
)",
      TypecheckFails(
          HasSubstr("bit count (32) does not match parameter bit count (8)")));
}

TEST(TypecheckV2FuzzTestTest, FuzzTestConstRangeMismatch) {
  EXPECT_THAT(
      R"(
const C = u32:0..1000;
#[fuzz_test(domains=`C`)]
fn f(x: u8) {}
)",
      TypecheckFails(
          HasSubstr("bit count (32) does not match parameter bit count (8)")));
}

TEST(TypecheckV2FuzzTestTest, FuzzTestCountMismatchTooMany) {
  EXPECT_THAT(
      R"(
#[fuzz_test(domains=`u32:0..1, u32:0..2`)]
fn f(x: u32) {}
)",
      TypecheckFails(HasSubstr("fuzz_test attribute has 2 domain arguments, "
                               "but function `f` has 1 parameter")));
}

TEST(TypecheckV2FuzzTestTest, FuzzTestCountMismatchTooFew) {
  EXPECT_THAT(
      R"(
#[fuzz_test(domains=`u32:0..1`)]
fn g(x: u32, y: u32) {}
)",
      TypecheckFails(HasSubstr("fuzz_test attribute has 1 domain argument, "
                               "but function `g` has 2 parameters")));
}

TEST(TypecheckV2FuzzTestTest, FuzzTestEnum) {
  EXPECT_THAT(R"(
enum E {
  E0 = 0,
  E1 = 1,
}
#[fuzz_test(domains=`[E::E0, E::E1]`)]
fn f(x: E) {}
)",
              TypecheckSucceeds(::testing::_));
}

TEST(TypecheckV2FuzzTestTest, FuzzTestImportedEnum) {
  constexpr std::string_view kImported = R"(
pub enum E {
  E0 = 0,
  E1 = 1,
}
)";
  constexpr std::string_view kProgram = R"(
import imported;
#[fuzz_test(domains=`[imported::E::E0, imported::E::E1]`)]
fn f(x: imported::E) {}
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data));
  EXPECT_THAT(TypecheckV2(kProgram, "main", &import_data), IsOk());
}

TEST(TypecheckV2FuzzTestTest, FuzzTestNoParameters) {
  EXPECT_THAT(
      R"(
#[fuzz_test]
fn f() {}
)",
      TypecheckFails(HasSubstr("Can only fuzz test functions with at least 1 "
                               "parameter; function `f` has 0")));
}

TEST(TypecheckV2FuzzTestTest, FuzzTestAttributeWithZeroArguments) {
  EXPECT_THAT(R"(
#[fuzz_test]
fn f(x: u32) {}
)",
              TypecheckSucceeds(::testing::_));
}

TEST(TypecheckV2FuzzTestTest, FuzzTestParametric) {
  EXPECT_THAT(
      R"(
#[fuzz_test(domains=`u32:0..1`)]
fn f<N: u32>(x: uN[N]) {
   x+uN[N]:1
}
)",
      TypecheckFails(HasSubstr("Cannot fuzz test parametric function `f`")));
}

TEST(TypecheckV2Test, FuzzTestDerivedStructDomainPartiallyNestedSuccess) {
  EXPECT_THAT(R"(
#[fuzz_domain("InnerDomain")]
struct Inner {
    y: u32,
    z: u8,
}

#[fuzz_domain("OuterDomain")]
struct Outer {
    x: Inner,
}

fn create_f_domain() -> OuterDomain {
   OuterDomain {
     x: InnerDomain {
       y: u32:0..10,
     },
   }
}

#[fuzz_test(domains=`create_f_domain()`)]
fn f(s: Outer) {}
)",
              TypecheckSucceeds(::testing::_));
}

TEST(TypecheckV2Test, FuzzTestStructDomainTypeAlias) {
  EXPECT_THAT(R"(
#[fuzz_domain("InnerDomain")]
struct Inner {
    y: u32,
}

type MyAlias = Inner;

#[fuzz_domain("OuterDomain")]
struct Outer {
    a: MyAlias,
}

fn create_f_domain() -> OuterDomain {
   OuterDomain {
     a: InnerDomain {
       y: u32:0..10,
     },
   }
}

#[fuzz_test(domains=`create_f_domain()`)]
fn f(s: Outer) {}
)",
              TypecheckSucceeds(::testing::_));
}

TEST(TypecheckV2Test, FuzzTestStructDomainArray) {
  EXPECT_THAT(R"(
#[fuzz_domain("InnerDomain")]
struct Inner {
    y: u32,
}

#[fuzz_domain("OuterDomain")]
struct Outer {
    b: Inner[2],
}

fn create_f_domain() -> OuterDomain {
   OuterDomain {
     b: [InnerDomain { y: u32:0..10 }, InnerDomain { y: u32:0..10 }],
   }
}

#[fuzz_test(domains=`create_f_domain()`)]
fn f(s: Outer) {}
)",
              TypecheckSucceeds(::testing::_));
}

TEST(TypecheckV2Test, FuzzTestStructDomainTuple) {
  EXPECT_THAT(R"(
#[fuzz_domain("InnerDomain")]
struct Inner {
    y: u32,
}

#[fuzz_domain("OuterDomain")]
struct Outer {
    c: (Inner, u8),
}

fn create_f_domain() -> OuterDomain {
   OuterDomain {
     c: (InnerDomain { y: u32:0..10 }, u8:0..10),
   }
}

#[fuzz_test(domains=`create_f_domain()`)]
fn f(s: Outer) {}
)",
              TypecheckSucceeds(::testing::_));
}

TEST(TypecheckV2FuzzTestTest, ArrayParam) {
  EXPECT_THAT(
      R"(
#[fuzz_test(domains=`(u32:0..10, u32:1..=11)`)]
fn fuzz_custom_array_domain(t: u32[2]) {}
)",
      TypecheckSucceeds(::testing::_));
}

TEST(TypecheckV2FuzzTestTest, ArrayParamSmallerSucceeds) {
  EXPECT_THAT(
      R"(
#[fuzz_test(domains=`(u32:0..10, u32:1..=11)`)]
fn fuzz_custom_array_domain(t: u32[4]) {}
)",
      TypecheckSucceeds(::testing::_));
}

TEST(TypecheckV2FuzzTestTest, ArrayParamBadDomainSizeMismatch) {
  EXPECT_THAT(
      R"(
#[fuzz_test(domains=`(u32:0..10, u32:1..=11, [u32:0])`)]
fn fuzz_custom_array_domain(t: u32[2]) {}
)",
      TypecheckFails(HasSubstr("domain array size (3) exceeds")));
}

TEST(TypecheckV2FuzzTestTest, ArrayParamBadDomainTypeMismatch) {
  EXPECT_THAT(
      R"(
#[fuzz_test(domains=`(u32:0..10, u32:1..=11, [u8:0])`)]
fn fuzz_custom_array_domain(t: u32[3]) {}
)",
      TypecheckFails(HasSubstr(
          "domain bit count (8) does not match parameter bit count (32)")));
}

TEST(TypecheckV2FuzzTestTest, ArrayParamNotTupleDomain) {
  EXPECT_THAT(
      R"(
#[fuzz_test(domains=`[u32:0..10, u32:1..11]`)]
fn fuzz_custom_array_domain(t: u32[2]) {}
)",
      TypecheckFails(HasSubstr(
          "Fuzz test domain for array parameter `t: u32[2]` must be a tuple")));
}

TEST(TypecheckV2FuzzTestTest, MixedArrayAndScalarSucceeds) {
  EXPECT_THAT(
      R"(
#[fuzz_test(domains=`(u32:0..10, u32:1..11), u32:0..5`)]
fn f(t: u32[2], s: u32) {}
)",
      TypecheckSucceeds(::testing::_));
}

}  // namespace
}  // namespace xls::dslx
