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

#include "xls/dslx/domain_utils.h"

#include <optional>
#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/type_system/type.h"
#include "xls/ir/bits.h"

namespace xls::dslx {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

TEST(DomainUtilsTest, ValidateFunctionDomainsSucceeds) {
  constexpr std::string_view kProgram = R"(
#[fuzz_test(domains=`u32:10..=12, [true]`)]
fn prop(x: u32, flag: bool) -> bool {
    flag && x >= u32:10
}
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      FuzzTestFunction * ftf,
      tm.module->GetMemberOrError<FuzzTestFunction>("prop"));
  ASSERT_TRUE(ftf->domains().has_value());

  XLS_EXPECT_OK(ValidateFunctionDomains(ftf->fn(), *ftf->domains(),
                                        *tm.type_info, tm.warnings, import_data,
                                        import_data.file_table()));
}

TEST(DomainUtilsTest, ValidateFunctionDomainsTypeMismatchFails) {
  constexpr std::string_view kProgram = R"(
#[fuzz_test(domains=`u32:10..=12, [true]`)]
fn prop(x: u32, flag: bool) -> bool {
    flag && x >= u32:10
}

fn wrong_types(x: u16, flag: bool) -> bool {
    flag && x >= u16:10
}
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      FuzzTestFunction * ftf,
      tm.module->GetMemberOrError<FuzzTestFunction>("prop"));
  ASSERT_TRUE(ftf->domains().has_value());
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * wrong_types,
      tm.module->GetMemberOrError<Function>("wrong_types"));

  EXPECT_THAT(ValidateFunctionDomains(*wrong_types, *ftf->domains(),
                                      *tm.type_info, tm.warnings, import_data,
                                      import_data.file_table()),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Domain bit count (32) does not match "
                                 "parameter bit count (16)")));
}

TEST(DomainUtilsTest, EnumerateArbitraryUnsignedScalar) {
  constexpr std::string_view kProgram = R"(
fn target(x: u2) -> u2 { x }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn,
                           tm.module->GetMemberOrError<Function>("target"));
  std::optional<Type*> u2_type = tm.type_info->GetItem(fn->params()[0]);
  ASSERT_TRUE(u2_type.has_value());

  EXPECT_THAT(EnumerateArbitraryDomain("x", **u2_type, /*max_points=*/4),
              IsOkAndHolds(ElementsAre(
                  InterpValue::MakeUBits(2, 0), InterpValue::MakeUBits(2, 1),
                  InterpValue::MakeUBits(2, 2), InterpValue::MakeUBits(2, 3))));
}

TEST(DomainUtilsTest, EnumerateArbitrarySignedScalar) {
  constexpr std::string_view kProgram = R"(
fn target(x: s2) -> s2 { x }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn,
                           tm.module->GetMemberOrError<Function>("target"));
  std::optional<Type*> s2_type = tm.type_info->GetItem(fn->params()[0]);
  ASSERT_TRUE(s2_type.has_value());

  EXPECT_THAT(EnumerateArbitraryDomain("x", **s2_type, /*max_points=*/4),
              IsOkAndHolds(ElementsAre(
                  InterpValue::MakeSBits(2, -2), InterpValue::MakeSBits(2, -1),
                  InterpValue::MakeSBits(2, 0), InterpValue::MakeSBits(2, 1))));
}

TEST(DomainUtilsTest, EnumerateArbitraryZeroBitScalar) {
  constexpr std::string_view kProgram = R"(
fn target(x: uN[0]) -> uN[0] { x }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn,
                           tm.module->GetMemberOrError<Function>("target"));
  std::optional<Type*> u0_type = tm.type_info->GetItem(fn->params()[0]);
  ASSERT_TRUE(u0_type.has_value());

  EXPECT_THAT(EnumerateArbitraryDomain("x", **u0_type, /*max_points=*/4),
              IsOkAndHolds(ElementsAre(InterpValue::MakeZeroValue(
                  /*is_signed=*/false, /*bit_count=*/0))));
}

TEST(DomainUtilsTest, EnumerateArbitraryBool) {
  constexpr std::string_view kProgram = R"(
fn target(flag: bool) -> bool { flag }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn,
                           tm.module->GetMemberOrError<Function>("target"));
  std::optional<Type*> bool_type = tm.type_info->GetItem(fn->params()[0]);
  ASSERT_TRUE(bool_type.has_value());

  EXPECT_THAT(EnumerateArbitraryDomain("flag", **bool_type, /*max_points=*/4),
              IsOkAndHolds(ElementsAre(InterpValue::MakeBool(false),
                                       InterpValue::MakeBool(true))));
}

TEST(DomainUtilsTest, EnumerateArbitraryEnum) {
  constexpr std::string_view kProgram = R"(
enum Color : u2 {
    RED = 0,
    GREEN = 1,
    BLUE = 2,
}

fn target(c: Color) -> Color { c }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn,
                           tm.module->GetMemberOrError<Function>("target"));
  XLS_ASSERT_OK_AND_ASSIGN(EnumDef * color_def,
                           tm.module->GetMemberOrError<EnumDef>("Color"));
  std::optional<Type*> enum_type = tm.type_info->GetItem(fn->params()[0]);
  ASSERT_TRUE(enum_type.has_value());

  EXPECT_THAT(
      EnumerateArbitraryDomain("c", **enum_type, /*max_points=*/4),
      IsOkAndHolds(ElementsAre(
          InterpValue::MakeEnum(UBits(0, 2), /*is_signed=*/false, color_def),
          InterpValue::MakeEnum(UBits(1, 2), /*is_signed=*/false, color_def),
          InterpValue::MakeEnum(UBits(2, 2), /*is_signed=*/false, color_def))));
}

TEST(DomainUtilsTest, EnumerateArbitraryEnumExceedsMaxPointsFails) {
  constexpr std::string_view kProgram = R"(
enum Color : u2 {
    RED = 0,
    GREEN = 1,
    BLUE = 2,
}

fn target(c: Color) -> Color { c }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn,
                           tm.module->GetMemberOrError<Function>("target"));
  std::optional<Type*> enum_type = tm.type_info->GetItem(fn->params()[0]);
  ASSERT_TRUE(enum_type.has_value());

  EXPECT_THAT(EnumerateArbitraryDomain("c", **enum_type, /*max_points=*/2),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("exceeds maximum size")));
}

TEST(DomainUtilsTest, EnumerateArbitraryCompositeStruct) {
  constexpr std::string_view kProgram = R"(
struct Pair {
    flag: bool,
    val: u1,
}

fn target(p: Pair) -> u1 { u1:0 }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn,
                           tm.module->GetMemberOrError<Function>("target"));
  std::optional<Type*> pair_type = tm.type_info->GetItem(fn->params()[0]);
  ASSERT_TRUE(pair_type.has_value());

  EXPECT_THAT(EnumerateArbitraryDomain("p", **pair_type, /*max_points=*/16,
                                       /*allow_composite=*/true),
              IsOkAndHolds(ElementsAre(
                  InterpValue::MakeTuple({InterpValue::MakeBool(false),
                                          InterpValue::MakeBool(false)}),
                  InterpValue::MakeTuple({InterpValue::MakeBool(false),
                                          InterpValue::MakeBool(true)}),
                  InterpValue::MakeTuple({InterpValue::MakeBool(true),
                                          InterpValue::MakeBool(false)}),
                  InterpValue::MakeTuple({InterpValue::MakeBool(true),
                                          InterpValue::MakeBool(true)}))));
}

TEST(DomainUtilsTest, EnumerateArbitraryCompositeTuple) {
  constexpr std::string_view kProgram = R"(
fn target(t: (bool, u1)) -> u1 { t.1 }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn,
                           tm.module->GetMemberOrError<Function>("target"));
  std::optional<Type*> tuple_type = tm.type_info->GetItem(fn->params()[0]);
  ASSERT_TRUE(tuple_type.has_value());

  EXPECT_THAT(EnumerateArbitraryDomain("t", **tuple_type, /*max_points=*/16,
                                       /*allow_composite=*/true),
              IsOkAndHolds(ElementsAre(
                  InterpValue::MakeTuple({InterpValue::MakeBool(false),
                                          InterpValue::MakeBool(false)}),
                  InterpValue::MakeTuple({InterpValue::MakeBool(false),
                                          InterpValue::MakeBool(true)}),
                  InterpValue::MakeTuple({InterpValue::MakeBool(true),
                                          InterpValue::MakeBool(false)}),
                  InterpValue::MakeTuple({InterpValue::MakeBool(true),
                                          InterpValue::MakeBool(true)}))));
}

TEST(DomainUtilsTest, EnumerateArbitraryCompositeArray) {
  constexpr std::string_view kProgram = R"(
fn target(a: bool[2]) -> bool { a[0] }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn,
                           tm.module->GetMemberOrError<Function>("target"));
  std::optional<Type*> array_type = tm.type_info->GetItem(fn->params()[0]);
  ASSERT_TRUE(array_type.has_value());
  XLS_ASSERT_OK_AND_ASSIGN(
      InterpValue ff, InterpValue::MakeArray({InterpValue::MakeBool(false),
                                              InterpValue::MakeBool(false)}));
  XLS_ASSERT_OK_AND_ASSIGN(
      InterpValue ft, InterpValue::MakeArray({InterpValue::MakeBool(false),
                                              InterpValue::MakeBool(true)}));
  XLS_ASSERT_OK_AND_ASSIGN(
      InterpValue tf, InterpValue::MakeArray({InterpValue::MakeBool(true),
                                              InterpValue::MakeBool(false)}));
  XLS_ASSERT_OK_AND_ASSIGN(
      InterpValue tt, InterpValue::MakeArray({InterpValue::MakeBool(true),
                                              InterpValue::MakeBool(true)}));

  EXPECT_THAT(EnumerateArbitraryDomain("a", **array_type, /*max_points=*/16,
                                       /*allow_composite=*/true),
              IsOkAndHolds(ElementsAre(ff, ft, tf, tt)));
}

TEST(DomainUtilsTest, EnumerateArbitraryCompositeDisallowedFails) {
  constexpr std::string_view kProgram = R"(
struct Pair {
    flag: bool,
    val: u1,
}

fn target(p: Pair) -> u1 { u1:0 }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn,
                           tm.module->GetMemberOrError<Function>("target"));
  std::optional<Type*> pair_type = tm.type_info->GetItem(fn->params()[0]);
  ASSERT_TRUE(pair_type.has_value());

  EXPECT_THAT(EnumerateArbitraryDomain("p", **pair_type, /*max_points=*/16,
                                       /*allow_composite=*/false),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("unsupported type")));
}

TEST(DomainUtilsTest, EnumerateArbitraryWidthExceedsMaxPointsFails) {
  constexpr std::string_view kProgram = R"(
fn target(x: u8) -> u8 { x }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn,
                           tm.module->GetMemberOrError<Function>("target"));
  std::optional<Type*> u8_type = tm.type_info->GetItem(fn->params()[0]);
  ASSERT_TRUE(u8_type.has_value());

  EXPECT_THAT(
      EnumerateArbitraryDomain("x", **u8_type, /*max_points=*/16),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("exceeds the maximum single-parameter width")));
}

TEST(DomainUtilsTest, EnumerateArbitraryNonPositiveMaxPointsFails) {
  constexpr std::string_view kProgram = R"(
fn target(x: u2) -> u2 { x }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn,
                           tm.module->GetMemberOrError<Function>("target"));
  std::optional<Type*> u2_type = tm.type_info->GetItem(fn->params()[0]);
  ASSERT_TRUE(u2_type.has_value());

  EXPECT_THAT(EnumerateArbitraryDomain("x", **u2_type, /*max_points=*/0),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("max_points must be positive")));
}

TEST(DomainUtilsTest, EnumerateDomainForOneParamInclusiveRange) {
  constexpr std::string_view kProgram = R"(
#[fuzz_test(domains=`u32:10..=12`)]
fn prop(x: u32) -> bool { x >= u32:10 }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      FuzzTestFunction * ftf,
      tm.module->GetMemberOrError<FuzzTestFunction>("prop"));
  ASSERT_TRUE(ftf->domains().has_value());
  std::optional<Type*> fn_type_opt = tm.type_info->GetItem(&ftf->fn());
  ASSERT_TRUE(fn_type_opt.has_value());
  const FunctionType& fn_type = (*fn_type_opt)->AsFunction();

  EXPECT_THAT(EnumerateDomainForOneParam(
                  &import_data, tm.type_info, "x", *fn_type.params()[0],
                  (*ftf->domains())->members()[0], /*max_points=*/10),
              IsOkAndHolds(ElementsAre(InterpValue::MakeUBits(32, 10),
                                       InterpValue::MakeUBits(32, 11),
                                       InterpValue::MakeUBits(32, 12))));
}

TEST(DomainUtilsTest, EnumerateDomainForOneParamExclusiveRange) {
  constexpr std::string_view kProgram = R"(
#[fuzz_test(domains=`u32:10..12`)]
fn prop(x: u32) -> bool { x >= u32:10 }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      FuzzTestFunction * ftf,
      tm.module->GetMemberOrError<FuzzTestFunction>("prop"));
  ASSERT_TRUE(ftf->domains().has_value());
  std::optional<Type*> fn_type_opt = tm.type_info->GetItem(&ftf->fn());
  ASSERT_TRUE(fn_type_opt.has_value());
  const FunctionType& fn_type = (*fn_type_opt)->AsFunction();

  EXPECT_THAT(EnumerateDomainForOneParam(
                  &import_data, tm.type_info, "x", *fn_type.params()[0],
                  (*ftf->domains())->members()[0], /*max_points=*/10),
              IsOkAndHolds(ElementsAre(InterpValue::MakeUBits(32, 10),
                                       InterpValue::MakeUBits(32, 11))));
}

TEST(DomainUtilsTest, EnumerateDomainForOneParamSignedRange) {
  constexpr std::string_view kProgram = R"(
#[fuzz_test(domains=`s4:-1..=1`)]
fn prop(x: s4) -> bool { x >= s4:-1 }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      FuzzTestFunction * ftf,
      tm.module->GetMemberOrError<FuzzTestFunction>("prop"));
  ASSERT_TRUE(ftf->domains().has_value());
  std::optional<Type*> fn_type_opt = tm.type_info->GetItem(&ftf->fn());
  ASSERT_TRUE(fn_type_opt.has_value());
  const FunctionType& fn_type = (*fn_type_opt)->AsFunction();

  EXPECT_THAT(EnumerateDomainForOneParam(
                  &import_data, tm.type_info, "x", *fn_type.params()[0],
                  (*ftf->domains())->members()[0], /*max_points=*/10),
              IsOkAndHolds(ElementsAre(InterpValue::MakeSBits(4, -1),
                                       InterpValue::MakeSBits(4, 0),
                                       InterpValue::MakeSBits(4, 1))));
}

TEST(DomainUtilsTest, EnumerateDomainForOneParamEmptyRangeFails) {
  constexpr std::string_view kProgram = R"(
#[fuzz_test(domains=`u32:5..5`)]
fn prop(x: u32) -> bool { true }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      FuzzTestFunction * ftf,
      tm.module->GetMemberOrError<FuzzTestFunction>("prop"));
  ASSERT_TRUE(ftf->domains().has_value());
  std::optional<Type*> fn_type_opt = tm.type_info->GetItem(&ftf->fn());
  ASSERT_TRUE(fn_type_opt.has_value());
  const FunctionType& fn_type = (*fn_type_opt)->AsFunction();

  EXPECT_THAT(
      EnumerateDomainForOneParam(
          &import_data, tm.type_info, "x", *fn_type.params()[0],
          (*ftf->domains())->members()[0], /*max_points=*/10),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("empty domain")));
}

TEST(DomainUtilsTest, EnumerateDomainForOneParamRangeExceedsMaxPointsFails) {
  constexpr std::string_view kProgram = R"(
#[fuzz_test(domains=`u32:10..=15`)]
fn prop(x: u32) -> bool { true }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      FuzzTestFunction * ftf,
      tm.module->GetMemberOrError<FuzzTestFunction>("prop"));
  ASSERT_TRUE(ftf->domains().has_value());
  std::optional<Type*> fn_type_opt = tm.type_info->GetItem(&ftf->fn());
  ASSERT_TRUE(fn_type_opt.has_value());
  const FunctionType& fn_type = (*fn_type_opt)->AsFunction();

  EXPECT_THAT(EnumerateDomainForOneParam(
                  &import_data, tm.type_info, "x", *fn_type.params()[0],
                  (*ftf->domains())->members()[0], /*max_points=*/3),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("exceeds maximum size")));
}

TEST(DomainUtilsTest, EnumerateDomainForOneParamEmptyTupleIsArbitrary) {
  constexpr std::string_view kProgram = R"(
#[fuzz_test(domains=`()`)]
fn prop(x: u2) -> bool { true }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      FuzzTestFunction * ftf,
      tm.module->GetMemberOrError<FuzzTestFunction>("prop"));
  ASSERT_TRUE(ftf->domains().has_value());
  std::optional<Type*> fn_type_opt = tm.type_info->GetItem(&ftf->fn());
  ASSERT_TRUE(fn_type_opt.has_value());
  const FunctionType& fn_type = (*fn_type_opt)->AsFunction();

  EXPECT_THAT(EnumerateDomainForOneParam(
                  &import_data, tm.type_info, "x", *fn_type.params()[0],
                  (*ftf->domains())->members()[0], /*max_points=*/10),
              IsOkAndHolds(ElementsAre(
                  InterpValue::MakeUBits(2, 0), InterpValue::MakeUBits(2, 1),
                  InterpValue::MakeUBits(2, 2), InterpValue::MakeUBits(2, 3))));
}

TEST(DomainUtilsTest, EnumerateDomainForOneParamInlineStruct) {
  constexpr std::string_view kProgram = R"(
struct Pair {
    a: u2,
    b: bool,
}

#[fuzz_test(domains=`Pair { a: u2:1..3 }`)]
fn prop(p: Pair) -> bool { true }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      FuzzTestFunction * ftf,
      tm.module->GetMemberOrError<FuzzTestFunction>("prop"));
  ASSERT_TRUE(ftf->domains().has_value());
  std::optional<Type*> fn_type_opt = tm.type_info->GetItem(&ftf->fn());
  ASSERT_TRUE(fn_type_opt.has_value());
  const FunctionType& fn_type = (*fn_type_opt)->AsFunction();

  EXPECT_THAT(EnumerateDomainForOneParam(&import_data, tm.type_info, "p",
                                         *fn_type.params()[0],
                                         (*ftf->domains())->members()[0],
                                         /*max_points=*/10),
              IsOkAndHolds(ElementsAre(
                  InterpValue::MakeTuple({InterpValue::MakeUBits(2, 1),
                                          InterpValue::MakeBool(false)}),
                  InterpValue::MakeTuple({InterpValue::MakeUBits(2, 1),
                                          InterpValue::MakeBool(true)}),
                  InterpValue::MakeTuple({InterpValue::MakeUBits(2, 2),
                                          InterpValue::MakeBool(false)}),
                  InterpValue::MakeTuple({InterpValue::MakeUBits(2, 2),
                                          InterpValue::MakeBool(true)}))));
}

TEST(DomainUtilsTest, EnumerateDomainForOneParamTuple) {
  constexpr std::string_view kProgram = R"(
#[fuzz_test(domains=`(u8:1..3, [true])`)]
fn prop(t: (u8, bool)) -> bool { t.1 }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      FuzzTestFunction * ftf,
      tm.module->GetMemberOrError<FuzzTestFunction>("prop"));
  ASSERT_TRUE(ftf->domains().has_value());
  std::optional<Type*> fn_type_opt = tm.type_info->GetItem(&ftf->fn());
  ASSERT_TRUE(fn_type_opt.has_value());
  const FunctionType& fn_type = (*fn_type_opt)->AsFunction();

  EXPECT_THAT(EnumerateDomainForOneParam(&import_data, tm.type_info, "t",
                                         *fn_type.params()[0],
                                         (*ftf->domains())->members()[0],
                                         /*max_points=*/10),
              IsOkAndHolds(ElementsAre(
                  InterpValue::MakeTuple({InterpValue::MakeUBits(8, 1),
                                          InterpValue::MakeBool(true)}),
                  InterpValue::MakeTuple({InterpValue::MakeUBits(8, 2),
                                          InterpValue::MakeBool(true)}))));
}

TEST(DomainUtilsTest, EnumerateDomainForOneParamArrayWithTrailingArbitrary) {
  constexpr std::string_view kProgram = R"(
#[fuzz_test(domains=`(u1:1..=1,)`)]
fn prop(a: u1[2]) -> bool { a[0] == u1:1 }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      FuzzTestFunction * ftf,
      tm.module->GetMemberOrError<FuzzTestFunction>("prop"));
  ASSERT_TRUE(ftf->domains().has_value());
  std::optional<Type*> fn_type_opt = tm.type_info->GetItem(&ftf->fn());
  ASSERT_TRUE(fn_type_opt.has_value());
  const FunctionType& fn_type = (*fn_type_opt)->AsFunction();
  XLS_ASSERT_OK_AND_ASSIGN(
      InterpValue row0, InterpValue::MakeArray({InterpValue::MakeUBits(1, 1),
                                                InterpValue::MakeBool(false)}));
  XLS_ASSERT_OK_AND_ASSIGN(
      InterpValue row1, InterpValue::MakeArray({InterpValue::MakeUBits(1, 1),
                                                InterpValue::MakeBool(true)}));

  EXPECT_THAT(EnumerateDomainForOneParam(
                  &import_data, tm.type_info, "a", *fn_type.params()[0],
                  (*ftf->domains())->members()[0], /*max_points=*/10),
              IsOkAndHolds(ElementsAre(row0, row1)));
}

TEST(DomainUtilsTest, EnumerateDomainForOneParamConstRange) {
  constexpr std::string_view kProgram = R"(
const R = u8:1..4;

#[fuzz_test(domains=`R`)]
fn prop(x: u8) -> bool { x >= u8:1 }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      FuzzTestFunction * ftf,
      tm.module->GetMemberOrError<FuzzTestFunction>("prop"));
  ASSERT_TRUE(ftf->domains().has_value());
  std::optional<Type*> fn_type_opt = tm.type_info->GetItem(&ftf->fn());
  ASSERT_TRUE(fn_type_opt.has_value());
  const FunctionType& fn_type = (*fn_type_opt)->AsFunction();

  EXPECT_THAT(EnumerateDomainForOneParam(
                  &import_data, tm.type_info, "x", *fn_type.params()[0],
                  (*ftf->domains())->members()[0], /*max_points=*/10),
              IsOkAndHolds(ElementsAre(InterpValue::MakeUBits(8, 1),
                                       InterpValue::MakeUBits(8, 2),
                                       InterpValue::MakeUBits(8, 3))));
}

TEST(DomainUtilsTest, EnumerateDomainForOneParamConstEmptyTuple) {
  constexpr std::string_view kProgram = R"(
const ANY = ();

#[fuzz_test(domains=`ANY`)]
fn prop(x: u2) -> bool { true }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      FuzzTestFunction * ftf,
      tm.module->GetMemberOrError<FuzzTestFunction>("prop"));
  ASSERT_TRUE(ftf->domains().has_value());
  std::optional<Type*> fn_type_opt = tm.type_info->GetItem(&ftf->fn());
  ASSERT_TRUE(fn_type_opt.has_value());
  const FunctionType& fn_type = (*fn_type_opt)->AsFunction();

  EXPECT_THAT(EnumerateDomainForOneParam(
                  &import_data, tm.type_info, "x", *fn_type.params()[0],
                  (*ftf->domains())->members()[0], /*max_points=*/10),
              IsOkAndHolds(ElementsAre(
                  InterpValue::MakeUBits(2, 0), InterpValue::MakeUBits(2, 1),
                  InterpValue::MakeUBits(2, 2), InterpValue::MakeUBits(2, 3))));
}

TEST(DomainUtilsTest, EnumerateDomainForOneParamDerivedStructFromHelper) {
  constexpr std::string_view kProgram = R"(
#[fuzz_domain("PairDomain")]
struct Pair {
    a: u2,
    b: bool,
}

fn make_domain() -> PairDomain {
    PairDomain { a: u2:1..3 }
}

#[fuzz_test(domains=`make_domain()`)]
fn prop(p: Pair) -> bool { true }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      FuzzTestFunction * ftf,
      tm.module->GetMemberOrError<FuzzTestFunction>("prop"));
  ASSERT_TRUE(ftf->domains().has_value());
  std::optional<Type*> fn_type_opt = tm.type_info->GetItem(&ftf->fn());
  ASSERT_TRUE(fn_type_opt.has_value());
  const FunctionType& fn_type = (*fn_type_opt)->AsFunction();

  EXPECT_THAT(EnumerateDomainForOneParam(&import_data, tm.type_info, "p",
                                         *fn_type.params()[0],
                                         (*ftf->domains())->members()[0],
                                         /*max_points=*/10),
              IsOkAndHolds(ElementsAre(
                  InterpValue::MakeTuple({InterpValue::MakeUBits(2, 1),
                                          InterpValue::MakeBool(false)}),
                  InterpValue::MakeTuple({InterpValue::MakeUBits(2, 1),
                                          InterpValue::MakeBool(true)}),
                  InterpValue::MakeTuple({InterpValue::MakeUBits(2, 2),
                                          InterpValue::MakeBool(false)}),
                  InterpValue::MakeTuple({InterpValue::MakeUBits(2, 2),
                                          InterpValue::MakeBool(true)}))));
}

TEST(DomainUtilsTest, EnumerateDomainForOneParamConstTuple) {
  constexpr std::string_view kProgram = R"(
const T = (u8:1..3, [false]);

#[fuzz_test(domains=`T`)]
fn prop(t: (u8, bool)) -> bool { !t.1 }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      FuzzTestFunction * ftf,
      tm.module->GetMemberOrError<FuzzTestFunction>("prop"));
  ASSERT_TRUE(ftf->domains().has_value());
  std::optional<Type*> fn_type_opt = tm.type_info->GetItem(&ftf->fn());
  ASSERT_TRUE(fn_type_opt.has_value());
  const FunctionType& fn_type = (*fn_type_opt)->AsFunction();

  EXPECT_THAT(EnumerateDomainForOneParam(&import_data, tm.type_info, "t",
                                         *fn_type.params()[0],
                                         (*ftf->domains())->members()[0],
                                         /*max_points=*/10),
              IsOkAndHolds(ElementsAre(
                  InterpValue::MakeTuple({InterpValue::MakeUBits(8, 1),
                                          InterpValue::MakeBool(false)}),
                  InterpValue::MakeTuple({InterpValue::MakeUBits(8, 2),
                                          InterpValue::MakeBool(false)}))));
}

TEST(DomainUtilsTest,
     EnumerateDomainForOneParamConstArrayWithTrailingArbitrary) {
  constexpr std::string_view kProgram = R"(
const A = ([u1:1],);

#[fuzz_test(domains=`A`)]
fn prop(a: u1[2]) -> bool { a[0] == u1:1 }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      FuzzTestFunction * ftf,
      tm.module->GetMemberOrError<FuzzTestFunction>("prop"));
  ASSERT_TRUE(ftf->domains().has_value());
  std::optional<Type*> fn_type_opt = tm.type_info->GetItem(&ftf->fn());
  ASSERT_TRUE(fn_type_opt.has_value());
  const FunctionType& fn_type = (*fn_type_opt)->AsFunction();
  XLS_ASSERT_OK_AND_ASSIGN(
      InterpValue row0, InterpValue::MakeArray({InterpValue::MakeUBits(1, 1),
                                                InterpValue::MakeBool(false)}));
  XLS_ASSERT_OK_AND_ASSIGN(
      InterpValue row1, InterpValue::MakeArray({InterpValue::MakeUBits(1, 1),
                                                InterpValue::MakeBool(true)}));

  EXPECT_THAT(EnumerateDomainForOneParam(
                  &import_data, tm.type_info, "a", *fn_type.params()[0],
                  (*ftf->domains())->members()[0], /*max_points=*/10),
              IsOkAndHolds(ElementsAre(row0, row1)));
}

TEST(DomainUtilsTest, EnumerateDomainForOneParamElementOfArray) {
  constexpr std::string_view kProgram = R"(
#[fuzz_test(domains=`[u8:3, u8:7]`)]
fn prop(x: u8) -> bool { x >= u8:3 }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      FuzzTestFunction * ftf,
      tm.module->GetMemberOrError<FuzzTestFunction>("prop"));
  ASSERT_TRUE(ftf->domains().has_value());
  std::optional<Type*> fn_type_opt = tm.type_info->GetItem(&ftf->fn());
  ASSERT_TRUE(fn_type_opt.has_value());
  const FunctionType& fn_type = (*fn_type_opt)->AsFunction();

  EXPECT_THAT(EnumerateDomainForOneParam(
                  &import_data, tm.type_info, "x", *fn_type.params()[0],
                  (*ftf->domains())->members()[0], /*max_points=*/10),
              IsOkAndHolds(ElementsAre(InterpValue::MakeUBits(8, 3),
                                       InterpValue::MakeUBits(8, 7))));
}

TEST(DomainUtilsTest,
     EnumerateDomainForOneParamElementOfArrayExceedsMaxPointsFails) {
  constexpr std::string_view kProgram = R"(
#[fuzz_test(domains=`[u8:1, u8:2, u8:3]`)]
fn prop(x: u8) -> bool { true }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      FuzzTestFunction * ftf,
      tm.module->GetMemberOrError<FuzzTestFunction>("prop"));
  ASSERT_TRUE(ftf->domains().has_value());
  std::optional<Type*> fn_type_opt = tm.type_info->GetItem(&ftf->fn());
  ASSERT_TRUE(fn_type_opt.has_value());
  const FunctionType& fn_type = (*fn_type_opt)->AsFunction();

  EXPECT_THAT(EnumerateDomainForOneParam(
                  &import_data, tm.type_info, "x", *fn_type.params()[0],
                  (*ftf->domains())->members()[0], /*max_points=*/2),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("exceeds maximum size")));
}

TEST(DomainUtilsTest, EnumerateDomainForOneParamEmptyElementOfArrayFails) {
  constexpr std::string_view kProgram = R"(
const EMPTY = u8[0]:[];

fn target(x: u8) -> u8 { x }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(ConstantDef * empty_def,
                           tm.module->GetConstantDef("EMPTY"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn,
                           tm.module->GetMemberOrError<Function>("target"));
  std::optional<Type*> u8_type = tm.type_info->GetItem(fn->params()[0]);
  ASSERT_TRUE(u8_type.has_value());

  EXPECT_THAT(
      EnumerateDomainForOneParam(&import_data, tm.type_info, "x", **u8_type,
                                 empty_def->value(), /*max_points=*/10),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("empty domain")));
}

TEST(DomainUtilsTest, EnumerateDomainForOneParamUnsupportedValueFails) {
  constexpr std::string_view kProgram = R"(
const SCALAR = u8:5;

fn target(x: u8) -> u8 { x }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(ConstantDef * scalar_def,
                           tm.module->GetConstantDef("SCALAR"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn,
                           tm.module->GetMemberOrError<Function>("target"));
  std::optional<Type*> u8_type = tm.type_info->GetItem(fn->params()[0]);
  ASSERT_TRUE(u8_type.has_value());

  EXPECT_THAT(EnumerateDomainForOneParam(&import_data, tm.type_info, "x",
                                         **u8_type, scalar_def->value(),
                                         /*max_points=*/10),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Unsupported domain value")));
}

TEST(DomainUtilsTest, EnumerateFunctionDomainsWithExplicitDomains) {
  constexpr std::string_view kProgram = R"(
#[fuzz_test(domains=`u32:10..=12, [true]`)]
fn prop(x: u32, flag: bool) -> bool {
    flag && x >= u32:10
}
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      FuzzTestFunction * ftf,
      tm.module->GetMemberOrError<FuzzTestFunction>("prop"));
  ASSERT_TRUE(ftf->domains().has_value());
  const Function& fn = ftf->fn();
  std::optional<Type*> fn_type_opt = tm.type_info->GetItem(&fn);
  ASSERT_TRUE(fn_type_opt.has_value());
  const FunctionType& fn_type = (*fn_type_opt)->AsFunction();

  EXPECT_THAT(
      EnumerateFunctionDomains(&import_data, tm.type_info, fn, fn_type,
                               ftf->domains(), /*max_points=*/10),
      IsOkAndHolds(ElementsAre(ElementsAre(InterpValue::MakeUBits(32, 10),
                                           InterpValue::MakeBool(true)),
                               ElementsAre(InterpValue::MakeUBits(32, 11),
                                           InterpValue::MakeBool(true)),
                               ElementsAre(InterpValue::MakeUBits(32, 12),
                                           InterpValue::MakeBool(true)))));
}

TEST(DomainUtilsTest, EnumerateFunctionDomainsWithoutExplicitDomains) {
  constexpr std::string_view kProgram = R"(
fn target(x: u1, flag: bool) -> bool { flag }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn,
                           tm.module->GetMemberOrError<Function>("target"));
  std::optional<Type*> fn_type_opt = tm.type_info->GetItem(fn);
  ASSERT_TRUE(fn_type_opt.has_value());
  const FunctionType& fn_type = (*fn_type_opt)->AsFunction();

  EXPECT_THAT(
      EnumerateFunctionDomains(&import_data, tm.type_info, *fn, fn_type,
                               /*domains_tuple=*/std::nullopt,
                               /*max_points=*/10),
      IsOkAndHolds(ElementsAre(ElementsAre(InterpValue::MakeBool(false),
                                           InterpValue::MakeBool(false)),
                               ElementsAre(InterpValue::MakeBool(false),
                                           InterpValue::MakeBool(true)),
                               ElementsAre(InterpValue::MakeBool(true),
                                           InterpValue::MakeBool(false)),
                               ElementsAre(InterpValue::MakeBool(true),
                                           InterpValue::MakeBool(true)))));
}

TEST(DomainUtilsTest, EnumerateFunctionDomainsWrongArityFails) {
  constexpr std::string_view kProgram = R"(
#[fuzz_test(domains=`u32:10..=12, [true]`)]
fn prop(x: u32, flag: bool) -> bool {
    flag && x >= u32:10
}

fn single_param(x: u32) -> bool { x >= u32:10 }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      FuzzTestFunction * ftf,
      tm.module->GetMemberOrError<FuzzTestFunction>("prop"));
  ASSERT_TRUE(ftf->domains().has_value());
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * single_param,
      tm.module->GetMemberOrError<Function>("single_param"));
  std::optional<Type*> fn_type_opt = tm.type_info->GetItem(single_param);
  ASSERT_TRUE(fn_type_opt.has_value());
  const FunctionType& fn_type = (*fn_type_opt)->AsFunction();

  EXPECT_THAT(EnumerateFunctionDomains(&import_data, tm.type_info,
                                       *single_param, fn_type, ftf->domains(),
                                       /*max_points=*/10),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Domain tuple has 2 arguments, but function "
                                 "`single_param` has 1 parameter")));
}

TEST(DomainUtilsTest,
     EnumerateFunctionDomainsSingleParamExceedsMaxPointsFails) {
  constexpr std::string_view kProgram = R"(
#[fuzz_test(domains=`u32:10..=12, [true]`)]
fn prop(x: u32, flag: bool) -> bool {
    flag && x >= u32:10
}
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      FuzzTestFunction * ftf,
      tm.module->GetMemberOrError<FuzzTestFunction>("prop"));
  ASSERT_TRUE(ftf->domains().has_value());
  const Function& fn = ftf->fn();
  std::optional<Type*> fn_type_opt = tm.type_info->GetItem(&fn);
  ASSERT_TRUE(fn_type_opt.has_value());
  const FunctionType& fn_type = (*fn_type_opt)->AsFunction();

  EXPECT_THAT(EnumerateFunctionDomains(&import_data, tm.type_info, fn, fn_type,
                                       ftf->domains(), /*max_points=*/2),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("exceeds maximum size")));
}

TEST(DomainUtilsTest, EnumerateFunctionDomainsProductExceedsMaxPointsFails) {
  constexpr std::string_view kProgram = R"(
#[fuzz_test(domains=`u8:0..2, u8:0..3`)]
fn prop(x: u8, y: u8) -> bool { true }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      FuzzTestFunction * ftf,
      tm.module->GetMemberOrError<FuzzTestFunction>("prop"));
  ASSERT_TRUE(ftf->domains().has_value());
  const Function& fn = ftf->fn();
  std::optional<Type*> fn_type_opt = tm.type_info->GetItem(&fn);
  ASSERT_TRUE(fn_type_opt.has_value());
  const FunctionType& fn_type = (*fn_type_opt)->AsFunction();

  EXPECT_THAT(EnumerateFunctionDomains(&import_data, tm.type_info, fn, fn_type,
                                       ftf->domains(), /*max_points=*/5),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("exceeds maximum size")));
}

TEST(DomainUtilsTest, EnumerateArbitraryXnBitVector) {
  constexpr std::string_view kProgram = R"(
fn target(x: xN[false][2]) -> xN[false][2] { x }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn,
                           tm.module->GetMemberOrError<Function>("target"));
  std::optional<Type*> xn_type = tm.type_info->GetItem(fn->params()[0]);
  ASSERT_TRUE(xn_type.has_value());

  EXPECT_THAT(EnumerateArbitraryDomain("x", **xn_type, /*max_points=*/16,
                                       /*allow_composite=*/true),
              IsOkAndHolds(ElementsAre(
                  InterpValue::MakeUBits(2, 0), InterpValue::MakeUBits(2, 1),
                  InterpValue::MakeUBits(2, 2), InterpValue::MakeUBits(2, 3))));
}

TEST(DomainUtilsTest, EnumerateDomainForOneParamXnConstRange) {
  constexpr std::string_view kProgram = R"(
const R = xN[false][4]:1..xN[false][4]:3;
const R_REF = R;

fn target(x: xN[false][4]) -> xN[false][4] { x }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(ConstantDef * ref_def,
                           tm.module->GetConstantDef("R_REF"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn,
                           tm.module->GetMemberOrError<Function>("target"));
  std::optional<Type*> xn_type = tm.type_info->GetItem(fn->params()[0]);
  ASSERT_TRUE(xn_type.has_value());

  EXPECT_THAT(EnumerateDomainForOneParam(&import_data, tm.type_info, "x",
                                         **xn_type, ref_def->value(),
                                         /*max_points=*/10),
              IsOkAndHolds(ElementsAre(InterpValue::MakeUBits(4, 1),
                                       InterpValue::MakeUBits(4, 2))));
}

TEST(DomainUtilsTest, EnumerateDomainForOneParamXnTupleFails) {
  constexpr std::string_view kProgram = R"(
const T = (u4:0..2,);

fn target(x: xN[false][4]) -> xN[false][4] { x }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(ConstantDef * tuple_def,
                           tm.module->GetConstantDef("T"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn,
                           tm.module->GetMemberOrError<Function>("target"));
  std::optional<Type*> xn_type = tm.type_info->GetItem(fn->params()[0]);
  ASSERT_TRUE(xn_type.has_value());

  EXPECT_THAT(EnumerateDomainForOneParam(&import_data, tm.type_info, "x",
                                         **xn_type, tuple_def->value(),
                                         /*max_points=*/10),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Tuple domain expression")));
}

TEST(DomainUtilsTest, EnumerateDomainForOneParamStructForNonStructFails) {
  constexpr std::string_view kProgram = R"(
struct Pair {
    a: u2,
    b: bool,
}

#[fuzz_test(domains=`Pair { a: u2:0..2 }`)]
fn prop(p: Pair) -> bool { true }

fn scalar_fn(x: u8) -> u8 { x }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      FuzzTestFunction * ftf,
      tm.module->GetMemberOrError<FuzzTestFunction>("prop"));
  ASSERT_TRUE(ftf->domains().has_value());
  XLS_ASSERT_OK_AND_ASSIGN(Function * scalar_fn,
                           tm.module->GetMemberOrError<Function>("scalar_fn"));
  std::optional<Type*> u8_type = tm.type_info->GetItem(scalar_fn->params()[0]);
  ASSERT_TRUE(u8_type.has_value());

  EXPECT_THAT(
      EnumerateDomainForOneParam(&import_data, tm.type_info, "x", **u8_type,
                                 (*ftf->domains())->members()[0],
                                 /*max_points=*/10),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("cannot be used for non-struct")));
}

TEST(DomainUtilsTest, EnumerateDomainForOneParamInlineStructUnknownFieldFails) {
  constexpr std::string_view kProgram = R"(
struct Pair {
    a: u2,
    b: bool,
}

struct Single {
    a: u2,
}

#[fuzz_test(domains=`Pair { a: u2:0..2 }`)]
fn prop(p: Pair) -> bool { true }

fn single_fn(s: Single) -> u2 { s.a }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      FuzzTestFunction * ftf,
      tm.module->GetMemberOrError<FuzzTestFunction>("prop"));
  ASSERT_TRUE(ftf->domains().has_value());
  XLS_ASSERT_OK_AND_ASSIGN(Function * single_fn,
                           tm.module->GetMemberOrError<Function>("single_fn"));
  std::optional<Type*> single_type =
      tm.type_info->GetItem(single_fn->params()[0]);
  ASSERT_TRUE(single_type.has_value());

  EXPECT_THAT(
      EnumerateDomainForOneParam(&import_data, tm.type_info, "s", **single_type,
                                 (*ftf->domains())->members()[0],
                                 /*max_points=*/10),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("unknown field `b`")));
}

TEST(DomainUtilsTest, EnumerateDomainForOneParamTupleSizeMismatchFails) {
  constexpr std::string_view kProgram = R"(
#[fuzz_test(domains=`(u8:1..3, [true])`)]
fn prop(t: (u8, bool)) -> bool { t.1 }

fn triple_fn(t: (u8, bool, u8)) -> u8 { t.0 }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      FuzzTestFunction * ftf,
      tm.module->GetMemberOrError<FuzzTestFunction>("prop"));
  ASSERT_TRUE(ftf->domains().has_value());
  XLS_ASSERT_OK_AND_ASSIGN(Function * triple_fn,
                           tm.module->GetMemberOrError<Function>("triple_fn"));
  std::optional<Type*> triple_type =
      tm.type_info->GetItem(triple_fn->params()[0]);
  ASSERT_TRUE(triple_type.has_value());

  EXPECT_THAT(
      EnumerateDomainForOneParam(&import_data, tm.type_info, "t", **triple_type,
                                 (*ftf->domains())->members()[0],
                                 /*max_points=*/10),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Domain tuple size (2) does not match")));
}

TEST(DomainUtilsTest, EnumerateDomainForOneParamOversizedArrayDomainFails) {
  constexpr std::string_view kProgram = R"(
#[fuzz_test(domains=`([u2:0], [u2:1], [u2:2])`)]
fn prop(a: u2[3]) -> bool { true }

fn small_arr_fn(a: u2[2]) -> u2 { a[0] }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      FuzzTestFunction * ftf,
      tm.module->GetMemberOrError<FuzzTestFunction>("prop"));
  ASSERT_TRUE(ftf->domains().has_value());
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * small_arr_fn,
      tm.module->GetMemberOrError<Function>("small_arr_fn"));
  std::optional<Type*> arr_type =
      tm.type_info->GetItem(small_arr_fn->params()[0]);
  ASSERT_TRUE(arr_type.has_value());

  EXPECT_THAT(
      EnumerateDomainForOneParam(&import_data, tm.type_info, "a", **arr_type,
                                 (*ftf->domains())->members()[0],
                                 /*max_points=*/10),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Domain array size (3) exceeds")));
}

TEST(DomainUtilsTest, EnumerateDomainForOneParamTupleForStructFails) {
  constexpr std::string_view kProgram = R"(
struct Pair {
    a: u2,
    b: bool,
}

#[fuzz_test(domains=`(u2:0..2, [true])`)]
fn prop(t: (u2, bool)) -> bool { t.1 }

fn struct_fn(p: Pair) -> bool { p.b }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      FuzzTestFunction * ftf,
      tm.module->GetMemberOrError<FuzzTestFunction>("prop"));
  ASSERT_TRUE(ftf->domains().has_value());
  XLS_ASSERT_OK_AND_ASSIGN(Function * struct_fn,
                           tm.module->GetMemberOrError<Function>("struct_fn"));
  std::optional<Type*> pair_type =
      tm.type_info->GetItem(struct_fn->params()[0]);
  ASSERT_TRUE(pair_type.has_value());

  EXPECT_THAT(
      EnumerateDomainForOneParam(&import_data, tm.type_info, "p", **pair_type,
                                 (*ftf->domains())->members()[0],
                                 /*max_points=*/10),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Tuple domain expression")));
}

TEST(DomainUtilsTest, EnumerateDomainForOneParamConstTupleForStructFails) {
  constexpr std::string_view kProgram = R"(
struct Pair {
    a: u2,
    b: bool,
}

const T = (u2:0..2, [true]);
const T_REF = T;

fn struct_fn(p: Pair) -> bool { p.b }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(ConstantDef * ref_def,
                           tm.module->GetConstantDef("T_REF"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * struct_fn,
                           tm.module->GetMemberOrError<Function>("struct_fn"));
  std::optional<Type*> pair_type =
      tm.type_info->GetItem(struct_fn->params()[0]);
  ASSERT_TRUE(pair_type.has_value());

  EXPECT_THAT(EnumerateDomainForOneParam(&import_data, tm.type_info, "p",
                                         **pair_type, ref_def->value(),
                                         /*max_points=*/10),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("cannot be used for struct")));
}

TEST(DomainUtilsTest, EnumerateDomainForOneParamConstStructForTupleFails) {
  constexpr std::string_view kProgram = R"(
#[fuzz_domain("PairDomain")]
struct Pair {
    a: u2,
    b: bool,
}

fn make_domain() -> PairDomain {
    PairDomain { a: u2:0..2 }
}

#[fuzz_test(domains=`make_domain()`)]
fn prop(p: Pair) -> bool { true }

fn tuple_fn(t: (u2, bool)) -> bool { t.1 }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      FuzzTestFunction * ftf,
      tm.module->GetMemberOrError<FuzzTestFunction>("prop"));
  ASSERT_TRUE(ftf->domains().has_value());
  XLS_ASSERT_OK_AND_ASSIGN(Function * tuple_fn,
                           tm.module->GetMemberOrError<Function>("tuple_fn"));
  std::optional<Type*> tuple_type =
      tm.type_info->GetItem(tuple_fn->params()[0]);
  ASSERT_TRUE(tuple_type.has_value());

  EXPECT_THAT(
      EnumerateDomainForOneParam(&import_data, tm.type_info, "t", **tuple_type,
                                 (*ftf->domains())->members()[0],
                                 /*max_points=*/10),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("cannot be used for non-struct")));
}

TEST(DomainUtilsTest, EnumerateDomainForOneParamConstStructWrongSizeFails) {
  constexpr std::string_view kProgram = R"(
#[fuzz_domain("PairDomain")]
struct Pair {
    a: u2,
    b: bool,
}

struct Single {
    a: u2,
}

fn make_domain() -> PairDomain {
    PairDomain { a: u2:0..2 }
}

#[fuzz_test(domains=`make_domain()`)]
fn prop(p: Pair) -> bool { true }

fn single_fn(s: Single) -> u2 { s.a }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      FuzzTestFunction * ftf,
      tm.module->GetMemberOrError<FuzzTestFunction>("prop"));
  ASSERT_TRUE(ftf->domains().has_value());
  XLS_ASSERT_OK_AND_ASSIGN(Function * single_fn,
                           tm.module->GetMemberOrError<Function>("single_fn"));
  std::optional<Type*> single_type =
      tm.type_info->GetItem(single_fn->params()[0]);
  ASSERT_TRUE(single_type.has_value());

  EXPECT_THAT(
      EnumerateDomainForOneParam(&import_data, tm.type_info, "s", **single_type,
                                 (*ftf->domains())->members()[0],
                                 /*max_points=*/10),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Domain struct size (2) does not match")));
}

TEST(DomainUtilsTest, EnumerateDomainForOneParamConstNonTupleForStructFails) {
  constexpr std::string_view kProgram = R"(
struct Pair {
    a: u2,
    b: bool,
}

const R = u8:0..2;
const R_REF = R;

fn struct_fn(p: Pair) -> bool { p.b }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(ConstantDef * ref_def,
                           tm.module->GetConstantDef("R_REF"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * struct_fn,
                           tm.module->GetMemberOrError<Function>("struct_fn"));
  std::optional<Type*> pair_type =
      tm.type_info->GetItem(struct_fn->params()[0]);
  ASSERT_TRUE(pair_type.has_value());

  EXPECT_THAT(EnumerateDomainForOneParam(&import_data, tm.type_info, "p",
                                         **pair_type, ref_def->value(),
                                         /*max_points=*/10),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("expects a struct")));
}

TEST(DomainUtilsTest, EnumerateDomainForOneParamConstTupleWrongSizeFails) {
  constexpr std::string_view kProgram = R"(
const T = (u8:0..2, [true]);
const T_REF = T;

fn triple_fn(t: (u8, bool, u8)) -> u8 { t.0 }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(ConstantDef * ref_def,
                           tm.module->GetConstantDef("T_REF"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * triple_fn,
                           tm.module->GetMemberOrError<Function>("triple_fn"));
  std::optional<Type*> triple_type =
      tm.type_info->GetItem(triple_fn->params()[0]);
  ASSERT_TRUE(triple_type.has_value());

  EXPECT_THAT(EnumerateDomainForOneParam(&import_data, tm.type_info, "t",
                                         **triple_type, ref_def->value(),
                                         /*max_points=*/10),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Domain tuple size (2) does not match")));
}

TEST(DomainUtilsTest, EnumerateDomainForOneParamConstNonTupleForTupleFails) {
  constexpr std::string_view kProgram = R"(
const R = u8:0..2;
const R_REF = R;

fn tuple_fn(t: (u8, bool)) -> bool { t.1 }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(ConstantDef * ref_def,
                           tm.module->GetConstantDef("R_REF"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * tuple_fn,
                           tm.module->GetMemberOrError<Function>("tuple_fn"));
  std::optional<Type*> tuple_type =
      tm.type_info->GetItem(tuple_fn->params()[0]);
  ASSERT_TRUE(tuple_type.has_value());

  EXPECT_THAT(EnumerateDomainForOneParam(&import_data, tm.type_info, "t",
                                         **tuple_type, ref_def->value(),
                                         /*max_points=*/10),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("expects a tuple")));
}

TEST(DomainUtilsTest,
     EnumerateDomainForOneParamConstOversizedArrayDomainFails) {
  constexpr std::string_view kProgram = R"(
const A = ([u2:0], [u2:1], [u2:2]);
const A_REF = A;

fn small_arr_fn(a: u2[2]) -> u2 { a[0] }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(ConstantDef * ref_def,
                           tm.module->GetConstantDef("A_REF"));
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * small_arr_fn,
      tm.module->GetMemberOrError<Function>("small_arr_fn"));
  std::optional<Type*> arr_type =
      tm.type_info->GetItem(small_arr_fn->params()[0]);
  ASSERT_TRUE(arr_type.has_value());

  EXPECT_THAT(EnumerateDomainForOneParam(&import_data, tm.type_info, "a",
                                         **arr_type, ref_def->value(),
                                         /*max_points=*/10),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Domain array size (3) exceeds")));
}

TEST(DomainUtilsTest, EnumerateDomainForOneParamConstRangeForArrayFails) {
  constexpr std::string_view kProgram = R"(
const R = u2:0..2;
const R_REF = R;

fn arr_fn(a: u2[2]) -> u2 { a[0] }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(ConstantDef * ref_def,
                           tm.module->GetConstantDef("R_REF"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * arr_fn,
                           tm.module->GetMemberOrError<Function>("arr_fn"));
  std::optional<Type*> arr_type = tm.type_info->GetItem(arr_fn->params()[0]);
  ASSERT_TRUE(arr_type.has_value());

  EXPECT_THAT(EnumerateDomainForOneParam(&import_data, tm.type_info, "a",
                                         **arr_type, ref_def->value(),
                                         /*max_points=*/10),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected array or tuple of domains")));
}

TEST(DomainUtilsTest, EnumerateDomainForOneParamRangeBitWidthMismatchFails) {
  constexpr std::string_view kProgram = R"(
#[fuzz_test(domains=`u16:0..3`)]
fn prop(x: u16) -> bool { true }

fn narrow_fn(x: u8) -> u8 { x }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      FuzzTestFunction * ftf,
      tm.module->GetMemberOrError<FuzzTestFunction>("prop"));
  ASSERT_TRUE(ftf->domains().has_value());
  XLS_ASSERT_OK_AND_ASSIGN(Function * narrow_fn,
                           tm.module->GetMemberOrError<Function>("narrow_fn"));
  std::optional<Type*> u8_type = tm.type_info->GetItem(narrow_fn->params()[0]);
  ASSERT_TRUE(u8_type.has_value());

  EXPECT_THAT(
      EnumerateDomainForOneParam(&import_data, tm.type_info, "x", **u8_type,
                                 (*ftf->domains())->members()[0],
                                 /*max_points=*/10),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Domain bit count (16) does not match")));
}

TEST(DomainUtilsTest, EnumerateDomainForOneParamRangeSignednessMismatchFails) {
  constexpr std::string_view kProgram = R"(
#[fuzz_test(domains=`s4:0..2`)]
fn prop(x: s4) -> bool { true }

fn unsigned_fn(x: u4) -> u4 { x }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      FuzzTestFunction * ftf,
      tm.module->GetMemberOrError<FuzzTestFunction>("prop"));
  ASSERT_TRUE(ftf->domains().has_value());
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * unsigned_fn,
      tm.module->GetMemberOrError<Function>("unsigned_fn"));
  std::optional<Type*> u4_type =
      tm.type_info->GetItem(unsigned_fn->params()[0]);
  ASSERT_TRUE(u4_type.has_value());

  EXPECT_THAT(
      EnumerateDomainForOneParam(&import_data, tm.type_info, "x", **u4_type,
                                 (*ftf->domains())->members()[0],
                                 /*max_points=*/10),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Domain signedness (signed) does not match")));
}

TEST(DomainUtilsTest, EnumerateDomainForOneParamElementOfEnum) {
  constexpr std::string_view kProgram = R"(
enum Color : u2 {
    RED = 0,
    GREEN = 1,
    BLUE = 2,
}

#[fuzz_test(domains=`[Color::RED, Color::BLUE]`)]
fn prop(c: Color) -> bool { true }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      FuzzTestFunction * ftf,
      tm.module->GetMemberOrError<FuzzTestFunction>("prop"));
  ASSERT_TRUE(ftf->domains().has_value());
  XLS_ASSERT_OK_AND_ASSIGN(EnumDef * color_def,
                           tm.module->GetMemberOrError<EnumDef>("Color"));
  std::optional<Type*> fn_type_opt = tm.type_info->GetItem(&ftf->fn());
  ASSERT_TRUE(fn_type_opt.has_value());
  const FunctionType& fn_type = (*fn_type_opt)->AsFunction();

  EXPECT_THAT(
      EnumerateDomainForOneParam(&import_data, tm.type_info, "c",
                                 *fn_type.params()[0],
                                 (*ftf->domains())->members()[0],
                                 /*max_points=*/10),
      IsOkAndHolds(ElementsAre(
          InterpValue::MakeEnum(UBits(0, 2), /*is_signed=*/false, color_def),
          InterpValue::MakeEnum(UBits(2, 2), /*is_signed=*/false, color_def))));
}

TEST(DomainUtilsTest, EnumerateDomainForOneParamElementOfNonBitsForBitsFails) {
  constexpr std::string_view kProgram = R"(
enum Color : u2 {
    RED = 0,
    GREEN = 1,
}

#[fuzz_test(domains=`[Color::RED]`)]
fn prop(c: Color) -> bool { true }

fn bits_fn(x: u2) -> u2 { x }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      FuzzTestFunction * ftf,
      tm.module->GetMemberOrError<FuzzTestFunction>("prop"));
  ASSERT_TRUE(ftf->domains().has_value());
  XLS_ASSERT_OK_AND_ASSIGN(Function * bits_fn,
                           tm.module->GetMemberOrError<Function>("bits_fn"));
  std::optional<Type*> u2_type = tm.type_info->GetItem(bits_fn->params()[0]);
  ASSERT_TRUE(u2_type.has_value());

  EXPECT_THAT(
      EnumerateDomainForOneParam(&import_data, tm.type_info, "x", **u2_type,
                                 (*ftf->domains())->members()[0],
                                 /*max_points=*/10),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Expected bits domain value")));
}

TEST(DomainUtilsTest, EnumerateDomainForOneParamElementOfNonEnumForEnumFails) {
  constexpr std::string_view kProgram = R"(
enum Color : u2 {
    RED = 0,
    GREEN = 1,
}

#[fuzz_test(domains=`[u2:0]`)]
fn prop(x: u2) -> bool { true }

fn enum_fn(c: Color) -> Color { c }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      FuzzTestFunction * ftf,
      tm.module->GetMemberOrError<FuzzTestFunction>("prop"));
  ASSERT_TRUE(ftf->domains().has_value());
  XLS_ASSERT_OK_AND_ASSIGN(Function * enum_fn,
                           tm.module->GetMemberOrError<Function>("enum_fn"));
  std::optional<Type*> enum_type = tm.type_info->GetItem(enum_fn->params()[0]);
  ASSERT_TRUE(enum_type.has_value());

  EXPECT_THAT(
      EnumerateDomainForOneParam(&import_data, tm.type_info, "c", **enum_type,
                                 (*ftf->domains())->members()[0],
                                 /*max_points=*/10),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Expected enum domain value")));
}

TEST(DomainUtilsTest, EnumerateDomainForOneParamElementOfWrongEnumFails) {
  constexpr std::string_view kProgram = R"(
enum Color : u2 {
    RED = 0,
    GREEN = 1,
}

enum OtherColor : u2 {
    A = 0,
    B = 1,
}

#[fuzz_test(domains=`[OtherColor::A]`)]
fn prop(c: OtherColor) -> bool { true }

fn color_fn(c: Color) -> Color { c }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      FuzzTestFunction * ftf,
      tm.module->GetMemberOrError<FuzzTestFunction>("prop"));
  ASSERT_TRUE(ftf->domains().has_value());
  XLS_ASSERT_OK_AND_ASSIGN(Function * color_fn,
                           tm.module->GetMemberOrError<Function>("color_fn"));
  std::optional<Type*> color_type =
      tm.type_info->GetItem(color_fn->params()[0]);
  ASSERT_TRUE(color_type.has_value());

  EXPECT_THAT(
      EnumerateDomainForOneParam(&import_data, tm.type_info, "c", **color_type,
                                 (*ftf->domains())->members()[0],
                                 /*max_points=*/10),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Enum domain type mismatch")));
}

TEST(DomainUtilsTest, EnumerateDomainForOneParamScalarDomainForTokenFails) {
  constexpr std::string_view kProgram = R"(
#[fuzz_test(domains=`[u8:0]`)]
fn prop(x: u8) -> bool { true }

fn token_fn(t: token) -> token { t }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      FuzzTestFunction * ftf,
      tm.module->GetMemberOrError<FuzzTestFunction>("prop"));
  ASSERT_TRUE(ftf->domains().has_value());
  XLS_ASSERT_OK_AND_ASSIGN(Function * token_fn,
                           tm.module->GetMemberOrError<Function>("token_fn"));
  std::optional<Type*> token_type =
      tm.type_info->GetItem(token_fn->params()[0]);
  ASSERT_TRUE(token_type.has_value());

  EXPECT_THAT(
      EnumerateDomainForOneParam(&import_data, tm.type_info, "t", **token_type,
                                 (*ftf->domains())->members()[0],
                                 /*max_points=*/10),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Unsupported parameter type")));
}

TEST(DomainUtilsTest, EnumerateArbitraryZeroBitArrayExceedsMaxPointsFails) {
  constexpr std::string_view kProgram = R"(
fn target(a: uN[0][100]) -> uN[0] { a[0] }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn,
                           tm.module->GetMemberOrError<Function>("target"));
  std::optional<Type*> arr_type = tm.type_info->GetItem(fn->params()[0]);
  ASSERT_TRUE(arr_type.has_value());

  EXPECT_THAT(EnumerateArbitraryDomain("a", **arr_type, /*max_points=*/10,
                                       /*allow_composite=*/true),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("exceeds the maximum domain size")));
}

TEST(DomainUtilsTest,
     EnumerateDomainForOneParamArraySizeExceedsMaxPointsFails) {
  constexpr std::string_view kProgram = R"(
const T = ((),);

fn target(a: uN[0][100]) -> uN[0] { a[0] }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(ConstantDef * tuple_def,
                           tm.module->GetConstantDef("T"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn,
                           tm.module->GetMemberOrError<Function>("target"));
  std::optional<Type*> arr_type = tm.type_info->GetItem(fn->params()[0]);
  ASSERT_TRUE(arr_type.has_value());

  EXPECT_THAT(EnumerateDomainForOneParam(&import_data, tm.type_info, "a",
                                         **arr_type, tuple_def->value(),
                                         /*max_points=*/10),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("exceeds the maximum domain size")));
}

TEST(DomainUtilsTest,
     EnumerateDomainForOneParamConstArraySizeExceedsMaxPointsFails) {
  constexpr std::string_view kProgram = R"(
const A = ((),);
const A_REF = A;

fn target(a: uN[0][100]) -> uN[0] { a[0] }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(ConstantDef * ref_def,
                           tm.module->GetConstantDef("A_REF"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn,
                           tm.module->GetMemberOrError<Function>("target"));
  std::optional<Type*> arr_type = tm.type_info->GetItem(fn->params()[0]);
  ASSERT_TRUE(arr_type.has_value());

  EXPECT_THAT(EnumerateDomainForOneParam(&import_data, tm.type_info, "a",
                                         **arr_type, ref_def->value(),
                                         /*max_points=*/10),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("exceeds the maximum domain size")));
}

}  // namespace
}  // namespace xls::dslx
