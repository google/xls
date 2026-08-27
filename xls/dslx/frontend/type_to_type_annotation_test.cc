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

#include "xls/dslx/frontend/type_to_type_annotation.h"

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/channel_direction.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/type_system/type.h"

namespace xls::dslx {
namespace {

using ::absl_testing::IsOkAndHolds;

struct TypeAndModule {
  const Type* type;
  TypecheckedModule tm;
};

absl::StatusOr<TypeAndModule> ExtractReturnType(std::string_view program,
                                                std::string_view fn_name,
                                                ImportData& import_data) {
  XLS_ASSIGN_OR_RETURN(
      TypecheckedModule tm,
      ParseAndTypecheck(program, "test.x", "test", &import_data));
  XLS_ASSIGN_OR_RETURN(Function * f,
                       tm.module->GetMemberOrError<Function>(fn_name));
  XLS_ASSIGN_OR_RETURN(Type * f_type, tm.type_info->GetItemOrError(f));
  FunctionType* func_type = dynamic_cast<FunctionType*>(f_type);
  XLS_RET_CHECK(func_type != nullptr);
  return TypeAndModule{&func_type->return_type(), std::move(tm)};
}

TEST(TypeToTypeAnnotationTest, BitsTypeUnsigned) {
  auto import_data = CreateImportDataForTest();
  constexpr std::string_view kProgram = R"(
fn f() -> u32 {
    u32:42
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(TypeAndModule tam,
                           ExtractReturnType(kProgram, "f", import_data));

  XLS_ASSERT_OK_AND_ASSIGN(
      TypeAnnotation * ta,
      CreateTypeAnnotation(*tam.tm.module, *tam.type, tam.tm.module->span()));

  auto* builtin_ta = dynamic_cast<BuiltinTypeAnnotation*>(ta);
  ASSERT_NE(builtin_ta, nullptr);
  EXPECT_EQ(builtin_ta->builtin_type(), BuiltinType::kU32);
  EXPECT_EQ(builtin_ta->GetBitCount(), 32);
  EXPECT_THAT(builtin_ta->GetSignedness(), IsOkAndHolds(false));
}

TEST(TypeToTypeAnnotationTest, BitsTypeSigned) {
  auto import_data = CreateImportDataForTest();
  constexpr std::string_view kProgram = R"(
fn f() -> s16 {
    s16:42
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(TypeAndModule tam,
                           ExtractReturnType(kProgram, "f", import_data));

  XLS_ASSERT_OK_AND_ASSIGN(
      TypeAnnotation * ta,
      CreateTypeAnnotation(*tam.tm.module, *tam.type, tam.tm.module->span()));

  auto* builtin_ta = dynamic_cast<BuiltinTypeAnnotation*>(ta);
  ASSERT_NE(builtin_ta, nullptr);
  EXPECT_EQ(builtin_ta->builtin_type(), BuiltinType::kS16);
  EXPECT_EQ(builtin_ta->GetBitCount(), 16);
  EXPECT_THAT(builtin_ta->GetSignedness(), IsOkAndHolds(true));
}

TEST(TypeToTypeAnnotationTest, TokenType) {
  auto import_data = CreateImportDataForTest();
  constexpr std::string_view kProgram = R"(
fn f(t: token) -> token {
    t
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(TypeAndModule tam,
                           ExtractReturnType(kProgram, "f", import_data));

  XLS_ASSERT_OK_AND_ASSIGN(
      TypeAnnotation * ta,
      CreateTypeAnnotation(*tam.tm.module, *tam.type, tam.tm.module->span()));

  auto* builtin_ta = dynamic_cast<BuiltinTypeAnnotation*>(ta);
  ASSERT_NE(builtin_ta, nullptr);
  EXPECT_EQ(builtin_ta->builtin_type(), BuiltinType::kToken);
}

TEST(TypeToTypeAnnotationTest, ChannelType) {
  auto import_data = CreateImportDataForTest();
  constexpr std::string_view kProgram = R"(
fn f(c: chan<u32> in) -> chan<u32> in {
    c
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(TypeAndModule tam,
                           ExtractReturnType(kProgram, "f", import_data));

  XLS_ASSERT_OK_AND_ASSIGN(
      TypeAnnotation * ta,
      CreateTypeAnnotation(*tam.tm.module, *tam.type, tam.tm.module->span()));

  auto* chan_ta = dynamic_cast<ChannelTypeAnnotation*>(ta);
  ASSERT_NE(chan_ta, nullptr);
  EXPECT_EQ(chan_ta->direction(), ChannelDirection::kIn);

  TypeAnnotation* payload = chan_ta->payload();
  ASSERT_NE(payload, nullptr);
  auto* builtin_payload = dynamic_cast<BuiltinTypeAnnotation*>(payload);
  ASSERT_NE(builtin_payload, nullptr);
  EXPECT_EQ(builtin_payload->builtin_type(), BuiltinType::kU32);
}

TEST(TypeToTypeAnnotationTest, TupleType) {
  auto import_data = CreateImportDataForTest();
  constexpr std::string_view kProgram = R"(
fn f() -> (u32, bool, s8) {
    (u32:0, false, s8:0)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(TypeAndModule tam,
                           ExtractReturnType(kProgram, "f", import_data));

  XLS_ASSERT_OK_AND_ASSIGN(
      TypeAnnotation * ta,
      CreateTypeAnnotation(*tam.tm.module, *tam.type, tam.tm.module->span()));

  auto* tuple_ta = dynamic_cast<TupleTypeAnnotation*>(ta);
  ASSERT_NE(tuple_ta, nullptr);
  ASSERT_EQ(tuple_ta->members().size(), 3);

  EXPECT_EQ(tuple_ta->members()[0]->ToString(), "u32");
  EXPECT_EQ(tuple_ta->members()[1]->ToString(), "bool");
  EXPECT_EQ(tuple_ta->members()[2]->ToString(), "s8");
}

TEST(TypeToTypeAnnotationTest, ArrayType) {
  auto import_data = CreateImportDataForTest();
  constexpr std::string_view kProgram = R"(
fn f() -> u32[3] {
    u32[3]:[u32:1, u32:2, u32:3]
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(TypeAndModule tam,
                           ExtractReturnType(kProgram, "f", import_data));

  XLS_ASSERT_OK_AND_ASSIGN(
      TypeAnnotation * ta,
      CreateTypeAnnotation(*tam.tm.module, *tam.type, tam.tm.module->span()));

  auto* array_ta = dynamic_cast<ArrayTypeAnnotation*>(ta);
  ASSERT_NE(array_ta, nullptr);

  Expr* dim = array_ta->dim();
  ASSERT_NE(dim, nullptr);
  auto* dim_num = dynamic_cast<Number*>(dim);
  ASSERT_NE(dim_num, nullptr);
  EXPECT_THAT(dim_num->GetAsUint64(*tam.tm.module->file_table()),
              IsOkAndHolds(3));
}

TEST(TypeToTypeAnnotationTest, StructType) {
  auto import_data = CreateImportDataForTest();
  constexpr std::string_view kProgram = R"(
struct S {
    x: u32,
    y: bool,
}
fn f() -> S {
    S { x: u32:0, y: false }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(TypeAndModule tam,
                           ExtractReturnType(kProgram, "f", import_data));

  XLS_ASSERT_OK_AND_ASSIGN(
      TypeAnnotation * ta,
      CreateTypeAnnotation(*tam.tm.module, *tam.type, tam.tm.module->span()));

  EXPECT_EQ(ta->ToString(), "S");

  auto* ref_ta = dynamic_cast<TypeRefTypeAnnotation*>(ta);
  ASSERT_NE(ref_ta, nullptr);

  TypeRef* tr = ref_ta->type_ref();
  ASSERT_NE(tr, nullptr);

  TypeDefinition td = tr->type_definition();
  EXPECT_TRUE(std::holds_alternative<StructDef*>(td));
  StructDef* sd = std::get<StructDef*>(td);
  EXPECT_EQ(sd->identifier(), "S");
}

TEST(TypeToTypeAnnotationTest, EnumType) {
  auto import_data = CreateImportDataForTest();
  constexpr std::string_view kProgram = R"(
enum E : u2 {
    A = 0,
    B = 1,
}
fn f() -> E {
    E::A
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(TypeAndModule tam,
                           ExtractReturnType(kProgram, "f", import_data));

  XLS_ASSERT_OK_AND_ASSIGN(
      TypeAnnotation * ta,
      CreateTypeAnnotation(*tam.tm.module, *tam.type, tam.tm.module->span()));

  EXPECT_EQ(ta->ToString(), "E");

  auto* ref_ta = dynamic_cast<TypeRefTypeAnnotation*>(ta);
  ASSERT_NE(ref_ta, nullptr);

  TypeRef* tr = ref_ta->type_ref();
  ASSERT_NE(tr, nullptr);

  TypeDefinition td = tr->type_definition();
  EXPECT_TRUE(std::holds_alternative<EnumDef*>(td));
  EnumDef* ed = std::get<EnumDef*>(td);
  EXPECT_EQ(ed->identifier(), "E");
}

TEST(TypeToTypeAnnotationTest, ImportedStructType) {
  auto import_data = CreateImportDataForTest();
  constexpr std::string_view kImported = R"(
pub struct ImportedStruct {
    a: u32
}
)";
  constexpr std::string_view kProgram = R"(
import fake_import;
fn f() -> fake_import::ImportedStruct {
    fake_import::ImportedStruct { a: u32:42 }
}
)";
  XLS_ASSERT_OK(
      ParseAndTypecheck(kImported, "fake_import.x", "fake_import", &import_data)
          .status());

  XLS_ASSERT_OK_AND_ASSIGN(TypeAndModule tam,
                           ExtractReturnType(kProgram, "f", import_data));

  Module new_module("new_module", /*fs_path=*/std::nullopt,
                    *tam.tm.module->file_table());

  NameDef* name_def = new_module.Make<NameDef>(
      tam.tm.module->span(), "fake_import", /*definer=*/nullptr);
  Import* new_import = new_module.Make<Import>(
      tam.tm.module->span(), std::vector<std::string>{"fake_import"}, *name_def,
      /*alias=*/std::nullopt);
  name_def->set_definer(new_import);
  XLS_ASSERT_OK(
      new_module.AddTop(new_import, /*make_collision_error=*/nullptr));

  XLS_ASSERT_OK_AND_ASSIGN(
      TypeAnnotation * ta,
      CreateTypeAnnotation(new_module, *tam.type, tam.tm.module->span(),
                           tam.tm.module, tam.tm.type_info));

  EXPECT_EQ(ta->ToString(), "fake_import::ImportedStruct");

  auto* ref_ta = dynamic_cast<TypeRefTypeAnnotation*>(ta);
  ASSERT_NE(ref_ta, nullptr);

  TypeRef* tr = ref_ta->type_ref();
  ASSERT_NE(tr, nullptr);

  TypeDefinition td = tr->type_definition();
  EXPECT_TRUE(std::holds_alternative<ColonRef*>(td));
  ColonRef* cr = std::get<ColonRef*>(td);
  EXPECT_EQ(cr->ToString(), "fake_import::ImportedStruct");
}

TEST(TypeToTypeAnnotationTest, ImportedEnumType) {
  auto import_data = CreateImportDataForTest();
  constexpr std::string_view kImported = R"(
pub enum ImportedEnum : u2 {
    A = 0,
    B = 1,
}
)";
  constexpr std::string_view kProgram = R"(
import fake_import;
fn f() -> fake_import::ImportedEnum {
    fake_import::ImportedEnum::A
}
)";
  XLS_ASSERT_OK(
      ParseAndTypecheck(kImported, "fake_import.x", "fake_import", &import_data)
          .status());

  XLS_ASSERT_OK_AND_ASSIGN(TypeAndModule tam,
                           ExtractReturnType(kProgram, "f", import_data));

  Module new_module("new_module", /*fs_path=*/std::nullopt,
                    *tam.tm.module->file_table());

  NameDef* name_def = new_module.Make<NameDef>(
      tam.tm.module->span(), "fake_import", /*definer=*/nullptr);
  Import* new_import = new_module.Make<Import>(
      tam.tm.module->span(), std::vector<std::string>{"fake_import"}, *name_def,
      /*alias=*/std::nullopt);
  name_def->set_definer(new_import);
  XLS_ASSERT_OK(
      new_module.AddTop(new_import, /*make_collision_error=*/nullptr));

  XLS_ASSERT_OK_AND_ASSIGN(
      TypeAnnotation * ta,
      CreateTypeAnnotation(new_module, *tam.type, tam.tm.module->span(),
                           tam.tm.module, tam.tm.type_info));

  EXPECT_EQ(ta->ToString(), "fake_import::ImportedEnum");

  auto* ref_ta = dynamic_cast<TypeRefTypeAnnotation*>(ta);
  ASSERT_NE(ref_ta, nullptr);

  TypeRef* tr = ref_ta->type_ref();
  ASSERT_NE(tr, nullptr);

  TypeDefinition td = tr->type_definition();
  EXPECT_TRUE(std::holds_alternative<ColonRef*>(td));
  ColonRef* cr = std::get<ColonRef*>(td);
  EXPECT_EQ(cr->ToString(), "fake_import::ImportedEnum");
}

}  // namespace
}  // namespace xls::dslx
