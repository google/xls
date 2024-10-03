// Copyright 2020 The XLS Authors
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

#include "xls/dslx/frontend/ast.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/frontend/ast_test_utils.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"

namespace xls::dslx {
namespace {

using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using testing::HasSubstr;

TEST(AstTest, ModuleWithConstant) {
  FileTable file_table;
  Module m("test", /*fs_path=*/std::nullopt, file_table);
  const Span fake_span;
  Number* number = m.Make<Number>(fake_span, std::string("42"),
                                  NumberKind::kOther, /*type=*/nullptr);
  NameDef* name_def = m.Make<NameDef>(fake_span, std::string("MOL"), nullptr);
  ConstantDef* constant_def =
      m.Make<ConstantDef>(fake_span, name_def, /*type_annotation=*/nullptr,
                          number, /*is_public=*/false);
  name_def->set_definer(constant_def);
  XLS_ASSERT_OK(m.AddTop(constant_def, /*make_collision_error=*/nullptr));

  EXPECT_EQ(m.ToString(), "const MOL = 42;");
}

TEST(AstTest, ModuleWithStructAndImpl) {
  FileTable file_table;
  Module m("test", /*fs_path=*/std::nullopt, file_table);
  const Span fake_span;
  const std::vector<ParametricBinding*> empty_binding;

  // One member var: `member_var: bool`.
  BuiltinType bool_type = BuiltinTypeFromString("bool").value();
  TypeAnnotation* bool_type_annot = m.Make<BuiltinTypeAnnotation>(
      Span::Fake(), bool_type, m.GetOrCreateBuiltinNameDef(bool_type));
  std::vector<StructMember> members{
      StructMember(fake_span, "member_var", bool_type_annot)};

  // Struct definition.
  NameDef* name_def =
      m.Make<NameDef>(fake_span, std::string("MyStruct"), nullptr);
  StructDef* struct_def =
      m.Make<StructDef>(fake_span, name_def, empty_binding, members, false);
  name_def->set_definer(struct_def);
  XLS_ASSERT_OK(m.AddTop(struct_def, /*make_collision_error=*/nullptr));

  // Impl definition.
  TypeRef* type_ref = m.Make<TypeRef>(fake_span, TypeDefinition(struct_def));
  TypeAnnotation* type_annot = m.Make<TypeRefTypeAnnotation>(
      fake_span, type_ref, /*parametrics=*/std::vector<ExprOrType>{});
  Impl* impl = m.Make<Impl>(fake_span, type_annot, /*is_public=*/false);
  XLS_ASSERT_OK(m.AddTop(impl, /*make_collision_error=*/nullptr));

  constexpr std::string_view kExpected = R"(struct MyStruct {
    member_var: bool,
}
impl MyStruct {})";

  EXPECT_EQ(m.ToString(), kExpected);
}

TEST(AstTest, GetNumberAsInt64) {
  struct Example {
    std::string text;
    uint64_t want;
  } kCases[] = {
      {.text = "0b0", .want = 0},
      {.text = "0b1", .want = 1},
      {.text = "0b10", .want = 2},
      {.text = "0b11", .want = 3},
      {.text = "0b100", .want = 4},
      {.text = "0b1000", .want = 8},
      {.text = "0b1011", .want = 11},
      {.text = "0b1_1000", .want = 24},
      {.text = "0b1_1001", .want = 25},
      {.text = "0b1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_"
               "1111_"
               "1111_1111_1111",
       .want = static_cast<uint64_t>(-1)},
      {.text = "-1", .want = static_cast<uint64_t>(-1)},
  };
  FileTable file_table;
  Module m("test", /*fs_path=*/std::nullopt, file_table);
  auto make_num = [&m](std::string text) {
    const Span fake_span;
    return m.Make<Number>(fake_span, text, NumberKind::kOther,
                          /*type=*/nullptr);
  };
  for (const Example& example : kCases) {
    EXPECT_THAT(make_num(example.text)->GetAsUint64(file_table),
                IsOkAndHolds(example.want));
  }

  EXPECT_THAT(make_num("0b")->GetAsUint64(file_table),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Could not convert 0b to a number")));
}

TEST(AstTest, CharacterNumberToStringTest) {
  struct Example {
    std::string text;
    std::string want;
  } kCases[] = {
      {.text = R"(4)", .want = R"('4')"},  {.text = R"(2)", .want = R"('2')"},
      {.text = R"(X)", .want = R"('X')"},  {.text = R"(l)", .want = R"('l')"},
      {.text = R"(S)", .want = R"('S')"},  {.text = R"(")", .want = R"('"')"},
      {.text = R"(')", .want = R"('\'')"}, {.text = R"(\)", .want = R"('\\')"},
  };
  FileTable file_table;
  Module m("test", /*fs_path=*/std::nullopt, file_table);
  auto make_char_num = [&m](std::string text) {
    const Span fake_span;
    return m.Make<Number>(fake_span, text, NumberKind::kCharacter,
                          /*type=*/nullptr);
  };
  for (const Example& example : kCases) {
    EXPECT_THAT(make_char_num(example.text)->ToString(), example.want);
  }
}

TEST(AstTest, GetBuiltinTypeSignedness) {
  XLS_ASSERT_OK_AND_ASSIGN(bool is_signed,
                           GetBuiltinTypeSignedness(BuiltinType::kBool));
  EXPECT_FALSE(is_signed);
  XLS_ASSERT_OK_AND_ASSIGN(is_signed,
                           GetBuiltinTypeSignedness(BuiltinType::kS1));
  EXPECT_TRUE(is_signed);
  XLS_ASSERT_OK_AND_ASSIGN(is_signed,
                           GetBuiltinTypeSignedness(BuiltinType::kU1));
  EXPECT_FALSE(is_signed);
  XLS_ASSERT_OK_AND_ASSIGN(is_signed,
                           GetBuiltinTypeSignedness(BuiltinType::kSN));
  EXPECT_TRUE(is_signed);
  XLS_ASSERT_OK_AND_ASSIGN(is_signed,
                           GetBuiltinTypeSignedness(BuiltinType::kUN));
  EXPECT_FALSE(is_signed);
  XLS_ASSERT_OK_AND_ASSIGN(is_signed,
                           GetBuiltinTypeSignedness(BuiltinType::kBits));
  EXPECT_FALSE(is_signed);

  // Tokens don't have signedness.
  EXPECT_THAT(GetBuiltinTypeSignedness(BuiltinType::kToken),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Type \"token\" has no defined signedness.")));
}

TEST(AstTest, GetBuiltinTypeBitCount) {
  XLS_ASSERT_OK_AND_ASSIGN(int64_t bit_count,
                           GetBuiltinTypeBitCount(BuiltinType::kBool));
  EXPECT_EQ(bit_count, 1);
  XLS_ASSERT_OK_AND_ASSIGN(bit_count, GetBuiltinTypeBitCount(BuiltinType::kS1));
  EXPECT_EQ(bit_count, 1);
  XLS_ASSERT_OK_AND_ASSIGN(bit_count,
                           GetBuiltinTypeBitCount(BuiltinType::kS64));
  EXPECT_EQ(bit_count, 64);
  XLS_ASSERT_OK_AND_ASSIGN(bit_count, GetBuiltinTypeBitCount(BuiltinType::kU1));
  EXPECT_EQ(bit_count, 1);
  XLS_ASSERT_OK_AND_ASSIGN(bit_count,
                           GetBuiltinTypeBitCount(BuiltinType::kU64));
  EXPECT_EQ(bit_count, 64);
  XLS_ASSERT_OK_AND_ASSIGN(bit_count, GetBuiltinTypeBitCount(BuiltinType::kSN));
  EXPECT_EQ(bit_count, 0);
  XLS_ASSERT_OK_AND_ASSIGN(bit_count, GetBuiltinTypeBitCount(BuiltinType::kUN));
  EXPECT_EQ(bit_count, 0);
  XLS_ASSERT_OK_AND_ASSIGN(bit_count,
                           GetBuiltinTypeBitCount(BuiltinType::kBits));
  EXPECT_EQ(bit_count, 0);
  XLS_ASSERT_OK_AND_ASSIGN(bit_count,
                           GetBuiltinTypeBitCount(BuiltinType::kToken));
  EXPECT_EQ(bit_count, 0);
}

// See comment on `MakeCastWithinLtComparison()` -- we need to insert parens
// appropriately here.
TEST(AstTest, ToStringCastWithinLtComparison) {
  auto [file_table, module, lt] = MakeCastWithinLtComparison();

  EXPECT_EQ(lt->ToString(), "(x as t) < x");
}

TEST(AstTest, GetFuncParam) {
  // Create an empty function
  //  fn f(p: u32) -> u32 {}

  FileTable file_table;
  const Span fake_span;
  Module m("test", /*fs_path=*/std::nullopt, file_table);

  BuiltinNameDef* builtin_name_def = m.GetOrCreateBuiltinNameDef("u32");
  BuiltinTypeAnnotation* u32_type_annotation = m.Make<BuiltinTypeAnnotation>(
      fake_span, BuiltinType::kU32, builtin_name_def);

  NameDef* func_name_def =
      m.Make<NameDef>(fake_span, std::string("f"), nullptr);
  NameDef* param_name_def =
      m.Make<NameDef>(fake_span, std::string("p"), nullptr);

  std::vector<Param*> params;
  params.push_back(m.Make<Param>(param_name_def, u32_type_annotation));

  std::vector<ParametricBinding*> parametric_bindings;

  StatementBlock* block =
      m.Make<StatementBlock>(fake_span, std::vector<Statement*>{}, true);

  Function* f =
      m.Make<Function>(fake_span, func_name_def, parametric_bindings, params,
                       u32_type_annotation, block, FunctionTag::kNormal, false);

  XLS_ASSERT_OK_AND_ASSIGN(Param * found_param, f->GetParamByName("p"));
  EXPECT_EQ(found_param, params[0]);

  EXPECT_THAT(f->GetParamByName("not_a_param"),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("Param 'not_a_param' not a parameter")));
}

}  // namespace
}  // namespace xls::dslx
