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
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xls/common/attribute_data.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/frontend/ast_test_utils.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"

namespace xls::dslx {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

class AstTest : public testing::Test {
 public:
  FileTable file_table;
  Module m{"test", /*fs_path=*/std::nullopt, file_table};
  const Span fake_span;
};

TEST_F(AstTest, ModuleWithConstant) {
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

TEST_F(AstTest, ModuleWithStructAndImpl) {
  const std::vector<ParametricBinding*> empty_binding;

  // One member var: `member_var: bool`.
  BuiltinType bool_type = BuiltinTypeFromString("bool").value();
  TypeAnnotation* bool_type_annot = m.Make<BuiltinTypeAnnotation>(
      fake_span, bool_type, m.GetOrCreateBuiltinNameDef(bool_type));
  NameDef* field_name_def =
      m.Make<NameDef>(fake_span, std::string("member_var"), nullptr);
  std::vector<StructMemberNode*> members{m.Make<StructMemberNode>(
      fake_span, field_name_def, fake_span, bool_type_annot)};

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
  Impl* impl = m.Make<Impl>(fake_span, type_annot, std::vector<ImplMember>{},
                            /*is_public=*/false);
  XLS_ASSERT_OK(m.AddTop(impl, /*make_collision_error=*/nullptr));

  constexpr std::string_view kExpected = R"(struct MyStruct {
    member_var: bool,
}
impl MyStruct {
})";

  EXPECT_EQ(m.ToString(), kExpected);
}

TEST_F(AstTest, GetNumberAsInt64) {
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
  auto make_num = [this](std::string text) {
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

TEST_F(AstTest, CharacterNumberToStringTest) {
  struct Example {
    std::string text;
    std::string want;
  } kCases[] = {
      {.text = R"(4)", .want = R"('4')"},  {.text = R"(2)", .want = R"('2')"},
      {.text = R"(X)", .want = R"('X')"},  {.text = R"(l)", .want = R"('l')"},
      {.text = R"(S)", .want = R"('S')"},  {.text = R"(")", .want = R"('"')"},
      {.text = R"(')", .want = R"('\'')"}, {.text = R"(\)", .want = R"('\\')"},
  };
  auto make_char_num = [this](std::string text) {
    return m.Make<Number>(fake_span, text, NumberKind::kCharacter,
                          /*type=*/nullptr);
  };
  for (const Example& example : kCases) {
    EXPECT_THAT(make_char_num(example.text)->ToString(), example.want);
  }
}

TEST_F(AstTest, GetBuiltinTypeSignedness) {
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

TEST_F(AstTest, GetBuiltinTypeBitCount) {
  int64_t bit_count = GetBuiltinTypeBitCount(BuiltinType::kBool);
  EXPECT_EQ(bit_count, 1);
  bit_count = GetBuiltinTypeBitCount(BuiltinType::kS1);
  EXPECT_EQ(bit_count, 1);
  bit_count = GetBuiltinTypeBitCount(BuiltinType::kS64);
  EXPECT_EQ(bit_count, 64);
  bit_count = GetBuiltinTypeBitCount(BuiltinType::kU1);
  EXPECT_EQ(bit_count, 1);
  bit_count = GetBuiltinTypeBitCount(BuiltinType::kU64);
  EXPECT_EQ(bit_count, 64);
  bit_count = GetBuiltinTypeBitCount(BuiltinType::kSN);
  EXPECT_EQ(bit_count, 0);
  bit_count = GetBuiltinTypeBitCount(BuiltinType::kUN);
  EXPECT_EQ(bit_count, 0);
  bit_count = GetBuiltinTypeBitCount(BuiltinType::kBits);
  EXPECT_EQ(bit_count, 0);
  bit_count = GetBuiltinTypeBitCount(BuiltinType::kToken);
  EXPECT_EQ(bit_count, 0);
}

// See comment on `MakeCastWithinLtComparison()` -- we need to insert parens
// appropriately here.
TEST_F(AstTest, ToStringCastWithinLtComparison) {
  auto [file_table, module, lt] = MakeCastWithinLtComparison();

  EXPECT_EQ(lt->ToString(), "(x as t) < x");
}

TEST_F(AstTest, GetFuncParam) {
  // Create an empty function
  //  fn f(p: u32) -> u32 {}

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

  Function* f = m.Make<Function>(fake_span, func_name_def, parametric_bindings,
                                 params, u32_type_annotation, block,
                                 FunctionTag::kNormal, false, false);

  XLS_ASSERT_OK_AND_ASSIGN(Param * found_param, f->GetParamByName("p"));
  EXPECT_EQ(found_param, params[0]);

  EXPECT_THAT(f->GetParamByName("not_a_param"),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("Param 'not_a_param' not a parameter")));
}

TEST_F(AstTest, IsConstantNameRef) {
  NameDef* name_def =
      m.Make<NameDef>(fake_span, std::string("MyStruct"), nullptr);
  NameRef* name_ref = m.Make<NameRef>(fake_span, "name", name_def);

  EXPECT_FALSE(IsConstant(name_ref));
}

TEST_F(AstTest, IsConstantBinopOfNameRefs) {
  NameDef* name_def =
      m.Make<NameDef>(fake_span, std::string("MyStruct"), nullptr);
  NameRef* left = m.Make<NameRef>(fake_span, "name", name_def);
  NameRef* right = m.Make<NameRef>(fake_span, "name", name_def);
  Binop* binop =
      m.Make<Binop>(fake_span, BinopKind::kAdd, left, right, fake_span);

  EXPECT_FALSE(IsConstant(binop));
}

TEST_F(AstTest, IsConstantBinopOfNumbers) {
  Number* left = m.Make<Number>(fake_span, std::string("41"),
                                NumberKind::kOther, /*type=*/nullptr);
  Number* right = m.Make<Number>(fake_span, std::string("42"),
                                 NumberKind::kOther, /*type=*/nullptr);
  Binop* binop =
      m.Make<Binop>(fake_span, BinopKind::kAdd, left, right, fake_span);

  EXPECT_TRUE(IsConstant(binop));
}

TEST_F(AstTest, IsConstantUnopOfNameRefs) {
  NameDef* name_def =
      m.Make<NameDef>(fake_span, std::string("MyStruct"), nullptr);
  NameRef* operand = m.Make<NameRef>(fake_span, "name", name_def);
  Unop* unop = m.Make<Unop>(fake_span, UnopKind::kNegate, operand, fake_span);

  EXPECT_FALSE(IsConstant(unop));
}

TEST_F(AstTest, IsConstantUnopOfNumbers) {
  Number* operand = m.Make<Number>(fake_span, std::string("41"),
                                   NumberKind::kOther, /*type=*/nullptr);
  Unop* unop = m.Make<Unop>(fake_span, UnopKind::kNegate, operand, fake_span);

  EXPECT_TRUE(IsConstant(unop));
}

TEST_F(AstTest, IsConstantNumber) {
  Number* number = m.Make<Number>(fake_span, std::string("42"),
                                  NumberKind::kOther, /*type=*/nullptr);

  EXPECT_TRUE(IsConstant(number));
}

// Tests the IsConstant predicate on an array with a single `name` reference.
//
// Since the name reference is not constant, the array is not constant.
TEST_F(AstTest, IsConstantArrayOfNameRefs) {
  NameDef* name_def =
      m.Make<NameDef>(fake_span, std::string("MyStruct"), nullptr);
  NameRef* operand = m.Make<NameRef>(fake_span, "name", name_def);
  Array* array = m.Make<Array>(fake_span, std::vector<Expr*>{operand}, false);

  EXPECT_FALSE(IsConstant(array));
}

// Tests the IsConstant predicate on an array with a single `number` literal.
// Since the number literal is constant, the array is constant.
TEST_F(AstTest, IsConstantArrayOfNumbers) {
  Number* operand = m.Make<Number>(fake_span, std::string("41"),
                                   NumberKind::kOther, /*type=*/nullptr);
  Array* array = m.Make<Array>(fake_span, std::vector<Expr*>{operand}, false);

  EXPECT_TRUE(IsConstant(array));
}

// Tests the IsConstant predicate on an empty array.
// Since there are no members, the array is constant.
TEST_F(AstTest, IsConstantEmptyArray) {
  Array* array = m.Make<Array>(fake_span, std::vector<Expr*>{}, false);

  EXPECT_TRUE(IsConstant(array));
}

TEST_F(AstTest, AttributeToStringFuzzTestNoArgs) {
  Attribute* attr = m.Make<Attribute>(
      fake_span, std::nullopt, AttributeData(AttributeKind::kFuzzTest, {}));
  EXPECT_EQ(attr->ToString(), "#[fuzz_test]");
}

TEST_F(AstTest, AttributeToStringFuzzTestSingleArg) {
  Attribute* attr =
      m.Make<Attribute>(fake_span, fake_span,
                        AttributeData(AttributeKind::kFuzzTest, {"u32:0..1"}));
  EXPECT_EQ(attr->ToString(), "#[fuzz_test(`u32:0..1`)]");
}

TEST_F(AstTest, AttributeToStringFuzzTestMultipleArgs) {
  Attribute* attr = m.Make<Attribute>(
      fake_span, fake_span,
      AttributeData(AttributeKind::kFuzzTest, {"u32:0..1", "u32:10..20"}));
  EXPECT_EQ(attr->ToString(), "#[fuzz_test(`u32:0..1`, `u32:10..20`)]");
}

TEST_F(AstTest, AttributeToStringGenericArg) {
  Attribute* attr =
      m.Make<Attribute>(fake_span, fake_span,
                        AttributeData(AttributeKind::kTest, {"some_string"}));
  EXPECT_EQ(attr->ToString(), "#[test(some_string)]");
}

TEST_F(AstTest, AttributeToStringQuickcheckNoArgs) {
  Attribute* attr = m.Make<Attribute>(
      fake_span, std::nullopt, AttributeData(AttributeKind::kQuickcheck, {}));
  EXPECT_EQ(attr->ToString(), "#[quickcheck]");
}

TEST_F(AstTest, AttributeToStringQuickcheckExhaustive) {
  Attribute* attr = m.Make<Attribute>(
      fake_span, fake_span,
      AttributeData(AttributeKind::kQuickcheck, {std::string("exhaustive")}));
  EXPECT_EQ(attr->ToString(), "#[quickcheck(exhaustive)]");
}

TEST_F(AstTest, AttributeToStringQuickcheckTestCount) {
  AttributeData::IntKeyValueArgument arg("test_count", 1000);
  Attribute* attr = m.Make<Attribute>(
      fake_span, fake_span, AttributeData(AttributeKind::kQuickcheck, {arg}));
  EXPECT_EQ(attr->ToString(), "#[quickcheck(test_count = 1000)]");
}

TEST_F(AstTest, AttributeToStringExternVerilogStringLiteral) {
  AttributeData::StringLiteralArgument arg{"my_module"};
  Attribute* attr =
      m.Make<Attribute>(fake_span, fake_span,
                        AttributeData(AttributeKind::kExternVerilog, {arg}));
  EXPECT_EQ(attr->ToString(), "#[extern_verilog(\"my_module\")]");
}

}  // namespace
}  // namespace xls::dslx
