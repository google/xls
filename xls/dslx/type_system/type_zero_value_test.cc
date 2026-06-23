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

#include "xls/dslx/type_system/type_zero_value.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"

namespace xls::dslx {
namespace {
SumType MakeOuterSumWithInhabitedNestedSumPayload(Module& module) {
  const Span kFakeSpan = Span::Fake();

  auto* inner_name = module.Make<NameDef>(kFakeSpan, "Inner", nullptr);
  auto* inner_unit_name = module.Make<NameDef>(kFakeSpan, "InnerUnit", nullptr);
  auto* inner_unit = module.Make<SumVariant>(
      kFakeSpan, inner_unit_name, SumVariant::PayloadShape::kUnit,
      std::vector<TypeAnnotation*>{}, std::vector<StructMemberNode*>{});
  auto* inner_def = module.Make<SumDef>(
      kFakeSpan, inner_name, std::vector<ParametricBinding*>{},
      std::vector<SumVariant*>{inner_unit}, /*is_public=*/false);
  inner_name->set_definer(inner_def);
  std::vector<SumTypeVariant> inner_variants;
  inner_variants.push_back(SumTypeVariant::MakeUnit(*inner_unit));
  SumType inner_type(*inner_def, std::move(inner_variants));

  auto* outer_name = module.Make<NameDef>(kFakeSpan, "Outer", nullptr);
  auto* wrapped_name = module.Make<NameDef>(kFakeSpan, "Wrapped", nullptr);
  auto* wrapped = module.Make<SumVariant>(
      kFakeSpan, wrapped_name, SumVariant::PayloadShape::kTuple,
      std::vector<TypeAnnotation*>{module.Make<TypeRefTypeAnnotation>(
          kFakeSpan, module.Make<TypeRef>(kFakeSpan, inner_def),
          std::vector<ExprOrType>{})},
      std::vector<StructMemberNode*>{});
  auto* outer_def = module.Make<SumDef>(
      kFakeSpan, outer_name, std::vector<ParametricBinding*>{},
      std::vector<SumVariant*>{wrapped}, /*is_public=*/false);
  outer_name->set_definer(outer_def);
  std::vector<std::unique_ptr<Type>> wrapped_members;
  wrapped_members.push_back(inner_type.CloneToUnique());
  std::vector<SumTypeVariant> outer_variants;
  outer_variants.push_back(
      SumTypeVariant::MakeTuple(*wrapped, std::move(wrapped_members)));
  return SumType(*outer_def, std::move(outer_variants));
}

TEST(TypeZeroValueTest, ConstructsNestedSumPayloadZeroValue) {
  FileTable file_table;
  Module module("test", /*fs_path=*/std::nullopt, file_table);
  SumType outer_type = MakeOuterSumWithInhabitedNestedSumPayload(module);
  ImportData import_data = CreateImportDataForTest();

  XLS_ASSERT_OK_AND_ASSIGN(
      InterpValue result, MakeZeroValue(outer_type, import_data, Span::Fake()));
  const std::vector<InterpValue>& outer = result.GetValuesOrDie();
  ASSERT_EQ(outer.size(), 2);
  EXPECT_TRUE(outer.at(0).IsUBits());
  EXPECT_TRUE(outer.at(0).GetBitsOrDie().IsZero());

  const std::vector<InterpValue>& payload = outer.at(1).GetValuesOrDie();
  ASSERT_EQ(payload.size(), 1);
  const std::vector<InterpValue>& inner = payload.at(0).GetValuesOrDie();
  ASSERT_EQ(inner.size(), 2);
  EXPECT_TRUE(inner.at(0).IsUBits());
  EXPECT_TRUE(inner.at(0).GetBitsOrDie().IsZero());
  EXPECT_TRUE(inner.at(1).GetValuesOrDie().empty());
}

TEST(TypeZeroValueTest, ConstructsDeeplyNestedSumZerosWithoutRevalidation) {
  constexpr int kNestingDepth = 32;
  const Span span = Span::Fake();
  FileTable file_table;
  Module module("test", /*fs_path=*/std::nullopt, file_table);
  std::unique_ptr<SumType> current;
  SumDef* previous_definition = nullptr;

  for (int depth = 0; depth < kNestingDepth; ++depth) {
    auto* sum_name =
        module.Make<NameDef>(span, "Nested" + std::to_string(depth), nullptr);
    auto* variant_name = module.Make<NameDef>(span, "Zero", nullptr);
    std::vector<TypeAnnotation*> payload_annotations;
    std::vector<std::unique_ptr<Type>> payload_members;
    if (previous_definition != nullptr) {
      payload_annotations.push_back(module.Make<TypeRefTypeAnnotation>(
          span, module.Make<TypeRef>(span, previous_definition),
          std::vector<ExprOrType>{}));
      payload_members.push_back(current->CloneToUnique());
    }
    auto* variant = module.Make<SumVariant>(
        span, variant_name,
        previous_definition == nullptr ? SumVariant::PayloadShape::kUnit
                                       : SumVariant::PayloadShape::kTuple,
        payload_annotations, std::vector<StructMemberNode*>{});
    auto* definition = module.Make<SumDef>(
        span, sum_name, std::vector<ParametricBinding*>{},
        std::vector<SumVariant*>{variant}, /*is_public=*/false);
    sum_name->set_definer(definition);
    std::vector<SumTypeVariant> variants;
    if (previous_definition == nullptr) {
      variants.push_back(SumTypeVariant::MakeUnit(*variant));
    } else {
      variants.push_back(
          SumTypeVariant::MakeTuple(*variant, std::move(payload_members)));
    }
    current = std::make_unique<SumType>(*definition, std::move(variants));
    previous_definition = definition;
  }

  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue result,
                           MakeZeroValue(*current, import_data, span));
  const InterpValue* value = &result;
  for (int depth = 0; depth < kNestingDepth; ++depth) {
    const std::vector<InterpValue>& encoded = value->GetValuesOrDie();
    ASSERT_EQ(encoded.size(), 2);
    EXPECT_TRUE(encoded.at(0).GetBitsOrDie().IsZero());
    const std::vector<InterpValue>& payload = encoded.at(1).GetValuesOrDie();
    if (depth + 1 == kNestingDepth) {
      EXPECT_TRUE(payload.empty());
    } else {
      ASSERT_EQ(payload.size(), 1);
      value = &payload.at(0);
    }
  }
}

TEST(TypeZeroValueTest, UsesExplicitZeroDiscriminantInsteadOfDenseStorageTag) {
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck("fn id(x: u32) -> u32 { x }", "fake.x", "fake",
                        &import_data, nullptr));
  const Span span = tm.module->span();
  auto* tag_type = tm.module->Make<BuiltinTypeAnnotation>(
      span, BuiltinType::kU2,
      tm.module->GetOrCreateBuiltinNameDef(BuiltinType::kU2));
  auto* u8 = tm.module->Make<BuiltinTypeAnnotation>(
      span, BuiltinType::kU8,
      tm.module->GetOrCreateBuiltinNameDef(BuiltinType::kU8));
  auto* u16 = tm.module->Make<BuiltinTypeAnnotation>(
      span, BuiltinType::kU16,
      tm.module->GetOrCreateBuiltinNameDef(BuiltinType::kU16));
  auto* request_discriminant =
      tm.module->Make<Number>(span, "1", NumberKind::kOther, tag_type);
  auto* idle_discriminant =
      tm.module->Make<Number>(span, "0", NumberKind::kOther, tag_type);
  tm.type_info->NoteConstExpr(request_discriminant,
                              InterpValue::MakeUBits(2, 1));
  tm.type_info->NoteConstExpr(idle_discriminant, InterpValue::MakeUBits(2, 0));

  auto* sum_name = tm.module->Make<NameDef>(span, "Message", nullptr);
  auto* request_name = tm.module->Make<NameDef>(span, "Request", nullptr);
  auto* idle_name = tm.module->Make<NameDef>(span, "Idle", nullptr);
  auto* request = tm.module->Make<SumVariant>(
      span, request_name, SumVariant::PayloadShape::kTuple,
      std::vector<TypeAnnotation*>{u8}, std::vector<StructMemberNode*>{},
      request_discriminant);
  auto* idle = tm.module->Make<SumVariant>(
      span, idle_name, SumVariant::PayloadShape::kTuple,
      std::vector<TypeAnnotation*>{u16}, std::vector<StructMemberNode*>{},
      idle_discriminant);
  auto* sum_def = tm.module->Make<SumDef>(
      span, sum_name, std::vector<ParametricBinding*>{},
      std::vector<SumVariant*>{request, idle}, /*is_public=*/false, tag_type);
  sum_name->set_definer(sum_def);
  XLS_ASSERT_OK(tm.module->AddTop(sum_def, /*make_collision_error=*/nullptr));

  std::vector<SumTypeVariant> variants;
  std::vector<std::unique_ptr<Type>> request_members;
  request_members.push_back(BitsType::MakeU8());
  variants.push_back(
      SumTypeVariant::MakeTuple(*request, std::move(request_members)));
  std::vector<std::unique_ptr<Type>> idle_members;
  idle_members.push_back(std::make_unique<BitsType>(false, 16));
  variants.push_back(SumTypeVariant::MakeTuple(*idle, std::move(idle_members)));
  SumType sum_type(*sum_def, std::move(variants));

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue result,
                           MakeZeroValue(sum_type, import_data, span));
  const std::vector<InterpValue>& encoded = result.GetValuesOrDie();
  ASSERT_EQ(encoded.size(), 2);
  EXPECT_EQ(encoded.at(0).GetBitValueUnsigned().value(), 1);
  const std::vector<InterpValue>& payload = encoded.at(1).GetValuesOrDie();
  ASSERT_EQ(payload.size(), 2);
  EXPECT_EQ(payload.at(0).GetBitCount().value(), 8);
  EXPECT_EQ(payload.at(1).GetBitCount().value(), 16);
  EXPECT_TRUE(payload.at(0).GetBitsOrDie().IsZero());
  EXPECT_TRUE(payload.at(1).GetBitsOrDie().IsZero());
  EXPECT_FALSE(MakeAllOnesValue(sum_type, import_data, span).ok());

  tm.type_info->NoteConstExpr(idle_discriminant, InterpValue::MakeUBits(2, 2));
  EXPECT_FALSE(MakeZeroValue(sum_type, import_data, span).ok());
}

TEST(TypeZeroValueTest, RejectsAnEmptySumWithoutAZeroVariant) {
  const Span span = Span::Fake();
  FileTable file_table;
  Module module("test", /*fs_path=*/std::nullopt, file_table);
  auto* name = module.Make<NameDef>(span, "Never", nullptr);
  auto* def =
      module.Make<SumDef>(span, name, std::vector<ParametricBinding*>{},
                          std::vector<SumVariant*>{}, /*is_public=*/false);
  name->set_definer(def);
  SumType never(*def, std::vector<SumTypeVariant>{});
  ImportData import_data = CreateImportDataForTest();

  EXPECT_FALSE(MakeZeroValue(never, import_data, span).ok());
}

TEST(TypeZeroValueTest, EmptyArraysDoNotMaterializeUninhabitedSumElements) {
  const Span span = Span::Fake();
  FileTable file_table;
  Module module("test", /*fs_path=*/std::nullopt, file_table);
  auto* never_name = module.Make<NameDef>(span, "Never", nullptr);
  auto* never_def =
      module.Make<SumDef>(span, never_name, std::vector<ParametricBinding*>{},
                          std::vector<SumVariant*>{}, /*is_public=*/false);
  never_name->set_definer(never_def);
  SumType never(*never_def, std::vector<SumTypeVariant>{});
  ArrayType empty(never.CloneToUnique(), TypeDim::CreateU32(0));
  ImportData import_data = CreateImportDataForTest();

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue zero,
                           MakeZeroValue(empty, import_data, span));
  EXPECT_TRUE(zero.GetValuesOrDie().empty());
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue ones,
                           MakeAllOnesValue(empty, import_data, span));
  EXPECT_TRUE(ones.GetValuesOrDie().empty());
}

}  // namespace
}  // namespace xls::dslx
