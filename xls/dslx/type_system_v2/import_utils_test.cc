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

#include "xls/dslx/type_system_v2/import_utils.h"

#include <optional>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_cloner.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/type_system_v2/type_annotation_utils.h"

namespace xls::dslx {
namespace {

using ::testing::HasSubstr;

TEST(ImportUtilsTest, WarmConstructorRetainsAliasParametricExpressions) {
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      auto module, ParseModule(R"(
#![feature(type_inference_v2)]
enum E<N: u32> { Value(uN[N]) }
type A = E<u32:8>;
type B = A;
const X = B::Value(u8:0);
const Y = B::Value(u8:1);
)",
                               "test.x", "test", import_data.file_table()));
  XLS_ASSERT_OK_AND_ASSIGN(auto* alias,
                           module->GetMemberOrError<TypeAlias>("A"));
  const auto* annotation =
      dynamic_cast<const TypeRefTypeAnnotation*>(&alias->type_annotation());
  ASSERT_NE(annotation, nullptr);
  XLS_ASSERT_OK_AND_ASSIGN(auto* sum, module->GetMemberOrError<SumDef>("E"));
  for (const char* name : {"X", "X", "Y"}) {
    XLS_ASSERT_OK_AND_ASSIGN(auto* constant, module->GetConstantDef(name));
    const auto* invocation = dynamic_cast<const Invocation*>(constant->value());
    ASSERT_NE(invocation, nullptr);
    XLS_ASSERT_OK_AND_ASSIGN(auto resolved,
                             ResolveSumConstructor(invocation, import_data));
    ASSERT_TRUE(resolved.has_value());
    EXPECT_EQ(resolved->sum_ref.def, sum);
    EXPECT_EQ(resolved->variant, sum->GetVariant("Value").value());
    EXPECT_EQ(resolved->sum_ref.parametrics, annotation->parametrics());
  }
  EXPECT_FALSE(import_data.HasInferenceTable());
}

TEST(ImportUtilsTest, AliasCacheKeepsOuterAnnotationAndConstructorOrigin) {
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      auto module, ParseModule(R"(
#![feature(type_inference_v2)]
enum E<N: u32> { Value(uN[N]) }
type A = E;
type B = A;
const X = B::Value(u8:0);
const Y = B::Value(u16:0);
struct S<N: u32> { value: uN[N] }
type SA = S;
type SB = SA;
const SX = SB { value: u8:0 };
const SY = SB { value: u16:0 };
)",
                               "test.x", "test", import_data.file_table()));
  XLS_ASSERT_OK_AND_ASSIGN(auto* sum_alias,
                           module->GetMemberOrError<TypeAlias>("B"));
  XLS_ASSERT_OK_AND_ASSIGN(auto* struct_alias,
                           module->GetMemberOrError<TypeAlias>("SB"));
  for (const char* name : {"X", "Y"}) {
    XLS_ASSERT_OK_AND_ASSIGN(auto* constant, module->GetConstantDef(name));
    const auto* origin = dynamic_cast<const Invocation*>(constant->value());
    ASSERT_NE(origin, nullptr);
    auto* reference = module->Make<TypeRef>(origin->span(), sum_alias);
    auto* annotation = module->Make<TypeRefTypeAnnotation>(
        origin->span(), reference, std::vector<ExprOrType>{}, origin);
    XLS_ASSERT_OK_AND_ASSIGN(auto resolved, GetSumRef(annotation, import_data));
    ASSERT_TRUE(resolved.has_value());
    EXPECT_EQ(resolved->construction_origin, origin);
    const auto* reconstructed = CreateSumAnnotation(*module, *resolved)
                                    ->AsAnnotation<TypeRefTypeAnnotation>();
    EXPECT_EQ(reconstructed->construction_origin(), origin);
  }
  for (const char* name : {"SX", "SY"}) {
    XLS_ASSERT_OK_AND_ASSIGN(auto* constant, module->GetConstantDef(name));
    const auto* origin = dynamic_cast<const StructInstance*>(constant->value());
    ASSERT_NE(origin, nullptr);
    auto* reference = module->Make<TypeRef>(origin->span(), struct_alias);
    auto* annotation = module->Make<TypeRefTypeAnnotation>(
        origin->span(), reference, std::vector<ExprOrType>{}, origin);
    XLS_ASSERT_OK_AND_ASSIGN(auto resolved,
                             GetStructOrProcRef(annotation, import_data));
    ASSERT_TRUE(resolved.has_value());
    EXPECT_EQ(resolved->construction_origin, origin);
    EXPECT_EQ(resolved->type_ref_type_annotation, annotation);
    const auto* reconstructed = CreateStructOrProcAnnotation(*module, *resolved)
                                    ->AsAnnotation<TypeRefTypeAnnotation>();
    EXPECT_EQ(reconstructed->construction_origin(), origin);
  }
}

TEST(ImportUtilsTest,
     CloneResolvesSubstitutedParametricsAfterOriginalWasCached) {
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      auto module, ParseModule(R"(
#![feature(type_inference_v2)]
enum E<N: u32> { Value(uN[N]) }
const X = E<u32:8>::Value(u8:0);
const WIDTH = u32:16;
)",
                               "test.x", "test", import_data.file_table()));
  XLS_ASSERT_OK_AND_ASSIGN(auto* original, module->GetConstantDef("X"));
  XLS_ASSERT_OK_AND_ASSIGN(auto* width, module->GetConstantDef("WIDTH"));
  const auto* invocation = dynamic_cast<const Invocation*>(original->value());
  ASSERT_NE(invocation, nullptr);
  XLS_ASSERT_OK_AND_ASSIGN(auto resolved,
                           ResolveSumConstructor(invocation, import_data));
  ASSERT_TRUE(resolved.has_value());
  ASSERT_EQ(resolved->sum_ref.parametrics.size(), 1);
  Expr* original_width = std::get<Expr*>(resolved->sum_ref.parametrics[0]);
  XLS_ASSERT_OK_AND_ASSIGN(
      AstNode * cloned,
      CloneAst(invocation,
               [&](const AstNode* node, Module* target,
                   const absl::flat_hash_map<const AstNode*, AstNode*>& mapping)
                   -> std::optional<AstNode*> {
                 if (node == original_width) {
                   return width->value();
                 } else {
                   return PreserveTypeDefinitionsReplacer(node, target,
                                                          mapping);
                 }
               }));
  const auto* cloned_invocation = dynamic_cast<const Invocation*>(cloned);
  ASSERT_NE(cloned_invocation, nullptr);
  EXPECT_NE(cloned_invocation->callee(), invocation->callee());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto cloned_resolved,
      ResolveSumConstructor(cloned_invocation, import_data));
  ASSERT_TRUE(cloned_resolved.has_value());
  EXPECT_EQ(cloned_resolved->sum_ref.def, resolved->sum_ref.def);
  EXPECT_EQ(cloned_resolved->variant, resolved->variant);
  EXPECT_EQ(cloned_resolved->sum_ref.parametrics,
            std::vector<ExprOrType>{width->value()});
  EXPECT_EQ(resolved->sum_ref.parametrics,
            std::vector<ExprOrType>{original_width});
}

TEST(ImportUtilsTest, AliasSuffixPreservesFirstDuplicateParametricDiagnostic) {
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      auto module, ParseModule(R"(
#![feature(type_inference_v2)]
enum E<N: u32> { Value(uN[N]) }
type A = E<u32:8>;
type B = A<u32:16>;
const X = B<u32:32>::Value(u32:0);
)",
                               "test.x", "test", import_data.file_table()));
  XLS_ASSERT_OK_AND_ASSIGN(auto* constant, module->GetConstantDef("X"));
  const auto* invocation = dynamic_cast<const Invocation*>(constant->value());
  ASSERT_NE(invocation, nullptr);
  const auto first = ResolveSumConstructor(invocation, import_data);
  ASSERT_FALSE(first.ok());
  EXPECT_THAT(first.status().message(),
              HasSubstr("Parametric values defined multiple times for "
                        "annotation: `A<u32:16>`"));
  const auto repeated = ResolveSumConstructor(invocation, import_data);
  EXPECT_EQ(repeated.status(), first.status());
}

}  // namespace
}  // namespace xls::dslx
