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

#include "xls/dslx/type_system_v2/inference_table.h"

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/casts.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_cloner.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/parser.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/scanner.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system_v2/inference_table_converter.h"
#include "xls/dslx/type_system_v2/inference_table_to_type_info.h"
#include "xls/dslx/type_system_v2/type_annotation_utils.h"
#include "xls/dslx/type_system_v2/type_system_test_utils.h"
#include "xls/dslx/warning_collector.h"
#include "xls/dslx/warning_kind.h"

namespace xls::dslx {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

class InferenceTableTest : public ::testing::Test {
 public:
  void SetUp() override {
    import_data_.emplace(CreateImportDataForTest());
    warning_collector_.emplace(kAllWarningsSet);
    module_ =
        std::make_unique<Module>("test", /*fs_path=*/std::nullopt, file_table_);
    table_ = InferenceTable::Create(*module_);
  }

  absl::StatusOr<TypeInfo*> ConvertTableToTypeInfo() {
    XLS_ASSIGN_OR_RETURN(
        std::unique_ptr<InferenceTableConverter> converter,
        InferenceTableToTypeInfo(*table_, *module_, *import_data_,
                                 *warning_collector_, file_table_));
    return converter->GetBaseTypeInfo();
  }

  absl::StatusOr<std::string> ConvertTableToTypeInfoString() {
    XLS_ASSIGN_OR_RETURN(TypeInfo * ti, ConvertTableToTypeInfo());
    return TypeInfoToString(*ti, file_table_);
  }

  void ParseAndInitModuleAndTable(std::string_view program) {
    Scanner scanner(file_table_, file_table_.GetOrCreate("fake.x"),
                    std::string(program));
    Parser parser("fake", &scanner);
    XLS_ASSERT_OK_AND_ASSIGN(module_, parser.ParseModule());
    table_ = InferenceTable::Create(*module_);
  }

  FileTable file_table_;
  std::optional<ImportData> import_data_;
  std::optional<WarningCollector> warning_collector_;
  std::unique_ptr<Module> module_;
  std::unique_ptr<InferenceTable> table_;
};

TEST_F(InferenceTableTest, TypeInfoForEmptyTable) {
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           ConvertTableToTypeInfoString());
  EXPECT_EQ(type_info_string, "");
}

TEST_F(InferenceTableTest, AddOneSimpleAnnotation) {
  // Just one def for x:u32.
  NameDef* x = module_->Make<NameDef>(Span::Fake(), "x", /*definer=*/nullptr);
  TypeAnnotation* annotation = module_->Make<BuiltinTypeAnnotation>(
      Span::Fake(), BuiltinType::kU32,
      module_->GetOrCreateBuiltinNameDef("u32"));
  XLS_EXPECT_OK(table_->SetTypeAnnotation(x, annotation));
  EXPECT_EQ(table_->GetTypeAnnotation(x), annotation);
  EXPECT_EQ(table_->GetTypeVariable(x), std::nullopt);
}

TEST_F(InferenceTableTest, SetTypeVariableToNonInferenceVariable) {
  // Set type for `x` to a `NameRef` to `N` which is not an inference variable.
  // It should fail because it needs to be an inference variable.
  NameDef* x = module_->Make<NameDef>(Span::Fake(), "x", /*definer=*/nullptr);
  NameDef* n = module_->Make<NameDef>(Span::Fake(), "N", /*definer=*/nullptr);
  NameRef* n_ref = module_->Make<NameRef>(Span::Fake(), "N", n);
  EXPECT_THAT(table_->SetTypeVariable(x, n_ref),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(InferenceTableTest, SetTypeVariableToNonType) {
  // Set type for `x` to `N` which is an integer-kind inference variable. It
  // should fail because it needs to be type-kind.
  NameDef* x = module_->Make<NameDef>(Span::Fake(), "x", /*definer=*/nullptr);
  NameDef* n = module_->Make<NameDef>(Span::Fake(), "N", /*definer=*/nullptr);
  XLS_ASSERT_OK_AND_ASSIGN(
      const NameRef* n_var,
      table_->DefineInternalVariable(InferenceVariableKind::kInteger, n, "N"));
  EXPECT_THAT(table_->SetTypeVariable(x, n_var),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(InferenceTableTest, AddAnnotationsWithConflictingSignedness) {
  // Apply the same type variable to the LHS and RHS of an addition, then claim
  // that the LHS (x) is annotated as u32 and the RHS (y) is annotated as s32.
  // This should be allowed by the table, as it's caught at the conversion
  // stage.
  NameDef* x = module_->Make<NameDef>(Span::Fake(), "x", /*definer=*/nullptr);
  NameDef* y = module_->Make<NameDef>(Span::Fake(), "y", /*definer=*/nullptr);
  NameRef* x_ref = module_->Make<NameRef>(Span::Fake(), "x", x);
  NameRef* y_ref = module_->Make<NameRef>(Span::Fake(), "y", y);
  AstNode* add_node = module_->Make<Binop>(Span::Fake(), BinopKind::kAdd, x_ref,
                                           y_ref, Span::Fake());

  TypeAnnotation* u32_annotation = module_->Make<BuiltinTypeAnnotation>(
      Span::Fake(), BuiltinType::kU32,
      module_->GetOrCreateBuiltinNameDef("u32"));
  TypeAnnotation* s32_annotation = module_->Make<BuiltinTypeAnnotation>(
      Span::Fake(), BuiltinType::kS32,
      module_->GetOrCreateBuiltinNameDef("s32"));

  XLS_ASSERT_OK_AND_ASSIGN(const NameRef* t0,
                           table_->DefineInternalVariable(
                               InferenceVariableKind::kType, add_node, "T0"));
  XLS_EXPECT_OK(table_->SetTypeVariable(x_ref, t0));
  XLS_EXPECT_OK(table_->SetTypeVariable(y_ref, t0));
  XLS_EXPECT_OK(table_->SetTypeAnnotation(x_ref, u32_annotation));
  XLS_EXPECT_OK(table_->SetTypeAnnotation(y_ref, s32_annotation));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<const TypeAnnotation*> annotations,
      table_->GetTypeAnnotationsForTypeVariable(std::nullopt, t0));
  EXPECT_THAT(annotations, ElementsAre(u32_annotation, s32_annotation));
}

TEST_F(InferenceTableTest, AddAnnotationsWithSameSignedness) {
  // Apply the same type variable to the LHS and RHS of an addition, then claim
  // that the LHS (x) and the RHS (y) are both annotated as u32.
  NameDef* x = module_->Make<NameDef>(Span::Fake(), "x", /*definer=*/nullptr);
  NameDef* y = module_->Make<NameDef>(Span::Fake(), "y", /*definer=*/nullptr);
  NameRef* x_ref = module_->Make<NameRef>(Span::Fake(), "x", x);
  NameRef* y_ref = module_->Make<NameRef>(Span::Fake(), "y", y);
  AstNode* add_node = module_->Make<Binop>(Span::Fake(), BinopKind::kAdd, x_ref,
                                           y_ref, Span::Fake());

  TypeAnnotation* u32_annotation = module_->Make<BuiltinTypeAnnotation>(
      Span::Fake(), BuiltinType::kU32,
      module_->GetOrCreateBuiltinNameDef("u32"));

  XLS_ASSERT_OK_AND_ASSIGN(const NameRef* t0,
                           table_->DefineInternalVariable(
                               InferenceVariableKind::kType, add_node, "T0"));
  XLS_EXPECT_OK(table_->SetTypeVariable(x_ref, t0));
  XLS_EXPECT_OK(table_->SetTypeVariable(y_ref, t0));
  XLS_EXPECT_OK(table_->SetTypeAnnotation(x_ref, u32_annotation));
  XLS_EXPECT_OK(table_->SetTypeAnnotation(y_ref, u32_annotation));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<const TypeAnnotation*> annotations,
      table_->GetTypeAnnotationsForTypeVariable(std::nullopt, t0));
  EXPECT_THAT(annotations, ElementsAre(u32_annotation, u32_annotation));
}

TEST_F(InferenceTableTest, ParametricVariable) {
  ParseAndInitModuleAndTable(R"(
    fn foo<N: u32>(a: uN[N]) -> uN[N] { a }
    fn bar() {
      foo<u32:4>(u4:1);
      foo<u32:5>(u5:3);
    }
)");

  XLS_ASSERT_OK_AND_ASSIGN(const Function* foo,
                           module_->GetMemberOrError<Function>("foo"));
  ASSERT_EQ(foo->parametric_bindings().size(), 1);
  ASSERT_EQ(foo->params().size(), 1);
  XLS_ASSERT_OK(
      table_->DefineParametricVariable(*foo->parametric_bindings()[0]));
  for (const Param* param : foo->params()) {
    XLS_ASSERT_OK(table_->SetTypeAnnotation(param, param->type_annotation()));
  }
  const NameDef* n = foo->parametric_bindings()[0]->name_def();
  XLS_ASSERT_OK_AND_ASSIGN(const Function* bar,
                           module_->GetMemberOrError<Function>("bar"));
  ASSERT_EQ(bar->body()->statements().size(), 2);
  const Invocation* invocation1 = down_cast<const Invocation*>(
      ToAstNode(bar->body()->statements().at(0)->wrapped()));
  const Invocation* invocation2 = down_cast<const Invocation*>(
      ToAstNode(bar->body()->statements().at(1)->wrapped()));
  XLS_ASSERT_OK_AND_ASSIGN(
      const ParametricContext* parametric_context1,
      table_->AddParametricInvocation(*invocation1, *foo, bar,
                                      /*caller_invocation=*/std::nullopt,
                                      /*self_type=*/std::nullopt));
  XLS_ASSERT_OK_AND_ASSIGN(
      const ParametricContext* parametric_context2,
      table_->AddParametricInvocation(*invocation2, *foo, bar,
                                      /*caller_invocation=*/std::nullopt,
                                      /*self_type=*/std::nullopt));

  EXPECT_THAT(table_->GetParametricInvocations(),
              ElementsAre(parametric_context1, parametric_context2));

  std::optional<ParametricContextScopedExpr> parametric_inv1_n_value =
      table_->GetParametricValue(*n, *parametric_context1);
  std::optional<ParametricContextScopedExpr> parametric_inv2_n_value =
      table_->GetParametricValue(*n, *parametric_context2);
  ASSERT_TRUE(parametric_inv1_n_value.has_value());
  ASSERT_TRUE(parametric_inv2_n_value.has_value());
  // These exprs are scoped to `nullopt` invocation because they reside in the
  // non-parametric calling context.
  EXPECT_EQ(parametric_inv1_n_value->context(), std::nullopt);
  EXPECT_EQ(parametric_inv2_n_value->context(), std::nullopt);
  EXPECT_EQ(parametric_inv1_n_value->expr()->ToString(), "u32:4");
  EXPECT_EQ(parametric_inv2_n_value->expr()->ToString(), "u32:5");
}

TEST_F(InferenceTableTest, ParametricVariableWithDefault) {
  ParseAndInitModuleAndTable(R"(
    fn foo<M: u32 = {u32:4}, N: u32 = {M * M}>(a: uN[M], b: uN[N])
        -> uN[M + N] { a ++ b }

    fn bar() {
      // Use both defaults.
      foo(u4:1);
      // Use the default for N only.
      foo<u32:5>(u5:3, u25:4);
    }
)");

  XLS_ASSERT_OK_AND_ASSIGN(const Function* foo,
                           module_->GetMemberOrError<Function>("foo"));
  ASSERT_EQ(foo->parametric_bindings().size(), 2);
  const NameDef* m = foo->parametric_bindings()[0]->name_def();
  const NameDef* n = foo->parametric_bindings()[1]->name_def();
  ASSERT_EQ(foo->params().size(), 2);
  XLS_ASSERT_OK(
      table_->DefineParametricVariable(*foo->parametric_bindings()[0]));
  XLS_ASSERT_OK(
      table_->DefineParametricVariable(*foo->parametric_bindings()[1]));
  for (const Param* param : foo->params()) {
    XLS_ASSERT_OK(table_->SetTypeAnnotation(param, param->type_annotation()));
  }
  XLS_ASSERT_OK_AND_ASSIGN(const Function* bar,
                           module_->GetMemberOrError<Function>("bar"));
  ASSERT_EQ(bar->body()->statements().size(), 2);
  const Invocation* invocation1 = down_cast<const Invocation*>(
      ToAstNode(bar->body()->statements().at(0)->wrapped()));
  const Invocation* invocation2 = down_cast<const Invocation*>(
      ToAstNode(bar->body()->statements().at(1)->wrapped()));
  XLS_ASSERT_OK_AND_ASSIGN(
      const ParametricContext* parametric_context1,
      table_->AddParametricInvocation(*invocation1, *foo, bar,
                                      /*caller_invocation=*/std::nullopt,
                                      /*self_type=*/std::nullopt));
  XLS_ASSERT_OK_AND_ASSIGN(
      const ParametricContext* parametric_context2,
      table_->AddParametricInvocation(*invocation2, *foo, bar,
                                      /*caller_invocation=*/std::nullopt,
                                      /*self_type=*/std::nullopt));

  EXPECT_THAT(table_->GetParametricInvocations(),
              ElementsAre(parametric_context1, parametric_context2));

  std::optional<ParametricContextScopedExpr> parametric_inv1_m_value =
      table_->GetParametricValue(*m, *parametric_context1);
  std::optional<ParametricContextScopedExpr> parametric_inv1_n_value =
      table_->GetParametricValue(*n, *parametric_context1);
  std::optional<ParametricContextScopedExpr> parametric_inv2_m_value =
      table_->GetParametricValue(*m, *parametric_context2);
  std::optional<ParametricContextScopedExpr> parametric_inv2_n_value =
      table_->GetParametricValue(*n, *parametric_context2);
  ASSERT_TRUE(parametric_inv1_m_value.has_value());
  ASSERT_TRUE(parametric_inv1_n_value.has_value());
  ASSERT_TRUE(parametric_inv2_m_value.has_value());
  ASSERT_TRUE(parametric_inv2_n_value.has_value());

  // Exprs that reside in the callee are scoped to the callee invocation.
  EXPECT_EQ(parametric_inv1_m_value->context(), parametric_context1);
  EXPECT_EQ(parametric_inv2_m_value->context(), std::nullopt);
  EXPECT_EQ(parametric_inv1_n_value->context(), parametric_context1);
  EXPECT_EQ(parametric_inv2_n_value->context(), parametric_context2);
  EXPECT_EQ(parametric_inv1_m_value->expr()->ToString(), "u32:4");
  EXPECT_EQ(parametric_inv2_m_value->expr()->ToString(), "u32:5");
  EXPECT_EQ(parametric_inv1_n_value->expr()->ToString(), "M * M");
  EXPECT_EQ(parametric_inv2_n_value->expr()->ToString(), "M * M");

  EXPECT_THAT(
      table_->ToString(),
      AllOf(
          HasSubstr("Node: b: uN[N]"), HasSubstr("Annotation: uN[N]"),
          HasSubstr("Node: a: uN[M]"), HasSubstr("Annotation: uN[M]"),
          HasSubstr("Node: M"), HasSubstr("Annotation: u32"),
          HasSubstr("Node: N"), HasSubstr("Variable: N"),
          HasSubstr("Variable: M"), HasSubstr("Parametric contexts:"),
          HasSubstr("ParametricContext(id=0, parent_id=none, self_type=none, "
                    "node=foo(u4:1), data=(foo, caller: bar))"),
          HasSubstr("ParametricContext(id=1, parent_id=none, self_type=none, "
                    "node=foo<u32:5>(u5:3, u25:4), data=(foo, caller: bar)")));
}

TEST_F(InferenceTableTest, ParametricVariableWithArrayAnnotation) {
  ParseAndInitModuleAndTable(R"(
    // The point here is to exercise the conversion of `uN[32]` to an
    // `InferenceVariableKind`, as opposed to the more common `u32`.
    fn foo<M: uN[32]>(a: uN[M]) -> uN[M] { a }

    fn bar() {
      foo<u32:5>(u5:3);
    }
)");

  XLS_ASSERT_OK_AND_ASSIGN(const Function* foo,
                           module_->GetMemberOrError<Function>("foo"));
  ASSERT_EQ(foo->parametric_bindings().size(), 1);
  const NameDef* m = foo->parametric_bindings()[0]->name_def();
  ASSERT_EQ(foo->params().size(), 1);
  XLS_ASSERT_OK(
      table_->DefineParametricVariable(*foo->parametric_bindings()[0]));
  for (const Param* param : foo->params()) {
    XLS_ASSERT_OK(table_->SetTypeAnnotation(param, param->type_annotation()));
  }
  XLS_ASSERT_OK_AND_ASSIGN(const Function* bar,
                           module_->GetMemberOrError<Function>("bar"));
  ASSERT_EQ(bar->body()->statements().size(), 1);
  const Invocation* invocation = down_cast<const Invocation*>(
      ToAstNode(bar->body()->statements().at(0)->wrapped()));
  XLS_ASSERT_OK_AND_ASSIGN(
      const ParametricContext* parametric_context,
      table_->AddParametricInvocation(*invocation, *foo, bar,
                                      /*caller_invocation=*/std::nullopt,
                                      /*self_type=*/std::nullopt));

  std::optional<ParametricContextScopedExpr> parametric_inv_m_value =
      table_->GetParametricValue(*m, *parametric_context);
  ASSERT_TRUE(parametric_inv_m_value.has_value());
  EXPECT_EQ(parametric_inv_m_value->context(), std::nullopt);
  EXPECT_EQ(parametric_inv_m_value->expr()->ToString(), "u32:5");
  EXPECT_THAT(table_->ToString(),
              AllOf(HasSubstr("Node: a: uN[M]"), HasSubstr("Annotation: uN[M]"),
                    HasSubstr("Node: M"), HasSubstr("Annotation: uN[32]"),
                    HasSubstr("Variable: M")));
}

TEST_F(InferenceTableTest, ParametricVariableWithUnsupportedAnnotation) {
  ParseAndInitModuleAndTable(R"(
    struct T {}
    fn foo<X: T>() -> T { X }

    fn bar() {
      let t = T{};
      foo<t>();
    }
)");

  XLS_ASSERT_OK_AND_ASSIGN(const Function* foo,
                           module_->GetMemberOrError<Function>("foo"));
  ASSERT_EQ(foo->parametric_bindings().size(), 1);
  EXPECT_THAT(
      table_->DefineParametricVariable(*foo->parametric_bindings()[0]),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Inference variables of type T are not supported")));
}

TEST_F(InferenceTableTest, Clone) {
  NameDef* x = module_->Make<NameDef>(Span::Fake(), "x", /*definer=*/nullptr);
  NameDef* y = module_->Make<NameDef>(Span::Fake(), "y", /*definer=*/nullptr);
  NameRef* x_ref = module_->Make<NameRef>(Span::Fake(), "x", x);
  NameRef* y_ref = module_->Make<NameRef>(Span::Fake(), "y", y);
  AstNode* add_node = module_->Make<Binop>(Span::Fake(), BinopKind::kAdd, x_ref,
                                           y_ref, Span::Fake());
  XLS_ASSERT_OK_AND_ASSIGN(const NameRef* t0,
                           table_->DefineInternalVariable(
                               InferenceVariableKind::kType, add_node, "T0"));

  const TypeAnnotation* annotation =
      CreateU32Annotation(*module_, Span::Fake());

  XLS_EXPECT_OK(table_->SetTypeVariable(add_node, t0));
  XLS_EXPECT_OK(table_->SetTypeVariable(y_ref, t0));
  XLS_EXPECT_OK(table_->SetTypeAnnotation(add_node, annotation));
  XLS_EXPECT_OK(table_->SetTypeAnnotation(x_ref, annotation));

  XLS_ASSERT_OK_AND_ASSIGN(AstNode * clone,
                           table_->Clone(add_node, &NoopCloneReplacer));

  Binop* cloned_add_node = down_cast<Binop*>(clone);
  EXPECT_EQ(table_->GetTypeAnnotation(cloned_add_node), annotation);
  EXPECT_EQ(table_->GetTypeAnnotation(cloned_add_node->lhs()), annotation);
  EXPECT_EQ(table_->GetTypeAnnotation(cloned_add_node->rhs()), std::nullopt);

  EXPECT_EQ(table_->GetTypeVariable(cloned_add_node), t0);
  EXPECT_EQ(table_->GetTypeVariable(cloned_add_node->lhs()), std::nullopt);
  EXPECT_EQ(table_->GetTypeVariable(cloned_add_node->rhs()), t0);
}

}  // namespace
}  // namespace xls::dslx
