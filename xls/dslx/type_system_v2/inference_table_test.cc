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
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/strings/substitute.h"
#include "xls/common/casts.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/parser.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/scanner.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system_v2/inference_table_to_type_info.h"
#include "xls/dslx/warning_collector.h"
#include "xls/dslx/warning_kind.h"

namespace xls::dslx {
namespace {

using ::absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::ContainsRegex;
using ::testing::HasSubstr;

class InferenceTableTest : public ::testing::Test {
 public:
  void SetUp() override {
    import_data_.emplace(CreateImportDataForTest());
    warning_collector_.emplace(kAllWarningsSet);
    module_ =
        std::make_unique<Module>("test", /*fs_path=*/std::nullopt, file_table_);
    table_ = InferenceTable::Create(*module_, file_table_);
  }

  absl::StatusOr<TypeInfo*> ConvertTableToTypeInfo() {
    XLS_ASSIGN_OR_RETURN(TypeInfo * ti, InferenceTableToTypeInfo(
                                            *table_, *module_, *import_data_,
                                            *warning_collector_, file_table_));
    return ti;
  }

  absl::StatusOr<std::string> TypeInfoToString(const TypeInfo& ti) {
    if (ti.dict().empty()) {
      return "";
    }
    std::vector<std::string> strings;
    for (const auto& [node, type] : ti.dict()) {
      Span span = node->GetSpan().has_value() ? *node->GetSpan() : Span::Fake();
      strings.push_back(absl::Substitute("span: $0, node: `$1`, type: $2",
                                         span.ToString(file_table_),
                                         node->ToString(), type->ToString()));
    }
    absl::c_sort(strings);
    return strings.size() == 1
               ? strings[0]
               : absl::Substitute("\n$0\n", absl::StrJoin(strings, "\n"));
  }

  absl::StatusOr<std::string> ConvertTableToTypeInfoString() {
    XLS_ASSIGN_OR_RETURN(TypeInfo * ti, ConvertTableToTypeInfo());
    return TypeInfoToString(*ti);
  }

  void ParseAndInitModuleAndTable(std::string_view program) {
    Scanner scanner(file_table_, file_table_.GetOrCreate("fake.x"),
                    std::string(program));
    Parser parser("fake", &scanner);
    XLS_ASSERT_OK_AND_ASSIGN(module_, parser.ParseModule());
    table_ = InferenceTable::Create(*module_, file_table_);
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

TEST_F(InferenceTableTest, TypeInfoForOneSimpleAnnotation) {
  // Just one def for x:u32.
  NameDef* x = module_->Make<NameDef>(Span::Fake(), "x", /*definer=*/nullptr);
  TypeAnnotation* annotation = module_->Make<BuiltinTypeAnnotation>(
      Span::Fake(), BuiltinType::kU32,
      module_->GetOrCreateBuiltinNameDef("u32"));
  XLS_EXPECT_OK(table_->SetTypeAnnotation(x, annotation));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           ConvertTableToTypeInfoString());
  EXPECT_EQ(type_info_string,
            "span: <no-file>:1:1-1:1, node: `x`, type: uN[32]");
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
      NameRef * n_var,
      table_->DefineInternalVariable(InferenceVariableKind::kInteger, n, "N"));
  EXPECT_THAT(table_->SetTypeVariable(x, n_var),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(InferenceTableTest, SignednessMismatch) {
  // Apply the same type variable to the LHS and RHS of an addition, then claim
  // that the LHS (x) is annotated as u32 and the RHS (y) is annotated as s32.
  // Upon associating s32 with y, we should get a signedness mismatch.
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

  XLS_ASSERT_OK_AND_ASSIGN(
      NameRef * t0, table_->DefineInternalVariable(InferenceVariableKind::kType,
                                                   add_node, "T0"));
  XLS_EXPECT_OK(table_->SetTypeVariable(x_ref, t0));
  XLS_EXPECT_OK(table_->SetTypeVariable(y_ref, t0));
  XLS_EXPECT_OK(table_->SetTypeAnnotation(x_ref, u32_annotation));

  EXPECT_THAT(
      table_->SetTypeAnnotation(y_ref, s32_annotation),
      StatusIs(absl::StatusCode::kInvalidArgument,
               ContainsRegex("signed vs. unsigned mismatch.*s32.*vs. u32")));
}

TEST_F(InferenceTableTest, SignednessAgreement) {
  // Apply the same type variable to the LHS and RHS of an addition, then claim
  // that the LHS (x) and the RHS (y) are both annotated as u32. This should not
  // error.
  NameDef* x = module_->Make<NameDef>(Span::Fake(), "x", /*definer=*/nullptr);
  NameDef* y = module_->Make<NameDef>(Span::Fake(), "y", /*definer=*/nullptr);
  NameRef* x_ref = module_->Make<NameRef>(Span::Fake(), "x", x);
  NameRef* y_ref = module_->Make<NameRef>(Span::Fake(), "y", y);
  AstNode* add_node = module_->Make<Binop>(Span::Fake(), BinopKind::kAdd, x_ref,
                                           y_ref, Span::Fake());

  TypeAnnotation* u32_annotation = module_->Make<BuiltinTypeAnnotation>(
      Span::Fake(), BuiltinType::kU32,
      module_->GetOrCreateBuiltinNameDef("u32"));

  XLS_ASSERT_OK_AND_ASSIGN(
      NameRef * t0, table_->DefineInternalVariable(InferenceVariableKind::kType,
                                                   add_node, "T0"));
  XLS_EXPECT_OK(table_->SetTypeVariable(x_ref, t0));
  XLS_EXPECT_OK(table_->SetTypeVariable(y_ref, t0));
  XLS_EXPECT_OK(table_->SetTypeAnnotation(x_ref, u32_annotation));
  XLS_EXPECT_OK(table_->SetTypeAnnotation(y_ref, u32_annotation));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           ConvertTableToTypeInfoString());
  EXPECT_EQ(type_info_string, R"(
span: <no-file>:1:1-1:1, node: `x`, type: uN[32]
span: <no-file>:1:1-1:1, node: `y`, type: uN[32]
)");
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
  XLS_ASSERT_OK_AND_ASSIGN(const Function* bar,
                           module_->GetMemberOrError<Function>("bar"));
  ASSERT_EQ(bar->body()->statements().size(), 2);
  const Invocation* invocation1 = down_cast<const Invocation*>(
      ToAstNode(bar->body()->statements().at(0)->wrapped()));
  const Invocation* invocation2 = down_cast<const Invocation*>(
      ToAstNode(bar->body()->statements().at(1)->wrapped()));
  XLS_ASSERT_OK(
      table_->AddParametricInvocation(*invocation1, *foo, *bar,
                                      /*caller_invocation=*/std::nullopt));
  XLS_ASSERT_OK(
      table_->AddParametricInvocation(*invocation2, *foo, *bar,
                                      /*caller_invocation=*/std::nullopt));

  XLS_ASSERT_OK_AND_ASSIGN(TypeInfo * ti, ConvertTableToTypeInfo());
  std::optional<TypeInfo*> invocation_ti1 =
      ti->GetInvocationTypeInfo(invocation1, ParametricEnv());
  std::optional<TypeInfo*> invocation_ti2 =
      ti->GetInvocationTypeInfo(invocation2, ParametricEnv());
  EXPECT_TRUE(invocation_ti1.has_value());
  EXPECT_TRUE(invocation_ti2.has_value());
  XLS_ASSERT_OK_AND_ASSIGN(std::string invocation_ti1_string,
                           TypeInfoToString(**invocation_ti1));
  XLS_ASSERT_OK_AND_ASSIGN(std::string invocation_ti2_string,
                           TypeInfoToString(**invocation_ti2));
  EXPECT_EQ(invocation_ti1_string, R"(
span: fake.x:2:20-2:28, node: `a: uN[N]`, type: uN[4]
span: fake.x:2:26-2:27, node: `N`, type: uN[32]
span: fake.x:2:26-2:27, node: `u32`, type: typeof(uN[32])
)");
  EXPECT_EQ(invocation_ti2_string, R"(
span: fake.x:2:20-2:28, node: `a: uN[N]`, type: uN[5]
span: fake.x:2:26-2:27, node: `N`, type: uN[32]
span: fake.x:2:26-2:27, node: `u32`, type: typeof(uN[32])
)");
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
  XLS_ASSERT_OK(
      table_->AddParametricInvocation(*invocation1, *foo, *bar,
                                      /*caller_invocation=*/std::nullopt));
  XLS_ASSERT_OK(
      table_->AddParametricInvocation(*invocation2, *foo, *bar,
                                      /*caller_invocation=*/std::nullopt));

  XLS_ASSERT_OK_AND_ASSIGN(TypeInfo * ti, ConvertTableToTypeInfo());
  std::optional<TypeInfo*> invocation_ti1 =
      ti->GetInvocationTypeInfo(invocation1, ParametricEnv());
  std::optional<TypeInfo*> invocation_ti2 =
      ti->GetInvocationTypeInfo(invocation2, ParametricEnv());
  EXPECT_TRUE(invocation_ti1.has_value());
  EXPECT_TRUE(invocation_ti2.has_value());
  XLS_ASSERT_OK_AND_ASSIGN(std::string invocation_ti1_string,
                           TypeInfoToString(**invocation_ti1));
  XLS_ASSERT_OK_AND_ASSIGN(std::string invocation_ti2_string,
                           TypeInfoToString(**invocation_ti2));
  EXPECT_THAT(invocation_ti1_string,
              AllOf(HasSubstr("node: `a: uN[M]`, type: uN[4]"),
                    HasSubstr("node: `b: uN[N]`, type: uN[16]")));
  EXPECT_THAT(invocation_ti2_string,
              AllOf(HasSubstr("node: `a: uN[M]`, type: uN[5]"),
                    HasSubstr("node: `b: uN[N]`, type: uN[25]")));
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
  XLS_ASSERT_OK(
      table_->AddParametricInvocation(*invocation, *foo, *bar,
                                      /*caller_invocation=*/std::nullopt));

  XLS_ASSERT_OK_AND_ASSIGN(TypeInfo * ti, ConvertTableToTypeInfo());
  std::optional<TypeInfo*> invocation_ti =
      ti->GetInvocationTypeInfo(invocation, ParametricEnv());
  EXPECT_TRUE(invocation_ti.has_value());
  XLS_ASSERT_OK_AND_ASSIGN(std::string invocation_ti1_string,
                           TypeInfoToString(**invocation_ti));
  EXPECT_THAT(invocation_ti1_string,
              HasSubstr("node: `a: uN[M]`, type: uN[5]"));
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

TEST_F(InferenceTableTest, ParametricVariableAndParametricCaller) {
  ParseAndInitModuleAndTable(R"(
    fn foo<M: u32 = {u32:4}, N: u32 = {M * M}>(a: uN[M], b: uN[N])
        -> uN[M + N] { a ++ b }

    fn bar<X: u32>() {
      foo<X>(zero!<uN[X]>(), zero!<uN[X * X]>());
    }

    fn baz() {
      bar<u32:5>();
      bar<u32:7>();
    }
)");

  // Get `foo` and its relevant nodes.
  XLS_ASSERT_OK_AND_ASSIGN(const Function* foo,
                           module_->GetMemberOrError<Function>("foo"));
  ASSERT_EQ(foo->parametric_bindings().size(), 2);
  ASSERT_EQ(foo->params().size(), 2);
  XLS_ASSERT_OK(
      table_->DefineParametricVariable(*foo->parametric_bindings()[0]));
  XLS_ASSERT_OK(
      table_->DefineParametricVariable(*foo->parametric_bindings()[1]));
  for (const Param* param : foo->params()) {
    XLS_ASSERT_OK(table_->SetTypeAnnotation(param, param->type_annotation()));
  }

  // Get `bar` and its relevant nodes.
  XLS_ASSERT_OK_AND_ASSIGN(const Function* bar,
                           module_->GetMemberOrError<Function>("bar"));
  ASSERT_EQ(bar->parametric_bindings().size(), 1);
  XLS_ASSERT_OK(
      table_->DefineParametricVariable(*bar->parametric_bindings()[0]));
  ASSERT_EQ(bar->body()->statements().size(), 1);
  const Invocation* foo_invocation_node = down_cast<const Invocation*>(
      ToAstNode(bar->body()->statements().at(0)->wrapped()));

  // Get `baz` and its relevant nodes.
  XLS_ASSERT_OK_AND_ASSIGN(const Function* baz,
                           module_->GetMemberOrError<Function>("baz"));
  ASSERT_EQ(baz->body()->statements().size(), 2);
  const Invocation* bar_invocation_node1 = down_cast<const Invocation*>(
      ToAstNode(baz->body()->statements().at(0)->wrapped()));
  const Invocation* bar_invocation_node2 = down_cast<const Invocation*>(
      ToAstNode(baz->body()->statements().at(1)->wrapped()));

  // Add the 4 parametric invocations to the table.
  XLS_ASSERT_OK_AND_ASSIGN(
      const ParametricInvocation* bar_invocation1,
      table_->AddParametricInvocation(*bar_invocation_node1, *bar, *baz,
                                      /*caller_invocation=*/std::nullopt));
  XLS_ASSERT_OK_AND_ASSIGN(
      const ParametricInvocation* bar_invocation2,
      table_->AddParametricInvocation(*bar_invocation_node2, *bar, *baz,
                                      /*caller_invocation=*/std::nullopt));
  XLS_ASSERT_OK(table_->AddParametricInvocation(*foo_invocation_node, *foo,
                                                *bar, bar_invocation1));
  XLS_ASSERT_OK(table_->AddParametricInvocation(*foo_invocation_node, *foo,
                                                *bar, bar_invocation2));

  // Check what we get for the 2 invocations of `foo`.
  XLS_ASSERT_OK_AND_ASSIGN(TypeInfo * ti, ConvertTableToTypeInfo());
  std::optional<TypeInfo*> invocation_ti1 = ti->GetInvocationTypeInfo(
      foo_invocation_node,
      ParametricEnv(absl::flat_hash_map<std::string, InterpValue>{
          {"X", InterpValue::MakeU32(5)}}));
  std::optional<TypeInfo*> invocation_ti2 = ti->GetInvocationTypeInfo(
      foo_invocation_node,
      ParametricEnv(absl::flat_hash_map<std::string, InterpValue>{
          {"X", InterpValue::MakeU32(7)}}));
  EXPECT_TRUE(invocation_ti1.has_value());
  EXPECT_TRUE(invocation_ti2.has_value());
  XLS_ASSERT_OK_AND_ASSIGN(std::string invocation_ti1_string,
                           TypeInfoToString(**invocation_ti1));
  XLS_ASSERT_OK_AND_ASSIGN(std::string invocation_ti2_string,
                           TypeInfoToString(**invocation_ti2));
  EXPECT_THAT(invocation_ti1_string,
              AllOf(HasSubstr("node: `a: uN[M]`, type: uN[5]"),
                    HasSubstr("node: `b: uN[N]`, type: uN[25]")));
  EXPECT_THAT(invocation_ti2_string,
              AllOf(HasSubstr("node: `a: uN[M]`, type: uN[7]"),
                    HasSubstr("node: `b: uN[N]`, type: uN[49]")));
}

TEST_F(InferenceTableTest, TooManyParametricsInInvocation) {
  ParseAndInitModuleAndTable(R"(
    fn foo<N: u32>(a: uN[N]) -> uN[N] { a }
    fn bar() {
      foo<u32:4, u32:5>(u4:1);
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
  XLS_ASSERT_OK_AND_ASSIGN(const Function* bar,
                           module_->GetMemberOrError<Function>("bar"));
  ASSERT_EQ(bar->body()->statements().size(), 1);
  const Invocation* invocation = down_cast<const Invocation*>(
      ToAstNode(bar->body()->statements().at(0)->wrapped()));
  EXPECT_THAT(
      table_->AddParametricInvocation(*invocation, *foo, *bar,
                                      /*caller_invocation=*/std::nullopt),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Too many parametric values supplied")));
}

}  // namespace
}  // namespace xls::dslx
