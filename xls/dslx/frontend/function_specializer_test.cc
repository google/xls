// Copyright 2025 The XLS Authors
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

#include "xls/dslx/frontend/function_specializer.h"

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/match.h"
#include "gtest/gtest.h"
#include "xls/common/casts.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/replace_invocations.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/typecheck_module.h"
#include "xls/dslx/warning_collector.h"

namespace xls::dslx {
namespace {

TEST(FunctionSpecializerTest, SpecializesParametricFunction) {
  constexpr std::string_view kProgram =
      R"(fn scale<M: u32>(x: bits[M]) -> bits[M] {
  let shifted = x << M;
  shifted
}
)";

  std::unique_ptr<ImportData> import_data = CreateImportDataPtrForTest();
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckedModule typechecked,
                           ParseAndTypecheck(kProgram, "test_module.x",
                                             "test_module", import_data.get()));

  Module* module = typechecked.module;
  ASSERT_NE(module, nullptr);

  auto functions = module->GetFunctionByName();
  ASSERT_TRUE(functions.contains("scale"));
  Function* scale_fn = functions.at("scale");

  ASSERT_FALSE(scale_fn->parametric_bindings().empty());
  const ParametricBinding* binding = scale_fn->parametric_bindings().front();
  InterpValue binding_value = InterpValue::MakeUBits(/*bit_count=*/32, 32);
  absl::flat_hash_map<std::string, InterpValue> env_bindings;
  env_bindings.emplace(binding->identifier(), binding_value);
  ParametricEnv env(env_bindings);

  XLS_ASSERT_OK_AND_ASSIGN(
      Function * specialized,
      InsertFunctionSpecialization(scale_fn, env, "scale_M32"));

  EXPECT_FALSE(specialized->IsParametric());
  EXPECT_EQ(specialized->identifier(), "scale_M32");

  ASSERT_EQ(specialized->params().size(), 1);
  auto* array_type = down_cast<ArrayTypeAnnotation*>(
      specialized->params()[0]->type_annotation());
  auto* literal_dim = down_cast<Number*>(array_type->dim());
  EXPECT_EQ(literal_dim->text(), "0x20");

  StatementBlock* specialized_body = specialized->body();
  ASSERT_EQ(specialized_body->statements().size(), 2);

  const Statement::Wrapped& shifted_statement =
      specialized_body->statements().front()->wrapped();
  ASSERT_TRUE(std::holds_alternative<Let*>(shifted_statement));
  auto* let_stmt = std::get<Let*>(shifted_statement);
  auto* shift = down_cast<Binop*>(let_stmt->rhs());
  auto* shift_amount = down_cast<Number*>(shift->rhs());
  EXPECT_EQ(shift_amount->text(), "0x20");

  ASSERT_TRUE(module->GetFunction("scale_M32").has_value());
  EXPECT_EQ(module->GetFunctionNames(),
            (std::vector<std::string>{"scale", "scale_M32"}));
}

TEST(FunctionSpecializerTest, SpecializedParametersRebindNameRefs) {
  constexpr std::string_view kProgram =
      R"(fn passthrough<N: u32>(x: bits[N]) -> bits[N] {
  x
}
)";

  std::unique_ptr<ImportData> import_data = CreateImportDataPtrForTest();
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckedModule typechecked,
                           ParseAndTypecheck(kProgram, "test_module.x",
                                             "test_module", import_data.get()));

  Module* module = typechecked.module;
  ASSERT_NE(module, nullptr);

  auto functions = module->GetFunctionByName();
  ASSERT_TRUE(functions.contains("passthrough"));
  Function* passthrough_fn = functions.at("passthrough");

  ASSERT_FALSE(passthrough_fn->parametric_bindings().empty());
  const ParametricBinding* binding =
      passthrough_fn->parametric_bindings().front();
  InterpValue binding_value = InterpValue::MakeUBits(/*bit_count=*/32, 8);
  absl::flat_hash_map<std::string, InterpValue> env_bindings;
  env_bindings.emplace(binding->identifier(), binding_value);
  ParametricEnv env(env_bindings);

  XLS_ASSERT_OK_AND_ASSIGN(
      Function * specialized,
      InsertFunctionSpecialization(passthrough_fn, env,
                                   "passthrough_N8"));

  ASSERT_EQ(specialized->params().size(), 1);
  Param* specialized_param = specialized->params()[0];
  NameDef* specialized_name_def = specialized_param->name_def();

  StatementBlock* specialized_body = specialized->body();
  ASSERT_EQ(specialized_body->statements().size(), 1);
  const Statement::Wrapped& specialized_wrapped =
      specialized_body->statements().front()->wrapped();
  ASSERT_TRUE(std::holds_alternative<Expr*>(specialized_wrapped));
  Expr* ret_expr = std::get<Expr*>(specialized_wrapped);

  auto* name_ref = down_cast<NameRef*>(ret_expr);
  ASSERT_TRUE(std::holds_alternative<const NameDef*>(name_ref->name_def()));
  const NameDef* bound_name_def =
      std::get<const NameDef*>(name_ref->name_def());
  EXPECT_EQ(bound_name_def, specialized_name_def);

  // Typecheck the module again to ensure the specialized function integrates.
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> cloned, CloneModule(*module));
  std::unique_ptr<ImportData> tc_import_data = CreateImportDataPtrForTest();
  WarningCollector warnings(tc_import_data->enabled_warnings());
  std::filesystem::path module_path = module->fs_path().value_or(
      std::filesystem::path("test_module.x"));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModuleInfo> module_info,
      TypecheckModule(std::move(cloned), module_path.string(),
                      tc_import_data.get(), &warnings));
  EXPECT_NE(module_info->type_info(), nullptr);
  EXPECT_TRUE(module_info->module().GetFunction("passthrough_N8").has_value());
}

TEST(FunctionSpecializerTest, SpecializedFunctionReceivesSyntheticSpans) {
  constexpr std::string_view kProgram =
      R"(fn add_one<N: u32>(x: bits[N]) -> bits[N] {
  x + uN[N]:1
}

fn twice<N: u32>(x: bits[N]) -> bits[N] {
  add_one<N>(add_one<N>(x))
}
)";

  std::unique_ptr<ImportData> import_data = CreateImportDataPtrForTest();
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckedModule typechecked,
                           ParseAndTypecheck(kProgram, "span_test.x",
                                             "span_test", import_data.get()));

  Module* module = typechecked.module;
  ASSERT_NE(module, nullptr);

  auto functions = module->GetFunctionByName();
  ASSERT_TRUE(functions.contains("add_one"));
  Function* add_one = functions.at("add_one");

  ASSERT_FALSE(add_one->parametric_bindings().empty());
  const ParametricBinding* binding = add_one->parametric_bindings().front();
  absl::flat_hash_map<std::string, InterpValue> env_bindings;
  env_bindings.emplace(binding->identifier(),
                       InterpValue::MakeUBits(/*bit_count=*/32, 4));
  ParametricEnv env(env_bindings);

  XLS_ASSERT_OK_AND_ASSIGN(Function * specialized,
                           InsertFunctionSpecialization(add_one, env,
                                                        "add_one_N4"));

  ASSERT_NE(specialized, nullptr);
  FileTable& files = *module->file_table();
  std::string_view original_filename = add_one->span().GetFilename(files);
  std::string_view specialized_filename = specialized->span().GetFilename(files);

  EXPECT_NE(original_filename, specialized_filename);
  EXPECT_TRUE(absl::StrContains(specialized_filename, "<specialization:"));

  // The specialized function span should enclose its body span.
  ASSERT_NE(specialized->body(), nullptr);
  EXPECT_TRUE(specialized->span().Contains(specialized->body()->span()));

  // All statements in the body should be in the synthetic file and contained
  // within the function span.
  for (Statement* stmt : specialized->body()->statements()) {
    ASSERT_NE(stmt, nullptr);
    std::optional<Span> stmt_span = stmt->GetSpan();
    ASSERT_TRUE(stmt_span.has_value());
    EXPECT_EQ(stmt_span->GetFilename(files), specialized_filename);
    EXPECT_TRUE(specialized->span().Contains(*stmt_span));
  }

  // Function parameters should also be placed in the synthetic file.
  for (Param* param : specialized->params()) {
    ASSERT_NE(param, nullptr);
    EXPECT_EQ(param->span().GetFilename(files), specialized_filename);
    EXPECT_TRUE(specialized->span().Contains(param->span()));
  }
}

TEST(FunctionSpecializerTest, SpecializedInvocationParametricsAreConcrete) {
  constexpr std::string_view kProgram =
      R"(fn repeat<COUNT: u32, N: u32>(x: uN[N]) -> uN[N][COUNT] {
  uN[N][COUNT]:[x, ...]
}

fn select_poly<N: u32>(polys: uN[6][N], selector: uN[N]) -> uN[6] {
  let repeated = repeat<N, N>(selector);
  let first = repeated[u32:0];
  if first == selector { polys[u32:0] } else { polys[u32:1] }
}
)";

  std::unique_ptr<ImportData> import_data = CreateImportDataPtrForTest();
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckedModule typechecked,
                           ParseAndTypecheck(kProgram, "invocation_test.x",
                                             "invocation_test",
                                             import_data.get()));

  Module* module = typechecked.module;
  ASSERT_NE(module, nullptr);

  auto functions = module->GetFunctionByName();
  ASSERT_TRUE(functions.contains("select_poly"));
  Function* select_poly = functions.at("select_poly");

  ASSERT_FALSE(select_poly->parametric_bindings().empty());
  const ParametricBinding* binding = select_poly->parametric_bindings().front();
  absl::flat_hash_map<std::string, InterpValue> env_bindings;
  env_bindings.emplace(binding->identifier(),
                       InterpValue::MakeUBits(/*bit_count=*/32, 34));
  ParametricEnv env(env_bindings);

  XLS_ASSERT_OK_AND_ASSIGN(Function * specialized,
                           InsertFunctionSpecialization(select_poly, env,
                                                        "select_poly_34"));

  StatementBlock* body = specialized->body();
  ASSERT_NE(body, nullptr);
  ASSERT_FALSE(body->statements().empty());

  const Statement::Wrapped& first_wrapped =
      body->statements().front()->wrapped();
  ASSERT_TRUE(std::holds_alternative<Let*>(first_wrapped));
  Let* repeated_let = std::get<Let*>(first_wrapped);

  Expr* rhs = repeated_let->rhs();
  auto* invocation = dynamic_cast<Invocation*>(rhs);
  ASSERT_NE(invocation, nullptr);

  const std::vector<ExprOrType>& parametrics =
      invocation->explicit_parametrics();
  ASSERT_FALSE(parametrics.empty());
  ASSERT_TRUE(std::holds_alternative<Expr*>(parametrics.front()));
  Expr* param_expr = std::get<Expr*>(parametrics.front());
  auto* number = dynamic_cast<Number*>(param_expr);
  ASSERT_NE(number, nullptr);
  EXPECT_EQ(number->text(), "0x22");

  const std::string specialized_filename =
      std::string(specialized->span().GetFilename(*module->file_table()));

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> cloned_module,
                           CloneModule(*module));
  std::unique_ptr<ImportData> tc_import_data = CreateImportDataPtrForTest();
  tc_import_data->file_table().GetOrCreate("invocation_test.x");
  tc_import_data->file_table().GetOrCreate(specialized_filename);
  std::filesystem::path module_path = module->fs_path().value_or(
      std::filesystem::path("invocation_test.x"));

  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule retyped,
      TypecheckModule(std::move(cloned_module), module_path.string(),
                      tc_import_data.get()));
  EXPECT_NE(retyped.type_info, nullptr);
  const Function* caller = retyped.module->GetFunction("select_poly").value();
  const Function* callee = retyped.module->GetFunction("repeat").value();
  const Function* callers_arr[] = {caller};

  InvocationRewriteRule rule;
  rule.from_callee = callee;
  rule.to_callee = callee;
  const InvocationRewriteRule rules_arr[] = {rule};

  absl::StatusOr<TypecheckedModule> rewritten = ReplaceInvocationsInModule(
      retyped, absl::MakeSpan(callers_arr), absl::MakeSpan(rules_arr),
      *tc_import_data, "invocation_test.rewrite");
  ASSERT_TRUE(rewritten.ok()) << rewritten.status();
}

TEST(FunctionSpecializerTest, SpecializedInvocationSupportsRewrite) {
  constexpr std::string_view kProgram =
      R"(fn repeat<COUNT: u32, N: u32>(x: uN[N]) -> uN[N][COUNT] {
  uN[N][COUNT]:[x, ...]
}

fn select_poly<N: u32>(polys: uN[6][N], selector: uN[N]) -> uN[6] {
  let repeated = repeat<N, N>(selector);
  let first = repeated[u32:0];
  if first == selector { polys[u32:0] } else { polys[u32:1] }
}
)";

  std::unique_ptr<ImportData> import_data = CreateImportDataPtrForTest();
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckedModule typechecked,
                           ParseAndTypecheck(kProgram, "rewrite_test.x",
                                             "rewrite_test",
                                             import_data.get()));

  Module* module = typechecked.module;
  ASSERT_NE(module, nullptr);

  Function* select_poly = module->GetFunctionByName().at("select_poly");
  const ParametricBinding* binding = select_poly->parametric_bindings().front();
  absl::flat_hash_map<std::string, InterpValue> env_bindings;
  env_bindings.emplace(binding->identifier(),
                       InterpValue::MakeUBits(/*bit_count=*/32, 12));
  ParametricEnv env(env_bindings);

  XLS_ASSERT_OK_AND_ASSIGN(Function * specialized,
                           InsertFunctionSpecialization(select_poly, env,
                                                        "select_poly_12"));
  ASSERT_NE(specialized, nullptr);

  const std::string specialized_filename =
      std::string(specialized->span().GetFilename(*module->file_table()));

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> cloned_module,
                           CloneModule(*module));
  std::unique_ptr<ImportData> rewrite_import_data =
      CreateImportDataPtrForTest();

  rewrite_import_data->file_table().GetOrCreate("rewrite_test.x");
  rewrite_import_data->file_table().GetOrCreate(specialized_filename);

  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule retyped,
      TypecheckModule(std::move(cloned_module), "rewrite_test.x",
                      rewrite_import_data.get()));

  const Function* caller = retyped.module->GetFunction("select_poly").value();
  const Function* callee = retyped.module->GetFunction("repeat").value();
  const Function* callers_arr[] = {caller};

  InvocationRewriteRule rule;
  rule.from_callee = callee;
  rule.to_callee = callee;
  const InvocationRewriteRule rules_arr[] = {rule};

  absl::StatusOr<TypecheckedModule> replaced = ReplaceInvocationsInModule(
      retyped, absl::MakeSpan(callers_arr), absl::MakeSpan(rules_arr),
      *rewrite_import_data, "rewrite_test.rewrite");
  ASSERT_TRUE(replaced.ok()) << replaced.status();
}

TEST(FunctionSpecializerTest, TypeAnnotationsSubstituteParametricBindings) {
  constexpr std::string_view kProgram =
      R"(fn slice<M: u32>(x: bits[M]) -> bits[M + 8] {
  let y: bits[M + 8] = x ++ u8:0;
  y
}
)";

  std::unique_ptr<ImportData> import_data = CreateImportDataPtrForTest();
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckedModule typechecked,
                           ParseAndTypecheck(kProgram, "ta_module.x",
                                             "ta_module", import_data.get()));

  Module* module = typechecked.module;
  ASSERT_NE(module, nullptr);

  std::optional<Function*> slice_fn = module->GetFunction("slice");
  ASSERT_TRUE(slice_fn.has_value());

  const ParametricBinding* binding =
      slice_fn.value()->parametric_bindings().front();
  InterpValue binding_value = InterpValue::MakeUBits(/*bit_count=*/32, 16);
  absl::flat_hash_map<std::string, InterpValue> env_bindings;
  env_bindings.emplace(binding->identifier(), binding_value);
  ParametricEnv env(env_bindings);

  XLS_ASSERT_OK_AND_ASSIGN(
      Function * specialized,
      InsertFunctionSpecialization(slice_fn.value(), env, "slice_M16"));

  ASSERT_EQ(specialized->params().size(), 1);
  Param* specialized_param = specialized->params()[0];
  auto* param_type =
      down_cast<ArrayTypeAnnotation*>(specialized_param->type_annotation());
  ASSERT_NE(param_type, nullptr);
  auto* param_dim = dynamic_cast<Number*>(param_type->dim());
  ASSERT_NE(param_dim, nullptr);
  EXPECT_EQ(param_dim->text(), "0x10");

  auto* return_type =
      down_cast<ArrayTypeAnnotation*>(specialized->return_type());
  ASSERT_NE(return_type, nullptr);
  auto* return_dim = dynamic_cast<Binop*>(return_type->dim());
  ASSERT_NE(return_dim, nullptr);
  EXPECT_EQ(return_dim->binop_kind(), BinopKind::kAdd);
  auto* return_dim_lhs = dynamic_cast<Number*>(return_dim->lhs());
  ASSERT_NE(return_dim_lhs, nullptr);
  EXPECT_EQ(return_dim_lhs->text(), "0x10");
  auto* return_dim_rhs = dynamic_cast<Number*>(return_dim->rhs());
  ASSERT_NE(return_dim_rhs, nullptr);
  ASSERT_NE(module->file_table(), nullptr);
  const FileTable& file_table = *module->file_table();
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t return_dim_rhs_value,
                           return_dim_rhs->GetAsUint64(file_table));
  EXPECT_EQ(return_dim_rhs_value, 8);

  StatementBlock* body = specialized->body();
  ASSERT_EQ(body->statements().size(), 2);
  const Statement::Wrapped& let_wrapped =
      body->statements().front()->wrapped();
  ASSERT_TRUE(std::holds_alternative<Let*>(let_wrapped));
  auto* let_stmt = std::get<Let*>(let_wrapped);
  auto* let_type =
      down_cast<ArrayTypeAnnotation*>(let_stmt->type_annotation());
  ASSERT_NE(let_type, nullptr);
  auto* let_dim = dynamic_cast<Binop*>(let_type->dim());
  ASSERT_NE(let_dim, nullptr);
  EXPECT_EQ(let_dim->binop_kind(), BinopKind::kAdd);
  auto* let_dim_lhs = dynamic_cast<Number*>(let_dim->lhs());
  ASSERT_NE(let_dim_lhs, nullptr);
  EXPECT_EQ(let_dim_lhs->text(), "0x10");
  auto* let_dim_rhs = dynamic_cast<Number*>(let_dim->rhs());
  ASSERT_NE(let_dim_rhs, nullptr);
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t let_dim_rhs_value,
                           let_dim_rhs->GetAsUint64(file_table));
  EXPECT_EQ(let_dim_rhs_value, 8);
}

}  // namespace
}  // namespace xls::dslx
