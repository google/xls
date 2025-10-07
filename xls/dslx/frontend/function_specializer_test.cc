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

}  // namespace
}  // namespace xls::dslx
