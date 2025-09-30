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

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "gtest/gtest.h"
#include "xls/common/casts.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type_info.h"

namespace xls::dslx {
namespace {

TEST(FunctionSpecializerTest, SpecializesParametricFunction) {
  constexpr std::string_view kProgram =
      R"(fn scale<M: u32>(x: bits[M]) -> bits[M] {
  let shifted = x << M;
  shifted
}

fn call() -> bits[32] {
  scale(bits[32]:0x1)
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
  ASSERT_TRUE(functions.contains("call"));
  Function* scale_fn = functions.at("scale");
  Function* call_fn = functions.at("call");

  StatementBlock* call_body = call_fn->body();
  ASSERT_EQ(call_body->statements().size(), 1);
  const Statement::Wrapped& wrapped =
      call_body->statements().front()->wrapped();
  ASSERT_TRUE(std::holds_alternative<Expr*>(wrapped));
  auto* invocation = down_cast<Invocation*>(std::get<Expr*>(wrapped));

  TypeInfo* root_type_info = typechecked.type_info;
  ASSERT_NE(root_type_info, nullptr);

  std::optional<TypeInfo*> instantiation_type_info =
      root_type_info->GetInvocationTypeInfo(invocation, ParametricEnv());
  ASSERT_TRUE(instantiation_type_info.has_value());

  std::optional<const ParametricEnv*> callee_env =
      root_type_info->GetInvocationCalleeBindings(invocation, ParametricEnv());
  ASSERT_TRUE(callee_env.has_value());

  XLS_ASSERT_OK_AND_ASSIGN(
      Function * specialized,
      InsertFunctionSpecialization(scale_fn, **callee_env, "scale_M32"));

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
            (std::vector<std::string>{"scale", "scale_M32", "call"}));
}

}  // namespace
}  // namespace xls::dslx
