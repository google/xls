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

#include "xls/dslx/type_system/type_info.h"

#include <optional>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xls/common/casts.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/type_system/parametric_env.h"

namespace xls::dslx {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

TEST(TypeInfoTest, Instantiate) {
  FileTable file_table;
  Module module("test", /*fs_path=*/std::nullopt, file_table);
  TypeInfoOwner owner;
  XLS_ASSERT_OK_AND_ASSIGN(TypeInfo * type_info, owner.New(&module));
  EXPECT_EQ(type_info->parent(), nullptr);
}

// Tests our internal-error reporting path if a bad parametric environment is
// given when building up the type information.
TEST(TypeInfoTest, AddingBadCallerEnvGivesError) {
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckedModule tm,
                           ParseAndTypecheck(R"(
fn p<X: u32, Y: u32>() -> u32 {
  X+Y
}

fn main() -> u32 {
  p<u32:2, u32:3>()
})",
                                             "test.x", "test", &import_data));

  Function* main = tm.module->GetFunctionByName().at("main");

  const Invocation* invoke_p = down_cast<const Invocation*>(
      ToAstNode(main->body()->statements().at(0)->wrapped()));
  ASSERT_NE(invoke_p, nullptr);

  // Main has no parametric env so anything with a value inside is bad.
  const ParametricEnv bad_caller_env(
      absl::flat_hash_map<std::string, InterpValue>{
          {"A", InterpValue::MakeU32(42)},
      });

  const ParametricEnv valid_callee_env(
      absl::flat_hash_map<std::string, InterpValue>{
          {"X", InterpValue::MakeU32(42)},
          {"Y", InterpValue::MakeU32(64)},
      });

  // We should not be able to add a caller environment in `main()`.
  EXPECT_THAT(tm.type_info->AddInvocationTypeInfo(
                  *invoke_p, /*caller=*/main, bad_caller_env, valid_callee_env,
                  /*derived_type_info=*/nullptr),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("caller `main` given env with key `A` not "
                                 "present in parametric keys: {}")));
}

}  // namespace
}  // namespace xls::dslx
