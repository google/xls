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

#include "xls/dslx/interpreter.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/parse_and_typecheck.h"

namespace xls::dslx {
namespace {

TEST(InterpreterTest, RunIdentityFn) {
  auto module = absl::make_unique<Module>("test");
  Pos fake_pos("<fake>", 0, 0);
  Span fake_span(fake_pos, fake_pos);
  auto* u32 = module->Make<BuiltinTypeAnnotation>(fake_span, BuiltinType::kU32);
  auto* x = module->Make<NameDef>(fake_span, "x", /*definer=*/nullptr);
  auto* id = module->Make<NameDef>(fake_span, "id", /*definer=*/nullptr);
  auto* x_ref = module->Make<NameRef>(fake_span, "x", x);
  std::vector<Param*> params = {module->Make<Param>(x, u32)};
  std::vector<ParametricBinding*> parametrics;
  auto* function = module->Make<Function>(fake_span, id, parametrics, params,
                                          u32, x_ref, /*is_public=*/false);

  module->AddTop(function);

  ImportCache import_cache;

  // Populate a type information entity so we can resolve it.
  XLS_ASSERT_OK(import_cache.type_info_owner().New(module.get()).status());

  Interpreter interp(module.get(), /*typecheck=*/nullptr,
                     /*additional_search_paths=*/{},
                     /*import_cache=*/&import_cache);
  InterpValue mol = InterpValue::MakeU32(42);
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue result, interp.RunFunction("id", {mol}));
  EXPECT_TRUE(mol.Eq(result));
}

TEST(InterpreterTest, RunTokenIdentityFn) {
  absl::string_view program = "fn id(t: token) -> token { t }";
  ImportCache import_cache;
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(program, "test.x", "test", &import_cache,
                        /*additional_search_paths=*/{}));
  Interpreter interp(tm.module, /*typecheck=*/nullptr,
                     /*additional_search_paths=*/{},
                     /*import_cache=*/&import_cache);
  InterpValue tok = InterpValue::MakeToken();
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue result, interp.RunFunction("id", {tok}));
  EXPECT_TRUE(result.Eq(tok));
  EXPECT_EQ(result.ToString(), tok.ToString());
}

}  // namespace
}  // namespace xls::dslx
