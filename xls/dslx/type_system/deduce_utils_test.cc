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

#include "xls/dslx/type_system/deduce_utils.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "xls/common/casts.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/bindings.h"
#include "xls/dslx/frontend/parser.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/proc.h"
#include "xls/dslx/frontend/scanner.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"

namespace xls::dslx {
namespace {

TEST(DeduceUtilsTest, ValidateNumber) {
  Scanner scanner("test.x", "42 256");
  Parser parser("test", &scanner);
  Bindings bindings;
  XLS_ASSERT_OK_AND_ASSIGN(Expr * e, parser.ParseExpression(bindings));
  auto* ft = down_cast<Number*>(e);

  XLS_ASSERT_OK_AND_ASSIGN(e, parser.ParseExpression(bindings));
  auto* tfs = down_cast<Number*>(e);

  auto u8 = BitsType::MakeU8();
  XLS_ASSERT_OK(ValidateNumber(*ft, *u8));

  // 256 does not fit in a u8.
  ASSERT_THAT(ValidateNumber(*tfs, *u8),
              status_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  testing::HasSubstr(
                      "Value '256' does not fit in the bitwidth of a uN[8]")));
}

TEST(ProcConfigIrConverterTest, ResolveProcNameRef) {
  Module module("test_module", /*fs_path=*/std::nullopt);
  NameDef* name_def = module.Make<NameDef>(Span::Fake(), "proc_name", nullptr);
  NameDef* config_name_def =
      module.Make<NameDef>(Span::Fake(), "config_name", nullptr);
  NameDef* next_name_def =
      module.Make<NameDef>(Span::Fake(), "next_name", nullptr);
  NameDef* init_name_def =
      module.Make<NameDef>(Span::Fake(), "init_name", nullptr);
  BuiltinTypeAnnotation* return_type = module.Make<BuiltinTypeAnnotation>(
      Span::Fake(), BuiltinType::kU32,
      module.GetOrCreateBuiltinNameDef(dslx::BuiltinType::kU32));
  Number* body =
      module.Make<Number>(Span::Fake(), "7", NumberKind::kOther, nullptr);
  Statement* body_stmt = module.Make<Statement>(body);

  StatementBlock* block = module.Make<StatementBlock>(
      Span::Fake(), std::vector<Statement*>{body_stmt},
      /*trailing_semi=*/false);

  Function* config = module.Make<Function>(
      Span::Fake(), config_name_def,
      /*parametric_bindings=*/std::vector<ParametricBinding*>(),
      /*params=*/std::vector<Param*>(), return_type, block,
      FunctionTag::kProcConfig, /*is_public=*/true);
  Function* next = module.Make<Function>(
      Span::Fake(), next_name_def,
      /*parametric_bindings=*/std::vector<ParametricBinding*>(),
      /*params=*/std::vector<Param*>(), return_type, block,
      FunctionTag::kProcNext, /*is_public=*/true);
  Function* init = module.Make<Function>(
      Span::Fake(), init_name_def,
      /*parametric_bindings=*/std::vector<ParametricBinding*>(),
      /*params=*/std::vector<Param*>(), return_type, block,
      FunctionTag::kProcNext, /*is_public=*/true);
  const std::vector<ParametricBinding*> bindings;
  const ProcLikeBody proc_body{
      .stmts = {},
      .config = config,
      .next = next,
      .init = init,
      .members = {},
  };
  Proc* original_proc =
      module.Make<Proc>(Span::Fake(), /*body_span=*/Span::Fake(), name_def,
                        /*parametric_bindings=*/bindings, proc_body,
                        /*is_public=*/true);
  XLS_ASSERT_OK(module.AddTop(original_proc, /*make_collision_error=*/nullptr));
  name_def->set_definer(original_proc);

  TypeInfoOwner type_info_owner;
  XLS_ASSERT_OK_AND_ASSIGN(TypeInfo * type_info, type_info_owner.New(&module));

  NameRef* name_ref = module.Make<NameRef>(Span::Fake(), "proc_name", name_def);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * p, ResolveProc(name_ref, type_info));
  EXPECT_EQ(p, original_proc);
}

TEST(ProcConfigIrConverterTest, ResolveProcColonRef) {
  std::vector<std::string> import_tokens{"robs", "dslx", "import_module"};
  ImportTokens subject(import_tokens);
  ModuleInfo module_info(
      std::make_unique<Module>("import_module", /*fs_path=*/std::nullopt),
      /*type_info=*/nullptr, "robs/dslx/import_module.x");
  Module* import_module = &module_info.module();

  NameDef* name_def =
      import_module->Make<NameDef>(Span::Fake(), "proc_name", nullptr);
  NameDef* config_name_def =
      import_module->Make<NameDef>(Span::Fake(), "config_name", nullptr);
  NameDef* next_name_def =
      import_module->Make<NameDef>(Span::Fake(), "next_name", nullptr);
  NameDef* init_name_def =
      import_module->Make<NameDef>(Span::Fake(), "init_name", nullptr);
  BuiltinTypeAnnotation* return_type =
      import_module->Make<BuiltinTypeAnnotation>(
          Span::Fake(), BuiltinType::kU32,
          import_module->GetOrCreateBuiltinNameDef(dslx::BuiltinType::kU32));
  Number* body = import_module->Make<Number>(Span::Fake(), "7",
                                             NumberKind::kOther, nullptr);
  Statement* body_stmt = import_module->Make<Statement>(body);

  StatementBlock* block = import_module->Make<StatementBlock>(
      Span::Fake(), std::vector<Statement*>{body_stmt},
      /*trailing_semi=*/false);

  Function* config = import_module->Make<Function>(
      Span::Fake(), config_name_def,
      /*parametric_bindings=*/std::vector<ParametricBinding*>(),
      /*params=*/std::vector<Param*>(), return_type, block,
      FunctionTag::kProcConfig, /*is_public=*/true);
  Function* next = import_module->Make<Function>(
      Span::Fake(), next_name_def,
      /*parametric_bindings=*/std::vector<ParametricBinding*>(),
      /*params=*/std::vector<Param*>(), return_type, block,
      FunctionTag::kProcNext, /*is_public=*/true);
  Function* init = import_module->Make<Function>(
      Span::Fake(), init_name_def,
      /*parametric_bindings=*/std::vector<ParametricBinding*>(),
      /*params=*/std::vector<Param*>(), return_type, block,
      FunctionTag::kProcInit, /*is_public=*/true);
  std::vector<ProcMember*> members;
  std::vector<ParametricBinding*> bindings;
  ProcLikeBody proc_body = {
      .stmts = {},
      .config = config,
      .next = next,
      .init = init,
      .members = members,
  };
  Proc* original_proc = import_module->Make<Proc>(
      Span::Fake(), /*body_span=*/Span::Fake(), name_def, bindings, proc_body,
      /*is_public=*/true);
  XLS_ASSERT_OK(
      import_module->AddTop(original_proc, /*make_collision_error=*/nullptr));
  name_def->set_definer(original_proc);

  Module module("test_module", /*fs_path=*/std::nullopt);
  NameDef* module_def =
      module.Make<NameDef>(Span::Fake(), "import_module", nullptr);
  Import* import = module.Make<Import>(Span::Fake(), import_tokens, *module_def,
                                       std::nullopt);
  module_def->set_definer(import);
  NameRef* module_ref =
      module.Make<NameRef>(Span::Fake(), "import_module", module_def);
  ColonRef* colon_ref =
      module.Make<ColonRef>(Span::Fake(), module_ref, "proc_name");

  TypeInfoOwner type_info_owner;
  XLS_ASSERT_OK_AND_ASSIGN(TypeInfo * type_info, type_info_owner.New(&module));
  XLS_ASSERT_OK_AND_ASSIGN(TypeInfo * imported_type_info,
                           type_info_owner.New(import_module));
  type_info->AddImport(import, import_module, imported_type_info);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * p, ResolveProc(colon_ref, type_info));
  EXPECT_EQ(p, original_proc);
}

}  // namespace
}  // namespace xls::dslx
