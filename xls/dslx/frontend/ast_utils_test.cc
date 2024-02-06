// Copyright 2021 The XLS Authors
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
#include "xls/dslx/frontend/ast_utils.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/parser.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/scanner.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/type_system/type_info.h"

namespace xls::dslx {
namespace {

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
      Span::Fake(), BuiltinType::kU32, module.GetOrCreateBuiltinNameDef("u32"));
  Number* body =
      module.Make<Number>(Span::Fake(), "7", NumberKind::kOther, nullptr);
  Statement* body_stmt = module.Make<Statement>(body);

  Block* block =
      module.Make<Block>(Span::Fake(), std::vector<Statement*>{body_stmt},
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
  std::vector<ProcMember*> members;
  std::vector<ParametricBinding*> bindings;
  Proc* original_proc =
      module.Make<Proc>(Span::Fake(), name_def, config_name_def, next_name_def,
                        bindings, members, config, next, init,
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
          import_module->GetOrCreateBuiltinNameDef("u32"));
  Number* body = import_module->Make<Number>(Span::Fake(), "7",
                                             NumberKind::kOther, nullptr);
  Statement* body_stmt = import_module->Make<Statement>(body);

  Block* block = import_module->Make<Block>(Span::Fake(),
                                            std::vector<Statement*>{body_stmt},
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
  Proc* original_proc = import_module->Make<Proc>(
      Span::Fake(), name_def, config_name_def, next_name_def, bindings, members,
      config, next, init, /*is_public=*/true);
  XLS_ASSERT_OK(
      import_module->AddTop(original_proc, /*make_collision_error=*/nullptr));
  name_def->set_definer(original_proc);

  Module module("test_module", /*fs_path=*/std::nullopt);
  NameDef* module_def =
      module.Make<NameDef>(Span::Fake(), "import_module", nullptr);
  Import* import = module.Make<Import>(Span::Fake(), import_tokens, module_def,
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

TEST(ProcConfigIrConverterTest, BitVectorTests) {
  constexpr std::string_view kDslxText = R"(type MyType = u37;
type MyTuple = (u23, u1);
type MyArray = bits[22][33];
enum MyEnum : u7 { kA = 0, }
type MyEnumAlias = MyEnum;
struct MyStruct { x: u17 }

// A single struct which has many members of different types. This is an easy
// way of gathering together many TypeAnnotations and referring to them.
struct TheStruct {
  a: u32,
  b: s12,
  c: uN[111],
  d: sN[32],
  e: bits[312],
  f: bool,
  g: MyType,
  h: MyEnum,
  i: MyEnumAlias,
  j: (u23, u1),
  k: bits[22][33],
  l: MyStruct,
  m: MyTuple,
  n: MyArray,
}
 )";
  XLS_ASSERT_OK_AND_ASSIGN(auto module,
                           ParseModule(kDslxText, "fake_path.x", "the_module"));

  XLS_ASSERT_OK_AND_ASSIGN(std::vector<AstNode*> nodes,
                           CollectUnder(module.get(), /*want_types=*/true));
  StructDef* the_struct_def = nullptr;
  for (AstNode* node : nodes) {
    if (auto* struct_def = dynamic_cast<StructDef*>(node);
        struct_def != nullptr && struct_def->identifier() == "TheStruct") {
      the_struct_def = struct_def;
      break;
    }
  }
  ASSERT_NE(the_struct_def, nullptr);
  auto get_type_metadata = [&](std::string_view name) {
    for (const StructMember& member : the_struct_def->members()) {
      if (member.name == name) {
        return ExtractBitVectorMetadata(member.type);
      }
    }
    XLS_LOG(FATAL) << "Unknown field: " << name;
  };

  EXPECT_EQ(std::get<int64_t>(get_type_metadata("a")->bit_count), 32);
  EXPECT_FALSE(get_type_metadata("a")->is_signed);
  EXPECT_EQ(get_type_metadata("a")->kind, BitVectorKind::kBitType);

  EXPECT_EQ(std::get<int64_t>(get_type_metadata("b")->bit_count), 12);
  EXPECT_TRUE(get_type_metadata("b")->is_signed);
  EXPECT_EQ(get_type_metadata("b")->kind, BitVectorKind::kBitType);

  EXPECT_EQ(std::get<Expr*>(get_type_metadata("c")->bit_count)->ToString(),
            "111");
  EXPECT_FALSE(get_type_metadata("c")->is_signed);
  EXPECT_EQ(get_type_metadata("c")->kind, BitVectorKind::kBitType);

  EXPECT_EQ(std::get<Expr*>(get_type_metadata("d")->bit_count)->ToString(),
            "32");
  EXPECT_TRUE(get_type_metadata("d")->is_signed);
  EXPECT_EQ(get_type_metadata("d")->kind, BitVectorKind::kBitType);

  EXPECT_EQ(std::get<Expr*>(get_type_metadata("e")->bit_count)->ToString(),
            "312");
  EXPECT_FALSE(get_type_metadata("e")->is_signed);
  EXPECT_EQ(get_type_metadata("e")->kind, BitVectorKind::kBitType);

  EXPECT_EQ(std::get<int64_t>(get_type_metadata("f")->bit_count), 1);
  EXPECT_FALSE(get_type_metadata("f")->is_signed);
  EXPECT_EQ(get_type_metadata("f")->kind, BitVectorKind::kBitType);

  EXPECT_EQ(std::get<int64_t>(get_type_metadata("g")->bit_count), 37);
  EXPECT_FALSE(get_type_metadata("g")->is_signed);
  EXPECT_EQ(get_type_metadata("g")->kind, BitVectorKind::kBitTypeAlias);

  EXPECT_EQ(std::get<int64_t>(get_type_metadata("h")->bit_count), 7);
  EXPECT_FALSE(get_type_metadata("h")->is_signed);
  EXPECT_EQ(get_type_metadata("h")->kind, BitVectorKind::kEnumType);

  EXPECT_EQ(std::get<int64_t>(get_type_metadata("i")->bit_count), 7);
  EXPECT_FALSE(get_type_metadata("i")->is_signed);
  EXPECT_EQ(get_type_metadata("i")->kind, BitVectorKind::kEnumTypeAlias);

  EXPECT_FALSE(get_type_metadata("j").has_value());
  EXPECT_FALSE(get_type_metadata("k").has_value());
  EXPECT_FALSE(get_type_metadata("l").has_value());
  EXPECT_FALSE(get_type_metadata("m").has_value());
  EXPECT_FALSE(get_type_metadata("n").has_value());
}

// Tests that the ResolveLocalStructDef can see through transitive aliases.
TEST(ResolveLocalStructDef, StructDefInTransitiveAlias) {
  const std::string kProgram = R"(struct S {
}

type MyS1 = S;
type MyS2 = MyS1;
)";
  Scanner scanner("test.x", kProgram);
  Parser parser("test", &scanner);
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> module,
                           parser.ParseModule());

  std::optional<ModuleMember*> maybe_my_s = module->FindMemberWithName("MyS2");
  ASSERT_TRUE(maybe_my_s.has_value());
  auto* type_alias = std::get<TypeAlias*>(*maybe_my_s.value());
  auto* aliased = type_alias->type_annotation();
  auto* type_ref_type_annotation =
      dynamic_cast<TypeRefTypeAnnotation*>(aliased);
  ASSERT_NE(type_ref_type_annotation, nullptr);
  TypeDefinition td = type_ref_type_annotation->type_ref()->type_definition();

  XLS_ASSERT_OK_AND_ASSIGN(StructDef * resolved, ResolveLocalStructDef(td));

  std::optional<ModuleMember*> maybe_s = module->FindMemberWithName("S");
  ASSERT_TRUE(maybe_s.has_value());
  auto* s = std::get<StructDef*>(*maybe_s.value());
  EXPECT_EQ(s, resolved);
}

}  // namespace
}  // namespace xls::dslx
