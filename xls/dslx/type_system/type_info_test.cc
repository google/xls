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
#include <variant>
#include <vector>

#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/substitute.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/frontend/aggregate_construction.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/builtin_stubs_utils.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/parser.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/scanner.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/typecheck_test_utils.h"

namespace xls::dslx {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::UnorderedElementsAre;

std::vector<ParametricEnv> GetCalleeBindings(const TypeInfo& ti, Function* f,
                                             bool unique) {
  std::vector<InvocationCalleeData> invocations =
      unique ? ti.GetUniqueInvocationCalleeData(f)
             : ti.GetAllInvocationCalleeData(f);
  std::vector<ParametricEnv> all_callee_bindings;
  all_callee_bindings.reserve(invocations.size());
  for (const InvocationCalleeData& data : invocations) {
    all_callee_bindings.push_back(data.callee_bindings);
  }
  return all_callee_bindings;
}

TEST(TypeInfoTest, Instantiate) {
  FileTable file_table;
  Module module("test", /*fs_path=*/std::nullopt, file_table);
  TypeInfoOwner owner;
  XLS_ASSERT_OK_AND_ASSIGN(TypeInfo * type_info,
                           owner.New(file_table, TypeInfo::kRootName));
  EXPECT_EQ(type_info->parent(), nullptr);
}

TEST(TypeInfoTest, SumConstructorSubjectsAreInherited) {
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckedModule tm,
                           ParseAndTypecheck(R"(
enum E { Unit, Tuple(u32) }
const UNIT = E::Unit;
const TUPLE = E::Tuple(u32:0);
)",
                                             "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(SumDef * sum,
                           tm.module->GetMemberOrError<SumDef>("E"));
  XLS_ASSERT_OK_AND_ASSIGN(ConstantDef * unit,
                           tm.module->GetConstantDef("UNIT"));
  XLS_ASSERT_OK_AND_ASSIGN(ConstantDef * tuple,
                           tm.module->GetConstantDef("TUPLE"));
  auto* unit_ref = dynamic_cast<const ColonRef*>(unit->value());
  auto* invocation = dynamic_cast<const Invocation*>(tuple->value());
  ASSERT_NE(unit_ref, nullptr);
  ASSERT_NE(invocation, nullptr);
  auto* tuple_ref = dynamic_cast<const ColonRef*>(invocation->callee());
  ASSERT_NE(tuple_ref, nullptr);

  TypeInfoOwner owner;
  XLS_ASSERT_OK_AND_ASSIGN(TypeInfo * child, owner.New(import_data.file_table(),
                                                       "child", tm.type_info));
  XLS_ASSERT_OK_AND_ASSIGN(
      TypeInfo * grandchild,
      owner.New(import_data.file_table(), "grandchild", child));
  for (const TypeInfo* type_info : {tm.type_info, child, grandchild}) {
    XLS_ASSERT_OK_AND_ASSIGN(TypeInfo::ResolvedColonRefSubject subject,
                             type_info->GetResolvedColonRefSubject(tuple_ref));
    EXPECT_EQ(std::get<SumDef*>(subject), sum);
    EXPECT_TRUE(type_info->IsSumConstructor(unit_ref));
    EXPECT_TRUE(type_info->IsSumConstructor(invocation));
  }
}

TEST(TypeInfoTest, BoundConstructionFollowsCallerContextAndParentLookup) {
  FileTable file_table;
  Scanner scanner(file_table, file_table.GetOrCreate("test.x"), R"(
    #![feature(generics)]
    struct A { x: u8, y: u8 }
    struct B { y: u8, x: u8 }
    fn make<T: type>() -> T { T { y: u8:2, x: u8:1 } }
)");
  Parser parser("test", &scanner);
  XLS_ASSERT_OK_AND_ASSIGN(auto module, parser.ParseModule());
  XLS_ASSERT_OK_AND_ASSIGN(StructDef * a,
                           module->GetMemberOrError<StructDef>("A"));
  XLS_ASSERT_OK_AND_ASSIGN(StructDef * b,
                           module->GetMemberOrError<StructDef>("B"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * make,
                           module->GetMemberOrError<Function>("make"));
  const auto* expression = absl::down_cast<const StructInstance*>(
      ToAstNode(make->body()->statements()[0]->wrapped()));
  const BoundConstruction bound_a(expression, a,
                                  NamedFieldCorrespondence{1, 0});
  const BoundConstruction bound_b(expression, b,
                                  NamedFieldCorrespondence{0, 1});
  TypeInfoOwner owner;
  XLS_ASSERT_OK_AND_ASSIGN(TypeInfo * root,
                           owner.New(file_table, TypeInfo::kRootName));
  XLS_ASSERT_OK_AND_ASSIGN(TypeInfo * caller_a,
                           owner.New(file_table, "caller_a", root));
  XLS_ASSERT_OK_AND_ASSIGN(TypeInfo * caller_b,
                           owner.New(file_table, "caller_b", root));
  XLS_ASSERT_OK_AND_ASSIGN(TypeInfo * nested,
                           owner.New(file_table, "nested", caller_a));
  caller_a->SetBoundConstruction(expression, &bound_a);
  caller_b->SetBoundConstruction(expression, &bound_b);
  EXPECT_EQ(root->GetBoundConstruction(expression), std::nullopt);
  ASSERT_EQ(nested->GetBoundConstruction(expression), &bound_a);
  ASSERT_EQ(caller_b->GetBoundConstruction(expression), &bound_b);
  EXPECT_EQ((*nested->GetBoundConstruction(expression))->GetMember(0),
            expression->members()[1].second);
  EXPECT_EQ((*caller_b->GetBoundConstruction(expression))->GetMember(0),
            expression->members()[0].second);
  nested->SetBoundConstruction(expression, &bound_b);
  EXPECT_EQ(nested->GetBoundConstruction(expression), &bound_b);
  EXPECT_EQ(caller_a->GetBoundConstruction(expression), &bound_a);
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

  const Invocation* invoke_p = absl::down_cast<const Invocation*>(
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
                  *invoke_p, /*callee=*/nullptr, /*caller=*/main,
                  bad_caller_env, valid_callee_env,
                  /*derived_type_info=*/nullptr),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("caller `main` given env with key `A` not "
                                 "present in parametric keys: {}")));
}

TEST(TypeInfoTest, GetUniqueInvocationCalleeDataNonParametric) {
  const std::string kInvocation = R"(
fn f() -> u32 { u32:42 }
fn main() -> u32 { f() }
)";
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(kInvocation));

  std::optional<Function*> f = result.tm.module->GetFunction("f");
  ASSERT_TRUE(f.has_value());
  auto f_invocations = result.tm.type_info->GetUniqueInvocationCalleeData(*f);
  // No parametric envs
  EXPECT_TRUE(f_invocations.empty());

  std::optional<Function*> main = result.tm.module->GetFunction("main");
  ASSERT_TRUE(main.has_value());
  auto main_invocations =
      result.tm.type_info->GetUniqueInvocationCalleeData(*main);
  EXPECT_TRUE(main_invocations.empty());
}

TEST(TypeInfoTest, GetUniqueInvocationCalleeDataOneParametricCall) {
  const std::string kInvocation = R"(
fn f<N: u32>() -> u32 { u32:42 }
fn main() -> u32 { f<u32:0>() }
)";
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(kInvocation));

  std::optional<Function*> f = result.tm.module->GetFunction("f");
  ASSERT_TRUE(f.has_value());

  auto invocations = result.tm.type_info->GetUniqueInvocationCalleeData(*f);
  EXPECT_EQ(invocations.size(), 1);
}

TEST(TypeInfoTest, FunctionCallGraphBasic) {
  ImportData import_data = CreateImportDataForTest();
  const std::string kProgram = R"(
fn leaf(x: u32) -> u32 { x }

fn caller(x: u32) -> u32 {
  leaf(x) + leaf(x)
}

fn unused(x: u32) -> u32 { x }
)";
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "graph.x", "graph", &import_data));

  auto graph = tm.type_info->GetFunctionCallGraph(tm.module);
  const Function* leaf = tm.module->GetFunction("leaf").value();
  const Function* caller = tm.module->GetFunction("caller").value();
  ASSERT_TRUE(graph.contains(caller));
  EXPECT_THAT(graph.at(caller), ElementsAre(leaf));
  ASSERT_TRUE(graph.contains(leaf));
  EXPECT_TRUE(graph.at(leaf).empty());

  const Function* unused = tm.module->GetFunction("unused").value();
  ASSERT_TRUE(graph.contains(unused));
  EXPECT_TRUE(graph.at(unused).empty());
}

TEST(TypeInfoTest, FunctionCallGraphHandlesMapAndIncludesBuiltins) {
  ImportData import_data = CreateImportDataForTest();
  const std::string kProgram = R"(
fn inc(x: u32) -> u32 { x + u32:1 }

fn apply(xs: u32[3]) -> u32[3] {
  map(xs, inc)
}

fn uses_builtin(x: u32) -> u32 { clz(x) }
)";
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "graph_map.x", "graph_map", &import_data,
                        nullptr));

  auto graph = tm.type_info->GetFunctionCallGraph(tm.module);
  const Function* inc = tm.module->GetFunction("inc").value();
  const Function* apply = tm.module->GetFunction("apply").value();
  ASSERT_TRUE(graph.contains(apply));
  EXPECT_THAT(graph.at(apply), ElementsAre(inc));

  const Function* uses_builtin = tm.module->GetFunction("uses_builtin").value();
  ASSERT_TRUE(graph.contains(uses_builtin));
  const std::vector<const Function*>& uses_builtin_callees =
      graph.at(uses_builtin);
  ASSERT_THAT(uses_builtin_callees, testing::SizeIs(1));
  const Function* builtin_callee = uses_builtin_callees.front();
  EXPECT_EQ(builtin_callee->identifier(), "clz");
  EXPECT_EQ(builtin_callee->owner()->name(), kBuiltinStubsModuleName);
}

TEST(TypeInfoTest, FunctionCallGraphHandlesIntermoduleInvocations) {
  ImportData import_data = CreateImportDataForTest();
  const std::string kImported = R"(
pub fn increment(x: u32) -> u32 { x + u32:1 }
)";
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule imported_tm,
      ParseAndTypecheck(kImported, "imported.x", "imported", &import_data));

  const std::string kProgram = R"(
import imported;

fn local_call(x: u32) -> u32 { imported::increment(x) }

fn entry(x: u32) -> u32 { local_call(x) }
)";
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "main.x", "main", &import_data));

  auto graph = tm.type_info->GetFunctionCallGraph(tm.module);
  const Function* local_call = tm.module->GetFunction("local_call").value();
  const Function* entry = tm.module->GetFunction("entry").value();
  const Function* imported_increment =
      imported_tm.module->GetFunction("increment").value();

  ASSERT_TRUE(graph.contains(local_call));
  EXPECT_THAT(graph.at(local_call), ElementsAre(imported_increment));
  ASSERT_TRUE(graph.contains(entry));
  EXPECT_THAT(graph.at(entry), ElementsAre(local_call));
}

TEST(TypeInfoTest, FunctionCallGraphIncludesProcSpawns) {
  ImportData import_data = CreateImportDataForTest();
  const std::string kProgram = R"(
proc worker<N: u32> {
  value: uN[N];
  init { zero!<uN[N]>() }
  config(value: uN[N]) { (value,) }
  next(state: uN[N]) { state }
}

proc main {
  init { () }
  config() {
    spawn worker<u32:8>(u8:0);
    ()
  }
  next(state: ()) { state }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckedModule tm,
                           ParseAndTypecheck(kProgram, "spawn_graph.x",
                                             "spawn_graph", &import_data));

  auto graph = tm.type_info->GetFunctionCallGraph(tm.module);
  const Function* main_config = tm.module->GetFunction("main.config").value();
  const Function* worker_config =
      tm.module->GetFunction("worker.config").value();
  const Function* worker_init = tm.module->GetFunction("worker.init").value();
  const Function* worker_next = tm.module->GetFunction("worker.next").value();

  ASSERT_TRUE(graph.contains(main_config));
  EXPECT_THAT(graph.at(main_config),
              UnorderedElementsAre(worker_config, worker_init, worker_next));
}

TEST(TypeInfoTest, GetUniqueInvocationCalleeDataMultipleParametricCalls) {
  const std::string kInvocation = R"(
fn f<N: u32>() -> u32 { u32:42 }
fn main() -> u32 { f<u32:0>() + f<u32:0>() + f<u32:1>() }
)";
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(kInvocation));

  std::optional<Function*> f = result.tm.module->GetFunction("f");
  ASSERT_TRUE(f.has_value());

  std::vector<ParametricEnv> all_callee_bindings =
      GetCalleeBindings(*result.tm.type_info, *f, /*unique=*/true);
  EXPECT_THAT(all_callee_bindings,
              UnorderedElementsAre(
                  ParametricEnv(absl::flat_hash_map<std::string, InterpValue>{
                      {"N", InterpValue::MakeU32(0)},
                  }),
                  ParametricEnv(absl::flat_hash_map<std::string, InterpValue>{
                      {"N", InterpValue::MakeU32(1)},
                  })));
}

TEST(TypeInfoTest,
     GetUniqueInvocationCalleeDataMultipleAndRepeatedParametricCalls) {
  const std::string kInvocation = R"(
fn f<N: u32>() -> u32 { u32:42 }
fn main() -> u32 { f<u32:0>() + f<u32:1>() }
fn main2() -> u32 { f<u32:1>() + f<u32:0>() + f<u32:2>() }
)";
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(kInvocation));

  std::optional<Function*> f = result.tm.module->GetFunction("f");
  ASSERT_TRUE(f.has_value());

  auto invocations = result.tm.type_info->GetUniqueInvocationCalleeData(*f);
  EXPECT_EQ(invocations.size(), 3);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(invocations[i].callee_bindings,
              ParametricEnv(absl::flat_hash_map<std::string, InterpValue>{
                  {"N", InterpValue::MakeU32(i)},
              }));
  }
}

TEST(TypeInfoTest, GetAllInvocationCalleeDataMultipleParametricCalls) {
  const std::string kInvocation = R"(
fn f<N: u32>() -> u32 { u32:42 }
fn main() -> u32 { f<u32:0>() + f<u32:0>() + f<u32:1>() }
)";
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(kInvocation));

  std::optional<Function*> f = result.tm.module->GetFunction("f");
  ASSERT_TRUE(f.has_value());

  std::vector<ParametricEnv> all_callee_bindings =
      GetCalleeBindings(*result.tm.type_info, *f, /*unique=*/false);
  EXPECT_THAT(all_callee_bindings,
              UnorderedElementsAre(
                  ParametricEnv(absl::flat_hash_map<std::string, InterpValue>{{
                      "N",
                      InterpValue::MakeU32(0),
                  }}),
                  ParametricEnv(absl::flat_hash_map<std::string, InterpValue>{{
                      "N",
                      InterpValue::MakeU32(0),
                  }}),
                  ParametricEnv(absl::flat_hash_map<std::string, InterpValue>{{
                      "N",
                      InterpValue::MakeU32(1),
                  }})));

  auto unique_invocations =
      result.tm.type_info->GetUniqueInvocationCalleeData(*f);
  ASSERT_EQ(unique_invocations.size(), 2);

  std::vector<ParametricEnv> unique_callee_bindings;
  unique_callee_bindings.reserve(unique_invocations.size());
  for (const InvocationCalleeData& data : unique_invocations) {
    unique_callee_bindings.push_back(data.callee_bindings);
  }

  EXPECT_THAT(unique_callee_bindings,
              UnorderedElementsAre(
                  ParametricEnv(absl::flat_hash_map<std::string, InterpValue>{{
                      "N",
                      InterpValue::MakeU32(0),
                  }}),
                  ParametricEnv(absl::flat_hash_map<std::string, InterpValue>{{
                      "N",
                      InterpValue::MakeU32(1),
                  }})));
}

TEST(TypeInfoTest, GetUniqueInvocationCalleeDataParametricProc) {
  const std::string kInvocation = R"(
proc spawnee<N: u32>{
  init { }
  config() {()}
  next(state: ()) { state }
}

proc main {
  init { }
  config() {spawn spawnee<u32:0>(); spawn spawnee<u32:1>(); () }
  next(state: ()) { state }
}

proc main2 {
  init { }
  config() {
    spawn spawnee<u32:1>();
    spawn spawnee<u32:0>();
    spawn spawnee<u32:2>();
    ()}
  next(state: ()) { state }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(kInvocation));

  std::optional<Function*> next_fn =
      result.tm.module->GetFunction("spawnee.next");
  ASSERT_TRUE(next_fn.has_value());

  auto next_invocations =
      result.tm.type_info->GetUniqueInvocationCalleeData(*next_fn);
  EXPECT_EQ(next_invocations.size(), 3);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(next_invocations[i].callee_bindings,
              ParametricEnv(absl::flat_hash_map<std::string, InterpValue>{
                  {"N", InterpValue::MakeU32(i)},
              }));
    EXPECT_EQ(next_invocations[i].invocation->args().size(), 1);
    EXPECT_EQ(next_invocations[i].invocation->args()[0]->ToString(),
              absl::Substitute("spawnee.init<u32:$0>()", i));
  }

  std::optional<Function*> config_fn =
      result.tm.module->GetFunction("spawnee.config");
  auto config_invocations =
      result.tm.type_info->GetUniqueInvocationCalleeData(*config_fn);
  EXPECT_EQ(config_invocations.size(), 3);
  for (auto invocation : config_invocations) {
    // The config function in this test has no arguments.
    EXPECT_EQ(invocation.invocation->args().size(), 0);
  }
}

TEST(TypeInfoTest, GetUniqueInvocationCalleeDataProcWithConfigArgs) {
  const std::string kInvocation = R"(
proc spawnee<N: u32>{
  a: uN[N];
  init { zero!<uN[N]>() }
  config(x: uN[N]) {(x,)}
  next(state: uN[N]) { state }
}

proc main {
  init { }
  config() {
    spawn spawnee<u32:8>(u8:0);
    spawn spawnee<u32:16>(u16:1);
    ()
  }
  next(state: ()) { state }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(kInvocation));

  std::optional<Function*> next_fn =
      result.tm.module->GetFunction("spawnee.next");
  ASSERT_TRUE(next_fn.has_value());

  auto next_invocations =
      result.tm.type_info->GetUniqueInvocationCalleeData(*next_fn);
  EXPECT_EQ(next_invocations.size(), 2);
  EXPECT_EQ(next_invocations[0].callee_bindings,
            ParametricEnv(absl::flat_hash_map<std::string, InterpValue>{
                {"N", InterpValue::MakeU32(8)},
            }));
  EXPECT_EQ(next_invocations[1].callee_bindings,
            ParametricEnv(absl::flat_hash_map<std::string, InterpValue>{
                {"N", InterpValue::MakeU32(16)},
            }));

  std::optional<Function*> config_fn =
      result.tm.module->GetFunction("spawnee.config");
  auto config_invocations =
      result.tm.type_info->GetUniqueInvocationCalleeData(*config_fn);
  EXPECT_EQ(config_invocations.size(), 2);
  auto args8 = config_invocations[0].invocation->args();
  EXPECT_EQ(args8.size(), 1);
  EXPECT_EQ(args8[0]->ToString(), "u8:0");

  auto args16 = config_invocations[1].invocation->args();
  EXPECT_EQ(args16.size(), 1);
  EXPECT_EQ(args16[0]->ToString(), "u16:1");
}

}  // namespace
}  // namespace xls::dslx
