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
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/substitute.h"
#include "xls/common/casts.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/builtin_stubs_utils.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
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
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, Typecheck(kInvocation));

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
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, Typecheck(kInvocation));

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

  auto graph = tm.type_info->GetFunctionCallGraph();
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
                        nullptr, {TypeInferenceVersion::kVersion2}));

  auto graph = tm.type_info->GetFunctionCallGraph();
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

  auto graph = tm.type_info->GetFunctionCallGraph();
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

  auto graph = tm.type_info->GetFunctionCallGraph();
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
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, Typecheck(kInvocation));

  std::optional<Function*> f = result.tm.module->GetFunction("f");
  ASSERT_TRUE(f.has_value());

  auto invocations = result.tm.type_info->GetUniqueInvocationCalleeData(*f);
  // There are two unique invocations of f.
  EXPECT_EQ(invocations.size(), 2);
  // They should be in original-call order.
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(invocations[i].callee_bindings,
              ParametricEnv(absl::flat_hash_map<std::string, InterpValue>{
                  {"N", InterpValue::MakeU32(i)},
              }));
  }
}

TEST(TypeInfoTest,
     GetUniqueInvocationCalleeDataMultipleAndRepeatedParametricCalls) {
  const std::string kInvocation = R"(
fn f<N: u32>() -> u32 { u32:42 }
fn main() -> u32 { f<u32:0>() + f<u32:1>() }
fn main2() -> u32 { f<u32:1>() + f<u32:0>() + f<u32:2>() }
)";
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, Typecheck(kInvocation));

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
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, Typecheck(kInvocation));

  std::optional<Function*> f = result.tm.module->GetFunction("f");
  ASSERT_TRUE(f.has_value());

  auto all_invocations = result.tm.type_info->GetAllInvocationCalleeData(*f);
  ASSERT_EQ(all_invocations.size(), 3);
  EXPECT_EQ(all_invocations[0].callee_bindings,
            ParametricEnv(absl::flat_hash_map<std::string, InterpValue>{{
                "N",
                InterpValue::MakeU32(0),
            }}));
  EXPECT_EQ(all_invocations[1].callee_bindings,
            ParametricEnv(absl::flat_hash_map<std::string, InterpValue>{{
                "N",
                InterpValue::MakeU32(0),
            }}));
  EXPECT_EQ(all_invocations[2].callee_bindings,
            ParametricEnv(absl::flat_hash_map<std::string, InterpValue>{{
                "N",
                InterpValue::MakeU32(1),
            }}));
  EXPECT_NE(all_invocations[0].invocation, all_invocations[1].invocation);

  auto unique_invocations =
      result.tm.type_info->GetUniqueInvocationCalleeData(*f);
  ASSERT_EQ(unique_invocations.size(), 2);
  EXPECT_EQ(unique_invocations[0].callee_bindings,
            ParametricEnv(absl::flat_hash_map<std::string, InterpValue>{{
                "N",
                InterpValue::MakeU32(0),
            }}));
  EXPECT_EQ(unique_invocations[1].callee_bindings,
            ParametricEnv(absl::flat_hash_map<std::string, InterpValue>{{
                "N",
                InterpValue::MakeU32(1),
            }}));
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
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, Typecheck(kInvocation));

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
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, Typecheck(kInvocation));

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
