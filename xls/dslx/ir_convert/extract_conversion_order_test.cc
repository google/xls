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

#include "xls/dslx/ir_convert/extract_conversion_order.h"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/proc.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/type_system/parametric_env.h"

namespace xls::dslx {
namespace {

using ::testing::ElementsAre;

MATCHER_P2(IdentifierAndProcId, identifier_matcher, proc_id_matcher, "") {
  if (!arg.proc_id().has_value()) {
    return false;
  }
  return ExplainMatchResult(identifier_matcher, arg.f()->identifier(),
                            result_listener) &&
         ExplainMatchResult(proc_id_matcher, arg.proc_id()->ToString(),
                            result_listener);
}

TEST(ExtractConversionOrderTest, SimpleLinearCallgraph) {
  constexpr std::string_view kProgram = R"(
fn g() -> u32 { u32:42 }
fn f() -> u32 { g() }
fn main() -> u32 { f() }
)";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<ConversionRecord> order,
                           GetOrder(tm.module, tm.type_info));
  ASSERT_EQ(3, order.size());
  EXPECT_EQ(order[0].f()->identifier(), "g");
  EXPECT_EQ(order[1].f()->identifier(), "f");
  EXPECT_EQ(order[2].f()->identifier(), "main");
}

TEST(ExtractConversionOrderTest, Parametric) {
  constexpr std::string_view kProgram = R"(
fn f<N: u32>(x: bits[N]) -> u32 { N }
fn main() -> u32 { f(u2:0) }
)";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<ConversionRecord> order,
                           GetOrder(tm.module, tm.type_info));
  ASSERT_EQ(2, order.size());
  EXPECT_EQ(order[0].f()->identifier(), "f");
  EXPECT_EQ(order[0].parametric_env(),
            ParametricEnv(absl::flat_hash_map<std::string, InterpValue>{
                {"N", InterpValue::MakeUBits(/*bit_count=*/32, /*value=*/2)}}));
  EXPECT_EQ(order[1].f()->identifier(), "main");
  EXPECT_EQ(order[1].parametric_env(), ParametricEnv());
}

TEST(ExtractConversionOrderTest, TransitiveParametric) {
  constexpr std::string_view kProgram = R"(
fn g<M: u32>(x: bits[M]) -> u32 { M }
fn f<N: u32>(x: bits[N]) -> u32 { g(x) }
fn main() -> u32 { f(u2:0) }
)";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<ConversionRecord> order,
                           GetOrder(tm.module, tm.type_info));
  ASSERT_EQ(3, order.size());
  EXPECT_EQ(order[0].f()->identifier(), "g");
  order[0].parametric_env(),
      ParametricEnv(absl::flat_hash_map<std::string, InterpValue>{
          {"M", InterpValue::MakeUBits(/*bit_count=*/32, /*value=*/2)}});
  EXPECT_EQ(order[1].f()->identifier(), "f");
  EXPECT_EQ(order[1].parametric_env(),
            ParametricEnv(absl::flat_hash_map<std::string, InterpValue>{
                {"N", InterpValue::MakeUBits(/*bit_count=*/32, /*value=*/2)}}));
  EXPECT_EQ(order[2].f()->identifier(), "main");
  EXPECT_EQ(order[2].parametric_env(), ParametricEnv());
}

TEST(ExtractConversionOrderTest, UseTreeEntryCallInParametric) {
  constexpr std::string_view kProgram = R"(#![feature(use_syntax)]
use std::is_pow2;
fn f<N: u32>(x: bits[N]) -> bool { is_pow2(x) }
fn main() -> bool { f(u2:3) }
)";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<ConversionRecord> order,
                           GetOrder(tm.module, tm.type_info));
  ASSERT_EQ(3, order.size());
  EXPECT_EQ(order[0].f()->identifier(), "is_pow2");
  EXPECT_EQ(order[0].parametric_env(),
            ParametricEnv(absl::flat_hash_map<std::string, InterpValue>{
                {"N", InterpValue::MakeUBits(/*bit_count=*/32, /*value=*/2)}}));
  EXPECT_EQ(order[1].f()->identifier(), "f");
  EXPECT_EQ(order[1].parametric_env(),
            ParametricEnv(absl::flat_hash_map<std::string, InterpValue>{
                {"N", InterpValue::MakeUBits(/*bit_count=*/32, /*value=*/2)}}));
  EXPECT_EQ(order[2].f()->identifier(), "main");
  EXPECT_EQ(order[2].parametric_env(), ParametricEnv());
}

TEST(ExtractConversionOrderTest, BuiltinIsElided) {
  constexpr std::string_view kProgram = R"(
fn main() -> u32 { fail!("failure", u32:0) }
)";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<ConversionRecord> order,
                           GetOrder(tm.module, tm.type_info));
  ASSERT_EQ(1, order.size());
  EXPECT_EQ(order[0].f()->identifier(), "main");
  EXPECT_EQ(order[0].parametric_env(), ParametricEnv());
}

TEST(ExtractConversionOrderTest, GetOrderForEntryFunctionWithFunctions) {
  constexpr std::string_view kProgram = R"(
fn g() -> u32 { u32:42 }
fn f() -> u32 { g() }
fn main() -> u32 { f() }
)";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           tm.module->GetMemberOrError<Function>("main"));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<ConversionRecord> order,
                           GetOrderForEntry(f, tm.type_info));
  ASSERT_EQ(3, order.size());
  EXPECT_EQ(order[0].f()->identifier(), "g");
  EXPECT_FALSE(order[0].IsTop());
  EXPECT_EQ(order[1].f()->identifier(), "f");
  EXPECT_FALSE(order[1].IsTop());
  EXPECT_EQ(order[2].f()->identifier(), "main");
  EXPECT_TRUE(order[2].IsTop());
  XLS_ASSERT_OK_AND_ASSIGN(f, tm.module->GetMemberOrError<Function>("f"));
  XLS_ASSERT_OK_AND_ASSIGN(order, GetOrderForEntry(f, tm.type_info));
  ASSERT_EQ(2, order.size());
  EXPECT_EQ(order[0].f()->identifier(), "g");
  EXPECT_FALSE(order[0].IsTop());
  EXPECT_EQ(order[1].f()->identifier(), "f");
  EXPECT_TRUE(order[1].IsTop());
}

TEST(ExtractConversionOrderTest, GetOrderForEntryFunctionWithConst) {
  constexpr std::string_view kProgram = R"(
fn id(x: u32) -> u32 { x }

const MY_VALUE = id(u32:42);

fn entry() -> u32 { MY_VALUE }
)";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           tm.module->GetMemberOrError<Function>("entry"));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<ConversionRecord> order,
                           GetOrderForEntry(f, tm.type_info));
  ASSERT_EQ(1, order.size());
  EXPECT_EQ(order[0].f()->identifier(), "entry");
  EXPECT_TRUE(order[0].IsTop());
}

TEST(ExtractConversionOrderTest, GetOrderForEntryFunctionSingleFunction) {
  constexpr std::string_view kProgram = R"(
fn main() -> u32 { u32:42 }
)";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           tm.module->GetMemberOrError<Function>("main"));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<ConversionRecord> order,
                           GetOrderForEntry(f, tm.type_info));
  ASSERT_EQ(1, order.size());
  EXPECT_EQ(order[0].f()->identifier(), "main");
  EXPECT_TRUE(order[0].IsTop());
}

TEST(ExtractConversionOrderTest,
     GetOrderForEntryFunctionWithFunctionReoccurence) {
  constexpr std::string_view kProgram = R"(
fn h() -> u32 { u32:42 }
fn g() -> u32 { h() }
fn f() -> u32 { let x:u32 = g(); x + h() }
fn main() -> u32 { f() }
)";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           tm.module->GetMemberOrError<Function>("main"));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<ConversionRecord> order,
                           GetOrderForEntry(f, tm.type_info));
  ASSERT_EQ(4, order.size());
  EXPECT_EQ(order[0].f()->identifier(), "h");
  EXPECT_FALSE(order[0].IsTop());
  EXPECT_EQ(order[1].f()->identifier(), "g");
  EXPECT_FALSE(order[1].IsTop());
  EXPECT_EQ(order[2].f()->identifier(), "f");
  EXPECT_FALSE(order[2].IsTop());
  EXPECT_EQ(order[3].f()->identifier(), "main");
  EXPECT_TRUE(order[3].IsTop());
}

TEST(ExtractConversionOrderTest, GetOrderForEntryFunctionWithDiamondCallGraph) {
  constexpr std::string_view kProgram = R"(
fn i() -> u32 { u32:42 }
fn h() -> u32 { i() }
fn g() -> u32 { i() }
fn f() -> u32 { g() + h() }
fn main() -> u32 { f() }
)";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           tm.module->GetMemberOrError<Function>("main"));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<ConversionRecord> order,
                           GetOrderForEntry(f, tm.type_info));
  ASSERT_EQ(5, order.size());
  EXPECT_EQ(order[0].f()->identifier(), "i");
  EXPECT_FALSE(order[0].IsTop());
  EXPECT_EQ(order[1].f()->identifier(), "g");
  EXPECT_FALSE(order[1].IsTop());
  EXPECT_EQ(order[2].f()->identifier(), "h");
  EXPECT_FALSE(order[2].IsTop());
  EXPECT_EQ(order[3].f()->identifier(), "f");
  EXPECT_FALSE(order[3].IsTop());
  EXPECT_EQ(order[4].f()->identifier(), "main");
  EXPECT_TRUE(order[4].IsTop());
}

// TODO(vmirian) 2-2-2022 Consider creating a struct containing the program,
// the golden result to verify for proc order tests.
TEST(ExtractConversionOrderTest, BasicProcWithEntry) {
  constexpr std::string_view kProgram = R"(
proc foo {
  init { () }
  config() { () }
  next(state: ()) { () }
}

proc main {
  init { () }
  config() {
    spawn foo();
    ()
  }
  next(state: ()) { () }
}
)";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  std::vector<ConversionRecord> order;
  XLS_ASSERT_OK_AND_ASSIGN(Proc * main,
                           tm.module->GetMemberOrError<Proc>("main"));
  XLS_ASSERT_OK_AND_ASSIGN(order, GetOrderForEntry(main, tm.type_info));
  ASSERT_EQ(5, order.size());
  ASSERT_TRUE(order[0].proc_id().has_value());
  ASSERT_TRUE(order[1].proc_id().has_value());
  ASSERT_TRUE(order[2].proc_id().has_value());
  ASSERT_TRUE(order[3].proc_id().has_value());
  EXPECT_EQ(order[0].f()->identifier(), "foo.init");
  EXPECT_EQ(order[0].proc_id().value().ToString(), "main->foo:0");
  EXPECT_FALSE(order[0].IsTop());
  EXPECT_EQ(order[1].f()->identifier(), "main.config");
  EXPECT_EQ(order[1].proc_id().value().ToString(), "main:0");
  EXPECT_FALSE(order[1].IsTop());
  EXPECT_EQ(order[2].f()->identifier(), "foo.config");
  EXPECT_EQ(order[2].proc_id().value().ToString(), "main->foo:0");
  EXPECT_FALSE(order[2].IsTop());
  EXPECT_EQ(order[3].f()->identifier(), "main.next");
  EXPECT_EQ(order[3].proc_id().value().ToString(), "main:0");
  EXPECT_TRUE(order[3].IsTop());
  EXPECT_EQ(order[4].f()->identifier(), "foo.next");
  EXPECT_EQ(order[4].proc_id().value().ToString(), "main->foo:0");
  EXPECT_FALSE(order[4].IsTop());
}

TEST(ExtractConversionOrderTest, BasicProc) {
  constexpr std::string_view kProgram = R"(
proc foo {
  init { () }
  config() { () }
  next(state: ()) { () }
}

proc main {
  init { () }
  config() {
    spawn foo();
    ()
  }
  next(state: ()) { () }
}
)";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<ConversionRecord> order,
                           GetOrder(tm.module, tm.type_info));
  ASSERT_EQ(5, order.size());
  ASSERT_TRUE(order[0].proc_id().has_value());
  ASSERT_TRUE(order[1].proc_id().has_value());
  ASSERT_TRUE(order[2].proc_id().has_value());
  ASSERT_TRUE(order[3].proc_id().has_value());
  ASSERT_TRUE(order[4].proc_id().has_value());
  EXPECT_EQ(order[0].f()->identifier(), "foo.init");
  EXPECT_EQ(order[0].proc_id().value().ToString(), "main->foo:0");
  EXPECT_EQ(order[1].f()->identifier(), "main.config");
  EXPECT_EQ(order[1].proc_id().value().ToString(), "main:0");
  EXPECT_EQ(order[2].f()->identifier(), "foo.config");
  EXPECT_EQ(order[2].proc_id().value().ToString(), "main->foo:0");
  EXPECT_EQ(order[3].f()->identifier(), "main.next");
  EXPECT_EQ(order[3].proc_id().value().ToString(), "main:0");
  EXPECT_EQ(order[4].f()->identifier(), "foo.next");
  EXPECT_EQ(order[4].proc_id().value().ToString(), "main->foo:0");
}

TEST(ExtractConversionOrderTest, ProcNetworkWithEntry) {
  constexpr std::string_view kProgram = R"(
fn f0() -> u32 {
  u32:42
}

fn f1() -> u32 {
  u32:24
}

proc p2 {
  init { u32:0 }
  config() { () }

  next(x: u32) {
    f0()
  }
}

proc p1 {
  init { u32:0 }
  config() {
    spawn p2();
    ()
  }
  next(i: u32) {
    i
  }
}

proc p0 {
  init { u32:1 }
  config() {
    spawn p2();
    spawn p1();
    ()
  }
  next(i: u32) {
    let j = f1();
    f0() + j
  }
}

proc main {
  init { () }
  config() {
    spawn p0();
    spawn p1();
    spawn p2();
    ()
  }
  next(state: ()) { () }
}
)";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  std::vector<ConversionRecord> order;
  XLS_ASSERT_OK_AND_ASSIGN(Proc * main,
                           tm.module->GetMemberOrError<Proc>("main"));
  XLS_ASSERT_OK_AND_ASSIGN(order, GetOrderForEntry(main, tm.type_info));
  ASSERT_EQ(21, order.size());
  ASSERT_TRUE(order[0].proc_id().has_value());
  ASSERT_FALSE(order[1].proc_id().has_value());
  ASSERT_TRUE(order[2].proc_id().has_value());
  ASSERT_TRUE(order[3].proc_id().has_value());
  ASSERT_FALSE(order[4].proc_id().has_value());
  ASSERT_TRUE(order[5].proc_id().has_value());
  ASSERT_TRUE(order[6].proc_id().has_value());
  ASSERT_TRUE(order[7].proc_id().has_value());
  ASSERT_TRUE(order[8].proc_id().has_value());
  ASSERT_TRUE(order[9].proc_id().has_value());
  ASSERT_TRUE(order[10].proc_id().has_value());
  ASSERT_TRUE(order[11].proc_id().has_value());
  ASSERT_TRUE(order[12].proc_id().has_value());
  ASSERT_TRUE(order[13].proc_id().has_value());
  ASSERT_TRUE(order[14].proc_id().has_value());
  ASSERT_TRUE(order[15].proc_id().has_value());
  ASSERT_TRUE(order[16].proc_id().has_value());
  ASSERT_TRUE(order[17].proc_id().has_value());
  ASSERT_TRUE(order[18].proc_id().has_value());
  ASSERT_TRUE(order[19].proc_id().has_value());
  ASSERT_TRUE(order[20].proc_id().has_value());
  EXPECT_EQ(order[0].f()->identifier(), "p2.init");
  EXPECT_FALSE(order[0].IsTop());
  EXPECT_EQ(order[0].proc_id().value().ToString(), "main->p0:0->p2:0");
  EXPECT_EQ(order[1].f()->identifier(), "f0");
  EXPECT_FALSE(order[1].IsTop());
  EXPECT_EQ(order[2].f()->identifier(), "p1.init");
  EXPECT_EQ(order[2].proc_id().value().ToString(), "main->p0:0->p1:0");
  EXPECT_FALSE(order[2].IsTop());
  EXPECT_EQ(order[3].f()->identifier(), "p0.init");
  EXPECT_EQ(order[3].proc_id().value().ToString(), "main->p0:0");
  EXPECT_FALSE(order[3].IsTop());
  EXPECT_EQ(order[4].f()->identifier(), "f1");
  EXPECT_FALSE(order[4].IsTop());
  EXPECT_EQ(order[5].f()->identifier(), "main.config");
  EXPECT_EQ(order[5].proc_id().value().ToString(), "main:0");
  EXPECT_FALSE(order[5].IsTop());
  EXPECT_EQ(order[6].f()->identifier(), "p2.config");
  EXPECT_EQ(order[6].proc_id().value().ToString(), "main->p2:0");
  EXPECT_FALSE(order[6].IsTop());
  EXPECT_EQ(order[7].f()->identifier(), "p1.config");
  EXPECT_EQ(order[7].proc_id().value().ToString(), "main->p1:0");
  EXPECT_FALSE(order[7].IsTop());
  EXPECT_EQ(order[8].f()->identifier(), "p2.config");
  EXPECT_EQ(order[8].proc_id().value().ToString(), "main->p1:0->p2:0");
  EXPECT_FALSE(order[8].IsTop());
  EXPECT_EQ(order[9].f()->identifier(), "p0.config");
  EXPECT_EQ(order[9].proc_id().value().ToString(), "main->p0:0");
  EXPECT_FALSE(order[9].IsTop());
  EXPECT_EQ(order[10].f()->identifier(), "p1.config");
  EXPECT_EQ(order[10].proc_id().value().ToString(), "main->p0:0->p1:0");
  EXPECT_FALSE(order[10].IsTop());
  EXPECT_EQ(order[11].f()->identifier(), "p2.config");
  EXPECT_EQ(order[11].proc_id().value().ToString(), "main->p0:0->p1:0->p2:0");
  EXPECT_FALSE(order[11].IsTop());
  EXPECT_EQ(order[12].f()->identifier(), "p2.config");
  EXPECT_EQ(order[12].proc_id().value().ToString(), "main->p0:0->p2:0");
  EXPECT_FALSE(order[12].IsTop());
  EXPECT_EQ(order[13].f()->identifier(), "main.next");
  EXPECT_EQ(order[13].proc_id().value().ToString(), "main:0");
  EXPECT_TRUE(order[13].IsTop());
  EXPECT_EQ(order[14].f()->identifier(), "p2.next");
  EXPECT_EQ(order[14].proc_id().value().ToString(), "main->p0:0->p2:0");
  EXPECT_FALSE(order[14].IsTop());
  EXPECT_EQ(order[15].f()->identifier(), "p2.next");
  EXPECT_EQ(order[15].proc_id().value().ToString(), "main->p0:0->p1:0->p2:0");
  EXPECT_FALSE(order[15].IsTop());
  EXPECT_EQ(order[16].f()->identifier(), "p1.next");
  EXPECT_EQ(order[16].proc_id().value().ToString(), "main->p0:0->p1:0");
  EXPECT_FALSE(order[16].IsTop());
  EXPECT_EQ(order[17].f()->identifier(), "p0.next");
  EXPECT_EQ(order[17].proc_id().value().ToString(), "main->p0:0");
  EXPECT_FALSE(order[17].IsTop());
}

TEST(ExtractConversionOrderTest, ProcNetwork) {
  constexpr std::string_view kProgram = R"(
fn f0() -> u32 {
  u32:42
}

fn f1() -> u32 {
  u32:24
}

proc p2 {
  init { u32:3 }
  config() { () }

  next(x: u32) {
    f0()
  }
}

proc p1 {
  init { u32:8 }
  config() {
    spawn p2();
    ()
  }
  next(i: u32) {
    i
  }
}

proc p0 {
  init { u32:1000 }
  config() {
    spawn p2();
    spawn p1();
    ()
  }
  next(i: u32) {
    let j = f1();
    f0() + j
  }
}

proc main {
  init { () }
  config() {
    spawn p0();
    spawn p1();
    spawn p2();
    ()
  }
  next(state: ()) { () }
}
)";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<ConversionRecord> order,
                           GetOrder(tm.module, tm.type_info));
  ASSERT_EQ(21, order.size());
  ASSERT_FALSE(order[0].proc_id().has_value());
  ASSERT_FALSE(order[1].proc_id().has_value());
  ASSERT_TRUE(order[2].proc_id().has_value());
  ASSERT_TRUE(order[3].proc_id().has_value());
  ASSERT_TRUE(order[4].proc_id().has_value());
  ASSERT_TRUE(order[5].proc_id().has_value());
  ASSERT_TRUE(order[6].proc_id().has_value());
  ASSERT_TRUE(order[7].proc_id().has_value());
  ASSERT_TRUE(order[8].proc_id().has_value());
  ASSERT_TRUE(order[9].proc_id().has_value());
  ASSERT_TRUE(order[10].proc_id().has_value());
  ASSERT_TRUE(order[11].proc_id().has_value());
  ASSERT_TRUE(order[12].proc_id().has_value());
  ASSERT_TRUE(order[13].proc_id().has_value());
  ASSERT_TRUE(order[14].proc_id().has_value());
  ASSERT_TRUE(order[15].proc_id().has_value());
  ASSERT_TRUE(order[16].proc_id().has_value());
  ASSERT_TRUE(order[17].proc_id().has_value());
  ASSERT_TRUE(order[18].proc_id().has_value());
  ASSERT_TRUE(order[19].proc_id().has_value());
  ASSERT_TRUE(order[20].proc_id().has_value());

  EXPECT_EQ(order[0].f()->identifier(), "f0");
  EXPECT_EQ(order[1].f()->identifier(), "f1");
  EXPECT_EQ(order[2].f()->identifier(), "p2.init");
  EXPECT_EQ(order[2].proc_id().value().ToString(), "main->p0:0->p2:0");
  EXPECT_EQ(order[3].f()->identifier(), "p1.init");
  EXPECT_EQ(order[3].proc_id().value().ToString(), "main->p0:0->p1:0");
  EXPECT_EQ(order[4].f()->identifier(), "p0.init");
  EXPECT_EQ(order[4].proc_id().value().ToString(), "main->p0:0");
  EXPECT_EQ(order[5].f()->identifier(), "main.config");
  EXPECT_EQ(order[5].proc_id().value().ToString(), "main:0");
  EXPECT_EQ(order[6].f()->identifier(), "p2.config");
  EXPECT_EQ(order[6].proc_id().value().ToString(), "main->p2:0");
  EXPECT_EQ(order[7].f()->identifier(), "p1.config");
  EXPECT_EQ(order[7].proc_id().value().ToString(), "main->p1:0");
  EXPECT_EQ(order[8].f()->identifier(), "p2.config");
  EXPECT_EQ(order[8].proc_id().value().ToString(), "main->p1:0->p2:0");
  EXPECT_EQ(order[9].f()->identifier(), "p0.config");
  EXPECT_EQ(order[9].proc_id().value().ToString(), "main->p0:0");
  EXPECT_EQ(order[10].f()->identifier(), "p1.config");
  EXPECT_EQ(order[10].proc_id().value().ToString(), "main->p0:0->p1:0");
  EXPECT_EQ(order[11].f()->identifier(), "p2.config");
  EXPECT_EQ(order[11].proc_id().value().ToString(), "main->p0:0->p1:0->p2:0");
  EXPECT_EQ(order[12].f()->identifier(), "p2.config");
  EXPECT_EQ(order[12].proc_id().value().ToString(), "main->p0:0->p2:0");
  EXPECT_EQ(order[13].f()->identifier(), "main.next");
  EXPECT_EQ(order[13].proc_id().value().ToString(), "main:0");
  EXPECT_EQ(order[14].f()->identifier(), "p2.next");
  EXPECT_EQ(order[14].proc_id().value().ToString(), "main->p0:0->p2:0");
  EXPECT_EQ(order[15].f()->identifier(), "p2.next");
  EXPECT_EQ(order[15].proc_id().value().ToString(), "main->p0:0->p1:0->p2:0");
  EXPECT_EQ(order[16].f()->identifier(), "p1.next");
  EXPECT_EQ(order[16].proc_id().value().ToString(), "main->p0:0->p1:0");
  EXPECT_EQ(order[17].f()->identifier(), "p0.next");
  EXPECT_EQ(order[17].proc_id().value().ToString(), "main->p0:0");
  EXPECT_EQ(order[18].f()->identifier(), "p2.next");
  EXPECT_EQ(order[18].proc_id().value().ToString(), "main->p1:0->p2:0");
  EXPECT_EQ(order[19].f()->identifier(), "p1.next");
  EXPECT_EQ(order[19].proc_id().value().ToString(), "main->p1:0");
  EXPECT_EQ(order[20].f()->identifier(), "p2.next");
  EXPECT_EQ(order[20].proc_id().value().ToString(), "main->p2:0");
}

TEST(ExtractConversionOrderTest, ProcNetworkWithTwoTopLevelProcs) {
  constexpr std::string_view kProgram = R"(
proc p2 {
  init { () }
  config() { () }
  next(state: ()) { () }
}

proc p1 {
  init { () }
  config() { () }
  next(state: ()) { () }
}

proc p0 {
  init { () }
  config() {
    spawn p1();
    spawn p2();
    ()
  }
  next(state: ()) { () }
}

proc main {
  init { () }
  config() {
    spawn p1();
    spawn p2();
    ()
  }
  next(state: ()) { () }
}
)";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<ConversionRecord> order,
                           GetOrder(tm.module, tm.type_info));
  ASSERT_EQ(14, order.size());
  ASSERT_TRUE(order[0].proc_id().has_value());
  ASSERT_TRUE(order[1].proc_id().has_value());
  ASSERT_TRUE(order[2].proc_id().has_value());
  ASSERT_TRUE(order[3].proc_id().has_value());
  ASSERT_TRUE(order[4].proc_id().has_value());
  ASSERT_TRUE(order[5].proc_id().has_value());
  ASSERT_TRUE(order[6].proc_id().has_value());
  ASSERT_TRUE(order[7].proc_id().has_value());
  ASSERT_TRUE(order[8].proc_id().has_value());
  ASSERT_TRUE(order[9].proc_id().has_value());
  ASSERT_TRUE(order[10].proc_id().has_value());
  ASSERT_TRUE(order[11].proc_id().has_value());
  ASSERT_TRUE(order[12].proc_id().has_value());
  ASSERT_TRUE(order[13].proc_id().has_value());
  EXPECT_EQ(order[0].f()->identifier(), "p1.init");
  EXPECT_EQ(order[0].proc_id().value().ToString(), "p0->p1:0");
  EXPECT_EQ(order[1].f()->identifier(), "p2.init");
  EXPECT_EQ(order[1].proc_id().value().ToString(), "p0->p2:0");
  EXPECT_EQ(order[2].f()->identifier(), "p0.config");
  EXPECT_EQ(order[2].proc_id().value().ToString(), "p0:0");
  EXPECT_EQ(order[3].f()->identifier(), "p2.config");
  EXPECT_EQ(order[3].proc_id().value().ToString(), "p0->p2:0");

  EXPECT_EQ(order[4].f()->identifier(), "p1.config");
  EXPECT_EQ(order[4].proc_id().value().ToString(), "p0->p1:0");
  EXPECT_EQ(order[5].f()->identifier(), "p0.next");
  EXPECT_EQ(order[5].proc_id().value().ToString(), "p0:0");
  EXPECT_EQ(order[6].f()->identifier(), "p1.next");
  EXPECT_EQ(order[6].proc_id().value().ToString(), "p0->p1:0");
  EXPECT_EQ(order[7].f()->identifier(), "p2.next");
  EXPECT_EQ(order[7].proc_id().value().ToString(), "p0->p2:0");
  EXPECT_EQ(order[8].f()->identifier(), "main.config");
  EXPECT_EQ(order[8].proc_id().value().ToString(), "main:0");
  EXPECT_EQ(order[9].f()->identifier(), "p2.config");
  EXPECT_EQ(order[9].proc_id().value().ToString(), "main->p2:0");
  EXPECT_EQ(order[10].f()->identifier(), "p1.config");
  EXPECT_EQ(order[10].proc_id().value().ToString(), "main->p1:0");
  EXPECT_EQ(order[11].f()->identifier(), "main.next");
  EXPECT_EQ(order[11].proc_id().value().ToString(), "main:0");
  EXPECT_EQ(order[12].f()->identifier(), "p1.next");
  EXPECT_EQ(order[12].proc_id().value().ToString(), "main->p1:0");
  EXPECT_EQ(order[13].f()->identifier(), "p2.next");
  EXPECT_EQ(order[13].proc_id().value().ToString(), "main->p2:0");
}

TEST(ExtractConversionOrderTest, ProcChainWithMultiLevelMultiInstanceCounts) {
  constexpr std::string_view kProgram = R"(
proc C {
  init { () }
  config() { () }
  next(state: ()) { () }
}

proc B {
  init { () }
  config() {
    spawn C();
    ()
  }
  next(state: ()) { () }
}

proc E {
  init { () }
  config() {
    spawn B();
    ()
  }
  next(state: ()) { () }
}

proc D {
  init { () }
  config() {
    spawn B();
    ()
  }
  next(state: ()) { () }
}

proc A {
  init { () }
  config() {
    spawn D();
    spawn B();
    spawn B();
    spawn E();
    ()
  }
  next(state: ()) { () }
}
)";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<ConversionRecord> order,
                           GetOrder(tm.module, tm.type_info));
  EXPECT_THAT(order,
              ElementsAre(IdentifierAndProcId("C.init", "A->D:0->B:0->C:0"),
                          IdentifierAndProcId("B.init", "A->D:0->B:0"),
                          IdentifierAndProcId("D.init", "A->D:0"),
                          IdentifierAndProcId("E.init", "A->E:0"),
                          IdentifierAndProcId("A.config", "A:0"),
                          IdentifierAndProcId("E.config", "A->E:0"),
                          IdentifierAndProcId("B.config", "A->E:0->B:0"),
                          IdentifierAndProcId("C.config", "A->E:0->B:0->C:0"),
                          IdentifierAndProcId("B.config", "A->B:1"),
                          IdentifierAndProcId("C.config", "A->B:1->C:0"),
                          IdentifierAndProcId("B.config", "A->B:0"),
                          IdentifierAndProcId("C.config", "A->B:0->C:0"),
                          IdentifierAndProcId("D.config", "A->D:0"),
                          IdentifierAndProcId("B.config", "A->D:0->B:0"),
                          IdentifierAndProcId("C.config", "A->D:0->B:0->C:0"),
                          IdentifierAndProcId("A.next", "A:0"),
                          IdentifierAndProcId("C.next", "A->D:0->B:0->C:0"),
                          IdentifierAndProcId("B.next", "A->D:0->B:0"),
                          IdentifierAndProcId("D.next", "A->D:0"),
                          IdentifierAndProcId("C.next", "A->B:0->C:0"),
                          IdentifierAndProcId("B.next", "A->B:0"),
                          IdentifierAndProcId("C.next", "A->B:1->C:0"),
                          IdentifierAndProcId("B.next", "A->B:1"),
                          IdentifierAndProcId("C.next", "A->E:0->B:0->C:0"),
                          IdentifierAndProcId("B.next", "A->E:0->B:0"),
                          IdentifierAndProcId("E.next", "A->E:0")));
}

TEST(GetTopLevelProcsTest, OnlyOneParametricProc) {
  constexpr std::string_view kProgram = R"(
proc np {
  init { () }
  config() { () }
  next(state: ()) { () }
}

proc p<N: u32> {
  init { () }
  config() { () }
  next(state: ()) { () }
}
)";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Proc*> top_level_procs,
                           GetTopLevelProcs(tm.module, tm.type_info));
  ASSERT_EQ(top_level_procs.size(), 1);
  EXPECT_EQ(top_level_procs.at(0)->identifier(), "np");
}

}  // namespace
}  // namespace xls::dslx
