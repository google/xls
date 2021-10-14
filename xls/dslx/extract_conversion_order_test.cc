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

#include "xls/dslx/extract_conversion_order.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/parse_and_typecheck.h"

namespace xls::dslx {
namespace {

TEST(ExtractConversionOrderTest, SimpleLinearCallgraph) {
  const char* program = R"(
fn g() -> u32 { u32:42 }
fn f() -> u32 { g() }
fn main() -> u32 { f() }
)";
  auto import_data = ImportData::CreateForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(program, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<ConversionRecord> order,
                           GetOrder(tm.module, tm.type_info));
  ASSERT_EQ(3, order.size());
  EXPECT_EQ(order[0].f()->identifier(), "g");
  EXPECT_EQ(order[1].f()->identifier(), "f");
  EXPECT_EQ(order[2].f()->identifier(), "main");
}

TEST(ExtractConversionOrderTest, Parametric) {
  const char* program = R"(
fn f<N: u32>(x: bits[N]) -> u32 { N }
fn main() -> u32 { f(u2:0) }
)";
  auto import_data = ImportData::CreateForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(program, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<ConversionRecord> order,
                           GetOrder(tm.module, tm.type_info));
  ASSERT_EQ(2, order.size());
  EXPECT_EQ(order[0].f()->identifier(), "f");
  EXPECT_EQ(order[0].symbolic_bindings(),
            SymbolicBindings(absl::flat_hash_map<std::string, InterpValue>{
                {"N", InterpValue::MakeUBits(/*bit_count=*/32, /*value=*/2)}}));
  EXPECT_EQ(order[1].f()->identifier(), "main");
  EXPECT_EQ(order[1].symbolic_bindings(), SymbolicBindings());
}

TEST(ExtractConversionOrderTest, TransitiveParametric) {
  const char* program = R"(
fn g<M: u32>(x: bits[M]) -> u32 { M }
fn f<N: u32>(x: bits[N]) -> u32 { g(x) }
fn main() -> u32 { f(u2:0) }
)";
  auto import_data = ImportData::CreateForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(program, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<ConversionRecord> order,
                           GetOrder(tm.module, tm.type_info));
  ASSERT_EQ(3, order.size());
  EXPECT_EQ(order[0].f()->identifier(), "g");
  order[0].symbolic_bindings(),
      SymbolicBindings(absl::flat_hash_map<std::string, InterpValue>{
          {"M", InterpValue::MakeUBits(/*bit_count=*/32, /*value=*/2)}});
  EXPECT_EQ(order[1].f()->identifier(), "f");
  EXPECT_EQ(order[1].symbolic_bindings(),
            SymbolicBindings(absl::flat_hash_map<std::string, InterpValue>{
                {"N", InterpValue::MakeUBits(/*bit_count=*/32, /*value=*/2)}}));
  EXPECT_EQ(order[2].f()->identifier(), "main");
  EXPECT_EQ(order[2].symbolic_bindings(), SymbolicBindings());
}

TEST(ExtractConversionOrderTest, BuiltinIsElided) {
  const char* program = R"(
fn main() -> u32 { fail!(u32:0) }
)";
  auto import_data = ImportData::CreateForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(program, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<ConversionRecord> order,
                           GetOrder(tm.module, tm.type_info));
  ASSERT_EQ(1, order.size());
  EXPECT_EQ(order[0].f()->identifier(), "main");
  EXPECT_EQ(order[0].symbolic_bindings(), SymbolicBindings());
}

TEST(ExtractConversionOrderTest, GetOrderForEntryFunctionWithFunctions) {
  const char* program = R"(
fn g() -> u32 { u32:42 }
fn f() -> u32 { g() }
fn main() -> u32 { f() }
)";
  auto import_data = ImportData::CreateForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(program, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, tm.module->GetFunctionOrError("main"));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<ConversionRecord> order,
                           GetOrderForEntry(f, tm.type_info));
  ASSERT_EQ(3, order.size());
  EXPECT_EQ(order[0].f()->identifier(), "g");
  EXPECT_EQ(order[1].f()->identifier(), "f");
  EXPECT_EQ(order[2].f()->identifier(), "main");
  XLS_ASSERT_OK_AND_ASSIGN(f, tm.module->GetFunctionOrError("f"));
  XLS_ASSERT_OK_AND_ASSIGN(order, GetOrderForEntry(f, tm.type_info));
  ASSERT_EQ(2, order.size());
  EXPECT_EQ(order[0].f()->identifier(), "g");
  EXPECT_EQ(order[1].f()->identifier(), "f");
}

TEST(ExtractConversionOrderTest, GetOrderForEntryFunctionWithConst) {
  const char* program = R"(
fn id(x: u32) -> u32 { x }

const MY_VALUE = id(u32:42);

fn entry() -> u32 { MY_VALUE }
)";
  auto import_data = ImportData::CreateForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(program, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           tm.module->GetFunctionOrError("entry"));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<ConversionRecord> order,
                           GetOrderForEntry(f, tm.type_info));
  ASSERT_EQ(2, order.size());
  EXPECT_EQ(order[0].f()->identifier(), "id");
  EXPECT_EQ(order[1].f()->identifier(), "entry");
}

TEST(ExtractConversionOrderTest, GetOrderForEntryFunctionSingleFunction) {
  const char* program = R"(
fn main() -> u32 { u32:42 }
)";
  auto import_data = ImportData::CreateForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(program, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, tm.module->GetFunctionOrError("main"));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<ConversionRecord> order,
                           GetOrderForEntry(f, tm.type_info));
  ASSERT_EQ(1, order.size());
  EXPECT_EQ(order[0].f()->identifier(), "main");
}

TEST(ExtractConversionOrderTest,
     GetOrderForEntryFunctionWithFunctionReoccurence) {
  const char* program = R"(
fn h() -> u32 { u32:42 }
fn g() -> u32 { h() }
fn f() -> u32 { let x:u32 = g(); x + h() }
fn main() -> u32 { f() }
)";
  auto import_data = ImportData::CreateForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(program, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, tm.module->GetFunctionOrError("main"));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<ConversionRecord> order,
                           GetOrderForEntry(f, tm.type_info));
  ASSERT_EQ(4, order.size());
  EXPECT_EQ(order[0].f()->identifier(), "h");
  EXPECT_EQ(order[1].f()->identifier(), "g");
  EXPECT_EQ(order[2].f()->identifier(), "f");
  EXPECT_EQ(order[3].f()->identifier(), "main");
}

TEST(ExtractConversionOrderTest, GetOrderForEntryFunctionWithDiamondCallGraph) {
  const char* program = R"(
fn i() -> u32 { u32:42 }
fn h() -> u32 { i() }
fn g() -> u32 { i() }
fn f() -> u32 { g() + h() }
fn main() -> u32 { f() }
)";
  auto import_data = ImportData::CreateForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(program, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, tm.module->GetFunctionOrError("main"));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<ConversionRecord> order,
                           GetOrderForEntry(f, tm.type_info));
  ASSERT_EQ(5, order.size());
  EXPECT_EQ(order[0].f()->identifier(), "i");
  EXPECT_EQ(order[1].f()->identifier(), "g");
  EXPECT_EQ(order[2].f()->identifier(), "h");
  EXPECT_EQ(order[3].f()->identifier(), "f");
  EXPECT_EQ(order[4].f()->identifier(), "main");
}

TEST(ExtractConversionOrderTest, BasicProc) {
  const char* program = R"(
proc foo {
  config() { () }
  next() { () }
}

proc main {
  config() {
    spawn foo()();
    ()
  }
  next() { () }
}
)";
  auto import_data = ImportData::CreateForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(program, "test.x", "test", &import_data));
  std::vector<ConversionRecord> order;
  XLS_ASSERT_OK_AND_ASSIGN(Proc * main, tm.module->GetProcOrError("main"));
  XLS_ASSERT_OK_AND_ASSIGN(order, GetOrderForEntry(main, tm.type_info));
  ASSERT_EQ(4, order.size());
  ASSERT_TRUE(order[0].proc_id().has_value());
  ASSERT_TRUE(order[1].proc_id().has_value());
  ASSERT_TRUE(order[2].proc_id().has_value());
  ASSERT_TRUE(order[3].proc_id().has_value());
  EXPECT_EQ(order[0].f()->identifier(), "main_config");
  EXPECT_EQ(order[0].proc_id().value().ToString(), "main:0");
  EXPECT_EQ(order[1].f()->identifier(), "foo_config");
  EXPECT_EQ(order[1].proc_id().value().ToString(), "main->foo:0");
  EXPECT_EQ(order[2].f()->identifier(), "main_next");
  EXPECT_EQ(order[2].proc_id().value().ToString(), "main:0");
  EXPECT_EQ(order[3].f()->identifier(), "foo_next");
  EXPECT_EQ(order[3].proc_id().value().ToString(), "main->foo:0");
}

TEST(ExtractConversionOrderTest, ProcNetwork) {
  const char* program = R"(
fn f0() -> u32 {
  u32:42
}

fn f1() -> u32 {
  u32:24
}

proc p2 {
  config() { () }

  next(x: u32) {
    f0()
  }
}

proc p1 {
  config() {
    spawn p2()(u32:0);
    ()
  }
  next(i: u32) {
    i
  }
}

proc p0 {
  config() {
    spawn p2()(u32:1);
    spawn p1()(u32:2);
    ()
  }
  next(i: u32) {
    let j = f1();
    f0() + j
  }
}

proc main {
  config() {
    spawn p0()(u32:3);
    spawn p1()(u32:4);
    spawn p2()(u32:5);
    ()
  }
  next() { () }
}
)";
  auto import_data = ImportData::CreateForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(program, "test.x", "test", &import_data));
  std::vector<ConversionRecord> order;
  XLS_ASSERT_OK_AND_ASSIGN(Proc * main, tm.module->GetProcOrError("main"));
  XLS_ASSERT_OK_AND_ASSIGN(order, GetOrderForEntry(main, tm.type_info));
  ASSERT_EQ(18, order.size());
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
  EXPECT_EQ(order[0].f()->identifier(), "f0");
  EXPECT_EQ(order[1].f()->identifier(), "f1");
  EXPECT_EQ(order[2].f()->identifier(), "main_config");
  EXPECT_EQ(order[2].proc_id().value().ToString(), "main:0");
  EXPECT_EQ(order[3].f()->identifier(), "p2_config");
  EXPECT_EQ(order[3].proc_id().value().ToString(), "main->p2:0");
  EXPECT_EQ(order[4].f()->identifier(), "p1_config");
  EXPECT_EQ(order[4].proc_id().value().ToString(), "main->p1:0");
  EXPECT_EQ(order[5].f()->identifier(), "p2_config");
  EXPECT_EQ(order[5].proc_id().value().ToString(), "main->p1->p2:0");
  EXPECT_EQ(order[6].f()->identifier(), "p0_config");
  EXPECT_EQ(order[6].proc_id().value().ToString(), "main->p0:0");
  EXPECT_EQ(order[7].f()->identifier(), "p1_config");
  EXPECT_EQ(order[7].proc_id().value().ToString(), "main->p0->p1:0");
  EXPECT_EQ(order[8].f()->identifier(), "p2_config");
  EXPECT_EQ(order[8].proc_id().value().ToString(), "main->p0->p1->p2:0");
  EXPECT_EQ(order[9].f()->identifier(), "p2_config");
  EXPECT_EQ(order[9].proc_id().value().ToString(), "main->p0->p2:0");
  EXPECT_EQ(order[10].f()->identifier(), "main_next");
  EXPECT_EQ(order[10].proc_id().value().ToString(), "main:0");
  EXPECT_EQ(order[11].f()->identifier(), "p2_next");
  EXPECT_EQ(order[11].proc_id().value().ToString(), "main->p0->p2:0");
  EXPECT_EQ(order[12].f()->identifier(), "p2_next");
  EXPECT_EQ(order[12].proc_id().value().ToString(), "main->p0->p1->p2:0");
  EXPECT_EQ(order[13].f()->identifier(), "p1_next");
  EXPECT_EQ(order[13].proc_id().value().ToString(), "main->p0->p1:0");
  EXPECT_EQ(order[14].f()->identifier(), "p0_next");
  EXPECT_EQ(order[14].proc_id().value().ToString(), "main->p0:0");
  EXPECT_EQ(order[15].f()->identifier(), "p2_next");
  EXPECT_EQ(order[15].proc_id().value().ToString(), "main->p1->p2:0");
  EXPECT_EQ(order[16].f()->identifier(), "p1_next");
  EXPECT_EQ(order[16].proc_id().value().ToString(), "main->p1:0");
  EXPECT_EQ(order[17].f()->identifier(), "p2_next");
  EXPECT_EQ(order[17].proc_id().value().ToString(), "main->p2:0");
}

}  // namespace
}  // namespace xls::dslx
