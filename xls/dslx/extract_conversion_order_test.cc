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
  ImportData import_data;
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(program, "test.x", "test", &import_data,
                        /*additional_search_paths=*/{}));
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
  ImportData import_data;
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(program, "test.x", "test", &import_data,
                        /*additional_search_paths=*/{}));
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
  ImportData import_data;
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(program, "test.x", "test", &import_data,
                        /*additional_search_paths=*/{}));
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
  ImportData import_data;
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(program, "test.x", "test", &import_data,
                        /*additional_search_paths=*/{}));
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
  ImportData import_data;
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(program, "test.x", "test", &import_data,
                        /*additional_search_paths=*/{}));
  EXPECT_TRUE(tm.module->GetFunction("main").has_value());
  std::vector<ConversionRecord> order;
  XLS_ASSERT_OK_AND_ASSIGN(
      order,
      GetOrderForEntry(tm.module->GetFunction("main").value(), tm.type_info));
  ASSERT_EQ(3, order.size());
  EXPECT_EQ(order[0].f()->identifier(), "g");
  EXPECT_EQ(order[1].f()->identifier(), "f");
  EXPECT_EQ(order[2].f()->identifier(), "main");
  XLS_ASSERT_OK_AND_ASSIGN(
      order,
      GetOrderForEntry(tm.module->GetFunction("f").value(), tm.type_info));
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
  ImportData import_data;
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(program, "test.x", "test", &import_data,
                        /*additional_search_paths=*/{}));
  EXPECT_TRUE(tm.module->GetFunction("entry").has_value());
  std::vector<ConversionRecord> order;
  XLS_ASSERT_OK_AND_ASSIGN(
      order,
      GetOrderForEntry(tm.module->GetFunction("entry").value(), tm.type_info));
  ASSERT_EQ(2, order.size());
  EXPECT_EQ(order[0].f()->identifier(), "id");
  EXPECT_EQ(order[1].f()->identifier(), "entry");
}

TEST(ExtractConversionOrderTest, GetOrderForEntryFunctionSingleFunction) {
  const char* program = R"(
fn main() -> u32 { u32:42 }
)";
  ImportData import_data;
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(program, "test.x", "test", &import_data,
                        /*additional_search_paths=*/{}));
  EXPECT_TRUE(tm.module->GetFunction("main").has_value());
  std::vector<ConversionRecord> order;
  XLS_ASSERT_OK_AND_ASSIGN(
      order,
      GetOrderForEntry(tm.module->GetFunction("main").value(), tm.type_info));
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
  ImportData import_data;
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(program, "test.x", "test", &import_data,
                        /*additional_search_paths=*/{}));
  EXPECT_TRUE(tm.module->GetFunction("main").has_value());
  std::vector<ConversionRecord> order;
  XLS_ASSERT_OK_AND_ASSIGN(
      order,
      GetOrderForEntry(tm.module->GetFunction("main").value(), tm.type_info));
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
  ImportData import_data;
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(program, "test.x", "test", &import_data,
                        /*additional_search_paths=*/{}));
  EXPECT_TRUE(tm.module->GetFunction("main").has_value());
  std::vector<ConversionRecord> order;
  XLS_ASSERT_OK_AND_ASSIGN(
      order,
      GetOrderForEntry(tm.module->GetFunction("main").value(), tm.type_info));
  ASSERT_EQ(5, order.size());
  EXPECT_EQ(order[0].f()->identifier(), "i");
  EXPECT_EQ(order[1].f()->identifier(), "g");
  EXPECT_EQ(order[2].f()->identifier(), "h");
  EXPECT_EQ(order[3].f()->identifier(), "f");
  EXPECT_EQ(order[4].f()->identifier(), "main");
}

}  // namespace
}  // namespace xls::dslx
