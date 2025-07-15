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

#include "xls/dslx/ir_convert/get_conversion_records.h"

#include <string>
#include <string_view>
#include <vector>

#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/ir_convert/extract_conversion_order.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/type_system/parametric_env.h"

namespace xls::dslx {
namespace {

TEST(GetConversionRecordsTest, SimpleLinearCallgraph) {
  constexpr std::string_view kProgram = R"(
fn g() -> u32 { u32:42 }
fn f() -> u32 { g() }
fn main() -> u32 { f() }
)";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<ConversionRecord> order,
      GetConversionRecords(tm.module, tm.type_info, false));
  ASSERT_EQ(3, order.size());
  EXPECT_EQ(order[0].f()->identifier(), "g");
  EXPECT_EQ(order[1].f()->identifier(), "f");
  EXPECT_EQ(order[2].f()->identifier(), "main");
}

TEST(GetConversionRecordsTest, MultipleCallsToSameFunction) {
  constexpr std::string_view kProgram = R"(
fn g() -> u32 { u32:42 }
fn main() -> u32 { g() + g() }
)";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<ConversionRecord> order,
      GetConversionRecords(tm.module, tm.type_info, false));
  ASSERT_EQ(2, order.size());
  EXPECT_EQ(order[0].f()->identifier(), "g");
  EXPECT_EQ(order[1].f()->identifier(), "main");
}

TEST(GetConversionRecordsTest, ParametricFn) {
  constexpr std::string_view kProgram = R"(
fn f<N: u32>(x: bits[N]) -> u32 { N }
fn main() -> u32 { f(u2:0) }
)";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<ConversionRecord> order,
      GetConversionRecords(tm.module, tm.type_info, false));
  ASSERT_EQ(2, order.size());
  EXPECT_EQ(order[0].f()->identifier(), "f");
  EXPECT_EQ(order[0].parametric_env(),
            ParametricEnv(absl::flat_hash_map<std::string, InterpValue>{
                {"N", InterpValue::MakeUBits(/*bit_count=*/32, /*value=*/2)}}));
  EXPECT_EQ(order[1].f()->identifier(), "main");
  EXPECT_EQ(order[1].parametric_env(), ParametricEnv());
}

TEST(GetConversionRecordsTest, ParametricFnCalledTwice) {
  constexpr std::string_view kProgram = R"(
fn f<N: u32>(x: bits[N]) -> u32 { N }
fn main() -> u32 { f(u2:0) + f(u2:1) }
)";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<ConversionRecord> order,
      GetConversionRecords(tm.module, tm.type_info, false));
  ASSERT_EQ(2, order.size());
  EXPECT_EQ(order[0].f()->identifier(), "f");
  EXPECT_EQ(order[0].parametric_env(),
            ParametricEnv(absl::flat_hash_map<std::string, InterpValue>{
                {"N", InterpValue::MakeUBits(/*bit_count=*/32, /*value=*/2)}}));
  EXPECT_EQ(order[1].f()->identifier(), "main");
  EXPECT_EQ(order[1].parametric_env(), ParametricEnv());
}

TEST(GetConversionRecordsTest, ParametricFnCalledDifferentParametrics) {
  constexpr std::string_view kProgram = R"(
fn f<N: u32>(x: bits[N]) -> u32 { N }
fn main() -> u32 { f(u2:0) + f(u3:1) }
)";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<ConversionRecord> order,
      GetConversionRecords(tm.module, tm.type_info, false));
  // Two for f, one for main.
  ASSERT_EQ(3, order.size());
  EXPECT_EQ(order[0].f()->identifier(), "f");
  EXPECT_EQ(order[0].parametric_env(),
            ParametricEnv(absl::flat_hash_map<std::string, InterpValue>{
                {"N", InterpValue::MakeUBits(/*bit_count=*/32, /*value=*/2)}}));
  EXPECT_EQ(order[1].f()->identifier(), "f");
  EXPECT_EQ(order[1].parametric_env(),
            ParametricEnv(absl::flat_hash_map<std::string, InterpValue>{
                {"N", InterpValue::MakeUBits(/*bit_count=*/32, /*value=*/3)}}));
  EXPECT_EQ(order[2].f()->identifier(), "main");
  EXPECT_EQ(order[2].parametric_env(), ParametricEnv());
}

TEST(GetConversionRecordsTest, ParametricProc) {
  constexpr std::string_view kProgram = R"(
proc P<N: u32> {
  x: uN[N];
  init {zero!<uN[N]>()}
  config(x: uN[N]) { (x, ) }
  next(x: uN[N]) { x }
}

proc top {
  init{}
  config() {
    spawn P<u32:2>(u2:1);
    spawn P<u32:2>(u2:1);
    spawn P<u32:4>(u4:2);
    ()
  }
  next(x: ()) { () }
}
)";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<ConversionRecord> order,
      GetConversionRecords(tm.module, tm.type_info, false));
  ASSERT_EQ(3, order.size());
  EXPECT_EQ(order[0].f()->identifier(), "P.next");
  EXPECT_EQ(order[0].parametric_env(),
            ParametricEnv(absl::flat_hash_map<std::string, InterpValue>{
                {"N", InterpValue::MakeUBits(/*bit_count=*/32, /*value=*/2)}}));
  EXPECT_EQ(order[1].f()->identifier(), "P.next");
  EXPECT_EQ(order[1].parametric_env(),
            ParametricEnv(absl::flat_hash_map<std::string, InterpValue>{
                {"N", InterpValue::MakeUBits(/*bit_count=*/32, /*value=*/4)}}));
  EXPECT_EQ(order[2].f()->identifier(), "top.next");
  EXPECT_EQ(order[2].parametric_env(), ParametricEnv());
}

TEST(GetConversionRecordsTest, TransitiveParametric) {
  constexpr std::string_view kProgram = R"(
fn g<M: u32>(x: bits[M]) -> u32 { M }
fn f<N: u32>(x: bits[N]) -> u32 { g(x) }
fn main() -> u32 { f(u2:0) }
)";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<ConversionRecord> order,
      GetConversionRecords(tm.module, tm.type_info, false));
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

TEST(GetConversionRecordsTest, BuiltinIsElided) {
  constexpr std::string_view kProgram = R"(
fn main() -> u32 { fail!("failure", u32:0) }
)";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<ConversionRecord> order,
      GetConversionRecords(tm.module, tm.type_info, false));
  ASSERT_EQ(1, order.size());
  EXPECT_EQ(order[0].f()->identifier(), "main");
  EXPECT_EQ(order[0].parametric_env(), ParametricEnv());
}

TEST(GetConversionRecordsTest, BasicProc) {
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
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<ConversionRecord> order,
      GetConversionRecords(tm.module, tm.type_info, false));
  ASSERT_EQ(2, order.size());
  EXPECT_EQ(order[0].f()->identifier(), "foo.next");
  EXPECT_EQ(order[1].f()->identifier(), "main.next");
}

TEST(GetConversionRecordsTest, ProcNetwork) {
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
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<ConversionRecord> order,
      GetConversionRecords(tm.module, tm.type_info, false));
  ASSERT_EQ(6, order.size());

  EXPECT_EQ(order[0].f()->identifier(), "f0");
  EXPECT_EQ(order[1].f()->identifier(), "f1");
  EXPECT_EQ(order[2].f()->identifier(), "p2.next");
  EXPECT_EQ(order[3].f()->identifier(), "p1.next");
  EXPECT_EQ(order[4].f()->identifier(), "p0.next");
  EXPECT_EQ(order[5].f()->identifier(), "main.next");
}

TEST(GetConversionRecordsTest, ProcNetworkWithTwoTopLevelProcs) {
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
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<ConversionRecord> order,
      GetConversionRecords(tm.module, tm.type_info, false));
  ASSERT_EQ(4, order.size());
  EXPECT_EQ(order[0].f()->identifier(), "p2.next");
  EXPECT_EQ(order[1].f()->identifier(), "p1.next");
  EXPECT_EQ(order[2].f()->identifier(), "p0.next");
  EXPECT_EQ(order[3].f()->identifier(), "main.next");
}

TEST(GetConversionRecordsTest, TestFunction) {
  constexpr std::string_view kProgram = R"(
fn f<N: u32>(x: bits[N]) -> u32 { N }

#[test]
fn my_test() -> bool { f<u32:8>(u8:1) == u32:8 }
)";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<ConversionRecord> order,
                           GetConversionRecords(tm.module, tm.type_info, true));
  ASSERT_EQ(2, order.size());
  EXPECT_EQ(order[0].f()->identifier(), "f");
  EXPECT_EQ(order[0].parametric_env(),
            ParametricEnv(absl::flat_hash_map<std::string, InterpValue>{
                {"N", InterpValue::MakeUBits(/*bit_count=*/32, /*value=*/8)}}));
  EXPECT_EQ(order[1].f()->identifier(), "my_test");
  EXPECT_EQ(order[1].parametric_env(), ParametricEnv());
}

TEST(GetConversionRecordsTest, TestFunctionSkipped) {
  constexpr std::string_view kProgram = R"(
fn f<N: u32>(x: bits[N]) -> u32 { N }

#[test]
fn my_test() -> bool { f<u32:8>(u8:1) == u32:8 }
)";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<ConversionRecord> order,
      GetConversionRecords(tm.module, tm.type_info, false));
  ASSERT_EQ(1, order.size());
  // Even though the function is called from a test and the test is ignored,
  // there are still types in the type info for the parametric to be evaluated.
  EXPECT_EQ(order[0].f()->identifier(), "f");
  EXPECT_EQ(order[0].parametric_env(),
            ParametricEnv(absl::flat_hash_map<std::string, InterpValue>{
                {"N", InterpValue::MakeUBits(/*bit_count=*/32, /*value=*/8)}}));
}

TEST(GetConversionRecordsTest, TestProc) {
  constexpr std::string_view kProgram = R"(
proc P<N: u32> {
  x: uN[N];
  init {zero!<uN[N]>()}
  config(x: uN[N]) { (x, ) }
  next(x: uN[N]) { x }
}

#[test_proc]
proc test {
  init{}
  config(terminator: chan<bool> out) {
    spawn P<u32:4>(u4:2);
    ()
  }
  next(x: ()) { () }
}
)";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<ConversionRecord> order,
      GetConversionRecords(tm.module, tm.type_info, false));
  ASSERT_EQ(2, order.size());
  EXPECT_EQ(order[0].f()->identifier(), "P.next");
  EXPECT_EQ(order[0].parametric_env(),
            ParametricEnv(absl::flat_hash_map<std::string, InterpValue>{
                {"N", InterpValue::MakeUBits(/*bit_count=*/32, /*value=*/4)}}));
  EXPECT_EQ(order[1].f()->identifier(), "test.next");
}

TEST(GetConversionRecordsTest, Quickcheck) {
  constexpr std::string_view kProgram = R"(
#[quickcheck]
fn identity(x: u32) -> bool { x == x }
)";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<ConversionRecord> order,
      GetConversionRecords(tm.module, tm.type_info, false));
  ASSERT_EQ(1, order.size());
  EXPECT_EQ(order[0].f()->identifier(), "identity");
  EXPECT_EQ(order[0].parametric_env(), ParametricEnv());
}

TEST(GetConversionRecordsTest, ImplWithFn) {
  constexpr std::string_view kProgram = R"(
struct S {
  field: u32
}

impl S {
  fn f(self: Self) -> u32 { self.field }
}
)";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<ConversionRecord> order,
      GetConversionRecords(tm.module, tm.type_info, false));
  ASSERT_EQ(1, order.size());
  EXPECT_EQ(order[0].f()->identifier(), "f");
  EXPECT_EQ(order[0].parametric_env(), ParametricEnv());
}

}  // namespace

}  // namespace xls::dslx
