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

TEST(GetConversionRecordsTest, ParametricFnCalled2x) {
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
}  // namespace

}  // namespace xls::dslx
