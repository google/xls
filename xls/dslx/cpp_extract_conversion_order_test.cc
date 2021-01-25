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

#include "xls/dslx/cpp_extract_conversion_order.h"

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
  ImportCache import_cache;
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(program, "test.x", "test", &import_cache));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<ConversionRecord> order,
                           GetOrder(tm.module.get(), tm.type_info));
  ASSERT_EQ(3, order.size());
  EXPECT_EQ(order[0].f->identifier(), "g");
  EXPECT_EQ(order[1].f->identifier(), "f");
  EXPECT_EQ(order[2].f->identifier(), "main");
}

TEST(ExtractConversionOrderTest, Parametric) {
  const char* program = R"(
fn f<N: u32>(x: bits[N]) -> u32 { N }
fn main() -> u32 { f(u2:0) }
)";
  ImportCache import_cache;
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(program, "test.x", "test", &import_cache));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<ConversionRecord> order,
                           GetOrder(tm.module.get(), tm.type_info));
  ASSERT_EQ(2, order.size());
  EXPECT_EQ(order[0].f->identifier(), "f");
  EXPECT_EQ(
      order[0].bindings,
      SymbolicBindings(absl::flat_hash_map<std::string, int64>{{"N", 2}}));
  EXPECT_EQ(order[1].f->identifier(), "main");
  EXPECT_EQ(order[1].bindings, SymbolicBindings());
  tm.type_info->ClearTypeInfoRefsForGc();
}

TEST(ExtractConversionOrderTest, TransitiveParametric) {
  const char* program = R"(
fn g<M: u32>(x: bits[M]) -> u32 { M }
fn f<N: u32>(x: bits[N]) -> u32 { g(x) }
fn main() -> u32 { f(u2:0) }
)";
  ImportCache import_cache;
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(program, "test.x", "test", &import_cache));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<ConversionRecord> order,
                           GetOrder(tm.module.get(), tm.type_info));
  ASSERT_EQ(3, order.size());
  EXPECT_EQ(order[0].f->identifier(), "g");
  EXPECT_EQ(
      order[0].bindings,
      SymbolicBindings(absl::flat_hash_map<std::string, int64>{{"M", 2}}));
  EXPECT_EQ(order[1].f->identifier(), "f");
  EXPECT_EQ(
      order[1].bindings,
      SymbolicBindings(absl::flat_hash_map<std::string, int64>{{"N", 2}}));
  EXPECT_EQ(order[2].f->identifier(), "main");
  EXPECT_EQ(order[2].bindings, SymbolicBindings());
  tm.type_info->ClearTypeInfoRefsForGc();
}

TEST(ExtractConversionOrderTest, BuiltinIsElided) {
  const char* program = R"(
fn main() -> u32 { fail!(u32:0) }
)";
  ImportCache import_cache;
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(program, "test.x", "test", &import_cache));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<ConversionRecord> order,
                           GetOrder(tm.module.get(), tm.type_info));
  ASSERT_EQ(1, order.size());
  EXPECT_EQ(order[0].f->identifier(), "main");
  EXPECT_EQ(order[0].bindings, SymbolicBindings());
}

}  // namespace
}  // namespace xls::dslx
