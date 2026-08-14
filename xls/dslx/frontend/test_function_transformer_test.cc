// Copyright 2026 The XLS Authors
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

#include "xls/dslx/frontend/test_function_transformer.h"

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "gtest/gtest.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/parse_and_typecheck.h"

namespace xls::dslx {
namespace {

TEST(TestFunctionTransformerTest, NoTestFunctionsUnchanged) {
  constexpr std::string_view kProgram = R"(fn main() -> u32 {
    u32:0
})";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  TestFunctionTransformer transformer(*tm.module, *tm.type_info);
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> new_module,
                           transformer.TransformTestFunctions());
  EXPECT_EQ(new_module->ToString(), kProgram);

  new_module->SetName(absl::StrCat(new_module->name(), "_cloned"));
  XLS_EXPECT_OK(
      TypecheckModule(std::move(new_module), "test_cloned.x", &import_data)
          .status());
}

TEST(TestFunctionTransformerTest, TestFunctionNoSpawnsUnchanged) {
  constexpr std::string_view kProgram = R"(#[test]
fn main() {
    ()
})";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  TestFunctionTransformer transformer(*tm.module, *tm.type_info);
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> new_module,
                           transformer.TransformTestFunctions());
  EXPECT_EQ(new_module->ToString(), kProgram);

  new_module->SetName(absl::StrCat(new_module->name(), "_cloned"));
  XLS_EXPECT_OK(
      TypecheckModule(std::move(new_module), "test_cloned.x", &import_data)
          .status());
}

TEST(TestFunctionTransformerTest, TestFunctionWithSpawnRemoved) {
  constexpr std::string_view kProgram = R"(proc P { }
impl P {
    fn new() -> Self { P {  } }
}

#[test]
fn main() {
    let p = P::new();
    p.spawn();
})";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  TestFunctionTransformer transformer(*tm.module, *tm.type_info);
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> new_module,
                           transformer.TransformTestFunctions());
  EXPECT_FALSE(absl::StrContains(new_module->ToString(), "fn main()"));
  EXPECT_TRUE(absl::StrContains(new_module->ToString(),
                                "__test__terminator: chan<bool> out"));
  EXPECT_TRUE(absl::StrContains(new_module->ToString(),
                                "#[test]\nproc __test__proc__main"));

  new_module->SetName(absl::StrCat(new_module->name(), "_cloned"));
  XLS_EXPECT_OK(
      TypecheckModule(std::move(new_module), "test_cloned.x", &import_data)
          .status());
}

TEST(TestFunctionTransformerTest, TestFunctionWithSetupAndRuntime) {
  constexpr std::string_view kProgram = R"(proc P {
    r: chan<u32> in,
}
impl P {
    fn new(ri: chan<u32> in) -> Self { P { r:ri  } }
}

#[test]
fn main() {
    let (s, r) = chan<u32>("pin");
    let p = P::new(r);
    p.spawn();
    send(token(), s, u32:0);
})";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  TestFunctionTransformer transformer(*tm.module, *tm.type_info);
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> new_module,
                           transformer.TransformTestFunctions());

  std::string cloned_code = new_module->ToString();
  EXPECT_FALSE(absl::StrContains(cloned_code, "fn main()"));

  size_t impl_pos = cloned_code.find("impl __test__proc__main");
  ASSERT_NE(impl_pos, std::string::npos);
  std::string impl_code = cloned_code.substr(impl_pos);

  size_t new_pos = impl_code.find("fn new(");
  size_t next_pos = impl_code.find("fn next(");

  ASSERT_NE(new_pos, std::string::npos);
  ASSERT_NE(next_pos, std::string::npos);

  // Setup statements must be in 'new'
  std::string new_body = impl_code.substr(new_pos, next_pos - new_pos);
  EXPECT_TRUE(absl::StrContains(new_body, "chan<u32>(\"pin\")"));
  EXPECT_TRUE(absl::StrContains(new_body, "P::new(r)"));
  EXPECT_TRUE(absl::StrContains(new_body, "p.spawn()"));

  // Runtime statements must be in 'next'
  std::string next_body = impl_code.substr(next_pos);
  EXPECT_TRUE(absl::StrContains(next_body, "send(token(), s, u32:0)"));
  EXPECT_TRUE(absl::StrContains(next_body, "self.__test__terminator"));

  // TODO(davidplass): This test passes because TypecheckModule on the direct
  // AST does not re-evaluate lexical scoping for already-linked NameRef nodes.
  // This will change when we build "variable promotion" of locals to fields.
  new_module->SetName(absl::StrCat(new_module->name(), "_cloned"));
  XLS_EXPECT_OK(
      TypecheckModule(std::move(new_module), "test_cloned.x", &import_data)
          .status());
}

TEST(TestFunctionTransformerTest, TestFunctionSimpleConstantLet) {
  constexpr std::string_view kProgram = R"(
proc P {
}
impl P {
    fn new(x: u32) -> Self { P {} }
}

#[test]
fn main() {
    let x = u32:42;
    let p = P::new(x);
    p.spawn();
})";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  TestFunctionTransformer transformer(*tm.module, *tm.type_info);
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> new_module,
                           transformer.TransformTestFunctions());

  std::string cloned_code = new_module->ToString();

  size_t impl_pos = cloned_code.find("impl __test__proc__main");
  ASSERT_NE(impl_pos, std::string::npos);
  std::string impl_code = cloned_code.substr(impl_pos);

  size_t new_pos = impl_code.find("fn new(");
  size_t next_pos = impl_code.find("fn next(");

  ASSERT_NE(new_pos, std::string::npos);
  ASSERT_NE(next_pos, std::string::npos);

  std::string new_body = impl_code.substr(new_pos, next_pos - new_pos);

  // All setup statements must be in 'new'
  EXPECT_TRUE(absl::StrContains(new_body, "let x = u32:42;"));
  EXPECT_TRUE(absl::StrContains(new_body, "let p = P::new(x);"));
  EXPECT_TRUE(absl::StrContains(new_body, "p.spawn();"));
  // Struct instance is returned by "new"; terminator channel is added to the
  // struct instance.
  EXPECT_TRUE(absl::StrContains(
      new_body,
      "__test__proc__main { __test__terminator: __test__terminator }"));

  new_module->SetName(absl::StrCat(new_module->name(), "_cloned"));
  XLS_EXPECT_OK(
      TypecheckModule(std::move(new_module), "test_cloned.x", &import_data)
          .status());
}

TEST(TestFunctionTransformerTest, TestFunctionWithAssertions) {
  constexpr std::string_view kProgram = R"(
proc P {
}
impl P {
    fn new() -> Self { P {} }
}

#[test]
fn main() {
    let (tx, rx) = chan<u32>("c");
    let p = P::new();
    p.spawn();
    send(token(), tx, u32:42);
    assert_eq(u32:1, u32:1);
})";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  TestFunctionTransformer transformer(*tm.module, *tm.type_info);
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> new_module,
                           transformer.TransformTestFunctions());

  std::string cloned_code = new_module->ToString();

  size_t impl_pos = cloned_code.find("impl __test__proc__main");
  ASSERT_NE(impl_pos, std::string::npos);
  std::string impl_code = cloned_code.substr(impl_pos);

  size_t new_pos = impl_code.find("fn new(");
  size_t next_pos = impl_code.find("fn next(");

  ASSERT_NE(new_pos, std::string::npos);
  ASSERT_NE(next_pos, std::string::npos);

  // Setup in 'new'
  std::string new_body = impl_code.substr(new_pos, next_pos - new_pos);
  EXPECT_TRUE(absl::StrContains(new_body, "chan<u32>(\"c\")"));
  EXPECT_TRUE(absl::StrContains(new_body, "P::new()"));
  EXPECT_TRUE(absl::StrContains(new_body, "p.spawn()"));

  // Runtime and assertions in 'next'
  std::string next_body = impl_code.substr(next_pos);
  EXPECT_TRUE(absl::StrContains(next_body, "send(token(), tx, u32:42)"));
  EXPECT_TRUE(absl::StrContains(next_body, "assert_eq(u32:1, u32:1)"));

  // TODO(davidplass): This test passes because TypecheckModule on the direct
  // AST does not re-evaluate lexical scoping for already-linked NameRef nodes.
  // This will change when we build "variable promotion" of locals to fields.
  new_module->SetName(absl::StrCat(new_module->name(), "_cloned"));
  XLS_EXPECT_OK(
      TypecheckModule(std::move(new_module), "test_cloned.x", &import_data)
          .status());
}

TEST(TestFunctionTransformerTest, TestFunctionMultipleSpawns) {
  constexpr std::string_view kProgram = R"(
proc P {
}
impl P {
    fn new() -> Self { P {} }
}

#[test]
fn main() {
    let (tx1, rx1) = chan<u32>("c1");
    let (tx2, rx2) = chan<u32>("c2");
    let p1 = P::new();
    let p2 = P::new();
    p1.spawn();
    p2.spawn();
    send(token(), tx1, u32:1);
    send(token(), tx2, u32:2);
})";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  TestFunctionTransformer transformer(*tm.module, *tm.type_info);
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> new_module,
                           transformer.TransformTestFunctions());

  std::string cloned_code = new_module->ToString();

  size_t impl_pos = cloned_code.find("impl __test__proc__main");
  ASSERT_NE(impl_pos, std::string::npos);
  std::string impl_code = cloned_code.substr(impl_pos);

  size_t new_pos = impl_code.find("fn new(");
  size_t next_pos = impl_code.find("fn next(");

  ASSERT_NE(new_pos, std::string::npos);
  ASSERT_NE(next_pos, std::string::npos);

  // Setup in 'new'
  std::string new_body = impl_code.substr(new_pos, next_pos - new_pos);
  EXPECT_TRUE(absl::StrContains(new_body, "chan<u32>(\"c1\")"));
  EXPECT_TRUE(absl::StrContains(new_body, "chan<u32>(\"c2\")"));
  EXPECT_TRUE(absl::StrContains(new_body, "p1 = P::new()"));
  EXPECT_TRUE(absl::StrContains(new_body, "p2 = P::new()"));
  EXPECT_TRUE(absl::StrContains(new_body, "p1.spawn()"));
  EXPECT_TRUE(absl::StrContains(new_body, "p2.spawn()"));

  // Runtime in 'next'
  std::string next_body = impl_code.substr(next_pos);
  EXPECT_TRUE(absl::StrContains(next_body, "send(token(), tx1, u32:1)"));
  EXPECT_TRUE(absl::StrContains(next_body, "send(token(), tx2, u32:2)"));

  // TODO(davidplass): This test passes because TypecheckModule on the direct
  // AST does not re-evaluate lexical scoping for already-linked NameRef nodes.
  // This will change when we build "variable promotion" of locals to fields.
  new_module->SetName(absl::StrCat(new_module->name(), "_cloned"));
  XLS_EXPECT_OK(
      TypecheckModule(std::move(new_module), "test_cloned.x", &import_data)
          .status());
}

// Tests that the statements in the test function are cloned into the new
// module.
TEST(TestFunctionTransformerTest, StatementCloningOwnerTest) {
  constexpr std::string_view kProgram = R"(
proc P {
}
impl P {
    fn new(x: u32) -> Self { P {} }
}

#[test]
fn main() {
    let x = u32:42;
    let p = P::new(x);
    p.spawn();
}
)";
  std::unique_ptr<Module> new_module;
  {
    auto import_data = CreateImportDataForTest();
    XLS_ASSERT_OK_AND_ASSIGN(
        TypecheckedModule tm,
        ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
    TestFunctionTransformer transformer(*tm.module, *tm.type_info);
    XLS_ASSERT_OK_AND_ASSIGN(new_module, transformer.TransformTestFunctions());
  }
  std::string cloned_code = new_module->ToString();
  EXPECT_TRUE(absl::StrContains(cloned_code, "let x = u32:42;"));
}

}  // namespace
}  // namespace xls::dslx
