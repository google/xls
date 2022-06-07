// Copyright 2022 The XLS Authors
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
#include "xls/dslx/ast_cloner.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/ast.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/parse_and_typecheck.h"

namespace xls::dslx {
namespace {

TEST(AstClonerTest, BasicOperation) {
  constexpr absl::string_view kProgram = R"(
fn main() -> u32 {
  let a = u32:0;
  let b = u32:1;
  u32:3
})";

  constexpr absl::string_view kExpected = R"(let a = u32:0;
let b = u32:1;
u32:3)";

  XLS_ASSERT_OK_AND_ASSIGN(auto module,
                           ParseModule(kProgram, "fake_path.x", "the_module"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, module->GetFunctionOrError("main"));
  XLS_ASSERT_OK_AND_ASSIGN(AstNode * clone, CloneAst(f->body()));
  EXPECT_EQ(kExpected, clone->ToString());
}

TEST(AstClonerTest, NameRefs) {
  constexpr absl::string_view kProgram = R"(
fn main() -> u32 {
  let a = u32:0;
  a
})";

  constexpr absl::string_view kExpected = R"(let a = u32:0;
a)";

  XLS_ASSERT_OK_AND_ASSIGN(auto module,
                           ParseModule(kProgram, "fake_path.x", "the_module"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, module->GetFunctionOrError("main"));
  XLS_ASSERT_OK_AND_ASSIGN(AstNode * clone, CloneAst(f->body()));
  EXPECT_EQ(kExpected, clone->ToString());
}

TEST(AstClonerTest, XlsTuple) {
  constexpr absl::string_view kProgram = R"(
fn main() -> (u32, u32) {
  let a = u32:0;
  let b = u32:1;
  (a, b)
}
)";

  constexpr absl::string_view kExpected = R"(let a = u32:0;
let b = u32:1;
(a, b))";

  XLS_ASSERT_OK_AND_ASSIGN(auto module,
                           ParseModule(kProgram, "fake_path.x", "the_module"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, module->GetFunctionOrError("main"));
  XLS_ASSERT_OK_AND_ASSIGN(AstNode * clone, CloneAst(f->body()));
  EXPECT_EQ(kExpected, clone->ToString());
}

TEST(AstClonerTest, BasicFunction) {
  constexpr absl::string_view kProgram = R"(fn main() -> (u32, u32) {
  let a = u32:0;
  let b = u32:1;
  (a, b)
})";

  XLS_ASSERT_OK_AND_ASSIGN(auto module,
                           ParseModule(kProgram, "fake_path.x", "the_module"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, module->GetFunctionOrError("main"));
  XLS_ASSERT_OK_AND_ASSIGN(AstNode * clone, CloneAst(f));
  EXPECT_EQ(kProgram, clone->ToString());
}

TEST(AstClonerTest, StructDefAndInstance) {
  constexpr absl::string_view kProgram = R"(
struct MyStruct {
  a: u32,
  b: s64
}

fn main() -> MyStruct {
  MyStruct { a: u32:0, b: s64:0xbeef }
}
)";

  constexpr absl::string_view kExpectedStructDef = R"(struct MyStruct {
  a: u32,
  b: s64,
})";

  constexpr absl::string_view kExpectedFunction = R"(fn main() -> MyStruct {
  MyStruct { a: u32:0, b: s64:0xbeef }
})";

  XLS_ASSERT_OK_AND_ASSIGN(auto module,
                           ParseModule(kProgram, "fake_path.x", "the_module"));
  StructDef* struct_def = module->GetStructDefs().at(0);
  XLS_ASSERT_OK_AND_ASSIGN(AstNode * clone, CloneAst(struct_def));
  EXPECT_EQ(kExpectedStructDef, clone->ToString());

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, module->GetFunctionOrError("main"));
  XLS_ASSERT_OK_AND_ASSIGN(clone, CloneAst(f));
  EXPECT_EQ(kExpectedFunction, clone->ToString());
}

TEST(AstClonerTest, ColonRefToImportedStruct) {
  constexpr absl::string_view kProgram = R"(
import my.module as foo

fn main() -> foo::ImportedStruct {
  let bar = foo::ImportedStruct { a: u32:0, b: s64:0xbeef };
  bar.b
})";

  constexpr absl::string_view kExpectedFunction =
      R"(fn main() -> foo::ImportedStruct {
  let bar = foo::ImportedStruct { a: u32:0, b: s64:0xbeef };
  bar.b
})";

  XLS_ASSERT_OK_AND_ASSIGN(auto module,
                           ParseModule(kProgram, "fake_path.x", "the_module"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, module->GetFunctionOrError("main"));
  XLS_ASSERT_OK_AND_ASSIGN(AstNode * clone, CloneAst(f));
  EXPECT_EQ(kExpectedFunction, clone->ToString());
}

TEST(AstClonerTest, ArraysAndConstantDefs) {
  constexpr absl::string_view kProgram = R"(
const ARRAY_SIZE = uN[32]:5;
fn main() -> u32[ARRAY_SIZE] {
  u32[ARRAY_SIZE]:[u32:0, u32:1, u32:2, ...]
})";

  constexpr absl::string_view kExpectedFunction =
      R"(fn main() -> u32[ARRAY_SIZE] {
  u32[ARRAY_SIZE]:[u32:0, u32:1, u32:2, ...]
})";

  XLS_ASSERT_OK_AND_ASSIGN(auto module,
                           ParseModule(kProgram, "fake_path.x", "the_module"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, module->GetFunctionOrError("main"));
  XLS_ASSERT_OK_AND_ASSIGN(AstNode * clone, CloneAst(f));
  EXPECT_EQ(kExpectedFunction, clone->ToString());
}

}  // namespace
}  // namespace xls::dslx
