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
#include "xls/dslx/frontend/ast_cloner.h"

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <variant>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/casts.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/command_line_utils.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/parse_and_typecheck.h"

namespace xls::dslx {
namespace {

using ::testing::IsEmpty;

std::optional<TypeRef*> FindFirstTypeRef(AstNode* node) {
  if (auto type_ref = dynamic_cast<TypeRef*>(node); type_ref) {
    return type_ref;
  }
  for (AstNode* child : node->GetChildren(true)) {
    std::optional type_ref = FindFirstTypeRef(child);
    if (type_ref.has_value()) {
      return type_ref;
    }
  }
  return std::nullopt;
}

TEST(AstClonerTest, BasicOperation) {
  constexpr std::string_view kProgram = R"(
fn main() -> u32 {
    let a = u32:0;
    let b = u32:1;
    u32:3
})";

  constexpr std::string_view kExpected = R"({
    let a = u32:0;
    let b = u32:1;
    u32:3
})";

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(auto module, ParseModule(kProgram, "fake_path.x",
                                                    "the_module", file_table));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           module->GetMemberOrError<Function>("main"));
  StatementBlock* body_expr = f->body();
  XLS_ASSERT_OK_AND_ASSIGN(AstNode * clone, CloneAst(body_expr));
  EXPECT_EQ(kExpected, clone->ToString());
  XLS_ASSERT_OK(VerifyClone(body_expr, clone, *module->file_table()));
}

TEST(AstClonerTest, NameRefs) {
  constexpr std::string_view kProgram = R"(
fn main() -> u32 {
    let a = u32:0;
    a
})";

  constexpr std::string_view kExpected = R"({
    let a = u32:0;
    a
})";

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(auto module, ParseModule(kProgram, "fake_path.x",
                                                    "the_module", file_table));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           module->GetMemberOrError<Function>("main"));
  StatementBlock* body_expr = f->body();
  XLS_ASSERT_OK_AND_ASSIGN(AstNode * clone, CloneAst(body_expr));
  EXPECT_EQ(kExpected, clone->ToString());
  XLS_ASSERT_OK(VerifyClone(body_expr, clone, *module->file_table()));
}

TEST(AstClonerTest, FunctionRef) {
  constexpr std::string_view kProgram = R"(
fn f<X:u32>() -> u32 { X }

fn main() -> u32[3] {
    map([u32:1, u32:2, u32:3], f<u32:4>)
})";

  constexpr std::string_view kExpected = R"({
    map([u32:1, u32:2, u32:3], f<u32:4>)
})";

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(auto module, ParseModule(kProgram, "fake_path.x",
                                                    "the_module", file_table));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           module->GetMemberOrError<Function>("main"));
  StatementBlock* body_expr = f->body();
  XLS_ASSERT_OK_AND_ASSIGN(AstNode * clone, CloneAst(body_expr));
  EXPECT_EQ(kExpected, clone->ToString());
  XLS_ASSERT_OK(VerifyClone(body_expr, clone, *module->file_table()));
}

TEST(AstClonerTest, ReplaceOneOfTwoNameRefs) {
  constexpr std::string_view kProgram = R"(
fn main() -> u32 {
    let a = u32:0;
    let b = a + 2;
    b
})";

  constexpr std::string_view kExpected = R"({
    let a = u32:0;
    let b = 3 + 2;
    b
})";

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(auto module, ParseModule(kProgram, "fake_path.x",
                                                    "the_module", file_table));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           module->GetMemberOrError<Function>("main"));
  Number* a_replacement =
      module->Make<Number>(Span::Fake(), "3", NumberKind::kOther,
                           /*type=*/nullptr);
  StatementBlock* body_expr = f->body();
  XLS_ASSERT_OK_AND_ASSIGN(
      AstNode * clone,
      CloneAst(body_expr,
               [&](const AstNode* original_node) -> std::optional<AstNode*> {
                 if (const auto* name_ref =
                         dynamic_cast<const NameRef*>(original_node);
                     name_ref && name_ref->identifier() == "a") {
                   return a_replacement;
                 }
                 return std::nullopt;
               }));
  EXPECT_EQ(kExpected, clone->ToString());
  XLS_ASSERT_OK(VerifyClone(body_expr, clone, *module->file_table()));
}

TEST(AstClonerTest, XlsTuple) {
  constexpr std::string_view kProgram = R"(
fn main() -> (u32, u32) {
    let a = u32:0;
    let b = u32:1;
    (a, b)
}
)";

  constexpr std::string_view kExpected = R"({
    let a = u32:0;
    let b = u32:1;
    (a, b)
})";

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(auto module, ParseModule(kProgram, "fake_path.x",
                                                    "the_module", file_table));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           module->GetMemberOrError<Function>("main"));
  StatementBlock* body_expr = f->body();
  XLS_ASSERT_OK_AND_ASSIGN(AstNode * clone, CloneAst(body_expr));
  EXPECT_EQ(kExpected, clone->ToString());
  XLS_ASSERT_OK(VerifyClone(body_expr, clone, *module->file_table()));
}

TEST(AstClonerTest, BasicFunction) {
  constexpr std::string_view kProgram = R"(fn main() -> (u32, u32) {
    let a = u32:0;
    let b = u32:1;
    (a, b)
})";

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(auto module, ParseModule(kProgram, "fake_path.x",
                                                    "the_module", file_table));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           module->GetMemberOrError<Function>("main"));
  XLS_ASSERT_OK_AND_ASSIGN(AstNode * clone, CloneAst(f));
  EXPECT_EQ(kProgram, clone->ToString());
  XLS_ASSERT_OK(VerifyClone(f, clone, *module->file_table()));
}

TEST(AstClonerTest, StructDefAndInstance) {
  constexpr std::string_view kProgram = R"(
struct MyStruct {
    a: u32,
    b: s64
}

fn main() -> MyStruct {
    MyStruct { a: u32:0, b: s64:0xbeef }
}
)";

  constexpr std::string_view kExpectedStructDef = R"(struct MyStruct {
    a: u32,
    b: s64,
})";

  constexpr std::string_view kExpectedFunction = R"(fn main() -> MyStruct {
    MyStruct { a: u32:0, b: s64:0xbeef }
})";

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(auto module, ParseModule(kProgram, "fake_path.x",
                                                    "the_module", file_table));
  StructDef* struct_def = module->GetStructDefs().at(0);
  XLS_ASSERT_OK_AND_ASSIGN(AstNode * clone, CloneAst(struct_def));
  EXPECT_EQ(kExpectedStructDef, clone->ToString());

  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           module->GetMemberOrError<Function>("main"));
  XLS_ASSERT_OK_AND_ASSIGN(clone, CloneAst(f));
  EXPECT_EQ(kExpectedFunction, clone->ToString());
  XLS_ASSERT_OK(VerifyClone(f, clone, *module->file_table()));
}

TEST(AstClonerTest, ProcDefAndImpl) {
  constexpr std::string_view kProgram = R"(
proc MyProc {
    a: u32,
    b: s64
}

impl MyProc {}
)";

  constexpr std::string_view kExpectedProcDef = R"(proc MyProc {
    a: u32,
    b: s64,
})";

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(auto module, ParseModule(kProgram, "fake_path.x",
                                                    "the_module", file_table));
  ProcDef* proc_def = module->GetProcDefs().at(0);
  XLS_ASSERT_OK_AND_ASSIGN(AstNode * clone, CloneAst(proc_def));
  EXPECT_EQ(kExpectedProcDef, clone->ToString());
}

TEST(AstClonerTest, ParametricStructDefAndImpl) {
  constexpr std::string_view kProgram = R"(
struct MyStruct<N: u32> {
    a: u32[N],
    b: s64
}

impl MyStruct {}
)";

  constexpr std::string_view kExpectedProcDef = R"(struct MyStruct<N: u32> {
    a: u32[N],
    b: s64,
})";

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(auto module, ParseModule(kProgram, "fake_path.x",
                                                    "the_module", file_table));
  StructDef* struct_def = module->GetStructDefs().at(0);
  XLS_ASSERT_OK_AND_ASSIGN(AstNode * clone, CloneAst(struct_def));
  EXPECT_EQ(kExpectedProcDef, clone->ToString());
}

TEST(AstClonerTest, SimpleParametricProc) {
  constexpr std::string_view kProgram = R"(
proc p<N: u32> {
    config() { () }
    init { () }
    next(state: ()) { () }
}
)";
  constexpr std::string_view kExpected = R"(fn p.config<N: u32>() -> () {
    ()
}
fn p.init<N: u32>() -> () {
    ()
}
fn p.next<N: u32>(state: ()) -> () {
    ()
}
proc p<N: u32> {
    config() {
        ()
    }
    init {
        ()
    }
    next(state: ()) {
        ()
    }
})";

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(auto module, ParseModule(kProgram, "fake_path.x",
                                                    "the_module", file_table));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> clone,
                           CloneModule(module.get()));
  EXPECT_EQ(kExpected, clone->ToString());
}

TEST(AstClonerTest, StructDefAndImpl) {
  constexpr std::string_view kProgram = R"(
struct MyStruct {
    a: u32,
    b: s64
}

impl MyStruct {
    const MY_CONST = u32:5;
}

fn main() -> u32 {
    MyStruct::MY_CONST
}
)";

  constexpr std::string_view kExpectedStructDef = R"(struct MyStruct {
    a: u32,
    b: s64,
})";

  constexpr std::string_view kExpectedImpl = R"(impl MyStruct {
    const MY_CONST = u32:5;
})";

  constexpr std::string_view kExpectedFunction = R"(fn main() -> u32 {
    MyStruct::MY_CONST
})";

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(auto module, ParseModule(kProgram, "fake_path.x",
                                                    "the_module", file_table));

  Impl* impl = module->GetImpls().at(0);
  XLS_ASSERT_OK_AND_ASSIGN(AstNode * impl_clone, CloneAst(impl));
  EXPECT_EQ(kExpectedImpl, impl_clone->ToString());

  StructDef* struct_def = module->GetStructDefs().at(0);
  XLS_ASSERT_OK_AND_ASSIGN(AstNode * struct_clone, CloneAst(struct_def));
  EXPECT_EQ(kExpectedStructDef, struct_clone->ToString());
  Impl* cloned_impl = dynamic_cast<StructDef*>(struct_clone)->impl().value();
  EXPECT_EQ(kExpectedImpl, cloned_impl->ToString());

  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           module->GetMemberOrError<Function>("main"));
  XLS_ASSERT_OK_AND_ASSIGN(AstNode * clone, CloneAst(f));
  EXPECT_EQ(kExpectedFunction, clone->ToString());
  XLS_ASSERT_OK(VerifyClone(f, clone, *module->file_table()));
}

TEST(AstClonerTest, ColonRefToImportedStruct) {
  constexpr std::string_view kProgram = R"(
import my.module as foo;

fn main() -> foo::ImportedStruct {
    let bar = foo::ImportedStruct { a: u32:0, b: s64:0xbeef };
    bar.b
})";

  constexpr std::string_view kExpectedFunction =
      R"(fn main() -> foo::ImportedStruct {
    let bar = foo::ImportedStruct { a: u32:0, b: s64:0xbeef };
    bar.b
})";

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(auto module, ParseModule(kProgram, "fake_path.x",
                                                    "the_module", file_table));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           module->GetMemberOrError<Function>("main"));
  XLS_ASSERT_OK_AND_ASSIGN(AstNode * clone, CloneAst(f));
  EXPECT_EQ(kExpectedFunction, clone->ToString());
  XLS_ASSERT_OK(VerifyClone(f, clone, *module->file_table()));
}

TEST(AstClonerTest, ArraysAndConstantDefs) {
  constexpr std::string_view kProgram = R"(
const ARRAY_SIZE = uN[32]:5;
fn main() -> u32[ARRAY_SIZE] {
    u32[ARRAY_SIZE]:[u32:0, u32:1, u32:2, ...]
})";

  constexpr std::string_view kExpectedFunction =
      R"(fn main() -> u32[ARRAY_SIZE] {
    u32[ARRAY_SIZE]:[u32:0, u32:1, u32:2, ...]
})";

  constexpr std::string_view kExpectedConstant =
      R"(const ARRAY_SIZE = uN[32]:5;)";

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(auto module, ParseModule(kProgram, "fake_path.x",
                                                    "the_module", file_table));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           module->GetMemberOrError<Function>("main"));
  XLS_ASSERT_OK_AND_ASSIGN(AstNode * clone, CloneAst(f));
  EXPECT_EQ(kExpectedFunction, clone->ToString());

  XLS_ASSERT_OK_AND_ASSIGN(ConstantDef * constant,
                           module->GetConstantDef("ARRAY_SIZE"));
  XLS_ASSERT_OK_AND_ASSIGN(clone, CloneAst(constant));
  EXPECT_EQ(kExpectedConstant, clone->ToString());

  XLS_ASSERT_OK(VerifyClone(f, clone, *module->file_table()));
}

TEST(AstClonerTest, Binops) {
  constexpr std::string_view kProgram = R"(fn main() -> u13 {
    u13:5 + u13:500
})";

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(auto module, ParseModule(kProgram, "fake_path.x",
                                                    "the_module", file_table));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           module->GetMemberOrError<Function>("main"));
  XLS_ASSERT_OK_AND_ASSIGN(AstNode * clone, CloneAst(f));
  EXPECT_EQ(kProgram, clone->ToString());
  XLS_ASSERT_OK(VerifyClone(f, clone, *module->file_table()));
}

TEST(AstClonerTest, Unops) {
  constexpr std::string_view kProgram = R"(fn main() -> u13 {
    -u13:500
})";

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(auto module, ParseModule(kProgram, "fake_path.x",
                                                    "the_module", file_table));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           module->GetMemberOrError<Function>("main"));
  XLS_ASSERT_OK_AND_ASSIGN(AstNode * clone, CloneAst(f));
  EXPECT_EQ(kProgram, clone->ToString());
  XLS_ASSERT_OK(VerifyClone(f, clone, *module->file_table()));
}

TEST(AstClonerTest, Casts) {
  constexpr std::string_view kProgram = R"(fn main() -> u13 {
    -u17:500 as u13
})";

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(auto module, ParseModule(kProgram, "fake_path.x",
                                                    "the_module", file_table));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           module->GetMemberOrError<Function>("main"));
  XLS_ASSERT_OK_AND_ASSIGN(AstNode * clone, CloneAst(f));
  EXPECT_EQ(kProgram, clone->ToString());
  XLS_ASSERT_OK(VerifyClone(f, clone, *module->file_table()));
}

TEST(AstClonerTest, Procs) {
  constexpr std::string_view kProgram = R"(proc MyProc {
    a: u32;
    b: u64;
    config() {
        (u32:7, u64:0xfffffffffffff)
    }
    init {
        u19:0
    }
    next(state: u19) {
        (a as u64 + b) as u19
    }
})";

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(auto module, ParseModule(kProgram, "fake_path.x",
                                                    "the_module", file_table));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * p, module->GetMemberOrError<Proc>("MyProc"));
  XLS_ASSERT_OK_AND_ASSIGN(AstNode * clone, CloneAst(p));
  EXPECT_EQ(kProgram, clone->ToString());
  XLS_ASSERT_OK(VerifyClone(p, clone, *module->file_table()));
}

TEST(AstClonerTest, TestFunctions) {
  constexpr std::string_view kProgram = R"(#[test]
fn my_test() {
    let a = u32:0;
    let _ = assert_eq(u32:0, a);
    ()
})";

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(auto module, ParseModule(kProgram, "fake_path.x",
                                                    "the_module", file_table));
  XLS_ASSERT_OK_AND_ASSIGN(TestFunction * tf,
                           module->GetMemberOrError<TestFunction>("my_test"));
  XLS_ASSERT_OK_AND_ASSIGN(AstNode * clone, CloneAst(tf));
  EXPECT_EQ(kProgram, clone->ToString());
  XLS_ASSERT_OK(VerifyClone(tf, clone, file_table));
}

TEST(AstClonerTest, TestProcs) {
  constexpr std::string_view kProgram = R"(#[test_proc]
proc my_test_proc {
    a: u32;
    b: uN[127];
    terminator: chan<bool> out;
    config(terminator: chan<bool> out) {
        (u32:0, uN[127]:127, terminator)
    }
    init {
        u64:1
    }
    next(state: u64) {
        state + a as u64 + b as u64
    }
})";

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(auto module, ParseModule(kProgram, "fake_path.x",
                                                    "the_module", file_table));
  XLS_ASSERT_OK_AND_ASSIGN(TestProc * tp,
                           module->GetMemberOrError<TestProc>("my_test_proc"));
  XLS_ASSERT_OK_AND_ASSIGN(AstNode * clone, CloneAst(tp));
  EXPECT_EQ(kProgram, clone->ToString());
  XLS_ASSERT_OK(VerifyClone(tp, clone, file_table));
}

TEST(AstClonerTest, EnumDef) {
  constexpr std::string_view kProgram = R"(enum MyEnum : u32 {
    PET = 0,
    ALL = 1,
    DOGS = 2,
    FOREVER = 3,
})";

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(auto module, ParseModule(kProgram, "fake_path.x",
                                                    "the_module", file_table));
  XLS_ASSERT_OK_AND_ASSIGN(EnumDef * enum_def,
                           module->GetMemberOrError<EnumDef>("MyEnum"));
  XLS_ASSERT_OK_AND_ASSIGN(AstNode * clone, CloneAst(enum_def));
  EXPECT_EQ(kProgram, clone->ToString());
  XLS_ASSERT_OK(VerifyClone(enum_def, clone, file_table));
}

TEST(AstClonerTest, TypeAlias) {
  constexpr std::string_view kProgram =
      R"(type RobsUnnecessaryType = uN[0xbeef];)";

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(auto module, ParseModule(kProgram, "fake_path.x",
                                                    "the_module", file_table));
  XLS_ASSERT_OK_AND_ASSIGN(
      TypeAlias * type_alias,
      module->GetMemberOrError<TypeAlias>("RobsUnnecessaryType"));
  XLS_ASSERT_OK_AND_ASSIGN(AstNode * clone, CloneAst(type_alias));
  EXPECT_EQ(kProgram, clone->ToString());
  XLS_ASSERT_OK(VerifyClone(type_alias, clone, file_table));
}

// This is an interesting case because the start and limit of the slice are
// nullptr.
TEST(AstClonerTest, SliceWithNullptrs) {
  constexpr std::string_view kProgram =
      R"(const MOL = u32:42; const MOL2 = MOL[:];)";

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(auto module, ParseModule(kProgram, "fake_path.x",
                                                    "the_module", file_table));
  XLS_ASSERT_OK_AND_ASSIGN(ConstantDef * top_member,
                           module->GetMemberOrError<ConstantDef>("MOL2"));
  XLS_ASSERT_OK_AND_ASSIGN(AstNode * clone, CloneAst(top_member));
  EXPECT_EQ("const MOL2 = MOL[:];", clone->ToString());
  XLS_ASSERT_OK(VerifyClone(top_member, clone, file_table));
}

TEST(AstClonerTest, PreserveTypeDefinitionsReplacer) {
  constexpr std::string_view kProgram =
      R"(
type my_type = u32;
fn foo() -> u32 {
  zero!<my_type>()
}
)";

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(auto module, ParseModule(kProgram, "fake_path.x",
                                                    "the_module", file_table));
  XLS_ASSERT_OK_AND_ASSIGN(Function * foo,
                           module->GetMemberOrError<Function>("foo"));
  XLS_ASSERT_OK_AND_ASSIGN(AstNode * clone,
                           CloneAst(foo, &PreserveTypeDefinitionsReplacer));
  std::optional<TypeRef*> type_ref = FindFirstTypeRef(foo);
  ASSERT_TRUE(type_ref.has_value());
  std::optional<TypeRef*> cloned_type_ref = FindFirstTypeRef(clone);
  ASSERT_TRUE(cloned_type_ref.has_value());
  EXPECT_EQ((*cloned_type_ref)->type_definition(),
            (*type_ref)->type_definition());
}

TEST(AstClonerTest, QuickCheck) {
  constexpr std::string_view kProgram = R"(#[quickcheck(test_count=1000)]
fn my_quickcheck(a: u32, b: u64, c: sN[128]) {
    (a + b) + c == a + (b + c)
})";

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(auto module, ParseModule(kProgram, "fake_path.x",
                                                    "the_module", file_table));
  XLS_ASSERT_OK_AND_ASSIGN(
      QuickCheck * quick_check,
      module->GetMemberOrError<QuickCheck>("my_quickcheck"));
  XLS_ASSERT_OK_AND_ASSIGN(AstNode * clone, CloneAst(quick_check));
  EXPECT_EQ(kProgram, clone->ToString());
  XLS_ASSERT_OK(VerifyClone(quick_check, clone, file_table));
}

TEST(AstClonerTest, CloneModule) {
  constexpr std::string_view kProgram = R"(import my_import;
enum MyEnum : u8 {
    DOGS = 0,
    ARE = 1,
    GOOD = 2,
}

fn my_function(a: u32) -> u16 {
    a as u16
}

proc my_proc {
    a: u8;
    b: u32;
    init {
        u16:0
    }
    config() {
        (u8:32, u32:8)
    }
    next(state: u16) {
        let x = my_function(state as u32);
        a as u16 + b as u16 + x
    }
})";

  // Note that we're dealing with a post-parsing AST, which means that the proc
  // config and next functions will be present as top-level functions.
  constexpr std::string_view kExpected = R"(import my_import;
enum MyEnum : u8 {
    DOGS = 0,
    ARE = 1,
    GOOD = 2,
}
fn my_function(a: u32) -> u16 {
    a as u16
}
fn my_proc.init() -> u16 {
    u16:0
}
fn my_proc.config() -> (u8, u32) {
    (u8:32, u32:8)
}
fn my_proc.next(state: u16) -> u16 {
    let x = my_function(state as u32);
    a as u16 + b as u16 + x
}
proc my_proc {
    a: u8;
    b: u32;
    config() {
        (u8:32, u32:8)
    }
    init {
        u16:0
    }
    next(state: u16) {
        let x = my_function(state as u32);
        a as u16 + b as u16 + x
    }
})";

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(auto module, ParseModule(kProgram, "fake_path.x",
                                                    "the_module", file_table));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> clone,
                           CloneModule(module.get()));
  EXPECT_EQ(kExpected, clone->ToString());
  XLS_ASSERT_OK(VerifyClone(module.get(), clone.get(), file_table));
}

TEST(AstClonerTest, IndexVariants) {
  constexpr std::string_view kProgram = R"(fn main() {
    let array = u32[5]:[u32:0, u32:1, u32:2, u32:3, u32:4];
    let index = array[2];
    let slice = array[3][0:2];
    let width_slice = array[3][array[0]+:u4];
    ()
})";

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(auto module, ParseModule(kProgram, "fake_path.x",
                                                    "the_module", file_table));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> clone,
                           CloneModule(module.get()));
  EXPECT_EQ(kProgram, clone->ToString());
  XLS_ASSERT_OK(VerifyClone(module.get(), clone.get(), file_table));
}

TEST(AstClonerTest, SplatStructInstance) {
  constexpr std::string_view kProgram = R"(struct MyStruct {
    a: u32,
    b: u33,
    c: u34,
}
fn main() {
    let x: MyStruct = MyStruct { a: u32:0, b: u33:1, c: u34:0xbeef };
    let y: MyStruct = MyStruct { a: u32:0xf00d, c: u34:0xbef0, ..x };
    ()
})";

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(auto module, ParseModule(kProgram, "fake_path.x",
                                                    "the_module", file_table));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> clone,
                           CloneModule(module.get()));
  EXPECT_EQ(kProgram, clone->ToString());
  XLS_ASSERT_OK(VerifyClone(module.get(), clone.get(), file_table));
}

TEST(AstClonerTest, Ternary) {
  constexpr std::string_view kProgram =
      R"(fn main(a: u32, b: u32, c: u32) -> u32 {
    if a > u32:5 { b } else { c }
})";

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(auto module, ParseModule(kProgram, "fake_path.x",
                                                    "the_module", file_table));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> clone,
                           CloneModule(module.get()));
  EXPECT_EQ(kProgram, clone->ToString());
  XLS_ASSERT_OK(VerifyClone(module.get(), clone.get(), file_table));
}

TEST(AstClonerTest, FormatMacro) {
  constexpr std::string_view kProgram = R"(fn main(x: u32) -> u32 {
    let _ = trace_fmt!("x is {}, {:#x} in hex and {:#b} in binary", x, x, x);
    ()
})";

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(auto module, ParseModule(kProgram, "fake_path.x",
                                                    "the_module", file_table));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> clone,
                           CloneModule(module.get()));
  EXPECT_EQ(kProgram, clone->ToString());
}

TEST(AstClonerTest, Match) {
  // Try to every potential NameDefTree Leaf type (NameRef, NameDef,
  // WildcardPattern, Number, ColonRef).
  constexpr std::string_view kProgram = R"(import foo;
fn main(x: u32, y: u32) -> u32 {
    match (x, y) {
        (u32:0, y) => y,
        (u32:1, a) => a + u32:100,
        (u32:2, _) => foo::IMPORTED_CONSTANT,
        (_, u32:100) => u32:200,
    }
})";

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(auto module, ParseModule(kProgram, "fake_path.x",
                                                    "the_module", file_table));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> clone,
                           CloneModule(module.get()));
  EXPECT_EQ(kProgram, clone->ToString());
}

TEST(AstClonerTest, String) {
  constexpr std::string_view kProgram = R"(fn main() -> u8[13] {
    "dogs are good"
})";

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(auto module, ParseModule(kProgram, "fake_path.x",
                                                    "the_module", file_table));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> clone,
                           CloneModule(module.get()));
  EXPECT_EQ(kProgram, clone->ToString());
}

TEST(AstClonerTest, ConstAssert) {
  constexpr std::string_view kProgram = R"(fn main() {
    const_assert!(true);
})";

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(auto module, ParseModule(kProgram, "fake_path.x",
                                                    "the_module", file_table));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> clone,
                           CloneModule(module.get()));
  EXPECT_EQ(kProgram, clone->ToString());
}

TEST(AstClonerTest, NormalFor) {
  constexpr std::string_view kProgram = R"(fn main() -> u32 {
    for (i, a): (u32, u32) in range(0, u32:100) {
        i + a
    }(u32:0)
})";

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(auto module, ParseModule(kProgram, "fake_path.x",
                                                    "the_module", file_table));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> clone,
                           CloneModule(module.get()));
  EXPECT_EQ(kProgram, clone->ToString());
}

TEST(AstClonerTest, TupleIndex) {
  constexpr std::string_view kProgram = R"(fn main() -> u32 {
    (u8:8, u16:16, u32:32, u64:64).2
})";

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(auto module, ParseModule(kProgram, "fake_path.x",
                                                    "the_module", file_table));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> clone,
                           CloneModule(module.get()));
  EXPECT_EQ(kProgram, clone->ToString());
}

TEST(AstClonerTest, SendsAndRecvsAndSpawns) {
  constexpr std::string_view kProgram = R"(import other_module;
proc MyProc {
    input_p: chan<u32> out;
    output_c: chan<u64> out;
    config() {
        let (input_p, input_c) = chan<u32>("input");
        let (output_p, output_c) = chan<u64>("output");
        spawn other_module::OtherProc(input_c, output_p);
        (input_p, output_c)
    }
    init {
        u32:0
    }
    next(state: u32) {
        let tok = send(join(), input_p, state);
        let tok = send_if(tok, input_p, state > u32:32, state);
        let (tok1, state) = recv(tok, output_c);
        let (tok2, foo) = recv_if(tok, output_c, state > u32:32, u64:0);
        let tok = join(tok1, tok2);
        state + foo
    }
})";
  constexpr std::string_view kExpected = R"(import other_module;
fn MyProc.config() -> (chan<u32> out, chan<u64> out) {
    let (input_p, input_c) = chan<u32>("input");
    let (output_p, output_c) = chan<u64>("output");
    spawn other_module::OtherProc(input_c, output_p);
    (input_p, output_c)
}
fn MyProc.init() -> u32 {
    u32:0
}
fn MyProc.next(state: u32) -> u32 {
    let tok = send(join(), input_p, state);
    let tok = send_if(tok, input_p, state > u32:32, state);
    let (tok1, state) = recv(tok, output_c);
    let (tok2, foo) = recv_if(tok, output_c, state > u32:32, u64:0);
    let tok = join(tok1, tok2);
    state + foo
}
proc MyProc {
    input_p: chan<u32> out;
    output_c: chan<u64> out;
    config() {
        let (input_p, input_c) = chan<u32>("input");
        let (output_p, output_c) = chan<u64>("output");
        spawn other_module::OtherProc(input_c, output_p);
        (input_p, output_c)
    }
    init {
        u32:0
    }
    next(state: u32) {
        let tok = send(join(), input_p, state);
        let tok = send_if(tok, input_p, state > u32:32, state);
        let (tok1, state) = recv(tok, output_c);
        let (tok2, foo) = recv_if(tok, output_c, state > u32:32, u64:0);
        let tok = join(tok1, tok2);
        state + foo
    }
})";

  FileTable file_table;
  absl::StatusOr<std::unique_ptr<Module>> module_or =
      ParseModule(kProgram, "fake_path.x", "the_module", file_table);
  if (!module_or.ok()) {
    TryPrintError(
        module_or.status(),
        [&](std::string_view path) -> absl::StatusOr<std::string> {
          return std::string(kProgram);
        },
        file_table);
  }
  XLS_ASSERT_OK(module_or.status());
  Module* module = module_or.value().get();
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> clone, CloneModule(module));
  EXPECT_EQ(kExpected, clone->ToString());
}

// A ConstRef doesn't own its underlying NameDef. So don't clone it.
TEST(AstClonerTest, DoesntCloneConstRefNameDefs) {
  constexpr std::string_view kProgram = R"(
const FOO = u32:42;
fn bar() -> u32{
  FOO
}
)";

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(auto module, ParseModule(kProgram, "fake_path.x",
                                                    "the_module", file_table));
  XLS_ASSERT_OK_AND_ASSIGN(Function * main,
                           module->GetMemberOrError<Function>("bar"));
  StatementBlock* body_expr = main->body();
  ASSERT_EQ(body_expr->statements().size(), 1);
  ASSERT_TRUE(
      std::holds_alternative<Expr*>(body_expr->statements().at(0)->wrapped()));
  ConstRef* orig_ref = down_cast<ConstRef*>(
      std::get<Expr*>(body_expr->statements().at(0)->wrapped()));
  XLS_ASSERT_OK_AND_ASSIGN(AstNode * clone, CloneAst(orig_ref));
  ConstRef* new_ref = down_cast<ConstRef*>(clone);
  EXPECT_EQ(orig_ref->name_def(), new_ref->name_def());
}

TEST(AstClonerTest, CloneAstClonesVerbatimNode) {
  constexpr std::string_view kProgram = "const FOO = u32:42;";
  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(auto module, ParseModule(kProgram, "fake_path.x",
                                                    "the_module", file_table));

  VerbatimNode original(module.get(), Span(), "foo");
  XLS_ASSERT_OK_AND_ASSIGN(AstNode * clone, CloneAst(&original));
  VerbatimNode* clone_node = down_cast<VerbatimNode*>(clone);
  EXPECT_EQ(original.text(), clone_node->text());
  EXPECT_EQ(original.span(), clone_node->span());
}

TEST(AstClonerTest, CloneStatementBlockSkipsEmptyVerbatimNode) {
  constexpr std::string_view kProgram = "const FOO = {u32:42};";
  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(auto module, ParseModule(kProgram, "fake_path.x",
                                                    "the_module", file_table));

  Number number(module.get(), Span(), "3", NumberKind::kOther, nullptr);
  Statement statement(module.get(), &number);
  StatementBlock statement_block(module.get(), Span(), {&statement},
                                 /*trailing_semi=*/true);

  VerbatimNode empty_verbatim(module.get(), Span());
  XLS_ASSERT_OK_AND_ASSIGN(
      AstNode * clone,
      CloneAst(&statement_block,
               [&](const AstNode* node) -> std::optional<AstNode*> {
                 if (node->kind() == AstNodeKind::kStatement) {
                   // Replace with a verbatim node with no text.
                   return &empty_verbatim;
                 }
                 return std::nullopt;
               }));

  // The clone should have zero children now that we replaced the original node
  // with an empty VerbatimNode.
  StatementBlock* clone_node = down_cast<StatementBlock*>(clone);
  EXPECT_THAT(clone_node->statements(), IsEmpty());
}

TEST(AstClonerTest, CloneStatementBlockUsesVerbatimNode) {
  constexpr std::string_view kProgram = "const FOO = {u32:42};";
  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(auto module, ParseModule(kProgram, "fake_path.x",
                                                    "the_module", file_table));

  Number number(module.get(), Span(), "3", NumberKind::kOther, nullptr);
  Statement statement(module.get(), &number);
  StatementBlock statement_block(module.get(), Span(), {&statement},
                                 /*trailing_semi=*/true);

  VerbatimNode verbatim(module.get(), Span(), "replaced");
  XLS_ASSERT_OK_AND_ASSIGN(
      AstNode * clone,
      CloneAst(&statement_block,
               [&](const AstNode* node) -> std::optional<AstNode*> {
                 if (node->kind() == AstNodeKind::kStatement) {
                   // Replace with the desired verbatim node
                   return &verbatim;
                 }
                 return std::nullopt;
               }));

  StatementBlock* clone_statement_block = down_cast<StatementBlock*>(clone);
  ASSERT_EQ(clone_statement_block->statements().size(), 1);

  Statement* first = clone_statement_block->statements().at(0);
  EXPECT_EQ(std::get<VerbatimNode*>(first->wrapped()), &verbatim);
}

TEST(AstClonerTest, CloneModuleClonesVerbatimNode) {
  constexpr std::string_view kProgram = "const FOO = u32:42;";
  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Module> module,
      ParseModule(kProgram, "fake_path.x", "the_module", file_table));

  VerbatimNode original(module.get(), Span(), "foo");
  XLS_ASSERT_OK(
      module.get()->AddTop(&original, /*make_collision_error=*/nullptr));

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> cloned_module,
                           CloneModule(module.get()));
  EXPECT_EQ(cloned_module->top().size(), 2);

  VerbatimNode* cloned_verbatim_node =
      std::get<VerbatimNode*>(cloned_module->top().at(1));
  ASSERT_NE(cloned_verbatim_node, nullptr);

  EXPECT_EQ(original.text(), cloned_verbatim_node->text());
  EXPECT_EQ(original.span(), cloned_verbatim_node->span());
}

}  // namespace
}  // namespace xls::dslx
