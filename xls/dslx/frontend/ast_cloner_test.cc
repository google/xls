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
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/casts.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/command_line_utils.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/parse_and_typecheck.h"

namespace xls::dslx {
namespace {

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

  XLS_ASSERT_OK_AND_ASSIGN(auto module,
                           ParseModule(kProgram, "fake_path.x", "the_module"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           module->GetMemberOrError<Function>("main"));
  Block* body_expr = f->body();
  XLS_ASSERT_OK_AND_ASSIGN(AstNode * clone, CloneAst(body_expr));
  EXPECT_EQ(kExpected, clone->ToString());
  XLS_ASSERT_OK(VerifyClone(body_expr, clone));
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

  XLS_ASSERT_OK_AND_ASSIGN(auto module,
                           ParseModule(kProgram, "fake_path.x", "the_module"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           module->GetMemberOrError<Function>("main"));
  Block* body_expr = f->body();
  XLS_ASSERT_OK_AND_ASSIGN(AstNode * clone, CloneAst(body_expr));
  EXPECT_EQ(kExpected, clone->ToString());
  XLS_ASSERT_OK(VerifyClone(body_expr, clone));
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

  XLS_ASSERT_OK_AND_ASSIGN(auto module,
                           ParseModule(kProgram, "fake_path.x", "the_module"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           module->GetMemberOrError<Function>("main"));
  Block* body_expr = f->body();
  XLS_ASSERT_OK_AND_ASSIGN(AstNode * clone, CloneAst(body_expr));
  EXPECT_EQ(kExpected, clone->ToString());
  XLS_ASSERT_OK(VerifyClone(body_expr, clone));
}

TEST(AstClonerTest, BasicFunction) {
  constexpr std::string_view kProgram = R"(fn main() -> (u32, u32) {
    let a = u32:0;
    let b = u32:1;
    (a, b)
})";

  XLS_ASSERT_OK_AND_ASSIGN(auto module,
                           ParseModule(kProgram, "fake_path.x", "the_module"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           module->GetMemberOrError<Function>("main"));
  XLS_ASSERT_OK_AND_ASSIGN(AstNode * clone, CloneAst(f));
  EXPECT_EQ(kProgram, clone->ToString());
  XLS_ASSERT_OK(VerifyClone(f, clone));
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

  XLS_ASSERT_OK_AND_ASSIGN(auto module,
                           ParseModule(kProgram, "fake_path.x", "the_module"));
  StructDef* struct_def = module->GetStructDefs().at(0);
  XLS_ASSERT_OK_AND_ASSIGN(AstNode * clone, CloneAst(struct_def));
  EXPECT_EQ(kExpectedStructDef, clone->ToString());

  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           module->GetMemberOrError<Function>("main"));
  XLS_ASSERT_OK_AND_ASSIGN(clone, CloneAst(f));
  EXPECT_EQ(kExpectedFunction, clone->ToString());
  XLS_ASSERT_OK(VerifyClone(f, clone));
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

  XLS_ASSERT_OK_AND_ASSIGN(auto module,
                           ParseModule(kProgram, "fake_path.x", "the_module"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           module->GetMemberOrError<Function>("main"));
  XLS_ASSERT_OK_AND_ASSIGN(AstNode * clone, CloneAst(f));
  EXPECT_EQ(kExpectedFunction, clone->ToString());
  XLS_ASSERT_OK(VerifyClone(f, clone));
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

  XLS_ASSERT_OK_AND_ASSIGN(auto module,
                           ParseModule(kProgram, "fake_path.x", "the_module"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           module->GetMemberOrError<Function>("main"));
  XLS_ASSERT_OK_AND_ASSIGN(AstNode * clone, CloneAst(f));
  EXPECT_EQ(kExpectedFunction, clone->ToString());
  XLS_ASSERT_OK(VerifyClone(f, clone));
}

TEST(AstClonerTest, Binops) {
  constexpr std::string_view kProgram = R"(fn main() -> u13 {
    u13:5 + u13:500
})";

  XLS_ASSERT_OK_AND_ASSIGN(auto module,
                           ParseModule(kProgram, "fake_path.x", "the_module"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           module->GetMemberOrError<Function>("main"));
  XLS_ASSERT_OK_AND_ASSIGN(AstNode * clone, CloneAst(f));
  EXPECT_EQ(kProgram, clone->ToString());
  XLS_ASSERT_OK(VerifyClone(f, clone));
}

TEST(AstClonerTest, Unops) {
  constexpr std::string_view kProgram = R"(fn main() -> u13 {
    -u13:500
})";

  XLS_ASSERT_OK_AND_ASSIGN(auto module,
                           ParseModule(kProgram, "fake_path.x", "the_module"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           module->GetMemberOrError<Function>("main"));
  XLS_ASSERT_OK_AND_ASSIGN(AstNode * clone, CloneAst(f));
  EXPECT_EQ(kProgram, clone->ToString());
  XLS_ASSERT_OK(VerifyClone(f, clone));
}

TEST(AstClonerTest, Casts) {
  constexpr std::string_view kProgram = R"(fn main() -> u13 {
    -u17:500 as u13
})";

  XLS_ASSERT_OK_AND_ASSIGN(auto module,
                           ParseModule(kProgram, "fake_path.x", "the_module"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           module->GetMemberOrError<Function>("main"));
  XLS_ASSERT_OK_AND_ASSIGN(AstNode * clone, CloneAst(f));
  EXPECT_EQ(kProgram, clone->ToString());
  XLS_ASSERT_OK(VerifyClone(f, clone));
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
    next(tok: token, state: u19) {
        (a as u64 + b) as u19
    }
})";

  XLS_ASSERT_OK_AND_ASSIGN(auto module,
                           ParseModule(kProgram, "fake_path.x", "the_module"));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * p, module->GetMemberOrError<Proc>("MyProc"));
  XLS_ASSERT_OK_AND_ASSIGN(AstNode * clone, CloneAst(p));
  EXPECT_EQ(kProgram, clone->ToString());
  XLS_ASSERT_OK(VerifyClone(p, clone));
}

TEST(AstClonerTest, TestFunctions) {
  constexpr std::string_view kProgram = R"(#[test]
fn my_test() {
    let a = u32:0;
    let _ = assert_eq(u32:0, a);
    ()
})";

  XLS_ASSERT_OK_AND_ASSIGN(auto module,
                           ParseModule(kProgram, "fake_path.x", "the_module"));
  XLS_ASSERT_OK_AND_ASSIGN(TestFunction * tf,
                           module->GetMemberOrError<TestFunction>("my_test"));
  XLS_ASSERT_OK_AND_ASSIGN(AstNode * clone, CloneAst(tf));
  EXPECT_EQ(kProgram, clone->ToString());
  XLS_ASSERT_OK(VerifyClone(tf, clone));
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
    next(tok: token, state: u64) {
        state + a as u64 + b as u64
    }
})";

  XLS_ASSERT_OK_AND_ASSIGN(auto module,
                           ParseModule(kProgram, "fake_path.x", "the_module"));
  XLS_ASSERT_OK_AND_ASSIGN(TestProc * tp,
                           module->GetMemberOrError<TestProc>("my_test_proc"));
  XLS_ASSERT_OK_AND_ASSIGN(AstNode * clone, CloneAst(tp));
  EXPECT_EQ(kProgram, clone->ToString());
  XLS_ASSERT_OK(VerifyClone(tp, clone));
}

TEST(AstClonerTest, EnumDef) {
  constexpr std::string_view kProgram = R"(enum MyEnum : u32 {
    PET = 0,
    ALL = 1,
    DOGS = 2,
    FOREVER = 3,
})";

  XLS_ASSERT_OK_AND_ASSIGN(auto module,
                           ParseModule(kProgram, "fake_path.x", "the_module"));
  XLS_ASSERT_OK_AND_ASSIGN(EnumDef * enum_def,
                           module->GetMemberOrError<EnumDef>("MyEnum"));
  XLS_ASSERT_OK_AND_ASSIGN(AstNode * clone, CloneAst(enum_def));
  EXPECT_EQ(kProgram, clone->ToString());
  XLS_ASSERT_OK(VerifyClone(enum_def, clone));
}

TEST(AstClonerTest, TypeAlias) {
  constexpr std::string_view kProgram =
      R"(type RobsUnnecessaryType = uN[0xbeef];)";

  XLS_ASSERT_OK_AND_ASSIGN(auto module,
                           ParseModule(kProgram, "fake_path.x", "the_module"));
  XLS_ASSERT_OK_AND_ASSIGN(
      TypeAlias * type_alias,
      module->GetMemberOrError<TypeAlias>("RobsUnnecessaryType"));
  XLS_ASSERT_OK_AND_ASSIGN(AstNode * clone, CloneAst(type_alias));
  EXPECT_EQ(kProgram, clone->ToString());
  XLS_ASSERT_OK(VerifyClone(type_alias, clone));
}

TEST(AstClonerTest, QuickCheck) {
  constexpr std::string_view kProgram = R"(#[quickcheck(test_count=1000)]
fn my_quickcheck(a: u32, b: u64, c: sN[128]) {
    (a + b) + c == a + (b + c)
})";

  XLS_ASSERT_OK_AND_ASSIGN(auto module,
                           ParseModule(kProgram, "fake_path.x", "the_module"));
  XLS_ASSERT_OK_AND_ASSIGN(
      QuickCheck * quick_check,
      module->GetMemberOrError<QuickCheck>("my_quickcheck"));
  XLS_ASSERT_OK_AND_ASSIGN(AstNode * clone, CloneAst(quick_check));
  EXPECT_EQ(kProgram, clone->ToString());
  XLS_ASSERT_OK(VerifyClone(quick_check, clone));
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
    next(tok: token, state: u16) {
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
fn my_proc.next(tok: token, state: u16) -> u16 {
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
    next(tok: token, state: u16) {
        let x = my_function(state as u32);
        a as u16 + b as u16 + x
    }
})";

  XLS_ASSERT_OK_AND_ASSIGN(auto module,
                           ParseModule(kProgram, "fake_path.x", "the_module"));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> clone,
                           CloneModule(module.get()));
  EXPECT_EQ(kExpected, clone->ToString());
  XLS_ASSERT_OK(VerifyClone(module.get(), clone.get()));
}

TEST(AstClonerTest, IndexVariants) {
  constexpr std::string_view kProgram = R"(fn main() {
    let array = u32[5]:[u32:0, u32:1, u32:2, u32:3, u32:4];
    let index = array[2];
    let slice = array[3][0:2];
    let width_slice = array[3][array[0]+:u4];
    ()
})";

  XLS_ASSERT_OK_AND_ASSIGN(auto module,
                           ParseModule(kProgram, "fake_path.x", "the_module"));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> clone,
                           CloneModule(module.get()));
  EXPECT_EQ(kProgram, clone->ToString());
  XLS_ASSERT_OK(VerifyClone(module.get(), clone.get()));
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

  XLS_ASSERT_OK_AND_ASSIGN(auto module,
                           ParseModule(kProgram, "fake_path.x", "the_module"));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> clone,
                           CloneModule(module.get()));
  EXPECT_EQ(kProgram, clone->ToString());
  XLS_ASSERT_OK(VerifyClone(module.get(), clone.get()));
}

TEST(AstClonerTest, Ternary) {
  constexpr std::string_view kProgram =
      R"(fn main(a: u32, b: u32, c: u32) -> u32 {
    if a > u32:5 { b } else { c }
})";

  XLS_ASSERT_OK_AND_ASSIGN(auto module,
                           ParseModule(kProgram, "fake_path.x", "the_module"));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> clone,
                           CloneModule(module.get()));
  EXPECT_EQ(kProgram, clone->ToString());
  XLS_ASSERT_OK(VerifyClone(module.get(), clone.get()));
}

TEST(AstClonerTest, FormatMacro) {
  constexpr std::string_view kProgram = R"(fn main(x: u32) -> u32 {
    let _ = trace_fmt!("x is {}, {:#x} in hex and {:#b} in binary", x, x, x);
    ()
})";

  XLS_ASSERT_OK_AND_ASSIGN(auto module,
                           ParseModule(kProgram, "fake_path.x", "the_module"));
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

  XLS_ASSERT_OK_AND_ASSIGN(auto module,
                           ParseModule(kProgram, "fake_path.x", "the_module"));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> clone,
                           CloneModule(module.get()));
  EXPECT_EQ(kProgram, clone->ToString());
}

TEST(AstClonerTest, String) {
  constexpr std::string_view kProgram = R"(fn main() -> u8[13] {
    "dogs are good"
})";

  XLS_ASSERT_OK_AND_ASSIGN(auto module,
                           ParseModule(kProgram, "fake_path.x", "the_module"));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> clone,
                           CloneModule(module.get()));
  EXPECT_EQ(kProgram, clone->ToString());
}

TEST(AstClonerTest, ConstAssert) {
  constexpr std::string_view kProgram = R"(fn main() {
    const_assert!(true);
})";

  XLS_ASSERT_OK_AND_ASSIGN(auto module,
                           ParseModule(kProgram, "fake_path.x", "the_module"));
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

  XLS_ASSERT_OK_AND_ASSIGN(auto module,
                           ParseModule(kProgram, "fake_path.x", "the_module"));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> clone,
                           CloneModule(module.get()));
  EXPECT_EQ(kProgram, clone->ToString());
}

TEST(AstClonerTest, TupleIndex) {
  constexpr std::string_view kProgram = R"(fn main() -> u32 {
    (u8:8, u16:16, u32:32, u64:64).2
})";

  XLS_ASSERT_OK_AND_ASSIGN(auto module,
                           ParseModule(kProgram, "fake_path.x", "the_module"));
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
    next(tok: token, state: u32) {
        let tok = send(tok, input_p, state);
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
fn MyProc.next(tok: token, state: u32) -> u32 {
    let tok = send(tok, input_p, state);
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
    next(tok: token, state: u32) {
        let tok = send(tok, input_p, state);
        let tok = send_if(tok, input_p, state > u32:32, state);
        let (tok1, state) = recv(tok, output_c);
        let (tok2, foo) = recv_if(tok, output_c, state > u32:32, u64:0);
        let tok = join(tok1, tok2);
        state + foo
    }
})";

  absl::StatusOr<std::unique_ptr<Module>> module_or =
      ParseModule(kProgram, "fake_path.x", "the_module");
  if (!module_or.ok()) {
    TryPrintError(module_or.status(),
                  [&](std::string_view path) -> absl::StatusOr<std::string> {
                    return std::string(kProgram);
                  });
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

  XLS_ASSERT_OK_AND_ASSIGN(auto module,
                           ParseModule(kProgram, "fake_path.x", "the_module"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * main,
                           module->GetMemberOrError<Function>("bar"));
  Block* body_expr = main->body();
  ASSERT_EQ(body_expr->statements().size(), 1);
  ASSERT_TRUE(
      std::holds_alternative<Expr*>(body_expr->statements().at(0)->wrapped()));
  ConstRef* orig_ref = down_cast<ConstRef*>(
      std::get<Expr*>(body_expr->statements().at(0)->wrapped()));
  XLS_ASSERT_OK_AND_ASSIGN(AstNode * clone, CloneAst(orig_ref));
  ConstRef* new_ref = down_cast<ConstRef*>(clone);
  EXPECT_EQ(orig_ref->name_def(), new_ref->name_def());
}

}  // namespace
}  // namespace xls::dslx
