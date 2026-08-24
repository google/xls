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
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/parse_and_typecheck.h"

namespace xls::dslx {
namespace {

// Helper function to transform a test function into a test proc, and typecheck
// the resulting module.
absl::StatusOr<std::string> TransformAndTypecheck(std::string_view program) {
  auto import_data = CreateImportDataForTest();
  XLS_ASSIGN_OR_RETURN(
      TypecheckedModule tm,
      ParseAndTypecheck(program, "test.x", "test", &import_data));
  TestFunctionTransformer transformer(*tm.module, *tm.type_info);
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Module> new_module,
                       transformer.TransformTestFunctions());

  // Remove "spawn" trait placeholder methods from the string.
  std::string transformed_code =
      absl::StrReplaceAll(new_module->ToString(), {
                                                      {"fn spawn(self) ;", ""},
                                                  });

  XLS_RETURN_IF_ERROR(ParseAndTypecheck(transformed_code, "transformed.x",
                                        "test_cloned", &import_data)
                          .status())
      << transformed_code;
  return transformed_code;
}

TEST(TestFunctionTransformerTest, NoTestFunctionsUnchanged) {
  constexpr std::string_view kProgram = R"(fn main() -> u32 {
    u32:0
})";
  XLS_ASSERT_OK_AND_ASSIGN(std::string cloned_code,
                           TransformAndTypecheck(kProgram));
  EXPECT_EQ(cloned_code, kProgram);
}

TEST(TestFunctionTransformerTest, TestFunctionNoSpawnsUnchanged) {
  constexpr std::string_view kProgram = R"(#[test]
fn main() {
    ()
})";
  XLS_ASSERT_OK_AND_ASSIGN(std::string cloned_code,
                           TransformAndTypecheck(kProgram));
  EXPECT_EQ(cloned_code, kProgram);
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
  XLS_ASSERT_OK_AND_ASSIGN(std::string cloned_code,
                           TransformAndTypecheck(kProgram));
  EXPECT_FALSE(absl::StrContains(cloned_code, "fn main()"));
  EXPECT_TRUE(
      absl::StrContains(cloned_code, "__test__terminator: chan<bool> out"));
  EXPECT_TRUE(
      absl::StrContains(cloned_code, "#[test]\nproc __test__proc__main"));
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
  XLS_ASSERT_OK_AND_ASSIGN(std::string cloned_code,
                           TransformAndTypecheck(kProgram));
  EXPECT_FALSE(absl::StrContains(cloned_code, "fn main()"));

  size_t impl_pos = cloned_code.find("impl __test__proc__main");
  ASSERT_NE(impl_pos, std::string::npos);
  std::string impl_code = cloned_code.substr(impl_pos);

  size_t new_pos = impl_code.find("fn new(");
  size_t next_pos = impl_code.find("fn next(");

  ASSERT_NE(new_pos, std::string::npos);
  ASSERT_NE(next_pos, std::string::npos);

  std::string new_body = impl_code.substr(new_pos, next_pos - new_pos);
  EXPECT_TRUE(absl::StrContains(new_body, "chan<u32>(\"pin\")"));
  EXPECT_TRUE(absl::StrContains(new_body, "P::new(r)"));
  EXPECT_TRUE(absl::StrContains(new_body, "p.spawn()"));

  std::string next_body = impl_code.substr(next_pos);
  EXPECT_TRUE(absl::StrContains(next_body, "send(token(), self.s, u32:0)"));
  EXPECT_TRUE(absl::StrContains(next_body, "self.__test__terminator"));
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
  XLS_ASSERT_OK_AND_ASSIGN(std::string cloned_code,
                           TransformAndTypecheck(kProgram));

  size_t impl_pos = cloned_code.find("impl __test__proc__main");
  ASSERT_NE(impl_pos, std::string::npos);
  std::string impl_code = cloned_code.substr(impl_pos);

  size_t new_pos = impl_code.find("fn new(");
  size_t next_pos = impl_code.find("fn next(");

  ASSERT_NE(new_pos, std::string::npos);
  ASSERT_NE(next_pos, std::string::npos);

  std::string new_body = impl_code.substr(new_pos, next_pos - new_pos);

  EXPECT_TRUE(absl::StrContains(new_body, "let x = u32:42;"));
  EXPECT_TRUE(absl::StrContains(new_body, "let p = P::new(x);"));
  EXPECT_TRUE(absl::StrContains(new_body, "p.spawn();"));

  // Struct instance is returned by "new"; terminator channel is added to the
  // struct instance.
  EXPECT_TRUE(absl::StrContains(
      new_body,
      "__test__proc__main { __test__terminator: __test__terminator }"));
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
  XLS_ASSERT_OK_AND_ASSIGN(std::string cloned_code,
                           TransformAndTypecheck(kProgram));

  size_t impl_pos = cloned_code.find("impl __test__proc__main");
  ASSERT_NE(impl_pos, std::string::npos);
  std::string impl_code = cloned_code.substr(impl_pos);

  size_t new_pos = impl_code.find("fn new(");
  size_t next_pos = impl_code.find("fn next(");

  ASSERT_NE(new_pos, std::string::npos);
  ASSERT_NE(next_pos, std::string::npos);

  std::string new_body = impl_code.substr(new_pos, next_pos - new_pos);
  EXPECT_TRUE(absl::StrContains(new_body, "chan<u32>(\"c\")"));
  EXPECT_TRUE(absl::StrContains(new_body, "P::new()"));
  EXPECT_TRUE(absl::StrContains(new_body, "p.spawn()"));

  std::string next_body = impl_code.substr(next_pos);
  EXPECT_TRUE(absl::StrContains(next_body, "send(token(), self.tx, u32:42)"));
  EXPECT_TRUE(absl::StrContains(next_body, "assert_eq(u32:1, u32:1)"));
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
  XLS_ASSERT_OK_AND_ASSIGN(std::string cloned_code,
                           TransformAndTypecheck(kProgram));

  size_t impl_pos = cloned_code.find("impl __test__proc__main");
  ASSERT_NE(impl_pos, std::string::npos);
  std::string impl_code = cloned_code.substr(impl_pos);

  size_t new_pos = impl_code.find("fn new(");
  size_t next_pos = impl_code.find("fn next(");

  ASSERT_NE(new_pos, std::string::npos);
  ASSERT_NE(next_pos, std::string::npos);

  std::string new_body = impl_code.substr(new_pos, next_pos - new_pos);
  EXPECT_TRUE(absl::StrContains(new_body, "chan<u32>(\"c1\")"));
  EXPECT_TRUE(absl::StrContains(new_body, "chan<u32>(\"c2\")"));
  EXPECT_TRUE(absl::StrContains(new_body, "p1 = P::new()"));
  EXPECT_TRUE(absl::StrContains(new_body, "p2 = P::new()"));
  EXPECT_TRUE(absl::StrContains(new_body, "p1.spawn()"));
  EXPECT_TRUE(absl::StrContains(new_body, "p2.spawn()"));

  std::string next_body = impl_code.substr(next_pos);
  EXPECT_TRUE(absl::StrContains(next_body, "send(token(), self.tx1, u32:1)"));
  EXPECT_TRUE(absl::StrContains(next_body, "send(token(), self.tx2, u32:2)"));
}

TEST(TestFunctionTransformerTest, TestFunctionWithDestructuredLets) {
  constexpr std::string_view kProgram = R"(
proc P {
}
impl P {
    fn new() -> Self { P {} }
}

#[test]
fn main() {
    let (tx, rx) = chan<u32>("c");
    let (p, (dummy1, dummy2)) = (P::new(), (u32:1, u32:2));
    p.spawn();
    send(token(), tx, u32:42);
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::string cloned_code,
                           TransformAndTypecheck(kProgram));

  size_t impl_pos = cloned_code.find("impl __test__proc__main");
  ASSERT_NE(impl_pos, std::string::npos);
  std::string impl_code = cloned_code.substr(impl_pos);

  size_t new_pos = impl_code.find("fn new(");
  size_t next_pos = impl_code.find("fn next(");

  ASSERT_NE(new_pos, std::string::npos);
  ASSERT_NE(next_pos, std::string::npos);

  std::string new_body = impl_code.substr(new_pos, next_pos - new_pos);
  EXPECT_TRUE(absl::StrContains(new_body, "let (tx, rx) = chan<u32>(\"c\");"));
  EXPECT_TRUE(absl::StrContains(
      new_body, "let (p, (dummy1, dummy2)) = (P::new(), (u32:1, u32:2));"));
  EXPECT_TRUE(absl::StrContains(new_body, "p.spawn();"));

  std::string next_body = impl_code.substr(next_pos);
  // tx must be promoted.
  EXPECT_TRUE(absl::StrContains(next_body, "send(token(), self.tx, u32:42)"));
}

TEST(TestFunctionTransformerTest, TestFunctionWithShadowedVariables) {
  constexpr std::string_view kProgram = R"(#![feature(explicit_state_access)]
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

    let x = u32:1;
    send(token(), tx, x);

    let x = u32:2;
    send(token(), tx, x);
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::string cloned_code,
                           TransformAndTypecheck(kProgram));

  size_t impl_pos = cloned_code.find("impl __test__proc__main");
  ASSERT_NE(impl_pos, std::string::npos);
  std::string impl_code = cloned_code.substr(impl_pos);

  size_t new_pos = impl_code.find("fn new(");
  size_t next_pos = impl_code.find("fn next(");

  ASSERT_NE(new_pos, std::string::npos);
  ASSERT_NE(next_pos, std::string::npos);

  std::string new_body = impl_code.substr(new_pos, next_pos - new_pos);
  EXPECT_TRUE(absl::StrContains(new_body, "let (tx, rx) = chan<u32>(\"c\");"))
      << new_body;
  EXPECT_TRUE(absl::StrContains(new_body, "let x = u32:1;")) << new_body;
  EXPECT_TRUE(absl::StrContains(new_body, "let x_1 = u32:2;")) << new_body;

  std::string next_body = impl_code.substr(next_pos);
  // tx and x and shadowed x must be promoted.
  EXPECT_TRUE(
      absl::StrContains(next_body, "send(token(), self.tx, read(self.x))"))
      << next_body;
  EXPECT_TRUE(
      absl::StrContains(next_body, "send(token(), self.tx, read(self.x_1))"))
      << next_body;
}

TEST(TestFunctionTransformerTest, TestFunctionWithForLoop) {
  constexpr std::string_view kProgram = R"(#![feature(explicit_state_access)]
proc P {
    r: chan<u32> in,
    s: chan<u32> out,
}
impl P {
    fn new(r: chan<u32> in, s: chan<u32> out) -> Self { P { r, s } }
    fn next(self) {
        let (tok, val) = recv(token(), self.r);
        send(tok, self.s, val + u32:1);
    }
}

#[test]
fn main() {
    let (tx_in, rx_in) = chan<u32>("in");
    let (tx_out, rx_out) = chan<u32>("out");
    P::new(rx_in, tx_out).spawn();

    let tok = token();
    let (tok, accum) = for (i, (tok, accum)) in u32:0..u32:5 {
        let tok = send(tok, tx_in, i);
        let (tok, val) = recv(tok, rx_out);
        (tok, accum + val)
    }((tok, u32:0));

    assert_eq(accum, u32:15);
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::string cloned_code,
                           TransformAndTypecheck(kProgram));

  size_t impl_pos = cloned_code.find("impl __test__proc__main");
  ASSERT_NE(impl_pos, std::string::npos);
  std::string impl_code = cloned_code.substr(impl_pos);

  size_t new_pos = impl_code.find("fn new(");
  size_t next_pos = impl_code.find("fn next(");

  ASSERT_NE(new_pos, std::string::npos);
  ASSERT_NE(next_pos, std::string::npos);

  std::string new_body = impl_code.substr(new_pos, next_pos - new_pos);
  EXPECT_TRUE(
      absl::StrContains(new_body, "let (tx_in, rx_in) = chan<u32>(\"in\");"))
      << new_body;
  EXPECT_TRUE(
      absl::StrContains(new_body, "let (tx_out, rx_out) = chan<u32>(\"out\");"))
      << new_body;

  std::string next_body = impl_code.substr(next_pos);
  // tx_in and rx_out must be promoted and rewritten inside the for loop.
  EXPECT_TRUE(absl::StrContains(next_body, "send(tok, self.tx_in, i)"))
      << next_body;
  EXPECT_TRUE(absl::StrContains(next_body, "recv(tok, self.rx_out)"))
      << next_body;
}

TEST(TestFunctionTransformerTest, TestFunctionWithUnrollFor) {
  constexpr std::string_view kProgram = R"(#![feature(explicit_state_access)]
proc P {
    r: chan<u32> in,
}
impl P {
    fn new(r: chan<u32> in) -> Self { P { r } }
}

#[test]
fn main() {
    let (tx, rx) = chan<u32>("c");
    P::new(rx).spawn();

    let my_array = unroll_for! (i, accum) in u32:0..u32:4 {
        accum
    }([u32:1, u32:2, u32:3, u32:4]);

    send(token(), tx, my_array[u32:0]);
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::string cloned_code,
                           TransformAndTypecheck(kProgram));

  size_t impl_pos = cloned_code.find("impl __test__proc__main");
  ASSERT_NE(impl_pos, std::string::npos);
  std::string impl_code = cloned_code.substr(impl_pos);

  size_t new_pos = impl_code.find("fn new(");
  size_t next_pos = impl_code.find("fn next(");

  ASSERT_NE(new_pos, std::string::npos);
  ASSERT_NE(next_pos, std::string::npos);

  std::string new_body = impl_code.substr(new_pos, next_pos - new_pos);
  EXPECT_TRUE(absl::StrContains(new_body, "let (tx, rx) = chan<u32>(\"c\");"))
      << new_body;
  EXPECT_TRUE(absl::StrContains(new_body, "let my_array = unroll_for!"))
      << new_body;

  std::string next_body = impl_code.substr(next_pos);
  // tx and my_array must be promoted.
  EXPECT_TRUE(absl::StrContains(
      next_body, "send(token(), self.tx, read(self.my_array)[u32:0])"))
      << next_body;
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
  // Note: does not use the helper function so we can test that the statements
  // are cloned into the new module, and not left as dangling pointers.
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

TEST(TestFunctionTransformerTest, TestFunctionWithPromotedTuple) {
  constexpr std::string_view kProgram = R"(#![feature(explicit_state_access)]
proc P {
    c: chan<u32> out,
}
impl P {
    fn new(c: chan<u32> out) -> Self { P { c } }
}

#[test]
fn main() {
    let (tx, rx) = chan<u32>("c");
    let t = (u32:42, u32:100);
    P::new(tx).spawn();
    let tok = send(token(), tx, t.0);
    send(tok, tx, t.1);
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::string cloned_code,
                           TransformAndTypecheck(kProgram));

  size_t impl_pos = cloned_code.find("impl __test__proc__main");
  ASSERT_NE(impl_pos, std::string::npos);
  std::string impl_code = cloned_code.substr(impl_pos);

  size_t new_pos = impl_code.find("fn new(");
  size_t next_pos = impl_code.find("fn next(");

  ASSERT_NE(new_pos, std::string::npos);
  ASSERT_NE(next_pos, std::string::npos);

  std::string new_body = impl_code.substr(new_pos, next_pos - new_pos);
  EXPECT_TRUE(absl::StrContains(new_body, "let t = (u32:42, u32:100);"));

  std::string next_body = impl_code.substr(next_pos);
  // tx and t must be promoted.
  EXPECT_TRUE(absl::StrContains(
      next_body, "let tok = send(token(), self.tx, read(self.t).0);"));
  EXPECT_TRUE(
      absl::StrContains(next_body, "send(tok, self.tx, read(self.t).1);"));
}

TEST(TestFunctionTransformerTest, TestFunctionWithUnpromotedTuple) {
  constexpr std::string_view kProgram = R"(#![feature(explicit_state_access)]
proc P {
    c: chan<u32> out,
}
impl P {
    fn new(c: chan<u32> out) -> Self { P { c } }
}

#[test]
fn main() {
    let (tx, rx) = chan<u32>("c");
    let my_tuple = (u32:1, u32:2);
    let sum = my_tuple.0 + my_tuple.1;
    P::new(tx).spawn();
    send(token(), tx, sum);
}
)";
  // This test should pass because 'my_tuple' is not promoted (only 'tx' and
  // 'sum' are).
  XLS_ASSERT_OK_AND_ASSIGN(std::string cloned_code,
                           TransformAndTypecheck(kProgram));

  size_t impl_pos = cloned_code.find("impl __test__proc__main");
  ASSERT_NE(impl_pos, std::string::npos);
  std::string impl_code = cloned_code.substr(impl_pos);

  size_t new_pos = impl_code.find("fn new(");
  size_t next_pos = impl_code.find("fn next(");

  ASSERT_NE(new_pos, std::string::npos);
  ASSERT_NE(next_pos, std::string::npos);

  std::string new_body = impl_code.substr(new_pos, next_pos - new_pos);
  EXPECT_TRUE(absl::StrContains(new_body, "let my_tuple = (u32:1, u32:2);"));
  EXPECT_TRUE(
      absl::StrContains(new_body, "let sum = my_tuple.0 + my_tuple.1;"));

  std::string next_body = impl_code.substr(next_pos);
  // tx and sum must be promoted, but my_tuple must NOT be.
  EXPECT_TRUE(
      absl::StrContains(next_body, "send(token(), self.tx, read(self.sum))"));
  EXPECT_FALSE(absl::StrContains(next_body, "self.my_tuple"));
}

TEST(TestFunctionTransformerTest, TestFunctionWithShadowedChannel) {
  constexpr std::string_view kProgram = R"(#![feature(explicit_state_access)]
proc P {
    r: chan<s32> in,
}
impl P {
    fn new(r: chan<s32> in) -> Self { P { r } }
}

#[test]
fn main() {
    let (tx, rx) = chan<s32>("c");
    let p = P::new(rx);
    p.spawn();
    let rx = s32:-1;
    send(token(), tx, rx);
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::string cloned_code,
                           TransformAndTypecheck(kProgram));

  size_t impl_pos = cloned_code.find("impl __test__proc__main");
  ASSERT_NE(impl_pos, std::string::npos);
  std::string impl_code = cloned_code.substr(impl_pos);

  size_t new_pos = impl_code.find("fn new(");
  size_t next_pos = impl_code.find("fn next(");

  ASSERT_NE(new_pos, std::string::npos);
  ASSERT_NE(next_pos, std::string::npos);

  std::string new_body = impl_code.substr(new_pos, next_pos - new_pos);
  EXPECT_TRUE(absl::StrContains(new_body, "let (tx, rx) = chan<s32>(\"c\");"));

  EXPECT_TRUE(absl::StrContains(new_body, "P::new(rx)"));
  EXPECT_TRUE(absl::StrContains(new_body, "let rx = s32:-1;"));

  std::string next_body = impl_code.substr(next_pos);
  // tx and the shadowed rx (s32) must be promoted. The original rx (channel) is
  // NOT promoted.
  EXPECT_TRUE(
      absl::StrContains(next_body, "send(token(), self.tx, read(self.rx))"));
  EXPECT_TRUE(absl::StrContains(next_body, "self.rx"));
}

TEST(TestFunctionTransformerTest, TestFunctionWithPromotedVariableInIf) {
  constexpr std::string_view kProgram = R"(#![feature(explicit_state_access)]
proc Dummy {
}
impl Dummy {
    fn new() -> Self { Dummy {} }
}

#[test]
fn main() {
    let (tx, rx) = chan<u32>("c");
    let dummy = Dummy::new();
    dummy.spawn();
    let x = u32:42;
    let cond = true;
    if cond {
        send(token(), tx, x)
    } else {
        token()
    };
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::string cloned_code,
                           TransformAndTypecheck(kProgram));

  size_t impl_pos = cloned_code.find("impl __test__proc__main");
  ASSERT_NE(impl_pos, std::string::npos);
  std::string impl_code = cloned_code.substr(impl_pos);

  size_t new_pos = impl_code.find("fn new(");
  size_t next_pos = impl_code.find("fn next(");

  ASSERT_NE(new_pos, std::string::npos);
  ASSERT_NE(next_pos, std::string::npos);

  std::string new_body = impl_code.substr(new_pos, next_pos - new_pos);
  EXPECT_TRUE(absl::StrContains(new_body, "let x = u32:42;"));

  std::string next_body = impl_code.substr(next_pos);
  // tx and x must be promoted.
  EXPECT_TRUE(absl::StrContains(next_body, "if read(self.cond) {"));
  EXPECT_TRUE(
      absl::StrContains(next_body, "send(token(), self.tx, read(self.x))"));
}

TEST(TestFunctionTransformerTest, TestFunctionWithPromotedVariableInMatch) {
  constexpr std::string_view kProgram = R"(#![feature(explicit_state_access)]
proc Dummy {
}
impl Dummy {
    fn new() -> Self { Dummy {} }
}

#[test]
fn main() {
    let (tx, rx) = chan<u32>("c");
    let dummy = Dummy::new();
    dummy.spawn();
    let x = u32:42;
    let val = u32:1;
    match val {
        u32:1 => send(token(), tx, x),
        _ => token(),
    };
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::string cloned_code,
                           TransformAndTypecheck(kProgram));

  size_t impl_pos = cloned_code.find("impl __test__proc__main");
  ASSERT_NE(impl_pos, std::string::npos);
  std::string impl_code = cloned_code.substr(impl_pos);

  size_t new_pos = impl_code.find("fn new(");
  size_t next_pos = impl_code.find("fn next(");

  ASSERT_NE(new_pos, std::string::npos);
  ASSERT_NE(next_pos, std::string::npos);

  std::string new_body = impl_code.substr(new_pos, next_pos - new_pos);
  EXPECT_TRUE(absl::StrContains(new_body, "let x = u32:42;"));

  std::string next_body = impl_code.substr(next_pos);
  EXPECT_TRUE(absl::StrContains(next_body, "match read(self.val) {"));
  // tx must be promoted.
  EXPECT_TRUE(absl::StrContains(
      next_body, "u32:1 => send(token(), self.tx, read(self.x))"));
}

TEST(TestFunctionTransformerTest, TestFunctionWithStatefulProcAndAssertions) {
  constexpr std::string_view kProgram = R"(#![feature(explicit_state_access)]
proc Counter {
    state: u32,
}
impl Counter {
    fn new() -> Self { Counter { state: u32:0 } }
    fn next(self) {
        let s = read(self.state);
        write(self.state, s + u32:1);
    }
}

#[test]
fn main() {
    let (tx, rx) = chan<u32>("c");
    let x = u32:42;
    let p = Counter::new();
    p.spawn();
    send(token(), tx, x);
    assert_eq(x, u32:42);
}
)";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  // We manually perform the transformation and verify the generated string
  // representation here instead of using `TransformAndTypecheck`.
  // `TransformAndTypecheck` fails because it attempts to re-typecheck the
  // string representation of the transformed module, which is currently buggy
  // due to how state members are printed by `ToString()` (i.e., wrapped in
  // `State`.)
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
  EXPECT_TRUE(absl::StrContains(new_body, "let x = u32:42;"));
  EXPECT_TRUE(absl::StrContains(new_body, "Counter::new()"));

  std::string next_body = impl_code.substr(next_pos);
  // tx and x must be promoted. assert_eq must be rewritten to use read(self.x)
  EXPECT_TRUE(
      absl::StrContains(next_body, "send(token(), self.tx, read(self.x))"));
  EXPECT_TRUE(absl::StrContains(next_body, "assert_eq(read(self.x), u32:42)"));
}

TEST(TestFunctionTransformerTest, TestFunctionWithNameCollision) {
  constexpr std::string_view kProgram = R"(#![feature(explicit_state_access)]
proc P {
    c: chan<u32> out,
}
impl P {
    fn new(c: chan<u32> out) -> Self {
        P { c }
    }
}

#[test]
fn main() {
    let (tx, rx) = chan<u32>("c");
    let x_1 = u32:100;
    let tok1 = send(token(), tx, x_1);
    let x = u32:42;
    let tok2 = send(tok1, tx, x);
    let x = u32:43;
    let tok3 = send(tok2, tx, x);
    P::new(tx).spawn();
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::string cloned_code,
                           TransformAndTypecheck(kProgram));

  size_t impl_pos = cloned_code.find("impl __test__proc__main");
  ASSERT_NE(impl_pos, std::string::npos);
  std::string impl_code = cloned_code.substr(impl_pos);

  size_t new_pos = impl_code.find("fn new(");
  size_t next_pos = impl_code.find("fn next(");

  ASSERT_NE(new_pos, std::string::npos);
  ASSERT_NE(next_pos, std::string::npos);

  std::string new_body = impl_code.substr(new_pos, next_pos - new_pos);
  EXPECT_TRUE(absl::StrContains(new_body, "let (tx, rx) = chan<u32>(\"c\");"));
  EXPECT_TRUE(absl::StrContains(new_body, "let x_1 = u32:100;"));
  // This one doesn't need to have an alternate name, because it's already
  // unique.
  EXPECT_TRUE(absl::StrContains(new_body, "let x = u32:42;")) << new_body;
  // This one does.
  EXPECT_TRUE(absl::StrContains(new_body, "let x_2 = u32:43;")) << new_body;

  std::string next_body = impl_code.substr(next_pos);
  EXPECT_TRUE(
      absl::StrContains(next_body, "send(token(), self.tx, read(self.x_1))"));
  EXPECT_TRUE(
      absl::StrContains(next_body, "send(tok1, self.tx, read(self.x))"));
  EXPECT_TRUE(
      absl::StrContains(next_body, "send(tok2, self.tx, read(self.x_2))"));
}

TEST(TestFunctionTransformerTest, TestFunctionWithImportedType) {
  auto import_data = CreateImportDataForTest();
  constexpr std::string_view kImported = R"(
pub struct ImportedStruct {
    a: u32
}
)";
  constexpr std::string_view kProgram = R"(#![feature(explicit_state_access)]
import fake_import;
proc P {
    c: chan<fake_import::ImportedStruct> out,
}
impl P {
    fn new(c: chan<fake_import::ImportedStruct> out) -> Self {
        P { c }
    }
}

#[test]
fn main() {
    let (tx, rx) = chan<fake_import::ImportedStruct>("c");
    let x = fake_import::ImportedStruct { a: u32:42 };
    P::new(tx).spawn();
    send(token(), tx, x);
}
)";
  XLS_ASSERT_OK(
      ParseAndTypecheck(kImported, "fake_import.x", "fake_import", &import_data)
          .status());

  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  TestFunctionTransformer transformer(*tm.module, *tm.type_info);
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> new_module,
                           transformer.TransformTestFunctions());

  std::string cloned_code = new_module->ToString();

  new_module->SetName(absl::StrCat(new_module->name(), "_cloned"));
  XLS_EXPECT_OK(
      TypecheckModule(std::move(new_module), "test_cloned.x", &import_data)
          .status());

  EXPECT_TRUE(absl::StrContains(cloned_code, "x: fake_import::ImportedStruct"));
}

TEST(TestFunctionTransformerTest, TestFunctionWithImportedEnum) {
  auto import_data = CreateImportDataForTest();
  constexpr std::string_view kImported = R"(
pub enum ImportedEnum : u32 {
    A = 1,
    B = 2,
}
)";
  constexpr std::string_view kProgram = R"(#![feature(explicit_state_access)]
import fake_import;
proc P {
    c: chan<fake_import::ImportedEnum> out,
}
impl P {
    fn new(c: chan<fake_import::ImportedEnum> out) -> Self {
        P { c }
    }
}

#[test]
fn main() {
    let (tx, rx) = chan<fake_import::ImportedEnum>("c");
    let x = fake_import::ImportedEnum::A;
    P::new(tx).spawn();
    send(token(), tx, x);
}
)";
  XLS_ASSERT_OK(
      ParseAndTypecheck(kImported, "fake_import.x", "fake_import", &import_data)
          .status());

  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  TestFunctionTransformer transformer(*tm.module, *tm.type_info);
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> new_module,
                           transformer.TransformTestFunctions());

  std::string cloned_code = new_module->ToString();

  new_module->SetName(absl::StrCat(new_module->name(), "_cloned"));
  XLS_EXPECT_OK(
      TypecheckModule(std::move(new_module), "test_cloned.x", &import_data)
          .status());

  EXPECT_TRUE(absl::StrContains(cloned_code, "x: fake_import::ImportedEnum"));
}

TEST(TestFunctionTransformerTest, TestFunctionWithPromotedTupleOfImportedType) {
  auto import_data = CreateImportDataForTest();
  constexpr std::string_view kImported = R"(
pub struct ImportedStruct {
    a: u32
}
)";
  constexpr std::string_view kProgram = R"(#![feature(explicit_state_access)]
import fake_import;
proc P {
    c: chan<fake_import::ImportedStruct> out,
}
impl P {
    fn new(c: chan<fake_import::ImportedStruct> out) -> Self {
        P { c }
    }
}

#[test]
fn main() {
    let (tx, rx) = chan<fake_import::ImportedStruct>("c");
    let t = (fake_import::ImportedStruct { a: u32:42 }, u32:100);
    P::new(tx).spawn();
    let tok = send(token(), tx, t.0);
}
)";
  XLS_ASSERT_OK(
      ParseAndTypecheck(kImported, "fake_import.x", "fake_import", &import_data)
          .status());

  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  TestFunctionTransformer transformer(*tm.module, *tm.type_info);
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> new_module,
                           transformer.TransformTestFunctions());

  std::string cloned_code = new_module->ToString();

  new_module->SetName(absl::StrCat(new_module->name(), "_cloned"));
  XLS_EXPECT_OK(
      TypecheckModule(std::move(new_module), "test_cloned.x", &import_data)
          .status());

  EXPECT_TRUE(
      absl::StrContains(cloned_code, "t: (fake_import::ImportedStruct, u32)"));
}

TEST(TestFunctionTransformerTest, TestFunctionWithPromotedArrayOfImportedType) {
  auto import_data = CreateImportDataForTest();
  constexpr std::string_view kImported = R"(
pub struct ImportedStruct {
    a: u32
}
)";
  constexpr std::string_view kProgram = R"(#![feature(explicit_state_access)]
import fake_import;
proc P {
    c: chan<fake_import::ImportedStruct> out,
}
impl P {
    fn new(c: chan<fake_import::ImportedStruct> out) -> Self {
        P { c }
    }
}

#[test]
fn main() {
    let (tx, rx) = chan<fake_import::ImportedStruct>("c");
    let a = [fake_import::ImportedStruct { a: u32:42 }, fake_import::ImportedStruct { a: u32:100 }];
    P::new(tx).spawn();
    let tok = send(token(), tx, a[0]);
}
)";
  XLS_ASSERT_OK(
      ParseAndTypecheck(kImported, "fake_import.x", "fake_import", &import_data)
          .status());

  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  TestFunctionTransformer transformer(*tm.module, *tm.type_info);
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> new_module,
                           transformer.TransformTestFunctions());

  std::string cloned_code = new_module->ToString();

  new_module->SetName(absl::StrCat(new_module->name(), "_cloned"));
  XLS_EXPECT_OK(
      TypecheckModule(std::move(new_module), "test_cloned.x", &import_data)
          .status());

  EXPECT_TRUE(
      absl::StrContains(cloned_code, "a: fake_import::ImportedStruct[2]"));
}

}  // namespace
}  // namespace xls::dslx
