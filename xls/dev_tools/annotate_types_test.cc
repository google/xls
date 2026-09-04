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

#include "xls/dev_tools/annotate_types.h"

#include <string>
#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"

namespace xls::dslx {
namespace {

using ::absl_testing::IsOkAndHolds;

TEST(AnnotateTypesTest, SimpleScalarLet) {
  constexpr std::string_view kProgram = R"(fn f() -> u32 {
    let x = u32:42;
    x
}
)";
  constexpr std::string_view kExpected = R"(fn f() -> u32 {
    let x: u32 = u32:42;
    x
}
)";
  EXPECT_THAT(AnnotateTypes(kProgram, "test_module"), IsOkAndHolds(kExpected));
}

TEST(AnnotateTypesTest, AlreadyAnnotatedLet) {
  constexpr std::string_view kProgram = R"(fn f() -> u32 {
    let x: u32 = u32:42;
    let y = u32:10;
    x + y
}
)";
  constexpr std::string_view kExpected = R"(fn f() -> u32 {
    let x: u32 = u32:42;
    let y: u32 = u32:10;
    x + y
}
)";
  EXPECT_THAT(AnnotateTypes(kProgram, "test_module"), IsOkAndHolds(kExpected));
}

TEST(AnnotateTypesTest, TupleLet) {
  constexpr std::string_view kProgram = R"(fn f() -> (u32, s16) {
    let (a, b) = (u32:1, s16:2);
    (a, b)
}
)";
  constexpr std::string_view kExpected = R"(fn f() -> (u32, s16) {
    let (a, b): (u32, s16) = (u32:1, s16:2);
    (a, b)
}
)";
  EXPECT_THAT(AnnotateTypes(kProgram, "test_module"), IsOkAndHolds(kExpected));
}

TEST(AnnotateTypesTest, SingleElementTupleLet) {
  constexpr std::string_view kProgram = R"(fn f() -> (u32,) {
    let (a,) = (u32:42,);
    let t = (u32:100,);
    (a + t.0,)
}
)";
  constexpr std::string_view kExpected = R"(fn f() -> (u32,) {
    let (a,): (u32,) = (u32:42,);
    let t: (u32,) = (u32:100,);
    (a + t.0,)
}
)";
  EXPECT_THAT(AnnotateTypes(kProgram, "test_module"), IsOkAndHolds(kExpected));
}

TEST(AnnotateTypesTest, ArrayLet) {
  constexpr std::string_view kProgram = R"(fn f() -> u32[3] {
    let a = u32[3]:[1, 2, 3];
    a
}
)";
  constexpr std::string_view kExpected = R"(fn f() -> u32[3] {
    let a: u32[3] = u32[3]:[1, 2, 3];
    a
}
)";
  EXPECT_THAT(AnnotateTypes(kProgram, "test_module"), IsOkAndHolds(kExpected));
}

TEST(AnnotateTypesTest, StructAndEnumLet) {
  constexpr std::string_view kProgram = R"(enum Color : u2 {
    RED = 0,
    BLUE = 1,
}

struct Point {
    x: u32,
    y: u32,
    color: Color,
}

fn make_point() -> Point {
    let c = Color::BLUE;
    let p = Point { x: u32:1, y: u32:2, color: c };
    p
}
)";
  constexpr std::string_view kExpected = R"(enum Color : u2 {
    RED = 0,
    BLUE = 1,
}

struct Point {
    x: u32,
    y: u32,
    color: Color,
}

fn make_point() -> Point {
    let c: Color = Color::BLUE;
    let p: Point = Point { x: u32:1, y: u32:2, color: c };
    p
}
)";
  EXPECT_THAT(AnnotateTypes(kProgram, "test_module"), IsOkAndHolds(kExpected));
}

TEST(AnnotateTypesTest, PreservesCommentsAndFormatting) {
  constexpr std::string_view kProgram = R"(// Top-level header comment
fn f(x: u32) -> u32 {
    // Comment before let
    let foo = x + u32:1; // Trailing line comment

    // Multi-line
    // comment block
    let bar = foo * u32:2;
    bar
}
)";
  constexpr std::string_view kExpected = R"(// Top-level header comment
fn f(x: u32) -> u32 {
    // Comment before let
    let foo: u32 = x + u32:1; // Trailing line comment

    // Multi-line
    // comment block
    let bar: u32 = foo * u32:2;
    bar
}
)";
  EXPECT_THAT(AnnotateTypes(kProgram, "test_module"), IsOkAndHolds(kExpected));
}

TEST(AnnotateTypesTest, MultiByteUtf8InComments) {
  constexpr std::string_view kProgram =
      "// \xe2\x9c\xa8 Multi-byte UTF-8 comment: \xf0\x9f\x9a\x80\n"
      "fn f() -> u32 {\n"
      "    // Non-ASCII: \xc3\xa9\xc3\xa0\xc3\xbc "
      "\xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e\n"
      "    let x = u32:42;\n"
      "    x\n"
      "}\n";
  constexpr std::string_view kExpected =
      "// \xe2\x9c\xa8 Multi-byte UTF-8 comment: \xf0\x9f\x9a\x80\n"
      "fn f() -> u32 {\n"
      "    // Non-ASCII: \xc3\xa9\xc3\xa0\xc3\xbc "
      "\xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e\n"
      "    let x: u32 = u32:42;\n"
      "    x\n"
      "}\n";
  EXPECT_THAT(AnnotateTypes(kProgram, "test_module"), IsOkAndHolds(kExpected));
}

TEST(AnnotateTypesTest, UninstantiatedParametricFunctionSkipped) {
  constexpr std::string_view kProgram = R"(fn f<N: u32>(x: uN[N]) -> uN[N] {
    let y = x;
    y
}
)";
  // Since f is uninstantiated and parametric, y cannot be concretely annotated,
  // so it is skipped.
  EXPECT_THAT(AnnotateTypes(kProgram, "test_module"), IsOkAndHolds(kProgram));
}

TEST(AnnotateTypesTest, NestedTupleAndWildcard) {
  constexpr std::string_view kProgram = R"(fn f() -> u32 {
    let (a, (_, c)) = (u32:1, (u8:2, u16:3));
    let _ = u32:42;
    a + (c as u32)
}
)";
  constexpr std::string_view kExpected = R"(fn f() -> u32 {
    let (a, (_, c)): (u32, (u8, u16)) = (u32:1, (u8:2, u16:3));
    let _: u32 = u32:42;
    a + (c as u32)
}
)";
  EXPECT_THAT(AnnotateTypes(kProgram, "test_module"), IsOkAndHolds(kExpected));
}

TEST(AnnotateTypesTest, NestedLetExpressions) {
  constexpr std::string_view kProgram = R"(fn f() -> u32 {
    let a = {
        let b = u32:10;
        b + u32:5
    };
    a
}
)";
  constexpr std::string_view kExpected = R"(fn f() -> u32 {
    let a: u32 = {
        let b: u32 = u32:10;
        b + u32:5
    };
    a
}
)";
  EXPECT_THAT(AnnotateTypes(kProgram, "test_module"), IsOkAndHolds(kExpected));
}

TEST(AnnotateTypesTest, MultipleFunctions) {
  constexpr std::string_view kProgram = R"(fn foo(x: u32) -> u32 {
    let y = x + u32:1;
    y
}

fn bar(z: s32) -> s32 {
    let w = z - s32:1;
    w
}
)";
  constexpr std::string_view kExpected = R"(fn foo(x: u32) -> u32 {
    let y: u32 = x + u32:1;
    y
}

fn bar(z: s32) -> s32 {
    let w: s32 = z - s32:1;
    w
}
)";
  EXPECT_THAT(AnnotateTypes(kProgram, "test_module"), IsOkAndHolds(kExpected));
}

TEST(AnnotateTypesTest, ProcBindings) {
  constexpr std::string_view kProgram = R"(proc MyProc {
    c: chan<u32> in;
    config(c: chan<u32> in) {
        let _ = ();
        (c,)
    }
    init {
        let init_val = u32:0;
        init_val
    }
    next(state: u32) {
        let tok = join();
        let (tok, val) = recv(tok, c);
        let next_state = state + val;
        next_state
    }
}
)";
  constexpr std::string_view kExpected = R"(proc MyProc {
    c: chan<u32> in;
    config(c: chan<u32> in) {
        let _: () = ();
        (c,)
    }
    init {
        let init_val: u32 = u32:0;
        init_val
    }
    next(state: u32) {
        let tok: token = join();
        let (tok, val): (token, u32) = recv(tok, c);
        let next_state: u32 = state + val;
        next_state
    }
}
)";
  EXPECT_THAT(AnnotateTypes(kProgram, "test_module"), IsOkAndHolds(kExpected));
}

TEST(AnnotateTypesTest, TypeInferenceV2Option) {
  constexpr std::string_view kProgram = R"(fn f() -> u32 {
    let x = u32:42;
    x
}
)";
  constexpr std::string_view kExpected = R"(fn f() -> u32 {
    let x: u32 = u32:42;
    x
}
)";
  AnnotateTypesOptions options;
  options.type_inference_v2 = true;
  EXPECT_THAT(AnnotateTypes(kProgram, "test_module", "input.x", options),
              IsOkAndHolds(kExpected));
}

TEST(AnnotateTypesTest, SyntaxErrorReturnsErrorStatus) {
  constexpr std::string_view kInvalidProgram = R"(fn f() -> u32 {
    let x = ;
}
)";
  EXPECT_FALSE(AnnotateTypes(kInvalidProgram, "test_module").ok());
}

}  // namespace
}  // namespace xls::dslx
