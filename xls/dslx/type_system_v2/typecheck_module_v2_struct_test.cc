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

#include <string>
#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system/typecheck_test_utils.h"
#include "xls/dslx/type_system_v2/matchers.h"

// Tests for struct definitions, instantiations, member access, and splatting
// (excluding `impl` blocks).
namespace xls::dslx {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::AllOf;
using ::testing::HasSubstr;

TEST(TypecheckV2StructTest, GlobalStructInstancePropagation) {
  EXPECT_THAT(
      R"(
struct S { field: u32 }
const X = S { field: 5 };
const Y = X;
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("5", "uN[32]"),
                              HasNodeWithType("Y", "S { field: uN[32] }"))));
}

TEST(TypecheckV2StructTest, GlobalStructInstanceContainingStructInstance) {
  EXPECT_THAT(R"(
struct S { field: u32 }
struct T { s: S }
const X = T { s: S { field: 5 } };
)",
              TypecheckSucceeds(
                  AllOf(HasNodeWithType("X", "T { s: S { field: uN[32] } }"),
                        HasNodeWithType("5", "uN[32]"))));
}

TEST(TypecheckV2StructTest, SplatParametricStructWithFunctionReturnValue) {
  // Note that triggering the creation of a struct `ParametricContext` at the
  // point of the `Y` declaration is necessary in order to test possible missing
  // handling of splatted members in the logic that creates the context. If we
  // were to have the 2 instance expressions in the same caller context, the
  // same struct context would be reused for Y.
  EXPECT_THAT(
      R"(
struct S<A: u32, B: u32> { field: uN[A], field2: uN[B] }
fn s_factory<A: u32, B: u32>(a: uN[A], b: uN[B]) -> S<A, B> {
  S {field: a, field2: b}
}
const X = s_factory<16, 32>(4, 5);
const Y = S<16, 32> { field: 3, ..X };
)",
      TypecheckSucceeds(
          HasNodeWithType("Y", "S { field: uN[16], field2: uN[32] }")));
}

TEST(TypecheckV2StructTest, UselessStructSplatWarning) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
struct S { field1: u32, field2: s32 }
const X = S { field1: u32:6, field2: s32:7 };
const Y = S { field1: u32:3, field2: s32:4, ..X };
)"));
  ASSERT_THAT(result.tm.warnings.warnings().size(), 1);
  EXPECT_EQ(result.tm.warnings.warnings()[0].message,
            "'Splatted' struct instance has all members of struct defined, "
            "consider removing the `..X`");
}

TEST(TypecheckV2StructTest, AccessOfStructMember) {
  EXPECT_THAT(
      R"(
struct S { x: u32 }
const X = S { x: u32:5 }.x;
)",
      TypecheckSucceeds(HasNodeWithType("X", "uN[32]")));
}

TEST(TypecheckV2StructTest, AccessOfNonexistentStructMemberFails) {
  EXPECT_THAT(
      R"(
struct S { x: u32 }
const X = S { x: u32:5 }.y;
)",
      TypecheckFails(HasSubstr("No member `y` in struct `S`")));
}

TEST(TypecheckV2StructTest, AccessOfMemberOfNonStructFails) {
  EXPECT_THAT(
      R"(
const X = (u32:1).y;
)",
      TypecheckFails(
          HasSubstr("Builtin type 'u32' does not have attribute 'y'")));
}

TEST(TypecheckV2StructTest, ParametricStructWithParametricCallInMemberType) {
  // Based upon https://github.com/google/xls/issues/2722.
  EXPECT_THAT(R"(
fn bar<N: u32>(a: uN[N]) -> uN[N] { a / 2 }

struct Foo<A: u32> {
    x: uN[bar(A)],
}

const C = Foo<32>{ x: u16:2 };
)",
              TypecheckSucceeds(HasNodeWithType("C", "Foo { x: uN[16] }")));
}

TEST(TypecheckV2StructTest, ParametricStructAnnotationWithoutParametrics) {
  EXPECT_THAT(
      R"(
struct Foo<A: u32> {}

const C = zero!<Foo>();
)",
      TypecheckFails(
          HasSubstr("Reference to parametric struct type `Foo` must have all "
                    "parametrics specified in this context")));
}

TEST(TypecheckV2StructTest, SumOfStructMembers) {
  EXPECT_THAT(
      R"(
struct S {
  x: s16,
  y: s16
}
const X = S { x: -1, y: -2 };
const Y = X.x + X.y;
)",
      TypecheckSucceeds(HasNodeWithType("Y", "sN[16]")));
}

TEST(TypecheckV2StructTest, AccessOfStructMemberFromFunctionReturnValue) {
  EXPECT_THAT(
      R"(
struct S { x: u32 }
fn f(a: u32) -> S { S { x: a } }
const X = f(2).x;

)",
      TypecheckSucceeds(HasNodeWithType("X", "uN[32]")));
}

TEST(TypecheckV2StructTest, AccessOfStructMemberUsedForParametricInference) {
  EXPECT_THAT(
      R"(
struct S<N: u32> { x: uN[N] }
fn f<N: u32>(a: uN[N]) -> uN[N] { a }
const X = f(S { x: u24:1 }.x);
)",
      TypecheckSucceeds(HasNodeWithType("X", "uN[24]")));
}

TEST(TypecheckV2StructTest,
     ParametricStructWithInferredParametricFromOtherStruct) {
  EXPECT_THAT(
      R"(
struct S<N: u32> {
  x: uN[N]
}
const X = S{ x: u24:5 };
const Y = S{ x: X.x };
)",
      TypecheckSucceeds(HasNodeWithType("Y", "S { x: uN[24] }")));
}

TEST(TypecheckV2StructTest, ParametricStructWithTooManyParametricsFails) {
  EXPECT_THAT(R"(
struct S<N: u32> {}
const X = S<16, 8>{};
)",
              TypecheckFails(HasSubstr("Too many parametric values supplied")));
}

TEST(TypecheckV2StructTest,
     ParametricStructWithInsufficientExplicitParametricsInfersParametrics) {
  EXPECT_THAT(
      R"(
struct S<M: u32, N: u32> {
  x: uN[M],
  y: uN[N]
}
const X = S<32>{x: u32:4, y: u32:5};
)",
      TypecheckSucceeds(HasNodeWithType("X", "S { x: uN[32], y: uN[32] }")));
}

TEST(TypecheckV2StructTest,
     ParametricStructWithOneExplicitAndOneDefaultedParametric) {
  EXPECT_THAT(
      R"(
struct S<M: u32, N: u32 = {M * 2}> {
  x: uN[M],
  y: uN[N]
}
const X = S<16>{x: u16:4, y: u32:5};
)",
      TypecheckSucceeds(HasNodeWithType("X", "S { x: uN[16], y: uN[32] }")));
}

TEST(TypecheckV2StructTest, ParametricStructAsFunctionArgument) {
  EXPECT_THAT(
      R"(
struct S<N: u32> {
  x: uN[N]
}
fn foo(a: S<24>) -> S<24> { a }
const X = foo(S<24> { x: u24:5 });
)",
      TypecheckSucceeds(HasNodeWithType("X", "S { x: uN[24] }")));
}

TEST(TypecheckV2StructTest,
     ParametricStructAsFunctionArgumentExplicitMismatchFails) {
  EXPECT_THAT(
      R"(
struct S<N: u32> {
  x: uN[N]
}
fn foo(a: S<24>) -> S<24> { a }
const X = foo(S<25> { x: u25:5 });
)",
      TypecheckFails(
          AllOf(HasSubstr("Value mismatch for parametric `N` of struct `S`"),
                HasSubstr("u32:24 vs. u32:25"))));
}

TEST(TypecheckV2StructTest, ParametricStructAsFunctionReturnValue) {
  EXPECT_THAT(
      R"(
struct S<N: u32> {
  x: uN[N]
}
fn foo(x: u24) -> S<24> { S { x } }
const X = foo(u24:5);
)",
      TypecheckSucceeds(
          AllOf(HasNodeWithType("X", "S { x: uN[24] }"),
                HasNodeWithType("S { x: x }", "S { x: uN[24] }"))));
}

TEST(TypecheckV2StructTest,
     ParametricStructFormalReturnValueWithTooManyParametrics) {
  EXPECT_THAT(
      R"(
struct S<N: u32> {}
fn foo() -> S<24, 25> { S {} }
)",
      TypecheckFails(HasSubstr("Too many parametric values supplied")));
}

TEST(TypecheckV2StructTest,
     ParametricStructFormalReturnValueWithWrongTypeParametric) {
  EXPECT_THAT(
      R"(
struct S<N: u32> {}
fn foo() -> S<u64:24> { S {} }
)",
      TypecheckFails(HasSizeMismatch("u64", "u32")));
}

TEST(TypecheckV2StructTest,
     ParametricStructFormalArgumentWithTooManyParametrics) {
  EXPECT_THAT(
      R"(
struct S<N: u32> {}
fn foo(a: S<24, 25>) {}
)",
      TypecheckFails(HasSubstr("Too many parametric values supplied")));
}

TEST(TypecheckV2StructTest,
     ParametricStructFormalArgumentWithWrongTypeParametric) {
  EXPECT_THAT(
      R"(
struct S<N: u32> {}
fn foo(a: S<u64:24>) {}
)",
      TypecheckFails(HasSizeMismatch("u64", "u32")));
}

TEST(TypecheckV2StructTest,
     ParametricStructAsFunctionReturnValueWithExplicitismatchFails) {
  EXPECT_THAT(
      R"(
struct S<N: u32> {
  x: uN[N]
}
fn foo(x: u24) -> S<24> { S<25> { x } }
const X = foo(u24:5);
)",
      TypecheckFails(
          AllOf(HasSubstr("Value mismatch for parametric `N` of struct `S`"),
                HasSubstr("u32:25 vs. u32:24"))));
}

TEST(TypecheckV2StructTest, ParametricStructWithWrongOrderParametricValues) {
  EXPECT_THAT(
      R"(
struct StructFoo<A: u32, B: u32> {
  x: uN[A],
  y: uN[B],
}

fn wrong_order<A: u32, B: u32>(x:uN[A], y:uN[B]) -> StructFoo<B, A> {
  StructFoo<B, A>{x, y}
}

fn test() -> StructFoo<32, 33> {
  wrong_order<32, 33>(2, 3)
}
)",
      TypecheckFails(HasSubstr("Value mismatch for parametric `A` of struct "
                               "`StructFoo`: u32:33 vs. u32:32")));
}

TEST(TypecheckV2StructTest,
     ParametricStructWithCorrectReverseOrderParametricValues) {
  EXPECT_THAT(
      R"(
struct StructFoo<A: u32, B: u32> {
  x: uN[A],
  y: uN[B],
}

fn wrong_order<A: u32, B: u32>(x:uN[A], y:uN[B]) -> StructFoo<B, A> {
  StructFoo<B, A>{x:y, y:x}
}

fn test() -> StructFoo<33, 32> {
  wrong_order<32, 33>(2, 3)
}
)",
      TypecheckSucceeds(HasNodeWithType("wrong_order<32, 33>(2, 3)",
                                        "StructFoo { x: uN[33], y: uN[32] }")));
}

TEST(TypecheckV2StructTest, ParametricStructWithConstantDimension) {
  EXPECT_THAT(
      R"(
const N = u32:4;
struct S<M: u32> {
  x: uN[M][N]
}
const X = S { x: [u24:1, u24:2, u24:3, u24:4] };
)",
      TypecheckSucceeds(HasNodeWithType("X", "S { x: uN[24][4] }")));
}

// Various samples of actual-argument compatibility with an `xN` field within a
// struct via a struct instantiation expression (based on original
// `typecheck_module_test`).
TEST(TypecheckV2StructTest, StructInstantiateParametricXnField) {
  EXPECT_THAT(
      R"(
struct XnWrapper<S: bool, N: u32> {
  field: xN[S][N]
}
fn f() -> XnWrapper<false, u32:8> { XnWrapper<false, u32:8> { field: u8:0 } }
fn g() -> XnWrapper<true, u32:8> { XnWrapper<true, u32:8> { field: s8:1 } }
fn h() -> XnWrapper<false, u32:8> { XnWrapper<false, u32:8> { field: xN[false][8]:2 } }
fn i() -> XnWrapper<true, u32:8> { XnWrapper<true, u32:8> { field: xN[true][8]:3 } }
)",
      TypecheckSucceeds(AllOf(
          HasNodeWithType("XnWrapper<false, u32:8> { field: u8:0 }",
                          "XnWrapper { field: uN[8] }"),
          HasNodeWithType("XnWrapper<true, u32:8> { field: s8:1 }",
                          "XnWrapper { field: sN[8] }"),
          HasNodeWithType("XnWrapper<false, u32:8> { field: xN[false][8]:2 }",
                          "XnWrapper { field: uN[8] }"),
          HasNodeWithType("XnWrapper<true, u32:8> { field: xN[true][8]:3 }",
                          "XnWrapper { field: sN[8] }"))));
}

TEST(TypecheckV2StructTest, StructFunctionArgument) {
  EXPECT_THAT(R"(
struct S { field: u32 }
fn f(s: S) {}
fn g() {
  f(S { field: 2 })
}
)",
              TypecheckSucceeds(AllOf(
                  HasNodeWithType("2", "uN[32]"),
                  HasNodeWithType("S { field: 2 }", "S { field: uN[32] }"))));
}

TEST(TypecheckV2StructTest, StructFunctionReturnValue) {
  EXPECT_THAT(R"(
struct S { field: u32 }
fn f(value: u32) -> S {
  S { field: value }
}
const X = f(2);
)",
              TypecheckSucceeds(AllOf(
                  HasNodeWithType("const X = f(2);", "S { field: uN[32] }"))));
}

TEST(TypecheckV2StructTest, InstantiationOfNonStruct) {
  EXPECT_THAT(
      "const X = u32 { foo: 1 };",
      TypecheckFails(HasSubstr(
          "Attempted to instantiate non-struct type `u32` as a struct.")));
}

TEST(TypecheckV2StructTest, ZeroMacroEmptyStruct) {
  EXPECT_THAT(R"(
struct S { }
const Y = zero!<S>();
)",
              TypecheckSucceeds(HasNodeWithType("Y", "S {}")));
}

TEST(TypecheckV2StructTest, ZeroMacroStruct) {
  EXPECT_THAT(
      R"(
struct S { a: u32, b: u32, }
const Y = zero!<S>();
)",
      TypecheckSucceeds(HasNodeWithType("Y", "S { a: uN[32], b: uN[32] }")));
}

TEST(TypecheckV2StructTest, ZeroMacroParametricStruct) {
  EXPECT_THAT(
      R"(
struct S<A: u32, B: u32> { a: uN[A], b: uN[B], }
const Y = zero!<S<16, 64>>();
)",
      TypecheckSucceeds(HasNodeWithType("Y", "S { a: uN[16], b: uN[64] }")));
}

TEST(TypecheckV2StructTest, ZeroMacroParametricStructInFn) {
  EXPECT_THAT(
      R"(
struct S<A: u32, B: u32> { a: uN[A], b: uN[B], }
fn f<N:u32, M: u32={N*4}>()-> S<N, M> { zero!<S<N, M>>() }
const Y = f<16>();
)",
      TypecheckSucceeds(HasNodeWithType("Y", "S { a: uN[16], b: uN[64] }")));
}

TEST(TypecheckV2StructTest, ZeroMacroImportedStructType) {
  constexpr std::string_view kImported = R"(
pub struct S { field: u32 }
)";
  constexpr std::string_view kProgram = R"(
import imported;
const Y = zero!<imported::S>();
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data));
  EXPECT_THAT(
      TypecheckV2(kProgram, "main", &import_data),
      IsOkAndHolds(HasTypeInfo(HasNodeWithType("Y", "S { field: uN[32] }"))));
}

TEST(TypecheckV2StructTest,
     ZeroMacroImportedStructTypeWithParametricsInOwnningModule) {
  constexpr std::string_view kImported = R"(
pub struct S<N: u32> { field: uN[N] }
pub type S32 = S<32>;
)";
  constexpr std::string_view kProgram = R"(
import imported;
const Y = zero!<imported::S32>();
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data));
  EXPECT_THAT(
      TypecheckV2(kProgram, "main", &import_data),
      IsOkAndHolds(HasTypeInfo(HasNodeWithType("Y", "S { field: uN[32] }"))));
}

TEST(TypecheckV2StructTest,
     ZeroMacroImportedStructTypeWithInlineParametricsInConsumingModule) {
  constexpr std::string_view kImported = R"(
pub struct S<N: u32> { field: uN[N] }
)";
  constexpr std::string_view kProgram = R"(
import imported;
const Y = zero!<imported::S<32>>();
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data));
  EXPECT_THAT(
      TypecheckV2(kProgram, "main", &import_data),
      IsOkAndHolds(HasTypeInfo(HasNodeWithType("Y", "S { field: uN[32] }"))));
}

TEST(TypecheckV2StructTest,
     ZeroMacroImportedStructTypeWithAliasParametricsInConsumingModule) {
  constexpr std::string_view kImported = R"(
pub struct S<N: u32> { field: uN[N] }
)";
  constexpr std::string_view kProgram = R"(
import imported;
type S32 = imported::S<32>;
const Y = zero!<S32>();
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data));
  EXPECT_THAT(
      TypecheckV2(kProgram, "main", &import_data),
      IsOkAndHolds(HasTypeInfo(HasNodeWithType("Y", "S { field: uN[32] }"))));
}

TEST(TypecheckV2StructTest,
     ZeroMacroImportedStructTypeWithParametricsInBothModulesFails) {
  constexpr std::string_view kImported = R"(
pub struct S<N: u32> { field: uN[N] }
pub type S32 = S<32>;
)";
  constexpr std::string_view kProgram = R"(
import imported;
const Y = zero!<imported::S32<32>>();
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data));
  EXPECT_THAT(TypecheckV2(kProgram, "main", &import_data).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Parametric values defined multiple times for "
                                 "annotation: `S<32>`")));
}

TEST(TypecheckV2StructTest, AllOnesMacroEmptyStruct) {
  EXPECT_THAT(R"(
struct S { }
const Y = all_ones!<S>();
)",
              TypecheckSucceeds(HasNodeWithType("Y", "S {}")));
}

TEST(TypecheckV2StructTest, AllOnesMacroStruct) {
  EXPECT_THAT(
      R"(
struct S { a: u32, b: u32, }
const Y = all_ones!<S>();
)",
      TypecheckSucceeds(HasNodeWithType("Y", "S { a: uN[32], b: uN[32] }")));
}

TEST(TypecheckV2StructTest, AllOnesMacroParametricStruct) {
  EXPECT_THAT(
      R"(
struct S<A: u32, B: u32> { a: uN[A], b: uN[B], }
const Y = all_ones!<S<16, 64>>();
)",
      TypecheckSucceeds(HasNodeWithType("Y", "S { a: uN[16], b: uN[64] }")));
}

TEST(TypecheckV2StructTest, ParametricsFromStructAndMethodInType) {
  EXPECT_THAT(
      R"(
struct S<M: u32> {
  a: uN[M]
}

impl S<M> {
  fn replicate<N: u32>(self) -> uN[M][N] { [self.a, ...] }
}

const X = S{a: u16:5}.replicate<3>();
const Y = S{a: u32:6}.replicate<4>();
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[16][3]"),
                              HasNodeWithType("Y", "uN[32][4]"))));
}

TEST(TypecheckV2StructTest, ParametricStructsWithSameBindingName) {
  EXPECT_THAT(
      R"(
struct S<M: u32> {
  a: uN[M]
}

struct T<M: u32> {
  a: uN[M]
}

impl S<M> {
  fn replicate<N: u32>(self) -> uN[M][N] { [self.a, ...] }
}

impl T<M> {
  fn replicate<N: u32>(self) -> uN[M][N] { [self.a, ...] }
}

const X = S{a: u16:5}.replicate<3>();
const Y = T{a: u32:6}.replicate<4>();
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[16][3]"),
                              HasNodeWithType("Y", "uN[32][4]"))));
}

TEST(TypecheckV2StructTest, ParametricsFromStructAndMethodBothInferred) {
  EXPECT_THAT(
      R"(
struct Data<M: u32> {
  a: uN[M]
}

impl Data<M> {
  fn combine<S: bool, N: u32>(self, b: xN[S][N]) -> (uN[M], xN[S][N]) {
    (self.a, b)
  }
}

const X = Data{a: u3:5}.combine(s7:-42);
const Y = Data{a: u7:120}.combine(u9:256);
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("X", "(uN[3], sN[7])"),
                              HasNodeWithType("Y", "(uN[7], uN[9])"))));
}

TEST(TypecheckV2StructTest, TypeAliasOnStructInParametricFn) {
  EXPECT_THAT(R"(
struct S<X: u32> {
  x: bits[X],
}

fn f<T: u32>() -> uN[T] {
  type MyS = S<T>;
  MyS { x: 1 }.x
}

fn main() {
  let x = f<8>();
  let y = f<15>();
}
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("x", "uN[8]"),
                                      HasNodeWithType("y", "uN[15]"))));
}

TEST(TypecheckV2StructTest, TypeAliasOfStructWithBoundParametrics) {
  EXPECT_THAT(R"(
struct S<X: u32, Y: u32> {
  x: bits[X],
  y: bits[Y],
}
type MyS = S<3, 4>;
fn f() -> MyS { MyS {x: 3, y: 4 } }
)",
              TypecheckSucceeds(AllOf(
                  HasNodeWithType("f", "() -> S { x: uN[3], y: uN[4] }"),
                  HasNodeWithType("MyS", "typeof(S { x: uN[3], y: uN[4] })"))));
}

TEST(TypecheckV2StructTest, ElementInTypeAliasOfStructWithBoundParametrics) {
  EXPECT_THAT(R"(
struct S<X: u32, Y: u32> {
  x: bits[X],
  y: bits[Y],
}
type MyS = S<3, 4>;
fn f() -> uN[3] {
  let x = MyS { x: 1, y: 1 };
  x.x
}
)",
              TypecheckSucceeds(AllOf(
                  HasNodeWithType("f", "() -> uN[3]"),
                  HasNodeWithType("MyS", "typeof(S { x: uN[3], y: uN[4] })"),
                  HasNodeWithType("x", "S { x: uN[3], y: uN[4] }"))));
}

TEST(TypecheckV2StructTest, ForWithDestructuredAcc) {
  EXPECT_THAT(
      R"(
const X = for (i, (x, y, (a, b))) : (u32, (u32, u32, (u32, u32))) in u32:0..3 {
  (x + i, y + i, (x + 2, y + 2))
  } ((0, 0, (1, 1)));
)",
      TypecheckSucceeds(
          AllOf(HasNodeWithType("X", "(uN[32], uN[32], (uN[32], uN[32]))"),
                HasNodeWithType("x", "uN[32]"), HasNodeWithType("y", "uN[32]"),
                HasNodeWithType("a", "uN[32]"), HasNodeWithType("b", "uN[32]"),
                HasNodeWithType("(x, y, (a, b))",
                                "(uN[32], uN[32], (uN[32], uN[32]))"),
                HasNodeWithType("(a, b)", "(uN[32], uN[32])"))));
}

TEST(TypecheckV2StructTest, SliceDestructuredAccumulator) {
  EXPECT_THAT(R"(
const X = for (i, (q, r)): (u32, (u8, u16)) in u32:0..1 {
    (q, r[0:s32:16])
}((0, 0));
)",
              TypecheckSucceeds(HasNodeWithType("X", "(uN[8], uN[16])")));
}

TEST(TypecheckV2StructTest, ProcAsStructMemberFails) {
  EXPECT_THAT(
      R"(
proc Foo {
  foo: u32,
  bar: bool,
}

impl Foo {}

struct Bar {
  the_proc: Foo
}
)",
      TypecheckFails(HasSubstr("Structs cannot contain procs as members.")));
}

TEST(TypecheckV2StructTest, ParametricStructInferenceUsingProcParametric) {
  constexpr std::string_view kImported = R"(
pub struct Data<N: u32> {
  value: uN[N]
}
)";
  constexpr std::string_view kProgram = R"(
import imported;

proc Counter<N: u32> {
  type value_t = imported::Data<N>;

  c: chan<value_t> out;
  init { value_t { value: 0 } }
  config(c: chan<value_t> out) {
    (c,)
  }
  next(i: value_t) {
    send(join(), c, i);
    let res = imported::Data { value: uN[N]:0 };
    res
  }
}

proc main {
  c16: chan<imported::Data<16>> in;
  c32: chan<imported::Data<32>> in;
  init { (join(), 0) }
  config() {
    let (p16, c16) = chan<imported::Data<16>>("my_chan16");
    let (p32, c32) = chan<imported::Data<32>>("my_chan32");
    spawn Counter<16>(p16);
    spawn Counter<32>(p32);
    (c16,c32)
  }
  next(state: (token, u48)) {
    let (tok16, v16) = recv(state.0, c16);
    let (tok32, v32) = recv(tok16, c32);
    (tok32, v32.value ++ v16.value)
  }
}
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data).status());
  EXPECT_THAT(TypecheckV2(kProgram, "main", &import_data),
              IsOkAndHolds(HasTypeInfo(
                  AllOf(HasNodeWithType("spawn Counter<16>(p16)", "()"),
                        HasNodeWithType("spawn Counter<32>(p32)", "()")))));
}

TEST(TypecheckV2StructTest, ImportConstantStructSizeMismatch) {
  constexpr std::string_view kImported = R"(
struct Point { x: uN[8] }

pub const MY_POINT = Point { x: 1 };
)";
  constexpr std::string_view kProgram = R"(
import imported;

fn main() -> uN[5] {
  let var = imported::MY_POINT;
  var.x
})";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data).status());
  EXPECT_THAT(TypecheckV2(kProgram, "main", &import_data),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSizeMismatch("uN[8]", "uN[5]")));
}

TEST(TypecheckV2StructTest, ImportStruct) {
  constexpr std::string_view kImported = R"(
pub struct S { x: u5[2] }
)";
  constexpr std::string_view kProgram = R"(
import imported;

fn main() -> u5 {
  let s = imported::S{x: [1, 2]};
  s.x[1]
})";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data).status());
  EXPECT_THAT(
      TypecheckV2(kProgram, "main", &import_data),
      IsOkAndHolds(HasTypeInfo(AllOf(HasNodeWithType("s", "S { x: uN[5][2] }"),
                                     HasNodeWithType("s.x", "uN[5]")))));
}

TEST(TypecheckV2StructTest, ImportReturnsStructWithForeignConstant) {
  constexpr std::string_view kFoo = R"(
pub struct S<N: u32> { x: uN[N] }
)";
  constexpr std::string_view kBar = R"(
import foo;

pub const BAR_N = u32:5;
type BAR_S = foo::S<BAR_N>;

pub fn bar() -> BAR_S {
  BAR_S { x: 2 }
}
)";
  constexpr std::string_view kBaz = R"(
import bar;

fn main() {
  const_assert!(bar::bar().x == u5:2);
}
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kFoo, "foo", &import_data).status());
  XLS_EXPECT_OK(TypecheckV2(kBar, "bar", &import_data).status());
  XLS_EXPECT_OK(TypecheckV2(kBaz, "baz", &import_data).status());
}

TEST(TypecheckV2StructTest, ImportStructAsTypeTwoLevels) {
  constexpr std::string_view kFirst = R"(
pub struct S { x: u5 }
)";
  constexpr std::string_view kSecond = R"(
import first;

pub type T = first::S;
)";
  constexpr std::string_view kProgram = R"(
import second;

fn main() -> u5 {
  let s = second::T{x: 1};
  s.x
})";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kFirst, "first", &import_data));
  XLS_EXPECT_OK(TypecheckV2(kSecond, "second", &import_data));
  EXPECT_THAT(
      TypecheckV2(kProgram, "main", &import_data),
      IsOkAndHolds(HasTypeInfo(AllOf(HasNodeWithType("s.x", "uN[5]"),
                                     HasNodeWithType("s", "S { x: uN[5] }")))));
}

TEST(TypecheckV2StructTest, ImportStructAsType) {
  constexpr std::string_view kImported = R"(
pub struct S { x: u5 }

impl S {
  fn X(self) -> u5 { self.x }
}
)";
  constexpr std::string_view kProgram = R"(
import imported;

type T = imported::S;

fn main() -> u5 {
  let s = T{x: 1};
  s.X()
})";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data).status());
  EXPECT_THAT(
      TypecheckV2(kProgram, "main", &import_data),
      IsOkAndHolds(HasTypeInfo(AllOf(HasNodeWithType("s.X()", "uN[5]"),
                                     HasNodeWithType("s", "S { x: uN[5] }")))));
}

TEST(TypecheckV2StructTest, ImportNonPublicStructUseStaticFunction) {
  constexpr std::string_view kImported = R"(
struct S {}

impl S {
  fn X() -> u5[2] { [1, 2] }
}
)";
  constexpr std::string_view kProgram = R"(
import imported;

fn main() -> u5 {
  imported::S::X()[0]
})";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data).status());
  EXPECT_THAT(
      TypecheckV2(kProgram, "main", &import_data),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("not public")));
}

TEST(TypecheckV2StructTest, PassReassignedStructFieldFromParametricFunction) {
  EXPECT_THAT(
      R"(
struct S { a: u32 }

fn s_passthrough(s: S) -> S { s }
fn u32_passthrough(a: u32) -> u32 { a }

fn f<N: u32>(s: S) -> u32 {
 let s = s_passthrough(s);
 u32_passthrough(s.a)
}

const X = f<1>(S { a: 1 });

)",
      TypecheckSucceeds(HasNodeWithType("X", "uN[32]")));
}

TEST(TypecheckV2StructTest, ParametricDefaultClog2InStruct) {
  constexpr std::string_view kImported = R"(
pub fn clog2<N: u32>(x: bits[N]) -> bits[N] {
    if x >= bits[N]:1 { (N as bits[N]) - clz(x - bits[N]:1) } else { bits[N]:0 }
}
)";
  constexpr std::string_view kProgram = R"(
import imported;

struct Foo<X: u32, Y: u32 = {imported::clog2(X)}> {
    a: uN[X],
    b: uN[Y],
}

fn make_zero_foo<X: u32>() -> Foo<X> {
    zero!<Foo<X>>()
}

fn main() -> Foo<30> {
    make_zero_foo<30>()
}

fn test() -> Foo<5> {
    make_zero_foo<5>()
})";

  auto import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data));
  EXPECT_THAT(
      TypecheckV2(kProgram, "main", &import_data),
      IsOkAndHolds(HasTypeInfo(AllOf(
          HasNodeWithType("make_zero_foo<5>()", "Foo { a: uN[5], b: uN[3] }"),
          HasNodeWithType("make_zero_foo<30>()",
                          "Foo { a: uN[30], b: uN[5] }")))));
}

TEST(TypecheckV2StructTest, CallFunctionOnStructMember) {
  EXPECT_THAT(
      R"(
struct G { }

impl G {
  fn x(self: Self) -> u32 {
     u32:1
  }
}

struct F { g: G }

impl F {
  fn y(self: Self) -> u32 {
    self.g.x()
  }
}

)",
      TypecheckSucceeds(HasNodeWithType("y", "(F { g: G {} }) -> uN[32]")));
}

TEST(TypecheckV2StructTest,
     InvocationInfersParametricForStructMemberInvocation) {
  EXPECT_THAT(R"(
pub struct MyStruct<SIZE: u32> { x: bits[SIZE] }

pub fn one<SIZE: u32>() -> MyStruct<SIZE> {
    MyStruct<SIZE> { x: bits[SIZE]:1 }
}

struct WrapStruct<SIZE: u32> { val: MyStruct<SIZE> }

fn wrap<SIZE: u32>(a: MyStruct<SIZE>) -> WrapStruct<SIZE> {
        WrapStruct { val: a }
}

fn test() {
  const F32_SIZE = u32:8;
  let o = one<F32_SIZE>();
  assert_eq(
      wrap(o),
      WrapStruct { val: one<F32_SIZE>() });
}
)",
              TypecheckSucceeds(HasNodeWithType("o", "MyStruct { x: uN[8] }")));
}

TEST(TypecheckV2StructTest, InvocationAsStructParameter) {
  EXPECT_THAT(R"(
struct MyStruct<A: u32> {
    x: uN[A],
    y: u32,
}

type S = MyStruct<1>;

fn zero() -> S {
    S { x: 0, y: 0 }
}

fn test() -> S {
    S { x: 1, ..zero() }
}
)",
              TypecheckSucceeds(HasNodeWithType("test", "() -> MyStruct")));
}

TEST(TypecheckV2StructTest, ImportedTypeDefinedUsingStructMember) {
  constexpr std::string_view kImported = R"(
pub struct Params {
    dim: u32,
}

pub const PARAMS = Params {
    dim: 4,
};

pub struct S {
    arr: u32[PARAMS.dim],
}
)";
  constexpr std::string_view kProgram = R"(
import imported;

type x = imported::S;
)";
  auto import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data));
  EXPECT_THAT(TypecheckV2(kProgram, "main", &import_data),
              IsOkAndHolds(HasTypeInfo(
                  HasNodeWithType("x", "typeof(S { arr: uN[32][4] })"))));
}

TEST(TypecheckV2StructTest, ImportConstantStruct) {
  constexpr std::string_view kImported = R"(
pub struct X { x: u32 }

pub const X_VAL = X { x: 30 };

pub type Word = bits[X_VAL.x];

)";
  constexpr std::string_view kProgram = R"(
import imported;

fn test() -> imported::Word {
    widening_cast<imported::Word>(u1:0)
}
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data));
  EXPECT_THAT(
      TypecheckV2(kProgram, "main", &import_data),
      IsOkAndHolds(HasTypeInfo(HasNodeWithType("test", "() -> uN[30]"))));
}

TEST(TypecheckV2StructTest,
     SplatImplicitParametricStructWithFunctionReturnValue) {
  EXPECT_THAT(
      R"(
struct S<A: u32, B: u32> { field: uN[A], field2: uN[B] }
fn s_factory<A: u32, B: u32>(a: uN[A], b: uN[B]) -> S<A, B> {
  S {field: a, field2: b}
}
const X = s_factory<16, 32>(4, 5);
const Y = S { field: 3, ..X };
)",
      TypecheckSucceeds(
          HasNodeWithType("Y", "S { field: uN[16], field2: uN[32] }")));
}

TEST(TypecheckV2StructTest,
     ParametricStructAsFunctionArgumentWithImplicitParametric) {
  EXPECT_THAT(
      R"(
struct S<N: u32> {
  x: uN[N]
}
fn foo(a: S<24>) -> S<24> { a }
const X = foo(S { x: u24:5 });
)",
      TypecheckSucceeds(HasNodeWithType("X", "S { x: uN[24] }")));
}

TEST(TypecheckV2StructTest,
     ParametricStructAsFunctionArgumentWithImplicitParametricMismatchFails) {
  EXPECT_THAT(
      R"(
struct S<N: u32> {
  x: uN[N]
}
fn foo(a: S<24>) -> S<24> { a }
const X = foo(S { x: u25:5 });
)",
      TypecheckFails(HasSubstr("Value mismatch for parametric `N` of struct "
                               "`S`: u32:24 vs. u32:25")));
}

TEST(TypecheckV2StructTest,
     ParametricStructAsFunctionReturnValueWithImplicitMismatchFails) {
  EXPECT_THAT(
      R"(
struct S<N: u32> {
  x: uN[N]
}
fn foo() -> S<24> { S { x: u25:5 } }
const X = foo();
)",
      TypecheckFails(HasSizeMismatch("u25", "uN[24]")));
}

TEST(TypecheckV2StructTest,
     ParametricStructAsParametricFunctionArgumentWithImplicitParametric) {
  EXPECT_THAT(
      R"(
struct S<N: u32> {
  x: uN[N]
}
fn foo<N: u32>(a: S<N>) -> S<N> { a }
const X = foo(S { x: u24:5 });
)",
      TypecheckSucceeds(HasNodeWithType("X", "S { x: uN[24] }")));
}

TEST(TypecheckV2StructTest,
     ParametricFunctionWithImplicitParametricStructReturnExpr) {
  EXPECT_THAT(
      R"(
struct S<N: u32> {
  x: uN[N]
}
fn foo<N: u32>(x: uN[N]) -> S<N> { S { x } }
const X = foo(u24:5);
)",
      TypecheckSucceeds(HasNodeWithType("X", "S { x: uN[24] }")));
}

TEST(TypecheckV2StructTest, ZeroMacroImplConstError) {
  EXPECT_THAT(R"(
struct S{}
impl S { const X = u32:10; }
const Y = zero!<S::X>();
)",
              TypecheckFails(HasSubstr("in `zero!<S::X>()`")));
}

TEST(TypecheckV2StructTest, AllOnesMacroImplConstError) {
  EXPECT_THAT(R"(
struct S{}
impl S { const X = u32:10; }
const Y = all_ones!<S::X>();
)",
              TypecheckFails(HasSubstr("in `all_ones!<S::X>()`")));
}

TEST(TypecheckV2StructTest, StaticConstantOnImpl) {
  EXPECT_THAT(R"(
struct Point {}
impl Point {
  const NUM_DIMS = u32:2;
}
const X = Point::NUM_DIMS;
)",
              TypecheckSucceeds(HasNodeWithType("X", "uN[32]")));
}

TEST(TypecheckV2StructTest, MissingFunctionOnImplFails) {
  EXPECT_THAT(
      R"(
struct Point {}
impl Point {
  const NUM_DIMS = u32:2;
}
const X = Point::num_dims();
)",
      TypecheckFails(HasSubstr(
          "Name 'num_dims' is not defined by the impl for struct 'Point'")));
}

TEST(TypecheckV2StructTest, ImplWithMissingConstantFails) {
  EXPECT_THAT(
      R"(
struct Point { x: u32, y: u32 }

impl Point {
    const NUM_DIMS = u32:2;
}

fn point_dims() -> u32 {
    Point::DIMENSIONS
}
)",
      TypecheckFails(HasSubstr(
          "Name 'DIMENSIONS' is not defined by the impl for struct 'Point'")));
}

TEST(TypecheckV2StructTest, MissingImplOnStructFails) {
  EXPECT_THAT(R"(
struct Point {}
const X = Point::num_dims();
)",
              TypecheckFails(
                  HasSubstr("Struct 'Point' has no impl defining 'num_dims'")));
}

TEST(TypecheckV2StructTest, ImplWithConstCalledAsFuncFails) {
  EXPECT_THAT(R"(
struct Point {}
impl Point {
    const num_dims = u32:4;
}
const X = Point::num_dims();
)",
              TypecheckFails(HasSubstr(
                  "Invocation callee `Point::num_dims` is not a function")));
}

TEST(TypecheckV2StructTest, StaticImplFunction) {
  EXPECT_THAT(R"(
struct Point {}
impl Point {
    fn num_dims() -> u32 { 2 }
}
const X = Point::num_dims();
)",
              TypecheckSucceeds(HasNodeWithType("X", "uN[32]")));
}

TEST(TypecheckV2StructTest, StaticImplFunctionWithWrongArgumentTypeFails) {
  EXPECT_THAT(R"(
struct Point {}
impl Point {
    fn foo(a: u32) -> u32 { a }
}
const X = Point::foo(u24:5);
)",
              TypecheckFails(HasSizeMismatch("u24", "u32")));
}

TEST(TypecheckV2StructTest, StaticImplFunctionCallWithMissingArgumentFails) {
  EXPECT_THAT(R"(
struct Point {}
impl Point {
    fn foo(a: u32) -> u32 { a }
}
const X = Point::foo();
)",
              TypecheckFails(HasSubstr("Expected 1 argument(s) but got 0")));
}

TEST(TypecheckV2StructTest, StaticImplFunctionCallWithExtraArgumentFails) {
  EXPECT_THAT(R"(
struct Point {}
impl Point {
    fn foo(a: u32) -> u32 { a }
}
const X = Point::foo(u32:1, u32:2);
)",
              TypecheckFails(HasSubstr("Expected 1 argument(s) but got 2")));
}

TEST(TypecheckV2StructTest, StaticImplFunctioUsingConst) {
  EXPECT_THAT(R"(
struct Point {}
impl Point {
    const DIMS = u32:2;

    fn num_dims() -> u32 {
        DIMS
    }
}

const X = uN[Point::num_dims()]:0;
)",
              TypecheckSucceeds(HasNodeWithType("X", "uN[2]")));
}

TEST(TypecheckV2StructTest, StaticConstUsingImplFunction) {
  EXPECT_THAT(R"(
struct Point {}
impl Point {
    fn num_dims() -> u32 { 2 }
    const DIMS = num_dims();
}

const X = uN[Point::DIMS]:0;
)",
              TypecheckSucceeds(HasNodeWithType("X", "uN[2]")));
}

TEST(TypecheckV2StructTest, ImplConstantUsedForParametricFunctionInference) {
  EXPECT_THAT(R"(
struct Foo {}
impl Foo {
  const X = u32:2;
}
fn f<N: u32>(a: uN[N]) -> uN[N] { a }
const Y = f(Foo::X);
)",
              TypecheckSucceeds(HasNodeWithType("Y", "uN[32]")));
}

TEST(TypecheckV2StructTest, ImplMethodCallsStaticImplFunction) {
  EXPECT_THAT(
      R"(
struct F<N: u32> { x: uN[N] }

type MyF = F<5>;

impl F<N> {
    fn static_fn() -> uN[N] {
        uN[N]:1
    }

    fn diff_x(self: Self) -> F<N> {
        F<N> { x: self.x - F<N>::static_fn() }
    }
}

const F_ST = MyF { x: 5 };
const G_ST = F_ST.diff_x();
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("F_ST", "F { x: uN[5] }"),
                              HasNodeWithType("G_ST", "F { x: uN[5] }"))));
}

TEST(TypecheckV2StructTest, ImplFunctionUsingStructMembers) {
  EXPECT_THAT(R"(
struct Point { x: u32, y: u32 }

impl Point {
    fn area(self) -> u32 {
        self.x * self.y
    }
}

const P = Point{x: u32:4, y:u32:2};
const Y = P.area();
const Z = uN[Y]:0;
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("Y", "uN[32]"),
                                      HasNodeWithType("Z", "uN[8]"))));
}

TEST(TypecheckV2StructTest, ImplFunctionReturnsSelf) {
  EXPECT_THAT(R"(
struct Point { x: u32, y: u32 }

impl Point {
    fn unit() -> Self { Point { x: u32:1, y: u32:1 } }
}

const P = Point::unit();
const X = uN[P.x]:0;
)",
              TypecheckSucceeds(
                  AllOf(HasNodeWithType("P", "Point { x: uN[32], y: uN[32] }"),
                        HasNodeWithType("X", "uN[1]"))));
}

TEST(TypecheckV2StructTest, ImplsForDifferentStructs) {
  EXPECT_THAT(R"(
struct Point { x: u32, y: u32 }

struct Line { a: Point, b: Point }

impl Point {
    fn area(self) -> u32 {
        self.x * self.y
    }
}

impl Line {
    fn height(self) -> u32 {
        self.b.y - self.a.y
    }
}

const P = Point{x: u32:4, y:u32:2};
const A = P.area(); // 8
const L = Line{a: P, b: Point{x: u32:4, y: u32:4}};
const H = L.height(); // 2
const Z = uN[A + H]:0;
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("A", "uN[32]"),
                                      HasNodeWithType("H", "uN[32]"),
                                      HasNodeWithType("Z", "uN[10]"))));
}

TEST(TypecheckV2StructTest, ImplFunctionUsingStructMembersIndirect) {
  EXPECT_THAT(R"(
struct Point { x: u32, y: u32 }

impl Point {
    fn area(self) -> u32 {
        self.x * self.y
    }
}

const P = Point{x: u32:4, y:u32:2};
const Y = P;
const W = Y;
const X = uN[W.area()]:0;
)",
              TypecheckSucceeds(HasNodeWithType("X", "uN[8]")));
}

TEST(TypecheckV2StructTest, InstanceMethodCalledStaticallyWithNoParamsFails) {
  EXPECT_THAT(R"(
struct Point { x: u32, y: u32 }

impl Point {
    fn area(self) -> u32 {
        self.x * self.y
    }
}

const P = Point::area();
)",
              TypecheckFails(HasSubstr("Expected 1 argument(s) but got 0")));
}

TEST(TypecheckV2StructTest, ImplFunctionCalledOnSelf) {
  EXPECT_THAT(R"(
struct Rect { width: u32, height: u32 }

impl Rect {
    const BORDER = u32:2;
    fn compute_width(self) -> u32 { self.width + BORDER * 2 }
    fn compute_height(self) -> u32 { self.height + BORDER * 2 }

    fn area(self) -> u32 {
        self.compute_width() * self.compute_height()
    }
}

const R = Rect { width: 2, height: 4 };
const Z = uN[R.area()]:0;
)",
              TypecheckSucceeds(HasNodeWithType("Z", "uN[48]")));
}

TEST(TypecheckV2StructTest, ParametricInstanceMethodCallingNonParametricOne) {
  XLS_EXPECT_OK(TypecheckV2(
      R"(
struct T {
   t: u32
}

impl T {
  fn foo(self) -> u32 { self.t }
}

struct S {
   a: T
}

impl S {
  fn foo<N: u32>(self) -> u32 { self.a.foo() }
}

fn main() -> u32 {
  let s = S { a: T { t: 5 } };
  s.foo<32>()
}
)"));
}

TEST(TypecheckV2StructTest, ParametricInstanceMethodWithParametricSignedness) {
  EXPECT_THAT(
      R"(
struct S {}

impl S {
  fn foo<S: bool, N: u32>(self, a: xN[S][N]) -> xN[S][N] { a + 1 }
}

const X = S{}.foo(u16:100);
const Y = S{}.foo(s32:200);
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[16]"),
                              HasNodeWithType("Y", "sN[32]"))));
}

TEST(TypecheckV2StructTest, ParametricInstanceMethodOfParametricStruct) {
  EXPECT_THAT(
      R"(
struct S<M: u32> {}

impl S<M> {
  fn add<N: u32>(self) -> u32 { M + N }
}

const X = S<4>{}.add<3>();
const Y = S<5>{}.add<6>();
const_assert!(X == 7);
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[32]"),
                              HasNodeWithType("Y", "uN[32]"))));
}

TEST(TypecheckV2StructTest,
     ParametricInstanceMethodOfParametricStructDuplicatesParametric) {
  EXPECT_THAT(
      R"(
struct S<M: u32> {}

impl S<M> {
  fn add<M: u32>(self) -> u32 { M }
}

const X = S<4>{}.add<3>();
const Y = S<5>{}.add<6>();
)",
      TypecheckFails(HasSubstr(
          "Parametric binding `M` shadows binding from struct definition")));
}

TEST(TypecheckV2StructTest, ParametricInstanceMethodUsingParametricConstant) {
  EXPECT_THAT(
      R"(
struct S<M: u32> {}

impl S<M> {
  const M_VALUE = M;
  fn add<N: u32>(self) -> u32 { M_VALUE + N }
}

const X = S<4>{}.add<3>();
const Y = S<5>{}.add<6>();
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[32]"),
                              HasNodeWithType("Y", "uN[32]"))));
}

TEST(TypecheckV2StructTest,
     ParametricInstanceMethodUsingStaticParametricAsDefault) {
  EXPECT_THAT(
      R"(
struct S<M: u32> {}

impl S<M> {
  fn add<N: u32, P: u32 = {M + N}>(self) -> uN[P] { uN[P]:0 }
}

const X = S<4>{}.add<3>();
const Y = S<5>{}.add<6>();
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[7]"),
                              HasNodeWithType("Y", "uN[11]"))));
}

TEST(TypecheckV2StructTest, StackedParametricInstanceMethodCalls) {
  EXPECT_THAT(
      R"(
struct Data<M: u32> {
  a: uN[M]
}

impl Data<M> {
  fn bar<N: u32>(self, b: uN[N]) -> uN[N] { b }

  fn foo<N: u32>(self, b: uN[N]) -> (uN[M], uN[N], u16) {
    (self.a, b, self.bar(b as u16))
  }
}

const X = Data{a: u3:5}.foo(u6:42);
const Y = Data{a: u7:120}.foo(u9:256);
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("X", "(uN[3], uN[6], uN[16])"),
                              HasNodeWithType("Y", "(uN[7], uN[9], uN[16])"))));
}

TEST(TypecheckV2StructTest, ImportStructImpl) {
  constexpr std::string_view kImported = R"(
pub struct S { x: u5 }

impl S {
  fn X(self) -> u5 { self.x }
}
)";
  constexpr std::string_view kProgram = R"(
import imported;

fn main() -> u5 {
  let s = imported::S{x: 1};
  s.X()
})";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data).status());
  EXPECT_THAT(
      TypecheckV2(kProgram, "main", &import_data),
      IsOkAndHolds(HasTypeInfo(AllOf(HasNodeWithType("s.X()", "uN[5]"),
                                     HasNodeWithType("s", "S { x: uN[5] }")))));
}

TEST(TypecheckV2StructTest, ImportImplUseStaticFunction) {
  constexpr std::string_view kImported = R"(
pub struct S {}

impl S {
  fn X() -> u5[2] { [1, 2] }
}
)";
  constexpr std::string_view kProgram = R"(
import imported;

fn main() -> u5 {
  imported::S::X()[0]
})";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data).status());
  EXPECT_THAT(
      TypecheckV2(kProgram, "main", &import_data),
      IsOkAndHolds(HasTypeInfo(HasNodeWithType("main", "() -> uN[5]"))));
}

TEST(TypecheckV2StructTest, ImportedMissingStaticFunctionOnImpl) {
  constexpr std::string_view kImported = R"(
pub struct S {}

impl S { }
)";
  constexpr std::string_view kProgram = R"(
import imported;

fn main() -> u5 {
  imported::S::X()[0]
})";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data).status());
  EXPECT_THAT(TypecheckV2(kProgram, "main", &import_data),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("is not defined by the impl")));
}

TEST(TypecheckV2StructTest, ImportedStaticFunctionOnStructWithoutImpl) {
  constexpr std::string_view kImported = R"(
pub struct S {}
)";
  constexpr std::string_view kProgram = R"(
import imported;

fn main() -> u5 {
  imported::S::X()[0]
})";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data).status());
  EXPECT_THAT(TypecheckV2(kProgram, "main", &import_data),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("no impl defining")));
}

TEST(TypecheckV2StructTest, ImportImplUseStaticConstantTypeMismatch) {
  constexpr std::string_view kImported = R"(
pub struct S {}

impl S {
  const SOME_ARRAY = u5[2]:[1, 2];
}
)";
  constexpr std::string_view kProgram = R"(
import imported;

fn main() -> u32 {
  imported::S::SOME_ARRAY[1]
})";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data).status());
  EXPECT_THAT(TypecheckV2(kProgram, "main", &import_data),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasTypeMismatch("u5", "u32")));
}

TEST(TypecheckV2StructTest, ImportedImplParametric) {
  constexpr std::string_view kImported = R"(
pub struct Empty<X: u32> { }

impl Empty<X> {
   const IMPORTED = 2 * X;
}

)";
  constexpr std::string_view kProgram = R"(
import imported;

fn main() -> uN[6] {
    type MyEmpty = imported::Empty<3>;
    uN[MyEmpty::IMPORTED]:0
})";
  auto import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data));
  EXPECT_THAT(TypecheckV2(kProgram, "main", &import_data),
              IsOkAndHolds(HasTypeInfo(
                  AllOf(HasNodeWithType("main", "() -> uN[6]"),
                        HasNodeWithType("uN[MyEmpty::IMPORTED]:0", "uN[6]")))));
}

TEST(TypecheckV2StructTest, ImportedImplTypeAlias) {
  constexpr std::string_view kImported = R"(
pub struct Empty<X: u32> { }

impl Empty<X> {
   const IMPORTED = u32:2 * X;
}

)";
  constexpr std::string_view kProgram = R"(
import imported;

type MyEmpty = imported::Empty<5>;

fn main() -> uN[10] {
    let var = uN[MyEmpty::IMPORTED]:0;
    var
})";
  auto import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data));
  EXPECT_THAT(TypecheckV2(kProgram, "main", &import_data),
              IsOkAndHolds(HasTypeInfo(HasNodeWithType("var", "uN[10]"))));
}

TEST(TypecheckV2StructTest, ImportedImplTypeAliasWithFunction) {
  constexpr std::string_view kImported = R"(
pub struct Empty { }

impl Empty {
   fn some_val() -> u5 {
       u5:4
   }
}

)";
  constexpr std::string_view kProgram = R"(
import imported;

type MyEmpty = imported::Empty;

fn main() -> uN[5] {
    MyEmpty::some_val()
})";
  auto import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data));
  EXPECT_THAT(TypecheckV2(kProgram, "main", &import_data),
              IsOkAndHolds(HasTypeInfo(
                  HasNodeWithType("MyEmpty::some_val()", "uN[5]"))));
}

TEST(TypecheckV2StructTest, ImportedImplConstant) {
  constexpr std::string_view kImported = R"(
pub struct X<EXP_SZ: u32, FRACTION_SZ: u32> { }

impl X<EXP_SZ, FRACTION_SZ> {
    const EXP_SIZE = EXP_SZ;
    const FRACTION_SIZE = FRACTION_SZ;
    const TOTAL_SIZE = u32:1 + EXP_SZ + FRACTION_SZ;
}

pub type MyX = X<u32:23, u32:8>;
)";

  constexpr std::string_view kProgram = R"(
import imported;

const MXU_RESULT_F32_PADDING = u32:5;
const PADDED_F32_W = imported::MyX::TOTAL_SIZE + MXU_RESULT_F32_PADDING;

const_assert!(PADDED_F32_W == u32:37);
)";
  auto import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data));
  EXPECT_THAT(TypecheckV2(kProgram, "main", &import_data),
              IsOkAndHolds(HasTypeInfo(
                  HasNodeWithType("MXU_RESULT_F32_PADDING", "uN[32]"))));
}

TEST(TypecheckV2StructTest, ImportedImplConstInType) {
  constexpr std::string_view kImported = R"(
pub struct MyStruct<SZ: u32> {
    val: bits[SZ],
}

impl MyStruct<SZ> {
    const SIZE = SZ;
}

pub type S3 = MyStruct<u32:3>;

  )";
  constexpr std::string_view kProgram = R"(
import imported;

type S3 = imported::S3;

struct FloatFormatSpec<SZ: u32> {
    exp_bias: sN[SZ],
    min_normal: imported::MyStruct<SZ>,
}

const S3_FORMAT_SPEC = FloatFormatSpec<S3::SIZE> {
    exp_bias: sN[S3::SIZE]:2,
    min_normal: imported::MyStruct<S3::SIZE> {
        val: u3:0b000,
    },
};
)";
  auto import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data));
  EXPECT_THAT(TypecheckV2(kProgram, "main", &import_data),
              IsOkAndHolds(HasTypeInfo(
                  HasNodeWithType("S3_FORMAT_SPEC",
                                  "FloatFormatSpec { exp_bias: sN[3], "
                                  "min_normal: MyStruct { val: uN[3] } }"))));
}

TEST(TypecheckV2StructTest, ImplFnUsesStructParametricInInvocation) {
  EXPECT_THAT(R"(
fn helper<N: u32>() -> uN[N] {
  uN[N]:1
}

struct st<N: u32> {}

impl st<N> {
  fn call(self, i: u32) -> uN[N] {
    helper<N>()
  }
}

fn my_conversion<N: u32>(arr: u32[3]) -> uN[N][3] {
  map(arr, st<N>{}.call)
}

const M = u32:0..3;
const ARR = my_conversion<16>(M);
const_assert!(ARR[1] == u16:1);
)",
              TypecheckSucceeds(HasNodeWithType("ARR", "uN[16][3]")));
}

TEST(TypecheckV2StructTest, InstanceMethodNotDefined) {
  EXPECT_THAT(
      R"(
struct Point { x: u32, y: u32 }

impl Point { }

const P = Point { x: u32:1, y: u32:4 };
const Z = uN[P.area()]:0;
)",
      TypecheckFails(HasSubstr(
          "Name 'area' is not defined by the impl for struct 'Point'")));
}

TEST(TypecheckV2StructTest, ImplFunctionUsingStructMembersAndArg) {
  EXPECT_THAT(R"(
struct Point { x: u32, y: u32 }

impl Point {
    fn area(self, a: u32, b: u32) -> u32 {
        self.x * self.y * a * b
    }
}

const P = Point{x: 4, y:2};
const Y = P.area(2, 1);
const Z = uN[Y]:0;
)",
              TypecheckSucceeds(HasNodeWithType("Z", "uN[16]")));
}

TEST(TypecheckV2StructTest, ImplFunctionUsingStructMembersExplicitSelfType) {
  EXPECT_THAT(R"(
struct Point { x: u32, y: u32 }

impl Point {
    fn area(self: Self) -> u32 {
        self.x * self.y
    }
}

const P = Point{x: u32:4, y:u32:2};
const A = P.area();
const Z = uN[A]:0;
)",
              TypecheckSucceeds(HasNodeWithType("Z", "uN[8]")));
}

TEST(TypecheckV2StructTest, ParametricImplConstant) {
  EXPECT_THAT(R"(
struct S<N: u32> {}

impl S<N> {
  const N_VALUE = N;
}

const X = uN[S<1>::N_VALUE]:0;
const Y = uN[S<2>::N_VALUE]:0;
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[1]"),
                                      HasNodeWithType("Y", "uN[2]"))));
}

TEST(TypecheckV2StructTest, ImplConstantForwardParametricToFunction) {
  EXPECT_THAT(R"(
fn f<A: u32>(a: uN[A]) -> uN[A] { a }

struct S<N: u32> {}

impl S<N> {
  const F_VALUE = f<N>(1);
}

// Note that this was only failing originally if a type alias rather than the
// actual struct name was used to access `F_VALUE`.
type S1 = S<1>;
type S2 = S<2>;

const X = S1::F_VALUE;
const Y = S2::F_VALUE;
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[1]"),
                                      HasNodeWithType("Y", "uN[2]"))));
}

TEST(TypecheckV2StructTest, SumOfImplConstantsFromDifferentParameterizations) {
  EXPECT_THAT(R"(
struct S<N: u32> {}

impl S<N> {
  const N_VALUE = N;
}

const X = uN[S<2>::N_VALUE + S<1>::N_VALUE]:0;
)",
              TypecheckSucceeds(HasNodeWithType("X", "uN[3]")));
}

TEST(TypecheckV2StructTest, ParametricConstantUsingConstantFromSameImpl) {
  EXPECT_THAT(R"(
struct S<N: u32> {}

fn f<N: u32>() -> u32 { N + 1 }

impl S<N> {
  const N_PLUS_1_VALUE = f<N>();
  const N_PLUS_2_VALUE = f<N_PLUS_1_VALUE>();
}

const X = uN[S<2>::N_PLUS_2_VALUE]:0;
const Y = uN[S<10>::N_PLUS_2_VALUE]:0;
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[4]"),
                                      HasNodeWithType("Y", "uN[12]"))));
}

TEST(TypecheckV2StructTest, StaticParametricImplFnUsingConstant) {
  EXPECT_THAT(R"(
struct S<N: u32> {}

impl S<N> {
  const N_PLUS_1_VALUE = N + 1;
  fn foo(a: u32) -> u32 { N_PLUS_1_VALUE + N + a }
}

// Note that we would likely need constexpr evaluator to be aware of impl
// TypeInfos in order to use these return values in a mandatory constexpr
// context.
const X = S<3>::foo(10);
const Y = S<4>::foo(10);
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[32]"),
                                      HasNodeWithType("Y", "uN[32]"))));
}

TEST(TypecheckV2StructTest, StaticImplFnUsingParametricForInterface) {
  EXPECT_THAT(R"(
struct S<N: u32> {}

impl S<N> {
  fn foo(a: uN[N]) -> uN[N] { a }
}

const X = S<16>::foo(10);
const Y = S<32>::foo(11);
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[16]"),
                                      HasNodeWithType("Y", "uN[32]"))));
}

TEST(TypecheckV2StructTest, ParametricConstantUsingConstantFromOtherImpl) {
  EXPECT_THAT(R"(
// Note that the entities in here are sensitive to positioning due to
// https://github.com/google/xls/issues/1911

struct S<N: u32> {}

fn f<N: u32>() -> u32 { N + 1 }

impl S<N> {
  const N_PLUS_1_VALUE = f<N>();
}

struct T<N: u32> {}

impl T<N> {
  const N_PLUS_2_VALUE = f<{S<N>::N_PLUS_1_VALUE}>();
}

const X = uN[T<2>::N_PLUS_2_VALUE]:0;
const Y = uN[T<10>::N_PLUS_2_VALUE]:0;
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[4]"),
                                      HasNodeWithType("Y", "uN[12]"))));
}

TEST(TypecheckV2StructTest, ParametricBasedImplConstantType) {
  EXPECT_THAT(R"(
struct S<N: u32> {}

impl S<N> {
  const C = uN[N]:0;
}

const C2 = S<2>::C;
const C3 = S<3>::C;
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("C2", "uN[2]"),
                                      HasNodeWithType("C3", "uN[3]"))));
}

TEST(TypecheckV2StructTest, ImplConstantUsingParametricDefault) {
  EXPECT_THAT(R"(
struct S<A: u32, B: u32 = {A * 2}> {}

impl S<A, B> {
  const C = B;
}

const X = uN[S<2>::C]:0;
const Y = uN[S<3>::C]:0;
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[4]"),
                                      HasNodeWithType("Y", "uN[6]"))));
}

TEST(TypecheckV2StructTest, ParametricImplConstantUsedWithMissingParametrics) {
  EXPECT_THAT(
      R"(
struct S<A: u32, B: u32 = {A * 2}> {}

impl S<A, B> {
  const C = B;
}

const X = S::C;
)",
      TypecheckFails(HasSubstr("Use of `S` with missing parametric(s): A")));
}

TEST(TypecheckV2StructTest, ParametricImplConstantUsedWithTooManyParametrics) {
  EXPECT_THAT(
      R"(
struct S<A: u32> {}

impl S<A> {
  const C = A;
}

const X = S<1, 2>::C;
)",
      TypecheckFails(
          HasSubstr("Too many parametric values supplied; limit: 1 given: 2")));
}

TEST(TypecheckV2StructTest, InstanceMethodReturningStaticParametricType) {
  EXPECT_THAT(
      R"(
struct S<A: u32> {}

impl S<A> {
  fn foo(self) -> uN[A] { uN[A]:0 }
}

const X = S<16>{}.foo();
const Y = S<32>{}.foo();
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[16]"),
                              HasNodeWithType("Y", "uN[32]"))));
}

TEST(TypecheckV2StructTest, InstanceMethodReturningParametricConstType) {
  EXPECT_THAT(
      R"(
struct S<A: u32> {}

impl S<A> {
  const C = A;
  fn foo(self) -> uN[C] { uN[C]:0 }
}

const X = S<16>{}.foo();
const Y = S<32>{}.foo();
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[16]"),
                              HasNodeWithType("Y", "uN[32]"))));
}

TEST(TypecheckV2StructTest, InstanceMethodTakingStaticParametricType) {
  EXPECT_THAT(
      R"(
struct S<A: u32> {}

impl S<A> {
  fn foo(self, a: uN[A]) -> uN[A] { a + 1 }
}

const X = S<16>{}.foo(100);
const Y = S<32>{}.foo(200);
const_assert!(X == 101);
const_assert!(Y == 201);
)",
      TypecheckSucceeds(AllOf(
          HasNodeWithType("X", "uN[16]"), HasNodeWithType("Y", "uN[32]"),
          HasNodeWithType("100", "uN[16]"), HasNodeWithType("200", "uN[32]"))));
}

TEST(TypecheckV2StructTest,
     InstanceMethodTakingStaticParametricTypeAndUsingSelf) {
  EXPECT_THAT(
      R"(
struct S<A: u32> {
  val: uN[A]
}

impl S<A> {
  fn foo(self, a: uN[A]) -> uN[A] { a + self.val }
}

const X = S<16>{val: 1}.foo(100);
const Y = S<32>{val: 2}.foo(200);
const_assert!(X == 101);
const_assert!(Y == 202);
)",
      TypecheckSucceeds(AllOf(
          HasNodeWithType("X", "uN[16]"), HasNodeWithType("Y", "uN[32]"),
          HasNodeWithType("100", "uN[16]"), HasNodeWithType("200", "uN[32]"))));
}

TEST(TypecheckV2StructTest, InstanceMethodTakingStaticConstType) {
  EXPECT_THAT(
      R"(
struct S<A: u32> {}

impl S<A> {
  const C = A + 1;
  fn foo(self, a: uN[C]) -> uN[C] { a + 1 }
}

const X = S<16>{}.foo(100);
const Y = S<32>{}.foo(200);
 const_assert!(X == 101);
)",
      TypecheckSucceeds(AllOf(
          HasNodeWithType("X", "uN[17]"), HasNodeWithType("Y", "uN[33]"),
          HasNodeWithType("100", "uN[17]"), HasNodeWithType("200", "uN[33]"))));
}

TEST(TypecheckV2StructTest, ParametricInstanceMethodTakingStaticConstType) {
  EXPECT_THAT(
      R"(
struct S<A: u32> {}

impl S<A> {
  const C = A + 1;
  fn foo<N: u32>(self, a: uN[C]) -> uN[C] { a + 1 }
}

const X = S<16>{}.foo<20>(100);
const Y = S<32>{}.foo<16>(200);
const_assert!(X == 101);
)",
      TypecheckSucceeds(AllOf(
          HasNodeWithType("X", "uN[17]"), HasNodeWithType("Y", "uN[33]"),
          HasNodeWithType("100", "uN[17]"), HasNodeWithType("200", "uN[33]"))));
}

TEST(TypecheckV2StructTest, ParametricInstanceMethod) {
  EXPECT_THAT(
      R"(
struct S {}

impl S {
  fn foo<N: u32>(self, a: uN[N]) -> uN[N] { a + 1 }
}

const X = S{}.foo(u16:100);
const Y = S{}.foo(u32:200);
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[16]"),
                              HasNodeWithType("Y", "uN[32]"))));
}

TEST(TypecheckV2StructTest, ParametricInstanceMethodUsingSelf) {
  EXPECT_THAT(
      R"(
struct S {
   a: u32
}

impl S {
  // Originally there was a bug that self would only be resolved in here if the
  // struct was also parametric.
  fn foo<N: u32>(self, b: uN[N]) -> uN[N + 32] { b ++ self.a }
}

const X = S{a: 1}.foo(u16:100);
const Y = S{a: 2}.foo(u32:200);
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[48]"),
                              HasNodeWithType("Y", "uN[64]"))));
}

TEST(TypecheckV2StructTest, PartialInferenceForStructInstance) {
  EXPECT_THAT(
      R"(
struct S<DefinedParametric: u32, InferredParametric: u32> {
    x: uN[InferredParametric],
}

impl S<DefinedParametric, InferredParametric> {
    fn call(self, i: u32) -> uN[DefinedParametric] {
        i as uN[DefinedParametric] + self.x
    }
}
fn main<N: u32>() -> uN[N] {
    let x: uN[N] = 5;
    S<N>{ x: x }.call(u32:1)
}

const ONE = main<16>();
const TWO = main<24>();
const_assert!(ONE == u16:6);
const_assert!(TWO == u24:6);
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("ONE", "uN[16]"),
                              HasNodeWithType("TWO", "uN[24]"))));
}

TEST(TypecheckV2StructTest, PartialInferenceForStructInstanceGeneric) {
  EXPECT_THAT(
      R"(
#![feature(generics)]

struct S<DefinedType: type, InferredType: type> {
    x: InferredType,
}

impl S<DefinedType, InferredType> {
    fn call(self, i: u32) -> DefinedType {
        i as DefinedType + self.x
    }
}
fn main<T: type>() -> T {
    let x: T = 5;
    S<T>{ x: x }.call(u32:1)
}

const ONE = main<u16>();
const TWO = main<u24>();
const_assert!(ONE == u16:6);
const_assert!(TWO == u24:6);
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("ONE", "uN[16]"),
                              HasNodeWithType("TWO", "uN[24]"))));
}

TEST(TypecheckV2StructTest, SimpleStructConstructorThenUse) {
  XLS_EXPECT_OK(TypecheckV2(
      R"(
struct S {
   a: u32
}

impl S {
  fn new(a: u32) -> Self {
    S { a }
  }

  fn foo(self) -> u32 { self.a }
}

const X = S::new(5).foo();
const_assert!(X == 5);
)"));
}

TEST(TypecheckV2StructTest, SimpleStructConstructorThenParametricUse) {
  XLS_EXPECT_OK(TypecheckV2(
      R"(
struct S {
   a: u32
}

impl S {
  fn new(a: u32) -> Self {
    S { a }
  }

  fn foo<N: u32>(self) -> u32 { self.a }
}

const X = S::new(5).foo<32>();
const_assert!(X == 5);
)"));
}

TEST(TypecheckV2StructTest, ParametricStructConstructorThenUse) {
  XLS_EXPECT_OK(TypecheckV2(
      R"(
struct S<N: u32> {
   a: uN[N]
}

impl S<N> {
  fn new(a: uN[N]) -> Self {
    S<N> { a }
  }

  fn foo(self) -> uN[N] { self.a }
}

const X = S<32>::new(5).foo();
const_assert!(X == 5);
)"));
}

TEST(TypecheckV2StructTest, StructInstanceFieldMissing) {
  EXPECT_THAT(
      R"(
struct MyStruct {
    x: u32,
    y: u8,
}

fn create_f_domain() -> MyStruct {
   MyStruct {
     x: u32:0..10,
   }
}
)",
      TypecheckFails(HasSubstr("is missing member(s): `y`")));
}

}  // namespace
}  // namespace xls::dslx
