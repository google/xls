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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/parse_and_typecheck.h"

namespace xls::dslx {
namespace {

using status_testing::StatusIs;
using testing::HasSubstr;

// Helper for parsing/typechecking a snippet of DSLX text.
absl::Status Typecheck(absl::string_view text) {
  ImportData import_data;
  return ParseAndTypecheck(text, "fake.x", "fake", &import_data,
                           /*additional_search_paths=*/{})
      .status();
}

TEST(TypecheckTest, ParametricWrongArgCount) {
  absl::string_view text = R"(
fn id<N: u32>(x: bits[N]) -> bits[N] { x }
fn f() -> u32 { id(u8:3, u8:4) }
)";
  EXPECT_THAT(
      Typecheck(text),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Expected 1 parameter(s) but got 2 argument(s)")));
}

TEST(TypecheckTest, ParametricTooManyExplicitSupplied) {
  absl::string_view text = R"(
fn id<X: u32>(x: bits[X]) -> bits[X] { x }
fn main() -> u32 { id<u32:32, u32:64>(u32:5) }
)";
  EXPECT_THAT(
      Typecheck(text),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Too many parametric values supplied; limit: 1 given: 2")));
}

TEST(TypecheckTest, Identity) {
  XLS_EXPECT_OK(Typecheck("fn f(x: u32) -> u32 { x }"));
  XLS_EXPECT_OK(Typecheck("fn f(x: bits[3], y: bits[4]) -> bits[3] { x }"));
  XLS_EXPECT_OK(Typecheck("fn f(x: bits[3], y: bits[4]) -> bits[4] { y }"));
  EXPECT_THAT(
      Typecheck("fn f(x: bits[3], y: bits[4]) -> bits[5] { y }"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("uN[4] vs uN[5]: Return type of function body")));
}

TEST(TypecheckTest, TokenIdentity) {
  XLS_EXPECT_OK(Typecheck("fn f(x: token) -> token { x }"));
}

TEST(TypecheckTest, Unit) {
  XLS_EXPECT_OK(Typecheck("fn f(x: u32) -> () { () }"));
  XLS_EXPECT_OK(Typecheck("fn f(x: u32) { () }"));
}

TEST(TypecheckTest, Arithmetic) {
  // Simple add.
  XLS_EXPECT_OK(Typecheck("fn f(x: u32, y: u32) -> u32 { x + y }"));
  // Wrong return type (implicitly unit).
  EXPECT_THAT(
      Typecheck("fn f(x: u32, y: u32) { x + y }"),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("uN[32] vs ()")));
  // Wrong return type (implicitly unit).
  EXPECT_THAT(
      Typecheck(R"(
      fn f<N: u32>(x: bits[N], y: bits[N]) { x + y }
      fn g() -> u64 { f(u64:5, u64:5) }
      )"),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("uN[64] vs ()")));
  // Mixing widths not permitted.
  EXPECT_THAT(Typecheck("fn f(x: u32, y: bits[4]) { x + y }"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("uN[32] vs uN[4]")));
  // Parametric same-width is ok!
  XLS_EXPECT_OK(
      Typecheck("fn f<N: u32>(x: bits[N], y: bits[N]) -> bits[N] { x + y }"));
}

TEST(TypecheckTest, Unary) {
  XLS_EXPECT_OK(Typecheck("fn f(x: u32) -> u32 { !x }"));
  XLS_EXPECT_OK(Typecheck("fn f(x: u32) -> u32 { -x }"));
}

TEST(TypecheckTest, Let) {
  XLS_EXPECT_OK(Typecheck("fn f() -> u32 { let x: u32 = u32:2; x }"));
  EXPECT_THAT(Typecheck(
                  R"(fn f() -> u32 {
        let x: u32 = u32:2;
        let y: bits[4] = bits[4]:3;
        y
      }
      )"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("uN[4] vs uN[32]")));
  XLS_EXPECT_OK(Typecheck(
      "fn f() -> u32 { let (x, y): (u32, bits[4]) = (u32:2, bits[4]:3); x }"));
}

TEST(TypecheckTest, LetBadRhs) {
  EXPECT_THAT(
      Typecheck(
          R"(fn f() -> bits[2] {
          let (x, (y, (z,))): (u32, (bits[4], (bits[2],))) = (u32:2, bits[4]:3);
          z
        })"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("did not match inferred type of right hand side")));
}

TEST(TypecheckTest, ParametricInvocation) {
  std::string program = R"(
fn p<N: u32>(x: bits[N]) -> bits[N] { x + bits[N]:1 }
fn f() -> u32 { p(u32:3) }
)";
  XLS_EXPECT_OK(Typecheck(program));
}

TEST(TypecheckTest, ParametricInvocationWithTuple) {
  std::string program = R"(
fn p<N: u32>(x: bits[N]) -> (bits[N], bits[N]) { (x, x) }
fn f() -> (u32, u32) { p(u32:3) }
)";
  XLS_EXPECT_OK(Typecheck(program));
}

TEST(TypecheckTest, DoubleParametricInvocation) {
  std::string program = R"(
fn p<N: u32>(x: bits[N]) -> bits[N] { x + bits[N]:1 }
fn o<M: u32>(x: bits[M]) -> bits[M] { p(x) }
fn f() -> u32 { o(u32:3) }
)";
  XLS_EXPECT_OK(Typecheck(program));
}

TEST(TypecheckTest, ParametricInvocationConflictingArgs) {
  std::string program = R"(
fn id<N: u32>(x: bits[N], y: bits[N]) -> bits[N] { x }
fn f() -> u32 { id(u8:3, u32:5) }
)";
  EXPECT_THAT(Typecheck(program), StatusIs(absl::StatusCode::kInvalidArgument,
                                           HasSubstr("saw: 8; then: 32")));
}

TEST(TypecheckTest, ParametricWrongKind) {
  std::string program = R"(
fn id<N: u32>(x: bits[N]) -> bits[N] { x }
fn f() -> u32 { id((u8:3,)) }
)";
  EXPECT_THAT(Typecheck(program), StatusIs(absl::StatusCode::kInvalidArgument,
                                           HasSubstr("different kinds")));
}

TEST(TypecheckTest, ParametricWrongNumberOfDims) {
  std::string program = R"(
fn id<N: u32, M: u32>(x: bits[N][M]) -> bits[N][M] { x }
fn f() -> u32 { id(u32:42) }
)";
  EXPECT_THAT(
      Typecheck(program),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("types are different kinds (array vs ubits)")));
}

TEST(TypecheckTest, RecursionCausesError) {
  std::string program = "fn f(x: u32) -> u32 { f(x) }";
  EXPECT_THAT(Typecheck(program),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Recursion detected while typechecking")));
}

TEST(TypecheckTest, ParametricRecursionCausesError) {
  std::string program = R"(
fn f<X: u32>(x: bits[X]) -> u32 { f(x) }
fn g() -> u32 { f(u32: 5) }
)";
  EXPECT_THAT(Typecheck(program),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Recursion detected while typechecking")));
}

TEST(TypecheckTest, HigherOrderRecursionCausesError) {
  std::string program = R"(
fn h<Y: u32>(y: bits[Y]) -> bits[Y] { h(y) }
fn g() -> u32[3] {
    let x0 = u32[3]:[0, 1, 2];
    map(x0, h)
}
)";
  EXPECT_THAT(Typecheck(program),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Recursion detected while typechecking")));
}

TEST(TypecheckTest, InvokeWrongArg) {
  std::string program = R"(
fn id_u32(x: u32) -> u32 { x }
fn f(x: u8) -> u8 { id_u32(x) }
)";
  EXPECT_THAT(
      Typecheck(program),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Mismatch between parameter and argument types")));
}

TEST(TypecheckTest, BadTupleType) {
  std::string program = R"(
fn f() -> u32 {
  let (a, b, c): (u32, u32) = (u32:1, u32:2, u32:3);
  a
}
)";
  EXPECT_THAT(
      Typecheck(program),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Annotated type did not match inferred type")));
}

TEST(TypecheckTest, LogicalAndOfComparisons) {
  XLS_EXPECT_OK(Typecheck("fn f(a: u8, b: u8) -> bool { a == b }"));
  XLS_EXPECT_OK(Typecheck(
      "fn f(a: u8, b: u8, c: u32, d: u32) -> bool { a == b && c == d }"));
}

TEST(TypecheckTest, Typedef) {
  XLS_EXPECT_OK(Typecheck(R"(
type MyTypeDef = (u32, u8);
fn id(x: MyTypeDef) -> MyTypeDef { x }
fn f() -> MyTypeDef { id((u32:42, u8:127)) }
)"));
}

TEST(TypecheckTest, For) {
  XLS_EXPECT_OK(Typecheck(R"(
fn f() -> u32 {
  for (i, accum): (u32, u32) in range(u32:0, u32:3) {
    let new_accum: u32 = accum + i;
    new_accum
  }(u32:0)
})"));
}

TEST(TypecheckTest, ForNoAnnotation) {
  XLS_EXPECT_OK(Typecheck(R"(
fn f() -> u32 {
  for (i, accum) in range(u32:0, u32:3) {
    accum
  }(u32:0)
})"));
}

TEST(TypecheckTest, ForBuiltinInBody) {
  XLS_EXPECT_OK(Typecheck(R"(
fn f() -> u32 {
  for (i, accum): (u32, u32) in range(u32:0, u32:3) {
    trace!(accum)
  }(u32:0)
})"));
}

TEST(TypecheckTest, ForNestedBindings) {
  XLS_EXPECT_OK(Typecheck(R"(
fn f(x: u32) -> (u32, u8) {
  for (i, (x, y)): (u32, (u32, u8)) in range(u32:0, u32:3) {
    (x, y)
  }((x, u8:42))
}
)"));
}

TEST(TypecheckTest, ForWithBadTypeTree) {
  EXPECT_THAT(
      Typecheck(R"(
fn f(x: u32) -> (u32, u8) {
  for (i, (x, y)): (u32, u8) in range(u32:0, u32:3) {
    (x, y)
  }((x, u8:42))
})"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("(uN[32], uN[8]) vs (uN[32], (uN[32], uN[8])): For-loop "
                    "annotated type did not match inferred type.")));
}

TEST(TypecheckTest, DerivedExprTypeMismatch) {
  EXPECT_THAT(
      Typecheck(R"(
fn p<X: u32, Y: bits[4] = X+X>(x: bits[X]) -> bits[X] { x }
fn f() -> u32 { p(u32:3) }
)"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "Annotated type of derived parametric value did not match")));
}

TEST(TypecheckTest, ParametricInstantiationVsArgOk) {
  XLS_EXPECT_OK(Typecheck(R"(
fn parametric<X: u32 = u32:5> (x: bits[X]) -> bits[X] { x }
fn main() -> bits[5] { parametric(u5:1) }
)"));
}

TEST(TypecheckTest, ParametricInstantiationVsArgError) {
  EXPECT_THAT(Typecheck(R"(
fn foo<X: u32 = u32: 5>(x: bits[X]) -> bits[X] { x }
fn bar() -> bits[10] { foo(u5:1) + foo(u10: 1) }
)"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Parametric constraint violated")));
}

TEST(TypecheckTest, ParametricInstantiationVsBodyOk) {
  XLS_EXPECT_OK(Typecheck(R"(
fn parametric<X: u32 = u32:5>() -> bits[5] { bits[X]:1 + bits[5]:1 }
fn main() -> bits[5] { parametric() }
)"));
}

TEST(TypecheckTest, ParametricInstantiationVsBodyError) {
  EXPECT_THAT(Typecheck(R"(
fn foo<X: u32 = u32:5>() -> bits[10] { bits[X]:1 + bits[10]:1 }
fn bar() -> bits[10] { foo() }
)"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("uN[5] vs uN[10]: Could not deduce type for "
                                 "binary operation '+'")));
}

TEST(TypecheckTest, ParametricInstantiationVsReturnOk) {
  XLS_EXPECT_OK(Typecheck(R"(
fn parametric<X: u32 = u32: 5>() -> bits[5] { bits[X]: 1 }
fn main() -> bits[5] { parametric() }
)"));
}

TEST(TypecheckTest, ParametricInstantiationVsReturnError) {
  EXPECT_THAT(
      Typecheck(R"(
fn foo<X: u32 = u32: 5>() -> bits[10] { bits[X]: 1 }
fn bar() -> bits[10] { foo() }
)"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Return type of function body for 'foo' did not match")));
}

TEST(TypecheckTest, ParametricIndirectInstantiationVsArgOk) {
  XLS_EXPECT_OK(Typecheck(R"(
fn foo<X: u32>(x1: bits[X], x2: bits[X]) -> bits[X] { x1 + x2 }
fn fazz<Y: u32>(y: bits[Y]) -> bits[Y] { foo(y, y + bits[Y]: 1) }
fn bar() -> bits[10] { fazz(u10: 1) }
)"));
}

TEST(TypecheckTest, ParametricInstantiationVsArgError2) {
  EXPECT_THAT(
      Typecheck(R"(
fn foo<X: u32>(x1: bits[X], x2: bits[X]) -> bits[X] { x1 + x2 }
fn fazz<Y: u32>(y: bits[Y]) -> bits[Y] { foo(y, y++y) }
fn bar() -> bits[10] { fazz(u10: 1) }
)"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Parametric value X was bound to different values")));
}

TEST(TypecheckTest, ParametricIndirectInstantiationVsBodyOk) {
  XLS_EXPECT_OK(Typecheck(R"(
fn foo<X: u32, R: u32 = X + X>(x: bits[X]) -> bits[R] {
  let a = bits[R]: 5;
  x++x + a
}
fn fazz<Y: u32, T: u32 = Y + Y>(y: bits[Y]) -> bits[T] { foo(y) }
fn bar() -> bits[10] { fazz(u5:1) }
)"));
}

TEST(TypecheckTest, ParametricIndirectInstantiationVsBodyError) {
  EXPECT_THAT(Typecheck(R"(
fn foo<X: u32, D: u32 = X + X>(x: bits[X]) -> bits[X] {
  let a = bits[D]:5;
  x + a
}
fn fazz<Y: u32>(y: bits[Y]) -> bits[Y] { foo(y) }
fn bar() -> bits[5] { fazz(u5:1) })"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("uN[5] vs uN[10]: Could not deduce type for "
                                 "binary operation '+'")));
}

TEST(TypecheckTest, ParametricIndirectInstantiationVsReturnOk) {
  XLS_EXPECT_OK(Typecheck(R"(
fn foo<X: u32, R: u32 = X + X>(x: bits[X]) -> bits[R] { x++x }
fn fazz<Y: u32, T: u32 = Y + Y>(y: bits[Y]) -> bits[T] { foo(y) }
fn bar() -> bits[10] { fazz(u5:1) }
)"));
}

TEST(TypecheckTest, ParametricIndirectInstantiationVsReturnError) {
  EXPECT_THAT(
      Typecheck(R"(
fn foo<X: u32, R: u32 = X + X>(x: bits[X]) -> bits[R] { x * x }
fn fazz<Y: u32, T: u32 = Y + Y>(y: bits[Y]) -> bits[T] { foo(y) }
fn bar() -> bits[10] { fazz(u5:1) }
)"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Return type of function body for 'foo' did not match")));
}

TEST(TypecheckTest, ParametricDerivedInstantiationVsArgOk) {
  XLS_EXPECT_OK(Typecheck(R"(
fn foo<X: u32, Y: u32 = X + X>(x: bits[X], y: bits[Y]) -> bits[X] { x }
fn bar() -> bits[5] { foo(u5:1, u10: 2) }
)"));
}

TEST(TypecheckTest, ParametricDerivedInstantiationVsArgError) {
  EXPECT_THAT(Typecheck(R"(
fn foo<X: u32, Y: u32 = X + X>(x: bits[X], y: bits[Y]) -> bits[X] { x }
fn bar() -> bits[5] { foo(u5:1, u11: 2) }
)"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Parametric constraint violated")));
}

TEST(TypecheckTest, ParametricDerivedInstantiationVsBodyOk) {
  XLS_EXPECT_OK(Typecheck(R"(
fn foo<W: u32, Z: u32 = W + W>(w: bits[W]) -> bits[1] {
    let val: bits[Z] = w++w + bits[Z]: 5;
    and_reduce(val)
}
fn bar() -> bits[1] { foo(u5: 5) + foo(u10: 10) }
)"));
}

TEST(TypecheckTest, ParametricDerivedInstantiationVsBodyError) {
  EXPECT_THAT(Typecheck(R"(
fn foo<W: u32, Z: u32 = W + W>(w: bits[W]) -> bits[1] {
  let val: bits[Z] = w + w;
  and_reduce(val)
}
fn bar() -> bits[1] { foo(u5:5) }
)"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("uN[10] vs uN[5]")));
}

TEST(TypecheckTest, ParametricDerivedInstantiationVsReturnOk) {
  XLS_EXPECT_OK(Typecheck(R"(
fn double<X: u32, Y: u32 = X + X> (x: bits[X]) -> bits[Y] { x++x }
fn foo<W: u32, Z: u32 = W + W> (w: bits[W]) -> bits[Z] { double(w) }
fn bar() -> bits[10] { foo(u5:1) }
)"));
}

TEST(TypecheckTest, ParametricDerivedInstantiationVsReturnError) {
  EXPECT_THAT(
      Typecheck(R"(
fn double<X: u32, Y: u32 = X + X>(x: bits[X]) -> bits[Y] { x + x }
fn foo<W: u32, Z: u32 = W + W>(w: bits[W]) -> bits[Z] { double(w) }
fn bar() -> bits[10] { foo(u5:1) }
)"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr(
                   "Return type of function body for 'double' did not match")));
}

TEST(TypecheckTest, ParametricDerivedInstantiationViaFnCall) {
  XLS_EXPECT_OK(Typecheck(R"(
fn double(n: u32) -> u32 { n * u32: 2 }
fn foo<W: u32, Z: u32 = double(W)>(w: bits[W]) -> bits[Z] { w++w }
fn bar() -> bits[10] { foo(u5:1) }
)"));
}

TEST(TypecheckTest, ParametricFnNotAlwaysPolymorphic) {
  EXPECT_THAT(Typecheck(R"(
fn foo<X: u32>(x: bits[X]) -> u1 {
  let non_polymorphic = x + u5: 1;
  u1:0
}
fn bar() -> bits[1] {
  foo(u5:5) ^ foo(u10:5)
}
)"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("uN[10] vs uN[5]: Could not deduce type for "
                                 "binary operation '+'")));
}

TEST(TypecheckTest, ParametricWidthSliceStartError) {
  EXPECT_THAT(Typecheck(R"(
fn make_u1<N: u32>(x: bits[N]) -> u1 {
  x[4 +: bits[1]]
}
fn bar() -> bits[1] {
  make_u1(u10:5) ^ make_u1(u2:1)
}
)"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Cannot fit slice start 4 in 2 bits")));
}

TEST(TypecheckTest, BitSliceOnParametricWidth) {
  XLS_EXPECT_OK(Typecheck(R"(
fn get_middle_bits<N: u32, R: u32 = N - u32:2>(x: bits[N]) -> bits[R] {
  x[1:-1]
}

fn caller() {
  let x1: u2 = get_middle_bits(u4:15);
  let x2: u4 = get_middle_bits(u6:63);
  ()
}
)"));
}

TEST(TypecheckTest, ParametricMapNonPolymorphic) {
  EXPECT_THAT(Typecheck(R"(
fn add_one<N: u32>(x: bits[N]) -> bits[N] { x + bits[5]:1 }

fn main() {
  let arr = [u5:1, u5:2, u5:3];
  let mapped_arr = map(arr, add_one);
  let type_error = add_one(u6:1);
  ()
}
)"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("uN[6] vs uN[5]")));
}

TEST(TypecheckTest, LetBindingInferredDoesNotMatchAnnotation) {
  EXPECT_THAT(Typecheck(R"(
fn f() -> u32 {
  let x: u32 = bits[4]:7;
  x
}
)"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Annotated type did not match inferred type "
                                 "of right hand side")));
}

TEST(TypecheckTest, UpdateBuiltin) {
  XLS_EXPECT_OK(Typecheck(R"(
fn f() -> u32[3] {
  let x: u32[3] = u32[3]:[0, 1, 2];
  update(x, u32:1, u32:3)
}
)"));
}

TEST(TypecheckTest, SliceBuiltin) {
  XLS_EXPECT_OK(Typecheck(R"(
fn f() -> u32[3] {
  let x: u32[2] = u32[2]:[0, 1];
  slice(x, u32:0, u32[3]:[0, 0, 0])
}
)"));
}

TEST(TypecheckTest, SelectBuiltin) {
  XLS_EXPECT_OK(Typecheck(R"(
fn f(x: bool) -> u32 {
  select(x, u32:1, u32:0)
}
)"));
}

TEST(TypecheckTest, SelectBuiltinDifferentOnFalse) {
  EXPECT_THAT(Typecheck(R"(
fn f(x: bool) -> u32 {
  select(x, u32:1, u8:0)
}
)"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Want arguments 1 and 2 to 'select' to be of "
                                 "the same type; got uN[32] vs uN[8]")));
}

TEST(TypecheckTest, EnumerateBuiltin) {
  XLS_EXPECT_OK(Typecheck(R"(
type MyTup = (u32, u2);
fn f(x: u2[7]) -> MyTup[7] {
  enumerate(x)
}
)"));
}

TEST(TypecheckTest, TernaryNonBoolean) {
  EXPECT_THAT(
      Typecheck(R"(
fn f(x: u32) -> u32 {
  u32:42 if x else u32:64
}
)"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Test type for conditional expression is not \"bool\"")));
}

TEST(TypecheckTest, AddWithCarryBuiltin) {
  XLS_EXPECT_OK(Typecheck(R"(
fn f(x: u32) -> (u1, u32) {
  add_with_carry(x, x)
}
)"));
}

TEST(TypecheckTest, BitSliceUpdateBuiltIn) {
  XLS_EXPECT_OK(Typecheck(R"(
fn f(x: u32, y: u17, z: u15) -> u32 {
  bit_slice_update(x, y, z)
}
)"));
}

TEST(TypecheckTest, UpdateIncompatibleValue) {
  EXPECT_THAT(Typecheck(R"(
fn f(x: u32[5]) -> u32[5] {
  update(x, u32:1, u8:0)
}
)"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("uN[32] to match argument 2 type uN[8]")));
}

TEST(TypecheckTest, UpdateMultidimIndex) {
  EXPECT_THAT(Typecheck(R"(
fn f(x: u32[6][5], i: u32[2]) -> u32[6][5] {
  update(x, i, u32[6]:[0, ...])
}
)"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Want argument 1 to be unsigned bits")));
}

TEST(TypecheckTest, MissingAnnotation) {
  XLS_EXPECT_OK(Typecheck(R"(
fn f() -> u32 {
  let x = u32:2;
  x + x
}
)"));
}

TEST(TypecheckTest, Index) {
  XLS_EXPECT_OK(Typecheck("fn f(x: uN[32][4]) -> u32 { x[u32:0] }"));
  XLS_EXPECT_OK(Typecheck("fn f(x: u32[5], i: u8) -> u32 { x[i] }"));
  EXPECT_THAT(
      Typecheck("fn f(x: u32, i: u8) -> u32 { x[i] }"),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("not an array")));
  EXPECT_THAT(Typecheck("fn f(x: u32[5], i: u8[5]) -> u32 { x[i] }"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("not (scalar) bits")));
}

TEST(TypecheckTest, OutOfRangeNumber) {
  XLS_EXPECT_OK(Typecheck("fn f() -> u8 { u8:255 }"));
  XLS_EXPECT_OK(Typecheck("fn f() -> u8 { u8:-1 }"));
  XLS_EXPECT_OK(Typecheck("fn f() -> u8 { u8:-2 }"));
  EXPECT_THAT(
      Typecheck("fn f() -> u8 { u8:256 }"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Value '256' does not fit in the bitwidth of a uN[8]")));
}

TEST(TypecheckTest, OutOfRangeNumberInConstantArray) {
  EXPECT_THAT(
      Typecheck("fn f() -> u8[3] { u8[3]:[1, 2, 256] }"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Value '256' does not fit in the bitwidth of a uN[8]")));
}

TEST(TypecheckTest, MatchArmMismatch) {
  EXPECT_THAT(
      Typecheck("fn f(x: u8) -> u8 { match x { u8:0 => u8:3, _ => u3:3 } }"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("match arm did not have the same type")));
}

TEST(TypecheckTest, ArrayInconsistency) {
  EXPECT_THAT(Typecheck(R"(
type Foo = (u8, u32);
fn f() -> Foo {
  let xs = Foo[2]:[(u8:0, u32:1), u32:2];
  xs[u32:1]
}
)"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("vs uN[32]: Array member did not have same "
                                 "type as other members.")));
}

TEST(TypecheckTest, ArrayOfConsts) {
  XLS_EXPECT_OK(Typecheck(R"(
fn f() -> u4 {
  let a: u4 = u4:1;
  let my_array = [a];
  a
}
)"));
}

TEST(TypecheckTest, ArrayEllipsis) {
  XLS_EXPECT_OK(Typecheck("fn main() -> u8[2] { u8[2]:[0, ...] }"));
}

TEST(TypecheckTest, BadArrayAddition) {
  EXPECT_THAT(Typecheck(R"(
fn f(a: bits[32][4], b: bits[32][4]) -> bits[32][4] {
  a + b
}
)"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Binary operations can only be applied")));
}

TEST(TypecheckTest, OneHotBadPrioType) {
  EXPECT_THAT(
      Typecheck(R"(
fn f(x: u7, prio: u2) -> u8 {
  one_hot(x, prio)
}
)"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Expected argument 1 to 'one_hot' to be a u1; got uN[2]")));
}

TEST(TypecheckTest, OneHotSelOfSigned) {
  XLS_EXPECT_OK(Typecheck(R"(
fn f() -> s4 {
  let a: s4 = s4:1;
  let b: s4 = s4:2;
  let s: u2 = u2:0b01;
  one_hot_sel(s, [a, b])
}
)"));
}

TEST(TypecheckTest, OverlargeEnumValue) {
  EXPECT_THAT(
      Typecheck(R"(
enum Foo : u1 {
  A = 0,
  B = 1,
  C = 2,
}
)"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Value '2' does not fit in the bitwidth of a uN[1]")));
}

TEST(TypecheckTest, CannotAddEnums) {
  EXPECT_THAT(
      Typecheck(R"(
enum Foo : u2 {
  A = 0,
  B = 1,
}
fn f() -> Foo {
  Foo::A + Foo::B
}
)"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Cannot use '+' on values with enum type Foo")));
}

TEST(TypecheckTest, WidthSlices) {
  XLS_EXPECT_OK(Typecheck("fn f(x: u32) -> bits[0] { x[0+:bits[0]] }"));
  XLS_EXPECT_OK(Typecheck("fn f(x: u32) -> u2 { x[32+:u2] }"));
  XLS_EXPECT_OK(Typecheck("fn f(x: u32) -> u1 { x[31+:u1] }"));
}

TEST(TypecheckTest, WidthSliceNegativeStartNumber) {
  // Start literal is treated as unsigned.
  XLS_EXPECT_OK(Typecheck("fn f(x: u32) -> u1 { x[-1+:u1] }"));
  XLS_EXPECT_OK(Typecheck("fn f(x: u32) -> u2 { x[-1+:u2] }"));
  XLS_EXPECT_OK(Typecheck("fn f(x: u32) -> u3 { x[-2+:u3] }"));
}

TEST(TypecheckTest, WidthSliceSignedStart) {
  // We reject signed start literals.
  EXPECT_THAT(
      Typecheck("fn f(start: s32, x: u32) -> u3 { x[start+:u3] }"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Start index for width-based slice must be unsigned")));
}

TEST(TypecheckTest, WidthSliceTupleStart) {
  EXPECT_THAT(
      Typecheck("fn f(start: (s32), x: u32) -> u3 { x[start+:u3] }"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Start expression for width slice must be bits typed")));
}

TEST(TypecheckTest, WidthSliceTupleSubject) {
  EXPECT_THAT(Typecheck("fn f(start: s32, x: (u32)) -> u3 { x[start+:u3] }"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Value to slice is not of 'bits' type")));
}

TEST(TypecheckTest, OverlargeWidthSlice) {
  EXPECT_THAT(Typecheck("fn f(x: u32) -> u33 { x[0+:u33] }"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Slice type must have <= original number of "
                                 "bits; attempted slice from 32 to 33 bits.")));
}

TEST(TypecheckTest, BadAttributeAccessOnTuple) {
  EXPECT_THAT(Typecheck(R"(
fn main() -> () {
  let x: (u32,) = (u32:42,);
  x.a
}
)"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected a struct for attribute access")));
}

TEST(TypecheckTest, BadAttributeAccessOnBits) {
  EXPECT_THAT(Typecheck(R"(
fn main() -> () {
  let x = u32:42;
  x.a
}
)"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected a struct for attribute access")));
}

TEST(TypecheckTest, BadArrayLiteralType) {
  EXPECT_THAT(Typecheck(R"(
fn main() -> s32[2] {
  s32:[1, 2]
}
)"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Annotated type for array literal must be an "
                                 "array type; got sbits s32")));
}

TEST(TypecheckTest, CharLiteralArray) {
  XLS_EXPECT_OK(Typecheck(R"(
fn main() -> u8[3] {
  u8[3]:['X', 'L', 'S']
}
)"));
}

TEST(TypecheckTest, BadEnumRef) {
  EXPECT_THAT(
      Typecheck(R"(
enum MyEnum : u1 { A = 0, B = 1 }
fn f() -> MyEnum { MyEnum::C }
)"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Name 'C' is not defined by the enum MyEnum")));
}

TEST(TypecheckTest, NominalTyping) {
  // Nominal typing not structural, e.g. OtherPoint cannot be passed where we
  // want a Point, even though their members are the same.
  EXPECT_THAT(Typecheck(R"(
struct Point { x: s8, y: u32 }
struct OtherPoint { x: s8, y: u32 }
fn f(x: Point) -> Point { x }
fn g() -> Point {
  let shp = OtherPoint { x: s8:255, y: u32:1024 };
  f(shp)
}
)"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("parameter type name: 'Point'; argument type "
                                 "name: 'OtherPoint'")));
}

TEST(TypecheckTest, ParametricWithConstantArrayEllipsis) {
  XLS_EXPECT_OK(Typecheck(R"(
fn p<N: u32>(_: bits[N]) -> u8[2] { u8[2]:[0, ...] }
fn main() -> u8[2] { p(false) }
)"));
}

TEST(TypecheckTest, BadQuickcheckFunctionRet) {
  EXPECT_THAT(Typecheck(R"(
#![quickcheck]
fn f() -> u5 { u5:1 }
)"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("must return a bool")));
}

TEST(TypecheckTest, BadQuickcheckFunctionParametrics) {
  EXPECT_THAT(
      Typecheck(R"(
#![quickcheck]
fn f<N: u32>() -> bool { true }
)"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Quickchecking parametric functions is unsupported")));
}

// Helper for struct instance based tests.
absl::Status TypecheckStructInstance(std::string program) {
  program = R"(
struct Point {
  x: s8,
  y: u32,
}
)" + program;
  return Typecheck(program);
}

TEST(TypecheckStructInstanceTest, AccessMissingMember) {
  EXPECT_THAT(
      TypecheckStructInstance("fn f(p: Point) -> () { p.z }"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Struct 'Point' does not have a member with name 'z'")));
}

TEST(TypecheckStructInstanceTest, WrongType) {
  EXPECT_THAT(TypecheckStructInstance(
                  "fn f() -> Point { Point { y: u8:42, x: s8:255 } }"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("uN[32] vs uN[8]")));
}

TEST(TypecheckStructInstanceTest, MissingFieldX) {
  EXPECT_THAT(
      TypecheckStructInstance("fn f() -> Point { Point { y: u32:42 } }"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Struct instance is missing member(s): 'x'")));
}

TEST(TypecheckStructInstanceTest, MissingFieldY) {
  EXPECT_THAT(
      TypecheckStructInstance("fn f() -> Point { Point { x: s8: 255 } }"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Struct instance is missing member(s): 'y'")));
}

TEST(TypecheckStructInstanceTest, OutOfOrderOk) {
  XLS_EXPECT_OK(TypecheckStructInstance(
      "fn f() -> Point { Point { y: u32:42, x: s8:255 } }"));
}

TEST(TypecheckStructInstanceTest, ProvideExtraFieldZ) {
  EXPECT_THAT(
      TypecheckStructInstance(
          "fn f() -> Point { Point { x: s8:255, y: u32:42, z: u32:1024 } }"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Struct \'Point\' has no member \'z\', but it was "
                         "provided by this instance.")));
}

TEST(TypecheckStructInstanceTest, DuplicateFieldY) {
  EXPECT_THAT(
      TypecheckStructInstance(
          "fn f() -> Point { Point { x: s8:255, y: u32:42, y: u32:1024 } }"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Duplicate value seen for \'y\' in this \'Point\' "
                         "struct instance.")));
}

TEST(TypecheckStructInstanceTest, StructIncompatibleWithTupleEquivalent) {
  EXPECT_THAT(
      TypecheckStructInstance(R"(
fn f(x: (s8, u32)) -> (s8, u32) { x }
fn g() -> (s8, u32) {
  let p = Point { x: s8:255, y: u32:1024 };
  f(p)
}
)"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("argument types are different kinds (tuple vs struct)")));
}

TEST(TypecheckStructInstanceTest, SplatWithDuplicate) {
  EXPECT_THAT(
      TypecheckStructInstance(
          "fn f(p: Point) -> Point { Point { x: s8:42, x: s8:64, ..p } }"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Duplicate value seen for \'x\' in this \'Point\' "
                         "struct instance.")));
}

TEST(TypecheckStructInstanceTest, SplatWithExtraFieldQ) {
  EXPECT_THAT(TypecheckStructInstance(
                  "fn f(p: Point) -> Point { Point { q: u32:42, ..p } }"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Struct 'Point' has no member 'q'")));
}

// Helper for parametric struct instance based tests.
absl::Status TypecheckParametricStructInstance(std::string program) {
  program = R"(
struct Point<N: u32, M: u32 = N + N> {
  x: bits[N],
  y: bits[M],
}
)" + program;
  return Typecheck(program);
}

TEST(TypecheckParametricStructInstanceTest, WrongDerivedType) {
  EXPECT_THAT(TypecheckParametricStructInstance(
                  "fn f() -> Point<32, 63> { Point { x: u32:5, y: u63:255 } }"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("uN[64] vs uN[63]")));
}

TEST(TypecheckParametricStructInstanceTest, TooManyParametricArgs) {
  EXPECT_THAT(
      TypecheckParametricStructInstance(
          "fn f() -> Point<u32:5, u32:10, u32:15> { Point { x: u5:5, y: "
          "u10:255 } }"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Expected 2 parametric arguments for 'Point'; got 3")));
}

TEST(TypecheckParametricStructInstanceTest, OutOfOrderOk) {
  XLS_EXPECT_OK(TypecheckParametricStructInstance(
      "fn f() -> Point<32, 64> { Point { y: u64:42, x: u32:255 } }"));
}

TEST(TypecheckParametricStructInstanceTest,
     OkInstantiationInParametricFunction) {
  XLS_EXPECT_OK(TypecheckParametricStructInstance(R"(
fn f<A: u32, B: u32>(x: bits[A], y: bits[B]) -> Point<A, B> { Point { x, y } }
fn main() {
  let _ = f(u5:1, u10:2);
  let _ = f(u14:1, u28:2);
  ()
}
)"));
}

TEST(TypecheckParametricStructInstanceTest, BadReturnType) {
  EXPECT_THAT(
      TypecheckParametricStructInstance(
          "fn f() -> Point<5, 10> { Point { x: u32:5, y: u64:255 } }"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "struct 'Point' structure: Point { x: uN[32], y: uN[64] } vs "
              "struct 'Point' structure: Point { x: uN[5], y: uN[10] }")));
}

// Bad struct type-parametric instantiation in parametric function.
TEST(TypecheckParametricStructInstanceTest, BadParametricInstantiation) {
  EXPECT_THAT(TypecheckParametricStructInstance(R"(
fn f<A: u32, B: u32>(x: bits[A], y: bits[B]) -> Point<A, B> {
  Point { x, y }
}

fn main() {
  let _ = f(u5:1, u10:2);
  let _ = f(u14:1, u15:2);
  ()
}
)"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("uN[28] vs uN[15]")));
}

TEST(TypecheckParametricStructInstanceTest, BadParametricSplatInstantiation) {
  EXPECT_THAT(TypecheckParametricStructInstance(R"(
fn f<A: u32, B: u32>(x: bits[A], y: bits[B]) -> Point<A, B> {
  let p = Point { x, y };
  Point { x: (x++x), ..p }
}

fn main() {
  let _ = f(u5:1, u10:2);
  ()
}
)"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("uN[20] vs uN[10]")));
}

}  // namespace
}  // namespace xls::dslx
