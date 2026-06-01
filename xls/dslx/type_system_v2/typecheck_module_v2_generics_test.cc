// Copyright 2024 The XLS Authors
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

#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system/typecheck_test_utils.h"
#include "xls/dslx/type_system_v2/matchers.h"
#include "xls/dslx/type_system_v2/type_system_test_utils.h"

namespace xls::dslx {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::AllOf;
using ::testing::HasSubstr;

TEST(TypecheckV2GenericsTest, ParentGenericParametric) {
  EXPECT_THAT(
      R"(
#![feature(generics)]

fn call<FnType: type>() -> FnType {
    zero!<FnType>()
}

fn main<T: type>() -> T {
    call<T>()
}

const ONE:u16 = main<u16>();
const TWO:u24 = main<u24>();
const_assert!(ONE == u16:0);
const_assert!(TWO == u24:0);
)",
      TypecheckSucceeds(AllOf((HasNodeWithType("ONE", "uN[16]"),
                               HasNodeWithType("TWO", "uN[24]")))));
}

TEST(TypecheckV2GenericsTest, StructImplUsesLocalType) {
  EXPECT_THAT(
      R"(
#![feature(generics)]

struct lm<LambdaType: type> {}

impl lm<LambdaType> {
    fn call(self, i: u32) -> LambdaType {
        zero!<LambdaType>()
    }
}

fn main<T: type>() -> T[5] {
    type LocalType = T;
    map(u32:0..5, lm<LocalType>{}.call)
}

const ONE = main<u16>();
const TWO = main<u24>();
const_assert!(ONE == [u16:0, 0, 0, 0, 0]);
const_assert!(TWO == [u24:0, 0, 0, 0, 0]);
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("ONE", "uN[16][5]"),
                              HasNodeWithType("TWO", "uN[24][5]"))));
}

TEST(TypecheckV2GenericsTest, StructImplUsesGenericTypeParametric) {
  EXPECT_THAT(
      R"(
#![feature(generics)]

struct lm<LambdaType: type> {}

impl lm<LambdaType> {
    fn call(self) -> LambdaType {
        zero!<LambdaType>()
    }
}

const ONE = lm<u16>{}.call();
const TWO = lm<u24>{}.call();
const_assert!(ONE == 0);
const_assert!(TWO == 0);
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("ONE", "uN[16]"),
                              HasNodeWithType("TWO", "uN[24]"))));
}

TEST(TypecheckV2GenericsTest, TypeValueParametricMismatch) {
  EXPECT_THAT(
      R"(
#![feature(generics)]

fn fake_decode<T: type>(x: u32) -> u32 { x }

const Y = fake_decode<u32:32>(u32:1);
)",
      TypecheckFails(HasSubstr("Expected parametric type, saw `u32:32`")));
}

TEST(TypecheckV2GenericsTest, GenericTypePassthroughFunction) {
  EXPECT_THAT(
      R"(
#![feature(generics)]

struct S { a: u32 }

fn foo<T: type>(a: T) -> T { a }

type MyInt = u34;

enum MyEnum { A = 0 }

const C = foo<u32>(5);
const D = foo<u16[3]>([1, 2, 3]);
const E = foo((s8:5, s8:6));
const F = foo<S>(S { a: 5 });
const G = foo<MyInt>(10);
const H = foo(MyEnum::A);
)",
      TypecheckSucceeds(AllOf(
          HasNodeWithType("C", "uN[32]"), HasNodeWithType("D", "uN[16][3]"),
          HasNodeWithType("E", "(sN[8], sN[8])"),
          HasNodeWithType("F", "S { a: uN[32] }"),
          HasNodeWithType("G", "uN[34]"), HasNodeWithType("H", "MyEnum"))));
}

TEST(TypecheckV2GenericsTest, SquareAnythingFunction) {
  XLS_ASSERT_OK(TypecheckV2(
      R"(
#![feature(generics)]

fn square<T: type>(a: T) -> T { a * a }

const_assert!(square(u32:5) == 25);
const_assert!(square<s64>(-12) == 144);
)"));
}

TEST(TypecheckV2GenericsTest, Add5ToAnythingFunction) {
  XLS_ASSERT_OK(TypecheckV2(
      R"(
#![feature(generics)]

fn add5<T: type>(a: T) -> T { a + 5 }

const_assert!(add5<u32>(5) == u32:10);
const_assert!(add5(s64:-12) == s64:-7);
)"));
}

TEST(TypecheckV2GenericsTest, GenericFunctionWithIllegalInstantiationFails) {
  EXPECT_THAT(R"(
#![feature(generics)]

fn foo<T: type>(a: T) -> T { a + 5 }
const C = foo<u32[2]>([1, 2]);
)",
              TypecheckFails(HasTypeMismatch("uN[32][2]", "uN[3]")));
}

TEST(TypecheckV2GenericsTest, ZeroWrapper) {
  XLS_ASSERT_OK(TypecheckV2(
      R"(
#![feature(generics)]

struct S { a: u32 }

fn foo<T: type>() -> T { zero!<T>() }

const_assert!(foo<u32>() == 0);
const_assert!(foo<s16>() == 0);
const_assert!(foo<u32[3]>() == [u32:0, ...]);
)"));
}

TEST(TypecheckV2GenericsTest, OptionalStruct) {
  XLS_ASSERT_OK(TypecheckV2(
      R"(
#![feature(generics)]

struct Optional<T: type> {
  has_value: bool,
  value: T
}

fn make_optional<T: type>(value: T) -> Optional<T> {
  Optional<T> { has_value: true, value: value }
}

fn make_nullopt<T: type>() -> Optional<T> {
  zero!<Optional<T>>()
}

const C = make_optional(u32:5);
const D = make_nullopt<u32>();
const E = make_optional<u32[3]>([1, 2, 3]);
const F = make_optional(make_optional(u32:5));

const_assert!(C.has_value);
const_assert!(C.value == 5);
const_assert!(!D.has_value);
const_assert!(E.has_value);
const_assert!(E.value == [u32:1, 2, 3]);
const_assert!(F.value.has_value);
const_assert!(F.value.value == 5);
)"));
}

TEST(TypecheckV2GenericsTest, CallFooOnAnything) {
  XLS_ASSERT_OK(TypecheckV2(
      R"(
#![feature(generics)]

struct U32Wrapper { a: u32 }
struct S64Wrapper { a: s64 }

impl U32Wrapper {
  fn foo(self) -> u32 { self.a + 5 }
}

impl S64Wrapper {
  fn foo(self) -> s64 { self.a * self.a }
}

fn call_foo<R: type, T: type>(value: T) -> R {
  value.foo()
}

const C = call_foo<u32>(U32Wrapper { a: 10 });
const D = call_foo<s64>(S64Wrapper { a: -2 });

const_assert!(C == 15);
const_assert!(D == 4);
)"));
}

TEST(TypecheckV2GenericsTest, ParametricImplFnCallsParametricImplFn) {
  EXPECT_THAT(
      R"(
#![feature(generics)]

struct inner<UType: type> {
  inner_var: UType
}

impl inner<UType> {
  fn call(self) -> UType {
    self.inner_var
  }
}

struct outer<N: u32> {
  outer_var: u32[N]
}

impl outer<N> {
  fn call(self) -> u32 {
    let i_var = self.outer_var[0];
    inner{inner_var: i_var}.call()
  }
}

fn nested(arr: u32[5]) -> u32 {
  outer{outer_var: arr}.call()
}

const RES = nested(u32:0..5);
const_assert!(RES == 0);
)",
      TypecheckSucceeds(
          AllOf(HasNodeWithType("RES", "uN[32]"),
                HasNodeWithType("outer<5>",
                                "typeof(outer { outer_var: uN[32][5] }"))));
}

TEST(TypecheckV2GenericsTest,
     ResolveAnnotationFromSeparateImplInvocationWithOtherStruct) {
  EXPECT_THAT(
      R"(
#![feature(generics)]

struct S2<T: type> {
  val: T,
}

impl S2<T> {
  fn call<U: type>(self, other: S2<U>) -> U {
    other.val
  }
}

fn structs<N: u32>() -> s32[N+1] {
  let my_val = map(0..N, |i| i as s32);
  let plus_one = map(0..N+1, |i| (i + 1) as s32);
  S2{val: my_val}.call(S2{val: plus_one})
}

const RES = structs<4>();
const_assert!(RES == [s32:1, 2, 3, 4, 5]);
)",
      TypecheckSucceeds(HasNodeWithType("RES", "sN[32][5]")));
}

TEST(TypecheckV2GenericsTest, ResolveAnnotationFromSeparateImplInvocation) {
  EXPECT_THAT(
      R"(
#![feature(generics)]

fn is_odd(i: u32) -> bool {
  i % 2 == 1
}

struct S2<odd_map_type: type> {
  odd_map: odd_map_type,
}

impl S2 {
  fn call(self, i: u32) -> u32 {
    if self.odd_map[i] {
      i + 2
    } else {
      i
    }
  }
}

fn add_two<N: u32>() -> u32 {
  let odd_map = map(0..N, is_odd);
  S2{odd_map: odd_map}.call(N - 1)
}

const RES = add_two<5>();
const_assert!(RES == u32:4);
)",
      TypecheckSucceeds(HasNodeWithType("RES", "uN[32]")));
}

TEST(TypecheckV2GenericsTest, GenericConstantAccess) {
  EXPECT_THAT(
      R"(
#![feature(generics)]

struct S1 {}

impl S1 {
  const SZ = u32:32;
  const C = u32:5;
}

struct S2 {}

impl S2 {
  const SZ = u32:64;
  const C = u64:6;
}

fn foo<T: type>() -> uN[T::SZ] {
  T::C
}

const S1_RESULT = foo<S1>();
const S2_RESULT = foo<S2>();

const_assert!(S1_RESULT == 5);
const_assert!(S2_RESULT == 6);
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("S1_RESULT", "uN[32]"),
                              HasNodeWithType("S2_RESULT", "uN[64]"))));
}

TEST(TypecheckV2GenericsTest, GenericEnumAccess) {
  EXPECT_THAT(
      R"(
#![feature(generics)]

enum E1 : u32 {
  FOO = 5
}

enum E2 : u32 {
  FOO = 6
}

fn add_foo<E: type>(a: u32) -> u32 {
  a + E::FOO as u32
}

const C = add_foo<E1>(10);
const D = add_foo<E2>(20);
const_assert!(C == 15);
const_assert!(D == 26);
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("C", "uN[32]"),
                              HasNodeWithType("D", "uN[32]"))));
}

TEST(TypecheckV2GenericsTest, GenericStructTypeExplicit) {
  XLS_ASSERT_OK(TypecheckV2(
      R"(
#![feature(generics)]

struct Anything<T: type> {
  value: T
}

const C = Anything<u32>{value: u32:5};
const D = Anything<u3>{value: u3:5};

const_assert!(C.value == 5);
const_assert!(D.value == 5);

)"));
}

TEST(TypecheckV2GenericsTest, GenericStructTypeExplicitTypeMismatch) {
  EXPECT_THAT(
      R"(
#![feature(generics)]

struct Anything<T: type> {
  value: T
}

const C = Anything<u16>{value: u32:5};

)",
      TypecheckFails(HasTypeMismatch("u32", "u16")));
}

TEST(TypecheckV2GenericsTest, GenericStructTypeInferred) {
  EXPECT_THAT(
      R"(
#![feature(generics)]

struct Anything<T: type> {
  value: T
}

const C = Anything{value: u32:5};
const D = Anything{value: u3:5};

const_assert!(C.value == 5);
const_assert!(D.value == 5);

)",
      TypecheckSucceeds(
          AllOf(HasNodeWithType("C", "Anything { value: uN[32] }"),
                HasNodeWithType("D", "Anything { value: uN[3] }"))));
}

TEST(TypecheckV2GenericsTest, GenericStructAndImpl) {
  EXPECT_THAT(
      R"(
#![feature(generics)]

struct Anything<T: type> {
  value: T
}

impl Anything {
  fn foo<S: type>(self, x: S) -> S {
    x + self.value
  }
}

const X = u32:5;
const X_STRUCT = Anything { value: X };
const Y = X_STRUCT.foo(u32:10);

const_assert!(Y == 15);
)",
      TypecheckSucceeds(HasNodeWithType("Y", "uN[32]")));
}

TEST(TypecheckV2GenericsTest, GenericStructAndImplWithMap) {
  EXPECT_THAT(
      R"(
#![feature(generics)]

struct Anything<T: type> {
  value: T
}

impl Anything {
  fn foo<U: type>(self, x: U) -> u32 {
    x + self.value
  }
}

fn main() -> u32 {
  let u = u32:4..10;
  let x = u32:10;
  let arr = map(u, Anything { value: x }.foo);
  arr[1]
}

const_assert!(main() == 15);
)",
      TypecheckSucceeds(HasNodeWithType("main", "() -> uN[32]")));
}

TEST(TypecheckV2GenericsTest, GenericStructAndImplWithMapNonParametricFn) {
  EXPECT_THAT(
      R"(
#![feature(generics)]

struct Anything<T: type> {
  value: T
}

impl Anything<T> {
  fn foo(self, x: u32) -> u32 {
    x + self.value
  }
}

fn main() -> u32 {
  let u = u32:4..10;
  let x = u32:10;
  let arr = map(u, Anything { value: x }.foo);
  arr[2]
}

const_assert!(main() == 16);
)",
      TypecheckSucceeds(HasNodeWithType("main", "() -> uN[32]")));
}

TEST(TypecheckV2GenericsTest, GenericStructAndImplTypeMismatch) {
  EXPECT_THAT(
      R"(
#![feature(generics)]

struct Anything<T: type> {
  value: T
}

impl Anything {
  fn foo<S: type>(self, x: S) -> S {
    x + self.value
  }
}

const X = u32:5;
const X_STRUCT = Anything { value: X };
const Y = X_STRUCT.foo(u16:10);
)",
      TypecheckFails(HasTypeMismatch("uN[32]", "uN[16]")));
}

TEST(TypecheckV2GenericsTest, GenericColonRefFunctionCall) {
  EXPECT_THAT(
      R"(
#![feature(generics)]

struct U32Adder {}

impl U32Adder {
    fn add(a: u32, b: u32) -> u32 { a + b }
}

struct Foo {
    value: u16
}

struct FooAdder {}

impl FooAdder {
    fn add(a: Foo, b: Foo) -> Foo { Foo { value: a.value + b.value } }
}

fn add_wrapper<ADDER: type, T: type>(a: T, b: T) -> T {
    ADDER::add(a, b)
}

const A = add_wrapper<U32Adder>(u32:1, u32:2);
const B = add_wrapper<FooAdder>(Foo { value: 3 }, Foo { value: 4 });
const_assert!(A == 3);
const_assert!(B == Foo { value: 7 });
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("A", "uN[32]"),
                              HasNodeWithType("B", "Foo { value: uN[16] }"))));
}

TEST(TypecheckV2GenericsTest, GenericColonRefCallToNonexistentFunction) {
  EXPECT_THAT(
      R"(
#![feature(generics)]

struct U32Adder {}

impl U32Adder {
    fn add(a: u32, b: u32) -> u32 { a + b }
}

fn sub_wrapper<ALG: type, T: type>(a: T, b: T) -> T {
    ALG::sub(a, b)
}

const A = sub_wrapper<U32Adder>(u32:1, u32:2);
)",
      TypecheckFails(HasSubstr(
          "Name 'sub' is not defined by the impl for struct 'U32Adder'")));
}

TEST(TypecheckV2GenericsTest, GenericColonRefCallWithArgTypeMismatch) {
  EXPECT_THAT(
      R"(
#![feature(generics)]

struct U32Adder {}

impl U32Adder {
    fn add(a: u32, b: u32) -> u32 { a + b }
}

fn add_wrapper<ADDER: type, T: type>(a: T, b: T) -> T {
    ADDER::add(a, b)
}

const A = add_wrapper<U32Adder>(u16:1, 2);
)",
      TypecheckFails(HasTypeMismatch("uN[16]", "uN[32]")));
}

TEST(TypecheckV2Test, ComparisonAsParametricArgument) {
  EXPECT_THAT(R"(
fn foo<S: bool>(a: xN[S][32]) -> xN[S][32] { a }
const Y = foo<{2 > 1}>(s32:5);
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("Y", "sN[32]"),
                                      HasNodeWithType("2", "uN[2]"),
                                      HasNodeWithType("1", "uN[2]"))));
}

TEST(TypecheckV2Test, ComparisonAsParametricArgumentWithConflictFails) {
  EXPECT_THAT(R"(
fn foo<S: bool>(a: xN[S][32]) -> xN[S][32] { a }
const Y = foo<{2 > 1}>(u32:5);
)",
              TypecheckFails(HasSignednessMismatch("xN[1][32]", "u32")));
}

TEST(TypecheckV2Test, ComparisonAndSumAsParametricArguments) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
const X = u32:1;
fn foo<S: bool, N: u32>(a: xN[S][N]) -> xN[S][N] { a }
const Y = foo<{X == 1}, {X + 3}>(s4:3);
)"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(type_info_string, HasSubstr("node: `Y`, type: sN[4]"));
}

TEST(TypecheckV2Test, ComparisonAndSumParametricArgumentsWithConflictFails) {
  EXPECT_THAT(R"(
const X = u32:1;
fn foo<S: bool, N: u32>(a: xN[S][N]) -> xN[S][N] { a }
const Y = foo<{X == 1}, {X + 4}>(s4:3);
)",
              TypecheckFails(HasSizeMismatch("xN[1][5]", "s4")));
}

TEST(TypecheckV2Test, InferenceOfParametricUsingAnnotationWithInvocation) {
  // This is based on a similar pattern that occurs in std.x, with the inference
  // of the implicit parametrics for `assert_eq` being the potential trouble
  // area.
  EXPECT_THAT(
      R"(
fn sizeof<S: bool, N: u32>(x: xN[S][N]) -> u32 { N }
fn foo() {
  let x = uN[32]:0xffffffff;
  let y: uN[sizeof(x) + u32:2] = x as uN[sizeof(x) + u32:2];
  assert_eq(y, uN[34]:0xffffffff);
}
)",
      TypecheckSucceeds(HasNodeWithType("y", "uN[34]")));
}

TEST(TypecheckV2Test, MultiTermSumOfParametricCalls) {
  EXPECT_THAT(
      R"(
fn foo<N: u32>(a: uN[N]) -> u32 { a as u32 }
// Note: the point is to make it obvious in manual runs if there is exponential
// growth in typechecking an expr like this. That was the case with the original
// rev of `HandleInvocation`.
const Y = foo(u32:1) + foo(u32:2) + foo(u32:3) + foo(u32:4) + foo(u32:5) +
    foo(u32:6) + foo(u32:7);
)",
      TypecheckSucceeds(HasNodeWithType("Y", "uN[32]")));
}

TEST(TypecheckV2Test, ParametricDefaultWithTypeBasedOnOtherParametric) {
  EXPECT_THAT(R"(
fn p<X: u32, Y: bits[X] = {u1:0}>(x: bits[X]) -> bits[X] { x }
const X = p(u1:0);
)",
              TypecheckSucceeds(HasNodeWithType("X", "uN[1]")));
}

TEST(TypecheckV2Test, UnassignedReturnValueIgnoredParametric) {
  EXPECT_THAT(
      R"(
fn ignored<N:u32>() -> uN[N] { zero!<uN[N]>() }

fn main() -> u32 {
  ignored<u32:31>();
  u32:1
}
)",
      TypecheckSucceeds(HasNodeWithType("ignored<u32:31>()", "uN[31]")));
}

TEST(TypecheckV2Test, ParametricLet) {
  EXPECT_THAT(R"(
fn f<N: u32>() -> uN[N] {
  let x = uN[N]:0;
  x
}

const X = f<4>();
const Y = f<16>();
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("X", "uN[4]"),
                                      HasNodeWithType("Y", "uN[16]"))));
}

TEST(TypecheckV2Test, ParametricContextStackingViaDefault) {
  EXPECT_THAT(
      R"(
fn g<A: u32>(x: uN[A]) -> u32 { 32 }
fn f<A: u32, B: u32 = {g(A)}>(a: uN[A])-> uN[B] { a }
const X = f(u32:5);
)",
      TypecheckSucceeds(HasNodeWithType("X", "uN[32]")));
}

TEST(TypecheckV2Test, ParametricValuesDefinedMultipleTimesInTypeAlias) {
  EXPECT_THAT(R"(
struct S<X: u32, Y: u32 = {X * 2}> {
  x: bits[X],
  y: bits[Y],
}
type MyS = S<3>;
type MySDouble = MyS<4>;
fn f() -> uN[4] {
  let x = MySDouble { x: 3, y: 4 };
  x.y
}
)",
              TypecheckFails(HasSubstr("Parametric values defined multiple "
                                       "times for annotation: `S<3>`")));
}

TEST(TypecheckV2Test, ParametricValuesNeverDefinedInTypeAlias) {
  EXPECT_THAT(
      R"(
struct S<X: u32, Y: u32 = {X * 2}> {
  x: bits[X],
  y: bits[Y],
}
type MyS = S;
fn f() -> MyS {
  MyS { x: 3, y: 4 };
}
)",
      TypecheckFails(HasSubstr(
          "Could not infer parametric(s) for instance of struct S: X")));
}

TEST(TypecheckV2Test, ImportedConstantSizeAsParametricValue) {
  constexpr std::string_view kImported = R"(
pub const SOME_CONSTANT = u32:8;
)";
  constexpr std::string_view kProgram = R"(
import imported;

fn foo<N: u32>(a: uN[N]) -> uN[N] { a }

const X = foo(uN[imported::SOME_CONSTANT]:0);
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data).status());
  EXPECT_THAT(TypecheckV2(kProgram, "main", &import_data),
              IsOkAndHolds(HasTypeInfo(HasNodeWithType("X", "uN[8]"))));
}

TEST(TypecheckV2Test, ImportTypeAliasWithParametrics) {
  constexpr std::string_view kImported = R"(
pub struct S<N: u32> {
 a: uN[N]
}

pub type S32 = S<32>;
)";
  constexpr std::string_view kProgram = R"(
import imported;

fn get_a(s: imported::S32) -> u32 {
  s.a
}
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data));
  XLS_EXPECT_OK(TypecheckV2(kProgram, "main", &import_data));
}

TEST(TypecheckV2Test, RangeAsArgumentParametric) {
  EXPECT_THAT(
      R"(
pub fn pass_back<N: u32>(input: u32[N]) -> u32[N] {
    input
}

fn test() {
    pass_back(0..4);
}

)",
      TypecheckSucceeds(
          AllOf(HasNodeWithType("test", "() -> ()"),
                HasNodeWithType("pass_back(0..4)", "uN[32][4]"))));
}

TEST(TypecheckV2Test, InferParametricWithRange) {
  EXPECT_THAT(R"(
pub fn infer_parametric<N: u32, M: u32>(true_indices: u32[M]) -> bool[N] {
    for (i, x): (u32, bool[N]) in true_indices {
        update(x, i, true)
    }(zero!<bool[N]>())
}

fn test() {
    infer_parametric<4>([0, 1, 2, 3]);
    let a = 0..4;
    infer_parametric<4>(a);
}

)",
              TypecheckSucceeds(
                  AllOf(HasNodeWithType("test", "() -> ()"),
                        HasNodeWithType("infer_parametric<4>(a)", "uN[1][4]"),
                        HasNodeWithType("a", "uN[32][4]"))));
}

TEST(TypecheckV2Test, UnusedDefinitionParametrics) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
fn f<A: u32>() {
  let a: uN[A] = 0;
}

fn g() {
  f<u32:2>();
}
)"));
  ASSERT_THAT(result.tm.warnings.warnings().size(), 1);
  EXPECT_EQ(result.tm.warnings.warnings()[0].message,
            "Definition of `a` (type `uN[2]`) is not used in function `f`");
}

TEST(TypecheckV2Test, ParametricTypeRolloverOk) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
fn p<N: u32, M: u32>() -> u32 {
  uN[N - M + u32:2]:1 as u32
}

fn main() -> u32 {
  p<u32:1, u32:2>()
}
)"));
  EXPECT_EQ(result.tm.warnings.warnings().size(), 0);
}

TEST(TypecheckV2Test, FuzzTestParametric) {
  EXPECT_THAT(
      R"(
#[fuzz_test(domains=`u32:0..1`)]
fn f<N: u32>(x: uN[N]) {
   x+uN[N]:1
}
)",
      TypecheckFails(HasSubstr("Cannot fuzz test parametric function `f`")));
}

TEST(TypecheckV2Test, XnAnnotationWithNonBoolLiteralSignednessFails) {
  EXPECT_THAT("const Y = xN[2][32]:1;",
              TypecheckFails(HasSizeMismatch("bool", "u2")));
}

TEST(TypecheckV2Test, XnAnnotationWithNonBoolConstantSignednessFails) {
  EXPECT_THAT(R"(
const X = u32:2;
const Y = xN[X][32]:1;
)",
              TypecheckFails(HasSizeMismatch("bool", "u32")));
}

TEST(TypecheckV2Test, ConcatOfBitsAsImplicitParametric) {
  EXPECT_THAT(R"(
fn f<A: u32>(a: uN[A]) -> uN[A] { a }
const X = f(u16:0 ++ u32:0);
)",
              TypecheckSucceeds(HasNodeWithType("X", "uN[48]")));
}

TEST(TypecheckV2Test, ConcatOfArrayAsImplicitParametric) {
  EXPECT_THAT(R"(
fn f<A: u32>(a: u16[A]) -> u16[A] { a }
const X = f([u16:0, 1, 2] ++ [u16:20]);
)",
              TypecheckSucceeds(HasNodeWithType("X", "uN[16][4]")));
}

TEST(TypecheckV2Test,
     ParametricFunctionTakingIntegerOfImplicitParameterizedSize) {
  EXPECT_THAT(R"(
fn foo<N: u32>(a: uN[N]) -> uN[N] { a }
const X = foo(u10:5);
const Y = foo(u11:5);
)",
              TypecheckSucceeds(
                  AllOf(HasNodeWithType("const X = foo(u10:5);", "uN[10]"),
                        HasNodeWithType("const Y = foo(u11:5);", "uN[11]"))));
}

TEST(TypecheckV2Test,
     ParametricFunctionTakingIntegerOfImplicitParameterizedSignedness) {
  EXPECT_THAT(R"(
fn foo<S: bool>(a: xN[S][32]) -> xN[S][32] { a }
const X = foo(u32:5);
const Y = foo(s32:5);
)",
              TypecheckSucceeds(
                  AllOf(HasNodeWithType("const X = foo(u32:5);", "uN[32]"),
                        HasNodeWithType("const Y = foo(s32:5);", "sN[32]"))));
}

TEST(TypecheckV2Test,
     ParametricFunctionTakingIntegerOfImplicitParameterizedSignednessAndSize) {
  EXPECT_THAT(R"(
fn foo<S: bool, N: u32>(a: xN[S][N]) -> xN[S][N] { a }
const X = foo(u10:5);
const Y = foo(s11:5);
)",
              TypecheckSucceeds(
                  AllOf(HasNodeWithType("const X = foo(u10:5);", "uN[10]"),
                        HasNodeWithType("const Y = foo(s11:5);", "sN[11]"))));
}

TEST(TypecheckV2Test, ParametricFunctionWithDefaultImplicitlyOverriddenFails) {
  EXPECT_THAT(R"(
fn foo<M: u32, N: u32 = {M + 1}>(a: uN[N]) -> uN[N] { a }
const X = foo<11>(u20:5);
)",
              TypecheckFails(HasSizeMismatch("u20", "uN[12]")));
}

TEST(TypecheckV2Test,
     ParametricFunctionTakingIntegerOfImplicitSignednessAndSizeWithSum) {
  EXPECT_THAT(R"(
const X = u32:3;
const Y = u32:4;
fn foo<N: u32>(a: uN[N]) -> uN[N] { a }
const Z = foo(X + Y + X + 50);
)",
              TypecheckSucceeds(
                  HasNodeWithType("const Z = foo(X + Y + X + 50);", "uN[32]")));
}

TEST(TypecheckV2Test, ParametricFunctionTakingArrayOfImplicitSize) {
  EXPECT_THAT(
      R"(
fn foo<N: u32>(a: u32[N]) -> u32[N] { a }
const X = foo([1, 2, 3]);
const Y = foo([4, 5, 6, 7]);
)",
      TypecheckSucceeds(
          AllOf(HasNodeWithType("const X = foo([1, 2, 3]);", "uN[32][3]"),
                HasNodeWithType("const Y = foo([4, 5, 6, 7]);", "uN[32][4]"),
                HasNodeWithType("1", "uN[32]"), HasNodeWithType("2", "uN[32]"),
                HasNodeWithType("3", "uN[32]"), HasNodeWithType("4", "uN[32]"),
                HasNodeWithType("5", "uN[32]"), HasNodeWithType("6", "uN[32]"),
                HasNodeWithType("7", "uN[32]"))));
}

TEST(TypecheckV2Test, ParametricFunctionImplicitParameterPropagation) {
  EXPECT_THAT(R"(
fn bar<A: u32, B: u32>(a: uN[A], b: uN[B]) -> uN[A] { a + 1 }
fn foo<A: u32, B: u32>(a: uN[A], b: uN[B]) -> uN[B] { bar(b, a) }
const X = foo(u23:4, u17:5);
)",
              TypecheckSucceeds(
                  HasNodeWithType("const X = foo(u23:4, u17:5);", "uN[17]")));
}

TEST(TypecheckV2Test, ParametricFunctionImplicitParameterExplicitPropagation) {
  EXPECT_THAT(R"(
fn bar<A: u32, B: u32>(a: uN[A], b: uN[B]) -> uN[A] { a + 1 }
fn foo<A: u32, B: u32>(a: uN[A], b: uN[B]) -> uN[B] { bar<B, A>(b, a) }
const X = foo(u23:4, u17:5);
)",
              TypecheckSucceeds(
                  HasNodeWithType("const X = foo(u23:4, u17:5);", "uN[17]")));
}

TEST(TypecheckV2Test, ParametricFunctionImplicitInvocationNesting) {
  EXPECT_THAT(R"(
fn foo<N: u32>(a: uN[N]) -> uN[N] { a + 1 }
const X = foo(foo(u24:4) + foo(u24:5));
)",
              TypecheckSucceeds(HasNodeWithType(
                  "const X = foo(foo(u24:4) + foo(u24:5));", "uN[24]")));
}

TEST(TypecheckV2Test,
     ParametricFunctionImplicitInvocationNestingWithExplicitOuter) {
  EXPECT_THAT(R"(
fn foo<N: u32>(a: uN[N]) -> uN[N] { a + 1 }
const X = foo<24>(foo(u24:4 + foo(u24:6)) + foo(u24:5));
)",
              TypecheckSucceeds(HasNodeWithType(
                  "const X = foo<24>(foo(u24:4 + foo(u24:6)) + foo(u24:5));",
                  "uN[24]")));
}

TEST(TypecheckV2Test,
     ParametricFunctionImplicitInvocationNestingWithExplicitInner) {
  EXPECT_THAT(R"(
fn foo<N: u32>(a: uN[N]) -> uN[N] { a + 1 }
const X = foo(foo<24>(4) + foo<24>(5));
)",
              TypecheckSucceeds(HasNodeWithType(
                  "const X = foo(foo<24>(4) + foo<24>(5));", "uN[24]")));
}

TEST(TypecheckV2Test,
     ParametricFunctionCallUsingGlobalConstantInImplicitParametricArgument) {
  EXPECT_THAT(R"(
fn foo<N: u32>(a: uN[N]) -> uN[N] { a }
const X = u3:1;
const Z = foo(X);
)",
              TypecheckSucceeds(HasNodeWithType("const Z = foo(X);", "uN[3]")));
}

TEST(TypecheckV2Test,
     ParametricFunctionCallWithImplicitParameterFollowedByTypePropagation) {
  EXPECT_THAT(R"(
fn foo<N: u32>(a: uN[N]) -> uN[N] { a }
const Y = foo(u15:1);
const Z = Y + 1;
)",
              TypecheckSucceeds(HasNodeWithType("const Z = Y + 1;", "uN[15]")));
}

TEST(TypecheckV2Test, ImplicitParametricBindingRollover) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
fn bar(n: u32) -> u32 { n - (u32:1 << 31) - u32:1 }
fn foo(n: u32) -> u32 { bar(n) }
fn p<N: u32, M: u32 = {foo(N)}>() {
}

fn main() {
  p<u32:0>();  // <-- cause an underflow to occur
}
)"));
  EXPECT_THAT(
      result.tm.warnings.warnings()[0].message,
      AllOf(HasSubstr("constexpr evaluation detected rollover in operation"),
            HasSubstr("left-hand value `0`"),
            HasSubstr("right-hand value `2147483648`"),
            HasSubstr("in bar\nin foo\nfrom fake.x:6:27-6:30")));
}

// TODO(erinzmoore): It should be possible to instantiate a generic type.
TEST(TypecheckV2Test, DISABLED_InstantiateGenericTypeAsStruct) {
  EXPECT_THAT(
      R"(
#![feature(generics)]

struct S {
  x: u32
}

fn main<T: type>() -> T {
  T { x: u32:5 }
}

const RES = main<S>();
const_assert!(RES == S{x: 5});
)",
      TypecheckSucceeds(HasNodeWithType("RES", "S { x: uN[32] }")));
}

}  // namespace
}  // namespace xls::dslx
