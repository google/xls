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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/type_system/typecheck_test_utils.h"
#include "xls/dslx/type_system_v2/matchers.h"

namespace xls::dslx {
namespace {

using ::testing::AllOf;

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

// TODO: This case currently doesn't pass const evaluation because the type of
// `self` isn't available in TypeInfo for the impl function.
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

}  // namespace
}  // namespace xls::dslx
