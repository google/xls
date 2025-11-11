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

}  // namespace
}  // namespace xls::dslx
