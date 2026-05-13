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
#include "xls/dslx/ir_convert/convert_options.h"
#include "xls/dslx/ir_convert/ir_converter_test_utils.h"
#include "xls/dslx/type_system/typecheck_test_utils.h"

namespace xls::dslx {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

TEST_F(IrConverterTest, NamedConstant) {
  constexpr std::string_view program =
      R"(fn f() -> u32 {
  let foo: u32 = u32:42;
  foo
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::string converted,
                           ConvertOneFunctionForTest(program, "f"));
  ExpectIr(converted);
}

TEST_F(IrConverterTest, Concat) {
  constexpr std::string_view program =
      R"(fn f(x: bits[31]) -> u32 {
  bits[1]:1 ++ x
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::string converted,
                           ConvertOneFunctionForTest(program, "f"));
  ExpectIr(converted);
}

TEST_F(IrConverterTest, ParameterNamePreservedOnLetAlias) {
  constexpr std::string_view program = R"(
pub fn my_fun(baz: u32) -> u32 {
  let foo = baz;
  foo
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(std::string converted,
                           ConvertOneFunctionForTest(program, "my_fun"));

  // Expect the parameter to retain its original DSLX name `baz` in IR.
  EXPECT_EQ(converted, R"(package test_module

file_number 0 "test_module.x"

top fn __test_module__my_fun(baz: bits[32] id=1) -> bits[32] {
  ret baz: bits[32] = param(name=baz, id=1)
}
)");
}

TEST_F(IrConverterTest, TwoPlusTwo) {
  constexpr std::string_view program =
      R"(fn two_plus_two() -> u32 {
  u32:2 + u32:2
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::string converted,
                           ConvertOneFunctionForTest(program, "two_plus_two"));
  ExpectIr(converted);
}

TEST_F(IrConverterTest, SignedDiv) {
  constexpr std::string_view program =
      R"(fn signed_div(x: s32, y: s32) -> s32 {
  x / y
})";
  XLS_ASSERT_OK_AND_ASSIGN(std::string converted,
                           ConvertOneFunctionForTest(program, "signed_div"));
  ExpectIr(converted);
}

TEST_F(IrConverterTest, NegativeX) {
  constexpr std::string_view program =
      R"(fn negate(x: u32) -> u32 {
  -x
})";
  XLS_ASSERT_OK_AND_ASSIGN(std::string converted,
                           ConvertOneFunctionForTest(program, "negate"));
  ExpectIr(converted);
}

TEST_F(IrConverterTest, LetBinding) {
  constexpr std::string_view program =
      R"(fn f() -> u32 {
  let x: u32 = u32:2;
  x+x
})";
  XLS_ASSERT_OK_AND_ASSIGN(std::string converted,
                           ConvertOneFunctionForTest(program, "f"));
  ExpectIr(converted);
}

TEST_F(IrConverterTest, ExtendConversions) {
  constexpr std::string_view program =
      R"(fn main(x: u8, y: s8) -> (u32, u32, s32, s32) {
  (x as u32, y as u32, x as s32, y as s32)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::string converted,
                           ConvertModuleForTest(program));
  ExpectIr(converted);
}

TEST_F(IrConverterTest, Conditional) {
  constexpr std::string_view program =
      R"(fn main(x: bool) -> u8 {
  if x { u8:42 } else { u8:24 }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::string converted,
                           ConvertModuleForTest(program));
  ExpectIr(converted);
}

TEST_F(IrConverterTest, ConditionalsPlusStuff) {
  constexpr std::string_view program =
      R"(
fn foo(a: bool) -> u8 {
  let x = if a { u8:42 } else { u8:24 } + u8:1;
  let y = u8:1 + if a { u8:42 } else { u8:24 };
  let z = if a { u8:42 } else { u8:24 } + if a { u8:42 } else { u8:24 };
  x + y + z
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::string converted,
                           ConvertModuleForTest(program));
  ExpectIr(converted);
}

TEST_F(IrConverterTest, ConstConditional) {
  constexpr std::string_view program =
      R"(
fn main() -> u32 {
  const if true {
    u16:42
  } else {
    u8:24
  } as u32
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(std::string converted,
                           ConvertModuleForTest(program));
  ExpectIr(converted);
}

TEST_F(IrConverterTest, ConstIfOfParametricInFn) {
  constexpr std::string_view kProgram = R"(
fn const_if_disparate_types<A: u32>() -> u32 {
    let data = const if A == u32:0 {
        u16:600
    } else if A == u32:1 {
        u32:70000
    } else {
        u8:4
    };

    data as u32
}

fn main() -> u32 { const_if_disparate_types<u32:0>() }
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::string converted,
                           ConvertModuleForTest(kProgram));
  ExpectIr(converted);
}

TEST_F(IrConverterTest, ConstantsWithConditionalsPlusStuff) {
  constexpr std::string_view program =
      R"(
const A = true;
const X = if A { u8:42 } else { u8:24 } + u8:1;
const Y = u8:1 + if A { u8:42 } else { u8:24 };
const Z = if A { u8:42 } else { u8:24 } + if A { u8:42 } else { u8:24 };
fn main(x: bool) -> u8 { X + Y + Z }
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::string converted,
                           ConvertModuleForTest(program));
  ExpectIr(converted);
}

TEST_F(IrConverterTest, WidthSlice) {
  constexpr std::string_view program =
      R"(
fn f(x: u32, y: u32) -> u8 {
  x[2+:u8]+x[y+:u8]
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::string converted,
                           ConvertModuleForTest(program));
  ExpectIr(converted);
}

TEST_F(IrConverterTest, BitSliceCast) {
  constexpr std::string_view program =
      R"(
fn main(x: u2) -> u1 {
  x as u1
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::string converted,
                           ConvertModuleForTest(program));
  ExpectIr(converted);
}

TEST_F(IrConverterTest, CastOfAdd) {
  constexpr std::string_view program =
      R"(
fn main(x: u8, y: u8) -> u32 {
  (x + y) as u32
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::string converted,
                           ConvertModuleForTest(program));
  ExpectIr(converted);
}

TEST_F(IrConverterTest, IdentityFinalArg) {
  constexpr std::string_view program =
      R"(
fn main(x0: u19, x3: u29) -> u29 {
  let x15: u29 = u29:0;
  let x17: u19 = (x0) + (x15 as u19);
  x3
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::string converted,
                           ConvertModuleForTest(program));
  ExpectIr(converted);
}

TEST_F(IrConverterTest, ModuleLevelConstantDims) {
  constexpr std::string_view program =
      R"(
const BATCH_SIZE = u32:17;

fn main(x: u32[BATCH_SIZE]) -> u32 {
  x[u32:16]
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::string converted,
                           ConvertModuleForTest(program));
  ExpectIr(converted);
}

TEST_F(IrConverterTest, BitSliceSyntax) {
  constexpr std::string_view program =
      R"(
fn f(x: u4) -> u2 {
  x[:2]+x[-2:]+x[1:3]+x[-3:-1]+x[0:-2]
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::string converted,
                           ConvertModuleForTest(program));
  ExpectIr(converted);
}

TEST_F(IrConverterTest, Identity) {
  constexpr std::string_view program =
      R"(fn main(x: u8) -> u8 {
  x
})";
  XLS_ASSERT_OK_AND_ASSIGN(std::string converted,
                           ConvertModuleForTest(program));
  ExpectIr(converted);
}

TEST_F(IrConverterTest, BitSliceUpdate) {
  constexpr std::string_view program =
      R"(
fn main(x: u32, y: u16, z: u8) -> u32 {
  bit_slice_update(x, y, z)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::string converted,
                           ConvertModuleForTest(program));
  ExpectIr(converted);
}

TEST_F(IrConverterTest, TokenIdentityFunction) {
  constexpr std::string_view program = "fn main(x: token) -> token { x }";
  XLS_ASSERT_OK_AND_ASSIGN(std::string converted,
                           ConvertModuleForTest(program));
  ExpectIr(converted);
}

TEST_F(IrConverterTest, ParameterShadowingModuleLevelConstant) {
  constexpr std::string_view program = R"(
  const FOO = u32:0;

  fn test1<FOO:u32>(x:u32) -> u32 {
    x + FOO
  }

  fn test2<FOO:u32>(x:u32) -> u32 {
    test1<FOO>(x) - FOO
  }

  fn main() -> u32 {
    let foo = test2<u32:3>(u32:3);
    foo
  }
  )";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      std::string converted,
      ConvertModuleForTest(program, kProcScopedChannelOptions, &import_data));
  ExpectIr(converted);
}

TEST_F(IrConverterTest, StringWithUnsignedRangeCharacter) {
  constexpr std::string_view program = R"(fn main() -> u8[1] {
  "\x80"  // -128 in signed char interpretation
})";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      std::string converted,
      ConvertOneFunctionForTest(program, "main", import_data));
  ExpectIr(converted);
}

TEST_F(IrConverterTest, WideningCastsUnErrors) {
  constexpr std::string_view program =
      R"(
fn main(x: u8) -> u32 {
  let x_32 = widening_cast<u32>(x);
  let x_4  = widening_cast<u4>(x_32);
  x_32 + widening_cast<u32>(x_4)
}
)";

  auto import_data = CreateImportDataForTest();
  EXPECT_THAT(ConvertOneFunctionForTest(program, "main", import_data),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Cannot cast from type `uN[32]` (32 bits) "
                                 "to `uN[4]` (4 bits) with widening_cast")));
}

TEST_F(IrConverterTest, WideningAndCheckedCasts) {
  constexpr std::string_view program =
      R"(
fn main(x: u8, y: s8) -> u32 {
  let x_32 = checked_cast<u32>(widening_cast<u32>(x));
  let y_16 = widening_cast<s16>(checked_cast<s7>(y));
  let y_times_two_32 = checked_cast<s32>(y_16) + widening_cast<s32>(y_16);
  x_32 + checked_cast<u32>(y_times_two_32)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(std::string converted,
                           ConvertModuleForTest(program));
  ExpectIr(converted);
}

TEST_F(IrConverterTest, ConstantInFn) {
  constexpr std::string_view program = R"(
fn f(input: u16) -> u16 {
  all_ones!<u16>() + input
}

fn main() -> u16 {
  f(u16:1)
}
)";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(std::string converted,
                           ConvertOneFunctionForTest(program, "main"));
  ExpectIr(converted);
}

TEST_F(IrConverterTest, ParametricConstantInFn) {
  constexpr std::string_view program = R"(
fn f<N:u32>(input: uN[N]) -> uN[N] {
  all_ones!<uN[N]>() + input
}

fn main() -> u16 {
  f<u32:16>(u16:1)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(std::string converted,
                           ConvertOneFunctionForTest(program, "main"));
  ExpectIr(converted);
}

TEST_F(IrConverterTest, UnrollFor) {
  // Based on UnrollForWithoutIndexAccTypeAnnotation
  constexpr std::string_view program = R"(
proc SomeProc {
  init { () }
  config() { }
  next(state: ()) {
    unroll_for! (i, a) in u32:0..u32:4 {
      a + i
    }(u32:0);
  }
})";

  XLS_ASSERT_OK_AND_ASSIGN(std::string converted,
                           ConvertOneFunctionForTest(program, "SomeProc"));
  ExpectIr(converted);
}

}  // namespace
}  // namespace xls::dslx
