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

#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system/typecheck_test_utils.h"
#include "xls/dslx/type_system_v2/matchers.h"

namespace xls::dslx {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::AllOf;

TEST(TypecheckV2Test, LambdaUsesParentFunctionParametricInReturn) {
  EXPECT_THAT(R"(
fn my_conversion<N: u32>(arr: u32[3]) -> uN[N][3] {
  map(arr, |x| -> uN[N] { x as uN[N] })
}

const M = u32:0..3;
const ARR = my_conversion<16>(M);
const_assert!(ARR[1] == u16:1);
)",
              TypecheckSucceeds(HasNodeWithType("ARR", "uN[16][3]")));
}

TEST(TypecheckV2Test, LambdaUsesParentFunctionParametricInFnCall) {
  EXPECT_THAT(R"(
fn helper<N: u32>() -> uN[N] {
  uN[N]:1
}

fn my_conversion<N: u32>(arr: u32[3]) -> uN[N][3] {
  map(arr, |x: u32| -> uN[N] {
    helper<N>()
  })
}

const M = u32:0..3;
const ARR = my_conversion<16>(M);
const_assert!(ARR[1] == u16:1);
)",
              TypecheckSucceeds(HasNodeWithType("ARR", "uN[16][3]")));
}

TEST(TypecheckV2Test, LambdaUsesParentFunctionParametricAndContextCapture) {
  EXPECT_THAT(R"(
fn my_conversion<N: u32>(arr: u32[3]) -> uN[N][3] {
  let delta = uN[N]:5;
  map(arr, |x| -> uN[N] { (x + N) as uN[N] + delta })
}

const M = u32:0..3;
const ARR = my_conversion<16>(M);
const_assert!(ARR[1] == u16:22);
)",
              TypecheckSucceeds(HasNodeWithType("ARR", "uN[16][3]")));
}

TEST(TypecheckV2Test, LambdaUsesParentFunctionParametricInBody) {
  EXPECT_THAT(R"(
fn add_N<N: u32>(arr: u32[3]) -> u32[3] {
  map(arr, |x| { x + N })
}

const M = u32:0..3;
const ARR = add_N<16>(M);
const_assert!(ARR[1] == 17);
)",
              TypecheckSucceeds(HasNodeWithType("ARR", "uN[32][3]")));
}

TEST(TypecheckV2Test, LambdaUsesParentFunctionParametricInBodyMultipleTimes) {
  EXPECT_THAT(R"(
fn add_N<N: u32>(arr: u32[3]) -> u32[3] {
  map(arr, |x| {
    let y = x + N;
    y + N
  })
}

const M = u32:0..3;
const ARR = add_N<16>(M);
const_assert!(ARR[1] == 33);
)",
              TypecheckSucceeds(HasNodeWithType("ARR", "uN[32][3]")));
}

TEST(TypecheckV2Test, LambdaUsesParentFunctionParametricInMacro) {
  EXPECT_THAT(R"(
fn zeros<N: u32>(arr: u32[3]) -> uN[N][3] {
  map(arr, |x| { zero!<uN[N]>() })
}

const M = u32:0..3;
const ARR = zeros<16>(M);
const_assert!(ARR[1] == 0);
)",
              TypecheckSucceeds(HasNodeWithType("ARR", "uN[16][3]")));
}

TEST(TypecheckV2Test, LambdaUsesParentFunctionParametricNoTypeInference) {
  EXPECT_THAT(R"(
fn add_N<N: u32>(arr: u32[3]) -> u32[3] {
  map(arr, |x| { let y: u32 = x + N; y })
}

const M = u32:0..3;
const ARR = add_N<16>(M);
const_assert!(ARR[1] == 17);
)",
              TypecheckSucceeds(HasNodeWithType("ARR", "uN[32][3]")));
}

TEST(TypecheckV2Test, LambdaUnusedParentFunctionParametric) {
  EXPECT_THAT(R"(
fn my_conversion<N: u32, M: u32>(arr: u32[M]) -> uN[N][M] {
  map(arr, |x| -> uN[N] { x as uN[N] })
}

const IN = u32:0..3;
const ARR = my_conversion<16, 3>(IN);
const_assert!(ARR[1] == u16:1);
)",
              TypecheckSucceeds(HasNodeWithType("ARR", "uN[16][3]")));
}

TEST(TypecheckV2Test, LambdaCallsFunction) {
  EXPECT_THAT(
      R"(
fn add_two(x: u32) -> u32 {
  x + 2
}

fn main() -> u32 {
  let ARR = map(u32:0..5, |i| -> u32 { add_two(i) });
  ARR[4]
}

const_assert!(main() == 6);
)",
      TypecheckSucceeds(HasNodeWithType("ARR", "uN[32][5]")));
}

TEST(TypecheckV2Test, LambdaCallsImportedFunction) {
  constexpr std::string_view kImported = R"(
pub fn add_two(x: u32) -> u32 {
  x + 2
}
)";
  constexpr std::string_view kProgram = R"(
import imported;
fn main() -> u32 {
  let ARR = map(u32:0..5, |i| -> u32 { imported::add_two(i) });
  ARR[4]
}

const_assert!(main() == 6);
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data));
  EXPECT_THAT(TypecheckV2(kProgram, "main", &import_data),
              IsOkAndHolds(HasTypeInfo(HasNodeWithType("ARR", "uN[32][5]"))));
}

TEST(TypecheckV2Test, LambdaWithTypeFromContext) {
  EXPECT_THAT(R"(
fn my_conversion(arr: u32[3]) -> uN[16][3] {
  let N = 16;
  map(arr, |x| -> uN[N] { x as uN[N] })
}

const M = u32:0..3;
const ARR = my_conversion(M);
const_assert!(ARR[1] == u16:1);
)",
              TypecheckSucceeds(HasNodeWithType("ARR", "uN[16][3]")));
}

TEST(TypecheckV2Test, LambdaWithExplicitTypes) {
  EXPECT_THAT(R"(
const M = u16:0..6;
const ARR = map(M, | i: u16 | -> u16 { u16:2 * i });
const_assert!(ARR[1] == u16:2);
)",
              TypecheckSucceeds(HasNodeWithType("ARR", "uN[16][6]")));
}

TEST(TypecheckV2Test, NestedLambdas) {
  EXPECT_THAT(
      R"(
import std;
fn main() -> u32[4][5] {
  let z = zero!<u32[4][5]>();
  map(std::enumerate(z), | tup | {
    let i = tup.0;
    let arr = tup.1;
    map(std::enumerate(arr), | tup2 | {
      let j = tup2.0;
      i + j
    })
  })
}

const_assert!(main() == [[u32:0, 1, 2, 3],
        [u32:1, 2, 3, 4],
        [u32:2, 3, 4, 5],
        [u32:3, 4, 5, 6],
        [u32:4, 5, 6, 7]]);

)",
      TypecheckSucceeds(HasNodeWithType("main", "() -> uN[32][4][5]")));
}

TEST(TypecheckV2Test, LambdaWithParamContextCapture) {
  EXPECT_THAT(
      R"(
fn main(x: u32) -> u32 {
  let arr = u32:0..2;
  let added = map(arr, |i| -> u32 { i + x });
  added[1]
}

const_assert!(main(2) == 3);
)",
      TypecheckSucceeds(HasNodeWithType("main", "(uN[32]) -> uN[32]")));
}

TEST(TypecheckV2Test, LambdaWithMultipleParams) {
  EXPECT_THAT(
      R"(
fn main() -> u32 {
  (|i, j| -> u32 {i * j})(u32:2, u32:4)
}
const_assert!(main() == 8);
)",
      TypecheckSucceeds(HasNodeWithType("main", "() -> uN[32]")));
}

TEST(TypecheckV2Test, LambdaWithMultipleParamsMismatch) {
  EXPECT_THAT(
      R"(
fn main() -> u32 {
  (|i, j: bool| -> u32 {i * j})(u32:2, u32:4)
}
)",
      TypecheckFails(HasSizeMismatch("bool", "u32")));
}

TEST(TypecheckV2Test, LambdaWithContextCapture) {
  EXPECT_THAT(
      R"(
fn main() -> u32 {
  const X = u32:4;
  const Y = u32:2;
  let ARR = map(u32:0..5, |i| -> u32 { X * Y * i });
  ARR[4]
}

const_assert!(main() == 32);
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("ARR", "uN[32][5]"),
                              HasNodeWithType("X", "uN[32]"))));
}

TEST(TypecheckV2Test, LambdaWithContextParamsTypeMismatch) {
  EXPECT_THAT(
      R"(
fn main() {
  const X = false;
  let ARR = map(0..5, |i| -> u32 { X * i });
}
)",
      TypecheckFails(HasSizeMismatch("uN[1]", "uN[32]")));
}

TEST(TypecheckV2Test, LambdaGeneratedValueAsType) {
  EXPECT_THAT(R"(
const X = u32:3;
const ARR = map(u32:0..5, |i: u32| -> u32 { X * i });
const TEST = uN[ARR[1]]:0;
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("ARR", "uN[32][5]"),
                                      HasNodeWithType("TEST", "uN[3]"))));
}

TEST(TypecheckV2Test, LambdaImplicitReturnAndParentFunctionParametric) {
  EXPECT_THAT(R"(
fn my_conversion<N: u32, M: u32>(arr: u32[M]) -> uN[N][M] {
  map(arr, |x| { x as uN[N] })
}

const IN = u32:0..3;
const ARR = my_conversion<16, 3>(IN);
const_assert!(ARR[1] == u16:1);
)",
              TypecheckSucceeds(HasNodeWithType("ARR", "uN[16][3]")));
}

TEST(TypecheckV2Test, LambdaWithImplicitReturn) {
  EXPECT_THAT(
      R"(
fn main() -> u32 {
  let arr = map(u32:0..2, |i| { i });
  arr[1]
}
const_assert!(main() == 1);

)",
      TypecheckSucceeds(HasNodeWithType("main", "() -> uN[32]")));
}

TEST(TypecheckV2Test, LambdaWithImplicitReturnNoBraces) {
  EXPECT_THAT(
      R"(
fn main() -> u32 {
  let arr = map(u32:0..2, |i| i + 3);
  arr[1]
}
const_assert!(main() == 4);

)",
      TypecheckSucceeds(HasNodeWithType("main", "() -> uN[32]")));
}

TEST(TypecheckV2Test, LambdaWithImplicitReturnAndContextCapture) {
  EXPECT_THAT(
      R"(
fn main() -> u32 {
  let arr = u32:0..2;
  let x = u32:2;
  let added = map(arr, |i| { i + x });
  added[1]
}

const_assert!(main() == 3);
)",
      TypecheckSucceeds(HasNodeWithType("main", "() -> uN[32]")));
}

TEST(TypecheckV2Test, LambdaWithImplicitReturnAndParamContextCapture) {
  EXPECT_THAT(
      R"(
fn main(x: u32) -> u32 {
  let arr = u32:0..2;
  let added = map(arr, |i| { i + x });
  added[1]
}

const_assert!(main(2) == 3);
)",
      TypecheckSucceeds(HasNodeWithType("main", "(uN[32]) -> uN[32]")));
}

TEST(TypecheckV2Test, LambdaWithImplicitParam) {
  EXPECT_THAT(
      R"(
const ARR = map(u16:0..6, | i | -> u16 { i });
const_assert!(ARR[1] == u16:1);
const_assert!(ARR[5] == u16:5);
)",
      TypecheckSucceeds(HasNodeWithType("ARR", "uN[16][6]")));
}

TEST(TypecheckV2Test, LambdaUsesMapResult) {
  EXPECT_THAT(R"(
fn is_odd(i: u32) -> bool {
  i % 2 == 1
}

fn add_two<N: u32>() -> u32[N] {
  let odd_map = map(0..N, is_odd);
  map(
    0..N,
    |i| -> u32 {
      if odd_map[i] {
        i + 2
      } else {
        i
      }
    }
  )
}

const RES = add_two<5>();
const_assert!(RES == [u32:0, 3, 2, 5, 4]);
const RES2 = add_two<3>();
const_assert!(RES2 == [u32:0, 3, 2]);
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("RES", "uN[32][5]"),
                                      HasNodeWithType("RES2", "uN[32][3]"))));
}

TEST(TypecheckV2Test, NestedLambdaIteratesOverGlobalConst) {
  EXPECT_THAT(
      R"(
const X = u32:2;
const Y = u32:3;
type Results = u1[X][Y];
type MyBit = u1;

fn nested() -> Results {
   map(0..Y, | y_idx | {
       map(0..X, | x_idx | -> MyBit {
           if (x_idx + y_idx) % 2 == 0 {
               1 as MyBit
           } else {
               0 as MyBit
           }
       })
   })
}

const RES = nested();
const EX = [
  [u1:1, u1:0],
  [u1:0, u1:1],
  [u1:1, u1:0],
];
const_assert!(RES == EX);

)",
      TypecheckSucceeds(
          AllOf(HasNodeWithType("RES", "uN[1][2][3]"),
                HasNodeWithType(
                    "lambda_capture_struct_at_fake.x:10:14-18:5",
                    "typeof(lambda_capture_struct_at_fake.x:10:14-18:5 {})"))));
}

TEST(TypecheckV2Test, NestedLambdaWithInnerAndOuterLoopVars) {
  EXPECT_THAT(
      R"(
const X = u32:2;
const Y = u32:3;
const COUNT = u32:4;
type word = u4;
type Results = word[X][Y];

fn nested() -> Results {
  let used_in_outer = [[false, false], [true, true], [true, false], [false, false]];
  for (ct, result): (u32, Results) in u32:0..COUNT {
      map(result, | res | -> word[X] {
          let used_in_inner = used_in_outer[ct];
          map(0..X, | x_idx | -> word {
              if used_in_inner[x_idx] {
                  res[x_idx] + 1
              } else {
                  res[x_idx]
              }
          })
      })
  }(zero!<Results>())
}

const RES = nested();
const EX = [
  [u4:2, u4:1],
  [u4:2, u4:1],
  [u4:2, u4:1],
];
const_assert!(RES == EX);

)",
      TypecheckSucceeds(HasNodeWithType("RES", "uN[4][2][3]")));
}

TEST(TypecheckV2Test, LambdaUsesUnrollForOutput) {
  EXPECT_THAT(
      R"(
const A = u32:1;
fn foo() -> u32[5] {
  let B = u32:2;
  const X = unroll_for! (i, a) in u32:0..5 {
    let C = B + i;
    let D = A * a;
    C + D
  }(u32:0);

  map(u32:0..5, |i| { i + X })
}

const FOO = foo();
const_assert!(FOO == [u32:20, 21, 22, 23, 24]);

)",
      TypecheckSucceeds(AllOf(
          HasNodeWithType("X", "uN[32]"), HasNodeWithType("FOO", "uN[32][5]"),
          HasNodeWithType("lambda_capture_struct_at_fake.x:13:17-13:30<u32>",
                          "typeof(lambda_capture_struct_at_fake.x:13:17-13:30 "
                          "{ X: uN[32] })"))));
}

TEST(TypecheckV2Test, LambdaUsesColonRef) {
  EXPECT_THAT(
      R"(
struct S {}
impl S {
  const C: u32 = 5;
}

const ARR = map(u32:0..6, | i | { S::C + i });
const_assert!(ARR == [u32:5, 6, 7, 8, 9, 10]);
)",
      TypecheckSucceeds(HasNodeWithType("ARR", "uN[32][6]")));
}

TEST(TypecheckV2Test, LambdaUsesLocalTypeInFnCall) {
  EXPECT_THAT(R"(
#![feature(generics)]

fn helper<T: type>() -> T {
  1 as T
}

fn my_conversion<N: u32>(arr: u32[3]) -> uN[N][3] {
  type MyN = uN[N];
  map(arr, |x: u32| -> uN[N] {
    helper<MyN>()
  })
}

const M = u32:0..3;
const ARR = my_conversion<16>(M);
const_assert!(ARR[1] == u16:1);
)",
              TypecheckSucceeds(HasNodeWithType("ARR", "uN[16][3]")));
}

TEST(TypecheckV2Test, LambdaUsesLocalTypeExplicitReturn) {
  EXPECT_THAT(
      R"(
fn main() -> u16[5] {
  type Int = u16;
  let res = map(u32:0..5, | i: u32 | -> Int {zero!<Int>() + i as Int});
  res
}

const ARR = main();
const_assert!(ARR == [u16:0, 1, 2, 3, 4]);
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("ARR", "uN[16][5]"),
                              HasNodeWithType("res", "uN[16][5]"))));
}

TEST(TypecheckV2Test, LambdaUsesLocalTypeImplicitReturn) {
  EXPECT_THAT(
      R"(
fn main() -> u16[5] {
  type Int = u16;
  let res = map(u32:0..5, | i: u32 | {zero!<Int>() + i as Int});
  res
}

const ARR = main();
const_assert!(ARR == [u16:0, 1, 2, 3, 4]);
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("ARR", "uN[16][5]"),
                              HasNodeWithType("res", "uN[16][5]"))));
}

TEST(TypecheckV2Test, LambdaUsesParentGenericTypeImplicitReturnType) {
  EXPECT_THAT(
      R"(
#![feature(generics)]

fn main<T: type>() -> T[5] {
  let res = map(u32:0..5, |i| {i as T});
  res
}

const ONE = main<u16>();
const TWO = main<u24>();
const_assert!(ONE == [u16:0, 1, 2, 3, 4]);
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("ONE", "uN[16][5]"),
                              HasNodeWithType("TWO", "uN[24][5]"))));
}

TEST(TypecheckV2Test, LambdaUsesParentGenericTypeAndContextCapture) {
  EXPECT_THAT(
      R"(
#![feature(generics)]

fn main<T: type>() -> T[5] {
  let x: T = 5;
  map(u32:0..5, |i| {i as T + x})
}

const ONE = main<u16>();
const TWO = main<u24>();
const_assert!(ONE == [u16:5, 6, 7, 8, 9]);
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("ONE", "uN[16][5]"),
                              HasNodeWithType("TWO", "uN[24][5]"))));
}

TEST(TypecheckV2Test, LambdaUsesLocalTypeFromParentGenericImplicitReturn) {
  EXPECT_THAT(
      R"(
#![feature(generics)]

fn main<OuterType: type>() -> OuterType[5] {
  let y : OuterType = 17;
  type LmType = OuterType;
  map(u32:0..5, |i| {y + i as LmType})
}

const ONE = main<u16>();
const TWO = main<u24>();
const_assert!(ONE == [u16:17, 18, 19, 20, 21]);
const_assert!(TWO == [u24:17, 18, 19, 20, 21]);
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("ONE", "uN[16][5]"),
                              HasNodeWithType("TWO", "uN[24][5]"))));
}

TEST(TypecheckV2Test, LambdaUsesLocalTypeFromParentGenericWithContextCapture) {
  EXPECT_THAT(
      R"(
#![feature(generics)]

fn main<OuterType: type>(x: OuterType) -> OuterType[5] {
  type LmType = OuterType;
  map(u32:0..5, |i| {i as LmType + x})
}

const ONE = main<u16>(u16:5);
const TWO = main<u24>(u24:3);
const_assert!(ONE == [u16:5, 6, 7, 8, 9]);
const_assert!(TWO == [u24:3, 4, 5, 6, 7]);
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("ONE", "uN[16][5]"),
                              HasNodeWithType("TWO", "uN[24][5]"))));
}

TEST(TypecheckV2Test, LambdaUsesLocalTypeFromParentGeneric) {
  EXPECT_THAT(
      R"(
#![feature(generics)]

fn main<OuterType: type>() -> OuterType[5] {
  type LmType = OuterType;
  map(u32:0..5, |i| -> LmType {i as LmType})
}

const ONE = main<u16>();
const TWO = main<u24>();
const_assert!(ONE == [u16:0, 1, 2, 3, 4]);
const_assert!(TWO == [u24:0, 1, 2, 3, 4]);
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("ONE", "uN[16][5]"),
                              HasNodeWithType("TWO", "uN[24][5]"))));
}

TEST(TypecheckV2Test, LambdaUsesParentGenericTypeExplicitReturn) {
  EXPECT_THAT(
      R"(
#![feature(generics)]

fn main<OuterType: type>() -> OuterType[5] {
  map(u32:0..5, |i| -> OuterType {i as OuterType})
}

const ONE = main<u16>();
const TWO = main<u24>();
const_assert!(ONE == [u16:0, 1, 2, 3, 4]);
const_assert!(TWO == [u24:0, 1, 2, 3, 4]);
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("ONE", "uN[16][5]"),
                              HasNodeWithType("TWO", "uN[24][5]"))));
}

TEST(TypecheckV2Test, NestedLambdaUsesLocalTypeFromParent) {
  EXPECT_THAT(
      R"(
fn main<N: u32>() -> uN[N][5][4] {
  type Int = uN[N];
  let z = Int:5;
  let res = map(u32:0..4, | j | -> Int[5] {
    map(u32:0..5, | i | {j as Int + i as Int + z})
  });
  res
}

const ONE = main<16>();
const TWO = main<32>();
const_assert!(ONE == [[u16:5, 6, 7, 8, 9],
                      [u16:6, 7, 8, 9, 10],
                      [u16:7, 8, 9, 10, 11],
                      [u16:8, 9, 10, 11, 12]]);
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("ONE", "uN[16][5][4]"),
                              HasNodeWithType("TWO", "uN[32][5][4]"))));
}

TEST(TypecheckV2Test, NestedLambdaUsesLocalGenericTypeFromParent) {
  EXPECT_THAT(
      R"(
#![feature(generics)]

fn main<T: type>() -> T[5][4] {
  type Int = T;
  let z = Int:5;
  map(u32:0..4, | j | -> Int[5] {
    map(u32:0..5, | i | { Int:1 })
  })
}

const ONE = main<u16>();
const TWO = main<u32>();
const_assert!(ONE == [[u16:1, 1, 1, 1, 1],
                      [u16:1, 1, 1, 1, 1],
                      [u16:1, 1, 1, 1, 1],
                      [u16:1, 1, 1, 1, 1]]);
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("ONE", "uN[16][5][4]"),
                              HasNodeWithType("TWO", "uN[32][5][4]"))));
}

TEST(TypecheckV2Test, LambdaUsesLocalStructType) {
  EXPECT_THAT(R"(
struct S<N: u32> {
  x: uN[N]
}

fn main() -> S<8>[5] {
  type MyS = S<8>;
  map(u8:0..5, |i| { MyS{x: i} })
}

const RES = main();
const_assert!(RES[0] == S<8>{x: 0});
)",
              TypecheckSucceeds(HasNodeWithType("RES", "S { x: uN[8] }[5]")));
}
TEST(TypecheckV2Test, LambdaUsesLocalStructTypeExplicitReturn) {
  EXPECT_THAT(R"(
struct S<N: u32> {
  x: uN[N]
}

fn main() -> S<8>[5] {
  type MyS = S<8>;
  map(u8:0..5, |i| -> MyS { MyS{x: i} })
}

const RES = main();
const_assert!(RES[0] == S<8>{x: 0});
)",
              TypecheckSucceeds(HasNodeWithType("RES", "S { x: uN[8] }[5]")));
}

TEST(TypecheckV2Test, LambdaUsesLocalStructTypeWithParentParametric) {
  EXPECT_THAT(
      R"(
struct S<N: u32> {
  x: uN[N]
}

fn main<N: u32>() -> S<N>[5] {
  type MyS = S<N>;
  map(uN[N]:0..5, |i| -> MyS { MyS{x: i} })
}

const ONE = main<16>();
const TWO = main<8>();
const_assert!(ONE[1] == S<16>{x: 1});
const_assert!(TWO[4] == S<8>{x: 4});
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("ONE", "S { x: uN[16] }[5]"),
                              HasNodeWithType("TWO", "S { x: uN[8] }[5]"))));
}

TEST(TypecheckV2Test, LambdaUsesLocalStructTypeAndParentParametricInBody) {
  EXPECT_THAT(
      R"(
struct S<N: u32> {
  x: uN[N]
}

fn main<N: u32>() -> S<N>[5] {
  type MyS = S;
  map(uN[N]:0..5, |i| { MyS<N>{x: i} })
}

const ONE = main<16>();
const TWO = main<8>();
const_assert!(ONE[1] == S<16>{x: 1});
const_assert!(TWO[4] == S<8>{x: 4});
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("ONE", "S { x: uN[16] }[5]"),
                              HasNodeWithType("TWO", "S { x: uN[8] }[5]"))));
}

TEST(TypecheckV2Test, LambdaUsesLocalStructTypeWithParentGeneric) {
  EXPECT_THAT(
      R"(
#![feature(generics)]

struct S<U: type> {
  x: U
}

fn main<T: type>() -> S<T>[5] {
  type MyS = S<T>;
  map(u32:0..5, |i| { zero!<MyS>() })
}

const ONE = main<u16>();
const TWO = main<u8>();
const_assert!(ONE[1] == S<u16>{x: 0});
const_assert!(TWO[4] == S<u8>{x: 0});
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("ONE", "S { x: uN[16] }[5]"),
                              HasNodeWithType("TWO", "S { x: uN[8] }[5]"))));
}

TEST(TypecheckV2Test, LambdaUsesLocalStructTypeMismatch) {
  EXPECT_THAT(
      R"(
struct S<N: u32> {
  x: uN[N]
}

fn main() -> S<16>[5] {
  type MyS = S<8>;
  map(u16:0..5, |i| { MyS{x: i} })
}

const RES = main();
)",
      TypecheckFails(HasSizeMismatch("uN[16]", "uN[8]")));
}

// TODO(erinzmoore): Enable once generic struct instantiation is supported.
TEST(TypecheckV2Test, DISABLED_LambdaUsesGenericTypeAsStruct) {
  EXPECT_THAT(
      R"(
#![feature(generics)]

struct S<N: u32> {
  x: uN[N]
}

fn main<T: type>() -> T[5] {
  type MyS = T;
  map(u32:0..5, |i| -> MyS { MyS{x: i} })
}

const ONE = main<S<16>>();
const TWO = main<S<8>>();
const_assert!(ONE[1] == S<u16>{x: 1});
const_assert!(TWO[4] == S<u8>{x: 4});
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("ONE", "S { x: uN[16] }[5]"),
                              HasNodeWithType("TWO", "S { x: uN[8] }[5]"))));
}

}  // namespace
}  // namespace xls::dslx
