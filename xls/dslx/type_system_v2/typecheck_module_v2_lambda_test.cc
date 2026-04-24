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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/dslx/type_system/typecheck_test_utils.h"
#include "xls/dslx/type_system_v2/matchers.h"

namespace xls::dslx {
namespace {

using ::testing::AllOf;

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
fn main() -> u32[4][5] {
  let z = zero!<u32[4][5]>();
  map(enumerate(z), | tup | {
    let i = tup.0;
    let arr = tup.1;
    map(enumerate(arr), | tup2 | {
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

TEST(TypecheckV2Test, NestedLambdaIteratesOverLocalConst) {
  EXPECT_THAT(
      R"(
fn nested() -> u1[2][3] {
   const X = u32:2;
   const Y = u32:3;
   map(0..Y, | y_idx: u32 | {
       map(0..X, | x_idx: u32 | {
           if (x_idx + y_idx) % 2 == 0 {
               u1:1
           } else {
               u1:0
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
      TypecheckSucceeds(AllOf(
          HasNodeWithType("RES", "uN[1][2][3]"),
          HasNodeWithType("lambda_capture_struct_at_fake.x:7:14-15:5::X",
                          "uN[32]"),
          HasNodeWithType("lambda_capture_struct_at_fake.x:8:18-14:9<u32>",
                          "typeof(lambda_capture_struct_at_fake.x:8:18-14:9 { "
                          "y_idx: uN[32] }"))));
}

TEST(TypecheckV2Test, NestedLambdaIteratesOverLocalConstWithExplicitReturn) {
  EXPECT_THAT(
      R"(
fn nested() -> u1[2][3] {
   const X = u32:2;
   const Y = u32:3;
   map(0..Y, | y_idx: u32 | -> u1[X] {
       map(0..X, | x_idx: u32 | {
           if (x_idx + y_idx) % 2 == 0 {
               u1:1
           } else {
               u1:0
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
                HasNodeWithType("lambda_capture_struct_at_fake.x:7:14-15:5::X",
                                "uN[32]"))));
}

TEST(TypecheckV2Test, NestedLambdaIteratesOverGlobalConst) {
  EXPECT_THAT(
      R"(
const X = u32:2;
const Y = u32:3;
type Results = u1[X][Y];

fn nested() -> Results {
   map(0..Y, | y_idx | {
       map(0..X, | x_idx | {
           if (x_idx + y_idx) % 2 == 0 {
               1
           } else {
               0
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
      TypecheckSucceeds(HasNodeWithType("RES", "uN[1][2][3]")));
}

TEST(TypecheckV2Test, LambdaUsesUnrollForOutput) {
  EXPECT_THAT(
      R"(
const A = u32:1;
fn foo() -> u32[5] {
  let B = u32:2;
  let X = unroll_for! (i, a) in u32:0..5 {
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

}  // namespace
}  // namespace xls::dslx
