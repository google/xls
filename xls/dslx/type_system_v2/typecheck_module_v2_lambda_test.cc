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
    let _ = N;
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

}  // namespace
}  // namespace xls::dslx
