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

// This file implements floating point fast (approximate) inverse square
// root. This should be able to compute 1.0 / sqrt(x) using fewer hardware
// resources than using a sqrt and division module, although this hasn't been
// benchmarked yet. Latency is expected to be lower as well. The tradeoff is
// that this offers slighlty less precision (error is < 0.2% in worst case). The
// accuracy-resources tradeoff can be adjusted by changing the number of
// Newton's method iterations (default 1).
//
// Note:
//  - Input denormals are treated as/flushed to 0. (denormals-are-zero / DAZ).
//  - Only round-to-nearest mode is supported.
//  - No exception flags are raised/reported.
//  - We emit a single, canonical representation for NaN (qnan) but accept
//    all NaN respresentations as input.
//
// Reference: https://en.wikipedia.org/wiki/Fast_inverse_square_root

import apfloat
import float32
import xls.modules.fp.fp32_add_2
import xls.modules.fp.fp32_mul_2

type F32 = float32::F32;

// Computes an approximation of 1.0 / sqrt(x). NUM_REFINEMENTS can be increased
// to tradeoff more hardware resources for more accuracy.
pub fn fp32_fast_rsqrt_config_refinements<NUM_REFINEMENTS: u32 = u32:1>(x: F32) -> F32 {
  const zero_point_five = F32 {sign: u1:0,
                               bexp: u8:0x7e,
                               fraction:  u23:0};
  const one_point_five  = F32 {sign: u1:0,
                               bexp: u8:0x7f,
                               fraction:  u1:1 ++ u22:0};
  const magic_number = u32:0x5f3759df;

  // Flush subnormal input.
  let x = float32::subnormals_to_zero(x);

  let approx = float32::unflatten(
                  magic_number - (float32::flatten(x) >> u32:1));
  let half_x = fp32_mul_2::fp32_mul_2(x, zero_point_five);

  // Refine solution w/ Newton's method.
  let result = for (idx, approx): (u32, F32) in range (u32:0, NUM_REFINEMENTS) {
    let prod = fp32_mul_2::fp32_mul_2(half_x, approx);
    let prod = fp32_mul_2::fp32_mul_2(prod, approx);
    let nprod = F32{sign: !prod.sign, bexp: prod.bexp, fraction: prod.fraction};
    let diff = fp32_add_2::fp32_add_2(one_point_five, nprod);
    fp32_mul_2::fp32_mul_2(approx, diff)
  } (approx);

  // I don't *think* it is possible to underflow / have a subnormal result
  // here. In order to have a subnormal result, x would have to be so large
  // that it overflows to infinity (handled below).

  // Special cases.
  // 1/sqrt(inf) -> 0, 1/sqrt(-inf) -> NaN (handled below along
  // with other negative numbers).
  let result = if float32::is_inf(x) { float32::zero(x.sign) } else { result };
  // 1/sqrt(x < 0) -> NaN
  let result = if x.sign == u1:1 { float32::qnan() } else { result };
  // 1/sqrt(NaN) -> NaN.
  let result = if float32::is_nan(x) { x } else { result };
  // 1/sqrt(0) -> inf, 1/sqrt(-0) -> -inf
  let result = if float32::is_zero_or_subnormal(x) { float32::inf(x.sign) }
               else { result };
  result
}

pub fn fp32_fast_rsqrt(x: F32) -> F32 {
  fp32_fast_rsqrt_config_refinements<u32:1>(x)
}

#[test]
fn fast_sqrt_test() {
  // Test Special cases.
  let _ = assert_eq(fp32_fast_rsqrt(float32::zero(u1:0)),
    float32::inf(u1:0));
  let _ = assert_eq(fp32_fast_rsqrt(float32::zero(u1:1)),
    float32::inf(u1:1));
  let _ = assert_eq(fp32_fast_rsqrt(float32::inf(u1:0)),
    float32::zero(u1:0));
  let _ = assert_eq(fp32_fast_rsqrt(float32::inf(u1:1)),
    float32::qnan());
  let _ = assert_eq(fp32_fast_rsqrt(float32::qnan()),
    float32::qnan());
  let _ = assert_eq(fp32_fast_rsqrt(float32::one(u1:1)),
    float32::qnan());
  let pos_denormal = F32{sign: u1:0, bexp: u8:0, fraction: u23:99};
  let _ = assert_eq(fp32_fast_rsqrt(pos_denormal),
    float32::inf(u1:0));
  let neg_denormal = F32{sign: u1:1, bexp: u8:0, fraction: u23:99};
  let _ = assert_eq(fp32_fast_rsqrt(neg_denormal),
    float32::inf(u1:1));
  ()
}

