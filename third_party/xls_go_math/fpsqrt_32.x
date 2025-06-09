// Copyright 2009 The Go Authors. All rights reserved.
// Copyright 2021 The XLS Authors
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This code is based on the go double sqrt implementation,
// available at https://golang.org/src/math/sqrt.go
// This implementation is in turn based on
// FreeBSD's /usr/src/lib/msun/src/e_sqrt.c
// ====================================================
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//
// Developed at SunPro, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this
// software is freely granted, provided that this notice
// is preserved.
// ====================================================
//
// __ieee754_sqrt(x)
// Return correctly rounded sqrt.
//           -----------------------------------------
//           | Use the hardware sqrt if you have one |
//           -----------------------------------------
// Method:
//   Bit by bit method using integer arithmetic. (Slow, but portable)
//   1. Normalization
//      Scale x to y in [1,4) with even powers of 2:
//      find an integer k such that  1 <= (y=x*2**(2k)) < 4, then
//              sqrt(x) = 2**k * sqrt(y)
//   2. Bit by bit computation
//      Let q  = sqrt(y) truncated to i bit after binary point (q = 1),
//           i                                                   0
//                                     i+1         2
//          s  = 2*q , and      y  =  2   * ( y - q  ).          (1)
//           i      i            i                 i
//
//      To compute q    from q , one checks whether
//                  i+1       i
//
//                            -(i+1) 2
//                      (q + 2      )  <= y.                     (2)
//                        i
//                                                            -(i+1)
//      If (2) is false, then q   = q ; otherwise q   = q  + 2      .
//                             i+1   i             i+1   i
//
//      With some algebraic manipulation, it is not difficult to see
//      that (2) is equivalent to
//                             -(i+1)
//                      s  +  2       <= y                       (3)
//                       i                i
//
//      The advantage of (3) is that s  and y  can be computed by
//                                    i      i
//      the following recurrence formula:
//          if (3) is false
//
//          s     =  s  ,       y    = y   ;                     (4)
//           i+1      i          i+1    i
//
//      otherwise,
//                         -i                      -(i+1)
//          s     =  s  + 2  ,  y    = y  -  s  - 2              (5)
//           i+1      i          i+1    i     i
//
//      One may easily use induction to prove (4) and (5).
//      Note. Since the left hand side of (3) contain only i+2 bits,
//            it is not necessary to do a full (53-bit) comparison
//            in (3).
//   3. Final rounding
//      After generating the 53 bits result, we compute one more bit.
//      Together with the remainder, we can decide whether the
//      result is exact, bigger than 1/2ulp, or less than 1/2ulp
//      (it will never equal to 1/2ulp).
//      The rounding mode can be detected by checking whether
//      huge + tiny is equal to huge, and whether huge - tiny is
//      equal to huge for some floating point number "huge" and "tiny".
//
// Notes:  Rounding mode detection omitted.

// This file implements [most of] IEEE 754 single-precision
// floating point square-root, with the following exceptions:
//  - Input denormals are treated as/flushed to 0.
//      (denormals-are-zero / DAZ).
//  - Only round-to-nearest mode is supported.
//  - No exception flags are raised/reported.
// In all other cases, results should be identical to other
// conforming implementations, modulo exact fraction
// values in the NaN case (we emit a single, canonical
// representation for NaN (qnan) but accept all NaN
// respresentations as input).

import float32;

type F32 = float32::F32;

pub fn fpsqrt_32(x: F32) -> F32 {
  // Flush subnormal input.
  let x = float32::subnormals_to_zero(x);

  let exp = float32::unbiased_exponent(x);

  let scaled_fixed_point_x = u1:0 ++ u8:1 ++ x.fraction;
  // If odd exp, double x to make it even.
  let scaled_fixed_point_x = if (exp as u8)[0:1] { scaled_fixed_point_x << u32:1 }
                             else { scaled_fixed_point_x };
  // exp = exp / 2, exponent of square root
  let exp = exp >> u8:1;

  // Generate sqrt(x) bit by bit.
  let scaled_fixed_point_x = scaled_fixed_point_x << u32:1;

  // s is scaled version of the square root calculated down to a
  let (scaled_fixed_point_x, sqrt_in_progress, _) =
    for (_, (scaled_fixed_point_x,
               sqrt_in_progress,
               shifting_bit_mask)):
        (u32, (u32,
               u32,
               u32))
        in u32:0..(u32:23 + u32:2) {

    let temp = (sqrt_in_progress << u32:1) | shifting_bit_mask;

    // Would be nice to have dslx if-blocks that can desugar
    // down to something like this automatically...
    let (sqrt_in_progress, scaled_fixed_point_x) =
    if temp <= scaled_fixed_point_x {
      (sqrt_in_progress | shifting_bit_mask,
      scaled_fixed_point_x - temp)
    } else {
      (sqrt_in_progress, scaled_fixed_point_x)
    };

    let scaled_fixed_point_x = scaled_fixed_point_x << u32:1;
    let shifting_bit_mask = shifting_bit_mask >> u32:1;

    (scaled_fixed_point_x,
     sqrt_in_progress,
     shifting_bit_mask)

  } ((scaled_fixed_point_x,      // scaled_fixed_point_x
      u32:0,                     // sqrt_in_progress
      u32:1 << u32:23 + u32:1)); // shifting_bit_mask

  // Final rounding.
  let sqrt_in_progress = if scaled_fixed_point_x != u32:0 {
    sqrt_in_progress + (u31:0 ++ sqrt_in_progress[0:1])
  } else {
    sqrt_in_progress
  };
  let scaled_fixed_point_x = (sqrt_in_progress >> u32:1) +
           ((float32::bias(exp - s8:1)) as u32 << u32:23);
  let result = float32::unflatten(scaled_fixed_point_x);

  // I don't *think* it is possible to underflow / have a subnormal result
  // here. In order to have a subnormal result, x would have to be
  // subnormal with x << sqrt(x). In this case, x would have been flushed
  // to 0. x==0 is handled below as a special case.

  // Special cases.
  // sqrt(inf) -> inf, sqrt(-inf) -> NaN (handled below along
  // with other negative numbers).
  let result = if float32::is_inf(x) { x } else { result };
  // sqrt(x < 0) -> NaN
  let result = if x.sign == u1:1 { float32::qnan() } else { result };
  // sqrt(NaN) -> NaN.
  let result = if float32::is_nan(x) { float32::qnan() } else { result };
  // x == -0 returns x rather than NaN.
  let result = if float32::is_zero_or_subnormal(x) { x } else { result };
  result
}

#[test]
fn sqrt_test() {
  // Test Special cases.
  assert_eq(fpsqrt_32(float32::zero(u1:0)),
    float32::zero(u1:0));
  assert_eq(fpsqrt_32(float32::zero(u1:1)),
    float32::zero(u1:1));
  assert_eq(fpsqrt_32(float32::inf(u1:0)),
    float32::inf(u1:0));
  assert_eq(fpsqrt_32(float32::inf(u1:1)),
    float32::qnan());
  assert_eq(fpsqrt_32(float32::qnan()),
    float32::qnan());
  assert_eq(fpsqrt_32(float32::one(u1:1)),
    float32::qnan());
  let pos_denormal = F32{sign: u1:0, bexp: u8:0, fraction: u23:99};
  assert_eq(fpsqrt_32(pos_denormal),
    float32::zero(u1:0));
  let neg_denormal = F32{sign: u1:1, bexp: u8:0, fraction: u23:99};
  assert_eq(fpsqrt_32(neg_denormal),
    float32::zero(u1:1));

  // Try some simple numbers.
  // sqrt(1).
  assert_eq(fpsqrt_32(float32::one(u1:0)),
    fpsqrt_32(float32::one(u1:0)));
  // sqrt(4).
  let four = F32 {sign: u1:0, bexp:
                  float32::bias(s8:2),
                  fraction: u23:0};
  let two = F32 {sign: u1:0, bexp:
                  float32::bias(s8:1),
                  fraction: u23:0};
  assert_eq(fpsqrt_32(four), two);
  // sqrt(9).
  let nine = F32 {sign: u1:0, bexp:
                  float32::bias(s8:3),
                  fraction: u2:0 ++ u1:1 ++ u20:0};
  let three = F32 {sign: u1:0, bexp:
                  float32::bias(s8:1),
                  fraction: u1:1 ++ u22:0};
  assert_eq(fpsqrt_32(nine), three);
  // sqrt(25).
  let twenty_five = F32 {sign: u1:0, bexp:
                  float32::bias(s8:4),
                  fraction: u4:0x9 ++ u19:0};
  let five = F32 {sign: u1:0, bexp:
                  float32::bias(s8:2),
                  fraction: u2:1 ++ u21:0};
  assert_eq(fpsqrt_32(twenty_five), five);
}
