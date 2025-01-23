// Copyright 2009 The Go Authors. All rights reserved.
// Copyright 2021 The XLS Authors
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This code is based on the go double sqrt implementation,
// available at https://golang.org/src/math/exp.go
// This implementation is in turn based on
// FreeBSD's /usr/src/lib/msun/src/e_exp.c
// ====================================================
// Copyright (C) 2004 by Sun Microsystems, Inc. All rights reserved.
//
// Permission to use, copy, modify, and distribute this
// software is freely granted, provided that this notice
// is preserved.
//
// exp(x)
// Returns the exponential of x.
//
// Method
//   1. Argument reduction:
//      Reduce x to an r so that |r| <= 0.5*ln2 ~ 0.34658.
//      Given x, find r and integer k such that
//
//               x = k*ln2 + r,  |r| <= 0.5*ln2.
//
//      Here r will be represented as r = hi-lo for better
//      accuracy.
//
//   2. Approximation of exp(r) by a special rational function on
//      the interval [0,0.34658]:
//      Write
//          R(r**2) = r*(exp(r)+1)/(exp(r)-1) = 2 + r*r/6 - r**4/360 + ...
//      We use a special Remez algorithm on [0,0.34658] to generate
//      a polynomial of degree 5 to approximate R. The maximum error
//      of this polynomial approximation is bounded by 2**-59
//      [This error is for the 64-bit implementation, whereas this dslx
//       implementation is 32-bit]. In other words,
//          R(z) ~ 2.0 + P1*z + P2*z**2 + P3*z**3 + P4*z**4 + P5*z**5
//      (where z=r*r, and the values of P1 to P5 are listed below)
//      and
//          |                  5          |     -59
//          | 2.0+P1*z+...+P5*z   -  R(z) | <= 2
//          |                             |
//      The computation of exp(r) thus becomes
//                             2*r
//              exp(r) = 1 + -------
//                            R - r
//                                 r*R1(r)
//                     = 1 + r + ----------- (for better accuracy)
//                                2 - R1(r)
//      where
//                               2       4             10
//              R1(r) = r - (P1*r  + P2*r  + ... + P5*r   ).
//
//   3. Scale back to obtain exp(x):
//      From step 1, we have
//         exp(x) = 2**k * exp(r)
//
// Special cases:
//      exp(INF) is INF, exp(NaN) is NaN;
//      exp(-INF) is 0, and
//      for finite argument, only exp(0)=1 is exact.
//
// Accuracy:
//      according to an error analysis, the error is always less than
//      1 ulp (unit in the last place).
//      [This accuracy is for the 64-bit implementation, whereas this dslx
//       implementation is 32-bit]
//
// Misc. info.
//      For IEEE double
//          if x >  7.09782712893383973096e+02 then exp(x) overflow
//          if x < -7.45133219101941108420e+02 then exp(x) underflow
//      [This is for the 64-bit implementation, whereas this dslx
//       implementation is 32-bit]
//
// Constants:
// The hexadecimal values are the intended ones for the following
// constants. The decimal values may be used, provided that the
// compiler will convert from decimal to binary accurately enough
// to produce the hexadecimal values shown.
// ====================================================

// This file implements floating point exp,
// which calcualtes e^x.
// Note:
//  - Input denormals are treated as/flushed to 0.
//      (denormals-are-zero / DAZ).  Similarly,
//      denormal results are flushed to 0.
//  - No exception flags are raised/reported.
//  - We emit a single, canonical representation for
//      NaN (qnan) but accept all NaN respresentations
//      as input

import std;
import float32;
import third_party.xls_berkeley_softfloat.fpdiv_2x32;

type F32 = float32::F32;


const LN2HI = float32::unflatten(u32:0x3f317218);
const LN2LO = float32::unflatten(u32:0x2f51cf7a);
const LOG2E = float32::unflatten(u32:0x3fb8aa3b);

const P1 = float32::unflatten(u32:0x3e2aaaab);
const P2 = float32::unflatten(u32:0xbb360b61);
const P3 = float32::unflatten(u32:0x388ab355);
const P4 = float32::unflatten(u32:0xb5ddea0e);
const P5 = float32::unflatten(u32:0x3331bb4c);

// Returns e^(hi-lo) * 2^k, where |r| <= ln(2)/2
fn expmulti(hi: F32, lo: F32, k: s32) -> F32 {
  const ONE = float32::one(u1:0);
  const TWO = F32{ sign: u1:0,
                   bexp: float32::bias(s8:1),
                   fraction: u23:0
                  };

  let range = float32::sub(hi, lo);
  let range_sq = float32::mul(range, range);

  // c := r - t*(P1+t*(P2+t*(P3+t*(P4+t*P5))))
  let c = float32::mul(range_sq, P5);
  let c = float32::add(c, P4);
  let c = float32::mul(range_sq, c);
  let c = float32::add(c, P3);
  let c = float32::mul(range_sq, c);
  let c = float32::add(c, P2);
  let c = float32::mul(range_sq, c);
  let c = float32::add(c, P1);
  let c = float32::mul(range_sq, c);
  let c = float32::sub(range, c);

  // y := 1 - ((lo - (r*c)/(2-c)) - hi)
  let numerator = float32::mul(range, c);
  let divisor = float32::sub(TWO, c);
  let div = fpdiv_2x32::fpdiv_2x32(numerator, divisor);
  let y = float32::sub(lo, div);
  let y = float32::sub(y, hi);
  let y = float32::sub(ONE, y);

  float32::ldexp(y, k)
}

// Returns e^x
pub fn fpexp_32(x: F32) -> F32 {
  const ZERO = float32::zero(u1:0);

  // Flush subnormals.
  let x = float32::subnormals_to_zero(x);

  let scaled_x = float32::mul(LOG2E, x);
  let signed_half = F32{ sign: x.sign,
                         bexp: float32::bias(s8:-1),
                         fraction:  u23:0
                        };
  let fp_k = float32::add(scaled_x, signed_half);
  let k = float32::cast_to_fixed<u32:32>(fp_k);

  // Reduce
  // TODO(jbaileyhandle): Cheaper to truncate fp_k directly?
  let fp_truncated_k = float32::cast_from_fixed_using_rne(k);
  let hi = float32::mul(LN2HI, fp_truncated_k);
  let hi = float32::sub(x, hi);
  let lo = float32::mul( LN2LO, fp_truncated_k);

  // Compute
  let result = expmulti(hi, lo, k);

  // Handle underflow.
  let result =
    if ((float32::is_nan(result) || float32::is_inf(result)) && x.sign) {
      float32::zero(u1:0)
    } else {
      result
    };
  // Handle overflow.
  let result = if (float32::is_nan(result) && !x.sign) { float32::inf(u1:0) }
               else { result };

  // Special cases.
  // exp(NaN) -> NaN
  let result = if float32::is_nan(x) { float32::qnan() }
               else { result };
  // exp(Inf) -> Inf
  let result = if (float32::is_inf(x) && x.sign == u1:0) { x }
               else { result };
  // exp(-Inf) -> 0
  let result = if (float32::is_inf(x) && x.sign == u1:1) { ZERO }
               else { result };
  result
}


#[test]
fn fpexp_32_test() {
  // Special cases.
  assert_eq(float32::qnan(), fpexp_32(float32::qnan()));
  assert_eq(float32::inf(u1:0), fpexp_32(float32::inf(u1:0)));
  assert_eq(float32::zero(u1:0), fpexp_32(float32::inf(u1:1)));

  // Very big positive input.
  let input = float32::unflatten(u32:0x5d5538f0);
  assert_eq(float32::inf(u1:0), fpexp_32(input));

  // Very big negative input.
  let input = float32::unflatten(u32:0xf12d483f);
  assert_eq(float32::zero(u1:0), fpexp_32(input));
}
