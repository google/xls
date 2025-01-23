// Copyright 2011 The Go Authors. All rights reserved.
// Copyright 2021 The XLS Authors
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// ====================================================
// Cephes Math Library Release 2.8:  June, 2000
// Copyright 1984, 1987, 1989, 1992, 2000 by Stephen L. Moshier
//
// The readme file at http://netlib.sandia.gov/cephes/ says:
//    Some software in this archive may be from the book _Methods and
// Programs for Mathematical Functions_ (Prentice-Hall or Simon & Schuster
// International, 1989) or from the Cephes Mathematical Library, a
// commercial product. In either event, it is copyrighted by the author.
// What you see here may be used freely but it comes with no support or
// guarantee.
//
//   The two known misprints in the book are repaired here in the
// source listings for the gamma function and the incomplete beta
// integral.
//
//   Stephen L. Moshier
//   moshier@na-net.ornl.gov
// ====================================================


// ====================================================
//      Circular sine
//
// SYNOPSIS:
//
// double x, y, sin();
// y = sin( x );
//
// DESCRIPTION:
//
// Range reduction is into intervals of pi/4.  The reduction error is nearly
// eliminated by contriving an extended precision modular arithmetic.
//
// Two polynomial approximating functions are employed.
// Between 0 and pi/4 the sine is approximated by
//      x  +  x**3 P(x**2).
// Between pi/4 and pi/2 the cosine is represented as
//      1  -  x**2 Q(x**2).
//
// ACCURACY:
//  ...
//
// Partial loss of accuracy begins to occur at x = 2**30 = 1.074e9.  The loss
// is not gradual, but jumps suddenly to about 1 part in 10e7.  Results may
// be meaningless for x > 2**49 = 5.6e14.
//
//      cos.c
//
//      Circular cosine
//
// SYNOPSIS:
//
// double x, y, cos();
// y = cos( x );
//
// DESCRIPTION:
//
// Range reduction is into intervals of pi/4.  The reduction error is nearly
// eliminated by contriving an extended precision modular arithmetic.
//
// Two polynomial approximating functions are employed.
// Between 0 and pi/4 the cosine is approximated by
//      1  -  x**2 Q(x**2).
// Between pi/4 and pi/2 the sine is represented as
//      x  +  x**3 P(x**2).
//
// ====================================================

// This file implements the cos function for double-recision
// floating point numbers using a taylor's series approximation.
// Note:
//  - Input denormals are treated as/flushed to 0.
//      (denormals-are-zero / DAZ).

import std;
import float32;

import third_party.xls_go_math.fp_trig_reduce;

type F32 = float32::F32;

const SIN_COEF =
      map(u32[6]:[0x2f2ec7e9, 0xb2d72f2d, 0x3638ef1b,
                  0xb9500d01, 0x3c088889, 0xbe2aaaab],
          float32::unflatten);
const COS_COEF =
      map(u32[6]:[0xad47d24d, 0x310f74ec, 0xb493f27c,
                  0x37d00d01, 0xbab60b61, 0x3d2aaaab],
          float32::unflatten);

// Helper function that calculates taylor
// series approximation of cos in range [0, Pi/4).
fn cos_taylor(z_sq: F32) -> F32 {
  const ONE = float32::one(u1:0);

  // cos = 1.0
  //       - 0.5*z_sq
  //       + z_sq*z_sq*(
  //          (
  //            (
  //              (
  //                (
  //                  (_cos[0]*z_sq)
  //                  +_cos[1]
  //                )*z_sq
  //                +_cos[2]
  //              )*z_sq
  //              +_cos[3]
  //            )*z_sq
  //            +_cos[4]
  //          )*z_sq
  //          +_cos[5]
  //        )

  let big_product = float32::mul(
                      z_sq,
                      float32::mul(
                        z_sq,

                        for(idx, acc): (u3, F32)
                          in range (u3:0, u3:5) {
                          float32::add(
                            float32::mul(
                              acc,
                              z_sq
                            ),
                            COS_COEF[idx+u3:1]
                          )
                        }(COS_COEF[u3:0])
                      )
                    );

  let half_z_sq = F32{bexp: std::bounded_minus_1(z_sq.bexp), ..z_sq};
  float32::sub(
    float32::add(
      ONE,
      big_product
    ),
    half_z_sq
  )
}

// Helper function that calculates taylor
// series approximation of sin in range [0, Pi/4)
fn sin_taylor(z:F32, z_sq: F32) -> F32 {
  // sin = z
  //       + z*z_sq*(
  //          (
  //            (
  //              (
  //                (
  //                  (_sin[0]*z_sq)
  //                  +_sin[1]
  //                )*z_sq
  //                +_sin[2]
  //              )*z_sq
  //              +_sin[3]
  //            )*z_sq
  //            +_sin[4]
  //          )*z_sq
  //          +_sin[5]
  //        )

  let big_product = float32::mul(
                      z,
                      float32::mul(
                        z_sq,

                        for(idx, acc): (u3, F32)
                          in range (u3:0, u3:5) {
                          float32::add(
                            float32::mul(
                              acc,
                              z_sq
                            ),
                            SIN_COEF[idx+u3:1]
                          )
                        }(SIN_COEF[u3:0])
                      )
                    );


  float32::add(
    z,
    big_product
  )
}

// Returns (sin(x), cos(x)), where x is in radians.
pub fn fp_sincos_32(x: F32) -> (F32, F32) {
  // Flush subnormals.
  let x = float32::subnormals_to_zero(x);
  let x_in_w_flush = x;

  // Make argument positive.
  let sin_sign = x.sign;
  let cos_sign = u1:0;
  let x = F32{sign: u1:0, ..x};

  // Reduce x to be in range[0, 2*Pi)
  let (j, z) = fp_trig_reduce::fp_trig_reduce_32(x);

  // Reflect across x axis.
  let (j, sin_sign, cos_sign) =
       if j > u3:3 {
         (j - u3:4, !sin_sign, !cos_sign)
       } else {
         (j, sin_sign, cos_sign)
       };

  // Adjust cos sign across y axis.
  let cos_sign =
              if (j > u3:1) { !cos_sign }
              else { cos_sign };

  let z_sq = float32::mul(z, z);
  let cos = cos_taylor(z_sq);
  let sin = sin_taylor(z, z_sq);
  // Swap sin and cos if, after trig reduction and
  // reflection across the x-axis, x is in the range
  // [PI/4, 3*PI/4).
  let (sin, cos) =
                 if (j == u3:1 || j == u3:2) { (cos, sin) }
                 else { (sin, cos) };

  let cos = if cos_sign { F32{sign: !cos.sign, ..cos} }
            else { cos };

  let sin =  if sin_sign { F32{sign: !sin.sign, ..sin} }
             else { sin };
  let result = (sin, cos);

  // Special cases.
  // x == +/-inf, NaN --> NaN
  let result =
            if (float32::is_nan(x) || float32::is_inf(x)) { (float32::qnan(), float32::qnan()) }
            else { result };
  // x == 0
  let result =
            if (float32::is_zero_or_subnormal(x)) { (x_in_w_flush, float32::one(u1:0)) }
            else { result };
  result
}

#[test]
fn test_fp_sincos_32() {
  let non_canonical_nan = float32::unflatten(
    float32::flatten(float32::qnan()) | u32:1);
  let denormal = F32{sign: u1:0, bexp: u8:0, fraction: u23:1};

  // Inf.
  assert_eq(fp_sincos_32(float32::inf(u1:0)),
            (float32::qnan(), float32::qnan()));
  assert_eq(fp_sincos_32(float32::inf(u1:1)),
            (float32::qnan(), float32::qnan()));
  // NaN
  assert_eq(fp_sincos_32(non_canonical_nan),
            (float32::qnan(), float32::qnan()));
  // 0
  assert_eq(fp_sincos_32(float32::zero(u1:0)),
            (float32::zero(u1:0), float32::one(u1:0)));
  assert_eq(fp_sincos_32(float32::zero(u1:1)),
            (float32::zero(u1:1), float32::one(u1:0)));

  // denormal
  assert_eq(fp_sincos_32(denormal),
            (float32::zero(u1:0), float32::one(u1:0)));
}

// Returns sin(x), where x is in radians.
pub fn fp_sin_32(x: F32) -> F32 {
  let (sin, _) = fp_sincos_32(x);
  sin
}

#[test]
fn test_fp_sin_32() {
  // Just testing that we got the wiring right.
  assert_eq(fp_sin_32(float32::zero(u1:0)),
            float32::zero(u1:0));
}

// Returns cos(x), where x is in radians.
pub fn fp_cos_32(x: F32) -> F32 {
  let (_, cos) = fp_sincos_32(x);
  cos
}

#[test]
fn test_fp_cos_32() {
  // Just testing that we got the wiring right.
  assert_eq(fp_cos_32(float32::zero(u1:0)),
            float32::one(u1:0));
}
