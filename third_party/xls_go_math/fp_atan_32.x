// Copyright 2009 The Go Authors. All rights reserved.
// Copyright 2026 The XLS Authors
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This code is based on the Go atan implementation,
// available at https://golang.org/src/math/atan.go
//
// The original C code, the long comment, and the constants below were
// from http://netlib.sandia.gov/cephes/cmath/atan.c, available from
// http://www.netlib.org/cephes/cmath.tgz.
// The go code is a version of the original C.
//
// atan.c
// Inverse circular tangent (arctangent)
//
// SYNOPSIS:
// double x, y, atan();
// y = atan( x );
//
// DESCRIPTION:
// Returns radian angle between -pi/2 and +pi/2 whose tangent is x.
//
// Range reduction is from three intervals into the interval from zero to 0.66.
// The approximant uses a rational function of degree 4/5 of the form
// x + x**3 P(x)/Q(x).
//
// ACCURACY:
//                      Relative error:
// arithmetic   domain    # trials  peak     rms
//    DEC       -10, 10   50000     2.4e-17  8.3e-18
//    IEEE      -10, 10   10^6      1.8e-16  5.0e-17
// [This accuracy is for the 64-bit implementation, whereas this dslx
//  implementation is 32-bit]
//
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

// This file implements the arctangent function for single-precision
// floating point numbers.
// Note:
//  - Input denormals are treated as/flushed to 0.
//      (denormals-are-zero / DAZ).  Similarly,
//      denormal results are flushed to 0.
//  - No exception flags are raised/reported.
//  - We emit a single, canonical representation for
//      NaN (qnan) but accept all NaN respresentations
//      as input
//
// Differences from the Go implementation:
//  - The coefficients and constants are rounded to single precision.
//  - The three range-reduction cases share a single divider and a single
//    evaluation of xatan, i.e. the reduced argument is selected before
//    xatan is evaluated. For the x > tan(3*pi/8) case,
//    pi/2 - xatan(1/x) is computed as pi/2 + xatan(-1/x).
//  - The low-order bits of pi/2 (resp. pi/4) are added to xatan's result
//    before adding pi/2 (resp. pi/4), which keeps them from being rounded
//    away in single precision.

import float32;
import third_party.xls_berkeley_softfloat.fpdiv_2x32;

type F32 = float32::F32;

// Coefficients of xatan, i.e. P0..P4 and Q0..Q4 in the Go implementation,
// rounded to single precision.
const P_COEF = map(
    u32[5]:[0xbf6003fd, 0xc18142a3, 0xc2960462, 0xc2f5c5f9, 0xc281b350], float32::unflatten);
const Q_COEF = map(
    u32[5]:[0x41c6de23, 0x432506ea, 0x43d870c7, 0x43f2b1f9, 0x43428cf8], float32::unflatten);

// pi/2 and pi/4 rounded to single precision.
const PIO2 = float32::unflatten(0x3fc90fdb);
const PIO4 = float32::unflatten(0x3f490fdb);

// pi/2 = PIO2 + MOREBITS (to roughly twice single precision).
const MOREBITS = float32::unflatten(0xb33bbd2e);
const HALF_MOREBITS = float32::unflatten(0xb2bbbd2e);

// tan(3*pi/8)
const TAN3PIO8 = float32::unflatten(0x401a827a);

// 0.66
const XATAN_MAX = float32::unflatten(0x3f28f5c3);

// xatan evaluates a series valid in the range [0, 0.66].
fn xatan(x: F32) -> F32 {
    // z := x * x
    let z = float32::mul(x, x);

    // P(z) = (((P0*z+P1)*z+P2)*z+P3)*z + P4
    let p = for (idx, acc): (u32, F32) in 1..5 {
        float32::add(float32::mul(acc, z), P_COEF[idx])
    }(P_COEF[0]);

    // Q(z) = ((((z+Q0)*z+Q1)*z+Q2)*z+Q3)*z + Q4
    let q = for (idx, acc): (u32, F32) in 1..5 {
        float32::add(float32::mul(acc, z), Q_COEF[idx])
    }(float32::add(z, Q_COEF[0]));

    // z = z * P(z) / Q(z)
    let z = fpdiv_2x32::fpdiv_2x32(float32::mul(z, p), q);

    // z = x*z + x
    float32::add(float32::mul(x, z), x)
}

// satan reduces its argument (known to be positive)
// to the range [0, 0.66] and calls xatan.
fn satan(x: F32) -> F32 {
    const ZERO = float32::zero(0);
    const ONE = float32::one(0);

    let small = float32::lte_2(x, XATAN_MAX);
    let large = float32::gt_2(x, TAN3PIO8);

    // Reduced argument:
    //   x <= 0.66:          x
    //   x > tan(3*pi/8):    -1/x
    //   otherwise:          (x-1)/(x+1)
    let (numerator, divisor) =
        if large { (float32::one(1), x) } else { (float32::sub(x, ONE), float32::add(x, ONE)) };
    let quotient = fpdiv_2x32::fpdiv_2x32(numerator, divisor);

    // satan(x) = base + (xatan(reduced) + lo), where base + lo is:
    //   x <= 0.66:          0
    //   x > tan(3*pi/8):    pi/2
    //   otherwise:          pi/4
    let (reduced, base, lo) = if small {
        (x, ZERO, ZERO)
    } else if large {
        (quotient, PIO2, MOREBITS)
    } else {
        (quotient, PIO4, HALF_MOREBITS)
    };

    float32::add(base, float32::add(xatan(reduced), lo))
}

// Returns the arctangent, in radians, of x.
//
// Special cases are:
//   fp_atan_32(±0) = ±0
//   fp_atan_32(±Inf) = ±pi/2
//   fp_atan_32(NaN) = NaN
pub fn fp_atan_32(x: F32) -> F32 {
    // Flush subnormals.
    let x = float32::subnormals_to_zero(x);

    // atan(-x) = -atan(x)
    let result = satan(F32 { sign: 0, ..x });
    let result = F32 { sign: x.sign, ..result };

    // Special cases.
    if float32::is_nan(x) {
        float32::qnan()
    } else if float32::is_inf(x) {
        F32 { sign: x.sign, ..PIO2 }
    } else if float32::is_zero_or_subnormal(x) {
        x
    } else {
        result
    }
}

#[test]
fn fp_atan_32_special_cases_test() {
    let non_canonical_nan = float32::unflatten(float32::flatten(float32::qnan()) | 1);
    let subnormal = F32 { sign: 0, bexp: 0, fraction: 1 };
    let neg_subnormal = F32 { sign: 1, ..subnormal };

    // NaN
    assert_eq(fp_atan_32(float32::qnan()), float32::qnan());
    assert_eq(fp_atan_32(non_canonical_nan), float32::qnan());
    // ±Inf
    assert_eq(fp_atan_32(float32::inf(0)), PIO2);
    assert_eq(fp_atan_32(float32::inf(1)), float32::negate(PIO2));
    // ±0
    assert_eq(fp_atan_32(float32::zero(0)), float32::zero(0));
    assert_eq(fp_atan_32(float32::zero(1)), float32::zero(1));
    // Subnormals are flushed to zero.
    assert_eq(fp_atan_32(subnormal), float32::zero(0));
    assert_eq(fp_atan_32(neg_subnormal), float32::zero(1));
}

// Returns the distance between `x` and `y` in units in the last place.
// Only meaningful for finite values with the same sign.
fn ulp_distance(x: F32, y: F32) -> u32 {
    let x_bits = float32::flatten(x);
    let y_bits = float32::flatten(y);
    if x_bits > y_bits { x_bits - y_bits } else { y_bits - x_bits }
}

#[test]
fn fp_atan_32_values_test() {
    // atan(±1) = ±pi/4
    assert_eq(fp_atan_32(float32::one(0)), PIO4);
    assert_eq(fp_atan_32(float32::one(1)), float32::negate(PIO4));
    // atan(x) = x for small x.
    let tiny = float32::unflatten(0x2f800000);  // 2^-32
    assert_eq(fp_atan_32(tiny), tiny);
    assert_eq(fp_atan_32(float32::negate(tiny)), float32::negate(tiny));
    // atan(x) rounds to pi/2 for large x.
    assert_eq(fp_atan_32(float32::max_normal(0)), PIO2);
    assert_eq(fp_atan_32(float32::max_normal(1)), float32::negate(PIO2));

    // Values from each range-reduction interval, checked to be within 1 ulp of
    // the correctly rounded single-precision result: (x, atan(x)).
    let cases = [
        // x <= 0.66
        (u32:0x3e800000, u32:0x3e7adbb0), // atan(0.25)
        (u32:0x3f000000, u32:0x3eed6338), // atan(0.5)
        (u32:0xbf000000, u32:0xbeed6338), // atan(-0.5)
        (u32:0x3f28f5c3, u32:0x3f1557ef), // atan(0.66)
        // 0.66 < x <= tan(3*pi/8)
        (u32:0x3f28f5c4, u32:0x3f1557f0), // atan(nextafter(0.66, inf))
        (u32:0x3fc00000, u32:0x3f7b985f), // atan(1.5)
        (u32:0x40000000, u32:0x3f8db70d), // atan(2)
        (u32:0xc0000000, u32:0xbf8db70d), // atan(-2)
        (u32:0x401a827a, u32:0x3f96cbe4), // atan(tan(3*pi/8))
        // x > tan(3*pi/8)
        (u32:0x401a827b, u32:0x3f96cbe4), // atan(nextafter(tan(3*pi/8), inf))
        (u32:0x40400000, u32:0x3f9fe0bb), // atan(3)
        (u32:0x41200000, u32:0x3fbc4de9), // atan(10)
        (u32:0xc1200000, u32:0xbfbc4de9), // atan(-10)
        (u32:0x461c4000, u32:0x3fc90c94), // atan(10000)
    ];
    for ((x_bits, expected_bits), ()) in cases {
        let actual = fp_atan_32(float32::unflatten(x_bits));
        let expected = float32::unflatten(expected_bits);
        assert_eq(actual.sign, expected.sign);
        assert_lt(ulp_distance(actual, expected), u32:2);
    }(());
}
