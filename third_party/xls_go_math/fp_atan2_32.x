// Copyright 2009 The Go Authors. All rights reserved.
// Copyright 2026 The XLS Authors
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This code is based on the Go atan2 implementation,
// available at https://golang.org/src/math/atan2.go

// This file implements the two-argument arctangent function for
// single-precision floating point numbers.
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
//  - The quadrant correction for x < 0 is chosen based on the sign of y
//    rather than the sign of atan(y/x), so that the correct sign is
//    returned when y/x underflows to zero.
//  - pi is added in two parts (PI_LO, then PI), so that the low-order bits
//    of pi are not rounded away in single precision.

import float32;
import third_party.xls_berkeley_softfloat.fpdiv_2x32;
import third_party.xls_go_math.fp_atan_32;

type F32 = float32::F32;

// pi, pi/2, pi/4 and 3*pi/4 rounded to single precision.
const PI = float32::unflatten(0x40490fdb);
const PIO2 = float32::unflatten(0x3fc90fdb);
const PIO4 = float32::unflatten(0x3f490fdb);
const THREE_PIO4 = float32::unflatten(0x4016cbe4);

// pi = PI + PI_LO (to roughly twice single precision).
const PI_LO = float32::unflatten(0xb3bbbd2e);

// Returns `x` with the sign of `sign`.
fn copysign(x: F32, sign: u1) -> F32 { F32 { sign, ..x } }

// Returns the arctangent of y/x, in radians, using the signs of the two
// arguments to determine the quadrant of the return value.
//
// Special cases are (in order):
//   fp_atan2_32(y, NaN) = NaN
//   fp_atan2_32(NaN, x) = NaN
//   fp_atan2_32(+0, x>=0) = +0
//   fp_atan2_32(-0, x>=0) = -0
//   fp_atan2_32(+0, x<=-0) = +pi
//   fp_atan2_32(-0, x<=-0) = -pi
//   fp_atan2_32(y>0, 0) = +pi/2
//   fp_atan2_32(y<0, 0) = -pi/2
//   fp_atan2_32(+Inf, +Inf) = +pi/4
//   fp_atan2_32(-Inf, +Inf) = -pi/4
//   fp_atan2_32(+Inf, -Inf) = 3pi/4
//   fp_atan2_32(-Inf, -Inf) = -3pi/4
//   fp_atan2_32(y, +Inf) = 0
//   fp_atan2_32(y>0, -Inf) = +pi
//   fp_atan2_32(y<0, -Inf) = -pi
//   fp_atan2_32(+Inf, x) = +pi/2
//   fp_atan2_32(-Inf, x) = -pi/2
// where subnormal inputs are treated as zeros of the same sign.
pub fn fp_atan2_32(y: F32, x: F32) -> F32 {
    // Flush subnormals.
    let y = float32::subnormals_to_zero(y);
    let x = float32::subnormals_to_zero(x);

    // Call atan and determine the quadrant.
    let q = fp_atan_32::fp_atan_32(fpdiv_2x32::fpdiv_2x32(y, x));
    let result = if x.sign {
        // y > 0: q + pi
        // y < 0: q - pi
        let (pi_hi, pi_lo) =
            if y.sign { (float32::negate(PI), float32::negate(PI_LO)) } else { (PI, PI_LO) };
        float32::add(float32::add(q, pi_lo), pi_hi)
    } else {
        q
    };

    // Special cases.
    if float32::is_nan(y) || float32::is_nan(x) {
        float32::qnan()
    } else if float32::is_zero_or_subnormal(y) {
        if x.sign { copysign(PI, y.sign) } else { copysign(float32::zero(0), y.sign) }
    } else if float32::is_zero_or_subnormal(x) {
        copysign(PIO2, y.sign)
    } else if float32::is_inf(x) {
        if x.sign {
            if float32::is_inf(y) { copysign(THREE_PIO4, y.sign) } else { copysign(PI, y.sign) }
        } else {
            if float32::is_inf(y) {
                copysign(PIO4, y.sign)
            } else {
                copysign(float32::zero(0), y.sign)
            }
        }
    } else if float32::is_inf(y) {
        copysign(PIO2, y.sign)
    } else {
        result
    }
}

// Returns the distance between `x` and `y` in units in the last place.
// Only meaningful for finite values with the same sign.
fn ulp_distance(x: F32, y: F32) -> u32 {
    let x_bits = float32::flatten(x);
    let y_bits = float32::flatten(y);
    if x_bits > y_bits { x_bits - y_bits } else { y_bits - x_bits }
}

#[test]
fn fp_atan2_32_special_cases_test() {
    let non_canonical_nan = float32::unflatten(float32::flatten(float32::qnan()) | 1);
    let pos_zero = float32::zero(0);
    let neg_zero = float32::zero(1);
    let pos_inf = float32::inf(0);
    let neg_inf = float32::inf(1);
    let pos_one = float32::one(0);
    let neg_one = float32::one(1);
    let pos_subnormal = F32 { sign: 0, bexp: 0, fraction: 1 };
    let neg_subnormal = F32 { sign: 1, ..pos_subnormal };
    let neg_pi = float32::negate(PI);
    let neg_pio2 = float32::negate(PIO2);
    let neg_pio4 = float32::negate(PIO4);
    let neg_three_pio4 = float32::negate(THREE_PIO4);

    // atan2(y, NaN) = NaN
    assert_eq(fp_atan2_32(pos_one, float32::qnan()), float32::qnan());
    assert_eq(fp_atan2_32(pos_inf, non_canonical_nan), float32::qnan());
    // atan2(NaN, x) = NaN
    assert_eq(fp_atan2_32(float32::qnan(), pos_one), float32::qnan());
    assert_eq(fp_atan2_32(non_canonical_nan, pos_zero), float32::qnan());
    // atan2(±0, x>=0) = ±0
    assert_eq(fp_atan2_32(pos_zero, pos_zero), pos_zero);
    assert_eq(fp_atan2_32(neg_zero, pos_zero), neg_zero);
    assert_eq(fp_atan2_32(pos_zero, pos_one), pos_zero);
    assert_eq(fp_atan2_32(neg_zero, pos_one), neg_zero);
    assert_eq(fp_atan2_32(pos_zero, pos_inf), pos_zero);
    assert_eq(fp_atan2_32(neg_zero, pos_inf), neg_zero);
    // atan2(±0, x<=-0) = ±pi
    assert_eq(fp_atan2_32(pos_zero, neg_zero), PI);
    assert_eq(fp_atan2_32(neg_zero, neg_zero), neg_pi);
    assert_eq(fp_atan2_32(pos_zero, neg_one), PI);
    assert_eq(fp_atan2_32(neg_zero, neg_one), neg_pi);
    assert_eq(fp_atan2_32(pos_zero, neg_inf), PI);
    assert_eq(fp_atan2_32(neg_zero, neg_inf), neg_pi);
    // atan2(y>0, 0) = +pi/2, atan2(y<0, 0) = -pi/2
    assert_eq(fp_atan2_32(pos_one, pos_zero), PIO2);
    assert_eq(fp_atan2_32(pos_one, neg_zero), PIO2);
    assert_eq(fp_atan2_32(neg_one, pos_zero), neg_pio2);
    assert_eq(fp_atan2_32(neg_one, neg_zero), neg_pio2);
    assert_eq(fp_atan2_32(pos_inf, pos_zero), PIO2);
    assert_eq(fp_atan2_32(neg_inf, neg_zero), neg_pio2);
    // atan2(±Inf, +Inf) = ±pi/4
    assert_eq(fp_atan2_32(pos_inf, pos_inf), PIO4);
    assert_eq(fp_atan2_32(neg_inf, pos_inf), neg_pio4);
    // atan2(±Inf, -Inf) = ±3pi/4
    assert_eq(fp_atan2_32(pos_inf, neg_inf), THREE_PIO4);
    assert_eq(fp_atan2_32(neg_inf, neg_inf), neg_three_pio4);
    // atan2(y, +Inf) = ±0
    assert_eq(fp_atan2_32(pos_one, pos_inf), pos_zero);
    assert_eq(fp_atan2_32(neg_one, pos_inf), neg_zero);
    // atan2(y>0, -Inf) = +pi, atan2(y<0, -Inf) = -pi
    assert_eq(fp_atan2_32(pos_one, neg_inf), PI);
    assert_eq(fp_atan2_32(neg_one, neg_inf), neg_pi);
    // atan2(±Inf, x) = ±pi/2
    assert_eq(fp_atan2_32(pos_inf, pos_one), PIO2);
    assert_eq(fp_atan2_32(pos_inf, neg_one), PIO2);
    assert_eq(fp_atan2_32(neg_inf, pos_one), neg_pio2);
    assert_eq(fp_atan2_32(neg_inf, neg_one), neg_pio2);
    // Subnormals are treated as zeros of the same sign.
    assert_eq(fp_atan2_32(pos_subnormal, pos_one), pos_zero);
    assert_eq(fp_atan2_32(neg_subnormal, pos_one), neg_zero);
    assert_eq(fp_atan2_32(pos_subnormal, neg_one), PI);
    assert_eq(fp_atan2_32(neg_subnormal, neg_one), neg_pi);
    assert_eq(fp_atan2_32(pos_one, pos_subnormal), PIO2);
    assert_eq(fp_atan2_32(neg_one, neg_subnormal), neg_pio2);
    assert_eq(fp_atan2_32(pos_subnormal, neg_subnormal), PI);
}

#[test]
fn fp_atan2_32_quadrants_test() {
    let pos_one = float32::one(0);
    let neg_one = float32::one(1);

    // The four diagonals.
    assert_eq(fp_atan2_32(pos_one, pos_one), PIO4);
    assert_eq(fp_atan2_32(neg_one, pos_one), float32::negate(PIO4));
    assert_eq(fp_atan2_32(pos_one, neg_one), THREE_PIO4);
    assert_eq(fp_atan2_32(neg_one, neg_one), float32::negate(THREE_PIO4));

    // y/x underflows to zero; the result must still be in the right quadrant.
    let tiny = float32::unflatten(0x00800000);  // 2^-126
    let huge = float32::max_normal(0);
    assert_eq(fp_atan2_32(tiny, huge), float32::zero(0));
    assert_eq(fp_atan2_32(float32::negate(tiny), huge), float32::zero(1));
    assert_eq(fp_atan2_32(tiny, float32::negate(huge)), PI);
    assert_eq(fp_atan2_32(float32::negate(tiny), float32::negate(huge)), float32::negate(PI));

    // y/x overflows to infinity.
    assert_eq(fp_atan2_32(huge, tiny), PIO2);
    assert_eq(fp_atan2_32(float32::negate(huge), tiny), float32::negate(PIO2));
    // For x < 0 the result is pi + atan(-inf), which is within 1 ulp of pi/2.
    let actual = fp_atan2_32(huge, float32::negate(tiny));
    assert_eq(actual.sign, u1:0);
    assert_lt(ulp_distance(actual, PIO2), u32:2);
    let actual = fp_atan2_32(float32::negate(huge), float32::negate(tiny));
    assert_eq(actual.sign, u1:1);
    assert_lt(ulp_distance(actual, float32::negate(PIO2)), u32:2);
}

#[test]
fn fp_atan2_32_values_test() {
    // Values from each quadrant, checked to be within 1 ulp of the correctly
    // rounded single-precision result: (y, x, atan2(y, x)).
    let cases = [
        (u32:0x3f800000, u32:0x40000000, u32:0x3eed6338), // atan2(1, 2)
        (u32:0x40000000, u32:0x3f800000, u32:0x3f8db70d), // atan2(2, 1)
        (u32:0x3f800000, u32:0xc0000000, u32:0x402b6374), // atan2(1, -2)
        (u32:0x40000000, u32:0xbf800000, u32:0x40023454), // atan2(2, -1)
        (u32:0xbf800000, u32:0xc0000000, u32:0xc02b6374), // atan2(-1, -2)
        (u32:0xc0000000, u32:0xbf800000, u32:0xc0023454), // atan2(-2, -1)
        (u32:0xbf800000, u32:0x40000000, u32:0xbeed6338), // atan2(-1, 2)
        (u32:0xc0000000, u32:0x3f800000, u32:0xbf8db70d), // atan2(-2, 1)
        (u32:0x3a83126f, u32:0xc0400000, u32:0x40490a65), // atan2(0.001, -3)
        (u32:0xba83126f, u32:0xc0400000, u32:0xc0490a65), // atan2(-0.001, -3)
        (u32:0x501502f9, u32:0x3a83126f, u32:0x3fc90fdb), // atan2(1e10, 0.001)
    ];
    for ((y_bits, x_bits, expected_bits), ()) in cases {
        let actual = fp_atan2_32(float32::unflatten(y_bits), float32::unflatten(x_bits));
        let expected = float32::unflatten(expected_bits);
        assert_eq(actual.sign, expected.sign);
        assert_lt(ulp_distance(actual, expected), u32:2);
    }(());
}
