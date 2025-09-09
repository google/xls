// Copyright 2023 The XLS Authors
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

#![feature(type_inference_v2)]

// This floating point adder splits the computation into two parallel paths
// for lower latency. Code based on papers by P.M. Seidel and G. Even:
//   * "On the design of fast IEEE floating-point adders" (2001)
//   * "Delay-optimized implementation of IEEE floating-point addition" (2004)

// --
// Note, this code currently is INCOMPLETE (parts of r_path() missing,
// and is replaced with regular apfloat::add(). The leading zero anticipator
// actually does not run in parallel but runs after the values are computed.
// (currently harder to implement, would be easier once
// https://github.com/google/xls/issues/510 is available)).
// This code is WIP. Don't use yet.

// Right now, the code resembles in parts how it would be done in Verilog for
// easier review and comparison.
// After complete implementation, the code will be refactored to be more
// canonical DSLX.

import std;
import apfloat;

// Given two numbers a, b, returns (a + b) and incremented sum (a + b + 1)
fn compound_adder<WIDTH: u32>(a: uN[WIDTH], b: uN[WIDTH]) -> (uN[WIDTH], uN[WIDTH]) {
    // TODO(hzeller): 2023-10-15 use an efficient implementation if synthesis
  // tool can't deduce this.
  (a + b, a + b + uN[WIDTH]:1)
}

#[test]
fn compound_adder_test() {
    assert_eq(compound_adder(u8:8, u8:19), (u8:27, u8:28));
    assert_eq(compound_adder(u8:253, u8:1), (u8:254, u8:255));
}

// Calculate difference of two positive values and return values in sign-magnitude
// form. Returns sign-magnitude tuple (|a| - |b| <= 0, abs(|a| - |b|)).
// Note, this returns -0 if (a == b), which is used in our application, which is good
// for testing if strictly |a| > |b|.
fn sign_magnitude_difference<WIDTH: u32>(a: uN[WIDTH], b: uN[WIDTH]) -> (bool, uN[WIDTH]) {
    // 1's complement internally, then use the following observation.
    //    abs(|A| - |B|) =   |A| + |~B| + 1 iff |A| - |B| >  0
    //                     ~(|A| + |~B|)    iff |A| - |B| <= 0
    // We use the compound_adder() to efficiently prepare sum + 1 to select result

    type WidthWithCarry = uN[WIDTH + u32:1];
    let (sum, incremented_sum) = compound_adder(a as WidthWithCarry, !b as WidthWithCarry);
    let a_is_less_equal: bool = !sum[-1:];  // Sign bit overflow in the carry

    let abs_difference = if a_is_less_equal { !sum } else { incremented_sum };
    (a_is_less_equal, abs_difference as uN[WIDTH])
}

#[test]
fn sign_magnitude_difference_test() {
    // The way we like our ones differencer is to yield a -0 for (a == b), i.e. result.0==(a >= b)
    assert_eq(sign_magnitude_difference(u8:0, u8:0), (true, u8:0));
    assert_eq(sign_magnitude_difference(u8:42, u8:42), (true, u8:0));
    assert_eq(sign_magnitude_difference(u8:255, u8:255), (true, u8:0));

    // Make sure this works for very small width; exhaustive for u1
    assert_eq(sign_magnitude_difference(u1:0, u1:0), (true, u1:0));
    assert_eq(sign_magnitude_difference(u1:1, u1:1), (true, u1:0));
    assert_eq(sign_magnitude_difference(u1:0, u1:1), (true, u1:1));
    assert_eq(sign_magnitude_difference(u1:1, u1:0), (false, u1:1));

    // Exhaustive for u2
    for (left, _): (u2, ()) in u2:0..u2:3 {
        for (right, _): (u2, ()) in u2:0..u2:3 {
            assert_eq(sign_magnitude_difference(left, right),
                      ((right >= left), if right >= left { right - left } else { left - right }));
        }(());
    }(());

    // Exhaustive for u8
    for (left, _): (u8, ()) in u8:0..u8:255 {
        for (right, _): (u8, ()) in u8:0..u8:255 {
            assert_eq(sign_magnitude_difference(left, right),
                      ((right >= left), if right >= left { right - left } else { left - right }));
        }(());
    }(());

    // Close to overflow is handled correctly
    assert_eq(sign_magnitude_difference(u8:255, u8:0), (false, u8:255));
    assert_eq(sign_magnitude_difference(u8:255, u8:5), (false, u8:250));
    assert_eq(sign_magnitude_difference(u8:0, u8:255), (true, u8:255));
    assert_eq(sign_magnitude_difference(u8:5, u8:255), (true, u8:250));
}

struct ZeroAnticipatorPrediction<SHIFT_WIDTH: u32> {
    quasinormalization_shift: uN[SHIFT_WIDTH],
    is_nonzero: bool,
    needs_correction: bool,
}

// Determine predicted leading zeros given the two operands of the difference.
// This can run in parallel to actually doing the difference
// TODO(hzeller): 2023-10-20 actually do this in parallel, for now, do that sequentially
//                and re-do the difference here to then call clz() on it.
fn leading_zero_anticipator<WIDTH: u32,
                            SHIFT_WIDTH: u32 = {std::clog2(WIDTH)}>(minuend: uN[WIDTH],
                                                                    subtrahend: uN[WIDTH])
    -> ZeroAnticipatorPrediction<SHIFT_WIDTH> {
    // TODO(hzeller): this should be calculated with the leading zero anticipator in parallel.
    // (Easier to implement after https://github.com/google/xls/issues/510  done)
    // For now: count zeroes after difference has arrived.
    // For better abstraction We can just do the same diff operation here again, it is CSE'ed.
    let (_, difference) = sign_magnitude_difference(minuend, subtrahend);
    ZeroAnticipatorPrediction {
        quasinormalization_shift: clz(difference) as uN[SHIFT_WIDTH],
        is_nonzero: difference != uN[WIDTH]:0,
        needs_correction: false  // Our pretend zero anticipator is always right.
    }
}

#[test]
fn leading_zero_anticipator_test() {
    assert_eq(ZeroAnticipatorPrediction {
                  quasinormalization_shift: u4:5, is_nonzero: true, needs_correction: false
    }, leading_zero_anticipator(u12:0x0ff, u12:0x08f));
    assert_eq(ZeroAnticipatorPrediction {
        quasinormalization_shift: u4:12, is_nonzero: false, needs_correction: false
    }, leading_zero_anticipator(u12:0x0ff, u12:0x0ff));
}

// The N-path: effective subtraction, with exponent difference {-1, 0, +1}.
// This can result in a large cancellation.
fn n_path<EXP_SZ: u32, FRACTION_SZ: u32>(a: apfloat::APFloat<EXP_SZ, FRACTION_SZ>,
                                         b: apfloat::APFloat<EXP_SZ, FRACTION_SZ>)
                                         -> (apfloat::APFloat<EXP_SZ, FRACTION_SZ>, bool) {
    // We assume to be in the near path with exponents difference in {-1, 0, +1}
    // So we only have to look at the last two bits
    // TODO(hzeller): 2023-10-21 would using 1's complement improve speed ?
    let small_exponent_difference = a.bexp as u2 - b.bexp as u2;

    // Assuming that we're in the valid range of { -1, 0, +1 } exponent difference, observing
    // -1 means that b definitely has the larger exponent.
    // Comment rmlarsen@ next line: [1+:bool] pretty obscure syntax.
    // Filed https://github.com/google/xls/issues/1180 to allow more readable ways
    let b_has_larger_exponent = small_exponent_difference[1+:bool];

    // Re-add hidden bit
    let a_significand = u1:1 ++ a.fraction;
    let b_significand = u1:1 ++ b.fraction;

    // Select minuend and shift for alignment.
    // Preshift (<<1)
    let (minuend_sign, minuend_exponent, minuend_significand) = if b_has_larger_exponent {
        (b.sign, b.bexp, b_significand ++ u1:0)
    } else {
        (a.sign, a.bexp, a_significand ++ u1:0)
    };

    let subtrahend_significand = if b_has_larger_exponent {
        u1:0 ++ a_significand  // Align: shift one right.
    } else {
        // https://github.com/google/xls/issues/1180 for discussion of improved syntax for next
        let need_align_right = small_exponent_difference[0+:bool];  // expa > expb
        if need_align_right { u1:0 ++ b_significand } else { b_significand ++ u1:0 }
    };

    // The difference uses two bits more than our significand (or FRACTION_SZ + 3)
    let (diff_sign, diff_magnitude) = sign_magnitude_difference(minuend_significand,
                                                                 subtrahend_significand);

    // Predict leading zeroes
    let lz_prediction = leading_zero_anticipator(minuend_significand, subtrahend_significand);

    // Use leading zero predition results to make decisions.
    let exact_cancellation = !lz_prediction.is_nonzero && !lz_prediction.needs_correction;
    let near_significand = diff_magnitude << lz_prediction.quasinormalization_shift;
    let near_fraction = if lz_prediction.needs_correction {
        near_significand[2+:uN[FRACTION_SZ]]
    } else {
        near_significand[1+:uN[FRACTION_SZ]]
    };

    let near_exponent = if exact_cancellation {
        uN[EXP_SZ]:0
    } else {
        minuend_exponent - lz_prediction.quasinormalization_shift as uN[EXP_SZ] -
            lz_prediction.needs_correction as uN[EXP_SZ]
    };

    let use_far_path = !diff_sign && diff_magnitude[-1:];
    let near_sign = if exact_cancellation { u1:0 } else { diff_sign ^ minuend_sign };

    type ResultType = apfloat::APFloat<EXP_SZ, FRACTION_SZ>;

    (ResultType { sign: near_sign, bexp: near_exponent, fraction: near_fraction }, use_far_path)
}

#[test]
fn n_path_test() {
    type F32 = apfloat::APFloat<8, 23>;

    // Almost three minus one brings us in range ..[1,2)
    let a = F32 { sign: u1:0, bexp: u8:0x80, fraction: u23:0x3fffff };  // 2.9999998
    let b = F32 { sign: u1:1, bexp: u8:0x7f, fraction: u23:0x000000 };  // -1
    let expected = F32 { sign: u1:0, bexp: u8:0x7f, fraction: u23:0x7ffffe };  // a + b = almosttwo
    assert_eq(n_path(a, b), (expected, false));

    // However 3 - 1 returns exactly 2, so that would be _outside_ range [1,2), so we
    // should use far path.
    let a = F32 { sign: u1:0, bexp: u8:0x80, fraction: u23:0x400000 };  // 3.0
    let b = F32 { sign: u1:1, bexp: u8:0x7f, fraction: u23:0x000000 };  // -1
    let expected = F32 { sign: u1:0, bexp: u8:0x80, fraction: u23:0x000000 };  // a + b = 2
    assert_eq(n_path(a, b), (expected, true));  // <- use far

    let a = F32 { sign: u1:1, bexp: u8:0x7f, fraction: u23:0x000000 };  // -1
    let b = F32 { sign: u1:0, bexp: u8:0x80, fraction: u23:0x000000 };  // 2
    let expected = F32 { sign: u1:0, bexp: u8:0x7f, fraction: u23:0x000000 };  // a + b = 1
    assert_eq(n_path(a, b), (expected, false));

    // +/- 1 exponent offset
    let a = F32 { sign: u1:0, bexp: u8:0x87, fraction: u23:0x210000 };  // 322
    let b = F32 { sign: u1:1, bexp: u8:0x86, fraction: u23:0x540000 };  // -212
    let expected = F32 { sign: u1:0, bexp: u8:0x85, fraction: u23:0x5c0000 };  // a + b = 110
    assert_eq(n_path(a, b), (expected, false));

    let a = F32 { sign: u1:0, bexp: u8:0x86, fraction: u23:0x540000 };  // 212
    let b = F32 { sign: u1:1, bexp: u8:0x87, fraction: u23:0x210000 };  // -322
    let expected = F32 { sign: u1:1, bexp: u8:0x85, fraction: u23:0x5c0000 };  // a + b = -110
    assert_eq(n_path(a, b), (expected, false));

    // Lots of cancellation
    let a = F32 { sign: u1:0, bexp: u8:0x96, fraction: u23:0x000000 };  // 8_388_608
    let b = F32 { sign: u1:1, bexp: u8:0x95, fraction: u23:0x7fffff };  // -8_388_607.5
    let expected = F32 { sign: u1:0, bexp: u8:0x7e, fraction: u23:0x000000 };  // a + b = 0.5
    assert_eq(n_path(a, b), (expected, false));

    // More bits than fit into the representation, only coarse-grained resoolution
    let a = F32 { sign: u1:0, bexp: u8:0x96, fraction: u23:0x7ffffb };  // 16_777_211
    let b = F32 { sign: u1:1, bexp: u8:0x96, fraction: u23:0x7ffffe };  // -16_777_214
    let expected = F32 { sign: u1:1, bexp: u8:0x80, fraction: u23:0x400000 };  // a + b = -3
    assert_eq(n_path(a, b), (expected, false));

    let a = F32 { sign: u1:0, bexp: u8:0x96, fraction: u23:0x7ffffe };  // 16_777_214
    let b = F32 { sign: u1:1, bexp: u8:0x96, fraction: u23:0x7ffffb };  // -16_777_211
    let expected = F32 { sign: u1:0, bexp: u8:0x80, fraction: u23:0x400000 };  // a + b = 3
    assert_eq(n_path(a, b), (expected, false));

    // Also expected: limited floating point resolution, so the following arrives at -2
    let a = F32 { sign: u1:0, bexp: u8:0x97, fraction: u23:0x000000 };  // 16_777_217
    let b = F32 { sign: u1:1, bexp: u8:0x97, fraction: u23:0x000001 };  // -16_777_218
    let expected = F32 { sign: u1:1, bexp: u8:0x80, fraction: u23:0x000000 };  // a + b = -2
    assert_eq(n_path(a, b), (expected, false));

    // Full cancellation, arriving at zero
    let a = F32 { sign: u1:0, bexp: u8:0x7f, fraction: u23:0x000000 };  // 1
    let b = F32 { sign: u1:1, bexp: u8:0x7f, fraction: u23:0x000000 };  // -1
    let expected = F32 { sign: u1:0, bexp: u8:0x00, fraction: u23:0x000000 };  // a + b = 0
    assert_eq(n_path(a, b), (expected, false));

    let a = F32 { sign: u1:0, bexp: u8:0x89, fraction: u23:0x1a4000 };  // 1234
    let b = F32 { sign: u1:1, bexp: u8:0x89, fraction: u23:0x1a4000 };  // -1234
    let expected = F32 { sign: u1:0, bexp: u8:0x00, fraction: u23:0x000000 };  // a + b = 0
    assert_eq(n_path(a, b), (expected, false));

    // Various fractional values
    let a = F32 { sign: u1:0, bexp: u8:0x7f, fraction: u23:0x000000 };  // 1
    let b = F32 { sign: u1:1, bexp: u8:0x7e, fraction: u23:0x000000 };  // -0.5
    let expected = F32 { sign: u1:0, bexp: u8:0x7e, fraction: u23:0x000000 };  // a + b = 0.5
    assert_eq(n_path(a, b), (expected, false));

    let a = F32 { sign: u1:0, bexp: u8:0x7f, fraction: u23:0x000000 };  // 1
    let b = F32 { sign: u1:1, bexp: u8:0x7e, fraction: u23:0x400000 };  // -0.75
    let expected = F32 { sign: u1:0, bexp: u8:0x7d, fraction: u23:0x000000 };  // a + b = 0.25
    assert_eq(n_path(a, b), (expected, false));

    let a = F32 { sign: u1:0, bexp: u8:0x7f, fraction: u23:0x000000 };  // 1
    let b = F32 { sign: u1:1, bexp: u8:0x7e, fraction: u23:0x200000 };  // -0.625
    let expected = F32 { sign: u1:0, bexp: u8:0x7d, fraction: u23:0x400000 };  // a + b = 0.375
    assert_eq(n_path(a, b), (expected, false));

    let a = F32 { sign: u1:0, bexp: u8:0x7f, fraction: u23:0x000000 };  // 1
    let b = F32 { sign: u1:1, bexp: u8:0x7f, fraction: u23:0x200000 };  // -1.25
    let expected = F32 { sign: u1:1, bexp: u8:0x7d, fraction: u23:0x000000 };  // a + b = -0.25
    assert_eq(n_path(a, b), (expected, false));

    // Some corner cases
    let a = F32 { sign: u1:0, bexp: u8:0x7f, fraction: u23:0x000001 };  // 1 + ulp(1)
    let b = F32 { sign: u1:1, bexp: u8:0x7f, fraction: u23:0x000000 };  // -1
    let expected = F32 { sign: u1:0, bexp: u8:0x68, fraction: u23:0x000000 };  // a + b = ulp(1)
    assert_eq(n_path(a, b), (expected, false));

    let a = F32 { sign: u1:0, bexp: u8:0x01, fraction: u23:0x000000 };  // MIN_NORM
    let b = F32 { sign: u1:1, bexp: u8:0x01, fraction: u23:0x000000 };  // -MIN_NORM
    let expected = F32 { sign: u1:0, bexp: u8:0x00, fraction: u23:0x000000 };  // a + b = 0
    assert_eq(n_path(a, b), (expected, false));

    let a = F32 { sign: u1:1, bexp: u8:0x28, fraction: u23:0x6b0000 };  // 'h946b0000
    let b = F32 { sign: u1:0, bexp: u8:0x28, fraction: u23:0x0e0000 };  // 'h140e0000
    let expected = F32 { sign: u1:1, bexp: u8:0x27, fraction: u23:0x3a0000 };  // a + b = 'h93ba0000
    assert_eq(n_path(a, b), (expected, false));

    let a = F32 { sign: u1:1, bexp: u8:0x28, fraction: u23:0x6b3b67 };  // 'h946b3b67
    let b = F32 { sign: u1:0, bexp: u8:0x28, fraction: u23:0x0e4c12 };  // 'h140e4c12
    let expected = F32 { sign: u1:1, bexp: u8:0x27, fraction: u23:0x39deaa };  // a + b = 'h93b9deaa
    assert_eq(n_path(a, b), (expected, false));
}

fn r_exponent_differencer
    <EXP_SZ: u32, FRACTION_SZ: u32,
     SIGNIFICAND_WIDTH: u32              = {u32:1 + FRACTION_SZ},  // + hidden bit
     LARGEST_MEANINGFUL_SHIFT_WIDTH: u32 = {std::clog2(SIGNIFICAND_WIDTH + u32:1)},
     UPPER_EXPONENT_WIDTH: u32           = {EXP_SZ - LARGEST_MEANINGFUL_SHIFT_WIDTH}>
    (a: apfloat::APFloat<EXP_SZ, FRACTION_SZ>, b: apfloat::APFloat<EXP_SZ, FRACTION_SZ>)
    -> (bool, uN[UPPER_EXPONENT_WIDTH], uN[LARGEST_MEANINGFUL_SHIFT_WIDTH], bool) {
    type ShiftableExponent = uN[LARGEST_MEANINGFUL_SHIFT_WIDTH];
    type UpperExponent = uN[UPPER_EXPONENT_WIDTH];

    let (is_negative, difference) = sign_magnitude_difference(a.bexp, b.bexp);
    // Split into upper and lower half
    let (upper, lower) = (difference[-UPPER_EXPONENT_WIDTH as s32:],
                          difference[:-UPPER_EXPONENT_WIDTH as s32]);
    let use_far = or_reduce(difference[1:]);

    (is_negative, upper, lower, use_far)
}

#[test]
fn r_exponent_differencer_test() {
    type F32 = apfloat::APFloat<8, 23>;
    type UpperBits = uN[3];
    type LowerBits = uN[5];  // clog2(1 + 23 + 1)

    let a = F32 { bexp: u8:42, sign: u1:0, fraction: u23:0 };
    let b = F32 { bexp: u8:48, sign: u1:0, fraction: u23:0 };
    assert_eq(r_exponent_differencer(a, b), (true, UpperBits:0, LowerBits:6, true));

    let a = F32 { bexp: u8:48, sign: u1:0, fraction: u23:0 };
    let b = F32 { bexp: u8:42, sign: u1:0, fraction: u23:0 };
    assert_eq(r_exponent_differencer(a, b), (false, UpperBits:0, LowerBits:6, true));
}

// R path works on all places where large cancellation can not occur: when the exponents
// differ at least by 2 and both signs are the same.
fn r_path<EXP_SZ: u32, FRACTION_SZ: u32>(a: apfloat::APFloat<EXP_SZ, FRACTION_SZ>,
                                         b: apfloat::APFloat<EXP_SZ, FRACTION_SZ>)
                                         -> (apfloat::APFloat<EXP_SZ, FRACTION_SZ>, bool) {
    const _SIGNIFICAND_WIDTH = FRACTION_SZ + u32:1;

    let (exp_diff_sign, up_diff, med_shift, use_far) = r_exponent_differencer(a, b);

    let exponent_difference_is_big = or_reduce(up_diff);

    let effective_subtraction = a.sign ^ b.sign;
    let a_significand = u1:1 ++ a.fraction;
    let b_significand = u1:1 ++ b.fraction;

    // 1's complement values to use in case we have a effective subtraction
    let (signed_a_significand, signed_b_significand) = if effective_subtraction {
        (!a_significand, !b_significand)
    } else {
        (a_significand, b_significand)
    };

    // In the case of an effective subtraction, we preshift the significands to the left so that
    // the final result is in the range of [1,4) instead of [1/2, 2).
    // This unifies the output range of the effective subtraction and effective addition.
    let (preshifted_signed_a, preshifted_signed_b) = if effective_subtraction {
        (signed_a_significand ++ u2:0b11, u1:0b1 ++ signed_b_significand ++ u1:0b1)
    } else {
        (u1:0b0 ++ signed_a_significand ++ u1:0b0, u2:0b00 ++ signed_a_significand)
    };
    let preshifted_signed_smaller_significand = if exp_diff_sign {
        preshifted_signed_a
    } else {
        preshifted_signed_b
    };

    const LARGEST_ALIGNMENT_SHIFT = (u32:1 << std::clog2(FRACTION_SZ + u32:2)) - u32:1;
    const ALIGNED_SIGNIFICAND_FRACTION_BITS = LARGEST_ALIGNMENT_SHIFT + FRACTION_SZ;

    type PaddedAlignedSignificand = sN[ALIGNED_SIGNIFICAND_FRACTION_BITS + u32:2];
    type AlignedSignificandWithRounding = uN[ALIGNED_SIGNIFICAND_FRACTION_BITS + u32:1];

    let _signed_small_significand = if exp_diff_sign {
        signed_b_significand
    } else {
        signed_a_significand
    };
    let top_part_bits = uN[LARGEST_ALIGNMENT_SHIFT]:0;
    let top_part = if effective_subtraction { !top_part_bits } else { top_part_bits };
    let max_shift_aligned_small_significand = top_part ++ signed_a_significand;

    let padding_bits = uN[LARGEST_ALIGNMENT_SHIFT - u32:2]:0;
    let sign_adjusted_padding = if effective_subtraction { !padding_bits } else { padding_bits };
    let padded_signed_small_significand:PaddedAlignedSignificand =
        std::to_signed(effective_subtraction ++
                       preshifted_signed_smaller_significand ++
                       sign_adjusted_padding);
    let medium_shift_aligned_signed_small_significand =
        (padded_signed_small_significand >> med_shift) as AlignedSignificandWithRounding;
    let _aligned_signed_small_significand = if exponent_difference_is_big {
        max_shift_aligned_small_significand
    } else {
        medium_shift_aligned_signed_small_significand
    };

    let larger_significand = if exp_diff_sign { b_significand } else { a_significand };
    let _preshifted_larger_significand = if effective_subtraction {
        larger_significand ++ u1:0
    } else {
        u1:0 ++ larger_significand
    };

    // Under the assumptions of the R-Path, we also know the sign of the larger input is
    // the sign of the output
    let _far_sign = if exp_diff_sign { b.sign } else { a.sign };

    // Missing: shifted add and rounding magic.

    // TODO(hzeller) Implement to end.
    // Until r_path() implementation complete: delegate to regular add.
    (apfloat::add(a, b), use_far)
}

#[test]
fn r_path_test() {
    type F32 = apfloat::APFloat<8, 23>;

    let a = F32 { sign: u1:0, bexp: u8:0x7f, fraction: u23:0x000000 };  // 1
    let b = F32 { sign: u1:0, bexp: u8:0x81, fraction: u23:0x000000 };  // 4
    let expected = F32 { sign: u1:0, bexp: u8:0x81, fraction: u23:0x200000 };  // a + b = 5
    assert_eq(r_path(a, b), (expected, true));

    let a = F32 { sign: u1:0, bexp: u8:0x7f, fraction: u23:0x000000 };  // 1
    let b = F32 { sign: u1:1, bexp: u8:0x81, fraction: u23:0x000000 };  // -4
    let expected = F32 { sign: u1:1, bexp: u8:0x80, fraction: u23:0x400000 };  // a + b = -3
    assert_eq(r_path(a, b), (expected, true));

    let a = F32 { sign: u1:0, bexp: u8:0x7f, fraction: u23:0x000000 };  // 1
    let b = F32 { sign: u1:1, bexp: u8:0x81, fraction: u23:0x400000 };  // -6
    let expected = F32 { sign: u1:1, bexp: u8:0x81, fraction: u23:0x200000 };  // a + b = -5
    assert_eq(r_path(a, b), (expected, true));

    let a = F32 { sign: u1:0, bexp: u8:0x7f, fraction: u23:0x000000 };  // 1
    let b = F32 { sign: u1:1, bexp: u8:0x7d, fraction: u23:0x000000 };  // -0.25
    let expected = F32 { sign: u1:0, bexp: u8:0x7e, fraction: u23:0x400000 };  // a + b = 0.75
    assert_eq(r_path(a, b), (expected, true));

    let a = F32 { sign: u1:0, bexp: u8:0xfe, fraction: u23:0x7fffff };  // FLT_MAX
    let b = F32 { sign: u1:0, bexp: u8:0xe7, fraction: u23:0x000000 };  // ulp(FLT_MAX)
    let expected = F32 { sign: u1:0, bexp: u8:0xff, fraction: u23:0x000000 };  // a + b = inf
    assert_eq(r_path(a, b), (expected, true));
}

pub fn add_seidel_even<EXP_SZ: u32, FRACTION_SZ: u32>(x: apfloat::APFloat<EXP_SZ, FRACTION_SZ>,
                                                      y: apfloat::APFloat<EXP_SZ, FRACTION_SZ>)
                                                      -> apfloat::APFloat<EXP_SZ, FRACTION_SZ> {
    let (near_result, near_says_use_far) = n_path(x, y);  // exponent -1..+1 delta; different sign.
    let (far_result, far_says_use_far) = r_path(x, y);    // everything else

    // Near and far path have been executed in parallel, now choose the result.
    let use_far_result = (x.sign == y.sign) || near_says_use_far || far_says_use_far;
    if use_far_result { far_result } else { near_result }
}

#[test]
fn add_seidel_even_test() {
    type F32 = apfloat::APFloat<8, 23>;

    let one: F32 = apfloat::one<u32:8, u32:23>(u1:0);
    let two = F32 { sign: u1:0, bexp: u8:0x80, fraction: u23:0x000000 };
    assert_eq(add_seidel_even(one, one), two);

    let fourty = F32 { sign: u1:0, bexp: u8:0x84, fraction: u23:0x200000 };
    let fourtytwo = F32 { sign: u1:0, bexp: u8:0x84, fraction: u23:0x280000 };
    assert_eq(add_seidel_even(fourty, two), fourtytwo);

    // Subnormals considered zero
    let a = F32 { sign: u1:1, bexp: u8:0x00, fraction: u23:0x008400 };  // subnormal
    let b = F32 { sign: u1:0, bexp: u8:0x13, fraction: u23:0x7c2018 };  // some random value
    let expected = F32 { sign: u1:0, bexp: u8:0x13, fraction: u23:0x7c2018 };  // a + b = same
    assert_eq(add_seidel_even(a, b), expected);

    // Infinity + something = infinity
    let a = F32 { sign: u1:0, bexp: u8:0xff, fraction: u23:0x000000 };  // inf
    let b = F32 { sign: u1:0, bexp: u8:0xf2, fraction: u23:0x401000 };  // some random normal
    let expected = F32 { sign: u1:0, bexp: u8:0xff, fraction: u23:0x000000 };  // a + b = inf
    assert_eq(add_seidel_even(a, b), expected);

    // -inf + inf => nan
    let a = F32 { sign: u1:1, bexp: u8:0xff, fraction: u23:0x000000 };  // hff800000
    let b = F32 { sign: u1:0, bexp: u8:0xff, fraction: u23:0x000000 };  // h7f800000
    let expected = F32 { sign: u1:0, bexp: u8:0xff, fraction: u23:0x400000 };
    //assert_eq(add_seidel_even(a, b), expected);
    assert_eq(apfloat::add(a, b), expected);  // <- TODO: use add_seidel_even() not apfloat

    // Far away results should just result in the far result
    let a = F32 { sign: u1:0, bexp: u8:0x11, fraction: u23:0x123456 };  // pretty tiny
    let b = F32 { sign: u1:1, bexp: u8:0x92, fraction: u23:0x000001 };  // some number
    let expected = F32 { sign: u1:1, bexp: u8:0x92, fraction: u23:0x000001 };  // a + b = same
    //assert_eq(add_seidel_even(a, b), expected);
    assert_eq(apfloat::add(a, b), expected);  // <- TODO: use add_seidel_even() not apfloat
}

// Simple top for evaluations
pub fn float32_top(a: apfloat::APFloat<8, 23>, b: apfloat::APFloat<8, 23>)
                   -> apfloat::APFloat<8, 23> {
    add_seidel_even(a, b)
}
