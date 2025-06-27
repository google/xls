// Copyright 2025 The XLS Authors
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

import std;
import apfloat;
import abs_diff;

import xls.modules.add_dual_path.common;
import xls.modules.add_dual_path.sign_magnitude;

type APFloat = apfloat::APFloat;
type AbsDiffResult = abs_diff::AbsDiffResult;

// The near path is designed for the "possible catastrophic cancellation" case, i.e.
// it does a count-leading-zeros on the result to see how to normalize the result.
//
// Because the exponent difference is at most 1, we just need a 2:1 mux for the y addend
// instead of a full shifter.
pub fn add_near_path_with_diff<USE_CLZT: bool, EXP_SZ: u32, FRACTION_SZ: u32>
    (x: APFloat<EXP_SZ, FRACTION_SZ>, y: APFloat<EXP_SZ, FRACTION_SZ>,
     exp_diff_result: AbsDiffResult<EXP_SZ>) -> APFloat<EXP_SZ, FRACTION_SZ> {
    const FRACTION_SZ_P1: u32 = FRACTION_SZ + u32:1;
    const FRACTION_SZ_P2: u32 = FRACTION_SZ_P1 + u32:1;
    const FRACTION_SZ_P3: u32 = FRACTION_SZ_P2 + u32:1;

    let x_addend: uN[FRACTION_SZ_P2] = common::ftz(x.bexp, x.fraction) ++ u1:0;
    let y_addend: uN[FRACTION_SZ_P2] = if abs_diff::is_zero(exp_diff_result) {
        common::ftz(y.bexp, y.fraction) ++ u1:0
    } else {
        // Exponent difference is 1, so we shift the conceptual "fraction with the hidden bit
        // prepended" by 1 bit to the right.
        let shifted = u2:0b01 ++ y.fraction;
        if y.bexp == uN[EXP_SZ]:0 { uN[FRACTION_SZ_P2]:0 } else { shifted }
    };

    let (result_sign, carry_result) =
        sign_magnitude::sign_magnitude_add_sub(x.sign, x_addend, y.sign, y_addend);
    let (carry, result): (u1, uN[FRACTION_SZ_P2]) = common::split_msbs<u32:1>(carry_result);

    if carry_result == uN[FRACTION_SZ_P3]:0 {
        let both_signed_zero =
            common::is_effective_signed_zero(x) && common::is_effective_signed_zero(y);
        apfloat::zero<EXP_SZ, FRACTION_SZ>(both_signed_zero)
    } else {
        if carry {
            let exp_out = x.bexp + uN[EXP_SZ]:1;
            // No shifting needed just selection of the high fraction bits and associated rounding.
            let (frac_out, lsbs): (uN[FRACTION_SZ], u2) = common::split_msbs<FRACTION_SZ>(result);
            let up_is_even = std::lsb(frac_out) == true;
            let (guard, sticky) = (lsbs[1+:u1], lsbs[0+:u1]);
            let round_up = match (guard, sticky, up_is_even) {
                (true, true, _) => true,
                (false, ..) => false,
                (true, false, _) => up_is_even,
            };
            let frac_out = if round_up { frac_out + uN[FRACTION_SZ]:1 } else { frac_out };
            let frac_out = if exp_out == all_ones!<uN[EXP_SZ]>() {
                uN[FRACTION_SZ]:0  // infinity fraction is 0
            } else {
                frac_out
            };
            APFloat<EXP_SZ, FRACTION_SZ> { sign: result_sign, bexp: exp_out, fraction: frac_out }
        } else {
            let left_shift =
                if USE_CLZT { std::clzt(result) as uN[EXP_SZ] } else { clz(result) as uN[EXP_SZ] };
            let exp_out =
                if left_shift <= x.bexp { x.bexp - left_shift } else { zero!<uN[EXP_SZ]>() };

            let shifted_result: uN[FRACTION_SZ_P2] = result << left_shift;
            let frac_out: uN[FRACTION_SZ] = shifted_result[1:-1];
            let frac_out = if exp_out == zero!<uN[EXP_SZ]>() {
                uN[FRACTION_SZ]:0  // zero/subnormal fraction is 0
            } else {
                frac_out
            };
            let up_is_even = std::lsb(frac_out) == true;
            let (frac_out, exp_out) = if std::lsb(shifted_result) && up_is_even {
                let new_frac = frac_out + uN[FRACTION_SZ]:1;
                let new_exp =
                    if new_frac == uN[FRACTION_SZ]:0 { exp_out + uN[EXP_SZ]:1 } else { exp_out };
                (new_frac, new_exp)
            } else {
                (frac_out, exp_out)
            };

            APFloat<EXP_SZ, FRACTION_SZ> {
                sign: result_sign,
                bexp: exp_out,
                fraction: frac_out,
                // lop off the hidden bit
            }
        }
    }
}

fn add_near_path<EXP_SZ: u32, FRACTION_SZ: u32>
    (x: APFloat<EXP_SZ, FRACTION_SZ>, y: APFloat<EXP_SZ, FRACTION_SZ>)
    -> APFloat<EXP_SZ, FRACTION_SZ> {
    let exp_diff_result = abs_diff::abs_diff(x.bexp, y.bexp);
    add_near_path_with_diff<common::USE_CLZT>(x, y, exp_diff_result)
}

// --

const BF16_BIAS = u8:127;
const BF16_ZERO = APFloat<u32:8, u32:7> { sign: false, bexp: u8:0, fraction: u7:0 };
const BF16_MINUS_ZERO = APFloat<u32:8, u32:7> { sign: true, bexp: u8:0, fraction: u7:0 };

// Smallest normal value is 1.0000_001 * 2^-126
const BF16_SMALLEST_NORMAL = APFloat<u32:8, u32:7> { sign: false, bexp: u8:1, fraction: u7:1 };

const BF16_ONE = APFloat<u32:8, u32:7> { sign: false, bexp: BF16_BIAS, fraction: u7:0 };
const BF16_MINUS_ONE = APFloat<u32:8, u32:7> { sign: true, bexp: BF16_BIAS, fraction: u7:0 };
const BF16_TWO = APFloat<u32:8, u32:7> { sign: false, bexp: BF16_BIAS + u8:1, fraction: u7:0 };

// 1.1 * 2^1
const BF16_THREE = APFloat<u32:8, u32:7> {
    sign: false,
    bexp: BF16_BIAS + u8:1,
    fraction: u1:0b1 ++ u6:0,
};

const BF16_FOUR = APFloat<u32:8, u32:7> { sign: false, bexp: BF16_BIAS + u8:2, fraction: u7:0 };

// 1.01 * 2^2
const BF16_FIVE = APFloat<u32:8, u32:7> {
    sign: false,
    bexp: BF16_BIAS + u8:2,
    fraction: u2:0b01 ++ u5:0,
};

const BF16_EIGHT = APFloat<u32:8, u32:7> { sign: false, bexp: BF16_BIAS + u8:3, fraction: u7:0 };

#[test]
fn test_0_0_add() { assert_eq(add_near_path(BF16_ZERO, BF16_ZERO), BF16_ZERO); }

#[test]
fn test_0_0_sub() { assert_eq(add_near_path(BF16_ZERO, BF16_MINUS_ZERO), BF16_ZERO); }

#[test]
fn test_1_1_add() { assert_eq(add_near_path(BF16_ONE, BF16_ONE), BF16_TWO); }

#[test]
fn test_1_1_sub() { assert_eq(add_near_path(BF16_ONE, BF16_MINUS_ONE), BF16_ZERO); }

#[test]
fn test_2_1_add() { assert_eq(add_near_path(BF16_TWO, BF16_ONE), BF16_THREE); }

#[test]
fn test_2_1_sub() { assert_eq(add_near_path(BF16_TWO, BF16_MINUS_ONE), BF16_ONE); }

#[test]
fn test_2_2_add() { assert_eq(add_near_path(BF16_TWO, BF16_TWO), BF16_FOUR); }

#[test]
fn test_2_3_add() { assert_eq(add_near_path(BF16_TWO, BF16_THREE), BF16_FIVE); }

#[test]
fn test_4_4_add() { assert_eq(add_near_path(BF16_FOUR, BF16_FOUR), BF16_EIGHT); }

#[test]
fn test_0_smallest_normal_add() {
    assert_eq(add_near_path(BF16_SMALLEST_NORMAL, BF16_ZERO), BF16_SMALLEST_NORMAL);
}

fn do_test_near_path_quickcheck_sample_with_sign
    (x_sign: bool, y_sign: bool, x_frac: u7, y_frac: u7, x_bexp: u8, exp_diff: u1)
    {
    let x = APFloat<u32:8, u32:7> { sign: x_sign, bexp: x_bexp, fraction: x_frac };
    let y = APFloat<u32:8, u32:7> { sign: y_sign, bexp: x_bexp - exp_diff as u8, fraction: y_frac };
    assert_eq(add_near_path(x, y), apfloat::add(x, y));
}

// As above, but with only positive-sign values.
fn do_test_near_path_quickcheck_sample(x_frac: u7, y_frac: u7, x_bexp: u8, exp_diff: u1) {
    do_test_near_path_quickcheck_sample_with_sign(false, false, x_frac, y_frac, x_bexp, exp_diff);
}

#[test]
fn test_qc_0_0_0_1() {
    // 0.0 + 1.0 * 2^-126
    do_test_near_path_quickcheck_sample(u7:0, u7:0, u8:1, u1:1);
}

#[test]
fn test_qc_0_1_1_0() { do_test_near_path_quickcheck_sample(u7:0, u7:1, u8:1, u1:0); }

#[test]
fn test_qc_0_1_2_1() {
    // 1.0 * 2^2 + 1.0000001 * 2^1
    do_test_near_path_quickcheck_sample(u7:0, u7:1, u8:2, u1:1);
}

#[test]
fn test_qc_0_2_254_0() { do_test_near_path_quickcheck_sample(u7:0, u7:2, u8:254, u1:0); }

#[test]
fn test_qc_0_3_1_0() { do_test_near_path_quickcheck_sample(u7:0, u7:3, u8:1, u1:0); }

#[test]
fn test_qc_0_3_2_1() { do_test_near_path_quickcheck_sample(u7:0, u7:3, u8:2, u1:1); }

#[test]
fn test_qc_0_127_2_1() { do_test_near_path_quickcheck_sample(u7:0, u7:127, u8:2, u1:1); }

#[quickcheck(test_count=100000)]
fn quickcheck_add_near_path(x_frac: u7, y_frac: u7, x_bexp: u8, exp_diff: u1) -> bool {
    let x = APFloat<u32:8, u32:7> { sign: false, bexp: x_bexp, fraction: x_frac };
    let y = APFloat<u32:8, u32:7> { sign: false, bexp: x_bexp - exp_diff as u8, fraction: y_frac };
    match (apfloat::tag(x), apfloat::tag(y)) {
        (apfloat::APFloatTag::NAN, _) | (_, apfloat::APFloatTag::NAN) => true,
        (apfloat::APFloatTag::INFINITY, _) | (_, apfloat::APFloatTag::INFINITY) => true,
        (apfloat::APFloatTag::SUBNORMAL, _) | (_, apfloat::APFloatTag::SUBNORMAL) => true,
        _ => add_near_path(x, y) == apfloat::add(x, y),
    }
}

#[test]
fn test_qc_with_sign_1_0_124_57_1_0() {
    do_test_near_path_quickcheck_sample_with_sign(true, false, u7:124, u7:57, u8:1, u1:0);
}

#[test]
fn test_qc_with_sign_1_0_71_104_5_10() {
    do_test_near_path_quickcheck_sample_with_sign(true, false, u7:71, u7:104, u8:5, u1:0);
}

#[test]
fn test_qc_with_sign_1_0_63_72_3_0() {
    do_test_near_path_quickcheck_sample_with_sign(true, false, u7:63, u7:72, u8:3, u1:0);
}

// -0.0 + -0.0 => -0.0
#[test]
fn test_qc_with_sign_1_1_0_0_0_0() {
    do_test_near_path_quickcheck_sample_with_sign(true, true, u7:0, u7:0, u8:0, u1:0);
}

#[quickcheck(test_count=100000)]
fn quickcheck_add_near_path_with_sign
    (x_sign: bool, y_sign: bool, x_frac: u7, y_frac: u7, x_bexp: u8, exp_diff: u1) -> bool {
    let x = APFloat<u32:8, u32:7> { sign: x_sign, bexp: x_bexp, fraction: x_frac };
    let y = APFloat<u32:8, u32:7> { sign: y_sign, bexp: x_bexp - exp_diff as u8, fraction: y_frac };
    match (apfloat::tag(x), apfloat::tag(y)) {
        (apfloat::APFloatTag::NAN, _) | (_, apfloat::APFloatTag::NAN) => true,
        (apfloat::APFloatTag::INFINITY, _) | (_, apfloat::APFloatTag::INFINITY) => true,
        (apfloat::APFloatTag::SUBNORMAL, _) | (_, apfloat::APFloatTag::SUBNORMAL) => true,
        _ => add_near_path(x, y) == apfloat::add(x, y),
    }
}
