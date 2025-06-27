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

// The far path is designed for the "no catastrophic cancellation" case, i.e.
// because the exponent difference is >= 2 we know that the hidden bit from the larger operand
// x will at worst give a borrow to the next least significant bit in the sum.
//
// This lets us elide the clz primitive on output and just use a 2:1 mux.
pub fn add_far_path_with_diff<EXP_SZ: u32, FRACTION_SZ: u32>
    (x: APFloat<EXP_SZ, FRACTION_SZ>, y: APFloat<EXP_SZ, FRACTION_SZ>,
     exp_diff_result: AbsDiffResult<EXP_SZ>) -> APFloat<EXP_SZ, FRACTION_SZ> {
    const FRACTION_SZ_P1: u32 = FRACTION_SZ + u32:1;
    const FRACTION_SZ_P2: u32 = FRACTION_SZ_P1 + u32:1;
    const FRACTION_SZ_P3: u32 = FRACTION_SZ_P2 + u32:1;
    const FRACTION_SZ_P4: u32 = FRACTION_SZ_P3 + u32:1;
    const FRACTION_SZ_P5: u32 = FRACTION_SZ_P4 + u32:1;

    let x_addend: uN[FRACTION_SZ_P4] = common::ftz(x.bexp, x.fraction) ++ u3:0;
    let y_addend: uN[FRACTION_SZ_P4] = common::ftz(y.bexp, y.fraction) ++ u3:0;

    // We align the smaller operand for addition.
    let y_addend_shifted: uN[FRACTION_SZ_P4] = y_addend >> exp_diff_result.uncorrected;
    let y_addend_shifted = y_addend_shifted >> exp_diff_result.correction;
    let shifted_mask = common::dynamic_mask<FRACTION_SZ_P4>(exp_diff_result.uncorrected);
    let shifted_mask = if exp_diff_result.correction {
        (shifted_mask << 1) | uN[FRACTION_SZ_P4]:1
    } else {
        shifted_mask
    };
    let shifted_off = y_addend & shifted_mask;
    let low_bit_jam = or_reduce(shifted_off);
    let y_addend_shifted_jammed: uN[FRACTION_SZ_P4] =
        y_addend_shifted | low_bit_jam as uN[FRACTION_SZ_P4];

    let (result_sign, wide_sum): (bool, uN[FRACTION_SZ_P5]) =
        sign_magnitude::sign_magnitude_add_sub(x.sign, x_addend, y.sign, y_addend_shifted_jammed);

    let (carry, sum): (bool, uN[FRACTION_SZ_P4]) = common::split_msbs<u32:1>(wide_sum);

    let (frac, exp, guard, sticky): (uN[FRACTION_SZ], uN[EXP_SZ], bool, bool) = if carry {
        let exp = x.bexp + uN[EXP_SZ]:1;

        // the carry bit is the hidden bit, so the msbs of the sum are the fraction
        let (frac, rest) = common::split_msbs<FRACTION_SZ>(sum);
        (frac, exp, std::msb(rest), or_reduce(rest[:-1]))
    } else if std::msb(sum) {
        // the top bit is the hidden bit, so we lop that off
        let (frac, rest) = common::split_msbs<FRACTION_SZ>(sum[:-1]);
        (frac, x.bexp, std::msb(rest), or_reduce(rest[:-1]))
    } else {
        let (frac, rest) = common::split_msbs<FRACTION_SZ>(sum[:-2]);
        (frac, x.bexp - uN[EXP_SZ]:1, std::msb(rest), or_reduce(rest[:-1]))
    };

    let is_frac_odd = std::lsb(frac) == true;
    let round_up = match (guard, sticky) {
        (true, true) => true,
        (true, false) => is_frac_odd,  // frac is odd, it's a tie => round up to even
        (false, _) => false,
    };

    let (frac_out, bexp_out): (uN[FRACTION_SZ], uN[EXP_SZ]) = if round_up {
        let frac = frac + uN[FRACTION_SZ]:1;
        let bexp = if frac == uN[FRACTION_SZ]:0 { exp + uN[EXP_SZ]:1 } else { exp };
        // For overflow to infinity, we need to set the fraction to 0.
        (frac, bexp)
    } else {
        (frac, exp)
    };

    // We need to give the canonical "fraction of zero" when the exponent has become infinity.
    let frac_out = if bexp_out == all_ones!<uN[EXP_SZ]>() { uN[FRACTION_SZ]:0 } else { frac_out };

    APFloat { sign: x.sign, bexp: bexp_out, fraction: frac_out }
}

fn add_far_path<EXP_SZ: u32, FRACTION_SZ: u32>
    (x: APFloat<EXP_SZ, FRACTION_SZ>, y: APFloat<EXP_SZ, FRACTION_SZ>)
    -> APFloat<EXP_SZ, FRACTION_SZ> {
    let exp_diff_result = abs_diff::abs_diff(x.bexp, y.bexp);
    add_far_path_with_diff(x, y, exp_diff_result)
}

fn do_test_quickcheck_with_sign
    (x_sign: bool, y_sign: bool, x_frac: u7, y_frac: u7, x_bexp: u8, exp_diff: u8)
    {
    let x = APFloat { sign: x_sign, bexp: x_bexp, fraction: x_frac };
    let y = APFloat { sign: y_sign, bexp: x_bexp - exp_diff as u8, fraction: y_frac };
    assert_eq(add_far_path(x, y), apfloat::add(x, y));
}

fn do_test_quickcheck(x_frac: u7, y_frac: u7, x_bexp: u8, exp_diff: u8) {
    do_test_quickcheck_with_sign(false, false, x_frac, y_frac, x_bexp, exp_diff);
}

// x_addend is  11111111
// y_addend is  10000010
// y_shifted is       10_000010
// sum is      100000001 (note: has carry)
//              ^^^^^^^\-guard
#[test]
fn test_qc_127_2_0_6__guard_computed_under_carry() { do_test_quickcheck(u7:127, u7:2, u8:7, u8:6); }

#[test]
fn test_qc_0_1_9_8() { do_test_quickcheck(u7:0, u7:1, u8:9, u8:8); }

#[test]
fn test_qc_0_0_3_2() { do_test_quickcheck(u7:0, u7:0, u8:3, u8:2); }

#[test]
fn test_qc_0_0_9_8() { do_test_quickcheck(u7:0, u7:0, u8:9, u8:8); }

#[test]
fn test_qc_0_0_10_9() { do_test_quickcheck(u7:0, u7:0, u8:10, u8:9); }

#[test]
fn test_qc_0_0_11_10() { do_test_quickcheck(u7:0, u7:0, u8:11, u8:10); }

#[test]
fn test_qc_0_3_3_2() { do_test_quickcheck(u7:0, u7:3, u8:3, u8:2); }

#[test]
fn test_qc_80_67_128_2() { do_test_quickcheck(u7:80, u7:67, u8:128, u8:2); }

#[test]
fn test_qc_119_92_180_2() { do_test_quickcheck(u7:119, u7:92, u8:180, u8:2); }

// Interesting case for rounding:
// x          11111111
// y          11111000
// y shifted  00001111 => guard = 1 sticky = 0
// result         111 (no hidden)
#[test]
fn test_qc_127_120_64_4() { do_test_quickcheck(u7:127, u7:120, u8:64, u8:4); }

#[test]
fn test_qc_118_127_125_3() { do_test_quickcheck(u7:118, u7:127, u8:125, u8:3); }

#[test]
fn test_qc_106_57_254_3() { do_test_quickcheck(u7:106, u7:57, u8:254, u8:3); }

#[test]
fn test_qc_127_2_74_6() { do_test_quickcheck(u7:127, u7:2, u8:74, u8:6); }

// Note: most of the time the value for the x bexp doesn't matter so we can just set it to the
// exp_diff plus one.

#[test]
fn test_qc_107_42_3_2() { do_test_quickcheck(u7:107, u7:42, u8:3, u8:2); }

#[test]
fn test_qc_98_124_3_2() { do_test_quickcheck(u7:98, u7:124, u8:3, u8:2); }

#[test]
fn test_qc_102_58_254_2() { do_test_quickcheck(u7:102, u7:58, u8:254, u8:2); }

#[quickcheck(test_count=100000)]
fn quickcheck_add_far_path(x_frac: u7, y_frac: u7, x_bexp: u8, exp_diff: u8) -> bool {
    let x = APFloat { sign: false, bexp: x_bexp, fraction: x_frac };
    let y = APFloat { sign: false, bexp: x_bexp - exp_diff as u8, fraction: y_frac };
    if exp_diff < u8:2 || y.bexp > x.bexp {
        true
    } else {
        match (apfloat::tag(x), apfloat::tag(y)) {
            (apfloat::APFloatTag::NAN, _) | (_, apfloat::APFloatTag::NAN) => true,
            (apfloat::APFloatTag::INFINITY, _) | (_, apfloat::APFloatTag::INFINITY) => true,
            (apfloat::APFloatTag::SUBNORMAL, _) | (_, apfloat::APFloatTag::SUBNORMAL) => true,
            _ => add_far_path(x, y) == apfloat::add(x, y),
        }
    }
}

#[test]
fn test_qc_with_sign_0_1_111_96_146_8() {
    do_test_quickcheck_with_sign(false, true, u7:111, u7:96, u8:146, u8:8);
}

// 1.0001000 * 2^(3-127) + -1.010_0001 * 2^(1-127)
#[test]
fn test_qc_with_sign_0_1_8_33_3_2() {
    do_test_quickcheck_with_sign(false, true, u7:8, u7:33, u8:3, u8:2);
}

#[test]
fn test_qc_with_sign_1_0_6_106_5_4() {
    do_test_quickcheck_with_sign(true, false, u7:6, u7:106, u8:5, u8:4);
}

#[quickcheck(test_count=100000)]
fn quickcheck_add_far_path_with_sign
    (x_sign: bool, y_sign: bool, x_frac: u7, y_frac: u7, x_bexp: u8, exp_diff: u8) -> bool {
    let x = APFloat { sign: x_sign, bexp: x_bexp, fraction: x_frac };
    let y = APFloat { sign: y_sign, bexp: x_bexp - exp_diff as u8, fraction: y_frac };
    if exp_diff < u8:2 || y.bexp > x.bexp {
        true
    } else {
        match (apfloat::tag(x), apfloat::tag(y)) {
            (apfloat::APFloatTag::NAN, _) | (_, apfloat::APFloatTag::NAN) => true,
            (apfloat::APFloatTag::INFINITY, _) | (_, apfloat::APFloatTag::INFINITY) => true,
            (apfloat::APFloatTag::SUBNORMAL, _) | (_, apfloat::APFloatTag::SUBNORMAL) => true,
            _ => add_far_path(x, y) == apfloat::add(x, y),
        }
    }
}
