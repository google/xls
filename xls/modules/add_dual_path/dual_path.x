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

import xls.modules.add_dual_path.near_path;
import xls.modules.add_dual_path.far_path;
import xls.modules.add_dual_path.common;

type APFloat = apfloat::APFloat;
type AbsDiffResult = abs_diff::AbsDiffResult;

enum WhichPath : u1 {
    Near = 0,
    Far = 1,
}

pub fn add_dual_path<EXP_SZ: u32, FRACTION_SZ: u32>
    (x: APFloat<EXP_SZ, FRACTION_SZ>, y: APFloat<EXP_SZ, FRACTION_SZ>)
    -> APFloat<EXP_SZ, FRACTION_SZ> {
    const FRACTION_SZ_P1: u32 = FRACTION_SZ + u32:1;

    let abs_diff_result: AbsDiffResult<EXP_SZ> = abs_diff::abs_diff(x.bexp, y.bexp);

    // Swap so that x has the larger exponent.
    let (x, y) = if abs_diff::is_x_larger(abs_diff_result) { (x, y) } else { (y, x) };

    let which_path = match (abs_diff_result.correction, abs_diff_result.uncorrected) {
        (_, uN[EXP_SZ]:0) | (u1:0, uN[EXP_SZ]:1) => WhichPath::Near,
        _ => WhichPath::Far,
    };

    let result = if which_path == WhichPath::Near {
        near_path::add_near_path_with_diff<common::USE_CLZT>(x, y, abs_diff_result)
    } else {
        far_path::add_far_path_with_diff(x, y, abs_diff_result)
    };

    if apfloat::is_nan(x) || apfloat::is_nan(y) {
        apfloat::qnan<EXP_SZ, FRACTION_SZ>()
    } else if apfloat::is_inf(x) && apfloat::is_inf(y) {
        if x.sign == y.sign {
            apfloat::inf<EXP_SZ, FRACTION_SZ>(x.sign)
        } else {
            apfloat::qnan<EXP_SZ, FRACTION_SZ>()
        }
    } else if apfloat::is_inf(x) {
        apfloat::inf<EXP_SZ, FRACTION_SZ>(x.sign)
    } else if apfloat::is_inf(y) {
        apfloat::inf<EXP_SZ, FRACTION_SZ>(y.sign)
    } else {
        result
    }
}

pub fn add_dual_path_bf16(x: APFloat<8, 7>, y: APFloat<8, 7>) -> APFloat<8, 7> {
    add_dual_path(x, y)
}

pub fn add_dual_path_f32(x: APFloat<8, 23>, y: APFloat<8, 23>) -> APFloat<8, 23> {
    add_dual_path(x, y)
}

// This tests the annoying case where the smaller mantissa is big enough to cause partial
// cancellation of the hidden bit.
#[test]
fn test_add_dual_path_far_path_partial_cancellation() {
    let x = APFloat { sign: false, bexp: u8:3, fraction: u7:0b001_0000 };
    let y = APFloat { sign: true, bexp: u8:1, fraction: u7:0b010_0001 };
    assert_eq(add_dual_path(x, y), apfloat::add(x, y));
}

#[quickcheck(test_count=100000)]
fn quickcheck_add_dual_path_nosign(x_frac: u7, y_frac: u7, x_bexp: u8, y_bexp: u8) -> bool {
    let x = APFloat { sign: false, bexp: x_bexp, fraction: x_frac };
    let y = APFloat { sign: false, bexp: y_bexp, fraction: y_frac };
    add_dual_path(x, y) == apfloat::add(x, y)
}

#[test]
fn test_inf_plus_neg_inf() {
    let x = APFloat { sign: false, bexp: u8:255, fraction: u7:0 };  // inf
    let y = APFloat { sign: true, bexp: u8:255, fraction: u7:0 };  // -inf
    assert_eq(add_dual_path(x, y), apfloat::add(x, y));
}

#[test]
fn test_zero_plus_neg_inf() {
    let x = APFloat { sign: false, bexp: u8:0, fraction: u7:0 };
    let y = APFloat { sign: true, bexp: u8:255, fraction: u7:0 };  // -inf
    assert_eq(add_dual_path(x, y), apfloat::add(x, y));
}

#[test]
fn test_add_signed_small_normal_plus_signed_denorm() {
    let x = APFloat { sign: true, bexp: u8:1, fraction: u7:1 };  // 1.0000001 * 2^(1-127)
    let y = APFloat { sign: true, bexp: u8:0, fraction: u7:1 };  // signed denorm
    assert_eq(add_dual_path(x, y), apfloat::add(x, y));
}

#[test]
fn test_add_two_negative_subnorms_gives_negative_zero() {
    let x = APFloat { sign: true, bexp: u8:0, fraction: u7:1 };
    let y = APFloat { sign: true, bexp: u8:0, fraction: u7:1 };
    assert_eq(add_dual_path(x, y), apfloat::add(x, y));
}
