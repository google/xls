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

import xls.modules.add_dual_path.common;

/// Performs an "add/sub" operation:
/// * if the signs are the same, adds the magnitudes
/// * if the signs are different, subtracts the smaller magnitude from the larger
/// Returns `(result_sign, result_magnitude)`.
pub fn sign_magnitude_add_sub<N: u32, WIDE: u32 = {N + u32:1}>
    (x_sign: bool, x_val: uN[N], y_sign: bool, y_val: uN[N]) -> (bool, uN[WIDE]) {
    // Same sign => normal unsigned add
    if x_sign == y_sign {
        (x_sign, common::add_with_carry_out(x_val, y_val))
    } else {
        // Opposite signs => subtract smaller magnitude from larger
        if x_val > y_val {
            (x_sign, (x_val - y_val) as uN[WIDE])
        } else if x_val < y_val {
            (y_sign, (y_val - x_val) as uN[WIDE])
        } else {
            // Exactly equal => can produce ±0
            (false, uN[WIDE]:0)
        }
    }
}

#[test]
fn test_sign_magnitude_add_sub() {
    assert_eq(sign_magnitude_add_sub(false, u8:0, false, u8:0), (false, u9:0));
    assert_eq(sign_magnitude_add_sub(false, u8:0, false, u8:1), (false, u9:1));
    assert_eq(sign_magnitude_add_sub(false, u8:1, false, u8:0), (false, u9:1));
    assert_eq(sign_magnitude_add_sub(false, u8:1, false, u8:1), (false, u9:2));
    assert_eq(sign_magnitude_add_sub(false, u8:1, true, u8:1), (false, u9:0));
    assert_eq(sign_magnitude_add_sub(true, u8:1, false, u8:1), (false, u9:0));
    assert_eq(sign_magnitude_add_sub(true, u8:1, true, u8:1), (true, u9:2));

    // See that we get the appropriate carry bit out when the addition overflows.
    assert_eq(sign_magnitude_add_sub(false, u8:255, false, u8:1), (false, u9:256));
}

#[quickcheck(exhaustive)]
fn prop_add_unsigned_numbers_gives_unsigned(x: u4, y: u4) -> bool {
    let (sign, result) = sign_magnitude_add_sub(false, x, false, y);
    sign == false && result >= x as u5 && result >= y as u5
}
