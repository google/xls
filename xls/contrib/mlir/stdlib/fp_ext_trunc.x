// Copyright 2024 The XLS Authors
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

// Extension and truncation of floating point values.

import apfloat;

pub const BF16_EXP_SZ = u32:8;  // Exponent bits
pub const BF16_FRACTION_SZ = u32:7;  // Fraction bits

pub const F32_EXP_SZ = u32:8;  // Exponent bits
pub const F32_FRACTION_SZ = u32:23;  // Fraction bits

type BF16 = apfloat::APFloat<BF16_EXP_SZ, BF16_FRACTION_SZ>;
type F32 = apfloat::APFloat<F32_EXP_SZ, F32_FRACTION_SZ>;

pub fn trunc(f: F32) -> BF16 {
    // Implementation based on Eigen's float_to_bfloat16_rtne:
    // https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/src/Core/arch/Default/BFloat16.h
    if apfloat::is_nan(f) {
        // Squash NaN to a quiet NaN (preserving sign) so rounding cannot turn sNaNs into +/-inf
        // or wrap high-payload NaNs into +/-0.0.
        let qnan = apfloat::qnan<BF16_EXP_SZ, BF16_FRACTION_SZ>();
        BF16 { sign: f.sign, bexp: qnan.bexp, fraction: qnan.fraction }
    } else {
        // 1. bit_cast<uint32_t>(ff)
        let input = f.sign ++ f.bexp ++ f.fraction;

        // 2. lsb = (input >> 16) & 1
        let lsb = input[16:17] as u32;

        // 3. rounding_bias = 0x7fff + lsb
        let rounding_bias = u32:0x7fff + lsb;

        // 4. input += rounding_bias
        let rounded_input = input + rounding_bias;

        // 5. output.value = static_cast<uint16_t>(input >> 16)
        //    Extract the upper 16 bits.
        let output_raw = rounded_input[16:32];

        // Re-pack into the BF16 struct
        BF16 { sign: output_raw[15:16], bexp: output_raw[7:15], fraction: output_raw[0:7] }
    }
}

pub fn ext(f: BF16) -> F32 {
    F32 {
        sign: f.sign,
        bexp: f.bexp,
        // BF16 fraction bits go in the MSB positions of the F32 fraction.
        // Pad with zeros at the LSB end.
        fraction: f.fraction ++ bits[F32_FRACTION_SZ - BF16_FRACTION_SZ]:0
    }
}

#[test]
fn trunc_nan_test() {
    let pos_qnan = BF16 { sign: u1:0, bexp: u8:0xff, fraction: u7:0x40 };
    let neg_qnan = BF16 { sign: u1:1, bexp: u8:0xff, fraction: u7:0x40 };

    // Signaling NaN whose upper 7 fraction bits are 0 must not truncate to +/-inf.
    assert_eq(trunc(F32 { sign: u1:0, bexp: u8:0xff, fraction: u23:0x000001 }), pos_qnan);
    assert_eq(trunc(F32 { sign: u1:1, bexp: u8:0xff, fraction: u23:0x000001 }), neg_qnan);

    // High-payload NaN must not overflow rounding addition into +/-0.0.
    assert_eq(trunc(F32 { sign: u1:0, bexp: u8:0xff, fraction: u23:0x7f8000 }), pos_qnan);
    assert_eq(trunc(F32 { sign: u1:1, bexp: u8:0xff, fraction: u23:0x7f8000 }), neg_qnan);
    assert_eq(trunc(F32 { sign: u1:0, bexp: u8:0xff, fraction: u23:0x7fffff }), pos_qnan);
    assert_eq(trunc(F32 { sign: u1:1, bexp: u8:0xff, fraction: u23:0x7fffff }), neg_qnan);

    // Standard quiet NaNs stay quiet NaNs with preserved sign.
    assert_eq(trunc(F32 { sign: u1:0, bexp: u8:0xff, fraction: u23:0x400000 }), pos_qnan);
    assert_eq(trunc(F32 { sign: u1:1, bexp: u8:0xff, fraction: u23:0x400000 }), neg_qnan);
}

#[test]
fn trunc_and_ext_edge_cases_test() {
    // Signed zeros and infinities.
    assert_eq(trunc(apfloat::zero<8, 23>(false)), apfloat::zero<8, 7>(false));
    assert_eq(trunc(apfloat::zero<8, 23>(true)), apfloat::zero<8, 7>(true));
    assert_eq(trunc(apfloat::inf<8, 23>(false)), apfloat::inf<8, 7>(false));
    assert_eq(trunc(apfloat::inf<8, 23>(true)), apfloat::inf<8, 7>(true));

    // Round-to-nearest-even ties (LSB=0 rounds down, LSB=1 rounds up).
    assert_eq(
        trunc(F32 { sign: u1:0, bexp: u8:127, fraction: u23:0x008000 }),
        BF16 { sign: u1:0, bexp: u8:127, fraction: u7:0x00 });
    assert_eq(
        trunc(F32 { sign: u1:0, bexp: u8:127, fraction: u23:0x018000 }),
        BF16 { sign: u1:0, bexp: u8:127, fraction: u7:0x02 });

    // Max finite F32 rounds up to +inf; max subnormal F32 rounds up to min_normal BF16.
    assert_eq(
        trunc(F32 { sign: u1:0, bexp: u8:0xfe, fraction: u23:0x7f8000 }),
        apfloat::inf<8, 7>(false));
    assert_eq(
        trunc(F32 { sign: u1:0, bexp: u8:0x00, fraction: u23:0x7f8000 }),
        BF16 { sign: u1:0, bexp: u8:0x01, fraction: u7:0x00 });

    // ext pads fraction with zeros.
    assert_eq(
        ext(BF16 { sign: u1:1, bexp: u8:0x7f, fraction: u7:0x55 }),
        F32 { sign: u1:1, bexp: u8:0x7f, fraction: u23:0x550000 });
}
