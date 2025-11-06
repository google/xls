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
    // Implementation based on
    // https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/src/Core/arch/Default/BFloat16.h#L405

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
    BF16 {
        sign: output_raw[15:16],
        bexp: output_raw[7:15],
        fraction: output_raw[0:7]
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
