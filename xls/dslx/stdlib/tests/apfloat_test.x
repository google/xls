// Copyright 2020 The XLS Authors
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

import apfloat;
import std;

#[test]
fn qnan_test() {
    let expected = apfloat::APFloat<8, 23> { sign: 0, bexp: 0xff, fraction: 0x400000 };
    let actual = apfloat::qnan<8, 23>();
    assert_eq(actual, expected);

    let expected = apfloat::APFloat<4, 2> { sign: 0, bexp: 0xf, fraction: 0x2 };
    let actual = apfloat::qnan<4, 2>();
    assert_eq(actual, expected);
}

#[test]
fn inf_test() {
    let expected = apfloat::APFloat<8, 23> { sign: 0, bexp: 0xff, fraction: 0x0 };
    let actual = apfloat::inf<8, 23>(0);
    assert_eq(actual, expected);

    let expected = apfloat::APFloat<4, 2> { sign: 0, bexp: 0xf, fraction: 0x0 };
    let actual = apfloat::inf<4, 2>(0);
    assert_eq(actual, expected);
}

#[test]
fn abs_test() {
    let expected = apfloat::APFloat<8, 23> { sign: 0, bexp: 0x7f, fraction: 0x0 };
    let actual =
        apfloat::abs<8, 23>(apfloat::APFloat<8, 23> { sign: 1, bexp: 0x7f, fraction: 0x0 });
    assert_eq(actual, expected);

    let expected = apfloat::APFloat<8, 23> { sign: 0, bexp: 0xff, fraction: 0x3645A2 };
    let actual =
        apfloat::abs<8, 23>(apfloat::APFloat<8, 23> { sign: 1, bexp: 0xff, fraction: 0x3645A2 });
    assert_eq(actual, expected);
}

#[test]
fn zero_test() {
    let expected = apfloat::APFloat<8, 23> { sign: 0, bexp: 0x0, fraction: 0x0 };
    let actual = apfloat::zero<8, 23>(0);
    assert_eq(actual, expected);

    let expected = apfloat::APFloat<4, 2> { sign: 1, bexp: 0x0, fraction: 0x0 };
    let actual = apfloat::zero<4, 2>(1);
    assert_eq(actual, expected);
}

#[test]
fn one_test() {
    let expected = apfloat::APFloat<8, 23> { sign: 0, bexp: 0x7f, fraction: 0x0 };
    let actual = apfloat::one<8, 23>(0);
    assert_eq(actual, expected);

    let expected = apfloat::APFloat<4, 2> { sign: 0, bexp: 0x7, fraction: 0x0 };
    let actual = apfloat::one<4, 2>(0);
    assert_eq(actual, expected);
}

#[test]
fn max_normal_test() {
    let expected = apfloat::APFloat<8, 23> { sign: 0, bexp: 0xfe, fraction: 0x7fffff };
    let actual = apfloat::max_normal<8, 23>(0);
    assert_eq(actual, expected);

    let expected = apfloat::APFloat<4, 2> { sign: 1, bexp: 0xe, fraction: 0x3 };
    let actual = apfloat::max_normal<4, 2>(1);
    assert_eq(actual, expected);
}

#[test]
fn test_max_normal_exp() {
    assert_eq(apfloat::max_normal_exp<8>(), s8:127);
    assert_eq(apfloat::max_normal_exp<11>(), s11:1023);
}

#[test]
fn test_min_normal_exp() {
    assert_eq(apfloat::min_normal_exp<8>(), s8:-126);
    assert_eq(apfloat::min_normal_exp<11>(), s11:-1022);
}

#[test]
fn exponent_bias_test() {
    const F32_EXP_SZ = u32:8;
    const F32_FRACTION_SZ = u32:23;
    let expected = s8:127;
    let input = apfloat::one<F32_EXP_SZ, F32_FRACTION_SZ>(false);
    let actual = apfloat::exponent_bias(input);
    assert_eq(actual, expected);

    const BF19_EXP_SZ = u32:8;
    const BF19_FRACTION_SZ = u32:10;
    let expected = s8:127;
    let input = apfloat::one<BF19_EXP_SZ, BF19_FRACTION_SZ>(false);
    let actual = apfloat::exponent_bias(input);
    assert_eq(actual, expected);

    const BF16_EXP_SZ = u32:8;
    const BF16_FRACTION_SZ = u32:7;
    let expected = s8:127;
    let input = apfloat::one<BF16_EXP_SZ, BF16_FRACTION_SZ>(false);
    let actual = apfloat::exponent_bias(input);
    assert_eq(actual, expected);

    const FP16_EXP_SZ = u32:5;
    const FP16_FRACTION_SZ = u32:10;
    let expected = s5:15;
    let input = apfloat::one<FP16_EXP_SZ, FP16_FRACTION_SZ>(false);
    let actual = apfloat::exponent_bias(input);
    assert_eq(actual, expected);
}

#[test]
fn unbiased_exponent_zero_test() {
    let expected = s8:-127;
    let actual = apfloat::unbiased_exponent<8, 23>(apfloat::zero<8, 23>(false));
    assert_eq(actual, expected);

    let actual = apfloat::unbiased_exponent<8, 23>(apfloat::zero<8, 23>(true));
    assert_eq(actual, expected);
}

#[test]
fn unbiased_exponent_one_test() {
    let expected = s8:0;
    let actual = apfloat::unbiased_exponent<8, 23>(apfloat::one<8, 23>(false));
    assert_eq(actual, expected);

    let actual = apfloat::unbiased_exponent<8, 23>(apfloat::one<8, 23>(true));
    assert_eq(actual, expected);
}

#[test]
fn unbiased_exponent_two_test() {
    let expected = s8:1;
    let two = apfloat::APFloat<8, 23> { sign: 0, bexp: 128, fraction: 0 };
    let actual = apfloat::unbiased_exponent<8, 23>(two);
    assert_eq(actual, expected);

    let minus_two = apfloat::APFloat<8, 23> { sign: 1, bexp: 128, fraction: 0 };
    let actual = apfloat::unbiased_exponent<8, 23>(minus_two);
    assert_eq(actual, expected);
}

#[test]
fn unbiased_exponent_half_test() {
    let half = apfloat::APFloat<8, 23> { sign: 0, bexp: 126, fraction: 0 };
    let expected = s8:-1;
    let actual = apfloat::unbiased_exponent<8, 23>(half);
    assert_eq(actual, expected);

    let minus_half = apfloat::APFloat<8, 23> { sign: 0, bexp: 126, fraction: 0 };
    let actual = apfloat::unbiased_exponent<8, 23>(minus_half);
    assert_eq(actual, expected);
}

#[test]
fn unbiased_exponent_subnormal_test() {
    let expected = s8:-127;
    let actual = apfloat::unbiased_exponent<8, 23>(
        apfloat::APFloat<8, 23> { sign: 0, bexp: 0, fraction: 42 });
    assert_eq(actual, expected);
}

#[test]
fn unbiased_exponent_inf_nan_test() {
    let expected = s8:-128;

    // inf
    let actual = apfloat::unbiased_exponent(apfloat::inf<8, 23>(false));
    assert_eq(actual, expected);

    // -inf
    let actual = apfloat::unbiased_exponent(apfloat::inf<8, 23>(true));
    assert_eq(actual, expected);

    // NaN
    let actual = apfloat::unbiased_exponent(apfloat::qnan<8, 23>());
    assert_eq(actual, expected);
}

#[test]
fn bias_test() {
    // Normal.
    let expected = u8:127;
    let actual = apfloat::bias<8>(0);
    assert_eq(expected, actual);

    // Inf or NaN.
    let expected = u8:255;
    let actual = apfloat::bias<8>(-128);
    assert_eq(expected, actual);

    // Zero or subnormal.
    let expected = u8:0;
    let actual = apfloat::bias<8>(-127);
    assert_eq(expected, actual);
}

#[test]
fn cast_from_fixed_using_rne_test() {
    // Zero is a special case.
    let zero_float = apfloat::zero<4, 4>(0);
    assert_eq(apfloat::cast_from_fixed_using_rne<4, 4>(sN[32]:0), zero_float);

    // +/-1
    let one_float = apfloat::one<4, 4>(0);
    assert_eq(apfloat::cast_from_fixed_using_rne<4, 4>(sN[32]:1), one_float);
    let none_float = apfloat::one<4, 4>(1);
    assert_eq(apfloat::cast_from_fixed_using_rne<4, 4>(sN[32]:-1), none_float);

    // +/-4
    let four_float = apfloat::APFloat<4, 4> { sign: 0, bexp: 9, fraction: 0 };
    assert_eq(apfloat::cast_from_fixed_using_rne<4, 4>(sN[32]:4), four_float);
    let nfour_float = apfloat::APFloat<4, 4> { sign: 1, bexp: 9, fraction: 0 };
    assert_eq(apfloat::cast_from_fixed_using_rne<4, 4>(sN[32]:-4), nfour_float);

    // Cast maximum representable exponent in target format.
    let max_representable = apfloat::APFloat<4, 4> { sign: 0, bexp: 14, fraction: 0 };
    assert_eq(apfloat::cast_from_fixed_using_rne<4, 4>(sN[32]:128), max_representable);

    // Cast minimum non-representable exponent in target format.
    assert_eq(apfloat::cast_from_fixed_using_rne<4, 4>(sN[32]:256), apfloat::inf<4, 4>(0));

    // Test rounding - maximum truncated bits that will round down, even fraction.
    let truncate = apfloat::APFloat<4, 4> { sign: 0, bexp: 14, fraction: 0 };
    assert_eq(apfloat::cast_from_fixed_using_rne<4, 4>(sN[32]:131), truncate);

    // Test rounding - maximum truncated bits that will round down, odd fraction.
    let truncate = apfloat::APFloat<4, 4> { sign: 0, bexp: 14, fraction: 1 };
    assert_eq(apfloat::cast_from_fixed_using_rne<4, 4>(sN[32]:139), truncate);

    // Test rounding - halfway and already even, round down
    let truncate = apfloat::APFloat<4, 4> { sign: 0, bexp: 14, fraction: 0 };
    assert_eq(apfloat::cast_from_fixed_using_rne<4, 4>(sN[32]:132), truncate);

    // Test rounding - halfway and odd, round up
    let round_up = apfloat::APFloat<4, 4> { sign: 0, bexp: 14, fraction: 2 };
    assert_eq(apfloat::cast_from_fixed_using_rne<4, 4>(sN[32]:140), round_up);

    // Test rounding - over halfway and even, round up
    let round_up = apfloat::APFloat<4, 4> { sign: 0, bexp: 14, fraction: 1 };
    assert_eq(apfloat::cast_from_fixed_using_rne<4, 4>(sN[32]:133), round_up);

    // Test rounding - over halfway and odd, round up
    let round_up = apfloat::APFloat<4, 4> { sign: 0, bexp: 14, fraction: 2 };
    assert_eq(apfloat::cast_from_fixed_using_rne<4, 4>(sN[32]:141), round_up);

    // Test rounding - Rounding up increases exponent.
    let round_inc_exponent = apfloat::APFloat<4, 4> { sign: 0, bexp: 14, fraction: 0 };
    assert_eq(apfloat::cast_from_fixed_using_rne<4, 4>(sN[32]:126), round_inc_exponent);
    assert_eq(apfloat::cast_from_fixed_using_rne<4, 4>(sN[32]:127), round_inc_exponent);

    // Test rounding - Rounding up overflows to infinity.
    assert_eq(apfloat::cast_from_fixed_using_rne<4, 4>(sN[32]:252), apfloat::inf<4, 4>(0));
    assert_eq(apfloat::cast_from_fixed_using_rne<4, 4>(sN[32]:254), apfloat::inf<4, 4>(0));
}

#[test]
fn cast_from_fixed_using_rne_exp_overflow_test() {
    // Regression test: when NUM_SRC_BITS > 2^(EXP_SZ + 1), num_trailing_nonzeros must not wrap
    // around when computing the unbiased exponent (e.g., 1 << 32 has 33 trailing bits, which
    // previously truncated to 1 in u5 and returned 1.0 instead of +inf).
    assert_eq(apfloat::cast_from_fixed_using_rne<4, 4>(s64:1 << 32), apfloat::inf<4, 4>(false));
    assert_eq(apfloat::cast_from_fixed_using_rne<4, 4>(-(s64:1 << 32)), apfloat::inf<4, 4>(true));
    assert_eq(apfloat::cast_from_fixed_using_rne<4, 4>(s64:1 << 33), apfloat::inf<4, 4>(false));
    assert_eq(
        apfloat::cast_from_fixed_using_rne<4, 4>(std::signed_min_value<64>()),
        apfloat::inf<4, 4>(true));
    assert_eq(
        apfloat::cast_from_fixed_using_rne<4, 4>(std::signed_max_value<64>()),
        apfloat::inf<4, 4>(false));
    // Also check FP8 E4M3 (EXP_SZ=4, FRACTION_SZ=3) and HF16 (EXP_SZ=5, FRACTION_SZ=10) with s65.
    assert_eq(apfloat::cast_from_fixed_using_rne<4, 3>(s64:1 << 32), apfloat::inf<4, 3>(false));
    assert_eq(apfloat::cast_from_fixed_using_rne<5, 10>(sN[65]:1 << 63), apfloat::inf<5, 10>(false));
    assert_eq(
        apfloat::cast_from_fixed_using_rne<5, 10>(std::signed_min_value<65>()),
        apfloat::inf<5, 10>(true));
}

#[test]
fn cast_from_fixed_using_rz_test() {
    const EXP_SZ = u32:5;
    const FRAC_SZ = u32:5;
    // This gives us the maximum number of bits before things go to infinity
    assert_eq(apfloat::max_normal_exp<EXP_SZ>(), sN[EXP_SZ]:15);

    type Float = apfloat::APFloat<EXP_SZ, FRAC_SZ>;
    assert_eq(
        apfloat::cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(sN[17]:0),
        apfloat::zero<EXP_SZ, FRAC_SZ>(false));
    assert_eq(
        apfloat::cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(sN[17]:1),
        apfloat::one<EXP_SZ, FRAC_SZ>(false));
    assert_eq(
        apfloat::cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(-sN[17]:1),
        apfloat::one<EXP_SZ, FRAC_SZ>(true));

    assert_eq(
        apfloat::cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(sN[17]:2),
        Float { sign: false, bexp: apfloat::bias(sN[EXP_SZ]:1), fraction: 0 });

    assert_eq(
        apfloat::cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(-sN[17]:2),
        Float { sign: true, bexp: apfloat::bias(sN[EXP_SZ]:1), fraction: 0 });

    assert_eq(
        apfloat::cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(sN[17]:3),
        Float { sign: false, bexp: apfloat::bias(sN[EXP_SZ]:1), fraction: 0b10000 });

    assert_eq(
        apfloat::cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(-sN[17]:3),
        Float { sign: true, bexp: apfloat::bias(sN[EXP_SZ]:1), fraction: 0b10000 });

    assert_eq(
        apfloat::cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(sN[17]:0b111000),
        Float { sign: false, bexp: apfloat::bias(sN[EXP_SZ]:5), fraction: 0b11000 });

    assert_eq(
        apfloat::cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(-sN[17]:0b111000),
        Float { sign: true, bexp: apfloat::bias(sN[EXP_SZ]:5), fraction: 0b11000 });

    assert_eq(
        apfloat::cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(sN[17]:0b1110000),
        Float { sign: false, bexp: apfloat::bias(sN[EXP_SZ]:6), fraction: 0b11000 });

    assert_eq(
        apfloat::cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(-sN[17]:0b1110000),
        Float { sign: true, bexp: apfloat::bias(sN[EXP_SZ]:6), fraction: 0b11000 });

    assert_eq(
        apfloat::cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(sN[17]:0b111111),
        Float { sign: false, bexp: apfloat::bias(sN[EXP_SZ]:5), fraction: 0b11111 });

    assert_eq(
        apfloat::cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(-sN[17]:0b111111),
        Float { sign: true, bexp: apfloat::bias(sN[EXP_SZ]:5), fraction: 0b11111 });

    assert_eq(
        apfloat::cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(sN[17]:0b1111110),
        Float { sign: false, bexp: apfloat::bias(sN[EXP_SZ]:6), fraction: 0b11111 });

    assert_eq(
        apfloat::cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(-sN[17]:0b1111110),
        Float { sign: true, bexp: apfloat::bias(sN[EXP_SZ]:6), fraction: 0b11111 });

    assert_eq(
        apfloat::cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(sN[17]:0b1111111),
        Float { sign: false, bexp: apfloat::bias(sN[EXP_SZ]:6), fraction: 0b11111 });

    assert_eq(
        apfloat::cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(-sN[17]:0b1111111),
        Float { sign: true, bexp: apfloat::bias(sN[EXP_SZ]:6), fraction: 0b11111 });

    assert_eq(
        apfloat::cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(sN[17]:0b01111111111111111),
        Float { sign: false, bexp: apfloat::bias(sN[EXP_SZ]:15), fraction: 0b11111 });

    assert_eq(
        apfloat::cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(-sN[17]:0b01111111111111111),
        Float { sign: true, bexp: apfloat::bias(sN[EXP_SZ]:15), fraction: 0b11111 });

    assert_eq(
        apfloat::cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(sN[17]:0b00000011111111111),
        Float { sign: false, bexp: apfloat::bias(sN[EXP_SZ]:10), fraction: 0b11111 });

    assert_eq(
        apfloat::cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(-sN[17]:0b00000011111111111),
        Float { sign: true, bexp: apfloat::bias(sN[EXP_SZ]:10), fraction: 0b11111 });

    assert_eq(
        apfloat::cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(sN[17]:0b00000011111111000),
        Float { sign: false, bexp: apfloat::bias(sN[EXP_SZ]:10), fraction: 0b11111 });

    assert_eq(
        apfloat::cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(-sN[17]:0b00000011111111000),
        Float { sign: true, bexp: apfloat::bias(sN[EXP_SZ]:10), fraction: 0b11111 });

    assert_eq(
        apfloat::cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(sN[20]:0b01000000000000000),
        Float { sign: false, bexp: apfloat::bias(sN[EXP_SZ]:15), fraction: 0 });

    assert_eq(
        apfloat::cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(-sN[20]:0b01000000000000000),
        Float { sign: true, bexp: apfloat::bias(sN[EXP_SZ]:15), fraction: 0 });

    assert_eq(
        apfloat::cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(sN[20]:0b010000000000000000),
        apfloat::max_normal<EXP_SZ, FRAC_SZ>(false));

    assert_eq(
        apfloat::cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(-sN[20]:0b010000000000000000),
        apfloat::max_normal<EXP_SZ, FRAC_SZ>(true));

    assert_eq(
        apfloat::cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(sN[20]:0b010000111000000000),
        apfloat::max_normal<EXP_SZ, FRAC_SZ>(false));

    assert_eq(
        apfloat::cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(-sN[20]:0b010001110000000000),
        apfloat::max_normal<EXP_SZ, FRAC_SZ>(true));

    assert_eq(
        apfloat::cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(sN[30]:0b010000111000000000),
        apfloat::max_normal<EXP_SZ, FRAC_SZ>(false));

    assert_eq(
        apfloat::cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(-sN[30]:0b010001110000000000),
        apfloat::max_normal<EXP_SZ, FRAC_SZ>(true));
}

#[test]
fn cast_from_fixed_using_rz_exp_overflow_test() {
    // Regression test: when NUM_SRC_BITS > 2^(EXP_SZ + 1), num_trailing_nonzeros must not wrap
    // around when computing the unbiased exponent.
    assert_eq(
        apfloat::cast_from_fixed_using_rz<4, 4>(s64:1 << 32), apfloat::max_normal<4, 4>(false));
    assert_eq(
        apfloat::cast_from_fixed_using_rz<4, 4>(-(s64:1 << 32)), apfloat::max_normal<4, 4>(true));
    assert_eq(
        apfloat::cast_from_fixed_using_rz<4, 4>(s64:1 << 33), apfloat::max_normal<4, 4>(false));
    assert_eq(
        apfloat::cast_from_fixed_using_rz<4, 4>(std::signed_min_value<64>()),
        apfloat::max_normal<4, 4>(true));
    assert_eq(
        apfloat::cast_from_fixed_using_rz<4, 4>(std::signed_max_value<64>()),
        apfloat::max_normal<4, 4>(false));
    // Also check FP8 E4M3 (EXP_SZ=4, FRACTION_SZ=3) and HF16 (EXP_SZ=5, FRACTION_SZ=10) with s65.
    assert_eq(
        apfloat::cast_from_fixed_using_rz<4, 3>(s64:1 << 32), apfloat::max_normal<4, 3>(false));
    assert_eq(
        apfloat::cast_from_fixed_using_rz<5, 10>(sN[65]:1 << 63), apfloat::max_normal<5, 10>(false));
    assert_eq(
        apfloat::cast_from_fixed_using_rz<5, 10>(std::signed_min_value<65>()),
        apfloat::max_normal<5, 10>(true));
}

#[test]
fn upcast_with_denorms_test() {
    const BF16_EXP_SZ = u32:8;
    const BF16_FRACTION_SZ = u32:7;
    const F64_EXP_SZ = u32:11;
    const F64_FRACTION_SZ = u32:52;

    let one_bf16 = apfloat::one<BF16_EXP_SZ, BF16_FRACTION_SZ>(0);
    let one_f64 = apfloat::one<F64_EXP_SZ, F64_FRACTION_SZ>(0);
    let one_dot_5_bf16 = apfloat::APFloat<BF16_EXP_SZ, BF16_FRACTION_SZ> {
        fraction: u7:1 << (BF16_FRACTION_SZ - 1),
        ..one_bf16
    };
    let one_dot_5_f64 = apfloat::APFloat<F64_EXP_SZ, F64_FRACTION_SZ> {
        fraction: u52:1 << (F64_FRACTION_SZ - 1),
        ..one_f64
    };
    let zero_f64 = apfloat::zero<F64_EXP_SZ, F64_FRACTION_SZ>(0);
    let neg_zero_f64 = apfloat::zero<F64_EXP_SZ, F64_FRACTION_SZ>(1);
    let zero_bf16 = apfloat::zero<BF16_EXP_SZ, BF16_FRACTION_SZ>(0);
    let neg_zero_bf16 = apfloat::zero<BF16_EXP_SZ, BF16_FRACTION_SZ>(1);

    assert_eq(apfloat::upcast_with_denorms<F64_EXP_SZ, F64_FRACTION_SZ>(one_bf16), one_f64);
    assert_eq(apfloat::upcast_with_denorms<F64_EXP_SZ, F64_FRACTION_SZ>(zero_bf16), zero_f64);
    assert_eq(
        apfloat::upcast_with_denorms<F64_EXP_SZ, F64_FRACTION_SZ>(neg_zero_bf16), neg_zero_f64);
    assert_eq(
        apfloat::upcast_with_denorms<F64_EXP_SZ, F64_FRACTION_SZ>(one_dot_5_bf16), one_dot_5_f64);
    assert_eq(
        apfloat::upcast_with_denorms<F64_EXP_SZ, F64_FRACTION_SZ>(
            apfloat::qnan<BF16_EXP_SZ, BF16_FRACTION_SZ>()),
        apfloat::qnan<F64_EXP_SZ, F64_FRACTION_SZ>());
    assert_eq(
        apfloat::upcast_with_denorms<F64_EXP_SZ, F64_FRACTION_SZ>(
            apfloat::inf<BF16_EXP_SZ, BF16_FRACTION_SZ>(0)),
        apfloat::inf<F64_EXP_SZ, F64_FRACTION_SZ>(0));
    assert_eq(
        apfloat::upcast_with_denorms<F64_EXP_SZ, F64_FRACTION_SZ>(
            apfloat::inf<BF16_EXP_SZ, BF16_FRACTION_SZ>(1)),
        apfloat::inf<F64_EXP_SZ, F64_FRACTION_SZ>(1));
    assert_eq(
        apfloat::upcast_with_denorms<F64_EXP_SZ, F64_FRACTION_SZ>(
            apfloat::zero<BF16_EXP_SZ, BF16_FRACTION_SZ>(0)),
        apfloat::zero<F64_EXP_SZ, F64_FRACTION_SZ>(0));
    assert_eq(
        apfloat::upcast_with_denorms<F64_EXP_SZ, F64_FRACTION_SZ>(
            apfloat::zero<BF16_EXP_SZ, BF16_FRACTION_SZ>(1)),
        apfloat::zero<F64_EXP_SZ, F64_FRACTION_SZ>(1));

    // same size
    assert_eq(apfloat::upcast_with_denorms<BF16_EXP_SZ, BF16_FRACTION_SZ>(zero_bf16), zero_bf16);
    assert_eq(
        apfloat::upcast_with_denorms<BF16_EXP_SZ, BF16_FRACTION_SZ>(neg_zero_bf16), neg_zero_bf16);
    assert_eq(apfloat::upcast_with_denorms<BF16_EXP_SZ, BF16_FRACTION_SZ>(one_bf16), one_bf16);
    assert_eq(
        apfloat::upcast_with_denorms<BF16_EXP_SZ, BF16_FRACTION_SZ>(one_dot_5_bf16), one_dot_5_bf16);
}

#[test]
fn upcast_with_denorms_denormal_test() {
    const BF16_EXP_SZ = u32:8;
    const BF16_FRACTION_SZ = u32:7;
    const F64_EXP_SZ = u32:11;
    const F64_FRACTION_SZ = u32:52;

    let denormal_bf16 = apfloat::APFloat {
        sign: u1:0,
        bexp: uN[BF16_EXP_SZ]:0,
        fraction: uN[BF16_FRACTION_SZ]:1 << (BF16_FRACTION_SZ - 1) | 1,
    };
    let neg_denormal_bf16 = apfloat::APFloat { sign: u1:1, ..denormal_bf16 };
    assert_eq(
        apfloat::upcast_with_denorms<BF16_EXP_SZ, BF16_FRACTION_SZ>(denormal_bf16), denormal_bf16);
    assert_eq(
        apfloat::upcast_with_denorms<BF16_EXP_SZ, BF16_FRACTION_SZ>(neg_denormal_bf16),
        neg_denormal_bf16);
    assert_eq(
        apfloat::upcast_with_denorms<F64_EXP_SZ, F64_FRACTION_SZ>(denormal_bf16),
        apfloat::APFloat {
            sign: u1:0,
            bexp: uN[F64_EXP_SZ]:0x380,
            fraction: uN[F64_FRACTION_SZ]:0x400000000000,
        });
    assert_eq(
        apfloat::upcast_with_denorms<F64_EXP_SZ, F64_FRACTION_SZ>(neg_denormal_bf16),
        apfloat::APFloat {
            sign: u1:1,
            bexp: uN[F64_EXP_SZ]:0x380,
            fraction: uN[F64_FRACTION_SZ]:0x400000000000,
        });

    let min_plus_denormal_bf16 =
        apfloat::APFloat { sign: u1:0, bexp: uN[BF16_EXP_SZ]:0, fraction: uN[BF16_FRACTION_SZ]:1 };

    let max_plus_denormal_bf16 = apfloat::APFloat {
        sign: u1:0,
        bexp: uN[BF16_EXP_SZ]:0,
        fraction: std::mask_bits<BF16_FRACTION_SZ>(),
    };
    assert_eq(
        apfloat::upcast_with_denorms<F64_EXP_SZ, F64_FRACTION_SZ>(min_plus_denormal_bf16),
        apfloat::APFloat { sign: u1:0, bexp: uN[F64_EXP_SZ]:0x37a, fraction: uN[F64_FRACTION_SZ]:0 });
    assert_eq(
        apfloat::upcast_with_denorms<F64_EXP_SZ, F64_FRACTION_SZ>(max_plus_denormal_bf16),
        apfloat::APFloat {
            sign: u1:0,
            bexp: uN[F64_EXP_SZ]:0x380,
            fraction: uN[F64_FRACTION_SZ]:0xfc00000000000,
        });

    // same exponent size upcast.
    const F32_EXP_SZ = u32:8;
    const F32_FRACTION_SZ = u32:23;
    assert_eq(
        apfloat::upcast_with_denorms<F32_EXP_SZ, F32_FRACTION_SZ>(min_plus_denormal_bf16),
        apfloat::APFloat {
            sign: u1:0,
            bexp: uN[F32_EXP_SZ]:0,
            fraction: uN[F32_FRACTION_SZ]:0x10000,
        });
    assert_eq(
        apfloat::upcast_with_denorms<F32_EXP_SZ, F32_FRACTION_SZ>(max_plus_denormal_bf16),
        apfloat::APFloat {
            sign: u1:0,
            bexp: uN[F32_EXP_SZ]:0,
            fraction: uN[F32_FRACTION_SZ]:0x7f0000,
        });

    // same fraction size upcast.
    assert_eq(
        apfloat::upcast_with_denorms<F64_EXP_SZ, BF16_FRACTION_SZ>(min_plus_denormal_bf16),
        apfloat::APFloat {
            sign: u1:0,
            bexp: uN[F64_EXP_SZ]:0x37a,
            fraction: uN[BF16_FRACTION_SZ]:0,
        });
    assert_eq(
        apfloat::upcast_with_denorms<F64_EXP_SZ, BF16_FRACTION_SZ>(max_plus_denormal_bf16),
        apfloat::APFloat {
            sign: u1:0,
            bexp: uN[F64_EXP_SZ]:0x380,
            fraction: uN[BF16_FRACTION_SZ]:0x7e,
        });
}

#[test]
fn upcast_daz_test() {
    const BF16_EXP_SZ = u32:8;
    const BF16_FRACTION_SZ = u32:7;
    const F64_EXP_SZ = u32:11;
    const F64_FRACTION_SZ = u32:52;

    let one_bf16 = apfloat::one<BF16_EXP_SZ, BF16_FRACTION_SZ>(0);
    let one_f64 = apfloat::one<F64_EXP_SZ, F64_FRACTION_SZ>(0);
    let one_dot_5_bf16 = apfloat::APFloat<BF16_EXP_SZ, BF16_FRACTION_SZ> {
        fraction: u7:1 << (BF16_FRACTION_SZ - 1),
        ..one_bf16
    };
    let one_dot_5_f64 = apfloat::APFloat<F64_EXP_SZ, F64_FRACTION_SZ> {
        fraction: u52:1 << (F64_FRACTION_SZ - 1),
        ..one_f64
    };
    let zero_f64 = apfloat::zero<F64_EXP_SZ, F64_FRACTION_SZ>(0);
    let neg_zero_f64 = apfloat::zero<F64_EXP_SZ, F64_FRACTION_SZ>(1);
    let zero_bf16 = apfloat::zero<BF16_EXP_SZ, BF16_FRACTION_SZ>(0);
    let neg_zero_bf16 = apfloat::zero<BF16_EXP_SZ, BF16_FRACTION_SZ>(1);

    assert_eq(apfloat::upcast_daz<F64_EXP_SZ, F64_FRACTION_SZ>(one_bf16), one_f64);
    assert_eq(apfloat::upcast_daz<F64_EXP_SZ, F64_FRACTION_SZ>(zero_bf16), zero_f64);
    assert_eq(apfloat::upcast_daz<F64_EXP_SZ, F64_FRACTION_SZ>(neg_zero_bf16), neg_zero_f64);
    assert_eq(apfloat::upcast_daz<F64_EXP_SZ, F64_FRACTION_SZ>(one_dot_5_bf16), one_dot_5_f64);
    assert_eq(
        apfloat::upcast_daz<F64_EXP_SZ, F64_FRACTION_SZ>(
            apfloat::qnan<BF16_EXP_SZ, BF16_FRACTION_SZ>()),
        apfloat::qnan<F64_EXP_SZ, F64_FRACTION_SZ>());
    assert_eq(
        apfloat::upcast_daz<F64_EXP_SZ, F64_FRACTION_SZ>(
            apfloat::inf<BF16_EXP_SZ, BF16_FRACTION_SZ>(0)),
        apfloat::inf<F64_EXP_SZ, F64_FRACTION_SZ>(0));
    assert_eq(
        apfloat::upcast_daz<F64_EXP_SZ, F64_FRACTION_SZ>(
            apfloat::inf<BF16_EXP_SZ, BF16_FRACTION_SZ>(1)),
        apfloat::inf<F64_EXP_SZ, F64_FRACTION_SZ>(1));
    assert_eq(
        apfloat::upcast_daz<F64_EXP_SZ, F64_FRACTION_SZ>(
            apfloat::zero<BF16_EXP_SZ, BF16_FRACTION_SZ>(0)),
        apfloat::zero<F64_EXP_SZ, F64_FRACTION_SZ>(0));
    assert_eq(
        apfloat::upcast_daz<F64_EXP_SZ, F64_FRACTION_SZ>(
            apfloat::zero<BF16_EXP_SZ, BF16_FRACTION_SZ>(1)),
        apfloat::zero<F64_EXP_SZ, F64_FRACTION_SZ>(1));

    // same size
    assert_eq(apfloat::upcast_daz<BF16_EXP_SZ, BF16_FRACTION_SZ>(zero_bf16), zero_bf16);
    assert_eq(apfloat::upcast_daz<BF16_EXP_SZ, BF16_FRACTION_SZ>(neg_zero_bf16), neg_zero_bf16);
    assert_eq(apfloat::upcast_daz<BF16_EXP_SZ, BF16_FRACTION_SZ>(one_bf16), one_bf16);
    assert_eq(apfloat::upcast_daz<BF16_EXP_SZ, BF16_FRACTION_SZ>(one_dot_5_bf16), one_dot_5_bf16);
}

#[test]
fn upcast_daz_denormal_test() {
    const BF16_EXP_SZ = u32:8;
    const BF16_FRACTION_SZ = u32:7;
    const F64_EXP_SZ = u32:11;
    const F64_FRACTION_SZ = u32:52;

    let zero_bf16 = apfloat::zero<BF16_EXP_SZ, BF16_FRACTION_SZ>(0);
    let neg_zero_bf16 = apfloat::zero<BF16_EXP_SZ, BF16_FRACTION_SZ>(1);
    let zero_f64 = apfloat::zero<F64_EXP_SZ, F64_FRACTION_SZ>(0);
    let neg_zero_f64 = apfloat::zero<F64_EXP_SZ, F64_FRACTION_SZ>(1);

    let denormal_bf16 = apfloat::APFloat {
        sign: u1:0,
        bexp: uN[BF16_EXP_SZ]:0,
        fraction: uN[BF16_FRACTION_SZ]:1 << (BF16_FRACTION_SZ - 1) | 1,
    };
    let neg_denormal_bf16 = apfloat::APFloat { sign: u1:1, ..denormal_bf16 };
    assert_eq(apfloat::upcast_daz<BF16_EXP_SZ, BF16_FRACTION_SZ>(denormal_bf16), zero_bf16);
    assert_eq(apfloat::upcast_daz<BF16_EXP_SZ, BF16_FRACTION_SZ>(neg_denormal_bf16), neg_zero_bf16);
    assert_eq(apfloat::upcast_daz<F64_EXP_SZ, F64_FRACTION_SZ>(denormal_bf16), zero_f64);
    assert_eq(apfloat::upcast_daz<F64_EXP_SZ, F64_FRACTION_SZ>(neg_denormal_bf16), neg_zero_f64);

    let min_plus_denormal_bf16 =
        apfloat::APFloat { sign: u1:0, bexp: uN[BF16_EXP_SZ]:0, fraction: uN[BF16_FRACTION_SZ]:1 };

    let max_plus_denormal_bf16 = apfloat::APFloat {
        sign: u1:0,
        bexp: uN[BF16_EXP_SZ]:0,
        fraction: std::mask_bits<BF16_FRACTION_SZ>(),
    };
    assert_eq(apfloat::upcast_daz<F64_EXP_SZ, F64_FRACTION_SZ>(min_plus_denormal_bf16), zero_f64);
    assert_eq(apfloat::upcast_daz<F64_EXP_SZ, F64_FRACTION_SZ>(max_plus_denormal_bf16), zero_f64);

    // same exponent size upcast.
    const F32_EXP_SZ = u32:8;
    const F32_FRACTION_SZ = u32:23;
    let zero_f32 = apfloat::zero<F32_EXP_SZ, F32_FRACTION_SZ>(0);
    assert_eq(apfloat::upcast_daz<F32_EXP_SZ, F32_FRACTION_SZ>(min_plus_denormal_bf16), zero_f32);
    assert_eq(apfloat::upcast_daz<F32_EXP_SZ, F32_FRACTION_SZ>(max_plus_denormal_bf16), zero_f32);

    // same fraction size upcast.
    assert_eq(
        apfloat::upcast_daz<F64_EXP_SZ, BF16_FRACTION_SZ>(min_plus_denormal_bf16),
        apfloat::zero<F64_EXP_SZ, BF16_FRACTION_SZ>(0));
    assert_eq(
        apfloat::upcast_daz<F64_EXP_SZ, BF16_FRACTION_SZ>(max_plus_denormal_bf16),
        apfloat::zero<F64_EXP_SZ, BF16_FRACTION_SZ>(0));
}

#[quickcheck]
fn does_downcast_stay_within_one_ulp_rounded(f: apfloat::APFloat<8, 13>) -> bool {
    type HF16 = apfloat::APFloat<5, 10>;
    let to_away = apfloat::downcast<HF16::FRACTION_SIZE, HF16::EXP_SIZE>(
        f, apfloat::RoundStyle::TIES_TO_AWAY);
    let to_even = apfloat::downcast<HF16::FRACTION_SIZE, HF16::EXP_SIZE>(
        f, apfloat::RoundStyle::TIES_TO_EVEN);

    let one_ulp_away = if apfloat::is_nan(to_away) {
        !apfloat::is_nan(to_even)
    } else if apfloat::is_inf(to_away) {
        // to-away rounded up to infinity but even didn't
        apfloat::unbiased_exponent(to_even) == apfloat::max_normal_exp<HF16::EXP_SIZE>() &&
        to_even.fraction == all_ones!<u10>()
    } else if to_even.bexp == 0 && to_even.fraction == 0 {
        // to_even rounded down to zero but to_away rounded away from it.
        to_away.bexp == 0 && to_away.fraction == 1
    } else if !apfloat::is_zero_or_subnormal(to_away) && apfloat::is_zero_or_subnormal(to_even) {
        // to-away rounded up to normal but even didn't
        to_away.bexp == 1 && to_away.fraction == 0 && to_even.fraction == all_ones!<u10>()
    } else if to_even.bexp != to_away.bexp {
        // to-away rounded up but to_even rounded down
        to_even.bexp + 1 == to_away.bexp && to_even.fraction == all_ones!<u10>() &&
        to_away.fraction == 0
    } else {
        // subnormals different rounding
        to_even.fraction + 1 == to_away.fraction
    };
    to_away.sign == to_even.sign && (to_away == to_even || one_ulp_away)
}

#[test]
fn downcast_special() {
    const F32_EXP_SZ = u32:8;
    const F32_FRACTION_SZ = u32:23;
    const FP16_EXP_SZ = u32:5;
    const FP16_FRACTION_SZ = u32:10;

    // qnan -> qnan
    let qnan_f32 = apfloat::qnan<F32_EXP_SZ, F32_FRACTION_SZ>();
    let qnan_fp16 = apfloat::qnan<FP16_EXP_SZ, FP16_FRACTION_SZ>();
    assert_eq(
        apfloat::downcast<FP16_FRACTION_SZ, FP16_EXP_SZ>(
            qnan_f32, apfloat::RoundStyle::TIES_TO_EVEN), qnan_fp16);
    assert_eq(
        apfloat::downcast<FP16_FRACTION_SZ, FP16_EXP_SZ>(
            qnan_f32, apfloat::RoundStyle::TIES_TO_AWAY), qnan_fp16);

    // inf -> inf
    let inf_f32 = apfloat::inf<F32_EXP_SZ, F32_FRACTION_SZ>(0);
    let inf_fp16 = apfloat::inf<FP16_EXP_SZ, FP16_FRACTION_SZ>(0);
    assert_eq(
        apfloat::downcast<FP16_FRACTION_SZ, FP16_EXP_SZ>(inf_f32, apfloat::RoundStyle::TIES_TO_EVEN),
        inf_fp16);
    assert_eq(
        apfloat::downcast<FP16_FRACTION_SZ, FP16_EXP_SZ>(inf_f32, apfloat::RoundStyle::TIES_TO_AWAY),
        inf_fp16);

    let minus_inf_f32 = apfloat::inf<F32_EXP_SZ, F32_FRACTION_SZ>(1);
    let minus_inf_fp16 = apfloat::inf<FP16_EXP_SZ, FP16_FRACTION_SZ>(1);
    assert_eq(
        apfloat::downcast<FP16_FRACTION_SZ, FP16_EXP_SZ>(
            minus_inf_f32, apfloat::RoundStyle::TIES_TO_EVEN), minus_inf_fp16);
    assert_eq(
        apfloat::downcast<FP16_FRACTION_SZ, FP16_EXP_SZ>(
            minus_inf_f32, apfloat::RoundStyle::TIES_TO_AWAY), minus_inf_fp16);

    // +/- 0.0 -> same-signed 0
    let zero_f32 = apfloat::zero<F32_EXP_SZ, F32_FRACTION_SZ>(0);
    let zero_fp16 = apfloat::zero<FP16_EXP_SZ, FP16_FRACTION_SZ>(0);
    assert_eq(
        apfloat::downcast<FP16_FRACTION_SZ, FP16_EXP_SZ>(
            zero_f32, apfloat::RoundStyle::TIES_TO_EVEN), zero_fp16);
    assert_eq(
        apfloat::downcast<FP16_FRACTION_SZ, FP16_EXP_SZ>(
            zero_f32, apfloat::RoundStyle::TIES_TO_AWAY), zero_fp16);

    let minus_zero_f32 = apfloat::zero<F32_EXP_SZ, F32_FRACTION_SZ>(1);
    let minus_zero_fp16 = apfloat::zero<FP16_EXP_SZ, FP16_FRACTION_SZ>(1);
    assert_eq(
        apfloat::downcast<FP16_FRACTION_SZ, FP16_EXP_SZ>(
            minus_zero_f32, apfloat::RoundStyle::TIES_TO_EVEN), minus_zero_fp16);
    assert_eq(
        apfloat::downcast<FP16_FRACTION_SZ, FP16_EXP_SZ>(
            minus_zero_f32, apfloat::RoundStyle::TIES_TO_AWAY), minus_zero_fp16);

    // stays normalized
    type F32 = apfloat::APFloat<F32_EXP_SZ, F32_FRACTION_SZ>;
    const SMALLER_EXP_SZ = F32_EXP_SZ - 4;
    type Smaller = apfloat::APFloat<SMALLER_EXP_SZ, F32_FRACTION_SZ>;
    let f = F32 {
        sign: 0,
        bexp: apfloat::bias<F32_EXP_SZ>(apfloat::min_normal_exp<SMALLER_EXP_SZ>() as s8),
        fraction: 0x123,
    };
    let exp = Smaller { sign: 0, bexp: 1, fraction: 0x123 };
    assert_eq(
        apfloat::downcast<F32_FRACTION_SZ, SMALLER_EXP_SZ>(f, apfloat::RoundStyle::TIES_TO_EVEN),
        exp);
    assert_eq(
        apfloat::downcast<F32_FRACTION_SZ, SMALLER_EXP_SZ>(f, apfloat::RoundStyle::TIES_TO_AWAY),
        exp);
}

#[test]
fn downcast_generates_subnormal() {
    // unbiased exponent is -1, 0, inf, or subnormal/zero
    type E2F5 = apfloat::APFloat<2, 5>;
    type E4F6 = apfloat::APFloat<4, 6>;

    // binary float 0b0.001000
    let not_subnormal = E4F6 {
        sign: false,
        bexp: apfloat::bias<E4F6::EXP_SIZE>(apfloat::min_normal_exp<E2F5::EXP_SIZE>() as s4 - 3),
        fraction: 0b000000,
    };
    let expected = E2F5 { sign: false, bexp: 0, fraction: 0b00100 };
    assert_eq(apfloat::downcast<5, 2>(not_subnormal, apfloat::RoundStyle::TIES_TO_EVEN), expected);
    assert_eq(apfloat::downcast<5, 2>(not_subnormal, apfloat::RoundStyle::TIES_TO_AWAY), expected);

    // Check rounding occurs.
    // slightly larger ranges
    type E6F8 = apfloat::APFloat<6, 8>;
    // Middle.
    let not_subnormal = E6F8 {
        sign: false,
        bexp: apfloat::bias<E6F8::EXP_SIZE>(apfloat::min_normal_exp<E4F6::EXP_SIZE>() as s6 - 3),
        // 0b1.0101_0000
        fraction: 0b0101_0000,
    };
    // 0b0.001010[10000]
    let expected_to_even = E4F6 { sign: false, bexp: 0, fraction: 0b00_1010 };
    let expected_to_away = E4F6 { sign: false, bexp: 0, fraction: 0b00_1011 };
    assert_eq(
        apfloat::downcast<6, 4>(not_subnormal, apfloat::RoundStyle::TIES_TO_EVEN), expected_to_even);
    assert_eq(
        apfloat::downcast<6, 4>(not_subnormal, apfloat::RoundStyle::TIES_TO_AWAY), expected_to_away);

    // always down.
    let not_subnormal = E6F8 {
        sign: false,
        bexp: apfloat::bias<E6F8::EXP_SIZE>(apfloat::min_normal_exp<E4F6::EXP_SIZE>() as s6 - 4),
        // 0b1.0101_0000
        fraction: 0b0101_0000,
    };
    // 0b0.000101[010000]
    let expected = E4F6 { sign: false, bexp: 0, fraction: 0b00_0101 };
    assert_eq(apfloat::downcast<6, 4>(not_subnormal, apfloat::RoundStyle::TIES_TO_EVEN), expected);
    assert_eq(apfloat::downcast<6, 4>(not_subnormal, apfloat::RoundStyle::TIES_TO_AWAY), expected);

    // always up.
    let not_subnormal = E6F8 {
        sign: false,
        bexp: apfloat::bias<E6F8::EXP_SIZE>(apfloat::min_normal_exp<E4F6::EXP_SIZE>() as s6 - 5),
        // 0b1.0101_0000
        fraction: 0b0101_0000,
    };
    // 0b0.000010[1010000]
    let expected = E4F6 { sign: false, bexp: 0, fraction: 0b00_0011 };
    assert_eq(apfloat::downcast<6, 4>(not_subnormal, apfloat::RoundStyle::TIES_TO_EVEN), expected);
    assert_eq(apfloat::downcast<6, 4>(not_subnormal, apfloat::RoundStyle::TIES_TO_AWAY), expected);

    // always up all the way off.
    let not_subnormal = E6F8 {
        sign: false,
        bexp: apfloat::bias<E6F8::EXP_SIZE>(apfloat::min_normal_exp<E4F6::EXP_SIZE>() as s6 - 7),
        // 0b1.0101_0000
        fraction: 0b0101_0000,
    };
    // 0b0.000000[101010000]
    let expected = E4F6 { sign: false, bexp: 0, fraction: 0b00_0001 };
    assert_eq(apfloat::downcast<6, 4>(not_subnormal, apfloat::RoundStyle::TIES_TO_EVEN), expected);
    assert_eq(apfloat::downcast<6, 4>(not_subnormal, apfloat::RoundStyle::TIES_TO_AWAY), expected);

    type F32 = apfloat::APFloat<8, 23>;
    type HF16 = apfloat::APFloat<5, 10>;
    let not_subnormal = F32 {
        sign: false,
        bexp: apfloat::bias<8>(apfloat::min_normal_exp<HF16::EXP_SIZE>() as s8 - 1),
        fraction: 0,
    };
    let expected = HF16 { sign: false, bexp: 0, fraction: 0b10_0000_0000 };
    assert_eq(apfloat::downcast<10, 5>(not_subnormal, apfloat::RoundStyle::TIES_TO_EVEN), expected);
    assert_eq(apfloat::downcast<10, 5>(not_subnormal, apfloat::RoundStyle::TIES_TO_AWAY), expected);

    let not_subnormal = F32 {
        sign: false,
        bexp: apfloat::bias<8>(apfloat::min_normal_exp<HF16::EXP_SIZE>() as s8 - 2),
        fraction: 0,
    };
    let expected = HF16 { sign: false, bexp: 0, fraction: 0b01_0000_0000 };
    assert_eq(apfloat::downcast<10, 5>(not_subnormal, apfloat::RoundStyle::TIES_TO_EVEN), expected);
    assert_eq(apfloat::downcast<10, 5>(not_subnormal, apfloat::RoundStyle::TIES_TO_AWAY), expected);

    let not_subnormal = F32 {
        sign: false,
        bexp: apfloat::bias<8>(apfloat::min_normal_exp<HF16::EXP_SIZE>() as s8 - 3),
        fraction: 0,
    };
    let expected = HF16 { sign: false, bexp: 0, fraction: 0b00_1000_0000 };
    assert_eq(apfloat::downcast<10, 5>(not_subnormal, apfloat::RoundStyle::TIES_TO_EVEN), expected);
    assert_eq(apfloat::downcast<10, 5>(not_subnormal, apfloat::RoundStyle::TIES_TO_AWAY), expected);

    let not_subnormal = F32 {
        sign: false,
        bexp: apfloat::bias<8>(apfloat::min_normal_exp<HF16::EXP_SIZE>() as s8 - 4),
        fraction: 0,
    };
    let expected = HF16 { sign: false, bexp: 0, fraction: 0b00_0100_0000 };
    assert_eq(apfloat::downcast<10, 5>(not_subnormal, apfloat::RoundStyle::TIES_TO_EVEN), expected);
    assert_eq(apfloat::downcast<10, 5>(not_subnormal, apfloat::RoundStyle::TIES_TO_AWAY), expected);

    let not_subnormal = F32 {
        sign: false,
        bexp: apfloat::bias<8>(apfloat::min_normal_exp<HF16::EXP_SIZE>() as s8 - 10),
        fraction: 0,
    };
    let expected = HF16 { sign: false, bexp: 0, fraction: 0b00_0000_0001 };
    assert_eq(apfloat::downcast<10, 5>(not_subnormal, apfloat::RoundStyle::TIES_TO_EVEN), expected);
    assert_eq(apfloat::downcast<10, 5>(not_subnormal, apfloat::RoundStyle::TIES_TO_AWAY), expected);

    let not_subnormal = F32 {
        sign: false,
        bexp: apfloat::bias<8>(apfloat::min_normal_exp<HF16::EXP_SIZE>() as s8 - 11),
        fraction: 0,
    };
    let expected_away = HF16 { sign: false, bexp: 0, fraction: 0b00_0000_0001 };
    let expected_even = HF16 { sign: false, bexp: 0, fraction: 0b00_0000_0000 };
    assert_eq(
        apfloat::downcast<10, 5>(not_subnormal, apfloat::RoundStyle::TIES_TO_EVEN), expected_even);
    assert_eq(
        apfloat::downcast<10, 5>(not_subnormal, apfloat::RoundStyle::TIES_TO_AWAY), expected_away);

    // Rounds up
    let not_subnormal = F32 {
        sign: false,
        bexp: apfloat::bias<8>(apfloat::min_normal_exp<HF16::EXP_SIZE>() as s8 - 3),
        fraction: u10:0b00_0000_0110 ++ u13:0,
    };
    let expected_away = HF16 { sign: false, bexp: 0, fraction: 0b00_1000_0001 };
    let expected_even = HF16 { sign: false, bexp: 0, fraction: 0b00_1000_0001 };
    assert_eq(
        apfloat::downcast<10, 5>(not_subnormal, apfloat::RoundStyle::TIES_TO_EVEN), expected_even);
    assert_eq(
        apfloat::downcast<10, 5>(not_subnormal, apfloat::RoundStyle::TIES_TO_AWAY), expected_away);

    // Rounds down
    let not_subnormal = F32 {
        sign: false,
        bexp: apfloat::bias<8>(apfloat::min_normal_exp<HF16::EXP_SIZE>() as s8 - 3),
        fraction: u10:0b00_0000_0011 ++ u13:0,
    };
    let expected_away = HF16 { sign: false, bexp: 0, fraction: 0b00_1000_0000 };
    let expected_even = HF16 { sign: false, bexp: 0, fraction: 0b00_1000_0000 };
    assert_eq(
        apfloat::downcast<10, 5>(not_subnormal, apfloat::RoundStyle::TIES_TO_EVEN), expected_even);
    assert_eq(
        apfloat::downcast<10, 5>(not_subnormal, apfloat::RoundStyle::TIES_TO_AWAY), expected_away);

    let not_subnormal = F32 {
        sign: false,
        bexp: apfloat::bias<8>(apfloat::min_normal_exp<HF16::EXP_SIZE>() as s8 - 3),
        fraction: u10:0b00_0000_0100 ++ u13:0,
    };
    let expected_away = HF16 { sign: false, bexp: 0, fraction: 0b00_1000_0001 };
    let expected_even = HF16 { sign: false, bexp: 0, fraction: 0b00_1000_0000 };
    assert_eq(
        apfloat::downcast<10, 5>(not_subnormal, apfloat::RoundStyle::TIES_TO_EVEN), expected_even);
    assert_eq(
        apfloat::downcast<10, 5>(not_subnormal, apfloat::RoundStyle::TIES_TO_AWAY), expected_away);

    let not_subnormal = F32 {
        sign: false,
        bexp: apfloat::bias<8>(apfloat::min_normal_exp<HF16::EXP_SIZE>() as s8 - 3),
        fraction: u10:0b00_0000_1100 ++ u13:0,
    };
    let expected_away = HF16 { sign: false, bexp: 0, fraction: 0b00_1000_0010 };
    let expected_even = HF16 { sign: false, bexp: 0, fraction: 0b00_1000_0010 };
    assert_eq(
        apfloat::downcast<10, 5>(not_subnormal, apfloat::RoundStyle::TIES_TO_EVEN), expected_even);
    assert_eq(
        apfloat::downcast<10, 5>(not_subnormal, apfloat::RoundStyle::TIES_TO_AWAY), expected_away);

    let not_subnormal = F32 {
        sign: false,
        bexp: apfloat::bias<8>(apfloat::min_normal_exp<HF16::EXP_SIZE>() as s8 - 3),
        fraction: 0b100_1001_0111_1011_1110_1001,
    };
    let expected_away = HF16 { sign: false, bexp: 0, fraction: 0b00_1100_1001 };
    let expected_even = HF16 { sign: false, bexp: 0, fraction: 0b00_1100_1001 };
    assert_eq(
        apfloat::downcast<10, 5>(not_subnormal, apfloat::RoundStyle::TIES_TO_EVEN), expected_even);
    assert_eq(
        apfloat::downcast<10, 5>(not_subnormal, apfloat::RoundStyle::TIES_TO_AWAY), expected_away);

    type F64 = apfloat::APFloat<11, 52>;
    let not_subnormal = F64 {
        sign: false,
        bexp: apfloat::bias<F64::EXP_SIZE>(apfloat::min_normal_exp<F32::EXP_SIZE>() as s11 - 3),
        fraction: 0b1000_1000_0110_0001_0010_1001_1010_0000_0001_0010_1100_1010_0001,
    };
    let expected = F32 { sign: false, bexp: 0, fraction: 0b001_1000_1000_0110_0001_0011 };
    assert_eq(apfloat::downcast<23, 8>(not_subnormal, apfloat::RoundStyle::TIES_TO_EVEN), expected);
    assert_eq(apfloat::downcast<23, 8>(not_subnormal, apfloat::RoundStyle::TIES_TO_AWAY), expected);
}

#[test]
fn downcast_rounds_subnormal_to_normal() {
    type E4F6 = apfloat::APFloat<4, 6>;
    type E6F8 = apfloat::APFloat<6, 8>;

    // binary float 0b0.000000000000111111111 = 0b1.11111111 * 2**-7
    let not_subnormal = E6F8 { sign: false, bexp: apfloat::bias<6>(-7), fraction: 0b1111_1111 };

    // ends up with a subnormal with 3 1s off the end of the mantisa and all
    // other bits 1. This rounds up to pushing it out of the subnormal regime.
    let expected = E4F6 { sign: false, bexp: apfloat::bias<4>(-6), fraction: 0 };
    assert_eq(apfloat::downcast<6, 4>(not_subnormal, apfloat::RoundStyle::TIES_TO_EVEN), expected);
    assert_eq(apfloat::downcast<6, 4>(not_subnormal, apfloat::RoundStyle::TIES_TO_AWAY), expected);
}

#[test]
fn downcast_raz_fp32_to_fp16_test() {
    const F32_EXP_SZ = u32:8;
    const F32_FRACTION_SZ = u32:23;
    const FP16_EXP_SZ = u32:5;
    const FP16_FRACTION_SZ = u32:10;

    // qnan -> qnan
    let qnan_f32 = apfloat::qnan<F32_EXP_SZ, F32_FRACTION_SZ>();
    let qnan_fp16 = apfloat::qnan<FP16_EXP_SZ, FP16_FRACTION_SZ>();
    assert_eq(
        apfloat::subnormals_to_zero(
            apfloat::downcast<FP16_FRACTION_SZ, FP16_EXP_SZ>(
                qnan_f32, apfloat::RoundStyle::TIES_TO_AWAY)), qnan_fp16);

    // inf -> inf
    let inf_f32 = apfloat::inf<F32_EXP_SZ, F32_FRACTION_SZ>(0);
    let inf_fp16 = apfloat::inf<FP16_EXP_SZ, FP16_FRACTION_SZ>(0);
    assert_eq(
        apfloat::subnormals_to_zero(
            apfloat::downcast<FP16_FRACTION_SZ, FP16_EXP_SZ>(
                inf_f32, apfloat::RoundStyle::TIES_TO_AWAY)), inf_fp16);

    let minus_inf_f32 = apfloat::inf<F32_EXP_SZ, F32_FRACTION_SZ>(1);
    let minus_inf_fp16 = apfloat::inf<FP16_EXP_SZ, FP16_FRACTION_SZ>(1);
    assert_eq(
        apfloat::subnormals_to_zero(
            apfloat::downcast<FP16_FRACTION_SZ, FP16_EXP_SZ>(
                minus_inf_f32, apfloat::RoundStyle::TIES_TO_AWAY)), minus_inf_fp16);

    // +/- 0.0 -> same-signed 0
    let zero_f32 = apfloat::zero<F32_EXP_SZ, F32_FRACTION_SZ>(0);
    let zero_fp16 = apfloat::zero<FP16_EXP_SZ, FP16_FRACTION_SZ>(0);
    assert_eq(
        apfloat::subnormals_to_zero(
            apfloat::downcast<FP16_FRACTION_SZ, FP16_EXP_SZ>(
                zero_f32, apfloat::RoundStyle::TIES_TO_AWAY)), zero_fp16);

    let minus_zero_f32 = apfloat::zero<F32_EXP_SZ, F32_FRACTION_SZ>(1);
    let minus_zero_fp16 = apfloat::zero<FP16_EXP_SZ, FP16_FRACTION_SZ>(1);
    assert_eq(
        apfloat::subnormals_to_zero(
            apfloat::downcast<FP16_FRACTION_SZ, FP16_EXP_SZ>(
                minus_zero_f32, apfloat::RoundStyle::TIES_TO_AWAY)), minus_zero_fp16);

    // subnormals flushed to same-signed 0.0
    let subnormal_f32 =
        apfloat::APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 0, bexp: 0, fraction: 0x7fffff };
    assert_eq(
        apfloat::subnormals_to_zero(
            apfloat::downcast<FP16_FRACTION_SZ, FP16_EXP_SZ>(
                subnormal_f32, apfloat::RoundStyle::TIES_TO_AWAY)), zero_fp16);

    let minus_subnormal_f32 =
        apfloat::APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 1, ..subnormal_f32 };
    assert_eq(
        apfloat::subnormals_to_zero(
            apfloat::downcast<FP16_FRACTION_SZ, FP16_EXP_SZ>(
                minus_subnormal_f32, apfloat::RoundStyle::TIES_TO_AWAY)), minus_zero_fp16);

    // normals
    let one_fp16 = apfloat::one<FP16_EXP_SZ, FP16_FRACTION_SZ>(0);
    let one_f32 = apfloat::one<F32_EXP_SZ, F32_FRACTION_SZ>(0);
    assert_eq(
        apfloat::subnormals_to_zero(
            apfloat::downcast<FP16_FRACTION_SZ, FP16_EXP_SZ>(
                one_f32, apfloat::RoundStyle::TIES_TO_AWAY)), one_fp16);

    let minus_one_fp16 = apfloat::APFloat<FP16_EXP_SZ, FP16_FRACTION_SZ> { sign: 1, ..one_fp16 };
    let minus_one_f32 = apfloat::APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 1, ..one_f32 };
    assert_eq(
        apfloat::subnormals_to_zero(
            apfloat::downcast<FP16_FRACTION_SZ, FP16_EXP_SZ>(
                minus_one_f32, apfloat::RoundStyle::TIES_TO_AWAY)), minus_one_fp16);

    // fraction with no rounding necessary
    let one_dot_5_f32 =
        apfloat::APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { fraction: 0x400000, ..one_f32 };
    let one_dot_5_fp16 =
        apfloat::APFloat<FP16_EXP_SZ, FP16_FRACTION_SZ> { fraction: 0x200, ..one_fp16 };
    assert_eq(
        apfloat::subnormals_to_zero(
            apfloat::downcast<FP16_FRACTION_SZ, FP16_EXP_SZ>(
                one_dot_5_f32, apfloat::RoundStyle::TIES_TO_AWAY)), one_dot_5_fp16);

    // rounds down
    let not_that_close_to_four_f32 =
        apfloat::APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 0, bexp: 0x80, fraction: 0x7fc000 };
    let almost_four_fp16 =
        apfloat::APFloat<FP16_EXP_SZ, FP16_FRACTION_SZ> { sign: 0, bexp: 0x10, fraction: 0x3fe };
    assert_eq(
        apfloat::subnormals_to_zero(
            apfloat::downcast<FP16_FRACTION_SZ, FP16_EXP_SZ>(
                not_that_close_to_four_f32, apfloat::RoundStyle::TIES_TO_AWAY)), almost_four_fp16);

    // rounds up tie to away
    let getting_closer_to_four_f32 =
        apfloat::APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 0, bexp: 0x80, fraction: 0x7fe000 };
    let four_fp16 =
        apfloat::APFloat<FP16_EXP_SZ, FP16_FRACTION_SZ> { sign: 0, bexp: 0x10, fraction: 0x3ff };
    assert_eq(
        apfloat::subnormals_to_zero(
            apfloat::downcast<FP16_FRACTION_SZ, FP16_EXP_SZ>(
                getting_closer_to_four_f32, apfloat::RoundStyle::TIES_TO_AWAY)), four_fp16);

    // rounds up tie to away.
    let even_closer_to_four_f32 =
        apfloat::APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 0, bexp: 0x80, fraction: 0x7ff000 };
    let four_fp16 =
        apfloat::APFloat<FP16_EXP_SZ, FP16_FRACTION_SZ> { sign: 0, bexp: 0x11, fraction: 0x0 };
    assert_eq(
        apfloat::subnormals_to_zero(
            apfloat::downcast<FP16_FRACTION_SZ, FP16_EXP_SZ>(
                even_closer_to_four_f32, apfloat::RoundStyle::TIES_TO_AWAY)), four_fp16);

    // rounds up
    let soooo_close_to_four_f32 =
        apfloat::APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 0, bexp: 0x80, fraction: 0x7ff100 };
    let four_fp16 =
        apfloat::APFloat<FP16_EXP_SZ, FP16_FRACTION_SZ> { sign: 0, bexp: 0x11, fraction: 0x0 };
    assert_eq(
        apfloat::subnormals_to_zero(
            apfloat::downcast<FP16_FRACTION_SZ, FP16_EXP_SZ>(
                soooo_close_to_four_f32, apfloat::RoundStyle::TIES_TO_AWAY)), four_fp16);

    // round up to inf
    let max_normal_f32 =
        apfloat::APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 0, bexp: 0xfe, fraction: 0x7fffff };
    assert_eq(
        apfloat::subnormals_to_zero(
            apfloat::downcast<FP16_FRACTION_SZ, FP16_EXP_SZ>(
                max_normal_f32, apfloat::RoundStyle::TIES_TO_AWAY)), inf_fp16);

    let minus_max_normal_f32 =
        apfloat::APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 1, bexp: 0xfe, fraction: 0x7fffff };
    let minus_inf_fp16 = apfloat::inf<FP16_EXP_SZ, FP16_FRACTION_SZ>(1);
    assert_eq(
        apfloat::subnormals_to_zero(
            apfloat::downcast<FP16_FRACTION_SZ, FP16_EXP_SZ>(
                minus_max_normal_f32, apfloat::RoundStyle::TIES_TO_AWAY)), minus_inf_fp16);
}

#[test]
fn downcast_rne_fp32_to_fp16_test() {
    const F32_EXP_SZ = u32:8;
    const F32_FRACTION_SZ = u32:23;
    const FP16_EXP_SZ = u32:5;
    const FP16_FRACTION_SZ = u32:10;

    // qnan -> qnan
    let qnan_f32 = apfloat::qnan<F32_EXP_SZ, F32_FRACTION_SZ>();
    let qnan_fp16 = apfloat::qnan<FP16_EXP_SZ, FP16_FRACTION_SZ>();
    assert_eq(apfloat::downcast_rne<FP16_FRACTION_SZ, FP16_EXP_SZ>(qnan_f32), qnan_fp16);

    // inf -> inf
    let inf_f32 = apfloat::inf<F32_EXP_SZ, F32_FRACTION_SZ>(0);
    let inf_fp16 = apfloat::inf<FP16_EXP_SZ, FP16_FRACTION_SZ>(0);
    assert_eq(apfloat::downcast_rne<FP16_FRACTION_SZ, FP16_EXP_SZ>(inf_f32), inf_fp16);

    let minus_inf_f32 = apfloat::inf<F32_EXP_SZ, F32_FRACTION_SZ>(1);
    let minus_inf_fp16 = apfloat::inf<FP16_EXP_SZ, FP16_FRACTION_SZ>(1);
    assert_eq(apfloat::downcast_rne<FP16_FRACTION_SZ, FP16_EXP_SZ>(minus_inf_f32), minus_inf_fp16);

    // +/- 0.0 -> same-signed 0
    let zero_f32 = apfloat::zero<F32_EXP_SZ, F32_FRACTION_SZ>(0);
    let zero_fp16 = apfloat::zero<FP16_EXP_SZ, FP16_FRACTION_SZ>(0);
    assert_eq(apfloat::downcast_rne<FP16_FRACTION_SZ, FP16_EXP_SZ>(zero_f32), zero_fp16);

    let minus_zero_f32 = apfloat::zero<F32_EXP_SZ, F32_FRACTION_SZ>(1);
    let minus_zero_fp16 = apfloat::zero<FP16_EXP_SZ, FP16_FRACTION_SZ>(1);
    assert_eq(apfloat::downcast_rne<FP16_FRACTION_SZ, FP16_EXP_SZ>(minus_zero_f32), minus_zero_fp16);

    // subnormals flushed to same-signed 0.0
    let subnormal_f32 =
        apfloat::APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 0, bexp: 0, fraction: 0x7fffff };
    assert_eq(apfloat::downcast_rne<FP16_FRACTION_SZ, FP16_EXP_SZ>(subnormal_f32), zero_fp16);

    let minus_subnormal_f32 =
        apfloat::APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 1, ..subnormal_f32 };
    assert_eq(
        apfloat::downcast_rne<FP16_FRACTION_SZ, FP16_EXP_SZ>(minus_subnormal_f32), minus_zero_fp16);

    // normals
    let one_fp16 = apfloat::one<FP16_EXP_SZ, FP16_FRACTION_SZ>(0);
    let one_f32 = apfloat::one<F32_EXP_SZ, F32_FRACTION_SZ>(0);
    assert_eq(apfloat::downcast_rne<FP16_FRACTION_SZ, FP16_EXP_SZ>(one_f32), one_fp16);

    let minus_one_fp16 = apfloat::APFloat<FP16_EXP_SZ, FP16_FRACTION_SZ> { sign: 1, ..one_fp16 };
    let minus_one_f32 = apfloat::APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 1, ..one_f32 };
    assert_eq(apfloat::downcast_rne<FP16_FRACTION_SZ, FP16_EXP_SZ>(minus_one_f32), minus_one_fp16);

    // fraction with no rounding necessary
    let one_dot_5_f32 =
        apfloat::APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { fraction: 0x400000, ..one_f32 };
    let one_dot_5_fp16 =
        apfloat::APFloat<FP16_EXP_SZ, FP16_FRACTION_SZ> { fraction: 0x200, ..one_fp16 };
    assert_eq(apfloat::downcast_rne<FP16_FRACTION_SZ, FP16_EXP_SZ>(one_dot_5_f32), one_dot_5_fp16);

    // rounds down
    let not_that_close_to_four_f32 =
        apfloat::APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 0, bexp: 0x80, fraction: 0x7fc000 };
    let almost_four_fp16 =
        apfloat::APFloat<FP16_EXP_SZ, FP16_FRACTION_SZ> { sign: 0, bexp: 0x10, fraction: 0x3fe };
    assert_eq(
        apfloat::downcast_rne<FP16_FRACTION_SZ, FP16_EXP_SZ>(not_that_close_to_four_f32),
        almost_four_fp16);

    // rounds down tie to even
    let getting_closer_to_four_f32 =
        apfloat::APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 0, bexp: 0x80, fraction: 0x7fe000 };
    let almost_four_fp16 =
        apfloat::APFloat<FP16_EXP_SZ, FP16_FRACTION_SZ> { sign: 0, bexp: 0x10, fraction: 0x3ff };
    assert_eq(
        apfloat::downcast_rne<FP16_FRACTION_SZ, FP16_EXP_SZ>(getting_closer_to_four_f32),
        almost_four_fp16);

    // rounds up tie to even.
    let even_closer_to_four_f32 =
        apfloat::APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 0, bexp: 0x80, fraction: 0x7ff000 };
    let four_fp16 =
        apfloat::APFloat<FP16_EXP_SZ, FP16_FRACTION_SZ> { sign: 0, bexp: 0x11, fraction: 0x0 };
    assert_eq(
        apfloat::downcast_rne<FP16_FRACTION_SZ, FP16_EXP_SZ>(even_closer_to_four_f32), four_fp16);

    // rounds up
    let soooo_close_to_four_f32 =
        apfloat::APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 0, bexp: 0x80, fraction: 0x7ff100 };
    let four_fp16 =
        apfloat::APFloat<FP16_EXP_SZ, FP16_FRACTION_SZ> { sign: 0, bexp: 0x11, fraction: 0x0 };
    assert_eq(
        apfloat::downcast_rne<FP16_FRACTION_SZ, FP16_EXP_SZ>(soooo_close_to_four_f32), four_fp16);

    // round up to inf
    let max_normal_f32 =
        apfloat::APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 0, bexp: 0xfe, fraction: 0x7fffff };
    assert_eq(apfloat::downcast_rne<FP16_FRACTION_SZ, FP16_EXP_SZ>(max_normal_f32), inf_fp16);

    let minus_max_normal_f32 =
        apfloat::APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 1, bexp: 0xfe, fraction: 0x7fffff };
    let minus_inf_fp16 = apfloat::inf<FP16_EXP_SZ, FP16_FRACTION_SZ>(1);
    assert_eq(
        apfloat::downcast_rne<FP16_FRACTION_SZ, FP16_EXP_SZ>(minus_max_normal_f32), minus_inf_fp16);
}

#[test]
fn downcast_rne_fp32_to_bf16_test() {
    const F32_EXP_SZ = u32:8;
    const F32_FRACTION_SZ = u32:23;
    const BF16_EXP_SZ = u32:8;
    const BF16_FRACTION_SZ = u32:7;

    // qnan -> qnan
    let qnan_f32 = apfloat::qnan<F32_EXP_SZ, F32_FRACTION_SZ>();
    let qnan_fp16 = apfloat::qnan<BF16_EXP_SZ, BF16_FRACTION_SZ>();
    assert_eq(apfloat::downcast_rne<BF16_FRACTION_SZ, BF16_EXP_SZ>(qnan_f32), qnan_fp16);

    // inf -> inf
    let inf_f32 = apfloat::inf<F32_EXP_SZ, F32_FRACTION_SZ>(0);
    let inf_fp16 = apfloat::inf<BF16_EXP_SZ, BF16_FRACTION_SZ>(0);
    assert_eq(apfloat::downcast_rne<BF16_FRACTION_SZ, BF16_EXP_SZ>(inf_f32), inf_fp16);

    let minus_inf_f32 = apfloat::inf<F32_EXP_SZ, F32_FRACTION_SZ>(1);
    let minus_inf_fp16 = apfloat::inf<BF16_EXP_SZ, BF16_FRACTION_SZ>(1);
    assert_eq(apfloat::downcast_rne<BF16_FRACTION_SZ, BF16_EXP_SZ>(minus_inf_f32), minus_inf_fp16);

    // +/- 0.0 -> same-signed 0
    let zero_f32 = apfloat::zero<F32_EXP_SZ, F32_FRACTION_SZ>(0);
    let zero_fp16 = apfloat::zero<BF16_EXP_SZ, BF16_FRACTION_SZ>(0);
    assert_eq(apfloat::downcast_rne<BF16_FRACTION_SZ, BF16_EXP_SZ>(zero_f32), zero_fp16);

    let minus_zero_f32 = apfloat::zero<F32_EXP_SZ, F32_FRACTION_SZ>(1);
    let minus_zero_fp16 = apfloat::zero<BF16_EXP_SZ, BF16_FRACTION_SZ>(1);
    assert_eq(apfloat::downcast_rne<BF16_FRACTION_SZ, BF16_EXP_SZ>(minus_zero_f32), minus_zero_fp16);

    // subnormals flushed to same-signed 0.0
    let subnormal_f32 =
        apfloat::APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 0, bexp: 0, fraction: 0x7fffff };
    assert_eq(apfloat::downcast_rne<BF16_FRACTION_SZ, BF16_EXP_SZ>(subnormal_f32), zero_fp16);

    let minus_subnormal_f32 =
        apfloat::APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 1, ..subnormal_f32 };
    assert_eq(
        apfloat::downcast_rne<BF16_FRACTION_SZ, BF16_EXP_SZ>(minus_subnormal_f32), minus_zero_fp16);

    // normals
    let one_bf16 = apfloat::one<BF16_EXP_SZ, BF16_FRACTION_SZ>(0);
    let one_f32 = apfloat::one<F32_EXP_SZ, F32_FRACTION_SZ>(0);
    assert_eq(apfloat::downcast_rne<BF16_FRACTION_SZ, BF16_EXP_SZ>(one_f32), one_bf16);

    let minus_one_bf16 = apfloat::APFloat<BF16_EXP_SZ, BF16_FRACTION_SZ> { sign: 1, ..one_bf16 };
    let minus_one_f32 = apfloat::APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 1, ..one_f32 };
    assert_eq(apfloat::downcast_rne<BF16_FRACTION_SZ, BF16_EXP_SZ>(minus_one_f32), minus_one_bf16);

    // fraction with no rounding necessary
    let one_dot_5_f32 =
        apfloat::APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { fraction: 0x400000, ..one_f32 };
    let one_dot_5_bf16 =
        apfloat::APFloat<BF16_EXP_SZ, BF16_FRACTION_SZ> { fraction: 0x40, ..one_bf16 };
    assert_eq(apfloat::downcast_rne<BF16_FRACTION_SZ, BF16_EXP_SZ>(one_dot_5_f32), one_dot_5_bf16);

    // rounds down
    let pi_f32 =
        apfloat::APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 0, bexp: 0x80, fraction: 0x490fdb };
    let pi_bf16 =
        apfloat::APFloat<BF16_EXP_SZ, BF16_FRACTION_SZ> { sign: 0, bexp: 0x80, fraction: 0x49 };
    assert_eq(apfloat::downcast_rne<BF16_FRACTION_SZ, BF16_EXP_SZ>(pi_f32), pi_bf16);

    // rounds up
    let one_third_f32 =
        apfloat::APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 0, bexp: 0x7d, fraction: 0x2aaaab };
    let one_third_bf16 =
        apfloat::APFloat<BF16_EXP_SZ, BF16_FRACTION_SZ> { sign: 0, bexp: 0x7d, fraction: 0x2b };
    assert_eq(apfloat::downcast_rne<BF16_FRACTION_SZ, BF16_EXP_SZ>(one_third_f32), one_third_bf16);

    // rounds down, tie to even
    let rne_down_f32 =
        apfloat::APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 0, bexp: 0x80, fraction: 0x408000 };
    let rne_down_bf16 =
        apfloat::APFloat<BF16_EXP_SZ, BF16_FRACTION_SZ> { sign: 0, bexp: 0x80, fraction: 0x40 };
    assert_eq(apfloat::downcast_rne<BF16_FRACTION_SZ, BF16_EXP_SZ>(rne_down_f32), rne_down_bf16);

    // rounds up, tie to even
    let rne_up_f32 =
        apfloat::APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 0, bexp: 0x80, fraction: 0x418000 };
    let rne_up_bf16 =
        apfloat::APFloat<BF16_EXP_SZ, BF16_FRACTION_SZ> { sign: 0, bexp: 0x80, fraction: 0x42 };
    assert_eq(apfloat::downcast_rne<BF16_FRACTION_SZ, BF16_EXP_SZ>(rne_up_f32), rne_up_bf16);

    // round up to inf
    let inf_bf16 = apfloat::inf<BF16_EXP_SZ, BF16_FRACTION_SZ>(0);
    let just_above_max_normal_bf16 =
        apfloat::APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 0, bexp: 0xfe, fraction: 0x7fe000 };
    assert_eq(
        apfloat::downcast_rne<BF16_FRACTION_SZ, BF16_EXP_SZ>(just_above_max_normal_bf16), inf_bf16);

    let just_below_max_normal_bf16 =
        apfloat::APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 1, bexp: 0xfe, fraction: 0x7fe000 };
    let minus_inf_bf16 = apfloat::inf<BF16_EXP_SZ, BF16_FRACTION_SZ>(1);
    assert_eq(
        apfloat::downcast_rne<BF16_FRACTION_SZ, BF16_EXP_SZ>(just_below_max_normal_bf16),
        minus_inf_bf16);

    // round up to inf
    let max_normal_f32 =
        apfloat::APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 0, bexp: 0xfe, fraction: 0x7fffff };
    assert_eq(apfloat::downcast_rne<BF16_FRACTION_SZ, BF16_EXP_SZ>(max_normal_f32), inf_bf16);

    let minus_max_normal_f32 =
        apfloat::APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 1, bexp: 0xfe, fraction: 0x7fffff };
    let minus_inf_bf16 = apfloat::inf<BF16_EXP_SZ, BF16_FRACTION_SZ>(1);
    assert_eq(
        apfloat::downcast_rne<BF16_FRACTION_SZ, BF16_EXP_SZ>(minus_max_normal_f32), minus_inf_bf16);
}

#[test]
fn ldexp_test() {
    // Test Special cases.
    assert_eq(apfloat::ldexp(apfloat::zero<8, 23>(0), s32:1), apfloat::zero<8, 23>(0));
    assert_eq(apfloat::ldexp(apfloat::zero<8, 23>(1), s32:1), apfloat::zero<8, 23>(1));
    assert_eq(apfloat::ldexp(apfloat::inf<8, 23>(0), s32:-1), apfloat::inf<8, 23>(0));
    assert_eq(apfloat::ldexp(apfloat::inf<8, 23>(1), s32:-1), apfloat::inf<8, 23>(1));
    assert_eq(apfloat::ldexp(apfloat::qnan<8, 23>(), s32:1), apfloat::qnan<8, 23>());

    // Subnormal input.
    let pos_denormal = apfloat::APFloat { sign: u1:0, bexp: u8:0, fraction: u23:99 };
    assert_eq(apfloat::ldexp(pos_denormal, s32:1), apfloat::zero<8, 23>(0));
    let neg_denormal = apfloat::APFloat { sign: u1:1, bexp: u8:0, fraction: u23:99 };
    assert_eq(apfloat::ldexp(neg_denormal, s32:1), apfloat::zero<8, 23>(1));

    // Output subnormal, flush to zero.
    assert_eq(apfloat::ldexp(pos_denormal, s32:-1), apfloat::zero<8, 23>(0));

    // Subnormal result rounds up to normal number.
    let frac = apfloat::APFloat { sign: u1:0, bexp: u8:10, fraction: u23:0x7fffff };
    let expected = apfloat::APFloat { sign: u1:0, bexp: u8:1, fraction: u23:0 };
    assert_eq(apfloat::ldexp(frac, s32:-10), expected);
    let frac = apfloat::APFloat { sign: u1:1, bexp: u8:10, fraction: u23:0x7fffff };
    let expected = apfloat::APFloat { sign: u1:1, bexp: u8:1, fraction: u23:0 };
    assert_eq(apfloat::ldexp(frac, s32:-10), expected);

    // Large positive input exponents.
    let frac = apfloat::APFloat { sign: u1:0, bexp: u8:128, fraction: u23:0x0 };
    let expected = apfloat::inf<8, 23>(0);
    assert_eq(apfloat::ldexp(frac, s32:0x7FFFFFFF - 1), expected);
    let frac = apfloat::APFloat { sign: u1:0, bexp: u8:128, fraction: u23:0x0 };
    let expected = apfloat::inf<8, 23>(0);
    assert_eq(apfloat::ldexp(frac, s32:0x7FFFFFFF), expected);
    let frac = apfloat::APFloat { sign: u1:1, bexp: u8:128, fraction: u23:0x0 };
    let expected = apfloat::inf<8, 23>(1);
    assert_eq(apfloat::ldexp(frac, s32:0x7FFFFFFF - 1), expected);
    let frac = apfloat::APFloat { sign: u1:1, bexp: u8:128, fraction: u23:0x0 };
    let expected = apfloat::inf<8, 23>(1);
    assert_eq(apfloat::ldexp(frac, s32:0x7FFFFFFF), expected);

    // Large negative input exponents.
    let frac = apfloat::APFloat { sign: u1:0, bexp: u8:126, fraction: u23:0x0 };
    let expected = apfloat::zero<8, 23>(0);
    assert_eq(apfloat::ldexp(frac, s32:0x80000000 + 0x1), expected);
    let frac = apfloat::APFloat { sign: u1:0, bexp: u8:126, fraction: u23:0x0 };
    let expected = apfloat::zero<8, 23>(0);
    assert_eq(apfloat::ldexp(frac, s32:0x80000000), expected);
    let frac = apfloat::APFloat { sign: u1:1, bexp: u8:126, fraction: u23:0x0 };
    let expected = apfloat::zero<8, 23>(1);
    assert_eq(apfloat::ldexp(frac, s32:0x80000000 + 0x1), expected);
    let frac = apfloat::APFloat { sign: u1:1, bexp: u8:126, fraction: u23:0x0 };
    let expected = apfloat::zero<8, 23>(1);
    assert_eq(apfloat::ldexp(frac, s32:0x80000000), expected);

    // Other large exponents from reported bug #462.
    let frac = apfloat::unflatten<8, 23>(u32:0xd3fefd2b);
    let expected = apfloat::inf<8, 23>(1);
    assert_eq(apfloat::ldexp(frac, s32:0x7ffffffd), expected);
    let frac = apfloat::unflatten<8, 23>(u32:0x36eba93e);
    let expected = apfloat::zero<8, 23>(0);
    assert_eq(apfloat::ldexp(frac, s32:0x80000010), expected);
    let frac = apfloat::unflatten<8, 23>(u32:0x8a87c096);
    let expected = apfloat::zero<8, 23>(1);
    assert_eq(apfloat::ldexp(frac, s32:0x80000013), expected);
    let frac = apfloat::unflatten<8, 23>(u32:0x71694e37);
    let expected = apfloat::inf<8, 23>(0);
    assert_eq(apfloat::ldexp(frac, s32:0x7fffffbe), expected);
}

#[test]
fn cast_to_fixed_test() {
    // Cast +/-0.0
    assert_eq(apfloat::cast_to_fixed<32>(apfloat::zero<8, 23>(0)), s32:0);
    assert_eq(apfloat::cast_to_fixed<32>(apfloat::zero<8, 23>(1)), s32:0);

    // Cast +/-1.0
    assert_eq(apfloat::cast_to_fixed<32>(apfloat::one<8, 23>(0)), s32:1);
    assert_eq(apfloat::cast_to_fixed<32>(apfloat::one<8, 23>(1)), s32:-1);

    // Cast +/-1.5 --> +/- 1
    let one_point_five = apfloat::APFloat<8, 23> { sign: 0, bexp: 0x7f, fraction: u1:1 ++ u22:0 };
    assert_eq(apfloat::cast_to_fixed<32>(one_point_five), s32:1);
    let n_one_point_five = apfloat::APFloat<8, 23> { sign: 1, bexp: 0x7f, fraction: u1:1 ++ u22:0 };
    assert_eq(apfloat::cast_to_fixed<32>(n_one_point_five), s32:-1);

    // Cast +/-4.0
    let four = apfloat::cast_from_fixed_using_rne<8, 23>(s32:4);
    let neg_four = apfloat::cast_from_fixed_using_rne<8, 23>(s32:-4);
    assert_eq(apfloat::cast_to_fixed<32>(four), s32:4);
    assert_eq(apfloat::cast_to_fixed<32>(neg_four), s32:-4);

    // Cast 7
    let seven = apfloat::cast_from_fixed_using_rne<8, 23>(s32:7);
    assert_eq(apfloat::cast_to_fixed<32>(seven), s32:7);

    // Cast big number (more digits left of decimal than hidden bit + fraction).
    let big_num = (u1:0 ++ std::mask_bits<23>() ++ u8:0) as s32;
    let fp_big_num = apfloat::cast_from_fixed_using_rne<8, 23>(big_num);
    assert_eq(apfloat::cast_to_fixed<32>(fp_big_num), big_num);

    // Cast large, non-overflowing numbers.
    let big_fit = apfloat::APFloat<8, 23> { sign: 0, bexp: 127 + 30, fraction: 0x7fffff };
    assert_eq(apfloat::cast_to_fixed<32>(big_fit), (u1:0 ++ u24:0xffffff ++ u7:0) as s32);
    let big_fit = apfloat::APFloat<8, 23> { sign: 1, bexp: 127 + 30, fraction: 0x7fffff };
    assert_eq(apfloat::cast_to_fixed<32>(big_fit), (0 - (u1:0 ++ u24:0xffffff ++ u7:0) as s32));

    // Cast barely overflowing postive number.
    let big_overflow = apfloat::APFloat<8, 23> { sign: 0, bexp: 127 + 31, fraction: 0x0 };
    assert_eq(apfloat::cast_to_fixed<32>(big_overflow), (u1:1 ++ u31:0) as s32);

    // This produces the largest negative int, but doesn't actually
    // overflow
    let max_negative = apfloat::APFloat<8, 23> { sign: 1, bexp: 127 + 31, fraction: 0x0 };
    assert_eq(apfloat::cast_to_fixed<32>(max_negative), (u1:1 ++ u31:0) as s32);

    // Negative overflow.
    let negative_overflow = apfloat::APFloat<8, 23> { sign: 1, bexp: 127 + 31, fraction: 0x1 };
    assert_eq(apfloat::cast_to_fixed<32>(negative_overflow), (u1:1 ++ u31:0) as s32);

    // NaN input.
    assert_eq(apfloat::cast_to_fixed<32>(apfloat::qnan<8, 23>()), (u1:1 ++ u31:0) as s32);
}

#[test]
fn test_fp_eq_2() {
    let neg_zero = apfloat::zero<8, 23>(1);
    let zero = apfloat::zero<8, 23>(0);
    let neg_one = apfloat::one<8, 23>(1);
    let one = apfloat::one<8, 23>(0);
    let two = apfloat::APFloat<8, 23> { bexp: one.bexp + 1, ..one };
    let neg_inf = apfloat::inf<8, 23>(1);
    let inf = apfloat::inf<8, 23>(0);
    let nan = apfloat::qnan<8, 23>();
    let denormal_1 = apfloat::unflatten<8, 23>(u32:1);
    let denormal_2 = apfloat::unflatten<8, 23>(u32:2);

    // Test unequal.
    assert_eq(apfloat::eq_2(one, two), u1:0);
    assert_eq(apfloat::eq_2(two, one), u1:0);

    // Test equal.
    assert_eq(apfloat::eq_2(neg_zero, zero), u1:1);
    assert_eq(apfloat::eq_2(one, one), u1:1);
    assert_eq(apfloat::eq_2(two, two), u1:1);

    // Test equal (subnormals and zero).
    assert_eq(apfloat::eq_2(zero, zero), u1:1);
    assert_eq(apfloat::eq_2(zero, neg_zero), u1:1);
    assert_eq(apfloat::eq_2(zero, denormal_1), u1:1);
    assert_eq(apfloat::eq_2(denormal_2, denormal_1), u1:1);

    // Test negatives.
    assert_eq(apfloat::eq_2(one, neg_one), u1:0);
    assert_eq(apfloat::eq_2(neg_one, one), u1:0);
    assert_eq(apfloat::eq_2(neg_one, neg_one), u1:1);

    // Special case - inf.
    assert_eq(apfloat::eq_2(inf, one), u1:0);
    assert_eq(apfloat::eq_2(neg_inf, inf), u1:0);
    assert_eq(apfloat::eq_2(inf, inf), u1:1);
    assert_eq(apfloat::eq_2(neg_inf, neg_inf), u1:1);

    // Special case - NaN (always returns false).
    assert_eq(apfloat::eq_2(one, nan), u1:0);
    assert_eq(apfloat::eq_2(neg_one, nan), u1:0);
    assert_eq(apfloat::eq_2(inf, nan), u1:0);
    assert_eq(apfloat::eq_2(nan, inf), u1:0);
    assert_eq(apfloat::eq_2(nan, nan), u1:0);
}

#[test]
fn test_fp_gt_2() {
    let zero = apfloat::zero<8, 23>(0);
    let neg_one = apfloat::one<8, 23>(1);
    let one = apfloat::one<8, 23>(0);
    let two = apfloat::APFloat<8, 23> { bexp: one.bexp + 1, ..one };
    let neg_two = apfloat::APFloat<8, 23> { bexp: neg_one.bexp + 1, ..neg_one };
    let neg_inf = apfloat::inf<8, 23>(1);
    let inf = apfloat::inf<8, 23>(0);
    let nan = apfloat::qnan<8, 23>();
    let denormal_1 = apfloat::unflatten<8, 23>(u32:1);
    let denormal_2 = apfloat::unflatten<8, 23>(u32:2);

    // Test unequal.
    assert_eq(apfloat::gt_2(one, two), u1:0);
    assert_eq(apfloat::gt_2(two, one), u1:1);

    // Test equal.
    assert_eq(apfloat::gt_2(one, one), u1:0);
    assert_eq(apfloat::gt_2(two, two), u1:0);
    assert_eq(apfloat::gt_2(denormal_1, denormal_2), u1:0);
    assert_eq(apfloat::gt_2(denormal_2, denormal_1), u1:0);
    assert_eq(apfloat::gt_2(denormal_1, zero), u1:0);

    // Test negatives.
    assert_eq(apfloat::gt_2(zero, neg_one), u1:1);
    assert_eq(apfloat::gt_2(neg_one, zero), u1:0);
    assert_eq(apfloat::gt_2(one, neg_one), u1:1);
    assert_eq(apfloat::gt_2(neg_one, one), u1:0);
    assert_eq(apfloat::gt_2(neg_one, neg_one), u1:0);
    assert_eq(apfloat::gt_2(neg_two, neg_two), u1:0);
    assert_eq(apfloat::gt_2(neg_one, neg_two), u1:1);
    assert_eq(apfloat::gt_2(neg_two, neg_one), u1:0);

    // Special case - inf.
    assert_eq(apfloat::gt_2(inf, one), u1:1);
    assert_eq(apfloat::gt_2(inf, neg_one), u1:1);
    assert_eq(apfloat::gt_2(inf, two), u1:1);
    assert_eq(apfloat::gt_2(neg_two, neg_inf), u1:1);
    assert_eq(apfloat::gt_2(inf, inf), u1:0);
    assert_eq(apfloat::gt_2(neg_inf, inf), u1:0);
    assert_eq(apfloat::gt_2(inf, neg_inf), u1:1);
    assert_eq(apfloat::gt_2(neg_inf, neg_inf), u1:0);

    // Special case - NaN (always returns false).
    assert_eq(apfloat::gt_2(one, nan), u1:0);
    assert_eq(apfloat::gt_2(nan, one), u1:0);
    assert_eq(apfloat::gt_2(neg_one, nan), u1:0);
    assert_eq(apfloat::gt_2(nan, neg_one), u1:0);
    assert_eq(apfloat::gt_2(inf, nan), u1:0);
    assert_eq(apfloat::gt_2(nan, inf), u1:0);
    assert_eq(apfloat::gt_2(nan, nan), u1:0);
}

#[test]
fn test_fp_gte_2() {
    let zero = apfloat::zero<8, 23>(0);
    let neg_one = apfloat::one<8, 23>(1);
    let one = apfloat::one<8, 23>(0);
    let two = apfloat::APFloat<8, 23> { bexp: one.bexp + 1, ..one };
    let neg_two = apfloat::APFloat<8, 23> { bexp: neg_one.bexp + 1, ..neg_one };
    let neg_inf = apfloat::inf<8, 23>(1);
    let inf = apfloat::inf<8, 23>(0);
    let nan = apfloat::qnan<8, 23>();
    let denormal_1 = apfloat::unflatten<8, 23>(u32:1);
    let denormal_2 = apfloat::unflatten<8, 23>(u32:2);

    // Test unequal.
    assert_eq(apfloat::gte_2(one, two), u1:0);
    assert_eq(apfloat::gte_2(two, one), u1:1);

    // Test equal.
    assert_eq(apfloat::gte_2(one, one), u1:1);
    assert_eq(apfloat::gte_2(two, two), u1:1);
    assert_eq(apfloat::gte_2(denormal_1, denormal_2), u1:1);
    assert_eq(apfloat::gte_2(denormal_2, denormal_1), u1:1);
    assert_eq(apfloat::gte_2(denormal_1, zero), u1:1);

    // Test negatives.
    assert_eq(apfloat::gte_2(zero, neg_one), u1:1);
    assert_eq(apfloat::gte_2(neg_one, zero), u1:0);
    assert_eq(apfloat::gte_2(one, neg_one), u1:1);
    assert_eq(apfloat::gte_2(neg_one, one), u1:0);
    assert_eq(apfloat::gte_2(neg_one, neg_one), u1:1);
    assert_eq(apfloat::gte_2(neg_two, neg_two), u1:1);
    assert_eq(apfloat::gte_2(neg_one, neg_two), u1:1);
    assert_eq(apfloat::gte_2(neg_two, neg_one), u1:0);

    // Special case - inf.
    assert_eq(apfloat::gte_2(inf, one), u1:1);
    assert_eq(apfloat::gte_2(inf, neg_one), u1:1);
    assert_eq(apfloat::gte_2(inf, two), u1:1);
    assert_eq(apfloat::gte_2(neg_two, neg_inf), u1:1);
    assert_eq(apfloat::gte_2(inf, inf), u1:1);
    assert_eq(apfloat::gte_2(neg_inf, inf), u1:0);
    assert_eq(apfloat::gte_2(inf, neg_inf), u1:1);
    assert_eq(apfloat::gte_2(neg_inf, neg_inf), u1:1);

    // Special case - NaN (always returns false).
    assert_eq(apfloat::gte_2(one, nan), u1:0);
    assert_eq(apfloat::gte_2(nan, one), u1:0);
    assert_eq(apfloat::gte_2(neg_one, nan), u1:0);
    assert_eq(apfloat::gte_2(nan, neg_one), u1:0);
    assert_eq(apfloat::gte_2(inf, nan), u1:0);
    assert_eq(apfloat::gte_2(nan, inf), u1:0);
    assert_eq(apfloat::gte_2(nan, nan), u1:0);
}

#[test]
fn test_fp_lte_2() {
    let zero = apfloat::zero<8, 23>(0);
    let neg_one = apfloat::one<8, 23>(1);
    let one = apfloat::one<8, 23>(0);
    let two = apfloat::APFloat<8, 23> { bexp: one.bexp + 1, ..one };
    let neg_two = apfloat::APFloat<8, 23> { bexp: neg_one.bexp + 1, ..neg_one };
    let neg_inf = apfloat::inf<8, 23>(1);
    let inf = apfloat::inf<8, 23>(0);
    let nan = apfloat::qnan<8, 23>();
    let denormal_1 = apfloat::unflatten<8, 23>(u32:1);
    let denormal_2 = apfloat::unflatten<8, 23>(u32:2);

    // Test unequal.
    assert_eq(apfloat::lte_2(one, two), u1:1);
    assert_eq(apfloat::lte_2(two, one), u1:0);

    // Test equal.
    assert_eq(apfloat::lte_2(one, one), u1:1);
    assert_eq(apfloat::lte_2(two, two), u1:1);
    assert_eq(apfloat::lte_2(denormal_1, denormal_2), u1:1);
    assert_eq(apfloat::lte_2(denormal_2, denormal_1), u1:1);
    assert_eq(apfloat::lte_2(denormal_1, zero), u1:1);

    // Test negatives.
    assert_eq(apfloat::lte_2(zero, neg_one), u1:0);
    assert_eq(apfloat::lte_2(neg_one, zero), u1:1);
    assert_eq(apfloat::lte_2(one, neg_one), u1:0);
    assert_eq(apfloat::lte_2(neg_one, one), u1:1);
    assert_eq(apfloat::lte_2(neg_one, neg_one), u1:1);
    assert_eq(apfloat::lte_2(neg_two, neg_two), u1:1);
    assert_eq(apfloat::lte_2(neg_one, neg_two), u1:0);
    assert_eq(apfloat::lte_2(neg_two, neg_one), u1:1);

    // Special case - inf.
    assert_eq(apfloat::lte_2(inf, one), u1:0);
    assert_eq(apfloat::lte_2(inf, neg_one), u1:0);
    assert_eq(apfloat::lte_2(inf, two), u1:0);
    assert_eq(apfloat::lte_2(neg_two, neg_inf), u1:0);
    assert_eq(apfloat::lte_2(inf, inf), u1:1);
    assert_eq(apfloat::lte_2(neg_inf, inf), u1:1);
    assert_eq(apfloat::lte_2(inf, neg_inf), u1:0);
    assert_eq(apfloat::lte_2(neg_inf, neg_inf), u1:1);

    // Special case - NaN (always returns false).
    assert_eq(apfloat::lte_2(one, nan), u1:0);
    assert_eq(apfloat::lte_2(nan, one), u1:0);
    assert_eq(apfloat::lte_2(neg_one, nan), u1:0);
    assert_eq(apfloat::lte_2(nan, neg_one), u1:0);
    assert_eq(apfloat::lte_2(inf, nan), u1:0);
    assert_eq(apfloat::lte_2(nan, inf), u1:0);
    assert_eq(apfloat::lte_2(nan, nan), u1:0);
}

#[test]
fn test_fp_lt_2() {
    let zero = apfloat::zero<8, 23>(0);
    let neg_one = apfloat::one<8, 23>(1);
    let one = apfloat::one<8, 23>(0);
    let two = apfloat::APFloat<8, 23> { bexp: one.bexp + 1, ..one };
    let neg_two = apfloat::APFloat<8, 23> { bexp: neg_one.bexp + 1, ..neg_one };
    let neg_inf = apfloat::inf<8, 23>(1);
    let inf = apfloat::inf<8, 23>(0);
    let nan = apfloat::qnan<8, 23>();
    let denormal_1 = apfloat::unflatten<8, 23>(u32:1);
    let denormal_2 = apfloat::unflatten<8, 23>(u32:2);

    // Test unequal.
    assert_eq(apfloat::lt_2(one, two), u1:1);
    assert_eq(apfloat::lt_2(two, one), u1:0);

    // Test equal.
    assert_eq(apfloat::lt_2(one, one), u1:0);
    assert_eq(apfloat::lt_2(two, two), u1:0);
    assert_eq(apfloat::lt_2(denormal_1, denormal_2), u1:0);
    assert_eq(apfloat::lt_2(denormal_2, denormal_1), u1:0);
    assert_eq(apfloat::lt_2(denormal_1, zero), u1:0);

    // Test negatives.
    assert_eq(apfloat::lt_2(zero, neg_one), u1:0);
    assert_eq(apfloat::lt_2(neg_one, zero), u1:1);
    assert_eq(apfloat::lt_2(one, neg_one), u1:0);
    assert_eq(apfloat::lt_2(neg_one, one), u1:1);
    assert_eq(apfloat::lt_2(neg_one, neg_one), u1:0);
    assert_eq(apfloat::lt_2(neg_two, neg_two), u1:0);
    assert_eq(apfloat::lt_2(neg_one, neg_two), u1:0);
    assert_eq(apfloat::lt_2(neg_two, neg_one), u1:1);

    // Special case - inf.
    assert_eq(apfloat::lt_2(inf, one), u1:0);
    assert_eq(apfloat::lt_2(inf, neg_one), u1:0);
    assert_eq(apfloat::lt_2(inf, two), u1:0);
    assert_eq(apfloat::lt_2(neg_two, neg_inf), u1:0);
    assert_eq(apfloat::lt_2(inf, inf), u1:0);
    assert_eq(apfloat::lt_2(neg_inf, inf), u1:1);
    assert_eq(apfloat::lt_2(inf, neg_inf), u1:0);
    assert_eq(apfloat::lt_2(neg_inf, neg_inf), u1:0);

    // Special case - NaN (always returns false).
    assert_eq(apfloat::lt_2(one, nan), u1:0);
    assert_eq(apfloat::lt_2(nan, one), u1:0);
    assert_eq(apfloat::lt_2(neg_one, nan), u1:0);
    assert_eq(apfloat::lt_2(nan, neg_one), u1:0);
    assert_eq(apfloat::lt_2(inf, nan), u1:0);
    assert_eq(apfloat::lt_2(nan, inf), u1:0);
    assert_eq(apfloat::lt_2(nan, nan), u1:0);
}

#[test]
fn to_int_test() {
    // +0
    let expected = s32:0;
    let actual =
        apfloat::to_int<8, 23, 32>(apfloat::APFloat<8, 23> { sign: 0, bexp: 0x0, fraction: 0x0 });
    assert_eq(expected, actual);

    // smallest positive subnormal
    let expected = s32:0;
    let actual =
        apfloat::to_int<8, 23, 32>(apfloat::APFloat<8, 23> { sign: 0, bexp: 0x0, fraction: 0x1 });
    assert_eq(expected, actual);

    // largest subnormal
    let expected = s32:0;
    let actual = apfloat::to_int<8, 23, 32>(
        apfloat::APFloat<8, 23> { sign: 0, bexp: 0x0, fraction: 0x7fffff });
    assert_eq(expected, actual);

    // smallest positive normal
    let expected = s32:0;
    let actual =
        apfloat::to_int<8, 23, 32>(apfloat::APFloat<8, 23> { sign: 0, bexp: 0x1, fraction: 0x0 });
    assert_eq(expected, actual);

    // largest < 1
    let expected = s32:0;
    let actual = apfloat::to_int<8, 23, 32>(
        apfloat::APFloat<8, 23> { sign: 0, bexp: 0x7e, fraction: 0x7fffff });
    assert_eq(expected, actual);

    // 1
    let expected = s32:1;
    let actual =
        apfloat::to_int<8, 23, 32>(apfloat::APFloat<8, 23> { sign: 0, bexp: 0x7f, fraction: 0x0 });
    assert_eq(expected, actual);

    let expected = s32:1;
    let actual = apfloat::to_int<8, 23, 32>(
        apfloat::APFloat<8, 23> { sign: 0, bexp: 0x7f, fraction: 0xa5a5 });
    assert_eq(expected, actual);

    // 2
    let expected = s32:2;
    let actual =
        apfloat::to_int<8, 23, 32>(apfloat::APFloat<8, 23> { sign: 0, bexp: 0x80, fraction: 0x0 });
    assert_eq(expected, actual);

    let expected = s32:2;
    let actual = apfloat::to_int<8, 23, 32>(
        apfloat::APFloat<8, 23> { sign: 0, bexp: 0x80, fraction: 0xa5a5 });
    assert_eq(expected, actual);

    let expected = s32:0xa5a5;
    let actual = apfloat::to_int<8, 23, 32>(
        apfloat::APFloat<8, 23> { sign: 0, bexp: 0x8e, fraction: 0x25a500 });
    assert_eq(expected, actual);

    let expected = s32:23;
    let actual = apfloat::to_int<8, 23, 32>(
        apfloat::APFloat<8, 23> { sign: 0, bexp: 0x83, fraction: 0x380000 });
    assert_eq(expected, actual);

    // unbiased exp == EXP_SZ = 23
    let expected = s32:0xffffff;
    let actual = apfloat::to_int<8, 23, 32>(
        apfloat::APFloat<8, 23> { sign: 0, bexp: 0x96, fraction: 0x7fffff });
    assert_eq(expected, actual);

    let expected = s16:0xa5a;
    let actual = apfloat::to_int<8, 23, 16>(
        apfloat::APFloat<8, 23> { sign: 0, bexp: 0x8a, fraction: 0x25a5a5 });
    assert_eq(expected, actual);

    let expected = s16:0xa5;
    let actual = apfloat::to_int<8, 23, 16>(
        apfloat::APFloat<8, 23> { sign: 0, bexp: 0x86, fraction: 0x25a5a5 });
    assert_eq(expected, actual);

    let expected = s16:0x14b;
    let actual = apfloat::to_int<8, 23, 16>(
        apfloat::APFloat<8, 23> { sign: 0, bexp: 0x87, fraction: 0x25a5a5 });
    assert_eq(expected, actual);

    let expected = s16:0x296;
    let actual = apfloat::to_int<8, 23, 16>(
        apfloat::APFloat<8, 23> { sign: 0, bexp: 0x88, fraction: 0x25a5a5 });
    assert_eq(expected, actual);

    let expected = s16::MAX;
    let actual = apfloat::to_int<8, 23, 16>(
        apfloat::APFloat<8, 23> { sign: 0, bexp: 0x8f, fraction: 0x25a5a5 });
    assert_eq(expected, actual);

    let expected = s24:0x14b4b;
    let actual = apfloat::to_int<8, 23, 24>(
        apfloat::APFloat<8, 23> { sign: 0, bexp: 0x8f, fraction: 0x25a5a5 });
    assert_eq(expected, actual);

    let expected = s32:0x14b4b;
    let actual = apfloat::to_int<8, 23, 32>(
        apfloat::APFloat<8, 23> { sign: 0, bexp: 0x8f, fraction: 0x25a5a5 });
    assert_eq(expected, actual);

    let expected = s16:0xa;
    let actual = apfloat::to_int<8, 23, 16>(
        apfloat::APFloat<8, 23> { sign: 0, bexp: 0x82, fraction: 0x25a5a5 });
    assert_eq(expected, actual);

    let expected = s16:0x5;
    let actual = apfloat::to_int<8, 23, 16>(
        apfloat::APFloat<8, 23> { sign: 0, bexp: 0x81, fraction: 0x25a5a5 });
    assert_eq(expected, actual);

    // s32::MAX + 1.0
    let expected = s32::MAX;
    let actual =
        apfloat::to_int<8, 23, 32>(apfloat::APFloat<8, 23> { sign: 0, bexp: 0x9e, fraction: 0x0 });
    assert_eq(expected, actual);

    // s32::MAX + 1.0, fits in s64 so no clamping
    let expected = (s32::MAX as s64) + 1;
    let actual =
        apfloat::to_int<8, 23, 64>(apfloat::APFloat<8, 23> { sign: 0, bexp: 0x9e, fraction: 0x0 });
    assert_eq(expected, actual);

    // large s64 sized integer as exact float32
    let expected = s64:0x1234_5000_0000_0000;
    let actual = apfloat::to_int<8, 23, 64>(
        apfloat::APFloat<8, 23> { sign: 0, bexp: 0xbb, fraction: 0x11a280 });
    assert_eq(expected, actual);

    // largest normal
    let expected = s32::MAX;
    let actual = apfloat::to_int<8, 23, 32>(
        apfloat::APFloat<8, 23> { sign: 0, bexp: 0xfe, fraction: 0x7fffff });
    assert_eq(expected, actual);

    // inf
    let expected = s32::MAX;
    let actual =
        apfloat::to_int<8, 23, 32>(apfloat::APFloat<8, 23> { sign: 0, bexp: 0xff, fraction: 0x0 });
    assert_eq(expected, actual);

    // -0
    let expected = s32:0;
    let actual =
        apfloat::to_int<8, 23, 32>(apfloat::APFloat<8, 23> { sign: 1, bexp: 0x0, fraction: 0x0 });
    assert_eq(expected, actual);

    // negative subnormal
    let expected = s32:0;
    let actual = apfloat::to_int<8, 23, 32>(
        apfloat::APFloat<8, 23> { sign: 1, bexp: 0x0, fraction: 0x7fffff });
    assert_eq(expected, actual);

    // -2
    let expected = s32:-2;
    let actual =
        apfloat::to_int<8, 23, 32>(apfloat::APFloat<8, 23> { sign: 1, bexp: 0x80, fraction: 0x0 });
    assert_eq(expected, actual);

    let expected = s32:-2;
    let actual =
        apfloat::to_int<8, 23, 32>(apfloat::APFloat<8, 23> { sign: 1, bexp: 0x80, fraction: 0x1 });
    assert_eq(expected, actual);

    // -inf
    const S32_MIN = s32:1 << 31;
    let expected = S32_MIN;
    let actual =
        apfloat::to_int<8, 23, 32>(apfloat::APFloat<8, 23> { sign: 1, bexp: 0xff, fraction: 0x0 });
    assert_eq(expected, actual);

    // nan
    let expected = s32:0;
    let actual = apfloat::to_int<8, 23, 32>(apfloat::qnan<8, 23>());
    assert_eq(expected, actual);
}

#[test]
fn to_uint_test() {
    // +0
    assert_eq(
        u32:0, apfloat::to_uint<32>(apfloat::APFloat<8, 23> { sign: 0, bexp: 0x0, fraction: 0x0 }));

    // smallest positive subnormal
    assert_eq(
        u32:0, apfloat::to_uint<32>(apfloat::APFloat<8, 23> { sign: 0, bexp: 0x0, fraction: 0x1 }));

    // largest subnormal
    assert_eq(
        u32:0,
        apfloat::to_uint<32>(apfloat::APFloat<8, 23> { sign: 0, bexp: 0x0, fraction: 0x7fffff }));

    // smallest positive normal
    assert_eq(
        u32:0, apfloat::to_uint<32>(apfloat::APFloat<8, 23> { sign: 0, bexp: 0x1, fraction: 0x0 }));

    // largest < 1
    assert_eq(
        u32:0,
        apfloat::to_uint<32>(apfloat::APFloat<8, 23> { sign: 0, bexp: 0x7e, fraction: 0x7fffff }));

    // 1
    assert_eq(
        u32:1, apfloat::to_uint<32>(apfloat::APFloat<8, 23> { sign: 0, bexp: 0x7f, fraction: 0x0 }));

    // smallest > 1
    assert_eq(
        u32:1, apfloat::to_uint<32>(apfloat::APFloat<8, 23> { sign: 0, bexp: 0x7f, fraction: 0x1 }));

    // pi
    assert_eq(
        u32:3,
        apfloat::to_uint<32>(apfloat::APFloat<8, 23> { sign: 0, bexp: 0x80, fraction: 0x490fdb }));

    // largest normal
    assert_eq(
        u32::MAX,
        apfloat::to_uint<32>(apfloat::APFloat<8, 23> { sign: 0, bexp: 0xfe, fraction: 0x7fffff }));

    // inf
    assert_eq(u32::MAX, apfloat::to_uint<32>(apfloat::inf<8, 23>(0)));

    // -0
    assert_eq(
        u32:0, apfloat::to_uint<32>(apfloat::APFloat<8, 23> { sign: 0, bexp: 0x0, fraction: 0x0 }));

    // -2
    assert_eq(
        u32:0, apfloat::to_uint<32>(apfloat::APFloat<8, 23> { sign: 1, bexp: 0x80, fraction: 0x0 }));

    // -2
    assert_eq(
        u32:0, apfloat::to_uint<32>(apfloat::APFloat<8, 23> { sign: 1, bexp: 0x80, fraction: 0x0 }));

    // largest < -2
    assert_eq(
        u32:0, apfloat::to_uint<32>(apfloat::APFloat<8, 23> { sign: 1, bexp: 0x80, fraction: 0x1 }));

    // -inf
    assert_eq(u32:0, apfloat::to_uint<32>(apfloat::inf<8, 23>(1)));

    // nan
    assert_eq(u32:0, apfloat::to_uint<32>(apfloat::qnan<8, 23>()));

    // s32::MAX + 1.0, should fit as uint32.
    assert_eq(
        u32:0x80000000,
        apfloat::to_uint<32>(apfloat::APFloat<8, 23> { sign: 0, bexp: 0x9e, fraction: 0x0 }));

    // basic spot check with different format (bfloat16)
    // smallest > 1
    assert_eq(
        u16:1, apfloat::to_uint<16>(apfloat::APFloat<8, 7> { sign: 0, bexp: 0x7f, fraction: 0x1 }));

    // largest normal
    assert_eq(
        u16::MAX,
        apfloat::to_uint<16>(apfloat::APFloat<8, 7> { sign: 0, bexp: 0x0fe, fraction: 0x7f }));

    // -2
    assert_eq(
        u4:0, apfloat::to_uint<4>(apfloat::APFloat<8, 7> { sign: 1, bexp: 0x80, fraction: 0 }));
}

#[quickcheck(exhaustive)]
fn add_clz_lza_equivalence(a: apfloat::APFloat<3, 3>, b: apfloat::APFloat<3, 3>) -> bool {
    apfloat::add<3, 3, true>(a, b) == apfloat::add<3, 3, false>(a, b)
}

#[quickcheck]
fn add_clz_lza_equivalence_f32(a: apfloat::APFloat<8, 23>, b: apfloat::APFloat<8, 23>) -> bool {
    apfloat::add<8, 23, true>(a, b) == apfloat::add<8, 23, false>(a, b)
}

// add is thoroughly tested elsewhere so a few simple tests is sufficient.
#[test]
fn test_sub() {
    let one = apfloat::one<8, 23>(0);
    let two = apfloat::add<8, 23>(one, one);
    let neg_two = apfloat::APFloat<8, 23> { sign: 1, ..two };
    let three = apfloat::add<8, 23>(one, two);
    let four = apfloat::add<8, 23>(two, two);

    assert_eq(apfloat::sub(four, one), three);
    assert_eq(apfloat::sub(four, two), two);
    assert_eq(apfloat::sub(four, three), one);
    assert_eq(apfloat::sub(three, two), one);
    assert_eq(apfloat::sub(two, four), neg_two);
}

#[test]
fn smoke() {
    type F32 = apfloat::APFloat<8, 23>;
    let zero = F32 { sign: 0, bexp: 0, fraction: 0 };
    let one_point_one = F32 { sign: 0, bexp: 127, fraction: 0xccccd };
    let twenty_seven_point_one = F32 { sign: 0, bexp: 131, fraction: 0x58cccd };
    let a = twenty_seven_point_one;
    let b = one_point_one;
    let c = zero;
    let expected = F32 { sign: 0, bexp: 0x83, fraction: 0x6e7ae2 };
    let actual = apfloat::fma<8, 23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn one_x_one_plus_one_f32() {
    type F32 = apfloat::APFloat<8, 23>;
    let one_point_zero = F32 { sign: 0, bexp: 127, fraction: 0 };
    let a = one_point_zero;
    let b = one_point_zero;
    let c = one_point_zero;
    let expected = F32 { sign: 0, bexp: 128, fraction: 0 };
    let actual = apfloat::fma<8, 23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn one_x_one_plus_one_f64() {
    type F64 = apfloat::APFloat<11, 52>;
    let one_point_zero = F64 { sign: 0, bexp: 1023, fraction: 0 };
    let a = one_point_zero;
    let b = one_point_zero;
    let c = one_point_zero;
    let expected = F64 { sign: 0, bexp: 1024, fraction: 0 };
    let actual = apfloat::fma<11, 52>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn one_x_one_plus_one_bf16() {
    type BF16 = apfloat::APFloat<8, 7>;
    let one_point_zero = BF16 { sign: 0, bexp: 127, fraction: 0 };
    let a = one_point_zero;
    let b = one_point_zero;
    let c = one_point_zero;
    let expected = BF16 { sign: 0, bexp: 128, fraction: 0 };
    let actual = apfloat::fma<8, 7>(a, b, c);
    assert_eq(expected, actual)
}

// Too complicated to be fully descriptive:
// (3250761 x -0.00542...) + 456.31...
// This set of tests will use the same inputs (or as close as is possible).
#[test]
fn manual_case_a_f32() {
    type F32 = apfloat::APFloat<8, 23>;
    let a = F32 { sign: 0, bexp: 0x97, fraction: 0x565d43 };
    let b = F32 { sign: 1, bexp: 0x77, fraction: 0x319a49 };
    let c = F32 { sign: 0, bexp: 0x87, fraction: 0x642891 };
    let expected = F32 { sign: 1, bexp: 0x90, fraction: 0x144598 };
    let actual = apfloat::fma<8, 23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn manual_case_a_f64() {
    type F64 = apfloat::APFloat<11, 52>;
    let a = F64 { sign: 0, bexp: 0x417, fraction: 0x565d43 };
    let b = F64 { sign: 1, bexp: 0x3f7, fraction: 0x319a49 };
    let c = F64 { sign: 0, bexp: 0x407, fraction: 0x642891 };
    let expected = F64 { sign: 1, bexp: 0x40e, fraction: 0xfe000010f26c7 };
    let actual = apfloat::fma<11, 52>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn manual_case_a_bf16() {
    type BF16 = apfloat::APFloat<8, 7>;
    let a = BF16 { sign: 0, bexp: 0x97, fraction: 0x2b };
    let b = BF16 { sign: 1, bexp: 0x77, fraction: 0x18 };
    let c = BF16 { sign: 0, bexp: 0x87, fraction: 0x32 };
    let expected = BF16 { sign: 1, bexp: 0x8f, fraction: 0x4a };
    let actual = apfloat::fma<8, 7>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn twenty_seven_point_one_x_twenty_seven_point_one_plus_zero() {
    type F32 = apfloat::APFloat<8, 23>;
    let zero = F32 { sign: 0, bexp: 0, fraction: 0 };
    let twenty_seven_point_one = F32 { sign: 0, bexp: 131, fraction: 0x58cccd };
    let a = twenty_seven_point_one;
    let b = twenty_seven_point_one;
    let c = zero;
    let expected = F32 { sign: 0, bexp: 0x88, fraction: 0x379a3e };
    let actual = apfloat::fma<8, 23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn twenty_seven_point_one_x_twenty_seven_point_one_plus_one() {
    type F32 = apfloat::APFloat<8, 23>;
    let one_point_zero = F32 { sign: 0, bexp: 127, fraction: 0 };
    let twenty_seven_point_one = F32 { sign: 0, bexp: 131, fraction: 0x58cccd };
    let a = twenty_seven_point_one;
    let b = twenty_seven_point_one;
    let c = one_point_zero;
    let expected = F32 { sign: 0, bexp: 0x88, fraction: 0x37da3e };
    let actual = apfloat::fma<8, 23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn twenty_seven_point_one_x_twenty_seven_point_one_plus_one_point_one() {
    type F32 = apfloat::APFloat<8, 23>;
    let one_point_one = F32 { sign: 0, bexp: 127, fraction: 0xccccd };
    let twenty_seven_point_one = F32 { sign: 0, bexp: 131, fraction: 0x58cccd };
    let a = twenty_seven_point_one;
    let b = twenty_seven_point_one;
    let c = one_point_one;
    let expected = F32 { sign: 0, bexp: 0x88, fraction: 0x37e0a4 };
    let actual = apfloat::fma<8, 23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_a() {
    type F32 = apfloat::APFloat<8, 23>;
    let a = F32 { sign: 0x1, bexp: 0x50, fraction: 0x1a8ddc };
    let b = F32 { sign: 0x1, bexp: 0xcb, fraction: 0xee7ac };
    let c = F32 { sign: 0x1, bexp: 0xb7, fraction: 0x609f18 };
    let expected = F32 { sign: 1, bexp: 0xb7, fraction: 0x609f18 };
    let actual = apfloat::fma<8, 23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_b() {
    type F32 = apfloat::APFloat<8, 23>;
    let a = F32 { sign: 0x0, bexp: 0x23, fraction: 0x4d3a41 };
    let b = F32 { sign: 0x0, bexp: 0x30, fraction: 0x35a901 };
    let c = F32 { sign: 0x0, bexp: 0x96, fraction: 0x627c62 };
    let expected = F32 { sign: 0, bexp: 0x96, fraction: 0x627c62 };
    let actual = apfloat::fma<8, 23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_c() {
    type F32 = apfloat::APFloat<8, 23>;
    let a = F32 { sign: 0x1, bexp: 0x71, fraction: 0x2f0932 };
    let b = F32 { sign: 0x0, bexp: 0xe5, fraction: 0x416b76 };
    let c = F32 { sign: 0x0, bexp: 0xcb, fraction: 0x5fd32a };
    let expected = F32 { sign: 1, bexp: 0xd8, fraction: 0x4386a };
    let actual = apfloat::fma<8, 23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_d() {
    type F32 = apfloat::APFloat<8, 23>;
    let a = F32 { sign: 0x0, bexp: 0xac, fraction: 0x1d0d22 };
    let b = F32 { sign: 0x0, bexp: 0xdb, fraction: 0x2fe688 };
    let c = F32 { sign: 0x0, bexp: 0xa9, fraction: 0x2be1d2 };
    let expected = F32 { sign: 0, bexp: 0xff, fraction: 0x0 };
    let actual = apfloat::fma<8, 23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_e() {
    type F32 = apfloat::APFloat<8, 23>;
    let a = F32 { sign: 0x0, bexp: 0x7b, fraction: 0x25e79f };
    let b = F32 { sign: 0x1, bexp: 0xff, fraction: 0x207370 };
    let c = F32 { sign: 0x1, bexp: 0x39, fraction: 0x6bb348 };
    let expected = apfloat::qnan<8, 23>();
    let actual = apfloat::fma<8, 23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_f() {
    type F32 = apfloat::APFloat<8, 23>;
    let a = F32 { sign: 0x1, bexp: 0xe0, fraction: 0x3cdaa8 };
    let b = F32 { sign: 0x1, bexp: 0x96, fraction: 0x52549c };
    let c = F32 { sign: 0x0, bexp: 0x1c, fraction: 0x21e0fd };
    let expected = F32 { sign: 0, bexp: 0xf8, fraction: 0x1b29c9 };
    let actual = apfloat::fma<8, 23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_g() {
    type F32 = apfloat::APFloat<8, 23>;
    let a = F32 { sign: 0x1, bexp: 0xc4, fraction: 0x73b59a };
    let b = F32 { sign: 0x0, bexp: 0xa6, fraction: 0x1631c0 };
    let c = F32 { sign: 0x0, bexp: 0x29, fraction: 0x5b3d33 };
    let expected = F32 { sign: 1, bexp: 0xec, fraction: 0xefbc5 };
    let actual = apfloat::fma<8, 23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_h() {
    type F32 = apfloat::APFloat<8, 23>;
    let a = F32 { sign: 0x1, bexp: 0x9b, fraction: 0x3f50d4 };
    let b = F32 { sign: 0x0, bexp: 0x7b, fraction: 0x4beeb5 };
    let c = F32 { sign: 0x1, bexp: 0x37, fraction: 0x6ad17c };
    let expected = F32 { sign: 1, bexp: 0x98, fraction: 0x18677d };
    let actual = apfloat::fma<8, 23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_i() {
    type F32 = apfloat::APFloat<8, 23>;
    let a = F32 { sign: 0x0, bexp: 0x66, fraction: 0x36e592 };
    let b = F32 { sign: 0x0, bexp: 0xc8, fraction: 0x2b5bf1 };
    let c = F32 { sign: 0x0, bexp: 0x52, fraction: 0x12900b };
    let expected = F32 { sign: 0, bexp: 0xaf, fraction: 0x74da11 };
    let actual = apfloat::fma<8, 23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_j() {
    type F32 = apfloat::APFloat<8, 23>;
    let a = F32 { sign: 0x1, bexp: 0x88, fraction: 0x0f0e03 };
    let b = F32 { sign: 0x1, bexp: 0xb9, fraction: 0x36006d };
    let c = F32 { sign: 0x1, bexp: 0xaa, fraction: 0x358b6b };
    let expected = F32 { sign: 0, bexp: 0xc2, fraction: 0x4b6865 };
    let actual = apfloat::fma<8, 23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_k() {
    type F32 = apfloat::APFloat<8, 23>;
    let a = F32 { sign: 0x1, bexp: 0x29, fraction: 0x2fd76d };
    let b = F32 { sign: 0x1, bexp: 0xce, fraction: 0x63eded };
    let c = F32 { sign: 0x0, bexp: 0xfd, fraction: 0x21adee };
    let expected = F32 { sign: 0, bexp: 0xfd, fraction: 0x21adee };
    let actual = apfloat::fma<8, 23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_l() {
    type F32 = apfloat::APFloat<8, 23>;
    let a = F32 { sign: 0x0, bexp: 0x6a, fraction: 0x09c1b9 };
    let b = F32 { sign: 0x1, bexp: 0x7c, fraction: 0x666a52 };
    let c = F32 { sign: 0x1, bexp: 0x80, fraction: 0x626bcf };
    let expected = F32 { sign: 1, bexp: 0x80, fraction: 0x626bcf };
    let actual = apfloat::fma<8, 23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_m() {
    type F32 = apfloat::APFloat<8, 23>;
    let a = F32 { sign: 0x1, bexp: 0x70, fraction: 0x41e2db };
    let b = F32 { sign: 0x1, bexp: 0xd1, fraction: 0x013c17 };
    let c = F32 { sign: 0x0, bexp: 0xb9, fraction: 0x30313f };
    let expected = F32 { sign: 0, bexp: 0xc2, fraction: 0x4419bf };
    let actual = apfloat::fma<8, 23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_n() {
    type F32 = apfloat::APFloat<8, 23>;
    let a = F32 { sign: 0x1, bexp: 0x33, fraction: 0x537374 };
    let b = F32 { sign: 0x0, bexp: 0x40, fraction: 0x78fa62 };
    let c = F32 { sign: 0x1, bexp: 0x09, fraction: 0x7cfb29 };
    let expected = F32 { sign: 1, bexp: 0x09, fraction: 0x7cfb36 };
    let actual = apfloat::fma<8, 23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_o() {
    type F32 = apfloat::APFloat<8, 23>;
    let a = F32 { sign: 0x0, bexp: 0x94, fraction: 0x1aeb90 };
    let b = F32 { sign: 0x1, bexp: 0x88, fraction: 0x1ab376 };
    let c = F32 { sign: 0x1, bexp: 0x9d, fraction: 0x15dd1e };
    let expected = F32 { sign: 1, bexp: 0x9e, fraction: 0x288cde };
    let actual = apfloat::fma<8, 23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_p() {
    type F32 = apfloat::APFloat<8, 23>;
    let a = F32 { sign: 0x0, bexp: 0x88, fraction: 0x1ebb00 };
    let b = F32 { sign: 0x1, bexp: 0xf6, fraction: 0x0950b6 };
    let c = F32 { sign: 0x0, bexp: 0xfd, fraction: 0x6c314b };
    let expected = F32 { sign: 1, bexp: 0xfe, fraction: 0x5e77d4 };
    let actual = apfloat::fma<8, 23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_q() {
    type F32 = apfloat::APFloat<8, 23>;
    let a = F32 { sign: 0x0, bexp: 0xda, fraction: 0x5b328f };
    let b = F32 { sign: 0x1, bexp: 0x74, fraction: 0x157da3 };
    let c = F32 { sign: 0x0, bexp: 0x1b, fraction: 0x6a3f25 };
    let expected = F32 { sign: 1, bexp: 0xd0, fraction: 0x000000 };
    let actual = apfloat::fma<8, 23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_r() {
    type F32 = apfloat::APFloat<8, 23>;
    let a = F32 { sign: 0x1, bexp: 0x34, fraction: 0x4da000 };
    let b = F32 { sign: 0x0, bexp: 0xf4, fraction: 0x4bc400 };
    let c = F32 { sign: 0x1, bexp: 0x33, fraction: 0x54476d };
    let expected = F32 { sign: 1, bexp: 0xaa, fraction: 0x23ab4f };
    let actual = apfloat::fma<8, 23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_s() {
    type F32 = apfloat::APFloat<8, 23>;
    let a = F32 { sign: 0x1, bexp: 0x27, fraction: 0x732d83 };
    let b = F32 { sign: 0x1, bexp: 0xbb, fraction: 0x4b2dcd };
    let c = F32 { sign: 0x0, bexp: 0x3a, fraction: 0x65e4bd };
    let expected = F32 { sign: 0, bexp: 0x64, fraction: 0x410099 };
    let actual = apfloat::fma<8, 23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_t() {
    type F32 = apfloat::APFloat<8, 23>;
    let a = F32 { sign: 0x0, bexp: 0x17, fraction: 0x070770 };
    let b = F32 { sign: 0x1, bexp: 0x86, fraction: 0x623b39 };
    let c = F32 { sign: 0x0, bexp: 0x1e, fraction: 0x6ea761 };
    let expected = F32 { sign: 1, bexp: 0x0c, fraction: 0x693bc0 };
    let actual = apfloat::fma<8, 23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_u() {
    type F32 = apfloat::APFloat<8, 23>;
    let a = F32 { sign: 0x0, bexp: 0xb1, fraction: 0x0c8800 };
    let b = F32 { sign: 0x1, bexp: 0xc6, fraction: 0x2b3800 };
    let c = F32 { sign: 0x0, bexp: 0x22, fraction: 0x00c677 };
    let expected = F32 { sign: 1, bexp: 0xf8, fraction: 0x3bfb2b };
    let actual = apfloat::fma<8, 23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_v() {
    type F32 = apfloat::APFloat<8, 23>;
    let a = F32 { sign: 0x0, bexp: 0x90, fraction: 0x04a800 };
    let b = F32 { sign: 0x1, bexp: 0x1f, fraction: 0x099cb0 };
    let c = F32 { sign: 0x0, bexp: 0x28, fraction: 0x4d6497 };
    let expected = F32 { sign: 1, bexp: 0x30, fraction: 0x0dd0cf };
    let actual = apfloat::fma<8, 23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_w() {
    type F32 = apfloat::APFloat<8, 23>;
    let a = F32 { sign: 0x0, bexp: 0x90, fraction: 0x0fdde1 };
    let b = F32 { sign: 0x0, bexp: 0xa8, fraction: 0x663085 };
    let c = F32 { sign: 0x0, bexp: 0x9b, fraction: 0x450d69 };
    let expected = F32 { sign: 0, bexp: 0xba, fraction: 0x015c9c };
    let actual = apfloat::fma<8, 23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_x() {
    type F32 = apfloat::APFloat<8, 23>;
    let a = F32 { sign: 0x1, bexp: 0x4c, fraction: 0x5ca821 };
    let b = F32 { sign: 0x0, bexp: 0x87, fraction: 0x14808c };
    let c = F32 { sign: 0x0, bexp: 0x1c, fraction: 0x585ccf };
    let expected = F32 { sign: 1, bexp: 0x55, fraction: 0x000000 };
    let actual = apfloat::fma<8, 23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_y() {
    type F32 = apfloat::APFloat<8, 23>;
    let a = F32 { sign: 0x0, bexp: 0xc5, fraction: 0x3a123b };
    let b = F32 { sign: 0x1, bexp: 0x3b, fraction: 0x7ee4d9 };
    let c = F32 { sign: 0x0, bexp: 0x4f, fraction: 0x1d4ddc };
    let expected = F32 { sign: 1, bexp: 0x82, fraction: 0x39446d };
    let actual = apfloat::fma<8, 23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_z() {
    type F32 = apfloat::APFloat<8, 23>;
    let a = F32 { sign: 0x1, bexp: 0xd4, fraction: 0x1b858b };
    let b = F32 { sign: 0x1, bexp: 0x9e, fraction: 0x59fa23 };
    let c = F32 { sign: 0x1, bexp: 0xb5, fraction: 0x3520e4 };
    let expected = F32 { sign: 0x0, bexp: 0xf4, fraction: 0x046c29 };
    let actual = apfloat::fma<8, 23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_aa() {
    type F32 = apfloat::APFloat<8, 23>;
    let a = F32 { sign: 0x1, bexp: 0x9b, fraction: 0x3ac78d };
    let b = F32 { sign: 0x0, bexp: 0x3b, fraction: 0x542cbb };
    let c = F32 { sign: 0x1, bexp: 0x09, fraction: 0x0c609e };
    let expected = F32 { sign: 0x1, bexp: 0x58, fraction: 0x1acde3 };
    let actual = apfloat::fma<8, 23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fma_subnormal_to_normal_rounding_test() {
    type F32 = apfloat::APFloat<8, 23>;
    let pos_zero = apfloat::zero<8, 23>(false);
    let neg_zero = apfloat::zero<8, 23>(true);
    let min_normal = F32 { sign: 0, bexp: 1, fraction: 0 };
    let neg_min_normal = F32 { sign: 1, bexp: 1, fraction: 0 };

    // Exact halfway between largest subnormal (2^-126 - 2^-149) and min_normal (2^-126):
    // 2^-126 - 2^-150 rounds to even -> min_normal.
    let a = F32 { sign: 0, bexp: 63, fraction: 0x7fffff };
    let b = F32 { sign: 0, bexp: 64, fraction: 0 };
    assert_eq(apfloat::fma<8, 23>(a, b, pos_zero), min_normal);
    assert_eq(apfloat::fma<8, 23>(apfloat::negate(a), b, neg_zero), neg_min_normal);

    // Same boundary via cancellation: 2^-126 - 2^-150 -> min_normal.
    let neg_2_m75 = F32 { sign: 1, bexp: 52, fraction: 0 };
    let pos_2_m75 = F32 { sign: 0, bexp: 52, fraction: 0 };
    assert_eq(apfloat::fma<8, 23>(neg_2_m75, pos_2_m75, min_normal), min_normal);

    // Strictly above midpoint: 2^-126 - 2^-151 -> min_normal.
    let neg_2_m76 = F32 { sign: 1, bexp: 51, fraction: 0 };
    assert_eq(apfloat::fma<8, 23>(neg_2_m76, pos_2_m75, min_normal), min_normal);

    // Largest subnormal (below midpoint): 2^-126 - 2^-149 stays subnormal and flushes to +0.0.
    let neg_2_m74 = F32 { sign: 1, bexp: 53, fraction: 0 };
    assert_eq(apfloat::fma<8, 23>(neg_2_m74, pos_2_m75, min_normal), pos_zero);

    // Same midpoint boundary for BF16: (2 - 2^-7) * 2^-64 * 2^-63 = 2^-126 - 2^-134 -> min_normal.
    type BF16 = apfloat::APFloat<8, 7>;
    let bf_a = BF16 { sign: 0, bexp: 63, fraction: 0x7f };
    let bf_b = BF16 { sign: 0, bexp: 64, fraction: 0 };
    let bf_min_normal = BF16 { sign: 0, bexp: 1, fraction: 0 };
    assert_eq(apfloat::fma<8, 7>(bf_a, bf_b, apfloat::zero<8, 7>(false)), bf_min_normal);
}

#[test]
fn fma_subnormal_intermediate_rounding_test() {
    type F32 = apfloat::APFloat<8, 23>;
    let a = F32 { sign: 0, bexp: 0x6d, fraction: 0x694e0d };
    let b = F32 { sign: 0, bexp: 0x0b, fraction: 0x6ffa53 };
    let c = F32 { sign: 0, bexp: 0x05, fraction: 0x04ee18 };
    let expected = F32 { sign: 0, bexp: 0x05, fraction: 0x05096f };
    let actual = apfloat::fma<8, 23>(a, b, c);
    assert_eq(expected, actual);
}

#[quickcheck]
fn fma_precise_for_small_integers
    (a_mag: u3, a_sign: bool, b_mag: u3, b_sign: bool, c_mag: u3, c_sign: bool) -> bool {
    // All i4 values are representable with full precision in f32 and all
    // intermediate calculations are exact.
    let a = apfloat::cast_from_fixed_using_rne<8, 23>((a_mag as u4) as s4);
    let a = if a_sign { apfloat::negate(a) } else { a };
    let b = apfloat::cast_from_fixed_using_rne<8, 23>((b_mag as u4) as s4);
    let b = if b_sign { apfloat::negate(b) } else { b };
    let c = apfloat::cast_from_fixed_using_rne<8, 23>((c_mag as u4) as s4);
    let c = if c_sign { apfloat::negate(c) } else { c };
    let expected = apfloat::fma<8, 23>(a, b, c);
    let actual = apfloat::add(apfloat::mul(a, b), c);
    expected == actual
}

#[test]
fn has_fractional_part_test() {
    const F32_EXP_SZ = u32:8;
    const F32_FRACTION_SZ = u32:23;
    type F32 = apfloat::APFloat<F32_EXP_SZ, F32_FRACTION_SZ>;
    let one_f32 = apfloat::one<F32_EXP_SZ, F32_FRACTION_SZ>(0);
    let big_f32 = F32 { sign: 0, bexp: apfloat::bias(sN[F32_EXP_SZ]:32), fraction: 0x123 };
    assert_eq(apfloat::has_fractional_part(one_f32), true);
    assert_eq(apfloat::has_fractional_part(big_f32), false);

    const BF16_EXP_SZ = u32:5;
    const BF16_FRACTION_SZ = u32:10;
    type BF16 = apfloat::APFloat<BF16_EXP_SZ, BF16_FRACTION_SZ>;
    let one_bf16 = apfloat::one<BF16_EXP_SZ, BF16_FRACTION_SZ>(0);
    let big_bf16 = BF16 { sign: 0, bexp: apfloat::bias(sN[BF16_EXP_SZ]:10), fraction: 0x12 };
    assert_eq(apfloat::has_fractional_part(one_bf16), true);
    assert_eq(apfloat::has_fractional_part(big_bf16), false);
}

#[test]
fn has_negative_exponent_test() {
    const EXP_SZ = u32:8;
    const FRACTION_SZ = u32:23;
    type F32 = apfloat::APFloat<EXP_SZ, FRACTION_SZ>;
    let zero_f32 = apfloat::zero<EXP_SZ, FRACTION_SZ>(0);
    let zero_dot_5_f32 = F32 { fraction: uN[FRACTION_SZ]:1 << (FRACTION_SZ - 1), ..zero_f32 };
    let one_f32 = apfloat::one<EXP_SZ, FRACTION_SZ>(0);
    assert_eq(apfloat::has_negative_exponent(zero_dot_5_f32), true);
    assert_eq(apfloat::has_negative_exponent(one_f32), false);
}

#[test]
fn ceil_with_denorms_fractional_midpoint_test() {
    const EXP_SZ = u32:8;
    const FRACTION_SZ = u32:23;
    type F32 = apfloat::APFloat<EXP_SZ, FRACTION_SZ>;
    let one_f32 = apfloat::one<EXP_SZ, FRACTION_SZ>(0);
    let two_f32 = apfloat::add(one_f32, one_f32);
    let minus_two_f32 = F32 { sign: 1, ..two_f32 };
    let three_f32 = apfloat::add(one_f32, two_f32);
    let two_dot_5_f32 = F32 { fraction: uN[FRACTION_SZ]:1 << (FRACTION_SZ - 2), ..two_f32 };
    let minus_two_dot_5_f32 = F32 { sign: 1, ..two_dot_5_f32 };
    assert_eq(apfloat::ceil_with_denorms(two_dot_5_f32), three_f32);
    assert_eq(apfloat::ceil_with_denorms(minus_two_dot_5_f32), minus_two_f32);
}

#[test]
fn ceil_with_denorms_fractional_test() {
    const EXP_SZ = u32:8;
    const FRACTION_SZ = u32:23;
    type F32 = apfloat::APFloat<EXP_SZ, FRACTION_SZ>;
    let one_f32 = apfloat::one<EXP_SZ, FRACTION_SZ>(0);
    let two_f32 = apfloat::add(one_f32, one_f32);
    let minus_two_f32 = F32 { sign: 1, ..two_f32 };
    let three_f32 = apfloat::add(one_f32, two_f32);
    let two_dot_0000002_f32 = F32 { fraction: 1, ..two_f32 };
    assert_eq(apfloat::ceil_with_denorms(two_dot_0000002_f32), three_f32);
    let minus_two_dot_0000002_f32 = F32 { sign: 1, ..two_dot_0000002_f32 };
    assert_eq(apfloat::ceil_with_denorms(minus_two_dot_0000002_f32), minus_two_f32);
}

#[test]
fn ceil_with_denorms_big_integral_test() {
    const EXP_SZ = u32:8;
    const FRACTION_SZ = u32:23;
    type F32 = apfloat::APFloat<EXP_SZ, FRACTION_SZ>;
    let big_f32 = F32 { sign: 0, bexp: apfloat::bias(sN[EXP_SZ]:32), fraction: 0x123 };
    assert_eq(apfloat::ceil_with_denorms(big_f32), big_f32);
}

#[test]
fn ceil_with_denorms_already_round_test() {
    const EXP_SZ = u32:8;
    const FRACTION_SZ = u32:23;
    let one_f32 = apfloat::one<EXP_SZ, FRACTION_SZ>(0);
    assert_eq(apfloat::ceil_with_denorms(one_f32), one_f32);
}

#[test]
fn ceil_with_denorms_special() {
    const EXP_SZ = u32:8;
    const FRACTION_SZ = u32:23;
    type F32 = apfloat::APFloat<EXP_SZ, FRACTION_SZ>;
    let inf_f32 = apfloat::inf<EXP_SZ, FRACTION_SZ>(0);
    assert_eq(apfloat::ceil_with_denorms(inf_f32), inf_f32);
    let minus_inf_f32 = F32 { sign: 1, ..inf_f32 };
    assert_eq(apfloat::ceil_with_denorms(minus_inf_f32), minus_inf_f32);
    let zero_f32 = apfloat::zero<EXP_SZ, FRACTION_SZ>(0);
    assert_eq(apfloat::ceil_with_denorms(zero_f32), zero_f32);
    let minus_zero_f32 = F32 { sign: 1, ..zero_f32 };
    assert_eq(apfloat::ceil_with_denorms(minus_zero_f32), minus_zero_f32);
    let qnan_f32 = apfloat::qnan<EXP_SZ, FRACTION_SZ>();
    assert_eq(apfloat::ceil_with_denorms(qnan_f32), qnan_f32);
}

#[test]
fn ceil_with_denorms_denormal_test() {
    const EXP_SZ = u32:8;
    const FRACTION_SZ = u32:23;
    type F32 = apfloat::APFloat<EXP_SZ, FRACTION_SZ>;
    let one_f32 = apfloat::one<EXP_SZ, FRACTION_SZ>(0);
    let denormal_f32 = F32 { sign: 0, bexp: 0, fraction: uN[FRACTION_SZ]:1 << (FRACTION_SZ - 1) };
    assert_eq(apfloat::ceil_with_denorms(denormal_f32), one_f32);
    let minus_zero_f32 = apfloat::zero<EXP_SZ, FRACTION_SZ>(1);
    let minus_denormal_f32 = F32 { sign: 1, ..denormal_f32 };
    assert_eq(apfloat::ceil_with_denorms(minus_denormal_f32), minus_zero_f32);
}

#[test]
fn ceil_daz_fractional_midpoint_test() {
    const EXP_SZ = u32:8;
    const FRACTION_SZ = u32:23;
    type F32 = apfloat::APFloat<EXP_SZ, FRACTION_SZ>;
    let one_f32 = apfloat::one<EXP_SZ, FRACTION_SZ>(0);
    let two_f32 = apfloat::add(one_f32, one_f32);
    let minus_two_f32 = F32 { sign: 1, ..two_f32 };
    let three_f32 = apfloat::add(one_f32, two_f32);
    let two_dot_5_f32 = F32 { fraction: uN[FRACTION_SZ]:1 << (FRACTION_SZ - 2), ..two_f32 };
    let minus_two_dot_5_f32 = F32 { sign: 1, ..two_dot_5_f32 };
    assert_eq(apfloat::ceil_daz(two_dot_5_f32), three_f32);
    assert_eq(apfloat::ceil_daz(minus_two_dot_5_f32), minus_two_f32);
}

#[test]
fn ceil_daz_fractional_test() {
    const EXP_SZ = u32:8;
    const FRACTION_SZ = u32:23;
    type F32 = apfloat::APFloat<EXP_SZ, FRACTION_SZ>;
    let one_f32 = apfloat::one<EXP_SZ, FRACTION_SZ>(0);
    let two_f32 = apfloat::add(one_f32, one_f32);
    let minus_two_f32 = F32 { sign: 1, ..two_f32 };
    let three_f32 = apfloat::add(one_f32, two_f32);
    let two_dot_0000002_f32 = F32 { fraction: 1, ..two_f32 };
    assert_eq(apfloat::ceil_daz(two_dot_0000002_f32), three_f32);
    let minus_two_dot_0000002_f32 = F32 { sign: 1, ..two_dot_0000002_f32 };
    assert_eq(apfloat::ceil_daz(minus_two_dot_0000002_f32), minus_two_f32);
}

#[test]
fn ceil_daz_big_integral_test() {
    const EXP_SZ = u32:8;
    const FRACTION_SZ = u32:23;
    type F32 = apfloat::APFloat<EXP_SZ, FRACTION_SZ>;
    let big_f32 = F32 { sign: 0, bexp: apfloat::bias(sN[EXP_SZ]:32), fraction: 0x123 };
    assert_eq(apfloat::ceil_daz(big_f32), big_f32);
}

#[test]
fn ceil_daz_already_round_test() {
    const EXP_SZ = u32:8;
    const FRACTION_SZ = u32:23;
    let one_f32 = apfloat::one<EXP_SZ, FRACTION_SZ>(0);
    assert_eq(apfloat::ceil_daz(one_f32), one_f32);
}

#[test]
fn ceil_daz_special() {
    const EXP_SZ = u32:8;
    const FRACTION_SZ = u32:23;
    type F32 = apfloat::APFloat<EXP_SZ, FRACTION_SZ>;
    let inf_f32 = apfloat::inf<EXP_SZ, FRACTION_SZ>(0);
    assert_eq(apfloat::ceil_daz(inf_f32), inf_f32);
    let minus_inf_f32 = F32 { sign: 1, ..inf_f32 };
    assert_eq(apfloat::ceil_daz(minus_inf_f32), minus_inf_f32);
    let zero_f32 = apfloat::zero<EXP_SZ, FRACTION_SZ>(0);
    assert_eq(apfloat::ceil_daz(zero_f32), zero_f32);
    let minus_zero_f32 = F32 { sign: 1, ..zero_f32 };
    assert_eq(apfloat::ceil_daz(minus_zero_f32), minus_zero_f32);
    let qnan_f32 = apfloat::qnan<EXP_SZ, FRACTION_SZ>();
    assert_eq(apfloat::ceil_daz(qnan_f32), qnan_f32);
}

#[test]
fn ceil_daz_denormal_test() {
    const EXP_SZ = u32:8;
    const FRACTION_SZ = u32:23;
    type F32 = apfloat::APFloat<EXP_SZ, FRACTION_SZ>;
    let zero_f32 = apfloat::zero<EXP_SZ, FRACTION_SZ>(0);
    let denormal_f32 = F32 { sign: 0, bexp: 0, fraction: uN[FRACTION_SZ]:1 << (FRACTION_SZ - 1) };
    assert_eq(apfloat::ceil_daz(denormal_f32), zero_f32);
    let minus_zero_f32 = apfloat::zero<EXP_SZ, FRACTION_SZ>(1);
    let minus_denormal_f32 = F32 { sign: 1, ..denormal_f32 };
    assert_eq(apfloat::ceil_daz(minus_denormal_f32), minus_zero_f32);
}

#[quickcheck]
fn ceil_daz_denorms_round_to_signed_zero(f: apfloat::APFloat<8, 7>) -> bool {
    !apfloat::is_zero_or_subnormal(f) || apfloat::ceil_daz(f) == apfloat::zero<8, 7>(f.sign)
}

#[test]
fn floor_with_denorms_fractional_midpoint_test() {
    const EXP_SZ = u32:8;
    const FRACTION_SZ = u32:23;
    type F32 = apfloat::APFloat<EXP_SZ, FRACTION_SZ>;
    let one_f32 = apfloat::one<EXP_SZ, FRACTION_SZ>(0);
    let two_f32 = apfloat::add(one_f32, one_f32);
    let three_f32 = apfloat::add(one_f32, two_f32);
    let minus_three_f32 = F32 { sign: 1, ..three_f32 };
    let two_dot_5_f32 = F32 { fraction: uN[FRACTION_SZ]:1 << (FRACTION_SZ - 2), ..two_f32 };
    let minus_two_dot_5_f32 = F32 { sign: 1, ..two_dot_5_f32 };
    assert_eq(apfloat::floor_with_denorms(two_dot_5_f32), two_f32);
    assert_eq(apfloat::floor_with_denorms(minus_two_dot_5_f32), minus_three_f32);
}

#[test]
fn floor_with_denorms_fractional_test() {
    const EXP_SZ = u32:8;
    const FRACTION_SZ = u32:23;
    type F32 = apfloat::APFloat<EXP_SZ, FRACTION_SZ>;
    let one_f32 = apfloat::one<EXP_SZ, FRACTION_SZ>(0);
    let two_f32 = apfloat::add(one_f32, one_f32);
    let three_f32 = apfloat::add(one_f32, two_f32);
    let minus_three_f32 = F32 { sign: 1, ..three_f32 };
    let two_dot_5000002_f32 =
        F32 { fraction: uN[FRACTION_SZ]:1 << (FRACTION_SZ - 2) | 1, ..two_f32 };
    let minus_two_dot_0000002_f32 = F32 { sign: 1, ..two_dot_5000002_f32 };
    assert_eq(apfloat::floor_with_denorms(two_dot_5000002_f32), two_f32);
    assert_eq(apfloat::floor_with_denorms(minus_two_dot_0000002_f32), minus_three_f32);
}

#[test]
fn floor_with_denorms_big_integral_test() {
    const EXP_SZ = u32:8;
    const FRACTION_SZ = u32:23;
    type F32 = apfloat::APFloat<EXP_SZ, FRACTION_SZ>;
    let big_f32 = F32 { sign: 0, bexp: apfloat::bias(sN[EXP_SZ]:32), fraction: 0x123 };
    assert_eq(apfloat::floor_with_denorms(big_f32), big_f32);
}

#[test]
fn floor_with_denorms_already_round_test() {
    const EXP_SZ = u32:8;
    const FRACTION_SZ = u32:23;
    let one_f32 = apfloat::one<EXP_SZ, FRACTION_SZ>(0);
    assert_eq(apfloat::floor_with_denorms(one_f32), one_f32);
}

#[test]
fn floor_with_denorms_special() {
    const EXP_SZ = u32:8;
    const FRACTION_SZ = u32:23;
    type F32 = apfloat::APFloat<EXP_SZ, FRACTION_SZ>;
    let inf_f32 = apfloat::inf<EXP_SZ, FRACTION_SZ>(0);
    assert_eq(apfloat::floor_with_denorms(inf_f32), inf_f32);
    let minus_inf_f32 = F32 { sign: 1, ..inf_f32 };
    assert_eq(apfloat::floor_with_denorms(minus_inf_f32), minus_inf_f32);
    let zero_f32 = apfloat::zero<EXP_SZ, FRACTION_SZ>(0);
    assert_eq(apfloat::floor_with_denorms(zero_f32), zero_f32);
    let minus_zero_f32 = F32 { sign: 1, ..zero_f32 };
    assert_eq(apfloat::floor_with_denorms(minus_zero_f32), minus_zero_f32);
    let qnan_f32 = apfloat::qnan<EXP_SZ, FRACTION_SZ>();
    assert_eq(apfloat::floor_with_denorms(qnan_f32), qnan_f32);
}

#[test]
fn floor_with_denorms_denormal_test() {
    const EXP_SZ = u32:8;
    const FRACTION_SZ = u32:23;
    type F32 = apfloat::APFloat<EXP_SZ, FRACTION_SZ>;
    let zero_f32 = apfloat::zero<EXP_SZ, FRACTION_SZ>(0);
    let denormal_f32 = F32 { sign: 0, bexp: 0, fraction: uN[FRACTION_SZ]:1 << (FRACTION_SZ - 1) };
    assert_eq(apfloat::floor_with_denorms(denormal_f32), zero_f32);
    let minus_one_f32 = apfloat::one<EXP_SZ, FRACTION_SZ>(1);
    let minus_denormal_f32 = F32 { sign: 1, ..denormal_f32 };
    assert_eq(apfloat::floor_with_denorms(minus_denormal_f32), minus_one_f32);
}

#[test]
fn floor_daz_fractional_midpoint_test() {
    const EXP_SZ = u32:8;
    const FRACTION_SZ = u32:23;
    type F32 = apfloat::APFloat<EXP_SZ, FRACTION_SZ>;
    let one_f32 = apfloat::one<EXP_SZ, FRACTION_SZ>(0);
    let two_f32 = apfloat::add(one_f32, one_f32);
    let three_f32 = apfloat::add(one_f32, two_f32);
    let minus_three_f32 = F32 { sign: 1, ..three_f32 };
    let two_dot_5_f32 = F32 { fraction: uN[FRACTION_SZ]:1 << (FRACTION_SZ - 2), ..two_f32 };
    let minus_two_dot_5_f32 = F32 { sign: 1, ..two_dot_5_f32 };
    assert_eq(apfloat::floor_daz(two_dot_5_f32), two_f32);
    assert_eq(apfloat::floor_daz(minus_two_dot_5_f32), minus_three_f32);
}

#[test]
fn floor_daz_fractional_test() {
    const EXP_SZ = u32:8;
    const FRACTION_SZ = u32:23;
    type F32 = apfloat::APFloat<EXP_SZ, FRACTION_SZ>;
    let one_f32 = apfloat::one<EXP_SZ, FRACTION_SZ>(0);
    let two_f32 = apfloat::add(one_f32, one_f32);
    let three_f32 = apfloat::add(one_f32, two_f32);
    let minus_three_f32 = F32 { sign: 1, ..three_f32 };
    let two_dot_5000002_f32 =
        F32 { fraction: uN[FRACTION_SZ]:1 << (FRACTION_SZ - 2) | 1, ..two_f32 };
    let minus_two_dot_0000002_f32 = F32 { sign: 1, ..two_dot_5000002_f32 };
    assert_eq(apfloat::floor_daz(two_dot_5000002_f32), two_f32);
    assert_eq(apfloat::floor_daz(minus_two_dot_0000002_f32), minus_three_f32);
}

#[test]
fn floor_daz_big_integral_test() {
    const EXP_SZ = u32:8;
    const FRACTION_SZ = u32:23;
    type F32 = apfloat::APFloat<EXP_SZ, FRACTION_SZ>;
    let big_f32 = F32 { sign: 0, bexp: apfloat::bias(sN[EXP_SZ]:32), fraction: 0x123 };
    assert_eq(apfloat::floor_daz(big_f32), big_f32);
}

#[test]
fn floor_daz_already_round_test() {
    const EXP_SZ = u32:8;
    const FRACTION_SZ = u32:23;
    let one_f32 = apfloat::one<EXP_SZ, FRACTION_SZ>(0);
    assert_eq(apfloat::floor_daz(one_f32), one_f32);
}

#[test]
fn floor_daz_special() {
    const EXP_SZ = u32:8;
    const FRACTION_SZ = u32:23;
    type F32 = apfloat::APFloat<EXP_SZ, FRACTION_SZ>;
    let inf_f32 = apfloat::inf<EXP_SZ, FRACTION_SZ>(0);
    assert_eq(apfloat::floor_daz(inf_f32), inf_f32);
    let minus_inf_f32 = F32 { sign: 1, ..inf_f32 };
    assert_eq(apfloat::floor_daz(minus_inf_f32), minus_inf_f32);
    let zero_f32 = apfloat::zero<EXP_SZ, FRACTION_SZ>(0);
    assert_eq(apfloat::floor_daz(zero_f32), zero_f32);
    let minus_zero_f32 = F32 { sign: 1, ..zero_f32 };
    assert_eq(apfloat::floor_daz(minus_zero_f32), minus_zero_f32);
    let qnan_f32 = apfloat::qnan<EXP_SZ, FRACTION_SZ>();
    assert_eq(apfloat::floor_daz(qnan_f32), qnan_f32);
}

#[test]
fn floor_daz_denormal_test() {
    const EXP_SZ = u32:8;
    const FRACTION_SZ = u32:23;
    type F32 = apfloat::APFloat<EXP_SZ, FRACTION_SZ>;
    let zero_f32 = apfloat::zero<EXP_SZ, FRACTION_SZ>(0);
    let denormal_f32 = F32 { sign: 0, bexp: 0, fraction: uN[FRACTION_SZ]:1 << (FRACTION_SZ - 1) };
    assert_eq(apfloat::floor_daz(denormal_f32), zero_f32);
    let minus_zero_f32 = apfloat::zero<EXP_SZ, FRACTION_SZ>(1);
    let minus_denormal_f32 = F32 { sign: 1, ..denormal_f32 };
    assert_eq(apfloat::floor_daz(minus_denormal_f32), minus_zero_f32);
}

#[quickcheck]
fn floor_daz_denorms_round_to_signed_zero(f: apfloat::APFloat<8, 7>) -> bool {
    !apfloat::is_zero_or_subnormal(f) || apfloat::floor_daz(f) == apfloat::zero<8, 7>(f.sign)
}

#[test]
fn trunc_fractional_midpoint_test() {
    const EXP_SZ = u32:8;
    const FRACTION_SZ = u32:23;
    type F32 = apfloat::APFloat<EXP_SZ, FRACTION_SZ>;
    let one_f32 = apfloat::one<EXP_SZ, FRACTION_SZ>(0);
    let two_f32 = apfloat::add(one_f32, one_f32);
    let two_dot_5_f32 = F32 { fraction: uN[FRACTION_SZ]:1 << (FRACTION_SZ - 2), ..two_f32 };
    let minus_two_f32 = F32 { sign: 1, ..two_f32 };
    let minus_two_dot_5_f32 = F32 { sign: 1, ..two_dot_5_f32 };
    assert_eq(apfloat::trunc(two_dot_5_f32), two_f32);
    assert_eq(apfloat::trunc(minus_two_dot_5_f32), minus_two_f32);
}

#[test]
fn trunc_fractional_test() {
    const EXP_SZ = u32:8;
    const FRACTION_SZ = u32:23;
    type F32 = apfloat::APFloat<EXP_SZ, FRACTION_SZ>;
    let one_f32 = apfloat::one<EXP_SZ, FRACTION_SZ>(0);
    let two_f32 = apfloat::add(one_f32, one_f32);
    let two_dot_5000002_f32 =
        F32 { fraction: uN[FRACTION_SZ]:1 << (FRACTION_SZ - 2) | 1, ..two_f32 };
    let minus_two_f32 = F32 { sign: 1, ..two_f32 };
    let minus_two_dot_0000002_f32 = F32 { sign: 1, ..two_dot_5000002_f32 };
    assert_eq(apfloat::trunc(two_dot_5000002_f32), two_f32);
    assert_eq(apfloat::trunc(minus_two_dot_0000002_f32), minus_two_f32);
}

#[test]
fn trunc_big_integral_test() {
    const EXP_SZ = u32:8;
    const FRACTION_SZ = u32:23;
    type F32 = apfloat::APFloat<EXP_SZ, FRACTION_SZ>;
    let big_f32 = F32 { sign: 0, bexp: apfloat::bias(sN[EXP_SZ]:32), fraction: 0x123 };
    assert_eq(apfloat::trunc(big_f32), big_f32);
}

#[test]
fn trunc_already_round_test() {
    const EXP_SZ = u32:8;
    const FRACTION_SZ = u32:23;
    let one_f32 = apfloat::one<EXP_SZ, FRACTION_SZ>(0);
    assert_eq(apfloat::trunc(one_f32), one_f32);
}

#[test]
fn trunc_special() {
    const EXP_SZ = u32:8;
    const FRACTION_SZ = u32:23;
    type F32 = apfloat::APFloat<EXP_SZ, FRACTION_SZ>;
    let inf_f32 = apfloat::inf<EXP_SZ, FRACTION_SZ>(0);
    assert_eq(apfloat::trunc(inf_f32), inf_f32);
    let minus_inf_f32 = F32 { sign: 1, ..inf_f32 };
    assert_eq(apfloat::trunc(minus_inf_f32), minus_inf_f32);
    let zero_f32 = apfloat::zero<EXP_SZ, FRACTION_SZ>(0);
    assert_eq(apfloat::trunc(zero_f32), zero_f32);
    let minus_zero_f32 = F32 { sign: 1, ..zero_f32 };
    assert_eq(apfloat::trunc(minus_zero_f32), minus_zero_f32);
    let qnan_f32 = apfloat::qnan<EXP_SZ, FRACTION_SZ>();
    assert_eq(apfloat::trunc(qnan_f32), qnan_f32);
}

#[test]
fn trunc_zero_fractional_test() {
    const EXP_SZ = u32:8;
    const FRACTION_SZ = u32:23;
    type F32 = apfloat::APFloat<EXP_SZ, FRACTION_SZ>;
    let zero_f32 = apfloat::zero<EXP_SZ, FRACTION_SZ>(0);
    let zero_dot_5_f32 = F32 { fraction: uN[FRACTION_SZ]:1 << (FRACTION_SZ - 1), ..zero_f32 };
    assert_eq(apfloat::trunc(zero_dot_5_f32), zero_f32);
    let minus_zero_f32 = apfloat::zero<EXP_SZ, FRACTION_SZ>(1);
    let minus_zero_dot_5_f32 = F32 { sign: 1, ..zero_dot_5_f32 };
    assert_eq(apfloat::trunc(minus_zero_dot_5_f32), minus_zero_f32);
}

#[test]
fn round_fractional_test() {
    const F32_EXP_SZ = u32:8;
    const F32_FRACTION_SZ = u32:23;
    let two_f32 =
        apfloat::APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 0, bexp: 0x80, fraction: 0 };
    let minus_two_f32 = apfloat::APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 1, ..two_f32 };
    let three_f32 =
        apfloat::APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 0, bexp: 0x80, fraction: 0x400000 };
    let minus_three_f32 = apfloat::APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 1, ..three_f32 };
    // round(2.0000002) == 2.0
    assert_eq(apfloat::round(apfloat::APFloat<8, 23> { fraction: 0x1, ..two_f32 }), two_f32);
    // round(-2.0000002) == -2.0
    assert_eq(
        apfloat::round(apfloat::APFloat<8, 23> { fraction: 0x1, ..minus_two_f32 }), minus_two_f32);
    // round(2.9999997) == 3.0
    assert_eq(apfloat::round(apfloat::APFloat<8, 23> { fraction: 0x3fffff, ..two_f32 }), three_f32);
    // round(-2.9999997) = -3.0
    assert_eq(
        apfloat::round(apfloat::APFloat<8, 23> { fraction: 0x3fffff, ..minus_two_f32 }),
        minus_three_f32);
}

#[test]
fn round_fractional_subnormal_test() {
    const F32_EXP_SZ = u32:8;
    const F32_FRACTION_SZ = u32:23;
    let zero_f32 = apfloat::zero<F32_EXP_SZ, F32_FRACTION_SZ>(0);
    let minus_zero_f32 = apfloat::zero<F32_EXP_SZ, F32_FRACTION_SZ>(1);

    // round(1e-45) == 0.0
    assert_eq(apfloat::round(apfloat::APFloat<8, 23> { fraction: 0x1, ..zero_f32 }), zero_f32);
    // round(-1e-45) == -0.0
    assert_eq(
        apfloat::round(apfloat::APFloat<8, 23> { fraction: 0x1, ..minus_zero_f32 }), minus_zero_f32);
    // round(1.1754942e-38) == 0.0
    assert_eq(apfloat::round(apfloat::APFloat<8, 23> { fraction: 0x7fffff, ..zero_f32 }), zero_f32);
    // round(-1.1754942e-38) == -0.0
    assert_eq(
        apfloat::round(apfloat::APFloat<8, 23> { fraction: 0x7fffff, ..minus_zero_f32 }),
        minus_zero_f32);
}

#[test]
fn round_fractional_negative_exp_test() {
    const F32_EXP_SZ = u32:8;
    const F32_FRACTION_SZ = u32:23;
    let zero_f32 = apfloat::zero<F32_EXP_SZ, F32_FRACTION_SZ>(0);
    let minus_zero_f32 = apfloat::zero<F32_EXP_SZ, F32_FRACTION_SZ>(1);

    let one_f32 = apfloat::one<F32_EXP_SZ, F32_FRACTION_SZ>(0);
    let minus_one_f32 = apfloat::one<F32_EXP_SZ, F32_FRACTION_SZ>(1);
    // round(1.1754945e-38) == 0.0
    assert_eq(
        apfloat::round(apfloat::APFloat<8, 23> { sign: 0, bexp: 0x1, fraction: 0x1 }), zero_f32);
    // round(-1.1754945e-38) == -0.0
    assert_eq(
        apfloat::round(apfloat::APFloat<8, 23> { sign: 1, bexp: 0x1, fraction: 0x1 }),
        minus_zero_f32);
    // round(0.49999997) == 0.0
    assert_eq(
        apfloat::round(apfloat::APFloat<8, 23> { sign: 0, bexp: 0x7d, fraction: 0x7fffff }),
        zero_f32);
    // round(-0.49999997) == -0.0
    assert_eq(
        apfloat::round(apfloat::APFloat<8, 23> { sign: 1, bexp: 0x7d, fraction: 0x7fffff }),
        minus_zero_f32);
    // round(0.5) == 0.0
    assert_eq(
        apfloat::round(apfloat::APFloat<8, 23> { sign: 0, bexp: 0x7e, fraction: 0 }), zero_f32);
    // round(-0.5) == -0.0
    assert_eq(
        apfloat::round(apfloat::APFloat<8, 23> { sign: 1, bexp: 0x7e, fraction: 0 }), minus_zero_f32);
    // round(0.50000006) == 1.0
    assert_eq(apfloat::round(apfloat::APFloat<8, 23> { sign: 0, bexp: 0x7e, fraction: 1 }), one_f32);
    // round(-0.50000006) == -1.0
    assert_eq(
        apfloat::round(apfloat::APFloat<8, 23> { sign: 1, bexp: 0x7e, fraction: 1 }), minus_one_f32);
    // round(0.9999999) == 1.0
    assert_eq(
        apfloat::round(apfloat::APFloat<8, 23> { sign: 0, bexp: 0x7e, fraction: 0x7ffffe }), one_f32);
    // round(-0.9999999) == -1.0
    assert_eq(
        apfloat::round(apfloat::APFloat<8, 23> { sign: 1, bexp: 0x7e, fraction: 0x7ffffe }),
        minus_one_f32);
}

#[test]
fn round_fractional_zero_exp_test() {
    const F32_EXP_SZ = u32:8;
    const F32_FRACTION_SZ = u32:23;
    let one_f32 = apfloat::one<F32_EXP_SZ, F32_FRACTION_SZ>(0);
    let minus_one_f32 = apfloat::one<F32_EXP_SZ, F32_FRACTION_SZ>(1);
    let two_f32 =
        apfloat::APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 0, bexp: 0x80, fraction: 0 };
    let minus_two_f32 = apfloat::APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 1, ..two_f32 };
    // round(1.0000001) == 1.0
    assert_eq(apfloat::round(apfloat::APFloat<8, 23> { fraction: 0x1, ..one_f32 }), one_f32);
    // round(-1.0000001) == -1.0
    assert_eq(
        apfloat::round(apfloat::APFloat<8, 23> { fraction: 0x1, ..minus_one_f32 }), minus_one_f32);
    // round(1.9999999) == 2.0
    assert_eq(apfloat::round(apfloat::APFloat<8, 23> { fraction: 0x7fffff, ..one_f32 }), two_f32);
    // round(-1.9999999) == -2.0
    assert_eq(
        apfloat::round(apfloat::APFloat<8, 23> { fraction: 0x7fffff, ..minus_one_f32 }),
        minus_two_f32);
}

#[test]
fn round_fractional_midpoint_test() {
    const F32_EXP_SZ = u32:8;
    const F32_FRACTION_SZ = u32:23;
    let two_f32 =
        apfloat::APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 0, bexp: 0x80, fraction: 0 };
    let minus_two_f32 = apfloat::APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 1, ..two_f32 };
    let three_f32 =
        apfloat::APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 0, bexp: 0x80, fraction: 0x400000 };
    let minus_three_f32 = apfloat::APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 1, ..three_f32 };
    let four_f32 =
        apfloat::APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 0, bexp: 0x81, fraction: 0 };
    let minus_four_f32 = apfloat::APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 1, ..four_f32 };
    // round(3.5) == 4.0
    assert_eq(apfloat::round(apfloat::APFloat<8, 23> { fraction: 0x600000, ..three_f32 }), four_f32);
    // round(-3.5) = -4.0
    assert_eq(
        apfloat::round(apfloat::APFloat<8, 23> { fraction: 0x600000, ..minus_three_f32 }),
        minus_four_f32);
    // round(2.5) == 2.0
    assert_eq(apfloat::round(apfloat::APFloat<8, 23> { fraction: 0x200000, ..two_f32 }), two_f32);
    // round(-2.5) == -2.0
    assert_eq(
        apfloat::round(apfloat::APFloat<8, 23> { fraction: 0x200000, ..minus_two_f32 }),
        minus_two_f32);
}

#[test]
fn round_already_round_test() {
    const F32_EXP_SZ = u32:8;
    const F32_FRACTION_SZ = u32:23;
    let one_f32 = apfloat::one<F32_EXP_SZ, F32_FRACTION_SZ>(0);
    assert_eq(apfloat::round(one_f32), one_f32);
}

#[test]
fn round_special() {
    const F32_EXP_SZ = u32:8;
    const F32_FRACTION_SZ = u32:23;
    let inf_f32 = apfloat::inf<F32_EXP_SZ, F32_FRACTION_SZ>(0);
    assert_eq(apfloat::round(inf_f32), inf_f32);
    let minus_inf_f32 = apfloat::APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 1, ..inf_f32 };
    assert_eq(apfloat::round(minus_inf_f32), minus_inf_f32);
    let zero_f32 = apfloat::zero<F32_EXP_SZ, F32_FRACTION_SZ>(0);
    assert_eq(apfloat::round(zero_f32), zero_f32);
    let minus_zero_f32 = apfloat::APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 1, ..zero_f32 };
    assert_eq(apfloat::round(minus_zero_f32), minus_zero_f32);
    let qnan_f32 = apfloat::qnan<F32_EXP_SZ, F32_FRACTION_SZ>();
    assert_eq(apfloat::round(qnan_f32), qnan_f32);
}

#[quickcheck]
fn round_never_assert_fails(f: apfloat::APFloat<8, 23>) -> bool {
    // nb asserts are checked.
    apfloat::round(f);
    true
}

#[quickcheck]
fn downcast_never_assert_fails
    (f: apfloat::APFloat<8, 23>, round_style: apfloat::RoundStyle) -> bool {
    // nb asserts are checked.
    apfloat::downcast<10, 8>(f, round_style);
    apfloat::downcast<7, 8>(f, round_style);
    apfloat::downcast<3, 4>(f, round_style);
    apfloat::downcast<2, 5>(f, round_style);
    true
}

#[quickcheck]
fn cast_from_fixed_using_rne_never_assert_fails(v: s64) -> bool {
    // nb asserts are checked.
    apfloat::cast_from_fixed_using_rne<4, 4>(v);
    apfloat::cast_from_fixed_using_rne<8, 8>(v);
    apfloat::cast_from_fixed_using_rne<8, 23>(v);
    apfloat::cast_from_fixed_using_rne<20, 43>(v);
    apfloat::cast_from_fixed_using_rne<25, 60>(v);
    true
}
