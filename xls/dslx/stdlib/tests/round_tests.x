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

#![feature(type_inference_v2)]

import std;
import round;

#[test]
fn test_wrappers_well_formed() {
    // ensures the "wrappers" are well-formed.
    let num_bits_rounded = u3:1;
    let unrounded = u5:0b0001_0;

    // unsigned
    assert_eq(
        (u1:0, u5:0b0001_0), round::round_u(round::RoundingMode::RNE, num_bits_rounded, unrounded));
    assert_eq((u1:0, u2:0b00), round::round_trunc_u<u32:3>(round::RoundingMode::RNE, unrounded));
    assert_eq((u1:0, u2:0b00), round::round_trunc_to_u<u32:2>(round::RoundingMode::RNE, unrounded));

    // sign & magnitude
    assert_eq(
        (u1:0, u5:0b0001_0),
        round::round_sm(
            round::RoundingMode::RNE, num_bits_rounded, round::Sign::Negative, unrounded));
    assert_eq(
        (u1:0, u2:0b00),
        round::round_trunc_sm<u32:3>(round::RoundingMode::RNE, round::Sign::Negative, unrounded));
    assert_eq(
        (u1:0, u2:0b00),
        round::round_trunc_to_sm<u32:2>(round::RoundingMode::RNE, round::Sign::Negative, unrounded));

    // signed
    assert_eq(
        (u1:0, s5:0b0001_0),
        round::round_s(round::RoundingMode::RNE, num_bits_rounded, unrounded as s5));
    assert_eq(
        (u1:0, s2:0b00), round::round_trunc_s<u32:3>(round::RoundingMode::RNE, unrounded as s5));
    assert_eq(
        (u1:0, s2:0b00), round::round_trunc_to_s<u32:2>(round::RoundingMode::RNE, unrounded as s5));
}

#[test]
fn test_can_round_all_bits_pow2() {
    // We may want to round away all N bits when N is a power of 2. In such a case, clog2(N+1)
    // is needed.
    let num_bits_rounded = u3:4;
    let unrounded = u4:0b1000;
    assert_eq(
        (u1:0, u4:0b0000), round::round_u(round::RoundingMode::RNE, num_bits_rounded, unrounded));
}

// ** Test N = 0 doesn't cause a compiler error **
#[test]
fn round_zero_width_unsigned_rne() {
    let (_, rounded) = round::round<false, u32:0>(
        round::RoundingMode::RNE, uN[0]:0, round::Sign::NonNegative, uN[0]:0);
    assert_eq(uN[0]:0, rounded);
}

#[test]
fn round_zero_width_dynamic_rtz() {
    let (overflow, rounded) = round::round<false, u32:0>(
        round::RoundingMode::RTZ, uN[0]:0, round::Sign::NonNegative, uN[0]:0);
    assert_eq(u1:0, overflow);
    assert_eq(uN[0]:0, rounded);
}

#[test]
fn round_u_zero_width_rtz() {
    let (overflow, rounded) = round::round_u<u32:0>(round::RoundingMode::RTZ, uN[0]:0, uN[0]:0);
    assert_eq(u1:0, overflow);
    assert_eq(uN[0]:0, rounded);
}

#[test]
fn round_trunc_s_all_bits_rtz() {
    // Forces round_trunc_s to compute R = N - num_bits_rounded = 0
    let (_, rounded) = round::round_trunc_s<u32:4, u32:4>(round::RoundingMode::RTZ, sN[4]:0);
    assert_eq(sN[0]:0, rounded);
}

#[test]
fn round_trunc_u_all_bits_rtz() {
    // Forces round_trunc_u to compute R = 0.
    let (_, rounded) = round::round_trunc_u<u32:4, u32:4>(round::RoundingMode::RTZ, uN[4]:0);
    assert_eq(uN[0]:0, rounded);
}

#[test]
fn round_trunc_to_u_zero_width_rtz() {
    let (overflow, rounded) =
        round::round_trunc_to_u<u32:0, u32:4>(round::RoundingMode::RTZ, uN[4]:0);
    assert_eq(u1:0, overflow);
    assert_eq(uN[0]:0, rounded);
}

#[test]
fn round_trunc_sm_all_bits_rtz() {
    // Forces round_trunc_sm to compute R = 0.
    let (_, rounded) = round::round_trunc_sm<u32:4, u32:4>(
        round::RoundingMode::RTZ, round::Sign::NonNegative, uN[4]:0);
    assert_eq(uN[0]:0, rounded);
}

#[test]
fn round_s_zero_width_rtz() {
    let (overflow, rounded) = round::round_s<u32:0>(round::RoundingMode::RTZ, uN[0]:0, sN[0]:0);
    assert_eq(u1:0, overflow);
    assert_eq(sN[0]:0, rounded);
}

#[test]
fn round_trunc_to_s_zero_width_rtz() {
    let (overflow, rounded) =
        round::round_trunc_to_s<u32:0, u32:4>(round::RoundingMode::RTZ, sN[4]:0);
    assert_eq(u1:0, overflow);
    assert_eq(sN[0]:0, rounded);
}

#[test]
fn round_sm_zero_width_rtz() {
    let (overflow, rounded) = round::round_sm<u32:0>(
        round::RoundingMode::RTZ, uN[0]:0, round::Sign::NonNegative, uN[0]:0);
    assert_eq(u1:0, overflow);
    assert_eq(uN[0]:0, rounded);
}

#[test]
fn round_trunc_to_sm_zero_width_rtz() {
    let (overflow, rounded) = round::round_trunc_to_sm<u32:0, u32:4>(
        round::RoundingMode::RTZ, round::Sign::NonNegative, uN[4]:0);
    assert_eq(u1:0, overflow);
    assert_eq(uN[0]:0, rounded);
}

// ** Test num_bits_rounded > N **

fn exercise_round_u_more_fractional_bits(num_bits_rounded: uN[3]) {
    let zero = u5:0;
    let half = u5:0b1_0000;
    let all_ones = u5:0b1_1111;

    assert_eq((u1:0, u5:0), round::round_u(round::RoundingMode::RNE, num_bits_rounded, zero));
    assert_eq((u1:0, u5:0), round::round_u(round::RoundingMode::RNE, num_bits_rounded, half));
    assert_eq((u1:0, u5:0), round::round_u(round::RoundingMode::RNE, num_bits_rounded, all_ones));

    assert_eq((u1:0, u5:0), round::round_u(round::RoundingMode::RNA, num_bits_rounded, zero));
    assert_eq((u1:0, u5:0), round::round_u(round::RoundingMode::RNA, num_bits_rounded, half));
    assert_eq((u1:0, u5:0), round::round_u(round::RoundingMode::RNA, num_bits_rounded, all_ones));

    assert_eq((u1:0, u5:0), round::round_u(round::RoundingMode::RTZ, num_bits_rounded, zero));
    assert_eq((u1:0, u5:0), round::round_u(round::RoundingMode::RTZ, num_bits_rounded, half));
    assert_eq((u1:0, u5:0), round::round_u(round::RoundingMode::RTZ, num_bits_rounded, all_ones));

    assert_eq((u1:0, u5:0), round::round_u(round::RoundingMode::RTN, num_bits_rounded, zero));
    assert_eq((u1:0, u5:0), round::round_u(round::RoundingMode::RTN, num_bits_rounded, half));
    assert_eq((u1:0, u5:0), round::round_u(round::RoundingMode::RTN, num_bits_rounded, all_ones));

    assert_eq((u1:0, u5:0), round::round_u(round::RoundingMode::RTP, num_bits_rounded, zero));
    assert_eq((u1:1, u5:0), round::round_u(round::RoundingMode::RTP, num_bits_rounded, half));
    assert_eq((u1:1, u5:0), round::round_u(round::RoundingMode::RTP, num_bits_rounded, all_ones));
}

fn exercise_round_s_more_fractional_bits(num_bits_rounded: uN[3]) {
    let zero = s5:0;
    let little_positive = s5:8;
    let little_negative = s5:-8;

    assert_eq((u1:0, s5:0), round::round_s(round::RoundingMode::RNE, num_bits_rounded, zero));
    assert_eq(
        (u1:0, s5:0), round::round_s(round::RoundingMode::RNE, num_bits_rounded, little_positive));
    assert_eq(
        (u1:0, s5:0), round::round_s(round::RoundingMode::RNE, num_bits_rounded, little_negative));

    assert_eq((u1:0, s5:0), round::round_s(round::RoundingMode::RNA, num_bits_rounded, zero));
    assert_eq(
        (u1:0, s5:0), round::round_s(round::RoundingMode::RNA, num_bits_rounded, little_positive));
    assert_eq(
        (u1:0, s5:0), round::round_s(round::RoundingMode::RNA, num_bits_rounded, little_negative));

    assert_eq((u1:0, s5:0), round::round_s(round::RoundingMode::RTZ, num_bits_rounded, zero));
    assert_eq(
        (u1:0, s5:0), round::round_s(round::RoundingMode::RTZ, num_bits_rounded, little_positive));
    assert_eq(
        (u1:0, s5:0), round::round_s(round::RoundingMode::RTZ, num_bits_rounded, little_negative));

    assert_eq((u1:0, s5:0), round::round_s(round::RoundingMode::RTN, num_bits_rounded, zero));
    assert_eq(
        (u1:0, s5:0), round::round_s(round::RoundingMode::RTN, num_bits_rounded, little_positive));
    assert_eq(
        (u1:1, s5:0), round::round_s(round::RoundingMode::RTN, num_bits_rounded, little_negative));

    assert_eq((u1:0, s5:0), round::round_s(round::RoundingMode::RTP, num_bits_rounded, zero));
    assert_eq(
        (u1:1, s5:0), round::round_s(round::RoundingMode::RTP, num_bits_rounded, little_positive));
    assert_eq(
        (u1:0, s5:0), round::round_s(round::RoundingMode::RTP, num_bits_rounded, little_negative));
}

fn exercise_round_sm_more_fractional_bits(num_bits_rounded: uN[3], sign: round::Sign) {
    let zero = u5:0;
    let half = u5:0b1_0000;
    let all_ones = u5:0b1_1111;

    assert_eq((u1:0, u5:0), round::round_sm(round::RoundingMode::RNE, num_bits_rounded, sign, zero));
    assert_eq((u1:0, u5:0), round::round_sm(round::RoundingMode::RNE, num_bits_rounded, sign, half));
    assert_eq(
        (u1:0, u5:0), round::round_sm(round::RoundingMode::RNE, num_bits_rounded, sign, all_ones));

    assert_eq((u1:0, u5:0), round::round_sm(round::RoundingMode::RNA, num_bits_rounded, sign, zero));
    assert_eq((u1:0, u5:0), round::round_sm(round::RoundingMode::RNA, num_bits_rounded, sign, half));
    assert_eq(
        (u1:0, u5:0), round::round_sm(round::RoundingMode::RNA, num_bits_rounded, sign, all_ones));

    assert_eq((u1:0, u5:0), round::round_sm(round::RoundingMode::RTZ, num_bits_rounded, sign, zero));
    assert_eq((u1:0, u5:0), round::round_sm(round::RoundingMode::RTZ, num_bits_rounded, sign, half));
    assert_eq(
        (u1:0, u5:0), round::round_sm(round::RoundingMode::RTZ, num_bits_rounded, sign, all_ones));

    let rtn_expected_overflow = if sign == round::Sign::Negative { u1:1 } else { u1:0 };
    assert_eq((u1:0, u5:0), round::round_sm(round::RoundingMode::RTN, num_bits_rounded, sign, zero));
    assert_eq(
        (rtn_expected_overflow, u5:0),
        round::round_sm(round::RoundingMode::RTN, num_bits_rounded, sign, half));
    assert_eq(
        (rtn_expected_overflow, u5:0),
        round::round_sm(round::RoundingMode::RTN, num_bits_rounded, sign, all_ones));

    let rtp_expected_overflow = if sign == round::Sign::NonNegative { u1:1 } else { u1:0 };
    assert_eq((u1:0, u5:0), round::round_sm(round::RoundingMode::RTP, num_bits_rounded, sign, zero));
    assert_eq(
        (rtp_expected_overflow, u5:0),
        round::round_sm(round::RoundingMode::RTP, num_bits_rounded, sign, half));
    assert_eq(
        (rtp_expected_overflow, u5:0),
        round::round_sm(round::RoundingMode::RTP, num_bits_rounded, sign, all_ones));
}

#[test]
fn test_round_more_fractional_bits_unsigned() {
    exercise_round_u_more_fractional_bits(u3:6);
    exercise_round_u_more_fractional_bits(u3:7);
}

#[test]
fn test_round_more_fractional_bits_signed() {
    exercise_round_s_more_fractional_bits(u3:6);
    exercise_round_s_more_fractional_bits(u3:7);
}

#[test]
fn test_round_more_fractional_bits_sign_magnitude() {
    exercise_round_sm_more_fractional_bits(u3:6, round::Sign::NonNegative);
    exercise_round_sm_more_fractional_bits(u3:7, round::Sign::NonNegative);
    exercise_round_sm_more_fractional_bits(u3:6, round::Sign::Negative);
    exercise_round_sm_more_fractional_bits(u3:7, round::Sign::Negative);
}

// Test specific behaviors below.

#[test]
fn test_round_unsigned_minimal_cases() {
    // num_bits_rounded = 0 is a no-op (early return path).
    assert_eq(
        (u1:0, u5:0b10101),
        round::round_u(round::RoundingMode::RTP, u3:0, u5:0b10101));

    let num_bits_rounded = u3:2;

    // Real value is 2.5 (10 / 2^2). Tie between 2 and 3.
    let unrounded = u5:0b01010;
    assert_eq(
        (u1:0, u5:0b01000),
        round::round_u(round::RoundingMode::RNE, num_bits_rounded, unrounded));
    assert_eq(
        (u1:0, u5:0b01100),
        round::round_u(round::RoundingMode::RNA, num_bits_rounded, unrounded));

    // Real value is 2.75 (11 / 2^2). Non-tie; nearest is 3.
    let unrounded = u5:0b01011;
    assert_eq(
        (u1:0, u5:0b01100),
        round::round_u(round::RoundingMode::RNE, num_bits_rounded, unrounded));

    // Real value is 7.75 (31 / 2^2). RTP would round to 8.0, which overflows u5 with 2
    // fractional bits.
    let unrounded = u5:0b11111;
    assert_eq(
        (u1:1, u5:0),
        round::round_u(round::RoundingMode::RTP, num_bits_rounded, unrounded));
}

#[test]
fn test_round_sign_magnitude_negative_floor_vs_ceil() {
    let num_bits_rounded = u3:2;
    let magnitude = u5:0b01001;  // Real magnitude is 2.25 (9 / 2^2)

    // Floor(-2.25) = -3.0
    assert_eq(
        (u1:0, u5:0b01100),
        round::round_sm(
            round::RoundingMode::RTN, num_bits_rounded, round::Sign::Negative, magnitude));

    // Ceil(-2.25) = -2.0
    assert_eq(
        (u1:0, u5:0b01000),
        round::round_sm(
            round::RoundingMode::RTP, num_bits_rounded, round::Sign::Negative, magnitude));
}

#[test]
fn test_round_signed_twos_complement_negative_cases() {
    let num_bits_rounded = u3:2;

    // Real value is -2.25 (-9 / 2^2).
    let unrounded = s5:-9;

    // Nearest(-2.25) = -2
    assert_eq(
        (u1:0, s5:-8),
        round::round_s(round::RoundingMode::RNE, num_bits_rounded, unrounded));

    // RTZ(-2.25) = -2
    assert_eq(
        (u1:0, s5:-8),
        round::round_s(round::RoundingMode::RTZ, num_bits_rounded, unrounded));

    // Floor(-2.25) = -3
    assert_eq(
        (u1:0, s5:-12),
        round::round_s(round::RoundingMode::RTN, num_bits_rounded, unrounded));
}

#[test]
fn test_round_signed_twos_complement_positive_overflow_sign_change() {
    // 0.9375 = 15 / 2^4; rounding to 1.0 cannot be represented in s5 with 4 fractional bits.
    let num_bits_rounded = u3:4;
    let unrounded = s5:15;
    assert_eq(
        (u1:1, s5:0),
        round::round_s(round::RoundingMode::RNE, num_bits_rounded, unrounded));
}

#[test]
fn test_round_signed_twos_complement_rounding_all_bits_negative_overflow() {
    // When every bit is fractional (num_bits_rounded == N), some modes would return -1 for a
    // negative tie like -0.5, which is not representable without any integer bits.
    let num_bits_rounded = u3:5;
    let unrounded = s5:-16;  // -0.5 = -16 / 2^5

    assert_eq(
        (u1:1, s5:0),
        round::round_s(round::RoundingMode::RTN, num_bits_rounded, unrounded));
    assert_eq(
        (u1:1, s5:0),
        round::round_s(round::RoundingMode::RNA, num_bits_rounded, unrounded));
}
