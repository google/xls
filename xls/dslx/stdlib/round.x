#![feature(type_inference_v2)]

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

// Implements rounding for all rounding modes defined by the IEEE 754 standard.
//
// It handles unsigned, signed (two's complement), and sign-and-magnitude values.
// It handles all 5 IEEE 754 rounding modes.
// Versions with compile-time 'num bits rounded' argument are provided, truncating the rounded-away
// bits.
// Versions with runtime 'num bits rounded' argument are provided, returning the full-width rounded
// result, with the rounded-away bits zeroed.
//
// Note: XLS prunes unused specializations. If callers pass compile-time constants for
// `num_bits_rounded` or restrict `RoundingMode` via an adapter, the optimizer keeps only the
// cases that remain reachable. (E.g. wrap this API with your own enum of three rounding modes
// and convert to `RoundingMode`, then the others will fold away.)
//
// Let's say you only want 3 rounding modes. Your public API should be a new enum with only
// those 3 rounding modes, and a function that takes that enum, and translates it to the
// RoundingMode enum of this library. XLS optimizer will remove the unused rounding modes from
// the optimized code.

import std;

// Rounding modes defined by the IEEE 754 standard.
//
// Note that the first two (RNE, RNA) always round to the nearest value, and when two potential
// results are equally close, a tie-breaking rule is applied. The last three (RTZ, RTN, RTP)
// first establish a direction on the Extended Real number line, and then round to the nearest
// value in that direction. If there is a closer value in the opposite direction, it is never
// returned.
//
// On naming: those that round to the nearest value begin with "RN", and those that always round
// in the same direction begin with "RT".
pub enum RoundingMode : u3 {
    // Round to Nearest, ties to Even. Of the two equally-close rounded results, the 'even'
    // result's lsb is 0.
    RNE = 0,
    // Rounds to Nearest, ties Away from zero. Of the two equally-close rounded results, the
    // value with the larger magnitude is returned.
    RNA = 1,
    // Round Toward Zero (i.e. floor(x) when x >= 0, or ceil(x) when x < 0)
    RTZ = 2,
    // Round Toward Negative infinity (i.e. floor(x))
    RTN = 3,
    // Round Toward Positive infinity (i.e. ceil(x))
    RTP = 4,
}

// Indicates a positive (more precisely: non-negative) or negative number.
pub enum Sign : u1 {
    // positive or zero
    NonNegative = 0,
    // negative
    Negative = 1,
}

// Rounds off the `num_bits_rounded` least significant bits. Returns (overflow, rounded result).
//
// Works for:
// - unsigned integers
//   - `sign` must be `NonNegative` (otherwise you have the sign-and-magnitude case, see below)
// - signed (two's complement) integers
//   - `sign` is ignored. The most significant bit of `unrounded` is used to determine the sign.
// - sign and magnitude values
//   - `sign` must be `Negative` when the represented value is a negative number, otherwise
//     `sign` must be `NonNegative`
//   - rounding may produce a zero magnitude from a negative input; callers must decide whether
//     to keep or flip the sign in that case
//
// The `num_bits_rounded` lsbs of the rounded result will always be 0.
//
// Users should interpret `unrounded` as a fixed-point quantity with num_bits_rounded fractional
// bits, being rounded to an integer. For unsigned inputs the corresponding Real value is
// unrounded / 2^num_bits_rounded, and the IEEE-754 rounding modes apply directly to that Real
// number. This viewpoint also explains the RNE tie case when every retained bit is discarded:
// the surviving integer portion is zero (an even value), so ties resolve towards zero.
//
// Overflow is 1 when the Real rounded result isn't a representable result (because the increase
// in magnitude requires a wider result type). Some non-exhaustive examples of when that can
// occur:
//  * RNE(3.5) = 4 -> overflow when round(RNE, 2, NonNegative, u4:0b11_10)
//  * RTN(-1.0625) = -2 -> overflow when round(RTN, 4, Negative, u5:0b1_0001)
//  * RTN(-0.03125) = -1 -> overflow when round(RTN, 5, NonNegative, s5:0b11111)
// The rounded result is 0 when overflow is 1.
//
// When num_bits_rounded > N, all source bits are treated as fractional. The rounded integer is 0
// unless the rounding mode requires +/-1, in which case overflow is signaled and 0 is returned.
//
// As mentioned above, during a tie, RNE looks at the least significant retained bit to
// determine round up or down. When there are no retained bits (i.e. num_bits_rounded >= N),
// round down is chosen. E.g.
// round(RNE, 4 bits, unsigned, u5:0b1_1000) -> rounds up (retained msb is 1)
// round(RNE, 4 bits, unsigned, u4:0b1000) -> rounds down (no retained bits)
pub fn round
    <S: bool, N: u32, W_NBR: u32 = {std::clog2(N + u32:1)}, NP1: u32 = {N + u32:1},
     W_SAFE: u32 = {std::max(N, u32:1)}>
    (rounding_mode: RoundingMode, num_bits_rounded: uN[W_NBR], sign: Sign, unrounded: xN[S][N])
    -> (u1, xN[S][N]) {
    // Compute sign bit while avoiding issues when N is zero.
    let unrounded_u = unrounded as uN[N];
    let sign_shift = std::usub_or_zero(N, u32:1) as uN[W_NBR];
    type SafeWord = uN[W_SAFE];
    let unrounded_sign_bit = std::lsb((unrounded_u as SafeWord) >> sign_shift);

    // determine sign when unrounded is two's complement
    let sign = if S {
        if unrounded_sign_bit == u1:1 { Sign::Negative } else { Sign::NonNegative }
    } else {
        sign
    };

    if N == u32:0 {
        (u1:0, xN[S][N]:0)
    } else if num_bits_rounded == uN[W_NBR]:0 {
        (u1:0, unrounded)
    } else if num_bits_rounded as u32 > N {
        let is_zero = unrounded_u == uN[N]:0;
        let is_strictly_negative = !is_zero && sign == Sign::Negative;
        let is_strictly_positive = !is_zero && sign == Sign::NonNegative;
        let overflow = match rounding_mode {
            RoundingMode::RTN => if is_strictly_negative { u1:1 } else { u1:0 },
            RoundingMode::RTP => if is_strictly_positive { u1:1 } else { u1:0 },
            _ => u1:0,
        };
        (overflow, xN[S][N]:0)
    } else {
        let negative_twos_complement = S && sign == Sign::Negative;

        // The bits rounded away; these bits are always 0 in the result.
        let rounded_bits = std::keep_lsbs(unrounded_u, num_bits_rounded);
        let rounded_bits_safe = rounded_bits as SafeWord;

        // The bits that will be returned, before any rounding adjustment.
        let retained_bits = std::clear_lsbs(unrounded_u, num_bits_rounded);

        // Note: zero retained bits means retained_bits_are_odd is false.
        let retained_bits_are_odd = std::lsb(retained_bits >> num_bits_rounded);

        let rounded_bits_are_nonzero = rounded_bits_safe != SafeWord:0;

        // This is the value of 0.5 when num_bits_rounded is interpreted as a negative binary
        // exponent (and by implication, `unrounded` is a binary fixed point value). We are
        // rounding the fixed point value to a nearby integer. This is 0.5 in this fixed point
        // format.
        let half_value = (SafeWord:1) << (num_bits_rounded as SafeWord - SafeWord:1);

        // as we defined half above, we use a similar definition of one
        let one = (uN[NP1]:1) << num_bits_rounded;
        let zero = uN[NP1]:0;

        // Beware rounded_gt_half when unrounded is two's complement and negative; it's
        // misleading.
        let rounded_gt_half = rounded_bits_safe > half_value;
        let rounded_eq_half = rounded_bits_safe == half_value;

        let adjustment = match rounding_mode {
            RoundingMode::RNE => {
                // round to nearest, ties to even
                // when |rounded_bits| > |half| or (|rounded_bits| == |half| and the retained bits
                // are odd)
                // the adjustment is:
                // unsigned -> 1
                // sign & magnitude, positive value -> 1
                // sign & magnitude, negative value -> 1
                // two's complement, positive value -> 1
                // two's complement, negative value is more complex, see below
                let tie_to_even = rounded_eq_half && retained_bits_are_odd;
                if negative_twos_complement {
                    // recall that rounded > 0.5 means the (negative two's complement) value is
                    // closer to 0 than half. E.g. -4 + 0.75 = -3.25 is closer to 0 than -3.5 is.
                    let closer_to_zero_than_half_is = rounded_gt_half;
                    if closer_to_zero_than_half_is || tie_to_even {
                        // RNE(-3.25) -> -3, retained=-4, thus adjustment=1
                        // RNE(-2.5) -> -2, retained=-3, thus adjustment=+1
                        one
                    } else {
                        // case: further from 0 than half is (e.g. -4 + 0.25 = -3.75 which is
                        // further from 0 than -3.5 is) OR rounded=0.5 and retained bits are even.
                        // RNE(-3.75) -> -4, retained=-4, thus adjustment=0
                        // RNE(-3.5) -> -4, retained=-4, thus adjustment=0
                        zero
                    }
                } else {
                    if rounded_gt_half || tie_to_even { one } else { zero }
                }
            },
            RoundingMode::RNA => {
                // round to nearest, ties away from zero
                // when |rounded_bits| >= |half| the adjustment is:
                // unsigned -> 1
                // sign & magnitude, positive value -> 1
                // sign & magnitude, negative value -> 1
                // two's complement, positive value -> 1
                // two's complement, negative value -> 0 (because truncation is toward -∞)
                //
                // you'll notice that RNE and RNA are the same w.r.t. the adjustment, and only
                // differ in the case of a tie (they agree when |rounded_bits| > |half|)
                if negative_twos_complement {
                    // recall that rounded > 0.5 means the (negative two's complement) value is
                    // closer to 0 than half. E.g. -4 + 0.75 = -3.25 is closer to 0 than -3.5 is.
                    let closer_to_zero_than_half_is = rounded_gt_half;
                    if closer_to_zero_than_half_is {
                        // RNA(-3.25) -> -3, retained=-4, thus adjustment=1
                        one
                    } else {
                        // RNA(-3.5) -> -4, retained=-4, thus adjustment=0
                        // RNA(-3.75) -> -4, retained=-4, thus adjustment=0
                        zero
                    }
                } else {
                    // unsigned or sign-magnitude and positive two's-complement
                    if rounded_gt_half || rounded_eq_half { one } else { zero }
                }
            },
            RoundingMode::RTZ => {
                // round toward zero
                // when rounded_bits != zero, the adjustment is:
                // unsigned -> 0
                // sign & magnitude, positive value -> 0
                // sign & magnitude, negative value -> 0
                // two's complement, positive value -> 0
                // two's complement, negative value -> 1
                if negative_twos_complement && rounded_bits_are_nonzero { one } else { zero }
            },
            RoundingMode::RTN => {
                // round toward negative infinity
                // when rounded_bits != zero, the adjustment is:
                // unsigned -> 0
                // sign & magnitude, positive value -> 0
                // sign & magnitude, negative value -> 1
                // two's complement, positive value -> 0
                // two's complement, negative value -> 0 (because truncation is toward -∞)
                if rounded_bits_are_nonzero {
                    match (S, sign) {
                        (false, Sign::Negative) => one,
                        _ => zero,
                    }
                } else {
                    zero
                }
            },
            RoundingMode::RTP => {
                // round toward positive infinity
                // when rounded_bits != zero, the adjustment is:
                // unsigned -> 1
                // sign & magnitude, positive value -> 1
                // sign & magnitude, negative value -> 0
                // two's complement, positive value -> 1
                // two's complement, negative value -> 1
                if rounded_bits_are_nonzero {
                    match (S, sign) {
                        (false, Sign::Negative) => zero,
                        _ => one,
                    }
                } else {
                    zero
                }
            },
        };

        let sum = retained_bits as uN[NP1] + adjustment;
        let (carry, rounded_u) = std::split_msbs<u32:1>(sum);

        let rounded_sign_bit = std::lsb((rounded_u as SafeWord) >> sign_shift);
        let sign_changed = S && rounded_sign_bit != unrounded_sign_bit;
        let adjustment_is_one = adjustment == one;
        let rounding_all_bits = (num_bits_rounded as u32) == N;
        let overflow = if !S {
            // Unsigned or sign-and-magnitude: any carry-out indicates overflow.
            carry
        } else {
            match (sign, adjustment_is_one) {
                // Positive argument, no adjustment - never overflows.
                (Sign::NonNegative, false) => false,
                (Sign::NonNegative, true) => {
                    // Positive argument, adjustment of +1.
                    //   - If we rounded away every bit, rely on the carry-out.
                    //   - If we kept at least one integer bit, check for sign change.
                    if rounding_all_bits { carry } else { sign_changed }
                },
                // Negative argument with +1 adjustment never overflows (-1 -> 0 etc.).
                (Sign::Negative, true) => false,
                (Sign::Negative, false) => {
                    // Negative argument with no adjustment.
                    // When every bit is rounded away the result is 0 (RTZ/RNE/RTP)
                    // or -1 (RTN/RNA). The latter is not representable because no integer bits
                    // remain.
                    if rounding_all_bits &&
                    (rounding_mode == RoundingMode::RTN || rounding_mode == RoundingMode::RNA) {
                        true
                    } else {
                        false
                    }
                },
            }
        };

        let rounded_u = if sign_changed {
            // handles cases like:
            // argument is two's complement, positive, and rounding away all bits
            // RNE(0.9375) = 1 -> overflow
            // round(RoundingMode::RNE, 4, NonNegative, s5:0b0_1111))
            // without this correction, result would be s5:0b1_0000, i.e. -1
            uN[N]:0
        } else {
            rounded_u
        };
        (overflow, rounded_u as xN[S][N])
    }
}

// Rounds an unsigned integer:
//  - rounds a runtime-specified number (`num_bits_rounded`) of least significant bits,
//  - returns the full-width rounded result with the least significant `num_bits_rounded` bits
//    zeroed.
// Returns (overflow, rounded result).
pub fn round_u<N: u32, W_NBR: u32 = {std::clog2(N + u32:1)}>
    (rounding_mode: RoundingMode, num_bits_rounded: uN[W_NBR], unrounded: uN[N]) -> (u1, uN[N]) {
    round(rounding_mode, num_bits_rounded, Sign::NonNegative, unrounded)
}

// Rounds an unsigned integer:
// - rounds a compile-time-constant (`num_bits_rounded`) number of least significant bits,
// - returns only the most significant bits (i.e., the rounded result), discarding the rounded-off
//   bits.
// Returns (overflow, rounded result).
pub fn round_trunc_u
    <NumBitsRounded: u32, N: u32, R: u32 = {N - NumBitsRounded},
     W_NBR: u32 = {std::clog2(N + u32:1)}>
    (rounding_mode: RoundingMode, unrounded: uN[N]) -> (u1, uN[R]) {
    const_assert!(NumBitsRounded <= N);
    let (overflow, rounded) = round_u(rounding_mode, NumBitsRounded as uN[W_NBR], unrounded);
    let (rounded_msbs, _) = std::split_msbs<R>(rounded);
    (overflow, rounded_msbs)
}

// Rounds an unsigned integer:
// - such that after rounding it is `AtMost` (a compile-time constant) bits wide,
// - returns only the most significant bits (i.e., the rounded result), discarding the rounded-off
//   bits.
// Returns (overflow, rounded result).
pub fn round_trunc_to_u
    <AtMost: u32, N: u32, R: u32 = {std::min(AtMost, N)},
     NumBitsRounded: u32 = {std::usub_or_zero(N, R)}, W_NBR: u32 = {std::clog2(N + u32:1)}>
    (rounding_mode: RoundingMode, unrounded: uN[N]) -> (u1, uN[R]) {
    if NumBitsRounded == u32:0 {
        // This no-op cast is required by the type checker. When this branch is not taken, this
        // cast op unifies the types of the branches.
        let unrounded = unrounded as uN[R];
        (u1:0, unrounded)
    } else {
        round_trunc_u<NumBitsRounded>(rounding_mode, unrounded)
    }
}

// Rounds a signed integer:
//  - rounds a runtime-specified number (`num_bits_rounded`) of least significant bits,
//  - returns the full-width rounded result with the least significant `num_bits_rounded` bits
//    zeroed.
// Returns (overflow, rounded result).
pub fn round_s<N: u32, W_NBR: u32 = {std::clog2(N + u32:1)}>
    (rounding_mode: RoundingMode, num_bits_rounded: uN[W_NBR], unrounded: sN[N]) -> (u1, sN[N]) {
    round(rounding_mode, num_bits_rounded, Sign::NonNegative, unrounded)
}

// Rounds a signed integer:
// - rounds a compile-time-constant (`num_bits_rounded`) number of least significant bits,
// - returns only the most significant bits (i.e., the rounded result), discarding the rounded-off
//   bits.
// Returns (overflow, rounded result).
pub fn round_trunc_s
    <num_bits_rounded: u32, N: u32, R: u32 = {N - num_bits_rounded},
     W_NBR: u32 = {std::clog2(N + u32:1)}>
    (rounding_mode: RoundingMode, unrounded: sN[N]) -> (u1, sN[R]) {
    const_assert!(num_bits_rounded <= N);
    if R == u32:0 {
        let (overflow, _) = round_s(rounding_mode, num_bits_rounded as uN[W_NBR], unrounded);
        (overflow, zero!<sN[R]>())
    } else {
        let (overflow, rounded) = round_s(rounding_mode, num_bits_rounded as uN[W_NBR], unrounded);
        let (rounded_msbs, _) = std::split_msbs<R>(rounded as uN[N]);
        (overflow, rounded_msbs as sN[R])
    }
}

// Rounds a signed integer:
// - such that after rounding it is `AtMost` (a compile-time constant) bits wide,
// - returns only the most significant bits (i.e., the rounded result), discarding the rounded-off
//   bits.
// Returns (overflow, rounded result).
pub fn round_trunc_to_s
    <AtMost: u32, N: u32, R: u32 = {std::min(AtMost, N)},
     NumBitsRounded: u32 = {std::usub_or_zero(N, R)}, W_NBR: u32 = {std::clog2(N + u32:1)}>
    (rounding_mode: RoundingMode, unrounded: sN[N]) -> (u1, sN[R]) {
    if NumBitsRounded == u32:0 {
        let unrounded = unrounded as sN[R];
        (u1:0, unrounded)
    } else {
        round_trunc_s<NumBitsRounded>(rounding_mode, unrounded)
    }
}

// Rounds a sign-and-magnitude integer:
//  - rounds a runtime-specified number (`num_bits_rounded`) of least significant bits,
//  - returns the full-width rounded result with the least significant `num_bits_rounded` bits
//    zeroed.
// Returns (overflow, rounded result).
pub fn round_sm<N: u32, W_NBR: u32 = {std::clog2(N + u32:1)}>
    (rounding_mode: RoundingMode, num_bits_rounded: uN[W_NBR], sign: Sign, magnitude: uN[N])
    -> (u1, uN[N]) {
    round(rounding_mode, num_bits_rounded, sign, magnitude)
}

// Rounds a sign-and-magnitude integer:
// - rounds a compile-time-constant (`num_bits_rounded`) number of least significant bits,
// - returns only the most significant bits (i.e., the rounded result), discarding the rounded-off
//   bits.
// Returns (overflow, rounded result).
pub fn round_trunc_sm
    <num_bits_rounded: u32, N: u32, R: u32 = {N - num_bits_rounded},
     W_NBR: u32 = {std::clog2(N + u32:1)}>
    (rounding_mode: RoundingMode, sign: Sign, magnitude: uN[N]) -> (u1, uN[R]) {
    const_assert!(num_bits_rounded <= N);
    let (overflow, rounded) =
        round_sm(rounding_mode, num_bits_rounded as uN[W_NBR], sign, magnitude);
    let (rounded_msbs, _) = std::split_msbs<R>(rounded);
    (overflow, rounded_msbs)
}

// Rounds a sign-and-magnitude integer:
// - such that after rounding it is `AtMost` (a compile-time constant) bits wide,
// - returns only the most significant bits (i.e., the rounded result), discarding the rounded-off
//   bits.
// Returns (overflow, rounded result).
pub fn round_trunc_to_sm
    <AtMost: u32, N: u32, R: u32 = {std::min(AtMost, N)},
     NumBitsRounded: u32 = {std::usub_or_zero(N, R)}, W_NBR: u32 = {std::clog2(N + u32:1)}>
    (rounding_mode: RoundingMode, sign: Sign, magnitude: uN[N]) -> (u1, uN[R]) {
    if NumBitsRounded == u32:0 {
        let magnitude = magnitude as uN[R];
        (u1:0, magnitude)
    } else {
        round_trunc_sm<NumBitsRounded>(rounding_mode, sign, magnitude)
    }
}
