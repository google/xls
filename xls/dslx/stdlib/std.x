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

// DSLX standard library routines.

pub fn sizeof<S: bool, N: u32>(x: xN[S][N]) -> u32 { N }

#[test]
fn sizeof_signed_test() {
    assert_eq(u32:0, sizeof(sN[0]:0));
    assert_eq(u32:1, sizeof(sN[1]:0));
    assert_eq(u32:2, sizeof(sN[2]:0));

    //TODO(tedhong): 2023-03-15 - update frontend to support below.
    //assert_eq(u32:0xffffffff, sizeof(uN[0xffffffff]:0));
}

#[test]
fn sizeof_unsigned_test() {
    assert_eq(u32:0, sizeof(uN[0]:0));
    assert_eq(u32:1, sizeof(uN[1]:0));
    assert_eq(u32:2, sizeof(uN[2]:0));

    //TODO(tedhong): 2023-03-15 - update frontend to support below.
    //assert_eq(u32:0xffffffff, sizeof(uN[0xffffffff]:0));
}

#[test]
fn use_sizeof_test() {
    let x = uN[32]:0xffffffff;
    let y: uN[sizeof(x) + u32:2] = x as uN[sizeof(x) + u32:2];
    assert_eq(y, uN[34]:0xffffffff);
}

// Returns the maximum signed value contained in N bits.
pub fn signed_max_value<N: u32, N_MINUS_ONE: u32 = {N - u32:1}>() -> sN[N] {
    ((sN[N]:1 << N_MINUS_ONE) - sN[N]:1) as sN[N]
}

#[test]
fn signed_max_value_test() {
    assert_eq(s8:0x7f, signed_max_value<u32:8>());
    assert_eq(s16:0x7fff, signed_max_value<u32:16>());
    assert_eq(s17:0xffff, signed_max_value<u32:17>());
    assert_eq(s32:0x7fffffff, signed_max_value<u32:32>());
}

// Returns the minimum signed value contained in N bits.
pub fn signed_min_value<N: u32, N_MINUS_ONE: u32 = {N - u32:1}>() -> sN[N] {
    (uN[N]:1 << N_MINUS_ONE) as sN[N]
}

#[test]
fn signed_min_value_test() {
    assert_eq(s8:-128, signed_min_value<u32:8>());
    assert_eq(s16:-32768, signed_min_value<u32:16>());
    assert_eq(s17:-65536, signed_min_value<u32:17>());
}

pub fn unsigned_min_value<N: u32>() -> uN[N] { zero!<uN[N]>() }

#[test]
fn unsigned_min_value_test() {
    assert_eq(u8:0, unsigned_min_value<u32:8>());
    assert_eq(u16:0, unsigned_min_value<u32:16>());
    assert_eq(u17:0, unsigned_min_value<u32:17>());
}

// Returns the maximum unsigned value contained in N bits.
pub fn unsigned_max_value<N: u32, N_PLUS_ONE: u32 = {N + u32:1}>() -> uN[N] {
    ((uN[N_PLUS_ONE]:1 << N) - uN[N_PLUS_ONE]:1) as uN[N]
}

#[test]
fn unsigned_max_value_test() {
    assert_eq(u8:0xff, unsigned_max_value<u32:8>());
    assert_eq(u16:0xffff, unsigned_max_value<u32:16>());
    assert_eq(u17:0x1ffff, unsigned_max_value<u32:17>());
    assert_eq(u32:0xffffffff, unsigned_max_value<u32:32>());
}

// Returns the maximum of two (signed or unsigned) integers.
pub fn max<S: bool, N: u32>(x: xN[S][N], y: xN[S][N]) -> xN[S][N] { if x > y { x } else { y } }

#[test]
fn max_test_signed() {
    assert_eq(s2:0, max(s2:0, s2:0));
    assert_eq(s2:1, max(s2:-1, s2:1));
    assert_eq(s7:-3, max(s7:-3, s7:-6));
}

#[test]
fn max_test_unsigned() {
    assert_eq(u1:1, max(u1:1, u1:0));
    assert_eq(u1:1, max(u1:1, u1:1));
    assert_eq(u2:3, max(u2:3, u2:2));
}

// Returns the minimum of two (signed or unsigned) integers.
pub fn min<S: bool, N: u32>(x: xN[S][N], y: xN[S][N]) -> xN[S][N] { if x < y { x } else { y } }

#[test]
fn min_test_unsigned() {
    assert_eq(u1:0, min(u1:1, u1:0));
    assert_eq(u1:1, min(u1:1, u1:1));
    assert_eq(u2:2, min(u2:3, u2:2));
}

#[test]
fn min_test_signed() {
    assert_eq(s1:0, min(s1:0, s1:0));
    assert_eq(s1:-1, min(s1:0, s1:-1));
    assert_eq(s1:-1, min(s1:-1, s1:0));
    assert_eq(s1:-1, min(s1:-1, s1:-1));

    assert_eq(s2:-2, min(s2:0, s2:-2));
    assert_eq(s2:-1, min(s2:0, s2:-1));
    assert_eq(s2:0, min(s2:0, s2:0));
    assert_eq(s2:0, min(s2:0, s2:1));

    assert_eq(s2:-2, min(s2:1, s2:-2));
    assert_eq(s2:-1, min(s2:1, s2:-1));
    assert_eq(s2:0, min(s2:1, s2:0));
    assert_eq(s2:1, min(s2:1, s2:1));

    assert_eq(s2:-2, min(s2:-2, s2:-2));
    assert_eq(s2:-2, min(s2:-2, s2:-1));
    assert_eq(s2:-2, min(s2:-2, s2:0));
    assert_eq(s2:-2, min(s2:-2, s2:1));

    assert_eq(s2:-2, min(s2:-1, s2:-2));
    assert_eq(s2:-1, min(s2:-1, s2:-1));
    assert_eq(s2:-1, min(s2:-1, s2:0));
    assert_eq(s2:-1, min(s2:-1, s2:1));
}

// Returns unsigned add of x (N bits) and y (M bits) as a max(N,M)+1 bit value.
pub fn uadd<N: u32, M: u32, R: u32 = {max(N, M) + u32:1}>(x: uN[N], y: uN[M]) -> uN[R] {
    (x as uN[R]) + (y as uN[R])
}

// Returns signed add of x (N bits) and y (M bits) as a max(N,M)+1 bit value.
pub fn sadd<N: u32, M: u32, R: u32 = {max(N, M) + u32:1}>(x: sN[N], y: sN[M]) -> sN[R] {
    (x as sN[R]) + (y as sN[R])
}

#[test]
fn uadd_test() {
    assert_eq(u4:4, uadd(u3:2, u3:2));
    assert_eq(u4:0b0100, uadd(u3:0b011, u3:0b001));
    assert_eq(u4:0b1110, uadd(u3:0b111, u3:0b111));
}

#[test]
fn sadd_test() {
    assert_eq(s4:4, sadd(s3:2, s3:2));
    assert_eq(s4:0, sadd(s3:1, s3:-1));
    assert_eq(s4:0b0100, sadd(s3:0b011, s2:0b01));
    assert_eq(s4:-8, sadd(s3:-4, s3:-4));
}

// Saturating subtraction: returns x - y, saturating at 0 if x < y.
pub fn usub_or_zero<N: u32>(x: uN[N], y: uN[N]) -> uN[N] { if x < y { uN[N]:0 } else { x - y } }

#[test]
fn usub_or_zero_test() {
    assert_eq(usub_or_zero(u8:0, u8:0), u8:0);
    assert_eq(usub_or_zero(u8:0, u8:1), u8:0);
    assert_eq(usub_or_zero(u8:0, u8:255), u8:0);

    assert_eq(usub_or_zero(u8:128, u8:0), u8:128);
    assert_eq(usub_or_zero(u8:128, u8:1), u8:127);
    assert_eq(usub_or_zero(u8:128, u8:128), u8:0);
    assert_eq(usub_or_zero(u8:128, u8:129), u8:0);
    assert_eq(usub_or_zero(u8:128, u8:255), u8:0);

    // Edge cases: max and min values
    assert_eq(usub_or_zero(u8:255, u8:255), u8:0);
    assert_eq(usub_or_zero(u8:255, u8:254), u8:1);
    assert_eq(usub_or_zero(u8:255, u8:0), u8:255);
    assert_eq(usub_or_zero(u8:1, u8:255), u8:0);
    assert_eq(usub_or_zero(u8:1, u8:2), u8:0);
    assert_eq(usub_or_zero(u8:1, u8:1), u8:0);
    assert_eq(usub_or_zero(u8:2, u8:1), u8:1);
    assert_eq(usub_or_zero(u8:2, u8:2), u8:0);
    assert_eq(usub_or_zero(u8:2, u8:3), u8:0);

    // 1-bit values
    assert_eq(usub_or_zero(u1:0, u1:0), u1:0);
    assert_eq(usub_or_zero(u1:1, u1:0), u1:1);
    assert_eq(usub_or_zero(u1:1, u1:1), u1:0);
    assert_eq(usub_or_zero(u1:0, u1:1), u1:0);

    // 4-bit values
    assert_eq(usub_or_zero(u4:0, u4:15), u4:0);
    assert_eq(usub_or_zero(u4:15, u4:0), u4:15);
    assert_eq(usub_or_zero(u4:8, u4:8), u4:0);
    assert_eq(usub_or_zero(u4:7, u4:8), u4:0);
    assert_eq(usub_or_zero(u4:8, u4:7), u4:1);

    // Large difference
    assert_eq(usub_or_zero(u16:0, u16:65535), u16:0);
    assert_eq(usub_or_zero(u16:65535, u16:0), u16:65535);
    assert_eq(usub_or_zero(u16:65535, u16:65535), u16:0);
    assert_eq(usub_or_zero(u16:1, u16:65535), u16:0);
    assert_eq(usub_or_zero(u16:65535, u16:1), u16:65534);
}

// Returns the value of x-1 with saturation at 0.
pub fn bounded_minus_1<N: u32>(x: uN[N]) -> uN[N] { usub_or_zero(x, uN[N]:1) }

// Returns unsigned mul of x (N bits) and y (M bits) as an N+M bit value.
pub fn umul<N: u32, M: u32, R: u32 = {N + M}>(x: uN[N], y: uN[M]) -> uN[R] {
    (x as uN[R]) * (y as uN[R])
}

// Returns signed mul of x (N bits) and y (M bits) as an N+M bit value.
pub fn smul<N: u32, M: u32, R: u32 = {N + M}>(x: sN[N], y: sN[M]) -> sN[R] {
    (x as sN[R]) * (y as sN[R])
}

#[test]
fn umul_test() {
    assert_eq(u6:4, umul(u3:2, u3:2));
    assert_eq(u4:0b0011, umul(u2:0b11, u2:0b01));
    assert_eq(u4:0b1001, umul(u2:0b11, u2:0b11));
}

#[test]
fn smul_test() {
    assert_eq(s6:4, smul(s3:2, s3:2));
    assert_eq(s4:0b1111, smul(s2:0b11, s2:0b01));
}

// Returns unsigned division of `n` (N bits) and `d` (M bits) as quotient (N bits) and remainder (M
// bits).
// If dividing by `0`, returns all `1`s for quotient and `n` for remainder.
// Implements binary long division; should be expected to use a large number of gates and have a
// slow critical path when using combinational codegen.
pub fn iterative_div_mod<N: u32, M: u32>(n: uN[N], d: uN[M]) -> (uN[N], uN[M]) {
    // Zero extend divisor by 1 bit.
    let divisor = d as uN[M + u32:1];

    for (i, (q, r)): (u32, (uN[N], uN[M])) in u32:0..N {
        // Shift the next bit of n into r.
        let r = r ++ n[(N - u32:1 - i)+:u1];
        let (q, r) = if r >= divisor {
            (q as uN[N - u32:1] ++ u1:1, r - divisor)
        } else {
            (q as uN[N - u32:1] ++ u1:0, r)
        };
        // Remove the MSB of r; guaranteed to be 0 because r < d.
        (q, r[0:M as s32])
    }((uN[N]:0, uN[M]:0))
}

// Returns unsigned division of `n` (N bits) and `d` (M bits) as quotient (N bits).
// If dividing by `0`, returns all `1`s for quotient.
pub fn iterative_div<N: u32, M: u32>(n: uN[N], d: uN[M]) -> uN[N] {
    let (q, r) = iterative_div_mod(n, d);
    q
}

#[test]
fn iterative_div_mod_test() {
    // Power of 2.
    assert_eq((u4:0, u4:8), iterative_div_mod(u4:8, u4:15));
    assert_eq((u4:1, u4:0), iterative_div_mod(u4:8, u4:8));
    assert_eq((u4:2, u4:0), iterative_div_mod(u4:8, u4:4));
    assert_eq((u4:4, u4:0), iterative_div_mod(u4:8, u4:2));
    assert_eq((u4:8, u4:0), iterative_div_mod(u4:8, u4:1));
    assert_eq((u4:8 / u4:0, u4:8), iterative_div_mod(u4:8, u4:0));
    assert_eq((u4:15, u4:8), iterative_div_mod(u4:8, u4:0));

    // Non-powers-of-2.
    assert_eq((u32:6, u32:0), iterative_div_mod(u32:18, u32:3));
    assert_eq((u32:6, u32:0), iterative_div_mod(u32:36, u32:6));
    assert_eq((u32:6, u32:0), iterative_div_mod(u32:48, u32:8));
    assert_eq((u32:20, u32:0), iterative_div_mod(u32:900, u32:45));

    // Results w/ remainder.
    assert_eq((u32:6, u32:2), iterative_div_mod(u32:20, u32:3));
    assert_eq((u32:6, u32:5), iterative_div_mod(u32:41, u32:6));
    assert_eq((u32:6, u32:7), iterative_div_mod(u32:55, u32:8));
    assert_eq((u32:20, u32:44), iterative_div_mod(u32:944, u32:45));

    // Arbitrary width.
    assert_eq((u5:6, u3:2), iterative_div_mod(u5:20, u3:3));
    assert_eq((u6:6, u4:5), iterative_div_mod(u6:41, u4:6));
    assert_eq((u6:6, u4:7), iterative_div_mod(u6:55, u4:8));
    assert_eq((u10:20, u6:44), iterative_div_mod(u10:944, u6:45));

    // Divide by 0.
    assert_eq((u4:0xf, u4:8), iterative_div_mod(u4:8, u4:0));
    assert_eq((u8:0xff, u8:64), iterative_div_mod(u8:64, u8:0));
    assert_eq((u1:1, u1:0), iterative_div_mod(u1:0, u1:0));
}

#[test]
fn iterative_div_test() {
    // Power of 2.
    assert_eq(u4:0, iterative_div(u4:8, u4:15));
    assert_eq(u4:1, iterative_div(u4:8, u4:8));
    assert_eq(u4:2, iterative_div(u4:8, u4:4));
    assert_eq(u4:4, iterative_div(u4:8, u4:2));
    assert_eq(u4:8, iterative_div(u4:8, u4:1));
    assert_eq(u4:8 / u4:0, iterative_div(u4:8, u4:0));
    assert_eq(u4:15, iterative_div(u4:8, u4:0));

    // Non-powers-of-2.
    assert_eq(u32:6, iterative_div(u32:18, u32:3));
    assert_eq(u32:6, iterative_div(u32:36, u32:6));
    assert_eq(u32:6, iterative_div(u32:48, u32:8));
    assert_eq(u32:20, iterative_div(u32:900, u32:45));

    // Results w/ remainder.
    assert_eq(u32:6, iterative_div(u32:20, u32:3));
    assert_eq(u32:6, iterative_div(u32:41, u32:6));
    assert_eq(u32:6, iterative_div(u32:55, u32:8));
    assert_eq(u32:20, iterative_div(u32:944, u32:45));

    // Divide by 0.
    assert_eq(u8:0xff, iterative_div(u8:64, u8:0));
}

// Extracts the LSb (least significant bit) from the value `x` and returns it.
pub fn lsb<S: bool, N: u32>(x: xN[S][N]) -> u1 { x as u1 }

#[test]
fn lsb_test() {
    // unsigned values
    assert_eq(u1:0, lsb(u2:0b00));
    assert_eq(u1:1, lsb(u2:0b01));
    assert_eq(u1:1, lsb(u2:0b11));
    assert_eq(u1:0, lsb(u2:0b10));
    assert_eq(u1:1, lsb(u2:0b11));

    // signed values
    assert_eq(u1:0, lsb(s2:0b00));
    assert_eq(u1:1, lsb(s2:0b01));
    assert_eq(u1:1, lsb(s2:0b11));
    assert_eq(u1:0, lsb(s2:0b10));
    assert_eq(u1:1, lsb(s2:0b11));
}

// Extracts the MSb (most significant bit) from the value `x` and returns it.
pub fn msb<S: bool, N: u32>(x: xN[S][N]) -> u1 { (x as uN[N])[-1:] }

#[test]
fn msb_test() {
    // unsigned values
    assert_eq(u1:0, msb(u2:0b00));
    assert_eq(u1:0, msb(u2:0b01));
    assert_eq(u1:1, msb(u2:0b11));
    assert_eq(u1:1, msb(u2:0b10));
    assert_eq(u1:1, msb(u2:0b11));

    // signed values
    assert_eq(u1:0, msb(s2:0b00));
    assert_eq(u1:0, msb(s2:0b01));
    assert_eq(u1:1, msb(s2:0b11));
    assert_eq(u1:1, msb(s2:0b10));
    assert_eq(u1:1, msb(s2:0b11));
}


// Returns the R LSbs of x.
// `x` must be at least R bits wide.
pub fn lsbs<R: u32, N: u32>(x: uN[N]) -> uN[R] {
  const_assert!(N >= R);
  x[0 +: uN[R]]
}

#[test]
fn lsbs_test() {
    assert_eq(lsbs<u32:1>(u2:1), u1:1);
    assert_eq(lsbs<u32:2>(u3:5), u2:1);
    assert_eq(lsbs<u32:3>(u5:30), u3:6);
}

// Returns the R MSbs of x.
// `x` must be at least R bits wide.
pub fn msbs<R: u32, N: u32>(x: uN[N]) -> uN[R] {
  const_assert!(N >= R);
  x[(N - R) +: uN[R]]
}

#[test]
fn msbs_test() {
    assert_eq(msbs<u32:1>(u2:1), u1:0);
    assert_eq(msbs<u32:2>(u3:5), u2:2);
    assert_eq(msbs<u32:3>(u5:30), u3:7);
}

// Splits bits into (N most significant bits, the remaining least significant
// bits).
//
// This function ensures that all bits of the argument are used.
pub fn split_msbs<N: u32, X: u32, Z: u32 = {X - N}, FROM_START: s32 = {Z as s32}>
    (x: bits[X]) -> (bits[N], bits[Z]) {
    // Can't split more bits than exist
    const_assert!(N <= X);

    let msbs = x[FROM_START:];
    let lsbs = x[0:FROM_START];
    (msbs, lsbs)
}

#[test]
fn test_split_msbs() {
    assert_eq(split_msbs<u32:0>(u5:0b10101), (uN[0]:0, u5:0b10101));
    assert_eq(split_msbs<u32:1>(u5:0b10101), (u1:0b1, u4:0b0101));
    assert_eq(split_msbs<u32:2>(u5:0b10101), (u2:0b10, u3:0b101));
    assert_eq(split_msbs<u32:3>(u5:0b10101), (u3:0b101, u2:0b01));
    assert_eq(split_msbs<u32:4>(u5:0b10101), (u4:0b1010, u1:0b1));
    assert_eq(split_msbs<u32:5>(u5:0b10101), (u5:0b10101, uN[0]:0));
}

#[quickcheck(exhaustive)]
fn prop_split_msbs(n: uN[4], o: uN[3]) -> bool {
    let (n2, o2) = split_msbs<4>(n ++ o);
    n == n2 && o == o2
}

// Splits bits into (the remaining most significant bits, N least significant
// bits).
//
// This function ensures that all bits of the argument are used.
pub fn split_lsbs<N: u32, X: u32, Y: u32 = {X - N}, FROM_START: s32 = {N as s32}>
    (x: bits[X]) -> (bits[Y], bits[N]) {
    // Can't split more bits than exist
    const_assert!(N <= X);

    let msbs = x[FROM_START:];
    let lsbs = x[0:FROM_START];
    (msbs, lsbs)
}

#[test]
fn test_split_lsbs() {
    assert_eq(split_lsbs<u32:5>(u5:0b10101), (uN[0]:0, u5:0b10101));
    assert_eq(split_lsbs<u32:4>(u5:0b10101), (u1:0b1, u4:0b0101));
    assert_eq(split_lsbs<u32:3>(u5:0b10101), (u2:0b10, u3:0b101));
    assert_eq(split_lsbs<u32:2>(u5:0b10101), (u3:0b101, u2:0b01));
    assert_eq(split_lsbs<u32:1>(u5:0b10101), (u4:0b1010, u1:0b1));
    assert_eq(split_lsbs<u32:0>(u5:0b10101), (u5:0b10101, uN[0]:0));
}

#[quickcheck(exhaustive)]
fn prop_split_lsbs(n: uN[4], o: uN[3]) -> bool {
    let (n2, o2) = split_lsbs<3>(n ++ o);
    n == n2 && o == o2
}

// Returns the absolute value of x as a signed number.
pub fn abs<BITS: u32>(x: sN[BITS]) -> sN[BITS] { if x < sN[BITS]:0 { -x } else { x } }

// Converts an array of N bools to a bits[N] value.
//
// The bool at index 0 in the array because the MSb (most significant bit) in
// the result.
pub fn convert_to_bits_msb0<N: u32>(x: bool[N]) -> uN[N] {
    for (i, accum): (u32, uN[N]) in u32:0..N {
        accum | (x[i] as uN[N]) << ((N - i - u32:1) as uN[N])
    }(uN[N]:0)
}

#[test]
fn convert_to_bits_msb0_test() {
    assert_eq(u3:0b000, convert_to_bits_msb0(bool[3]:[false, false, false]));
    assert_eq(u3:0b001, convert_to_bits_msb0(bool[3]:[false, false, true]));
    assert_eq(u3:0b010, convert_to_bits_msb0(bool[3]:[false, true, false]));
    assert_eq(u3:0b011, convert_to_bits_msb0(bool[3]:[false, true, true]));
    assert_eq(u3:0b100, convert_to_bits_msb0(bool[3]:[true, false, false]));
    assert_eq(u3:0b110, convert_to_bits_msb0(bool[3]:[true, true, false]));
    assert_eq(u3:0b111, convert_to_bits_msb0(bool[3]:[true, true, true]));
}

// Converts a bits[N] values to an array of N bools.
//
// This variant puts the LSb (least significant bit) of the word at index 0 in
// the resulting array.
pub fn convert_to_bools_lsb0<N: u32>(x: uN[N]) -> bool[N] {
    for (idx, partial): (u32, bool[N]) in u32:0..N {
        update(partial, idx, x[idx+:bool])
    }(bool[N]:[false, ...])
}

#[test]
fn convert_to_bools_lsb0_test() {
    assert_eq(convert_to_bools_lsb0(u1:1), bool[1]:[true]);
    assert_eq(convert_to_bools_lsb0(u2:0b01), bool[2]:[true, false]);
    assert_eq(convert_to_bools_lsb0(u2:0b10), bool[2]:[false, true]);
    assert_eq(convert_to_bools_lsb0(u3:0b000), bool[3]:[false, false, false]);
    assert_eq(convert_to_bools_lsb0(u3:0b001), bool[3]:[true, false, false]);
    assert_eq(convert_to_bools_lsb0(u3:0b010), bool[3]:[false, true, false]);
    assert_eq(convert_to_bools_lsb0(u3:0b011), bool[3]:[true, true, false]);
    assert_eq(convert_to_bools_lsb0(u3:0b100), bool[3]:[false, false, true]);
    assert_eq(convert_to_bools_lsb0(u3:0b110), bool[3]:[false, true, true]);
    assert_eq(convert_to_bools_lsb0(u3:0b111), bool[3]:[true, true, true]);
}

#[quickcheck]
fn convert_to_from_bools(x: u4) -> bool {
    convert_to_bits_msb0(array_rev(convert_to_bools_lsb0(x))) == x
}

// Returns (found, index) given array and the element to find within the array.
//
// Note that when found is false, the index is 0 -- 0 is provided instead of a
// value like -1 to prevent out-of-bounds accesses from occurring if the index
// is used in a match expression (which will eagerly evaluate all of its arms),
// to prevent it from creating an error at simulation time if the value is
// ultimately discarded from the unselected match arm.
pub fn find_index<BITS: u32, ELEMS: u32>(array: uN[BITS][ELEMS], x: uN[BITS]) -> (bool, u32) {
    // Compute all the positions that are equal to our target.
    let bools: bool[ELEMS] = for (i, accum): (u32, bool[ELEMS]) in u32:0..ELEMS {
        update(accum, i, array[i] == x)
    }((bool[ELEMS]:[false, ...]));

    let x: uN[ELEMS] = convert_to_bits_msb0(bools);
    let index = clz(x);
    let found: bool = or_reduce(x);
    (found, if found { index as u32 } else { u32:0 })
}

#[test]
fn find_index_test() {
    let haystack = u3[4]:[0b001, 0b010, 0b100, 0b111];
    assert_eq((true, u32:1), find_index(haystack, u3:0b010));
    assert_eq((true, u32:3), find_index(haystack, u3:0b111));
    assert_eq((false, u32:0), find_index(haystack, u3:0b000));
}

// Concatenates 3 values of arbitrary bitwidths to a single value.
pub fn concat3<X: u32, Y: u32, Z: u32, R: u32 = {X + Y + Z}>
    (x: bits[X], y: bits[Y], z: bits[Z]) -> bits[R] {
    x ++ y ++ z
}

#[test]
fn concat3_test() { assert_eq(u12:0b111000110010, concat3(u6:0b111000, u4:0b1100, u2:0b10)); }

// Returns the ceiling of (x divided by y).
pub fn ceil_div<N: u32>(x: uN[N], y: uN[N]) -> uN[N] {
    let usual = (x - uN[N]:1) / y + uN[N]:1;
    if x > uN[N]:0 { usual } else { uN[N]:0 }
}

#[test]
fn ceil_div_test() {
    assert_eq(ceil_div(u32:6, u32:2), u32:3);
    assert_eq(ceil_div(u32:5, u32:2), u32:3);
    assert_eq(ceil_div(u32:4, u32:2), u32:2);
    assert_eq(ceil_div(u32:3, u32:2), u32:2);
    assert_eq(ceil_div(u32:2, u32:2), u32:1);
    assert_eq(ceil_div(u32:1, u32:2), u32:1);
    assert_eq(ceil_div(u32:0, u32:2), u32:0);

    assert_eq(ceil_div(u8:6, u8:3), u8:2);
    assert_eq(ceil_div(u8:5, u8:3), u8:2);
    assert_eq(ceil_div(u8:4, u8:3), u8:2);
    assert_eq(ceil_div(u8:3, u8:3), u8:1);
    assert_eq(ceil_div(u8:2, u8:3), u8:1);
    assert_eq(ceil_div(u8:1, u8:3), u8:1);
    assert_eq(ceil_div(u8:0, u8:3), u8:0);
}

// Returns `x` rounded up to the nearest multiple of `y`.
pub fn round_up_to_nearest(x: u32, y: u32) -> u32 { (ceil_div(x, y) * y) as u32 }

#[test]
fn round_up_to_nearest_test() {
    assert_eq(u32:4, round_up_to_nearest(u32:3, u32:2));
    assert_eq(u32:4, round_up_to_nearest(u32:4, u32:2));
}

// Returns `x` rounded up to the nearest multiple of `y`, where `y` is a known positive power of 2.
// This functionality is the same as `round_up_to_nearest` but optimized when `y` is a power of 2.
pub fn round_up_to_nearest_pow2_unsigned<N: u32>(x: uN[N], y: uN[N]) -> uN[N] {
    (x + y - uN[N]:1) & -y
}

#[test]
fn test_round_up_to_nearest_pow2_unsigned() {
    assert_eq(round_up_to_nearest_pow2_unsigned(u16:0, u16:8), u16:0);
    assert_eq(round_up_to_nearest_pow2_unsigned(u16:1, u16:8), u16:8);
    assert_eq(round_up_to_nearest_pow2_unsigned(u16:7, u16:8), u16:8);
    assert_eq(round_up_to_nearest_pow2_unsigned(u16:8, u16:8), u16:8);
    assert_eq(round_up_to_nearest_pow2_unsigned(u16:9, u16:8), u16:16);

    assert_eq(round_up_to_nearest_pow2_unsigned(u16:9, u16:16), u16:16);
}

// Returns `x` rounded up to the nearest multiple of `y`, where `y` is a known positive power of 2.
// This functionality is the same as `round_up_to_nearest` but optimized when `y` is a power of 2.
pub fn round_up_to_nearest_pow2_signed<N: u32>(x: sN[N], y: uN[N]) -> sN[N] {
    (x + y as sN[N] - sN[N]:1) & -(y as sN[N])
}

#[test]
fn test_round_up_to_pow2_signed() {
    assert_eq(round_up_to_nearest_pow2_signed(s16:9, u16:8), s16:16);
    assert_eq(round_up_to_nearest_pow2_signed(s16:8, u16:8), s16:8);
    assert_eq(round_up_to_nearest_pow2_signed(s16:7, u16:8), s16:8);
    assert_eq(round_up_to_nearest_pow2_signed(s16:1, u16:8), s16:8);
    assert_eq(round_up_to_nearest_pow2_signed(s16:0, u16:8), s16:0);
    assert_eq(round_up_to_nearest_pow2_signed(s16:-1, u16:8), s16:0);
    assert_eq(round_up_to_nearest_pow2_signed(s16:-7, u16:8), s16:0);
    assert_eq(round_up_to_nearest_pow2_signed(s16:-8, u16:8), s16:-8);
    assert_eq(round_up_to_nearest_pow2_signed(s16:-9, u16:8), s16:-8);

    assert_eq(round_up_to_nearest_pow2_signed(s16:9, u16:16), s16:16);
    assert_eq(round_up_to_nearest_pow2_signed(s16:-9, u16:16), s16:0);
}

// Returns `floor(log2(x))`, with one exception:
//
// When x=0, this function differs from the true mathematical function:
// flog2(0) = 0
// floor(log2(0)) = -infinity
//
// This function is frequently used to calculate the number of bits required to
// represent an unsigned integer `n` to define flog2(0) = 0.
//
// Example: flog2(7) = 2, flog2(8) = 3.
pub fn flog2<N: u32>(x: bits[N]) -> bits[N] {
    if x >= bits[N]:1 { (N as bits[N]) - clz(x) - bits[N]:1 } else { bits[N]:0 }
}

#[test]
fn flog2_test() {
    assert_eq(u32:0, flog2(u32:0));
    assert_eq(u32:0, flog2(u32:1));
    assert_eq(u32:1, flog2(u32:2));
    assert_eq(u32:1, flog2(u32:3));
    assert_eq(u32:2, flog2(u32:4));
    assert_eq(u32:2, flog2(u32:5));
    assert_eq(u32:2, flog2(u32:6));
    assert_eq(u32:2, flog2(u32:7));
    assert_eq(u32:3, flog2(u32:8));
    assert_eq(u32:3, flog2(u32:9));
}

// Returns `ceiling(log2(x))`, with one exception:
//
// When x=0, this function differs from the true mathematical function:
// clog2(0) = 0
// ceiling(log2(0)) = -infinity
//
// This function is frequently used to calculate the number of bits required to
// represent `x` possibilities. With this interpretation, it is sensible
// to define clog2(0) = 0.
//
// Example: clog2(7) = 3
pub fn clog2<N: u32>(x: bits[N]) -> bits[N] {
    if x >= bits[N]:1 { (N as bits[N]) - clz(x - bits[N]:1) } else { bits[N]:0 }
}

#[test]
fn clog2_test() {
    assert_eq(u32:0, clog2(u32:0));
    assert_eq(u32:0, clog2(u32:1));
    assert_eq(u32:1, clog2(u32:2));
    assert_eq(u32:2, clog2(u32:3));
    assert_eq(u32:2, clog2(u32:4));
    assert_eq(u32:3, clog2(u32:5));
    assert_eq(u32:3, clog2(u32:6));
    assert_eq(u32:3, clog2(u32:7));
    assert_eq(u32:3, clog2(u32:8));
    assert_eq(u32:4, clog2(u32:9));
}

// Returns true when x is a non-zero power-of-two.
pub fn is_pow2<N: u32>(x: uN[N]) -> bool { x > uN[N]:0 && (x & (x - uN[N]:1) == uN[N]:0) }

#[test]
fn is_pow2_test() {
    assert_eq(is_pow2(u32:0), false);
    assert_eq(is_pow2(u32:1), true);
    assert_eq(is_pow2(u32:2), true);
    assert_eq(is_pow2(u32:3), false);
    assert_eq(is_pow2(u32:4), true);
    assert_eq(is_pow2(u32:5), false);
    assert_eq(is_pow2(u32:6), false);
    assert_eq(is_pow2(u32:7), false);
    assert_eq(is_pow2(u32:8), true);

    // Test parametric bitwidth.
    assert_eq(is_pow2(u8:0), false);
    assert_eq(is_pow2(u8:1), true);
    assert_eq(is_pow2(u8:2), true);
    assert_eq(is_pow2(u8:3), false);
    assert_eq(is_pow2(u8:4), true);
    assert_eq(is_pow2(u8:5), false);
    assert_eq(is_pow2(u8:6), false);
    assert_eq(is_pow2(u8:7), false);
    assert_eq(is_pow2(u8:8), true);
}

// Returns true when `x` is an even integer.
pub fn is_even<S: bool, N: u32>(x: xN[S][N]) -> bool { lsb(x as uN[N]) == u1:0 }

#[test]
fn is_even_test() {
    assert_eq(is_even(uN[0]:0), true);

    assert_eq(is_even(sN[0]:0), true);

    assert_eq(is_even(u1:0), true);
    assert_eq(is_even(u1:1), false);

    assert_eq(is_even(s1:0), true);
    assert_eq(is_even(s1:-1), false);

    assert_eq(is_even(u2:0), true);
    assert_eq(is_even(u2:1), false);
    assert_eq(is_even(u2:2), true);
    assert_eq(is_even(u2:3), false);

    assert_eq(is_even(s3:-4), true);
    assert_eq(is_even(s3:-3), false);
    assert_eq(is_even(s3:-2), true);
    assert_eq(is_even(s3:-1), false);
    assert_eq(is_even(s3:0), true);
    assert_eq(is_even(s3:1), false);
    assert_eq(is_even(s3:2), true);
    assert_eq(is_even(s3:3), false);
}

// Adjacent integers are always of opposite parity.
#[quickcheck(exhaustive)]
fn prop_is_even_adjacent_diff_unsigned(x: u4) -> bool {
    let even_x = is_even(x);
    let even_xp1 = is_even(x + u4:1);
    let even_xm1 = is_even(x - u4:1);
    even_x != even_xp1 && even_x != even_xm1
}

// Adjacent integers are always of opposite parity.
#[quickcheck(exhaustive)]
fn prop_is_even_adjacent_diff_signed(x: s4) -> bool {
    let even_x = is_even(x);
    let even_xp1 = is_even(x + s4:1);
    let even_xm1 = is_even(x - s4:1);
    even_x != even_xp1 && even_x != even_xm1
}

// Returns x % y where y must be a non-zero power-of-two.
pub fn mod_pow2<N: u32>(x: bits[N], y: bits[N]) -> bits[N] {
    // TODO(leary): 2020-06-11 Add assertion y is a power of two and non-zero.
    x & (y - bits[N]:1)
}

#[test]
fn mod_pow2_test() {
    assert_eq(u32:1, mod_pow2(u32:5, u32:4));
    assert_eq(u32:0, mod_pow2(u32:4, u32:4));
    assert_eq(u32:3, mod_pow2(u32:3, u32:4));
}

// Returns x / y where y must be a non-zero power-of-two.
pub fn div_pow2<N: u32>(x: bits[N], y: bits[N]) -> bits[N] {
    // TODO(leary): 2020-06-11 Add assertion y is a power of two and non-zero.
    x >> clog2(y)
}

#[test]
fn div_pow2_test() {
    assert_eq(u32:1, div_pow2(u32:5, u32:4));
    assert_eq(u32:1, div_pow2(u32:4, u32:4));
    assert_eq(u32:0, div_pow2(u32:3, u32:4));
}

// Returns a value with X bits set (of type bits[X]).
pub fn mask_bits<X: u32>() -> bits[X] { !bits[X]:0 }

#[test]
fn mask_bits_test() {
    assert_eq(u8:0xff, mask_bits<u32:8>());
    assert_eq(u13:0x1fff, mask_bits<u32:13>());
}

// The result of comparing two values.
pub enum Ordering : s2 {
    Less = -1,
    Equal = 0,
    Greater = 1,
}

fn compare_unsigned<N: u32>(lhs: uN[N], rhs: uN[N]) -> Ordering {
    // Zero-extend both to N+1 bits so that subtraction underflow sets the MSB to '1'
    let lhs_ext = lhs as uN[N + u32:1];
    let rhs_ext = rhs as uN[N + u32:1];

    // Subtract in N+1 bits; the top bit is 1 exactly when lhs<rhs
    let diff_ext: uN[N + u32:1] = lhs_ext - rhs_ext;
    let less = msb(diff_ext);

    let not_equal: u1 = lhs != rhs;

    //    Mapping:
    //      lhs<rhs  ⟹ less=1, not_equal=1 ⟹ bits = 11₂ ⟹ -1
    //      lhs=rhs  ⟹ less=0, not_equal=0 ⟹ bits = 00₂ ⟹  0
    //      lhs>rhs  ⟹ less=0, not_equal=1 ⟹ bits = 01₂ ⟹ +1

    let packed: u2 = less ++ not_equal;
    packed as Ordering
}

fn compare_signed<N: u32>(lhs: sN[N], rhs: sN[N]) -> Ordering {
    // Sign-extend both to N+1 bits so that subtraction overflow never occurs
    let lhs_sext = lhs as sN[N + u32:1];
    let rhs_sext = rhs as sN[N + u32:1];

    // Subtract in N+1 bits; the top bit is 1 exactly when lhs<rhs
    let diff_ext: sN[N + u32:1] = lhs_sext - rhs_sext;
    let less: u1 = msb(diff_ext);

    let not_equal: u1 = lhs != rhs;

    //    Mapping:
    //      lhs<rhs  ⟹ less=1, not_equal=1 ⟹ bits = 11₂ ⟹ -1
    //      lhs=rhs  ⟹ less=0, not_equal=0 ⟹ bits = 00₂ ⟹  0
    //      lhs>rhs  ⟹ less=0, not_equal=1 ⟹ bits = 01₂ ⟹ +1

    let packed: u2 = less ++ not_equal;
    packed as Ordering
}

#[quickcheck(exhaustive)]
fn prop_compare_unsigned_matches_corresponding_comparison(lhs: u4, rhs: u4) -> bool {
    match compare_unsigned(lhs, rhs) {
        Ordering::Less => lhs < rhs,
        Ordering::Equal => lhs == rhs,
        Ordering::Greater => lhs > rhs,
    }
}

#[quickcheck(exhaustive)]
fn prop_compare_signed_matches_corresponding_comparison(lhs: s4, rhs: s4) -> bool {
    match compare_signed(lhs, rhs) {
        Ordering::Less => lhs < rhs,
        Ordering::Equal => lhs == rhs,
        Ordering::Greater => lhs > rhs,
    }
}

// Compares two integers of the same sign and width.
//
// The reason to use this over a comparison operator such as `<` is that this exhaustively
// handles all 3 cases; you can match on the result.
pub fn compare<S: bool, N: u32>(lhs: xN[S][N], rhs: xN[S][N]) -> Ordering {
    if S {
        compare_signed(lhs as sN[N], rhs as sN[N])
    } else {
        compare_unsigned(lhs as uN[N], rhs as uN[N])
    }
}

#[test]
fn test_compare() {
    // Unsigned comparisons
    assert_eq(compare(u8:1, u8:2), Ordering::Less);
    assert_eq(compare(u8:2, u8:2), Ordering::Equal);
    assert_eq(compare(u8:3, u8:2), Ordering::Greater);

    // Signed comparisons
    assert_eq(compare(s8:-1, s8:0), Ordering::Less);
    assert_eq(compare(s8:0, s8:0), Ordering::Equal);
    assert_eq(compare(s8:1, s8:0), Ordering::Greater);
}

// "Explicit signed comparison" helpers for working with unsigned values, can be
// a bit more convenient and a bit more explicit intent than doing casting of
// left hand side and right hand side.

pub fn sge<N: u32>(x: uN[N], y: uN[N]) -> bool { (x as sN[N]) >= (y as sN[N]) }

pub fn sgt<N: u32>(x: uN[N], y: uN[N]) -> bool { (x as sN[N]) > (y as sN[N]) }

pub fn sle<N: u32>(x: uN[N], y: uN[N]) -> bool { (x as sN[N]) <= (y as sN[N]) }

pub fn slt<N: u32>(x: uN[N], y: uN[N]) -> bool { (x as sN[N]) < (y as sN[N]) }

#[test]
fn test_scmps() {
    assert_eq(sge(u2:3, u2:1), false);
    assert_eq(sgt(u2:3, u2:1), false);
    assert_eq(sle(u2:3, u2:1), true);
    assert_eq(slt(u2:3, u2:1), true);
}

// Performs integer exponentiation as in Hacker's Delight, section 11-3.
// Only nonnegative exponents are allowed, hence the uN parameter for spow.
pub fn upow<N: u32>(x: uN[N], n: uN[N]) -> uN[N] {
    let result = uN[N]:1;
    let p = x;

    let work = for (i, (n, p, result)) in u32:0..N {
        let result = if (n & uN[N]:1) == uN[N]:1 { result * p } else { result };

        (n >> 1, p * p, result)
    }((n, p, result));
    work.2
}

pub fn spow<N: u32>(x: sN[N], n: uN[N]) -> sN[N] {
    let result = sN[N]:1;
    let p = x;

    let work = for (i, (n, p, result)): (u32, (uN[N], sN[N], sN[N])) in u32:0..N {
        let result = if (n & uN[N]:1) == uN[N]:1 { result * p } else { result };

        (n >> uN[N]:1, p * p, result)
    }((n, p, result));
    work.2
}

#[test]
fn test_upow() {
    assert_eq(upow(u32:2, u32:2), u32:4);
    assert_eq(upow(u32:2, u32:20), u32:0x100000);
    assert_eq(upow(u32:3, u32:20), u32:0xcfd41b91);
    assert_eq(upow(u32:1, u32:20), u32:0x1);
    assert_eq(upow(u32:1, u32:20), u32:0x1);
}

#[test]
fn test_spow() {
    assert_eq(spow(s32:2, u32:2), s32:4);
    assert_eq(spow(s32:2, u32:20), s32:0x100000);
    assert_eq(spow(s32:3, u32:20), s32:0xcfd41b91);
    assert_eq(spow(s32:1, u32:20), s32:0x1);
    assert_eq(spow(s32:1, u32:20), s32:0x1);
}

// Count the number of bits that are 1.
pub fn popcount<N: u32>(x: bits[N]) -> bits[N] {
    let acc = for (i, acc): (u32, bits[N]) in u32:0..N {
        acc + (x[i+:u1] as bits[N])
    }(bits[N]:0);
    acc
}

#[test]
fn test_popcount() {
    assert_eq(popcount(u17:0xa5a5), u17:8);
    assert_eq(popcount(u17:0x1a5a5), u17:9);
    assert_eq(popcount(u1:0x0), u1:0);
    assert_eq(popcount(u1:0x1), u1:1);
    assert_eq(popcount(u32:0xffffffff), u32:32);
}

// Rotate `x` right by `y` bits.
pub fn rotr<N: u32>(x: bits[N], y: bits[N]) -> bits[N] {
    let y_mod = y % (N as bits[N]);
    (x >> y_mod) | (x << ((N as bits[N]) - y_mod))
}

#[test]
fn test_rotr() {
    assert_eq(bits[3]:0b101, rotr(bits[3]:0b011, bits[3]:1));
    assert_eq(bits[3]:0b011, rotr(bits[3]:0b110, bits[3]:1));
}

#[test]
fn test_rotr_zero_shift() {
    let x = u8:0xaa;
    let result = rotr(x, u8:0);
    assert_eq(result, x);  // Should be unchanged
}

#[quickcheck]
fn rotr_preserves_popcount(x: u8, y: u8) -> bool { popcount(x) == popcount(rotr(x, y)) }

// Rotate `x` left by `y` bits.
pub fn rotl<N: u32>(x: bits[N], y: bits[N]) -> bits[N] {
    let y_mod = y % (N as bits[N]);
    (x << y_mod) | (x >> ((N as bits[N]) - y_mod))
}

#[quickcheck]
fn rotr_then_rotl_is_original(x: u8, y: u8) -> bool { x == rotl(rotr(x, y), y) }

#[quickcheck]
fn rotl_then_rotr_is_original(x: u8, y: u8) -> bool { x == rotr(rotl(x, y), y) }

#[test]
fn test_rotl() {
    assert_eq(bits[3]:0b110, rotl(bits[3]:0b011, bits[3]:1));
    assert_eq(bits[3]:0b101, rotl(bits[3]:0b110, bits[3]:1));
}

#[test]
fn test_rotl_zero_shift() {
    let x = u8:0xaa;
    let result = rotl(x, u8:0);
    assert_eq(result, x);  // Should be unchanged
}

#[quickcheck]
fn rotl_preserves_popcount(x: u8, y: u8) -> bool { popcount(x) == popcount(rotl(x, y)) }

// Converts an unsigned number to a signed number of the same width.
//
// This is the moral equivalent of:
//
//     x as sN[std::sizeof(x)]
//
// That is, you might use it when you don't want to figure out the width of x
// in order to perform a cast, you just know that the unsigned number you have
// you want to turn signed.
pub fn to_signed<N: u32>(x: uN[N]) -> sN[N] { x as sN[N] }

#[test]
fn test_to_signed() {
    let x = u32:42;
    assert_eq(s32:42, to_signed(x));
    assert_eq(x as sN[sizeof(x)], to_signed(x));
    let x = u8:42;
    assert_eq(s8:42, to_signed(x));
    assert_eq(x as sN[sizeof(x)], to_signed(x));
}

// As with to_signed but for signed-to-unsigned conversion.
pub fn to_unsigned<N: u32>(x: sN[N]) -> uN[N] { x as uN[N] }

#[test]
fn test_to_unsigned() {
    let x = s32:42;
    assert_eq(u32:42, to_unsigned(x));
    assert_eq(x as uN[sizeof(x)], to_unsigned(x));
    let x = s8:42;
    assert_eq(u8:42, to_unsigned(x));
    assert_eq(x as uN[sizeof(x)], to_unsigned(x));
}

// Adds two unsigned integers and detects for overflow.
//
// uadd_with_overflow<V: u32>(x: uN[N], y : uN[M]) returns a 2-tuple
// indicating overflow (boolean) and a sum (x+y) as uN[V).  An overflow
// occurs if the result does not fit within a uN[V].
//
// Example usage:
//  let result : (bool, u16) = uadd_with_overflow<u32:16>(x, y);
//
pub fn uadd_with_overflow
    <V: u32, N: u32, M: u32, MAX_N_M: u32 = {max(N, M)}, MAX_N_M_V: u32 = {max(MAX_N_M, V)}>
    (x: uN[N], y: uN[M]) -> (bool, uN[V]) {

    let x_extended = widening_cast<uN[MAX_N_M_V + u32:1]>(x);
    let y_extended = widening_cast<uN[MAX_N_M_V + u32:1]>(y);

    let full_result: uN[MAX_N_M_V + u32:1] = x_extended + y_extended;
    let narrowed_result = full_result as uN[V];
    let overflow_detected = or_reduce(full_result[V as s32:]);

    (overflow_detected, narrowed_result)
}

#[test]
fn test_uadd_with_overflow() {
    assert_eq(uadd_with_overflow<u32:1>(u4:0, u5:0), (false, u1:0));
    assert_eq(uadd_with_overflow<u32:1>(u4:1, u5:0), (false, u1:1));
    assert_eq(uadd_with_overflow<u32:1>(u4:1, u5:1), (true, u1:0));
    assert_eq(uadd_with_overflow<u32:1>(u4:2, u5:1), (true, u1:1));

    assert_eq(uadd_with_overflow<u32:4>(u4:15, u3:0), (false, u4:15));
    assert_eq(uadd_with_overflow<u32:4>(u4:8, u3:7), (false, u4:15));
    assert_eq(uadd_with_overflow<u32:4>(u4:9, u3:7), (true, u4:0));
    assert_eq(uadd_with_overflow<u32:4>(u4:10, u3:6), (true, u4:0));
    assert_eq(uadd_with_overflow<u32:4>(u4:11, u3:6), (true, u4:1));
}

// Extract bits given a fixed-point integer with a constant offset.
//   i.e. let x_extended = x as uN[max(unsigned_sizeof(x) + FIXED_SHIFT, TO_EXCLUSIVE)];
//        (x_extended << FIXED_SHIFT)[FROM_INCLUSIVE:TO_EXCLUSIVE]
//
// This function behaves as-if x has reasonably infinite precision so that
// the result is zero-padded if FROM_INCLUSIVE or TO_EXCLUSIVE are out of
// range of the original x's bitwidth.
//
// If TO_EXCLUSIVE <= FROM_INCLUSIVE, the result will be a zero-bit uN[0].
pub fn extract_bits
    <FROM_INCLUSIVE: u32, TO_EXCLUSIVE: u32, FIXED_SHIFT: u32, N: u32,
     EXTRACT_WIDTH: u32 = {max(s32:0, TO_EXCLUSIVE as s32 - FROM_INCLUSIVE as s32) as u32}>
    (x: uN[N]) -> uN[EXTRACT_WIDTH] {
    if TO_EXCLUSIVE <= FROM_INCLUSIVE {
        uN[EXTRACT_WIDTH]:0
    } else {
        // With a non-zero fixed width, all lower bits of index < fixed_shift are
        // are zero.
        let lower_bits =
            uN[checked_cast<u32>(max(s32:0, FIXED_SHIFT as s32 - FROM_INCLUSIVE as s32))]:0;

        // Based on the input of N bits and a fixed shift, there are an effective
        // count of N + fixed_shift known bits.  All bits of index >
        // N + fixed_shift - 1 are zero's.
        const UPPER_BIT_COUNT = checked_cast<u32>(
            max(s32:0, N as s32 + FIXED_SHIFT as s32 - TO_EXCLUSIVE as s32 - s32:1));
        const UPPER_BITS = uN[UPPER_BIT_COUNT]:0;

        if FIXED_SHIFT < FROM_INCLUSIVE {
            // The bits extracted start within or after the middle span.
            //  upper_bits ++ middle_bits
            const FROM: s32 = min(FROM_INCLUSIVE as s32 - FIXED_SHIFT as s32, N as s32);
            const TO: s32 = min(TO_EXCLUSIVE as s32 - FIXED_SHIFT as s32, N as s32);
            let middle_bits = UPPER_BITS ++ x[FROM:TO];
            (UPPER_BITS ++ middle_bits) as uN[EXTRACT_WIDTH]
        } else if FIXED_SHIFT <= TO_EXCLUSIVE {
            // The bits extracted start within the fixed_shift span.
            const TO: s32 = min(TO_EXCLUSIVE as s32 - FIXED_SHIFT as s32, N as s32);
            let middle_bits = x[0:TO];

            (UPPER_BITS ++ middle_bits ++ lower_bits) as uN[EXTRACT_WIDTH]
        } else {
            uN[EXTRACT_WIDTH]:0
        }
    }
}

#[test]
fn test_extract_bits() {
    assert_eq(extract_bits<u32:4, u32:4, u32:0>(u4:0x9), uN[0]:0);
    assert_eq(extract_bits<u32:0, u32:4, u32:0>(u4:0x9), u4:0x9);  // 0b[1001]

    assert_eq(extract_bits<u32:0, u32:5, u32:0>(u4:0xf), u5:0xf);  // 0b[01111]
    assert_eq(extract_bits<u32:0, u32:5, u32:1>(u4:0xf), u5:0x1e);  // 0b0[11110]
    assert_eq(extract_bits<u32:0, u32:5, u32:2>(u4:0xf), u5:0x1c);  // 0b1[11100]
    assert_eq(extract_bits<u32:0, u32:5, u32:3>(u4:0xf), u5:0x18);  // 0b11[11000]
    assert_eq(extract_bits<u32:0, u32:5, u32:4>(u4:0xf), u5:0x10);  // 0b111[10000]
    assert_eq(extract_bits<u32:0, u32:5, u32:5>(u4:0xf), u5:0x0);  // 0b1111[00000]

    assert_eq(extract_bits<u32:2, u32:5, u32:0>(u4:0xf), u3:0x3);  // 0b[011]11
    assert_eq(extract_bits<u32:2, u32:5, u32:1>(u4:0xf), u3:0x7);  // 0b[111]10
    assert_eq(extract_bits<u32:2, u32:5, u32:2>(u4:0xf), u3:0x7);  // 0b1[111]00
    assert_eq(extract_bits<u32:2, u32:5, u32:3>(u4:0xf), u3:0x6);  // 0b11[110]00
    assert_eq(extract_bits<u32:2, u32:5, u32:4>(u4:0xf), u3:0x4);  // 0b111[100]00
    assert_eq(extract_bits<u32:2, u32:5, u32:5>(u4:0xf), u3:0x0);  // 0b1111[000]00

    assert_eq(extract_bits<u32:0, u32:4, u32:0>(u4:0xf), u4:0xf);  // 0b[1111]
    assert_eq(extract_bits<u32:0, u32:4, u32:1>(u4:0xf), u4:0xe);  // 0b1[1110]
    assert_eq(extract_bits<u32:0, u32:4, u32:2>(u4:0xf), u4:0xc);  // 0b11[1100]
    assert_eq(extract_bits<u32:0, u32:4, u32:3>(u4:0xf), u4:0x8);  // 0b111[1000]
    assert_eq(extract_bits<u32:0, u32:4, u32:4>(u4:0xf), u4:0x0);  // 0b1111[0000]
    assert_eq(extract_bits<u32:0, u32:4, u32:5>(u4:0xf), u4:0x0);  // 0b11110[0000]

    assert_eq(extract_bits<u32:1, u32:4, u32:0>(u4:0xf), u3:0x7);  // 0b[111]1
    assert_eq(extract_bits<u32:1, u32:4, u32:1>(u4:0xf), u3:0x7);  // 0b1[111]0
    assert_eq(extract_bits<u32:1, u32:4, u32:2>(u4:0xf), u3:0x6);  // 0b11[110]0
    assert_eq(extract_bits<u32:1, u32:4, u32:3>(u4:0xf), u3:0x4);  // 0b111[100]0
    assert_eq(extract_bits<u32:1, u32:4, u32:4>(u4:0xf), u3:0x0);  // 0b1111[000]0
    assert_eq(extract_bits<u32:1, u32:4, u32:5>(u4:0xf), u3:0x0);  // 0b11110[000]0

    assert_eq(extract_bits<u32:2, u32:4, u32:0>(u4:0xf), u2:0x3);  // 0b[11]11
    assert_eq(extract_bits<u32:2, u32:4, u32:1>(u4:0xf), u2:0x3);  // 0b1[11]10
    assert_eq(extract_bits<u32:2, u32:4, u32:2>(u4:0xf), u2:0x3);  // 0b11[11]00
    assert_eq(extract_bits<u32:2, u32:4, u32:3>(u4:0xf), u2:0x2);  // 0b111[10]00
    assert_eq(extract_bits<u32:2, u32:4, u32:4>(u4:0xf), u2:0x0);  // 0b1111[00]00
    assert_eq(extract_bits<u32:2, u32:4, u32:5>(u4:0xf), u2:0x0);  // 0b11110[00]00

    assert_eq(extract_bits<u32:3, u32:4, u32:0>(u4:0xf), u1:0x1);  // 0b[1]111
    assert_eq(extract_bits<u32:3, u32:4, u32:1>(u4:0xf), u1:0x1);  // 0b1[1]110
    assert_eq(extract_bits<u32:3, u32:4, u32:2>(u4:0xf), u1:0x1);  // 0b11[1]100
    assert_eq(extract_bits<u32:3, u32:4, u32:3>(u4:0xf), u1:0x1);  // 0b111[1]000
    assert_eq(extract_bits<u32:3, u32:4, u32:4>(u4:0xf), u1:0x0);  // 0b1111[0]000
    assert_eq(extract_bits<u32:3, u32:4, u32:5>(u4:0xf), u1:0x0);  // 0b11110[0]000

    assert_eq(extract_bits<u32:0, u32:3, u32:0>(u4:0xf), u3:0x7);  // 0b1[111]
    assert_eq(extract_bits<u32:0, u32:3, u32:1>(u4:0xf), u3:0x6);  // 0b11[110]
    assert_eq(extract_bits<u32:0, u32:3, u32:2>(u4:0xf), u3:0x4);  // 0b111[100]
    assert_eq(extract_bits<u32:0, u32:3, u32:3>(u4:0xf), u3:0x0);  // 0b1111[000]
    assert_eq(extract_bits<u32:0, u32:3, u32:4>(u4:0xf), u3:0x0);  // 0b11110[000]
    assert_eq(extract_bits<u32:0, u32:3, u32:5>(u4:0xf), u3:0x0);  // 0b111100[000]

    assert_eq(extract_bits<u32:1, u32:3, u32:0>(u4:0xf), u2:0x3);  // 0b1[11]1
    assert_eq(extract_bits<u32:1, u32:3, u32:1>(u4:0xf), u2:0x3);  // 0b11[11]0
    assert_eq(extract_bits<u32:1, u32:3, u32:2>(u4:0xf), u2:0x2);  // 0b111[10]0
    assert_eq(extract_bits<u32:1, u32:3, u32:3>(u4:0xf), u2:0x0);  // 0b1111[00]0
    assert_eq(extract_bits<u32:1, u32:3, u32:4>(u4:0xf), u2:0x0);  // 0b11110[00]0
    assert_eq(extract_bits<u32:1, u32:3, u32:5>(u4:0xf), u2:0x0);  // 0b111100[00]0

    assert_eq(extract_bits<u32:2, u32:3, u32:0>(u4:0xf), u1:0x1);  // 0b1[1]11
    assert_eq(extract_bits<u32:2, u32:3, u32:1>(u4:0xf), u1:0x1);  // 0b11[1]10
    assert_eq(extract_bits<u32:2, u32:3, u32:2>(u4:0xf), u1:0x1);  // 0b111[1]00
    assert_eq(extract_bits<u32:2, u32:3, u32:3>(u4:0xf), u1:0x0);  // 0b1111[0]00
    assert_eq(extract_bits<u32:2, u32:3, u32:4>(u4:0xf), u1:0x0);  // 0b11110[0]00
    assert_eq(extract_bits<u32:2, u32:3, u32:5>(u4:0xf), u1:0x0);  // 0b111100[0]00
}

// Multiplies two numbers and detects for overflow.
//
// umul_with_overflow<V: u32>(x: uN[N], y : uN[M]) returns a 2-tuple
// indicating overflow (boolean) and a product (x*y) as uN[V].  An overflow
// occurs if the result does not fit within a uN[V].
//
// Example usage:
//  let result : (bool, u16) = umul_with_overflow<u32:16>(x, y);
//
pub fn umul_with_overflow
    <V: u32, N: u32, M: u32, N_lower_bits: u32 = {N >> u32:1},
     N_upper_bits: u32 = {N - N_lower_bits}, M_lower_bits: u32 = {M >> u32:1},
     M_upper_bits: u32 = {M - M_lower_bits},
     Min_N_M_lower_bits: u32 = {min(N_lower_bits, M_lower_bits)}, N_Plus_M: u32 = {N + M}>
    (x: uN[N], y: uN[M]) -> (bool, uN[V]) {
    // Break x and y into two halves.
    // x = x1 ++ x0,
    // y = y1 ++ x1,
    let x1 = x[N_lower_bits as s32:];
    let x0 = x[s32:0:N_lower_bits as s32];

    let y1 = y[M_lower_bits as s32:];
    let y0 = y[s32:0:M_lower_bits as s32];

    // Bits [0 : N_lower_bits+M_lower_bits]]
    let x0y0: uN[N_lower_bits + M_lower_bits] = umul(x0, y0);

    // Bits [M_lower_bits +: N_upper_bits+M_lower_bits]
    let x1y0: uN[N_upper_bits + M_lower_bits] = umul(x1, y0);

    // Bits [N_lower_bits +: N_lower_bits+M_upper_bits]
    let x0y1: uN[M_upper_bits + N_lower_bits] = umul(x0, y1);

    // Bits [N_lower_bits+M_lower_bits += N_upper_bits+M_upper_bits]
    let x1y1: uN[N_upper_bits + M_upper_bits] = umul(x1, y1);

    // Break the above numbers into three buckets
    //  [0: min(N_lower_bits, M_lower_bits)] --> only from x0y0
    //  [min(N_lower_bits, M_lower_bits : V] --> need to add together
    //  [V : N+M] --> need to or_reduce
    let x0y0_a = extract_bits<u32:0, Min_N_M_lower_bits, u32:0>(x0y0);
    let x0y0_b = extract_bits<Min_N_M_lower_bits, V, u32:0>(x0y0);
    let x0y0_c = extract_bits<V, N_Plus_M, u32:0>(x0y0);

    // x1 has a shift of N_lower_bits
    let x1y0_b = extract_bits<Min_N_M_lower_bits, V, N_lower_bits>(x1y0);
    let x1y0_c = extract_bits<V, N_Plus_M, N_lower_bits>(x1y0);

    // y1 has a shift of M_lower_bits
    let x0y1_b = extract_bits<Min_N_M_lower_bits, V, M_lower_bits>(x0y1);
    let x0y1_c = extract_bits<V, N_Plus_M, M_lower_bits>(x0y1);

    // x1y1 has a shift of N_lower_bits + M_lower_bits
    let x1y1_b = extract_bits<Min_N_M_lower_bits, V, {N_lower_bits + M_lower_bits}>(x1y1);
    let x1y1_c = extract_bits<V, N_Plus_M, {N_lower_bits + M_lower_bits}>(x1y1);

    // Add partial shifts to obtain the narrowed results, keeping 2 bits for overflow.
    // (x0y0_b + x1y0_b + x1y1_b + x1y1_b) ++ x0y0a
    let x0y0_b_extended = widening_cast<uN[V - Min_N_M_lower_bits + u32:2]>(x0y0_b);
    let x0y1_b_extended = widening_cast<uN[V - Min_N_M_lower_bits + u32:2]>(x0y1_b);
    let x1y0_b_extended = widening_cast<uN[V - Min_N_M_lower_bits + u32:2]>(x1y0_b);
    let x1y1_b_extended = widening_cast<uN[V - Min_N_M_lower_bits + u32:2]>(x1y1_b);
    let narrowed_result_upper =
        x0y0_b_extended + x1y0_b_extended + x0y1_b_extended + x1y1_b_extended;
    let overflow_narrowed_result_upper_sum =
        or_reduce(narrowed_result_upper[V as s32 - Min_N_M_lower_bits as s32:]);

    let partial_narrowed_result =
        narrowed_result_upper[0:V as s32 - Min_N_M_lower_bits as s32] ++ x0y0_a;

    let narrowed_result = partial_narrowed_result[0:V as s32];
    let overflow_detected = or_reduce(x0y0_c) || or_reduce(x0y1_c) || or_reduce(x1y0_c) ||
                            or_reduce(x1y1_c) || or_reduce(partial_narrowed_result[V as s32:]) ||
                            overflow_narrowed_result_upper_sum;

    (overflow_detected, narrowed_result)
}

// TODO(tedhong): 2023-08-11 Exhaustively test umul_with_overflow for
// certain bitwith combinations.
#[test]
fn test_umul_with_overflow() {
    assert_eq(umul_with_overflow<u32:1>(u4:0, u4:0), (false, u1:0));
    assert_eq(umul_with_overflow<u32:1>(u4:15, u4:0), (false, u1:0));
    assert_eq(umul_with_overflow<u32:1>(u4:1, u4:1), (false, u1:1));
    assert_eq(umul_with_overflow<u32:1>(u4:2, u4:1), (true, u1:0));
    assert_eq(umul_with_overflow<u32:1>(u4:8, u4:8), (true, u1:0));
    assert_eq(umul_with_overflow<u32:1>(u4:15, u4:15), (true, u1:1));

    assert_eq(umul_with_overflow<u32:4>(u4:0, u3:0), (false, u4:0));
    assert_eq(umul_with_overflow<u32:4>(u4:2, u3:7), (false, u4:14));
    assert_eq(umul_with_overflow<u32:4>(u4:5, u3:3), (false, u4:15));
    assert_eq(umul_with_overflow<u32:4>(u4:4, u3:4), (true, u4:0));
    assert_eq(umul_with_overflow<u32:4>(u4:9, u3:2), (true, u4:2));
    assert_eq(umul_with_overflow<u32:4>(u4:15, u3:7), (true, u4:9));

    for (i, ()): (u32, ()) in u32:0..u32:7 {
        for (j, ()): (u32, ()) in u32:0..u32:15 {
            let result = i * j;
            let overflow = result > u32:15;

            assert_eq(umul_with_overflow<u32:4>(i as u3, j as u4), (overflow, result as u4))
        }(())
    }(());
}

pub fn is_unsigned_msb_set<N: u32>(x: uN[N]) -> bool { x[-1:] }

#[test]
fn is_unsigned_msb_set_test() {
    assert_eq(false, is_unsigned_msb_set(u8:0));
    assert_eq(false, is_unsigned_msb_set(u8:1));
    assert_eq(false, is_unsigned_msb_set(u8:127));
    assert_eq(true, is_unsigned_msb_set(u8:128));
    assert_eq(true, is_unsigned_msb_set(u8:129));
}

// What verilog calls a "part-select" to extract a particular subset of bits
// from a larger bits type. This does compile-time checking that the values are
// in-bounds.
//
// Note: for new code, prefer direct bit slicing syntax; e.g.
//
//  x[LSB +: bits[WIDTH]]
//
// This is given for help in porting code from Verilog to DSLX, e.g. if a user
// wants a more direct transcription.
pub fn vslice<MSB: u32, LSB: u32, IN: u32, OUT: u32 = {MSB - LSB + u32:1}>
    (x: bits[IN]) -> bits[OUT] {
    // This will help flag if the MSB and LSB are given out of order
    const_assert!(MSB >= LSB);
    x[LSB+:bits[OUT]]
}

#[test]
fn test_vslice() {
    assert_eq(vslice<u32:7, u32:0>(u8:0xab), u8:0xab);
    assert_eq(vslice<u32:3, u32:0>(u8:0xab), u4:0xb);
    assert_eq(vslice<u32:7, u32:4>(u8:0xab), u4:0xa);
    assert_eq(vslice<u32:0, u32:0>(u8:0xab), u1:1);
}

// Deprecated functions, to be removed in favor of sizeof() which is
// signedness-parameterized.
//
// TODO(https://github.com/google/xls/issues/1348): 2024-03-12 Add deprecated
// annotation to DSLX so we can tag functions that are deprecated like these
// are.

pub fn sizeof_signed<N: u32>(x: sN[N]) -> u32 { N }

pub fn sizeof_unsigned<N: u32>(x: uN[N]) -> u32 { N }

// Returns `x` with all but the least-significant `n` bits set to zero.
pub fn keep_lsbs<WIDTH: u32, N_WIDTH: u32>(x: uN[WIDTH], n: uN[N_WIDTH]) -> uN[WIDTH] {
    x & !(all_ones!<uN[WIDTH]>() << n)
}

#[test]
fn keep_lsbs_test() {
    assert_eq(u8:0x00, keep_lsbs(u8:0xab, u4:0));
    assert_eq(u8:0x01, keep_lsbs(u8:0xab, u4:1));
    assert_eq(u8:0x00, keep_lsbs(u8:0xaa, u4:1));
    assert_eq(u8:0x00, keep_lsbs(u8:0xa0, u4:4));
    assert_eq(u8:0x0a, keep_lsbs(u8:0xaa, u4:4));
    assert_eq(u8:0x08, keep_lsbs(u8:0xa8, u4:4));
    assert_eq(u8:0xa0, keep_lsbs(u8:0xa0, u4:8));
    assert_eq(u8:0xaa, keep_lsbs(u8:0xaa, u4:8));
    assert_eq(u8:0xa8, keep_lsbs(u8:0xa8, u4:8));
    assert_eq(u8:0xa0, keep_lsbs(u8:0xa0, u4:12));
    assert_eq(u8:0xaa, keep_lsbs(u8:0xaa, u4:12));
    assert_eq(u8:0xa8, keep_lsbs(u8:0xa8, u4:12));
}

// Returns `x` with all but the most-significant `n` bits set to zero.
pub fn keep_msbs<WIDTH: u32, N_WIDTH: u32>(x: uN[WIDTH], n: uN[N_WIDTH]) -> uN[WIDTH] {
    x & !(all_ones!<uN[WIDTH]>() >> n)
}

#[test]
fn keep_msbs_test() {
    assert_eq(u8:0x00, keep_msbs(u8:0xab, u4:0));
    assert_eq(u8:0x80, keep_msbs(u8:0xab, u4:1));
    assert_eq(u8:0x00, keep_msbs(u8:0x2b, u4:1));
    assert_eq(u8:0x00, keep_msbs(u8:0x0a, u4:4));
    assert_eq(u8:0xa0, keep_msbs(u8:0xaa, u4:4));
    assert_eq(u8:0x80, keep_msbs(u8:0x8a, u4:4));
    assert_eq(u8:0xa0, keep_msbs(u8:0xa0, u4:8));
    assert_eq(u8:0xaa, keep_msbs(u8:0xaa, u4:8));
    assert_eq(u8:0xa8, keep_msbs(u8:0xa8, u4:8));
    assert_eq(u8:0xa0, keep_msbs(u8:0xa0, u4:12));
    assert_eq(u8:0xaa, keep_msbs(u8:0xaa, u4:12));
    assert_eq(u8:0xa8, keep_msbs(u8:0xa8, u4:12));
}

// Returns `value` with the least-significant `n` bits set to zero.
pub fn clear_lsbs<WIDTH: u32, N_WIDTH: u32>(value: uN[WIDTH], n: uN[N_WIDTH]) -> uN[WIDTH] {
    value & (all_ones!<uN[WIDTH]>() << n)
}

#[test]
fn clear_lsbs_test() {
    assert_eq(u8:0xab, clear_lsbs(u8:0xab, u4:0));
    assert_eq(u8:0xaa, clear_lsbs(u8:0xab, u4:1));
    assert_eq(u8:0xaa, clear_lsbs(u8:0xaa, u4:1));
    assert_eq(u8:0xa0, clear_lsbs(u8:0xa0, u4:4));
    assert_eq(u8:0xa0, clear_lsbs(u8:0xaa, u4:4));
    assert_eq(u8:0xa0, clear_lsbs(u8:0xa8, u4:4));
    assert_eq(u8:0x00, clear_lsbs(u8:0xa0, u4:8));
    assert_eq(u8:0x00, clear_lsbs(u8:0xaa, u4:8));
    assert_eq(u8:0x00, clear_lsbs(u8:0xa8, u4:8));
    assert_eq(u8:0x00, clear_lsbs(u8:0xa0, u4:12));
    assert_eq(u8:0x00, clear_lsbs(u8:0xaa, u4:12));
    assert_eq(u8:0x00, clear_lsbs(u8:0xa8, u4:12));
}

// Returns `value` with the most-significant `n` bits set to zero.
pub fn clear_msbs<WIDTH: u32, N_WIDTH: u32>(value: uN[WIDTH], n: uN[N_WIDTH]) -> uN[WIDTH] {
    value & (all_ones!<uN[WIDTH]>() >> n)
}

#[test]
fn clear_msbs_test() {
    assert_eq(u8:0xab, clear_msbs(u8:0xab, u4:0));
    assert_eq(u8:0x2b, clear_msbs(u8:0xab, u4:1));
    assert_eq(u8:0x2b, clear_msbs(u8:0x2b, u4:1));
    assert_eq(u8:0x00, clear_msbs(u8:0xa0, u4:4));
    assert_eq(u8:0x0a, clear_msbs(u8:0xaa, u4:4));
    assert_eq(u8:0x08, clear_msbs(u8:0xa8, u4:4));
    assert_eq(u8:0x00, clear_msbs(u8:0xa0, u4:8));
    assert_eq(u8:0x00, clear_msbs(u8:0xaa, u4:8));
    assert_eq(u8:0x00, clear_msbs(u8:0xa8, u4:8));
    assert_eq(u8:0x00, clear_msbs(u8:0xa0, u4:12));
    assert_eq(u8:0x00, clear_msbs(u8:0xaa, u4:12));
    assert_eq(u8:0x00, clear_msbs(u8:0xa8, u4:12));
}

// Implementation of or_reduce_lsb() with choice of implementation.
// The "do_mask_impl" template parameter chooses the implementation and is
// subject to change while tested in different contexts.
// Public interface is or_reduce_lsb()
fn or_reduce_lsb_impl<DO_MASK_IMPL: bool, WIDTH: u32, N_WIDTH: u32>
    (value: uN[WIDTH], n: uN[N_WIDTH]) -> bool {
    if DO_MASK_IMPL {
        // Mask the relevant bits, then compare.
        keep_lsbs(value, n) != uN[WIDTH]:0
    } else {
        // shift out uninteresting bits, then compare.
        value << (WIDTH as uN[N_WIDTH] - n) != uN[WIDTH]:0
    }
}

#[test]
fn or_reduce_lsb_impl_impl_test() {
    assert_eq(false, or_reduce_lsb_impl<true>(u8:0xab, u4:0));
    assert_eq(true, or_reduce_lsb_impl<true>(u8:0xab, u4:1));
    assert_eq(false, or_reduce_lsb_impl<true>(u8:0xaa, u4:1));
    assert_eq(false, or_reduce_lsb_impl<true>(u8:0xa0, u4:4));
    assert_eq(true, or_reduce_lsb_impl<true>(u8:0xaa, u4:4));
    assert_eq(true, or_reduce_lsb_impl<true>(u8:0xa8, u4:4));

    assert_eq(false, or_reduce_lsb_impl<false>(u8:0xab, u4:0));
    assert_eq(true, or_reduce_lsb_impl<false>(u8:0xab, u4:1));
    assert_eq(false, or_reduce_lsb_impl<false>(u8:0xaa, u4:1));
    assert_eq(false, or_reduce_lsb_impl<false>(u8:0xa0, u4:4));
    assert_eq(true, or_reduce_lsb_impl<false>(u8:0xaa, u4:4));
    assert_eq(true, or_reduce_lsb_impl<false>(u8:0xa8, u4:4));
}

// or_reduce the lower "n" bits of "value". Return 'true', if any of the "n" lower
// bits is set.
pub fn or_reduce_lsb<WIDTH: u32, N_WIDTH: u32>(value: uN[WIDTH], n: uN[N_WIDTH]) -> bool {
    or_reduce_lsb_impl<true>(value, n)  // using impl. typically yielding best QoR
}

// Combine the values of the two clzt halfs. If left is saturated, zeroes on the
// right continue and need to be added; otherwise just keep left.
// The outputs are one bit wider than the input and the inputs are never
// larger than a leading one followed by zeros. This means that adding will
// either never carry or in the trivial 'carry'-case number just double.
// Thus, this whole operation can be lowered to muxing.
fn combine_clzt_halfs<N: u32>(left: uN[N], right: uN[N]) -> uN[N + u32:1] {
    match (left[-1:], right[-1:]) {
        (u1:1, u1:1) => left ++ u1:0,  // Both at maximum: add them, i.e. mult by 2
        (u1:1, u1:0) => u2:0b01 ++ right[:-1],  // right side less than max zero bits
        _ => u1:0 ++ left,  // right side can't contribute anymore
    }
}

#[test]
fn combine_clzt_halfs_test() {
    // Exhaustive test of all valid inputs

    // rhs does not matter if left side has less than full bits
    assert_eq(combine_clzt_halfs(u2:0b00, u2:0b00), u3:0);
    assert_eq(combine_clzt_halfs(u2:0b00, u2:0b01), u3:0);
    assert_eq(combine_clzt_halfs(u2:0b00, u2:0b10), u3:0);

    assert_eq(combine_clzt_halfs(u2:0b01, u2:0b00), u3:1);
    assert_eq(combine_clzt_halfs(u2:0b01, u2:0b01), u3:1);
    assert_eq(combine_clzt_halfs(u2:0b01, u2:0b10), u3:1);

    // Left side saturated count, add rhs.
    assert_eq(combine_clzt_halfs(u2:0b10, u2:0b00), u3:2);
    assert_eq(combine_clzt_halfs(u2:0b10, u2:0b01), u3:3);
    assert_eq(combine_clzt_halfs(u2:0b10, u2:0b10), u3:4);

    assert_eq(combine_clzt_halfs(u2:0b10, u2:0b10), u3:4);
}

// What we _actually_ want to write after https://github.com/google/xls/issues/510 is addressed

// Count leading zeroes using a tree of gates. Input is required to be a power of 2 bits.
// Use clzt() for general-purpose function allowing arbitrary bits.
// fn clzt_pow2<N: u32, RESULT_BITS: u32 = {clog2(N + u32:1)}>(value: bits[N]) -> uN[RESULT_BITS] {
//     const_assert!(is_pow2(N));
//     if N == u32:1 {
//         !value  // Trivial case
//     } else {
//         const N_HALF = (N >> 1) as s32;
//         combine_clzt_halfs(clzt_pow2(value[-N_HALF:]), clzt_pow2(value[:-N_HALF]))
//     }
// }

// ... Since we don't have recursion yet, expand manually into clzt_pow_$N()

fn clzt_pow2_1(value: bits[1]) -> uN[1] {
    !value  // Trivial case
}

// All the following are identical, just with lifting the N parameter as part
// of the symbol name to allow recursion.
fn clzt_pow2_2(value: bits[2]) -> uN[2] {
    const N_HALF = (u32:2 >> 1) as s32;
    combine_clzt_halfs(clzt_pow2_1(value[-N_HALF:]), clzt_pow2_1(value[:-N_HALF]))
}

fn clzt_pow2_4(value: bits[4]) -> uN[3] {
    const N_HALF = (u32:4 >> 1) as s32;
    combine_clzt_halfs(clzt_pow2_2(value[-N_HALF:]), clzt_pow2_2(value[:-N_HALF]))
}

fn clzt_pow2_8(value: bits[8]) -> uN[4] {
    const N_HALF = (u32:8 >> 1) as s32;
    combine_clzt_halfs(clzt_pow2_4(value[-N_HALF:]), clzt_pow2_4(value[:-N_HALF]))
}

fn clzt_pow2_16(value: bits[16]) -> uN[5] {
    const N_HALF = (u32:16 >> 1) as s32;
    combine_clzt_halfs(clzt_pow2_8(value[-N_HALF:]), clzt_pow2_8(value[:-N_HALF]))
}

fn clzt_pow2_32(value: bits[32]) -> uN[6] {
    const N_HALF = (u32:32 >> 1) as s32;
    combine_clzt_halfs(clzt_pow2_16(value[-N_HALF:]), clzt_pow2_16(value[:-N_HALF]))
}

fn clzt_pow2_64(value: bits[64]) -> uN[7] {
    const N_HALF = (u32:64 >> 1) as s32;
    combine_clzt_halfs(clzt_pow2_32(value[-N_HALF:]), clzt_pow2_32(value[:-N_HALF]))
}

fn clzt_pow2_128(value: bits[128]) -> uN[8] {
    const N_HALF = (u32:128 >> 1) as s32;
    combine_clzt_halfs(clzt_pow2_64(value[-N_HALF:]), clzt_pow2_64(value[:-N_HALF]))
}

fn clzt_pow2_256(value: bits[256]) -> uN[9] {
    const N_HALF = (u32:256 >> 1) as s32;
    combine_clzt_halfs(clzt_pow2_128(value[-N_HALF:]), clzt_pow2_128(value[:-N_HALF]))
}

fn clzt_pow2_512(value: bits[512]) -> uN[10] {
    const N_HALF = (u32:512 >> 1) as s32;
    combine_clzt_halfs(clzt_pow2_256(value[-N_HALF:]), clzt_pow2_256(value[:-N_HALF]))
}

// Count leading zeroes for power of 2 numbers.
// Since we can't have recursion, manually expand this here up to 64 bits
pub fn clzt_pow2<N: u32, RESULT_BITS: u32 = {clog2(N + u32:1)}>(value: bits[N]) -> uN[RESULT_BITS] {
    const_assert!(is_pow2(N));
    match N {
        // These casts for the arguments and return types should not be needed.
        u32:512 => clzt_pow2_512(value as uN[512]) as uN[RESULT_BITS],
        u32:256 => clzt_pow2_256(value as uN[256]) as uN[RESULT_BITS],
        u32:128 => clzt_pow2_128(value as uN[128]) as uN[RESULT_BITS],
        u32:64 => clzt_pow2_64(value as uN[64]) as uN[RESULT_BITS],
        u32:32 => clzt_pow2_32(value as uN[32]) as uN[RESULT_BITS],
        u32:16 => clzt_pow2_16(value as uN[16]) as uN[RESULT_BITS],
        u32:8 => clzt_pow2_8(value as uN[8]) as uN[RESULT_BITS],
        u32:4 => clzt_pow2_4(value as uN[4]) as uN[RESULT_BITS],
        u32:2 => clzt_pow2_2(value as uN[2]) as uN[RESULT_BITS],
        u32:1 => clzt_pow2_1(value as uN[1]) as uN[RESULT_BITS],
        _ => clz(value) as uN[RESULT_BITS],  // More bits ? Fall back to builtin for now.
    }
}

// count leading zero 'recursive'
#[test]
fn clzt_pow2_test() {
    // Individual explicit number
    assert_eq(clzt_pow2_2(u2:0), u2:2);
    assert_eq(clzt_pow2_2(u2:1), u2:1);
    assert_eq(clzt_pow2_2(u2:2), u2:0);
    assert_eq(clzt_pow2_2(u2:3), u2:0);

    assert_eq(clzt_pow2_4(u4:0b0000), u3:4);
    assert_eq(clzt_pow2_4(u4:0b0001), u3:3);
    assert_eq(clzt_pow2_4(u4:0b0010), u3:2);

    assert_eq(clzt_pow2_4(u4:0b0100), u3:1);
    assert_eq(clzt_pow2_4(u4:0b0101), u3:1);

    assert_eq(clzt_pow2_4(u4:0b1000), u3:0);
    assert_eq(clzt_pow2_4(u4:0b1001), u3:0);

    assert_eq(clzt_pow2_8(u8:0b10000000), u4:0);
    assert_eq(clzt_pow2_8(u8:0b01000000), u4:1);
    assert_eq(clzt_pow2_8(u8:0b00100000), u4:2);
    assert_eq(clzt_pow2_8(u8:0b00000001), u4:7);
    assert_eq(clzt_pow2_8(u8:0b00000000), u4:8);

    // Driver function
    assert_eq(clzt_pow2(u2:3), u2:0);
    assert_eq(clzt_pow2(u4:0b0010), u3:2);
    assert_eq(clzt_pow2(u8:0b01000000), u4:1);
}

// Given a number, what is the next power of two. E.g. 5 -> 8; 8 -> 8; 20 -> 32
pub fn next_pow2(n: u32) -> u32 { upow(u32:2, clog2(n)) }

#[test]
fn test_next_pow2() {
    assert_eq(next_pow2(u32:1), u32:1);
    assert_eq(next_pow2(u32:2), u32:2);
    assert_eq(next_pow2(u32:4), u32:4);
    assert_eq(next_pow2(u32:5), u32:8);
    assert_eq(next_pow2(u32:7), u32:8);
    assert_eq(next_pow2(u32:8), u32:8);
    assert_eq(next_pow2(u32:9), u32:16);
    assert_eq(next_pow2(u32:16), u32:16);
    assert_eq(next_pow2(u32:20), u32:32);
    assert_eq(next_pow2(u32:47), u32:64);
    assert_eq(next_pow2(u32:65), u32:128);
}

// General purpose clzt() number that works on any number, though most efficient for powers of two.
// Named std::clzt() with 't'-suffix for 'tree' and to disambiguate from builtin-clz()
pub fn clzt<N: u32, RESULT_BITS: u32 = {clog2(N + u32:1)}>(value: bits[N]) -> uN[RESULT_BITS] {
    const BITS_MISSING_UNTIL_POWER_TWO = next_pow2(N) - N;
    // To get a proper power of 2 number, pad the trailing end with non-zeroes.
    clzt_pow2(value ++ mask_bits<BITS_MISSING_UNTIL_POWER_TWO>()) as uN[RESULT_BITS]
}

#[test]
fn clzt_test() {
    assert_eq(clzt(u2:3), u2:0);
    assert_eq(clzt(u4:0b0010), u3:2);
    assert_eq(clzt(u8:0b01000000), u4:1);
    assert_eq(clzt(u9:0b010000000), u4:1);

    assert_eq(clzt(u8:0b000000000), u4:8);
    assert_eq(clzt(u9:0b0000000000), u4:9);

    // Make sure numbers outside the implemented range of clzt() fall back
    // to builtin clz()
    assert_eq(clzt(uN[717]:1 << 710), u10:6);

    const TEST_UP_TO = u32:555;  // Test a bit beyond implemented 512
    for (N, _): (u32, ()) in 0..TEST_UP_TO {
        let number = uN[TEST_UP_TO + 1]:1 << N;
        let expectd_leading_zeres = (TEST_UP_TO - N) as u10;
        assert_eq(clzt(number), expectd_leading_zeres);
    }(());
}

#[quickcheck]
fn prop_clzt_same_as_clz(x: u64) -> bool { clz(x) == clzt(x) as u64 }

/// Returns whether all the `items` are distinct (i.e. there are no duplicate
/// items) after the `valid` mask is applied.
pub fn distinct<COUNT: u32, N: u32, S: bool>(items: xN[S][N][COUNT], valid: bool[COUNT]) -> bool {
    const INIT_ALL_DISTINCT = true;
    for (i, all_distinct) in u32:0..COUNT {
        for (j, all_distinct) in u32:0..COUNT {
            if i != j && valid[i] && valid[j] && items[i] == items[j] {
                false
            } else {
                all_distinct
            }
        }(all_distinct)
    }(INIT_ALL_DISTINCT)
}

#[test]
fn test_simple_nondistinct() { assert_eq(distinct(u2[2]:[1, 1], bool[2]:[true, true]), false) }

#[test]
fn test_distinct_unsigned() {
    let items = u8[4]:[1, 2, 3, 2];
    let valid = bool[4]:[true, true, true, true];
    assert_eq(distinct(items, valid), false);
}

#[test]
fn test_distinct_signed() {
    let items = s8[3]:[-1, 0, 1];
    let valid = bool[3]:[true, true, true];
    assert_eq(distinct(items, valid), true);
}

#[test]
fn test_distinct_with_invalid() {
    let items = u8[4]:[1, 2, 3, 1];
    let valid = bool[4]:[true, true, true, false];
    assert_eq(distinct(items, valid), true);
}

#[quickcheck]
fn quickcheck_forced_duplicate(xs: u4[4], to_dupe: u2) -> bool {
    const ALL_VALID = bool[4]:[true, ...];
    let forced_dupe = update(xs, (to_dupe as u32 + u32:1) % u32:4, xs[to_dupe]);
    distinct(forced_dupe, ALL_VALID) == false
}

#[quickcheck]
fn quickcheck_distinct_all_valid_items_same(value: u4, valid: bool[4]) -> bool {
    let items = u4[4]:[value, ...];  // All items are the same.
    let num_valid = popcount(valid as u4) as u32;

    if num_valid <= u32:1 {
        // With 0 or 1 valid items, they are trivially distinct.
        distinct(items, valid) == true
    } else {
        // Since all valid items are the same, 'distinct' should return false.
        distinct(items, valid) == false
    }
}
