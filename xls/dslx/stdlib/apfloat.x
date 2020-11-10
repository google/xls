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

// Arbitrary-precision floating point routines.
import std

pub struct APFloat<EXP_SZ:u32, SFD_SZ:u32>  {
  sign: bits[1],  // sign bit
  bexp: bits[EXP_SZ],  // biased exponent
  sfd:  bits[SFD_SZ],  // significand (no hidden bit)
}

pub enum APFloatTag : u3 {
  NAN       = 0,
  INFINITY  = 1,
  SUBNORMAL = 2,
  ZERO      = 3,
  NORMAL    = 4,
}

pub fn qnan<EXP_SZ:u32, SFD_SZ:u32>() -> APFloat<EXP_SZ, SFD_SZ> {
  APFloat<EXP_SZ, SFD_SZ> {
    sign: bits[1]:0,
    bexp: std::mask_bits<EXP_SZ>() as bits[EXP_SZ],
    sfd: bits[SFD_SZ]:1 << ((SFD_SZ - u32:1) as bits[SFD_SZ])
  }
}

test qnan {
  let expected = APFloat<u32:8, u32:23> {
    sign: u1:0, bexp: u8:0xff, sfd: u23:0x400000,
  };
  let actual = qnan<u32:8, u32:23>();
  let _ = assert_eq(actual, expected);

  let expected = APFloat<u32:4, u32:2> {
    sign: u1:0, bexp: u4:0xf, sfd: u2:0x2,
  };
  let actual = qnan<u32:4, u32:2>();
  let _ = assert_eq(actual, expected);
  ()
}

pub fn zero<EXP_SZ:u32, SFD_SZ:u32>(sign: bits[1])
    -> APFloat<EXP_SZ, SFD_SZ> {
  APFloat<EXP_SZ, SFD_SZ>{
    sign: sign,
    bexp: bits[EXP_SZ]:0,
    sfd: bits[SFD_SZ]:0 }
}

test zero {
  let expected = APFloat<u32:8, u32:23> {
    sign: u1:0, bexp: u8:0x0, sfd: u23:0x0,
  };
  let actual = zero<u32:8, u32:23>(u1:0);
  let _ = assert_eq(actual, expected);

  let expected = APFloat<u32:4, u32:2> {
    sign: u1:1, bexp: u4:0x0, sfd: u2:0x0,
  };
  let actual = zero<u32:4, u32:2>(u1:1);
  let _ = assert_eq(actual, expected);
  ()
}

pub fn one<EXP_SZ:u32, SFD_SZ:u32, MASK_SZ:u32 = EXP_SZ - u32:1>(
    sign: bits[1])
    -> APFloat<EXP_SZ, SFD_SZ> {
  APFloat<EXP_SZ, SFD_SZ>{
    sign: sign,
    bexp: std::mask_bits<MASK_SZ>() as bits[EXP_SZ],
    sfd: bits[SFD_SZ]:0
  }
}

test one {
  let expected = APFloat<u32:8, u32:23> {
    sign: u1:0, bexp: u8:0x7f, sfd: u23:0x0,
  };
  let actual = one<u32:8, u32:23>(u1:0);
  let _ = assert_eq(actual, expected);

  let expected = APFloat<u32:4, u32:2> {
    sign: u1:0, bexp: u4:0x7, sfd: u2:0x0,
  };
  let actual = one<u32:4, u32:2>(u1:0);
  let _ = assert_eq(actual, expected);
  ()
}

pub fn inf<EXP_SZ:u32, SFD_SZ:u32>(sign: bits[1]) -> APFloat<EXP_SZ, SFD_SZ> {
  APFloat<EXP_SZ, SFD_SZ>{
    sign: sign,
    bexp: std::mask_bits<EXP_SZ>(),
    sfd: bits[SFD_SZ]:0
  }
}

test inf {
  let expected = APFloat<u32:8, u32:23> {
    sign: u1:0, bexp: u8:0xff, sfd: u23:0x0,
  };
  let actual = inf<u32:8, u32:23>(u1:0);
  let _ = assert_eq(actual, expected);

  let expected = APFloat<u32:4, u32:2> {
    sign: u1:0, bexp: u4:0xf, sfd: u2:0x0,
  };
  let actual = inf<u32:4, u32:2>(u1:0);
  let _ = assert_eq(actual, expected);
  ()
}

// Accessor helpers for the F32 typedef.
pub fn unbiased_exponent<EXP_SZ:u32, SFD_SZ:u32, UEXP_SZ:u32 = EXP_SZ + u32:1, MASK_SZ:u32 = EXP_SZ - u32:1>(
    f: APFloat<EXP_SZ, SFD_SZ>)
    -> bits[UEXP_SZ] {
  (f.bexp as bits[UEXP_SZ]) - (std::mask_bits<MASK_SZ>() as bits[UEXP_SZ])
}

test unbiased_exponent {
  let expected = u9:0x0;
  let actual = unbiased_exponent<u32:8, u32:23>(
      APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0x7f, sfd: u23:0 });
  let _ = assert_eq(actual, expected);
  ()
}

pub fn flatten<EXP_SZ:u32, SFD_SZ:u32, TOTAL_SZ:u32 = u32:1+EXP_SZ+SFD_SZ>(
    x: APFloat<EXP_SZ, SFD_SZ>) -> bits[TOTAL_SZ] {
  x.sign ++ x.bexp ++ x.sfd
}

pub fn unflatten<EXP_SZ:u32, SFD_SZ:u32,
    TOTAL_SZ:u32 = u32:1+EXP_SZ+SFD_SZ,
    SIGN_OFFSET:u32 = EXP_SZ+SFD_SZ>(
    x: bits[TOTAL_SZ]) -> APFloat<EXP_SZ, SFD_SZ> {
  APFloat<EXP_SZ, SFD_SZ>{
      sign: (x >> SIGN_OFFSET) as bits[1],
      bexp: (x >> SFD_SZ) as bits[EXP_SZ],
      sfd: x as bits[SFD_SZ],
  }
}

pub fn subnormals_to_zero<EXP_SZ:u32, SFD_SZ:u32>(
    x: APFloat<EXP_SZ, SFD_SZ>) -> APFloat<EXP_SZ, SFD_SZ> {
  zero(x.sign) if x.bexp == bits[EXP_SZ]:0 else x
}

// Returns a normalized APFloat with the given components. 'sfd_with_hidden' is the
// significand including the hidden bit. This function only normalizes in the
// direction of decreasing the exponent. Input must be a normal number or
// zero. Dernormals are flushed to zero in the result.
pub fn normalize<EXP_SZ:u32, SFD_SZ:u32, WIDE_SFD:u32 = SFD_SZ + u32:1>(
    sign: bits[1], exp: bits[EXP_SZ], sfd_with_hidden: bits[WIDE_SFD])
    -> APFloat<EXP_SZ, SFD_SZ> {
  let leading_zeros = clz(sfd_with_hidden) as bits[SFD_SZ]; // as bits[clog2(SFD_SZ)]?
  let zero_value = zero<EXP_SZ, SFD_SZ>(sign);
  let zero_sfd = WIDE_SFD as bits[SFD_SZ];
  let normalized_sfd = (sfd_with_hidden << (leading_zeros as bits[WIDE_SFD])) as bits[SFD_SZ];

  match (exp <= (leading_zeros as bits[EXP_SZ]), leading_zeros) {
    // Significand is zero.
    (_, zero_sfd) => zero_value,
    // Flush denormals to zero.
    (true, _) => zero_value,
    // Normalize.
    _ => APFloat { sign: sign,
                   bexp: exp - (leading_zeros as bits[EXP_SZ]),
                   sfd: normalized_sfd },
  }
}

test normalize {
  let expected = APFloat<u32:8, u32:23>{
      sign: u1:0, bexp: u8:0x12, sfd: u23:0x7e_dcba };
  let actual = normalize<u32:8, u32:23>(u1:0, u8:0x12, u24:0xfe_dcba);
  let _ = assert_eq(expected, actual);

  let expected = APFloat<u32:8, u32:23>{
      sign: u1:0, bexp: u8:0x0, sfd: u23:0x0 };
  let actual = normalize<u32:8, u32:23>(u1:0, u8:0x1, u24:0x0);
  let _ = assert_eq(expected, actual);

  let expected = APFloat<u32:8, u32:23>{
      sign: u1:0, bexp: u8:0x0, sfd: u23:0x0 };
  let actual = normalize<u32:8, u32:23>(u1:0, u8:0xfe, u24:0x0);
  let _ = assert_eq(expected, actual);

  let expected = APFloat<u32:8, u32:23>{
      sign: u1:1, bexp: u8:77, sfd: u23:0x0 };
  let actual = normalize<u32:8, u32:23>(u1:1, u8:100, u24:1);
  let _ = assert_eq(expected, actual);

  let expected = APFloat<u32:8, u32:23>{
      sign: u1:1, bexp: u8:2, sfd: u23:0b000_1111_0000_0101_0000_0000 };
  let actual = normalize<u32:8, u32:23>(
      u1:1, u8:10, u24:0b0000_0000_1000_1111_0000_0101);
  let _ = assert_eq(expected, actual);

  let expected = APFloat<u32:8, u32:23>{
      sign: u1:1, bexp: u8:10, sfd: u23:0b000_0000_1000_1111_0000_0101};
  let actual = normalize<u32:8, u32:23>(
      u1:1, u8:10, u24:0b1000_0000_1000_1111_0000_0101);
  let _ = assert_eq(expected, actual);

  // Denormals should be flushed to zero.
  let expected = zero<u32:8, u32:23>(u1:1);
  let actual = normalize<u32:8, u32:23>(
      u1:1, u8:5, u24:0b0000_0000_1000_1111_0000_0101);
  let _ = assert_eq(expected, actual);

  let expected = zero<u32:8, u32:23>(u1:0);
  let actual = normalize<u32:8, u32:23>(
      u1:0, u8:2, u24:0b0010_0000_1000_1111_0000_0101);
  let _ = assert_eq(expected, actual);
  ()
}

// Returns whether or not the given APFloat represents an infinite quantity.d
pub fn is_inf<EXP_SZ:u32, SFD_SZ:u32, MASK_SZ:u32 = EXP_SZ - u32:1>(
    x: APFloat<EXP_SZ, SFD_SZ>) -> u1 {
  (x.bexp == std::mask_bits<MASK_SZ>() & x.sfd == bits[SFD_SZ]:0)
}

// Returns whether or not the given F32 represents NaN.
pub fn is_nan<EXP_SZ:u32, SFD_SZ:u32, MASK_SZ:u32 = EXP_SZ - u32:1>(
    x: APFloat<EXP_SZ, SFD_SZ>) -> u1 {
  (x.bexp == std::mask_bits<MASK_SZ>() & x.sfd != bits[SFD_SZ]:0)
}
