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

// bfloat16 routines.
import apfloat

pub type BF16 = apfloat::APFloat<u32:8, u32:7>;
pub type FloatTag = apfloat::APFloatTag;

pub fn qnan() -> BF16 { apfloat::qnan<u32:8, u32:7>() }
pub fn zero(sign: u1) -> BF16 { apfloat::zero<u32:8, u32:7>(sign) }
pub fn one(sign: u1) -> BF16 { apfloat::one<u32:8, u32:7>(sign) }
pub fn inf(sign: u1) -> BF16 { apfloat::inf<u32:8, u32:7>(sign) }
pub fn unbiased_exponent(f: BF16) -> u9 {
  apfloat::unbiased_exponent<u32:8, u32:7>(f)
}
pub fn bias(unbiased_exponent_in: u9) -> u8 {
  apfloat::bias<u32:8, u32:7>(unbiased_exponent_in)
}
pub fn flatten(f: BF16) -> u16 { apfloat::flatten<u32:8, u32:7>(f) }
pub fn unflatten(f: u16) -> BF16 { apfloat::unflatten<u32:8, u32:7>(f) }
pub fn cast_from_fixed<NUM_SRC_BITS:u32>(s: sN[NUM_SRC_BITS]) -> BF16 {
  apfloat::cast_from_fixed<u32:8, u32:7>(s)
}
pub fn cast_to_fixed<NUM_DST_BITS:u32>(to_cast: BF16) -> sN[NUM_DST_BITS] {
  apfloat::cast_to_fixed<NUM_DST_BITS, u32:8, u32:7>(to_cast)
}
pub fn subnormals_to_zero(f: BF16) -> BF16 {
  apfloat::subnormals_to_zero<u32:8, u32:7>(f)
}

pub fn is_inf(f: BF16) -> u1 { apfloat::is_inf<u32:8, u32:7>(f) }
pub fn is_nan(f: BF16) -> u1 { apfloat::is_nan<u32:8, u32:7>(f) }
pub fn is_zero_or_subnormal(f: BF16) -> u1 {
  apfloat::is_zero_or_subnormal<u32:8, u32:7>(f)
}

pub fn eq_2(x: BF16, y: BF16) -> u1 {
  apfloat::eq_2<u32:8, u32:7>(x, y)
}

pub fn gt_2(x: BF16, y: BF16) -> u1 {
  apfloat::gt_2<u32:8, u32:7>(x, y)
}

pub fn gte_2(x: BF16, y: BF16) -> u1 {
  apfloat::gte_2<u32:8, u32:7>(x, y)
}

pub fn lt_2(x: BF16, y: BF16) -> u1 {
  apfloat::lt_2<u32:8, u32:7>(x, y)
}

pub fn lte_2(x: BF16, y: BF16) -> u1 {
  apfloat::lte_2<u32:8, u32:7>(x, y)
}

pub fn normalize(sign:u1, exp: u8, sfd_with_hidden: u8) -> BF16 {
  apfloat::normalize<u32:8, u32:7>(sign, exp, sfd_with_hidden)
}

pub fn tag(f: BF16) -> FloatTag {
  apfloat::tag(f)
}

// Increments the significand of the input BF16 by one and returns the
// normalized result. Input must be a normal *non-zero* number.
pub fn increment_sfd(input: BF16) -> BF16 {
  // Add the hidden bit and one (the increment amount) to the significand. If it
  // overflows 8 bits the number must be normalized.
  let new_sfd = u9:0x80 + (input.sfd as u9) + u9:1;
  let new_sfd_msb = new_sfd[8 +: u1];
  match (new_sfd_msb, input.bexp >= u8:0xfe) {
    // Overflow to infinity.
    (true, true) => inf(input.sign),
    // Significand overflowed, normalize.
    (true, false) => BF16 { sign: input.sign,
                            bexp: input.bexp + u8:1,
                            sfd: new_sfd[1 +: u7] },
    // No normalization required.
    (_, _) => BF16 { sign: input.sign,
                     bexp: input.bexp,
                     sfd: new_sfd[:7] },
  }
}

#![test]
fn increment_sfd_bf16_test() {
  // No normalization required.
  let _ = assert_eq(increment_sfd(BF16 { sign: u1:0, bexp: u8:42, sfd: u7:0 }),
                    BF16 { sign: u1:0, bexp: u8:42, sfd: u7:1 });
  let _ = assert_eq(increment_sfd(BF16 { sign: u1:1, bexp: u8:42, sfd: u7:0 }),
                    BF16 { sign: u1:1, bexp: u8:42, sfd: u7:1 });
  let _ = assert_eq(increment_sfd(BF16 { sign: u1:0, bexp: u8:42, sfd: u7:12 }),
                    BF16 { sign: u1:0, bexp: u8:42, sfd: u7:13 });
  let _ = assert_eq(increment_sfd(BF16 { sign: u1:0, bexp: u8:254, sfd: u7:0x3f }),
                    BF16 { sign: u1:0, bexp: u8:254, sfd: u7:0x40 });

  // Normalization required.
  let _ = assert_eq(increment_sfd(BF16 { sign: u1:1, bexp: u8:1, sfd: u7:0x7f }),
                    BF16 { sign: u1:1, bexp: u8:2, sfd: u7:0 });
  let _ = assert_eq(increment_sfd(BF16 { sign: u1:0, bexp: u8:123, sfd: u7:0x7f }),
                    BF16 { sign: u1:0, bexp: u8:124, sfd: u7:0 });

  // Overflow to infinity.
  let _ = assert_eq(increment_sfd(BF16 { sign: u1:0, bexp: u8:254, sfd: u7:0x7f }),
                    inf(u1:0));
  let _ = assert_eq(increment_sfd(BF16 { sign: u1:1, bexp: u8:254, sfd: u7:0x7f }),
                    inf(u1:1));
  ()
}
