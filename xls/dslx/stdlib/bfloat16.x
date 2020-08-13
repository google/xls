// Copyright 2020 Google LLC
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

// "Brain float" routines.

pub struct BF16 {
  sign: u1,  // sign bit
  bexp: u8,  // biased exponent
  sfd:  u7,  // significand (no hidden bit)
}

pub fn qnan() -> BF16 { BF16 { sign: u1:0, bexp: u8:0xff, sfd: u7:0x40 } }
pub fn zero(sign: u1) -> BF16 { BF16 { sign: sign, bexp: u8:0, sfd: u7:0 } }
pub fn inf(sign: u1) -> BF16 { BF16 { sign: sign, bexp: u8:0xff, sfd: u7:0 } }

// Increments the significand of the input BF16 by one and returns the
// normalized result. Input must be a normal *non-zero* number.
pub fn increment_sfd(input: BF16) -> BF16 {
  // Add the hidden bit and one (the increment amount) to the significand. If it
  // overflows 8 bits the number must be normalized.
  let new_sfd = u9:0x80 + (input.sfd as u9) + u9:1;
  let new_sfd_msb = new_sfd[8 +: u1];
  match (new_sfd_msb, input.bexp >= u8:0xfe) {
    // Overflow to infinity.
    (true, true) => inf(input.sign);
    // Significand overflowed, normalize.
    (true, false) => BF16 { sign: input.sign,
                            bexp: input.bexp + u8:1,
                            sfd: new_sfd[1 +: u7] };
    // No normalization required.
    (_, _) => BF16 { sign: input.sign,
                     bexp: input.bexp,
                     sfd: new_sfd[:7] };
  }
}

test increment_sfd_bf16 {
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

