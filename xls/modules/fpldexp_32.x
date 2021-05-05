// Copyright 2021 The XLS Authors
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

// This file implements floating point ldexp,
// which calcualtes fraction * 2^exp.
// Note:
//  - Input denormals are treated as/flushed to 0.
//      (denormals-are-zero / DAZ).  Similarly,
//      denormal results are flushed to 0.
//  - No exception flags are raised/reported.
//  - We emit a single, canonical representation for
//      NaN (qnan) but accept all NaN respresentations
//      as input

import std
import float32

type F32 = float32::F32;

// Returns fraction * 2^exp
pub fn fpldexp_32(fraction: F32, exp: s32) -> F32 {
  // TODO(jbaileyhandle):  Remove after testing.

  const max_exponent = float32::bias(u9:0) as s32;
  const min_exponent = s32:1 - (float32::bias(u9:0) as s32);

  // Flush subnormal input.
  let fraction = float32::subnormals_to_zero(fraction);

  // Increase the exponent of fraction by 'exp'.
  // If this was not a DAZ module, we'd have to deal
  // with denormal 'fraction' here.
  let exp = exp + signex(float32::unbiased_exponent(fraction), s32:0);
  let result = F32 {sign: fraction.sign,
                    bexp: float32::bias(exp as u9),
                    sfd: fraction.sfd };

  // Handle overflow.
  let result = float32::inf(fraction.sign) if exp > max_exponent
                                         else result;

  // Hanlde underflow, taking into account the case that underflow
  // rounds back up to a normal number.
  // If this was not a DAZ module, we'd have to deal
  // with denormal 'result' here.
  let underflow_result = F32{sign: fraction.sign, bexp: u8:1, sfd: u23:0}
    if (exp == min_exponent - s32:1) && (fraction.sfd == std::mask_bits<u32:23>())
    else float32::zero(fraction.sign);
  let result = underflow_result if exp < min_exponent
                                          else result;
  // Flush subnormal output.
  let result = float32::subnormals_to_zero(result);

  // Handle special cases.
  let result = fraction if ( float32::is_zero_or_subnormal(fraction)
                         || float32::is_inf(fraction) )
                    else result;
  let result = float32::qnan() if float32::is_nan(fraction)
                               else result;
  result
}

#![test]
fn fpldexp_32_test() {
  // Test Special cases.
  let _ = assert_eq(fpldexp_32(float32::zero(u1:0), s32:1),
    float32::zero(u1:0));
  let _ = assert_eq(fpldexp_32(float32::zero(u1:1), s32:1),
    float32::zero(u1:1));
  let _ = assert_eq(fpldexp_32(float32::inf(u1:0), s32:-1),
    float32::inf(u1:0));
  let _ = assert_eq(fpldexp_32(float32::inf(u1:1), s32:-1),
    float32::inf(u1:1));
  let _ = assert_eq(fpldexp_32(float32::qnan(), s32:1),
    float32::qnan());

  // Subnormal input.
  let pos_denormal = F32{sign: u1:0, bexp: u8:0, sfd: u23:99};
  let _ = assert_eq(fpldexp_32(pos_denormal, s32:1),
    float32::zero(u1:0));
  let neg_denormal = F32{sign: u1:1, bexp: u8:0, sfd: u23:99};
  let _ = assert_eq(fpldexp_32(neg_denormal, s32:1),
    float32::zero(u1:1));

  // Output subnormal, flush to zero.
  let almost_denormal = F32{sign: u1:0, bexp: u8:1, sfd: u23:99};
  let _ = assert_eq(fpldexp_32(pos_denormal, s32:-1),
    float32::zero(u1:0));

  // Subnormal result rounds up to normal number.
  let frac = F32{sign: u1:0, bexp: u8:10, sfd: u23:0x7fffff};
  let expected = F32{sign: u1:0, bexp: u8:1, sfd: u23:0};
  let _ = assert_eq(fpldexp_32(frac, s32:-10), expected);
  let frac = F32{sign: u1:1, bexp: u8:10, sfd: u23:0x7fffff};
  let expected = F32{sign: u1:1, bexp: u8:1, sfd: u23:0};
  let _ = assert_eq(fpldexp_32(frac, s32:-10), expected);

  ()
}

