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
pub fn fp32_ldexp(fraction: F32, exp: s32) -> F32 {
  // TODO(jbaileyhandle):  Remove after testing.

  const max_exponent = float32::bias(s8:0) as s33;
  const min_exponent = s33:1 - (float32::bias(s8:0) as s33);

  // Flush subnormal input.
  let fraction = float32::subnormals_to_zero(fraction);

  // Increase the exponent of fraction by 'exp'.
  // If this was not a DAZ module, we'd have to deal
  // with denormal 'fraction' here.
  let exp = signex(exp, s33:0)
              + signex(float32::unbiased_exponent(fraction), s33:0);
  let result = F32 {sign: fraction.sign,
                    bexp: float32::bias(exp as s8),
                    fraction: fraction.fraction };

  // Handle overflow.
  let result = if exp > max_exponent { float32::inf(fraction.sign) }
               else { result };

  // Hanlde underflow, taking into account the case that underflow
  // rounds back up to a normal number.
  // If this was not a DAZ module, we'd have to deal
  // with denormal 'result' here.
  let underflow_result =
    if exp == (min_exponent - s33:1) && fraction.fraction == std::mask_bits<u32:23>() {
      F32{sign: fraction.sign, bexp: u8:1, fraction: u23:0}
    } else {
      float32::zero(fraction.sign)
    };
  let result = if exp < min_exponent { underflow_result }
               else { result };
  // Flush subnormal output.
  let result = float32::subnormals_to_zero(result);

  // Handle special cases.
  let result = if float32::is_zero_or_subnormal(fraction) || float32::is_inf(fraction) { fraction }
               else { result };
  let result = if float32::is_nan(fraction) { float32::qnan() }
               else { result };
  result
}

#[test]
fn fp32_ldexp_test() {
  // Test Special cases.
  let _ = assert_eq(fp32_ldexp(float32::zero(u1:0), s32:1),
    float32::zero(u1:0));
  let _ = assert_eq(fp32_ldexp(float32::zero(u1:1), s32:1),
    float32::zero(u1:1));
  let _ = assert_eq(fp32_ldexp(float32::inf(u1:0), s32:-1),
    float32::inf(u1:0));
  let _ = assert_eq(fp32_ldexp(float32::inf(u1:1), s32:-1),
    float32::inf(u1:1));
  let _ = assert_eq(fp32_ldexp(float32::qnan(), s32:1),
    float32::qnan());

  // Subnormal input.
  let pos_denormal = F32{sign: u1:0, bexp: u8:0, fraction: u23:99};
  let _ = assert_eq(fp32_ldexp(pos_denormal, s32:1),
    float32::zero(u1:0));
  let neg_denormal = F32{sign: u1:1, bexp: u8:0, fraction: u23:99};
  let _ = assert_eq(fp32_ldexp(neg_denormal, s32:1),
    float32::zero(u1:1));

  // Output subnormal, flush to zero.
  let almost_denormal = F32{sign: u1:0, bexp: u8:1, fraction: u23:99};
  let _ = assert_eq(fp32_ldexp(pos_denormal, s32:-1),
    float32::zero(u1:0));

  // Subnormal result rounds up to normal number.
  let frac = F32{sign: u1:0, bexp: u8:10, fraction: u23:0x7fffff};
  let expected = F32{sign: u1:0, bexp: u8:1, fraction: u23:0};
  let _ = assert_eq(fp32_ldexp(frac, s32:-10), expected);
  let frac = F32{sign: u1:1, bexp: u8:10, fraction: u23:0x7fffff};
  let expected = F32{sign: u1:1, bexp: u8:1, fraction: u23:0};
  let _ = assert_eq(fp32_ldexp(frac, s32:-10), expected);

  // Large positive input exponents.
  let frac = F32{sign: u1:0, bexp: u8:128, fraction: u23:0x0};
  let expected = float32::inf(u1:0);
  let _ = assert_eq(fp32_ldexp(frac, s32:0x7FFFFFFF - s32:1), expected);
  let frac = F32{sign: u1:0, bexp: u8:128, fraction: u23:0x0};
  let expected = float32::inf(u1:0);
  let _ = assert_eq(fp32_ldexp(frac, s32:0x7FFFFFFF), expected);
  let frac = F32{sign: u1:1, bexp: u8:128, fraction: u23:0x0};
  let expected = float32::inf(u1:1);
  let _ = assert_eq(fp32_ldexp(frac, s32:0x7FFFFFFF - s32:1), expected);
  let frac = F32{sign: u1:1, bexp: u8:128, fraction: u23:0x0};
  let expected = float32::inf(u1:1);
  let _ = assert_eq(fp32_ldexp(frac, s32:0x7FFFFFFF), expected);

  // Large negative input exponents.
  let frac = F32{sign: u1:0, bexp: u8:126, fraction: u23:0x0};
  let expected = float32::zero(u1:0);
  let _ = assert_eq(fp32_ldexp(frac, s32:0x80000000 + s32:0x1), expected);
  let frac = F32{sign: u1:0, bexp: u8:126, fraction: u23:0x0};
  let expected = float32::zero(u1:0);
  let _ = assert_eq(fp32_ldexp(frac, s32:0x80000000), expected);
  let frac = F32{sign: u1:1, bexp: u8:126, fraction: u23:0x0};
  let expected = float32::zero(u1:1);
  let _ = assert_eq(fp32_ldexp(frac, s32:0x80000000 + s32:0x1), expected);
  let frac = F32{sign: u1:1, bexp: u8:126, fraction: u23:0x0};
  let expected = float32::zero(u1:1);
  let _ = assert_eq(fp32_ldexp(frac, s32:0x80000000), expected);

  // Other large exponents from reported bug #462.
  let frac = float32::unflatten(u32:0xd3fefd2b);
  let expected = float32::inf(u1:1);
  let _ = assert_eq(fp32_ldexp(frac, s32:0x7ffffffd), expected);
  let frac = float32::unflatten(u32:0x36eba93e);
  let expected = float32::zero(u1:0);
  let _ = assert_eq(fp32_ldexp(frac, s32:0x80000010), expected);
  let frac = float32::unflatten(u32:0x8a87c096);
  let expected = float32::zero(u1:1);
  let _ = assert_eq(fp32_ldexp(frac, s32:0x80000013), expected);
  let frac = float32::unflatten(u32:0x71694e37);
  let expected = float32::inf(u1:0);
  let _ = assert_eq(fp32_ldexp(frac, s32:0x7fffffbe), expected);

  ()
}
