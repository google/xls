// Copyright 2009 The Go Authors. All rights reserved.
// Copyright 2021 The XLS Authors
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This code code is based on the go Payne-Hanek implementation,
// available at: https://golang.org/src/math/trig_reduce.go
// That implementation in turn is based on
// "ARGUMENT REDUCTION FOR HUGE ARGUMENTS: Good to the Last Bit"
// K. C. Ng et al, March 24, 1992
// The license for the go implementation is reproduced below.

// ====================================================
// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
// ====================================================

// This file implements Payne-Hanek range reduction by Pi/4
// for x > 0. It returns the integer part mod 8 (j) and
// the fractional part (z) of x / (Pi/4).
// Note: This module will not give meaningful results for
// x == infinity or x = NaN.

import apfloat;
import float32;
import float64;
import std;

type F32 = float32::F32;
type F64 = float64::F64;
type APFloat = apfloat::APFloat;

// Binary digits of 4/Pi
// Format is 00...001.<mantissa>
const FOUR_DIV_PI_BITS = uN[1280]:0x000000000000000145f306dc9c882a53f84eafa3ea69bb81b6c52b3278872083fca2c757bd778ac36e48dc74849ba5c00c925dd413a32439fc3bd63962534e7dd1046bea5d768909d338e04d68befc827323ac7306a673e93908bf177bf250763ff12fffbc0b301fde5e2316b414da3eda6cfd9e4f96136e9e8c7ecd3cbfd45aea4f758fd7cbe2f67a0e73ef14a525d4d7f6bf623f1aba10ac06608df8f6d757;

// PI / 4.0
const PI_DIV_4_FP64 = float64::unflatten(u64:0x3fe921fb54442d18);
const PI_DIV_4_FP32 = float32::unflatten(u32:0x3f490fdb);

// Pyane-Hanke range reduction by Pi/4. Returns
// the integer part mod 8 and the fractional part
// of x / (Pi/4).
// In other words, calculate x mod 2*PI. Return
// the result in a format where j gives which of
// 8 sectors of radian size PI/4 the reduced
// x falls in and z gives the radian offset into
// this sector (note that this offset can be a
// negative offset into the previous sector).
pub fn fp_trig_reduce<EXP_SZ:u32, SFD_SZ:u32, UEXP_SZ:u32 = {EXP_SZ + u32:1}>(x: APFloat<EXP_SZ, SFD_SZ>, pi_div_4: APFloat<EXP_SZ, SFD_SZ>) -> (u3, APFloat<EXP_SZ, SFD_SZ>) {
  // Check if reduction is necessary.
  let reduction_needed = apfloat::gte_2(x, pi_div_4);

  // Decompose floating point.
  let fixed_x = uN[UEXP_SZ]:1 ++ x.fraction;
  // Extract out the integer and exponent such that,
  // x = fixed_x * 2^exp
  let exp = apfloat::unbiased_exponent<EXP_SZ, SFD_SZ>(x) - (SFD_SZ as sN[EXP_SZ]);

  // Grab 4/Pi bits.
  // Using the exponent of x to grab the appropriate set of
  // bits from FOUR_DIV_PI_BITS to multiply by s.t. the product
  // has exponent -61.
  let exp_sum = (exp + sN[EXP_SZ]:61) as uN[EXP_SZ];
  let pd4_bits = (FOUR_DIV_PI_BITS << exp_sum)[1088:1280];

  // Multiply x mantisssa by extracted FOUR_DIV_PI bits.
  // 192 bits was used for go 64-bit reference implementation.
  // We could probably use fewer bits for 32-bit...
  let prod = (pd4_bits * (fixed_x as uN[192]));

  // Upper 3 bits of hi are j.
  let j = prod[189:192];

  // Extract the fraction and find its magnitude.
  let raw_fraction_bits = prod[61:189];
  let hi = raw_fraction_bits[64:128];
  let lz = clz(hi);
  let fraction_bexp = apfloat::bias(-(sN[EXP_SZ]:1 + (lz as sN[EXP_SZ])));

  // Clear hidden bit and shift mantissa.
  const SFD_INDEX_START = u32:128 - SFD_SZ;
  let fraction_fraction = (raw_fraction_bits << (lz + u64:1))[SFD_INDEX_START as s32:128];

  // Construct fraction.
  let fraction = APFloat<EXP_SZ, SFD_SZ> {sign: u1:0,
                                          bexp: fraction_bexp,
                                          fraction: fraction_fraction};

  // Map zeros to origin (comment from reference go code... very cryptic).
  let (j, fraction) =
    if j[0:1] {
      (j + u3:1,
       apfloat::sub(fraction, apfloat::one<EXP_SZ, SFD_SZ>(u1:0)))
    } else {
      (j, fraction)
    };

  // Multiply fractional part by PI/4
  let fraction = apfloat::mul(fraction, pi_div_4);

  if reduction_needed { (j, fraction) }
  else { (u3:0, x) }
}

pub fn fp_trig_reduce_64(x: F64) -> (u3, F64) {
  fp_trig_reduce<u32:11, u32:52>(x, PI_DIV_4_FP64)
}

pub fn fp_trig_reduce_32(x: F32) -> (u3, F32) {
  fp_trig_reduce<u32:8, u32:23>(x, PI_DIV_4_FP32)
}

// This is just a smoke test validation - thorough validation will be achieved
// for 32 bit trig_reduce by testing 32 bit sin/cos.
#[test]
fn fp_trig_reduce_32_test() {
  let num = float32::unflatten(u32:0x45492679);
  let (j, fraction) = fp_trig_reduce_32(num);
  assert_eq(j, u3:2);
  assert_eq(fraction, float32::unflatten(u32:0xbe20e760));
}

// 64-bit version of the above test for comparison.
#[test]
fn fp_trig_reduce_64_test() {
  let num = float64::unflatten(u64:0x40a924cf20000000);
  let (j, fraction) = fp_trig_reduce_64(num);
  assert_eq(j, u3:2);
  assert_eq(fraction, float64::unflatten(u64:0xbfc41cebad677a69));
}

// Main / entry function that's only used for JIT.
pub fn main(x: F64) -> (u3, F64) {
  fp_trig_reduce_64(x)
}
