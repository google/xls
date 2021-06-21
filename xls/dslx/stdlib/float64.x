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

// 64-bit floating point routines.
import apfloat

// TODO(rspringer): Make u32:11 and u32:52 symbolic constants. Currently, such
// constants don't propagate correctly and fail to resolve when in parametric
// specifications.
pub type F64 = apfloat::APFloat<u32:11, u32:52>;
pub type FloatTag = apfloat::APFloatTag;

pub type TaggedF64 = (FloatTag, F64);

pub fn qnan() -> F64 {
  apfloat::qnan<u32:11, u32:52>()
}

pub fn zero(sign: u1) -> F64 { apfloat::zero<u32:11, u32:52>(sign) }
pub fn one(sign: u1) -> F64 { apfloat::one<u32:11, u32:52>(sign) }
pub fn inf(sign: u1) -> F64 { apfloat::inf<u32:11, u32:52>(sign) }
pub fn unbiased_exponent(f: F64) -> u12 {
  apfloat::unbiased_exponent<u32:11, u32:52>(f)
}
pub fn bias(unbiased_exponent_in: u12) -> u11 {
  apfloat::bias<u32:11, u32:52>(unbiased_exponent_in)
}
pub fn flatten(f: F64) -> u64 { apfloat::flatten<u32:11, u32:52>(f) }
pub fn unflatten(f: u64) -> F64 { apfloat::unflatten<u32:11, u32:52>(f) }
pub fn cast_from_fixed<NUM_SRC_BITS:u32>(s: sN[NUM_SRC_BITS]) -> F64 {
  apfloat::cast_from_fixed<u32:11, u32:52>(s)
}
pub fn cast_to_fixed<NUM_DST_BITS:u32>(to_cast: F64) -> sN[NUM_DST_BITS] {
  apfloat::cast_to_fixed<NUM_DST_BITS, u32:11, u32:52>(to_cast)
}
pub fn subnormals_to_zero(f: F64) -> F64 {
  apfloat::subnormals_to_zero<u32:11, u32:52>(f)
}

pub fn is_inf(f: F64) -> u1 { apfloat::is_inf<u32:11, u32:52>(f) }
pub fn is_nan(f: F64) -> u1 { apfloat::is_nan<u32:11, u32:52>(f) }
pub fn is_zero_or_subnormal(f: F64) -> u1 {
  apfloat::is_zero_or_subnormal<u32:11, u32:52>(f)
}

pub fn eq_2(x: F64, y: F64) -> u1 {
  apfloat::eq_2<u32:11, u32:52>(x, y)
}

pub fn gt_2(x: F64, y: F64) -> u1 {
  apfloat::gt_2<u32:11, u32:52>(x, y)
}

pub fn gte_2(x: F64, y: F64) -> u1 {
  apfloat::gte_2<u32:11, u32:52>(x, y)
}

pub fn lt_2(x: F64, y: F64) -> u1 {
  apfloat::lt_2<u32:11, u32:52>(x, y)
}

pub fn lte_2(x: F64, y: F64) -> u1 {
  apfloat::lte_2<u32:11, u32:52>(x, y)
}

pub fn normalize(sign:u1, exp: u11, sfd_with_hidden: u53) -> F64 {
  apfloat::normalize<u32:11, u32:52>(sign, exp, sfd_with_hidden)
}

pub fn tag(f: F64) -> FloatTag {
  apfloat::tag<u32:11, u32:52>(f)
}

#![test]
fn normalize_test() {
  let expected = F64 {
      sign: u1:0, bexp: u11:0x2, sfd: u52:0xf_fffe_dcba_0000 };
  let actual = normalize(u1:0, u11:0x12, u53:0x1f_fffe_dcba);
  let _ = assert_eq(expected, actual);

  let expected = F64 {
      sign: u1:0, bexp: u11:0x0, sfd: u52:0x0 };
  let actual = normalize(u1:0, u11:0x1, u53:0x0);
  let _ = assert_eq(expected, actual);

  let expected = F64 {
      sign: u1:0, bexp: u11:0x0, sfd: u52:0x0 };
  let actual = normalize(u1:0, u11:0xfe, u53:0x0);
  let _ = assert_eq(expected, actual);

  let expected = F64 {
      sign: u1:1, bexp: u11:0x4d, sfd: u52:0x0 };
  let actual = normalize(u1:1, u11:0x81, u53:1);
  let _ = assert_eq(expected, actual);
  ()
}

#![test]
fn tag_test() {
  let _ = assert_eq(tag(F64 { sign: u1:0, bexp: u11:0, sfd: u52:0 }), FloatTag::ZERO);
  let _ = assert_eq(tag(F64 { sign: u1:1, bexp: u11:0, sfd: u52:0 }), FloatTag::ZERO);
  let _ = assert_eq(tag(zero(u1:0)), FloatTag::ZERO);
  let _ = assert_eq(tag(zero(u1:1)), FloatTag::ZERO);

  let _ = assert_eq(tag(F64 { sign: u1:0, bexp: u11:0, sfd: u52:1 }), FloatTag::SUBNORMAL);
  let _ = assert_eq(tag(F64 { sign: u1:0, bexp: u11:0, sfd: u52:0x7f_ffff }), FloatTag::SUBNORMAL);

  let _ = assert_eq(tag(F64 { sign: u1:0, bexp: u11:12, sfd: u52:0 }), FloatTag::NORMAL);
  let _ = assert_eq(tag(F64 { sign: u1:1, bexp: u11:u11:0x7fe, sfd: u52:0x7f_ffff }), FloatTag::NORMAL);
  let _ = assert_eq(tag(F64 { sign: u1:1, bexp: u11:1, sfd: u52:1 }), FloatTag::NORMAL);

  let _ = assert_eq(tag(F64 { sign: u1:0, bexp: u11:0x7ff, sfd: u52:0 }), FloatTag::INFINITY);
  let _ = assert_eq(tag(F64 { sign: u1:1, bexp: u11:0x7ff, sfd: u52:0 }), FloatTag::INFINITY);
  let foo = inf(u1:0);
  let _ = trace!(foo);
  let _ = assert_eq(tag(inf(u1:0)), FloatTag::INFINITY);
  let _ = assert_eq(tag(inf(u1:1)), FloatTag::INFINITY);

  let _ = assert_eq(tag(F64 { sign: u1:0, bexp: u11:0x7ff, sfd: u52:1 }), FloatTag::NAN);
  let _ = assert_eq(tag(F64 { sign: u1:1, bexp: u11:0x7ff, sfd: u52:0x7f_ffff }), FloatTag::NAN);
  let _ = assert_eq(tag(qnan()), FloatTag::NAN);
  ()
}

