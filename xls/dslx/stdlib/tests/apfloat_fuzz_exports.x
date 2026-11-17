// Copyright 2026 The XLS Authors
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

// Packed-bit entrypoints wrapping apfloat stdlib operations across multiple
// floating-point formats for AOT-compiled C++ FuzzTest oracle testing.

import apfloat;
import float32;

// E4M3 (IEEE-754 S1E4M3)
pub fn e4m3_add(a: u8, b: u8) -> u8 {
    apfloat::flatten(apfloat::add(apfloat::unflatten<4, 3>(a), apfloat::unflatten<4, 3>(b)))
}

pub fn e4m3_mul(a: u8, b: u8) -> u8 {
    apfloat::flatten(apfloat::mul(apfloat::unflatten<4, 3>(a), apfloat::unflatten<4, 3>(b)))
}

pub fn e4m3_fma(a: u8, b: u8, c: u8) -> u8 {
    apfloat::flatten(
        apfloat::fma(
            apfloat::unflatten<4, 3>(a), apfloat::unflatten<4, 3>(b), apfloat::unflatten<4, 3>(c)))
}

pub fn e4m3_from_s64_rne(x: s64) -> u8 {
    apfloat::flatten(apfloat::cast_from_fixed_using_rne<4, 3>(x))
}

pub fn e4m3_from_s64_rz(x: s64) -> u8 {
    apfloat::flatten(apfloat::cast_from_fixed_using_rz<4, 3>(x))
}

// BF16 (IEEE-754 S1E8M7)
pub fn bf16_add(a: u16, b: u16) -> u16 {
    apfloat::flatten(apfloat::add(apfloat::unflatten<8, 7>(a), apfloat::unflatten<8, 7>(b)))
}

pub fn bf16_add_lza(a: u16, b: u16) -> u16 {
    apfloat::flatten(
        apfloat::add<8, 7, true>(apfloat::unflatten<8, 7>(a), apfloat::unflatten<8, 7>(b)))
}

pub fn bf16_sub(a: u16, b: u16) -> u16 {
    apfloat::flatten(apfloat::sub(apfloat::unflatten<8, 7>(a), apfloat::unflatten<8, 7>(b)))
}

pub fn bf16_mul(a: u16, b: u16) -> u16 {
    apfloat::flatten(apfloat::mul(apfloat::unflatten<8, 7>(a), apfloat::unflatten<8, 7>(b)))
}

pub fn bf16_fma(a: u16, b: u16, c: u16) -> u16 {
    apfloat::flatten(
        apfloat::fma(
            apfloat::unflatten<8, 7>(a), apfloat::unflatten<8, 7>(b), apfloat::unflatten<8, 7>(c)))
}

pub fn bf16_ldexp(a: u16, exp: s32) -> u16 {
    apfloat::flatten(apfloat::ldexp(apfloat::unflatten<8, 7>(a), exp))
}

pub fn bf16_from_s32_rne(x: s32) -> u16 {
    apfloat::flatten(apfloat::cast_from_fixed_using_rne<8, 7>(x))
}

pub fn bf16_from_s32_rz(x: s32) -> u16 {
    apfloat::flatten(apfloat::cast_from_fixed_using_rz<8, 7>(x))
}

pub fn bf16_to_s32(a: u16) -> s32 { apfloat::to_int<8, 7, 32>(apfloat::unflatten<8, 7>(a)) }

pub fn bf16_to_u32(a: u16) -> u32 { apfloat::to_uint<32>(apfloat::unflatten<8, 7>(a)) }

pub fn bf16_cmp(a: u16, b: u16) -> u8 {
    let x = apfloat::unflatten<8, 7>(a);
    let y = apfloat::unflatten<8, 7>(b);
    u3:0 ++ (apfloat::gte_2(x, y) as u1) ++ (apfloat::gt_2(x, y) as u1) ++
    (apfloat::lte_2(x, y) as u1) ++ (apfloat::lt_2(x, y) as u1) ++ (apfloat::eq_2(x, y) as u1)
}

pub fn bf16_ceil_with_denorms(a: u16) -> u16 {
    apfloat::flatten(apfloat::ceil_with_denorms(apfloat::unflatten<8, 7>(a)))
}

pub fn bf16_ceil_daz(a: u16) -> u16 {
    apfloat::flatten(apfloat::ceil_daz(apfloat::unflatten<8, 7>(a)))
}

pub fn bf16_floor_with_denorms(a: u16) -> u16 {
    apfloat::flatten(apfloat::floor_with_denorms(apfloat::unflatten<8, 7>(a)))
}

pub fn bf16_floor_daz(a: u16) -> u16 {
    apfloat::flatten(apfloat::floor_daz(apfloat::unflatten<8, 7>(a)))
}

pub fn bf16_trunc(a: u16) -> u16 { apfloat::flatten(apfloat::trunc(apfloat::unflatten<8, 7>(a))) }

pub fn bf16_round_rne(a: u16) -> u16 {
    apfloat::flatten(
        apfloat::round<8, 7, apfloat::RoundStyle::TIES_TO_EVEN>(apfloat::unflatten<8, 7>(a)))
}

pub fn bf16_round_rna(a: u16) -> u16 {
    apfloat::flatten(
        apfloat::round<8, 7, apfloat::RoundStyle::TIES_TO_AWAY>(apfloat::unflatten<8, 7>(a)))
}

// HF16 (IEEE-754 binary16, S1E5M10)
pub fn hf16_add(a: u16, b: u16) -> u16 {
    apfloat::flatten(apfloat::add(apfloat::unflatten<5, 10>(a), apfloat::unflatten<5, 10>(b)))
}

pub fn hf16_mul(a: u16, b: u16) -> u16 {
    apfloat::flatten(apfloat::mul(apfloat::unflatten<5, 10>(a), apfloat::unflatten<5, 10>(b)))
}

pub fn hf16_fma(a: u16, b: u16, c: u16) -> u16 {
    apfloat::flatten(
        apfloat::fma(
            apfloat::unflatten<5, 10>(a), apfloat::unflatten<5, 10>(b), apfloat::unflatten<5, 10>(c)))
}

pub fn hf16_from_s32_rne(x: s32) -> u16 {
    apfloat::flatten(apfloat::cast_from_fixed_using_rne<5, 10>(x))
}

pub fn hf16_from_s32_rz(x: s32) -> u16 {
    apfloat::flatten(apfloat::cast_from_fixed_using_rz<5, 10>(x))
}

// F32 (IEEE-754 binary32, S1E8M23) and cross-format casts
pub fn f32_add(a: u32, b: u32) -> u32 {
    apfloat::flatten(apfloat::add(apfloat::unflatten<8, 23>(a), apfloat::unflatten<8, 23>(b)))
}

pub fn f32_add_lza(a: u32, b: u32) -> u32 {
    apfloat::flatten(
        apfloat::add<8, 23, true>(apfloat::unflatten<8, 23>(a), apfloat::unflatten<8, 23>(b)))
}

pub fn f32_mul(a: u32, b: u32) -> u32 {
    apfloat::flatten(apfloat::mul(apfloat::unflatten<8, 23>(a), apfloat::unflatten<8, 23>(b)))
}

pub fn f32_fma(a: u32, b: u32, c: u32) -> u32 {
    apfloat::flatten(
        apfloat::fma(
            apfloat::unflatten<8, 23>(a), apfloat::unflatten<8, 23>(b), apfloat::unflatten<8, 23>(c)))
}

pub fn f32_ldexp(a: u32, exp: s32) -> u32 {
    apfloat::flatten(apfloat::ldexp(apfloat::unflatten<8, 23>(a), exp))
}

pub fn f32_from_s64_rne(x: s64) -> u32 {
    apfloat::flatten(apfloat::cast_from_fixed_using_rne<8, 23>(x))
}

pub fn f32_from_s64_rz(x: s64) -> u32 {
    apfloat::flatten(apfloat::cast_from_fixed_using_rz<8, 23>(x))
}

pub fn f32_to_s32(a: u32) -> s32 { apfloat::to_int<8, 23, 32>(apfloat::unflatten<8, 23>(a)) }

pub fn f32_to_u32(a: u32) -> u32 { apfloat::to_uint<32>(apfloat::unflatten<8, 23>(a)) }

pub fn f32_cast_to_fixed_s32(a: u32) -> s32 {
    apfloat::cast_to_fixed<32>(apfloat::unflatten<8, 23>(a))
}

pub fn f32_from_int32(x: s32) -> u32 { float32::flatten(float32::from_int32(x)) }

pub fn f32_from_uint32(x: u32) -> u32 { float32::flatten(float32::from_uint32(x)) }

pub fn f32_fast_rsqrt(a: u32) -> u32 {
    float32::flatten(float32::fast_rsqrt(float32::unflatten(a)))
}

pub fn f32_to_bf16_downcast_rne(a: u32) -> u16 {
    apfloat::flatten(apfloat::downcast_rne<7, 8>(apfloat::unflatten<8, 23>(a)))
}

pub fn f32_to_bf16_downcast_rna(a: u32) -> u16 {
    apfloat::flatten(
        apfloat::downcast<7, 8>(apfloat::unflatten<8, 23>(a), apfloat::RoundStyle::TIES_TO_AWAY))
}

pub fn f32_to_hf16_downcast_rne(a: u32) -> u16 {
    apfloat::flatten(apfloat::downcast_rne<10, 5>(apfloat::unflatten<8, 23>(a)))
}

pub fn f32_to_hf16_downcast(a: u32) -> u16 {
    apfloat::flatten(
        apfloat::downcast<10, 5>(apfloat::unflatten<8, 23>(a), apfloat::RoundStyle::TIES_TO_EVEN))
}

pub fn bf16_to_f32_upcast_denorms(a: u16) -> u32 {
    apfloat::flatten(apfloat::upcast_with_denorms<8, 23>(apfloat::unflatten<8, 7>(a)))
}

pub fn bf16_to_f32_upcast_daz(a: u16) -> u32 {
    apfloat::flatten(apfloat::upcast_daz<8, 23>(apfloat::unflatten<8, 7>(a)))
}

pub fn hf16_to_f32_upcast_denorms(a: u16) -> u32 {
    apfloat::flatten(apfloat::upcast_with_denorms<8, 23>(apfloat::unflatten<5, 10>(a)))
}

// F64 (IEEE-754 binary64, S1E11M52)
pub fn f64_add(a: u64, b: u64) -> u64 {
    apfloat::flatten(apfloat::add(apfloat::unflatten<11, 52>(a), apfloat::unflatten<11, 52>(b)))
}

pub fn f64_mul(a: u64, b: u64) -> u64 {
    apfloat::flatten(apfloat::mul(apfloat::unflatten<11, 52>(a), apfloat::unflatten<11, 52>(b)))
}

pub fn f64_fma(a: u64, b: u64, c: u64) -> u64 {
    apfloat::flatten(
        apfloat::fma(
            apfloat::unflatten<11, 52>(a), apfloat::unflatten<11, 52>(b),
            apfloat::unflatten<11, 52>(c)))
}
