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

import apfloat;

type APFloat = apfloat::APFloat;

// If true, uses the clzt helper to count leading zeros.
// If false, uses the clz built-in.
pub const USE_CLZT: bool = true;

pub fn split_msbs<MSB_SZ: u32, N: u32, LSB_SZ: u32 = {N - MSB_SZ}>
    (x: uN[N]) -> (uN[MSB_SZ], uN[LSB_SZ]) {
    let msbs: uN[MSB_SZ] = x[-(MSB_SZ as s32):];
    let rest: uN[LSB_SZ] = x[0:LSB_SZ as s32];
    (msbs, rest)
}

pub fn ftz<EXP_SZ: u32, FRACTION_SZ: u32, FRACTION_SZ_P1: u32 = {FRACTION_SZ + u32:1}>
    (bexp: uN[EXP_SZ], fraction: uN[FRACTION_SZ]) -> uN[FRACTION_SZ_P1] {
    if bexp == uN[EXP_SZ]:0 { uN[FRACTION_SZ_P1]:0 } else { u1:1 ++ fraction }
}

pub fn dynamic_mask<WIDTH: u32, N_WIDTH: u32>(n: uN[N_WIDTH]) -> uN[WIDTH] {
    (uN[WIDTH]:1 << n) - uN[WIDTH]:1
}

pub fn is_effective_signed_zero<EXP_SZ: u32, FRACTION_SZ: u32>
    (x: APFloat<EXP_SZ, FRACTION_SZ>) -> bool {
    x.sign && x.bexp == uN[EXP_SZ]:0
}

pub fn add_with_carry_out<WIDTH: u32, WIDTH_P1: u32 = {WIDTH + u32:1}>
    (x: uN[WIDTH], y: uN[WIDTH]) -> uN[WIDTH_P1] {
    x as uN[WIDTH_P1] + y as uN[WIDTH_P1]
}

pub fn is_subnormal<EXP_SZ: u32, FRACTION_SZ: u32>(x: APFloat<EXP_SZ, FRACTION_SZ>) -> bool {
    x.bexp == uN[EXP_SZ]:0 && x.fraction != uN[FRACTION_SZ]:0
}
