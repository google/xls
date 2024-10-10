// Copyright 2024 The XLS Authors
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

// Convenience wrappers for apfloat abs.


import apfloat;

pub const BF16_EXP_SZ = u32:8;  // Exponent bits
pub const BF16_FRACTION_SZ = u32:7;  // Fraction bits

pub const F32_EXP_SZ = u32:8;  // Exponent bits
pub const F32_FRACTION_SZ = u32:23;  // Fraction bits

type BF16 = apfloat::APFloat<BF16_EXP_SZ, BF16_FRACTION_SZ>;
type F32 = apfloat::APFloat<F32_EXP_SZ, F32_FRACTION_SZ>;

pub fn abs_bf16(x: BF16) -> BF16 {
  apfloat::abs(x)
}

pub fn abs_f32(x: F32) -> F32 {
  apfloat::abs(x)
}
