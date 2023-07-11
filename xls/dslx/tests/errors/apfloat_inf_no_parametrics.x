// Copyright 2023 The XLS Authors
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

pub fn mask_bits<X: u32>() -> bits[X] {
  !bits[X]:0
}

pub struct APFloat<EXP_SZ:u32, FRACTION_SZ:u32> {
  sign: bits[1],  // Sign bit.
  bexp: bits[EXP_SZ],  // Biased exponent.
  fraction:  bits[FRACTION_SZ],  // Fractional part (no hidden bit).
}

pub fn inf<EXP_SZ:u32, FRACTION_SZ:u32>(
           sign: bits[1]) -> APFloat<EXP_SZ, FRACTION_SZ> {
  APFloat<EXP_SZ, FRACTION_SZ>{
    sign: sign,
    bexp: mask_bits<EXP_SZ>(),
    fraction: bits[FRACTION_SZ]:0
  }
}

fn main() -> APFloat<8, 24> {
    inf(true)
}
