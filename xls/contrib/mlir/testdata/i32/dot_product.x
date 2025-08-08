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

import std;

// Calculates the dot product of fixed-point vectors a and b.
pub fn dot_product_fixed<BITCOUNT: u32, VECTOR_LENGTH: u32>
  (a: sN[BITCOUNT][VECTOR_LENGTH], b: sN[BITCOUNT][VECTOR_LENGTH])
  -> sN[BITCOUNT]{

  for(idx, acc): (u32, sN[BITCOUNT])
    in u32:0..VECTOR_LENGTH {

    let partial_product = a[idx] * b[idx];
    acc + partial_product
  } (sN[BITCOUNT]:0)
}

fn dot_product_fixed_test(a : s32[4], b: s32[4]) -> s32 {
   dot_product_fixed<u32:32, u32:4>(a, b)
}
