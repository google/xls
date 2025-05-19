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

// Parameterized fixed and floating point dot product
// implementations.

import float32;

type F32 = float32::F32;

// Calculates the dot product of fixed-point vectors a and b.
pub fn dot_product_fixed<BITCOUNT: u32, VECTOR_LENGTH: u32>
  (a: sN[BITCOUNT][VECTOR_LENGTH], b: sN[BITCOUNT][VECTOR_LENGTH])
  -> sN[BITCOUNT]{

  for(idx, acc): (u32, sN[BITCOUNT])
    in range (u32:0, VECTOR_LENGTH) {

    let partial_product = a[idx] * b[idx];
    acc + partial_product
  } (sN[BITCOUNT]:0)
}

// Calculates the dot product of 32-bit floating-point
// vectors a and b.
pub fn dot_product_float32<VECTOR_LENGTH: u32>
  (a: F32[VECTOR_LENGTH], b: F32[VECTOR_LENGTH])
  -> F32{

  for(idx, acc): (u32, F32)
    in range (u32:0, VECTOR_LENGTH) {
    let partial_product = float32::mul(a[idx], b[idx]);
    float32::add(acc, partial_product)
  } (float32::zero(u1:0))
}

#[test]
fn dot_product_fixed_test() {
   let a = s32[4]:[1, 2, 3, 4];
   let b = s32[4]:[5, 6, 7, 8];
   let result = dot_product_fixed<u32:32, u32:4>(a, b);
   assert_eq(result, s32:70);

   let a = s8[2]:[1, 2];
   let b = s8[2]:[5, 6];
   let result = dot_product_fixed<u32:8, u32:2>(a, b);
   assert_eq(result, s8:17);
}

#[test]
fn dot_product_float32_test() {
   let a = map(s32[4]:[1, 2, 3, 4], float32::cast_from_fixed_using_rne);
   let b = map(s32[4]:[5, 6, 7, 8], float32::cast_from_fixed_using_rne);
   let result = dot_product_float32(a, b);
   assert_eq(result, float32::cast_from_fixed_using_rne(s32:70));
}
