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

// Parameterized fixed and floating point fir filter
// implementations.  A fir filter, used in signal
// processing, is essentially a convolution of a
// filter of a series of samples.

import std;
import float32;

type F32 = float32::F32;


pub fn fir_filter_fixed<NUM_TAPS:u32, NUM_SAMPLES:u32,
  NUM_OUTPUTS:u32 = {NUM_SAMPLES - NUM_TAPS + u32:1}>
  (samples: s32[NUM_SAMPLES], coefficients: s32[NUM_TAPS])
  -> s32[NUM_OUTPUTS] {

  // Convolve filter coefficients over sample array.
  for(out_idx, fir_output): (u32, s32[NUM_OUTPUTS])
    in range(u32:0, NUM_OUTPUTS) {

    // Compute a single output datapoint.
    // There's probably a way to reuse dot-product here
    // (flatten sample array, bitslice, reverse, repack as array,
    // then pass to dot product?), but it seems like more
    // trouble than it's worth.
    let point_output = for(tap_idx, acc): (u32, s32)
      in range(u32:0, NUM_TAPS) {

      let sample_idx = out_idx + NUM_TAPS - tap_idx - u32:1;
      let product = coefficients[tap_idx] * samples[sample_idx];
      acc + product

    }(s32:0);

    update(fir_output, out_idx, point_output)

  }(s32[NUM_OUTPUTS]:[0,...])
}

pub fn fir_filter_float32<NUM_TAPS:u32, NUM_SAMPLES:u32,
  NUM_OUTPUTS:u32 = {NUM_SAMPLES - NUM_TAPS + u32:1}>
  (samples: F32[NUM_SAMPLES], coefficients: F32[NUM_TAPS])
  -> F32[NUM_OUTPUTS] {

  // Convolve filter coefficients over sample array.
  for(out_idx, fir_output): (u32, F32[NUM_OUTPUTS])
    in range(u32:0, NUM_OUTPUTS) {

    // Compute a single output datapoint.
    let point_output = for(tap_idx, acc): (u32, F32)
      in range(u32:0, NUM_TAPS) {

      let sample_idx = out_idx + NUM_TAPS - tap_idx - u32:1;
      let product = float32::mul(coefficients[tap_idx],
                                           samples[sample_idx]);
      float32::add(acc, product)

    }(float32::zero(u1:0));

    update(fir_output, out_idx, point_output)

  }(F32[NUM_OUTPUTS]:[float32::zero(u1:0),...])
}

#[test]
fn fir_filter_fixed_test() {
   let samples = s32[6]:[1, 2, 3, 4, 5, 6];
   let coefficients = s32[4]:[10, 11, -12, -13];
   let result = fir_filter_fixed(samples, coefficients);
   assert_eq(result, s32[3]:[36, 32, 28]);
}

#[test]
fn fir_filter_float32_test() {
   let samples = map(s32[6]:[1, 2, 3, 4, 5, 6], float32::cast_from_fixed_using_rne);
   let coefficients= map(s32[4]:[10, 11, -12, -13], float32::cast_from_fixed_using_rne);
   let result = fir_filter_float32(samples, coefficients);
   let expected = map(s32[3]:[36, 32, 28], float32::cast_from_fixed_using_rne);
   assert_eq(result, expected);
}
