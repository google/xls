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

#![feature(type_inference_v2)]

import float32;

type F32 = float32::F32;



pub fn fir_filter_float32<NUM_TAPS:u32, NUM_SAMPLES:u32,
  NUM_OUTPUTS:u32 = {NUM_SAMPLES - NUM_TAPS + u32:1}>
  (samples: F32[NUM_SAMPLES], coefficients: F32[NUM_TAPS])
  -> F32[NUM_OUTPUTS] {

  // Convolve filter coefficients over sample array.
  for(out_idx, fir_output): (u32, F32[NUM_OUTPUTS]) in u32:0..NUM_OUTPUTS {

    // Compute a single output datapoint.
    let point_output = for(tap_idx, acc): (u32, F32) in u32:0..NUM_TAPS {

      let sample_idx = out_idx + NUM_TAPS - tap_idx - u32:1;
      let product = float32::mul(coefficients[tap_idx],
                                           samples[sample_idx]);
      float32::add(acc, product)

    }(float32::zero(u1:0));

    update(fir_output, out_idx, point_output)

  }(F32[NUM_OUTPUTS]:[float32::zero(u1:0),...])
}

pub fn main(samples: F32[6], coefficients: F32[4]) -> F32[3] {
  fir_filter_float32(samples, coefficients)
}
