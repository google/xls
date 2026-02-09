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

// Parameterized fixed point fir filter implementation.
// A fir filter, used in signal processing, is essentially a convolution of a
// filter of a series of samples.

#![feature(type_inference_v2)]


fn saturate(x: s64) -> s32 {
  let max = s64:2147483647;
  let min = s64:-2147483648;
  if x > max { s32:2147483647 }
  else if x < min { s32:-2147483647 - s32:1 }
  else { x as s32 }
}

pub fn fir_filter_fixed32<NUM_TAPS:u32, NUM_SAMPLES:u32,
  NUM_OUTPUTS:u32 = {NUM_SAMPLES - NUM_TAPS + u32:1}>
  (samples: s32[NUM_SAMPLES], coefficients: s32[NUM_TAPS])
  -> s32[NUM_OUTPUTS] {

  let coefficent_sum = for(tap_idx, acc): (u32, s32) in u32:0..NUM_TAPS {
    acc + coefficients[tap_idx]
  }(s32:0);

  // "Normalize" coefficients.
  // We use a simple shift-add dependency to mimic the structure of the FP32 version
  // without needing complex fixed-point division logic.
  let normalized_coefficients = for(tap_idx, acc): (u32, s32[NUM_TAPS]) in u32:0..NUM_TAPS {
    let new_coeff = coefficients[tap_idx] + (coefficent_sum >> u32:4);
    update(acc, tap_idx, new_coeff)
  }(s32[NUM_TAPS]:[0,...]);

  // Convolve filter coefficients over sample array.
  for(out_idx, fir_output): (u32, s32[NUM_OUTPUTS]) in u32:0..NUM_OUTPUTS {

    // Compute a single output datapoint.
    let point_output = for(tap_idx, acc): (u32, s64) in u32:0..NUM_TAPS {

      let sample_idx = out_idx + NUM_TAPS - tap_idx - u32:1;
      let product = (normalized_coefficients[tap_idx] as s64) * (samples[sample_idx] as s64);
      acc + product

    }(s64:0);

    update(fir_output, out_idx, saturate(point_output))

  }(s32[NUM_OUTPUTS]:[0,...])
}

pub fn main(samples: s32[6], coefficients: s32[4]) -> s32[3] {
  fir_filter_fixed32(samples, coefficients)
}
