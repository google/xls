// MachSuite (BSD-3) License
//
// Copyright (c) 2014-2015, the President and Fellows of Harvard College.
// Copyright 2021 The XLS Authors
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the name of Harvard University nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Performs a FFT_SIZE-input fast fourier transfrom.
// Based on:
// https://github.com/breagen/MachSuite/blob/master/fft/strided/fft.c

import float32
import std
import third_party.xls_machsuite.fft.test_data.dslx_test_data
import xls.modules.fp.fp32_add_2
import xls.modules.fp.fp32_sub_2
import xls.modules.fp.fp32_mul_2

type F32 = float32::F32;

const FFT_SIZE = u32:1024;
const FFT_HALF_SIZE = u32:512;

fn fft_root_twiddle(root_index: u32, odd: u32, even: u32,
                    real: F32[FFT_SIZE], img: F32[FFT_SIZE],
                    real_twid: F32[FFT_HALF_SIZE],
                    img_twid: F32[FFT_HALF_SIZE])
                    -> (F32[FFT_SIZE], F32[FFT_SIZE]) {
  // Would be neat if we could overload operators for F32
  // or other types.
  let diff = fp32_sub_2::fp32_sub_2(
               fp32_mul_2::fp32_mul_2(real_twid[root_index],
                                      real[odd]),
               fp32_mul_2::fp32_mul_2(img_twid[root_index],
                                      img[odd]));
  let sum = fp32_add_2::fp32_add_2(
               fp32_mul_2::fp32_mul_2(real_twid[root_index],
                                      img[odd]),
               fp32_mul_2::fp32_mul_2(img_twid[root_index],
                                      real[odd]));
  let img = update(img, odd, sum);
  let real = update(real, odd, diff);
  (real, img)
}


// This does the bulk of the work in the body of the nested fft loop.
fn fft_inner_loop_body(log: u32, odd: u32, span: u32, real: F32[FFT_SIZE],
                       img: F32[FFT_SIZE], real_twid: F32[FFT_HALF_SIZE],
                       img_twid: F32[FFT_HALF_SIZE])
                       -> (F32[FFT_SIZE], F32[FFT_SIZE]) {
  let even = odd ^ span;

  let real_sum = fp32_add_2::fp32_add_2(real[even], real[odd]);
  let real_diff = fp32_sub_2::fp32_sub_2(real[even], real[odd]);
  let real = update(real, odd, real_diff);
  let real = update(real, even, real_sum);

  let img_sum = fp32_add_2::fp32_add_2(img[even], img[odd]);
  let img_diff = fp32_sub_2::fp32_sub_2(img[even], img[odd]);
  let img = update(img, odd, img_diff);
  let img = update(img, even, img_sum);

  let root_index = (even << log) & (FFT_SIZE - u32:1);
  if root_index != u32:0 {
    fft_root_twiddle(root_index, odd, even, real, img, real_twid, img_twid)
  } else {
    (real, img)
  }
}

// FFT_SIZE-input fast fourier transform, where inputs
// have already been preprocssed (e.g. sin / cos applied, twiddle
// factors generated) and separated into real and
// imaginary components.
// TODO(jbaileyhandle): Can we make this parametric in the
// number of inputs?
fn fft(real: F32[FFT_SIZE], img: F32[FFT_SIZE], real_twid: F32[FFT_HALF_SIZE],
       img_twid: F32[FFT_HALF_SIZE]) -> (F32[FFT_SIZE], F32[FFT_SIZE]) {

  for (log, (real, img)): (u32, (F32[FFT_SIZE], F32[FFT_SIZE]))
    in range(u32:0, std::clog2(FFT_SIZE)) {
    let span = FFT_SIZE >> log + u32:1;

    for (odd, (real, img)): (u32, (F32[FFT_SIZE], F32[FFT_SIZE]))
      in range(u32:0, u32:1024) {
      // ^  Weirdly, this won't compile if we use FFT_SIZE instead of
      // u32:1024. Build complains about AST node with non-const IR
      // value, even though FFT_SIZE is a const and it is used in the
      // range of the outer loop without issue.

      // We want to loop dynamically, going from span to 1023
      // while skipping half of the numbers in the range.
      // However, dslx restricts us to simply incrementing though a
      // static range. So, we iterate over [0, 1023) and only apply the
      // loop body for the 'real' iterations.
      // After loop unrolling and dead code elimination, these dead
      // iterations should be removed.  However, if we try to reuse the
      // same hardware for each iteration, the module may be idle
      // over half of the time...
      if odd >= span && ((odd & span) != u32:0) {
        fft_inner_loop_body(log, odd, span, real, img, real_twid, img_twid)
      } else {
        (real, img)
      }

    } ((real, img))
  } ((real, img))
}

#[test]
fn test_fft() {
  // Grab and format data.
  let real_in: F32[FFT_SIZE] =
    map(dslx_test_data::REAL_IN_FLAT, float32::unflatten);
  let img_in: F32[FFT_SIZE] =
    map(dslx_test_data::IMG_IN_FLAT, float32::unflatten);
  let real_twid_in: F32[FFT_HALF_SIZE] =
    map(dslx_test_data::REAL_TWID_IN_FLAT, float32::unflatten);
  let img_twid_in: F32[FFT_HALF_SIZE] =
    map(dslx_test_data::IMG_TWID_IN_FLAT, float32::unflatten);
  let real_out_expected: F32[FFT_SIZE] =
    map(dslx_test_data::REAL_OUT_FLAT, float32::unflatten);
  let img_out_expected: F32[FFT_SIZE] =
    map(dslx_test_data::IMG_OUT_FLAT, float32::unflatten);

  // Check fft implementation.
  let (real_out_observed, img_out_observed) = fft(
     real_in,
     img_in,
     real_twid_in,
     img_twid_in);
  let _ = assert_eq(real_out_observed, real_out_expected);
  let _ = assert_eq(img_out_observed, img_out_expected);
  ()
}
