// Copyright 2011-2013 Gerhard Reitmayr, TU Graz. All rights reserved.
// Copyright 2021 The XLS Authors
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file.

// This code is based on the KinnectFusion implementation
// available at: https://github.com/GerhardR/kfusion/blob/master/kfusion.cu

// Bilateral filter is a noise-reduction image filter that
// preserves edges.

import std
import float32
import xls.modules.fpadd_2x32
import xls.modules.fpmul_2x32
import xls.modules.fpsub_2x32
import third_party.xls_berkeley_softfloat.fpdiv_2x32
import third_party.xls_go_math.fpexp_32
import third_party.xls_kfusion.grab_image_window


type F32 = float32::F32;

pub const RADIUS = u32:2;
const WINDOW_WIDTH = RADIUS * u32:2 + u32:1;
const WINDOW_SIZE = WINDOW_WIDTH * WINDOW_WIDTH;
const GAUSSIAN_SIZE = WINDOW_WIDTH;
const DELTA = s32:4;
const E_DELTA = float32::unflatten(u32:0x3dcccccd); // 0.1f
const E_D_SQUARED_2 = fpmul_2x32::fpmul_2x32(
                 fpmul_2x32::fpmul_2x32(E_DELTA, E_DELTA),
                 float32::cast_from_fixed(s32:2)
               );

// Produces a guassian distribution.
// Could just input computed values as a constant,
// but this seems more transparent (values will still
// be computed statically).
fn generate_gaussian() -> F32[GAUSSIAN_SIZE] {
  for(idx, gaussian) : (u32, F32[GAUSSIAN_SIZE])
    in range(u32:0, GAUSSIAN_SIZE) {
    // gaussian[idx] = epxf(-(x*x) / (2*delta*delta))
    let x = (idx as s32) - s32:2;
    let x_f32 = float32::cast_from_fixed(x);
    let x_sq_f32 = fpmul_2x32::fpmul_2x32(x_f32, x_f32);
    let neg_x_sq_f32 = F32{sign: u1:1, .. x_sq_f32};

    let delta_sq_two_f32 = float32::cast_from_fixed(DELTA * DELTA * s32:2);
    
    let division = fpdiv_2x32::fpdiv_2x32(neg_x_sq_f32, delta_sq_two_f32);
    let gaussian_value = fpexp_32::fpexp_32(division);

    update(gaussian, idx as u32, gaussian_value)

  }(F32[GAUSSIAN_SIZE]:[float32::zero(u1:0),...])
}
const GAUSSIAN = generate_gaussian();

// Perform bilateral filter for a single image stencil / window.
pub fn bilateral_filter_kernel(in_window: F32[WINDOW_SIZE]) -> F32 {
  let center = in_window[WINDOW_SIZE >> u32:1];
  let (sum, t) = for(row_idx, (sum, t)): (u32, (F32, F32)) in range(u32:0, WINDOW_WIDTH) {
    for(col_idx, (sum, t)): (u32, (F32, F32)) in range(u32:0, WINDOW_WIDTH) {
      let pixel_idx = row_idx * WINDOW_WIDTH + col_idx;
      let pixel_val = in_window[pixel_idx];

      // mod = (pixel_val-center)^2
      let diff = fpsub_2x32::fpsub_2x32(pixel_val, center);
      let diff_sq = fpmul_2x32::fpmul_2x32(diff, diff);

      // factor = gaussian[row] * gaussian[col] * expf(-mod / E_D_SQUARED_2) 
      let neg_diff_sq = F32{sign: u1:1, ..diff_sq};
      let divide = fpdiv_2x32::fpdiv_2x32(neg_diff_sq, E_D_SQUARED_2);
      let expf = fpexp_32::fpexp_32(divide);
      let factor = fpmul_2x32::fpmul_2x32(
                     GAUSSIAN[row_idx],
                     fpmul_2x32::fpmul_2x32(
                        GAUSSIAN[col_idx],
                        expf
                      )
                    );

      // If pixel_val > 0,
      // t += factor * pixel_val
      // sum +- factor
      let t_add = fpmul_2x32::fpmul_2x32(factor, pixel_val);
      let new_t = fpadd_2x32::fpadd_2x32(t, t_add);
      let new_sum = fpadd_2x32::fpadd_2x32(sum, factor);
      let gt_zero = !float32::is_zero_or_subnormal(pixel_val)
                    && !pixel_val.sign;
      (new_sum, new_t) if gt_zero
               else (sum, t)
    } ((sum, t))
  } ((float32::zero(u1:0), float32::zero(u1:0)));

  fpdiv_2x32::fpdiv_2x32(t, sum)
}

#![test]
fn bilateral_filter_kernel_test() {
  let window = map(
                      u32[25]:[
                      0x3f800000,
                      0x3f8ccccd,
                      0x00000000,
                      0x3f8ccccd,
                      0x3f800000,
                      0x3f8ccccd,
                      0x3f800000,
                      0x3f8ccccd,
                      0x3f800000,
                      0x3f8ccccd,
                      0x3f800000,
                      0x3f8ccccd,
                      0x3f800000,
                      0x3f8ccccd,
                      0x3f800000,
                      0x4111999a,
                      0x41100000,
                      0x4111999a,
                      0x41100000,
                      0x4111999a,
                      0x41100000,
                      0x4111999a,
                      0x41100000,
                      0x4111999a,
                      0x41100000,
                      ],
                      float32::unflatten);

  let expected = float32::unflatten(u32:0x3f84e4db);
  let actual = bilateral_filter_kernel(window);
  let _ = assert_eq(expected, actual);
  ()
}
