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
import third_party.xls_kfusion.bilateral_filter_kernel


type F32 = float32::F32;

pub fn bilateral_filter<NUM_ROWS:u32, 
  NUM_COLS:u32, NUM_ELMS:u32 = NUM_ROWS*NUM_COLS>
  (in_img: F32[NUM_ELMS]) -> F32[NUM_ELMS] {

  // Iterate over pixels.
  for(row_idx, out_img): (u32, F32[NUM_ELMS]) 
    in range(u32:0, NUM_ROWS) {
    for(col_idx, out_img): (u32, F32[NUM_ELMS]) 
      in range(u32:0, NUM_COLS) {
      let pixel_idx = row_idx * NUM_COLS + col_idx;

      let window = 
        grab_image_window::grab_image_window<NUM_ROWS, NUM_COLS, bilateral_filter_kernel::RADIUS>(
          in_img, row_idx, col_idx);
      let new_pixel_val = bilateral_filter_kernel::bilateral_filter_kernel(window);

      update(out_img, pixel_idx, new_pixel_val)
    }(out_img)
  }(F32[NUM_ELMS]:[float32::zero(u1:0), ...])
}

// Rounding for testing...
fn round_2_lsb(x: F32) -> F32 {
  float32::round_to_nearest_sfd_idx(x, u32:2)
}

#![test]
fn bilateral_filter_test() {
  let in_img= map(
                  u32[30]:[
                    0x3f800000,
                    0x3f8ccccd,
                    0x00000000,
                    0x3f8ccccd,
                    0x3f800000,
                    0x3f8ccccd,
                    0x3f8ccccd,
                    0x3f800000,
                    0x3f8ccccd,
                    0x3f800000,
                    0x3f8ccccd,
                    0x3f800000,
                    0x3f800000,
                    0x3f8ccccd,
                    0x3f800000,
                    0x3f8ccccd,
                    0x3f800000,
                    0x3f8ccccd,
                    0x4111999a,
                    0x41100000,
                    0x4111999a,
                    0x41100000,
                    0x4111999a,
                    0x41100000,
                    0x41100000,
                    0x4111999a,
                    0x41100000,
                    0x4111999a,
                    0x41100000,
                    0x4111999a,
                  ],
                  float32::unflatten);

  let expected = map(
                  u32[30]:[
                    0x3f835a83,
                    0x3f88051b,
                    0x00000000,
                    0x3f897144,
                    0x3f868209,
                    0x3f89e262,
                    0x3f86a5dd,
                    0x3f84dd4e,
                    0x3f88189e,
                    0x3f861fce,
                    0x3f8923c6,
                    0x3f86a465,
                    0x3f840737,
                    0x3f87f776,
                    0x3f84e4db,
                    0x3f88b724,
                    0x3f85a312,
                    0x3f890cd8,
                    0x4110dba4,
                    0x41108f56,
                    0x4110f4ba,
                    0x4110a4e1,
                    0x41110a45,
                    0x4110bdf8,
                    0x41106694,
                    0x4110eb92,
                    0x41108a51,
                    0x41110f48,
                    0x4110ae08,
                    0x41113306,
                  ],
                  float32::unflatten);
  let expected = map(expected, round_2_lsb);

  let actual = map(bilateral_filter<u32:5, u32:6>(in_img), round_2_lsb);
  let idx = u32:0;
  let expected_0 = expected[idx];
  let actual_0 = actual[idx];
  let _ = assert_eq(expected_0, actual_0);
  ()
}
