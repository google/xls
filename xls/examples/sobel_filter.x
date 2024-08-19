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

// Floating point sobel filter implementation (fixed-point is TODO).
// A sobel filter, used in image and video processing, applies stencils
// on an image to sharpen edges.

import std;
import float32;
import third_party.xls_go_math.fpsqrt_32;

type F32 = float32::F32;

// Sobel stencils.
const X_STENCIL = s32[3][3]:[s32[3]:[-1,  0,  1],
                             s32[3]:[-2,  0,  2],
                             s32[3]:[-1,  0,  1]];
const Y_STENCIL = s32[3][3]:[s32[3]:[-1, -2, -1],
                             s32[3]:[ 0,  0,  0],
                             s32[3]:[ 1,  2,  1]];

// TODO(jbaileyhandle): Do we have a way to reshape multidimensional
// arrays or to apply map to multidemnsional arrays?
fn convert_triplet(input: s32[3]) -> F32[3] {
  map(input, float32::cast_from_fixed_using_rne)
}
const X_STENCIL_F32 = map(X_STENCIL, convert_triplet);
const Y_STENCIL_F32 = map(Y_STENCIL, convert_triplet);

// Apply a stencil 'stencil' on the image 'in_img' to calculate the pixel
// at row 'row_idx', column 'col_idx' in the output image.
pub fn apply_stencil_float32<NUM_ROWS:u32, NUM_COLS:u32, NUM_ELMS:u32 = {NUM_ROWS*NUM_COLS}>(
  in_img: F32[NUM_ELMS],
  row_idx:u32, col_idx:u32, stencil: F32[3][3]) -> F32 {

  for(row_offset, sum): (u32, F32)
    in range(u32:0, u32:3) {

    let img_row_idx = row_idx + row_offset;
    for(col_offset, sum): (u32, F32)
      in range(u32:0, u32:3) {
      let img_col_idx = col_idx + col_offset;

      let img_idx = img_row_idx*NUM_COLS + img_col_idx;
      let prod = float32::mul(
                    in_img[img_idx],
                    stencil[row_offset][col_offset]);

      // TODO(jbaileyhandle): This would be more efficient
      // as a reduction tree rather than a sequence of adds.
      float32::add(sum, prod)
    }(sum)
  }(float32::zero(u1:0))
}

#[test]
fn apply_stencil_float32_test() {
  // Do we have a way to reshape a 9-element array to a 3x3?
  let img1 = map(s32[16]:[1, 1, 1, 1,
                          1, -1, -2, 1,
                          1, -4, -3, 1,
                          1, 1, 1, 1], float32::cast_from_fixed_using_rne);

  assert_eq(apply_stencil_float32<u32:4, u32:4>(img1, u32:0, u32:0, X_STENCIL_F32),
                   float32::cast_from_fixed_using_rne(s32:-10));
  assert_eq(apply_stencil_float32<u32:4, u32:4>(img1, u32:0, u32:1, X_STENCIL_F32),
                   float32::cast_from_fixed_using_rne(s32:9));
  assert_eq(apply_stencil_float32<u32:4, u32:4>(img1, u32:1, u32:0, X_STENCIL_F32),
                   float32::cast_from_fixed_using_rne(s32:-11));
  assert_eq(apply_stencil_float32<u32:4, u32:4>(img1, u32:1, u32:1, X_STENCIL_F32),
                  float32::cast_from_fixed_using_rne(s32:12));

  assert_eq(apply_stencil_float32<u32:4, u32:4>(img1, u32:0, u32:0, Y_STENCIL_F32),
                   float32::cast_from_fixed_using_rne(s32:-14));
  assert_eq(apply_stencil_float32<u32:4, u32:4>(img1, u32:0, u32:1, Y_STENCIL_F32),
                   float32::cast_from_fixed_using_rne(s32:-13));
  assert_eq(apply_stencil_float32<u32:4, u32:4>(img1, u32:1, u32:0, Y_STENCIL_F32),
                   float32::cast_from_fixed_using_rne(s32:7));
  assert_eq(apply_stencil_float32<u32:4, u32:4>(img1, u32:1, u32:1, Y_STENCIL_F32),
                  float32::cast_from_fixed_using_rne(s32:8));
}

// Apply Sobel filter to input image 'in_img' with NUM_ROWS rows
// and NUM_COLS columns.  Output image has NUM_ROWS-2 rows and
// NUM_COLS-2 columns.
pub fn sobel_filter_float32<NUM_ROWS:u32, NUM_COLS:u32, NUM_ELMS:u32 = {NUM_ROWS*NUM_COLS},
  NUM_ROWS_OUT:u32 = {NUM_ROWS-u32:2}, NUM_COLS_OUT:u32 = {NUM_COLS-u32:2},
  NUM_ELMS_OUT:u32 = {NUM_ROWS_OUT*NUM_COLS_OUT}>
  (in_img: F32[NUM_ELMS]) -> F32[NUM_ELMS_OUT]{

  // Iterate over output pixels.
  for(out_row_idx, out_img): (u32, F32[NUM_ELMS_OUT])
    in range(u32:0, NUM_ROWS_OUT) {
    for(out_col_idx, out_img): (u32, F32[NUM_ELMS_OUT])
      in range(u32:0, NUM_COLS_OUT) {

      // Iterate over stencil.
      let x_stencil_out = apply_stencil_float32<NUM_ROWS, NUM_COLS>(in_img, out_row_idx, out_col_idx, X_STENCIL_F32);
      let x_sq = float32::mul(x_stencil_out, x_stencil_out);
      let y_stencil_out = apply_stencil_float32<NUM_ROWS, NUM_COLS>(in_img, out_row_idx, out_col_idx, Y_STENCIL_F32);
      let y_sq = float32::mul(y_stencil_out, y_stencil_out);
      let sum_sq = float32::add(x_sq, y_sq);
      let pixel_val = fpsqrt_32::fpsqrt_32(sum_sq);

      // Multi-dimensional update is really ugly - is there a cleaner way to do this?
      let out_idx = out_row_idx * NUM_COLS_OUT + out_col_idx;
      update(out_img, out_idx, pixel_val)
    }(out_img)
  }(F32[NUM_ELMS_OUT]:[float32::zero(u1:0), ...])
}

#[test]
fn sobel_filter_float32_test() {
  // Do we have a way to reshape a 9-element array to a 3x3?
  let img1 = map(s32[16]:[1, 1, 1, 1,
                          1, -1, -2, 1,
                          1, -4, -3, 1,
                          1, 1, 1, 1], float32::cast_from_fixed_using_rne);

  let sobel_out = sobel_filter_float32<u32:4, u32:4>(img1);
  // Truncate before comparison for simplicity.
  // Can't use map to cast_to_fixed for all elements at once because
  // map does not seem to support parametric functions.
  assert_eq(float32::cast_to_fixed<u32:32>(sobel_out[u32:0]), s32:17);
  assert_eq(float32::cast_to_fixed<u32:32>(sobel_out[u32:1]), s32:15);
  assert_eq(float32::cast_to_fixed<u32:32>(sobel_out[u32:2]), s32:13);
  assert_eq(float32::cast_to_fixed<u32:32>(sobel_out[u32:3]), s32:14);


  // Try non-square image.
  let img1 = map(s32[12]:[1, 1, 1, 1,
                          1, -1, -2, 1,
                          1, -4, -3, 1], float32::cast_from_fixed_using_rne);
  let sobel_out = sobel_filter_float32<u32:3, u32:4>(img1);
  assert_eq(float32::cast_to_fixed<u32:32>(sobel_out[u32:0]), s32:17);
  assert_eq(float32::cast_to_fixed<u32:32>(sobel_out[u32:1]), s32:15);

}

// TODO(jbaileyhandle): Fixed-point version (Images are generally fixed point).
// Note that sqrt is a floating-point op, so we would need to cast to float
// internally and then cast the output.
