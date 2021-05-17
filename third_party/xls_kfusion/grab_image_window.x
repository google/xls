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

import std
import float32

type F32 = float32::F32;

// Gather data from a NUM_ROWS x NUM_COLS image to apply 
// stencil operation to.
// That is, grab a WINDOW_WIDTH x WINDOW_SIZE block from
// 'in_img', centered on the pixel at ['center_row', 'center_col'].
// If this window extends beyond the bounds of 'in_img', we
// duplicate the pixels at the boundary for the window indexes
// that lay beyond the boundry.
pub fn grab_image_window<NUM_ROWS:u32, 
  NUM_COLS:u32, RADIUS:u32, NUM_ELMS:u32 = NUM_ROWS*NUM_COLS, 
  WINDOW_WIDTH:u32 = u32:2 * RADIUS + u32:1,
  WINDOW_SIZE:u32 = WINDOW_WIDTH * WINDOW_WIDTH>
  (in_img: F32[NUM_ELMS], center_row:u32, center_col:u32) -> F32[WINDOW_SIZE] {

  let pixel_idx = center_row * NUM_COLS + center_col;

  for(row_offset, window): (u32, F32[WINDOW_SIZE])
    in range(u32:0, WINDOW_WIDTH) {
    for(col_offset, window): (u32, F32[WINDOW_SIZE])
      in range(u32:0, WINDOW_WIDTH) {

      let row_idx = std::sclamp((center_row + row_offset - RADIUS) as s32, 
        s32:0, (NUM_ROWS as s32) - s32:1);
      let col_idx = std::sclamp((center_col + col_offset - RADIUS) as s32, 
        s32:0, (NUM_COLS as s32) - s32:1);
      let pixel_idx = row_idx * (NUM_COLS as s32) + col_idx;
      let pixel_val = in_img[pixel_idx];
      let window_idx = row_offset * WINDOW_WIDTH + col_offset;
      update(window, window_idx, pixel_val)

    }(window)
  }(F32[WINDOW_SIZE]:[float32::zero(u1:0), ...])
}

fn cast_to_s32(input:F32) -> s32 {
  float32::cast_to_fixed<u32:32>(input)
}

#![test]
fn grab_image_window_test() {
  let img1_fixed = s32[30]:[ 1,  2,  3,  4,  5,  6,
                             7,  8,  9, 10, 11, 12,
                            13, 14, 15, 16, 17, 18,
                            18, 19, 20, 21, 22, 23,
                            24, 25, 26, 27, 28, 29];
  let img1_f32 = map(img1_fixed, float32::cast_from_fixed);

  // "normal" window grab.
  let window_f32 = grab_image_window<u32:5, u32:6, u32:1>(img1_f32, u32:1, u32:1);
  let window_fixed = map(window_f32, cast_to_s32);
  let window_expected = s32[9]:[1, 2, 3, 7, 8, 9, 13, 14, 15];
  let _ = assert_eq(window_fixed, window_expected);

  // "normal" window grab.
  let window_f32 = grab_image_window<u32:5, u32:6, u32:1>(img1_f32, u32:2, u32:3);
  let window_fixed = map(window_f32, cast_to_s32);
  let window_expected = s32[9]:[9, 10, 11, 15, 16, 17, 20, 21, 22];
  let _ = assert_eq(window_fixed, window_expected);

  // Clamp top of window.
  let window_f32 = grab_image_window<u32:5, u32:6, u32:1>(img1_f32, u32:0, u32:1);
  let window_fixed = map(window_f32, cast_to_s32);
  let window_expected = s32[9]:[1, 2, 3, 1, 2, 3, 7, 8, 9];
  let _ = assert_eq(window_fixed, window_expected);

  // Clamp right hand side of window.
  let window_f32 = grab_image_window<u32:5, u32:6, u32:1>(img1_f32, u32:3, u32:5);
  let window_fixed = map(window_f32, cast_to_s32);
  let window_expected = s32[9]:[17, 18, 18, 22, 23, 23, 28, 29, 29];
  let _ = assert_eq(window_fixed, window_expected);

  // Clamp top and left hand side of window.
  let window_f32 = grab_image_window<u32:5, u32:6, u32:1>(img1_f32, u32:0, u32:0);
  let window_fixed = map(window_f32, cast_to_s32);
  let window_expected = s32[9]:[1, 1, 2, 1, 1, 2, 7, 7, 8];
  let _ = assert_eq(window_fixed, window_expected);

  // Clamp botom and right hand side of window.
  let window_f32 = grab_image_window<u32:5, u32:6, u32:1>(img1_f32, u32:4, u32:5);
  let window_fixed = map(window_f32, cast_to_s32);
  let window_expected = s32[9]:[22, 23, 23, 28, 29, 29, 28, 29, 29];
  let _ = assert_eq(window_fixed, window_expected);

  // Grab window with bigger radius.
  let window_f32 = grab_image_window<u32:5, u32:6, u32:2>(img1_f32, u32:2, u32:3);
  let window_fixed = map(window_f32, cast_to_s32);
  let window_expected = s32[25]:[ 2,  3,  4,  5,  6, 
                                  8,  9, 10, 11, 12, 
                                 14, 15, 16, 17, 18, 
                                 19, 20, 21, 22, 23, 
                                 25, 26, 27, 28, 29];
  let _ = assert_eq(window_fixed, window_expected);
  ()
}
