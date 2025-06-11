#![feature(type_inference_v2)]

// Copyright 2023 The XLS Authors
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

import xls.examples.sobel_filter;

import float32;

type F32 = float32::F32;

fn apply_stencil_float32_8x8(
    in_img: F32[64],
    row_idx:u32, col_idx:u32, stencil: F32[3][3]) -> F32 {
  sobel_filter::apply_stencil_float32<u32:8, u32:8>(in_img, row_idx, col_idx, stencil)
}
