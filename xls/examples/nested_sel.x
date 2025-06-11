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

// Small nested-select microbenchmark.

fn mysel(cond: bool, x: u8, y: u8) -> u8 {
  if cond { x } else { y }
}

fn main(p: u8, v: u8, w: u8, x: u8, y: u8, z: u8) -> u8 {
  let p_gt_32 = p > u8:32;
  let p_eq_7 = p == u8:7;
  let p_eq_6 = p == u8:6;
  let p_le_3 = p <= u8:3;
  mysel(p_le_3, v, mysel(p_eq_6, w, mysel(p_eq_7, x, mysel(p_gt_32, y, z))))
}
