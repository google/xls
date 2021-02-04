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

// This file implements most of IEEE-754 single-precision
// floating-point subtraction, with the following exceptions:
//  - Both input and output denormals are treated as/flushed to 0.
//  - Only round-to-nearest mode is supported.
//  - No exception flags are raised/reported.
// In all other cases, results should be identical to other
// conforming implementations (modulo exact significand
// values in the NaN case.
import float32
import xls.modules.fpadd_2x32

type F32 = float32::F32;

pub fn fpsub_2x32(x: F32, y: F32) -> F32 {
  let y = F32 {sign: !y.sign, ..y};
  fpadd_2x32::fpadd_2x32(x, y) 
}

// fpadd_2x32 is thoroughly tested elsewhere
// and fpsub_2x32 is a trivial modification of
// fpadd_2x32, so a few simple tests is sufficient.
#![test]
fn test_fpsub_2x32() {
  let one = float32::one(u1:0);
  let two = fpadd_2x32::fpadd_2x32(one, one);
  let neg_two = F32 {sign: u1:1, ..two};
  let three = fpadd_2x32::fpadd_2x32(one, two);
  let four = fpadd_2x32::fpadd_2x32(two, two);

  let _ = assert_eq(fpsub_2x32(four, one), three);
  let _ = assert_eq(fpsub_2x32(four, two), two);
  let _ = assert_eq(fpsub_2x32(four, three), one);
  let _ = assert_eq(fpsub_2x32(three, two), one);
  let _ = assert_eq(fpsub_2x32(two, four), neg_two);
  ()
}
