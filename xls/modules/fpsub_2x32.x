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
import xls.modules.apfloat_sub_2
import xls.modules.fpadd_2x32
import float32

type F32 = float32::F32;

pub fn fpsub_2x32(x: F32, y: F32) -> F32 {
  apfloat_sub_2::apfloat_sub_2<u32:8, u32:23>(x,y)
}
