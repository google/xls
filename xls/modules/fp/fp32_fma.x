// Copyright 2020 The XLS Authors
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

// This file implements most of a IEEE-754-compliant fused multiply-add
// operation, with the following exceptions:
//  - Both input and output denormals are treated as/flushed to 0 (internal
//    subnormals arising from the internal product are left intact.
//  - Only round-to-nearest mode is supported.
//  - No exception flags are raised/reported.
// In all other cases, results should be identical to other
// conforming implementations (modulo exact fraction values in the NaN case).
import float32
import xls.modules.fp.apfloat_fma

type F32 = float32::F32;
pub fn fp32_fma(a: F32, b: F32, c: F32) -> F32 {
  apfloat_fma::fma<u32:8, u32:23>(a, b, c)
}
