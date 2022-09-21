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
import float64
import xls.modules.fp.apfloat_fma

type F64 = float64::F64;
pub fn fp64_fma(a: F64, b: F64, c: F64) -> F64 {
  apfloat_fma::fma<u32:11, u32:52>(a, b, c)
}
