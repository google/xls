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

// A few "comparatively simple" routines to invoke via jit_wrapper entry
// points, exercising some interesting features of JIT invocation; e.g.
// matching floating point type, boolean, uint32_t, and fallibility.

import std;

pub fn umul_with_overflow_21_21_18(x: uN[21], y: uN[18]) -> (bool, uN[21]) {
  std::umul_with_overflow<u32:21>(x, y)
}

pub fn umul_with_overflow_35_32_18(x: uN[32], y: uN[18]) -> (bool, uN[35]) {
  std::umul_with_overflow<u32:35>(x, y)
}
