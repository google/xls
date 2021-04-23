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

// A few "comparatively simple" routines to invoke via jit_wrapper entry
// points, exercising some interesting features of JIT invocation; e.g.
// matching floating point type, boolean, uint32_t, and fallibility.

import float32

type F32 = float32::F32;

pub fn identity(x: F32) -> F32 { x }

pub fn is_inf(x: F32) -> bool { float32::is_inf(x) }

pub fn fail_on_42(x: u32) -> u32 {
  match x {
    u32:42 => fail!(x),
    _ => x
  }
}
