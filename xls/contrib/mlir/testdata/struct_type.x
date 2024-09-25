// Copyright 2024 The XLS Authors
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

pub struct float32 {
  sign: u1,
  bexp: u8,
  fraction: u23,
}

pub fn int_to_float(x: s32) -> float32 {
  float32 {sign: u1:1, bexp: u8:0x10, fraction: u23:0xbeef}
}

pub fn float_to_int(x: float32) -> s32 {
  s32:0xbeef
}
