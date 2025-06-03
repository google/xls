// Copyright 2022 The XLS Authors
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

const ARRAY_SIZE = u32:256;

// Microbenchmark which does some operations on a large array.
fn large_array(x: u32, indices: u8[ARRAY_SIZE]) -> u32[ARRAY_SIZE] {
   let a = u32[ARRAY_SIZE]:[x, u32: 0, ... ];
   let b = u32[ARRAY_SIZE]:[u32:0, x, ... ];
   for (i, c): (u32, u32[ARRAY_SIZE]) in u32:0..ARRAY_SIZE {
     let index: u8 = indices[i];
     update(c, index, a[index] + b[index] + c[index])
   }(u32[ARRAY_SIZE]:[u32:0, ...])
}
