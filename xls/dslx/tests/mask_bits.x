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

// Returns a value with X bits set (of type bits[X]).
pub fn mask_bits<X: u32, Y:u32= X + u32:1>() -> bits[X] {
  !bits[X]:0
}

#[test]
fn test_mask_bits() {
  let _ = assert_eq(u8:0xff, mask_bits<u32:8>());
  //let _ = assert_eq(u13:0x1fff, mask_bits<u32:13>());
  ()
}
