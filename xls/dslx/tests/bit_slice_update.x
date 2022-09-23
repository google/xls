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

fn main(x: u32, y: u5, z: u23) -> u32 {
   bit_slice_update(x, y, z)
}

#[test]
fn bit_slice_update_test() {
  let _ = assert_eq(u6:0b010111, bit_slice_update(u6:0b010101, u16:0, u2:0b11));
  let _ = assert_eq(u6:0b110101, bit_slice_update(u6:0b010101, u16:4, u2:0b11));
  let _ = assert_eq(u6:0b110000, bit_slice_update(u6:0b000000, u16:4, u2:0b11));
  let _ = assert_eq(u6:0b010101, bit_slice_update(u6:0b010101, u16:6, u2:0b11));
  let _ = assert_eq(u6:0b010101,
                    bit_slice_update(u6:0b010101,
                                     bits[1234]:0xffff_ffff_ffff_ffff_ffff_ffff,
                                     u2:0b11));
  let _ = assert_eq(bits[96]:0xffab_ffff_ffff_ffff_ffff_ffff,
                    bit_slice_update(bits[96]:0xffff_ffff_ffff_ffff_ffff_ffff,
                                     u17: 80,
                                     u8: 0xab));
  let _ = assert_eq(bits[16]:0xf12d,
                    bit_slice_update(bits[16]:0xabcd,
                                     u17: 4,
                                     bits[96]:0xffff_ffff_ffff_ffff_ffff_ff12));
  ()
}
