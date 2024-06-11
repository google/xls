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

#[test]
fn numerical_conversions() {
    let s8_m2 = s8:-2;
    let u8_m2 = u8:0xfe;
    // Sign extension (source type is signed).
    assert_eq(s32:-2, s8_m2 as s32);
    assert_eq(u32:0xfffffffe, s8_m2 as u32);
    assert_eq(s16:-2, s8_m2 as s16);
    assert_eq(u16:0xfffe, s8_m2 as u16);
    // Zero extension (source type is unsigned).
    assert_eq(u32:0xfe, u8_m2 as u32);
    assert_eq(s32:0xfe, u8_m2 as s32);
    // Nop (bitwidth is unchanged).
    assert_eq(s8:-2, s8_m2 as s8);
    assert_eq(u8:0xfe, s8_m2 as u8);
    assert_eq(u8:0xfe, u8_m2 as u8);
    assert_eq(s8:-2, u8_m2 as s8);
}
