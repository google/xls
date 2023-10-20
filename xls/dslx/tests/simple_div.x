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

fn main(a: u32, b: u32) -> u32 { a / b }

#[test]
fn simple_div_test() {
    assert_eq(u32:8, main(u32:42, u32:5));
    assert_eq(u32:8, u32:42 / u32:5);
    assert_eq(s32:-8, s32:-42 / s32:5);
    assert_eq(s32:-8, s32:42 / s32:-5);
    assert_eq(s32:8, s32:-42 / s32:-5);

    // Division by zero: defined to max positive bits.
    assert_eq(u32:0xffff_ffff, u32:42 / u32:0);
    assert_eq(s32:0x7fff_ffff, s32:42 / s32:0);
}
