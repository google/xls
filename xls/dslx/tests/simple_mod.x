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

fn main(a: u32, b: u32) -> u32 { a % b }

#[test]
fn simple_mod_test() {
    assert_eq(u32:2, main(u32:42, u32:5));
    assert_eq(u32:2, u32:42 % u32:5);
    assert_eq(s32:-2, s32:-42 % s32:5);
    assert_eq(s32:2, s32:42 % s32:-5);
    assert_eq(s32:-2, s32:-42 % s32:-5);

    // Division by zero. Defined to be zero.
    assert_eq(u32:0, u32:42 % u32:0);
    assert_eq(s32:0, s32:42 % s32:0);
}
