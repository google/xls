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

fn main(x: u8[4], start: u32) -> u8[3] { array_slice(x, start, u8[3]:[0, ...]) }

#[test]
fn slice_test() {
    let a: u8[4] = u8[4]:[4, ...];
    assert_eq(u8[3]:[4, 4, 4], main(a, u32:1));
    assert_eq(u8[3]:[4, 4, 4], u8[3]:[4, ...]);
    assert_eq(u8:4, (u8[3]:[4, ...])[u32:2]);
}
