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

fn main(array: u8[4]) -> (u32, u8)[4] { enumerate(array) }

#[test]
fn enumerate_test() {
    let my_array = u8[4]:[1, 2, 4, 8];
    let enumerated = main(my_array);
    assert_eq(enumerated[u32:0], (u32:0, u8:1));
    assert_eq(enumerated[u32:1], (u32:1, u8:2));
    assert_eq(enumerated[u32:2], (u32:2, u8:4));
    assert_eq(enumerated[u32:3], (u32:3, u8:8));
}
