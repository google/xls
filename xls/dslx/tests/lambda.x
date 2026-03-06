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

fn add_two(arr: u32[4]) -> u32[4] { map(arr, |i| -> u32 { i + u32:2 }) }

#[test]
fn test_add_two() {
    let arr = add_two(u32:0..4);
    assert_eq(arr, u32:2..6);
}

fn main() -> () {
    let arr = add_two(u32:0..4);
    const_assert!(arr[0] == 2);

    let my_val = u32:2;
    let x = (|i, j| -> u32 { my_val * i * j })(u32:1, u32:4);
    const_assert!(x == 8);
}
