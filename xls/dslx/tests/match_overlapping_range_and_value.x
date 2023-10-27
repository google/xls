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

fn f(x: u32, y: u32, z: u32) -> u32 {
    match x {
        u32:2 => x,
        u32:1..u32:3 => y,
        _ => z,
    }
}

fn main(x: u32) -> u32 { f(x, x + u32:1, x + u32:2) }

#[test]
fn test_main() {
    assert_eq(main(u32:0), u32:2);  // z value
    assert_eq(main(u32:1), u32:2);  // y value
    assert_eq(main(u32:2), u32:2);  // x value
    assert_eq(main(u32:3), u32:5);  // z value
    assert_eq(main(u32:4), u32:6);  // z value
}
