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

fn f(x: u32) -> u32 {
    match x {
        u32:2..u32:4 => u32:0,
        _ => x,
    }
}

fn main() -> u32[5] { u32[5]:[f(u32:0), f(u32:1), f(u32:2), f(u32:3), f(u32:4)] }

#[test]
fn test_main() { assert_eq(main(), u32[5]:[0, 1, 0, 0, 4]) }
