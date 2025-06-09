// Copyright 2021 The XLS Authors
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

struct MyStruct { x: u32 }

const C = MyStruct { x: u32:3 };

const C_ARRAY = MyStruct[1]:[C];

fn main() -> u32 {
    const C': MyStruct = C_ARRAY[u32:0];
    for (i, accum) in u32:0..C'.x {
        accum + i
    }(u32:0)
}

#[test]
fn test_main() { assert_eq(main(), u32:0 + u32:1 + u32:2) }
