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

// Contains a for loop that iterates up to a local const value.
//
// We fill all the bits of the const to show the value is interpreted as an
// unsigned limit.

fn main() -> u32 {
    const FOO = u2:3;
    for (i, accum): (u2, u32) in u2:0..FOO {
        accum + (i as u32)
    }(u32:0)
}

#[test]
fn main_test() { assert_eq(main(), u32:3) }
