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

fn p<N: u32>() -> u32 {
    for (i, accum): (u32, u32) in u32:0..N {
        accum + i
    }(u32:0)
}

fn main() -> u32 { p<u32:4>() }

#[test]
fn test_main() { assert_eq(main(), u32:6) }
