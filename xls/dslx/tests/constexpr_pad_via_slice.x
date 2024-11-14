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

// Test case reflecting https://github.com/google/xls/issues/214

// Pads an array up to "R" elements by filling with zeros in the higher indices.
fn pad<X: u32, R: u32, DIFF: u32 = {R - X}>(xs: u32[X]) -> u32[R] {
    if R >= X { xs ++ u32[DIFF]:[0, ...] } else { array_slice(xs, u32:0, u32[R]:[0, ...]) }
}

fn main(x: u32[2]) -> u32[4] { pad<u32:2, u32:4>(x) }

#[test]
fn test_pad() {
    assert_eq(u32[4]:[42, 42, 0, 0], main(u32[2]:[42, 42]));
    assert_eq(u32[4]:[42, 42, 0, 0], pad<u32:2, u32:4>(u32[2]:[42, 42]));
    assert_eq(u32[2]:[42, 0], pad<u32:1, u32:2>(u32[1]:[42]));
}
