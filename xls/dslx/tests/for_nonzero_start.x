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

const LOOP_START = u32:5;

fn main() -> u32 {
    for (i, j): (u32, u32) in LOOP_START..u32:10 {
        j + i
    }(u32:0)
}

#[test]
fn test_fn() {
    let x = main();
    assert_eq(x, u32:35)
}

#[test]
fn wide_negative_start() {
    for (i, _) in sN[65]:-1..0 {
        assert_eq(i, -1)
    }(())
}

#[test]
fn signed_range_crossing_zero() {
    let count = for (_, acc): (s4, u32) in s4:-8..s4:7 {
        acc + u32:1
    }(u32:0);
    assert_eq(count, u32:15);
}

#[test]
fn wide_range_array() {
    let start = uN[74]:0x1_ffff_ffff_ffff_ffff;
    let end = uN[74]:0x2_0000_0000_0000_0000;
    assert_eq(start..=end, [start, end]);
}
