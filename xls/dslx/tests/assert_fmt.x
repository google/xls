// Copyright 2025 The XLS Authors
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

fn main() {}

#[test]
fn test_assert_fmt_passes() {
    let i = u32:5;
    assert_fmt!(i == u32:5, "i should be 5 but is {}", i);
}

#[test]
fn test_assert_fmt_signed() {
    let i = s32:-1;
    assert_fmt!(i == s32:-1, "i should be -1 but is {:d}", i);
}

#[test]
fn test_assert_fmt_bool() {
    let i = bool:true;
    assert_fmt!(i, "i should be true but is {}", i);
}

#[test]
fn test_assert_fmt_in_unrolled_for() {
    unroll_for! (i, _): (u32, ()) in u32:0..u32:4 {
        assert_fmt!(i < u32:4, "i is {}!", i);
    }(());
}
