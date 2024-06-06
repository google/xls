// Copyright 2020 The XLS Authors
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

#[sv_type("cool_point")]
struct Point {
    some_super_long_and_wordy_field_name: u32,
    another_long_and_wordy_field_reach_line_limit: u32,
}

fn f(p: Point) -> u32 {
    p.some_super_long_and_wordy_field_name + p.another_long_and_wordy_field_reach_line_limit
}

fn main() -> u32 {
    f(
        Point {
            some_super_long_and_wordy_field_name: u32:42,
            another_long_and_wordy_field_reach_line_limit: u32:64
        })
}

#[test]
fn main_test() { assert_eq(u32:106, main()) }
