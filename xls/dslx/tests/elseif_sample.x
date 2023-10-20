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

fn elseif_sample(s: bool, x: u32, y: u32) -> u32 {
    if s == true {
        x
    } else if x == u32:7 {
        y
    } else {
        u32:42
    }
}

#[test]
fn elseif_sample_test() {
    assert_eq(u32:7, elseif_sample(true, u32:7, u32:1));
    assert_eq(u32:1, elseif_sample(false, u32:7, u32:1));
    assert_eq(u32:42, elseif_sample(false, u32:8, u32:1));
}

fn elseif_wrapper(x: u32) -> u8 {
    if x == u32:42 {
        u8:1
    } else if x == u32:64 {
        u8:2
    } else if x == u32:77 {
        u8:3
    } else {
        u8:4
    }
}

#[test]
fn elseif_wrapper_test() {
    assert_eq(u8:1, elseif_wrapper(u32:42));
    assert_eq(u8:2, elseif_wrapper(u32:64));
    assert_eq(u8:3, elseif_wrapper(u32:77));
    assert_eq(u8:4, elseif_wrapper(u32:128));
}

fn main() -> u32 { elseif_wrapper(u32:42) as u32 + elseif_sample(false, u32:7, u32:1) }

#[test]
fn main_test() { assert_eq(u32:2, main()) }
