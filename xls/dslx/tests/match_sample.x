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

fn match_sample(s: bool, x: u32, y: u32) -> u32 {
  match (s, x, y) {
    (true, _, _) => x,
    (false, u32:7, b) => b,
    _ => u32:42,
  }
}

#[test]
fn match_wildcard_test() {
  let _ = assert_eq(u32:7, match_sample(true, u32:7, u32:1));
  let _ = assert_eq(u32:1, match_sample(false, u32:7, u32:1));
  let _ = assert_eq(u32:42, match_sample(false, u32:8, u32:1));
  ()
}

fn match_wrapper(x: u32) -> u8 {
  match x {
    u32:42 => u8:1,
    u32:64 => u8:2,
    u32:77 => u8:3,
    _ => u8:4
  }
}

#[test]
fn match_wrapper_test() {
  let _: () = assert_eq(u8:1, match_wrapper(u32:42));
  let _: () = assert_eq(u8:2, match_wrapper(u32:64));
  let _: () = assert_eq(u8:3, match_wrapper(u32:77));
  let _: () = assert_eq(u8:4, match_wrapper(u32:128));
  ()
}

fn main() -> u32 {
  match_wrapper(u32:42) as u32 + match_sample(false, u32:7, u32:1)
}

#[test]
fn main_test() {
  assert_eq(u32:2, main())
}

