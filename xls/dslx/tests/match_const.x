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

const FOO = u8:42;

fn match_const(x: u8) -> u8 {
  match x {
    FOO => u8:0,
    _ => u8:42,
  }
}

#[test]
fn match_const_not_binding() {
  let _ = assert_eq(u8:42, match_const(u8:0));
  let _ = assert_eq(u8:42, match_const(u8:1));
  let _ = assert_eq(u8:0, match_const(u8:42));
  ()
}

fn h(t: (u8, (u16, u32))) -> u32 {
  match t {
    (FOO, (x, y)) => (x as u32) + y,
    (_, (y, u32:42)) => y as u32,
    _ => u32:7,
  }
}

#[test]
fn match_nested() {
  let _ = assert_eq(u32:3, h((u8:42, (u16:1, u32:2))));
  let _ = assert_eq(u32:1, h((u8:0, (u16:1, u32:42))));
  let _ = assert_eq(u32:7, h((u8:0, (u16:1, u32:0))));
  ()
}

fn main() -> u32 {
  match_const(u8:42) as u32 + h((u8:0, (u16:1, u32:42)))
}

#[test]
fn main_test() {
  assert_eq(u32:1, main())
}