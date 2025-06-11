#![feature(type_inference_v2)]

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

import xls.examples.dslx_module.capitalize;

pub enum Choice: u2 {
  CAPITALIZE = u2:0,
  SPONGE = u2: 1,
  NOTHING = u2:2,
}
pub fn maybe_capitalize<N: u32>(word: u8[N], state: Choice) -> u8[N] {
  if (state == Choice::CAPITALIZE) {
    capitalize::capitalize<N>(word)
  } else if (state == Choice::SPONGE) {
    capitalize::sponge_capitalize<N>(word)
  } else {
    word
  }
}

#[test]
fn capitalize_choice() {
  assert_eq(maybe_capitalize<u32:6>("foobar", Choice::CAPITALIZE), "FOOBAR");
  assert_eq(maybe_capitalize<u32:6>("FOOBAR", Choice::CAPITALIZE), "FOOBAR");
  assert_eq(maybe_capitalize<u32:6>("123456", Choice::CAPITALIZE), "123456");
  assert_eq(maybe_capitalize<u32:12>("123456foobar", Choice::CAPITALIZE), "123456FOOBAR");
}
#[test]
fn nothing_choice() {
  assert_eq(maybe_capitalize<u32:6>("foobar", Choice::NOTHING), "foobar");
  assert_eq(maybe_capitalize<u32:6>("FOOBAR", Choice::NOTHING), "FOOBAR");
  assert_eq(maybe_capitalize<u32:6>("123456", Choice::NOTHING), "123456");
  assert_eq(maybe_capitalize<u32:12>("123456foobar", Choice::NOTHING), "123456foobar");
}
#[test]
fn sponge_choice() {
  assert_eq(maybe_capitalize<u32:6>("foobar", Choice::SPONGE), "FoObAr");
  assert_eq(maybe_capitalize<u32:6>("FOOBAR", Choice::SPONGE), "FoObAr");
  assert_eq(maybe_capitalize<u32:6>("123456", Choice::SPONGE), "123456");
  assert_eq(maybe_capitalize<u32:12>("123456foobar", Choice::SPONGE), "123456FoObAr");
}
