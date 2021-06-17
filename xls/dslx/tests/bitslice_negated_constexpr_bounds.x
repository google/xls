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

// Ensures that we can be referentially transparent with negative index slices.

const FIVE_U32 = u32:5;
const FIVE_S32 = s32:5;

fn f(x: u32) -> bits[5] { x[-5:] }
fn g(x: u32) -> bits[5] { x[-FIVE_S32:] }
fn h(x: u32) -> bits[5] { x[(-FIVE_U32) as s32:] }

#![test]
fn main_test() {
  let x = u32:0xa000_0000;
  let want = u5:0b1010_0;
  let _ = assert_eq(f(x), g(x));
  let _ = assert_eq(g(x), h(x));
  let _ = assert_eq(f(x), want);
  let _ = assert_eq(g(x), want);
  let _ = assert_eq(h(x), want);
  ()
}
