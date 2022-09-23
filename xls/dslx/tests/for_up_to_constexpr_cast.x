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

// Contains a for loop that iterates up to a constexpr casted value.

const FOO = u2:3;
const CASTED = FOO as u4;

fn f() -> u32 {
  for (i, accum): (u4, u32) in u4:0..CASTED {
    accum + (i as u32)
  }(u32:0)
}

// As above, but puts the constexpr inline instead of binding to a module-level
// const.
fn g() -> u32 {
  for (i, accum): (u4, u32) in u4:0..FOO as u4 {
    accum + (i as u32)
  }(u32:0)
}

fn p<N: u2>() -> u32 {
  for (i, accum): (u4, u32) in u4:0..N as u4 {
    accum + (i as u32)
  }(u32:0)
}

fn main() -> u32 {
  f() + g() + p<FOO>()
}

#[test]
fn f_test() {
  let _ = assert_eq(f(), u32:3);
  let _ = assert_eq(g(), u32:3);
  let _ = assert_eq(p<FOO>(), u32:3);
  let _ = assert_eq(main(), u32:9);
  ()
}
