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

fn foo<A: u32, B: u32, C: u32 = A if A > B else B>(x: bits[A], y: bits[B]) -> bits[C] {
  let use_a = u1:1 if A > B else u1:0;
  ((x + (y as bits[A])) as bits[C]) if use_a else (((x as bits[B]) + y) as bits[C])
}

test parametric_with_comparison {
  let A = u32:8;
  let B = u32:16;
  let x = bits[8]:0xff;
  let y = bits[16]:0xff;
  let actual = foo(x, y) as u16;
  let expected = u16:0x1fe;
  let _ = assert_eq(actual, expected);

  let A = u32:16;
  let B = u32:8;
  let x = bits[16]:0xff;
  let y = bits[8]:0xff;
  let actual = foo(x, y) as u16;
  let expected = u16:0x1fe;
  let _ = assert_eq(actual, expected);
  ()
}
