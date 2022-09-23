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

fn foo<A: u32, B: u32, C: u32 = if A > B { A } else { B }>(x: bits[A], y: bits[B]) -> bits[C] {
  let use_a = if A > B { u1:1 } else { u1:0 };
  if use_a {
    ((x + (y as bits[A])) as bits[C])
  } else {
    (((x as bits[B]) + y) as bits[C])
  }
}

#[test]
fn parametric_with_comparison() {
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

// This set of functions tests what was an ambiguous case when
// parsing parametrics: is foo<A>(B)>(C) an invocation of foo
// parameterized by A with the argument B, or is it foo parameterized
// by "A<B" and invoked with C?
// We [now] use Rust generic expression rules, as of v1.51, which
// contain parametric expressions in curly braces, which disambiguates
// this sort of case.
fn callee<A: u32>(x: bits[A]) -> bits[A] {
  x + bits[A]:1
}

const X = u32:5;
const Y = u32:6;
const Z = u1:0;
const W = u1:1;

fn caller() -> u32{
  let x = u32:16;
  let y = callee<u32:32>(x);
  let z = callee<{(u32:32 > u32:16) as u32 + u32:4}>(x as u5) as u32;
  callee<{ (X > (Y) > (Z) > (W)) as u32 + u32:15 }>(u15:8) as u32
}
