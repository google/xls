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

// Tests for explicit instantiations of parametric structs and functions.
struct Generic<X:u32, Y:u32> {
  a: bits[X],
  b: bits[Y]
}

pub fn foo(a: bits[4]) -> Generic<u32:4, u32:8> {
  Generic<u32:4, u32:8>{ a: a, b: bits[8]:0 }
}

pub fn indirect_foo<X: u32, Y: u32 = (X * X) as u32>(a: bits[4]) -> Generic<{X as u32}, u32:8> {
  Generic<{X as u32}, u32:8>{ a: a as bits[X], b: bits[8]:32 }
}

pub fn instantiates_indirect_foo(a: bits[16]) -> Generic<u32:16, u32:8> {
  indirect_foo<u32:16>(a as bits[4])
}

pub fn parameterized_zero<C:u32, D:u32>(x: bits[C]) -> Generic<C, D> {
  Generic<C, D>{ a: bits[C]:0, b: bits[D]:1 }
}

pub fn two_param_indirect<E:u32, F:u32>(value: bits[E]) -> Generic<E, F> {
  parameterized_zero<E, F>(value)
}

#![test]
fn generic() {
  let actual = two_param_indirect<u32:1, u32:2>(u1:0);
  ()
}
