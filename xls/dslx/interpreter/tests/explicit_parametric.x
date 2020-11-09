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
struct Foo<X:u32, Y:u32> {
  a: bits[X],
  b: bits[Y]
}

pub fn FooZero(a: bits[4]) -> Foo<u32:4, u32:8> {
  Foo<u32:4, u32:8>{ a: a, b: bits[8]:0 }
}

pub fn IndirectFoo<X:u64, Y: u32 = (X * X) as u32>(a: bits[4]) -> Foo<X as u32, u32:8> {
  Foo<X as u32, u32:8>{ a: a as bits[X], b: bits[8]:32 }
}

pub fn InstantiatesIndirectFoo(a: bits[16]) -> Foo<u32:16, u32:8> {
  IndirectFoo<u64:16>(a as bits[4])
}
