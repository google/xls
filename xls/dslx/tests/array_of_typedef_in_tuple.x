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

type Foo = (u8, u32);
type Bar = (
  u16,
  Foo[2],
);
type Foo2 = Foo[2];

fn foo_add_all(x: Foo) -> u32 {
  (x.0 as u32) + x.1
}

fn bar_add_all(x: Bar) -> u32 {
  let foos: Foo2 = x.1;
  let foo0: Foo = foos[u32:0];
  let foo1: Foo = foos[u32:1];
  (x.0 as u32) + foo_add_all(foo0) + foo_add_all(foo1)
}

#[test]
fn bar_add_all_test() {
  let foo0: Foo = (u8:1, u32:2);
  let bar: Bar = (u16:3, [foo0, foo0]);
  assert_eq(u32:9, bar_add_all(bar))
}

fn main(x: Bar) -> u32 {
  bar_add_all(x)
}

