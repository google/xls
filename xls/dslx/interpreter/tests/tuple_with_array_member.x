// Copyright 2020 Google LLC
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

type Foo = (
  u32,
  u8[4],
);

fn flatten(x: u8[4]) -> u32 {
  x[u32:0] ++ x[u32:1] ++ x[u32:2] ++ x[u32:3]
}

fn add_members(x: Foo) -> u32 {
  x[u32:0] + flatten(x[u32:1])
}

test add_members {
  let x: Foo = (u32:0xdeadbeef, u8[4]:[0, 0, 0, 1]);
  assert_eq(u32:0xdeadbef0, add_members(x))
}

fn main(x: Foo) -> u32 {
  add_members(x)
}
