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

const THING_COUNT = u32:2;

type Foo = (u32[THING_COUNT]);

fn get_thing(x: Foo, i: u32) -> u32 {
    let things: u32[THING_COUNT] = x.0;
    things[i]
}

#[test]
fn foo_test() {
    let foo: Foo = (u32[THING_COUNT]:[42, 64],);
    assert_eq(u32:42, get_thing(foo, u32:0));
    assert_eq(u32:64, get_thing(foo, u32:1));
}
