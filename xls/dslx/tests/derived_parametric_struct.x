// Copyright 2024 The XLS Authors
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

struct StructFoo<A: u32, B: u32 = {u32:32}, C: u32 = {B / u32:2}> { a: uN[A], b: uN[B], c: uN[C] }

#[test]
fn Foo() {
    let foo = StructFoo<u32:5> { a: u5:0, b: u32:1, c: u16:0 };
    assert_eq(foo.c, u16:0);
}
