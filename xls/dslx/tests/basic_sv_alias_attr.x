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

#[sv_type("cool_type")]
type Foo = (u32, u8);

fn main(x: Foo) -> u32 { x.0 + (x.1 as u32) }

#[test]
fn main_test() {
    let foo: Foo = (u32:42, u8:3);
    assert_eq(u32:45, main(foo));
}
