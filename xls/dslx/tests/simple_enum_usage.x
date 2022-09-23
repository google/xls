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

enum Foo : u32 {
  A = 0,
  B = 1,
}

fn main(x: Foo) -> Foo {
  if x == Foo::A { Foo::B } else { Foo::A }
}

#[test]
fn test_main() {
  let _ = assert_eq(Foo::B, main(Foo::A));
  let _ = assert_eq(Foo::A, main(Foo::B));
  ()
}