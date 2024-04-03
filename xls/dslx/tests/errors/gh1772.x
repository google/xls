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

// Regression test for https://github.com/google/xls/issues/1172

pub struct Foo<WIDTH:u32> {
  a: uN[WIDTH],
}

fn sizeof_foo<WIDTH:u32>(x: Foo<WIDTH>) -> u32 {
  WIDTH
}

#[test]
fn struct_assign_test() {
  let x = Foo<5> { a:u17:1 };        // Want type Foo<5>, but with undetected typo ...
  assert_eq(sizeof_foo(x), u32:17);  // ... so actually got type Foo<17> instead
}
