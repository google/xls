// Copyright 2021 The XLS Authors
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

// Uses an enum type for a parametric expression.

enum MyEnum : u1 {
  A = false,
  B = true,
}

fn p<E: MyEnum>(x: MyEnum) -> bool {
  x == E
}

fn is_a(x: MyEnum) -> bool { p<MyEnum::A>(x) }
fn is_b(x: MyEnum) -> bool { p<MyEnum::B>(x) }

fn main(x: MyEnum) -> bool {
  is_a(x) || is_b(x)
}

#![test]
fn parametric_enum_value() {
  let _ = assert_eq(is_a(MyEnum::A), true);
  let _ = assert_eq(is_a(MyEnum::B), false);
  let _ = assert_eq(is_b(MyEnum::A), false);
  let _ = assert_eq(is_b(MyEnum::B), true);
  let _ = assert_eq(main(MyEnum::A), true);
  let _ = assert_eq(main(MyEnum::B), true);
  ()
}
