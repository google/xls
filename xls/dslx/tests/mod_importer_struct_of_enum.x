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

import xls.dslx.tests.mod_imported_struct_of_enum

type MyEnum2 = mod_imported_struct_of_enum::MyEnum;
type MyStruct2 = mod_imported_struct_of_enum::MyStruct;

fn main(x: u8) -> MyStruct2 {
  MyStruct2 { x: x as MyEnum2 }
}

#[test]
fn main_test() {
  let s: MyStruct2 = main(u8:42);
  assert_eq(s.x, MyEnum2::FOO)
}
