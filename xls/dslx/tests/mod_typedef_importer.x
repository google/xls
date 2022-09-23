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

import xls.dslx.tests.mod_imported_aliases

type MyEnum = mod_imported_aliases::MyEnumAlias;
type MyStruct = mod_imported_aliases::MyStructAlias;
type MyTuple = mod_imported_aliases::MyTupleAlias;

fn main(x: u8) -> MyTuple {
  let me: MyEnum = x as MyEnum;
  (MyStruct { me: me }, MyEnum::FOO)
}

#[test]
fn main_test() {
  let (ms, me): (MyStruct, MyEnum) = main(u8:64);
  let _ = assert_eq(MyEnum::BAR, ms.me);
  assert_eq(MyEnum::FOO, me)
}
