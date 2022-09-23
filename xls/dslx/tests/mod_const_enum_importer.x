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

import xls.dslx.tests.mod_simple_const_enum

type MyEnum = mod_simple_const_enum::MyEnum;

fn main(x: MyEnum) -> bool {
  x == mod_simple_const_enum::MY_FOO
}

#[test]
fn main_test() {
  let _ = assert_eq(main(MyEnum::FOO), true);
  let _ = assert_eq(main(MyEnum::BAR), false);
  ()
}
