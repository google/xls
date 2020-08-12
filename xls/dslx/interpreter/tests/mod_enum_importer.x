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

import xls.dslx.interpreter.tests.mod_imported

type MyEnum = mod_imported::MyEnum;

fn main(x: u8) -> MyEnum {
  x as MyEnum
}

test main {
  let _ = assert_eq(main(u8:42), MyEnum::FOO);
  let _ = assert_eq(main(u8:64), MyEnum::BAR);
  ()
}
