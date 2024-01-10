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

import xls.dslx.tests.mod_simple_enum;

fn f() -> mod_simple_enum::EnumType { zero!<mod_simple_enum::EnumType>() }

fn g() -> mod_simple_enum::EnumTypeAlias { zero!<mod_simple_enum::EnumTypeAlias>() }

fn main() -> (mod_simple_enum::EnumType, mod_simple_enum::EnumTypeAlias) { (f(), g()) }

#[test]
fn test_main() {
    let (a, b) = main();
    assert_eq(a, b);
    assert_eq(a, mod_simple_enum::EnumType::FIRST)
}
