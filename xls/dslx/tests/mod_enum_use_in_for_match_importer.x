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

// Imports a module that holds an enum definition, and uses it inside of a match
// expression inside of a for loop.

import xls.dslx.tests.mod_simple_enum;

type EnumType = mod_simple_enum::EnumType;

fn main(x: EnumType) -> bool {
    for (_, _): (u32, bool) in u32:0..u32:1 {
        match x {
            EnumType::FIRST => false,
            EnumType::SECOND => true,
        }
    }(false)
}

#[test]
fn test_main() {
    assert_eq(false, main(EnumType::FIRST));
    assert_eq(true, main(EnumType::SECOND));
}
