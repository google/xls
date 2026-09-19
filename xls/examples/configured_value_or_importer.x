// Copyright 2026 The XLS Authors
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

// Test & example for using configured values in an imported module.

import xls.examples.configured_value_or;

fn real_main
    ()
    -> (bool, u32, s32, configured_value_or::MyEnum, bool, u32, s32, configured_value_or::MyEnum) {
    let (a, b, c, d, e, f, g, h) = configured_value_or::main();
    (e, f, g, h, a, b, c, d)
}

#[test]
fn test_importee() {
    assert_eq(
        (
            true, u32:123, s32:-200, configured_value_or::MyEnum::B, false, u32:42, s32:-100,
            configured_value_or::MyEnum::C,
        ), real_main());
}

#[test]
fn test_imported() {
    assert_eq(
        (
            false, u32:42, s32:-100, configured_value_or::MyEnum::C, true, u32:123, s32:-200,
            configured_value_or::MyEnum::B,
        ), configured_value_or::main());
}
