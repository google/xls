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

import xls.dslx.tests.mod_imported_typedef;

fn main(x: s8) -> s32 { signex(x, s32:0) }

#[test]
fn test_main() {
    assert_eq(main(s8:-1), s32:-1);
    assert_eq(main(s8:1), s32:1);
}

#[test]
fn signex_builtin() {
    let x = s8:-1;
    let s: s32 = signex(x, s32:0);
    let u: u32 = signex(x, u32:0);
    assert_eq(s as u32, u)
}

#[test]
fn signex_builtin_convert_to_fewer_bits() {
    let x = s8:-1;
    let s: s4 = signex(x, s4:0);
    let u: u4 = signex(x, u4:0);
    assert_eq(s as u4, u);
    assert_eq(x as s4, s);
    assert_eq(x as u4, u);
}

#[test]
fn signex_builtin_convert_to_external_typedef() {
    let x = s8:-1;
    let u32_result: u32 = signex(x, mod_imported_typedef::MY_PUBLIC_CONST);
    assert_eq(u32_result, u32::MAX);
    let u42_result: u42 = signex(x, mod_imported_typedef::MyBits:0);
    assert_eq(u42_result, u42::MAX);
    // Now do the same with a zero value as the input to sign extend..
    let y = s8:0;
    let u32_result: u32 = signex(y, mod_imported_typedef::MY_PUBLIC_CONST);
    assert_eq(u32_result, u32:0);
    let u42_result: u42 = signex(y, mod_imported_typedef::MyBits:0);
    assert_eq(u42_result, u42:0);
}
