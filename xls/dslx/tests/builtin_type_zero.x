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

import xls.dslx.tests.number_of_imported_type_import as noiti;

type MyU128 = uN[128];
type MyU256 = uN[256];

fn main() -> MyU128 { MyU128::ZERO }

#[test]
fn test_builtin_max_values() {
    assert_eq(u1:0, u1::ZERO);
    assert_eq(u2:0, u2::ZERO);
    assert_eq(u3:0, u3::ZERO);
    assert_eq(u4:0, u4::ZERO);
    assert_eq(u5:0, u5::ZERO);
    assert_eq(u6:0, u6::ZERO);
    assert_eq(u7:0, u7::ZERO);
    assert_eq(u8:0, u8::ZERO);
    assert_eq(u16:0, u16::ZERO);
    assert_eq(u32:0, u32::ZERO);
    assert_eq(u32:0, noiti::my_type::ZERO);
    assert_eq(u64:0, u64::ZERO);

    // signed types
    assert_eq(s1:0, s1::ZERO);
    assert_eq(s2:0, s2::ZERO);
    assert_eq(s3:0, s3::ZERO);
    assert_eq(s4:0, s4::ZERO);
    assert_eq(s5:0, s5::ZERO);
    assert_eq(s6:0, s6::ZERO);
    assert_eq(s7:0, s7::ZERO);
    assert_eq(s8:0, s8::ZERO);
    assert_eq(s16:0, s16::ZERO);
    assert_eq(s32:0, s32::ZERO);
    assert_eq(s32:0, noiti::my_signed_type::ZERO);
    assert_eq(s64:0, s64::ZERO);

    // TODO(https://github.com/google/xls/issues/711): the following syntax is not
    // permitted at the moment, an alias is required.
    //
    // assert_eq(uN[128]:0, uN[128]::ZERO);
    assert_eq(MyU128:0, MyU128::ZERO);
    assert_eq(MyU128:0, main());
    assert_eq(MyU256:0, MyU256::ZERO);
}
