// Copyright 2022 The XLS Authors
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
type MyS65 = sN[65];

fn main() -> (MyU128, MyU128) { (MyU128::MAX, MyU128::MIN) }

#[test]
fn test_builtin_max_values() {
    // unsigned types max
    assert_eq(u1:0x1, u1::MAX);
    assert_eq(u2:0x3, u2::MAX);
    assert_eq(u3:0x7, u3::MAX);
    assert_eq(u4:0xf, u4::MAX);
    assert_eq(u5:0x1f, u5::MAX);
    assert_eq(u6:0x3f, u6::MAX);
    assert_eq(u7:0x7f, u7::MAX);
    assert_eq(u8:0xff, u8::MAX);
    assert_eq(u16:0xffff, u16::MAX);
    assert_eq(u32:0xffff_ffff, u32::MAX);
    assert_eq(u32:0xffff_ffff, noiti::my_type::MAX);
    assert_eq(u64:0xffff_ffff_ffff_ffff, u64::MAX);

    // unsigned types min
    assert_eq(u1:0, u1::MIN);
    assert_eq(u2:0, u2::MIN);
    assert_eq(u3:0, u3::MIN);
    assert_eq(u4:0, u4::MIN);

    // signed types max
    assert_eq(s1:0b0, s1::MAX);
    assert_eq(s2:0b01, s2::MAX);
    assert_eq(s3:0b011, s3::MAX);
    assert_eq(s4:0b0111, s4::MAX);
    assert_eq(s5:0xf, s5::MAX);
    assert_eq(s6:0x1f, s6::MAX);
    assert_eq(s7:0x3f, s7::MAX);
    assert_eq(s8:0x7f, s8::MAX);
    assert_eq(s16:0x7fff, s16::MAX);
    assert_eq(s32:0x7fff_ffff, s32::MAX);
    assert_eq(s32:0x7fff_ffff, noiti::my_signed_type::MAX);
    assert_eq(s64:0x7fff_ffff_ffff_ffff, s64::MAX);

    // signed types min
    assert_eq(s1:0b1, s1::MIN);
    assert_eq(s2:0b10, s2::MIN);
    assert_eq(s3:0b100, s3::MIN);
    assert_eq(s8:-128, s8::MIN);
    assert_eq(MyS65:0b10000000000000000000000000000000000000000000000000000000000000000, MyS65::MIN);

    // TODO(https://github.com/google/xls/issues/711): the following syntax is not
    // permitted at the moment, an alias is required.
    //
    // assert_eq(uN[128]:0xffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff, uN[128]::MAX);
    assert_eq(MyU128:0xffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff, MyU128::MAX);
    assert_eq((MyU128:0xffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff, MyU128:0), main());
    assert_eq(
        MyU256:0xffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff,
        MyU256::MAX);
}
