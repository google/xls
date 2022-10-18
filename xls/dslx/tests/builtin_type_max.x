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

import xls.dslx.tests.number_of_imported_type_import as noiti

type MyU128 = uN[128];
type MyU256 = uN[256];

fn main() -> MyU128 {
  MyU128::MAX
}

#[test]
fn test_builtin_max_values() {
  let _ = assert_eq(u1:0x1, u1::MAX);
  let _ = assert_eq(u2:0x3, u2::MAX);
  let _ = assert_eq(u3:0x7, u3::MAX);
  let _ = assert_eq(u4:0xf, u4::MAX);
  let _ = assert_eq(u5:0x1f, u5::MAX);
  let _ = assert_eq(u6:0x3f, u6::MAX);
  let _ = assert_eq(u7:0x7f, u7::MAX);
  let _ = assert_eq(u8:0xff, u8::MAX);
  let _ = assert_eq(u16:0xffff, u16::MAX);
  let _ = assert_eq(u32:0xffff_ffff, u32::MAX);
  let _ = assert_eq(u32:0xffff_ffff, noiti::my_type::MAX);
  let _ = assert_eq(u64:0xffff_ffff_ffff_ffff, u64::MAX);
  // TODO(https://github.com/google/xls/issues/711): the following syntax is not
  // permitted at the moment, an alias is required.
  //
  // let _ = assert_eq(uN[128]:0xffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff, uN[128]::MAX);
  let _ = assert_eq(MyU128:0xffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff, MyU128::MAX);
  let _ = assert_eq(MyU128:0xffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff, main());
  let _ = assert_eq(MyU256:0xffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff, MyU256::MAX);
}
