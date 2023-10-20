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

#[test]
fn main_test() {
    let x: uN[96] = uN[96]:0xaaaa_bbbb_cccc_dddd_eeee_ffff;
    let big: uN[96] = uN[96]:0x9999_9999_9999_9999_9999_9999;
    let four: uN[96] = uN[96]:0x4;

    // Test a value which fits in an int64_t as a signed number,
    // but not in a uint64_t an unsigned number.
    let does_not_fit_in_uint64: uN[65] = uN[65]:0x1_ffff_ffff_ffff_ffff;
    assert_eq(x >> big, uN[96]:0);
    assert_eq(x >> four, uN[96]:0x0aaa_abbb_bccc_cddd_deee_efff);
    assert_eq(x << big, uN[96]:0);
    assert_eq(x << does_not_fit_in_uint64, uN[96]:0);
    assert_eq(x << four, uN[96]:0xaaab_bbbc_cccd_ddde_eeef_fff0)
}
