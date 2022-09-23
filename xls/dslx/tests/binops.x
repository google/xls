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

// Tests for various sizes of binary operations.
// Simple test of these ops and that wide data types work...
// and as a starting point for debugging they don't.

fn shl_unsigned_type() -> sN[16] {
  let b = uN[3]:3;
  let x = sN[16]:5;
  x << b
}

fn shl_literal_power_of_two() -> sN[16] {
  let x = sN[16]:5;
  x << 4
}

fn shl_literal() -> sN[16] {
  let x = sN[16]:5;
  x << 3
}

fn shl_binary_literal() -> sN[16] {
  let x = sN[16]:5;
  x << 0b11
}

fn shl_hex_literal() -> sN[16] {
  let x = sN[16]:5;
  x << 0x3
}

fn shl_parametric<N: u32>() -> sN[16] {
  let x = sN[16]:5;
  x << N
}

fn shr_signed() -> sN[4] {
  let x = sN[4]:8;
  x >> 2
}

fn shr_unsigned() -> uN[4] {
  let x = uN[4]:8;
  x >> 2
}

#[test]
fn test_shifts() {
  let _ = assert_eq(s16:40, shl_unsigned_type());
  let _ = assert_eq(s16:40, shl_literal());
  let _ = assert_eq(s16:40, shl_binary_literal());
  let _ = assert_eq(s16:40, shl_hex_literal());
  let _ = assert_eq(s16:80, shl_literal_power_of_two());
  let _ = assert_eq(s16:80, shl_parametric<u32:4>());
  // TODO(https://github.com/google/xls/issues/471): 2022-01-27 Re-enable
  // (or modify) after resolving this issue.
  // let _ = assert_eq(s4:14, shr_signed());
  let _ = assert_eq(u4:2, shr_unsigned());
  ()
}

fn main32() -> sN[32] {
  let x = sN[32]:1000;
  let y = sN[32]:-1000;
  let add = x + y;
  let mul = add * y;
  let shl = mul << (y as u32);
  let shra = mul >> (x as u32);
  let shrl = (mul as u32) >> (x as u32);
  let sub = (shrl as s32) - y;
  sub / y
}

fn main1k() -> sN[1024] {
  let x = sN[1024]:1;
  let y = sN[1024]:-3;
  let add = x + y;
  let mul = add * y;
  let shl = mul << (y as u32);
  let shra = mul >> (x as u32);
  let shrl = (mul as u32) >> (x as u32);
  let sub = (shrl as sN[1024]) - y;
  sub / y
}

fn main() -> sN[128] {
  let x = sN[128]:1;
  let y = sN[128]:-3;
  let add = x + y;
  let mul = add * y;
  let shl = mul << (y as u32);
  let shra = mul >> (x as u32);
  let shrl = (mul as u32) >> (x as u32);
  let sub = (shrl as sN[128]) - y;
  sub / y
}

#[test]
fn test_main() {
  let _ = assert_eq(s32:-1, main32());
  let _ = assert_eq(sN[1024]:-2, main1k());
  let _ = assert_eq(sN[128]:-2, main());
  ()
}
