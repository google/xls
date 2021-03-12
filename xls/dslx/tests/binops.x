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

fn shll_unsigned_type() -> sN[16] {
  let b = uN[3]:3;
  let x = sN[16]:5;
  x << b
}

fn shll_literal_power_of_two() -> sN[16] {
  let x = sN[16]:5;
  x << 4
}

fn shll_literal() -> sN[16] {
  let x = sN[16]:5;
  x << 3
}

fn shll_binary_literal() -> sN[16] {
  let x = sN[16]:5;
  x << 0b11
}

fn shll_hex_literal() -> sN[16] {
  let x = sN[16]:5;
  x << 0x3
}

#![test]
fn test_shifts() {
  let _ = assert_eq(s16:40, shll_unsigned_type());
  let _ = assert_eq(s16:40, shll_literal());
  let _ = assert_eq(s16:40, shll_binary_literal());
  let _ = assert_eq(s16:40, shll_hex_literal());
  let _ = assert_eq(s16:80, shll_literal_power_of_two());
  ()
}

fn main32() -> sN[32] {
  let x = sN[32]:1000;
  let y = sN[32]:-1000;
  let add = x + y;
  let mul = add * y;
  let shll = mul << (y as u32);
  let shra = mul >>> (x as u32);
  let shrl = mul >> (x as u32);
  let sub = shrl - y;
  sub / y
}

fn main1k() -> sN[1024] {
  let x = sN[1024]:1;
  let y = sN[1024]:-3;
  let add = x + y;
  let mul = add * y;
  let shll = mul << (y as u32);
  let shra = mul >>> (x as u32);
  let shrl = mul >> (x as u32);
  let sub = shrl - y;
  sub / y
}

fn main() -> sN[128] {
  let x = sN[128]:1;
  let y = sN[128]:-3;
  let add = x + y;
  let mul = add * y;
  let shll = mul << (y as u32);
  let shra = mul >>> (x as u32);
  let shrl = mul >> (x as u32);
  let sub = shrl - y;
  sub / y
}

#![test]
fn test_main() {
  let _ = assert_eq(s32:-1, main32());
  let _ = assert_eq(sN[1024]:-2, main1k());
  let _ = assert_eq(sN[128]:-2, main());
  ()
}