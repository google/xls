// Copyright 2021 The XLS Authors
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

fn binop(x: u2, y: u6) -> u8 {
  x ++ y
}

fn binop_chain(x: u1, y: u2, z: u3) -> u6 {
  x ++ y ++ z
}

fn binop_parametric<M: u32, N: u32, R: u32 = M+N>(x: bits[M], y: bits[N]) -> bits[R] {
  x ++ y
}

#[test]
fn test_main() {
  let _ = assert_eq(u8:0b11000000, binop(u2:0b11, u6:0));
  let _ = assert_eq(u8:0b00000111, binop(u2:0, u6:0b111));
  let _ = assert_eq(u6:0b100111, binop_chain(u1:1, u2:0b00, u3:0b111));
  let _ = assert_eq(u6:0b001000, binop_chain(u1:0, u2:0b01, u3:0b000));
  let _ = assert_eq(u32:0xdeadbeef, binop_parametric(u16:0xdead, u16:0xbeef));
  ()
}

// Example given in the docs.
#[test]
fn test_docs() {
  let _ = assert_eq(u8:0b11000000, u2:0b11 ++ u6:0b000000);
  let _ = assert_eq(u8:0b00000111, u2:0b00 ++ u6:0b000111);
  let _ = assert_eq(u6:0b100111, u1:1 ++ u2:0b00 ++ u3:0b111);
  let _ = assert_eq(u6:0b001000, u1:0 ++ u2:0b01 ++ u3:0b000);
  let _ = assert_eq(u32:0xdeadbeef, u16:0xdead ++ u16:0xbeef);
  ()
}
