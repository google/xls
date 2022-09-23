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


fn main() -> uN[1] {
  let x0 = bits[32]:0xffffffff;
  let x1 = bits[32]:0x00000000 & (and_reduce(x0) as bits[32]);
  let x2 = bits[32]:0xa5a5a5a5 | (or_reduce(x1) as bits[32]);
  xor_reduce(x2)
}

#[test]
fn reductions() {
  let and0 = uN[32]:0xffffffff;
  let _: () = assert_eq(uN[1]:1, and_reduce(and0));

  let and1 = uN[32]:0x0;
  let _: () = assert_eq(uN[1]:0, and_reduce(and1));

  let and2 = uN[32]:0xa5a5a5a5;
  let _: () = assert_eq(uN[1]:0, and_reduce(and2));

  let or0 = uN[32]:0xffffffff;
  let _: () = assert_eq(uN[1]:1, or_reduce(or0));

  let or1 = uN[32]:0x0;
  let _: () = assert_eq(uN[1]:0, or_reduce(or1));

  let or2 = uN[32]:0xa5a5a5a5;
  let _: () = assert_eq(uN[1]:1, or_reduce(or2));

  let xor0 = uN[32]:0xffffffff;
  let _: () = assert_eq(uN[1]:0, xor_reduce(xor0));

  let xor1 = uN[32]:0x0;
  let _: () = assert_eq(uN[1]:0, xor_reduce(xor1));

  let xor2 = uN[32]:0xa5a5a5a5;
  let _: () = assert_eq(uN[1]:0, xor_reduce(xor2));

  let xor3 = uN[32]:0x00000001;
  let _: () = assert_eq(uN[1]:1, xor_reduce(xor3));

  let xor4 = uN[32]:0xb5a5a5a5;
  let _: () = assert_eq(uN[1]:1, xor_reduce(xor4));

  ()
}
