// Copyright 2020 Google LLC
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
  let x0 = bits[32]:0xffffffff in
  let x1 = bits[32]:0x00000000 & (and_reduce(x0) as bits[32]) in
  let x2 = bits[32]:0xa5a5a5a5 | (or_reduce(x1) as bits[32]) in
  xor_reduce(x2)
}

test reductions {
  let and0 = uN[32]:0xffffffff in
  let _: () = assert_eq(uN[1]:1, and_reduce(and0)) in

  let and1 = uN[32]:0x0 in
  let _: () = assert_eq(uN[1]:0, and_reduce(and1)) in

  let and2 = uN[32]:0xa5a5a5a5 in
  let _: () = assert_eq(uN[1]:0, and_reduce(and2)) in

  let or0 = uN[32]:0xffffffff in
  let _: () = assert_eq(uN[1]:1, or_reduce(or0)) in

  let or1 = uN[32]:0x0 in
  let _: () = assert_eq(uN[1]:0, or_reduce(or1)) in

  let or2 = uN[32]:0xa5a5a5a5 in
  let _: () = assert_eq(uN[1]:1, or_reduce(or2)) in

  let xor0 = uN[32]:0xffffffff in
  let _: () = assert_eq(uN[1]:0, xor_reduce(xor0)) in

  let xor1 = uN[32]:0x0 in
  let _: () = assert_eq(uN[1]:0, xor_reduce(xor1)) in

  let xor2 = uN[32]:0xa5a5a5a5 in
  let _: () = assert_eq(uN[1]:0, xor_reduce(xor2)) in

  let xor3 = uN[32]:0x00000001 in
  let _: () = assert_eq(uN[1]:1, xor_reduce(xor3)) in

  let xor4 = uN[32]:0xb5a5a5a5 in
  let _: () = assert_eq(uN[1]:1, xor_reduce(xor4)) in

  ()
}
