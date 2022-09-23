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

// (Dummy) wrapper around reverse.
fn wrapper<N: u32>(x: bits[N]) -> bits[N] {
  rev(x)
}

// Target for IR conversion that works on u3s.
fn main(x: u3) -> u3 {
  wrapper(x)
}

// Reverse examples.
#[test]
fn test_reverse() {
  let _ = assert_eq(u3:0b100, main(u3:0b001));
  let _ = assert_eq(u3:0b001, main(u3:0b100));
  let _ = assert_eq(bits[0]:0, rev(bits[0]:0));
  let _ = assert_eq(u1:1, rev(u1:1));
  let _ = assert_eq(u2:0b10, rev(u2:0b01));
  let _ = assert_eq(u2:0b00, rev(u2:0b00));
  ()
}

// Reversing a value twice gets you the original value.
#[quickcheck]
fn prop_double_reverse(x: u32) -> bool {
  x == rev(rev(x))
}

// Reversing a value means that the lsb becomes the msb.
#[quickcheck]
fn prop_lsb_becomes_msb(x: u32) -> bool {
  let reversed_x = rev(x);
  x[0:1] == reversed_x[-1:]
}
