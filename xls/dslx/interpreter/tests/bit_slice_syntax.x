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

// Identity function helper.
fn [N: u32] id(x: bits[N]) -> bits[N] { x }

// Slice wrapper to test parametric widths.
fn [N: u32, R: u32 = N - u32:2] get_middle_bits(x: bits[N]) -> bits[R] {
  // Slice middle bits out of parametric width.
  x[1:-1]
}

test bit_slice_syntax {
  let _ = assert_eq(u2:0b11, get_middle_bits(u4:0b0110)) in
  let _ = assert_eq(u3:0b101, get_middle_bits(u5:0b01010)) in

  let x = u6:0b100111 in
  // Slice out two bits.
  let _ = assert_eq(u2:0b11, x[0:2]) in
  let _ = assert_eq(u2:0b11, x[1:3]) in
  let _ = assert_eq(u2:0b01, x[2:4]) in
  let _ = assert_eq(u2:0b00, x[3:5]) in

  // Slice out three bits.
  let _ = assert_eq(u3:0b111, x[0:3]) in
  let _ = assert_eq(u3:0b011, x[1:4]) in
  let _ = assert_eq(u3:0b001, x[2:5]) in
  let _ = assert_eq(u3:0b100, x[3:6]) in

  // Slice out from the end.
  let _ = assert_eq(u1:0b1, x[-1:]) in
  let _ = assert_eq(u1:0b1, x[-1:6]) in
  let _ = assert_eq(u2:0b10, x[-2:]) in
  let _ = assert_eq(u2:0b10, x[-2:6]) in
  let _ = assert_eq(u3:0b100, x[-3:]) in
  let _ = assert_eq(u3:0b100, x[-3:6]) in
  let _ = assert_eq(u4:0b1001, x[-4:]) in
  let _ = assert_eq(u4:0b1001, x[-4:6]) in

  // Slice both relative to the end (MSb).
  let _ = assert_eq(u2:0b01, x[-4:-2]) in
  let _ = assert_eq(u2:0b11, x[-6:-4]) in

  // Slice out from the beginning (LSb).
  let _ = assert_eq(u5:0b00111, x[:-1]) in
  let _ = assert_eq(u4:0b0111, x[:-2]) in
  let _ = assert_eq(u3:0b111, x[:-3]) in
  let _ = assert_eq(u2:0b11, x[:-4]) in
  let _ = assert_eq(u1:0b1, x[:-5]) in

  // Slicing past the end just means we hit the end (as in Python).
  let _ = assert_eq(u1:0b1, x[5:7]) in
  let _ = assert_eq(u1:0b1, x[-7:1]) in
  let _ = assert_eq(bits[0]:0, x[-7:-6]) in
  let _ = assert_eq(bits[0]:0, x[-6:-6]) in
  let _ = assert_eq(bits[0]:0, x[6:6]) in
  let _ = assert_eq(bits[0]:0, x[6:7]) in
  let _ = assert_eq(u1:1, x[-6:-5]) in

  // Slice of a slice.
  let _ = assert_eq(u2:0b11, x[:4][1:3]) in

  // Slice of an invocation.
  let _ = assert_eq(u2:0b01, id(x)[2:4]) in

  // Explicit-width slices.
  let _ = assert_eq(u2:0b01, x[2+:u2]) in
  let _ = assert_eq(s3:0b100, x[3+:s3]) in
  let _ = assert_eq(u3:0b001, x[5+:u3]) in
  ()
}
