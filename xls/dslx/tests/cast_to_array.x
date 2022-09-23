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

fn cast_to_array(x: u6) -> u2[3] {
  x as u2[3]
}

fn cast_from_array(a: u2[3]) -> u6 {
  a as u6
}

fn concat_arrays(a: u2[3], b: u2[3]) -> u2[6] {
  a ++ b
}

#[test]
fn cast_to_array_test() {
  let a_value: u6 = u6:0b011011;
  let a: u2[3] = cast_to_array(a_value);
  let a_array = u2[3]:[1, 2, 3];
  let _ = assert_eq(a, a_array);
  // Note: converting back from array to bits gives the original value.
  let _ = assert_eq(a_value, cast_from_array(a));

  let b_value: u6 = u6:0b111001;
  let b_array: u2[3] = u2[3]:[3, 2, 1];
  let b: u2[3] = cast_to_array(b_value);
  let _ = assert_eq(b, b_array);
  let _ = assert_eq(b_value, cast_from_array(b));

  // Concatenation of bits is analogous to concatenation of their converted
  // arrays. That is:
  //
  //  convert(concat(a, b)) == concat(convert(a), convert(b))
  let concat_value: u12 = a_value ++ b_value;
  let concat_array: u2[6] = concat_value as u2[6];
  let _ = assert_eq(concat_array, concat_arrays(a_array, b_array));

  // Show a few classic "endianness" example using 8-bit array values.
  let x = u32:0xdeadbeef;
  let _ = assert_eq(x as u8[4], u8[4]:[0xde, 0xad, 0xbe, 0xef]);
  let y = u16:0xbeef;
  let _ = assert_eq(y as u8[2], u8[2]:[0xbe, 0xef]);

  ()
}
