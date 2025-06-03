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

// Performs a variable-length encoding of a 32-bit integer value.
//
// Returns (array of encoded bytes, variable length).
//
// See https://developers.google.com/protocol-buffers/docs/encoding#varints
pub fn varint_encode_u32(x: u32) -> (u8[5], u3) {
  let (varint_size, a, _): (u3, u8[5], u32) =
  for (i, (size, a, x)): (u32, (u3, u8[5], u32)) in u32:0..u32:5 {
    // Lop off the least significant seven bits.
    let lsb = x as u7;

    // Compute what bits are remaining of the original integral value.
    let remaining = x >> u32:7;

    // We put leading continue bits on bytes for the varint encoding in the
    // most significant bit position.
    let continue_bit = remaining != u32:0;

    // Update the byte entry in the varint-encoded byte array.
    let byte_entry: u8 = continue_bit ++ lsb;

    // Update the encoded byte array with this byte we've computed.
    let new_a: u8[5] = update(a, i, byte_entry);

    // When we have zero-valued bytes we're not incrementing the size anymore,
    // we had already pushed the last byte.
    let new_size = if byte_entry != u8:0 { (i+u32:1) as u3 } else { size };

    // Loop carry the size, array, and the remaining part of the original integer.
    (new_size, new_a, remaining)
  }((u3:0, u8[5]:[u8:0, ...], x));
  (a, varint_size)
}

#[test]
fn varint_encode_u32_test() {
  let (a, i) = varint_encode_u32(u32:300);
  assert_eq(u8[5]:[172, 2, 0, 0, 0], a);
  assert_eq(u3:2, i);
}
