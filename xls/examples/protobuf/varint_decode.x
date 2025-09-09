#![feature(type_inference_v2)]

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

import std;
import xls.examples.protobuf.varint_encode;

// Performs a decoding of a variable-length varint value.
//
// Returns (decoded u32, consumed bytes).
//
// See https://developers.google.com/protocol-buffers/docs/encoding#varints
pub fn varint_decode_u32<NUM_BYTES:u32={std::clog2(u32:32)},
                         LEN_WIDTH:u32={std::clog2(NUM_BYTES)}>(
 bytes: u8[NUM_BYTES]) -> (u32, uN[LEN_WIDTH]) {
  type LenType = uN[LEN_WIDTH];

  let (chunks, last_chunk, _saw_last_chunk) =
   for (i, (chunks, last_chunk, saw_last_chunk)):
   (u32, (u7[NUM_BYTES], LenType, bool)) in u32:0..NUM_BYTES {
    let current_byte = bytes[i];
    let lsbs = current_byte as u7;
    let msb = current_byte[-1:];
    if saw_last_chunk {
      (chunks, last_chunk, saw_last_chunk)
    } else {
      (update(chunks, i, lsbs), i as LenType, !msb)
    }
  }((u7[NUM_BYTES]:[u7:0, ...], LenType:0, bool:0));

  // TODO(google/xls#1110): enable when quickcheck works with side-effecting ops
  // if !saw_last_chunk { fail!("did_not_see_last_chunk", ()) } else { () };
  // trace_fmt!("chunks = {}, last_chunk = {}", chunks, last_chunk);

  const FLATTENED_CHUNK_BITS = NUM_BYTES * u32:7;
  type FlattenedChunkType = uN[FLATTENED_CHUNK_BITS];
  let flattened: FlattenedChunkType =
  for (i, flattened): (u32, FlattenedChunkType) in u32:0..NUM_BYTES {
    flattened | ((chunks[i]  as FlattenedChunkType) << (u32:7 * i))
  }(zero!<FlattenedChunkType>());
  // trace_fmt!("flattened = {}", flattened);
  const NUM_EXTRA_BITS = FLATTENED_CHUNK_BITS - u32:32;
  let msbs = (flattened >> u32:32) as uN[NUM_EXTRA_BITS];

  // TODO(google/xls#1110): enable when quickcheck works with side-effecting ops
  // if msbs != uN[NUM_EXTRA_BITS]:0 {
  //   fail!("did_not_fit_in_u32", ())
  // } else { () };

  (flattened as u32, last_chunk + u3:1)
}

#[test]
fn varint_decode_u32_test() {
  let (decoded, consumed) = varint_decode_u32(u8[5]:[172, 2, 0, 0, 0]);
  assert_eq(u32:300, decoded);
  assert_eq(u3:2, consumed);
  let (decoded, consumed) = varint_decode_u32(u8[5]:[172, 2, 172, 2, 0]);
  assert_eq(u32:300, decoded);
  assert_eq(u3:2, consumed);
}

#[quickcheck(test_count=1000000)]
fn varint_u32_decode_encode_identity(x: u32) -> bool {
  let (encoded, encoded_len) = varint_encode::varint_encode_u32(x);
  let (decoded, decoded_len) = varint_decode_u32(encoded);
  x == decoded && encoded_len == decoded_len
}
