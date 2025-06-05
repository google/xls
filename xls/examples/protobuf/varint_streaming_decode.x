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

#![feature(type_inference_v2)]

import std;
import xls.examples.protobuf.varint_decode;

// Convenience for use with map().
fn not(x: bool) -> bool { !x }

// Convenience to statically shift byte array left, filling with 'fill'.
fn byte_array_shl<SHIFT:u32, N:u32>(bytes: u8[N], fill: u8) -> u8[N] {
  for (i, arr): (u32, u8[N]) in u32:0..N {
    update(arr, i, if i + SHIFT < N { bytes[i + SHIFT] } else { fill })
  }(u8[N]:[u8:0, ...])
}

#[test]
fn byte_array_shl_test() {
  assert_eq(
    u8[4]:[u8:0, u8:1, u8:2, u8:3],
    byte_array_shl<u32:0>(u8[4]:[u8:0, u8:1, u8:2, u8:3], u8:7),
  );
  assert_eq(
    u8[4]:[u8:1, u8:2, u8:3, u8:7],
    byte_array_shl<u32:1>(u8[4]:[u8:0, u8:1, u8:2, u8:3], u8:7),
  );
  assert_eq(
    u8[4]:[u8:2, u8:3, u8:7, u8:7],
    byte_array_shl<u32:2>(u8[4]:[u8:0, u8:1, u8:2, u8:3], u8:7),
  );
  assert_eq(
    u8[4]:[u8:3, u8:7, u8:7, u8:7],
    byte_array_shl<u32:3>(u8[4]:[u8:0, u8:1, u8:2, u8:3], u8:7),
  );
  assert_eq(
    u8[4]:[u8:7, u8:7, u8:7, u8:7],
    byte_array_shl<u32:4>(u8[4]:[u8:0, u8:1, u8:2, u8:3], u8:7),
  );
}

struct State<NUM_BYTES:u32, NUM_BYTES_WIDTH:u32> {
  work_chunk: u8[NUM_BYTES],       // Chunk of bytes currently being worked on.
  len: uN[NUM_BYTES_WIDTH],        // Number of valid bytes in work_chunk
  old_bytes: u8[4],                // Leftover bytes from a previous work chunk. Guaranteed not to
                                   // have any varint terminators, else they'd already be decoded.
  old_bytes_len: u3,               // Number of valid bytes in old_bytes_len. Invalid bytes start
                                   // with index 0.
  drop_count: uN[NUM_BYTES_WIDTH], // Number of bytes to drop from work_chunk/input.
}

pub proc varint_streaming_u32_decode<
   INPUT_BYTES:u32,
   OUTPUT_WORDS:u32,
   BIG_SHIFT:u32,
   INPUT_BYTES_WIDTH:u32={std::clog2(INPUT_BYTES+u32:1)},
   OUTPUT_WORDS_WIDTH:u32={std::clog2(OUTPUT_WORDS+u32:1)},
   BIG_SHIFT_WIDTH:u32={std::clog2(BIG_SHIFT)},
   COMBINED_BYTES: u32 ={INPUT_BYTES + u32:4},
   COMBINED_BYTES_WIDTH: u32 ={std::clog2(COMBINED_BYTES + u32:1)},
> {
  bytes_in: chan<(u8[INPUT_BYTES], uN[INPUT_BYTES_WIDTH])> in;
  words_out: chan<(u32[OUTPUT_WORDS], uN[OUTPUT_WORDS_WIDTH])> out;

  config(bytes_in: chan<(u8[INPUT_BYTES], uN[INPUT_BYTES_WIDTH])> in,
         words_out: chan<(u32[OUTPUT_WORDS], uN[OUTPUT_WORDS_WIDTH])> out) {
    (bytes_in, words_out)
  }

  init {
    type MyState = State<INPUT_BYTES, INPUT_BYTES_WIDTH>;
    zero!<MyState>()
  }

  next(state: State<INPUT_BYTES, INPUT_BYTES_WIDTH>) {
    trace_fmt!("state={}", state);

    const_assert!(INPUT_BYTES >= OUTPUT_WORDS);
    const_assert!(BIG_SHIFT > u32:1 && BIG_SHIFT < INPUT_BYTES);

    type OutputWordArray = u32[OUTPUT_WORDS];
    type OutputIdx = uN[OUTPUT_WORDS_WIDTH];
    type InputIdx = uN[INPUT_BYTES_WIDTH];
    type BigShiftIdx = uN[BIG_SHIFT_WIDTH];
    type CombinedIdx = uN[COMBINED_BYTES_WIDTH];

    let terminators: bool[INPUT_BYTES] =  // terminators[i] -> word_chunk[i] terminates a varint
      map(map(state.work_chunk, std::is_unsigned_msb_set), not);
    // Remove terminators on invalid bytes.
    let terminators = for (i, terminators): (u32, bool[INPUT_BYTES]) in u32:0..INPUT_BYTES {
      if i < state.len as u32 { terminators } else { update(terminators, i, false) }
    }(terminators);
    let num_terminators = std::popcount(std::convert_to_bits_msb0(terminators)) as InputIdx;

    // Find the index of the last terminator that will be decoded in this proc iteration.
    // We can decode OUTPUT_WORDS varints per iteration, so count up to OUTPUT_WORDS and stop.
    let (_, last_terminator_idx): (OutputIdx, u32) =
    for (idx, (word_count, last_idx)): (u32, (OutputIdx, u32)) in u32:0..INPUT_BYTES {
      if terminators[idx] && word_count as u32 < OUTPUT_WORDS {
        (word_count + OutputIdx:1, idx)
      } else { (word_count, last_idx) }
    }((OutputIdx:0, u32:0));

    // Get a new input once we've processed the entire work chunk.
    let do_input = state.len == InputIdx:0;
    let (input_tok, (input_data, input_len)) = recv_if(
      join(), bytes_in, do_input, (u8[INPUT_BYTES]:[u8:0, ...], InputIdx:0));

    trace_fmt!("input_data={} input_len={}, do_input={}", input_data, input_len, do_input);

    let do_drop = state.drop_count != InputIdx:0;
    assert!(!(do_input && do_drop), "input_and_drop");

    // Each iteration, we either shift by 1 or BIG_SHIFT. Do the shifts now and select later.
    let work_chunk_shl_1 = byte_array_shl<u32:1>(state.work_chunk, u8:0);
    let work_chunk_shl_big = byte_array_shl<BIG_SHIFT>(state.work_chunk, u8:0);

    trace!(last_terminator_idx);
    trace!(num_terminators);

    // Do a big shift if we're dropping a big shift's worth of bytes or if the last terminator is
    // at or after the end of the big shift's window.
    let do_big_shift = (do_drop && state.drop_count as u32 >= BIG_SHIFT) ||
                       (!do_drop && last_terminator_idx as u32 >= BIG_SHIFT - u32:1);

    // compute next state
    let next_drop_count = if do_input {
      InputIdx:0
    } else if do_big_shift {
      InputIdx:1 + last_terminator_idx as InputIdx - BIG_SHIFT as InputIdx
    } else if do_drop {
      state.drop_count - InputIdx:1
    } else {
      last_terminator_idx as InputIdx
    };

    let next_len = if do_input {
      input_len
    } else if do_big_shift {
      state.len - BIG_SHIFT as InputIdx
    } else if do_drop {
      state.len - InputIdx:1
    } else {
      state.len - InputIdx:1
    };
    // word_chunk is either set to input, word_chunk << 1, or word_chunk << BIG_SHIFT
    let next_word_chunk = if do_input {
      input_data
     } else if do_big_shift {
      work_chunk_shl_big
     } else if do_drop {
      work_chunk_shl_1
     } else {
      work_chunk_shl_1
     };

    let (next_old_bytes, next_old_bytes_len) = if do_input {
      (state.old_bytes, state.old_bytes_len)
    } else if do_big_shift {
      (state.old_bytes, u3:0)
    } else if num_terminators == InputIdx:0 {
      let next_old_bytes_len = if state.old_bytes_len < u3:4 { state.old_bytes_len + u3:1 } else { state.old_bytes_len };
      (byte_array_shl<u32:1>(state.old_bytes, state.work_chunk[0]), next_old_bytes_len)
    } else {
      (state.old_bytes, u3:0)
    };
    trace_fmt!("next_old_bytes calc: do_input={} do_big_shift={} num_terminators={}", do_input, do_big_shift, num_terminators);
    let next_state = State {
      work_chunk: next_word_chunk,
      len: next_len,
      old_bytes: next_old_bytes,
      old_bytes_len: next_old_bytes_len,
      drop_count: next_drop_count,
    };

    // Compute and output decoded varints.
    let bytes = state.old_bytes ++ state.work_chunk;
    let bytes_len = state.len as CombinedIdx + CombinedIdx:4;

    let (output_words, num_output_words, _) =
    for (i, (output_words, num_output_words, bytes_taken)):
     (u32, (OutputWordArray, OutputIdx, CombinedIdx)) in u32:0..OUTPUT_WORDS {
      let idx = bytes_taken as u32;
      let encoded = for (j, encoded): (u32, u8[5]) in u32:0..u32:5 {
        let val = if j + idx < COMBINED_BYTES { bytes[j + idx] } else { u8: 0 };
        update(encoded, j, val)
      }(u8[5]:[u8:0, ...]);
      let (decoded, this_bytes_taken) = varint_decode::varint_decode_u32(encoded);
      let total_bytes_taken = bytes_taken + this_bytes_taken as CombinedIdx;
      trace_fmt!(
        "encoded={}, decoded={}, this_bytes_taken={}, total_bytes_taken={}, bytes_len={}",
         encoded, decoded, this_bytes_taken, total_bytes_taken, bytes_len);
      if total_bytes_taken <= bytes_len {
        (
          update(output_words, i, decoded),
          num_output_words + OutputIdx:1,
          bytes_taken + this_bytes_taken as CombinedIdx,
        )
      } else {
        (output_words, num_output_words, bytes_taken)
      }
    }((zero!<OutputWordArray>(), OutputIdx:0, (u3:4 - state.old_bytes_len) as CombinedIdx));

    trace_fmt!("ouput_words={}, num_output_words={}", output_words, num_output_words);

    let output_tok = send_if(
      input_tok,
      words_out,
      !do_input && !do_drop && num_output_words > OutputIdx:0,
      (output_words, num_output_words));

    next_state
  }
}

const TEST_INPUT_BYTES = u32:7;
const TEST_OUTPUT_WORDS = u32:3;
const TEST_BIG_SHIFT = u32:3;
const TEST_INPUT_BYTES_WIDTH = std::clog2(TEST_INPUT_BYTES);
const TEST_OUTPUT_WORDS_WIDTH = std::clog2(TEST_OUTPUT_WORDS);

#[test_proc]
proc varint_streaming_u32_decode_test {
  bytes_out: chan<(u8[TEST_INPUT_BYTES], uN[TEST_INPUT_BYTES_WIDTH])> out;
  words_in: chan<(u32[TEST_OUTPUT_WORDS], uN[TEST_OUTPUT_WORDS_WIDTH])> in;
  terminator: chan<bool> out;

  init { () }

  config (terminator: chan<bool> out) {
    let (bytes_s, bytes_r) =
      chan<(u8[TEST_INPUT_BYTES], uN[TEST_INPUT_BYTES_WIDTH])>("bytes");
    let (words_s, words_r) =
      chan<(u32[TEST_OUTPUT_WORDS], uN[TEST_OUTPUT_WORDS_WIDTH])>("words");
    spawn varint_streaming_u32_decode<TEST_INPUT_BYTES,
                                      TEST_OUTPUT_WORDS,
                                      TEST_BIG_SHIFT>(bytes_r, words_s);
    (bytes_s, words_r, terminator)
  }

  next(st:()) {
    // Pump in a bunch of small numbers.
    let tok = for (_, tok): (u32, token) in u32:0..u32:100 {
      send(tok, bytes_out,
        (u8[7]:[u8:0, u8:1, u8:0, u8:1, u8:0, u8:1, u8:0], u3:7))
    }(join());
    let tok = for (_, tok): (u32, token) in u32:0..u32:100 {
      let (tok, (recv_bytes, bytes_recvd)) = recv(tok, words_in);
      assert_eq(recv_bytes, u32[3]:[u32:0, u32:1, u32:0]);
      assert_eq(bytes_recvd, uN[TEST_OUTPUT_WORDS_WIDTH]:3);
      let (tok, (recv_bytes, bytes_recvd)) = recv(tok, words_in);
      assert_eq(recv_bytes, u32[3]:[u32:1, u32:0, u32:1]);
      assert_eq(bytes_recvd, uN[TEST_OUTPUT_WORDS_WIDTH]:3);
      let (tok, (recv_bytes, bytes_recvd)) = recv(tok, words_in);
      assert_eq(recv_bytes, u32[3]:[u32:0, u32:0, u32:0]);
      assert_eq(bytes_recvd, uN[TEST_OUTPUT_WORDS_WIDTH]:1);
      tok
    }(tok);

    // Try a big number that spans multiple inputs.
    let tok = send(tok, bytes_out, (u8[7]:[u8:172, u8:0, ...], u3:1));
    let tok = send(tok, bytes_out, (u8[7]:[u8:2, u8:0, ...], u3:1));
    let (tok, (recv_bytes, bytes_recvd)) = recv(tok, words_in);
    assert_eq(bytes_recvd, uN[TEST_OUTPUT_WORDS_WIDTH]:1);
    assert_eq(recv_bytes, u32[3]:[u32:300, u32:0, u32:0]);

    // Try multiple big numbers that span multiple inputs.
    let tok = send(tok, bytes_out, (u8[7]:[
      // WORD 1 = 2^31
      u8:128, u8:128, u8:128, u8:128, u8:8,
      // WORD 2 (first two bytes)= 2^30
      u8: 128, u8:128], u3:7));
    let tok = send(tok, bytes_out, (u8[7]:[
      // WORD 2 (last three bytes) = 2^30
      u8:128, u8:128, u8: 4,
      // WORD 3 (first four bytes) = 2^29
      u8:128, u8:128, u8:128, u8:128,
     ], u3:7));
    let tok = send(tok, bytes_out, (u8[7]:[
      // WORD 3 (last byte) = 2^29
      u8:2,
      // WORD 4 = 2^28
      u8:128, u8: 128, u8:128, u8:128, u8:0, ...
     ], u3:5));
    let tok = send(tok, bytes_out, (u8[7]:[
      u8:1,
      // WORD 5 = 2^27
      u8:128, u8:128, u8:128, u8:64, u8:0, u8:0], u3:5));

    // WORD 1
    let (tok, (recv_bytes, bytes_recvd)) = recv(tok, words_in);
    assert_eq(bytes_recvd, uN[TEST_OUTPUT_WORDS_WIDTH]:1);
    assert_eq(recv_bytes, u32[3]:[u32:1 << u32:31, u32:0, u32:0]);

    // WORD 2
    let (tok, (recv_bytes, bytes_recvd)) = recv(tok, words_in);
    assert_eq(bytes_recvd, uN[TEST_OUTPUT_WORDS_WIDTH]:1);
    assert_eq(recv_bytes, u32[3]:[u32:1 << u32:30, u32:0, u32:0]);

    // WORD 3
    let (tok, (recv_bytes, bytes_recvd)) = recv(tok, words_in);
    assert_eq(bytes_recvd, uN[TEST_OUTPUT_WORDS_WIDTH]:1);
    assert_eq(recv_bytes, u32[3]:[u32:1 << u32:29, u32:0, u32:0]);

    // WORD 4 + 5
    let (tok, (recv_bytes, bytes_recvd)) = recv(tok, words_in);
    assert_eq(bytes_recvd, uN[TEST_OUTPUT_WORDS_WIDTH]:2);
    assert_eq(recv_bytes, u32[3]:[u32:1 << u32:28, u32:1 << u32:27, u32:0]);

    // Try a full input that decodes into a full output
    let tok = send(tok, bytes_out, (u8[7]:[
      // WORD 1 = 2^28
      u8:128, u8:128, u8:128, u8:128, u8:1,
      // WORD 2 = 1
      u8:1,
      // WORD 3 = 0
      u8:0], u3:7));
    let (tok, (recv_bytes, bytes_recvd)) = recv(tok, words_in);
    assert_eq(bytes_recvd, uN[TEST_OUTPUT_WORDS_WIDTH]:3);
    assert_eq(recv_bytes, u32[3]:[u32:1 << u32:28, u32:1, u32:0]);

    send(tok, terminator, true);
  }
}
