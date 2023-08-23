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

import std
import xls.examples.protobuf.varint_decode

struct State<NUM_BYTES:u32, NUM_BYTES_WIDTH:u32={std::clog2(NUM_BYTES)}> {
  bytes: u8[NUM_BYTES],
  len: uN[NUM_BYTES_WIDTH],
}

pub proc varint_streaming_u32_decode<
  INPUT_BYTES:u32,
  OUTPUT_WORDS:u32,
  SCRATCHPAD_BYTES:u32={INPUT_BYTES},
  INPUT_BYTES_WIDTH:u32={std::clog2(INPUT_BYTES)},
  OUTPUT_WORDS_WIDTH:u32={std::clog2(OUTPUT_WORDS)},
  SCRATCHPAD_BYTES_WIDTH:u32={std::clog2(SCRATCHPAD_BYTES)},
> {
  bytes_in: chan<(u8[INPUT_BYTES], uN[INPUT_BYTES_WIDTH])> in;
  words_out: chan<(u32[OUTPUT_WORDS], uN[OUTPUT_WORDS_WIDTH])> out;

  config(bytes_in: chan<(u8[INPUT_BYTES], uN[INPUT_BYTES_WIDTH])> in,
         words_out: chan<(u32[OUTPUT_WORDS], uN[OUTPUT_WORDS_WIDTH])> out) {
    (bytes_in, words_out)
  }

  init {
    State<SCRATCHPAD_BYTES, SCRATCHPAD_BYTES_WIDTH> {
      bytes: u8[SCRATCHPAD_BYTES]:[u8:0, ...],
      len: uN[SCRATCHPAD_BYTES_WIDTH]:0,
    }
  }

  next(tok: token, state: State<SCRATCHPAD_BYTES, SCRATCHPAD_BYTES_WIDTH>) {
    const_assert!(SCRATCHPAD_BYTES >= INPUT_BYTES);

    type ScratchpadSum = uN[SCRATCHPAD_BYTES_WIDTH + u32:1];
    type OutputWordArray = u32[OUTPUT_WORDS];
    type OutputIdx = uN[OUTPUT_WORDS_WIDTH];
    type InputIdx = uN[INPUT_BYTES_WIDTH];
    type ScratchpadIdx = uN[SCRATCHPAD_BYTES_WIDTH];

    const MAX_NUM_BYTES_PER_WORD = std::clog2(u32:32);

    let has_space =
      (state.len as ScratchpadSum) + (INPUT_BYTES as ScratchpadSum) <=
      (SCRATCHPAD_BYTES as ScratchpadSum);
    type InputIdx= uN[INPUT_BYTES_WIDTH];
    let (input_tok, (input_data, input_len), _) = recv_if_non_blocking(
      tok, bytes_in, has_space, (u8[INPUT_BYTES]:[u8:0, ...], InputIdx:0));
    let state = State<SCRATCHPAD_BYTES, SCRATCHPAD_BYTES_WIDTH> {
      bytes: for (i, updated): (u32, u8[SCRATCHPAD_BYTES]) in u32:0..INPUT_BYTES {
        if (i as InputIdx) < input_len {
          update(updated, state.len + (i as ScratchpadIdx), input_data[i])
        } else {
          updated
        }
      }(state.bytes),
      len: state.len + (input_len as ScratchpadIdx),
    };

    let msbs =
    // find msbs for valid (idx < state.len) bytes, false for the rest.
    for (i, accum): (u32, bool[SCRATCHPAD_BYTES]) in u32:0..SCRATCHPAD_BYTES {
      if (i as ScratchpadIdx) < state.len { accum } else {
        update(accum, i, false)
      }
    }(map(state.bytes, std::is_unsigned_msb_set));
    // Keep state.len MSBs and zero out the lower bits.
    let num_bytes_with_msb_set =
      std::popcount(std::convert_to_bits_msb0(msbs)) as ScratchpadIdx;
    
    // each varint ends when a byte's msb is no longer set.
    let num_words_in_state = state.len - num_bytes_with_msb_set;

    let (output_words, num_output_words, bytes_taken) =
    for (i, (output_words, num_output_words, bytes_taken)):
     (u32, (OutputWordArray, OutputIdx, ScratchpadIdx)) in u32:0..OUTPUT_WORDS {
      if i < num_words_in_state as u32 {
        let taken_bytes =
        for (i, accum): (u32, u8[MAX_NUM_BYTES_PER_WORD]) in
         u32:0..MAX_NUM_BYTES_PER_WORD {
          update(accum, i, state.bytes[i + (bytes_taken as u32)])
        }(u8[MAX_NUM_BYTES_PER_WORD]:[u8:0, ...]);
        let (decoded, this_bytes_taken) = varint_decode::varint_decode_u32(
          taken_bytes);
        (
          update(output_words, i, decoded),
          num_output_words + OutputIdx:1,
          bytes_taken + (this_bytes_taken as ScratchpadIdx),
        )
      } else {
        (output_words, num_output_words, bytes_taken)
      }
    }((zero!<OutputWordArray>(), zero!<OutputIdx>(), zero!<ScratchpadIdx>()));

    let output_tok = send_if(
      input_tok,
      words_out,
      num_output_words > OutputIdx:0,
      (output_words, num_output_words));

    State {
      bytes: for (i, updated): (u32, u8[SCRATCHPAD_BYTES]) in
       u32:0..SCRATCHPAD_BYTES {
        if i + bytes_taken as u32 < SCRATCHPAD_BYTES {
          update(updated, i, state.bytes[i + bytes_taken as u32])
        } else { updated }
      }(zero!<u8[SCRATCHPAD_BYTES]>()),
      len: state.len - bytes_taken,
    }
  }
}

const TEST_INPUT_BYTES = u32:7;
const TEST_OUTPUT_WORDS = u32:3;
const TEST_SCRATCHPAD_BYTES = u32:13;
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
      chan<(u8[TEST_INPUT_BYTES], uN[TEST_INPUT_BYTES_WIDTH])>;
    let (words_s, words_r) =
      chan<(u32[TEST_OUTPUT_WORDS], uN[TEST_OUTPUT_WORDS_WIDTH])>;
    spawn varint_streaming_u32_decode<TEST_INPUT_BYTES,
                                      TEST_OUTPUT_WORDS,
                                      TEST_SCRATCHPAD_BYTES>(bytes_r, words_s);
    (bytes_s, words_r, terminator)
  }

  next(tok: token, st:()) {
    // Pump in a bunch of small numbers.
    let tok = for (_, tok): (u32, token) in u32:0..u32:100 {
      let tok = send(tok, bytes_out,
        (u8[7]:[u8:0, u8:1, u8:0, u8:1, u8:0, u8:1, u8:0], u3:7));
      let tok = send(tok, bytes_out,
        (u8[7]:[u8:1, u8:0, u8:1, u8:0, u8:1, u8:0, u8:1], u3:7));
      let tok = send(tok, bytes_out,
        (u8[7]:[u8:0, u8:1, u8:0, u8:1, u8:0, u8:1, u8:0], u3:7));
      let tok = send(tok, bytes_out,
        (u8[7]:[u8:1, u8:0, u8:1, u8:0, u8:1, u8:0, u8:1], u3:7));
      let tok = send(tok, bytes_out,
        (u8[7]:[u8:0, u8:1, u8:0, u8:1, u8:0, u8:1, u8:0], u3:7));
      send(tok, bytes_out,
        (u8[7]:[u8:1, u8:0, u8:1, u8:0, u8:1, u8:0, u8:1], u3:7))
    }(tok);
    let tok = for (_, tok): (u32, token) in u32:0..u32:100 {
      for (_, tok): (u32, token) in u32:0..u32:7 {
        let (tok, (recv_bytes, bytes_recvd)) = recv(tok, words_in);
        assert_eq(recv_bytes, u32[3]:[u32:0, u32:1, u32:0]);
        assert_eq(bytes_recvd, uN[TEST_OUTPUT_WORDS_WIDTH]:3);
        let (tok, (recv_bytes, bytes_recvd)) = recv(tok, words_in);
        assert_eq(recv_bytes, u32[3]:[u32:1, u32:0, u32:1]);
        assert_eq(bytes_recvd, uN[TEST_OUTPUT_WORDS_WIDTH]:3);

        tok
      }(tok)
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