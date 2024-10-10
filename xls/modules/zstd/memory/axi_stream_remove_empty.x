// Copyright 2024 The XLS Authors
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

// This file contains implementation of AxiStreamRemoveEmpty proc,
// which is used to remove bytes marked as containing no data in the Axi Stream

import std;
import xls.modules.zstd.memory.axi_st;

struct AxiStreamRemoveEmptyState<
    DATA_W: u32, DEST_W: u32, ID_W: u32,
    DATA_W_LOG2: u32 = {std::clog2(DATA_W + u32:1)},
> {
    data: uN[DATA_W],
    len: uN[DATA_W_LOG2],
    last: bool,
    id: uN[ID_W],
    dest: uN[DEST_W],
}


// Returns a tuple containing data and length, afer removing non-data
// bytes from the in_data varaiable, using information from keep and str fields
fn remove_empty_bytes<DATA_W: u32, DATA_W_DIV8: u32, DATA_W_LOG2: u32> (
    in_data: uN[DATA_W], keep: uN[DATA_W_DIV8], str: uN[DATA_W_DIV8]
) -> (uN[DATA_W], uN[DATA_W_LOG2]) {

    const EXT_OFFSET_W = DATA_W_LOG2 + u32:3;

    type Data = uN[DATA_W];
    type Str = uN[DATA_W_DIV8];
    type Keep = uN[DATA_W_DIV8];
    type Offset = uN[DATA_W_LOG2];
    type OffsetExt = uN[EXT_OFFSET_W];
    type Length = uN[DATA_W_LOG2];

    let (data, len, _) = for (i, (data, len, offset)): (u32, (Data, Length, Offset)) in range(u32:0, DATA_W_DIV8) {
        if str[i +: u1] & keep[i +: u1] {
            (
                data | (in_data & (Data:0xFF << (u32:8 * i))) >> (OffsetExt:8 * offset as OffsetExt),
                len + Length:8,
                offset,
            )
        } else {
            (data, len, offset + Offset:1)
        }
    }((Data:0, Length:0, Offset:0));
    (data, len)
}

// Returns the number of bytes that should be soted in the state in case we
// ar not able to send all of them in a single transaction.
fn get_overflow_len<DATA_W: u32, LENGTH_W: u32>(len1: uN[LENGTH_W], len2: uN[LENGTH_W]) -> uN[LENGTH_W] {
    const LENGTH_W_PLUS_ONE = LENGTH_W + u32:1;
    type LengthPlusOne = uN[LENGTH_W_PLUS_ONE];

    const MAX_DATA_LEN = DATA_W as LengthPlusOne;
    (len1 as LengthPlusOne + len2 as LengthPlusOne - MAX_DATA_LEN) as uN[LENGTH_W]
}

// Return the new mask for keep and str fields, calculated using new data length
fn get_mask<DATA_W: u32, DATA_W_DIV8: u32, DATA_W_LOG2: u32>(len: uN[DATA_W_LOG2]) -> uN[DATA_W_DIV8] {
    const MAX_LEN = DATA_W as uN[DATA_W_LOG2];
    const MASK = !uN[DATA_W_DIV8]:0;

    let shift = std::div_pow2((MAX_LEN - len), uN[DATA_W_LOG2]:8);
    MASK >> shift
}

// A proc that removes empty bytes from the Axi Stream and provides aligned data
// to other procs, allowing for a simpler implementation of the receiving side
// of the design.
pub proc AxiStreamRemoveEmpty<
    DATA_W: u32, DEST_W: u32, ID_W: u32,
    DATA_W_DIV8: u32 = {DATA_W / u32:8},
    DATA_W_LOG2: u32 = {std::clog2(DATA_W + u32:1)},
> {
    type AxiStream = axi_st::AxiStream<DATA_W, DEST_W, ID_W, DATA_W_DIV8>;
    type State = AxiStreamRemoveEmptyState<DATA_W, DEST_W, ID_W, DATA_W_LOG2>;

    type Offset = uN[DATA_W_LOG2];
    type Length = uN[DATA_W_LOG2];
    type Keep = uN[DATA_W_DIV8];
    type Str = uN[DATA_W_DIV8];
    type Data = uN[DATA_W];

    stream_in_r: chan<AxiStream> in;
    stream_out_s: chan<AxiStream> out;

    config (
        stream_in_r: chan<AxiStream> in,
        stream_out_s: chan<AxiStream> out,
    ) {
        (stream_in_r, stream_out_s)
    }

    init { zero!<State>() }

    next (state: State) {
        const MAX_LEN = DATA_W as Length;
        const MAX_MASK = !uN[DATA_W_DIV8]:0;

        let do_recv = !state.last;
        let (tok, stream_in) = recv_if(join(), stream_in_r, !state.last, zero!<AxiStream>());
        let (id, dest) = if !state.last {
            (stream_in.id, stream_in.dest)
        } else {
            (state.id, state.dest)
        };

        let (data, len) = remove_empty_bytes<DATA_W, DATA_W_DIV8, DATA_W_LOG2>(
            stream_in.data, stream_in.keep, stream_in.str
        );

        let empty_input_bytes = MAX_LEN - len;
        let empty_state_bytes = MAX_LEN - state.len;

        let exceeds_transfer = (empty_input_bytes < state.len);
        let exact_transfer = (empty_input_bytes == state.len);

        let combined_state_data = state.data | data << state.len;
        let combined_input_data = data | state.data << len;

        let overflow_len = get_overflow_len<DATA_W>(state.len, len);
        let sum_len = state.len + len;
        let sum_mask = get_mask<DATA_W, DATA_W_DIV8, DATA_W_LOG2>(sum_len);

        let (next_state, do_send, data) = if !state.last & exceeds_transfer {
            // flush and store
            (
                State {
                    data: data >> empty_state_bytes,
                    len: overflow_len,
                    last: stream_in.last,
                    id: stream_in.id,
                    dest: stream_in.dest,
                },
                true,
                AxiStream {
                    data: combined_state_data,
                    str: MAX_MASK,
                    keep: MAX_MASK,
                    last: false,
                    id, dest
                }
            )
        } else if state.last | stream_in.last | exact_transfer {
            // flush only
            (
                zero!<State>(),
                true,
                AxiStream {
                    data: combined_state_data,
                    str: sum_mask,
                    keep: sum_mask,
                    last: state.last | stream_in.last,
                    id, dest
                }
            )
        } else {
            // store
            (
                State {
                    data: combined_input_data,
                    len: sum_len,
                    ..state
                },
                false,
                zero!<AxiStream>(),
            )
        };

        send_if(tok, stream_out_s, do_send, data);
        next_state
    }
}


const INST_DATA_W = u32:32;
const INST_DEST_W = u32:32;
const INST_ID_W = u32:32;

const INST_DATA_W_DIV8 = INST_DATA_W / u32:8;
const INST_DATA_W_LOG2 = std::clog2(INST_DATA_W + u32:1);

type InstAxiStream = axi_st::AxiStream<INST_DATA_W, INST_DEST_W, INST_ID_W, INST_DATA_W_DIV8>;

proc AxiStreamRemoveEmptyInst {
    config (
        stream_in_r: chan<InstAxiStream> in,
        stream_out_s: chan<InstAxiStream> out,
    ) {
        spawn AxiStreamRemoveEmpty<INST_DATA_W, INST_DEST_W, INST_ID_W, INST_DATA_W_DIV8, INST_DATA_W_LOG2> (
            stream_in_r,
            stream_out_s
        );
    }

    init { }

    next (state:()) { }
}


const TEST_DATA_W = u32:32;
const TEST_DEST_W = u32:32;
const TEST_ID_W = u32:32;
const TEST_DATA_W_DIV8 = TEST_DATA_W / u32:8;

type TestAxiStream = axi_st::AxiStream<TEST_DATA_W, TEST_DEST_W, TEST_ID_W, TEST_DATA_W_DIV8>;

#[test_proc]
proc AxiStreamRemoveEmptyTest {
    terminator: chan<bool> out;
    stream_in_s: chan<TestAxiStream> out;
    stream_out_r: chan<TestAxiStream> in;

    config (
        terminator: chan<bool> out,
    ) {
        let (stream_in_s, stream_in_r) = chan<TestAxiStream>("stream_in");
        let (stream_out_s, stream_out_r) = chan<TestAxiStream>("stream_out");

        spawn AxiStreamRemoveEmpty<TEST_DATA_W, TEST_DEST_W, TEST_ID_W>(
            stream_in_r, stream_out_s
        );

        (terminator, stream_in_s, stream_out_r)
    }

    init { }

    next (state: ()) {

        type Data = uN[TEST_DATA_W];
        type Keep = uN[TEST_DATA_W_DIV8];
        type Str = uN[TEST_DATA_W_DIV8];
        type Id = uN[TEST_ID_W];
        type Dest = uN[TEST_DEST_W];

        let tok = join();

        // Test 1: All bits set, last set
        let tok = send(tok, stream_in_s, TestAxiStream {
            data: Data:0xAABB_CCDD,
            str: Str:0xF,
            keep: Keep:0xF,
            last: u1:1,
            id: Id:3,
            dest: Dest:0,
        });

        let (tok, stream_out) = recv(tok, stream_out_r);
        assert_eq(stream_out, TestAxiStream {
            data: Data:0xAABB_CCDD,
            str: Str:0xF,
            keep: Keep:0xF,
            last: u1:1,
            id: Id:3,
            dest: Dest:0,
        });

        // Test 2: Non of bits set, last set
        let tok = send(tok, stream_in_s, TestAxiStream {
            data: Data:0xAABB_CCDD,
            str: Str:0x0,
            keep: Keep:0x0,
            last: u1:1,
            id: Id:3,
            dest: Dest:0,
        });

        let (tok, stream_out) = recv(tok, stream_out_r);
        assert_eq(stream_out, TestAxiStream {
            data: Data:0x0,
            str: Str:0x0,
            keep: Keep:0x0,
            last: u1:1,
            id: Id:3,
            dest: Dest:0,
        });

        // Test 3: Some bits set, last set
        let tok = send(tok, stream_in_s, TestAxiStream {
            data: Data:0xAABB_CCDD,
            str: Str:0x5,
            keep: Keep:0x5,
            last: u1:1,
            id: Id:3,
            dest: Dest:0,
        });

        let (tok, stream_out) = recv(tok, stream_out_r);
        assert_eq(stream_out, TestAxiStream {
            data: Data:0xBBDD,
            str: Str:0x3,
            keep: Keep:0x3,
            last: u1:1,
            id: Id:3,
            dest: Dest:0,
        });

        // Test 4: Some bits set, last set in the last transfer.
        // The last transfer is aligned

        let tok = send(tok, stream_in_s, TestAxiStream {
            data: Data:0x0000_BBAA,
            str: Str:0b0011,
            keep: Keep:0b0011,
            last: u1:0,
            id: Id:0,
            dest: Dest:0,
        });
        let tok = send(tok, stream_in_s, TestAxiStream {
            data: Data:0xFFEE_DDCC,
            str: Str:0b1111,
            keep: Keep:0b1111,
            last: u1:0,
            id: Id:0,
            dest: Dest:0,
        });
        let tok = send(tok, stream_in_s, TestAxiStream {
            data: Data:0x0022_0011,
            str: Str:0b0101,
            keep: Keep:0b0101,
            last: u1:1,
            id: Id:0,
            dest: Dest:0,
        });

        let (tok, stream_out) = recv(tok, stream_out_r);
        assert_eq(stream_out, TestAxiStream {
            data: Data:0xDDCC_BBAA,
            str: Str:0xF,
            keep: Keep:0xF,
            last: u1:0,
            id: Id:0,
            dest: Dest:0,
        });
        let (tok, stream_out) = recv(tok, stream_out_r);
        assert_eq(stream_out, TestAxiStream {
            data: Data:0x2211_FFEE,
            str: Str:0xF,
            keep: Keep:0xF,
            last: u1:1,
            id: Id:0,
            dest: Dest:0,
        });

        // Test 5: Some bits set, last set in the last transfer.
        // The last transfer is not aligned

        let tok = send(tok, stream_in_s, TestAxiStream {
            data: Data:0x00BB_00AA,
            str: Str:0b0101,
            keep: Keep:0b0101,
            last: u1:0,
            id: Id:0,
            dest: Dest:0,
        });
        let tok = send(tok, stream_in_s, TestAxiStream {
            data: Data:0x00DD_CC00,
            str: Str:0b0110,
            keep: Keep:0b0110,
            last: u1:0,
            id: Id:0,
            dest: Dest:0,
        });
        let tok = send(tok, stream_in_s, TestAxiStream {
            data: Data:0xFF00_00EE,
            str: Str:0b1001,
            keep: Keep:0b1001,
            last: u1:1,
            id: Id:0,
            dest: Dest:0,
        });

        let (tok, stream_out) = recv(tok, stream_out_r);
        assert_eq(stream_out, TestAxiStream {
            data: Data:0xDDCC_BBAA,
            str: Str:0xF,
            keep: Keep:0xF,
            last: u1:0,
            id: Id:0,
            dest: Dest:0,
        });
        let (tok, stream_out) = recv(tok, stream_out_r);
        assert_eq(stream_out, TestAxiStream {
            data: Data:0xFFEE,
            str: Str:0x3,
            keep: Keep:0x3,
            last: u1:1,
            id: Id:0,
            dest: Dest:0,
        });

        send(tok, terminator, true);
    }
}
