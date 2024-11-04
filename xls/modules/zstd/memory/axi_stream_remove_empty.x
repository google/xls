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

pub struct ContinuousStream<
    DATA_W: u32,
    DEST_W: u32,
    ID_W: u32,
    DATA_W_LOG2: u32 = {std::clog2(DATA_W + u32:1)},
> {
    data: uN[DATA_W],
    len: uN[DATA_W_LOG2],
    id: uN[ID_W],
    dest: uN[DEST_W],
    last: u1
}

const INST_DATA_W = u32:32;
const INST_DATA_W_DIV8 = INST_DATA_W / u32:8;
const INST_DATA_W_LOG2 = std::clog2(INST_DATA_W + u32:1);
const INST_DEST_W = u32:32;
const INST_ID_W = u32:32;
const TEST_DATA_W = u32:32;
const TEST_DATA_W_DIV8 = TEST_DATA_W / u32:8;
const TEST_DATA_W_LOG2 = std::clog2(TEST_DATA_W + u32:1);
const TEST_DEST_W = u32:32;
const TEST_ID_W = u32:32;

// Returns a tuple containing data and length, afer removing non-data
// bytes from the in_data varaiable, using information from keep and str fields
pub proc RemoveEmptyBytes<
    DATA_W: u32, DEST_W: u32, ID_W: u32,
    DATA_W_DIV8: u32 = {DATA_W / u32:8},
    DATA_W_LOG2: u32 = {std::clog2(DATA_W + u32:1)},
    EXT_OFFSET_W: u32 = {(std::clog2(DATA_W + u32:1)) + u32:3},
> {
    type Data = uN[DATA_W];
    type Str = uN[DATA_W_DIV8];
    type Offset = uN[DATA_W_LOG2];
    type OffsetExt = uN[EXT_OFFSET_W];
    type Length = uN[DATA_W_LOG2];

    type AxiStream = axi_st::AxiStream<DATA_W, DEST_W, ID_W, DATA_W_DIV8>;
    type StrobedStream = ContinuousStream<DATA_W, DEST_W, ID_W>;

    stream_r: chan<AxiStream> in;
    continuous_stream_s: chan<StrobedStream> out;

    config (
        stream_r: chan<AxiStream> in,
        continuous_stream_s: chan<StrobedStream> out,
    ) {
        (stream_r, continuous_stream_s)
    }

    init { () }

    next (state: ()) {
        let (tok, frame) = recv(join(), stream_r);
        let (in_data, str) = (frame.data, frame.str);

        let (data, len, _) = unroll_for! (i, (data, len, offset)): (u32, (Data, Length, Offset)) in range(u32:0, DATA_W_DIV8) {
            if str[i +: u1] {
                (
                    data | (in_data & (Data:0xFF << (u32:8 * i))) >> (OffsetExt:8 * offset as OffsetExt),
                    len + Length:8,
                    offset,
                )
            } else {
                (data, len, offset + Offset:1)
            }
        }((Data:0, Length:0, Offset:0));

        let continuous_stream = StrobedStream {
            data: data,
            len: len,
            id: frame.id,
            dest: frame.dest,
            last: frame.last,
        };
        send(tok, continuous_stream_s, continuous_stream);
    }
}

pub proc RemoveEmptyBytesInst {
    type AxiStream = axi_st::AxiStream<INST_DATA_W, INST_DEST_W, INST_ID_W, INST_DATA_W_DIV8>;
    type StrobedStream = ContinuousStream<INST_DATA_W, INST_DEST_W, INST_ID_W>;

    config (
        stream_r: chan<AxiStream> in,
        continuous_stream_s: chan<StrobedStream> out,
    ) {
        spawn RemoveEmptyBytes<INST_DATA_W, INST_DEST_W, INST_ID_W>(
            stream_r, continuous_stream_s
        );
    }

    init { () }

    next (state: ()) {}
}

#[test_proc]
proc RemoveEmptyBytesTest {
    type TestAxiStream = axi_st::AxiStream<TEST_DATA_W, TEST_DEST_W, TEST_ID_W, TEST_DATA_W_DIV8>;
    type TestStrobedStream = ContinuousStream<TEST_DATA_W, TEST_DEST_W, TEST_ID_W>;
    terminator: chan<bool> out;
    stream_s: chan<TestAxiStream> out;
    continuous_stream_r: chan<TestStrobedStream> in;

    config (
        terminator: chan<bool> out,
    ) {
        let (stream_s, stream_r) = chan<TestAxiStream>("frame_data");
        let (continuous_stream_s, continuous_stream_r) = chan<TestStrobedStream>("bare_data");

        spawn RemoveEmptyBytes<TEST_DATA_W, TEST_DEST_W, TEST_ID_W>(
            stream_r, continuous_stream_s
        );

        (terminator, stream_s, continuous_stream_r)
    }

    init { }

    next (state: ()) {
        type Data = uN[TEST_DATA_W];
        type Str = uN[TEST_DATA_W_DIV8];
        type Id = uN[TEST_ID_W];
        type Dest = uN[TEST_DEST_W];
        type Length = uN[TEST_DATA_W_LOG2];

        let tok = join();

        let data = Data:0xDEADBEEF;
        let input_data: TestAxiStream[16] = [
            TestAxiStream{data: data, str: Str:0b0000, keep: Str:0b0000, id: Id:0, dest: Dest:0, last: false},
            TestAxiStream{data: data, str: Str:0b0001, keep: Str:0b0001, id: Id:0, dest: Dest:0, last: false},
            TestAxiStream{data: data, str: Str:0b0010, keep: Str:0b0010, id: Id:0, dest: Dest:0, last: false},
            TestAxiStream{data: data, str: Str:0b0011, keep: Str:0b0011, id: Id:0, dest: Dest:0, last: false},
            TestAxiStream{data: data, str: Str:0b0100, keep: Str:0b0100, id: Id:0, dest: Dest:0, last: false},
            TestAxiStream{data: data, str: Str:0b0101, keep: Str:0b0101, id: Id:0, dest: Dest:0, last: false},
            TestAxiStream{data: data, str: Str:0b0110, keep: Str:0b0110, id: Id:0, dest: Dest:0, last: false},
            TestAxiStream{data: data, str: Str:0b0111, keep: Str:0b0111, id: Id:0, dest: Dest:0, last: false},
            TestAxiStream{data: data, str: Str:0b1000, keep: Str:0b1000, id: Id:0, dest: Dest:0, last: false},
            TestAxiStream{data: data, str: Str:0b1001, keep: Str:0b1001, id: Id:0, dest: Dest:0, last: false},
            TestAxiStream{data: data, str: Str:0b1010, keep: Str:0b1010, id: Id:0, dest: Dest:0, last: false},
            TestAxiStream{data: data, str: Str:0b1011, keep: Str:0b1011, id: Id:0, dest: Dest:0, last: false},
            TestAxiStream{data: data, str: Str:0b1100, keep: Str:0b1100, id: Id:0, dest: Dest:0, last: false},
            TestAxiStream{data: data, str: Str:0b1101, keep: Str:0b1101, id: Id:0, dest: Dest:0, last: false},
            TestAxiStream{data: data, str: Str:0b1110, keep: Str:0b1110, id: Id:0, dest: Dest:0, last: false},
            TestAxiStream{data: data, str: Str:0b1111, keep: Str:0b1111, id: Id:0, dest: Dest:0, last: true}
        ];
        let expected_output: TestStrobedStream[16] = [
            TestStrobedStream{data: Data:0x00, len: Length:0, id: Id:0, dest: Dest:0, last: false},
            TestStrobedStream{data: Data:0xEF, len: Length:8, id: Id:0, dest: Dest:0, last: false},
            TestStrobedStream{data: Data:0xBE, len: Length:8, id: Id:0, dest: Dest:0, last: false},
            TestStrobedStream{data: Data:0xBEEF, len: Length:16, id: Id:0, dest: Dest:0, last: false},
            TestStrobedStream{data: Data:0xAD, len: Length:8, id: Id:0, dest: Dest:0, last: false},
            TestStrobedStream{data: Data:0xADEF, len: Length:16, id: Id:0, dest: Dest:0, last: false},
            TestStrobedStream{data: Data:0xADBE, len: Length:16, id: Id:0, dest: Dest:0, last: false},
            TestStrobedStream{data: Data:0xADBEEF, len: Length:24, id: Id:0, dest: Dest:0, last: false},
            TestStrobedStream{data: Data:0xDE, len: Length:8, id: Id:0, dest: Dest:0, last: false},
            TestStrobedStream{data: Data:0xDEEF, len: Length:16, id: Id:0, dest: Dest:0, last: false},
            TestStrobedStream{data: Data:0xDEBE, len: Length:16, id: Id:0, dest: Dest:0, last: false},
            TestStrobedStream{data: Data:0xDEBEEF, len: Length:24, id: Id:0, dest: Dest:0, last: false},
            TestStrobedStream{data: Data:0xDEAD, len: Length:16, id: Id:0, dest: Dest:0, last: false},
            TestStrobedStream{data: Data:0xDEADEF, len: Length:24, id: Id:0, dest: Dest:0, last: false},
            TestStrobedStream{data: Data:0xDEADBE, len: Length:24, id: Id:0, dest: Dest:0, last: false},
            TestStrobedStream{data: Data:0xDEADBEEF, len: Length:32, id: Id:0, dest: Dest:0, last: true}
        ];

        let tok = for (i, tok): (u32, token) in range(u32:0, u32:16) {
            let tok = send(tok, stream_s, input_data[i]);
            trace_fmt!("TestRemoveEmptyBytes: Sent #{} strobed packet: {:#x}", i + u32:1, input_data[i]);
            let (tok, continuous_stream) = recv(tok, continuous_stream_r);
            trace_fmt!("TestRemoveEmptyBytes: Received #{} continuous packet: {:#x}", i + u32:1, continuous_stream);
            assert_eq(continuous_stream, expected_output[i]);
            (tok)
        } (tok);

        send(tok, terminator, true);
    }
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
    let len_bytes = std::div_pow2(len, uN[DATA_W_LOG2]:8);
    let mask = (uN[DATA_W_DIV8]:1 << len_bytes as uN[DATA_W_DIV8]) - uN[DATA_W_DIV8]:1;

    mask
}

// A proc that removes empty bytes from the Axi Stream and provides aligned data
// to other procs, allowing for a simpler implementation of the receiving side
// of the design.
pub proc AxiStreamRemoveEmptyInternal<
    DATA_W: u32, DEST_W: u32, ID_W: u32,
    DATA_W_DIV8: u32 = {DATA_W / u32:8},
    DATA_W_LOG2: u32 = {std::clog2(DATA_W + u32:1)},
> {
    type AxiStream = axi_st::AxiStream<DATA_W, DEST_W, ID_W, DATA_W_DIV8>;
    type StrobedStream = ContinuousStream<DATA_W, DEST_W, ID_W>;

    type State = AxiStreamRemoveEmptyState<DATA_W, DEST_W, ID_W, DATA_W_LOG2>;

    type Offset = uN[DATA_W_LOG2];
    type Length = uN[DATA_W_LOG2];
    type Keep = uN[DATA_W_DIV8];
    type Str = uN[DATA_W_DIV8];
    type Data = uN[DATA_W];

    stream_in_r: chan<StrobedStream> in;
    stream_out_s: chan<AxiStream> out;

    config (
        stream_in_r: chan<StrobedStream> in,
        stream_out_s: chan<AxiStream> out,
    ) {
        (stream_in_r, stream_out_s)
    }

    init { zero!<State>() }

    next (state: State) {
        const MAX_LEN = DATA_W as Length;
        const MAX_MASK = !uN[DATA_W_DIV8]:0;

        let do_recv = !state.last;
        let (tok, stream_in) = recv_if(join(), stream_in_r, do_recv, zero!<StrobedStream>());
        let (id, dest, data, len) = if do_recv {
            (stream_in.id, stream_in.dest, stream_in.data, stream_in.len)
        } else {
            (state.id, state.dest, Data:0, Length:0)
        };

        let empty_input_bytes = MAX_LEN - len;
        let empty_state_bytes = MAX_LEN - state.len;

        let exceeds_transfer = (empty_input_bytes < state.len);
        let exact_transfer = (empty_input_bytes == state.len);

        let combined_state_data = state.data | data << state.len;

        let sum_len = state.len + len;

        let (next_state, do_send, data) = if !state.last & exceeds_transfer {
            // flush and store
            let overflow_len = get_overflow_len<DATA_W>(state.len, len);
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
            let sum_mask = get_mask<DATA_W, DATA_W_DIV8, DATA_W_LOG2>(sum_len);
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
                    data: combined_state_data,
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

type InstAxiStream = axi_st::AxiStream<INST_DATA_W, INST_DEST_W, INST_ID_W, INST_DATA_W_DIV8>;
type InstStrobedStream = ContinuousStream<INST_DATA_W, INST_DEST_W, INST_ID_W>;

proc AxiStreamRemoveEmptyInternalInst {
    config (
        stream_in_r: chan<InstStrobedStream> in,
        stream_out_s: chan<InstAxiStream> out,
    ) {
        spawn AxiStreamRemoveEmptyInternal<INST_DATA_W, INST_DEST_W, INST_ID_W, INST_DATA_W_DIV8, INST_DATA_W_LOG2> (
            stream_in_r,
            stream_out_s
        );
    }

    init { }

    next (state:()) { }
}

pub proc AxiStreamRemoveEmpty<
    DATA_W: u32, DEST_W: u32, ID_W: u32,
    DATA_W_DIV8: u32 = {DATA_W / u32:8},
    DATA_W_LOG2: u32 = {std::clog2(DATA_W + u32:1)},
> {
    type AxiStream = axi_st::AxiStream<DATA_W, DEST_W, ID_W, DATA_W_DIV8>;
    type StrobedStream = ContinuousStream<DATA_W, DEST_W, ID_W>;

    config (
        stream_in_r: chan<AxiStream> in,
        stream_out_s: chan<AxiStream> out,
    ) {
        let (continuous_stream_s, continuous_stream_r) = chan<StrobedStream, u32:0>("continuous_stream");

        spawn RemoveEmptyBytes<DATA_W, DEST_W, ID_W>(
            stream_in_r,
            continuous_stream_s
        );
        spawn AxiStreamRemoveEmptyInternal<DATA_W, DEST_W, ID_W, DATA_W_DIV8, DATA_W_LOG2> (
            continuous_stream_r,
            stream_out_s
        );

        ()
    }

    init { () }

    next (state: ()) {}
}

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

        // Test 6: Some bits set, last set in the last transfer.

        let tok = send(tok, stream_in_s, TestAxiStream {
            data: Data:0x0000_00B9,
            str: Str:0b0001,
            keep: Keep:0b0001,
            last: u1:0,
            id: Id:0,
            dest: Dest:0,
        });
        let tok = send(tok, stream_in_s, TestAxiStream {
            data: Data:0x0000_007F,
            str: Str:0b0001,
            keep: Keep:0b0001,
            last: u1:0,
            id: Id:0,
            dest: Dest:0,
        });
        let tok = send(tok, stream_in_s, TestAxiStream {
            data: Data:0x0000_0069,
            str: Str:0b0001,
            keep: Keep:0b0001,
            last: u1:0,
            id: Id:0,
            dest: Dest:0,
        });
        let tok = send(tok, stream_in_s, TestAxiStream {
            data: Data:0x00DF_5EF7,
            str: Str:0b0111,
            keep: Keep:0b0111,
            last: u1:0,
            id: Id:0,
            dest: Dest:0,
        });
        let tok = send(tok, stream_in_s, TestAxiStream {
            data: Data:0x0000_C735,
            str: Str:0b0011,
            keep: Keep:0b0011,
            last: u1:1,
            id: Id:0,
            dest: Dest:0,
        });

        let (tok, stream_out) = recv(tok, stream_out_r);
        assert_eq(stream_out, TestAxiStream {
            data: Data:0xF769_7FB9,
            str: Str:0xF,
            keep: Keep:0xF,
            last: u1:0,
            id: Id:0,
            dest: Dest:0,
        });
        let (tok, stream_out) = recv(tok, stream_out_r);
        assert_eq(stream_out, TestAxiStream {
            data: Data:0xC735_DF5E,
            str: Str:0xF,
            keep: Keep:0xF,
            last: u1:1,
            id: Id:0,
            dest: Dest:0,
        });

        // Test 7: Some bits set, last set in the last transfer.


        let tok = send(tok, stream_in_s, TestAxiStream {
            data: Data:0xf7697fb9,
            str: Str:0b1111,
            keep: Keep:0b1111,
            last: u1:0,
            id: Id:0,
            dest: Dest:0,
        });
        let tok = send(tok, stream_in_s, TestAxiStream {
            data: Data:0xc735df5e,
            str: Str:0b1111,
            keep: Keep:0b1111,
            last: u1:1,
            id: Id:0,
            dest: Dest:0,
        });
        let tok = send(tok, stream_in_s, TestAxiStream {
            data: Data:0x70d3da1f,
            str: Str:0b1111,
            keep: Keep:0b1111,
            last: u1:0,
            id: Id:0,
            dest: Dest:0,
        });
        let tok = send(tok, stream_in_s, TestAxiStream {
            data: Data:0x0000001d,
            str: Str:0b0001,
            keep: Keep:0b0001,
            last: u1:1,
            id: Id:0,
            dest: Dest:0,
        });
        let tok = send(tok, stream_in_s, TestAxiStream {
            data: Data:0x01eaf614,
            str: Str:0b1111,
            keep: Keep:0b1111,
            last: u1:0,
            id: Id:0,
            dest: Dest:0,
        });
        let tok = send(tok, stream_in_s, TestAxiStream {
            data: Data:0x00001734,
            str: Str:0b0011,
            keep: Keep:0b0011,
            last: u1:1,
            id: Id:0,
            dest: Dest:0,
        });
        let tok = send(tok, stream_in_s, TestAxiStream {
            data: Data:0xe935b870,
            str: Str:0b1111,
            keep: Keep:0b1111,
            last: u1:0,
            id: Id:0,
            dest: Dest:0,
        });
        let tok = send(tok, stream_in_s, TestAxiStream {
            data: Data:0x00f149f5,
            str: Str:0b0111,
            keep: Keep:0b0111,
            last: u1:1,
            id: Id:0,
            dest: Dest:0,
        });
        let tok = send(tok, stream_in_s, TestAxiStream {
            data: Data:0xf073eed1,
            str: Str:0b1111,
            keep: Keep:0b1111,
            last: u1:0,
            id: Id:0,
            dest: Dest:0,
        });
        let tok = send(tok, stream_in_s, TestAxiStream {
            data: Data:0xce97b5bd,
            str: Str:0b1111,
            keep: Keep:0b1111,
            last: u1:1,
            id: Id:0,
            dest: Dest:0,
        });
        let tok = send(tok, stream_in_s, TestAxiStream {
            data: Data:0x950cddd9,
            str: Str:0b1111,
            keep: Keep:0b1111,
            last: u1:0,
            id: Id:0,
            dest: Dest:0,
        });
        let tok = send(tok, stream_in_s, TestAxiStream {
            data: Data:0x08f0ebd4,
            str: Str:0b1111,
            keep: Keep:0b1111,
            last: u1:1,
            id: Id:0,
            dest: Dest:0,
        });
        let tok = send(tok, stream_in_s, TestAxiStream {
            data: Data:0xABEB9592,
            str: Str:0b1111,
            keep: Keep:0b1111,
            last: u1:0,
            id: Id:0,
            dest: Dest:0,
        });
        let tok = send(tok, stream_in_s, TestAxiStream {
            data: Data:0xB16E2D5C,
            str: Str:0b1111,
            keep: Keep:0b1111,
            last: u1:1,
            id: Id:0,
            dest: Dest:0,
        });
        let tok = send(tok, stream_in_s, TestAxiStream {
            data: Data:0x157CF9C6,
            str: Str:0b1111,
            keep: Keep:0b1111,
            last: u1:0,
            id: Id:0,
            dest: Dest:0,
        });
        let tok = send(tok, stream_in_s, TestAxiStream {
            data: Data:0x00000019,
            str: Str:0b0001,
            keep: Keep:0b0001,
            last: u1:1,
            id: Id:0,
            dest: Dest:0,
        });
        let (tok, stream_out) = recv(tok, stream_out_r);
        assert_eq(stream_out, TestAxiStream {
            data: Data:0xf7697fb9,
            str: Str:0b1111,
            keep: Keep:0b1111,
            last: u1:0,
            id: Id:0,
            dest: Dest:0,
        });
        let (tok, stream_out) = recv(tok, stream_out_r);
        assert_eq(stream_out, TestAxiStream {
            data: Data:0xc735df5e,
            str: Str:0b1111,
            keep: Keep:0b1111,
            last: u1:1,
            id: Id:0,
            dest: Dest:0,
        });
        let (tok, stream_out) = recv(tok, stream_out_r);
        assert_eq(stream_out, TestAxiStream {
            data: Data:0x70d3da1f,
            str: Str:0b1111,
            keep: Keep:0b1111,
            last: u1:0,
            id: Id:0,
            dest: Dest:0,
        });
        let (tok, stream_out) = recv(tok, stream_out_r);
        assert_eq(stream_out, TestAxiStream {
            data: Data:0x0000001d,
            str: Str:0b0001,
            keep: Keep:0b0001,
            last: u1:1,
            id: Id:0,
            dest: Dest:0,
        });
        let (tok, stream_out) = recv(tok, stream_out_r);
        assert_eq(stream_out, TestAxiStream {
            data: Data:0x01eaf614,
            str: Str:0b1111,
            keep: Keep:0b1111,
            last: u1:0,
            id: Id:0,
            dest: Dest:0,
        });
        let (tok, stream_out) = recv(tok, stream_out_r);
        assert_eq(stream_out, TestAxiStream {
            data: Data:0x00001734,
            str: Str:0b0011,
            keep: Keep:0b0011,
            last: u1:1,
            id: Id:0,
            dest: Dest:0,
        });
        let (tok, stream_out) = recv(tok, stream_out_r);
        assert_eq(stream_out, TestAxiStream {
            data: Data:0xe935b870,
            str: Str:0b1111,
            keep: Keep:0b1111,
            last: u1:0,
            id: Id:0,
            dest: Dest:0,
        });
        let (tok, stream_out) = recv(tok, stream_out_r);
        assert_eq(stream_out, TestAxiStream {
            data: Data:0x00f149f5,
            str: Str:0b0111,
            keep: Keep:0b0111,
            last: u1:1,
            id: Id:0,
            dest: Dest:0,
        });
        let (tok, stream_out) = recv(tok, stream_out_r);
        assert_eq(stream_out, TestAxiStream {
            data: Data:0xf073eed1,
            str: Str:0b1111,
            keep: Keep:0b1111,
            last: u1:0,
            id: Id:0,
            dest: Dest:0,
        });
        let (tok, stream_out) = recv(tok, stream_out_r);
        assert_eq(stream_out, TestAxiStream {
            data: Data:0xce97b5bd,
            str: Str:0b1111,
            keep: Keep:0b1111,
            last: u1:1,
            id: Id:0,
            dest: Dest:0,
        });
        let (tok, stream_out) = recv(tok, stream_out_r);
        assert_eq(stream_out, TestAxiStream {
            data: Data:0x950cddd9,
            str: Str:0b1111,
            keep: Keep:0b1111,
            last: u1:0,
            id: Id:0,
            dest: Dest:0,
        });
        let (tok, stream_out) = recv(tok, stream_out_r);
        assert_eq(stream_out, TestAxiStream {
            data: Data:0x08f0ebd4,
            str: Str:0b1111,
            keep: Keep:0b1111,
            last: u1:1,
            id: Id:0,
            dest: Dest:0,
        });
        let (tok, stream_out) = recv(tok, stream_out_r);
        assert_eq(stream_out, TestAxiStream {
            data: Data:0xABEB9592,
            str: Str:0b1111,
            keep: Keep:0b1111,
            last: u1:0,
            id: Id:0,
            dest: Dest:0,
        });
        let (tok, stream_out) = recv(tok, stream_out_r);
        assert_eq(stream_out, TestAxiStream {
            data: Data:0xB16E2D5C,
            str: Str:0b1111,
            keep: Keep:0b1111,
            last: u1:1,
            id: Id:0,
            dest: Dest:0,
        });
        let (tok, stream_out) = recv(tok, stream_out_r);
        assert_eq(stream_out, TestAxiStream {
            data: Data:0x157CF9C6,
            str: Str:0b1111,
            keep: Keep:0b1111,
            last: u1:0,
            id: Id:0,
            dest: Dest:0,
        });
        let (tok, stream_out) = recv(tok, stream_out_r);
        assert_eq(stream_out, TestAxiStream {
            data: Data:0x00000019,
            str: Str:0b0001,
            keep: Keep:0b0001,
            last: u1:1,
            id: Id:0,
            dest: Dest:0,
        });

        send(tok, terminator, true);
    }
}
