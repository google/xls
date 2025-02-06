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

// AXI Stream Add Empty
//
// This proc adds support for performing write requests under unaligned addresses.
// It receives write request and calculates the write address offset from 4-bytes alignment.
// The offset is used to shift and padd with zero bytes the data to write received on the AxiStream
// interface.
// Shifted data is passed down to the AxiWriter proc that handles the write requests.

import std;

import xls.modules.zstd.memory.common;
import xls.modules.zstd.memory.axi_st;
import xls.modules.zstd.memory.axi_writer;

enum AxiStreamAddEmptyFsm : u3 {
    RECV_REQUEST = 0,
    PASSTHROUGH = 1,
    INJECT_PADDING = 2,
    FORWARD_STREAM = 3,
    ERROR = 7,
}

struct AxiStreamAddEmptyState<
    DATA_W: u32,
    DEST_W: u32,
    ID_W: u32,
    DATA_W_LOG2: u32,
    DATA_W_DIV8: u32,
    ADDR_W: u32,
> {
    fsm: AxiStreamAddEmptyFsm,
    offset: uN[DATA_W_DIV8],
    shift: uN[DATA_W],
    adjusted_len: uN[ADDR_W],
    do_recv_raw_stream: bool,
    do_send_padded_stream: bool,
    frame_to_send: axi_st::AxiStream<DATA_W, DEST_W, ID_W, DATA_W_DIV8>,
    buffer_frame: axi_st::AxiStream<DATA_W, DEST_W, ID_W, DATA_W_DIV8>,
    buffer_offset: uN[DATA_W_DIV8],
    buffer_shift: uN[DATA_W],
    id: uN[ID_W],
    dest: uN[DEST_W],
}

pub proc AxiStreamAddEmpty<
    DATA_W: u32,
    DEST_W: u32,
    ID_W: u32,
    ADDR_W: u32,
    DATA_W_DIV8: u32 = {DATA_W / u32:8},
    DATA_W_LOG2: u32 = {std::clog2(DATA_W / u32:8)},
> {
    type Req = axi_writer::AxiWriterRequest<ADDR_W>;
    type AxiStream = axi_st::AxiStream<DATA_W, DEST_W, ID_W, DATA_W_DIV8>;
    type State = AxiStreamAddEmptyState<DATA_W, DEST_W, ID_W, DATA_W_LOG2, DATA_W_DIV8, ADDR_W>;
    type Fsm = AxiStreamAddEmptyFsm;

    type Addr = uN[ADDR_W];
    type Data = uN[DATA_W];
    type Strobe = uN[DATA_W_DIV8];

    write_req_r: chan<Req> in;
    raw_stream_r: chan<AxiStream> in;
    padded_stream_s: chan<AxiStream> out;

    config (
        write_req_r: chan<Req> in,
        raw_stream_r: chan<AxiStream> in,
        padded_stream_s: chan<AxiStream> out,
    ) {
        (
            write_req_r,
            raw_stream_r,
            padded_stream_s
        )
    }

    init { zero!<State>() }

    next (state: State) {
        let (tok, write_req) = recv_if(join(), write_req_r, state.fsm == Fsm::RECV_REQUEST, zero!<Req>());
        let (tok, raw_stream) = recv_if(join(), raw_stream_r, state.do_recv_raw_stream == true, AxiStream { last: true, ..zero!<AxiStream>()});
        let tok = send_if(join(), padded_stream_s, state.do_send_padded_stream, state.frame_to_send);

        let next_state = match(state.fsm) {
            Fsm::RECV_REQUEST => {
                let offset = common::offset<DATA_W_DIV8>(write_req.address) as Strobe;
                let goto_passthrough = offset == Strobe:0;
                let adjusted_len = write_req.length + offset as Addr;
                State {
                    fsm: if (goto_passthrough) { Fsm::PASSTHROUGH } else { Fsm::INJECT_PADDING },
                    offset: offset,
                    adjusted_len: adjusted_len,
                    do_recv_raw_stream: true,
                    ..state
                }
            },
            Fsm::PASSTHROUGH => {
                if (state.frame_to_send.last == true) {
                    State {
                        fsm: Fsm::RECV_REQUEST,
                        ..zero!<State>()
                    }
                } else {
                    State {
                        fsm: Fsm::PASSTHROUGH,
                        frame_to_send: raw_stream,
                        do_recv_raw_stream: !raw_stream.last,
                        do_send_padded_stream: true,
                        ..state
                    }
                }
            },
            Fsm::INJECT_PADDING => {
                let shift = (state.offset as Data * Data:8);
                let data_to_send = raw_stream.data << shift;
                let strb_to_send = raw_stream.str << state.offset;
                let keep_to_send = raw_stream.keep << state.offset;
                let buffer_offset = DATA_W_DIV8 as Strobe - state.offset;
                let buffer_shift = buffer_offset as Data * Data:8;
                let buffer_data = raw_stream.data >> buffer_shift;
                let buffer_strb = raw_stream.str >> buffer_offset;
                let buffer_keep = raw_stream.keep >> buffer_offset;
                let last = if (state.adjusted_len <= DATA_W_DIV8 as Addr) {raw_stream.last} else { false };
                State {
                    fsm: Fsm::FORWARD_STREAM,
                    shift: shift,
                    buffer_offset: buffer_offset,
                    buffer_shift: buffer_shift,
                    frame_to_send: AxiStream {
                        data: data_to_send,
                        str: strb_to_send,
                        keep: keep_to_send,
                        last: last,
                        ..raw_stream
                    },
                    buffer_frame: AxiStream {
                        data: buffer_data,
                        str: buffer_strb,
                        keep: buffer_keep,
                        ..raw_stream
                    },
                    do_recv_raw_stream: !raw_stream.last,
                    do_send_padded_stream: true,
                    id: raw_stream.id,
                    dest: raw_stream.dest,
                    ..state
                }
            },
            Fsm::FORWARD_STREAM => {
                if (state.frame_to_send.last == true) {
                    State {
                        fsm: Fsm::RECV_REQUEST,
                        ..zero!<State>()
                    }
                } else {
                    let data_to_send = (raw_stream.data << state.shift) | state.buffer_frame.data;
                    let strb_to_send = (raw_stream.str << state.offset) | state.buffer_frame.str;
                    let keep_to_send = (raw_stream.keep << state.offset) | state.buffer_frame.keep;
                    let buffer_data = raw_stream.data >> state.buffer_shift;
                    let buffer_strb = raw_stream.str >> state.buffer_offset;
                    let buffer_keep = raw_stream.keep >> state.buffer_offset;
                    // Current frame is last and there is no more data to send in the next frame
                    let finish_early = (buffer_strb == Strobe:0) && raw_stream.last;

                    State {
                        fsm: Fsm::FORWARD_STREAM,
                        frame_to_send: AxiStream {
                            data: data_to_send,
                            str: strb_to_send,
                            keep: keep_to_send,
                            last: if (finish_early) { true } else { state.buffer_frame.last },
                            id: state.id,
                            dest: state.dest,
                        },
                        buffer_frame: AxiStream {
                            data: buffer_data,
                            str: buffer_strb,
                            keep: buffer_keep,
                            id: state.id,
                            dest: state.dest,
                            ..raw_stream
                        },
                        do_recv_raw_stream: !raw_stream.last,
                        ..state
                    }
                }
            },
            Fsm::ERROR => {
                state
            },
        };

        next_state
    }
}

const INST_ADDR_W = u32:16;
const INST_DATA_W = u32:32;
const INST_DEST_W = u32:32;
const INST_ID_W = u32:32;
const INST_DATA_W_DIV8 = u32:4;
const INST_DATA_W_LOG2 = u32:6;

type InstAxiStream = axi_st::AxiStream<INST_DATA_W, INST_DEST_W, INST_ID_W, INST_DATA_W_DIV8>;
type InstReq = axi_writer::AxiWriterRequest<INST_ADDR_W>;

proc AxiStreamAddEmptyInst {
    config (
        write_req_r: chan<InstReq> in,
        raw_stream_r: chan<InstAxiStream> in,
        padded_stream_s: chan<InstAxiStream> out,
    ) {
        spawn AxiStreamAddEmpty<INST_DATA_W, INST_DEST_W, INST_ID_W, INST_ADDR_W> (
            write_req_r,
            raw_stream_r,
            padded_stream_s
        );
    }

    init { }

    next (state:()) { }
}

const TEST_ADDR_W = u32:16;
const TEST_DATA_W = u32:32;
const TEST_DEST_W = u32:32;
const TEST_ID_W = u32:32;
const TEST_DATA_W_DIV8 = u32:4;

type TestAxiStream = axi_st::AxiStream<TEST_DATA_W, TEST_DEST_W, TEST_ID_W, TEST_DATA_W_DIV8>;
type TestReq = axi_writer::AxiWriterRequest<TEST_ADDR_W>;

type TestAddr = uN[TEST_ADDR_W];
type TestLength = uN[TEST_ADDR_W];
type TestData = uN[TEST_DATA_W];
type TestStrobe = uN[TEST_DATA_W_DIV8];
type TestId = uN[TEST_ID_W];
type TestDest = uN[TEST_DEST_W];

const TEST_WRITE_REQUEST = TestReq[13]:[
    TestReq {
        address: TestAddr:0x0,
        length: TestLength:8
    },
    TestReq {
        address: TestAddr:0x1,
        length: TestLength:8
    },
    TestReq {
        address: TestAddr:0x2,
        length: TestLength:8
    },
    TestReq {
        address: TestAddr:0x3,
        length: TestLength:8
    },
    TestReq {
        address: TestAddr:0x4,
        length: TestLength:8
    },
    TestReq {
        address: TestAddr:0x4,
        length: TestLength:16
    },
    TestReq {
        address: TestAddr:0x5,
        length: TestLength:16
    },
    TestReq {
        address: TestAddr:0x6,
        length: TestLength:16
    },
    TestReq {
        address: TestAddr:0x7,
        length: TestLength:16
    },
    TestReq {
        address: TestAddr:0x8,
        length: TestLength:16
    },
    TestReq {
        address: TestAddr:0x3,
        length: TestLength:10
    },
    TestReq {
        address: TestAddr:0x1,
        length: TestLength:1
    },
    TestReq {
        address: TestAddr:0x1,
        length: TestLength:4
    },
];

const TEST_STREAM_IN = TestAxiStream[35]:[
    TestAxiStream {
        data: TestData:0xDEAD_BEEF,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: false,
        id: TestId:0,
        dest: TestDest:0,
    },
    TestAxiStream {
        data: TestData:0xCAFEBAAD,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: true,
        id: TestId:0,
        dest: TestDest:0,
    },

    TestAxiStream {
        data: TestData:0xDEAD_BEEF,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: false,
        id: TestId:1,
        dest: TestDest:1,
    },
    TestAxiStream {
        data: TestData:0xCAFEBAAD,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: true,
        id: TestId:1,
        dest: TestDest:1,
    },

    TestAxiStream {
        data: TestData:0xDEAD_BEEF,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: false,
        id: TestId:2,
        dest: TestDest:2,
    },
    TestAxiStream {
        data: TestData:0xCAFEBAAD,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: true,
        id: TestId:2,
        dest: TestDest:2,
    },

    TestAxiStream {
        data: TestData:0xDEAD_BEEF,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: false,
        id: TestId:3,
        dest: TestDest:3,
    },
    TestAxiStream {
        data: TestData:0xCAFEBAAD,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: true,
        id: TestId:3,
        dest: TestDest:3,
    },

    TestAxiStream {
        data: TestData:0xDEAD_BEEF,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: false,
        id: TestId:4,
        dest: TestDest:4,
    },
    TestAxiStream {
        data: TestData:0xCAFEBAAD,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: true,
        id: TestId:4,
        dest: TestDest:4,
    },

    TestAxiStream {
        data: TestData:0x8765_4321,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: false,
        id: TestId:4,
        dest: TestDest:4,
    },
    TestAxiStream {
        data: TestData:0xFFEDCBA9,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: false,
        id: TestId:4,
        dest: TestDest:4,
    },
    TestAxiStream {
        data: TestData:0x9ABCDEFF,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: false,
        id: TestId:4,
        dest: TestDest:4,
    },
    TestAxiStream {
        data: TestData:0x12345678,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: true,
        id: TestId:4,
        dest: TestDest:4,
    },

    TestAxiStream {
        data: TestData:0x8765_4321,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: false,
        id: TestId:5,
        dest: TestDest:5,
    },
    TestAxiStream {
        data: TestData:0xFFEDCBA9,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: false,
        id: TestId:5,
        dest: TestDest:5,
    },
    TestAxiStream {
        data: TestData:0x9ABCDEFF,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: false,
        id: TestId:5,
        dest: TestDest:5,
    },
    TestAxiStream {
        data: TestData:0x12345678,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: true,
        id: TestId:5,
        dest: TestDest:5,
    },

    TestAxiStream {
        data: TestData:0x8765_4321,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: false,
        id: TestId:6,
        dest: TestDest:6,
    },
    TestAxiStream {
        data: TestData:0xFFEDCBA9,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: false,
        id: TestId:6,
        dest: TestDest:6,
    },
    TestAxiStream {
        data: TestData:0x9ABCDEFF,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: false,
        id: TestId:6,
        dest: TestDest:6,
    },
    TestAxiStream {
        data: TestData:0x12345678,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: true,
        id: TestId:6,
        dest: TestDest:6,
    },

    TestAxiStream {
        data: TestData:0x8765_4321,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: false,
        id: TestId:7,
        dest: TestDest:7,
    },
    TestAxiStream {
        data: TestData:0xFFEDCBA9,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: false,
        id: TestId:7,
        dest: TestDest:7,
    },
    TestAxiStream {
        data: TestData:0x9ABCDEFF,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: false,
        id: TestId:7,
        dest: TestDest:7,
    },
    TestAxiStream {
        data: TestData:0x12345678,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: true,
        id: TestId:7,
        dest: TestDest:7,
    },

    TestAxiStream {
        data: TestData:0x8765_4321,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: false,
        id: TestId:8,
        dest: TestDest:8,
    },
    TestAxiStream {
        data: TestData:0xFFEDCBA9,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: false,
        id: TestId:8,
        dest: TestDest:8,
    },
    TestAxiStream {
        data: TestData:0x9ABCDEFF,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: false,
        id: TestId:8,
        dest: TestDest:8,
    },
    TestAxiStream {
        data: TestData:0x12345678,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: true,
        id: TestId:8,
        dest: TestDest:8,
    },

    TestAxiStream {
        data: TestData:0x1734_6B45,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: false,
        id: TestId:9,
        dest: TestDest:9,
    },
    TestAxiStream {
        data: TestData:0x0476_2A22,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: false,
        id: TestId:9,
        dest: TestDest:0,
    },
    TestAxiStream {
        data: TestData:0xE304,
        str: TestStrobe:0b0011,
        keep: TestStrobe:0b0011,
        last: true,
        id: TestId:9,
        dest: TestDest:9,
    },
    TestAxiStream {
        data: TestData:0xF1,
        str: TestStrobe:0b0001,
        keep: TestStrobe:0b0001,
        last: true,
        id: TestId:10,
        dest: TestDest:10,
    },
    TestAxiStream {
        data: TestData:0x01EAF614,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: true,
        id: TestId:11,
        dest: TestDest:11,
    },
];

const TEST_STREAM_OUT = TestAxiStream[43]:[
    TestAxiStream {
        data: TestData:0xDEAD_BEEF,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: false,
        id: TestId:0,
        dest: TestDest:0,
    },
    TestAxiStream {
        data: TestData:0xCAFEBAAD,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: true,
        id: TestId:0,
        dest: TestDest:0,
    },

    TestAxiStream {
        data: TestData:0xADBE_EF00,
        str: TestStrobe:0b1110,
        keep: TestStrobe:0b1110,
        last: false,
        id: TestId:1,
        dest: TestDest:1,
    },
    TestAxiStream {
        data: TestData:0xFEBA_ADDE,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: false,
        id: TestId:1,
        dest: TestDest:1,
    },
    TestAxiStream {
        data: TestData:0x0000_00CA,
        str: TestStrobe:0b0001,
        keep: TestStrobe:0b0001,
        last: true,
        id: TestId:1,
        dest: TestDest:1,
    },

    TestAxiStream {
        data: TestData:0xBEEF_0000,
        str: TestStrobe:0b1100,
        keep: TestStrobe:0b1100,
        last: false,
        id: TestId:2,
        dest: TestDest:2,
    },
    TestAxiStream {
        data: TestData:0xBAAD_DEAD,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: false,
        id: TestId:2,
        dest: TestDest:2,
    },
    TestAxiStream {
        data: TestData:0x0000_CAFE,
        str: TestStrobe:0b0011,
        keep: TestStrobe:0b0011,
        last: true,
        id: TestId:2,
        dest: TestDest:2,
    },

    TestAxiStream {
        data: TestData:0xEF00_0000,
        str: TestStrobe:0b1000,
        keep: TestStrobe:0b1000,
        last: false,
        id: TestId:3,
        dest: TestDest:3,
    },
    TestAxiStream {
        data: TestData:0xADDE_ADBE,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: false,
        id: TestId:3,
        dest: TestDest:3,
    },
    TestAxiStream {
        data: TestData:0x00CA_FEBA,
        str: TestStrobe:0b0111,
        keep: TestStrobe:0b0111,
        last: true,
        id: TestId:3,
        dest: TestDest:3,
    },

    TestAxiStream {
        data: TestData:0xDEAD_BEEF,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: false,
        id: TestId:4,
        dest: TestDest:4,
    },
    TestAxiStream {
        data: TestData:0xCAFEBAAD,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: true,
        id: TestId:4,
        dest: TestDest:4,
    },

    TestAxiStream {
        data: TestData:0x8765_4321,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: false,
        id: TestId:4,
        dest: TestDest:4,
    },
    TestAxiStream {
        data: TestData:0xFFEDCBA9,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: false,
        id: TestId:4,
        dest: TestDest:4,
    },
    TestAxiStream {
        data: TestData:0x9ABCDEFF,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: false,
        id: TestId:4,
        dest: TestDest:4,
    },
    TestAxiStream {
        data: TestData:0x12345678,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: true,
        id: TestId:4,
        dest: TestDest:4,
    },

    TestAxiStream {
        data: TestData:0x6543_2100,
        str: TestStrobe:0b1110,
        keep: TestStrobe:0b1110,
        last: false,
        id: TestId:5,
        dest: TestDest:5,
    },
    TestAxiStream {
        data: TestData:0xEDCBA987,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: false,
        id: TestId:5,
        dest: TestDest:5,
    },
    TestAxiStream {
        data: TestData:0xBCDEFFFF,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: false,
        id: TestId:5,
        dest: TestDest:5,
    },
    TestAxiStream {
        data: TestData:0x3456789A,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: false,
        id: TestId:5,
        dest: TestDest:5,
    },
    TestAxiStream {
        data: TestData:0x12,
        str: TestStrobe:0b0001,
        keep: TestStrobe:0b0001,
        last: true,
        id: TestId:5,
        dest: TestDest:5,
    },

    TestAxiStream {
        data: TestData:0x4321_0000,
        str: TestStrobe:0b1100,
        keep: TestStrobe:0b1100,
        last: false,
        id: TestId:6,
        dest: TestDest:6,
    },
    TestAxiStream {
        data: TestData:0xCBA98765,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: false,
        id: TestId:6,
        dest: TestDest:6,
    },
    TestAxiStream {
        data: TestData:0xDEFFFFED,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: false,
        id: TestId:6,
        dest: TestDest:6,
    },
    TestAxiStream {
        data: TestData:0x56789ABC,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: false,
        id: TestId:6,
        dest: TestDest:6,
    },
    TestAxiStream {
        data: TestData:0x1234,
        str: TestStrobe:0b0011,
        keep: TestStrobe:0b0011,
        last: true,
        id: TestId:6,
        dest: TestDest:6,
    },

    TestAxiStream {
        data: TestData:0x2100_0000,
        str: TestStrobe:0b1000,
        keep: TestStrobe:0b1000,
        last: false,
        id: TestId:7,
        dest: TestDest:7,
    },
    TestAxiStream {
        data: TestData:0xA9876543,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: false,
        id: TestId:7,
        dest: TestDest:7,
    },
    TestAxiStream {
        data: TestData:0xFFFFEDCB,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: false,
        id: TestId:7,
        dest: TestDest:7,
    },
    TestAxiStream {
        data: TestData:0x789ABCDE,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: false,
        id: TestId:7,
        dest: TestDest:7,
    },
    TestAxiStream {
        data: TestData:0x1234_56,
        str: TestStrobe:0b0111,
        keep: TestStrobe:0b0111,
        last: true,
        id: TestId:7,
        dest: TestDest:7,
    },

    TestAxiStream {
        data: TestData:0x8765_4321,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: false,
        id: TestId:8,
        dest: TestDest:8,
    },
    TestAxiStream {
        data: TestData:0xFFEDCBA9,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: false,
        id: TestId:8,
        dest: TestDest:8,
    },
    TestAxiStream {
        data: TestData:0x9ABCDEFF,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: false,
        id: TestId:8,
        dest: TestDest:8,
    },
    TestAxiStream {
        data: TestData:0x12345678,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: true,
        id: TestId:8,
        dest: TestDest:8,
    },

    TestAxiStream {
        data: TestData:0x45000000,
        str: TestStrobe:0b1000,
        keep: TestStrobe:0b1000,
        last: false,
        id: TestId:9,
        dest: TestDest:9,
    },
    TestAxiStream {
        data: TestData:0x2217346B,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: false,
        id: TestId:9,
        dest: TestDest:9,
    },
    TestAxiStream {
        data: TestData:0x0404762A,
        str: TestStrobe:0b1111,
        keep: TestStrobe:0b1111,
        last: false,
        id: TestId:9,
        dest: TestDest:9,
    },
    TestAxiStream {
        data: TestData:0x000000E3,
        str: TestStrobe:0b0001,
        keep: TestStrobe:0b0001,
        last: true,
        id: TestId:9,
        dest: TestDest:9,
    },
    TestAxiStream {
        data: TestData:0x0000F100,
        str: TestStrobe:0b0010,
        keep: TestStrobe:0b0010,
        last: true,
        id: TestId:10,
        dest: TestDest:10,
    },
    TestAxiStream {
        data: TestData:0xEAF61400,
        str: TestStrobe:0b1110,
        keep: TestStrobe:0b1110,
        last: false,
        id: TestId:11,
        dest: TestDest:11,
    },
    TestAxiStream {
        data: TestData:0x01,
        str: TestStrobe:0b0001,
        keep: TestStrobe:0b0001,
        last: true,
        id: TestId:11,
        dest: TestDest:11,
    },
];

#[test_proc]
proc AxiStreamAddEmptyTest {
    terminator: chan<bool> out;

    write_req_s: chan<TestReq> out;
    raw_stream_s: chan<TestAxiStream> out;
    padded_stream_r: chan<TestAxiStream> in;

    config (
        terminator: chan<bool> out,
    ) {
        let (write_req_s, write_req_r) = chan<TestReq>("write_req");
        let (raw_stream_s, raw_stream_r) = chan<TestAxiStream>("raw_stream");
        let (padded_stream_s, padded_stream_r) = chan<TestAxiStream>("stream_out");

        spawn AxiStreamAddEmpty<TEST_DATA_W, TEST_DEST_W, TEST_ID_W, TEST_ADDR_W> (
            write_req_r,
            raw_stream_r,
            padded_stream_s
        );

        (
            terminator,
            write_req_s,
            raw_stream_s,
            padded_stream_r,
        )
    }

    init { }

    next (state: ()) {
        let tok = for ((i, test_write_req), tok): ((u32, TestReq), token) in enumerate(TEST_WRITE_REQUEST) {
            let tok = send(tok, write_req_s, test_write_req);
            trace_fmt!("Sent #{} write request {:#x}", i + u32:1, test_write_req);
            tok
        }(join());

        let tok = for ((i, test_raw_stream), tok): ((u32, TestAxiStream), token) in enumerate(TEST_STREAM_IN) {
            let tok = send(tok, raw_stream_s, test_raw_stream);
            trace_fmt!("Sent #{} stream input {:#x}", i + u32:1, test_raw_stream);
            tok
        }(tok);

        let tok = for ((i, test_stream_out), tok): ((u32, TestAxiStream), token) in enumerate(TEST_STREAM_OUT) {
            let (tok, stream_out) = recv(tok, padded_stream_r);
            trace_fmt!("Received #{} stream output {:#x}", i + u32:1, stream_out);
            assert_eq(test_stream_out, stream_out);
            tok
        }(tok);

        send(tok, terminator, true);
    }
}
