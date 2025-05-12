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

// Memory Writer
//
// This module implements the communication with memory through the AXI interface
// that facilitates the write operations to the memory.
//
// Memory Writer proc is configured with a base address used for calculating
// addresses for pending write requests.
// Write Requests consist of the offset from the base address and the length
// of the data to write in bytes.
// Data to write is received through generic DSLX data packets which are then
// formed into AxiStream frames and passed along with the write request to
// the underlying procs.
// The first proc in the data path is the AxiStreamAddEmpty that prepares
// data to write for the next proc which is the AxiWriter.
// Axi writer takes the write request from MemWriter and data stream from AxiStreamAddEmpty
// and forms valid AXI write transactions.

import std;

import xls.modules.zstd.memory.axi;
import xls.modules.zstd.memory.axi_st;
import xls.modules.zstd.memory.common;
import xls.modules.zstd.memory.axi_writer;
import xls.modules.zstd.memory.axi_stream_remove_empty;
import xls.modules.zstd.memory.axi_stream_add_empty;

pub struct MemWriterReq<ADDR_W: u32> {
    addr: uN[ADDR_W],
    length: uN[ADDR_W],
}

pub struct MemWriterDataPacket<DATA_W: u32, ADDR_W: u32> {
    data: uN[DATA_W],
    length: uN[ADDR_W], // Expressed in bytes
    last: bool,
}

enum MemWriterFsm : u2 {
    RECV_REQ = 0,
    SEND_WRITE_REQ = 1,
    RECV_DATA = 2,
    SEND_DATA = 3,
}

struct MemWriterState<
    ADDR_W: u32,
    DATA_W: u32,
    DEST_W: u32,
    ID_W: u32,
    DATA_W_DIV8: u32
> {
    fsm: MemWriterFsm,
    req_len: sN[ADDR_W],
    axi_writer_req: axi_writer::AxiWriterRequest<ADDR_W>,
}

proc MemWriter<
    ADDR_W: u32, DATA_W: u32, DEST_W: u32, ID_W: u32, WRITER_ID: u32,
    DATA_W_DIV8: u32 = {DATA_W / u32:8},
    DATA_W_LOG2: u32 = {std::clog2(DATA_W / u32:8)}
> {
    type Req = MemWriterReq<ADDR_W>;
    type Data = MemWriterDataPacket<DATA_W, ADDR_W>;
    type AxiWriterReq = axi_writer::AxiWriterRequest<ADDR_W>;
    type AxiWriterResp = axi_writer::AxiWriterResp;
    type PaddingReq = axi_writer::AxiWriterRequest<ADDR_W>;
    type AxiStream = axi_st::AxiStream<DATA_W, DEST_W, ID_W, DATA_W_DIV8>;
    type AxiAW = axi::AxiAw<ADDR_W, ID_W>;
    type AxiW = axi::AxiW<DATA_W, DATA_W_DIV8>;
    type AxiB = axi::AxiB<ID_W>;
    type State = MemWriterState<ADDR_W, DATA_W, DATA_W_DIV8, DEST_W, ID_W>;
    type Fsm = MemWriterFsm;

    type Length = uN[ADDR_W];
    type sLength = sN[ADDR_W];
    type Strobe = uN[DATA_W_DIV8];
    type Id = uN[ID_W];
    type Dest = uN[DEST_W];

    req_in_r: chan<Req> in;
    data_in_r: chan<Data> in;
    axi_writer_req_s: chan<AxiWriterReq> out;
    padding_req_s: chan<PaddingReq> out;
    axi_st_raw_s: chan<AxiStream> out;
    resp_s: chan<AxiWriterResp> out;

    config(
        req_in_r: chan<Req> in,
        data_in_r: chan<Data> in,
        axi_aw_s: chan<AxiAW> out,
        axi_w_s: chan<AxiW> out,
        axi_b_r: chan<AxiB> in,
        resp_s: chan<AxiWriterResp> out,
    ) {
        let (axi_writer_req_s, axi_writer_req_r) = chan<AxiWriterReq, u32:0>("axi_writer_req");
        let (padding_req_s, padding_req_r) = chan<PaddingReq, u32:0>("padding_req");
        let (axi_st_raw_s, axi_st_raw_r) = chan<AxiStream, u32:0>("axi_st_raw");
        let (axi_st_clean_s, axi_st_clean_r) = chan<AxiStream, u32:0>("axi_st_clean");
        let (axi_st_padded_s, axi_st_padded_r) = chan<AxiStream, u32:0>("axi_st_padded");

        spawn axi_stream_remove_empty::AxiStreamRemoveEmpty<
            DATA_W, DEST_W, ID_W
        >(axi_st_raw_r, axi_st_clean_s);
        spawn axi_stream_add_empty::AxiStreamAddEmpty<
            DATA_W, DEST_W, ID_W, ADDR_W
        >(padding_req_r, axi_st_clean_r, axi_st_padded_s);
        spawn axi_writer::AxiWriter<
            ADDR_W, DATA_W, DEST_W, ID_W
        >(axi_writer_req_r, resp_s, axi_aw_s, axi_w_s, axi_b_r, axi_st_padded_r);

        (req_in_r, data_in_r, axi_writer_req_s, padding_req_s, axi_st_raw_s, resp_s)
    }

    init { zero!<State>() }

    next(state: State) {
        let tok_0 = join();
        let (tok_2, req_in) = recv_if(tok_0, req_in_r, state.fsm == Fsm::RECV_REQ, zero!<Req>());
        let tok_3 = send_if(tok_0, axi_writer_req_s, state.fsm == Fsm::SEND_WRITE_REQ, state.axi_writer_req);
        let tok_4 = send_if(tok_3, padding_req_s, state.fsm == Fsm::SEND_WRITE_REQ, state.axi_writer_req);
        let (tok_5, data_in) = recv_if(tok_0, data_in_r, state.fsm == Fsm::SEND_DATA, zero!<Data>());

        let next_state = match(state.fsm) {
            Fsm::RECV_REQ => {
                State {
                    fsm: Fsm::SEND_WRITE_REQ,
                    req_len: req_in.length as sLength,
                    axi_writer_req: AxiWriterReq {
                        address: req_in.addr,
                        length: req_in.length
                    },
                }
            },
            Fsm::SEND_WRITE_REQ => {
                State {
                    fsm: Fsm::SEND_DATA,
                    ..state
                }
            },
            Fsm::SEND_DATA => {
                let next_req_len = state.req_len - data_in.length as sLength;
                State {
                    fsm: if (next_req_len <= sLength:0) {Fsm::RECV_REQ} else {Fsm::SEND_DATA},
                    req_len: next_req_len,
                    ..state
                }
            },
            _ => {
                assert!(false, "Invalid state");
                state
            }
        };

        let raw_axi_st_frame = match(state.fsm) {
            Fsm::SEND_DATA => {
                let next_req_len = next_state.req_len;
                let str_keep = ((Length:1 << data_in.length) - Length:1) as Strobe;
                AxiStream {
                    data: data_in.data,
                    str: str_keep,
                    keep: str_keep,
                    last: (next_req_len <= sLength:0),
                    id: WRITER_ID as Id,
                    dest: WRITER_ID as Dest
                }
            },
            _ => {
                zero!<AxiStream>()
            }
        };

        let tok_6 = send_if(tok_5, axi_st_raw_s, state.fsm == Fsm::SEND_DATA, raw_axi_st_frame);

        next_state
    }
}

const INST_ADDR_W = u32:16;
const INST_DATA_W = u32:32;
const INST_DATA_W_DIV8 = INST_DATA_W / u32:8;
const INST_DEST_W = INST_DATA_W / u32:8;
const INST_ID_W = INST_DATA_W / u32:8;
const INST_DATA_W_LOG2 = u32:6;
const INST_WRITER_ID = u32:2;

proc MemWriterInst {
    type InstReq = MemWriterReq<INST_ADDR_W>;
    type InstData = MemWriterDataPacket<INST_DATA_W, INST_ADDR_W>;
    type InstAxiStream = axi_st::AxiStream<INST_DATA_W, INST_DEST_W, INST_ID_W, INST_DATA_W_DIV8>;
    type InstAxiAW = axi::AxiAw<INST_ADDR_W, INST_ID_W>;
    type InstAxiW = axi::AxiW<INST_DATA_W, INST_DATA_W_DIV8>;
    type InstAxiB = axi::AxiB<INST_ID_W>;
    type InstAxiWriterResp = axi_writer::AxiWriterResp;

    config(
        req_in_r: chan<InstReq> in,
        data_in_r: chan<InstData> in,
        axi_aw_s: chan<InstAxiAW> out,
        axi_w_s: chan<InstAxiW> out,
        axi_b_r: chan<InstAxiB> in,
        resp_s: chan<InstAxiWriterResp> out
    ) {
        spawn MemWriter<
            INST_ADDR_W, INST_DATA_W, INST_DEST_W, INST_ID_W, INST_WRITER_ID
        >(req_in_r, data_in_r, axi_aw_s, axi_w_s, axi_b_r, resp_s);
        ()
    }

    init { () }

    next(state: ()) {  }
}

const TEST_ADDR_W = u32:16;
const TEST_DATA_W = u32:32;
const TEST_DATA_W_DIV8 = TEST_DATA_W / u32:8;
const TEST_DEST_W = TEST_DATA_W / u32:8;
const TEST_ID_W = TEST_DATA_W / u32:8;
const TEST_DATA_W_LOG2 = u32:6;
const TEST_WRITER_ID = u32:2;

type TestReq = MemWriterReq<INST_ADDR_W>;
type TestData = MemWriterDataPacket<INST_DATA_W, INST_ADDR_W>;
type TestAxiWriterResp = axi_writer::AxiWriterResp;
type TestAxiWriterRespStatus = axi_writer::AxiWriterRespStatus;
type TestAxiStream = axi_st::AxiStream<INST_DATA_W, INST_DEST_W, INST_ID_W, INST_DATA_W_DIV8>;
type TestAxiAW = axi::AxiAw<INST_ADDR_W, INST_ID_W>;
type TestAxiW = axi::AxiW<INST_DATA_W, INST_DATA_W_DIV8>;
type TestAxiB = axi::AxiB<INST_ID_W>;
type TestAxiWriteResp = axi::AxiWriteResp;
type TestAxiAxBurst = axi::AxiAxBurst;
type TestAxiAxSize = axi::AxiAxSize;

type TestAddr = uN[TEST_ADDR_W];
type TestLength = uN[TEST_ADDR_W];
type TestDataBits = uN[TEST_DATA_W];
type TestStrobe = uN[TEST_DATA_W_DIV8];
type TestId = uN[TEST_ID_W];
type TestDest = uN[TEST_DEST_W];

#[test_proc]
proc MemWriterTest {
    terminator: chan<bool> out;
    req_in_s: chan<TestReq> out;
    data_in_s: chan<TestData> out;
    axi_aw_r: chan<TestAxiAW> in;
    axi_w_r: chan<TestAxiW> in;
    axi_b_s: chan<TestAxiB> out;
    resp_r: chan<TestAxiWriterResp> in;

    config(
        terminator: chan<bool> out,
    ) {
        let (req_in_s, req_in_r) = chan<TestReq>("req_in");
        let (data_in_s, data_in_r) = chan<TestData>("data_in");
        let (axi_aw_s, axi_aw_r) = chan<TestAxiAW>("axi_aw");
        let (axi_w_s, axi_w_r) = chan<TestAxiW>("axi_w");
        let (axi_b_s, axi_b_r) = chan<TestAxiB>("axi_b");
        let (resp_s, resp_r) = chan<TestAxiWriterResp>("resp");
        spawn MemWriter<
            TEST_ADDR_W, TEST_DATA_W, TEST_DEST_W, TEST_ID_W, TEST_WRITER_ID
        >(req_in_r, data_in_r, axi_aw_s, axi_w_s, axi_b_r, resp_s);
        (terminator, req_in_s, data_in_s, axi_aw_r, axi_w_r, axi_b_s, resp_r)
    }

    init { () }

    next(state: ()) {
        let tok = join();

        // Aligned single transfer
        let tok = send(tok, req_in_s, TestReq {
            addr: TestAddr:0x0,
            length: TestLength:4
        });
        let tok = send(tok, data_in_s, TestData {
            data: TestDataBits:0x44332211,
            length: TestLength:4,
            last: true,
        });
        let (tok, aw) = recv(tok, axi_aw_r);
        assert_eq(aw, TestAxiAW {
            id: TestId:1,
            addr: TestAddr:0,
            size: TestAxiAxSize::MAX_4B_TRANSFER,
            len: u8:0,
            burst: TestAxiAxBurst::INCR,
        });
        let (tok, w) = recv(tok, axi_w_r);
        assert_eq(w, TestAxiW {
            data: TestDataBits:0x44332211,
            strb: TestStrobe:0xF,
            last: true,
        });
        let tok = send(tok, axi_b_s, TestAxiB {
            resp: TestAxiWriteResp::OKAY,
            id: TestId:1,
        });
        let (tok, resp) = recv(tok, resp_r);
        assert_eq(resp, TestAxiWriterResp{status: TestAxiWriterRespStatus::OKAY});

        // Unaligned single transfer
        let tok = send(tok, req_in_s, TestReq {
            addr: TestAddr:0x10,
            length: TestLength:1
        });
        let tok = send(tok, data_in_s, TestData {
            data: TestDataBits:0x00000055,
            length: TestLength:1,
            last: true,
        });
        let (tok, aw) = recv(tok, axi_aw_r);
        assert_eq(aw, TestAxiAW {
            id: TestId:2,
            addr: TestAddr:0x10,
            size: TestAxiAxSize::MAX_4B_TRANSFER,
            len: u8:0,
            burst: TestAxiAxBurst::INCR,
        });
        let (tok, w) = recv(tok, axi_w_r);
        assert_eq(w, TestAxiW {
            data: TestDataBits:0x00000055,
            strb: TestStrobe:0x1,
            last: true,
        });
        let tok = send(tok, axi_b_s, TestAxiB {
            resp: TestAxiWriteResp::OKAY,
            id: TestId:2,
        });
        let (tok, resp) = recv(tok, resp_r);
        assert_eq(resp, TestAxiWriterResp{status: TestAxiWriterRespStatus::OKAY});

        // Unaligned single transfer
        let tok = send(tok, req_in_s, TestReq {
            addr: TestAddr:0x24,
            length: TestLength:2
        });
        let tok = send(tok, data_in_s, TestData {
            data: TestDataBits:0x00007766,
            length: TestLength:2,
            last: true,
        });
        let (tok, aw) = recv(tok, axi_aw_r);
        assert_eq(aw, TestAxiAW {
            id: TestId:3,
            addr: TestAddr:0x24,
            size: TestAxiAxSize::MAX_4B_TRANSFER,
            len: u8:0,
            burst: TestAxiAxBurst::INCR,
        });
        let (tok, w) = recv(tok, axi_w_r);
        assert_eq(w, TestAxiW {
            data: TestDataBits:0x00007766,
            strb: TestStrobe:0x3,
            last: true,
        });
        let tok = send(tok, axi_b_s, TestAxiB {
            resp: TestAxiWriteResp::OKAY,
            id: TestId:3,
        });
        let (tok, resp) = recv(tok, resp_r);
        assert_eq(resp, TestAxiWriterResp{status: TestAxiWriterRespStatus::OKAY});

        // Unaligned single transfer
        let tok = send(tok, req_in_s, TestReq {
            addr: TestAddr:0x38,
            length: TestLength:3
        });
        let tok = send(tok, data_in_s, TestData {
            data: TestDataBits:0x00AA9988,
            length: TestLength:3,
            last: true,
        });
        let (tok, aw) = recv(tok, axi_aw_r);
        assert_eq(aw, TestAxiAW {
            id: TestId:4,
            addr: TestAddr:0x38,
            size: TestAxiAxSize::MAX_4B_TRANSFER,
            len: u8:0,
            burst: TestAxiAxBurst::INCR,
        });
        let (tok, w) = recv(tok, axi_w_r);
        assert_eq(w, TestAxiW {
            data: TestDataBits:0x00AA9988,
            strb: TestStrobe:0x7,
            last: true,
        });
        let tok = send(tok, axi_b_s, TestAxiB {
            resp: TestAxiWriteResp::OKAY,
            id: TestId:4,
        });
        let (tok, resp) = recv(tok, resp_r);
        assert_eq(resp, TestAxiWriterResp{status: TestAxiWriterRespStatus::OKAY});

        // Unaligned single transfer
        let tok = send(tok, req_in_s, TestReq {
            addr: TestAddr:0x71,
            length: TestLength:1
        });
        let tok = send(tok, data_in_s, TestData {
            data: TestDataBits:0x00000088,
            length: TestLength:1,
            last: true,
        });
        let (tok, aw) = recv(tok, axi_aw_r);
        assert_eq(aw, TestAxiAW {
            id: TestId:5,
            addr: TestAddr:0x70,
            size: TestAxiAxSize::MAX_4B_TRANSFER,
            len: u8:0,
            burst: TestAxiAxBurst::INCR,
        });
        let (tok, w) = recv(tok, axi_w_r);
        assert_eq(w, TestAxiW {
            data: TestDataBits:0x00008800,
            strb: TestStrobe:0x2,
            last: true,
        });
        let tok = send(tok, axi_b_s, TestAxiB {
            resp: TestAxiWriteResp::OKAY,
            id: TestId:5,
        });
        let (tok, resp) = recv(tok, resp_r);
        assert_eq(resp, TestAxiWriterResp{status: TestAxiWriterRespStatus::OKAY});

        // Unaligned 2 transfers
        let tok = send(tok, req_in_s, TestReq {
            addr: TestAddr:0xf3,
            length: TestLength:3
        });
        let tok = send(tok, data_in_s, TestData {
            data: TestDataBits:0x00112233,
            length: TestLength:3,
            last: true,
        });
        let (tok, aw) = recv(tok, axi_aw_r);
        assert_eq(aw, TestAxiAW {
            id: TestId:6,
            addr: TestAddr:0xf0,
            size: TestAxiAxSize::MAX_4B_TRANSFER,
            len: u8:1,
            burst: TestAxiAxBurst::INCR,
        });
        let (tok, w) = recv(tok, axi_w_r);
        assert_eq(w, TestAxiW {
            data: TestDataBits:0x33000000,
            strb: TestStrobe:0x8,
            last: false,
        });
        let (tok, w) = recv(tok, axi_w_r);
        assert_eq(w, TestAxiW {
            data: TestDataBits:0x00001122,
            strb: TestStrobe:0x3,
            last: true,
        });
        let tok = send(tok, axi_b_s, TestAxiB {
            resp: TestAxiWriteResp::OKAY,
            id: TestId:6,
        });
        let (tok, resp) = recv(tok, resp_r);
        assert_eq(resp, TestAxiWriterResp{status: TestAxiWriterRespStatus::OKAY});

        // Unligned 3 transfers
        let tok = send(tok, req_in_s, TestReq {
            addr: TestAddr:0x1f3,
            length: TestLength:7
        });
        let tok = send(tok, data_in_s, TestData {
            data: TestDataBits:0x11223344,
            length: TestLength:4,
            last: false,
        });
        let tok = send(tok, data_in_s, TestData {
            data: TestDataBits:0x00556677,
            length: TestLength:3,
            last: true,
        });
        let (tok, aw) = recv(tok, axi_aw_r);
        assert_eq(aw, TestAxiAW {
            id: TestId:7,
            addr: TestAddr:0x1f0,
            size: TestAxiAxSize::MAX_4B_TRANSFER,
            len: u8:2,
            burst: TestAxiAxBurst::INCR,
        });
        let (tok, w) = recv(tok, axi_w_r);
        assert_eq(w, TestAxiW {
            data: TestDataBits:0x44000000,
            strb: TestStrobe:0x8,
            last: false,
        });
        let (tok, w) = recv(tok, axi_w_r);
        assert_eq(w, TestAxiW {
            data: TestDataBits:0x77112233,
            strb: TestStrobe:0xF,
            last: false,
        });
        let (tok, w) = recv(tok, axi_w_r);
        assert_eq(w, TestAxiW {
            data: TestDataBits:0x00005566,
            strb: TestStrobe:0x3,
            last: true,
        });
        let tok = send(tok, axi_b_s, TestAxiB {
            resp: TestAxiWriteResp::OKAY,
            id: TestId:7,
        });
        let (tok, resp) = recv(tok, resp_r);
        assert_eq(resp, TestAxiWriterResp{status: TestAxiWriterRespStatus::OKAY});

        // Crossing AXI 4kB boundary, aligned 2 burst transfers
        let tok = send(tok, req_in_s, TestReq {
            addr: TestAddr:0x0FFC,
            length: TestLength:8
        });
        let tok = send(tok, data_in_s, TestData {
            data: TestDataBits:0x44332211,
            length: TestLength:4,
            last: false,
        });
        let tok = send(tok, data_in_s, TestData {
            data: TestDataBits:0x88776655,
            length: TestLength:4,
            last: true,
        });
        let (tok, aw) = recv(tok, axi_aw_r);
        assert_eq(aw, TestAxiAW {
            id: TestId:8,
            addr: TestAddr:0xFFC,
            size: TestAxiAxSize::MAX_4B_TRANSFER,
            len: u8:0,
            burst: TestAxiAxBurst::INCR,
        });
        let (tok, w) = recv(tok, axi_w_r);
        assert_eq(w, TestAxiW {
            data: TestDataBits:0x44332211,
            strb: TestStrobe:0b1111,
            last: true,
        });
        let tok = send(tok, axi_b_s, TestAxiB {
            resp: TestAxiWriteResp::OKAY,
            id: TestId:8,
        });
        let (tok, aw) = recv(tok, axi_aw_r);
        assert_eq(aw, TestAxiAW {
            id: TestId:9,
            addr: TestAddr:0x1000,
            size: TestAxiAxSize::MAX_4B_TRANSFER,
            len: u8:0,
            burst: TestAxiAxBurst::INCR,
        });
        let (tok, w) = recv(tok, axi_w_r);
        assert_eq(w, TestAxiW {
            data: TestDataBits:0x88776655,
            strb: TestStrobe:0b1111,
            last: true,
        });
        let tok = send(tok, axi_b_s, TestAxiB {
            resp: TestAxiWriteResp::OKAY,
            id: TestId:9,
        });
        let (tok, resp) = recv(tok, resp_r);
        assert_eq(resp, TestAxiWriterResp{status: TestAxiWriterRespStatus::OKAY});

        // Crossing AXI 4kB boundary, unaligned 2 burst transfers
        let tok = send(tok, req_in_s, TestReq {
            addr: TestAddr:0x1FFF,
            length: TestLength:7
        });
        let tok = send(tok, data_in_s, TestData {
            data: TestDataBits:0x44332211,
            length: TestLength:4,
            last: false,
        });
        let tok = send(tok, data_in_s, TestData {
            data: TestDataBits:0x00776655,
            length: TestLength:3,
            last: true,
        });
        let (tok, aw) = recv(tok, axi_aw_r);
        assert_eq(aw, TestAxiAW {
            id: TestId:10,
            addr: TestAddr:0x1FFC,
            size: TestAxiAxSize::MAX_4B_TRANSFER,
            len: u8:0,
            burst: TestAxiAxBurst::INCR,
        });
        let (tok, w) = recv(tok, axi_w_r);
        assert_eq(w, TestAxiW {
            data: TestDataBits:0x11000000,
            strb: TestStrobe:0b1000,
            last: true,
        });
        let tok = send(tok, axi_b_s, TestAxiB {
            resp: TestAxiWriteResp::OKAY,
            id: TestId:7,
        });
        let (tok, aw) = recv(tok, axi_aw_r);
        assert_eq(aw, TestAxiAW {
            id: TestId:11,
            addr: TestAddr:0x2000,
            size: TestAxiAxSize::MAX_4B_TRANSFER,
            len: u8:1,
            burst: TestAxiAxBurst::INCR,
        });
        let (tok, w) = recv(tok, axi_w_r);
        assert_eq(w, TestAxiW {
            data: TestDataBits:0x55443322,
            strb: TestStrobe:0b1111,
            last: false,
        });
        let (tok, w) = recv(tok, axi_w_r);
        assert_eq(w, TestAxiW {
            data: TestDataBits:0x00007766,
            strb: TestStrobe:0b0011,
            last: true,
        });
        let tok = send(tok, axi_b_s, TestAxiB {
            resp: TestAxiWriteResp::OKAY,
            id: TestId:11,
        });
        let (tok, resp) = recv(tok, resp_r);
        assert_eq(resp, TestAxiWriterResp{status: TestAxiWriterRespStatus::OKAY});

        // Unligned 3 transfers
        let tok = send(tok, req_in_s, TestReq {
            addr: TestAddr:0x1f3,
            length: TestLength:15
        });
        let tok = send(tok, data_in_s, TestData {
            data: TestDataBits:0x11223344,
            length: TestLength:4,
            last: false,
        });
        let tok = send(tok, data_in_s, TestData {
            data: TestDataBits:0x00005566,
            length: TestLength:2,
            last: false,
        });
        let tok = send(tok, data_in_s, TestData {
            data: TestDataBits:0x778899aa,
            length: TestLength:4,
            last: false,
        });
        let tok = send(tok, data_in_s, TestData {
            data: TestDataBits:0x00bbccdd,
            length: TestLength:3,
            last: false,
        });
        let tok = send(tok, data_in_s, TestData {
            data: TestDataBits:0x0000eeff,
            length: TestLength:2,
            last: true,
        });
        let (tok, aw) = recv(tok, axi_aw_r);
        assert_eq(aw, TestAxiAW {
            id: TestId:12,
            addr: TestAddr:0x1f0,
            size: TestAxiAxSize::MAX_4B_TRANSFER,
            len: u8:4,
            burst: TestAxiAxBurst::INCR,
        });
        let (tok, w) = recv(tok, axi_w_r);
        assert_eq(w, TestAxiW {
            data: TestDataBits:0x44000000,
            strb: TestStrobe:0x8,
            last: false,
        });
        let (tok, w) = recv(tok, axi_w_r);
        assert_eq(w, TestAxiW {
            data: TestDataBits:0x66112233,
            strb: TestStrobe:0xF,
            last: false,
        });
        let (tok, w) = recv(tok, axi_w_r);
        assert_eq(w, TestAxiW {
            data: TestDataBits:0x8899aa55,
            strb: TestStrobe:0xf,
            last: false,
        });
        let (tok, w) = recv(tok, axi_w_r);
        assert_eq(w, TestAxiW {
            data: TestDataBits:0xbbccdd77,
            strb: TestStrobe:0xf,
            last: false,
        });
        let (tok, w) = recv(tok, axi_w_r);
        assert_eq(w, TestAxiW {
            data: TestDataBits:0x0000eeff,
            strb: TestStrobe:0x3,
            last: true,
        });
        let tok = send(tok, axi_b_s, TestAxiB {
            resp: TestAxiWriteResp::OKAY,
            id: TestId:12,
        });
        let (tok, resp) = recv(tok, resp_r);
        assert_eq(resp, TestAxiWriterResp{status: TestAxiWriterRespStatus::OKAY});

        send(tok, terminator, true);
    }
}
