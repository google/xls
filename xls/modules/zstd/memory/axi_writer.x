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

// AXI Writer
//
// Part of the main controller, which translates write requests
// (address and length tuples) into AXI Write Transactions.
// Data to write is read from the AXI Stream interface which comes from the
// AxiStreamAddEmpty proc which is responsible for preparing the data for writes
// under unaligned addresses

import std;

import xls.modules.zstd.memory.axi;
import xls.modules.zstd.memory.axi_st;
import xls.modules.zstd.memory.common;

pub struct AxiWriterRequest<ADDR_W: u32> {
    address: uN[ADDR_W],
    length: uN[ADDR_W]
}

enum AxiWriterFsm : u4 {
    IDLE = 0,
    TRANSFER_LENGTH = 1,
    CALC_NEXT_TRANSFER = 2,
    TRANSFER_DISPATCH = 3,
    AXI_WRITE_AW = 4,
    AXI_WRITE_W = 5,
    AXI_WRITE_B = 6,
    RESP_OK = 7,
    ERROR = 15,
}

pub enum AxiWriterRespStatus : u1 {
    OKAY = 0,
    ERROR = 1,
}

pub struct AxiWriterResp {
    status: AxiWriterRespStatus
}

struct AxiWriterState<
    ADDR_W: u32,
    DATA_W: u32,
    ID_W: u32,
    DATA_W_DIV8: u32,
    LANE_W: u32
> {
    fsm: AxiWriterFsm,
    transfer_data: AxiWriterRequest<ADDR_W>,
    aw_bundle: axi::AxiAw<ADDR_W, ID_W>,
    w_bundle: axi::AxiW<DATA_W, DATA_W_DIV8>,
    b_bundle: axi::AxiB<ID_W>,
    burst_counter: u8,
    burst_end: u8,
    recv_new_write_req: bool,
    transaction_len: uN[ADDR_W],
    bytes_to_4k: uN[ADDR_W],
    bytes_to_max_axi_burst: uN[ADDR_W],
    address_align_offset: uN[ADDR_W],
    req_low_lane: uN[LANE_W],
    req_high_lane: uN[LANE_W],
}

pub proc AxiWriter<
    ADDR_W: u32, DATA_W: u32, DEST_W: u32, ID_W: u32,
    DATA_W_DIV8: u32 = {DATA_W / u32:8},
    LANE_W: u32 = {std::clog2(DATA_W / u32:8)}
> {
    type Req = AxiWriterRequest<ADDR_W>;
    type AxiStream = axi_st::AxiStream<DATA_W, DEST_W, ID_W, DATA_W_DIV8>;
    type AxiAw = axi::AxiAw<ADDR_W, ID_W>;
    type AxiW = axi::AxiW<DATA_W, DATA_W_DIV8>;
    type AxiB = axi::AxiB<ID_W>;
    type Resp = axi::AxiWriteResp;
    type AxiAxSize = axi::AxiAxSize;
    type AxiAxBurst = axi::AxiAxBurst;
    type State = AxiWriterState<ADDR_W, DATA_W, ID_W, DATA_W_DIV8, LANE_W>;
    type Fsm = AxiWriterFsm;

    type Addr = uN[ADDR_W];
    type Lane = uN[LANE_W];
    type Id = uN[ID_W];
    type Length = u8;

    ch_write_req: chan<Req> in;
    ch_write_resp: chan<AxiWriterResp> out;
    ch_axi_aw: chan<AxiAw> out;
    ch_axi_w: chan<AxiW> out;
    ch_axi_b: chan<AxiB> in;
    ch_axi_st_read: chan<AxiStream> in;

    config(
        ch_write_req: chan<Req> in,
        ch_write_resp: chan<AxiWriterResp> out,
        ch_axi_aw: chan<AxiAw> out,
        ch_axi_w: chan<AxiW> out,
        ch_axi_b: chan<AxiB> in,
        ch_axi_st_read: chan<AxiStream> in
    ) {
        (ch_write_req, ch_write_resp, ch_axi_aw, ch_axi_w, ch_axi_b, ch_axi_st_read)
    }

    init {
        State {
            recv_new_write_req: true,
            ..zero!<State>()
        }
    }

    next(state: State) {
        const BYTES_IN_TRANSFER = DATA_W_DIV8 as Addr;
        const MAX_AXI_BURST_BYTES = Addr:256 * BYTES_IN_TRANSFER;

        let tok_0 = join();

        // Address Generator
        let (tok_1, recv_transfer_data) = recv_if(
            tok_0, ch_write_req, state.fsm == Fsm::IDLE && state.recv_new_write_req,
            state.transfer_data);

        let tok_2 = send_if(tok_0, ch_axi_aw, state.fsm == Fsm::AXI_WRITE_AW, state.aw_bundle);

        let (tok_3, r_data) = recv_if(
            tok_0, ch_axi_st_read, state.fsm == Fsm::AXI_WRITE_W,
            zero!<AxiStream>());

        // Wait for B
        let (tok_5, b_data) = recv_if(
            tok_0, ch_axi_b, state.fsm == Fsm::AXI_WRITE_B,
            AxiB { resp: Resp::OKAY, id: Id:0 });

        let req_end = state.fsm == Fsm::RESP_OK && state.recv_new_write_req;
        let error_state = state.fsm == Fsm::ERROR;
        let resp = if (error_state) {AxiWriterResp{status: AxiWriterRespStatus::ERROR}} else {AxiWriterResp{status: AxiWriterRespStatus::OKAY}};
        let do_handle_resp = error_state | req_end;
        let tok_6 = send_if(tok_0, ch_write_resp, do_handle_resp, resp);

        let next_state = match(state.fsm) {
            Fsm::IDLE => {
                let bytes_to_4k = common::bytes_to_4k_boundary(recv_transfer_data.address);
                let address_align_offset = common::offset<DATA_W_DIV8>(recv_transfer_data.address) as Addr;
                let bytes_to_max_axi_burst = MAX_AXI_BURST_BYTES - address_align_offset as Addr;
                State {
                    fsm: Fsm::TRANSFER_LENGTH,
                    transfer_data: recv_transfer_data,
                    address_align_offset: address_align_offset,
                    bytes_to_4k: bytes_to_4k,
                    bytes_to_max_axi_burst: bytes_to_max_axi_burst,
                    ..state
                }
            },
            Fsm::TRANSFER_LENGTH => {
                let tran_len = std::min(state.transfer_data.length, std::min(state.bytes_to_4k, state.bytes_to_max_axi_burst));
                State {
                    fsm: Fsm::CALC_NEXT_TRANSFER,
                    transaction_len: tran_len,
                    ..state
                }
            },
            Fsm::CALC_NEXT_TRANSFER => {
                let next_address = state.transfer_data.address + state.transaction_len;
                let next_length = state.transfer_data.length - state.transaction_len;
                let next_transfer_data = Req {
                    address: next_address,
                    length: next_length,
                };
                let (req_low_lane, req_high_lane) = common::get_lanes<DATA_W_DIV8>(state.transfer_data.address, state.transaction_len);
                let aw_addr = common::align<DATA_W_DIV8>(recv_transfer_data.address);

                State {
                    fsm: Fsm::TRANSFER_DISPATCH,
                    aw_bundle: AxiAw {
                        addr: aw_addr,
                        ..state.aw_bundle
                    },
                    transfer_data: next_transfer_data,
                    req_low_lane: req_low_lane,
                    req_high_lane: req_high_lane,
                    ..state
                }
            },
            Fsm::TRANSFER_DISPATCH => {
                let recv_new_write_req = state.transfer_data.length == Addr:0;
                let id = state.aw_bundle.id + Id:1;
                let full_transaction_len = state.transaction_len + state.address_align_offset;
                let div = std::div_pow2(full_transaction_len, DATA_W_DIV8 as Addr);
                let rem = std::mod_pow2(full_transaction_len, DATA_W_DIV8 as Addr);
                let len = if (rem == Addr:0) { (div - Addr:1) as Length } else { div as Length };

                State {
                    fsm: Fsm::AXI_WRITE_AW,
                    aw_bundle: AxiAw {
                        id: id,
                        size: common::axsize<DATA_W_DIV8>(),
                        len: len,
                        burst: AxiAxBurst::INCR,
                        ..state.aw_bundle
                    },
                    burst_end: len,
                    recv_new_write_req: recv_new_write_req,
                    ..state
                }
            },
            Fsm::AXI_WRITE_AW => {

                State {
                    fsm: Fsm::AXI_WRITE_W,
                    ..state
                }
            },
            Fsm::AXI_WRITE_W => {
                let next_burst_counter = state.burst_counter + Length:1;

                let (next_fsm, req_low_lane, req_high_lane) = if (state.burst_counter == state.burst_end) {
                    (Fsm::AXI_WRITE_B, state.req_low_lane, state.req_high_lane)
                } else {
                    (Fsm::AXI_WRITE_W, Lane:0, state.req_high_lane)
                };

                State {
                    fsm: next_fsm,
                    burst_counter: next_burst_counter,
                    req_low_lane: req_low_lane,
                    req_high_lane: req_high_lane,
                    ..state
                }
            },
            Fsm::AXI_WRITE_B => {
                if (b_data.resp == Resp::OKAY) {
                    State {
                        fsm: Fsm::RESP_OK,
                        b_bundle: b_data,
                        burst_counter: Length:0,
                        ..state
                    }
                } else {
                    State {
                        fsm: Fsm::ERROR,
                        b_bundle: b_data,
                        ..state
                    }
                }
            },
            Fsm::RESP_OK => {
                State {
                    fsm: Fsm::IDLE,
                    ..state
                }
            },
            Fsm::ERROR => {
                State {
                    fsm: Fsm::IDLE,
                    ..state
                }
            },
            _ => {
                assert!(false, "Invalid state");
                State {
                    fsm: Fsm::ERROR,
                    ..state
                }
            }
        };

        let w_bundle = match(state.fsm) {
            Fsm::AXI_WRITE_W => {
                let last = state.burst_counter == state.burst_end;
                let low_lane = state.req_low_lane;
                let high_lane = if (last) { state.req_high_lane } else {Lane:3};
                let mask = common::lane_mask<DATA_W_DIV8>(low_lane, high_lane);

                AxiW {
                    data: r_data.data,
                    strb: mask,
                    last: last,
                }
            },
            _ => {
                zero!<AxiW>()
            }
        };

        // Send W
        let tok_4 = send_if(
            tok_3, ch_axi_w, state.fsm == Fsm::AXI_WRITE_W, w_bundle);

        next_state
    }
}

const INST_ADDR_W = u32:16;
const INST_DATA_W = u32:32;
const INST_DATA_W_DIV8 = INST_DATA_W / u32:8;
const INST_DEST_W = INST_DATA_W / u32:8;
const INST_ID_W = INST_DATA_W / u32:8;

proc AxiWriterInst {
    type InstReq = AxiWriterRequest<INST_ADDR_W>;
    type InstAxiWriterResp = AxiWriterResp;
    type InstAxiStream = axi_st::AxiStream<INST_DATA_W, INST_DEST_W, INST_ID_W, INST_DATA_W_DIV8>;
    type InstAxiAw = axi::AxiAw<INST_ADDR_W, INST_ID_W>;
    type InstAxiW = axi::AxiW<INST_DATA_W, INST_DATA_W_DIV8>;
    type InstAxiB = axi::AxiB<INST_ID_W>;

    config(ch_write_req: chan<InstReq> in,
           ch_write_resp: chan<InstAxiWriterResp> out,
           ch_axi_aw: chan<InstAxiAw> out,
           ch_axi_w: chan<InstAxiW> out,
           ch_axi_b: chan<InstAxiB> in,
           ch_axi_st_read: chan<InstAxiStream> in) {

        spawn AxiWriter<
            INST_ADDR_W, INST_DATA_W, INST_DEST_W, INST_ID_W>(
            ch_write_req, ch_write_resp, ch_axi_aw, ch_axi_w, ch_axi_b, ch_axi_st_read);
        ()
    }

    init { () }

    next(state: ()) {  }
}

const TEST_ADDR_W = u32:16;
const TEST_DATA_W = u32:32;
const TEST_DATA_W_DIV8 = INST_DATA_W / u32:8;
const TEST_DEST_W = INST_DATA_W / u32:8;
const TEST_ID_W = INST_DATA_W / u32:8;

#[test_proc]
proc AxiWriterTest {
    type TestReq = AxiWriterRequest<TEST_ADDR_W>;
    type TestAxiWriterResp = AxiWriterResp;
    type TestAxiStream = axi_st::AxiStream<TEST_DATA_W, TEST_DEST_W, TEST_ID_W, TEST_DATA_W_DIV8>;
    type TestAxiAw = axi::AxiAw<TEST_ADDR_W, TEST_ID_W>;
    type TestAxiW = axi::AxiW<TEST_DATA_W, TEST_DATA_W_DIV8>;
    type TestAxiB = axi::AxiB<TEST_ID_W>;

    type TestAxiWriteResp = axi::AxiWriteResp;
    type TestAxiWriterRespStatus = AxiWriterRespStatus;
    type TestAxiAxBurst = axi::AxiAxBurst;
    type TestAxiAxSize = axi::AxiAxSize;
    type TestAddr = uN[TEST_ADDR_W];
    type TestLength = uN[TEST_ADDR_W];
    type TestDataBits = uN[TEST_DATA_W];
    type TestStrobe = uN[TEST_DATA_W_DIV8];
    type TestId = uN[TEST_ID_W];
    type TestDest = uN[TEST_DEST_W];

    terminator: chan<bool> out;
    write_req_s: chan<TestReq> out;
    write_resp_r: chan<TestAxiWriterResp> in;
    axi_aw_r: chan<TestAxiAw> in;
    axi_w_r: chan<TestAxiW> in;
    axi_b_s: chan<TestAxiB> out;
    axi_st_read_s: chan<TestAxiStream> out;

    config(
        terminator: chan<bool> out,
    ) {
        let (write_req_s, write_req_r) = chan<TestReq>("write_req");
        let (write_resp_s, write_resp_r) = chan<TestAxiWriterResp>("write_resp");
        let (axi_aw_s, axi_aw_r) = chan<TestAxiAw>("axi_aw");
        let (axi_w_s, axi_w_r) = chan<TestAxiW>("axi_w");
        let (axi_b_s, axi_b_r) = chan<TestAxiB>("axi_b");
        let (axi_st_read_s, axi_st_read_r) = chan<TestAxiStream>("axi_st");

        spawn AxiWriter<
            TEST_ADDR_W, TEST_DATA_W, TEST_DEST_W, TEST_ID_W
        >(write_req_r, write_resp_s, axi_aw_s, axi_w_s, axi_b_r, axi_st_read_r);
        (terminator, write_req_s, write_resp_r, axi_aw_r, axi_w_r, axi_b_s, axi_st_read_s)
    }

    init { () }

    next(state: ()) {
        let tok = join();

        // Aligned single transfer
        let tok = send(tok, write_req_s, TestReq {
            address: TestAddr:0x0,
            length: TestLength:4
        });
        let tok = send(tok, axi_st_read_s, TestAxiStream {
            data: TestDataBits:0x11223344,
            str: TestStrobe:0b1111,
            keep: TestStrobe:0b1111,
            id: TestId:1,
            dest: TestDest:0,
            last: true,
        });
        let (tok, aw) = recv(tok, axi_aw_r);
        assert_eq(aw, TestAxiAw {
            id: TestId:1,
            addr: TestAddr:0,
            size: TestAxiAxSize::MAX_4B_TRANSFER,
            len: u8:0,
            burst: TestAxiAxBurst::INCR,
        });
        let (tok, w) = recv(tok, axi_w_r);
        assert_eq(w, TestAxiW {
            data: TestDataBits:0x11223344,
            strb: TestStrobe:0xF,
            last: true,
        });
        let tok = send(tok, axi_b_s, TestAxiB {
            resp: TestAxiWriteResp::OKAY,
            id: TestId:1,
        });
        let (tok, resp) = recv(tok, write_resp_r);
        assert_eq(resp, TestAxiWriterResp{status: TestAxiWriterRespStatus::OKAY});

        // Unaligned 2 transfers
        let tok = send(tok, write_req_s, TestReq {
            address: TestAddr:0xf3,
            length: TestLength:3
        });
        let tok = send(tok, axi_st_read_s, TestAxiStream {
            data: TestDataBits:0x11000000,
            str: TestStrobe:0b1000,
            keep: TestStrobe:0b1000,
            id: TestId:2,
            dest: TestDest:0,
            last: false,
        });
        let tok = send(tok, axi_st_read_s, TestAxiStream {
            data: TestDataBits:0x00003322,
            str: TestStrobe:0b0011,
            keep: TestStrobe:0b0011,
            id: TestId:2,
            dest: TestDest:0,
            last: true,
        });
        let (tok, aw) = recv(tok, axi_aw_r);
        assert_eq(aw, TestAxiAw {
            id: TestId:2,
            addr: TestAddr:0xf0,
            size: TestAxiAxSize::MAX_4B_TRANSFER,
            len: u8:1,
            burst: TestAxiAxBurst::INCR,
        });
        let (tok, w) = recv(tok, axi_w_r);
        assert_eq(w, TestAxiW {
            data: TestDataBits:0x11000000,
            strb: TestStrobe:0b1000,
            last: false,
        });
        let (tok, w) = recv(tok, axi_w_r);
        assert_eq(w, TestAxiW {
            data: TestDataBits:0x00003322,
            strb: TestStrobe:0b0011,
            last: true,
        });
        let tok = send(tok, axi_b_s, TestAxiB {
            resp: TestAxiWriteResp::OKAY,
            id: TestId:2,
        });
        let (tok, resp) = recv(tok, write_resp_r);
        assert_eq(resp, TestAxiWriterResp{status: TestAxiWriterRespStatus::OKAY});

        // Aligned 2 transfers
        let tok = send(tok, write_req_s, TestReq {
            address: TestAddr:0x100,
            length: TestLength:8
        });
        let tok = send(tok, axi_st_read_s, TestAxiStream {
            data: TestDataBits:0x44332211,
            str: TestStrobe:0b1111,
            keep: TestStrobe:0b1111,
            id: TestId:3,
            dest: TestDest:0,
            last: false,
        });
        let tok = send(tok, axi_st_read_s, TestAxiStream {
            data: TestDataBits:0x88776655,
            str: TestStrobe:0b1111,
            keep: TestStrobe:0b1111,
            id: TestId:3,
            dest: TestDest:0,
            last: true,
        });
        let (tok, aw) = recv(tok, axi_aw_r);
        assert_eq(aw, TestAxiAw {
            id: TestId:3,
            addr: TestAddr:0x100,
            size: TestAxiAxSize::MAX_4B_TRANSFER,
            len: u8:1,
            burst: TestAxiAxBurst::INCR,
        });
        let (tok, w) = recv(tok, axi_w_r);
        assert_eq(w, TestAxiW {
            data: TestDataBits:0x44332211,
            strb: TestStrobe:0b1111,
            last: false,
        });
        let (tok, w) = recv(tok, axi_w_r);
        assert_eq(w, TestAxiW {
            data: TestDataBits:0x88776655,
            strb: TestStrobe:0b1111,
            last: true,
        });
        let tok = send(tok, axi_b_s, TestAxiB {
            resp: TestAxiWriteResp::OKAY,
            id: TestId:3,
        });
        let (tok, resp) = recv(tok, write_resp_r);
        assert_eq(resp, TestAxiWriterResp{status: TestAxiWriterRespStatus::OKAY});

        // Unligned 3 transfers
        let tok = send(tok, write_req_s, TestReq {
            address: TestAddr:0x1F3,
            length: TestLength:7
        });
        let tok = send(tok, axi_st_read_s, TestAxiStream {
            data: TestDataBits:0x11000000,
            str: TestStrobe:0b1000,
            keep: TestStrobe:0b1000,
            id: TestId:4,
            dest: TestDest:0,
            last: false,
        });
        let tok = send(tok, axi_st_read_s, TestAxiStream {
            data: TestDataBits:0x55443322,
            str: TestStrobe:0b1111,
            keep: TestStrobe:0b1111,
            id: TestId:4,
            dest: TestDest:0,
            last: false,
        });
        let tok = send(tok, axi_st_read_s, TestAxiStream {
            data: TestDataBits:0x00007766,
            str: TestStrobe:0b0011,
            keep: TestStrobe:0b0011,
            id: TestId:4,
            dest: TestDest:0,
            last: true,
        });
        let (tok, aw) = recv(tok, axi_aw_r);
        assert_eq(aw, TestAxiAw {
            id: TestId:4,
            addr: TestAddr:0x1F0,
            size: TestAxiAxSize::MAX_4B_TRANSFER,
            len: u8:2,
            burst: TestAxiAxBurst::INCR,
        });
        let (tok, w) = recv(tok, axi_w_r);
        assert_eq(w, TestAxiW {
            data: TestDataBits:0x11000000,
            strb: TestStrobe:0b1000,
            last: false,
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
            id: TestId:4,
        });
        let (tok, resp) = recv(tok, write_resp_r);
        assert_eq(resp, TestAxiWriterResp{status: TestAxiWriterRespStatus::OKAY});

        // Crossing AXI 4kB boundary, aligned 2 burst transfers
        let tok = send(tok, write_req_s, TestReq {
            address: TestAddr:0x0FFC,
            length: TestLength:8
        });
        let tok = send(tok, axi_st_read_s, TestAxiStream {
            data: TestDataBits:0x44332211,
            str: TestStrobe:0b1111,
            keep: TestStrobe:0b1111,
            id: TestId:5,
            dest: TestDest:0,
            last: false,
        });
        let tok = send(tok, axi_st_read_s, TestAxiStream {
            data: TestDataBits:0x88776655,
            str: TestStrobe:0b1111,
            keep: TestStrobe:0b1111,
            id: TestId:5,
            dest: TestDest:0,
            last: true,
        });
        let (tok, aw) = recv(tok, axi_aw_r);
        assert_eq(aw, TestAxiAw {
            id: TestId:5,
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
            id: TestId:5,
        });
        let (tok, aw) = recv(tok, axi_aw_r);
        assert_eq(aw, TestAxiAw {
            id: TestId:6,
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
            id: TestId:6,
        });
        let (tok, resp) = recv(tok, write_resp_r);
        assert_eq(resp, TestAxiWriterResp{status: TestAxiWriterRespStatus::OKAY});

        // Crossing AXI 4kB boundary, unaligned 2 burst transfers
        let tok = send(tok, write_req_s, TestReq {
            address: TestAddr:0x1FFF,
            length: TestLength:7
        });
        let tok = send(tok, axi_st_read_s, TestAxiStream {
            data: TestDataBits:0x11000000,
            str: TestStrobe:0b1000,
            keep: TestStrobe:0b1000,
            id: TestId:7,
            dest: TestDest:0,
            last: true,
        });
        let tok = send(tok, axi_st_read_s, TestAxiStream {
            data: TestDataBits:0x55443322,
            str: TestStrobe:0b1111,
            keep: TestStrobe:0b1111,
            id: TestId:7,
            dest: TestDest:0,
            last: false,
        });
        let tok = send(tok, axi_st_read_s, TestAxiStream {
            data: TestDataBits:0x00007766,
            str: TestStrobe:0b0011,
            keep: TestStrobe:0b0011,
            id: TestId:7,
            dest: TestDest:0,
            last: true,
        });
        let (tok, aw) = recv(tok, axi_aw_r);
        assert_eq(aw, TestAxiAw {
            id: TestId:7,
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
        assert_eq(aw, TestAxiAw {
            id: TestId:8,
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
            id: TestId:8,
        });
        let (tok, resp) = recv(tok, write_resp_r);
        assert_eq(resp, TestAxiWriterResp{status: TestAxiWriterRespStatus::OKAY});

        send(tok, terminator, true);
    }
}
