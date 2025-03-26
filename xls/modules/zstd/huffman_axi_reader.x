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

// This file contains the implementation of Huffmann data preprocessor.

import std;
import xls.modules.zstd.memory.axi as axi;
import xls.modules.zstd.memory.mem_reader as mem_reader;

pub struct HuffmanAxiReaderCtrl<AXI_ADDR_W: u32> {
    base_addr: uN[AXI_ADDR_W],
    len: uN[AXI_ADDR_W],
}

pub struct HuffmanAxiReaderData {
    data: u8,
    last: bool,
}

struct HuffmanAxiReaderState<AXI_DATA_W: u32, AXI_ADDR_W: u32,
                             AXI_DATA_DIV8: u32, // = {AXI_DATA_W / u32:8},
                             AXI_DATA_DIV8_W: u32, // = {std::clog2({AXI_DATA_W / u32:8})}
                             > {
    ctrl: HuffmanAxiReaderCtrl<AXI_ADDR_W>,
    bytes_requested: uN[AXI_ADDR_W],
    bytes_sent: uN[AXI_ADDR_W],
}

pub proc HuffmanAxiReader<AXI_DATA_W: u32, AXI_ADDR_W: u32, AXI_ID_W: u32, AXI_DEST_W: u32,
                          AXI_DATA_DIV8: u32 = {AXI_DATA_W / u32:8},
                          AXI_DATA_DIV8_W: u32 = {std::clog2({AXI_DATA_W / u32:8})},
                          > {
    type AxiAr = axi::AxiAr<AXI_ADDR_W, AXI_ID_W>;
    type AxiR = axi::AxiR<AXI_DATA_W, AXI_ID_W>;

    type Ctrl = HuffmanAxiReaderCtrl<AXI_ADDR_W>;
    type Data = HuffmanAxiReaderData;

    type MemRdReq = mem_reader::MemReaderReq<AXI_ADDR_W>;
    type MemRdResp = mem_reader::MemReaderResp<AXI_DATA_W, AXI_ADDR_W>;

    type State = HuffmanAxiReaderState<AXI_DATA_W, AXI_ADDR_W, AXI_DATA_DIV8, AXI_DATA_DIV8_W>;

    ctrl_r: chan<Ctrl> in;
    mem_rd_req_s: chan<MemRdReq> out;
    mem_rd_resp_r: chan<MemRdResp> in;
    data_s: chan<Data> out;

    config (
        ctrl_r: chan<Ctrl> in,
        axi_r_r: chan<AxiR> in,
        axi_ar_s: chan<AxiAr> out,
        data_s: chan<Data> out,
    ) {
        let (mem_rd_req_s, mem_rd_req_r) = chan<MemRdReq, u32:1>("mem_rd_req");
        let (mem_rd_resp_s, mem_rd_resp_r) = chan<MemRdResp, u32:1>("mem_rd_resp");

        spawn mem_reader::MemReader<AXI_DATA_W, AXI_ADDR_W, AXI_DEST_W, AXI_ID_W, u32:1> (
            mem_rd_req_r,
            mem_rd_resp_s,
            axi_ar_s,
            axi_r_r
        );

        (
            ctrl_r,
            mem_rd_req_s,
            mem_rd_resp_r,
            data_s,
        )
    }

    init { zero!<State>() }

    next (state: State) {
        const BYTES_PER_TRANSACTION = (AXI_ADDR_W / u32:8) as u8;

        // receive and store ctrl
        let (_, ctrl, ctrl_valid) = recv_if_non_blocking(join(), ctrl_r, state.ctrl.len == state.bytes_sent, zero!<Ctrl>());

        let state = if ctrl_valid {
            trace_fmt!("Received CTRL {:#x}", ctrl);
            State {
                ctrl: ctrl,
                ..zero!<State>()
            }
        } else { state };

        // send AXI read request
        // this could be optimized to read multiple bytes per AXI transaction
        let addr = state.ctrl.base_addr + state.ctrl.len - uN[AXI_ADDR_W]:1 - state.bytes_requested;
        let mem_rd_req = MemRdReq {
            addr: addr,
            length: uN[AXI_ADDR_W]:1
        };
        let do_send_mem_rd_req = (state.bytes_requested < state.ctrl.len);
        send_if(join(), mem_rd_req_s, do_send_mem_rd_req, mem_rd_req);
        if (do_send_mem_rd_req) {
            trace_fmt!("Sent memory read request {:#x}", mem_rd_req);
        } else {};

        let state = if do_send_mem_rd_req {
            State {
                bytes_requested: state.bytes_requested + uN[AXI_ADDR_W]:1,
                ..state
            }
        } else {
            state
        };

        // receive data
        let do_read_mem_rd_resp = (state.bytes_requested > state.bytes_sent) && (state.bytes_sent < state.bytes_requested);
        let (tok, mem_rd_resp, mem_rd_resp_valid) = recv_if_non_blocking(join(), mem_rd_resp_r, do_read_mem_rd_resp, zero!<MemRdResp>());
        if mem_rd_resp_valid {
            trace_fmt!("Received memory read response {:#x}", mem_rd_resp);
        } else {};

        // send data
        let last = mem_rd_resp_valid && ((state.bytes_sent + uN[AXI_ADDR_W]:1) == state.ctrl.len);
        let data = Data {
            data: mem_rd_resp.data as u8,
            last: last,
        };
        let tok = send_if(tok, data_s, mem_rd_resp_valid, data);
        if mem_rd_resp_valid {
            trace_fmt!("Sent output data {:#x}", data);
        } else {};

        let state = if last {
            zero!<State>()
        } else if mem_rd_resp_valid {
            State {
                bytes_sent: state.bytes_sent + uN[AXI_ADDR_W]:1,
                ..state
            }
        } else { state };

        state
    }
}

const INST_AXI_DATA_W = u32:64;
const INST_AXI_ADDR_W = u32:16;
const INST_AXI_ID_W = u32:4;
const INST_AXI_DEST_W = u32:4;

proc HuffmanAxiReaderInst {
    type InstHuffmanAxiReaderCtrl = HuffmanAxiReaderCtrl<INST_AXI_ADDR_W>;

    type InstAxiAr = axi::AxiAr<INST_AXI_ADDR_W, INST_AXI_ID_W>;
    type InstAxiR = axi::AxiR<INST_AXI_DATA_W, INST_AXI_ID_W>;

    config (
        ctrl_r: chan<InstHuffmanAxiReaderCtrl> in,
        axi_r_r: chan<InstAxiR> in,
        axi_ar_s: chan<InstAxiAr> out,
        data_s: chan<HuffmanAxiReaderData> out,
    ) {
        spawn HuffmanAxiReader<INST_AXI_DATA_W, INST_AXI_ADDR_W, INST_AXI_ID_W, INST_AXI_DEST_W>(
            ctrl_r,
            axi_r_r,
            axi_ar_s,
            data_s,
        );
    }

    init { }

    next (state: ()) { }
}

const TEST_AXI_DATA_W = u32:64;
const TEST_AXI_ADDR_W = u32:16;
const TEST_AXI_ID_W = u32:4;
const TEST_AXI_DEST_W = u32:4;
const TEST_AXI_DATA_DIV8 = TEST_AXI_DATA_W / u32:8;
const TEST_AXI_DATA_DIV8_W = std::clog2(TEST_AXI_DATA_DIV8);

type TestHuffmanAxiReaderCtrl = HuffmanAxiReaderCtrl<TEST_AXI_ADDR_W>;

type TestAxiAr = axi::AxiAr<TEST_AXI_ADDR_W, TEST_AXI_ID_W>;
type TestAxiR = axi::AxiR<TEST_AXI_DATA_W, TEST_AXI_ID_W>;

struct TestAxiData {
    addr: uN[TEST_AXI_ADDR_W],
    data: uN[TEST_AXI_DATA_W],
    len: u8,
    last: bool,
}

const TEST_DATA_CTRL = TestHuffmanAxiReaderCtrl[3]:[
    TestHuffmanAxiReaderCtrl {
        base_addr: uN[TEST_AXI_ADDR_W]:0,
        len: uN[TEST_AXI_ADDR_W]:1,
    },
    TestHuffmanAxiReaderCtrl {
        base_addr: uN[TEST_AXI_ADDR_W]:128,
        len: uN[TEST_AXI_ADDR_W]:4,
    },
    TestHuffmanAxiReaderCtrl {
        base_addr: uN[TEST_AXI_ADDR_W]:64,
        len: uN[TEST_AXI_ADDR_W]:2,
    },
];

const TEST_DATA_AXI = TestAxiData[7]:[
    TestAxiData { addr: uN[TEST_AXI_ADDR_W]:0, data: uN[TEST_AXI_DATA_W]:0x0123456789ABCDF0, len: u8:0, last: true, },
    TestAxiData { addr: uN[TEST_AXI_ADDR_W]:131, data: uN[TEST_AXI_DATA_W]:0x8899AABBCCDDEEFF, len: u8:0, last: true, },
    TestAxiData { addr: uN[TEST_AXI_ADDR_W]:130, data: uN[TEST_AXI_DATA_W]:0x8899AABBCCDDEEFF, len: u8:0, last: true, },
    TestAxiData { addr: uN[TEST_AXI_ADDR_W]:129, data: uN[TEST_AXI_DATA_W]:0x8899AABBCCDDEEFF, len: u8:0, last: true, },
    TestAxiData { addr: uN[TEST_AXI_ADDR_W]:128, data: uN[TEST_AXI_DATA_W]:0x8899AABBCCDDEEFF, len: u8:0, last: true, },
    TestAxiData { addr: uN[TEST_AXI_ADDR_W]:65, data: uN[TEST_AXI_DATA_W]:0xDEADBEEFFEEBDAED, len: u8:0, last: false, },
    TestAxiData { addr: uN[TEST_AXI_ADDR_W]:64, data: uN[TEST_AXI_DATA_W]:0xDEADBEEFFEEBDAED, len: u8:0, last: true, },
];

const TEST_DATA_OUT = HuffmanAxiReaderData[7]:[
    HuffmanAxiReaderData { data: u8:0xF0, last: true, },
    HuffmanAxiReaderData { data: u8:0xCC, last: false, },
    HuffmanAxiReaderData { data: u8:0xDD, last: false, },
    HuffmanAxiReaderData { data: u8:0xEE, last: false, },
    HuffmanAxiReaderData { data: u8:0xFF, last: true, },
    HuffmanAxiReaderData { data: u8:0xDA, last: false, },
    HuffmanAxiReaderData { data: u8:0xED, last: true, },
];

#[test_proc]
proc HuffmanAxiReader_test {
    terminator: chan <bool> out;

    ctrl_s: chan<TestHuffmanAxiReaderCtrl> out;
    axi_r_s: chan<TestAxiR> out;
    axi_ar_r: chan<TestAxiAr> in;
    data_r: chan<HuffmanAxiReaderData> in;

    config (terminator: chan <bool> out) {
        let (ctrl_s, ctrl_r) = chan<TestHuffmanAxiReaderCtrl>("ctrl");
        let (axi_r_s, axi_r_r) = chan<TestAxiR>("axi_r");
        let (axi_ar_s, axi_ar_r) = chan<TestAxiAr>("axi_ar");
        let (data_s, data_r) = chan<HuffmanAxiReaderData>("data");

        spawn HuffmanAxiReader<TEST_AXI_DATA_W, TEST_AXI_ADDR_W, TEST_AXI_ID_W, TEST_AXI_DEST_W> (
            ctrl_r,
            axi_r_r,
            axi_ar_s,
            data_s
        );

        (
            terminator,
            ctrl_s,
            axi_r_s, axi_ar_r,
            data_r,
        )
    }

    init { }

    next (state: ()) {
        let tok = join();

        let tok = for ((i, test_ctrl), tok): ((u32, TestHuffmanAxiReaderCtrl), token) in enumerate(TEST_DATA_CTRL) {
            let tok = send(tok, ctrl_s, test_ctrl);
            trace_fmt!("Sent #{} ctrl {:#x}", i + u32:1, test_ctrl);
            tok
        }(tok);

        let tok = for ((i, test_axi), tok): ((u32, TestAxiData), token) in enumerate(TEST_DATA_AXI) {
            let (tok, axi_req) = recv(tok, axi_ar_r);
            trace_fmt!("Received #{} AXI request {:#x}", i + u32:1, axi_req);
            let aligned_addr = test_axi.addr & !(test_axi.addr % TEST_AXI_DATA_DIV8 as uN[TEST_AXI_ADDR_W]);

            assert_eq(aligned_addr, axi_req.addr);
            assert_eq(test_axi.len, axi_req.len);

            let axi_resp = TestAxiR {
                id: axi_req.id,
                data: test_axi.data,
                resp: axi::AxiReadResp::OKAY,
                last: test_axi.last,
            };
            let tok = send(tok, axi_r_s, axi_resp);
            trace_fmt!("Sent #{} AXI response {:#x}", i + u32:1, axi_resp);

            tok
        }(tok);

        let tok = for ((i, test_data), tok): ((u32, HuffmanAxiReaderData), token) in enumerate(TEST_DATA_OUT) {
            let (tok, data) = recv(tok, data_r);
            trace_fmt!("Received #{} data {:#x}", i + u32:1, data);

            assert_eq(test_data.data as u8, data.data);
            assert_eq(test_data.last, data.last);

            tok
        }(tok);

        send(tok, terminator, true);
    }
}
