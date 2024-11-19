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

import xls.modules.zstd.memory.axi as axi;

pub struct HuffmanAxiReaderCtrl<AXI_ADDR_W: u32> {
    base_addr: uN[AXI_ADDR_W],
    len: uN[AXI_ADDR_W],
}

pub struct HuffmanAxiReaderData {
    data: u8,
    last: bool,
}

struct HuffmanAxiReaderState<AXI_DATA_W: u32, AXI_ADDR_W: u32> {
    ctrl: HuffmanAxiReaderCtrl<AXI_ADDR_W>,
    bytes_requested: uN[AXI_ADDR_W],
    bytes_sent: uN[AXI_ADDR_W],
}

pub proc HuffmanAxiReader<AXI_DATA_W: u32, AXI_ADDR_W: u32, AXI_ID_W: u32> {
    // FIXME: Replace hard-coded values with proc params AXI_DATA_W, AXI_ID_W
    type AxiR = axi::AxiR<u32:32, u32:32>;
    // FIXME: Replace hard-coded values with proc params AXI_ADDR_W, AXI_ID_W
    type AxiAr = axi::AxiAr<u32:32, u32:32>;

    // FIXME: Replace hard-coded values with proc params AXI_ADDR_W
    type Ctrl = HuffmanAxiReaderCtrl<u32:32>;
    type Data = HuffmanAxiReaderData;

    // FIXME: Replace hard-coded values with proc params AXI_DATA_W, AXI_ADDR_W
    type State = HuffmanAxiReaderState<u32:32, u32:32>;

    ctrl_r: chan<Ctrl> in;
    axi_r_r: chan<AxiR> in;
    axi_ar_s: chan<AxiAr> out;
    data_s: chan<Data> out;

    config (
        ctrl_r: chan<Ctrl> in,
        axi_r_r: chan<AxiR> in,
        axi_ar_s: chan<AxiAr> out,
        data_s: chan<Data> out,
    ) {
        (
            ctrl_r,
            axi_r_r,
            axi_ar_s,
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
        let axi_ar = AxiAr {
            id: uN[AXI_ID_W]:0,
            addr: addr,
            ..zero!<AxiAr>()
        };
        let do_send_axi_req = (state.bytes_requested < state.ctrl.len);
        send_if(join(), axi_ar_s, do_send_axi_req, axi_ar);
        if (do_send_axi_req) {
            trace_fmt!("Sent AXI read request {:#x}", axi_ar);
        } else {};

        let state = if do_send_axi_req {
            State {
                bytes_requested: state.bytes_requested + uN[AXI_ADDR_W]:1,
                ..state
            }
        } else {
            state
        };

        // receive data from AXI
        let do_read_axi_resp = (state.bytes_requested > state.bytes_sent) && (state.bytes_sent < state.bytes_requested);
        let (tok, axi_r, axi_r_valid) = recv_if_non_blocking(join(), axi_r_r, do_read_axi_resp, zero!<AxiR>());

        // send data
        let last = axi_r_valid && ((state.bytes_sent + uN[AXI_ADDR_W]:1) == state.ctrl.len);
        let tok = send_if(tok, data_s, axi_r_valid, Data {
            data: axi_r.data as u8,
            last: last,
        });
        let state = if last {
            zero!<State>()
        } else if axi_r_valid {
            trace_fmt!("Received AXI read response {:#x}", axi_r);
            State {
                bytes_sent: state.bytes_sent + uN[AXI_ADDR_W]:1,
                ..state
            }
        } else { state };

        state
    }
}

const INST_AXI_DATA_W = u32:32;
const INST_AXI_ADDR_W = u32:32;
const INST_AXI_ID_W = u32:32;

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
        spawn HuffmanAxiReader<INST_AXI_DATA_W, INST_AXI_ADDR_W, INST_AXI_ID_W>(
            ctrl_r,
            axi_r_r,
            axi_ar_s,
            data_s,
        );
    }

    init { }

    next (state: ()) { }
}

const TEST_AXI_DATA_W = u32:32;
const TEST_AXI_ADDR_W = u32:32;
const TEST_AXI_ID_W = u32:32;

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
    TestAxiData { addr: uN[TEST_AXI_ADDR_W]:0, data: uN[TEST_AXI_DATA_W]:0x12, len: u8:0, last: true, },
    TestAxiData { addr: uN[TEST_AXI_ADDR_W]:131, data: uN[TEST_AXI_DATA_W]:0xAA, len: u8:0, last: true, },
    TestAxiData { addr: uN[TEST_AXI_ADDR_W]:130, data: uN[TEST_AXI_DATA_W]:0xBB, len: u8:0, last: true, },
    TestAxiData { addr: uN[TEST_AXI_ADDR_W]:129, data: uN[TEST_AXI_DATA_W]:0xCC, len: u8:0, last: true, },
    TestAxiData { addr: uN[TEST_AXI_ADDR_W]:128, data: uN[TEST_AXI_DATA_W]:0xDD, len: u8:0, last: true, },
    TestAxiData { addr: uN[TEST_AXI_ADDR_W]:65, data: uN[TEST_AXI_DATA_W]:0x44, len: u8:0, last: false, },
    TestAxiData { addr: uN[TEST_AXI_ADDR_W]:64, data: uN[TEST_AXI_DATA_W]:0x55, len: u8:0, last: true, },
];

const TEST_DATA_OUT = HuffmanAxiReaderData[7]:[
    HuffmanAxiReaderData { data: u8:0x12, last: true, },
    HuffmanAxiReaderData { data: u8:0xAA, last: false, },
    HuffmanAxiReaderData { data: u8:0xBB, last: false, },
    HuffmanAxiReaderData { data: u8:0xCC, last: false, },
    HuffmanAxiReaderData { data: u8:0xDD, last: true, },
    HuffmanAxiReaderData { data: u8:0x44, last: false, },
    HuffmanAxiReaderData { data: u8:0x55, last: true, },
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

        spawn HuffmanAxiReader<TEST_AXI_DATA_W, TEST_AXI_ADDR_W, TEST_AXI_ID_W> (
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

            assert_eq(test_axi.addr, axi_req.addr);
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
