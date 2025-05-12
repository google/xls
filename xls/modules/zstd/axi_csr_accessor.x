// Copyright 2023-2024 The XLS Authors
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

// This file contains implementation of a proc that handles CSRs. It provides
// an AXI interface for reading and writing the values as well as separate
// request/response channels. Apart from that it has an output channel which
// notifies aboud changes made to CSRs.

import std;
import xls.modules.zstd.memory.axi;
import xls.modules.zstd.csr_config;


struct AxiCsrAccessorState<ID_W: u32, ADDR_W:u32, DATA_W:u32, REGS_N: u32> {
    w_id: uN[ID_W],
    w_addr: uN[ADDR_W],
    r_id: uN[ID_W],
    r_addr: uN[ADDR_W],
}

pub proc AxiCsrAccessor<
    ID_W: u32, ADDR_W: u32, DATA_W: u32, REGS_N: u32,
    DATA_W_DIV8: u32 = { DATA_W / u32:8 },
    LOG2_REGS_N: u32 = { std::clog2(REGS_N) },
    LOG2_DATA_W_DIV8: u32 = { std::clog2(DATA_W / u32:8) },
> {
    type AxiAw = axi::AxiAw<ADDR_W, ID_W>;
    type AxiW = axi::AxiW<DATA_W, DATA_W_DIV8>;
    type AxiB = axi::AxiB<ID_W>;
    type AxiAr = axi::AxiAr<ADDR_W, ID_W>;
    type AxiR = axi::AxiR<DATA_W, ID_W>;

    type RdReq = csr_config::CsrRdReq<LOG2_REGS_N>;
    type RdResp = csr_config::CsrRdResp<LOG2_REGS_N, DATA_W>;
    type WrReq = csr_config::CsrWrReq<LOG2_REGS_N, DATA_W>;
    type WrResp = csr_config::CsrWrResp;

    type State = AxiCsrAccessorState<ID_W, ADDR_W, DATA_W, REGS_N>;
    type Data = uN[DATA_W];
    type RegN = uN[LOG2_REGS_N];

    axi_aw_r: chan<AxiAw> in;
    axi_w_r: chan<AxiW> in;
    axi_b_s: chan<AxiB> out;
    axi_ar_r: chan<AxiAr> in;
    axi_r_s: chan<AxiR> out;

    csr_rd_req_s: chan<RdReq> out;
    csr_rd_resp_r: chan<RdResp> in;
    csr_wr_req_s: chan<WrReq> out;
    csr_wr_resp_r: chan<WrResp> in;

    config (
        axi_aw_r: chan<AxiAw> in,
        axi_w_r: chan<AxiW> in,
        axi_b_s: chan<AxiB> out,
        axi_ar_r: chan<AxiAr> in,
        axi_r_s: chan<AxiR> out,

        csr_rd_req_s: chan<RdReq> out,
        csr_rd_resp_r: chan<RdResp> in,
        csr_wr_req_s: chan<WrReq> out,
        csr_wr_resp_r: chan<WrResp> in,
    ) {
        (
            axi_aw_r, axi_w_r, axi_b_s,
            axi_ar_r, axi_r_s,
            csr_rd_req_s, csr_rd_resp_r,
            csr_wr_req_s, csr_wr_resp_r,
        )
    }

    init {
        zero!<State>()
    }

    next (state: State) {
        let tok_0 = join();
        // write to CSR via AXI
        let (tok_1_1, axi_aw, axi_aw_valid) = recv_non_blocking(tok_0, axi_aw_r, AxiAw {id: state.w_id, addr: state.w_addr, ..zero!<AxiAw>()});
        // validate axi aw
        assert!(!(axi_aw_valid && axi_aw.addr as u32 >= (REGS_N << LOG2_DATA_W_DIV8)), "invalid_aw_addr");
        assert!(!(axi_aw_valid && axi_aw.len != u8:0), "invalid_aw_len");

        let (tok_1_2, axi_w, axi_w_valid) = recv_non_blocking(tok_1_1, axi_w_r, zero!<AxiW>());

        // Send WriteRequest to CSRs
        let data_w = if axi_w_valid {
            trace_fmt!("[CSR ACCESSOR] received csr write at {:#x}", axi_w);

            let (w_data, _, _) = for (i, (w_data, strb, mask)): (u32, (uN[DATA_W], uN[DATA_W_DIV8], uN[DATA_W])) in range(u32:0, DATA_W_DIV8) {
                let w_data = if axi_w.strb as u1 {
                    w_data | (axi_w.data & mask)
                } else {
                    w_data
                };
                (
                    w_data,
                    strb >> u32:1,
                    mask << u32:8,
                )
            }((uN[DATA_W]:0, axi_w.strb, uN[DATA_W]:0xFF));
            w_data
        } else {
            uN[DATA_W]:0
        };

        let wr_req = WrReq {
            csr: (axi_aw.addr >> LOG2_DATA_W_DIV8) as uN[LOG2_REGS_N],
            value: data_w
        };

        let tok_1_3 = send_if(tok_1_2, csr_wr_req_s, axi_w_valid, wr_req);

        let (tok_2_1, csr_wr_resp, csr_wr_resp_valid) = recv_non_blocking(tok_0, csr_wr_resp_r, zero!<WrResp>());
        let axi_write_resp = AxiB {
            resp: axi::AxiWriteResp::OKAY,
            id: axi_aw.id,
        };
        let tok_2_2 = send_if(tok_2_1, axi_b_s, csr_wr_resp_valid, axi_write_resp);


        // Send ReadRequest to CSRs
        let (tok_3_1, axi_ar, axi_ar_valid) = recv_non_blocking(tok_0, axi_ar_r, AxiAr {id: state.r_id, addr: state.r_addr, ..zero!<AxiAr>()});
        // validate ar bundle
        assert!(!(axi_ar_valid && axi_ar.addr as u32 >= (REGS_N << LOG2_DATA_W_DIV8)), "invalid_ar_addr");
        assert!(!(axi_ar_valid && axi_ar.len != u8:0), "invalid_ar_len");
        let rd_req = RdReq {
            csr: (axi_ar.addr >> LOG2_DATA_W_DIV8) as uN[LOG2_REGS_N],
        };
        let tok_3_2 = send_if(tok_3_1, csr_rd_req_s, axi_ar_valid, rd_req);

        let (tok_4_1, csr_rd_resp, csr_rd_resp_valid) = recv_non_blocking(tok_0, csr_rd_resp_r, zero!<RdResp>());

        let axi_read_resp = AxiR {
            id: axi_ar.id,
            data: csr_rd_resp.value,
            resp: axi::AxiReadResp::OKAY,
            last: true,
        };
        let tok_4_2 = send_if(tok_4_1, axi_r_s, csr_rd_resp_valid, axi_read_resp);

        State {
            w_id: axi_aw.id,
            w_addr: axi_aw.addr,
            r_id: axi_ar.id,
            r_addr: axi_ar.addr,
        }
    }
}

const INST_ID_W = u32:4;
const INST_DATA_W = u32:32;
const INST_ADDR_W = u32:16;
const INST_REGS_N = u32:16;
const INST_DATA_W_DIV8 = INST_DATA_W / u32:8;
const INST_LOG2_REGS_N = std::clog2(INST_REGS_N);

proc AxiCsrAccessorInst {
    type InstAxiAw = axi::AxiAw<INST_ADDR_W, INST_ID_W>;
    type InstAxiW = axi::AxiW<INST_DATA_W, INST_DATA_W_DIV8>;
    type InstAxiB = axi::AxiB<INST_ID_W>;
    type InstAxiAr = axi::AxiAr<INST_ADDR_W, INST_ID_W>;
    type InstAxiR = axi::AxiR<INST_DATA_W, INST_ID_W>;

    type InstCsrRdReq = csr_config::CsrRdReq<INST_LOG2_REGS_N>;
    type InstCsrRdResp = csr_config::CsrRdResp<INST_LOG2_REGS_N, INST_DATA_W>;
    type InstCsrWrReq = csr_config::CsrWrReq<INST_LOG2_REGS_N, INST_DATA_W>;
    type InstCsrWrResp = csr_config::CsrWrResp;
    type InstCsrChange = csr_config::CsrChange<INST_LOG2_REGS_N>;

    config(
        axi_aw_r: chan<InstAxiAw> in,
        axi_w_r: chan<InstAxiW> in,
        axi_b_s: chan<InstAxiB> out,
        axi_ar_r: chan<InstAxiAr> in,
        axi_r_s: chan<InstAxiR> out,


        csr_rd_req_s: chan<InstCsrRdReq> out,
        csr_rd_resp_r: chan<InstCsrRdResp> in,
        csr_wr_req_s: chan<InstCsrWrReq> out,
        csr_wr_resp_r: chan<InstCsrWrResp> in,
    ) {
        spawn AxiCsrAccessor<INST_ID_W, INST_ADDR_W, INST_DATA_W, INST_REGS_N> (
            axi_aw_r, axi_w_r, axi_b_s,
            axi_ar_r, axi_r_s,
            csr_rd_req_s, csr_rd_resp_r,
            csr_wr_req_s, csr_wr_resp_r,
        );
    }

    init { }

    next (state: ()) { }
}

const TEST_ID_W = u32:4;
const TEST_DATA_W = u32:32;
const TEST_ADDR_W = u32:16;
const TEST_REGS_N = u32:4;
const TEST_DATA_W_DIV8 = TEST_DATA_W / u32:8;
const TEST_LOG2_REGS_N = std::clog2(TEST_REGS_N);
const TEST_LOG2_DATA_W_DIV8 = std::clog2(TEST_DATA_W_DIV8);

type TestCsr = uN[TEST_LOG2_REGS_N];
type TestValue = uN[TEST_DATA_W];

struct TestData {
    csr: uN[TEST_LOG2_REGS_N],
    value: uN[TEST_DATA_W],
}

const TEST_DATA = TestData[20]:[
    TestData{ csr: TestCsr:0, value: TestValue:0xca32_9f4a },
    TestData{ csr: TestCsr:1, value: TestValue:0x0fb3_fa42 },
    TestData{ csr: TestCsr:2, value: TestValue:0xe7ee_da41 },
    TestData{ csr: TestCsr:3, value: TestValue:0xef51_f98c },
    TestData{ csr: TestCsr:0, value: TestValue:0x97a3_a2d2 },
    TestData{ csr: TestCsr:0, value: TestValue:0xea06_e94b },
    TestData{ csr: TestCsr:1, value: TestValue:0x5fac_17ce },
    TestData{ csr: TestCsr:3, value: TestValue:0xf9d8_9938 },
    TestData{ csr: TestCsr:2, value: TestValue:0xc262_2d2e },
    TestData{ csr: TestCsr:2, value: TestValue:0xb4dd_424e },
    TestData{ csr: TestCsr:1, value: TestValue:0x01f9_b9e4 },
    TestData{ csr: TestCsr:1, value: TestValue:0x3020_6eec },
    TestData{ csr: TestCsr:3, value: TestValue:0x3124_87b5 },
    TestData{ csr: TestCsr:0, value: TestValue:0x0a49_f5e3 },
    TestData{ csr: TestCsr:2, value: TestValue:0xde3b_5d0f },
    TestData{ csr: TestCsr:3, value: TestValue:0x5948_c1b3 },
    TestData{ csr: TestCsr:0, value: TestValue:0xa26d_851f },
    TestData{ csr: TestCsr:3, value: TestValue:0x3fa9_59c0 },
    TestData{ csr: TestCsr:1, value: TestValue:0x4efd_dd09 },
    TestData{ csr: TestCsr:1, value: TestValue:0x6d75_058a },
];

#[test_proc]
proc AxiCsrAccessorTest {
    type TestAxiAw = axi::AxiAw<TEST_ADDR_W, TEST_ID_W>;
    type TestAxiW = axi::AxiW<TEST_DATA_W, TEST_DATA_W_DIV8>;
    type TestAxiB = axi::AxiB<TEST_ID_W>;
    type TestAxiAr = axi::AxiAr<TEST_ADDR_W, TEST_ID_W>;
    type TestAxiR = axi::AxiR<TEST_DATA_W, TEST_ID_W>;


    type TestCsrRdReq = csr_config::CsrRdReq<TEST_LOG2_REGS_N>;
    type TestCsrRdResp = csr_config::CsrRdResp<TEST_LOG2_REGS_N, TEST_DATA_W>;
    type TestCsrWrReq = csr_config::CsrWrReq<TEST_LOG2_REGS_N, TEST_DATA_W>;
    type TestCsrWrResp = csr_config::CsrWrResp;
    type TestCsrChange = csr_config::CsrChange<TEST_LOG2_REGS_N>;

    terminator: chan<bool> out;

    axi_aw_s: chan<TestAxiAw> out;
    axi_w_s: chan<TestAxiW> out;
    axi_b_r: chan<TestAxiB> in;
    axi_ar_s: chan<TestAxiAr> out;
    axi_r_r: chan<TestAxiR> in;

    csr_rd_req_r: chan<TestCsrRdReq> in;
    csr_rd_resp_s: chan<TestCsrRdResp> out;
    csr_wr_req_r: chan<TestCsrWrReq> in;
    csr_wr_resp_s: chan<TestCsrWrResp> out;

    config (terminator: chan<bool> out) {
        let (axi_aw_s, axi_aw_r) = chan<TestAxiAw>("axi_aw");
        let (axi_w_s, axi_w_r) = chan<TestAxiW>("axi_w");
        let (axi_b_s, axi_b_r) = chan<TestAxiB>("axi_b");
        let (axi_ar_s, axi_ar_r) = chan<TestAxiAr>("axi_ar");
        let (axi_r_s, axi_r_r) = chan<TestAxiR>("axi_r");

        let (csr_rd_req_s, csr_rd_req_r) = chan<TestCsrRdReq>("csr_rd_req");
        let (csr_rd_resp_s, csr_rd_resp_r) = chan<TestCsrRdResp>("csr_rd_resp");

        let (csr_wr_req_s, csr_wr_req_r) = chan<TestCsrWrReq>("csr_wr_req");
        let (csr_wr_resp_s, csr_wr_resp_r) = chan<TestCsrWrResp>("csr_wr_resp");

        spawn AxiCsrAccessor<TEST_ID_W, TEST_ADDR_W, TEST_DATA_W, TEST_REGS_N> (
            axi_aw_r, axi_w_r, axi_b_s,
            axi_ar_r, axi_r_s,
            csr_rd_req_s, csr_rd_resp_r,
            csr_wr_req_s, csr_wr_resp_r,
        );

        (
            terminator,
            axi_aw_s, axi_w_s, axi_b_r,
            axi_ar_s, axi_r_r,
            csr_rd_req_r, csr_rd_resp_s,
            csr_wr_req_r, csr_wr_resp_s,
        )
    }

    init { }

    next (state: ()) {
        // test writing via AXI
        let tok = for ((i, test_data), tok): ((u32, TestData), token) in enumerate(TEST_DATA) {
            // write CSR via AXI
            let axi_aw = TestAxiAw {
                id: i as uN[TEST_ID_W],
                addr: (test_data.csr as uN[TEST_ADDR_W]) << TEST_LOG2_DATA_W_DIV8,
                size: axi::AxiAxSize::MAX_4B_TRANSFER,
                len: u8:0,
                burst: axi::AxiAxBurst::FIXED,
            };
            let tok = send(tok, axi_aw_s, axi_aw);
            trace_fmt!("Sent #{} AXI AW: {:#x}", i + u32:1, axi_aw);

            let axi_w = TestAxiW {
                data: test_data.value,
                strb: !uN[TEST_DATA_W_DIV8]:0,
                last: true,
            };
            let tok = send(tok, axi_w_s, axi_w);
            trace_fmt!("Sent #{} AXI W: {:#x}", i + u32:1, axi_w);

            let expected_wr_req = TestCsrWrReq {
                csr: test_data.csr,
                value: test_data.value
            };
            let (tok, wr_req) = recv(tok, csr_wr_req_r);
            trace_fmt!("Received #{} CSR WriteRequest: {:#x}", i + u32:1, wr_req);
            assert_eq(expected_wr_req, wr_req);

            let tok = send(tok, csr_wr_resp_s, TestCsrWrResp{});
            trace_fmt!("Sent #{} CsrWrResp", i + u32:1);
            let (tok, axi_b) = recv(tok, axi_b_r);
            trace_fmt!("Received #{} AXI B: {:#x}", i + u32:1, axi_b);
            let expected_axi_resp = TestAxiB{
                resp: axi::AxiWriteResp::OKAY,
                id: i as uN[TEST_ID_W],
            };
            assert_eq(expected_axi_resp, axi_b);

            // read CSRs via AXI
            let axi_ar = TestAxiAr {
                id: i as uN[TEST_ID_W],
                addr: (test_data.csr as uN[TEST_ADDR_W]) << TEST_LOG2_DATA_W_DIV8,
                len: u8:0,
                ..zero!<TestAxiAr>()
            };
            let tok = send(tok, axi_ar_s, axi_ar);
            trace_fmt!("Sent #{} AXI AR: {:#x}", i + u32:1, axi_ar);

            let expected_rd_req = TestCsrRdReq {
                csr: test_data.csr,
            };
            let (tok, rd_req) = recv(tok, csr_rd_req_r);
            trace_fmt!("Received #{} CSR ReadRequest: {:#x}", i + u32:1, rd_req);
            assert_eq(expected_rd_req, rd_req);
            let rd_resp = TestCsrRdResp {
                csr: test_data.csr,
                value: test_data.value
            };
            let tok = send(tok, csr_rd_resp_s, rd_resp);
            trace_fmt!("Sent #{} CsrRdResp: {:#x}", i + u32:1, rd_resp);

            let (tok, axi_r) = recv(tok, axi_r_r);
            trace_fmt!("Received #{} AXI R: {:#x}", i + u32:1, axi_r);
            let expected_axi_rd_resp = TestAxiR{
                id: i as uN[TEST_ID_W],
                data: test_data.value,
                resp: axi::AxiReadResp::OKAY,
                last: true,
            };
            assert_eq(expected_axi_rd_resp, axi_r);

            tok
        }(join());

        send(tok, terminator, true);
    }
}
