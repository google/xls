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

pub struct CsrRdReq<LOG2_REGS_N: u32> {
    csr: uN[LOG2_REGS_N],
}

pub struct CsrRdResp<LOG2_REGS_N: u32, DATA_W: u32> {
    csr: uN[LOG2_REGS_N],
    value: uN[DATA_W],
}

pub struct CsrWrReq<LOG2_REGS_N: u32, DATA_W: u32> {
    csr: uN[LOG2_REGS_N],
    value: uN[DATA_W],
}

pub struct CsrWrResp { }

pub struct CsrChange<LOG2_REGS_N: u32> {
    csr: uN[LOG2_REGS_N],
}

struct CsrConfigState<ID_W: u32, ADDR_W:u32, DATA_W:u32, REGS_N: u32> {
    register_file: uN[DATA_W][REGS_N],
}

pub proc CsrConfig<
    ID_W: u32, ADDR_W: u32, DATA_W: u32, REGS_N: u32,
    //REGS_INIT: u64[64] = {u64[64]:[u64:0, ...]},
    DATA_W_DIV8: u32 = { DATA_W / u32:8 },
    LOG2_REGS_N: u32 = { std::clog2(REGS_N) },
> {

    type RdReq = CsrRdReq<LOG2_REGS_N>;
    type RdResp = CsrRdResp<LOG2_REGS_N, DATA_W>;
    type WrReq = CsrWrReq<LOG2_REGS_N, DATA_W>;
    type WrResp = CsrWrResp;
    type Change = CsrChange<LOG2_REGS_N>;

    type State = CsrConfigState<ID_W, ADDR_W, DATA_W, REGS_N>;
    type Data = uN[DATA_W];
    type RegN = uN[LOG2_REGS_N];

    ext_csr_rd_req_r: chan<RdReq> in;
    ext_csr_rd_resp_s: chan<RdResp> out;
    ext_csr_wr_req_r: chan<WrReq> in;
    ext_csr_wr_resp_s: chan<WrResp> out;

    csr_rd_req_r: chan<RdReq> in;
    csr_rd_resp_s: chan<RdResp> out;
    csr_wr_req_r: chan<WrReq> in;
    csr_wr_resp_s: chan<WrResp> out;

    csr_change_s: chan<Change> out;

    config (
        ext_csr_rd_req_r: chan<RdReq> in,
        ext_csr_rd_resp_s: chan<RdResp> out,
        ext_csr_wr_req_r: chan<WrReq> in,
        ext_csr_wr_resp_s: chan<WrResp> out,

        csr_rd_req_r: chan<RdReq> in,
        csr_rd_resp_s: chan<RdResp> out,
        csr_wr_req_r: chan<WrReq> in,
        csr_wr_resp_s: chan<WrResp> out,
        csr_change_s: chan<Change> out,
    ) {
        (
            ext_csr_rd_req_r, ext_csr_rd_resp_s,
            ext_csr_wr_req_r, ext_csr_wr_resp_s,
            csr_rd_req_r, csr_rd_resp_s,
            csr_wr_req_r, csr_wr_resp_s,
            csr_change_s,
        )
    }

    init {
        zero!<State>()
    }

    next (state: State) {
        let register_file = state.register_file;

        let tok_0 = join();

        // write to CSR
        let (tok_1_1_1, ext_csr_wr_req, ext_csr_wr_req_valid) = recv_non_blocking(tok_0, ext_csr_wr_req_r, zero!<WrReq>());
        let (tok_1_1_2, csr_wr_req, csr_wr_req_valid) = recv_non_blocking(tok_0, csr_wr_req_r, zero!<WrReq>());

        // Mux the Write Requests from External and Internal sources
        // Write requests from external source take precedence before internal writes
        let wr_req = if (ext_csr_wr_req_valid) {
            ext_csr_wr_req
        } else if {csr_wr_req_valid} {
            csr_wr_req
        } else {
            zero!<WrReq>()
        };

        let wr_req_valid = ext_csr_wr_req_valid | csr_wr_req_valid;

        let register_file = if wr_req_valid {
            update(register_file, wr_req.csr as u32, wr_req.value)
        } else {
            register_file
        };

        // Send Write Response
        let tok_1_1 = join(tok_1_1_1, tok_1_1_2);
        let tok_1_2_1 = send_if(tok_1_1, ext_csr_wr_resp_s, ext_csr_wr_req_valid, WrResp {});
        let tok_1_2_2 = send_if(tok_1_1, csr_wr_resp_s, csr_wr_req_valid, WrResp {});

        // Send change notification
        let tok_1_2 = join(tok_1_2_1, tok_1_2_2);
        let tok_1_3 = send_if(tok_1_2, csr_change_s, wr_req_valid, Change { csr: wr_req.csr });


        // Read from CSRs
        let (tok_2_1, ext_csr_rd_req, ext_csr_req_valid) = recv_non_blocking(tok_0, ext_csr_rd_req_r, zero!<RdReq>());

        send_if(tok_2_1, ext_csr_rd_resp_s, ext_csr_req_valid, RdResp {
            csr: ext_csr_rd_req.csr,
            value: register_file[ext_csr_rd_req.csr as u32],
        });

        let (tok_3_1, csr_rd_req, csr_req_valid) = recv_non_blocking(tok_0, csr_rd_req_r, zero!<RdReq>());
        send_if(tok_3_1, csr_rd_resp_s, csr_req_valid, RdResp {
            csr: csr_rd_req.csr,
            value: register_file[csr_rd_req.csr as u32],
        });

        State {
            register_file: register_file,
        }
    }
}

const INST_ID_W = u32:32;
const INST_DATA_W = u32:32;
const INST_ADDR_W = u32:2;
const INST_REGS_N = u32:4;
const INST_DATA_W_DIV8 = INST_DATA_W / u32:8;
const INST_LOG2_REGS_N = std::clog2(INST_REGS_N);

proc CsrConfigInst {
    type InstCsrRdReq = CsrRdReq<INST_LOG2_REGS_N>;
    type InstCsrRdResp = CsrRdResp<INST_LOG2_REGS_N, INST_DATA_W>;
    type InstCsrWrReq = CsrWrReq<INST_LOG2_REGS_N, INST_DATA_W>;
    type InstCsrWrResp = CsrWrResp;
    type InstCsrChange = CsrChange<INST_LOG2_REGS_N>;

    config(
        ext_csr_rd_req_r: chan<InstCsrRdReq> in,
        ext_csr_rd_resp_s: chan<InstCsrRdResp> out,
        ext_csr_wr_req_r: chan<InstCsrWrReq> in,
        ext_csr_wr_resp_s: chan<InstCsrWrResp> out,

        csr_rd_req_r: chan<InstCsrRdReq> in,
        csr_rd_resp_s: chan<InstCsrRdResp> out,
        csr_wr_req_r: chan<InstCsrWrReq> in,
        csr_wr_resp_s: chan<InstCsrWrResp> out,
        csr_change_s: chan<InstCsrChange> out,
    ) {
        spawn CsrConfig<INST_ID_W, INST_ADDR_W, INST_DATA_W, INST_REGS_N> (
            ext_csr_rd_req_r, ext_csr_rd_resp_s,
            ext_csr_wr_req_r, ext_csr_wr_resp_s,
            csr_rd_req_r, csr_rd_resp_s,
            csr_wr_req_r, csr_wr_resp_s,
            csr_change_s,
        );
    }

    init { }

    next (state: ()) { }
}

const TEST_ID_W = u32:32;
const TEST_DATA_W = u32:32;
const TEST_ADDR_W = u32:2;
const TEST_REGS_N = u32:4;
const TEST_DATA_W_DIV8 = TEST_DATA_W / u32:8;
const TEST_LOG2_REGS_N = std::clog2(TEST_REGS_N);

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
proc CsrConfig_test {
    type TestCsrRdReq = CsrRdReq<TEST_LOG2_REGS_N>;
    type TestCsrRdResp = CsrRdResp<TEST_LOG2_REGS_N, TEST_DATA_W>;
    type TestCsrWrReq = CsrWrReq<TEST_LOG2_REGS_N, TEST_DATA_W>;
    type TestCsrWrResp = CsrWrResp;
    type TestCsrChange = CsrChange<TEST_LOG2_REGS_N>;

    terminator: chan<bool> out;

    ext_csr_rd_req_s: chan<TestCsrRdReq> out;
    ext_csr_rd_resp_r: chan<TestCsrRdResp> in;
    ext_csr_wr_req_s: chan<TestCsrWrReq> out;
    ext_csr_wr_resp_r: chan<TestCsrWrResp> in;

    csr_rd_req_s: chan<TestCsrRdReq> out;
    csr_rd_resp_r: chan<TestCsrRdResp> in;
    csr_wr_req_s: chan<TestCsrWrReq> out;
    csr_wr_resp_r: chan<TestCsrWrResp> in;

    csr_change_r: chan<TestCsrChange> in;

    config (terminator: chan<bool> out) {
        let (ext_csr_rd_req_s, ext_csr_rd_req_r) = chan<TestCsrRdReq>("ext_csr_rd_req");
        let (ext_csr_rd_resp_s, ext_csr_rd_resp_r) = chan<TestCsrRdResp>("ext_csr_rd_resp");

        let (ext_csr_wr_req_s, ext_csr_wr_req_r) = chan<TestCsrWrReq>("ext_csr_wr_req");
        let (ext_csr_wr_resp_s, ext_csr_wr_resp_r) = chan<TestCsrWrResp>("ext_csr_wr_resp");

        let (csr_rd_req_s, csr_rd_req_r) = chan<TestCsrRdReq>("csr_rd_req");
        let (csr_rd_resp_s, csr_rd_resp_r) = chan<TestCsrRdResp>("csr_rd_resp");

        let (csr_wr_req_s, csr_wr_req_r) = chan<TestCsrWrReq>("csr_wr_req");
        let (csr_wr_resp_s, csr_wr_resp_r) = chan<TestCsrWrResp>("csr_wr_resp");

        let (csr_change_s, csr_change_r) = chan<TestCsrChange>("csr_change");

        spawn CsrConfig<TEST_ID_W, TEST_ADDR_W, TEST_DATA_W, TEST_REGS_N> (
            ext_csr_rd_req_r, ext_csr_rd_resp_s,
            ext_csr_wr_req_r, ext_csr_wr_resp_s,
            csr_rd_req_r, csr_rd_resp_s,
            csr_wr_req_r, csr_wr_resp_s,
            csr_change_s,
        );

        (
            terminator,
            ext_csr_rd_req_s, ext_csr_rd_resp_r,
            ext_csr_wr_req_s, ext_csr_wr_resp_r,
            csr_rd_req_s, csr_rd_resp_r,
            csr_wr_req_s, csr_wr_resp_r,
            csr_change_r,
        )
    }

    init { }

    next (state: ()) {
        let expected_values = zero!<uN[TEST_DATA_W][TEST_REGS_N]>();

        // Test Writes through external interface
        let (tok, expected_values) = for ((i, test_data), (tok, expected_values)): ((u32, TestData), (token, uN[TEST_DATA_W][TEST_REGS_N])) in enumerate(TEST_DATA) {
            // write CSR via external interface
            let wr_req = TestCsrWrReq {
                csr: test_data.csr,
                value: test_data.value,
            };
            let tok = send(tok, ext_csr_wr_req_s, wr_req);
            trace_fmt!("Sent #{} WrReq through external interface: {:#x}", i + u32:1, wr_req);

            let (tok, wr_resp) = recv(tok, ext_csr_wr_resp_r);
            trace_fmt!("Received #{} WrResp through external interface: {:#x}", i + u32:1, wr_resp);

            // read CSR change
            let (tok, csr_change) = recv(tok, csr_change_r);
            trace_fmt!("Received #{} CSR change {:#x}", i + u32:1, csr_change);

            assert_eq(test_data.csr, csr_change.csr);

            // update expected values
            let expected_values = update(expected_values, test_data.csr as u32, test_data.value);

            let tok = for (test_csr, tok): (u32, token) in u32:0..u32:4 {
                let rd_req = TestCsrRdReq {
                    csr: test_csr as TestCsr,
                };
                let expected_rd_resp = TestCsrRdResp{
                    csr: test_csr as TestCsr,
                    value: expected_values[test_csr as u32]
                };

                // Read CSR via external interface
                let tok = send(tok, ext_csr_rd_req_s, rd_req);
                trace_fmt!("Sent #{} RdReq through external interface: {:#x}", i + u32:1, rd_req);
                let (tok, rd_resp) = recv(tok, ext_csr_rd_resp_r);
                trace_fmt!("Received #{} RdResp through external interface: {:#x}", i + u32:1, rd_resp);
                assert_eq(expected_rd_resp, rd_resp);

                // Read CSR via internal interface
                let tok = send(tok, csr_rd_req_s, rd_req);
                trace_fmt!("Sent #{} RdReq through internal interface: {:#x}", i + u32:1, rd_req);
                let (tok, csr_rd_resp) = recv(tok, csr_rd_resp_r);
                trace_fmt!("Received #{} RdResp through internal interface: {:#x}", i + u32:1, csr_rd_resp);
                assert_eq(expected_rd_resp, csr_rd_resp);
                tok
            }(tok);

            (tok, expected_values)
        }((join(), expected_values));

        // Test writes via internal interface
        let (tok, _) = for ((i, test_data), (tok, expected_values)): ((u32, TestData), (token, uN[TEST_DATA_W][TEST_REGS_N])) in enumerate(TEST_DATA) {
            // write CSR via request channel
            let csr_wr_req = TestCsrWrReq {
                csr: test_data.csr,
                value: test_data.value,
            };
            let tok = send(tok, csr_wr_req_s, csr_wr_req);
            trace_fmt!("Sent #{} WrReq through internal interface: {:#x}", i + u32:1, csr_wr_req);

            let (tok, csr_wr_resp) = recv(tok, csr_wr_resp_r);
            trace_fmt!("Received #{} WrResp through internal interface {:#x}", i + u32:1, csr_wr_resp);

            // read CSR change
            let (tok, csr_change) = recv(tok, csr_change_r);
            trace_fmt!("Received #{} CSR change {:#x}", i + u32:1, csr_change);
            assert_eq(test_data.csr, csr_change.csr);

            // update expected values
            let expected_values = update(expected_values, test_data.csr as u32, test_data.value);

            let tok = for (test_csr, tok): (u32, token) in u32:0..u32:4 {
                let rd_req = TestCsrRdReq {
                    csr: test_csr as TestCsr,
                };
                let expected_rd_resp = TestCsrRdResp{
                    csr: test_csr as TestCsr,
                    value: expected_values[test_csr as u32]
                };

                // Read CSR via external interface
                let tok = send(tok, ext_csr_rd_req_s, rd_req);
                trace_fmt!("Sent #{} RdReq through external interface: {:#x}", i + u32:1, rd_req);
                let (tok, rd_resp) = recv(tok, ext_csr_rd_resp_r);
                trace_fmt!("Received #{} RdResp through external interface: {:#x}", i + u32:1, rd_resp);
                assert_eq(expected_rd_resp, rd_resp);

                // Read CSR via internal interface
                let tok = send(tok, csr_rd_req_s, rd_req);
                trace_fmt!("Sent #{} RdReq through internal interface: {:#x}", i + u32:1, rd_req);
                let (tok, csr_rd_resp) = recv(tok, csr_rd_resp_r);
                trace_fmt!("Received #{} RdResp through internal interface: {:#x}", i + u32:1, csr_rd_resp);
                assert_eq(expected_rd_resp, csr_rd_resp);
                tok
            }(tok);

            (tok, expected_values)
        }((join(), expected_values));

        send(tok, terminator, true);
    }
}
