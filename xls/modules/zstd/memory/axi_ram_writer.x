// Copyright 2025 The XLS Authors
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

import std;

import xls.modules.zstd.math;
import xls.modules.zstd.memory.axi;
import xls.modules.zstd.memory.mem_writer;
import xls.examples.ram;

struct AxiRamWriterSync<ID_W: u32> {
    id: uN[ID_W],
    error_code: axi::AxiWriteResp
}

struct AxiRamWriterRequesterInternalConf<RAM_ADDR_W: u32, ID_W: u32> {
    addr: uN[RAM_ADDR_W],
    burst: axi::AxiAxBurst,
    transaction_size: u8,
    id: uN[ID_W],
    addr_overflow: bool,
}

struct AxiRamWriterRequesterInternalState<RAM_ADDR_W: u32, ID_W: u32> {
    active: bool,
    conf: AxiRamWriterRequesterInternalConf<RAM_ADDR_W, ID_W>,
}

proc AxiRamWriterRequesterInternal<
    ADDR_W: u32, DATA_W: u32, ID_W: u32, RAM_SIZE: u32,
    RAM_ADDR_W: u32 = {std::clog2(RAM_SIZE)},
    NUM_PARTITIONS: u32 = { DATA_W / u32:8 },
> {
    type Conf = AxiRamWriterRequesterInternalConf<RAM_ADDR_W, ID_W>;
    type State = AxiRamWriterRequesterInternalState<RAM_ADDR_W, ID_W>;
    type AxiAw = axi::AxiAw<ADDR_W, ID_W>;
    type AxiW = axi::AxiW<DATA_W, NUM_PARTITIONS>;
    type WriteReq = ram::WriteReq<RAM_ADDR_W, DATA_W, NUM_PARTITIONS>;
    type WriteResp = ram::WriteResp<DATA_W>;
    type Sync = AxiRamWriterSync<ID_W>;

    conf_r: chan<Conf> in;
    sync_s: chan<Sync> out;

    axi_w_r: chan<AxiW> in;
    wr_req_s: chan<WriteReq> out;
    wr_resp_r: chan<WriteResp> in;

    init { zero!<State>()}

    config(
        conf_r: chan<Conf> in,
        sync_s: chan<Sync> out,
        axi_w_r: chan<AxiW> in,
        wr_req_s: chan<WriteReq> out,
        wr_resp_r: chan<WriteResp> in,
    ) {
        (conf_r, sync_s, axi_w_r, wr_req_s, wr_resp_r)
    }

    next(state: State) {
        let tok = join();

        if !state.active {
            let (tok, conf) = recv(tok, conf_r);
            if conf.addr_overflow {
                trace_fmt!("Overflow error: {:#x}", conf);
                let tok = send(tok, sync_s, Sync { id: state.conf.id, error_code: axi::AxiWriteResp::DECERR });
                State { active: false, conf: zero!<Conf>() }
            } else {
                trace_fmt!("Receiving new burst, conf: {:#x}", conf);
                State { active: true, conf }
            }
        } else {
            let (tok, axi_w) = recv(tok, axi_w_r);
            trace_fmt!("Received data packet: {:#x}, writing to {:#x}", axi_w, state.conf.addr);

            let (new_addr, supported_mode) = if state.conf.burst == axi::AxiAxBurst::INCR {
                (state.conf.addr + uN[RAM_ADDR_W]:1, true)
            } else if state.conf.burst == axi::AxiAxBurst::FIXED {
                (state.conf.addr, true)
            } else {
                (state.conf.addr, false)
            };

            if supported_mode {
                let tok = send(tok, wr_req_s, WriteReq {
                    addr: state.conf.addr,
                    data: axi_w.data,
                    mask: axi_w.strb,
                });
                let (tok, _) = recv(tok, wr_resp_r); // synchronize burst packets with Ram

                if axi_w.last {
                    let tok = send(tok, sync_s, Sync { id: state.conf.id, error_code: axi::AxiWriteResp::OKAY });
                    State { active: false, conf: zero!<Conf>() }
                } else {
                    State {
                        active: true,
                        conf: Conf { addr: new_addr, ..state.conf }
                    }
                }
            } else {
                trace_fmt!("Unsupported burst type error: {:#x}", state.conf);
                let tok = send(tok, sync_s, Sync { id: state.conf.id, error_code: axi::AxiWriteResp::UNSUPPORTED });
                State { active: false, conf: zero!<Conf>() }
            }
        }
    }
}

proc AxiRamWriterRequester<
    ADDR_W: u32, DATA_W: u32, ID_W: u32, RAM_SIZE: u32,
    RAM_ADDR_W: u32 = {std::clog2(RAM_SIZE)},
    NUM_PARTITIONS: u32 = { DATA_W / u32:8 },
    DATA_W_BYTES: u32 = {DATA_W / u32:8 }
> {
    type AxiAw = axi::AxiAw<ADDR_W, ID_W>;
    type AxiW = axi::AxiW<DATA_W, NUM_PARTITIONS>;
    type AxiB = axi::AxiB<ID_W>;
    type WriteReq = ram::WriteReq<RAM_ADDR_W, DATA_W, NUM_PARTITIONS>;
    type WriteResp = ram::WriteResp<DATA_W>;
    type Conf = AxiRamWriterRequesterInternalConf<RAM_ADDR_W, ID_W>;
    type Sync = AxiRamWriterSync<ID_W>;
    type Addr = uN[ADDR_W];
    const CONF_FIFO_DEPTH=u32:1;

    axi_aw_r: chan<AxiAw> in;
    axi_w_r: chan<AxiW> in;
    axi_b_s: chan<AxiB> out;
    wr_req_s: chan<WriteReq> out;
    conf_s: chan<Conf> out;
    sync_s: chan<Sync> out;

    init { }

    config(
        axi_aw_r: chan<AxiAw> in,
        axi_w_r: chan<AxiW> in,
        axi_b_s: chan<AxiB> out,
        wr_req_s: chan<WriteReq> out,
        wr_resp_r: chan<WriteResp> in,
        sync_s: chan<Sync> out,
    ) {
        let (conf_s, conf_r) = chan<Conf, CONF_FIFO_DEPTH>("conf");

        spawn AxiRamWriterRequesterInternal<
            ADDR_W, DATA_W, ID_W, RAM_SIZE,
            RAM_ADDR_W, NUM_PARTITIONS
        > (
            conf_r, sync_s, axi_w_r, wr_req_s, wr_resp_r
        );

        (axi_aw_r, axi_w_r, axi_b_s, wr_req_s, conf_s, sync_s)
    }

    next(state: ()) {
        let (tok, aw_bundle) = recv(join(), axi_aw_r);
        trace_fmt!("Received AW bundle: {:#x}", aw_bundle);
        // convert received address to ram cell address
        let addr = (aw_bundle.addr / DATA_W_BYTES);
        let tok = send(tok, conf_s, Conf {
            addr: addr as uN[RAM_ADDR_W],
            burst: aw_bundle.burst,
            transaction_size: aw_bundle.len,
            id: aw_bundle.id,
            addr_overflow: addr > std::unsigned_max_value<RAM_ADDR_W>() as uN[ADDR_W] // address invalid condition
        });
    }
}

proc AxiRamWriterResponder<
    ADDR_W: u32, DATA_W: u32, ID_W: u32, RAM_SIZE: u32,
    RAM_ADDR_W: u32 = {std::clog2(RAM_SIZE)},
    NUM_PARTITIONS: u32 = { DATA_W / u32:8 },
> {
    type AxiB = axi::AxiB<ID_W>;
    type Sync = AxiRamWriterSync<ID_W>;

    axi_b_s: chan<AxiB> out;
    sync_r: chan<Sync> in;

    init { }

    config(
        axi_b_s: chan<AxiB> out,
        sync_r: chan<Sync> in,
    ) {
        (axi_b_s, sync_r)
    }

    next(state: ()) {
        let (tok, sync) = recv(join(), sync_r);
        let axi_response = AxiB {
            id: sync.id,
            resp: sync.error_code
        };

        trace_fmt!("AXI response {:#x}", axi_response);
        send(tok, axi_b_s, axi_response);
    }
}

const SYNC_FIFO_DEPTH=u32:1;
pub proc AxiRamWriter<
    ADDR_W: u32, DATA_W: u32,ID_W: u32, RAM_SIZE: u32,
    RAM_ADDR_W: u32 = {std::clog2(RAM_SIZE)},
    NUM_PARTITIONS: u32 = { DATA_W / u32:8 },
> {
    type AxiAw = axi::AxiAw<ADDR_W, ID_W>;
    type AxiW = axi::AxiW<DATA_W, NUM_PARTITIONS>;
    type AxiB = axi::AxiB<ID_W>;
    type WriteReq = ram::WriteReq<RAM_ADDR_W, DATA_W, NUM_PARTITIONS>;
    type WriteResp = ram::WriteResp<DATA_W>;
    type Sync = AxiRamWriterSync<ID_W>;

    init { }

    config(
        // AXI interface
        axi_aw_r: chan<AxiAw> in,
        axi_w_r: chan<AxiW> in,
        axi_b_s: chan<AxiB> out,
        // RAM interface
        wr_req_s: chan<WriteReq> out,
        wr_resp_r: chan<WriteResp> in,
    ) {
        let (sync_s, sync_r) = chan<Sync, SYNC_FIFO_DEPTH>("sync");

        spawn AxiRamWriterRequester<
            ADDR_W, DATA_W, ID_W, RAM_SIZE,
            RAM_ADDR_W, NUM_PARTITIONS,
        >(axi_aw_r, axi_w_r, axi_b_s, wr_req_s, wr_resp_r, sync_s);

        spawn AxiRamWriterResponder<
            ADDR_W, DATA_W, ID_W, RAM_SIZE,
            RAM_ADDR_W, NUM_PARTITIONS,
        >(axi_b_s, sync_r);
    }

    next(state: ()) { }
}

const INST_ADDR_W = u32:32;
const INST_DATA_W = u32:32;
const INST_ID_W = u32:8;
const INST_RAM_SIZE = u32:128;
const INST_RAM_ADDR_W: u32 = {std::clog2(INST_RAM_SIZE)};
const INST_NUM_PARTITIONS = INST_DATA_W / u32:8;

proc AxiRamWriterInst{
    type AxiAw = axi::AxiAw<INST_ADDR_W, INST_ID_W>;
    type AxiW = axi::AxiW<INST_DATA_W, INST_NUM_PARTITIONS>;
    type AxiB = axi::AxiB<INST_ID_W>;

    type WriteReq = ram::WriteReq<INST_RAM_ADDR_W, INST_DATA_W, INST_NUM_PARTITIONS>;
    type WriteResp = ram::WriteResp<INST_DATA_W>;

    init { }

    config(
        // AXI interface
        axi_aw_r: chan<AxiAw> in,
        axi_w_r: chan<AxiW> in,
        axi_b_s: chan<AxiB> out,
        // RAM interface
        wr_req_s: chan<WriteReq> out,
        wr_resp_r: chan<WriteResp> in,
    ) {
        spawn AxiRamWriter<
            INST_ADDR_W, INST_DATA_W, INST_ID_W, INST_RAM_SIZE,
            INST_RAM_ADDR_W, INST_NUM_PARTITIONS,
        > (axi_aw_r, axi_w_r, axi_b_s, wr_req_s, wr_resp_r);
    }

    next(state: ()) { }
}

const TEST_ADDR_W = u32:32;
const TEST_DATA_W = u32:32;
const TEST_ID_W = u32:8;
const TEST_RAM_SIZE = u32:128;
const TEST_RAM_ADDR_W: u32 = {std::clog2(INST_RAM_SIZE)};
const TEST_RAM_WORD = u32:8;
const TEST_NUM_PARTITIONS = TEST_DATA_W / u32:8;
const TEST_RAM_REQ_MASK_ALL = std::unsigned_max_value<TEST_NUM_PARTITIONS>();

type TestAxiAw = axi::AxiAw<TEST_ADDR_W, TEST_ID_W>;
type TestAxiW = axi::AxiW<TEST_DATA_W, TEST_NUM_PARTITIONS>;
type TestAxiB = axi::AxiB<TEST_ID_W>;
type TestReadReq = ram::ReadReq<TEST_RAM_ADDR_W, TEST_NUM_PARTITIONS>;
type TestReadResp = ram::ReadResp<TEST_DATA_W>;
type TestWriteReq = ram::WriteReq<TEST_RAM_ADDR_W, TEST_DATA_W, TEST_NUM_PARTITIONS>;
type TestWriteResp = ram::WriteResp;
type TestAxiSize   = axi::AxiAxSize;
type TestAxiBurst  = axi::AxiAxBurst;
type TestAxiId   = uN[TEST_ID_W];
type TestAxiAddr   = uN[TEST_ADDR_W];
type TestDataBits = uN[TEST_DATA_W];
type TestStrobe = uN[TEST_NUM_PARTITIONS];
type TestRamAddr = bits[TEST_RAM_ADDR_W];


const ZERO_AXI_AW = zero!<TestAxiAw>();

#[test_proc]
proc AxiRamWriterTest {
    type RamAddr = uN[TEST_RAM_ADDR_W];

    terminator: chan<bool> out;
    axi_aw_s: chan<TestAxiAw> out;
    axi_w_s: chan<TestAxiW> out;
    axi_b_r: chan<TestAxiB> in;
    rd_req_s: chan<TestReadReq> out;
    rd_resp_r: chan<TestReadResp> in;

    init {}

    config(
        terminator: chan<bool> out,
    ) {
        let (rd_req_s, rd_req_r) = chan<TestReadReq>("rd_req");
        let (rd_resp_s, rd_resp_r) = chan<TestReadResp>("rd_resp");
        let (wr_req_s, wr_req_r) = chan<TestWriteReq>("wr_req");
        let (wr_resp_s, wr_resp_r) = chan<TestWriteResp>("wr_resp");

        spawn ram::RamModel<TEST_DATA_W, TEST_RAM_SIZE, TEST_RAM_WORD> (
            rd_req_r, rd_resp_s, wr_req_r, wr_resp_s
        );

        let (axi_aw_s, axi_aw_r) = chan<TestAxiAw>("axi_aw");
        let (axi_w_s, axi_w_r) = chan<TestAxiW>("axi_w");
        let (axi_b_s, axi_b_r) = chan<TestAxiB>("axi_b");

        spawn AxiRamWriter<
            TEST_ADDR_W, TEST_DATA_W, TEST_ID_W, TEST_RAM_SIZE,
            TEST_RAM_ADDR_W, TEST_NUM_PARTITIONS
        > (axi_aw_r, axi_w_r, axi_b_s, wr_req_s, wr_resp_r);
        (
            terminator, axi_aw_s, axi_w_s, axi_b_r, rd_req_s, rd_resp_r,
        )
    }
    next(state: ()) {
        let tok =
        for((_, (axi_aw, n_axi_w, n_expected, n_addresses)), tok): ((u32, (TestAxiAw, TestAxiW[3], TestDataBits[3], RamAddr[3])), token) in enumerate(
            [
                (
                    TestAxiAw {
                        id: TestAxiId:7,
                        addr: TestAxiAddr:0x4,
                        size: TestAxiSize::MAX_4B_TRANSFER,
                        len: u8:2,
                        burst: TestAxiBurst::INCR
                    },
                    [
                        TestAxiW {
                            data: TestDataBits:0xBEEF,
                            strb: u4:0b1111,
                            last: false,
                        },
                        TestAxiW {
                            data: TestDataBits:0x89ABCDEF,
                            strb: u4: 0b1111,
                            last: false
                        },
                        TestAxiW {
                            data: TestDataBits:0xBADB055,
                            strb: u4: 0b1111,
                            last: true
                        },
                    ],
                    [
                        TestDataBits:0xBEEF,
                        TestDataBits:0x89ABCDEF,
                        TestDataBits: 0xBADB055
                    ],
                    [
                        RamAddr:0x1, RamAddr:0x2, RamAddr:0x3
                    ]
                ),
                (
                    TestAxiAw {
                        id: TestAxiId:7,
                        addr: TestAxiAddr:0x4,
                        size: TestAxiSize::MAX_4B_TRANSFER,
                        len: u8:2,
                        burst: TestAxiBurst::FIXED
                    },
                    [
                        TestAxiW {
                            data: TestDataBits:0xCC,
                            strb: u4:0b0001,
                            last: false,
                        },
                        TestAxiW {
                            data: TestDataBits:0xBEEF0000,
                            strb: u4:0b1100,
                            last: false,
                        },
                        TestAxiW {
                            data: TestDataBits:0x88888888,
                            strb: u4:0b0001,
                            last: true,
                        },
                    ],
                    [
                        TestDataBits:0xBEEFBE88,
                        TestDataBits:0xBEEFBE88,
                        TestDataBits:0xBEEFBE88,
                    ],
                    [
                        RamAddr: 0x1, RamAddr:0x1, RamAddr:0x1
                    ]
                )
            ]
        ) {
            let tok = send(tok, axi_aw_s, axi_aw);
            let tok =
            for ((_, axi_w), tok) in enumerate(n_axi_w) {
                let tok = send(tok, axi_w_s, axi_w);
                tok
            }(tok);

            let (tok, resp) = recv(tok, axi_b_r);
            assert_eq(
                resp, TestAxiB {
                    resp: axi::AxiWriteResp::OKAY,
                    id: axi_aw.id,
                }
            );

            let tok =
            for ((i, expected), tok) in enumerate(n_expected) {
                let tok = send(tok, rd_req_s, TestReadReq { addr: n_addresses[i], mask: TEST_RAM_REQ_MASK_ALL });
                let (tok, resp) = recv(tok, rd_resp_r);
                assert_eq(expected, resp.data);
                tok
            }(tok);

            tok
        }(join());

        send(tok, terminator, true);
    }
}

type TestMemWriterReq = mem_writer::MemWriterReq<TEST_ADDR_W>;
type TestMemWriterData = mem_writer::MemWriterDataPacket<TEST_DATA_W, TEST_ADDR_W>;
type TestMemWriterResp = mem_writer::MemWriterResp;
type TestMemWriterStatus = mem_writer::MemWriterRespStatus;
const TEST_WRITER_ID = u32:1;

#[test_proc]
proc AxiRamWriterMemWriterTest {
    terminator: chan<bool> out;
    rd_req_s: chan<TestReadReq> out;
    rd_resp_r: chan<TestReadResp> in;
    mem_wr_req_s: chan<TestMemWriterReq> out;
    mem_wr_data_s: chan<TestMemWriterData> out;
    mem_wr_resp_r: chan<TestMemWriterResp> in;

    init {}

    config(
        terminator: chan<bool> out,
    ) {
        let (rd_req_s, rd_req_r) = chan<TestReadReq>("rd_req");
        let (rd_resp_s, rd_resp_r) = chan<TestReadResp>("rd_resp");
        let (wr_req_s, wr_req_r) = chan<TestWriteReq>("wr_req");
        let (wr_resp_s, wr_resp_r) = chan<TestWriteResp>("wr_resp");

        let (mem_wr_req_s, mem_wr_req_r) = chan<TestMemWriterReq>("mem_wr_data");
        let (mem_wr_data_s, mem_wr_data_r) = chan<TestMemWriterData>("mem_wr_data");
        let (mem_wr_resp_s, mem_wr_resp_r) = chan<TestMemWriterResp>("mem_wr_data");

        spawn ram::RamModel<TEST_DATA_W, TEST_RAM_SIZE, TEST_RAM_WORD> (
            rd_req_r, rd_resp_s, wr_req_r, wr_resp_s
        );

        let (axi_aw_s, axi_aw_r) = chan<TestAxiAw>("axi_aw");
        let (axi_w_s, axi_w_r) = chan<TestAxiW>("axi_w");
        let (axi_b_s, axi_b_r) = chan<TestAxiB>("axi_b");

        spawn AxiRamWriter<
            TEST_ADDR_W, TEST_DATA_W, TEST_ID_W, TEST_RAM_SIZE,
            TEST_RAM_ADDR_W, TEST_NUM_PARTITIONS
        > (axi_aw_r, axi_w_r, axi_b_s, wr_req_s, wr_resp_r);

        spawn mem_writer::MemWriter<TEST_ADDR_W, TEST_DATA_W, TEST_ADDR_W, TEST_ID_W, TEST_WRITER_ID>(
            mem_wr_req_r, mem_wr_data_r,
            axi_aw_s, axi_w_s, axi_b_r,
            mem_wr_resp_s,
        );

        (
            terminator, rd_req_s, rd_resp_r, mem_wr_req_s, mem_wr_data_s, mem_wr_resp_r
        )
    }
    next(state: ()) {
        let tok =
        for ((_, (req, data, expected)), tok): ((u32, (TestMemWriterReq, TestMemWriterData, u32)), token) in enumerate(
            [
                (
                    TestMemWriterReq { addr: TestAxiAddr:0, length: u32:4 },
                    TestMemWriterData { data: TestDataBits: 0xB16B00B5,  last: true, length: u32: 4},
                    TestDataBits: 0xB16B00B5,
                ),
                (
                    TestMemWriterReq { addr: TestAxiAddr:0, length: u32:3 },
                    TestMemWriterData { data: TestDataBits: 0x888888,  last: true, length: u32: 3},
                    TestDataBits: 0xB1888888,
                ), // test masking
                (
                    TestMemWriterReq { addr: TestAxiAddr:1, length: u32:3 },
                    TestMemWriterData { data: TestDataBits: 0xAAAAAAAA,  last: true, length: u32: 3},
                    TestDataBits: 0xAAAAAA88,
                ), // test unaligned write
                (
                    TestMemWriterReq { addr: TestAxiAddr:4, length: u32:4 },
                    TestMemWriterData { data: TestDataBits: 0xB105F00D,  last: true, length: u32: 4},
                    TestDataBits: 0xB105F00D,
                ),
                (
                    TestMemWriterReq { addr: TestAxiAddr:6, length: u32:2 },
                    TestMemWriterData { data: TestDataBits: 0x0BAD,  last: true, length: u32: 2},
                    TestDataBits: 0xBADF00D,
                ), // test unaligned write
                (
                    TestMemWriterReq { addr: TestAxiAddr:4, length: u32:2 },
                    TestMemWriterData { data: TestDataBits: 0xB055,  last: true, length: u32: 2},
                    TestDataBits: 0x0BADB055,
                ), // test masking
                (
                    TestMemWriterReq { addr: TestAxiAddr:0, length: u32:4 },
                    TestMemWriterData { data: TestDataBits: 0xDEADBEEF,  last: true, length: u32: 4},
                    TestDataBits: 0xDEADBEEF,
                ),
                (
                    TestMemWriterReq { addr: TestAxiAddr:0x3, length: u32:1 },
                    TestMemWriterData { data: TestDataBits: 0xBA,  last: true, length: u32: 1},
                    TestDataBits: 0xBAADBEEF,
                ), // test masking

            ]
        ) {
            // write data
            let tok = send(tok, mem_wr_req_s, req);
            let tok = send(tok, mem_wr_data_s, data);
            let (tok, resp) = recv(tok, mem_wr_resp_r);
            assert_eq(resp.status, TestMemWriterStatus::OKAY);

            // read directly from memory cell to see if it's correct
            let cell = req.addr as TestRamAddr / TEST_NUM_PARTITIONS as TestRamAddr;
            let tok = send(tok, rd_req_s, TestReadReq { addr: cell as TestRamAddr, mask: TEST_RAM_REQ_MASK_ALL });
            let (tok, resp) = recv(tok, rd_resp_r);
            assert_eq(expected, resp.data);

            tok
        }(join());

        // test multiple data transactions
        let tok = send(tok, mem_wr_req_s, TestMemWriterReq { addr: TestAxiAddr:0x40, length: u32:16 });
        let tok = send(tok, mem_wr_data_s, TestMemWriterData { data: TestDataBits: 0x12345678,  last: false, length: u32: 4});
        let tok = send(tok, mem_wr_data_s, TestMemWriterData { data: TestDataBits: 0x89ABCDEF,  last: false, length: u32: 4});
        let tok = send(tok, mem_wr_data_s, TestMemWriterData { data: TestDataBits: 0xFEDCBA98,  last: false, length: u32: 4});
        let tok = send(tok, mem_wr_data_s, TestMemWriterData { data: TestDataBits: 0x76543210,  last: true, length: u32: 4});
        let (tok, resp) = recv(tok, mem_wr_resp_r);
        assert_eq(resp.status, TestMemWriterStatus::OKAY);
        // read directly from memory cells to see if it's correct
        let cell = TestRamAddr:0x40 / TEST_NUM_PARTITIONS as TestRamAddr;
        let tok = send(tok, rd_req_s, TestReadReq { addr: cell, mask: TEST_RAM_REQ_MASK_ALL });
        let (tok, resp) = recv(tok, rd_resp_r);
        assert_eq(u32:0x12345678, resp.data);
        let cell = TestRamAddr:0x44 / TEST_NUM_PARTITIONS as TestRamAddr;
        let tok = send(tok, rd_req_s, TestReadReq { addr: cell, mask: TEST_RAM_REQ_MASK_ALL });
        let (tok, resp) = recv(tok, rd_resp_r);
        assert_eq(u32:0x89ABCDEF, resp.data);
        let cell = TestRamAddr:0x48 / TEST_NUM_PARTITIONS as TestRamAddr;
        let tok = send(tok, rd_req_s, TestReadReq { addr: cell, mask: TEST_RAM_REQ_MASK_ALL });
        let (tok, resp) = recv(tok, rd_resp_r);
        assert_eq(u32:0xFEDCBA98, resp.data);
        let cell = TestRamAddr:0x4C / TEST_NUM_PARTITIONS as TestRamAddr;
        let tok = send(tok, rd_req_s, TestReadReq { addr: cell, mask: TEST_RAM_REQ_MASK_ALL });
        let (tok, resp) = recv(tok, rd_resp_r);
        assert_eq(u32:0x76543210, resp.data);

        // test invalid address
        let tok = send(tok, mem_wr_req_s, TestMemWriterReq { addr: TEST_NUM_PARTITIONS as TestAxiAddr * TEST_RAM_SIZE as TestAxiAddr, length: u32:4 });
        let tok = send(tok, mem_wr_data_s, TestMemWriterData { data: TestDataBits: 0x12345678,  last: false, length: u32: 4});
        let (tok, resp) = recv(tok, mem_wr_resp_r);

        assert_eq(resp.status, TestMemWriterStatus::ERROR);
        send(tok, terminator, true);
    }
}
