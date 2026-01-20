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

// This file contains implementation of RleBlockEncoder
//
// The proc is supposed to:
// 1. run a heuristic to determine if a block can be RLE Encoded (RleBlockEncoderSampler)
//    1.1 uniformly sample from the entire block
//    1.2 see if all samples are equal
//    1.3 early exit if not
// 2. run the encoding (check if all symbols are equal along the way) (RleBlockEncoderFullSearch)
// 3. if the encoding was successful return a **single** RLE pair (symbol, length)

import std;

import xls.examples.ram;
import xls.modules.zstd.common;
import xls.modules.zstd.memory.mem_reader;
import xls.modules.zstd.memory.axi;
import xls.modules.zstd.memory.axi_ram_reader;
import xls.modules.zstd.memory.axi_ram_writer;
import xls.modules.zstd.memory.mem_writer;
import xls.modules.zstd.mem_reader_simple_arbiter;

pub struct RleBlockEncoderReq<ADDR_W: u32> {
    addr: uN[ADDR_W],  // start address
    length: uN[ADDR_W] // length of the block
  }

pub enum RleBlockEncoderStatus: u2 {
    OK = 0,
    INCOMPRESSIBLE = 1,
    ERROR = 2
}

pub struct RleBlockEncoderResp<LENGTH_W: u32> {
    symbol: u8,
    length: uN[LENGTH_W],
    status: RleBlockEncoderStatus
}

struct RleBlockEncoderSamplerResp {
    value: u8,
    read_success: bool,
    all_equal: bool,
}

struct RleBlockEncoderInternalConfig<ADDR_W: u32> {
    address: uN[ADDR_W],
    stride: uN[ADDR_W],
    left: u32
}

struct RleBlockEncoderSamplerState<ADDR_W: u32> {
    active: bool,
    previous: u8,
    conf: RleBlockEncoderInternalConfig<ADDR_W>,
    first_read: bool
}

struct RleBlockEncoderFullSearchState<ADDR_W: u32> {
    active: bool,
    check_symbol: u8,
    all_equal: bool,
    all_reads_successful: bool,
    queried_left: u32,
    conf: RleBlockEncoderInternalConfig<ADDR_W>,
    first_read: bool
}


fn entire_word_of_equal_bytes<DATA_W: u32>(word: uN[DATA_W], to_check: u8) -> bool {
    const DATA_W_B = DATA_W as u8 / u8:8;
    let symbol = word as u8;

    unroll_for! (i, eq) : (u8, bool) in u8:0..DATA_W_B {
        let next_byte = (word >> (i as uN[DATA_W] * uN[DATA_W]:8)) as u8;
        if i >= to_check {
            eq
        } else {
            eq && (next_byte == symbol)
        }
    }(true)
}

pub proc RleBlockEncoderFullSearch<ADDR_W: u32, DATA_W:u32, LENGTH_W: u32, MAX_TX_PER_REQ: u32> {
    type MemReaderReq = mem_reader::MemReaderReq<ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<DATA_W, ADDR_W>;
    type MemReaderStatus = mem_reader::MemReaderStatus;

    type State = RleBlockEncoderFullSearchState<ADDR_W>;
    type Resp = RleBlockEncoderSamplerResp;
    type Config = RleBlockEncoderInternalConfig<ADDR_W>;

    mem_rd_req_s: chan<MemReaderReq> out;
    mem_rd_resp_r: chan<MemReaderResp> in;
    req_r: chan<Config> in;
    resp_s: chan<Resp> out;

    config(
        mem_rd_req_s: chan<MemReaderReq> out,
        mem_rd_resp_r: chan<MemReaderResp> in,
        req_r: chan<Config> in,
        resp_s: chan<Resp> out,
    ) {
        (
            mem_rd_req_s, mem_rd_resp_r,
            req_r, resp_s,
        )
    }

    init { zero!<State>() }

    next(state: State) {
        let tok = join();
        let conf = state.conf;

        if !state.active {
            let (tok, conf) = recv(tok, req_r);

            let query_size = if conf.left < MAX_TX_PER_REQ { conf.left } else { MAX_TX_PER_REQ };
            let tok = send(tok, mem_rd_req_s, MemReaderReq {
                addr: conf.address,
                length: query_size
            });

            State {
                active: true,
                all_equal: true,
                conf: Config {
                    left: conf.left,
                    address: conf.address + query_size,
                    ..conf
                },
                queried_left: query_size,
                first_read: true,
                all_reads_successful: true,
                ..zero!<State>()
            }

        } else if state.conf.left == u32:0
        || (state.queried_left == u32:0 && (!state.all_equal || !state.all_reads_successful)) {

            let tok = send(tok, resp_s, Resp {
                value: state.check_symbol,
                read_success: state.all_reads_successful,
                all_equal: state.all_equal
            });

            zero!<State>()
        } else if state.queried_left == u32:0 {
            // reducing read requests mechanism
            // 1. every MAX_TX_PER_REQ tokens make a new read request
            // 2. also check if it should early-abort (notice the above branch)
            let query_size = if conf.left < MAX_TX_PER_REQ { conf.left } else { MAX_TX_PER_REQ };
            let tok = send(tok, mem_rd_req_s, MemReaderReq {
                addr: conf.address,
                length: query_size
            });

            State {
                queried_left: query_size,
                conf: Config {
                    address: conf.address + query_size,
                    ..conf
                },
                ..state
            }
        } else {
            let (tok, resp) = recv(tok, mem_rd_resp_r);
            let check_symbol = if state.first_read {
                resp.data as u8
            } else {
                state.check_symbol
            };
            let all_equal_next = entire_word_of_equal_bytes(resp.data, resp.length as u8) && resp.data as u8 == check_symbol && state.all_equal;

            State {
                active: true,
                all_equal: all_equal_next,
                conf: Config {
                    left: conf.left - resp.length,
                    ..conf
                },
                queried_left: state.queried_left - resp.length,
                check_symbol: check_symbol,
                first_read: false,
                all_reads_successful: resp.status == MemReaderStatus::OKAY && state.all_reads_successful
            }
        }

    }

}


pub proc RleBlockEncoderSampler<ADDR_W: u32, DATA_W: u32, LENGTH_W: u32> {
    type MemReaderReq = mem_reader::MemReaderReq<ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<DATA_W, ADDR_W>;
    type MemReaderStatus = mem_reader::MemReaderStatus;

    type State = RleBlockEncoderSamplerState<ADDR_W>;
    type Resp = RleBlockEncoderSamplerResp;
    type Config = RleBlockEncoderInternalConfig<ADDR_W>;

    mem_rd_req_s: chan<MemReaderReq> out;
    mem_rd_resp_r: chan<MemReaderResp> in;
    req_r: chan<Config> in;
    resp_s: chan<Resp> out;

    config(
        mem_rd_req_s: chan<MemReaderReq> out,
        mem_rd_resp_r: chan<MemReaderResp> in,
        req_r: chan<Config> in,
        resp_s: chan<Resp> out,
    ) {
        (
            mem_rd_req_s, mem_rd_resp_r,
            req_r, resp_s,
        )
    }

    init { zero!<State>() }

    next(state: State) {
        const DATA_W_B = DATA_W / u32:8;

        let tok = join();
        let conf = state.conf;

        if !state.active {
            let (tok,conf) = recv(tok, req_r);
            State {
                active: true,
                conf: Config {
                    address: conf.address,
                    left: conf.left,
                    ..conf
                },
                first_read: true,
                ..zero!<State>()
            }
        } else if state.conf.left == u32:0 {
            let tok = send(tok, resp_s, Resp {
                value: state.previous,
                read_success: true,
                all_equal: true
            });
            zero!<State>()
        } else {
            let tok = send(tok, mem_rd_req_s, MemReaderReq {
                addr: conf.address,
                length: DATA_W_B
            });
            let (tok, mem_resp) = recv(tok, mem_rd_resp_r);
            let equal = mem_resp.data as u8 == state.previous && entire_word_of_equal_bytes(mem_resp.data, DATA_W_B as u8);

            if mem_resp.status == MemReaderStatus::OKAY && (state.first_read || equal) {
                State {
                    active: true,
                    conf: Config {
                        address: conf.address + conf.stride,
                        left: conf.left - u32:1,
                        ..conf
                    },
                    previous: mem_resp.data as u8,
                    first_read: false
                }
            } else {
                let tok = send(tok, resp_s, Resp {
                    value: mem_resp.data as u8,
                    read_success: mem_resp.status == MemReaderStatus::OKAY,
                    all_equal: false
                });

                State {
                    ..zero!<State>()
                }
            }
        }
    }
}

pub proc RleBlockEncoder<ADDR_W: u32, DATA_W: u32, LENGTH_W: u32, SAMPLE_COUNT: u32, MAX_TX_PER_REQ: u32 = {u32:1024}> {
    type Req = RleBlockEncoderReq<ADDR_W>;
    type Resp = RleBlockEncoderResp<LENGTH_W>;
    type Status = RleBlockEncoderStatus;
    type MemReaderReq = mem_reader::MemReaderReq<ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<DATA_W, ADDR_W>;
    type LoopResp = RleBlockEncoderSamplerResp;
    type LoopConfig = RleBlockEncoderInternalConfig<ADDR_W>;

    req: chan<Req> in;
    resp: chan<Resp> out;
    loop_req_s: chan<LoopConfig> out;
    loop_resp_r: chan<LoopResp> in;
    heuristic_req_s: chan<LoopConfig> out;
    heuristic_resp_r: chan<LoopResp> in;

    config(
        req: chan<Req> in,
        resp: chan<Resp> out,
        mem_rd_req_s: chan<MemReaderReq> out,
        mem_rd_resp_r: chan<MemReaderResp> in
    ) {
        let (loop_req_s, loop_req_r) = chan<LoopConfig, u32:1>("loop_req");
        let (loop_resp_s, loop_resp_r) = chan<LoopResp, u32:1>("loop_resp");
        let (heuristic_req_s, heuristic_req_r) = chan<LoopConfig, u32:1>("loop_req");
        let (heuristic_resp_s, heuristic_resp_r) = chan<LoopResp, u32:1>("loop_resp");

        let (n_mem_rd_req_s, n_mem_rd_req_r) = chan<MemReaderReq, u32:1>[2]("n_mem_rd_req");
        let (n_mem_rd_resp_s, n_mem_rd_resp_r) = chan<MemReaderResp, u32:1>[2]("n_mem_rd_resp");

        spawn mem_reader_simple_arbiter::MemReaderSimpleArbiter<ADDR_W, DATA_W, u32:2> (
            n_mem_rd_req_r, n_mem_rd_resp_s,
            mem_rd_req_s, mem_rd_resp_r,
        );

        spawn RleBlockEncoderFullSearch<ADDR_W, DATA_W, LENGTH_W, MAX_TX_PER_REQ>
        (
            n_mem_rd_req_s[0], n_mem_rd_resp_r[0],
            loop_req_r, loop_resp_s,
        );

        spawn RleBlockEncoderSampler<ADDR_W, DATA_W, LENGTH_W>
        (
            n_mem_rd_req_s[1], n_mem_rd_resp_r[1],
            heuristic_req_r, heuristic_resp_s,
        );

        (
            req, resp,
            loop_req_s, loop_resp_r,
            heuristic_req_s, heuristic_resp_r,
        )
    }

    init { }

    next(state: ()) {
        let (tok, req) = recv(join(), req);
        let stride = req.length / (SAMPLE_COUNT);

        // Only do sample scan for data large enough
        // so that sample scan doesn't perform full scan actually.
        let do_sample_scan: bool = stride > (DATA_W / u32:8);

        let tok1 = send_if(
            tok,
            heuristic_req_s,
            do_sample_scan,
            LoopConfig {
                address: req.addr,
                stride: stride,
                left: SAMPLE_COUNT
            }
        );

        let (tok1, loop_resp) = recv_if(tok1, heuristic_resp_r, do_sample_scan, zero!<LoopResp>());

        let do_full_scan: bool = (do_sample_scan && loop_resp.all_equal) || !do_sample_scan;

        let tok2 = send_if(
            tok,
            loop_req_s,
            do_full_scan,
            LoopConfig {
                address: req.addr,
                stride: uN[ADDR_W]:1,
                left: req.length
            }
        );

        let (result, symbol) = if do_sample_scan && !loop_resp.read_success {
            (Status::ERROR, u8:0)
        } else if do_sample_scan && !loop_resp.all_equal {
            (Status::INCOMPRESSIBLE, u8:0)
        } else {
            let (tok2, loop_resp) = recv(tok2, loop_resp_r);
            if !loop_resp.read_success {
                (Status::ERROR, u8:0)
            } else if !loop_resp.all_equal {
                (Status::INCOMPRESSIBLE, u8:0)
            } else {
                (Status::OK, loop_resp.value)
            }
        };

        let tok = send(tok, resp, Resp {
            symbol: symbol,
            length: req.length,
            status: result,
        });
    }
}

const INST_ADDR_W = u32:32;
const INST_DATA_W = u32:64;
const INST_LENGTH_W = u32:32;
const INST_SAMPLE_COUNT = u32:8;

proc RleBlockEncoderInst {
    type Req = RleBlockEncoderReq<INST_ADDR_W>;
    type Resp = RleBlockEncoderResp<INST_LENGTH_W>;
    type MemReaderReq = mem_reader::MemReaderReq<INST_ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<INST_DATA_W, INST_ADDR_W>;
    type Status = RleBlockEncoderStatus;

    config(
        req: chan<Req> in,
        resp: chan<Resp> out,
        mem_rd_req_s: chan<MemReaderReq> out,
        mem_rd_resp_r: chan<MemReaderResp> in,
    ) {
        spawn RleBlockEncoder<INST_ADDR_W, INST_DATA_W, INST_LENGTH_W, INST_SAMPLE_COUNT>
        (
            req, resp,
            mem_rd_req_s, mem_rd_resp_r
        );
    }

    init {  }
    next(state: ()) {  }
}


const TEST_ADDR_W = u32:32;
const TEST_AXI_ADDR_W = u32:32;
const TEST_AXI_DATA_W = u32:64;
const TEST_AXI_DEST_W = u32:8;
const TEST_AXI_ID_W = u32:4;
const TEST_WRITER_ID = u32:3;

const TEST_RAM_SIZE = u32:200;
const TEST_RAM_DATA_W = u32:64;
const TEST_RAM_WORD_PARTITION_SIZE = u32:8;
const TEST_RAM_NUM_PARTITIONS = TEST_RAM_DATA_W / TEST_RAM_WORD_PARTITION_SIZE;
const TEST_RAM_SIMULTANEOUS_RW_BEHAVIOR = ram::SimultaneousReadWriteBehavior::READ_BEFORE_WRITE;
const TEST_RAM_INITIALIZED = true;
const TEST_RAM_ASSERT_VALID_READ = true;

const TEST_LENGTH_W = u32:32;
const TEST_SAMPLE_COUNT = u32:2;
const TEST_MAX_TX_PER_REQ = u32:16;

type TestLen = bits[TEST_LENGTH_W];
type TestReq = RleBlockEncoderReq<TEST_ADDR_W>;
type TestResp = RleBlockEncoderResp<TEST_LENGTH_W>;
type TestStatus = RleBlockEncoderStatus;
type TestAddr = bits[TEST_ADDR_W];

const TEST_CASES = [
    (
        [
            u64:0x00FF_FFFF_FFFF_FFFF,
            u64:0x0000_0000_0000_0000,
            u64:0x0000_0000_0000_0000,
            u64:0x0000_0000_0000_0000,
            u64:0x0000_0000_0000_0000,
            u64:0x0000_0000_0000_0000,
        ],
        TestLen:7,
        TestResp {
            symbol: u8:0xFF,
            length: TestLen:7,
            status: TestStatus::OK
        }
    ),
    (
        [
            u64:0xFFFF_FFFF_FFFF_FFFF,
            u64:0xFFFF_FFFF_FFFF_FFFF,
            u64:0xFFFF_FFFF_FFFF_FFFF,
            u64:0xFFFF_FFFF_FFFF_FFFF,
            u64:0xFFFF_FFFF_FFFF_FFFF,
            u64:0xFFFF_FFFF_FFFF_FFFF,
        ],
        TestLen:48,
        TestResp {
            symbol: u8:0xFF,
            length: TestLen:48,
            status: TestStatus::OK
        }
    ),
    (
        [
            u64:0xBBBB_BBBB_BBBB_BBBB,
            u64:0xBBBB_BBBB_BBBB_BBBB,
            u64:0xBBBB_BBBB_BBBB_BBBB,
            u64:0xFFFF_FFFF_FFFF_FFFF,
            u64:0xFFFF_FFFF_FFFF_FFFF,
            u64:0xFFFF_FFFF_FFFF_FFFF,
        ],
        TestLen:24,
        TestResp {
            symbol: u8:0xBB,
            length: TestLen:24,
            status: TestStatus::OK
        }
    ),
    (
        [
           u64:0xFFFF_FFFF_FFFF_FFBB,
           u64:0xFFFF_FFFF_FFFF_FFFF,
           u64:0xFFFF_FFFF_FFFF_FFFF,
           u64:0xFFFF_FFFF_FFFF_FFFF,
           u64:0xFFFF_FFFF_FFFF_FFFF,
           u64:0xFFFF_FFFF_FFFF_FFFF,
        ],
        TestLen:48,
        TestResp {
           symbol: u8:0x0,
           length: TestLen:48,
           status: TestStatus::INCOMPRESSIBLE
        }
    ),
    (
        [
           u64:0xFFFF_FFFF_FFFF_FFFF,
           u64:0xFFFF_FFFF_FFFF_FFFF,
           u64:0xFFFF_FFFF_FFFF_FFFF,
           u64:0xFFFF_FFFF_FFFF_FFFF,
           u64:0xFFFF_FFFF_FFFF_FFFF,
           u64:0xFFFF_FFFF_FFFF_FFFF,
        ],
        TestLen:4,
        TestResp {
           symbol: u8:0xFF,
           length: TestLen:4,
           status: TestStatus::OK
        }
    ),
    (
        [
           u64:0xFFFF_FFFF_FFFF_FFFF,
           u64:0xFFFF_FFFF_FFFF_FFFF,
           u64:0xFFFF_FFFF_FFFF_FFFF,
           u64:0xFFFF_FFFF_FFFF_FFFF,
           u64:0xFFFF_FFFF_FFFF_FFFF,
           u64:0xFFFF_FFFF_FFFF_FF88,
        ],
        TestLen:48,
        TestResp {
           symbol: u8:0x0,
           length: TestLen:48,
           status: TestStatus::INCOMPRESSIBLE
        }
    ),
    (
        [
           u64:0x88FF_FFFF_FFFF_FFFF,
           u64:0xFFFF_FFFF_FFFF_FFFF,
           u64:0xFFFF_FFFF_FFFF_FFFF,
           u64:0xFFFF_FFFF_FFFF_FFFF,
           u64:0xFFFF_FFFF_FFFF_FFFF,
           u64:0xFFFF_FFFF_FFFF_FF88,
        ],
        TestLen:48,
        TestResp {
           symbol: u8:0x0,
           length: TestLen:48,
           status: TestStatus::INCOMPRESSIBLE
        }
    ),
    (
        [
            u64:0xFFFF_FFFF_FFFF_FFFF,
            u64:0xFFFF_FFFF_FFFF_FFFF,
            u64:0xFFFF_77FF_FFFF_FFFF,
            u64:0xFFFF_FFFF_FFFF_FFFF,
            u64:0xFFFF_FFFF_FFFF_FFFF,
            u64:0xFFFF_FFFF_FFFF_FFFF,
        ],
        TestLen:48,
        TestResp {
            symbol: u8:0x00,
            length: TestLen:48,
            status: TestStatus::INCOMPRESSIBLE
        }
    ),
    (
        [
            u64:0xFFFF_FFFF_FFFF_FFFF,
            u64:0xFFFF_FFFF_FFFF_FFFF,
            u64:0xFFFF_FFFF_FFFF_FFFF,
            u64:0xFFFF_FFFF_FFFF_FF77,
            u64:0xFFFF_FFFF_FFFF_FFFF,
            u64:0xFFFF_FFFF_FFFF_FFFF,
        ],
        TestLen:48,
        TestResp {
            symbol: u8:0x0,
            length: TestLen:48,
            status: TestStatus::INCOMPRESSIBLE
        }
    )
];

#[test_proc]
proc RleBlockEncoderTest {
    type AxiAr = axi::AxiAr<TEST_AXI_ADDR_W, TEST_AXI_ID_W>;
    type AxiR = axi::AxiR<TEST_AXI_DATA_W, TEST_AXI_ID_W>;
    type AxiAw = axi::AxiAw<TEST_AXI_ADDR_W, TEST_AXI_ID_W>;
    type AxiW = axi::AxiW<TEST_AXI_DATA_W, TEST_RAM_NUM_PARTITIONS>;
    type AxiB = axi::AxiB<TEST_AXI_ID_W>;

    type TestReadReq = ram::ReadReq<TEST_ADDR_W, TEST_RAM_NUM_PARTITIONS>;
    type TestWriteReq = ram::WriteReq<TEST_ADDR_W, TEST_RAM_DATA_W, TEST_RAM_NUM_PARTITIONS>;
    type TestReadResp = ram::ReadResp<TEST_RAM_DATA_W>;
    type TestWriteResp = ram::WriteResp;
    type TestMemReaderReq = mem_reader::MemReaderReq<TEST_ADDR_W>;
    type TestMemReaderResp = mem_reader::MemReaderResp<TEST_RAM_DATA_W, TEST_ADDR_W>;
    type TestMemWriterReq = mem_writer::MemWriterReq<TEST_ADDR_W>;
    type TestMemWriterData = mem_writer::MemWriterDataPacket<TEST_RAM_DATA_W, TEST_ADDR_W>;
    type TestMemWriterResp = mem_writer::MemWriterResp;
    type TestMemWriterStatus = mem_writer::MemWriterRespStatus;

    terminator: chan<bool> out;
    mem_wr_req_s:  chan<TestMemWriterReq> out;
    mem_wr_data_s: chan<TestMemWriterData> out;
    mem_wr_resp_r: chan<TestMemWriterResp> in;
    req_s: chan<TestReq> out;
    resp_r: chan<TestResp> in;

    init {  }

    config(terminator: chan<bool> out) {

        // IO for RAM
        let (rd_req_s, rd_req_r) = chan<TestReadReq>("input_rd_req");
        let (rd_resp_s, rd_resp_r) = chan<TestReadResp>("input_rd_resp");
        let (wr_req_s, wr_req_r) = chan<TestWriteReq>("input_wr_req");
        let (wr_resp_s, wr_resp_r) = chan<TestWriteResp>("input_wr_resp");

        // IO for AxiRamReader <-> MemReader
        let (input_axi_ar_s, input_axi_ar_r) = chan<AxiAr>("input_axi_ar");
        let (input_axi_r_s, input_axi_r_r) = chan<AxiR>("input_axi_r");

        // IO for MemReader <-> Encoder
        let (mem_rd_req_s, mem_rd_req_r) = chan<TestMemReaderReq>("mem_rd_req");
        let (mem_rd_resp_s, mem_rd_resp_r) = chan<TestMemReaderResp>("mem_rd_resp");

        // IO for AxiRamWriter <-> MemWriter
        let (output_axi_aw_s, output_axi_aw_r) = chan<AxiAw>("output_axi_aw");
        let (output_axi_w_s, output_axi_w_r) = chan<AxiW>("output_axi_w");
        let (output_axi_b_s, output_axi_b_r) = chan<AxiB>("output_axi_b");

        // IO for MemWriter <-> Encoder
        let (mem_wr_req_s, mem_wr_req_r) = chan<TestMemWriterReq>("mem_wr_req");
        let (mem_wr_data_s, mem_wr_data_r) = chan<TestMemWriterData>("mem_wr_data");
        let (mem_wr_resp_s, mem_wr_resp_r) = chan<TestMemWriterResp>("mem_wr_resp");

        // IO for RleBlockEncoder
        let (req_s, req_r) = chan<TestReq>("req");
        let (resp_s, resp_r) = chan<TestResp>("resp");

        spawn ram::RamModel<
            TEST_RAM_DATA_W, TEST_RAM_SIZE, TEST_RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_RW_BEHAVIOR, TEST_RAM_INITIALIZED, TEST_RAM_ASSERT_VALID_READ, TEST_AXI_ADDR_W
        >(
            rd_req_r, rd_resp_s,
            wr_req_r, wr_resp_s
        );

        spawn axi_ram_reader::AxiRamReader<
            TEST_AXI_ADDR_W, TEST_AXI_DATA_W, TEST_AXI_DEST_W, TEST_AXI_ID_W, TEST_RAM_SIZE
        >(
            input_axi_ar_r, input_axi_r_s,
            rd_req_s, rd_resp_r
        );

        spawn mem_reader::MemReader<TEST_AXI_DATA_W, TEST_AXI_ADDR_W, TEST_AXI_DEST_W, TEST_AXI_ID_W>(
            mem_rd_req_r, mem_rd_resp_s,
            input_axi_ar_s, input_axi_r_r
        );

        spawn axi_ram_writer::AxiRamWriter<
            TEST_AXI_ADDR_W, TEST_AXI_DATA_W, TEST_AXI_ID_W, TEST_RAM_SIZE, TEST_ADDR_W, TEST_RAM_NUM_PARTITIONS,
        >(
            output_axi_aw_r, output_axi_w_r, output_axi_b_s,
            wr_req_s, wr_resp_r
        );

        spawn mem_writer::MemWriter<
            TEST_AXI_ADDR_W, TEST_AXI_DATA_W, TEST_AXI_DEST_W, TEST_AXI_ID_W, TEST_WRITER_ID
        >(
            mem_wr_req_r, mem_wr_data_r,
            output_axi_aw_s, output_axi_w_s, output_axi_b_r,
            mem_wr_resp_s
        );

        spawn RleBlockEncoder<TEST_ADDR_W, TEST_RAM_DATA_W, TEST_LENGTH_W, TEST_SAMPLE_COUNT, TEST_MAX_TX_PER_REQ>
        (
            req_r, resp_s,
            mem_rd_req_s, mem_rd_resp_r
        );

        (
            terminator,
            mem_wr_req_s, mem_wr_data_s, mem_wr_resp_r,
            req_s, resp_r
        )
    }

    next(state: ()) {
        let tok = for ((_, (TEST_DATA, size_bytes, expected)), tok) in enumerate(TEST_CASES) {
            // arrange
            for ((i, test_data), tok) in enumerate(TEST_DATA) {
                let tok = send(tok, mem_wr_req_s,  TestMemWriterReq {
                    addr: i * TEST_RAM_DATA_W / u32:8 as TestAddr,
                    length: TEST_RAM_DATA_W / u32:8
                });
                let tok = send(tok, mem_wr_data_s, TestMemWriterData {
                    data: test_data,
                    last: true,
                    length: TEST_RAM_DATA_W
                });
                let (tok, _) = recv(tok, mem_wr_resp_r);
                tok
            }(tok);

            // act
            let tok = send(tok, req_s, TestReq {
                addr: uN[TEST_ADDR_W]:0,
                length: size_bytes
            });
            let (tok, response) = recv(tok, resp_r);

            // assert
            if (response != expected) {
                trace_fmt!("Received {:#x} != {:#x}", response, expected);
            } else {};
            assert_eq(response, expected);
            tok
        }(join());
        send(tok, terminator, true);
    }
}

