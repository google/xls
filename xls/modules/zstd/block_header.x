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

// This file contains utilities and type definitions related to
// ZSTD Block Header parsing and the implementation of BlockHeaderWriter proc.
// More information about the ZSTD Block Header can be found in:
// https://datatracker.ietf.org/doc/html/rfc8878#section-3.1.1.2

import std;

import xls.modules.zstd.common as common;
import xls.modules.zstd.memory.mem_writer as mem_writer;

type BlockType = common::BlockType;
type BlockSize = common::BlockSize;

type MemWriterResp = mem_writer::MemWriterResp;
type MemWriterRespStatus = mem_writer::MemWriterRespStatus;

// Status values reported by the block header parsing function
pub enum BlockHeaderStatus: u2 {
    OK = 0,
    CORRUPTED = 1,
    NO_ENOUGH_DATA = 2,
}

// Structure for data obtained from decoding Block_Header
pub struct BlockHeader {
    last: bool,
    btype: BlockType,
    size: BlockSize,
}

struct BlockHeaderWriterReq<ADDR_W: u32> {
    addr: uN[ADDR_W],
    header: BlockHeader,
}

// Auxiliary constant that can be used to initialize Proc's state
// with empty FrameHeader, because `zero!` cannot be used in that context
pub const ZERO_BLOCK_HEADER = zero!<BlockHeader>();

// Extracts Block_Header fields from 24-bit chunk of data
// that is assumed to be a valid Block_Header
pub fn extract_block_header(data:u24) -> BlockHeader {
    BlockHeader {
        size: data[3:24],
        btype: data[1:3] as BlockType,
        last: data[0:1],
    }
}

enum BlockHeaderWriterStatus: u1 {
    OKAY = 0,
    ERROR = 1
}

struct BlockHeaderWriterResp {
    status: BlockHeaderWriterStatus,
}

pub proc BlockHeaderWriter<DATA_W: u32, ADDR_W: u32> {
    type MemWriterReq = mem_writer::MemWriterReq<ADDR_W>;
    type MemWriterDataPacket = mem_writer::MemWriterDataPacket<DATA_W, ADDR_W>;

    req_r: chan<BlockHeaderWriterReq<ADDR_W>> in;
    resp_s: chan<BlockHeaderWriterResp> out;
    mem_wr_req_s: chan<MemWriterReq> out;
    mem_wr_data_s: chan<MemWriterDataPacket> out;
    mem_wr_resp_r: chan<MemWriterResp> in;

    config(
        req_r: chan<BlockHeaderWriterReq<ADDR_W>> in,
        resp_s: chan<BlockHeaderWriterResp> out,
        mem_wr_req_s: chan<MemWriterReq> out,
        mem_wr_data_s: chan<MemWriterDataPacket> out,
        mem_wr_resp_r: chan<MemWriterResp> in,
    ) {
        (req_r, resp_s, mem_wr_req_s, mem_wr_data_s, mem_wr_resp_r)
    }

    init {}
    next(state: ()) {
        let tok = join();
        let (tok, request) = recv(tok, req_r);

        // write this to memory at address request.addr
        let bytes = request.header.size as u21 ++
                    request.header.btype as u2 ++
                    request.header.last as u1;

        let writer_request = MemWriterReq {
            addr: request.addr,
            length: uN[ADDR_W]:3,
        };
        let tok = send(tok, mem_wr_req_s, writer_request);

        // Pass data to MemWriter
        let writer_data = MemWriterDataPacket {
            data: bytes as uN[DATA_W],
            length: uN[ADDR_W]:3,
            last: true,
        };

        let tok = send(tok, mem_wr_data_s, writer_data);

        // Check response from MemWriter
        let (tok, memory_response) = recv(tok, mem_wr_resp_r);

        let status = if (memory_response.status == MemWriterRespStatus::OKAY) {
            BlockHeaderWriterStatus::OKAY
        } else {
            BlockHeaderWriterStatus::ERROR
        };

        send(tok, resp_s, BlockHeaderWriterResp {status});
    }
}

const INST_DATA_W = u32:64;
const INST_ADDR_W = u32:32;

proc BlockHeaderWriterInst {
    type MemWriterReq = mem_writer::MemWriterReq<INST_ADDR_W>;
    type MemWriterDataPacket = mem_writer::MemWriterDataPacket<INST_DATA_W, INST_ADDR_W>;

    config(
        req_r: chan<BlockHeaderWriterReq<INST_ADDR_W>> in,
        resp_s: chan<BlockHeaderWriterResp> out,
        mem_wr_req_s: chan<MemWriterReq> out,
        mem_wr_data_s: chan<MemWriterDataPacket> out,
        mem_wr_resp_r: chan<MemWriterResp> in,
    ) {
        spawn BlockHeaderWriter<INST_DATA_W, INST_ADDR_W>(req_r, resp_s, mem_wr_req_s, mem_wr_data_s, mem_wr_resp_r);
    }

    init { }
    next (state: ()) { }
}

const TEST_ADDR_W = u32:64;
const TEST_DATA_W = u32:32;

type MemWriterReq = mem_writer::MemWriterReq<TEST_ADDR_W>;
type MemWriterDataPacket = mem_writer::MemWriterDataPacket<TEST_DATA_W, TEST_ADDR_W>;

struct TestCase {
    request: BlockHeaderWriterReq<TEST_ADDR_W>,
    expected_mem_writer_req: MemWriterReq,
    expected_mem_writer_data: MemWriterDataPacket,
    mem_writer_resp: MemWriterResp,
    expected_resp: BlockHeaderWriterResp,
}

const TEST_CASES = TestCase[3]:[
    TestCase {
        request: BlockHeaderWriterReq {
            addr: uN[TEST_ADDR_W]:0x43210000,
            header: BlockHeader {
                last: true,
                btype: BlockType::RAW,
                size: BlockSize:0x1000,
            },
        },
        expected_mem_writer_req: MemWriterReq {
            addr: uN[TEST_ADDR_W]:0x43210000,
            length: uN[TEST_ADDR_W]:3,
        },
        expected_mem_writer_data: MemWriterDataPacket{
            data: (u21:0x1000 ++ u2:0 ++ u1:1) as uN[TEST_DATA_W],
            length: uN[TEST_ADDR_W]: 3,
            last: true,
        },
        mem_writer_resp: MemWriterResp {
            status: MemWriterRespStatus::OKAY,
        },
        expected_resp: BlockHeaderWriterResp {
            status: BlockHeaderWriterStatus::OKAY,
        },
    },
    TestCase {
        request: BlockHeaderWriterReq {
            addr: uN[TEST_ADDR_W]:0xaaaaaaa0,
            header: BlockHeader {
                last: false,
                btype: BlockType::COMPRESSED,
                size: BlockSize:0x20,
            },
        },
        expected_mem_writer_req: MemWriterReq {
            addr: uN[TEST_ADDR_W]:0xaaaaaaa0,
            length: uN[TEST_ADDR_W]:3,
        },
        expected_mem_writer_data: MemWriterDataPacket {
            data: (u21:0x20 ++ u2:2 ++ u1:0) as uN[TEST_DATA_W],
            length: uN[TEST_ADDR_W]: 3,
            last: true,
        },
        mem_writer_resp: MemWriterResp {
            status: MemWriterRespStatus::OKAY,
        },
        expected_resp: BlockHeaderWriterResp {
            status: BlockHeaderWriterStatus::OKAY,
        },
    },
    TestCase {
        request: BlockHeaderWriterReq {
            addr: uN[TEST_ADDR_W]:0xc0c0a000,
            header: BlockHeader {
                last: true,
                btype: BlockType::RLE,
                size: BlockSize:0x60,
            },
        },
        expected_mem_writer_req: MemWriterReq {
            addr: uN[TEST_ADDR_W]:0xc0c0a000,
            length: uN[TEST_ADDR_W]:3,
        },
        expected_mem_writer_data: MemWriterDataPacket {
            data: (u21:0x60 ++ u2:1 ++ u1:1) as uN[TEST_DATA_W],
            length: uN[TEST_ADDR_W]: 3,
            last: true,
        },
        mem_writer_resp: MemWriterResp {
            status: MemWriterRespStatus::ERROR,
        },
        expected_resp: BlockHeaderWriterResp {
            status: BlockHeaderWriterStatus::ERROR,
        },
    },
];

#[test_proc]
proc Tester {
    type MemWriterReq = mem_writer::MemWriterReq<TEST_ADDR_W>;
    type MemWriterDataPacket = mem_writer::MemWriterDataPacket<TEST_DATA_W, TEST_ADDR_W>;

    terminator: chan<bool> out;

    // IO for BlockHeaderWriter
    req_s: chan<BlockHeaderWriterReq<TEST_ADDR_W>> out;
    resp_r: chan<BlockHeaderWriterResp> in;
    mem_wr_req_r: chan<MemWriterReq> in;
    mem_wr_data_r: chan<MemWriterDataPacket> in;
    mem_wr_resp_s: chan<MemWriterResp> out;

    init {}

    config(terminator: chan<bool> out) {
        let (req_s, req_r) = chan<BlockHeaderWriterReq<TEST_ADDR_W>>("req");
        let (resp_s, resp_r) = chan<BlockHeaderWriterResp>("resp");
        let (mem_wr_req_s, mem_wr_req_r) = chan<MemWriterReq>("mem_req");
        let (mem_wr_data_s, mem_wr_data_r) = chan<MemWriterDataPacket>("mem_data");
        let (mem_wr_resp_s, mem_wr_resp_r) = chan<MemWriterResp>("mem_resp");

        spawn BlockHeaderWriter<TEST_DATA_W, TEST_ADDR_W>(req_r, resp_s, mem_wr_req_s, mem_wr_data_s, mem_wr_resp_r);

        (terminator, req_s, resp_r, mem_wr_req_r, mem_wr_data_r, mem_wr_resp_s)
    }

    next(state: ()) {
        let tok = join();
        for (test_case, ()) : (TestCase, ()) in TEST_CASES {
            // send request
            let tok = send(tok, req_s, test_case.request);

            // verify request to MemWriter
            let (tok, val) = recv(tok, mem_wr_req_r);
            assert_eq(val, test_case.expected_mem_writer_req);

            // verify data to MemWriter
            let (tok, val) = recv(tok, mem_wr_data_r);
            assert_eq(val, test_case.expected_mem_writer_data);

            // send memory response
            let tok = send(tok, mem_wr_resp_s, test_case.mem_writer_resp);

            // verify final response from BlockHeaderWriter
            let (tok, val) = recv(tok, resp_r);
            assert_eq(val, test_case.expected_resp);
        }(());

        send(tok, terminator, true);
    }
}
