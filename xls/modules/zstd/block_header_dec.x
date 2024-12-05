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

import xls.modules.zstd.block_header as block_header;
import xls.modules.zstd.common as common;
import xls.modules.zstd.memory.mem_reader as mem_reader;

type BlockSize = common::BlockSize;
type BlockType = common::BlockType;
type BlockHeader = block_header::BlockHeader;

pub struct BlockHeaderDecoderReq<ADDR_W: u32> {
    addr: uN[ADDR_W],
}

pub enum BlockHeaderDecoderStatus: u2 {
   OKAY = 0,
   CORRUPTED = 1,
   MEMORY_ACCESS_ERROR = 2,
}

pub struct BlockHeaderDecoderResp {
    status: BlockHeaderDecoderStatus,
    header: BlockHeader,
    rle_symbol: u8,
}

pub proc BlockHeaderDecoder<DATA_W: u32, ADDR_W: u32> {
    type Req = BlockHeaderDecoderReq<ADDR_W>;
    type Resp = BlockHeaderDecoderResp<ADDR_W>;

    type MemReaderReq = mem_reader::MemReaderReq<ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<DATA_W, ADDR_W>;
    type MemReaderStatus = mem_reader::MemReaderStatus;

    type Status = BlockHeaderDecoderStatus;
    type Length = uN[ADDR_W];
    type Addr = uN[ADDR_W];

    req_r: chan<Req> in;
    resp_s: chan<Resp> out;
    mem_req_s: chan<MemReaderReq> out;
    mem_resp_r: chan<MemReaderResp> in;

    config (
        req_r: chan<Req> in,
        resp_s: chan<Resp> out,
        mem_req_s: chan<MemReaderReq> out,
        mem_resp_r: chan<MemReaderResp> in,
    ) {
        (req_r, resp_s, mem_req_s, mem_resp_r)
    }

    init { }

    next (state: ()) {
        let tok0 = join();

        // receive request
        let (tok1_0, req, req_valid) = recv_non_blocking(tok0, req_r, zero!<Req>());

        // send memory read request
        let mem_req = MemReaderReq {addr: req.addr, length: Length:4 };
        let tok2_0 = send_if(tok1_0, mem_req_s, req_valid, mem_req);

        // receive memory read response
        let (tok1_1, mem_resp, mem_resp_valid) = recv_non_blocking(tok0, mem_resp_r, zero!<MemReaderResp>());

        let header = block_header::extract_block_header(mem_resp.data as u24);
        let rle_symbol = mem_resp.data[u32:24 +: u8];
        let status = match ( mem_resp.status == MemReaderStatus::OKAY, header.btype != BlockType::RESERVED) {
            (true,  true) => Status::OKAY,
            (true, false) => Status::CORRUPTED,
            (   _,     _) => Status::MEMORY_ACCESS_ERROR,
        };

        let resp = Resp { status, header, rle_symbol };
        let tok2_1 = send_if(tok1_1, resp_s, mem_resp_valid, resp);
    }
}

const INST_DATA_W = u32:64;
const INST_ADDR_W = u32:16;

proc BlockHeaderDecoderInst {
    type Req = BlockHeaderDecoderReq<INST_ADDR_W>;
    type Resp = BlockHeaderDecoderResp;
    type MemReaderReq = mem_reader::MemReaderReq<INST_ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<INST_DATA_W, INST_ADDR_W>;

    config (
        req_r: chan<Req> in,
        resp_s: chan<Resp> out,
        mem_req_s: chan<MemReaderReq> out,
        mem_resp_r: chan<MemReaderResp> in,
    ) {
        spawn BlockHeaderDecoder<INST_DATA_W, INST_ADDR_W>( req_r, resp_s, mem_req_s, mem_resp_r);
    }

    init { }
    next (state: ()) { }
}

const TEST_DATA_W = u32:32;
const TEST_ADDR_W = u32:32;

fn header_to_raw(header: BlockHeader, rle_symbol: u8) -> u32 {
    rle_symbol ++ header.size ++ (header.btype as u2) ++ header.last
}


#[test_proc]
proc BlockHeaderDecoderTest {
    type Req = BlockHeaderDecoderReq<TEST_ADDR_W>;
    type Resp = BlockHeaderDecoderResp;

    type MemReaderReq = mem_reader::MemReaderReq<TEST_ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<TEST_DATA_W, TEST_ADDR_W>;
    type MemReaderStatus = mem_reader::MemReaderStatus;

    type Data = uN[TEST_DATA_W];
    type Addr = uN[TEST_ADDR_W];
    type Length = uN[TEST_ADDR_W];

    terminator: chan<bool> out;

    req_s: chan<Req> out;
    resp_r: chan<Resp> in;

    mem_req_r: chan<MemReaderReq> in;
    mem_resp_s: chan<MemReaderResp> out;

    config (terminator: chan<bool> out) {

        let (req_s, req_r) = chan<Req>("req");
        let (resp_s, resp_r) = chan<Resp>("resp");

        let (mem_req_s, mem_req_r) = chan<MemReaderReq>("mem_req");
        let (mem_resp_s, mem_resp_r) = chan<MemReaderResp>("mem_resp");

        spawn BlockHeaderDecoder<TEST_DATA_W, TEST_ADDR_W> (
            req_r, resp_s, mem_req_s, mem_resp_r
        );

        (terminator, req_s, resp_r, mem_req_r, mem_resp_s)
    }

    init { }

    next (state: ()) {
        const LENGTH = Length:4;

        let tok = join();

        // Test Raw
        let addr = Addr:0x1234;
        let header = BlockHeader { size: BlockSize:0x100, btype: BlockType::RAW, last: true};
        let rle_symbol = u8:0;

        let req = Req { addr };
        let tok = send(tok, req_s, req);

        let (tok, mem_req) = recv(tok, mem_req_r);
        assert_eq(mem_req, MemReaderReq { addr, length: LENGTH });

        let mem_resp = MemReaderResp {
            status: MemReaderStatus::OKAY,
            data: checked_cast<Data>(header_to_raw(header, rle_symbol)),
            length: LENGTH,
            last: true,
        };
        let tok = send(tok, mem_resp_s, mem_resp);
        let (tok, resp) = recv(tok, resp_r);
        assert_eq(resp, Resp {
            status: BlockHeaderDecoderStatus::OKAY,
            header: header,
            rle_symbol: rle_symbol
        });

        // Test RLE
        let addr = Addr:0x2000;
        let header = BlockHeader { size: BlockSize:0x40, btype: BlockType::RLE, last: false};
        let rle_symbol = u8:123;

        let req = Req { addr };
        let tok = send(tok, req_s, req);

        let (tok, mem_req) = recv(tok, mem_req_r);
        assert_eq(mem_req, MemReaderReq { addr, length: LENGTH });

        let mem_resp = MemReaderResp {
            status: MemReaderStatus::OKAY,
            data: checked_cast<Data>(header_to_raw(header, rle_symbol)),
            length: LENGTH,
            last: true,
        };
        let tok = send(tok, mem_resp_s, mem_resp);
        let (tok, resp) = recv(tok, resp_r);
        assert_eq(resp, Resp {
            status: BlockHeaderDecoderStatus::OKAY,
            header: header,
            rle_symbol: rle_symbol
        });

        // Test COMPRESSED
        let addr = Addr:0x2000;
        let header = BlockHeader { size: BlockSize:0x40, btype: BlockType::COMPRESSED, last: true};
        let rle_symbol = u8:0;

        let req = Req { addr };
        let tok = send(tok, req_s, req);

        let (tok, mem_req) = recv(tok, mem_req_r);
        assert_eq(mem_req, MemReaderReq { addr, length: LENGTH });

        let mem_resp = MemReaderResp {
            status: MemReaderStatus::OKAY,
            data: checked_cast<Data>(header_to_raw(header, rle_symbol)),
            length: LENGTH,
            last: true,
        };
        let tok = send(tok, mem_resp_s, mem_resp);
        let (tok, resp) = recv(tok, resp_r);
        assert_eq(resp, Resp {
            status: BlockHeaderDecoderStatus::OKAY,
            header: header,
            rle_symbol: rle_symbol
        });

        // Test RESERVED
        let addr = Addr:0x2000;
        let header = BlockHeader { size: BlockSize:0x40, btype: BlockType::RESERVED, last: true};
        let rle_symbol = u8:0;

        let req = Req { addr };
        let tok = send(tok, req_s, req);

        let (tok, mem_req) = recv(tok, mem_req_r);
        assert_eq(mem_req, MemReaderReq { addr, length: LENGTH });

        let mem_resp = MemReaderResp {
            status: MemReaderStatus::OKAY,
            data: checked_cast<Data>(header_to_raw(header, rle_symbol)),
            length: LENGTH,
            last: true,
        };
        let tok = send(tok, mem_resp_s, mem_resp);
        let (tok, resp) = recv(tok, resp_r);
        assert_eq(resp, Resp {
            status: BlockHeaderDecoderStatus::CORRUPTED,
            header: header,
            rle_symbol: rle_symbol
        });

        // Test memory error
        let addr = Addr:0x2000;
        let header = BlockHeader { size: BlockSize:0x40, btype: BlockType::RESERVED, last: true};
        let rle_symbol = u8:0;

        let req = Req { addr };
        let tok = send(tok, req_s, req);

        let (tok, mem_req) = recv(tok, mem_req_r);
        assert_eq(mem_req, MemReaderReq { addr, length: LENGTH });

        let mem_resp = MemReaderResp {
            status: MemReaderStatus::ERROR,
            data: checked_cast<Data>(header_to_raw(header, rle_symbol)),
            length: LENGTH,
            last: true,
        };
        let tok = send(tok, mem_resp_s, mem_resp);
        let (tok, resp) = recv(tok, resp_r);
        assert_eq(resp, Resp {
            status: BlockHeaderDecoderStatus::MEMORY_ACCESS_ERROR,
            header: header,
            rle_symbol: rle_symbol
        });

        send(tok, terminator, true);
    }
}
