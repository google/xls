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

// This file contains the implementation of LiteralsDecoder.

import std;

import xls.examples.ram;
import xls.modules.zstd.common as common;
import xls.modules.zstd.literals_block_header_dec as literals_block_header_dec;
import xls.modules.zstd.literals_buffer as literals_buffer;
import xls.modules.zstd.memory.axi as axi;
import xls.modules.zstd.memory.axi_ram_reader;
import xls.modules.zstd.memory.mem_reader as mem_reader;
import xls.modules.zstd.parallel_rams as parallel_rams;
import xls.modules.zstd.ram_printer as ram_printer;
import xls.modules.zstd.raw_literals_dec as raw_literals_dec;
import xls.modules.zstd.rle_literals_dec as rle_literals_dec;
import xls.modules.zstd.huffman_literals_dec as huffman_literals_dec;

type CopyOrMatchContent = common::CopyOrMatchContent;
type CopyOrMatchLength = common::CopyOrMatchLength;
type LitData = common::LitData;
type LitLength = common::LitLength;
type LiteralsBlockType = literals_block_header_dec::LiteralsBlockType;
type LiteralsBufferCtrl = common::LiteralsBufferCtrl;
type LiteralsData = common::LiteralsData;
type LiteralsDataWithSync = common::LiteralsDataWithSync;
type LiteralsPathCtrl = common::LiteralsPathCtrl;
type SequenceExecutorMessageType = common::SequenceExecutorMessageType;
type SequenceExecutorPacket = common::SequenceExecutorPacket<common::SYMBOL_WIDTH>;
type Streams = common::Streams;

pub struct LiteralsDecoderCtrlReq<AXI_ADDR_W: u32> {
    addr: uN[AXI_ADDR_W],
    literals_last: bool
}

pub enum LiteralsDecoderCtrlStatus: u1 {
    OKAY = 0,
    ERROR = 1,
}

pub struct LiteralsDecoderCtrlResp {
    status: LiteralsDecoderCtrlStatus
}

struct LiteralsDecoderCtrlState<AXI_ADDR_W: u32> {
    id: u32,
    req: LiteralsDecoderCtrlReq<AXI_ADDR_W>,
    req_valid: bool,
    decoding_raw_literals: bool,
    decoding_rle_literals: bool,
    decoding_huffman_literals: bool,
}

proc LiteralsDecoderCtrl<AXI_ADDR_W: u32> {
    type CtrlReq = LiteralsDecoderCtrlReq<AXI_ADDR_W>;
    type CtrlResp = LiteralsDecoderCtrlResp;
    type HeaderReq = literals_block_header_dec::LiteralsHeaderDecoderReq<AXI_ADDR_W>;
    type HeaderResp = literals_block_header_dec::LiteralsHeaderDecoderResp;
    type RawReq = raw_literals_dec::RawLiteralsDecoderReq<AXI_ADDR_W>;
    type RawResp = raw_literals_dec::RawLiteralsDecoderResp;
    type RawRespStatus = raw_literals_dec::RawLiteralsDecoderStatus;
    type RleReq = rle_literals_dec::RleLiteralsDecoderReq<AXI_ADDR_W>;
    type RleResp = rle_literals_dec::RleLiteralsDecoderResp;
    type RleRespStatus = rle_literals_dec::RleLiteralsDecoderStatus;
    type HuffmanReq = huffman_literals_dec::HuffmanLiteralsDecoderReq<AXI_ADDR_W>;
    type HuffmanResp = huffman_literals_dec::HuffmanLiteralsDecoderResp;
    type HuffmanRespStatus = huffman_literals_dec::HuffmanLiteralsDecoderStatus;

    type Status = LiteralsDecoderCtrlStatus;
    type State = LiteralsDecoderCtrlState;

    // Literals Decoder control
    lit_ctrl_req_r: chan<CtrlReq> in;
    lit_ctrl_resp_s: chan<CtrlResp> out;
    lit_ctrl_header_s: chan<HeaderResp> out;

    // Literals Header Decoder
    lit_header_req_s: chan<HeaderReq> out;
    lit_header_resp_r: chan<HeaderResp> in;

    // Raw Literals Decoder
    raw_lit_req_s: chan<RawReq> out;
    raw_lit_resp_r: chan<RawResp> in;

    // Rle Literals Decoder
    rle_lit_req_s: chan<RleReq> out;
    rle_lit_resp_r: chan<RleResp> in;

    // Huffman Literals Decoder
    huffman_lit_req_s: chan<HuffmanReq> out;
    huffman_lit_resp_r: chan<HuffmanResp> in;

    init {
        zero!<State>()
    }

    config (
        // Literals Decoder control
        lit_ctrl_req_r: chan<CtrlReq> in,
        lit_ctrl_resp_s: chan<CtrlResp> out,
        lit_ctrl_header_s: chan<HeaderResp> out,

        // Literals Header Decoder
        lit_header_req_s: chan<HeaderReq> out,
        lit_header_resp_r: chan<HeaderResp> in,

        // Raw Literals Decoder
        raw_lit_req_s: chan<RawReq> out,
        raw_lit_resp_r: chan<RawResp> in,

        // Rle Literals Decoder
        rle_lit_req_s: chan<RleReq> out,
        rle_lit_resp_r: chan<RleResp> in,

        // Huffman Literals Decoder
        huffman_lit_req_s: chan<HuffmanReq> out,
        huffman_lit_resp_r: chan<HuffmanResp> in
    ) {
        (
            lit_ctrl_req_r, lit_ctrl_resp_s, lit_ctrl_header_s,
            lit_header_req_s, lit_header_resp_r,
            raw_lit_req_s, raw_lit_resp_r,
            rle_lit_req_s, rle_lit_resp_r,
            huffman_lit_req_s, huffman_lit_resp_r
        )
    }

    next (state: State) {
        let tok = join();
        // Try receiving response from Raw-, Rle- and HuffmanLiteralsDecoder procs to free
        // resources at the very begining of next() evaluation
        let do_recv_raw_resp = state.decoding_raw_literals;
        let (tok, raw_resp, raw_resp_valid) = recv_if_non_blocking(tok, raw_lit_resp_r, do_recv_raw_resp, zero!<RawResp>());
        let decoding_raw_literals = if (raw_resp_valid) {
            trace_fmt!("received RawResp: {:#x}", raw_resp);
            false
        } else {
            state.decoding_raw_literals
        };

        let do_recv_rle_resp = state.decoding_rle_literals;
        let (tok, rle_resp, rle_resp_valid) = recv_if_non_blocking(tok, rle_lit_resp_r, do_recv_rle_resp, zero!<RleResp>());
        let decoding_rle_literals = if (rle_resp_valid) {
            trace_fmt!("received RleResp: {:#x}", rle_resp);
            false
        } else {
            state.decoding_rle_literals
        };

        let do_recv_huffman_resp = state.decoding_huffman_literals;
        let (tok, huffman_resp, huffman_resp_valid) = recv_if_non_blocking(tok, huffman_lit_resp_r, do_recv_huffman_resp, zero!<HuffmanResp>());
        let decoding_huffman_literals = if (huffman_resp_valid) {
            trace_fmt!("received HuffmanResp: {:#x}", huffman_resp);
            false
        } else {
            state.decoding_huffman_literals
        };

        // Receive new literals decoding request if previous was handled
        let tok = join();
        let do_recv_ctrl_req = !state.req_valid;
        let (tok, ctrl_req, ctrl_req_valid) = recv_if_non_blocking(tok, lit_ctrl_req_r, do_recv_ctrl_req, zero!<CtrlReq>());

        if (ctrl_req_valid) {
            trace_fmt!("received CtrlReq: {:#x}", ctrl_req);
        } else {};

        let (new_ctrl_req, new_ctrl_req_valid) = if (ctrl_req_valid) {
            (ctrl_req, true)
        } else {
            (state.req, state.req_valid)
        };

        // There's no harm in trying to receive header decoding response in every next() evaluation
        let (tok, header_resp, header_resp_valid) = recv_non_blocking(tok, lit_header_resp_r, zero!<HeaderResp>());
        if (header_resp_valid) {
            trace_fmt!("received HeaderReq: {:#x}", header_resp);
        } else {};

        send_if(tok, lit_ctrl_header_s, header_resp_valid, header_resp);

        // Send literals header decoding request right after receiving CtrlRequest
        let header_req = HeaderReq {
            addr: new_ctrl_req.addr
        };
        let do_send_header_req = ctrl_req_valid;
        let tok = send_if(tok, lit_header_req_s, do_send_header_req, header_req);
        if (do_send_header_req) {
            trace_fmt!("send HeaderReq: {:#x}", header_req);
        } else {};

        // Address of the beginning of the actual literals in the Literals Section
        let literals_addr = state.req.addr + header_resp.length as uN[AXI_ADDR_W];

        // Send raw literals decoding request right after receiving decoded literals header
        let raw_req = RawReq {
            addr: literals_addr,
            length: header_resp.header.regenerated_size as uN[AXI_ADDR_W],
            id: state.id,
            literals_last: state.req.literals_last
        };
        let do_send_raw_req = header_resp_valid && (header_resp.header.literal_type == LiteralsBlockType::RAW) && !state.decoding_raw_literals;
        let tok = send_if(tok, raw_lit_req_s, do_send_raw_req, raw_req);
        let decoding_raw_literals = if (do_send_raw_req) {
            trace_fmt!("send RawReq: {:#x}", raw_req);
            true
        } else {
            decoding_raw_literals
        };

        // Send rle literals decoding request right after receiving decoded literals header
        let rle_req = RleReq {
            symbol: header_resp.symbol,
            length: header_resp.header.regenerated_size,
            id: state.id,
            literals_last: state.req.literals_last
        };
        let do_send_rle_req = header_resp_valid && (header_resp.header.literal_type == LiteralsBlockType::RLE) && !state.decoding_rle_literals;
        let tok = send_if(tok, rle_lit_req_s, do_send_rle_req, rle_req);
        let decoding_rle_literals = if (do_send_rle_req) {
            trace_fmt!("send RleReq: {:#x}", rle_req);
            true
        } else {
            decoding_rle_literals
        };

        // Send huffman literals decoding request right after receiving decoded literals header
        let huffman_new_config = match(header_resp.header.literal_type) {
            LiteralsBlockType::COMP => true,
            LiteralsBlockType::COMP_4 => true,
            LiteralsBlockType::TREELESS => false,
            LiteralsBlockType::TREELESS_4 => false,
            _ => false,
        };
        let huffman_multi_stream = match(header_resp.header.literal_type) {
            LiteralsBlockType::COMP => false,
            LiteralsBlockType::COMP_4 => true,
            LiteralsBlockType::TREELESS => false,
            LiteralsBlockType::TREELESS_4 => true,
            _ => false,
        };
        let huffman_req = HuffmanReq {
            base_addr: literals_addr,
            len: header_resp.header.compressed_size as uN[AXI_ADDR_W],
            new_config: huffman_new_config,
            multi_stream: huffman_multi_stream,
            id: state.id,
            literals_last: state.req.literals_last
        };
        let huffman_literals_type = header_resp.header.literal_type == LiteralsBlockType::COMP ||
                                    header_resp.header.literal_type == LiteralsBlockType::COMP_4 ||
                                    header_resp.header.literal_type == LiteralsBlockType::TREELESS ||
                                    header_resp.header.literal_type == LiteralsBlockType::TREELESS_4;
        let do_send_huffman_req = header_resp_valid && huffman_literals_type && !state.decoding_huffman_literals;
        let tok = send_if(tok, huffman_lit_req_s, do_send_huffman_req, huffman_req);
        let decoding_huffman_literals = if (do_send_huffman_req) {
            trace_fmt!("send HuffmanReq: {:#x}", huffman_req);
            true
        } else {
            decoding_huffman_literals
        };

        // Handle response after literals were decoded
        let do_send_resp = raw_resp_valid ||
                           rle_resp_valid ||
                           huffman_resp_valid;
        let new_ctrl_req_valid = if (do_send_resp) {
            false
        } else {
            new_ctrl_req_valid
        };
        // ERROR status is coded by non-zero integer
        // RleLiteralsDecoder cannot fail
        // Invalid (not received) response defaults to OKAY
        let resp = if (raw_resp.status == RawRespStatus::ERROR ||
                       huffman_resp.status == HuffmanRespStatus::ERROR) {
            CtrlResp { status: Status::ERROR }
        } else {
            CtrlResp { status: Status::OKAY }
        };
        let tok = send_if(tok, lit_ctrl_resp_s, do_send_resp, resp);

        let new_id = if (do_send_resp) {
            if (state.req_valid && state.req.literals_last) {
                u32:0
            } else {
                state.id + u32:1
            }
        } else {
            state.id
        };

        if (do_send_resp) {
            trace_fmt!("send CtrlResp: {:#x}", resp);
        } else {};

        let next_state = State {
            id: new_id,
            req: new_ctrl_req,
            req_valid: new_ctrl_req_valid,
            decoding_raw_literals: decoding_raw_literals,
            decoding_rle_literals: decoding_rle_literals,
            decoding_huffman_literals: decoding_huffman_literals,
        };

        next_state
    }
}

const INST_AXI_ADDR_W = u32:16;
proc LiteralsDecoderCtrlInst {
    type CtrlReq = LiteralsDecoderCtrlReq<INST_AXI_ADDR_W>;
    type CtrlResp = LiteralsDecoderCtrlResp;
    type HeaderReq = literals_block_header_dec::LiteralsHeaderDecoderReq<INST_AXI_ADDR_W>;
    type HeaderResp = literals_block_header_dec::LiteralsHeaderDecoderResp;
    type RawReq = raw_literals_dec::RawLiteralsDecoderReq<INST_AXI_ADDR_W>;
    type RawResp = raw_literals_dec::RawLiteralsDecoderResp;
    type RleReq = rle_literals_dec::RleLiteralsDecoderReq<INST_AXI_ADDR_W>;
    type RleResp = rle_literals_dec::RleLiteralsDecoderResp;
    type HuffmanReq = huffman_literals_dec::HuffmanLiteralsDecoderReq<INST_AXI_ADDR_W>;
    type HuffmanResp = huffman_literals_dec::HuffmanLiteralsDecoderResp;

    init {}

    config (
        // Literals Decoder control
        lit_ctrl_req_r: chan<CtrlReq> in,
        lit_ctrl_resp_s: chan<CtrlResp> out,
        lit_ctrl_header_s: chan<HeaderResp> out,

        // Literals Header Decoder
        lit_header_req_s: chan<HeaderReq> out,
        lit_header_resp_r: chan<HeaderResp> in,

        // Raw Literals Decoder
        raw_lit_req_s: chan<RawReq> out,
        raw_lit_resp_r: chan<RawResp> in,

        // Rle Literals Decoder
        rle_lit_req_s: chan<RleReq> out,
        rle_lit_resp_r: chan<RleResp> in,

        // Huffman Literals Decoder
        huffman_lit_req_s: chan<HuffmanReq> out,
        huffman_lit_resp_r: chan<HuffmanResp> in
    ) {
        spawn LiteralsDecoderCtrl<INST_AXI_ADDR_W>(
            lit_ctrl_req_r, lit_ctrl_resp_s, lit_ctrl_header_s,
            lit_header_req_s, lit_header_resp_r,
            raw_lit_req_s, raw_lit_resp_r,
            rle_lit_req_s, rle_lit_resp_r,
            huffman_lit_req_s, huffman_lit_resp_r
        );
    }

    next (state: ()) {}
}

const TEST_AXI_ADDR_W = u32:16;
const TEST_AXI_DATA_W = u32:64;

#[test_proc]
proc LiteralsDecoderCtrl_test {
    type CtrlReq = LiteralsDecoderCtrlReq<TEST_AXI_ADDR_W>;
    type CtrlResp = LiteralsDecoderCtrlResp;
    type CtrlStatus = LiteralsDecoderCtrlStatus;
    type HeaderReq = literals_block_header_dec::LiteralsHeaderDecoderReq<TEST_AXI_ADDR_W>;
    type HeaderResp = literals_block_header_dec::LiteralsHeaderDecoderResp;
    type HeaderStatus = literals_block_header_dec::LiteralsHeaderDecoderStatus;
    type Header = literals_block_header_dec::LiteralsHeader;
    type LiteralsBlockType = literals_block_header_dec::LiteralsBlockType;
    type RawReq = raw_literals_dec::RawLiteralsDecoderReq<TEST_AXI_ADDR_W>;
    type RawResp = raw_literals_dec::RawLiteralsDecoderResp;
    type RawStatus = raw_literals_dec::RawLiteralsDecoderStatus;
    type RleReq = rle_literals_dec::RleLiteralsDecoderReq<TEST_AXI_ADDR_W>;
    type RleResp = rle_literals_dec::RleLiteralsDecoderResp;
    type RleStatus = rle_literals_dec::RleLiteralsDecoderStatus;
    type HuffmanReq = huffman_literals_dec::HuffmanLiteralsDecoderReq<TEST_AXI_ADDR_W>;
    type HuffmanResp = huffman_literals_dec::HuffmanLiteralsDecoderResp;
    type HuffmanStatus = huffman_literals_dec::HuffmanLiteralsDecoderStatus;

    type Addr = uN[TEST_AXI_ADDR_W];

    terminator: chan<bool> out;

    // Literals Decoder control
    lit_ctrl_req_s: chan<CtrlReq> out;
    lit_ctrl_resp_r: chan<CtrlResp> in;
    lit_ctrl_header_r: chan<HeaderResp> in;

    // Literals Header Decoder
    lit_header_req_r: chan<HeaderReq> in;
    lit_header_resp_s: chan<HeaderResp> out;

    // Raw Literals Decoder
    raw_lit_req_r: chan<RawReq> in;
    raw_lit_resp_s: chan<RawResp> out;

    // Rle Literals Decoder
    rle_lit_req_r: chan<RleReq> in;
    rle_lit_resp_s: chan<RleResp> out;

    // Huffman Literals Decoder
    huffman_lit_req_r: chan<HuffmanReq> in;
    huffman_lit_resp_s: chan<HuffmanResp> out;

    config (terminator: chan<bool> out) {
        // Literals Decoder control
        let (lit_ctrl_req_s, lit_ctrl_req_r) = chan<CtrlReq>("lit_ctrl_req");
        let (lit_ctrl_resp_s, lit_ctrl_resp_r) = chan<CtrlResp>("lit_ctrl_resp");
        let (lit_ctrl_header_s, lit_ctrl_header_r) = chan<HeaderResp>("lit_ctrl_resp");

        // Literals Header Decoder
        let (lit_header_req_s, lit_header_req_r) = chan<HeaderReq>("lit_header_req");
        let (lit_header_resp_s, lit_header_resp_r) = chan<HeaderResp>("lit_header_resp");

        // Raw Literals Decoder
        let (raw_lit_req_s, raw_lit_req_r) = chan<RawReq>("raw_lit_req");
        let (raw_lit_resp_s, raw_lit_resp_r) = chan<RawResp>("raw_lit_resp");

        // Rle Literals Decoder
        let (rle_lit_req_s, rle_lit_req_r) = chan<RleReq>("rle_lit_req");
        let (rle_lit_resp_s, rle_lit_resp_r) = chan<RleResp>("rle_lit_resp");

        // Huffman Literals Decoder
        let (huffman_lit_req_s, huffman_lit_req_r) = chan<HuffmanReq>("huffman_lit_req");
        let (huffman_lit_resp_s, huffman_lit_resp_r) = chan<HuffmanResp>("huffman_lit_resp");

        spawn LiteralsDecoderCtrl<TEST_AXI_ADDR_W>(
            lit_ctrl_req_r, lit_ctrl_resp_s, lit_ctrl_header_s,
            lit_header_req_s, lit_header_resp_r,
            raw_lit_req_s, raw_lit_resp_r,
            rle_lit_req_s, rle_lit_resp_r,
            huffman_lit_req_s, huffman_lit_resp_r
        );

        (
            terminator,
            lit_ctrl_req_s, lit_ctrl_resp_r, lit_ctrl_header_r,
            lit_header_req_r, lit_header_resp_s,
            raw_lit_req_r, raw_lit_resp_s,
            rle_lit_req_r, rle_lit_resp_s,
            huffman_lit_req_r, huffman_lit_resp_s
        )
    }

    init {}

    next (state: ()) {
        let tok = join();

        let lit_ctrl_reqs: CtrlReq[6] = [
            CtrlReq { addr: Addr:0x4, literals_last: false },
            CtrlReq { addr: Addr:0x34, literals_last: false },
            CtrlReq { addr: Addr:0x234, literals_last: true },
            CtrlReq { addr: Addr:0x1234, literals_last: false },
            CtrlReq { addr: Addr:0x2345, literals_last: false },
            CtrlReq { addr: Addr:0x3456, literals_last: true },
        ];

        let lit_header_resps: HeaderResp[6] = [
            HeaderResp {
                header: Header {
                    literal_type: LiteralsBlockType::RAW,
                    regenerated_size: u20:0x10,
                    compressed_size: u20:0x20
                },
                symbol: u8:0x00,
                length: u3:5,
                status: HeaderStatus::OKAY
            },
            HeaderResp {
                header: Header {
                    literal_type: LiteralsBlockType::RAW,
                    regenerated_size: u20:0x20,
                    compressed_size: u20:0x10
                },
                symbol: u8:0x00,
                length: u3:3,
                status: HeaderStatus::OKAY
            },
            HeaderResp {
                header: Header {
                    literal_type: LiteralsBlockType::RLE,
                    regenerated_size: u20:0x15,
                    compressed_size: u20:0x20
                },
                symbol: u8:0x5B,
                length: u3:4,
                status: HeaderStatus::OKAY
            },
            HeaderResp {
                header: Header {
                    literal_type: LiteralsBlockType::RAW,
                    regenerated_size: u20:0x10,
                    compressed_size: u20:0x20
                },
                symbol: u8:0x00,
                length: u3:5,
                status: HeaderStatus::OKAY
            },
            HeaderResp {
                header: Header {
                    literal_type: LiteralsBlockType::RLE,
                    regenerated_size: u20:0x35,
                    compressed_size: u20:0x20
                },
                symbol: u8:0x6C,
                length: u3:3,
                status: HeaderStatus::OKAY
            },
            HeaderResp {
                header: Header {
                    literal_type: LiteralsBlockType::RAW,
                    regenerated_size: u20:0x10,
                    compressed_size: u20:0x20
                },
                symbol: u8:0x00,
                length: u3:5,
                status: HeaderStatus::OKAY
            }
        ];

        // IDs of decoding requests
        // Should be zero after each ctrl request with literals_last == true
        let req_ids: u32[6] = [
            u32:0,
            u32:1,
            u32:2,
            u32:0,
            u32:1,
            u32:2,
        ];

        // Test logic
        let tok = for (i, tok): (u32, token) in range(u32:0, u32:6) {
            let lit_ctrl_req = lit_ctrl_reqs[i];
            let expected_lit_header_req = HeaderReq { addr: lit_ctrl_req.addr };
            let lit_header_resp = lit_header_resps[i];

            let tok = send(tok, lit_ctrl_req_s, lit_ctrl_req);
            trace_fmt!("Test: Sent CtrlReq: {:#x}", lit_ctrl_req);

            let (tok, lit_header_req) = recv(tok, lit_header_req_r);
            trace_fmt!("Test: Received HeaderReq: {:#x}", lit_header_req);
            assert_eq(lit_header_req, expected_lit_header_req);
            let tok = send(tok, lit_header_resp_s, lit_header_resp);
            trace_fmt!("Test: Sent HeaderResp: {:#x}", lit_header_resp);

            if (lit_header_resp.header.literal_type == LiteralsBlockType::RAW) {
                let (tok, raw_lit_req) = recv(tok, raw_lit_req_r);
                trace_fmt!("Test: Received RawReq: {:#x}", raw_lit_req);
                let expected_raw_lit_req = RawReq {
                    id: req_ids[i],
                    addr: lit_ctrl_reqs[i].addr + lit_header_resps[i].length as Addr,
                    length: lit_header_resps[i].header.regenerated_size as Addr,
                    literals_last: lit_ctrl_reqs[i].literals_last
                };
                assert_eq(raw_lit_req, expected_raw_lit_req);
                let raw_lit_resp = RawResp { status: RawStatus::OKAY };
                let tok = send(tok, raw_lit_resp_s, raw_lit_resp);
                trace_fmt!("Test: Sent RawResp: {:#x}", raw_lit_resp);
            } else if (lit_header_resp.header.literal_type == LiteralsBlockType::RLE) {
                let (tok, rle_lit_req) = recv(tok, rle_lit_req_r);
                trace_fmt!("Test: Received RleReq: {:#x}", rle_lit_req);
                let expected_rle_lit_req = RleReq {
                    id: req_ids[i],
                    symbol: lit_header_resps[i].symbol,
                    length: lit_header_resps[i].header.regenerated_size,
                    literals_last: lit_ctrl_reqs[i].literals_last
                };
                assert_eq(rle_lit_req, expected_rle_lit_req);
                let rle_lit_resp = RleResp { status: RleStatus::OKAY };
                let tok = send(tok, rle_lit_resp_s, rle_lit_resp);
                trace_fmt!("Test: Sent RleResp: {:#x}", rle_lit_resp);
            } else {
                //let (tok, huffman_lit_req) = recv(tok, huffman_lit_req_r);
                //trace_fmt!("Test: Received HuffmanReq: {:#x}", huffman_lit_req);
                //let expected_huffman_lit_req = HuffmanReq {
                //};
                //assert_eq(huffman_lit_req, expected_huffman_lit_req);
                //let huffman_lit_resp = HuffmanResp { status: HuffmanStatus::OKAY };
                //let tok = send(tok, huffman_lit_resp_s, huffman_lit_resp);
                //trace_fmt!("Test: Sent HuffmanResp: {:#x}", huffman_lit_resp);
            };

            let (tok, lit_ctrl_resp) = recv(tok, lit_ctrl_resp_r);
            trace_fmt!("Test: Received CtrlResp: {:#x}", lit_ctrl_resp);
            let expected_lit_ctrl_resp = CtrlResp { status: CtrlStatus::OKAY };
            assert_eq(lit_ctrl_resp, expected_lit_ctrl_resp);

            tok
        }(tok);

        send(tok, terminator, true);
    }
}

pub proc LiteralsDecoder<
    HISTORY_BUFFER_SIZE_KB: u32,
    // AXI parameters
    AXI_DATA_W: u32, AXI_ADDR_W: u32, AXI_ID_W: u32, AXI_DEST_W: u32,

    HUFFMAN_WEIGHTS_DPD_RAM_ADDR_WIDTH: u32, HUFFMAN_WEIGHTS_DPD_RAM_DATA_WIDTH: u32, HUFFMAN_WEIGHTS_DPD_RAM_NUM_PARTITIONS: u32,
    HUFFMAN_WEIGHTS_TMP_RAM_ADDR_WIDTH: u32, HUFFMAN_WEIGHTS_TMP_RAM_DATA_WIDTH: u32, HUFFMAN_WEIGHTS_TMP_RAM_NUM_PARTITIONS: u32,
    HUFFMAN_WEIGHTS_TMP2_RAM_ADDR_WIDTH: u32, HUFFMAN_WEIGHTS_TMP2_RAM_DATA_WIDTH: u32, HUFFMAN_WEIGHTS_TMP2_RAM_NUM_PARTITIONS: u32,
    HUFFMAN_WEIGHTS_FSE_RAM_ADDR_WIDTH: u32, HUFFMAN_WEIGHTS_FSE_RAM_DATA_WIDTH: u32, HUFFMAN_WEIGHTS_FSE_RAM_NUM_PARTITIONS: u32,

    // Huffman weights memory parameters
    HUFFMAN_WEIGHTS_RAM_ADDR_WIDTH: u32 = {huffman_literals_dec::WEIGHTS_ADDR_WIDTH},
    HUFFMAN_WEIGHTS_RAM_DATA_WIDTH: u32 = {huffman_literals_dec::WEIGHTS_DATA_WIDTH},
    HUFFMAN_WEIGHTS_RAM_NUM_PARTITIONS: u32 = {huffman_literals_dec::WEIGHTS_NUM_PARTITIONS},

    // Huffman prescan memory parameters
    HUFFMAN_PRESCAN_RAM_ADDR_WIDTH: u32 = {huffman_literals_dec::PRESCAN_ADDR_WIDTH},
    HUFFMAN_PRESCAN_RAM_DATA_WIDTH: u32 = {huffman_literals_dec::PRESCAN_DATA_WIDTH},
    HUFFMAN_PRESCAN_RAM_NUM_PARTITIONS: u32 = {huffman_literals_dec::PRESCAN_NUM_PARTITIONS},

    // Literals buffer memory parameters
    LITERALS_BUFFER_RAM_ADDR_WIDTH: u32 = {parallel_rams::ram_addr_width(HISTORY_BUFFER_SIZE_KB)},
    LITERALS_BUFFER_RAM_SIZE: u32 = {parallel_rams::ram_size(HISTORY_BUFFER_SIZE_KB)},
    LITERALS_BUFFER_RAM_DATA_WIDTH: u32 = {literals_buffer::RAM_DATA_WIDTH},
    LITERALS_BUFFER_RAM_NUM_PARTITIONS: u32 = {literals_buffer::RAM_NUM_PARTITIONS},
> {
    type ReadReq = ram::ReadReq<LITERALS_BUFFER_RAM_ADDR_WIDTH, LITERALS_BUFFER_RAM_NUM_PARTITIONS>;
    type ReadResp = ram::ReadResp<LITERALS_BUFFER_RAM_DATA_WIDTH>;
    type WriteReq = ram::WriteReq<LITERALS_BUFFER_RAM_ADDR_WIDTH, LITERALS_BUFFER_RAM_DATA_WIDTH, LITERALS_BUFFER_RAM_NUM_PARTITIONS>;
    type WriteResp = ram::WriteResp;
    type MemAxiAr = axi::AxiAr<AXI_ADDR_W, AXI_ID_W>;
    type MemAxiR = axi::AxiR<AXI_DATA_W, AXI_ID_W>;

    type CtrlReq = LiteralsDecoderCtrlReq<AXI_ADDR_W>;
    type CtrlResp = LiteralsDecoderCtrlResp;
    type BufferCtrl = common::LiteralsBufferCtrl;
    type BufferOut = common::SequenceExecutorPacket<common::SYMBOL_WIDTH>;

    // TODO: make sure those can use the same parameters
    type HuffmanWeightsReadReq    = ram::ReadReq<HUFFMAN_WEIGHTS_RAM_ADDR_WIDTH, HUFFMAN_WEIGHTS_RAM_NUM_PARTITIONS>;
    type HuffmanWeightsReadResp   = ram::ReadResp<HUFFMAN_WEIGHTS_RAM_DATA_WIDTH>;
    type HuffmanWeightsWriteReq   = ram::WriteReq<HUFFMAN_WEIGHTS_RAM_ADDR_WIDTH, HUFFMAN_WEIGHTS_RAM_DATA_WIDTH, HUFFMAN_WEIGHTS_RAM_NUM_PARTITIONS>;
    type HuffmanWeightsWriteResp  = ram::WriteResp;

    type HuffmanPrescanReadReq    = ram::ReadReq<HUFFMAN_PRESCAN_RAM_ADDR_WIDTH, HUFFMAN_PRESCAN_RAM_NUM_PARTITIONS>;
    type HuffmanPrescanReadResp   = ram::ReadResp<HUFFMAN_PRESCAN_RAM_DATA_WIDTH>;
    type HuffmanPrescanWriteReq   = ram::WriteReq<HUFFMAN_PRESCAN_RAM_ADDR_WIDTH, HUFFMAN_PRESCAN_RAM_DATA_WIDTH, HUFFMAN_PRESCAN_RAM_NUM_PARTITIONS>;
    type HuffmanPrescanWriteResp  = ram::WriteResp;

    type HuffmanWeightsDpdRamRdReq = ram::ReadReq<HUFFMAN_WEIGHTS_DPD_RAM_ADDR_WIDTH, HUFFMAN_WEIGHTS_DPD_RAM_NUM_PARTITIONS>;
    type HuffmanWeightsDpdRamRdResp = ram::ReadResp<HUFFMAN_WEIGHTS_DPD_RAM_DATA_WIDTH>;
    type HuffmanWeightsDpdRamWrReq = ram::WriteReq<HUFFMAN_WEIGHTS_DPD_RAM_ADDR_WIDTH, HUFFMAN_WEIGHTS_DPD_RAM_DATA_WIDTH, HUFFMAN_WEIGHTS_DPD_RAM_NUM_PARTITIONS>;
    type HuffmanWeightsDpdRamWrResp = ram::WriteResp;

    type HuffmanWeightsTmpRamRdReq = ram::ReadReq<HUFFMAN_WEIGHTS_TMP_RAM_ADDR_WIDTH, HUFFMAN_WEIGHTS_TMP_RAM_NUM_PARTITIONS>;
    type HuffmanWeightsTmpRamRdResp = ram::ReadResp<HUFFMAN_WEIGHTS_TMP_RAM_DATA_WIDTH>;
    type HuffmanWeightsTmpRamWrReq = ram::WriteReq<HUFFMAN_WEIGHTS_TMP_RAM_ADDR_WIDTH, HUFFMAN_WEIGHTS_TMP_RAM_DATA_WIDTH, HUFFMAN_WEIGHTS_TMP_RAM_NUM_PARTITIONS>;
    type HuffmanWeightsTmpRamWrResp = ram::WriteResp;

    type HuffmanWeightsTmp2RamRdReq = ram::ReadReq<HUFFMAN_WEIGHTS_TMP2_RAM_ADDR_WIDTH, HUFFMAN_WEIGHTS_TMP2_RAM_NUM_PARTITIONS>;
    type HuffmanWeightsTmp2RamRdResp = ram::ReadResp<HUFFMAN_WEIGHTS_TMP2_RAM_DATA_WIDTH>;
    type HuffmanWeightsTmp2RamWrReq = ram::WriteReq<HUFFMAN_WEIGHTS_TMP2_RAM_ADDR_WIDTH, HUFFMAN_WEIGHTS_TMP2_RAM_DATA_WIDTH, HUFFMAN_WEIGHTS_TMP2_RAM_NUM_PARTITIONS>;
    type HuffmanWeightsTmp2RamWrResp = ram::WriteResp;

    type HuffmanWeightsFseRamRdReq = ram::ReadReq<HUFFMAN_WEIGHTS_FSE_RAM_ADDR_WIDTH, HUFFMAN_WEIGHTS_FSE_RAM_NUM_PARTITIONS>;
    type HuffmanWeightsFseRamRdResp = ram::ReadResp<HUFFMAN_WEIGHTS_FSE_RAM_DATA_WIDTH>;
    type HuffmanWeightsFseRamWrReq = ram::WriteReq<HUFFMAN_WEIGHTS_FSE_RAM_ADDR_WIDTH, HUFFMAN_WEIGHTS_FSE_RAM_DATA_WIDTH, HUFFMAN_WEIGHTS_FSE_RAM_NUM_PARTITIONS>;
    type HuffmanWeightsFseRamWrResp = ram::WriteResp;

    type HeaderResp = literals_block_header_dec::LiteralsHeaderDecoderResp;

    config (
        // AXI Literals Header Decoder (manager)
        lit_header_axi_ar_s: chan<MemAxiAr> out,
        lit_header_axi_r_r: chan<MemAxiR> in,

        // AXI Raw Literals Decoder (manager)
        raw_lit_axi_ar_s: chan<MemAxiAr> out,
        raw_lit_axi_r_r: chan<MemAxiR> in,

        // AXI Huffman Literals Decoder (manager)
        huffman_lit_axi_ar_s: chan<MemAxiAr> out,
        huffman_lit_axi_r_r: chan<MemAxiR> in,

        // AXI Huffman Jump Table Decoder (manager)
        huffman_jump_table_axi_ar_s: chan<MemAxiAr> out,
        huffman_jump_table_axi_r_r: chan<MemAxiR> in,

        // AXI Huffman Weights Header Decoder (manager)
        huffman_weights_header_axi_ar_s: chan<MemAxiAr> out,
        huffman_weights_header_axi_r_r: chan<MemAxiR> in,

        // AXI Huffman Weights RAW Decoder (manager)
        huffman_weights_raw_axi_ar_s: chan<MemAxiAr> out,
        huffman_weights_raw_axi_r_r: chan<MemAxiR> in,

        // AXI Huffman Weights FSE Decoder (manager)
        huffman_weights_fse_lookup_dec_axi_ar_s: chan<MemAxiAr> out,
        huffman_weights_fse_lookup_dec_axi_r_r: chan<MemAxiR> in,
        huffman_weights_fse_decoder_dec_axi_ar_s: chan<MemAxiAr> out,
        huffman_weights_fse_decoder_dec_axi_r_r: chan<MemAxiR> in,

        // Literals Decoder control
        lit_ctrl_req_r: chan<CtrlReq> in,
        lit_ctrl_resp_s: chan<CtrlResp> out,
        lit_ctrl_header_s: chan<HeaderResp> out,

        // Literals Decoder output control
        lit_buf_ctrl_r: chan<BufferCtrl> in,
        lit_buf_out_s: chan<BufferOut> out,

        // Internal memory
        rd_req_m0_s: chan<ReadReq> out,
        rd_req_m1_s: chan<ReadReq> out,
        rd_req_m2_s: chan<ReadReq> out,
        rd_req_m3_s: chan<ReadReq> out,
        rd_req_m4_s: chan<ReadReq> out,
        rd_req_m5_s: chan<ReadReq> out,
        rd_req_m6_s: chan<ReadReq> out,
        rd_req_m7_s: chan<ReadReq> out,
        rd_resp_m0_r: chan<ReadResp> in,
        rd_resp_m1_r: chan<ReadResp> in,
        rd_resp_m2_r: chan<ReadResp> in,
        rd_resp_m3_r: chan<ReadResp> in,
        rd_resp_m4_r: chan<ReadResp> in,
        rd_resp_m5_r: chan<ReadResp> in,
        rd_resp_m6_r: chan<ReadResp> in,
        rd_resp_m7_r: chan<ReadResp> in,
        wr_req_m0_s: chan<WriteReq> out,
        wr_req_m1_s: chan<WriteReq> out,
        wr_req_m2_s: chan<WriteReq> out,
        wr_req_m3_s: chan<WriteReq> out,
        wr_req_m4_s: chan<WriteReq> out,
        wr_req_m5_s: chan<WriteReq> out,
        wr_req_m6_s: chan<WriteReq> out,
        wr_req_m7_s: chan<WriteReq> out,
        wr_resp_m0_r: chan<WriteResp> in,
        wr_resp_m1_r: chan<WriteResp> in,
        wr_resp_m2_r: chan<WriteResp> in,
        wr_resp_m3_r: chan<WriteResp> in,
        wr_resp_m4_r: chan<WriteResp> in,
        wr_resp_m5_r: chan<WriteResp> in,
        wr_resp_m6_r: chan<WriteResp> in,
        wr_resp_m7_r: chan<WriteResp> in,

        // Huffman weights memory
        huffman_lit_weights_mem_rd_req_s: chan<HuffmanWeightsReadReq> out,
        huffman_lit_weights_mem_rd_resp_r: chan<HuffmanWeightsReadResp> in,
        huffman_lit_weights_mem_wr_req_s: chan<HuffmanWeightsWriteReq> out,
        huffman_lit_weights_mem_wr_resp_r: chan<HuffmanWeightsWriteResp> in,
        // Huffman prescan memory
        huffman_lit_prescan_mem_rd_req_s: chan<HuffmanPrescanReadReq> out,
        huffman_lit_prescan_mem_rd_resp_r: chan<HuffmanPrescanReadResp> in,
        huffman_lit_prescan_mem_wr_req_s: chan<HuffmanPrescanWriteReq> out,
        huffman_lit_prescan_mem_wr_resp_r: chan<HuffmanPrescanWriteResp> in,

        huffman_lit_weights_dpd_rd_req_s: chan<HuffmanWeightsDpdRamRdReq> out,
        huffman_lit_weights_dpd_rd_resp_r: chan<HuffmanWeightsDpdRamRdResp> in,
        huffman_lit_weights_dpd_wr_req_s: chan<HuffmanWeightsDpdRamWrReq> out,
        huffman_lit_weights_dpd_wr_resp_r: chan<HuffmanWeightsDpdRamWrResp> in,

        huffman_lit_weights_tmp_rd_req_s: chan<HuffmanWeightsTmpRamRdReq> out,
        huffman_lit_weights_tmp_rd_resp_r: chan<HuffmanWeightsTmpRamRdResp> in,
        huffman_lit_weights_tmp_wr_req_s: chan<HuffmanWeightsTmpRamWrReq> out,
        huffman_lit_weights_tmp_wr_resp_r: chan<HuffmanWeightsTmpRamWrResp> in,

        huffman_lit_weights_tmp2_rd_req_s: chan<HuffmanWeightsTmp2RamRdReq> out,
        huffman_lit_weights_tmp2_rd_resp_r: chan<HuffmanWeightsTmp2RamRdResp> in,
        huffman_lit_weights_tmp2_wr_req_s: chan<HuffmanWeightsTmp2RamWrReq> out,
        huffman_lit_weights_tmp2_wr_resp_r: chan<HuffmanWeightsTmp2RamWrResp> in,

        huffman_lit_weights_fse_rd_req_s: chan<HuffmanWeightsFseRamRdReq> out,
        huffman_lit_weights_fse_rd_resp_r: chan<HuffmanWeightsFseRamRdResp> in,
        huffman_lit_weights_fse_wr_req_s: chan<HuffmanWeightsFseRamWrReq> out,
        huffman_lit_weights_fse_wr_resp_r: chan<HuffmanWeightsFseRamWrResp> in,
    ) {
        type HeaderReq = literals_block_header_dec::LiteralsHeaderDecoderReq<AXI_ADDR_W>;
        type HeaderResp = literals_block_header_dec::LiteralsHeaderDecoderResp;
        type RawReq = raw_literals_dec::RawLiteralsDecoderReq<AXI_ADDR_W>;
        type RawResp = raw_literals_dec::RawLiteralsDecoderResp;
        type RleReq = rle_literals_dec::RleLiteralsDecoderReq<AXI_ADDR_W>;
        type RleResp = rle_literals_dec::RleLiteralsDecoderResp;
        type HuffmanReq = huffman_literals_dec::HuffmanLiteralsDecoderReq<AXI_ADDR_W>;
        type HuffmanResp = huffman_literals_dec::HuffmanLiteralsDecoderResp;
        type MemReaderReq  = mem_reader::MemReaderReq<AXI_ADDR_W>;
        type MemReaderResp = mem_reader::MemReaderResp<AXI_DATA_W, AXI_ADDR_W>;

        const CHANNEL_DEPTH = u32:1;
        // Literals Header Decoder
        let (lit_header_mem_rd_req_s, lit_header_mem_rd_req_r) = chan<MemReaderReq, CHANNEL_DEPTH>("lit_header_mem_rd_req");
        let (lit_header_mem_rd_resp_s, lit_header_mem_rd_resp_r) = chan<MemReaderResp, CHANNEL_DEPTH>("lit_header_mem_rd_resp");

        spawn mem_reader::MemReader<AXI_DATA_W, AXI_ADDR_W, AXI_DEST_W, AXI_ID_W, CHANNEL_DEPTH>(
           lit_header_mem_rd_req_r, lit_header_mem_rd_resp_s,
           lit_header_axi_ar_s, lit_header_axi_r_r
        );

        let (lit_header_req_s,  lit_header_req_r) = chan<HeaderReq, CHANNEL_DEPTH>("lit_header_req");
        let (lit_header_resp_s, lit_header_resp_r) = chan<HeaderResp, CHANNEL_DEPTH>("lit_header_resp");

        spawn literals_block_header_dec::LiteralsHeaderDecoder<AXI_DATA_W, AXI_ADDR_W>(
            lit_header_req_r, lit_header_resp_s,
            lit_header_mem_rd_req_s, lit_header_mem_rd_resp_r
        );

        // Raw Literals Decoder
        let (raw_lit_mem_rd_req_s, raw_lit_mem_rd_req_r) = chan<MemReaderReq, CHANNEL_DEPTH>("raw_lit_mem_rd_req");
        let (raw_lit_mem_rd_resp_s, raw_lit_mem_rd_resp_r) = chan<MemReaderResp, CHANNEL_DEPTH>("raw_lit_mem_rd_resp");

        spawn mem_reader::MemReader<AXI_DATA_W, AXI_ADDR_W, AXI_DEST_W, AXI_ID_W, CHANNEL_DEPTH>(
           raw_lit_mem_rd_req_r, raw_lit_mem_rd_resp_s,
           raw_lit_axi_ar_s, raw_lit_axi_r_r
        );

        let (raw_lit_req_s,  raw_lit_req_r) = chan<RawReq, CHANNEL_DEPTH>("raw_lit_req");
        let (raw_lit_resp_s, raw_lit_resp_r) = chan<RawResp, CHANNEL_DEPTH>("raw_lit_resp");
        let (raw_lit_output_s, raw_lit_output_r) = chan<LiteralsDataWithSync, CHANNEL_DEPTH>("raw_lit_output");

        spawn raw_literals_dec::RawLiteralsDecoder<AXI_DATA_W, AXI_ADDR_W>(
            raw_lit_req_r, raw_lit_resp_s, raw_lit_output_s,
            raw_lit_mem_rd_req_s, raw_lit_mem_rd_resp_r
        );

        // Rle Literals Decoder
        let (rle_lit_req_s,  rle_lit_req_r) = chan<RleReq, CHANNEL_DEPTH>("rle_lit_req");
        let (rle_lit_resp_s, rle_lit_resp_r) = chan<RleResp, CHANNEL_DEPTH>("rle_lit_resp");
        let (rle_lit_output_s, rle_lit_output_r) = chan<LiteralsDataWithSync, CHANNEL_DEPTH>("rle_lit_output");

        spawn rle_literals_dec::RleLiteralsDecoder<AXI_DATA_W>(
            rle_lit_req_r, rle_lit_resp_s, rle_lit_output_s
        );

        // Huffman Literals Decoder
        let (huffman_lit_req_s,  huffman_lit_req_r) = chan<HuffmanReq, CHANNEL_DEPTH>("huffman_lit_req");
        let (huffman_lit_resp_s, huffman_lit_resp_r) = chan<HuffmanResp, CHANNEL_DEPTH>("huffman_lit_resp");
        let (huffman_lit_output_s, huffman_lit_output_r) = chan<LiteralsDataWithSync, CHANNEL_DEPTH>("huffman_lit_output");

        spawn huffman_literals_dec::HuffmanLiteralsDecoder<
            AXI_DATA_W, AXI_ADDR_W, AXI_ID_W, AXI_DEST_W,
            HUFFMAN_WEIGHTS_DPD_RAM_ADDR_WIDTH, HUFFMAN_WEIGHTS_DPD_RAM_DATA_WIDTH, HUFFMAN_WEIGHTS_DPD_RAM_NUM_PARTITIONS,
            HUFFMAN_WEIGHTS_TMP_RAM_ADDR_WIDTH, HUFFMAN_WEIGHTS_TMP_RAM_DATA_WIDTH, HUFFMAN_WEIGHTS_TMP_RAM_NUM_PARTITIONS,
            HUFFMAN_WEIGHTS_TMP2_RAM_ADDR_WIDTH, HUFFMAN_WEIGHTS_TMP2_RAM_DATA_WIDTH, HUFFMAN_WEIGHTS_TMP2_RAM_NUM_PARTITIONS,
            HUFFMAN_WEIGHTS_FSE_RAM_ADDR_WIDTH, HUFFMAN_WEIGHTS_FSE_RAM_DATA_WIDTH, HUFFMAN_WEIGHTS_FSE_RAM_NUM_PARTITIONS,
            HUFFMAN_WEIGHTS_RAM_ADDR_WIDTH, HUFFMAN_WEIGHTS_RAM_DATA_WIDTH, HUFFMAN_WEIGHTS_RAM_NUM_PARTITIONS,
            HUFFMAN_PRESCAN_RAM_ADDR_WIDTH, HUFFMAN_PRESCAN_RAM_DATA_WIDTH, HUFFMAN_PRESCAN_RAM_NUM_PARTITIONS
        >(
            huffman_lit_req_r, huffman_lit_resp_s, huffman_lit_output_s,
            huffman_lit_axi_ar_s, huffman_lit_axi_r_r,
            huffman_jump_table_axi_ar_s, huffman_jump_table_axi_r_r,
            huffman_weights_header_axi_ar_s, huffman_weights_header_axi_r_r,
            huffman_weights_raw_axi_ar_s, huffman_weights_raw_axi_r_r,
            huffman_weights_fse_lookup_dec_axi_ar_s, huffman_weights_fse_lookup_dec_axi_r_r,
            huffman_weights_fse_decoder_dec_axi_ar_s, huffman_weights_fse_decoder_dec_axi_r_r,
            huffman_lit_weights_mem_rd_req_s, huffman_lit_weights_mem_rd_resp_r,
            huffman_lit_weights_mem_wr_req_s, huffman_lit_weights_mem_wr_resp_r,
            huffman_lit_prescan_mem_rd_req_s, huffman_lit_prescan_mem_rd_resp_r,
            huffman_lit_prescan_mem_wr_req_s, huffman_lit_prescan_mem_wr_resp_r,
            huffman_lit_weights_dpd_rd_req_s, huffman_lit_weights_dpd_rd_resp_r,
            huffman_lit_weights_dpd_wr_req_s, huffman_lit_weights_dpd_wr_resp_r,
            huffman_lit_weights_tmp_rd_req_s, huffman_lit_weights_tmp_rd_resp_r,
            huffman_lit_weights_tmp_wr_req_s, huffman_lit_weights_tmp_wr_resp_r,
            huffman_lit_weights_tmp2_rd_req_s, huffman_lit_weights_tmp2_rd_resp_r,
            huffman_lit_weights_tmp2_wr_req_s, huffman_lit_weights_tmp2_wr_resp_r,
            huffman_lit_weights_fse_rd_req_s, huffman_lit_weights_fse_rd_resp_r,
            huffman_lit_weights_fse_wr_req_s,  huffman_lit_weights_fse_wr_resp_r,
        );

        // Literals Buffer
        spawn literals_buffer::LiteralsBuffer<
            HISTORY_BUFFER_SIZE_KB,
            LITERALS_BUFFER_RAM_SIZE,
            LITERALS_BUFFER_RAM_ADDR_WIDTH
        > (
            raw_lit_output_r, rle_lit_output_r, huffman_lit_output_r,
            lit_buf_ctrl_r, lit_buf_out_s,
            rd_req_m0_s, rd_req_m1_s, rd_req_m2_s, rd_req_m3_s,
            rd_req_m4_s, rd_req_m5_s, rd_req_m6_s, rd_req_m7_s,
            rd_resp_m0_r, rd_resp_m1_r, rd_resp_m2_r, rd_resp_m3_r,
            rd_resp_m4_r, rd_resp_m5_r, rd_resp_m6_r, rd_resp_m7_r,
            wr_req_m0_s, wr_req_m1_s, wr_req_m2_s, wr_req_m3_s,
            wr_req_m4_s, wr_req_m5_s, wr_req_m6_s, wr_req_m7_s,
            wr_resp_m0_r, wr_resp_m1_r, wr_resp_m2_r, wr_resp_m3_r,
            wr_resp_m4_r, wr_resp_m5_r, wr_resp_m6_r, wr_resp_m7_r,
        );

        spawn LiteralsDecoderCtrl<AXI_ADDR_W> (
            lit_ctrl_req_r, lit_ctrl_resp_s, lit_ctrl_header_s,
            lit_header_req_s, lit_header_resp_r,
            raw_lit_req_s, raw_lit_resp_r,
            rle_lit_req_s, rle_lit_resp_r,
            huffman_lit_req_s, huffman_lit_resp_r,
        );

        ()
    }

    init { }

    next (state: ()) { }
}

const ZSTD_HISTORY_BUFFER_SIZE_KB: u32 = u32:64;
const ZSTD_RAM_ADDR_WIDTH: u32 = parallel_rams::ram_addr_width(ZSTD_HISTORY_BUFFER_SIZE_KB);
const INST_AXI_DATA_W:u32 = u32:64;
const INST_AXI_ID_W:u32 = u32:4;
const INST_AXI_DEST_W:u32 = u32:4;

const INST_HUFFMAN_WEIGHTS_RAM_ADDR_WIDTH = huffman_literals_dec::INST_WEIGHTS_RAM_ADDR_WIDTH;
const INST_HUFFMAN_WEIGHTS_RAM_DATA_WIDTH = huffman_literals_dec::INST_WEIGHTS_RAM_DATA_WIDTH;
const INST_HUFFMAN_WEIGHTS_RAM_NUM_PARTITIONS = huffman_literals_dec::INST_WEIGHTS_RAM_NUM_PARTITIONS;

const INST_HUFFMAN_PRESCAN_RAM_ADDR_WIDTH = huffman_literals_dec::INST_PRESCAN_RAM_ADDR_WIDTH;
const INST_HUFFMAN_PRESCAN_RAM_DATA_WIDTH = huffman_literals_dec::INST_PRESCAN_RAM_DATA_WIDTH;
const INST_HUFFMAN_PRESCAN_RAM_NUM_PARTITIONS = huffman_literals_dec::INST_PRESCAN_RAM_NUM_PARTITIONS;

const INST_HUFFMAN_WEIGHTS_DPD_RAM_DATA_WIDTH = u32:16;
const INST_HUFFMAN_WEIGHTS_DPD_RAM_SIZE = u32:256;
const INST_HUFFMAN_WEIGHTS_DPD_RAM_ADDR_WIDTH = std::clog2(INST_HUFFMAN_WEIGHTS_DPD_RAM_SIZE);
const INST_HUFFMAN_WEIGHTS_DPD_RAM_WORD_PARTITION_SIZE = INST_HUFFMAN_WEIGHTS_DPD_RAM_DATA_WIDTH;
const INST_HUFFMAN_WEIGHTS_DPD_RAM_NUM_PARTITIONS = ram::num_partitions(
    INST_HUFFMAN_WEIGHTS_DPD_RAM_WORD_PARTITION_SIZE, INST_HUFFMAN_WEIGHTS_DPD_RAM_DATA_WIDTH
);
const INST_HUFFMAN_WEIGHTS_FSE_RAM_DATA_WIDTH = u32:32;
const INST_HUFFMAN_WEIGHTS_FSE_RAM_SIZE = u32:256;
const INST_HUFFMAN_WEIGHTS_FSE_RAM_ADDR_WIDTH = std::clog2(INST_HUFFMAN_WEIGHTS_FSE_RAM_SIZE);
const INST_HUFFMAN_WEIGHTS_FSE_RAM_WORD_PARTITION_SIZE = INST_HUFFMAN_WEIGHTS_FSE_RAM_DATA_WIDTH / u32:3;
const INST_HUFFMAN_WEIGHTS_FSE_RAM_NUM_PARTITIONS = ram::num_partitions(
    INST_HUFFMAN_WEIGHTS_FSE_RAM_WORD_PARTITION_SIZE, INST_HUFFMAN_WEIGHTS_FSE_RAM_DATA_WIDTH
);

const INST_HUFFMAN_WEIGHTS_TMP_RAM_DATA_WIDTH = u32:16;
const INST_HUFFMAN_WEIGHTS_TMP_RAM_SIZE = u32:256;
const INST_HUFFMAN_WEIGHTS_TMP_RAM_ADDR_WIDTH = std::clog2(INST_HUFFMAN_WEIGHTS_TMP_RAM_SIZE);
const INST_HUFFMAN_WEIGHTS_TMP_RAM_WORD_PARTITION_SIZE = INST_HUFFMAN_WEIGHTS_TMP_RAM_DATA_WIDTH;
const INST_HUFFMAN_WEIGHTS_TMP_RAM_NUM_PARTITIONS = ram::num_partitions(
    INST_HUFFMAN_WEIGHTS_TMP_RAM_WORD_PARTITION_SIZE, INST_HUFFMAN_WEIGHTS_TMP_RAM_DATA_WIDTH
);

const INST_HUFFMAN_WEIGHTS_TMP2_RAM_DATA_WIDTH = u32:8;
const INST_HUFFMAN_WEIGHTS_TMP2_RAM_SIZE = u32:512;
const INST_HUFFMAN_WEIGHTS_TMP2_RAM_ADDR_WIDTH = std::clog2(INST_HUFFMAN_WEIGHTS_TMP2_RAM_SIZE);
const INST_HUFFMAN_WEIGHTS_TMP2_RAM_WORD_PARTITION_SIZE = INST_HUFFMAN_WEIGHTS_TMP2_RAM_DATA_WIDTH;
const INST_HUFFMAN_WEIGHTS_TMP2_RAM_NUM_PARTITIONS = ram::num_partitions(
    INST_HUFFMAN_WEIGHTS_TMP2_RAM_WORD_PARTITION_SIZE, INST_HUFFMAN_WEIGHTS_TMP2_RAM_DATA_WIDTH
);

proc LiteralsDecoderInst {
    type ReadReq = ram::ReadReq<ZSTD_RAM_ADDR_WIDTH, literals_buffer::RAM_NUM_PARTITIONS>;
    type ReadResp = ram::ReadResp<literals_buffer::RAM_DATA_WIDTH>;
    type WriteReq = ram::WriteReq<ZSTD_RAM_ADDR_WIDTH, literals_buffer::RAM_DATA_WIDTH, literals_buffer::RAM_NUM_PARTITIONS>;
    type WriteResp = ram::WriteResp;
    type MemAxiAr = axi::AxiAr<INST_AXI_ADDR_W, INST_AXI_ID_W>;
    type MemAxiR = axi::AxiR<INST_AXI_DATA_W, INST_AXI_ID_W>;

    type CtrlReq = LiteralsDecoderCtrlReq<INST_AXI_ADDR_W>;
    type CtrlResp = LiteralsDecoderCtrlResp;
    type BufferCtrl = common::LiteralsBufferCtrl;
    type BufferOut = common::SequenceExecutorPacket<common::SYMBOL_WIDTH>;

    type HuffmanWeightsReadReq    = ram::ReadReq<INST_HUFFMAN_WEIGHTS_RAM_ADDR_WIDTH, INST_HUFFMAN_WEIGHTS_RAM_NUM_PARTITIONS>;
    type HuffmanWeightsReadResp   = ram::ReadResp<INST_HUFFMAN_WEIGHTS_RAM_DATA_WIDTH>;
    type HuffmanWeightsWriteReq   = ram::WriteReq<INST_HUFFMAN_WEIGHTS_RAM_ADDR_WIDTH, INST_HUFFMAN_WEIGHTS_RAM_DATA_WIDTH, INST_HUFFMAN_WEIGHTS_RAM_NUM_PARTITIONS>;
    type HuffmanWeightsWriteResp  = ram::WriteResp;

    type HuffmanPrescanReadReq    = ram::ReadReq<INST_HUFFMAN_PRESCAN_RAM_ADDR_WIDTH, INST_HUFFMAN_PRESCAN_RAM_NUM_PARTITIONS>;
    type HuffmanPrescanReadResp   = ram::ReadResp<INST_HUFFMAN_PRESCAN_RAM_DATA_WIDTH>;
    type HuffmanPrescanWriteReq   = ram::WriteReq<INST_HUFFMAN_PRESCAN_RAM_ADDR_WIDTH, INST_HUFFMAN_PRESCAN_RAM_DATA_WIDTH, INST_HUFFMAN_PRESCAN_RAM_NUM_PARTITIONS>;
    type HuffmanPrescanWriteResp  = ram::WriteResp;

    type HuffmanWeightsDpdRamRdReq = ram::ReadReq<INST_HUFFMAN_WEIGHTS_DPD_RAM_ADDR_WIDTH, INST_HUFFMAN_WEIGHTS_DPD_RAM_NUM_PARTITIONS>;
    type HuffmanWeightsDpdRamRdResp = ram::ReadResp<INST_HUFFMAN_WEIGHTS_DPD_RAM_DATA_WIDTH>;
    type HuffmanWeightsDpdRamWrReq = ram::WriteReq<INST_HUFFMAN_WEIGHTS_DPD_RAM_ADDR_WIDTH, INST_HUFFMAN_WEIGHTS_DPD_RAM_DATA_WIDTH, INST_HUFFMAN_WEIGHTS_DPD_RAM_NUM_PARTITIONS>;
    type HuffmanWeightsDpdRamWrResp = ram::WriteResp;

    type HuffmanWeightsTmpRamRdReq = ram::ReadReq<INST_HUFFMAN_WEIGHTS_TMP_RAM_ADDR_WIDTH, INST_HUFFMAN_WEIGHTS_TMP_RAM_NUM_PARTITIONS>;
    type HuffmanWeightsTmpRamRdResp = ram::ReadResp<INST_HUFFMAN_WEIGHTS_TMP_RAM_DATA_WIDTH>;
    type HuffmanWeightsTmpRamWrReq = ram::WriteReq<INST_HUFFMAN_WEIGHTS_TMP_RAM_ADDR_WIDTH, INST_HUFFMAN_WEIGHTS_TMP_RAM_DATA_WIDTH, INST_HUFFMAN_WEIGHTS_TMP_RAM_NUM_PARTITIONS>;
    type HuffmanWeightsTmpRamWrResp = ram::WriteResp;

    type HuffmanWeightsTmp2RamRdReq = ram::ReadReq<INST_HUFFMAN_WEIGHTS_TMP2_RAM_ADDR_WIDTH, INST_HUFFMAN_WEIGHTS_TMP2_RAM_NUM_PARTITIONS>;
    type HuffmanWeightsTmp2RamRdResp = ram::ReadResp<INST_HUFFMAN_WEIGHTS_TMP2_RAM_DATA_WIDTH>;
    type HuffmanWeightsTmp2RamWrReq = ram::WriteReq<INST_HUFFMAN_WEIGHTS_TMP2_RAM_ADDR_WIDTH, INST_HUFFMAN_WEIGHTS_TMP2_RAM_DATA_WIDTH, INST_HUFFMAN_WEIGHTS_TMP2_RAM_NUM_PARTITIONS>;
    type HuffmanWeightsTmp2RamWrResp = ram::WriteResp;

    type HuffmanWeightsFseRamRdReq = ram::ReadReq<INST_HUFFMAN_WEIGHTS_FSE_RAM_ADDR_WIDTH, INST_HUFFMAN_WEIGHTS_FSE_RAM_NUM_PARTITIONS>;
    type HuffmanWeightsFseRamRdResp = ram::ReadResp<INST_HUFFMAN_WEIGHTS_FSE_RAM_DATA_WIDTH>;
    type HuffmanWeightsFseRamWrReq = ram::WriteReq<INST_HUFFMAN_WEIGHTS_FSE_RAM_ADDR_WIDTH, INST_HUFFMAN_WEIGHTS_FSE_RAM_DATA_WIDTH, INST_HUFFMAN_WEIGHTS_FSE_RAM_NUM_PARTITIONS>;
    type HuffmanWeightsFseRamWrResp = ram::WriteResp;

    type HeaderResp = literals_block_header_dec::LiteralsHeaderDecoderResp;

    config (
        // AXI Literals Header Decoder (manager)
        lit_header_axi_ar_s: chan<MemAxiAr> out,
        lit_header_axi_r_r: chan<MemAxiR> in,

        // AXI Raw Literals Decoder (manager)
        raw_lit_axi_ar_s: chan<MemAxiAr> out,
        raw_lit_axi_r_r: chan<MemAxiR> in,

        // AXI Huffman Literals Decoder (manager)
        huffman_lit_axi_ar_s: chan<MemAxiAr> out,
        huffman_lit_axi_r_r: chan<MemAxiR> in,

        // AXI Huffman Jump Table Decoder (manager)
        huffman_jump_table_axi_ar_s: chan<MemAxiAr> out,
        huffman_jump_table_axi_r_r: chan<MemAxiR> in,

        // AXI Huffman Weights Header Decoder (manager)
        huffman_weights_header_axi_ar_s: chan<MemAxiAr> out,
        huffman_weights_header_axi_r_r: chan<MemAxiR> in,

        // AXI Huffman Weights RAW Decoder (manager)
        huffman_weights_raw_axi_ar_s: chan<MemAxiAr> out,
        huffman_weights_raw_axi_r_r: chan<MemAxiR> in,

        // AXI Huffman Weights FSE Decoder (manager)
        huffman_weights_fse_lookup_dec_axi_ar_s: chan<MemAxiAr> out,
        huffman_weights_fse_lookup_dec_axi_r_r: chan<MemAxiR> in,
        huffman_weights_fse_decoder_dec_axi_ar_s: chan<MemAxiAr> out,
        huffman_weights_fse_decoder_dec_axi_r_r: chan<MemAxiR> in,

        // Literals Decoder control
        lit_ctrl_req_r: chan<CtrlReq> in,
        lit_ctrl_resp_s: chan<CtrlResp> out,
        lit_ctrl_header_s: chan<HeaderResp> out,

        // Literals Decoder output control
        lit_buf_ctrl_r: chan<BufferCtrl> in,
        lit_buf_out_s: chan<BufferOut> out,

        // Internal memory
        rd_req_m0_s: chan<ReadReq> out,
        rd_req_m1_s: chan<ReadReq> out,
        rd_req_m2_s: chan<ReadReq> out,
        rd_req_m3_s: chan<ReadReq> out,
        rd_req_m4_s: chan<ReadReq> out,
        rd_req_m5_s: chan<ReadReq> out,
        rd_req_m6_s: chan<ReadReq> out,
        rd_req_m7_s: chan<ReadReq> out,
        rd_resp_m0_r: chan<ReadResp> in,
        rd_resp_m1_r: chan<ReadResp> in,
        rd_resp_m2_r: chan<ReadResp> in,
        rd_resp_m3_r: chan<ReadResp> in,
        rd_resp_m4_r: chan<ReadResp> in,
        rd_resp_m5_r: chan<ReadResp> in,
        rd_resp_m6_r: chan<ReadResp> in,
        rd_resp_m7_r: chan<ReadResp> in,
        wr_req_m0_s: chan<WriteReq> out,
        wr_req_m1_s: chan<WriteReq> out,
        wr_req_m2_s: chan<WriteReq> out,
        wr_req_m3_s: chan<WriteReq> out,
        wr_req_m4_s: chan<WriteReq> out,
        wr_req_m5_s: chan<WriteReq> out,
        wr_req_m6_s: chan<WriteReq> out,
        wr_req_m7_s: chan<WriteReq> out,
        wr_resp_m0_r: chan<WriteResp> in,
        wr_resp_m1_r: chan<WriteResp> in,
        wr_resp_m2_r: chan<WriteResp> in,
        wr_resp_m3_r: chan<WriteResp> in,
        wr_resp_m4_r: chan<WriteResp> in,
        wr_resp_m5_r: chan<WriteResp> in,
        wr_resp_m6_r: chan<WriteResp> in,
        wr_resp_m7_r: chan<WriteResp> in,

        // Huffman weights memory
        huffman_lit_weights_mem_rd_req_s: chan<HuffmanWeightsReadReq> out,
        huffman_lit_weights_mem_rd_resp_r: chan<HuffmanWeightsReadResp> in,
        huffman_lit_weights_mem_wr_req_s: chan<HuffmanWeightsWriteReq> out,
        huffman_lit_weights_mem_wr_resp_r: chan<HuffmanWeightsWriteResp> in,
        // Huffman prescan memory
        huffman_lit_prescan_mem_rd_req_s: chan<HuffmanPrescanReadReq> out,
        huffman_lit_prescan_mem_rd_resp_r: chan<HuffmanPrescanReadResp> in,
        huffman_lit_prescan_mem_wr_req_s: chan<HuffmanPrescanWriteReq> out,
        huffman_lit_prescan_mem_wr_resp_r: chan<HuffmanPrescanWriteResp> in,

        huffman_lit_weights_dpd_rd_req_s: chan<HuffmanWeightsDpdRamRdReq> out,
        huffman_lit_weights_dpd_rd_resp_r: chan<HuffmanWeightsDpdRamRdResp> in,
        huffman_lit_weights_dpd_wr_req_s: chan<HuffmanWeightsDpdRamWrReq> out,
        huffman_lit_weights_dpd_wr_resp_r: chan<HuffmanWeightsDpdRamWrResp> in,

        huffman_lit_weights_tmp_rd_req_s: chan<HuffmanWeightsTmpRamRdReq> out,
        huffman_lit_weights_tmp_rd_resp_r: chan<HuffmanWeightsTmpRamRdResp> in,
        huffman_lit_weights_tmp_wr_req_s: chan<HuffmanWeightsTmpRamWrReq> out,
        huffman_lit_weights_tmp_wr_resp_r: chan<HuffmanWeightsTmpRamWrResp> in,

        huffman_lit_weights_tmp2_rd_req_s: chan<HuffmanWeightsTmp2RamRdReq> out,
        huffman_lit_weights_tmp2_rd_resp_r: chan<HuffmanWeightsTmp2RamRdResp> in,
        huffman_lit_weights_tmp2_wr_req_s: chan<HuffmanWeightsTmp2RamWrReq> out,
        huffman_lit_weights_tmp2_wr_resp_r: chan<HuffmanWeightsTmp2RamWrResp> in,

        huffman_lit_weights_fse_rd_req_s: chan<HuffmanWeightsFseRamRdReq> out,
        huffman_lit_weights_fse_rd_resp_r: chan<HuffmanWeightsFseRamRdResp> in,
        huffman_lit_weights_fse_wr_req_s: chan<HuffmanWeightsFseRamWrReq> out,
        huffman_lit_weights_fse_wr_resp_r: chan<HuffmanWeightsFseRamWrResp> in,
    ) {

        spawn LiteralsDecoder<
            ZSTD_HISTORY_BUFFER_SIZE_KB,
            INST_AXI_DATA_W, INST_AXI_ADDR_W, INST_AXI_ID_W, INST_AXI_DEST_W,
            INST_HUFFMAN_WEIGHTS_DPD_RAM_ADDR_WIDTH, INST_HUFFMAN_WEIGHTS_DPD_RAM_DATA_WIDTH, INST_HUFFMAN_WEIGHTS_DPD_RAM_NUM_PARTITIONS,
            INST_HUFFMAN_WEIGHTS_TMP_RAM_ADDR_WIDTH, INST_HUFFMAN_WEIGHTS_TMP_RAM_DATA_WIDTH, INST_HUFFMAN_WEIGHTS_TMP_RAM_NUM_PARTITIONS,
            INST_HUFFMAN_WEIGHTS_TMP2_RAM_ADDR_WIDTH, INST_HUFFMAN_WEIGHTS_TMP2_RAM_DATA_WIDTH, INST_HUFFMAN_WEIGHTS_TMP2_RAM_NUM_PARTITIONS,
            INST_HUFFMAN_WEIGHTS_FSE_RAM_ADDR_WIDTH, INST_HUFFMAN_WEIGHTS_FSE_RAM_DATA_WIDTH, INST_HUFFMAN_WEIGHTS_FSE_RAM_NUM_PARTITIONS,
            INST_HUFFMAN_WEIGHTS_RAM_ADDR_WIDTH, INST_HUFFMAN_WEIGHTS_RAM_DATA_WIDTH, INST_HUFFMAN_WEIGHTS_RAM_NUM_PARTITIONS,
            INST_HUFFMAN_PRESCAN_RAM_ADDR_WIDTH, INST_HUFFMAN_PRESCAN_RAM_DATA_WIDTH, INST_HUFFMAN_PRESCAN_RAM_NUM_PARTITIONS
        > (
            // AXI Literals Header Decoder (manager)
            lit_header_axi_ar_s, lit_header_axi_r_r,
            // AXI Raw Literals Decoder (manager)
            raw_lit_axi_ar_s, raw_lit_axi_r_r,
            // AXI Huffman Literals Decoder (manager)
            huffman_lit_axi_ar_s, huffman_lit_axi_r_r,
            // AXI Huffman Jump Table Decoder (manager)
            huffman_jump_table_axi_ar_s, huffman_jump_table_axi_r_r,
            // AXI Huffman Weights Header Decoder (manager)
            huffman_weights_header_axi_ar_s, huffman_weights_header_axi_r_r,
            // AXI Huffman Weights RAW Decoder (manager)
            huffman_weights_raw_axi_ar_s, huffman_weights_raw_axi_r_r,
            // AXI Huffman Weights FSE Decoder (manager)
            huffman_weights_fse_lookup_dec_axi_ar_s, huffman_weights_fse_lookup_dec_axi_r_r,
            huffman_weights_fse_decoder_dec_axi_ar_s, huffman_weights_fse_decoder_dec_axi_r_r,
            // Literals Decoder control
            lit_ctrl_req_r, lit_ctrl_resp_s, lit_ctrl_header_s,
            // Literals Decoder output control
            lit_buf_ctrl_r, lit_buf_out_s,
            // Internal memory
            rd_req_m0_s, rd_req_m1_s, rd_req_m2_s, rd_req_m3_s,
            rd_req_m4_s, rd_req_m5_s, rd_req_m6_s, rd_req_m7_s,
            rd_resp_m0_r, rd_resp_m1_r, rd_resp_m2_r, rd_resp_m3_r,
            rd_resp_m4_r, rd_resp_m5_r, rd_resp_m6_r, rd_resp_m7_r,
            wr_req_m0_s, wr_req_m1_s, wr_req_m2_s, wr_req_m3_s,
            wr_req_m4_s, wr_req_m5_s, wr_req_m6_s, wr_req_m7_s,
            wr_resp_m0_r, wr_resp_m1_r, wr_resp_m2_r, wr_resp_m3_r,
            wr_resp_m4_r, wr_resp_m5_r, wr_resp_m6_r, wr_resp_m7_r,
            // Huffman weights memory
            huffman_lit_weights_mem_rd_req_s, huffman_lit_weights_mem_rd_resp_r,
            huffman_lit_weights_mem_wr_req_s, huffman_lit_weights_mem_wr_resp_r,
            // Huffman prescan memory
            huffman_lit_prescan_mem_rd_req_s, huffman_lit_prescan_mem_rd_resp_r,
            huffman_lit_prescan_mem_wr_req_s, huffman_lit_prescan_mem_wr_resp_r,

            huffman_lit_weights_dpd_rd_req_s, huffman_lit_weights_dpd_rd_resp_r,
            huffman_lit_weights_dpd_wr_req_s, huffman_lit_weights_dpd_wr_resp_r,

            huffman_lit_weights_tmp_rd_req_s, huffman_lit_weights_tmp_rd_resp_r,
            huffman_lit_weights_tmp_wr_req_s, huffman_lit_weights_tmp_wr_resp_r,

            huffman_lit_weights_tmp2_rd_req_s, huffman_lit_weights_tmp2_rd_resp_r,
            huffman_lit_weights_tmp2_wr_req_s, huffman_lit_weights_tmp2_wr_resp_r,

            huffman_lit_weights_fse_rd_req_s, huffman_lit_weights_fse_rd_resp_r,
            huffman_lit_weights_fse_wr_req_s, huffman_lit_weights_fse_wr_resp_r,
        );
    }

    init {}

    next (state: ()) {}
}

const TEST_HISTORY_BUFFER_SIZE_KB:u32 = u32:1;

// Parameters for the AXI bus connecting LiteralsBlockHeaderDecoder,
// RawLiteralsDecoder and HuffmanLiteralsDecoder to the system memory
const TEST_AXI_RAM_ADDR_W:u32 = u32:32;
const TEST_AXI_RAM_DATA_W:u32 = u32:64;
const TEST_AXI_RAM_ID_W:u32 = u32:8;
const TEST_AXI_RAM_DEST_W:u32 = u32:8;

// Parameters for RamModels used for mocking the system memory for
// the LiteralsBlockHeaderDecoder, RawLiteralsDecoder and HuffmanLiteralsDecoder
const TEST_AXI_RAM_MODEL_DATA_WIDTH:u32 = TEST_AXI_RAM_DATA_W;
const TEST_AXI_RAM_MODEL_SIZE:u32 = u32:1024;
const TEST_AXI_RAM_MODEL_ADDR_WIDTH:u32 = std::clog2(TEST_AXI_RAM_MODEL_SIZE);
const TEST_AXI_RAM_MODEL_WORD_PARTITION_SIZE:u32 = u32:8;
const TEST_AXI_RAM_MODEL_NUM_PARTITIONS:u32 = ram::num_partitions(TEST_AXI_RAM_MODEL_WORD_PARTITION_SIZE, TEST_AXI_RAM_MODEL_DATA_WIDTH);
const TEST_AXI_RAM_MODEL_BASE_ADDR:u32 = u32:0;
const TEST_AXI_RAM_MODEL_SIMULTANEOUS_READ_WRITE_BEHAVIOR = ram::SimultaneousReadWriteBehavior::READ_BEFORE_WRITE;
const TEST_AXI_RAM_MODEL_INITIALIZED = true;
const TEST_AXI_RAM_MODEL_ASSERT_VALID_READ = true;
const TEST_AXI_RAM_MODEL_NUM = u32:1;

// Parameters for RamModels used for mocking the LiteralsBuffer internal memory
const TEST_LITERALS_BUFFER_RAM_MODEL_DATA_WIDTH:u32 = literals_buffer::RAM_DATA_WIDTH;
const TEST_LITERALS_BUFFER_RAM_MODEL_SIZE:u32 = parallel_rams::ram_size(TEST_HISTORY_BUFFER_SIZE_KB);
const TEST_LITERALS_BUFFER_RAM_MODEL_ADDR_WIDTH:u32 = parallel_rams::ram_addr_width(TEST_HISTORY_BUFFER_SIZE_KB);
const TEST_LITERALS_BUFFER_RAM_MODEL_WORD_PARTITION_SIZE:u32 = literals_buffer::RAM_WORD_PARTITION_SIZE;
const TEST_LITERALS_BUFFER_RAM_MODEL_NUM_PARTITIONS:u32 = literals_buffer::RAM_NUM_PARTITIONS;
const TEST_LITERALS_BUFFER_RAM_MODEL_SIMULTANEOUS_READ_WRITE_BEHAVIOR = ram::SimultaneousReadWriteBehavior::READ_BEFORE_WRITE;
const TEST_LITERALS_BUFFER_RAM_MODEL_INITIALIZED = true;
const TEST_LITERALS_BUFFER_RAM_MODEL_ASSERT_VALID_READ = true;
const TEST_LITERALS_BUFFER_RAM_MODEL_NUM = literals_buffer::RAM_NUM;

// Parameters for RamModels used for mocking the HuffmanLiteralsDecoder prescan weights memory
const TEST_HUFFMAN_PRESCAN_RAM_MODEL_DATA_WIDTH:u32 = huffman_literals_dec::TEST_PRESCAN_RAM_DATA_WIDTH;
const TEST_HUFFMAN_PRESCAN_RAM_MODEL_SIZE:u32 = huffman_literals_dec::TEST_PRESCAN_RAM_SIZE;
const TEST_HUFFMAN_PRESCAN_RAM_MODEL_ADDR_WIDTH:u32 = huffman_literals_dec::TEST_PRESCAN_RAM_ADDR_WIDTH;
const TEST_HUFFMAN_PRESCAN_RAM_MODEL_WORD_PARTITION_SIZE:u32 = huffman_literals_dec::TEST_PRESCAN_WORD_PARTITION_SIZE;
const TEST_HUFFMAN_PRESCAN_RAM_MODEL_NUM_PARTITIONS:u32 = huffman_literals_dec::TEST_PRESCAN_RAM_NUM_PARTITIONS;
const TEST_HUFFMAN_PRESCAN_RAM_MODEL_SIMULTANEOUS_READ_WRITE_BEHAVIOR = ram::SimultaneousReadWriteBehavior::READ_BEFORE_WRITE;
const TEST_HUFFMAN_PRESCAN_RAM_MODEL_INITIALIZED = true;
const TEST_HUFFMAN_PRESCAN_RAM_MODEL_ASSERT_VALID_READ = true;

// Parameters for RamModels used for mocking the HuffmanLiteralsDecoder internal weights memory
const TEST_HUFFMAN_WEIGHTS_RAM_MODEL_DATA_WIDTH:u32 = huffman_literals_dec::TEST_WEIGHTS_RAM_DATA_WIDTH;
const TEST_HUFFMAN_WEIGHTS_RAM_MODEL_SIZE:u32 = huffman_literals_dec::TEST_WEIGHTS_RAM_SIZE;
const TEST_HUFFMAN_WEIGHTS_RAM_MODEL_ADDR_WIDTH:u32 = huffman_literals_dec::TEST_WEIGHTS_RAM_ADDR_WIDTH;
const TEST_HUFFMAN_WEIGHTS_RAM_MODEL_WORD_PARTITION_SIZE:u32 = huffman_literals_dec::TEST_WEIGHTS_WORD_PARTITION_SIZE;
const TEST_HUFFMAN_WEIGHTS_RAM_MODEL_NUM_PARTITIONS:u32 = huffman_literals_dec::TEST_WEIGHTS_RAM_NUM_PARTITIONS;
const TEST_HUFFMAN_WEIGHTS_RAM_MODEL_SIMULTANEOUS_READ_WRITE_BEHAVIOR = ram::SimultaneousReadWriteBehavior::READ_BEFORE_WRITE;
const TEST_HUFFMAN_WEIGHTS_RAM_MODEL_INITIALIZED = true;
const TEST_HUFFMAN_WEIGHTS_RAM_MODEL_ASSERT_VALID_READ = true;

const TEST_HUFFMAN_WEIGHTS_DPD_RAM_MODEL_DATA_WIDTH = u32:16;
const TEST_HUFFMAN_WEIGHTS_DPD_RAM_MODEL_SIZE = u32:256;
const TEST_HUFFMAN_WEIGHTS_DPD_RAM_MODEL_ADDR_WIDTH = std::clog2(TEST_HUFFMAN_WEIGHTS_DPD_RAM_MODEL_SIZE);
const TEST_HUFFMAN_WEIGHTS_DPD_RAM_MODEL_WORD_PARTITION_SIZE = TEST_HUFFMAN_WEIGHTS_DPD_RAM_MODEL_DATA_WIDTH;
const TEST_HUFFMAN_WEIGHTS_DPD_RAM_MODEL_NUM_PARTITIONS = ram::num_partitions(
    TEST_HUFFMAN_WEIGHTS_DPD_RAM_MODEL_WORD_PARTITION_SIZE, TEST_HUFFMAN_WEIGHTS_DPD_RAM_MODEL_DATA_WIDTH);
const TEST_HUFFMAN_WEIGHTS_DPD_RAM_MODEL_SIMULTANEOUS_READ_WRITE_BEHAVIOR = ram::SimultaneousReadWriteBehavior::READ_BEFORE_WRITE;
const TEST_HUFFMAN_WEIGHTS_DPD_RAM_MODEL_INITIALIZED = true;
const TEST_HUFFMAN_WEIGHTS_DPD_RAM_MODEL_ASSERT_VALID_READ = true;

const TEST_HUFFMAN_WEIGHTS_FSE_RAM_MODEL_DATA_WIDTH = u32:32;
const TEST_HUFFMAN_WEIGHTS_FSE_RAM_MODEL_SIZE = u32:256;
const TEST_HUFFMAN_WEIGHTS_FSE_RAM_MODEL_ADDR_WIDTH = std::clog2(TEST_HUFFMAN_WEIGHTS_FSE_RAM_MODEL_SIZE);
const TEST_HUFFMAN_WEIGHTS_FSE_RAM_MODEL_WORD_PARTITION_SIZE = TEST_HUFFMAN_WEIGHTS_FSE_RAM_MODEL_DATA_WIDTH / u32:3;
const TEST_HUFFMAN_WEIGHTS_FSE_RAM_MODEL_NUM_PARTITIONS = ram::num_partitions(
    TEST_HUFFMAN_WEIGHTS_FSE_RAM_MODEL_WORD_PARTITION_SIZE, TEST_HUFFMAN_WEIGHTS_FSE_RAM_MODEL_DATA_WIDTH);
const TEST_HUFFMAN_WEIGHTS_FSE_RAM_MODEL_SIMULTANEOUS_READ_WRITE_BEHAVIOR = ram::SimultaneousReadWriteBehavior::READ_BEFORE_WRITE;
const TEST_HUFFMAN_WEIGHTS_FSE_RAM_MODEL_INITIALIZED = true;
const TEST_HUFFMAN_WEIGHTS_FSE_RAM_MODEL_ASSERT_VALID_READ = true;

const TEST_HUFFMAN_WEIGHTS_TMP_RAM_MODEL_DATA_WIDTH = u32:16;
const TEST_HUFFMAN_WEIGHTS_TMP_RAM_MODEL_SIZE = u32:256;
const TEST_HUFFMAN_WEIGHTS_TMP_RAM_MODEL_ADDR_WIDTH = std::clog2(TEST_HUFFMAN_WEIGHTS_TMP_RAM_MODEL_SIZE);
const TEST_HUFFMAN_WEIGHTS_TMP_RAM_MODEL_WORD_PARTITION_SIZE = TEST_HUFFMAN_WEIGHTS_TMP_RAM_MODEL_DATA_WIDTH;
const TEST_HUFFMAN_WEIGHTS_TMP_RAM_MODEL_NUM_PARTITIONS = ram::num_partitions(
    TEST_HUFFMAN_WEIGHTS_TMP_RAM_MODEL_WORD_PARTITION_SIZE, TEST_HUFFMAN_WEIGHTS_TMP_RAM_MODEL_DATA_WIDTH);
const TEST_HUFFMAN_WEIGHTS_TMP_RAM_MODEL_SIMULTANEOUS_READ_WRITE_BEHAVIOR = ram::SimultaneousReadWriteBehavior::READ_BEFORE_WRITE;
const TEST_HUFFMAN_WEIGHTS_TMP_RAM_MODEL_INITIALIZED = true;
const TEST_HUFFMAN_WEIGHTS_TMP_RAM_MODEL_ASSERT_VALID_READ = true;

const TEST_HUFFMAN_WEIGHTS_TMP2_RAM_MODEL_DATA_WIDTH = u32:8;
const TEST_HUFFMAN_WEIGHTS_TMP2_RAM_MODEL_SIZE = u32:512;
const TEST_HUFFMAN_WEIGHTS_TMP2_RAM_MODEL_ADDR_WIDTH = std::clog2(TEST_HUFFMAN_WEIGHTS_TMP2_RAM_MODEL_SIZE);
const TEST_HUFFMAN_WEIGHTS_TMP2_RAM_MODEL_WORD_PARTITION_SIZE = TEST_HUFFMAN_WEIGHTS_TMP2_RAM_MODEL_DATA_WIDTH;
const TEST_HUFFMAN_WEIGHTS_TMP2_RAM_MODEL_NUM_PARTITIONS = ram::num_partitions(
    TEST_HUFFMAN_WEIGHTS_TMP2_RAM_MODEL_WORD_PARTITION_SIZE, TEST_HUFFMAN_WEIGHTS_TMP2_RAM_MODEL_DATA_WIDTH);
const TEST_HUFFMAN_WEIGHTS_TMP2_RAM_MODEL_SIMULTANEOUS_READ_WRITE_BEHAVIOR = ram::SimultaneousReadWriteBehavior::READ_BEFORE_WRITE;
const TEST_HUFFMAN_WEIGHTS_TMP2_RAM_MODEL_INITIALIZED = true;
const TEST_HUFFMAN_WEIGHTS_TMP2_RAM_MODEL_ASSERT_VALID_READ = true;

#[test_proc]
proc LiteralsDecoder_test {
    // LiteralsBuffer internal memory
    type LiteralsBufferRamRdReq = ram::ReadReq<TEST_LITERALS_BUFFER_RAM_MODEL_ADDR_WIDTH, TEST_LITERALS_BUFFER_RAM_MODEL_NUM_PARTITIONS>;
    type LiteralsBufferRamRdResp = ram::ReadResp<TEST_LITERALS_BUFFER_RAM_MODEL_DATA_WIDTH>;
    type LiteralsBufferRamWrReq = ram::WriteReq<TEST_LITERALS_BUFFER_RAM_MODEL_ADDR_WIDTH, TEST_LITERALS_BUFFER_RAM_MODEL_DATA_WIDTH, TEST_LITERALS_BUFFER_RAM_MODEL_NUM_PARTITIONS>;
    type LiteralsBufferRamWrResp = ram::WriteResp;

    // System bus
    type MemAxiR = axi::AxiR<TEST_AXI_RAM_DATA_W, TEST_AXI_RAM_ID_W>;
    type MemAxiAr = axi::AxiAr<TEST_AXI_RAM_ADDR_W, TEST_AXI_RAM_ID_W>;

    // System bus external memory
    type AxiRamRdReq = ram::ReadReq<TEST_AXI_RAM_MODEL_ADDR_WIDTH, TEST_AXI_RAM_MODEL_NUM_PARTITIONS>;
    type AxiRamRdResp = ram::ReadResp<TEST_AXI_RAM_MODEL_DATA_WIDTH>;
    type AxiRamWrReq = ram::WriteReq<TEST_AXI_RAM_MODEL_ADDR_WIDTH, TEST_AXI_RAM_MODEL_DATA_WIDTH, TEST_AXI_RAM_MODEL_NUM_PARTITIONS>;
    type AxiRamWrResp = ram::WriteResp;

    // Huffman weights internal memory
    type HuffmanWeightsRamRdReq   = ram::ReadReq<TEST_HUFFMAN_WEIGHTS_RAM_MODEL_ADDR_WIDTH, TEST_HUFFMAN_WEIGHTS_RAM_MODEL_NUM_PARTITIONS>;
    type HuffmanWeightsRamRdResp  = ram::ReadResp<TEST_HUFFMAN_WEIGHTS_RAM_MODEL_DATA_WIDTH>;
    type HuffmanWeightsRamWrReq  = ram::WriteReq<TEST_HUFFMAN_WEIGHTS_RAM_MODEL_ADDR_WIDTH, TEST_HUFFMAN_WEIGHTS_RAM_MODEL_DATA_WIDTH, TEST_HUFFMAN_WEIGHTS_RAM_MODEL_NUM_PARTITIONS>;
    type HuffmanWeightsRamWrResp = ram::WriteResp;

    // Huffman prescan internal memory
    type HuffmanPrescanRamRdReq   = ram::ReadReq<TEST_HUFFMAN_PRESCAN_RAM_MODEL_ADDR_WIDTH, TEST_HUFFMAN_PRESCAN_RAM_MODEL_NUM_PARTITIONS>;
    type HuffmanPrescanRamRdResp  = ram::ReadResp<TEST_HUFFMAN_PRESCAN_RAM_MODEL_DATA_WIDTH>;
    type HuffmanPrescanRamWrReq  = ram::WriteReq<TEST_HUFFMAN_PRESCAN_RAM_MODEL_ADDR_WIDTH, TEST_HUFFMAN_PRESCAN_RAM_MODEL_DATA_WIDTH, TEST_HUFFMAN_PRESCAN_RAM_MODEL_NUM_PARTITIONS>;
    type HuffmanPrescanRamWrResp = ram::WriteResp;

    type HuffmanWeightsDpdRamRdReq = ram::ReadReq<TEST_HUFFMAN_WEIGHTS_DPD_RAM_MODEL_ADDR_WIDTH, TEST_HUFFMAN_WEIGHTS_DPD_RAM_MODEL_NUM_PARTITIONS>;
    type HuffmanWeightsDpdRamRdResp = ram::ReadResp<TEST_HUFFMAN_WEIGHTS_DPD_RAM_MODEL_DATA_WIDTH>;
    type HuffmanWeightsDpdRamWrReq = ram::WriteReq<TEST_HUFFMAN_WEIGHTS_DPD_RAM_MODEL_ADDR_WIDTH, TEST_HUFFMAN_WEIGHTS_DPD_RAM_MODEL_DATA_WIDTH, TEST_HUFFMAN_WEIGHTS_DPD_RAM_MODEL_NUM_PARTITIONS>;
    type HuffmanWeightsDpdRamWrResp = ram::WriteResp;

    type HuffmanWeightsTmpRamRdReq = ram::ReadReq<TEST_HUFFMAN_WEIGHTS_TMP_RAM_MODEL_ADDR_WIDTH, TEST_HUFFMAN_WEIGHTS_TMP_RAM_MODEL_NUM_PARTITIONS>;
    type HuffmanWeightsTmpRamRdResp = ram::ReadResp<TEST_HUFFMAN_WEIGHTS_TMP_RAM_MODEL_DATA_WIDTH>;
    type HuffmanWeightsTmpRamWrReq = ram::WriteReq<TEST_HUFFMAN_WEIGHTS_TMP_RAM_MODEL_ADDR_WIDTH, TEST_HUFFMAN_WEIGHTS_TMP_RAM_MODEL_DATA_WIDTH, TEST_HUFFMAN_WEIGHTS_TMP_RAM_MODEL_NUM_PARTITIONS>;
    type HuffmanWeightsTmpRamWrResp = ram::WriteResp;

    type HuffmanWeightsTmp2RamRdReq = ram::ReadReq<TEST_HUFFMAN_WEIGHTS_TMP2_RAM_MODEL_ADDR_WIDTH, TEST_HUFFMAN_WEIGHTS_TMP2_RAM_MODEL_NUM_PARTITIONS>;
    type HuffmanWeightsTmp2RamRdResp = ram::ReadResp<TEST_HUFFMAN_WEIGHTS_TMP2_RAM_MODEL_DATA_WIDTH>;
    type HuffmanWeightsTmp2RamWrReq = ram::WriteReq<TEST_HUFFMAN_WEIGHTS_TMP2_RAM_MODEL_ADDR_WIDTH, TEST_HUFFMAN_WEIGHTS_TMP2_RAM_MODEL_DATA_WIDTH, TEST_HUFFMAN_WEIGHTS_TMP2_RAM_MODEL_NUM_PARTITIONS>;
    type HuffmanWeightsTmp2RamWrResp = ram::WriteResp;

    type HuffmanWeightsFseRamRdReq = ram::ReadReq<TEST_HUFFMAN_WEIGHTS_FSE_RAM_MODEL_ADDR_WIDTH, TEST_HUFFMAN_WEIGHTS_FSE_RAM_MODEL_NUM_PARTITIONS>;
    type HuffmanWeightsFseRamRdResp = ram::ReadResp<TEST_HUFFMAN_WEIGHTS_FSE_RAM_MODEL_DATA_WIDTH>;
    type HuffmanWeightsFseRamWrReq = ram::WriteReq<TEST_HUFFMAN_WEIGHTS_FSE_RAM_MODEL_ADDR_WIDTH, TEST_HUFFMAN_WEIGHTS_FSE_RAM_MODEL_DATA_WIDTH, TEST_HUFFMAN_WEIGHTS_FSE_RAM_MODEL_NUM_PARTITIONS>;
    type HuffmanWeightsFseRamWrResp = ram::WriteResp;

    // Control and output
    type CtrlReq = LiteralsDecoderCtrlReq<TEST_AXI_RAM_ADDR_W>;
    type CtrlResp = LiteralsDecoderCtrlResp;
    type CtrlStatus = LiteralsDecoderCtrlStatus;
    type BufferCtrl = common::LiteralsBufferCtrl;
    type BufferOut = common::SequenceExecutorPacket<common::SYMBOL_WIDTH>;

    type AxiRamData = uN[TEST_AXI_RAM_MODEL_DATA_WIDTH];
    type AxiRamAddr = uN[TEST_AXI_RAM_MODEL_ADDR_WIDTH];
    type AxiRamMask = uN[TEST_AXI_RAM_MODEL_NUM_PARTITIONS];

    type AxiAddr = uN[TEST_AXI_RAM_ADDR_W];

    type HeaderResp = literals_block_header_dec::LiteralsHeaderDecoderResp;

    terminator: chan<bool> out;

    // Literals Decoder control
    ctrl_req_s: chan<CtrlReq> out;
    ctrl_resp_r: chan<CtrlResp> in;
    ctrl_header_r: chan<HeaderResp> in;

    // Output control
    buf_ctrl_s: chan<BufferCtrl> out;
    buf_out_r: chan<BufferOut> in;

    print_start_s: chan<()> out;
    print_finish_r: chan<()> in;

    ram_wr_req_header_s : chan<AxiRamWrReq> out;
    ram_wr_resp_header_r : chan<AxiRamWrResp> in;
    ram_wr_req_raw_s : chan<AxiRamWrReq> out;
    ram_wr_resp_raw_r : chan<AxiRamWrResp> in;
    ram_wr_req_huffman_s : chan<AxiRamWrReq> out;
    ram_wr_resp_huffman_r : chan<AxiRamWrResp> in;
    ram_wr_req_huffman_jump_table_s : chan<AxiRamWrReq> out;
    ram_wr_resp_huffman_jump_table_r : chan<AxiRamWrResp> in;
    ram_wr_req_huffman_weights_header_s : chan<AxiRamWrReq> out;
    ram_wr_resp_huffman_weights_header_r : chan<AxiRamWrResp> in;
    ram_wr_req_huffman_weights_raw_s : chan<AxiRamWrReq> out;
    ram_wr_resp_huffman_weights_raw_r : chan<AxiRamWrResp> in;
    ram_wr_req_huffman_weights_fse_lookup_dec_s : chan<AxiRamWrReq> out;
    ram_wr_resp_huffman_weights_fse_lookup_dec_r : chan<AxiRamWrResp> in;
    ram_wr_req_huffman_weights_fse_decoder_dec_s : chan<AxiRamWrReq> out;
    ram_wr_resp_huffman_weights_fse_decoder_dec_r : chan<AxiRamWrResp> in;

    config (terminator: chan<bool> out) {
        let (lit_header_axi_ar_s, lit_header_axi_ar_r) = chan<MemAxiAr>("lit_header_axi_ar");
        let (lit_header_axi_r_s, lit_header_axi_r_r) = chan<MemAxiR>("lit_header_axi_r");

        let (raw_lit_axi_ar_s, raw_lit_axi_ar_r) = chan<MemAxiAr>("raw_lit_axi_ar");
        let (raw_lit_axi_r_s, raw_lit_axi_r_r) = chan<MemAxiR>("raw_lit_axi_r");

        let (huffman_lit_axi_ar_s, huffman_lit_axi_ar_r) = chan<MemAxiAr>("huffman_lit_axi_ar");
        let (huffman_lit_axi_r_s, huffman_lit_axi_r_r) = chan<MemAxiR>("huffman_lit_axi_r");

        let (huffman_jump_table_axi_ar_s, huffman_jump_table_axi_ar_r) = chan<MemAxiAr>("huffman_jump_table_axi_ar");
        let (huffman_jump_table_axi_r_s, huffman_jump_table_axi_r_r) = chan<MemAxiR>("huffman_jump_table_axi_r");

        let (huffman_weights_header_axi_ar_s, huffman_weights_header_axi_ar_r) = chan<MemAxiAr>("huffman_weights_header_axi_ar");
        let (huffman_weights_header_axi_r_s, huffman_weights_header_axi_r_r) = chan<MemAxiR>("huffman_weights_header_axi_r");

        let (huffman_weights_raw_axi_ar_s, huffman_weights_raw_axi_ar_r) = chan<MemAxiAr>("huffman_weights_raw_axi_ar");
        let (huffman_weights_raw_axi_r_s, huffman_weights_raw_axi_r_r) = chan<MemAxiR>("huffman_weights_raw_axi_r");

        let (huffman_weights_fse_lookup_dec_axi_ar_s, huffman_weights_fse_lookup_dec_axi_ar_r) = chan<MemAxiAr>("huffman_weights_fse_lookup_dec_axi_ar");
        let (huffman_weights_fse_lookup_dec_axi_r_s, huffman_weights_fse_lookup_dec_axi_r_r) = chan<MemAxiR>("huffman_weights_fse_lookup_dec_axi_r_r");

        let (huffman_weights_fse_decoder_dec_axi_ar_s, huffman_weights_fse_decoder_dec_axi_ar_r) = chan<MemAxiAr>("huffman_weights_fse_decoder_dec_axi_ar");
        let (huffman_weights_fse_decoder_dec_axi_r_s, huffman_weights_fse_decoder_dec_axi_r_r) = chan<MemAxiR>("huffman_weights_fse_decoder_dec_axi_r");

        let (ctrl_req_s, ctrl_req_r) = chan<CtrlReq>("ctrl_req");
        let (ctrl_resp_s, ctrl_resp_r) = chan<CtrlResp>("ctrl_resp");
        let (ctrl_header_s, ctrl_header_r) = chan<HeaderResp>("ctrl_header");

        let (buf_ctrl_s, buf_ctrl_r) = chan<BufferCtrl>("buf_ctrl");
        let (buf_out_s, buf_out_r) = chan<BufferOut>("buf_out");

        let (print_start_s, print_start_r) = chan<()>("print_start");
        let (print_finish_s, print_finish_r) = chan<()>("print_finish");

        let (ram_rd_req_s,  ram_rd_req_r) = chan<LiteralsBufferRamRdReq>[literals_buffer::RAM_NUM]("ram_rd_req");
        let (ram_rd_resp_s, ram_rd_resp_r) = chan<LiteralsBufferRamRdResp>[literals_buffer::RAM_NUM]("ram_rd_resp");
        let (ram_wr_req_s,  ram_wr_req_r) = chan<LiteralsBufferRamWrReq>[literals_buffer::RAM_NUM]("ram_wr_req");
        let (ram_wr_resp_s, ram_wr_resp_r) = chan<LiteralsBufferRamWrResp>[literals_buffer::RAM_NUM]("ram_wr_resp");

        let (huffman_lit_weights_mem_rd_req_s, huffman_lit_weights_mem_rd_req_r) = chan<HuffmanWeightsRamRdReq>("huffman_lit_weights_mem_rd_req");
        let (huffman_lit_weights_mem_rd_resp_s, huffman_lit_weights_mem_rd_resp_r) = chan<HuffmanWeightsRamRdResp>("huffman_lit_weights_mem_rd_resp");
        let (huffman_lit_weights_mem_wr_req_s, huffman_lit_weights_mem_wr_req_r) = chan<HuffmanWeightsRamWrReq>("huffman_lit_weights_mem_wr_req");
        let (huffman_lit_weights_mem_wr_resp_s, huffman_lit_weights_mem_wr_resp_r) = chan<HuffmanWeightsRamWrResp>("huffman_lit_weights_mem_wr_resp");

        let (huffman_lit_prescan_mem_rd_req_s, huffman_lit_prescan_mem_rd_req_r) = chan<HuffmanPrescanRamRdReq>("huffman_lit_prescan_mem_rd_req");
        let (huffman_lit_prescan_mem_rd_resp_s, huffman_lit_prescan_mem_rd_resp_r) = chan<HuffmanPrescanRamRdResp>("huffman_lit_prescan_mem_rd_resp");
        let (huffman_lit_prescan_mem_wr_req_s, huffman_lit_prescan_mem_wr_req_r) = chan<HuffmanPrescanRamWrReq>("huffman_lit_prescan_mem_wr_req");
        let (huffman_lit_prescan_mem_wr_resp_s, huffman_lit_prescan_mem_wr_resp_r) = chan<HuffmanPrescanRamWrResp>("huffman_lit_prescan_mem_wr_resp");

        let (huffman_lit_weights_dpd_rd_req_s, huffman_lit_weights_dpd_rd_req_r) = chan<HuffmanWeightsDpdRamRdReq>("huffman_lit_weights_dpd_rd_req");
        let (huffman_lit_weights_dpd_rd_resp_s, huffman_lit_weights_dpd_rd_resp_r) = chan<HuffmanWeightsDpdRamRdResp>("huffman_lit_weights_dpd_rd_resp");
        let (huffman_lit_weights_dpd_wr_req_s, huffman_lit_weights_dpd_wr_req_r) = chan<HuffmanWeightsDpdRamWrReq>("huffman_lit_weights_dpd_wr_req");
        let (huffman_lit_weights_dpd_wr_resp_s, huffman_lit_weights_dpd_wr_resp_r) = chan<HuffmanWeightsDpdRamWrResp>("huffman_lit_weights_dpd_wr_resp");

        let (huffman_lit_weights_tmp_rd_req_s, huffman_lit_weights_tmp_rd_req_r) = chan<HuffmanWeightsTmpRamRdReq>("huffman_lit_weights_tmp_rd_req");
        let (huffman_lit_weights_tmp_rd_resp_s, huffman_lit_weights_tmp_rd_resp_r) = chan<HuffmanWeightsTmpRamRdResp>("huffman_lit_weights_tmp_rd_resp");
        let (huffman_lit_weights_tmp_wr_req_s, huffman_lit_weights_tmp_wr_req_r) = chan<HuffmanWeightsTmpRamWrReq>("huffman_lit_weights_tmp_wr_req");
        let (huffman_lit_weights_tmp_wr_resp_s, huffman_lit_weights_tmp_wr_resp_r) = chan<HuffmanWeightsTmpRamWrResp>("huffman_lit_weights_tmp_wr_resp");

        let (huffman_lit_weights_tmp2_rd_req_s, huffman_lit_weights_tmp2_rd_req_r) = chan<HuffmanWeightsTmp2RamRdReq>("huffman_lit_weights_tmp2_rd_req");
        let (huffman_lit_weights_tmp2_rd_resp_s, huffman_lit_weights_tmp2_rd_resp_r) = chan<HuffmanWeightsTmp2RamRdResp>("huffman_lit_weights_tmp2_rd_resp");
        let (huffman_lit_weights_tmp2_wr_req_s, huffman_lit_weights_tmp2_wr_req_r) = chan<HuffmanWeightsTmp2RamWrReq>("huffman_lit_weights_tmp2_wr_req");
        let (huffman_lit_weights_tmp2_wr_resp_s, huffman_lit_weights_tmp2_wr_resp_r) = chan<HuffmanWeightsTmp2RamWrResp>("huffman_lit_weights_tmp2_wr_resp");

        let (huffman_lit_weights_fse_rd_req_s, huffman_lit_weights_fse_rd_req_r) = chan<HuffmanWeightsFseRamRdReq>("huffman_lit_weights_fse_rd_req");
        let (huffman_lit_weights_fse_rd_resp_s, huffman_lit_weights_fse_rd_resp_r) = chan<HuffmanWeightsFseRamRdResp>("huffman_lit_weights_fse_rd_resp");
        let (huffman_lit_weights_fse_wr_req_s, huffman_lit_weights_fse_wr_req_r) = chan<HuffmanWeightsFseRamWrReq>("huffman_lit_weights_fse_wr_req");
        let (huffman_lit_weights_fse_wr_resp_s, huffman_lit_weights_fse_wr_resp_r) = chan<HuffmanWeightsFseRamWrResp>("huffman_lit_weights_fse_wr_resp");

        spawn LiteralsDecoder<
            TEST_HISTORY_BUFFER_SIZE_KB,
            TEST_AXI_RAM_DATA_W, TEST_AXI_RAM_ADDR_W, TEST_AXI_RAM_ID_W, TEST_AXI_RAM_DEST_W,
            TEST_HUFFMAN_WEIGHTS_DPD_RAM_MODEL_ADDR_WIDTH, TEST_HUFFMAN_WEIGHTS_DPD_RAM_MODEL_DATA_WIDTH, TEST_HUFFMAN_WEIGHTS_DPD_RAM_MODEL_NUM_PARTITIONS,
            TEST_HUFFMAN_WEIGHTS_TMP_RAM_MODEL_ADDR_WIDTH, TEST_HUFFMAN_WEIGHTS_TMP_RAM_MODEL_DATA_WIDTH, TEST_HUFFMAN_WEIGHTS_TMP_RAM_MODEL_NUM_PARTITIONS,
            TEST_HUFFMAN_WEIGHTS_TMP2_RAM_MODEL_ADDR_WIDTH, TEST_HUFFMAN_WEIGHTS_TMP2_RAM_MODEL_DATA_WIDTH, TEST_HUFFMAN_WEIGHTS_TMP2_RAM_MODEL_NUM_PARTITIONS,
            TEST_HUFFMAN_WEIGHTS_FSE_RAM_MODEL_ADDR_WIDTH, TEST_HUFFMAN_WEIGHTS_FSE_RAM_MODEL_DATA_WIDTH, TEST_HUFFMAN_WEIGHTS_FSE_RAM_MODEL_NUM_PARTITIONS,
            TEST_HUFFMAN_WEIGHTS_RAM_MODEL_ADDR_WIDTH, TEST_HUFFMAN_WEIGHTS_RAM_MODEL_DATA_WIDTH, TEST_HUFFMAN_WEIGHTS_RAM_MODEL_NUM_PARTITIONS,
            TEST_HUFFMAN_PRESCAN_RAM_MODEL_ADDR_WIDTH, TEST_HUFFMAN_PRESCAN_RAM_MODEL_DATA_WIDTH, TEST_HUFFMAN_PRESCAN_RAM_MODEL_NUM_PARTITIONS,
        > (
            lit_header_axi_ar_s, lit_header_axi_r_r,
            raw_lit_axi_ar_s, raw_lit_axi_r_r,
            huffman_lit_axi_ar_s, huffman_lit_axi_r_r,
            huffman_jump_table_axi_ar_s, huffman_jump_table_axi_r_r,
            huffman_weights_header_axi_ar_s, huffman_weights_header_axi_r_r,
            huffman_weights_raw_axi_ar_s, huffman_weights_raw_axi_r_r,
            huffman_weights_fse_lookup_dec_axi_ar_s, huffman_weights_fse_lookup_dec_axi_r_r,
            huffman_weights_fse_decoder_dec_axi_ar_s, huffman_weights_fse_decoder_dec_axi_r_r,
            ctrl_req_r, ctrl_resp_s, ctrl_header_s,
            buf_ctrl_r, buf_out_s,
            ram_rd_req_s[0], ram_rd_req_s[1], ram_rd_req_s[2], ram_rd_req_s[3],
            ram_rd_req_s[4], ram_rd_req_s[5], ram_rd_req_s[6], ram_rd_req_s[7],
            ram_rd_resp_r[0], ram_rd_resp_r[1], ram_rd_resp_r[2], ram_rd_resp_r[3],
            ram_rd_resp_r[4], ram_rd_resp_r[5], ram_rd_resp_r[6], ram_rd_resp_r[7],
            ram_wr_req_s[0], ram_wr_req_s[1], ram_wr_req_s[2], ram_wr_req_s[3],
            ram_wr_req_s[4], ram_wr_req_s[5], ram_wr_req_s[6], ram_wr_req_s[7],
            ram_wr_resp_r[0], ram_wr_resp_r[1], ram_wr_resp_r[2], ram_wr_resp_r[3],
            ram_wr_resp_r[4], ram_wr_resp_r[5], ram_wr_resp_r[6], ram_wr_resp_r[7],
            huffman_lit_weights_mem_rd_req_s, huffman_lit_weights_mem_rd_resp_r,
            huffman_lit_weights_mem_wr_req_s, huffman_lit_weights_mem_wr_resp_r,
            huffman_lit_prescan_mem_rd_req_s, huffman_lit_prescan_mem_rd_resp_r,
            huffman_lit_prescan_mem_wr_req_s, huffman_lit_prescan_mem_wr_resp_r,
            huffman_lit_weights_dpd_rd_req_s, huffman_lit_weights_dpd_rd_resp_r,
            huffman_lit_weights_dpd_wr_req_s, huffman_lit_weights_dpd_wr_resp_r,
            huffman_lit_weights_tmp_rd_req_s, huffman_lit_weights_tmp_rd_resp_r,
            huffman_lit_weights_tmp_wr_req_s, huffman_lit_weights_tmp_wr_resp_r,
            huffman_lit_weights_tmp2_rd_req_s, huffman_lit_weights_tmp2_rd_resp_r,
            huffman_lit_weights_tmp2_wr_req_s, huffman_lit_weights_tmp2_wr_resp_r,
            huffman_lit_weights_fse_rd_req_s, huffman_lit_weights_fse_rd_resp_r,
            huffman_lit_weights_fse_wr_req_s, huffman_lit_weights_fse_wr_resp_r,
        );

        spawn ram_printer::RamPrinter<
            TEST_LITERALS_BUFFER_RAM_MODEL_DATA_WIDTH,
            TEST_LITERALS_BUFFER_RAM_MODEL_SIZE,
            TEST_LITERALS_BUFFER_RAM_MODEL_NUM_PARTITIONS,
            TEST_LITERALS_BUFFER_RAM_MODEL_ADDR_WIDTH,
            TEST_LITERALS_BUFFER_RAM_MODEL_NUM
        > (
            print_start_r, print_finish_s, ram_rd_req_s, ram_rd_resp_r
        );

        spawn ram::RamModel<
            TEST_LITERALS_BUFFER_RAM_MODEL_DATA_WIDTH,
            TEST_LITERALS_BUFFER_RAM_MODEL_SIZE,
            TEST_LITERALS_BUFFER_RAM_MODEL_WORD_PARTITION_SIZE,
            TEST_LITERALS_BUFFER_RAM_MODEL_SIMULTANEOUS_READ_WRITE_BEHAVIOR,
            TEST_LITERALS_BUFFER_RAM_MODEL_INITIALIZED
        > (
            ram_rd_req_r[0], ram_rd_resp_s[0], ram_wr_req_r[0], ram_wr_resp_s[0]
        );
        spawn ram::RamModel<
            TEST_LITERALS_BUFFER_RAM_MODEL_DATA_WIDTH,
            TEST_LITERALS_BUFFER_RAM_MODEL_SIZE,
            TEST_LITERALS_BUFFER_RAM_MODEL_WORD_PARTITION_SIZE,
            TEST_LITERALS_BUFFER_RAM_MODEL_SIMULTANEOUS_READ_WRITE_BEHAVIOR,
            TEST_LITERALS_BUFFER_RAM_MODEL_INITIALIZED
        > (
            ram_rd_req_r[1], ram_rd_resp_s[1], ram_wr_req_r[1], ram_wr_resp_s[1]
        );
        spawn ram::RamModel<
            TEST_LITERALS_BUFFER_RAM_MODEL_DATA_WIDTH,
            TEST_LITERALS_BUFFER_RAM_MODEL_SIZE,
            TEST_LITERALS_BUFFER_RAM_MODEL_WORD_PARTITION_SIZE,
            TEST_LITERALS_BUFFER_RAM_MODEL_SIMULTANEOUS_READ_WRITE_BEHAVIOR,
            TEST_LITERALS_BUFFER_RAM_MODEL_INITIALIZED
        > (
            ram_rd_req_r[2], ram_rd_resp_s[2], ram_wr_req_r[2], ram_wr_resp_s[2]
        );
        spawn ram::RamModel<
            TEST_LITERALS_BUFFER_RAM_MODEL_DATA_WIDTH,
            TEST_LITERALS_BUFFER_RAM_MODEL_SIZE,
            TEST_LITERALS_BUFFER_RAM_MODEL_WORD_PARTITION_SIZE,
            TEST_LITERALS_BUFFER_RAM_MODEL_SIMULTANEOUS_READ_WRITE_BEHAVIOR,
            TEST_LITERALS_BUFFER_RAM_MODEL_INITIALIZED
        > (
            ram_rd_req_r[3], ram_rd_resp_s[3], ram_wr_req_r[3], ram_wr_resp_s[3]
        );
        spawn ram::RamModel<
            TEST_LITERALS_BUFFER_RAM_MODEL_DATA_WIDTH,
            TEST_LITERALS_BUFFER_RAM_MODEL_SIZE,
            TEST_LITERALS_BUFFER_RAM_MODEL_WORD_PARTITION_SIZE,
            TEST_LITERALS_BUFFER_RAM_MODEL_SIMULTANEOUS_READ_WRITE_BEHAVIOR,
            TEST_LITERALS_BUFFER_RAM_MODEL_INITIALIZED
        > (
            ram_rd_req_r[4], ram_rd_resp_s[4], ram_wr_req_r[4], ram_wr_resp_s[4]
        );
        spawn ram::RamModel<
            TEST_LITERALS_BUFFER_RAM_MODEL_DATA_WIDTH,
            TEST_LITERALS_BUFFER_RAM_MODEL_SIZE,
            TEST_LITERALS_BUFFER_RAM_MODEL_WORD_PARTITION_SIZE,
            TEST_LITERALS_BUFFER_RAM_MODEL_SIMULTANEOUS_READ_WRITE_BEHAVIOR,
            TEST_LITERALS_BUFFER_RAM_MODEL_INITIALIZED
        > (
            ram_rd_req_r[5], ram_rd_resp_s[5], ram_wr_req_r[5], ram_wr_resp_s[5]
        );
        spawn ram::RamModel<
            TEST_LITERALS_BUFFER_RAM_MODEL_DATA_WIDTH,
            TEST_LITERALS_BUFFER_RAM_MODEL_SIZE,
            TEST_LITERALS_BUFFER_RAM_MODEL_WORD_PARTITION_SIZE,
            TEST_LITERALS_BUFFER_RAM_MODEL_SIMULTANEOUS_READ_WRITE_BEHAVIOR,
            TEST_LITERALS_BUFFER_RAM_MODEL_INITIALIZED
        > (
            ram_rd_req_r[6], ram_rd_resp_s[6], ram_wr_req_r[6], ram_wr_resp_s[6]
        );
        spawn ram::RamModel<
            TEST_LITERALS_BUFFER_RAM_MODEL_DATA_WIDTH,
            TEST_LITERALS_BUFFER_RAM_MODEL_SIZE,
            TEST_LITERALS_BUFFER_RAM_MODEL_WORD_PARTITION_SIZE,
            TEST_LITERALS_BUFFER_RAM_MODEL_SIMULTANEOUS_READ_WRITE_BEHAVIOR,
            TEST_LITERALS_BUFFER_RAM_MODEL_INITIALIZED
        > (
            ram_rd_req_r[7], ram_rd_resp_s[7], ram_wr_req_r[7], ram_wr_resp_s[7]
        );

        // Mock RAM for Literals Header MemReader
        let (ram_rd_req_header_s, ram_rd_req_header_r) = chan<AxiRamRdReq>("ram_rd_req_header");
        let (ram_rd_resp_header_s, ram_rd_resp_header_r) = chan<AxiRamRdResp>("ram_rd_resp_header");
        let (ram_wr_req_header_s, ram_wr_req_header_r) = chan<AxiRamWrReq>("ram_wr_req_header");
        let (ram_wr_resp_header_s, ram_wr_resp_header_r) = chan<AxiRamWrResp>("ram_wr_resp_header");

        spawn ram::RamModel<
            TEST_AXI_RAM_MODEL_DATA_WIDTH,
            TEST_AXI_RAM_MODEL_SIZE,
            TEST_AXI_RAM_MODEL_WORD_PARTITION_SIZE,
            TEST_AXI_RAM_MODEL_SIMULTANEOUS_READ_WRITE_BEHAVIOR,
            TEST_AXI_RAM_MODEL_INITIALIZED,
            TEST_AXI_RAM_MODEL_ASSERT_VALID_READ,
            TEST_AXI_RAM_MODEL_ADDR_WIDTH
        > (
            ram_rd_req_header_r, ram_rd_resp_header_s, ram_wr_req_header_r, ram_wr_resp_header_s
        );

        spawn axi_ram_reader::AxiRamReader<
            TEST_AXI_RAM_ADDR_W, TEST_AXI_RAM_DATA_W, TEST_AXI_RAM_DEST_W, TEST_AXI_RAM_ID_W,
            TEST_AXI_RAM_MODEL_SIZE, TEST_AXI_RAM_MODEL_BASE_ADDR, TEST_AXI_RAM_MODEL_DATA_WIDTH, TEST_AXI_RAM_MODEL_ADDR_WIDTH
        > (
            lit_header_axi_ar_r, lit_header_axi_r_s,
            ram_rd_req_header_s, ram_rd_resp_header_r
        );

        // Mock RAM for RawLiterals MemReader
        let (ram_rd_req_raw_s, ram_rd_req_raw_r) = chan<AxiRamRdReq>("ram_rd_req_raw");
        let (ram_rd_resp_raw_s, ram_rd_resp_raw_r) = chan<AxiRamRdResp>("ram_rd_resp_raw");
        let (ram_wr_req_raw_s, ram_wr_req_raw_r) = chan<AxiRamWrReq>("ram_wr_req_raw");
        let (ram_wr_resp_raw_s, ram_wr_resp_raw_r) = chan<AxiRamWrResp>("ram_wr_resp_raw");

        spawn ram::RamModel<
            TEST_AXI_RAM_MODEL_DATA_WIDTH,
            TEST_AXI_RAM_MODEL_SIZE,
            TEST_AXI_RAM_MODEL_WORD_PARTITION_SIZE,
            TEST_AXI_RAM_MODEL_SIMULTANEOUS_READ_WRITE_BEHAVIOR,
            TEST_AXI_RAM_MODEL_INITIALIZED,
            TEST_AXI_RAM_MODEL_ASSERT_VALID_READ,
            TEST_AXI_RAM_MODEL_ADDR_WIDTH
        > (
            ram_rd_req_raw_r, ram_rd_resp_raw_s, ram_wr_req_raw_r, ram_wr_resp_raw_s
        );

        spawn axi_ram_reader::AxiRamReader<
            TEST_AXI_RAM_ADDR_W, TEST_AXI_RAM_DATA_W, TEST_AXI_RAM_DEST_W, TEST_AXI_RAM_ID_W,
            TEST_AXI_RAM_MODEL_SIZE, TEST_AXI_RAM_MODEL_BASE_ADDR, TEST_AXI_RAM_MODEL_DATA_WIDTH, TEST_AXI_RAM_MODEL_ADDR_WIDTH
        > (
            raw_lit_axi_ar_r, raw_lit_axi_r_s,
            ram_rd_req_raw_s, ram_rd_resp_raw_r
        );

        // Mock RAM for HuffmanLiteralsDecoder MemReader
        let (ram_rd_req_huffman_s, ram_rd_req_huffman_r) = chan<AxiRamRdReq>("ram_rd_req_huffman");
        let (ram_rd_resp_huffman_s, ram_rd_resp_huffman_r) = chan<AxiRamRdResp>("ram_rd_resp_huffman");
        let (ram_wr_req_huffman_s, ram_wr_req_huffman_r) = chan<AxiRamWrReq>("ram_wr_req_huffman");
        let (ram_wr_resp_huffman_s, ram_wr_resp_huffman_r) = chan<AxiRamWrResp>("ram_wr_resp_huffman");

        spawn ram::RamModel<
            TEST_AXI_RAM_MODEL_DATA_WIDTH,
            TEST_AXI_RAM_MODEL_SIZE,
            TEST_AXI_RAM_MODEL_WORD_PARTITION_SIZE,
            TEST_AXI_RAM_MODEL_SIMULTANEOUS_READ_WRITE_BEHAVIOR,
            TEST_AXI_RAM_MODEL_INITIALIZED,
            TEST_AXI_RAM_MODEL_ASSERT_VALID_READ,
            TEST_AXI_RAM_MODEL_ADDR_WIDTH
        > (
            ram_rd_req_huffman_r, ram_rd_resp_huffman_s, ram_wr_req_huffman_r, ram_wr_resp_huffman_s
        );

        spawn axi_ram_reader::AxiRamReader<
            TEST_AXI_RAM_ADDR_W, TEST_AXI_RAM_DATA_W, TEST_AXI_RAM_DEST_W, TEST_AXI_RAM_ID_W,
            TEST_AXI_RAM_MODEL_SIZE, TEST_AXI_RAM_MODEL_BASE_ADDR, TEST_AXI_RAM_MODEL_DATA_WIDTH, TEST_AXI_RAM_MODEL_ADDR_WIDTH
        > (
            huffman_lit_axi_ar_r, huffman_lit_axi_r_s,
            ram_rd_req_huffman_s, ram_rd_resp_huffman_r
        );

        // Mock RAM for Huffman Jump Table decoder MemReader
        let (ram_rd_req_huffman_jump_table_s, ram_rd_req_huffman_jump_table_r) = chan<AxiRamRdReq>("ram_rd_req_huffman_jump_table");
        let (ram_rd_resp_huffman_jump_table_s, ram_rd_resp_huffman_jump_table_r) = chan<AxiRamRdResp>("ram_rd_resp_huffman_jump_table");
        let (ram_wr_req_huffman_jump_table_s, ram_wr_req_huffman_jump_table_r) = chan<AxiRamWrReq>("ram_wr_req_huffman_jump_table");
        let (ram_wr_resp_huffman_jump_table_s, ram_wr_resp_huffman_jump_table_r) = chan<AxiRamWrResp>("ram_wr_resp_huffman_jump_table");

        spawn ram::RamModel<
            TEST_AXI_RAM_MODEL_DATA_WIDTH,
            TEST_AXI_RAM_MODEL_SIZE,
            TEST_AXI_RAM_MODEL_WORD_PARTITION_SIZE,
            TEST_AXI_RAM_MODEL_SIMULTANEOUS_READ_WRITE_BEHAVIOR,
            TEST_AXI_RAM_MODEL_INITIALIZED,
            TEST_AXI_RAM_MODEL_ASSERT_VALID_READ,
            TEST_AXI_RAM_MODEL_ADDR_WIDTH
        > (
            ram_rd_req_huffman_jump_table_r, ram_rd_resp_huffman_jump_table_s, ram_wr_req_huffman_jump_table_r, ram_wr_resp_huffman_jump_table_s
        );

        spawn axi_ram_reader::AxiRamReader<
            TEST_AXI_RAM_ADDR_W, TEST_AXI_RAM_DATA_W, TEST_AXI_RAM_DEST_W, TEST_AXI_RAM_ID_W,
            TEST_AXI_RAM_MODEL_SIZE, TEST_AXI_RAM_MODEL_BASE_ADDR, TEST_AXI_RAM_MODEL_DATA_WIDTH, TEST_AXI_RAM_MODEL_ADDR_WIDTH
        > (
            huffman_jump_table_axi_ar_r, huffman_jump_table_axi_r_s,
            ram_rd_req_huffman_jump_table_s, ram_rd_resp_huffman_jump_table_r
        );

        // Mock RAM for HuffmanWeights header decoder MemReader
        let (ram_rd_req_huffman_weights_header_s, ram_rd_req_huffman_weights_header_r) = chan<AxiRamRdReq>("ram_rd_req_huffman_weights_header");
        let (ram_rd_resp_huffman_weights_header_s, ram_rd_resp_huffman_weights_header_r) = chan<AxiRamRdResp>("ram_rd_resp_huffman_weights_header");
        let (ram_wr_req_huffman_weights_header_s, ram_wr_req_huffman_weights_header_r) = chan<AxiRamWrReq>("ram_wr_req_huffman_weights_header");
        let (ram_wr_resp_huffman_weights_header_s, ram_wr_resp_huffman_weights_header_r) = chan<AxiRamWrResp>("ram_wr_resp_huffman_weights_header");

        spawn ram::RamModel<
            TEST_AXI_RAM_MODEL_DATA_WIDTH,
            TEST_AXI_RAM_MODEL_SIZE,
            TEST_AXI_RAM_MODEL_WORD_PARTITION_SIZE,
            TEST_AXI_RAM_MODEL_SIMULTANEOUS_READ_WRITE_BEHAVIOR,
            TEST_AXI_RAM_MODEL_INITIALIZED,
            TEST_AXI_RAM_MODEL_ASSERT_VALID_READ,
            TEST_AXI_RAM_MODEL_ADDR_WIDTH
        > (
            ram_rd_req_huffman_weights_header_r, ram_rd_resp_huffman_weights_header_s, ram_wr_req_huffman_weights_header_r, ram_wr_resp_huffman_weights_header_s
        );

        spawn axi_ram_reader::AxiRamReader<
            TEST_AXI_RAM_ADDR_W, TEST_AXI_RAM_DATA_W, TEST_AXI_RAM_DEST_W, TEST_AXI_RAM_ID_W,
            TEST_AXI_RAM_MODEL_SIZE, TEST_AXI_RAM_MODEL_BASE_ADDR, TEST_AXI_RAM_MODEL_DATA_WIDTH, TEST_AXI_RAM_MODEL_ADDR_WIDTH
        > (
            huffman_weights_header_axi_ar_r, huffman_weights_header_axi_r_s,
            ram_rd_req_huffman_weights_header_s, ram_rd_resp_huffman_weights_header_r
        );

        // Mock RAM for HuffmanWeights raw decoder MemReader
        let (ram_rd_req_huffman_weights_raw_s, ram_rd_req_huffman_weights_raw_r) = chan<AxiRamRdReq>("ram_rd_req_huffman_weights_raw");
        let (ram_rd_resp_huffman_weights_raw_s, ram_rd_resp_huffman_weights_raw_r) = chan<AxiRamRdResp>("ram_rd_resp_huffman_weights_raw");
        let (ram_wr_req_huffman_weights_raw_s, ram_wr_req_huffman_weights_raw_r) = chan<AxiRamWrReq>("ram_wr_req_huffman_weights_raw");
        let (ram_wr_resp_huffman_weights_raw_s, ram_wr_resp_huffman_weights_raw_r) = chan<AxiRamWrResp>("ram_wr_resp_huffman_weights_raw");

        spawn ram::RamModel<
            TEST_AXI_RAM_MODEL_DATA_WIDTH,
            TEST_AXI_RAM_MODEL_SIZE,
            TEST_AXI_RAM_MODEL_WORD_PARTITION_SIZE,
            TEST_AXI_RAM_MODEL_SIMULTANEOUS_READ_WRITE_BEHAVIOR,
            TEST_AXI_RAM_MODEL_INITIALIZED,
            TEST_AXI_RAM_MODEL_ASSERT_VALID_READ,
            TEST_AXI_RAM_MODEL_ADDR_WIDTH
        > (
            ram_rd_req_huffman_weights_raw_r, ram_rd_resp_huffman_weights_raw_s, ram_wr_req_huffman_weights_raw_r, ram_wr_resp_huffman_weights_raw_s
        );

        spawn axi_ram_reader::AxiRamReader<
            TEST_AXI_RAM_ADDR_W, TEST_AXI_RAM_DATA_W, TEST_AXI_RAM_DEST_W, TEST_AXI_RAM_ID_W,
            TEST_AXI_RAM_MODEL_SIZE, TEST_AXI_RAM_MODEL_BASE_ADDR, TEST_AXI_RAM_MODEL_DATA_WIDTH, TEST_AXI_RAM_MODEL_ADDR_WIDTH
        > (
            huffman_weights_raw_axi_ar_r, huffman_weights_raw_axi_r_s,
            ram_rd_req_huffman_weights_raw_s, ram_rd_resp_huffman_weights_raw_r
        );

        // Mock RAM for HuffmanWeights FseLookupDecoder MemReader
        let (ram_rd_req_huffman_weights_fse_lookup_dec_s, ram_rd_req_huffman_weights_fse_lookup_dec_r) = chan<AxiRamRdReq>("ram_rd_req_huffman_weights_fse_lookup_dec");
        let (ram_rd_resp_huffman_weights_fse_lookup_dec_s, ram_rd_resp_huffman_weights_fse_lookup_dec_r) = chan<AxiRamRdResp>("ram_rd_resp_huffman_weights_fse_lookup_dec");
        let (ram_wr_req_huffman_weights_fse_lookup_dec_s, ram_wr_req_huffman_weights_fse_lookup_dec_r) = chan<AxiRamWrReq>("ram_wr_req_huffman_weights_fse_lookup_dec");
        let (ram_wr_resp_huffman_weights_fse_lookup_dec_s, ram_wr_resp_huffman_weights_fse_lookup_dec_r) = chan<AxiRamWrResp>("ram_wr_resp_huffman_weights_fse_lookup_dec");

        spawn ram::RamModel<
            TEST_AXI_RAM_MODEL_DATA_WIDTH,
            TEST_AXI_RAM_MODEL_SIZE,
            TEST_AXI_RAM_MODEL_WORD_PARTITION_SIZE,
            TEST_AXI_RAM_MODEL_SIMULTANEOUS_READ_WRITE_BEHAVIOR,
            TEST_AXI_RAM_MODEL_INITIALIZED,
            TEST_AXI_RAM_MODEL_ASSERT_VALID_READ,
            TEST_AXI_RAM_MODEL_ADDR_WIDTH
        > (
            ram_rd_req_huffman_weights_fse_lookup_dec_r, ram_rd_resp_huffman_weights_fse_lookup_dec_s,
            ram_wr_req_huffman_weights_fse_lookup_dec_r, ram_wr_resp_huffman_weights_fse_lookup_dec_s
        );

        spawn axi_ram_reader::AxiRamReader<
            TEST_AXI_RAM_ADDR_W, TEST_AXI_RAM_DATA_W, TEST_AXI_RAM_DEST_W, TEST_AXI_RAM_ID_W,
            TEST_AXI_RAM_MODEL_SIZE, TEST_AXI_RAM_MODEL_BASE_ADDR, TEST_AXI_RAM_MODEL_DATA_WIDTH, TEST_AXI_RAM_MODEL_ADDR_WIDTH
        > (
            huffman_weights_fse_lookup_dec_axi_ar_r, huffman_weights_fse_lookup_dec_axi_r_s,
            ram_rd_req_huffman_weights_fse_lookup_dec_s, ram_rd_resp_huffman_weights_fse_lookup_dec_r
        );

        // Mock RAM for HuffmanWeights FseDecoder MemReader
        let (ram_rd_req_huffman_weights_fse_decoder_dec_s, ram_rd_req_huffman_weights_fse_decoder_dec_r) = chan<AxiRamRdReq>("ram_rd_req_huffman_weights_fse_decoder_dec");
        let (ram_rd_resp_huffman_weights_fse_decoder_dec_s, ram_rd_resp_huffman_weights_fse_decoder_dec_r) = chan<AxiRamRdResp>("ram_rd_resp_huffman_weights_fse_decoder_dec");
        let (ram_wr_req_huffman_weights_fse_decoder_dec_s, ram_wr_req_huffman_weights_fse_decoder_dec_r) = chan<AxiRamWrReq>("ram_wr_req_huffman_weights_fse_decoder_dec");
        let (ram_wr_resp_huffman_weights_fse_decoder_dec_s, ram_wr_resp_huffman_weights_fse_decoder_dec_r) = chan<AxiRamWrResp>("ram_wr_resp_huffman_weights_fse_decoder_dec");

        spawn ram::RamModel<
            TEST_AXI_RAM_MODEL_DATA_WIDTH,
            TEST_AXI_RAM_MODEL_SIZE,
            TEST_AXI_RAM_MODEL_WORD_PARTITION_SIZE,
            TEST_AXI_RAM_MODEL_SIMULTANEOUS_READ_WRITE_BEHAVIOR,
            TEST_AXI_RAM_MODEL_INITIALIZED,
            TEST_AXI_RAM_MODEL_ASSERT_VALID_READ,
            TEST_AXI_RAM_MODEL_ADDR_WIDTH
        > (
            ram_rd_req_huffman_weights_fse_decoder_dec_r, ram_rd_resp_huffman_weights_fse_decoder_dec_s,
            ram_wr_req_huffman_weights_fse_decoder_dec_r, ram_wr_resp_huffman_weights_fse_decoder_dec_s
        );

        spawn axi_ram_reader::AxiRamReader<
            TEST_AXI_RAM_ADDR_W, TEST_AXI_RAM_DATA_W, TEST_AXI_RAM_DEST_W, TEST_AXI_RAM_ID_W,
            TEST_AXI_RAM_MODEL_SIZE, TEST_AXI_RAM_MODEL_BASE_ADDR, TEST_AXI_RAM_MODEL_DATA_WIDTH, TEST_AXI_RAM_MODEL_ADDR_WIDTH
        > (
            huffman_weights_fse_decoder_dec_axi_ar_r, huffman_weights_fse_decoder_dec_axi_r_s,
            ram_rd_req_huffman_weights_fse_decoder_dec_s, ram_rd_resp_huffman_weights_fse_decoder_dec_r
        );

        // Huffman weigths memory
        spawn ram::RamModel<
            TEST_HUFFMAN_WEIGHTS_RAM_MODEL_DATA_WIDTH,
            TEST_HUFFMAN_WEIGHTS_RAM_MODEL_SIZE,
            TEST_HUFFMAN_WEIGHTS_RAM_MODEL_WORD_PARTITION_SIZE,
            TEST_HUFFMAN_WEIGHTS_RAM_MODEL_SIMULTANEOUS_READ_WRITE_BEHAVIOR,
            TEST_HUFFMAN_WEIGHTS_RAM_MODEL_INITIALIZED,
            TEST_HUFFMAN_WEIGHTS_RAM_MODEL_ASSERT_VALID_READ,
            TEST_HUFFMAN_WEIGHTS_RAM_MODEL_ADDR_WIDTH,
        > (
            huffman_lit_weights_mem_rd_req_r, huffman_lit_weights_mem_rd_resp_s,
            huffman_lit_weights_mem_wr_req_r, huffman_lit_weights_mem_wr_resp_s
        );

        // Huffman prescan memory
        spawn ram::RamModel<
            TEST_HUFFMAN_PRESCAN_RAM_MODEL_DATA_WIDTH,
            TEST_HUFFMAN_PRESCAN_RAM_MODEL_SIZE,
            TEST_HUFFMAN_PRESCAN_RAM_MODEL_WORD_PARTITION_SIZE,
            TEST_HUFFMAN_PRESCAN_RAM_MODEL_SIMULTANEOUS_READ_WRITE_BEHAVIOR,
            TEST_HUFFMAN_PRESCAN_RAM_MODEL_INITIALIZED,
            TEST_HUFFMAN_PRESCAN_RAM_MODEL_ASSERT_VALID_READ,
            TEST_HUFFMAN_PRESCAN_RAM_MODEL_ADDR_WIDTH
        > (
            huffman_lit_prescan_mem_rd_req_r, huffman_lit_prescan_mem_rd_resp_s,
            huffman_lit_prescan_mem_wr_req_r, huffman_lit_prescan_mem_wr_resp_s
        );

        spawn ram::RamModel<
            TEST_HUFFMAN_WEIGHTS_DPD_RAM_MODEL_DATA_WIDTH,
            TEST_HUFFMAN_WEIGHTS_DPD_RAM_MODEL_SIZE,
            TEST_HUFFMAN_WEIGHTS_DPD_RAM_MODEL_WORD_PARTITION_SIZE,
            TEST_HUFFMAN_WEIGHTS_DPD_RAM_MODEL_SIMULTANEOUS_READ_WRITE_BEHAVIOR,
            TEST_HUFFMAN_WEIGHTS_DPD_RAM_MODEL_INITIALIZED,
            TEST_HUFFMAN_WEIGHTS_DPD_RAM_MODEL_ASSERT_VALID_READ,
            TEST_HUFFMAN_WEIGHTS_DPD_RAM_MODEL_ADDR_WIDTH
        > (
            huffman_lit_weights_dpd_rd_req_r, huffman_lit_weights_dpd_rd_resp_s,
            huffman_lit_weights_dpd_wr_req_r, huffman_lit_weights_dpd_wr_resp_s
        );


        spawn ram::RamModel<
            TEST_HUFFMAN_WEIGHTS_TMP_RAM_MODEL_DATA_WIDTH,
            TEST_HUFFMAN_WEIGHTS_TMP_RAM_MODEL_SIZE,
            TEST_HUFFMAN_WEIGHTS_TMP_RAM_MODEL_WORD_PARTITION_SIZE,
            TEST_HUFFMAN_WEIGHTS_TMP_RAM_MODEL_SIMULTANEOUS_READ_WRITE_BEHAVIOR,
            TEST_HUFFMAN_WEIGHTS_TMP_RAM_MODEL_INITIALIZED,
            TEST_HUFFMAN_WEIGHTS_TMP_RAM_MODEL_ASSERT_VALID_READ,
            TEST_HUFFMAN_WEIGHTS_TMP_RAM_MODEL_ADDR_WIDTH
        > (
            huffman_lit_weights_tmp_rd_req_r, huffman_lit_weights_tmp_rd_resp_s,
            huffman_lit_weights_tmp_wr_req_r, huffman_lit_weights_tmp_wr_resp_s
        );

        spawn ram::RamModel<
            TEST_HUFFMAN_WEIGHTS_TMP2_RAM_MODEL_DATA_WIDTH,
            TEST_HUFFMAN_WEIGHTS_TMP2_RAM_MODEL_SIZE,
            TEST_HUFFMAN_WEIGHTS_TMP2_RAM_MODEL_WORD_PARTITION_SIZE,
            TEST_HUFFMAN_WEIGHTS_TMP2_RAM_MODEL_SIMULTANEOUS_READ_WRITE_BEHAVIOR,
            TEST_HUFFMAN_WEIGHTS_TMP2_RAM_MODEL_INITIALIZED,
            TEST_HUFFMAN_WEIGHTS_TMP2_RAM_MODEL_ASSERT_VALID_READ,
            TEST_HUFFMAN_WEIGHTS_TMP2_RAM_MODEL_ADDR_WIDTH
        > (
            huffman_lit_weights_tmp2_rd_req_r, huffman_lit_weights_tmp2_rd_resp_s,
            huffman_lit_weights_tmp2_wr_req_r, huffman_lit_weights_tmp2_wr_resp_s
        );

        spawn ram::RamModel<
            TEST_HUFFMAN_WEIGHTS_FSE_RAM_MODEL_DATA_WIDTH,
            TEST_HUFFMAN_WEIGHTS_FSE_RAM_MODEL_SIZE,
            TEST_HUFFMAN_WEIGHTS_FSE_RAM_MODEL_WORD_PARTITION_SIZE,
            TEST_HUFFMAN_WEIGHTS_FSE_RAM_MODEL_SIMULTANEOUS_READ_WRITE_BEHAVIOR,
            TEST_HUFFMAN_WEIGHTS_FSE_RAM_MODEL_INITIALIZED,
            TEST_HUFFMAN_WEIGHTS_FSE_RAM_MODEL_ASSERT_VALID_READ,
            TEST_HUFFMAN_WEIGHTS_FSE_RAM_MODEL_ADDR_WIDTH
        > (
            huffman_lit_weights_fse_rd_req_r, huffman_lit_weights_fse_rd_resp_s,
            huffman_lit_weights_fse_wr_req_r, huffman_lit_weights_fse_wr_resp_s
        );

        (
            terminator,
            ctrl_req_s, ctrl_resp_r, ctrl_header_r,
            buf_ctrl_s, buf_out_r,
            print_start_s, print_finish_r,
            ram_wr_req_header_s, ram_wr_resp_header_r,
            ram_wr_req_raw_s, ram_wr_resp_raw_r,
            ram_wr_req_huffman_s, ram_wr_resp_huffman_r,
            ram_wr_req_huffman_jump_table_s, ram_wr_resp_huffman_jump_table_r,
            ram_wr_req_huffman_weights_header_s, ram_wr_resp_huffman_weights_header_r,
            ram_wr_req_huffman_weights_raw_s, ram_wr_resp_huffman_weights_raw_r,
            ram_wr_req_huffman_weights_fse_lookup_dec_s, ram_wr_resp_huffman_weights_fse_lookup_dec_r,
            ram_wr_req_huffman_weights_fse_decoder_dec_s, ram_wr_resp_huffman_weights_fse_decoder_dec_r
        )
    }

    init { }

    next (state: ()) {
        const TEST_MEMORY: AxiRamWrReq[21] = [
            // Literals #0 (RAW; 8 bytes)
            // Header: 0x40
            AxiRamWrReq { addr: AxiRamAddr:0x0, data: AxiRamData:0x5734_65A6_DB5D_B040, mask: AxiRamMask:0xFF },    // AXI addr: 0x0
            AxiRamWrReq { addr: AxiRamAddr:0x1, data: AxiRamData:0x16, mask: AxiRamMask:0xFF },                     // AXI addr: 0x8

            // Literals #1 (RLE; 4 bytes)
            // Header: 0x21
            AxiRamWrReq { addr: AxiRamAddr:0x2, data: AxiRamData:0x2321, mask: AxiRamMask:0xFF },                   // AXI addr: 0x10

            // Literals #2 (RLE; 2 bytes)
            // Header: 0x11
            AxiRamWrReq { addr: AxiRamAddr:0x4, data: AxiRamData:0x3511, mask: AxiRamMask:0xFF },                   // AXI addr: 0x20

            // Literals #3 (RAW; 15 bytes)
            // Header: 0x78
            AxiRamWrReq { addr: AxiRamAddr:0x6, data: AxiRamData:0xFB41_C67B_6053_7078, mask: AxiRamMask:0xFF },    // AXI addr: 0x30
            AxiRamWrReq { addr: AxiRamAddr:0x7, data: AxiRamData:0x9B0F_9CE1_BAA9_6D4C, mask: AxiRamMask:0xFF },    // AXI addr: 0x38

            // Literals #4 (Huffman; 6 bytes)
            // Header: 0x01_80_42 (0b0000000110_0000000100_00_10)
            AxiRamWrReq { addr: AxiRamAddr:0x10, data: (u8:0b0000_0001 ++ u24:0x100234 ++ u8:0x84 ++ u24:0x01_80_42) as AxiRamData, mask: AxiRamMask:0xFF }, // AXI addr: 0x80
            AxiRamWrReq { addr: AxiRamAddr:0x11, data: u8:0b00001_1_01 as AxiRamData, mask: AxiRamMask:0xFF },      // AXI addr: 0x88

            // Literals #5 (RLE; 12 bytes)
            // Header: 0x61
            AxiRamWrReq { addr: AxiRamAddr:0x20, data: AxiRamData:0x5A61, mask: AxiRamMask:0xFF },                  // AXI addr: 0x100

            // Literals #6 (Huffman; 4 bytes)
            // Header: 0x00_80_43 (0b0000000010_0000000100_00_11)
            AxiRamWrReq { addr: AxiRamAddr:0x30, data: (u16:0b00001_0001_0000_01_1 ++ u24:0x00_80_43) as AxiRamData, mask: AxiRamMask:0xFF }, // AXI addr: 0x180

            // Literals #7 (RLE; 0 bytes)
            // Header: 0x01
            AxiRamWrReq { addr: AxiRamAddr:0x40, data: AxiRamData:0xFF01, mask: AxiRamMask:0xFF },                  // AXI addr: 0x200

            // Literals #8 (Huffman; 18 bytes)
            // Header: 0x04_81_06 (0b0000010010_0000010010_01_10)
            AxiRamWrReq { addr: AxiRamAddr:0x50, data: (u8:0x02 ++ u24:0x100234 ++ u8:0x84 ++ u24:0x04_81_06) as AxiRamData, mask: AxiRamMask:0xFF }, // AXI addr: 0x280
            AxiRamWrReq { addr: AxiRamAddr:0x51, data: (u8:0b0000_0001 ++ u16:0b00001_1_01_0000_0001 ++ u40:0x0002_0002_00) as AxiRamData, mask: AxiRamMask:0xFF }, // AXI addr: 0x288
            AxiRamWrReq { addr: AxiRamAddr:0x52, data: (u16:0b00001_1_01_0000_0001 ++ u16:0b00001_1_01_0000_0001 ++ u8:0b00001_1_01) as AxiRamData, mask: AxiRamMask:0xFF }, // AXI addr: 290

            // Literals #9 (RAW; 31 bytes)
            // Header: 0xF8
            AxiRamWrReq { addr: AxiRamAddr:0x60, data: AxiRamData:0x943E_9618_34C2_47F8, mask: AxiRamMask:0xFF },   // AXI addr: 0x300
            AxiRamWrReq { addr: AxiRamAddr:0x61, data: AxiRamData:0x02D0_E8D7_289A_BE60, mask: AxiRamMask:0xFF },   // AXI addr: 0x308
            AxiRamWrReq { addr: AxiRamAddr:0x62, data: AxiRamData:0x64C3_8BE1_FA8D_12BC, mask: AxiRamMask:0xFF },   // AXI addr: 0x310
            AxiRamWrReq { addr: AxiRamAddr:0x63, data: AxiRamData:0x1963_F1CE_21C2_94F8, mask: AxiRamMask:0xFF },   // AXI addr: 0x318

            // Literals #10 (Huffman; 15 bytes)
            // Header: 0x03_80_E7 (0b0000001110_0000001110_01_11)
            AxiRamWrReq { addr: AxiRamAddr:0x70, data: (u40:0x02_0002_0002 ++ u24:0x03_80_E7) as AxiRamData, mask: AxiRamMask:0xFF }, // AXI addr: 0x350
            AxiRamWrReq { addr: AxiRamAddr:0x71, data: (u8:0b0000_0001 ++ u16:0b00001_1_01_0000_0001 ++ u16:0b00001_1_01_0000_0001 ++ u16:0b00001_1_01_0000_0001 ++ u8:0x00) as AxiRamData, mask: AxiRamMask:0xFF }, // AXI addr: 0x358
            AxiRamWrReq { addr: AxiRamAddr:0x72, data: u8:0b00001_1_01 as AxiRamData, mask: AxiRamMask:0xFF }, // AXI addr: 0x358
        ];

        const TEST_CTRL: CtrlReq[11] = [
            CtrlReq {addr: AxiAddr:0x0, literals_last: false},
            CtrlReq {addr: AxiAddr:0x10, literals_last: false},
            CtrlReq {addr: AxiAddr:0x20, literals_last: false},
            CtrlReq {addr: AxiAddr:0x30, literals_last: true},
            CtrlReq {addr: AxiAddr:0x80, literals_last: true},
            CtrlReq {addr: AxiAddr:0x100, literals_last: false},
            CtrlReq {addr: AxiAddr:0x180, literals_last: true},
            CtrlReq {addr: AxiAddr:0x200, literals_last: false},
            CtrlReq {addr: AxiAddr:0x280, literals_last: true},
            CtrlReq {addr: AxiAddr:0x300, literals_last: true},
            CtrlReq {addr: AxiAddr:0x380, literals_last: true},
        ];

        const TEST_EXPECTED_RESP: CtrlResp[11] = [
            CtrlResp {status: CtrlStatus::OKAY},
            CtrlResp {status: CtrlStatus::OKAY},
            CtrlResp {status: CtrlStatus::OKAY},
            CtrlResp {status: CtrlStatus::OKAY},
            CtrlResp {status: CtrlStatus::OKAY},
            CtrlResp {status: CtrlStatus::OKAY},
            CtrlResp {status: CtrlStatus::OKAY},
            CtrlResp {status: CtrlStatus::OKAY},
            CtrlResp {status: CtrlStatus::OKAY},
            CtrlResp {status: CtrlStatus::OKAY},
            CtrlResp {status: CtrlStatus::OKAY},
        ];

        const TEST_BUF_CTRL: LiteralsBufferCtrl[10] = [
            LiteralsBufferCtrl {length: u32:8, last: false},
            LiteralsBufferCtrl {length: u32:4 , last: false},
            LiteralsBufferCtrl {length: u32:1 , last: false},
            LiteralsBufferCtrl {length: u32:16, last: true},
            LiteralsBufferCtrl {length: u32:4, last: true},
            LiteralsBufferCtrl {length: u32:12, last: false},
            LiteralsBufferCtrl {length: u32:4, last: true},
            LiteralsBufferCtrl {length: u32:16, last: true},
            LiteralsBufferCtrl {length: u32:31, last: true},
            LiteralsBufferCtrl {length: u32:16, last: true},
        ];

        const TEST_EXPECTED_LITERALS: SequenceExecutorPacket[17] = [
            // Literals #0 (RAW)
            SequenceExecutorPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                length: CopyOrMatchLength:8,
                content: CopyOrMatchContent:0x1657_3465_A6DB_5DB0,
                last: true
            },
            // Literals #1 (RLE)
            SequenceExecutorPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                length: CopyOrMatchLength:4,
                content: CopyOrMatchContent:0x2323_2323,
                last: true
            },
            // Literals #2 (RLE)
            SequenceExecutorPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                length: CopyOrMatchLength:1,
                content: CopyOrMatchContent:0x35,
                last: false
            },
            // Literals #3 (RAW)
            SequenceExecutorPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                length: CopyOrMatchLength:8,
                content: CopyOrMatchContent:0xFB41_C67B_6053_7035,
                last: false
            },
            SequenceExecutorPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                length: CopyOrMatchLength:8,
                content: CopyOrMatchContent:0x9B0F_9CE1_BAA9_6D4C,
                last: true
            },
            // Literals #4 (Huffman)
            SequenceExecutorPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                length: CopyOrMatchLength:4,
                content: CopyOrMatchContent:0x0504_0100,
                last: true
            },
            // Literals #5 (RLE)
            SequenceExecutorPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                length: CopyOrMatchLength:8,
                content: CopyOrMatchContent:0x5A5A_5A5A_5A5A_5A5A,
                last: false
            },
            SequenceExecutorPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                length: CopyOrMatchLength:4,
                content: CopyOrMatchContent:0x5A5A_5A5A,
                last: true
            },
            // Literals #6 (Huffman)
            SequenceExecutorPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                length: CopyOrMatchLength:4,
                content: CopyOrMatchContent:0x0001_0405,
                last: true
            },
            // Literals #7 (RLE)
            // EMPTY
            // Literals #8 (Huffman)
            SequenceExecutorPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                length: CopyOrMatchLength:8,
                content: CopyOrMatchContent:0x0504_0100_0504_0100,
                last: false
            },
            SequenceExecutorPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                length: CopyOrMatchLength:8,
                content: CopyOrMatchContent:0x0504_0100_0504_0100,
                last: true
            },
            // Literals #9 (RAW)
            SequenceExecutorPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                length: CopyOrMatchLength:8,
                content: CopyOrMatchContent:0x6094_3E96_1834_C247,
                last: false
            },
            SequenceExecutorPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                length: CopyOrMatchLength:8,
                content: CopyOrMatchContent:0xBC02_D0E8_D728_9ABE,
                last: false
            },
            SequenceExecutorPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                length: CopyOrMatchLength:8,
                content: CopyOrMatchContent:0xF864_C38B_E1FA_8D12,
                last: false
            },
            SequenceExecutorPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                length: CopyOrMatchLength:7,
                content: CopyOrMatchContent:0x19_63F1_CE21_C294,
                last: true
            },
            // Literals #10 (Huffman)
            SequenceExecutorPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                length: CopyOrMatchLength:8,
                content: CopyOrMatchContent:0x0504_0100_0504_0100,
                last: false
            },
            SequenceExecutorPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                length: CopyOrMatchLength:8,
                content: CopyOrMatchContent:0x0504_0100_0504_0100,
                last: true
            },
        ];

        let tok = join();

        trace_fmt!("Filling system memory mock");
        let tok = for ((i, mem_req), tok):((u32, AxiRamWrReq), token) in enumerate(TEST_MEMORY) {
            trace_fmt!("Sent memory write request #{}: {:#x}", i + u32:1, mem_req);
            let tok = send(tok, ram_wr_req_header_s, mem_req);
            let (tok, _) = recv(tok, ram_wr_resp_header_r);
            let tok = send(tok, ram_wr_req_raw_s, mem_req);
            let (tok, _) = recv(tok, ram_wr_resp_raw_r);
            let tok = send(tok, ram_wr_req_huffman_s, mem_req);
            let (tok, _) = recv(tok, ram_wr_resp_huffman_r);
            let tok = send(tok, ram_wr_req_huffman_jump_table_s, mem_req);
            let (tok, _) = recv(tok, ram_wr_resp_huffman_jump_table_r);
            let tok = send(tok, ram_wr_req_huffman_weights_header_s, mem_req);
            let (tok, _) = recv(tok, ram_wr_resp_huffman_weights_header_r);
            let tok = send(tok, ram_wr_req_huffman_weights_raw_s, mem_req);
            let (tok, _) = recv(tok, ram_wr_resp_huffman_weights_raw_r);
            let tok = send(tok, ram_wr_req_huffman_weights_fse_lookup_dec_s, mem_req);
            let (tok, _) = recv(tok, ram_wr_resp_huffman_weights_fse_lookup_dec_r);
            let tok = send(tok, ram_wr_req_huffman_weights_fse_decoder_dec_s, mem_req);
            let (tok, _) = recv(tok, ram_wr_resp_huffman_weights_fse_decoder_dec_r);
            tok
        }(tok);

        assert_eq(array_size(TEST_CTRL), array_size(TEST_EXPECTED_RESP));

        trace_fmt!("Sending literals decoding requests");
        let tok = for ((i, test_ctrl), tok): ((u32, CtrlReq), token) in enumerate(TEST_CTRL) {
            let tok = send(tok, ctrl_req_s, test_ctrl);
            trace_fmt!("Sent #{} literals decoding request: {:#x}", i + u32:1, test_ctrl);
            let (tok, resp) = recv(tok, ctrl_resp_r);
            trace_fmt!("Received #{} literals decoding response {:#x}", i + u32:1, resp);
            assert_eq(TEST_EXPECTED_RESP[i], resp);
            tok
        }(tok);

        trace_fmt!("Sending literals buffer requests");
        let tok = for ((i, test_buf_ctrl), tok): ((u32, LiteralsBufferCtrl), token) in enumerate(TEST_BUF_CTRL) {
            let tok = send(tok, buf_ctrl_s, test_buf_ctrl);
            trace_fmt!("Sent #{} literals buffer request {:#x}", i + u32:1, test_buf_ctrl);
            tok
        }(tok);

        // receive and check packets
        let tok = for ((i, test_exp_literals), tok): ((u32, SequenceExecutorPacket), token) in enumerate(TEST_EXPECTED_LITERALS) {
            let (tok, literals) = recv(tok, buf_out_r);
            trace_fmt!("Received #{} literals packet {:#x}", i + u32:1, literals);
            trace_fmt!("Expected {:#x}", test_exp_literals);
            assert_eq(test_exp_literals, literals);
            tok
        }(tok);

        //// print RAM content
        //let tok = send(tok, print_start_s, ());
        //let (tok, _) = recv(tok, print_finish_r);

        send(tok, terminator, true);
    }
}

// TODO: Uncomment this test when fixed: https://github.com/google/xls/issues/1502
// type RamData = uN[literals_buffer::RAM_DATA_WIDTH];

// // Expected RAM content after each ctrl
// const TEST_EXPECTED_RAM_CONTENT = RamData[literals_buffer::RAM_NUM][10][7]:[
//     [
//         [RamData:0x016, RamData:0x057, RamData:0x034, RamData:0x065, RamData:0x0a6, RamData:0x0db, RamData:0x05d, RamData:0x0b0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//     ],
//     [
//         [RamData:0x016, RamData:0x057, RamData:0x034, RamData:0x065, RamData:0x0a6, RamData:0x0db, RamData:0x05d, RamData:0x0b0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:0x023, RamData:0x023, RamData:0x023, RamData:0x023],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//     ],
//     [
//         [RamData:0x016, RamData:0x057, RamData:0x034, RamData:0x065, RamData:0x0a6, RamData:0x0db, RamData:0x05d, RamData:0x0b0],
//         [RamData:  0x0, RamData:  0x0, RamData:0x035, RamData:0x035, RamData:0x023, RamData:0x023, RamData:0x023, RamData:0x023],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//     ],
//     [
//         [RamData:0x016, RamData:0x057, RamData:0x034, RamData:0x065, RamData:0x0a6, RamData:0x0db, RamData:0x05d, RamData:0x0b0],
//         [RamData:0x053, RamData:0x070, RamData:0x035, RamData:0x035, RamData:0x023, RamData:0x023, RamData:0x023, RamData:0x023],
//         [RamData:0x1a9, RamData:0x16d, RamData:0x04c, RamData:0x0fb, RamData:0x041, RamData:0x0c6, RamData:0x07b, RamData:0x060],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:0x19b, RamData:0x10f, RamData:0x19c, RamData:0x1e1, RamData:0x1ba],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//     ],
//     [
//         [RamData:0x016, RamData:0x057, RamData:0x034, RamData:0x065, RamData:0x0a6, RamData:0x0db, RamData:0x05d, RamData:0x0b0],
//         [RamData:0x053, RamData:0x070, RamData:0x035, RamData:0x035, RamData:0x023, RamData:0x023, RamData:0x023, RamData:0x023],
//         [RamData:0x1a9, RamData:0x16d, RamData:0x04c, RamData:0x0fb, RamData:0x041, RamData:0x0c6, RamData:0x07b, RamData:0x060],
//         [RamData:0x05a, RamData:0x05a, RamData:0x05a, RamData:0x19b, RamData:0x10f, RamData:0x19c, RamData:0x1e1, RamData:0x1ba],
//         [RamData:0x05a, RamData:0x05a, RamData:0x05a, RamData:0x05a, RamData:0x05a, RamData:0x05a, RamData:0x05a, RamData:0x05a],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:0x05a],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//     ],
//     [
//         [RamData:0x016, RamData:0x057, RamData:0x034, RamData:0x065, RamData:0x0a6, RamData:0x0db, RamData:0x05d, RamData:0x0b0],
//         [RamData:0x053, RamData:0x070, RamData:0x035, RamData:0x035, RamData:0x023, RamData:0x023, RamData:0x023, RamData:0x023],
//         [RamData:0x1a9, RamData:0x16d, RamData:0x04c, RamData:0x0fb, RamData:0x041, RamData:0x0c6, RamData:0x07b, RamData:0x060],
//         [RamData:0x05a, RamData:0x05a, RamData:0x05a, RamData:0x19b, RamData:0x10f, RamData:0x19c, RamData:0x1e1, RamData:0x1ba],
//         [RamData:0x05a, RamData:0x05a, RamData:0x05a, RamData:0x05a, RamData:0x05a, RamData:0x05a, RamData:0x05a, RamData:0x05a],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:0x05a],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//     ],
//     [
//         [RamData:0x016, RamData:0x057, RamData:0x034, RamData:0x065, RamData:0x0a6, RamData:0x0db, RamData:0x05d, RamData:0x0b0],
//         [RamData:0x053, RamData:0x070, RamData:0x035, RamData:0x035, RamData:0x023, RamData:0x023, RamData:0x023, RamData:0x023],
//         [RamData:0x1a9, RamData:0x16d, RamData:0x04c, RamData:0x0fb, RamData:0x041, RamData:0x0c6, RamData:0x07b, RamData:0x060],
//         [RamData:0x05a, RamData:0x05a, RamData:0x05a, RamData:0x19b, RamData:0x10f, RamData:0x19c, RamData:0x1e1, RamData:0x1ba],
//         [RamData:0x05a, RamData:0x05a, RamData:0x05a, RamData:0x05a, RamData:0x05a, RamData:0x05a, RamData:0x05a, RamData:0x05a],
//         [RamData:0x094, RamData:0x03e, RamData:0x096, RamData:0x018, RamData:0x034, RamData:0x0c2, RamData:0x047, RamData:0x05a],
//         [RamData:0x002, RamData:0x0d0, RamData:0x0e8, RamData:0x0d7, RamData:0x028, RamData:0x09a, RamData:0x0be, RamData:0x060],
//         [RamData:0x064, RamData:0x0c3, RamData:0x08b, RamData:0x0e1, RamData:0x0fa, RamData:0x08d, RamData:0x012, RamData:0x0bc],
//         [RamData:0x119, RamData:0x163, RamData:0x1f1, RamData:0x1ce, RamData:0x121, RamData:0x1c2, RamData:0x194, RamData:0x0f8],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//     ],
// ];

// const CYCLES_PER_RAM_READ = u32:16;

// #[test_proc]
// proc LiteralsDecoderRamContent_test {
//     terminator: chan<bool> out;

//     literals_ctrl_s: chan<LiteralsPathCtrl> out;
//     literals_data_s: chan<LiteralsData> out;
//     literals_buf_ctrl_s: chan<LiteralsBufferCtrl> out;
//     literals_r: chan<SequenceExecutorPacket> in;

//     ram_rd_req_m0_s: chan<TestReadReq> out;
//     ram_rd_req_m1_s: chan<TestReadReq> out;
//     ram_rd_req_m2_s: chan<TestReadReq> out;
//     ram_rd_req_m3_s: chan<TestReadReq> out;
//     ram_rd_req_m4_s: chan<TestReadReq> out;
//     ram_rd_req_m5_s: chan<TestReadReq> out;
//     ram_rd_req_m6_s: chan<TestReadReq> out;
//     ram_rd_req_m7_s: chan<TestReadReq> out;

//     ram_rd_resp_m0_r: chan<TestReadResp> in;
//     ram_rd_resp_m1_r: chan<TestReadResp> in;
//     ram_rd_resp_m2_r: chan<TestReadResp> in;
//     ram_rd_resp_m3_r: chan<TestReadResp> in;
//     ram_rd_resp_m4_r: chan<TestReadResp> in;
//     ram_rd_resp_m5_r: chan<TestReadResp> in;
//     ram_rd_resp_m6_r: chan<TestReadResp> in;
//     ram_rd_resp_m7_r: chan<TestReadResp> in;

//     config (terminator: chan<bool> out) {
//         let (literals_ctrl_s, literals_ctrl_r) = chan<LiteralsPathCtrl>("literals_ctrl");
//         let (literals_buf_ctrl_s, literals_buf_ctrl_r) = chan<LiteralsBufferCtrl>("literals_buf_ctrl");
//         let (literals_s, literals_r) = chan<SequenceExecutorPacket>("literals");

//         let (ram_rd_req_s,  ram_rd_req_r) = chan<TestReadReq>[literals_buffer::RAM_NUM]("ram_rd_req");
//         let (ram_rd_resp_s, ram_rd_resp_r) = chan<TestReadResp>[literals_buffer::RAM_NUM]("ram_rd_resp");
//         let (ram_wr_req_s,  ram_wr_req_r) = chan<TestWriteReq>[literals_buffer::RAM_NUM]("ram_wr_req");
//         let (ram_wr_resp_s, ram_wr_resp_r) = chan<TestWriteResp>[literals_buffer::RAM_NUM]("ram_wr_resp");

//         spawn LiteralsDecoder<TEST_HISTORY_BUFFER_SIZE_KB>(
//             literals_ctrl_r,
//             literals_buf_ctrl_r, literals_s,
//             ram_rd_req_s[0], ram_rd_req_s[1], ram_rd_req_s[2], ram_rd_req_s[3],
//             ram_rd_req_s[4], ram_rd_req_s[5], ram_rd_req_s[6], ram_rd_req_s[7],
//             ram_rd_resp_r[0], ram_rd_resp_r[1], ram_rd_resp_r[2], ram_rd_resp_r[3],
//             ram_rd_resp_r[4], ram_rd_resp_r[5], ram_rd_resp_r[6], ram_rd_resp_r[7],
//             ram_wr_req_s[0], ram_wr_req_s[1], ram_wr_req_s[2], ram_wr_req_s[3],
//             ram_wr_req_s[4], ram_wr_req_s[5], ram_wr_req_s[6], ram_wr_req_s[7],
//             ram_wr_resp_r[0], ram_wr_resp_r[1], ram_wr_resp_r[2], ram_wr_resp_r[3],
//             ram_wr_resp_r[4], ram_wr_resp_r[5], ram_wr_resp_r[6], ram_wr_resp_r[7]
//         );

//         spawn ram::RamModel<
//             literals_buffer::RAM_DATA_WIDTH, TEST_RAM_SIZE, literals_buffer::RAM_WORD_PARTITION_SIZE,
//             TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED>
//             (ram_rd_req_r[0], ram_rd_resp_s[0], ram_wr_req_r[0], ram_wr_resp_s[0]);
//         spawn ram::RamModel<
//             literals_buffer::RAM_DATA_WIDTH, TEST_RAM_SIZE, literals_buffer::RAM_WORD_PARTITION_SIZE,
//             TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED>
//             (ram_rd_req_r[1], ram_rd_resp_s[1], ram_wr_req_r[1], ram_wr_resp_s[1]);
//         spawn ram::RamModel<
//             literals_buffer::RAM_DATA_WIDTH, TEST_RAM_SIZE, literals_buffer::RAM_WORD_PARTITION_SIZE,
//             TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED>
//             (ram_rd_req_r[2], ram_rd_resp_s[2], ram_wr_req_r[2], ram_wr_resp_s[2]);
//         spawn ram::RamModel<
//             literals_buffer::RAM_DATA_WIDTH, TEST_RAM_SIZE, literals_buffer::RAM_WORD_PARTITION_SIZE,
//             TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED>
//             (ram_rd_req_r[3], ram_rd_resp_s[3], ram_wr_req_r[3], ram_wr_resp_s[3]);
//         spawn ram::RamModel<
//             literals_buffer::RAM_DATA_WIDTH, TEST_RAM_SIZE, literals_buffer::RAM_WORD_PARTITION_SIZE,
//             TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED>
//             (ram_rd_req_r[4], ram_rd_resp_s[4], ram_wr_req_r[4], ram_wr_resp_s[4]);
//         spawn ram::RamModel<
//             literals_buffer::RAM_DATA_WIDTH, TEST_RAM_SIZE, literals_buffer::RAM_WORD_PARTITION_SIZE,
//             TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED>
//             (ram_rd_req_r[5], ram_rd_resp_s[5], ram_wr_req_r[5], ram_wr_resp_s[5]);
//         spawn ram::RamModel<
//             literals_buffer::RAM_DATA_WIDTH, TEST_RAM_SIZE, literals_buffer::RAM_WORD_PARTITION_SIZE,
//             TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED>
//             (ram_rd_req_r[6], ram_rd_resp_s[6], ram_wr_req_r[6], ram_wr_resp_s[6]);
//         spawn ram::RamModel<
//             literals_buffer::RAM_DATA_WIDTH, TEST_RAM_SIZE, literals_buffer::RAM_WORD_PARTITION_SIZE,
//             TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED>
//             (ram_rd_req_r[7], ram_rd_resp_s[7], ram_wr_req_r[7], ram_wr_resp_s[7]);

//         (
//             terminator,
//             literals_ctrl_s, literals_data_s,
//             literals_buf_ctrl_s, literals_r,
//             ram_rd_req_s[0], ram_rd_req_s[1], ram_rd_req_s[2], ram_rd_req_s[3],
//             ram_rd_req_s[4], ram_rd_req_s[5], ram_rd_req_s[6], ram_rd_req_s[7],
//             ram_rd_resp_r[0], ram_rd_resp_r[1], ram_rd_resp_r[2], ram_rd_resp_r[3],
//             ram_rd_resp_r[4], ram_rd_resp_r[5], ram_rd_resp_r[6], ram_rd_resp_r[7],
//         )
//     }

//     init { u32:0 }

//     next (state: u32) {
//         // send literals
//         let ok = if (state == u32:0) {
//             for ((i, test_data), tok): ((u32, LiteralsData), token) in enumerate(TEST_DATA) {
//                 let tok = send(tok, literals_data_s, test_data);
//                 trace_fmt!("Sent #{} literals data, {:#x}", i + u32:1, test_data);
//                 tok
//             }(tok)
//         } else { tok };

//         // send ctrl and read RAM content
//         let tok = for ((i, test_ctrl), tok): ((u32, LiteralsPathCtrl), token) in enumerate(TEST_CTRL) {
//             if (state == i * CYCLES_PER_RAM_READ) {
//                 let tok = send(tok, literals_ctrl_s, test_ctrl);
//                 trace_fmt!("Sent #{} literals ctrl, {:#x}", i + u32:1, test_ctrl);
//                 tok
//             } else if (state == (i + u32:1) * CYCLES_PER_RAM_READ - u32:1) {
//                 for (addr, tok): (u32, token) in range(u32:0, u32:10) {
//                     let read_req = TestReadReq {
//                         addr: addr as uN[TEST_RAM_ADDR_WIDTH],
//                         mask: u1:1
//                     };

//                     let tok = send(tok, ram_rd_req_m0_s, read_req);
//                     let tok = send(tok, ram_rd_req_m1_s, read_req);
//                     let tok = send(tok, ram_rd_req_m2_s, read_req);
//                     let tok = send(tok, ram_rd_req_m3_s, read_req);
//                     let tok = send(tok, ram_rd_req_m4_s, read_req);
//                     let tok = send(tok, ram_rd_req_m5_s, read_req);
//                     let tok = send(tok, ram_rd_req_m6_s, read_req);
//                     let tok = send(tok, ram_rd_req_m7_s, read_req);

//                     let (tok, ram_rd_resp_m0) = recv(tok, ram_rd_resp_m0_r);
//                     let (tok, ram_rd_resp_m1) = recv(tok, ram_rd_resp_m1_r);
//                     let (tok, ram_rd_resp_m2) = recv(tok, ram_rd_resp_m2_r);
//                     let (tok, ram_rd_resp_m3) = recv(tok, ram_rd_resp_m3_r);
//                     let (tok, ram_rd_resp_m4) = recv(tok, ram_rd_resp_m4_r);
//                     let (tok, ram_rd_resp_m5) = recv(tok, ram_rd_resp_m5_r);
//                     let (tok, ram_rd_resp_m6) = recv(tok, ram_rd_resp_m6_r);
//                     let (tok, ram_rd_resp_m7) = recv(tok, ram_rd_resp_m7_r);
//                     trace_fmt!(
//                         "Received RAM read responses: [{:#x}, {:#x}, {:#x}, {:#x}, {:#x}, {:#x}, {:#x}, {:#x}]",
//                         ram_rd_resp_m7.data, ram_rd_resp_m6.data, ram_rd_resp_m5.data, ram_rd_resp_m4.data,
//                         ram_rd_resp_m3.data, ram_rd_resp_m2.data, ram_rd_resp_m1.data, ram_rd_resp_m0.data,
//                     );

//                     assert_eq(TEST_EXPECTED_RAM_CONTENT[i][addr][7], ram_rd_resp_m0.data);
//                     assert_eq(TEST_EXPECTED_RAM_CONTENT[i][addr][6], ram_rd_resp_m1.data);
//                     assert_eq(TEST_EXPECTED_RAM_CONTENT[i][addr][5], ram_rd_resp_m2.data);
//                     assert_eq(TEST_EXPECTED_RAM_CONTENT[i][addr][4], ram_rd_resp_m3.data);
//                     assert_eq(TEST_EXPECTED_RAM_CONTENT[i][addr][3], ram_rd_resp_m4.data);
//                     assert_eq(TEST_EXPECTED_RAM_CONTENT[i][addr][2], ram_rd_resp_m5.data);
//                     assert_eq(TEST_EXPECTED_RAM_CONTENT[i][addr][1], ram_rd_resp_m6.data);
//                     assert_eq(TEST_EXPECTED_RAM_CONTENT[i][addr][0], ram_rd_resp_m7.data);

//                     tok
//                 }(tok)
//             } else {
//                 tok
//             }
//         }(tok);

//         send_if(tok, terminator, state == array_size(TEST_CTRL) * CYCLES_PER_RAM_READ, true);

//         state + u32:1
//     }
// }
