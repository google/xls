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

// This file contains work-in-progress ZSTD decoder implementation
// More information about ZSTD decoding can be found in:
// https://datatracker.ietf.org/doc/html/rfc8878

import std;
import xls.examples.ram;
import xls.modules.zstd.axi_csr_accessor;
import xls.modules.zstd.common;
import xls.modules.zstd.memory.axi;
import xls.modules.zstd.csr_config;
import xls.modules.zstd.memory.mem_reader;
import xls.modules.zstd.frame_header_dec;
import xls.modules.zstd.block_header;
import xls.modules.zstd.block_header_dec;
import xls.modules.zstd.raw_block_dec;
import xls.modules.zstd.rle_block_dec;
import xls.modules.zstd.dec_mux;
import xls.modules.zstd.sequence_executor;
import xls.modules.zstd.repacketizer;

type BlockSize = common::BlockSize;
type BlockType = common::BlockType;
type BlockHeader = block_header::BlockHeader;

enum ZstdDecoderInternalFsm: u4 {
    IDLE = 0,
    READ_CONFIG = 1,
    DECODE_FRAME_HEADER = 2,
    DECODE_BLOCK_HEADER = 3,
    DECODE_RAW_BLOCK = 4,
    DECODE_RLE_BLOCK = 5,
    DECODE_COMPRESSED_BLOCK = 6,
    DECODE_CHECKSUM = 7,
    FINISH = 8,
    ERROR = 13,
    INVALID = 15,
}

enum ZstdDecoderStatus: u3 {
    OKAY = 0,
    FINISHED = 1,
    FRAME_HEADER_CORRUPTED = 2,
    FRAME_HEADER_UNSUPPORTED_WINDOW_SIZE = 3,
    BLOCK_HEADER_CORRUPTED = 4,
}

pub enum Csr: u3 {
    STATUS = 0,         // Keeps the code describing the current state of the ZSTD Decoder
    START = 1,          // Writing 1 when decoder is in IDLE state starts the decoding process
    RESET = 2,          // Writing 1 will reset the decoder to the IDLE state
    INPUT_BUFFER = 3,   // Keeps the base address for the input buffer that is used for storing the frame to decode
    OUTPUT_BUFFER = 4,  // Keeps the base address for the output buffer, ZSTD Decoder will write the decoded frame into memory starting from this address.
    WHO_AM_I = 5,       // Contains the identification number of the ZSTD Decoder
}

fn csr<LOG2_REGS_N: u32>(c: Csr) -> uN[LOG2_REGS_N] {
    c as uN[LOG2_REGS_N]
}

struct ZstdDecoderInternalState<AXI_DATA_W: u32, AXI_ADDR_W: u32, LOG2_REGS_N: u32> {
    fsm: ZstdDecoderInternalFsm,

    // Reading CSRs
    conf_cnt: uN[LOG2_REGS_N],
    conf_send: bool,
    input_buffer: uN[AXI_ADDR_W],
    input_buffer_valid: bool,
    output_buffer: uN[AXI_ADDR_W],
    output_buffer_valid: bool,

    // Writing to CSRs
    csr_wr_req: csr_config::CsrWrReq<LOG2_REGS_N, AXI_DATA_W>,
    csr_wr_req_valid: bool,

    // BH address
    bh_addr: uN[AXI_ADDR_W],

    // Block
    block_addr: uN[AXI_ADDR_W],
    block_length: uN[AXI_ADDR_W],
    block_last: bool,
    block_id: u32,
    block_rle_symbol: u8,

    // Req
    req_sent: bool,
}

proc ZstdDecoderInternal<
    AXI_DATA_W: u32, AXI_ADDR_W: u32, REGS_N: u32,
    LOG2_REGS_N:u32 = {std::clog2(REGS_N)},
    HB_RAM_N:u32 = {u32:8},
> {

    type State = ZstdDecoderInternalState<AXI_DATA_W, AXI_ADDR_W, LOG2_REGS_N>;
    type Fsm = ZstdDecoderInternalFsm;
    type Reg = uN[LOG2_REGS_N];
    type Data = uN[AXI_DATA_W];
    type Addr = uN[AXI_ADDR_W];

    type CsrRdReq = csr_config::CsrRdReq<LOG2_REGS_N>;
    type CsrRdResp = csr_config::CsrRdResp<LOG2_REGS_N, AXI_DATA_W>;
    type CsrWrReq = csr_config::CsrWrReq<LOG2_REGS_N, AXI_DATA_W>;
    type CsrWrResp = csr_config::CsrWrResp;
    type CsrChange = csr_config::CsrChange<LOG2_REGS_N>;

    type MemReaderReq  = mem_reader::MemReaderReq<AXI_ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<AXI_DATA_W, AXI_ADDR_W>;

    type FrameHeaderDecoderStatus = frame_header_dec::FrameHeaderDecoderStatus;
    type FrameHeaderDecoderReq = frame_header_dec::FrameHeaderDecoderReq<AXI_ADDR_W>;
    type FrameHeaderDecoderResp = frame_header_dec::FrameHeaderDecoderResp;

    type BlockHeaderDecoderStatus = block_header_dec::BlockHeaderDecoderStatus;
    type BlockHeaderDecoderReq = block_header_dec::BlockHeaderDecoderReq<AXI_ADDR_W>;
    type BlockHeaderDecoderResp = block_header_dec::BlockHeaderDecoderResp;

    type RawBlockDecoderStatus = raw_block_dec::RawBlockDecoderStatus;
    type RawBlockDecoderReq = raw_block_dec::RawBlockDecoderReq<AXI_ADDR_W>;
    type RawBlockDecoderResp = raw_block_dec::RawBlockDecoderResp;

    type RleBlockDecoderStatus = rle_block_dec::RleBlockDecoderStatus;
    type RleBlockDecoderReq = rle_block_dec::RleBlockDecoderReq<AXI_ADDR_W>;
    type RleBlockDecoderResp = rle_block_dec::RleBlockDecoderResp;

    // CsrConfig
    csr_rd_req_s: chan<CsrRdReq> out;
    csr_rd_resp_r: chan<CsrRdResp> in;
    csr_wr_req_s: chan<CsrWrReq> out;
    csr_wr_resp_r: chan<CsrWrResp> in;
    csr_change_r: chan<CsrChange> in;

    // MemReader + FameHeaderDecoder
    fh_req_s: chan<FrameHeaderDecoderReq> out;
    fh_resp_r: chan<FrameHeaderDecoderResp> in;

    // MemReader + BlockHeaderDecoder
    bh_req_s: chan<BlockHeaderDecoderReq> out;
    bh_resp_r: chan<BlockHeaderDecoderResp> in;

    // MemReader + RawBlockDecoder
    raw_req_s: chan<RawBlockDecoderReq> out;
    raw_resp_r: chan<RawBlockDecoderResp> in;

    // MemReader + RleBlockDecoder
    rle_req_s: chan<RleBlockDecoderReq> out;
    rle_resp_r: chan<RleBlockDecoderResp> in;

    notify_s: chan<()> out;
    reset_s: chan<()> out;

    init {
        zero!<State>()
    }

    config(
        csr_rd_req_s: chan<CsrRdReq> out,
        csr_rd_resp_r: chan<CsrRdResp> in,
        csr_wr_req_s: chan<CsrWrReq> out,
        csr_wr_resp_r: chan<CsrWrResp> in,
        csr_change_r: chan<CsrChange> in,

        // MemReader + FameHeaderDecoder
        fh_req_s: chan<FrameHeaderDecoderReq> out,
        fh_resp_r: chan<FrameHeaderDecoderResp> in,

        // MemReader + BlockHeaderDecoder
        bh_req_s: chan<BlockHeaderDecoderReq> out,
        bh_resp_r: chan<BlockHeaderDecoderResp> in,

        // MemReader + RawBlockDecoder
        raw_req_s: chan<RawBlockDecoderReq> out,
        raw_resp_r: chan<RawBlockDecoderResp> in,

        // MemReader + RleBlockDecoder
        rle_req_s: chan<RleBlockDecoderReq> out,
        rle_resp_r: chan<RleBlockDecoderResp> in,

        notify_s: chan<()> out,
        reset_s: chan<()> out,
    ) {
        (
            csr_rd_req_s, csr_rd_resp_r, csr_wr_req_s, csr_wr_resp_r, csr_change_r,
            fh_req_s, fh_resp_r,
            bh_req_s, bh_resp_r,
            raw_req_s, raw_resp_r,
            rle_req_s, rle_resp_r,
            notify_s, reset_s,
        )
    }

    next (state: State) {
        let tok0 = join();

        const CSR_REQS = CsrRdReq<LOG2_REGS_N>[2]:[
            CsrRdReq {csr: csr<LOG2_REGS_N>(Csr::INPUT_BUFFER)},
            CsrRdReq {csr: csr<LOG2_REGS_N>(Csr::OUTPUT_BUFFER)}
        ];

        const CSR_REQS_MAX = checked_cast<Reg>(array_size(CSR_REQS) - u32:1);

        let (tok1_0, csr_change, csr_change_valid) = recv_non_blocking(tok0, csr_change_r, zero!<CsrChange>());
        let is_start = (csr_change_valid && (csr_change.csr == csr<LOG2_REGS_N>(Csr::START)));

        let is_reset = (csr_change_valid && (csr_change.csr == csr<LOG2_REGS_N>(Csr::RESET)));
        let tok = send_if(tok0, reset_s, is_reset, ());
        if is_reset {
            trace_fmt!("[[RESET]]");
        } else {};

        if csr_change_valid {
            trace_fmt!("[CSR CHANGE] {:#x}", csr_change);
        } else {};

        let do_send_csr_req = (state.fsm == Fsm::READ_CONFIG) && (!state.conf_send);
        let csr_req = CSR_REQS[state.conf_cnt];
        let tok1_1 = send_if(tok0, csr_rd_req_s, do_send_csr_req, csr_req);
        if do_send_csr_req {
            trace_fmt!("[READ_CONFIG] Sending read request {:#x}", csr_req);
        } else {};

        let do_recv_csr_resp = (state.fsm == Fsm::READ_CONFIG);
        let (tok1_2, csr_data, csr_data_valid) = recv_if_non_blocking(tok0, csr_rd_resp_r, do_recv_csr_resp, zero!<CsrRdResp>());
        if csr_data_valid {
            trace_fmt!("[READ_CONFIG] Received CSR data: {:#x}", csr_data);
        } else {};

        let do_send_fh_req = (state.fsm == Fsm::DECODE_FRAME_HEADER) && !state.req_sent;
        let fh_req = FrameHeaderDecoderReq { addr: state.input_buffer };
        let tok1_3 = send_if(tok0, fh_req_s, do_send_fh_req, fh_req);
        if do_send_fh_req {
            trace_fmt!("[DECODE_FRAME_HEADER] Sending FH request {:#x}", fh_req);
        } else {};

        let do_recv_fh_resp = (state.fsm == Fsm::DECODE_FRAME_HEADER);
        let (tok1_4, fh_resp, fh_resp_valid) = recv_if_non_blocking(tok0, fh_resp_r, do_recv_fh_resp, zero!<FrameHeaderDecoderResp>());
        if fh_resp_valid {
            trace_fmt!("[DECODE_FRAME_HEADER]: Received FH {:#x}", fh_resp);
        } else {};

        let do_send_notify = (state.fsm == Fsm::ERROR || state.fsm == Fsm::FINISH);
        let tok = send_if(tok0, notify_s, do_send_notify, ());
        if do_send_notify {
            trace_fmt!("[[NOTIFY]]");
        } else {};

        let tok1_5 = send_if(tok0, csr_wr_req_s, state.csr_wr_req_valid, state.csr_wr_req);
        let (tok, _, _) = recv_non_blocking(tok0, csr_wr_resp_r, zero!<CsrWrResp>());
        if state.csr_wr_req_valid {
            trace_fmt!("[[CSR_WR_REQ]] Request: {:#x}", state.csr_wr_req);
        } else {};

        let do_send_bh_req = (state.fsm == Fsm::DECODE_BLOCK_HEADER) && !state.req_sent;
        let bh_req = BlockHeaderDecoderReq { addr: state.bh_addr };
        let tok1_6 = send_if(tok0, bh_req_s, do_send_bh_req, bh_req);
        if do_send_bh_req {
            trace_fmt!("[DECODE_BLOCK_HEADER]: Sending BH request: {:#x}", bh_req);
        } else {};

        let do_recv_bh_resp = (state.fsm == Fsm::DECODE_BLOCK_HEADER);
        let (tok1_4, bh_resp, bh_resp_valid) = recv_if_non_blocking(tok0, bh_resp_r, do_recv_bh_resp, zero!<BlockHeaderDecoderResp>());
        if bh_resp_valid {
            trace_fmt!("[DECODE_BLOCK_HEADER]: Received BH {:#x}", bh_resp);
        } else {};

        let do_send_raw_req = (state.fsm == Fsm::DECODE_RAW_BLOCK) && !state.req_sent;
        let raw_req = RawBlockDecoderReq {
            id: state.block_id,
            last_block: state.block_last,
            addr: state.block_addr,
            length: state.block_length,
        };
        let tok1_6 = send_if(tok0, raw_req_s, do_send_raw_req, raw_req);
        if do_send_raw_req {
            trace_fmt!("[DECODE_RAW_BLOCK]: Sending RAW request: {:#x}", raw_req);
        } else {};

        let do_recv_raw_resp = (state.fsm == Fsm::DECODE_RAW_BLOCK);
        let (tok1_7, raw_resp, raw_resp_valid) = recv_if_non_blocking(tok0, raw_resp_r, do_recv_raw_resp, zero!<RawBlockDecoderResp>());
        if raw_resp_valid {
            trace_fmt!("[DECODE_RAW_BLOCK]: Received RAW {:#x}", raw_resp);
        } else {};

        let do_send_rle_req = (state.fsm == Fsm::DECODE_RLE_BLOCK) && !state.req_sent;
        let rle_req = RleBlockDecoderReq {
            id: state.block_id,
            symbol: state.block_rle_symbol,
            length: checked_cast<BlockSize>(state.block_length),
            last_block: state.block_last,
        };
        let tok1_7 = send_if(tok0, rle_req_s, do_send_rle_req, rle_req);
        if do_send_rle_req {
            trace_fmt!("[DECODE_RLE_BLOCK]: Sending RLE request: {:#x}", rle_req);
        } else {};

        let do_recv_rle_resp = (state.fsm == Fsm::DECODE_RLE_BLOCK);
        let (tok1_8, rle_resp, rle_resp_valid) = recv_if_non_blocking(tok0, rle_resp_r, do_recv_rle_resp, zero!<RleBlockDecoderResp>());
        if raw_resp_valid {
            trace_fmt!("[DECODE_RLE_BLOCK]: Received RAW {:#x}", raw_resp);
        } else {};

        let new_state = match (state.fsm) {
            Fsm::IDLE => {
                trace_fmt!("[IDLE]");
                 if is_start {
                     State { fsm: Fsm::READ_CONFIG, conf_cnt: CSR_REQS_MAX, ..zero!<State>() }
                 } else { zero!<State>() }
            },

            Fsm::READ_CONFIG => {
                trace_fmt!("[READ_CONFIG]");
                let is_input_buffer_csr = (csr_data.csr == csr<LOG2_REGS_N>(Csr::INPUT_BUFFER));
                let input_buffer = if csr_data_valid && is_input_buffer_csr { checked_cast<Addr>(csr_data.value) } else { state.input_buffer };
                let input_buffer_valid = if csr_data_valid && is_input_buffer_csr { true } else { state.input_buffer_valid };

                let is_output_buffer_csr = (csr_data.csr == csr<LOG2_REGS_N>(Csr::OUTPUT_BUFFER));
                let output_buffer = if (csr_data_valid && is_output_buffer_csr) { checked_cast<Addr>(csr_data.value) } else { state.output_buffer };
                let output_buffer_valid = if (csr_data_valid && is_output_buffer_csr) { true } else { state.output_buffer_valid };

                let all_collected = input_buffer_valid & output_buffer_valid;
                let fsm = if all_collected { Fsm::DECODE_FRAME_HEADER } else { Fsm::READ_CONFIG };

                let conf_send = (state.conf_cnt == Reg:0);
                let conf_cnt = if conf_send { Reg:0 } else {state.conf_cnt - Reg:1};

                State {
                    fsm, conf_cnt, conf_send, input_buffer, input_buffer_valid, output_buffer, output_buffer_valid,
                ..zero!<State>()
                }
            },

            Fsm::DECODE_FRAME_HEADER => {
                trace_fmt!("[DECODE_FRAME_HEADER]");
                let error = (fh_resp.status != FrameHeaderDecoderStatus::OKAY);

                let csr_wr_req_valid = (fh_resp_valid && error);
                let csr_wr_req = CsrWrReq {
                    csr: csr<LOG2_REGS_N>(Csr::STATUS),
                    value: checked_cast<Data>(fh_resp.status),
                };

                let fsm = match (fh_resp_valid, error) {
                    ( true,  false) => Fsm::DECODE_BLOCK_HEADER,
                    ( true,   true) => Fsm::ERROR,
                    (    _,      _) => Fsm::DECODE_FRAME_HEADER,
                };

                let bh_addr = state.input_buffer + fh_resp.length as Addr;
                let req_sent = if !fh_resp_valid && !error { true } else { false };
                State {fsm, csr_wr_req, csr_wr_req_valid, bh_addr, req_sent, ..state }
            },

            Fsm::DECODE_BLOCK_HEADER => {
                trace_fmt!("[DECODE_BLOCK_HEADER]");
                let error = (bh_resp.status != BlockHeaderDecoderStatus::OKAY);

                let csr_wr_req_valid = (bh_resp_valid && error);
                let csr_wr_req = CsrWrReq {
                    csr:   csr<LOG2_REGS_N>(Csr::STATUS),
                    value: checked_cast<Data>(bh_resp.status),
                };

                let fsm = match (bh_resp_valid, error, bh_resp.header.btype) {
                    ( true,  false, BlockType::RAW       ) => Fsm::DECODE_RAW_BLOCK,
                    ( true,  false, BlockType::RLE       ) => Fsm::DECODE_RLE_BLOCK,
                    ( true,  false, BlockType::COMPRESSED) => Fsm::ERROR,
                    ( true,   true,                     _) => Fsm::ERROR,
                    (    _,      _,                     _) => Fsm::DECODE_BLOCK_HEADER,
                };

                let (block_addr, block_length, block_last, block_rle_symbol, bh_addr) = if bh_resp_valid {
                    let block_addr = state.bh_addr + Addr:3;
                    let block_length = checked_cast<Addr>(bh_resp.header.size);
                    let block_rle_symbol = bh_resp.rle_symbol;
                    let bh_addr = if bh_resp.header.btype == BlockType::RLE {
                        block_addr + Addr:1
                    } else {
                        block_addr + block_length
                    };

                    trace_fmt!("bh_addr: {:#x}", bh_addr);

                    (block_addr, block_length, bh_resp.header.last, block_rle_symbol, bh_addr)
                } else {
                    (state.block_addr, state.block_length, state.block_last, state.block_rle_symbol, state.bh_addr)
                };

                let req_sent = if !bh_resp_valid && !error { true } else { false };
                State {
                    fsm, bh_addr, req_sent,
                    block_addr, block_length, block_last, block_rle_symbol,
                    csr_wr_req, csr_wr_req_valid,
                    ..state
                }
            },

            Fsm::DECODE_RAW_BLOCK => {
                trace_fmt!("[DECODE_RAW_BLOCK]");

                let error = (raw_resp.status != RawBlockDecoderStatus::OKAY);

                let csr_wr_req_valid = (raw_resp_valid && error);
                let csr_wr_req = CsrWrReq {
                    csr:   csr<LOG2_REGS_N>(Csr::STATUS),
                    value: checked_cast<Data>(raw_resp.status),
                };

                let fsm = match (raw_resp_valid, error, state.block_last) {
                    (true,  false,  false) => Fsm::DECODE_BLOCK_HEADER,
                    (true,  false,   true) => Fsm::DECODE_CHECKSUM,
                    (true,   true,      _) => Fsm::ERROR,
                    (   _,      _,      _) => Fsm::DECODE_RAW_BLOCK,
                };

                let req_sent = if !raw_resp_valid && !error { true } else { false };
                let block_id = if raw_resp_valid { state.block_id + u32:1} else {state.block_id };

                let state = State {fsm, block_id, csr_wr_req, csr_wr_req_valid, req_sent, ..state};
                if fsm == Fsm::DECODE_BLOCK_HEADER {
                    trace_fmt!("Going to decode block header: {:#x}", state);
                } else {};

                state
            },

            Fsm::DECODE_RLE_BLOCK => {
                trace_fmt!("[DECODE_RLE_BLOCK]");
                let error = (rle_resp.status != RleBlockDecoderStatus::OKAY);

                let csr_wr_req_valid = (rle_resp_valid && error);
                let csr_wr_req = CsrWrReq {
                    csr:   csr<LOG2_REGS_N>(Csr::STATUS),
                    value: checked_cast<Data>(rle_resp.status),
                };

                let fsm = match (rle_resp_valid, error, state.block_last) {
                    (true,  false,  false) => Fsm::DECODE_BLOCK_HEADER,
                    (true,  false,   true) => Fsm::DECODE_CHECKSUM,
                    (true,   true,      _) => Fsm::ERROR,
                    (   _,      _,      _) => Fsm::DECODE_RLE_BLOCK,
                };

                let req_sent = if !rle_resp_valid && !error { true } else { false };
                let block_id = if rle_resp_valid { state.block_id + u32:1} else {state.block_id };

                let state = State {fsm, block_id, csr_wr_req, csr_wr_req_valid, req_sent, ..state};
                if fsm == Fsm::DECODE_BLOCK_HEADER {
                    trace_fmt!("Going to decode block header: {:#x}", state);
                } else {};

                state
            },

            Fsm::DECODE_CHECKSUM => {
                trace_fmt!("[DECODE_CHECKSUM]");
                State {fsm: Fsm::FINISH, ..zero!<State>() }

            },

            Fsm::ERROR => {
                 trace_fmt!("[ERROR]");
                 State { fsm: Fsm::IDLE, ..zero!<State>() }
            },

            Fsm::FINISH => {
                trace_fmt!("[FINISH]");
                let csr_wr_req_valid = true;
                let csr_wr_req = CsrWrReq {
                    csr: csr<LOG2_REGS_N>(Csr::STATUS),
                    value: checked_cast<Data>(fh_resp.status),
                };

                State { fsm: Fsm::IDLE, csr_wr_req, csr_wr_req_valid, ..zero!<State>() }
            },

            _ => zero!<State>(),
        };

        new_state
    }
}

const TEST_AXI_DATA_W = u32:64;
const TEST_AXI_ADDR_W = u32:32;
const TEST_REGS_N = u32:5;
const TEST_LOG2_REGS_N = std::clog2(TEST_REGS_N);

#[test_proc]
proc ZstdDecoderInternalTest {

    type BlockType = common::BlockType;
    type BlockSize = common::BlockSize;
    type BlockHeader = block_header::BlockHeader;
    type BlockHeaderDecoderStatus = block_header_dec::BlockHeaderDecoderStatus;

    type FrameHeaderDecoderReq = frame_header_dec::FrameHeaderDecoderReq;
    type FrameHeaderDecoderResp = frame_header_dec::FrameHeaderDecoderResp;
    type FrameHeaderDecoderStatus = frame_header_dec::FrameHeaderDecoderStatus;
    type FrameContentSize = frame_header_dec::FrameContentSize;
    type FrameHeader = frame_header_dec::FrameHeader;
    type WindowSize = frame_header_dec::WindowSize;
    type DictionaryId = frame_header_dec::DictionaryId;

    type CsrRdReq = csr_config::CsrRdReq<TEST_LOG2_REGS_N>;
    type CsrRdResp = csr_config::CsrRdResp<TEST_LOG2_REGS_N, TEST_AXI_DATA_W>;
    type CsrWrReq = csr_config::CsrWrReq<TEST_LOG2_REGS_N, TEST_AXI_DATA_W>;
    type CsrWrResp = csr_config::CsrWrResp;
    type CsrChange = csr_config::CsrChange<TEST_LOG2_REGS_N>;

    type FrameHeaderDecoderReq = frame_header_dec::FrameHeaderDecoderReq<TEST_AXI_ADDR_W>;
    type FrameHeaderDecoderResp = frame_header_dec::FrameHeaderDecoderResp;

    type BlockHeaderDecoderReq = block_header_dec::BlockHeaderDecoderReq<TEST_AXI_ADDR_W>;
    type BlockHeaderDecoderResp = block_header_dec::BlockHeaderDecoderResp;

    type RawBlockDecoderReq = raw_block_dec::RawBlockDecoderReq<TEST_AXI_ADDR_W>;
    type RawBlockDecoderResp = raw_block_dec::RawBlockDecoderResp;
    type RawBlockDecoderStatus = raw_block_dec::RawBlockDecoderStatus;

    type RleBlockDecoderReq = rle_block_dec::RleBlockDecoderReq<TEST_AXI_ADDR_W>;
    type RleBlockDecoderResp = rle_block_dec::RleBlockDecoderResp;
    type RleBlockDecoderStatus = rle_block_dec::RleBlockDecoderStatus;

    terminator: chan<bool> out;

    csr_rd_req_r: chan<CsrRdReq> in;
    csr_rd_resp_s: chan<CsrRdResp> out;
    csr_wr_req_r: chan<CsrWrReq> in;
    csr_wr_resp_s: chan<CsrWrResp> out;
    csr_change_s: chan<CsrChange> out;

    fh_req_r: chan<FrameHeaderDecoderReq> in;
    fh_resp_s: chan<FrameHeaderDecoderResp> out;

    bh_req_r:  chan<BlockHeaderDecoderReq> in;
    bh_resp_s: chan<BlockHeaderDecoderResp> out;

    raw_req_r: chan<RawBlockDecoderReq> in;
    raw_resp_s: chan<RawBlockDecoderResp> out;

    rle_req_r: chan<RleBlockDecoderReq> in;
    rle_resp_s: chan<RleBlockDecoderResp> out;

    notify_r: chan<()> in;
    reset_r: chan<()> in;

    init {}

    config(terminator: chan<bool> out) {
        let (csr_rd_req_s, csr_rd_req_r) = chan<CsrRdReq>("csr_rd_req");
        let (csr_rd_resp_s, csr_rd_resp_r) = chan<CsrRdResp>("csr_rd_resp");
        let (csr_wr_req_s, csr_wr_req_r) = chan<CsrWrReq>("csr_wr_req");
        let (csr_wr_resp_s, csr_wr_resp_r) = chan<CsrWrResp>("csr_wr_resp");
        let (csr_change_s, csr_change_r) = chan<CsrChange>("csr_change");

        let (fh_req_s, fh_req_r) = chan<FrameHeaderDecoderReq>("fh_req");
        let (fh_resp_s, fh_resp_r) = chan<FrameHeaderDecoderResp>("fh_resp");

        let (bh_req_s, bh_req_r) = chan<BlockHeaderDecoderReq>("bh_req");
        let (bh_resp_s, bh_resp_r) = chan<BlockHeaderDecoderResp>("bh_resp");

        let (raw_req_s,  raw_req_r) = chan<RawBlockDecoderReq>("raw_req");
        let (raw_resp_s, raw_resp_r) = chan<RawBlockDecoderResp>("raw_resp");

        let (rle_req_s, rle_req_r) = chan<RleBlockDecoderReq>("rle_req");
        let (rle_resp_s, rle_resp_r) = chan<RleBlockDecoderResp>("rle_resp");

        let (notify_s, notify_r) = chan<()>("notify");
        let (reset_s, reset_r) = chan<()>("reset");

        spawn ZstdDecoderInternal<TEST_AXI_DATA_W, TEST_AXI_ADDR_W, TEST_REGS_N>(
            csr_rd_req_s, csr_rd_resp_r, csr_wr_req_s, csr_wr_resp_r, csr_change_r,
            fh_req_s, fh_resp_r,
            bh_req_s, bh_resp_r,
            raw_req_s, raw_resp_r,
            rle_req_s, rle_resp_r,
            notify_s, reset_s,
        );

        (
            terminator,
            csr_rd_req_r, csr_rd_resp_s, csr_wr_req_r, csr_wr_resp_s, csr_change_s,
            fh_req_r, fh_resp_s,
            bh_req_r, bh_resp_s,
            raw_req_r, raw_resp_s,
            rle_req_r, rle_resp_s,
            notify_r, reset_r,
        )
    }

    next (state: ()) {
        type Addr = uN[TEST_AXI_ADDR_W];
        type Length = uN[TEST_AXI_ADDR_W];

        let tok = join();

        // Error in frame header

        let tok = send(tok, csr_change_s, CsrChange { csr: csr<TEST_LOG2_REGS_N>(Csr::START)});
        let (tok, csr_data) = recv(tok, csr_rd_req_r);
        assert_eq(csr_data, CsrRdReq {csr: csr<TEST_LOG2_REGS_N>(Csr::OUTPUT_BUFFER)});
        let (tok, csr_data) = recv(tok, csr_rd_req_r);
        assert_eq(csr_data, CsrRdReq {csr: csr<TEST_LOG2_REGS_N>(Csr::INPUT_BUFFER)});

        send(tok, csr_rd_resp_s, CsrRdResp {
            csr: csr<TEST_LOG2_REGS_N>(Csr::INPUT_BUFFER),
            value: uN[TEST_AXI_DATA_W]:0x1000
        });
        send(tok, csr_rd_resp_s, CsrRdResp {
            csr: csr<TEST_LOG2_REGS_N>(Csr::OUTPUT_BUFFER),
            value: uN[TEST_AXI_DATA_W]:0x2000
        });
        let (tok, fh_req) = recv(tok, fh_req_r);
        assert_eq(fh_req, FrameHeaderDecoderReq { addr: Addr:0x1000 });

        let tok = send(tok, fh_resp_s, FrameHeaderDecoderResp {
            status: FrameHeaderDecoderStatus::CORRUPTED,
            header: FrameHeader {
                window_size: WindowSize:100,
                frame_content_size: FrameContentSize:200,
                dictionary_id: DictionaryId:123,
                content_checksum_flag: u1:1,
            },
            length: u5:3,
        });


        let (tok, ()) = recv(tok, notify_r);

        // Correct case
        let tok = send(tok, csr_change_s, CsrChange { csr: csr<TEST_LOG2_REGS_N>(Csr::START)});
        let (tok, csr_data) = recv(tok, csr_rd_req_r);
        assert_eq(csr_data, CsrRdReq {csr: csr<TEST_LOG2_REGS_N>(Csr::OUTPUT_BUFFER)});
        let (tok, csr_data) = recv(tok, csr_rd_req_r);
        assert_eq(csr_data, CsrRdReq {csr: csr<TEST_LOG2_REGS_N>(Csr::INPUT_BUFFER)});

        send(tok, csr_rd_resp_s, CsrRdResp {
            csr: csr<TEST_LOG2_REGS_N>(Csr::INPUT_BUFFER),
            value: uN[TEST_AXI_DATA_W]:0x1000
        });
        send(tok, csr_rd_resp_s, CsrRdResp {
            csr: csr<TEST_LOG2_REGS_N>(Csr::OUTPUT_BUFFER),
            value: uN[TEST_AXI_DATA_W]:0x2000
        });
        let (tok, fh_req) = recv(tok, fh_req_r);
        assert_eq(fh_req, FrameHeaderDecoderReq { addr: Addr:0x1000 });

        let tok = send(tok, fh_resp_s, FrameHeaderDecoderResp {
            status: FrameHeaderDecoderStatus::OKAY,
            header: FrameHeader {
                window_size: WindowSize:100,
                frame_content_size: FrameContentSize:200,
                dictionary_id: DictionaryId:123,
                content_checksum_flag: u1:1,
            },
            length: u5:3,
        });

        let (tok, bh_req) = recv(tok, bh_req_r);
        assert_eq(bh_req, BlockHeaderDecoderReq {
            addr: Addr:0x1003,
        });

        let tok = send(tok, bh_resp_s, BlockHeaderDecoderResp {
            status: BlockHeaderDecoderStatus::OKAY,
            header: BlockHeader {
                last: false,
                btype: BlockType::RAW,
                size: BlockSize:0x1000,
            },
            rle_symbol: u8:0,
        });

        let (tok, raw_req) = recv(tok, raw_req_r);
        assert_eq(raw_req, RawBlockDecoderReq {
            last_block: false,
            id: u32:0,
            addr: Addr:0x1006,
            length: Length:0x1000
        });

        let tok = send(tok, raw_resp_s, RawBlockDecoderResp {
            status: RawBlockDecoderStatus::OKAY,
        });

        let (tok, bh_req) = recv(tok, bh_req_r);
        assert_eq(bh_req, BlockHeaderDecoderReq {
            addr: Addr:0x2006
        });
        let tok = send(tok, bh_resp_s, BlockHeaderDecoderResp {
            status: BlockHeaderDecoderStatus::OKAY,
            header: BlockHeader {
                last: false,
                btype: BlockType::RLE,
                size: BlockSize:0x1000,
            },
            rle_symbol: u8:123,
        });

        let (tok, rle_req) = recv(tok, rle_req_r);
        assert_eq(rle_req, RleBlockDecoderReq {
            id: u32:1,
            symbol: u8:123,
            last_block: false,
            length: checked_cast<BlockSize>(Length:0x1000),
        });
        let tok = send(tok, rle_resp_s, RleBlockDecoderResp {
            status: RleBlockDecoderStatus::OKAY,
        });

        let (tok, bh_req) = recv(tok, bh_req_r);
        assert_eq(bh_req, BlockHeaderDecoderReq {
            addr: Addr:0x200A,
        });

        let tok = send(tok, bh_resp_s, BlockHeaderDecoderResp {
            status: BlockHeaderDecoderStatus::OKAY,
            header: BlockHeader {
                last: true,
                btype: BlockType::RAW,
                size: BlockSize:0x1000,
            },
            rle_symbol: u8:0,
        });

        let (tok, raw_req) = recv(tok, raw_req_r);
        assert_eq(raw_req, RawBlockDecoderReq {
            last_block: true,
            id: u32:2,
            addr: Addr:0x200D,
            length: Length:0x1000
        });

        let tok = send(tok, raw_resp_s, RawBlockDecoderResp {
            status: RawBlockDecoderStatus::OKAY,
        });

        let (tok, ()) = recv(tok, notify_r);

        send(tok, terminator, true);
    }
}


pub proc ZstdDecoder<
    // AXI parameters
    AXI_DATA_W: u32, AXI_ADDR_W: u32, AXI_ID_W: u32, AXI_DEST_W: u32,
    // decoder parameters
    REGS_N: u32, WINDOW_LOG_MAX: u32,
    HB_ADDR_W: u32, HB_DATA_W: u32, HB_NUM_PARTITIONS: u32, HB_SIZE_KB: u32,
    // calculated parameters
    AXI_DATA_W_DIV8: u32 = {AXI_DATA_W / u32:8},
    LOG2_REGS_N: u32 = {std::clog2(REGS_N)},
    HB_RAM_N: u32 = {u32:8},
> {
    type CsrAxiAr = axi::AxiAr<AXI_ADDR_W, AXI_ID_W>;
    type CsrAxiR = axi::AxiR<AXI_DATA_W, AXI_ID_W>;
    type CsrAxiAw = axi::AxiAw<AXI_ADDR_W, AXI_ID_W>;
    type CsrAxiW = axi::AxiW<AXI_DATA_W, AXI_DATA_W_DIV8>;
    type CsrAxiB = axi::AxiB<AXI_ID_W>;

    type CsrRdReq = csr_config::CsrRdReq<LOG2_REGS_N>;
    type CsrRdResp = csr_config::CsrRdResp<LOG2_REGS_N, AXI_DATA_W>;
    type CsrWrReq = csr_config::CsrWrReq<LOG2_REGS_N, AXI_DATA_W>;
    type CsrWrResp = csr_config::CsrWrResp;
    type CsrChange = csr_config::CsrChange<LOG2_REGS_N>;

    type MemAxiAr = axi::AxiAr<AXI_ADDR_W, AXI_ID_W>;
    type MemAxiR = axi::AxiR<AXI_DATA_W, AXI_ID_W>;
    type MemAxiAw = axi::AxiAw<AXI_ADDR_W, AXI_ID_W>;
    type MemAxiW = axi::AxiW<AXI_DATA_W, AXI_DATA_W_DIV8>;
    type MemAxiB = axi::AxiB<AXI_ID_W>;

    type MemReaderReq  = mem_reader::MemReaderReq<AXI_ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<AXI_DATA_W, AXI_ADDR_W>;

    type FrameHeaderDecoderReq = frame_header_dec::FrameHeaderDecoderReq<AXI_ADDR_W>;
    type FrameHeaderDecoderResp = frame_header_dec::FrameHeaderDecoderResp;

    type BlockHeaderDecoderReq = block_header_dec::BlockHeaderDecoderReq<AXI_ADDR_W>;
    type BlockHeaderDecoderResp = block_header_dec::BlockHeaderDecoderResp;

    type RawBlockDecoderReq = raw_block_dec::RawBlockDecoderReq<AXI_ADDR_W>;
    type RawBlockDecoderResp = raw_block_dec::RawBlockDecoderResp;
    type ExtendedBlockDataPacket = common::ExtendedBlockDataPacket;

    type RleBlockDecoderReq = rle_block_dec::RleBlockDecoderReq<AXI_ADDR_W>;
    type RleBlockDecoderResp = rle_block_dec::RleBlockDecoderResp;

    type SequenceExecutorPacket = common::SequenceExecutorPacket;
    type ZstdDecodedPacket = common::ZstdDecodedPacket;

    type RamRdReq = ram::ReadReq<HB_ADDR_W, HB_NUM_PARTITIONS>;
    type RamRdResp = ram::ReadResp<HB_DATA_W>;
    type RamWrReq = ram::WriteReq<HB_ADDR_W, HB_DATA_W, HB_NUM_PARTITIONS>;
    type RamWrResp = ram::WriteResp;

    // Complex Block Decoder
    cmp_output_s: chan<ExtendedBlockDataPacket> out;

    init {}

    config(
        // AXI Ctrl (subordinate)
        csr_axi_aw_r: chan<CsrAxiAw> in,
        csr_axi_w_r: chan<CsrAxiW> in,
        csr_axi_b_s: chan<CsrAxiB> out,
        csr_axi_ar_r: chan<CsrAxiAr> in,
        csr_axi_r_s: chan<CsrAxiR> out,

        // AXI Frame Header Decoder (manager)
        fh_axi_ar_s: chan<MemAxiAr> out,
        fh_axi_r_r: chan<MemAxiR> in,

        //// AXI Block Header Decoder (manager)
        bh_axi_ar_s: chan<MemAxiAr> out,
        bh_axi_r_r: chan<MemAxiR> in,

        //// AXI RAW Block Decoder (manager)
        raw_axi_ar_s: chan<MemAxiAr> out,
        raw_axi_r_r: chan<MemAxiR> in,

        // History Buffer
        ram_rd_req_0_s: chan<RamRdReq> out,
        ram_rd_req_1_s: chan<RamRdReq> out,
        ram_rd_req_2_s: chan<RamRdReq> out,
        ram_rd_req_3_s: chan<RamRdReq> out,
        ram_rd_req_4_s: chan<RamRdReq> out,
        ram_rd_req_5_s: chan<RamRdReq> out,
        ram_rd_req_6_s: chan<RamRdReq> out,
        ram_rd_req_7_s: chan<RamRdReq> out,
        ram_rd_resp_0_r: chan<RamRdResp> in,
        ram_rd_resp_1_r: chan<RamRdResp> in,
        ram_rd_resp_2_r: chan<RamRdResp> in,
        ram_rd_resp_3_r: chan<RamRdResp> in,
        ram_rd_resp_4_r: chan<RamRdResp> in,
        ram_rd_resp_5_r: chan<RamRdResp> in,
        ram_rd_resp_6_r: chan<RamRdResp> in,
        ram_rd_resp_7_r: chan<RamRdResp> in,
        ram_wr_req_0_s: chan<RamWrReq> out,
        ram_wr_req_1_s: chan<RamWrReq> out,
        ram_wr_req_2_s: chan<RamWrReq> out,
        ram_wr_req_3_s: chan<RamWrReq> out,
        ram_wr_req_4_s: chan<RamWrReq> out,
        ram_wr_req_5_s: chan<RamWrReq> out,
        ram_wr_req_6_s: chan<RamWrReq> out,
        ram_wr_req_7_s: chan<RamWrReq> out,
        ram_wr_resp_0_r: chan<RamWrResp> in,
        ram_wr_resp_1_r: chan<RamWrResp> in,
        ram_wr_resp_2_r: chan<RamWrResp> in,
        ram_wr_resp_3_r: chan<RamWrResp> in,
        ram_wr_resp_4_r: chan<RamWrResp> in,
        ram_wr_resp_5_r: chan<RamWrResp> in,
        ram_wr_resp_6_r: chan<RamWrResp> in,
        ram_wr_resp_7_r: chan<RamWrResp> in,

        // Decoder output
        output_s: chan<ZstdDecodedPacket> out,
        notify_s: chan<()> out,
        reset_s: chan<()> out,
    ) {
        const CHANNEL_DEPTH = u32:1;

        // CSRs

        let (ext_csr_rd_req_s, ext_csr_rd_req_r) = chan<CsrRdReq, CHANNEL_DEPTH>("csr_rd_req");
        let (ext_csr_rd_resp_s, ext_csr_rd_resp_r) = chan<CsrRdResp, CHANNEL_DEPTH>("csr_rd_resp");
        let (ext_csr_wr_req_s, ext_csr_wr_req_r) = chan<CsrWrReq, CHANNEL_DEPTH>("csr_wr_req");
        let (ext_csr_wr_resp_s, ext_csr_wr_resp_r) = chan<CsrWrResp, CHANNEL_DEPTH>("csr_wr_resp");

        let (csr_rd_req_s, csr_rd_req_r) = chan<CsrRdReq, CHANNEL_DEPTH>("csr_rd_req");
        let (csr_rd_resp_s, csr_rd_resp_r) = chan<CsrRdResp, CHANNEL_DEPTH>("csr_rd_resp");
        let (csr_wr_req_s, csr_wr_req_r) = chan<CsrWrReq, CHANNEL_DEPTH>("csr_wr_req");
        let (csr_wr_resp_s, csr_wr_resp_r) = chan<CsrWrResp, CHANNEL_DEPTH>("csr_wr_resp");

        let (csr_change_s, csr_change_r) = chan<CsrChange, CHANNEL_DEPTH>("csr_change");

        spawn axi_csr_accessor::AxiCsrAccessor<AXI_ID_W, AXI_ADDR_W, AXI_DATA_W, REGS_N>(
            csr_axi_aw_r, csr_axi_w_r, csr_axi_b_s, // csr write from AXI
            csr_axi_ar_r, csr_axi_r_s,              // csr read from AXI
            ext_csr_rd_req_s, ext_csr_rd_resp_r,    // csr read to CsrConfig
            ext_csr_wr_req_s, ext_csr_wr_resp_r,    // csr write to CsrConfig
        );

        spawn csr_config::CsrConfig<AXI_ID_W, AXI_ADDR_W, AXI_DATA_W, REGS_N>(
            ext_csr_rd_req_r, ext_csr_rd_resp_s,    // csr read from AxiCsrAccessor
            ext_csr_wr_req_r, ext_csr_wr_resp_s,    // csr write from AxiCsrAccessor
            csr_rd_req_r, csr_rd_resp_s,            // csr read from design
            csr_wr_req_r, csr_wr_resp_s,            // csr write from design
            csr_change_s,                           // notification about csr change
        );

        // Frame Header

        let (fh_mem_rd_req_s,  fh_mem_rd_req_r) = chan<MemReaderReq, CHANNEL_DEPTH>("fh_mem_rd_req");
        let (fh_mem_rd_resp_s, fh_mem_rd_resp_r) = chan<MemReaderResp, CHANNEL_DEPTH>("fh_mem_rd_resp");

        spawn mem_reader::MemReader<AXI_DATA_W, AXI_ADDR_W, AXI_DEST_W, AXI_ID_W, CHANNEL_DEPTH>(
           fh_mem_rd_req_r, fh_mem_rd_resp_s,
           fh_axi_ar_s, fh_axi_r_r,
        );

        let (fh_req_s, fh_req_r) = chan<FrameHeaderDecoderReq, CHANNEL_DEPTH>("fh_req");
        let (fh_resp_s, fh_resp_r) = chan<FrameHeaderDecoderResp, CHANNEL_DEPTH>("fh_resp");

        spawn frame_header_dec::FrameHeaderDecoder<WINDOW_LOG_MAX, AXI_DATA_W, AXI_ADDR_W>(
            fh_mem_rd_req_s, fh_mem_rd_resp_r,
            fh_req_r, fh_resp_s,
        );

        // Block Header

        let (bh_mem_rd_req_s, bh_mem_rd_req_r) = chan<MemReaderReq, CHANNEL_DEPTH>("bh_mem_rd_req");
        let (bh_mem_rd_resp_s, bh_mem_rd_resp_r) = chan<MemReaderResp, CHANNEL_DEPTH>("bh_mem_rd_resp");

        spawn mem_reader::MemReader<AXI_DATA_W, AXI_ADDR_W, AXI_DEST_W, AXI_ID_W, CHANNEL_DEPTH>(
            bh_mem_rd_req_r, bh_mem_rd_resp_s,
            bh_axi_ar_s, bh_axi_r_r,
        );

        let (bh_req_s, bh_req_r) = chan<BlockHeaderDecoderReq, CHANNEL_DEPTH>("bh_req");
        let (bh_resp_s, bh_resp_r) = chan<BlockHeaderDecoderResp, CHANNEL_DEPTH>("bh_resp");

        spawn block_header_dec::BlockHeaderDecoder<AXI_DATA_W, AXI_ADDR_W>(
            bh_req_r, bh_resp_s,
            bh_mem_rd_req_s, bh_mem_rd_resp_r,
        );

        // Raw Block Decoder

        let (raw_mem_rd_req_s, raw_mem_rd_req_r) = chan<MemReaderReq, CHANNEL_DEPTH>("raw_mem_rd_req");
        let (raw_mem_rd_resp_s, raw_mem_rd_resp_r) = chan<MemReaderResp, CHANNEL_DEPTH>("raw_mem_rd_resp");

        spawn mem_reader::MemReader<AXI_DATA_W, AXI_ADDR_W, AXI_DEST_W, AXI_ID_W, CHANNEL_DEPTH>(
           raw_mem_rd_req_r, raw_mem_rd_resp_s,
           raw_axi_ar_s, raw_axi_r_r,
        );

        let (raw_req_s,  raw_req_r) = chan<RawBlockDecoderReq, CHANNEL_DEPTH>("raw_req");
        let (raw_resp_s, raw_resp_r) = chan<RawBlockDecoderResp, CHANNEL_DEPTH>("raw_resp");
        let (raw_output_s, raw_output_r) = chan<ExtendedBlockDataPacket, CHANNEL_DEPTH>("raw_output");

        spawn raw_block_dec::RawBlockDecoder<AXI_DATA_W, AXI_ADDR_W>(
            raw_req_r, raw_resp_s, raw_output_s,
            raw_mem_rd_req_s, raw_mem_rd_resp_r,
        );

        // RLE Block Decoder

        let (rle_req_s,  rle_req_r) = chan<RleBlockDecoderReq, CHANNEL_DEPTH>("rle_req");
        let (rle_resp_s, rle_resp_r) = chan<RleBlockDecoderResp, CHANNEL_DEPTH>("rle_resp");
        let (rle_output_s, rle_output_r) = chan<ExtendedBlockDataPacket, CHANNEL_DEPTH>("rle_output");

        spawn rle_block_dec::RleBlockDecoder<AXI_DATA_W>(
            rle_req_r, rle_resp_s, rle_output_s
        );

        // Collecting Packets

        let (cmp_output_s, cmp_output_r) = chan<ExtendedBlockDataPacket, CHANNEL_DEPTH>("cmp_output");
        let (seq_exec_input_s, seq_exec_input_r) = chan<SequenceExecutorPacket, CHANNEL_DEPTH>("demux_output");

        spawn dec_mux::DecoderMux(
            raw_output_r, rle_output_r, cmp_output_r,
            seq_exec_input_s,
        );

        // Sequence Execution

        let (seq_exec_looped_s, seq_exec_looped_r) = chan<SequenceExecutorPacket, CHANNEL_DEPTH>("seq_exec_looped");
        let (seq_exec_output_s, seq_exec_output_r) = chan<ZstdDecodedPacket, CHANNEL_DEPTH>("seq_exec_output");

        spawn sequence_executor::SequenceExecutor<HB_SIZE_KB>(
            seq_exec_input_r, seq_exec_output_s,
            seq_exec_looped_r, seq_exec_looped_s,
            ram_rd_req_0_s, ram_rd_req_1_s, ram_rd_req_2_s, ram_rd_req_3_s,
            ram_rd_req_4_s, ram_rd_req_5_s, ram_rd_req_6_s, ram_rd_req_7_s,
            ram_rd_resp_0_r, ram_rd_resp_1_r, ram_rd_resp_2_r, ram_rd_resp_3_r,
            ram_rd_resp_4_r, ram_rd_resp_5_r, ram_rd_resp_6_r, ram_rd_resp_7_r,
            ram_wr_req_0_s, ram_wr_req_1_s, ram_wr_req_2_s, ram_wr_req_3_s,
            ram_wr_req_4_s, ram_wr_req_5_s, ram_wr_req_6_s, ram_wr_req_7_s,
            ram_wr_resp_0_r, ram_wr_resp_1_r, ram_wr_resp_2_r, ram_wr_resp_3_r,
            ram_wr_resp_4_r, ram_wr_resp_5_r, ram_wr_resp_6_r, ram_wr_resp_7_r
        );

        // Repacketizer

        spawn repacketizer::Repacketizer(seq_exec_output_r, output_s);

        // Zstd Decoder Control

        spawn ZstdDecoderInternal<AXI_DATA_W, AXI_ADDR_W, REGS_N> (
            csr_rd_req_s, csr_rd_resp_r, csr_wr_req_s, csr_wr_resp_r, csr_change_r,
            fh_req_s, fh_resp_r,
            bh_req_s, bh_resp_r,
            raw_req_s, raw_resp_r,
            rle_req_s, rle_resp_r,
            notify_s, reset_s,
        );

        (cmp_output_s,)
    }

    next (state: ()) {
        send_if(join(), cmp_output_s, false, zero!<ExtendedBlockDataPacket>());
    }
}

const INST_AXI_DATA_W = u32:64;
const INST_AXI_ADDR_W = u32:16;
const INST_AXI_ID_W = u32:4;
const INST_AXI_DEST_W = u32:4;
const INST_REGS_N = u32:16;
const INST_WINDOW_LOG_MAX = u32:30;
const INST_HB_ADDR_W = sequence_executor::ZSTD_RAM_ADDR_WIDTH;
const INST_HB_DATA_W = sequence_executor::RAM_DATA_WIDTH;
const INST_HB_NUM_PARTITIONS = sequence_executor::RAM_NUM_PARTITIONS;
const INST_HB_SIZE_KB = sequence_executor::ZSTD_HISTORY_BUFFER_SIZE_KB;

const INST_LOG2_REGS_N = std::clog2(INST_REGS_N);
const INST_AXI_DATA_W_DIV8 = INST_AXI_DATA_W / u32:8;
const INST_HB_RAM_N = u32:8;

proc ZstdDecoderInternalInst {
    type State = ZstdDecoderInternalState;
    type Fsm = ZstdDecoderInternalFsm;

    type CsrRdReq = csr_config::CsrRdReq<INST_LOG2_REGS_N>;
    type CsrRdResp = csr_config::CsrRdResp<INST_LOG2_REGS_N, INST_AXI_DATA_W>;
    type CsrWrReq = csr_config::CsrWrReq<INST_LOG2_REGS_N, INST_AXI_DATA_W>;
    type CsrWrResp = csr_config::CsrWrResp;
    type CsrChange = csr_config::CsrChange<INST_LOG2_REGS_N>;

    type FrameHeaderDecoderReq = frame_header_dec::FrameHeaderDecoderReq<INST_AXI_ADDR_W>;
    type FrameHeaderDecoderResp = frame_header_dec::FrameHeaderDecoderResp;

    type BlockHeaderDecoderReq = block_header_dec::BlockHeaderDecoderReq<INST_AXI_ADDR_W>;
    type BlockHeaderDecoderResp = block_header_dec::BlockHeaderDecoderResp;

    type RawBlockDecoderReq = raw_block_dec::RawBlockDecoderReq<INST_AXI_ADDR_W>;
    type RawBlockDecoderResp = raw_block_dec::RawBlockDecoderResp;

    type RleBlockDecoderReq = rle_block_dec::RleBlockDecoderReq<INST_AXI_ADDR_W>;
    type RleBlockDecoderResp = rle_block_dec::RleBlockDecoderResp;

    init { }

    config(
        csr_rd_req_s: chan<CsrRdReq> out,
        csr_rd_resp_r: chan<CsrRdResp> in,
        csr_wr_req_s: chan<CsrWrReq> out,
        csr_wr_resp_r: chan<CsrWrResp> in,
        csr_change_r: chan<CsrChange> in,

        // MemReader + FameHeaderDecoder
        fh_req_s: chan<FrameHeaderDecoderReq> out,
        fh_resp_r: chan<FrameHeaderDecoderResp> in,

        // MemReader + BlockHeaderDecoder
        bh_req_s: chan<BlockHeaderDecoderReq> out,
        bh_resp_r: chan<BlockHeaderDecoderResp> in,

        // MemReader + RawBlockDecoder
        raw_req_s: chan<RawBlockDecoderReq> out,
        raw_resp_r: chan<RawBlockDecoderResp> in,

        // MemReader + RleBlockDecoder
        rle_req_s: chan<RleBlockDecoderReq> out,
        rle_resp_r: chan<RleBlockDecoderResp> in,

        // IRQ
        notify_s: chan<()> out,
        reset_s: chan<()> out,
    ) {
        spawn ZstdDecoderInternal<
            INST_AXI_DATA_W, INST_AXI_ADDR_W, INST_REGS_N,
        > (
            csr_rd_req_s, csr_rd_resp_r, csr_wr_req_s, csr_wr_resp_r, csr_change_r,
            fh_req_s, fh_resp_r,
            bh_req_s, bh_resp_r,
            raw_req_s, raw_resp_r,
            rle_req_s, rle_resp_r,
            notify_s, reset_s,
        );

    }

    next(state: ()) {}
}

proc ZstdDecoderInst {
    type CsrAxiAr = axi::AxiAr<INST_AXI_ADDR_W, INST_AXI_ID_W>;
    type CsrAxiR = axi::AxiR<INST_AXI_DATA_W, INST_AXI_ID_W>;
    type CsrAxiAw = axi::AxiAw<INST_AXI_ADDR_W, INST_AXI_ID_W>;
    type CsrAxiW = axi::AxiW<INST_AXI_DATA_W, INST_AXI_DATA_W_DIV8>;
    type CsrAxiB = axi::AxiB<INST_AXI_ID_W>;

    type MemAxiAr = axi::AxiAr<INST_AXI_ADDR_W, INST_AXI_ID_W>;
    type MemAxiR = axi::AxiR<INST_AXI_DATA_W, INST_AXI_ID_W>;
    type MemAxiAw = axi::AxiAw<INST_AXI_ADDR_W, INST_AXI_ID_W>;
    type MemAxiW = axi::AxiW<INST_AXI_DATA_W, INST_AXI_DATA_W_DIV8>;
    type MemAxiB = axi::AxiB<INST_AXI_ID_W>;

    type RamRdReq = ram::ReadReq<INST_HB_ADDR_W, INST_HB_NUM_PARTITIONS>;
    type RamRdResp = ram::ReadResp<INST_HB_DATA_W>;
    type RamWrReq = ram::WriteReq<INST_HB_ADDR_W, INST_HB_DATA_W, INST_HB_NUM_PARTITIONS>;
    type RamWrResp = ram::WriteResp;

    type ZstdDecodedPacket = common::ZstdDecodedPacket;

    init { }

    config(
        // AXI Ctrl (subordinate)
        csr_axi_aw_r: chan<CsrAxiAw> in,
        csr_axi_w_r: chan<CsrAxiW> in,
        csr_axi_b_s: chan<CsrAxiB> out,
        csr_axi_ar_r: chan<CsrAxiAr> in,
        csr_axi_r_s: chan<CsrAxiR> out,

        // AXI Frame Header Decoder (manager)
        fh_axi_ar_s: chan<MemAxiAr> out,
        fh_axi_r_r: chan<MemAxiR> in,

        // AXI Block Header Decoder (manager)
        bh_axi_ar_s: chan<MemAxiAr> out,
        bh_axi_r_r: chan<MemAxiR> in,

        // AXI RAW Block Decoder (manager)
        raw_axi_ar_s: chan<MemAxiAr> out,
        raw_axi_r_r: chan<MemAxiR> in,

        // History Buffer
        ram_rd_req_0_s: chan<RamRdReq> out,
        ram_rd_req_1_s: chan<RamRdReq> out,
        ram_rd_req_2_s: chan<RamRdReq> out,
        ram_rd_req_3_s: chan<RamRdReq> out,
        ram_rd_req_4_s: chan<RamRdReq> out,
        ram_rd_req_5_s: chan<RamRdReq> out,
        ram_rd_req_6_s: chan<RamRdReq> out,
        ram_rd_req_7_s: chan<RamRdReq> out,
        ram_rd_resp_0_r: chan<RamRdResp> in,
        ram_rd_resp_1_r: chan<RamRdResp> in,
        ram_rd_resp_2_r: chan<RamRdResp> in,
        ram_rd_resp_3_r: chan<RamRdResp> in,
        ram_rd_resp_4_r: chan<RamRdResp> in,
        ram_rd_resp_5_r: chan<RamRdResp> in,
        ram_rd_resp_6_r: chan<RamRdResp> in,
        ram_rd_resp_7_r: chan<RamRdResp> in,
        ram_wr_req_0_s: chan<RamWrReq> out,
        ram_wr_req_1_s: chan<RamWrReq> out,
        ram_wr_req_2_s: chan<RamWrReq> out,
        ram_wr_req_3_s: chan<RamWrReq> out,
        ram_wr_req_4_s: chan<RamWrReq> out,
        ram_wr_req_5_s: chan<RamWrReq> out,
        ram_wr_req_6_s: chan<RamWrReq> out,
        ram_wr_req_7_s: chan<RamWrReq> out,
        ram_wr_resp_0_r: chan<RamWrResp> in,
        ram_wr_resp_1_r: chan<RamWrResp> in,
        ram_wr_resp_2_r: chan<RamWrResp> in,
        ram_wr_resp_3_r: chan<RamWrResp> in,
        ram_wr_resp_4_r: chan<RamWrResp> in,
        ram_wr_resp_5_r: chan<RamWrResp> in,
        ram_wr_resp_6_r: chan<RamWrResp> in,
        ram_wr_resp_7_r: chan<RamWrResp> in,

        // Decoder output
        output_s: chan<ZstdDecodedPacket> out,
        notify_s: chan<()> out,
        reset_s: chan<()> out,
    ) {
        spawn ZstdDecoder<
            INST_AXI_DATA_W, INST_AXI_ADDR_W, INST_AXI_ID_W, INST_AXI_DEST_W,
            INST_REGS_N, INST_WINDOW_LOG_MAX,
            INST_HB_ADDR_W, INST_HB_DATA_W, INST_HB_NUM_PARTITIONS, INST_HB_SIZE_KB,
        >(
            csr_axi_aw_r, csr_axi_w_r, csr_axi_b_s, csr_axi_ar_r, csr_axi_r_s,
            fh_axi_ar_s, fh_axi_r_r,
            bh_axi_ar_s, bh_axi_r_r,
            raw_axi_ar_s, raw_axi_r_r,
            ram_rd_req_0_s, ram_rd_req_1_s, ram_rd_req_2_s, ram_rd_req_3_s,
            ram_rd_req_4_s, ram_rd_req_5_s, ram_rd_req_6_s, ram_rd_req_7_s,
            ram_rd_resp_0_r, ram_rd_resp_1_r, ram_rd_resp_2_r, ram_rd_resp_3_r,
            ram_rd_resp_4_r, ram_rd_resp_5_r, ram_rd_resp_6_r, ram_rd_resp_7_r,
            ram_wr_req_0_s, ram_wr_req_1_s, ram_wr_req_2_s, ram_wr_req_3_s,
            ram_wr_req_4_s, ram_wr_req_5_s, ram_wr_req_6_s, ram_wr_req_7_s,
            ram_wr_resp_0_r, ram_wr_resp_1_r, ram_wr_resp_2_r, ram_wr_resp_3_r,
            ram_wr_resp_4_r, ram_wr_resp_5_r, ram_wr_resp_6_r, ram_wr_resp_7_r,
            output_s, notify_s, reset_s,
        );
    }

    next (state: ()) {}
}
