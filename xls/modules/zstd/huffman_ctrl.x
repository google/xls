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

// This file contains Huffman decoder control and sequence proc implementation.

import xls.modules.zstd.common as common;
import xls.modules.zstd.memory.mem_reader as mem_reader;
import xls.modules.zstd.huffman_common as hcommon;
import xls.modules.zstd.huffman_axi_reader as axi_reader;
import xls.modules.zstd.huffman_code_builder as code_builder;
import xls.modules.zstd.huffman_data_preprocessor as data_preprocessor;
import xls.modules.zstd.huffman_decoder as decoder;
import xls.modules.zstd.huffman_prescan as prescan;
import xls.modules.zstd.huffman_weights_dec as weights_dec;


enum HuffmanControlAndSequenceFSM: u2 {
    IDLE = 0,
    DECODING = 1,
}

pub struct HuffmanControlAndSequenceCtrl<AXI_ADDR_W: u32> {
    base_addr: uN[AXI_ADDR_W],
    len: uN[AXI_ADDR_W],
    new_config: bool,
    multi_stream: bool,
    id: u32,
    literals_last: bool,
}

pub enum HuffmanControlAndSequenceStatus: u1 {
    OKAY = 0,
    ERROR = 1,
}

pub struct HuffmanControlAndSequenceResp {
    status: HuffmanControlAndSequenceStatus
}

struct HuffmanControlAndSequenceState<AXI_ADDR_W: u32> {
    fsm: HuffmanControlAndSequenceFSM,
    weights_dec_pending: bool,
    stream_dec_pending: bool,
    multi_stream_dec_pending: bool,
    multi_stream_decodings_finished: u3,
    jump_table_dec_pending: bool,
    jump_table_req_sent: bool,
    tree_description_size: uN[AXI_ADDR_W],
    ctrl: HuffmanControlAndSequenceCtrl<AXI_ADDR_W>,
    stream_sizes: uN[AXI_ADDR_W][4],
    prescan_start_sent: bool,
}

const JUMP_TABLE_SIZE = u32:6;

pub proc HuffmanControlAndSequence<AXI_ADDR_W: u32, AXI_DATA_W: u32> {
    type AxiReaderCtrl = axi_reader::HuffmanAxiReaderCtrl<AXI_ADDR_W>;
    type DataPreprocessorStart = data_preprocessor::HuffmanDataPreprocessorStart;
    type DecoderStart = decoder::HuffmanDecoderStart;
    type WeightsDecReq = weights_dec::HuffmanWeightsDecoderReq<AXI_ADDR_W>;
    type WeightsDecResp = weights_dec::HuffmanWeightsDecoderResp<AXI_ADDR_W>;
    type MemReaderReq  = mem_reader::MemReaderReq<AXI_ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<AXI_DATA_W, AXI_ADDR_W>;


    type State = HuffmanControlAndSequenceState<AXI_ADDR_W>;
    type FSM = HuffmanControlAndSequenceFSM;
    type Ctrl = HuffmanControlAndSequenceCtrl<AXI_ADDR_W>;
    type Resp = HuffmanControlAndSequenceResp;
    type Status = HuffmanControlAndSequenceStatus;

    ctrl_r: chan<Ctrl> in;
    resp_s: chan<Resp> out;

    // Huffman tree description decoder
    weights_dec_req_s: chan<WeightsDecReq> out;
    weights_dec_resp_r: chan<WeightsDecResp> in;

    // prescan
    prescan_start_s: chan<bool> out;

    // code builder
    code_builder_start_s: chan<bool> out;

    // AXI reader
    axi_reader_ctrl_s: chan<AxiReaderCtrl> out;

    // data preprocess
    data_preprocess_start_s: chan<DataPreprocessorStart> out;

    // decoder
    decoder_start_s: chan<DecoderStart> out;
    decoder_done_r: chan<()> in;

    // MemReader interface for fetching the Jump Table
    mem_rd_req_s: chan<MemReaderReq> out;
    mem_rd_resp_r: chan<MemReaderResp> in;


    config (
        ctrl_r: chan<Ctrl> in,
        resp_s: chan<Resp> out,
        weights_dec_req_s: chan<WeightsDecReq> out,
        weights_dec_resp_r: chan<WeightsDecResp> in,
        prescan_start_s: chan<bool> out,
        code_builder_start_s: chan<bool> out,
        axi_reader_ctrl_s: chan<AxiReaderCtrl> out,
        data_preprocess_start_s: chan<DataPreprocessorStart> out,
        decoder_start_s: chan<DecoderStart> out,
        decoder_done_r: chan<()> in,
        mem_rd_req_s: chan<MemReaderReq> out,
        mem_rd_resp_r: chan<MemReaderResp> in,
    ) {
        (
            ctrl_r, resp_s,
            weights_dec_req_s,
            weights_dec_resp_r,
            prescan_start_s,
            code_builder_start_s,
            axi_reader_ctrl_s,
            data_preprocess_start_s,
            decoder_start_s,
            decoder_done_r,
            mem_rd_req_s, mem_rd_resp_r,
        )
    }

    init {
        zero!<State>()
    }

    next (state: State) {
        // receive start
        let (tok, ctrl, ctrl_valid) = recv_if_non_blocking(join(), ctrl_r, state.fsm == FSM::IDLE, zero!<Ctrl>());
        if (ctrl_valid) { trace_fmt!("Received Ctrl: {:#x}", ctrl); } else {};

        let state = if ctrl_valid {
            State {
                fsm: FSM::DECODING,
                ctrl: ctrl,
                weights_dec_pending: ctrl.new_config,
                multi_stream_dec_pending: ctrl.multi_stream,
                jump_table_dec_pending: ctrl.multi_stream,
                ..state
            }
        } else {
            state
        };

        // send start to prescan and code builder
        let new_config = ctrl_valid & ctrl.new_config;

        // New config means the requirement to read and decode new Huffman Tree Description
        // Delegate this task to HuffmanWeightsDecoder
        let weights_dec_req = WeightsDecReq {
            addr: ctrl.base_addr
        };
        send_if(tok, weights_dec_req_s, new_config, weights_dec_req);
        if (new_config) { trace_fmt!("Sent Weights Decoding Request: {:#x}", weights_dec_req); } else {};

        // recv response
        let (tok, weights_dec_resp, weights_dec_resp_valid) = recv_if_non_blocking(tok, weights_dec_resp_r, state.weights_dec_pending, zero!<WeightsDecResp>());
        if (weights_dec_resp_valid) { trace_fmt!("Received Weights Decoding response: {:#x}", weights_dec_resp); } else {};
        let state = if weights_dec_resp_valid {
            trace_fmt!("Tree description size: {:#x}", weights_dec_resp.tree_description_size);
            State {
                weights_dec_pending: false,
                tree_description_size: weights_dec_resp.tree_description_size,
                ..state
            }
        } else {
            state
        };

        // Fetch the Jump Table if neccessary
        let jump_table_req = MemReaderReq {
            addr: state.ctrl.base_addr + state.tree_description_size,
            length: JUMP_TABLE_SIZE as uN[AXI_ADDR_W],
        };
        let do_send_jump_table_req = !state.weights_dec_pending && state.jump_table_dec_pending && !state.jump_table_req_sent;
        let tok = send_if(tok, mem_rd_req_s, do_send_jump_table_req, jump_table_req);
        if do_send_jump_table_req {
            trace_fmt!("Sent Jump Table read request {:#x}", jump_table_req);
        } else {};
        let (tok, jump_table_raw, jump_table_valid) = recv_if_non_blocking(tok, mem_rd_resp_r, state.jump_table_dec_pending, zero!<MemReaderResp>());
        let stream_sizes = jump_table_raw.data[0:48] as u16[3];
        let total_streams_size = state.ctrl.len - state.tree_description_size;
        let stream_sizes = uN[AXI_ADDR_W][4]:[
            stream_sizes[0] as uN[AXI_ADDR_W],
            stream_sizes[1] as uN[AXI_ADDR_W],
            stream_sizes[2] as uN[AXI_ADDR_W],
            total_streams_size - JUMP_TABLE_SIZE as uN[AXI_ADDR_W] - (stream_sizes[0] + stream_sizes[1] + stream_sizes[2]) as uN[AXI_ADDR_W]
        ];
        if jump_table_valid {
            trace_fmt!("Received Jump Table: {:#x}", jump_table_raw);
            trace_fmt!("Total streams size: {:#x}", total_streams_size);
            trace_fmt!("Stream sizes: {:#x}", stream_sizes);
        } else {};
        let state = if do_send_jump_table_req {
            State {
                jump_table_req_sent: true,
                ..state
            }
        } else if jump_table_valid {
            State {
                jump_table_dec_pending: false,
                jump_table_req_sent: false,
                stream_sizes: stream_sizes,
                ..state
            }
        } else {
            state
        };

        let start_decoding = (
            (state.fsm == FSM::DECODING) &
            (!state.weights_dec_pending) &
            (!state.stream_dec_pending) &
            (!state.jump_table_dec_pending) &
            (
                (!state.multi_stream_dec_pending) ||
                (state.multi_stream_dec_pending && state.multi_stream_decodings_finished != u3:4)
            )
        );
        let send_prescan_start = start_decoding & (!state.prescan_start_sent);
        send_if(tok, prescan_start_s, send_prescan_start, true);
        send_if(tok, code_builder_start_s, send_prescan_start, true);
        if (send_prescan_start) { trace_fmt!("Sent START to prescan and code builder"); } else {};

        let state = if send_prescan_start {
            State {
                prescan_start_sent: send_prescan_start,
                ..state
            }
        } else {
            state
        };

        let stream_sizes = state.stream_sizes;
        let (huffman_stream_addr, huffman_stream_len) = match(state.ctrl.new_config, state.ctrl.multi_stream, state.multi_stream_decodings_finished) {
            (false, false, _) => (state.ctrl.base_addr, state.ctrl.len),
            (true, false, _) => ((state.ctrl.base_addr + state.tree_description_size), (state.ctrl.len - state.tree_description_size)),

            (false, true, u3:0) => ((state.ctrl.base_addr + JUMP_TABLE_SIZE as uN[AXI_ADDR_W]), (stream_sizes[0] as uN[AXI_ADDR_W])),
            (false, true, u3:1) => ((state.ctrl.base_addr + JUMP_TABLE_SIZE as uN[AXI_ADDR_W] + stream_sizes[0]), (stream_sizes[1] as uN[AXI_ADDR_W])),
            (false, true, u3:2) => ((state.ctrl.base_addr + JUMP_TABLE_SIZE as uN[AXI_ADDR_W] + stream_sizes[0] + stream_sizes[1]), (stream_sizes[2] as uN[AXI_ADDR_W])),
            (false, true, u3:3) => ((state.ctrl.base_addr + JUMP_TABLE_SIZE as uN[AXI_ADDR_W] + stream_sizes[0] + stream_sizes[1] + stream_sizes[2]), (stream_sizes[3] as uN[AXI_ADDR_W])),

            (true, true, u3:0) => ((state.ctrl.base_addr + state.tree_description_size + JUMP_TABLE_SIZE as uN[AXI_ADDR_W]), (stream_sizes[0] as uN[AXI_ADDR_W])),
            (true, true, u3:1) => ((state.ctrl.base_addr + state.tree_description_size + JUMP_TABLE_SIZE as uN[AXI_ADDR_W] + stream_sizes[0]), (stream_sizes[1] as uN[AXI_ADDR_W])),
            (true, true, u3:2) => ((state.ctrl.base_addr + state.tree_description_size + JUMP_TABLE_SIZE as uN[AXI_ADDR_W] + stream_sizes[0] + stream_sizes[1]), (stream_sizes[2] as uN[AXI_ADDR_W])),
            (true, true, u3:3) => ((state.ctrl.base_addr + state.tree_description_size + JUMP_TABLE_SIZE as uN[AXI_ADDR_W] + stream_sizes[0] + stream_sizes[1] + stream_sizes[2]), (stream_sizes[3] as uN[AXI_ADDR_W])),

            (_, _, _) => (state.ctrl.base_addr, state.ctrl.len)
        };

        // send address and length to AXI reader
        let axi_reader_ctrl = AxiReaderCtrl {
            base_addr: huffman_stream_addr,
            len: huffman_stream_len,
        };
        send_if(tok, axi_reader_ctrl_s, start_decoding, axi_reader_ctrl);
        if (start_decoding) { trace_fmt!("Sent request to AXI reader: {:#x}", axi_reader_ctrl); } else {};

        // send reconfigure/keep to data preprocessor and decoder
        let config = if (state.multi_stream_decodings_finished > u3:0) {
            false
        } else {
            state.ctrl.new_config
        };
        let preprocessor_start = DataPreprocessorStart {
            new_config: config,
        };
        send_if(tok, data_preprocess_start_s, start_decoding, preprocessor_start);
        if start_decoding { trace_fmt!("Sent preprocessor start: {:#x}", preprocessor_start); } else {};
        let decoder_start = DecoderStart {
            new_config: config,
            id: state.ctrl.id,  // sending only if ctrl is valid
            literals_last: state.ctrl.literals_last,
            last_stream: !state.ctrl.multi_stream || (state.ctrl.multi_stream && state.multi_stream_decodings_finished == u3:3),
        };
        send_if(tok, decoder_start_s, start_decoding, decoder_start);
        if start_decoding { trace_fmt!("Sent decoder start: {:#x}", decoder_start); } else {};
        let state = if start_decoding {
            State {
                stream_dec_pending: true,
                ..state
            }
        } else {
            state
        };

        // receive done
        let (_, _, decoder_done_valid) = recv_if_non_blocking(tok, decoder_done_r, state.fsm == FSM::DECODING, ());
        if (decoder_done_valid) { trace_fmt!("Received Decoder Done"); } else {};
        let multi_stream_decodings_finished = if state.multi_stream_dec_pending {
            state.multi_stream_decodings_finished + u3:1
        } else {
            state.multi_stream_decodings_finished
        };

        let state = if decoder_done_valid {
            State {
                stream_dec_pending: false,
                multi_stream_decodings_finished: multi_stream_decodings_finished,
                ..state
            }
        } else {
            state
        };

        let state = if (multi_stream_decodings_finished == u3:4) {
            trace_fmt!("Multi-Stream decoding done");
            State {
                multi_stream_dec_pending: false,
                multi_stream_decodings_finished: u3:0,
                ..state
            }
        } else {
            state
        };

        let resp = Resp { status: Status::OKAY };
        let do_send_resp = decoder_done_valid && !state.multi_stream_dec_pending && state.multi_stream_decodings_finished == u3:0;
        send_if(tok, resp_s, do_send_resp, resp);
        if (do_send_resp) { trace_fmt!("Sent Ctrl response: {:#x}", resp); } else {};

        if do_send_resp {
            zero!<State>()
        } else {
            state
        }
    }
}


const INST_AXI_ADDR_W = u32:32;
const INST_AXI_DATA_W = u32:64;

proc HuffmanControlAndSequenceInst {
    type AxiReaderCtrl = axi_reader::HuffmanAxiReaderCtrl<INST_AXI_ADDR_W>;
    type DataPreprocessorStart = data_preprocessor::HuffmanDataPreprocessorStart;
    type DecoderStart = decoder::HuffmanDecoderStart;
    type WeightsDecReq = weights_dec::HuffmanWeightsDecoderReq<INST_AXI_ADDR_W>;
    type WeightsDecResp = weights_dec::HuffmanWeightsDecoderResp<INST_AXI_ADDR_W>;
    type MemReaderReq = mem_reader::MemReaderReq<INST_AXI_ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<INST_AXI_DATA_W, INST_AXI_ADDR_W>;

    config (
        ctrl_r: chan<HuffmanControlAndSequenceCtrl<INST_AXI_ADDR_W>> in,
        resp_s: chan<HuffmanControlAndSequenceResp> out,
        weights_dec_req_s: chan<WeightsDecReq> out,
        weights_dec_resp_r: chan<WeightsDecResp> in,
        prescan_start_s: chan<bool> out,
        code_builder_start_s: chan<bool> out,
        axi_reader_ctrl_s: chan<AxiReaderCtrl> out,
        data_preprocess_start_s: chan<DataPreprocessorStart> out,
        decoder_start_s: chan<DecoderStart> out,
        decoder_done_r: chan<()> in,
        mem_rd_req_s: chan<MemReaderReq> out,
        mem_rd_resp_r: chan<MemReaderResp> in,
    ) {
        spawn HuffmanControlAndSequence<INST_AXI_ADDR_W, INST_AXI_DATA_W>(
            ctrl_r, resp_s,
            weights_dec_req_s,
            weights_dec_resp_r,
            prescan_start_s,
            code_builder_start_s,
            axi_reader_ctrl_s,
            data_preprocess_start_s,
            decoder_start_s,
            decoder_done_r,
            mem_rd_req_s,
            mem_rd_resp_r,
        );
    }

    init { }

    next (state: ()) { }
}


const TEST_AXI_ADDR_W = u32:32;
const TEST_AXI_DATA_W = u32:64;

#[test_proc]
proc HuffmanControlAndSequence_test {
    type Ctrl = HuffmanControlAndSequenceCtrl<TEST_AXI_ADDR_W>;
    type Resp = HuffmanControlAndSequenceResp;
    type Status = HuffmanControlAndSequenceStatus;
    type AxiReaderCtrl = axi_reader::HuffmanAxiReaderCtrl<TEST_AXI_ADDR_W>;
    type DataPreprocessorStart = data_preprocessor::HuffmanDataPreprocessorStart;
    type DecoderStart = decoder::HuffmanDecoderStart;
    type WeightsDecReq = weights_dec::HuffmanWeightsDecoderReq<TEST_AXI_ADDR_W>;
    type WeightsDecResp = weights_dec::HuffmanWeightsDecoderResp<TEST_AXI_ADDR_W>;
    type WeightsDecStatus = weights_dec::HuffmanWeightsDecoderStatus;
    type MemReaderReq = mem_reader::MemReaderReq<TEST_AXI_ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<TEST_AXI_DATA_W, TEST_AXI_ADDR_W>;

    terminator: chan<bool> out;

    ctrl_s: chan<HuffmanControlAndSequenceCtrl<INST_AXI_ADDR_W>> out;
    resp_r: chan<HuffmanControlAndSequenceResp> in;
    weights_dec_req_r: chan<WeightsDecReq> in;
    weights_dec_resp_s: chan<WeightsDecResp> out;
    prescan_start_r: chan<bool> in;
    code_builder_start_r: chan<bool> in;
    axi_reader_ctrl_r: chan<AxiReaderCtrl> in;
    data_preprocess_start_r: chan<DataPreprocessorStart> in;
    decoder_start_r: chan<DecoderStart> in;
    decoder_done_s: chan<()> out;
    mem_rd_req_r: chan<MemReaderReq> in;
    mem_rd_resp_s: chan<MemReaderResp> out;

    config (terminator: chan<bool> out) {
        let (ctrl_s, ctrl_r) = chan<Ctrl>("ctrl");
        let (resp_s, resp_r) = chan<Resp>("resp");
        let (weights_dec_req_s, weights_dec_req_r) = chan<WeightsDecReq>("weights_dec_req");
        let (weights_dec_resp_s, weights_dec_resp_r) = chan<WeightsDecResp>("weights_dec_resp");
        let (prescan_start_s, prescan_start_r) = chan<bool>("prescan_start");
        let (code_builder_start_s, code_builder_start_r) = chan<bool>("code_builder_start");
        let (axi_reader_ctrl_s, axi_reader_ctrl_r) = chan<AxiReaderCtrl>("axi_reader_ctrl");
        let (data_preprocess_start_s, data_preprocess_start_r) = chan<DataPreprocessorStart>("data_preprocess_start");
        let (decoder_start_s, decoder_start_r) = chan<DecoderStart>("decoder_start");
        let (decoder_done_s, decoder_done_r) = chan<()>("decoder_done");
        let (mem_rd_req_s, mem_rd_req_r) = chan<MemReaderReq>("mem_rd_req");
        let (mem_rd_resp_s, mem_rd_resp_r) = chan<MemReaderResp>("mem_rd_resp");

        spawn HuffmanControlAndSequence<TEST_AXI_ADDR_W, TEST_AXI_DATA_W>(
            ctrl_r, resp_s,
            weights_dec_req_s,
            weights_dec_resp_r,
            prescan_start_s,
            code_builder_start_s,
            axi_reader_ctrl_s,
            data_preprocess_start_s,
            decoder_start_s,
            decoder_done_r,
            mem_rd_req_s,
            mem_rd_resp_r
        );

        (
            terminator,
            ctrl_s, resp_r,
            weights_dec_req_r,
            weights_dec_resp_s,
            prescan_start_r,
            code_builder_start_r,
            axi_reader_ctrl_r,
            data_preprocess_start_r,
            decoder_start_r,
            decoder_done_s,
            mem_rd_req_r,
            mem_rd_resp_s
        )
    }

    init { }

    next (state: ()) {
        let tok = join();
        // Single Stream
        // Without new config
        trace_fmt!("[TEST] Case #1");
        let ctrl = Ctrl {
            base_addr: uN[TEST_AXI_ADDR_W]:0x1,
            len: uN[TEST_AXI_ADDR_W]:0x2,
            new_config: false,
            multi_stream: false,
            id: u32:10,
            literals_last: true,
        };
        let tok = send(tok, ctrl_s, ctrl);

        let (tok, prescan_start) = recv(tok, prescan_start_r);
        trace_fmt!("[TEST] Received prescan START");
        assert_eq(true, prescan_start);

        let (tok, code_builder_start) = recv(tok, code_builder_start_r);
        trace_fmt!("[TEST] Received code builder START");
        assert_eq(true, code_builder_start);

        let (tok, axi_reader_ctrl) = recv(tok, axi_reader_ctrl_r);
        trace_fmt!("[TEST] Received AXI reader CTRL");
        assert_eq(AxiReaderCtrl {base_addr: ctrl.base_addr, len: ctrl.len}, axi_reader_ctrl);

        let (tok, data_preprocess_start) = recv(tok, data_preprocess_start_r);
        trace_fmt!("[TEST] Received data preprocess START");
        assert_eq(DataPreprocessorStart {new_config: ctrl.new_config}, data_preprocess_start);

        let (tok, decoder_start) = recv(tok, decoder_start_r);
        trace_fmt!("[TEST] Received decoder START");
        assert_eq(DecoderStart {new_config: ctrl.new_config, id: ctrl.id, literals_last: ctrl.literals_last, last_stream: true }, decoder_start);

        let tok = send(tok, decoder_done_s, ());
        let (tok, resp) = recv(tok, resp_r);
        trace_fmt!("[TEST] Received resp");
        assert_eq(Resp {status: Status::OKAY}, resp);

        // Single Stream
        // With new config
        trace_fmt!("[TEST] Case #2");
        let ctrl = Ctrl {
            base_addr: uN[TEST_AXI_ADDR_W]:0x1,
            len: uN[TEST_AXI_ADDR_W]:0x50,
            new_config: true,
            multi_stream: false,
            id: u32:0,
            literals_last: false,
        };
        let tok = send(tok, ctrl_s, ctrl);

        let (tok, weights_dec_req) = recv(tok, weights_dec_req_r);
        trace_fmt!("[TEST] Received weights decode request");
        assert_eq(WeightsDecReq {addr: uN[TEST_AXI_ADDR_W]:0x1}, weights_dec_req);

        // Signal Weight decoding done
        let tree_description_size = uN[TEST_AXI_ADDR_W]:0x25;
        let tok = send(tok, weights_dec_resp_s, WeightsDecResp{
            status: WeightsDecStatus::OKAY,
            tree_description_size: tree_description_size
        });

        let (tok, prescan_start) = recv(tok, prescan_start_r);
        trace_fmt!("[TEST] Received prescan START");
        assert_eq(true, prescan_start);

        let (tok, code_builder_start) = recv(tok, code_builder_start_r);
        trace_fmt!("[TEST] Received code builder START");
        assert_eq(true, code_builder_start);

        let (tok, axi_reader_ctrl) = recv(tok, axi_reader_ctrl_r);
        trace_fmt!("[TEST] Received AXI reader CTRL");
        assert_eq(AxiReaderCtrl {base_addr: ctrl.base_addr + tree_description_size, len: ctrl.len - tree_description_size}, axi_reader_ctrl);

        let (tok, data_preprocess_start) = recv(tok, data_preprocess_start_r);
        trace_fmt!("[TEST] Received data preprocess START");
        assert_eq(DataPreprocessorStart {new_config: ctrl.new_config}, data_preprocess_start);

        let (tok, decoder_start) = recv(tok, decoder_start_r);
        trace_fmt!("[TEST] Received decoder START");
        assert_eq(DecoderStart {new_config: ctrl.new_config, id: ctrl.id, literals_last: ctrl.literals_last, last_stream: true }, decoder_start);

        let tok = send(tok, decoder_done_s, ());
        let (tok, resp) = recv(tok, resp_r);
        trace_fmt!("[TEST] Received resp");
        assert_eq(Resp {status: Status::OKAY}, resp);

        // 4 Streams
        // Without new config
        trace_fmt!("[TEST] Case #3");
        let ctrl = Ctrl {
            base_addr: uN[TEST_AXI_ADDR_W]:0x1,
            len: uN[TEST_AXI_ADDR_W]:0xF,
            new_config: false,
            multi_stream: true,
            id: u32:10,
            literals_last: true,
        };
        let tok = send(tok, ctrl_s, ctrl);

        let (tok, mem_rd_req) = recv(tok, mem_rd_req_r);
        trace_fmt!("[TEST] Received jump table read request");
        assert_eq(MemReaderReq {addr: uN[TEST_AXI_ADDR_W]:0x1, length: uN[TEST_AXI_ADDR_W]:0x6}, mem_rd_req);

        let tok = send(tok, mem_rd_resp_s, MemReaderResp {status: mem_reader::MemReaderStatus::OKAY, data: uN[TEST_AXI_DATA_W]:0x0003_0002_0001, length: uN[TEST_AXI_ADDR_W]:0x6, last:true});

        const TEST_STREAM_ADDR = uN[TEST_AXI_ADDR_W][4]:[
            ctrl.base_addr + JUMP_TABLE_SIZE,
            ctrl.base_addr + JUMP_TABLE_SIZE + uN[TEST_AXI_ADDR_W]:0x3,
            ctrl.base_addr + JUMP_TABLE_SIZE + uN[TEST_AXI_ADDR_W]:0x5,
            ctrl.base_addr + JUMP_TABLE_SIZE + uN[TEST_AXI_ADDR_W]:0x6,
        ];
        const TEST_STREAM_LENGTH = uN[TEST_AXI_ADDR_W][4]:[
            uN[TEST_AXI_ADDR_W]:0x3,
            uN[TEST_AXI_ADDR_W]:0x2,
            uN[TEST_AXI_ADDR_W]:0x1,
            uN[TEST_AXI_ADDR_W]:0x3,
        ];

        let (tok, prescan_start) = recv(tok, prescan_start_r);
        trace_fmt!("[TEST] Received prescan START");
        assert_eq(true, prescan_start);

        let (tok, code_builder_start) = recv(tok, code_builder_start_r);
        trace_fmt!("[TEST] Received code builder START");
        assert_eq(true, code_builder_start);

        for (i, tok) in u32:0..u32:4 {
            trace_fmt!("[TEST] Stream #{}", i);

            let (tok, axi_reader_ctrl) = recv(tok, axi_reader_ctrl_r);
            trace_fmt!("[TEST] Received AXI reader CTRL");
            assert_eq(AxiReaderCtrl {base_addr: TEST_STREAM_ADDR[i], len: TEST_STREAM_LENGTH[i]}, axi_reader_ctrl);

            let (tok, data_preprocess_start) = recv(tok, data_preprocess_start_r);
            trace_fmt!("[TEST] Received data preprocess START");
            assert_eq(DataPreprocessorStart {new_config: ctrl.new_config}, data_preprocess_start);

            let (tok, decoder_start) = recv(tok, decoder_start_r);
            trace_fmt!("[TEST] Received decoder START");
            assert_eq(DecoderStart {new_config: ctrl.new_config, id: ctrl.id, literals_last: ctrl.literals_last, last_stream: (i == u32:3) }, decoder_start);

            let tok = send(tok, decoder_done_s, ());

            tok
        }(tok);

        let (tok, resp) = recv(tok, resp_r);
        trace_fmt!("[TEST] Received resp");
        assert_eq(Resp {status: Status::OKAY}, resp);

        // 4 Streams
        // With new config
        trace_fmt!("[TEST] Case #4");
        let ctrl = Ctrl {
            base_addr: uN[TEST_AXI_ADDR_W]:0x1,
            len: uN[TEST_AXI_ADDR_W]:0x50,
            new_config: true,
            multi_stream: true,
            id: u32:0,
            literals_last: false,
        };
        let tok = send(tok, ctrl_s, ctrl);

        let (tok, weights_dec_req) = recv(tok, weights_dec_req_r);
        trace_fmt!("[TEST] Received weights decode request");
        assert_eq(WeightsDecReq {addr: uN[TEST_AXI_ADDR_W]:0x1}, weights_dec_req);

        // Signal Weight decoding done
        let tree_description_size = uN[TEST_AXI_ADDR_W]:0x25;
        let tok = send(tok, weights_dec_resp_s, WeightsDecResp{
            status: WeightsDecStatus::OKAY,
            tree_description_size: tree_description_size
        });

        let (tok, mem_rd_req) = recv(tok, mem_rd_req_r);
        trace_fmt!("[TEST] Received jump table read request");
        assert_eq(MemReaderReq {addr: uN[TEST_AXI_ADDR_W]:0x26, length: uN[TEST_AXI_ADDR_W]:0x6}, mem_rd_req);

        let tok = send(tok, mem_rd_resp_s, MemReaderResp {status: mem_reader::MemReaderStatus::OKAY, data: uN[TEST_AXI_DATA_W]:0x0003_0002_0001, length: uN[TEST_AXI_ADDR_W]:0x6, last:true});

        const TEST_STREAM_ADDR = uN[TEST_AXI_ADDR_W][4]:[
            ctrl.base_addr + tree_description_size + JUMP_TABLE_SIZE,
            ctrl.base_addr + tree_description_size + JUMP_TABLE_SIZE + uN[TEST_AXI_ADDR_W]:0x3,
            ctrl.base_addr + tree_description_size + JUMP_TABLE_SIZE + uN[TEST_AXI_ADDR_W]:0x5,
            ctrl.base_addr + tree_description_size + JUMP_TABLE_SIZE + uN[TEST_AXI_ADDR_W]:0x6,
        ];
        const TEST_STREAM_LENGTH = uN[TEST_AXI_ADDR_W][4]:[
            uN[TEST_AXI_ADDR_W]:0x3,
            uN[TEST_AXI_ADDR_W]:0x2,
            uN[TEST_AXI_ADDR_W]:0x1,
            uN[TEST_AXI_ADDR_W]:0x1F,
        ];

        let (tok, prescan_start) = recv(tok, prescan_start_r);
        trace_fmt!("[TEST] Received prescan START");
        assert_eq(true, prescan_start);

        let (tok, code_builder_start) = recv(tok, code_builder_start_r);
        trace_fmt!("[TEST] Received code builder START");
        assert_eq(true, code_builder_start);

        for (i, tok) in u32:0..u32:4 {
            trace_fmt!("[TEST] Stream #{}", i);

            let (tok, axi_reader_ctrl) = recv(tok, axi_reader_ctrl_r);
            trace_fmt!("[TEST] Received AXI reader CTRL");
            assert_eq(AxiReaderCtrl {base_addr: TEST_STREAM_ADDR[i], len: TEST_STREAM_LENGTH[i]}, axi_reader_ctrl);

            let (tok, data_preprocess_start) = recv(tok, data_preprocess_start_r);
            trace_fmt!("[TEST] Received data preprocess START");
            assert_eq(DataPreprocessorStart {new_config: ctrl.new_config && (i == u32:0)}, data_preprocess_start);

            let (tok, decoder_start) = recv(tok, decoder_start_r);
            trace_fmt!("[TEST] Received decoder START");
            assert_eq(DecoderStart {new_config: ctrl.new_config && (i == u32:0), id: ctrl.id, literals_last: ctrl.literals_last, last_stream: (i == u32:3) }, decoder_start);

            let tok = send(tok, decoder_done_s, ());

            tok
        }(tok);

        send(tok, terminator, true);
    }
}
