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

const JUMP_TABLE_SIZE = u32:6;

struct HuffmanControlAndSequenceMultiStreamHandlerConfig<AXI_ADDR_W: u32> {
    multi_stream: bool,
    id: u32,
    literals_last: bool,
    base_addr: uN[AXI_ADDR_W],
    new_config: bool,
    tree_description_size: uN[AXI_ADDR_W],
    stream_sizes: uN[AXI_ADDR_W][4], // for multi-stream
    stream_len: uN[AXI_ADDR_W]
}

struct HuffmanControlAndSequenceMultiStreamHandlerState<AXI_ADDR_W: u32> {
    active: bool,
    stream_no: u2,
    config: HuffmanControlAndSequenceMultiStreamHandlerConfig<AXI_ADDR_W>,
}

struct HuffmanControlAndSequenceInternalRequest<AXI_ADDR_W: u32> {

}

// pub proc HuffmanControlAndSequenceInternal<AXI_ADDR_W: u32> {
//     type State = HuffmanControlAndSequenceMultiStreamHandlerState<AXI_ADDR_W>;
//     type DecoderStart = decoder::HuffmanDecoderStart;
//     type Config = HuffmanControlAndSequenceMultiStreamHandlerConfig<AXI_ADDR_W>;
//     type AxiReaderCtrl = axi_reader::HuffmanAxiReaderCtrl<AXI_ADDR_W>;
//     type DataPreprocessorStart = data_preprocessor::HuffmanDataPreprocessorStart;

//     config_r: chan<Config> in;
//     done_s: chan<()> out;
//     decoder_start_s: chan<DecoderStart> out;
//     decoder_done_r: chan<()> in;
//     axi_reader_ctrl_s: chan<AxiReaderCtrl> out;
//     data_preprocess_start_s: chan<DataPreprocessorStart> out;

//     config(

//         decoder_start_s: chan<DecoderStart> out,
//         decoder_done_r: chan<()> in,
//         axi_reader_ctrl_s: chan<AxiReaderCtrl> out,
//         data_preprocess_start_s: chan<DataPreprocessorStart> out,
//     ) {
//         (
//             config_r, done_s,
//             decoder_start_s, decoder_done_r,
//             axi_reader_ctrl_s, data_preprocess_start_s
//         )
//     }

//     init { }

//     next(state: ()) {
//         let tok = join();
//         let (tok, req) = recv(tok, )
//         let tok = send(tok, axi_reader_ctrl_s, req.axi_reader_ctrl);
//         trace_fmt!("[HuffmanControlAndSequence] Sent request to AXI reader: {:#x}", axi_reader_ctrl);
//         let tok = send(tok, data_preprocess_start_s, req.preprocessor_start);
//         trace_fmt!("[HuffmanControlAndSequence] Sent preprocessor start: {:#x}", preprocessor_start);
//         let tok = send(tok, decoder_start_s, req.decoder_start);
//         trace_fmt!("[HuffmanControlAndSequence] Sent decoder start: {:#x}", decoder_start);
//         let (tok, done) = recv(tok, decoder_done_r);
//         trace_fmt!("[HuffmanControlAndSequence] Received Decoder Done {}", done);

//     }
// }

pub proc HuffmanControlAndSequenceMultiStreamHandler<AXI_ADDR_W: u32> {
    type State = HuffmanControlAndSequenceMultiStreamHandlerState<AXI_ADDR_W>;
    type DecoderStart = decoder::HuffmanDecoderStart;
    type Config = HuffmanControlAndSequenceMultiStreamHandlerConfig<AXI_ADDR_W>;
    type AxiReaderCtrl = axi_reader::HuffmanAxiReaderCtrl<AXI_ADDR_W>;
    type DataPreprocessorStart = data_preprocessor::HuffmanDataPreprocessorStart;

    const MAX_STREAM_NO = u2:3;

    config_r: chan<Config> in;
    done_s: chan<()> out;
    decoder_start_s: chan<DecoderStart> out;
    decoder_done_r: chan<()> in;
    axi_reader_ctrl_s: chan<AxiReaderCtrl> out;
    data_preprocess_start_s: chan<DataPreprocessorStart> out;

    config(
        config_r: chan<Config> in,
        done_s: chan<()> out,
        decoder_start_s: chan<DecoderStart> out,
        decoder_done_r: chan<()> in,
        axi_reader_ctrl_s: chan<AxiReaderCtrl> out,
        data_preprocess_start_s: chan<DataPreprocessorStart> out,
    ) {
        (
            config_r, done_s,
            decoder_start_s, decoder_done_r,
            axi_reader_ctrl_s, data_preprocess_start_s
        )
    }

    init { zero!<State>() }

    next(state: State) {
        let config = state.config;
        let multi_stream = config.multi_stream;
        let stream_no = state.stream_no;
        let new_config = config.new_config;
        let td_size = config.tree_description_size;
        let base_addr = config.base_addr;
        let stream_sizes = config.stream_sizes;
        let stream_len = config.stream_len;

        let (huffman_stream_addr, huffman_stream_len) = match(new_config, multi_stream, stream_no) {
            (false, false, _) => (base_addr, stream_len),
            (true, false, _) => ((base_addr + td_size), (stream_len - td_size)),

            (false, true, u2:0) => ((base_addr + JUMP_TABLE_SIZE as uN[AXI_ADDR_W]), (stream_sizes[0] as uN[AXI_ADDR_W])),
            (false, true, u2:1) => ((base_addr + JUMP_TABLE_SIZE as uN[AXI_ADDR_W] + stream_sizes[0]), (stream_sizes[1] as uN[AXI_ADDR_W])),
            (false, true, u2:2) => ((base_addr + JUMP_TABLE_SIZE as uN[AXI_ADDR_W] + stream_sizes[0] + stream_sizes[1]), (stream_sizes[2] as uN[AXI_ADDR_W])),
            (false, true, u2:3) => ((base_addr + JUMP_TABLE_SIZE as uN[AXI_ADDR_W] + stream_sizes[0] + stream_sizes[1] + stream_sizes[2]), (stream_sizes[3] as uN[AXI_ADDR_W])),

            (true, true, u2:0) => ((base_addr + td_size + JUMP_TABLE_SIZE as uN[AXI_ADDR_W]), (stream_sizes[0] as uN[AXI_ADDR_W])),
            (true, true, u2:1) => ((base_addr + td_size + JUMP_TABLE_SIZE as uN[AXI_ADDR_W] + stream_sizes[0]), (stream_sizes[1] as uN[AXI_ADDR_W])),
            (true, true, u2:2) => ((base_addr + td_size + JUMP_TABLE_SIZE as uN[AXI_ADDR_W] + stream_sizes[0] + stream_sizes[1]), (stream_sizes[2] as uN[AXI_ADDR_W])),
            (true, true, u2:3) => ((base_addr + td_size + JUMP_TABLE_SIZE as uN[AXI_ADDR_W] + stream_sizes[0] + stream_sizes[1] + stream_sizes[2]), (stream_sizes[3] as uN[AXI_ADDR_W])),
        };
        let preprocessor_start = DataPreprocessorStart { new_config: new_config && stream_no == u2:0 };
        let axi_reader_ctrl = AxiReaderCtrl {
            base_addr: huffman_stream_addr,
            len: huffman_stream_len,
        };
        let decoder_start = DecoderStart {
            new_config: new_config && stream_no == u2:0,
            id: config.id,
            literals_last: config.literals_last,
            last_stream: !multi_stream || stream_no == MAX_STREAM_NO,
        };

        if !state.active {
            let (tok, config) = recv(join(), config_r);
            trace_fmt!("[HuffmanControlAndSequence] Multi-stream handler received {}", config);

            State {
                active: true,
                stream_no: u2:0,
                config: config
            }
        } else {
            if multi_stream {
                trace_fmt!("[HuffmanControlAndSequence] Processing multi-stream: {}/{}", stream_no as u32 + u32:1, MAX_STREAM_NO as u32 + u32:1);
            } else {
                trace_fmt!("[HuffmanControlAndSequence] Processing single stream");
            };
            let tok = send(join(), axi_reader_ctrl_s, axi_reader_ctrl);
            trace_fmt!("[HuffmanControlAndSequence] Sent request to AXI reader: {:#x}", axi_reader_ctrl);
            let tok = send(tok, data_preprocess_start_s, preprocessor_start);
            trace_fmt!("[HuffmanControlAndSequence] Sent preprocessor start: {:#x}", preprocessor_start);
            let tok = send(tok, decoder_start_s, decoder_start);
            trace_fmt!("[HuffmanControlAndSequence] Sent decoder start: {:#x}", decoder_start);
            let (tok, done) = recv(tok, decoder_done_r);
            trace_fmt!("[HuffmanControlAndSequence] Received Decoder Done {}", done);
            let last_iter = (!multi_stream || stream_no == MAX_STREAM_NO) && done == ();
            let tok = send_if(tok, done_s, last_iter, ());

            State {
                active: !last_iter,
                stream_no: stream_no + u2:1,
                config: config
            }
        }
    }
}

pub proc HuffmanControlAndSequence<AXI_ADDR_W: u32, AXI_DATA_W: u32> {
    type AxiReaderCtrl = axi_reader::HuffmanAxiReaderCtrl<AXI_ADDR_W>;
    type DataPreprocessorStart = data_preprocessor::HuffmanDataPreprocessorStart;
    type DecoderStart = decoder::HuffmanDecoderStart;
    type WeightsDecReq = weights_dec::HuffmanWeightsDecoderReq<AXI_ADDR_W>;
    type WeightsDecResp = weights_dec::HuffmanWeightsDecoderResp<AXI_ADDR_W>;
    type MemReaderReq  = mem_reader::MemReaderReq<AXI_ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<AXI_DATA_W, AXI_ADDR_W>;

    type MultiStreamConfig = HuffmanControlAndSequenceMultiStreamHandlerConfig<AXI_ADDR_W>;
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

    // MemReader interface for fetching the Jump Table
    mem_rd_req_s: chan<MemReaderReq> out;
    mem_rd_resp_r: chan<MemReaderResp> in;

    multi_stream_config_s: chan<MultiStreamConfig> out;
    multi_stream_resp_r: chan<()> in;

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
        let (multi_stream_config_s, multi_stream_config_r) = chan<MultiStreamConfig, u32:1>("multi_stream_config");
        let (multi_stream_resp_s, multi_stream_resp_r) = chan<(), u32:1>("multi_stream_resp");

        spawn HuffmanControlAndSequenceMultiStreamHandler<AXI_ADDR_W>
        (
            multi_stream_config_r, multi_stream_resp_s,
            decoder_start_s, decoder_done_r,
            axi_reader_ctrl_s,
            data_preprocess_start_s
        );

        (
            ctrl_r, resp_s,
            weights_dec_req_s,
            weights_dec_resp_r,
            prescan_start_s,
            code_builder_start_s,
            mem_rd_req_s, mem_rd_resp_r,
            multi_stream_config_s, multi_stream_resp_r
        )
    }

    init {}

    next (state: ()) {
        // receive start
        let tok = join();
        let (tok, ctrl) = recv(tok, ctrl_r);
        trace_fmt!("[HuffmanControlAndSequence] Received Ctrl: {:#x}", ctrl);

        let new_config = ctrl.new_config;
        let multi_stream = ctrl.multi_stream;

        // New config means the requirement to read and decode new Huffman Tree Description
        // Delegate this task to HuffmanWeightsDecoder
        let (tok, tree_description_size) = if new_config {
            let weights_dec_req = WeightsDecReq {
                addr: ctrl.base_addr
            };
            let tok = send(tok, weights_dec_req_s, weights_dec_req);
            trace_fmt!("[HuffmanControlAndSequence] Sent Weights Decoding Request: {:#x}", weights_dec_req);
            let (tok, weights_dec_resp) = recv(tok, weights_dec_resp_r);
            trace_fmt!("[HuffmanControlAndSequence] Received Weights Decoding response: {:#x}", weights_dec_resp);
            (tok, weights_dec_resp.tree_description_size)
        } else {
           (tok, uN[AXI_ADDR_W]:0)
        };
        trace_fmt!("[HuffmanControlAndSequence] Tree description size: {:#x}", tree_description_size);

        let (tok, stream_sizes, stream_len) = if multi_stream {
            // Fetch the Jump Table if neccessary
            let jump_table_req = MemReaderReq {
                addr: ctrl.base_addr + tree_description_size,
                length: JUMP_TABLE_SIZE as uN[AXI_ADDR_W],
            };
            let tok = send(tok, mem_rd_req_s, jump_table_req);
            trace_fmt!("[HuffmanControlAndSequence] Sent Jump Table read request {:#x}", jump_table_req);
            let (tok, jump_table_raw) = recv(tok, mem_rd_resp_r);

            let stream_sizes = jump_table_raw.data[0:48] as u16[3];
            let total_streams_size = ctrl.len - tree_description_size;
            let stream_sizes = uN[AXI_ADDR_W][4]:[
                stream_sizes[0] as uN[AXI_ADDR_W],
                stream_sizes[1] as uN[AXI_ADDR_W],
                stream_sizes[2] as uN[AXI_ADDR_W],
                total_streams_size - JUMP_TABLE_SIZE as uN[AXI_ADDR_W] - (stream_sizes[0] + stream_sizes[1] + stream_sizes[2]) as uN[AXI_ADDR_W]
            ];

            trace_fmt!("[HuffmanControlAndSequence] Received Jump Table: {:#x}", jump_table_raw);
            trace_fmt!("[HuffmanControlAndSequence] Total streams size: {:#x}", total_streams_size);
            trace_fmt!("[HuffmanControlAndSequence] Stream sizes: {:#x}", stream_sizes);

            (tok, stream_sizes, uN[AXI_ADDR_W]:0)
        } else {
            (tok, zero!<uN[AXI_ADDR_W][4]>(), ctrl.len)
        };

        let tok = send(tok, prescan_start_s, true);
        let tok = send(tok, code_builder_start_s, true);
        trace_fmt!("[HuffmanControlAndSequence] Sent START to prescan and code builder");

        let tok = send(
            tok, multi_stream_config_s, MultiStreamConfig {
                multi_stream: multi_stream,
                id: ctrl.id,
                literals_last: ctrl.literals_last,
                base_addr: ctrl.base_addr,
                new_config: new_config,
                tree_description_size: tree_description_size,
                stream_sizes: stream_sizes,
                stream_len: stream_len
            }
        );
        let (tok, _) = recv(tok, multi_stream_resp_r);

        let resp = Resp { status: Status::OKAY };
        let tok = send(tok, resp_s, resp);
        trace_fmt!("[HuffmanControlAndSequence] Sent Ctrl response: {:#x}", resp);
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
