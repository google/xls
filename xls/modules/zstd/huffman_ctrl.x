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
import xls.modules.zstd.huffman_common as hcommon;
import xls.modules.zstd.huffman_axi_reader as axi_reader;
import xls.modules.zstd.huffman_code_builder as code_builder;
import xls.modules.zstd.huffman_data_preprocessor as data_preprocessor;
import xls.modules.zstd.huffman_decoder as decoder;
import xls.modules.zstd.huffman_prescan as prescan;


enum HuffmanControlAndSequenceFSM: u2 {
    IDLE = 0,
    DECODING = 1,
}

pub struct HuffmanControlAndSequenceCtrl<AXI_ADDR_W: u32> {
    base_addr: uN[AXI_ADDR_W],
    len: uN[AXI_ADDR_W],
    new_config: bool,
}

struct HuffmanControlAndSequenceState {
    fsm: HuffmanControlAndSequenceFSM,
}

pub proc HuffmanControlAndSequence<AXI_ADDR_W: u32> {
    type AxiReaderCtrl = axi_reader::HuffmanAxiReaderCtrl<AXI_ADDR_W>;
    type DataPreprocessorStart = data_preprocessor::HuffmanDataPreprocessorStart;
    type DecoderStart = decoder::HuffmanDecoderStart;

    type State = HuffmanControlAndSequenceState;
    type FSM = HuffmanControlAndSequenceFSM;
    type Ctrl = HuffmanControlAndSequenceCtrl<AXI_ADDR_W>;

    ctrl_r: chan<Ctrl> in;

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

    config (
        ctrl_r: chan<Ctrl> in,
        prescan_start_s: chan<bool> out,
        code_builder_start_s: chan<bool> out,
        axi_reader_ctrl_s: chan<AxiReaderCtrl> out,
        data_preprocess_start_s: chan<DataPreprocessorStart> out,
        decoder_start_s: chan<DecoderStart> out,
        decoder_done_r: chan<()> in,
    ) {
        (
            ctrl_r,
            prescan_start_s,
            code_builder_start_s,
            axi_reader_ctrl_s,
            data_preprocess_start_s,
            decoder_start_s,
            decoder_done_r,
        )
    }

    init {
        zero!<State>()
    }

    next (state: State) {
        // receive start
        let (tok, ctrl, ctrl_valid) = recv_if_non_blocking(join(), ctrl_r, state.fsm == FSM::IDLE, zero!<Ctrl>());

        let state = if ctrl_valid {
            State {
                fsm: FSM::DECODING,
            }
        } else {
            state
        };

        // send start to prescan and code builder
        let new_config = ctrl_valid & ctrl.new_config;

        if new_config {
            trace_fmt!("Sending start to prescan and code builder");
        } else {};
        send_if(tok, prescan_start_s, new_config, true);
        send_if(tok, code_builder_start_s, new_config, true);

        // send address and length to AXI reader
        if ctrl_valid {
            trace_fmt!("Sending ctrl to AXI reader");
        } else {};
        send_if(tok, axi_reader_ctrl_s, ctrl_valid, AxiReaderCtrl {
            base_addr: ctrl.base_addr,
            len: ctrl.len,
        });

        // send reconfigure/keep to data preprocessor and decoder
        if ctrl_valid {
            trace_fmt!("Sending start to data preprocessor and decoder");
        } else {};
        send_if(tok, data_preprocess_start_s, ctrl_valid, DataPreprocessorStart {
            new_config: new_config,
        });
        send_if(tok, decoder_start_s, ctrl_valid, DecoderStart {
            new_config: new_config,
        });

        // receive done
        let (_, _, decoder_done_valid) = recv_if_non_blocking(tok, decoder_done_r, state.fsm == FSM::DECODING, ());
        if decoder_done_valid {
            trace_fmt!("Received decoder done");
        } else {};

        if decoder_done_valid {
            State {
                fsm: FSM::IDLE
            }
        } else {
            state
        }
    }
}


const INST_AXI_ADDR_W = u32:32;

proc HuffmanControlAndSequenceInst {
    type AxiReaderCtrl = axi_reader::HuffmanAxiReaderCtrl<INST_AXI_ADDR_W>;
    type DataPreprocessorStart = data_preprocessor::HuffmanDataPreprocessorStart;
    type DecoderStart = decoder::HuffmanDecoderStart;

    config (
        ctrl_r: chan<HuffmanControlAndSequenceCtrl<INST_AXI_ADDR_W>> in,
        prescan_start_s: chan<bool> out,
        code_builder_start_s: chan<bool> out,
        axi_reader_ctrl_s: chan<AxiReaderCtrl> out,
        data_preprocess_start_s: chan<DataPreprocessorStart> out,
        decoder_start_s: chan<DecoderStart> out,
        decoder_done_r: chan<()> in,
    ) {
        spawn HuffmanControlAndSequence<INST_AXI_ADDR_W>(
            ctrl_r,
            prescan_start_s,
            code_builder_start_s,
            axi_reader_ctrl_s,
            data_preprocess_start_s,
            decoder_start_s,
            decoder_done_r,
        );
    }

    init { }

    next (state: ()) { }
}


const TEST_AXI_ADDR_W = u32:32;

#[test_proc]
proc HuffmanControlAndSequence_test {
    type Ctrl = HuffmanControlAndSequenceCtrl<INST_AXI_ADDR_W>;
    type AxiReaderCtrl = axi_reader::HuffmanAxiReaderCtrl<TEST_AXI_ADDR_W>;
    type DataPreprocessorStart = data_preprocessor::HuffmanDataPreprocessorStart;
    type DecoderStart = decoder::HuffmanDecoderStart;

    terminator: chan<bool> out;

    ctrl_s: chan<HuffmanControlAndSequenceCtrl<INST_AXI_ADDR_W>> out;
    prescan_start_r: chan<bool> in;
    code_builder_start_r: chan<bool> in;
    axi_reader_ctrl_r: chan<AxiReaderCtrl> in;
    data_preprocess_start_r: chan<DataPreprocessorStart> in;
    decoder_start_r: chan<DecoderStart> in;
    decoder_done_s: chan<()> out;

    config (terminator: chan<bool> out) {
        let (ctrl_s, ctrl_r) = chan<Ctrl>("ctrl");
        let (prescan_start_s, prescan_start_r) = chan<bool>("prescan_start");
        let (code_builder_start_s, code_builder_start_r) = chan<bool>("code_builder_start");
        let (axi_reader_ctrl_s, axi_reader_ctrl_r) = chan<AxiReaderCtrl>("axi_reader_ctrl");
        let (data_preprocess_start_s, data_preprocess_start_r) = chan<DataPreprocessorStart>("data_preprocess_start");
        let (decoder_start_s, decoder_start_r) = chan<DecoderStart>("decoder_start");
        let (decoder_done_s, decoder_done_r) = chan<()>("decoder_done");
        
        spawn HuffmanControlAndSequence<INST_AXI_ADDR_W>(
            ctrl_r,
            prescan_start_s,
            code_builder_start_s,
            axi_reader_ctrl_s,
            data_preprocess_start_s,
            decoder_start_s,
            decoder_done_r,
        );

        (
            terminator,
            ctrl_s,
            prescan_start_r,
            code_builder_start_r,
            axi_reader_ctrl_r,
            data_preprocess_start_r,
            decoder_start_r,
            decoder_done_s,
        )
    }

    init { }

    next (state: ()) {
        let tok = join();

        // without new config
        let ctrl = Ctrl {
            base_addr: uN[TEST_AXI_ADDR_W]:0x1,
            len: uN[TEST_AXI_ADDR_W]:0x2,
            new_config: false,
        };
        let tok = send(tok, ctrl_s, ctrl);

        let (tok, axi_reader_ctrl) = recv(tok, axi_reader_ctrl_r);
        assert_eq(AxiReaderCtrl {base_addr: ctrl.base_addr, len: ctrl.len}, axi_reader_ctrl);

        let (tok, data_preprocess_start) = recv(tok, data_preprocess_start_r);
        assert_eq(DataPreprocessorStart {new_config: ctrl.new_config}, data_preprocess_start);

        let (tok, decoder_start) = recv(tok, decoder_start_r);
        assert_eq(DecoderStart {new_config: ctrl.new_config}, decoder_start);

        let tok = send(tok, decoder_done_s, ());

        // with new config
        let ctrl = Ctrl {
            base_addr: uN[TEST_AXI_ADDR_W]:0x1,
            len: uN[TEST_AXI_ADDR_W]:0x2,
            new_config: true,
        };
        let tok = send(tok, ctrl_s, ctrl);

        let (tok, prescan_start) = recv(tok, prescan_start_r);
        assert_eq(true, prescan_start);

        let (tok, code_builder_start) = recv(tok, code_builder_start_r);
        assert_eq(true, code_builder_start);

        let (tok, axi_reader_ctrl) = recv(tok, axi_reader_ctrl_r);
        assert_eq(AxiReaderCtrl {base_addr: ctrl.base_addr, len: ctrl.len}, axi_reader_ctrl);

        let (tok, data_preprocess_start) = recv(tok, data_preprocess_start_r);
        assert_eq(DataPreprocessorStart {new_config: ctrl.new_config}, data_preprocess_start);

        let (tok, decoder_start) = recv(tok, decoder_start_r);
        assert_eq(DecoderStart {new_config: ctrl.new_config}, decoder_start);

        let tok = send(tok, decoder_done_s, ());

        send(tok, terminator, true);
    }
}
