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

// This file contains the implementation of Huffmann data preprocessor.
// 1. Buffer data
// 2. Remove prefix
// 3. Forward it to HuffmanDecoder

import std;

import xls.modules.zstd.huffman_common as hcommon;
import xls.modules.zstd.huffman_axi_reader as huffman_axi_reader;

type Config = hcommon::CodeBuilderToPreDecoderOutput;

pub const H_DATA_W = hcommon::MAX_CODE_LEN * u32:8;
pub const H_DATA_W_LOG2 = std::clog2(H_DATA_W + u32:1);

pub type Data = uN[H_DATA_W];

pub type CodeLen = uN[H_DATA_W_LOG2];

const MAX_PREFIX_LEN = u4:7;

enum HuffmanDataPreprocessorFSM: u2 {
    IDLE = 0,
    AWAITING_CONFIG = 1,
    READ_DATA = 2,
}

pub struct HuffmanDataPreprocessorStart {
    new_config: bool
}

pub struct HuffmanDataPreprocessorData {
    data: Data,
    data_len: CodeLen,
    code_length: CodeLen[H_DATA_W],
    last: bool,
}

struct HuffmanDataPreprocessorState {
    lookahead_config: Config
}

struct HuffmanDataPreprocessorRxState {
    active: bool,
    in_buff: Data,
    in_len: CodeLen
}

struct HuffmanDataPreprocessorRxResp {
    data: Data,
    data_len: CodeLen,
    last: bool
}

pub proc HuffmanDataPreprocessorRx {
    type State = HuffmanDataPreprocessorRxState;
    type Resp = HuffmanDataPreprocessorRxResp;
    type DataIn = huffman_axi_reader::HuffmanAxiReaderData;

    req_r: chan<()> in;
    resp_s: chan<Resp> out;
    data_r: chan<DataIn> in;

    config(
        req_r: chan<()> in,
        resp_s: chan<Resp> out,
        data_r: chan<DataIn> in
    ) {
        (req_r, resp_s, data_r)
    }
    init { zero!<State>() }
    next(state: State) {
        let tok = join();
        if !state.active {
            let (tok, _) = recv(tok, req_r);
            State {
                active: true,
                ..zero!<State>()
            }
        } else {
            let (tok, data) = recv(tok, data_r);
            let next_in_len = state.in_len + CodeLen:8;
            let next_in_buff = state.in_buff | ((rev(data.data) as Data) << state.in_len);
            let last_iter = data.last || next_in_len >= H_DATA_W as CodeLen;
            trace_fmt!("[HuffmanDataPreprocessor] Data in state: {:#b} (len: {})", next_in_buff, next_in_len);
            let tok = send_if(tok, resp_s, last_iter, Resp {
                data: next_in_buff,
                data_len: next_in_len,
                last: data.last
            });

            State {
                active: !last_iter,
                in_buff: next_in_buff,
                in_len: next_in_len
            }
        }
    }
}

pub proc HuffmanDataPreprocessor {
    type State = HuffmanDataPreprocessorState;
    type FSM = HuffmanDataPreprocessorFSM;
    type Start = HuffmanDataPreprocessorStart;
    type DataIn = huffman_axi_reader::HuffmanAxiReaderData;
    type PreprocessedData = HuffmanDataPreprocessorData;
    type RxResp = HuffmanDataPreprocessorRxResp;

    start_r: chan<Start> in;
    lookahead_config_r: chan<Config> in;

    rx_req_s: chan<()> out;
    rx_resp_r: chan<RxResp> in;
    preprocessed_data_s: chan<PreprocessedData> out;

    config (
        start_r: chan<Start> in,
        lookahead_config_r: chan<Config> in,
        data_r: chan<DataIn> in,
        preprocessed_data_s: chan<PreprocessedData> out,
    ) {
        let (rx_req_s, rx_req_r) = chan<(), u32:1>("hdp_rx_req");
        let (rx_resp_s, rx_resp_r) = chan<RxResp, u32:1>("hdp_rx_resp");

        spawn HuffmanDataPreprocessorRx (
            rx_req_r, rx_resp_s, data_r
        );

        (
            start_r, lookahead_config_r,
            rx_req_s, rx_resp_r,
            preprocessed_data_s,
        )
    }

    init { zero!<State>() }

    next (state: State) {
        let tok = join();
        let (tok, start) = recv(tok, start_r);
        let (tok, lookahead_config) = recv_if(tok, lookahead_config_r, start.new_config, state.lookahead_config);
        if start.new_config {
            trace_fmt!("[HuffmanDataPreprocessor] Received config {:#x}", lookahead_config);
        } else {};

        let tok = send(tok, rx_req_s, ());
        let (tok, data) = recv(tok, rx_resp_r);
        trace_fmt!("[HuffmanDataPreprocessor] RX data {:#b}", data);

        // remove prefix
        let data_bits = data.data;
        let (prefix_len, _) = for (i, (prefix_len, stop)): (u32, (u4, bool)) in range(u32:0, MAX_PREFIX_LEN as u32) {
            if stop || (data_bits >> i) as u1 {
                (
                    prefix_len,
                    true,
                )
            } else {
                (
                    prefix_len + u4:1,
                    stop,
                )
            }
        }((u4:1, false));
        trace_fmt!("[HuffmanDataPreprocessor] Prefix len: {}", prefix_len);


        let data_bits = data_bits >> prefix_len;
        let data_bits_len = data.data_len - prefix_len as CodeLen;

        let tok = send(tok, preprocessed_data_s, PreprocessedData {
            data: data_bits,
            data_len: data_bits_len,
            last: data.last,
            code_length: zero!<CodeLen[H_DATA_W]>(), // unused, kept not to break proc interface (for now)
        });

        trace_fmt!("[HuffmanDataPreprocessor] Sent preprocessed data {:#x} (length {})", data_bits, data_bits_len);

        State {
            lookahead_config: lookahead_config
        }
    }
}

const TEST_START = HuffmanDataPreprocessorStart[2]:[
    HuffmanDataPreprocessorStart {
        new_config: true,
    },
    HuffmanDataPreprocessorStart {
        new_config: true,
    },
];

const TEST_CONFIG = Config[2]:[
    Config {
        max_code_length: uN[hcommon::WEIGHT_LOG]:6,
        valid_weights: [false, true, false, true, true, false, true, false, false, false, false, false]
    },
    Config {
        max_code_length: uN[hcommon::WEIGHT_LOG]:9,
        valid_weights: [false, true, false, false, true, false, false, true, false, true, false, false]
    }
];

const TEST_DATA = huffman_axi_reader::HuffmanAxiReaderData[12]:[
    // #1
    huffman_axi_reader::HuffmanAxiReaderData {
        data: u8:0b01010000,
        last: false,
    },
    huffman_axi_reader::HuffmanAxiReaderData {
        data: u8:0b01011011,
        last: false,
    },
    huffman_axi_reader::HuffmanAxiReaderData {
        data: u8:0b01000001,
        last: false,
    },
    huffman_axi_reader::HuffmanAxiReaderData {
        data: u8:0b01010011,
        last: true,
    },
    // #2
    huffman_axi_reader::HuffmanAxiReaderData {
        data: u8:0b00110100,
        last: false,
    },
    huffman_axi_reader::HuffmanAxiReaderData {
        data: u8:0b11110001,
        last: false,
    },
    huffman_axi_reader::HuffmanAxiReaderData {
        data: u8:0b01010000,
        last: false,
    },
    huffman_axi_reader::HuffmanAxiReaderData {
        data: u8:0b00101010,
        last: false,
    },
    huffman_axi_reader::HuffmanAxiReaderData {
        data: u8:0b11010100,
        last: false,
    },
    huffman_axi_reader::HuffmanAxiReaderData {
        data: u8:0b01000010,
        last: false,
    },
    huffman_axi_reader::HuffmanAxiReaderData {
        data: u8:0b01010101,
        last: false,
    },
    huffman_axi_reader::HuffmanAxiReaderData {
        data: u8:0b10010101,
        last: true,
    },
];

const TEST_PREPROCESSED_DATA = HuffmanDataPreprocessorData[2]:[
    HuffmanDataPreprocessorData {
        data: Data:0b110_010_1_010000_010_110_1_1_010000_010,
        data_len: CodeLen:30,
        last: true,
        code_length: [
            CodeLen:3, CodeLen:1, CodeLen:6, CodeLen:6, CodeLen:4, CodeLen:3, CodeLen:3, CodeLen:1,
            CodeLen:3, CodeLen:1, CodeLen:1, CodeLen:3, CodeLen:1, CodeLen:1, CodeLen:3, CodeLen:1,
            CodeLen:6, CodeLen:6, CodeLen:4, CodeLen:3, CodeLen:3, CodeLen:1, CodeLen:3, CodeLen:1,
            CodeLen:3, CodeLen:1, CodeLen:3, CodeLen:3, CodeLen:1, CodeLen:1, CodeLen:0, CodeLen:0,
            CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0,
            CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0,
            CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0,
            CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0,
            CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0,
            CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0,
            CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0,
            CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0,
        ],
    },
    HuffmanDataPreprocessorData {
        data: Data:0b1_010_100_110_1_010_100_100_001000_1_010_110_1_010_100000_010_101000_1_1_110_010_1,
        data_len: CodeLen:61,
        last: true,
        code_length: [
            CodeLen:1, CodeLen:3, CodeLen:1, CodeLen:3, CodeLen:3, CodeLen:1, CodeLen:1, CodeLen:1,
            CodeLen:1, CodeLen:4, CodeLen:3, CodeLen:3, CodeLen:1, CodeLen:3, CodeLen:1, CodeLen:3,
            CodeLen:1, CodeLen:6, CodeLen:6, CodeLen:6, CodeLen:4, CodeLen:3, CodeLen:3, CodeLen:1,
            CodeLen:3, CodeLen:1, CodeLen:3, CodeLen:1, CodeLen:3, CodeLen:1, CodeLen:1, CodeLen:3,
            CodeLen:1, CodeLen:3, CodeLen:1, CodeLen:4, CodeLen:3, CodeLen:3, CodeLen:1, CodeLen:6,
            CodeLen:4, CodeLen:3, CodeLen:3, CodeLen:1, CodeLen:3, CodeLen:3, CodeLen:1, CodeLen:3,
            CodeLen:1, CodeLen:3, CodeLen:1, CodeLen:3, CodeLen:1, CodeLen:1, CodeLen:3, CodeLen:3,
            CodeLen:1, CodeLen:3, CodeLen:1, CodeLen:3, CodeLen:1, CodeLen:0, CodeLen:0, CodeLen:0,
            CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0,
            CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0,
            CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0,
            CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0,
        ],
    }
];

#[test_proc]
proc HuffmanDataPreprocessor_test {
    type State = HuffmanDataPreprocessorState;
    type Start = HuffmanDataPreprocessorStart;
    type Data = huffman_axi_reader::HuffmanAxiReaderData;
    type PreprocessedData = HuffmanDataPreprocessorData;

    terminator_s: chan<bool> out;

    start_s: chan<Start> out;
    lookahead_config_s: chan<Config> out;
    data_s: chan<Data> out;

    preprocessed_data_r: chan<PreprocessedData> in;

    config (terminator_s: chan<bool> out) {
        let (start_s, start_r) = chan<Start>("start");
        let (lookahead_config_s, lookahead_config_r) = chan<Config>("lookahead_config");
        let (data_s, data_r) = chan<Data>("data");
        let (preprocessed_data_s, preprocessed_data_r) = chan<PreprocessedData>("preprocessed_data");

        spawn HuffmanDataPreprocessor(
            start_r,
            lookahead_config_r,
            data_r,
            preprocessed_data_s,
        );

        (
            terminator_s,
            start_s,
            lookahead_config_s,
            data_s,
            preprocessed_data_r,
        )
    }

    init { }

    next (state: ()) {
        let tok = join();

        let (tok, _, _) = for ((i, test_start), (tok, cfg_idx, data_idx)): ((u32, Start), (token, u32, u32)) in enumerate(TEST_START) {
            let tok = send(tok, start_s, test_start);
            trace_fmt!("Sent #{} start {:#x}", i + u32:1, test_start);

            let (tok, cfg_idx) = if test_start.new_config {
                let tok = send(tok, lookahead_config_s, TEST_CONFIG[cfg_idx]);
                trace_fmt!("Sent #{} config {:#x}", cfg_idx + u32:1, TEST_CONFIG[cfg_idx]);
                (tok, cfg_idx + u32:1)
            } else { (tok, cfg_idx) };

            let (tok, data_idx, _) = for (_, (tok, data_idx, do_send)) in range(u32:0, hcommon::MAX_CODE_LEN) {
                if data_idx < array_size(TEST_DATA) {
                    let data = TEST_DATA[data_idx];

                    if do_send {
                        let tok = send(tok, data_s, data);
                        trace_fmt!("Sent #{} data {:#x}", data_idx + u32:1, data);
                        (tok, data_idx + u32:1, !data.last)
                    } else {
                        (tok, data_idx, false)
                    }
                } else { (tok, data_idx, false) }
            }((tok, data_idx, true));

            let (tok, preprocessed_data) = recv(tok, preprocessed_data_r);
            trace_fmt!("Received #{} preprocessed data {:#x}", i + u32:1, preprocessed_data);
            assert_eq(TEST_PREPROCESSED_DATA[i], preprocessed_data);

            (tok, cfg_idx, data_idx)
        }((tok, u32:0, u32:0));

        send(tok, terminator_s, true);
    }
}
