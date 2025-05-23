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
    fsm: HuffmanDataPreprocessorFSM,
    lookahead_config: Config,
    data_in: Data,
    data_in_len: CodeLen,
    data_in_last: bool,
    data_in_ready: bool,
    data_out: Data,
    data_out_len: CodeLen,
    data_out_last: bool,
    remove_prefix: bool,
}

pub proc HuffmanDataPreprocessor {
    type State = HuffmanDataPreprocessorState;
    type FSM = HuffmanDataPreprocessorFSM;
    type Start = HuffmanDataPreprocessorStart;
    type DataIn = huffman_axi_reader::HuffmanAxiReaderData;
    type PreprocessedData = HuffmanDataPreprocessorData;

    start_r: chan<Start> in;
    lookahead_config_r: chan<Config> in;
    data_r: chan<DataIn> in;

    preprocessed_data_s: chan<PreprocessedData> out;

    config (
        start_r: chan<Start> in,
        lookahead_config_r: chan<Config> in,
        data_r: chan<DataIn> in,
        preprocessed_data_s: chan<PreprocessedData> out,
    ) {
        (
            start_r,
            lookahead_config_r,
            data_r,
            preprocessed_data_s,
        )
    }

    init { zero!<State>() }

    next (state: State) {
        let tok = join();

        // wait for start
        let (tok, start, start_valid) = recv_if_non_blocking(tok, start_r, state.fsm == FSM::IDLE, zero!<Start>());

        let state = if start_valid {
            let fsm = if start.new_config {
                trace_fmt!("[HuffmanDataPreprocessor] Waiting for new config");
                FSM::AWAITING_CONFIG
            } else {
                FSM::READ_DATA
            };
            State {
                fsm: fsm,
                ..state
            }
        } else { state };

        // wait for config
        let (tok, config, config_valid) = recv_if_non_blocking(
            tok,
            lookahead_config_r,
            state.fsm == FSM::AWAITING_CONFIG,
            zero!<Config>()
        );

        let state = if config_valid {
            trace_fmt!("[HuffmanDataPreprocessor] Received config {:#x}", config);
            State {
                fsm: FSM::READ_DATA,
                lookahead_config: config,
                remove_prefix: true,
                ..state
            }
        } else { state };

        // receive data
        let do_read_data = state.fsm == FSM::READ_DATA && (state.data_in_len < H_DATA_W as CodeLen);
        let (tok, data, data_valid) = recv_if_non_blocking(tok, data_r, do_read_data, zero!<DataIn>());

        // process data
        let state = if data_valid {
            trace_fmt!("[HuffmanDataPreprocessor] Received data {:#b}", data);
            trace_fmt!("[HuffmanDataPreprocessor] Data in state {:#b} (length {})", state.data_in, state.data_in_len);
            State {
                data_in: state.data_in | ((rev(data.data) as Data) << state.data_in_len),
                data_in_len: state.data_in_len + CodeLen:8,
                data_in_last: data.last,
                data_in_ready: data.last || ((state.data_in_len + CodeLen:8) == (H_DATA_W as CodeLen)),
                ..state
            }
        } else { state };

        let state = if state.data_in_ready && state.data_out_len == CodeLen:0 {
            State {
                data_out: state.data_in,
                data_out_len: state.data_in_len,
                data_out_last: state.data_in_last,
                data_in: uN[H_DATA_W]:0,
                data_in_len: CodeLen:0,
                data_in_last: false,
                data_in_ready: false,
                ..state
            }
        } else {
            state
        };

        let do_process_data = state.data_out_len > CodeLen:0;

        let processed_data = if do_process_data {
            let data_bits = state.data_out;
            let data_bits_len = state.data_out_len;
            trace_fmt!("[HuffmanDataPreprocessor] Processing data {:#b} (length {})", state.data_out, state.data_out_len);

            // remove prefix
            let prefix_len = if state.remove_prefix {
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
                prefix_len
            } else {
                u4:0
            };

            let data_bits = data_bits >> prefix_len;
            let data_bits_len = data_bits_len - prefix_len as CodeLen;

            // compute Huffman code lengths

            // compute number of zeros
            let (code_lengths, _) = for (i, (code_lengths, num_zeros)): (u32, (CodeLen[H_DATA_W], CodeLen)) in range(u32:0, H_DATA_W) {
                // reverse order
                let n = H_DATA_W - u32:1 - i;
                if n < data_bits_len as u32 {
                    // if non zero then reset counter, otherwise increment
                    let num_zeros = if (data_bits >> n) as u1 {
                        CodeLen:0
                    } else {
                        num_zeros + CodeLen:1
                    };
                    // clip code len by max code length
                    let code_len = if num_zeros >= state.lookahead_config.max_code_length as CodeLen {
                        state.lookahead_config.max_code_length as CodeLen
                    } else {
                        num_zeros + CodeLen:1
                    };
                    (
                        update(code_lengths, n, code_len), num_zeros
                    )
                } else {
                    (code_lengths, num_zeros)
                }
            }((zero!<CodeLen[H_DATA_W]>(), CodeLen:0));

            // round up number of zeros to possible length
            let code_lengths = for (i, code_lengths): (u32, CodeLen[H_DATA_W]) in range(u32:0, H_DATA_W) {
                if i < data_bits_len as u32 {
                    let length = for (weight, length): (u32, CodeLen) in range(u32:0, hcommon::MAX_WEIGHT + u32:1) {
                        let weight_valid = state.lookahead_config.valid_weights[weight];
                        let number_of_bits = if weight > u32:0 {
                            state.lookahead_config.max_code_length as u32 + u32:1 - weight
                        } else {
                            u32:0
                        };
                        if (code_lengths[i] <= number_of_bits as CodeLen) && weight_valid {
                            number_of_bits as CodeLen
                        } else {
                            length
                        }
                    }(code_lengths[i]);
                    update(code_lengths, i, length)
                } else {
                    code_lengths
                }
            }(code_lengths);

            PreprocessedData {
                data: data_bits,
                data_len: data_bits_len,
                last: state.data_out_last,
                code_length: code_lengths,
            }

        } else { zero!<PreprocessedData>() };

        let tok = send_if(tok, preprocessed_data_s, do_process_data, processed_data);
        if do_process_data {
            trace_fmt!("[HuffmanDataPreprocessor] Sent preprocessed data {:#x} (length {})", processed_data.data, processed_data.data_len);
        } else {};

        let state = if do_process_data {
            State {
                data_out: uN[H_DATA_W]:0,
                data_out_len: CodeLen:0,
                remove_prefix: processed_data.last,
                ..state
            }
        } else { state };

        state
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
