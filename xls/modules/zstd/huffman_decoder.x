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

// This file contains the implementation of Huffman decoder.

import std;

import xls.modules.zstd.common as common;
import xls.modules.zstd.huffman_common as hcommon;
import xls.modules.zstd.huffman_data_preprocessor as huffman_data_preprocessor;

type Codes = hcommon::CodeBuilderToDecoderOutput;
type CodeLen = huffman_data_preprocessor::CodeLen;

const SYMBOLS_N = u32:1 << common::SYMBOL_WIDTH;

const H_DATA_W = hcommon::MAX_CODE_LEN * u32:8;
const H_DATA_W_LOG2 = std::clog2(H_DATA_W + u32:1);

const BUFF_W = H_DATA_W * u32:2;
const BUFF_W_LOG2 = std::clog2(BUFF_W + u32:1);

enum HuffmanDecoderFSM: u3 {
    IDLE = 0,
    AWAITING_CONFIG = 1,
    READ_DATA = 2,
    DECODE = 3,
}

pub struct HuffmanDecoderStart {
    new_config: bool,
    id: u32,
    literals_last: bool,
    last_stream: bool,    // 4'th huffman coded stream decoding for multi_stream
                          // or single stream decoding
}

struct HuffmanDecoderState {
    fsm: HuffmanDecoderFSM,
    symbol_config_id: u5,
    symbol_valid: bool[SYMBOLS_N],
    symbol_code: uN[hcommon::MAX_WEIGHT][SYMBOLS_N],
    symbol_code_len: uN[hcommon::WEIGHT_LOG][SYMBOLS_N],
    data_len: uN[BUFF_W_LOG2],
    data: uN[BUFF_W],
    data_last: bool,
    code_length: CodeLen[BUFF_W],
    decoded_literals: uN[common::SYMBOL_WIDTH][u32:8],
    decoded_literals_len: u4,
    id: u32,
    literals_last: bool,
    last_stream: bool,
}

fn extend_buff_array<N: u32, M:u32>(buff: CodeLen[N], buff_len: u32, array: CodeLen[M]) -> CodeLen[N] {
    const ELEM_SIZE = huffman_data_preprocessor::H_DATA_W_LOG2;

    let buff_flat = buff as uN[ELEM_SIZE * N];
    let array_flat = array as uN[ELEM_SIZE * M];
    let buff_flat = (
        buff_flat |
        (array_flat as uN[ELEM_SIZE * N] << (ELEM_SIZE * (N - M - buff_len)))
    );
    buff_flat as CodeLen[N]
}

#[test]
fn extend_buff_array_test() {
    assert_eq(
        CodeLen[8]:[CodeLen:1, CodeLen:2, CodeLen:3, CodeLen:4, CodeLen:5, CodeLen:0, CodeLen:0, CodeLen:0],
        extend_buff_array(
            CodeLen[8]:[CodeLen:1, CodeLen:2, CodeLen:3, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0],
            u32:3,
            CodeLen[2]:[CodeLen:4, CodeLen:5],
        ),
    );
    assert_eq(
        CodeLen[8]:[CodeLen:1, CodeLen:2, CodeLen:3, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0],
        extend_buff_array(
            zero!<CodeLen[8]>(),
            u32:0,
            CodeLen[3]:[CodeLen:1, CodeLen:2, CodeLen:3],
        ),
    );
}

fn shift_buff_array<N: u32>(buff: CodeLen[N], shift: u32) -> CodeLen[N] {
    const ELEM_SIZE = huffman_data_preprocessor::H_DATA_W_LOG2;

    let buff_flat = buff as uN[ELEM_SIZE * N];
    let buff_flat = buff_flat << (ELEM_SIZE * shift);
    buff_flat as CodeLen[N]
}

#[test]
fn shift_buff_array_test() {
    assert_eq(
        CodeLen[8]:[CodeLen:4, CodeLen:5, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0],
        shift_buff_array(
            CodeLen[8]:[CodeLen:1, CodeLen:2, CodeLen:3, CodeLen:4, CodeLen:5, CodeLen:0, CodeLen:0, CodeLen:0],
            u32:3,
        ),
    );
    assert_eq(
        CodeLen[8]:[CodeLen:1, CodeLen:2, CodeLen:3, CodeLen:4, CodeLen:5, CodeLen:0, CodeLen:0, CodeLen:0],
        shift_buff_array(
            CodeLen[8]:[CodeLen:1, CodeLen:2, CodeLen:3, CodeLen:4, CodeLen:5, CodeLen:0, CodeLen:0, CodeLen:0],
            u32:0,
        ),
    );
}

pub proc HuffmanDecoder {
    type State = HuffmanDecoderState;
    type FSM = HuffmanDecoderFSM;
    type Start = HuffmanDecoderStart;
    type Data = huffman_data_preprocessor::HuffmanDataPreprocessorData;

    start_r: chan<Start> in;
    codes_r: chan<Codes> in;
    data_r: chan<Data> in;

    done_s: chan<()> out;
    decoded_literals_s: chan<common::LiteralsDataWithSync> out;

    config (
        start_r: chan<Start> in,
        codes_r: chan<Codes> in,
        data_r: chan<Data> in,
        done_s: chan<()> out,
        decoded_literals_s: chan<common::LiteralsDataWithSync> out,
    ) {
        (
            start_r,
            codes_r,
            data_r,
            done_s,
            decoded_literals_s,
        )
    }

    init { zero!<State>() }

    next (state: State) {
        let tok = join();

        // wait for start
        let (tok, start, start_valid) = recv_if_non_blocking(
            tok, start_r, state.fsm == FSM::IDLE, zero!<HuffmanDecoderStart>()
        );

        let state = if start_valid {
            if start.new_config {
                trace_fmt!("[HuffmanDecoder] {} -> AWAITING_CONFIG", state.fsm);
                assert!(state.fsm == FSM::IDLE, "invalid_state_transition");
                State {
                    fsm: FSM::AWAITING_CONFIG,
                    symbol_config_id: u5:0,
                    id: start.id,
                    literals_last: start.literals_last,
                    last_stream: start.last_stream,
                    ..state
                }
            } else {
                trace_fmt!("[HuffmanDecoder] {} -> READ_DATA", state.fsm);
                assert!(state.fsm == FSM::IDLE, "invalid_state_transition");
                State {
                    fsm: FSM::READ_DATA,
                    id: start.id,
                    literals_last: start.literals_last,
                    last_stream: start.last_stream,
                    ..state
                }
            }
        } else { state };

        // wait for config
        let (tok, config) = recv_if(
            tok,
            codes_r,
            state.fsm == FSM::AWAITING_CONFIG,
            zero!<Codes>()
        );

        let state = if state.fsm == FSM::AWAITING_CONFIG {
            let (symbol_valid, symbol_code, symbol_code_len) =
                for (i, (symbol_valid, symbol_code, symbol_code_len)):
                    (
                        u32,
                        (
                            bool[SYMBOLS_N],
                            uN[hcommon::MAX_WEIGHT][SYMBOLS_N],
                            uN[hcommon::WEIGHT_LOG][SYMBOLS_N],
                        )
                    ) in range(u32:0, hcommon::PARALLEL_ACCESS_WIDTH) {
                (
                    update(symbol_valid, (state.symbol_config_id as u32 * u32:8) + i, config.symbol_valid[i]),
                    update(symbol_code, (state.symbol_config_id as u32 * u32:8) + i, config.code[i]),
                    update(symbol_code_len, (state.symbol_config_id as u32 * u32:8) + i, config.code_length[i]),
                )
            }((state.symbol_valid, state.symbol_code, state.symbol_code_len));
            trace_fmt!("[HuffmanDecoder] state.symbol_config_id+1: {:#x}", state.symbol_config_id as u32 + u32:1);
            trace_fmt!("[HuffmanDecoder] SYMBOLS_N: {:#x}", SYMBOLS_N);
            trace_fmt!("[HuffmanDecoder] hcommon::PARALLEL_ACCESS_WIDTH: {:#x}", hcommon::PARALLEL_ACCESS_WIDTH);
            let fsm = if (state.symbol_config_id as u32 + u32:1) == (SYMBOLS_N / hcommon::PARALLEL_ACCESS_WIDTH) {
                trace_fmt!("[HuffmanDecoder] {} -> READ_DATA", state.fsm);
                assert!(state.fsm == FSM::AWAITING_CONFIG, "invalid_state_transition");
                trace_fmt!("[HuffmanDecoder] Received codes:");
                for (i, ()) in range(u32:0, SYMBOLS_N) {
                    if symbol_valid[i] {
                        trace_fmt!("[HuffmanDecoder]   {:#b} (len {}) -> {:#x}", symbol_code[i], symbol_code_len[i], i);
                    } else {};
                }(());
                FSM::READ_DATA
            } else {
                state.fsm
            };
            State {
                fsm: fsm,
                symbol_config_id: state.symbol_config_id + u5:1,
                symbol_valid: symbol_valid,
                symbol_code: symbol_code,
                symbol_code_len: symbol_code_len,
                ..state
            }
        } else { state };

        // receive data
        let (tok, data, data_valid) = recv_if_non_blocking(
            tok, data_r, (state.fsm == FSM::READ_DATA) && (state.data_len as u32 < H_DATA_W), zero!<Data>()
        );

        let state = if data_valid {
            trace_fmt!("[HuffmanDecoder] {} -> DECODE", state.fsm);
            assert!(state.fsm == FSM::READ_DATA, "invalid_state_transition");
            trace_fmt!("[HuffmanDecoder] Received data: {:#b} (len: {})", data.data, data.data_len);
            State {
                fsm: FSM::DECODE,
                data_len: state.data_len + data.data_len as uN[BUFF_W_LOG2],
                data: state.data | (data.data as uN[BUFF_W] << state.data_len),
                data_last: data.last,
                code_length: extend_buff_array(state.code_length, state.data_len as u32, data.code_length),
                ..state
            }
        } else {
            state
        };

        // decode data
        let state = if (
            state.fsm == FSM::DECODE &&
            state.data_len > uN[BUFF_W_LOG2]:0
        ) {
            // greedily take longest match, won't skip anything since no symbol is a prefix of another (Huffman property)
            let (literal, matched) = for(i, (literal, matched)) : (u32, (uN[common::SYMBOL_WIDTH], bool)) in range(u32:0, SYMBOLS_N) {
                let test_length = state.symbol_code_len[i];
                let data_mask = (!uN[hcommon::MAX_WEIGHT]:0) >> (hcommon::MAX_WEIGHT - test_length as u32);
                let data_masked = state.data as uN[hcommon::MAX_WEIGHT] & data_mask;
                if (
                    state.symbol_valid[i] && (data_masked == state.symbol_code[i]) && state.data_len >= test_length as uN[BUFF_W_LOG2]
                ) {
                    trace_fmt!("decoded {:#b} as {:#x} (length={}, data_length={})", data_masked, i, test_length, state.data_len);
                    (i as uN[common::SYMBOL_WIDTH], true)
                } else {
                    (literal, matched)
                }
            }((uN[common::SYMBOL_WIDTH]:0, false));
            let length = state.symbol_code_len[literal];

            // shift buffer
            if matched {
                State {
                    decoded_literals: update(state.decoded_literals, state.decoded_literals_len, literal),
                    decoded_literals_len: state.decoded_literals_len + u4:1,
                    data_len: state.data_len - length as uN[BUFF_W_LOG2],
                    data: state.data >> length,
                    code_length: shift_buff_array(state.code_length, state.code_length[0] as u32),
                    ..state
                }
            } else {
                // means we've got dangling bits from the next package
                State {
                    fsm: FSM::READ_DATA,
                    ..state
                }
            }

        } else if (state.fsm == FSM::DECODE && state.data_len > uN[BUFF_W_LOG2]:0) {
            trace_fmt!(
                "[HuffmanDecoder] ERROR: data_len is {} which is shorter than the code length {}",
                state.data_len,
                state.code_length[0]
            );
            assert!(state.data_len >= state.code_length[0] as uN[BUFF_W_LOG2], "invalid_data_or_code_length");
            state
        } else {
            state
        };

        // send literals
        let do_send_literals = (
            state.decoded_literals_len == u4:8 ||
            (state.decoded_literals_len > u4:0 && state.data_len == uN[BUFF_W_LOG2]:0)
        );

        let data = if do_send_literals {
            for (i, data): (u32, common::LitData) in range(u32:0, u32:8) {
                data | (state.decoded_literals[i] as common::LitData << (common::SYMBOL_WIDTH * i))
            }(zero!<common::LitData>())
        } else {
            zero!<common::LitData>()
        };

        let done = (state.data_len == uN[BUFF_W_LOG2]:0) && (state.fsm == FSM::DECODE);
        let decoded_literals = common::LiteralsDataWithSync{
            data: data,
            length: state.decoded_literals_len as common::LitLength,
            last: done && state.last_stream,
            id: state.id,
            literals_last: state.literals_last,
        };
        let tok = send_if(tok, decoded_literals_s, do_send_literals, decoded_literals);
        if (do_send_literals) {
           trace_fmt!("[HuffmanDecoder] Sent decoded literals: {:#x}", decoded_literals);
        } else {};

        let state = if do_send_literals {
            let fsm = if state.data_len == uN[BUFF_W_LOG2]:0 {
                if state.data_last {
                    trace_fmt!("[HuffmanDecoder] {} -> IDLE", state.fsm);
                    FSM::IDLE
                } else {
                    trace_fmt!("[HuffmanDecoder] {} -> READ_DATA", state.fsm);
                    FSM::READ_DATA
                }
            } else {
                trace_fmt!("[HuffmanDecoder] {} -> DECODE", state.fsm);
                FSM::DECODE
            };
            assert!(state.fsm == FSM::DECODE, "invalid_state_transition");
            State {
                fsm: fsm,
                decoded_literals_len: u4:0,
                decoded_literals: zero!<uN[common::SYMBOL_WIDTH][u32:8]>(),
                ..state
            }
        } else {
            state
        };

        let tok = send_if(tok, done_s, done, ());

        state
    }
}

type TestCodeLen = uN[hcommon::WEIGHT_LOG];
type TestCode = uN[hcommon::MAX_WEIGHT];

struct SymbolData {
    symbol_valid: bool,
    code_length: TestCodeLen,
    code: TestCode,
}

// helper function to improve readability of test data
fn generate_codes(data: SymbolData[8]) -> Codes {
    Codes {
        symbol_valid: [
            data[0].symbol_valid, data[1].symbol_valid, data[2].symbol_valid, data[3].symbol_valid,
            data[4].symbol_valid, data[5].symbol_valid, data[6].symbol_valid, data[7].symbol_valid,
        ],
        code_length: [
            data[0].code_length, data[1].code_length, data[2].code_length, data[3].code_length,
            data[4].code_length, data[5].code_length, data[6].code_length, data[7].code_length,
        ],
        code: [
            data[0].code, data[1].code, data[2].code, data[3].code,
            data[4].code, data[5].code, data[6].code, data[7].code,
        ],
    }
}

const TEST_START = HuffmanDecoderStart[3]:[
    HuffmanDecoderStart { new_config: true, id: u32:0, literals_last: false, last_stream: true },
    HuffmanDecoderStart { new_config: false, id: u32:1, literals_last: true, last_stream: true },
    HuffmanDecoderStart { new_config: true, id: u32:0, literals_last: false, last_stream: true },
];

// config #1
// 0b1      -> 0x06
// 0b100    -> 0x03
// 0b010    -> 0x00
// 0b110    -> 0x02
// 0b1000   -> 0x1B
// 0b000000 -> 0xB6
// 0b010000 -> 0xB5
// 0b100000 -> 0x0D
// 0b110000 -> 0xB2
//
// config #2
// 0b1         -> 0x47
// 0b001       -> 0x41
// 0b010       -> 0xD2
// 0b011       -> 0x8A
// 0b000001    -> 0x7A
// 0b000010    -> 0xDA
// 0b000011    -> 0x45
// 0b000100    -> 0xD3
// 0b000101    -> 0x89
// 0b000110    -> 0x8D
// 0b000111    -> 0xD1
// 0b000000001 -> 0xAC
// 0b000000010 -> 0x8F
// 0b000000011 -> 0xDB
// 0b000000100 -> 0xD4
// 0b000000101 -> 0xFE
// 0b000000110 -> 0xDE
// 0b000000111 -> 0xD7

const TEST_CODES = Codes[64]:[
    // config #1
    generate_codes([    // 0x00 - 0x07
        SymbolData { symbol_valid: true, code_length: TestCodeLen:3, code: TestCode:0b010 },
        zero!<SymbolData>(),
        SymbolData { symbol_valid: true, code_length: TestCodeLen:3, code: TestCode:0b110 },
        SymbolData { symbol_valid: true, code_length: TestCodeLen:3, code: TestCode:0b100 },
        zero!<SymbolData>(),
        zero!<SymbolData>(),
        SymbolData { symbol_valid: true, code_length: TestCodeLen:1, code: TestCode:0b1 },
        zero!<SymbolData>(),
    ]),
    zero!<Codes>(),     // 0x08 - 0x0F
    zero!<Codes>(),     // 0x10 - 0x17
    generate_codes([    // 0x18 - 0x1F
        zero!<SymbolData>(),
        zero!<SymbolData>(),
        zero!<SymbolData>(),
        SymbolData { symbol_valid: true, code_length: TestCodeLen:4, code: TestCode:0b1000 },
        zero!<SymbolData>(),
        zero!<SymbolData>(),
        zero!<SymbolData>(),
        zero!<SymbolData>(),
    ]),
    zero!<Codes>(),     // 0x20 - 0x27
    zero!<Codes>(),     // 0x28 - 0x2F
    zero!<Codes>(),     // 0x30 - 0x37
    zero!<Codes>(),     // 0x38 - 0x3F
    zero!<Codes>(),     // 0x40 - 0x47
    zero!<Codes>(),     // 0x48 - 0x4F
    zero!<Codes>(),     // 0x50 - 0x67
    zero!<Codes>(),     // 0x58 - 0x5F
    zero!<Codes>(),     // 0x60 - 0x67
    zero!<Codes>(),     // 0x68 - 0x6F
    zero!<Codes>(),     // 0x70 - 0x77
    zero!<Codes>(),     // 0x78 - 0x7F
    zero!<Codes>(),     // 0x80 - 0x87
    zero!<Codes>(),     // 0x88 - 0x8F
    zero!<Codes>(),     // 0x90 - 0x97
    zero!<Codes>(),     // 0x98 - 0x9F
    zero!<Codes>(),     // 0xA0 - 0xA7
    zero!<Codes>(),     // 0xA8 - 0xAF
    generate_codes([    // 0xB0 - 0xB7
        zero!<SymbolData>(),
        zero!<SymbolData>(),
        SymbolData { symbol_valid: true, code_length: TestCodeLen:6, code: TestCode:0b110000 },
        zero!<SymbolData>(),
        zero!<SymbolData>(),
        SymbolData { symbol_valid: true, code_length: TestCodeLen:6, code: TestCode:0b010000 },
        SymbolData { symbol_valid: true, code_length: TestCodeLen:6, code: TestCode:0b000000 },
        zero!<SymbolData>(),
    ]),
    zero!<Codes>(),     // 0xB8 - 0xBF
    zero!<Codes>(),     // 0xC0 - 0xC7
    zero!<Codes>(),     // 0xC8 - 0xCF
    zero!<Codes>(),     // 0xD0 - 0xD7
    zero!<Codes>(),     // 0xD8 - 0xDF
    zero!<Codes>(),     // 0xE0 - 0xE7
    zero!<Codes>(),     // 0xE8 - 0xEF
    zero!<Codes>(),     // 0xF0 - 0xF7
    zero!<Codes>(),     // 0xF8 - 0xFF
    // config #2
    zero!<Codes>(),     // 0x00 - 0x07
    zero!<Codes>(),     // 0x08 - 0x0F
    zero!<Codes>(),     // 0x10 - 0x17
    zero!<Codes>(),     // 0x18 - 0x1F
    zero!<Codes>(),     // 0x20 - 0x27
    zero!<Codes>(),     // 0x28 - 0x2F
    zero!<Codes>(),     // 0x30 - 0x37
    zero!<Codes>(),     // 0x38 - 0x3F
    generate_codes([    // 0x40 - 0x47
        zero!<SymbolData>(),
        SymbolData { symbol_valid: true, code_length: TestCodeLen:3, code: TestCode:0b100 },
        zero!<SymbolData>(),
        zero!<SymbolData>(),
        zero!<SymbolData>(),
        SymbolData { symbol_valid: true, code_length: TestCodeLen:6, code: TestCode:0b110000 },
        zero!<SymbolData>(),
        SymbolData { symbol_valid: true, code_length: TestCodeLen:1, code: TestCode:0b1 },
    ]),
    zero!<Codes>(),     // 0x48 - 0x4F
    zero!<Codes>(),     // 0x50 - 0x67
    zero!<Codes>(),     // 0x58 - 0x5F
    zero!<Codes>(),     // 0x60 - 0x67
    zero!<Codes>(),     // 0x68 - 0x6F
    zero!<Codes>(),     // 0x70 - 0x77
    generate_codes([    // 0x78 - 0x7F
        zero!<SymbolData>(),
        zero!<SymbolData>(),
        SymbolData { symbol_valid: true, code_length: TestCodeLen:6, code: TestCode:0b100000 },
        zero!<SymbolData>(),
        zero!<SymbolData>(),
        zero!<SymbolData>(),
        zero!<SymbolData>(),
        zero!<SymbolData>(),
    ]),
    zero!<Codes>(),     // 0x80 - 0x87
    generate_codes([    // 0x88 - 0x8F
        zero!<SymbolData>(),
        SymbolData { symbol_valid: true, code_length: TestCodeLen:6, code: TestCode:0b101000 },
        SymbolData { symbol_valid: true, code_length: TestCodeLen:3, code: TestCode:0b110 },
        zero!<SymbolData>(),
        zero!<SymbolData>(),
        SymbolData { symbol_valid: true, code_length: TestCodeLen:6, code: TestCode:0b110000 },
        zero!<SymbolData>(),
        SymbolData { symbol_valid: true, code_length: TestCodeLen:9, code: TestCode:0b0100000000 },
    ]),
    zero!<Codes>(),     // 0x90 - 0x97
    zero!<Codes>(),     // 0x98 - 0x9F
    zero!<Codes>(),     // 0xA0 - 0xA7
    generate_codes([    // 0xA8 - 0xAF
        zero!<SymbolData>(),
        zero!<SymbolData>(),
        zero!<SymbolData>(),
        zero!<SymbolData>(),
        SymbolData { symbol_valid: true, code_length: TestCodeLen:9, code: TestCode:0b100000000 },
        zero!<SymbolData>(),
        zero!<SymbolData>(),
        zero!<SymbolData>(),
    ]),
    zero!<Codes>(),     // 0xB0 - 0xB7
    zero!<Codes>(),     // 0xB8 - 0xBF
    zero!<Codes>(),     // 0xC0 - 0xC7
    zero!<Codes>(),     // 0xC8 - 0xCF
    generate_codes([    // 0xD0 - 0xD7
        zero!<SymbolData>(),
        SymbolData { symbol_valid: true, code_length: TestCodeLen:6, code: TestCode:0b111000 },
        SymbolData { symbol_valid: true, code_length: TestCodeLen:3, code: TestCode:0b010 },
        SymbolData { symbol_valid: true, code_length: TestCodeLen:6, code: TestCode:0b001000 },
        SymbolData { symbol_valid: true, code_length: TestCodeLen:9, code: TestCode:0b001000000 },
        zero!<SymbolData>(),
        zero!<SymbolData>(),
        SymbolData { symbol_valid: true, code_length: TestCodeLen:9, code: TestCode:0b111000000 },
    ]),
    generate_codes([    // 0xD8 - 0xDF
        zero!<SymbolData>(),
        zero!<SymbolData>(),
        SymbolData { symbol_valid: true, code_length: TestCodeLen:6, code: TestCode:0b010000 },
        SymbolData { symbol_valid: true, code_length: TestCodeLen:9, code: TestCode:0b110000000 },
        zero!<SymbolData>(),
        zero!<SymbolData>(),
        SymbolData { symbol_valid: true, code_length: TestCodeLen:9, code: TestCode:0b011000000 },
        zero!<SymbolData>(),
    ]),
    zero!<Codes>(),     // 0xE0 - 0xE7
    zero!<Codes>(),     // 0xE8 - 0xEF
    zero!<Codes>(),     // 0xF0 - 0xF7
    generate_codes([    // 0xF8 - 0xFF
        zero!<SymbolData>(),
        zero!<SymbolData>(),
        zero!<SymbolData>(),
        zero!<SymbolData>(),
        zero!<SymbolData>(),
        zero!<SymbolData>(),
        SymbolData { symbol_valid: true, code_length: TestCodeLen:9, code: TestCode:0b101000000 },
        zero!<SymbolData>(),
    ]),
];

const TEST_DATA = huffman_data_preprocessor::HuffmanDataPreprocessorData[3]:[
    huffman_data_preprocessor::HuffmanDataPreprocessorData {
        data: huffman_data_preprocessor::Data:0x32a0b682,
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
    huffman_data_preprocessor::HuffmanDataPreprocessorData {
        data: huffman_data_preprocessor::Data:0x32a0b682,
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
    huffman_data_preprocessor::HuffmanDataPreprocessorData {
        data: huffman_data_preprocessor::Data:0b1_010_100_110_1_010_100_100_001000_1_010_110_1_010_100000_010_101000_1_1_110_010_1,
        data_len: CodeLen:61,
        last: true,
        code_length: [
            CodeLen:1, CodeLen:3, CodeLen:1, CodeLen:3, CodeLen:3, CodeLen:1, CodeLen:1, CodeLen:1,
            CodeLen:1, CodeLen:6, CodeLen:3, CodeLen:3, CodeLen:1, CodeLen:3, CodeLen:1, CodeLen:3,
            CodeLen:1, CodeLen:9, CodeLen:6, CodeLen:6, CodeLen:6, CodeLen:3, CodeLen:3, CodeLen:1,
            CodeLen:3, CodeLen:1, CodeLen:3, CodeLen:1, CodeLen:3, CodeLen:1, CodeLen:1, CodeLen:3,
            CodeLen:1, CodeLen:3, CodeLen:1, CodeLen:6, CodeLen:3, CodeLen:3, CodeLen:1, CodeLen:6,
            CodeLen:6, CodeLen:3, CodeLen:3, CodeLen:1, CodeLen:3, CodeLen:3, CodeLen:1, CodeLen:3,
            CodeLen:1, CodeLen:3, CodeLen:1, CodeLen:3, CodeLen:1, CodeLen:1, CodeLen:3, CodeLen:3,
            CodeLen:1, CodeLen:3, CodeLen:1, CodeLen:3, CodeLen:1, CodeLen:0, CodeLen:0, CodeLen:0,
            CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0,
            CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0,
            CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0,
            CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0, CodeLen:0,
        ],
    },
];

const TEST_LITERALS = common::LiteralsDataWithSync[7]:[
    common::LiteralsDataWithSync {
        data: common::LitData:0x06B5_0002_0606_B500,
        length: common::LitLength:8,
        last: false,
        id: u32:0,
        literals_last: false,
    },
    common::LiteralsDataWithSync {
        data: common::LitData:0x0200,
        length: common::LitLength:2,
        last: true,
        id: u32:0,
        literals_last: false,
    },
    common::LiteralsDataWithSync {
        data: common::LitData:0x06B5_0002_0606_B500,
        length: common::LitLength:8,
        last: false,
        id: u32:1,
        literals_last: true,
    },
    common::LiteralsDataWithSync {
        data: common::LitData:0x0200,
        length: common::LitLength:2,
        last: true,
        id: u32:1,
        literals_last: true,
    },
    common::LiteralsDataWithSync {
        data: common::LitData:0x7AD2_8947_478A_D247,
        length: common::LitLength:8,
        last: false,
        id: u32:0,
        literals_last: false,
    },
    common::LiteralsDataWithSync {
        data: common::LitData:0x4141_D347_D28A_47D2,
        length: common::LitLength:8,
        last: false,
        id: u32:0,
        literals_last: false,
    },
    common::LiteralsDataWithSync {
        data: common::LitData:0x47D2_418A_47D2,
        length: common::LitLength:6,
        last: true,
        id: u32:0,
        literals_last: false,
    },
];

#[test_proc]
proc HuffmanDecoder_test {
    type Start = HuffmanDecoderStart;
    type Data = huffman_data_preprocessor::HuffmanDataPreprocessorData;

    terminator_s: chan<bool> out;

    start_s: chan<Start> out;
    codes_s: chan<Codes> out;
    data_s: chan<Data> out;

    done_r: chan<()> in;
    decoded_literals_r: chan<common::LiteralsDataWithSync> in;

    config (terminator_s: chan<bool> out) {
        let (start_s, start_r) = chan<Start>("start");
        let (codes_s, codes_r) = chan<Codes>("codes");
        let (data_s, data_r) = chan<Data>("data");
        let (done_s, done_r) = chan<()>("done");
        let (decoded_literals_s, decoded_literals_r) = chan<common::LiteralsDataWithSync>("decoded_literals");

        spawn HuffmanDecoder(
            start_r, codes_r, data_r,
            done_s, decoded_literals_s,
        );
        (
            terminator_s,
            start_s,
            codes_s,
            data_s,
            done_r,
            decoded_literals_r,
        )
    }

    init { }

    next (state: ()) {
        let tok = join();

        let (tok, _) = for ((i, start), (tok, codes_idx)): ((u32, Start), (token, u32)) in enumerate(TEST_START) {
            // send start
            let tok = send(tok, start_s, start);
            trace_fmt!("Sent #{} start {:#x}", i + u32:1, start);

            // send codes if required
            let (tok, codes_idx) = if start.new_config {
                for (_, (tok, codes_idx)): (u32, (token, u32)) in range(u32:0, SYMBOLS_N / hcommon::PARALLEL_ACCESS_WIDTH) {
                    let tok = send(tok, codes_s, TEST_CODES[codes_idx]);
                    trace_fmt!("Send #{} codes {:#x}", codes_idx + u32:1, TEST_CODES[codes_idx]);
                    (tok, codes_idx + u32:1)
                }((tok, codes_idx))
            } else {
                (tok, codes_idx)
            };

            // send data
            let tok = send(tok, data_s, TEST_DATA[i]);
            trace_fmt!("Sent #{} data {:#x}", i + u32:1, TEST_DATA[i]);

            (tok, codes_idx)
        }((tok, u32:0));

        let tok = for ((i, expected_literals), tok): ((u32, common::LiteralsDataWithSync), token) in enumerate(TEST_LITERALS) {
            // receive literals
            let (tok, literals) = recv(tok, decoded_literals_r);
            trace_fmt!("Received #{} literals {:#x}", i + u32:1, literals);

            assert_eq(expected_literals, literals);

            // receive done
            let tok = if expected_literals.last {
                let (tok, _) = recv(tok, done_r);
                tok
            } else {
                tok
            };

            tok
        }(tok);

        send(tok, terminator_s, true);
    }

}
