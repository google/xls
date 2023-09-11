// Copyright 2023 The XLS Authors
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

import xls.examples.ram as ram
import xls.modules.dbe.common as dbe
import xls.modules.dbe.common_test as test
import xls.modules.dbe.lz4_decoder as lz4_decoder

type Mark = dbe::Mark;
type TokenKind = dbe::TokenKind;
type Token = dbe::Token;
type PlainData = dbe::PlainData;
type Lz4Token = dbe::Lz4Token;
type Lz4Data = dbe::Lz4Data;
type Lz4DecoderRamHbReadReq = lz4_decoder::Lz4DecoderRamHbReadReq;
type Lz4DecoderRamHbReadResp = lz4_decoder::Lz4DecoderRamHbReadResp;
type Lz4DecoderRamHbWriteReq = lz4_decoder::Lz4DecoderRamHbWriteReq;

const SYMBOL_WIDTH = dbe::LZ4_SYMBOL_WIDTH;
const MATCH_OFFSET_WIDTH = dbe::LZ4_MATCH_OFFSET_WIDTH;
const MATCH_LENGTH_WIDTH = dbe::LZ4_MATCH_LENGTH_WIDTH;

/// Version of decoder that uses RamModel, intended for tests only
pub proc decoder_model {
    init{()}

    config (
        encoded_data: chan<Token<
                SYMBOL_WIDTH,
                MATCH_OFFSET_WIDTH,
                MATCH_LENGTH_WIDTH
            >> in,
        plain_data: chan<PlainData<SYMBOL_WIDTH>> out,
    ) {
        let (hb_wr_req_s, hb_wr_req_r) = chan<Lz4DecoderRamHbWriteReq>;
        let (hb_wr_comp_s, hb_wr_comp_r) = chan<ram::WriteResp>;
        let (hb_rd_req_s, hb_rd_req_r) = chan<Lz4DecoderRamHbReadReq>;
        let (hb_rd_resp_s, hb_rd_resp_r) = chan<Lz4DecoderRamHbReadResp>;
        
        spawn ram::RamModel<
                SYMBOL_WIDTH,
                {u32:1 << MATCH_OFFSET_WIDTH},
                SYMBOL_WIDTH
            > (hb_rd_req_r, hb_rd_resp_s, hb_wr_req_r, hb_wr_comp_s);

        spawn lz4_decoder::decoder(
                encoded_data, plain_data,
                hb_rd_req_s, hb_rd_resp_r, hb_wr_req_s, hb_wr_comp_r,
            );
    }

    next (tok: token, state: ()) {
    }
}

// Reference input
const TEST_SIMPLE_INPUT_LEN = u32:32;
const TEST_SIMPLE_INPUT = Lz4Token[TEST_SIMPLE_INPUT_LEN]: [
    Token{kind: TokenKind::UNMATCHED_SYMBOL, symbol: u8:12, match_offset: u16:0, match_length: u16:0, mark: Mark::NONE},
    Token{kind: TokenKind::UNMATCHED_SYMBOL, symbol: u8: 1, match_offset: u16:0, match_length: u16:0, mark: Mark::NONE},
    Token{kind: TokenKind::UNMATCHED_SYMBOL, symbol: u8:15, match_offset: u16:0, match_length: u16:0, mark: Mark::NONE},
    Token{kind: TokenKind::UNMATCHED_SYMBOL, symbol: u8: 9, match_offset: u16:0, match_length: u16:0, mark: Mark::NONE},
    Token{kind: TokenKind::UNMATCHED_SYMBOL, symbol: u8:11, match_offset: u16:0, match_length: u16:0, mark: Mark::NONE},
    Token{kind: TokenKind::MATCH, symbol: u8:0, match_offset: u16: 1, match_length: u16: 1, mark: Mark::NONE},
    Token{kind: TokenKind::MATCH, symbol: u8:0, match_offset: u16: 0, match_length: u16: 2, mark: Mark::NONE},
    Token{kind: TokenKind::MATCH, symbol: u8:0, match_offset: u16: 4, match_length: u16: 0, mark: Mark::NONE},
    Token{kind: TokenKind::UNMATCHED_SYMBOL, symbol: u8:15, match_offset: u16:0, match_length: u16:0, mark: Mark::NONE},
    Token{kind: TokenKind::MATCH, symbol: u8:0, match_offset: u16: 1, match_length: u16: 0, mark: Mark::NONE},
    Token{kind: TokenKind::MATCH, symbol: u8:0, match_offset: u16: 2, match_length: u16:13, mark: Mark::NONE},
    Token{kind: TokenKind::UNMATCHED_SYMBOL, symbol: u8: 1, match_offset: u16:0, match_length: u16:0, mark: Mark::NONE},
    Token{kind: TokenKind::MATCH, symbol: u8:0, match_offset: u16: 5, match_length: u16: 0, mark: Mark::NONE},
    Token{kind: TokenKind::MATCH, symbol: u8:0, match_offset: u16: 5, match_length: u16: 2, mark: Mark::NONE},
    Token{kind: TokenKind::UNMATCHED_SYMBOL, symbol: u8: 7, match_offset: u16:0, match_length: u16:0, mark: Mark::NONE},
    Token{kind: TokenKind::UNMATCHED_SYMBOL, symbol: u8: 2, match_offset: u16:0, match_length: u16:0, mark: Mark::NONE},
    Token{kind: TokenKind::MATCH, symbol: u8:0, match_offset: u16: 0, match_length: u16:11, mark: Mark::NONE},
    Token{kind: TokenKind::MATCH, symbol: u8:0, match_offset: u16: 1, match_length: u16: 6, mark: Mark::NONE},
    Token{kind: TokenKind::MATCH, symbol: u8:0, match_offset: u16: 5, match_length: u16:15, mark: Mark::NONE},
    Token{kind: TokenKind::UNMATCHED_SYMBOL, symbol: u8: 3, match_offset: u16:0, match_length: u16:0, mark: Mark::NONE},
    Token{kind: TokenKind::UNMATCHED_SYMBOL, symbol: u8: 9, match_offset: u16:0, match_length: u16:0, mark: Mark::NONE},
    Token{kind: TokenKind::MATCH, symbol: u8:0, match_offset: u16: 5, match_length: u16: 1, mark: Mark::NONE},
    Token{kind: TokenKind::MATCH, symbol: u8:0, match_offset: u16: 3, match_length: u16:13, mark: Mark::NONE},
    Token{kind: TokenKind::UNMATCHED_SYMBOL, symbol: u8:14, match_offset: u16:0, match_length: u16:0, mark: Mark::NONE},
    Token{kind: TokenKind::MATCH, symbol: u8:0, match_offset: u16: 1, match_length: u16: 2, mark: Mark::NONE},
    Token{kind: TokenKind::MATCH, symbol: u8:0, match_offset: u16: 0, match_length: u16: 0, mark: Mark::NONE},
    Token{kind: TokenKind::MATCH, symbol: u8:0, match_offset: u16: 7, match_length: u16: 0, mark: Mark::NONE},
    Token{kind: TokenKind::UNMATCHED_SYMBOL, symbol: u8:15, match_offset: u16:0, match_length: u16:0, mark: Mark::NONE},
    Token{kind: TokenKind::MATCH, symbol: u8:0, match_offset: u16: 4, match_length: u16: 0, mark: Mark::NONE},
    Token{kind: TokenKind::MATCH, symbol: u8:0, match_offset: u16: 5, match_length: u16:11, mark: Mark::NONE},
    Token{kind: TokenKind::UNMATCHED_SYMBOL, symbol: u8: 8, match_offset: u16:0, match_length: u16:0, mark: Mark::NONE},
    Token{kind: TokenKind::MARKER, symbol: u8:0, match_offset: u16:0, match_length: u16:0, mark: Mark::END}
];
// Reference output
const TEST_SIMPLE_OUTPUT_LEN = u32:109;
const TEST_SIMPLE_OUTPUT = Lz4Data[TEST_SIMPLE_OUTPUT_LEN]: [
    PlainData{is_marker: false, data: u8:12, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:1, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:15, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:9, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:11, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:9, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:11, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:11, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:11, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:11, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:9, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:15, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:9, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:9, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:15, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:9, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:9, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:15, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:9, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:9, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:15, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:9, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:9, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:15, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:9, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:9, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:15, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:1, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:9, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:15, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:9, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:9, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:7, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:3, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:9, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:3, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:9, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:3, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:9, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:3, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:9, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:3, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:9, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:14, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:9, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:14, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:9, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:9, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:15, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:14, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:14, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:9, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:9, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:15, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:14, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:14, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:9, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:9, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:15, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:14, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:8, mark: Mark::NONE},
    PlainData{is_marker: true, data: u8:0, mark: Mark::END}
];

#[test_proc]
proc test_simple {
    term: chan<bool> out;
    recv_last_r: chan<bool> in;

    init {()}

    config(term: chan<bool> out) {        
        // tokens from sender and to decoder
        let (send_toks_s, send_toks_r) = chan<Lz4Token>;
        // symbols & last indicator from receiver
        let (recv_data_s, recv_data_r) = chan<Lz4Data>;
        let (recv_last_s, recv_last_r) = chan<bool>;


        spawn test::token_sender<
            TEST_SIMPLE_INPUT_LEN, SYMBOL_WIDTH, MATCH_OFFSET_WIDTH,
            MATCH_LENGTH_WIDTH
        >(TEST_SIMPLE_INPUT, send_toks_s);
        spawn decoder_model(send_toks_r, recv_data_s);
        spawn test::data_validator<TEST_SIMPLE_OUTPUT_LEN, SYMBOL_WIDTH>(
            TEST_SIMPLE_OUTPUT, recv_data_r, recv_last_s);

        (term, recv_last_r)
    }

    next (tok: token, state: ()) {
        let (tok, recv_term) = recv(tok, recv_last_r);
        send(tok, term, recv_term);
    }
}
