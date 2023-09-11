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
import xls.modules.dbe.lz4_encoder as lz4_encoder

type Mark = dbe::Mark;
type TokenKind = dbe::TokenKind;
type Token = dbe::Token;
type PlainData = dbe::PlainData;
type Lz4Token = dbe::Lz4Token;
type Lz4Data = dbe::Lz4Data;
type Lz4EncoderRamHbReadReq = lz4_encoder::Lz4EncoderRamHbReadReq;
type Lz4EncoderRamHbReadResp = lz4_encoder::Lz4EncoderRamHbReadResp;
type Lz4EncoderRamHbWriteReq = lz4_encoder::Lz4EncoderRamHbWriteReq;
type Lz4Encoder8kRamHtReadReq = lz4_encoder::Lz4Encoder8kRamHtReadReq;
type Lz4Encoder8kRamHtReadResp = lz4_encoder::Lz4Encoder8kRamHtReadResp;
type Lz4Encoder8kRamHtWriteReq = lz4_encoder::Lz4Encoder8kRamHtWriteReq;

const SYMBOL_WIDTH = dbe::LZ4_SYMBOL_WIDTH;
const MATCH_OFFSET_WIDTH = dbe::LZ4_MATCH_OFFSET_WIDTH;
const MATCH_LENGTH_WIDTH = dbe::LZ4_MATCH_LENGTH_WIDTH;
const HASH_SYMBOLS = dbe::LZ4_HASH_SYMBOLS;
const HASH_WIDTH = dbe::LZ4_HASH_WIDTH_8K;

/// Version of encoder that uses RamModel, intended for tests only
pub proc encoder_8k_model {
    init{()}

    config (
        plain_data: chan<Lz4Data> in,
        encoded_data: chan<Lz4Token> out,
    ) {
        let (hb_rd_req_s, hb_rd_req_r) = chan<Lz4EncoderRamHbReadReq>;
        let (hb_rd_resp_s, hb_rd_resp_r) = chan<Lz4EncoderRamHbReadResp>;
        let (hb_wr_req_s, hb_wr_req_r) = chan<Lz4EncoderRamHbWriteReq>;
        let (hb_wr_comp_s, hb_wr_comp_r) = chan<ram::WriteResp>;
        let (ht_rd_req_s, ht_rd_req_r) = chan<Lz4Encoder8kRamHtReadReq>;
        let (ht_rd_resp_s, ht_rd_resp_r) = chan<Lz4Encoder8kRamHtReadResp>;
        let (ht_wr_req_s, ht_wr_req_r) = chan<Lz4Encoder8kRamHtWriteReq>;
        let (ht_wr_comp_s, ht_wr_comp_r) = chan<ram::WriteResp>;

        spawn ram::RamModel<
            SYMBOL_WIDTH, {u32:1<<MATCH_OFFSET_WIDTH}, SYMBOL_WIDTH,
            ram::SimultaneousReadWriteBehavior::READ_BEFORE_WRITE,
            true
        >(hb_rd_req_r, hb_rd_resp_s, hb_wr_req_r, hb_wr_comp_s);
        spawn ram::RamModel<
            MATCH_OFFSET_WIDTH, {u32:1<<HASH_WIDTH}, MATCH_OFFSET_WIDTH,
            ram::SimultaneousReadWriteBehavior::READ_BEFORE_WRITE,
            true
        >(ht_rd_req_r, ht_rd_resp_s, ht_wr_req_r, ht_wr_comp_s);

        spawn lz4_encoder::encoder<HASH_WIDTH, false> (
            plain_data, encoded_data,
            hb_rd_req_s, hb_rd_resp_r, hb_wr_req_s, hb_wr_comp_r,
            ht_rd_req_s, ht_rd_resp_r, ht_wr_req_s, ht_wr_comp_r,
        );
    }

    next (tok: token, state: ()) {
    }
}

const TEST_DATA_LEN = u32:10;
const TEST_DATA = Lz4Data[TEST_DATA_LEN]: [
    // Check basic LT and MATCH generation
    PlainData{is_marker: false, data: u8:1, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:1, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:1, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    // Final token sequence
    PlainData{is_marker: false, data: u8:7, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:7, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:7, mark: Mark::NONE},
    PlainData{is_marker: true, data: u8:0, mark: Mark::END},
];

const TEST_TOKENS_LEN = u32:11;
const TEST_TOKENS = Lz4Token[TEST_TOKENS_LEN]: [
    Token{kind: TokenKind::UNMATCHED_SYMBOL, symbol: u8:1, match_offset: u16:0, match_length: u16:0, mark: Mark::NONE},
    Token{kind: TokenKind::UNMATCHED_SYMBOL, symbol: u8:2, match_offset: u16:0, match_length: u16:0, mark: Mark::NONE},
    Token{kind: TokenKind::MATCHED_SYMBOL, symbol: u8:1, match_offset: u16:0, match_length: u16:0, mark: Mark::NONE},
    Token{kind: TokenKind::MATCHED_SYMBOL, symbol: u8:2, match_offset: u16:0, match_length: u16:0, mark: Mark::NONE},
    Token{kind: TokenKind::MATCHED_SYMBOL, symbol: u8:1, match_offset: u16:0, match_length: u16:0, mark: Mark::NONE},
    Token{kind: TokenKind::MATCHED_SYMBOL, symbol: u8:2, match_offset: u16:0, match_length: u16:0, mark: Mark::NONE},
    Token{kind: TokenKind::MATCH, symbol: u8:0, match_offset: u16:1, match_length: u16:3, mark: Mark::NONE},
    // Final token sequence
    Token{kind: TokenKind::UNMATCHED_SYMBOL, symbol: u8:7, match_offset: u16:0, match_length: u16:0, mark: Mark::NONE},
    Token{kind: TokenKind::UNMATCHED_SYMBOL, symbol: u8:7, match_offset: u16:0, match_length: u16:0, mark: Mark::NONE},
    Token{kind: TokenKind::UNMATCHED_SYMBOL, symbol: u8:7, match_offset: u16:0, match_length: u16:0, mark: Mark::NONE},
    Token{kind: TokenKind::MARKER, symbol: u8:0, match_offset: u16:0, match_length: u16:0, mark: Mark::END},
];

#[test_proc]
proc test_encoder_8k_simple {
    term: chan<bool> out;
    recv_term_r: chan<bool> in;

    init {()}

    config(term: chan<bool> out) {
        let (send_data_s, send_data_r) = chan<Lz4Data>;
        let (enc_toks_s, enc_toks_r) = chan<Lz4Token>;
        let (recv_term_s, recv_term_r) = chan<bool>;

        spawn test::data_sender<TEST_DATA_LEN, SYMBOL_WIDTH>
            (TEST_DATA, send_data_s);
        spawn encoder_8k_model(
            send_data_r, enc_toks_s,
        );
        spawn test::token_validator<
            TEST_TOKENS_LEN, SYMBOL_WIDTH, MATCH_OFFSET_WIDTH,
            MATCH_LENGTH_WIDTH
        >(TEST_TOKENS, enc_toks_r, recv_term_s);

        (term, recv_term_r)
    }

    next (tok: token, state: ()) {
        let (tok, recv_term) = recv(tok, recv_term_r);
        send(tok, term, recv_term);
    }
}
