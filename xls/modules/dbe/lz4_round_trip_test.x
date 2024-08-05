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
import xls.modules.dbe.lz4_encoder_test as lz4_encoder_test
import xls.modules.dbe.lz4_decoder as lz4_decoder
import xls.modules.dbe.lz4_decoder_test as lz4_decoder_test

type Mark = dbe::Mark;
type TokenKind = dbe::TokenKind;
type Token = dbe::Token;
type PlainData = dbe::PlainData;
type Lz4Token = dbe::Lz4Token;
type Lz4Data = dbe::Lz4Data;

const TEST_DATA_LEN = u32:22;
const TEST_DATA = Lz4Data[TEST_DATA_LEN]: [
    PlainData{is_marker: false, data: u8:1, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:3, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:4, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:5, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:6, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:7, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:1, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:1, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:1, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:3, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:4, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:5, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:6, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:7, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:7, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:7, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:7, mark: Mark::NONE},
    PlainData{is_marker: true, data: u8:0, mark: Mark::END},
];

/// Round-trip test
#[test_proc]
proc test_encoder_8k_round_trip {
    term: chan<bool> out;
    recv_term_r: chan<bool> in;

    init {()}

    config(term: chan<bool> out) {
        let (send_data_s, send_data_r) = chan<Lz4Data>;
        let (enc_toks_s, enc_toks_r) = chan<Lz4Token>;
        let (recv_data_s, recv_data_r) = chan<Lz4Data>;
        let (recv_term_s, recv_term_r) = chan<bool>;

        spawn test::data_sender<TEST_DATA_LEN, dbe::LZ4_SYMBOL_WIDTH>
            (TEST_DATA, send_data_s);
        spawn lz4_encoder_test::encoder_8k_model(
            send_data_r, enc_toks_s,
        );
        spawn lz4_decoder_test::decoder_model(
            enc_toks_r, recv_data_s,
        );
        spawn test::data_validator<TEST_DATA_LEN, dbe::LZ4_SYMBOL_WIDTH>
            (TEST_DATA, recv_data_r, recv_term_s);

        (term, recv_term_r)
    }

    next (tok: token, state: ()) {
        let (tok, recv_term) = recv(tok, recv_term_r);
        send(tok, term, recv_term);
    }
}
