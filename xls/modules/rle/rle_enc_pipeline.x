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

// This file implements a parametric RLE encoding pipeline

import std
import xls.modules.rle.rle_interface as rle_interface
import xls.modules.rle.rle_enc as rle_enc
import xls.modules.rle.muxer as muxer

type PreData = rle_interface::PreData;
type EncData = rle_interface::EncData;
type MixData = rle_interface::MixData;

// RLE encoder with simple (symbol, counter) output stream
pub proc RLEEncPipeline<SYMBOL_WORD_WIDTH:u32> {
    init {()}

    config (
        symbol_r: chan<PreData<SYMBOL_WORD_WIDTH>>  in,
        word_s:   chan<MixData<SYMBOL_WORD_WIDTH>> out,
    ) {
        let (data_s, data_r) = chan<EncData<SYMBOL_WORD_WIDTH, SYMBOL_WORD_WIDTH>, 1>;
        spawn rle_enc::RLEEnc<SYMBOL_WORD_WIDTH, SYMBOL_WORD_WIDTH> (symbol_r, data_s);
        spawn muxer::StreamMuxerSimple<SYMBOL_WORD_WIDTH> (data_r, word_s);
        ()
    }

    next (tok: token, state: ()) {()}
}

// RLE encoding pipeline specialization for the codegen
proc RLEEncPipeline8 {
    init {()}

    config (
        symbol_r: chan<PreData<8>>  in,
        word_s:   chan<MixData<8>> out,
    ) {
        spawn RLEEncPipeline<u32:8>(symbol_r, word_s);
        ()
    }

    next (tok: token, state: ()) {()}
}

// Tests

const TEST_WORD_WIDTH = u32:32;

type TestRLEInData  = PreData<TEST_WORD_WIDTH>;
type TestRLEOutData = MixData<TEST_WORD_WIDTH>;

const TEST_RECV_WORD_LOOKUP = TestRLEOutData[18]:[
    TestRLEOutData{word: bits[TEST_WORD_WIDTH]:0xA, last: bool:false},
    TestRLEOutData{word: bits[TEST_WORD_WIDTH]:0x3, last: bool:false},
    TestRLEOutData{word: bits[TEST_WORD_WIDTH]:0xB, last: bool:false},
    TestRLEOutData{word: bits[TEST_WORD_WIDTH]:0x1, last: bool:true},
    TestRLEOutData{word: bits[TEST_WORD_WIDTH]:0xB, last: bool:false},
    TestRLEOutData{word: bits[TEST_WORD_WIDTH]:0x2, last: bool:false},
    TestRLEOutData{word: bits[TEST_WORD_WIDTH]:0x1, last: bool:false},
    TestRLEOutData{word: bits[TEST_WORD_WIDTH]:0x1, last: bool:false},
    TestRLEOutData{word: bits[TEST_WORD_WIDTH]:0xC, last: bool:false},
    TestRLEOutData{word: bits[TEST_WORD_WIDTH]:0x6, last: bool:false},
    TestRLEOutData{word: bits[TEST_WORD_WIDTH]:0x3, last: bool:false},
    TestRLEOutData{word: bits[TEST_WORD_WIDTH]:0x3, last: bool:false},
    TestRLEOutData{word: bits[TEST_WORD_WIDTH]:0x2, last: bool:false},
    TestRLEOutData{word: bits[TEST_WORD_WIDTH]:0x2, last: bool:true},
    TestRLEOutData{word: bits[TEST_WORD_WIDTH]:0x1, last: bool:false},
    TestRLEOutData{word: bits[TEST_WORD_WIDTH]:0x1, last: bool:true},
    TestRLEOutData{word: bits[TEST_WORD_WIDTH]:0x2, last: bool:false},
    TestRLEOutData{word: bits[TEST_WORD_WIDTH]:0x1, last: bool:true},
];

const TEST_RECV_WORD_LEN = array_size(TEST_RECV_WORD_LOOKUP);

type StreamMuxerSimpleTesterState = muxer::StreamMuxerSimpleTesterState;

// main proc used to test the RLE
#[test_proc]
proc RLEEncTester {
    terminator: chan<bool> out;         // test termination request
    word_r: chan<TestRLEOutData> in;    // data read from the tested MixSimple

    init {(zero!<StreamMuxerSimpleTesterState>())}

    config(terminator: chan<bool> out) {
        let (symbol_s, symbol_r) = chan<TestRLEInData>;
        let (word_s, word_r) = chan<TestRLEOutData>;

        spawn rle_enc::RLEEncTestSender(symbol_s);
        spawn RLEEncPipeline<TEST_WORD_WIDTH>(symbol_r, word_s);
        (terminator, word_r)
    }

    next(tok: token, state: StreamMuxerSimpleTesterState) {
        let (recv_tok, word) = recv(tok, word_r);

        let recv_count = state.recv_count + u32:1;

        let _ = trace_fmt!(
            "Received {} transactions, word: 0x{:x}, total of {} words",
            recv_count, word.word, recv_count
        );

        // checks for expected values

        let exp_word = TEST_RECV_WORD_LOOKUP[state.recv_count];
        let _ = assert_eq(exp_word, word);

        // check the total count after the last expected receive matches
        // with the expected value

        let exp_cnt_reached = recv_count == TEST_RECV_WORD_LEN;

        let _ = if exp_cnt_reached {
            let _ = send(recv_tok, terminator, true);
        } else {()};

        // state for the next iteration

        StreamMuxerSimpleTesterState {
            recv_count: recv_count,
        }
    }
}
