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

// This file implements a parametric RLE decoder processing pipeline

import std
import xls.modules.rle.rle_interface as rle_interface
import xls.modules.rle.rle_dec as rle_dec
import xls.modules.rle.demuxer as demuxer

type PreData = rle_interface::PreData;
type EncData = rle_interface::EncData;
type MixData = rle_interface::MixData;

// RLE decodeing pipeline
pub proc RLEDecPipeline<SYMBOL_WORD_WIDTH:u32> {
    init {()}

    config (
        word_r:   chan<MixData<SYMBOL_WORD_WIDTH>> in,
        symbol_s: chan<PreData<SYMBOL_WORD_WIDTH>> out,
    ) {
        let (data_s, data_r) = chan<EncData<SYMBOL_WORD_WIDTH, SYMBOL_WORD_WIDTH>, 1>;
        spawn demuxer::StreamDemuxerSimple<SYMBOL_WORD_WIDTH> (word_r, data_s);
        spawn rle_dec::RLEDec<SYMBOL_WORD_WIDTH, SYMBOL_WORD_WIDTH> (data_r, symbol_s);
        ()
    }

    next (tok: token, state: ()) {()}
}

// RLE decoding pipeline specialization for the codegen
proc RLEDecPipeline8 {
    init {()}

    config (
        word_r:   chan<MixData<8>> in,
        symbol_s: chan<PreData<8>> out,
    ) {
        spawn RLEDecPipeline<u32:8>(word_r, symbol_s);
        ()
    }

    next (tok: token, state: ()) {()}
}

// Tests

const TEST_WORD_WIDTH = u32:32;

type TestRLEInData  = MixData<TEST_WORD_WIDTH>;
type TestRLEOutData = PreData<TEST_WORD_WIDTH>;

const TEST_RECV_SYMBOL_LOOKUP = TestRLEOutData[20]:[
    TestRLEOutData{symbol: bits[TEST_WORD_WIDTH]:0xA, last: bool:false},
    TestRLEOutData{symbol: bits[TEST_WORD_WIDTH]:0xA, last: bool:false},
    TestRLEOutData{symbol: bits[TEST_WORD_WIDTH]:0xA, last: bool:false},
    TestRLEOutData{symbol: bits[TEST_WORD_WIDTH]:0xB, last: bool:false},
    TestRLEOutData{symbol: bits[TEST_WORD_WIDTH]:0xB, last: bool:false},
    TestRLEOutData{symbol: bits[TEST_WORD_WIDTH]:0x1, last: bool:false},
    TestRLEOutData{symbol: bits[TEST_WORD_WIDTH]:0xC, last: bool:false},
    TestRLEOutData{symbol: bits[TEST_WORD_WIDTH]:0xC, last: bool:false},
    TestRLEOutData{symbol: bits[TEST_WORD_WIDTH]:0xC, last: bool:true},
    TestRLEOutData{symbol: bits[TEST_WORD_WIDTH]:0xC, last: bool:false},
    TestRLEOutData{symbol: bits[TEST_WORD_WIDTH]:0xC, last: bool:false},
    TestRLEOutData{symbol: bits[TEST_WORD_WIDTH]:0xC, last: bool:false},
    TestRLEOutData{symbol: bits[TEST_WORD_WIDTH]:0x3, last: bool:false},
    TestRLEOutData{symbol: bits[TEST_WORD_WIDTH]:0x3, last: bool:false},
    TestRLEOutData{symbol: bits[TEST_WORD_WIDTH]:0x3, last: bool:true},
    TestRLEOutData{symbol: bits[TEST_WORD_WIDTH]:0x2, last: bool:false},
    TestRLEOutData{symbol: bits[TEST_WORD_WIDTH]:0x2, last: bool:true},
    TestRLEOutData{symbol: bits[TEST_WORD_WIDTH]:0x1, last: bool:false},
    TestRLEOutData{symbol: bits[TEST_WORD_WIDTH]:0x2, last: bool:false},
    TestRLEOutData{symbol: bits[TEST_WORD_WIDTH]:0x3, last: bool:true},
];

const TEST_RECV_SYMBOL_LEN = array_size(TEST_RECV_SYMBOL_LOOKUP);

type RLEDecTesterState = rle_dec::RLEDecTesterState;

// main proc used to test the RLE
#[test_proc]
proc RLEDecTester {
    terminator: chan<bool> out;        // test termination request
    symbol_r: chan<TestRLEOutData> in; // data read from the tested MixSimple

    init {(zero!<RLEDecTesterState>())}

    config(terminator: chan<bool> out) {
        let (word_s, word_r) = chan<TestRLEInData>;
        let (symbol_s, symbol_r) = chan<TestRLEOutData>;

        spawn demuxer::StreamDemuxerTestSender(word_s);
        spawn RLEDecPipeline<TEST_WORD_WIDTH>(word_r, symbol_s);
        (terminator, symbol_r)
    }

    next(tok: token, state: RLEDecTesterState) {
        let (recv_tok, symbol) = recv(tok, symbol_r);

        let total_count = state.total_count + u32:1;

        let _ = trace_fmt!(
            "Received {} transactions, symbol: 0x{:x}, last: {}",
            total_count, symbol.symbol, symbol.last
        );

        // checks for expected values

        let exp_symbol = TEST_RECV_SYMBOL_LOOKUP[state.total_count];
        let _ = assert_eq(exp_symbol, symbol);

        // check the total count after the last expected receive matches
        // with the expected value

        let exp_cnt_reached = total_count == TEST_RECV_SYMBOL_LEN;

        let _ = if exp_cnt_reached {
            let _ = send(recv_tok, terminator, true);
        } else {()};

        // state for the next iteration

        RLEDecTesterState {
            total_count: total_count,
        }
    }
}
