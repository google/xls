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

// This file implements a parametric RLE encoder
//
// The encoder uses Run Length Encoding (RLE) to compress the input stream
// with symbol (data to compress). The output contains packets of data with
// the symbol and the number of their occurrences in the input stream.

import std
import xls.modules.rle.rle_interface as rle_interface

type EncData = rle_interface::EncData;
type MixData = rle_interface::MixData;

// Structure to preserve state of the RLEEnc for the next iteration
struct StreamDemuxerState<WORD_WIDTH: u32> {
    symbol: bits[WORD_WIDTH], // symbol
    write: bool,              // send <symbol, count> pair
}

pub proc StreamDemuxerSimple<WORD_WIDTH: u32> {
    word_r: chan<MixData<WORD_WIDTH>> in;
    data_s: chan<EncData<WORD_WIDTH, WORD_WIDTH>> out;

    init {(StreamDemuxerState<WORD_WIDTH> {
            symbol:  bits[WORD_WIDTH]:0,
            write:   bool:false,
        }
    )}

    config(
        word_r: chan<MixData<WORD_WIDTH>> in,
        data_s: chan<EncData<WORD_WIDTH, WORD_WIDTH>> out,
    ) {(word_r, data_s)}

    next (tok: token, state: StreamDemuxerState<WORD_WIDTH>) {
        let (tok, word) = recv(tok, word_r);
        let tok = send_if(tok, data_s, state.write,
            EncData<WORD_WIDTH> {
                symbol: state.symbol,
                count:  word.word,
                last:   word.last,
            });
        StreamDemuxerState<WORD_WIDTH> {
            symbol: word.word,
            write:  !state.write,
        }
    }
}

// Stream Demuxer specialization for the codegen
proc StreamDemuxerSimple32 {
    init {()}

    config (
        word_r: chan<MixData<32>> in,
        data_s: chan<EncData<32, 32>> out,
    ) {
        spawn StreamDemuxerSimple<u32:32>(word_r, data_s);
        ()
    }
    next (tok: token, state: ()) {
        ()
    }
}

// Tests

type TestEncData = EncData<32, 32>;
type TestMixData = MixData<32>;

const TEST_SEND_WORD_LOOKUP = TestMixData[20]:[
    TestMixData{word: bits[32]:0xA, last: bool:false},
    TestMixData{word: bits[32]:0x3, last: bool:false},
    TestMixData{word: bits[32]:0xB, last: bool:false},
    TestMixData{word: bits[32]:0x2, last: bool:false},
    TestMixData{word: bits[32]:0x1, last: bool:false},
    TestMixData{word: bits[32]:0x1, last: bool:false},
    TestMixData{word: bits[32]:0xC, last: bool:false},
    TestMixData{word: bits[32]:0x3, last: bool:true},
    TestMixData{word: bits[32]:0xC, last: bool:false},
    TestMixData{word: bits[32]:0x3, last: bool:false},
    TestMixData{word: bits[32]:0x3, last: bool:false},
    TestMixData{word: bits[32]:0x3, last: bool:true},
    TestMixData{word: bits[32]:0x2, last: bool:false},
    TestMixData{word: bits[32]:0x2, last: bool:true},
    TestMixData{word: bits[32]:0x1, last: bool:false},
    TestMixData{word: bits[32]:0x1, last: bool:false},
    TestMixData{word: bits[32]:0x2, last: bool:false},
    TestMixData{word: bits[32]:0x1, last: bool:false},
    TestMixData{word: bits[32]:0x3, last: bool:false},
    TestMixData{word: bits[32]:0x1, last: bool:true},
];

const TEST_RECV_DATA_LOOKUP = TestEncData[10]:[
    TestEncData{symbol: bits[32]:0xA, count: bits[32]:0x3, last: bool:false},
    TestEncData{symbol: bits[32]:0xB, count: bits[32]:0x2, last: bool:false},
    TestEncData{symbol: bits[32]:0x1, count: bits[32]:0x1, last: bool:false},
    TestEncData{symbol: bits[32]:0xC, count: bits[32]:0x3, last: bool:true},
    TestEncData{symbol: bits[32]:0xC, count: bits[32]:0x3, last: bool:false},
    TestEncData{symbol: bits[32]:0x3, count: bits[32]:0x3, last: bool:true},
    TestEncData{symbol: bits[32]:0x2, count: bits[32]:0x2, last: bool:true},
    TestEncData{symbol: bits[32]:0x1, count: bits[32]:0x1, last: bool:false},
    TestEncData{symbol: bits[32]:0x2, count: bits[32]:0x1, last: bool:false},
    TestEncData{symbol: bits[32]:0x3, count: bits[32]:0x1, last: bool:true},
];

const TEST_SEND_WORD_LEN = array_size(TEST_SEND_WORD_LOOKUP);
const TEST_RECV_DATA_LEN = array_size(TEST_RECV_DATA_LOOKUP);

// structure to preserve state of the RLE encoder tester for the next iteration
pub struct StreamDemuxerSimpleTesterState {
    recv_count: u32,                   // number of received transactions
}

// auxiliary proc for sending data to the tested Stream Demuxer
proc StreamDemuxerTestSender {
    word_s: chan<TestMixData> out;  // words sent to the tested module
    init {(u32:0)}
    config ( word_s: chan<TestMixData> out) { (word_s,) }
    next (tok: token, count: u32) {
        if count < TEST_SEND_WORD_LEN {
            let test_word = TEST_SEND_WORD_LOOKUP[count];
            let send_tok  = send(tok, word_s, test_word);
            let _ = trace_fmt!("Sent {} transactions, word: 0x{:x}, last: {}",
                count, test_word.word, test_word.last);
            count + u32:1
         } else {
            count
         }
    }
}

// main proc used to test the MixSimple
#[test_proc]
proc StreamDemuxerSimpleTester {
    terminator: chan<bool> out;     // test termination request
    data_r: chan<TestEncData> in;   // data read from the tested MixSimple

    init {(zero!<StreamDemuxerSimpleTesterState>())}

    config(terminator: chan<bool> out) {
        let (word_s, word_r) = chan<TestMixData>;
        let (data_s, data_r) = chan<TestEncData>;

        spawn StreamDemuxerTestSender(word_s);
        spawn StreamDemuxerSimple<u32:32>(word_r, data_s);
        (terminator, data_r)
    }

    next(tok: token, state: StreamDemuxerSimpleTesterState) {
        let (recv_tok, test_data) = recv(tok, data_r);

        let recv_count = state.recv_count + u32:1;

        let _ = trace_fmt!(
            "Received {} transactions, symbol: 0x{:x}, count:{}, last: {}",
            recv_count, test_data.symbol, test_data.count, test_data.last
        );

        // checks for expected values

        let exp_data = TEST_RECV_DATA_LOOKUP[state.recv_count];
        let _ = assert_eq(exp_data, test_data);

        // check the total count after the last expected receive matches
        // with the expected value

        let _ = if (recv_count == TEST_RECV_DATA_LEN) {
            send(recv_tok, terminator, true);
        } else {()};

        // state for the next iteration

        StreamDemuxerSimpleTesterState {
            recv_count: recv_count,
        }
    }
}
