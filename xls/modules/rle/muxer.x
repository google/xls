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
struct StreamMuxerState<WORD_WIDTH: u32> {
    counter: bits[WORD_WIDTH],  // symbol count
    read: bool,                 // read <symbol, count> pair
    last: bool,                 // last word
}

pub proc StreamMuxerSimple<WORD_WIDTH: u32> {
    data_r: chan<EncData<WORD_WIDTH, WORD_WIDTH>> in;
    word_s: chan<MixData<WORD_WIDTH>> out;

    init {(StreamMuxerState<WORD_WIDTH> {
            counter: bits[WORD_WIDTH]:0,
            read:    bool:true,
            last:    bool:false,
        }
    )}

    config(
        data_r: chan<EncData<WORD_WIDTH, WORD_WIDTH>> in,
        word_s: chan<MixData<WORD_WIDTH>> out
    ) {(data_r, word_s)}

    next (tok: token, state: StreamMuxerState<WORD_WIDTH>) {
        let (tok, words) = recv_if(tok, data_r, state.read,
            EncData<WORD_WIDTH> {
                symbol: state.counter,
                count:  bits[WORD_WIDTH]:0,
                last:   bool:false,
            });
        let last = state.last & !state.read;
        let tok = send(tok, word_s, MixData<WORD_WIDTH> {
            word: words.symbol,
            last: last,
        });
        StreamMuxerState<WORD_WIDTH> {
            counter: words.count,
            read:    !state.read,
            last:    words.last,
        }
    }
}

// Stream Muxer specialization for the codegen
proc StreamMuxerSimple32 {
    init {()}

    config (
        data_r: chan<EncData<32, 32>> in,
        word_s: chan<MixData<32>> out,
    ) {
        spawn StreamMuxerSimple<u32:32>(data_r, word_s);
        ()
    }
    next (tok: token, state: ()) {
        ()
    }
}

// Tests

type TestEncData = EncData<32, 32>;
type TestMixData = MixData<32>;

const TEST_SEND_DATA_LOOKUP = TestEncData[10]:[
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

const TEST_RECV_WORD_LOOKUP = TestMixData[20]:[
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

const TEST_SEND_DATA_LEN = array_size(TEST_SEND_DATA_LOOKUP);
const TEST_RECV_WORD_LEN = array_size(TEST_RECV_WORD_LOOKUP);

// structure to preserve state of the RLE encoder tester for the next iteration
pub struct StreamMuxerSimpleTesterState {
    recv_count: u32,                   // number of received transactions
}

// auxiliary proc for sending data to the tested Stream Muxer
proc StreamMuxerTestSender {
    data_s: chan<EncData<32, 32>> out;  // symbol sent to the tested module
    init {(u32:0)}
    config ( data_s: chan<EncData<32, 32>> out) { (data_s,) }
    next (tok: token, count: u32) {
        if count < TEST_SEND_DATA_LEN {
            let test_data = TEST_SEND_DATA_LOOKUP[count];
            let send_tok  = send(tok, data_s, test_data);
            let _ = trace_fmt!("Sent {} transactions, symbol: 0x{:x}, last: {}",
                test_data.count, test_data.symbol, test_data.last);
            count + u32:1
         } else {
            count
         }
    }
}

// main proc used to test the MixSimple
#[test_proc]
proc StreamMuxerSimpleTester {
    terminator: chan<bool> out;     // test termination request
    word_r: chan<TestMixData> in;   // data read from the tested MixSimple

    init {(zero!<StreamMuxerSimpleTesterState>())}

    config(terminator: chan<bool> out) {
        let (data_s, data_r) = chan<TestEncData>;
        let (word_s, word_r) = chan<TestMixData>;

        spawn StreamMuxerTestSender(data_s);
        spawn StreamMuxerSimple<u32:32>(data_r, word_s);
        (terminator, word_r)
    }

    next(tok: token, state: StreamMuxerSimpleTesterState) {
        let (recv_tok, test_word) = recv(tok, word_r);

        let recv_count = state.recv_count + u32:1;

        let _ = trace_fmt!(
            "Received {} transactions, word: 0x{:x}, total of {} words",
            recv_count, test_word.word, recv_count
        );

        // checks for expected values

        let exp_word = TEST_RECV_WORD_LOOKUP[state.recv_count];
        let _ = assert_eq(exp_word, test_word);

        // check the total count after the last expected receive matches
        // with the expected value

        let _ = if (recv_count == TEST_RECV_WORD_LEN) {
            send(recv_tok, terminator, true);
        } else {()};

        // state for the next iteration

        StreamMuxerSimpleTesterState {
            recv_count: recv_count,
        }
    }
}
