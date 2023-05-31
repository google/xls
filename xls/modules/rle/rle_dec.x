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

// This file implements a parametric RLE decoder
//
// The RLE decoder decompresses incoming stream of
// (`symbol`, `count`) pairs, representing that `symbol` was
// repeated `count` times in original stream. Output stream
// should be equal to the RLE encoder input stream.
// Both input and output channels use additional `last` flag
// that indicates whether the packet ends the transmission.
// Decoder in its current form only propagates last signal.
// The behavior of the decoder is presented on the waveform below:
//                      ──────╥─────╥─────╥─────╥─────╥─────╥─────╥─────╥────
// next evaluation      XXXXXX║ 0   ║ 1   ║ 2   ║ 3   ║ 4   ║ 5   ║ 6   ║ ...
//                      ──────╨─────╨─────╨─────╨─────╨─────╨─────╨─────╨────
// do_recv                    ┌─────┐     ┌─────────────────┐           ┌────
//                      ──────┘     └─────┘                 └───────────┘    
//                      ──────╥─────╥─────╥─────╥─────╥─────╥────────────────
// symbol, count        XXXXXX║ A,2 ║XXXXX║ B,1 ║ B,1 ║ C,3 ║XXXXXXXXXXXXXXXX
// (input channel)      ──────╨─────╨─────╨─────╨─────╨─────╨────────────────
// last                                   ┌─────┐     ┌─────┐                
// (input channel)      ──────────────────┘     └─────┘     └────────────────
//                      ╥─────╥───────────╥───────────╥─────────────────╥────
// state.symbol         ║ 0   ║ A         ║ B         ║ C               ║ 0  
// (set state value)    ╨─────╨───────────╨───────────╨─────────────────╨────
//                      ╥─────╥─────╥─────╥───────────╥─────╥─────╥─────╥────
// state.count          ║ 0   ║ 1   ║ 0   ║ 0         ║ 2   ║ 1   ║ 0   ║ 0  
// (set state value)    ╨─────╨─────╨─────╨───────────╨─────╨─────╨─────╨────
//                                                               
//                      ──────╥───────────╥───────────╥─────────────────╥────
// symbol               XXXXXX║ A         ║ B         ║ C               ║XXXX
// (output channel)     ──────╨───────────╨───────────╨─────────────────╨────
// last                                   ┌─────┐                 ┌─────┐    
// (output channel)     ──────────────────┘     └─────────────────┘     └────


import std
import xls.modules.rle.rle_interface as rle_interface

type DecInData  = rle_interface::EncData;
type DecOutData = rle_interface::PreData;

// structure to preserve the state of an RLE decoder
struct RLEDecState<SYMBOL_WIDTH: u32, COUNT_WIDTH: u32> {
    // symbol to be repeated on output
    symbol: bits[SYMBOL_WIDTH],
    // count of symbols that has to be send
    count: bits[COUNT_WIDTH],
    // send last when repeat ends
    last: bool,
}
 // RLE decoder implementation
pub proc RLEDec<SYMBOL_WIDTH: u32, COUNT_WIDTH: u32> {
    input_r: chan<DecInData<SYMBOL_WIDTH, COUNT_WIDTH>> in;
    output_s: chan<DecOutData<SYMBOL_WIDTH>> out;

    init {(
        RLEDecState<SYMBOL_WIDTH, COUNT_WIDTH> {
            symbol: bits[SYMBOL_WIDTH]:0,
            count:  bits[COUNT_WIDTH]:0,
            last:   bool:false,
        }
    )}

    config (
        input_r: chan<DecInData<SYMBOL_WIDTH, COUNT_WIDTH>> in,
        output_s: chan<DecOutData<SYMBOL_WIDTH>> out,
    ) {(input_r, output_s)}

    next (tok: token, state: RLEDecState<SYMBOL_WIDTH, COUNT_WIDTH>) {
        let zero_input = DecInData { symbol: bits[SYMBOL_WIDTH]:0, count: bits[COUNT_WIDTH]:0, last: false };
        let empty = state.count == bits[COUNT_WIDTH]:0;
        let (input_tok, input) = recv_if(tok, input_r, empty, zero_input);
        let (next_symbol, next_count, next_last) = if (empty) {
            let t_count = input.count - bits[COUNT_WIDTH]:1;
            (input.symbol, t_count, input.last)
        } else {
            let t_count = state.count - bits[COUNT_WIDTH]:1;
            (state.symbol, t_count, state.last)
        };
        let send_last = next_last & (next_count == bits[COUNT_WIDTH]:0);
        let data_tok = send(input_tok, output_s, DecOutData {symbol: next_symbol, last: send_last});
        RLEDecState {
            symbol: next_symbol,
            count: next_count,
            last: next_last,
        }
    }
}


// RLE decoder specialization for the codegen
proc RLEDec32 {
    init {()}

    config (
        input_r: chan<DecInData<32, 2>> in,
        output_s: chan<DecOutData<32>> out,
    ) {
        spawn RLEDec<u32:32, u32:2>(input_r, output_s);
        ()
    }

    next (tok: token, state: ()) {
        ()
    }
}

// Tests

const TEST_SYMBOL_WIDTH = u32:32;
const TEST_COUNT_WIDTH  = u32:2;

type TestSymbol = bits[TEST_SYMBOL_WIDTH];
type TestCount  = bits[TEST_COUNT_WIDTH];
type TestDecInData  = DecInData<TEST_SYMBOL_WIDTH, TEST_COUNT_WIDTH>;
type TestDecOutData = DecOutData<TEST_SYMBOL_WIDTH>;


const TEST_SEND_LOOKUP = TestDecInData[10]:[
    TestDecInData {symbol: TestSymbol:0xA, count: TestCount:0x3, last:false},
    TestDecInData {symbol: TestSymbol:0xB, count: TestCount:0x1, last:true},
    TestDecInData {symbol: TestSymbol:0xB, count: TestCount:0x2, last:false},
    TestDecInData {symbol: TestSymbol:0x1, count: TestCount:0x1, last:false},
    TestDecInData {symbol: TestSymbol:0xC, count: TestCount:0x3, last:false},
    TestDecInData {symbol: TestSymbol:0xC, count: TestCount:0x3, last:false},
    TestDecInData {symbol: TestSymbol:0x3, count: TestCount:0x3, last:false},
    TestDecInData {symbol: TestSymbol:0x2, count: TestCount:0x2, last:true},
    TestDecInData {symbol: TestSymbol:0x1, count: TestCount:0x1, last:true},
    TestDecInData {symbol: TestSymbol:0x2, count: TestCount:0x1, last:true},
];

const TEST_RECV_LOOKUP = TestDecOutData[20]:[
    TestDecOutData {symbol: TestSymbol:0xA, last: false},
    TestDecOutData {symbol: TestSymbol:0xA, last: false},
    TestDecOutData {symbol: TestSymbol:0xA, last: false},
    TestDecOutData {symbol: TestSymbol:0xB, last: true},
    TestDecOutData {symbol: TestSymbol:0xB, last: false},
    TestDecOutData {symbol: TestSymbol:0xB, last: false},
    TestDecOutData {symbol: TestSymbol:0x1, last: false},
    TestDecOutData {symbol: TestSymbol:0xC, last: false},
    TestDecOutData {symbol: TestSymbol:0xC, last: false},
    TestDecOutData {symbol: TestSymbol:0xC, last: false},
    TestDecOutData {symbol: TestSymbol:0xC, last: false},
    TestDecOutData {symbol: TestSymbol:0xC, last: false},
    TestDecOutData {symbol: TestSymbol:0xC, last: false},
    TestDecOutData {symbol: TestSymbol:0x3, last: false},
    TestDecOutData {symbol: TestSymbol:0x3, last: false},
    TestDecOutData {symbol: TestSymbol:0x3, last: false},
    TestDecOutData {symbol: TestSymbol:0x2, last: false},
    TestDecOutData {symbol: TestSymbol:0x2, last: true},
    TestDecOutData {symbol: TestSymbol:0x1, last: true},
    TestDecOutData {symbol: TestSymbol:0x2, last: true},
];

// structure to preserve state of the RLE decoder tester for the next iteration
pub struct RLEDecTesterState {
    total_count: u32,     // total number of received symbols
}

// auxiliary proc for sending data to the tested RLE decoder
proc RLEDecTestSender {
    dec_input_s: chan<TestDecInData> out;

    init {(u32:0)}

    config (
        dec_input_s: chan<TestDecInData> out,
    ) {(dec_input_s,)}

    next (tok: token, counter: u32) {
        if counter < array_size(TEST_SEND_LOOKUP) {
            let data = TEST_SEND_LOOKUP[counter];
            let send_tok = send(tok, dec_input_s, data);
            let _ = trace_fmt!("Sent {} transactions, symbol: 0x{:x}, count: {}, last: {}",
                counter, data.symbol, data.count, data.last);
            counter + u32:1
         } else {
            counter
         }
    }
}

// main proc used to test the RLE decoder
#[test_proc]
proc RLEDecTester {
    terminator: chan<bool> out;            // test termination request
    dec_output_r: chan<TestDecOutData> in; // data read from the tested RLE decoder

    init {(RLEDecTesterState {total_count: u32:0})}

    config(terminator: chan<bool> out) {
        let (dec_input_s, dec_input_r)   = chan<TestDecInData>;
        let (dec_output_s, dec_output_r) = chan<TestDecOutData>;

        spawn RLEDecTestSender(dec_input_s);
        spawn RLEDec<TEST_SYMBOL_WIDTH, TEST_COUNT_WIDTH>(dec_input_r, dec_output_s);
        (terminator, dec_output_r)
    }

    next(tok: token, state: RLEDecTesterState) {
        let (recv_tok, dec_output) = recv(tok, dec_output_r);

        let total_count = state.total_count + u32:1;

        let _ = trace_fmt!(
            "Received {} transactions, symbol: 0x{:x}, last: {}",
            total_count, dec_output.symbol, dec_output.last
        );

        // checks for expected values

        let exp_dec_output = TEST_RECV_LOOKUP[state.total_count];
        let _ = assert_eq(dec_output, exp_dec_output);

        // finish the test if we received all the data correctly

        let finish = total_count == array_size(TEST_RECV_LOOKUP);
        let _ = send_if(recv_tok, terminator, finish, true);

        // state for the next iteration

        RLEDecTesterState {
            total_count: total_count,
        }
    }
}
