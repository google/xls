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
// The encoder uses Run Length Encoding (RLE) to compress the input stream of
// repeating symbols to the output stream that contains the symbols and
// the number of its consequect occurrences in the input stream.
// Both the input and the output channels use additional `last` flag
// that indicates whether the packet ends the transmission. After sending
// the last packet encoder dumps all the data to the output stream.
// The behavior of the encoder is presented on the waveform below:
//                      ──────╥─────╥─────╥─────╥─────╥─────╥─────╥─────╥────
// next evaluation      XXXXXX║ 0   ║ 1   ║ 2   ║ 3   ║ 4   ║ 5   ║ 6   ║ ...
//                      ──────╨─────╨─────╨─────╨─────╨─────╨─────╨─────╨────
//                      ──────╥───────────╥─────╥─────╥─────╥─────╥──────────
// symbol               XXXXXX║ A         ║ B   ║XXXXX║ B   ║ C   ║XXXXXXXXXX
// (input channel)      ──────╨───────────╨─────╨─────╨─────╨─────╨──────────
// last                                   ┌─────┐           ┌─────┐
// (input channel)      ──────────────────┘     └───────────┘     └──────────
//                      ╥─────╥─────╥─────╥─────╥─────╥─────╥─────╥──────────
// state.prev_symbol    ║ 0   ║ A   ║ A   ║ B   ║ 0   ║ B   ║ C   ║ 0
// (set state value)    ╨─────╨─────╨─────╨─────╨─────╨─────╨─────╨──────────
//                      ╥─────╥─────╥─────╥─────╥─────╥─────╥─────╥──────────
// state.prev_count     ║ 0   ║ 1   ║ 2   ║ 1   ║ 0   ║ 1   ║ 1   ║ 0
// (set state value)    ╨─────╨─────╨─────╨─────╨─────╨─────╨─────╨──────────
//
// do_send                                ┌───────────┐     ┌───────────┐
//                      ──────────────────┘           └─────┘           └────
//                      ──────────────────╥─────╥─────╥─────╥─────╥─────╥────
// symbol, count        XXXXXXXXXXXXXXXXXX║ A,2 ║ B,1 ║XXXXX║ B,1 ║ C,1 ║XXXX
// (output channel)     ──────────────────╨─────╨─────╨─────╨─────╨─────╨────
// last                                         ┌─────┐           ┌─────┐
// (output channel)     ────────────────────────┘     └───────────┘     └────

import std
import xls.modules.rle.rle_interface as rle_interface

type EncInData  = rle_interface::PreData;
type EncOutData = rle_interface::EncData;

// structure to preserve the state of an RLE encoder
struct RLEEncState<SYMBOL_WIDTH: u32, COUNT_WIDTH: u32> {
    // symbol from the previous RLEEnc::next evaluation,
    // valid if prev_count > 0
    prev_symbol: bits[SYMBOL_WIDTH],
    // symbol count from the previous RLEEnc::next evaluation.
    // zero means that the previous evaluation sent all the data and
    // we start counting from the beginning
    prev_count: bits[COUNT_WIDTH],
    // flag indicating that the previous symbol was the last one
    // in the transmission
    prev_last: bool,
}

// RLE encoder implementation
pub proc RLEEnc<SYMBOL_WIDTH: u32, COUNT_WIDTH: u32> {
    input_r: chan<EncInData<SYMBOL_WIDTH>> in;
    output_s: chan<EncOutData<SYMBOL_WIDTH, COUNT_WIDTH>> out;

    init {(
        RLEEncState<SYMBOL_WIDTH, COUNT_WIDTH> {
            prev_symbol: bits[SYMBOL_WIDTH]:0,
            prev_count: bits[COUNT_WIDTH]:0,
            prev_last: false,
        }
    )}

    config (
        input_r: chan<EncInData<SYMBOL_WIDTH>> in,
        output_s: chan<EncOutData<SYMBOL_WIDTH, COUNT_WIDTH>> out,
    ) {(input_r, output_s)}

    next (tok: token, state: RLEEncState<SYMBOL_WIDTH, COUNT_WIDTH>) {
        let zero_input = EncInData { symbol: bits[SYMBOL_WIDTH]:0, last: false };
        let (input_tok, input) = recv_if(tok, input_r, !state.prev_last, zero_input);

        let prev_symbol_valid = state.prev_count != bits[COUNT_WIDTH]:0;
        let symbol_differ = prev_symbol_valid && (input.symbol != state.prev_symbol);
        let overflow = state.prev_count == std::unsigned_max_value<COUNT_WIDTH>();

        let (symbol, count, last) = if (state.prev_last) {
            (
                bits[SYMBOL_WIDTH]:0,
                bits[COUNT_WIDTH]:0,
                false
            )
        } else if (symbol_differ || overflow) {
            (
                input.symbol,
                bits[COUNT_WIDTH]:1,
                input.last,
            )
        } else {
            (
                input.symbol,
                state.prev_count + bits[COUNT_WIDTH]:1,
                input.last,
            )
        };

        let data = EncOutData {
            symbol: state.prev_symbol,
            count: state.prev_count,
            last: state.prev_last
        };

        let do_send = state.prev_last || symbol_differ || overflow;
        let data_tok = send_if(input_tok, output_s, do_send, data);

        RLEEncState {
            prev_symbol: symbol,
            prev_count: count,
            prev_last: last,
        }
    }
}

// RLE encoder specialization for the codegen
proc RLEEnc32 {
    init {()}

    config (
        input_r: chan<EncInData<32>> in,
        output_s: chan<EncOutData<32, 2>> out,
    ) {
        spawn RLEEnc<u32:32, u32:2>(input_r, output_s);
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
type TestEncInData  = EncInData<TEST_SYMBOL_WIDTH>;
type TestEncOutData = EncOutData<TEST_SYMBOL_WIDTH, TEST_COUNT_WIDTH>;

const TEST_SEND_LOOKUP = TestEncInData[20]:[
    TestEncInData {symbol: TestSymbol:0xA, last: false},
    TestEncInData {symbol: TestSymbol:0xA, last: false},
    TestEncInData {symbol: TestSymbol:0xA, last: false},
    TestEncInData {symbol: TestSymbol:0xB, last: true},
    TestEncInData {symbol: TestSymbol:0xB, last: false},
    TestEncInData {symbol: TestSymbol:0xB, last: false},
    TestEncInData {symbol: TestSymbol:0x1, last: false},
    TestEncInData {symbol: TestSymbol:0xC, last: false},
    TestEncInData {symbol: TestSymbol:0xC, last: false},
    TestEncInData {symbol: TestSymbol:0xC, last: false},
    TestEncInData {symbol: TestSymbol:0xC, last: false},
    TestEncInData {symbol: TestSymbol:0xC, last: false},
    TestEncInData {symbol: TestSymbol:0xC, last: false},
    TestEncInData {symbol: TestSymbol:0x3, last: false},
    TestEncInData {symbol: TestSymbol:0x3, last: false},
    TestEncInData {symbol: TestSymbol:0x3, last: false},
    TestEncInData {symbol: TestSymbol:0x2, last: false},
    TestEncInData {symbol: TestSymbol:0x2, last: true},
    TestEncInData {symbol: TestSymbol:0x1, last: true},
    TestEncInData {symbol: TestSymbol:0x2, last: true},
];

const TEST_RECV_LOOKUP = TestEncOutData[10]:[
    TestEncOutData {symbol: TestSymbol:0xA, count: TestCount:0x3, last:false},
    TestEncOutData {symbol: TestSymbol:0xB, count: TestCount:0x1, last:true},
    TestEncOutData {symbol: TestSymbol:0xB, count: TestCount:0x2, last:false},
    TestEncOutData {symbol: TestSymbol:0x1, count: TestCount:0x1, last:false},
    TestEncOutData {symbol: TestSymbol:0xC, count: TestCount:0x3, last:false},
    TestEncOutData {symbol: TestSymbol:0xC, count: TestCount:0x3, last:false},
    TestEncOutData {symbol: TestSymbol:0x3, count: TestCount:0x3, last:false},
    TestEncOutData {symbol: TestSymbol:0x2, count: TestCount:0x2, last:true},
    TestEncOutData {symbol: TestSymbol:0x1, count: TestCount:0x1, last:true},
    TestEncOutData {symbol: TestSymbol:0x2, count: TestCount:0x1, last:true},
];

// structure to preserve state of the RLE encoder tester for the next iteration
struct RLEEncTesterState {
    recv_count: u32,      // number of received transactions
    total_count: u32,     // total number of received symbols
    prev: TestEncOutData, // previous data received from the RLE encoder
}

// auxiliary proc for sending data to the tested RLE encoder
proc RLEEncTestSender {
    enc_input_s: chan<TestEncInData> out;

    init {(u32:0)}

    config (
        enc_input_s: chan<TestEncInData> out,
    ) {(enc_input_s,)}

    next (tok: token, counter: u32) {
        if counter < array_size(TEST_SEND_LOOKUP) {
            let data = TEST_SEND_LOOKUP[counter];
            let send_tok = send(tok, enc_input_s, data);
            let _ = trace_fmt!("Sent {} transactions, symbol: 0x{:x}, last: {}", counter, data.symbol, data.last);
            counter + u32:1
         } else {
            counter
         }
    }
}

// main proc used to test the RLE encoder
#[test_proc]
proc RLEEncTester {
    terminator: chan<bool> out;            // test termination request
    enc_output_r: chan<TestEncOutData> in; // data read from the tested RLE encoder

    init {(
        RLEEncTesterState {
            recv_count: u32:0,
            total_count: u32:0,
            prev: TestEncOutData  {
                symbol: TestSymbol:0,
                count: TestCount:0,
                last: false,
            },
        }
    )}

    config(terminator: chan<bool> out) {
        let (enc_input_s, enc_input_r)   = chan<TestEncInData>;
        let (enc_output_s, enc_output_r) = chan<TestEncOutData>;

        spawn RLEEncTestSender(enc_input_s);
        spawn RLEEnc<TEST_SYMBOL_WIDTH, TEST_COUNT_WIDTH>(enc_input_r, enc_output_s);
        (terminator, enc_output_r)
    }

    next(tok: token, state: RLEEncTesterState) {
        let (recv_tok, enc_output) = recv(tok, enc_output_r);

        let recv_count = state.recv_count + u32:1;
        let total_count = state.total_count + enc_output.count as u32;

        let _ = trace_fmt!(
            "Received {} transactions, symbol: 0x{:x}, count: {}, last: {}, total of {} symbols",
            recv_count, enc_output.symbol, enc_output.count, enc_output.last, total_count
        );

        // checks for expected values

        let exp_enc_output = TEST_RECV_LOOKUP[state.recv_count];
        let _ = assert_eq(enc_output, exp_enc_output);

        // check for general properties of the received data

        let max_count = std::unsigned_max_value<TEST_COUNT_WIDTH>();

        let _ = assert_eq(enc_output.count > TestCount:0, true);
        let _ = assert_eq(enc_output.count <= max_count, true);

        // if the symbol repeats, check that the symbol counter has reached
        // its maximum value or if the symbol was last in the transaction

        let prev_data_valid = state.recv_count != u32:0;
        let _ = if prev_data_valid {
            let data_repeats = state.prev.symbol == enc_output.symbol;
            let overflow = state.prev.count == max_count;
            let last = state.prev.last;

            let _ = if data_repeats {
                let _ = assert_eq(overflow || last, true);
            } else {()};
        } else {()};

        // check the total count after the last expected receive matches
        // with the expected value

        let max_recv_count = array_size(TEST_RECV_LOOKUP);
        let max_total_count = array_size(TEST_SEND_LOOKUP);
        let exp_recv_reached = recv_count == max_recv_count;
        let exp_total_reached = total_count == max_total_count;

        let _ = if exp_recv_reached {
            let _ = assert_eq(exp_total_reached, true);
        } else {
            let _ = assert_lt(recv_count, max_recv_count);
            let _ = assert_lt(total_count, max_total_count);
        };

        // finish the test if we received all the data correctly

        let finish = exp_recv_reached && exp_total_reached;
        let _ = send_if(recv_tok, terminator, finish, true);

        // state for the next iteration

        RLEEncTesterState {
            recv_count: recv_count,
            total_count: total_count,
            prev: enc_output,
        }
    }
}
