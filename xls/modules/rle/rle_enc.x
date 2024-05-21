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

import std;
import xls.modules.rle.rle_common as rle_common;

type EncInData  = rle_common::PlainData;
type EncOutData = rle_common::CompressedData;

// structure to preserve the state of an RLE encoder
struct RunLengthEncoderState<SYMBOL_WIDTH: u32, COUNT_WIDTH: u32> {
    // symbol from the previous RunLengthEncoder::next evaluation,
    // valid if prev_count > 0
    prev_symbol: bits[SYMBOL_WIDTH],
    // symbol count from the previous RunLengthEncoder::next evaluation.
    // zero means that the previous evaluation sent all the data and
    // we start counting from the beginning
    prev_count: bits[COUNT_WIDTH],
    // flag indicating that the previous symbol was the last one
    // in the transmission
    prev_last: bool,
}

// RLE encoder implementation
pub proc RunLengthEncoder<SYMBOL_WIDTH: u32, COUNT_WIDTH: u32> {
    input_r: chan<EncInData<SYMBOL_WIDTH>> in;
    output_s: chan<EncOutData<SYMBOL_WIDTH, COUNT_WIDTH>> out;

    init {(
        RunLengthEncoderState<SYMBOL_WIDTH, COUNT_WIDTH> {
            prev_symbol: bits[SYMBOL_WIDTH]:0,
            prev_count: bits[COUNT_WIDTH]:0,
            prev_last: false,
        }
    )}

    config (
        input_r: chan<EncInData<SYMBOL_WIDTH>> in,
        output_s: chan<EncOutData<SYMBOL_WIDTH, COUNT_WIDTH>> out,
    ) {(input_r, output_s)}

    next (state: RunLengthEncoderState<SYMBOL_WIDTH, COUNT_WIDTH>) {
        let zero_input = EncInData {
          symbol: bits[SYMBOL_WIDTH]:0,
          last: false
        };
        let (input_tok, input) = recv_if(
            join(), input_r, !state.prev_last, zero_input);

        let prev_symbol_valid = state.prev_count != bits[COUNT_WIDTH]:0;
        let symbol_differ = prev_symbol_valid && (
            input.symbol != state.prev_symbol);
        let overflow =
            state.prev_count == std::unsigned_max_value<COUNT_WIDTH>();

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

        RunLengthEncoderState {
            prev_symbol: symbol,
            prev_count: count,
            prev_last: last,
        }
    }
}

// RLE encoder specialization for the codegen
proc RunLengthEncoder32 {

    init {()}

    config (
        input_r: chan<EncInData<32>> in,
        output_s: chan<EncOutData<32, 2>> out,
    ) {
        spawn RunLengthEncoder<u32:32, u32:2>(input_r, output_s);
        ()
    }

    next (state: ()) {
        ()
    }
}

// Tests

const TEST_COMMON_SYMBOL_WIDTH = u32:32;
// Make counter large enough so that it overflows only in overflow testcase.
const TEST_COMMON_COUNT_WIDTH = u32:32;

type TestCommonSymbol     = bits[TEST_COMMON_SYMBOL_WIDTH];
type TestCommonCount      = bits[TEST_COMMON_COUNT_WIDTH];
type TestCommonEncInData  = EncInData<TEST_COMMON_SYMBOL_WIDTH>;
type TestCommonEncOutData =
    EncOutData<TEST_COMMON_SYMBOL_WIDTH, TEST_COMMON_COUNT_WIDTH>;

// Simple transaction without overflow
const COUNT_SYMBOL_TEST_SYMBOL_WIDTH = TEST_COMMON_SYMBOL_WIDTH;
const COUNT_SYMBOL_TEST_COUNT_WIDTH  = TEST_COMMON_COUNT_WIDTH;

type CountSymbolTestStimulus   = TestCommonSymbol;
type CountSymbolTestSymbol     = TestCommonSymbol;
type CountSymbolTestCount      = TestCommonCount;
type CountSymbolTestEncInData  = TestCommonEncInData;
type CountSymbolTestEncOutData = TestCommonEncOutData;

#[test_proc]
proc RunLengthEncoderCountSymbolTest {
  terminator: chan<bool> out;
  enc_input_s: chan<CountSymbolTestEncInData> out;
  enc_output_r: chan<CountSymbolTestEncOutData> in;

  init {()}
  config (terminator: chan<bool> out) {
    let (enc_input_s, enc_input_r) = chan<CountSymbolTestEncInData>("enc_input");
    let (enc_output_s, enc_output_r) = chan<CountSymbolTestEncOutData>("enc_output");

    spawn RunLengthEncoder<COUNT_SYMBOL_TEST_SYMBOL_WIDTH, COUNT_SYMBOL_TEST_COUNT_WIDTH>(
        enc_input_r, enc_output_s);
    (terminator, enc_input_s, enc_output_r)
  }
  next (state:()) {
    let CountSymbolTestTestStimuli: CountSymbolTestStimulus[4] = [
      CountSymbolTestStimulus:0xA, CountSymbolTestStimulus:0xA,
      CountSymbolTestStimulus:0xA, CountSymbolTestStimulus:0xB,
    ];
    let tok = for ((counter, symbol), tok):
                  ((u32, CountSymbolTestStimulus) , token)
                  in enumerate(CountSymbolTestTestStimuli) {
      let last = counter == (array_size(CountSymbolTestTestStimuli) - u32:1);
      let stimulus = CountSymbolTestEncInData{symbol: symbol, last: last};
      let tok = send(tok, enc_input_s, stimulus);
      trace_fmt!("Sent {} stimuli, symbol: 0x{:x}, last: {}",
        counter, stimulus.symbol, stimulus.last);
      (tok)
    }(join());
    let CountSymbolTestTestOutput:
        (CountSymbolTestSymbol, CountSymbolTestCount)[2] = [
      (CountSymbolTestSymbol:0xA, CountSymbolTestCount:0x3),
      (CountSymbolTestSymbol:0xB, CountSymbolTestCount:0x1),
    ];
    let tok = for ((counter, (symbol, count)), tok):
        ((u32, (CountSymbolTestSymbol, CountSymbolTestCount)) , token)
        in enumerate(CountSymbolTestTestOutput) {
      let last = counter == (array_size(CountSymbolTestTestOutput) - u32:1);
      let expected = CountSymbolTestEncOutData{
          symbol: symbol, count: count, last: last};
      let (tok, enc_output) = recv(tok, enc_output_r);
      trace_fmt!(
        "Received {} pairs, symbol: 0x{:x}, count: {}, last: {}",
        counter, enc_output.symbol, enc_output.count, enc_output.last
      );
      assert_eq(enc_output, expected);
      (tok)
    }(tok);
    send(tok, terminator, true);
  }
}

// Transaction with counter overflow
const OVERFLOW_SYMBOL_WIDTH = TEST_COMMON_SYMBOL_WIDTH;
const OVERFLOW_COUNT_WIDTH  = u32:2;

type OverflowStimulus   = TestCommonSymbol;
type OverflowSymbol     = TestCommonSymbol;
type OverflowCount      = bits[OVERFLOW_COUNT_WIDTH];
type OverflowEncInData  = TestCommonEncInData;
type OverflowEncOutData =
    EncOutData<OVERFLOW_SYMBOL_WIDTH, OVERFLOW_COUNT_WIDTH>;

#[test_proc]
proc RunLengthEncoderOverflowTest {
  terminator: chan<bool> out;
  enc_input_s: chan<OverflowEncInData> out;
  enc_output_r: chan<OverflowEncOutData> in;

  init {()}
  config (terminator: chan<bool> out) {
    let (enc_input_s, enc_input_r) = chan<OverflowEncInData>("enc_input");
    let (enc_output_s, enc_output_r) = chan<OverflowEncOutData>("enc_output");

    spawn RunLengthEncoder<OVERFLOW_SYMBOL_WIDTH, OVERFLOW_COUNT_WIDTH>(
        enc_input_r, enc_output_s);
    (terminator, enc_input_s, enc_output_r)
  }
  next (state:()) {
    let OverflowTestStimuli: OverflowStimulus[14] = [
      OverflowStimulus:0xB, OverflowStimulus:0xB,
      OverflowStimulus:0x1, OverflowStimulus:0xC,
      OverflowStimulus:0xC, OverflowStimulus:0xC,
      OverflowStimulus:0xC, OverflowStimulus:0xC,
      OverflowStimulus:0xC, OverflowStimulus:0x3,
      OverflowStimulus:0x3, OverflowStimulus:0x3,
      OverflowStimulus:0x2, OverflowStimulus:0x2,
    ];
    let tok = for ((counter, symbol), tok):
                  ((u32, OverflowStimulus) , token)
                  in enumerate(OverflowTestStimuli) {
      let last = counter == (
          array_size(OverflowTestStimuli) - u32:1);
      let stimulus = OverflowEncInData{symbol: symbol, last: last};
      let tok = send(tok, enc_input_s, stimulus);
      trace_fmt!("Sent {} stimuli, symbol: 0x{:x}, last: {}",
        counter, stimulus.symbol, stimulus.last);
      (tok)
    }(join());
    let OverflowTestOutput:
        (OverflowSymbol, OverflowCount)[6] = [
      (OverflowSymbol:0xB, OverflowCount:0x2),
      (OverflowSymbol:0x1, OverflowCount:0x1),
      (OverflowSymbol:0xC, OverflowCount:0x3),
      (OverflowSymbol:0xC, OverflowCount:0x3),
      (OverflowSymbol:0x3, OverflowCount:0x3),
      (OverflowSymbol:0x2, OverflowCount:0x2),
    ];
    let tok = for ((counter, (symbol, count)), tok):
        ((u32, (OverflowSymbol, OverflowCount)) , token)
        in enumerate(OverflowTestOutput) {
      let last = counter == (array_size(OverflowTestOutput) - u32:1);
      let expected = OverflowEncOutData{
          symbol: symbol, count: count, last: last};
      let (tok, enc_output) = recv(tok, enc_output_r);
      trace_fmt!(
        "Received {} pairs, symbol: 0x{:x}, count: {}, last: {}",
        counter, enc_output.symbol, enc_output.count, enc_output.last
      );
      assert_eq(enc_output, expected);
      (tok)
    }(tok);
    send(tok, terminator, true);
  }
}

// Check that RLE encoder will create 2 `last` output packets,
// when 2 `last` input packets were consumed.
const LAST_AFTER_LAST_SYMBOL_WIDTH = TEST_COMMON_SYMBOL_WIDTH;
const LAST_AFTER_LAST_COUNT_WIDTH  = TEST_COMMON_COUNT_WIDTH;

type LastAfterLastStimulus   = TestCommonEncInData;
type LastAfterLastSymbol     = TestCommonSymbol;
type LastAfterLastCount      = TestCommonCount;
type LastAfterLastEncInData  = TestCommonEncInData;
type LastAfterLastEncOutData = TestCommonEncOutData;
type LastAfterLastOutput     = TestCommonEncOutData;

#[test_proc]
proc RunLengthEncoderLastAfterLastTest {
  terminator: chan<bool> out;
  enc_input_s: chan<LastAfterLastEncInData> out;
  enc_output_r: chan<LastAfterLastEncOutData> in;

  init {()}
  config (terminator: chan<bool> out) {
    let (enc_input_s, enc_input_r) = chan<LastAfterLastEncInData>("enc_input");
    let (enc_output_s, enc_output_r) = chan<LastAfterLastEncOutData>("enc_output");

    spawn RunLengthEncoder<LAST_AFTER_LAST_SYMBOL_WIDTH, LAST_AFTER_LAST_COUNT_WIDTH>(
        enc_input_r, enc_output_s);
    (terminator, enc_input_s, enc_output_r)
  }
  next (state:()) {
    let LastAfterLastTestStimuli: LastAfterLastStimulus[2] = [
      LastAfterLastStimulus {symbol: LastAfterLastSymbol:0x1, last: true},
      LastAfterLastStimulus {symbol: LastAfterLastSymbol:0x1, last: true},
    ];
    let tok = for ((counter, stimuli), tok):
        ((u32, LastAfterLastStimulus) , token)
        in enumerate(LastAfterLastTestStimuli) {
      let tok = send(tok, enc_input_s, stimuli);
      trace_fmt!("Sent {} transactions, symbol: 0x{:x}, last: {}",
        counter, stimuli.symbol, stimuli.last);
      (tok)
    }(join());
    let LastAfterLastTestOutput: LastAfterLastOutput[2] = [
      LastAfterLastOutput {
        symbol: LastAfterLastSymbol:0x1,
        count: LastAfterLastCount:0x1,
        last:true},
      LastAfterLastOutput {
        symbol: LastAfterLastSymbol:0x1,
        count: LastAfterLastCount:0x1,
        last:true},
    ];
    let tok = for ((counter, expected), tok):
        ((u32, LastAfterLastOutput) , token)
        in enumerate(LastAfterLastTestOutput) {
      let (tok, enc_output) = recv(tok, enc_output_r);
      trace_fmt!(
        "Received {} pairs, symbol: 0x{:x}, count: {}, last: {}",
        counter, enc_output.symbol, enc_output.count, enc_output.last
      );
      assert_eq(enc_output, expected);

      (tok)
    }(tok);
    send(tok, terminator, true);
  }
}

// Check overflow condition trigger on packet with `last`
const OVERFLOW_WITH_LAST_SYMBOL_WIDTH = TEST_COMMON_SYMBOL_WIDTH;
const OVERFLOW_WITH_LAST_COUNT_WIDTH  = u32:2;

type OverflowWithLastStimulus   = TestCommonSymbol;
type OverflowWithLastSymbol     = TestCommonSymbol;
type OverflowWithLastCount      =
    bits[OVERFLOW_WITH_LAST_COUNT_WIDTH];
type OverflowWithLastEncInData  = TestCommonEncInData;
type OverflowWithLastEncOutData =
    EncOutData<OVERFLOW_WITH_LAST_SYMBOL_WIDTH,
               OVERFLOW_WITH_LAST_COUNT_WIDTH>;

#[test_proc]
proc RunLengthEncoderOverflowWithLastTest {
  terminator: chan<bool> out;
  enc_input_s: chan<OverflowWithLastEncInData> out;
  enc_output_r: chan<OverflowWithLastEncOutData> in;

  init {()}
  config (terminator: chan<bool> out) {
    let (enc_input_s, enc_input_r) = chan<OverflowWithLastEncInData>("enc_input");
    let (enc_output_s, enc_output_r) =
        chan<OverflowWithLastEncOutData>("enc_output");

    spawn RunLengthEncoder<OVERFLOW_WITH_LAST_SYMBOL_WIDTH,
                 OVERFLOW_WITH_LAST_COUNT_WIDTH>(
        enc_input_r, enc_output_s);
    (terminator, enc_input_s, enc_output_r)
  }
  next (state:()) {
    let OverflowWithLastTestStimuli: OverflowWithLastStimulus[4] = [
      OverflowWithLastStimulus:0xC, OverflowWithLastStimulus:0xC,
      OverflowWithLastStimulus:0xC, OverflowWithLastStimulus:0xC,
    ];
    let tok = for ((counter, symbol), tok):
                  ((u32, OverflowWithLastStimulus) , token)
                  in enumerate(OverflowWithLastTestStimuli) {
      let last = counter == (
          array_size(OverflowWithLastTestStimuli) - u32:1);
      let stimulus = OverflowWithLastEncInData{symbol: symbol, last: last};
      let tok = send(tok, enc_input_s, stimulus);
      trace_fmt!("Sent {} stimuli, symbol: 0x{:x}, last: {}",
        counter, stimulus.symbol, stimulus.last);
      (tok)
    }(join());
    let OverflowWithLastTestOutput:
        (OverflowWithLastSymbol, OverflowWithLastCount)[2] = [
      (OverflowWithLastSymbol:0xC, OverflowWithLastCount:0x3),
      (OverflowWithLastSymbol:0xC, OverflowWithLastCount:0x1),
    ];
    let tok = for ((counter, (symbol, count)), tok):
        ((u32, (OverflowWithLastSymbol, OverflowWithLastCount)) , token)
        in enumerate(OverflowWithLastTestOutput) {
      let last = counter == (array_size(OverflowWithLastTestOutput) - u32:1);
      let expected = OverflowWithLastEncOutData{
          symbol: symbol, count: count, last: last};
      let (tok, enc_output) = recv(tok, enc_output_r);
      trace_fmt!(
        "Received {} pairs, symbol: 0x{:x}, count: {}, last: {}",
        counter, enc_output.symbol, enc_output.count, enc_output.last
      );
      assert_eq(enc_output, expected);
      (tok)
    }(tok);
    send(tok, terminator, true);
  }
}
