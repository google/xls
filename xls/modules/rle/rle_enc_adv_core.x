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

import std
import xls.modules.rle.rle_common as rle_common

type EncInData  = rle_common::PlainData;
type EncOutData = rle_common::CompressedData;


// structure to preserve the state of an RLE encoder
struct RunLengthEncoderAdvancedCoreState<SYMBOL_WIDTH: u32, COUNT_WIDTH: u32> {

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
pub proc RunLengthEncoderAdvancedCoreStage<SYMBOL_WIDTH: u32, COUNT_WIDTH: u32,
  INPUT_WIDTH:u32> {

  input_r: chan<(EncOutData<SYMBOL_WIDTH, COUNT_WIDTH, INPUT_WIDTH>,
    u32, u32)> in;
  output_s: chan<(EncOutData<SYMBOL_WIDTH, COUNT_WIDTH, INPUT_WIDTH>, u32)> out;

  init {(
    RunLengthEncoderAdvancedCoreState<SYMBOL_WIDTH, COUNT_WIDTH> {
      prev_symbol: bits[SYMBOL_WIDTH]:0,
      prev_count: bits[COUNT_WIDTH]:0,
      prev_last: false,
    }
  )}

  config (
    input_r:  chan<(EncOutData<SYMBOL_WIDTH, COUNT_WIDTH, INPUT_WIDTH>,
      u32, u32)> in,
    output_s: chan<(EncOutData<SYMBOL_WIDTH, COUNT_WIDTH, INPUT_WIDTH>,
      u32)> out,
  ) {(input_r, output_s)}

  next (tok: token, state: RunLengthEncoderAdvancedCoreState<SYMBOL_WIDTH, COUNT_WIDTH>) {
    let empty = (
      EncOutData<SYMBOL_WIDTH, COUNT_WIDTH, INPUT_WIDTH>{
        symbols: zero!<bits[SYMBOL_WIDTH][INPUT_WIDTH]>(),
        counts: zero!<bits[COUNT_WIDTH][INPUT_WIDTH]>(),
        last: false
      },
      u32:0,
      u32:0,
    );
    let (tok, input) = recv_if(tok, input_r, !state.prev_last, empty);

    let state_has_valid_symbol =
      state.prev_count != zero!<bits[COUNT_WIDTH]>();
    let symbols_diff = state.prev_symbol != input.0.symbols[0] &&
      state_has_valid_symbol;

    let overflow =
      (input.0.counts[input.1] as bits[COUNT_WIDTH + u32:1] +
       state.prev_count as bits[COUNT_WIDTH + u32:1])[COUNT_WIDTH as s32:];

    let pack_count = input.0.counts[input.1] + state.prev_count +
      overflow as bits[COUNT_WIDTH];

    let total_pair_count = match(symbols_diff, overflow, state.prev_last) {
      (false, false, false) => input.2,
      _ => input.2 + u32:1,
    };

    // Never true if state.prev_last is set as input.0.last is false
    let combine_last = input.0.last && total_pair_count <= INPUT_WIDTH;

    let (symbols, counts) = match(symbols_diff, overflow) {
      (true, _)      => (
        (state.prev_symbol as bits[SYMBOL_WIDTH][1]) ++ input.0.symbols,
        (state.prev_count as bits[COUNT_WIDTH][1]) ++ input.0.counts,
      ),
      (false, true)  => {
        let (symbols, counts, _) = for (idx, (symbols, counts, input_idx))
          in range(u32:0, INPUT_WIDTH + u32:1) {
          if idx == input.1 {
            (update(symbols, idx, input.0.symbols[input_idx]),
             update(counts, idx, std::unsigned_max_value<COUNT_WIDTH>()),
             input_idx)
          } else if idx == (input.1 + u32:1) {
            (update(symbols, idx, input.0.symbols[input_idx]),
             update(counts, idx, pack_count),
             input_idx + u32:1)
          } else {
            (update(symbols, idx, input.0.symbols[input_idx]),
             update(counts, idx, input.0.counts[input_idx]),
             input_idx + u32:1)
          }
        }((zero!<bits[SYMBOL_WIDTH][INPUT_WIDTH + u32:1]>(),
           zero!<bits[COUNT_WIDTH][INPUT_WIDTH + u32:1]>(),
           u32:0));
        (symbols, counts)
      },
      (false, false) => (
        input.0.symbols ++ bits[SYMBOL_WIDTH]:0 as bits[SYMBOL_WIDTH][1],
        update(input.0.counts, input.1, pack_count) ++
          bits[COUNT_WIDTH]:0 as bits[COUNT_WIDTH][1],
      ),
      _ => (
        zero!<bits[SYMBOL_WIDTH][INPUT_WIDTH + u32:1]>(),
        zero!<bits[COUNT_WIDTH][INPUT_WIDTH + u32:1]>(),
      ),
    };

    let (symbols_to_send, counts_to_send, last_to_send, pair_count_to_send) =
      match (combine_last, state.prev_last) {
        (false, false) => {
          let idx = total_pair_count - u32:1;
          let _symbols = update(symbols, idx, zero!<bits[SYMBOL_WIDTH]>());
          let _counts = update(counts, idx, zero!<bits[COUNT_WIDTH]>());
          (slice(_symbols, u32:0, zero!<bits[SYMBOL_WIDTH][INPUT_WIDTH]>()),
           slice(_counts, u32:0, zero!<bits[COUNT_WIDTH][INPUT_WIDTH]>()),
           false, idx)
        },
        _ => (slice(symbols, u32:0, zero!<bits[SYMBOL_WIDTH][INPUT_WIDTH]>()),
          slice(counts, u32:0, zero!<bits[COUNT_WIDTH][INPUT_WIDTH]>()),
          true, total_pair_count),
    };

    let (symbol_to_state, count_to_state, last_to_state) = {
      let idx = total_pair_count - u32:1;
      match (combine_last, input.0.last) {
        (false, true) => (symbols[idx], counts[idx], true),
        (false, false) => (symbols[idx], counts[idx], false),
        _ => (bits[SYMBOL_WIDTH]:0, bits[COUNT_WIDTH]:0, false),
      }
    };

    let output = (
      EncOutData<SYMBOL_WIDTH, COUNT_WIDTH, INPUT_WIDTH> {
        symbols: symbols_to_send,
        counts: counts_to_send,
        last: last_to_send,
      },
      pair_count_to_send,
    );
    let send = pair_count_to_send > u32:0 || last_to_send;

    let tok = send_if(tok, output_s, send, output);
    RunLengthEncoderAdvancedCoreState{
      prev_symbol: symbol_to_state,
      prev_count: count_to_state,
      prev_last:  last_to_state
    }
  }
}

// Tests

const TEST_COMMON_SYMBOL_WIDTH = u32:32;
const TEST_COMMON_COUNT_WIDTH = u32:32;
const TEST_COMMON_INPUT_WIDTH  = u32:4;

type TestCommonSymbol     = bits[TEST_COMMON_SYMBOL_WIDTH];
type TestCommonCount      = bits[TEST_COMMON_COUNT_WIDTH];

type TestCommonSymbols    = TestCommonSymbol[TEST_COMMON_INPUT_WIDTH];
type TestCommonCounts     = TestCommonCount[TEST_COMMON_INPUT_WIDTH];

type TestCommonStimulus   = (TestCommonSymbols, TestCommonCounts, u32, u32);

type TestCommonEncInData  = EncOutData<
  TEST_COMMON_SYMBOL_WIDTH, TEST_COMMON_COUNT_WIDTH, TEST_COMMON_INPUT_WIDTH>;
type TestCommonEncInTuple = (TestCommonEncInData, u32, u32);

type TestCommonEncOutData = EncOutData<
  TEST_COMMON_SYMBOL_WIDTH, TEST_COMMON_COUNT_WIDTH, TEST_COMMON_INPUT_WIDTH>;
type TestCommonEncOutTuple = (TestCommonEncOutData, u32);

// Transaction with counter overflow
const TEST_OVERFLOW_SYMBOL_WIDTH = u32:32;
const TEST_OVERFLOW_COUNT_WIDTH = u32:2;
const TEST_OVERFLOW_INPUT_WIDTH  = u32:4;

type TestOverflowSymbol     = bits[TEST_OVERFLOW_SYMBOL_WIDTH];
type TestOverflowCount      = bits[TEST_OVERFLOW_COUNT_WIDTH];

type TestOverflowSymbols    = TestOverflowSymbol[TEST_OVERFLOW_INPUT_WIDTH];
type TestOverflowCounts     = TestOverflowCount[TEST_OVERFLOW_INPUT_WIDTH];

type TestOverflowStimulus   = (TestOverflowSymbols, TestOverflowCounts, u32, u32);

type TestOverflowEncInData  = EncOutData<TEST_OVERFLOW_SYMBOL_WIDTH,
  TEST_OVERFLOW_COUNT_WIDTH, TEST_OVERFLOW_INPUT_WIDTH>;
type TestOverflowEncInTuple  = (TestOverflowEncInData, u32, u32);

type TestOverflowEncOutData = EncOutData<TEST_OVERFLOW_SYMBOL_WIDTH,
  TEST_OVERFLOW_COUNT_WIDTH, TEST_OVERFLOW_INPUT_WIDTH>;
type TestOverflowEncOutTuple = (TestOverflowEncOutData, u32);


// Check empty transaction
#[test_proc]
proc PacketOnlyWithLastSet {
  terminator: chan<bool> out;
  enc_input_s: chan<TestCommonEncInTuple> out;
  enc_output_r: chan<TestCommonEncOutTuple> in;
  init{()}
  config(
    terminator: chan<bool> out
  ) {
    let (input_s, input_r) = chan<TestCommonEncInTuple>;
    let (output_s, output_r) = chan<TestCommonEncOutTuple>;
    spawn RunLengthEncoderAdvancedCoreStage<
      TEST_COMMON_SYMBOL_WIDTH, TEST_COMMON_COUNT_WIDTH,
      TEST_COMMON_INPUT_WIDTH>(input_r, output_s);
    (terminator, input_s, output_r)
  }
  next(tok: token, state: ()) {
    let TestCommonTestStimuli: TestCommonStimulus[1] = [
      (TestCommonSymbols: [0x0, 0x0, 0x0, 0x0],
       TestCommonCounts: [0x0, 0x0, 0x0, 0x0],
       u32:0, u32:0),
    ];

    let tok = for ((counter, stimulus), tok):
                  ((u32, TestCommonStimulus) , token)
                  in enumerate(TestCommonTestStimuli) {
      let last = counter == (array_size(TestCommonTestStimuli) - u32:1);
      let _stimulus = (
        TestCommonEncInData{
          symbols: stimulus.0,
          counts: stimulus.1,
          last: last
        }, stimulus.2, stimulus.3
      );
      let tok = send(tok, enc_input_s, _stimulus);
      trace_fmt!("Sent {} stimuli, symbols: {:x}, counts: {:x}, last: {}, propagation: {}, pair_count: {}",
        counter, _stimulus.0.symbols, _stimulus.0.counts, _stimulus.0.last,
        _stimulus.1, _stimulus.2);
      (tok)
    }(tok);

    let TestCommonTestOutput: (TestCommonSymbols, TestCommonCounts, u32)[1] = [
      (TestCommonSymbols: [0x0, 0x0, 0x0, 0x0],
       TestCommonCounts: [0x0, 0x0, 0x0, 0x0], u32:0),
    ];

    let tok = for ((counter, (symbols, counts, pairs)), tok):
       ((u32, (TestCommonSymbols, TestCommonCounts, u32)) , token)
       in enumerate(TestCommonTestOutput) {
      let last = counter == (array_size(TestCommonTestOutput) - u32:1);
      let expected = (
        TestCommonEncOutData{
          symbols: symbols,
          counts: counts,
          last: last
        }, pairs
      );
      let (tok, enc_output) = recv(tok, enc_output_r);
      trace_fmt!(
        "Received {} pairs, symbols: {:x}, counts: {}, last: {}, pairs: {}",
        counter, enc_output.0.symbols, enc_output.0.counts, enc_output.0.last,
        enc_output.1);
      assert_eq(enc_output, expected);
      (tok)
    }(tok);

    send(tok, terminator, true);
  }
}

// Test state interaction with packet that starts with pair which
// has `symbol` equal to RLE's `state.prev_symbol` and doesn't cause
// counter overflow
#[test_proc]
proc CombineWithoutOverflow {
  terminator: chan<bool> out;
  enc_input_s: chan<TestCommonEncInTuple> out;
  enc_output_r: chan<TestCommonEncOutTuple> in;
  init{()}
  config(
    terminator: chan<bool> out
  ) {
    let (input_s, input_r) = chan<TestCommonEncInTuple>;
    let (output_s, output_r) = chan<TestCommonEncOutTuple>;
    spawn RunLengthEncoderAdvancedCoreStage<
      TEST_COMMON_SYMBOL_WIDTH, TEST_COMMON_COUNT_WIDTH,
      TEST_COMMON_INPUT_WIDTH>(input_r, output_s);
    (terminator, input_s, output_r)
  }
  next(tok: token, state: ()) {
    let TestCommonTestStimuli: TestCommonStimulus[2] = [
      (TestCommonSymbols: [0x1, 0x0, 0x0, 0x0],
       TestCommonCounts: [0x1, 0x0, 0x0, 0x0],
       u32:0, u32:1),
      (TestCommonSymbols: [0x1, 0x2, 0x3, 0x4],
       TestCommonCounts: [0x1, 0x1, 0x1, 0x1],
       u32:0, u32:4),
    ];

    let tok = for ((counter, stimulus), tok):
                  ((u32, TestCommonStimulus) , token)
                  in enumerate(TestCommonTestStimuli) {
      let last = counter == (array_size(TestCommonTestStimuli) - u32:1);
      let _stimulus = (
        TestCommonEncInData{
          symbols: stimulus.0,
          counts: stimulus.1,
          last: last
        }, stimulus.2, stimulus.3
      );
      let tok = send(tok, enc_input_s, _stimulus);
      trace_fmt!("Sent {} stimuli, symbols: {:x}, counts: {:x}, last: {}, propagation: {}, pair_count: {}",
        counter, _stimulus.0.symbols, _stimulus.0.counts, _stimulus.0.last,
        _stimulus.1, _stimulus.2);
      (tok)
    }(tok);

    let TestCommonTestOutput: (TestCommonSymbols, TestCommonCounts, u32)[1] = [
      (TestCommonSymbols: [0x1, 0x2, 0x3, 0x4],
       TestCommonCounts: [0x2, 0x1, 0x1, 0x1], u32:4),
    ];

    let tok = for ((counter, (symbols, counts, pairs)), tok):
       ((u32, (TestCommonSymbols, TestCommonCounts, u32)) , token)
       in enumerate(TestCommonTestOutput) {
      let last = counter == (array_size(TestCommonTestOutput) - u32:1);
      let expected = (
        TestCommonEncOutData{
          symbols: symbols,
          counts: counts,
          last: last
        }, pairs
      );
      let (tok, enc_output) = recv(tok, enc_output_r);
      trace_fmt!(
        "Received {} pairs, symbols: {:x}, counts: {}, last: {}, pairs: {}",
        counter, enc_output.0.symbols, enc_output.0.counts, enc_output.0.last,
        enc_output.1);
      assert_eq(enc_output, expected);
      (tok)
    }(tok);

    send(tok, terminator, true);
  }
}

// Test state interaction with packet that starts with pair which
// has `symbol` equal to RLE's `state.prev_symbol` and does cause
// counter overflow
#[test_proc]
proc CombineWithOverflow {
  terminator: chan<bool> out;
  enc_input_s: chan<TestOverflowEncInTuple> out;
  enc_output_r: chan<TestOverflowEncOutTuple> in;
  init{()}
  config(
    terminator: chan<bool> out
  ) {
    let (input_s, input_r) = chan<TestOverflowEncInTuple>;
    let (output_s, output_r) = chan<TestOverflowEncOutTuple>;
    spawn RunLengthEncoderAdvancedCoreStage<
      TEST_OVERFLOW_SYMBOL_WIDTH, TEST_OVERFLOW_COUNT_WIDTH,
      TEST_OVERFLOW_INPUT_WIDTH>(input_r, output_s);
    (terminator, input_s, output_r)
  }
  next(tok: token, state: ()) {
    let TestOverflowTestStimuli: TestOverflowStimulus[2] = [
      (TestOverflowSymbols: [0x1, 0x0, 0x0, 0x0],
       TestOverflowCounts: [0x1, 0x0, 0x0, 0x0],
       u32:0, u32:1),
      (TestOverflowSymbols: [0x1, 0x2, 0x3, 0x4],
       TestOverflowCounts: [0x3, 0x1, 0x1, 0x1],
       u32:0, u32:4),
    ];

    let tok = for ((counter, stimulus), tok):
                  ((u32, TestOverflowStimulus) , token)
                  in enumerate(TestOverflowTestStimuli) {
      let last = counter == (array_size(TestOverflowTestStimuli) - u32:1);
      let _stimulus = (
        TestOverflowEncInData{
          symbols: stimulus.0,
          counts: stimulus.1,
          last: last
        }, stimulus.2, stimulus.3
      );
      let tok = send(tok, enc_input_s, _stimulus);
      trace_fmt!("Sent {} stimuli, symbols: {:x}, counts: {:x}, last: {}, propagation: {}, pair_count: {}",
        counter, _stimulus.0.symbols, _stimulus.0.counts, _stimulus.0.last,
        _stimulus.1, _stimulus.2);
      (tok)
    }(tok);

    let TestOverflowTestOutput: (TestOverflowSymbols, TestOverflowCounts, u32)[2] = [
      (TestOverflowSymbols: [0x1, 0x1, 0x2, 0x3],
       TestOverflowCounts: [0x3, 0x1, 0x1, 0x1], u32:4),
      (TestOverflowSymbols: [0x4, 0x0, 0x0, 0x0],
       TestOverflowCounts: [0x1, 0x0, 0x0, 0x0], u32:1),
    ];

    let tok = for ((counter, (symbols, counts, pairs)), tok):
       ((u32, (TestOverflowSymbols, TestOverflowCounts, u32)) , token)
       in enumerate(TestOverflowTestOutput) {
      let last = counter == (array_size(TestOverflowTestOutput) - u32:1);
      let expected = (
        TestOverflowEncOutData{
          symbols: symbols,
          counts: counts,
          last: last
        }, pairs
      );
      let (tok, enc_output) = recv(tok, enc_output_r);
      trace_fmt!(
        "Received {} pairs, symbols: {:x}, counts: {}, last: {}, pairs: {}",
        counter, enc_output.0.symbols, enc_output.0.counts, enc_output.0.last,
        enc_output.1);
      assert_eq(enc_output, expected);
      (tok)
    }(tok);

    send(tok, terminator, true);
  }
}

// Check that state is correctly updated and contains lastest
// symbol, count pair
#[test_proc]
proc CombineAfterStateChange {
  terminator: chan<bool> out;
  enc_input_s: chan<TestCommonEncInTuple> out;
  enc_output_r: chan<TestCommonEncOutTuple> in;
  init{()}
  config(
    terminator: chan<bool> out
  ) {
    let (input_s, input_r) = chan<TestCommonEncInTuple>;
    let (output_s, output_r) = chan<TestCommonEncOutTuple>;
    spawn RunLengthEncoderAdvancedCoreStage<
      TEST_COMMON_SYMBOL_WIDTH, TEST_COMMON_COUNT_WIDTH,
      TEST_COMMON_INPUT_WIDTH>(input_r, output_s);
    (terminator, input_s, output_r)
  }
  next(tok: token, state: ()) {
    let TestCommonTestStimuli: TestCommonStimulus[3] = [
      (TestCommonSymbols: [0x1, 0x0, 0x0, 0x0],
       TestCommonCounts: [0x1, 0x0, 0x0, 0x0],
       u32:0, u32:1),
      (TestCommonSymbols: [0x2, 0x3, 0x4, 0x5],
       TestCommonCounts: [0x1, 0x1, 0x1, 0x1],
       u32:0, u32:4),
      (TestCommonSymbols: [0x5, 0x0, 0x0, 0x0],
       TestCommonCounts: [0x1, 0x0, 0x0, 0x0],
       u32:0, u32:1),
    ];

    let tok = for ((counter, stimulus), tok):
                  ((u32, TestCommonStimulus) , token)
                  in enumerate(TestCommonTestStimuli) {
      let last = counter == (array_size(TestCommonTestStimuli) - u32:1);
      let _stimulus = (
        TestCommonEncInData{
          symbols: stimulus.0,
          counts: stimulus.1,
          last: last
        }, stimulus.2, stimulus.3
      );
      let tok = send(tok, enc_input_s, _stimulus);
      trace_fmt!("Sent {} stimuli, symbols: {:x}, counts: {:x}, last: {}, propagation: {}, pair_count: {}",
        counter, _stimulus.0.symbols, _stimulus.0.counts, _stimulus.0.last,
        _stimulus.1, _stimulus.2);
      (tok)
    }(tok);

    let TestCommonTestOutput: (TestCommonSymbols, TestCommonCounts, u32)[2] = [
      (TestCommonSymbols: [0x1, 0x2, 0x3, 0x4],
       TestCommonCounts: [0x1, 0x1, 0x1, 0x1], u32:4),
      (TestCommonSymbols: [0x5, 0x0, 0x0, 0x0],
       TestCommonCounts: [0x2, 0x0, 0x0, 0x0], u32:1),
    ];

    let tok = for ((counter, (symbols, counts, pairs)), tok):
       ((u32, (TestCommonSymbols, TestCommonCounts, u32)) , token)
       in enumerate(TestCommonTestOutput) {
      let last = counter == (array_size(TestCommonTestOutput) - u32:1);
      let expected = (
        TestCommonEncOutData{
          symbols: symbols,
          counts: counts,
          last: last
        }, pairs
      );
      let (tok, enc_output) = recv(tok, enc_output_r);
      trace_fmt!(
        "Received {} pairs, symbols: {:x}, counts: {}, last: {}, pairs: {}",
        counter, enc_output.0.symbols, enc_output.0.counts, enc_output.0.last,
        enc_output.1);
      assert_eq(enc_output, expected);
      (tok)
    }(tok);

    send(tok, terminator, true);
  }
}

// Check last_combine is correctly used
#[test_proc]
proc CombineStateWithLastPacket {
  terminator: chan<bool> out;
  enc_input_s: chan<TestCommonEncInTuple> out;
  enc_output_r: chan<TestCommonEncOutTuple> in;
  init{()}
  config(
    terminator: chan<bool> out
  ) {
    let (input_s, input_r) = chan<TestCommonEncInTuple>;
    let (output_s, output_r) = chan<TestCommonEncOutTuple>;
    spawn RunLengthEncoderAdvancedCoreStage<
      TEST_COMMON_SYMBOL_WIDTH, TEST_COMMON_COUNT_WIDTH,
      TEST_COMMON_INPUT_WIDTH>(input_r, output_s);
    (terminator, input_s, output_r)
  }
  next(tok: token, state: ()) {
    let TestCommonTestStimuli: TestCommonStimulus[2] = [
      (TestCommonSymbols: [0x1, 0x0, 0x0, 0x0],
       TestCommonCounts: [0x1, 0x0, 0x0, 0x0],
       u32:0, u32:1),
      (TestCommonSymbols: [0x2, 0x3, 0x4, 0x0],
       TestCommonCounts: [0x1, 0x1, 0x1, 0x0],
       u32:0, u32:3),
    ];

    let tok = for ((counter, stimulus), tok):
                  ((u32, TestCommonStimulus) , token)
                  in enumerate(TestCommonTestStimuli) {
      let last = counter == (array_size(TestCommonTestStimuli) - u32:1);
      let _stimulus = (
        TestCommonEncInData{
          symbols: stimulus.0,
          counts: stimulus.1,
          last: last
        }, stimulus.2, stimulus.3
      );
      let tok = send(tok, enc_input_s, _stimulus);
      trace_fmt!("Sent {} stimuli, symbols: {:x}, counts: {:x}, last: {}, propagation: {}, pair_count: {}",
        counter, _stimulus.0.symbols, _stimulus.0.counts, _stimulus.0.last,
        _stimulus.1, _stimulus.2);
      (tok)
    }(tok);

    let TestCommonTestOutput: (TestCommonSymbols, TestCommonCounts, u32)[1] = [
      (TestCommonSymbols: [0x1, 0x2, 0x3, 0x4],
       TestCommonCounts: [0x1, 0x1, 0x1, 0x1], u32:4),
    ];

    let tok = for ((counter, (symbols, counts, pairs)), tok):
       ((u32, (TestCommonSymbols, TestCommonCounts, u32)) , token)
       in enumerate(TestCommonTestOutput) {
      let last = counter == (array_size(TestCommonTestOutput) - u32:1);
      let expected = (
        TestCommonEncOutData{
          symbols: symbols,
          counts: counts,
          last: last
        }, pairs
      );
      let (tok, enc_output) = recv(tok, enc_output_r);
      trace_fmt!(
        "Received {} pairs, symbols: {:x}, counts: {}, last: {}, pairs: {}",
        counter, enc_output.0.symbols, enc_output.0.counts, enc_output.0.last,
        enc_output.1);
      assert_eq(enc_output, expected);
      (tok)
    }(tok);

    send(tok, terminator, true);
  }
}

// Check that state is correctly cleared after `last` is seen
#[test_proc]
proc NoStateSipllAfterLast {
  terminator: chan<bool> out;
  enc_input_s: chan<TestCommonEncInTuple> out;
  enc_output_r: chan<TestCommonEncOutTuple> in;
  init{()}
  config(
    terminator: chan<bool> out
  ) {
    let (input_s, input_r) = chan<TestCommonEncInTuple>;
    let (output_s, output_r) = chan<TestCommonEncOutTuple>;
    spawn RunLengthEncoderAdvancedCoreStage<
      TEST_COMMON_SYMBOL_WIDTH, TEST_COMMON_COUNT_WIDTH,
      TEST_COMMON_INPUT_WIDTH>(input_r, output_s);
    (terminator, input_s, output_r)
  }
  next(tok: token, state: ()) {
    let TestCommonTestStimuli: TestCommonEncInTuple[2] = [
      (TestCommonEncInData {
        symbols: TestCommonSymbols:[0x1, 0x0, 0x0, 0x0],
        counts: TestCommonCounts:[0x1, 0x0, 0x0, 0x0],
        last: true
      }, u32:0, u32:1),
      (TestCommonEncInData {
        symbols: TestCommonSymbols:[0x1, 0x0, 0x0, 0x0],
        counts: TestCommonCounts:[0x1, 0x0, 0x0, 0x0],
        last: true
      }, u32:0, u32:1),
    ];

    let tok = for ((counter, stimulus), tok):
                  ((u32, TestCommonEncInTuple) , token)
                  in enumerate(TestCommonTestStimuli) {
      let tok = send(tok, enc_input_s, stimulus);
      trace_fmt!("Sent {} stimuli, symbols: {:x}, counts: {:x}, last: {}, propagation: {}, pair_count: {}",
        counter, stimulus.0.symbols, stimulus.0.counts, stimulus.0.last,
        stimulus.1, stimulus.2);
      (tok)
    }(tok);

    let TestCommonTestOutput: TestCommonEncOutTuple[2] = [
      (TestCommonEncOutData {
        symbols: [0x1, 0x0, 0x0, 0x0],
        counts: [0x1, 0x0, 0x0, 0x0],
        last: true,
      }, u32:1),
      (TestCommonEncOutData {
        symbols: [0x1, 0x0, 0x0, 0x0],
        counts: [0x1, 0x0, 0x0, 0x0],
        last: true,
      }, u32:1),
    ];

    let tok = for ((counter, expected), tok):
       ((u32, TestCommonEncOutTuple) , token)
       in enumerate(TestCommonTestOutput) {
      let (tok, enc_output) = recv(tok, enc_output_r);
      trace_fmt!(
        "Received {} pairs, symbols: {:x}, counts: {}, last: {}, pairs: {}",
        counter, enc_output.0.symbols, enc_output.0.counts, enc_output.0.last,
        enc_output.1);
      assert_eq(enc_output, expected);
      (tok)
    }(tok);

    send(tok, terminator, true);
  }
}
