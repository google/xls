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

struct RunLengthEncoderAdvancedAdjustWidthStageState<
  SYMBOL_WIDTH:u32, COUNT_WIDTH:u32, PAIR_COUNT:u32> {

  stored_pairs: (bits[SYMBOL_WIDTH], bits[COUNT_WIDTH])[PAIR_COUNT],
  stored_pairs_count: u32,
  stored_last: bool,
}

pub proc RunLengthEncoderAdvancedAdjustWidthStage<
  SYMBOL_WIDTH: u32, COUNT_WIDTH: u32, INPUT_WIDTH:u32, OUTPUT_WIDTH:u32> {

  input_r: chan<(EncOutData<SYMBOL_WIDTH, COUNT_WIDTH, INPUT_WIDTH>, u32)> in;
  output_s: chan<EncOutData<SYMBOL_WIDTH, COUNT_WIDTH, OUTPUT_WIDTH>> out;

  init {(
    RunLengthEncoderAdvancedAdjustWidthStageState<
      SYMBOL_WIDTH, COUNT_WIDTH, INPUT_WIDTH> {
        stored_pairs: zero!<(bits[SYMBOL_WIDTH], bits[COUNT_WIDTH])[INPUT_WIDTH]>(),
        stored_pairs_count: u32:0,
        stored_last: false,
    }
  )}
  config (
    input_r: chan<(EncOutData<SYMBOL_WIDTH, COUNT_WIDTH, INPUT_WIDTH>, u32)> in,
    output_s: chan<EncOutData<SYMBOL_WIDTH, COUNT_WIDTH, OUTPUT_WIDTH>> out,
  ) {(input_r, output_s)}

  next (tok: token, state: RunLengthEncoderAdvancedAdjustWidthStageState<
      SYMBOL_WIDTH, COUNT_WIDTH, INPUT_WIDTH>) {
    let recv_next_portion = !state.stored_last &&
      (state.stored_pairs_count <= OUTPUT_WIDTH);
    let empty_input = (
      EncOutData<SYMBOL_WIDTH, COUNT_WIDTH, INPUT_WIDTH> {
        symbols: zero!<bits[SYMBOL_WIDTH][INPUT_WIDTH]>(),
        counts:  zero!<bits[COUNT_WIDTH][INPUT_WIDTH]>(),
        last:    false
      },
      u32:0,
    );
    let (tok, input) = recv_if(tok, input_r, recv_next_portion, empty_input);

    let (pairs_to_send, pairs_to_send_count, input_processed_count) =
      for (idx, (pairs, pairs_count, input_count)) in range(u32:0, OUTPUT_WIDTH) {
        if pairs_count < state.stored_pairs_count {
          (update(pairs, pairs_count, state.stored_pairs[pairs_count]),
            pairs_count + u32:1, input_count)
        } else if input_count < input.1 {
          let input_pair = (input.0.symbols[input_count],
            input.0.counts[input_count]);
          (update(pairs, pairs_count, input_pair),
            pairs_count + u32:1, input_count + u32:1)
        } else {
          (pairs, pairs_count, input_count)
        }
    } ((zero!<(bits[SYMBOL_WIDTH], bits[COUNT_WIDTH])[OUTPUT_WIDTH]>(),
        u32:0, u32:0));

    // If we recv data, all stored pairs are being send
    let (pairs_to_state, pairs_count_to_state) = if recv_next_portion {
      for (idx, (pairs, pairs_count)) in range(u32:0, INPUT_WIDTH) {
        if input_processed_count <= idx && idx < input.1 {
          (update(pairs, pairs_count,
            (input.0.symbols[idx], input.0.counts[idx])),
           pairs_count + u32:1)
        } else {
          (pairs, pairs_count)
        }
      }((zero!<(bits[SYMBOL_WIDTH], bits[COUNT_WIDTH])[INPUT_WIDTH]>(), u32:0))
    } else {
      for (idx, (pairs, pairs_count)) in range(u32:0, INPUT_WIDTH){
        if pairs_to_send_count <= idx && idx < state.stored_pairs_count {
          (
            update(pairs, pairs_count, state.stored_pairs[idx]),
            pairs_count + u32:1
          )
        } else {
          (pairs, pairs_count)
        }
      } ((state.stored_pairs, u32:0))
    };

    let last_to_send  = (state.stored_last || input.0.last) &&
      (pairs_count_to_state == u32:0);
    let last_to_state = (state.stored_last || input.0.last) &&
      (pairs_count_to_state != u32:0);

    let (symbols_to_send, counts_to_send) =
      for (idx, (symbols, counts)) in range(u32:0, OUTPUT_WIDTH) {
        let pair = pairs_to_send[idx];
        (update(symbols, idx, pair.0), update(counts, idx, pair.1))
    }((zero!<bits[SYMBOL_WIDTH][OUTPUT_WIDTH]>(),
       zero!<bits[COUNT_WIDTH][OUTPUT_WIDTH]>()));

    let output = EncOutData<SYMBOL_WIDTH, COUNT_WIDTH, OUTPUT_WIDTH> {
      symbols: symbols_to_send,
      counts: counts_to_send,
      last: last_to_send,
    };
    let tok = send(tok, output_s, output);

    RunLengthEncoderAdvancedAdjustWidthStageState<
      SYMBOL_WIDTH, COUNT_WIDTH, INPUT_WIDTH> {
        stored_pairs: pairs_to_state,
        stored_pairs_count: pairs_count_to_state,
        stored_last: last_to_state,
    }
  }
}

const TEST_COMMON_SYMBOL_WIDTH = u32:32;
const TEST_COMMON_COUNT_WIDTH  = u32:32;
const TEST_COMMON_INPUT_WIDTH  = u32:4;
const TEST_COMMON_OUTPUT_WIDTH = u32:2;

type TestCommonSymbol     = bits[TEST_COMMON_SYMBOL_WIDTH];
type TestCommonCount      = bits[TEST_COMMON_COUNT_WIDTH];

type TestCommonSymbolsIn  = TestCommonSymbol[TEST_COMMON_INPUT_WIDTH];
type TestCommonCountsIn   = TestCommonCount[TEST_COMMON_INPUT_WIDTH];
type TestCommonSymbolsOut = TestCommonSymbol[TEST_COMMON_OUTPUT_WIDTH];
type TestCommonCountsOut  = TestCommonCount[TEST_COMMON_OUTPUT_WIDTH];

type TestCommonStimulus   = (TestCommonSymbolsIn, TestCommonCountsIn, u32);

type TestCommonEncInData  = EncOutData<TEST_COMMON_SYMBOL_WIDTH,
  TEST_COMMON_COUNT_WIDTH, TEST_COMMON_INPUT_WIDTH>;
type TestCommonEncInTuple = (TestCommonEncInData, u32);

type TestCommonEncOutData = EncOutData<
  TEST_COMMON_SYMBOL_WIDTH, TEST_COMMON_COUNT_WIDTH, TEST_COMMON_OUTPUT_WIDTH>;

// Test simple case, where only last is set.
// Check handling of empty transaction.
#[test_proc]
proc PacketContainsOnlyLast {
  terminator: chan<bool> out;
  enc_input_s: chan<TestCommonEncInTuple> out;
  enc_output_r: chan<TestCommonEncOutData> in;
  init{()}
  config(terminator: chan<bool> out) {
    let (input_s, input_r) = chan<TestCommonEncInTuple>;
    let (output_s, output_r) = chan<TestCommonEncOutData>;
    spawn RunLengthEncoderAdvancedAdjustWidthStage<
      TEST_COMMON_SYMBOL_WIDTH, TEST_COMMON_COUNT_WIDTH,
      TEST_COMMON_INPUT_WIDTH, TEST_COMMON_OUTPUT_WIDTH>(input_r, output_s);
    (terminator, input_s, output_r)
  }
  next(tok:token, state:()) {
    let TestCommonTestStimuli: TestCommonStimulus[1] = [
      (TestCommonSymbolsIn:[0x1, 0x2, 0x3, 0x4],
       TestCommonCountsIn:[0x0, 0x0, 0x0, 0x0],
       u32:0),
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
        }, stimulus.2,
      );
      let tok = send(tok, enc_input_s, _stimulus);
      trace_fmt!("Sent {} stimuli, symbols: {:x}, counts: {:x}, last: {}, pair_count: {}",
        counter, _stimulus.0.symbols, _stimulus.0.counts, _stimulus.0.last, _stimulus.1);
      (tok)
    }(tok);

    let TestCommonTestOutput: (TestCommonSymbolsOut, TestCommonCountsOut)[1] = [
      ([TestCommonSymbol:0x0, TestCommonSymbol:0x0],
       [TestCommonCount:0x0, TestCommonCount:0x0]),
    ];
    let tok = for ((counter, (symbols, counts)), tok):
       ((u32, (TestCommonSymbolsOut, TestCommonCountsOut)) , token)
       in enumerate(TestCommonTestOutput) {
      let last = counter == (array_size(TestCommonTestOutput) - u32:1);
      let expected = TestCommonEncOutData{
        symbols: symbols,
        counts: counts,
        last: last
      };
      let (tok, enc_output) = recv(tok, enc_output_r);
      trace_fmt!(
        "Received {} pairs, symbols: {:x}, counts: {}, last: {}",
        counter, enc_output.symbols, enc_output.counts, enc_output.last);
      assert_eq(enc_output, expected);
      (tok)
    }(tok);

    send(tok, terminator, true);
    ()
  }
}

// Test simple case, where number of incoming pairs <=
// max output width
#[test_proc]
proc InputPairCountLessOrEqualOutputWidth {
  terminator: chan<bool> out;
  enc_input_s: chan<TestCommonEncInTuple> out;
  enc_output_r: chan<TestCommonEncOutData> in;
  init{()}
  config(terminator: chan<bool> out) {
    let (input_s, input_r) = chan<TestCommonEncInTuple>;
    let (output_s, output_r) = chan<TestCommonEncOutData>;
    spawn RunLengthEncoderAdvancedAdjustWidthStage<
    TEST_COMMON_SYMBOL_WIDTH, TEST_COMMON_COUNT_WIDTH,
    TEST_COMMON_INPUT_WIDTH, TEST_COMMON_OUTPUT_WIDTH>(input_r, output_s);
    (terminator, input_s, output_r)
  }
  next(tok:token, state:()) {
    let TestCommonTestStimuli: TestCommonStimulus[1] = [
      (TestCommonSymbolsIn:[0x1, 0x2, 0x3, 0x4],
       TestCommonCountsIn:[0x1, 0x1, 0x0, 0x0],
       u32:2),
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
        }, stimulus.2,
      );
      let tok = send(tok, enc_input_s, _stimulus);
      trace_fmt!("Sent {} stimuli, symbols: {:x}, counts: {:x}, last: {}, pair_count: {}",
        counter, _stimulus.0.symbols, _stimulus.0.counts, _stimulus.0.last, _stimulus.1);
      (tok)
    }(tok);

    let TestCommonTestOutput: (TestCommonSymbolsOut, TestCommonCountsOut)[1] = [
      ([TestCommonSymbol:0x1, TestCommonSymbol:0x2],
       [TestCommonCount:0x1, TestCommonCount:0x1]),
    ];
    let tok = for ((counter, (symbols, counts)), tok):
       ((u32, (TestCommonSymbolsOut, TestCommonCountsOut)) , token)
       in enumerate(TestCommonTestOutput) {
      let last = counter == (array_size(TestCommonTestOutput) - u32:1);
      let expected = TestCommonEncOutData{
        symbols: symbols,
        counts: counts,
        last: last
      };
      let (tok, enc_output) = recv(tok, enc_output_r);
      trace_fmt!(
        "Received {} pairs, symbols: {:x}, counts: {}, last: {}",
        counter, enc_output.symbols, enc_output.counts, enc_output.last);
      assert_eq(enc_output, expected);
      (tok)
    }(tok);

    send(tok, terminator, true);
    ()
  }
}

// Test simple case, where number of incoming pairs = input width
// It checks serialization
#[test_proc]
proc InputFullyFilled {
  terminator: chan<bool> out;
  enc_input_s: chan<TestCommonEncInTuple> out;
  enc_output_r: chan<TestCommonEncOutData> in;
  init{()}
  config(terminator: chan<bool> out) {
    let (input_s, input_r) = chan<TestCommonEncInTuple>;
    let (output_s, output_r) = chan<TestCommonEncOutData>;
    spawn RunLengthEncoderAdvancedAdjustWidthStage<
    TEST_COMMON_SYMBOL_WIDTH, TEST_COMMON_COUNT_WIDTH,
    TEST_COMMON_INPUT_WIDTH, TEST_COMMON_OUTPUT_WIDTH>(input_r, output_s);
    (terminator, input_s, output_r)
  }
  next(tok:token, state:()) {
    let TestCommonTestStimuli: TestCommonStimulus[1] = [
      (TestCommonSymbolsIn:[0x1, 0x2, 0x3, 0x4],
       TestCommonCountsIn:[0x1, 0x1, 0x1, 0x1],
       u32:4),
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
        }, stimulus.2,
      );
      let tok = send(tok, enc_input_s, _stimulus);
      trace_fmt!("Sent {} stimuli, symbols: {:x}, counts: {:x}, last: {}, pair_count: {}",
        counter, _stimulus.0.symbols, _stimulus.0.counts, _stimulus.0.last, _stimulus.1);
      (tok)
    }(tok);

    let TestCommonTestOutput: (TestCommonSymbolsOut, TestCommonCountsOut)[2] = [
      (TestCommonSymbolsOut:[0x1, 0x2],
       TestCommonCountsOut:[0x1, 0x1]),
      (TestCommonSymbolsOut:[0x3, 0x4],
       TestCommonCountsOut:[0x1, 0x1]),
    ];
    let tok = for ((counter, (symbols, counts)), tok):
       ((u32, (TestCommonSymbolsOut, TestCommonCountsOut)) , token)
       in enumerate(TestCommonTestOutput) {
      let last = counter == (array_size(TestCommonTestOutput) - u32:1);
      let expected = TestCommonEncOutData{
        symbols: symbols,
        counts: counts,
        last: last
      };
      let (tok, enc_output) = recv(tok, enc_output_r);
      trace_fmt!(
        "Received {} pairs, symbols: {:x}, counts: {}, last: {}",
        counter, enc_output.symbols, enc_output.counts, enc_output.last);
      assert_eq(enc_output, expected);
      (tok)
    }(tok);

    send(tok, terminator, true);
    ()
  }
}

// Send 2-packet ransaction, where first packet has 3 pairs, and second one has
// single pair. Checks that only 2 output packets are produced.
#[test_proc]
proc CombineStateWithNextInputPairs {
  terminator: chan<bool> out;
  enc_input_s: chan<TestCommonEncInTuple> out;
  enc_output_r: chan<TestCommonEncOutData> in;
  init{()}
  config(terminator: chan<bool> out) {
    let (input_s, input_r) = chan<TestCommonEncInTuple>;
    let (output_s, output_r) = chan<TestCommonEncOutData>;
    spawn RunLengthEncoderAdvancedAdjustWidthStage<
    TEST_COMMON_SYMBOL_WIDTH, TEST_COMMON_COUNT_WIDTH,
    TEST_COMMON_INPUT_WIDTH, TEST_COMMON_OUTPUT_WIDTH>(input_r, output_s);
    (terminator, input_s, output_r)
  }
  next(tok:token, state:()) {
    let TestCommonTestStimuli: TestCommonStimulus[2] = [
      (TestCommonSymbolsIn:[0x1, 0x2, 0x3, 0x4],
       TestCommonCountsIn:[0x1, 0x1, 0x1, 0x0],
       u32:3),
      (TestCommonSymbolsIn:[0x4, 0x2, 0x3, 0x4],
       TestCommonCountsIn:[0x1, 0x0, 0x0, 0x0],
       u32:1),
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
        }, stimulus.2,
      );
      let tok = send(tok, enc_input_s, _stimulus);
      trace_fmt!("Sent {} stimuli, symbols: {:x}, counts: {:x}, last: {}, pair_count: {}",
        counter, _stimulus.0.symbols, _stimulus.0.counts, _stimulus.0.last, _stimulus.1);
      (tok)
    }(tok);

    let TestCommonTestOutput: (TestCommonSymbolsOut, TestCommonCountsOut)[2] = [
      ([TestCommonSymbol:0x1, TestCommonSymbol:0x2],
       [TestCommonCount:0x1, TestCommonCount:0x1]),
      ([TestCommonSymbol:0x3, TestCommonSymbol:0x4],
       [TestCommonCount:0x1, TestCommonCount:0x1]),
    ];
    let tok = for ((counter, (symbols, counts)), tok):
       ((u32, (TestCommonSymbolsOut, TestCommonCountsOut)) , token)
       in enumerate(TestCommonTestOutput) {
      let last = counter == (array_size(TestCommonTestOutput) - u32:1);
      let expected = TestCommonEncOutData{
        symbols: symbols,
        counts: counts,
        last: last
      };
      let (tok, enc_output) = recv(tok, enc_output_r);
      trace_fmt!(
        "Received {} pairs, symbols: {:x}, counts: {}, last: {}",
        counter, enc_output.symbols, enc_output.counts, enc_output.last);
      assert_eq(enc_output, expected);
      (tok)
    }(tok);

    send(tok, terminator, true);
    ()
  }
}

// Send 2-packet ransaction, where first packet has 3 pairs, and second one has
// 4 pairs. Checks that only 4 output packets are produced.
#[test_proc]
proc PairStateCombineWithStateUpdate {
  terminator: chan<bool> out;
  enc_input_s: chan<TestCommonEncInTuple> out;
  enc_output_r: chan<TestCommonEncOutData> in;
  init{()}
  config(terminator: chan<bool> out) {
    let (input_s, input_r) = chan<TestCommonEncInTuple>;
    let (output_s, output_r) = chan<TestCommonEncOutData>;
    spawn RunLengthEncoderAdvancedAdjustWidthStage<
    TEST_COMMON_SYMBOL_WIDTH, TEST_COMMON_COUNT_WIDTH,
    TEST_COMMON_INPUT_WIDTH, TEST_COMMON_OUTPUT_WIDTH>(input_r, output_s);
    (terminator, input_s, output_r)
  }
  next(tok:token, state:()) {
    let TestCommonTestStimuli: TestCommonStimulus[2] = [
      (TestCommonSymbolsIn:[0x1, 0x2, 0x3, 0x4],
       TestCommonCountsIn:[0x1, 0x1, 0x1, 0x0],
       u32:3),
      (TestCommonSymbolsIn:[0x4, 0x5, 0x6, 0x7],
       TestCommonCountsIn:[0x1, 0x1, 0x1, 0x1],
       u32:4),
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
        }, stimulus.2,
      );
      let tok = send(tok, enc_input_s, _stimulus);
      trace_fmt!("Sent {} stimuli, symbols: {:x}, counts: {:x}, last: {}, pair_count: {}",
        counter, _stimulus.0.symbols, _stimulus.0.counts, _stimulus.0.last, _stimulus.1);
      (tok)
    }(tok);

    let TestCommonTestOutput: (TestCommonSymbolsOut, TestCommonCountsOut)[4] = [
      (TestCommonSymbolsOut:[0x1, 0x2],
       TestCommonCountsOut:[0x1, 0x1]),
      (TestCommonSymbolsOut:[0x3, 0x4],
       TestCommonCountsOut:[0x1, 0x1]),
      (TestCommonSymbolsOut:[0x5, 0x6],
       TestCommonCountsOut:[0x1, 0x1]),
      (TestCommonSymbolsOut:[0x7, 0x0],
       TestCommonCountsOut:[0x1, 0x0]),
    ];
    let tok = for ((counter, (symbols, counts)), tok):
       ((u32, (TestCommonSymbolsOut, TestCommonCountsOut)) , token)
       in enumerate(TestCommonTestOutput) {
      let last = counter == (array_size(TestCommonTestOutput) - u32:1);
      let expected = TestCommonEncOutData{
        symbols: symbols,
        counts: counts,
        last: last
      };
      let (tok, enc_output) = recv(tok, enc_output_r);
      trace_fmt!(
        "Received {} pairs, symbols: {:x}, counts: {}, last: {}",
        counter, enc_output.symbols, enc_output.counts, enc_output.last);
      assert_eq(enc_output, expected);
      (tok)
    }(tok);

    send(tok, terminator, true);
    ()
  }
}

// Send 2 transactions, where first one has 3 pairs, and second one has
// single pair. Checks that 2 transactions will be created, first one with 2
// output packets and second one with single packet.
#[test_proc]
proc NoStateSipllAfterLast {
  terminator: chan<bool> out;
  enc_input_s: chan<TestCommonEncInTuple> out;
  enc_output_r: chan<TestCommonEncOutData> in;
  init{()}
  config(terminator: chan<bool> out) {
    let (input_s, input_r) = chan<TestCommonEncInTuple>;
    let (output_s, output_r) = chan<TestCommonEncOutData>;
    spawn RunLengthEncoderAdvancedAdjustWidthStage<
    TEST_COMMON_SYMBOL_WIDTH, TEST_COMMON_COUNT_WIDTH,
    TEST_COMMON_INPUT_WIDTH, TEST_COMMON_OUTPUT_WIDTH>(input_r, output_s);
    (terminator, input_s, output_r)
  }
  next(tok:token, state:()) {
    let TestCommonTestStimuli: TestCommonEncInTuple[2] = [
      (TestCommonEncInData{
        symbols:[0x1, 0x2, 0x3, 0x4],
        counts:[0x1, 0x1, 0x1, 0x0],
        last:true
      }, u32:3),
      (TestCommonEncInData{
        symbols:[0x4, 0x2, 0x3, 0x4],
        counts:[0x1, 0x0, 0x0, 0x0],
        last:true
      }, u32:1),
    ];
    let tok = for ((counter, stimulus), tok):
                  ((u32, TestCommonEncInTuple) , token)
                  in enumerate(TestCommonTestStimuli) {
      let tok = send(tok, enc_input_s, stimulus);
      trace_fmt!("Sent {} stimuli, symbols: {:x}, counts: {:x}, last: {}, pair_count: {}",
        counter, stimulus.0.symbols, stimulus.0.counts, stimulus.0.last, stimulus.1);
      (tok)
    }(tok);

    let TestCommonTestOutput: TestCommonEncOutData[3] = [
      TestCommonEncOutData {
        symbols: [0x1, 0x2],
        counts: [0x1, 0x1],
        last: false,
      },
      TestCommonEncOutData {
        symbols: [0x3, 0x0],
        counts: [0x1, 0x0],
        last: true,
      },
      TestCommonEncOutData {
        symbols: [0x4, 0x0],
        counts: [0x1, 0x0],
        last: true
      },
    ];
    let tok = for ((counter, expected), tok):
       ((u32, TestCommonEncOutData) , token)
       in enumerate(TestCommonTestOutput) {
      let (tok, enc_output) = recv(tok, enc_output_r);
      trace_fmt!(
        "Received {} pairs, symbols: {:x}, counts: {}, last: {}",
        counter, enc_output.symbols, enc_output.counts, enc_output.last);
      assert_eq(enc_output, expected);
      (tok)
    }(tok);

    send(tok, terminator, true);
    ()
  }
}
