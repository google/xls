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

pub proc RunLengthEncoderAdvancedRealignStage<
  SYMBOL_WIDTH: u32, COUNT_WIDTH: u32,INPUT_WIDTH:u32> {

  input_r: chan<EncOutData<SYMBOL_WIDTH, COUNT_WIDTH, INPUT_WIDTH>> in;
  output_s: chan<(EncOutData<SYMBOL_WIDTH, COUNT_WIDTH, INPUT_WIDTH>, u32, u32)> out;

  init{()}
  config(
    input_r: chan<EncOutData<SYMBOL_WIDTH, COUNT_WIDTH, INPUT_WIDTH>> in,
    output_s: chan<(EncOutData<SYMBOL_WIDTH, COUNT_WIDTH, INPUT_WIDTH>, u32, u32)> out,
  ) {(input_r, output_s)}

  next (tok:token, state: ()) {
    let (tok, input) = recv(tok, input_r);
    let (symbols, counts, end) =
      for (idx, (symbols, counts, insert_place)) in range(u32:0, INPUT_WIDTH) {
        if input.counts[idx] != bits[COUNT_WIDTH]:0 {
          (
            update(symbols, insert_place, input.symbols[idx]),
            update(counts, insert_place, input.counts[idx]),
            insert_place + u32:1
          )
        } else {
          (symbols, counts, insert_place)
        }
    } ((zero!<bits[SYMBOL_WIDTH][INPUT_WIDTH]>(),
        zero!<bits[COUNT_WIDTH][INPUT_WIDTH]>(),
        u32:0));
    let output = EncOutData<SYMBOL_WIDTH, COUNT_WIDTH, INPUT_WIDTH> {
      symbols: symbols,
      counts: counts,
      last: input.last,
    };
    let (prop_count, _, _) =
      for (idx, (p_count, symbol, valid)) in range(u32:1, INPUT_WIDTH) {
        let symbol_eq = symbols[idx] == symbol;
        match (valid, symbol_eq) {
          (false, _) => (p_count, symbol, valid),
          (true, true) => (p_count + u32:1, symbol, valid),
          (true, false) => (p_count, symbol, false),
          _ => (u32:0, bits[SYMBOL_WIDTH]:0, false),
        }
    } ((u32:0, symbols[u32:0], counts[u32:0] != bits[COUNT_WIDTH]:0));
    send(tok, output_s, (output, prop_count, end));
  }
}

// Test RealignStage

// Transaction without overflow
const TEST_COMMON_SYMBOL_WIDTH = u32:32;
const TEST_COMMON_COUNT_WIDTH  = u32:32;
const TEST_COMMON_SYMBOL_COUNT = u32:4;

type TestCommonSymbol     = bits[TEST_COMMON_SYMBOL_WIDTH];
type TestCommonCount      = bits[TEST_COMMON_COUNT_WIDTH];

type TestCommonSymbols    = TestCommonSymbol[TEST_COMMON_SYMBOL_COUNT];
type TestCommonCounts     = TestCommonCount[TEST_COMMON_SYMBOL_COUNT];

type TestCommonStimulus   = (TestCommonSymbols, TestCommonCounts);

type TestCommonEncData =  EncOutData<
  TEST_COMMON_SYMBOL_WIDTH, TEST_COMMON_COUNT_WIDTH, TEST_COMMON_SYMBOL_COUNT>;

type TestCommonEncInData  = TestCommonEncData;
type TestCommonEncOutData = (TestCommonEncData, u32, u32);

// Transaction with counter overflow
const OVERFLOW_COUNT_WIDTH  = u32:2;

type OverflowSymbol     = TestCommonSymbol;
type OverflowCount      = bits[OVERFLOW_COUNT_WIDTH];

type OverflowSymbols    = OverflowSymbol[TEST_COMMON_SYMBOL_COUNT];
type OverflowCounts     = OverflowCount[TEST_COMMON_SYMBOL_COUNT];

type OverflowStimulus   = (TestCommonSymbols, OverflowCounts);

type OverflowEncData =  EncOutData<
  TEST_COMMON_SYMBOL_WIDTH, OVERFLOW_COUNT_WIDTH, TEST_COMMON_SYMBOL_COUNT>;

type OverflowEncInData  = OverflowEncData;
type OverflowEncOutData = (OverflowEncData, u32, u32);

#[test_proc]
proc RunLengthEncoderAdvancedRealignStageNoPairs {
  terminator: chan<bool> out;
  enc_input_s: chan<TestCommonEncInData> out;
  enc_output_r: chan<TestCommonEncOutData> in;

  init {()}
  config (terminator: chan<bool> out) {
    let (enc_input_s, enc_input_r) = chan<TestCommonEncInData>;
    let (enc_output_s, enc_output_r) = chan<TestCommonEncOutData>;

    spawn RunLengthEncoderAdvancedRealignStage<
      TEST_COMMON_SYMBOL_WIDTH, TEST_COMMON_COUNT_WIDTH, TEST_COMMON_SYMBOL_COUNT>(
        enc_input_r, enc_output_s);
    (terminator, enc_input_s, enc_output_r)
  }
  next (tok: token, state:()) {
    let TestCommonTestStimuli: TestCommonStimulus[1] = [
      ([TestCommonSymbol:0x0, TestCommonSymbol:0x0,
        TestCommonSymbol:0x0, TestCommonSymbol:0x0,],
       [TestCommonCount:0x0, TestCommonCount:0x0,
        TestCommonCount:0x0, TestCommonCount:0x0,]
      ),
    ];
    let tok = for ((counter, stimulus), tok):
                  ((u32, TestCommonStimulus) , token)
                  in enumerate(TestCommonTestStimuli) {
      let last = counter == (array_size(TestCommonTestStimuli) - u32:1);
      let _stimulus = TestCommonEncInData{
        symbols: stimulus.0,
        counts: stimulus.1,
        last: last
      };
      let tok = send(tok, enc_input_s, _stimulus);
      trace_fmt!("Sent {} stimuli, symbols: {:x}, counts: {:x}, lasts: {}",
        counter, _stimulus.symbols, _stimulus.counts, _stimulus.last);
      (tok)
    }(tok);
    let TestCommonTestOutput:
        (TestCommonSymbols, TestCommonCounts, u32, u32)[1] = [
      ([TestCommonSymbol:0x0, TestCommonSymbol:0x0,
        TestCommonSymbol:0x0, TestCommonSymbol:0x0,],
       [TestCommonCount:0x0, TestCommonCount:0x0,
        TestCommonCount:0x0, TestCommonCount:0x0,],
       u32:0,
       u32:0,
      ),
    ];
    let tok = for ((counter, (symbols, counts, propagation, end)), tok):
        ((u32, (TestCommonSymbols, TestCommonCounts, u32, u32)) , token)
        in enumerate(TestCommonTestOutput) {
      let last = counter == (array_size(TestCommonTestOutput) - u32:1);
      let expected = (
        TestCommonEncData{
          symbols: symbols,
          counts: counts,
          last: last
        },
        propagation,
        end,
      );
      let (tok, enc_output) = recv(tok, enc_output_r);
      trace_fmt!(
        "Received {} pairs, symbols: {:x}, counts: {}, last: {}, propagation: {}",
        counter, enc_output.0.symbols, enc_output.0.counts,
        enc_output.0.last, enc_output.1
      );
      assert_eq(enc_output, expected);
      (tok)
    }(tok);
    send(tok, terminator, true);
  }
}

#[test_proc]
proc RunLengthEncoderAdvancedRealignStageAllPairsFilled {
  terminator: chan<bool> out;
  enc_input_s: chan<TestCommonEncInData> out;
  enc_output_r: chan<TestCommonEncOutData> in;

  init {()}
  config (terminator: chan<bool> out) {
    let (enc_input_s, enc_input_r) = chan<TestCommonEncInData>;
    let (enc_output_s, enc_output_r) = chan<TestCommonEncOutData>;

    spawn RunLengthEncoderAdvancedRealignStage<
      TEST_COMMON_SYMBOL_WIDTH, TEST_COMMON_COUNT_WIDTH, TEST_COMMON_SYMBOL_COUNT>(
        enc_input_r, enc_output_s);
    (terminator, enc_input_s, enc_output_r)
  }
  next (tok: token, state:()) {
    let TestCommonTestStimuli: TestCommonStimulus[1] = [
      ([TestCommonSymbol:0x1, TestCommonSymbol:0x2,
        TestCommonSymbol:0x3, TestCommonSymbol:0x4,],
       [TestCommonCount:0x1, TestCommonCount:0x1,
        TestCommonCount:0x1, TestCommonCount:0x1,]
      ),
    ];
    let tok = for ((counter, stimulus), tok):
                  ((u32, TestCommonStimulus) , token)
                  in enumerate(TestCommonTestStimuli) {
      let last = counter == (array_size(TestCommonTestStimuli) - u32:1);
      let _stimulus = TestCommonEncInData{
        symbols: stimulus.0,
        counts: stimulus.1,
        last: last
      };
      let tok = send(tok, enc_input_s, _stimulus);
      trace_fmt!("Sent {} stimuli, symbols: {:x}, counts: {:x}, last: {}",
        counter, _stimulus.symbols, _stimulus.counts, _stimulus.last);
      (tok)
    }(tok);
    let TestCommonTestOutput:
        (TestCommonSymbols, TestCommonCounts, u32, u32)[1] = [
      ([TestCommonSymbol:0x1, TestCommonSymbol:0x2,
        TestCommonSymbol:0x3, TestCommonSymbol:0x4,],
       [TestCommonCount:0x1, TestCommonCount:0x1,
        TestCommonCount:0x1, TestCommonCount:0x1,],
       u32:0,
       u32:4,
      ),
    ];
    let tok = for ((counter, (symbols, counts, propagation, end)), tok):
        ((u32, (TestCommonSymbols, TestCommonCounts, u32, u32)) , token)
        in enumerate(TestCommonTestOutput) {
      let last = counter == (array_size(TestCommonTestOutput) - u32:1);
      let expected = (
        TestCommonEncData{
          symbols: symbols,
          counts: counts,
          last: last
        },
        propagation,
        end,
      );
      let (tok, enc_output) = recv(tok, enc_output_r);
      trace_fmt!(
        "Received {} pairs, symbols: {:x}, counts: {}, last: {}, propagation: {}",
        counter, enc_output.0.symbols, enc_output.0.counts,
        enc_output.0.last, enc_output.1
      );
      assert_eq(enc_output, expected);
      (tok)
    }(tok);
    send(tok, terminator, true);
  }
}

#[test_proc]
proc RunLengthEncoderAdvancedRealignStageFarAwayPair {
  terminator: chan<bool> out;
  enc_input_s: chan<TestCommonEncInData> out;
  enc_output_r: chan<TestCommonEncOutData> in;

  init {()}
  config (terminator: chan<bool> out) {
    let (enc_input_s, enc_input_r) = chan<TestCommonEncInData>;
    let (enc_output_s, enc_output_r) = chan<TestCommonEncOutData>;

    spawn RunLengthEncoderAdvancedRealignStage<
      TEST_COMMON_SYMBOL_WIDTH, TEST_COMMON_COUNT_WIDTH, TEST_COMMON_SYMBOL_COUNT>(
        enc_input_r, enc_output_s);
    (terminator, enc_input_s, enc_output_r)
  }
  next (tok: token, state:()) {
    let TestCommonTestStimuli: TestCommonStimulus[1] = [
      ([TestCommonSymbol:0x0, TestCommonSymbol:0x0,
        TestCommonSymbol:0x0, TestCommonSymbol:0x1,],
       [TestCommonCount:0x0, TestCommonCount:0x0,
        TestCommonCount:0x0, TestCommonCount:0x4,]
      ),
    ];
    let tok = for ((counter, stimulus), tok):
                  ((u32, TestCommonStimulus) , token)
                  in enumerate(TestCommonTestStimuli) {
      let last = counter == (array_size(TestCommonTestStimuli) - u32:1);
      let _stimulus = TestCommonEncInData{
        symbols: stimulus.0,
        counts: stimulus.1,
        last: last
      };
      let tok = send(tok, enc_input_s, _stimulus);
      trace_fmt!("Sent {} stimuli, symbols: {:x}, counts: {:x}, lasts: {}",
        counter, _stimulus.symbols, _stimulus.counts, _stimulus.last);
      (tok)
    }(tok);
    let TestCommonTestOutput:
        (TestCommonSymbol[4], TestCommonCount[4], u32, u32)[1] = [
      ([TestCommonSymbol:0x1, TestCommonSymbol:0x0,
        TestCommonSymbol:0x0, TestCommonSymbol:0x0,],
       [TestCommonCount:0x4, TestCommonCount:0x0,
        TestCommonCount:0x0, TestCommonCount:0x0,],
       u32:0,
       u32:1,
      ),
    ];
    let tok = for ((counter, (symbols, counts, propagation, end)), tok):
        ((u32, (TestCommonSymbols, TestCommonCounts, u32, u32)) , token)
        in enumerate(TestCommonTestOutput) {
      let last = counter == (array_size(TestCommonTestOutput) - u32:1);
      let expected = (
        TestCommonEncData{
          symbols: symbols,
          counts: counts,
          last: last
        },
        propagation,
        end,
      );
      let (tok, enc_output) = recv(tok, enc_output_r);
      trace_fmt!(
        "Received {} pairs, symbols: {:x}, counts: {}, last: {}, propagation: {}",
        counter, enc_output.0.symbols, enc_output.0.counts,
        enc_output.0.last, enc_output.1
      );
      assert_eq(enc_output, expected);
      (tok)
    }(tok);
    send(tok, terminator, true);
  }
}

#[test_proc]
proc RunLengthEncoderAdvancedRealignStagePropagataion {
  terminator: chan<bool> out;
  enc_input_s: chan<OverflowEncInData> out;
  enc_output_r: chan<OverflowEncOutData> in;

  init {()}
  config (terminator: chan<bool> out) {
    let (enc_input_s, enc_input_r) = chan<OverflowEncInData>;
    let (enc_output_s, enc_output_r) = chan<OverflowEncOutData>;

    spawn RunLengthEncoderAdvancedRealignStage<
      TEST_COMMON_SYMBOL_WIDTH, OVERFLOW_COUNT_WIDTH, TEST_COMMON_SYMBOL_COUNT>(
        enc_input_r, enc_output_s);
    (terminator, enc_input_s, enc_output_r)
  }
  next (tok: token, state:()) {
    let OverflowTestStimuli: OverflowStimulus[1] = [
      ([OverflowSymbol:0x0, OverflowSymbol:0x0,
        OverflowSymbol:0x1, OverflowSymbol:0x1,],
       [OverflowCount:0x0, OverflowCount:0x0,
        OverflowCount:0x3, OverflowCount:0x1,]
      ),
    ];
    let tok = for ((counter, stimulus), tok):
                  ((u32, OverflowStimulus) , token)
                  in enumerate(OverflowTestStimuli) {
      let last = counter == (array_size(OverflowTestStimuli) - u32:1);
      let _stimulus = TestCommonEncInData{
        symbols: stimulus.0,
        counts: stimulus.1,
        last: last
      };
      let tok = send(tok, enc_input_s, _stimulus);
      trace_fmt!("Sent {} stimuli, symbols: {:x}, counts: {:x}, lasts: {}",
        counter, _stimulus.symbols, _stimulus.counts, _stimulus.last);
      (tok)
    }(tok);
    let OverflowTestOutput:
        (OverflowSymbol[4], OverflowCount[4], u32, u32)[1] = [
      ([OverflowSymbol:0x1, OverflowSymbol:0x1,
        OverflowSymbol:0x0, OverflowSymbol:0x0,],
       [OverflowCount:0x3, OverflowCount:0x1,
        OverflowCount:0x0, OverflowCount:0x0,],
       u32:1,
       u32:2,
      ),
    ];
    let tok = for ((counter, (symbols, counts, propagation, end)), tok):
        ((u32, (TestCommonSymbols, OverflowCounts, u32, u32)) , token)
        in enumerate(OverflowTestOutput) {
      let last = counter == (array_size(OverflowTestOutput) - u32:1);
      let expected = (
        TestCommonEncData{
          symbols: symbols,
          counts: counts,
          last: last
        },
        propagation,
        end,
      );
      let (tok, enc_output) = recv(tok, enc_output_r);
      trace_fmt!(
        "Received {} pairs, symbols: {:x}, counts: {}, last: {}, propagation: {}",
        counter, enc_output.0.symbols, enc_output.0.counts,
        enc_output.0.last, enc_output.1
      );
      assert_eq(enc_output, expected);
      (tok)
    }(tok);
    send(tok, terminator, true);
  }
}
