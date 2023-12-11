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

pub proc RunLengthEncoderAdvancedReduceStage<
  SYMBOL_WIDTH: u32, COUNT_WIDTH: u32,INPUT_WIDTH:u32> {

  input_r: chan<EncInData<SYMBOL_WIDTH, INPUT_WIDTH>> in;
  output_s: chan<EncOutData<SYMBOL_WIDTH, COUNT_WIDTH, INPUT_WIDTH>> out;

  init {()}
  config (
    input_r: chan<EncInData<SYMBOL_WIDTH, INPUT_WIDTH>> in,
    output_s: chan<EncOutData<SYMBOL_WIDTH, COUNT_WIDTH, INPUT_WIDTH>> out,
  ) {(input_r, output_s)}

  next (tok: token, state: ()) {
    let (tok, input) = recv(tok, input_r);
    let (symbols, counts, symbol, count) =
      for (idx, (symbols, counts, symbol, count)) in range(u32:1, INPUT_WIDTH) {
        let _symbol = input.symbols[idx];
        let valid   = input.symbol_valids[idx];
        let symbols_diff = symbol != _symbol;
        let overflow = count == std::unsigned_max_value<COUNT_WIDTH>();
        match (valid, overflow, symbols_diff) {
          (false, _, _)   => (symbols, counts, symbol, count),
          (true, false, false) => (symbols, counts,
            symbol, count + bits[COUNT_WIDTH]: 1),
          _ => (
            update(symbols, idx - u32:1, symbol),
            update(counts, idx - u32:1, count),
            _symbol,
            bits[COUNT_WIDTH]: 1,
          ),
        }
    } ((zero!<bits[SYMBOL_WIDTH][INPUT_WIDTH]>(),
        zero!<bits[COUNT_WIDTH][INPUT_WIDTH]>(),
        input.symbols[u32:0],
        input.symbol_valids[u32:0] as bits[COUNT_WIDTH]));
    let symbols = update(symbols, INPUT_WIDTH - u32:1, symbol);
    let counts = update(counts, INPUT_WIDTH - u32:1, count);
    let output = EncOutData<SYMBOL_WIDTH, COUNT_WIDTH, INPUT_WIDTH> {
      symbols: symbols,
      counts: counts,
      last: input.last,
    };
    let not_empty = input.last ||
      or_reduce(std::convert_to_bits_msb0(input.symbol_valids));
    send_if(tok, output_s, not_empty, output);
  }
}

// Test ReduceStage

// Transaction without overflow
const TEST_COMMON_SYMBOL_WIDTH = u32:32;
const TEST_COMMON_COUNT_WIDTH  = u32:32;
const TEST_COMMON_SYMBOL_COUNT = u32:4;

type TestCommonSymbol     = bits[TEST_COMMON_SYMBOL_WIDTH];
type TestCommonCount      = bits[TEST_COMMON_COUNT_WIDTH];

type TestCommonSymbols    = TestCommonSymbol[TEST_COMMON_SYMBOL_COUNT];
type TestCommonCounts     = TestCommonCount[TEST_COMMON_SYMBOL_COUNT];

type TestCommonStimulus   = (
  TestCommonSymbols, bits[1][TEST_COMMON_SYMBOL_COUNT]);

type TestCommonEncInData  = EncInData<
  TEST_COMMON_SYMBOL_WIDTH, TEST_COMMON_SYMBOL_COUNT>;
type TestCommonEncOutData = EncOutData<
  TEST_COMMON_SYMBOL_WIDTH, TEST_COMMON_COUNT_WIDTH, TEST_COMMON_SYMBOL_COUNT>;

// Transaction with counter overflow
const OVERFLOW_COUNT_WIDTH  = u32:2;

type OverflowSymbol     = TestCommonSymbol;
type OverflowCount      = bits[OVERFLOW_COUNT_WIDTH];

type OverflowSymbols    = OverflowSymbol[TEST_COMMON_SYMBOL_COUNT];
type OverflowCounts     = OverflowCount[TEST_COMMON_SYMBOL_COUNT];

type OverflowStimulus   = (
  TestCommonSymbols, bits[1][TEST_COMMON_SYMBOL_COUNT]);

type OverflowEncInData  = TestCommonEncInData;
type OverflowEncOutData = EncOutData<
  TEST_COMMON_SYMBOL_WIDTH, OVERFLOW_COUNT_WIDTH, TEST_COMMON_SYMBOL_COUNT>;


#[test_proc]
proc RunLengthEncoderAdvancedReduceStageNoSymbols {
  terminator: chan<bool> out;
  enc_input_s: chan<TestCommonEncInData> out;
  enc_output_r: chan<TestCommonEncOutData> in;

  init {()}
  config (terminator: chan<bool> out) {
    let (enc_input_s, enc_input_r) = chan<TestCommonEncInData>;
    let (enc_output_s, enc_output_r) = chan<TestCommonEncOutData>;

    spawn RunLengthEncoderAdvancedReduceStage<
      TEST_COMMON_SYMBOL_WIDTH, TEST_COMMON_COUNT_WIDTH, TEST_COMMON_SYMBOL_COUNT>(
        enc_input_r, enc_output_s);
    (terminator, enc_input_s, enc_output_r)
  }
  next (tok: token, state:()) {
    let TestCommonTestStimuli: TestCommonStimulus[2] = [
      ([TestCommonSymbol:0x0, TestCommonSymbol:0x0,
        TestCommonSymbol:0x0,  TestCommonSymbol:0x0],
       [bits[1]:0, bits[1]:0, bits[1]:0, bits[1]:0],
      ),
      ([TestCommonSymbol:0x0, TestCommonSymbol:0x0,
        TestCommonSymbol:0x0,  TestCommonSymbol:0x0],
       [bits[1]:0, bits[1]:0, bits[1]:0, bits[1]:0],
      )
    ];
    let tok = for ((counter, stimulus), tok):
                  ((u32, TestCommonStimulus) , token)
                  in enumerate(TestCommonTestStimuli) {
      let last = counter == (array_size(TestCommonTestStimuli) - u32:1);
      let _stimulus = TestCommonEncInData{
        symbols: stimulus.0,
        symbol_valids: stimulus.1,
        last: last
      };
      let tok = send(tok, enc_input_s, _stimulus);
      trace_fmt!("Sent {} stimuli, symbols: {:x}, symbol_valids: {:x}, lasts: {}",
        counter, _stimulus.symbols, _stimulus.symbol_valids, _stimulus.last);
      (tok)
    }(tok);
    let TestCommonTestOutput:
        (TestCommonSymbols, TestCommonCounts)[1] = [
      ([TestCommonSymbol:0x0, TestCommonSymbol:0x0,
        TestCommonSymbol:0x0, TestCommonSymbol:0x0,],
       [TestCommonCount:0x0, TestCommonCount:0x0,
        TestCommonCount:0x0, TestCommonCount:0x0,]
      ),
    ];
    let tok = for ((counter, (symbols, counts)), tok):
        ((u32, (TestCommonSymbols, TestCommonCounts)) , token)
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
        counter, enc_output.symbols, enc_output.counts, enc_output.last
      );
      assert_eq(enc_output, expected);
      (tok)
    }(tok);
    send(tok, terminator, true);
  }
}

#[test_proc]
proc RunLengthEncoderAdvancedReduceStageNonRepeatingSymbols {
  terminator: chan<bool> out;
  enc_input_s: chan<TestCommonEncInData> out;
  enc_output_r: chan<TestCommonEncOutData> in;

  init {()}
  config (terminator: chan<bool> out) {
    let (enc_input_s, enc_input_r) = chan<TestCommonEncInData>;
    let (enc_output_s, enc_output_r) = chan<TestCommonEncOutData>;

    spawn RunLengthEncoderAdvancedReduceStage<
      TEST_COMMON_SYMBOL_WIDTH, TEST_COMMON_COUNT_WIDTH, TEST_COMMON_SYMBOL_COUNT>(
        enc_input_r, enc_output_s);
    (terminator, enc_input_s, enc_output_r)
  }
  next (tok: token, state:()) {
    let TestCommonTestStimuli: TestCommonStimulus[1] = [
      ([TestCommonSymbol:0x1, TestCommonSymbol:0x2,
        TestCommonSymbol:0x3, TestCommonSymbol:0x4],
       [bits[1]:1, bits[1]:1, bits[1]:1, bits[1]:1],
      )
    ];
    let tok = for ((counter, stimulus), tok):
                  ((u32, TestCommonStimulus) , token)
                  in enumerate(TestCommonTestStimuli) {
      let last = counter == (array_size(TestCommonTestStimuli) - u32:1);
      let _stimulus = TestCommonEncInData{
        symbols: stimulus.0,
        symbol_valids: stimulus.1,
        last: last
      };
      let tok = send(tok, enc_input_s, _stimulus);
      trace_fmt!("Sent {} stimuli, symbols: {:x}, symbol_valids: {:x}, lasts: {}",
        counter, _stimulus.symbols, _stimulus.symbol_valids, _stimulus.last);
      (tok)
    }(tok);
    let TestCommonTestOutput:
        (TestCommonSymbol[4], TestCommonCount[4])[1] = [
      ([TestCommonSymbol:0x1, TestCommonSymbol:0x2,
        TestCommonSymbol:0x3, TestCommonSymbol:0x4,],
       [TestCommonCount:0x1, TestCommonCount:0x1,
        TestCommonCount:0x1, TestCommonCount:0x1,]
      ),
    ];
    let tok = for ((counter, (symbols, counts)), tok):
        ((u32, (TestCommonSymbol[4], TestCommonCount[4])) , token)
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
        counter, enc_output.symbols, enc_output.counts, enc_output.last
      );
      assert_eq(enc_output, expected);
      (tok)
    }(tok);
    send(tok, terminator, true);
  }
}

#[test_proc]
proc RunLengthEncoderAdvancedReduceStageRepeatingSymbolsNoOverflow {
  terminator: chan<bool> out;
  enc_input_s: chan<TestCommonEncInData> out;
  enc_output_r: chan<TestCommonEncOutData> in;

  init {()}
  config (terminator: chan<bool> out) {
    let (enc_input_s, enc_input_r) = chan<TestCommonEncInData>;
    let (enc_output_s, enc_output_r) = chan<TestCommonEncOutData>;

    spawn RunLengthEncoderAdvancedReduceStage<
      TEST_COMMON_SYMBOL_WIDTH, TEST_COMMON_COUNT_WIDTH, TEST_COMMON_SYMBOL_COUNT>(
        enc_input_r, enc_output_s);
    (terminator, enc_input_s, enc_output_r)
  }
  next (tok: token, state:()) {
    let TestCommonTestStimuli: TestCommonStimulus[1] = [
      ([TestCommonSymbol:0x1, TestCommonSymbol:0x1,
        TestCommonSymbol:0x1, TestCommonSymbol:0x1],
       [bits[1]:1, bits[1]:1, bits[1]:1, bits[1]:1],
      )
    ];
    let tok = for ((counter, stimulus), tok):
                  ((u32, TestCommonStimulus) , token)
                  in enumerate(TestCommonTestStimuli) {
      let last = counter == (array_size(TestCommonTestStimuli) - u32:1);
      let _stimulus = TestCommonEncInData{
        symbols: stimulus.0,
        symbol_valids: stimulus.1,
        last: last
      };
      let tok = send(tok, enc_input_s, _stimulus);
      trace_fmt!("Sent {} stimuli, symbols: {:x}, symbol_valids: {:x}, lasts: {}",
        counter, _stimulus.symbols, _stimulus.symbol_valids, _stimulus.last);
      (tok)
    }(tok);
    let TestCommonTestOutput:
        (TestCommonSymbol[4], TestCommonCount[4])[1] = [
      ([TestCommonSymbol:0x0, TestCommonSymbol:0x0,
        TestCommonSymbol:0x0, TestCommonSymbol:0x1,],
       [TestCommonCount:0x0, TestCommonCount:0x0,
        TestCommonCount:0x0, TestCommonCount:0x4,]
      ),
    ];
    let tok = for ((counter, (symbols, counts)), tok):
        ((u32, (TestCommonSymbol[4], TestCommonCount[4])) , token)
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
        counter, enc_output.symbols, enc_output.counts, enc_output.last
      );
      assert_eq(enc_output, expected);
      (tok)
    }(tok);
    send(tok, terminator, true);
  }
}

#[test_proc]
proc RunLengthEncoderAdvancedReduceStageRepeatingSymbolsOverflow {
  terminator: chan<bool> out;
  enc_input_s: chan<OverflowEncInData> out;
  enc_output_r: chan<OverflowEncOutData> in;

  init {()}
  config (terminator: chan<bool> out) {
    let (enc_input_s, enc_input_r) = chan<OverflowEncInData>;
    let (enc_output_s, enc_output_r) = chan<OverflowEncOutData>;

    spawn RunLengthEncoderAdvancedReduceStage<
      TEST_COMMON_SYMBOL_WIDTH, OVERFLOW_COUNT_WIDTH, TEST_COMMON_SYMBOL_COUNT>(
        enc_input_r, enc_output_s);
    (terminator, enc_input_s, enc_output_r)
  }
  next (tok: token, state:()) {
    let OverflowTestStimuli: OverflowStimulus[1] = [
      ([OverflowSymbol:0x1, OverflowSymbol:0x1,
        OverflowSymbol:0x1, OverflowSymbol:0x1],
       [bits[1]:1, bits[1]:1, bits[1]:1, bits[1]:1],
      )
    ];
    let tok = for ((counter, stimulus), tok):
                  ((u32, OverflowStimulus) , token)
                  in enumerate(OverflowTestStimuli) {
      let last = counter == (array_size(OverflowTestStimuli) - u32:1);
      let _stimulus = OverflowEncInData{
        symbols: stimulus.0,
        symbol_valids: stimulus.1,
        last: last
      };
      let tok = send(tok, enc_input_s, _stimulus);
      trace_fmt!("Sent {} stimuli, symbols: {:x}, symbol_valids: {:x}, lasts: {}",
        counter, _stimulus.symbols, _stimulus.symbol_valids, _stimulus.last);
      (tok)
    }(tok);
    let OverflowTestOutput:
        (OverflowSymbol[4], OverflowCount[4])[1] = [
      ([OverflowSymbol:0x0, OverflowSymbol:0x0,
        OverflowSymbol:0x1, OverflowSymbol:0x1,],
       [OverflowCount:0x0, OverflowCount:0x0,
        OverflowCount:0x3, OverflowCount:0x1,]
      ),
    ];
    let tok = for ((counter, (symbols, counts)), tok):
        ((u32, (OverflowSymbol[4], OverflowCount[4])) , token)
        in enumerate(OverflowTestOutput) {
      let last = counter == (array_size(OverflowTestOutput) - u32:1);
      let expected = OverflowEncOutData{
          symbols: symbols,
          counts: counts,
          last: last
      };
      let (tok, enc_output) = recv(tok, enc_output_r);
      trace_fmt!(
        "Received {} pairs, symbols: {:x}, counts: {}, last: {}",
        counter, enc_output.symbols, enc_output.counts, enc_output.last
      );
      assert_eq(enc_output, expected);
      (tok)
    }(tok);
    send(tok, terminator, true);
  }
}
