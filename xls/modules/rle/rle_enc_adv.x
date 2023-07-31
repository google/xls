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

// This file implements a parametric multisymbol RLE encoder
//
// The encoder uses Run Length Encoding (RLE) to compress the input stream of
// repeating symbols to the output stream that contains the symbols and
// the number of its consecutive occurrences in the input stream.
// Both the input and the output channels use additional `last` flag
// that indicates whether the packet ends the transmission. After sending
// the last packet encoder dumps all the data to the output stream.
// The behavior of the encoder is presented on the waveform below:

// This encoder is implemented as a net of 4 processes.
// 1. Reduce stage - this process takes incoming symbols and symbol_valid
// and reduces them into symbol count pairs. This stage is stateless.
// 2. Realign stage - this process moves pairs emitted from previous stage
// so that they are align to the left, it also calculates propagation distance
// for the first pair.
// 3. Core stage - this stage is stateful. It takes align pairs,
// and combines them with its state.It outputs multiple symbol/count pairs.
// 4 - Adjust Width stage - this stage takes output from the core stage.
// If output can handle more or equal number of pairs as
// input number of symbols. This stage does nothing.
// If the output is narrower than the input,
// this stage will serialize symbol counter pairs.


import std
import xls.modules.rle.rle_common as rle_common

import xls.modules.rle.rle_enc_adv_reduce_stage as reduce_stage
import xls.modules.rle.rle_enc_adv_realign_stage as realign_stage
import xls.modules.rle.rle_enc_adv_core as core
import xls.modules.rle.rle_enc_adv_adjust_width_stage as adjust_width_stage


type EncInData  = rle_common::PlainData;
type EncOutData = rle_common::CompressedData;


// RLE encoder implementation
pub proc RunLengthEncoderAdvanced<SYMBOL_WIDTH: u32, COUNT_WIDTH: u32,
  INPUT_WIDTH:u32, OUTPUT_WIDTH:u32> {

  init {()}

  config (
    input_r: chan<EncInData<SYMBOL_WIDTH, INPUT_WIDTH>> in,
    output_s: chan<EncOutData<SYMBOL_WIDTH, COUNT_WIDTH, OUTPUT_WIDTH>> out,
  ) {
    let (reduced_s, reduced_r) = chan<
      EncOutData<SYMBOL_WIDTH, COUNT_WIDTH, INPUT_WIDTH>>;
    spawn reduce_stage::RunLengthEncoderAdvancedReduceStage<
      SYMBOL_WIDTH, COUNT_WIDTH, INPUT_WIDTH>(input_r, reduced_s);

    let (realigned_s, realigned_r) = chan<
      (EncOutData<SYMBOL_WIDTH, COUNT_WIDTH, INPUT_WIDTH>, u32, u32)>;
    spawn realign_stage::RunLengthEncoderAdvancedRealignStage<
      SYMBOL_WIDTH, COUNT_WIDTH, INPUT_WIDTH>(reduced_r, realigned_s);

    let (compressed_s, compressed_r) = chan<
      (EncOutData<SYMBOL_WIDTH, COUNT_WIDTH, INPUT_WIDTH>, u32)>;

    spawn core::RunLengthEncoderAdvancedCoreStage<
      SYMBOL_WIDTH, COUNT_WIDTH, INPUT_WIDTH>(realigned_r, compressed_s);
    spawn adjust_width_stage::RunLengthEncoderAdvancedAdjustWidthStage<
      SYMBOL_WIDTH, COUNT_WIDTH, INPUT_WIDTH, OUTPUT_WIDTH>(
        compressed_r, output_s);
    ()
  }

  next (tok: token, state: ()) {()}
}

// RLE encoder specialization for the codegen
proc RunLengthEncoder8_8_4_2 {

    init {()}

    config (
        input_r: chan<EncInData<8, 4>> in,
        output_s: chan<EncOutData<8, 8, 2>> out,
    ) {
        spawn RunLengthEncoderAdvanced<u32:8, u32:8, u32:4, u32:2>(
          input_r, output_s);
        ()
    }

    next (tok: token, state: ()) {
        ()
    }
}

// Testing
// Each subprocess is tested individually.

const TEST_COMMON_SYMBOL_WIDTH = u32:32;
const TEST_COMMON_COUNT_WIDTH = u32:32;
const TEST_COMMON_INPUT_WIDTH = u32:4;
const TEST_COMMON_OUTPUT_WIDTH = u32:2;

type TestCommonSymbol     = bits[TEST_COMMON_SYMBOL_WIDTH];
type TestCommonCount      = bits[TEST_COMMON_COUNT_WIDTH];

type TestCommonSymbolsIn      = TestCommonSymbol[TEST_COMMON_INPUT_WIDTH];
type TestCommonSymbolValidsIn = bits[1][TEST_COMMON_INPUT_WIDTH];
type TestCommonStimulus       = (TestCommonSymbolsIn, TestCommonSymbolValidsIn);

type TestCommonSymbolsOut = TestCommonSymbol[TEST_COMMON_OUTPUT_WIDTH];
type TestCommonCountsOut  = TestCommonCount[TEST_COMMON_OUTPUT_WIDTH];
type TestCommonOutputs   = (TestCommonSymbolsOut, TestCommonCountsOut);

type TestCommonEncInData  = EncInData<
  TEST_COMMON_SYMBOL_WIDTH, TEST_COMMON_INPUT_WIDTH>;
type TestCommonEncOutData = EncOutData<
  TEST_COMMON_SYMBOL_WIDTH, TEST_COMMON_COUNT_WIDTH, TEST_COMMON_OUTPUT_WIDTH>;


#[test_proc]
proc ConsumeMultipleSymbolRepetitionsAtOnce {
  terminator: chan<bool> out;
  enc_input_s: chan<TestCommonEncInData> out;
  enc_output_r: chan<TestCommonEncOutData> in;

  init{()}

  config(terminator: chan<bool> out) {
    let (enc_input_s, enc_input_r) = chan<TestCommonEncInData>;
    let (enc_output_s, enc_output_r) = chan<TestCommonEncOutData>;
    spawn RunLengthEncoderAdvanced<
      TEST_COMMON_SYMBOL_WIDTH, TEST_COMMON_COUNT_WIDTH,
      TEST_COMMON_INPUT_WIDTH, TEST_COMMON_OUTPUT_WIDTH
    >(enc_input_r, enc_output_s);
    (terminator, enc_input_s, enc_output_r)
  }

  next(tok: token, state: ()) {
        let TestCommonTestStimuli: TestCommonStimulus[4] = [
      (TestCommonSymbolsIn:[0x1, 0x1, 0x1, 0x1],
       TestCommonSymbolValidsIn:[0x1, 0x1, 0x1, 0x1]),
      (TestCommonSymbolsIn:[0x1, 0x1, 0x1, 0x1],
       TestCommonSymbolValidsIn:[0x1, 0x1, 0x1, 0x1]),
      (TestCommonSymbolsIn:[0x1, 0x1, 0x1, 0x1],
       TestCommonSymbolValidsIn:[0x1, 0x1, 0x1, 0x1]),
      (TestCommonSymbolsIn:[0x1, 0x1, 0x1, 0x1],
       TestCommonSymbolValidsIn:[0x1, 0x1, 0x1, 0x1]),
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
      trace_fmt!("Sent {} stimuli, symbols: {:x}, last: {}",
        counter, _stimulus.symbols, _stimulus.last);
      (tok)
    }(tok);

    let TestCommonTestOutput: TestCommonOutputs[1] = [
      (TestCommonSymbolsOut:[0x1, 0x0],
       TestCommonCountsOut:[0x10, 0x0]),
    ];

    let tok = for ((counter, output), tok):
       ((u32, TestCommonOutputs) , token)
       in enumerate(TestCommonTestOutput) {
      let last = counter == (array_size(TestCommonTestOutput) - u32:1);
      let expected = TestCommonEncOutData{
        symbols: output.0,
        counts: output.1,
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

#[test_proc]
proc ConsumeMultipleSymbolsAtOnce {
  terminator: chan<bool> out;
  enc_input_s: chan<TestCommonEncInData> out;
  enc_output_r: chan<TestCommonEncOutData> in;

  init{()}

  config(terminator: chan<bool> out) {
    let (enc_input_s, enc_input_r) = chan<TestCommonEncInData>;
    let (enc_output_s, enc_output_r) = chan<TestCommonEncOutData>;
    spawn RunLengthEncoderAdvanced<
      TEST_COMMON_SYMBOL_WIDTH, TEST_COMMON_COUNT_WIDTH,
      TEST_COMMON_INPUT_WIDTH, TEST_COMMON_OUTPUT_WIDTH
    >(enc_input_r, enc_output_s);
    (terminator, enc_input_s, enc_output_r)
  }

  next(tok: token, state: ()) {
        let TestCommonTestStimuli: TestCommonStimulus[1] = [
      (TestCommonSymbolsIn:[0x1, 0x1, 0x2, 0x2],
       TestCommonSymbolValidsIn:[0x1, 0x1, 0x1, 0x1]),
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
      trace_fmt!("Sent {} stimuli, symbols: {:x}, last: {}",
        counter, _stimulus.symbols, _stimulus.last);
      (tok)
    }(tok);

    let TestCommonTestOutput: TestCommonOutputs[1] = [
      (TestCommonSymbolsOut:[0x1, 0x2],
       TestCommonCountsOut:[0x2, 0x2]),
    ];

    let tok = for ((counter, output), tok):
       ((u32, TestCommonOutputs) , token)
       in enumerate(TestCommonTestOutput) {
      let last = counter == (array_size(TestCommonTestOutput) - u32:1);
      let expected = TestCommonEncOutData{
        symbols: output.0,
        counts: output.1,
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

#[test_proc]
proc ConsumePacketWithInvalidSymbols {
  terminator: chan<bool> out;
  enc_input_s: chan<TestCommonEncInData> out;
  enc_output_r: chan<TestCommonEncOutData> in;

  init{()}

  config(terminator: chan<bool> out) {
    let (enc_input_s, enc_input_r) = chan<TestCommonEncInData>;
    let (enc_output_s, enc_output_r) = chan<TestCommonEncOutData>;
    spawn RunLengthEncoderAdvanced<
      TEST_COMMON_SYMBOL_WIDTH, TEST_COMMON_COUNT_WIDTH,
      TEST_COMMON_INPUT_WIDTH, TEST_COMMON_OUTPUT_WIDTH
    >(enc_input_r, enc_output_s);
    (terminator, enc_input_s, enc_output_r)
  }

  next(tok: token, state: ()) {
        let TestCommonTestStimuli: TestCommonStimulus[4] = [
      (TestCommonSymbolsIn:[0x1, 0x0, 0x1, 0x0],
       TestCommonSymbolValidsIn:[0x1, 0x0, 0x1, 0x0]),
      (TestCommonSymbolsIn:[0x1, 0x1, 0x0, 0x1],
       TestCommonSymbolValidsIn:[0x1, 0x1, 0x0, 0x1]),
      (TestCommonSymbolsIn:[0x0, 0x1, 0x1, 0x0],
       TestCommonSymbolValidsIn:[0x0, 0x1, 0x1, 0x0]),
      (TestCommonSymbolsIn:[0x0, 0x0, 0x0, 0x0],
       TestCommonSymbolValidsIn:[0x0, 0x0, 0x0, 0x0]),
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
      trace_fmt!("Sent {} stimuli, symbols: {:x}, last: {}",
        counter, _stimulus.symbols, _stimulus.last);
      (tok)
    }(tok);

    let TestCommonTestOutput: TestCommonOutputs[1] = [
      (TestCommonSymbolsOut:[0x1, 0x0],
       TestCommonCountsOut:[0x7, 0x0]),
    ];

    let tok = for ((counter, output), tok):
       ((u32, TestCommonOutputs) , token)
       in enumerate(TestCommonTestOutput) {
      let last = counter == (array_size(TestCommonTestOutput) - u32:1);
      let expected = TestCommonEncOutData{
        symbols: output.0,
        counts: output.1,
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

#[test_proc]
proc ConsumePacketWithAllDiffSymbols {
  terminator: chan<bool> out;
  enc_input_s: chan<TestCommonEncInData> out;
  enc_output_r: chan<TestCommonEncOutData> in;

  init{()}

  config(terminator: chan<bool> out) {
    let (enc_input_s, enc_input_r) = chan<TestCommonEncInData>;
    let (enc_output_s, enc_output_r) = chan<TestCommonEncOutData>;
    spawn RunLengthEncoderAdvanced<
      TEST_COMMON_SYMBOL_WIDTH, TEST_COMMON_COUNT_WIDTH,
      TEST_COMMON_INPUT_WIDTH, TEST_COMMON_OUTPUT_WIDTH
    >(enc_input_r, enc_output_s);
    (terminator, enc_input_s, enc_output_r)
  }

  next(tok: token, state: ()) {
        let TestCommonTestStimuli: TestCommonStimulus[1] = [
      (TestCommonSymbolsIn:[0x1, 0x2, 0x3, 0x4],
       TestCommonSymbolValidsIn:[0x1, 0x1, 0x1, 0x1]),
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
      trace_fmt!("Sent {} stimuli, symbols: {:x}, last: {}",
        counter, _stimulus.symbols, _stimulus.last);
      (tok)
    }(tok);

    let TestCommonTestOutput: TestCommonOutputs[2] = [
      (TestCommonSymbolsOut:[0x1, 0x2],
       TestCommonCountsOut:[0x1, 0x1]),
      (TestCommonSymbolsOut:[0x3, 0x4],
       TestCommonCountsOut:[0x1, 0x1]),
    ];

    let tok = for ((counter, output), tok):
       ((u32, TestCommonOutputs) , token)
       in enumerate(TestCommonTestOutput) {
      let last = counter == (array_size(TestCommonTestOutput) - u32:1);
      let expected = TestCommonEncOutData{
        symbols: output.0,
        counts: output.1,
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

#[test_proc]
proc ConsumePacketsWhereLastSymbolRepeats {
  terminator: chan<bool> out;
  enc_input_s: chan<TestCommonEncInData> out;
  enc_output_r: chan<TestCommonEncOutData> in;

  init{()}

  config(terminator: chan<bool> out) {
    let (enc_input_s, enc_input_r) = chan<TestCommonEncInData>;
    let (enc_output_s, enc_output_r) = chan<TestCommonEncOutData>;
    spawn RunLengthEncoderAdvanced<
      TEST_COMMON_SYMBOL_WIDTH, TEST_COMMON_COUNT_WIDTH,
      TEST_COMMON_INPUT_WIDTH, TEST_COMMON_OUTPUT_WIDTH
    >(enc_input_r, enc_output_s);
    (terminator, enc_input_s, enc_output_r)
  }

  next(tok: token, state: ()) {
        let TestCommonTestStimuli: TestCommonStimulus[2] = [
      (TestCommonSymbolsIn:[0x1, 0x2, 0x3, 0x4],
       TestCommonSymbolValidsIn:[0x1, 0x1, 0x1, 0x1]),
      (TestCommonSymbolsIn:[0x4, 0x4, 0x4, 0x4],
       TestCommonSymbolValidsIn:[0x1, 0x1, 0x1, 0x1]),
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
      trace_fmt!("Sent {} stimuli, symbols: {:x}, last: {}",
        counter, _stimulus.symbols, _stimulus.last);
      (tok)
    }(tok);

    let TestCommonTestOutput: TestCommonOutputs[2] = [
      (TestCommonSymbolsOut:[0x1, 0x2],
       TestCommonCountsOut:[0x1, 0x1]),
      (TestCommonSymbolsOut:[0x3, 0x4],
       TestCommonCountsOut:[0x1, 0x5]),
    ];

    let tok = for ((counter, output), tok):
       ((u32, TestCommonOutputs) , token)
       in enumerate(TestCommonTestOutput) {
      let last = counter == (array_size(TestCommonTestOutput) - u32:1);
      let expected = TestCommonEncOutData{
        symbols: output.0,
        counts: output.1,
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
