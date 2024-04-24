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


import std;
import xls.modules.rle.rle_common as rle_common;

type DecInData  = rle_common::CompressedData;
type DecOutData = rle_common::PlainData;

// structure to preserve the state of an RLE decoder
struct RunLengthDecoderState<SYMBOL_WIDTH: u32, COUNT_WIDTH: u32> {
  // symbol to be repeated on output
  symbol: bits[SYMBOL_WIDTH],
  // count of symbols that has to be send
  count: bits[COUNT_WIDTH],
  // send last when repeat ends
  last: bool,
}
 // RLE decoder implementation
pub proc RunLengthDecoder<SYMBOL_WIDTH: u32, COUNT_WIDTH: u32> {
  input_r: chan<DecInData<SYMBOL_WIDTH, COUNT_WIDTH>> in;
  output_s: chan<DecOutData<SYMBOL_WIDTH>> out;

  init {(
    RunLengthDecoderState<SYMBOL_WIDTH, COUNT_WIDTH> {
      symbol: bits[SYMBOL_WIDTH]:0,
      count:  bits[COUNT_WIDTH]:0,
      last:   bool:false,
    }
  )}

  config (
    input_r: chan<DecInData<SYMBOL_WIDTH, COUNT_WIDTH>> in,
    output_s: chan<DecOutData<SYMBOL_WIDTH>> out,
  ) {(input_r, output_s)}

  next (tok: token, state: RunLengthDecoderState<SYMBOL_WIDTH, COUNT_WIDTH>) {
    let state_input = DecInData {
      symbol: state.symbol,
      count: state.count,
      last: state.last
    };
    let recv_next_symbol = (state.count == bits[COUNT_WIDTH]:0);
    let (tok, input) = recv_if(tok, input_r, recv_next_symbol, state_input);
    let next_count = if input.count == bits[COUNT_WIDTH]:0 {
        fail!("invalid_count_0", input.count)
    } else {
        input.count - bits[COUNT_WIDTH]:1
    };
    let done_sending = (next_count == bits[COUNT_WIDTH]:0);
    let send_last = input.last && done_sending;
    let data_tok = send(tok, output_s, DecOutData {
      symbol: input.symbol,
      last: send_last
    });
    if (send_last) {
      zero!<RunLengthDecoderState>()
    } else {
      RunLengthDecoderState {
        symbol: input.symbol,
        count: next_count,
        last: input.last,
      }
    }
  }
}


// RLE decoder specialization for the codegen
proc RunLengthDecoder32 {
  init {()}

  config (
    input_r: chan<DecInData<32, 2>> in,
    output_s: chan<DecOutData<32>> out,
  ) {
    spawn RunLengthDecoder<u32:32, u32:2>(input_r, output_s);
    ()
  }

  next (tok: token, state: ()) {
    ()
  }
}

// Tests

const TEST_SYMBOL_WIDTH = u32:32;
const TEST_COUNT_WIDTH  = u32:32;

type TestSymbol     = bits[TEST_SYMBOL_WIDTH];
type TestCount      = bits[TEST_COUNT_WIDTH];
type TestStimulus   = (TestSymbol, TestCount);
type TestDecInData  = DecInData<TEST_SYMBOL_WIDTH, TEST_COUNT_WIDTH>;
type TestDecOutData = DecOutData<TEST_SYMBOL_WIDTH>;

// Check RLE decoder on a transaction
#[test_proc]
proc RunLengthDecoderTransactionTest {
  terminator: chan<bool> out;            // test termination request
  dec_input_s: chan<TestDecInData> out;
  dec_output_r: chan<TestDecOutData> in;

  init {()}

  config(terminator: chan<bool> out) {
    let (dec_input_s, dec_input_r)   = chan<TestDecInData>("dec_input");
    let (dec_output_s, dec_output_r) = chan<TestDecOutData>("dec_output");

    spawn RunLengthDecoder<TEST_SYMBOL_WIDTH, TEST_COUNT_WIDTH>(
      dec_input_r, dec_output_s);
    (terminator, dec_input_s, dec_output_r)
  }

  next(tok: token, state: ()) {
    let TransactionTestStimuli: TestStimulus[6] =[
      (TestSymbol:0xB, TestCount:0x2),
      (TestSymbol:0x1, TestCount:0x1),
      (TestSymbol:0xC, TestCount:0x3),
      (TestSymbol:0xC, TestCount:0x3),
      (TestSymbol:0x3, TestCount:0x3),
      (TestSymbol:0x2, TestCount:0x2),
    ];
    let tok = for ((counter, stimulus), tok):
        ((u32, (TestSymbol, TestCount)) , token)
        in enumerate(TransactionTestStimuli) {
      let last = counter == (array_size(TransactionTestStimuli) - u32:1);
      let data_in = TestDecInData{
        symbol: stimulus.0,
        count: stimulus.1,
        last: last
      };
      let tok = send(tok, dec_input_s, data_in);
      trace_fmt!("Sent {} stimuli, symbol: 0x{:x}, count:{}, last: {}",
          counter + u32:1, data_in.symbol, data_in.count, data_in.last);
      (tok)
    }(tok);
    let TransationTestOutputs: TestSymbol[14] = [
      TestSymbol: 0xB, TestSymbol: 0xB,
      TestSymbol: 0x1, TestSymbol: 0xC,
      TestSymbol: 0xC, TestSymbol: 0xC,
      TestSymbol: 0xC, TestSymbol: 0xC,
      TestSymbol: 0xC, TestSymbol: 0x3,
      TestSymbol: 0x3, TestSymbol: 0x3,
      TestSymbol: 0x2, TestSymbol: 0x2,
    ];
    let tok = for ((counter, symbol), tok):
        ((u32, TestSymbol) , token)
        in enumerate(TransationTestOutputs) {
      let last = counter == (array_size(TransationTestOutputs) - u32:1);
      let data_out = TestDecOutData{
        symbol: symbol,
        last: last
      };
      let (tok, dec_output) = recv(tok, dec_output_r);
      trace_fmt!(
          "Received {} transactions, symbol: 0x{:x}, last: {}",
          counter, dec_output.symbol, dec_output.last
      );
      assert_eq(dec_output, data_out);
      (tok)
    }(tok);
    send(tok, terminator, true);
  }
}

// Check that RLE decoder will create 2 `last` output packets,
// when 2 `last` input packets were consumed.
#[test_proc]
proc RunLengthDecoderLastAfterLastTest {
  terminator: chan<bool> out;            // test termination request
  dec_input_s: chan<TestDecInData> out;
  dec_output_r: chan<TestDecOutData> in;

  init {()}

  config(terminator: chan<bool> out) {
    let (dec_input_s, dec_input_r)   = chan<TestDecInData>("dec_input");
    let (dec_output_s, dec_output_r) = chan<TestDecOutData>("dec_output");

    spawn RunLengthDecoder<TEST_SYMBOL_WIDTH, TEST_COUNT_WIDTH>(
        dec_input_r, dec_output_s);
    (terminator, dec_input_s, dec_output_r)
  }

  next(tok: token, state: ()) {
    let LastAfterLastTestStimuli: TestDecInData[2] =[
      TestDecInData {
        symbol: TestSymbol:0x1,
        count: TestCount:0x1,
        last:true
      },
      TestDecInData {
        symbol: TestSymbol:0x2,
        count: TestCount:0x1,
        last:true
      },
    ];
    let tok = for ((counter, stimulus), tok):
        ((u32, TestDecInData) , token)
        in enumerate(LastAfterLastTestStimuli) {
      let tok = send(tok, dec_input_s, stimulus);
      trace_fmt!("Sent {} stimuli, symbol: 0x{:x}, count:{}, last: {}",
          counter + u32:1, stimulus.symbol, stimulus.count, stimulus.last);
      (tok)
    }(tok);
    let LastAfterLastTestOutputs: TestDecOutData[2] = [
      TestDecOutData{symbol: TestSymbol: 0x1, last: true},
      TestDecOutData{symbol: TestSymbol: 0x2, last: true},
    ];
    let tok = for ((counter, output), tok):
        ((u32, TestDecOutData) , token)
        in enumerate(LastAfterLastTestOutputs) {
      let (tok, dec_output) = recv(tok, dec_output_r);
      trace_fmt!(
          "Received {} transactions, symbol: 0x{:x}, last: {}",
          counter + u32:1, dec_output.symbol, dec_output.last
      );
      assert_eq(dec_output, output);
      (tok)
    }(tok);
    send(tok, terminator, true);
  }
}
