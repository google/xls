// Copyright 2024 The XLS Authors
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

// Tests run-length encoding with a massive data set.

import std;
import xls.modules.rle.rle_common as rle_common;
import xls.modules.rle.rle_enc as rle_enc;

const SYMBOL_WIDTH = u32:32;
const COUNT_WIDTH = SYMBOL_WIDTH + u32:1;

type EncInData = rle_common::PlainData<SYMBOL_WIDTH>;
type EncOutData = rle_common::CompressedData<SYMBOL_WIDTH, COUNT_WIDTH>;

struct GeneratorState { cur_sym: u32, cur_rep: u32, target_rep: u32 }

proc GenerateCount {
    output: chan<EncInData> out;

    config(output: chan<EncInData> out) { (output,) }

    init { GeneratorState { cur_sym: u32:1, cur_rep: u32:0, target_rep: u32:1 } }

    next(state: GeneratorState) {
        let sym = if state.cur_sym == u32:0 { u32:1 } else { state.cur_sym };
        let reps = state.cur_rep + u32:1;
        send(join(), output, EncInData { symbol: state.cur_sym, last: false });
        if reps >= state.target_rep {
            let next_rep = state.target_rep + u32:1;
            GeneratorState {
                cur_sym: sym + u32:1,
                cur_rep: u32:0,
                target_rep: if next_rep == u32:0 { u32:1 } else { next_rep }
            }
        } else {
            GeneratorState { cur_sym: sym, cur_rep: reps, target_rep: state.target_rep }
        }
    }
}

pub proc CompressCount<REPS: u32> {
    terminate: chan<bool> out;
    compressed_res: chan<EncOutData> in;

    config(terminate: chan<bool> out) {
        let (enc_input_s, enc_input_r) = chan<EncInData>("enc_input");
        let (enc_output_s, enc_output_r) = chan<EncOutData>("enc_output");
        spawn GenerateCount(enc_input_s);
        spawn rle_enc::RunLengthEncoder<u32:32, u32:33>(enc_input_r, enc_output_s);
        (terminate, enc_output_r)
    }

    init { u32:1 }

    next(cnt: u32) {
        let (tok, v) = recv(join(), compressed_res);

        assert_eq(v, EncOutData { symbol: cnt, count: cnt as u33, last: false });

        send_if(tok, terminate, cnt == REPS, u1:1);
        cnt + u32:1
    }
}
