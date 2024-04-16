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

import std;
import xls.modules.shift_buffer.shift_buffer;

const TEST_DATA_WIDTH = u32:64;
const TEST_DATA_WIDTH_X2 = u32:128;
const TEST_LENGTH_WIDTH = std::clog2(TEST_DATA_WIDTH) + u32:1;

proc ShiftBufferInst {
    type Input = shift_buffer::ShiftBufferPacket;
    type Ctrl = shift_buffer::ShiftBufferCtrl;
    type Output = shift_buffer::ShiftBufferOutput;
    input_r: chan<Input<TEST_DATA_WIDTH, TEST_LENGTH_WIDTH>> in;
    ctrl_r: chan<Ctrl<TEST_LENGTH_WIDTH>> in;
    output_s: chan<Output<TEST_DATA_WIDTH, TEST_LENGTH_WIDTH>> out;

    config(input_r: chan<Input<TEST_DATA_WIDTH, TEST_LENGTH_WIDTH>> in,
           ctrl_r: chan<Ctrl<TEST_LENGTH_WIDTH>> in,
           output_s: chan<Output<TEST_DATA_WIDTH, TEST_LENGTH_WIDTH>> out) {

        spawn shift_buffer::ShiftBuffer<TEST_DATA_WIDTH, TEST_LENGTH_WIDTH>(ctrl_r, input_r, output_s);

        (input_r, ctrl_r, output_s)
    }

    init {  }

    next(state: ()) {}
}

proc ShiftBufferAlignerInst {
    type Input = shift_buffer::ShiftBufferPacket<TEST_DATA_WIDTH, TEST_LENGTH_WIDTH>;
    type Inter = shift_buffer::ShiftBufferPacket<TEST_DATA_WIDTH_X2, TEST_LENGTH_WIDTH>;

    config(input: chan<Input> in, inter: chan<Inter> out) {
        spawn shift_buffer::ShiftBufferAligner<TEST_DATA_WIDTH, TEST_LENGTH_WIDTH>(input, inter);
    }

    init {  }

    next(state: ()) {  }
}

proc ShiftBufferStorageInst {
    type Ctrl = shift_buffer::ShiftBufferCtrl<TEST_LENGTH_WIDTH>;
    type Inter = shift_buffer::ShiftBufferPacket<TEST_DATA_WIDTH_X2, TEST_LENGTH_WIDTH>;
    type Output = shift_buffer::ShiftBufferOutput<TEST_DATA_WIDTH, TEST_LENGTH_WIDTH>;

    config(ctrl: chan<Ctrl> in, inter: chan<Inter> in, output: chan<Output> out) {
        spawn shift_buffer::ShiftBufferStorage<TEST_DATA_WIDTH, TEST_LENGTH_WIDTH>(ctrl, inter, output);
    }

    init {  }

    next(state: ()) {  }
}


