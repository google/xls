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

// This file contains the implementation of a buffering proc responsible for
// reading the data stream and dividing it into the smallest packets of the
// requested size.

import std;
import xls.modules.zstd.buffer;

fn buffer_width(data_width: u32) -> u32 { data_width * u32:2 }

pub struct ShiftBufferInput<DATA_WIDTH: u32, LENGTH_WIDTH: u32> {
    data: uN[DATA_WIDTH],
    length: uN[LENGTH_WIDTH],
    last: bool,
}

pub struct ShiftBufferOutput<DATA_WIDTH: u32, LENGTH_WIDTH: u32> {
    data: uN[DATA_WIDTH],
    length: uN[LENGTH_WIDTH],
    last: bool,
}

pub enum ShiftBufferOutputType : u1 {
    DATA = 0,
    CTRL = 1,
}

pub struct ShiftBufferCtrl<LENGTH_WIDTH: u32> {
    length: uN[LENGTH_WIDTH],
    output: ShiftBufferOutputType,
}

enum ShiftBufferStatus : u1 {
    REFILL = 0,
    FLUSH = 1,
}

struct ShiftBufferState<BUFFER_WIDTH: u32, LENGTH_WIDTH: u32> {
    status: ShiftBufferStatus,
    buff: buffer::Buffer<BUFFER_WIDTH>,
    ctrl: ShiftBufferCtrl<LENGTH_WIDTH>,
    ctrl_valid: bool,
    last: bool,
}

pub fn mask_data<DATA_WIDTH: u32, LENGTH_WIDTH: u32>
    (data: bits[DATA_WIDTH], length: bits[LENGTH_WIDTH]) -> bits[DATA_WIDTH] {
    type Data = bits[DATA_WIDTH];
    type Length = bits[LENGTH_WIDTH];

    let mask = if length != Length:0 { (Data:1 << length) - Data:1 } else { Data:0 };
    data & mask
}

pub proc ShiftBuffer<DATA_WIDTH: u32, LENGTH_WIDTH: u32, BUFFER_WIDTH: u32 = {buffer_width(DATA_WIDTH)}> {
    in_data_r: chan<ShiftBufferInput<DATA_WIDTH, LENGTH_WIDTH>> in;
    in_ctrl_r: chan<ShiftBufferCtrl<LENGTH_WIDTH>> in;
    out_data_s: chan<ShiftBufferOutput<DATA_WIDTH, LENGTH_WIDTH>> out;
    out_ctrl_s: chan<ShiftBufferOutput<DATA_WIDTH, LENGTH_WIDTH>> out;

    config(in_data_r: chan<ShiftBufferInput<DATA_WIDTH, LENGTH_WIDTH>> in,
           in_ctrl_r: chan<ShiftBufferCtrl<LENGTH_WIDTH>> in,
           out_data_s: chan<ShiftBufferOutput<DATA_WIDTH, LENGTH_WIDTH>> out,
           out_ctrl_s: chan<ShiftBufferOutput<DATA_WIDTH, LENGTH_WIDTH>> out) {
        (in_data_r, in_ctrl_r, out_data_s, out_ctrl_s)
    }

    init {
        type BufferState = ShiftBufferState<BUFFER_WIDTH, LENGTH_WIDTH>;
        zero!<BufferState>()
    }

    next(state: ShiftBufferState<BUFFER_WIDTH, LENGTH_WIDTH>) {
        type Data = uN[DATA_WIDTH];
        type Length = uN[LENGTH_WIDTH];
        type BufferData = uN[BUFFER_WIDTH];
        type Input = ShiftBufferInput<DATA_WIDTH, LENGTH_WIDTH>;
        type Ctrl = ShiftBufferCtrl<LENGTH_WIDTH>;
        type Output = ShiftBufferOutput<DATA_WIDTH, LENGTH_WIDTH>;
        type State = ShiftBufferState<BUFFER_WIDTH, LENGTH_WIDTH>;
        type Status = ShiftBufferStatus;
        type OutputType = ShiftBufferOutputType;

        let tok0 = join();

        // Receive control
        let (tok1_0, ctrl, ctrl_valid) =
            recv_if_non_blocking(tok0, in_ctrl_r, !state.ctrl_valid, state.ctrl);

        // Receive data
        let can_fit = buffer::buffer_can_fit(state.buff, Data:0);
        let do_recv_data = can_fit && state.status == Status::REFILL;
        let (tok1_1, recv_data, recv_data_valid) =
            recv_if_non_blocking(tok0, in_data_r, do_recv_data, zero!<Input>());

        let tok1 = join(tok1_0, tok1_1);

        // Handle the request
        // Uses buffer from the previous next evaluation (from state) to prevent
        // creating long combinatorial logic that would allow receiving and
        // sending back the data in the same cycle.

        let has_valid_request = ctrl_valid || state.ctrl_valid;
        let has_enough_data = buffer::buffer_has_at_least(state.buff, ctrl.length as u32);

        let (data, do_send, new_buff, new_ctrl, new_ctrl_valid) = if has_valid_request && has_enough_data {
            let (buff, data) = buffer::buffer_pop(state.buff, ctrl.length as u32);
            let last = buff.length == u32:0 && state.last;
            let output = Output { data: data as Data, length: ctrl.length, last };
            let do_send = ctrl.length != Length:0;
            (output, do_send, buff, zero!<Ctrl>(), false)
        } else {
            (zero!<Output>(), false, state.buff, ctrl, has_valid_request)
        };

        let do_send_data = do_send && ctrl.output == OutputType::DATA;
        let tok2_0 = send_if(tok1, out_data_s, do_send_data, data);
        let do_send_ctrl = do_send && ctrl.output == OutputType::CTRL;
        let tok2_1 = send_if(tok1, out_ctrl_s, do_send_ctrl, data);
        let tok2 = join(tok2_0, tok2_1);

        // Handle input data
        let (new_buff, new_last) = if can_fit && recv_data_valid {
            (
                buffer::buffer_append_with_length(
                    new_buff, recv_data.data as BufferData, recv_data.length as u32),
                recv_data.last,
            )
        } else {
            (new_buff, state.last)
        };

        // Handle state change
        let new_status = if state.status == Status::REFILL && recv_data_valid && recv_data.last {
            Status::FLUSH
        } else if state.status == Status::FLUSH && state.buff.length == ctrl.length as u32 {
            Status::REFILL
        } else {
            state.status
        };

        State {
            status: new_status,
            buff: new_buff,
            ctrl: new_ctrl,
            ctrl_valid: new_ctrl_valid,
            last: new_last
        }
    }
}

const INST_DATA_WIDTH = u32:64;
const INST_LENGTH_WIDTH = std::clog2(INST_DATA_WIDTH) + u32:1;

proc ShiftBufferInst {
    config(in_data_r: chan<ShiftBufferInput<INST_DATA_WIDTH, INST_LENGTH_WIDTH>> in,
           in_ctrl_r: chan<ShiftBufferCtrl<INST_LENGTH_WIDTH>> in,
           out_data_s: chan<ShiftBufferOutput<INST_DATA_WIDTH, INST_LENGTH_WIDTH>> out,
           out_ctrl_s: chan<ShiftBufferOutput<INST_DATA_WIDTH, INST_LENGTH_WIDTH>> out) {
        spawn ShiftBuffer<INST_DATA_WIDTH, INST_LENGTH_WIDTH>(
            in_data_r, in_ctrl_r, out_data_s, out_ctrl_s);
    }

    init {  }

    next(state: ()) {  }
}

const TEST_DATA_WIDTH = u32:64;
const TEST_LENGTH_WIDTH = std::clog2(TEST_DATA_WIDTH) + u32:1;

#[test_proc]
proc ShiftBufferTest {
    terminator: chan<bool> out;
    in_data_s: chan<ShiftBufferInput<TEST_DATA_WIDTH, TEST_LENGTH_WIDTH>> out;
    in_ctrl_s: chan<ShiftBufferCtrl<TEST_LENGTH_WIDTH>> out;
    out_data_r: chan<ShiftBufferOutput<TEST_DATA_WIDTH, TEST_LENGTH_WIDTH>> in;
    out_ctrl_r: chan<ShiftBufferOutput<TEST_DATA_WIDTH, TEST_LENGTH_WIDTH>> in;

    config(terminator: chan<bool> out) {
        let (in_data_s, in_data_r) = chan<ShiftBufferInput<TEST_DATA_WIDTH, TEST_LENGTH_WIDTH>>("in_data");
        let (in_ctrl_s, in_ctrl_r) = chan<ShiftBufferCtrl<TEST_LENGTH_WIDTH>>("in_ctrl");
        let (out_data_s, out_data_r) = chan<ShiftBufferOutput<TEST_DATA_WIDTH, TEST_LENGTH_WIDTH>>("out_data");
        let (out_ctrl_s, out_ctrl_r) = chan<ShiftBufferOutput<TEST_DATA_WIDTH, TEST_LENGTH_WIDTH>>("out_ctrl");

        spawn ShiftBuffer<TEST_DATA_WIDTH, TEST_LENGTH_WIDTH>(
            in_data_r, in_ctrl_r, out_data_s, out_ctrl_s);
        (terminator, in_data_s, in_ctrl_s, out_data_r, out_ctrl_r)
    }

    init {  }

    next(state: ()) {
        type Data = uN[TEST_DATA_WIDTH];
        type Length = uN[TEST_LENGTH_WIDTH];
        type OutputType = ShiftBufferOutputType;
        type Input = ShiftBufferInput;
        type Output = ShiftBufferOutput;
        type Ctrl = ShiftBufferCtrl;

        let tok = join();

        let tok = send(tok, in_data_s, Input { data: Data:0xDD_44, length: Length:16, last: false });
        let tok = send(tok, in_data_s, Input { data: Data:0xAA_11_BB_22_CC_33, length: Length:48, last: false });
        let tok = send(tok, in_data_s, Input { data: Data:0xEE_55_FF_66_00_77_11_88, length: Length:64, last: false });

        let tok = send(tok, in_ctrl_s, Ctrl { length: Length:8, output: OutputType::DATA });
        let (tok, output) = recv(tok, out_data_r);
        assert_eq(output, Output { data: Data:0x44, length: Length:8, last: false });

        let tok = send(tok, in_ctrl_s, Ctrl { length: Length:4, output: OutputType::DATA });
        let (tok, output) = recv(tok, out_data_r);
        assert_eq(output, Output { data: Data:0xD, length: Length:4, last: false });

        let tok = send(tok, in_ctrl_s, Ctrl { length: Length:64, output: OutputType::CTRL });
        let (tok, output) = recv(tok, out_ctrl_r);
        assert_eq(output, Output { data: Data:0x18_8A_A1_1B_B2_2C_C3_3D, length: Length:64, last: false });

        let tok = send(tok, in_data_s, Input { data: Data:0x44_BB_55_CC, length: Length:32, last: false });
        let tok = send(tok, in_data_s, Input { data: Data:0x22_99_33_AA, length: Length:32, last: true });
        let tok = send(tok, in_data_s, Input { data: Data:0x66_DD_77_EE_88_FF_99_00, length: Length:64, last: false });

        let tok = send(tok, in_ctrl_s, Ctrl { length: Length:4, output: OutputType::CTRL });
        let (tok, output) = recv(tok, out_ctrl_r);
        assert_eq(output, Output { data: Data:0x1, length: Length:4, last: false });

        let tok = send(tok, in_ctrl_s, Ctrl { length: Length:64, output: OutputType::DATA });
        let (tok, output) = recv(tok, out_data_r);
        assert_eq(output, Output { data: Data:0x55_CC_EE_55_FF_66_00_77, length: Length:64, last: false });

        let tok = send(tok, in_ctrl_s, Ctrl { length: Length:16, output: OutputType::DATA });
        let (tok, output) = recv(tok, out_data_r);
        assert_eq(output, Output { data: Data:0x44_BB, length: Length:16, last: false });

        let tok = send(tok, in_ctrl_s, Ctrl { length: Length:32, output: OutputType::CTRL });
        let (tok, output) = recv(tok, out_ctrl_r);
        assert_eq(output, Output { data: Data:0x22_99_33_AA, length: Length:32, last: true });

        let tok = send(tok, in_ctrl_s, Ctrl { length: Length:64, output: OutputType::DATA });

        let (tok, output) = recv(tok, out_data_r);
        assert_eq(output, Output { data: Data:0x66_DD_77_EE_88_FF_99_00, length: Length:64, last: false });

        send(tok, terminator, true);
    }
}
