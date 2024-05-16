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

pub struct ShiftBufferInput<DATA_W: u32, LENGTH_W: u32> {
    data: uN[DATA_W],
    length: uN[LENGTH_W],
    last: bool,
}

pub struct ShiftBufferOutput<DATA_W: u32, LENGTH_W: u32> {
    data: uN[DATA_W],
    length: uN[LENGTH_W],
    last: bool,
}

pub struct ShiftBufferCtrl<LENGTH_W: u32> {
    length: uN[LENGTH_W],
}

enum ShiftBufferStatus : u1 {
    REFILL = 0,
    FLUSH = 1,
}

struct ShiftBufferState<BUFFER_W: u32, LENGTH_W: u32> {
    status: ShiftBufferStatus,
    buff: buffer::Buffer<BUFFER_W>,
    ctrl: ShiftBufferCtrl<LENGTH_W>,
    ctrl_valid: bool,
    last: bool,
}

pub fn mask_data<DATA_W: u32, LENGTH_W: u32> (data: bits[DATA_W], length: bits[LENGTH_W]) -> bits[DATA_W] {
    type Data = bits[DATA_W];
    type Length = bits[LENGTH_W];

    let mask = if length != Length:0 { (Data:1 << length) - Data:1 } else { Data:0 };
    data & mask
}

pub proc ShiftBuffer<
    DATA_W: u32, LENGTH_W: u32,
    BUFFER_W: u32 = {buffer_width(DATA_W)}
> {
    type Data = uN[DATA_W];
    type Length = uN[LENGTH_W];
    type BufferData = uN[BUFFER_W];
    type Input = ShiftBufferInput<DATA_W, LENGTH_W>;
    type Ctrl = ShiftBufferCtrl<LENGTH_W>;
    type Output = ShiftBufferOutput<DATA_W, LENGTH_W>;
    type State = ShiftBufferState<BUFFER_W, LENGTH_W>;
    type Status = ShiftBufferStatus;

    in_data_r: chan<Input> in;
    in_ctrl_r: chan<Ctrl> in;
    out_data_s: chan<Output> out;

    config(
        in_data_r: chan<Input> in,
        in_ctrl_r: chan<Ctrl> in,
        out_data_s: chan<Output> out,
    ) { (in_data_r, in_ctrl_r, out_data_s) }

    init { zero!<State>() }

    next(state: State) {

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

        let tok2_0 = send_if(tok1, out_data_s, do_send, data);

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

const INST_DATA_W = u32:64;
const INST_LENGTH_W = std::clog2(INST_DATA_W) + u32:1;

proc ShiftBufferInst {
    type Input = ShiftBufferInput<INST_DATA_W, INST_LENGTH_W>;
    type Ctrl = ShiftBufferCtrl<INST_LENGTH_W>;
    type Output = ShiftBufferOutput<INST_DATA_W, INST_LENGTH_W>;

    config(
        data_r: chan<Input> in,
        ctrl_r: chan<Ctrl> in,
        out_s: chan<Output> out
    ) {
        spawn ShiftBuffer<INST_DATA_W, INST_LENGTH_W>(data_r, ctrl_r, out_s);
    }

    init {  }

    next(state: ()) {  }
}

const TEST_DATA_W = u32:64;
const TEST_LENGTH_W = std::clog2(TEST_DATA_W) + u32:1;

#[test_proc]
proc ShiftBufferTest {
    type Input = ShiftBufferInput<TEST_DATA_W, TEST_LENGTH_W>;
    type Ctrl = ShiftBufferCtrl<TEST_LENGTH_W>;
    type Output = ShiftBufferOutput<TEST_DATA_W, TEST_LENGTH_W>;
    type Data = uN[TEST_DATA_W];
    type Length = uN[TEST_LENGTH_W];

    terminator: chan<bool> out;
    data_s: chan<Input> out;
    ctrl_s: chan<Ctrl> out;
    out_r: chan<Output> in;

    config(terminator: chan<bool> out) {
        let (data_s, data_r) = chan<Input>("in_data");
        let (ctrl_s, ctrl_r) = chan<Ctrl>("in_ctrl");
        let (out_s, out_r) = chan<Output>("out_data");

        spawn ShiftBuffer<TEST_DATA_W, TEST_LENGTH_W>(data_r, ctrl_r, out_s);
        (terminator, data_s, ctrl_s, out_r)
    }

    init {  }

    next(state: ()) {

        let tok = join();
        let tok = send(tok, data_s, Input { data: Data:0xDD_44, length: Length:16, last: false });
        let tok = send(tok, data_s, Input { data: Data:0xAA_11_BB_22_CC_33, length: Length:48, last: false });
        let tok = send(tok, data_s, Input { data: Data:0xEE_55_FF_66_00_77_11_88, length: Length:64, last: false });

        let tok = send(tok, ctrl_s, Ctrl { length: Length:8 });
        let (tok, output) = recv(tok, out_r);
        assert_eq(output, Output { data: Data:0x44, length: Length:8, last: false });

        let tok = send(tok, ctrl_s, Ctrl { length: Length:4 });
        let (tok, output) = recv(tok, out_r);
        assert_eq(output, Output { data: Data:0xD, length: Length:4, last: false });

        let tok = send(tok, ctrl_s, Ctrl { length: Length:64 });
        let (tok, output) = recv(tok, out_r);
        assert_eq(output, Output { data: Data:0x18_8A_A1_1B_B2_2C_C3_3D, length: Length:64, last: false });

        let tok = send(tok, data_s, Input { data: Data:0x44_BB_55_CC, length: Length:32, last: false });
        let tok = send(tok, data_s, Input { data: Data:0x22_99_33_AA, length: Length:32, last: true });
        let tok = send(tok, data_s, Input { data: Data:0x66_DD_77_EE_88_FF_99_00, length: Length:64, last: false });

        let tok = send(tok, ctrl_s, Ctrl { length: Length:4 });
        let (tok, output) = recv(tok, out_r);
        assert_eq(output, Output { data: Data:0x1, length: Length:4, last: false });

        let tok = send(tok, ctrl_s, Ctrl { length: Length:64 });
        let (tok, output) = recv(tok, out_r);
        assert_eq(output, Output { data: Data:0x55_CC_EE_55_FF_66_00_77, length: Length:64, last: false });

        let tok = send(tok, ctrl_s, Ctrl { length: Length:16 });
        let (tok, output) = recv(tok, out_r);
        assert_eq(output, Output { data: Data:0x44_BB, length: Length:16, last: false });

        let tok = send(tok, ctrl_s, Ctrl { length: Length:32 });
        let (tok, output) = recv(tok, out_r);
        assert_eq(output, Output { data: Data:0x22_99_33_AA, length: Length:32, last: true });

        let tok = send(tok, ctrl_s, Ctrl { length: Length:64 });

        let (tok, output) = recv(tok, out_r);
        assert_eq(output, Output { data: Data:0x66_DD_77_EE_88_FF_99_00, length: Length:64, last: false });

        send(tok, terminator, true);
    }
}
