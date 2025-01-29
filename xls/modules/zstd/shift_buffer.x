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
import xls.modules.zstd.math;

pub enum ShiftBufferStatus : u1 {
    OK = 0,
    ERROR = 1,
}

pub fn length_width(data_width: u32) -> u32 {
    std::clog2(data_width + u32:1)
}

// Common definition for buffer input and output payload
pub struct ShiftBufferPacket<DATA_WIDTH: u32, LENGTH_WIDTH: u32> {
    data: uN[DATA_WIDTH],
    length: uN[LENGTH_WIDTH],
}

// Output structure - packet with embedded status of the buffer operation
pub type ShiftBufferOutput = ShiftBufferPacket;

// Buffer pop command
pub struct ShiftBufferCtrl<LENGTH_WIDTH: u32> {
    length: uN[LENGTH_WIDTH]
}

struct ShiftBufferAlignerState<LENGTH_WIDTH: u32> {
    ptr: uN[LENGTH_WIDTH]
}

pub proc ShiftBufferAligner<
    DATA_WIDTH: u32,
    LENGTH_WIDTH: u32 = {length_width(DATA_WIDTH)},
    DATA_WIDTH_X2: u32 = {DATA_WIDTH * u32:2},
> {
    type Length = uN[LENGTH_WIDTH];
    type Data = uN[DATA_WIDTH];
    type DataX2 = uN[DATA_WIDTH_X2];

    type State = ShiftBufferAlignerState<LENGTH_WIDTH>;
    type Input = ShiftBufferPacket<DATA_WIDTH, LENGTH_WIDTH>;
    type Inter = ShiftBufferPacket<DATA_WIDTH_X2, LENGTH_WIDTH>;

    input_r: chan<Input> in;
    inter_s: chan<Inter> out;

    config(
        input_r: chan<Input> in,
        inter_s: chan<Inter> out,
    ) {
        (input_r, inter_s)
    }

    init {zero!<State>()}

    next(state: State) {
        // FIXME: Remove when https://github.com/google/xls/issues/1368 is resolved
        type Inter = ShiftBufferPacket<DATA_WIDTH_X2, LENGTH_WIDTH>;

        let tok = join();

        let (tok0, data) = recv(tok, input_r);
        let tok0 = send(tok0, inter_s, Inter {
            length: data.length,
            data: math::logshiftl(data.data as DataX2, state.ptr),
        });

        State {ptr: (state.ptr + data.length) % (DATA_WIDTH as Length) }
    }
}

const ALIGNER_TEST_DATA_WIDTH = u32:64;
const ALIGNER_TEST_LENGTH_WIDTH = length_width(ALIGNER_TEST_DATA_WIDTH);
const ALIGNER_TEST_DATA_WIDTH_X2 = ALIGNER_TEST_DATA_WIDTH * u32:2;

#[test_proc]
proc ShiftBufferAlignerTest {
    terminator: chan<bool> out;
    type Input = ShiftBufferPacket<ALIGNER_TEST_DATA_WIDTH, ALIGNER_TEST_LENGTH_WIDTH>;
    type Inter = ShiftBufferPacket<ALIGNER_TEST_DATA_WIDTH_X2, ALIGNER_TEST_LENGTH_WIDTH>;

    type Data = uN[ALIGNER_TEST_DATA_WIDTH];
    type Length = uN[ALIGNER_TEST_LENGTH_WIDTH];
    type DataX2 = uN[ALIGNER_TEST_DATA_WIDTH_X2];

    input_s: chan<Input> out;
    inter_r: chan<Inter> in;

    config(terminator: chan<bool> out) {
        let (input_s, input_r) = chan<Input>("input");
        let (inter_s, inter_r) = chan<Inter>("inter");

        spawn ShiftBufferAligner<ALIGNER_TEST_DATA_WIDTH>(input_r, inter_s);

        (terminator, input_s, inter_r)
    }

    init {  }

    next(state: ()) {
        let tok = send(join(), input_s, Input { data: Data:0xAABB_CCDD, length: Length:32});
        let tok = send(tok, input_s, Input { data: Data:0x1122, length: Length:16});
        let tok = send(tok, input_s, Input { data: Data:0x33, length: Length:8});
        let tok = send(tok, input_s, Input { data: Data:0x44, length: Length:8});
        let tok = send(tok, input_s, Input { data: Data:0xFFFF, length: Length:4});
        let tok = send(tok, input_s, Input { data: Data:0x0, length: Length:0});
        let tok = send(tok, input_s, Input { data: Data:0x0, length: Length:4});
        let tok = send(tok, input_s, Input { data: Data:0x1, length: Length:1});
        let tok = send(tok, input_s, Input { data: Data:0xF, length: Length:3});
        let tok = send(tok, input_s, Input { data: Data:0xF, length: Length:4});

        let (tok, data) = recv(tok, inter_r);
        assert_eq(data, Inter { data: DataX2: 0xAABB_CCDD, length: Length: 32});
        let (tok, data) = recv(tok, inter_r);
        assert_eq(data, Inter { data: DataX2: 0x1122_0000_0000, length: Length: 16});
        let (tok, data) = recv(tok, inter_r);
        assert_eq(data, Inter { data: DataX2: 0x33_0000_0000_0000, length: Length: 8});
        let (tok, data) = recv(tok, inter_r);
        assert_eq(data, Inter { data: DataX2: 0x4400_0000_0000_0000, length: Length: 8});
        let (tok, data) = recv(tok, inter_r);
        assert_eq(data, Inter { data: DataX2:0xFFFF, length: Length:4});
        let (tok, data) = recv(tok, inter_r);
        assert_eq(data, Inter { data: DataX2:0x0, length: Length:0});
        let (tok, data) = recv(tok, inter_r);
        assert_eq(data, Inter { data: DataX2:0x00, length: Length:4});
        let (tok, data) = recv(tok, inter_r);
        assert_eq(data, Inter { data: DataX2:0x100, length: Length:1});
        let (tok, data) = recv(tok, inter_r);
        assert_eq(data, Inter { data: DataX2:0x1E00, length: Length:3});
        let (tok, data) = recv(tok, inter_r);
        assert_eq(data, Inter { data: DataX2:0xF000, length: Length:4});

        send(tok, terminator, true);
    }
}

struct ShiftBufferStorageState<BUFFER_WIDTH: u32, LENGTH_WIDTH: u32> {
    buffer: bits[BUFFER_WIDTH],  // The storage element.
    buffer_cnt: bits[LENGTH_WIDTH + u32:2],  // Number of valid bits in the buffer.
    read_ptr: bits[LENGTH_WIDTH + u32:2],  // First occupied bit in the buffer when buffer_cnt > 0.
    write_ptr: bits[LENGTH_WIDTH + u32:2],  // First free bit in the buffer.
    cmd: ShiftBufferCtrl<LENGTH_WIDTH>,  // Received command of ShiftBufferCtrl type.
    cmd_valid: bool,  // Field cmd is valid.
}

pub proc ShiftBufferStorage<DATA_WIDTH: u32, LENGTH_WIDTH: u32> {
    type Buffer = bits[DATA_WIDTH * u32:3];
    type BufferLength = bits[LENGTH_WIDTH + u32:2]; // TODO: where does this "+ u32:2" come from? shouldn't it be number_of_bits_required_to_represent(DATA_WIDTH * u32:3)?
    type Data = bits[DATA_WIDTH];
    type DataLength = bits[LENGTH_WIDTH];
    type State = ShiftBufferStorageState;
    type Ctrl = ShiftBufferCtrl;
    type Inter = ShiftBufferPacket;
    type Output = ShiftBufferOutput;
    ctrl: chan<Ctrl<LENGTH_WIDTH>> in;
    inter: chan<Inter<{DATA_WIDTH * u32:2}, LENGTH_WIDTH>> in;
    output: chan<Output<DATA_WIDTH, LENGTH_WIDTH>> out;

    config(
        ctrl: chan<Ctrl<LENGTH_WIDTH>> in,
        inter: chan<Inter<{DATA_WIDTH * u32:2}, LENGTH_WIDTH>> in,
        output: chan<Output<DATA_WIDTH, LENGTH_WIDTH>> out,
    ) {
        (ctrl, inter, output)
    }

    init {
        type State = ShiftBufferStorageState<{DATA_WIDTH * u32:3}, LENGTH_WIDTH>;
        zero!<State>()
    }

    next(state: State<{DATA_WIDTH * u32:3}, LENGTH_WIDTH>) {
        type State = ShiftBufferStorageState<{DATA_WIDTH * u32:3}, LENGTH_WIDTH>;
        type Ctrl = ShiftBufferCtrl<LENGTH_WIDTH>;
        type Inter = ShiftBufferPacket<{DATA_WIDTH * u32:2}, LENGTH_WIDTH>;
        type Output = ShiftBufferOutput<DATA_WIDTH, LENGTH_WIDTH>;
        type OutputPayload = ShiftBufferPacket<DATA_WIDTH, LENGTH_WIDTH>;
        type OutputStatus = ShiftBufferStatus;
        type DataLength = bits[LENGTH_WIDTH];
        // trace_fmt!("state: {:#x}", state);

        const MAX_BUFFER_CNT = (DATA_WIDTH * u32:3) as BufferLength;

        let shift_buffer_right = state.read_ptr >= (DATA_WIDTH as BufferLength);
        // trace_fmt!("shift_buffer_right: {:#x}", shift_buffer_right);
        let shift_data_left =
            state.write_ptr >= (DATA_WIDTH as BufferLength) && !shift_buffer_right;
        // trace_fmt!("shift_data_left: {:#x}", shift_data_left);
        let recv_new_input = state.write_ptr < (DATA_WIDTH * u32:2) as BufferLength;
        // trace_fmt!("recv_new_input: {:#x}", recv_new_input);
        let has_enough_data = (state.cmd.length as BufferLength <= state.buffer_cnt);
        let send_response = state.cmd_valid && has_enough_data;
        // trace_fmt!("send_response: {:#x}", send_response);
        let recv_new_cmd = !state.cmd_valid || send_response;
        // trace_fmt!("recv_new_cmd: {:#x}", recv_new_cmd);

        let tok = join();

        // Shift buffer if required
        let (new_buffer, new_read_ptr, new_write_ptr) = if shift_buffer_right {
            (state.buffer >> DATA_WIDTH,
            state.read_ptr - DATA_WIDTH as BufferLength,
            state.write_ptr - DATA_WIDTH as BufferLength)
        } else {
            (state.buffer,
            state.read_ptr,
            state.write_ptr)
        };

        // if (shift_buffer_right) {
        //     trace_fmt!("Shifted data");
        //     trace_fmt!("new_buffer: {:#x}", new_buffer);
        //     trace_fmt!("new_read_ptr: {}", new_read_ptr);
        //     trace_fmt!("new_write_ptr: {}", new_write_ptr);
        // } else { () };

        // Handle incoming writes
        let (tok_input, wdata, wdata_valid) = recv_if_non_blocking(tok, inter, recv_new_input, zero!<Inter>());

        let (new_buffer, new_write_ptr) = if wdata_valid {
            // Shift data if required
            let new_data = if shift_data_left {
                wdata.data as Buffer << DATA_WIDTH
            } else {
                wdata.data as Buffer
            };
            let new_buffer = new_buffer | new_data;
            let new_write_ptr = new_write_ptr + wdata.length as BufferLength;

            (new_buffer, new_write_ptr)
        } else {
            (new_buffer, new_write_ptr)
        };

        // if (wdata_valid) {
        //     trace_fmt!("Received aligned data {:#x}", wdata);
        //     trace_fmt!("new_buffer: {:#x}", new_buffer);
        //     trace_fmt!("new_write_ptr: {}", new_write_ptr);
        // } else { () };

        // Handle incoming reads
        let (tok_ctrl, new_cmd, new_cmd_valid) =
            recv_if_non_blocking(tok, ctrl, recv_new_cmd, state.cmd);

        // if (new_cmd_valid) {
        //     trace_fmt!("Received new cmd: {}", new_cmd);
        // } else {()};
        let new_cmd_valid = if recv_new_cmd { new_cmd_valid } else { state.cmd_valid };
        // Handle current read

        let (rdata, new_read_ptr) = if send_response {
            let new_read_ptr = new_read_ptr + state.cmd.length as BufferLength;
            let rdata = Output {
                length: state.cmd.length,
                data: math::mask(math::logshiftr(state.buffer, state.read_ptr) as Data, state.cmd.length),
            };

            // trace_fmt!("rdata: {:#x}", rdata);
            // trace_fmt!("new_read_ptr: {}", new_read_ptr);

            (rdata, new_read_ptr)
        } else {
            (zero!<Output>(), new_read_ptr)
        };

        let tok = join(tok_input, tok_ctrl);
        send_if(tok, output, send_response, rdata);
        // if (send_response) {
        //     trace_fmt!("Sent out rdata: {:#x}", rdata);
        // } else {()};

        let new_buffer_cnt = new_write_ptr - new_read_ptr;

        let new_state = State {
            buffer: new_buffer,
            buffer_cnt: new_buffer_cnt,
            read_ptr: new_read_ptr,
            write_ptr: new_write_ptr,
            cmd: new_cmd,
            cmd_valid: new_cmd_valid,
        };

        new_state
    }
}

const STORAGE_TEST_DATA_WIDTH = u32:64;
const STORAGE_TEST_LENGTH_WIDTH = length_width(STORAGE_TEST_DATA_WIDTH);
const STORAGE_TEST_DATA_WIDTH_X2 = STORAGE_TEST_DATA_WIDTH * u32:2;

#[test_proc]
proc ShiftBufferStorageTest {
    terminator: chan<bool> out;
    type Ctrl = ShiftBufferCtrl<STORAGE_TEST_LENGTH_WIDTH>;
    type Inter = ShiftBufferPacket<STORAGE_TEST_DATA_WIDTH_X2, STORAGE_TEST_LENGTH_WIDTH>;
    type Output = ShiftBufferOutput<STORAGE_TEST_DATA_WIDTH, STORAGE_TEST_LENGTH_WIDTH>;
    type OutputPayload = ShiftBufferPacket<STORAGE_TEST_DATA_WIDTH, STORAGE_TEST_LENGTH_WIDTH>;
    type OutputStatus = ShiftBufferStatus;

    type Length = uN[STORAGE_TEST_LENGTH_WIDTH];
    type Data = uN[STORAGE_TEST_DATA_WIDTH];
    type DataX2 = uN[STORAGE_TEST_DATA_WIDTH_X2];

    ctrl_s: chan<Ctrl> out;
    inter_s: chan<Inter> out;
    output_r: chan<Output> in;

    config(terminator: chan<bool> out) {
        let (ctrl_s, ctrl_r) = chan<Ctrl>("ctrl");
        let (inter_s, inter_r) = chan<Inter>("inter");
        let (output_s, output_r) = chan<Output>("output");

        spawn ShiftBufferStorage<STORAGE_TEST_DATA_WIDTH, STORAGE_TEST_LENGTH_WIDTH>(ctrl_r, inter_r, output_s);

        (terminator, ctrl_s, inter_s, output_r)
    }

    init {  }

    next(state: ()) {
        // Single input, single output packet 32bit buffering
        let tok = send(join(), inter_s, Inter { data: DataX2: 0xAABB_CCDD, length: Length: 32});

        // Multiple input packets, single output 32bit buffering
        let tok = send(tok, inter_s, Inter { data: DataX2: 0x3344_0000_0000, length: Length: 16});
        let tok = send(tok, inter_s, Inter { data: DataX2: 0x22_0000_0000_0000, length: Length: 8});
        let tok = send(tok, inter_s, Inter { data: DataX2: 0x1100_0000_0000_0000, length: Length: 8});

        // Small consecutive single input, single output 8bit buffering
        let tok = send(tok, inter_s, Inter { data: DataX2: 0x55, length: Length: 8});
        let tok = send(tok, inter_s, Inter { data: DataX2: 0x6600, length: Length: 8});

        // Multiple input packets, single output 64bit buffering
        let tok = send(tok, inter_s, Inter { data: DataX2: 0xDDEE_0000, length: Length: 16});
        let tok = send(tok, inter_s, Inter { data: DataX2: 0xBBCC_0000_0000, length: Length: 16});
        let tok = send(tok, inter_s, Inter { data: DataX2: 0x99AA_0000_0000_0000, length: Length: 16});
        let tok = send(tok, inter_s, Inter { data: DataX2: 0x7788, length: Length: 16});

        // Single input packet, single output 64bit buffering
        let tok = send(tok, inter_s, Inter { data: DataX2: 0x1122_3344_5566_7788_0000, length: Length: 64});

        // Single 64bit input packet, multiple output packets of different sizes
        let tok = send(tok, inter_s, Inter { data: DataX2: 0xEEFF_0011_CCDD_BBAA_0000, length: Length: 64});

        // Account for leftover 0xEEFF from the previous packet
        let tok = send(tok, inter_s, Inter { data: DataX2: 0x1122_0000, length: Length: 16});
        // Should operate on flushed buffer
        let tok = send(tok, inter_s, Inter { data: DataX2: 0x3344_0000_0000, length: Length: 16});

        // Input packets additionally span across 2 shift buffer aligner shift domains
        let tok = send(tok, inter_s, Inter { data: DataX2: 0x7788_0000_0000_0000, length: Length: 16});
        let tok = send(tok, inter_s, Inter { data: DataX2: 0x5566, length: Length: 16});

        // Single input, single output packet 32bit buffering
        let tok = send(tok, ctrl_s, Ctrl { length: Length:32});
        let (tok, data) = recv(tok, output_r);
        assert_eq(data, Output { data: Data: 0xAABB_CCDD, length: Length: 32});

        // Multiple input packets, single output 32bit buffering
        let tok = send(tok, ctrl_s, Ctrl { length: Length:32});
        let (tok, data) = recv(tok, output_r);
        assert_eq(data, Output { data: Data: 0x1122_3344, length: Length: 32});

        // Small consecutive single input, single output 8bit buffering
        let tok = send(tok, ctrl_s, Ctrl { length: Length:8});
        let (tok, data) = recv(tok, output_r);
        assert_eq(data, Output { data: Data: 0x55, length: Length: 8});
        let tok = send(tok, ctrl_s, Ctrl { length: Length:8});
        let (tok, data) = recv(tok, output_r);
        assert_eq(data, Output { data: Data: 0x66, length: Length: 8});

        // Multiple input packets, single output 64bit buffering
        let tok = send(tok, ctrl_s, Ctrl { length: Length:64});
        let (tok, data) = recv(tok, output_r);
        assert_eq(data, Output { data: Data: 0x7788_99AA_BBCC_DDEE, length: Length: 64});

        // Single input packet, single output 64bit buffering
        let tok = send(tok, ctrl_s, Ctrl { length: Length:64});
        let (tok, data) = recv(tok, output_r);
        assert_eq(data, Output { data: Data: 0x1122_3344_5566_7788, length: Length: 64});

        // Single 64bit input packet, multiple output packets of different sizes
        let tok = send(tok, ctrl_s, Ctrl { length: Length:8});
        let (tok, data) = recv(tok, output_r);
        assert_eq(data, Output { data: Data: 0xAA, length: Length: 8});
        let tok = send(tok, ctrl_s, Ctrl { length: Length:8});
        let (tok, data) = recv(tok, output_r);
        assert_eq(data, Output { data: Data: 0xBB, length: Length: 8});
        let tok = send(tok, ctrl_s, Ctrl { length: Length:16});
        let (tok, data) = recv(tok, output_r);
        assert_eq(data, Output { data: Data: 0xCCDD, length: Length: 16});
        let tok = send(tok, ctrl_s, Ctrl { length: Length:32});
        let (tok, data) = recv(tok, output_r);
        assert_eq(data, Output { data: Data: 0xEEFF_0011, length: Length: 32});

        let tok = send(tok, ctrl_s, Ctrl { length: Length:16});
        let (tok, data) = recv(tok, output_r);
        assert_eq(data, Output { data: Data: 0x1122, length: Length: 16});
        let tok = send(tok, ctrl_s, Ctrl { length: Length:16});
        let (tok, data) = recv(tok, output_r);
        assert_eq(data, Output { data: Data: 0x3344, length: Length: 16});

        let tok = send(tok, ctrl_s, Ctrl { length: Length:32});
        let (tok, data) = recv(tok, output_r);
        assert_eq(data, Output { data: Data: 0x5566_7788, length: Length: 32});

        // Test attempting to read more data than available in the buffer
        // This should wait indefinitely, we test this by checking that we can't
        // receive data over the next consecutive 100 iterations
        let tok = send(tok, ctrl_s, Ctrl { length: Length:64});
        let tok = for (_, tok): (u32, token) in u32:1..u32:100 {
            let (tok, _, data_valid) = recv_non_blocking(tok, output_r, zero!<Output>());
            assert_eq(data_valid, false);
            tok
        }(tok);

        // Refill the buffer with more data - not enough to reply to the earlier request for 64b
        let tok = send(tok, inter_s, Inter { data: DataX2: 0xDEAD_BEEF_0000, length: Length: 32});
        // Check that we can't receive still
        let tok = for (_, tok): (u32, token) in u32:1..u32:100 {
            let (tok, _, data_valid) = recv_non_blocking(tok, output_r, zero!<Output>());
            assert_eq(data_valid, false);
            tok
        }(tok);

        // Refill buffer with enough data
        let tok = send(tok, inter_s, Inter { data: DataX2: 0xF00B_A4BA_0000_0000_0000, length: Length: 32});
        // Now we should be able to receive a response for 64b request
        let (tok, data) = recv(tok, output_r);
        assert_eq(data, Output { data: Data: 0xF00BA4BA_DEADBEEF, length: Length: 64});

        send(tok, terminator, true);
    }
}

pub proc ShiftBuffer<DATA_WIDTH: u32, LENGTH_WIDTH: u32> {
    type Input = ShiftBufferPacket;
    type Ctrl = ShiftBufferCtrl;
    type Inter = ShiftBufferPacket;
    type Output = ShiftBufferOutput;

    config(ctrl: chan<Ctrl<LENGTH_WIDTH>> in, input: chan<Input<DATA_WIDTH, LENGTH_WIDTH>> in,
           output: chan<Output<DATA_WIDTH, LENGTH_WIDTH>> out) {
        let (inter_out, inter_in) =
            chan<ShiftBufferPacket<{DATA_WIDTH * u32:2}, LENGTH_WIDTH>, u32:1>("inter");
        spawn ShiftBufferAligner<DATA_WIDTH, LENGTH_WIDTH>(input, inter_out);
        spawn ShiftBufferStorage<DATA_WIDTH, LENGTH_WIDTH>(ctrl, inter_in, output);
    }

    init {  }

    next(state: ()) { }
}

const INST_DATA_WIDTH = u32:64;
const INST_DATA_WIDTH_X2 = u32:128;
const INST_LENGTH_WIDTH = std::clog2(INST_DATA_WIDTH) + u32:1;

proc ShiftBufferInst {
    type Input = ShiftBufferPacket;
    type Ctrl = ShiftBufferCtrl;
    type Output = ShiftBufferOutput;
    input_r: chan<Input<INST_DATA_WIDTH, INST_LENGTH_WIDTH>> in;
    ctrl_r: chan<Ctrl<INST_LENGTH_WIDTH>> in;
    output_s: chan<Output<INST_DATA_WIDTH, INST_LENGTH_WIDTH>> out;

    config(input_r: chan<Input<INST_DATA_WIDTH, INST_LENGTH_WIDTH>> in,
           ctrl_r: chan<Ctrl<INST_LENGTH_WIDTH>> in,
           output_s: chan<Output<INST_DATA_WIDTH, INST_LENGTH_WIDTH>> out) {

        spawn ShiftBuffer<INST_DATA_WIDTH, INST_LENGTH_WIDTH>(ctrl_r, input_r, output_s);

        (input_r, ctrl_r, output_s)
    }

    init {  }

    next(state: ()) {}
}

proc ShiftBufferAlignerInst {
    type Input = ShiftBufferPacket<INST_DATA_WIDTH, INST_LENGTH_WIDTH>;
    type Inter = ShiftBufferPacket<INST_DATA_WIDTH_X2, INST_LENGTH_WIDTH>;

    config(input: chan<Input> in, inter: chan<Inter> out) {
        spawn ShiftBufferAligner<INST_DATA_WIDTH, INST_LENGTH_WIDTH>(input, inter);
    }

    init {  }

    next(state: ()) {  }
}

proc ShiftBufferStorageInst {
    type Ctrl = ShiftBufferCtrl<INST_LENGTH_WIDTH>;
    type Inter = ShiftBufferPacket<INST_DATA_WIDTH_X2, INST_LENGTH_WIDTH>;
    type Output = ShiftBufferOutput<INST_DATA_WIDTH, INST_LENGTH_WIDTH>;

    config(ctrl: chan<Ctrl> in, inter: chan<Inter> in, output: chan<Output> out) {
        spawn ShiftBufferStorage<INST_DATA_WIDTH, INST_LENGTH_WIDTH>(ctrl, inter, output);
    }

    init {  }

    next(state: ()) {  }
}

const TEST_DATA_WIDTH = u32:64;
const TEST_LENGTH_WIDTH = std::clog2(TEST_DATA_WIDTH) + u32:1; // TODO: other places in the code use length_width(TEST_DATA_WIDTH) which is clog2(TEST_DATA_WIDTH + 1) instead, why clog2(TEST_DATA_WIDTH) + 1 here?

#[test_proc]
proc ShiftBufferTest {
    type Input = ShiftBufferPacket;
    type Ctrl = ShiftBufferCtrl;
    type Output = ShiftBufferOutput;

    terminator: chan<bool> out;
    input_s: chan<Input<TEST_DATA_WIDTH, TEST_LENGTH_WIDTH>> out;
    ctrl_s: chan<Ctrl<TEST_LENGTH_WIDTH>> out;
    data_r: chan<Output<TEST_DATA_WIDTH, TEST_LENGTH_WIDTH>> in;

    config(terminator: chan<bool> out) {
        let (input_s, input_r) = chan<Input<TEST_DATA_WIDTH, TEST_LENGTH_WIDTH>, u32:1>("input");
        let (ctrl_s, ctrl_r) = chan<Ctrl<TEST_LENGTH_WIDTH>, u32:1>("ctrl");
        let (data_s, data_r) = chan<Output<TEST_DATA_WIDTH, TEST_LENGTH_WIDTH>, u32:1>("data");

        spawn ShiftBuffer<TEST_DATA_WIDTH, TEST_LENGTH_WIDTH>(ctrl_r, input_r, data_s);

        (terminator, input_s, ctrl_s, data_r)
    }

    init {  }

    next(state: ()) {
        type Data = bits[TEST_DATA_WIDTH];
        type Length = bits[TEST_LENGTH_WIDTH];
        type Input = ShiftBufferPacket;
        type Output = ShiftBufferOutput<TEST_DATA_WIDTH, TEST_LENGTH_WIDTH>;
        type OutputPayload = ShiftBufferPacket;
        type OutputStatus = ShiftBufferStatus;
        type Ctrl = ShiftBufferCtrl;

        let tok = send(join(), input_s, Input { data: Data:0xDD_44, length: Length:16 });
        let tok = send(tok, input_s, Input { data: Data:0xAA_11_BB_22_CC_33, length: Length:48 });
        let tok = send(tok, input_s, Input { data: Data:0xEE_55_FF_66_00_77_11_88, length: Length:64 });

        // Single input, single output packet 32bit buffering
        let tok = send(join(), input_s, Input { data: Data: 0xAABB_CCDD, length: Length: 32});

        // Multiple input packets, single output 32bit buffering
        let tok = send(tok, input_s, Input { data: Data: 0x3344, length: Length: 16});
        let tok = send(tok, input_s, Input { data: Data: 0x22, length: Length: 8});
        let tok = send(tok, input_s, Input { data: Data: 0x11, length: Length: 8});

        // Small consecutive single input, single output 8bit buffering
        let tok = send(tok, input_s, Input { data: Data: 0x55, length: Length: 8});
        let tok = send(tok, input_s, Input { data: Data: 0x66, length: Length: 8});

        // Multiple input packets, single output 64bit buffering
        let tok = send(tok, input_s, Input { data: Data: 0xDDEE, length: Length: 16});
        let tok = send(tok, input_s, Input { data: Data: 0xBBCC, length: Length: 16});
        let tok = send(tok, input_s, Input { data: Data: 0x99AA, length: Length: 16});
        let tok = send(tok, input_s, Input { data: Data: 0x7788, length: Length: 16});

        // Single input packet, single output 64bit buffering
        let tok = send(tok, input_s, Input { data: Data: 0x1122_3344_5566_7788, length: Length: 64});

        // Single 64bit input packet, multiple output packets of different sizes
        let tok = send(tok, input_s, Input { data: Data: 0xEEFF_0011_CCDD_BBAA, length: Length: 64});

        // Account for leftover 0xEEFF from the previous packet
        let tok = send(tok, input_s, Input { data: Data: 0x1122, length: Length: 16});
        // Should operate on flushed buffer
        let tok = send(tok, input_s, Input { data: Data: 0x3344, length: Length: 16});

        // Input packets additionally span across 2 shift buffer aligner shift domains
        let tok = send(tok, input_s, Input { data: Data: 0x7788, length: Length: 16});
        let tok = send(tok, input_s, Input { data: Data: 0x5566, length: Length: 16});

        let tok = send(tok, ctrl_s, Ctrl { length: Length:8 });
        let (tok, output) = recv(tok, data_r);
        assert_eq(output, Output { data: Data:0x44, length: Length:8 });

        let tok = send(tok, ctrl_s, Ctrl { length: Length:4 });
        let (tok, output) = recv(tok, data_r);
        assert_eq(output, Output { data: Data:0xD, length: Length:4 });
        let tok = send(tok, ctrl_s, Ctrl { length: Length:4 });
        let (tok, output) = recv(tok, data_r);
        assert_eq(output, Output { data: Data:0xD, length: Length:4 });

        let tok = send(tok, ctrl_s, Ctrl { length: Length:48 });
        let (tok, output) = recv(tok, data_r);
        assert_eq(output, Output { data: Data:0xAA_11_BB_22_CC_33, length: Length:48 });

        let tok = send(tok, ctrl_s, Ctrl { length: Length:64 });
        let (tok, output) = recv(tok, data_r);
        assert_eq(output, Output { data: Data:0xEE_55_FF_66_00_77_11_88, length: Length:64 });

        // Single input, single output packet 32bit buffering
        let tok = send(tok, ctrl_s, Ctrl { length: Length:32});
        let (tok, data) = recv(tok, data_r);
        assert_eq(data, Output { data: Data: 0xAABB_CCDD, length: Length: 32});

        // Multiple input packets, single output 32bit buffering
        let tok = send(tok, ctrl_s, Ctrl { length: Length:32});
        let (tok, data) = recv(tok, data_r);
        assert_eq(data, Output { data: Data: 0x1122_3344, length: Length: 32});

        // Small consecutive single input, single output 8bit buffering
        let tok = send(tok, ctrl_s, Ctrl { length: Length:8});
        let (tok, data) = recv(tok, data_r);
        assert_eq(data, Output { data: Data: 0x55, length: Length: 8});
        let tok = send(tok, ctrl_s, Ctrl { length: Length:8});
        let (tok, data) = recv(tok, data_r);
        assert_eq(data, Output { data: Data: 0x66, length: Length: 8});

        // Multiple input packets, single output 64bit buffering
        let tok = send(tok, ctrl_s, Ctrl { length: Length:64});
        let (tok, data) = recv(tok, data_r);
        assert_eq(data, Output { data: Data: 0x7788_99AA_BBCC_DDEE, length: Length: 64});

        // Single input packet, single output 64bit buffering
        let tok = send(tok, ctrl_s, Ctrl { length: Length:64});
        let (tok, data) = recv(tok, data_r);
        assert_eq(data, Output { data: Data: 0x1122_3344_5566_7788, length: Length: 64});

        // Single 64bit input packet, multiple output packets of different sizes
        let tok = send(tok, ctrl_s, Ctrl { length: Length:8});
        let (tok, data) = recv(tok, data_r);
        assert_eq(data, Output { data: Data: 0xAA, length: Length: 8});
        let tok = send(tok, ctrl_s, Ctrl { length: Length:8});
        let (tok, data) = recv(tok, data_r);
        assert_eq(data, Output { data: Data: 0xBB, length: Length: 8});
        let tok = send(tok, ctrl_s, Ctrl { length: Length:16});
        let (tok, data) = recv(tok, data_r);
        assert_eq(data, Output { data: Data: 0xCCDD, length: Length: 16});
        let tok = send(tok, ctrl_s, Ctrl { length: Length:32});
        let (tok, data) = recv(tok, data_r);
        assert_eq(data, Output { data: Data: 0xEEFF_0011, length: Length: 32});

        let tok = send(tok, ctrl_s, Ctrl { length: Length:16});
        let (tok, data) = recv(tok, data_r);
        assert_eq(data, Output { data: Data: 0x1122, length: Length: 16});
        let tok = send(tok, ctrl_s, Ctrl { length: Length:16});
        let (tok, data) = recv(tok, data_r);
        assert_eq(data, Output { data: Data: 0x3344, length: Length: 16});

        let tok = send(tok, ctrl_s, Ctrl { length: Length:32});
        let (tok, data) = recv(tok, data_r);
        assert_eq(data, Output { data: Data: 0x5566_7788, length: Length: 32});

        // Test attempting to read more data than available in the buffer
        // This should wait indefinitely, we test this by checking that we can't
        // receive data over the next consecutive 100 iterations
        let tok = send(tok, ctrl_s, Ctrl { length: Length:64});
        let tok = for (_, tok): (u32, token) in u32:1..u32:100 {
            let (tok, _, data_valid) = recv_non_blocking(tok, data_r, zero!<Output>());
            assert_eq(data_valid, false);
            tok
        }(tok);

        // Refill the buffer with more data - not enough to reply to the earlier request for 64b
        let tok = send(tok, input_s, Input { data: Data: 0xDEAD_BEEF, length: Length: 32});
        // Check that we can't receive still
        let tok = for (_, tok): (u32, token) in u32:1..u32:100 {
            let (tok, _, data_valid) = recv_non_blocking(tok, data_r, zero!<Output>());
            assert_eq(data_valid, false);
            tok
        }(tok);

        // Refill buffer with enough data
        let tok = send(tok, input_s, Input { data: Data: 0xF00B_A4BA, length: Length: 32});
        // Now we should be able to receive a response for 64b request
        let (tok, data) = recv(tok, data_r);
        assert_eq(data, Output { data: Data: 0xF00BA4BA_DEADBEEF, length: Length: 64});

        send(tok, terminator, true);
    }
}
