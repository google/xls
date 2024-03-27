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

// This file contains a proc responsible for decoding probability frequencies
// to probability distribution, as described in:
// https://datatracker.ietf.org/doc/html/rfc8878#section-4.1.1

import std;
import xls.examples.ram;
import xls.modules.zstd.common;
import xls.modules.zstd.shift_buffer;

pub struct Remainder { value: u1, valid: bool }

pub struct FSEProbaFreqDecoderCtrl {
    remainder: Remainder,
    finished: bool,
}

enum Status : u4 {
    SEND_ACCURACY_LOG_REQ = 0,
    RECV_ACCURACY_LOG = 1,
    SEND_SYMBOL_REQ = 2,
    RECV_SYMBOL = 3,
    RECV_ZERO_PROBA = 4,
    WRITE_ZERO_PROBA = 5,
    WAIT_FOR_COMPLETION = 6,
    INVALID = 15,
}

struct State {
    status: Status,
    // accuracy log used in the FSE decoding table
    accuracy_log: u32,
    // remaining bit that can be a leftover from parsing small probability frequencies
    remainder: Remainder,
    // indicates if one more packet with zero probabilities is expected
    next_recv_zero: bool,
    // information about remaining probability points
    remaining_proba: u32,
    // number of received probability symbols
    symbol_count: u32,
    // number of probability symbols written to RAM
    written_symbol_count: u32,
    // number of processed zero probabilities
    zero_proba_count: u32,
}

type DecoderCtrl = FSEProbaFreqDecoderCtrl;
type SequenceData = common::SequenceData;

// Adapter for input data, converting the data to a shift buffer input type
pub proc FSEInputBuffer<DATA_WIDTH: u32, LENGTH_WIDTH: u32> {
    type BufferInput = shift_buffer::ShiftBufferInput<DATA_WIDTH, LENGTH_WIDTH>;
    type BufferCtrl = shift_buffer::ShiftBufferCtrl<LENGTH_WIDTH>;
    type BufferOutput = shift_buffer::ShiftBufferOutput<DATA_WIDTH, LENGTH_WIDTH>;

    in_data_r: chan<SequenceData> in;
    buff_in_data_s: chan<BufferInput> out;

    config(in_data_r: chan<SequenceData> in,
           in_ctrl_r: chan<BufferCtrl> in,
           out_data_s: chan<BufferOutput> out,
           out_ctrl_s: chan<BufferOutput> out) {

        let (buff_in_data_s, buff_in_data_r) = chan<BufferInput, u32:1>;

        spawn shift_buffer::ShiftBuffer<DATA_WIDTH, LENGTH_WIDTH>
            (buff_in_data_r, in_ctrl_r, out_data_s, out_ctrl_s);

        (in_data_r, buff_in_data_s)
    }

    init {  }

    next(tok: token, state: ()) {
        type Data = common::BlockData;
        type Length = bits[LENGTH_WIDTH];

        let (tok, recv_data, recv_valid) =
            recv_non_blocking(tok, in_data_r, zero!<SequenceData>());

        let shift_buffer_data = BufferInput {
            data: recv_data.bytes as Data, length: recv_data.length as Length, last: recv_data.last
        };

        let tok = send_if(tok, buff_in_data_s, recv_valid, shift_buffer_data);

        if recv_valid {
            trace_fmt!("[IO] Sent the following packet to ShiftBuffer: {:#x}", shift_buffer_data);
        } else {};

    }
}

// calculates bit_with of the next probability frequency based on the remaining probability points
fn get_bit_width(remaining_proba: u32) -> u32 { std::flog2(remaining_proba + u32:1) + u32:1 }

// calculates mask for small probability frequency values
fn get_lower_mask(bit_width: u32) -> u32 { (u32:1 << (bit_width - u32:1)) - u32:1 }

// calculates threshold for a duplicated "upper" range of small probability frequencies
fn get_threshold(bit_width: u32, remaining_proba: u32) -> u32 {
    (u32:1 << bit_width) - u32:1 - (remaining_proba + u32:1)
}

// get the adjusted stream value for calculating probability points
fn get_adjusted_value(data: u32, remainder: Remainder) -> u32 {
    if remainder.valid { (data << u32:1) | (remainder.value as u32) } else { data }
}

// proc for filling probability frequencies table
pub proc FSEProbaFreqDecoder<
    RAM_DATA_WIDTH: u32,
    RAM_SIZE: u32,
    RAM_WORD_PARTITION_SIZE: u32,
    RAM_ADDR_WIDTH: u32 = {std::clog2(RAM_SIZE)},
    RAM_NUM_PARTITIONS: u32 = {ram::num_partitions(RAM_WORD_PARTITION_SIZE, RAM_DATA_WIDTH)},
    DATA_WIDTH: u32 = {common::DATA_WIDTH},
    LENGTH_WIDTH: u32 = {common::BLOCK_PACKET_WIDTH}
> {
    type Length = bits[LENGTH_WIDTH];
    type BufferCtrl = shift_buffer::ShiftBufferCtrl<LENGTH_WIDTH>;
    type BufferOutput = shift_buffer::ShiftBufferOutput<DATA_WIDTH, LENGTH_WIDTH>;
    type BufferOutputType = shift_buffer::ShiftBufferOutputType;
    type RamWriteReq = ram::WriteReq<RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>;
    type RamWriteResp = ram::WriteResp;
    type RamReadReq = ram::ReadReq<RAM_ADDR_WIDTH, RAM_NUM_PARTITIONS>;
    type RamReadResp = ram::ReadResp<RAM_DATA_WIDTH>;
    type RamAddr = bits[RAM_ADDR_WIDTH];
    type RamData = bits[RAM_DATA_WIDTH];

    fse_table_ctrl_s: chan<DecoderCtrl> out;
    buff_in_ctrl_s: chan<BufferCtrl> out;
    buff_out_data_r: chan<BufferOutput> in;
    buff_out_ctrl_r: chan<BufferOutput> in;
    rd_req_s: chan<RamReadReq> out;
    rd_resp_r: chan<RamReadResp> in;
    wr_req_s: chan<RamWriteReq> out;
    wr_resp_r: chan<RamWriteResp> in;

    config(
        fse_table_ctrl_s: chan<DecoderCtrl> out,
        buff_in_ctrl_s: chan<BufferCtrl> out,
        buff_out_data_r: chan<BufferOutput> in,
        buff_out_ctrl_r: chan<BufferOutput> in,
        rd_req_s: chan<RamReadReq> out,
        rd_resp_r: chan<RamReadResp> in,
        wr_req_s: chan<RamWriteReq> out,
        wr_resp_r: chan<RamWriteResp> in) {
        (
            fse_table_ctrl_s,
            buff_in_ctrl_s, buff_out_data_r, buff_out_ctrl_r,
            rd_req_s, rd_resp_r, wr_req_s, wr_resp_r,
        )
    }

    init { zero!<State>() }

    next(tok: token, state: State) {
        type RamWriteResp = ram::WriteResp;
        type RamWriteReq = ram::WriteReq<RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>;
        type RamReadReq = ram::ReadReq<RAM_ADDR_WIDTH, RAM_NUM_PARTITIONS>;
        type RamReadResp = ram::ReadResp<RAM_DATA_WIDTH>;

        let (do_buff_ctrl_recv, do_buff_data_recv) = match (state.status) {
            Status::RECV_ACCURACY_LOG => (true, false),
            Status::RECV_SYMBOL => (false, true),
            Status::RECV_ZERO_PROBA => (false, true),
            _ => (false, false),
        };

        let (tok, out_ctrl) = recv_if(tok, buff_out_ctrl_r, do_buff_ctrl_recv, zero!<BufferOutput>());
        if do_buff_ctrl_recv {
            trace_fmt!("Received from the ctrl output of ShiftBuffer");
        } else {};

        let (tok, out_data) = recv_if(tok, buff_out_data_r, do_buff_data_recv, zero!<BufferOutput>());
        if do_buff_data_recv {
            trace_fmt!("Received from the data output of ShiftBuffer");
        } else {};

        let (tok, wr_resp, wr_resp_valid) = recv_non_blocking(tok, wr_resp_r, zero!<RamWriteResp>());
        let written_symbol_count = if wr_resp_valid {
            state.written_symbol_count + u32:1
        } else {
            state.written_symbol_count
        };

        let (buffer_ctrl_option, ram_option, new_state, finish_ctrl) = match state.status {
            Status::SEND_ACCURACY_LOG_REQ => {
                trace_fmt!("[STATE]: SEND_ACCURACY_LOG_REQ");
                (
                    (true, BufferCtrl { length: Length:4, output: BufferOutputType::CTRL }),
                    (false, zero!<RamWriteReq>()),
                    State { status: Status::RECV_ACCURACY_LOG, written_symbol_count, ..state },
                    zero!<DecoderCtrl>(),
                )
            },
            Status::RECV_ACCURACY_LOG => {
                trace_fmt!("[STATE]: RECV_ACCURACY_LOG_CTRL");
                let accuracy_log = u32:5 + out_ctrl.data as u32;

                trace_fmt!("[FSE]: Accuracy log is {}", accuracy_log);
                (
                    (false, zero!<BufferCtrl>()),
                    (false, zero!<RamWriteReq>()),
                    State {
                        status: Status::SEND_SYMBOL_REQ,
                        accuracy_log,
                        remaining_proba: u32:1 << accuracy_log,
                        written_symbol_count,
                        ..state
                    },
                    zero!<DecoderCtrl>(),
                )
            },
            Status::SEND_SYMBOL_REQ => {
                trace_fmt!("[STATE] SEND_SYMBOL_REQ");
                let bit_width = get_bit_width(state.remaining_proba);

                (
                    (true, BufferCtrl { length: bit_width as Length, output: BufferOutputType::DATA }),
                    (false, zero!<RamWriteReq>()),
                    State { status: Status::RECV_SYMBOL, written_symbol_count, ..state },
                    zero!<DecoderCtrl>(),
                )
            },
            Status::RECV_SYMBOL => {
                trace_fmt!("[STATE]: RECV_SYMBOL");
                let bit_width = get_bit_width(state.remaining_proba);
                let lower_mask = get_lower_mask(bit_width);
                let threshold = get_threshold(bit_width, state.remaining_proba);

                let mask = out_data.length as u32 - u32:1;
                let data = out_data.data as u32;

                trace_fmt!("[FSE: {}]: value before adjustments: {} ({:#x})", state.symbol_count, data, data);
                let value = get_adjusted_value(data, state.remainder);

                trace_fmt!("[FSE: {}]: bit_width: {:#x}", state.symbol_count, bit_width);
                trace_fmt!("[FSE: {}]: lower_mask: {} ({:#x})", state.symbol_count, lower_mask, lower_mask);
                trace_fmt!("[FSE: {}]: threshold: {} ({:#x})", state.symbol_count, threshold, threshold);
                trace_fmt!("[FSE: {}]: value with remainder: {} ({:#x})", state.symbol_count, value, value);

                let (remainder, value) = if (value & lower_mask) < threshold {
                    trace_fmt!("[FSE: {}]: Small number", state.symbol_count);
                    (
                        Remainder { value: value[bit_width - u32:1+:u1], valid: true },
                        value & lower_mask,
                    )
                } else if value > lower_mask {
                    trace_fmt!("[FSE: {}]: Adjusted number", state.symbol_count);
                    (zero!<Remainder>(), value - threshold)
                } else {
                    trace_fmt!("[FSE: {}]: Normal number", state.symbol_count);
                    (zero!<Remainder>(), value)
                };

                trace_fmt!("[FSE: {}]: remainder: {}", state.symbol_count, remainder);
                trace_fmt!("[FSE: {}]: value: {}", state.symbol_count, value);

                let proba = value as s32 - s32:1;
                let proba_points = if proba < s32:0 { u32:1 } else { proba as u32 };
                assert!(proba_points <= state.remaining_proba, "corrupted_data");

                let remaining_proba = state.remaining_proba - proba_points;
                trace_fmt!("[FSE: {}]: proba: {}", state.symbol_count, proba);
                trace_fmt!("[FSE: {}]: remaining_proba: {} ({:#x})",
                state.symbol_count, remaining_proba, remaining_proba);

                let symbol_count = state.symbol_count + u32:1;
                let remainder_count = if remainder.valid { u32:1 } else { u32:0 };

                if remaining_proba == u32:0 {
                    (
                        (false, zero!<BufferCtrl>()),
                        (true, RamWriteReq {
                            addr: state.symbol_count as RamAddr,
                            data: proba as RamData,
                            mask: std::unsigned_max_value<RAM_NUM_PARTITIONS>()
                        }),
                        State {
                            status: Status::WAIT_FOR_COMPLETION,
                            written_symbol_count,
                            symbol_count,
                            remaining_proba,
                            remainder,
                        ..state
                        },
                        zero!<DecoderCtrl>(),
                    )
                } else if remaining_proba > u32:0 && proba != s32:0 {
                    let next_bit_width = get_bit_width(remaining_proba) - remainder_count;
                    (
                        (true, BufferCtrl { length: next_bit_width as Length, output: BufferOutputType::DATA }),
                        (true, RamWriteReq {
                            addr: state.symbol_count as RamAddr,
                            data: proba as RamData,
                            mask: std::unsigned_max_value<RAM_NUM_PARTITIONS>()
                        }),
                        State {
                            status: Status::RECV_SYMBOL,
                            written_symbol_count,
                            symbol_count,
                            remaining_proba,
                            remainder,
                        ..state
                        },
                        zero!<DecoderCtrl>(),
                    )
                } else if remaining_proba > u32:0 && proba == s32:0 {
                    let next_bit_width = u32:2 - remainder_count;
                    (
                        (true, BufferCtrl { length: next_bit_width, output: BufferOutputType::DATA }),
                        (true, RamWriteReq {
                            addr: state.symbol_count as RamAddr,
                            data: proba as RamData,
                            mask: std::unsigned_max_value<RAM_NUM_PARTITIONS>()
                        }),
                        State {
                            status: Status::RECV_ZERO_PROBA,
                            written_symbol_count,
                            symbol_count,
                            remaining_proba,
                            remainder,
                        ..state
                        },
                        zero!<DecoderCtrl>(),
                    )
                } else {
                    fail!(
                        "unhandled_case",
                        (
                            (false, zero!<BufferCtrl>()),
                            (false, zero!<RamWriteReq>()),
                            State { status: Status::INVALID, ..zero!<State>() },
                            zero!<DecoderCtrl>(),
                        ))
                }
            },
            Status::RECV_ZERO_PROBA => {
                trace_fmt!("[STATE]: RECV_ZERO_PROBA");

                let zero_proba_count = out_data.data as u32;
                let zero_proba_length = out_data.length as u32;

                trace_fmt!("[FSE]: {:#x}", out_data);
                let zero_proba_count = get_adjusted_value(zero_proba_count, state.remainder);

                trace_fmt!("[FSE]: zero_proba_count: {}, length: {}",
                zero_proba_count, zero_proba_length);
                if zero_proba_count == u32:0 {
                    let new_status = if state.remaining_proba > u32:0 {
                        Status::SEND_SYMBOL_REQ
                    } else if state.remaining_proba == u32:0 {
                        Status::WAIT_FOR_COMPLETION
                    } else {
                        Status::INVALID
                    };

                    (
                        (true, zero!<BufferCtrl>()),
                        (false, zero!<RamWriteReq>()),
                        State {
                            status: new_status, remainder: zero!<Remainder>(), written_symbol_count, ..state},
                        zero!<DecoderCtrl>(),
                    )
                } else {
                    let next_recv_zero = zero_proba_count == u32:3;
                    (
                        (false, zero!<BufferCtrl>()),
                        (false, zero!<RamWriteReq>()),
                        State {
                            status: Status::WRITE_ZERO_PROBA,
                            remainder: zero!<Remainder>(),
                            written_symbol_count,
                            zero_proba_count,
                            next_recv_zero,
                        ..state
                        },
                        zero!<DecoderCtrl>(),
                    )
                }
            },
            Status::WRITE_ZERO_PROBA => {
                trace_fmt!("[STATUS]: WRITE_ZERO_PROBA");
                let zero_proba_count = state.zero_proba_count - u32:1;
                let symbol_count = state.symbol_count + u32:1;

                let write_req = RamWriteReq {
                    addr: state.symbol_count as RamAddr,
                    data: RamData:0,
                    mask: std::unsigned_max_value<RAM_NUM_PARTITIONS>()
                };

                if zero_proba_count == u32:0 && state.next_recv_zero == true {
                    (
                        (true, BufferCtrl { length: Length:2, output: BufferOutputType::DATA }),
                        (true, write_req),
                        State {
                            status: Status::RECV_ZERO_PROBA,
                            next_recv_zero: false,
                            written_symbol_count,
                            zero_proba_count,
                            symbol_count,
                        ..state
                        },
                        zero!<DecoderCtrl>(),
                    )
                } else if zero_proba_count == u32:0 && state.next_recv_zero == false {
                    (
                        (false, zero!<BufferCtrl>()),
                        (true, write_req),
                        State {
                            status: Status::SEND_SYMBOL_REQ,
                            next_recv_zero: false,
                            zero_proba_count: u32:0,
                            written_symbol_count,
                            symbol_count,
                        ..state
                        },
                        zero!<DecoderCtrl>(),
                    )
                } else {
                    (
                        (false, zero!<BufferCtrl>()),
                        (true, write_req),
                        State {
                            status: Status::WRITE_ZERO_PROBA,
                            zero_proba_count,
                            symbol_count,
                            written_symbol_count,
                        ..state
                        },
                        zero!<DecoderCtrl>(),
                    )
                }
            },
            Status::WAIT_FOR_COMPLETION => {
                trace_fmt!("[STATE]: WAIT_FOR_COMPLETION");
                if written_symbol_count == state.symbol_count {
                    (
                        (false, zero!<BufferCtrl>()),
                        (false, zero!<RamWriteReq>()),
                        zero!<State>(),
                        DecoderCtrl {
                            remainder: state.remainder,
                            finished: true
                        },
                    )
                } else {
                    (
                        (false, zero!<BufferCtrl>()),
                        (false, zero!<RamWriteReq>()),
                        state,
                        zero!<DecoderCtrl>(),
                    )
                }
            },
            _ => {
                trace_fmt!("Invalid state");
                fail!(
                    "not_handled",
                    (
                        (false, zero!<BufferCtrl>()),
                        (false, zero!<RamWriteReq>()),
                        state,
                        zero!<DecoderCtrl>(),
                    ))
            },
        };

        let (do_send_ctrl, ctrl_data) = buffer_ctrl_option;
        let tok = send_if(tok, buff_in_ctrl_s, do_send_ctrl, ctrl_data);
        if do_send_ctrl {
            trace_fmt!("[IO]: Send the following control to ShiftBuffer: {:#x}", ctrl_data);
        } else {};

        let (do_send_ram, ram_data) = ram_option;
        let tok = send_if(tok, wr_req_s, do_send_ram, ram_data);
        if do_send_ram {
            trace_fmt!("[IO]: Send the following request to ram: {:#x}", ram_data);
        } else {};

        let (do_send_finish, finish_data) = (finish_ctrl.finished, finish_ctrl);
        let tok = send_if(tok, fse_table_ctrl_s, do_send_finish, finish_data);
        if do_send_finish {
            trace_fmt!("[IO]: Send the information about finished parsing: {:#x}", finish_data);
        } else {};

        // unused channels
        send_if(tok, rd_req_s, false, zero!<RamReadReq>());
        recv_if(tok, rd_resp_r, false, zero!<RamReadResp>());

        new_state
    }
}


const INST_RAM_SIZE = common::FSE_MAX_SYMBOLS;
const INST_RAM_ADDR_WIDTH = std::clog2(INST_RAM_SIZE);
const INST_RAM_DATA_WIDTH = get_bit_width(u32:1 << common::FSE_MAX_ACCURACY_LOG);
const INST_RAM_WORD_PARTITION_SIZE = INST_RAM_DATA_WIDTH;
const INST_RAM_NUM_PARTITIONS = ram::num_partitions(INST_RAM_WORD_PARTITION_SIZE, INST_RAM_DATA_WIDTH);
const INST_DATA_WIDTH = common::DATA_WIDTH;
const INST_LENGTH_WIDTH = common::BLOCK_PACKET_WIDTH;

proc FSEProbaFreqDecoderInst {
    rd_req_s: chan<ram::ReadReq<INST_RAM_ADDR_WIDTH, INST_RAM_NUM_PARTITIONS>> out;
    rd_resp_r: chan<ram::ReadResp<INST_RAM_DATA_WIDTH>> in;

    init {}

    config(
        fse_table_ctrl_s: chan<DecoderCtrl> out,
        buff_in_ctrl_s: chan<shift_buffer::ShiftBufferCtrl<INST_LENGTH_WIDTH>> out,
        buff_out_data_r: chan<shift_buffer::ShiftBufferOutput<INST_DATA_WIDTH, INST_LENGTH_WIDTH>> in,
        buff_out_ctrl_r: chan<shift_buffer::ShiftBufferOutput<INST_DATA_WIDTH, INST_LENGTH_WIDTH>> in,
        rd_req_s: chan<ram::ReadReq<INST_RAM_ADDR_WIDTH, INST_RAM_NUM_PARTITIONS>> out,
        rd_resp_r: chan<ram::ReadResp<INST_RAM_DATA_WIDTH>> in,
        wr_req_s: chan<ram::WriteReq<INST_RAM_ADDR_WIDTH, INST_RAM_DATA_WIDTH, INST_RAM_NUM_PARTITIONS>> out,
        wr_resp_r: chan<ram::WriteResp> in
    ) {
        spawn FSEProbaFreqDecoder<INST_RAM_DATA_WIDTH, INST_RAM_SIZE, INST_RAM_WORD_PARTITION_SIZE>(
            fse_table_ctrl_s,
            buff_in_ctrl_s, buff_out_data_r, buff_out_ctrl_r,
            rd_req_s, rd_resp_r, wr_req_s, wr_resp_r);

        (rd_req_s, rd_resp_r)
    }

    next(tok: token, state: ()) { }
}


const TEST_RAM_DATA_WIDTH = u32:8;
const TEST_RAM_SIZE = u32:100;
const TEST_RAM_ADDR_WIDTH = std::clog2(TEST_RAM_SIZE);
const TEST_RAM_WORD_PARTITION_SIZE = TEST_RAM_DATA_WIDTH;
const TEST_RAM_NUM_PARTITIONS = ram::num_partitions(TEST_RAM_WORD_PARTITION_SIZE, TEST_RAM_DATA_WIDTH);
const TEST_DATA_WIDTH = common::DATA_WIDTH;
const TEST_LENGTH_WIDTH = common::BLOCK_PACKET_WIDTH;

#[test_proc]
proc FSEProbaFreqDecoderTest {
    type ReadReq = ram::ReadReq<TEST_RAM_ADDR_WIDTH, TEST_RAM_NUM_PARTITIONS>;
    type ReadResp = ram::ReadResp<TEST_RAM_DATA_WIDTH>;
    type WriteReq = ram::WriteReq<TEST_RAM_ADDR_WIDTH, TEST_RAM_DATA_WIDTH, TEST_RAM_NUM_PARTITIONS>;
    type WriteResp = ram::WriteResp;
    type BufferCtrl = shift_buffer::ShiftBufferCtrl<TEST_LENGTH_WIDTH>;
    type BufferOutput = shift_buffer::ShiftBufferOutput<TEST_DATA_WIDTH, TEST_LENGTH_WIDTH>;
    type BufferOutputType = shift_buffer::ShiftBufferOutputType;

    terminator: chan<bool> out;
    seq_data_s: chan<SequenceData> out;
    fse_table_ctrl_r: chan<DecoderCtrl> in;
    rd_req_s: chan<ReadReq> out;
    rd_resp_r: chan<ReadResp> in;
    wr_req_s: chan<WriteReq> out;
    wr_resp_r: chan<WriteResp> in;

    config(terminator: chan<bool> out) {
        // RAM channels
        let (rd_req_s, rd_req_r) = chan<ReadReq>;
        let (rd_resp_s, rd_resp_r) = chan<ReadResp>;
        let (wr_req_s, wr_req_r) = chan<WriteReq>;
        let (wr_resp_s, wr_resp_r) = chan<WriteResp>;

        // FSEProbaFreqDecoder channels
        let (seq_data_s, seq_data_r) = chan<SequenceData>;
        let (fse_table_ctrl_s, fse_table_ctrl_r) = chan<DecoderCtrl>;

        let (buff_in_ctrl_s, buff_in_ctrl_r) = chan<BufferCtrl>;
        let (buff_out_data_s, buff_out_data_r) = chan<BufferOutput>;
        let (buff_out_ctrl_s, buff_out_ctrl_r) = chan<BufferOutput>;

        spawn FSEInputBuffer<TEST_DATA_WIDTH, TEST_LENGTH_WIDTH>(
            seq_data_r, buff_in_ctrl_r, buff_out_data_s, buff_out_ctrl_s);

        spawn FSEProbaFreqDecoder<TEST_RAM_DATA_WIDTH, TEST_RAM_SIZE, TEST_RAM_WORD_PARTITION_SIZE>(
            fse_table_ctrl_s,
            buff_in_ctrl_s, buff_out_data_r, buff_out_ctrl_r,
            rd_req_s, rd_resp_r, wr_req_s, wr_resp_r);

        spawn ram::RamModel<TEST_RAM_DATA_WIDTH, TEST_RAM_SIZE, TEST_RAM_WORD_PARTITION_SIZE>(
            rd_req_r, rd_resp_s, wr_req_r, wr_resp_s);

        (
            terminator,
            seq_data_s, fse_table_ctrl_r,
            rd_req_s, rd_resp_r, wr_req_s, wr_resp_r,
        )
    }

    init {  }

    next(tok: token, state: ()) {

        trace_fmt!("TEST CASE 1");
        // * accuracy_log = 8
        // * probability frequencies:
        // | value | probability | bits (real) | symbol number |
        // | ----- | ----------- | ----------  | ------------- |
        // | 97    | 96          | 8(9)        | 0             |
        // | 117   | 116         | 8           | 1             |
        // | 55    | 36          | 6           | 2             |
        // | 1     | 0           | 4(3)        | 3             |
        // | 2*    | 0 0 0       | (2)         | 4 5 6         |
        // | 1*    | 0           | (2)         | 7             |
        // | 3     | 2           | 4(3)        | 8             |
        // | 1     | 0           | 3           | 9             |
        // | 0     | -1          | 3           | 10            |
        // | 7     | 6           | 3           | 11            |

        let tok = send(
            tok, seq_data_s,
            common::SequenceData {
                bytes: u64:0b111_000_00_001_011_01_11_001_110111_01110101_01100001_0011,
                length: u32:47,
                last: false
            });
        let (tok, _) = recv(tok, fse_table_ctrl_r);

        trace_fmt!("TEST CASE 2");
        // * accuracy_log = 9
        // * probability frequencies:
        // | value | probability | bits (real) | symbol number |
        // | ----- | ----------- | ----------  | ------------- |
        // | 1022  | 512         | 10          | 0             |
        // | 0     | -1          | 2(1)        | 1             |

        let tok = send(tok, seq_data_s, common::SequenceData {
            bytes: u64:0b00_1111111110_0100,
            length: u32:16,
            last: false
        });
        let (tok, _) = recv(tok, fse_table_ctrl_r);

        trace_fmt!("TEST CASE 3");
        // * accuracy_log = 9
        // * probability frequencies:
        // | value | probability | bits (real) | symbol number |
        // | ----- | ----------- | ----------  | ------------- |
        // | 1022  | 512         | 10          | 0             |
        // | 2     | -1          | 2(1)        | 1             |
        let tok = send(tok, seq_data_s, common::SequenceData {
            bytes: u64:0b10_1111111110_0100,
            length: u32:16,
            last: false
        });
        let (tok, _) = recv(tok, fse_table_ctrl_r);
        let tok = send(tok, terminator, true);
    }
}
