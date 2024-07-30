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
import xls.modules.zstd.ram_wr_handler as ram_wr;

type Remainder = common::FseRemainder;
type DecoderCtrl = common::FseProbaFreqDecoderCtrl;
type AccuracyLog = common::FseAccuracyLog;
type RemainingProba = common::FseRemainingProba;
type SymbolCount = common::FseSymbolCount;
type SequenceData = common::SequenceData;

const SYMBOL_COUNT_WIDTH = common::FSE_SYMBOL_COUNT_WIDTH;
const ACCURACY_LOG_WIDTH = common::FSE_ACCURACY_LOG_WIDTH;

enum Status : u3 {
    SEND_ACCURACY_LOG_REQ = 0,
    RECV_ACCURACY_LOG = 1,
    SEND_SYMBOL_REQ = 2,
    RECV_SYMBOL = 3,
    RECV_ZERO_PROBA = 4,
    WRITE_ZERO_PROBA = 5,
    WAIT_FOR_COMPLETION = 6,
    INVALID = 7,
}

struct State {
    status: Status,
    // accuracy log used in the FSE decoding table
    accuracy_log: AccuracyLog,
    // remaining bit that can be a leftover from parsing small probability frequencies
    remainder: Remainder,
    // indicates if one more packet with zero probabilities is expected
    next_recv_zero: bool,
    // information about remaining probability points
    remaining_proba: RemainingProba,
    // number of received probability symbols
    symbol_count: SymbolCount,
    // number of probability symbols written to RAM
    written_symbol_count: SymbolCount,
    // number of processed zero probability symbols
    zero_proba_count: SymbolCount,
}

// Adapter for input data, converting the data to a shift buffer input type
pub proc FseInputBuffer<DATA_WIDTH: u32, LENGTH_WIDTH: u32> {
    type BufferInput = shift_buffer::ShiftBufferInput<DATA_WIDTH, LENGTH_WIDTH>;
    type BufferCtrl = shift_buffer::ShiftBufferCtrl<LENGTH_WIDTH>;
    type BufferOutput = shift_buffer::ShiftBufferOutput<DATA_WIDTH, LENGTH_WIDTH>;
    type Data = common::BlockData;
    type Length = bits[LENGTH_WIDTH];

    in_data_r: chan<SequenceData> in;
    buff_in_data_s: chan<BufferInput> out;

    config(
        in_data_r: chan<SequenceData> in,
        in_ctrl_r: chan<BufferCtrl> in,
        out_data_s: chan<BufferOutput> out,
        out_ctrl_s: chan<BufferOutput> out
    ) {
        let (buff_in_data_s, buff_in_data_r) = chan<BufferInput, u32:1>("buff_in_data");
        spawn shift_buffer::ShiftBuffer<DATA_WIDTH, LENGTH_WIDTH>(
            buff_in_data_r, in_ctrl_r, out_data_s, out_ctrl_s);

        (in_data_r, buff_in_data_s)
    }

    init {  }

    next(state: ()) {
        let tok0 = join();
        let (tok1, recv_data, recv_valid) = recv_non_blocking(tok0, in_data_r, zero!<SequenceData>());

        let shift_buffer_data = BufferInput {
            data: recv_data.bytes as Data,
            length: recv_data.length as Length,
            last: recv_data.last
        };

        send_if(tok1, buff_in_data_s, recv_valid, shift_buffer_data);
    }
}

// calculates bit_with of the next probability frequency based on the remaining probability points
fn get_bit_width(remaining_proba: RemainingProba) -> u16 {
    let highest_set_bit = std::flog2(remaining_proba as u32 + u32:1);
    highest_set_bit as u16 + u16:1
}

// calculates mask for small probability frequency values
fn get_lower_mask(bit_width: u16) -> u16 { (u16:1 << (bit_width - u16:1)) - u16:1 }

// calculates threshold for a duplicated "upper" range of small probability frequencies
fn get_threshold(bit_width: u16, remaining_proba: u16) -> u16 {
    (u16:1 << bit_width) - u16:1 - (remaining_proba + u16:1)
}

// get the adjusted stream value for calculating probability points
fn get_adjusted_value(data: u16, remainder: Remainder) -> u16 {
    if remainder.valid { (data << u16:1) | (remainder.value as u16) } else { data }
}

// proc for filling probability frequencies table
pub proc FseProbaFreqDecoder<
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
    resp_in_s: chan<bool> out;
    resp_out_r: chan<SymbolCount> in;
    rd_req_s: chan<RamReadReq> out;
    rd_resp_r: chan<RamReadResp> in;
    wr_req_s: chan<RamWriteReq> out;

    config(
        fse_table_ctrl_s: chan<DecoderCtrl> out,
        buff_in_ctrl_s: chan<BufferCtrl> out,
        buff_out_data_r: chan<BufferOutput> in,
        buff_out_ctrl_r: chan<BufferOutput> in,
        rd_req_s: chan<RamReadReq> out,
        rd_resp_r: chan<RamReadResp> in,
        wr_req_s: chan<RamWriteReq> out,
        wr_resp_r: chan<RamWriteResp> in
    ) {
        let (resp_in_s, resp_in_r) = chan<bool, u32:1>("resp_in");
        let (resp_out_s, resp_out_r) = chan<SymbolCount, u32:1>("resp_out");

        spawn ram_wr::RamWrRespHandler<SYMBOL_COUNT_WIDTH>(resp_in_r, resp_out_s, wr_resp_r);
        (
            fse_table_ctrl_s,
            buff_in_ctrl_s, buff_out_data_r, buff_out_ctrl_r,
            resp_in_s, resp_out_r,
            rd_req_s, rd_resp_r, wr_req_s,
        )
    }

    init { zero!<State>() }

    next(state: State) {
        type BufferCtrl = shift_buffer::ShiftBufferCtrl<LENGTH_WIDTH>;
        type BufferOutput = shift_buffer::ShiftBufferOutput<DATA_WIDTH, LENGTH_WIDTH>;
        type RamWriteReq = ram::WriteReq<RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>;
        type RamWriteResp = ram::WriteResp;
        type RamReadReq = ram::ReadReq<RAM_ADDR_WIDTH, RAM_NUM_PARTITIONS>;
        type RamReadResp = ram::ReadResp<RAM_DATA_WIDTH>;

        let tok0 = join();

        let (do_buff_ctrl_recv, do_buff_data_recv) = match (state.status) {
            Status::RECV_ACCURACY_LOG => (true, false),
            Status::RECV_SYMBOL => (false, true),
            Status::RECV_ZERO_PROBA => (false, true),
            _ => (false, false),
        };

        let (tok1_0, out_ctrl) =
            recv_if(tok0, buff_out_ctrl_r, do_buff_ctrl_recv, zero!<BufferOutput>());
        let (tok1_1, out_data) =
            recv_if(tok0, buff_out_data_r, do_buff_data_recv, zero!<BufferOutput>());

        let (tok1_2, written_symbol_count, written_symb_count_valid) =
            recv_non_blocking(tok0, resp_out_r, state.written_symbol_count);

        let tok1 = join(tok1_0, tok1_1, tok1_2);

        let (buffer_ctrl_option, ram_option, new_state, finish_ctrl) = match state.status {
            Status::SEND_ACCURACY_LOG_REQ => {
                (
                    (true, BufferCtrl { length: ACCURACY_LOG_WIDTH as Length, output: BufferOutputType::CTRL }),
                    (false, zero!<RamWriteReq>()),
                    State { status: Status::RECV_ACCURACY_LOG, written_symbol_count, ..state },
                    zero!<DecoderCtrl>(),
                )
            },
            Status::RECV_ACCURACY_LOG => {
                let accuracy_log = AccuracyLog:5 + out_ctrl.data as AccuracyLog;
                let remaining_proba = RemainingProba:1 << accuracy_log;

                (
                    (false, zero!<BufferCtrl>()),
                    (false, zero!<RamWriteReq>()),
                    State {
                        status: Status::SEND_SYMBOL_REQ,
                        accuracy_log,
                        remaining_proba,
                        written_symbol_count,
                    ..state
                    },
                    zero!<DecoderCtrl>(),
                )
            },
            Status::SEND_SYMBOL_REQ => {
                let bit_width = get_bit_width(state.remaining_proba);
                (
                    (true, BufferCtrl { length: bit_width as Length, output: BufferOutputType::DATA }),
                    (false, zero!<RamWriteReq>()),
                    State { status: Status::RECV_SYMBOL, written_symbol_count, ..state },
                    zero!<DecoderCtrl>(),
                )
            },
            Status::RECV_SYMBOL => {
                let bit_width = get_bit_width(state.remaining_proba);
                let lower_mask = get_lower_mask(bit_width);
                let threshold = get_threshold(bit_width, state.remaining_proba as u16);

                let mask = (u16:1 << out_data.length) - u16:1;
                let data = out_data.data as u16;
                assert!(data & mask == data, "data should not contain additional bits");

                let value = get_adjusted_value(data, state.remainder);
                let (remainder, value) = if (value & lower_mask) < threshold {
                    (Remainder { value: value[bit_width - u16:1+:u1], valid: true }, value & lower_mask)
                } else if value > lower_mask {
                    (zero!<Remainder>(), value - threshold)
                } else {
                    (zero!<Remainder>(), value)
                };

                let proba = value as s16 - s16:1;
                let proba_points = if proba < s16:0 { RemainingProba:1 } else { proba as RemainingProba };
                assert!(proba_points <= state.remaining_proba, "corrupted_data");

                let remaining_proba = state.remaining_proba - proba_points;
                let symbol_count = state.symbol_count + SymbolCount:1;
                let remainder_count = if remainder.valid { u16:1 } else { u16:0 };

                // received all the symbols
                if remaining_proba == RemainingProba:0 {
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
                // there are remaining symbols, and next symbol is normal
                } else if remaining_proba > RemainingProba:0 && proba != s16:0 {
                    let next_bit_width = get_bit_width(remaining_proba) - remainder_count;
                    (
                        (true, BufferCtrl {
                            length: next_bit_width as Length,
                            output: BufferOutputType::DATA
                        }),
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
                // there are remaining symbols, and next data is info about zero probability
                } else if remaining_proba > RemainingProba:0 && proba == s16:0 {
                    let next_bit_width = u16:2 - remainder_count;
                    (
                        (true, BufferCtrl {
                            length: next_bit_width as Length,
                            output: BufferOutputType::DATA
                        }),
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
                let zero_proba_count = out_data.data as SymbolCount;
                let zero_proba_length = out_data.length as SymbolCount;
                let zero_proba_count = get_adjusted_value(zero_proba_count as u16, state.remainder) as SymbolCount;

                // all zero probabilitis received
                if zero_proba_count == SymbolCount:0 {
                    let new_status = if state.remaining_proba > RemainingProba:0 {
                        Status::SEND_SYMBOL_REQ
                    } else if state.remaining_proba == RemainingProba:0 {
                        Status::WAIT_FOR_COMPLETION
                    } else {
                        Status::INVALID
                    };

                    (
                        (true, zero!<BufferCtrl>()),
                        (false, zero!<RamWriteReq>()),
                        State {
                            status: new_status,
                            remainder: zero!<Remainder>(),
                            written_symbol_count,
                        ..state
                        },
                        zero!<DecoderCtrl>(),
                    )
                // some zero probabilities left
                } else {
                    let next_recv_zero = zero_proba_count == SymbolCount:3;
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
                let zero_proba_count = state.zero_proba_count - SymbolCount:1;
                let symbol_count = state.symbol_count + SymbolCount:1;

                let write_req = RamWriteReq {
                    addr: state.symbol_count as RamAddr,
                    data: RamData:0,
                    mask: std::unsigned_max_value<RAM_NUM_PARTITIONS>()
                };

                if zero_proba_count == SymbolCount:0 && state.next_recv_zero == true {
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
                } else if zero_proba_count == SymbolCount:0 && state.next_recv_zero == false {
                    (
                        (false, zero!<BufferCtrl>()), (true, write_req),
                        State {
                            status: Status::SEND_SYMBOL_REQ,
                            next_recv_zero: false,
                            zero_proba_count: SymbolCount:0,
                            written_symbol_count,
                            symbol_count,
                        ..state
                        }, zero!<DecoderCtrl>(),
                    )
                } else {
                    (
                        (false, zero!<BufferCtrl>()), (true, write_req),
                        State {
                            status: Status::WRITE_ZERO_PROBA,
                            zero_proba_count,
                            symbol_count,
                            written_symbol_count,
                        ..state
                        }, zero!<DecoderCtrl>(),
                    )
                }
            },
            Status::WAIT_FOR_COMPLETION => {
                if written_symbol_count == state.symbol_count {
                    (
                        (false, zero!<BufferCtrl>()),
                        (false, zero!<RamWriteReq>()),
                        zero!<State>(),
                        DecoderCtrl { remainder: state.remainder, finished: true },
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
        let tok2_0 = send_if(tok1, buff_in_ctrl_s, do_send_ctrl, ctrl_data);

        let (do_send_ram, ram_data) = ram_option;
        let tok2_1 = send_if(tok1, wr_req_s, do_send_ram, ram_data);
        let tok2_2 = send_if(tok1, resp_in_s, do_send_ram, state.symbol_count == SymbolCount:0);

        let (do_send_finish, finish_data) = (finish_ctrl.finished, finish_ctrl);
        let tok2_3 = send_if(tok1, fse_table_ctrl_s, do_send_finish, finish_data);

        // unused channels
        send_if(tok0, rd_req_s, false, zero!<RamReadReq>());
        recv_if(tok0, rd_resp_r, false, zero!<RamReadResp>());

        new_state
    }
}

const INST_RAM_SIZE = common::FSE_MAX_SYMBOLS;
const INST_RAM_ADDR_WIDTH = std::clog2(INST_RAM_SIZE);
const INST_RAM_DATA_WIDTH = get_bit_width(RemainingProba:1 << common::FSE_MAX_ACCURACY_LOG) as u32;
const INST_RAM_WORD_PARTITION_SIZE = INST_RAM_DATA_WIDTH;
const INST_RAM_NUM_PARTITIONS = ram::num_partitions(INST_RAM_WORD_PARTITION_SIZE, INST_RAM_DATA_WIDTH);
const INST_DATA_WIDTH = common::DATA_WIDTH;
const INST_LENGTH_WIDTH = common::BLOCK_PACKET_WIDTH;

proc FseProbaFreqDecoderInst {
    rd_req_s: chan<ram::ReadReq<INST_RAM_ADDR_WIDTH, INST_RAM_NUM_PARTITIONS>> out;
    rd_resp_r: chan<ram::ReadResp<INST_RAM_DATA_WIDTH>> in;

    config(
        fse_table_ctrl_s: chan<DecoderCtrl> out,
        buff_in_ctrl_s: chan<shift_buffer::ShiftBufferCtrl<INST_LENGTH_WIDTH>> out,
        buff_out_data_r: chan<shift_buffer::ShiftBufferOutput<INST_DATA_WIDTH, INST_LENGTH_WIDTH>> in,
        buff_out_ctrl_r: chan<shift_buffer::ShiftBufferOutput<INST_DATA_WIDTH, INST_LENGTH_WIDTH>> in,
        rd_req_s: chan<ram::ReadReq<INST_RAM_ADDR_WIDTH, INST_RAM_NUM_PARTITIONS>> out,
        rd_resp_r: chan<ram::ReadResp<INST_RAM_DATA_WIDTH>> in,
        wr_req_s: chan<ram::WriteReq<INST_RAM_ADDR_WIDTH, INST_RAM_DATA_WIDTH, INST_RAM_NUM_PARTITIONS>> out,
        wr_resp_r: chan<ram::WriteResp> in) {

        spawn FseProbaFreqDecoder<INST_RAM_DATA_WIDTH, INST_RAM_SIZE, INST_RAM_WORD_PARTITION_SIZE>(
            fse_table_ctrl_s,
            buff_in_ctrl_s, buff_out_data_r, buff_out_ctrl_r,
            rd_req_s, rd_resp_r, wr_req_s, wr_resp_r
        );

        (rd_req_s, rd_resp_r)
    }

    init { }
    next(state: ()) { }
}

const TEST_RAM_DATA_WIDTH = u32:16;
const TEST_RAM_SIZE = u32:100;
const TEST_RAM_ADDR_WIDTH = std::clog2(TEST_RAM_SIZE);
const TEST_RAM_WORD_PARTITION_SIZE = TEST_RAM_DATA_WIDTH;
const TEST_RAM_NUM_PARTITIONS = ram::num_partitions(TEST_RAM_WORD_PARTITION_SIZE, TEST_RAM_DATA_WIDTH);
const TEST_DATA_WIDTH = common::DATA_WIDTH;
const TEST_LENGTH_WIDTH = common::BLOCK_PACKET_WIDTH;

#[test_proc]
proc FseProbaFreqDecoderTest {
    type ReadReq = ram::ReadReq<TEST_RAM_ADDR_WIDTH, TEST_RAM_NUM_PARTITIONS>;
    type ReadResp = ram::ReadResp<TEST_RAM_DATA_WIDTH>;
    type WriteReq = ram::WriteReq<TEST_RAM_ADDR_WIDTH, TEST_RAM_DATA_WIDTH, TEST_RAM_NUM_PARTITIONS>;
    type WriteResp = ram::WriteResp;
    type BufferCtrl = shift_buffer::ShiftBufferCtrl<TEST_LENGTH_WIDTH>;
    type BufferOutput = shift_buffer::ShiftBufferOutput<TEST_DATA_WIDTH, TEST_LENGTH_WIDTH>;
    type BufferOutputType = shift_buffer::ShiftBufferOutputType;
    type RamAddr = bits[TEST_RAM_ADDR_WIDTH];
    type RamData = uN[TEST_RAM_DATA_WIDTH];
    type RamDataSigned = sN[TEST_RAM_DATA_WIDTH];

    terminator: chan<bool> out;
    seq_data_s: chan<SequenceData> out;
    fse_table_ctrl_r: chan<DecoderCtrl> in;
    rd_req_s: chan<ReadReq> out;
    rd_resp_r: chan<ReadResp> in;
    wr_req_s: chan<WriteReq> out;
    wr_resp_r: chan<WriteResp> in;

    config(terminator: chan<bool> out) {
        // RAM channels
        let (rd_req_s, rd_req_r) = chan<ReadReq>("rd_req");
        let (rd_resp_s, rd_resp_r) = chan<ReadResp>("rd_resp");
        let (wr_req_s, wr_req_r) = chan<WriteReq>("wr_req");
        let (wr_resp_s, wr_resp_r) = chan<WriteResp>("wr_resp");

        // FseProbaFreqDecoder channels
        let (seq_data_s, seq_data_r) = chan<SequenceData>("seq_data");
        let (fse_table_ctrl_s, fse_table_ctrl_r) = chan<DecoderCtrl>("fse_table_ctrl_s");

        let (buff_in_ctrl_s, buff_in_ctrl_r) = chan<BufferCtrl>("buff_in_ctrl");
        let (buff_out_data_s, buff_out_data_r) = chan<BufferOutput>("buff_out_data");
        let (buff_out_ctrl_s, buff_out_ctrl_r) = chan<BufferOutput>("buff_out_ctrl");

        spawn FseInputBuffer<TEST_DATA_WIDTH, TEST_LENGTH_WIDTH>(
            seq_data_r, buff_in_ctrl_r, buff_out_data_s, buff_out_ctrl_s);

        spawn FseProbaFreqDecoder<TEST_RAM_DATA_WIDTH, TEST_RAM_SIZE, TEST_RAM_WORD_PARTITION_SIZE>(
            fse_table_ctrl_s,
            buff_in_ctrl_s, buff_out_data_r, buff_out_ctrl_r,
            rd_req_s, rd_resp_r, wr_req_s, wr_resp_r);

        spawn ram::RamModel<TEST_RAM_DATA_WIDTH, TEST_RAM_SIZE, TEST_RAM_WORD_PARTITION_SIZE>(
            rd_req_r, rd_resp_s, wr_req_r, wr_resp_s);

        (terminator, seq_data_s, fse_table_ctrl_r, rd_req_s, rd_resp_r, wr_req_s, wr_resp_r)
    }

    init { }

    next(state: ()) {
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
        // | 6     | 5           | 3           | 11            |

        const EXPECTED_RAM_CONTENTS = RamData[12]:[
            RamData:96,
            RamData:116,
            RamData:36,
            RamData:0,
            RamData:0, RamData:0, RamData:0,
            RamData:0,
            RamData:2,
            RamData:0,
            RamDataSigned:-1 as RamData,
            RamData:5
        ];

        let tok = join();

        let tok = send(tok, seq_data_s, common::SequenceData {
            bytes: u64:0b111_000_00_001_011_01_11_001_110111_01110101_01100001_0011,
            length: u32:47,
            last: false
        });
        let (tok, _) = recv(tok, fse_table_ctrl_r);

        for ((i, exp_val), tok): ((u32, RamData), token) in enumerate(EXPECTED_RAM_CONTENTS) {
            let tok = send(tok, rd_req_s, ReadReq {
                addr: i as RamAddr,
                mask: std::unsigned_max_value<TEST_RAM_NUM_PARTITIONS>(),
            });

            let (tok, recv_data) = recv(tok, rd_resp_r);
            assert_eq(recv_data.data, exp_val);
            tok
        }((tok));

        // * accuracy_log = 9
        // * probability frequencies:
        // | value | probability | bits (real) | symbol number |
        // | ----- | ----------- | ----------  | ------------- |
        // | 1022  | 511         | 10          | 0             |
        // | 0     | -1          | 2(1)        | 1             |

        const EXPECTED_RAM_CONTENTS = RamData[2]:[
            RamData:511,
            RamDataSigned:-1 as RamData,
        ];

        let tok = send(tok, seq_data_s, common::SequenceData {
            bytes: u64:0b00_1111111110_0100,
            length: u32:16,
            last: false
        });
        let (tok, _) = recv(tok, fse_table_ctrl_r);

        for ((i, exp_val), tok): ((u32, RamData), token) in enumerate(EXPECTED_RAM_CONTENTS) {
            let tok = send(tok, rd_req_s, ReadReq {
                addr: i as RamAddr,
                mask: std::unsigned_max_value<TEST_RAM_NUM_PARTITIONS>(),
            });

            let (tok, recv_data) = recv(tok, rd_resp_r);
            assert_eq(recv_data.data, exp_val);
            tok
        }((tok));

        // * accuracy_log = 9
        // * probability frequencies:
        // | value | probability | bits (real) | symbol number |
        // | ----- | ----------- | ----------  | ------------- |
        // | 1022  | 511         | 10          | 0             |
        // | 2     | -1          | 2(1)        | 1             |

        const EXPECTED_RAM_CONTENTS = RamData[2]:[
            RamData:511,
            RamDataSigned:-1 as RamData,
        ];

        let tok = send(tok, seq_data_s, common::SequenceData {
            bytes: u64:0b10_1111111110_0100,
            length: u32:16,
            last: false
        });
        let (tok, _) = recv(tok, fse_table_ctrl_r);

        for ((i, exp_val), tok): ((u32, RamData), token) in enumerate(EXPECTED_RAM_CONTENTS) {
            let tok = send(tok, rd_req_s, ReadReq {
                addr: i as RamAddr,
                mask: std::unsigned_max_value<TEST_RAM_NUM_PARTITIONS>(),
            });

            let (tok, recv_data) = recv(tok, rd_resp_r);
            assert_eq(recv_data.data, exp_val);
            tok
        }((tok));

        let tok = send(tok, terminator, true);
    }
}
