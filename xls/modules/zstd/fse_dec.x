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
import xls.examples.ram;
import xls.modules.zstd.common;
import xls.modules.zstd.math;
import xls.modules.zstd.refilling_shift_buffer;
import xls.modules.zstd.fse_table_creator;

type FseTableRecord = common::FseTableRecord;

//
// type Base = u16;
// type Symbol = u16;
// type NumOfBits = u16;
//
// struct FseTableRecord {
//     symbol: Symbol,
//     num_of_bits: NumOfBits,
//     base: Base
// }
//
// pub fn bits_to_fse_record(bit: u48) -> FseTableRecord {
//     FseTableRecord {
//         symbol: bit[0:16],
//         num_of_bits: bit[16:32],
//         base: bit[32:48]
//     }
// }
//
// fn fse_record_to_bits(record: FseTableRecord) -> u48 {
//     record.base ++ record.num_of_bits ++ record.symbol
// }



type BlockSyncData = common::BlockSyncData;
type SequenceExecutorMessageType = common::SequenceExecutorMessageType;
type SequenceExecutorPacket = common::SequenceExecutorPacket<common::SYMBOLS_IN_PACKET>;
type CommandConstructorData = common::CommandConstructorData;

type CopyOrMatchLength = common::CopyOrMatchLength;
type CopyOrMatchContent = common::CopyOrMatchContent;

type RefillingSBCtrl = refilling_shift_buffer::RefillingShiftBufferCtrl;
type RefillingSBOutput = refilling_shift_buffer::RefillingShiftBufferOutput;

pub enum FseDecoderStatus: u1 {
    OK = 0,
    ERROR = 1,
}

pub struct FseDecoderCtrl {
    sync: BlockSyncData,
    sequences_count: u24,
    literals_count: u20,
    of_acc_log: u7,
    ll_acc_log: u7,
    ml_acc_log: u7,
}

pub struct FseDecoderFinish {
    status: FseDecoderStatus
}

// 3.1.1.3.2.1.1. Sequence Codes for Lengths and Offsets
const SEQ_MAX_CODES_LL = u8:35;
const SEQ_MAX_CODES_ML = u8:51;

const SEQ_LITERAL_LENGTH_BASELINES = u32[36]:[
    0,  1,  2,   3,   4,   5,    6,    7,    8,    9,     10,    11,
    12, 13, 14,  15,  16,  18,   20,   22,   24,   28,    32,    40,
    48, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536
];
const SEQ_LITERAL_LENGTH_EXTRA_BITS = u8[36]:[
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  1,  1,
    1, 1, 2, 2, 3, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
];

const SEQ_MATCH_LENGTH_BASELINES = u32[53]:[
    3,  4,   5,   6,   7,    8,    9,    10,   11,    12,    13,   14, 15, 16,
    17, 18,  19,  20,  21,   22,   23,   24,   25,    26,    27,   28, 29, 30,
    31, 32,  33,  34,  35,   37,   39,   41,   43,    47,    51,   59, 67, 83,
    99, 131, 259, 515, 1027, 2051, 4099, 8195, 16387, 32771, 65539
];
const SEQ_MATCH_LENGTH_EXTRA_BITS = u8[53]:[
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  1,  1,  1, 1,
    2, 2, 3, 3, 4, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
];

enum SEQ_PART : u2 {
    LiteralLength = 0,
    Offset = 1,
    MatchLength = 2,
}

enum FseDecoderFSM : u5 {
    RECV_CTRL = 0,
    PADDING = 1,
    INIT_OF_STATE = 2,
    INIT_ML_STATE = 3,
    INIT_LL_STATE = 4,
    SEND_RAM_RD_REQ = 5,
    RECV_RAM_RD_RESP = 6,
    READ_OF_BITS = 7,
    READ_ML_BITS = 8,
    READ_LL_BITS = 9,
    UPDATE_OF_STATE = 10,
    UPDATE_ML_STATE = 11,
    UPDATE_LL_STATE = 12,
    SEND_COMMAND_LITERAL = 13,
    SEND_COMMAND_SEQUENCE = 14,
    SEND_LEFTOVER_LITERALS_REQ = 15,
    SEND_FINISH = 16,
}

struct FseDecoderState {
    fsm: FseDecoderFSM,
    ctrl: FseDecoderCtrl,
    sequences_count: u24,
    literals_count: u20,
    of: u64,
    ll: u64,
    ml: u64,
    of_fse_table_record: FseTableRecord,
    ll_fse_table_record: FseTableRecord,
    ml_fse_table_record: FseTableRecord,
    of_fse_table_record_valid: bool,
    ll_fse_table_record_valid: bool,
    ml_fse_table_record_valid: bool,
    of_state: u16,
    ll_state: u16,
    ml_state: u16,
    read_bits: u16,
    read_bits_length: u7,
    read_bits_needed: u7,
    sent_buf_ctrl: bool,
    shift_buffer_error: bool,
    padding: u4,
}

pub proc FseDecoder<
    RAM_DATA_W: u32, RAM_ADDR_W: u32, RAM_NUM_PARTITIONS:u32,
    AXI_DATA_W: u32,
    REFILLING_SB_DATA_W: u32 = {AXI_DATA_W},
    REFILLING_SB_LENGTH_W: u32 = {refilling_shift_buffer::length_width(REFILLING_SB_DATA_W)},
> {
    type FseRamRdReq = ram::ReadReq<RAM_ADDR_W, RAM_NUM_PARTITIONS>;
    type FseRamRdResp = ram::ReadResp<RAM_DATA_W>;

    type RefillingSBCtrl = refilling_shift_buffer::RefillingShiftBufferCtrl<REFILLING_SB_LENGTH_W>;
    type RefillingSBOutput = refilling_shift_buffer::RefillingShiftBufferOutput<REFILLING_SB_DATA_W, REFILLING_SB_LENGTH_W>;

    // control
    ctrl_r: chan<FseDecoderCtrl> in;
    finish_s: chan<FseDecoderFinish> out;

    // shift buffer
    rsb_ctrl_s: chan<RefillingSBCtrl> out;
    rsb_data_r: chan<RefillingSBOutput> in;

    // output command
    command_s: chan<CommandConstructorData> out;

    // RAMs
    ll_fse_rd_req_s: chan<FseRamRdReq> out;
    ll_fse_rd_resp_r: chan<FseRamRdResp> in;

    ml_fse_rd_req_s: chan<FseRamRdReq> out;
    ml_fse_rd_resp_r: chan<FseRamRdResp> in;

    of_fse_rd_req_s: chan<FseRamRdReq> out;
    of_fse_rd_resp_r: chan<FseRamRdResp> in;

    config (
        ctrl_r: chan<FseDecoderCtrl> in,
        finish_s: chan<FseDecoderFinish> out,
        rsb_ctrl_s: chan<RefillingSBCtrl> out,
        rsb_data_r: chan<RefillingSBOutput> in,
        command_s: chan<CommandConstructorData> out,
        ll_fse_rd_req_s: chan<FseRamRdReq> out,
        ll_fse_rd_resp_r: chan<FseRamRdResp> in,
        ml_fse_rd_req_s: chan<FseRamRdReq> out,
        ml_fse_rd_resp_r: chan<FseRamRdResp> in,
        of_fse_rd_req_s: chan<FseRamRdReq> out,
        of_fse_rd_resp_r: chan<FseRamRdResp> in,
    ) {
        (
            ctrl_r, finish_s,
            rsb_ctrl_s, rsb_data_r,
            command_s,
            ll_fse_rd_req_s, ll_fse_rd_resp_r,
            ml_fse_rd_req_s, ml_fse_rd_resp_r,
            of_fse_rd_req_s, of_fse_rd_resp_r,
        )
    }

    init { zero!<FseDecoderState>() }

    next (state: FseDecoderState) {
        type RamAddr = uN[RAM_ADDR_W];
        const RAM_MASK_ALL = std::unsigned_max_value<RAM_NUM_PARTITIONS>();

        let tok0 = join();

        // receive ctrl
        let (_, ctrl, ctrl_valid) = recv_if_non_blocking(tok0, ctrl_r, state.fsm == FseDecoderFSM::RECV_CTRL, zero!<FseDecoderCtrl>());
        if ctrl_valid {
            trace_fmt!("ctrl: {:#x}", ctrl);
        } else {};
        let state = if ctrl_valid {
            FseDecoderState {
                ctrl: ctrl,
                sequences_count: ctrl.sequences_count,
                ..state
            }
        } else { state };

        // receive ram read response
        let (_, ll_rd_resp, ll_rd_resp_valid) = recv_if_non_blocking(tok0, ll_fse_rd_resp_r, state.fsm == FseDecoderFSM::RECV_RAM_RD_RESP, zero!<FseRamRdResp>());
        let (_, ml_rd_resp, ml_rd_resp_valid) = recv_if_non_blocking(tok0, ml_fse_rd_resp_r, state.fsm == FseDecoderFSM::RECV_RAM_RD_RESP, zero!<FseRamRdResp>());
        let (_, of_rd_resp, of_rd_resp_valid) = recv_if_non_blocking(tok0, of_fse_rd_resp_r, state.fsm == FseDecoderFSM::RECV_RAM_RD_RESP, zero!<FseRamRdResp>());

        let ll_fse_table_record = fse_table_creator::bits_to_fse_record(ll_rd_resp.data);
        let ml_fse_table_record = fse_table_creator::bits_to_fse_record(ml_rd_resp.data);
        let of_fse_table_record = fse_table_creator::bits_to_fse_record(of_rd_resp.data);

        // if ll_rd_resp_valid {
        //     trace_fmt!("ll_fse_table_record: {:#x}", ll_fse_table_record);
        // } else {};
        // if ml_rd_resp_valid {
        //     trace_fmt!("ml_fse_table_record: {:#x}", ml_fse_table_record);
        // } else {};
        // if of_rd_resp_valid {
        //     trace_fmt!("of_fse_table_record: {:#x}", of_fse_table_record);
        // } else {};
        // validate LL and ML symbols
        assert!(!(ll_rd_resp_valid && ll_fse_table_record.symbol > SEQ_MAX_CODES_LL), "invalid_literal_length_symbol");
        assert!(!(ml_rd_resp_valid && ml_fse_table_record.symbol > SEQ_MAX_CODES_ML), "invalid_match_length_symbol");

        // request records
        let do_send_ram_rd_req = state.fsm == FseDecoderFSM::SEND_RAM_RD_REQ;

        let ll_req = FseRamRdReq { addr: state.ll_state as RamAddr, mask: RAM_MASK_ALL};
        let ml_req = FseRamRdReq { addr: state.ml_state as RamAddr, mask: RAM_MASK_ALL};
        let of_req = FseRamRdReq { addr: state.of_state as RamAddr, mask: RAM_MASK_ALL};

        send_if(tok0, ll_fse_rd_req_s, do_send_ram_rd_req, ll_req);
        send_if(tok0, ml_fse_rd_req_s, do_send_ram_rd_req, ml_req);
        send_if(tok0, of_fse_rd_req_s, do_send_ram_rd_req, of_req);

        if do_send_ram_rd_req {
            trace_fmt!("ll_req: {:#x}", ll_req);
            trace_fmt!("ml_req: {:#x}", ml_req);
            trace_fmt!("of_req: {:#x}", of_req);
        } else {};

        // read bits
        let do_read_bits = (
            state.fsm == FseDecoderFSM::PADDING ||
            state.fsm == FseDecoderFSM::INIT_OF_STATE ||
            state.fsm == FseDecoderFSM::INIT_ML_STATE ||
            state.fsm == FseDecoderFSM::INIT_LL_STATE ||
            state.fsm == FseDecoderFSM::READ_OF_BITS ||
            state.fsm == FseDecoderFSM::READ_ML_BITS ||
            state.fsm == FseDecoderFSM::READ_LL_BITS ||
            state.fsm == FseDecoderFSM::UPDATE_OF_STATE ||
            state.fsm == FseDecoderFSM::UPDATE_ML_STATE ||
            state.fsm == FseDecoderFSM::UPDATE_LL_STATE
        );
        let do_send_buf_ctrl = do_read_bits && !state.sent_buf_ctrl;

        let buf_ctrl_length = if ((state.read_bits_needed - state.read_bits_length) > REFILLING_SB_DATA_W as u7) {
            REFILLING_SB_DATA_W as u7
        } else {
            state.read_bits_needed - state.read_bits_length
        };

        if do_send_buf_ctrl {
            trace_fmt!("Asking for {:#x} data", buf_ctrl_length);
        } else {};

        send_if(tok0, rsb_ctrl_s, do_send_buf_ctrl, RefillingSBCtrl {
            length: buf_ctrl_length,
        });

        let state = if do_send_buf_ctrl {
            FseDecoderState { sent_buf_ctrl: do_send_buf_ctrl, ..state }
        } else { state };

        let recv_sb_output = (do_read_bits && state.sent_buf_ctrl);
        let (_, buf_data, buf_data_valid) = recv_if_non_blocking(tok0, rsb_data_r, recv_sb_output, zero!<RefillingSBOutput>());
        if buf_data_valid {
            trace_fmt!("[FseDecoder] Received data {:#x} in state {}", buf_data, state.fsm);
        } else { };

        let state = if do_read_bits & buf_data_valid {
            FseDecoderState {
                sent_buf_ctrl: false,
                read_bits: math::logshiftl(buf_data.data as u16, state.read_bits_length) | state.read_bits,
                read_bits_length: state.read_bits_length + buf_data.length,
                shift_buffer_error: state.shift_buffer_error | buf_data.error,
                ..state
            }
        } else { state };

        // send command
        let literals_count = state.literals_count + state.ll as u20;
        let command_data = if state.fsm == FseDecoderFSM::SEND_COMMAND_LITERAL {
            trace_fmt!("(ll: {:#x}, ml: {:#x}, of: {:#x}", state.ll, state.ml, state.of);
            SequenceExecutorPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                length: state.ll,
                content: CopyOrMatchContent:0,
                last: false,
            }
        } else if state.fsm == FseDecoderFSM::SEND_COMMAND_SEQUENCE {
            SequenceExecutorPacket {
                msg_type: SequenceExecutorMessageType::SEQUENCE,
                length: state.ml,
                content: state.of,
                last: (state.sequences_count == u24:1) && (state.literals_count == state.ctrl.literals_count),
            }
        } else {
           SequenceExecutorPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                length: (state.ctrl.literals_count - state.literals_count) as CopyOrMatchLength,
                content: CopyOrMatchContent:0,
                last: true,
            }
        };
        let do_send_command = (
            (command_data.length != CopyOrMatchLength:0) &&
            (
                state.fsm == FseDecoderFSM::SEND_COMMAND_LITERAL ||
                state.fsm == FseDecoderFSM::SEND_COMMAND_SEQUENCE ||
                state.fsm == FseDecoderFSM::SEND_LEFTOVER_LITERALS_REQ
            )
        );
        let command = CommandConstructorData {
            sync: state.ctrl.sync,
            data: command_data,
        };

        if do_send_command {
            trace_fmt!("[FseDecoder] Sending command: {:#x}", command);
        } else {};
        send_if(tok0, command_s, do_send_command, command);

        // send finish
        send_if(tok0, finish_s, state.fsm == FseDecoderFSM::SEND_FINISH, FseDecoderFinish {
            status: if state.shift_buffer_error { FseDecoderStatus::ERROR } else { FseDecoderStatus::OK }
        });

        // update state
        match (state.fsm) {
            FseDecoderFSM::RECV_CTRL => {
                if (ctrl_valid) {
                    trace_fmt!("[FseDecoder]: Moving to PADDING");
                    if ctrl.sequences_count == u24:0 {
                        FseDecoderState {
                            fsm: FseDecoderFSM::SEND_LEFTOVER_LITERALS_REQ,
                            ctrl: ctrl,
                            ..state
                        }
                    } else {
                        FseDecoderState {
                            fsm: FseDecoderFSM::PADDING,
                            ctrl: ctrl,
                            read_bits: u16:0,
                            read_bits_length: u7:0,
                            read_bits_needed: u7:1,
                            ..state
                        }
                    }
                } else { state }
            },
            FseDecoderFSM::PADDING => {
                if (state.read_bits_needed == state.read_bits_length && !state.sent_buf_ctrl) {
                    trace_fmt!("[FseDecoder]: Moving to INIT_LL_STATE");

                    let padding = state.padding + u4:1;
                    assert!(padding <= u4:8, "invalid_padding");

                    let padding_available = (state.read_bits as u1 == u1:0);
                    if padding_available {
                        FseDecoderState {
                            fsm: FseDecoderFSM::PADDING,
                            read_bits: u16:0,
                            read_bits_length: u7:0,
                            read_bits_needed: u7:1,
                            padding, ..state
                        }
                    } else {
                        trace_fmt!("padding is: {:#x}", padding);
                        FseDecoderState {
                            fsm: FseDecoderFSM::INIT_LL_STATE,
                            read_bits: u16:0,
                            read_bits_length: u7:0,
                            read_bits_needed: state.ctrl.ll_acc_log,
                            ..state
                        }
                    }
                } else { state }
            },

            FseDecoderFSM::INIT_LL_STATE => {
                if (state.read_bits_needed == state.read_bits_length && !state.sent_buf_ctrl) {
                    trace_fmt!("[FseDecoder]: Moving to INIT_OF_STATE");
                    FseDecoderState {
                        fsm: FseDecoderFSM::INIT_OF_STATE,
                        ll_state: state.read_bits,
                        read_bits: u16:0,
                        read_bits_length: u7:0,
                        read_bits_needed: state.ctrl.of_acc_log,
                        ..state
                    }
                } else { state }
            },
            FseDecoderFSM::INIT_OF_STATE => {
                if (state.read_bits_needed == state.read_bits_length && !state.sent_buf_ctrl) {
                    trace_fmt!("[FseDecoder]: Moving to INIT_ML_STATE");
                    FseDecoderState {
                        fsm: FseDecoderFSM::INIT_ML_STATE,
                        of_state: state.read_bits,
                        read_bits: u16:0,
                        read_bits_length: u7:0,
                        read_bits_needed: state.ctrl.ml_acc_log,
                        ..state
                    }
                } else { state }
            },
            FseDecoderFSM::INIT_ML_STATE => {
                if (state.read_bits_needed == state.read_bits_length && !state.sent_buf_ctrl) {
                    trace_fmt!("[FseDecoder]: Moving to RAM_RD_REQ");
                    FseDecoderState {
                        fsm: FseDecoderFSM::SEND_RAM_RD_REQ,
                        ml_state: state.read_bits,
                        read_bits: u16:0,
                        read_bits_length: u7:0,
                        read_bits_needed: u7:0,
                        ..state
                    }
                } else { state }
            },
            FseDecoderFSM::SEND_RAM_RD_REQ => {
                // trace_fmt!("State LL: {} ML: {} OF: {}", state.ll_state, state.ml_state, state.of_state);
                trace_fmt!("State LL: {:#x} ML: {:#x} OF: {:#x}", state.ll_state, state.ml_state, state.of_state);
                FseDecoderState {
                    fsm: FseDecoderFSM::RECV_RAM_RD_RESP,
                    ll_fse_table_record_valid: false,
                    ml_fse_table_record_valid: false,
                    of_fse_table_record_valid: false,
                    ..state
                }
            },
            FseDecoderFSM::RECV_RAM_RD_RESP => {
                trace_fmt!("RECV_RAM_RD_RESP");
                // save fse records in state
                let state = if ll_rd_resp_valid {
                    FseDecoderState { ll_fse_table_record: ll_fse_table_record, ll_fse_table_record_valid: true, ..state }
                } else { state };
                let state = if ml_rd_resp_valid {
                    FseDecoderState { ml_fse_table_record: ml_fse_table_record, ml_fse_table_record_valid: true, ..state }
                } else { state };
                let state = if of_rd_resp_valid {
                    FseDecoderState { of_fse_table_record: of_fse_table_record, of_fse_table_record_valid: true, ..state }
                } else { state };

                if (state.ll_fse_table_record_valid &&
                    state.ml_fse_table_record_valid &&
                    state.of_fse_table_record_valid
                ) {
                    trace_fmt!("all states received: {:#x}", state);
                    FseDecoderState {
                        fsm: FseDecoderFSM::READ_OF_BITS,
                        read_bits: u16:0,
                        read_bits_length: u7:0,
                        read_bits_needed: state.of_fse_table_record.symbol as u7,
                        ..state
                    }
                } else { state }
            },
            FseDecoderFSM::READ_OF_BITS => {
                if ((state.read_bits_needed == state.read_bits_length) && !state.sent_buf_ctrl) {
                    trace_fmt!("of_code: {:#x}", state.of_fse_table_record.symbol);
                    FseDecoderState {
                        fsm: FseDecoderFSM::READ_ML_BITS,
                        of: (math::logshiftl(u32:1, state.of_fse_table_record.symbol) + state.read_bits as u32) as u64,
                        read_bits: u16:0,
                        read_bits_length: u7:0,
                        read_bits_needed: SEQ_MATCH_LENGTH_EXTRA_BITS[state.ml_fse_table_record.symbol] as u7,
                        ..state
                    }
                } else { state }
            },
            FseDecoderFSM::READ_ML_BITS => {
                if ((state.read_bits_needed == state.read_bits_length) && !state.sent_buf_ctrl) {
                    FseDecoderState {
                        fsm: FseDecoderFSM::READ_LL_BITS,
                        ml: (SEQ_MATCH_LENGTH_BASELINES[state.ml_fse_table_record.symbol] + state.read_bits as u32) as u64,
                        read_bits: u16:0,
                        read_bits_length: u7:0,
                        read_bits_needed: SEQ_LITERAL_LENGTH_EXTRA_BITS[state.ll_fse_table_record.symbol] as u7,
                        ..state
                    }
                } else { state }

            },
            FseDecoderFSM::READ_LL_BITS => {
                if ((state.read_bits_needed == state.read_bits_length) && !state.sent_buf_ctrl) {
                    if state.sequences_count == u24:1 {
                        // skip state update for last sequence
                        FseDecoderState {
                            fsm: FseDecoderFSM::SEND_COMMAND_LITERAL,
                            of_state: state.of_fse_table_record.base + state.read_bits,
                            ll: (SEQ_LITERAL_LENGTH_BASELINES[state.ll_fse_table_record.symbol] + state.read_bits as u32) as u64,
                            read_bits: u16:0,
                            read_bits_length: u7:0,
                            read_bits_needed: u7:0,
                            ..state
                        }
                    } else {
                        FseDecoderState {
                            fsm: FseDecoderFSM::UPDATE_LL_STATE,
                            ll: (SEQ_LITERAL_LENGTH_BASELINES[state.ll_fse_table_record.symbol] + state.read_bits as u32) as u64,
                            read_bits: u16:0,
                            read_bits_length: u7:0,
                            read_bits_needed: state.ll_fse_table_record.num_of_bits as u7,
                            ..state
                        }
                    }
                } else { state }
            },
            FseDecoderFSM::UPDATE_LL_STATE => {
                trace_fmt!("Values LL: {:#x} ML: {:#x} OF: {:#x}", state.ll, state.ml, state.of);
                if ((state.read_bits_needed == state.read_bits_length) && !state.sent_buf_ctrl) {
                    FseDecoderState {
                        fsm: FseDecoderFSM::UPDATE_ML_STATE,
                        ll_state: state.ll_fse_table_record.base + state.read_bits,
                        read_bits: u16:0,
                        read_bits_length: u7:0,
                        read_bits_needed: state.ml_fse_table_record.num_of_bits as u7,
                        ..state
                    }
                } else { state }
            },
            FseDecoderFSM::UPDATE_ML_STATE => {
                if ((state.read_bits_needed == state.read_bits_length) && !state.sent_buf_ctrl) {
                    FseDecoderState {
                        fsm: FseDecoderFSM::UPDATE_OF_STATE,
                        ml_state: state.ml_fse_table_record.base + state.read_bits,
                        read_bits: u16:0,
                        read_bits_length: u7:0,
                        read_bits_needed: state.of_fse_table_record.num_of_bits as u7,
                        ..state
                    }
                } else { state }
            },
            FseDecoderFSM::UPDATE_OF_STATE => {
                if ((state.read_bits_needed == state.read_bits_length) && !state.sent_buf_ctrl) {
                    FseDecoderState {
                        fsm: FseDecoderFSM::SEND_COMMAND_LITERAL,
                        of_state: state.of_fse_table_record.base + state.read_bits,
                        read_bits: u16:0,
                        read_bits_length: u7:0,
                        read_bits_needed: u7:0,
                        ..state
                    }
                } else { state }
            },
            FseDecoderFSM::SEND_COMMAND_LITERAL => {
                    trace_fmt!("LITERALS_COUNT: {:#x}/{:#x}", literals_count, state.ctrl.literals_count);
                    FseDecoderState {
                        fsm: FseDecoderFSM::SEND_COMMAND_SEQUENCE,
                            literals_count, ..state
                    }
            },
            FseDecoderFSM::SEND_COMMAND_SEQUENCE => {
                if (state.sequences_count == u24:1) {
                    if state.literals_count < state.ctrl.literals_count {
                        trace_fmt!("Going to LEFTOVER");
                        FseDecoderState {
                            fsm: FseDecoderFSM::SEND_LEFTOVER_LITERALS_REQ,
                            sequences_count: u24:0,
                            ..state
                        }
                    } else if state.literals_count == state.ctrl.literals_count {
                        trace_fmt!("Going to FINISH");
                        FseDecoderState {
                            fsm: FseDecoderFSM::SEND_FINISH,
                            sequences_count: u24:0,
                            ..state
                        }
                    } else {
                        fail!("too_many_literals", zero!<FseDecoderState>())
                    }
                } else {
                    FseDecoderState {
                        fsm: FseDecoderFSM::SEND_RAM_RD_REQ,
                        sequences_count: state.sequences_count - u24:1,
                        ..state
                    }
                }
            },
           FseDecoderFSM::SEND_LEFTOVER_LITERALS_REQ => {
                FseDecoderState {
                    fsm:FseDecoderFSM::SEND_FINISH,
                    ..zero!<FseDecoderState>()
                }
            },

            FseDecoderFSM::SEND_FINISH => {
                FseDecoderState {
                    fsm:FseDecoderFSM::RECV_CTRL,
                    ..zero!<FseDecoderState>()
                }
            },
            _ => {
                fail!("impossible_case", state)
            },
        }
    }
}

const INST_RAM_SIZE = common::FSE_MAX_SYMBOLS;
const INST_RAM_ADDR_W = std::clog2(INST_RAM_SIZE);
const INST_RAM_DATA_W = u32:32;
const INST_RAM_WORD_PARTITION_SIZE = INST_RAM_DATA_W / u32:3;
const INST_RAM_NUM_PARTITIONS = ram::num_partitions(
    INST_RAM_WORD_PARTITION_SIZE, INST_RAM_DATA_W);
const INST_AXI_DATA_W = u32:64;
const INST_REFILLING_SB_DATA_W = INST_AXI_DATA_W;
const INST_REFILLING_SB_LENGTH_W = refilling_shift_buffer::length_width(INST_REFILLING_SB_DATA_W);

pub proc FseDecoderInst {
    type FseRamRdReq = ram::ReadReq<INST_RAM_ADDR_W, INST_RAM_NUM_PARTITIONS>;
    type FseRamRdResp = ram::ReadResp<INST_RAM_DATA_W>;

    type RefillingSBCtrl = refilling_shift_buffer::RefillingShiftBufferCtrl<INST_REFILLING_SB_LENGTH_W>;
    type RefillingSBOutput = refilling_shift_buffer::RefillingShiftBufferOutput<INST_REFILLING_SB_DATA_W, INST_REFILLING_SB_LENGTH_W>;

    config (
        ctrl_r: chan<FseDecoderCtrl> in,
        finish_s: chan<FseDecoderFinish> out,
        rsb_ctrl_s: chan<RefillingSBCtrl> out,
        rsb_data_r: chan<RefillingSBOutput> in,
        command_s: chan<CommandConstructorData> out,
        ll_fse_rd_req_s: chan<FseRamRdReq> out,
        ll_fse_rd_resp_r: chan<FseRamRdResp> in,
        ml_fse_rd_req_s: chan<FseRamRdReq> out,
        ml_fse_rd_resp_r: chan<FseRamRdResp> in,
        of_fse_rd_req_s: chan<FseRamRdReq> out,
        of_fse_rd_resp_r: chan<FseRamRdResp> in,
    ) {
        spawn FseDecoder<
            INST_RAM_DATA_W, INST_RAM_ADDR_W, INST_RAM_NUM_PARTITIONS,
            INST_AXI_DATA_W,
        >(
            ctrl_r, finish_s,
            rsb_ctrl_s, rsb_data_r,
            command_s,
            ll_fse_rd_req_s, ll_fse_rd_resp_r,
            ml_fse_rd_req_s, ml_fse_rd_resp_r,
            of_fse_rd_req_s, of_fse_rd_resp_r,
        );
    }

    init { () }

    next (state: ()) {}
}

// test data was generated using decodecorpus and educational_decoder from zstd repository
// block #0 seed: 58602
// block #1 seed: 48401

const TEST_OF_TABLE = u32[256][2]:[[
    u32:0x00_03_0008, u32:0x02_02_0004, u32:0x03_02_0014, u32:0x03_02_0018, u32:0x04_03_0008, u32:0x00_03_0010, u32:0x02_02_0008, u32:0x03_02_001c,
    u32:0x03_01_0000, u32:0x04_03_0010, u32:0x02_02_000c, u32:0x02_02_0010, u32:0x03_01_0002, u32:0x04_03_0018, u32:0x00_03_0018, u32:0x02_02_0014,
    u32:0x03_01_0004, u32:0x03_01_0006, u32:0x04_02_0000, u32:0x02_02_0018, u32:0x02_02_001c, u32:0x03_01_0008, u32:0x03_01_000a, u32:0x00_02_0000,
    u32:0x02_01_0000, u32:0x03_01_000c, u32:0x03_01_000e, u32:0x04_02_0004, u32:0x00_02_0004, u32:0x02_01_0002, u32:0x03_01_0010, u32:0x03_01_0012,
    u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0031, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000,
    u32:0x31_51_0100, u32:0x00_00_0101, u32:0x00_00_0003, u32:0x00_00_0101, u32:0x00_00_0301, u32:0x00_00_0101, u32:0x00_00_0301, u32:0x00_00_0100,
    u32:0x03_08_0101, u32:0x02_00_0103, u32:0x02_04_0101, u32:0x02_00_0001, u32:0x03_14_0101, u32:0x03_00_0301, u32:0x02_18_0100, u32:0x02_00_0101,
    u32:0x01_08_0000, u32:0x03_00_0000, u32:0x02_10_0000, u32:0x02_00_0000, u32:0x01_08_0031, u32:0x03_00_0000, u32:0x03_1c_0000, u32:0x02_00_0000,
    u32:0x01_00_0103, u32:0x01_00_0101, u32:0x02_10_0303, u32:0x02_00_0101, u32:0x02_0c_0301, u32:0x01_00_0101, u32:0x01_10_0301, u32:0x02_00_0103,
    u32:0x01_02_0000, u32:0x01_00_0002, u32:0x01_18_0000, u32:0x02_00_0200, u32:0x02_18_0000, u32:0x01_00_0200, u32:0x01_14_0002, u32:0x01_00_0000,
    u32:0x00_04_0000, u32:0x00_00_0000, u32:0x00_06_0000, u32:0x00_00_0000, u32:0x00_00_0051, u32:0x00_00_0000, u32:0x00_18_0000, u32:0x00_00_0000,
    u32:0x51_1c_0008, u32:0x00_00_000c, u32:0x00_08_000e, u32:0x00_00_0010, u32:0x00_0a_0008, u32:0x00_00_0010, u32:0x00_00_0012, u32:0x00_00_0014,
    u32:0x08_00_0016, u32:0x00_00_0010, u32:0x04_0c_0018, u32:0x00_00_001a, u32:0x14_0e_001c, u32:0x00_00_0018, u32:0x18_04_0018, u32:0x00_00_001e,
    u32:0x08_04_0000, u32:0x00_00_0001, u32:0x10_02_0000, u32:0x00_00_0002, u32:0x08_10_0003, u32:0x00_00_0004, u32:0x1c_12_0005, u32:0x00_00_0000,
    u32:0x00_00_0006, u32:0x00_00_0007, u32:0x10_00_0008, u32:0x00_00_0004, u32:0x0c_00_0004, u32:0x00_00_0009, u32:0x10_00_000a, u32:0x00_00_000b,
    u32:0x02_31_0000, u32:0x00_00_0000, u32:0x18_00_0000, u32:0x00_00_0000, u32:0x18_00_0411, u32:0x00_00_0000, u32:0x14_00_0000, u32:0x00_00_0000,
    u32:0x04_00_3230, u32:0x00_01_3020, u32:0x06_01_2030, u32:0x00_01_3233, u32:0x00_03_3033, u32:0x00_00_2020, u32:0x18_01_3030, u32:0x00_01_3020,
    u32:0x1c_01_2031, u32:0x00_03_3033, u32:0x08_01_3333, u32:0x00_01_2020, u32:0x0a_01_3830, u32:0x00_03_3020, u32:0x00_00_2031, u32:0x00_01_3333,
    u32:0x00_01_3333, u32:0x00_01_2020, u32:0x0c_03_3030, u32:0x00_01_3020, u32:0x0e_01_2031, u32:0x00_01_3033, u32:0x04_01_3032, u32:0x00_00_2020,
    u32:0x04_01_6530, u32:0x00_01_3020, u32:0x02_01_2031, u32:0x00_03_3032, u32:0x10_00_3133, u32:0x00_01_2020, u32:0x12_01_3030, u32:0x00_01_3020,
    u32:0x00_00_2031, u32:0x00_00_3032, u32:0x00_00_3032, u32:0x00_00_2020, u32:0x00_00_3231, u32:0x00_00_3020, u32:0x00_00_2031, u32:0x00_00_3032,
    u32:0x31_31_3133, u32:0x00_00_2020, u32:0x00_00_3030, u32:0x00_00_3020, u32:0x00_00_2030, u32:0x00_00_3033, u32:0x00_00_3233, u32:0x00_00_2020,
    u32:0x00_03_000a, u32:0x01_01_0000, u32:0x01_01_0000, u32:0x01_01_0000, u32:0x03_03_0000, u32:0x00_03_0000, u32:0x01_01_0000, u32:0x01_01_0000,
    u32:0x01_01_0000, u32:0x03_03_0000, u32:0x01_01_0000, u32:0x01_01_0000, u32:0x01_01_0000, u32:0x03_03_0000, u32:0x00_03_0000, u32:0x01_01_0000,
    u32:0x01_00_0000, u32:0x01_00_0000, u32:0x03_02_0000, u32:0x01_00_0000, u32:0x01_00_0000, u32:0x01_00_0000, u32:0x01_00_0000, u32:0x00_02_0000,
    u32:0x01_00_0000, u32:0x01_00_0000, u32:0x01_00_0000, u32:0x03_02_0000, u32:0x00_02_0000, u32:0x01_00_0000, u32:0x01_00_0000, u32:0x01_00_0000,
    u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000,
    u32:0x31_51_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000,
    u32:0x03_08_0000, u32:0x01_00_0000, u32:0x01_0c_0000, u32:0x01_00_0000, u32:0x03_0e_0000, u32:0x03_00_0000, u32:0x01_10_0000, u32:0x01_00_0000,
    u32:0x01_08_0000, u32:0x03_00_0000, u32:0x01_10_0000, u32:0x01_00_0000, u32:0x01_12_0000, u32:0x03_00_0000, u32:0x03_14_0000, u32:0x01_00_0000,
    u32:0x00_16_0000, u32:0x00_00_0000, u32:0x02_10_0000, u32:0x00_00_0000, u32:0x00_18_0000, u32:0x00_00_0000, u32:0x00_1a_0000, u32:0x02_00_0000,
    u32:0x00_1c_0000, u32:0x00_00_0000, u32:0x00_18_0000, u32:0x02_00_0000, u32:0x02_18_0000, u32:0x00_00_0000, u32:0x00_1e_0000, u32:0x00_00_0000,
],[
    u32:0x00_05_0000, u32:0x06_04_0000, u32:0x09_05_0000, u32:0x0f_05_0000, u32:0x15_05_0000, u32:0x03_05_0000, u32:0x07_04_0000, u32:0x0c_05_0000,
    u32:0x12_05_0000, u32:0x17_05_0000, u32:0x05_05_0000, u32:0x08_04_0000, u32:0x0e_05_0000, u32:0x14_05_0000, u32:0x02_05_0000, u32:0x07_04_0010,
    u32:0x0b_05_0000, u32:0x11_05_0000, u32:0x16_05_0000, u32:0x04_05_0000, u32:0x08_04_0010, u32:0x0d_05_0000, u32:0x13_05_0000, u32:0x01_05_0000,
    u32:0x06_04_0010, u32:0x0a_05_0000, u32:0x10_05_0000, u32:0x1c_05_0000, u32:0x1b_05_0000, u32:0x1a_05_0000, u32:0x19_05_0000, u32:0x18_05_0000,
    u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0051, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000,
    u32:0x31_51_0100, u32:0x00_00_0302, u32:0x00_00_0605, u32:0x00_00_0a08, u32:0x00_00_100d, u32:0x00_00_1613, u32:0x00_00_1c19, u32:0x00_00_211f,
    u32:0x05_00_2523, u32:0x04_00_2927, u32:0x05_00_2d2b, u32:0x05_00_0201, u32:0x05_00_0403, u32:0x05_00_0706, u32:0x04_00_0c09, u32:0x05_00_120f,
    u32:0x05_00_1815, u32:0x05_00_1e1b, u32:0x05_00_2220, u32:0x04_00_2624, u32:0x05_00_2a28, u32:0x05_00_012c, u32:0x05_00_0201, u32:0x04_00_0504,
    u32:0x05_00_0807, u32:0x05_00_0e0b, u32:0x05_00_1411, u32:0x05_00_1a17, u32:0x04_00_341d, u32:0x05_00_3233, u32:0x05_00_3031, u32:0x05_00_2e2f,
    u32:0x04_00_0000, u32:0x05_00_0000, u32:0x05_00_0000, u32:0x05_00_0000, u32:0x05_00_0051, u32:0x05_00_0000, u32:0x05_10_0000, u32:0x05_00_0000,
    u32:0x00_00_0406, u32:0x00_00_0505, u32:0x00_00_0505, u32:0x00_00_0605, u32:0x00_00_0606, u32:0x00_00_0606, u32:0x00_00_0606, u32:0x00_00_0606,
    u32:0x51_10_0606, u32:0x00_00_0606, u32:0x00_00_0606, u32:0x00_00_0404, u32:0x00_00_0505, u32:0x00_00_0505, u32:0x00_00_0606, u32:0x00_00_0606,
    u32:0x00_10_0606, u32:0x00_00_0606, u32:0x00_00_0606, u32:0x00_00_0606, u32:0x00_00_0606, u32:0x00_00_0406, u32:0x00_00_0404, u32:0x00_00_0505,
    u32:0x00_00_0505, u32:0x00_00_0606, u32:0x00_00_0606, u32:0x00_00_0606, u32:0x00_00_0606, u32:0x00_00_0606, u32:0x00_00_0606, u32:0x00_00_0606,
    u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0091, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000,
    u32:0x00_51_0000, u32:0x00_00_0000, u32:0x00_00_0020, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x10_00_0000, u32:0x00_00_0000,
    u32:0x00_00_0000, u32:0x00_01_0000, u32:0x00_02_0000, u32:0x00_03_0000, u32:0x00_05_0000, u32:0x00_06_0000, u32:0x00_08_0000, u32:0x00_0a_0000,
    u32:0x10_0d_0000, u32:0x00_10_0000, u32:0x00_13_0000, u32:0x00_16_0000, u32:0x00_19_0000, u32:0x00_1c_0000, u32:0x00_1f_0010, u32:0x00_21_0000,
    u32:0x10_23_0020, u32:0x00_25_0000, u32:0x00_27_0020, u32:0x00_29_0000, u32:0x00_2b_0000, u32:0x00_2d_0000, u32:0x00_01_0000, u32:0x00_02_0000,
    u32:0x00_03_0000, u32:0x00_04_0000, u32:0x00_06_0000, u32:0x00_07_0000, u32:0x00_09_0000, u32:0x00_0c_0000, u32:0x00_0f_0000, u32:0x00_12_0000,
    u32:0x00_15_0000, u32:0x00_18_0000, u32:0x00_1b_0000, u32:0x00_1e_0020, u32:0x00_20_0030, u32:0x00_22_0010, u32:0x00_24_0020, u32:0x00_26_0020,
    u32:0x51_28_0020, u32:0x00_2a_0020, u32:0x00_2c_0000, u32:0x00_01_0000, u32:0x00_01_0000, u32:0x00_02_0000, u32:0x00_04_0000, u32:0x00_05_0000,
    u32:0x00_07_0000, u32:0x01_08_0000, u32:0x02_0b_0000, u32:0x03_0e_0000, u32:0x05_11_0000, u32:0x06_14_0000, u32:0x08_17_0000, u32:0x0a_1a_0000,
    u32:0x0d_1d_0000, u32:0x10_34_0000, u32:0x13_33_0000, u32:0x16_32_0000, u32:0x19_31_0411, u32:0x1c_30_0000, u32:0x1f_2f_0000, u32:0x21_2e_0000,
    u32:0x23_00_6430, u32:0x25_00_3020, u32:0x27_00_2030, u32:0x29_00_3436, u32:0x2b_00_3033, u32:0x2d_00_2020, u32:0x01_00_3532, u32:0x02_00_3020,
    u32:0x03_51_2030, u32:0x04_00_3033, u32:0x06_00_3333, u32:0x07_00_2020, u32:0x09_00_3630, u32:0x0c_00_3020, u32:0x0f_00_2030, u32:0x12_00_3333,
    u32:0x15_06_3333, u32:0x18_04_2020, u32:0x1b_05_3730, u32:0x1e_05_3020, u32:0x20_05_2035, u32:0x22_05_3033, u32:0x24_05_3032, u32:0x26_06_2020,
    u32:0x28_06_3032, u32:0x2a_06_3020, u32:0x2c_06_2035, u32:0x01_06_3032, u32:0x01_06_3533, u32:0x02_06_2020, u32:0x04_06_3230, u32:0x05_06_3020,
    u32:0x07_06_2036, u32:0x08_06_3032, u32:0x0b_06_3032, u32:0x0e_06_2020, u32:0x11_06_3430, u32:0x14_06_3020, u32:0x17_04_2036, u32:0x1a_04_3032,
    u32:0x1d_05_3633, u32:0x34_05_2020, u32:0x33_05_6131, u32:0x32_05_3020, u32:0x31_06_2034, u32:0x30_06_3033, u32:0x2f_06_3233, u32:0x2e_06_2020,
    u32:0x00_06_000a, u32:0x00_06_0000, u32:0x00_06_0000, u32:0x00_06_0000, u32:0x00_06_0000, u32:0x00_06_0000, u32:0x00_06_0000, u32:0x00_06_0000,
    u32:0x51_06_0000, u32:0x00_06_0000, u32:0x00_06_0000, u32:0x00_04_0000, u32:0x00_04_0000, u32:0x00_04_0000, u32:0x00_05_0000, u32:0x00_05_0000,
]];

const TEST_ML_TABLE = u32[256][2]:[[
    u32:0x00_03_0008, u32:0x01_01_000c, u32:0x01_01_000e, u32:0x01_01_0010, u32:0x03_03_0008, u32:0x00_03_0010, u32:0x01_01_0012, u32:0x01_01_0014,
    u32:0x01_01_0016, u32:0x03_03_0010, u32:0x01_01_0018, u32:0x01_01_001a, u32:0x01_01_001c, u32:0x03_03_0018, u32:0x00_03_0018, u32:0x01_01_001e,
    u32:0x01_00_0000, u32:0x01_00_0001, u32:0x03_02_0000, u32:0x01_00_0002, u32:0x01_00_0003, u32:0x01_00_0004, u32:0x01_00_0005, u32:0x00_02_0000,
    u32:0x01_00_0006, u32:0x01_00_0007, u32:0x01_00_0008, u32:0x03_02_0004, u32:0x00_02_0004, u32:0x01_00_0009, u32:0x01_00_000a, u32:0x01_00_000b,
    u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0411, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000,
    u32:0x31_51_3030, u32:0x00_00_3520, u32:0x00_00_2031, u32:0x00_00_3033, u32:0x00_00_3033, u32:0x00_00_2020, u32:0x00_00_3030, u32:0x00_00_3020,
    u32:0x03_08_2030, u32:0x01_00_3533, u32:0x01_0c_3333, u32:0x01_00_2020, u32:0x03_0e_3130, u32:0x03_00_3020, u32:0x01_10_2063, u32:0x01_00_3333,
    u32:0x01_08_3333, u32:0x03_00_2020, u32:0x01_10_3130, u32:0x01_00_3020, u32:0x01_12_2030, u32:0x03_00_3033, u32:0x03_14_3032, u32:0x01_00_2020,
    u32:0x00_16_3130, u32:0x00_00_3120, u32:0x02_10_2032, u32:0x00_00_3032, u32:0x00_18_3033, u32:0x00_00_2020, u32:0x00_1a_3030, u32:0x02_00_3020,
    u32:0x00_1c_2030, u32:0x00_00_3032, u32:0x00_18_3032, u32:0x02_00_2020, u32:0x02_18_3030, u32:0x00_00_3120, u32:0x00_1e_2061, u32:0x00_00_3032,
    u32:0x00_00_3136, u32:0x00_00_2020, u32:0x00_01_3030, u32:0x00_00_3020, u32:0x00_00_2030, u32:0x00_00_3033, u32:0x00_02_3233, u32:0x00_00_2020,
    u32:0x51_03_000a, u32:0x00_00_0000, u32:0x00_04_0000, u32:0x00_00_0000, u32:0x00_05_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000,
    u32:0x08_06_0000, u32:0x00_00_0000, u32:0x0c_07_0000, u32:0x00_00_0000, u32:0x0e_08_0000, u32:0x00_00_0000, u32:0x10_04_0000, u32:0x00_00_0000,
    u32:0x08_04_0000, u32:0x00_00_0000, u32:0x10_09_0000, u32:0x00_00_0000, u32:0x12_0a_0000, u32:0x00_00_0000, u32:0x14_0b_0000, u32:0x00_00_0000,
    u32:0x16_00_0000, u32:0x00_00_0000, u32:0x10_00_0000, u32:0x00_00_0000, u32:0x18_00_0000, u32:0x00_00_0000, u32:0x1a_00_0000, u32:0x00_00_0000,
    u32:0x1c_11_0000, u32:0x00_04_0000, u32:0x18_00_0000, u32:0x00_00_0000, u32:0x18_00_0000, u32:0x00_00_0000, u32:0x1e_00_0000, u32:0x00_00_0000,
    u32:0x00_31_0000, u32:0x00_30_0000, u32:0x01_20_0000, u32:0x00_33_0000, u32:0x00_31_0000, u32:0x00_20_0000, u32:0x02_30_0000, u32:0x00_30_0000,
    u32:0x03_30_0000, u32:0x00_30_0000, u32:0x04_20_0000, u32:0x00_20_0000, u32:0x05_30_0000, u32:0x00_30_0000, u32:0x00_20_0000, u32:0x00_33_0000,
    u32:0x06_30_0000, u32:0x00_20_0000, u32:0x07_30_0000, u32:0x00_30_0000, u32:0x08_30_0000, u32:0x00_30_0000, u32:0x04_20_0000, u32:0x00_20_0000,
    u32:0x04_30_0000, u32:0x00_37_0000, u32:0x09_20_0000, u32:0x00_32_0000, u32:0x0a_30_0000, u32:0x00_20_0000, u32:0x0b_30_0000, u32:0x00_30_0000,
    u32:0x00_30_0000, u32:0x00_30_0000, u32:0x00_20_0000, u32:0x00_20_0000, u32:0x00_30_0000, u32:0x00_30_0000, u32:0x00_20_0000, u32:0x00_32_0000,
    u32:0x11_30_0000, u32:0x04_20_0000, u32:0x00_30_0000, u32:0x00_30_0000, u32:0x00_30_0000, u32:0x00_30_0000, u32:0x00_20_0000, u32:0x00_20_0000,
    u32:0x31_30_0000, u32:0x31_30_0000, u32:0x20_20_0000, u32:0x33_33_0000, u32:0x30_30_0000, u32:0x20_20_0000, u32:0x30_30_0000, u32:0x30_30_0000,
    u32:0x30_30_0000, u32:0x30_30_0000, u32:0x20_20_0000, u32:0x20_20_0000, u32:0x33_32_0000, u32:0x30_30_0000, u32:0x20_20_0000, u32:0x33_33_0000,
    u32:0x30_30_0000, u32:0x20_20_0000, u32:0x30_30_0000, u32:0x30_30_0000, u32:0x30_30_0000, u32:0x30_30_0000, u32:0x20_20_0000, u32:0x20_20_0000,
    u32:0x33_32_0000, u32:0x30_30_0000, u32:0x20_20_0000, u32:0x32_32_0000, u32:0x30_30_0000, u32:0x20_20_0000, u32:0x30_30_0000, u32:0x30_30_0000,
    u32:0x30_30_0000, u32:0x30_30_0000, u32:0x20_20_0000, u32:0x20_20_0000, u32:0x32_33_0000, u32:0x30_30_0000, u32:0x20_20_0000, u32:0x32_33_0000,
    u32:0x30_33_0000, u32:0x20_20_0000, u32:0x30_30_0000, u32:0x30_30_0000, u32:0x30_30_0000, u32:0x30_30_0000, u32:0x20_20_0000, u32:0x20_20_0000,
    u32:0x33_0a_0000, u32:0x30_00_0000, u32:0x20_00_0000, u32:0x33_00_0000, u32:0x30_00_0000, u32:0x20_00_0000, u32:0x30_00_0000, u32:0x30_00_0000,
    u32:0x30_00_0000, u32:0x30_00_0000, u32:0x20_00_0000, u32:0x20_00_0000, u32:0x32_00_0000, u32:0x30_00_0000, u32:0x20_00_0000, u32:0x30_00_0000,
    u32:0x30_00_0000, u32:0x20_00_0000, u32:0x30_00_0000, u32:0x30_00_0000, u32:0x30_00_0000, u32:0x30_00_0000, u32:0x20_00_0000, u32:0x20_00_0000,
    u32:0x32_00_0000, u32:0x30_00_0000, u32:0x20_00_0000, u32:0x30_00_0000, u32:0x30_00_0000, u32:0x20_00_0000, u32:0x30_00_0000, u32:0x30_00_0000,
],[
    u32:0x00_06_0000, u32:0x01_04_0000, u32:0x02_05_0020, u32:0x03_05_0000, u32:0x05_05_0000, u32:0x06_05_0000, u32:0x08_05_0000, u32:0x0a_06_0000,
    u32:0x0d_06_0000, u32:0x10_06_0000, u32:0x13_06_0000, u32:0x16_06_0000, u32:0x19_06_0000, u32:0x1c_06_0000, u32:0x1f_06_0000, u32:0x21_06_0000,
    u32:0x23_06_0000, u32:0x25_06_0000, u32:0x27_06_0000, u32:0x29_06_0000, u32:0x2b_06_0000, u32:0x2d_06_0000, u32:0x01_04_0010, u32:0x02_04_0000,
    u32:0x03_05_0020, u32:0x04_05_0000, u32:0x06_05_0020, u32:0x07_05_0000, u32:0x09_06_0000, u32:0x0c_06_0000, u32:0x0f_06_0000, u32:0x12_06_0000,
    u32:0x15_06_0000, u32:0x18_06_0000, u32:0x1b_06_0000, u32:0x1e_06_0000, u32:0x20_06_0000, u32:0x22_06_0000, u32:0x24_06_0000, u32:0x26_06_0000,
    u32:0x28_06_0000, u32:0x2a_06_0000, u32:0x2c_06_0000, u32:0x01_04_0020, u32:0x01_04_0030, u32:0x02_04_0010, u32:0x04_05_0020, u32:0x05_05_0020,
    u32:0x07_05_0020, u32:0x08_05_0020, u32:0x0b_06_0000, u32:0x0e_06_0000, u32:0x11_06_0000, u32:0x14_06_0000, u32:0x17_06_0000, u32:0x1a_06_0000,
    u32:0x1d_06_0000, u32:0x34_06_0000, u32:0x33_06_0000, u32:0x32_06_0000, u32:0x31_06_0000, u32:0x30_06_0000, u32:0x2f_06_0000, u32:0x2e_06_0000,
    u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0411, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000,
    u32:0x51_91_3030, u32:0x00_00_3920, u32:0x00_00_2031, u32:0x00_00_3033, u32:0x00_00_3033, u32:0x00_00_2020, u32:0x00_00_3030, u32:0x00_00_3020,
    u32:0x06_00_2030, u32:0x04_00_3933, u32:0x05_00_3333, u32:0x05_00_2020, u32:0x05_20_3530, u32:0x05_00_3020, u32:0x05_00_2030, u32:0x06_00_3333,
    u32:0x06_00_3333, u32:0x06_00_2020, u32:0x06_00_3530, u32:0x06_00_3020, u32:0x06_00_2030, u32:0x06_00_3033, u32:0x06_00_3032, u32:0x06_00_2020,
    u32:0x06_00_3630, u32:0x06_00_3020, u32:0x06_00_2030, u32:0x06_00_3032, u32:0x06_00_3033, u32:0x06_00_2020, u32:0x04_00_3630, u32:0x04_00_3020,
    u32:0x05_00_2030, u32:0x05_00_3032, u32:0x05_00_3032, u32:0x05_00_2020, u32:0x06_00_3430, u32:0x06_00_3020, u32:0x06_00_2030, u32:0x06_00_3032,
    u32:0x06_00_3033, u32:0x06_00_2020, u32:0x06_00_3630, u32:0x06_00_3020, u32:0x06_00_2030, u32:0x06_00_3033, u32:0x06_00_3233, u32:0x06_00_2020,
    u32:0x06_00_000a, u32:0x06_00_0000, u32:0x06_00_0000, u32:0x04_00_0000, u32:0x04_10_0000, u32:0x04_00_0000, u32:0x05_00_0000, u32:0x05_00_0000,
    u32:0x05_20_0000, u32:0x05_00_0000, u32:0x06_00_0000, u32:0x06_00_0000, u32:0x06_20_0000, u32:0x06_00_0000, u32:0x06_00_0000, u32:0x06_00_0000,
    u32:0x06_00_0000, u32:0x06_00_0000, u32:0x06_00_0000, u32:0x06_00_0000, u32:0x06_00_0000, u32:0x06_00_0000, u32:0x06_00_0000, u32:0x06_00_0000,
    u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000,
    u32:0x91_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000,
    u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x20_00_0000, u32:0x00_00_0000, u32:0x00_20_0000, u32:0x00_00_0000,
    u32:0x00_30_0000, u32:0x00_00_0000, u32:0x00_10_0000, u32:0x00_00_0000, u32:0x00_20_0000, u32:0x00_00_0000, u32:0x00_20_0000, u32:0x00_00_0000,
    u32:0x00_20_0000, u32:0x00_00_0000, u32:0x00_20_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000,
    u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000,
    u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000,
    u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x10_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000,
    u32:0x20_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x20_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000,
    u32:0x00_11_0000, u32:0x00_04_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000,
    u32:0x00_30_0000, u32:0x00_30_0000, u32:0x00_20_0000, u32:0x00_33_0000, u32:0x00_30_0000, u32:0x00_20_0000, u32:0x00_30_0000, u32:0x00_30_0000,
    u32:0x00_30_0000, u32:0x00_30_0000, u32:0x00_20_0000, u32:0x00_20_0000, u32:0x00_30_0000, u32:0x00_30_0000, u32:0x00_20_0000, u32:0x00_33_0000,
    u32:0x00_30_0000, u32:0x00_20_0000, u32:0x00_30_0000, u32:0x00_30_0000, u32:0x00_30_0000, u32:0x00_30_0000, u32:0x20_20_0000, u32:0x00_20_0000,
    u32:0x30_30_0000, u32:0x00_30_0000, u32:0x10_20_0000, u32:0x00_32_0000, u32:0x20_30_0000, u32:0x00_20_0000, u32:0x20_30_0000, u32:0x00_30_0000,
]];

const TEST_LL_TABLE = u32[256][2]:[[
    u32:0x00_01_000e, u32:0x00_01_0010, u32:0x00_01_0012, u32:0x00_01_0014, u32:0x01_02_0004, u32:0x00_01_0016, u32:0x00_01_0018, u32:0x00_01_001a,
    u32:0x01_02_0008, u32:0x01_02_000c, u32:0x00_01_001c, u32:0x00_01_001e, u32:0x00_00_0000, u32:0x01_02_0010, u32:0x00_00_0001, u32:0x00_00_0002,
    u32:0x00_00_0003, u32:0x01_02_0014, u32:0x01_02_0018, u32:0x00_00_0004, u32:0x00_00_0005, u32:0x00_00_0006, u32:0x01_02_001c, u32:0x00_00_0007,
    u32:0x00_00_0008, u32:0x00_00_0009, u32:0x00_00_000a, u32:0x01_01_0000, u32:0x00_00_000b, u32:0x00_00_000c, u32:0x00_00_000d, u32:0x01_01_0002,
    u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0031, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000,
    u32:0x31_51_0200, u32:0x00_00_0303, u32:0x00_00_0004, u32:0x00_00_0302, u32:0x00_00_0403, u32:0x00_00_0202, u32:0x00_00_0403, u32:0x00_00_0200,
    u32:0x01_0e_0303, u32:0x01_00_0204, u32:0x01_10_0302, u32:0x01_00_0003, u32:0x02_12_0302, u32:0x01_00_0403, u32:0x01_14_0200, u32:0x01_00_0303,
    u32:0x02_04_0000, u32:0x02_00_0000, u32:0x01_16_0000, u32:0x01_00_0000, u32:0x00_18_0031, u32:0x02_00_0000, u32:0x00_1a_0000, u32:0x00_00_0000,
    u32:0x00_08_0203, u32:0x02_00_0202, u32:0x02_0c_0303, u32:0x00_00_0202, u32:0x00_1c_0301, u32:0x00_00_0202, u32:0x02_1e_0301, u32:0x00_00_0203,
    u32:0x00_00_0101, u32:0x00_00_0202, u32:0x00_10_0102, u32:0x01_00_0201, u32:0x00_01_0101, u32:0x00_00_0201, u32:0x00_02_0102, u32:0x01_00_0101,
    u32:0x00_03_0000, u32:0x00_00_0000, u32:0x00_14_0000, u32:0x00_00_0000, u32:0x00_18_0051, u32:0x00_00_0000, u32:0x00_04_0000, u32:0x00_00_0000,
    u32:0x51_05_0008, u32:0x00_00_0004, u32:0x00_06_0014, u32:0x00_00_0018, u32:0x00_1c_0008, u32:0x00_00_0010, u32:0x00_07_0008, u32:0x00_00_001c,
    u32:0x0e_08_0000, u32:0x00_00_0010, u32:0x10_09_000c, u32:0x00_00_0010, u32:0x12_0a_0002, u32:0x00_00_0018, u32:0x14_00_0018, u32:0x00_00_0014,
    u32:0x04_0b_0004, u32:0x00_00_0006, u32:0x16_0c_0000, u32:0x00_00_0018, u32:0x18_0d_001c, u32:0x00_00_0008, u32:0x1a_02_000a, u32:0x00_00_0000,
    u32:0x08_00_0000, u32:0x00_00_000c, u32:0x0c_00_000e, u32:0x00_00_0004, u32:0x1c_00_0004, u32:0x00_00_0002, u32:0x1e_00_0010, u32:0x00_00_0012,
    u32:0x00_31_0000, u32:0x00_00_0000, u32:0x10_00_0000, u32:0x00_00_0000, u32:0x01_00_0031, u32:0x00_00_0000, u32:0x02_00_0000, u32:0x00_00_0000,
    u32:0x03_00_0100, u32:0x00_02_0101, u32:0x14_03_0003, u32:0x00_03_0101, u32:0x18_04_0301, u32:0x00_00_0101, u32:0x04_02_0301, u32:0x00_03_0100,
    u32:0x05_03_0101, u32:0x00_04_0103, u32:0x06_02_0101, u32:0x00_02_0001, u32:0x1c_03_0101, u32:0x00_04_0301, u32:0x07_00_0100, u32:0x00_02_0101,
    u32:0x08_03_0000, u32:0x00_03_0000, u32:0x09_04_0000, u32:0x00_02_0000, u32:0x0a_02_0031, u32:0x00_03_0000, u32:0x00_03_0000, u32:0x00_00_0000,
    u32:0x0b_02_0103, u32:0x00_03_0101, u32:0x0c_03_0303, u32:0x00_04_0101, u32:0x0d_00_0301, u32:0x00_02_0101, u32:0x02_03_0301, u32:0x00_03_0103,
    u32:0x00_00_0000, u32:0x00_00_0002, u32:0x00_00_0000, u32:0x00_00_0200, u32:0x00_00_0000, u32:0x00_00_0200, u32:0x00_00_0002, u32:0x00_00_0000,
    u32:0x31_31_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0051, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000,
    u32:0x00_03_0008, u32:0x02_02_000c, u32:0x03_02_000e, u32:0x03_02_0010, u32:0x04_03_0008, u32:0x00_03_0010, u32:0x02_02_0012, u32:0x03_02_0014,
    u32:0x03_01_0016, u32:0x04_03_0010, u32:0x02_02_0018, u32:0x02_02_001a, u32:0x03_01_001c, u32:0x04_03_0018, u32:0x00_03_0018, u32:0x02_02_001e,
    u32:0x03_01_0000, u32:0x03_01_0001, u32:0x04_02_0000, u32:0x02_02_0002, u32:0x02_02_0003, u32:0x03_01_0004, u32:0x03_01_0005, u32:0x00_02_0000,
    u32:0x02_01_0006, u32:0x03_01_0007, u32:0x03_01_0008, u32:0x04_02_0004, u32:0x00_02_0004, u32:0x02_01_0009, u32:0x03_01_000a, u32:0x03_01_000b,
    u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0411, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000,
    u32:0x31_51_3030, u32:0x00_00_3520, u32:0x00_00_2031, u32:0x00_00_3033, u32:0x00_00_3033, u32:0x00_00_2020, u32:0x00_00_3030, u32:0x00_00_3020,
    u32:0x03_08_2030, u32:0x02_00_3533, u32:0x02_04_3333, u32:0x02_00_2020, u32:0x03_14_3230, u32:0x03_00_3020, u32:0x02_18_2034, u32:0x02_00_3333,
    u32:0x01_08_3333, u32:0x03_00_2020, u32:0x02_10_3230, u32:0x02_00_3020, u32:0x01_08_2030, u32:0x03_00_3033, u32:0x03_1c_3032, u32:0x02_00_2020,
    u32:0x01_00_3130, u32:0x01_00_3020, u32:0x02_10_2038, u32:0x02_00_3032, u32:0x02_0c_3033, u32:0x01_00_2020, u32:0x01_10_3130, u32:0x02_00_3020,
    u32:0x01_02_2030, u32:0x01_00_3032, u32:0x01_18_3032, u32:0x02_00_2020, u32:0x02_18_3130, u32:0x01_00_3120, u32:0x01_14_2030, u32:0x01_00_3032,
],[
    u32:0x00_02_0010, u32:0x00_02_0014, u32:0x01_03_0008, u32:0x03_03_0008, u32:0x0d_03_0008, u32:0x00_02_0018, u32:0x00_02_001c, u32:0x03_03_0010,
    u32:0x05_03_0008, u32:0x0d_03_0010, u32:0x00_01_0000, u32:0x01_03_0010, u32:0x03_03_0018, u32:0x0d_03_0018, u32:0x00_01_0002, u32:0x00_01_0004,
    u32:0x01_03_0018, u32:0x05_03_0010, u32:0x0d_02_0000, u32:0x00_01_0006, u32:0x01_02_0000, u32:0x03_02_0000, u32:0x05_03_0018, u32:0x00_01_0008,
    u32:0x00_01_000a, u32:0x01_02_0004, u32:0x05_02_0000, u32:0x0d_02_0004, u32:0x00_01_000c, u32:0x00_01_000e, u32:0x03_02_0004, u32:0x05_02_0004,
    u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0031, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000,
    u32:0x31_51_0600, u32:0x00_00_0f09, u32:0x00_00_0315, u32:0x00_00_0c07, u32:0x00_00_1712, u32:0x00_00_0805, u32:0x00_00_140e, u32:0x00_00_0702,
    u32:0x02_10_110b, u32:0x02_00_0416, u32:0x03_14_0d08, u32:0x03_00_0113, u32:0x03_08_0a06, u32:0x02_00_1c10, u32:0x02_08_1a1b, u32:0x03_00_1819,
    u32:0x03_08_0000, u32:0x03_00_0000, u32:0x01_18_0000, u32:0x03_00_0000, u32:0x03_1c_0031, u32:0x03_00_0000, u32:0x01_10_0000, u32:0x01_00_0000,
    u32:0x03_08_0405, u32:0x03_00_0505, u32:0x02_10_0505, u32:0x01_00_0504, u32:0x02_00_0505, u32:0x02_00_0405, u32:0x03_10_0505, u32:0x01_00_0405,
    u32:0x01_18_0505, u32:0x02_00_0505, u32:0x02_18_0504, u32:0x02_00_0505, u32:0x01_02_0504, u32:0x01_00_0505, u32:0x02_04_0505, u32:0x02_00_0505,
    u32:0x00_18_0000, u32:0x00_00_0000, u32:0x00_10_0000, u32:0x00_00_0000, u32:0x00_00_0051, u32:0x00_00_0000, u32:0x00_06_0000, u32:0x00_00_0000,
    u32:0x51_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_18_0000, u32:0x00_00_0000, u32:0x00_08_0000, u32:0x00_00_0000,
    u32:0x10_0a_0000, u32:0x00_00_0000, u32:0x14_04_0000, u32:0x00_00_0000, u32:0x08_00_0000, u32:0x00_00_0000, u32:0x08_04_0000, u32:0x00_00_0010,
    u32:0x08_0c_0000, u32:0x00_00_0000, u32:0x18_0e_0000, u32:0x00_00_0000, u32:0x1c_04_0010, u32:0x00_00_0000, u32:0x10_04_0000, u32:0x00_00_0000,
    u32:0x08_00_0010, u32:0x00_00_0000, u32:0x10_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x10_00_0000, u32:0x00_00_0000,
    u32:0x18_31_0000, u32:0x00_00_0000, u32:0x18_00_0000, u32:0x00_00_0000, u32:0x02_00_0051, u32:0x00_00_0000, u32:0x04_00_0000, u32:0x00_00_0000,
    u32:0x18_00_0100, u32:0x00_06_0302, u32:0x10_09_0605, u32:0x00_0f_0a08, u32:0x00_15_100d, u32:0x00_03_1613, u32:0x06_07_1c19, u32:0x00_0c_211f,
    u32:0x00_12_2523, u32:0x00_17_2927, u32:0x00_05_2d2b, u32:0x00_08_0201, u32:0x18_0e_0403, u32:0x00_14_0706, u32:0x08_02_0c09, u32:0x00_07_120f,
    u32:0x0a_0b_1815, u32:0x00_11_1e1b, u32:0x04_16_2220, u32:0x00_04_2624, u32:0x00_08_2a28, u32:0x00_0d_012c, u32:0x04_13_0201, u32:0x00_01_0504,
    u32:0x0c_06_0807, u32:0x00_0a_0e0b, u32:0x0e_10_1411, u32:0x00_1c_1a17, u32:0x04_1b_341d, u32:0x00_1a_3233, u32:0x04_19_3031, u32:0x00_18_2e2f,
    u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0051, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000,
    u32:0x31_31_0406, u32:0x00_00_0505, u32:0x00_00_0505, u32:0x00_00_0605, u32:0x00_00_0606, u32:0x00_00_0606, u32:0x00_00_0606, u32:0x00_00_0606,
    u32:0x00_05_0606, u32:0x06_04_0606, u32:0x09_05_0606, u32:0x0f_05_0404, u32:0x15_05_0505, u32:0x03_05_0505, u32:0x07_04_0606, u32:0x0c_05_0606,
    u32:0x12_05_0606, u32:0x17_05_0606, u32:0x05_05_0606, u32:0x08_04_0606, u32:0x0e_05_0606, u32:0x14_05_0406, u32:0x02_05_0404, u32:0x07_04_0505,
    u32:0x0b_05_0505, u32:0x11_05_0606, u32:0x16_05_0606, u32:0x04_05_0606, u32:0x08_04_0606, u32:0x0d_05_0606, u32:0x13_05_0606, u32:0x01_05_0606,
    u32:0x06_04_0000, u32:0x0a_05_0000, u32:0x10_05_0000, u32:0x1c_05_0000, u32:0x1b_05_0091, u32:0x1a_05_0000, u32:0x19_05_0000, u32:0x18_05_0000,
    u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0020, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000,
    u32:0x31_51_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000, u32:0x00_00_0000,
    u32:0x05_00_0000, u32:0x04_00_0000, u32:0x05_00_0000, u32:0x05_00_0000, u32:0x05_00_0000, u32:0x05_00_0000, u32:0x04_00_0010, u32:0x05_00_0000,
    u32:0x05_00_0020, u32:0x05_00_0000, u32:0x05_00_0020, u32:0x04_00_0000, u32:0x05_00_0000, u32:0x05_00_0000, u32:0x05_00_0000, u32:0x04_00_0000,
    u32:0x05_00_0000, u32:0x05_00_0000, u32:0x05_00_0000, u32:0x05_00_0000, u32:0x04_00_0000, u32:0x05_00_0000, u32:0x05_00_0000, u32:0x05_00_0000,
    u32:0x04_00_0000, u32:0x05_00_0000, u32:0x05_00_0000, u32:0x05_00_0020, u32:0x05_00_0030, u32:0x05_00_0010, u32:0x05_10_0020, u32:0x05_00_0020,
]];

const TEST_SYNC = BlockSyncData[2]:[
    BlockSyncData {id: u32:1234, last_block: false},
    BlockSyncData {id: u32:1235, last_block: true},
];

const TEST_CTRL = FseDecoderCtrl[2]:[
    FseDecoderCtrl {
        sync: TEST_SYNC[0],
        sequences_count: u24:8,
        literals_count: u20:0,
        of_acc_log: u7:5,
        ll_acc_log: u7:5,
        ml_acc_log: u7:5,
    },
    FseDecoderCtrl {
        sync: TEST_SYNC[1],
        sequences_count: u24:7,
        literals_count: u20:0,
        of_acc_log: u7:5,
        ll_acc_log: u7:5,
        ml_acc_log: u7:6,
    },
];


const TEST_AXI_DATA_W = u32:64;
const TEST_REFILLING_SB_DATA_W = TEST_AXI_DATA_W;
const TEST_REFILLING_SB_LENGTH_W = refilling_shift_buffer::length_width(TEST_REFILLING_SB_DATA_W);
const TEST_RAM_DATA_W = u32:32;
const TEST_RAM_SIZE = common::FSE_MAX_SYMBOLS;
const TEST_RAM_ADDR_W = std::clog2(TEST_RAM_SIZE);
const TEST_RAM_WORD_PARTITION_SIZE = TEST_RAM_DATA_W / u32:3;
const TEST_RAM_NUM_PARTITIONS = ram::num_partitions(
    TEST_RAM_WORD_PARTITION_SIZE, TEST_RAM_DATA_W);

type TestRefillingSBOutput = refilling_shift_buffer::RefillingShiftBufferOutput<TEST_REFILLING_SB_DATA_W, TEST_REFILLING_SB_LENGTH_W>;

const TEST_DATA_0 = TestRefillingSBOutput[48]:[
    // init states
    TestRefillingSBOutput { error: false, data: u64:0b11111, length: u7:5},
    TestRefillingSBOutput { error: false, data: u64:0b101, length: u7:5},
    TestRefillingSBOutput { error: false, data: u64:0b10, length: u7:5},
    // symbols (seq #0)
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:0},
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:0},
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:0},
    // state update
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:1},
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:1},
    TestRefillingSBOutput { error: false, data: u64:0b100, length: u7:3},
    // symbols (seq #1)
    TestRefillingSBOutput { error: false, data: u64:0b10, length: u7:2},
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:0},
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:0},
    // state update
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:1},
    TestRefillingSBOutput { error: false, data: u64:0b110, length: u7:3},
    TestRefillingSBOutput { error: false, data: u64:0b10, length: u7:2},
    // symbols (seq #2)
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:3},
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:0},
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:0},
    // state update
    TestRefillingSBOutput { error: false, data: u64:0b10, length: u7:2},
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:0},
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:1},
    // symbols (seq #3)
    TestRefillingSBOutput { error: false, data: u64:0b11, length: u7:3},
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:0},
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:0},
    // state update
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:0},
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:1},
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:1},
    // symbols (seq #4)
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:4},
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:0},
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:0},
    // state update
    TestRefillingSBOutput { error: false, data: u64:0b1, length: u7:1},
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:0},
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:3},
    // symbols (seq #5)
    TestRefillingSBOutput { error: false, data: u64:0b101, length: u7:3},
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:0},
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:0},
    // state update
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:0},
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:1},
    TestRefillingSBOutput { error: false, data: u64:0b1, length: u7:1},
    // symbols (seq #6)
    TestRefillingSBOutput { error: false, data: u64:0b11, length: u7:2},
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:0},
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:0},
    // state update
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:0},
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:2},
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:2},
    // symbols (seq #7)
    TestRefillingSBOutput { error: false, data: u64:0b1000, length: u7:4},
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:0},
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:0},
    // no state update for last sequence
];

const TEST_DATA_1 = TestRefillingSBOutput[42]:[
    // init states
    TestRefillingSBOutput { error: false, data: u64:0b10000, length: u7:5},
    TestRefillingSBOutput { error: false, data: u64:0b1110, length: u7:5},
    TestRefillingSBOutput { error: false, data: u64:0b11001, length: u7:6},
    // symbols (seq #0)
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:2},
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:0},
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:0},
    // state update
    TestRefillingSBOutput { error: false, data: u64:0b110, length: u7:3},
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:5},
    TestRefillingSBOutput { error: false, data: u64:0b1110, length: u7:5},
    // symbols (seq #1)
    TestRefillingSBOutput { error: false, data: u64:0b10, length: u7:2},
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:0},
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:0},
    // state update
    TestRefillingSBOutput { error: false, data: u64:0b10, length: u7:2},
    TestRefillingSBOutput { error: false, data: u64:0b1, length: u7:6},
    TestRefillingSBOutput { error: false, data: u64:0b101, length: u7:5},
    // symbols (seq #2)
    TestRefillingSBOutput { error: false, data: u64:0b110, length: u7:3},
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:0},
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:0},
    // state update
    TestRefillingSBOutput { error: false, data: u64:0b11, length: u7:2},
    TestRefillingSBOutput { error: false, data: u64:0b1, length: u7:4},
    TestRefillingSBOutput { error: false, data: u64:0b10011, length: u7:5},
    // symbols (seq #3)
    TestRefillingSBOutput { error: false, data: u64:0b11, length: u7:4},
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:0},
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:0},
    // state update
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:2},
    TestRefillingSBOutput { error: false, data: u64:0b1, length: u7:4},
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:5},
    // symbols (seq #4)
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:0},
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:0},
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:0},
    // state update
    TestRefillingSBOutput { error: false, data: u64:0b10, length: u7:3},
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:4},
    TestRefillingSBOutput { error: false, data: u64:0b1010, length: u7:5},
    // symbols (seq #5)
    TestRefillingSBOutput { error: false, data: u64:0b1110, length: u7:5},
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:0},
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:0},
    // state update
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:1},
    TestRefillingSBOutput { error: false, data: u64:0b11, length: u7:6},
    TestRefillingSBOutput { error: false, data: u64:0b10011, length: u7:5},
    // symbols (seq #6)
    TestRefillingSBOutput { error: false, data: u64:0b10, length: u7:4},
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:0},
    TestRefillingSBOutput { error: false, data: u64:0b0, length: u7:0},
    // no state update for last sequence
];
// FIXME: test error propagation with TestRefillingSBOutput { error: true, ...}


fn test_command(block_idx: u32, msg_type: SequenceExecutorMessageType, length: CopyOrMatchLength, content: CopyOrMatchContent, last: bool) -> CommandConstructorData {
    CommandConstructorData {
        sync: TEST_SYNC[block_idx],
        data: SequenceExecutorPacket {
            msg_type: msg_type,
            length: length,
            content: content,
            last: last,
        },
    }
}

const TEST_EXPECTED_COMMANDS_0 = CommandConstructorData[16]:[
    // block #0
    // seq #0
    test_command(u32:0, SequenceExecutorMessageType::LITERAL, CopyOrMatchLength:1, CopyOrMatchContent:0, false),
    test_command(u32:0, SequenceExecutorMessageType::SEQUENCE, CopyOrMatchLength:4, CopyOrMatchContent:1, false),
    // seq #1
    test_command(u32:0, SequenceExecutorMessageType::LITERAL, CopyOrMatchLength:0, CopyOrMatchContent:0, false),
    test_command(u32:0, SequenceExecutorMessageType::SEQUENCE, CopyOrMatchLength:3, CopyOrMatchContent:6, false),
    // seq #2
    test_command(u32:0, SequenceExecutorMessageType::LITERAL, CopyOrMatchLength:1, CopyOrMatchContent:0, false),
    test_command(u32:0, SequenceExecutorMessageType::SEQUENCE, CopyOrMatchLength:4, CopyOrMatchContent:8, false),
    // seq #3
    test_command(u32:0, SequenceExecutorMessageType::LITERAL, CopyOrMatchLength:0, CopyOrMatchContent:0, false),
    test_command(u32:0, SequenceExecutorMessageType::SEQUENCE, CopyOrMatchLength:4, CopyOrMatchContent:11, false),
    // seq #4
    test_command(u32:0, SequenceExecutorMessageType::LITERAL, CopyOrMatchLength:0, CopyOrMatchContent:0, false),
    test_command(u32:0, SequenceExecutorMessageType::SEQUENCE, CopyOrMatchLength:4, CopyOrMatchContent:16, false),
    // seq #5
    test_command(u32:0, SequenceExecutorMessageType::LITERAL, CopyOrMatchLength:0, CopyOrMatchContent:0, false),
    test_command(u32:0, SequenceExecutorMessageType::SEQUENCE, CopyOrMatchLength:4, CopyOrMatchContent:13, false),
    // seq #6
    test_command(u32:0, SequenceExecutorMessageType::LITERAL, CopyOrMatchLength:0, CopyOrMatchContent:0, false),
    test_command(u32:0, SequenceExecutorMessageType::SEQUENCE, CopyOrMatchLength:6, CopyOrMatchContent:7, false),
    // seq #7
    test_command(u32:0, SequenceExecutorMessageType::LITERAL, CopyOrMatchLength:0, CopyOrMatchContent:0, false),
    test_command(u32:0, SequenceExecutorMessageType::SEQUENCE, CopyOrMatchLength:3, CopyOrMatchContent:24, true),
];

const TEST_EXPECTED_COMMANDS_1 = CommandConstructorData[14]:[
    // block #1
    // seq #0
    test_command(u32:1, SequenceExecutorMessageType::LITERAL, CopyOrMatchLength:1, CopyOrMatchContent:0, false),
    test_command(u32:1, SequenceExecutorMessageType::SEQUENCE, CopyOrMatchLength:7, CopyOrMatchContent:4, false),
    // seq #1
    test_command(u32:1, SequenceExecutorMessageType::LITERAL, CopyOrMatchLength:3, CopyOrMatchContent:0, false),
    test_command(u32:1, SequenceExecutorMessageType::SEQUENCE, CopyOrMatchLength:3, CopyOrMatchContent:6, false),
    // seq #2
    test_command(u32:1, SequenceExecutorMessageType::LITERAL, CopyOrMatchLength:0, CopyOrMatchContent:0, false),
    test_command(u32:1, SequenceExecutorMessageType::SEQUENCE, CopyOrMatchLength:4, CopyOrMatchContent:14, false),
    // seq #3
    test_command(u32:1, SequenceExecutorMessageType::LITERAL, CopyOrMatchLength:5, CopyOrMatchContent:0, false),
    test_command(u32:1, SequenceExecutorMessageType::SEQUENCE, CopyOrMatchLength:4, CopyOrMatchContent:19, false),
    // seq #4
    test_command(u32:1, SequenceExecutorMessageType::LITERAL, CopyOrMatchLength:13, CopyOrMatchContent:0, false),
    test_command(u32:1, SequenceExecutorMessageType::SEQUENCE, CopyOrMatchLength:4, CopyOrMatchContent:1, false),
    // seq #5
    test_command(u32:1, SequenceExecutorMessageType::LITERAL, CopyOrMatchLength:0, CopyOrMatchContent:0, false),
    test_command(u32:1, SequenceExecutorMessageType::SEQUENCE, CopyOrMatchLength:3, CopyOrMatchContent:46, false),
    // seq #6
    test_command(u32:1, SequenceExecutorMessageType::LITERAL, CopyOrMatchLength:0, CopyOrMatchContent:0, false),
    test_command(u32:1, SequenceExecutorMessageType::SEQUENCE, CopyOrMatchLength:6, CopyOrMatchContent:18, true),
];

//#[test_proc]
//proc FseDecoderTest {
//    type FseRamRdReq = ram::ReadReq<TEST_RAM_ADDR_W, TEST_RAM_NUM_PARTITIONS>;
//    type FseRamRdResp = ram::ReadResp<TEST_RAM_DATA_W>;
//
//    type FseRamWrReq = ram::WriteReq<TEST_RAM_ADDR_W, TEST_RAM_DATA_W, TEST_RAM_NUM_PARTITIONS>;
//    type FseRamWrResp = ram::WriteResp;
//
//    type RefillingSBCtrl = refilling_shift_buffer::RefillingShiftBufferCtrl<TEST_REFILLING_SB_LENGTH_W>;
//    type RefillingSBOutput = refilling_shift_buffer::RefillingShiftBufferOutput<TEST_REFILLING_SB_DATA_W, TEST_REFILLING_SB_LENGTH_W>;
//
//    terminator: chan<bool> out;
//
//    ctrl_s: chan<FseDecoderCtrl> out;
//    finish_r: chan<FseDecoderFinish> in;
//
//    rsb_ctrl_r: chan<RefillingSBCtrl> in;
//    rsb_data_s: chan<RefillingSBOutput> out;
//
//    command_r: chan<CommandConstructorData> in;
//
//    ll_fse_wr_req_s: chan<FseRamWrReq> out;
//    ll_fse_wr_resp_r: chan<FseRamWrResp> in;
//
//    ml_fse_wr_req_s: chan<FseRamWrReq> out;
//    ml_fse_wr_resp_r: chan<FseRamWrResp> in;
//
//    of_fse_wr_req_s: chan<FseRamWrReq> out;
//    of_fse_wr_resp_r: chan<FseRamWrResp> in;
//
//    config (terminator: chan<bool> out) {
//        let (ctrl_s, ctrl_r) = chan<FseDecoderCtrl>("ctrl");
//        let (finish_s, finish_r) = chan<FseDecoderFinish>("finish");
//
//        let (rsb_ctrl_s, rsb_ctrl_r) = chan<RefillingSBCtrl>("rsb_ctrl");
//        let (rsb_data_s, rsb_data_r) = chan<RefillingSBOutput>("rsb_out_data");
//
//        let (command_s, command_r) = chan<CommandConstructorData>("command");
//
//        // RAM with FSE lookup for Literal Lengths
//        let (ll_fse_rd_req_s, ll_fse_rd_req_r) = chan<FseRamRdReq>("ll_fse_rd_req");
//        let (ll_fse_rd_resp_s, ll_fse_rd_resp_r) = chan<FseRamRdResp>("ll_fse_rd_resp");
//        let (ll_fse_wr_req_s, ll_fse_wr_req_r) = chan<FseRamWrReq>("ll_fse_wr_req");
//        let (ll_fse_wr_resp_s, ll_fse_wr_resp_r) = chan<FseRamWrResp>("ll_fse_wr_resp");
//
//        spawn ram::RamModel<
//            TEST_RAM_DATA_W,
//            TEST_RAM_SIZE,
//            TEST_RAM_WORD_PARTITION_SIZE,
//        >(ll_fse_rd_req_r, ll_fse_rd_resp_s, ll_fse_wr_req_r, ll_fse_wr_resp_s);
//
//        // RAM with FSE lookup for Match Lengths
//        let (ml_fse_rd_req_s, ml_fse_rd_req_r) = chan<FseRamRdReq>("ml_fse_rd_req");
//        let (ml_fse_rd_resp_s, ml_fse_rd_resp_r) = chan<FseRamRdResp>("ml_fse_rd_resp");
//        let (ml_fse_wr_req_s, ml_fse_wr_req_r) = chan<FseRamWrReq>("ml_fse_wr_req");
//        let (ml_fse_wr_resp_s, ml_fse_wr_resp_r) = chan<FseRamWrResp>("ml_fse_wr_resp");
//
//        spawn ram::RamModel<
//            TEST_RAM_DATA_W,
//            TEST_RAM_SIZE,
//            TEST_RAM_WORD_PARTITION_SIZE,
//        >(ml_fse_rd_req_r, ml_fse_rd_resp_s, ml_fse_wr_req_r, ml_fse_wr_resp_s);
//
//        // RAM with FSE lookup for Offsets
//        let (of_fse_rd_req_s, of_fse_rd_req_r) = chan<FseRamRdReq>("of_fse_rd_req");
//        let (of_fse_rd_resp_s, of_fse_rd_resp_r) = chan<FseRamRdResp>("of_fse_rd_resp");
//        let (of_fse_wr_req_s, of_fse_wr_req_r) = chan<FseRamWrReq>("of_fse_wr_req");
//        let (of_fse_wr_resp_s, of_fse_wr_resp_r) = chan<FseRamWrResp>("of_fse_wr_resp");
//
//        spawn ram::RamModel<
//            TEST_RAM_DATA_W,
//            TEST_RAM_SIZE,
//            TEST_RAM_WORD_PARTITION_SIZE,
//        >(of_fse_rd_req_r, of_fse_rd_resp_s, of_fse_wr_req_r, of_fse_wr_resp_s);
//
//        spawn FseDecoder<
//            TEST_RAM_DATA_W, TEST_RAM_ADDR_W, TEST_RAM_NUM_PARTITIONS,
//            TEST_AXI_DATA_W,
//        >(
//            ctrl_r, finish_s,
//            rsb_ctrl_s, rsb_data_r,
//            command_s,
//            ll_fse_rd_req_s, ll_fse_rd_resp_r,
//            ml_fse_rd_req_s, ml_fse_rd_resp_r,
//            of_fse_rd_req_s, of_fse_rd_resp_r,
//        );
//
//        (
//            terminator,
//            ctrl_s, finish_r,
//            rsb_ctrl_r, rsb_data_s,
//            command_r,
//            ll_fse_wr_req_s, ll_fse_wr_resp_r,
//            ml_fse_wr_req_s, ml_fse_wr_resp_r,
//            of_fse_wr_req_s, of_fse_wr_resp_r,
//        )
//    }
//
//    init { u32:0 }
//
//    next (state: u32) {
//        let tok = join();
//
//        // write OF table
//        let tok = for ((i, of_record), tok): ((u32, u32), token) in enumerate(TEST_OF_TABLE[state]) {
//            let tok = send(tok, of_fse_wr_req_s, FseRamWrReq {
//                addr: i as u8,
//                data: of_record,
//                mask: u4:0xf,
//            });
//            let (tok, _) = recv(tok, of_fse_wr_resp_r);
//            tok
//        }(tok);
//
//        // write ML table
//        let tok = for ((i, ml_record), tok): ((u32, u32), token) in enumerate(TEST_ML_TABLE[state]) {
//            let tok = send(tok, ml_fse_wr_req_s, FseRamWrReq {
//                addr: i as u8,
//                data: ml_record,
//                mask: u4:0xf,
//            });
//            let (tok, _) = recv(tok, ml_fse_wr_resp_r);
//            tok
//        }(tok);
//
//        // write LL table
//        let tok = for ((i, ll_record), tok): ((u32, u32), token) in enumerate(TEST_LL_TABLE[state]) {
//            let tok = send(tok, ll_fse_wr_req_s, FseRamWrReq {
//                addr: i as u8,
//                data: ll_record,
//                mask: u4:0xf,
//            });
//            let (tok, _) = recv(tok, ll_fse_wr_resp_r);
//            tok
//        }(tok);
//
//        // send ctrl
//        let tok = send(tok, ctrl_s, TEST_CTRL[state]);
//        trace_fmt!("Sent ctrl {:#x}", TEST_CTRL[state]);
//
//        match state {
//            u32:0 => {
//                // block #0
//                // send data
//                let tok = for ((i, data), tok): ((u32, RefillingSBOutput), token) in enumerate(TEST_DATA_0) {
//                    let (tok, buf_ctrl) = recv(tok, rsb_ctrl_r);
//                    trace_fmt!("Received #{} buf ctrl {:#x}", i + u32:1, buf_ctrl);
//                    assert_eq(RefillingSBCtrl {length: data.length}, buf_ctrl);
//                    let tok = send(tok, rsb_data_s, data);
//                    trace_fmt!("Sent #{} buf data {:#x}", i + u32:1, data);
//                    tok
//                }(tok);
//
//                // recv commands
//                let tok = for ((i, expected_cmd), tok): ((u32, CommandConstructorData), token) in enumerate(TEST_EXPECTED_COMMANDS_0) {
//                    let (tok, cmd) = recv(tok, command_r);
//                    trace_fmt!("Received #{} cmd {:#x}", i + u32:1, cmd);
//                    assert_eq(expected_cmd, cmd);
//                    tok
//                }(tok);
//
//                // recv finish
//                let (tok, _) = recv(tok, finish_r);
//            },
//            u32:1 => {
//                // block #1
//                // send data
//                let tok = for ((i, data), tok): ((u32, RefillingSBOutput), token) in enumerate(TEST_DATA_1) {
//                    let (tok, buf_ctrl) = recv(tok, rsb_ctrl_r);
//                    trace_fmt!("Received #{} buf ctrl {:#x}", i + u32:1, buf_ctrl);
//                    assert_eq(RefillingSBCtrl {length: data.length}, buf_ctrl);
//                    let tok = send(tok, rsb_data_s, data);
//                    trace_fmt!("Sent #{} buf data {:#x}", i + u32:1, data);
//                    tok
//                }(tok);
//
//                // recv commands
//                let tok = for ((i, expected_cmd), tok): ((u32, CommandConstructorData), token) in enumerate(TEST_EXPECTED_COMMANDS_1) {
//                    let (tok, cmd) = recv(tok, command_r);
//                    trace_fmt!("Received #{} cmd {:#x}", i + u32:1, cmd);
//                    assert_eq(expected_cmd, cmd);
//                    tok
//                }(tok);
//
//                // recv finish
//                let (tok, _) = recv(tok, finish_r);
//
//                send(tok, terminator, true);
//            },
//        };
//
//        state + u32:1
//    }
//}
