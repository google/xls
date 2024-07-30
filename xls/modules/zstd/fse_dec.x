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
import xls.modules.zstd.shift_buffer;

type FseRamReadReq = common::SeqDecFseRamReadReq;
type FseRamReadResp = common::SeqDecFseRamReadResp;
type FseRamWriteReq = common::SeqDecFseRamWriteReq;
type FseRamWriteResp = common::SeqDecFseRamWriteResp;
type FseTableRecord = common::FseTableRecord;

type ShiftBufferCtrl = common::SeqDecShiftBufferCtrl;
type ShiftBufferInput = common::SeqDecShiftBufferInput;
type ShiftBufferOutput = common::SeqDecShiftBufferOutput;

type BlockSyncData = common::BlockSyncData;
type SequenceExecutorMessageType = common::SequenceExecutorMessageType;
type SequenceExecutorPacket = common::SequenceExecutorPacket;
type CommandConstructorData = common::CommandConstructorData;

type CopyOrMatchLength = common::CopyOrMatchLength;
type CopyOrMatchContent = common::CopyOrMatchContent;

pub struct FseDecoderCtrl {
    sync: BlockSyncData,
    sequences_count: u24,
    of_acc_log: u7,
    ll_acc_log: u7,
    ml_acc_log: u7,
}
pub struct FseDecoderFinish { }

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

enum FseDecoderFSM : u4 {
    RECV_CTRL = 0,
    INIT_OF_STATE = 1,
    INIT_ML_STATE = 2,
    INIT_LL_STATE = 3,
    SEND_RAM_RD_REQ = 4,
    RECV_RAM_RD_RESP = 5,
    READ_OF_BITS = 6,
    READ_ML_BITS = 7,
    READ_LL_BITS = 8,
    UPDATE_OF_STATE = 9,
    UPDATE_ML_STATE = 10,
    UPDATE_LL_STATE = 11,
    SEND_COMMAND_LITERAL = 12,
    SEND_COMMAND_SEQUENCE = 13,
    SEND_FINISH = 14,
}

struct FseDecoderState {
    fsm: FseDecoderFSM,
    ctrl: FseDecoderCtrl,
    sequences_count: u24,
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
}

proc FseDecoder {
    // control
    ctrl_r: chan<FseDecoderCtrl> in;
    finish_s: chan<FseDecoderFinish> out;

    // shift buffer
    shift_buffer_ctrl_s: chan<ShiftBufferCtrl> out;
    shift_buffer_out_data_r: chan<ShiftBufferOutput> in;
    shift_buffer_out_ctrl_r: chan<ShiftBufferOutput> in;

    // output command
    command_s: chan<CommandConstructorData> out;

    // RAMs
    ll_def_fse_rd_req_s: chan<FseRamReadReq> out;
    ll_def_fse_rd_resp_r: chan<FseRamReadResp> in;

    ll_fse_rd_req_s: chan<FseRamReadReq> out;
    ll_fse_rd_resp_r: chan<FseRamReadResp> in;

    ml_def_fse_rd_req_s: chan<FseRamReadReq> out;
    ml_def_fse_rd_resp_r: chan<FseRamReadResp> in;

    ml_fse_rd_req_s: chan<FseRamReadReq> out;
    ml_fse_rd_resp_r: chan<FseRamReadResp> in;

    of_def_fse_rd_req_s: chan<FseRamReadReq> out;
    of_def_fse_rd_resp_r: chan<FseRamReadResp> in;

    of_fse_rd_req_s: chan<FseRamReadReq> out;
    of_fse_rd_resp_r: chan<FseRamReadResp> in;

    config (
        ctrl_r: chan<FseDecoderCtrl> in,
        finish_s: chan<FseDecoderFinish> out,
        shift_buffer_ctrl_s: chan<ShiftBufferCtrl> out,
        shift_buffer_out_data_r: chan<ShiftBufferOutput> in,
        shift_buffer_out_ctrl_r: chan<ShiftBufferOutput> in,
        command_s: chan<CommandConstructorData> out,
        ll_def_fse_rd_req_s: chan<FseRamReadReq> out,
        ll_def_fse_rd_resp_r: chan<FseRamReadResp> in,
        ll_fse_rd_req_s: chan<FseRamReadReq> out,
        ll_fse_rd_resp_r: chan<FseRamReadResp> in,
        ml_def_fse_rd_req_s: chan<FseRamReadReq> out,
        ml_def_fse_rd_resp_r: chan<FseRamReadResp> in,
        ml_fse_rd_req_s: chan<FseRamReadReq> out,
        ml_fse_rd_resp_r: chan<FseRamReadResp> in,
        of_def_fse_rd_req_s: chan<FseRamReadReq> out,
        of_def_fse_rd_resp_r: chan<FseRamReadResp> in,
        of_fse_rd_req_s: chan<FseRamReadReq> out,
        of_fse_rd_resp_r: chan<FseRamReadResp> in,
    ) {
        (
            ctrl_r, finish_s,
            shift_buffer_ctrl_s, shift_buffer_out_data_r, shift_buffer_out_ctrl_r,
            command_s,
            ll_def_fse_rd_req_s, ll_def_fse_rd_resp_r, ll_fse_rd_req_s, ll_fse_rd_resp_r,
            ml_def_fse_rd_req_s, ml_def_fse_rd_resp_r, ml_fse_rd_req_s, ml_fse_rd_resp_r,
            of_def_fse_rd_req_s, of_def_fse_rd_resp_r, of_fse_rd_req_s, of_fse_rd_resp_r,
        )
    }

    init { zero!<FseDecoderState>() }

    next (state: FseDecoderState) {
        type RamAddr = uN[common::SEQDEC_FSE_RAM_ADDR_WIDTH];
        const RAM_MASK_ALL = std::unsigned_max_value<common::SEQDEC_FSE_RAM_NUM_PARTITIONS>();

        let tok0 = join();

        // receive ctrl
        let (_, ctrl, ctrl_valid) = recv_if_non_blocking(tok0, ctrl_r, state.fsm == FseDecoderFSM::RECV_CTRL, zero!<FseDecoderCtrl>());

        let state = if ctrl_valid {
            FseDecoderState {
                ctrl: ctrl,
                sequences_count: ctrl.sequences_count,
                ..state
            }
        } else { state };

        // receive ram read response
        let (_, ll_rd_resp, ll_rd_resp_valid) = recv_if_non_blocking(tok0, ll_def_fse_rd_resp_r, state.fsm == FseDecoderFSM::RECV_RAM_RD_RESP, zero!<FseRamReadResp>());
        let (_, ml_rd_resp, ml_rd_resp_valid) = recv_if_non_blocking(tok0, ml_def_fse_rd_resp_r, state.fsm == FseDecoderFSM::RECV_RAM_RD_RESP, zero!<FseRamReadResp>());
        let (_, of_rd_resp, of_rd_resp_valid) = recv_if_non_blocking(tok0, of_def_fse_rd_resp_r, state.fsm == FseDecoderFSM::RECV_RAM_RD_RESP, zero!<FseRamReadResp>());

        let (_, _, _) = recv_if_non_blocking(tok0, ll_fse_rd_resp_r, false, zero!<FseRamReadResp>());
        let (_, _, _) = recv_if_non_blocking(tok0, ml_fse_rd_resp_r, false, zero!<FseRamReadResp>());
        let (_, _, _) = recv_if_non_blocking(tok0, of_fse_rd_resp_r, false, zero!<FseRamReadResp>());

        let ll_fse_table_record = FseTableRecord {
            symbol: ll_rd_resp.data[24:32],
            num_of_bits: ll_rd_resp.data[16:24],
            base: ll_rd_resp.data[0:16],
        };
        let ml_fse_table_record = FseTableRecord {
            symbol: ml_rd_resp.data[24:32],
            num_of_bits: ml_rd_resp.data[16:24],
            base: ml_rd_resp.data[0:16],
        };
        let of_fse_table_record = FseTableRecord {
            symbol: of_rd_resp.data[24:32],
            num_of_bits: of_rd_resp.data[16:24],
            base: of_rd_resp.data[0:16],
        };

        // validate LL and ML symbols
        assert!(!(ll_rd_resp_valid && ll_fse_table_record.symbol > SEQ_MAX_CODES_LL), "invalid_literal_length_symbol");
        assert!(!(ml_rd_resp_valid && ml_fse_table_record.symbol > SEQ_MAX_CODES_ML), "invalid_match_length_symbol");

        // save fse records in state
        let state = if state.fsm == FseDecoderFSM::RECV_RAM_RD_RESP {
            let state = if ll_rd_resp_valid {
                FseDecoderState { ll_fse_table_record: ll_fse_table_record, ll_fse_table_record_valid: true, ..state }
            } else { state };
            let state = if ml_rd_resp_valid {
                FseDecoderState { ml_fse_table_record: ml_fse_table_record, ml_fse_table_record_valid: true, ..state }
            } else { state };
            let state = if of_rd_resp_valid {
                FseDecoderState { of_fse_table_record: of_fse_table_record, of_fse_table_record_valid: true, ..state }
            } else { state };
            state
        } else { state };

        // request records
        let do_send_ram_rd_req = state.fsm == FseDecoderFSM::SEND_RAM_RD_REQ;

        send_if(tok0, ll_def_fse_rd_req_s, do_send_ram_rd_req, FseRamReadReq { addr: state.ll_state as RamAddr, mask: RAM_MASK_ALL});
        send_if(tok0, ml_def_fse_rd_req_s, do_send_ram_rd_req, FseRamReadReq { addr: state.ml_state as RamAddr, mask: RAM_MASK_ALL});
        send_if(tok0, of_def_fse_rd_req_s, do_send_ram_rd_req, FseRamReadReq { addr: state.of_state as RamAddr, mask: RAM_MASK_ALL});

        send_if(tok0, ll_fse_rd_req_s, false, FseRamReadReq { addr: state.ll_state as RamAddr, mask: RAM_MASK_ALL });
        send_if(tok0, ml_fse_rd_req_s, false, FseRamReadReq { addr: state.ml_state as RamAddr, mask: RAM_MASK_ALL });
        send_if(tok0, of_fse_rd_req_s, false, FseRamReadReq { addr: state.of_state as RamAddr, mask: RAM_MASK_ALL });

        // read bits
        let do_read_bits = (
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

        let buf_ctrl_length = if ((state.read_bits_needed - state.read_bits_length) > common::SEQDEC_SHIFT_BUFFER_DATA_WIDTH as u7) {
            common::SEQDEC_SHIFT_BUFFER_DATA_WIDTH as u7
        } else {
            state.read_bits_needed - state.read_bits_length
        };

        send_if(tok0, shift_buffer_ctrl_s, do_send_buf_ctrl, ShiftBufferCtrl {
            length: buf_ctrl_length,
            output: shift_buffer::ShiftBufferOutputType::DATA,
        });

        let state = if do_send_buf_ctrl {
            FseDecoderState { sent_buf_ctrl: do_send_buf_ctrl, ..state }
        } else { state };

        let (_, buf_data, buf_data_valid) = recv_if_non_blocking(tok0, shift_buffer_out_data_r, do_read_bits && state.sent_buf_ctrl, zero!<ShiftBufferOutput>());
        let (_, _, _) = recv_if_non_blocking(tok0, shift_buffer_out_ctrl_r, do_read_bits && state.sent_buf_ctrl, zero!<ShiftBufferOutput>());

        let state = if do_read_bits & buf_data_valid {
            FseDecoderState {
                sent_buf_ctrl: false,
                read_bits: (buf_data.data as u16 << state.read_bits_length) | state.read_bits,
                read_bits_length: state.read_bits_length + buf_data.length,
                ..state
            }
        } else { state };

        // send command
        let command_data = if state.fsm == FseDecoderFSM::SEND_COMMAND_LITERAL {
            SequenceExecutorPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                length: state.ll,
                content: CopyOrMatchContent:0,
                last: false,
            }
        } else {
            SequenceExecutorPacket {
                msg_type: SequenceExecutorMessageType::SEQUENCE,
                length: state.ml,
                content: state.of,
                last: state.sequences_count == u24:1,
            }
        };
        let do_send_command = state.fsm == FseDecoderFSM::SEND_COMMAND_LITERAL || state.fsm == FseDecoderFSM::SEND_COMMAND_SEQUENCE;
        send_if(tok0, command_s, do_send_command, CommandConstructorData {
            sync: state.ctrl.sync,
            data: command_data,
        });

        // send finish
        send_if(tok0, finish_s, state.fsm == FseDecoderFSM::SEND_FINISH, FseDecoderFinish{});

        // update state
        match (state.fsm) {
            FseDecoderFSM::RECV_CTRL => {
                if (ctrl_valid) {
                    FseDecoderState {
                        fsm: FseDecoderFSM::INIT_LL_STATE,
                        ctrl: ctrl,
                        read_bits: u16:0,
                        read_bits_length: u7:0,
                        read_bits_needed: state.ctrl.ll_acc_log,
                        ..state
                    }
                } else { state }
            },
            FseDecoderFSM::INIT_LL_STATE => {
                if (state.read_bits_needed == state.read_bits_length && !state.sent_buf_ctrl) {
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
                if (state.ll_fse_table_record_valid &&
                    state.ml_fse_table_record_valid &&
                    state.of_fse_table_record_valid
                ) {
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
                    FseDecoderState {
                        fsm: FseDecoderFSM::READ_ML_BITS,
                        of: ((u32:1 << state.of_fse_table_record.symbol) + state.read_bits as u32) as u64,
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
                FseDecoderState {
                    fsm: FseDecoderFSM::SEND_COMMAND_SEQUENCE,
                    ..state
                }
            },
            FseDecoderFSM::SEND_COMMAND_SEQUENCE => {
                if (state.sequences_count == u24:1) {
                    FseDecoderState {
                        fsm: FseDecoderFSM::SEND_FINISH,
                        sequences_count: u24:0,
                        ..state
                    }
                } else {
                    FseDecoderState {
                        fsm: FseDecoderFSM::SEND_RAM_RD_REQ,
                        sequences_count: state.sequences_count - u24:1,
                        ..state
                    }
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
        of_acc_log: u7:5,
        ll_acc_log: u7:5,
        ml_acc_log: u7:5,
    },
    FseDecoderCtrl {
        sync: TEST_SYNC[1],
        sequences_count: u24:7,
        of_acc_log: u7:5,
        ll_acc_log: u7:5,
        ml_acc_log: u7:6,
    },
];

const TEST_DATA_0 = ShiftBufferOutput[48]:[
    // init states
    ShiftBufferOutput {data: u64:0b11111, length: u7:5, last: false},
    ShiftBufferOutput {data: u64:0b101, length: u7:5, last: false},
    ShiftBufferOutput {data: u64:0b10, length: u7:5, last: false},
    // symbols (seq #0)
    ShiftBufferOutput {data: u64:0b0, length: u7:0, last: false},
    ShiftBufferOutput {data: u64:0b0, length: u7:0, last: false},
    ShiftBufferOutput {data: u64:0b0, length: u7:0, last: false},
    // state update
    ShiftBufferOutput {data: u64:0b0, length: u7:1, last: false},
    ShiftBufferOutput {data: u64:0b0, length: u7:1, last: false},
    ShiftBufferOutput {data: u64:0b100, length: u7:3, last: false},
    // symbols (seq #1)
    ShiftBufferOutput {data: u64:0b10, length: u7:2, last: false},
    ShiftBufferOutput {data: u64:0b0, length: u7:0, last: false},
    ShiftBufferOutput {data: u64:0b0, length: u7:0, last: false},
    // state update
    ShiftBufferOutput {data: u64:0b0, length: u7:1, last: false},
    ShiftBufferOutput {data: u64:0b110, length: u7:3, last: false},
    ShiftBufferOutput {data: u64:0b10, length: u7:2, last: false},
    // symbols (seq #2)
    ShiftBufferOutput {data: u64:0b0, length: u7:3, last: false},
    ShiftBufferOutput {data: u64:0b0, length: u7:0, last: false},
    ShiftBufferOutput {data: u64:0b0, length: u7:0, last: false},
    // state update
    ShiftBufferOutput {data: u64:0b10, length: u7:2, last: false},
    ShiftBufferOutput {data: u64:0b0, length: u7:0, last: false},
    ShiftBufferOutput {data: u64:0b0, length: u7:1, last: false},
    // symbols (seq #3)
    ShiftBufferOutput {data: u64:0b11, length: u7:3, last: false},
    ShiftBufferOutput {data: u64:0b0, length: u7:0, last: false},
    ShiftBufferOutput {data: u64:0b0, length: u7:0, last: false},
    // state update
    ShiftBufferOutput {data: u64:0b0, length: u7:0, last: false},
    ShiftBufferOutput {data: u64:0b0, length: u7:1, last: false},
    ShiftBufferOutput {data: u64:0b0, length: u7:1, last: false},
    // symbols (seq #4)
    ShiftBufferOutput {data: u64:0b0, length: u7:4, last: false},
    ShiftBufferOutput {data: u64:0b0, length: u7:0, last: false},
    ShiftBufferOutput {data: u64:0b0, length: u7:0, last: false},
    // state update
    ShiftBufferOutput {data: u64:0b1, length: u7:1, last: false},
    ShiftBufferOutput {data: u64:0b0, length: u7:0, last: false},
    ShiftBufferOutput {data: u64:0b0, length: u7:3, last: false},
    // symbols (seq #5)
    ShiftBufferOutput {data: u64:0b101, length: u7:3, last: false},
    ShiftBufferOutput {data: u64:0b0, length: u7:0, last: false},
    ShiftBufferOutput {data: u64:0b0, length: u7:0, last: false},
    // state update
    ShiftBufferOutput {data: u64:0b0, length: u7:0, last: false},
    ShiftBufferOutput {data: u64:0b0, length: u7:1, last: false},
    ShiftBufferOutput {data: u64:0b1, length: u7:1, last: false},
    // symbols (seq #6)
    ShiftBufferOutput {data: u64:0b11, length: u7:2, last: false},
    ShiftBufferOutput {data: u64:0b0, length: u7:0, last: false},
    ShiftBufferOutput {data: u64:0b0, length: u7:0, last: false},
    // state update
    ShiftBufferOutput {data: u64:0b0, length: u7:0, last: false},
    ShiftBufferOutput {data: u64:0b0, length: u7:2, last: false},
    ShiftBufferOutput {data: u64:0b0, length: u7:2, last: false},
    // symbols (seq #7)
    ShiftBufferOutput {data: u64:0b1000, length: u7:4, last: false},
    ShiftBufferOutput {data: u64:0b0, length: u7:0, last: false},
    ShiftBufferOutput {data: u64:0b0, length: u7:0, last: true},
    // no state update for last sequence
];

const TEST_DATA_1 = ShiftBufferOutput[42]:[
    // init states
    ShiftBufferOutput {data: u64:0b10000, length: u7:5, last: false},
    ShiftBufferOutput {data: u64:0b1110, length: u7:5, last: false},
    ShiftBufferOutput {data: u64:0b11001, length: u7:6, last: false},
    // symbols (seq #0)
    ShiftBufferOutput {data: u64:0b0, length: u7:2, last: false},
    ShiftBufferOutput {data: u64:0b0, length: u7:0, last: false},
    ShiftBufferOutput {data: u64:0b0, length: u7:0, last: false},
    // state update
    ShiftBufferOutput {data: u64:0b110, length: u7:3, last: false},
    ShiftBufferOutput {data: u64:0b0, length: u7:5, last: false},
    ShiftBufferOutput {data: u64:0b1110, length: u7:5, last: false},
    // symbols (seq #1)
    ShiftBufferOutput {data: u64:0b10, length: u7:2, last: false},
    ShiftBufferOutput {data: u64:0b0, length: u7:0, last: false},
    ShiftBufferOutput {data: u64:0b0, length: u7:0, last: false},
    // state update
    ShiftBufferOutput {data: u64:0b10, length: u7:2, last: false},
    ShiftBufferOutput {data: u64:0b1, length: u7:6, last: false},
    ShiftBufferOutput {data: u64:0b101, length: u7:5, last: false},
    // symbols (seq #2)
    ShiftBufferOutput {data: u64:0b110, length: u7:3, last: false},
    ShiftBufferOutput {data: u64:0b0, length: u7:0, last: false},
    ShiftBufferOutput {data: u64:0b0, length: u7:0, last: false},
    // state update
    ShiftBufferOutput {data: u64:0b11, length: u7:2, last: false},
    ShiftBufferOutput {data: u64:0b1, length: u7:4, last: false},
    ShiftBufferOutput {data: u64:0b10011, length: u7:5, last: false},
    // symbols (seq #3)
    ShiftBufferOutput {data: u64:0b11, length: u7:4, last: false},
    ShiftBufferOutput {data: u64:0b0, length: u7:0, last: false},
    ShiftBufferOutput {data: u64:0b0, length: u7:0, last: false},
    // state update
    ShiftBufferOutput {data: u64:0b0, length: u7:2, last: false},
    ShiftBufferOutput {data: u64:0b1, length: u7:4, last: false},
    ShiftBufferOutput {data: u64:0b0, length: u7:5, last: false},
    // symbols (seq #4)
    ShiftBufferOutput {data: u64:0b0, length: u7:0, last: false},
    ShiftBufferOutput {data: u64:0b0, length: u7:0, last: false},
    ShiftBufferOutput {data: u64:0b0, length: u7:0, last: false},
    // state update
    ShiftBufferOutput {data: u64:0b10, length: u7:3, last: false},
    ShiftBufferOutput {data: u64:0b0, length: u7:4, last: false},
    ShiftBufferOutput {data: u64:0b1010, length: u7:5, last: false},
    // symbols (seq #5)
    ShiftBufferOutput {data: u64:0b1110, length: u7:5, last: false},
    ShiftBufferOutput {data: u64:0b0, length: u7:0, last: false},
    ShiftBufferOutput {data: u64:0b0, length: u7:0, last: false},
    // state update
    ShiftBufferOutput {data: u64:0b0, length: u7:1, last: false},
    ShiftBufferOutput {data: u64:0b11, length: u7:6, last: false},
    ShiftBufferOutput {data: u64:0b10011, length: u7:5, last: false},
    // symbols (seq #6)
    ShiftBufferOutput {data: u64:0b10, length: u7:4, last: false},
    ShiftBufferOutput {data: u64:0b0, length: u7:0, last: false},
    ShiftBufferOutput {data: u64:0b0, length: u7:0, last: true},
    // no state update for last sequence
];

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

#[test_proc]
proc FseDecoderTest {
    terminator: chan<bool> out;

    ctrl_s: chan<FseDecoderCtrl> out;
    finish_r: chan<FseDecoderFinish> in;

    shift_buffer_ctrl_r: chan<ShiftBufferCtrl> in;
    shift_buffer_out_data_s: chan<ShiftBufferOutput> out;
    shift_buffer_out_ctrl_s: chan<ShiftBufferOutput> out;
    
    command_r: chan<CommandConstructorData> in;

    ll_def_fse_wr_req_s: chan<FseRamWriteReq> out;
    ll_def_fse_wr_resp_r: chan<FseRamWriteResp> in;

    ll_fse_wr_req_s: chan<FseRamWriteReq> out;
    ll_fse_wr_resp_r: chan<FseRamWriteResp> in;

    ml_def_fse_wr_req_s: chan<FseRamWriteReq> out;
    ml_def_fse_wr_resp_r: chan<FseRamWriteResp> in;

    ml_fse_wr_req_s: chan<FseRamWriteReq> out;
    ml_fse_wr_resp_r: chan<FseRamWriteResp> in;

    of_def_fse_wr_req_s: chan<FseRamWriteReq> out;
    of_def_fse_wr_resp_r: chan<FseRamWriteResp> in;

    of_fse_wr_req_s: chan<FseRamWriteReq> out;
    of_fse_wr_resp_r: chan<FseRamWriteResp> in;

    config (terminator: chan<bool> out) {
        let (ctrl_s, ctrl_r) = chan<FseDecoderCtrl>("ctrl");
        let (finish_s, finish_r) = chan<FseDecoderFinish>("finish");

        let (shift_buffer_ctrl_s, shift_buffer_ctrl_r) = chan<ShiftBufferCtrl>("shift_buffer_ctrl");
        let (shift_buffer_out_data_s, shift_buffer_out_data_r) = chan<ShiftBufferOutput>("shift_buffer_out_data");
        let (shift_buffer_out_ctrl_s, shift_buffer_out_ctrl_r) = chan<ShiftBufferOutput>("shift_buffer_out_ctrl");

        let (command_s, command_r) = chan<CommandConstructorData>("command");

        // RAM with default FSE lookup for Literal Lengths
        let (ll_def_fse_rd_req_s, ll_def_fse_rd_req_r) = chan<FseRamReadReq>("ll_def_fse_rd_req");
        let (ll_def_fse_rd_resp_s, ll_def_fse_rd_resp_r) = chan<FseRamReadResp>("ll_def_fse_rd_resp");
        let (ll_def_fse_wr_req_s, ll_def_fse_wr_req_r) = chan<FseRamWriteReq>("ll_def_fse_wr_req");
        let (ll_def_fse_wr_resp_s, ll_def_fse_wr_resp_r) = chan<FseRamWriteResp>("ll_def_fse_wr_resp");

        spawn ram::RamModel<
            common::SEQDEC_FSE_RAM_DATA_WIDTH,
            common::SEQDEC_FSE_RAM_SIZE,
            common::SEQDEC_FSE_RAM_WORD_PARTITION_SIZE
        >(ll_def_fse_rd_req_r, ll_def_fse_rd_resp_s, ll_def_fse_wr_req_r, ll_def_fse_wr_resp_s);

        // RAM for FSE lookup for Literal Lengths
        let (ll_fse_rd_req_s, ll_fse_rd_req_r) = chan<FseRamReadReq>("ll_fse_rd_req");
        let (ll_fse_rd_resp_s, ll_fse_rd_resp_r) = chan<FseRamReadResp>("ll_fse_rd_resp");
        let (ll_fse_wr_req_s, ll_fse_wr_req_r) = chan<FseRamWriteReq>("ll_fse_wr_req");
        let (ll_fse_wr_resp_s, ll_fse_wr_resp_r) = chan<FseRamWriteResp>("ll_fse_wr_resp");

        spawn ram::RamModel<
            common::SEQDEC_FSE_RAM_DATA_WIDTH,
            common::SEQDEC_FSE_RAM_SIZE,
            common::SEQDEC_FSE_RAM_WORD_PARTITION_SIZE
        >(ll_fse_rd_req_r, ll_fse_rd_resp_s, ll_fse_wr_req_r, ll_fse_wr_resp_s);

        // RAM with default FSE lookup for Match Lengths
        let (ml_def_fse_rd_req_s, ml_def_fse_rd_req_r) = chan<FseRamReadReq>("ml_def_fse_rd_req");
        let (ml_def_fse_rd_resp_s, ml_def_fse_rd_resp_r) = chan<FseRamReadResp>("ml_def_fse_rd_resp");
        let (ml_def_fse_wr_req_s, ml_def_fse_wr_req_r) = chan<FseRamWriteReq>("ml_def_fse_wr_req");
        let (ml_def_fse_wr_resp_s, ml_def_fse_wr_resp_r) = chan<FseRamWriteResp>("ml_def_fse_wr_resp");

        spawn ram::RamModel<
            common::SEQDEC_FSE_RAM_DATA_WIDTH,
            common::SEQDEC_FSE_RAM_SIZE,
            common::SEQDEC_FSE_RAM_WORD_PARTITION_SIZE
        >(ml_def_fse_rd_req_r, ml_def_fse_rd_resp_s, ml_def_fse_wr_req_r, ml_def_fse_wr_resp_s);

        // RAM for FSE lookup for Match Lengths
        let (ml_fse_rd_req_s, ml_fse_rd_req_r) = chan<FseRamReadReq>("ml_fse_rd_req");
        let (ml_fse_rd_resp_s, ml_fse_rd_resp_r) = chan<FseRamReadResp>("ml_fse_rd_resp");
        let (ml_fse_wr_req_s, ml_fse_wr_req_r) = chan<FseRamWriteReq>("ml_fse_wr_req");
        let (ml_fse_wr_resp_s, ml_fse_wr_resp_r) = chan<FseRamWriteResp>("ml_fse_wr_resp");

        spawn ram::RamModel<
            common::SEQDEC_FSE_RAM_DATA_WIDTH,
            common::SEQDEC_FSE_RAM_SIZE,
            common::SEQDEC_FSE_RAM_WORD_PARTITION_SIZE
        >(ml_fse_rd_req_r, ml_fse_rd_resp_s, ml_fse_wr_req_r, ml_fse_wr_resp_s);

        // RAM with default FSE lookup for Offsets
        let (of_def_fse_rd_req_s, of_def_fse_rd_req_r) = chan<FseRamReadReq>("of_def_fse_rd_req");
        let (of_def_fse_rd_resp_s, of_def_fse_rd_resp_r) = chan<FseRamReadResp>("of_def_fse_rd_resp");
        let (of_def_fse_wr_req_s, of_def_fse_wr_req_r) = chan<FseRamWriteReq>("of_def_fse_wr_req");
        let (of_def_fse_wr_resp_s, of_def_fse_wr_resp_r) = chan<FseRamWriteResp>("of_def_fse_wr_resp");

        spawn ram::RamModel<
            common::SEQDEC_FSE_RAM_DATA_WIDTH,
            common::SEQDEC_FSE_RAM_SIZE,
            common::SEQDEC_FSE_RAM_WORD_PARTITION_SIZE
        >(of_def_fse_rd_req_r, of_def_fse_rd_resp_s, of_def_fse_wr_req_r, of_def_fse_wr_resp_s);

        // RAM for FSE lookup for Offsets
        let (of_fse_rd_req_s, of_fse_rd_req_r) = chan<FseRamReadReq>("of_fse_rd_req");
        let (of_fse_rd_resp_s, of_fse_rd_resp_r) = chan<FseRamReadResp>("of_fse_rd_resp");
        let (of_fse_wr_req_s, of_fse_wr_req_r) = chan<FseRamWriteReq>("of_fse_wr_req");
        let (of_fse_wr_resp_s, of_fse_wr_resp_r) = chan<FseRamWriteResp>("of_fse_wr_resp");

        spawn ram::RamModel<
            common::SEQDEC_FSE_RAM_DATA_WIDTH,
            common::SEQDEC_FSE_RAM_SIZE,
            common::SEQDEC_FSE_RAM_WORD_PARTITION_SIZE
        >(of_fse_rd_req_r, of_fse_rd_resp_s, of_fse_wr_req_r, of_fse_wr_resp_s);

        spawn FseDecoder (
            ctrl_r, finish_s,
            shift_buffer_ctrl_s, shift_buffer_out_data_r, shift_buffer_out_ctrl_r,
            command_s,
            ll_def_fse_rd_req_s, ll_def_fse_rd_resp_r,
            ll_fse_rd_req_s, ll_fse_rd_resp_r,
            ml_def_fse_rd_req_s, ml_def_fse_rd_resp_r,
            ml_fse_rd_req_s, ml_fse_rd_resp_r,
            of_def_fse_rd_req_s, of_def_fse_rd_resp_r,
            of_fse_rd_req_s, of_fse_rd_resp_r,
        );

        (
            terminator,
            ctrl_s, finish_r,
            shift_buffer_ctrl_r, shift_buffer_out_data_s, shift_buffer_out_ctrl_s,
            command_r,
            ll_def_fse_wr_req_s, ll_def_fse_wr_resp_r, ll_fse_wr_req_s, ll_fse_wr_resp_r,
            ml_def_fse_wr_req_s, ml_def_fse_wr_resp_r, ml_fse_wr_req_s, ml_fse_wr_resp_r,
            of_def_fse_wr_req_s, of_def_fse_wr_resp_r, of_fse_wr_req_s, of_fse_wr_resp_r,
        )
    }

    init {}

    next (state: ()) {
        let tok = join();

        // block #0
        // write OF table
        let tok = for ((i, of_record), tok): ((u32, u32), token) in enumerate(TEST_OF_TABLE[0]) {
            let tok = send(tok, of_def_fse_wr_req_s, FseRamWriteReq {
                addr: i as u8,
                data: of_record,
                mask: u4:0xf,
            });
            let (tok, _) = recv(tok, of_def_fse_wr_resp_r);
            tok
        }(tok);

        // write ML table
        let tok = for ((i, ml_record), tok): ((u32, u32), token) in enumerate(TEST_ML_TABLE[0]) {
            let tok = send(tok, ml_def_fse_wr_req_s, FseRamWriteReq {
                addr: i as u8,
                data: ml_record,
                mask: u4:0xf,
            });
            let (tok, _) = recv(tok, ml_def_fse_wr_resp_r);
            tok
        }(tok);

        // write LL table
        let tok = for ((i, ll_record), tok): ((u32, u32), token) in enumerate(TEST_LL_TABLE[0]) {
            let tok = send(tok, ll_def_fse_wr_req_s, FseRamWriteReq {
                addr: i as u8,
                data: ll_record,
                mask: u4:0xf,
            });
            let (tok, _) = recv(tok, ll_def_fse_wr_resp_r);
            tok
        }(tok);

        // send ctrl
        let tok = send(tok, ctrl_s, TEST_CTRL[0]);
        trace_fmt!("Sent ctrl {:#x}", TEST_CTRL[0]);

        // send data
        let tok = for ((i, data), tok): ((u32, ShiftBufferOutput), token) in enumerate(TEST_DATA_0) {
            let (tok, buf_ctrl) = recv(tok, shift_buffer_ctrl_r);
            trace_fmt!("Received #{} buf ctrl {:#x}", i + u32:1, buf_ctrl);
            assert_eq(ShiftBufferCtrl {length: data.length, output: shift_buffer::ShiftBufferOutputType::DATA}, buf_ctrl);
            let tok = send(tok, shift_buffer_out_data_s, data);
            trace_fmt!("Sent #{} buf data {:#x}", i + u32:1, data);
            tok
        }(tok);

        // recv commands
        let tok = for ((i, expected_cmd), tok): ((u32, CommandConstructorData), token) in enumerate(TEST_EXPECTED_COMMANDS_0) {
            let (tok, cmd) = recv(tok, command_r);
            trace_fmt!("Received #{} cmd {:#x}", i + u32:1, cmd);
            assert_eq(expected_cmd, cmd);
            tok
        }(tok);

        // recv finish
        let (tok, _) = recv(tok, finish_r);

        // block #1
        // write OF table
        let tok = for ((i, of_record), tok): ((u32, u32), token) in enumerate(TEST_OF_TABLE[1]) {
            let tok = send(tok, of_def_fse_wr_req_s, FseRamWriteReq {
                addr: i as u8,
                data: of_record,
                mask: u4:0xf,
            });
            let (tok, _) = recv(tok, of_def_fse_wr_resp_r);
            tok
        }(tok);

        // write ML table
        let tok = for ((i, ml_record), tok): ((u32, u32), token) in enumerate(TEST_ML_TABLE[1]) {
            let tok = send(tok, ml_def_fse_wr_req_s, FseRamWriteReq {
                addr: i as u8,
                data: ml_record,
                mask: u4:0xf,
            });
            let (tok, _) = recv(tok, ml_def_fse_wr_resp_r);
            tok
        }(tok);

        // write LL table
        let tok = for ((i, ll_record), tok): ((u32, u32), token) in enumerate(TEST_LL_TABLE[1]) {
            let tok = send(tok, ll_def_fse_wr_req_s, FseRamWriteReq {
                addr: i as u8,
                data: ll_record,
                mask: u4:0xf,
            });
            let (tok, _) = recv(tok, ll_def_fse_wr_resp_r);
            tok
        }(tok);

        // send ctrl
        let tok = send(tok, ctrl_s, TEST_CTRL[1]);
        trace_fmt!("Sent ctrl {:#x}", TEST_CTRL[1]);

        // send data
        let tok = for ((i, data), tok): ((u32, ShiftBufferOutput), token) in enumerate(TEST_DATA_1) {
            let (tok, buf_ctrl) = recv(tok, shift_buffer_ctrl_r);
            trace_fmt!("Received #{} buf ctrl {:#x}", i + u32:1, buf_ctrl);
            assert_eq(ShiftBufferCtrl {length: data.length, output: shift_buffer::ShiftBufferOutputType::DATA}, buf_ctrl);
            let tok = send(tok, shift_buffer_out_data_s, data);
            trace_fmt!("Sent #{} buf data {:#x}", i + u32:1, data);
            tok
        }(tok);

        // recv commands
        let tok = for ((i, expected_cmd), tok): ((u32, CommandConstructorData), token) in enumerate(TEST_EXPECTED_COMMANDS_1) {
            let (tok, cmd) = recv(tok, command_r);
            trace_fmt!("Received #{} cmd {:#x}", i + u32:1, cmd);
            assert_eq(expected_cmd, cmd);
            tok
        }(tok);

        // recv finish
        let (tok, _) = recv(tok, finish_r);

        send(tok, terminator, true);
    }
}
