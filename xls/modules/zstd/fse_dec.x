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
import xls.modules.zstd.refilling_shift_buffer;
import xls.modules.zstd.fse_table_creator;

type FseTableRecord = common::FseTableRecord;

type BlockSyncData = common::BlockSyncData;
type SequenceExecutorMessageType = common::SequenceExecutorMessageType;
type SequenceExecutorPacket = common::SequenceExecutorPacket<common::SYMBOLS_IN_PACKET>;
type CommandConstructorData = common::CommandConstructorData;

type CopyOrMatchLength = common::CopyOrMatchLength;
type CopyOrMatchContent = common::CopyOrMatchContent;

type RefillingSBCtrl = refilling_shift_buffer::RefillingShiftBufferCtrl;
type RefillingSBOutput = refilling_shift_buffer::RefillingShiftBufferOutput;

pub enum FseDecoderStatus : u1 {
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

pub struct FseDecoderFinish { status: FseDecoderStatus }

pub fn extract_triplet(num: u64, sz0: u8, sz1: u8, sz2: u8) -> (u64, u64, u64) {
    let shifted2 = num;
    let shifted1 = shifted2 >> sz2;
    let shifted0 = shifted1 >> sz1;
    (
        shifted0 & ((u64:1 << sz0) - u64:1), shifted1 & ((u64:1 << sz1) - u64:1),
        shifted2 & ((u64:1 << sz2) - u64:1),
    )
}

// 3.1.1.3.2.1.1. Sequence Codes for Lengths and Offsets
const SEQ_MAX_CODES_LL = u8:35;
const SEQ_MAX_CODES_ML = u8:51;

const SEQ_LITERAL_LENGTH_BASELINES = u32[36]:[
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 28, 32, 40, 48, 64,
    128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
];
const SEQ_LITERAL_LENGTH_EXTRA_BITS = u8[36]:[
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 6, 7, 8, 9, 10, 11,
    12, 13, 14, 15, 16,
];

const SEQ_MATCH_LENGTH_BASELINES = u32[53]:[
    3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    28, 29, 30, 31, 32, 33, 34, 35, 37, 39, 41, 43, 47, 51, 59, 67, 83, 99, 131, 259, 515, 1027,
    2051, 4099, 8195, 16387, 32771, 65539,
];
const SEQ_MATCH_LENGTH_EXTRA_BITS = u8[53]:[
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
];

enum SEQ_PART : u2 {
    LiteralLength = 0,
    Offset = 1,
    MatchLength = 2,
}

enum FseDecoderFSM : u4 {
    RECV_CTRL = 0,
    PADDING = 1,
    INIT_STATE = 2,
    SEND_RAM_RD_REQ = 3,
    RECV_RAM_RD_RESP = 4,
    READ_BITS = 5,
    UPDATE_STATE = 6,
    SEND_COMMAND = 7,
    SEND_LEFTOVER_LITERALS_REQ = 8,
    SEND_FINISH = 9,
}

enum FseDecoderCommandFSM : u2 {
    IDLE = 0,
    LITERAL = 1,
    SEQUENCE = 2,
}

struct FseDecoderCommandReq {
    send_second: bool,
    cmd0: CommandConstructorData,
    cmd1: CommandConstructorData,
}

struct FseDecoderCommandState { fsm: FseDecoderCommandFSM, req: FseDecoderCommandReq }

proc FseDecoderCommand {
    type State = FseDecoderCommandState;
    type Req = FseDecoderCommandReq;
    type Fsm = FseDecoderCommandFSM;
    req_r: chan<Req> in;
    command_s: chan<CommandConstructorData> out;

    config(req_r: chan<Req> in, command_s: chan<CommandConstructorData> out) { (req_r, command_s) }

    init { zero!<State>() }

    next(state: State) {
        let tok = join();
        match (state.fsm) {
            Fsm::IDLE => {
                let (tok, req) = recv(tok, req_r);
                State { fsm: Fsm::LITERAL, req }
            },
            Fsm::LITERAL => {
                let tok = send(tok, command_s, state.req.cmd0);
                State {
                    fsm: if state.req.send_second { Fsm::SEQUENCE } else { Fsm::IDLE },
                    ..state
                }
            },
            Fsm::SEQUENCE => {
                let tok = send(tok, command_s, state.req.cmd1);
                zero!<State>()
            },
        }
    }
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
    read_bits: u64,
    read_bits_length: u7,
    read_bits_needed: u7,
    sent_buf_ctrl: bool,
    shift_buffer_error: bool,
    padding: u4,
}

pub proc FseDecoder<RAM_DATA_W: u32, RAM_ADDR_W: u32, RAM_NUM_PARTITIONS: u32, AXI_DATA_W: u32, REFILLING_SB_DATA_W:
u32 = {
    AXI_DATA_W}, REFILLING_SB_LENGTH_W:
u32 = {
    refilling_shift_buffer::length_width(REFILLING_SB_DATA_W)}>
{
    type FseRamRdReq = ram::ReadReq<RAM_ADDR_W, RAM_NUM_PARTITIONS>;
    type FseRamRdResp = ram::ReadResp<RAM_DATA_W>;
    type RefillingSBCtrl = refilling_shift_buffer::RefillingShiftBufferCtrl<REFILLING_SB_LENGTH_W>;
    type RefillingSBOutput =
    refilling_shift_buffer::RefillingShiftBufferOutput<REFILLING_SB_DATA_W, REFILLING_SB_LENGTH_W>;
    // control
    ctrl_r: chan<FseDecoderCtrl> in;
    finish_s: chan<FseDecoderFinish> out;
    // shift buffer
    rsb_ctrl_s: chan<RefillingSBCtrl> out;
    rsb_data_r: chan<RefillingSBOutput> in;
    // RAMs
    ll_fse_rd_req_s: chan<FseRamRdReq> out;
    ll_fse_rd_resp_r: chan<FseRamRdResp> in;
    ml_fse_rd_req_s: chan<FseRamRdReq> out;
    ml_fse_rd_resp_r: chan<FseRamRdResp> in;
    of_fse_rd_req_s: chan<FseRamRdReq> out;
    of_fse_rd_resp_r: chan<FseRamRdResp> in;
    cmd_s: chan<FseDecoderCommandReq> out;

    config(ctrl_r: chan<FseDecoderCtrl> in, finish_s: chan<FseDecoderFinish> out,
           rsb_ctrl_s: chan<RefillingSBCtrl> out, rsb_data_r: chan<RefillingSBOutput> in,
           command_s: chan<CommandConstructorData> out, ll_fse_rd_req_s: chan<FseRamRdReq> out,
           ll_fse_rd_resp_r: chan<FseRamRdResp> in, ml_fse_rd_req_s: chan<FseRamRdReq> out,
           ml_fse_rd_resp_r: chan<FseRamRdResp> in, of_fse_rd_req_s: chan<FseRamRdReq> out,
           of_fse_rd_resp_r: chan<FseRamRdResp> in) {
        let (cmd_s, cmd_r) = chan<FseDecoderCommandReq, u32:1>("fse_dec_cmd_req");

        spawn FseDecoderCommand(cmd_r, command_s);

        (
            ctrl_r, finish_s, rsb_ctrl_s, rsb_data_r, ll_fse_rd_req_s, ll_fse_rd_resp_r,
            ml_fse_rd_req_s, ml_fse_rd_resp_r, of_fse_rd_req_s, of_fse_rd_resp_r, cmd_s,
        )
    }

    init { zero!<FseDecoderState>() }

    next(state: FseDecoderState) {
        type RamAddr = uN[RAM_ADDR_W];
        const RAM_MASK_ALL = std::unsigned_max_value<RAM_NUM_PARTITIONS>();

        let tok0 = join();

        // receive ctrl
        let (_, ctrl, ctrl_valid) = recv_if_non_blocking(
            tok0, ctrl_r, state.fsm == FseDecoderFSM::RECV_CTRL, zero!<FseDecoderCtrl>());
        if ctrl_valid { trace_fmt!("ctrl: {:#x}", ctrl); } else {  };
        let state = if ctrl_valid {
            FseDecoderState { ctrl, sequences_count: ctrl.sequences_count, ..state }
        } else {
            state
        };

        // receive ram read response
        let (_, ll_rd_resp, ll_rd_resp_valid) = recv_if_non_blocking(
            tok0, ll_fse_rd_resp_r, state.fsm == FseDecoderFSM::RECV_RAM_RD_RESP,
            zero!<FseRamRdResp>());
        let (_, ml_rd_resp, ml_rd_resp_valid) = recv_if_non_blocking(
            tok0, ml_fse_rd_resp_r, state.fsm == FseDecoderFSM::RECV_RAM_RD_RESP,
            zero!<FseRamRdResp>());
        let (_, of_rd_resp, of_rd_resp_valid) = recv_if_non_blocking(
            tok0, of_fse_rd_resp_r, state.fsm == FseDecoderFSM::RECV_RAM_RD_RESP,
            zero!<FseRamRdResp>());

        let ll_fse_table_record = fse_table_creator::bits_to_fse_record(ll_rd_resp.data);
        let ml_fse_table_record = fse_table_creator::bits_to_fse_record(ml_rd_resp.data);
        let of_fse_table_record = fse_table_creator::bits_to_fse_record(of_rd_resp.data);

        // validate LL and ML symbols
        assert!(
            !(ll_rd_resp_valid && ll_fse_table_record.symbol > SEQ_MAX_CODES_LL),
            "invalid_literal_length_symbol");
        assert!(
            !(ml_rd_resp_valid && ml_fse_table_record.symbol > SEQ_MAX_CODES_ML),
            "invalid_match_length_symbol");

        // request records
        let do_send_ram_rd_req = state.fsm == FseDecoderFSM::SEND_RAM_RD_REQ;

        let ll_req = FseRamRdReq { addr: state.ll_state as RamAddr, mask: RAM_MASK_ALL };
        let ml_req = FseRamRdReq { addr: state.ml_state as RamAddr, mask: RAM_MASK_ALL };
        let of_req = FseRamRdReq { addr: state.of_state as RamAddr, mask: RAM_MASK_ALL };

        send_if(tok0, ll_fse_rd_req_s, do_send_ram_rd_req, ll_req);
        send_if(tok0, ml_fse_rd_req_s, do_send_ram_rd_req, ml_req);
        send_if(tok0, of_fse_rd_req_s, do_send_ram_rd_req, of_req);

        if do_send_ram_rd_req {
            trace_fmt!("ll_req: {:#x}", ll_req);
            trace_fmt!("ml_req: {:#x}", ml_req);
            trace_fmt!("of_req: {:#x}", of_req);
        } else {


        };

        // read bits
        let do_read_bits = (state.fsm == FseDecoderFSM::PADDING ||
                           state.fsm == FseDecoderFSM::INIT_STATE ||
                           state.fsm == FseDecoderFSM::READ_BITS ||
                           state.fsm == FseDecoderFSM::UPDATE_STATE);
        let do_send_buf_ctrl = do_read_bits && !state.sent_buf_ctrl;

        let buf_ctrl_length =
            if (state.read_bits_needed - state.read_bits_length) > REFILLING_SB_DATA_W as u7 {
                REFILLING_SB_DATA_W as u7
            } else {
                state.read_bits_needed - state.read_bits_length
            };

        if do_send_buf_ctrl { trace_fmt!("Asking for {:#x} data", buf_ctrl_length); } else {  };

        send_if(tok0, rsb_ctrl_s, do_send_buf_ctrl, RefillingSBCtrl { length: buf_ctrl_length });

        let state = if do_send_buf_ctrl {
            FseDecoderState { sent_buf_ctrl: do_send_buf_ctrl, ..state }
        } else {
            state
        };

        let recv_sb_output = (do_read_bits && state.sent_buf_ctrl);
        let (_, buf_data, buf_data_valid) =
            recv_if_non_blocking(tok0, rsb_data_r, recv_sb_output, zero!<RefillingSBOutput>());
        if buf_data_valid {
            trace_fmt!("[FseDecoder] Received data {:#x} in state {}", buf_data, state.fsm);
        } else {


        };

        let state = if do_read_bits & buf_data_valid {
            FseDecoderState {
                sent_buf_ctrl: false,
                read_bits: (buf_data.data << state.read_bits_length) | state.read_bits,
                read_bits_length: state.read_bits_length + buf_data.length,
                shift_buffer_error: state.shift_buffer_error | buf_data.error,
                ..state
            }
        } else {
            state
        };

        // send command

        let do_send_command = (state.fsm == FseDecoderFSM::SEND_COMMAND ||
                              state.fsm == FseDecoderFSM::SEND_LEFTOVER_LITERALS_REQ);

        let command = if state.fsm == FseDecoderFSM::SEND_COMMAND {
            FseDecoderCommandReq {
                send_second: true,
                cmd0: CommandConstructorData {
                    sync: state.ctrl.sync,
                    data: SequenceExecutorPacket {
                        msg_type: SequenceExecutorMessageType::LITERAL,
                        length: state.ll,
                        content: CopyOrMatchContent:0,
                        last: false,
                    },
                },
                cmd1: CommandConstructorData {
                    sync: state.ctrl.sync,
                    data: SequenceExecutorPacket {
                        msg_type: SequenceExecutorMessageType::SEQUENCE,
                        length: state.ml,
                        content: state.of,
                        last:
                            (state.sequences_count == u24:1) &&
                            (state.literals_count == state.ctrl.literals_count),
                    },
                },
            }
        } else {
            FseDecoderCommandReq {
                send_second: false,
                cmd0: CommandConstructorData {
                    sync: state.ctrl.sync,
                    data: SequenceExecutorPacket {
                        msg_type: SequenceExecutorMessageType::LITERAL,
                        length:
                            (state.ctrl.literals_count - state.literals_count) as CopyOrMatchLength,
                        content: CopyOrMatchContent:0,
                        last: true,
                    },
                },
                cmd1: zero!<CommandConstructorData>(),
            }
        };

        if do_send_command { trace_fmt!("[FseDecoder] Sending command: {:#x}", command); } else {  };
        send_if(tok0, cmd_s, do_send_command, command);

        // send finish
        send_if(
            tok0, finish_s, state.fsm == FseDecoderFSM::SEND_FINISH,
            FseDecoderFinish {
                status: if state.shift_buffer_error {
                    FseDecoderStatus::ERROR
                } else {
                    FseDecoderStatus::OK
                },
            });

        // update state
        match (state.fsm) {
            FseDecoderFSM::RECV_CTRL => {
                if ctrl_valid {
                    trace_fmt!("[FseDecoder]: Moving to PADDING");
                    if ctrl.sequences_count == u24:0 {
                        FseDecoderState {
                            fsm: FseDecoderFSM::SEND_LEFTOVER_LITERALS_REQ,
                            ctrl,
                            ..state
                        }
                    } else {
                        FseDecoderState {
                            fsm: FseDecoderFSM::PADDING,
                            ctrl,
                            read_bits: u64:0,
                            read_bits_length: u7:0,
                            read_bits_needed: u7:1,
                            ..state
                        }
                    }
                } else {
                    state
                }
                },
            FseDecoderFSM::PADDING => {
                if state.read_bits_needed == state.read_bits_length && !state.sent_buf_ctrl {
                    trace_fmt!("[FseDecoder]: Moving to INIT_LL_STATE");

                    let padding = state.padding + u4:1;
                    assert!(padding <= u4:8, "invalid_padding");

                    let padding_available = (state.read_bits as u1 == u1:0);
                    if padding_available {
                        FseDecoderState {
                            fsm: FseDecoderFSM::PADDING,
                            read_bits: u64:0,
                            read_bits_length: u7:0,
                            read_bits_needed: u7:1,
                            padding,
                            ..state
                        }
                    } else {
                        trace_fmt!("padding is: {:#x}", padding);
                        FseDecoderState {
                            fsm: FseDecoderFSM::INIT_STATE,
                            read_bits: u64:0,
                            read_bits_length: u7:0,
                            read_bits_needed:
                                state.ctrl.of_acc_log + state.ctrl.ll_acc_log +
                                state.ctrl.ml_acc_log,
                            ..state
                        }
                    }
                } else {
                    state
                }
                },
            FseDecoderFSM::INIT_STATE => {
                if state.read_bits_needed == state.read_bits_length && !state.sent_buf_ctrl {
                    trace_fmt!("[FseDecoder]: Moving to INIT_OF_STATE");
                    // order: ll, of, ml
                    let (ll_state, of_state, ml_state) = extract_triplet(
                        state.read_bits, state.ctrl.ll_acc_log as u8, state.ctrl.of_acc_log as u8,
                        state.ctrl.ml_acc_log as u8);

                    FseDecoderState {
                        fsm: FseDecoderFSM::SEND_RAM_RD_REQ,
                        ll_state: ll_state as u16,
                        ml_state: ml_state as u16,
                        of_state: of_state as u16,
                        read_bits: u64:0,
                        read_bits_length: u7:0,
                        read_bits_needed: u7:0,
                        ..state
                    }
                } else {
                    state
                }
            },
            FseDecoderFSM::SEND_RAM_RD_REQ => {
                // trace_fmt!("State LL: {} ML: {} OF: {}", state.ll_state, state.ml_state,
                // state.of_state);
                trace_fmt!("State LL: {:#x} ML: {:#x} OF: {:#x}",
                state.ll_state, state.ml_state, state.of_state);
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
                    FseDecoderState {
                        ll_fse_table_record,
                        ll_fse_table_record_valid: true,
                        ..state
                    }
                } else {
                    state
                };
                let state = if ml_rd_resp_valid {
                    FseDecoderState {
                        ml_fse_table_record,
                        ml_fse_table_record_valid: true,
                        ..state
                    }
                } else {
                    state
                };
                let state = if of_rd_resp_valid {
                    FseDecoderState {
                        of_fse_table_record,
                        of_fse_table_record_valid: true,
                        ..state
                    }
                } else {
                    state
                };

                if state.ll_fse_table_record_valid && state.ml_fse_table_record_valid &&
                state.of_fse_table_record_valid {
                    trace_fmt!("all states received: {:#x}", state);
                    FseDecoderState {
                        fsm: FseDecoderFSM::READ_BITS,
                        read_bits: u64:0,
                        read_bits_length: u7:0,
                        read_bits_needed:
                            (state.of_fse_table_record.symbol as u8 +
                            SEQ_MATCH_LENGTH_EXTRA_BITS[state.ml_fse_table_record.symbol] +
                            SEQ_LITERAL_LENGTH_EXTRA_BITS[state.ll_fse_table_record.symbol]) as
                            u7,
                        ..state
                    }
                } else {
                    state
                }
            },
            FseDecoderFSM::READ_BITS => {
                if (state.read_bits_needed == state.read_bits_length) && !state.sent_buf_ctrl {
                    // order: of, ml, ll
                    let (of_bits, ml_bits, ll_bits) = extract_triplet(
                        state.read_bits, state.of_fse_table_record.symbol as u8,
                        SEQ_MATCH_LENGTH_EXTRA_BITS[state.ml_fse_table_record.symbol],
                        SEQ_LITERAL_LENGTH_EXTRA_BITS[state.ll_fse_table_record.symbol]);
                    let of = ((u32:1 << state.of_fse_table_record.symbol) + of_bits as u32) as u64;
                    let ml = (SEQ_MATCH_LENGTH_BASELINES[state.ml_fse_table_record.symbol] +
                             ml_bits as u32) as
                             u64;
                    let ll = (SEQ_LITERAL_LENGTH_BASELINES[state.ll_fse_table_record.symbol] +
                             ll_bits as u32) as
                             u64;

                    if state.sequences_count == u24:1 {
                        let of_state = state.of_fse_table_record.base + of_bits as u16;
                        FseDecoderState {
                            fsm: FseDecoderFSM::SEND_COMMAND,
                            of,
                            ml,
                            ll,
                            of_state,
                            read_bits: u64:0,
                            read_bits_length: u7:0,
                            read_bits_needed: u7:0,
                            literals_count: state.literals_count + ll as u20,
                            ..state
                        }
                    } else {
                        FseDecoderState {
                            fsm: FseDecoderFSM::UPDATE_STATE,
                            of,
                            ml,
                            ll,
                            read_bits: u64:0,
                            read_bits_length: u7:0,
                            read_bits_needed:
                                (state.ml_fse_table_record.num_of_bits +
                                state.of_fse_table_record.num_of_bits +
                                state.ll_fse_table_record.num_of_bits) as
                                u7,
                            literals_count: state.literals_count + ll as u20,
                            ..state
                        }
                    }
                } else {
                    state
                }
            },
            FseDecoderFSM::UPDATE_STATE => {
                trace_fmt!("Values LL: {:#x} ML: {:#x} OF: {:#x}", state.ll, state.ml, state.of);
                // order: ll, ml, of
                let (ll_bits, ml_bits, of_bits) = extract_triplet(
                    state.read_bits, state.ll_fse_table_record.num_of_bits,
                    state.ml_fse_table_record.num_of_bits, state.of_fse_table_record.num_of_bits);
                let ll_state = state.ll_fse_table_record.base + ll_bits as u16;
                let ml_state = state.ml_fse_table_record.base + ml_bits as u16;
                let of_state = state.of_fse_table_record.base + of_bits as u16;

                if (state.read_bits_needed == state.read_bits_length) && !state.sent_buf_ctrl {
                    FseDecoderState {
                        fsm: FseDecoderFSM::SEND_COMMAND,
                        ll_state,
                        ml_state,
                        of_state,
                        read_bits: u64:0,
                        read_bits_length: u7:0,
                        read_bits_needed: u7:0,
                        ..state
                    }
                } else {
                    state
                }
            },
            FseDecoderFSM::SEND_COMMAND => {
                if state.sequences_count == u24:1 {
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
                        trace_fmt!("Fails state: {:#x}", state);
                        fail!("too_many_literals", state)
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
                FseDecoderState { fsm: FseDecoderFSM::SEND_FINISH, ..zero!<FseDecoderState>() }
                },
            FseDecoderFSM::SEND_FINISH => {
                FseDecoderState { fsm: FseDecoderFSM::RECV_CTRL, ..zero!<FseDecoderState>() }
                },
            _ => { fail!("impossible_case", state) },
        }
    }
}

const INST_RAM_SIZE = common::FSE_MAX_SYMBOLS;
const INST_RAM_ADDR_W = std::clog2(INST_RAM_SIZE);
const INST_RAM_DATA_W = u32:32;
const INST_RAM_WORD_PARTITION_SIZE = INST_RAM_DATA_W / u32:3;
const INST_RAM_NUM_PARTITIONS = ram::num_partitions(INST_RAM_WORD_PARTITION_SIZE, INST_RAM_DATA_W);
const INST_AXI_DATA_W = u32:64;
const INST_REFILLING_SB_DATA_W = INST_AXI_DATA_W;
const INST_REFILLING_SB_LENGTH_W = refilling_shift_buffer::length_width(INST_REFILLING_SB_DATA_W);

pub proc FseDecoderInst {
    type FseRamRdReq = ram::ReadReq<INST_RAM_ADDR_W, INST_RAM_NUM_PARTITIONS>;
    type FseRamRdResp = ram::ReadResp<INST_RAM_DATA_W>;
    type RefillingSBCtrl =
    refilling_shift_buffer::RefillingShiftBufferCtrl<INST_REFILLING_SB_LENGTH_W>;
    type RefillingSBOutput =
    refilling_shift_buffer::RefillingShiftBufferOutput<INST_REFILLING_SB_DATA_W, INST_REFILLING_SB_LENGTH_W>;

    config(ctrl_r: chan<FseDecoderCtrl> in, finish_s: chan<FseDecoderFinish> out,
           rsb_ctrl_s: chan<RefillingSBCtrl> out, rsb_data_r: chan<RefillingSBOutput> in,
           command_s: chan<CommandConstructorData> out, ll_fse_rd_req_s: chan<FseRamRdReq> out,
           ll_fse_rd_resp_r: chan<FseRamRdResp> in, ml_fse_rd_req_s: chan<FseRamRdReq> out,
           ml_fse_rd_resp_r: chan<FseRamRdResp> in, of_fse_rd_req_s: chan<FseRamRdReq> out,
           of_fse_rd_resp_r: chan<FseRamRdResp> in) {
        spawn FseDecoder<INST_RAM_DATA_W, INST_RAM_ADDR_W, INST_RAM_NUM_PARTITIONS, INST_AXI_DATA_W>(
            ctrl_r, finish_s, rsb_ctrl_s, rsb_data_r, command_s, ll_fse_rd_req_s, ll_fse_rd_resp_r,
            ml_fse_rd_req_s, ml_fse_rd_resp_r, of_fse_rd_req_s, of_fse_rd_resp_r);
    }

    init { () }

    next(state: ()) {  }
}
