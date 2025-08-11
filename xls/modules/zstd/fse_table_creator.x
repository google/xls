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

// FseTableCreator generates a decoding table from a probability distribution.
// The algorithm for creating the decoding lookup is described in:
// https://datatracker.ietf.org/doc/html/rfc8878#section-4.1.1.

import std;
import xls.examples.ram;
import xls.modules.zstd.common;
import xls.modules.zstd.ram_wr_handler as ram_wr;
import xls.modules.zstd.fse_table_iterator as fse_table_iterator;

type SymbolCount = common::FseSymbolCount;
type AccuracyLog = common::FseAccuracyLog;

enum Status : u4 {
    RECEIVE_START = 0,
    TEST_NEGATIVE_PROB = 1,
    HANDLE_NEGATIVE_PROB = 2,
    TEST_POSITIVE_PROB = 3,
    HANDLE_POSITIVE_PROB = 4,
    HANDLE_POSITIVE_PROB_WRITE_STATE_DESC = 5,
    INNER_FOR_GET_POS = 6,
    INNER_FOR_WRITE_SYM = 7,
    LAST_FOR = 8,
    GET_STATE_DESC = 9,
    SET_STATE_DESC = 10,
    SEND_FINISH = 11,
    START_ITERATING_POS = 12,
}

struct FseTableCreatorState {
    status: Status,
    req: bool,
    idx: u10,
    // TODO: num_symbs is u8, possibly other fields as well
    num_symbs: u8,
    curr_symbol: u8,
    state_desc_for_symbol: u16,
    accuracy_log: u16,
    high_threshold: u16,
    inner_for_idx: u16,
    inner_for_range: u16,
    dpd_data: u16,
    pos: u16,
}

type FseTableRecord = common::FseTableRecord;

pub struct FseStartMsg { num_symbs: SymbolCount, accuracy_log: AccuracyLog }

pub fn fse_record_to_bits(record: FseTableRecord) -> u32 {
    record.base ++ record.num_of_bits ++ record.symbol
}

#[test]
fn test_fse_record_to_bits() {
    let bit = fse_record_to_bits(
        FseTableRecord { symbol: u8:0x17, num_of_bits: u8:0x05, base: u16:0x0020 }
    );
    assert_eq(bit, u32:0x0020_05_17);
}

pub fn bits_to_fse_record(bit: u32) -> FseTableRecord {
    FseTableRecord {
        symbol: bit[0:8],
        num_of_bits: bit[8:16],
        base: bit[16:32]
    }
}

#[test]
fn test_bits_to_fse_record() {
    let record = bits_to_fse_record(u32:0x0020_05_17);
    assert_eq(record, FseTableRecord { symbol: u8:0x17, num_of_bits: u8:0x05, base: u16:0x0020 });
}

pub proc FseTableCreator<
    // Default Probability Distribution RAM parameters
    DPD_RAM_DATA_WIDTH: u32, DPD_RAM_ADDR_WIDTH: u32, DPD_RAM_NUM_PARTITIONS: u32,
    // FSE lookup table parameters
    FSE_RAM_DATA_WIDTH: u32, FSE_RAM_ADDR_WIDTH: u32, FSE_RAM_NUM_PARTITIONS: u32,
    // Temp RAM parameters
    TMP_RAM_DATA_WIDTH: u32, TMP_RAM_ADDR_WIDTH: u32, TMP_RAM_NUM_PARTITIONS: u32,
    TMP2_RAM_DATA_WIDTH: u32, TMP2_RAM_ADDR_WIDTH: u32, TMP2_RAM_NUM_PARTITIONS: u32,
> {
    type State = FseTableCreatorState;

    type DpdRamReadReq = ram::ReadReq<DPD_RAM_ADDR_WIDTH, DPD_RAM_NUM_PARTITIONS>;
    type DpdRamReadResp = ram::ReadResp<DPD_RAM_DATA_WIDTH>;

    type FseRamWriteReq = ram::WriteReq<FSE_RAM_ADDR_WIDTH, FSE_RAM_DATA_WIDTH, FSE_RAM_NUM_PARTITIONS>;
    type FseRamWriteResp = ram::WriteResp;

    type TmpRamWriteReq = ram::WriteReq<TMP_RAM_ADDR_WIDTH, TMP_RAM_DATA_WIDTH, TMP_RAM_NUM_PARTITIONS>;
    type TmpRamWriteResp = ram::WriteResp;
    type TmpRamReadReq = ram::ReadReq<TMP_RAM_ADDR_WIDTH, TMP_RAM_NUM_PARTITIONS>;
    type TmpRamReadResp = ram::ReadResp<TMP_RAM_DATA_WIDTH>;

    type Tmp2RamWriteReq = ram::WriteReq<TMP2_RAM_ADDR_WIDTH, TMP2_RAM_DATA_WIDTH, TMP2_RAM_NUM_PARTITIONS>;
    type Tmp2RamWriteResp = ram::WriteResp;
    type Tmp2RamReadReq = ram::ReadReq<TMP2_RAM_ADDR_WIDTH, TMP2_RAM_NUM_PARTITIONS>;
    type Tmp2RamReadResp = ram::ReadResp<TMP2_RAM_DATA_WIDTH>;

    type TestRamWriteResp = ram::WriteResp;

    type IterCtrl = common::FseTableCreatorCtrl;
    type IterIndex = common::FseTableIndex;

    dpd_rd_req_s: chan<DpdRamReadReq> out;
    dpd_rd_resp_r: chan<DpdRamReadResp> in;

    // a request to start creating the FSE decoding table
    fse_table_start_r: chan<FseStartMsg> in;
    // a response with information that the table has been saved to RAM
    fse_table_finish_s: chan<()> out;

    fse_wr_req_s: chan<FseRamWriteReq> out;
    fse_wr_resp_r: chan<FseRamWriteResp> in;

    tmp_rd_req_s: chan<TmpRamReadReq> out;
    tmp_rd_resp_r: chan<TmpRamReadResp> in;
    tmp_wr_req_s: chan<TmpRamWriteReq> out;
    tmp_wr_resp_r: chan<TmpRamWriteResp> in;

    tmp2_rd_req_s: chan<Tmp2RamReadReq> out;
    tmp2_rd_resp_r: chan<Tmp2RamReadResp> in;
    tmp2_wr_req_s: chan<Tmp2RamWriteReq> out;
    tmp2_wr_resp_r: chan<Tmp2RamWriteResp> in;

    it_ctrl_s: chan<IterCtrl> out;
    it_index_r: chan<IterIndex> in;

    config(
        fse_table_start_r: chan<FseStartMsg> in,
        fse_table_finish_s: chan<()> out,

        // RAM with default probability distribution
        dpd_rd_req_s: chan<DpdRamReadReq> out,
        dpd_rd_resp_r: chan<DpdRamReadResp> in,

        // Ram with FSE decoding table
        fse_wr_req_s: chan<FseRamWriteReq> out,
        fse_wr_resp_r: chan<FseRamWriteResp> in,

        tmp_rd_req_s: chan<TmpRamReadReq> out,
        tmp_rd_resp_r: chan<TmpRamReadResp> in,
        tmp_wr_req_s: chan<TmpRamWriteReq> out,
        tmp_wr_resp_r: chan<TmpRamWriteResp> in,

        tmp2_rd_req_s: chan<Tmp2RamReadReq> out,
        tmp2_rd_resp_r: chan<Tmp2RamReadResp> in,
        tmp2_wr_req_s: chan<Tmp2RamWriteReq> out,
        tmp2_wr_resp_r: chan<Tmp2RamWriteResp> in,
    ) {
        let (it_ctrl_s, it_ctrl_r) = chan<IterCtrl, u32:1>("it_ctrl");
        let (it_index_s, it_index_r) = chan<IterIndex, u32:1>("it_index");
        spawn fse_table_iterator::FseTableIterator(it_ctrl_r, it_index_s);

        (
            dpd_rd_req_s, dpd_rd_resp_r,
            fse_table_start_r, fse_table_finish_s,
            fse_wr_req_s, fse_wr_resp_r,
            tmp_rd_req_s, tmp_rd_resp_r, tmp_wr_req_s, tmp_wr_resp_r,
            tmp2_rd_req_s, tmp2_rd_resp_r, tmp2_wr_req_s, tmp2_wr_resp_r,
            it_ctrl_s, it_index_r
        )
    }

    init { zero!<State>() }

    next(state: State) {
        const DPD_RAM_REQ_MASK_ALL = std::unsigned_max_value<DPD_RAM_NUM_PARTITIONS>();
        const FSE_RAM_REQ_MASK_ALL = std::unsigned_max_value<FSE_RAM_NUM_PARTITIONS>();
        const FSE_RAM_REQ_MASK_SYMBOL = uN[FSE_RAM_NUM_PARTITIONS]:1;
        const TMP_RAM_REQ_MASK_ALL = std::unsigned_max_value<TMP_RAM_NUM_PARTITIONS>();
        const TMP2_RAM_REQ_MASK_ALL = std::unsigned_max_value<TMP2_RAM_NUM_PARTITIONS>();

        let tok0 = join();

        let receive_start = (state.status == Status::RECEIVE_START);
        let (tok1, fse_start_msg) = recv_if(tok0, fse_table_start_r, receive_start, zero!<FseStartMsg>());

        let get_dpd_data = state.status == Status::TEST_NEGATIVE_PROB ||
                           state.status == Status::TEST_POSITIVE_PROB ||
                           state.status == Status::HANDLE_POSITIVE_PROB;

        let send_dpd_req = get_dpd_data && state.req;
        let addr = if send_dpd_req {
            checked_cast<uN[DPD_RAM_ADDR_WIDTH]>(state.idx)
        } else {
            uN[DPD_RAM_ADDR_WIDTH]:0
        };
        let tok_dpd_req = send_if(tok0, dpd_rd_req_s, send_dpd_req,
            DpdRamReadReq {
                addr: addr,
                mask: DPD_RAM_REQ_MASK_ALL
            });
        let get_dpd_resp = get_dpd_data && !state.req;
        let (tok_dpd_resp, dpd_resp) = recv_if(tok0, dpd_rd_resp_r, get_dpd_resp, zero!<DpdRamReadResp>());

        let handle_negative_prob_req = state.status == Status::HANDLE_NEGATIVE_PROB;
        let decreased_high_threshold = state.high_threshold - u16:1;
        let fse_wr_req = if handle_negative_prob_req {
            Tmp2RamWriteReq {
                addr: checked_cast<uN[TMP2_RAM_ADDR_WIDTH]>(decreased_high_threshold),
                data: checked_cast<uN[TMP2_RAM_DATA_WIDTH]>(state.idx),
                mask: TMP2_RAM_REQ_MASK_ALL,
            }
        } else {
            zero!<Tmp2RamWriteReq>()
        };
        let tok3 = send_if(tok0, tmp2_wr_req_s, handle_negative_prob_req, fse_wr_req);
        let handle_negative_prob_resp = (state.status == Status::HANDLE_NEGATIVE_PROB);
        let (tok3, _) = recv_if(tok3, tmp2_wr_resp_r, handle_negative_prob_resp, FseRamWriteResp {});

        let addr = if handle_negative_prob_req {
            checked_cast<uN[TMP_RAM_ADDR_WIDTH]>(state.idx)
        } else {
            uN[TMP_RAM_ADDR_WIDTH]:0
        };
        let tok5 = send_if(tok0, tmp_wr_req_s, handle_negative_prob_req,
            TmpRamWriteReq {
                addr: addr,
                data: checked_cast<uN[TMP_RAM_DATA_WIDTH]>(u16:1),
                mask: TMP_RAM_REQ_MASK_ALL
            });
        let (tok5, _) = recv_if(tok5, tmp_wr_resp_r, handle_negative_prob_resp, TestRamWriteResp {});

        let handle_positive_prob_write_state_desc = (state.status == Status::HANDLE_POSITIVE_PROB_WRITE_STATE_DESC);
        let addr = if handle_positive_prob_write_state_desc {
            checked_cast<uN[TMP_RAM_ADDR_WIDTH]>(state.idx)
        } else {
            uN[TMP_RAM_ADDR_WIDTH]:0
        };
        let tok6 = send_if(tok0, tmp_wr_req_s, handle_positive_prob_write_state_desc,
            TmpRamWriteReq {
                addr: addr,
                data: checked_cast<uN[TMP_RAM_DATA_WIDTH]>(state.dpd_data),
                mask: TMP_RAM_REQ_MASK_ALL
            }
        );
        let (tok6, _) = recv_if(tok6, tmp_wr_resp_r, handle_positive_prob_write_state_desc, TmpRamWriteResp {});

        let inner_for_start_counting = state.status == Status::START_ITERATING_POS;
        let negative_proba_count = (u16:1 << state.accuracy_log) - state.high_threshold;
        let tok7 = send_if( tok0, it_ctrl_s, inner_for_start_counting,
            IterCtrl {
                accuracy_log: checked_cast<AccuracyLog>(state.accuracy_log),
                negative_proba_count: checked_cast<SymbolCount>(negative_proba_count),
            }
        );
        let inner_for_get_pos = (state.status == Status::INNER_FOR_GET_POS);
        let (_, pos) = recv_if(tok0, it_index_r, inner_for_get_pos, zero!<IterIndex>());

        let inner_for_write_sym = state.status == Status::INNER_FOR_WRITE_SYM;
        let idx = if inner_for_write_sym {
            checked_cast<uN[TMP2_RAM_DATA_WIDTH]>(state.idx)
        } else {
            uN[TMP2_RAM_DATA_WIDTH]:0
        };
        let tok4 = send_if( tok0, tmp2_wr_req_s, inner_for_write_sym,
            Tmp2RamWriteReq {
                addr: checked_cast<uN[TMP2_RAM_ADDR_WIDTH]>(state.pos),
                data: idx,
                mask: TMP2_RAM_REQ_MASK_ALL,
            }
        );

        let (tok4, _) = recv_if(tok4, tmp2_wr_resp_r, inner_for_write_sym, FseRamWriteResp {});

        let last_for = state.status == Status::LAST_FOR;
        let tok8 = send_if(tok0, tmp2_rd_req_s, last_for,
            Tmp2RamReadReq {
                addr: checked_cast<uN[TMP2_RAM_ADDR_WIDTH]>(state.idx),
                mask: TMP2_RAM_REQ_MASK_ALL,
            }
        );
        let (tok8, fse_resp) = recv_if(tok8, tmp2_rd_resp_r, last_for, zero!<Tmp2RamReadResp>());
        let fse_record_symbol = fse_resp.data;

        let get_state_desc = state.status == Status::GET_STATE_DESC;
        let symbol = state.curr_symbol;
        let tok8 = send_if(tok8, tmp_rd_req_s, get_state_desc,
            TmpRamReadReq {
                addr: checked_cast<uN[TMP_RAM_ADDR_WIDTH]>(symbol),
                mask: TMP_RAM_REQ_MASK_ALL
            }
        );
        let (tok8, tmp_resp) = recv_if(tok8, tmp_rd_resp_r, get_state_desc, zero!<TmpRamReadResp>());

        let set_state_desc = state.status == Status::SET_STATE_DESC;
        let tok9 = send_if(tok8, tmp_wr_req_s, set_state_desc,
            TmpRamWriteReq {
                addr: checked_cast<uN[TMP_RAM_ADDR_WIDTH]>(symbol),
                data: checked_cast<uN[TMP_RAM_DATA_WIDTH]>(state.state_desc_for_symbol + u16:1),
                mask: TMP_RAM_REQ_MASK_ALL
            }
        );
        let (tok9, _) = recv_if(tok9, tmp_wr_resp_r, set_state_desc, TmpRamWriteResp {});

        let num_bits = state.accuracy_log - common::highest_set_bit(state.state_desc_for_symbol);
        let size = u16:1 << state.accuracy_log;
        let new_state_base = (state.state_desc_for_symbol << num_bits) - size;

        let complete_record = FseTableRecord {
            symbol: symbol,
            num_of_bits: checked_cast<u8>(num_bits),
            base: new_state_base
        };
        let complete_record_as_bits = fse_record_to_bits(complete_record);

        let fse_wr_req = FseRamWriteReq {
                addr: checked_cast<uN[FSE_RAM_ADDR_WIDTH]>(state.idx),
                data: checked_cast<uN[FSE_RAM_DATA_WIDTH]>(complete_record_as_bits),
                mask: FSE_RAM_REQ_MASK_ALL
        };
        let tok10 = send_if(tok8, fse_wr_req_s, set_state_desc, fse_wr_req);
        let (tok10, _) = recv_if(tok10, fse_wr_resp_r, set_state_desc, FseRamWriteResp {});

        let send_finish = state.status == Status::SEND_FINISH;
        let tok11 = send_if(tok0, fse_table_finish_s, send_finish, ());

        // trace_fmt!("fse lookup state: {:#x}", state);

        if state.req && (
               state.status == Status::TEST_NEGATIVE_PROB ||
               state.status == Status::TEST_POSITIVE_PROB ||
               state.status == Status::HANDLE_POSITIVE_PROB) {
            State { req: false, ..state }
        } else {
            match (state.status) {
                Status::RECEIVE_START => {
                    State {
                        status: Status::TEST_NEGATIVE_PROB,
                        req: true,
                        num_symbs: checked_cast<u8>(fse_start_msg.num_symbs),
                        accuracy_log: checked_cast<u16>(fse_start_msg.accuracy_log),
                        high_threshold: u16:1 << fse_start_msg.accuracy_log,
                    ..state
                    }
                },
                Status::TEST_NEGATIVE_PROB => {
                    if dpd_resp.data == s16:-1 as u16 {
                        State { status: Status::HANDLE_NEGATIVE_PROB, ..state }
                    } else {
                        let next_idx = state.idx + u10:1;
                        if next_idx < checked_cast<u10>(state.num_symbs) {
                            State { status: Status::TEST_NEGATIVE_PROB, req: true, idx: next_idx, ..state }
                        } else {
                            State { status: Status::START_ITERATING_POS, req: true, idx: u10:0, ..state }
                        }
                    }
                },
                Status::HANDLE_NEGATIVE_PROB => {
                    // https://github.com/facebook/zstd/blob/9f42fa0a043aa389534cf10ff086976c4c6b10a6/doc/educational_decoder/zstd_decompress.c#L2143-L2146
                    let next_idx = state.idx + u10:1;
                    if next_idx < checked_cast<u10>(state.num_symbs) {
                        State { status: Status::TEST_NEGATIVE_PROB, req: true, idx: next_idx, high_threshold: decreased_high_threshold, ..state }
                    } else {
                        State { status: Status::START_ITERATING_POS, req: true, idx: u10:0, high_threshold: decreased_high_threshold, ..state }
                    }
                },
                Status::START_ITERATING_POS => {
                    State { status: Status::TEST_POSITIVE_PROB, ..state }
                },
                Status::TEST_POSITIVE_PROB => {
                    if dpd_resp.data as s16 > s16:0 {
                        State { status: Status::HANDLE_POSITIVE_PROB, req: true, ..state }
                    } else {
                        let next_idx = state.idx + u10:1;
                        if next_idx < checked_cast<u10>(state.num_symbs) {
                            State { status: Status::TEST_POSITIVE_PROB, req: true, idx: next_idx, ..state }
                        } else {
                            State { status: Status::LAST_FOR, idx: u10:0, ..state }
                        }
                    }
                },
                Status::HANDLE_POSITIVE_PROB => {
                    // https://github.com/facebook/zstd/blob/9f42fa0a043aa389534cf10ff086976c4c6b10a6/doc/educational_decoder/zstd_decompress.c#L2161
                    State { status: Status::HANDLE_POSITIVE_PROB_WRITE_STATE_DESC, dpd_data: dpd_resp.data, ..state }
                },
                Status::HANDLE_POSITIVE_PROB_WRITE_STATE_DESC => {
                    State { status: Status::INNER_FOR_GET_POS, inner_for_idx: u16:0, inner_for_range: checked_cast<u16>(state.dpd_data), ..state }
                },
                Status::INNER_FOR_GET_POS => {
                    // https://github.com/facebook/zstd/blob/9f42fa0a043aa389534cf10ff086976c4c6b10a6/doc/educational_decoder/zstd_decompress.c#L2165
                    State { status: Status::INNER_FOR_WRITE_SYM, pos: checked_cast<u16>(pos), ..state }
                },
                Status::INNER_FOR_WRITE_SYM => {
                    let next_idx = state.inner_for_idx + u16:1;
                    if next_idx < state.inner_for_range {
                        State { status: Status::INNER_FOR_GET_POS, inner_for_idx: next_idx, ..state }
                    } else {
                        assert!(pos == IterIndex:0, "corruption_detected_while_decompressing");
                        let next_idx = state.idx + u10:1;
                        if next_idx < checked_cast<u10>(state.num_symbs) {
                            State { status: Status::TEST_POSITIVE_PROB, req: true, idx: next_idx, ..state }
                        } else {
                            State { status: Status::LAST_FOR, idx: u10:0, ..state }
                        }
                    }
                },
                Status::LAST_FOR => {
                    // https://github.com/facebook/zstd/blob/9f42fa0a043aa389534cf10ff086976c4c6b10a6/doc/educational_decoder/zstd_decompress.c#L2183
                    State { status: Status::GET_STATE_DESC, curr_symbol: fse_record_symbol, ..state }
                },
                Status::GET_STATE_DESC => {
                    // https://github.com/facebook/zstd/blob/9f42fa0a043aa389534cf10ff086976c4c6b10a6/doc/educational_decoder/zstd_decompress.c#L2184
                    State { status: Status::SET_STATE_DESC, state_desc_for_symbol: tmp_resp.data, ..state }
                },
                Status::SET_STATE_DESC => {
                    let next_idx = state.idx + u10:1;
                    if next_idx as u16 < size {
                        State { status: Status::LAST_FOR, idx: next_idx, ..state }
                    } else {
                        State { status: Status::SEND_FINISH, ..state }
                    }
                },
                Status::SEND_FINISH => { State { status: Status::RECEIVE_START, ..zero!<State>() } },
                _ => fail!("impossible_case", zero!<State>()),
            }
        }
    }
}

const TEST_DPD_RAM_DATA_WIDTH = u32:16;
const TEST_DPD_RAM_SIZE = u32:256;
const TEST_DPD_RAM_ADDR_WIDTH = std::clog2(TEST_DPD_RAM_SIZE);
const TEST_DPD_RAM_WORD_PARTITION_SIZE = TEST_DPD_RAM_DATA_WIDTH;
const TEST_DPD_RAM_NUM_PARTITIONS = ram::num_partitions(
    TEST_DPD_RAM_WORD_PARTITION_SIZE, TEST_DPD_RAM_DATA_WIDTH);

const TEST_FSE_RAM_DATA_WIDTH = u32:32;
const TEST_FSE_RAM_SIZE = u32:1 << common::FSE_MAX_ACCURACY_LOG;
const TEST_FSE_RAM_ADDR_WIDTH = std::clog2(TEST_FSE_RAM_SIZE);
const TEST_FSE_RAM_WORD_PARTITION_SIZE = TEST_FSE_RAM_DATA_WIDTH;
const TEST_FSE_RAM_NUM_PARTITIONS = ram::num_partitions(
    TEST_FSE_RAM_WORD_PARTITION_SIZE, TEST_FSE_RAM_DATA_WIDTH);

const TEST_TMP_RAM_DATA_WIDTH = u32:16;
const TEST_TMP_RAM_SIZE = u32:256;
const TEST_TMP_RAM_ADDR_WIDTH = std::clog2(TEST_TMP_RAM_SIZE);
const TEST_TMP_RAM_WORD_PARTITION_SIZE = TEST_TMP_RAM_DATA_WIDTH;
const TEST_TMP_RAM_NUM_PARTITIONS = ram::num_partitions(
    TEST_TMP_RAM_WORD_PARTITION_SIZE, TEST_TMP_RAM_DATA_WIDTH);

const TEST_TMP2_RAM_DATA_WIDTH = u32:8;
const TEST_TMP2_RAM_SIZE = u32:512;
const TEST_TMP2_RAM_ADDR_WIDTH = std::clog2(TEST_TMP2_RAM_SIZE);
const TEST_TMP2_RAM_WORD_PARTITION_SIZE = TEST_TMP2_RAM_DATA_WIDTH;
const TEST_TMP2_RAM_NUM_PARTITIONS = ram::num_partitions(
    TEST_TMP2_RAM_WORD_PARTITION_SIZE, TEST_TMP2_RAM_DATA_WIDTH);

proc FseTableCreatorInst {
    type DpdRamReadReq = ram::ReadReq<TEST_DPD_RAM_ADDR_WIDTH, TEST_DPD_RAM_NUM_PARTITIONS>;
    type DpdRamReadResp = ram::ReadResp<TEST_DPD_RAM_DATA_WIDTH>;

    type FseRamReadReq = ram::ReadReq<TEST_FSE_RAM_ADDR_WIDTH, TEST_FSE_RAM_NUM_PARTITIONS>;
    type FseRamReadResp = ram::ReadResp<TEST_FSE_RAM_DATA_WIDTH>;
    type FseRamWriteReq = ram::WriteReq<TEST_FSE_RAM_ADDR_WIDTH, TEST_FSE_RAM_DATA_WIDTH, TEST_FSE_RAM_NUM_PARTITIONS>;
    type FseRamWriteResp = ram::WriteResp;

    type TmpRamWriteReq = ram::WriteReq<TEST_TMP_RAM_ADDR_WIDTH, TEST_TMP_RAM_DATA_WIDTH, TEST_TMP_RAM_NUM_PARTITIONS>;
    type TmpRamWriteResp = ram::WriteResp;
    type TmpRamReadReq = ram::ReadReq<TEST_TMP_RAM_ADDR_WIDTH, TEST_TMP_RAM_NUM_PARTITIONS>;
    type TmpRamReadResp = ram::ReadResp<TEST_TMP_RAM_DATA_WIDTH>;

    type Tmp2RamWriteReq = ram::WriteReq<TEST_TMP2_RAM_ADDR_WIDTH, TEST_TMP2_RAM_DATA_WIDTH, TEST_TMP2_RAM_NUM_PARTITIONS>;
    type Tmp2RamWriteResp = ram::WriteResp;
    type Tmp2RamReadReq = ram::ReadReq<TEST_TMP2_RAM_ADDR_WIDTH, TEST_TMP2_RAM_NUM_PARTITIONS>;
    type Tmp2RamReadResp = ram::ReadResp<TEST_TMP2_RAM_DATA_WIDTH>;

    config(
        fse_table_start_r: chan<FseStartMsg> in,
        fse_table_finish_s: chan<()> out,

        dpd_rd_req_s: chan<DpdRamReadReq> out,
        dpd_rd_resp_r: chan<DpdRamReadResp> in,

        fse_wr_req_s: chan<FseRamWriteReq> out,
        fse_wr_resp_r: chan<FseRamWriteResp> in,

        tmp_rd_req_s: chan<TmpRamReadReq> out,
        tmp_rd_resp_r: chan<TmpRamReadResp> in,
        tmp_wr_req_s: chan<TmpRamWriteReq> out,
        tmp_wr_resp_r: chan<TmpRamWriteResp> in,

        tmp2_rd_req_s: chan<Tmp2RamReadReq> out,
        tmp2_rd_resp_r: chan<Tmp2RamReadResp> in,
        tmp2_wr_req_s: chan<Tmp2RamWriteReq> out,
        tmp2_wr_resp_r: chan<Tmp2RamWriteResp> in,
    ) {
        spawn FseTableCreator<
            TEST_DPD_RAM_DATA_WIDTH, TEST_DPD_RAM_ADDR_WIDTH, TEST_DPD_RAM_NUM_PARTITIONS,
            TEST_FSE_RAM_DATA_WIDTH, TEST_FSE_RAM_ADDR_WIDTH, TEST_FSE_RAM_NUM_PARTITIONS,
            TEST_TMP_RAM_DATA_WIDTH, TEST_TMP_RAM_ADDR_WIDTH, TEST_TMP_RAM_NUM_PARTITIONS,
            TEST_TMP2_RAM_DATA_WIDTH, TEST_TMP2_RAM_ADDR_WIDTH, TEST_TMP2_RAM_NUM_PARTITIONS,
        >(
            fse_table_start_r, fse_table_finish_s,
            dpd_rd_req_s, dpd_rd_resp_r,
            fse_wr_req_s, fse_wr_resp_r,
            tmp_rd_req_s, tmp_rd_resp_r, tmp_wr_req_s, tmp_wr_resp_r,
            tmp2_rd_req_s, tmp2_rd_resp_r, tmp2_wr_req_s,tmp2_wr_resp_r,
        );
    }

    init {  }

    next(state: ()) {  }
}

const TEST_OFFSET_CODE_TABLE = FseTableRecord[32]:[
    FseTableRecord { symbol: u8:0, num_of_bits: u8:5, base: u16:0 },
    FseTableRecord { symbol: u8:6, num_of_bits: u8:4, base: u16:0 },
    FseTableRecord { symbol: u8:9, num_of_bits: u8:5, base: u16:0 },
    FseTableRecord { symbol: u8:15, num_of_bits: u8:5, base: u16:0 },
    FseTableRecord { symbol: u8:21, num_of_bits: u8:5, base: u16:0 },
    FseTableRecord { symbol: u8:3, num_of_bits: u8:5, base: u16:0 },
    FseTableRecord { symbol: u8:7, num_of_bits: u8:4, base: u16:0 },
    FseTableRecord { symbol: u8:12, num_of_bits: u8:5, base: u16:0 },
    FseTableRecord { symbol: u8:18, num_of_bits: u8:5, base: u16:0 },
    FseTableRecord { symbol: u8:23, num_of_bits: u8:5, base: u16:0 },
    FseTableRecord { symbol: u8:5, num_of_bits: u8:5, base: u16:0 },
    FseTableRecord { symbol: u8:8, num_of_bits: u8:4, base: u16:0 },
    FseTableRecord { symbol: u8:14, num_of_bits: u8:5, base: u16:0 },
    FseTableRecord { symbol: u8:20, num_of_bits: u8:5, base: u16:0 },
    FseTableRecord { symbol: u8:2, num_of_bits: u8:5, base: u16:0 },
    FseTableRecord { symbol: u8:7, num_of_bits: u8:4, base: u16:16 },
    FseTableRecord { symbol: u8:11, num_of_bits: u8:5, base: u16:0 },
    FseTableRecord { symbol: u8:17, num_of_bits: u8:5, base: u16:0 },
    FseTableRecord { symbol: u8:22, num_of_bits: u8:5, base: u16:0 },
    FseTableRecord { symbol: u8:4, num_of_bits: u8:5, base: u16:0 },
    FseTableRecord { symbol: u8:8, num_of_bits: u8:4, base: u16:16 },
    FseTableRecord { symbol: u8:13, num_of_bits: u8:5, base: u16:0 },
    FseTableRecord { symbol: u8:19, num_of_bits: u8:5, base: u16:0 },
    FseTableRecord { symbol: u8:1, num_of_bits: u8:5, base: u16:0 },
    FseTableRecord { symbol: u8:6, num_of_bits: u8:4, base: u16:16 },
    FseTableRecord { symbol: u8:10, num_of_bits: u8:5, base: u16:0 },
    FseTableRecord { symbol: u8:16, num_of_bits: u8:5, base: u16:0 },
    FseTableRecord { symbol: u8:28, num_of_bits: u8:5, base: u16:0 },
    FseTableRecord { symbol: u8:27, num_of_bits: u8:5, base: u16:0 },
    FseTableRecord { symbol: u8:26, num_of_bits: u8:5, base: u16:0 },
    FseTableRecord { symbol: u8:25, num_of_bits: u8:5, base: u16:0 },
    FseTableRecord { symbol: u8:24, num_of_bits: u8:5, base: u16:0 },
];

#[test_proc]
proc FseTableCreatorTest {
    type DpdRamReadReq = ram::ReadReq<TEST_DPD_RAM_ADDR_WIDTH, TEST_DPD_RAM_NUM_PARTITIONS>;
    type DpdRamReadResp = ram::ReadResp<TEST_DPD_RAM_DATA_WIDTH>;
    type DpdRamWriteReq = ram::WriteReq<TEST_DPD_RAM_ADDR_WIDTH, TEST_DPD_RAM_DATA_WIDTH, TEST_DPD_RAM_NUM_PARTITIONS>;
    type DpdRamWriteResp = ram::WriteResp;

    type FseRamReadReq = ram::ReadReq<TEST_FSE_RAM_ADDR_WIDTH, TEST_FSE_RAM_NUM_PARTITIONS>;
    type FseRamReadResp = ram::ReadResp<TEST_FSE_RAM_DATA_WIDTH>;
    type FseRamWriteReq = ram::WriteReq<TEST_FSE_RAM_ADDR_WIDTH, TEST_FSE_RAM_DATA_WIDTH, TEST_FSE_RAM_NUM_PARTITIONS>;
    type FseRamWriteResp = ram::WriteResp;

    type TmpRamReadReq = ram::ReadReq<TEST_TMP_RAM_ADDR_WIDTH, TEST_TMP_RAM_NUM_PARTITIONS>;
    type TmpRamReadResp = ram::ReadResp<TEST_TMP_RAM_DATA_WIDTH>;
    type TmpRamWriteReq = ram::WriteReq<TEST_TMP_RAM_ADDR_WIDTH, TEST_TMP_RAM_DATA_WIDTH, TEST_TMP_RAM_NUM_PARTITIONS>;
    type TmpRamWriteResp = ram::WriteResp;

    type Tmp2RamWriteReq = ram::WriteReq<TEST_TMP2_RAM_ADDR_WIDTH, TEST_TMP2_RAM_DATA_WIDTH, TEST_TMP2_RAM_NUM_PARTITIONS>;
    type Tmp2RamWriteResp = ram::WriteResp;
    type Tmp2RamReadReq = ram::ReadReq<TEST_TMP2_RAM_ADDR_WIDTH, TEST_TMP2_RAM_NUM_PARTITIONS>;
    type Tmp2RamReadResp = ram::ReadResp<TEST_TMP2_RAM_DATA_WIDTH>;

    terminator: chan<bool> out;
    fse_table_start_s: chan<FseStartMsg> out;
    fse_table_finish_r: chan<()> in;

    dpd_wr_req_s: chan<DpdRamWriteReq> out;
    dpd_wr_resp_r: chan<DpdRamWriteResp> in;

    fse_rd_req_s: chan<FseRamReadReq> out;
    fse_rd_resp_r: chan<FseRamReadResp> in;

    config(terminator: chan<bool> out) {
        let (dpd_rd_req_s, dpd_rd_req_r) = chan<DpdRamReadReq>("dpd_rd_req");
        let (dpd_rd_resp_s, dpd_rd_resp_r) = chan<DpdRamReadResp>("dpd_rd_resp");
        let (dpd_wr_req_s, dpd_wr_req_r) = chan<DpdRamWriteReq>("dpd_wr_req");
        let (dpd_wr_resp_s, dpd_wr_resp_r) = chan<DpdRamWriteResp>("dpd_wr_resp");

        spawn ram::RamModel<
            TEST_DPD_RAM_DATA_WIDTH, TEST_DPD_RAM_SIZE, TEST_DPD_RAM_WORD_PARTITION_SIZE>(
            dpd_rd_req_r, dpd_rd_resp_s, dpd_wr_req_r, dpd_wr_resp_s);

        let (fse_rd_req_s, fse_rd_req_r) = chan<FseRamReadReq>("fse_rd_req");
        let (fse_rd_resp_s, fse_rd_resp_r) = chan<FseRamReadResp>("fse_rd_resp");
        let (fse_wr_req_s, fse_wr_req_r) = chan<FseRamWriteReq>("fse_wr_req");
        let (fse_wr_resp_s, fse_wr_resp_r) = chan<FseRamWriteResp>("fse_wr_resp");

        spawn ram::RamModel<
            TEST_FSE_RAM_DATA_WIDTH, TEST_FSE_RAM_SIZE, TEST_FSE_RAM_WORD_PARTITION_SIZE>(
            fse_rd_req_r, fse_rd_resp_s, fse_wr_req_r, fse_wr_resp_s);

        let (tmp_rd_req_s, tmp_rd_req_r) = chan<TmpRamReadReq>("tmp_rd_req");
        let (tmp_rd_resp_s, tmp_rd_resp_r) = chan<TmpRamReadResp>("tmp_rd_resp");
        let (tmp_wr_req_s, tmp_wr_req_r) = chan<TmpRamWriteReq>("tmp_wr_req");
        let (tmp_wr_resp_s, tmp_wr_resp_r) = chan<TmpRamWriteResp>("tmp_wr_resp");

        spawn ram::RamModel<
            TEST_TMP_RAM_DATA_WIDTH, TEST_TMP_RAM_SIZE, TEST_TMP_RAM_WORD_PARTITION_SIZE>(
            tmp_rd_req_r, tmp_rd_resp_s, tmp_wr_req_r, tmp_wr_resp_s);

        let (tmp2_rd_req_s, tmp2_rd_req_r) = chan<Tmp2RamReadReq>("tmp2_rd_req");
        let (tmp2_rd_resp_s, tmp2_rd_resp_r) = chan<Tmp2RamReadResp>("tmp2_rd_resp");
        let (tmp2_wr_req_s, tmp2_wr_req_r) = chan<Tmp2RamWriteReq>("tmp2_wr_req");
        let (tmp2_wr_resp_s, tmp2_wr_resp_r) = chan<Tmp2RamWriteResp>("tmp2_wr_resp");

        spawn ram::RamModel<
            TEST_TMP2_RAM_DATA_WIDTH, TEST_TMP2_RAM_SIZE, TEST_TMP2_RAM_WORD_PARTITION_SIZE>(
            tmp2_rd_req_r, tmp2_rd_resp_s, tmp2_wr_req_r, tmp2_wr_resp_s);

        let (fse_table_start_s, fse_table_start_r) = chan<FseStartMsg>("fse_table_start");
        let (fse_table_finish_s, fse_table_finish_r) = chan<()>("fse_table_finish");

        spawn FseTableCreator<
            TEST_DPD_RAM_DATA_WIDTH, TEST_DPD_RAM_ADDR_WIDTH, TEST_DPD_RAM_NUM_PARTITIONS,
            TEST_FSE_RAM_DATA_WIDTH, TEST_FSE_RAM_ADDR_WIDTH, TEST_FSE_RAM_NUM_PARTITIONS,
            TEST_TMP_RAM_DATA_WIDTH, TEST_TMP_RAM_ADDR_WIDTH, TEST_TMP_RAM_NUM_PARTITIONS,
            TEST_TMP2_RAM_DATA_WIDTH, TEST_TMP2_RAM_ADDR_WIDTH, TEST_TMP2_RAM_NUM_PARTITIONS,
        >(
            fse_table_start_r, fse_table_finish_s,
            dpd_rd_req_s, dpd_rd_resp_r,
            fse_wr_req_s, fse_wr_resp_r,
            tmp_rd_req_s, tmp_rd_resp_r, tmp_wr_req_s, tmp_wr_resp_r,
            tmp2_rd_req_s, tmp2_rd_resp_r, tmp2_wr_req_s, tmp2_wr_resp_r
        );

        (
            terminator,
            fse_table_start_s, fse_table_finish_r,
            dpd_wr_req_s, dpd_wr_resp_r,
            fse_rd_req_s, fse_rd_resp_r,
        )
    }

    init {  }

    next(state: ()) {
        const DPD_RAM_REQ_MASK_ALL = std::unsigned_max_value<TEST_DPD_RAM_NUM_PARTITIONS>();
        const FSE_RAM_REQ_MASK_ALL = std::unsigned_max_value<TEST_FSE_RAM_NUM_PARTITIONS>();

        let tok = join();

        let dist_arr_length = array_size(common::FSE_OFFSET_DEFAULT_DIST);
        let accuracy_log = AccuracyLog:5;
        // 1. Fill the DPD Ram with default probability distribution
        let tok = for (idx, tok): (u32, token) in u32:0..dist_arr_length {
            let tok = send(
                tok, dpd_wr_req_s,
                DpdRamWriteReq {
                    addr: checked_cast<uN[TEST_DPD_RAM_ADDR_WIDTH]>(idx),
                    data: checked_cast<uN[TEST_DPD_RAM_DATA_WIDTH]>(
                        std::to_unsigned(common::FSE_OFFSET_DEFAULT_DIST[idx])
                    ),
                    mask: DPD_RAM_REQ_MASK_ALL
                });
            let (tok, _) = recv(tok, dpd_wr_resp_r);
            (tok)
        }(tok);
        // 2. send start request over the fse_table_start_s channel
        let tok = send(tok, fse_table_start_s, FseStartMsg {
            num_symbs: checked_cast<SymbolCount>(dist_arr_length),
            accuracy_log
        });
        // 3. wait for finish response on fse_table_finish_r channel
        let (tok, _) = recv(tok, fse_table_finish_r);
        // 4. Read FSE Ram and verify values
        // (https://datatracker.ietf.org/doc/html/rfc8878#section-appendix.a)
        let code_length = u16:1 << accuracy_log;
        let tok = for (idx, tok): (u16, token) in u16:0..code_length {
            let tok = send(tok, fse_rd_req_s,
                FseRamReadReq {
                    addr: checked_cast<uN[TEST_FSE_RAM_ADDR_WIDTH]>(idx),
                    mask: FSE_RAM_REQ_MASK_ALL
                }
            );
            let (tok, resp) = recv(tok, fse_rd_resp_r);
            let fse_record = bits_to_fse_record(resp.data);
            assert_eq(fse_record.symbol, TEST_OFFSET_CODE_TABLE[idx].symbol);
            assert_eq(fse_record.num_of_bits, TEST_OFFSET_CODE_TABLE[idx].num_of_bits);
            assert_eq(fse_record.base, TEST_OFFSET_CODE_TABLE[idx].base);
            (tok)
        }(tok);

        let tok = send(tok, terminator, true);
    }
}
