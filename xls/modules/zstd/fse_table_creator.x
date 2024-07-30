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
    FINISH = 12,
}

struct FseTableCreatorState {
    status: Status,
    req: bool,
    idx: u16,
    num_symbs: u16,
    curr_symbol: u16,
    state_desc_for_symbol: u16,
    accuracy_log: u16,
    high_threshold: u16,
    inner_for_idx: u16,
    inner_for_range: u16,
    dpd_data: u16,
    pos: u16,
}

struct FseTableRecord {
    symbol: u16,
    num_of_bits: u16,
    base: u16
}

struct FseStartMsg { num_symbs: SymbolCount, accuracy_log: AccuracyLog }

fn fse_record_to_bits(record: FseTableRecord) -> u48 {
    record.base ++ record.num_of_bits ++ record.symbol
}

#[test]
fn test_fse_record_to_bits() {
    let bit = fse_record_to_bits(
        FseTableRecord { symbol: u16:0x0017, num_of_bits: u16:0x0005, base: u16:0x0020 }
    );
    assert_eq(bit, u48:0x0020_0005_0017);
}

fn bits_to_fse_record(bit: u48) -> FseTableRecord {
    FseTableRecord {
        symbol: bit[0:16],
        num_of_bits: bit[16:32],
        base: bit[32:48]
    }
}

#[test]
fn test_bits_to_fse_record() {
    let record = bits_to_fse_record(u48:0x0020_0005_0017);
    assert_eq(record, FseTableRecord { symbol: u16:0x0017, num_of_bits: u16:0x0005, base: u16:0x0020 });
}

proc FseTableCreator<
    // Default Probability Distribution RAM parameters
    DPD_RAM_DATA_WIDTH: u32,
    DPD_RAM_SIZE: u32,
    DPD_RAM_WORD_PARTITION_SIZE: u32,

    // FSE lookup table parameters
    FSE_RAM_DATA_WIDTH: u32,
    FSE_RAM_SIZE: u32,
    FSE_RAM_WORD_PARTITION_SIZE: u32,

    TMP_RAM_DATA_WIDTH: u32,
    TMP_RAM_SIZE: u32,
    TMP_RAM_WORD_PARTITION_SIZE: u32,

    // values computed from other params
    DPD_RAM_ADDR_WIDTH: u32 = {std::clog2(DPD_RAM_SIZE)},
    DPD_RAM_NUM_PARTITIONS: u32 = {ram::num_partitions(DPD_RAM_WORD_PARTITION_SIZE, DPD_RAM_DATA_WIDTH)},
    FSE_RAM_ADDR_WIDTH: u32 = {std::clog2(FSE_RAM_SIZE)},
    FSE_RAM_NUM_PARTITIONS: u32 = {ram::num_partitions(FSE_RAM_WORD_PARTITION_SIZE, FSE_RAM_DATA_WIDTH)},
    TMP_RAM_ADDR_WIDTH: u32 = {std::clog2(TMP_RAM_SIZE)},
    TMP_RAM_NUM_PARTITIONS: u32 = {ram::num_partitions(TMP_RAM_WORD_PARTITION_SIZE, TMP_RAM_DATA_WIDTH)}
> {
    type State = FseTableCreatorState;

    type DpdRamWriteReq = ram::WriteReq<DPD_RAM_ADDR_WIDTH, DPD_RAM_DATA_WIDTH, DPD_RAM_NUM_PARTITIONS>;
    type DpdRamWriteResp = ram::WriteResp;
    type DpdRamReadReq = ram::ReadReq<DPD_RAM_ADDR_WIDTH, DPD_RAM_NUM_PARTITIONS>;
    type DpdRamReadResp = ram::ReadResp<DPD_RAM_DATA_WIDTH>;

    type FseRamWriteReq = ram::WriteReq<FSE_RAM_ADDR_WIDTH, FSE_RAM_DATA_WIDTH, FSE_RAM_NUM_PARTITIONS>;
    type FseRamWriteResp = ram::WriteResp;
    type FseRamReadReq = ram::ReadReq<FSE_RAM_ADDR_WIDTH, FSE_RAM_NUM_PARTITIONS>;
    type FseRamReadResp = ram::ReadResp<FSE_RAM_DATA_WIDTH>;

    type TmpRamWriteReq = ram::WriteReq<TMP_RAM_ADDR_WIDTH, TMP_RAM_DATA_WIDTH, TMP_RAM_NUM_PARTITIONS>;
    type TmpRamWriteResp = ram::WriteResp;
    type TmpRamReadReq = ram::ReadReq<TMP_RAM_ADDR_WIDTH, TMP_RAM_NUM_PARTITIONS>;
    type TmpRamReadResp = ram::ReadResp<TMP_RAM_DATA_WIDTH>;

    type IterCtrl = common::FseTableCreatorCtrl;
    type IterIndex = common::FseTableIndex;

    dpd_rd_req_s: chan<DpdRamReadReq> out;
    dpd_rd_resp_r: chan<DpdRamReadResp> in;
    dpd_wr_req_s: chan<DpdRamWriteReq> out;
    dpd_wr_resp_r: chan<DpdRamWriteResp> in;

    // a request to start creating the FSE decoding table
    fse_table_start_r: chan<FseStartMsg> in;
    // a response with information that the table has been saved to RAM
    fse_table_finish_s: chan<()> out;

    fse_rd_req_s: chan<FseRamReadReq> out;
    fse_rd_resp_r: chan<FseRamReadResp> in;
    fse_wr_req_s: chan<FseRamWriteReq> out;
    fse_wr_resp_r: chan<FseRamWriteResp> in;

    tmp_rd_req_s: chan<TmpRamReadReq> out;
    tmp_rd_resp_r: chan<TmpRamReadResp> in;
    tmp_wr_req_s: chan<TmpRamWriteReq> out;
    tmp_wr_resp_r: chan<TmpRamWriteResp> in;

    it_ctrl_s: chan<IterCtrl> out;
    it_index_r: chan<IterIndex> in;

    config(
        fse_table_start_r: chan<FseStartMsg> in,
        fse_table_finish_s: chan<()> out,

        // RAM with default probability distribution
        dpd_rd_req_s: chan<DpdRamReadReq> out,
        dpd_rd_resp_r: chan<DpdRamReadResp> in,
        dpd_wr_req_s: chan<DpdRamWriteReq> out,
        dpd_wr_resp_r: chan<DpdRamWriteResp> in,

        // Ram with FSE decoding table
        fse_rd_req_s: chan<FseRamReadReq> out,
        fse_rd_resp_r: chan<FseRamReadResp> in,
        fse_wr_req_s: chan<FseRamWriteReq> out,
        fse_wr_resp_r: chan<FseRamWriteResp> in,

        tmp_rd_req_s: chan<TmpRamReadReq> out,
        tmp_rd_resp_r: chan<TmpRamReadResp> in,
        tmp_wr_req_s: chan<TmpRamWriteReq> out,
        tmp_wr_resp_r: chan<TmpRamWriteResp> in
    ) {
        let (it_ctrl_s, it_ctrl_r) = chan<IterCtrl, u32:1>("it_ctrl");
        let (it_index_s, it_index_r) = chan<IterIndex, u32:1>("it_index");
        spawn fse_table_iterator::FseTableIterator(it_ctrl_r, it_index_s);

        (
            dpd_rd_req_s, dpd_rd_resp_r, dpd_wr_req_s, dpd_wr_resp_r,
            fse_table_start_r, fse_table_finish_s,
            fse_rd_req_s, fse_rd_resp_r, fse_wr_req_s, fse_wr_resp_r,
            tmp_rd_req_s, tmp_rd_resp_r, tmp_wr_req_s, tmp_wr_resp_r,
            it_ctrl_s, it_index_r
        )
    }

    init { zero!<State>() }

    next(state: State) {
        const DPD_RAM_REQ_MASK_ALL = std::unsigned_max_value<DPD_RAM_NUM_PARTITIONS>();
        const FSE_RAM_REQ_MASK_ALL = std::unsigned_max_value<FSE_RAM_NUM_PARTITIONS>();
        const FSE_RAM_REQ_MASK_SYMBOL = u3:0b001;
        const TMP_RAM_REQ_MASK_ALL = std::unsigned_max_value<TMP_RAM_NUM_PARTITIONS>();

        // Type definitions repeated because of https://github.com/google/xls/issues/1368
        type DpdRamReadReq = ram::ReadReq<DPD_RAM_ADDR_WIDTH, DPD_RAM_NUM_PARTITIONS>;
        type FseRamWriteReq = ram::WriteReq<FSE_RAM_ADDR_WIDTH, FSE_RAM_DATA_WIDTH, FSE_RAM_NUM_PARTITIONS>;
        type FseRamWriteResp = ram::WriteResp;
        type FseRamReadReq = ram::ReadReq<FSE_RAM_ADDR_WIDTH, FSE_RAM_NUM_PARTITIONS>;
        type TmpRamWriteReq = ram::WriteReq<TMP_RAM_ADDR_WIDTH, TMP_RAM_DATA_WIDTH, TMP_RAM_NUM_PARTITIONS>;
        type TestRamWriteResp = ram::WriteResp;
        type TmpRamReadReq = ram::ReadReq<TMP_RAM_ADDR_WIDTH, TMP_RAM_NUM_PARTITIONS>;

        let tok0 = join();

        // dummy operations on unused channels
        send_if(tok0, dpd_wr_req_s, false, zero!<DpdRamWriteReq>());
        recv_if(tok0, dpd_wr_resp_r, false, zero!<DpdRamWriteResp>());

        let receive_start = (state.status == Status::RECEIVE_START);
        let (tok1, fse_start_msg) = recv_if(tok0, fse_table_start_r, receive_start, zero!<FseStartMsg>());

        let get_dpd_data = state.status == Status::TEST_NEGATIVE_PROB ||
                           state.status == Status::TEST_POSITIVE_PROB ||
                           state.status == Status::HANDLE_POSITIVE_PROB;

        let send_dpd_req = get_dpd_data && state.req;
        let tok_dpd_req = send_if(tok0, dpd_rd_req_s, send_dpd_req,
            DpdRamReadReq {
                addr: checked_cast<uN[DPD_RAM_ADDR_WIDTH]>(state.idx),
                mask: DPD_RAM_REQ_MASK_ALL
            });
        let get_dpd_resp = get_dpd_data && !state.req;
        let (tok_dpd_resp, dpd_resp) = recv_if(tok0, dpd_rd_resp_r, get_dpd_resp, zero!<DpdRamReadResp>());

        let handle_negative_prob_req = state.status == Status::HANDLE_NEGATIVE_PROB;
        let decreased_high_threshold = state.high_threshold - u16:1;
        let index_as_symbol_record = FseTableRecord {
            symbol: state.idx,
            num_of_bits: u16:0,
            base: u16:0
        };
        let fse_record_as_bits = fse_record_to_bits(index_as_symbol_record);
        let fse_wr_req = if handle_negative_prob_req {
            FseRamWriteReq {
                addr: checked_cast<uN[FSE_RAM_ADDR_WIDTH]>(decreased_high_threshold),
                data: checked_cast<uN[FSE_RAM_DATA_WIDTH]>(fse_record_as_bits),
                mask: FSE_RAM_REQ_MASK_SYMBOL
            }
        } else {
            zero!<FseRamWriteReq>()
        };
        let tok3 = send_if(tok0, fse_wr_req_s, handle_negative_prob_req, fse_wr_req);
        let handle_negative_prob_resp = (state.status == Status::HANDLE_NEGATIVE_PROB);
        let (tok3, _) = recv_if(tok3, fse_wr_resp_r, handle_negative_prob_resp, FseRamWriteResp {});

        let tok5 = send_if(tok0, tmp_wr_req_s, handle_negative_prob_req,
            TmpRamWriteReq {
                addr: checked_cast<uN[TMP_RAM_ADDR_WIDTH]>(state.idx),
                data: checked_cast<uN[TMP_RAM_DATA_WIDTH]>(u16:1),
                mask: TMP_RAM_REQ_MASK_ALL
            });
        let (tok5, _) = recv_if(tok5, tmp_wr_resp_r, handle_negative_prob_resp, TestRamWriteResp {});

        let handle_positive_prob_write_state_desc = (state.status == Status::HANDLE_POSITIVE_PROB_WRITE_STATE_DESC);
        let tok6 = send_if(tok0, tmp_wr_req_s, handle_positive_prob_write_state_desc,
            TmpRamWriteReq {
                addr: checked_cast<uN[TMP_RAM_ADDR_WIDTH]>(state.idx),
                data: checked_cast<uN[TMP_RAM_DATA_WIDTH]>(state.dpd_data),
                mask: TMP_RAM_REQ_MASK_ALL
            }
        );
        let (tok6, _) = recv_if(tok6, tmp_wr_resp_r, handle_positive_prob_write_state_desc, TmpRamWriteResp {});

        let inner_for_get_pos = (state.status == Status::INNER_FOR_GET_POS);
        let negative_proba_count = (u16:1 << state.accuracy_log) - state.high_threshold;
        let tok7 = send_if( tok0, it_ctrl_s, inner_for_get_pos,
            IterCtrl {
                accuracy_log: checked_cast<AccuracyLog>(state.accuracy_log),
                negative_proba_count: checked_cast<SymbolCount>(negative_proba_count),
            }
        );
        let (tok7, pos) = recv_if(tok7, it_index_r, inner_for_get_pos, zero!<IterIndex>());

        let inner_for_write_sym = state.status == Status::INNER_FOR_WRITE_SYM;
        let tok4 = send_if( tok0, fse_wr_req_s, inner_for_write_sym,
            FseRamWriteReq {
                addr: checked_cast<uN[FSE_RAM_ADDR_WIDTH]>(state.pos),
                data: checked_cast<uN[FSE_RAM_DATA_WIDTH]>(fse_record_as_bits),
                mask: FSE_RAM_REQ_MASK_SYMBOL
            }
        );

        let (tok4, _) = recv_if(tok4, fse_wr_resp_r, inner_for_write_sym, FseRamWriteResp {});

        let last_for = state.status == Status::LAST_FOR;
        let tok8 = send_if(tok0, fse_rd_req_s, last_for,
            FseRamReadReq {
                addr: checked_cast<uN[FSE_RAM_ADDR_WIDTH]>(state.idx),
                mask: FSE_RAM_REQ_MASK_SYMBOL
            }
        );
        let (tok8, fse_resp) = recv_if(tok8, fse_rd_resp_r, last_for, zero!<FseRamReadResp>());
        let fse_record = bits_to_fse_record(fse_resp.data);

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
            num_of_bits: num_bits,
            base: new_state_base
        };
        let complete_record_as_bits = fse_record_to_bits(complete_record);
        let tok10 = send_if(tok8, fse_wr_req_s, set_state_desc,
            FseRamWriteReq {
                addr: checked_cast<uN[FSE_RAM_ADDR_WIDTH]>(state.idx),
                data: checked_cast<uN[FSE_RAM_DATA_WIDTH]>(complete_record_as_bits),
                mask: FSE_RAM_REQ_MASK_ALL
            }
        );
        let (tok10, _) = recv_if(tok10, fse_wr_resp_r, set_state_desc, FseRamWriteResp {});

        let send_finish = state.status == Status::SEND_FINISH;
        let tok11 = send_if(tok0, fse_table_finish_s, send_finish, ());

        let tok0 = join(
            tok1, tok_dpd_req, tok_dpd_resp, tok3, tok5, tok6, tok7, tok4, tok8, tok9, tok10, tok11);
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
                        num_symbs: checked_cast<u16>(fse_start_msg.num_symbs),
                        accuracy_log: checked_cast<u16>(fse_start_msg.accuracy_log),
                        high_threshold: u16:1 << fse_start_msg.accuracy_log,
                    ..state
                    }
                },
                Status::TEST_NEGATIVE_PROB => {
                    if dpd_resp.data == s16:-1 as u16 {
                        State { status: Status::HANDLE_NEGATIVE_PROB, ..state }
                    } else {
                        let next_idx = state.idx + u16:1;
                        if next_idx < state.num_symbs {
                            State { status: Status::TEST_NEGATIVE_PROB, req: true, idx: next_idx, ..state }
                        } else {
                            State { status: Status::TEST_POSITIVE_PROB, idx: u16:0, ..state }
                        }
                    }
                },
                Status::HANDLE_NEGATIVE_PROB => {
                    // https://github.com/facebook/zstd/blob/9f42fa0a043aa389534cf10ff086976c4c6b10a6/doc/educational_decoder/zstd_decompress.c#L2143-L2146
                    let next_idx = state.idx + u16:1;
                    if next_idx < state.num_symbs {
                        State { status: Status::TEST_NEGATIVE_PROB, req: true, idx: next_idx, high_threshold: decreased_high_threshold, ..state }
                    } else {
                        State { status: Status::TEST_POSITIVE_PROB, req: true, idx: u16:0, high_threshold: decreased_high_threshold, ..state }
                    }
                },
                Status::TEST_POSITIVE_PROB => {
                    if dpd_resp.data as s16 > s16:0 {
                        State { status: Status::HANDLE_POSITIVE_PROB, req: true, ..state }
                    } else {
                        let next_idx = state.idx + u16:1;
                        if next_idx < state.num_symbs {
                            State { status: Status::TEST_POSITIVE_PROB, req: true, idx: next_idx, ..state }
                        } else {
                            State { status: Status::LAST_FOR, idx: u16:0, ..state }
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
                        let next_idx = state.idx + u16:1;
                        if next_idx < state.num_symbs {
                            State { status: Status::TEST_POSITIVE_PROB, req: true, idx: next_idx, ..state }
                        } else {
                            State { status: Status::LAST_FOR, idx: u16:0, ..state }
                        }
                    }
                },
                Status::LAST_FOR => {
                    // https://github.com/facebook/zstd/blob/9f42fa0a043aa389534cf10ff086976c4c6b10a6/doc/educational_decoder/zstd_decompress.c#L2183
                    State { status: Status::GET_STATE_DESC, curr_symbol: fse_record.symbol, ..state }
                },
                Status::GET_STATE_DESC => {
                    // https://github.com/facebook/zstd/blob/9f42fa0a043aa389534cf10ff086976c4c6b10a6/doc/educational_decoder/zstd_decompress.c#L2184
                    State { status: Status::SET_STATE_DESC, state_desc_for_symbol: tmp_resp.data, ..state }
                },
                Status::SET_STATE_DESC => {
                    let next_idx = state.idx + u16:1;
                    if next_idx < size {
                        State { status: Status::LAST_FOR, idx: next_idx, ..state }
                    } else {
                        State { status: Status::SEND_FINISH, ..state }
                    }
                },
                Status::SEND_FINISH => { State { status: Status::FINISH, ..state } },
                Status::FINISH => { state },
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

const TEST_FSE_RAM_DATA_WIDTH = u32:48;
const TEST_FSE_RAM_SIZE = u32:256;
const TEST_FSE_RAM_ADDR_WIDTH = std::clog2(TEST_FSE_RAM_SIZE);
const TEST_FSE_RAM_WORD_PARTITION_SIZE = TEST_FSE_RAM_DATA_WIDTH / u32:3;
const TEST_FSE_RAM_NUM_PARTITIONS = ram::num_partitions(
    TEST_FSE_RAM_WORD_PARTITION_SIZE, TEST_FSE_RAM_DATA_WIDTH);

const TEST_TMP_RAM_DATA_WIDTH = u32:16;
const TEST_TMP_RAM_SIZE = u32:256;
const TEST_TMP_RAM_ADDR_WIDTH = std::clog2(TEST_TMP_RAM_SIZE);
const TEST_TMP_RAM_WORD_PARTITION_SIZE = TEST_TMP_RAM_DATA_WIDTH;
const TEST_TMP_RAM_NUM_PARTITIONS = ram::num_partitions(
    TEST_TMP_RAM_WORD_PARTITION_SIZE, TEST_TMP_RAM_DATA_WIDTH);

proc FseTableCreatorInst {
    type DpdRamReadReq = ram::ReadReq<TEST_DPD_RAM_ADDR_WIDTH, TEST_DPD_RAM_NUM_PARTITIONS>;
    type DpdRamReadResp = ram::ReadResp<TEST_DPD_RAM_DATA_WIDTH>;
    type DpdRamWriteReq = ram::WriteReq<TEST_DPD_RAM_ADDR_WIDTH, TEST_DPD_RAM_DATA_WIDTH, TEST_DPD_RAM_NUM_PARTITIONS>;
    type DpdRamWriteResp = ram::WriteResp;

    type FseRamReadReq = ram::ReadReq<TEST_FSE_RAM_ADDR_WIDTH, TEST_FSE_RAM_NUM_PARTITIONS>;
    type FseRamReadResp = ram::ReadResp<TEST_FSE_RAM_DATA_WIDTH>;
    type FseRamWriteReq = ram::WriteReq<TEST_FSE_RAM_ADDR_WIDTH, TEST_FSE_RAM_DATA_WIDTH, TEST_FSE_RAM_NUM_PARTITIONS>;
    type FseRamWriteResp = ram::WriteResp;

    type TmpRamWriteReq = ram::WriteReq<TEST_TMP_RAM_ADDR_WIDTH, TEST_TMP_RAM_DATA_WIDTH, TEST_TMP_RAM_NUM_PARTITIONS>;
    type TmpRamWriteResp = ram::WriteResp;
    type TmpRamReadReq = ram::ReadReq<TEST_TMP_RAM_ADDR_WIDTH, TEST_TMP_RAM_NUM_PARTITIONS>;
    type TmpRamReadResp = ram::ReadResp<TEST_TMP_RAM_DATA_WIDTH>;

    dpd_rd_req_s: chan<DpdRamReadReq> out;
    dpd_rd_resp_r: chan<DpdRamReadResp> in;
    dpd_wr_req_s: chan<DpdRamWriteReq> out;
    dpd_wr_resp_r: chan<DpdRamWriteResp> in;

    fse_table_start_r: chan<FseStartMsg> in;
    fse_table_finish_s: chan<()> out;

    fse_rd_req_s: chan<FseRamReadReq> out;
    fse_rd_resp_r: chan<FseRamReadResp> in;
    fse_wr_req_s: chan<FseRamWriteReq> out;
    fse_wr_resp_r: chan<FseRamWriteResp> in;

    tmp_rd_req_s: chan<TmpRamReadReq> out;
    tmp_rd_resp_r: chan<TmpRamReadResp> in;
    tmp_wr_req_s: chan<TmpRamWriteReq> out;
    tmp_wr_resp_r: chan<TmpRamWriteResp> in;

    config(
        fse_table_start_r: chan<FseStartMsg> in,
        fse_table_finish_s: chan<()> out,

        dpd_rd_req_s: chan<DpdRamReadReq> out,
        dpd_rd_resp_r: chan<DpdRamReadResp> in,
        dpd_wr_req_s: chan<DpdRamWriteReq> out,
        dpd_wr_resp_r: chan<DpdRamWriteResp> in,

        fse_rd_req_s: chan<FseRamReadReq> out,
        fse_rd_resp_r: chan<FseRamReadResp> in,
        fse_wr_req_s: chan<FseRamWriteReq> out,
        fse_wr_resp_r: chan<FseRamWriteResp> in,

        tmp_rd_req_s: chan<TmpRamReadReq> out,
        tmp_rd_resp_r: chan<TmpRamReadResp> in,
        tmp_wr_req_s: chan<TmpRamWriteReq> out,
        tmp_wr_resp_r: chan<TmpRamWriteResp> in
    ) {
        spawn FseTableCreator<
            TEST_DPD_RAM_DATA_WIDTH, TEST_DPD_RAM_SIZE, TEST_DPD_RAM_WORD_PARTITION_SIZE,
            TEST_FSE_RAM_DATA_WIDTH, TEST_FSE_RAM_SIZE, TEST_FSE_RAM_WORD_PARTITION_SIZE,
            TEST_TMP_RAM_DATA_WIDTH, TEST_TMP_RAM_SIZE, TEST_TMP_RAM_WORD_PARTITION_SIZE
        >(
            fse_table_start_r, fse_table_finish_s,
            dpd_rd_req_s, dpd_rd_resp_r, dpd_wr_req_s, dpd_wr_resp_r,
            fse_rd_req_s, fse_rd_resp_r, fse_wr_req_s, fse_wr_resp_r,
            tmp_rd_req_s, tmp_rd_resp_r, tmp_wr_req_s, tmp_wr_resp_r
        );
        (
            dpd_rd_req_s, dpd_rd_resp_r, dpd_wr_req_s, dpd_wr_resp_r,
            fse_table_start_r, fse_table_finish_s,
            fse_rd_req_s, fse_rd_resp_r, fse_wr_req_s, fse_wr_resp_r,
            tmp_rd_req_s, tmp_rd_resp_r, tmp_wr_req_s, tmp_wr_resp_r
        )
    }

    init {  }

    next(state: ()) {  }
}

const TEST_OFFSET_CODE_TABLE = FseTableRecord[32]:[
    FseTableRecord { symbol: u16:0, num_of_bits: u16:5, base: u16:0 },
    FseTableRecord { symbol: u16:6, num_of_bits: u16:4, base: u16:0 },
    FseTableRecord { symbol: u16:9, num_of_bits: u16:5, base: u16:0 },
    FseTableRecord { symbol: u16:15, num_of_bits: u16:5, base: u16:0 },
    FseTableRecord { symbol: u16:21, num_of_bits: u16:5, base: u16:0 },
    FseTableRecord { symbol: u16:3, num_of_bits: u16:5, base: u16:0 },
    FseTableRecord { symbol: u16:7, num_of_bits: u16:4, base: u16:0 },
    FseTableRecord { symbol: u16:12, num_of_bits: u16:5, base: u16:0 },
    FseTableRecord { symbol: u16:18, num_of_bits: u16:5, base: u16:0 },
    FseTableRecord { symbol: u16:23, num_of_bits: u16:5, base: u16:0 },
    FseTableRecord { symbol: u16:5, num_of_bits: u16:5, base: u16:0 },
    FseTableRecord { symbol: u16:8, num_of_bits: u16:4, base: u16:0 },
    FseTableRecord { symbol: u16:14, num_of_bits: u16:5, base: u16:0 },
    FseTableRecord { symbol: u16:20, num_of_bits: u16:5, base: u16:0 },
    FseTableRecord { symbol: u16:2, num_of_bits: u16:5, base: u16:0 },
    FseTableRecord { symbol: u16:7, num_of_bits: u16:4, base: u16:16 },
    FseTableRecord { symbol: u16:11, num_of_bits: u16:5, base: u16:0 },
    FseTableRecord { symbol: u16:17, num_of_bits: u16:5, base: u16:0 },
    FseTableRecord { symbol: u16:22, num_of_bits: u16:5, base: u16:0 },
    FseTableRecord { symbol: u16:4, num_of_bits: u16:5, base: u16:0 },
    FseTableRecord { symbol: u16:8, num_of_bits: u16:4, base: u16:16 },
    FseTableRecord { symbol: u16:13, num_of_bits: u16:5, base: u16:0 },
    FseTableRecord { symbol: u16:19, num_of_bits: u16:5, base: u16:0 },
    FseTableRecord { symbol: u16:1, num_of_bits: u16:5, base: u16:0 },
    FseTableRecord { symbol: u16:6, num_of_bits: u16:4, base: u16:16 },
    FseTableRecord { symbol: u16:10, num_of_bits: u16:5, base: u16:0 },
    FseTableRecord { symbol: u16:16, num_of_bits: u16:5, base: u16:0 },
    FseTableRecord { symbol: u16:28, num_of_bits: u16:5, base: u16:0 },
    FseTableRecord { symbol: u16:27, num_of_bits: u16:5, base: u16:0 },
    FseTableRecord { symbol: u16:26, num_of_bits: u16:5, base: u16:0 },
    FseTableRecord { symbol: u16:25, num_of_bits: u16:5, base: u16:0 },
    FseTableRecord { symbol: u16:24, num_of_bits: u16:5, base: u16:0 },
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

    terminator: chan<bool> out;
    fse_table_start_s: chan<FseStartMsg> out;
    fse_table_finish_r: chan<()> in;

    dpd_rd_req_s: chan<DpdRamReadReq> out;
    dpd_rd_resp_r: chan<DpdRamReadResp> in;
    dpd_wr_req_s: chan<DpdRamWriteReq> out;
    dpd_wr_resp_r: chan<DpdRamWriteResp> in;

    fse_rd_req_s: chan<FseRamReadReq> out;
    fse_rd_resp_r: chan<FseRamReadResp> in;
    fse_wr_req_s: chan<FseRamWriteReq> out;
    fse_wr_resp_r: chan<FseRamWriteResp> in;

    tmp_rd_req_s: chan<TmpRamReadReq> out;
    tmp_rd_resp_r: chan<TmpRamReadResp> in;
    tmp_wr_req_s: chan<TmpRamWriteReq> out;
    tmp_wr_resp_r: chan<TmpRamWriteResp> in;

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

        let (fse_table_start_s, fse_table_start_r) = chan<FseStartMsg>("fse_table_start");
        let (fse_table_finish_s, fse_table_finish_r) = chan<()>("fse_table_finish");

        spawn FseTableCreator<
            TEST_DPD_RAM_DATA_WIDTH, TEST_DPD_RAM_SIZE, TEST_DPD_RAM_WORD_PARTITION_SIZE,
            TEST_FSE_RAM_DATA_WIDTH, TEST_FSE_RAM_SIZE, TEST_FSE_RAM_WORD_PARTITION_SIZE,
            TEST_TMP_RAM_DATA_WIDTH, TEST_TMP_RAM_SIZE, TEST_TMP_RAM_WORD_PARTITION_SIZE
        >(
            fse_table_start_r, fse_table_finish_s,
            dpd_rd_req_s, dpd_rd_resp_r, dpd_wr_req_s, dpd_wr_resp_r,
            fse_rd_req_s, fse_rd_resp_r, fse_wr_req_s, fse_wr_resp_r,
            tmp_rd_req_s, tmp_rd_resp_r, tmp_wr_req_s, tmp_wr_resp_r
        );

        (
            terminator,
            fse_table_start_s, fse_table_finish_r,
            dpd_rd_req_s, dpd_rd_resp_r, dpd_wr_req_s, dpd_wr_resp_r,
            fse_rd_req_s, fse_rd_resp_r, fse_wr_req_s, fse_wr_resp_r,
            tmp_rd_req_s, tmp_rd_resp_r, tmp_wr_req_s, tmp_wr_resp_r,
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
        let tok = for (idx, tok): (u32, token) in range(u32:0, dist_arr_length) {
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
        let tok = for (idx, tok): (u16, token) in range(u16:0, code_length) {
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
