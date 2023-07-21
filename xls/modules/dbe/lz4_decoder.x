// Copyright 2023 The XLS Authors
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

import std
import xls.examples.ram as ram
import xls.modules.dbe.common as dbe

type Mark = dbe::Mark;
type TokenKind = dbe::TokenKind;
type Token = dbe::Token;
type PlainData = dbe::PlainData;
type RamWriteReq = ram::WriteReq;
type RamWriteResp = ram::WriteResp;
type RamReadReq = ram::ReadReq;
type RamReadResp = ram::ReadResp;


fn RamZeroReadReq<ADDR_WIDTH: u32, NUM_PARTITIONS: u32>
    () -> RamReadReq<ADDR_WIDTH, NUM_PARTITIONS> {
    RamReadReq<ADDR_WIDTH, NUM_PARTITIONS> {
        addr:uN[ADDR_WIDTH]:0,
        mask:uN[NUM_PARTITIONS]:0,
    }
}

fn RamZeroReadResp<DATA_WIDTH: u32>
    () -> RamReadResp<DATA_WIDTH> {
    RamReadResp<DATA_WIDTH> {
        data:uN[DATA_WIDTH]:0,
    }
}

fn RamZeroWriteReq<ADDR_WIDTH: u32, DATA_WIDTH: u32, NUM_PARTITIONS: u32>
    () -> RamWriteReq<ADDR_WIDTH, DATA_WIDTH, NUM_PARTITIONS> {
    RamWriteReq<ADDR_WIDTH, DATA_WIDTH, NUM_PARTITIONS> {
        addr:uN[ADDR_WIDTH]:0,
        data:uN[DATA_WIDTH]:0,
        mask:uN[NUM_PARTITIONS]:0,
    }
}

fn RamZeroWriteResp() -> RamWriteResp {
    RamWriteResp {}
}

/// The behavior of our decoder is defined as follows.
/// 
/// A. History buffer is a buffer of (1<<PTR_WIDTH) cells in size, each cell
///    is capable of storing one symbol (uN[SYM_WIDTH] number).
/// B. Cells in the buffer are addressed using indices, which take values from
///    0 to (1<<PTR_WIDTH)-1.
/// C. The history buffer has a write pointer and a read pointer associated
///    with it.
/// D. Each pointer can store a single cell index value in range from
///    0 to (1<<PTR_WIDTH)-1 inclusive.
/// E. Initial values of write and read pointers are unspecified.
/// F. The pointers can be incremented and decremented. Whenever as a result
///    of such operation the pointer value goes outside of the range specified
///    above, it is wrapped around to the other side of the range using rules
///    of unsigned integer binary overflow / underflow.
/// G. Whenever a symbol is "written to" the history buffer, the symbol is
///    written to the cell pointed to by the write pointer, and after that the
///    write pointer is incremented by 1.
///
/// H. The operation of decoder is described as follows:
///     1. Consume a Token from the input stream.
///     2. Perform actions specified in 'I'.
///     3. Repeat from step 1.
///   
/// I. The actions performed depend on the value of `kind` field of a received
///    Token object:
///    LITERAL:
///     1. Write `lt_sym` symbol to the history buffer.
///     2. Send PlainData object to the output data stream, wherein `sym` is
///        equal to `lt_sym` and `last` flag is equal to this Token's `last`
///        flag.
///    COPY_POINTER:
///     1. Set read pointer to be equal to the write pointer, then decrement
///        it by `cp_off + 1`.
///     2. Send PlainData object to the output data stream with `sym` set to
///        the symbol pointed to by the read pointer, and `last` flag set iff
///        the Token's `last` is set and this is the last symbol to be emitted
///        for this Token.
///     3. Write symbol pointed to by the read pointer to the history buffer.
///     4. Increment read pointer by 1.
///     5. Repeat steps 2-4 `cp_count` times.
///         NOTE:
///             Total number of executions of steps 2-4 is `cp_count+1`.
///             - If `cp_count` is 0, steps 2-4 are executed once
///             unconditionally, and then they are repeated 0 times (not
///             executed), so the total number of executions is 1.
///             - If `cp_count` is 2, steps 2-4 are first executed once
///             unconditionally, and then they are repeated 2 times, so
///             the total number of executions is 3.
///
/// To make decoder useful, it has to be complemented by the corresponding
/// encoder that is defined as follows (vagueness is intended):
/// 
/// - Given a sequence of PlainData objects, the encoder should produce such
///   a sequence of Token objects that the decoder reading that sequence
///   will produce a sequence of PlainData objects precisely matching the
///   original sequence.
///


struct State<SYM_WIDTH: u32, PTR_WIDTH: u32, CNT_WIDTH: u32,
                HB_SIZE: u32 = {u32:1<<PTR_WIDTH}> {
    // Write pointer - next location _to be written_
    wr_ptr: uN[PTR_WIDTH],
    // Read pointer - next location _to be read_
    rd_ptr: uN[PTR_WIDTH],
    // Read counter - remaining number of symbols to be read out from the
    // history buffer
    rd_ctr_rem: uN[CNT_WIDTH],
    // Whether we're in the ERROR state
    err: bool,
    // Helps to track which cells in HB are written and which ones are not
    // (if some CP token points to an unwritten cell, we detect that and
    // generate an error, as otherwise that will make decoder output garbage
    // data that may leak some stale data from previous blocks)
    hb_all_valid: bool,
}

pub proc decoder_base<SYM_WIDTH: u32, PTR_WIDTH: u32, CNT_WIDTH: u32> {
    i_encoded: chan<Token<SYM_WIDTH, PTR_WIDTH, CNT_WIDTH>> in;
    o_data: chan<PlainData<SYM_WIDTH>> out;

    /// History buffer RAM
    /// Size: (1<<PTR_WIDTH) x SYM_WIDTH
    o_ram_hb_wr_req: chan<ram::WriteReq<PTR_WIDTH, SYM_WIDTH, 1>> out;
    i_ram_hb_wr_resp: chan<ram::WriteResp> in;
    o_ram_hb_rd_req: chan<ram::ReadReq<PTR_WIDTH, 1>> out;
    i_ram_hb_rd_resp: chan<ram::ReadResp<SYM_WIDTH>> in;

    init {
        State<SYM_WIDTH, PTR_WIDTH, CNT_WIDTH> {
            wr_ptr: uN[PTR_WIDTH]:0,
            rd_ptr: uN[PTR_WIDTH]:0,
            rd_ctr_rem: uN[CNT_WIDTH]:0,
            err: false,
            hb_all_valid: false,
        }
    }

    config (
        i_encoded: chan<Token<SYM_WIDTH, PTR_WIDTH, CNT_WIDTH>> in,
        o_symb: chan<PlainData<SYM_WIDTH>> out,
        // RAM
        o_ram_hb_wr_req: chan<ram::WriteReq<PTR_WIDTH, SYM_WIDTH, 1>> out,
        i_ram_hb_wr_resp: chan<ram::WriteResp> in,
        o_ram_hb_rd_req: chan<ram::ReadReq<PTR_WIDTH, 1>> out,
        i_ram_hb_rd_resp: chan<ram::ReadResp<SYM_WIDTH>> in,
    ) {
        (
            i_encoded, o_symb,
            o_ram_hb_wr_req, i_ram_hb_wr_resp,
            o_ram_hb_rd_req, i_ram_hb_rd_resp,
        )
    }

    next (tok: token, state: State<SYM_WIDTH, PTR_WIDTH, CNT_WIDTH>) {
        let is_copying = state.rd_ctr_rem != uN[CNT_WIDTH]:0;
        let do_recv = !is_copying;

        // Receive token
        let (tok, rx) = recv_if(tok, i_encoded, do_recv,
            dbe::zero_token<SYM_WIDTH, PTR_WIDTH, CNT_WIDTH>());

        let rx_lt = do_recv && rx.kind == TokenKind::LT;
        let rx_cp = do_recv && rx.kind == TokenKind::CP;
        let rx_mark = do_recv && rx.kind == TokenKind::MK;
        let rx_end = rx_mark && rx.mark == Mark::END;
        let rx_error = rx_mark && dbe::is_error(rx.mark);
        let rx_reset = rx_mark && rx.mark == Mark::RESET;
        let rx_unexpected_mark = rx_mark && !(rx_end || rx_reset || rx_error);

        // Are we going to output a literal symbol or a symbol from HB?
        let emit_sym_lit = rx_lt;
        let emit_sym_from_hb = rx_cp || is_copying;

        // Reset, set or increment read pointer
        let rd_ptr = if rx_reset {
            uN[PTR_WIDTH]:0
        } else if rx_cp {
            state.wr_ptr - rx.cp_off - uN[PTR_WIDTH]:1
        } else if is_copying {
            state.rd_ptr + uN[PTR_WIDTH]:1
        } else {
            state.rd_ptr
        };

        // Check whether rd_ptr points to an unwritten HB cell
        let rd_ptr_valid = (rd_ptr < state.wr_ptr) || state.hb_all_valid;

        // HB READ - read the symbol from history buffer
        let do_hb_read = emit_sym_from_hb && rd_ptr_valid;
        let tok = send_if(tok, o_ram_hb_rd_req, do_hb_read,
            ram::ReadWordReq<u32:1>(rd_ptr));
        let (tok, hb_rd) = recv_if(tok, i_ram_hb_rd_resp, do_hb_read,
            RamZeroReadResp<SYM_WIDTH>());

        // Generate ERROR_INVAL_CP when we should've read from HB but
        // we couldn't because CP points to an invalid location
        let err_inval_cp = emit_sym_from_hb && !do_hb_read;

        // Send decoded data symbol or Mark
        let zero_data = dbe::zero_plain_data<SYM_WIDTH>();
        let (data_valid, data) = if rx_reset {
            // Propagate RESET in all states
            (true, PlainData {
                is_mark: true,
                mark: Mark::RESET,
                ..zero_data
            })
        } else if state.err {
            // Do not send anything else while in an error state
            (false, zero_data)
        } else if rx_error || rx_end {
            // Propagate ERROR and END tokens
            (true, PlainData {
                is_mark: true,
                mark: rx.mark,
                ..zero_data
            })
        } else if rx_unexpected_mark {
            // Generate ERROR_BAD_MARK
            (true, PlainData {
                is_mark: true,
                mark: Mark::ERROR_BAD_MARK,
                ..zero_data
            })
        } else if err_inval_cp {
            // Generate ERROR_INVAL_CP
            (true, PlainData {
                is_mark: true,
                mark: Mark::ERROR_INVAL_CP,
                ..zero_data
            })
        } else if emit_sym_lit {
            // Replicate symbol from LT token
            (true, PlainData {
                is_mark: false,
                data: rx.lt_sym,
                ..zero_data
            })
        } else if emit_sym_from_hb {
            // Replicate symbol from HB
            (true, PlainData {
                is_mark: false,
                data: hb_rd.data,
                ..zero_data
            })
        } else {
            (false, zero_data)
        };
        let tok = send_if(tok, o_data, data_valid, data);

        // HB WRITE - write emitted symbol to the history buffer
        let do_hb_write = data_valid && !data.is_mark;
        let tok = send_if(tok, o_ram_hb_wr_req, do_hb_write,
            ram::WriteWordReq<u32:1>(state.wr_ptr, data.data));
        let (tok, _) = recv_if(tok, i_ram_hb_wr_resp, do_hb_write,
            RamZeroWriteResp());

        // Reset/increment write pointer
        let wr_ptr = if rx_reset {
            uN[PTR_WIDTH]:0
        } else if do_hb_write {
            state.wr_ptr + uN[PTR_WIDTH]:1
        } else {
            state.wr_ptr
        };

        // When write pointer wraps over to 0 after the increment,
        // that means we've filled the complete HB, so set a corresponding flag
        let hb_all_valid = if rx_reset {
            false
        } else if do_hb_write && wr_ptr == uN[PTR_WIDTH]:0 {
            true
        } else {
            state.hb_all_valid
        };

        // Read count set & decrement
        let rd_ctr_rem = if rx_reset {
            uN[CNT_WIDTH]:0
        } else if rx_cp {
            rx.cp_cnt
        } else if is_copying {
            state.rd_ctr_rem - uN[CNT_WIDTH]:1
        } else {
            state.rd_ctr_rem
        };

        // Enter/exit error state
        let err = if rx_reset {
            false
        } else if rx_error || rx_unexpected_mark || err_inval_cp {
            true
        } else {
            state.err
        };

        State{
            wr_ptr: wr_ptr,
            rd_ptr: rd_ptr,
            rd_ctr_rem: rd_ctr_rem,
            hb_all_valid: hb_all_valid,
            err: err,
        }
    }
}

/// Version of `decoder` with embedded RamModel
/// Intended to be used only for tests
pub proc decoder_base_modelram<SYM_WIDTH: u32, PTR_WIDTH: u32, CNT_WIDTH: u32> {
    init{()}

    config (
        i_encoded: chan<Token<SYM_WIDTH, PTR_WIDTH, CNT_WIDTH>> in,
        o_data: chan<PlainData<SYM_WIDTH>> out,
    ) {
        let (o_hb_wr_req, i_hb_wr_req) =
            chan<ram::WriteReq<PTR_WIDTH, SYM_WIDTH, 1>>;
        let (o_hb_wr_resp, i_hb_wr_resp) =
            chan<ram::WriteResp>;
        let (o_hb_rd_req, i_hb_rd_req) =
            chan<ram::ReadReq<PTR_WIDTH, 1>>;
        let (o_hb_rd_resp, i_hb_rd_resp) =
            chan<ram::ReadResp<SYM_WIDTH>>;
        
        spawn ram::RamModel<SYM_WIDTH, {u32:1<<PTR_WIDTH}, SYM_WIDTH>
            (i_hb_rd_req, o_hb_rd_resp, i_hb_wr_req, o_hb_wr_resp);

        spawn decoder_base<SYM_WIDTH, PTR_WIDTH, CNT_WIDTH>
            (
                i_encoded, o_data,
                o_hb_wr_req, i_hb_wr_resp,
                o_hb_rd_req, i_hb_rd_resp,
            );
    }

    next (tok: token, state: ()) {
    }
}

/// Version of `decoder_base` compatible with classic LZ4 algorithm
pub const LZ4C_SYM_WIDTH = u32:8;
pub const LZ4C_PTR_WIDTH = u32:16;
pub const LZ4C_CNT_WIDTH = u32:16;

pub proc decoder {
    init{}

    config (
        i_encoded:
            chan<Token<LZ4C_SYM_WIDTH, LZ4C_PTR_WIDTH, LZ4C_CNT_WIDTH>> in,
        o_data: chan<PlainData<LZ4C_SYM_WIDTH>> out,
        o_ram_hb_wr_req:
            chan<ram::WriteReq<LZ4C_PTR_WIDTH, LZ4C_SYM_WIDTH, 1>> out,
        i_ram_hb_wr_resp: chan<ram::WriteResp> in,
        o_ram_hb_rd_req: chan<ram::ReadReq<LZ4C_PTR_WIDTH, 1>> out,
        i_ram_hb_rd_resp: chan<ram::ReadResp<LZ4C_SYM_WIDTH>> in,
    ) {
        spawn decoder_base
            <LZ4C_SYM_WIDTH, LZ4C_PTR_WIDTH, LZ4C_CNT_WIDTH>
            (
                i_encoded, o_data,
                o_ram_hb_wr_req, i_ram_hb_wr_resp,
                o_ram_hb_rd_req, i_ram_hb_rd_resp,
            );
    }

    next(tok: token, st: ()) {
    }
}

/// Version of `decoder` with embedded RamModel
/// Intended to be used only for tests
pub proc decoder_modelram {
    init{()}

    config (
        i_encoded:
            chan<Token<LZ4C_SYM_WIDTH, LZ4C_PTR_WIDTH, LZ4C_CNT_WIDTH>> in,
        o_symb: chan<PlainData<LZ4C_SYM_WIDTH>> out,
    ) {
        let (o_hb_wr_req, i_hb_wr_req) =
            chan<ram::WriteReq<LZ4C_PTR_WIDTH, LZ4C_SYM_WIDTH, 1>>;
        let (o_hb_wr_resp, i_hb_wr_resp) =
            chan<ram::WriteResp>;
        let (o_hb_rd_req, i_hb_rd_req) =
            chan<ram::ReadReq<LZ4C_PTR_WIDTH, 1>>;
        let (o_hb_rd_resp, i_hb_rd_resp) =
            chan<ram::ReadResp<LZ4C_SYM_WIDTH>>;
        
        spawn ram::RamModel
            <LZ4C_SYM_WIDTH, {u32:1<<LZ4C_PTR_WIDTH}, LZ4C_SYM_WIDTH>
            (i_hb_rd_req, o_hb_rd_resp, i_hb_wr_req, o_hb_wr_resp);

        spawn decoder
            (
                i_encoded, o_symb,
                o_hb_wr_req, i_hb_wr_resp,
                o_hb_rd_req, i_hb_rd_resp,
            );
    }

    next (tok: token, state: ()) {
    }
}


///
/// Tests
///
import xls.modules.dbe.common_test as test

const TST_SYM_WIDTH = u32:4;
const TST_PTR_WIDTH = u32:3;
const TST_CNT_WIDTH = u32:4;


// Reference input
const TST_SIMPLE_INPUT_LEN = u32:32;
const TST_SIMPLE_INPUT = Token<TST_SYM_WIDTH, TST_PTR_WIDTH, TST_CNT_WIDTH>[TST_SIMPLE_INPUT_LEN]: [
    Token{kind: TokenKind::LT, lt_sym: uN[4]:12, cp_off: uN[3]:0, cp_cnt: uN[4]:0, mark: Mark::NONE},
    Token{kind: TokenKind::LT, lt_sym: uN[4]: 1, cp_off: uN[3]:0, cp_cnt: uN[4]:0, mark: Mark::NONE},
    Token{kind: TokenKind::LT, lt_sym: uN[4]:15, cp_off: uN[3]:0, cp_cnt: uN[4]:0, mark: Mark::NONE},
    Token{kind: TokenKind::LT, lt_sym: uN[4]: 9, cp_off: uN[3]:0, cp_cnt: uN[4]:0, mark: Mark::NONE},
    Token{kind: TokenKind::LT, lt_sym: uN[4]:11, cp_off: uN[3]:0, cp_cnt: uN[4]:0, mark: Mark::NONE},
    Token{kind: TokenKind::CP, lt_sym: uN[4]:0, cp_off: uN[3]: 1, cp_cnt: uN[4]: 1, mark: Mark::NONE},
    Token{kind: TokenKind::CP, lt_sym: uN[4]:0, cp_off: uN[3]: 0, cp_cnt: uN[4]: 2, mark: Mark::NONE},
    Token{kind: TokenKind::CP, lt_sym: uN[4]:0, cp_off: uN[3]: 4, cp_cnt: uN[4]: 0, mark: Mark::NONE},
    Token{kind: TokenKind::LT, lt_sym: uN[4]:15, cp_off: uN[3]:0, cp_cnt: uN[4]:0, mark: Mark::NONE},
    Token{kind: TokenKind::CP, lt_sym: uN[4]:0, cp_off: uN[3]: 1, cp_cnt: uN[4]: 0, mark: Mark::NONE},
    Token{kind: TokenKind::CP, lt_sym: uN[4]:0, cp_off: uN[3]: 2, cp_cnt: uN[4]:13, mark: Mark::NONE},
    Token{kind: TokenKind::LT, lt_sym: uN[4]: 1, cp_off: uN[3]:0, cp_cnt: uN[4]:0, mark: Mark::NONE},
    Token{kind: TokenKind::CP, lt_sym: uN[4]:0, cp_off: uN[3]: 5, cp_cnt: uN[4]: 0, mark: Mark::NONE},
    Token{kind: TokenKind::CP, lt_sym: uN[4]:0, cp_off: uN[3]: 5, cp_cnt: uN[4]: 2, mark: Mark::NONE},
    Token{kind: TokenKind::LT, lt_sym: uN[4]: 7, cp_off: uN[3]:0, cp_cnt: uN[4]:0, mark: Mark::NONE},
    Token{kind: TokenKind::LT, lt_sym: uN[4]: 2, cp_off: uN[3]:0, cp_cnt: uN[4]:0, mark: Mark::NONE},
    Token{kind: TokenKind::CP, lt_sym: uN[4]:0, cp_off: uN[3]: 0, cp_cnt: uN[4]:11, mark: Mark::NONE},
    Token{kind: TokenKind::CP, lt_sym: uN[4]:0, cp_off: uN[3]: 1, cp_cnt: uN[4]: 6, mark: Mark::NONE},
    Token{kind: TokenKind::CP, lt_sym: uN[4]:0, cp_off: uN[3]: 5, cp_cnt: uN[4]:15, mark: Mark::NONE},
    Token{kind: TokenKind::LT, lt_sym: uN[4]: 3, cp_off: uN[3]:0, cp_cnt: uN[4]:0, mark: Mark::NONE},
    Token{kind: TokenKind::LT, lt_sym: uN[4]: 9, cp_off: uN[3]:0, cp_cnt: uN[4]:0, mark: Mark::NONE},
    Token{kind: TokenKind::CP, lt_sym: uN[4]:0, cp_off: uN[3]: 5, cp_cnt: uN[4]: 1, mark: Mark::NONE},
    Token{kind: TokenKind::CP, lt_sym: uN[4]:0, cp_off: uN[3]: 3, cp_cnt: uN[4]:13, mark: Mark::NONE},
    Token{kind: TokenKind::LT, lt_sym: uN[4]:14, cp_off: uN[3]:0, cp_cnt: uN[4]:0, mark: Mark::NONE},
    Token{kind: TokenKind::CP, lt_sym: uN[4]:0, cp_off: uN[3]: 1, cp_cnt: uN[4]: 2, mark: Mark::NONE},
    Token{kind: TokenKind::CP, lt_sym: uN[4]:0, cp_off: uN[3]: 0, cp_cnt: uN[4]: 0, mark: Mark::NONE},
    Token{kind: TokenKind::CP, lt_sym: uN[4]:0, cp_off: uN[3]: 7, cp_cnt: uN[4]: 0, mark: Mark::NONE},
    Token{kind: TokenKind::LT, lt_sym: uN[4]:15, cp_off: uN[3]:0, cp_cnt: uN[4]:0, mark: Mark::NONE},
    Token{kind: TokenKind::CP, lt_sym: uN[4]:0, cp_off: uN[3]: 4, cp_cnt: uN[4]: 0, mark: Mark::NONE},
    Token{kind: TokenKind::CP, lt_sym: uN[4]:0, cp_off: uN[3]: 5, cp_cnt: uN[4]:11, mark: Mark::NONE},
    Token{kind: TokenKind::LT, lt_sym: uN[4]: 8, cp_off: uN[3]:0, cp_cnt: uN[4]:0, mark: Mark::NONE},
    Token{kind: TokenKind::MK, lt_sym: uN[4]:0, cp_off: uN[3]:0, cp_cnt: uN[4]:0, mark: Mark::END}
];
// Reference output
const TST_SIMPLE_OUTPUT_LEN = u32:109;
const TST_SIMPLE_OUTPUT = PlainData<TST_SYM_WIDTH>[TST_SIMPLE_OUTPUT_LEN]: [
    PlainData{is_mark: false, data: uN[4]:12, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:1, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:15, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:9, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:11, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:9, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:11, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:11, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:11, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:11, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:9, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:15, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:9, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:9, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:15, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:9, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:9, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:15, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:9, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:9, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:15, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:9, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:9, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:15, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:9, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:9, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:15, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:1, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:9, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:15, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:9, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:9, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:7, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:2, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:2, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:2, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:2, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:2, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:2, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:2, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:2, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:2, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:2, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:2, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:2, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:2, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:2, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:2, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:2, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:2, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:2, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:2, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:2, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:2, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:2, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:2, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:2, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:2, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:2, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:2, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:2, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:2, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:2, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:2, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:2, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:2, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:2, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:2, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:2, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:3, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:9, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:2, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:2, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:3, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:9, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:2, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:2, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:3, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:9, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:2, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:2, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:3, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:9, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:2, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:2, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:3, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:9, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:14, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:9, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:14, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:9, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:9, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:2, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:15, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:14, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:14, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:9, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:9, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:2, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:15, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:14, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:14, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:9, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:9, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:2, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:15, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:14, mark: Mark::NONE},
    PlainData{is_mark: false, data: uN[4]:8, mark: Mark::NONE},
    PlainData{is_mark: true, data: uN[4]:0, mark: Mark::END}
];

#[test_proc]
proc test_simple {
    o_term: chan<bool> out;
    recv_last_r: chan<bool> in;

    init {()}

    config(o_term: chan<bool> out) {        
        // tokens from sender and to decoder
        let (send_toks_s, send_toks_r) =
            chan<Token<TST_SYM_WIDTH, TST_PTR_WIDTH, TST_CNT_WIDTH>>;
        // symbols & last indicator from receiver
        let (recv_data_s, recv_data_r) = chan<PlainData<TST_SYM_WIDTH>>;
        let (recv_last_s, recv_last_r) = chan<bool>;


        spawn test::token_sender
            <TST_SIMPLE_INPUT_LEN, TST_SYM_WIDTH, TST_PTR_WIDTH, TST_CNT_WIDTH>
            (TST_SIMPLE_INPUT, send_toks_s);
        spawn decoder_base_modelram<TST_SYM_WIDTH, TST_PTR_WIDTH, TST_CNT_WIDTH>(
            send_toks_r, recv_data_s);
        spawn test::data_validator<TST_SIMPLE_OUTPUT_LEN, TST_SYM_WIDTH>(
            TST_SIMPLE_OUTPUT, recv_data_r, recv_last_s);

        (o_term, recv_last_r)
    }

    next (tok: token, state: ()) {
        // Here, test::data_validator validates the data received from decoder
        // We just forward its o_term to our o_term
        let (tok, v) = recv(tok, recv_last_r);
        send(tok, o_term, v);
    }
}
