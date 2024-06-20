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
type AbstractRamWriteReq = ram::WriteReq;
type AbstractRamWriteResp = ram::WriteResp;
type AbstractRamReadReq = ram::ReadReq;
type AbstractRamReadResp = ram::ReadResp;


/// The behavior of our decoder is defined as follows.
/// 
/// A. History buffer is a buffer of (1<<MATCH_OFFSET_WIDTH) cells in size,
///    each cell is capable of storing one symbol (uN[SYMBOL_WIDTH] number).
/// B. Cells in the buffer are addressed using indices, which take values from
///    0 to (1<<MATCH_OFFSET_WIDTH)-1.
/// C. The history buffer has a write pointer and a read pointer associated
///    with it.
/// D. Each pointer can store a single cell index value in range from
///    0 to (1<<MATCH_OFFSET_WIDTH)-1 inclusive.
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
///    SYMBOL:
///     1. Write `literal` symbol to the history buffer.
///     2. Send PlainData object to the output data stream, wherein `sym` is
///        equal to `literal` and `last` flag is equal to this Token's `last`
///        flag.
///    MATCH:
///     1. Set read pointer to be equal to the write pointer, then decrement
///        it by `match_offset + 1`.
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


struct State<SYMBOL_WIDTH: u32, MATCH_OFFSET_WIDTH: u32, MATCH_LENGTH_WIDTH: u32> {
    // Write pointer - next location _to be written_
    wr_ptr: uN[MATCH_OFFSET_WIDTH],
    // Read pointer - next location _to be read_
    rd_ptr: uN[MATCH_OFFSET_WIDTH],
    // Read counter - remaining number of symbols to be read out from the
    // history buffer
    rd_ctr_rem: uN[MATCH_LENGTH_WIDTH],
    // Whether we're in the ERROR state
    err: bool,
    // Helps to track which cells in HB are written and which ones are not
    // (if some MATCH token points to an unwritten cell, we detect that and
    // generate an error, as otherwise that will make decoder output garbage
    // data that may leak some stale data from previous blocks)
    hb_all_valid: bool,
}

pub proc decoder_base<
    SYMBOL_WIDTH: u32, MATCH_OFFSET_WIDTH: u32, MATCH_LENGTH_WIDTH: u32
> {
    encoded_data: chan<Token<SYMBOL_WIDTH, MATCH_OFFSET_WIDTH, MATCH_LENGTH_WIDTH>> in;
    plain_data: chan<PlainData<SYMBOL_WIDTH>> out;

    /// History buffer RAM
    /// Size: (1<<MATCH_OFFSET_WIDTH) x SYMBOL_WIDTH
    ram_hb_rd_req: chan<ram::ReadReq<MATCH_OFFSET_WIDTH, 1>> out;
    ram_hb_rd_resp: chan<ram::ReadResp<SYMBOL_WIDTH>> in;
    ram_hb_wr_req: chan<ram::WriteReq<MATCH_OFFSET_WIDTH, SYMBOL_WIDTH, 1>> out;
    ram_hb_wr_comp: chan<ram::WriteResp> in;

    init {
        State<SYMBOL_WIDTH, MATCH_OFFSET_WIDTH, MATCH_LENGTH_WIDTH> {
            wr_ptr: uN[MATCH_OFFSET_WIDTH]:0,
            rd_ptr: uN[MATCH_OFFSET_WIDTH]:0,
            rd_ctr_rem: uN[MATCH_LENGTH_WIDTH]:0,
            err: false,
            hb_all_valid: false,
        }
    }

    config (
        encoded_data:
            chan<
                Token<SYMBOL_WIDTH, MATCH_OFFSET_WIDTH, MATCH_LENGTH_WIDTH>
            > in,
        plain_data: chan<PlainData<SYMBOL_WIDTH>> out,
        // RAM
        ram_hb_rd_req: chan<ram::ReadReq<MATCH_OFFSET_WIDTH, 1>> out,
        ram_hb_rd_resp: chan<ram::ReadResp<SYMBOL_WIDTH>> in,
        ram_hb_wr_req:
            chan<
                ram::WriteReq<MATCH_OFFSET_WIDTH, SYMBOL_WIDTH, 1>
            > out,
        ram_hb_wr_comp: chan<ram::WriteResp> in,
    ) {
        (
            encoded_data, plain_data,
            ram_hb_rd_req, ram_hb_rd_resp, ram_hb_wr_req, ram_hb_wr_comp,
        )
    }

    next (
        tok: token,
        state: State<SYMBOL_WIDTH, MATCH_OFFSET_WIDTH, MATCH_LENGTH_WIDTH>
    ) {
        type DecToken =
            Token<SYMBOL_WIDTH, MATCH_OFFSET_WIDTH, MATCH_LENGTH_WIDTH>;
        type DecData = PlainData<SYMBOL_WIDTH>;
        type HbRamReadResp = AbstractRamReadResp<SYMBOL_WIDTH>;

        let is_copying = state.rd_ctr_rem != uN[MATCH_LENGTH_WIDTH]:0;
        let do_recv = !is_copying;

        // Receive token
        let (tok, rx) = recv_if(tok, encoded_data, do_recv,
            zero!<DecToken>());

        let rx_lt = do_recv && rx.kind == TokenKind::UNMATCHED_SYMBOL;
        let rx_cp = do_recv && rx.kind == TokenKind::MATCH;
        let rx_mark = do_recv && rx.kind == TokenKind::MARKER;
        let rx_end = rx_mark && rx.mark == Mark::END;
        let rx_error = rx_mark && dbe::is_error(rx.mark);
        let rx_reset = rx_mark && rx.mark == Mark::RESET;
        let rx_unexpected_mark = rx_mark && !(rx_end || rx_reset || rx_error);

        // Are we going to output a literal symbol or a symbol from HB?
        let emit_sym_lit = rx_lt;
        let emit_sym_from_hb = rx_cp || is_copying;

        // Reset, set or increment read pointer
        let rd_ptr = if rx_reset {
            uN[MATCH_OFFSET_WIDTH]:0
        } else if rx_cp {
            state.wr_ptr - rx.match_offset - uN[MATCH_OFFSET_WIDTH]:1
        } else if is_copying {
            state.rd_ptr + uN[MATCH_OFFSET_WIDTH]:1
        } else {
            state.rd_ptr
        };

        // Check whether rd_ptr points to an unwritten HB cell
        let rd_ptr_valid = (rd_ptr < state.wr_ptr) || state.hb_all_valid;

        // HB READ - read the symbol from history buffer
        let do_hb_read = emit_sym_from_hb && rd_ptr_valid;
        let tok = send_if(tok, ram_hb_rd_req, do_hb_read,
            ram::ReadWordReq<u32:1>(rd_ptr));
        let (tok, hb_rd) = recv_if(tok, ram_hb_rd_resp, do_hb_read,
            zero!<HbRamReadResp>());

        // Generate ERROR_INVAL_CP when we should've read from HB but
        // we couldn't because MATCH points to an invalid location
        let err_inval_cp = emit_sym_from_hb && !do_hb_read;

        // Send decoded data symbol or Mark
        let zero_data = zero!<DecData>();
        let (data_valid, data) = if rx_reset {
            // Propagate RESET in all states
            (true, DecData {
                is_marker: true,
                mark: Mark::RESET,
                ..zero_data
            })
        } else if state.err {
            // Do not send anything else while in an error state
            (false, zero_data)
        } else if rx_error || rx_end {
            // Propagate ERROR and END tokens
            (true, DecData {
                is_marker: true,
                mark: rx.mark,
                ..zero_data
            })
        } else if rx_unexpected_mark {
            // Generate ERROR_BAD_MARK
            (true, DecData {
                is_marker: true,
                mark: Mark::ERROR_BAD_MARK,
                ..zero_data
            })
        } else if err_inval_cp {
            // Generate ERROR_INVAL_CP
            (true, DecData {
                is_marker: true,
                mark: Mark::ERROR_INVAL_CP,
                ..zero_data
            })
        } else if emit_sym_lit {
            // Replicate symbol from LITERAL token
            (true, DecData {
                is_marker: false,
                data: rx.symbol,
                ..zero_data
            })
        } else if emit_sym_from_hb {
            // Replicate symbol from HB
            (true, DecData {
                is_marker: false,
                data: hb_rd.data,
                ..zero_data
            })
        } else {
            (false, zero_data)
        };
        let tok = send_if(tok, plain_data, data_valid, data);

        // HB WRITE - write emitted symbol to the history buffer
        let do_hb_write = data_valid && !data.is_marker;
        let tok = send_if(tok, ram_hb_wr_req, do_hb_write,
            ram::WriteWordReq<u32:1>(state.wr_ptr, data.data));
        let (tok, _) = recv_if(tok, ram_hb_wr_comp, do_hb_write,
            zero!<AbstractRamWriteResp>());

        // Reset/increment write pointer
        let wr_ptr = if rx_reset {
            uN[MATCH_OFFSET_WIDTH]:0
        } else if do_hb_write {
            state.wr_ptr + uN[MATCH_OFFSET_WIDTH]:1
        } else {
            state.wr_ptr
        };

        // When write pointer wraps over to 0 after the increment,
        // that means we've filled the complete HB, so set a corresponding flag
        let hb_all_valid = if rx_reset {
            false
        } else if do_hb_write && wr_ptr == uN[MATCH_OFFSET_WIDTH]:0 {
            true
        } else {
            state.hb_all_valid
        };

        // Read count set & decrement
        let rd_ctr_rem = if rx_reset {
            uN[MATCH_LENGTH_WIDTH]:0
        } else if rx_cp {
            rx.match_length
        } else if is_copying {
            state.rd_ctr_rem - uN[MATCH_LENGTH_WIDTH]:1
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

/// Version of decoder compatible with classic LZ4 algorithm

type Lz4Token = dbe::Lz4Token;
type Lz4Data = dbe::Lz4Data;
pub type Lz4DecoderRamHbReadReq = ram::ReadReq<dbe::LZ4_MATCH_OFFSET_WIDTH, 1>;
pub type Lz4DecoderRamHbReadResp = ram::ReadResp<dbe::LZ4_SYMBOL_WIDTH>;
pub type Lz4DecoderRamHbWriteReq = ram::WriteReq<
        dbe::LZ4_MATCH_OFFSET_WIDTH,
        dbe::LZ4_SYMBOL_WIDTH, 1
    >;

pub proc decoder {
    init{}

    config (
        encoded_data: chan<Lz4Token> in,
        plain_data: chan<Lz4Data> out,
        ram_hb_rd_req: chan<Lz4DecoderRamHbReadReq> out,
        ram_hb_rd_resp: chan<Lz4DecoderRamHbReadResp> in,
        ram_hb_wr_req: chan<Lz4DecoderRamHbWriteReq> out,
        ram_hb_wr_comp: chan<ram::WriteResp> in,
    ) {
        spawn decoder_base<
                dbe::LZ4_SYMBOL_WIDTH,
                dbe::LZ4_MATCH_OFFSET_WIDTH,
                dbe::LZ4_MATCH_LENGTH_WIDTH
            > (
                encoded_data, plain_data,
                ram_hb_rd_req, ram_hb_rd_resp, ram_hb_wr_req, ram_hb_wr_comp,
            );
    }

    next(tok: token, st: ()) {
    }
}
