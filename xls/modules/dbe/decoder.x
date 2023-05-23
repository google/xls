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
import xls.modules.dbe.common as dbe

type TokenKind = dbe::TokenKind;
type Token = dbe::Token;
type PlainData = dbe::PlainData;


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
///        it by `cp_count + 1`.
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
    // History buffer
    hb: uN[SYM_WIDTH][HB_SIZE],
    // Write pointer - next location _to be written_
    wr_ptr: uN[PTR_WIDTH],
    // Read pointer - next location _to be read_
    rd_ptr: uN[PTR_WIDTH],
    // Read counter - remaining number of symbols to be read out from the
    // history buffer
    rd_ctr_rem: uN[CNT_WIDTH],
    // Flag that remembers value of 'last' for CP tokens
    cp_last: bool,
}

pub proc decoder<SYM_WIDTH: u32, PTR_WIDTH: u32, CNT_WIDTH: u32,
                    HB_SIZE: u32 = {u32:1<<PTR_WIDTH}> {
    i_encoded: chan<Token<SYM_WIDTH, PTR_WIDTH, CNT_WIDTH>> in;
    o_data: chan<PlainData<SYM_WIDTH>> out;

    init {
        State<SYM_WIDTH, PTR_WIDTH, CNT_WIDTH> {
            hb: uN[SYM_WIDTH][HB_SIZE]:[uN[SYM_WIDTH]:0, ...],
            wr_ptr: uN[PTR_WIDTH]:0,
            rd_ptr: uN[PTR_WIDTH]:0,
            rd_ctr_rem: uN[CNT_WIDTH]:0,
            cp_last: false,
        }
    }

    config (
        i_encoded: chan<Token<SYM_WIDTH, PTR_WIDTH, CNT_WIDTH>> in,
        o_symb: chan<PlainData<SYM_WIDTH>> out
    ) {
        (i_encoded, o_symb)
    }

    next (tok: token, state: State<SYM_WIDTH, PTR_WIDTH, CNT_WIDTH>) {
        let is_copying = state.rd_ctr_rem != uN[CNT_WIDTH]:0;

        // Token Rx
        let (tok, enc) = recv_if(tok, i_encoded, !is_copying, Token {
            kind: TokenKind::LT,
            last: false,
            lt_sym: uN[SYM_WIDTH]:0,
            cp_off: uN[PTR_WIDTH]:0,
            cp_cnt: uN[CNT_WIDTH]:0,
        });

        let is_cp = !is_copying && enc.kind == TokenKind::CP;
        let is_lit = !is_copying && enc.kind == TokenKind::LT;
        let is_sym_from_hist = is_copying || is_cp;
        let cp_last = if is_cp {enc.last} else {state.cp_last};

        // Read pointer set
        let rd_ptr = if is_cp {
            state.wr_ptr - enc.cp_off - uN[PTR_WIDTH]:1
        } else {
            state.rd_ptr
        };
        // Get the data item to be sent now
        let (data_valid, data) = if is_lit {
            (true, PlainData {
                sym: enc.lt_sym,
                last: enc.last
            })
        } else if is_sym_from_hist {
            (true, PlainData {
                sym: state.hb[rd_ptr],
                last: cp_last && state.rd_ctr_rem == uN[CNT_WIDTH]:1
            })
        } else {
            (false, PlainData {
                sym: uN[SYM_WIDTH]:0,
                last: false
            })
        };
        // Send the data item
        let tok = send_if(tok, o_data, data_valid, data);

        // Write the symbol to the history buffer
        let hb = if data_valid {
            update(state.hb, state.wr_ptr, data.sym)
        } else {
            state.hb
        };
        // Write pointer increment
        let wr_ptr = if data_valid {
            state.wr_ptr + uN[PTR_WIDTH]:1
        } else {
            state.wr_ptr
        };
        // Read pointer increment
        let rd_ptr = if is_sym_from_hist {
            rd_ptr + uN[PTR_WIDTH]:1
        }  else {
            rd_ptr
        };
        // Read count set & decrement
        let rd_ctr_rem = if is_cp {
            enc.cp_cnt
        } else if is_copying {
            state.rd_ctr_rem - uN[CNT_WIDTH]:1
        } else {
            state.rd_ctr_rem
        };

        State{
            hb: hb,
            wr_ptr: wr_ptr,
            rd_ptr: rd_ptr,
            rd_ctr_rem: rd_ctr_rem,
            cp_last: cp_last,
        }
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
    Token{kind: TokenKind::LT, last: false, lt_sym: uN[4]:12, cp_off: uN[3]:0, cp_cnt: uN[4]:0},
    Token{kind: TokenKind::LT, last: false, lt_sym: uN[4]: 1, cp_off: uN[3]:0, cp_cnt: uN[4]:0},
    Token{kind: TokenKind::LT, last: false, lt_sym: uN[4]:15, cp_off: uN[3]:0, cp_cnt: uN[4]:0},
    Token{kind: TokenKind::LT, last: false, lt_sym: uN[4]: 9, cp_off: uN[3]:0, cp_cnt: uN[4]:0},
    Token{kind: TokenKind::LT, last: false, lt_sym: uN[4]:11, cp_off: uN[3]:0, cp_cnt: uN[4]:0},
    Token{kind: TokenKind::CP, last: false, lt_sym: uN[4]:0, cp_off: uN[3]: 1, cp_cnt: uN[4]: 1},
    Token{kind: TokenKind::CP, last: false, lt_sym: uN[4]:0, cp_off: uN[3]: 0, cp_cnt: uN[4]: 2},
    Token{kind: TokenKind::CP, last: false, lt_sym: uN[4]:0, cp_off: uN[3]: 4, cp_cnt: uN[4]: 0},
    Token{kind: TokenKind::LT, last: false, lt_sym: uN[4]:15, cp_off: uN[3]:0, cp_cnt: uN[4]:0},
    Token{kind: TokenKind::CP, last: false, lt_sym: uN[4]:0, cp_off: uN[3]: 1, cp_cnt: uN[4]: 0},
    Token{kind: TokenKind::CP, last: false, lt_sym: uN[4]:0, cp_off: uN[3]: 2, cp_cnt: uN[4]:13},
    Token{kind: TokenKind::LT, last: false, lt_sym: uN[4]: 1, cp_off: uN[3]:0, cp_cnt: uN[4]:0},
    Token{kind: TokenKind::CP, last: false, lt_sym: uN[4]:0, cp_off: uN[3]: 5, cp_cnt: uN[4]: 0},
    Token{kind: TokenKind::CP, last: false, lt_sym: uN[4]:0, cp_off: uN[3]: 5, cp_cnt: uN[4]: 2},
    Token{kind: TokenKind::LT, last: false, lt_sym: uN[4]: 7, cp_off: uN[3]:0, cp_cnt: uN[4]:0},
    Token{kind: TokenKind::LT, last: false, lt_sym: uN[4]: 2, cp_off: uN[3]:0, cp_cnt: uN[4]:0},
    Token{kind: TokenKind::CP, last: false, lt_sym: uN[4]:0, cp_off: uN[3]: 0, cp_cnt: uN[4]:11},
    Token{kind: TokenKind::CP, last: false, lt_sym: uN[4]:0, cp_off: uN[3]: 1, cp_cnt: uN[4]: 6},
    Token{kind: TokenKind::CP, last: false, lt_sym: uN[4]:0, cp_off: uN[3]: 5, cp_cnt: uN[4]:15},
    Token{kind: TokenKind::LT, last: false, lt_sym: uN[4]: 3, cp_off: uN[3]:0, cp_cnt: uN[4]:0},
    Token{kind: TokenKind::LT, last: false, lt_sym: uN[4]: 9, cp_off: uN[3]:0, cp_cnt: uN[4]:0},
    Token{kind: TokenKind::CP, last: false, lt_sym: uN[4]:0, cp_off: uN[3]: 5, cp_cnt: uN[4]: 1},
    Token{kind: TokenKind::CP, last: false, lt_sym: uN[4]:0, cp_off: uN[3]: 3, cp_cnt: uN[4]:13},
    Token{kind: TokenKind::LT, last: false, lt_sym: uN[4]:14, cp_off: uN[3]:0, cp_cnt: uN[4]:0},
    Token{kind: TokenKind::CP, last: false, lt_sym: uN[4]:0, cp_off: uN[3]: 1, cp_cnt: uN[4]: 2},
    Token{kind: TokenKind::CP, last: false, lt_sym: uN[4]:0, cp_off: uN[3]: 0, cp_cnt: uN[4]: 0},
    Token{kind: TokenKind::CP, last: false, lt_sym: uN[4]:0, cp_off: uN[3]: 7, cp_cnt: uN[4]: 0},
    Token{kind: TokenKind::LT, last: false, lt_sym: uN[4]:15, cp_off: uN[3]:0, cp_cnt: uN[4]:0},
    Token{kind: TokenKind::CP, last: false, lt_sym: uN[4]:0, cp_off: uN[3]: 4, cp_cnt: uN[4]: 0},
    Token{kind: TokenKind::CP, last: false, lt_sym: uN[4]:0, cp_off: uN[3]: 5, cp_cnt: uN[4]:11},
    Token{kind: TokenKind::LT, last: false, lt_sym: uN[4]: 8, cp_off: uN[3]:0, cp_cnt: uN[4]:0},
    Token{kind: TokenKind::CP, last: true, lt_sym: uN[4]:0, cp_off: uN[3]: 6, cp_cnt: uN[4]: 8}            
];
// Reference output
const TST_SIMPLE_OUTPUT_LEN = u32:117;
const TST_SIMPLE_OUTPUT = uN[TST_SYM_WIDTH][TST_SIMPLE_OUTPUT_LEN]: [
    12,  1, 15,  9, 11,  9, 11, 11, 11, 11,  9, 15,  9,  9, 15,  9,
    9, 15,  9,  9, 15,  9,  9, 15,  9,  9, 15,  1,  9, 15,  9,  9,
    7,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
    2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
    2,  2,  2,  2,  2,  3,  9,  2,  2,  3,  9,  2,  2,  3,  9,  2,
    2,  3,  9,  2,  2,  3,  9, 14,  9, 14,  9,  9,  2, 15, 14, 14,
    9,  9,  2, 15, 14, 14,  9,  9,  2, 15, 14,  8, 14,  9,  9,  2,
    15, 14,  8, 14,  9
];

#[test_proc]
proc test_simple {
    o_term: chan<bool> out;
    recv_last_r: chan<bool> in;

    init {()}

    config(o_term: chan<bool> out) {        
        // tokens from sender and to decoder
        let (send_toks_s, send_toks_r) = chan<Token<TST_SYM_WIDTH, TST_PTR_WIDTH, TST_CNT_WIDTH>>;
        // symbols & last indicator from receiver
        let (recv_data_s, recv_data_r) = chan<PlainData<TST_SYM_WIDTH>>;
        let (recv_last_s, recv_last_r) = chan<bool>;


        spawn test::token_sender
            <TST_SIMPLE_INPUT_LEN, TST_SYM_WIDTH, TST_PTR_WIDTH, TST_CNT_WIDTH>
            (TST_SIMPLE_INPUT, send_toks_s);
        spawn decoder<TST_SYM_WIDTH, TST_PTR_WIDTH, TST_CNT_WIDTH>(
            send_toks_r, recv_data_s);
        /// NOTE(2023-06-20):
        /// can not pass o_term directly to data_validator for some reason
        spawn test::data_validator<TST_SIMPLE_OUTPUT_LEN, TST_SYM_WIDTH>(
            TST_SIMPLE_OUTPUT, recv_data_r, recv_last_s);

        (o_term, recv_last_r)
    }

    next (tok: token, state: ()) {
        // Here, test::data_validator validates the data received from decoder
        // We only forward its o_term to our o_term
        let (tok, v) = recv(tok, recv_last_r);
        let _ = send(tok, o_term, v);
    }
}
