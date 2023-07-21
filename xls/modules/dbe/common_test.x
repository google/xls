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

import xls.modules.dbe.common as dbe

type Mark = dbe::Mark;
type TokenKind = dbe::TokenKind;
type Token = dbe::Token;
type PlainData = dbe::PlainData;

///
/// This library contains helper processes and functions used by DBE tests
///

pub proc data_sender<NUM_SYMS: u32, SYM_WIDTH:u32> {
    data: PlainData<SYM_WIDTH>[NUM_SYMS];
    o_data: chan<PlainData<SYM_WIDTH>> out;

    init { u32:0 }

    config (
        data: PlainData<SYM_WIDTH>[NUM_SYMS],
        o_data: chan<PlainData<SYM_WIDTH>> out,
    ) {
        (data, o_data)
    }

    next (tok: token, state: u32) {
        let next_idx = state + u32:1;
        let is_done = (state >= NUM_SYMS);

        if (!is_done) {
            let tosend = data[state];
            trace_fmt!("Sending {}", tosend);
            send(tok, o_data, tosend);
            next_idx
        } else {
            state
        }
    }
}

pub proc data_validator<NUM_SYMS: u32, SYM_WIDTH:u32> {
    ref_data: PlainData<SYM_WIDTH>[NUM_SYMS];
    i_data: chan<PlainData<SYM_WIDTH>> in;
    o_term: chan<bool> out;

    init { u32:0 }

    config(
        ref_data: PlainData<SYM_WIDTH>[NUM_SYMS],
        i_data: chan<PlainData<SYM_WIDTH>> in,
        o_term: chan<bool> out
    ) {
        (ref_data, i_data, o_term)
    }

    next (tok: token, state: u32) {
        // state = [0, NUM_SYMS-1] - expect data
        // state >= NUM_SYMS - expect nothing
        let next_idx = state + u32:1;
        let is_end = (state == NUM_SYMS - u32:1);
        let is_done = (state >= NUM_SYMS);

        let (tok, rx) = recv(tok, i_data);
        trace_fmt!("Received {}", rx);

        let (fail, next_state) = if is_done {
            // Shouldn't get here
            (true, state)
        } else {
            let expect = ref_data[state];
            let fail = if (rx != expect) {
                trace_fmt!("MISMATCH! Expected {}, got {}", expect, rx);
                true
            } else {
                false
            };
            (fail, next_idx)
        };

        send_if(tok, o_term, fail || is_end, !fail);

        next_state
    }
}

pub proc token_sender<
        NUM_TOKS: u32, SYM_WIDTH: u32, PTR_WIDTH: u32, CNT_WIDTH: u32> {
    toks: Token<SYM_WIDTH, PTR_WIDTH, CNT_WIDTH>[NUM_TOKS];
    o_toks: chan<Token<SYM_WIDTH, PTR_WIDTH, CNT_WIDTH>> out;

    init { u32:0 }

    config (
        toks: Token<SYM_WIDTH, PTR_WIDTH, CNT_WIDTH>[NUM_TOKS],
        o_toks: chan<Token<SYM_WIDTH, PTR_WIDTH, CNT_WIDTH>> out,
    ) {
        (toks, o_toks)
    }

    next (tok: token, state: u32) {
        let next_idx = state + u32:1;
        let is_done = (state >= NUM_TOKS);

        if (!is_done) {
            let tosend = toks[state];
            trace_fmt!("Sending {}", tosend);
            send(tok, o_toks, tosend);
            next_idx
        } else {
            state
        }
    }
}

pub proc token_validator<
        NUM_TOKS: u32, SYM_WIDTH: u32, PTR_WIDTH: u32, CNT_WIDTH: u32> {
    ref_toks: Token<SYM_WIDTH, PTR_WIDTH, CNT_WIDTH>[NUM_TOKS];
    i_token: chan<Token<SYM_WIDTH, PTR_WIDTH, CNT_WIDTH>> in;
    o_term: chan<bool> out;

    init { u32:0 }

    config(
        ref_toks: Token<SYM_WIDTH, PTR_WIDTH, CNT_WIDTH>[NUM_TOKS],
        i_token: chan<Token<SYM_WIDTH, PTR_WIDTH, CNT_WIDTH>> in,
        o_term: chan<bool> out
    ) {
        (ref_toks, i_token, o_term)
    }

    next (tok: token, state: u32) {
        // state = [0, NUM_TOKS-1] - expect token
        // state >= NUM_TOKS - expect nothing
        let next_idx = state + u32:1;
        let is_end = (state == NUM_TOKS - u32:1);
        let is_done = (state >= NUM_TOKS);

        let (tok, rx) = recv(tok, i_token);
        trace_fmt!("Received {}", rx);

        let (fail, next_state) = if is_done {
            // Shouldn't get here
            (true, state)
        } else {
            let expect = ref_toks[state];
            let fail = if (rx != expect) {
                trace_fmt!("MISMATCH! Expected {}, got {}", expect, rx);
                true
            } else {
                false
            };
            (fail, next_idx)
        };

        send_if(tok, o_term, fail || is_end, !fail);

        next_state
    }
}
