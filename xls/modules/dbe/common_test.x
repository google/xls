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

type TokenKind = dbe::TokenKind;
type Token = dbe::Token;
type PlainData = dbe::PlainData;

///
/// This library contains helper processes and functions used by DBE tests
///

pub proc data_sender<NUM_SYMS: u32, SYM_WIDTH:u32> {
    syms: uN[SYM_WIDTH][NUM_SYMS];
    o_data: chan<PlainData<SYM_WIDTH>> out;

    init { u32:0 }

    config (
        syms: uN[SYM_WIDTH][NUM_SYMS],
        o_data: chan<PlainData<SYM_WIDTH>> out,
    ) {
        (syms, o_data)
    }

    next (tok: token, state: u32) {
        let next_idx = state + u32:1;
        let is_last = (next_idx == NUM_SYMS);
        let is_done = (state >= NUM_SYMS);

        let _ = if (!is_done) {
            let sym = syms[state];
            let data = PlainData{
                sym: sym,
                last: is_last
            };
            let _ = send(tok, o_data, data);
            let _ = trace_fmt!("Sent {}", data);
            ()
        } else {()};

        next_idx
    }
}

pub proc data_validator<NUM_SYMS: u32, SYM_WIDTH:u32> {
    ref_syms: uN[SYM_WIDTH][NUM_SYMS];
    i_data: chan<PlainData<SYM_WIDTH>> in;
    o_term: chan<bool> out;

    init { u32:0 }

    config(
        ref_syms: uN[SYM_WIDTH][NUM_SYMS],
        i_data: chan<PlainData<SYM_WIDTH>> in,
        o_term: chan<bool> out
    ) {
        (ref_syms, i_data, o_term)
    }

    next (tok: token, state: u32) {
        let next_idx = state + u32:1;
        let is_done = (state >= NUM_SYMS);

        let (tok, data) = recv(tok, i_data);
        let sym = data.sym;
        let last = data.last;
        trace_fmt!("Received {}", data);

        let (fail, next_state) = if (!is_done) {
            let exp_sym = ref_syms[state];
            let exp_last = (next_idx == NUM_SYMS);
            let fail = if (exp_sym != sym || exp_last != last) {
                trace_fmt!(
                    "MISMATCH! Expected sym:{} last:{}, got sym:{} last:{}",
                    exp_sym, exp_last, sym, last);
                true
            } else {
                false
            };
            (fail, next_idx)
        } else {
            (true, state)
        };

        let _ = send_if(tok, o_term, fail || last, !fail);

        next_state
    }
}

/// Dummy consumer for PlainData channels
pub proc data_blackhole<N: u32> {
    i_str: chan<PlainData<N>> in;

    init {()}

    config(
        i_str: chan<PlainData<N>> in
    ) {
        (i_str,)
    }

    next (tok: token, state: ()) {
        let _ = recv(tok, i_str);
        ()
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
        let is_last = (next_idx == NUM_TOKS);
        let is_done = (state >= NUM_TOKS);

        let _ = if (!is_done) {
            let tosend = toks[state];
            let _ = send_if(tok, o_toks, !is_done, tosend);
            let _ = trace_fmt!("Sent last:{} tok:{}", is_last, tosend);
            ()
        } else {()};

        next_idx
    }
}

pub proc token_sniffer<SYM_WIDTH: u32, PTR_WIDTH: u32, CNT_WIDTH: u32> {
    i_str: chan<Token<SYM_WIDTH, PTR_WIDTH, CNT_WIDTH>> in;
    o_str: chan<Token<SYM_WIDTH, PTR_WIDTH, CNT_WIDTH>> out;

    init {()}

    config(
        i_str: chan<Token<SYM_WIDTH, PTR_WIDTH, CNT_WIDTH>> in,
        o_str: chan<Token<SYM_WIDTH, PTR_WIDTH, CNT_WIDTH>> out,
    ) {
        (i_str, o_str)
    }

    next(tok: token, state: ()) {
        let (tok, v) = recv(tok, i_str);
        let _ = trace_fmt!("[sniff] {}", v);
        let _ = send(tok, o_str, v);
        ()
    }
}
