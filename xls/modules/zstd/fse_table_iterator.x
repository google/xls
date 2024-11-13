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

// This proc provides the order in which the FSE decoding table should be
// filled with symbols. The algorithm is described in:
// https://datatracker.ietf.org/doc/html/rfc8878#section-4.1.1

import std;
import xls.modules.zstd.common;

type Reset = bool;
type Index = common::FseTableIndex;
type Ctrl = common::FseTableCreatorCtrl;

type AccuracyLog = common::FseAccuracyLog;
type SymbolCount = common::FseSymbolCount;

enum Status : u1 {
    CONFIGURE = 0,
    SEND = 1,
}

struct State { status: Status, ctrl: Ctrl, cnt: u16, pos: u16 }

pub proc FseTableIterator {
    ctrl_r: chan<Ctrl> in;
    idx_s: chan<Index> out;

    config(
        ctrl_r: chan<Ctrl> in,
        idx_s: chan<Index> out
    ) { (ctrl_r, idx_s) }

    init { zero!<State>() }

    next(state: State) {
        const ZERO_STATE = zero!<State>();
        const ZERO_IDX_OPTION = (false, u16:0);

        let tok0 = join();

        let do_recv_ctrl = state.status == Status::CONFIGURE;
        let (tok1, ctrl) = recv_if(tok0, ctrl_r, do_recv_ctrl, zero!<Ctrl>());

        let ((do_send_idx, idx), new_state) = match (state.status) {
            Status::CONFIGURE => {
                ((true, u16:0), State { ctrl, status: Status::SEND, ..ZERO_STATE })
                },
            Status::SEND => {
                let size = u16:1 << state.ctrl.accuracy_log;
                let high_threshold = size - state.ctrl.negative_proba_count as u16;
                let step = (size >> 1) + (size >> 3) + u16:3;
                let mask = size - u16:1;

                let pos = (state.pos + step) & mask;

                let valid = pos < high_threshold;
                let next_cnt = state.cnt + u16:1;
                let last = (valid && (next_cnt == high_threshold - u16:1));

                if last {
                    ((true, pos), ZERO_STATE)
                } else if valid {
                    ((true, pos), State { cnt: next_cnt, pos, ..state })
                } else {
                    (ZERO_IDX_OPTION, State { cnt: state.cnt, pos, ..state })
                }
            },
            _ => fail!("incorrect_state", (ZERO_IDX_OPTION, ZERO_STATE)),
        };

        let tok2 = send_if(tok1, idx_s, do_send_idx, checked_cast<Index>(idx));
        if do_send_idx { trace_fmt!("[IO]: Send index: {}", idx); } else {  };

        new_state
    }
}

const TEST_EXPECTRED_IDX = Index[27]:[
    Index:0, Index:23, Index:14, Index:5, Index:19, Index:10, Index:1, Index:24, Index:15, Index:6,
    Index:20, Index:11, Index:2, Index:25, Index:16, Index:7, Index:21, Index:12, Index:3, Index:26,
    Index:17, Index:8, Index:22, Index:13, Index:4, Index:18, Index:9,
];

#[test_proc]
proc FseTableIteratorTest {
    terminator: chan<bool> out;
    ctrl_s: chan<Ctrl> out;
    idx_r: chan<Index> in;

    config(terminator: chan<bool> out) {
        let (ctrl_s, ctrl_r) = chan<Ctrl>("ctrl");
        let (idx_s, idx_r) = chan<Index>("idx");

        spawn FseTableIterator(ctrl_r, idx_s);
        (terminator, ctrl_s, idx_r)
    }

    init {  }

    next(state: ()) {
        let tok = join();
        let tok = send(
            tok, ctrl_s, Ctrl { accuracy_log: AccuracyLog:5, negative_proba_count: SymbolCount:5 });
        let tok = for (exp_idx, tok): (Index, token) in TEST_EXPECTRED_IDX {
            let (tok, idx) = recv(tok, idx_r);
            assert_eq(idx, exp_idx);
            (tok)
        }(tok);

        send(tok, terminator, true);
    }
}
