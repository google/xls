// Copyright 2025 The XLS Authors
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
import xls.modules.zstd.memory.mem_reader;

struct MemReaderMuxState<
    N: u32,
    N_WIDTH: u32 = {std::clog2(N + u32:1)}
> {
    sel: uN[N_WIDTH],
    active: bool,
}

pub proc MemReaderMux<
    ADDR_W: u32, DATA_W: u32, N: u32,
    INIT_SEL: u32 = {u32:0},
    N_WIDTH: u32 = {std::clog2(N + u32:1)}
> {
    type MemReaderReq = mem_reader::MemReaderReq<ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<DATA_W, ADDR_W>;

    type State = MemReaderMuxState<N>;
    type Sel = uN[N_WIDTH];

    sel_req_r: chan<Sel> in;
    sel_resp_s: chan<()> out;

    n_req_r: chan<MemReaderReq>[N] in;
    n_resp_s: chan<MemReaderResp>[N] out;

    req_s: chan<MemReaderReq> out;
    resp_r: chan<MemReaderResp> in;

    config(
        sel_req_r: chan<Sel> in,
        sel_resp_s: chan<()> out,

        n_req_r: chan<MemReaderReq>[N] in,
        n_resp_s: chan<MemReaderResp>[N] out,

        req_s: chan<MemReaderReq> out,
        resp_r: chan<MemReaderResp> in,
    ) {
        (
           sel_req_r, sel_resp_s,
           n_req_r, n_resp_s,
           req_s, resp_r,
        )
    }

    init {
        State {
            sel: checked_cast<Sel>(INIT_SEL),
            active: false,
        }
    }

    next(state: State) {
        let tok0 = join();

        let (tok1_0, n_req, n_req_valid) = recv_if_non_blocking(tok0, n_req_r[state.sel], !state.active, zero!<MemReaderReq>());
        let tok2_0 = send_if(tok1_0, req_s, n_req_valid, n_req);

        let active = state.active || n_req_valid;

        let (tok2_1, resp, resp_valid) = recv_if_non_blocking(tok1_0, resp_r, active, zero!<MemReaderResp>());
        let tok3_0 = send_if(tok2_1, n_resp_s[state.sel], resp_valid, resp);

        let active = (state.active || n_req_valid) && !(resp_valid && resp.last);

        let (tok3_1, sel, sel_valid) = recv_if_non_blocking(tok2_1, sel_req_r, !active, state.sel);
        let tok4_0 = send_if(tok3_1, sel_resp_s, !active && sel_valid, ());

        State { active, sel }
    }
}

const TEST_ADDR_W = u32:32;
const TEST_DATA_W = u32:32;
const TEST_N = u32:5;
const TEST_N_WIDTH = std::clog2(TEST_N + u32:1);

#[test_proc]
proc MemReaderMuxTest {
    type MemReaderReq = mem_reader::MemReaderReq<TEST_ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<TEST_DATA_W, TEST_ADDR_W>;

    type State = MemReaderMuxState<TEST_N>;
    type Sel = uN[TEST_N_WIDTH];

    terminator: chan<bool> out;

    sel_req_s: chan<Sel> out;
    sel_resp_r: chan<()> in;

    n_req_s: chan<MemReaderReq>[TEST_N] out;
    n_resp_r: chan<MemReaderResp>[TEST_N] in;

    req_r: chan<MemReaderReq> in;
    resp_s: chan<MemReaderResp> out;

    init {}
    config(terminator: chan<bool> out,) {

        let (sel_req_s, sel_req_r) = chan<Sel>("sel_req");
        let (sel_resp_s, sel_resp_r) = chan<()>("sel_resp");

        let (n_req_s, n_req_r) = chan<MemReaderReq>[TEST_N]("n_req");
        let (n_resp_s, n_resp_r) = chan<MemReaderResp>[TEST_N]("n_resp");

        let (req_s, req_r) = chan<MemReaderReq>("req");
        let (resp_s, resp_r) = chan<MemReaderResp>("resp");

        spawn MemReaderMux<TEST_ADDR_W, TEST_DATA_W, TEST_N>(
            sel_req_r, sel_resp_s,
            n_req_r, n_resp_s,
            req_s, resp_r,
        );

        (
            terminator,
            sel_req_s, sel_resp_r,
            n_req_s, n_resp_r,
            req_r, resp_s,
        )
    }

    next(state: ()) {
        let tok = join();

        let tok = send(tok, sel_req_s, Sel:3);
        let (tok, _) = recv(tok, sel_resp_r);

        let tok = send(tok, n_req_s[3], zero!<MemReaderReq>());
        let (tok, _) = recv(tok, req_r);

        // Cannot switch during the transmission
        let tok = send(tok, sel_req_s, Sel:1);

        let tok = send(tok, resp_s, zero!<MemReaderResp>());
        let (tok, _) = recv(tok, n_resp_r[3]);

        // Now we should be able to receive the select
        let (tok, _) = recv(tok, sel_resp_r);

        let tok = send(tok, n_req_s[1], zero!<MemReaderReq>());
        let (tok, _) = recv(tok, req_r);

        let tok = send(tok, resp_s, zero!<MemReaderResp>());
        let (tok, _) = recv(tok, n_resp_r[1]);

        send(tok, terminator, true);
    }
}
