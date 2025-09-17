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
import xls.modules.zstd.memory.mem_writer;

struct MemWriterMuxState<
    N: u32,
    N_WIDTH: u32 = {std::clog2(N + u32:1)}
> {
    sel: uN[N_WIDTH],
    active: bool,
}

pub proc MemWriterMux<
    ADDR_W: u32, DATA_W: u32, N: u32,
    INIT_SEL: u32 = {u32:0},
    N_WIDTH: u32 = {std::clog2(N + u32:1)}
> {
    type MemWriterReq = mem_writer::MemWriterReq<ADDR_W>;
    type MemWriterResp = mem_writer::MemWriterResp;
    type MemWriterData = mem_writer::MemWriterDataPacket<DATA_W, ADDR_W>;

    type State = MemWriterMuxState<N>;
    type Sel = uN[N_WIDTH];

    sel_req_r: chan<Sel> in;
    sel_resp_s: chan<()> out;

    n_req_r: chan<MemWriterReq>[N] in;
    n_data_r: chan<MemWriterData>[N] in;
    n_resp_s: chan<MemWriterResp>[N] out;

    req_s: chan<MemWriterReq> out;
    data_s: chan<MemWriterData> out;
    resp_r: chan<MemWriterResp> in;

    config(
        sel_req_r: chan<Sel> in,
        sel_resp_s: chan<()> out,

        n_req_r: chan<MemWriterReq>[N] in,
        n_data_r: chan<MemWriterData>[N] in,
        n_resp_s: chan<MemWriterResp>[N] out,

        req_s: chan<MemWriterReq> out,
        data_s: chan<MemWriterData> out,
        resp_r: chan<MemWriterResp> in,
    ) {
        (
           sel_req_r, sel_resp_s,
           n_req_r, n_data_r, n_resp_s,
           req_s, data_s, resp_r,
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

        let (tok1_0, n_req, n_req_valid) = unroll_for! (i, (tok, resp, valid)): (uN[N_WIDTH], (token, MemWriterReq, bool)) in range(uN[N_WIDTH]:0, N as uN[N_WIDTH]) {
            let (tok, r, v) = recv_if_non_blocking(tok, n_req_r[i], state.sel == i && !state.active, zero!<MemWriterReq>());
            if v { (tok, r, true) } else { (tok, resp, valid) }
        }((tok0, zero!<MemWriterReq>(), false));

        let tok2_0 = send_if(tok1_0, req_s, n_req_valid, n_req);

        let active = state.active || n_req_valid;

        let (tok2_1, n_data, n_data_valid) = unroll_for! (i, (tok, resp, valid)): (uN[N_WIDTH], (token, MemWriterData, bool)) in range(uN[N_WIDTH]:0, N as uN[N_WIDTH]) {
            let (tok, r, v) = recv_if_non_blocking(tok, n_data_r[i], state.sel == i && active, zero!<MemWriterData>());
            if v { (tok, r, true) } else { (tok, resp, valid) }
        }((tok1_0, zero!<MemWriterData>(), false));

        let tok3_0 = send_if(tok2_1, data_s, n_data_valid, n_data);

        let (tok2_2, resp, resp_valid) = recv_if_non_blocking(tok1_0, resp_r, active, zero!<MemWriterResp>());

        let tok3_1 = unroll_for! (i, tok): (uN[N_WIDTH], token) in range(uN[N_WIDTH]:0, N as uN[N_WIDTH]) {
            send_if(tok, n_resp_s[i], state.sel == i && resp_valid, resp)
        }(tok2_2);
        let active = (state.active || n_req_valid) && !(resp_valid);
        let (tok3_2, sel, sel_valid) = recv_if_non_blocking(tok2_2, sel_req_r, !active, state.sel);
        let tok4_0 = send_if(tok3_2, sel_resp_s, sel_valid, ());
        State { active, sel }
    }
}


const INST_ADDR_W = u32:32;
const INST_DATA_W = u32:32;
const INST_N = u32:5;
const INST_N_WIDTH = std::clog2(INST_N + u32:1);

proc MemWriterMuxInst {
    type MemWriterReq = mem_writer::MemWriterReq<INST_ADDR_W>;
    type MemWriterResp = mem_writer::MemWriterResp;
    type MemWriterData = mem_writer::MemWriterDataPacket<INST_DATA_W, INST_ADDR_W>;

    type State = MemWriterMuxState<INST_N>;
    type Sel = uN[INST_N_WIDTH];

    init {}
    config(
        sel_req_r: chan<Sel> in,
        sel_resp_s: chan<()> out,
        n_req_r: chan<MemWriterReq>[INST_N] in,
        n_data_r: chan<MemWriterData>[INST_N] in,
        n_resp_s: chan<MemWriterResp>[INST_N] out,
        req_s: chan<MemWriterReq> out,
        data_s: chan<MemWriterData> out,
        resp_r: chan<MemWriterResp> in,
     ) {
        spawn MemWriterMux<INST_ADDR_W, INST_DATA_W, u32:5>(
            sel_req_r, sel_resp_s,
            n_req_r, n_data_r, n_resp_s,
            req_s, data_s, resp_r,
        );

        ()
    }

    next(state: ()) {}
}

const TEST_ADDR_W = u32:32;
const TEST_DATA_W = u32:32;
const TEST_N = u32:5;
const TEST_N_WIDTH = std::clog2(TEST_N + u32:1);

#[test_proc]
proc MemWriterMuxTest {
    type MemWriterReq = mem_writer::MemWriterReq<TEST_ADDR_W>;
    type MemWriterResp = mem_writer::MemWriterResp;
    type MemWriterData = mem_writer::MemWriterDataPacket<TEST_DATA_W, TEST_ADDR_W>;

    type State = MemWriterMuxState<TEST_N>;
    type Sel = uN[TEST_N_WIDTH];

    terminator: chan<bool> out;

    sel_req_s: chan<Sel> out;
    sel_resp_r: chan<()> in;

    n_req_s: chan<MemWriterReq>[TEST_N] out;
    n_data_s: chan<MemWriterData>[TEST_N] out;
    n_resp_r: chan<MemWriterResp>[TEST_N] in;

    req_r: chan<MemWriterReq> in;
    data_r: chan<MemWriterData> in;
    resp_s: chan<MemWriterResp> out;

    init {}
    config(terminator: chan<bool> out,) {

        let (sel_req_s, sel_req_r) = chan<Sel>("sel_req");
        let (sel_resp_s, sel_resp_r) = chan<()>("sel_resp");

        let (n_req_s, n_req_r) = chan<MemWriterReq>[TEST_N]("n_req");
        let (n_data_s, n_data_r) = chan<MemWriterData>[TEST_N]("n_data");
        let (n_resp_s, n_resp_r) = chan<MemWriterResp>[TEST_N]("n_resp");

        let (req_s, req_r) = chan<MemWriterReq>("req");
        let (data_s, data_r) = chan<MemWriterData>("data");
        let (resp_s, resp_r) = chan<MemWriterResp>("resp");

        spawn MemWriterMux<TEST_ADDR_W, TEST_DATA_W, TEST_N>(
            sel_req_r, sel_resp_s,
            n_req_r, n_data_r, n_resp_s,
            req_s, data_s, resp_r,
        );

        (
            terminator,
            sel_req_s, sel_resp_r,
            n_req_s, n_data_s, n_resp_r,
            req_r, data_r, resp_s,
        )
    }

    next(state: ()) {
        let tok = join();

        let tok = send(tok, sel_req_s, Sel:3);
        let (tok, _) = recv(tok, sel_resp_r);

        let tok = send(tok, n_req_s[3], zero!<MemWriterReq>());
        let (tok, _) = recv(tok, req_r);

        // Cannot switch during the transmission
        let tok = send(tok, sel_req_s, Sel:1);

        let tok = send(tok, n_data_s[3], zero!<MemWriterData>());
        let (tok, _) = recv(tok, data_r);

        let tok = send(tok, resp_s, zero!<MemWriterResp>());
        let (tok, _) = recv(tok, n_resp_r[3]);

        // Now we should be able to receive the select
        let (tok, _) = recv(tok, sel_resp_r);

        let tok = send(tok, n_req_s[1], zero!<MemWriterReq>());
        let (tok, _) = recv(tok, req_r);

        let tok = send(tok, n_data_s[1], zero!<MemWriterData>());
        let (tok, _) = recv(tok, data_r);

        let tok = send(tok, resp_s, zero!<MemWriterResp>());
        let (tok, _) = recv(tok, n_resp_r[1]);

        send(tok, terminator, true);
    }
}
