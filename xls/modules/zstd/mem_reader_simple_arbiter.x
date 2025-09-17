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
import xls.modules.zstd.mem_reader_mux;
import xls.modules.zstd.memory.mem_reader;

struct MemReaderSimpleArbiterState<
    N: u32,
    N_WIDTH: u32 = {std::clog2(N + u32:1)}
>{
    cnt: uN[N_WIDTH],
}

pub proc MemReaderSimpleArbiter<
    ADDR_W: u32, DATA_W: u32, N: u32,
    N_WIDTH: u32 = {std::clog2(N + u32:1)}
> {
    type MemReaderReq = mem_reader::MemReaderReq<ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<DATA_W, ADDR_W>;

    type Sel = uN[N_WIDTH];
    type State = MemReaderSimpleArbiterState<N>;

    sel_req_s: chan<Sel> out;
    sel_resp_r: chan<()> in;

    config(
        n_req_r: chan<MemReaderReq>[N] in,
        n_resp_s: chan<MemReaderResp>[N] out,
        req_s: chan<MemReaderReq> out,
        resp_r: chan<MemReaderResp> in,
    ) {

        let (sel_req_s, sel_req_r) = chan<Sel>("sel_req");
        let (sel_resp_s, sel_resp_r) = chan<()>("sel_resp");

        spawn mem_reader_mux::MemReaderMux<ADDR_W, DATA_W, N>(
            sel_req_r, sel_resp_s,
            n_req_r, n_resp_s,
            req_s, resp_r,
        );

        (sel_req_s, sel_resp_r)
    }

    init { zero!<State>() }

    next(state: State) {
        let tok0 = join();

        let tok = send(join(), sel_req_s, state.cnt);
        let (tok, _) = recv(tok, sel_resp_r);

        if state.cnt == N as uN[N_WIDTH] - uN[N_WIDTH]:1 {
            zero!<State>()
        } else {
            State {cnt: state.cnt + uN[N_WIDTH]:1 }
        }
    }
}

const TEST_ADDR_W = u32:32;
const TEST_DATA_W = u32:32;
const TEST_NUM_PARTITIONS = u32:32;
const TEST_N = u32:5;
const TEST_N_WIDTH = std::clog2(TEST_N + u32:1);

#[test_proc]
proc MemReaderSimpleArbiterTest {
    type MemReaderReq = mem_reader::MemReaderReq<TEST_ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<TEST_DATA_W, TEST_ADDR_W>;

    type Addr = uN[TEST_ADDR_W];
    type Length = uN[TEST_ADDR_W];

    terminator: chan<bool> out;

    n_req_s: chan<MemReaderReq>[TEST_N] out;
    n_resp_r: chan<MemReaderResp>[TEST_N] in;

    req_r: chan<MemReaderReq> in;
    resp_s: chan<MemReaderResp> out;

    init {}
    config(terminator: chan<bool> out,) {

        let (n_req_s, n_req_r) = chan<MemReaderReq>[TEST_N]("n_req");
        let (n_resp_s, n_resp_r) = chan<MemReaderResp>[TEST_N]("n_resp");

        let (req_s, req_r) = chan<MemReaderReq>("req");
        let (resp_s, resp_r) = chan<MemReaderResp>("resp");

        spawn MemReaderSimpleArbiter<TEST_ADDR_W, TEST_DATA_W, TEST_N>(
            n_req_r, n_resp_s,
            req_s, resp_r,
        );

        (
            terminator,
            n_req_s, n_resp_r,
            req_r, resp_s,
        )
    }

    next(state: ()) {
        let tok = unroll_for! (i, tok): (u32, token) in range(u32:0, TEST_N) {
            let req = MemReaderReq {
                addr: i as Addr,
                length: (TEST_DATA_W / u32:8) as Length,
            };
            trace_fmt!("Sending request {:#x} on channel: {}", req, i);
            let tok = send(tok, n_req_s[i], req);

            let resp = MemReaderResp {
                last: true,
                ..zero!<MemReaderResp>()
            };

            trace_fmt!("Sending response {:#x} on channel: {}", resp, i);
            let tok = send(tok, resp_s, resp);

            tok
        }(join());

        let tok = unroll_for! (i, tok): (u32, token) in range(u32:0, TEST_N) {
            let (tok, req) = recv(tok, req_r);
            trace_fmt!("Received req {:#x} on channel {}", req, i);
            let (tok, resp) = recv(tok, n_resp_r[i]);
            trace_fmt!("Received response {:#x} on channel: {}", resp, i);

            tok
        }(join());

        send(tok, terminator, true);
    }
}

