// Copyright 2023-2024 The XLS Authors
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

// Generic Physical Function
//
// The gpf proc mocks a real accelerator function.
// Performed algorithm is meant to be simple, i.e.
// output data = input data + 1

import xls.modules.dma.bus.axi_st_pkg;

enum PfBehavior : u1 {
    INCREMENT = 0,
    INVERT = 1,
}

pub fn pf_incr<N: u32>(d: uN[N]) -> uN[N] { d + uN[N]:1 }

pub fn pf_inv<N: u32>(d: uN[N]) -> uN[N] { !d & uN[N]:0xff }

#[test]
fn test_pf() {
    assert_eq(pf_incr(u32:0), u32:1);
    assert_eq(pf_incr(u32:1), u32:2);
    assert_eq(pf_incr(u32:100), u32:101);

    assert_eq(pf_inv(u32:0x00), u32:0xff);
    assert_eq(pf_inv(u32:0x0f), u32:0xf0);
    assert_eq(pf_inv(u32:0xf0), u32:0x0f);
}

type AxiStreamBundle = axi_st_pkg::AxiStreamBundle;

proc gpf<DATA_W: u32, DATA_W_DIV8: u32, DEST_W: u32, ID_W: u32, PF_BEHAVIOR: PfBehavior> {
    ch_i: chan<AxiStreamBundle<DATA_W, DATA_W_DIV8, ID_W, DEST_W>> in;
    ch_o: chan<AxiStreamBundle<DATA_W, DATA_W_DIV8, ID_W, DEST_W>> out;

    config(ch_i: chan<AxiStreamBundle<DATA_W, DATA_W_DIV8, ID_W, DEST_W>> in,
           ch_o: chan<AxiStreamBundle<DATA_W, DATA_W_DIV8, ID_W, DEST_W>> out) {
        (ch_i, ch_o)
    }

    init { u32:0 }

    next(tok: token, state: u32) {
        trace!(state);
        let (tok, read_data) = recv(tok, ch_i);
        let state = state + u32:1;

        let data = if PF_BEHAVIOR == PfBehavior::INCREMENT {
            pf_incr(read_data.tdata)
        } else if PF_BEHAVIOR == PfBehavior::INVERT {
            pf_inv(read_data.tdata)
        } else {
            pf_incr(read_data.tdata)
        };

        let axi_packet = AxiStreamBundle<DATA_W, DATA_W_DIV8, ID_W, DEST_W> {
            tdata: data,
            tstr: read_data.tstr,
            tkeep: read_data.tkeep,
            tlast: read_data.tlast,
            tid: read_data.tid,
            tdest: read_data.tdest
        };
        let tok = send(tok, ch_o, axi_packet);
        trace_fmt!("GPF: sent packet #{} to output", state);
        state
    }
}

const TEST_0_DATA_W = u32:8;
const TEST_0_DATA_W_DIV8 = u32:1;
const TEST_0_ID_W = u32:1;
const TEST_0_DEST_W = u32:1;

#[test_proc]
proc test_gpf_increment {
    ch_i: chan<AxiStreamBundle<TEST_0_DATA_W, TEST_0_DATA_W_DIV8, TEST_0_ID_W, TEST_0_DEST_W>> out;
    ch_o: chan<AxiStreamBundle<TEST_0_DATA_W, TEST_0_DATA_W_DIV8, TEST_0_ID_W, TEST_0_DEST_W>> in;
    terminator: chan<bool> out;

    config(terminator: chan<bool> out) {
        let (ch_i_s, ch_i_r) =
            chan<AxiStreamBundle<TEST_0_DATA_W, TEST_0_DATA_W_DIV8, TEST_0_ID_W, TEST_0_DEST_W>>;
        let (ch_o_s, ch_o_r) =
            chan<AxiStreamBundle<TEST_0_DATA_W, TEST_0_DATA_W_DIV8, TEST_0_ID_W, TEST_0_DEST_W>>;
        spawn gpf<
            TEST_0_DATA_W, TEST_0_DATA_W_DIV8, TEST_0_ID_W, TEST_0_DEST_W, PfBehavior::INCREMENT>(
            ch_i_r, ch_o_s);
        (ch_i_s, ch_o_r, terminator)
    }

    init { () }

    next(tok: token, state: ()) {
        let data = uN[TEST_0_DATA_W]:15;
        let axi_packet = AxiStreamBundle<TEST_0_DATA_W, TEST_0_DATA_W_DIV8, TEST_0_ID_W, TEST_0_DEST_W>
        {
            tdata: data,
            tstr: uN[TEST_0_DATA_W_DIV8]:0,
            tkeep: uN[TEST_0_DATA_W_DIV8]:0,
            tlast: u1:1,
            tid: uN[TEST_0_ID_W]:0,
            tdest: uN[TEST_0_DEST_W]:0
        };

        let tok = send(tok, ch_i, axi_packet);
        let (tok, axi_packet_r) = recv(tok, ch_o);
        let r_data = axi_packet_r.tdata;

        trace_fmt!("Data W: {}, Data R: {}", data, r_data);
        assert_eq(data + uN[TEST_0_DATA_W]:1, r_data);

        let tok = send(tok, terminator, true);
    }
}

#[test_proc]
proc test_gpf_invert {
    ch_i: chan<AxiStreamBundle<TEST_0_DATA_W, TEST_0_DATA_W_DIV8, TEST_0_ID_W, TEST_0_DEST_W>> out;
    ch_o: chan<AxiStreamBundle<TEST_0_DATA_W, TEST_0_DATA_W_DIV8, TEST_0_ID_W, TEST_0_DEST_W>> in;
    terminator: chan<bool> out;

    config(terminator: chan<bool> out) {
        let (ch_i_s, ch_i_r) =
            chan<AxiStreamBundle<TEST_0_DATA_W, TEST_0_DATA_W_DIV8, TEST_0_ID_W, TEST_0_DEST_W>>;
        let (ch_o_s, ch_o_r) =
            chan<AxiStreamBundle<TEST_0_DATA_W, TEST_0_DATA_W_DIV8, TEST_0_ID_W, TEST_0_DEST_W>>;
        spawn gpf<TEST_0_DATA_W, TEST_0_DATA_W_DIV8, TEST_0_ID_W, TEST_0_DEST_W, PfBehavior::INVERT>(
            ch_i_r, ch_o_s);
        (ch_i_s, ch_o_r, terminator)
    }

    init { () }

    next(tok: token, state: ()) {
        let data = uN[TEST_0_DATA_W]:15;
        let axi_packet = AxiStreamBundle<TEST_0_DATA_W, TEST_0_DATA_W_DIV8, TEST_0_ID_W, TEST_0_DEST_W>
        {
            tdata: data,
            tstr: uN[TEST_0_DATA_W_DIV8]:0,
            tkeep: uN[TEST_0_DATA_W_DIV8]:0,
            tlast: u1:1,
            tid: uN[TEST_0_ID_W]:0,
            tdest: uN[TEST_0_DEST_W]:0
        };

        let tok = send(tok, ch_i, axi_packet);
        let (tok, axi_packet_r) = recv(tok, ch_o);
        let r_data = axi_packet_r.tdata;

        trace_fmt!("Data W: {}, Data R: {}", data, r_data);
        assert_eq(!data & uN[TEST_0_DATA_W]:0xff, r_data);

        let tok = send(tok, terminator, true);
    }
}
