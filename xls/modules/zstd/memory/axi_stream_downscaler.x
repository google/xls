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


// This file contains implementation of AxiStreamDownscaler that can be used
// to convert AxiStream transactions from a wider bus, to multiple transactions
// on more narrow bus.

import std;
import xls.modules.zstd.memory.axi_st;

struct AxiStreamDownscalerState<
    IN_W: u32, OUT_W: u32, DEST_W: u32, ID_W: u32,
    IN_W_DIV8: u32, // = {IN_W / u32:8},
    RATIO_W: u32, // = {std::clog2((IN_W / OUT_W) + u32:1)},
> {
    in_data: axi_st::AxiStream<IN_W, DEST_W, ID_W, IN_W_DIV8>,
    i: uN[RATIO_W],
}

// A proc responsible for converting Axi Stream transactions from a wider bus,
// to multiple transactions on more narrow bus
pub proc AxiStreamDownscaler<
    IN_W: u32, OUT_W: u32, DEST_W: u32, ID_W: u32,
    IN_W_DIV8: u32 = {IN_W / u32:8},
    OUT_W_DIV8: u32 = {OUT_W / u32:8},
    RATIO: u32 = {IN_W / OUT_W},
    RATIO_W: u32 = {std::clog2((IN_W / OUT_W) + u32:1)}
> {
    type State = AxiStreamDownscalerState<IN_W, OUT_W, DEST_W, ID_W, IN_W_DIV8, RATIO_W>;
    type InStream = axi_st::AxiStream<IN_W, DEST_W, ID_W, IN_W_DIV8>;
    type OutStream = axi_st::AxiStream<OUT_W, DEST_W, ID_W, OUT_W_DIV8>;

    // Assumptions related to parameters
    const_assert!(IN_W >= OUT_W); // input should be wider than output
    const_assert!(IN_W % OUT_W == u32:0); // output width should be a multiple of input width

    // checks for parameters
    const_assert!(RATIO == IN_W / OUT_W);
    const_assert!(RATIO_W == std::clog2((IN_W / OUT_W) + u32:1));

    in_r: chan<InStream> in;
    out_s: chan<OutStream> out;

    config(
        in_r: chan<InStream> in,
        out_s: chan<OutStream> out
    ) { (in_r, out_s) }

    init { zero!<State>() }

    next(state: State) {
        const MAX_ITER = RATIO as uN[RATIO_W] - uN[RATIO_W]:1;

        let tok0 = join();

        let do_recv = (state.i == uN[RATIO_W]:0);
        let (tok1, in_data) = recv_if(tok0, in_r, do_recv, state.in_data);

        let is_last_iter = (state.i == MAX_ITER);

        let data = in_data.data[OUT_W      * state.i as u32 +: uN[OUT_W]];
        let keep = in_data.keep[OUT_W_DIV8 * state.i as u32 +: uN[OUT_W_DIV8]];
        let str  =  in_data.str[OUT_W_DIV8 * state.i as u32 +: uN[OUT_W_DIV8]];
        let id   = in_data.id;
        let dest = in_data.dest;
        let last = if is_last_iter { in_data.last } else { u1:0 };

        let out_data = OutStream { data, keep, str, last, id, dest };

        let tok = send(tok1, out_s, out_data);

        if is_last_iter {
            zero!<State>()
        } else {
            let i = state.i + uN[RATIO_W]:1;
            State { in_data, i }
        }
    }
}


const INST_IN_W = u32:128;
const INST_IN_W_DIV8 = INST_IN_W / u32:8;
const INST_OUT_W = u32:32;
const INST_OUT_W_DIV8 = INST_OUT_W / u32:8;
const INST_DEST_W = u32:8;
const INST_ID_W = u32:8;

proc AxiStreamDownscalerInst {
    type InStream = axi_st::AxiStream<INST_IN_W, INST_DEST_W, INST_ID_W, INST_IN_W_DIV8>;
    type OutStream = axi_st::AxiStream<INST_OUT_W, INST_DEST_W, INST_ID_W, INST_OUT_W_DIV8>;

    config(
        in_r: chan<InStream> in,
        out_s: chan<OutStream> out
    ) {
        spawn AxiStreamDownscaler<
            INST_IN_W, INST_OUT_W, INST_DEST_W, INST_ID_W
        >(in_r, out_s);
    }

    init {  }

    next(state: ()) {  }
}


const TEST_IN_W = u32:128;
const TEST_IN_W_DIV8 = TEST_IN_W / u32:8;
const TEST_OUT_W = u32:32;
const TEST_OUT_W_DIV8 = TEST_OUT_W / u32:8;
const TEST_DEST_W = u32:8;
const TEST_ID_W = u32:8;
const TEST_RATIO = TEST_IN_W / TEST_OUT_W;
const TEST_RATIO_W = std::clog2((TEST_IN_W / TEST_OUT_W) + u32:1);

#[test_proc]
proc AxiStreamWitdhDownscalerTest {
    type InStream = axi_st::AxiStream<TEST_IN_W, TEST_DEST_W, TEST_ID_W, TEST_IN_W_DIV8>;
    type OutStream = axi_st::AxiStream<TEST_OUT_W, TEST_DEST_W, TEST_ID_W, TEST_OUT_W_DIV8>;
    type InData = uN[TEST_IN_W];
    type InStr = uN[TEST_IN_W_DIV8];
    type InKeep = uN[TEST_IN_W_DIV8];
    type OutData = uN[TEST_OUT_W];
    type OutStr = uN[TEST_OUT_W_DIV8];
    type OutKeep = uN[TEST_OUT_W_DIV8];
    type Id = uN[TEST_ID_W];
    type Dest = uN[TEST_DEST_W];

    terminator: chan<bool> out;
    in_s: chan<InStream> out;
    out_r: chan<OutStream> in;

    config(terminator: chan<bool> out) {
        let (in_s, in_r) = chan<InStream>("in");
        let (out_s, out_r) = chan<OutStream>("out");

        spawn AxiStreamDownscaler<
            TEST_IN_W, TEST_OUT_W, TEST_DEST_W, TEST_ID_W
        > (in_r, out_s);

        (terminator, in_s, out_r)
    }

    init { }

    next(state: ()) {
        let tok = join();

        // Test 1
        let tok = send(tok, in_s, InStream {
            data: InData:0xAAAA_BBBB_CCCC_DDDD_1111_2222_3333_4444,
            str: InStr:0x0FF0,
            keep: InKeep:0x0FF0,
            last: u1:1,
            id: Id:0xAB,
            dest: Dest:0xCD
        });

        let (tok, data) = recv(tok, out_r);
        assert_eq(data, OutStream {
            data: OutData:0x3333_4444,
            str: OutStr:0x0,
            keep: OutKeep:0x0,
            last: u1:0,
            id: Id:0xAB,
            dest: Dest:0xCD
        });

        let (tok, data) = recv(tok, out_r);
        assert_eq(data, OutStream {
            data: OutData:0x1111_2222,
            str: OutStr:0xF,
            keep: OutKeep:0xF,
            last: u1:0,
            id: Id:0xAB,
            dest: Dest:0xCD
        });

        let (tok, data) = recv(tok, out_r);
        assert_eq(data, OutStream {
            data: OutData:0xCCCC_DDDD,
            str: OutStr:0xF,
            keep: OutKeep:0xF,
            last: u1:0,
            id: Id:0xAB,
            dest: Dest:0xCD
        });

        let (tok, data) = recv(tok, out_r);
        assert_eq(data, OutStream {
            data: OutData:0xAAAA_BBBB,
            str: OutStr:0x0,
            keep: OutKeep:0x0,
            last: u1:1,
            id: Id:0xAB,
            dest: Dest:0xCD
        });

        // Test 2
        let tok = send(tok, in_s, InStream {
            data: InData:0xAAAA_BBBB_CCCC_DDDD_1111_2222_3333_4444,
            str: InStr:0x1234,
            keep: InKeep:0x1234,
            last: u1:0,
            id: Id:0xAB,
            dest: Dest:0xCD
        });

        let (tok, data) = recv(tok, out_r);
        assert_eq(data, OutStream {
            data: OutData:0x3333_4444,
            str: OutStr:0x4,
            keep: OutKeep:0x4,
            last: u1:0,
            id: Id:0xAB,
            dest: Dest:0xCD
        });

        let (tok, data) = recv(tok, out_r);
        assert_eq(data, OutStream {
            data: OutData:0x1111_2222,
            str: OutStr:0x3,
            keep: OutKeep:0x3,
            last: u1:0,
            id: Id:0xAB,
            dest: Dest:0xCD
        });

        let (tok, data) = recv(tok, out_r);
        assert_eq(data, OutStream {
            data: OutData:0xCCCC_DDDD,
            str: OutStr:0x2,
            keep: OutKeep:0x2,
            last: u1:0,
            id: Id:0xAB,
            dest: Dest:0xCD
        });

        let (tok, data) = recv(tok, out_r);
        assert_eq(data, OutStream {
            data: OutData:0xAAAA_BBBB,
            str: OutStr:0x1,
            keep: OutKeep:0x1,
            last: u1:0,
            id: Id:0xAB,
            dest: Dest:0xCD
        });

        send(tok, terminator, true);
    }
}
