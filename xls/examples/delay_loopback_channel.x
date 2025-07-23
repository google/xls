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

// This implements a parametric delay proc implemented using loopback channels that are ultimately
// generated as FIFOs.

#![feature(type_inference_v2)]

import std;

// TODO(google/xls#889): Make top parametric when supported.
const DATA_WIDTH = u32:32;

const DELAY = u32:2048;

const INIT_DATA = u32:3;

const LOGDELAY = std::clog2(DELAY + u32:1);

struct DelayState { initial_output_count: uN[LOGDELAY], occupancy: uN[LOGDELAY] }

fn increment<N: u32>(x: uN[N]) -> uN[N] { x + uN[N]:1 }

fn decrement<N: u32>(x: uN[N]) -> uN[N] { x - uN[N]:1 }

fn max(x: u32, y: u32) -> u32 { if x > y { x } else { y } }

fn eq<M: u32, N: u32, MAX: u32 = {max(M, N)}>(x: uN[M], y: uN[N]) -> bool {
    x as uN[MAX] == y as uN[MAX]
}

// A proc that implements a delay.
// For the first DELAY transactions, the output is INIT_DATA. After that, the output transactions
// are the inputs delayed by DELAY.
pub proc Delay {
    data_in: chan<uN[DATA_WIDTH]> in;
    data_out: chan<uN[DATA_WIDTH]> out;
    loopback_push: chan<uN[DATA_WIDTH]> out;
    loopback_pop: chan<uN[DATA_WIDTH]> in;

    config(data_in: chan<bits[DATA_WIDTH]> in, data_out: chan<bits[DATA_WIDTH]> out) {
        let (loopback_push, loopback_pop) = chan<uN[DATA_WIDTH], DELAY>("loopback");
        (data_in, data_out, loopback_push, loopback_pop)
    }

    init {
        type MyDelay = DelayState;
        zero!<MyDelay>()
    }

    next(state: DelayState) {
        trace_fmt!("state = {}", state);
        let init_done = eq(state.initial_output_count, DELAY);
        let (recv_tok, input_data, data_in_valid) =
            recv_if_non_blocking(join(), data_in, !eq(state.occupancy, DELAY), uN[DATA_WIDTH]:0);
        let send_tok = send_if(recv_tok, loopback_push, data_in_valid, input_data);
        let (recv_tok, loopback_data, loopback_valid) =
            recv_if_non_blocking(join(), loopback_pop, init_done, INIT_DATA as uN[DATA_WIDTH]);
        let send_tok = send_if(recv_tok, data_out, !init_done || loopback_valid, loopback_data);
        trace_fmt!("data_in_valid={}, loopback_valid={}", data_in_valid, loopback_valid);
        let next_output_count = if init_done {
            state.initial_output_count
        } else {
            increment(state.initial_output_count)
        };
        let next_occupancy = match (data_in_valid, loopback_valid) {
            (true, false) => increment(state.occupancy),
            (false, true) => decrement(state.occupancy),
            _ => state.occupancy,
        };
        DelayState { initial_output_count: next_output_count, occupancy: next_occupancy }
    }
}

#[test_proc]
proc DelayTest {
    terminator: chan<bool> out;
    data_out: chan<uN[DATA_WIDTH]> out;
    data_in: chan<uN[DATA_WIDTH]> in;

    config(terminator: chan<bool> out) {
        let (data_in_c, data_in_p) = chan<uN[DATA_WIDTH], DELAY>("data_in");
        let (data_out_c, data_out_p) = chan<uN[DATA_WIDTH], DELAY>("data_out");
        spawn Delay(data_in_p, data_out_c);
        (terminator, data_in_c, data_out_p)
    }

    init { () }

    next(state: ()) {
        // Check that the first DELAY outputs are INIT_DATA.
        let tok = for (_, tok): (u32, token) in u32:0..DELAY {
            let (tok, value) = recv(tok, data_in);
            assert_eq(value, INIT_DATA);
            tok
        }(join());
        // Queue up a bunch of inputs.
        let tok = for (i, tok): (u32, token) in u32:0..u32:2 * DELAY {
            send(tok, data_out, i)
        }(tok);
        // Check the outputs.
        let tok = for (i, tok): (u32, token) in u32:0..u32:2 * DELAY {
            let (tok, value) = recv(tok, data_in);
            assert_eq(value, i);
            tok
        }(tok);
        send(tok, terminator, true);
    }
}
