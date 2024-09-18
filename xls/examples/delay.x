// Copyright 2022 The XLS Authors
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

// This file implements a parametric delay proc implemented using external RAM.
//
// The delay proc reproduces the input you send to it DELAY transactions after
// you sent them. The first DELAY values are chosen via a parameter to the proc
// called INIT_DATA. The delay is implemented with a double-wide single-port
// RAM. It alternates between reading and writing two values to the RAM, and the
// delay proc's state holds on to the other read/write depending on whether it
// is reading from or writing to the RAM. This proc can run fully pipelined,
// i.e. consuming an input and producing an output every cycle.
import std;

import xls.examples.ram;

type RamReq = ram::RWRamReq;
type RamResp = ram::RWRamResp;

fn double(x: u32) -> u32 { x * u32:2 }
fn addr_width(size: u32) -> u32 { std::clog2(size) }
fn half_floor(size: u32) -> u32 { (size >> u32: 1) }
fn is_odd(size: u32) -> bool { size[0:1] != u1:0 }

struct DelayState<DATA_WIDTH:u32, ADDR_WIDTH:u32> {
  // Current index into the RAM
  idx: bits[ADDR_WIDTH],
  // For the first DELAY cycles, the RAM has not been filled and whatever you
  // read from it will be garbage, during which init_done will be false. When
  // the RAM has been filled, init_done will be true and reads will be valid.
  init_done: bool,
  // If this is a write stage, keeps the second output from the previous read
  // stage.
  prev_read: bits[DATA_WIDTH],
  // If this is a write stage, keeps the previous input from the previous read
  // stage.
  prev_write: bits[DATA_WIDTH],
  // If true, delay is in the read stage, else it is a write stage.
  is_read_stage: bool,
}

// Parametric proc that has either 0 or 1 delay. Used to handle the case where a
// delay proc has odd delay.
proc Delay0or1<DATA_WIDTH:u32, DELAY_IS_ONE:bool, INIT_DATA:u32> {
    data_in: chan<bits[DATA_WIDTH]> in;
    data_out: chan<bits[DATA_WIDTH]> out;

    init { INIT_DATA }

    config(data_in: chan<bits[DATA_WIDTH]> in,
        data_out: chan<bits[DATA_WIDTH]> out) {
        (data_in, data_out)
    }

    next (prev_recv: bits[DATA_WIDTH]) {
        let (recv_tok, next_recv) = recv(join(), data_in);
        let to_send = if (DELAY_IS_ONE) { prev_recv } else { next_recv };
        trace!(DELAY_IS_ONE);
        let send_tok = send(join(), data_out, to_send);
        (next_recv)
    }
}

// Only works on DELAY even, the wrapper handles the case with odd-DELAY values.
proc DelayInternal<DATA_WIDTH:u32, DELAY:u32, INIT_DATA:u32={u32:0},
                   ADDR_WIDTH:u32={addr_width(half_floor(DELAY))},
                   DOUBLE_DATA_WIDTH:u32={double(DATA_WIDTH)},
                   HALF_FLOOR_DELAY:u32={half_floor(DELAY)}> {
    data_in: chan<bits[DATA_WIDTH]> in;
    data_out: chan<bits[DATA_WIDTH]> out;
    ram_req: chan<RamReq<ADDR_WIDTH, DOUBLE_DATA_WIDTH, 0>> out;
    ram_resp: chan<RamResp<DOUBLE_DATA_WIDTH>> in;
    ram_wr_comp: chan<()> in;
    type DelayInternalRamReq = RamReq<ADDR_WIDTH, DOUBLE_DATA_WIDTH, 0>;

    init {
        DelayState {
            idx: bits[ADDR_WIDTH]: 0,
            init_done: false,
            prev_read: bits[DATA_WIDTH]: 0,
            prev_write: bits[DATA_WIDTH]: 0,
            is_read_stage: true,
        }
    }

    config(data_in: chan<bits[DATA_WIDTH]> in,
           data_out: chan<bits[DATA_WIDTH]> out,
           ram_req: chan<RamReq<ADDR_WIDTH, DOUBLE_DATA_WIDTH, 0>> out,
           ram_resp: chan<RamResp<DOUBLE_DATA_WIDTH>> in,
           ram_wr_comp: chan<()> in) {
        (data_in, data_out, ram_req, ram_resp, ram_wr_comp)
    }

    next(state: DelayState<DATA_WIDTH, ADDR_WIDTH>) {
        let we = !state.is_read_stage;
        let re = state.is_read_stage && state.init_done;

        let (tok, next_write) = recv(join(), data_in);

        let data = next_write ++ state.prev_write;

        let tok = send(tok, ram_req, DelayInternalRamReq {
            addr: state.idx,
            data: data,
            write_mask: (),
            read_mask: (),
            we: we,
            re: re,
        });
        let zero_resp = RamResp<DOUBLE_DATA_WIDTH> {
          data: bits[DOUBLE_DATA_WIDTH]:0
        };
        let (read_tok, resp) = recv_if(tok, ram_resp, re, zero_resp);
        let empty_tuple: () = ();
        let (write_tok, _) = recv_if(tok, ram_wr_comp, we, empty_tuple);
        let tok = join(read_tok, write_tok);

        let current_data = if state.init_done {
            if state.is_read_stage {
                resp.data[0:(DATA_WIDTH as s32)]
            } else {
                state.prev_read
            }
        } else {
            INIT_DATA
        };

        let tok = send(read_tok, data_out, current_data);

        if state.is_read_stage {
            let next_data = resp.data[(DATA_WIDTH as s32):];
            DelayState {
                prev_read: next_data,
                prev_write: next_write,
                is_read_stage: false,
                ..state
            }
        } else {
            let next_idx = state.idx + bits[ADDR_WIDTH]:1;
            let next_idx = if next_idx as u32 >= half_floor(DELAY) {
                bits[ADDR_WIDTH]: 0
            } else {
                next_idx
            };
            let next_init_done = state.init_done || (next_idx == uN[ADDR_WIDTH]:0);
            DelayState {
                idx: next_idx,
                is_read_stage: true,
                init_done: next_init_done,
                ..state
            }
        }
    }
}

// A proc that implements a delay.
// For the first DELAY transactions, the output is INIT_DATA. After that, the
// output transactions are the inputs delayed by DELAY.
pub proc Delay<DATA_WIDTH:u32, DELAY:u32, INIT_DATA:u32={u32:0},
               ADDR_WIDTH:u32={addr_width(half_floor(DELAY))},
               DOUBLE_DATA_WIDTH:u32={double(DATA_WIDTH)},
               HALF_FLOOR_DELAY:u32={half_floor(DELAY)},
               DELAY_IS_ODD:bool={is_odd(DELAY)}> {
    data_in: chan<bits[DATA_WIDTH]> in;
    data_out: chan<bits[DATA_WIDTH]> out;
    ram_req: chan<RamReq<ADDR_WIDTH, DOUBLE_DATA_WIDTH, 0>> out;
    ram_resp: chan<RamResp<DOUBLE_DATA_WIDTH>> in;
    ram_wr_comp: chan<()> in;

    init { () }

    config(data_in: chan<bits[DATA_WIDTH]> in,
        data_out: chan<bits[DATA_WIDTH]> out,
        ram_req: chan<RamReq<ADDR_WIDTH, DOUBLE_DATA_WIDTH, 0>> out,
        ram_resp: chan<RamResp<DOUBLE_DATA_WIDTH>> in,
        ram_wr_comp: chan<()> in) {
        let (internal_data_s, internal_data_r) = chan<bits[DATA_WIDTH]>("even_delay");
        spawn DelayInternal<DATA_WIDTH, DELAY, INIT_DATA, ADDR_WIDTH,
                            DOUBLE_DATA_WIDTH, HALF_FLOOR_DELAY>(
            data_in, internal_data_s, ram_req, ram_resp, ram_wr_comp);
        spawn Delay0or1<DATA_WIDTH, DELAY_IS_ODD, INIT_DATA>(internal_data_r, data_out);
        (data_in, data_out, ram_req, ram_resp, ram_wr_comp)
    }

    next (state: ()) {
        ()
    }
}


// Define a concretized Delay for codegen.
type DelayRamReq32x2048 = RamReq<10, 64, 0>;
type DelayRamResp32x2048 = RamResp<64>;

pub proc Delay32x2048_init3 {
    data_in: chan<u32> in;
    data_out: chan<u32> out;
    ram_req: chan<DelayRamReq32x2048> out;
    ram_resp: chan<DelayRamResp32x2048> in;
    ram_wr_comp: chan<()> in;

    init { () }

    config(data_in: chan<u32> in, data_out: chan<u32> out,
           ram_req: chan<DelayRamReq32x2048> out,
           ram_resp: chan<DelayRamResp32x2048> in,
           ram_wr_comp: chan<()> in) {
        spawn Delay<u32:32, u32:2048, u32:3>(
            data_in, data_out, ram_req, ram_resp, ram_wr_comp);
        (data_in, data_out, ram_req, ram_resp, ram_wr_comp)
    }

    next(state: ()) {
        ()
    }
}

const TEST0_DELAY = u32:2048;

#[test_proc]
proc delay_smoke_test_even {
    data_in_r: chan<u32> out;
    data_out_s: chan<u32> in;
    terminator: chan<bool> out;

    init { () }

    config(terminator: chan<bool> out) {
        let (ram_req_s, ram_req_r) = chan<RamReq<10, 64, 0>>("ram_req");
        let (ram_resp_s, ram_resp_r) = chan<RamResp<64>>("ram_resp");
        let (ram_wr_comp_s, ram_wr_comp_r) = chan<()>("ram_wr_comp");
        spawn ram::SinglePortRamModel<u32:64, u32:1024>(
            ram_req_r, ram_resp_s, ram_wr_comp_s);

        let (data_in_s, data_in_r) = chan<u32>("data_in");
        let (data_out_s, data_out_r) = chan<u32>("data_out");
        spawn Delay<u32:32, TEST0_DELAY, u32:3>(
            data_in_r, data_out_s, ram_req_s, ram_resp_r, ram_wr_comp_r);

        (data_in_s, data_out_r, terminator)
    }

    next(state: ()) {
        let stok = for (i, tok): (u32, token) in range(u32:0, TEST0_DELAY*u32:5) {
            trace!(i);
            send(tok, data_in_r, i)
        } (join());
        // first, receive the inits
        let rtok = for (i, tok): (u32, token) in range(u32:0, TEST0_DELAY) {
            trace!(i);
            let (tok, result) = recv(tok, data_out_s);
            assert_eq(result, u32:3);
            tok
        } (join());
        // after the inits, check the delayed outputs
        let rtok = for (i, tok) : (u32, token) in range(u32:0, TEST0_DELAY*u32:4) {
            trace!(i);
            let (tok, result) = recv(tok, data_out_s);
            assert_eq(result, i);
            tok
        } (rtok);

        let tok = join(stok, rtok);
        send(tok, terminator, true);
    }
}

const TEST1_DELAY = u32:2047;

#[test_proc]
    proc delay_smoke_test_odd {
    data_in_r: chan<u32> out;
    data_out_s: chan<u32> in;
    terminator: chan<bool> out;

    init { () }

    config(terminator: chan<bool> out) {
        let (ram_req_s, ram_req_r) = chan<RamReq<10, 64, 0>>("ram_req");
        let (ram_resp_s, ram_resp_r) = chan<RamResp<64>>("ram_resp");
        let (ram_wr_comp_s, ram_wr_comp_r) = chan<()>("ram_wr_comp");
        spawn ram::SinglePortRamModel<u32:64, u32:1024>(
            ram_req_r, ram_resp_s, ram_wr_comp_s);

        let (data_in_s, data_in_r) = chan<u32>("data_in");
        let (data_out_s, data_out_r) = chan<u32>("data_out");
        spawn Delay<u32:32, TEST1_DELAY, u32:3>(
            data_in_r, data_out_s, ram_req_s, ram_resp_r, ram_wr_comp_r);
        (data_in_s, data_out_r, terminator)
    }

    next(state: ()) {
        let stok = for (i, tok): (u32, token) in range(u32:0, TEST1_DELAY*u32:5) {
            send(tok, data_in_r, i)
        } (join());
        // first, receive the inits
        let rtok = for (_, tok): (u32, token) in range(u32:0, TEST1_DELAY) {
            let (tok, result) = recv(tok, data_out_s);
            assert_eq(result, u32:3);
            tok
        } (join());
        // after the inits, check the delayed outputs
        let rtok = for (i, tok): (u32, token) in range(u32:0, TEST1_DELAY*u32:4) {
            let (tok, result) = recv(tok, data_out_s);
            assert_eq(result, i);
            tok
        } (rtok);

        let tok = join(stok, rtok);
        send(tok, terminator, true);
    }
}
