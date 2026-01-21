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
import xls.examples.ram;

pub struct HistoryBufferReq<
    ADDR_W: u32,
    DATA_W: u32,
    NUM_PARTITIONS: u32> {
    addr: uN[ADDR_W],
    data: uN[DATA_W],
    write_mask: uN[NUM_PARTITIONS],
    read_mask: uN[NUM_PARTITIONS],
}

pub struct HistoryBufferResp<DATA_W: u32> {
    data: uN[DATA_W]
}

pub type HistoryBufferWrComp = ();

fn to_single_req<
    P_ADDR_W: u32, P_DATA_W: u32, P_NUM_PARTITIONS: u32, S_ADDR_W: u32, S_DATA_W: u32>
    (preq: HistoryBufferReq<P_ADDR_W, P_DATA_W, P_NUM_PARTITIONS>) ->
    (ram::RWRamReq<S_ADDR_W, S_DATA_W>[P_NUM_PARTITIONS], u32[P_NUM_PARTITIONS],
    bool[P_NUM_PARTITIONS], bool[P_NUM_PARTITIONS]) {

    let base = preq.addr / P_NUM_PARTITIONS;
    let offset = preq.addr % P_NUM_PARTITIONS;
    type RWRamReq = ram::RWRamReq<S_ADDR_W, S_DATA_W>;
    const ZERO_REQS = RWRamReq[P_NUM_PARTITIONS]:[zero!<RWRamReq>(), ...];
    const ZERO_ORDER = u32[P_NUM_PARTITIONS]:[u32:0, ...];
    const ZERO_DO_SEND = bool[P_NUM_PARTITIONS]:[false, ...];
    const ZERO_DO_COMP = bool[P_NUM_PARTITIONS]:[false, ...];

    unroll_for! (i, (reqs, poss, do_rd, do_wr)) :
        (u32, (RWRamReq[P_NUM_PARTITIONS], u32[P_NUM_PARTITIONS], bool[P_NUM_PARTITIONS],
            bool[P_NUM_PARTITIONS])) in u32:0..P_NUM_PARTITIONS {
        let bank = (i - offset as u32) % P_NUM_PARTITIONS;
        let data = preq.data[bank * u32:8 +: u8] as u8;
        let addr = (base + (i < offset as u32) as uN[P_ADDR_W]) as uN[S_ADDR_W];
        let we = preq.write_mask[bank +: u1];
        let re = preq.read_mask[bank +: u1];

        let req = ram::RWRamReq<S_ADDR_W, S_DATA_W> {
            addr, data, we, re,
            write_mask: (),
            read_mask: (),
        };

        trace_fmt!("bank = {:#x}, data = {:#x}, addr = {:#x}, we = {:#x}, re = {:#x}", bank, data, addr, we, re);

        let reqs = update(reqs, i, req);
        let poss = update(poss, i, bank as u32);
        let do_rd = update(do_rd, i, re);
        let do_wr = update(do_wr, i, we);
        (reqs, poss, do_rd, do_wr)
    }((ZERO_REQS, ZERO_ORDER, ZERO_DO_SEND, ZERO_DO_COMP))
}

const S_DATA_W = u32:8;
const P_ADDR_W = u32:32;

#[test]
fn to_single_req_64bit_width_test() {
    const P_DATA_W = u32:64;
    const P_NUM_PARTITIONS = P_DATA_W / S_DATA_W;
    const P_MASK_W = P_NUM_PARTITIONS;
    const S_ADDR_W = S_DATA_W - std::clog2(P_NUM_PARTITIONS);

    type PAddr = uN[P_ADDR_W];
    type PData = uN[P_DATA_W];
    type PMask = uN[P_MASK_W];
    type SAddr = uN[S_ADDR_W];
    type SData = uN[S_DATA_W];

    let req = HistoryBufferReq<P_ADDR_W, P_DATA_W, P_NUM_PARTITIONS> {
        addr: PAddr:0,
        data: PData:0x88_77_66_55_44_33_22_11,
        write_mask: PMask:0b1111_1111,
        read_mask: PMask:0b0000_0000,
    };
    let (reqs, orders, _, _) = to_single_req<P_ADDR_W, P_DATA_W, P_NUM_PARTITIONS, S_ADDR_W, S_DATA_W>(req);
    let exp_reqs_1 = ram::RWRamReq<S_ADDR_W, S_DATA_W>[8]:[
        ram::RWRamReq<S_ADDR_W, S_DATA_W> { addr: SAddr:0, data: SData:0x11, write_mask: (), read_mask: (), we: true, re: false },
        ram::RWRamReq<S_ADDR_W, S_DATA_W> { addr: SAddr:0, data: SData:0x22, write_mask: (), read_mask: (), we: true, re: false },
        ram::RWRamReq<S_ADDR_W, S_DATA_W> { addr: SAddr:0, data: SData:0x33, write_mask: (), read_mask: (), we: true, re: false },
        ram::RWRamReq<S_ADDR_W, S_DATA_W> { addr: SAddr:0, data: SData:0x44, write_mask: (), read_mask: (), we: true, re: false },
        ram::RWRamReq<S_ADDR_W, S_DATA_W> { addr: SAddr:0, data: SData:0x55, write_mask: (), read_mask: (), we: true, re: false },
        ram::RWRamReq<S_ADDR_W, S_DATA_W> { addr: SAddr:0, data: SData:0x66, write_mask: (), read_mask: (), we: true, re: false },
        ram::RWRamReq<S_ADDR_W, S_DATA_W> { addr: SAddr:0, data: SData:0x77, write_mask: (), read_mask: (), we: true, re: false },
        ram::RWRamReq<S_ADDR_W, S_DATA_W> { addr: SAddr:0, data: SData:0x88, write_mask: (), read_mask: (), we: true, re: false },
    ];
    trace_fmt!("Reqs: {}", reqs);
    trace_fmt!("Orders: {}", orders);
    assert_eq(reqs, exp_reqs_1);

    let req = HistoryBufferReq<P_ADDR_W, P_DATA_W, P_NUM_PARTITIONS> {
        addr: PAddr:0x1,
        data: PData:0xAA_BB_CC_DD_EE_FF_99_88,
        write_mask: PMask:0b1111_1111,
        read_mask: PMask:0b0000_0000,
    };
    let (reqs, orders, _, _) = to_single_req<P_ADDR_W, P_DATA_W, P_NUM_PARTITIONS, S_ADDR_W, S_DATA_W>(req);
    let exp_reqs_2 = ram::RWRamReq<S_ADDR_W, S_DATA_W>[8]:[
        ram::RWRamReq<S_ADDR_W, S_DATA_W> { addr: SAddr:1, data: SData:0xAA, write_mask: (), read_mask: (), we: true, re: false },
        ram::RWRamReq<S_ADDR_W, S_DATA_W> { addr: SAddr:0, data: SData:0x88, write_mask: (), read_mask: (), we: true, re: false },
        ram::RWRamReq<S_ADDR_W, S_DATA_W> { addr: SAddr:0, data: SData:0x99, write_mask: (), read_mask: (), we: true, re: false },
        ram::RWRamReq<S_ADDR_W, S_DATA_W> { addr: SAddr:0, data: SData:0xFF, write_mask: (), read_mask: (), we: true, re: false },
        ram::RWRamReq<S_ADDR_W, S_DATA_W> { addr: SAddr:0, data: SData:0xEE, write_mask: (), read_mask: (), we: true, re: false },
        ram::RWRamReq<S_ADDR_W, S_DATA_W> { addr: SAddr:0, data: SData:0xDD, write_mask: (), read_mask: (), we: true, re: false },
        ram::RWRamReq<S_ADDR_W, S_DATA_W> { addr: SAddr:0, data: SData:0xCC, write_mask: (), read_mask: (), we: true, re: false },
        ram::RWRamReq<S_ADDR_W, S_DATA_W> { addr: SAddr:0, data: SData:0xBB, write_mask: (), read_mask: (), we: true, re: false },
    ];
    trace_fmt!("Reqs: {}", reqs);
    trace_fmt!("Orders: {}", orders);

    assert_eq(reqs, exp_reqs_2);

    let req = HistoryBufferReq<P_ADDR_W, P_DATA_W, P_NUM_PARTITIONS> {
        addr: PAddr:0xA,
        data: PData:0x08_07_06_05_04_03_02_01,
        write_mask: PMask:0b1111_1111,
        read_mask: PMask:0b0000_0000,
    };
    let (reqs, orders, _, _) = to_single_req<P_ADDR_W, P_DATA_W, P_NUM_PARTITIONS, S_ADDR_W, S_DATA_W>(req);
    let exp_reqs_3 = ram::RWRamReq<S_ADDR_W, S_DATA_W>[8]:[
        ram::RWRamReq<S_ADDR_W, S_DATA_W> { addr: SAddr:2, data: SData:0x07, write_mask: (), read_mask: (), we: true, re: false },
        ram::RWRamReq<S_ADDR_W, S_DATA_W> { addr: SAddr:2, data: SData:0x08, write_mask: (), read_mask: (), we: true, re: false },
        ram::RWRamReq<S_ADDR_W, S_DATA_W> { addr: SAddr:1, data: SData:0x01, write_mask: (), read_mask: (), we: true, re: false },
        ram::RWRamReq<S_ADDR_W, S_DATA_W> { addr: SAddr:1, data: SData:0x02, write_mask: (), read_mask: (), we: true, re: false },
        ram::RWRamReq<S_ADDR_W, S_DATA_W> { addr: SAddr:1, data: SData:0x03, write_mask: (), read_mask: (), we: true, re: false },
        ram::RWRamReq<S_ADDR_W, S_DATA_W> { addr: SAddr:1, data: SData:0x04, write_mask: (), read_mask: (), we: true, re: false },
        ram::RWRamReq<S_ADDR_W, S_DATA_W> { addr: SAddr:1, data: SData:0x05, write_mask: (), read_mask: (), we: true, re: false },
        ram::RWRamReq<S_ADDR_W, S_DATA_W> { addr: SAddr:1, data: SData:0x06, write_mask: (), read_mask: (), we: true, re: false },
    ];
    trace_fmt!("Reqs: {}", reqs);
    trace_fmt!("Orders: {}", orders);

    assert_eq(reqs, exp_reqs_3);
}

#[test]
fn to_single_req_128bit_width_test() {
    const P_DATA_W = u32:128;
    const P_NUM_PARTITIONS = P_DATA_W / S_DATA_W;
    const P_MASK_W = P_NUM_PARTITIONS;
    const S_ADDR_W = S_DATA_W - std::clog2(P_NUM_PARTITIONS);

    type PAddr = uN[P_ADDR_W];
    type PData = uN[P_DATA_W];
    type PMask = uN[P_MASK_W];
    type SAddr = uN[S_ADDR_W];
    type SData = uN[S_DATA_W];

    let req = HistoryBufferReq<P_ADDR_W, P_DATA_W, P_NUM_PARTITIONS> {
        addr: PAddr:0xA,
        data: PData:0x10_0F_0E_0D_0C_0B_0A_09_08_07_06_05_04_03_02_01,
        write_mask: PMask:0b1111_1111_1111_1111,
        read_mask: PMask:0b0000_0000_0000_0000,
    };
    let (reqs, orders, _, _) = to_single_req<P_ADDR_W, P_DATA_W, P_NUM_PARTITIONS, S_ADDR_W, S_DATA_W>(req);
    let exp_reqs = ram::RWRamReq<S_ADDR_W, S_DATA_W>[P_NUM_PARTITIONS]:[
        ram::RWRamReq<S_ADDR_W, S_DATA_W> { addr: SAddr:1, data: SData:0x07, write_mask: (), read_mask: (), we: true, re: false },
        ram::RWRamReq<S_ADDR_W, S_DATA_W> { addr: SAddr:1, data: SData:0x08, write_mask: (), read_mask: (), we: true, re: false },
        ram::RWRamReq<S_ADDR_W, S_DATA_W> { addr: SAddr:1, data: SData:0x09, write_mask: (), read_mask: (), we: true, re: false },
        ram::RWRamReq<S_ADDR_W, S_DATA_W> { addr: SAddr:1, data: SData:0x0A, write_mask: (), read_mask: (), we: true, re: false },
        ram::RWRamReq<S_ADDR_W, S_DATA_W> { addr: SAddr:1, data: SData:0x0B, write_mask: (), read_mask: (), we: true, re: false },
        ram::RWRamReq<S_ADDR_W, S_DATA_W> { addr: SAddr:1, data: SData:0x0C, write_mask: (), read_mask: (), we: true, re: false },
        ram::RWRamReq<S_ADDR_W, S_DATA_W> { addr: SAddr:1, data: SData:0x0D, write_mask: (), read_mask: (), we: true, re: false },
        ram::RWRamReq<S_ADDR_W, S_DATA_W> { addr: SAddr:1, data: SData:0x0E, write_mask: (), read_mask: (), we: true, re: false },
        ram::RWRamReq<S_ADDR_W, S_DATA_W> { addr: SAddr:1, data: SData:0x0F, write_mask: (), read_mask: (), we: true, re: false },
        ram::RWRamReq<S_ADDR_W, S_DATA_W> { addr: SAddr:1, data: SData:0x10, write_mask: (), read_mask: (), we: true, re: false },
        ram::RWRamReq<S_ADDR_W, S_DATA_W> { addr: SAddr:0, data: SData:0x01, write_mask: (), read_mask: (), we: true, re: false },
        ram::RWRamReq<S_ADDR_W, S_DATA_W> { addr: SAddr:0, data: SData:0x02, write_mask: (), read_mask: (), we: true, re: false },
        ram::RWRamReq<S_ADDR_W, S_DATA_W> { addr: SAddr:0, data: SData:0x03, write_mask: (), read_mask: (), we: true, re: false },
        ram::RWRamReq<S_ADDR_W, S_DATA_W> { addr: SAddr:0, data: SData:0x04, write_mask: (), read_mask: (), we: true, re: false },
        ram::RWRamReq<S_ADDR_W, S_DATA_W> { addr: SAddr:0, data: SData:0x05, write_mask: (), read_mask: (), we: true, re: false },
        ram::RWRamReq<S_ADDR_W, S_DATA_W> { addr: SAddr:0, data: SData:0x06, write_mask: (), read_mask: (), we: true, re: false },
    ];
    trace_fmt!("Reqs: {}", reqs);
    trace_fmt!("Orders: {}", orders);

    assert_eq(reqs, exp_reqs);
}

fn pack_response<P_DATA_W: u32, S_DATA_W: u32>(
    resps: ram::RWRamResp<S_DATA_W>, poss: u32, data: uN[P_DATA_W]) -> uN[P_DATA_W]{
    bit_slice_update(data, poss as u32 * u32:8, resps.data)
}

fn to_parallel_resp<P_DATA_W: u32, S_DATA_W: u32>(
    resps: ram::RWRamResp<S_DATA_W>[8], poss: u32[8]) -> HistoryBufferResp<P_DATA_W> {
    let data = unroll_for! (i, data) : (u32, uN[P_DATA_W]) in u32:0..u32:8 {
        pack_response<P_DATA_W, S_DATA_W>(resps[i], poss[i], data)
    }(uN[P_DATA_W]:0);
    HistoryBufferResp { data }
}

#[test]
fn to_parallel_resp_test() {
    const P_DATA_W = u32:64;
    const P_NUM_PARTITIONS = P_DATA_W / S_DATA_W;
    const P_MASK_W = P_NUM_PARTITIONS;
    const S_ADDR_W = S_DATA_W - std::clog2(P_NUM_PARTITIONS);

    type PAddr = uN[P_ADDR_W];
    type PData = uN[P_DATA_W];
    type PMask = uN[P_MASK_W];
    type SAddr = uN[S_ADDR_W];
    type SData = uN[S_DATA_W];

    let resps_1 = ram::RWRamResp<S_DATA_W>[8]:[
        ram::RWRamResp<S_DATA_W> { data: SData:0x11 },
        ram::RWRamResp<S_DATA_W> { data: SData:0x22 },
        ram::RWRamResp<S_DATA_W> { data: SData:0x33 },
        ram::RWRamResp<S_DATA_W> { data: SData:0x44 },
        ram::RWRamResp<S_DATA_W> { data: SData:0x55 },
        ram::RWRamResp<S_DATA_W> { data: SData:0x66 },
        ram::RWRamResp<S_DATA_W> { data: SData:0x77 },
        ram::RWRamResp<S_DATA_W> { data: SData:0x88 },
    ];
    let orders_1 = u32[8]:[ u32:0, u32:1, u32:2, u32:3, u32:4, u32:5, u32:6, u32:7 ];
    let resp = to_parallel_resp<P_DATA_W, S_DATA_W>(resps_1, orders_1);
    let exp_resp1 = HistoryBufferResp { data: PData:0x8877_6655_4433_2211 };
    assert_eq(resp, exp_resp1);

    let resps_2 = ram::RWRamResp<S_DATA_W>[8]:[
        ram::RWRamResp<S_DATA_W> { data: SData:0xAA },
        ram::RWRamResp<S_DATA_W> { data: SData:0x88 },
        ram::RWRamResp<S_DATA_W> { data: SData:0x99 },
        ram::RWRamResp<S_DATA_W> { data: SData:0xFF },
        ram::RWRamResp<S_DATA_W> { data: SData:0xEE },
        ram::RWRamResp<S_DATA_W> { data: SData:0xDD },
        ram::RWRamResp<S_DATA_W> { data: SData:0xCC },
        ram::RWRamResp<S_DATA_W> { data: SData:0xBB },
    ];
    let orders_2 = u32[8]:[ u32:7, u32:0, u32:1, u32:2, u32:3, u32:4, u32:5, u32:6 ];
    let resp = to_parallel_resp<P_DATA_W, S_DATA_W>(resps_2, orders_2);
        let exp_resp2 = HistoryBufferResp {
        data: PData:0xAA_BB_CC_DD_EE_FF_99_88,
    };

    trace_fmt!("resp: {:#x}", resp);
    trace_fmt!("exp_resp2: {:#x}", exp_resp2);

    assert_eq(resp, exp_resp2);
}

proc HistoryBuffer1rw<
    RAM_DATA_W: u32,
    RAM_ADDR_W: u32,
    SINGLE_RAM_DATA_W: u32,
    RAM_NUM_PARTITIONS: u32 = {RAM_DATA_W / SINGLE_RAM_DATA_W},
    SINGLE_RAM_ADDR_W: u32 = {RAM_ADDR_W - std::clog2(RAM_NUM_PARTITIONS)},
> {
    type InputReq = HistoryBufferReq<RAM_ADDR_W, RAM_DATA_W, RAM_NUM_PARTITIONS>;
    type InputResp = HistoryBufferResp<RAM_DATA_W>;
    type InputWrComp = HistoryBufferWrComp;

    type OutputReq = ram::RWRamReq<SINGLE_RAM_ADDR_W, SINGLE_RAM_DATA_W>;
    type OutputResp = ram::RWRamResp<SINGLE_RAM_DATA_W>;
    type OutputWrComp = ();

    req_r: chan<InputReq> in;
    resp_s: chan<InputResp> out;
    wr_comp_s: chan<InputWrComp> out;

    req_n_s: chan<OutputReq>[RAM_NUM_PARTITIONS] out;
    resp_n_r: chan<OutputResp>[RAM_NUM_PARTITIONS] in;
    wr_comp_n_r: chan<OutputWrComp>[RAM_NUM_PARTITIONS] in;

    init {}

    config(
        req_r: chan<InputReq> in,
        resp_s: chan<InputResp> out,
        wr_comp_s: chan<InputWrComp> out,

        req_n_s: chan<OutputReq>[RAM_NUM_PARTITIONS] out,
        resp_n_r: chan<OutputResp>[RAM_NUM_PARTITIONS] in,
        wr_comp_n_r: chan<OutputWrComp>[RAM_NUM_PARTITIONS] in
    ) {
        (
            req_r, resp_s, wr_comp_s,
            req_n_s, resp_n_r, wr_comp_n_r,
        )
    }

    next(state: ()) {
        const ZERO_RESP = zero!<OutputResp>();
        const ZERO_WR_COMP = zero!<OutputWrComp>();

        let (req_tok, req) = recv(join(), req_r);
        trace_fmt!("Received request to parallel ram: {:#x}" , req);
        let (reqs, orders, do_rd, do_wr) = to_single_req<RAM_ADDR_W, RAM_DATA_W,
            RAM_NUM_PARTITIONS, SINGLE_RAM_ADDR_W, SINGLE_RAM_DATA_W>(req);

        let (tok, resps_tok, comps_tok, resp_data) =
            unroll_for! (i, (tok, all_resps, all_comps, data)):
            (u32, (token, token, token, uN[RAM_DATA_W])) in u32:0..RAM_NUM_PARTITIONS {

                if (do_rd[i] | do_wr[i]) { trace_fmt!("Sending request to ram {}", i); };
                let tok_req = send_if(tok, req_n_s[i], (do_rd[i] | do_wr[i]), reqs[i]);

                if do_rd[i] { trace_fmt!("Waiting for read response from ram {}", i); };
                let (tok_resp, resp) = recv_if(tok_req, resp_n_r[i], do_rd[i], ZERO_RESP);

                let output_resp = pack_response<RAM_DATA_W, SINGLE_RAM_DATA_W>(resp, orders[i], data);

                if do_wr[i] { trace_fmt!("Waiting for write completion from ram {}", i); };
                let (tok_comp, _) = recv_if(tok_req, wr_comp_n_r[i], do_wr[i], ZERO_WR_COMP);

                (tok, join(tok_resp, all_resps), join(tok_comp, all_comps), output_resp)
        }((req_tok, req_tok, req_tok, uN[RAM_DATA_W]:0));

        let resp = HistoryBufferResp { data: resp_data };

        let (do_resp, do_wr_comp) =
            unroll_for! (i, (do_read, do_write)) : (u32, (bool, bool)) in u32:0..RAM_NUM_PARTITIONS {
                let new_do_rd = do_read | do_rd[i];
                let new_do_wr = do_write | do_wr[i];
                (new_do_rd, new_do_wr)
        }((false, false));

        if do_resp { trace_fmt!("Sending response {:#x}", resp);};
        let resp_tok = send_if(resps_tok, resp_s, do_resp, resp);

        if do_wr_comp { trace_fmt!("Sending completion"); };
        let resp_tok = send_if(comps_tok, wr_comp_s, do_wr_comp, ());
    }
}

const TEST_RAM_DATA_W = u32:64;
const TEST_RAM_SIZE = u32:1024;
const TEST_NUM_PARTITIONS = TEST_RAM_DATA_W / S_DATA_W;
const TEST_SINGLE_RAM_ADDR_W = P_ADDR_W - std::clog2(TEST_NUM_PARTITIONS);
const TEST_SINGLE_RAM_SIZE = TEST_RAM_SIZE / S_DATA_W;
const TEST_RAM_MASK = TEST_NUM_PARTITIONS;

#[test_proc]
proc HistoryBuffer1rwTest {
    type InputReq = HistoryBufferReq<P_ADDR_W, TEST_RAM_DATA_W, TEST_NUM_PARTITIONS>;
    type InputResp = HistoryBufferResp<TEST_RAM_DATA_W>;
    type InputWrComp = HistoryBufferWrComp;

    type OutputReq = ram::RWRamReq<TEST_SINGLE_RAM_ADDR_W, S_DATA_W>;
    type OutputResp = ram::RWRamResp<S_DATA_W>;
    type OutputWrComp = ();

    type IAddr = uN[P_ADDR_W];
    type IData = uN[TEST_RAM_DATA_W];
    type IMask = uN[TEST_RAM_MASK];

    terminator: chan<bool> out;
    req_s: chan<InputReq> out;
    resp_r: chan<InputResp> in;
    wr_comp_r: chan<InputWrComp> in;

    init {}

    config(terminator: chan<bool> out) {
        let (req_s, req_r) = chan<InputReq>("req");
        let (resp_s, resp_r) = chan<InputResp>("resp");
        let (wr_comp_s, wr_comp_r) = chan<InputWrComp>("wr_comp");

        let (req_n_s, req_n_r) = chan<OutputReq>[TEST_NUM_PARTITIONS]("req_n");
        let (resp_n_s, resp_n_r) = chan<OutputResp>[TEST_NUM_PARTITIONS]("resp_n");
        let (wr_comp_n_s, wr_comp_n_r) = chan<OutputWrComp>[TEST_NUM_PARTITIONS]("wr_comp_n");

        unroll_for! (i, _) : (u32, ()) in u32:0..TEST_NUM_PARTITIONS {
            spawn ram::SinglePortRamModel<S_DATA_W,
                TEST_SINGLE_RAM_SIZE, S_DATA_W, TEST_SINGLE_RAM_ADDR_W>
                (req_n_r[i], resp_n_s[i], wr_comp_n_s[i]);
        }(());

        spawn HistoryBuffer1rw<
            TEST_RAM_DATA_W, P_ADDR_W, S_DATA_W
        >(
            req_r, resp_s, wr_comp_s,
            req_n_s, resp_n_r, wr_comp_n_r,
        );

        (terminator, req_s, resp_r, wr_comp_r)
    }

    next(state: ()) {
        let tok = send(join(), req_s, InputReq {
            addr: IAddr:0x4,
            data: IData:0x88_77_66_55_44_33_22_11,
            write_mask: IMask:0b1001_0101,
            read_mask: IMask:0b0000_0000,
        });
        let (tok, _) = recv(tok, wr_comp_r);

        let tok = send(join(), req_s, InputReq {
            addr: IAddr:0x8,
            data: IData:0x0,
            read_mask: IMask:0b1111_1111,
            write_mask: IMask:0b0000_0000,
        });
        let (tok, resp) = recv(tok, resp_r);
        trace_fmt!("Received response: {:#x}", resp);
        assert_eq(resp, InputResp {
            data: IData:0x00_00_00_00_88_00_00_55,
        });

        let tok = send(join(), req_s, InputReq {
            addr: IAddr:0x9,
            data: IData:0xAA_BB,
            write_mask: IMask:0b0000_0011,
            read_mask: IMask:0b0000_0000,
        });
        let (tok, _) = recv(tok, wr_comp_r);

        let tok = send(join(), req_s, InputReq {
            addr: IAddr:0x8,
            data: IData:0x0,
            read_mask: IMask:0b1111_1111,
            write_mask: IMask:0b0000_0000,
        });
        let (tok, resp) = recv(tok, resp_r);
        trace_fmt!("Received response: {:#x}", resp);
        assert_eq(resp, InputResp {
            data: IData:0x00_00_00_00_88_AA_BB_55,
        });

        send(tok, terminator, true);
    }
}

pub proc HistoryBuffer<
    RAM_DATA_W: u32,
    RAM_ADDR_W: u32,
    SINGLE_RAM_DATA_W: u32,
    RAM_NUM_PARTITIONS: u32 = {RAM_DATA_W / SINGLE_RAM_DATA_W},
    SINGLE_RAM_ADDR_W: u32 = {RAM_ADDR_W - std::clog2(RAM_NUM_PARTITIONS)},
> {
    type InputReq = HistoryBufferReq<RAM_ADDR_W, RAM_DATA_W, RAM_NUM_PARTITIONS>;
    type InputResp = HistoryBufferResp<RAM_DATA_W>;
    type InputWrComp = ();

    type OutputReq = ram::RWRamReq<SINGLE_RAM_ADDR_W, SINGLE_RAM_DATA_W>;
    type OutputResp = ram::RWRamResp<SINGLE_RAM_DATA_W>;
    type OutputWrComp = ();

    init { }

    config(
        req0_r: chan<InputReq> in,
        resp0_s: chan<InputResp> out,
        wr_comp0_s: chan<InputWrComp> out,

        req1_r: chan<InputReq> in,
        resp1_s: chan<InputResp> out,
        wr_comp1_s: chan<InputWrComp> out,

        req0n_s: chan<OutputReq>[RAM_NUM_PARTITIONS] out,
        resp0n_r: chan<OutputResp>[RAM_NUM_PARTITIONS] in,
        wr_comp0n_r: chan<OutputWrComp>[RAM_NUM_PARTITIONS] in,

        req1n_s: chan<OutputReq>[RAM_NUM_PARTITIONS] out,
        resp1n_r: chan<OutputResp>[RAM_NUM_PARTITIONS] in,
        wr_comp1n_r: chan<OutputWrComp>[RAM_NUM_PARTITIONS] in
    ) {
        spawn HistoryBuffer1rw<RAM_DATA_W, RAM_ADDR_W, SINGLE_RAM_DATA_W> (
            req0_r, resp0_s, wr_comp0_s,
            req0n_s, resp0n_r, wr_comp0n_r,
        );

        spawn HistoryBuffer1rw<RAM_DATA_W, RAM_ADDR_W, SINGLE_RAM_DATA_W> (
            req1_r, resp1_s, wr_comp1_s,
            req1n_s, resp1n_r, wr_comp1n_r,
        );

        ()
    }

    next(state: ()) { }
}
