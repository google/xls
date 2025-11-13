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
    P_ADDR_W: u32, P_DATA_W: u32, P_NUM_PARTITIONS: u32,
    S_ADDR_W: u32, S_DATA_W: u32
>(preq: HistoryBufferReq<P_ADDR_W, P_DATA_W, P_NUM_PARTITIONS>) -> (ram::RWRamReq<S_ADDR_W, S_DATA_W>[8], u3[8], bool[8], bool[8]) {

    let base = preq.addr / uN[P_ADDR_W]:8;
    let offset = preq.addr % uN[P_ADDR_W]:8;
    type RWRamReq = ram::RWRamReq<S_ADDR_W, S_DATA_W>;
    const ZERO_REQS = RWRamReq[8]:[zero!<RWRamReq>(), ...];
    const ZERO_ORDER = u3[8]:[u3:0, ...];
    const ZERO_DO_SEND = bool[8]:[false, ...];
    const ZERO_DO_COMP = bool[8]:[false, ...];

    unroll_for! (i, (reqs, poss, do_rd, do_wr)) : (u32, (RWRamReq[8], u3[8], bool[8], bool[8])) in u32:0..u32:8 {
        let bank = (i - offset as u32) % u32:8;
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
        let poss = update(poss, i, bank as u3);
        let do_rd = update(do_rd, i, re);
        let do_wr = update(do_wr, i, we);
        (reqs, poss, do_rd, do_wr)
    }((ZERO_REQS, ZERO_ORDER, ZERO_DO_SEND, ZERO_DO_COMP))
}

#[test]
fn to_single_req_test() {
    const P_ADDR_W = u32:32;
    const P_DATA_W = u32:64;
    const P_NUM_PARTITIONS = u32:8;
    const P_MASK_W = u32:8;
    const S_ADDR_W = u32:29;
    const S_DATA_W = u32:8;

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

fn to_parallel_resp<P_DATA_W: u32, S_DATA_W: u32>(
    resps: ram::RWRamResp<S_DATA_W>[8], poss: u3[8]) -> HistoryBufferResp<P_DATA_W> {
    let data = unroll_for! (i, data) : (u32, uN[P_DATA_W]) in u32:0..u32:8 {
        bit_slice_update(data, poss[i] as u32 * u32:8, resps[i].data)
    }(uN[P_DATA_W]:0);
    HistoryBufferResp { data }
}

#[test]
fn to_parallel_resp_test() {
    const P_ADDR_W = u32:32;
    const P_DATA_W = u32:64;
    const P_MASK_W = u32:8;
    const S_ADDR_W = u32:29;
    const S_DATA_W = u32:8;

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
    let orders_1 = u3[8]:[ u3:0, u3:1, u3:2, u3:3, u3:4, u3:5, u3:6, u3:7 ];
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
    let orders_2 = u3[8]:[ u3:7, u3:0, u3:1, u3:2, u3:3, u3:4, u3:5, u3:6 ];
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
    RAM_SIZE: u64,
    SINGLE_RAM_DATA_W: u32 = {RAM_DATA_W / u32:8},
    SINGLE_RAM_ADDR_W: u32 = {RAM_ADDR_W - u32:3},
    SINGLE_RAM_SIZE_W: u32 = {RAM_SIZE as u32 / u32:8},
    RAM_NUM_PARTITIONS: u32 = {RAM_DATA_W / u32:8},
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

    req_n_s: chan<OutputReq>[8] out;
    resp_n_r: chan<OutputResp>[8] in;
    wr_comp_n_r: chan<OutputWrComp>[8] in;

    init {}

    config(
        req_r: chan<InputReq> in,
        resp_s: chan<InputResp> out,
        wr_comp_s: chan<InputWrComp> out,

        req_n_s: chan<OutputReq>[8] out,
        resp_n_r: chan<OutputResp>[8] in,
        wr_comp_n_r: chan<OutputWrComp>[8] in
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
        let (reqs, orders, do_rd, do_wr) = to_single_req<
            RAM_ADDR_W, RAM_DATA_W, RAM_NUM_PARTITIONS, SINGLE_RAM_ADDR_W, SINGLE_RAM_DATA_W
        >(req);

        if (do_rd[0] | do_wr[0]) { trace_fmt!("Sending request to ram 0"); };
        let req_tok_0 = send_if(req_tok, req_n_s[0], (do_rd[0] | do_wr[0]), reqs[0]);
        if (do_rd[1] | do_wr[1]) { trace_fmt!("Sending request to ram 1"); };
        let req_tok_1 = send_if(req_tok, req_n_s[1], (do_rd[1] | do_wr[1]), reqs[1]);
        if (do_rd[2] | do_wr[2]) { trace_fmt!("Sending request to ram 2"); };
        let req_tok_2 = send_if(req_tok, req_n_s[2], (do_rd[2] | do_wr[2]), reqs[2]);
        if (do_rd[3] | do_wr[3]) { trace_fmt!("Sending request to ram 3"); };
        let req_tok_3 = send_if(req_tok, req_n_s[3], (do_rd[3] | do_wr[3]), reqs[3]);
        if (do_rd[4] | do_wr[4]) { trace_fmt!("Sending request to ram 4"); };
        let req_tok_4 = send_if(req_tok, req_n_s[4], (do_rd[4] | do_wr[4]), reqs[4]);
        if (do_rd[5] | do_wr[5]) { trace_fmt!("Sending request to ram 5"); };
        let req_tok_5 = send_if(req_tok, req_n_s[5], (do_rd[5] | do_wr[5]), reqs[5]);
        if (do_rd[6] | do_wr[6]) { trace_fmt!("Sending request to ram 6"); };
        let req_tok_6 = send_if(req_tok, req_n_s[6], (do_rd[6] | do_wr[6]), reqs[6]);
        if (do_rd[7] | do_wr[7]) { trace_fmt!("Sending request to ram 7"); };
        let req_tok_7 = send_if(req_tok, req_n_s[7], (do_rd[7] | do_wr[7]), reqs[7]);

        if do_rd[0]  { trace_fmt!("Waiting for read response from ram 0"); };
        let (resp_tok_0, resp_0) = recv_if(req_tok_0, resp_n_r[0], do_rd[0], ZERO_RESP);
        if do_rd[1]  { trace_fmt!("Waiting for read response from ram 1"); };
        let (resp_tok_1, resp_1) = recv_if(req_tok_1, resp_n_r[1], do_rd[1], ZERO_RESP);
        if do_rd[2]  { trace_fmt!("Waiting for read response from ram 2"); };
        let (resp_tok_2, resp_2) = recv_if(req_tok_2, resp_n_r[2], do_rd[2], ZERO_RESP);
        if do_rd[3]  { trace_fmt!("Waiting for read response from ram 3"); };
        let (resp_tok_3, resp_3) = recv_if(req_tok_3, resp_n_r[3], do_rd[3], ZERO_RESP);
        if do_rd[4]  { trace_fmt!("Waiting for read response from ram 4"); };
        let (resp_tok_4, resp_4) = recv_if(req_tok_4, resp_n_r[4], do_rd[4], ZERO_RESP);
        if do_rd[5]  { trace_fmt!("Waiting for read response from ram 5"); };
        let (resp_tok_5, resp_5) = recv_if(req_tok_5, resp_n_r[5], do_rd[5], ZERO_RESP);
        if do_rd[6]  { trace_fmt!("Waiting for read response from ram 6"); };
        let (resp_tok_6, resp_6) = recv_if(req_tok_6, resp_n_r[6], do_rd[6], ZERO_RESP);
        if do_rd[7]  { trace_fmt!("Waiting for read response from ram 7"); };
        let (resp_tok_7, resp_7) = recv_if(req_tok_7, resp_n_r[7], do_rd[7], ZERO_RESP);

        let resps_tok = join(
            resp_tok_0, resp_tok_1, resp_tok_2, resp_tok_3,
            resp_tok_4, resp_tok_5, resp_tok_6, resp_tok_7,
        );
        type OutputResp = ram::RWRamResp<SINGLE_RAM_DATA_W>;
        let resp = to_parallel_resp<RAM_DATA_W, SINGLE_RAM_DATA_W>(
            OutputResp[u32:8]:[ resp_0, resp_1, resp_2, resp_3, resp_4, resp_5, resp_6, resp_7 ],
            orders
        );
        let do_resp = do_rd[0] | do_rd[1] | do_rd[2] | do_rd[3]
                    | do_rd[4] | do_rd[5] | do_rd[6] | do_rd[7];
        if do_resp { trace_fmt!("Sending response {:#x}", resp);};
        let resp_tok = send_if(resps_tok, resp_s, do_resp, resp);

        if do_wr[0]  { trace_fmt!("Waiting for write completion from ram 0"); };
        let (comp_tok_0, _) = recv_if(req_tok_0, wr_comp_n_r[0], do_wr[0], ZERO_WR_COMP);
        if do_wr[1]  { trace_fmt!("Waiting for write completion from ram 1"); };
        let (comp_tok_1, _) = recv_if(req_tok_1, wr_comp_n_r[1], do_wr[1], ZERO_WR_COMP);
        if do_wr[2]  { trace_fmt!("Waiting for write completion from ram 2"); };
        let (comp_tok_2, _) = recv_if(req_tok_2, wr_comp_n_r[2], do_wr[2], ZERO_WR_COMP);
        if do_wr[3]  { trace_fmt!("Waiting for write completion from ram 3"); };
        let (comp_tok_3, _) = recv_if(req_tok_3, wr_comp_n_r[3], do_wr[3], ZERO_WR_COMP);
        if do_wr[4]  { trace_fmt!("Waiting for write completion from ram 4"); };
        let (comp_tok_4, _) = recv_if(req_tok_4, wr_comp_n_r[4], do_wr[4], ZERO_WR_COMP);
        if do_wr[5]  { trace_fmt!("Waiting for write completion from ram 5"); };
        let (comp_tok_5, _) = recv_if(req_tok_5, wr_comp_n_r[5], do_wr[5], ZERO_WR_COMP);
        if do_wr[6]  { trace_fmt!("Waiting for write completion from ram 6"); };
        let (comp_tok_6, _) = recv_if(req_tok_6, wr_comp_n_r[6], do_wr[6], ZERO_WR_COMP);
        if do_wr[7]  { trace_fmt!("Waiting for write completion from ram 7"); };
        let (comp_tok_7, _) = recv_if(req_tok_7, wr_comp_n_r[7], do_wr[7], ZERO_WR_COMP);

        let comps_tok = join(
            comp_tok_0, comp_tok_1, comp_tok_2, comp_tok_3,
            comp_tok_4, comp_tok_5, comp_tok_6, comp_tok_7,
        );
        let do_wr_comp = do_wr[0] | do_wr[1] | do_wr[2] | do_wr[3]
                       | do_wr[4] | do_wr[5] | do_wr[6] | do_wr[7];
        if do_wr_comp { trace_fmt!("Sending completion"); };
        let resp_tok = send_if(resps_tok, wr_comp_s, do_wr_comp, ());
    }
}

const TEST_RAM_DATA_W = u32:64;
const TEST_RAM_ADDR_W = u32:32;
const TEST_RAM_SIZE = u64:1024;
const TEST_RAM_MASK = TEST_RAM_DATA_W / u32:8;
const TEST_RAM_NUM_PARTITIONS = TEST_RAM_DATA_W / u32:8;

const TEST_SINGLE_RAM_DATA_W = TEST_RAM_DATA_W / u32:8;
const TEST_SINGLE_RAM_ADDR_W = TEST_RAM_ADDR_W - u32:3;
const TEST_SINGLE_RAM_SIZE = TEST_RAM_SIZE as u32 / u32:8;

#[test_proc]
proc HistoryBuffer1rwTest {
    type InputReq = HistoryBufferReq<TEST_RAM_ADDR_W, TEST_RAM_DATA_W, TEST_RAM_NUM_PARTITIONS>;
    type InputResp = HistoryBufferResp<TEST_RAM_DATA_W>;
    type InputWrComp = HistoryBufferWrComp;

    type OutputReq = ram::RWRamReq<TEST_SINGLE_RAM_ADDR_W, TEST_SINGLE_RAM_DATA_W>;
    type OutputResp = ram::RWRamResp<TEST_SINGLE_RAM_DATA_W>;
    type OutputWrComp = ();

    type IAddr = uN[TEST_RAM_ADDR_W];
    type IData = uN[TEST_RAM_DATA_W];
    type IMask = uN[TEST_RAM_MASK];

    type SAddr = uN[TEST_SINGLE_RAM_ADDR_W];
    type SData = uN[TEST_SINGLE_RAM_DATA_W];

    terminator: chan<bool> out;
    req_s: chan<InputReq> out;
    resp_r: chan<InputResp> in;
    wr_comp_r: chan<InputWrComp> in;

    init {}

    config(terminator: chan<bool> out) {
        let (req_s, req_r) = chan<InputReq>("req");
        let (resp_s, resp_r) = chan<InputResp>("resp");
        let (wr_comp_s, wr_comp_r) = chan<InputWrComp>("wr_comp");

        let (req_n_s, req_n_r) = chan<OutputReq>[8]("req_n");
        let (resp_n_s, resp_n_r) = chan<OutputResp>[8]("resp_n");
        let (wr_comp_n_s, wr_comp_n_r) = chan<OutputWrComp>[8]("wr_comp_n");

        spawn ram::SinglePortRamModel<
            TEST_SINGLE_RAM_DATA_W, TEST_SINGLE_RAM_SIZE,
            TEST_SINGLE_RAM_DATA_W, TEST_SINGLE_RAM_ADDR_W
        > (req_n_r[0], resp_n_s[0], wr_comp_n_s[0]);
        spawn ram::SinglePortRamModel<
            TEST_SINGLE_RAM_DATA_W, TEST_SINGLE_RAM_SIZE,
            TEST_SINGLE_RAM_DATA_W, TEST_SINGLE_RAM_ADDR_W
        > (req_n_r[1], resp_n_s[1], wr_comp_n_s[1]);
        spawn ram::SinglePortRamModel<
            TEST_SINGLE_RAM_DATA_W, TEST_SINGLE_RAM_SIZE,
            TEST_SINGLE_RAM_DATA_W, TEST_SINGLE_RAM_ADDR_W>
            (req_n_r[2], resp_n_s[2], wr_comp_n_s[2]);
        spawn ram::SinglePortRamModel<
            TEST_SINGLE_RAM_DATA_W, TEST_SINGLE_RAM_SIZE,
            TEST_SINGLE_RAM_DATA_W, TEST_SINGLE_RAM_ADDR_W
        > (req_n_r[3], resp_n_s[3], wr_comp_n_s[3]);
        spawn ram::SinglePortRamModel<
            TEST_SINGLE_RAM_DATA_W, TEST_SINGLE_RAM_SIZE,
            TEST_SINGLE_RAM_DATA_W, TEST_SINGLE_RAM_ADDR_W
        > (req_n_r[4], resp_n_s[4], wr_comp_n_s[4]);
        spawn ram::SinglePortRamModel<
            TEST_SINGLE_RAM_DATA_W, TEST_SINGLE_RAM_SIZE,
            TEST_SINGLE_RAM_DATA_W, TEST_SINGLE_RAM_ADDR_W
        > (req_n_r[5], resp_n_s[5], wr_comp_n_s[5]);
        spawn ram::SinglePortRamModel<
            TEST_SINGLE_RAM_DATA_W, TEST_SINGLE_RAM_SIZE,
            TEST_SINGLE_RAM_DATA_W, TEST_SINGLE_RAM_ADDR_W
        > (req_n_r[6], resp_n_s[6], wr_comp_n_s[6]);
        spawn ram::SinglePortRamModel<
            TEST_SINGLE_RAM_DATA_W, TEST_SINGLE_RAM_SIZE,
            TEST_SINGLE_RAM_DATA_W, TEST_SINGLE_RAM_ADDR_W
        > (req_n_r[7], resp_n_s[7], wr_comp_n_s[7]);

        spawn HistoryBuffer1rw<
            TEST_RAM_DATA_W, TEST_RAM_ADDR_W, TEST_RAM_SIZE
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
    RAM_SIZE: u64,
    SINGLE_RAM_DATA_W: u32 = {RAM_DATA_W / u32:8},
    SINGLE_RAM_ADDR_W: u32 = {RAM_ADDR_W},
    RAM_NUM_PARTITIONS: u32 = {RAM_DATA_W / u32:8},
    SINGLE_RAM_SIZE_W: u32 = {RAM_SIZE as u32 / RAM_NUM_PARTITIONS},
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

        req0n_s: chan<OutputReq>[8] out,
        resp0n_r: chan<OutputResp>[8] in,
        wr_comp0n_r: chan<OutputWrComp>[8] in,

        req1n_s: chan<OutputReq>[8] out,
        resp1n_r: chan<OutputResp>[8] in,
        wr_comp1n_r: chan<OutputWrComp>[8] in
    ) {
        spawn HistoryBuffer1rw<RAM_DATA_W, RAM_ADDR_W, RAM_SIZE, SINGLE_RAM_DATA_W, SINGLE_RAM_ADDR_W, SINGLE_RAM_SIZE_W, RAM_NUM_PARTITIONS> (
            req0_r, resp0_s, wr_comp0_s,
            req0n_s, resp0n_r, wr_comp0n_r,
        );

        spawn HistoryBuffer1rw<RAM_DATA_W, RAM_ADDR_W, RAM_SIZE, SINGLE_RAM_DATA_W, SINGLE_RAM_ADDR_W, SINGLE_RAM_SIZE_W, RAM_NUM_PARTITIONS> (
            req1_r, resp1_s, wr_comp1_s,
            req1n_s, resp1n_r, wr_comp1n_r,
        );

        ()
    }

    next(state: ()) { }
}
