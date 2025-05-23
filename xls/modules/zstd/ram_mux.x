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

// This file contains utilities related to ZSTD Block Header parsing.
// More information about the ZSTD Block Header can be found in:
// https://datatracker.ietf.org/doc/html/rfc8878#section-3.1.1.2


import std;
import xls.examples.ram;

struct RamMuxState { sel: u1, cnt0: u32, cnt1: u32 }

pub proc RamMux<
    ADDR_WIDTH: u32, DATA_WIDTH: u32, NUM_PARTITIONS: u32,
    INIT_SEL: u1 = {u1:0}
> {
    type ReadReq = ram::ReadReq<ADDR_WIDTH, NUM_PARTITIONS>;
    type ReadResp = ram::ReadResp<DATA_WIDTH>;
    type WriteReq = ram::WriteReq<ADDR_WIDTH, DATA_WIDTH, NUM_PARTITIONS>;
    type WriteResp = ram::WriteResp;

    sel_r: chan<u1> in;

    rd_req0_r: chan<ReadReq> in;
    rd_resp0_s: chan<ReadResp> out;
    wr_req0_r: chan<WriteReq> in;
    wr_resp0_s: chan<WriteResp> out;
    rd_req1_r: chan<ReadReq> in;
    rd_resp1_s: chan<ReadResp> out;
    wr_req1_r: chan<WriteReq> in;
    wr_resp1_s: chan<WriteResp> out;
    rd_req_s: chan<ReadReq> out;
    rd_resp_r: chan<ReadResp> in;
    wr_req_s: chan<WriteReq> out;
    wr_resp_r: chan<WriteResp> in;

    config(
        sel_r: chan<u1> in,
        rd_req0_r: chan<ReadReq> in,
        rd_resp0_s: chan<ReadResp> out,
        wr_req0_r: chan<WriteReq> in,
        wr_resp0_s: chan<WriteResp> out,
        rd_req1_r: chan<ReadReq> in,
        rd_resp1_s: chan<ReadResp> out,
        wr_req1_r: chan<WriteReq> in,
        wr_resp1_s: chan<WriteResp> out,
        rd_req_s: chan<ReadReq> out,
        rd_resp_r: chan<ReadResp> in,
        wr_req_s: chan<WriteReq> out,
        wr_resp_r: chan<WriteResp> in
    ) {
        (
            sel_r,
            rd_req0_r, rd_resp0_s, wr_req0_r, wr_resp0_s,
            rd_req1_r, rd_resp1_s, wr_req1_r, wr_resp1_s,
            rd_req_s, rd_resp_r, wr_req_s, wr_resp_r,
        )
    }

    init {
        RamMuxState {
            sel: INIT_SEL, ..zero!<RamMuxState>()
        }
    }

    next(state: RamMuxState) {
        let tok0 = join();

        let sel = state.sel;
        let (cnt0, cnt1) = (state.cnt0, state.cnt1);

        // receive requests from channel 0
        let (tok1_0, rd_req0, rd_req0_valid) =
            recv_if_non_blocking(tok0, rd_req0_r, sel == u1:0, zero!<ReadReq>());
        let cnt0 = if rd_req0_valid { cnt0 + u32:1 } else { cnt0 };

        let (tok1_1, wr_req0, wr_req0_valid) =
            recv_if_non_blocking(tok0, wr_req0_r, sel == u1:0, zero!<WriteReq>());
        let cnt0 = if wr_req0_valid { cnt0 + u32:1 } else { cnt0 };

        // receive requests from channel 1
        let (tok1_2, rd_req1, rd_req1_valid) =
            recv_if_non_blocking(tok0, rd_req1_r, sel == u1:1, zero!<ReadReq>());
        let cnt1 = if rd_req1_valid { cnt1 + u32:1 } else { cnt1 };

        let (tok1_3, wr_req1, wr_req1_valid) =
            recv_if_non_blocking(tok0, wr_req1_r, sel == u1:1, zero!<WriteReq>());
        let cnt1 = if wr_req1_valid { cnt1 + u32:1 } else { cnt1 };

        // receive responses from output channel
        let (tok1_4, rd_resp, rd_resp_valid) =
            recv_non_blocking(tok0, rd_resp_r, zero!<ReadResp>());
        let (tok1_5, wr_resp, wr_resp_valid) =
            recv_non_blocking(tok0, wr_resp_r, zero!<WriteResp>());

        let tok1 = join(tok1_0, tok1_1, tok1_2, tok1_3, tok1_4, tok1_5);

        // prepare output values
        let (rd_req, rd_req_valid, wr_req, wr_req_valid) = if sel == u1:0 {
            (rd_req0, rd_req0_valid, wr_req0, wr_req0_valid)
        } else {
            (rd_req1, rd_req1_valid, wr_req1, wr_req1_valid)
        };

        // send requests to output channel
        let tok2_0 = send_if(tok1, rd_req_s, rd_req_valid, rd_req);
        let tok2_1 = send_if(tok1, wr_req_s, wr_req_valid, wr_req);

        // send responses to channel 0
        let rd_resp0_cond = (sel == u1:0 && rd_resp_valid);
        let tok2_2 = send_if(tok1, rd_resp0_s, rd_resp0_cond, rd_resp);
        let cnt0 = if rd_resp0_cond { cnt0 - u32:1 } else { cnt0 };

        let wr_resp0_cond = (sel == u1:0 && wr_resp_valid);
        let tok2_3 = send_if(tok1, wr_resp0_s, wr_resp0_cond, wr_resp);
        let cnt0 = if wr_resp0_cond { cnt0 - u32:1 } else { cnt0 };

        // send responses to channel 1
        let rd_resp1_cond = (sel == u1:1 && rd_resp_valid);
        let tok2_4 = send_if(tok1, rd_resp1_s, rd_resp1_cond, rd_resp);
        let cnt1 = if rd_resp1_cond { cnt1 - u32:1 } else { cnt1 };

        let wr_resp1_cond = (sel == u1:1 && wr_resp_valid);
        let tok2_5 = send_if(tok1, wr_resp1_s, wr_resp1_cond, wr_resp);
        let cnt1 = if wr_resp1_cond { cnt1 - u32:1 } else { cnt1 };

        // handle select
        let (tok2_6, sel, sel_valid) =
            recv_if_non_blocking(tok1, sel_r, cnt0 == u32:0 && cnt1 == u32:0, state.sel);

        RamMuxState { sel, cnt0, cnt1 }
    }
}

const MUX_TEST_SIZE = u32:32;
const MUX_TEST_DATA_WIDTH = u32:8;
const MUX_TEST_ADDR_WIDTH = std::clog2(MUX_TEST_SIZE);
const MUX_TEST_WORD_PARTITION_SIZE = u32:1;
const MUX_TEST_NUM_PARTITIONS = ram::num_partitions(MUX_TEST_WORD_PARTITION_SIZE, MUX_TEST_DATA_WIDTH);

type MuxTestAddr = uN[MUX_TEST_ADDR_WIDTH];
type MuxTestData = uN[MUX_TEST_DATA_WIDTH];

fn MuxTestWriteWordReq (addr: MuxTestAddr, data: MuxTestData) ->
    ram::WriteReq<MUX_TEST_ADDR_WIDTH, MUX_TEST_DATA_WIDTH, MUX_TEST_NUM_PARTITIONS> {
    ram::WriteWordReq<MUX_TEST_NUM_PARTITIONS>(addr, data)
}

fn MuxTestReadWordReq(addr: MuxTestAddr) ->
    ram::ReadReq<MUX_TEST_ADDR_WIDTH, MUX_TEST_NUM_PARTITIONS> {
    ram::ReadWordReq<MUX_TEST_NUM_PARTITIONS>(addr)
}

#[test_proc]
proc RamMuxTest {
    terminator: chan<bool> out;
    sel_s: chan<u1> out;

    type ReadReq = ram::ReadReq<MUX_TEST_ADDR_WIDTH, MUX_TEST_NUM_PARTITIONS>;
    type ReadResp = ram::ReadResp<MUX_TEST_DATA_WIDTH>;
    type WriteReq = ram::WriteReq<MUX_TEST_ADDR_WIDTH, MUX_TEST_DATA_WIDTH, MUX_TEST_NUM_PARTITIONS>;
    type WriteResp = ram::WriteResp;


    rd_req0_s: chan<ReadReq> out;
    rd_resp0_r: chan<ReadResp> in;
    wr_req0_s: chan<WriteReq> out;
    wr_resp0_r: chan<WriteResp> in;
    rd_req1_s: chan<ReadReq> out;
    rd_resp1_r: chan<ReadResp> in;
    wr_req1_s: chan<WriteReq> out;
    wr_resp1_r: chan<WriteResp> in;

    config(terminator: chan<bool> out) {
        let (sel_s, sel_r) = chan<u1>("sel");

        let (rd_req0_s, rd_req0_r) = chan<ReadReq>("rd_req0");
        let (rd_resp0_s, rd_resp0_r) = chan<ReadResp>("rd_resp0");
        let (wr_req0_s, wr_req0_r) = chan<WriteReq>("wr_req0");
        let (wr_resp0_s, wr_resp0_r) = chan<WriteResp>("wr_resp0");

        let (rd_req1_s, rd_req1_r) = chan<ReadReq>("rd_req1");
        let (rd_resp1_s, rd_resp1_r) = chan<ReadResp>("rd_resp1");
        let (wr_req1_s, wr_req1_r) = chan<WriteReq>("rd_req1");
        let (wr_resp1_s, wr_resp1_r) = chan<WriteResp>("wr_resp1");

        let (rd_req_s, rd_req_r) = chan<ReadReq>("rd_req");
        let (rd_resp_s, rd_resp_r) = chan<ReadResp>("rd_resp");
        let (wr_req_s, wr_req_r) = chan<WriteReq>("wr_req");
        let (wr_resp_s, wr_resp_r) = chan<WriteResp>("wr_resp");

        spawn RamMux<MUX_TEST_ADDR_WIDTH, MUX_TEST_DATA_WIDTH, MUX_TEST_NUM_PARTITIONS>(
            sel_r, rd_req0_r, rd_resp0_s, wr_req0_r, wr_resp0_s, rd_req1_r, rd_resp1_s, wr_req1_r,
            wr_resp1_s, rd_req_s, rd_resp_r, wr_req_s, wr_resp_r);

        spawn ram::RamModel<MUX_TEST_DATA_WIDTH, MUX_TEST_SIZE, MUX_TEST_WORD_PARTITION_SIZE>(
            rd_req_r, rd_resp_s, wr_req_r, wr_resp_s);
        (
            terminator, sel_s, rd_req0_s, rd_resp0_r, wr_req0_s, wr_resp0_r, rd_req1_s, rd_resp1_r,
            wr_req1_s, wr_resp1_r,
        )
    }

    init {  }

    next(state: ()) {
        let tok = join();
        let req = MuxTestWriteWordReq(MuxTestAddr:0, MuxTestData:0xAB);
        let tok = send(tok, wr_req0_s, req);
        let (tok, _) = recv(tok, wr_resp0_r);
        let tok = send(tok, rd_req0_s, MuxTestReadWordReq(req.addr));
        let (tok, resp) = recv(tok, rd_resp0_r);
        assert_eq(resp.data, req.data);

        let req = MuxTestWriteWordReq(MuxTestAddr:1, MuxTestData:0xCD);
        let tok = send(tok, wr_req1_s, req);
        let tok = send(tok, sel_s, u1:1);
        let (tok, _) = recv(tok, wr_resp1_r);
        let tok = send(tok, rd_req1_s, MuxTestReadWordReq(req.addr));
        let (tok, resp) = recv(tok, rd_resp1_r);
        assert_eq(resp.data, req.data);

        let tok = send(tok, terminator, true);
    }
}
