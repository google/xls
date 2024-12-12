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

// This file contains a RamDemux implementation that can be used to connect
// a single proc with two RAM instances, by using a single RAM interface and
// switching between the RAMs, when requested. The switching occurs only after
// each request has received the corresponding response.
// Additionally, a "naive" implementation is provided that does not ensure
// any synchronization when switching RAMs.

import std;
import xls.examples.ram;
import xls.modules.zstd.ram_demux;

pub proc RamDemux3<
    ADDR_WIDTH: u32,
    DATA_WIDTH: u32,
    NUM_PARTITIONS: u32,
    INIT_SEL: u2 = {u2:0},
    QUEUE_LEN: u32 = {u32:5},
    D1_INIT_SEL: u1 = {INIT_SEL == u2:1 || INIT_SEL == u2:2},
    D2_INIT_SEL: u1 = {INIT_SEL == u2:2 || INIT_SEL == u2:3},
> {
    type ReadReq = ram::ReadReq<ADDR_WIDTH, NUM_PARTITIONS>;
    type ReadResp = ram::ReadResp<DATA_WIDTH>;
    type WriteReq = ram::WriteReq<ADDR_WIDTH, DATA_WIDTH, NUM_PARTITIONS>;
    type WriteResp = ram::WriteResp;

    sel_req_r: chan<u2> in;
    sel_resp_s: chan<()> out;
    d1_sel_req_s: chan<u1> out;
    d1_sel_resp_r: chan<()> in;
    d2_sel_req_s: chan<u1> out;
    d2_sel_resp_r: chan<()> in;

    config(
        sel_req_r: chan<u2> in,
        sel_resp_s: chan<()> out,

        rd_req_r: chan<ReadReq> in,
        rd_resp_s: chan<ReadResp> out,
        wr_req_r: chan<WriteReq> in,
        wr_resp_s: chan<WriteResp> out,

        rd_req0_s: chan<ReadReq> out,
        rd_resp0_r: chan<ReadResp> in,
        wr_req0_s: chan<WriteReq> out,
        wr_resp0_r: chan<WriteResp> in,

        rd_req1_s: chan<ReadReq> out,
        rd_resp1_r: chan<ReadResp> in,
        wr_req1_s: chan<WriteReq> out,
        wr_resp1_r: chan<WriteResp> in,

        rd_req2_s: chan<ReadReq> out,
        rd_resp2_r: chan<ReadResp> in,
        wr_req2_s: chan<WriteReq> out,
        wr_resp2_r: chan<WriteResp> in

    ) {
        const CHANNEL_DEPTH = u32:1;

        let (d1_sel_req_s, d1_sel_req_r) = chan<u1, CHANNEL_DEPTH>("d1_sel_req");
        let (d1_sel_resp_s, d1_sel_resp_r) = chan<(), CHANNEL_DEPTH>("d1_sel_resp");

        let (tmp_rd_req_s, tmp_rd_req_r) = chan<ReadReq, CHANNEL_DEPTH>("tmp_rd_req");
        let (tmp_rd_resp_s, tmp_rd_resp_r) = chan<ReadResp, CHANNEL_DEPTH>("tmp_rd_resp");
        let (tmp_wr_req_s, tmp_wr_req_r) = chan<WriteReq, CHANNEL_DEPTH>("tmp_wr_req");
        let (tmp_wr_resp_s, tmp_wr_resp_r) = chan<WriteResp, CHANNEL_DEPTH>("tmp_wr_resp");

        spawn ram_demux::RamDemux<ADDR_WIDTH, DATA_WIDTH, NUM_PARTITIONS, D1_INIT_SEL, QUEUE_LEN>(
            d1_sel_req_r, d1_sel_resp_s,
            rd_req_r, rd_resp_s, wr_req_r, wr_resp_s,
            rd_req0_s, rd_resp0_r, wr_req0_s, wr_resp0_r,
            tmp_rd_req_s, tmp_rd_resp_r, tmp_wr_req_s, tmp_wr_resp_r,
        );

        let (d2_sel_req_s, d2_sel_req_r) = chan<u1, CHANNEL_DEPTH>("d2_sel_req");
        let (d2_sel_resp_s, d2_sel_resp_r) = chan<(), CHANNEL_DEPTH>("d2_sel_resp");

        spawn ram_demux::RamDemux<ADDR_WIDTH, DATA_WIDTH, NUM_PARTITIONS, D2_INIT_SEL, QUEUE_LEN>(
            d2_sel_req_r, d2_sel_resp_s,
            tmp_rd_req_r, tmp_rd_resp_s, tmp_wr_req_r, tmp_wr_resp_s,
            rd_req1_s, rd_resp1_r, wr_req1_s, wr_resp1_r,
            rd_req2_s, rd_resp2_r, wr_req2_s, wr_resp2_r
        );

        (
            sel_req_r, sel_resp_s,
            d1_sel_req_s, d1_sel_resp_r,
            d2_sel_req_s, d2_sel_resp_r,
        )
    }

    init { }

    next(state: ()) {
        let tok = join();
        let (tok, sel) = recv(tok, sel_req_r);

        let (sel1, sel2) = match sel {
            u2:0 => (u1:0, u1:0),
            u2:1 => (u1:1, u1:0),
            u2:2 => (u1:1, u1:1),
            _    => (u1:0, u1:1),
        };

        let tok1_0 = send(tok, d1_sel_req_s, sel1);
        let (tok2_0, ()) = recv(tok1_0, d1_sel_resp_r);

        let tok1_1 = send(tok, d2_sel_req_s, sel2);
        let (tok2_1, ()) = recv(tok, d2_sel_resp_r);

        let tok2 = join(tok2_0, tok2_1);
        send(tok2, sel_resp_s, ());
    }
}

const RAM_SIZE = u32:32;
const RAM_DATA_WIDTH = u32:8;
const RAM_ADDR_WIDTH = std::clog2(RAM_SIZE);
const RAM_WORD_PARTITION_SIZE = u32:1;
const RAM_NUM_PARTITIONS = ram::num_partitions(RAM_WORD_PARTITION_SIZE, RAM_DATA_WIDTH);

pub proc RamDemux3Inst {
    type ReadReq = ram::ReadReq<RAM_ADDR_WIDTH, RAM_NUM_PARTITIONS>;
    type ReadResp = ram::ReadResp<RAM_DATA_WIDTH>;
    type WriteReq = ram::WriteReq<RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>;
    type WriteResp = ram::WriteResp;

    config(
        sel_req_r: chan<u2> in,
        sel_resp_s: chan<()> out,

        rd_req_r: chan<ReadReq> in,
        rd_resp_s: chan<ReadResp> out,
        wr_req_r: chan<WriteReq> in,
        wr_resp_s: chan<WriteResp> out,

        rd_req0_s: chan<ReadReq> out,
        rd_resp0_r: chan<ReadResp> in,
        wr_req0_s: chan<WriteReq> out,
        wr_resp0_r: chan<WriteResp> in,

        rd_req1_s: chan<ReadReq> out,
        rd_resp1_r: chan<ReadResp> in,
        wr_req1_s: chan<WriteReq> out,
        wr_resp1_r: chan<WriteResp> in,

        rd_req2_s: chan<ReadReq> out,
        rd_resp2_r: chan<ReadResp> in,
        wr_req2_s: chan<WriteReq> out,
        wr_resp2_r: chan<WriteResp> in

    ) {
        spawn RamDemux3<RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>(
            sel_req_r, sel_resp_s,
            rd_req_r, rd_resp_s, wr_req_r, wr_resp_s,
            rd_req0_s, rd_resp0_r, wr_req0_s, wr_resp0_r,
            rd_req1_s, rd_resp1_r, wr_req1_s, wr_resp1_r,
            rd_req2_s, rd_resp2_r, wr_req2_s, wr_resp2_r
        );
    }

    init {  }

    next(state: ()) {  }
}

const TEST_RAM_SIZE = u32:32;
const TEST_RAM_DATA_WIDTH = u32:8;
const TEST_RAM_ADDR_WIDTH = std::clog2(TEST_RAM_SIZE);
const TEST_RAM_WORD_PARTITION_SIZE = u32:1;
const TEST_RAM_NUM_PARTITIONS = ram::num_partitions(TEST_RAM_WORD_PARTITION_SIZE, TEST_RAM_DATA_WIDTH);
const TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR = ram::SimultaneousReadWriteBehavior::READ_BEFORE_WRITE;
const TEST_RAM_INITIALIZED = true;
const TEST_DEMUX_INIT_SEL = u2:0;
const TEST_DEMUX_QUEUE_LEN = u32:5;

#[test_proc]
proc RamDemux3Test {

    type WriteReq  = ram::WriteReq<TEST_RAM_ADDR_WIDTH, TEST_RAM_DATA_WIDTH, TEST_RAM_NUM_PARTITIONS>;
    type ReadResp  = ram::ReadResp<TEST_RAM_DATA_WIDTH>;
    type ReadReq   = ram::ReadReq<TEST_RAM_ADDR_WIDTH, TEST_RAM_NUM_PARTITIONS>;
    type WriteResp = ram::WriteResp;

    type Addr = uN[TEST_RAM_ADDR_WIDTH];
    type Data = uN[TEST_RAM_DATA_WIDTH];
    type Mask = uN[TEST_RAM_NUM_PARTITIONS];

    terminator: chan<bool> out;

    sel_req_s: chan<u2> out;
    sel_resp_r: chan<()> in;

    rd_req_s: chan<ReadReq> out;
    rd_resp_r: chan<ReadResp> in;
    wr_req_s: chan<WriteReq> out;
    wr_resp_r: chan<WriteResp> in;

    rd_req0_s: chan<ReadReq> out;
    rd_resp0_r: chan<ReadResp> in;
    wr_req0_s: chan<WriteReq> out;
    wr_resp0_r: chan<WriteResp> in;

    rd_req1_s: chan<ReadReq> out;
    rd_resp1_r: chan<ReadResp> in;
    wr_req1_s: chan<WriteReq> out;
    wr_resp1_r: chan<WriteResp> in;

    rd_req2_s: chan<ReadReq> out;
    rd_resp2_r: chan<ReadResp> in;
    wr_req2_s: chan<WriteReq> out;
    wr_resp2_r: chan<WriteResp> in;

    config(terminator: chan<bool> out) {
        let (sel_req_s, sel_req_r) = chan<u2>("sel_req");
        let (sel_resp_s, sel_resp_r) = chan<()>("sel_resp");

        let (rd_req_s, rd_req_r) = chan<ReadReq>("rd_req");
        let (rd_resp_s, rd_resp_r) = chan<ReadResp>("rd_resp");
        let (wr_req_s, wr_req_r) = chan<WriteReq>("wr_req");
        let (wr_resp_s, wr_resp_r) = chan<WriteResp>("wr_resp");

        let (rd_req0_s, rd_req0_r) = chan<ReadReq>("rd_req0");
        let (rd_resp0_s, rd_resp0_r) = chan<ReadResp>("rd_resp0");
        let (wr_req0_s, wr_req0_r) = chan<WriteReq>("wr_req0");
        let (wr_resp0_s, wr_resp0_r) = chan<WriteResp>("wr_resp0");

        let (rd_req1_s, rd_req1_r) = chan<ReadReq>("rd_req1");
        let (rd_resp1_s, rd_resp1_r) = chan<ReadResp>("rd_resp1");
        let (wr_req1_s, wr_req1_r) = chan<WriteReq>("wr_req1");
        let (wr_resp1_s, wr_resp1_r) = chan<WriteResp>("wr_resp1");

        let (rd_req2_s, rd_req2_r) = chan<ReadReq>("rd_req2");
        let (rd_resp2_s, rd_resp2_r) = chan<ReadResp>("rd_resp2");
        let (wr_req2_s, wr_req2_r) = chan<WriteReq>("wr_req2");
        let (wr_resp2_s, wr_resp2_r) = chan<WriteResp>("wr_resp2");

        spawn RamDemux3<
            TEST_RAM_ADDR_WIDTH, TEST_RAM_DATA_WIDTH, TEST_RAM_NUM_PARTITIONS,
            TEST_DEMUX_INIT_SEL, TEST_DEMUX_QUEUE_LEN
        >(
            sel_req_r, sel_resp_s,
            rd_req_r, rd_resp_s, wr_req_r, wr_resp_s,
            rd_req0_s, rd_resp0_r, wr_req0_s, wr_resp0_r,
            rd_req1_s, rd_resp1_r, wr_req1_s, wr_resp1_r,
            rd_req2_s, rd_resp2_r, wr_req2_s, wr_resp2_r,
        );

        spawn ram::RamModel<
            TEST_RAM_DATA_WIDTH, TEST_RAM_SIZE, TEST_RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED
        >(rd_req0_r, rd_resp0_s, wr_req0_r, wr_resp0_s);

        spawn ram::RamModel<
            TEST_RAM_DATA_WIDTH, TEST_RAM_SIZE, TEST_RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED
        >(rd_req1_r, rd_resp1_s, wr_req1_r, wr_resp1_s);

        spawn ram::RamModel<
            TEST_RAM_DATA_WIDTH, TEST_RAM_SIZE, TEST_RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED
        >(rd_req2_r, rd_resp2_s, wr_req2_r, wr_resp2_s);

        (
            terminator, sel_req_s, sel_resp_r,
            rd_req_s,  rd_resp_r,  wr_req_s,  wr_resp_r,
            rd_req0_s, rd_resp0_r, wr_req0_s, wr_resp0_r,
            rd_req1_s, rd_resp1_r, wr_req1_s, wr_resp1_r,
            rd_req2_s, rd_resp2_r, wr_req2_s, wr_resp2_r,
        )
    }

    init {  }

    next(state: ()) {
        let tok = join();

        // Writes

        let tok = send(tok, sel_req_s, u2:0);
        let (tok, _) = recv(tok, sel_resp_r);

        let tok = send(tok, wr_req_s, WriteReq { addr: Addr:0, data: Data:0xA, mask: !Mask:0 });
        let (tok, _) = recv(tok, wr_resp_r);

        let tok = send(tok, sel_req_s, u2:1);
        let (tok, _) = recv(tok, sel_resp_r);

        let tok = send(tok, wr_req_s, WriteReq { addr: Addr:0, data: Data:0xB, mask: !Mask:0 });
        let (tok, _) = recv(tok, wr_resp_r);

        let tok = send(tok, sel_req_s, u2:2);
        let (tok, _) = recv(tok, sel_resp_r);

        let tok = send(tok, wr_req_s, WriteReq { addr: Addr:0, data: Data:0xC, mask: !Mask:0 });
        let (tok, _) = recv(tok, wr_resp_r);

        // Reads

        let tok = send(tok, sel_req_s, u2:0);
        let (tok, _) = recv(tok, sel_resp_r);

        let tok = send(tok, rd_req_s, ReadReq { addr: Addr:0, mask: !Mask:0 });
        let (tok, resp) = recv(tok, rd_resp_r);
        trace_fmt!("Value read from the first RAM: {:#x}", resp);

        let tok = send(tok, sel_req_s, u2:1);
        let (tok, _) = recv(tok, sel_resp_r);

        let tok = send(tok, rd_req_s, ReadReq { addr: Addr:0, mask: !Mask:0 });
        let (tok, resp) = recv(tok, rd_resp_r);
        trace_fmt!("Value read from the second RAM: {:#x}", resp);

        let tok = send(tok, sel_req_s, u2:2);
        let (tok, _) = recv(tok, sel_resp_r);

        let tok = send(tok, rd_req_s, ReadReq { addr: Addr:0, mask: !Mask:0 });
        let (tok, resp) = recv(tok, rd_resp_r);
        trace_fmt!("Value read from the third RAM: {:#x}", resp);

        let tok = send(tok, terminator, true);
    }
}
