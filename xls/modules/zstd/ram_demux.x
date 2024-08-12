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

// First bit of queue is not used to simplify the implementation.
// Queue end is encoded using one-hot and if it is equal to 1,
// then the queue is empty. Queue length should be greater or equal
// to RAM latency, otherways the demux might not work properly.
struct RamDemuxState<QUEUE_LEN: u32 = {u32:5}> {
    sel: u1,
    sel_q_rd: uN[QUEUE_LEN + u32:1],
    sel_q_wr: uN[QUEUE_LEN + u32:1],
    sel_q_rd_end: uN[QUEUE_LEN + u32:1],
    sel_q_wr_end: uN[QUEUE_LEN + u32:1],
}

proc RamDemux<
    ADDR_WIDTH: u32,
    DATA_WIDTH: u32,
    NUM_PARTITIONS: u32,
    INIT_SEL: u1 = {u1:0},
    QUEUE_LEN: u32 = {u32:5}
> {
    type ReadReq = ram::ReadReq<ADDR_WIDTH, NUM_PARTITIONS>;
    type ReadResp = ram::ReadResp<DATA_WIDTH>;
    type WriteReq = ram::WriteReq<ADDR_WIDTH, DATA_WIDTH, NUM_PARTITIONS>;
    type WriteResp = ram::WriteResp;

    type Queue = uN[QUEUE_LEN + u32:1];

    sel_req_r: chan<u1> in;
    sel_resp_s: chan<()> out;

    rd_req_r: chan<ReadReq> in;
    rd_resp_s: chan<ReadResp> out;
    wr_req_r: chan<WriteReq> in;
    wr_resp_s: chan<WriteResp> out;

    rd_req0_s: chan<ReadReq> out;
    rd_resp0_r: chan<ReadResp> in;
    wr_req0_s: chan<WriteReq> out;
    wr_resp0_r: chan<WriteResp> in;

    rd_req1_s: chan<ReadReq> out;
    rd_resp1_r: chan<ReadResp> in;
    wr_req1_s: chan<WriteReq> out;
    wr_resp1_r: chan<WriteResp> in;

    config(
        sel_req_r: chan<u1> in,
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
        wr_resp1_r: chan<WriteResp> in
    ) {
        (
            sel_req_r, sel_resp_s,
            rd_req_r, rd_resp_s, wr_req_r, wr_resp_s,
            rd_req0_s, rd_resp0_r, wr_req0_s, wr_resp0_r,
            rd_req1_s, rd_resp1_r, wr_req1_s, wr_resp1_r,
        )
    }

    init {
        RamDemuxState<QUEUE_LEN> {
            sel: INIT_SEL,
            sel_q_rd: Queue:0,
            sel_q_wr: Queue:0,
            sel_q_rd_end: Queue:1,
            sel_q_wr_end: Queue:1
        }
    }

    next(state: RamDemuxState<QUEUE_LEN>) {
        let sel = state.sel;
        let sel_q_rd = state.sel_q_rd;
        let sel_q_wr = state.sel_q_wr;
        let sel_q_rd_end = state.sel_q_rd_end;
        let sel_q_wr_end = state.sel_q_wr_end;

        let tok = join();

        // receive requests from input channel
        // conditional reading is not required here ase the queue would
        // never be full (assuming its length is greater or equal to RAM
        // latency), as there would be at maxiumum one new request added
        // to queue per cycle and the response for the first one should
        // be received after number of cycles equal to RAM latency (which
        // is less or equal to queue length)
        let (tok1_0, rd_req, rd_req_valid) = recv_non_blocking(tok, rd_req_r, zero!<ReadReq>());
        let (sel_q_rd_end, sel_q_rd) = if rd_req_valid {
            (sel_q_rd_end << u32:1, (sel_q_rd << u32:1) | ((sel as Queue) << u32:1))
        } else {
            (sel_q_rd_end, sel_q_rd)
        };

        let (tok1_1, wr_req, wr_req_valid) = recv_non_blocking(tok, wr_req_r, zero!<WriteReq>());
        let (sel_q_wr_end, sel_q_wr) = if wr_req_valid {
            (sel_q_wr_end << u32:1, (sel_q_wr << u32:1) | ((sel as Queue) << u32:1))
        } else {
            (sel_q_wr_end, sel_q_wr)
        };

        // send requests to output channel 0
        let rd_req0_cond = ((sel_q_rd >> u32:1) as u1 == u1:0 && rd_req_valid);
        let tok1_2 = send_if(tok, rd_req0_s, rd_req0_cond, rd_req);

        let wr_req0_cond = ((sel_q_wr >> u32:1) as u1 == u1:0 && wr_req_valid);
        let tok1_3 = send_if(tok, wr_req0_s, wr_req0_cond, wr_req);

        // send requests to output channel 1
        let rd_req1_cond = ((sel_q_rd >> u32:1) as u1 == u1:1 && rd_req_valid);
        let tok1_4 = send_if(tok, rd_req1_s, rd_req1_cond, rd_req);

        let wr_req1_cond = ((sel_q_wr >> u32:1) as u1 == u1:1 && wr_req_valid);
        let tok1_5 = send_if(tok, wr_req1_s, wr_req1_cond, wr_req);

        // join tokens
        let tok1 = join(tok1_0, tok1_1, tok1_2, tok1_3, tok1_4, tok1_5);

        // check which channel should be used for read/write
        let rd_resp_ch = if (sel_q_rd & sel_q_rd_end) == Queue:0 { u1:0 } else { u1:1 };
        let wr_resp_ch = if (sel_q_wr & sel_q_wr_end) == Queue:0 { u1:0 } else { u1:1 };

        // receive responses from output channel 0
        let (tok2_0, rd_resp0, rd_resp0_valid) =
            recv_if_non_blocking(tok1, rd_resp0_r, rd_resp_ch == u1:0, zero!<ReadResp>());
        let (tok2_1, wr_resp0, wr_resp0_valid) =
            recv_if_non_blocking(tok1, wr_resp0_r, wr_resp_ch == u1:0, zero!<WriteResp>());

        // receive responses from output channel 1
        let (tok2_2, rd_resp1, rd_resp1_valid) =
            recv_if_non_blocking(tok1, rd_resp1_r, rd_resp_ch == u1:1, zero!<ReadResp>());
        let (tok2_3, wr_resp1, wr_resp1_valid) =
            recv_if_non_blocking(tok1, wr_resp1_r, wr_resp_ch == u1:1, zero!<WriteResp>());

        // prepare read output values
        let (rd_resp, rd_resp_valid) = if rd_resp_ch == u1:0 {
            (rd_resp0, rd_resp0_valid)
        } else {
            (rd_resp1, rd_resp1_valid)
        };

        // prepare write output values
        let (wr_resp, wr_resp_valid) = if wr_resp_ch == u1:0 {
            (wr_resp0, wr_resp0_valid)
        } else {
            (wr_resp1, wr_resp1_valid)
        };

        // send responses to input channel
        let tok2_4 = send_if(tok1, rd_resp_s, rd_resp_valid, rd_resp);
        let sel_q_rd_end = if rd_resp_valid { sel_q_rd_end >> u32:1 } else { sel_q_rd_end };

        let tok2_5 = send_if(tok1, wr_resp_s, wr_resp_valid, wr_resp);
        let sel_q_wr_end = if wr_resp_valid { sel_q_wr_end >> u32:1 } else { sel_q_wr_end };

        // handle select
        let (tok1_6, sel, sel_valid) = recv_non_blocking(tok, sel_req_r, sel);

        let tok1_7 = send_if(tok1_6, sel_resp_s, sel_valid, ());

        RamDemuxState<QUEUE_LEN> { sel, sel_q_rd, sel_q_wr, sel_q_rd_end, sel_q_wr_end }
    }
}

const TEST_RAM_SIZE = u32:32;
const TEST_RAM_DATA_WIDTH = u32:8;
const TEST_RAM_ADDR_WIDTH = std::clog2(TEST_RAM_SIZE);
const TEST_RAM_WORD_PARTITION_SIZE = u32:1;
const TEST_RAM_NUM_PARTITIONS = ram::num_partitions(
    TEST_RAM_WORD_PARTITION_SIZE, TEST_RAM_DATA_WIDTH);
const TEST_DEMUX_INIT_SEL = u1:0;
const TEST_DEMUX_QUEUE_LEN = u32:5;

type TestWriteReq = ram::WriteReq<TEST_RAM_ADDR_WIDTH, TEST_RAM_DATA_WIDTH, TEST_RAM_NUM_PARTITIONS>;
type TestReadResp = ram::ReadResp<TEST_RAM_DATA_WIDTH>;
type TestReadReq = ram::ReadReq<TEST_RAM_ADDR_WIDTH, TEST_RAM_NUM_PARTITIONS>;
type TestWriteResp = ram::WriteResp;
type TestDemuxAddr = uN[TEST_RAM_ADDR_WIDTH];
type TestDemuxData = uN[TEST_RAM_DATA_WIDTH];

fn TestDemuxWriteWordReq(addr: TestDemuxAddr, data: TestDemuxData) -> TestWriteReq {
    ram::WriteWordReq<TEST_RAM_NUM_PARTITIONS>(addr, data)
}

fn TestDemuxReadWordReq(addr: TestDemuxAddr) -> TestReadReq {
    ram::ReadWordReq<TEST_RAM_NUM_PARTITIONS>(addr)
}

#[test_proc]
proc RamDemuxTest {
    terminator: chan<bool> out;

    sel_req_s: chan<u1> out;
    sel_resp_r: chan<()> in;

    rd_req_s: chan<TestReadReq> out;
    rd_resp_r: chan<TestReadResp> in;
    wr_req_s: chan<TestWriteReq> out;
    wr_resp_r: chan<TestWriteResp> in;

    rd_req0_s: chan<TestReadReq> out;
    rd_resp0_r: chan<TestReadResp> in;
    wr_req0_s: chan<TestWriteReq> out;
    wr_resp0_r: chan<TestWriteResp> in;

    rd_req1_s: chan<TestReadReq> out;
    rd_resp1_r: chan<TestReadResp> in;
    wr_req1_s: chan<TestWriteReq> out;
    wr_resp1_r: chan<TestWriteResp> in;

    config(terminator: chan<bool> out) {
        let (sel_req_s, sel_req_r) = chan<u1>("sel_req");
        let (sel_resp_s, sel_resp_r) = chan<()>("sel_resp");

        let (rd_req_s, rd_req_r) = chan<TestReadReq>("rd_req");
        let (rd_resp_s, rd_resp_r) = chan<TestReadResp>("rd_resp");
        let (wr_req_s, wr_req_r) = chan<TestWriteReq>("wr_req");
        let (wr_resp_s, wr_resp_r) = chan<TestWriteResp>("wr_resp");

        let (rd_req0_s, rd_req0_r) = chan<TestReadReq>("rd_req0");
        let (rd_resp0_s, rd_resp0_r) = chan<TestReadResp>("rd_resp0");
        let (wr_req0_s, wr_req0_r) = chan<TestWriteReq>("wr_req0");
        let (wr_resp0_s, wr_resp0_r) = chan<TestWriteResp>("wr_resp0");

        let (rd_req1_s, rd_req1_r) = chan<TestReadReq>("rd_req1");
        let (rd_resp1_s, rd_resp1_r) = chan<TestReadResp>("rd_resp1");
        let (wr_req1_s, wr_req1_r) = chan<TestWriteReq>("wr_req1");
        let (wr_resp1_s, wr_resp1_r) = chan<TestWriteResp>("wr_resp1");

        spawn RamDemux<
            TEST_RAM_ADDR_WIDTH, TEST_RAM_DATA_WIDTH, TEST_RAM_NUM_PARTITIONS,
            TEST_DEMUX_INIT_SEL, TEST_DEMUX_QUEUE_LEN
        >(
            sel_req_r, sel_resp_s,
            rd_req_r, rd_resp_s, wr_req_r, wr_resp_s,
            rd_req0_s, rd_resp0_r, wr_req0_s, wr_resp0_r,
            rd_req1_s, rd_resp1_r, wr_req1_s, wr_resp1_r
        );

        spawn ram::RamModel<TEST_RAM_DATA_WIDTH, TEST_RAM_SIZE, TEST_RAM_WORD_PARTITION_SIZE>(
            rd_req0_r, rd_resp0_s, wr_req0_r, wr_resp0_s);

        spawn ram::RamModel<TEST_RAM_DATA_WIDTH, TEST_RAM_SIZE, TEST_RAM_WORD_PARTITION_SIZE>(
            rd_req1_r, rd_resp1_s, wr_req1_r, wr_resp1_s);
        (
            terminator, sel_req_s, sel_resp_r,
            rd_req_s, rd_resp_r, wr_req_s, wr_resp_r,
            rd_req0_s, rd_resp0_r, wr_req0_s, wr_resp0_r,
            rd_req1_s, rd_resp1_r, wr_req1_s, wr_resp1_r,
        )
    }

    init {  }

    next(state: ()) {
        let tok = join();
        // test case 0: write data with demux to ram0 and read directly
        let addr = TestDemuxAddr:0;
        // request
        let req = TestDemuxWriteWordReq(addr, TestDemuxData:0x12);
        // set sel to 0
        let tok = send(tok, sel_req_s, u1:0);
        let (tok, _) = recv(tok, sel_resp_r);
        // write via demux
        let tok = send(tok, wr_req_s, req);
        let (tok, _) = recv(tok, wr_resp_r);
        // read directly from ram0
        let tok = send(tok, rd_req0_s, TestDemuxReadWordReq(addr));
        let (tok, resp) = recv(tok, rd_resp0_r);
        assert_eq(resp.data, req.data);

        // test case 1: write data with demux to ram1 and read directly
        let addr = TestDemuxAddr:1;
        // request
        let req = TestDemuxWriteWordReq(addr, TestDemuxData:0x34);
        // set sel to 1
        let tok = send(tok, sel_req_s, u1:1);
        let (tok, _) = recv(tok, sel_resp_r);
        // write via demux
        let tok = send(tok, wr_req_s, req);
        let (tok, _) = recv(tok, wr_resp_r);
        // read directly from ram1
        let tok = send(tok, rd_req1_s, TestDemuxReadWordReq(addr));
        let (tok, resp) = recv(tok, rd_resp1_r);
        assert_eq(resp.data, req.data);

        // test case 2: write data directly to ram0 and read with demux
        let addr = TestDemuxAddr:0;
        // request
        let req = TestDemuxWriteWordReq(addr, TestDemuxData:0x56);
        // write directly to ram0
        let tok = send(tok, wr_req0_s, req);
        let (tok, _) = recv(tok, wr_resp0_r);
        // set sel to 0
        let tok = send(tok, sel_req_s, u1:0);
        let (tok, _) = recv(tok, sel_resp_r);
        // read via demux
        let tok = send(tok, rd_req_s, TestDemuxReadWordReq(addr));
        let (tok, resp) = recv(tok, rd_resp_r);
        assert_eq(resp.data, req.data);

        // test case 3: write data directly to ram1 and read with demux
        let addr = TestDemuxAddr:0;
        // request
        let req = TestDemuxWriteWordReq(addr, TestDemuxData:0x78);
        // write directly to ram1
        let tok = send(tok, wr_req1_s, req);
        let (tok, _) = recv(tok, wr_resp1_r);
        // set sel to 1
        let tok = send(tok, sel_req_s, u1:1);
        let (tok, _) = recv(tok, sel_resp_r);
        // read via demux
        let tok = send(tok, rd_req_s, TestDemuxReadWordReq(addr));
        let (tok, resp) = recv(tok, rd_resp_r);
        assert_eq(resp.data, req.data);

        // test case 4: try to switch sel during write
        let addr = TestDemuxAddr:1;
        // request
        let req0 = TestDemuxWriteWordReq(addr, TestDemuxData:0xAB);
        let req1 = TestDemuxWriteWordReq(addr, TestDemuxData:0xCD);
        // set sel to 0
        let tok = send(tok, sel_req_s, u1:0);
        let (tok, _) = recv(tok, sel_resp_r);
        // start write via demux
        let tok = send(tok, wr_req_s, req0);
        // set sel to 1 during read
        let tok = send(tok, sel_req_s, u1:1);
        let (tok, _) = recv(tok, sel_resp_r);
        // finish write via demux
        let (tok, _) = recv(tok, wr_resp_r);
        // perform second write
        let tok = send(tok, wr_req_s, req1);
        let (tok, _) = recv(tok, wr_resp_r);
        // read directly from ram0 and assert data from req0 was written
        let tok = send(tok, rd_req0_s, TestDemuxReadWordReq(addr));
        let (tok, resp) = recv(tok, rd_resp0_r);
        assert_eq(resp.data, req0.data);
        // read directly from ram1 and assert data from req1 was written
        let tok = send(tok, rd_req1_s, TestDemuxReadWordReq(addr));
        let (tok, resp) = recv(tok, rd_resp1_r);
        assert_eq(resp.data, req1.data);

        // test case 5: try to switch sel during read
        let addr = TestDemuxAddr:1;
        // request
        let req0 = TestDemuxWriteWordReq(addr, TestDemuxData:0xAB);
        // write directly to ram0
        let tok = send(tok, wr_req0_s, req0);
        let (tok, _) = recv(tok, wr_resp0_r);
        let req1 = TestDemuxWriteWordReq(addr, TestDemuxData:0xCD);
        // write directly to ram1
        let tok = send(tok, wr_req1_s, req1);
        let (tok, _) = recv(tok, wr_resp1_r);
        // set sel to 0
        let tok = send(tok, sel_req_s, u1:0);
        let (tok, _) = recv(tok, sel_resp_r);
        // start read via demux
        let tok = send(tok, rd_req_s, TestDemuxReadWordReq(addr));
        // set sel to 1 during read
        let tok = send(tok, sel_req_s, u1:1);
        let (tok, _) = recv(tok, sel_resp_r);
        // finish read via demux
        let (tok, resp0) = recv(tok, rd_resp_r);
        // perform second read
        let tok = send(tok, rd_req_s, TestDemuxReadWordReq(addr));
        let (tok, resp1) = recv(tok, rd_resp_r);
        // assert that first read returned data from ram0
        assert_eq(resp0.data, req0.data);
        // assert that second read returned data from ram1
        assert_eq(resp1.data, req1.data);

        // test case 6: sending more write requests than queue can hold
        // set sel to 0
        let tok = send(tok, sel_req_s, u1:0);
        let (tok, _) = recv(tok, sel_resp_r);
        // send 8 write requests
        let tok = for (i, tok): (u32, token) in range(u32:0, TEST_DEMUX_QUEUE_LEN + u32:3) {
            let req = TestDemuxWriteWordReq(i as TestDemuxAddr, i as TestDemuxData);
            let tok = send(tok, wr_req_s, req);
            let (tok, _) = recv(tok, wr_resp_r);
            tok
        }(tok);
        // read values directly from ram
        let tok = for (i, tok): (u32, token) in range(u32:0, TEST_DEMUX_QUEUE_LEN + u32:3) {
            let req0 = TestDemuxReadWordReq(i as TestDemuxAddr);
            let tok = send(tok, rd_req0_s, req0);
            let (tok, resp0) = recv(tok, rd_resp0_r);
            assert_eq(resp0.data, i as TestDemuxData);
            tok
        }(tok);

        // test case 7: sending more read requests than queue can hold
        // set sel to 1
        let tok = send(tok, sel_req_s, u1:1);
        let (tok, _) = recv(tok, sel_resp_r);
        // write values directly to ram
        let tok = for (i, tok): (u32, token) in range(u32:0, TEST_DEMUX_QUEUE_LEN + u32:3) {
            let req1 = TestDemuxWriteWordReq(i as TestDemuxAddr, i as TestDemuxData);
            let tok = send(tok, wr_req1_s, req1);
            let (tok, _) = recv(tok, wr_resp1_r);
            tok
        }(tok);
        // send 8 write requests
        let tok = for (i, tok): (u32, token) in range(u32:0, TEST_DEMUX_QUEUE_LEN + u32:3) {
            let req = TestDemuxReadWordReq(i as TestDemuxAddr);
            let tok = send(tok, rd_req_s, req);
            let (tok, resp) = recv(tok, rd_resp_r);
            assert_eq(resp.data, i as TestDemuxData);
            tok
        }(tok);

        let tok = send(tok, terminator, true);
    }
}

const RAM_SIZE = u32:32;
const RAM_DATA_WIDTH = u32:8;
const RAM_ADDR_WIDTH = std::clog2(RAM_SIZE);
const RAM_WORD_PARTITION_SIZE = u32:1;
const RAM_NUM_PARTITIONS = ram::num_partitions(RAM_WORD_PARTITION_SIZE, RAM_DATA_WIDTH);

// Sample for codegen
pub proc RamDemuxInst {
    type ReadReq = ram::ReadReq<RAM_ADDR_WIDTH, RAM_NUM_PARTITIONS>;
    type ReadResp = ram::ReadResp<RAM_DATA_WIDTH>;
    type WriteReq = ram::WriteReq<RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>;
    type WriteResp = ram::WriteResp;

    config(
        sel_req_r: chan<u1> in,
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
        wr_resp1_r: chan<WriteResp> in
    ) {
        spawn RamDemux<RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>(
            sel_req_r, sel_resp_s,
            rd_req_r, rd_resp_s, wr_req_r, wr_resp_s,
            rd_req0_s, rd_resp0_r, wr_req0_s, wr_resp0_r,
            rd_req1_s, rd_resp1_r, wr_req1_s, wr_resp1_r
        );
    }

    init {  }

    next(state: ()) {  }
}

struct RamDemuxNaiveState { sel: u1 }

// This implementation does not support sel switching during read/write operation
proc RamDemuxNaive<
    ADDR_WIDTH: u32,
    DATA_WIDTH: u32,
    NUM_PARTITIONS: u32,
    INIT_SEL: u1 = {u1:0}
> {
    type ReadReq = ram::ReadReq<ADDR_WIDTH, NUM_PARTITIONS>;
    type ReadResp = ram::ReadResp<DATA_WIDTH>;
    type WriteReq = ram::WriteReq<ADDR_WIDTH, DATA_WIDTH, NUM_PARTITIONS>;
    type WriteResp = ram::WriteResp;

    sel_req_r: chan<u1> in;
    sel_resp_s: chan<()> out;

    rd_req_r: chan<ReadReq> in;
    rd_resp_s: chan<ReadResp> out;
    wr_req_r: chan<WriteReq> in;
    wr_resp_s: chan<WriteResp> out;

    rd_req0_s: chan<ReadReq> out;
    rd_resp0_r: chan<ReadResp> in;
    wr_req0_s: chan<WriteReq> out;
    wr_resp0_r: chan<WriteResp> in;

    rd_req1_s: chan<ReadReq> out;
    rd_resp1_r: chan<ReadResp> in;
    wr_req1_s: chan<WriteReq> out;
    wr_resp1_r: chan<WriteResp> in;

    config(
        sel_req_r: chan<u1> in,
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
        wr_resp1_r: chan<WriteResp> in
    ) {
        (
            sel_req_r, sel_resp_s,
            rd_req_r, rd_resp_s, wr_req_r, wr_resp_s,
            rd_req0_s, rd_resp0_r, wr_req0_s, wr_resp0_r,
            rd_req1_s, rd_resp1_r, wr_req1_s, wr_resp1_r,
        )
    }

    init { RamDemuxNaiveState { sel: INIT_SEL } }

    next(state: RamDemuxNaiveState) {
        let tok = join();

        let sel = state.sel;

        // receive requests from input channel
        let (tok1_0, rd_req, rd_req_valid) = recv_non_blocking(tok, rd_req_r, zero!<ReadReq>());
        let (tok1_1, wr_req, wr_req_valid) = recv_non_blocking(tok, wr_req_r, zero!<WriteReq>());

        // send requests to output channel 0
        let rd_req0_cond = (sel == u1:0 && rd_req_valid);
        let tok1_2 = send_if(tok, rd_req0_s, rd_req0_cond, rd_req);

        let wr_req0_cond = (sel == u1:0 && wr_req_valid);
        let tok1_3 = send_if(tok, wr_req0_s, wr_req0_cond, wr_req);

        // send requests to output channel 1
        let rd_req1_cond = (sel == u1:1 && rd_req_valid);
        let tok1_4 = send_if(tok, rd_req1_s, rd_req1_cond, rd_req);

        let wr_req1_cond = (sel == u1:1 && wr_req_valid);
        let tok1_5 = send_if(tok, wr_req1_s, wr_req1_cond, wr_req);

        // join tokens
        let tok1 = join(tok1_0, tok1_1, tok1_2, tok1_3, tok1_4, tok1_5);

        // receive responses from output channel 0
        let (tok2_0, rd_resp0, rd_resp0_valid) =
            recv_if_non_blocking(tok1, rd_resp0_r, sel == u1:0, zero!<ReadResp>());
        let (tok2_1, wr_resp0, wr_resp0_valid) =
            recv_if_non_blocking(tok1, wr_resp0_r, sel == u1:0, zero!<WriteResp>());

        // receive responses from output channel 1
        let (tok2_2, rd_resp1, rd_resp1_valid) =
            recv_if_non_blocking(tok1, rd_resp1_r, sel == u1:1, zero!<ReadResp>());
        let (tok2_3, wr_resp1, wr_resp1_valid) =
            recv_if_non_blocking(tok1, wr_resp1_r, sel == u1:1, zero!<WriteResp>());

        // prepare output values
        let (rd_resp, rd_resp_valid, wr_resp, wr_resp_valid) = if sel == u1:0 {
            (rd_resp0, rd_resp0_valid, wr_resp0, wr_resp0_valid)
        } else {
            (rd_resp1, rd_resp1_valid, wr_resp1, wr_resp1_valid)
        };

        // send responses to input channel
        let tok2_4 = send_if(tok1, rd_resp_s, rd_resp_valid, rd_resp);
        let tok2_5 = send_if(tok1, wr_resp_s, wr_resp_valid, wr_resp);

        // handle select
        let (tok1_6, sel, sel_valid) = recv_non_blocking(tok, sel_req_r, sel);

        let tok1_7 = send_if(tok1_6, sel_resp_s, sel_valid, ());

        RamDemuxNaiveState { sel }
    }
}

#[test_proc]
proc RamDemuxNaiveTest {
    terminator: chan<bool> out;

    sel_req_s: chan<u1> out;
    sel_resp_r: chan<()> in;

    rd_req_s: chan<TestReadReq> out;
    rd_resp_r: chan<TestReadResp> in;
    wr_req_s: chan<TestWriteReq> out;
    wr_resp_r: chan<TestWriteResp> in;

    rd_req0_s: chan<TestReadReq> out;
    rd_resp0_r: chan<TestReadResp> in;
    wr_req0_s: chan<TestWriteReq> out;
    wr_resp0_r: chan<TestWriteResp> in;

    rd_req1_s: chan<TestReadReq> out;
    rd_resp1_r: chan<TestReadResp> in;
    wr_req1_s: chan<TestWriteReq> out;
    wr_resp1_r: chan<TestWriteResp> in;

    config(terminator: chan<bool> out) {
        let (sel_req_s, sel_req_r) = chan<u1>("sel_req");
        let (sel_resp_s, sel_resp_r) = chan<()>("sel_resp");

        let (rd_req_s, rd_req_r) = chan<TestReadReq>("rd_req");
        let (rd_resp_s, rd_resp_r) = chan<TestReadResp>("rd_resp");
        let (wr_req_s, wr_req_r) = chan<TestWriteReq>("wr_req");
        let (wr_resp_s, wr_resp_r) = chan<TestWriteResp>("wr_resp");

        let (rd_req0_s, rd_req0_r) = chan<TestReadReq>("rd_req0");
        let (rd_resp0_s, rd_resp0_r) = chan<TestReadResp>("rd_resp0");
        let (wr_req0_s, wr_req0_r) = chan<TestWriteReq>("wr_req0");
        let (wr_resp0_s, wr_resp0_r) = chan<TestWriteResp>("wr_resp0");

        let (rd_req1_s, rd_req1_r) = chan<TestReadReq>("wr_req1");
        let (rd_resp1_s, rd_resp1_r) = chan<TestReadResp>("wr_resp1");
        let (wr_req1_s, wr_req1_r) = chan<TestWriteReq>("wr_req1");
        let (wr_resp1_s, wr_resp1_r) = chan<TestWriteResp>("wr_resp1");

        spawn RamDemuxNaive<TEST_RAM_ADDR_WIDTH, TEST_RAM_DATA_WIDTH, TEST_RAM_NUM_PARTITIONS>(
            sel_req_r, sel_resp_s,
            rd_req_r, rd_resp_s, wr_req_r, wr_resp_s,
            rd_req0_s, rd_resp0_r, wr_req0_s, wr_resp0_r,
            rd_req1_s, rd_resp1_r, wr_req1_s, wr_resp1_r
        );

        spawn ram::RamModel<TEST_RAM_DATA_WIDTH, TEST_RAM_SIZE, TEST_RAM_WORD_PARTITION_SIZE>(
            rd_req0_r, rd_resp0_s, wr_req0_r, wr_resp0_s);

        spawn ram::RamModel<TEST_RAM_DATA_WIDTH, TEST_RAM_SIZE, TEST_RAM_WORD_PARTITION_SIZE>(
            rd_req1_r, rd_resp1_s, wr_req1_r, wr_resp1_s);
        (
            terminator, sel_req_s, sel_resp_r,
            rd_req_s, rd_resp_r, wr_req_s, wr_resp_r,
            rd_req0_s, rd_resp0_r, wr_req0_s, wr_resp0_r,
            rd_req1_s, rd_resp1_r, wr_req1_s, wr_resp1_r,
        )
    }

    init {  }

    next(state: ()) {
        let tok = join();
        // test case 0: write data with demux to ram0 and read directly
        let addr = TestDemuxAddr:0;
        // request
        let req = TestDemuxWriteWordReq(addr, TestDemuxData:0x12);
        // set sel to 0
        let tok = send(tok, sel_req_s, u1:0);
        let (tok, _) = recv(tok, sel_resp_r);
        // write via demux
        let tok = send(tok, wr_req_s, req);
        let (tok, _) = recv(tok, wr_resp_r);
        // read directly from ram0
        let tok = send(tok, rd_req0_s, TestDemuxReadWordReq(addr));
        let (tok, resp) = recv(tok, rd_resp0_r);
        assert_eq(resp.data, req.data);

        // test case 1: write data with demux to ram1 and read directly
        let addr = TestDemuxAddr:1;
        // request
        let req = TestDemuxWriteWordReq(addr, TestDemuxData:0x34);
        // set sel to 1
        let tok = send(tok, sel_req_s, u1:1);
        let (tok, _) = recv(tok, sel_resp_r);
        // write via demux
        let tok = send(tok, wr_req_s, req);
        let (tok, _) = recv(tok, wr_resp_r);
        // read directly from ram1
        let tok = send(tok, rd_req1_s, TestDemuxReadWordReq(addr));
        let (tok, resp) = recv(tok, rd_resp1_r);
        assert_eq(resp.data, req.data);

        // test case 2: write data directly to ram0 and read with demux
        let addr = TestDemuxAddr:0;
        // request
        let req = TestDemuxWriteWordReq(addr, TestDemuxData:0x56);
        // write directly to ram0
        let tok = send(tok, wr_req0_s, req);
        let (tok, _) = recv(tok, wr_resp0_r);
        // set sel to 0
        let tok = send(tok, sel_req_s, u1:0);
        let (tok, _) = recv(tok, sel_resp_r);
        // read via demux
        let tok = send(tok, rd_req_s, TestDemuxReadWordReq(addr));
        let (tok, resp) = recv(tok, rd_resp_r);
        assert_eq(resp.data, req.data);

        // test case 3: write data directly to ram1 and read with demux
        let addr = TestDemuxAddr:0;
        // request
        let req = TestDemuxWriteWordReq(addr, TestDemuxData:0x78);
        // write directly to ram1
        let tok = send(tok, wr_req1_s, req);
        let (tok, _) = recv(tok, wr_resp1_r);
        // set sel to 1
        let tok = send(tok, sel_req_s, u1:1);
        let (tok, _) = recv(tok, sel_resp_r);
        // read via demux
        let tok = send(tok, rd_req_s, TestDemuxReadWordReq(addr));
        let (tok, resp) = recv(tok, rd_resp_r);
        assert_eq(resp.data, req.data);

        // test cases 4 and 5 from RamDemuxTest are not relevant here as this naive
        // implementation does not support sel switching during read/write operations

        let tok = send(tok, terminator, true);
    }
}

// Sample for codegen
pub proc RamDemuxNaiveInst {
    type ReadReq = ram::ReadReq<RAM_ADDR_WIDTH, RAM_NUM_PARTITIONS>;
    type ReadResp = ram::ReadResp<RAM_DATA_WIDTH>;
    type WriteReq = ram::WriteReq<RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>;
    type WriteResp = ram::WriteResp;

    config(
        sel_req_r: chan<u1> in,
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
        wr_resp1_r: chan<WriteResp> in
    ) {
        spawn RamDemuxNaive<RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>(
            sel_req_r, sel_resp_s,
            rd_req_r, rd_resp_s, wr_req_r, wr_resp_s,
            rd_req0_s, rd_resp0_r, wr_req0_s, wr_resp0_r,
            rd_req1_s, rd_resp1_r, wr_req1_s, wr_resp1_r
        );
    }

    init {  }

    next(state: ()) {  }
}
