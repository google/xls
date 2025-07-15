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

pub proc RamPassthroughRead<
    ADDR_WIDTH: u32,
    DATA_WIDTH: u32,
    NUM_PARTITIONS: u32,
> {
    type ReadReq = ram::ReadReq<ADDR_WIDTH, NUM_PARTITIONS>;
    type ReadResp = ram::ReadResp<DATA_WIDTH>;

    rd_req_r: chan<ReadReq> in;
    rd_resp_s: chan<ReadResp> out;
    rd_req_s: chan<ReadReq> out;
    rd_resp_r: chan<ReadResp> in;

    config(
        rd_req_r: chan<ReadReq> in,
        rd_resp_s: chan<ReadResp> out,
        rd_req_s: chan<ReadReq> out,
        rd_resp_r: chan<ReadResp> in,
    ) {
        (
            rd_req_r, rd_resp_s,
            rd_req_s, rd_resp_r,
        )
    }

    init {}

    next(state: ()) {
        let tok0 = join();

        let (tok_rd_req, rd_req) = recv(join(), rd_req_r);
        let tok_sent_rd_req = send(tok_rd_req, rd_req_s, rd_req);

        let (tok_rd_resp, rd_resp) = recv(tok_sent_rd_req, rd_resp_r);
        let tok_sent_rd_resp = send(tok_rd_resp, rd_resp_s, rd_resp);
    }
}

pub proc RamPassthroughWrite<
    ADDR_WIDTH: u32,
    DATA_WIDTH: u32,
    NUM_PARTITIONS: u32,
> {
    type WriteReq = ram::WriteReq<ADDR_WIDTH, DATA_WIDTH, NUM_PARTITIONS>;
    type WriteResp = ram::WriteResp;

    wr_req_r: chan<WriteReq> in;
    wr_resp_s: chan<WriteResp> out;
    wr_req_s: chan<WriteReq> out;
    wr_resp_r: chan<WriteResp> in;

    config(
        wr_req_r: chan<WriteReq> in,
        wr_resp_s: chan<WriteResp> out,
        wr_req_s: chan<WriteReq> out,
        wr_resp_r: chan<WriteResp> in,
    ) {
        (
            wr_req_r, wr_resp_s,
            wr_req_s, wr_resp_r,
        )
    }

    init {}

    next(state: ()) {
        let (tok_wr_req, wr_req) = recv(join(), wr_req_r);
        let tok_sent_wr_req = send(tok_wr_req, wr_req_s, wr_req);

        let (tok_wr_resp, wr_resp) = recv(tok_sent_wr_req, wr_resp_r);
        let tok_sent_wr_resp = send(tok_wr_resp, wr_resp_s, wr_resp);
    }
}

pub proc RamPassthrough<
    ADDR_WIDTH: u32,
    DATA_WIDTH: u32,
    NUM_PARTITIONS: u32,
> {
    type ReadReq = ram::ReadReq<ADDR_WIDTH, NUM_PARTITIONS>;
    type ReadResp = ram::ReadResp<DATA_WIDTH>;
    type WriteReq = ram::WriteReq<ADDR_WIDTH, DATA_WIDTH, NUM_PARTITIONS>;
    type WriteResp = ram::WriteResp;

    config(
        rd_req_r: chan<ReadReq> in,
        rd_resp_s: chan<ReadResp> out,
        wr_req_r: chan<WriteReq> in,
        wr_resp_s: chan<WriteResp> out,

        rd_req_s: chan<ReadReq> out,
        rd_resp_r: chan<ReadResp> in,
        wr_req_s: chan<WriteReq> out,
        wr_resp_r: chan<WriteResp> in,
    ) {
        spawn RamPassthroughRead<ADDR_WIDTH, DATA_WIDTH, NUM_PARTITIONS>(rd_req_r, rd_resp_s, rd_req_s, rd_resp_r);
        spawn RamPassthroughWrite<ADDR_WIDTH, DATA_WIDTH, NUM_PARTITIONS>(wr_req_r, wr_resp_s, wr_req_s, wr_resp_r);
    }

    init {}
    next(state: ()) { }
}

const RAM_SIZE = u32:1024;
const RAM_DATA_WIDTH = u32:64;
const RAM_ADDR_WIDTH = std::clog2(RAM_SIZE);
const RAM_WORD_PARTITION_SIZE = u32:1;
const RAM_NUM_PARTITIONS = ram::num_partitions(RAM_WORD_PARTITION_SIZE, RAM_DATA_WIDTH);

pub proc RamPassthroughInst {
    type ReadReq = ram::ReadReq<RAM_ADDR_WIDTH, RAM_NUM_PARTITIONS>;
    type ReadResp = ram::ReadResp<RAM_DATA_WIDTH>;
    type WriteReq = ram::WriteReq<RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>;
    type WriteResp = ram::WriteResp;

    config(
        rd_req_r: chan<ReadReq> in,
        rd_resp_s: chan<ReadResp> out,
        wr_req_r: chan<WriteReq> in,
        wr_resp_s: chan<WriteResp> out,

        rd_req_s: chan<ReadReq> out,
        rd_resp_r: chan<ReadResp> in,
        wr_req_s: chan<WriteReq> out,
        wr_resp_r: chan<WriteResp> in,
    ) {
        spawn RamPassthrough<RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>(
            rd_req_r, rd_resp_s, wr_req_r, wr_resp_s,
            rd_req_s, rd_resp_r, wr_req_s, wr_resp_r,
        );
    }

    init {  }
    next(state: ()) {  }
}
