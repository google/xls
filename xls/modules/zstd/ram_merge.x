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

pub proc RamMerge<
    RAM_ADDR_W: u32,
    RAM_DATA_W: u32,
    RAM_NUM_PARTITIONS: u32
> {
    type ReadReq = ram::ReadReq<RAM_ADDR_W, RAM_NUM_PARTITIONS>;
    type ReadResp = ram::ReadResp<RAM_DATA_W>;
    type WriteReq = ram::WriteReq<RAM_ADDR_W, RAM_DATA_W, RAM_NUM_PARTITIONS>;
    type WriteResp = ram::WriteResp;

    init {}

    read_side_rd_req_r: chan<ReadReq> in;
    read_side_rd_resp_s: chan<ReadResp> out;

    write_side_wr_req_r: chan<WriteReq> in;
    write_side_wr_resp_s: chan<WriteResp> out;

    merge_side_rd_req_s: chan<ReadReq> out;
    merge_side_rd_resp_r: chan<ReadResp> in;
    merge_side_wr_req_s: chan<WriteReq> out;
    merge_side_wr_resp_r: chan<WriteResp> in;

    config(
        read_side_rd_req_r: chan<ReadReq> in,
        read_side_rd_resp_s: chan<ReadResp> out,

        write_side_wr_req_r: chan<WriteReq> in,
        write_side_wr_resp_s: chan<WriteResp> out,

        merge_side_rd_req_s: chan<ReadReq> out,
        merge_side_rd_resp_r: chan<ReadResp> in,
        merge_side_wr_req_s: chan<WriteReq> out,
        merge_side_wr_resp_r: chan<WriteResp> in,
    ) {
        (
            read_side_rd_req_r, read_side_rd_resp_s,

            write_side_wr_req_r, write_side_wr_resp_s,

            merge_side_rd_req_s, merge_side_rd_resp_r,
            merge_side_wr_req_s, merge_side_wr_resp_r
        )
    }

    next (state: ()) {
        let tok = join();

        // Passthrough Requests
        let (tok_rd, rd_req, rd_req_valid) = recv_non_blocking(tok, read_side_rd_req_r, zero!<ReadReq>());
        let (tok_rd, rd_resp, rd_resp_valid) = recv_non_blocking(tok_rd, merge_side_rd_resp_r, zero!<ReadResp>());
        let tok_rd = send_if(tok_rd, merge_side_rd_req_s, rd_req_valid, rd_req);
        let tok_rd = send_if(tok_rd, read_side_rd_resp_s, rd_resp_valid, rd_resp);

        let (tok_wr, wr_req, wr_req_valid) = recv_non_blocking(tok, write_side_wr_req_r, zero!<WriteReq>());
        let (tok_wr, wr_resp, wr_resp_valid) = recv_non_blocking(tok_wr, merge_side_wr_resp_r, zero!<WriteResp>());
        let tok_wr = send_if(tok_wr, merge_side_wr_req_s, wr_req_valid, wr_req);
        let tok_wr = send_if(tok_wr, write_side_wr_resp_s, wr_resp_valid, wr_resp);

        let tok_joined = join(tok_rd, tok_wr);
    }
}
