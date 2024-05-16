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

// This file contains the implementation of a proc responsible for receiving
// RAM completions and counting their number. The proc might be used to simplify
// the implementation of other procedures.

import std;
import xls.examples.ram;

proc RamWrRespHandler<CNT_WIDTH: u32> {
    type Reset = bool;
    type WriteCnt = bits[CNT_WIDTH];
    type WriteResp = ram::WriteResp;

    input_r: chan<Reset> in;
    output_s: chan<WriteCnt> out;
    wr_resp_r: chan<WriteResp> in;

    config(
        input_r: chan<Reset> in,
        output_s: chan<WriteCnt> out,
        wr_resp_r: chan<WriteResp> in
    ) {
        (input_r, output_s, wr_resp_r)
    }

    init { WriteCnt:0 }

    next(tok0: token, wr_cnt: WriteCnt) {
        let (tok1, reset) = recv(tok0, input_r);
        recv(tok1, wr_resp_r);

        let wr_cnt = if reset { WriteCnt:1 } else { wr_cnt };
        send(tok1, output_s, wr_cnt);

        wr_cnt + WriteCnt:1
    }
}

const INST_CNT_WIDTH = u32:32;
proc RamWrRespHandlerInst {
    type Reset = bool;
    type WriteCnt = bits[INST_CNT_WIDTH];
    type WriteResp = ram::WriteResp;

    config(
        input_r: chan<Reset> in,
        output_s: chan<WriteCnt> out,
        wr_resp_r: chan<WriteResp> in
    ) {
        spawn RamWrRespHandler<INST_CNT_WIDTH>(input_r, output_s, wr_resp_r);
    }

    init { }
    next(tok0: token, state: ()) { }
}

const TEST_CNT_WIDTH = u32:32;
const TEST_RAM_DATA_WIDTH = u32:8;
const TEST_RAM_SIZE = u32:256;
const TEST_RAM_ADDR_WIDTH = std::clog2(TEST_RAM_SIZE);
const TEST_RAM_WORD_PARTITION_SIZE = TEST_RAM_DATA_WIDTH;
const TEST_RAM_NUM_PARTITIONS = ram::num_partitions(TEST_RAM_WORD_PARTITION_SIZE, TEST_RAM_DATA_WIDTH);

const TEST_SYMBOLS_TO_SEND = u32:12;

struct RamWrRespHandlerTestState { rd_cnt: u32, wr_cnt: u32 }

#[test_proc]
proc RamWrRespHandlerTest {
    type RamReadReq = ram::ReadReq<TEST_RAM_ADDR_WIDTH, TEST_RAM_NUM_PARTITIONS>;
    type RamReadResp = ram::ReadResp<TEST_RAM_DATA_WIDTH>;
    type RamWriteReq = ram::WriteReq<TEST_RAM_ADDR_WIDTH, TEST_RAM_DATA_WIDTH, TEST_RAM_NUM_PARTITIONS>;
    type RamWriteResp = ram::WriteResp;
    type RamAddr = bits[TEST_RAM_ADDR_WIDTH];
    type RamData = bits[TEST_RAM_DATA_WIDTH];
    type State = RamWrRespHandlerTestState;
    type CntWidth = bits[TEST_CNT_WIDTH];

    terminator: chan<bool> out;
    rd_req_s: chan<RamReadReq> out;
    rd_resp_r: chan<RamReadResp> in;
    wr_req_s: chan<RamWriteReq> out;
    wr_resp_r: chan<RamWriteResp> in;
    resp_in_s: chan<bool> out;
    resp_out_r: chan<CntWidth> in;

    config(terminator: chan<bool> out) {
        let (rd_req_s, rd_req_r) = chan<RamReadReq>("rd_req");
        let (rd_resp_s, rd_resp_r) = chan<RamReadResp>("rd_resp");
        let (wr_req_s, wr_req_r) = chan<RamWriteReq>("wr_req");
        let (wr_resp_s, wr_resp_r) = chan<RamWriteResp>("wr_resp");

        let (resp_in_s, resp_in_r) = chan<bool>("resp_in");
        let (resp_out_s, resp_out_r) = chan<u32>("resp_out");

        spawn RamWrRespHandler<TEST_CNT_WIDTH>(resp_in_r, resp_out_s, wr_resp_r);

        spawn ram::RamModel<TEST_RAM_DATA_WIDTH, TEST_RAM_SIZE, TEST_RAM_WORD_PARTITION_SIZE>(
            rd_req_r, rd_resp_s, wr_req_r, wr_resp_s);

        (terminator, rd_req_s, rd_resp_r, wr_req_s, wr_resp_r, resp_in_s, resp_out_r)
    }

    init {
        type State = RamWrRespHandlerTestState;
        zero!<State>()
    }

    next(tok0: token, state: State) {
        let start = (state.rd_cnt == u32:0);
        const MASK = std::unsigned_max_value<TEST_RAM_NUM_PARTITIONS>();

        let (tok1, wr_cnt, _) = recv_non_blocking(tok0, resp_out_r, state.wr_cnt);

        let do_send_ram = state.rd_cnt < TEST_SYMBOLS_TO_SEND;
        let wr_req = RamWriteReq {
            addr: state.rd_cnt as RamAddr,
            data: state.rd_cnt as RamData,
            mask: MASK
        };

        let tok2_0 = send_if(tok1, wr_req_s, do_send_ram, wr_req);
        let tok2_1 = send_if(tok1, resp_in_s, do_send_ram, start);

        let do_terminate = (state.wr_cnt == (TEST_SYMBOLS_TO_SEND - u32:1));
        let tok2_2 = send_if(tok1, terminator, do_terminate, true);

        if do_terminate {
            zero!<State>()
        } else {
            let rd_cnt = state.rd_cnt + u32:1;
            State { rd_cnt, wr_cnt }
        }
    }
}
