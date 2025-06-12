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

// The proc should just pass incoming data as literals to LiteralsBuffer.
// Packets of 0 length are not passed further and a warning is log instead.

import xls.modules.zstd.common;
import xls.modules.zstd.memory.mem_reader as mem_reader;

type LiteralsDataWithSync = common::LiteralsDataWithSync;
type LitData = common::LitData;
type LitLength = common::LitLength;
type LitID = common::LitID;

pub struct RawLiteralsDecoderReq<ADDR_W: u32> {
    id: u32,
    addr: uN[ADDR_W],
    length: uN[ADDR_W],
    literals_last: bool,
}

pub enum RawLiteralsDecoderStatus: u1 {
    OKAY = 0,
    ERROR = 1,
}

pub struct RawLiteralsDecoderResp {
    status: RawLiteralsDecoderStatus,
}

struct RawLiteralsDecoderState {
    id: u32,
    literals_last: bool,
}

pub proc RawLiteralsDecoder<DATA_W: u32, ADDR_W: u32> {
    type Req = RawLiteralsDecoderReq<ADDR_W>;
    type Resp = RawLiteralsDecoderResp;
    type Output = LiteralsDataWithSync;
    type State = RawLiteralsDecoderState;
    type Status = RawLiteralsDecoderStatus;
    type MemReaderReq = mem_reader::MemReaderReq<ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<DATA_W, ADDR_W>;
    type MemReaderStatus = mem_reader::MemReaderStatus;

    // decoder input
    req_r: chan<Req> in;
    resp_s: chan<Resp> out;

    // decoder output
    output_s: chan<Output> out;

    // memory interface
    mem_req_s: chan<MemReaderReq> out;
    mem_resp_r: chan<MemReaderResp> in;

    init { zero!<State>() }

    config(
        req_r: chan<Req> in,
        resp_s: chan<Resp> out,
        output_s: chan<Output> out,

        mem_req_s: chan<MemReaderReq> out,
        mem_resp_r: chan<MemReaderResp> in,
    ) {
        (
            req_r, resp_s, output_s,
            mem_req_s, mem_resp_r,
        )
    }

    next(state: State) {
        let tok0 = join();

        // receive request
        let (tok1_0, req, req_valid) = recv_non_blocking(tok0, req_r, zero!<Req>());

        // update ID and last in state
        let state = if req_valid {
            State { id: req.id, literals_last: req.literals_last}
        } else { state };

        // send memory read request
        let req = MemReaderReq { addr: req.addr, length: req.length };
        let tok2_0 = send_if(tok1_0, mem_req_s, req_valid, req);

        // receive memory read response
        let (tok1_1, mem_resp, mem_resp_valid) = recv_non_blocking(tok0, mem_resp_r, zero!<MemReaderResp>());
        let mem_resp_error = (mem_resp.status != MemReaderStatus::OKAY);

        // prepare output data, decoded RAW block is always a literal
        let output_data = Output {
            last: mem_resp.last,
            literals_last: state.literals_last,
            id: state.id as LitID,
            data: checked_cast<LitData>(mem_resp.data),
            length: checked_cast<LitLength>(mem_resp.length),
        };

        // send output data
        let mem_resp_correct = mem_resp_valid && !mem_resp_error;
        let tok2_1 = send_if(tok1_1, output_s, mem_resp_correct, output_data);

        // send response after block end
        let resp = if mem_resp_correct {
            Resp { status: Status::OKAY }
        } else {
            Resp { status: Status::ERROR }
        };

        let do_send_resp = mem_resp_valid && mem_resp.last;
        let tok2_2 = send_if(tok1_1, resp_s, do_send_resp, resp);

        state
    }
}

const INST_DATA_W = u32:64;
const INST_ADDR_W = u32:16;

pub proc RawLiteralsDecoderInst {
    type Req = RawLiteralsDecoderReq<INST_ADDR_W>;
    type Resp = RawLiteralsDecoderResp;
    type Output = LiteralsDataWithSync;
    type MemReaderReq = mem_reader::MemReaderReq<INST_ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<INST_DATA_W, INST_ADDR_W>;

    init { () }

    config(
        req_r: chan<Req> in,
        resp_s: chan<Resp> out,
        output_s: chan<Output> out,

        mem_req_s: chan<MemReaderReq> out,
        mem_resp_r: chan<MemReaderResp> in,
    ) {
        spawn RawLiteralsDecoder<INST_DATA_W, INST_ADDR_W>(
            req_r, resp_s, output_s, mem_req_s, mem_resp_r
        );
    }

    next(state: ()) {}
}

const TEST_DATA_W = u32:64;
const TEST_ADDR_W = u32:16;

#[test_proc]
proc RawLiteralsDecoderTest {
    type Req = RawLiteralsDecoderReq<TEST_ADDR_W>;
    type Resp = RawLiteralsDecoderResp;
    type Output = LiteralsDataWithSync;
    type State = RawLiteralsDecoderState;
    type Status = RawLiteralsDecoderStatus;
    type MemReaderReq = mem_reader::MemReaderReq<TEST_ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<TEST_DATA_W, TEST_ADDR_W>;
    type MemReaderStatus = mem_reader::MemReaderStatus;

    type Data = uN[TEST_DATA_W];
    type Addr = uN[TEST_ADDR_W];
    type Length = uN[TEST_ADDR_W];

    terminator: chan<bool> out;
    // decoder input
    req_s: chan<Req> out;
    resp_r: chan<Resp> in;

    // decoder output
    output_r: chan<Output> in;

    // memory interface
    mem_req_r: chan<MemReaderReq> in;
    mem_resp_s: chan<MemReaderResp> out;

    config(terminator: chan<bool> out) {
        let (req_s, req_r) = chan<Req>("req");
        let (resp_s, resp_r) = chan<Resp>("resp");
        let (output_s, output_r) = chan<Output>("output");

        let (mem_req_s, mem_req_r) = chan<MemReaderReq>("mem_req");
        let (mem_resp_s, mem_resp_r) = chan<MemReaderResp>("mem_resp");

        spawn RawLiteralsDecoder<TEST_DATA_W, TEST_ADDR_W>(
            req_r, resp_s, output_s, mem_req_s, mem_resp_r
        );

        (terminator, req_s, resp_r, output_r, mem_req_r, mem_resp_s)
    }

    init {  }

    next(state: ()) {

        let tok = join();

        // Test 0
        let req = Req { id: u32:0, literals_last: false, addr: Addr:0, length: Length:8 };
        let tok = send(tok, req_s, req);

        let (tok, mem_req) = recv(tok, mem_req_r);
        assert_eq(mem_req, MemReaderReq { addr: Addr:0, length: Length:8 });

        let mem_resp = MemReaderResp {
            status: mem_reader::MemReaderStatus::OKAY,
            data: Data:0x1122_3344,
            length: Length:8,
            last: true,
        };
        let tok = send(tok, mem_resp_s, mem_resp);
        let (tok, output) = recv(tok, output_r);
        assert_eq(output, Output {
            last: true,
            literals_last: false,
            id: u32:0,
            data: Data:0x1122_3344,
            length: LitLength:8,
        });

        // Test 1
        let req = Req { id: u32:1, literals_last: true, addr: Addr:0x1001, length: Length:15 };
        let tok = send(tok, req_s, req);

        let (tok, mem_req) = recv(tok, mem_req_r);
        assert_eq(mem_req, MemReaderReq { addr: Addr:0x1001, length: Length:15 });

        let mem_resp = MemReaderResp {
            status: mem_reader::MemReaderStatus::OKAY,
            data: Data:0x1122_3344_5566_7788,
            length: Length:8,
            last: false
        };
        let tok = send(tok, mem_resp_s, mem_resp);

        let mem_resp = MemReaderResp {
            status: mem_reader::MemReaderStatus::OKAY,
            data: Data:0xAA_BBCC_DDEE_FF99,
            length: Length:7,
            last: true,
        };
        let tok = send(tok, mem_resp_s, mem_resp);

        let (tok, output) = recv(tok, output_r);
        assert_eq(output, Output {
            last: false,
            literals_last: true,
            id: u32:1,
            data: Data:0x1122_3344_5566_7788,
            length: LitLength:8,
        });

        let (tok, output) = recv(tok, output_r);
        assert_eq(output, Output {
            last: true,
            literals_last: true,
            id: u32:1,
            data: Data:0xAA_BBCC_DDEE_FF99,
            length: LitLength:7,
        });

        // Test 2
        let req = Req {id: u32:2, literals_last: false, addr: Addr:0x2000, length: Length:0 };
        let tok = send(tok, req_s, req);

        let (tok, mem_req) = recv(tok, mem_req_r);
        assert_eq(mem_req, MemReaderReq { addr: Addr:0x2000, length: Length:0 });

        let mem_resp = MemReaderResp {
            status: mem_reader::MemReaderStatus::OKAY,
            data: Data:0x0,
            length: Length:0,
            last: true,
        };
        let tok = send(tok, mem_resp_s, mem_resp);
        let (tok, output) = recv(tok, output_r);
        assert_eq(output, Output {
            last: true,
            literals_last: false,
            id: u32:2,
            data: Data:0x0,
            length: LitLength:0,
        });

        send(tok, terminator, true);
    }
}
