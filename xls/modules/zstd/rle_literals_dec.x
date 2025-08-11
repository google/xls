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

// This file contains the implementation of RleLiteralsDecoder responsible for decoding
// ZSTD RLE Literals. More information about Rle Literals's format can be found in:
// https://datatracker.ietf.org/doc/html/rfc8878#section-3.1.1.3.1

import std;

import xls.modules.zstd.common;

type LiteralsDataWithSync = common::LiteralsDataWithSync;
type RleLitData = common::RleLitData;
type RleLitRepeat = common::RleLitRepeat;
type LitData = common::LitData;
type LitID = common::LitID;
type LitLength = common::LitLength;

pub enum RleLiteralsDecoderStatus: u1 {
    OKAY = 0,
}

pub struct RleLiteralsDecoderReq {
    id: u32,
    symbol: u8,
    length: RleLitRepeat,
    literals_last: bool,
}

pub struct RleLiteralsDecoderResp {
    status: RleLiteralsDecoderStatus
}

struct RleLiteralsDecoderState {
    req: RleLiteralsDecoderReq,
    req_valid: bool,
}

pub proc RleLiteralsDecoder<DATA_W: u32> {
    type Req = RleLiteralsDecoderReq;
    type Resp = RleLiteralsDecoderResp;
    type Output = LiteralsDataWithSync;

    type State = RleLiteralsDecoderState;

    req_r: chan<Req> in;
    resp_s: chan<Resp> out;
    output_s: chan<Output> out;

    config( req_r: chan<Req> in,
        resp_s: chan<Resp> out,
        output_s: chan<Output> out,
    ) { (req_r, resp_s, output_s) }

    init { zero!<State>() }

    next(state: State) {
        const MAX_OUTPUT_SYMBOLS = (DATA_W / u32:8);
        const MAX_LEN = MAX_OUTPUT_SYMBOLS as RleLitRepeat;

        let tok0 = join();

        let (tok1, req) = recv_if(tok0, req_r, !state.req_valid, state.req);

        let last = req.length <= MAX_LEN;
        let length = if last { req.length } else { MAX_LEN };
        let data = unroll_for! (i, data): (u32, uN[DATA_W]) in u32:0..MAX_OUTPUT_SYMBOLS {
            bit_slice_update(data, i * u32:8, req.symbol)
        }(uN[DATA_W]:0);

        let output = Output {
            last: last,
            literals_last: req.literals_last,
            id: req.id,
            data: checked_cast<LitData>(data),
            length: checked_cast<LitLength>(length),
        };

        send_if(tok1, resp_s, last, zero!<Resp>());
        send(tok1, output_s, output);

        if last {
            zero!<State>()
        } else {
            let length = req.length - MAX_LEN;
            State {
                req: Req { length, ..req },
                req_valid: true,
            }
        }
    }
}

const INST_DATA_W = u32:64;

pub proc RleLiteralsDecoderInst {
    type Req = RleLiteralsDecoderReq;
    type Resp = RleLiteralsDecoderResp;
    type Output = LiteralsDataWithSync;

    config(
        req_r: chan<Req> in,
        resp_s: chan<Resp> out,
        output_s: chan<Output> out,
    ) {
        spawn RleLiteralsDecoder<INST_DATA_W>(
            req_r, resp_s, output_s
        );
    }

    init { () }

    next(state: ()) {}
}

const TEST_DATA_W = u32:64;

#[test_proc]
proc RleLiteralsDecoder_test {
    type Req = RleLiteralsDecoderReq;
    type Resp = RleLiteralsDecoderResp;
    type Output = LiteralsDataWithSync;
    type Status = RleLiteralsDecoderStatus;

    terminator: chan<bool> out;
    req_s: chan<Req> out;
    resp_r: chan<Resp> in;
    out_r: chan<Output> in;

    config (terminator: chan<bool> out) {
        let (req_s, req_r) = chan<Req>("req");
        let (resp_s, resp_r) = chan<Resp>("resp");
        let (out_s, out_r) = chan<Output>("output");

        spawn RleLiteralsDecoder<TEST_DATA_W>(
            req_r, resp_s, out_s
        );

        (terminator, req_s, resp_r, out_r)
    }

    init { }

    next(state: ()) {
        let tok = join();
        let test_rle_req: Req[6] = [
            Req {symbol: RleLitData:0x11, length: RleLitRepeat:11, id: LitID:0, literals_last: false},
            Req {symbol: RleLitData:0x22, length: RleLitRepeat:3, id: LitID:1, literals_last: false},
            Req {symbol: RleLitData:0x33, length: RleLitRepeat:16, id: LitID:2, literals_last: false},
            Req {symbol: RleLitData:0x55, length: RleLitRepeat:2, id: LitID:3, literals_last: false},
            Req {symbol: RleLitData:0x66, length: RleLitRepeat:20, id: LitID:4, literals_last: false},
            Req {symbol: RleLitData:0x00, length: RleLitRepeat:0, id: LitID:5, literals_last: true},
        ];
        let test_rle_resp: Resp[6] = [
            Resp {status: Status::OKAY},
            Resp {status: Status::OKAY},
            Resp {status: Status::OKAY},
            Resp {status: Status::OKAY},
            Resp {status: Status::OKAY},
            Resp {status: Status::OKAY},
        ];

        let test_out_data: LiteralsDataWithSync[10] = [
            // 1st literal
            LiteralsDataWithSync {data: LitData:0x1111_1111_1111_1111, length: LitLength:8, id: LitID:0, last: false, literals_last: false},
            LiteralsDataWithSync {data: LitData:0x1111_1111_1111_1111, length: LitLength:3, id: LitID:0, last: true, literals_last: false},
            // 2nd literal
            LiteralsDataWithSync {data: LitData:0x2222_2222_2222_2222, length: LitLength:3, id: LitID:1, last: true, literals_last: false},
            // 3rd literal
            LiteralsDataWithSync {data: LitData:0x3333_3333_3333_3333, length: LitLength:8, id: LitID:2, last: false, literals_last: false},
            LiteralsDataWithSync {data: LitData:0x3333_3333_3333_3333, length: LitLength:8, id: LitID:2, last: true, literals_last: false},
            // 5th literal
            LiteralsDataWithSync {data: LitData:0x5555_5555_5555_5555, length: LitLength:2, id: LitID:3, last: true, literals_last: false},
            // 6th literal
            LiteralsDataWithSync {data: LitData:0x6666_6666_6666_6666, length: LitLength:8, id: LitID:4, last: false, literals_last: false},
            LiteralsDataWithSync {data: LitData:0x6666_6666_6666_6666, length: LitLength:8, id: LitID:4, last: false, literals_last: false},
            LiteralsDataWithSync {data: LitData:0x6666_6666_6666_6666, length: LitLength:4, id: LitID:4, last: true, literals_last: false},
            // 7th literal
            LiteralsDataWithSync {data: LitData:0x0000_0000_0000_0000, length: LitLength:0, id: LitID:5, last: true, literals_last: true},
        ];

        // Test #0
        let req = test_rle_req[0];
        let resp = test_rle_resp[0];
        let tok = send(tok, req_s, req);
        trace_fmt!("Sent req: {:#x}", req);
        let (tok, out_data) = recv(tok, out_r);
        trace_fmt!("Received batched data: {:#x}", out_data);
        assert_eq(out_data, test_out_data[0]);
        let (tok, out_data) = recv(tok, out_r);
        trace_fmt!("Received batched data: {:#x}", out_data);
        assert_eq(out_data, test_out_data[1]);
        let (tok, rle_resp) = recv(tok, resp_r);
        trace_fmt!("Received resp: {:#x}", rle_resp);
        assert_eq(rle_resp, resp);

        // Test #1
        let req = test_rle_req[1];
        let resp = test_rle_resp[1];
        let tok = send(tok, req_s, req);
        trace_fmt!("Sent req: {:#x}", req);
        let (tok, out_data) = recv(tok, out_r);
        trace_fmt!("Received batched data: {:#x}", out_data);
        assert_eq(out_data, test_out_data[2]);
        let (tok, rle_resp) = recv(tok, resp_r);
        trace_fmt!("Received resp: {:#x}", rle_resp);
        assert_eq(rle_resp, resp);

        // Test #2
        let req = test_rle_req[2];
        let resp = test_rle_resp[2];
        let tok = send(tok, req_s, req);
        trace_fmt!("Sent req: {:#x}", req);
        let (tok, out_data) = recv(tok, out_r);
        trace_fmt!("Received batched data: {:#x}", out_data);
        assert_eq(out_data, test_out_data[3]);
        let (tok, out_data) = recv(tok, out_r);
        trace_fmt!("Received batched data: {:#x}", out_data);
        assert_eq(out_data, test_out_data[4]);
        let (tok, rle_resp) = recv(tok, resp_r);
        trace_fmt!("Received resp: {:#x}", rle_resp);
        assert_eq(rle_resp, resp);

        // Test #3
        let req = test_rle_req[3];
        let resp = test_rle_resp[3];
        let tok = send(tok, req_s, req);
        trace_fmt!("Sent req: {:#x}", req);
        let (tok, out_data) = recv(tok, out_r);
        trace_fmt!("Received batched data: {:#x}", out_data);
        assert_eq(out_data, test_out_data[5]);
        let (tok, rle_resp) = recv(tok, resp_r);
        trace_fmt!("Received resp: {:#x}", rle_resp);
        assert_eq(rle_resp, resp);

        // Test #4
        let req = test_rle_req[4];
        let resp = test_rle_resp[4];
        let tok = send(tok, req_s, req);
        trace_fmt!("Sent req: {:#x}", req);
        let (tok, out_data) = recv(tok, out_r);
        trace_fmt!("Received batched data: {:#x}", out_data);
        assert_eq(out_data, test_out_data[6]);
        let (tok, out_data) = recv(tok, out_r);
        trace_fmt!("Received batched data: {:#x}", out_data);
        assert_eq(out_data, test_out_data[7]);
        let (tok, out_data) = recv(tok, out_r);
        trace_fmt!("Received batched data: {:#x}", out_data);
        assert_eq(out_data, test_out_data[8]);
        let (tok, rle_resp) = recv(tok, resp_r);
        trace_fmt!("Received resp: {:#x}", rle_resp);
        assert_eq(rle_resp, resp);

        // Test #5
        let req = test_rle_req[5];
        let resp = test_rle_resp[5];
        let tok = send(tok, req_s, req);
        trace_fmt!("Sent req: {:#x}", req);
        let (tok, out_data) = recv(tok, out_r);
        trace_fmt!("Received batched data: {:#x}", out_data);
        assert_eq(out_data, test_out_data[9]);
        let (tok, rle_resp) = recv(tok, resp_r);
        trace_fmt!("Received resp: {:#x}", rle_resp);
        assert_eq(rle_resp, resp);

        send(tok, terminator, true);
    }
}
