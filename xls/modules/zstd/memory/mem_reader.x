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

import std;

import xls.modules.zstd.memory.axi;
import xls.modules.zstd.memory.axi_st;

import xls.modules.zstd.memory.axi_reader;
import xls.modules.zstd.memory.axi_stream_downscaler;
import xls.modules.zstd.memory.axi_stream_remove_empty;


// This module provides the MemReader and MemReaderAdv procs for handling
// read transactions on the AXI bus. Both readers spawn helper components
// that simplifies interactions with them for other procs.
// Use MemReader when the data width on both the DSLX and AXI sides
// is the same. For cases where the AXI width is a multiple of the DSLX width,
// use MemReaderAdv. Other configurations are not supported.

// Enum containing information about the status of the response
pub enum MemReaderStatus : u1 {
    OKAY = 0,
    ERROR = 1,
}

// Request that can be submited to MemReader to read data from an AXI bus
pub struct MemReaderReq<DSLX_ADDR_W: u32> {
    addr: uN[DSLX_ADDR_W],   //
    length: uN[DSLX_ADDR_W]  //
}

// Response received rom the MemReader proc
pub struct MemReaderResp<DSLX_DATA_W: u32, DSLX_ADDR_W: u32> {
    status: MemReaderStatus,  // status of the request
    data: uN[DSLX_DATA_W],    // data read from the AXI bus
    length: uN[DSLX_ADDR_W],  // length of the data in bytes
    last: bool,               // if this is the last packet to expect as a response
}

enum MemReaderFsm : u3 {
    REQUEST = 0,
    RESPONSE = 1,
    RESPONSE_ZERO = 2,
    RESPONSE_ERROR = 3,
}

struct MemReaderState<AXI_ADDR_W: u32, DSLX_ADDR_W: u32> {
    fsm: MemReaderFsm,
    error: bool,
    base: uN[AXI_ADDR_W],
}

// A proc implementing the logic for issuing requests to AxiReader,
// receiving the data, and convering the data to the specified output format.
proc MemReaderInternal<
    // DSLX side parameters
    DSLX_DATA_W: u32, DSLX_ADDR_W: u32,
    // AXI side parameters
    AXI_DATA_W: u32, AXI_ADDR_W: u32, AXI_DEST_W: u32, AXI_ID_W: u32,
    // parameters calculated from other values
    DSLX_DATA_W_DIV8: u32 = {DSLX_DATA_W / u32:8},
    AXI_DATA_W_DIV8: u32 = {AXI_DATA_W / u32:8},
    AXI_TO_DSLX_RATIO: u32 = {AXI_DATA_W / DSLX_DATA_W},
    AXI_TO_DSLX_RATIO_W: u32 = {std::clog2((AXI_DATA_W / DSLX_DATA_W) + u32:1)}
> {
    type Req = MemReaderReq<DSLX_ADDR_W>;
    type Resp = MemReaderResp<DSLX_DATA_W, DSLX_ADDR_W>;
    type Length = uN[DSLX_ADDR_W];

    type AxiReaderReq = axi_reader::AxiReaderReq<AXI_ADDR_W>;
    type AxiReaderError = axi_reader::AxiReaderError;
    type AxiAr = axi::AxiAr<AXI_ADDR_W, AXI_ID_W>;
    type AxiR = axi::AxiR<AXI_DATA_W, AXI_ID_W>;
    type AxiStreamInput = axi_st::AxiStream<AXI_DATA_W, AXI_DEST_W, AXI_ID_W, AXI_DATA_W_DIV8>;
    type AxiStreamOutput = axi_st::AxiStream<DSLX_DATA_W, AXI_DEST_W, AXI_ID_W, DSLX_DATA_W_DIV8>;

    type State = MemReaderState<AXI_ADDR_W, DSLX_ADDR_W>;
    type Fsm = MemReaderFsm;

    // Assumptions related to parameters
    const_assert!(DSLX_DATA_W % u32:8 == u32:0);       // DSLX-side data width should be divisible by 8
    const_assert!(AXI_DATA_W % u32:8 == u32:0);        // AXI-side data width should be divisible by 8
    const_assert!(AXI_DATA_W >= DSLX_DATA_W);          // AXI-side width should be wider or has the same width as DSLX-side
    const_assert!(AXI_DATA_W % DSLX_DATA_W == u32:0);  // DSLX-side width should be a multiple of AXI-side width

    // checks for parameters
    const_assert!(DSLX_DATA_W_DIV8 == DSLX_DATA_W / u32:8);
    const_assert!(AXI_DATA_W_DIV8 == AXI_DATA_W / u32:8);
    const_assert!(AXI_TO_DSLX_RATIO == AXI_DATA_W / DSLX_DATA_W);
    const_assert!(AXI_TO_DSLX_RATIO_W == std::clog2((AXI_DATA_W / DSLX_DATA_W) + u32:1));

    req_r: chan<Req> in;
    resp_s: chan<Resp> out;

    reader_req_s: chan<AxiReaderReq> out;
    reader_err_r: chan<AxiReaderError> in;

    axi_st_out_r: chan<AxiStreamOutput> in;

    config(
        req_r: chan<Req> in,
        resp_s: chan<Resp> out,
        reader_req_s: chan<AxiReaderReq> out,
        reader_err_r: chan<AxiReaderError> in,
        axi_st_out_r: chan<AxiStreamOutput> in,
    ) {
        (req_r, resp_s, reader_req_s, reader_err_r, axi_st_out_r)
    }

    init { zero!<State>() }

    next(state: State) {
        type Resp = MemReaderResp<DSLX_DATA_W, DSLX_ADDR_W>;
        type DslxData = uN[DSLX_DATA_W];
        type DslxLength = uN[DSLX_ADDR_W];
        type AxiLength = uN[AXI_ADDR_W];
        type AxiStr = uN[AXI_DATA_W_DIV8];
        type AxiKeep = uN[AXI_DATA_W_DIV8];
        type Status = MemReaderStatus;

        const READER_RESP_ERROR = Resp { status: Status::ERROR, ..zero!<Resp>() };
        const READER_RESP_ZERO = Resp { status: Status::OKAY, last: true, ..zero!<Resp>() };

        let tok0 = join();
        let (tok, error_info, error) =
            recv_non_blocking(tok0, reader_err_r, zero!<AxiReaderError>());

        // Request
        let do_handle_req = !error && (state.fsm == Fsm::REQUEST);
        let (tok, req) = recv_if(tok0, req_r, do_handle_req, zero!<Req>());
        let is_zero_len = (req.length == uN[DSLX_ADDR_W]:0);

        let reader_req = axi_reader::AxiReaderReq {
            addr: req.addr,
            len: req.length
        };
        let do_send_reader_req = !error && !is_zero_len && (state.fsm == Fsm::REQUEST);
        let tok = send_if(tok0, reader_req_s, do_send_reader_req, reader_req);

        let do_handle_resp = !error && (state.fsm == Fsm::RESPONSE);
        let (tok, st) = recv_if(tok0, axi_st_out_r, do_handle_resp, zero!<AxiStreamOutput>());

        let length = std::popcount(st.str | st.keep) as Length;
        let reader_resp_ok = Resp { status: Status::OKAY, data: st.data, length, last: st.last };

        let reader_resp = if state.fsm == Fsm::RESPONSE_ERROR {
            READER_RESP_ERROR
        } else if state.fsm == Fsm::RESPONSE_ZERO {
            READER_RESP_ZERO
        } else {
            reader_resp_ok
        };

        let do_send_resp = do_handle_resp ||
                           (state.fsm == Fsm::RESPONSE_ERROR) ||
                           (state.fsm == Fsm::RESPONSE_ZERO);
        let tok = send_if(tok0, resp_s, do_send_resp, reader_resp);

        let next_state = match (state.fsm) {
            Fsm::REQUEST => {
                if error {
                    State { fsm: Fsm::RESPONSE_ERROR, ..zero!<State>() }
                } else if is_zero_len {
                    State { fsm: Fsm::RESPONSE_ZERO, ..state }
                } else {
                    State { fsm: Fsm::RESPONSE, ..state }
                }
            },
            Fsm::RESPONSE => {
                if error {
                    State { fsm: Fsm::RESPONSE_ERROR, ..zero!<State>() }
                } else if st.last {
                    State { fsm: Fsm::REQUEST, ..state }
                } else {
                    State { fsm: Fsm::RESPONSE, ..state }
                }
            },
            Fsm::RESPONSE_ZERO => {
                if error {
                    State { fsm: Fsm::RESPONSE_ERROR, ..zero!<State>() }
                } else {
                    State { fsm: Fsm::REQUEST, ..state }
                }
            },
            Fsm::RESPONSE_ERROR => {
                if error {
                    State { fsm: Fsm::RESPONSE_ERROR, ..zero!<State>() }
                } else {
                    State { fsm: Fsm::REQUEST, ..state }
                }
            },
        };

        next_state
    }
}

// A proc that integrates other procs to create a functional design for
// performing AXI read transactions. It allows for connecting narrow DSLX-side
// with wider AXI-side, if the wider side has to be a multiple of the narrower side.
pub proc MemReaderAdv<
    // DSLX side parameters
    DSLX_DATA_W: u32, DSLX_ADDR_W: u32,
    // AXI side parameters
    AXI_DATA_W: u32, AXI_ADDR_W: u32, AXI_DEST_W: u32, AXI_ID_W: u32,
    // parameters calculated from other values
    DSLX_DATA_W_DIV8: u32 = {DSLX_DATA_W / u32:8},
    AXI_DATA_W_DIV8: u32 = {AXI_DATA_W / u32:8},
    AXI_TO_DSLX_RATIO: u32 = {AXI_DATA_W / DSLX_DATA_W},
    AXI_TO_DSLX_RATIO_W: u32 = {std::clog2((AXI_DATA_W / DSLX_DATA_W) + u32:1)}
> {
    type Req = MemReaderReq<DSLX_ADDR_W>;
    type Resp = MemReaderResp<DSLX_DATA_W, DSLX_ADDR_W>;

    type AxiReaderReq = axi_reader::AxiReaderReq<AXI_ADDR_W>;
    type AxiR = axi::AxiR<AXI_DATA_W, AXI_ID_W>;
    type AxiAr = axi::AxiAr<AXI_ADDR_W, AXI_ID_W>;
    type AxiStreamInput = axi_st::AxiStream<AXI_DATA_W, AXI_DEST_W, AXI_ID_W, AXI_DATA_W_DIV8>;
    type AxiStreamOutput = axi_st::AxiStream<DSLX_DATA_W, AXI_DEST_W, AXI_ID_W, DSLX_DATA_W_DIV8>;
    type AxiReaderError = axi_reader::AxiReaderError;

    config(
        req_r: chan<Req> in,
        resp_s: chan<Resp> out,
        axi_ar_s: chan<AxiAr> out,
        axi_r_r: chan<AxiR> in
    ) {
        let (reader_req_s, reader_req_r) = chan<AxiReaderReq, u32:0>("reader_req");
        let (reader_err_s, reader_err_r) = chan<AxiReaderError, u32:0>("reader_err");

        let (axi_st_in_s, axi_st_in_r) = chan<AxiStreamInput, u32:0>("axi_st_in");
        let (axi_st_remove_s, axi_st_remove_r) = chan<AxiStreamOutput, u32:0>("axi_st_remove");
        let (axi_st_out_s, axi_st_out_r) = chan<AxiStreamOutput, u32:0>("axi_st_out");

        spawn MemReaderInternal<
            DSLX_DATA_W, DSLX_ADDR_W,
            AXI_DATA_W, AXI_ADDR_W, AXI_DEST_W, AXI_ID_W,
        >(req_r, resp_s, reader_req_s, reader_err_r, axi_st_out_r);

        spawn axi_reader::AxiReader<
            AXI_ADDR_W, AXI_DATA_W, AXI_DEST_W, AXI_ID_W
        >(reader_req_r, axi_ar_s, axi_r_r, axi_st_in_s, reader_err_s);

        spawn axi_stream_downscaler::AxiStreamDownscaler<
            AXI_DATA_W, DSLX_DATA_W, AXI_DEST_W, AXI_ID_W
        >(axi_st_in_r, axi_st_remove_s);

        spawn axi_stream_remove_empty::AxiStreamRemoveEmpty<
            DSLX_DATA_W, AXI_DEST_W, AXI_ID_W
        >(axi_st_remove_r, axi_st_out_s);

        ()
    }

    init { }
    next(state: ()) { }
}

// A proc that integrates other procs to create a functional design for
// performing AXI read transactions. The proc allows for interfacing with
// AXI bus that has the same data width as DSLX-side of the design.
pub proc MemReader<
    DATA_W: u32, ADDR_W: u32, DEST_W: u32, ID_W: u32,
    CHANNEL_DEPTH: u32 = {u32:0},
    DATA_W_DIV8: u32 = {DATA_W / u32:8},
> {
    type Req = MemReaderReq<ADDR_W>;
    type Resp = MemReaderResp<DATA_W, ADDR_W>;

    type AxiReaderReq = axi_reader::AxiReaderReq<ADDR_W>;
    type AxiR = axi::AxiR<DATA_W, ID_W>;
    type AxiAr = axi::AxiAr<ADDR_W, ID_W>;
    type AxiStreamInput = axi_st::AxiStream<DATA_W, DEST_W, ID_W, DATA_W_DIV8>;
    type AxiStreamOutput = axi_st::AxiStream<DATA_W, DEST_W, ID_W, DATA_W_DIV8>;
    type AxiReaderError = axi_reader::AxiReaderError;

    config(
        req_r: chan<Req> in,
        resp_s: chan<Resp> out,
        axi_ar_s: chan<AxiAr> out,
        axi_r_r: chan<AxiR> in
    ) {
        let (reader_req_s, reader_req_r) = chan<AxiReaderReq, CHANNEL_DEPTH>("reader_req");
        let (reader_err_s, reader_err_r) = chan<AxiReaderError, CHANNEL_DEPTH>("reader_err");

        let (axi_st_in_s, axi_st_in_r) = chan<AxiStreamInput, CHANNEL_DEPTH>("axi_st_in");
        let (axi_st_out_s, axi_st_out_r) = chan<AxiStreamOutput, CHANNEL_DEPTH>("axi_st_out");

        spawn MemReaderInternal<
            DATA_W, ADDR_W, DATA_W, ADDR_W, DEST_W, ID_W
        >(req_r, resp_s, reader_req_s, reader_err_r, axi_st_out_r);

        spawn axi_reader::AxiReader<
            ADDR_W, DATA_W, DEST_W, ID_W
        >(reader_req_r, axi_ar_s, axi_r_r, axi_st_in_s, reader_err_s);

        spawn axi_stream_remove_empty::AxiStreamRemoveEmpty<
            DATA_W, DEST_W, ID_W
        >(axi_st_in_r, axi_st_out_s);
        ()
    }

    init { }
    next(state: ()) { }
}


const INST_ADV_AXI_DATA_W = u32:128;
const INST_ADV_AXI_ADDR_W = u32:16;
const INST_ADV_AXI_DEST_W = u32:8;
const INST_ADV_AXI_ID_W = u32:8;
const INST_ADV_AXI_DATA_W_DIV8 = INST_ADV_AXI_DATA_W / u32:8;

const INST_ADV_DSLX_ADDR_W = u32:16;
const INST_ADV_DSLX_DATA_W = u32:64;
const INST_ADV_DSLX_DATA_W_DIV8 = INST_ADV_DSLX_DATA_W / u32:8;

proc MemReaderAdvInst {
    type Req = MemReaderReq<INST_ADV_DSLX_ADDR_W>;
    type Resp = MemReaderResp<INST_ADV_DSLX_DATA_W, INST_ADV_DSLX_ADDR_W>;

    type AxiReaderReq = axi_reader::AxiReaderReq<INST_ADV_AXI_ADDR_W>;
    type AxiAr = axi::AxiAr<INST_ADV_AXI_ADDR_W, INST_ADV_AXI_ID_W>;
    type AxiR = axi::AxiR<INST_ADV_AXI_DATA_W, INST_ADV_AXI_ID_W>;
    type AxiStream = axi_st::AxiStream<INST_ADV_AXI_DATA_W, INST_ADV_AXI_DEST_W, INST_ADV_AXI_ID_W, INST_ADV_AXI_DATA_W_DIV8>;

    type State = MemReaderState<INST_ADV_AXI_ADDR_W, INST_ADV_DSLX_ADDR_W>;
    type Fsm = MemReaderFsm;

    config(
        req_r: chan<Req> in,
        resp_s: chan<Resp> out,
        axi_ar_s: chan<AxiAr> out,
        axi_r_r: chan<AxiR> in
    ) {
        spawn MemReaderAdv<
            INST_ADV_DSLX_DATA_W, INST_ADV_DSLX_ADDR_W,
            INST_ADV_AXI_DATA_W, INST_ADV_AXI_ADDR_W, INST_ADV_AXI_DEST_W, INST_ADV_AXI_ID_W,
        >(req_r, resp_s, axi_ar_s, axi_r_r);

        ()
    }

    init { }
    next(state: ()) { }
}

proc MemReaderInternalInst {
    type Req = MemReaderReq<INST_ADV_DSLX_ADDR_W>;
    type Resp = MemReaderResp<INST_ADV_DSLX_DATA_W, INST_ADV_DSLX_ADDR_W>;

    type AxiReaderReq = axi_reader::AxiReaderReq<INST_ADV_AXI_ADDR_W>;
    type AxiReaderError = axi_reader::AxiReaderError;

    type AxiAr = axi::AxiAr<INST_ADV_AXI_ADDR_W, INST_ADV_AXI_ID_W>;
    type AxiR = axi::AxiR<INST_ADV_AXI_DATA_W, INST_ADV_AXI_ID_W>;
    type AxiStreamOutput = axi_st::AxiStream<INST_ADV_DSLX_DATA_W, INST_ADV_AXI_DEST_W, INST_ADV_AXI_ID_W, INST_ADV_DSLX_DATA_W_DIV8>;

    config(
        req_r: chan<Req> in,
        resp_s: chan<Resp> out,
        reader_req_s: chan<AxiReaderReq> out,
        reader_err_r: chan<AxiReaderError> in,
        axi_st_out_r: chan<AxiStreamOutput> in
    ) {
        spawn MemReaderInternal<
            INST_ADV_DSLX_DATA_W, INST_ADV_DSLX_ADDR_W,
            INST_ADV_AXI_DATA_W, INST_ADV_AXI_ADDR_W, INST_ADV_AXI_DEST_W, INST_ADV_AXI_ID_W,
        >(req_r, resp_s, reader_req_s, reader_err_r, axi_st_out_r);
        ()
    }

    init { }
    next(state: ()) { }
}

const INST_DATA_W = u32:64;
const INST_ADDR_W = u32:16;
const INST_DEST_W = u32:8;
const INST_ID_W = u32:8;
const INST_DATA_W_DIV8 = INST_DATA_W / u32:8;


proc MemReaderInst {
    type Req = MemReaderReq<INST_ADDR_W>;
    type Resp = MemReaderResp<INST_DATA_W, INST_ADDR_W>;

    type AxiReaderReq = axi_reader::AxiReaderReq<INST_ADDR_W>;
    type AxiAr = axi::AxiAr<INST_ADDR_W, INST_ID_W>;
    type AxiR = axi::AxiR<INST_DATA_W, INST_ID_W>;
    type AxiStream = axi_st::AxiStream<INST_DATA_W, INST_DEST_W, INST_ID_W, INST_DATA_W_DIV8>;

    type State = MemReaderState<INST_ADDR_W, INST_ADDR_W>;
    type Fsm = MemReaderFsm;

    config(
        req_r: chan<Req> in,
        resp_s: chan<Resp> out,
        axi_ar_s: chan<AxiAr> out,
        axi_r_r: chan<AxiR> in
    ) {
        spawn MemReader<
            INST_DATA_W, INST_ADDR_W, INST_DEST_W, INST_ID_W,
        >(req_r, resp_s, axi_ar_s, axi_r_r);

        ()
    }

    init { }
    next(state: ()) { }
}

const TEST_AXI_DATA_W = u32:128;
const TEST_AXI_ADDR_W = u32:16;
const TEST_AXI_DEST_W = u32:8;
const TEST_AXI_ID_W = u32:8;
const TEST_AXI_DATA_W_DIV8 = TEST_AXI_DATA_W / u32:8;

const TEST_DSLX_ADDR_W = u32:16;
const TEST_DSLX_DATA_W = u32:64;
const TEST_DSLX_DATA_W_DIV8 = TEST_DSLX_DATA_W / u32:8;

#[test_proc]
proc MemReaderTest {
    type Req = MemReaderReq<TEST_DSLX_ADDR_W>;
    type Resp = MemReaderResp<TEST_DSLX_DATA_W, TEST_DSLX_ADDR_W>;
    type Fsm = MemReaderFsm;

    type AxiReaderReq = axi_reader::AxiReaderReq<TEST_AXI_ADDR_W>;
    type AxiAr = axi::AxiAr<TEST_AXI_ADDR_W, TEST_AXI_ID_W>;
    type AxiR = axi::AxiR<TEST_AXI_DATA_W, TEST_AXI_ID_W>;
    type AxiStream = axi_st::AxiStream<TEST_AXI_DATA_W, TEST_AXI_DEST_W, TEST_AXI_ID_W, TEST_AXI_DATA_W_DIV8>;

    type Addr = uN[TEST_AXI_ADDR_W];
    type Length = uN[TEST_DSLX_ADDR_W];
    type Data = uN[TEST_DSLX_DATA_W];

    type AxiAddr = uN[TEST_AXI_ADDR_W];
    type AxiBurst = axi::AxiAxBurst;
    type AxiCache = axi::AxiArCache;
    type AxiData = uN[TEST_AXI_DATA_W];
    type AxiId = uN[TEST_AXI_ID_W];
    type AxiLast = bool;
    type AxiLength = uN[8];
    type AxiProt = uN[3];
    type AxiQos = uN[4];
    type AxiRegion = uN[4];
    type AxiResp = axi::AxiReadResp;
    type AxiSize = axi::AxiAxSize;

    type Status = MemReaderStatus;

    terminator: chan<bool> out;
    req_s: chan<Req> out;
    resp_r: chan<Resp> in;
    axi_ar_r: chan<AxiAr> in;
    axi_r_s: chan<AxiR> out;

    config(terminator: chan<bool> out) {
        let (req_s, req_r) = chan<Req>("req");
        let (resp_s, resp_r) = chan<Resp>("resp");
        let (axi_ar_s, axi_ar_r) = chan<AxiAr>("axi_ar");
        let (axi_r_s, axi_r_r) = chan<AxiR>("axi_r");

        spawn MemReaderAdv<
            TEST_DSLX_DATA_W, TEST_DSLX_ADDR_W,
            TEST_AXI_DATA_W, TEST_AXI_ADDR_W, TEST_AXI_DEST_W, TEST_AXI_ID_W,
        >(req_r, resp_s, axi_ar_s, axi_r_r);
        (terminator, req_s, resp_r, axi_ar_r, axi_r_s)
    }

    init { }

    next(state: ()) {
        let tok = join();

                // empty transfers, should be just confirmed internaly
        let tok = send(tok, req_s, Req { addr: Addr:0x1100, length: Length:0x0 });
        let (tok, resp) = recv(tok, resp_r);
        assert_eq(resp, Resp {
            status: Status::OKAY,
            data: Data:0,
            length: Length:0,
            last: true
        });

        // aligned transfer shorter than full AXI-side,
        // that fits one DSLX-side width

        let tok = send(tok, req_s, Req { addr: Addr:0x1100, length: Length:0x1 });
        let (tok, ar) = recv(tok, axi_ar_r);
        assert_eq(ar, AxiAr {
            id: AxiId:0x0,
            addr: AxiAddr:0x1100,
            region: AxiRegion:0x0,
            len: AxiLength:0x0,
            size: AxiSize::MAX_16B_TRANSFER,
            burst: AxiBurst::INCR,
            cache: AxiCache::DEV_NO_BUF,
            prot: AxiProt:0x0,
            qos: AxiQos:0x0
        });

        let tok = send(tok, axi_r_s, AxiR {
            id: AxiId:0x0,
            data: AxiData:0x1122_3344_5566_7788_9900_AABB_CCDD_EEFF,
            resp: AxiResp::OKAY,
            last: AxiLast:true
        });

        let (tok, resp) = recv(tok, resp_r);
        assert_eq(resp, Resp {
            status: Status::OKAY,
            data: Data:0xFF,
            length: Length:1,
            last: true
        });

        // unaligned transfer shorter than full AXI-side,
        // that fits one DSLX-side width

        let tok = send(tok, req_s, Req { addr: Addr:0x1001, length: Length:0x1 });
        let (tok, ar) = recv(tok, axi_ar_r);
        assert_eq(ar, AxiAr {
            id: AxiId:0x0,
            addr: AxiAddr:0x1000,
            region: AxiRegion:0x0,
            len: AxiLength:0x0,
            size: AxiSize::MAX_16B_TRANSFER,
            burst: AxiBurst::INCR,
            cache: AxiCache::DEV_NO_BUF,
            prot: AxiProt:0x0,
            qos: AxiQos:0x0
        });

        let tok = send(tok, axi_r_s, AxiR {
            id: AxiId:0x0,
            data: AxiData:0x1122_3344_5566_7788_9900_AABB_CCDD_EEFF,
            resp: AxiResp::OKAY,
            last: AxiLast:true
        });

        let (tok, resp) = recv(tok, resp_r);
        assert_eq(resp, Resp {
            status: Status::OKAY,
            data: Data:0xEE,
            length: Length:1,
            last: true
        });

        // unaligned transfer shorter than full AXI-side,
        // that fits one DSLX-side width and crosess 4k boundary

        let tok = send(tok, req_s, Req { addr: Addr:0xFFF, length: Length:0x2 });
        let (tok, ar) = recv(tok, axi_ar_r);
        assert_eq(ar, AxiAr {
            id: AxiId:0x0,
            addr: AxiAddr:0xFF0,
            region: AxiRegion:0x0,
            len: AxiLength:0x0,
            size: AxiSize::MAX_16B_TRANSFER,
            burst: AxiBurst::INCR,
            cache: AxiCache::DEV_NO_BUF,
            prot: AxiProt:0x0,
            qos: AxiQos:0x0
        });

        let tok = send(tok, axi_r_s, AxiR {
            id: AxiId:0x0,
            data: AxiData:0x1122_3344_5566_7788_9900_AABB_CCDD_EE55,
            // Addresses:    ^ 0xFFF                              ^ 0xFF0
            resp: AxiResp::OKAY,
            last: AxiLast:true
        });

        let (tok, ar) = recv(tok, axi_ar_r);
        assert_eq(ar, AxiAr {
            id: AxiId:0x0,
            addr: AxiAddr:0x1000,
            region: AxiRegion:0x0,
            len: AxiLength:0x0,
            size: AxiSize::MAX_16B_TRANSFER,
            burst: AxiBurst::INCR,
            cache: AxiCache::DEV_NO_BUF,
            prot: AxiProt:0x0,
            qos: AxiQos:0x0
        });

        let tok = send(tok, axi_r_s, AxiR {
            id: AxiId:0x0,
            data: AxiData:0x5522_3344_5566_7788_9900_AABB_CCDD_EEFF,
            // Addresses:    ^ 0x100F                             ^ 0x1000
            resp: AxiResp::OKAY,
            last: AxiLast:true
        });

        let (tok, resp) = recv(tok, resp_r);
        assert_eq(resp, Resp {
            status: Status::OKAY,
            data: Data:0xFF11,
            //     0x1000 ^ ^ 0x0FFF
            length: Length:2,
            last: true
        });


        let tok = send(tok, req_s, Req { addr: Addr:0xFFF, length: Length:0x10 });
        let (tok, ar) = recv(tok, axi_ar_r);
        assert_eq(ar, AxiAr {
            id: AxiId:0x0,
            addr: AxiAddr:0xFF0,
            region: AxiRegion:0x0,
            len: AxiLength:0x0,
            size: AxiSize::MAX_16B_TRANSFER,
            burst: AxiBurst::INCR,
            cache: AxiCache::DEV_NO_BUF,
            prot: AxiProt:0x0,
            qos: AxiQos:0x0
        });

        let tok = send(tok, axi_r_s, AxiR {
            id: AxiId:0x0,
            data: AxiData:0x1122_3344_5566_7788_9900_AABB_CCDD_EEFF,
            resp: AxiResp::OKAY,
            last: AxiLast:true
        });

        let (tok, ar) = recv(tok, axi_ar_r);
        assert_eq(ar, AxiAr {
            id: AxiId:0x0,
            addr: AxiAddr:0x1000,
            region: AxiRegion:0x0,
            len: AxiLength:0x0,
            size: AxiSize::MAX_16B_TRANSFER,
            burst: AxiBurst::INCR,
            cache: AxiCache::DEV_NO_BUF,
            prot: AxiProt:0x0,
            qos: AxiQos:0x0
        });

        let tok = send(tok, axi_r_s, AxiR {
            id: AxiId:0x0,
            data: AxiData:0x1122_3344_5566_7788_9900_AABB_CCDD_EEFF,
            resp: AxiResp::OKAY,
            last: AxiLast:true
        });

        let (tok, resp) = recv(tok, resp_r);
        assert_eq(resp, Resp {
            status: Status::OKAY,
            data: Data:0x00AA_BBCC_DDEE_FF11,
            length: Length:8,
            last: false
        });

        let (tok, resp) = recv(tok, resp_r);
        assert_eq(resp, Resp {
            status: Status::OKAY,
            data: Data:0x2233_4455_6677_8899,
            length: Length:8,
            last: true
        });

        let tok = send(tok, req_s, Req { addr: Addr:0x1001, length: Length:17 });
        let (tok, ar) = recv(tok, axi_ar_r);
        assert_eq(ar, AxiAr {
            id: AxiId:0x0,
            addr: AxiAddr:0x1000,
            region: AxiRegion:0x0,
            len: AxiLength:0x1,
            size: AxiSize::MAX_16B_TRANSFER,
            burst: AxiBurst::INCR,
            cache: AxiCache::DEV_NO_BUF,
            prot: AxiProt:0x0,
            qos: AxiQos:0x0
        });

        let tok = send(tok, axi_r_s, AxiR {
            id: AxiId:0x0,
            data: AxiData:0x1122_3344_5566_7788_9900_AABB_CCDD_EEFF,
            resp: AxiResp::OKAY,
            last: AxiLast:false
        });

        let tok = send(tok, axi_r_s, AxiR {
            id: AxiId:0x0,
            data: AxiData:0x1122_3344_5566_7788_9900_AABB_CCDD_EEFF,
            resp: AxiResp::OKAY,
            last: AxiLast:true
        });

        let (tok, resp) = recv(tok, resp_r);
        assert_eq(resp, Resp {
            status: Status::OKAY,
            data: Data:0x8899_00AA_BBCC_DDEE,
            length: Length:8,
            last: false
        });

        let (tok, resp) = recv(tok, resp_r);
        assert_eq(resp, Resp {
            status: Status::OKAY,
            data: Data:0xFF11_2233_4455_6677,
            length: Length:8,
            last: false
        });

        let (tok, resp) = recv(tok, resp_r);
        assert_eq(resp, Resp {
            status: Status::OKAY,
            data: Data:0xEE,
            length: Length:1,
            last: true
        });


        send(tok, terminator, true);
    }
}
