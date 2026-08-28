// Copyright 2023-2024 The XLS Authors
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

// Frontend Reader
//
// Part of the main controller, which translates
// (address, length) tuples from the address generator
// into AXI Read Transactions. Data received from AXI
// is written to the AXI Stream interface (expected FIFO).
//

import std;
import xls.modules.dma.bus.axi_pkg;
import xls.modules.dma.bus.axi_st_pkg;
import xls.modules.dma.common;
import xls.modules.dma.config;

type TransferDescBundle = common::TransferDescBundle;
type AxiArBundle = axi_pkg::AxiArBundle;
type AxiRBundle = axi_pkg::AxiRBundle;
type AxiStreamBundle = axi_st_pkg::AxiStreamBundle;

enum FrontendReaderStatusEnum : u3 {
    IDLE = 0,
    AXI_READ_REQ = 1,
    AXI_READ_RSP = 2,
    SEND_AXI_ST = 3,
}

struct FrontendReaderState<ADDR_W: u32, DATA_W: u32, ID_W: u32> {
    status: FrontendReaderStatusEnum,
    transfer_data: TransferDescBundle<ADDR_W>,
    r_bundle: AxiRBundle<DATA_W, ID_W>,
    burst_counter: u32,
}

// FIXME: Overflow issues, ensure correct back pressure.
// If we request more data from AXI than could be put into the FIFO, then
// we will overflow. To solve this problem, the FIFO should give us information
// about free space fs=(FIFO_SIZE-|ptr_w-ptr_r|).
//     if fs < transfer.length, then wait
// On the other hand, we can deassert ready on the System Bus and block it until
// there is sufficient space to end the transfer.
proc FrontendReader<ADDR_W: u32, DATA_W: u32, DATA_W_DIV8: u32, DEST_W: u32, ID_W: u32> {
    ch_addr_gen_req: chan<TransferDescBundle<ADDR_W>> in;
    ch_addr_gen_rsp: chan<()> out;
    ch_axi_ar: chan<AxiArBundle<ADDR_W, ID_W>> out;
    ch_axi_r: chan<AxiRBundle<DATA_W, ID_W>> in;
    ch_axi_st_write: chan<AxiStreamBundle<DATA_W, DATA_W_DIV8, DEST_W, ID_W>> out;

    config(ch_addr_gen_req: chan<TransferDescBundle<ADDR_W>> in, ch_addr_gen_rsp: chan<()> out,
           ch_axi_ar: chan<AxiArBundle<ADDR_W, ID_W>> out,
           ch_axi_r: chan<AxiRBundle<DATA_W, ID_W>> in,
           ch_axi_st_write: chan<AxiStreamBundle<DATA_W, DATA_W_DIV8, DEST_W, ID_W>> out) {
        (ch_addr_gen_req, ch_addr_gen_rsp, ch_axi_ar, ch_axi_r, ch_axi_st_write)
    }

    init {
        (FrontendReaderState<ADDR_W, DATA_W, ID_W> {
            status: FrontendReaderStatusEnum::IDLE,
            transfer_data: common::zeroTransferDescBundle<ADDR_W>(),
            r_bundle: axi_pkg::zeroAxiRBundle<DATA_W, ID_W>(),
            burst_counter: u32:0
        })
    }

    next(tok: token, state: FrontendReaderState<ADDR_W, DATA_W, ID_W>) {
        trace!(state);

        let (tok, next_transfer_data, goto_read_req) = recv_if_non_blocking(
            tok, ch_addr_gen_req, state.status == FrontendReaderStatusEnum::IDLE,
            state.transfer_data);

        if goto_read_req {
            trace_fmt!("[READER] Data transfer order [AG] = {}", next_transfer_data);
        } else {


        };

        let axi_read_req = axi_pkg::simpleAxiArBundle<ADDR_W, ID_W>(
            state.transfer_data.address, uN[ID_W]:0, state.transfer_data.length as u8);
        let tok = send_if(
            tok, ch_axi_ar, state.status == FrontendReaderStatusEnum::AXI_READ_REQ, axi_read_req);

        let (tok, axi_read_rsp, goto_stream_req) = recv_if_non_blocking(
            tok, ch_axi_r, state.status == FrontendReaderStatusEnum::AXI_READ_RSP,
            axi_pkg::zeroAxiRBundle<DATA_W, ID_W>());
        let next_r_bundle = if goto_stream_req { axi_read_rsp } else { state.r_bundle };

        let next_burst_counter =
            if goto_stream_req { state.burst_counter + u32:1 } else { state.burst_counter };

        let axi_read_req = AxiStreamBundle<DATA_W, DATA_W_DIV8, DEST_W, ID_W> {
            tdata: next_r_bundle.rdata,
            tstr: std::unsigned_max_value<DATA_W_DIV8>(),
            tkeep: uN[DATA_W_DIV8]:0,
            tlast: next_burst_counter == state.transfer_data.length,
            tid: uN[ID_W]:0,
            tdest: uN[DEST_W]:0
        };
        let tok = send_if(
            tok, ch_axi_st_write, state.status == FrontendReaderStatusEnum::SEND_AXI_ST,
            axi_read_req);

        let goto_idle = (state.status == FrontendReaderStatusEnum::SEND_AXI_ST) &&
                        (next_burst_counter == state.transfer_data.length);

        let tok = send_if(tok, ch_addr_gen_rsp, goto_idle, ());

        let next_burst_counter = if goto_idle { u32:0 } else { next_burst_counter };

        // Next state logic
        let nextStatus = if state.status == FrontendReaderStatusEnum::IDLE {
            if goto_read_req {
                FrontendReaderStatusEnum::AXI_READ_REQ
            } else {
                FrontendReaderStatusEnum::IDLE
            }
        } else if state.status == FrontendReaderStatusEnum::AXI_READ_REQ {
            FrontendReaderStatusEnum::AXI_READ_RSP
        } else if state.status == FrontendReaderStatusEnum::AXI_READ_RSP {
            if goto_stream_req {
                FrontendReaderStatusEnum::SEND_AXI_ST
            } else {
                FrontendReaderStatusEnum::AXI_READ_RSP
            }
        } else if state.status == FrontendReaderStatusEnum::SEND_AXI_ST {
            if goto_idle {
                FrontendReaderStatusEnum::IDLE
            } else {
                FrontendReaderStatusEnum::AXI_READ_RSP
            }
        } else {
            FrontendReaderStatusEnum::IDLE
        };

        // trace_fmt!("Next state = {}", nextStatus);
        FrontendReaderState {
            status: nextStatus,
            transfer_data: next_transfer_data,
            r_bundle: next_r_bundle,
            burst_counter: next_burst_counter
        }
    }
}

const TEST_0_ADDR_W = u32:32;
const TEST_0_DATA_W = u32:32;
const TEST_0_DATA_W_DIV8 = u32:4;
const TEST_0_DEST_W = u32:4;
const TEST_0_ID_W = u32:4;

#[test_proc]
proc testFrontendReader {
    ch_addr_gen_req: chan<TransferDescBundle<TEST_0_ADDR_W>> out;
    ch_addr_gen_rsp: chan<()> in;
    ch_axi_ar: chan<AxiArBundle<TEST_0_ADDR_W, TEST_0_ID_W>> in;
    ch_axi_r: chan<AxiRBundle<TEST_0_DATA_W, TEST_0_ID_W>> out;
    ch_axi_st_write:
    chan<AxiStreamBundle<TEST_0_DATA_W, TEST_0_DATA_W_DIV8, TEST_0_DEST_W, TEST_0_ID_W>> in;
    terminator: chan<bool> out;

    config(terminator: chan<bool> out) {
        let (ch_addr_gen_req_s, ch_addr_gen_req_r) = chan<TransferDescBundle<TEST_0_ADDR_W>>;
        let (ch_addr_gen_rsp_s, ch_addr_gen_rsp_r) = chan<()>;
        let (ch_axi_ar_s, ch_axi_ar_r) = chan<AxiArBundle<TEST_0_ADDR_W, TEST_0_ID_W>>;
        let (ch_axi_r_s, ch_axi_r_r) = chan<AxiRBundle<TEST_0_DATA_W, TEST_0_ID_W>>;
        let (ch_axi_st_write_s, ch_axi_st_write_r) =
            chan<AxiStreamBundle<TEST_0_DATA_W, TEST_0_DATA_W_DIV8, TEST_0_DEST_W, TEST_0_ID_W>>;
        spawn FrontendReader<
            TEST_0_ADDR_W, TEST_0_DATA_W, TEST_0_DATA_W_DIV8, TEST_0_DEST_W, TEST_0_ID_W>(
            ch_addr_gen_req_r, ch_addr_gen_rsp_s, ch_axi_ar_s, ch_axi_r_r, ch_axi_st_write_s);
        (
            ch_addr_gen_req_s, ch_addr_gen_rsp_r, ch_axi_ar_r, ch_axi_r_s, ch_axi_st_write_r,
            terminator,
        )
    }

    init { () }

    next(tok: token, state: ()) {
        let BASE_ADDR = uN[TEST_0_ADDR_W]:1000;
        let BASE_DATA = uN[TEST_0_DATA_W]:200;
        let NUM_TRANSFER = u32:2;
        let NUM_BURST = u32:3;
        let ID = uN[TEST_0_ID_W]:0;

        let tok = for (i, tok): (u32, token) in u32:0..NUM_TRANSFER {
            // Configuration from the AG
            let tok = send(
                tok, ch_addr_gen_req,
                TransferDescBundle<TEST_0_ADDR_W> { address: (BASE_ADDR + i), length: NUM_BURST });

            // AXI AR
            let (tok, test_axi_ar) = recv(tok, ch_axi_ar);
            trace_fmt!("test_axi_ar = {}", test_axi_ar);

            // AXI R
            let tok = for (j, tok): (u32, token) in uN[TEST_0_ADDR_W]:0..NUM_BURST {
                let tok = send(
                    tok, ch_axi_r,
                    AxiRBundle<TEST_0_DATA_W, TEST_0_ID_W> {
                        rid: ID,
                        rdata: BASE_DATA + (i + u32:10 * j as uN[TEST_0_DATA_W]),
                        rresp: axi_pkg::AXI_READ_RESPONSE_CODES::OKAY,
                        rlast: (j == (NUM_BURST - u32:1)) as u1
                    });
                tok
            }(tok);

            // FIFO
            let tok = for (j, tok): (u32, token) in uN[TEST_0_ADDR_W]:0..NUM_BURST {
                let (tok, r_fifo_data) = recv(tok, ch_axi_st_write);
                trace_fmt!("r_fifo_data = {}", r_fifo_data);
                assert_eq(r_fifo_data.tdata, BASE_DATA + (i + u32:10 * j as uN[TEST_0_DATA_W]));
                // Signal done
                tok
            }(tok);
            let (tok, _) = recv(tok, ch_addr_gen_rsp);
            tok
        }(tok);

        let tok = send(tok, terminator, true);
    }
}

proc frontend_reader {
    config(ch_addr_gen_req: chan<TransferDescBundle<config::TOP_ADDR_W>> in,
           ch_addr_gen_rsp: chan<()> out,
           ch_axi_ar: chan<AxiArBundle<config::TOP_ADDR_W, config::TOP_ID_W>> out,
           ch_axi_r: chan<AxiRBundle<config::TOP_DATA_W, config::TOP_ID_W>> in,
           ch_axi_st_write: chan<AxiStreamBundle<config::TOP_DATA_W, config::TOP_DATA_W_DIV8, config::TOP_DEST_W, config::TOP_ID_W>> out) {

        spawn FrontendReader<
            config::TOP_ADDR_W, config::TOP_DATA_W, config::TOP_DATA_W_DIV8, config::TOP_DEST_W, config::TOP_ID_W>(
            ch_addr_gen_req, ch_addr_gen_rsp, ch_axi_ar, ch_axi_r, ch_axi_st_write);
        ()
    }

    init { () }

    next(tok: token, state: ()) {  }
}
