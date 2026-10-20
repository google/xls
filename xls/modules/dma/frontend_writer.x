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

// Frontend Writer
//
// Part of the main controller, which translates
// (address, length) tuples from the address generator
// into AXI Write Transactions. Data sent to AXI
// is read from the AXI Stream interface (FIFO).
//
//

import std;
import xls.modules.dma.bus.axi_pkg;
import xls.modules.dma.bus.axi_st_pkg;
import xls.modules.dma.common;
import xls.modules.dma.config;

type TransferDescBundle = common::TransferDescBundle;
type AxiAwBundle = axi_pkg::AxiAwBundle;
type AxiWBundle = axi_pkg::AxiWBundle;
type AxiBBundle = axi_pkg::AxiBBundle;
type AxiStreamBundle = axi_st_pkg::AxiStreamBundle;

enum FrontendWriterStatusEnum : u3 {
    IDLE = 0,
    AXI_WRITE_AW = 1,
    READ_AXI_ST = 2,
    AXI_WRITE_W = 3,
    AXI_WRITE_B = 4,
}

struct FrontendWriterState<ADDR_W: u32, DATA_W: u32, ID_W: u32, STRB_W: u32> {
    status: FrontendWriterStatusEnum,
    transfer_data: TransferDescBundle<ADDR_W>,
    aw_bundle: AxiAwBundle<ADDR_W, ID_W>,
    w_bundle: AxiWBundle<DATA_W, STRB_W>,
    b_bundle: AxiBBundle<ID_W>,
    burst_counter: u32,
}

proc FrontendWriter<ADDR_W: u32, DATA_W: u32, DATA_W_DIV8: u32, DEST_W: u32, ID_W: u32, STRB_W: u32>
{
    ch_addr_gen_req: chan<TransferDescBundle<ADDR_W>> in;
    ch_addr_gen_rsp: chan<()> out;
    ch_axi_aw: chan<AxiAwBundle<ADDR_W, ID_W>> out;
    ch_axi_w: chan<AxiWBundle<DATA_W, STRB_W>> out;
    ch_axi_b: chan<AxiBBundle<ID_W>> in;
    ch_axi_st_read: chan<AxiStreamBundle<DATA_W, DATA_W_DIV8, DEST_W, ID_W>> in;

    config(ch_addr_gen_req: chan<TransferDescBundle<ADDR_W>> in, ch_addr_gen_rsp: chan<()> out,
           ch_axi_aw: chan<AxiAwBundle<ADDR_W, ID_W>> out,
           ch_axi_w: chan<AxiWBundle<DATA_W, STRB_W>> out, ch_axi_b: chan<AxiBBundle<ID_W>> in,
           ch_axi_st_read: chan<AxiStreamBundle<DATA_W, DATA_W_DIV8, DEST_W, ID_W>> in) {
        (ch_addr_gen_req, ch_addr_gen_rsp, ch_axi_aw, ch_axi_w, ch_axi_b, ch_axi_st_read)
    }

    init {
        (FrontendWriterState<ADDR_W, DATA_W, ID_W, STRB_W> {
            status: FrontendWriterStatusEnum::IDLE,
            transfer_data:
            TransferDescBundle<ADDR_W> { address: uN[ADDR_W]:0, length: uN[ADDR_W]:0 },
            aw_bundle:
            AxiAwBundle<ADDR_W, ID_W> {
                awid: uN[ID_W]:0,
                awaddr: uN[ADDR_W]:0,
                awsize: axi_pkg::AXI_AXSIZE_ENCODING::MAX_8B_TRANSFER,
                awlen: uN[8]:0,
                awburst: axi_pkg::AXI_AXBURST_ENCODING::FIXED
            },
            w_bundle:
            AxiWBundle<DATA_W, STRB_W> { wdata: uN[DATA_W]:0, wstrb: uN[STRB_W]:0, wlast: u1:0 },
            b_bundle:
            AxiBBundle<ID_W> { bresp: axi_pkg::AXI_WRITE_RESPONSE_CODES::OKAY, bid: uN[ID_W]:0 },
            burst_counter: u32:0
        })
    }

    next(tok: token, state: FrontendWriterState<ADDR_W, DATA_W, ID_W, STRB_W>) {
        // Address Generator
        let (tok, next_transfer_data, goto_axi_write_aw) = recv_if_non_blocking(
            tok, ch_addr_gen_req, state.status == FrontendWriterStatusEnum::IDLE,
            state.transfer_data);

        // Send AW
        let next_aw_bundle = if goto_axi_write_aw {
            trace_fmt!("[WRITER] Data transfer order [AG] = {}", next_transfer_data);
            AxiAwBundle<ADDR_W, ID_W> {
                awid: uN[ID_W]:0,
                awaddr: next_transfer_data.address,
                awsize: axi_pkg::AXI_AXSIZE_ENCODING::MAX_8B_TRANSFER,
                awlen: next_transfer_data.length as u8,
                awburst: axi_pkg::AXI_AXBURST_ENCODING::FIXED
            }
        } else {
            state.aw_bundle
        };
        let tok = send_if(tok, ch_axi_aw, goto_axi_write_aw, next_aw_bundle);

        let goto_read_axi_st = state.status == FrontendWriterStatusEnum::AXI_WRITE_AW;

        // here
        let (tok, r_data, goto_axi_write_w) = recv_if_non_blocking(
            tok, ch_axi_st_read, state.status == FrontendWriterStatusEnum::READ_AXI_ST,
            axi_st_pkg::zeroAxiStreamBundle<DATA_W, DATA_W_DIV8, DEST_W, ID_W>());

        let next_w_bundle = if goto_axi_write_w {
            AxiWBundle<DATA_W, STRB_W> {
                wdata: r_data.tdata,
                wstrb: std::unsigned_max_value<STRB_W>(),
                wlast: state.burst_counter == (state.transfer_data.length - u32:1)
            }
        } else {
            state.w_bundle
        };

        // Send W
        let tok = send_if(
            tok, ch_axi_w, state.status == FrontendWriterStatusEnum::AXI_WRITE_W, state.w_bundle);

        let next_burst_counter =
            if goto_axi_write_w { state.burst_counter + u32:1 } else { state.burst_counter };
        trace_fmt!("next burst counter = {}", next_burst_counter);

        // B
        let goto_axi_write_b = (state.status == FrontendWriterStatusEnum::AXI_WRITE_W) &&
                               (next_burst_counter == state.transfer_data.length);

        // Wait for B
        let (tok, b_data, goto_idle) = recv_if_non_blocking(
            tok, ch_axi_b, state.status == FrontendWriterStatusEnum::AXI_WRITE_B,
            axi_pkg::zeroAxiBBundle<ID_W>());
        let next_b_bundle = if goto_idle {
            trace_fmt!("b_data = {}", b_data);
            b_data
        } else {
            state.b_bundle
        };

        let next_burst_counter = if goto_idle { u32:0 } else { next_burst_counter };

        // TODO: If B channel response is not OKAY, signal an error
        let tok = send_if(tok, ch_addr_gen_rsp, goto_idle, ());

        // Next state logic
        let nextStatus = if state.status == FrontendWriterStatusEnum::IDLE {
            if goto_axi_write_aw {
                FrontendWriterStatusEnum::AXI_WRITE_AW
            } else {
                FrontendWriterStatusEnum::IDLE
            }
        } else if state.status == FrontendWriterStatusEnum::AXI_WRITE_AW {
            if goto_read_axi_st {
                FrontendWriterStatusEnum::READ_AXI_ST
            } else {
                FrontendWriterStatusEnum::AXI_WRITE_AW
            }
        } else if state.status == FrontendWriterStatusEnum::READ_AXI_ST {
            if goto_axi_write_w {
                FrontendWriterStatusEnum::AXI_WRITE_W
            } else {
                FrontendWriterStatusEnum::READ_AXI_ST
            }
        } else if state.status == FrontendWriterStatusEnum::AXI_WRITE_W {
            if goto_axi_write_b {
                FrontendWriterStatusEnum::AXI_WRITE_B
            } else {
                FrontendWriterStatusEnum::READ_AXI_ST
            }
        } else if state.status == FrontendWriterStatusEnum::AXI_WRITE_B {
            if goto_idle {
                FrontendWriterStatusEnum::IDLE
            } else {
                FrontendWriterStatusEnum::AXI_WRITE_B
            }
        } else {
            FrontendWriterStatusEnum::IDLE
        };

        // trace_fmt!("NextState = {}", nextStatus);
        FrontendWriterState<ADDR_W, DATA_W, ID_W, STRB_W> {
            status: nextStatus,
            transfer_data: next_transfer_data,
            aw_bundle: next_aw_bundle,
            w_bundle: next_w_bundle,
            b_bundle: next_b_bundle,
            burst_counter: next_burst_counter
        }
    }
}

const TEST_0_ADDR_W = u32:32;
const TEST_0_DATA_W = u32:32;
const TEST_0_DATA_W_DIV8 = u32:4;
const TEST_0_DEST_W = u32:32;
const TEST_0_ID_W = u32:32;
const TEST_0_STRB_W = u32:32;

#[test_proc]
proc testFrontendWriter {
    ch_addr_gen_req: chan<TransferDescBundle<TEST_0_ADDR_W>> out;
    ch_addr_gen_rsp: chan<()> in;
    ch_axi_aw: chan<AxiAwBundle<TEST_0_ADDR_W, TEST_0_ID_W>> in;
    ch_axi_w: chan<AxiWBundle<TEST_0_DATA_W, TEST_0_STRB_W>> in;
    ch_axi_b: chan<AxiBBundle<TEST_0_ID_W>> out;
    ch_axi_st_read:
    chan<AxiStreamBundle<TEST_0_DATA_W, TEST_0_DATA_W_DIV8, TEST_0_DEST_W, TEST_0_ID_W>> out;
    terminator: chan<bool> out;

    config(terminator: chan<bool> out) {
        let (ch_addr_gen_req_s, ch_addr_gen_req_r) = chan<TransferDescBundle<TEST_0_ADDR_W>>;
        let (ch_addr_gen_rsp_s, ch_addr_gen_rsp_r) = chan<()>;
        let (ch_axi_aw_s, ch_axi_aw_r) = chan<AxiAwBundle<TEST_0_ADDR_W, TEST_0_ID_W>>;
        let (ch_axi_w_s, ch_axi_w_r) = chan<AxiWBundle<TEST_0_DATA_W, TEST_0_STRB_W>>;
        let (ch_axi_b_s, ch_axi_b_r) = chan<AxiBBundle<TEST_0_ID_W>>;
        let (ch_axi_st_read_s, ch_axi_st_read_r) =
            chan<AxiStreamBundle<TEST_0_DATA_W, TEST_0_DATA_W_DIV8, TEST_0_DEST_W, TEST_0_ID_W>>;
        spawn FrontendWriter<
            TEST_0_ADDR_W, TEST_0_DATA_W, TEST_0_DATA_W_DIV8, TEST_0_DEST_W, TEST_0_ID_W, TEST_0_STRB_W>(
            ch_addr_gen_req_r, ch_addr_gen_rsp_s, ch_axi_aw_s, ch_axi_w_s, ch_axi_b_r,
            ch_axi_st_read_r);
        (
            ch_addr_gen_req_s, ch_addr_gen_rsp_r, ch_axi_aw_r, ch_axi_w_r, ch_axi_b_s,
            ch_axi_st_read_s, terminator,
        )
    }

    init { () }

    next(tok: token, state: ()) {
        let BASE_ADDR = uN[TEST_0_ADDR_W]:1000;
        let BASE_DATA = uN[TEST_0_DATA_W]:200;
        let NUM_TRANSFER = u32:9;
        let NUM_BURST = u32:2;

        let tok = for (i, tok): (u32, token) in u32:0..NUM_TRANSFER {
            // Start transfer
            let tok = send(
                tok, ch_addr_gen_req,
                TransferDescBundle<TEST_0_ADDR_W> { address: BASE_ADDR + i, length: NUM_BURST });
            // Provide stream data
            let tok = for (j, tok): (u32, token) in uN[TEST_0_ADDR_W]:0..NUM_BURST {
                trace_fmt!("Burst j={}", j);
                let st_data = axi_st_pkg::simpleAxiStreamBundle<
                    TEST_0_DATA_W, TEST_0_DATA_W_DIV8, TEST_0_DEST_W, TEST_0_ID_W>(
                    BASE_DATA + (i + u32:10 * j as uN[TEST_0_DATA_W]));
                let tok = send(tok, ch_axi_st_read, st_data);
                trace_fmt!("Sent st_data = {}", st_data);
                tok
            }(tok);
            trace_fmt!("----------------------------------------");
            // Handle AXI Write
            let (tok, aw) = recv(tok, ch_axi_aw);
            trace_fmt!("SBUS: AW = {}", aw);

            let tok = for (j, tok): (u32, token) in uN[TEST_0_ADDR_W]:0..NUM_BURST {
                let (tok, w) = recv(tok, ch_axi_w);
                trace_fmt!("SBUS:  W = {}", w);
                let tok = send(tok, ch_axi_b, axi_pkg::zeroAxiBBundle<TEST_0_ID_W>());
                assert_eq(w.wdata, BASE_DATA + (i + u32:10 * j as uN[TEST_0_DATA_W]));
                tok
            }(tok);
            // End transfer
            let (tok, transfer_done) = recv(tok, ch_addr_gen_rsp);
            trace_fmt!("transfer_done = {}", transfer_done);
            tok
        }(tok);
        let tok = send(tok, terminator, true);
    }
}

proc frontend_writer {
    config(ch_addr_gen_req: chan<TransferDescBundle<config::TOP_ADDR_W>> in,
           ch_addr_gen_rsp: chan<()> out,
           ch_axi_aw: chan<AxiAwBundle<config::TOP_ADDR_W, config::TOP_ID_W>> out,
           ch_axi_w: chan<AxiWBundle<config::TOP_DATA_W, config::TOP_STRB_W>> out,
           ch_axi_b: chan<AxiBBundle<config::TOP_ID_W>> in,
           ch_axi_st_read: chan<AxiStreamBundle<config::TOP_DATA_W, config::TOP_DATA_W_DIV8, config::TOP_DEST_W, config::TOP_ID_W>> in) {

        spawn FrontendWriter<
            config::TOP_ADDR_W, config::TOP_DATA_W, config::TOP_DATA_W_DIV8, config::TOP_DEST_W, config::TOP_ID_W, config::TOP_STRB_W>(
            ch_addr_gen_req, ch_addr_gen_rsp, ch_axi_aw, ch_axi_w, ch_axi_b, ch_axi_st_read);
        ()
    }

    init { () }

    next(tok: token, state: ()) {  }
}
