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

import std;
import xls.modules.dma.bus.axi_pkg;
import xls.modules.dma.common;
import xls.modules.dma.config;
import xls.modules.dma.csr;

// FIXME: casting imported types is a workaround
// https://github.com/google/xls/issues/1030
type MainCtrlBundle = common::MainCtrlBundle;

type AxiAwBundle = axi_pkg::AxiAwBundle;
type AxiWBundle = axi_pkg::AxiWBundle;
type AxiBBundle = axi_pkg::AxiBBundle;
type AxiArBundle = axi_pkg::AxiArBundle;
type AxiRBundle = axi_pkg::AxiRBundle;

struct axi_csr_state<ADDR_W: u32, DATA_W: u32> {
    waddr: uN[ADDR_W],
    wdata: uN[DATA_W],
    raddr: uN[ADDR_W],
    rdata: uN[DATA_W],
}

// AXI4 subordinate receives AXI4 transactions and translates them into simple
// read/writes for CSR
proc AxiCsr<ID_W: u32, ADDR_W: u32, DATA_W: u32, STRB_W: u32, REGS_N: u32> {
    aw_ch: chan<AxiAwBundle<ADDR_W, ID_W>> in;
    w_ch: chan<AxiWBundle<DATA_W, STRB_W>> in;
    b_ch: chan<AxiBBundle<ID_W>> out;
    ar_ch: chan<AxiArBundle<ADDR_W, ID_W>> in;
    r_ch: chan<AxiRBundle<DATA_W, ID_W>> out;
    read_req: chan<csr::ReadReq<ADDR_W>> out;
    read_resp: chan<csr::ReadResp<DATA_W>> in;
    write_req: chan<csr::WriteReq<ADDR_W, DATA_W>> out;
    write_resp: chan<csr::WriteResp> in;
    ch_writer_start: chan<u1> out;
    ch_writer_configuration: chan<MainCtrlBundle<ADDR_W>> out;
    ch_writer_busy: chan<u1> in;
    ch_writer_done: chan<u1> in;
    ch_reader_start: chan<u1> out;
    ch_reader_configuration: chan<MainCtrlBundle<ADDR_W>> out;
    ch_reader_busy: chan<u1> in;
    ch_reader_done: chan<u1> in;

    config(aw_ch: chan<AxiAwBundle<ADDR_W, ID_W>> in, w_ch: chan<AxiWBundle<DATA_W, STRB_W>> in,
           b_ch: chan<AxiBBundle<ID_W>> out, ar_ch: chan<AxiArBundle<ADDR_W, ID_W>> in,
           r_ch: chan<AxiRBundle<DATA_W, ID_W>> out, ch_writer_start: chan<u1> out,
           ch_writer_configuration: chan<MainCtrlBundle<ADDR_W>> out, ch_writer_busy: chan<u1> in,
           ch_writer_done: chan<u1> in, ch_reader_start: chan<u1> out,
           ch_reader_configuration: chan<MainCtrlBundle<ADDR_W>> out, ch_reader_busy: chan<u1> in,
           ch_reader_done: chan<u1> in, reader_sync_req: chan<()> in, reader_sync_rsp: chan<()> out,
           writer_sync_req: chan<()> in, writer_sync_rsp: chan<()> out) {
        let (read_req_s, read_req_r) = chan<csr::ReadReq<ADDR_W>>;
        let (read_resp_s, read_resp_r) = chan<csr::ReadResp<DATA_W>>;
        let (write_req_s, write_req_r) = chan<csr::WriteReq<ADDR_W, DATA_W>>;
        let (write_resp_s, write_resp_r) = chan<csr::WriteResp>;

        spawn csr::Csr<ADDR_W, DATA_W, REGS_N>(
            read_req_r, read_resp_s, write_req_r, write_resp_s, ch_writer_start,
            ch_writer_configuration, ch_writer_busy, ch_writer_done, ch_reader_start,
            ch_reader_configuration, ch_reader_busy, ch_reader_done, reader_sync_req,
            reader_sync_rsp, writer_sync_req, writer_sync_rsp);
        (
            aw_ch, w_ch, b_ch, ar_ch, r_ch, read_req_s, read_resp_r, write_req_s, write_resp_r,
            ch_writer_start, ch_writer_configuration, ch_writer_busy, ch_writer_done,
            ch_reader_start, ch_reader_configuration, ch_reader_busy, ch_reader_done,
        )
    }

    init {
        axi_csr_state<ADDR_W, DATA_W> {
            waddr: uN[ADDR_W]:0, wdata: uN[DATA_W]:0, raddr: uN[ADDR_W]:0, rdata: uN[DATA_W]:0
        }
    }

    next(tok: token, state: axi_csr_state<ADDR_W, DATA_W>) {
        // AW Channel Handler
        let (tok, aw_payload, aw_valid) = recv_non_blocking(tok, aw_ch, zero!<AxiAwBundle>());

        // CSR Addresses in software are expressed in bytes.
        // CSR Addresses in hardware are expressed in words.
        // The AXI address must be translated: divided by 4 with default config.
        // FIXME: Address translation is only correct with 32-bit configuration.
        // In general, this should be: addr /= len(word)/len(byte)
        let w_addr = if aw_valid { aw_payload.awaddr >> uN[ADDR_W]:2 } else { state.waddr };

        // W channel Handler
        let (tok, w_payload, w_valid) = recv_non_blocking(tok, w_ch, zero!<AxiWBundle>());
        let w_data = if w_valid { w_payload.wdata } else { state.wdata };

        // Handle write to CSR
        let tok = send_if(tok, write_req, w_valid, csr::WriteWordReq(w_addr, w_data));
        let (tok, _, csr_write_valid) = recv_non_blocking(tok, write_resp, csr::WriteResp {});

        // B Channel Handlers
        let b_msg = AxiBBundle { bresp: axi_pkg::AXI_WRITE_RESPONSE_CODES::OKAY, bid: uN[ID_W]:0 };
        let tok = send_if(tok, b_ch, csr_write_valid, b_msg);

        // AR Channel Handler
        let zero_AxiArBundle = axi_pkg::zeroAxiArBundle<ADDR_W, ID_W>();
        let (tok, ar_payload, ar_valid) = recv_non_blocking(tok, ar_ch, zero_AxiArBundle);

        // CSR Addresses in software are expressed in bytes.
        // CSR Addresses in hardware are expressed in words.
        // The AXI address must be translated: divided by 4 with default config.
        // FIXME: Address translation is only correct with 32-bit configuration.
        // In general, this should be: addr /= len(word)/len(byte)
        let r_addr = if ar_valid { ar_payload.araddr >> uN[ADDR_W]:2 } else { state.raddr };

        // Handle Read from CSR
        let tok = send_if(tok, read_req, ar_valid, csr::ReadWordReq(r_addr));
        let (tok, r_data, csr_read_valid) =
            recv_non_blocking(tok, read_resp, zero!<csr::ReadResp>());

        // R Channel Handler
        let tok = send_if(
            tok, r_ch, csr_read_valid,
            AxiRBundle {
                rid: uN[ID_W]:0,
                rdata: r_data.data,
                rresp: axi_pkg::AXI_READ_RESPONSE_CODES::OKAY,
                rlast: uN[1]:1
            });

        axi_csr_state { waddr: w_addr, wdata: w_data, raddr: r_addr, rdata: r_data.data }
    }
}

// Tests
const TEST_ID_W = u32:4;
const TEST_ADDR_W = config::CSR_ADDR_W;
const TEST_DATA_W = config::CSR_DATA_W;
const TEST_STRB_W = config::CSR_DATA_W / u32:8;
const TEST_REGS_N = config::CSR_REGS_N;

#[test_proc]
proc test_axi_csr {
    aw_ch: chan<AxiAwBundle<TEST_ADDR_W, TEST_ID_W>> out;
    w_ch: chan<AxiWBundle<TEST_DATA_W, TEST_STRB_W>> out;
    b_ch: chan<AxiBBundle<TEST_ID_W>> in;
    ar_ch: chan<AxiArBundle<TEST_ADDR_W, TEST_ID_W>> out;
    r_ch: chan<AxiRBundle<TEST_DATA_W, TEST_ID_W>> in;
    ch_writer_start: chan<u1> in;
    ch_writer_configuration: chan<MainCtrlBundle<TEST_ADDR_W>> in;
    ch_writer_busy: chan<u1> out;
    ch_writer_done: chan<u1> out;
    ch_reader_start: chan<u1> in;
    ch_reader_configuration: chan<MainCtrlBundle<TEST_ADDR_W>> in;
    ch_reader_busy: chan<u1> out;
    ch_reader_done: chan<u1> out;
    reader_sync_req: chan<()> out;
    reader_sync_rsp: chan<()> in;
    writer_sync_req: chan<()> out;
    writer_sync_rsp: chan<()> in;
    terminator: chan<bool> out;

    config(terminator: chan<bool> out) {
        let (aw_req_s, aw_req_r) = chan<AxiAwBundle<TEST_ADDR_W, TEST_ID_W>>;
        let (w_req_s, w_req_r) = chan<AxiWBundle<TEST_DATA_W, TEST_STRB_W>>;
        let (b_req_s, b_req_r) = chan<AxiBBundle<TEST_ID_W>>;
        let (ar_ch_s, ar_ch_r) = chan<AxiArBundle<TEST_ADDR_W, TEST_ID_W>>;
        let (r_ch_s, r_ch_r) = chan<AxiRBundle<TEST_DATA_W, TEST_ID_W>>;
        let (ch_writer_start_s, ch_writer_start_r) = chan<u1>;
        let (ch_writer_configuration_s, ch_writer_configuration_r) =
            chan<MainCtrlBundle<TEST_ADDR_W>>;
        let (ch_writer_busy_s, ch_writer_busy_r) = chan<u1>;
        let (ch_writer_done_s, ch_writer_done_r) = chan<u1>;
        let (ch_reader_start_s, ch_reader_start_r) = chan<u1>;
        let (ch_reader_configuration_s, ch_reader_configuration_r) =
            chan<MainCtrlBundle<TEST_ADDR_W>>;
        let (ch_reader_busy_s, ch_reader_busy_r) = chan<u1>;
        let (ch_reader_done_s, ch_reader_done_r) = chan<u1>;

        let (reader_sync_req_s, reader_sync_req_r) = chan<()>;
        let (reader_sync_rsp_s, reader_sync_rsp_r) = chan<()>;
        let (writer_sync_req_s, writer_sync_req_r) = chan<()>;
        let (writer_sync_rsp_s, writer_sync_rsp_r) = chan<()>;

        spawn AxiCsr<TEST_ID_W, TEST_ADDR_W, TEST_DATA_W, TEST_STRB_W, TEST_REGS_N>(
            aw_req_r, w_req_r, b_req_s, ar_ch_r, r_ch_s, ch_writer_start_s,
            ch_writer_configuration_s, ch_writer_busy_r, ch_writer_done_r, ch_reader_start_s,
            ch_reader_configuration_s, ch_reader_busy_r, ch_reader_done_r, reader_sync_req_r,
            reader_sync_rsp_s, writer_sync_req_r, writer_sync_rsp_s);
        (
            aw_req_s, w_req_s, b_req_r, ar_ch_s, r_ch_r, ch_writer_start_r,
            ch_writer_configuration_r, ch_writer_busy_s, ch_writer_done_s, ch_reader_start_r,
            ch_reader_configuration_r, ch_reader_busy_s, ch_reader_done_s, reader_sync_req_s,
            reader_sync_rsp_r, writer_sync_req_s, writer_sync_rsp_r, terminator,
        )
    }

    init { () }

    next(tok: token, state: ()) {
        let id = uN[TEST_ID_W]:0;

        // Write to all CSRs
        for (i, tok): (u32, token) in u32:0..TEST_REGS_N {
            let addr = (i << u32:2) as uN[TEST_ADDR_W];
            let data = (u32:0x00000033 + i) as uN[TEST_DATA_W];

            let aw = axi_pkg::simpleAxiAwBundle<TEST_ADDR_W, TEST_ID_W>(addr, id);
            let w = axi_pkg::simpleAxiWBundle<TEST_DATA_W, TEST_STRB_W>(data);

            let tok = send(tok, aw_ch, aw);
            let tok = send(tok, w_ch, w);
            let (tok, b_resp) = recv(tok, b_ch);
            assert_eq(b_resp.bresp, axi_pkg::AXI_WRITE_RESPONSE_CODES::OKAY);
            (tok)
        }(tok);

        // Read all values and compare with writes
        for (i, tok): (u32, token) in u32:0..TEST_REGS_N {
            let addr = (i << u32:2) as uN[TEST_ADDR_W];
            let ar = axi_pkg::simpleAxiArBundle<TEST_ADDR_W, TEST_ID_W>(addr, id, u8:0);
            let tok = send(tok, ar_ch, ar);
            let (tok, rcv) = recv(tok, r_ch);
            assert_eq(rcv.rdata, (u32:0x00000033 + i) as uN[TEST_DATA_W]);
            (tok)
        }(tok);

        let tok = send(tok, terminator, true);
    }
}

// Verilog example
proc axi_csr {
    config(aw_ch: chan<AxiAwBundle<config::TOP_ADDR_W, config::TOP_ID_W>> in,
           w_ch: chan<AxiWBundle<config::TOP_DATA_W, config::TOP_STRB_W>> in,
           b_ch: chan<AxiBBundle<config::TOP_ID_W>> out,
           ar_ch: chan<AxiArBundle<config::TOP_ADDR_W, config::TOP_ID_W>> in,
           r_ch: chan<AxiRBundle<config::TOP_DATA_W, config::TOP_ID_W>> out,
           ch_writer_start: chan<u1> out,
           ch_writer_configuration: chan<MainCtrlBundle<config::TOP_ADDR_W>> out,
           ch_writer_busy: chan<u1> in, ch_writer_done: chan<u1> in, ch_reader_start: chan<u1> out,
           ch_reader_configuration: chan<MainCtrlBundle<config::TOP_ADDR_W>> out,
           ch_reader_busy: chan<u1> in, ch_reader_done: chan<u1> in, reader_sync_req: chan<()> in,
           reader_sync_rsp: chan<()> out, writer_sync_req: chan<()> in,
           writer_sync_rsp: chan<()> out) {
        spawn AxiCsr<
            config::TOP_ID_W, config::TOP_ADDR_W, config::TOP_DATA_W, config::TOP_STRB_W, config::TOP_REGS_N>(
            aw_ch, w_ch, b_ch, ar_ch, r_ch, ch_writer_start, ch_writer_configuration,
            ch_writer_busy, ch_writer_done, ch_reader_start, ch_reader_configuration,
            ch_reader_busy, ch_reader_done, reader_sync_req, reader_sync_rsp, writer_sync_req,
            writer_sync_rsp);
        ()
    }

    init { () }

    next(tok: token, state: ()) {  }
}
