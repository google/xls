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

// Main Controller
//
// The Controller connects the AXI System Bus to the GPF via AXI-Stream FIFOs.
// It is controlled by the state of the CSRs.
// It is currently assumed that all buses in the design have equal data width.

import std;
import xls.modules.dma.bus.axi_pkg;
import xls.modules.dma.bus.axi_st_pkg;
import xls.modules.dma.common;
import xls.modules.dma.config;
import xls.modules.dma.address_generator;
import xls.modules.dma.axi_csr;
import xls.modules.dma.frontend_reader;
import xls.modules.dma.frontend_writer;
import xls.modules.dma.gpf;

type MainCtrlBundle = common::MainCtrlBundle;
type TransferDescBundle = common::TransferDescBundle;
type AxiArBundle = axi_pkg::AxiArBundle;
type AxiRBundle = axi_pkg::AxiRBundle;
type AxiAwBundle = axi_pkg::AxiAwBundle;
type AxiWBundle = axi_pkg::AxiWBundle;
type AxiBBundle = axi_pkg::AxiBBundle;
type AxiStreamBundle = axi_st_pkg::AxiStreamBundle;

proc MainController<ADDR_W: u32, DATA_W: u32, DATA_W_DIV8: u32, DEST_W: u32, ID_W: u32, REGS_N: u32, STRB_W:
u32>
{
    ch_axi_ctrl_aw: chan<AxiAwBundle<ADDR_W, ID_W>> in;
    ch_axi_ctrl_w: chan<AxiWBundle<DATA_W, STRB_W>> in;
    ch_axi_ctrl_b: chan<AxiBBundle<ID_W>> out;
    ch_axi_ctrl_ar: chan<AxiArBundle<ADDR_W, ID_W>> in;
    ch_axi_ctrl_r: chan<AxiRBundle<DATA_W, ID_W>> out;
    ch_axi_data_aw: chan<AxiAwBundle<ADDR_W, ID_W>> out;
    ch_axi_data_w: chan<AxiWBundle<DATA_W, STRB_W>> out;
    ch_axi_data_b: chan<AxiBBundle<ID_W>> in;
    ch_axi_data_ar: chan<AxiArBundle<ADDR_W, ID_W>> out;
    ch_axi_data_r: chan<AxiRBundle<DATA_W, ID_W>> in;
    ch_axi_st_write: chan<AxiStreamBundle<DATA_W, DATA_W_DIV8, DEST_W, ID_W>> out;
    ch_axi_st_read: chan<AxiStreamBundle<DATA_W, DATA_W_DIV8, DEST_W, ID_W>> in;
    reader_sync_req: chan<()> in;
    reader_sync_rsp: chan<()> out;
    writer_sync_req: chan<()> in;
    writer_sync_rsp: chan<()> out;

    config(ch_axi_ctrl_aw: chan<AxiAwBundle<ADDR_W, ID_W>> in,
           ch_axi_ctrl_w: chan<AxiWBundle<DATA_W, STRB_W>> in,
           ch_axi_ctrl_b: chan<AxiBBundle<ID_W>> out,
           ch_axi_ctrl_ar: chan<AxiArBundle<ADDR_W, ID_W>> in,
           ch_axi_ctrl_r: chan<AxiRBundle<DATA_W, ID_W>> out,
           ch_axi_data_aw: chan<AxiAwBundle<ADDR_W, ID_W>> out,
           ch_axi_data_w: chan<AxiWBundle<DATA_W, STRB_W>> out,
           ch_axi_data_b: chan<AxiBBundle<ID_W>> in,
           ch_axi_data_ar: chan<AxiArBundle<ADDR_W, ID_W>> out,
           ch_axi_data_r: chan<AxiRBundle<DATA_W, ID_W>> in,
           ch_axi_st_write: chan<AxiStreamBundle<DATA_W, DATA_W_DIV8, DEST_W, ID_W>> out,
           ch_axi_st_read: chan<AxiStreamBundle<DATA_W, DATA_W_DIV8, DEST_W, ID_W>> in,
           reader_sync_req: chan<()> in, reader_sync_rsp: chan<()> out,
           writer_sync_req: chan<()> in, writer_sync_rsp: chan<()> out) {
        let (ch_writer_start_s, ch_writer_start_r) = chan<u1>;
        let (ch_writer_configuration_s, ch_writer_configuration_r) = chan<MainCtrlBundle<ADDR_W>>;
        let (ch_writer_busy_s, ch_writer_busy_r) = chan<u1>;
        let (ch_writer_done_s, ch_writer_done_r) = chan<u1>;
        let (ch_reader_start_s, ch_reader_start_r) = chan<u1>;
        let (ch_reader_configuration_s, ch_reader_configuration_r) = chan<MainCtrlBundle<ADDR_W>>;
        let (ch_reader_busy_s, ch_reader_busy_r) = chan<u1>;
        let (ch_reader_done_s, ch_reader_done_r) = chan<u1>;
        let (ch_reader_addr_gen_req_s, ch_reader_addr_gen_req_r) = chan<TransferDescBundle<ADDR_W>>;
        let (ch_reader_addr_gen_rsp_s, ch_reader_addr_gen_rsp_r) = chan<()>;
        let (ch_writer_addr_gen_req_s, ch_writer_addr_gen_req_r) = chan<TransferDescBundle<ADDR_W>>;
        let (ch_writer_addr_gen_rsp_s, ch_writer_addr_gen_rsp_r) = chan<()>;

        // Control path
        spawn axi_csr::AxiCsr<ID_W, ADDR_W, DATA_W, STRB_W, REGS_N>(
            ch_axi_ctrl_aw, ch_axi_ctrl_w, ch_axi_ctrl_b, ch_axi_ctrl_ar, ch_axi_ctrl_r,
            ch_writer_start_s, ch_writer_configuration_s, ch_writer_busy_r, ch_writer_done_r,
            ch_reader_start_s, ch_reader_configuration_s, ch_reader_busy_r, ch_reader_done_r,
            reader_sync_req, reader_sync_rsp, writer_sync_req, writer_sync_rsp);

        // Read data path
        spawn address_generator::AddressGenerator<ADDR_W, DATA_W_DIV8>(
            ch_reader_configuration_r, ch_reader_start_r, ch_reader_busy_s, ch_reader_done_s,
            ch_reader_addr_gen_req_s, ch_reader_addr_gen_rsp_r);

        spawn frontend_reader::FrontendReader<ADDR_W, DATA_W, DATA_W_DIV8, DEST_W, ID_W>(
            ch_reader_addr_gen_req_r, ch_reader_addr_gen_rsp_s, ch_axi_data_ar, ch_axi_data_r,
            ch_axi_st_write);

        // Write data_path
        spawn address_generator::AddressGenerator<ADDR_W, DATA_W_DIV8>(
            ch_writer_configuration_r, ch_writer_start_r, ch_writer_busy_s, ch_writer_done_s,
            ch_writer_addr_gen_req_s, ch_writer_addr_gen_rsp_r);

        spawn frontend_writer::FrontendWriter<ADDR_W, DATA_W, DATA_W_DIV8, DEST_W, ID_W, STRB_W>(
            ch_writer_addr_gen_req_r, ch_writer_addr_gen_rsp_s, ch_axi_data_aw, ch_axi_data_w,
            ch_axi_data_b, ch_axi_st_read);

        // TODO: Spawn IRQ CTRL

        (
            ch_axi_ctrl_aw, ch_axi_ctrl_w, ch_axi_ctrl_b, ch_axi_ctrl_ar, ch_axi_ctrl_r,
            ch_axi_data_aw, ch_axi_data_w, ch_axi_data_b, ch_axi_data_ar, ch_axi_data_r,
            ch_axi_st_write, ch_axi_st_read, reader_sync_req, reader_sync_rsp, writer_sync_req,
            writer_sync_rsp,
        )
    }

    init { () }

    next(tok: token, state: ()) {  }
}

// Start processing
// 000011 =  3 - in single mode
// 001111 = 15 - in single mode, disable sync
// 110011 = 51 - in loop mode
// 111111 = 63 - in loop mode, disable sync

const TEST_ADDR_W = u32:32;
const TEST_DATA_W = u32:32;
const TEST_DATA_W_DIV8 = TEST_DATA_W / u32:8;
const TEST_DEST_W = TEST_DATA_W / u32:8;
const TEST_ID_W = TEST_DATA_W / u32:8;
const TEST_REGS_N = u32:14;
const TEST_STRB_W = TEST_DATA_W / u32:8;

type AdrDatPair = (uN[TEST_ADDR_W], uN[TEST_DATA_W]);

// FIXME: Tests interfere with each other!

// #[test_proc]
// proc TestSingleMode {
//     ch_axi_ctrl_aw: chan<AxiAwBundle<TEST_ADDR_W, TEST_ID_W>> out;
//     ch_axi_ctrl_w: chan<AxiWBundle<TEST_DATA_W, TEST_STRB_W>> out;
//     ch_axi_ctrl_b: chan<AxiBBundle<TEST_ID_W>> in;
//     ch_axi_ctrl_ar: chan<AxiArBundle<TEST_ADDR_W, TEST_ID_W>> out;
//     ch_axi_ctrl_r: chan<AxiRBundle<TEST_DATA_W, TEST_ID_W>> in;
//     ch_axi_data_aw: chan<AxiAwBundle<TEST_ADDR_W, TEST_ID_W>> in;
//     ch_axi_data_w: chan<AxiWBundle<TEST_DATA_W, TEST_STRB_W>> in;
//     ch_axi_data_b: chan<AxiBBundle<TEST_ID_W>> out;
//     ch_axi_data_ar: chan<AxiArBundle<TEST_ADDR_W, TEST_ID_W>> in;
//     ch_axi_data_r: chan<AxiRBundle<TEST_DATA_W, TEST_ID_W>> out;
//     reader_sync_req: chan<()> out;
//     reader_sync_rsp: chan<()> in;
//     writer_sync_req: chan<()> out;
//     writer_sync_rsp: chan<()> in;
//     terminator: chan<bool> out;

//     config(terminator: chan<bool> out) {
//         let (ch_axi_ctrl_aw_s, ch_axi_ctrl_aw_r) = chan<AxiAwBundle<TEST_ADDR_W, TEST_ID_W>>;
//         let (ch_axi_ctrl_w_s, ch_axi_ctrl_w_r) = chan<AxiWBundle<TEST_DATA_W, TEST_STRB_W>>;
//         let (ch_axi_ctrl_b_s, ch_axi_ctrl_b_r) = chan<AxiBBundle<TEST_ID_W>>;
//         let (ch_axi_ctrl_ar_s, ch_axi_ctrl_ar_r) = chan<AxiArBundle<TEST_ADDR_W, TEST_ID_W>>;
//         let (ch_axi_ctrl_r_s, ch_axi_ctrl_r_r) = chan<AxiRBundle<TEST_DATA_W, TEST_ID_W>>;
//         let (ch_axi_data_aw_s, ch_axi_data_aw_r) = chan<AxiAwBundle<TEST_ADDR_W, TEST_ID_W>>;
//         let (ch_axi_data_w_s, ch_axi_data_w_r) = chan<AxiWBundle<TEST_DATA_W, TEST_STRB_W>>;
//         let (ch_axi_data_b_s, ch_axi_data_b_r) = chan<AxiBBundle<TEST_ID_W>>;
//         let (ch_axi_data_ar_s, ch_axi_data_ar_r) = chan<AxiArBundle<TEST_ADDR_W, TEST_ID_W>>;
//         let (ch_axi_data_r_s, ch_axi_data_r_r) = chan<AxiRBundle<TEST_DATA_W, TEST_ID_W>>;
//         let (reader_sync_req_s, reader_sync_req_r) = chan<()>;
//         let (reader_sync_rsp_s, reader_sync_rsp_r) = chan<()>;
//         let (writer_sync_req_s, writer_sync_req_r) = chan<()>;
//         let (writer_sync_rsp_s, writer_sync_rsp_r) = chan<()>;

//         let (ch_axi_st_write_s, ch_axi_st_write_r) =
//             chan<AxiStreamBundle<TEST_DATA_W, TEST_DATA_W_DIV8, TEST_DEST_W, TEST_ID_W>>;
//         let (ch_axi_st_read_s, ch_axi_st_read_r) =
//             chan<AxiStreamBundle<TEST_DATA_W, TEST_DATA_W_DIV8, TEST_DEST_W, TEST_ID_W>>;

//         spawn MainController<
//             TEST_ADDR_W, TEST_DATA_W, TEST_DATA_W_DIV8, TEST_DEST_W, TEST_ID_W, TEST_REGS_N,
//             TEST_STRB_W>(
//             ch_axi_ctrl_aw_r, ch_axi_ctrl_w_r, ch_axi_ctrl_b_s, ch_axi_ctrl_ar_r, ch_axi_ctrl_r_s,
//             ch_axi_data_aw_s, ch_axi_data_w_s, ch_axi_data_b_r, ch_axi_data_ar_s, ch_axi_data_r_r,
//             ch_axi_st_write_s, ch_axi_st_read_r,reader_sync_req_r,
//             reader_sync_rsp_s, writer_sync_req_r, writer_sync_rsp_s);

//         spawn gpf::gpf<
//             TEST_DATA_W, TEST_DATA_W_DIV8, TEST_DEST_W, TEST_ID_W, gpf::PfBehavior::INCREMENT>(
//             ch_axi_st_write_r, ch_axi_st_read_s);

//         (
//             ch_axi_ctrl_aw_s, ch_axi_ctrl_w_s, ch_axi_ctrl_b_r, ch_axi_ctrl_ar_s, ch_axi_ctrl_r_r,
//             ch_axi_data_aw_r, ch_axi_data_w_r, ch_axi_data_b_s, ch_axi_data_ar_r,
//             ch_axi_data_r_s,reader_sync_req_s,
//             reader_sync_rsp_r, writer_sync_req_s, writer_sync_rsp_r,
//             terminator,
//         )
//     }

//     init { () }

//     next(tok: token, state: ()) {
//         let CTRL_WORD = uN[TEST_DATA_W]:3;

//         let id = uN[TEST_ID_W]:0;
//         let rw_config = MainCtrlBundle<TEST_ADDR_W> {
//             start_address: u32:0x1000, line_count: u32:5, line_length: u32:6, line_stride: u32:0
//         };
//         let init_csr_values = AdrDatPair[u32:10]:[
//             (config::READER_START_ADDRESS, rw_config.start_address),
//             (config::READER_LINE_LENGTH, rw_config.line_length),
//             (config::READER_LINE_COUNT, rw_config.line_count),
//             (config::READER_STRIDE_BETWEEN_LINES, rw_config.line_stride),
//             (config::WRITER_START_ADDRESS, rw_config.start_address),
//             (config::WRITER_LINE_LENGTH, rw_config.line_length),
//             (config::WRITER_LINE_COUNT, rw_config.line_count),
//             (config::WRITER_STRIDE_BETWEEN_LINES, rw_config.line_stride),
//             (config::INTERRUPT_MASK_REGISTER, uN[TEST_DATA_W]:3),
//             (config::CONTROL_REGISTER, CTRL_WORD),
//         ];

//         for (i, tok): (u32, token) in u32:0..u32:10 {
//             let addr = (init_csr_values[i]).0 << uN[TEST_ADDR_W]:2;
//             let data = (init_csr_values[i]).1;
//             let w = axi_pkg::simpleAxiWBundle<TEST_DATA_W, TEST_STRB_W>(data);
//             let aw = axi_pkg::simpleAxiAwBundle<TEST_ADDR_W, TEST_ID_W>(addr, id);
//             let tok = send(tok, ch_axi_ctrl_aw, aw);
//             let tok = send(tok, ch_axi_ctrl_w, w);
//             let (tok, b_resp) = recv(tok, ch_axi_ctrl_b);
//             assert_eq(b_resp.bresp, axi_pkg::AXI_WRITE_RESPONSE_CODES::OKAY);
//             (tok)
//         }(tok);

//         // Read all values and compare with writes
//         for (i, tok): (u32, token) in u32:0..10 {
//             let addr = (init_csr_values[i]).0 << uN[TEST_ADDR_W]:2;
//             let ar = axi_pkg::simpleAxiArBundle(addr, id, u8:1);
//             let tok = send(tok, ch_axi_ctrl_ar, ar);
//             let (tok, rcv) = recv(tok, ch_axi_ctrl_r);
//             if i != 9 {
//                 assert_eq(rcv.rdata, ((init_csr_values[i]).1) as uN[TEST_DATA_W])
//             } else {};
//             (tok)
//         }(tok);
//         trace_fmt!("[32;1mAXI Control Bus: PASS[0m");

//         if CTRL_WORD[2:4] == u2:0 {
//             // Synchronize to external
//             let tok = send(tok, reader_sync_req, ());
//             let (tok, _) = recv(tok, reader_sync_rsp);

//             let tok = send(tok, writer_sync_req, ());
//             let (tok, _) = recv(tok, writer_sync_rsp);
//         } else {};

//         // Initialize system memory
//         let MEM_SIZE = rw_config.line_count * rw_config.line_length;
//         let system_memory = for (i, system_memory): (u32, uN[TEST_DATA_W][MEM_SIZE]) in
//             u32:0..MEM_SIZE {
//             update(system_memory, i, (i + u32:1) as uN[TEST_DATA_W])
//         }(uN[TEST_DATA_W][MEM_SIZE]:[uN[TEST_DATA_W]:0, ...]);

//         // assert_eq(u32:1, u32:0);

//         let system_memory_copy = for (_, mem): (u32, uN[TEST_DATA_W][MEM_SIZE]) in
//             u32:0..rw_config.line_length {
//             // Handle AXI Read
//             let (tok, axi_ar) = recv(tok, ch_axi_data_ar);
//             let addr = (axi_ar.araddr - rw_config.start_address) >> 2;
//             let tok = for (i, tok): (u32, token) in u32:0..rw_config.line_count {
//                 let tok = send(
//                     tok, ch_axi_data_r,
//                     axi_pkg::simpleAxiRBundle<TEST_DATA_W, TEST_ID_W>(
//                         system_memory[addr + i], id));
//                 tok
//             }(tok);

//             // Handle AXI Write

//             let (tok, aw) = recv(tok, ch_axi_data_aw);

//             let mem = for (i, mem): (u32, uN[TEST_DATA_W][MEM_SIZE]) in
//             u32:0..rw_config.line_count
//             {
//                 let (tok, w) = recv(tok, ch_axi_data_w);
//                 let addr = (aw.awaddr - rw_config.start_address) >> 2;
//                 let mem = update(mem, addr + i, w.wdata);
//                 mem
//             }(mem);

//             let tok = send(tok, ch_axi_data_b, axi_pkg::zeroAxiBBundle<TEST_ID_W>());
//             mem
//         }(uN[TEST_DATA_W][MEM_SIZE]:[uN[TEST_DATA_W]:0, ...]);

//         trace_fmt!("System memory = {}", system_memory);
//         trace_fmt!("System memory copy = {}", system_memory_copy);

//         // TODO: CSR registers are not updated for a few more "next cycles"
//         // I would love a while loop here, otherwise I have to rewrite the whole test
//         // to use `next` as a sort-of while loop
//         for (_, tok): (u32, token) in u32:0..u32:3 {
//             let addr = config::STATUS_REGISTER << uN[TEST_ADDR_W]:2;
//             let ar = axi_pkg::simpleAxiArBundle(addr, id, u8:1);
//             let tok = send(tok, ch_axi_ctrl_ar, ar);
//             let (tok, _) = recv(tok, ch_axi_ctrl_r);
//             (tok)
//         }(tok);

//         // Check Interrupt Status register
//         let addr = config::INTERRUPT_STATUS_REGISTER << uN[TEST_ADDR_W]:2;
//         let ar = axi_pkg::simpleAxiArBundle(addr, id, u8:1);
//         let tok = send(tok, ch_axi_ctrl_ar, ar);
//         let (tok, rcv) = recv(tok, ch_axi_ctrl_r);
//         assert_eq(rcv.rdata, uN[TEST_DATA_W]:3);

//         // Clear interrupts
//         let addr = config::INTERRUPT_STATUS_REGISTER << uN[TEST_ADDR_W]:2;
//         let data = uN[TEST_DATA_W]:0;
//         let w = axi_pkg::simpleAxiWBundle<TEST_DATA_W, TEST_STRB_W>(data);
//         let aw = axi_pkg::simpleAxiAwBundle<TEST_ADDR_W, TEST_ID_W>(addr, id);
//         let tok = send(tok, ch_axi_ctrl_aw, aw);
//         let tok = send(tok, ch_axi_ctrl_w, w);
//         let (tok, b_resp) = recv(tok, ch_axi_ctrl_b);
//         assert_eq(b_resp.bresp, axi_pkg::AXI_WRITE_RESPONSE_CODES::OKAY);

//         trace_fmt!("[32;1m AXI Data Bus: PASS[0m");

//         let tok = send(tok, terminator, true);
//     }
// }

#[test_proc]
proc TestImageInverse {
    ch_axi_ctrl_aw: chan<AxiAwBundle<TEST_ADDR_W, TEST_ID_W>> out;
    ch_axi_ctrl_w: chan<AxiWBundle<TEST_DATA_W, TEST_STRB_W>> out;
    ch_axi_ctrl_b: chan<AxiBBundle<TEST_ID_W>> in;
    ch_axi_ctrl_ar: chan<AxiArBundle<TEST_ADDR_W, TEST_ID_W>> out;
    ch_axi_ctrl_r: chan<AxiRBundle<TEST_DATA_W, TEST_ID_W>> in;
    ch_axi_data_aw: chan<AxiAwBundle<TEST_ADDR_W, TEST_ID_W>> in;
    ch_axi_data_w: chan<AxiWBundle<TEST_DATA_W, TEST_STRB_W>> in;
    ch_axi_data_b: chan<AxiBBundle<TEST_ID_W>> out;
    ch_axi_data_ar: chan<AxiArBundle<TEST_ADDR_W, TEST_ID_W>> in;
    ch_axi_data_r: chan<AxiRBundle<TEST_DATA_W, TEST_ID_W>> out;
    reader_sync_req: chan<()> out;
    reader_sync_rsp: chan<()> in;
    writer_sync_req: chan<()> out;
    writer_sync_rsp: chan<()> in;
    terminator: chan<bool> out;

    config(terminator: chan<bool> out) {
        let (ch_axi_ctrl_aw_s, ch_axi_ctrl_aw_r) = chan<AxiAwBundle<TEST_ADDR_W, TEST_ID_W>>;
        let (ch_axi_ctrl_w_s, ch_axi_ctrl_w_r) = chan<AxiWBundle<TEST_DATA_W, TEST_STRB_W>>;
        let (ch_axi_ctrl_b_s, ch_axi_ctrl_b_r) = chan<AxiBBundle<TEST_ID_W>>;
        let (ch_axi_ctrl_ar_s, ch_axi_ctrl_ar_r) = chan<AxiArBundle<TEST_ADDR_W, TEST_ID_W>>;
        let (ch_axi_ctrl_r_s, ch_axi_ctrl_r_r) = chan<AxiRBundle<TEST_DATA_W, TEST_ID_W>>;
        let (ch_axi_data_aw_s, ch_axi_data_aw_r) = chan<AxiAwBundle<TEST_ADDR_W, TEST_ID_W>>;
        let (ch_axi_data_w_s, ch_axi_data_w_r) = chan<AxiWBundle<TEST_DATA_W, TEST_STRB_W>>;
        let (ch_axi_data_b_s, ch_axi_data_b_r) = chan<AxiBBundle<TEST_ID_W>>;
        let (ch_axi_data_ar_s, ch_axi_data_ar_r) = chan<AxiArBundle<TEST_ADDR_W, TEST_ID_W>>;
        let (ch_axi_data_r_s, ch_axi_data_r_r) = chan<AxiRBundle<TEST_DATA_W, TEST_ID_W>>;
        let (reader_sync_req_s, reader_sync_req_r) = chan<()>;
        let (reader_sync_rsp_s, reader_sync_rsp_r) = chan<()>;
        let (writer_sync_req_s, writer_sync_req_r) = chan<()>;
        let (writer_sync_rsp_s, writer_sync_rsp_r) = chan<()>;

        let (ch_axi_st_write_s, ch_axi_st_write_r) =
            chan<AxiStreamBundle<TEST_DATA_W, TEST_DATA_W_DIV8, TEST_DEST_W, TEST_ID_W>>;
        let (ch_axi_st_read_s, ch_axi_st_read_r) =
            chan<AxiStreamBundle<TEST_DATA_W, TEST_DATA_W_DIV8, TEST_DEST_W, TEST_ID_W>>;

        spawn MainController<
            TEST_ADDR_W, TEST_DATA_W, TEST_DATA_W_DIV8, TEST_DEST_W, TEST_ID_W, TEST_REGS_N, TEST_STRB_W>(
            ch_axi_ctrl_aw_r, ch_axi_ctrl_w_r, ch_axi_ctrl_b_s, ch_axi_ctrl_ar_r, ch_axi_ctrl_r_s,
            ch_axi_data_aw_s, ch_axi_data_w_s, ch_axi_data_b_r, ch_axi_data_ar_s, ch_axi_data_r_r,
            ch_axi_st_write_s, ch_axi_st_read_r, reader_sync_req_r, reader_sync_rsp_s,
            writer_sync_req_r, writer_sync_rsp_s);

        spawn gpf::gpf<
            TEST_DATA_W, TEST_DATA_W_DIV8, TEST_DEST_W, TEST_ID_W, gpf::PfBehavior::INVERT>(
            ch_axi_st_write_r, ch_axi_st_read_s);

        (
            ch_axi_ctrl_aw_s, ch_axi_ctrl_w_s, ch_axi_ctrl_b_r, ch_axi_ctrl_ar_s, ch_axi_ctrl_r_r,
            ch_axi_data_aw_r, ch_axi_data_w_r, ch_axi_data_b_s, ch_axi_data_ar_r, ch_axi_data_r_s,
            reader_sync_req_s, reader_sync_rsp_r, writer_sync_req_s, writer_sync_rsp_r, terminator,
        )
    }

    init { () }

    next(tok: token, state: ()) {
        let CTRL_WORD = uN[TEST_DATA_W]:15;

        let id = uN[TEST_ID_W]:0;
        let rw_config = MainCtrlBundle<TEST_ADDR_W> {
            start_address: u32:0x1000, line_count: u32:27, line_length: u32:1, line_stride: u32:0
        };
        let init_csr_values = AdrDatPair[u32:10]:[
            (config::READER_START_ADDRESS, rw_config.start_address),
            (config::READER_LINE_LENGTH, rw_config.line_length),
            (config::READER_LINE_COUNT, rw_config.line_count),
            (config::READER_STRIDE_BETWEEN_LINES, rw_config.line_stride),
            (config::WRITER_START_ADDRESS, rw_config.start_address),
            (config::WRITER_LINE_LENGTH, rw_config.line_length),
            (config::WRITER_LINE_COUNT, rw_config.line_count),
            (config::WRITER_STRIDE_BETWEEN_LINES, rw_config.line_stride),
            (config::INTERRUPT_MASK_REGISTER, uN[TEST_DATA_W]:3),
            (config::CONTROL_REGISTER, CTRL_WORD),
        ];

        for (i, tok): (u32, token) in u32:0..u32:10 {
            let addr = (init_csr_values[i]).0 << uN[TEST_ADDR_W]:2;
            let data = (init_csr_values[i]).1;
            let w = axi_pkg::simpleAxiWBundle<TEST_DATA_W, TEST_STRB_W>(data);
            let aw = axi_pkg::simpleAxiAwBundle<TEST_ADDR_W, TEST_ID_W>(addr, id);
            let tok = send(tok, ch_axi_ctrl_aw, aw);
            let tok = send(tok, ch_axi_ctrl_w, w);
            let (tok, b_resp) = recv(tok, ch_axi_ctrl_b);
            assert_eq(b_resp.bresp, axi_pkg::AXI_WRITE_RESPONSE_CODES::OKAY);
            (tok)
        }(tok);

        // Read all values and compare with writes
        for (i, tok): (u32, token) in u32:0..10 {
            let addr = (init_csr_values[i]).0 << uN[TEST_ADDR_W]:2;
            let ar = axi_pkg::simpleAxiArBundle(addr, id, u8:1);
            let tok = send(tok, ch_axi_ctrl_ar, ar);
            let (tok, rcv) = recv(tok, ch_axi_ctrl_r);
            if i != 9 {
                assert_eq(rcv.rdata, ((init_csr_values[i]).1) as uN[TEST_DATA_W]);
            } else {


            };
            (tok)
        }(tok);
        trace_fmt!("[32;1mAXI Control Bus: PASS[0m");

        // Initialize system memory
        let MEM_SIZE = rw_config.line_count * rw_config.line_length;
        let system_memory = uN[TEST_DATA_W][MEM_SIZE]:[
            uN[TEST_DATA_W]:0x00, uN[TEST_DATA_W]:0x00, uN[TEST_DATA_W]:0x00, uN[TEST_DATA_W]:0x82,
            uN[TEST_DATA_W]:0x04, uN[TEST_DATA_W]:0x7e, uN[TEST_DATA_W]:0x44, uN[TEST_DATA_W]:0x04,
            uN[TEST_DATA_W]:0x02, uN[TEST_DATA_W]:0x28, uN[TEST_DATA_W]:0x04, uN[TEST_DATA_W]:0x02,
            uN[TEST_DATA_W]:0x10, uN[TEST_DATA_W]:0x04, uN[TEST_DATA_W]:0x7e, uN[TEST_DATA_W]:0x28,
            uN[TEST_DATA_W]:0x04, uN[TEST_DATA_W]:0x40, uN[TEST_DATA_W]:0x44, uN[TEST_DATA_W]:0x04,
            uN[TEST_DATA_W]:0x40, uN[TEST_DATA_W]:0x82, uN[TEST_DATA_W]:0x7c, uN[TEST_DATA_W]:0x7e,
            uN[TEST_DATA_W]:0x00, uN[TEST_DATA_W]:0x00, uN[TEST_DATA_W]:0x00,
        ];

        let system_memory_copy = for (_, mem): (u32, uN[TEST_DATA_W][MEM_SIZE]) in
            u32:0..rw_config.line_length {
            // Handle AXI Read
            let (tok, axi_ar) = recv(tok, ch_axi_data_ar);
            let addr = (axi_ar.araddr - rw_config.start_address) >> 2;
            let tok = for (i, tok): (u32, token) in u32:0..rw_config.line_count {
                let tok = send(
                    tok, ch_axi_data_r,
                    axi_pkg::simpleAxiRBundle<TEST_DATA_W, TEST_ID_W>(
                        system_memory[addr + i], id));
                tok
            }(tok);

            // Handle AXI Write
            let (tok, aw) = recv(tok, ch_axi_data_aw);
            let mem = for (i, mem): (u32, uN[TEST_DATA_W][MEM_SIZE]) in
                u32:0..rw_config.line_count {
                let (tok, w) = recv(tok, ch_axi_data_w);
                let addr = (aw.awaddr - rw_config.start_address) >> 2;
                let mem = update(mem, addr + i, w.wdata);
                mem
            }(mem);

            let tok = send(tok, ch_axi_data_b, axi_pkg::zeroAxiBBundle<TEST_ID_W>());
            mem
        }(uN[TEST_DATA_W][MEM_SIZE]:[uN[TEST_DATA_W]:0, ...]);

        trace_fmt!("System memory = {:x}", system_memory);
        trace_fmt!("System memory copy = {:x}", system_memory_copy);
        let golden_data = uN[TEST_DATA_W][MEM_SIZE]:[
            uN[TEST_DATA_W]:0xff, uN[TEST_DATA_W]:0xff, uN[TEST_DATA_W]:0xff, uN[TEST_DATA_W]:0x7d,
            uN[TEST_DATA_W]:0xfb, uN[TEST_DATA_W]:0x81, uN[TEST_DATA_W]:0xbb, uN[TEST_DATA_W]:0xfb,
            uN[TEST_DATA_W]:0xfd, uN[TEST_DATA_W]:0xd7, uN[TEST_DATA_W]:0xfb, uN[TEST_DATA_W]:0xfd,
            uN[TEST_DATA_W]:0xef, uN[TEST_DATA_W]:0xfb, uN[TEST_DATA_W]:0x81, uN[TEST_DATA_W]:0xd7,
            uN[TEST_DATA_W]:0xfb, uN[TEST_DATA_W]:0xbf, uN[TEST_DATA_W]:0xbb, uN[TEST_DATA_W]:0xfb,
            uN[TEST_DATA_W]:0xbf, uN[TEST_DATA_W]:0x7d, uN[TEST_DATA_W]:0x83, uN[TEST_DATA_W]:0x81,
            uN[TEST_DATA_W]:0xff, uN[TEST_DATA_W]:0xff, uN[TEST_DATA_W]:0xff,
        ];
        assert_eq(golden_data, system_memory_copy);
        trace_fmt!("[32;1m AXI Data Bus: PASS[0m");

        let tok = send(tok, terminator, true);
    }
}

// #[test_proc]
// proc TestLoopMode {
//     ch_axi_ctrl_aw: chan<AxiAwBundle<TEST_ADDR_W, TEST_ID_W>> out;
//     ch_axi_ctrl_w: chan<AxiWBundle<TEST_DATA_W, TEST_STRB_W>> out;
//     ch_axi_ctrl_b: chan<AxiBBundle<TEST_ID_W>> in;
//     ch_axi_ctrl_ar: chan<AxiArBundle<TEST_ADDR_W, TEST_ID_W>> out;
//     ch_axi_ctrl_r: chan<AxiRBundle<TEST_DATA_W, TEST_ID_W>> in;
//     ch_axi_data_aw: chan<AxiAwBundle<TEST_ADDR_W, TEST_ID_W>> in;
//     ch_axi_data_w: chan<AxiWBundle<TEST_DATA_W, TEST_STRB_W>> in;
//     ch_axi_data_b: chan<AxiBBundle<TEST_ID_W>> out;
//     ch_axi_data_ar: chan<AxiArBundle<TEST_ADDR_W, TEST_ID_W>> in;
//     ch_axi_data_r: chan<AxiRBundle<TEST_DATA_W, TEST_ID_W>> out;
//     reader_sync_req: chan<()> out;
//     reader_sync_rsp: chan<()> in;
//     writer_sync_req: chan<()> out;
//     writer_sync_rsp: chan<()> in;
//     terminator: chan<bool> out;

//     config(terminator: chan<bool> out) {
//         let (ch_axi_ctrl_aw_s, ch_axi_ctrl_aw_r) = chan<AxiAwBundle<TEST_ADDR_W, TEST_ID_W>>;
//         let (ch_axi_ctrl_w_s, ch_axi_ctrl_w_r) = chan<AxiWBundle<TEST_DATA_W, TEST_STRB_W>>;
//         let (ch_axi_ctrl_b_s, ch_axi_ctrl_b_r) = chan<AxiBBundle<TEST_ID_W>>;
//         let (ch_axi_ctrl_ar_s, ch_axi_ctrl_ar_r) = chan<AxiArBundle<TEST_ADDR_W, TEST_ID_W>>;
//         let (ch_axi_ctrl_r_s, ch_axi_ctrl_r_r) = chan<AxiRBundle<TEST_DATA_W, TEST_ID_W>>;
//         let (ch_axi_data_aw_s, ch_axi_data_aw_r) = chan<AxiAwBundle<TEST_ADDR_W, TEST_ID_W>>;
//         let (ch_axi_data_w_s, ch_axi_data_w_r) = chan<AxiWBundle<TEST_DATA_W, TEST_STRB_W>>;
//         let (ch_axi_data_b_s, ch_axi_data_b_r) = chan<AxiBBundle<TEST_ID_W>>;
//         let (ch_axi_data_ar_s, ch_axi_data_ar_r) = chan<AxiArBundle<TEST_ADDR_W, TEST_ID_W>>;
//         let (ch_axi_data_r_s, ch_axi_data_r_r) = chan<AxiRBundle<TEST_DATA_W, TEST_ID_W>>;
//         let (reader_sync_req_s, reader_sync_req_r) = chan<()>;
//         let (reader_sync_rsp_s, reader_sync_rsp_r) = chan<()>;
//         let (writer_sync_req_s, writer_sync_req_r) = chan<()>;
//         let (writer_sync_rsp_s, writer_sync_rsp_r) = chan<()>;

//         let (ch_axi_st_write_s, ch_axi_st_write_r) =
//             chan<AxiStreamBundle<TEST_DATA_W, TEST_DATA_W_DIV8, TEST_DEST_W, TEST_ID_W>>;
//         let (ch_axi_st_read_s, ch_axi_st_read_r) =
//             chan<AxiStreamBundle<TEST_DATA_W, TEST_DATA_W_DIV8, TEST_DEST_W, TEST_ID_W>>;

//         spawn MainController<
//             TEST_ADDR_W, TEST_DATA_W, TEST_DATA_W_DIV8, TEST_DEST_W, TEST_ID_W, TEST_REGS_N,
//             TEST_STRB_W>(
//             ch_axi_ctrl_aw_r, ch_axi_ctrl_w_r, ch_axi_ctrl_b_s, ch_axi_ctrl_ar_r, ch_axi_ctrl_r_s,
//             ch_axi_data_aw_s, ch_axi_data_w_s, ch_axi_data_b_r, ch_axi_data_ar_s, ch_axi_data_r_r,
//             ch_axi_st_write_s, ch_axi_st_read_r,reader_sync_req_r,
//             reader_sync_rsp_s, writer_sync_req_r, writer_sync_rsp_s);

//         spawn gpf::gpf<
//             TEST_DATA_W, TEST_DATA_W_DIV8, TEST_DEST_W, TEST_ID_W, gpf::PfBehavior::INVERT>(
//             ch_axi_st_write_r, ch_axi_st_read_s);

//         (
//             ch_axi_ctrl_aw_s, ch_axi_ctrl_w_s, ch_axi_ctrl_b_r, ch_axi_ctrl_ar_s, ch_axi_ctrl_r_r,
//             ch_axi_data_aw_r, ch_axi_data_w_r, ch_axi_data_b_s, ch_axi_data_ar_r,
//             ch_axi_data_r_s,reader_sync_req_s,
//             reader_sync_rsp_r, writer_sync_req_s, writer_sync_rsp_r,
//             terminator,
//         )
//     }

//     init { () }

//     next(tok: token, state: ()) {
//         let id = uN[TEST_ID_W]:0;
//         let rw_config = MainCtrlBundle<TEST_ADDR_W> {
//             start_address: u32:0x1000, line_count: u32:27, line_length: u32:1, line_stride: u32:0
//         };
//         let init_csr_values = AdrDatPair[u32:10]:[
//             (config::READER_START_ADDRESS, rw_config.start_address),
//             (config::READER_LINE_LENGTH, rw_config.line_length),
//             (config::READER_LINE_COUNT, rw_config.line_count),
//             (config::READER_STRIDE_BETWEEN_LINES, rw_config.line_stride),
//             (config::WRITER_START_ADDRESS, rw_config.start_address),
//             (config::WRITER_LINE_LENGTH, rw_config.line_length),
//             (config::WRITER_LINE_COUNT, rw_config.line_count),
//             (config::WRITER_STRIDE_BETWEEN_LINES, rw_config.line_stride),
//             (config::INTERRUPT_MASK_REGISTER, uN[TEST_DATA_W]:3),
//             (config::CONTROL_REGISTER, uN[TEST_DATA_W]:63),
//         ];

//         for (i, tok): (u32, token) in u32:0..u32:10 {
//             let addr = (init_csr_values[i]).0 << uN[TEST_ADDR_W]:2;
//             let data = (init_csr_values[i]).1;
//             let w = axi_pkg::simpleAxiWBundle<TEST_DATA_W, TEST_STRB_W>(data);
//             let aw = axi_pkg::simpleAxiAwBundle<TEST_ADDR_W, TEST_ID_W>(addr, id);
//             let tok = send(tok, ch_axi_ctrl_aw, aw);
//             let tok = send(tok, ch_axi_ctrl_w, w);
//             let (tok, b_resp) = recv(tok, ch_axi_ctrl_b);
//             assert_eq(b_resp.bresp, axi_pkg::AXI_WRITE_RESPONSE_CODES::OKAY);
//             (tok)
//         }(tok);

//         // Read all values and compare with writes
//         for (i, tok): (u32, token) in u32:0..10 {
//             let addr = (init_csr_values[i]).0 << uN[TEST_ADDR_W]:2;
//             let ar = axi_pkg::simpleAxiArBundle(addr, id, u8:1);
//             let tok = send(tok, ch_axi_ctrl_ar, ar);
//             let (tok, rcv) = recv(tok, ch_axi_ctrl_r);
//             assert_eq(rcv.rdata, ((init_csr_values[i]).1) as uN[TEST_DATA_W]);
//             (tok)
//         }(tok);
//         trace_fmt!("[32;1mAXI Control Bus: PASS[0m");

//         for (_, tok): (u32, token) in u32:0..u32:3 {
//             // Synchronize to external
//             let tok = send(tok, reader_sync_req, ());
//             let (tok, _) = recv(tok, reader_sync_rsp);

//             let tok = send(tok, writer_sync_req, ());
//             let (tok, _) = recv(tok, writer_sync_rsp);

//             // Initialize system memory
//             let MEM_SIZE = rw_config.line_count * rw_config.line_length;
//             let system_memory = uN[TEST_DATA_W][MEM_SIZE]:[
//                 uN[TEST_DATA_W]:0x00, uN[TEST_DATA_W]:0x00, uN[TEST_DATA_W]:0x00,
//                 uN[TEST_DATA_W]:0x82,
//                 uN[TEST_DATA_W]:0x04, uN[TEST_DATA_W]:0x7e, uN[TEST_DATA_W]:0x44,
//                 uN[TEST_DATA_W]:0x04,
//                 uN[TEST_DATA_W]:0x02, uN[TEST_DATA_W]:0x28, uN[TEST_DATA_W]:0x04,
//                 uN[TEST_DATA_W]:0x02,
//                 uN[TEST_DATA_W]:0x10, uN[TEST_DATA_W]:0x04, uN[TEST_DATA_W]:0x7e,
//                 uN[TEST_DATA_W]:0x28,
//                 uN[TEST_DATA_W]:0x04, uN[TEST_DATA_W]:0x40, uN[TEST_DATA_W]:0x44,
//                 uN[TEST_DATA_W]:0x04,
//                 uN[TEST_DATA_W]:0x40, uN[TEST_DATA_W]:0x82, uN[TEST_DATA_W]:0x7c,
//                 uN[TEST_DATA_W]:0x7e,
//                 uN[TEST_DATA_W]:0x00, uN[TEST_DATA_W]:0x00, uN[TEST_DATA_W]:0x00,
//             ];

//             let system_memory_copy = for (_, mem): (u32, uN[TEST_DATA_W][MEM_SIZE]) in
//                 u32:0..rw_config.line_length {
//                 // Handle AXI Read
//                 let (tok, axi_ar) = recv(tok, ch_axi_data_ar);
//                 let addr = (axi_ar.araddr - rw_config.start_address) >> 2;
//                 let tok = for (i, tok): (u32, token) in u32:0..rw_config.line_count {
//                     let tok = send(
//                         tok, ch_axi_data_r,
//                         axi_pkg::simpleAxiRBundle<TEST_DATA_W, TEST_ID_W>(
//                             system_memory[addr + i], id));
//                     tok
//                 }(tok);

//                 // Handle AXI Write
//                 let (tok, aw) = recv(tok, ch_axi_data_aw);
//                 let mem = for (i, mem): (u32, uN[TEST_DATA_W][MEM_SIZE]) in
//                 u32:0..rw_config.line_count {
//                     let (tok, w) = recv(tok, ch_axi_data_w);
//                     let addr = (aw.awaddr - rw_config.start_address) >> 2;
//                     let mem = update(mem, addr + i, w.wdata);
//                     mem
//                 }(mem);

//                 let tok = send(tok, ch_axi_data_b, axi_pkg::zeroAxiBBundle<TEST_ID_W>());
//                 mem
//             }(uN[TEST_DATA_W][MEM_SIZE]:[uN[TEST_DATA_W]:0, ...]);

//             trace_fmt!("System memory = {:x}", system_memory);
//             trace_fmt!("System memory copy = {:x}", system_memory_copy);
//             let golden_data = uN[TEST_DATA_W][MEM_SIZE]:[
//                 uN[TEST_DATA_W]:0xff, uN[TEST_DATA_W]:0xff, uN[TEST_DATA_W]:0xff,
//                 uN[TEST_DATA_W]:0x7d,
//                 uN[TEST_DATA_W]:0xfb, uN[TEST_DATA_W]:0x81, uN[TEST_DATA_W]:0xbb,
//                 uN[TEST_DATA_W]:0xfb,
//                 uN[TEST_DATA_W]:0xfd, uN[TEST_DATA_W]:0xd7, uN[TEST_DATA_W]:0xfb,
//                 uN[TEST_DATA_W]:0xfd,
//                 uN[TEST_DATA_W]:0xef, uN[TEST_DATA_W]:0xfb, uN[TEST_DATA_W]:0x81,
//                 uN[TEST_DATA_W]:0xd7,
//                 uN[TEST_DATA_W]:0xfb, uN[TEST_DATA_W]:0xbf, uN[TEST_DATA_W]:0xbb,
//                 uN[TEST_DATA_W]:0xfb,
//                 uN[TEST_DATA_W]:0xbf, uN[TEST_DATA_W]:0x7d, uN[TEST_DATA_W]:0x83,
//                 uN[TEST_DATA_W]:0x81,
//                 uN[TEST_DATA_W]:0xff, uN[TEST_DATA_W]:0xff, uN[TEST_DATA_W]:0xff,
//             ];
//             assert_eq(golden_data, system_memory_copy);
//             trace_fmt!("[32;1m AXI Data Bus: PASS[0m");
//             (tok)
//         }(tok);
//         let tok = send(tok, terminator, true);
//     }
// }

proc main_controller {
    config(ch_axi_ctrl_aw: chan<AxiAwBundle<config::TOP_ADDR_W, config::TOP_ID_W>> in,
           ch_axi_ctrl_w: chan<AxiWBundle<config::TOP_DATA_W, config::TOP_STRB_W>> in,
           ch_axi_ctrl_b: chan<AxiBBundle<config::TOP_ID_W>> out,
           ch_axi_ctrl_ar: chan<AxiArBundle<config::TOP_ADDR_W, config::TOP_ID_W>> in,
           ch_axi_ctrl_r: chan<AxiRBundle<config::TOP_DATA_W, config::TOP_ID_W>> out,
           ch_axi_data_aw: chan<AxiAwBundle<config::TOP_ADDR_W, config::TOP_ID_W>> out,
           ch_axi_data_w: chan<AxiWBundle<config::TOP_DATA_W, config::TOP_STRB_W>> out,
           ch_axi_data_b: chan<AxiBBundle<config::TOP_ID_W>> in,
           ch_axi_data_ar: chan<AxiArBundle<config::TOP_ADDR_W, config::TOP_ID_W>> out,
           ch_axi_data_r: chan<AxiRBundle<config::TOP_DATA_W, config::TOP_ID_W>> in,
           ch_axi_st_write: chan<AxiStreamBundle<config::TOP_DATA_W, config::TOP_DATA_W_DIV8, config::TOP_DEST_W, config::TOP_ID_W>> out,
           ch_axi_st_read: chan<AxiStreamBundle<config::TOP_DATA_W, config::TOP_DATA_W_DIV8, config::TOP_DEST_W, config::TOP_ID_W>> in,
           reader_sync_req: chan<()> in, reader_sync_rsp: chan<()> out,
           writer_sync_req: chan<()> in, writer_sync_rsp: chan<()> out) {

        spawn MainController<
            config::TOP_ADDR_W, config::TOP_DATA_W, config::TOP_DATA_W_DIV8, config::TOP_DEST_W, config::TOP_ID_W, config::TOP_REGS_N, config::TOP_STRB_W>(
            ch_axi_ctrl_aw, ch_axi_ctrl_w, ch_axi_ctrl_b, ch_axi_ctrl_ar, ch_axi_ctrl_r,
            ch_axi_data_aw, ch_axi_data_w, ch_axi_data_b, ch_axi_data_ar, ch_axi_data_r,
            ch_axi_st_write, ch_axi_st_read, reader_sync_req, reader_sync_rsp, writer_sync_req,
            writer_sync_rsp);
        ()
    }

    init { () }

    next(tok: token, state: ()) {  }
}
