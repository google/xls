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
import xls.examples.ram;
import xls.modules.zstd.common;
import xls.modules.zstd.memory.axi;
import xls.modules.zstd.csr_config;
import xls.modules.zstd.sequence_executor;
import xls.modules.zstd.zstd_frame_testcases;
import xls.modules.zstd.memory.axi_ram;
import xls.modules.zstd.zstd_dec;

const TEST_WINDOW_LOG_MAX = u32:30;

const TEST_AXI_DATA_W = u32:64;
const TEST_AXI_ADDR_W = u32:32;
const TEST_AXI_ID_W = u32:8;
const TEST_AXI_DEST_W = u32:8;
const TEST_AXI_DATA_W_DIV8 = TEST_AXI_DATA_W / u32:8;

const TEST_REGS_N = u32:5;
const TEST_LOG2_REGS_N = std::clog2(TEST_REGS_N);

const TEST_HB_RAM_N = u32:8;
const TEST_HB_ADDR_W = sequence_executor::ZSTD_RAM_ADDR_WIDTH;
const TEST_HB_DATA_W = sequence_executor::RAM_DATA_WIDTH;
const TEST_HB_NUM_PARTITIONS = sequence_executor::RAM_NUM_PARTITIONS;
const TEST_HB_SIZE_KB = sequence_executor::ZSTD_HISTORY_BUFFER_SIZE_KB;
const TEST_HB_RAM_SIZE = sequence_executor::ZSTD_RAM_SIZE;
const TEST_HB_RAM_WORD_PARTITION_SIZE = sequence_executor::RAM_WORD_PARTITION_SIZE;
const TEST_HB_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR = sequence_executor::TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR;
const TEST_HB_RAM_INITIALIZED = sequence_executor::TEST_RAM_INITIALIZED;
const TEST_HB_RAM_ASSERT_VALID_READ:bool = false;

const TEST_RAM_DATA_W:u32 = TEST_AXI_DATA_W;
const TEST_RAM_SIZE:u32 = u32:16384;
const TEST_RAM_ADDR_W:u32 = std::clog2(TEST_RAM_SIZE);
const TEST_RAM_WORD_PARTITION_SIZE:u32 = u32:8;
const TEST_RAM_NUM_PARTITIONS:u32 = ram::num_partitions(TEST_RAM_WORD_PARTITION_SIZE, TEST_RAM_DATA_W);
const TEST_RAM_BASE_ADDR:u32 = u32:0;
const TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR = ram::SimultaneousReadWriteBehavior::READ_BEFORE_WRITE;
const TEST_RAM_INITIALIZED = true;

const TEST_MOCK_OUTPUT_RAM_SIZE:u32 = TEST_RAM_SIZE / TEST_AXI_DATA_W_DIV8;

fn csr_addr(c: zstd_dec::Csr) -> uN[TEST_AXI_ADDR_W] {
    (c as uN[TEST_AXI_ADDR_W]) << 3
}

#[test_proc]
proc ZstdDecoderTest {
    type CsrAxiAr = axi::AxiAr<TEST_AXI_ADDR_W, TEST_AXI_ID_W>;
    type CsrAxiR = axi::AxiR<TEST_AXI_DATA_W, TEST_AXI_ID_W>;
    type CsrAxiAw = axi::AxiAw<TEST_AXI_ADDR_W, TEST_AXI_ID_W>;
    type CsrAxiW = axi::AxiW<TEST_AXI_DATA_W, TEST_AXI_DATA_W_DIV8>;
    type CsrAxiB = axi::AxiB<TEST_AXI_ID_W>;

    type CsrRdReq = csr_config::CsrRdReq<TEST_LOG2_REGS_N>;
    type CsrRdResp = csr_config::CsrRdResp<TEST_LOG2_REGS_N, TEST_AXI_DATA_W>;
    type CsrWrReq = csr_config::CsrWrReq<TEST_LOG2_REGS_N, TEST_AXI_DATA_W>;
    type CsrWrResp = csr_config::CsrWrResp;
    type CsrChange = csr_config::CsrChange<TEST_LOG2_REGS_N>;

    type MemAxiAr = axi::AxiAr<TEST_AXI_ADDR_W, TEST_AXI_ID_W>;
    type MemAxiR = axi::AxiR<TEST_AXI_DATA_W, TEST_AXI_ID_W>;
    type MemAxiAw = axi::AxiAw<TEST_AXI_ADDR_W, TEST_AXI_ID_W>;
    type MemAxiW = axi::AxiW<TEST_AXI_DATA_W, TEST_AXI_DATA_W_DIV8>;
    type MemAxiB = axi::AxiB<TEST_AXI_ID_W>;

    type RamRdReqHB = ram::ReadReq<TEST_HB_ADDR_W, TEST_HB_NUM_PARTITIONS>;
    type RamRdRespHB = ram::ReadResp<TEST_HB_DATA_W>;
    type RamWrReqHB = ram::WriteReq<TEST_HB_ADDR_W, TEST_HB_DATA_W, TEST_HB_NUM_PARTITIONS>;
    type RamWrRespHB = ram::WriteResp;

    type RamRdReq = ram::ReadReq<TEST_RAM_ADDR_W, TEST_RAM_NUM_PARTITIONS>;
    type RamRdResp = ram::ReadResp<TEST_RAM_DATA_W>;
    type RamWrReq = ram::WriteReq<TEST_RAM_ADDR_W, TEST_RAM_DATA_W, TEST_RAM_NUM_PARTITIONS>;
    type RamWrResp = ram::WriteResp;

    type ZstdDecodedPacket = common::ZstdDecodedPacket;
    terminator: chan<bool> out;
    csr_axi_aw_s: chan<CsrAxiAw> out;
    csr_axi_w_s: chan<CsrAxiW> out;
    csr_axi_b_r: chan<CsrAxiB> in;
    csr_axi_ar_s: chan<CsrAxiAr> out;
    csr_axi_r_r: chan<CsrAxiR> in;
    fh_axi_ar_r: chan<MemAxiAr> in;
    fh_axi_r_s: chan<MemAxiR> out;
    bh_axi_ar_r: chan<MemAxiAr> in;
    bh_axi_r_s: chan<MemAxiR> out;
    raw_axi_ar_r: chan<MemAxiAr> in;
    raw_axi_r_s: chan<MemAxiR> out;
    output_axi_aw_r: chan<MemAxiAw> in;
    output_axi_w_r: chan<MemAxiW> in;
    output_axi_b_s: chan<MemAxiB> out;

    ram_rd_req_r: chan<RamRdReqHB>[8] in;
    ram_rd_resp_s: chan<RamRdRespHB>[8] out;
    ram_wr_req_r: chan<RamWrReqHB>[8] in;
    ram_wr_resp_s: chan<RamWrRespHB>[8] out;

    ram_wr_req_fh_s: chan<RamWrReq> out;
    ram_wr_req_bh_s: chan<RamWrReq> out;
    ram_wr_req_raw_s: chan<RamWrReq> out;
    raw_wr_resp_fh_r: chan<RamWrResp> in;
    raw_wr_resp_bh_r: chan<RamWrResp> in;
    raw_wr_resp_raw_r: chan<RamWrResp> in;

    output_r: chan<ZstdDecodedPacket> in;
    notify_r: chan<()> in;
    reset_r: chan<()> in;

    init {}

    config(terminator: chan<bool> out) {

        let (csr_axi_aw_s, csr_axi_aw_r) = chan<CsrAxiAw>("csr_axi_aw");
        let (csr_axi_w_s, csr_axi_w_r) = chan<CsrAxiW>("csr_axi_w");
        let (csr_axi_b_s, csr_axi_b_r) = chan<CsrAxiB>("csr_axi_b");
        let (csr_axi_ar_s, csr_axi_ar_r) = chan<CsrAxiAr>("csr_axi_ar");
        let (csr_axi_r_s, csr_axi_r_r) = chan<CsrAxiR>("csr_axi_r");

        let (fh_axi_ar_s, fh_axi_ar_r) = chan<MemAxiAr>("fh_axi_ar");
        let (fh_axi_r_s, fh_axi_r_r) = chan<MemAxiR>("fh_axi_r");

        let (bh_axi_ar_s, bh_axi_ar_r) = chan<MemAxiAr>("bh_axi_ar");
        let (bh_axi_r_s, bh_axi_r_r) = chan<MemAxiR>("bh_axi_r");

        let (raw_axi_ar_s, raw_axi_ar_r) = chan<MemAxiAr>("raw_axi_ar");
        let (raw_axi_r_s, raw_axi_r_r) = chan<MemAxiR>("raw_axi_r");

        let (output_axi_aw_s, output_axi_aw_r) = chan<MemAxiAw>("output_axi_aw");
        let (output_axi_w_s, output_axi_w_r) = chan<MemAxiW>("output_axi_w");
        let (output_axi_b_s, output_axi_b_r) = chan<MemAxiB>("output_axi_b");

        let (ram_rd_req_s, ram_rd_req_r) = chan<RamRdReqHB>[8]("ram_rd_req");
        let (ram_rd_resp_s, ram_rd_resp_r) = chan<RamRdRespHB>[8]("ram_rd_resp");
        let (ram_wr_req_s, ram_wr_req_r) = chan<RamWrReqHB>[8]("ram_wr_req");
        let (ram_wr_resp_s, ram_wr_resp_r) = chan<RamWrRespHB>[8]("ram_wr_resp");

        let (ram_rd_req_fh_s, ram_rd_req_fh_r) = chan<RamRdReq>("ram_rd_req_fh");
        let (ram_rd_req_bh_s, ram_rd_req_bh_r) = chan<RamRdReq>("ram_rd_req_bh");
        let (ram_rd_req_raw_s, ram_rd_req_raw_r) = chan<RamRdReq>("ram_rd_req_raw");
        let (ram_rd_resp_fh_s, ram_rd_resp_fh_r) = chan<RamRdResp>("ram_rd_resp_fh");
        let (ram_rd_resp_bh_s, ram_rd_resp_bh_r) = chan<RamRdResp>("ram_rd_resp_bh");
        let (ram_rd_resp_raw_s, ram_rd_resp_raw_r) = chan<RamRdResp>("ram_rd_resp_raw");

        let (ram_wr_req_fh_s, ram_wr_req_fh_r) = chan<RamWrReq>("ram_wr_req_fh");
        let (ram_wr_req_bh_s, ram_wr_req_bh_r) = chan<RamWrReq>("ram_wr_req_bh");
        let (ram_wr_req_raw_s, ram_wr_req_raw_r) = chan<RamWrReq>("ram_wr_req_raw");
        let (ram_wr_resp_fh_s, ram_wr_resp_fh_r) = chan<RamWrResp>("ram_wr_resp_fh");
        let (ram_wr_resp_bh_s, ram_wr_resp_bh_r) = chan<RamWrResp>("ram_wr_resp_bh");
        let (ram_wr_resp_raw_s, ram_wr_resp_raw_r) = chan<RamWrResp>("ram_wr_resp_raw");

        let (output_s, output_r) = chan<ZstdDecodedPacket>("output");
        let (notify_s, notify_r) = chan<()>("notify");
        let (reset_s, reset_r) = chan<()>("reset");

        spawn zstd_dec::ZstdDecoder<
            TEST_AXI_DATA_W, TEST_AXI_ADDR_W, TEST_AXI_ID_W, TEST_AXI_DEST_W,
            TEST_REGS_N, TEST_WINDOW_LOG_MAX,
            TEST_HB_ADDR_W, TEST_HB_DATA_W, TEST_HB_NUM_PARTITIONS, TEST_HB_SIZE_KB,
        >(
            csr_axi_aw_r, csr_axi_w_r, csr_axi_b_s, csr_axi_ar_r, csr_axi_r_s,
            fh_axi_ar_s, fh_axi_r_r,
            bh_axi_ar_s, bh_axi_r_r,
            raw_axi_ar_s, raw_axi_r_r,
            output_axi_aw_s, output_axi_w_s, output_axi_b_r,
            ram_rd_req_s[0], ram_rd_req_s[1], ram_rd_req_s[2], ram_rd_req_s[3],
            ram_rd_req_s[4], ram_rd_req_s[5], ram_rd_req_s[6], ram_rd_req_s[7],
            ram_rd_resp_r[0], ram_rd_resp_r[1], ram_rd_resp_r[2], ram_rd_resp_r[3],
            ram_rd_resp_r[4], ram_rd_resp_r[5], ram_rd_resp_r[6], ram_rd_resp_r[7],
            ram_wr_req_s[0], ram_wr_req_s[1], ram_wr_req_s[2], ram_wr_req_s[3],
            ram_wr_req_s[4], ram_wr_req_s[5], ram_wr_req_s[6], ram_wr_req_s[7],
            ram_wr_resp_r[0], ram_wr_resp_r[1], ram_wr_resp_r[2], ram_wr_resp_r[3],
            ram_wr_resp_r[4], ram_wr_resp_r[5], ram_wr_resp_r[6], ram_wr_resp_r[7],
            output_s, notify_s, reset_s,
        );

        spawn ram::RamModel<
            TEST_HB_DATA_W, TEST_HB_RAM_SIZE, TEST_HB_RAM_WORD_PARTITION_SIZE,
            TEST_HB_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_HB_RAM_INITIALIZED,
            TEST_HB_RAM_ASSERT_VALID_READ
        > (ram_rd_req_r[0], ram_rd_resp_s[0], ram_wr_req_r[0], ram_wr_resp_s[0]);

        spawn ram::RamModel<
            TEST_HB_DATA_W, TEST_HB_RAM_SIZE, TEST_HB_RAM_WORD_PARTITION_SIZE,
            TEST_HB_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_HB_RAM_INITIALIZED,
            TEST_HB_RAM_ASSERT_VALID_READ
        > (ram_rd_req_r[1], ram_rd_resp_s[1], ram_wr_req_r[1], ram_wr_resp_s[1]);

        spawn ram::RamModel<
            TEST_HB_DATA_W, TEST_HB_RAM_SIZE, TEST_HB_RAM_WORD_PARTITION_SIZE,
            TEST_HB_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_HB_RAM_INITIALIZED,
            TEST_HB_RAM_ASSERT_VALID_READ
        > (ram_rd_req_r[2], ram_rd_resp_s[2], ram_wr_req_r[2], ram_wr_resp_s[2]);

        spawn ram::RamModel<
            TEST_HB_DATA_W, TEST_HB_RAM_SIZE, TEST_HB_RAM_WORD_PARTITION_SIZE,
            TEST_HB_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_HB_RAM_INITIALIZED,
            TEST_HB_RAM_ASSERT_VALID_READ
        > (ram_rd_req_r[3], ram_rd_resp_s[3], ram_wr_req_r[3], ram_wr_resp_s[3]);

        spawn ram::RamModel<
            TEST_HB_DATA_W, TEST_HB_RAM_SIZE, TEST_HB_RAM_WORD_PARTITION_SIZE,
            TEST_HB_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_HB_RAM_INITIALIZED,
            TEST_HB_RAM_ASSERT_VALID_READ
        > (ram_rd_req_r[4], ram_rd_resp_s[4], ram_wr_req_r[4], ram_wr_resp_s[4]);

        spawn ram::RamModel<
            TEST_HB_DATA_W, TEST_HB_RAM_SIZE, TEST_HB_RAM_WORD_PARTITION_SIZE,
            TEST_HB_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_HB_RAM_INITIALIZED,
            TEST_HB_RAM_ASSERT_VALID_READ
        > (ram_rd_req_r[5], ram_rd_resp_s[5], ram_wr_req_r[5], ram_wr_resp_s[5]);

        spawn ram::RamModel<
            TEST_HB_DATA_W, TEST_HB_RAM_SIZE, TEST_HB_RAM_WORD_PARTITION_SIZE,
            TEST_HB_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_HB_RAM_INITIALIZED,
            TEST_HB_RAM_ASSERT_VALID_READ
        > (ram_rd_req_r[6], ram_rd_resp_s[6], ram_wr_req_r[6], ram_wr_resp_s[6]);

        spawn ram::RamModel<
            TEST_HB_DATA_W, TEST_HB_RAM_SIZE, TEST_HB_RAM_WORD_PARTITION_SIZE,
            TEST_HB_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_HB_RAM_INITIALIZED,
            TEST_HB_RAM_ASSERT_VALID_READ
        > (ram_rd_req_r[7], ram_rd_resp_s[7], ram_wr_req_r[7], ram_wr_resp_s[7]);

        spawn ram::RamModel<
            TEST_RAM_DATA_W, TEST_RAM_SIZE, TEST_RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED,
        > (ram_rd_req_fh_r, ram_rd_resp_fh_s, ram_wr_req_fh_r, ram_wr_resp_fh_s);

        spawn ram::RamModel<
            TEST_RAM_DATA_W, TEST_RAM_SIZE, TEST_RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED,
        > (ram_rd_req_bh_r, ram_rd_resp_bh_s, ram_wr_req_bh_r, ram_wr_resp_bh_s);

        spawn ram::RamModel<
            TEST_RAM_DATA_W, TEST_RAM_SIZE, TEST_RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED,
        > (ram_rd_req_raw_r, ram_rd_resp_raw_s, ram_wr_req_raw_r, ram_wr_resp_raw_s);

        spawn axi_ram::AxiRamReader<
            TEST_AXI_ADDR_W, TEST_AXI_DATA_W, TEST_AXI_DEST_W, TEST_AXI_ID_W,
            TEST_RAM_SIZE, TEST_RAM_BASE_ADDR, TEST_RAM_DATA_W, TEST_RAM_ADDR_W,
        >(fh_axi_ar_r, fh_axi_r_s, ram_rd_req_fh_s, ram_rd_resp_fh_r);

        spawn axi_ram::AxiRamReader<
            TEST_AXI_ADDR_W, TEST_AXI_DATA_W, TEST_AXI_DEST_W, TEST_AXI_ID_W,
            TEST_RAM_SIZE, TEST_RAM_BASE_ADDR, TEST_RAM_DATA_W, TEST_RAM_ADDR_W,
        >(bh_axi_ar_r, bh_axi_r_s, ram_rd_req_bh_s, ram_rd_resp_bh_r);

        spawn axi_ram::AxiRamReader<
            TEST_AXI_ADDR_W, TEST_AXI_DATA_W, TEST_AXI_DEST_W, TEST_AXI_ID_W,
            TEST_RAM_SIZE, TEST_RAM_BASE_ADDR, TEST_RAM_DATA_W, TEST_RAM_ADDR_W,
        >(raw_axi_ar_r, raw_axi_r_s, ram_rd_req_raw_s, ram_rd_resp_raw_r);

        (
            terminator,
            csr_axi_aw_s, csr_axi_w_s, csr_axi_b_r, csr_axi_ar_s, csr_axi_r_r,
            fh_axi_ar_r, fh_axi_r_s,
            bh_axi_ar_r, bh_axi_r_s,
            raw_axi_ar_r, raw_axi_r_s,
            output_axi_aw_r, output_axi_w_r, output_axi_b_s,
            ram_rd_req_r, ram_rd_resp_s, ram_wr_req_r, ram_wr_resp_s,
            ram_wr_req_fh_s, ram_wr_req_bh_s, ram_wr_req_raw_s,
            ram_wr_resp_fh_r, ram_wr_resp_bh_r, ram_wr_resp_raw_r,
            output_r, notify_r, reset_r,
        )
    }

    next (state: ()) {
        trace_fmt!("Test start");
        let frames_count = array_size(zstd_frame_testcases::FRAMES);

        let tok = join();
        let tok = unroll_for! (test_i, tok): (u32, token) in range(u32:0, frames_count) {
            trace_fmt!("Loading testcase {:x}", test_i + u32:1);
            let frame = zstd_frame_testcases::FRAMES[test_i];
            let tok = for (i, tok): (u32, token) in range(u32:0, frame.array_length) {
                let req = RamWrReq {
                    addr: i as uN[TEST_RAM_ADDR_W],
                    data: frame.data[i] as uN[TEST_RAM_DATA_W],
                    mask: uN[TEST_RAM_NUM_PARTITIONS]:0xFF
                };
                let tok = send(tok, ram_wr_req_fh_s, req);
                let tok = send(tok, ram_wr_req_bh_s, req);
                let tok = send(tok, ram_wr_req_raw_s, req);
                tok
            }(tok);

            trace_fmt!("Running decoder on testcase {:x}", test_i + u32:1);
            let addr_req = axi::AxiAw {
                id: uN[TEST_AXI_ID_W]:0,
                addr: uN[TEST_AXI_ADDR_W]:0,
                size: axi::AxiAxSize::MAX_4B_TRANSFER,
                len: u8:0,
                burst: axi::AxiAxBurst::FIXED,
            };
            let data_req = axi::AxiW {
                data: uN[TEST_AXI_DATA_W]:0,
                strb: uN[TEST_AXI_DATA_W_DIV8]:0xFF,
                last: u1:1,
            };

            // reset the decoder
            trace_fmt!("Sending reset");
            let tok = send(tok, csr_axi_aw_s, axi::AxiAw {
                addr: csr_addr<TEST_AXI_ADDR_W>(zstd_dec::Csr::RESET),
                ..addr_req
            });
            let tok = send(tok, csr_axi_w_s, axi::AxiW {
                data: uN[TEST_AXI_DATA_W]:0x1,
                ..data_req
            });
            trace_fmt!("Sent reset");
            let (tok, _) = recv(tok, csr_axi_b_r);
            // Wait for reset notification before issuing further CSR writes
            let (tok, _) = recv(tok, reset_r);
            // configure input buffer address
            let tok = send(tok, csr_axi_aw_s, axi::AxiAw {
                addr: csr_addr<TEST_AXI_ADDR_W>(zstd_dec::Csr::INPUT_BUFFER),
                ..addr_req
            });
            let tok = send(tok, csr_axi_w_s, axi::AxiW {
                data: uN[TEST_AXI_DATA_W]:0x0,
                ..data_req
            });
            let (tok, _) = recv(tok, csr_axi_b_r);
            // configure output buffer address
            let tok = send(tok, csr_axi_aw_s, axi::AxiAw {
                addr: csr_addr<TEST_AXI_ADDR_W>(zstd_dec::Csr::OUTPUT_BUFFER),
                ..addr_req
            });
            let tok = send(tok, csr_axi_w_s, axi::AxiW {
                data: uN[TEST_AXI_DATA_W]:0x1000,
                ..data_req
            });
            let (tok, _) = recv(tok, csr_axi_b_r);
            // start decoder
            let tok = send(tok, csr_axi_aw_s, axi::AxiAw {
                addr: csr_addr<TEST_AXI_ADDR_W>(zstd_dec::Csr::START),
                ..addr_req
            });
            let tok = send(tok, csr_axi_w_s, axi::AxiW {
                data: uN[TEST_AXI_DATA_W]:0x1,
                ..data_req
            });
            let (tok, _) = recv(tok, csr_axi_b_r);

            let decomp_frame = zstd_frame_testcases::DECOMPRESSED_FRAMES[test_i];
            let tok = for (decomp_i, tok): (u32, token) in range(u32:0, decomp_frame.array_length) {
                let (tok, decomp_packet) = recv(tok, output_r);
                assert_eq(decomp_packet.data, decomp_frame.data[decomp_i]);
                assert_eq(decomp_packet.last, decomp_i == (decomp_frame.length - u32:1) / u32:8);
                tok
            }(tok);

            // Test ZstdDecoder memory output interface
            // Mock the output memory buffer as a DSLX array
            // It is required to handle AXI write transactions and to write the incoming data to
            // the DSXL array.
            // The number of AXI transactions is not known beforehand because it depends on the
            // length of the decoded data and the address of the output buffer. The same goes
            // with the lengths of the particular AXI burst transactions (the number of transfers).
            // Because of that we cannot write for loops to handle AXI transactions dynamically.
            // As a workaround, the loops are constrained with upper bounds for AXI transactions
            // required for writing maximal supported payload and maximal possible burst transfer
            // size.

            // It is possible to decode payloads up to 16kB
            // The smallest possible AXI transaction will transfer 1 byte of data
            let MAX_AXI_TRANSACTIONS = u32:16384;
            // The maximal number if beats in AXI burst transaction
            let MAX_AXI_TRANSFERS = u32:256;
            // Actual size of decompressed payload for current test
            let DECOMPRESSED_BYTES = zstd_frame_testcases::DECOMPRESSED_FRAMES[test_i].length;
            trace_fmt!("ZstdDecTest: Start receiving output");
            let (tok, final_output_memory, final_output_memory_id, final_transfered_bytes) =
                for (axi_transaction, (tok, output_memory, output_memory_id, transfered_bytes)):
                    (u32, (token, uN[TEST_AXI_DATA_W][TEST_MOCK_OUTPUT_RAM_SIZE], u32, u32))
                     in range(u32:0, MAX_AXI_TRANSACTIONS) {
                if (transfered_bytes < DECOMPRESSED_BYTES) {
                    trace_fmt!("ZstdDecTest: Handle AXI Write transaction #{}", axi_transaction);
                    let (tok, axi_aw) = recv(tok, output_axi_aw_r);
                    trace_fmt!("ZstdDecTest: Received AXI AW: {:#x}", axi_aw);
                    let (tok, internal_output_memory, internal_output_memory_id, internal_transfered_bytes) =
                        for (axi_transfer, (tok, out_mem, out_mem_id, transf_bytes)):
                            (u32, (token, uN[TEST_AXI_DATA_W][TEST_MOCK_OUTPUT_RAM_SIZE], u32, u32))
                             in range(u32:0, MAX_AXI_TRANSFERS) {
                        if (axi_transfer as u8 <= axi_aw.len) {
                            // Receive AXI burst beat transfers
                            let (tok, axi_w) = recv(tok, output_axi_w_r);
                            trace_fmt!("ZstdDecTest: Received AXI W #{}: {:#x}", axi_transfer, axi_w);
                            let strobe_cnt = std::popcount(axi_w.strb) as u32;
                            // Assume continuous strobe, e.g.: 0b1111; 0b0111; 0b0011; 0b0001; 0b0000
                            let strobe_mask = (uN[TEST_AXI_DATA_W]:1 << (strobe_cnt * u32:8) as uN[TEST_AXI_DATA_W]) - uN[TEST_AXI_DATA_W]:1;
                            let strobed_data = axi_w.data & strobe_mask;
                            trace_fmt!("ZstdDecTest: write out_mem[{}] = {:#x}", out_mem_id, strobed_data);
                            let mem = update(out_mem, out_mem_id, (out_mem[out_mem_id] & !strobe_mask) | strobed_data);
                            let id = out_mem_id + u32:1;
                            let bytes_written = transf_bytes + strobe_cnt;
                            trace_fmt!("ZstdDecTest: bytes written: {}", bytes_written);
                            (tok, mem, id, bytes_written)
                        } else {
                            (tok, out_mem, out_mem_id, transf_bytes)
                        }
                    // Pass outer loop accumulator as initial accumulator for inner loop
                    }((tok, output_memory, output_memory_id, transfered_bytes));
                    let axi_b = axi::AxiB{resp: axi::AxiWriteResp::OKAY, id: axi_aw.id};
                    let tok = send(tok, output_axi_b_s, axi_b);
                    trace_fmt!("ZstdDecTest: Sent AXI B #{}: {:#x}", axi_transaction, axi_b);
                    (tok, internal_output_memory, internal_output_memory_id, internal_transfered_bytes)
                } else {
                    (tok, output_memory, output_memory_id, transfered_bytes)
                }
            }((tok, uN[TEST_AXI_DATA_W][TEST_MOCK_OUTPUT_RAM_SIZE]:[uN[TEST_AXI_DATA_W]:0, ...], u32:0, u32:0));
            trace_fmt!("ZstdDecTest: Finished receiving output");

            assert_eq(final_transfered_bytes, DECOMPRESSED_BYTES);
            assert_eq(final_output_memory_id, decomp_frame.array_length);
            for (memory_id, _): (u32, ()) in range(u32:0, decomp_frame.array_length) {
                assert_eq(final_output_memory[memory_id], decomp_frame.data[memory_id]);
            }(());

            let (tok, ()) = recv(tok, notify_r);
            trace_fmt!("Finished decoding testcase {:x} correctly", test_i + u32:1);
            tok
        }(tok);

        send(tok, terminator, true);
    }
}

