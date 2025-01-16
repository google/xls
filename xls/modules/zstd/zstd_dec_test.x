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
import xls.modules.zstd.memory.axi_ram;
import xls.modules.zstd.zstd_dec;
import xls.modules.zstd.comp_block_dec;
import xls.modules.zstd.sequence_dec;
import xls.modules.zstd.memory.mem_reader;
import xls.modules.zstd.huffman_literals_dec;
import xls.modules.zstd.parallel_rams;
import xls.modules.zstd.literals_buffer;
import xls.modules.zstd.fse_table_creator;
import xls.modules.zstd.ram_mux;
// import xls.modules.zstd.zstd_frame_testcases as comp_frame;
// import xls.modules.zstd.comp_frame;
import xls.modules.zstd.comp_frame_huffman as comp_frame;

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
const TEST_RAM_SIZE:u32 = u32:512;
const TEST_RAM_ADDR_W:u32 = std::clog2(TEST_RAM_SIZE);
const TEST_RAM_WORD_PARTITION_SIZE:u32 = u32:8;
const TEST_RAM_NUM_PARTITIONS:u32 = ram::num_partitions(TEST_RAM_WORD_PARTITION_SIZE, TEST_RAM_DATA_W);
const TEST_RAM_BASE_ADDR:u32 = u32:0;
const TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR = ram::SimultaneousReadWriteBehavior::READ_BEFORE_WRITE;
const TEST_RAM_INITIALIZED = true;

const TEST_DPD_RAM_DATA_W = u32:16;
const TEST_DPD_RAM_SIZE = u32:256;
const TEST_DPD_RAM_ADDR_W = std::clog2(TEST_DPD_RAM_SIZE);
const TEST_DPD_RAM_WORD_PARTITION_SIZE = TEST_DPD_RAM_DATA_W;
const TEST_DPD_RAM_NUM_PARTITIONS = ram::num_partitions(
    TEST_DPD_RAM_WORD_PARTITION_SIZE, TEST_DPD_RAM_DATA_W);

const TEST_FSE_RAM_DATA_W = u32:32;
const TEST_FSE_RAM_SIZE = u32:256;
const TEST_FSE_RAM_ADDR_W = std::clog2(TEST_FSE_RAM_SIZE);
const TEST_FSE_RAM_WORD_PARTITION_SIZE = TEST_FSE_RAM_DATA_W / u32:3;
const TEST_FSE_RAM_NUM_PARTITIONS = ram::num_partitions(
    TEST_FSE_RAM_WORD_PARTITION_SIZE, TEST_FSE_RAM_DATA_W);

const TEST_TMP_RAM_DATA_W = u32:16;
const TEST_TMP_RAM_SIZE = u32:256;
const TEST_TMP_RAM_ADDR_W = std::clog2(TEST_TMP_RAM_SIZE);
const TEST_TMP_RAM_WORD_PARTITION_SIZE = TEST_TMP_RAM_DATA_W;
const TEST_TMP_RAM_NUM_PARTITIONS = ram::num_partitions(
    TEST_TMP_RAM_WORD_PARTITION_SIZE, TEST_TMP_RAM_DATA_W);

// Huffman weights memory parameters
const HUFFMAN_WEIGHTS_RAM_SIZE: u32 = huffman_literals_dec::RAM_SIZE;
const HUFFMAN_WEIGHTS_RAM_ADDR_W: u32 = huffman_literals_dec::WEIGHTS_ADDR_WIDTH;
const HUFFMAN_WEIGHTS_RAM_DATA_W: u32 = huffman_literals_dec::WEIGHTS_DATA_WIDTH;
const HUFFMAN_WEIGHTS_RAM_PARTITION_WORD_SIZE: u32 = huffman_literals_dec::WEIGHTS_PARTITION_WORD_SIZE;
const HUFFMAN_WEIGHTS_RAM_NUM_PARTITIONS: u32 = huffman_literals_dec::WEIGHTS_NUM_PARTITIONS;

// Huffman prescan memory parameters
const HUFFMAN_PRESCAN_RAM_SIZE: u32 = huffman_literals_dec::RAM_SIZE;
const HUFFMAN_PRESCAN_RAM_ADDR_W: u32 = huffman_literals_dec::PRESCAN_ADDR_WIDTH;
const HUFFMAN_PRESCAN_RAM_DATA_W: u32 = huffman_literals_dec::PRESCAN_DATA_WIDTH;
const HUFFMAN_PRESCAN_RAM_PARTITION_WORD_SIZE: u32 = huffman_literals_dec::PRESCAN_PARTITION_WORD_SIZE;
const HUFFMAN_PRESCAN_RAM_NUM_PARTITIONS: u32 = huffman_literals_dec::PRESCAN_NUM_PARTITIONS;

const HISTORY_BUFFER_SIZE_KB = common::HISTORY_BUFFER_SIZE_KB;

// Literals buffer memory parameters
const LITERALS_BUFFER_RAM_ADDR_W: u32 = parallel_rams::ram_addr_width(HISTORY_BUFFER_SIZE_KB);
const LITERALS_BUFFER_RAM_SIZE: u32 = parallel_rams::ram_size(HISTORY_BUFFER_SIZE_KB);
const LITERALS_BUFFER_RAM_DATA_W: u32 = literals_buffer::RAM_DATA_WIDTH;
const LITERALS_BUFFER_RAM_NUM_PARTITIONS: u32 = literals_buffer::RAM_NUM_PARTITIONS;
const LITERALS_BUFFER_RAM_WORD_PARTITION_SIZE: u32 = LITERALS_BUFFER_RAM_DATA_W;

const AXI_CHAN_N = u32:10;

const TEST_MOCK_OUTPUT_RAM_SIZE:u32 = TEST_RAM_SIZE;

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

    type Req = comp_block_dec::CompressBlockDecoderReq<TEST_AXI_ADDR_W>;
    type Resp = comp_block_dec::CompressBlockDecoderResp;

    type SequenceDecReq = sequence_dec::SequenceDecoderReq<TEST_AXI_ADDR_W>;
    type SequenceDecResp = sequence_dec::SequenceDecoderResp;

    type MemReaderReq  = mem_reader::MemReaderReq<TEST_AXI_ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<TEST_AXI_DATA_W, TEST_AXI_ADDR_W>;

    type MemAxiAr = axi::AxiAr<TEST_AXI_ADDR_W, TEST_AXI_ID_W>;
    type MemAxiR = axi::AxiR<TEST_AXI_DATA_W, TEST_AXI_ID_W>;
    type MemAxiAw = axi::AxiAw<TEST_AXI_ADDR_W, TEST_AXI_ID_W>;
    type MemAxiW = axi::AxiW<TEST_AXI_DATA_W, TEST_AXI_DATA_W_DIV8>;
    type MemAxiB = axi::AxiB<TEST_AXI_ID_W>;

    type DpdRamRdReq = ram::ReadReq<TEST_DPD_RAM_ADDR_W, TEST_DPD_RAM_NUM_PARTITIONS>;
    type DpdRamRdResp = ram::ReadResp<TEST_DPD_RAM_DATA_W>;
    type DpdRamWrReq = ram::WriteReq<TEST_DPD_RAM_ADDR_W, TEST_DPD_RAM_DATA_W, TEST_DPD_RAM_NUM_PARTITIONS>;
    type DpdRamWrResp = ram::WriteResp;

    type TmpRamRdReq = ram::ReadReq<TEST_TMP_RAM_ADDR_W, TEST_TMP_RAM_NUM_PARTITIONS>;
    type TmpRamRdResp = ram::ReadResp<TEST_TMP_RAM_DATA_W>;
    type TmpRamWrReq = ram::WriteReq<TEST_TMP_RAM_ADDR_W, TEST_TMP_RAM_DATA_W, TEST_TMP_RAM_NUM_PARTITIONS>;
    type TmpRamWrResp = ram::WriteResp;

    type FseRamRdReq = ram::ReadReq<TEST_FSE_RAM_ADDR_W, TEST_FSE_RAM_NUM_PARTITIONS>;
    type FseRamRdResp = ram::ReadResp<TEST_FSE_RAM_DATA_W>;
    type FseRamWrReq = ram::WriteReq<TEST_FSE_RAM_ADDR_W, TEST_FSE_RAM_DATA_W, TEST_FSE_RAM_NUM_PARTITIONS>;
    type FseRamWrResp = ram::WriteResp;

    type SequenceExecutorPacket = common::SequenceExecutorPacket<common::SYMBOL_WIDTH>;
    type CommandConstructorData = common::CommandConstructorData;

    type HuffmanWeightsReadReq    = ram::ReadReq<HUFFMAN_WEIGHTS_RAM_ADDR_W, HUFFMAN_WEIGHTS_RAM_NUM_PARTITIONS>;
    type HuffmanWeightsReadResp   = ram::ReadResp<HUFFMAN_WEIGHTS_RAM_DATA_W>;
    type HuffmanWeightsWriteReq   = ram::WriteReq<HUFFMAN_WEIGHTS_RAM_ADDR_W, HUFFMAN_WEIGHTS_RAM_DATA_W, HUFFMAN_WEIGHTS_RAM_NUM_PARTITIONS>;
    type HuffmanWeightsWriteResp  = ram::WriteResp;
    type HuffmanPrescanReadReq    = ram::ReadReq<HUFFMAN_PRESCAN_RAM_ADDR_W, HUFFMAN_PRESCAN_RAM_NUM_PARTITIONS>;
    type HuffmanPrescanReadResp   = ram::ReadResp<HUFFMAN_PRESCAN_RAM_DATA_W>;
    type HuffmanPrescanWriteReq   = ram::WriteReq<HUFFMAN_PRESCAN_RAM_ADDR_W, HUFFMAN_PRESCAN_RAM_DATA_W, HUFFMAN_PRESCAN_RAM_NUM_PARTITIONS>;
    type HuffmanPrescanWriteResp  = ram::WriteResp;

    type LitBufRamRdReq = ram::ReadReq<LITERALS_BUFFER_RAM_ADDR_W, LITERALS_BUFFER_RAM_NUM_PARTITIONS>;
    type LitBufRamRdResp = ram::ReadResp<LITERALS_BUFFER_RAM_DATA_W>;
    type LitBufRamWrReq = ram::WriteReq<LITERALS_BUFFER_RAM_ADDR_W, LITERALS_BUFFER_RAM_DATA_W, LITERALS_BUFFER_RAM_NUM_PARTITIONS>;
    type LitBufRamWrResp = ram::WriteResp;

    terminator: chan<bool> out;

    csr_axi_aw_s: chan<CsrAxiAw> out;
    csr_axi_w_s: chan<CsrAxiW> out;
    csr_axi_b_r: chan<CsrAxiB> in;
    csr_axi_ar_s: chan<CsrAxiAr> out;
    csr_axi_r_r: chan<CsrAxiR> in;

    fh_axi_ar_r: chan<MemAxiAr> in;
    fh_axi_r_s: chan<MemAxiR> out;
    fh_ram_wr_req_s: chan<RamWrReq> out;
    fh_raw_wr_resp_r: chan<RamWrResp> in;

    bh_axi_ar_r: chan<MemAxiAr> in;
    bh_axi_r_s: chan<MemAxiR> out;
    bh_ram_wr_req_s: chan<RamWrReq> out;
    bh_raw_wr_resp_r: chan<RamWrResp> in;

    raw_axi_ar_r: chan<MemAxiAr> in;
    raw_axi_r_s: chan<MemAxiR> out;
    raw_ram_wr_req_s: chan<RamWrReq> out;
    raw_raw_wr_resp_r: chan<RamWrResp> in;

    comp_ram_wr_req_s: chan<RamWrReq>[AXI_CHAN_N] out;
    comp_ram_wr_resp_r: chan<RamWrResp>[AXI_CHAN_N] in;

    output_axi_aw_r: chan<MemAxiAw> in;
    output_axi_w_r: chan<MemAxiW> in;
    output_axi_b_s: chan<MemAxiB> out;

    hb_ram_rd_req_r: chan<RamRdReqHB>[8] in;
    hb_ram_rd_resp_s: chan<RamRdRespHB>[8] out;
    hb_ram_wr_req_r: chan<RamWrReqHB>[8] in;
    hb_ram_wr_resp_s: chan<RamWrRespHB>[8] out;

    ll_sel_test_s: chan<u1> out;
    ll_def_test_rd_req_s: chan<FseRamRdReq> out;
    ll_def_test_rd_resp_r: chan<FseRamRdResp> in;
    ll_def_test_wr_req_s: chan<FseRamWrReq> out;
    ll_def_test_wr_resp_r: chan<FseRamWrResp> in;

    ml_sel_test_s: chan<u1> out;
    ml_def_test_rd_req_s: chan<FseRamRdReq> out;
    ml_def_test_rd_resp_r: chan<FseRamRdResp> in;
    ml_def_test_wr_req_s: chan<FseRamWrReq> out;
    ml_def_test_wr_resp_r: chan<FseRamWrResp> in;

    of_sel_test_s: chan<u1> out;
    of_def_test_rd_req_s: chan<FseRamRdReq> out;
    of_def_test_rd_resp_r: chan<FseRamRdResp> in;
    of_def_test_wr_req_s: chan<FseRamWrReq> out;
    of_def_test_wr_resp_r: chan<FseRamWrResp> in;

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
        let (fh_ram_wr_req_s, fh_ram_wr_req_r) = chan<RamWrReq>("fh_ram_wr_req");
        let (fh_ram_wr_resp_s, fh_ram_wr_resp_r) = chan<RamWrResp>("fh_ram_wr_resp");
        let (fh_ram_rd_req_s, fh_ram_rd_req_r) = chan<RamRdReq>("fh_ram_rd_req");
        let (fh_ram_rd_resp_s, fh_ram_rd_resp_r) = chan<RamRdResp>("fh_ram_rd_resp");

        let (bh_axi_ar_s, bh_axi_ar_r) = chan<MemAxiAr>("bh_axi_ar");
        let (bh_axi_r_s, bh_axi_r_r) = chan<MemAxiR>("bh_axi_r");
        let (bh_ram_rd_req_s, bh_ram_rd_req_r) = chan<RamRdReq>("bh_ram_rd_req");
        let (bh_ram_rd_resp_s, bh_ram_rd_resp_r) = chan<RamRdResp>("bh_ram_rd_resp");
        let (bh_ram_wr_req_s, bh_ram_wr_req_r) = chan<RamWrReq>("bh_ram_wr_req");
        let (bh_ram_wr_resp_s, bh_ram_wr_resp_r) = chan<RamWrResp>("bh_ram_wr_resp");

        let (raw_axi_ar_s, raw_axi_ar_r) = chan<MemAxiAr>("raw_axi_ar");
        let (raw_axi_r_s, raw_axi_r_r) = chan<MemAxiR>("raw_axi_r");
        let (raw_ram_rd_req_s, raw_ram_rd_req_r) = chan<RamRdReq>("raw_ram_rd_req");
        let (raw_ram_rd_resp_s, raw_ram_rd_resp_r) = chan<RamRdResp>("raw_ram_rd_resp");
        let (raw_ram_wr_req_s, raw_ram_wr_req_r) = chan<RamWrReq>("raw_ram_wr_req");
        let (raw_ram_wr_resp_s, raw_ram_wr_resp_r) = chan<RamWrResp>("raw_ram_wr_resp");

        let (output_axi_aw_s, output_axi_aw_r) = chan<MemAxiAw>("output_axi_aw");
        let (output_axi_w_s, output_axi_w_r) = chan<MemAxiW>("output_axi_w");
        let (output_axi_b_s, output_axi_b_r) = chan<MemAxiB>("output_axi_b");

        let (hb_ram_rd_req_s, hb_ram_rd_req_r) = chan<RamRdReqHB>[8]("hb_ram_rd_req");
        let (hb_ram_rd_resp_s, hb_ram_rd_resp_r) = chan<RamRdRespHB>[8]("hb_ram_rd_resp");
        let (hb_ram_wr_req_s, hb_ram_wr_req_r) = chan<RamWrReqHB>[8]("hb_ram_wr_req");
        let (hb_ram_wr_resp_s, hb_ram_wr_resp_r) = chan<RamWrRespHB>[8]("hb_ram_wr_resp");

        let (notify_s, notify_r) = chan<()>("notify");
        let (reset_s, reset_r) = chan<()>("reset");

        // Huffman weights memory
        let (huffman_lit_weights_mem_rd_req_s, huffman_lit_weights_mem_rd_req_r) = chan<HuffmanWeightsReadReq>("huffman_lit_weights_mem_rd_req");
        let (huffman_lit_weights_mem_rd_resp_s, huffman_lit_weights_mem_rd_resp_r) = chan<HuffmanWeightsReadResp>("huffman_lit_weights_mem_rd_resp");
        let (huffman_lit_weights_mem_wr_req_s, huffman_lit_weights_mem_wr_req_r) = chan<HuffmanWeightsWriteReq>("huffman_lit_weights_mem_wr_req");
        let (huffman_lit_weights_mem_wr_resp_s, huffman_lit_weights_mem_wr_resp_r) = chan<HuffmanWeightsWriteResp>("huffman_lit_weights_mem_wr_resp");

        spawn ram::RamModel<
            HUFFMAN_WEIGHTS_RAM_DATA_W, HUFFMAN_WEIGHTS_RAM_SIZE, HUFFMAN_WEIGHTS_RAM_PARTITION_WORD_SIZE,
            TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED
        >(
            huffman_lit_weights_mem_rd_req_r, huffman_lit_weights_mem_rd_resp_s,
            huffman_lit_weights_mem_wr_req_r, huffman_lit_weights_mem_wr_resp_s,
        );

        // Huffman prescan memory
        let (huffman_lit_prescan_mem_rd_req_s, huffman_lit_prescan_mem_rd_req_r) = chan<HuffmanPrescanReadReq>("huffman_lit_prescan_mem_rd_req");
        let (huffman_lit_prescan_mem_rd_resp_s, huffman_lit_prescan_mem_rd_resp_r) = chan<HuffmanPrescanReadResp>("huffman_lit_prescan_mem_rd_resp");
        let (huffman_lit_prescan_mem_wr_req_s, huffman_lit_prescan_mem_wr_req_r) = chan<HuffmanPrescanWriteReq>("huffman_lit_prescan_mem_wr_req");
        let (huffman_lit_prescan_mem_wr_resp_s, huffman_lit_prescan_mem_wr_resp_r) = chan<HuffmanPrescanWriteResp>("huffman_lit_prescan_mem_wr_resp");

        spawn ram::RamModel<
            HUFFMAN_PRESCAN_RAM_DATA_W, HUFFMAN_PRESCAN_RAM_SIZE, HUFFMAN_PRESCAN_RAM_PARTITION_WORD_SIZE,
            TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED
        >(
            huffman_lit_prescan_mem_rd_req_r, huffman_lit_prescan_mem_rd_resp_s,
            huffman_lit_prescan_mem_wr_req_r, huffman_lit_prescan_mem_wr_resp_s,
        );

        // AXI channels for various blocks
        let (comp_axi_ar_s, comp_axi_ar_r) = chan<MemAxiAr>[AXI_CHAN_N]("comp_axi_ar");
        let (comp_axi_r_s, comp_axi_r_r) = chan<MemAxiR>[AXI_CHAN_N]("comp_axi_r");

        let (comp_ram_rd_req_s,  comp_ram_rd_req_r) = chan<RamRdReq>[AXI_CHAN_N]("comp_ram_rd_req");
        let (comp_ram_rd_resp_s, comp_ram_rd_resp_r) = chan<RamRdResp>[AXI_CHAN_N]("comp_ram_rd_resp");
        let (comp_ram_wr_req_s,  comp_ram_wr_req_r) = chan<RamWrReq>[AXI_CHAN_N]("comp_ram_wr_req");
        let (comp_ram_wr_resp_s, comp_ram_wr_resp_r) = chan<RamWrResp>[AXI_CHAN_N]("comp_ram_wr_resp");

        unroll_for! (i, ()): (u32, ()) in range(u32:0, AXI_CHAN_N) {
            spawn ram::RamModel<
                TEST_RAM_DATA_W, TEST_RAM_SIZE, TEST_RAM_WORD_PARTITION_SIZE,
                TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED
            >(comp_ram_rd_req_r[i], comp_ram_rd_resp_s[i], comp_ram_wr_req_r[i], comp_ram_wr_resp_s[i]);

            spawn axi_ram::AxiRamReader<
                TEST_AXI_ADDR_W, TEST_AXI_DATA_W, TEST_AXI_DEST_W, TEST_AXI_ID_W, TEST_RAM_SIZE,
                TEST_RAM_BASE_ADDR, TEST_RAM_DATA_W, TEST_RAM_ADDR_W
            >(comp_axi_ar_r[i], comp_axi_r_s[i], comp_ram_rd_req_s[i], comp_ram_rd_resp_r[i]);
        }(());

        // Literals buffer RAMs
        let (litbuf_rd_req_s,  litbuf_rd_req_r) = chan<LitBufRamRdReq>[u32:8]("litbuf_rd_req");
        let (litbuf_rd_resp_s, litbuf_rd_resp_r) = chan<LitBufRamRdResp>[u32:8]("litbuf_rd_resp");
        let (litbuf_wr_req_s,  litbuf_wr_req_r) = chan<LitBufRamWrReq>[u32:8]("litbuf_wr_req");
        let (litbuf_wr_resp_s, litbuf_wr_resp_r) = chan<LitBufRamWrResp>[u32:8]("litbuf_wr_resp");
        unroll_for! (i, ()): (u32, ()) in range(u32:0, u32:8) {
            spawn ram::RamModel<
                LITERALS_BUFFER_RAM_DATA_W, LITERALS_BUFFER_RAM_SIZE, LITERALS_BUFFER_RAM_WORD_PARTITION_SIZE,
                TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED
            >(
                litbuf_rd_req_r[i], litbuf_rd_resp_s[i], litbuf_wr_req_r[i], litbuf_wr_resp_s[i]
            );
        }(());

        // RAMs for FSE decoder
        // DPD RAM
        let (dpd_rd_req_s, dpd_rd_req_r) = chan<DpdRamRdReq>("dpd_rd_req");
        let (dpd_rd_resp_s, dpd_rd_resp_r) = chan<DpdRamRdResp>("dpd_rd_resp");
        let (dpd_wr_req_s, dpd_wr_req_r) = chan<DpdRamWrReq>("dpd_wr_req");
        let (dpd_wr_resp_s, dpd_wr_resp_r) = chan<DpdRamWrResp>("dpd_wr_resp");
        spawn ram::RamModel<
            TEST_DPD_RAM_DATA_W, TEST_DPD_RAM_SIZE, TEST_DPD_RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED
        >(
            dpd_rd_req_r, dpd_rd_resp_s, dpd_wr_req_r, dpd_wr_resp_s,
        );

        // TMP RAM
        let (tmp_rd_req_s, tmp_rd_req_r) = chan<TmpRamRdReq>("tmp_rd_req");
        let (tmp_rd_resp_s, tmp_rd_resp_r) = chan<TmpRamRdResp>("tmp_rd_resp");
        let (tmp_wr_req_s, tmp_wr_req_r) = chan<TmpRamWrReq>("tmp_wr_req");
        let (tmp_wr_resp_s, tmp_wr_resp_r) = chan<TmpRamWrResp>("tmp_wr_resp");
        spawn ram::RamModel<
            TEST_TMP_RAM_DATA_W, TEST_TMP_RAM_SIZE, TEST_TMP_RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED
        >(
            tmp_rd_req_r, tmp_rd_resp_s, tmp_wr_req_r, tmp_wr_resp_s,
        );

        // FSE RAMs
        let (fse_rd_req_s, fse_rd_req_r) = chan<FseRamRdReq>[u32:6]("fse_rd_req");
        let (fse_rd_resp_s, fse_rd_resp_r) = chan<FseRamRdResp>[u32:6]("fse_rd_resp");
        let (fse_wr_req_s, fse_wr_req_r) = chan<FseRamWrReq>[u32:6]("fse_wr_req");
        let (fse_wr_resp_s, fse_wr_resp_r) = chan<FseRamWrResp>[u32:6]("fse_wr_resp");
        unroll_for! (i, ()): (u32, ()) in range(u32:0, u32:6) {
            spawn ram::RamModel<
                TEST_FSE_RAM_DATA_W, TEST_FSE_RAM_SIZE, TEST_FSE_RAM_WORD_PARTITION_SIZE,
                TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED
            >(
                fse_rd_req_r[i], fse_rd_resp_s[i], fse_wr_req_r[i], fse_wr_resp_s[i]
            );
        }(());

        // Default LL

        let (ll_sel_test_s, ll_sel_test_r) = chan<u1>("ll_sel_test");

        let (ll_def_test_rd_req_s, ll_def_test_rd_req_r) = chan<FseRamRdReq>("ll_def_test_rd_req");
        let (ll_def_test_rd_resp_s, ll_def_test_rd_resp_r) = chan<FseRamRdResp>("ll_def_test_rd_resp");
        let (ll_def_test_wr_req_s, ll_def_test_wr_req_r) = chan<FseRamWrReq>("ll_def_test_wr_req");
        let (ll_def_test_wr_resp_s, ll_def_test_wr_resp_r) = chan<FseRamWrResp>("ll_def_test_wr_resp");

        let (ll_def_fse_rd_req_s, ll_def_fse_rd_req_r) = chan<FseRamRdReq>("ll_def_fse_rd_req");
        let (ll_def_fse_rd_resp_s, ll_def_fse_rd_resp_r) = chan<FseRamRdResp>("ll_def_fse_rd_resp");
        let (ll_def_fse_wr_req_s, ll_def_fse_wr_req_r) = chan<FseRamWrReq>("ll_def_fse_wr_req");
        let (ll_def_fse_wr_resp_s, ll_def_fse_wr_resp_r) = chan<FseRamWrResp>("ll_def_fse_wr_resp");

        spawn ram_mux::RamMux<
            TEST_FSE_RAM_ADDR_W,
            TEST_FSE_RAM_DATA_W,
            TEST_FSE_RAM_NUM_PARTITIONS,
        >(
            ll_sel_test_r,
            ll_def_test_rd_req_r, ll_def_test_rd_resp_s, ll_def_test_wr_req_r, ll_def_test_wr_resp_s,
            ll_def_fse_rd_req_r, ll_def_fse_rd_resp_s, ll_def_fse_wr_req_r, ll_def_fse_wr_resp_s,
            fse_rd_req_s[0], fse_rd_resp_r[0], fse_wr_req_s[0], fse_wr_resp_r[0],
        );

        // Default ML

        let (ml_sel_test_s, ml_sel_test_r) = chan<u1>("ml_sel_test");

        let (ml_def_test_rd_req_s, ml_def_test_rd_req_r) = chan<FseRamRdReq>("ml_def_test_rd_req");
        let (ml_def_test_rd_resp_s, ml_def_test_rd_resp_r) = chan<FseRamRdResp>("ml_def_test_rd_resp");
        let (ml_def_test_wr_req_s, ml_def_test_wr_req_r) = chan<FseRamWrReq>("ml_def_test_wr_req");
        let (ml_def_test_wr_resp_s, ml_def_test_wr_resp_r) = chan<FseRamWrResp>("ml_def_test_wr_resp");

        let (ml_def_fse_rd_req_s, ml_def_fse_rd_req_r) = chan<FseRamRdReq>("ml_def_fse_rd_req");
        let (ml_def_fse_rd_resp_s, ml_def_fse_rd_resp_r) = chan<FseRamRdResp>("ml_def_fse_rd_resp");
        let (ml_def_fse_wr_req_s, ml_def_fse_wr_req_r) = chan<FseRamWrReq>("ml_def_fse_wr_req");
        let (ml_def_fse_wr_resp_s, ml_def_fse_wr_resp_r) = chan<FseRamWrResp>("ml_def_fse_wr_resp");

        spawn ram_mux::RamMux<
            TEST_FSE_RAM_ADDR_W,
            TEST_FSE_RAM_DATA_W,
            TEST_FSE_RAM_NUM_PARTITIONS,
        >(
            ml_sel_test_r,
            ml_def_test_rd_req_r, ml_def_test_rd_resp_s, ml_def_test_wr_req_r, ml_def_test_wr_resp_s,
            ml_def_fse_rd_req_r, ml_def_fse_rd_resp_s, ml_def_fse_wr_req_r, ml_def_fse_wr_resp_s,
            fse_rd_req_s[2], fse_rd_resp_r[2], fse_wr_req_s[2], fse_wr_resp_r[2],
        );

        // Default OF

        let (of_sel_test_s, of_sel_test_r) = chan<u1>("of_sel_test");

        let (of_def_test_rd_req_s, of_def_test_rd_req_r) = chan<FseRamRdReq>("of_def_test_rd_req");
        let (of_def_test_rd_resp_s, of_def_test_rd_resp_r) = chan<FseRamRdResp>("of_def_test_rd_resp");
        let (of_def_test_wr_req_s, of_def_test_wr_req_r) = chan<FseRamWrReq>("of_def_test_wr_req");
        let (of_def_test_wr_resp_s, of_def_test_wr_resp_r) = chan<FseRamWrResp>("of_def_test_wr_resp");

        let (of_def_fse_rd_req_s, of_def_fse_rd_req_r) = chan<FseRamRdReq>("of_def_fse_rd_req");
        let (of_def_fse_rd_resp_s, of_def_fse_rd_resp_r) = chan<FseRamRdResp>("of_def_fse_rd_resp");
        let (of_def_fse_wr_req_s, of_def_fse_wr_req_r) = chan<FseRamWrReq>("of_def_fse_wr_req");
        let (of_def_fse_wr_resp_s, of_def_fse_wr_resp_r) = chan<FseRamWrResp>("of_def_fse_wr_resp");

        spawn ram_mux::RamMux<
            TEST_FSE_RAM_ADDR_W,
            TEST_FSE_RAM_DATA_W,
            TEST_FSE_RAM_NUM_PARTITIONS,
        >(
            of_sel_test_r,
            of_def_test_rd_req_r, of_def_test_rd_resp_s, of_def_test_wr_req_r, of_def_test_wr_resp_s,
            of_def_fse_rd_req_r, of_def_fse_rd_resp_s, of_def_fse_wr_req_r, of_def_fse_wr_resp_s,
            fse_rd_req_s[4], fse_rd_resp_r[4], fse_wr_req_s[4], fse_wr_resp_r[4],
        );

        spawn zstd_dec::ZstdDecoder<
            TEST_AXI_DATA_W, TEST_AXI_ADDR_W, TEST_AXI_ID_W, TEST_AXI_DEST_W,
            TEST_REGS_N, TEST_WINDOW_LOG_MAX,
            TEST_HB_ADDR_W, TEST_HB_DATA_W, TEST_HB_NUM_PARTITIONS, TEST_HB_SIZE_KB,
            TEST_DPD_RAM_ADDR_W, TEST_DPD_RAM_DATA_W, TEST_DPD_RAM_NUM_PARTITIONS,
            TEST_TMP_RAM_ADDR_W, TEST_TMP_RAM_DATA_W, TEST_TMP_RAM_NUM_PARTITIONS,
            TEST_FSE_RAM_ADDR_W, TEST_FSE_RAM_DATA_W, TEST_FSE_RAM_NUM_PARTITIONS,
            HISTORY_BUFFER_SIZE_KB, AXI_CHAN_N,
        >(
            csr_axi_aw_r, csr_axi_w_r, csr_axi_b_s, csr_axi_ar_r, csr_axi_r_s,
            fh_axi_ar_s, fh_axi_r_r,
            bh_axi_ar_s, bh_axi_r_r,
            raw_axi_ar_s, raw_axi_r_r,
            comp_axi_ar_s, comp_axi_r_r,
            dpd_rd_req_s, dpd_rd_resp_r,
            dpd_wr_req_s, dpd_wr_resp_r,
            tmp_rd_req_s, tmp_rd_resp_r,
            tmp_wr_req_s, tmp_wr_resp_r,

            // Channels for accessing FSE tables with muxed default FSE tables
            [ll_def_fse_rd_req_s, fse_rd_req_s[1], ml_def_fse_rd_req_s, fse_rd_req_s[3], of_def_fse_rd_req_s, fse_rd_req_s[5]],
            [ll_def_fse_rd_resp_r, fse_rd_resp_r[1], ml_def_fse_rd_resp_r, fse_rd_resp_r[3], of_def_fse_rd_resp_r, fse_rd_resp_r[5]],
            [ll_def_fse_wr_req_s, fse_wr_req_s[1], ml_def_fse_wr_req_s, fse_wr_req_s[3], of_def_fse_wr_req_s, fse_wr_req_s[5]],
            [ll_def_fse_wr_resp_r, fse_wr_resp_r[1], ml_def_fse_wr_resp_r, fse_wr_resp_r[3], of_def_fse_wr_resp_r, fse_wr_resp_r[5]],

            litbuf_rd_req_s, litbuf_rd_resp_r,
            litbuf_wr_req_s, litbuf_wr_resp_r,
            huffman_lit_weights_mem_rd_req_s, huffman_lit_weights_mem_rd_resp_r,
            huffman_lit_weights_mem_wr_req_s, huffman_lit_weights_mem_wr_resp_r,
            huffman_lit_prescan_mem_rd_req_s, huffman_lit_prescan_mem_rd_resp_r,
            huffman_lit_prescan_mem_wr_req_s, huffman_lit_prescan_mem_wr_resp_r,
            output_axi_aw_s, output_axi_w_s, output_axi_b_r,

            // RAMs for SequenceExecutor
            hb_ram_rd_req_s[0], hb_ram_rd_req_s[1], hb_ram_rd_req_s[2], hb_ram_rd_req_s[3],
            hb_ram_rd_req_s[4], hb_ram_rd_req_s[5], hb_ram_rd_req_s[6], hb_ram_rd_req_s[7],
            hb_ram_rd_resp_r[0], hb_ram_rd_resp_r[1], hb_ram_rd_resp_r[2], hb_ram_rd_resp_r[3],
            hb_ram_rd_resp_r[4], hb_ram_rd_resp_r[5], hb_ram_rd_resp_r[6], hb_ram_rd_resp_r[7],
            hb_ram_wr_req_s[0], hb_ram_wr_req_s[1], hb_ram_wr_req_s[2], hb_ram_wr_req_s[3],
            hb_ram_wr_req_s[4], hb_ram_wr_req_s[5], hb_ram_wr_req_s[6], hb_ram_wr_req_s[7],
            hb_ram_wr_resp_r[0], hb_ram_wr_resp_r[1], hb_ram_wr_resp_r[2], hb_ram_wr_resp_r[3],
            hb_ram_wr_resp_r[4], hb_ram_wr_resp_r[5], hb_ram_wr_resp_r[6], hb_ram_wr_resp_r[7],

            notify_s, reset_s,
        );

        unroll_for! (i, ()): (u32, ()) in range(u32:0, u32:8) {
            spawn ram::RamModel<
                TEST_HB_DATA_W, TEST_HB_RAM_SIZE, TEST_HB_RAM_WORD_PARTITION_SIZE,
                TEST_HB_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_HB_RAM_INITIALIZED,
                TEST_HB_RAM_ASSERT_VALID_READ
            >(hb_ram_rd_req_r[i], hb_ram_rd_resp_s[i], hb_ram_wr_req_r[i], hb_ram_wr_resp_s[i]);
        }(());


        spawn ram::RamModel<
            TEST_RAM_DATA_W, TEST_RAM_SIZE, TEST_RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED,
        > (fh_ram_rd_req_r, fh_ram_rd_resp_s, fh_ram_wr_req_r, fh_ram_wr_resp_s);

        spawn ram::RamModel<
            TEST_RAM_DATA_W, TEST_RAM_SIZE, TEST_RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED,
        > (bh_ram_rd_req_r, bh_ram_rd_resp_s, bh_ram_wr_req_r, bh_ram_wr_resp_s);

        spawn ram::RamModel<
            TEST_RAM_DATA_W, TEST_RAM_SIZE, TEST_RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED,
        > (raw_ram_rd_req_r, raw_ram_rd_resp_s, raw_ram_wr_req_r, raw_ram_wr_resp_s);

        spawn axi_ram::AxiRamReader<
            TEST_AXI_ADDR_W, TEST_AXI_DATA_W, TEST_AXI_DEST_W, TEST_AXI_ID_W,
            TEST_RAM_SIZE, TEST_RAM_BASE_ADDR, TEST_RAM_DATA_W, TEST_RAM_ADDR_W,
        >(fh_axi_ar_r, fh_axi_r_s, fh_ram_rd_req_s, fh_ram_rd_resp_r);

        spawn axi_ram::AxiRamReader<
            TEST_AXI_ADDR_W, TEST_AXI_DATA_W, TEST_AXI_DEST_W, TEST_AXI_ID_W,
            TEST_RAM_SIZE, TEST_RAM_BASE_ADDR, TEST_RAM_DATA_W, TEST_RAM_ADDR_W,
        >(bh_axi_ar_r, bh_axi_r_s, bh_ram_rd_req_s, bh_ram_rd_resp_r);

        spawn axi_ram::AxiRamReader<
            TEST_AXI_ADDR_W, TEST_AXI_DATA_W, TEST_AXI_DEST_W, TEST_AXI_ID_W,
            TEST_RAM_SIZE, TEST_RAM_BASE_ADDR, TEST_RAM_DATA_W, TEST_RAM_ADDR_W,
        >(raw_axi_ar_r, raw_axi_r_s, raw_ram_rd_req_s, raw_ram_rd_resp_r);

        (
            terminator,
            csr_axi_aw_s, csr_axi_w_s, csr_axi_b_r, csr_axi_ar_s, csr_axi_r_r,
            fh_axi_ar_r, fh_axi_r_s, fh_ram_wr_req_s, fh_ram_wr_resp_r,
            bh_axi_ar_r, bh_axi_r_s, bh_ram_wr_req_s, bh_ram_wr_resp_r,
            raw_axi_ar_r, raw_axi_r_s, raw_ram_wr_req_s, raw_ram_wr_resp_r,
            comp_ram_wr_req_s, comp_ram_wr_resp_r,
            output_axi_aw_r, output_axi_w_r, output_axi_b_s,
            hb_ram_rd_req_r, hb_ram_rd_resp_s, hb_ram_wr_req_r, hb_ram_wr_resp_s,
            ll_sel_test_s, ll_def_test_rd_req_s, ll_def_test_rd_resp_r, ll_def_test_wr_req_s, ll_def_test_wr_resp_r,
            ml_sel_test_s, ml_def_test_rd_req_s, ml_def_test_rd_resp_r, ml_def_test_wr_req_s, ml_def_test_wr_resp_r,
            of_sel_test_s, of_def_test_rd_req_s, of_def_test_rd_resp_r, of_def_test_wr_req_s, of_def_test_wr_resp_r,
            notify_r, reset_r,
        )
    }

    next (state: ()) {
        trace_fmt!("Test start");
        let frames_count = array_size(comp_frame::FRAMES);

        let tok = join();

        // FILL THE LL DEFAULT RAM
        trace_fmt!("Filling LL default FSE table");
        let tok = send(tok, ll_sel_test_s, u1:0);
        let tok = unroll_for! (i, tok): (u32, token) in range(u32:0, array_size(sequence_dec::DEFAULT_LL_TABLE)) {
            let req = FseRamWrReq {
                addr: i as uN[TEST_FSE_RAM_ADDR_W],
                data: fse_table_creator::fse_record_to_bits(sequence_dec::DEFAULT_LL_TABLE[i]),
                mask: !uN[TEST_FSE_RAM_NUM_PARTITIONS]:0,
            };
            let tok = send(tok, ll_def_test_wr_req_s, req);
            let (tok, _) = recv(tok, ll_def_test_wr_resp_r);
            tok
        }(tok);
        let tok = send(tok, ll_sel_test_s, u1:1);

        // FILL THE OF DEFAULT RAM
        trace_fmt!("Filling OF default FSE table");
        let tok = send(tok, of_sel_test_s, u1:0);
        let tok = unroll_for! (i, tok): (u32, token) in range(u32:0, array_size(sequence_dec::DEFAULT_OF_TABLE)) {
            let req = FseRamWrReq {
                addr: i as uN[TEST_FSE_RAM_ADDR_W],
                data: fse_table_creator::fse_record_to_bits(sequence_dec::DEFAULT_OF_TABLE[i]),
                mask: !uN[TEST_FSE_RAM_NUM_PARTITIONS]:0,
            };
            let tok = send(tok, of_def_test_wr_req_s, req);
            let (tok, _) = recv(tok, of_def_test_wr_resp_r);
            tok
        }(tok);
        let tok = send(tok, of_sel_test_s, u1:1);

        // FILL THE ML DEFAULT RAM
        trace_fmt!("Filling ML default FSE table");
        let tok = send(tok, ml_sel_test_s, u1:0);
        let tok = unroll_for! (i, tok): (u32, token) in range(u32:0, array_size(sequence_dec::DEFAULT_ML_TABLE)) {
            let req = FseRamWrReq {
                addr: i as uN[TEST_FSE_RAM_ADDR_W],
                data: fse_table_creator::fse_record_to_bits(sequence_dec::DEFAULT_ML_TABLE[i]),
                mask: !uN[TEST_FSE_RAM_NUM_PARTITIONS]:0,
            };
            let tok = send(tok, ml_def_test_wr_req_s, req);
            let (tok, _) = recv(tok, ml_def_test_wr_resp_r);
            tok
        }(tok);
        let tok = send(tok, ml_sel_test_s, u1:1);

        let tok = unroll_for! (test_i, tok): (u32, token) in range(u32:0, frames_count) {
            trace_fmt!("Loading testcase {:x}", test_i + u32:1);
            let frame = comp_frame::FRAMES[test_i];
            let tok = for (i, tok): (u32, token) in range(u32:0, frame.array_length) {
                let req = RamWrReq {
                    addr: i as uN[TEST_RAM_ADDR_W],
                    data: frame.data[i] as uN[TEST_RAM_DATA_W],
                    mask: uN[TEST_RAM_NUM_PARTITIONS]:0xFF
                };
                let tok = send(tok, fh_ram_wr_req_s, req);
                let tok = send(tok, bh_ram_wr_req_s, req);
                let tok = send(tok, raw_ram_wr_req_s, req);
                for (i, tok): (u32, token) in range(u32:0, AXI_CHAN_N) {
                    send(tok, comp_ram_wr_req_s[i], req)
                }(tok)
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

            let decomp_frame = comp_frame::DECOMPRESSED_FRAMES[test_i];
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
            let DECOMPRESSED_BYTES = comp_frame::DECOMPRESSED_FRAMES[test_i].length;
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
                trace_fmt!("Comparing {} output packet: {:#x} ?= {:#x}", memory_id,  final_output_memory[memory_id], decomp_frame.data[memory_id]);
                assert_eq(final_output_memory[memory_id], decomp_frame.data[memory_id]);
            }(());

            let (tok, ()) = recv(tok, notify_r);
            trace_fmt!("Finished decoding testcase {:x} correctly", test_i + u32:1);
            tok
        }(tok);

        send(tok, terminator, true);
    }
}

