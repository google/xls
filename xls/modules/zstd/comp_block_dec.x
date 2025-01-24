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
import xls.modules.zstd.memory.axi_ram;
import xls.modules.zstd.common;
import xls.modules.zstd.huffman_literals_dec;
import xls.modules.zstd.parallel_rams;
import xls.modules.zstd.literals_buffer;
import xls.modules.zstd.sequence_dec;
import xls.modules.zstd.literals_block_header_dec;
import xls.modules.zstd.literals_decoder;
import xls.modules.zstd.command_constructor;
import xls.modules.zstd.memory.axi;
import xls.modules.zstd.memory.mem_reader;
import xls.modules.zstd.fse_proba_freq_dec;
import xls.modules.zstd.fse_table_creator;
import xls.modules.zstd.ram_mux;

type SequenceExecutorPacket = common::SequenceExecutorPacket<common::SYMBOL_WIDTH>;
type ExtendedPacket = common::ExtendedBlockDataPacket;
type SequenceExecutorMessageType = common::SequenceExecutorMessageType;
type BlockDataPacket = common::BlockDataPacket;
type BlockSize = common::BlockSize;
type BlockSyncData = common::BlockSyncData;

pub enum CompressBlockDecoderStatus: u1 {
    OK = 0,
    ERROR = 1,
}

pub struct CompressBlockDecoderReq<AXI_ADDR_W: u32> {
    addr: uN[AXI_ADDR_W],
    length: BlockSize,
    id: u32,
    last_block: bool,
}
pub struct CompressBlockDecoderResp {
    status: CompressBlockDecoderStatus
}

pub proc CompressBlockDecoder<
    // AXI parameters
    AXI_DATA_W: u32, AXI_ADDR_W: u32, AXI_ID_W: u32, AXI_DEST_W: u32,

    // FSE lookup table RAMs for sequences
    SEQ_DPD_RAM_ADDR_W: u32, SEQ_DPD_RAM_DATA_W: u32, SEQ_DPD_RAM_NUM_PARTITIONS: u32,
    SEQ_TMP_RAM_ADDR_W: u32, SEQ_TMP_RAM_DATA_W: u32, SEQ_TMP_RAM_NUM_PARTITIONS: u32,
    SEQ_TMP2_RAM_ADDR_W: u32, SEQ_TMP2_RAM_DATA_W: u32, SEQ_TMP2_RAM_NUM_PARTITIONS: u32,
    SEQ_FSE_RAM_ADDR_W: u32, SEQ_FSE_RAM_DATA_W: u32, SEQ_FSE_RAM_NUM_PARTITIONS: u32,

    // for literals decoder
    HUFFMAN_WEIGHTS_DPD_RAM_ADDR_W: u32, HUFFMAN_WEIGHTS_DPD_RAM_DATA_W: u32, HUFFMAN_WEIGHTS_DPD_RAM_NUM_PARTITIONS: u32,
    HUFFMAN_WEIGHTS_TMP_RAM_ADDR_W: u32, HUFFMAN_WEIGHTS_TMP_RAM_DATA_W: u32, HUFFMAN_WEIGHTS_TMP_RAM_NUM_PARTITIONS: u32,
    HUFFMAN_WEIGHTS_TMP2_RAM_ADDR_W: u32, HUFFMAN_WEIGHTS_TMP2_RAM_DATA_W: u32, HUFFMAN_WEIGHTS_TMP2_RAM_NUM_PARTITIONS: u32,
    HUFFMAN_WEIGHTS_FSE_RAM_ADDR_W: u32, HUFFMAN_WEIGHTS_FSE_RAM_DATA_W: u32, HUFFMAN_WEIGHTS_FSE_RAM_NUM_PARTITIONS: u32,

    HISTORY_BUFFER_SIZE_KB: u32 = {common::HISTORY_BUFFER_SIZE_KB},

    // FSE proba
    FSE_PROBA_DIST_W: u32 = {u32:16},
    FSE_PROBA_MAX_DISTS: u32 = {u32:256},

    // constants
    AXI_DATA_W_DIV8: u32 = {AXI_DATA_W / u32:8},

    // Huffman weights memory parameters
    HUFFMAN_WEIGHTS_RAM_ADDR_W: u32 = {huffman_literals_dec::WEIGHTS_ADDR_WIDTH},
    HUFFMAN_WEIGHTS_RAM_DATA_W: u32 = {huffman_literals_dec::WEIGHTS_DATA_WIDTH},
    HUFFMAN_WEIGHTS_RAM_NUM_PARTITIONS: u32 = {huffman_literals_dec::WEIGHTS_NUM_PARTITIONS},
    // Huffman prescan memory parameters
    HUFFMAN_PRESCAN_RAM_ADDR_W: u32 = {huffman_literals_dec::PRESCAN_ADDR_WIDTH},
    HUFFMAN_PRESCAN_RAM_DATA_W: u32 = {huffman_literals_dec::PRESCAN_DATA_WIDTH},
    HUFFMAN_PRESCAN_RAM_NUM_PARTITIONS: u32 = {huffman_literals_dec::PRESCAN_NUM_PARTITIONS},
    // Literals buffer memory parameters
    LITERALS_BUFFER_RAM_ADDR_W: u32 = {parallel_rams::ram_addr_width(HISTORY_BUFFER_SIZE_KB)},
    LITERALS_BUFFER_RAM_DATA_W: u32 = {literals_buffer::RAM_DATA_WIDTH},
    LITERALS_BUFFER_RAM_NUM_PARTITIONS: u32 = {literals_buffer::RAM_NUM_PARTITIONS},
> {
    type Req = CompressBlockDecoderReq<AXI_ADDR_W>;
    type Resp = CompressBlockDecoderResp;

    type SequenceDecReq = sequence_dec::SequenceDecoderReq<AXI_ADDR_W>;
    type SequenceDecResp = sequence_dec::SequenceDecoderResp;

    type MemReaderReq  = mem_reader::MemReaderReq<AXI_ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<AXI_DATA_W, AXI_ADDR_W>;

    type MemAxiAr = axi::AxiAr<AXI_ADDR_W, AXI_ID_W>;
    type MemAxiR = axi::AxiR<AXI_DATA_W, AXI_ID_W>;
    type MemAxiAw = axi::AxiAw<AXI_ADDR_W, AXI_ID_W>;
    type MemAxiW = axi::AxiW<AXI_DATA_W, AXI_DATA_W_DIV8>;
    type MemAxiB = axi::AxiB<AXI_ID_W>;

    type SeqDpdRamRdReq = ram::ReadReq<SEQ_DPD_RAM_ADDR_W, SEQ_DPD_RAM_NUM_PARTITIONS>;
    type SeqDpdRamRdResp = ram::ReadResp<SEQ_DPD_RAM_DATA_W>;
    type SeqDpdRamWrReq = ram::WriteReq<SEQ_DPD_RAM_ADDR_W, SEQ_DPD_RAM_DATA_W, SEQ_DPD_RAM_NUM_PARTITIONS>;
    type SeqDpdRamWrResp = ram::WriteResp;

    type SeqTmpRamRdReq = ram::ReadReq<SEQ_TMP_RAM_ADDR_W, SEQ_TMP_RAM_NUM_PARTITIONS>;
    type SeqTmpRamRdResp = ram::ReadResp<SEQ_TMP_RAM_DATA_W>;
    type SeqTmpRamWrReq = ram::WriteReq<SEQ_TMP_RAM_ADDR_W, SEQ_TMP_RAM_DATA_W, SEQ_TMP_RAM_NUM_PARTITIONS>;
    type SeqTmpRamWrResp = ram::WriteResp;

    type SeqTmp2RamRdReq = ram::ReadReq<SEQ_TMP2_RAM_ADDR_W, SEQ_TMP2_RAM_NUM_PARTITIONS>;
    type SeqTmp2RamRdResp = ram::ReadResp<SEQ_TMP2_RAM_DATA_W>;
    type SeqTmp2RamWrReq = ram::WriteReq<SEQ_TMP2_RAM_ADDR_W, SEQ_TMP2_RAM_DATA_W, SEQ_TMP2_RAM_NUM_PARTITIONS>;
    type SeqTmp2RamWrResp = ram::WriteResp;

    type SeqFseRamRdReq = ram::ReadReq<SEQ_FSE_RAM_ADDR_W, SEQ_FSE_RAM_NUM_PARTITIONS>;
    type SeqFseRamRdResp = ram::ReadResp<SEQ_FSE_RAM_DATA_W>;
    type SeqFseRamWrReq = ram::WriteReq<SEQ_FSE_RAM_ADDR_W, SEQ_FSE_RAM_DATA_W, SEQ_FSE_RAM_NUM_PARTITIONS>;
    type SeqFseRamWrResp = ram::WriteResp;

    type HuffmanWeightsDpdRamRdReq = ram::ReadReq<HUFFMAN_WEIGHTS_DPD_RAM_ADDR_W, HUFFMAN_WEIGHTS_DPD_RAM_NUM_PARTITIONS>;
    type HuffmanWeightsDpdRamRdResp = ram::ReadResp<HUFFMAN_WEIGHTS_DPD_RAM_DATA_W>;
    type HuffmanWeightsDpdRamWrReq = ram::WriteReq<HUFFMAN_WEIGHTS_DPD_RAM_ADDR_W, HUFFMAN_WEIGHTS_DPD_RAM_DATA_W, HUFFMAN_WEIGHTS_DPD_RAM_NUM_PARTITIONS>;
    type HuffmanWeightsDpdRamWrResp = ram::WriteResp;

    type HuffmanWeightsTmpRamRdReq = ram::ReadReq<HUFFMAN_WEIGHTS_TMP_RAM_ADDR_W, HUFFMAN_WEIGHTS_TMP_RAM_NUM_PARTITIONS>;
    type HuffmanWeightsTmpRamRdResp = ram::ReadResp<HUFFMAN_WEIGHTS_TMP_RAM_DATA_W>;
    type HuffmanWeightsTmpRamWrReq = ram::WriteReq<HUFFMAN_WEIGHTS_TMP_RAM_ADDR_W, HUFFMAN_WEIGHTS_TMP_RAM_DATA_W, HUFFMAN_WEIGHTS_TMP_RAM_NUM_PARTITIONS>;
    type HuffmanWeightsTmpRamWrResp = ram::WriteResp;

    type HuffmanWeightsTmp2RamRdReq = ram::ReadReq<HUFFMAN_WEIGHTS_TMP2_RAM_ADDR_W, HUFFMAN_WEIGHTS_TMP2_RAM_NUM_PARTITIONS>;
    type HuffmanWeightsTmp2RamRdResp = ram::ReadResp<HUFFMAN_WEIGHTS_TMP2_RAM_DATA_W>;
    type HuffmanWeightsTmp2RamWrReq = ram::WriteReq<HUFFMAN_WEIGHTS_TMP2_RAM_ADDR_W, HUFFMAN_WEIGHTS_TMP2_RAM_DATA_W, HUFFMAN_WEIGHTS_TMP2_RAM_NUM_PARTITIONS>;
    type HuffmanWeightsTmp2RamWrResp = ram::WriteResp;

    type HuffmanWeightsFseRamRdReq = ram::ReadReq<HUFFMAN_WEIGHTS_FSE_RAM_ADDR_W, HUFFMAN_WEIGHTS_FSE_RAM_NUM_PARTITIONS>;
    type HuffmanWeightsFseRamRdResp = ram::ReadResp<HUFFMAN_WEIGHTS_FSE_RAM_DATA_W>;
    type HuffmanWeightsFseRamWrReq = ram::WriteReq<HUFFMAN_WEIGHTS_FSE_RAM_ADDR_W, HUFFMAN_WEIGHTS_FSE_RAM_DATA_W, HUFFMAN_WEIGHTS_FSE_RAM_NUM_PARTITIONS>;
    type HuffmanWeightsFseRamWrResp = ram::WriteResp;

    type LiteralsHeaderDecoderResp = literals_block_header_dec::LiteralsHeaderDecoderResp;
    type LiteralsBlockType = literals_block_header_dec::LiteralsBlockType;
    type LiteralsDecReq = literals_decoder::LiteralsDecoderCtrlReq<AXI_ADDR_W>;
    type LiteralsDecResp = literals_decoder::LiteralsDecoderCtrlResp;
    type LiteralsBufCtrl = common::LiteralsBufferCtrl;
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

    type AxiAddrW = uN[AXI_ADDR_W];

    req_r: chan<Req> in;
    resp_s: chan<Resp> out;

    lit_ctrl_req_s: chan<LiteralsDecReq> out;
    lit_ctrl_resp_r: chan<LiteralsDecResp> in;
    lit_ctrl_header_r: chan<LiteralsHeaderDecoderResp> in;

    seq_dec_req_s: chan<SequenceDecReq> out;
    seq_dec_resp_r: chan<SequenceDecResp> in;

    init {}

    config(
        req_r: chan<Req> in,
        resp_s: chan<Resp> out,

        // output from Command constructor to Sequence executor
        cmd_constr_out_s: chan<ExtendedPacket> out,

        // Sequence Decoder channels

        // Sequence Conf Decoder (manager)
        scd_axi_ar_s: chan<MemAxiAr> out,
        scd_axi_r_r: chan<MemAxiR> in,

        // Fse Lookup Decoder (manager)
        fld_axi_ar_s: chan<MemAxiAr> out,
        fld_axi_r_r: chan<MemAxiR> in,

        // FSE decoder (manager)
        fd_axi_ar_s: chan<MemAxiAr> out,
        fd_axi_r_r: chan<MemAxiR> in,

        // RAMs for FSE decoder
        dpd_rd_req_s: chan<SeqDpdRamRdReq> out,
        dpd_rd_resp_r: chan<SeqDpdRamRdResp> in,
        dpd_wr_req_s: chan<SeqDpdRamWrReq> out,
        dpd_wr_resp_r: chan<SeqDpdRamWrResp> in,

        tmp_rd_req_s: chan<SeqTmpRamRdReq> out,
        tmp_rd_resp_r: chan<SeqTmpRamRdResp> in,
        tmp_wr_req_s: chan<SeqTmpRamWrReq> out,
        tmp_wr_resp_r: chan<SeqTmpRamWrResp> in,

        tmp2_rd_req_s: chan<SeqTmp2RamRdReq> out,
        tmp2_rd_resp_r: chan<SeqTmp2RamRdResp> in,
        tmp2_wr_req_s: chan<SeqTmp2RamWrReq> out,
        tmp2_wr_resp_r: chan<SeqTmp2RamWrResp> in,

        ll_def_fse_rd_req_s: chan<SeqFseRamRdReq> out,
        ll_def_fse_rd_resp_r: chan<SeqFseRamRdResp> in,
        ll_def_fse_wr_req_s: chan<SeqFseRamWrReq> out,
        ll_def_fse_wr_resp_r: chan<SeqFseRamWrResp> in,

        ll_fse_rd_req_s: chan<SeqFseRamRdReq> out,
        ll_fse_rd_resp_r: chan<SeqFseRamRdResp> in,
        ll_fse_wr_req_s: chan<SeqFseRamWrReq> out,
        ll_fse_wr_resp_r: chan<SeqFseRamWrResp> in,

        ml_def_fse_rd_req_s: chan<SeqFseRamRdReq> out,
        ml_def_fse_rd_resp_r: chan<SeqFseRamRdResp> in,
        ml_def_fse_wr_req_s: chan<SeqFseRamWrReq> out,
        ml_def_fse_wr_resp_r: chan<SeqFseRamWrResp> in,

        ml_fse_rd_req_s: chan<SeqFseRamRdReq> out,
        ml_fse_rd_resp_r: chan<SeqFseRamRdResp> in,
        ml_fse_wr_req_s: chan<SeqFseRamWrReq> out,
        ml_fse_wr_resp_r: chan<SeqFseRamWrResp> in,

        of_def_fse_rd_req_s: chan<SeqFseRamRdReq> out,
        of_def_fse_rd_resp_r: chan<SeqFseRamRdResp> in,
        of_def_fse_wr_req_s: chan<SeqFseRamWrReq> out,
        of_def_fse_wr_resp_r: chan<SeqFseRamWrResp> in,

        of_fse_rd_req_s: chan<SeqFseRamRdReq> out,
        of_fse_rd_resp_r: chan<SeqFseRamRdResp> in,
        of_fse_wr_req_s: chan<SeqFseRamWrReq> out,
        of_fse_wr_resp_r: chan<SeqFseRamWrResp> in,

        // Literals decoder channels

        // AXI Literals Header Decoder (manager)
        lit_header_axi_ar_s: chan<MemAxiAr> out,
        lit_header_axi_r_r: chan<MemAxiR> in,

        // AXI Raw Literals Decoder (manager)
        raw_lit_axi_ar_s: chan<MemAxiAr> out,
        raw_lit_axi_r_r: chan<MemAxiR> in,

        // AXI Huffman Literals Decoder (manager)
        huffman_lit_axi_ar_s: chan<MemAxiAr> out,
        huffman_lit_axi_r_r: chan<MemAxiR> in,

        // AXI Huffman Jump Table Decoder (manager)
        huffman_jump_table_axi_ar_s: chan<MemAxiAr> out,
        huffman_jump_table_axi_r_r: chan<MemAxiR> in,

        // AXI Huffman Weights Header Decoder (manager)
        huffman_weights_header_axi_ar_s: chan<MemAxiAr> out,
        huffman_weights_header_axi_r_r: chan<MemAxiR> in,

        // AXI Huffman Weights RAW Decoder (manager)
        huffman_weights_raw_axi_ar_s: chan<MemAxiAr> out,
        huffman_weights_raw_axi_r_r: chan<MemAxiR> in,

        // AXI Huffman Weights FSE Decoder (manager)
        huffman_weights_fse_lookup_dec_axi_ar_s: chan<MemAxiAr> out,
        huffman_weights_fse_lookup_dec_axi_r_r: chan<MemAxiR> in,

        // AXI Huffman Weights FSE Decoder (manager)
        huffman_weights_fse_decoder_dec_axi_ar_s: chan<MemAxiAr> out,
        huffman_weights_fse_decoder_dec_axi_r_r: chan<MemAxiR> in,

        // Literals buffer internal memory
        rd_req_m0_s: chan<LitBufRamRdReq> out,
        rd_req_m1_s: chan<LitBufRamRdReq> out,
        rd_req_m2_s: chan<LitBufRamRdReq> out,
        rd_req_m3_s: chan<LitBufRamRdReq> out,
        rd_req_m4_s: chan<LitBufRamRdReq> out,
        rd_req_m5_s: chan<LitBufRamRdReq> out,
        rd_req_m6_s: chan<LitBufRamRdReq> out,
        rd_req_m7_s: chan<LitBufRamRdReq> out,
        rd_resp_m0_r: chan<LitBufRamRdResp> in,
        rd_resp_m1_r: chan<LitBufRamRdResp> in,
        rd_resp_m2_r: chan<LitBufRamRdResp> in,
        rd_resp_m3_r: chan<LitBufRamRdResp> in,
        rd_resp_m4_r: chan<LitBufRamRdResp> in,
        rd_resp_m5_r: chan<LitBufRamRdResp> in,
        rd_resp_m6_r: chan<LitBufRamRdResp> in,
        rd_resp_m7_r: chan<LitBufRamRdResp> in,
        wr_req_m0_s: chan<LitBufRamWrReq> out,
        wr_req_m1_s: chan<LitBufRamWrReq> out,
        wr_req_m2_s: chan<LitBufRamWrReq> out,
        wr_req_m3_s: chan<LitBufRamWrReq> out,
        wr_req_m4_s: chan<LitBufRamWrReq> out,
        wr_req_m5_s: chan<LitBufRamWrReq> out,
        wr_req_m6_s: chan<LitBufRamWrReq> out,
        wr_req_m7_s: chan<LitBufRamWrReq> out,
        wr_resp_m0_r: chan<LitBufRamWrResp> in,
        wr_resp_m1_r: chan<LitBufRamWrResp> in,
        wr_resp_m2_r: chan<LitBufRamWrResp> in,
        wr_resp_m3_r: chan<LitBufRamWrResp> in,
        wr_resp_m4_r: chan<LitBufRamWrResp> in,
        wr_resp_m5_r: chan<LitBufRamWrResp> in,
        wr_resp_m6_r: chan<LitBufRamWrResp> in,
        wr_resp_m7_r: chan<LitBufRamWrResp> in,

        // Huffman weights memory
        huffman_lit_weights_mem_rd_req_s: chan<HuffmanWeightsReadReq> out,
        huffman_lit_weights_mem_rd_resp_r: chan<HuffmanWeightsReadResp> in,
        huffman_lit_weights_mem_wr_req_s: chan<HuffmanWeightsWriteReq> out,
        huffman_lit_weights_mem_wr_resp_r: chan<HuffmanWeightsWriteResp> in,

        // Huffman prescan memory
        huffman_lit_prescan_mem_rd_req_s: chan<HuffmanPrescanReadReq> out,
        huffman_lit_prescan_mem_rd_resp_r: chan<HuffmanPrescanReadResp> in,
        huffman_lit_prescan_mem_wr_req_s: chan<HuffmanPrescanWriteReq> out,
        huffman_lit_prescan_mem_wr_resp_r: chan<HuffmanPrescanWriteResp> in,

        huffman_lit_weights_dpd_rd_req_s: chan<HuffmanWeightsDpdRamRdReq> out,
        huffman_lit_weights_dpd_rd_resp_r: chan<HuffmanWeightsDpdRamRdResp> in,
        huffman_lit_weights_dpd_wr_req_s: chan<HuffmanWeightsDpdRamWrReq> out,
        huffman_lit_weights_dpd_wr_resp_r: chan<HuffmanWeightsDpdRamWrResp> in,

        huffman_lit_weights_tmp_rd_req_s: chan<HuffmanWeightsTmpRamRdReq> out,
        huffman_lit_weights_tmp_rd_resp_r: chan<HuffmanWeightsTmpRamRdResp> in,
        huffman_lit_weights_tmp_wr_req_s: chan<HuffmanWeightsTmpRamWrReq> out,
        huffman_lit_weights_tmp_wr_resp_r: chan<HuffmanWeightsTmpRamWrResp> in,

        huffman_lit_weights_tmp2_rd_req_s: chan<HuffmanWeightsTmp2RamRdReq> out,
        huffman_lit_weights_tmp2_rd_resp_r: chan<HuffmanWeightsTmp2RamRdResp> in,
        huffman_lit_weights_tmp2_wr_req_s: chan<HuffmanWeightsTmp2RamWrReq> out,
        huffman_lit_weights_tmp2_wr_resp_r: chan<HuffmanWeightsTmp2RamWrResp> in,

        huffman_lit_weights_fse_rd_req_s: chan<HuffmanWeightsFseRamRdReq> out,
        huffman_lit_weights_fse_rd_resp_r: chan<HuffmanWeightsFseRamRdResp> in,
        huffman_lit_weights_fse_wr_req_s: chan<HuffmanWeightsFseRamWrReq> out,
        huffman_lit_weights_fse_wr_resp_r: chan<HuffmanWeightsFseRamWrResp> in,
    ) {
        // TODO: for consistency all MemReaders should be in toplevel ZSTD decoder
        // so we should move them up in the hierarchy from LiteralsDecoder
        // and SequenceDecoder to the toplevel
        const CHANNEL_DEPTH = u32:1;

        let (lit_ctrl_req_s, lit_ctrl_req_r) = chan<LiteralsDecReq, CHANNEL_DEPTH>("lit_ctrl_req");
        let (lit_ctrl_resp_s, lit_ctrl_resp_r) = chan<LiteralsDecResp, CHANNEL_DEPTH>("lit_ctrl_resp");
        let (lit_ctrl_header_s, lit_ctrl_header_r) = chan<LiteralsHeaderDecoderResp, CHANNEL_DEPTH>("lit_header");

        let (lit_buf_ctrl_s, lit_buf_ctrl_r) = chan<LiteralsBufCtrl, CHANNEL_DEPTH>("lit_buf_ctrl");
        let (lit_buf_out_s, lit_buf_out_r) = chan<SequenceExecutorPacket, CHANNEL_DEPTH>("lit_buf_out");

        spawn literals_decoder::LiteralsDecoder<
            HISTORY_BUFFER_SIZE_KB,
            AXI_DATA_W, AXI_ADDR_W, AXI_ID_W, AXI_DEST_W,
            HUFFMAN_WEIGHTS_DPD_RAM_ADDR_W, HUFFMAN_WEIGHTS_DPD_RAM_DATA_W, HUFFMAN_WEIGHTS_DPD_RAM_NUM_PARTITIONS,
            HUFFMAN_WEIGHTS_TMP_RAM_ADDR_W, HUFFMAN_WEIGHTS_TMP_RAM_DATA_W, HUFFMAN_WEIGHTS_TMP_RAM_NUM_PARTITIONS,
            HUFFMAN_WEIGHTS_TMP2_RAM_ADDR_W, HUFFMAN_WEIGHTS_TMP2_RAM_DATA_W, HUFFMAN_WEIGHTS_TMP2_RAM_NUM_PARTITIONS,
            HUFFMAN_WEIGHTS_FSE_RAM_ADDR_W, HUFFMAN_WEIGHTS_FSE_RAM_DATA_W, HUFFMAN_WEIGHTS_FSE_RAM_NUM_PARTITIONS,
            HUFFMAN_WEIGHTS_RAM_ADDR_W, HUFFMAN_WEIGHTS_RAM_DATA_W, HUFFMAN_WEIGHTS_RAM_NUM_PARTITIONS,
            HUFFMAN_PRESCAN_RAM_ADDR_W, HUFFMAN_PRESCAN_RAM_DATA_W, HUFFMAN_PRESCAN_RAM_NUM_PARTITIONS,
        >(
            lit_header_axi_ar_s, lit_header_axi_r_r,
            raw_lit_axi_ar_s, raw_lit_axi_r_r,
            huffman_lit_axi_ar_s, huffman_lit_axi_r_r,
            huffman_jump_table_axi_ar_s, huffman_jump_table_axi_r_r,
            huffman_weights_header_axi_ar_s, huffman_weights_header_axi_r_r,
            huffman_weights_raw_axi_ar_s, huffman_weights_raw_axi_r_r,
            huffman_weights_fse_lookup_dec_axi_ar_s, huffman_weights_fse_lookup_dec_axi_r_r,
            huffman_weights_fse_decoder_dec_axi_ar_s, huffman_weights_fse_decoder_dec_axi_r_r,
            lit_ctrl_req_r, lit_ctrl_resp_s, lit_ctrl_header_s,
            lit_buf_ctrl_r, lit_buf_out_s,
            rd_req_m0_s, rd_req_m1_s, rd_req_m2_s, rd_req_m3_s,
            rd_req_m4_s, rd_req_m5_s, rd_req_m6_s, rd_req_m7_s,
            rd_resp_m0_r, rd_resp_m1_r, rd_resp_m2_r, rd_resp_m3_r,
            rd_resp_m4_r, rd_resp_m5_r, rd_resp_m6_r, rd_resp_m7_r,
            wr_req_m0_s, wr_req_m1_s, wr_req_m2_s, wr_req_m3_s,
            wr_req_m4_s, wr_req_m5_s, wr_req_m6_s, wr_req_m7_s,
            wr_resp_m0_r, wr_resp_m1_r, wr_resp_m2_r, wr_resp_m3_r,
            wr_resp_m4_r, wr_resp_m5_r, wr_resp_m6_r, wr_resp_m7_r,
            huffman_lit_weights_mem_rd_req_s, huffman_lit_weights_mem_rd_resp_r,
            huffman_lit_weights_mem_wr_req_s, huffman_lit_weights_mem_wr_resp_r,
            huffman_lit_prescan_mem_rd_req_s, huffman_lit_prescan_mem_rd_resp_r,
            huffman_lit_prescan_mem_wr_req_s, huffman_lit_prescan_mem_wr_resp_r,
            huffman_lit_weights_dpd_rd_req_s, huffman_lit_weights_dpd_rd_resp_r,
            huffman_lit_weights_dpd_wr_req_s, huffman_lit_weights_dpd_wr_resp_r,
            huffman_lit_weights_tmp_rd_req_s, huffman_lit_weights_tmp_rd_resp_r,
            huffman_lit_weights_tmp_wr_req_s, huffman_lit_weights_tmp_wr_resp_r,
            huffman_lit_weights_tmp2_rd_req_s, huffman_lit_weights_tmp2_rd_resp_r,
            huffman_lit_weights_tmp2_wr_req_s, huffman_lit_weights_tmp2_wr_resp_r,
            huffman_lit_weights_fse_rd_req_s, huffman_lit_weights_fse_rd_resp_r,
            huffman_lit_weights_fse_wr_req_s, huffman_lit_weights_fse_wr_resp_r,
        );

        let (seq_dec_req_s, seq_dec_req_r) = chan<SequenceDecReq, CHANNEL_DEPTH>("seq_dec_req");
        let (seq_dec_resp_s, seq_dec_resp_r) = chan<SequenceDecResp, CHANNEL_DEPTH>("seq_dec_resp");
        let (seq_dec_command_s, seq_dec_command_r) = chan<CommandConstructorData, CHANNEL_DEPTH>("seq_dec_command");

        spawn sequence_dec::SequenceDecoder<
            AXI_ADDR_W, AXI_DATA_W, AXI_DEST_W, AXI_ID_W,
            SEQ_DPD_RAM_ADDR_W, SEQ_DPD_RAM_DATA_W, SEQ_DPD_RAM_NUM_PARTITIONS,
            SEQ_TMP_RAM_ADDR_W, SEQ_TMP_RAM_DATA_W, SEQ_TMP_RAM_NUM_PARTITIONS,
            SEQ_TMP2_RAM_ADDR_W, SEQ_TMP2_RAM_DATA_W, SEQ_TMP2_RAM_NUM_PARTITIONS,
            SEQ_FSE_RAM_ADDR_W, SEQ_FSE_RAM_DATA_W, SEQ_FSE_RAM_NUM_PARTITIONS,
        >(
            scd_axi_ar_s, scd_axi_r_r,
            fld_axi_ar_s, fld_axi_r_r,
            fd_axi_ar_s, fd_axi_r_r,
            seq_dec_req_r, seq_dec_resp_s,
            seq_dec_command_s,
            dpd_rd_req_s, dpd_rd_resp_r, dpd_wr_req_s, dpd_wr_resp_r,
            tmp_rd_req_s, tmp_rd_resp_r, tmp_wr_req_s, tmp_wr_resp_r,
            tmp2_rd_req_s, tmp2_rd_resp_r, tmp2_wr_req_s, tmp2_wr_resp_r,
            ll_def_fse_rd_req_s, ll_def_fse_rd_resp_r, ll_def_fse_wr_req_s, ll_def_fse_wr_resp_r,
            ll_fse_rd_req_s, ll_fse_rd_resp_r, ll_fse_wr_req_s, ll_fse_wr_resp_r,
            ml_def_fse_rd_req_s, ml_def_fse_rd_resp_r, ml_def_fse_wr_req_s, ml_def_fse_wr_resp_r,
            ml_fse_rd_req_s, ml_fse_rd_resp_r, ml_fse_wr_req_s, ml_fse_wr_resp_r,
            of_def_fse_rd_req_s, of_def_fse_rd_resp_r, of_def_fse_wr_req_s, of_def_fse_wr_resp_r,
            of_fse_rd_req_s, of_fse_rd_resp_r, of_fse_wr_req_s, of_fse_wr_resp_r,
        );

        spawn command_constructor::CommandConstructor(
            seq_dec_command_r,
            cmd_constr_out_s,
            lit_buf_out_r,
            lit_buf_ctrl_s,
        );

        (
            req_r, resp_s,
            lit_ctrl_req_s, lit_ctrl_resp_r, lit_ctrl_header_r,
            seq_dec_req_s, seq_dec_resp_r,
        )
    }

    next(_: ()) {
        let tok = join();

        let (tok_req, req) = recv(tok, req_r);
        trace_fmt!("[CompressBlockDecoder] Received request: {:#x}", req);

        let lit_ctrl_req = LiteralsDecReq {
            addr: req.addr,
            literals_last: req.last_block,
        };
        let tok_lit1 = send(tok_req, lit_ctrl_req_s, lit_ctrl_req);
        trace_fmt!("[CompressBlockDecoder] Sent lit_ctrl_req: {:#x}", lit_ctrl_req);

        let (tok_lit2, lit_header) = recv(tok_lit1, lit_ctrl_header_r);
        trace_fmt!("[CompressBlockDecoder] Received lit_header: {:#x}", lit_header);

        let seq_section_offset = lit_header.length as AxiAddrW + match (lit_header.header.literal_type) {
            LiteralsBlockType::RAW => lit_header.header.regenerated_size,
            LiteralsBlockType::RLE => u20:1,
            LiteralsBlockType::COMP | LiteralsBlockType::COMP_4 => lit_header.header.compressed_size,
            LiteralsBlockType::TREELESS | LiteralsBlockType::TREELESS_4 => lit_header.header.compressed_size,
            _ => fail!("comp_block_dec_unreachable", u20:0),
        } as AxiAddrW;

        let seq_section_start = req.addr + seq_section_offset;
        let seq_section_end = req.addr + req.length as AxiAddrW;

        let (tok_fin_lit, lit_resp) = recv(tok_lit1, lit_ctrl_resp_r);
        trace_fmt!("[CompressBlockDecoder] Received lit_ctrl_resp: {:#x}", lit_resp);

        let seq_req = SequenceDecReq {
            start_addr: seq_section_start,
            end_addr: seq_section_end,
            sync:  BlockSyncData {
                id: req.id,
                last_block: req.last_block,
           },
           literals_count: lit_header.header.regenerated_size,
        };

        trace_fmt!("[CompressBlockDecoder] Sending sequence req: {:#x}", seq_req);
        let tok_seq = send(tok_fin_lit, seq_dec_req_s, seq_req);

        let (tok_fin_seq, seq_resp) = recv(tok_seq, seq_dec_resp_r);
        trace_fmt!("[CompressBlockDecoder] Received sequence resp: {:#x}", seq_resp);

        let tok_finish = join(tok_fin_lit, tok_fin_seq);
        send(tok_finish, resp_s, Resp {
            status: CompressBlockDecoderStatus::OK
        });
    }
}

const TEST_CASE_RAM_DATA_W = u32:64;
const TEST_CASE_RAM_SIZE = u32:256;
const TEST_CASE_RAM_ADDR_W = std::clog2(TEST_CASE_RAM_SIZE);
const TEST_CASE_RAM_WORD_PARTITION_SIZE = TEST_CASE_RAM_DATA_W / u32:8;
const TEST_CASE_RAM_NUM_PARTITIONS = ram::num_partitions(
    TEST_CASE_RAM_WORD_PARTITION_SIZE, TEST_CASE_RAM_DATA_W);
const TEST_CASE_RAM_BASE_ADDR = u32:0;

const TEST_AXI_DATA_W = u32:64;
const TEST_AXI_ADDR_W = u32:32;
const TEST_AXI_ID_W = u32:4;
const TEST_AXI_DEST_W = u32:4;
const TEST_AXI_DATA_W_DIV8 = TEST_AXI_DATA_W / u32:8;

const TEST_SEQ_DPD_RAM_DATA_W = u32:16;
const TEST_SEQ_DPD_RAM_SIZE = u32:256;
const TEST_SEQ_DPD_RAM_ADDR_W = std::clog2(TEST_SEQ_DPD_RAM_SIZE);
const TEST_SEQ_DPD_RAM_WORD_PARTITION_SIZE = TEST_SEQ_DPD_RAM_DATA_W;
const TEST_SEQ_DPD_RAM_NUM_PARTITIONS = ram::num_partitions(
    TEST_SEQ_DPD_RAM_WORD_PARTITION_SIZE, TEST_SEQ_DPD_RAM_DATA_W);

const TEST_SEQ_FSE_RAM_DATA_W = u32:32;
const TEST_SEQ_FSE_RAM_SIZE = u32:1 << common::FSE_MAX_ACCURACY_LOG;
const TEST_SEQ_FSE_RAM_ADDR_W = std::clog2(TEST_SEQ_FSE_RAM_SIZE);
const TEST_SEQ_FSE_RAM_WORD_PARTITION_SIZE = TEST_SEQ_FSE_RAM_DATA_W;
const TEST_SEQ_FSE_RAM_NUM_PARTITIONS = ram::num_partitions(
    TEST_SEQ_FSE_RAM_WORD_PARTITION_SIZE, TEST_SEQ_FSE_RAM_DATA_W);

const TEST_SEQ_TMP_RAM_DATA_W = u32:16;
const TEST_SEQ_TMP_RAM_SIZE = u32:256;
const TEST_SEQ_TMP_RAM_ADDR_W = std::clog2(TEST_SEQ_TMP_RAM_SIZE);
const TEST_SEQ_TMP_RAM_WORD_PARTITION_SIZE = TEST_SEQ_TMP_RAM_DATA_W;
const TEST_SEQ_TMP_RAM_NUM_PARTITIONS = ram::num_partitions(
    TEST_SEQ_TMP_RAM_WORD_PARTITION_SIZE, TEST_SEQ_TMP_RAM_DATA_W);

const TEST_SEQ_TMP2_RAM_DATA_W = u32:8;
const TEST_SEQ_TMP2_RAM_SIZE = u32:512;
const TEST_SEQ_TMP2_RAM_ADDR_W = std::clog2(TEST_SEQ_TMP2_RAM_SIZE);
const TEST_SEQ_TMP2_RAM_WORD_PARTITION_SIZE = TEST_SEQ_TMP2_RAM_DATA_W;
const TEST_SEQ_TMP2_RAM_NUM_PARTITIONS = ram::num_partitions(TEST_SEQ_TMP2_RAM_WORD_PARTITION_SIZE, TEST_SEQ_TMP2_RAM_DATA_W);

const TEST_RAM_SIM_RW_BEHAVIOR = ram::SimultaneousReadWriteBehavior::READ_BEFORE_WRITE;
const TEST_RAM_INITIALIZED = true;

const HISTORY_BUFFER_SIZE_KB = common::HISTORY_BUFFER_SIZE_KB;

const TEST_HUFFMAN_WEIGHTS_DPD_RAM_DATA_W = u32:16;
const TEST_HUFFMAN_WEIGHTS_DPD_RAM_SIZE = u32:256;
const TEST_HUFFMAN_WEIGHTS_DPD_RAM_ADDR_W = std::clog2(TEST_HUFFMAN_WEIGHTS_DPD_RAM_SIZE);
const TEST_HUFFMAN_WEIGHTS_DPD_RAM_WORD_PARTITION_SIZE = TEST_HUFFMAN_WEIGHTS_DPD_RAM_DATA_W;
const TEST_HUFFMAN_WEIGHTS_DPD_RAM_NUM_PARTITIONS = ram::num_partitions(
    TEST_HUFFMAN_WEIGHTS_DPD_RAM_WORD_PARTITION_SIZE, TEST_HUFFMAN_WEIGHTS_DPD_RAM_DATA_W);

const TEST_HUFFMAN_WEIGHTS_FSE_RAM_DATA_W = u32:32;
const TEST_HUFFMAN_WEIGHTS_FSE_RAM_SIZE = u32:256;
const TEST_HUFFMAN_WEIGHTS_FSE_RAM_ADDR_W = std::clog2(TEST_HUFFMAN_WEIGHTS_FSE_RAM_SIZE);
const TEST_HUFFMAN_WEIGHTS_FSE_RAM_WORD_PARTITION_SIZE = TEST_HUFFMAN_WEIGHTS_FSE_RAM_DATA_W / u32:3;
const TEST_HUFFMAN_WEIGHTS_FSE_RAM_NUM_PARTITIONS = ram::num_partitions(
    TEST_HUFFMAN_WEIGHTS_FSE_RAM_WORD_PARTITION_SIZE, TEST_HUFFMAN_WEIGHTS_FSE_RAM_DATA_W);

const TEST_HUFFMAN_WEIGHTS_TMP_RAM_DATA_W = u32:16;
const TEST_HUFFMAN_WEIGHTS_TMP_RAM_SIZE = u32:256;
const TEST_HUFFMAN_WEIGHTS_TMP_RAM_ADDR_W = std::clog2(TEST_HUFFMAN_WEIGHTS_TMP_RAM_SIZE);
const TEST_HUFFMAN_WEIGHTS_TMP_RAM_WORD_PARTITION_SIZE = TEST_HUFFMAN_WEIGHTS_TMP_RAM_DATA_W;
const TEST_HUFFMAN_WEIGHTS_TMP_RAM_NUM_PARTITIONS = ram::num_partitions(
    TEST_HUFFMAN_WEIGHTS_TMP_RAM_WORD_PARTITION_SIZE, TEST_HUFFMAN_WEIGHTS_TMP_RAM_DATA_W);

const TEST_HUFFMAN_WEIGHTS_TMP2_RAM_DATA_W = u32:8;
const TEST_HUFFMAN_WEIGHTS_TMP2_RAM_SIZE = u32:512;
const TEST_HUFFMAN_WEIGHTS_TMP2_RAM_ADDR_W = std::clog2(TEST_HUFFMAN_WEIGHTS_TMP2_RAM_SIZE);
const TEST_HUFFMAN_WEIGHTS_TMP2_RAM_WORD_PARTITION_SIZE = TEST_HUFFMAN_WEIGHTS_TMP2_RAM_DATA_W;
const TEST_HUFFMAN_WEIGHTS_TMP2_RAM_NUM_PARTITIONS = ram::num_partitions(
    TEST_HUFFMAN_WEIGHTS_TMP2_RAM_WORD_PARTITION_SIZE, TEST_HUFFMAN_WEIGHTS_TMP2_RAM_DATA_W);

const TEST_HUFFMAN_WEIGHTS_RAM_DATA_W: u32 = huffman_literals_dec::WEIGHTS_DATA_WIDTH;
const TEST_HUFFMAN_WEIGHTS_RAM_SIZE = huffman_literals_dec::RAM_SIZE;
const TEST_HUFFMAN_WEIGHTS_RAM_ADDR_W: u32 = huffman_literals_dec::WEIGHTS_ADDR_WIDTH;
const TEST_HUFFMAN_WEIGHTS_RAM_WORD_PARTITION_SIZE = huffman_literals_dec::WEIGHTS_PARTITION_WORD_SIZE;
const TEST_HUFFMAN_WEIGHTS_RAM_NUM_PARTITIONS: u32 = huffman_literals_dec::WEIGHTS_NUM_PARTITIONS;

const TEST_HUFFMAN_PRESCAN_RAM_DATA_W: u32 = huffman_literals_dec::PRESCAN_DATA_WIDTH;
const TEST_HUFFMAN_PRESCAN_RAM_SIZE = huffman_literals_dec::RAM_SIZE;
const TEST_HUFFMAN_PRESCAN_RAM_ADDR_W: u32 = huffman_literals_dec::PRESCAN_ADDR_WIDTH;
const TEST_HUFFMAN_PRESCAN_RAM_WORD_PARTITION_SIZE = huffman_literals_dec::PRESCAN_PARTITION_WORD_SIZE;
const TEST_HUFFMAN_PRESCAN_RAM_NUM_PARTITIONS: u32 = huffman_literals_dec::PRESCAN_NUM_PARTITIONS;

const TEST_LITERALS_BUFFER_RAM_ADDR_W: u32 = parallel_rams::ram_addr_width(HISTORY_BUFFER_SIZE_KB);
const TEST_LITERALS_BUFFER_RAM_SIZE: u32 = parallel_rams::ram_size(HISTORY_BUFFER_SIZE_KB);
const TEST_LITERALS_BUFFER_RAM_DATA_W: u32 = literals_buffer::RAM_DATA_WIDTH;
const TEST_LITERALS_BUFFER_RAM_NUM_PARTITIONS: u32 = literals_buffer::RAM_NUM_PARTITIONS;
const TEST_LITERALS_BUFFER_RAM_WORD_PARTITION_SIZE: u32 = TEST_LITERALS_BUFFER_RAM_DATA_W;

const AXI_CHAN_N = u32:11;

// testcase format:
// - block length (without block header, essentially length of sequences + literals sections),
// - literals and sequences sections as they appear in memory
// - expected output size
// - expected output
const COMP_BLOCK_DEC_TESTCASES: (u32, u64[64], u32, ExtendedPacket[128])[7] = [
    // RAW
    (
        // Test case 0
        // raw literals (18) + sequences with 3 predefined tables (2)
        //
        // last block generated with:
        // ./decodecorpus -pdata2.out -odata2.in -s7110  --block-type=2 --content-size --literal-type=0 --max-block-size-log=5
        u32:0x1C,
        u64[64]:[
            u64:0x0, u64:0x0,        // 0x000
            u64:0x0, u64:0x0,        // 0x010
            u64:0x0, u64:0x0,        // 0x020
            u64:0x0, u64:0x0,        // 0x030
            u64:0x0, u64:0x0,        // 0x040
            u64:0x0, u64:0x0,        // 0x050
            u64:0x0, u64:0x0,        // 0x060
            u64:0x0, u64:0x0,        // 0x070
            u64:0x0, u64:0x0,        // 0x080
            u64:0x0, u64:0x0,        // 0x090
            u64:0x0, u64:0x0,        // 0x0A0
            u64:0x0, u64:0x0,        // 0x0B0
            u64:0x0, u64:0x0,        // 0x0C0
            u64:0x0, u64:0x0,        // 0x0D0
            u64:0x0, u64:0x0,        // 0x0E0
            u64:0x0, u64:0x0,        // 0x0F0
            u64:0x1fba7f9f15523990,
            u64:0x43e75b86b1dfe343,
            u64:0xc0423000200d6c6,
            u64:0x252c492,
            u64:0, ...
        ],
        u32:6,
        ExtendedPacket[128]:[
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:0, data: u64:0x431fba7f9f155239, length: u32:8 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:0, data: u64:0xe75b86b1dfe3, length: u32:6 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::SEQUENCE,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:0, data: u64:0x192, length: u32:8 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:0, data: u64:0xd6c643, length: u32:3 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::SEQUENCE,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:0, data: u64:0x223, length: u32:3 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                packet: BlockDataPacket { last: true, last_block: false, id: u32:0, data: u64:0x00, length: u32:1 }
            },
            zero!<ExtendedPacket>(), ...
        ]
    ),
    (
        // Test case 1
        // raw literals (64) + sequences with 3 predefined tables (1)
        //
        // last block generated with:
        // ./decodecorpus -pdata2.out -odata2.in -s35304 --block-type=2 --content-size --literal-type=0 --max-block-size-log=7
        u32:0x48,
        u64[64]:[
            u64:0x0, u64:0x0,        // 0x000
            u64:0x0, u64:0x0,        // 0x010
            u64:0x0, u64:0x0,        // 0x020
            u64:0x0, u64:0x0,        // 0x030
            u64:0x0, u64:0x0,        // 0x040
            u64:0x0, u64:0x0,        // 0x050
            u64:0x0, u64:0x0,        // 0x060
            u64:0x0, u64:0x0,        // 0x070
            u64:0x0, u64:0x0,        // 0x080
            u64:0x0, u64:0x0,        // 0x090
            u64:0x0, u64:0x0,        // 0x0A0
            u64:0x0, u64:0x0,        // 0x0B0
            u64:0x0, u64:0x0,        // 0x0C0
            u64:0x0, u64:0x0,        // 0x0D0
            u64:0x0, u64:0x0,        // 0x0E0
            u64:0x0, u64:0x0,        // 0x0F0
            u64:0xc792801500520404,
            u64:0x9be2210a8b13a2bb,
            u64:0x291994532c422e15,
            u64:0x1c37a8940c112bcd,
            u64:0xc95f959fa34764de,
            u64:0x57c1079b679780bb,
            u64:0x7a819dd90c2f2b97,
            u64:0x5a829f58ba369e42,
            u64:0x13d608b30001d27d,
            u64:0, ...
        ],
        u32:10,
        ExtendedPacket[128]:[
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:1, data: u64:0xa2bbc79280150052, length: u32:8 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:1, data: u64:0x2e159be2210a8b13, length: u32:8 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:1, data: u64:0x2bcd291994532c42, length: u32:8 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:1, data: u64:0x64de1c37a8940c11, length: u32:8 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:1, data: u64:0x9fa347, length: u32:3 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::SEQUENCE,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:1, data: u64:0x116, length: u32:4 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:1, data: u64:0x9b679780bbc95f95, length: u32:8 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:1, data: u64:0xd90c2f2b9757c107, length: u32:8 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:1, data: u64:0x58ba369e427a819d, length: u32:8 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                packet: BlockDataPacket { last: true, last_block: false, id: u32:1, data: u64:0xd27d5a829f, length: u32:5 }
            },
            zero!<ExtendedPacket>(), ...
        ]
    ),
    // RLE
    (
        // Test case 2
        // RLE literals (13) + sequences with 3 predefined tables (15)
        //
        // last block generated with:
        // ./decodecorpus -pdata2.out -odata2.in -s52123 --block-type=2 --content-size --literal-type=1 --max-block-size-log=7
        u32:0x35,
        u64[64]:[
            u64:0x0, u64:0x0,        // 0x000
            u64:0x0, u64:0x0,        // 0x010
            u64:0x0, u64:0x0,        // 0x020
            u64:0x0, u64:0x0,        // 0x030
            u64:0x0, u64:0x0,        // 0x040
            u64:0x0, u64:0x0,        // 0x050
            u64:0x0, u64:0x0,        // 0x060
            u64:0x0, u64:0x0,        // 0x070
            u64:0x0, u64:0x0,        // 0x080
            u64:0x0, u64:0x0,        // 0x090
            u64:0x0, u64:0x0,        // 0x0A0
            u64:0x0, u64:0x0,        // 0x0B0
            u64:0x0, u64:0x0,        // 0x0C0
            u64:0x0, u64:0x0,        // 0x0D0
            u64:0x0, u64:0x0,        // 0x0E0
            u64:0x0, u64:0x0,        // 0x0F0
            u64:0xf006ace2000f7669,
            u64:0xdd540313be00074e,
            u64:0xb005607a005e2056,
            u64:0xa8e58056222e0c33,
            u64:0x5404c001f64c80a,
            u64:0x834002e100f7dce,
            u64:0x40381ea080,
            u64:0, ...
        ],
        u32:30,
        ExtendedPacket[128]:[
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:2, data: u64:0x0, length: u32:0 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::SEQUENCE,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:2, data: u64:0x1f50, length: u32:3 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:2, data: u64:0x76767676, length: u32:4 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::SEQUENCE,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:2, data: u64:0x21a, length: u32:3 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:2, data: u64:0x0, length: u32:0 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::SEQUENCE,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:2, data: u64:0x2, length: u32:3 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:2, data: u64:0x0, length: u32:0 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::SEQUENCE,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:2, data: u64:0x1bee, length: u32:3 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:2, data: u64:0x76, length: u32:1 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::SEQUENCE,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:2, data: u64:0x2026, length: u32:3 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:2, data: u64:0x0, length: u32:0 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::SEQUENCE,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:2, data: u64:0x1d93, length: u32:3 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:2, data: u64:0x76, length: u32:1 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::SEQUENCE,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:2, data: u64:0x2a39, length: u32:3 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:2, data: u64:0x76, length: u32:1 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::SEQUENCE,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:2, data: u64:0x3111, length: u32:3 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:2, data: u64:0x76, length: u32:1 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::SEQUENCE,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:2, data: u64:0xe76, length: u32:4 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:2, data: u64:0x0, length: u32:0 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::SEQUENCE,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:2, data: u64:0x303d, length: u32:3 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:2, data: u64:0x0, length: u32:0 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::SEQUENCE,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:2, data: u64:0x3, length: u32:3 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:2, data: u64:0x0, length: u32:0 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::SEQUENCE,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:2, data: u64:0x36ea, length: u32:3 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:2, data: u64:0x7676767676, length: u32:5 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::SEQUENCE,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:2, data: u64:0x53be, length: u32:3 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:2, data: u64:0x0, length: u32:0 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::SEQUENCE,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:2, data: u64:0x14ef, length: u32:3 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:2, data: u64:0x0, length: u32:0 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::SEQUENCE,
                packet: BlockDataPacket { last: true, last_block: false, id: u32:2, data: u64:0x2ce2, length: u32:4 }
            },
            zero!<ExtendedPacket>(), ...
        ],
    ),
    (
        // Test case 3
        // RLE literals (102) + sequences with 3 predefined tables (2)
        //
        // last block generated with:
        // ./decodecorpus -pdata2.out -odata2.in -s52352 --block-type=2 --content-size --literal-type=1 --max-block-size-log=7
        u32:0xa,
        u64[64]:[
            u64:0x0, u64:0x0,        // 0x000
            u64:0x0, u64:0x0,        // 0x010
            u64:0x0, u64:0x0,        // 0x020
            u64:0x0, u64:0x0,        // 0x030
            u64:0x0, u64:0x0,        // 0x040
            u64:0x0, u64:0x0,        // 0x050
            u64:0x0, u64:0x0,        // 0x060
            u64:0x0, u64:0x0,        // 0x070
            u64:0x0, u64:0x0,        // 0x080
            u64:0x0, u64:0x0,        // 0x090
            u64:0x0, u64:0x0,        // 0x0A0
            u64:0x0, u64:0x0,        // 0x0B0
            u64:0x0, u64:0x0,        // 0x0C0
            u64:0x0, u64:0x0,        // 0x0D0
            u64:0x0, u64:0x0,        // 0x0E0
            u64:0x0, u64:0x0,        // 0x0F0
            u64:0x42184c0002f50665,
            u64:0x9570,
            u64:0, ...
        ],
        u32:16,
        ExtendedPacket[128]:[
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:3, data: u64:0xf5f5f5f5f5f5f5f5, length: u32:8 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:3, data: u64:0xf5f5f5f5f5f5, length: u32:6 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::SEQUENCE,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:3, data: u64:0x2, length: u32:4 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:3, data: u64:0xf5f5f5f5, length: u32:4 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::SEQUENCE,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:3, data: u64:0x4c, length: u32:6 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:3, data: u64:0xf5f5f5f5f5f5f5f5, length: u32:8 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:3, data: u64:0xf5f5f5f5f5f5f5f5, length: u32:8 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:3, data: u64:0xf5f5f5f5f5f5f5f5, length: u32:8 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:3, data: u64:0xf5f5f5f5f5f5f5f5, length: u32:8 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:3, data: u64:0xf5f5f5f5f5f5f5f5, length: u32:8 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:3, data: u64:0xf5f5f5f5f5f5f5f5, length: u32:8 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:3, data: u64:0xf5f5f5f5f5f5f5f5, length: u32:8 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:3, data: u64:0xf5f5f5f5f5f5f5f5, length: u32:8 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:3, data: u64:0xf5f5f5f5f5f5f5f5, length: u32:8 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:3, data: u64:0xf5f5f5f5f5f5f5f5, length: u32:8 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                packet: BlockDataPacket { last: true, last_block: false, id: u32:3, data: u64:0xf5f5f5f5, length: u32:4 }
            },
            zero!<ExtendedPacket>(), ...
        ]
    ),
    // Corner cases
    (
        // Test case 4
        // RLE literals (0) + sequences with 3 predefined tables (0)
        //
        // last block generated with:
        // ./decodecorpus -pdata2.out -odata2.in -s10761 --block-type=2 --content-size --literal-type=1 --max-block-size-log=7
        u32:0x3,
        u64[64]:[
            u64:0x0, u64:0x0,        // 0x000
            u64:0x0, u64:0x0,        // 0x010
            u64:0x0, u64:0x0,        // 0x020
            u64:0x0, u64:0x0,        // 0x030
            u64:0x0, u64:0x0,        // 0x040
            u64:0x0, u64:0x0,        // 0x050
            u64:0x0, u64:0x0,        // 0x060
            u64:0x0, u64:0x0,        // 0x070
            u64:0x0, u64:0x0,        // 0x080
            u64:0x0, u64:0x0,        // 0x090
            u64:0x0, u64:0x0,        // 0x0A0
            u64:0x0, u64:0x0,        // 0x0B0
            u64:0x0, u64:0x0,        // 0x0C0
            u64:0x0, u64:0x0,        // 0x0D0
            u64:0x0, u64:0x0,        // 0x0E0
            u64:0x0, u64:0x0,        // 0x0F0
            u64:0x1501,
            u64:0, ...
        ],
        u32:1,
        ExtendedPacket[128]:[
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                packet: BlockDataPacket { last: true, last_block: false, id: u32:4, data: u64:0x0, length: u32:0 }
            },
            zero!<ExtendedPacket>(), ...
        ]
    ),
    (
        // Test case 5
        // RLE literals (0) + sequences with 3 predefined tables (2)
        // last block generated with:
        //./decodecorpus -pdata2.out -odata2.in -s7294 --block-type=2 --content-size --literal-type=1 --max-block-size-log=7
        u32:0xc,
        u64[64]:[
            u64:0x0, u64:0x0,        // 0x000
            u64:0x0, u64:0x0,        // 0x010
            u64:0x0, u64:0x0,        // 0x020
            u64:0x0, u64:0x0,        // 0x030
            u64:0x0, u64:0x0,        // 0x040
            u64:0x0, u64:0x0,        // 0x050
            u64:0x0, u64:0x0,        // 0x060
            u64:0x0, u64:0x0,        // 0x070
            u64:0x0, u64:0x0,        // 0x080
            u64:0x0, u64:0x0,        // 0x090
            u64:0x0, u64:0x0,        // 0x0A0
            u64:0x0, u64:0x0,        // 0x0B0
            u64:0x0, u64:0x0,        // 0x0C0
            u64:0x0, u64:0x0,        // 0x0D0
            u64:0x0, u64:0x0,        // 0x0E0
            u64:0x0, u64:0x0,        // 0x0F0
            u64:0x6006ab770002fa01,
            u64:0x1020070,
            u64:0, ...
        ],
        u32:4,
        ExtendedPacket[128]:[
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:5, data: u64:0x0, length: u32:0 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::SEQUENCE,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:5, data: u64:0xf06, length: u32:3 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                packet: BlockDataPacket { last: false, last_block: false, id: u32:5, data: u64:0, length: u32:0 }
            },
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::SEQUENCE,
                packet: BlockDataPacket { last: true, last_block: false, id: u32:5, data: u64:0x2b77, length: u32:4 }
            },
            zero!<ExtendedPacket>(), ...
        ],
    ),
    (
        // Test case 6
        // RAW literals (2) + sequences with 3 predefined tables (0)
        // last block generated with:
        //./decodecorpus -pdata2.out -odata2.in -s38193 --block-type=2 --content-size --literal-type=0 --max-block-size-log=7
        u32:0x4,
        u64[64]:[
            u64:0x0, u64:0x0,        // 0x000
            u64:0x0, u64:0x0,        // 0x010
            u64:0x0, u64:0x0,        // 0x020
            u64:0x0, u64:0x0,        // 0x030
            u64:0x0, u64:0x0,        // 0x040
            u64:0x0, u64:0x0,        // 0x050
            u64:0x0, u64:0x0,        // 0x060
            u64:0x0, u64:0x0,        // 0x070
            u64:0x0, u64:0x0,        // 0x080
            u64:0x0, u64:0x0,        // 0x090
            u64:0x0, u64:0x0,        // 0x0A0
            u64:0x0, u64:0x0,        // 0x0B0
            u64:0x0, u64:0x0,        // 0x0C0
            u64:0x0, u64:0x0,        // 0x0D0
            u64:0x0, u64:0x0,        // 0x0E0
            u64:0x0, u64:0x0,        // 0x0F0
            u64:0x215a10,
            u64:0, ...
        ],
        u32:1,
        ExtendedPacket[128]:[
            ExtendedPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                packet: BlockDataPacket { last: true, last_block: false, id: u32:6, data: u64:0x215a, length: u32:2 }
            },
            zero!<ExtendedPacket>(), ...
        ],
    )
];

#[test_proc]
proc CompressBlockDecoderTest {
    type Req = CompressBlockDecoderReq<TEST_AXI_ADDR_W>;
    type Resp = CompressBlockDecoderResp;

    type SequenceDecReq = sequence_dec::SequenceDecoderReq<TEST_AXI_ADDR_W>;
    type SequenceDecResp = sequence_dec::SequenceDecoderResp;

    type MemReaderReq  = mem_reader::MemReaderReq<TEST_AXI_ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<TEST_AXI_DATA_W, TEST_AXI_ADDR_W>;

    type MemAxiAr = axi::AxiAr<TEST_AXI_ADDR_W, TEST_AXI_ID_W>;
    type MemAxiR = axi::AxiR<TEST_AXI_DATA_W, TEST_AXI_ID_W>;
    type MemAxiAw = axi::AxiAw<TEST_AXI_ADDR_W, TEST_AXI_ID_W>;
    type MemAxiW = axi::AxiW<TEST_AXI_DATA_W, TEST_AXI_DATA_W_DIV8>;
    type MemAxiB = axi::AxiB<TEST_AXI_ID_W>;

    type SeqDpdRamRdReq = ram::ReadReq<TEST_SEQ_DPD_RAM_ADDR_W, TEST_SEQ_DPD_RAM_NUM_PARTITIONS>;
    type SeqDpdRamRdResp = ram::ReadResp<TEST_SEQ_DPD_RAM_DATA_W>;
    type SeqDpdRamWrReq = ram::WriteReq<TEST_SEQ_DPD_RAM_ADDR_W, TEST_SEQ_DPD_RAM_DATA_W, TEST_SEQ_DPD_RAM_NUM_PARTITIONS>;
    type SeqDpdRamWrResp = ram::WriteResp;

    type SeqTmpRamRdReq = ram::ReadReq<TEST_SEQ_TMP_RAM_ADDR_W, TEST_SEQ_TMP_RAM_NUM_PARTITIONS>;
    type SeqTmpRamRdResp = ram::ReadResp<TEST_SEQ_TMP_RAM_DATA_W>;
    type SeqTmpRamWrReq = ram::WriteReq<TEST_SEQ_TMP_RAM_ADDR_W, TEST_SEQ_TMP_RAM_DATA_W, TEST_SEQ_TMP_RAM_NUM_PARTITIONS>;
    type SeqTmpRamWrResp = ram::WriteResp;

    type SeqTmp2RamRdReq = ram::ReadReq<TEST_SEQ_TMP2_RAM_ADDR_W, TEST_SEQ_TMP2_RAM_NUM_PARTITIONS>;
    type SeqTmp2RamRdResp = ram::ReadResp<TEST_SEQ_TMP2_RAM_DATA_W>;
    type SeqTmp2RamWrReq = ram::WriteReq<TEST_SEQ_TMP2_RAM_ADDR_W, TEST_SEQ_TMP2_RAM_DATA_W, TEST_SEQ_TMP2_RAM_NUM_PARTITIONS>;
    type SeqTmp2RamWrResp = ram::WriteResp;

    type SeqFseRamRdReq = ram::ReadReq<TEST_SEQ_FSE_RAM_ADDR_W, TEST_SEQ_FSE_RAM_NUM_PARTITIONS>;
    type SeqFseRamRdResp = ram::ReadResp<TEST_SEQ_FSE_RAM_DATA_W>;
    type SeqFseRamWrReq = ram::WriteReq<TEST_SEQ_FSE_RAM_ADDR_W, TEST_SEQ_FSE_RAM_DATA_W, TEST_SEQ_FSE_RAM_NUM_PARTITIONS>;
    type SeqFseRamWrResp = ram::WriteResp;

    type LiteralsHeaderDecoderResp = literals_block_header_dec::LiteralsHeaderDecoderResp;
    type LiteralsBlockType = literals_block_header_dec::LiteralsBlockType;
    type LiteralsDecReq = literals_decoder::LiteralsDecoderCtrlReq<TEST_AXI_ADDR_W>;
    type LiteralsDecResp = literals_decoder::LiteralsDecoderCtrlResp;
    type LiteralsBufCtrl = common::LiteralsBufferCtrl;

    type SequenceExecutorPacket = common::SequenceExecutorPacket<common::SYMBOL_WIDTH>;
    type CommandConstructorData = common::CommandConstructorData;

    type HuffmanWeightsReadReq    = ram::ReadReq<TEST_HUFFMAN_WEIGHTS_RAM_ADDR_W, TEST_HUFFMAN_WEIGHTS_RAM_NUM_PARTITIONS>;
    type HuffmanWeightsReadResp   = ram::ReadResp<TEST_HUFFMAN_WEIGHTS_RAM_DATA_W>;
    type HuffmanWeightsWriteReq   = ram::WriteReq<TEST_HUFFMAN_WEIGHTS_RAM_ADDR_W, TEST_HUFFMAN_WEIGHTS_RAM_DATA_W, TEST_HUFFMAN_WEIGHTS_RAM_NUM_PARTITIONS>;
    type HuffmanWeightsWriteResp  = ram::WriteResp;

    type HuffmanPrescanReadReq    = ram::ReadReq<TEST_HUFFMAN_PRESCAN_RAM_ADDR_W, TEST_HUFFMAN_PRESCAN_RAM_NUM_PARTITIONS>;
    type HuffmanPrescanReadResp   = ram::ReadResp<TEST_HUFFMAN_PRESCAN_RAM_DATA_W>;
    type HuffmanPrescanWriteReq   = ram::WriteReq<TEST_HUFFMAN_PRESCAN_RAM_ADDR_W, TEST_HUFFMAN_PRESCAN_RAM_DATA_W, TEST_HUFFMAN_PRESCAN_RAM_NUM_PARTITIONS>;
    type HuffmanPrescanWriteResp  = ram::WriteResp;

    type HuffmanWeightsDpdRamRdReq = ram::ReadReq<TEST_HUFFMAN_WEIGHTS_DPD_RAM_ADDR_W, TEST_HUFFMAN_WEIGHTS_DPD_RAM_NUM_PARTITIONS>;
    type HuffmanWeightsDpdRamRdResp = ram::ReadResp<TEST_HUFFMAN_WEIGHTS_DPD_RAM_DATA_W>;
    type HuffmanWeightsDpdRamWrReq = ram::WriteReq<TEST_HUFFMAN_WEIGHTS_DPD_RAM_ADDR_W, TEST_HUFFMAN_WEIGHTS_DPD_RAM_DATA_W, TEST_HUFFMAN_WEIGHTS_DPD_RAM_NUM_PARTITIONS>;
    type HuffmanWeightsDpdRamWrResp = ram::WriteResp;

    type HuffmanWeightsTmpRamRdReq = ram::ReadReq<TEST_HUFFMAN_WEIGHTS_TMP_RAM_ADDR_W, TEST_HUFFMAN_WEIGHTS_TMP_RAM_NUM_PARTITIONS>;
    type HuffmanWeightsTmpRamRdResp = ram::ReadResp<TEST_HUFFMAN_WEIGHTS_TMP_RAM_DATA_W>;
    type HuffmanWeightsTmpRamWrReq = ram::WriteReq<TEST_HUFFMAN_WEIGHTS_TMP_RAM_ADDR_W, TEST_HUFFMAN_WEIGHTS_TMP_RAM_DATA_W, TEST_HUFFMAN_WEIGHTS_TMP_RAM_NUM_PARTITIONS>;
    type HuffmanWeightsTmpRamWrResp = ram::WriteResp;

    type HuffmanWeightsTmp2RamRdReq = ram::ReadReq<TEST_HUFFMAN_WEIGHTS_TMP2_RAM_ADDR_W, TEST_HUFFMAN_WEIGHTS_TMP2_RAM_NUM_PARTITIONS>;
    type HuffmanWeightsTmp2RamRdResp = ram::ReadResp<TEST_HUFFMAN_WEIGHTS_TMP2_RAM_DATA_W>;
    type HuffmanWeightsTmp2RamWrReq = ram::WriteReq<TEST_HUFFMAN_WEIGHTS_TMP2_RAM_ADDR_W, TEST_HUFFMAN_WEIGHTS_TMP2_RAM_DATA_W, TEST_HUFFMAN_WEIGHTS_TMP2_RAM_NUM_PARTITIONS>;
    type HuffmanWeightsTmp2RamWrResp = ram::WriteResp;

    type HuffmanWeightsFseRamRdReq = ram::ReadReq<TEST_HUFFMAN_WEIGHTS_FSE_RAM_ADDR_W, TEST_HUFFMAN_WEIGHTS_FSE_RAM_NUM_PARTITIONS>;
    type HuffmanWeightsFseRamRdResp = ram::ReadResp<TEST_HUFFMAN_WEIGHTS_FSE_RAM_DATA_W>;
    type HuffmanWeightsFseRamWrReq = ram::WriteReq<TEST_HUFFMAN_WEIGHTS_FSE_RAM_ADDR_W, TEST_HUFFMAN_WEIGHTS_FSE_RAM_DATA_W, TEST_HUFFMAN_WEIGHTS_FSE_RAM_NUM_PARTITIONS>;
    type HuffmanWeightsFseRamWrResp = ram::WriteResp;

    type LitBufRamRdReq = ram::ReadReq<TEST_LITERALS_BUFFER_RAM_ADDR_W, TEST_LITERALS_BUFFER_RAM_NUM_PARTITIONS>;
    type LitBufRamRdResp = ram::ReadResp<TEST_LITERALS_BUFFER_RAM_DATA_W>;
    type LitBufRamWrReq = ram::WriteReq<TEST_LITERALS_BUFFER_RAM_ADDR_W, TEST_LITERALS_BUFFER_RAM_DATA_W, TEST_LITERALS_BUFFER_RAM_NUM_PARTITIONS>;
    type LitBufRamWrResp = ram::WriteResp;

    type TestcaseRamRdReq = ram::ReadReq<TEST_CASE_RAM_ADDR_W, TEST_CASE_RAM_NUM_PARTITIONS>;
    type TestcaseRamRdResp = ram::ReadResp<TEST_CASE_RAM_DATA_W>;
    type TestcaseRamWrReq = ram::WriteReq<TEST_CASE_RAM_ADDR_W, TEST_CASE_RAM_DATA_W, TEST_CASE_RAM_NUM_PARTITIONS>;
    type TestcaseRamWrResp = ram::WriteResp;

    terminator: chan<bool> out;
    req_s: chan<Req> out;
    resp_r: chan<Resp> in;
    cmd_constr_out_r: chan<ExtendedPacket> in;
    axi_ram_wr_req_s: chan<TestcaseRamWrReq>[AXI_CHAN_N] out;
    axi_ram_wr_resp_r: chan<TestcaseRamWrResp>[AXI_CHAN_N] in;

    ll_sel_test_s: chan<u1> out;
    ll_def_test_rd_req_s: chan<SeqFseRamRdReq> out;
    ll_def_test_rd_resp_r: chan<SeqFseRamRdResp> in;
    ll_def_test_wr_req_s: chan<SeqFseRamWrReq> out;
    ll_def_test_wr_resp_r: chan<SeqFseRamWrResp> in;

    ml_sel_test_s: chan<u1> out;
    ml_def_test_rd_req_s: chan<SeqFseRamRdReq> out;
    ml_def_test_rd_resp_r: chan<SeqFseRamRdResp> in;
    ml_def_test_wr_req_s: chan<SeqFseRamWrReq> out;
    ml_def_test_wr_resp_r: chan<SeqFseRamWrResp> in;

    of_sel_test_s: chan<u1> out;
    of_def_test_rd_req_s: chan<SeqFseRamRdReq> out;
    of_def_test_rd_resp_r: chan<SeqFseRamRdResp> in;
    of_def_test_wr_req_s: chan<SeqFseRamWrReq> out;
    of_def_test_wr_resp_r: chan<SeqFseRamWrResp> in;

    init {}
    config(terminator: chan<bool> out) {
        let (req_s, req_r) = chan<Req>("req");
        let (resp_s, resp_r) = chan<Resp>("resp");

        // output from Command constructor to Sequence executor
        let (cmd_constr_out_s, cmd_constr_out_r) = chan<ExtendedPacket>("cmd_constr_out");

        // Huffman weights memory
        let (huffman_lit_weights_mem_rd_req_s, huffman_lit_weights_mem_rd_req_r) = chan<HuffmanWeightsReadReq>("huffman_lit_weights_mem_rd_req");
        let (huffman_lit_weights_mem_rd_resp_s, huffman_lit_weights_mem_rd_resp_r) = chan<HuffmanWeightsReadResp>("huffman_lit_weights_mem_rd_resp");
        let (huffman_lit_weights_mem_wr_req_s, huffman_lit_weights_mem_wr_req_r) = chan<HuffmanWeightsWriteReq>("huffman_lit_weights_mem_wr_req");
        let (huffman_lit_weights_mem_wr_resp_s, huffman_lit_weights_mem_wr_resp_r) = chan<HuffmanWeightsWriteResp>("huffman_lit_weights_mem_wr_resp");

        // Huffman prescan memory
        let (huffman_lit_prescan_mem_rd_req_s, huffman_lit_prescan_mem_rd_req_r) = chan<HuffmanPrescanReadReq>("huffman_lit_prescan_mem_rd_req");
        let (huffman_lit_prescan_mem_rd_resp_s, huffman_lit_prescan_mem_rd_resp_r) = chan<HuffmanPrescanReadResp>("huffman_lit_prescan_mem_rd_resp");
        let (huffman_lit_prescan_mem_wr_req_s, huffman_lit_prescan_mem_wr_req_r) = chan<HuffmanPrescanWriteReq>("huffman_lit_prescan_mem_wr_req");
        let (huffman_lit_prescan_mem_wr_resp_s, huffman_lit_prescan_mem_wr_resp_r) = chan<HuffmanPrescanWriteResp>("huffman_lit_prescan_mem_wr_resp");

        let (huffman_lit_weights_dpd_rd_req_s, huffman_lit_weights_dpd_rd_req_r) = chan<HuffmanWeightsDpdRamRdReq>("huffman_lit_weights_dpd_rd_req");
        let (huffman_lit_weights_dpd_rd_resp_s, huffman_lit_weights_dpd_rd_resp_r) = chan<HuffmanWeightsDpdRamRdResp>("huffman_lit_weights_dpd_rd_resp_r");
        let (huffman_lit_weights_dpd_wr_req_s, huffman_lit_weights_dpd_wr_req_r) = chan<HuffmanWeightsDpdRamWrReq>("huffman_lit_weights_dpd_wr_req");
        let (huffman_lit_weights_dpd_wr_resp_s, huffman_lit_weights_dpd_wr_resp_r) = chan<HuffmanWeightsDpdRamWrResp>("huffman_lit_weights_dpd_wr_resp");

        let (huffman_lit_weights_tmp_rd_req_s, huffman_lit_weights_tmp_rd_req_r) = chan<HuffmanWeightsTmpRamRdReq>("huffman_lit_weights_tmp_rd_req");
        let (huffman_lit_weights_tmp_rd_resp_s, huffman_lit_weights_tmp_rd_resp_r) = chan<HuffmanWeightsTmpRamRdResp>("huffman_lit_weights_tmp_rd_resp");
        let (huffman_lit_weights_tmp_wr_req_s, huffman_lit_weights_tmp_wr_req_r) = chan<HuffmanWeightsTmpRamWrReq>("huffman_lit_weights_tmp_wr_req");
        let (huffman_lit_weights_tmp_wr_resp_s, huffman_lit_weights_tmp_wr_resp_r) = chan<HuffmanWeightsTmpRamWrResp>("huffman_lit_weights_tmp_wr_resp");

        let (huffman_lit_weights_tmp2_rd_req_s, huffman_lit_weights_tmp2_rd_req_r) = chan<HuffmanWeightsTmp2RamRdReq>("huffman_lit_weights_tmp2_rd_req");
        let (huffman_lit_weights_tmp2_rd_resp_s, huffman_lit_weights_tmp2_rd_resp_r) = chan<HuffmanWeightsTmp2RamRdResp>("huffman_lit_weights_tmp2_rd_resp");
        let (huffman_lit_weights_tmp2_wr_req_s, huffman_lit_weights_tmp2_wr_req_r) = chan<HuffmanWeightsTmp2RamWrReq>("huffman_lit_weights_tmp2_wr_req");
        let (huffman_lit_weights_tmp2_wr_resp_s, huffman_lit_weights_tmp2_wr_resp_r) = chan<HuffmanWeightsTmp2RamWrResp>("huffman_lit_weights_tmp2_wr_resp");

        let (huffman_lit_weights_fse_rd_req_s, huffman_lit_weights_fse_rd_req_r) = chan<HuffmanWeightsFseRamRdReq>("huffman_lit_weights_fse_rd_req");
        let (huffman_lit_weights_fse_rd_resp_s, huffman_lit_weights_fse_rd_resp_r) = chan<HuffmanWeightsFseRamRdResp>("huffman_lit_weights_fse_rd_resp_r");
        let (huffman_lit_weights_fse_wr_req_s, huffman_lit_weights_fse_wr_req_r) = chan<HuffmanWeightsFseRamWrReq>("huffman_lit_weights_fse_wr_req");
        let (huffman_lit_weights_fse_wr_resp_s, huffman_lit_weights_fse_wr_resp_r) = chan<HuffmanWeightsFseRamWrResp>("huffman_lit_weights_fse_wr_resp");

        spawn ram::RamModel<TEST_HUFFMAN_WEIGHTS_DPD_RAM_DATA_W, TEST_HUFFMAN_WEIGHTS_DPD_RAM_SIZE,
                            TEST_HUFFMAN_WEIGHTS_DPD_RAM_WORD_PARTITION_SIZE,
                            TEST_RAM_SIM_RW_BEHAVIOR, TEST_RAM_INITIALIZED
        >(
            huffman_lit_weights_dpd_rd_req_r, huffman_lit_weights_dpd_rd_resp_s,
            huffman_lit_weights_dpd_wr_req_r, huffman_lit_weights_dpd_wr_resp_s,
        );

        spawn ram::RamModel<TEST_HUFFMAN_WEIGHTS_TMP_RAM_DATA_W, TEST_HUFFMAN_WEIGHTS_TMP_RAM_SIZE,
                            TEST_HUFFMAN_WEIGHTS_TMP_RAM_WORD_PARTITION_SIZE,
                            TEST_RAM_SIM_RW_BEHAVIOR, TEST_RAM_INITIALIZED
        >(
            huffman_lit_weights_tmp_rd_req_r, huffman_lit_weights_tmp_rd_resp_s,
            huffman_lit_weights_tmp_wr_req_r, huffman_lit_weights_tmp_wr_resp_s,
        );

        spawn ram::RamModel<TEST_HUFFMAN_WEIGHTS_TMP2_RAM_DATA_W, TEST_HUFFMAN_WEIGHTS_TMP2_RAM_SIZE,
                            TEST_HUFFMAN_WEIGHTS_TMP2_RAM_WORD_PARTITION_SIZE,
                            TEST_RAM_SIM_RW_BEHAVIOR, TEST_RAM_INITIALIZED
        >(
            huffman_lit_weights_tmp2_rd_req_r, huffman_lit_weights_tmp2_rd_resp_s,
            huffman_lit_weights_tmp2_wr_req_r, huffman_lit_weights_tmp2_wr_resp_s,
        );

        spawn ram::RamModel<TEST_HUFFMAN_WEIGHTS_FSE_RAM_DATA_W, TEST_HUFFMAN_WEIGHTS_FSE_RAM_SIZE,
                            TEST_HUFFMAN_WEIGHTS_FSE_RAM_WORD_PARTITION_SIZE,
                            TEST_RAM_SIM_RW_BEHAVIOR, TEST_RAM_INITIALIZED
        >(
            huffman_lit_weights_fse_rd_req_r, huffman_lit_weights_fse_rd_resp_s,
            huffman_lit_weights_fse_wr_req_r, huffman_lit_weights_fse_wr_resp_s,
        );

        spawn ram::RamModel<TEST_HUFFMAN_PRESCAN_RAM_DATA_W, TEST_HUFFMAN_PRESCAN_RAM_SIZE,
                            TEST_HUFFMAN_PRESCAN_RAM_WORD_PARTITION_SIZE,
                            TEST_RAM_SIM_RW_BEHAVIOR, TEST_RAM_INITIALIZED
        >(
            huffman_lit_prescan_mem_rd_req_r, huffman_lit_prescan_mem_rd_resp_s,
            huffman_lit_prescan_mem_wr_req_r, huffman_lit_prescan_mem_wr_resp_s
        );

        spawn ram::RamModel<TEST_HUFFMAN_WEIGHTS_RAM_DATA_W, TEST_HUFFMAN_WEIGHTS_RAM_SIZE,
                            TEST_HUFFMAN_WEIGHTS_RAM_WORD_PARTITION_SIZE,
                            TEST_RAM_SIM_RW_BEHAVIOR, TEST_RAM_INITIALIZED
        >(
            huffman_lit_weights_mem_rd_req_r, huffman_lit_weights_mem_rd_resp_s,
            huffman_lit_weights_mem_wr_req_r, huffman_lit_weights_mem_wr_resp_s
        );

        // AXI channels for various blocks
        let (axi_ram_rd_req_s, axi_ram_rd_req_r) = chan<TestcaseRamRdReq>[AXI_CHAN_N]("axi_ram_rd_req");
        let (axi_ram_rd_resp_s, axi_ram_rd_resp_r) = chan<TestcaseRamRdResp>[AXI_CHAN_N]("axi_ram_rd_resp");
        let (axi_ram_wr_req_s, axi_ram_wr_req_r) = chan<TestcaseRamWrReq>[AXI_CHAN_N]("axi_ram_wr_req");
        let (axi_ram_wr_resp_s, axi_ram_wr_resp_r) = chan<TestcaseRamWrResp>[AXI_CHAN_N]("axi_ram_wr_resp");
        let (axi_ram_ar_s, axi_ram_ar_r) = chan<MemAxiAr>[AXI_CHAN_N]("axi_ram_ar");
        let (axi_ram_r_s, axi_ram_r_r) = chan<MemAxiR>[AXI_CHAN_N]("axi_ram_r");
        unroll_for! (i, ()): (u32, ()) in range(u32:0, AXI_CHAN_N) {
            spawn ram::RamModel<
                TEST_CASE_RAM_DATA_W, TEST_CASE_RAM_SIZE, TEST_CASE_RAM_WORD_PARTITION_SIZE,
                TEST_RAM_SIM_RW_BEHAVIOR, TEST_RAM_INITIALIZED
            >(
                axi_ram_rd_req_r[i], axi_ram_rd_resp_s[i], axi_ram_wr_req_r[i], axi_ram_wr_resp_s[i]
            );
            spawn axi_ram::AxiRamReader<
                TEST_AXI_ADDR_W, TEST_AXI_DATA_W, TEST_AXI_DEST_W, TEST_AXI_ID_W, TEST_CASE_RAM_SIZE,
                TEST_CASE_RAM_BASE_ADDR, TEST_CASE_RAM_DATA_W, TEST_CASE_RAM_ADDR_W
            >(
                axi_ram_ar_r[i], axi_ram_r_s[i], axi_ram_rd_req_s[i], axi_ram_rd_resp_r[i]
            );
        }(());

        // Literals buffer RAMs
        let (litbuf_rd_req_s,  litbuf_rd_req_r) = chan<LitBufRamRdReq>[u32:8]("litbuf_rd_req");
        let (litbuf_rd_resp_s, litbuf_rd_resp_r) = chan<LitBufRamRdResp>[u32:8]("litbuf_rd_resp");
        let (litbuf_wr_req_s,  litbuf_wr_req_r) = chan<LitBufRamWrReq>[u32:8]("litbuf_wr_req");
        let (litbuf_wr_resp_s, litbuf_wr_resp_r) = chan<LitBufRamWrResp>[u32:8]("litbuf_wr_resp");
        unroll_for! (i, ()): (u32, ()) in range(u32:0, u32:8) {
            spawn ram::RamModel<
                TEST_LITERALS_BUFFER_RAM_DATA_W, TEST_LITERALS_BUFFER_RAM_SIZE, TEST_LITERALS_BUFFER_RAM_WORD_PARTITION_SIZE,
                TEST_RAM_SIM_RW_BEHAVIOR, TEST_RAM_INITIALIZED
            >(
                litbuf_rd_req_r[i], litbuf_rd_resp_s[i], litbuf_wr_req_r[i], litbuf_wr_resp_s[i]
            );
        }(());

        // RAMs for FSE decoder
        // DPD RAM
        let (dpd_rd_req_s, dpd_rd_req_r) = chan<SeqDpdRamRdReq>("dpd_rd_req");
        let (dpd_rd_resp_s, dpd_rd_resp_r) = chan<SeqDpdRamRdResp>("dpd_rd_resp");
        let (dpd_wr_req_s, dpd_wr_req_r) = chan<SeqDpdRamWrReq>("dpd_wr_req");
        let (dpd_wr_resp_s, dpd_wr_resp_r) = chan<SeqDpdRamWrResp>("dpd_wr_resp");
        spawn ram::RamModel<TEST_SEQ_DPD_RAM_DATA_W, TEST_SEQ_DPD_RAM_SIZE, TEST_SEQ_DPD_RAM_WORD_PARTITION_SIZE,
                            TEST_RAM_SIM_RW_BEHAVIOR, TEST_RAM_INITIALIZED
        >(
            dpd_rd_req_r, dpd_rd_resp_s, dpd_wr_req_r, dpd_wr_resp_s,
        );

        // TMP RAM
        let (tmp_rd_req_s, tmp_rd_req_r) = chan<SeqTmpRamRdReq>("tmp_rd_req");
        let (tmp_rd_resp_s, tmp_rd_resp_r) = chan<SeqTmpRamRdResp>("tmp_rd_resp");
        let (tmp_wr_req_s, tmp_wr_req_r) = chan<SeqTmpRamWrReq>("tmp_wr_req");
        let (tmp_wr_resp_s, tmp_wr_resp_r) = chan<SeqTmpRamWrResp>("tmp_wr_resp");
        spawn ram::RamModel<
            TEST_SEQ_TMP_RAM_DATA_W, TEST_SEQ_TMP_RAM_SIZE, TEST_SEQ_TMP_RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIM_RW_BEHAVIOR, TEST_RAM_INITIALIZED
        >(
            tmp_rd_req_r, tmp_rd_resp_s, tmp_wr_req_r, tmp_wr_resp_s,
        );

        let (tmp2_rd_req_s, tmp2_rd_req_r) = chan<SeqTmp2RamRdReq>("tmp2_rd_req");
        let (tmp2_rd_resp_s, tmp2_rd_resp_r) = chan<SeqTmp2RamRdResp>("tmp2_rd_resp");
        let (tmp2_wr_req_s, tmp2_wr_req_r) = chan<SeqTmp2RamWrReq>("tmp2_wr_req");
        let (tmp2_wr_resp_s, tmp2_wr_resp_r) = chan<SeqTmp2RamWrResp>("tmp2_wr_resp");
        spawn ram::RamModel<
            TEST_SEQ_TMP2_RAM_DATA_W, TEST_SEQ_TMP2_RAM_SIZE, TEST_SEQ_TMP2_RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIM_RW_BEHAVIOR, TEST_RAM_INITIALIZED
        >(
            tmp2_rd_req_r, tmp2_rd_resp_s, tmp2_wr_req_r, tmp2_wr_resp_s,
        );

        // FSE RAMs
        let (fse_rd_req_s, fse_rd_req_r) = chan<SeqFseRamRdReq>[u32:6]("tmp_rd_req");
        let (fse_rd_resp_s, fse_rd_resp_r) = chan<SeqFseRamRdResp>[u32:6]("tmp_rd_resp");
        let (fse_wr_req_s, fse_wr_req_r) = chan<SeqFseRamWrReq>[u32:6]("tmp_wr_req");
        let (fse_wr_resp_s, fse_wr_resp_r) = chan<SeqFseRamWrResp>[u32:6]("tmp_wr_resp");
        unroll_for! (i, ()): (u32, ()) in range(u32:0, u32:6) {
            spawn ram::RamModel<
                TEST_SEQ_FSE_RAM_DATA_W, TEST_SEQ_FSE_RAM_SIZE, TEST_SEQ_FSE_RAM_WORD_PARTITION_SIZE,
                TEST_RAM_SIM_RW_BEHAVIOR, TEST_RAM_INITIALIZED
            >(
                fse_rd_req_r[i], fse_rd_resp_s[i], fse_wr_req_r[i], fse_wr_resp_s[i]
            );
        }(());

        // Default LL

        let (ll_sel_test_s, ll_sel_test_r) = chan<u1>("ll_sel_test");

        let (ll_def_test_rd_req_s, ll_def_test_rd_req_r) = chan<SeqFseRamRdReq>("ll_def_test_rd_req");
        let (ll_def_test_rd_resp_s, ll_def_test_rd_resp_r) = chan<SeqFseRamRdResp>("ll_def_test_rd_resp");
        let (ll_def_test_wr_req_s, ll_def_test_wr_req_r) = chan<SeqFseRamWrReq>("ll_def_test_wr_req");
        let (ll_def_test_wr_resp_s, ll_def_test_wr_resp_r) = chan<SeqFseRamWrResp>("ll_def_test_wr_resp");

        let (ll_def_fse_rd_req_s, ll_def_fse_rd_req_r) = chan<SeqFseRamRdReq>("ll_def_fse_rd_req");
        let (ll_def_fse_rd_resp_s, ll_def_fse_rd_resp_r) = chan<SeqFseRamRdResp>("ll_def_fse_rd_resp");
        let (ll_def_fse_wr_req_s, ll_def_fse_wr_req_r) = chan<SeqFseRamWrReq>("ll_def_fse_wr_req");
        let (ll_def_fse_wr_resp_s, ll_def_fse_wr_resp_r) = chan<SeqFseRamWrResp>("ll_def_fse_wr_resp");

        spawn ram_mux::RamMux<
            TEST_SEQ_FSE_RAM_ADDR_W, TEST_SEQ_FSE_RAM_DATA_W, TEST_SEQ_FSE_RAM_NUM_PARTITIONS,
        >(
            ll_sel_test_r,
            ll_def_test_rd_req_r, ll_def_test_rd_resp_s, ll_def_test_wr_req_r, ll_def_test_wr_resp_s,
            ll_def_fse_rd_req_r, ll_def_fse_rd_resp_s, ll_def_fse_wr_req_r, ll_def_fse_wr_resp_s,
            fse_rd_req_s[0], fse_rd_resp_r[0], fse_wr_req_s[0], fse_wr_resp_r[0],
        );

        // Default ML

        let (ml_sel_test_s, ml_sel_test_r) = chan<u1>("ml_sel_test");

        let (ml_def_test_rd_req_s, ml_def_test_rd_req_r) = chan<SeqFseRamRdReq>("ml_def_test_rd_req");
        let (ml_def_test_rd_resp_s, ml_def_test_rd_resp_r) = chan<SeqFseRamRdResp>("ml_def_test_rd_resp");
        let (ml_def_test_wr_req_s, ml_def_test_wr_req_r) = chan<SeqFseRamWrReq>("ml_def_test_wr_req");
        let (ml_def_test_wr_resp_s, ml_def_test_wr_resp_r) = chan<SeqFseRamWrResp>("ml_def_test_wr_resp");

        let (ml_def_fse_rd_req_s, ml_def_fse_rd_req_r) = chan<SeqFseRamRdReq>("ml_def_fse_rd_req");
        let (ml_def_fse_rd_resp_s, ml_def_fse_rd_resp_r) = chan<SeqFseRamRdResp>("ml_def_fse_rd_resp");
        let (ml_def_fse_wr_req_s, ml_def_fse_wr_req_r) = chan<SeqFseRamWrReq>("ml_def_fse_wr_req");
        let (ml_def_fse_wr_resp_s, ml_def_fse_wr_resp_r) = chan<SeqFseRamWrResp>("ml_def_fse_wr_resp");

        spawn ram_mux::RamMux<
            TEST_SEQ_FSE_RAM_ADDR_W, TEST_SEQ_FSE_RAM_DATA_W, TEST_SEQ_FSE_RAM_NUM_PARTITIONS,
        >(
            ml_sel_test_r,
            ml_def_test_rd_req_r, ml_def_test_rd_resp_s, ml_def_test_wr_req_r, ml_def_test_wr_resp_s,
            ml_def_fse_rd_req_r, ml_def_fse_rd_resp_s, ml_def_fse_wr_req_r, ml_def_fse_wr_resp_s,
            fse_rd_req_s[2], fse_rd_resp_r[2], fse_wr_req_s[2], fse_wr_resp_r[2],
        );

        // Default OF

        let (of_sel_test_s, of_sel_test_r) = chan<u1>("of_sel_test");

        let (of_def_test_rd_req_s, of_def_test_rd_req_r) = chan<SeqFseRamRdReq>("of_def_test_rd_req");
        let (of_def_test_rd_resp_s, of_def_test_rd_resp_r) = chan<SeqFseRamRdResp>("of_def_test_rd_resp");
        let (of_def_test_wr_req_s, of_def_test_wr_req_r) = chan<SeqFseRamWrReq>("of_def_test_wr_req");
        let (of_def_test_wr_resp_s, of_def_test_wr_resp_r) = chan<SeqFseRamWrResp>("of_def_test_wr_resp");

        let (of_def_fse_rd_req_s, of_def_fse_rd_req_r) = chan<SeqFseRamRdReq>("of_def_fse_rd_req");
        let (of_def_fse_rd_resp_s, of_def_fse_rd_resp_r) = chan<SeqFseRamRdResp>("of_def_fse_rd_resp");
        let (of_def_fse_wr_req_s, of_def_fse_wr_req_r) = chan<SeqFseRamWrReq>("of_def_fse_wr_req");
        let (of_def_fse_wr_resp_s, of_def_fse_wr_resp_r) = chan<SeqFseRamWrResp>("of_def_fse_wr_resp");

        spawn ram_mux::RamMux<
            TEST_SEQ_FSE_RAM_ADDR_W, TEST_SEQ_FSE_RAM_DATA_W, TEST_SEQ_FSE_RAM_NUM_PARTITIONS,
        >(
            of_sel_test_r,
            of_def_test_rd_req_r, of_def_test_rd_resp_s, of_def_test_wr_req_r, of_def_test_wr_resp_s,
            of_def_fse_rd_req_r, of_def_fse_rd_resp_s, of_def_fse_wr_req_r, of_def_fse_wr_resp_s,
            fse_rd_req_s[4], fse_rd_resp_r[4], fse_wr_req_s[4], fse_wr_resp_r[4],
        );

        spawn CompressBlockDecoder<
            TEST_AXI_DATA_W, TEST_AXI_ADDR_W, TEST_AXI_ID_W, TEST_AXI_DEST_W,

            TEST_SEQ_DPD_RAM_ADDR_W, TEST_SEQ_DPD_RAM_DATA_W, TEST_SEQ_DPD_RAM_NUM_PARTITIONS,
            TEST_SEQ_TMP_RAM_ADDR_W, TEST_SEQ_TMP_RAM_DATA_W, TEST_SEQ_TMP_RAM_NUM_PARTITIONS,
            TEST_SEQ_TMP2_RAM_ADDR_W, TEST_SEQ_TMP2_RAM_DATA_W, TEST_SEQ_TMP2_RAM_NUM_PARTITIONS,
            TEST_SEQ_FSE_RAM_ADDR_W, TEST_SEQ_FSE_RAM_DATA_W, TEST_SEQ_FSE_RAM_NUM_PARTITIONS,

            TEST_HUFFMAN_WEIGHTS_DPD_RAM_ADDR_W, TEST_HUFFMAN_WEIGHTS_DPD_RAM_DATA_W, TEST_HUFFMAN_WEIGHTS_DPD_RAM_NUM_PARTITIONS,
            TEST_HUFFMAN_WEIGHTS_TMP_RAM_ADDR_W, TEST_HUFFMAN_WEIGHTS_TMP_RAM_DATA_W, TEST_HUFFMAN_WEIGHTS_TMP_RAM_NUM_PARTITIONS,
            TEST_HUFFMAN_WEIGHTS_TMP2_RAM_ADDR_W, TEST_HUFFMAN_WEIGHTS_TMP2_RAM_DATA_W, TEST_HUFFMAN_WEIGHTS_TMP2_RAM_NUM_PARTITIONS,
            TEST_HUFFMAN_WEIGHTS_FSE_RAM_ADDR_W, TEST_HUFFMAN_WEIGHTS_FSE_RAM_DATA_W, TEST_HUFFMAN_WEIGHTS_FSE_RAM_NUM_PARTITIONS,
        >(
            req_r, resp_s,
            cmd_constr_out_s,
            axi_ram_ar_s[0], axi_ram_r_r[0],
            axi_ram_ar_s[1], axi_ram_r_r[1],
            axi_ram_ar_s[2], axi_ram_r_r[2],
            dpd_rd_req_s, dpd_rd_resp_r, dpd_wr_req_s, dpd_wr_resp_r,
            tmp_rd_req_s, tmp_rd_resp_r, tmp_wr_req_s, tmp_wr_resp_r,
            tmp2_rd_req_s, tmp2_rd_resp_r, tmp2_wr_req_s, tmp2_wr_resp_r,
            ll_def_fse_rd_req_s, ll_def_fse_rd_resp_r, ll_def_fse_wr_req_s, ll_def_fse_wr_resp_r,
            fse_rd_req_s[1], fse_rd_resp_r[1], fse_wr_req_s[1], fse_wr_resp_r[1],
            ml_def_fse_rd_req_s, ml_def_fse_rd_resp_r, ml_def_fse_wr_req_s, ml_def_fse_wr_resp_r,
            fse_rd_req_s[3], fse_rd_resp_r[3], fse_wr_req_s[3], fse_wr_resp_r[3],
            of_def_fse_rd_req_s, of_def_fse_rd_resp_r, of_def_fse_wr_req_s, of_def_fse_wr_resp_r,
            fse_rd_req_s[5], fse_rd_resp_r[5], fse_wr_req_s[5], fse_wr_resp_r[5],
            axi_ram_ar_s[3], axi_ram_r_r[3],
            axi_ram_ar_s[4], axi_ram_r_r[4],
            axi_ram_ar_s[5], axi_ram_r_r[5],
            axi_ram_ar_s[6], axi_ram_r_r[6],
            axi_ram_ar_s[7], axi_ram_r_r[7],
            axi_ram_ar_s[8], axi_ram_r_r[8],
            axi_ram_ar_s[9], axi_ram_r_r[9],
            axi_ram_ar_s[10], axi_ram_r_r[10],
            litbuf_rd_req_s[0], litbuf_rd_req_s[1], litbuf_rd_req_s[2], litbuf_rd_req_s[3],
            litbuf_rd_req_s[4], litbuf_rd_req_s[5], litbuf_rd_req_s[6], litbuf_rd_req_s[7],
            litbuf_rd_resp_r[0], litbuf_rd_resp_r[1], litbuf_rd_resp_r[2], litbuf_rd_resp_r[3],
            litbuf_rd_resp_r[4], litbuf_rd_resp_r[5], litbuf_rd_resp_r[6], litbuf_rd_resp_r[7],
            litbuf_wr_req_s[0], litbuf_wr_req_s[1], litbuf_wr_req_s[2], litbuf_wr_req_s[3],
            litbuf_wr_req_s[4], litbuf_wr_req_s[5], litbuf_wr_req_s[6], litbuf_wr_req_s[7],
            litbuf_wr_resp_r[0], litbuf_wr_resp_r[1], litbuf_wr_resp_r[2], litbuf_wr_resp_r[3],
            litbuf_wr_resp_r[4], litbuf_wr_resp_r[5], litbuf_wr_resp_r[6], litbuf_wr_resp_r[7],
            huffman_lit_weights_mem_rd_req_s, huffman_lit_weights_mem_rd_resp_r,
            huffman_lit_weights_mem_wr_req_s, huffman_lit_weights_mem_wr_resp_r,
            huffman_lit_prescan_mem_rd_req_s, huffman_lit_prescan_mem_rd_resp_r,
            huffman_lit_prescan_mem_wr_req_s, huffman_lit_prescan_mem_wr_resp_r,
            huffman_lit_weights_dpd_rd_req_s, huffman_lit_weights_dpd_rd_resp_r,
            huffman_lit_weights_dpd_wr_req_s, huffman_lit_weights_dpd_wr_resp_r,
            huffman_lit_weights_tmp_rd_req_s, huffman_lit_weights_tmp_rd_resp_r,
            huffman_lit_weights_tmp_wr_req_s, huffman_lit_weights_tmp_wr_resp_r,
            huffman_lit_weights_tmp2_rd_req_s, huffman_lit_weights_tmp2_rd_resp_r,
            huffman_lit_weights_tmp2_wr_req_s, huffman_lit_weights_tmp2_wr_resp_r,
            huffman_lit_weights_fse_rd_req_s, huffman_lit_weights_fse_rd_resp_r,
            huffman_lit_weights_fse_wr_req_s, huffman_lit_weights_fse_wr_resp_r,
        );

        (
            terminator,
            req_s, resp_r,
            cmd_constr_out_r,
            axi_ram_wr_req_s, axi_ram_wr_resp_r,

            ll_sel_test_s,
            ll_def_test_rd_req_s, ll_def_test_rd_resp_r, ll_def_test_wr_req_s, ll_def_test_wr_resp_r,

            ml_sel_test_s,
            ml_def_test_rd_req_s, ml_def_test_rd_resp_r, ml_def_test_wr_req_s, ml_def_test_wr_resp_r,

            of_sel_test_s,
            of_def_test_rd_req_s, of_def_test_rd_resp_r, of_def_test_wr_req_s, of_def_test_wr_resp_r,
        )
    }

    next(state: ()) {
        let tok = join();

        // FILL THE LL DEFAULT RAM
        trace_fmt!("Filling LL default FSE table");
        let tok = send(tok, ll_sel_test_s, u1:0);
        let tok = unroll_for! (i, tok): (u32, token) in range(u32:0, array_size(sequence_dec::DEFAULT_LL_TABLE)) {
            let req = SeqFseRamWrReq {
                addr: i as uN[TEST_SEQ_FSE_RAM_ADDR_W],
                data: fse_table_creator::fse_record_to_bits(sequence_dec::DEFAULT_LL_TABLE[i]),
                mask: !uN[TEST_SEQ_FSE_RAM_NUM_PARTITIONS]:0,
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
            let req = SeqFseRamWrReq {
                addr: i as uN[TEST_SEQ_FSE_RAM_ADDR_W],
                data: fse_table_creator::fse_record_to_bits(sequence_dec::DEFAULT_OF_TABLE[i]),
                mask: !uN[TEST_SEQ_FSE_RAM_NUM_PARTITIONS]:0,
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
            let req = SeqFseRamWrReq {
                addr: i as uN[TEST_SEQ_FSE_RAM_ADDR_W],
                data: fse_table_creator::fse_record_to_bits(sequence_dec::DEFAULT_ML_TABLE[i]),
                mask: !uN[TEST_SEQ_FSE_RAM_NUM_PARTITIONS]:0,
            };
            let tok = send(tok, ml_def_test_wr_req_s, req);
            let (tok, _) = recv(tok, ml_def_test_wr_resp_r);
            tok
        }(tok);
        let tok = send(tok, ml_sel_test_s, u1:1);

        let tok = unroll_for!(test_i, tok): (u32, token) in range(u32:0, array_size(COMP_BLOCK_DEC_TESTCASES)) {
            let (input_length, input, output_length, output) = COMP_BLOCK_DEC_TESTCASES[test_i];

            trace_fmt!("Loading testcase {}", test_i);
            let tok = for ((i, input_data), tok): ((u32, u64), token) in enumerate(input) {
                let req = TestcaseRamWrReq {
                    addr: i as uN[TEST_CASE_RAM_ADDR_W],
                    data: input_data as uN[TEST_CASE_RAM_DATA_W],
                    mask: !uN[TEST_CASE_RAM_NUM_PARTITIONS]:0
                };
                // Write to all RAMs
                let tok = unroll_for! (j, tok): (u32, token) in range(u32:0, AXI_CHAN_N) {
                    let tok = send(tok, axi_ram_wr_req_s[j], req);
                    let (tok, _) = recv(tok, axi_ram_wr_resp_r[j]);
                    tok
                }(tok);
                tok
            }(tok);

            trace_fmt!("Starting processing testcase {}", test_i);

            let req = Req {
                addr: uN[TEST_AXI_ADDR_W]:0x100,
                length: input_length as BlockSize,
                id: test_i,
                last_block: false,
            };

            trace_fmt!("Sending request to compressed block decoder: {}", req);
            let tok = send(tok, req_s, req);

            let tok = for (i, tok): (u32, token) in range(u32:0, output_length) {
                let expected_packet = output[i];
                let (tok, recvd_packet) = recv(tok, cmd_constr_out_r);
                trace_fmt!("Received {} command constructor packet: {:#x}", i, recvd_packet);
                assert_eq(expected_packet, recvd_packet);
                tok
            }(tok);

            let (tok, _) = recv(tok, resp_r);
            trace_fmt!("Finished processing testcase {}", test_i);
            tok
        }(tok);

        send(tok, terminator, true);
    }
}

