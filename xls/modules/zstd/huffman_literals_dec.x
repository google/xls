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

// This file contains Huffman literals decoder proc implementation.

import std;
import xls.modules.zstd.common as common;
import xls.modules.zstd.huffman_common as hcommon;
import xls.modules.zstd.huffman_axi_reader as axi_reader;
import xls.modules.zstd.huffman_code_builder as code_builder;
import xls.modules.zstd.huffman_data_preprocessor as data_preprocessor;
import xls.modules.zstd.huffman_decoder as decoder;
import xls.modules.zstd.huffman_prescan as prescan;
import xls.modules.zstd.huffman_ctrl as ctrl;
import xls.modules.zstd.huffman_weights_dec as weights_dec;
import xls.modules.zstd.memory.axi as axi;
import xls.modules.zstd.memory.axi_ram;
import xls.modules.zstd.memory.mem_reader as mem_reader;
import xls.examples.ram;

pub fn WeightPreScanMetaDataSize() -> u32 {
    prescan::WeightPreScanMetaDataSize()
}

pub type HuffmanLiteralsDecoderReq = ctrl::HuffmanControlAndSequenceCtrl;
pub type HuffmanLiteralsDecoderResp = ctrl::HuffmanControlAndSequenceResp;
pub type HuffmanLiteralsDecoderStatus = ctrl::HuffmanControlAndSequenceStatus;

pub const RAM_SIZE = prescan::RAM_SIZE;
pub const WEIGHTS_ADDR_WIDTH = prescan::RAM_ADDR_WIDTH;
pub const WEIGHTS_DATA_WIDTH = prescan::RAM_ACCESS_WIDTH;
pub const WEIGHTS_PARTITION_WORD_SIZE = WEIGHTS_DATA_WIDTH / u32:8;
pub const WEIGHTS_NUM_PARTITIONS = ram::num_partitions(WEIGHTS_PARTITION_WORD_SIZE, WEIGHTS_DATA_WIDTH);
// pub const WEIGHTS_NUM_PARTITIONS: u32 = u32:1;

pub const PRESCAN_ADDR_WIDTH: u32 = prescan::RAM_ADDR_WIDTH;
pub const PRESCAN_DATA_WIDTH: u32 = prescan::WeightPreScanMetaDataSize();
pub const PRESCAN_PARTITION_WORD_SIZE: u32 = PRESCAN_DATA_WIDTH;
pub const PRESCAN_NUM_PARTITIONS = ram::num_partitions(PRESCAN_PARTITION_WORD_SIZE, PRESCAN_DATA_WIDTH);

// pub const PRESCAN_NUM_PARTITIONS: u32 = u32:1;

pub proc HuffmanLiteralsDecoder<
    AXI_DATA_W: u32, AXI_ADDR_W: u32, AXI_ID_W: u32, AXI_DEST_W: u32,
    WEIGHTS_DPD_RAM_ADDR_W: u32, WEIGHTS_DPD_RAM_DATA_W: u32, WEIGHTS_DPD_RAM_NUM_PARTITIONS: u32,
    WEIGHTS_TMP_RAM_ADDR_W: u32, WEIGHTS_TMP_RAM_DATA_W: u32, WEIGHTS_TMP_RAM_NUM_PARTITIONS: u32,
    WEIGHTS_TMP2_RAM_ADDR_W: u32, WEIGHTS_TMP2_RAM_DATA_W: u32, WEIGHTS_TMP2_RAM_NUM_PARTITIONS: u32,

    WEIGHTS_FSE_RAM_ADDR_W: u32, WEIGHTS_FSE_RAM_DATA_W: u32, WEIGHTS_FSE_RAM_NUM_PARTITIONS: u32,
    WEIGHTS_RAM_ADDR_WIDTH: u32 = {WEIGHTS_ADDR_WIDTH},
    WEIGHTS_RAM_DATA_WIDTH: u32 = {WEIGHTS_DATA_WIDTH},
    WEIGHTS_RAM_NUM_PARTITIONS: u32 = {WEIGHTS_NUM_PARTITIONS},
    PRESCAN_RAM_ADDR_WIDTH: u32 = {PRESCAN_ADDR_WIDTH},
    PRESCAN_RAM_DATA_WIDTH: u32 = {PRESCAN_DATA_WIDTH},
    PRESCAN_RAM_NUM_PARTITIONS: u32 = {PRESCAN_NUM_PARTITIONS},
    > {
    type AxiR = axi::AxiR<AXI_DATA_W, AXI_ID_W>;
    type AxiAr = axi::AxiAr<AXI_ADDR_W, AXI_ID_W>;

    type MemReaderReq  = mem_reader::MemReaderReq<AXI_ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<AXI_DATA_W, AXI_ADDR_W>;

    type WeightsRamRdReq  = ram::ReadReq<WEIGHTS_RAM_ADDR_WIDTH, WEIGHTS_RAM_NUM_PARTITIONS>;
    type WeightsRamRdResp = ram::ReadResp<WEIGHTS_RAM_DATA_WIDTH>;
    type WeightsRamWrReq = ram::WriteReq<WEIGHTS_RAM_ADDR_WIDTH, WEIGHTS_RAM_DATA_WIDTH, WEIGHTS_RAM_NUM_PARTITIONS>;
    type WeightsRamWrResp = ram::WriteResp;

    type PrescanRamRdReq = ram::ReadReq<PRESCAN_RAM_ADDR_WIDTH, PRESCAN_RAM_NUM_PARTITIONS>;
    type PrescanRamRdResp = ram::ReadResp<PRESCAN_RAM_DATA_WIDTH>;
    type PrescanRamWrReq = ram::WriteReq<PRESCAN_RAM_ADDR_WIDTH, PRESCAN_RAM_DATA_WIDTH, PRESCAN_RAM_NUM_PARTITIONS>;
    type PrescanRamWrResp = ram::WriteResp;

    // Weights FSE RAMs
    type WeightsDpdRamRdReq = ram::ReadReq<WEIGHTS_DPD_RAM_ADDR_W, WEIGHTS_DPD_RAM_NUM_PARTITIONS>;
    type WeightsDpdRamRdResp = ram::ReadResp<WEIGHTS_DPD_RAM_DATA_W>;
    type WeightsDpdRamWrReq = ram::WriteReq<WEIGHTS_DPD_RAM_ADDR_W, WEIGHTS_DPD_RAM_DATA_W, WEIGHTS_DPD_RAM_NUM_PARTITIONS>;
    type WeightsDpdRamWrResp = ram::WriteResp;

    type WeightsTmpRamRdReq = ram::ReadReq<WEIGHTS_TMP_RAM_ADDR_W, WEIGHTS_TMP_RAM_NUM_PARTITIONS>;
    type WeightsTmpRamRdResp = ram::ReadResp<WEIGHTS_TMP_RAM_DATA_W>;
    type WeightsTmpRamWrReq = ram::WriteReq<WEIGHTS_TMP_RAM_ADDR_W, WEIGHTS_TMP_RAM_DATA_W, WEIGHTS_TMP_RAM_NUM_PARTITIONS>;
    type WeightsTmpRamWrResp = ram::WriteResp;

    type WeightsTmp2RamRdReq = ram::ReadReq<WEIGHTS_TMP2_RAM_ADDR_W, WEIGHTS_TMP2_RAM_NUM_PARTITIONS>;
    type WeightsTmp2RamRdResp = ram::ReadResp<WEIGHTS_TMP2_RAM_DATA_W>;
    type WeightsTmp2RamWrReq = ram::WriteReq<WEIGHTS_TMP2_RAM_ADDR_W, WEIGHTS_TMP2_RAM_DATA_W, WEIGHTS_TMP2_RAM_NUM_PARTITIONS>;
    type WeightsTmp2RamWrResp = ram::WriteResp;

    type WeightsFseRamRdReq = ram::ReadReq<WEIGHTS_FSE_RAM_ADDR_W, WEIGHTS_FSE_RAM_NUM_PARTITIONS>;
    type WeightsFseRamRdResp = ram::ReadResp<WEIGHTS_FSE_RAM_DATA_W>;
    type WeightsFseRamWrReq = ram::WriteReq<WEIGHTS_FSE_RAM_ADDR_W, WEIGHTS_FSE_RAM_DATA_W, WEIGHTS_FSE_RAM_NUM_PARTITIONS>;
    type WeightsFseRamWrResp = ram::WriteResp;

    type WeightsDecReq = weights_dec::HuffmanWeightsDecoderReq<AXI_ADDR_W>;
    type WeightsDecResp = weights_dec::HuffmanWeightsDecoderResp<AXI_ADDR_W>;

    type HuffmanAxiReaderCtrl = axi_reader::HuffmanAxiReaderCtrl<AXI_ADDR_W>;

    type Ctrl = HuffmanLiteralsDecoderReq<AXI_ADDR_W>;
    type Resp = HuffmanLiteralsDecoderResp;

    config (
        // ctrl
        ctrl_r: chan<Ctrl> in,
        resp_s: chan<Resp> out,
        // output literals
        decoded_literals_s: chan<common::LiteralsDataWithSync> out,
        // AXI interface - reverse reader
        axi_ar_s: chan<AxiAr> out,
        axi_r_r: chan<AxiR> in,
        // AXI interface - Huffman Jump Table decoder
        jump_table_axi_ar_s: chan<AxiAr> out,
        jump_table_axi_r_r: chan<AxiR> in,
        // AXI interface - Huffman tree description header decoder
        weights_header_dec_axi_ar_s: chan<AxiAr> out,
        weights_header_dec_axi_r_r: chan<AxiR> in,
        // AXI interface - RAW Huffman tree description decoder
        weights_raw_dec_axi_ar_s: chan<AxiAr> out,
        weights_raw_dec_axi_r_r: chan<AxiR> in,
        // AXI interface - FSE Huffman tree description decoder
        weights_fse_lookup_dec_axi_ar_s: chan<AxiAr> out,
        weights_fse_lookup_dec_axi_r_r: chan<AxiR> in,
        weights_fse_decoder_dec_axi_ar_s: chan<AxiAr> out,
        weights_fse_decoder_dec_axi_r_r: chan<AxiR> in,
        // weight memory
        weights_ram_rd_req_s: chan<WeightsRamRdReq> out,
        weights_ram_rd_resp_r: chan<WeightsRamRdResp> in,
        weights_ram_wr_req_s: chan<WeightsRamWrReq> out,
        weights_ram_wr_resp_r: chan<WeightsRamWrResp> in,
        // prescan memory
        prescan_ram_rd_req_s: chan<PrescanRamRdReq> out,
        prescan_ram_rd_resp_r: chan<PrescanRamRdResp> in,
        prescan_ram_wr_req_s: chan<PrescanRamWrReq> out,
        prescan_ram_wr_resp_r: chan<PrescanRamWrResp> in,
        // Weights FSE RAMs
        weights_dpd_rd_req_s: chan<WeightsDpdRamRdReq> out,
        weights_dpd_rd_resp_r: chan<WeightsDpdRamRdResp> in,
        weights_dpd_wr_req_s: chan<WeightsDpdRamWrReq> out,
        weights_dpd_wr_resp_r: chan<WeightsDpdRamWrResp> in,

        weights_tmp_rd_req_s: chan<WeightsTmpRamRdReq> out,
        weights_tmp_rd_resp_r: chan<WeightsTmpRamRdResp> in,
        weights_tmp_wr_req_s: chan<WeightsTmpRamWrReq> out,
        weights_tmp_wr_resp_r: chan<WeightsTmpRamWrResp> in,

        weights_tmp2_rd_req_s: chan<WeightsTmp2RamRdReq> out,
        weights_tmp2_rd_resp_r: chan<WeightsTmp2RamRdResp> in,
        weights_tmp2_wr_req_s: chan<WeightsTmp2RamWrReq> out,
        weights_tmp2_wr_resp_r: chan<WeightsTmp2RamWrResp> in,

        weights_fse_rd_req_s: chan<WeightsFseRamRdReq> out,
        weights_fse_rd_resp_r: chan<WeightsFseRamRdResp> in,
        weights_fse_wr_req_s: chan<WeightsFseRamWrReq> out,
        weights_fse_wr_resp_r: chan<WeightsFseRamWrResp> in,
    ) {
        let (prescan_start_s, prescan_start_r) = chan<bool, u32:1>("prescan_start");
        let (code_builder_start_s, code_builder_start_r) = chan<bool, u32:1>("code_buider");
        let (axi_reader_ctrl_s, axi_reader_ctrl_r) = chan<HuffmanAxiReaderCtrl, u32:1>("axi_reader_ctrl");
        let (data_preprocess_start_s, data_preprocess_start_r) = chan<data_preprocessor::HuffmanDataPreprocessorStart, u32:1>("data_preprocess_start");
        let (decoder_start_s, decoder_start_r) = chan<decoder::HuffmanDecoderStart, u32:1>("decoder_start");
        let (decoder_done_s, decoder_done_r) = chan<(), u32:1>("decoder_done");
        let (prescan_response_s, prescan_response_r) = chan<hcommon::WeightPreScanOutput, u32:1>("prescan_response");
        let (code_builder_codes_s, code_builder_codes_r) = chan<hcommon::CodeBuilderToDecoderOutput, u32:1>("code_builder_codes");
        let (lookahead_config_s, lookahead_config_r) = chan<hcommon::CodeBuilderToPreDecoderOutput, u32:1>("lookahead_config");
        let (axi_data_s, axi_data_r) = chan<axi_reader::HuffmanAxiReaderData, u32:1>("axi_data");
        let (preprocessed_data_s, preprocessed_data_r) = chan<data_preprocessor::HuffmanDataPreprocessorData, u32:1>("preprocessed_data");
        let (weights_dec_req_s, weights_dec_req_r) = chan<WeightsDecReq, u32:1>("weights_dec_req");
        let (weights_dec_resp_s, weights_dec_resp_r) = chan<WeightsDecResp, u32:1>("weights_dec_resp");
        let (jump_table_mem_rd_req_s, jump_table_mem_rd_req_r) = chan<MemReaderReq, u32:1>("jump_table_req");
        let (jump_table_mem_rd_resp_s, jump_table_mem_rd_resp_r) = chan<MemReaderResp, u32:1>("jump_table_resp");
        let (weights_header_dec_mem_rd_req_s, weights_header_dec_mem_rd_req_r) = chan<MemReaderReq, u32:1>("weights_dec_mem_rd_req");
        let (weights_header_dec_mem_rd_resp_s, weights_header_dec_mem_rd_resp_r) = chan<MemReaderResp, u32:1>("weights_dec_mem_rd_resp");
        let (weights_raw_dec_mem_rd_req_s, weights_raw_dec_mem_rd_req_r) = chan<MemReaderReq, u32:1>("weights_dec_mem_rd_req");
        let (weights_raw_dec_mem_rd_resp_s, weights_raw_dec_mem_rd_resp_r) = chan<MemReaderResp, u32:1>("weights_dec_mem_rd_resp");
        let (weights_fse_lookup_dec_mem_rd_req_s, weights_fse_lookup_dec_mem_rd_req_r) = chan<MemReaderReq, u32:1>("weights_lookup_dec_mem_rd_req");
        let (weights_fse_lookup_dec_mem_rd_resp_s, weights_fse_lookup_dec_mem_rd_resp_r) = chan<MemReaderResp, u32:1>("weights_lookup_dec_mem_rd_resp");
        let (weights_fse_decoder_dec_mem_rd_req_s, weights_fse_decoder_dec_mem_rd_req_r) = chan<MemReaderReq, u32:1>("weights_decoder_dec_mem_rd_req");
        let (weights_fse_decoder_dec_mem_rd_resp_s, weights_fse_decoder_dec_mem_rd_resp_r) = chan<MemReaderResp, u32:1>("weights_decoder_dec_mem_rd_resp");

        // code builder loopback
        let (weights_pow_sum_loopback_s, weights_pow_sum_loopback_r) = chan<uN[hcommon::MAX_WEIGHT + u32:2], u32:1>("weights_pow_sum_loopback");

        spawn ctrl::HuffmanControlAndSequence<AXI_ADDR_W, AXI_DATA_W>(
            ctrl_r, resp_s,
            weights_dec_req_s, weights_dec_resp_r,
            prescan_start_s,
            code_builder_start_s,
            axi_reader_ctrl_s,
            data_preprocess_start_s,
            decoder_start_s,
            decoder_done_r,
            jump_table_mem_rd_req_s,
            jump_table_mem_rd_resp_r,
        );

        spawn mem_reader::MemReader<AXI_DATA_W, AXI_ADDR_W, AXI_DEST_W, AXI_ID_W>(
           jump_table_mem_rd_req_r, jump_table_mem_rd_resp_s,
           jump_table_axi_ar_s, jump_table_axi_r_r
        );

        spawn mem_reader::MemReader<AXI_DATA_W, AXI_ADDR_W, AXI_DEST_W, AXI_ID_W>(
           weights_header_dec_mem_rd_req_r, weights_header_dec_mem_rd_resp_s,
           weights_header_dec_axi_ar_s, weights_header_dec_axi_r_r
        );

        spawn mem_reader::MemReader<AXI_DATA_W, AXI_ADDR_W, AXI_DEST_W, AXI_ID_W>(
           weights_raw_dec_mem_rd_req_r, weights_raw_dec_mem_rd_resp_s,
           weights_raw_dec_axi_ar_s, weights_raw_dec_axi_r_r
        );

        spawn mem_reader::MemReader<AXI_DATA_W, AXI_ADDR_W, AXI_DEST_W, AXI_ID_W>(
           weights_fse_lookup_dec_mem_rd_req_r, weights_fse_lookup_dec_mem_rd_resp_s,
           weights_fse_lookup_dec_axi_ar_s, weights_fse_lookup_dec_axi_r_r
        );

        spawn mem_reader::MemReader<AXI_DATA_W, AXI_ADDR_W, AXI_DEST_W, AXI_ID_W>(
           weights_fse_decoder_dec_mem_rd_req_r, weights_fse_decoder_dec_mem_rd_resp_s,
           weights_fse_decoder_dec_axi_ar_s, weights_fse_decoder_dec_axi_r_r
        );

        spawn weights_dec::HuffmanWeightsDecoder<
            AXI_ADDR_W, AXI_DATA_W, AXI_ID_W,
            WEIGHTS_RAM_ADDR_WIDTH, WEIGHTS_RAM_DATA_WIDTH, WEIGHTS_RAM_NUM_PARTITIONS,
            WEIGHTS_DPD_RAM_ADDR_W, WEIGHTS_DPD_RAM_DATA_W, WEIGHTS_DPD_RAM_NUM_PARTITIONS,
            WEIGHTS_TMP_RAM_ADDR_W, WEIGHTS_TMP_RAM_DATA_W, WEIGHTS_TMP_RAM_NUM_PARTITIONS,
            WEIGHTS_TMP2_RAM_ADDR_W, WEIGHTS_TMP2_RAM_DATA_W, WEIGHTS_TMP2_RAM_NUM_PARTITIONS,
            WEIGHTS_FSE_RAM_ADDR_W, WEIGHTS_FSE_RAM_DATA_W, WEIGHTS_FSE_RAM_NUM_PARTITIONS,
        >(
            weights_dec_req_r, weights_dec_resp_s,
            weights_header_dec_mem_rd_req_s, weights_header_dec_mem_rd_resp_r,
            weights_raw_dec_mem_rd_req_s, weights_raw_dec_mem_rd_resp_r,
            weights_fse_lookup_dec_mem_rd_req_s, weights_fse_lookup_dec_mem_rd_resp_r,
            weights_fse_decoder_dec_mem_rd_req_s, weights_fse_decoder_dec_mem_rd_resp_r,
            weights_ram_wr_req_s, weights_ram_wr_resp_r,
            weights_dpd_rd_req_s, weights_dpd_rd_resp_r, weights_dpd_wr_req_s, weights_dpd_wr_resp_r,
            weights_tmp_rd_req_s, weights_tmp_rd_resp_r, weights_tmp_wr_req_s, weights_tmp_wr_resp_r,
            weights_tmp2_rd_req_s, weights_tmp2_rd_resp_r, weights_tmp2_wr_req_s, weights_tmp2_wr_resp_r,
            weights_fse_rd_req_s, weights_fse_rd_resp_r, weights_fse_wr_req_s, weights_fse_wr_resp_r,
        );

        spawn prescan::WeightPreScan(
            prescan_start_r,
            weights_ram_rd_req_s,
            weights_ram_rd_resp_r,
            prescan_response_s,
            prescan_ram_rd_req_s,
            prescan_ram_rd_resp_r,
            prescan_ram_wr_req_s,
            prescan_ram_wr_resp_r,
        );

        spawn code_builder::WeightCodeBuilder(
            code_builder_start_r,
            prescan_response_r,
            code_builder_codes_s,
            lookahead_config_s,
            weights_pow_sum_loopback_s,
            weights_pow_sum_loopback_r,
        );

        spawn axi_reader::HuffmanAxiReader<AXI_DATA_W, AXI_ADDR_W, AXI_ID_W, AXI_DEST_W>(
            axi_reader_ctrl_r,
            axi_r_r,
            axi_ar_s,
            axi_data_s,
        );

        spawn data_preprocessor::HuffmanDataPreprocessor(
            data_preprocess_start_r,
            lookahead_config_r,
            axi_data_r,
            preprocessed_data_s,
        );

        spawn decoder::HuffmanDecoder(
            decoder_start_r,
            code_builder_codes_r,
            preprocessed_data_r,
            decoder_done_s,
            decoded_literals_s,
        );

        ()
    }

    init { }

    next (state: ()) { }
}

const INST_AXI_DATA_W = u32:64;
const INST_AXI_ADDR_W = u32:16;
const INST_AXI_ID_W = u32:4;
const INST_AXI_DEST_W = u32:4;

pub const INST_WEIGHTS_RAM_ADDR_WIDTH = WEIGHTS_ADDR_WIDTH;
pub const INST_WEIGHTS_RAM_DATA_WIDTH = WEIGHTS_DATA_WIDTH;
pub const INST_WEIGHTS_RAM_NUM_PARTITIONS = WEIGHTS_NUM_PARTITIONS;
pub const INST_PRESCAN_RAM_ADDR_WIDTH = PRESCAN_ADDR_WIDTH;
pub const INST_PRESCAN_RAM_DATA_WIDTH = PRESCAN_DATA_WIDTH;
pub const INST_PRESCAN_RAM_NUM_PARTITIONS = PRESCAN_NUM_PARTITIONS;

const INST_WEIGHTS_DPD_RAM_DATA_W = u32:16;
const INST_WEIGHTS_DPD_RAM_SIZE = u32:256;
const INST_WEIGHTS_DPD_RAM_ADDR_W = std::clog2(INST_WEIGHTS_DPD_RAM_SIZE);
const INST_WEIGHTS_DPD_RAM_WORD_PARTITION_SIZE = INST_WEIGHTS_DPD_RAM_DATA_W;
const INST_WEIGHTS_DPD_RAM_NUM_PARTITIONS = ram::num_partitions(
    INST_WEIGHTS_DPD_RAM_WORD_PARTITION_SIZE, INST_WEIGHTS_DPD_RAM_DATA_W
);

const INST_WEIGHTS_FSE_RAM_DATA_W = u32:32;
const INST_WEIGHTS_FSE_RAM_SIZE = u32:256;
const INST_WEIGHTS_FSE_RAM_ADDR_W = std::clog2(INST_WEIGHTS_FSE_RAM_SIZE);
const INST_WEIGHTS_FSE_RAM_WORD_PARTITION_SIZE = INST_WEIGHTS_FSE_RAM_DATA_W / u32:3;
const INST_WEIGHTS_FSE_RAM_NUM_PARTITIONS = ram::num_partitions(
    INST_WEIGHTS_FSE_RAM_WORD_PARTITION_SIZE, INST_WEIGHTS_FSE_RAM_DATA_W
);

const INST_WEIGHTS_TMP_RAM_DATA_W = u32:16;
const INST_WEIGHTS_TMP_RAM_SIZE = u32:256;
const INST_WEIGHTS_TMP_RAM_ADDR_W = std::clog2(INST_WEIGHTS_TMP_RAM_SIZE);
const INST_WEIGHTS_TMP_RAM_WORD_PARTITION_SIZE = INST_WEIGHTS_TMP_RAM_DATA_W;
const INST_WEIGHTS_TMP_RAM_NUM_PARTITIONS = ram::num_partitions(
    INST_WEIGHTS_TMP_RAM_WORD_PARTITION_SIZE, INST_WEIGHTS_TMP_RAM_DATA_W
);

const INST_WEIGHTS_TMP2_RAM_DATA_W = u32:8;
const INST_WEIGHTS_TMP2_RAM_SIZE = u32:512;
const INST_WEIGHTS_TMP2_RAM_ADDR_W = std::clog2(INST_WEIGHTS_TMP2_RAM_SIZE);
const INST_WEIGHTS_TMP2_RAM_WORD_PARTITION_SIZE = INST_WEIGHTS_TMP2_RAM_DATA_W;
const INST_WEIGHTS_TMP2_RAM_NUM_PARTITIONS = ram::num_partitions(
    INST_WEIGHTS_TMP2_RAM_WORD_PARTITION_SIZE, INST_WEIGHTS_TMP2_RAM_DATA_W
);

proc HuffmanLiteralsDecoderInst {
    type Ctrl = HuffmanLiteralsDecoderReq<INST_AXI_ADDR_W>;
    type Resp = HuffmanLiteralsDecoderResp;
    type AxiR = axi::AxiR<INST_AXI_DATA_W, INST_AXI_ID_W>;
    type AxiAr = axi::AxiAr<INST_AXI_ADDR_W, INST_AXI_ID_W>;

    type WeightsRamRdReq  = ram::ReadReq<INST_WEIGHTS_RAM_ADDR_WIDTH, INST_WEIGHTS_RAM_NUM_PARTITIONS>;
    type WeightsRamRdResp = ram::ReadResp<INST_WEIGHTS_RAM_DATA_WIDTH>;
    type WeightsRamWrReq = ram::WriteReq<INST_WEIGHTS_RAM_ADDR_WIDTH, INST_WEIGHTS_RAM_DATA_WIDTH, INST_WEIGHTS_RAM_NUM_PARTITIONS>;
    type WeightsRamWrResp = ram::WriteResp;

    type PrescanRamRdReq = ram::ReadReq<INST_PRESCAN_RAM_ADDR_WIDTH, INST_PRESCAN_RAM_NUM_PARTITIONS>;
    type PrescanRamRdResp = ram::ReadResp<INST_PRESCAN_RAM_DATA_WIDTH>;
    type PrescanRamWrReq = ram::WriteReq<INST_PRESCAN_RAM_ADDR_WIDTH, INST_PRESCAN_RAM_DATA_WIDTH, INST_PRESCAN_RAM_NUM_PARTITIONS>;
    type PrescanRamWrResp = ram::WriteResp;

    type WeightsDpdRamRdReq = ram::ReadReq<INST_WEIGHTS_DPD_RAM_ADDR_W, INST_WEIGHTS_DPD_RAM_NUM_PARTITIONS>;
    type WeightsDpdRamRdResp = ram::ReadResp<INST_WEIGHTS_DPD_RAM_DATA_W>;
    type WeightsDpdRamWrReq = ram::WriteReq<INST_WEIGHTS_DPD_RAM_ADDR_W, INST_WEIGHTS_DPD_RAM_DATA_W, INST_WEIGHTS_DPD_RAM_NUM_PARTITIONS>;
    type WeightsDpdRamWrResp = ram::WriteResp;

    type WeightsTmpRamRdReq = ram::ReadReq<INST_WEIGHTS_TMP_RAM_ADDR_W, INST_WEIGHTS_TMP_RAM_NUM_PARTITIONS>;
    type WeightsTmpRamRdResp = ram::ReadResp<INST_WEIGHTS_TMP_RAM_DATA_W>;
    type WeightsTmpRamWrReq = ram::WriteReq<INST_WEIGHTS_TMP_RAM_ADDR_W, INST_WEIGHTS_TMP_RAM_DATA_W, INST_WEIGHTS_TMP_RAM_NUM_PARTITIONS>;
    type WeightsTmpRamWrResp = ram::WriteResp;

    type WeightsTmp2RamRdReq = ram::ReadReq<INST_WEIGHTS_TMP2_RAM_ADDR_W, INST_WEIGHTS_TMP2_RAM_NUM_PARTITIONS>;
    type WeightsTmp2RamRdResp = ram::ReadResp<INST_WEIGHTS_TMP2_RAM_DATA_W>;
    type WeightsTmp2RamWrReq = ram::WriteReq<INST_WEIGHTS_TMP2_RAM_ADDR_W, INST_WEIGHTS_TMP2_RAM_DATA_W, INST_WEIGHTS_TMP2_RAM_NUM_PARTITIONS>;
    type WeightsTmp2RamWrResp = ram::WriteResp;

    type WeightsFseRamRdReq = ram::ReadReq<INST_WEIGHTS_FSE_RAM_ADDR_W, INST_WEIGHTS_FSE_RAM_NUM_PARTITIONS>;
    type WeightsFseRamRdResp = ram::ReadResp<INST_WEIGHTS_FSE_RAM_DATA_W>;
    type WeightsFseRamWrReq = ram::WriteReq<INST_WEIGHTS_FSE_RAM_ADDR_W, INST_WEIGHTS_FSE_RAM_DATA_W, INST_WEIGHTS_FSE_RAM_NUM_PARTITIONS>;
    type WeightsFseRamWrResp = ram::WriteResp;

    config (
        ctrl_r: chan<Ctrl> in,
        resp_s: chan<Resp> out,
        decoded_literals_s: chan<common::LiteralsDataWithSync> out,
        axi_ar_s: chan<AxiAr> out,
        axi_r_r: chan<AxiR> in,

        jump_table_axi_ar_s: chan<AxiAr> out,
        jump_table_axi_r_r: chan<AxiR> in,

        weights_header_dec_axi_ar_s: chan<AxiAr> out,
        weights_header_dec_axi_r_r: chan<AxiR> in,

        weights_raw_dec_axi_ar_s: chan<AxiAr> out,
        weights_raw_dec_axi_r_r: chan<AxiR> in,

        weights_fse_lookup_dec_axi_ar_s: chan<AxiAr> out,
        weights_fse_lookup_dec_axi_r_r: chan<AxiR> in,
        weights_fse_decoder_dec_axi_ar_s: chan<AxiAr> out,
        weights_fse_decoder_dec_axi_r_r: chan<AxiR> in,

        weights_ram_rd_req_s: chan<WeightsRamRdReq> out,
        weights_ram_rd_resp_r: chan<WeightsRamRdResp> in,
        weights_ram_wr_req_s: chan<WeightsRamWrReq> out,
        weights_ram_wr_resp_r: chan<WeightsRamWrResp> in,

        prescan_ram_rd_req_s: chan<PrescanRamRdReq> out,
        prescan_ram_rd_resp_r: chan<PrescanRamRdResp> in,
        prescan_ram_wr_req_s: chan<PrescanRamWrReq> out,
        prescan_ram_wr_resp_r: chan<PrescanRamWrResp> in,

        weights_dpd_rd_req_s: chan<WeightsDpdRamRdReq> out,
        weights_dpd_rd_resp_r: chan<WeightsDpdRamRdResp> in,
        weights_dpd_wr_req_s: chan<WeightsDpdRamWrReq> out,
        weights_dpd_wr_resp_r: chan<WeightsDpdRamWrResp> in,

        weights_tmp_rd_req_s: chan<WeightsTmpRamRdReq> out,
        weights_tmp_rd_resp_r: chan<WeightsTmpRamRdResp> in,
        weights_tmp_wr_req_s: chan<WeightsTmpRamWrReq> out,
        weights_tmp_wr_resp_r: chan<WeightsTmpRamWrResp> in,

        weights_tmp2_rd_req_s: chan<WeightsTmp2RamRdReq> out,
        weights_tmp2_rd_resp_r: chan<WeightsTmp2RamRdResp> in,
        weights_tmp2_wr_req_s: chan<WeightsTmp2RamWrReq> out,
        weights_tmp2_wr_resp_r: chan<WeightsTmp2RamWrResp> in,

        weights_fse_rd_req_s: chan<WeightsFseRamRdReq> out,
        weights_fse_rd_resp_r: chan<WeightsFseRamRdResp> in,
        weights_fse_wr_req_s: chan<WeightsFseRamWrReq> out,
        weights_fse_wr_resp_r: chan<WeightsFseRamWrResp> in,
    ) {
        spawn HuffmanLiteralsDecoder<
            INST_AXI_DATA_W, INST_AXI_ADDR_W, INST_AXI_ID_W, INST_AXI_DEST_W,
            INST_WEIGHTS_DPD_RAM_ADDR_W, INST_WEIGHTS_DPD_RAM_DATA_W, INST_WEIGHTS_DPD_RAM_NUM_PARTITIONS,
            INST_WEIGHTS_TMP_RAM_ADDR_W, INST_WEIGHTS_TMP_RAM_DATA_W, INST_WEIGHTS_TMP_RAM_NUM_PARTITIONS,
            INST_WEIGHTS_TMP2_RAM_ADDR_W, INST_WEIGHTS_TMP2_RAM_DATA_W, INST_WEIGHTS_TMP2_RAM_NUM_PARTITIONS,

            INST_WEIGHTS_FSE_RAM_ADDR_W, INST_WEIGHTS_FSE_RAM_DATA_W, INST_WEIGHTS_FSE_RAM_NUM_PARTITIONS,
            INST_WEIGHTS_RAM_ADDR_WIDTH, INST_WEIGHTS_RAM_DATA_WIDTH, INST_WEIGHTS_RAM_NUM_PARTITIONS,
            INST_PRESCAN_RAM_ADDR_WIDTH, INST_PRESCAN_RAM_DATA_WIDTH, INST_PRESCAN_RAM_NUM_PARTITIONS
        >(
            ctrl_r, resp_s,
            decoded_literals_s,
            axi_ar_s, axi_r_r,
            jump_table_axi_ar_s, jump_table_axi_r_r,
            weights_header_dec_axi_ar_s, weights_header_dec_axi_r_r,
            weights_raw_dec_axi_ar_s, weights_raw_dec_axi_r_r,
            weights_fse_lookup_dec_axi_ar_s, weights_fse_lookup_dec_axi_r_r,
            weights_fse_decoder_dec_axi_ar_s, weights_fse_decoder_dec_axi_r_r,
            weights_ram_rd_req_s, weights_ram_rd_resp_r,
            weights_ram_wr_req_s, weights_ram_wr_resp_r,
            prescan_ram_rd_req_s, prescan_ram_rd_resp_r,
            prescan_ram_wr_req_s, prescan_ram_wr_resp_r,
            weights_dpd_rd_req_s, weights_dpd_rd_resp_r,
            weights_dpd_wr_req_s, weights_dpd_wr_resp_r,
            weights_tmp_rd_req_s, weights_tmp_rd_resp_r,
            weights_tmp_wr_req_s, weights_tmp_wr_resp_r,
            weights_tmp2_rd_req_s, weights_tmp2_rd_resp_r,
            weights_tmp2_wr_req_s, weights_tmp2_wr_resp_r,
            weights_fse_rd_req_s, weights_fse_rd_resp_r,
            weights_fse_wr_req_s, weights_fse_wr_resp_r,
        );
    }

    init { }

    next (state: ()) { }
}

const TEST_AXI_RAM_DATA_W = u32:64;
const TEST_AXI_RAM_ADDR_W = u32:32;
const TEST_AXI_RAM_ID_W = u32:32;
const TEST_AXI_RAM_DEST_W = u32:32;
const TEST_AXI_RAM_DATA_DIV8 = TEST_AXI_RAM_DATA_W / u32:8;
const TEST_AXI_RAM_DATA_DIV8_W = std::clog2(TEST_AXI_RAM_DATA_DIV8);

// Parameters for RamModels used for mocking the system memory for
// the LiteralsBlockHeaderDecoder, RawLiteralsDecoder and HuffmanLiteralsDecoder
const TEST_AXI_RAM_MODEL_DATA_WIDTH:u32 = TEST_AXI_RAM_DATA_W;
const TEST_AXI_RAM_MODEL_SIZE:u32 = u32:2048;
const TEST_AXI_RAM_MODEL_ADDR_WIDTH:u32 = std::clog2(TEST_AXI_RAM_MODEL_SIZE);
const TEST_AXI_RAM_MODEL_WORD_PARTITION_SIZE:u32 = u32:8;
const TEST_AXI_RAM_MODEL_NUM_PARTITIONS:u32 = ram::num_partitions(TEST_AXI_RAM_MODEL_WORD_PARTITION_SIZE, TEST_AXI_RAM_MODEL_DATA_WIDTH);
const TEST_AXI_RAM_MODEL_BASE_ADDR:u32 = u32:0;
const TEST_AXI_RAM_MODEL_SIMULTANEOUS_READ_WRITE_BEHAVIOR = ram::SimultaneousReadWriteBehavior::READ_BEFORE_WRITE;
const TEST_AXI_RAM_MODEL_INITIALIZED = true;
const TEST_AXI_RAM_MODEL_ASSERT_VALID_READ = true;
const TEST_AXI_RAM_MODEL_NUM = u32:1;

pub const TEST_WEIGHTS_RAM_SIZE = prescan::RAM_SIZE;
pub const TEST_WEIGHTS_RAM_ADDR_WIDTH = WEIGHTS_ADDR_WIDTH;
pub const TEST_WEIGHTS_RAM_DATA_WIDTH = WEIGHTS_DATA_WIDTH;

pub const TEST_WEIGHTS_RAM_NUM_PARTITIONS = WEIGHTS_NUM_PARTITIONS;
pub const TEST_WEIGHTS_WORD_PARTITION_SIZE = WEIGHTS_PARTITION_WORD_SIZE;
pub const TEST_PRESCAN_RAM_ADDR_WIDTH = PRESCAN_ADDR_WIDTH;
pub const TEST_PRESCAN_RAM_DATA_WIDTH = PRESCAN_DATA_WIDTH;
pub const TEST_PRESCAN_RAM_NUM_PARTITIONS = PRESCAN_NUM_PARTITIONS;
pub const TEST_PRESCAN_RAM_SIZE = prescan::RAM_SIZE;
pub const TEST_PRESCAN_WORD_PARTITION_SIZE = prescan::WeightPreScanMetaDataSize();

const TEST_WEIGHTS_DPD_RAM_DATA_W = u32:16;
const TEST_WEIGHTS_DPD_RAM_SIZE = u32:256;
const TEST_WEIGHTS_DPD_RAM_ADDR_W = std::clog2(TEST_WEIGHTS_DPD_RAM_SIZE);
const TEST_WEIGHTS_DPD_RAM_WORD_PARTITION_SIZE = TEST_WEIGHTS_DPD_RAM_DATA_W;
const TEST_WEIGHTS_DPD_RAM_NUM_PARTITIONS = ram::num_partitions(
    TEST_WEIGHTS_DPD_RAM_WORD_PARTITION_SIZE, TEST_WEIGHTS_DPD_RAM_DATA_W
);

const TEST_WEIGHTS_FSE_RAM_DATA_W = u32:32;
const TEST_WEIGHTS_FSE_RAM_SIZE = u32:256;
const TEST_WEIGHTS_FSE_RAM_ADDR_W = std::clog2(TEST_WEIGHTS_FSE_RAM_SIZE);
const TEST_WEIGHTS_FSE_RAM_WORD_PARTITION_SIZE = TEST_WEIGHTS_FSE_RAM_DATA_W / u32:3;
const TEST_WEIGHTS_FSE_RAM_NUM_PARTITIONS = ram::num_partitions(
    TEST_WEIGHTS_FSE_RAM_WORD_PARTITION_SIZE, TEST_WEIGHTS_FSE_RAM_DATA_W
);

const TEST_WEIGHTS_TMP_RAM_DATA_W = u32:16;
const TEST_WEIGHTS_TMP_RAM_SIZE = u32:256;
const TEST_WEIGHTS_TMP_RAM_ADDR_W = std::clog2(TEST_WEIGHTS_TMP_RAM_SIZE);
const TEST_WEIGHTS_TMP_RAM_WORD_PARTITION_SIZE = TEST_WEIGHTS_TMP_RAM_DATA_W;
const TEST_WEIGHTS_TMP_RAM_NUM_PARTITIONS = ram::num_partitions(
    TEST_WEIGHTS_TMP_RAM_WORD_PARTITION_SIZE, TEST_WEIGHTS_TMP_RAM_DATA_W
);

const TEST_WEIGHTS_TMP2_RAM_DATA_W = u32:8;
const TEST_WEIGHTS_TMP2_RAM_SIZE = u32:512;
const TEST_WEIGHTS_TMP2_RAM_ADDR_W = std::clog2(TEST_WEIGHTS_TMP2_RAM_SIZE);
const TEST_WEIGHTS_TMP2_RAM_WORD_PARTITION_SIZE = TEST_WEIGHTS_TMP2_RAM_DATA_W;
const TEST_WEIGHTS_TMP2_RAM_NUM_PARTITIONS = ram::num_partitions(
    TEST_WEIGHTS_TMP2_RAM_WORD_PARTITION_SIZE, TEST_WEIGHTS_TMP2_RAM_DATA_W
);

type TestCtrl = HuffmanLiteralsDecoderReq<TEST_AXI_RAM_ADDR_W>;
type TestResp = HuffmanLiteralsDecoderResp;
type TestAxiR = axi::AxiR<TEST_AXI_RAM_DATA_W, TEST_AXI_RAM_ID_W>;
type TestAxiAr = axi::AxiAr<TEST_AXI_RAM_ADDR_W, TEST_AXI_RAM_ID_W>;

// System bus external memory
type TestAxiRamRdReq = ram::ReadReq<TEST_AXI_RAM_MODEL_ADDR_WIDTH, TEST_AXI_RAM_MODEL_NUM_PARTITIONS>;
type TestAxiRamRdResp = ram::ReadResp<TEST_AXI_RAM_MODEL_DATA_WIDTH>;
type TestAxiRamWrReq = ram::WriteReq<TEST_AXI_RAM_MODEL_ADDR_WIDTH, TEST_AXI_RAM_MODEL_DATA_WIDTH, TEST_AXI_RAM_MODEL_NUM_PARTITIONS>;
type TestAxiRamWrResp = ram::WriteResp;

type TestWeightsRamRdReq  = ram::ReadReq<TEST_WEIGHTS_RAM_ADDR_WIDTH, TEST_WEIGHTS_RAM_NUM_PARTITIONS>;
type TestWeightsRamRdResp = ram::ReadResp<TEST_WEIGHTS_RAM_DATA_WIDTH>;
type TestWeightsRamWrReq = ram::WriteReq<TEST_WEIGHTS_RAM_ADDR_WIDTH, TEST_WEIGHTS_RAM_DATA_WIDTH, TEST_WEIGHTS_RAM_NUM_PARTITIONS>;
type TestWeightsRamWrResp = ram::WriteResp;
type TestPrescanRamRdReq = ram::ReadReq<TEST_PRESCAN_RAM_ADDR_WIDTH, TEST_PRESCAN_RAM_NUM_PARTITIONS>;
type TestPrescanRamRdResp = ram::ReadResp<TEST_PRESCAN_RAM_DATA_WIDTH>;
type TestPrescanRamWrReq = ram::WriteReq<TEST_PRESCAN_RAM_ADDR_WIDTH, TEST_PRESCAN_RAM_DATA_WIDTH, TEST_PRESCAN_RAM_NUM_PARTITIONS>;
type TestPrescanRamWrResp = ram::WriteResp;

type TestRamEntry = uN[TEST_WEIGHTS_RAM_DATA_WIDTH];

type TestAxiRamData = uN[TEST_AXI_RAM_MODEL_DATA_WIDTH];
type TestAxiRamAddr = uN[TEST_AXI_RAM_MODEL_ADDR_WIDTH];
type TestAxiRamMask = uN[TEST_AXI_RAM_MODEL_NUM_PARTITIONS];

type TestWeightsDpdRamRdReq = ram::ReadReq<TEST_WEIGHTS_DPD_RAM_ADDR_W, TEST_WEIGHTS_DPD_RAM_NUM_PARTITIONS>;
type TestWeightsDpdRamRdResp = ram::ReadResp<TEST_WEIGHTS_DPD_RAM_DATA_W>;
type TestWeightsDpdRamWrReq = ram::WriteReq<TEST_WEIGHTS_DPD_RAM_ADDR_W, TEST_WEIGHTS_DPD_RAM_DATA_W, TEST_WEIGHTS_DPD_RAM_NUM_PARTITIONS>;
type TestWeightsDpdRamWrResp = ram::WriteResp;

type TestWeightsTmpRamRdReq = ram::ReadReq<TEST_WEIGHTS_TMP_RAM_ADDR_W, TEST_WEIGHTS_TMP_RAM_NUM_PARTITIONS>;
type TestWeightsTmpRamRdResp = ram::ReadResp<TEST_WEIGHTS_TMP_RAM_DATA_W>;
type TestWeightsTmpRamWrReq = ram::WriteReq<TEST_WEIGHTS_TMP_RAM_ADDR_W, TEST_WEIGHTS_TMP_RAM_DATA_W, TEST_WEIGHTS_TMP_RAM_NUM_PARTITIONS>;
type TestWeightsTmpRamWrResp = ram::WriteResp;

type TestWeightsTmp2RamRdReq = ram::ReadReq<TEST_WEIGHTS_TMP2_RAM_ADDR_W, TEST_WEIGHTS_TMP2_RAM_NUM_PARTITIONS>;
type TestWeightsTmp2RamRdResp = ram::ReadResp<TEST_WEIGHTS_TMP2_RAM_DATA_W>;
type TestWeightsTmp2RamWrReq = ram::WriteReq<TEST_WEIGHTS_TMP2_RAM_ADDR_W, TEST_WEIGHTS_TMP2_RAM_DATA_W, TEST_WEIGHTS_TMP2_RAM_NUM_PARTITIONS>;
type TestWeightsTmp2RamWrResp = ram::WriteResp;

type TestWeightsFseRamRdReq = ram::ReadReq<TEST_WEIGHTS_FSE_RAM_ADDR_W, TEST_WEIGHTS_FSE_RAM_NUM_PARTITIONS>;
type TestWeightsFseRamRdResp = ram::ReadResp<TEST_WEIGHTS_FSE_RAM_DATA_W>;
type TestWeightsFseRamWrReq = ram::WriteReq<TEST_WEIGHTS_FSE_RAM_ADDR_W, TEST_WEIGHTS_FSE_RAM_DATA_W, TEST_WEIGHTS_FSE_RAM_NUM_PARTITIONS>;
type TestWeightsFseRamWrResp = ram::WriteResp;

// Data for test case
// Source: Example from RFC 8878, 4.2.2. Huffman-Coded Streams
// https://datatracker.ietf.org/doc/html/rfc8878#huffman_coded_streams
// Weights taken from Table 25
// Bitstream fixed to encode literal sequence "0145"
// See https://www.rfc-editor.org/errata/eid8195

const TEST_MEMORY: TestAxiRamWrReq[7] = [
    // Literals #0
    // Length: 6 bytes
    // New config, 1 Stream
    // HTD Header: 0x84 (Direct representation, HTD length: 3)
    // Huffman Tree Description
    // code         symbol  length  weight
    // N/A          0x03    0       0
    // 0b0000       0x04    4       1
    // 0b0001       0x05    4       1      // last weight implicit
    // 0b001        0x02    3       2
    // 0b01         0x01    2       3
    // 0b1          0x00    1       4
    // 0b00001      padding

    TestAxiRamWrReq { addr: TestAxiRamAddr:0x0, data: (u16:0b00001_1_01_0000_0001 ++ u24:0x100234 ++ u8:0x84) as TestAxiRamData, mask: TestAxiRamMask:0xFF },
    //                           AXI addr: 0x0                                  ^               ^          ^
    //                                                       Huffman-coded stream             HTD HTD Header

    // Literals #1
    // Length: 2 bytes
    // Old config, 1 Stream
    TestAxiRamWrReq { addr: TestAxiRamAddr:0x20, data: TestAxiRamData:0b00001_0001_0000_01_1, mask: TestAxiRamMask:0xFF },
    //                           AXI addr: 0x100                                           ^
    //                                                                  Huffman-coded stream


    // Literals #2
    // Length: 18 bytes
    // New config, 4 Streams
    // HTD Header: 0x84 (Direct representation, HTD length: 3 + HTD_header (1 byte))
    // Jump Table: 0x0002_0002_0002 (Stream1: 2 bytes; Stream2: 2 bytes; Stream3: 2 bytes)
    // Huffman Tree Description
    // code         symbol  length  weight
    // N/A          0x03    0       0
    // 0b0000       0x04    4       1
    // 0b0001       0x05    4       1      // last weight implicit
    // 0b001        0x02    3       2
    // 0b01         0x01    2       3
    // 0b1          0x00    1       4
    // 0b00001      padding
    TestAxiRamWrReq { addr: TestAxiRamAddr:0x40, data: (u32:0x0002_0002 ++ u24:0x100234 ++ u8:0x84) as TestAxiRamData, mask: TestAxiRamMask:0xFF },
    //                           AXI addr: 0x200                      ^               ^          ^
    //                                                       Jump table             HTD HTD Header
    TestAxiRamWrReq { addr: TestAxiRamAddr:0x41, data: (u16:0b00001_1_01_0000_0001 ++ u16:0b00001_1_01_0000_0001 ++ u16:0b00001_1_01_0000_0001 ++ u16:0x0002) as TestAxiRamData, mask: TestAxiRamMask:0xFF },
    //                           AXI addr: 0x208                                 ^                             ^                             ^             ^
    //                                                      Huffman-coded stream #3       Huffman-coded stream #2       Huffman-coded stream #1         Jump table continued
    TestAxiRamWrReq { addr: TestAxiRamAddr:0x42, data: TestAxiRamData:0b00001_1_01_0000_0001, mask: TestAxiRamMask:0xFF },
    //                           AXI addr: 0x210                                           ^
    //                                                                Huffman-coded stream #4

    // Literals #3
    // Length: 14 bytes
    // Old config, 4 Streams
    // Jump Table: 0x0002_0002_0002 (Stream1: 2 bytes; Stream2: 2 bytes; Stream3: 2 bytes)
    TestAxiRamWrReq { addr: TestAxiRamAddr:0x60, data: (u16:0b00001_1_01_0000_0001 ++ u48:0x0002_0002_0002) as TestAxiRamData, mask: TestAxiRamMask:0xFF },
    //                           AXI addr: 0x300                                 ^                       ^
    //                                                      Huffman-coded stream #1                  Jump table
    TestAxiRamWrReq { addr: TestAxiRamAddr:0x61, data: (u16:0b00001_1_01_0000_0001 ++ u16:0b00001_1_01_0000_0001 ++ u16:0b00001_1_01_0000_0001) as TestAxiRamData, mask: TestAxiRamMask:0xFF },
    //                           AXI addr: 0x308                                 ^                             ^                             ^
    //                                                      Huffman-coded stream #4       Huffman-coded stream #3       Huffman-coded stream #2
];

const TEST_CTRL: TestCtrl[4] = [
    // Literals #0
    TestCtrl {
        base_addr: uN[TEST_AXI_RAM_ADDR_W]:0x0,
        len: uN[TEST_AXI_RAM_ADDR_W]:0x6,
        new_config: true,
        multi_stream: false,
        id: u32:0,
        literals_last: false,
    },

    // Literals #1
    TestCtrl {
        base_addr: uN[TEST_AXI_RAM_ADDR_W]:0x100,
        len: uN[TEST_AXI_RAM_ADDR_W]:0x2,
        new_config: false,
        multi_stream: false,
        id: u32:1,
        literals_last: false,
    },

    // Literals #2
    TestCtrl {
        base_addr: uN[TEST_AXI_RAM_ADDR_W]:0x200,
        len: uN[TEST_AXI_RAM_ADDR_W]:0x12,
        new_config: true,
        multi_stream: true,
        id: u32:2,
        literals_last: false,
    },

    // Literals #3
    TestCtrl {
        base_addr: uN[TEST_AXI_RAM_ADDR_W]:0x300,
        len: uN[TEST_AXI_RAM_ADDR_W]:0xE,
        new_config: false,
        multi_stream: true,
        id: u32:3,
        literals_last: true,
    },
];

const TEST_DECODED_LITERALS = common::LiteralsDataWithSync[10]:[
    // Literals #0
    common::LiteralsDataWithSync {
        data: common::LitData:0x0504_0100,
        length: common::LitLength:4,
        last: true,
        id: u32:0,
        literals_last: false,
    },
    // Literals #1
    common::LiteralsDataWithSync {
        data: common::LitData:0x0001_0405,
        length: common::LitLength:4,
        last: true,
        id: u32:1,
        literals_last: false,
    },
    // Literals #2
    common::LiteralsDataWithSync {
        data: common::LitData:0x0504_0100,
        length: common::LitLength:4,
        last: false,
        id: u32:2,
        literals_last: false,
    },
    common::LiteralsDataWithSync {
        data: common::LitData:0x0504_0100,
        length: common::LitLength:4,
        last: false,
        id: u32:2,
        literals_last: false,
    },
    common::LiteralsDataWithSync {
        data: common::LitData:0x0504_0100,
        length: common::LitLength:4,
        last: false,
        id: u32:2,
        literals_last: false,
    },
    common::LiteralsDataWithSync {
        data: common::LitData:0x0504_0100,
        length: common::LitLength:4,
        last: true,
        id: u32:2,
        literals_last: false,
    },
    // Literals #3
    common::LiteralsDataWithSync {
        data: common::LitData:0x0504_0100,
        length: common::LitLength:4,
        last: false,
        id: u32:3,
        literals_last: true,
    },
    common::LiteralsDataWithSync {
        data: common::LitData:0x0504_0100,
        length: common::LitLength:4,
        last: false,
        id: u32:3,
        literals_last: true,
    },
    common::LiteralsDataWithSync {
        data: common::LitData:0x0504_0100,
        length: common::LitLength:4,
        last: false,
        id: u32:3,
        literals_last: true,
    },
    common::LiteralsDataWithSync {
        data: common::LitData:0x0504_0100,
        length: common::LitLength:4,
        last: true,
        id: u32:3,
        literals_last: true,
    },
];

#[test_proc]
proc HuffmanLiteralsDecoder_test {
    type Status = HuffmanLiteralsDecoderStatus;

    terminator: chan<bool> out;

    ctrl_s: chan<TestCtrl> out;
    resp_r: chan<TestResp> in;
    decoded_literals_r: chan<common::LiteralsDataWithSync> in;
    ram_wr_req_huffman_s : chan<TestAxiRamWrReq> out;
    ram_wr_resp_huffman_r : chan<TestAxiRamWrResp> in;
    ram_wr_req_jump_table_s : chan<TestAxiRamWrReq> out;
    ram_wr_resp_jump_table_r : chan<TestAxiRamWrResp> in;
    ram_wr_req_huffman_weights_header_s : chan<TestAxiRamWrReq> out;
    ram_wr_resp_huffman_weights_header_r : chan<TestAxiRamWrResp> in;
    ram_wr_req_huffman_weights_raw_s : chan<TestAxiRamWrReq> out;
    ram_wr_resp_huffman_weights_raw_r : chan<TestAxiRamWrResp> in;
    ram_wr_req_huffman_weights_fse_lookup_s : chan<TestAxiRamWrReq> out;
    ram_wr_resp_huffman_weights_fse_lookup_r : chan<TestAxiRamWrResp> in;
    ram_wr_req_huffman_weights_fse_decoder_s : chan<TestAxiRamWrReq> out;
    ram_wr_resp_huffman_weights_fse_decoder_r : chan<TestAxiRamWrResp> in;

    config (terminator: chan<bool> out) {
        let (ctrl_s, ctrl_r) = chan<TestCtrl>("ctrl");
        let (resp_s, resp_r) = chan<TestResp>("resp");
        let (decoded_literals_s, decoded_literals_r) = chan<common::LiteralsDataWithSync>("decoded_literals");
        let (axi_ar_s, axi_ar_r) = chan<TestAxiAr>("axi_ar");
        let (axi_r_s, axi_r_r) = chan<TestAxiR>("axi_r");
        let (jump_table_axi_ar_s, jump_table_axi_ar_r) = chan<TestAxiAr>("jump_table_axi_ar");
        let (jump_table_axi_r_s, jump_table_axi_r_r) = chan<TestAxiR>("jump_table_axi_r");
        let (weights_header_dec_axi_ar_s, weights_header_dec_axi_ar_r) = chan<TestAxiAr>("weights_header_dec_axi_ar");
        let (weights_header_dec_axi_r_s, weights_header_dec_axi_r_r) = chan<TestAxiR>("weights_header_dec_axi_r");
        let (weights_raw_dec_axi_ar_s, weights_raw_dec_axi_ar_r) = chan<TestAxiAr>("weights_raw_dec_axi_ar");
        let (weights_raw_dec_axi_r_s, weights_raw_dec_axi_r_r) = chan<TestAxiR>("weights_raw_dec_axi_r");
        let (weights_fse_lookup_dec_axi_ar_s, weights_fse_lookup_dec_axi_ar_r) = chan<TestAxiAr>("weights_fse_lookup_dec_axi_ar");
        let (weights_fse_lookup_dec_axi_r_s, weights_fse_lookup_dec_axi_r_r) = chan<TestAxiR>("weights_fse_lookup_dec_axi_r");
        let (weights_fse_decoder_dec_axi_ar_s, weights_fse_decoder_dec_axi_ar_r) = chan<TestAxiAr>("weights_fse_decoder_dec_axi_ar");
        let (weights_fse_decoder_dec_axi_r_s, weights_fse_decoder_dec_axi_r_r) = chan<TestAxiR>("weights_fse_decoder_dec_axi_r");

        // weights internal memory
        let (weights_ram_rd_req_s, weights_ram_rd_req_r) = chan<TestWeightsRamRdReq>("weights_ram_rd_req");
        let (weights_ram_rd_resp_s, weights_ram_rd_resp_r) = chan<TestWeightsRamRdResp>("weights_ram_rd_resp");
        let (weights_ram_wr_req_s, weights_ram_wr_req_r) = chan<TestWeightsRamWrReq>("weights_ram_wr_req");
        let (weights_ram_wr_resp_s, weights_ram_wr_resp_r) = chan<TestWeightsRamWrResp>("weights_ram_wr_resp");

        // prescan internal memory
        let (prescan_ram_wr_req_s, prescan_ram_wr_req_r) = chan<TestPrescanRamWrReq, u32:1>("prescan_ram_wr_req");
        let (prescan_ram_wr_resp_s, prescan_ram_wr_resp_r) = chan<TestPrescanRamWrResp, u32:1>("prescan_ram_wr_resp");
        let (prescan_ram_rd_req_s, prescan_ram_rd_req_r) = chan<TestPrescanRamRdReq, u32:1>("prescan_ram_rd_req");
        let (prescan_ram_rd_resp_s, prescan_ram_rd_resp_r) = chan<TestPrescanRamRdResp, u32:1>("prescan_ram_rd_resp");

        // Weights FSE RAMs
        let (weights_dpd_rd_req_s, weights_dpd_rd_req_r) = chan<TestWeightsDpdRamRdReq>("weights_dpd_rd_req");
        let (weights_dpd_rd_resp_s, weights_dpd_rd_resp_r) = chan<TestWeightsDpdRamRdResp>("weights_dpd_rd_resp");
        let (weights_dpd_wr_req_s, weights_dpd_wr_req_r) = chan<TestWeightsDpdRamWrReq>("weights_dpd_wr_req");
        let (weights_dpd_wr_resp_s, weights_dpd_wr_resp_r) = chan<TestWeightsDpdRamWrResp>("weights_dpd_wr_resp");

        spawn ram::RamModel<
            TEST_WEIGHTS_DPD_RAM_DATA_W,
            TEST_WEIGHTS_DPD_RAM_SIZE,
            TEST_WEIGHTS_DPD_RAM_WORD_PARTITION_SIZE
        >(weights_dpd_rd_req_r, weights_dpd_rd_resp_s, weights_dpd_wr_req_r, weights_dpd_wr_resp_s);

        let (weights_tmp_rd_req_s, weights_tmp_rd_req_r) = chan<TestWeightsTmpRamRdReq>("weights_tmp_rd_req");
        let (weights_tmp_rd_resp_s, weights_tmp_rd_resp_r) = chan<TestWeightsTmpRamRdResp>("weights_tmp_rd_resp");
        let (weights_tmp_wr_req_s, weights_tmp_wr_req_r) = chan<TestWeightsTmpRamWrReq>("weights_tmp_wr_req");
        let (weights_tmp_wr_resp_s, weights_tmp_wr_resp_r) = chan<TestWeightsTmpRamWrResp>("weights_tmp_wr_resp");

        spawn ram::RamModel<
            TEST_WEIGHTS_TMP_RAM_DATA_W,
            TEST_WEIGHTS_TMP_RAM_SIZE,
            TEST_WEIGHTS_TMP_RAM_WORD_PARTITION_SIZE
        >(weights_tmp_rd_req_r, weights_tmp_rd_resp_s, weights_tmp_wr_req_r, weights_tmp_wr_resp_s);

        let (weights_tmp2_rd_req_s, weights_tmp2_rd_req_r) = chan<TestWeightsTmp2RamRdReq>("weights_tmp_rd_req");
        let (weights_tmp2_rd_resp_s, weights_tmp2_rd_resp_r) = chan<TestWeightsTmp2RamRdResp>("weights_tmp_rd_resp");
        let (weights_tmp2_wr_req_s, weights_tmp2_wr_req_r) = chan<TestWeightsTmp2RamWrReq>("weights_tmp_wr_req");
        let (weights_tmp2_wr_resp_s, weights_tmp2_wr_resp_r) = chan<TestWeightsTmp2RamWrResp>("weights_tmp_wr_resp");

        spawn ram::RamModel<
            TEST_WEIGHTS_TMP2_RAM_DATA_W,
            TEST_WEIGHTS_TMP2_RAM_SIZE,
            TEST_WEIGHTS_TMP2_RAM_WORD_PARTITION_SIZE
        >(weights_tmp2_rd_req_r, weights_tmp2_rd_resp_s, weights_tmp2_wr_req_r, weights_tmp2_wr_resp_s);

        let (weights_fse_rd_req_s, weights_fse_rd_req_r) = chan<TestWeightsFseRamRdReq>("weights_tmp_rd_req");
        let (weights_fse_rd_resp_s, weights_fse_rd_resp_r) = chan<TestWeightsFseRamRdResp>("weights_tmp_rd_resp");
        let (weights_fse_wr_req_s, weights_fse_wr_req_r) = chan<TestWeightsFseRamWrReq>("weights_tmp_wr_req");
        let (weights_fse_wr_resp_s, weights_fse_wr_resp_r) = chan<TestWeightsFseRamWrResp>("weights_tmp_wr_resp");

        spawn ram::RamModel<
            TEST_WEIGHTS_FSE_RAM_DATA_W,
            TEST_WEIGHTS_FSE_RAM_SIZE,
            TEST_WEIGHTS_FSE_RAM_WORD_PARTITION_SIZE
        >(weights_fse_rd_req_r, weights_fse_rd_resp_s, weights_fse_wr_req_r, weights_fse_wr_resp_s);

        spawn HuffmanLiteralsDecoder<
            TEST_AXI_RAM_DATA_W, TEST_AXI_RAM_ADDR_W, TEST_AXI_RAM_ID_W, TEST_AXI_RAM_DEST_W,
            TEST_WEIGHTS_DPD_RAM_ADDR_W, TEST_WEIGHTS_DPD_RAM_DATA_W, TEST_WEIGHTS_DPD_RAM_NUM_PARTITIONS,
            TEST_WEIGHTS_TMP_RAM_ADDR_W, TEST_WEIGHTS_TMP_RAM_DATA_W, TEST_WEIGHTS_TMP_RAM_NUM_PARTITIONS,
            TEST_WEIGHTS_TMP2_RAM_ADDR_W, TEST_WEIGHTS_TMP2_RAM_DATA_W, TEST_WEIGHTS_TMP2_RAM_NUM_PARTITIONS,
            TEST_WEIGHTS_FSE_RAM_ADDR_W, TEST_WEIGHTS_FSE_RAM_DATA_W, TEST_WEIGHTS_FSE_RAM_NUM_PARTITIONS,
            TEST_WEIGHTS_RAM_ADDR_WIDTH, TEST_WEIGHTS_RAM_DATA_WIDTH, TEST_WEIGHTS_RAM_NUM_PARTITIONS,
            TEST_PRESCAN_RAM_ADDR_WIDTH, TEST_PRESCAN_RAM_DATA_WIDTH, TEST_PRESCAN_RAM_NUM_PARTITIONS,
        >(
            ctrl_r, resp_s, decoded_literals_s,
            axi_ar_s, axi_r_r,
            jump_table_axi_ar_s, jump_table_axi_r_r,
            weights_header_dec_axi_ar_s, weights_header_dec_axi_r_r,
            weights_raw_dec_axi_ar_s, weights_raw_dec_axi_r_r,
            weights_fse_lookup_dec_axi_ar_s, weights_fse_lookup_dec_axi_r_r,
            weights_fse_decoder_dec_axi_ar_s, weights_fse_decoder_dec_axi_r_r,
            weights_ram_rd_req_s, weights_ram_rd_resp_r,
            weights_ram_wr_req_s, weights_ram_wr_resp_r,
            prescan_ram_rd_req_s, prescan_ram_rd_resp_r,
            prescan_ram_wr_req_s, prescan_ram_wr_resp_r,
            weights_dpd_rd_req_s, weights_dpd_rd_resp_r, weights_dpd_wr_req_s, weights_dpd_wr_resp_r,
            weights_tmp_rd_req_s, weights_tmp_rd_resp_r, weights_tmp_wr_req_s, weights_tmp_wr_resp_r,
            weights_tmp2_rd_req_s, weights_tmp2_rd_resp_r, weights_tmp2_wr_req_s, weights_tmp2_wr_resp_r,
            weights_fse_rd_req_s, weights_fse_rd_resp_r, weights_fse_wr_req_s, weights_fse_wr_resp_r,
        );

        // Mock RAM for HuffmanLiteralsDecoder MemReader
        let (ram_rd_req_huffman_s, ram_rd_req_huffman_r) = chan<TestAxiRamRdReq>("ram_rd_req_huffman");
        let (ram_rd_resp_huffman_s, ram_rd_resp_huffman_r) = chan<TestAxiRamRdResp>("ram_rd_resp_huffman");
        let (ram_wr_req_huffman_s, ram_wr_req_huffman_r) = chan<TestAxiRamWrReq>("ram_wr_req_huffman");
        let (ram_wr_resp_huffman_s, ram_wr_resp_huffman_r) = chan<TestAxiRamWrResp>("ram_wr_resp_huffman");

        spawn ram::RamModel<
            TEST_AXI_RAM_MODEL_DATA_WIDTH,
            TEST_AXI_RAM_MODEL_SIZE,
            TEST_AXI_RAM_MODEL_WORD_PARTITION_SIZE,
            TEST_AXI_RAM_MODEL_SIMULTANEOUS_READ_WRITE_BEHAVIOR,
            TEST_AXI_RAM_MODEL_INITIALIZED,
            TEST_AXI_RAM_MODEL_ASSERT_VALID_READ,
            TEST_AXI_RAM_MODEL_ADDR_WIDTH
        > (
            ram_rd_req_huffman_r, ram_rd_resp_huffman_s, ram_wr_req_huffman_r, ram_wr_resp_huffman_s
        );

        spawn axi_ram::AxiRamReader<
            TEST_AXI_RAM_ADDR_W, TEST_AXI_RAM_DATA_W, TEST_AXI_RAM_DEST_W, TEST_AXI_RAM_ID_W,
            TEST_AXI_RAM_MODEL_SIZE, TEST_AXI_RAM_MODEL_BASE_ADDR, TEST_AXI_RAM_MODEL_DATA_WIDTH, TEST_AXI_RAM_MODEL_ADDR_WIDTH
        > (
            axi_ar_r, axi_r_s,
            ram_rd_req_huffman_s, ram_rd_resp_huffman_r
        );

        // Mock RAM for Huffman Jump Table decoder MemReader
        let (ram_rd_req_jump_table_s, ram_rd_req_jump_table_r) = chan<TestAxiRamRdReq>("ram_rd_req_jump_table");
        let (ram_rd_resp_jump_table_s, ram_rd_resp_jump_table_r) = chan<TestAxiRamRdResp>("ram_rd_resp_jump_table");
        let (ram_wr_req_jump_table_s, ram_wr_req_jump_table_r) = chan<TestAxiRamWrReq>("ram_wr_req_jump_table");
        let (ram_wr_resp_jump_table_s, ram_wr_resp_jump_table_r) = chan<TestAxiRamWrResp>("ram_wr_resp_jump_table");

        spawn ram::RamModel<
            TEST_AXI_RAM_MODEL_DATA_WIDTH,
            TEST_AXI_RAM_MODEL_SIZE,
            TEST_AXI_RAM_MODEL_WORD_PARTITION_SIZE,
            TEST_AXI_RAM_MODEL_SIMULTANEOUS_READ_WRITE_BEHAVIOR,
            TEST_AXI_RAM_MODEL_INITIALIZED,
            TEST_AXI_RAM_MODEL_ASSERT_VALID_READ,
            TEST_AXI_RAM_MODEL_ADDR_WIDTH
        > (
            ram_rd_req_jump_table_r, ram_rd_resp_jump_table_s, ram_wr_req_jump_table_r, ram_wr_resp_jump_table_s
        );

        spawn axi_ram::AxiRamReader<
            TEST_AXI_RAM_ADDR_W, TEST_AXI_RAM_DATA_W, TEST_AXI_RAM_DEST_W, TEST_AXI_RAM_ID_W,
            TEST_AXI_RAM_MODEL_SIZE, TEST_AXI_RAM_MODEL_BASE_ADDR, TEST_AXI_RAM_MODEL_DATA_WIDTH, TEST_AXI_RAM_MODEL_ADDR_WIDTH
        > (
            jump_table_axi_ar_r, jump_table_axi_r_s,
            ram_rd_req_jump_table_s, ram_rd_resp_jump_table_r
        );

        // Mock RAM for HuffmanWeights header decoder MemReader
        let (ram_rd_req_huffman_weights_header_s, ram_rd_req_huffman_weights_header_r) = chan<TestAxiRamRdReq>("ram_rd_req_huffman_weights_header");
        let (ram_rd_resp_huffman_weights_header_s, ram_rd_resp_huffman_weights_header_r) = chan<TestAxiRamRdResp>("ram_rd_resp_huffman_weights_header");
        let (ram_wr_req_huffman_weights_header_s, ram_wr_req_huffman_weights_header_r) = chan<TestAxiRamWrReq>("ram_wr_req_huffman_weights_header");
        let (ram_wr_resp_huffman_weights_header_s, ram_wr_resp_huffman_weights_header_r) = chan<TestAxiRamWrResp>("ram_wr_resp_huffman_weights_header");

        spawn ram::RamModel<
            TEST_AXI_RAM_MODEL_DATA_WIDTH,
            TEST_AXI_RAM_MODEL_SIZE,
            TEST_AXI_RAM_MODEL_WORD_PARTITION_SIZE,
            TEST_AXI_RAM_MODEL_SIMULTANEOUS_READ_WRITE_BEHAVIOR,
            TEST_AXI_RAM_MODEL_INITIALIZED,
            TEST_AXI_RAM_MODEL_ASSERT_VALID_READ,
            TEST_AXI_RAM_MODEL_ADDR_WIDTH
        > (
            ram_rd_req_huffman_weights_header_r, ram_rd_resp_huffman_weights_header_s, ram_wr_req_huffman_weights_header_r, ram_wr_resp_huffman_weights_header_s
        );

        spawn axi_ram::AxiRamReader<
            TEST_AXI_RAM_ADDR_W, TEST_AXI_RAM_DATA_W, TEST_AXI_RAM_DEST_W, TEST_AXI_RAM_ID_W,
            TEST_AXI_RAM_MODEL_SIZE, TEST_AXI_RAM_MODEL_BASE_ADDR, TEST_AXI_RAM_MODEL_DATA_WIDTH, TEST_AXI_RAM_MODEL_ADDR_WIDTH
        > (
            weights_header_dec_axi_ar_r, weights_header_dec_axi_r_s,
            ram_rd_req_huffman_weights_header_s, ram_rd_resp_huffman_weights_header_r
        );

        // Mock RAM for HuffmanWeights raw decoder MemReader
        let (ram_rd_req_huffman_weights_raw_s, ram_rd_req_huffman_weights_raw_r) = chan<TestAxiRamRdReq>("ram_rd_req_huffman_weights_raw");
        let (ram_rd_resp_huffman_weights_raw_s, ram_rd_resp_huffman_weights_raw_r) = chan<TestAxiRamRdResp>("ram_rd_resp_huffman_weights_raw");
        let (ram_wr_req_huffman_weights_raw_s, ram_wr_req_huffman_weights_raw_r) = chan<TestAxiRamWrReq>("ram_wr_req_huffman_weights_raw");
        let (ram_wr_resp_huffman_weights_raw_s, ram_wr_resp_huffman_weights_raw_r) = chan<TestAxiRamWrResp>("ram_wr_resp_huffman_weights_raw");

        spawn ram::RamModel<
            TEST_AXI_RAM_MODEL_DATA_WIDTH,
            TEST_AXI_RAM_MODEL_SIZE,
            TEST_AXI_RAM_MODEL_WORD_PARTITION_SIZE,
            TEST_AXI_RAM_MODEL_SIMULTANEOUS_READ_WRITE_BEHAVIOR,
            TEST_AXI_RAM_MODEL_INITIALIZED,
            TEST_AXI_RAM_MODEL_ASSERT_VALID_READ,
            TEST_AXI_RAM_MODEL_ADDR_WIDTH
        > (
            ram_rd_req_huffman_weights_raw_r, ram_rd_resp_huffman_weights_raw_s, ram_wr_req_huffman_weights_raw_r, ram_wr_resp_huffman_weights_raw_s
        );

        spawn axi_ram::AxiRamReader<
            TEST_AXI_RAM_ADDR_W, TEST_AXI_RAM_DATA_W, TEST_AXI_RAM_DEST_W, TEST_AXI_RAM_ID_W,
            TEST_AXI_RAM_MODEL_SIZE, TEST_AXI_RAM_MODEL_BASE_ADDR, TEST_AXI_RAM_MODEL_DATA_WIDTH, TEST_AXI_RAM_MODEL_ADDR_WIDTH
        > (
            weights_raw_dec_axi_ar_r, weights_raw_dec_axi_r_s,
            ram_rd_req_huffman_weights_raw_s, ram_rd_resp_huffman_weights_raw_r
        );

        // Mock RAM for HuffmanWeights fse decoder MemReader
        let (ram_rd_req_huffman_weights_fse_lookup_s, ram_rd_req_huffman_weights_fse_lookup_r) = chan<TestAxiRamRdReq>("ram_rd_req_huffman_weights_fse_lookup");
        let (ram_rd_resp_huffman_weights_fse_lookup_s, ram_rd_resp_huffman_weights_fse_lookup_r) = chan<TestAxiRamRdResp>("ram_rd_resp_huffman_weights_fse_lookup");
        let (ram_wr_req_huffman_weights_fse_lookup_s, ram_wr_req_huffman_weights_fse_lookup_r) = chan<TestAxiRamWrReq>("ram_wr_req_huffman_weights_fse_lookup");
        let (ram_wr_resp_huffman_weights_fse_lookup_s, ram_wr_resp_huffman_weights_fse_lookup_r) = chan<TestAxiRamWrResp>("ram_wr_resp_huffman_weights_fse_lookup");

        spawn ram::RamModel<
            TEST_AXI_RAM_MODEL_DATA_WIDTH,
            TEST_AXI_RAM_MODEL_SIZE,
            TEST_AXI_RAM_MODEL_WORD_PARTITION_SIZE,
            TEST_AXI_RAM_MODEL_SIMULTANEOUS_READ_WRITE_BEHAVIOR,
            TEST_AXI_RAM_MODEL_INITIALIZED,
            TEST_AXI_RAM_MODEL_ASSERT_VALID_READ,
            TEST_AXI_RAM_MODEL_ADDR_WIDTH
        > (
            ram_rd_req_huffman_weights_fse_lookup_r, ram_rd_resp_huffman_weights_fse_lookup_s, ram_wr_req_huffman_weights_fse_lookup_r, ram_wr_resp_huffman_weights_fse_lookup_s
        );

        spawn axi_ram::AxiRamReader<
            TEST_AXI_RAM_ADDR_W, TEST_AXI_RAM_DATA_W, TEST_AXI_RAM_DEST_W, TEST_AXI_RAM_ID_W,
            TEST_AXI_RAM_MODEL_SIZE, TEST_AXI_RAM_MODEL_BASE_ADDR, TEST_AXI_RAM_MODEL_DATA_WIDTH, TEST_AXI_RAM_MODEL_ADDR_WIDTH
        > (
            weights_fse_lookup_dec_axi_ar_r, weights_fse_lookup_dec_axi_r_s,
            ram_rd_req_huffman_weights_fse_lookup_s, ram_rd_resp_huffman_weights_fse_lookup_r
        );

        let (ram_rd_req_huffman_weights_fse_decoder_s, ram_rd_req_huffman_weights_fse_decoder_r) = chan<TestAxiRamRdReq>("ram_rd_req_huffman_weights_fse");
        let (ram_rd_resp_huffman_weights_fse_decoder_s, ram_rd_resp_huffman_weights_fse_decoder_r) = chan<TestAxiRamRdResp>("ram_rd_resp_huffman_weights_fse");
        let (ram_wr_req_huffman_weights_fse_decoder_s, ram_wr_req_huffman_weights_fse_decoder_r) = chan<TestAxiRamWrReq>("ram_wr_req_huffman_weights_fse");
        let (ram_wr_resp_huffman_weights_fse_decoder_s, ram_wr_resp_huffman_weights_fse_decoder_r) = chan<TestAxiRamWrResp>("ram_wr_resp_huffman_weights_fse");

        spawn ram::RamModel<
            TEST_AXI_RAM_MODEL_DATA_WIDTH,
            TEST_AXI_RAM_MODEL_SIZE,
            TEST_AXI_RAM_MODEL_WORD_PARTITION_SIZE,
            TEST_AXI_RAM_MODEL_SIMULTANEOUS_READ_WRITE_BEHAVIOR,
            TEST_AXI_RAM_MODEL_INITIALIZED,
            TEST_AXI_RAM_MODEL_ASSERT_VALID_READ,
            TEST_AXI_RAM_MODEL_ADDR_WIDTH
        > (
            ram_rd_req_huffman_weights_fse_decoder_r, ram_rd_resp_huffman_weights_fse_decoder_s, ram_wr_req_huffman_weights_fse_decoder_r, ram_wr_resp_huffman_weights_fse_decoder_s
        );

        spawn axi_ram::AxiRamReader<
            TEST_AXI_RAM_ADDR_W, TEST_AXI_RAM_DATA_W, TEST_AXI_RAM_DEST_W, TEST_AXI_RAM_ID_W,
            TEST_AXI_RAM_MODEL_SIZE, TEST_AXI_RAM_MODEL_BASE_ADDR, TEST_AXI_RAM_MODEL_DATA_WIDTH, TEST_AXI_RAM_MODEL_ADDR_WIDTH
        > (
            weights_fse_decoder_dec_axi_ar_r, weights_fse_decoder_dec_axi_r_s,
            ram_rd_req_huffman_weights_fse_decoder_s, ram_rd_resp_huffman_weights_fse_decoder_r
        );

        spawn ram::RamModel<
            TEST_WEIGHTS_RAM_DATA_WIDTH, TEST_WEIGHTS_RAM_SIZE, TEST_WEIGHTS_WORD_PARTITION_SIZE
            >(
            weights_ram_rd_req_r, weights_ram_rd_resp_s,
            weights_ram_wr_req_r, weights_ram_wr_resp_s,
        );

        spawn ram::RamModel<
            TEST_PRESCAN_RAM_DATA_WIDTH, TEST_PRESCAN_RAM_SIZE, TEST_PRESCAN_WORD_PARTITION_SIZE
            >(
            prescan_ram_rd_req_r, prescan_ram_rd_resp_s,
            prescan_ram_wr_req_r, prescan_ram_wr_resp_s,
        );

        (
            terminator,
            ctrl_s, resp_r, decoded_literals_r,
            ram_wr_req_huffman_s, ram_wr_resp_huffman_r,
            ram_wr_req_jump_table_s, ram_wr_resp_jump_table_r,
            ram_wr_req_huffman_weights_header_s, ram_wr_resp_huffman_weights_header_r,
            ram_wr_req_huffman_weights_raw_s, ram_wr_resp_huffman_weights_raw_r,
            ram_wr_req_huffman_weights_fse_lookup_s, ram_wr_resp_huffman_weights_fse_lookup_r,
            ram_wr_req_huffman_weights_fse_decoder_s, ram_wr_resp_huffman_weights_fse_decoder_r,
        )

    }

    init { }

    next (state: ()) {
        let tok = join();

        trace_fmt!("Filling system memory mock");
        let tok = for ((i, mem_req), tok):((u32, TestAxiRamWrReq), token) in enumerate(TEST_MEMORY) {
            trace_fmt!("Sent memory write request #{}: {:#x}", i + u32:1, mem_req);
            let tok = send(tok, ram_wr_req_huffman_s, mem_req);
            let (tok, _) = recv(tok, ram_wr_resp_huffman_r);
            let tok = send(tok, ram_wr_req_jump_table_s, mem_req);
            let (tok, _) = recv(tok, ram_wr_resp_jump_table_r);
            let tok = send(tok, ram_wr_req_huffman_weights_header_s, mem_req);
            let (tok, _) = recv(tok, ram_wr_resp_huffman_weights_header_r);
            let tok = send(tok, ram_wr_req_huffman_weights_raw_s, mem_req);
            let (tok, _) = recv(tok, ram_wr_resp_huffman_weights_raw_r);
            let tok = send(tok, ram_wr_req_huffman_weights_fse_lookup_s, mem_req);
            let (tok, _) = recv(tok, ram_wr_resp_huffman_weights_fse_lookup_r);
            let tok = send(tok, ram_wr_req_huffman_weights_fse_decoder_s, mem_req);
            let (tok, _) = recv(tok, ram_wr_resp_huffman_weights_fse_decoder_r);
            tok
        }(tok);
        trace_fmt!("Filling system memory mock done");

        // Send Huffman Literals decoding requests
        let tok = for ((i, ctrl_req), tok):((u32, TestCtrl), token) in enumerate(TEST_CTRL) {
            let tok = send(tok, ctrl_s, ctrl_req);
            trace_fmt!("Sent #{} ctrl {:#x}", i + u32:1, ctrl_req);
            tok
        }(tok);

        // receive decoded literals
        let tok = for ((i, expected_decoded_literals), tok):((u32, common::LiteralsDataWithSync), token) in enumerate(TEST_DECODED_LITERALS) {
            trace_fmt!("Waiting for #{} decoded literals", i + u32:1);
            let (tok, decoded_literals) = recv(tok, decoded_literals_r);
            trace_fmt!("Received #{} decoded literals {:#x}", i + u32:1, decoded_literals);
            assert_eq(expected_decoded_literals, decoded_literals);

            if (decoded_literals.last) { trace_fmt!("Waiting for #{} decoding response", i + u32:1); } else {};
            let (tok, resp) = recv_if(tok, resp_r, decoded_literals.last, zero!<TestResp>());
            if (decoded_literals.last) { trace_fmt!("Received #{} decoding response {:#x}", i + u32:1, resp); } else {};
            assert_eq(TestResp {status: Status::OKAY}, resp);

            tok
        }(tok);

        send(tok, terminator, true);
    }
}

// TODO: implement tests with the following Huffman Tree
//const TEST_DATA_LEN_0 = u32:64;
//const TEST_DATA_0 = (
//    u8:0b001_1_010_0 ++  // 0x34 <- last byte in the memory
//    u8:0b11_1_1_0001 ++  // 0xF1
//    u8:0b01_010_000 ++   // 0x50
//    u8:0b001_010_1_0 ++  // 0x2A
//    u8:0b11_010_1_00 ++  // 0xD4
//    u8:0b0100_001_0 ++   // 0x42
//    u8:0b01_010_1_01 ++  // 0x55
//    u8:0b1_001_010_1     // 0x95 <- first byte in the memory
//);
//
//// code         symbol  length  weight
//// 0b1          0x47    1       9
//// 0b001        0x41    3       7
//// 0b010        0x8A    3       7
//// 0b011        0xD2    3       7
//// 0b000001     0x45    6       4
//// 0b000010     0x7A    6       4
//// 0b000011     0x89    6       4
//// 0b000100     0x8D    6       4
//// 0b000101     0xD1    6       4
//// 0b000110     0xD3    6       4
//// 0b000111     0xDA    6       4
//// 0b000000000  0x12    9       1
//// 0b000000001  0x8F    9       1
//// 0b000000010  0xAC    9       1
//// 0b000000011  0xD4    9       1
//// 0b000000100  0xD7    9       1
//// 0b000000101  0xDB    9       1
//// 0b000000110  0xDE    9       1
//// 0b000000111  0xFE    9       1
//
//const TEST_WEIGHT_MEMORY_0 = TestRamEntry[32]:[
//    //             x0 x1 x2 x3 x4 x5 x6 x7                 x8 x9 xA xB xC xD xE xF
//    TestRamEntry:0x_0__0__0__0__0__0__0__0, TestRamEntry:0x_0__0__0__0__0__0__0__0, // 0x0x
//    TestRamEntry:0x_0__0__1__0__0__0__0__0, TestRamEntry:0x_0__0__0__0__0__0__0__0, // 0x1x
//    TestRamEntry:0x_0__0__0__0__0__0__0__0, TestRamEntry:0x_0__0__0__0__0__0__0__0, // 0x2x
//    TestRamEntry:0x_0__0__0__0__0__0__0__0, TestRamEntry:0x_0__0__0__0__0__0__0__0, // 0x3x
//    TestRamEntry:0x_0__7__0__0__0__4__0__9, TestRamEntry:0x_0__0__0__0__0__0__0__0, // 0x4x
//    TestRamEntry:0x_0__0__0__0__0__0__0__0, TestRamEntry:0x_0__0__0__0__0__0__0__0, // 0x5x
//    TestRamEntry:0x_0__0__0__0__0__0__0__0, TestRamEntry:0x_0__0__0__0__0__0__0__0, // 0x6x
//    TestRamEntry:0x_0__0__0__0__0__0__0__0, TestRamEntry:0x_0__0__4__0__0__0__0__0, // 0x7x
//    TestRamEntry:0x_0__0__0__0__0__0__0__0, TestRamEntry:0x_0__4__7__0__0__4__0__1, // 0x8x
//    TestRamEntry:0x_0__0__0__0__0__0__0__0, TestRamEntry:0x_0__0__0__0__0__0__0__0, // 0x9x
//    TestRamEntry:0x_0__0__0__0__0__0__0__0, TestRamEntry:0x_0__0__0__0__1__0__0__0, // 0xAx
//    TestRamEntry:0x_0__0__0__0__0__0__0__0, TestRamEntry:0x_0__0__0__0__0__0__0__0, // 0xBx
//    TestRamEntry:0x_0__0__0__0__0__0__0__0, TestRamEntry:0x_0__0__0__0__0__0__0__0, // 0xCx
//    TestRamEntry:0x_0__4__7__4__1__0__0__1, TestRamEntry:0x_0__0__4__1__0__0__1__0, // 0xDx
//    TestRamEntry:0x_0__0__0__0__0__0__0__0, TestRamEntry:0x_0__0__0__0__0__0__0__0, // 0xEx
//    TestRamEntry:0x_0__0__0__0__0__0__0__0, TestRamEntry:0x_0__0__0__0__0__0__1__0, // 0xFx
//];
//
//const TEST_DECODED_LITERALS_0 = common::LiteralsDataWithSync[3]:[
//    common::LiteralsDataWithSync {
//        data: common::LitData:0x458A_D147_47D2_8A47,
//        length: common::LitLength:8,
//        last: false,
//        id: u32:0,
//        literals_last: false,
//    },
//    common::LiteralsDataWithSync {
//        data: common::LitData:0x4141_8D47_8AD2_478A,
//        length: common::LitLength:8,
//        last: false,
//        id: u32:0,
//        literals_last: false,
//    },
//    common::LiteralsDataWithSync {
//        data: common::LitData:0x478A_41D2_478A,
//        length: common::LitLength:6,
//        last: true,
//        id: u32:0,
//        literals_last: false,
//    },
//];
//
//// data for test case #1 (same config)
//const TEST_CTRL_1 = TestCtrl {
//    base_addr: uN[TEST_AXI_RAM_ADDR_W]:0x20,
//    len: uN[TEST_AXI_RAM_ADDR_W]:0x4,
//    new_config: false,
//    multi_stream: false,
//    id: u32:1,
//    literals_last: true,
//};
//
//const TEST_DATA_LEN_1 = u32:32;
//const TEST_DATA_1 = (
//    u8:0b001_011_1_1 ++ // 0x2F <- last byte in the memory
//    u8:0b1_1_000000 ++  // 0xC0
//    u8:0b000_0_000 ++   // 0x00
//    u8:0b0010_1_010     // 0x2A <- first byte in the memory
//);
//
//const TEST_DECODED_LITERALS_1 = common::LiteralsDataWithSync[2]:[
//    common::LiteralsDataWithSync {
//        data: common::LitData:0x47AC_1247_4747_47D2,
//        length: common::LitLength:8,
//        last: false,
//        id: u32:1,
//        literals_last: true,
//    },
//    common::LiteralsDataWithSync {
//        data: common::LitData:0x8A,
//        length: common::LitLength:1,
//        last: true,
//        id: u32:1,
//        literals_last: true,
//    },
//];

