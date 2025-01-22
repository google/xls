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
import xls.modules.zstd.memory.mem_reader;
import xls.modules.zstd.memory.axi_ram;
import xls.modules.zstd.fse_table_creator;
import xls.modules.zstd.refilling_shift_buffer;
import xls.modules.zstd.fse_proba_freq_dec;
import xls.modules.shift_buffer.shift_buffer;
import xls.modules.zstd.comp_lookup_dec;
import xls.modules.zstd.rle_lookup_dec;
import xls.modules.zstd.refilling_shift_buffer_mux;
import xls.modules.zstd.ram_mux;

type AccuracyLog = common::FseAccuracyLog;

pub struct FseLookupDecoderReq { is_rle: bool }
pub type FseLookupDecoderStatus = common::LookupDecoderStatus;
pub type FseLookupDecoderResp = common::LookupDecoderResp;

pub proc FseLookupDecoder<
    AXI_DATA_W: u32,
    DPD_RAM_DATA_W: u32, DPD_RAM_ADDR_W: u32, DPD_RAM_NUM_PARTITIONS: u32,
    TMP_RAM_DATA_W: u32, TMP_RAM_ADDR_W: u32, TMP_RAM_NUM_PARTITIONS: u32,
    TMP2_RAM_DATA_W: u32, TMP2_RAM_ADDR_W: u32, TMP2_RAM_NUM_PARTITIONS: u32,
    FSE_RAM_DATA_W: u32, FSE_RAM_ADDR_W: u32, FSE_RAM_NUM_PARTITIONS: u32,
    SB_LENGTH_W: u32 = {refilling_shift_buffer::length_width(AXI_DATA_W)},
> {
    type Req = FseLookupDecoderReq;
    type Resp = FseLookupDecoderResp;

    type DpdRamRdReq = ram::ReadReq<DPD_RAM_ADDR_W, DPD_RAM_NUM_PARTITIONS>;
    type DpdRamRdResp = ram::ReadResp<DPD_RAM_DATA_W>;
    type DpdRamWrReq = ram::WriteReq<DPD_RAM_ADDR_W, DPD_RAM_DATA_W, DPD_RAM_NUM_PARTITIONS>;
    type DpdRamWrResp = ram::WriteResp;

    type FseRamRdReq = ram::ReadReq<FSE_RAM_ADDR_W, FSE_RAM_NUM_PARTITIONS>;
    type FseRamRdResp = ram::ReadResp<FSE_RAM_DATA_W>;
    type FseRamWrReq = ram::WriteReq<FSE_RAM_ADDR_W, FSE_RAM_DATA_W, FSE_RAM_NUM_PARTITIONS>;
    type FseRamWrResp = ram::WriteResp;

    type TmpRamRdReq = ram::ReadReq<TMP_RAM_ADDR_W, TMP_RAM_NUM_PARTITIONS>;
    type TmpRamRdResp = ram::ReadResp<TMP_RAM_DATA_W>;
    type TmpRamWrReq = ram::WriteReq<TMP_RAM_ADDR_W, TMP_RAM_DATA_W, TMP_RAM_NUM_PARTITIONS>;
    type TmpRamWrResp = ram::WriteResp;

    type Tmp2RamRdReq = ram::ReadReq<TMP2_RAM_ADDR_W, TMP2_RAM_NUM_PARTITIONS>;
    type Tmp2RamRdResp = ram::ReadResp<TMP2_RAM_DATA_W>;
    type Tmp2RamWrReq = ram::WriteReq<TMP2_RAM_ADDR_W, TMP2_RAM_DATA_W, TMP2_RAM_NUM_PARTITIONS>;
    type Tmp2RamWrResp = ram::WriteResp;

    type SBOutput = refilling_shift_buffer::RefillingShiftBufferOutput<AXI_DATA_W, SB_LENGTH_W>;
    type SBCtrl = refilling_shift_buffer::RefillingShiftBufferCtrl<SB_LENGTH_W>;

    type LookupDecoderReq = common::LookupDecoderReq;
    type LookupDecoderResp = common::LookupDecoderResp;


    init {}

    fse_lookup_dec_req_r: chan<Req> in;
    fse_lookup_dec_resp_s: chan<Resp> out;

    comp_lookup_req_s: chan<LookupDecoderReq> out;
    comp_lookup_resp_r: chan<LookupDecoderResp> in;

    rle_lookup_req_s: chan<LookupDecoderReq> out;
    rle_lookup_resp_r: chan<LookupDecoderResp> in;

    shift_buffer_sel_req_s: chan<u1> out;
    shift_buffer_sel_resp_r: chan<()> in;

    fse_ram_sel_req_s: chan<u1> out;

    fse_rd_req0_s: chan<FseRamRdReq> out;
    fse_rd_resp0_r: chan<FseRamRdResp> in;

    fse_rd_req1_s: chan<FseRamRdReq> out;
    fse_rd_resp1_r: chan<FseRamRdResp> in;

    fse_rd_req_r: chan<FseRamRdReq> in;
    fse_rd_resp_s: chan<FseRamRdResp> out;

    config(
        fse_lookup_dec_req_r: chan<Req> in,
        fse_lookup_dec_resp_s: chan<Resp> out,

        dpd_rd_req_s: chan<DpdRamRdReq> out,
        dpd_rd_resp_r: chan<DpdRamRdResp> in,
        dpd_wr_req_s: chan<DpdRamWrReq> out,
        dpd_wr_resp_r: chan<DpdRamWrResp> in,

        tmp_rd_req_s: chan<TmpRamRdReq> out,
        tmp_rd_resp_r: chan<TmpRamRdResp> in,
        tmp_wr_req_s: chan<TmpRamWrReq> out,
        tmp_wr_resp_r: chan<TmpRamWrResp> in,

        tmp2_rd_req_s: chan<Tmp2RamRdReq> out,
        tmp2_rd_resp_r: chan<Tmp2RamRdResp> in,
        tmp2_wr_req_s: chan<Tmp2RamWrReq> out,
        tmp2_wr_resp_r: chan<Tmp2RamWrResp> in,

        fse_wr_req_s: chan<FseRamWrReq> out,
        fse_wr_resp_r: chan<FseRamWrResp> in,

        shift_buffer_ctrl_s: chan<SBCtrl> out,
        shift_buffer_data_r: chan<SBOutput> in,
    ) {
        const CHANNEL_DEPTH = u32:1;

        let (shift_buffer_sel_req_s, shift_buffer_sel_req_r) = chan<u1, CHANNEL_DEPTH>("shift_buffer_sel_req");
        let (shift_buffer_sel_resp_s, shift_buffer_sel_resp_r) = chan<(), CHANNEL_DEPTH>("shift_buffer_sel_resp");

        let (shift_buffer_ctrl0_s, shift_buffer_ctrl0_r) = chan<SBCtrl, CHANNEL_DEPTH>("shift_buffer_ctrl0");
        let (shift_buffer_data0_s, shift_buffer_data0_r) = chan<SBOutput, CHANNEL_DEPTH>("shift_buffer_data0");

        let (shift_buffer_ctrl1_s, shift_buffer_ctrl1_r) = chan<SBCtrl, CHANNEL_DEPTH>("shift_buffer_ctrl1");
        let (shift_buffer_data1_s, shift_buffer_data1_r) = chan<SBOutput, CHANNEL_DEPTH>("shift_buffer_data1");

        spawn refilling_shift_buffer_mux::RefillingShiftBufferMux<AXI_DATA_W, SB_LENGTH_W>(
            shift_buffer_sel_req_r, shift_buffer_sel_resp_s,
            shift_buffer_ctrl0_r, shift_buffer_data0_s,
            shift_buffer_ctrl1_r, shift_buffer_data1_s,
            shift_buffer_ctrl_s, shift_buffer_data_r,
        );

        let (fse_ram_sel_req_s, fse_ram_sel_req_r) = chan<u1, CHANNEL_DEPTH>("fse_ram_sel_req");

        let (fse_rd_req_s, fse_rd_req_r) = chan<FseRamRdReq, CHANNEL_DEPTH>("fse_rd_req");
        let (fse_rd_resp_s, fse_rd_resp_r) = chan<FseRamRdResp, CHANNEL_DEPTH>("fse_rd_resp");

        let (fse_rd_req0_s, fse_rd_req0_r) = chan<FseRamRdReq, CHANNEL_DEPTH>("fse_rd_req0");
        let (fse_rd_resp0_s, fse_rd_resp0_r) = chan<FseRamRdResp, CHANNEL_DEPTH>("fse_rd_resp0");
        let (fse_wr_req0_s, fse_wr_req0_r) = chan<FseRamWrReq, CHANNEL_DEPTH>("fse_wr_req0");
        let (fse_wr_resp0_s, fse_wr_resp0_r) = chan<FseRamWrResp, CHANNEL_DEPTH>("fse_wr_resp0");

        let (fse_rd_req1_s, fse_rd_req1_r) = chan<FseRamRdReq, CHANNEL_DEPTH>("fse_wr_req1");
        let (fse_rd_resp1_s, fse_rd_resp1_r) = chan<FseRamRdResp, CHANNEL_DEPTH>("fse_wr_resp1");
        let (fse_wr_req1_s, fse_wr_req1_r) = chan<FseRamWrReq, CHANNEL_DEPTH>("fse_wr_req1");
        let (fse_wr_resp1_s, fse_wr_resp1_r) = chan<FseRamWrResp, CHANNEL_DEPTH>("fse_wr_resp1");

        spawn ram_mux::RamMux<
            FSE_RAM_ADDR_W, FSE_RAM_DATA_W, FSE_RAM_NUM_PARTITIONS,
        >(
            fse_ram_sel_req_r,
            fse_rd_req0_r, fse_rd_resp0_s, fse_wr_req0_r, fse_wr_resp0_s,
            fse_rd_req1_r, fse_rd_resp1_s, fse_wr_req1_r, fse_wr_resp1_s,
            fse_rd_req_s, fse_rd_resp_r, fse_wr_req_s, fse_wr_resp_r,
        );

        let (comp_lookup_req_s, comp_lookup_req_r) = chan<LookupDecoderReq, CHANNEL_DEPTH>("comp_lookup_req");
        let (comp_lookup_resp_s, comp_lookup_resp_r) = chan<LookupDecoderResp, CHANNEL_DEPTH>("comp_lookup_resp");

        spawn comp_lookup_dec::CompLookupDecoder<
            AXI_DATA_W,
            DPD_RAM_DATA_W, DPD_RAM_ADDR_W, DPD_RAM_NUM_PARTITIONS,
            TMP_RAM_DATA_W, TMP_RAM_ADDR_W, TMP_RAM_NUM_PARTITIONS,
            TMP2_RAM_DATA_W, TMP2_RAM_ADDR_W, TMP2_RAM_NUM_PARTITIONS,
            FSE_RAM_DATA_W, FSE_RAM_ADDR_W, FSE_RAM_NUM_PARTITIONS,
        >(
            comp_lookup_req_r, comp_lookup_resp_s,
            dpd_rd_req_s, dpd_rd_resp_r, dpd_wr_req_s, dpd_wr_resp_r,
            tmp_rd_req_s, tmp_rd_resp_r, tmp_wr_req_s, tmp_wr_resp_r,
            tmp2_rd_req_s, tmp2_rd_resp_r, tmp2_wr_req_s, tmp2_wr_resp_r,

            fse_wr_req0_s, fse_wr_resp0_r,
            shift_buffer_ctrl0_s, shift_buffer_data0_r,
        );

        let (rle_lookup_req_s, rle_lookup_req_r) = chan<LookupDecoderReq, CHANNEL_DEPTH>("rle_lookup_req");
        let (rle_lookup_resp_s, rle_lookup_resp_r) = chan<LookupDecoderResp, CHANNEL_DEPTH>("rle_lookup_resp");

        spawn rle_lookup_dec::RleLookupDecoder<
            AXI_DATA_W, FSE_RAM_DATA_W, FSE_RAM_ADDR_W, FSE_RAM_NUM_PARTITIONS,
        >(
            rle_lookup_req_r, rle_lookup_resp_s,
            fse_wr_req1_s, fse_wr_resp1_r,
            shift_buffer_ctrl1_s, shift_buffer_data1_r,
        );

        (
            fse_lookup_dec_req_r, fse_lookup_dec_resp_s,
            comp_lookup_req_s, comp_lookup_resp_r,
            rle_lookup_req_s, rle_lookup_resp_r,

            shift_buffer_sel_req_s, shift_buffer_sel_resp_r,
            fse_ram_sel_req_s,

            fse_rd_req0_s, fse_rd_resp0_r,
            fse_rd_req1_s, fse_rd_resp1_r,

            fse_rd_req_r, fse_rd_resp_s,
        )
    }

    next(state: ()) {
        let tok0 = join();

        let (tok1, req) = recv(tok0, fse_lookup_dec_req_r);

        let sel = (req.is_rle == true);
        let tok2_0 = send(tok1, shift_buffer_sel_req_s, sel);
        let (tok3_0, _) = recv(tok2_0, shift_buffer_sel_resp_r);

        let tok2_1 = send(tok1, fse_ram_sel_req_s, sel);
        // let (tok, _) = recv(tok, fse_ram_sel_resp_r);

        let tok3 = join(tok2_1, tok3_0);

        let tok4_0 = send_if(tok3, rle_lookup_req_s, req.is_rle, LookupDecoderReq {});
        let (tok5_0, rle_lookup_resp) = recv_if(tok4_0, rle_lookup_resp_r, req.is_rle, zero!<LookupDecoderResp>());

        let tok4_1 = send_if(tok3, comp_lookup_req_s, !req.is_rle, LookupDecoderReq {});
        let (tok5_1, comp_lookup_resp) = recv_if(tok4_1, comp_lookup_resp_r, !req.is_rle, zero!<LookupDecoderResp>());

        let tok5 = join(tok5_0, tok5_1);

        let resp = if req.is_rle { rle_lookup_resp } else { comp_lookup_resp };
        let tok6 = send(tok5, fse_lookup_dec_resp_s, resp);

        // unused channels
        send_if(tok0, fse_rd_req0_s, false, zero!<FseRamRdReq>());
        recv_if(tok0, fse_rd_resp0_r, false, zero!<FseRamRdResp>());

        send_if(tok0, fse_rd_req1_s, false, zero!<FseRamRdReq>());
        recv_if(tok0, fse_rd_resp1_r, false, zero!<FseRamRdResp>());

        send_if(tok0, fse_rd_resp_s, false, zero!<FseRamRdResp>());
        recv_if(tok0, fse_rd_req_r, false, zero!<FseRamRdReq>());
    }
}


const TEST_AXI_DATA_WIDTH = u32:64;
const TEST_AXI_ADDR_WIDTH = u32:32;
const TEST_AXI_ID_WIDTH = u32:8;
const TEST_AXI_DEST_WIDTH = u32:8;
const TEST_SB_LENGTH_WIDTH = refilling_shift_buffer::length_width(TEST_AXI_DATA_WIDTH);

const TEST_CASE_RAM_DATA_WIDTH = u32:64;
const TEST_CASE_RAM_SIZE = u32:256;
const TEST_CASE_RAM_ADDR_WIDTH = std::clog2(TEST_CASE_RAM_SIZE);
const TEST_CASE_RAM_WORD_PARTITION_SIZE = TEST_CASE_RAM_DATA_WIDTH;
const TEST_CASE_RAM_NUM_PARTITIONS = ram::num_partitions(
    TEST_CASE_RAM_WORD_PARTITION_SIZE, TEST_CASE_RAM_DATA_WIDTH);
const TEST_CASE_RAM_BASE_ADDR = u32:0;

const TEST_DPD_RAM_DATA_WIDTH = u32:16;
const TEST_DPD_RAM_SIZE = u32:256;
const TEST_DPD_RAM_ADDR_WIDTH = std::clog2(TEST_DPD_RAM_SIZE);
const TEST_DPD_RAM_WORD_PARTITION_SIZE = TEST_DPD_RAM_DATA_WIDTH;
const TEST_DPD_RAM_NUM_PARTITIONS = ram::num_partitions(
    TEST_DPD_RAM_WORD_PARTITION_SIZE, TEST_DPD_RAM_DATA_WIDTH);

const TEST_FSE_RAM_DATA_WIDTH = u32:32;
const TEST_FSE_RAM_SIZE = u32:1 << common::FSE_MAX_ACCURACY_LOG;
const TEST_FSE_RAM_ADDR_WIDTH = std::clog2(TEST_FSE_RAM_SIZE);
const TEST_FSE_RAM_WORD_PARTITION_SIZE = TEST_FSE_RAM_DATA_WIDTH;
const TEST_FSE_RAM_NUM_PARTITIONS = ram::num_partitions(
    TEST_FSE_RAM_WORD_PARTITION_SIZE, TEST_FSE_RAM_DATA_WIDTH);

const TEST_TMP_RAM_DATA_WIDTH = u32:16;
const TEST_TMP_RAM_SIZE = u32:256;
const TEST_TMP_RAM_ADDR_WIDTH = std::clog2(TEST_TMP_RAM_SIZE);
const TEST_TMP_RAM_WORD_PARTITION_SIZE = TEST_TMP_RAM_DATA_WIDTH;
const TEST_TMP_RAM_NUM_PARTITIONS = ram::num_partitions(
    TEST_TMP_RAM_WORD_PARTITION_SIZE, TEST_TMP_RAM_DATA_WIDTH);

const TEST_TMP2_RAM_DATA_WIDTH = u32:8;
const TEST_TMP2_RAM_SIZE = u32:512;
const TEST_TMP2_RAM_ADDR_WIDTH = std::clog2(TEST_TMP2_RAM_SIZE);
const TEST_TMP2_RAM_WORD_PARTITION_SIZE = TEST_TMP2_RAM_DATA_WIDTH;
const TEST_TMP2_RAM_NUM_PARTITIONS = ram::num_partitions(
    TEST_TMP2_RAM_WORD_PARTITION_SIZE, TEST_TMP2_RAM_DATA_WIDTH);

const TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR = ram::SimultaneousReadWriteBehavior::READ_BEFORE_WRITE;
const TEST_RAM_INITIALIZED = true;

type FseTableRecord = common::FseTableRecord;

const COMP_LOOKUP_DECODER_TESTCASES: (u64[64], FseTableRecord[TEST_FSE_RAM_SIZE], FseLookupDecoderReq, FseLookupDecoderResp)[5] = [
    // RLE
    (
        u64[64]:[u64:0xA, u64:0, ...],
        FseTableRecord[TEST_FSE_RAM_SIZE]:[
            FseTableRecord { symbol: u8:0xa, num_of_bits: u8:0x0, base: u16:0x0 },
            zero!<FseTableRecord>(), ...
        ],
        FseLookupDecoderReq { is_rle: true },
        FseLookupDecoderResp { status: FseLookupDecoderStatus::OK, accuracy_log: AccuracyLog:0 }
    ),
    (
        u64[64]:[u64:0x2, u64:0, ...],
        FseTableRecord[TEST_FSE_RAM_SIZE]:[
            FseTableRecord { symbol: u8:0x2, num_of_bits: u8:0x0, base: u16:0x0 },
            zero!<FseTableRecord>(), ...
        ],
        FseLookupDecoderReq { is_rle: true },
        FseLookupDecoderResp { status: FseLookupDecoderStatus::OK, accuracy_log: AccuracyLog:0 }
    ),
    (
        u64[64]:[u64:0x7, u64:0, ...],
        FseTableRecord[TEST_FSE_RAM_SIZE]:[
            FseTableRecord { symbol: u8:0x7, num_of_bits: u8:0x0, base: u16:0x0 },
            zero!<FseTableRecord>(), ...
        ],
        FseLookupDecoderReq { is_rle: true },
        FseLookupDecoderResp { status: FseLookupDecoderStatus::OK, accuracy_log: AccuracyLog:0 }
    ),

    // COMPRESSED
    (
        u64[64]:[u64:0x72AAAAABBB1D25C0, u64:0, ...],
        FseTableRecord[TEST_FSE_RAM_SIZE]:[
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x1, base: u16:0x16 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x1, base: u16:0x18 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x1, base: u16:0x1a },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x1, base: u16:0x1c },
            FseTableRecord { symbol: u8:0x2, num_of_bits: u8:0x5, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x1, base: u16:0x1e },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x0, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x0, base: u16:0x1 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x0, base: u16:0x2 },
            FseTableRecord { symbol: u8:0x5, num_of_bits: u8:0x5, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x0, base: u16:0x3 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x0, base: u16:0x4 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x0, base: u16:0x5 },
            FseTableRecord { symbol: u8:0x1, num_of_bits: u8:0x5, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x0, base: u16:0x6 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x0, base: u16:0x7 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x0, base: u16:0x8 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x0, base: u16:0x9 },
            FseTableRecord { symbol: u8:0x4, num_of_bits: u8:0x5, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x0, base: u16:0xa },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x0, base: u16:0xb },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x0, base: u16:0xc },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x0, base: u16:0xd },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x0, base: u16:0xe },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x0, base: u16:0xf },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x0, base: u16:0x10 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x0, base: u16:0x11 },
            FseTableRecord { symbol: u8:0x3, num_of_bits: u8:0x5, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x0, base: u16:0x12 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x0, base: u16:0x13 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x0, base: u16:0x14 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x0, base: u16:0x15 },
            zero!<FseTableRecord>(), ...
        ],
        FseLookupDecoderReq { is_rle: false },
        FseLookupDecoderResp { status: FseLookupDecoderStatus::OK, accuracy_log: AccuracyLog:5 }
    ),
    (
        u64[64]:[u64:0x41081C158003A5D0, u64:0, ...],
        FseTableRecord[TEST_FSE_RAM_SIZE]:[
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x1, base: u16:0x18 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x1, base: u16:0x1a },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x1, base: u16:0x1c },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x1, base: u16:0x1e },
            FseTableRecord { symbol: u8:0x1, num_of_bits: u8:0x5, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x0, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x0, base: u16:0x1 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x0, base: u16:0x2 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x0, base: u16:0x3 },
            FseTableRecord { symbol: u8:0x4, num_of_bits: u8:0x5, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x0, base: u16:0x4 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x0, base: u16:0x5 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x0, base: u16:0x6 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x0, base: u16:0x7 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x0, base: u16:0x8 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x0, base: u16:0x9 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x0, base: u16:0xa },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x0, base: u16:0xb },
            FseTableRecord { symbol: u8:0x3, num_of_bits: u8:0x5, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x0, base: u16:0xc },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x0, base: u16:0xd },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x0, base: u16:0xe },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x0, base: u16:0xf },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x0, base: u16:0x10 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x0, base: u16:0x11 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x0, base: u16:0x12 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x0, base: u16:0x13 },
            FseTableRecord { symbol: u8:0x2, num_of_bits: u8:0x5, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x0, base: u16:0x14 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x0, base: u16:0x15 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x0, base: u16:0x16 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x0, base: u16:0x17 },
            zero!<FseTableRecord>(), ...
        ],
        FseLookupDecoderReq { is_rle: false },
        FseLookupDecoderResp { status: FseLookupDecoderStatus::OK, accuracy_log: AccuracyLog:5 }
    ),
];

#[test_proc]
proc FseLookupDecoderTest {
    type Req = FseLookupDecoderReq;
    type Resp = FseLookupDecoderResp;

    type MemReaderReq  = mem_reader::MemReaderReq<TEST_AXI_ADDR_WIDTH>;
    type MemReaderResp = mem_reader::MemReaderResp<TEST_AXI_DATA_WIDTH, TEST_AXI_ADDR_WIDTH>;

    type DpdRamWrReq = ram::WriteReq<TEST_DPD_RAM_ADDR_WIDTH, TEST_DPD_RAM_DATA_WIDTH, TEST_DPD_RAM_NUM_PARTITIONS>;
    type DpdRamWrResp = ram::WriteResp;
    type DpdRamRdReq = ram::ReadReq<TEST_DPD_RAM_ADDR_WIDTH, TEST_DPD_RAM_NUM_PARTITIONS>;
    type DpdRamRdResp = ram::ReadResp<TEST_DPD_RAM_DATA_WIDTH>;

    type FseRamRdReq = ram::ReadReq<TEST_FSE_RAM_ADDR_WIDTH, TEST_FSE_RAM_NUM_PARTITIONS>;
    type FseRamRdResp = ram::ReadResp<TEST_FSE_RAM_DATA_WIDTH>;
    type FseRamWrReq = ram::WriteReq<TEST_FSE_RAM_ADDR_WIDTH, TEST_FSE_RAM_DATA_WIDTH, TEST_FSE_RAM_NUM_PARTITIONS>;
    type FseRamWrResp = ram::WriteResp;

    type TmpRamRdReq = ram::ReadReq<TEST_TMP_RAM_ADDR_WIDTH, TEST_TMP_RAM_NUM_PARTITIONS>;
    type TmpRamRdResp = ram::ReadResp<TEST_TMP_RAM_DATA_WIDTH>;
    type TmpRamWrReq = ram::WriteReq<TEST_TMP_RAM_ADDR_WIDTH, TEST_TMP_RAM_DATA_WIDTH, TEST_TMP_RAM_NUM_PARTITIONS>;
    type TmpRamWrResp = ram::WriteResp;

    type Tmp2RamRdReq = ram::ReadReq<TEST_TMP2_RAM_ADDR_WIDTH, TEST_TMP2_RAM_NUM_PARTITIONS>;
    type Tmp2RamRdResp = ram::ReadResp<TEST_TMP2_RAM_DATA_WIDTH>;
    type Tmp2RamWrReq = ram::WriteReq<TEST_TMP2_RAM_ADDR_WIDTH, TEST_TMP2_RAM_DATA_WIDTH, TEST_TMP2_RAM_NUM_PARTITIONS>;
    type Tmp2RamWrResp = ram::WriteResp;

    type TestcaseRamRdReq = ram::ReadReq<TEST_CASE_RAM_ADDR_WIDTH, TEST_CASE_RAM_NUM_PARTITIONS>;
    type TestcaseRamRdResp = ram::ReadResp<TEST_CASE_RAM_DATA_WIDTH>;
    type TestcaseRamWrReq = ram::WriteReq<TEST_CASE_RAM_ADDR_WIDTH, TEST_CASE_RAM_DATA_WIDTH, TEST_CASE_RAM_NUM_PARTITIONS>;
    type TestcaseRamWrResp = ram::WriteResp;

    type RefillStartReq = refilling_shift_buffer::RefillStart<TEST_AXI_ADDR_WIDTH>;
    type SBOutput = refilling_shift_buffer::RefillingShiftBufferOutput<TEST_AXI_DATA_WIDTH, TEST_SB_LENGTH_WIDTH>;
    type SBCtrl = refilling_shift_buffer::RefillingShiftBufferCtrl<TEST_SB_LENGTH_WIDTH>;

    type AxiR = axi::AxiR<TEST_AXI_DATA_WIDTH, TEST_AXI_ID_WIDTH>;
    type AxiAr = axi::AxiAr<TEST_AXI_ADDR_WIDTH, TEST_AXI_ID_WIDTH>;

    terminator: chan<bool> out;
    req_s: chan<Req> out;
    resp_r: chan<Resp> in;
    fse_rd_req_s: chan<FseRamRdReq> out;
    fse_rd_resp_r: chan<FseRamRdResp> in;
    fse_wr_req_s: chan<FseRamWrReq> out;
    fse_wr_resp_r: chan<FseRamWrResp> in;
    testcase_wr_req_s: chan<TestcaseRamWrReq> out;
    testcase_wr_resp_r: chan<TestcaseRamWrResp> in;
    refill_req_s: chan<RefillStartReq> out;
    stop_flush_req_s: chan<()> out;
    flushing_done_r: chan<()> in;

    config(terminator: chan<bool> out) {
        let (req_s, req_r) = chan<Req>("req");
        let (resp_s, resp_r) = chan<Resp>("resp");
        let (mem_rd_req_s, mem_rd_req_r) = chan<MemReaderReq>("mem_rd_req");
        let (mem_rd_resp_s, mem_rd_resp_r) = chan<MemReaderResp>("mem_rd_resp");

        let (dpd_rd_req_s, dpd_rd_req_r) = chan<DpdRamRdReq>("dpd_rd_req");
        let (dpd_rd_resp_s, dpd_rd_resp_r) = chan<DpdRamRdResp>("dpd_rd_resp");
        let (dpd_wr_req_s, dpd_wr_req_r) = chan<DpdRamWrReq>("dpd_wr_req");
        let (dpd_wr_resp_s, dpd_wr_resp_r) = chan<DpdRamWrResp>("dpd_wr_resp");

        let (tmp_rd_req_s, tmp_rd_req_r) = chan<TmpRamRdReq>("tmp_rd_req");
        let (tmp_rd_resp_s, tmp_rd_resp_r) = chan<TmpRamRdResp>("tmp_rd_resp");
        let (tmp_wr_req_s, tmp_wr_req_r) = chan<TmpRamWrReq>("tmp_wr_req");
        let (tmp_wr_resp_s, tmp_wr_resp_r) = chan<TmpRamWrResp>("tmp_wr_resp");

        let (tmp2_rd_req_s, tmp2_rd_req_r) = chan<Tmp2RamRdReq>("tmp2_rd_req");
        let (tmp2_rd_resp_s, tmp2_rd_resp_r) = chan<Tmp2RamRdResp>("tmp2_rd_resp");
        let (tmp2_wr_req_s, tmp2_wr_req_r) = chan<Tmp2RamWrReq>("tmp2_wr_req");
        let (tmp2_wr_resp_s, tmp2_wr_resp_r) = chan<Tmp2RamWrResp>("tmp2_wr_resp");

        let (fse_rd_req_s, fse_rd_req_r) = chan<FseRamRdReq>("fse_rd_req");
        let (fse_rd_resp_s, fse_rd_resp_r) = chan<FseRamRdResp>("fse_rd_resp");
        let (fse_wr_req_s, fse_wr_req_r) = chan<FseRamWrReq>("fse_wr_req");
        let (fse_wr_resp_s, fse_wr_resp_r) = chan<FseRamWrResp>("fse_wr_resp");

        let (testcase_rd_req_s, testcase_rd_req_r) = chan<TestcaseRamRdReq>("testcase_rd_req");
        let (testcase_rd_resp_s, testcase_rd_resp_r) = chan<TestcaseRamRdResp>("testcase_rd_resp");
        let (testcase_wr_req_s, testcase_wr_req_r) = chan<TestcaseRamWrReq>("testcase_wr_req");
        let (testcase_wr_resp_s, testcase_wr_resp_r) = chan<TestcaseRamWrResp>("testcase_wr_resp");

        let (buffer_ctrl_s, buffer_ctrl_r) = chan<SBCtrl>("buffer_ctrl");
        let (buffer_data_out_s, buffer_data_out_r) = chan<SBOutput>("buffer_data_out");

        spawn FseLookupDecoder<
            TEST_AXI_DATA_WIDTH,
            TEST_DPD_RAM_DATA_WIDTH, TEST_DPD_RAM_ADDR_WIDTH, TEST_DPD_RAM_NUM_PARTITIONS,
            TEST_TMP_RAM_DATA_WIDTH, TEST_TMP_RAM_ADDR_WIDTH, TEST_TMP_RAM_NUM_PARTITIONS,
            TEST_TMP2_RAM_DATA_WIDTH, TEST_TMP2_RAM_ADDR_WIDTH, TEST_TMP2_RAM_NUM_PARTITIONS,
            TEST_FSE_RAM_DATA_WIDTH, TEST_FSE_RAM_ADDR_WIDTH, TEST_FSE_RAM_NUM_PARTITIONS,
        >(
            req_r, resp_s,
            dpd_rd_req_s, dpd_rd_resp_r, dpd_wr_req_s, dpd_wr_resp_r,
            tmp_rd_req_s, tmp_rd_resp_r, tmp_wr_req_s, tmp_wr_resp_r,
            tmp2_rd_req_s, tmp2_rd_resp_r, tmp2_wr_req_s, tmp2_wr_resp_r,
            fse_wr_req_s, fse_wr_resp_r,
            buffer_ctrl_s, buffer_data_out_r,
        );

        spawn ram::RamModel<
            TEST_DPD_RAM_DATA_WIDTH, TEST_DPD_RAM_SIZE, TEST_DPD_RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED,
        >(dpd_rd_req_r, dpd_rd_resp_s, dpd_wr_req_r, dpd_wr_resp_s);

        spawn ram::RamModel<
            TEST_FSE_RAM_DATA_WIDTH, TEST_FSE_RAM_SIZE, TEST_FSE_RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED,
        >(fse_rd_req_r, fse_rd_resp_s, fse_wr_req_r, fse_wr_resp_s);

        spawn ram::RamModel<
            TEST_TMP_RAM_DATA_WIDTH, TEST_TMP_RAM_SIZE, TEST_TMP_RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED,
        >(tmp_rd_req_r, tmp_rd_resp_s, tmp_wr_req_r, tmp_wr_resp_s);

        spawn ram::RamModel<
            TEST_TMP2_RAM_DATA_WIDTH, TEST_TMP2_RAM_SIZE, TEST_TMP2_RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED,
        >(tmp2_rd_req_r, tmp2_rd_resp_s, tmp2_wr_req_r, tmp2_wr_resp_s);

        spawn ram::RamModel<
            TEST_CASE_RAM_DATA_WIDTH, TEST_CASE_RAM_SIZE, TEST_CASE_RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED,
        >(testcase_rd_req_r, testcase_rd_resp_s, testcase_wr_req_r, testcase_wr_resp_s);

        let (testcase_axi_r_s, testcase_axi_r_r) = chan<AxiR>("testcase_axi_r");
        let (testcase_axi_ar_s, testcase_axi_ar_r) = chan<AxiAr>("testcase_axi_ar");

        spawn axi_ram::AxiRamReader<
            TEST_AXI_ADDR_WIDTH, TEST_AXI_DATA_WIDTH, TEST_AXI_DEST_WIDTH, TEST_AXI_ID_WIDTH,
            TEST_CASE_RAM_SIZE, TEST_CASE_RAM_BASE_ADDR, TEST_CASE_RAM_DATA_WIDTH,
            TEST_CASE_RAM_ADDR_WIDTH, TEST_CASE_RAM_NUM_PARTITIONS,
        >(testcase_axi_ar_r, testcase_axi_r_s, testcase_rd_req_s, testcase_rd_resp_r);

        spawn mem_reader::MemReader<
            TEST_AXI_DATA_WIDTH, TEST_AXI_ADDR_WIDTH, TEST_AXI_DEST_WIDTH, TEST_AXI_ID_WIDTH
        >(mem_rd_req_r, mem_rd_resp_s, testcase_axi_ar_s, testcase_axi_r_r);

        let (refill_req_s, refill_req_r) = chan<RefillStartReq>("start_req");
        let (stop_flush_req_s, stop_flush_req_r) = chan<()>("stop_flush_req");
        let (flushing_done_s, flushing_done_r) = chan<()>("flushing_done");

        spawn refilling_shift_buffer::RefillingShiftBuffer<TEST_AXI_DATA_WIDTH, TEST_AXI_ADDR_WIDTH>(
            mem_rd_req_s, mem_rd_resp_r,
            refill_req_r, stop_flush_req_r,
            buffer_ctrl_r, buffer_data_out_s,
            flushing_done_s,
        );

        (
            terminator, req_s, resp_r, fse_rd_req_s, fse_rd_resp_r,
            fse_wr_req_s, fse_wr_resp_r, testcase_wr_req_s, testcase_wr_resp_r,
            refill_req_s, stop_flush_req_s, flushing_done_r,
        )
    }

    init {}

    next(_: ()) {
        let tok = join();

        let tok = unroll_for!(test_i, tok): (u32, token) in range(u32:0, array_size(COMP_LOOKUP_DECODER_TESTCASES)) {
            let (input, output, req, exp_resp) = COMP_LOOKUP_DECODER_TESTCASES[test_i];

            trace_fmt!("Loading testcase {:x}", test_i);

            let tok = for ((i, input_data), tok): ((u32, u64), token) in enumerate(input) {
                let req = TestcaseRamWrReq {
                    addr: i as uN[TEST_CASE_RAM_ADDR_WIDTH],
                    data: input_data as uN[TEST_CASE_RAM_DATA_WIDTH],
                    mask: uN[TEST_CASE_RAM_NUM_PARTITIONS]:0x1
                };
                let tok = send(tok, testcase_wr_req_s, req);
                let (tok, _) = recv(tok, testcase_wr_resp_r);
                tok
            }(tok);

            trace_fmt!("Running FSE lookup decoder on testcase {:x}", test_i);
            let tok = send(tok, refill_req_s, RefillStartReq {
                start_addr: uN[TEST_AXI_ADDR_WIDTH]:0x0
            });

            let tok = send(tok, req_s, req);
            let (tok, resp) = recv(tok, resp_r);
            assert_eq(resp, exp_resp);

            let tok = for ((i, output_data), tok): ((u32, FseTableRecord), token) in enumerate(output) {
                let req = FseRamRdReq {
                    addr: i as uN[TEST_FSE_RAM_ADDR_WIDTH],
                    mask: std::unsigned_max_value<TEST_FSE_RAM_NUM_PARTITIONS>(),
                };
                let tok = send(tok, fse_rd_req_s, req);
                let (tok, resp) = recv(tok, fse_rd_resp_r);
                assert_eq(fse_table_creator::bits_to_fse_record(resp.data), output_data);

                // erase output for next test to start with clean memory
                let clear_req = FseRamWrReq {
                    addr: i as uN[TEST_FSE_RAM_ADDR_WIDTH],
                    mask: std::unsigned_max_value<TEST_FSE_RAM_NUM_PARTITIONS>(),
                    data: uN[TEST_FSE_RAM_DATA_WIDTH]:0x0,
                };
                let tok = send(tok, fse_wr_req_s, clear_req);
                let (tok, _) = recv(tok, fse_wr_resp_r);
                tok
            }(tok);

            let tok = send(tok, stop_flush_req_s, ());
            let (tok, ()) = recv(tok, flushing_done_r);

            tok
        }(tok);

        send(tok, terminator, true);
    }
}
