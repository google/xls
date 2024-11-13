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

type AccuracyLog = common::FseAccuracyLog;

pub enum FseLookupDecoderStatus: u1 {
    OK = 0,
    ERROR = 1,
}

pub struct FseLookupDecoderReq<AXI_ADDR_W: u32> {
    addr: uN[AXI_ADDR_W]
}

pub struct FseLookupDecoderResp {
    status: FseLookupDecoderStatus,
    accuracy_log: common::FseAccuracyLog
}

pub proc FseLookupDecoder<
    AXI_DATA_W: u32, AXI_ADDR_W: u32,
    DPD_RAM_DATA_W: u32, DPD_RAM_ADDR_W: u32, DPD_RAM_NUM_PARTITIONS: u32,
    TMP_RAM_DATA_W: u32, TMP_RAM_ADDR_W: u32, TMP_RAM_NUM_PARTITIONS: u32,
    FSE_RAM_DATA_W: u32, FSE_RAM_ADDR_W: u32, FSE_RAM_NUM_PARTITIONS: u32,
    SB_LENGTH_W: u32 = {refilling_shift_buffer::length_width(AXI_DATA_W)},
> {
    type Req = FseLookupDecoderReq<AXI_ADDR_W>;
    type Resp = FseLookupDecoderResp;
    type Status = FseLookupDecoderStatus;

    type FseTableStart = fse_table_creator::FseStartMsg;

    type MemReaderReq  = mem_reader::MemReaderReq<AXI_ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<AXI_DATA_W, AXI_ADDR_W>;
    type MemReaderStatus = mem_reader::MemReaderStatus;

    type DpdRamWrReq = ram::WriteReq<DPD_RAM_ADDR_W, DPD_RAM_DATA_W, DPD_RAM_NUM_PARTITIONS>;
    type DpdRamWrResp = ram::WriteResp;
    type DpdRamRdReq = ram::ReadReq<DPD_RAM_ADDR_W, DPD_RAM_NUM_PARTITIONS>;
    type DpdRamRdResp = ram::ReadResp<DPD_RAM_DATA_W>;

    type FseRamRdReq = ram::ReadReq<FSE_RAM_ADDR_W, FSE_RAM_NUM_PARTITIONS>;
    type FseRamRdResp = ram::ReadResp<FSE_RAM_DATA_W>;
    type FseRamWrReq = ram::WriteReq<FSE_RAM_ADDR_W, FSE_RAM_DATA_W, FSE_RAM_NUM_PARTITIONS>;
    type FseRamWrResp = ram::WriteResp;

    type TmpRamRdReq = ram::ReadReq<TMP_RAM_ADDR_W, TMP_RAM_NUM_PARTITIONS>;
    type TmpRamRdResp = ram::ReadResp<TMP_RAM_DATA_W>;
    type TmpRamWrReq = ram::WriteReq<TMP_RAM_ADDR_W, TMP_RAM_DATA_W, TMP_RAM_NUM_PARTITIONS>;
    type TmpRamWrResp = ram::WriteResp;

    type RefillerStartReq = refilling_shift_buffer::RefillStart<AXI_ADDR_W>;
    type RefillerError = refilling_shift_buffer::RefillError;
    type SBOutput = refilling_shift_buffer::RefillingShiftBufferOutput<AXI_DATA_W, SB_LENGTH_W>;
    type SBCtrl = refilling_shift_buffer::RefillingShiftBufferCtrl<SB_LENGTH_W>;

    type FsePFDecReq = fse_proba_freq_dec::FseProbaFreqDecoderReq;
    type FsePFDecResp = fse_proba_freq_dec::FseProbaFreqDecoderResp;
    type FsePFDecStatus = fse_proba_freq_dec::FseProbaFreqDecoderStatus;

    req_r: chan<Req> in;
    resp_s: chan<Resp> out;

    start_req_s: chan<RefillerStartReq> out;
    stop_flush_req_s: chan<()> out;
    flushing_done_r: chan<()> in;

    fse_pf_dec_req_s: chan<FsePFDecReq> out;
    fse_pf_dec_resp_r: chan<FsePFDecResp> in;
    fse_table_start_s: chan<FseTableStart> out;
    fse_table_finish_r: chan<()> in;

    init {}

    config(
        req_r: chan<Req> in,
        resp_s: chan<Resp> out,

        mem_rd_req_s: chan<MemReaderReq> out,
        mem_rd_resp_r: chan<MemReaderResp> in,

        dpd_rd_req_s: chan<DpdRamRdReq> out,
        dpd_rd_resp_r: chan<DpdRamRdResp> in,
        dpd_wr_req_s: chan<DpdRamWrReq> out,
        dpd_wr_resp_r: chan<DpdRamWrResp> in,

        tmp_rd_req_s: chan<TmpRamRdReq> out,
        tmp_rd_resp_r: chan<TmpRamRdResp> in,
        tmp_wr_req_s: chan<TmpRamWrReq> out,
        tmp_wr_resp_r: chan<TmpRamWrResp> in,

        fse_rd_req_s: chan<FseRamRdReq> out,
        fse_rd_resp_r: chan<FseRamRdResp> in,
        fse_wr_req_s: chan<FseRamWrReq> out,
        fse_wr_resp_r: chan<FseRamWrResp> in,
    ) {
        const CHANNEL_DEPTH = u32:1;

        let (fse_table_start_s, fse_table_start_r) = chan<FseTableStart, CHANNEL_DEPTH>("fse_table_start");
        let (fse_table_finish_s, fse_table_finish_r) = chan<(), CHANNEL_DEPTH>("fse_table_finish");

        spawn fse_table_creator::FseTableCreator<
            DPD_RAM_DATA_W, DPD_RAM_ADDR_W, DPD_RAM_NUM_PARTITIONS,
            FSE_RAM_DATA_W, FSE_RAM_ADDR_W, FSE_RAM_NUM_PARTITIONS,
            TMP_RAM_DATA_W, TMP_RAM_ADDR_W, TMP_RAM_NUM_PARTITIONS,
        >(
            fse_table_start_r, fse_table_finish_s,
            dpd_rd_req_s, dpd_rd_resp_r,
            fse_rd_req_s, fse_rd_resp_r, fse_wr_req_s, fse_wr_resp_r,
            tmp_rd_req_s, tmp_rd_resp_r, tmp_wr_req_s, tmp_wr_resp_r,
        );

        let (start_req_s, start_req_r) = chan<RefillerStartReq, CHANNEL_DEPTH>("start_req");
        let (stop_flush_req_s, stop_flush_req_r) = chan<(), CHANNEL_DEPTH>("stop_flush_req");
        let (buffer_ctrl_s, buffer_ctrl_r) = chan<SBCtrl, CHANNEL_DEPTH>("buffer_ctrl");
        let (buffer_data_out_s, buffer_data_out_r) = chan<SBOutput, CHANNEL_DEPTH>("buffer_data_out");
        let (flushing_done_s, flushing_done_r) = chan<(), CHANNEL_DEPTH>("flushing_done");

        spawn refilling_shift_buffer::RefillingShiftBuffer<AXI_DATA_W, AXI_ADDR_W>(
            mem_rd_req_s,
            mem_rd_resp_r,
            start_req_r,
            stop_flush_req_r,
            buffer_ctrl_r,
            buffer_data_out_s,
            flushing_done_s,
        );

        let (fse_pf_dec_req_s, fse_pf_dec_req_r) = chan<FsePFDecReq, CHANNEL_DEPTH>("fse_pf_dec_req");
        let (fse_pf_dec_resp_s, fse_pf_dec_resp_r) = chan<FsePFDecResp, CHANNEL_DEPTH>("fse_pf_dec_resp");

        spawn fse_proba_freq_dec::FseProbaFreqDecoder<
            DPD_RAM_DATA_W, DPD_RAM_ADDR_W, DPD_RAM_NUM_PARTITIONS,
        >(
            fse_pf_dec_req_r, fse_pf_dec_resp_s,
            buffer_ctrl_s, buffer_data_out_r,
            dpd_wr_req_s, dpd_wr_resp_r,
        );

        (
            req_r, resp_s,
            start_req_s,
            stop_flush_req_s,
            flushing_done_r,
            fse_pf_dec_req_s, fse_pf_dec_resp_r,
            fse_table_start_s, fse_table_finish_r,
        )
    }

    next(state: ()) {
        let tok = join();
        let (tok, start_req) = recv(tok, req_r);

        // start refilling shift buffer
        let tok_dec_pf1 = send(tok, start_req_s, RefillerStartReq {
            start_addr: start_req.addr
        });
        // start FSE probability frequency decoder
        let tok_dec_pf2 = send(tok, fse_pf_dec_req_s, FsePFDecReq {});

        // wait for completion from FSE probability frequency decoder
        let tok = join(tok_dec_pf1, tok_dec_pf2);
        let (tok_dec_resp, pf_dec_res) = recv(tok, fse_pf_dec_resp_r);

        // flush refilling shift buffer (regardless of any errors)
        let tok_flush = send(tok_dec_resp, stop_flush_req_s, ());
        recv(tok_flush, flushing_done_r);

        let pf_dec_ok = pf_dec_res.status == FsePFDecStatus::OK;
        // run FSE Table creation conditional or previous processing succeeding
        let tok = send_if(tok_dec_resp, fse_table_start_s, pf_dec_ok, FseTableStart {
            num_symbs: pf_dec_res.symbol_count,
            accuracy_log: pf_dec_res.accuracy_log,
        });
        // wait for completion from FSE table creator
        let (tok, ()) = recv_if(tok, fse_table_finish_r, pf_dec_ok, ());

        let resp = if pf_dec_ok {
            Resp { status: Status::OK, accuracy_log: pf_dec_res.accuracy_log }
        } else {
            Resp { status: Status::ERROR, ..zero!<Resp>() }
        };
        send(tok, resp_s, resp);
    }
}


const TEST_AXI_DATA_WIDTH = u32:64;
const TEST_AXI_ADDR_WIDTH = u32:32;
const TEST_AXI_ID_WIDTH = u32:8;
const TEST_AXI_DEST_WIDTH = u32:8;

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
const TEST_FSE_RAM_SIZE = u32:256;
const TEST_FSE_RAM_ADDR_WIDTH = std::clog2(TEST_FSE_RAM_SIZE);
const TEST_FSE_RAM_WORD_PARTITION_SIZE = TEST_FSE_RAM_DATA_WIDTH / u32:3;
const TEST_FSE_RAM_NUM_PARTITIONS = ram::num_partitions(
    TEST_FSE_RAM_WORD_PARTITION_SIZE, TEST_FSE_RAM_DATA_WIDTH);

const TEST_TMP_RAM_DATA_WIDTH = u32:16;
const TEST_TMP_RAM_SIZE = u32:256;
const TEST_TMP_RAM_ADDR_WIDTH = std::clog2(TEST_TMP_RAM_SIZE);
const TEST_TMP_RAM_WORD_PARTITION_SIZE = TEST_TMP_RAM_DATA_WIDTH;
const TEST_TMP_RAM_NUM_PARTITIONS = ram::num_partitions(
    TEST_TMP_RAM_WORD_PARTITION_SIZE, TEST_TMP_RAM_DATA_WIDTH);

const TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR = ram::SimultaneousReadWriteBehavior::READ_BEFORE_WRITE;
const TEST_RAM_INITIALIZED = true;

type FseTableRecord = common::FseTableRecord;

const FSE_LOOKUP_DECODER_TESTCASES: (u64[64], FseTableRecord[TEST_FSE_RAM_SIZE], FseLookupDecoderResp)[10] = [
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
        FseLookupDecoderResp { status: FseLookupDecoderStatus::OK, accuracy_log: AccuracyLog:5 }
    ),
    (
        u64[64]:[u64:0x1861862062081932, u64:0xC18628A106184184, u64:0x850720FACC49238, u64:0, ...],
        FseTableRecord[TEST_FSE_RAM_SIZE]:[
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x3, base: u16:0x10 },
            FseTableRecord { symbol: u8:0x18, num_of_bits: u8:0x4, base: u16:0x30 },
            FseTableRecord { symbol: u8:0x11, num_of_bits: u8:0x5, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x3, base: u16:0x18 },
            FseTableRecord { symbol: u8:0x19, num_of_bits: u8:0x3, base: u16:0x28 },
            FseTableRecord { symbol: u8:0x16, num_of_bits: u8:0x4, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x7, num_of_bits: u8:0x6, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x1a, num_of_bits: u8:0x4, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x18, num_of_bits: u8:0x4, base: u16:0x40 },
            FseTableRecord { symbol: u8:0x10, num_of_bits: u8:0x5, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x3, base: u16:0x20 },
            FseTableRecord { symbol: u8:0x19, num_of_bits: u8:0x3, base: u16:0x30 },
            FseTableRecord { symbol: u8:0x15, num_of_bits: u8:0x5, base: u16:0x40 },
            FseTableRecord { symbol: u8:0x6, num_of_bits: u8:0x6, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x1a, num_of_bits: u8:0x4, base: u16:0x10 },
            FseTableRecord { symbol: u8:0x17, num_of_bits: u8:0x4, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x10, num_of_bits: u8:0x5, base: u16:0x20 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x3, base: u16:0x28 },
            FseTableRecord { symbol: u8:0x19, num_of_bits: u8:0x3, base: u16:0x38 },
            FseTableRecord { symbol: u8:0x15, num_of_bits: u8:0x5, base: u16:0x60 },
            FseTableRecord { symbol: u8:0x5, num_of_bits: u8:0x6, base: u16:0x40 },
            FseTableRecord { symbol: u8:0x1a, num_of_bits: u8:0x4, base: u16:0x20 },
            FseTableRecord { symbol: u8:0x17, num_of_bits: u8:0x4, base: u16:0x10 },
            FseTableRecord { symbol: u8:0xe, num_of_bits: u8:0x6, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x3, base: u16:0x30 },
            FseTableRecord { symbol: u8:0x19, num_of_bits: u8:0x3, base: u16:0x40 },
            FseTableRecord { symbol: u8:0x14, num_of_bits: u8:0x5, base: u16:0x20 },
            FseTableRecord { symbol: u8:0x3, num_of_bits: u8:0x6, base: u16:0x40 },
            FseTableRecord { symbol: u8:0x19, num_of_bits: u8:0x3, base: u16:0x48 },
            FseTableRecord { symbol: u8:0x17, num_of_bits: u8:0x4, base: u16:0x20 },
            FseTableRecord { symbol: u8:0xc, num_of_bits: u8:0x7, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x3, base: u16:0x38 },
            FseTableRecord { symbol: u8:0x18, num_of_bits: u8:0x4, base: u16:0x50 },
            FseTableRecord { symbol: u8:0x13, num_of_bits: u8:0x6, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x2, num_of_bits: u8:0x6, base: u16:0x40 },
            FseTableRecord { symbol: u8:0x19, num_of_bits: u8:0x3, base: u16:0x50 },
            FseTableRecord { symbol: u8:0x16, num_of_bits: u8:0x4, base: u16:0x10 },
            FseTableRecord { symbol: u8:0xa, num_of_bits: u8:0x7, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x3, base: u16:0x40 },
            FseTableRecord { symbol: u8:0x18, num_of_bits: u8:0x4, base: u16:0x60 },
            FseTableRecord { symbol: u8:0x12, num_of_bits: u8:0x6, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x1, num_of_bits: u8:0x6, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x19, num_of_bits: u8:0x3, base: u16:0x58 },
            FseTableRecord { symbol: u8:0x16, num_of_bits: u8:0x4, base: u16:0x20 },
            FseTableRecord { symbol: u8:0x8, num_of_bits: u8:0x6, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x1b, num_of_bits: u8:0x7, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x18, num_of_bits: u8:0x4, base: u16:0x70 },
            FseTableRecord { symbol: u8:0x11, num_of_bits: u8:0x5, base: u16:0x20 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x3, base: u16:0x48 },
            FseTableRecord { symbol: u8:0x19, num_of_bits: u8:0x3, base: u16:0x60 },
            FseTableRecord { symbol: u8:0x15, num_of_bits: u8:0x4, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x7, num_of_bits: u8:0x6, base: u16:0x40 },
            FseTableRecord { symbol: u8:0x1a, num_of_bits: u8:0x4, base: u16:0x30 },
            FseTableRecord { symbol: u8:0x18, num_of_bits: u8:0x3, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x10, num_of_bits: u8:0x5, base: u16:0x40 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x3, base: u16:0x50 },
            FseTableRecord { symbol: u8:0x19, num_of_bits: u8:0x3, base: u16:0x68 },
            FseTableRecord { symbol: u8:0x15, num_of_bits: u8:0x4, base: u16:0x10 },
            FseTableRecord { symbol: u8:0x5, num_of_bits: u8:0x5, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x1a, num_of_bits: u8:0x4, base: u16:0x40 },
            FseTableRecord { symbol: u8:0x17, num_of_bits: u8:0x4, base: u16:0x30 },
            FseTableRecord { symbol: u8:0xf, num_of_bits: u8:0x7, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x3, base: u16:0x58 },
            FseTableRecord { symbol: u8:0x19, num_of_bits: u8:0x3, base: u16:0x70 },
            FseTableRecord { symbol: u8:0x14, num_of_bits: u8:0x5, base: u16:0x40 },
            FseTableRecord { symbol: u8:0x4, num_of_bits: u8:0x6, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x19, num_of_bits: u8:0x3, base: u16:0x78 },
            FseTableRecord { symbol: u8:0x17, num_of_bits: u8:0x4, base: u16:0x40 },
            FseTableRecord { symbol: u8:0xd, num_of_bits: u8:0x6, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x3, base: u16:0x60 },
            FseTableRecord { symbol: u8:0x19, num_of_bits: u8:0x2, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x14, num_of_bits: u8:0x5, base: u16:0x60 },
            FseTableRecord { symbol: u8:0x3, num_of_bits: u8:0x5, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x19, num_of_bits: u8:0x2, base: u16:0x4 },
            FseTableRecord { symbol: u8:0x16, num_of_bits: u8:0x4, base: u16:0x30 },
            FseTableRecord { symbol: u8:0xb, num_of_bits: u8:0x6, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x3, base: u16:0x68 },
            FseTableRecord { symbol: u8:0x18, num_of_bits: u8:0x3, base: u16:0x8 },
            FseTableRecord { symbol: u8:0x13, num_of_bits: u8:0x6, base: u16:0x40 },
            FseTableRecord { symbol: u8:0x2, num_of_bits: u8:0x5, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x19, num_of_bits: u8:0x2, base: u16:0x8 },
            FseTableRecord { symbol: u8:0x16, num_of_bits: u8:0x4, base: u16:0x40 },
            FseTableRecord { symbol: u8:0x9, num_of_bits: u8:0x6, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x3, base: u16:0x70 },
            FseTableRecord { symbol: u8:0x18, num_of_bits: u8:0x3, base: u16:0x10 },
            FseTableRecord { symbol: u8:0x11, num_of_bits: u8:0x5, base: u16:0x40 },
            FseTableRecord { symbol: u8:0x1, num_of_bits: u8:0x6, base: u16:0x40 },
            FseTableRecord { symbol: u8:0x19, num_of_bits: u8:0x2, base: u16:0xc },
            FseTableRecord { symbol: u8:0x16, num_of_bits: u8:0x4, base: u16:0x50 },
            FseTableRecord { symbol: u8:0x8, num_of_bits: u8:0x6, base: u16:0x40 },
            FseTableRecord { symbol: u8:0x1a, num_of_bits: u8:0x4, base: u16:0x50 },
            FseTableRecord { symbol: u8:0x18, num_of_bits: u8:0x3, base: u16:0x18 },
            FseTableRecord { symbol: u8:0x11, num_of_bits: u8:0x5, base: u16:0x60 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x3, base: u16:0x78 },
            FseTableRecord { symbol: u8:0x19, num_of_bits: u8:0x2, base: u16:0x10 },
            FseTableRecord { symbol: u8:0x15, num_of_bits: u8:0x4, base: u16:0x20 },
            FseTableRecord { symbol: u8:0x6, num_of_bits: u8:0x6, base: u16:0x40 },
            FseTableRecord { symbol: u8:0x1a, num_of_bits: u8:0x4, base: u16:0x60 },
            FseTableRecord { symbol: u8:0x17, num_of_bits: u8:0x4, base: u16:0x50 },
            FseTableRecord { symbol: u8:0x10, num_of_bits: u8:0x5, base: u16:0x60 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x2, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x19, num_of_bits: u8:0x2, base: u16:0x14 },
            FseTableRecord { symbol: u8:0x15, num_of_bits: u8:0x4, base: u16:0x30 },
            FseTableRecord { symbol: u8:0x5, num_of_bits: u8:0x5, base: u16:0x20 },
            FseTableRecord { symbol: u8:0x1a, num_of_bits: u8:0x4, base: u16:0x70 },
            FseTableRecord { symbol: u8:0x17, num_of_bits: u8:0x4, base: u16:0x60 },
            FseTableRecord { symbol: u8:0xe, num_of_bits: u8:0x6, base: u16:0x40 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x2, base: u16:0x4 },
            FseTableRecord { symbol: u8:0x19, num_of_bits: u8:0x2, base: u16:0x18 },
            FseTableRecord { symbol: u8:0x14, num_of_bits: u8:0x4, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x4, num_of_bits: u8:0x6, base: u16:0x40 },
            FseTableRecord { symbol: u8:0x19, num_of_bits: u8:0x2, base: u16:0x1c },
            FseTableRecord { symbol: u8:0x17, num_of_bits: u8:0x4, base: u16:0x70 },
            FseTableRecord { symbol: u8:0xd, num_of_bits: u8:0x6, base: u16:0x40 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x2, base: u16:0x8 },
            FseTableRecord { symbol: u8:0x18, num_of_bits: u8:0x3, base: u16:0x20 },
            FseTableRecord { symbol: u8:0x14, num_of_bits: u8:0x4, base: u16:0x10 },
            FseTableRecord { symbol: u8:0x3, num_of_bits: u8:0x5, base: u16:0x20 },
            FseTableRecord { symbol: u8:0x19, num_of_bits: u8:0x2, base: u16:0x20 },
            FseTableRecord { symbol: u8:0x16, num_of_bits: u8:0x4, base: u16:0x60 },
            FseTableRecord { symbol: u8:0xb, num_of_bits: u8:0x6, base: u16:0x40 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x2, base: u16:0xc },
            FseTableRecord { symbol: u8:0x18, num_of_bits: u8:0x3, base: u16:0x28 },
            FseTableRecord { symbol: u8:0x12, num_of_bits: u8:0x6, base: u16:0x40 },
            FseTableRecord { symbol: u8:0x2, num_of_bits: u8:0x5, base: u16:0x20 },
            FseTableRecord { symbol: u8:0x19, num_of_bits: u8:0x2, base: u16:0x24 },
            FseTableRecord { symbol: u8:0x16, num_of_bits: u8:0x4, base: u16:0x70 },
            FseTableRecord { symbol: u8:0x9, num_of_bits: u8:0x6, base: u16:0x40 },
            zero!<FseTableRecord>(), ...
        ],
        FseLookupDecoderResp { status: FseLookupDecoderStatus::OK, accuracy_log: AccuracyLog:7 }
    ),
    (
        u64[64]:[u64:0x60C3082082085072, u64:0x1C06F8077D850F20, u64:0, ...],
        FseTableRecord[TEST_FSE_RAM_SIZE]:[
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x5, base: u16:0x40 },
            FseTableRecord { symbol: u8:0xd, num_of_bits: u8:0x3, base: u16:0x38 },
            FseTableRecord { symbol: u8:0xc, num_of_bits: u8:0x3, base: u16:0x18 },
            FseTableRecord { symbol: u8:0x4, num_of_bits: u8:0x7, base: u16:0x0 },
            FseTableRecord { symbol: u8:0xe, num_of_bits: u8:0x3, base: u16:0x70 },
            FseTableRecord { symbol: u8:0xd, num_of_bits: u8:0x3, base: u16:0x40 },
            FseTableRecord { symbol: u8:0xa, num_of_bits: u8:0x5, base: u16:0x60 },
            FseTableRecord { symbol: u8:0xf, num_of_bits: u8:0x5, base: u16:0x40 },
            FseTableRecord { symbol: u8:0xd, num_of_bits: u8:0x3, base: u16:0x48 },
            FseTableRecord { symbol: u8:0xc, num_of_bits: u8:0x3, base: u16:0x20 },
            FseTableRecord { symbol: u8:0x1, num_of_bits: u8:0x4, base: u16:0x10 },
            FseTableRecord { symbol: u8:0xe, num_of_bits: u8:0x3, base: u16:0x78 },
            FseTableRecord { symbol: u8:0xc, num_of_bits: u8:0x3, base: u16:0x28 },
            FseTableRecord { symbol: u8:0xa, num_of_bits: u8:0x4, base: u16:0x0 },
            FseTableRecord { symbol: u8:0xf, num_of_bits: u8:0x5, base: u16:0x60 },
            FseTableRecord { symbol: u8:0xd, num_of_bits: u8:0x3, base: u16:0x50 },
            FseTableRecord { symbol: u8:0xb, num_of_bits: u8:0x4, base: u16:0x60 },
            FseTableRecord { symbol: u8:0x1, num_of_bits: u8:0x4, base: u16:0x20 },
            FseTableRecord { symbol: u8:0xe, num_of_bits: u8:0x2, base: u16:0x0 },
            FseTableRecord { symbol: u8:0xc, num_of_bits: u8:0x3, base: u16:0x30 },
            FseTableRecord { symbol: u8:0x9, num_of_bits: u8:0x5, base: u16:0x20 },
            FseTableRecord { symbol: u8:0xe, num_of_bits: u8:0x2, base: u16:0x4 },
            FseTableRecord { symbol: u8:0xd, num_of_bits: u8:0x3, base: u16:0x58 },
            FseTableRecord { symbol: u8:0xb, num_of_bits: u8:0x4, base: u16:0x70 },
            FseTableRecord { symbol: u8:0x1, num_of_bits: u8:0x4, base: u16:0x30 },
            FseTableRecord { symbol: u8:0xe, num_of_bits: u8:0x2, base: u16:0x8 },
            FseTableRecord { symbol: u8:0xc, num_of_bits: u8:0x3, base: u16:0x38 },
            FseTableRecord { symbol: u8:0x9, num_of_bits: u8:0x5, base: u16:0x40 },
            FseTableRecord { symbol: u8:0xe, num_of_bits: u8:0x2, base: u16:0xc },
            FseTableRecord { symbol: u8:0xd, num_of_bits: u8:0x3, base: u16:0x60 },
            FseTableRecord { symbol: u8:0xb, num_of_bits: u8:0x3, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x5, base: u16:0x60 },
            FseTableRecord { symbol: u8:0xe, num_of_bits: u8:0x2, base: u16:0x10 },
            FseTableRecord { symbol: u8:0xc, num_of_bits: u8:0x3, base: u16:0x40 },
            FseTableRecord { symbol: u8:0x8, num_of_bits: u8:0x6, base: u16:0x0 },
            FseTableRecord { symbol: u8:0xe, num_of_bits: u8:0x2, base: u16:0x14 },
            FseTableRecord { symbol: u8:0xd, num_of_bits: u8:0x3, base: u16:0x68 },
            FseTableRecord { symbol: u8:0xb, num_of_bits: u8:0x3, base: u16:0x8 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x4, base: u16:0x0 },
            FseTableRecord { symbol: u8:0xe, num_of_bits: u8:0x2, base: u16:0x18 },
            FseTableRecord { symbol: u8:0xc, num_of_bits: u8:0x3, base: u16:0x48 },
            FseTableRecord { symbol: u8:0x6, num_of_bits: u8:0x7, base: u16:0x0 },
            FseTableRecord { symbol: u8:0xe, num_of_bits: u8:0x2, base: u16:0x1c },
            FseTableRecord { symbol: u8:0xd, num_of_bits: u8:0x3, base: u16:0x70 },
            FseTableRecord { symbol: u8:0xb, num_of_bits: u8:0x3, base: u16:0x10 },
            FseTableRecord { symbol: u8:0xf, num_of_bits: u8:0x4, base: u16:0x0 },
            FseTableRecord { symbol: u8:0xd, num_of_bits: u8:0x3, base: u16:0x78 },
            FseTableRecord { symbol: u8:0xc, num_of_bits: u8:0x3, base: u16:0x50 },
            FseTableRecord { symbol: u8:0x3, num_of_bits: u8:0x7, base: u16:0x0 },
            FseTableRecord { symbol: u8:0xe, num_of_bits: u8:0x2, base: u16:0x20 },
            FseTableRecord { symbol: u8:0xd, num_of_bits: u8:0x2, base: u16:0x0 },
            FseTableRecord { symbol: u8:0xa, num_of_bits: u8:0x4, base: u16:0x10 },
            FseTableRecord { symbol: u8:0xf, num_of_bits: u8:0x4, base: u16:0x10 },
            FseTableRecord { symbol: u8:0xd, num_of_bits: u8:0x2, base: u16:0x4 },
            FseTableRecord { symbol: u8:0xc, num_of_bits: u8:0x3, base: u16:0x58 },
            FseTableRecord { symbol: u8:0x1, num_of_bits: u8:0x4, base: u16:0x40 },
            FseTableRecord { symbol: u8:0xe, num_of_bits: u8:0x2, base: u16:0x24 },
            FseTableRecord { symbol: u8:0xc, num_of_bits: u8:0x3, base: u16:0x60 },
            FseTableRecord { symbol: u8:0xa, num_of_bits: u8:0x4, base: u16:0x20 },
            FseTableRecord { symbol: u8:0xe, num_of_bits: u8:0x2, base: u16:0x28 },
            FseTableRecord { symbol: u8:0xd, num_of_bits: u8:0x2, base: u16:0x8 },
            FseTableRecord { symbol: u8:0xb, num_of_bits: u8:0x3, base: u16:0x18 },
            FseTableRecord { symbol: u8:0x1, num_of_bits: u8:0x4, base: u16:0x50 },
            FseTableRecord { symbol: u8:0xe, num_of_bits: u8:0x2, base: u16:0x2c },
            FseTableRecord { symbol: u8:0xc, num_of_bits: u8:0x3, base: u16:0x68 },
            FseTableRecord { symbol: u8:0x9, num_of_bits: u8:0x5, base: u16:0x60 },
            FseTableRecord { symbol: u8:0xe, num_of_bits: u8:0x2, base: u16:0x30 },
            FseTableRecord { symbol: u8:0xd, num_of_bits: u8:0x2, base: u16:0xc },
            FseTableRecord { symbol: u8:0xb, num_of_bits: u8:0x3, base: u16:0x20 },
            FseTableRecord { symbol: u8:0x1, num_of_bits: u8:0x4, base: u16:0x60 },
            FseTableRecord { symbol: u8:0xe, num_of_bits: u8:0x2, base: u16:0x34 },
            FseTableRecord { symbol: u8:0xc, num_of_bits: u8:0x3, base: u16:0x70 },
            FseTableRecord { symbol: u8:0x9, num_of_bits: u8:0x4, base: u16:0x0 },
            FseTableRecord { symbol: u8:0xe, num_of_bits: u8:0x2, base: u16:0x38 },
            FseTableRecord { symbol: u8:0xd, num_of_bits: u8:0x2, base: u16:0x10 },
            FseTableRecord { symbol: u8:0xb, num_of_bits: u8:0x3, base: u16:0x28 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x4, base: u16:0x10 },
            FseTableRecord { symbol: u8:0xe, num_of_bits: u8:0x2, base: u16:0x3c },
            FseTableRecord { symbol: u8:0xc, num_of_bits: u8:0x3, base: u16:0x78 },
            FseTableRecord { symbol: u8:0x7, num_of_bits: u8:0x6, base: u16:0x0 },
            FseTableRecord { symbol: u8:0xe, num_of_bits: u8:0x2, base: u16:0x40 },
            FseTableRecord { symbol: u8:0xd, num_of_bits: u8:0x2, base: u16:0x14 },
            FseTableRecord { symbol: u8:0xb, num_of_bits: u8:0x3, base: u16:0x30 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x4, base: u16:0x20 },
            FseTableRecord { symbol: u8:0xe, num_of_bits: u8:0x2, base: u16:0x44 },
            FseTableRecord { symbol: u8:0xc, num_of_bits: u8:0x2, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x5, num_of_bits: u8:0x7, base: u16:0x0 },
            FseTableRecord { symbol: u8:0xe, num_of_bits: u8:0x2, base: u16:0x48 },
            FseTableRecord { symbol: u8:0xd, num_of_bits: u8:0x2, base: u16:0x18 },
            FseTableRecord { symbol: u8:0xa, num_of_bits: u8:0x4, base: u16:0x30 },
            FseTableRecord { symbol: u8:0xf, num_of_bits: u8:0x4, base: u16:0x20 },
            FseTableRecord { symbol: u8:0xd, num_of_bits: u8:0x2, base: u16:0x1c },
            FseTableRecord { symbol: u8:0xc, num_of_bits: u8:0x2, base: u16:0x4 },
            FseTableRecord { symbol: u8:0x2, num_of_bits: u8:0x7, base: u16:0x0 },
            FseTableRecord { symbol: u8:0xe, num_of_bits: u8:0x2, base: u16:0x4c },
            FseTableRecord { symbol: u8:0xd, num_of_bits: u8:0x2, base: u16:0x20 },
            FseTableRecord { symbol: u8:0xa, num_of_bits: u8:0x4, base: u16:0x40 },
            FseTableRecord { symbol: u8:0xf, num_of_bits: u8:0x4, base: u16:0x30 },
            FseTableRecord { symbol: u8:0xd, num_of_bits: u8:0x2, base: u16:0x24 },
            FseTableRecord { symbol: u8:0xb, num_of_bits: u8:0x3, base: u16:0x38 },
            FseTableRecord { symbol: u8:0x1, num_of_bits: u8:0x4, base: u16:0x70 },
            FseTableRecord { symbol: u8:0xe, num_of_bits: u8:0x2, base: u16:0x50 },
            FseTableRecord { symbol: u8:0xc, num_of_bits: u8:0x2, base: u16:0x8 },
            FseTableRecord { symbol: u8:0xa, num_of_bits: u8:0x4, base: u16:0x50 },
            FseTableRecord { symbol: u8:0xe, num_of_bits: u8:0x2, base: u16:0x54 },
            FseTableRecord { symbol: u8:0xd, num_of_bits: u8:0x2, base: u16:0x28 },
            FseTableRecord { symbol: u8:0xb, num_of_bits: u8:0x3, base: u16:0x40 },
            FseTableRecord { symbol: u8:0x1, num_of_bits: u8:0x3, base: u16:0x0 },
            FseTableRecord { symbol: u8:0xe, num_of_bits: u8:0x2, base: u16:0x58 },
            FseTableRecord { symbol: u8:0xc, num_of_bits: u8:0x2, base: u16:0xc },
            FseTableRecord { symbol: u8:0x9, num_of_bits: u8:0x4, base: u16:0x10 },
            FseTableRecord { symbol: u8:0xe, num_of_bits: u8:0x2, base: u16:0x5c },
            FseTableRecord { symbol: u8:0xd, num_of_bits: u8:0x2, base: u16:0x2c },
            FseTableRecord { symbol: u8:0xb, num_of_bits: u8:0x3, base: u16:0x48 },
            FseTableRecord { symbol: u8:0x1, num_of_bits: u8:0x3, base: u16:0x8 },
            FseTableRecord { symbol: u8:0xe, num_of_bits: u8:0x2, base: u16:0x60 },
            FseTableRecord { symbol: u8:0xc, num_of_bits: u8:0x2, base: u16:0x10 },
            FseTableRecord { symbol: u8:0x8, num_of_bits: u8:0x6, base: u16:0x40 },
            FseTableRecord { symbol: u8:0xe, num_of_bits: u8:0x2, base: u16:0x64 },
            FseTableRecord { symbol: u8:0xd, num_of_bits: u8:0x2, base: u16:0x30 },
            FseTableRecord { symbol: u8:0xb, num_of_bits: u8:0x3, base: u16:0x50 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x4, base: u16:0x30 },
            FseTableRecord { symbol: u8:0xe, num_of_bits: u8:0x2, base: u16:0x68 },
            FseTableRecord { symbol: u8:0xc, num_of_bits: u8:0x2, base: u16:0x14 },
            FseTableRecord { symbol: u8:0x7, num_of_bits: u8:0x6, base: u16:0x40 },
            FseTableRecord { symbol: u8:0xe, num_of_bits: u8:0x2, base: u16:0x6c },
            FseTableRecord { symbol: u8:0xd, num_of_bits: u8:0x2, base: u16:0x34 },
            FseTableRecord { symbol: u8:0xb, num_of_bits: u8:0x3, base: u16:0x58 },
            zero!<FseTableRecord>(), ...
        ],
        FseLookupDecoderResp { status: FseLookupDecoderStatus::OK, accuracy_log: AccuracyLog:7 }
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
        FseLookupDecoderResp { status: FseLookupDecoderStatus::OK, accuracy_log: AccuracyLog:5 }
    ),
    (
        u64[64]:[u64:0x1101141108088A1, u64:0xA210842108421011, u64:0xAC90E792007A5B4, u64:0, ...],
        FseTableRecord[TEST_FSE_RAM_SIZE]:[
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x3, base: u16:0x8 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x3, base: u16:0x10 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x3, base: u16:0x18 },
            FseTableRecord { symbol: u8:0x1, num_of_bits: u8:0x6, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x8, num_of_bits: u8:0x6, base: u16:0x0 },
            FseTableRecord { symbol: u8:0xe, num_of_bits: u8:0x6, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x11, num_of_bits: u8:0x6, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x14, num_of_bits: u8:0x6, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x17, num_of_bits: u8:0x6, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x18, num_of_bits: u8:0x4, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x19, num_of_bits: u8:0x3, base: u16:0x20 },
            FseTableRecord { symbol: u8:0x19, num_of_bits: u8:0x3, base: u16:0x28 },
            FseTableRecord { symbol: u8:0x19, num_of_bits: u8:0x3, base: u16:0x30 },
            FseTableRecord { symbol: u8:0x19, num_of_bits: u8:0x3, base: u16:0x38 },
            FseTableRecord { symbol: u8:0x1a, num_of_bits: u8:0x3, base: u16:0x10 },
            FseTableRecord { symbol: u8:0x1a, num_of_bits: u8:0x3, base: u16:0x18 },
            FseTableRecord { symbol: u8:0x1a, num_of_bits: u8:0x3, base: u16:0x20 },
            FseTableRecord { symbol: u8:0x1b, num_of_bits: u8:0x3, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x1b, num_of_bits: u8:0x3, base: u16:0x8 },
            FseTableRecord { symbol: u8:0x1b, num_of_bits: u8:0x3, base: u16:0x10 },
            FseTableRecord { symbol: u8:0x1c, num_of_bits: u8:0x4, base: u16:0x10 },
            FseTableRecord { symbol: u8:0x1c, num_of_bits: u8:0x4, base: u16:0x20 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x3, base: u16:0x20 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x3, base: u16:0x28 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x3, base: u16:0x30 },
            FseTableRecord { symbol: u8:0x4, num_of_bits: u8:0x6, base: u16:0x0 },
            FseTableRecord { symbol: u8:0xc, num_of_bits: u8:0x6, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x10, num_of_bits: u8:0x6, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x13, num_of_bits: u8:0x6, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x16, num_of_bits: u8:0x6, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x18, num_of_bits: u8:0x4, base: u16:0x10 },
            FseTableRecord { symbol: u8:0x19, num_of_bits: u8:0x2, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x19, num_of_bits: u8:0x2, base: u16:0x4 },
            FseTableRecord { symbol: u8:0x19, num_of_bits: u8:0x2, base: u16:0x8 },
            FseTableRecord { symbol: u8:0x19, num_of_bits: u8:0x2, base: u16:0xc },
            FseTableRecord { symbol: u8:0x1a, num_of_bits: u8:0x3, base: u16:0x28 },
            FseTableRecord { symbol: u8:0x1a, num_of_bits: u8:0x3, base: u16:0x30 },
            FseTableRecord { symbol: u8:0x1a, num_of_bits: u8:0x3, base: u16:0x38 },
            FseTableRecord { symbol: u8:0x1a, num_of_bits: u8:0x2, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x1b, num_of_bits: u8:0x3, base: u16:0x18 },
            FseTableRecord { symbol: u8:0x1b, num_of_bits: u8:0x3, base: u16:0x20 },
            FseTableRecord { symbol: u8:0x1c, num_of_bits: u8:0x4, base: u16:0x30 },
            FseTableRecord { symbol: u8:0x1c, num_of_bits: u8:0x3, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x3, base: u16:0x38 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x2, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x2, base: u16:0x4 },
            FseTableRecord { symbol: u8:0x3, num_of_bits: u8:0x6, base: u16:0x0 },
            FseTableRecord { symbol: u8:0xa, num_of_bits: u8:0x6, base: u16:0x0 },
            FseTableRecord { symbol: u8:0xf, num_of_bits: u8:0x6, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x12, num_of_bits: u8:0x6, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x15, num_of_bits: u8:0x6, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x18, num_of_bits: u8:0x4, base: u16:0x20 },
            FseTableRecord { symbol: u8:0x18, num_of_bits: u8:0x4, base: u16:0x30 },
            FseTableRecord { symbol: u8:0x19, num_of_bits: u8:0x2, base: u16:0x10 },
            FseTableRecord { symbol: u8:0x19, num_of_bits: u8:0x2, base: u16:0x14 },
            FseTableRecord { symbol: u8:0x19, num_of_bits: u8:0x2, base: u16:0x18 },
            FseTableRecord { symbol: u8:0x19, num_of_bits: u8:0x2, base: u16:0x1c },
            FseTableRecord { symbol: u8:0x1a, num_of_bits: u8:0x2, base: u16:0x4 },
            FseTableRecord { symbol: u8:0x1a, num_of_bits: u8:0x2, base: u16:0x8 },
            FseTableRecord { symbol: u8:0x1a, num_of_bits: u8:0x2, base: u16:0xc },
            FseTableRecord { symbol: u8:0x1b, num_of_bits: u8:0x3, base: u16:0x28 },
            FseTableRecord { symbol: u8:0x1b, num_of_bits: u8:0x3, base: u16:0x30 },
            FseTableRecord { symbol: u8:0x1b, num_of_bits: u8:0x3, base: u16:0x38 },
            FseTableRecord { symbol: u8:0x1c, num_of_bits: u8:0x3, base: u16:0x8 },
            zero!<FseTableRecord>(), ...
        ],
        FseLookupDecoderResp { status: FseLookupDecoderStatus::OK, accuracy_log: AccuracyLog:6 }
    ),
    (
        u64[64]:[u64:0x4AF830AC90E7920, u64:0, ...],
        FseTableRecord[TEST_FSE_RAM_SIZE]:[
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x1, base: u16:0x2 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x1, base: u16:0x4 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x1, base: u16:0x6 },
            FseTableRecord { symbol: u8:0x1, num_of_bits: u8:0x2, base: u16:0xc },
            FseTableRecord { symbol: u8:0x2, num_of_bits: u8:0x4, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x1, base: u16:0x8 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x1, base: u16:0xa },
            FseTableRecord { symbol: u8:0x1, num_of_bits: u8:0x2, base: u16:0x10 },
            FseTableRecord { symbol: u8:0x1, num_of_bits: u8:0x2, base: u16:0x14 },
            FseTableRecord { symbol: u8:0x4, num_of_bits: u8:0x5, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x1, base: u16:0xc },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x1, base: u16:0xe },
            FseTableRecord { symbol: u8:0x1, num_of_bits: u8:0x2, base: u16:0x18 },
            FseTableRecord { symbol: u8:0x1, num_of_bits: u8:0x2, base: u16:0x1c },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x1, base: u16:0x10 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x1, base: u16:0x12 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x1, base: u16:0x14 },
            FseTableRecord { symbol: u8:0x1, num_of_bits: u8:0x1, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x3, num_of_bits: u8:0x5, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x1, base: u16:0x16 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x1, base: u16:0x18 },
            FseTableRecord { symbol: u8:0x1, num_of_bits: u8:0x1, base: u16:0x2 },
            FseTableRecord { symbol: u8:0x1, num_of_bits: u8:0x1, base: u16:0x4 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x1, base: u16:0x1a },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x1, base: u16:0x1c },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x1, base: u16:0x1e },
            FseTableRecord { symbol: u8:0x1, num_of_bits: u8:0x1, base: u16:0x6 },
            FseTableRecord { symbol: u8:0x2, num_of_bits: u8:0x4, base: u16:0x10 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x0, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x0, base: u16:0x1 },
            FseTableRecord { symbol: u8:0x1, num_of_bits: u8:0x1, base: u16:0x8 },
            FseTableRecord { symbol: u8:0x1, num_of_bits: u8:0x1, base: u16:0xa },
            zero!<FseTableRecord>(), ...
        ],
        FseLookupDecoderResp { status: FseLookupDecoderStatus::OK, accuracy_log: AccuracyLog:5 }
    ),
    (
        u64[64]:[u64:0xF47FFEBBFF1D25C0, u64:0, ...],
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
        FseLookupDecoderResp { status: FseLookupDecoderStatus::OK, accuracy_log: AccuracyLog:5 }
    ),
    (
        u64[64]:[u64:0xA84DF134544CA40, u64:0xEEC609988403B0C, u64:0, ...],
        FseTableRecord[TEST_FSE_RAM_SIZE]:[
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x4, base: u16:0x10 },
            FseTableRecord { symbol: u8:0x2, num_of_bits: u8:0x4, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x6, num_of_bits: u8:0x4, base: u16:0x10 },
            FseTableRecord { symbol: u8:0x12, num_of_bits: u8:0x3, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x16, num_of_bits: u8:0x4, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x1, num_of_bits: u8:0x3, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x5, num_of_bits: u8:0x3, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x7, num_of_bits: u8:0x4, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x14, num_of_bits: u8:0x4, base: u16:0x10 },
            FseTableRecord { symbol: u8:0x17, num_of_bits: u8:0x4, base: u16:0x10 },
            FseTableRecord { symbol: u8:0x1, num_of_bits: u8:0x3, base: u16:0x8 },
            FseTableRecord { symbol: u8:0x6, num_of_bits: u8:0x3, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x12, num_of_bits: u8:0x3, base: u16:0x8 },
            FseTableRecord { symbol: u8:0x16, num_of_bits: u8:0x4, base: u16:0x10 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x3, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x5, num_of_bits: u8:0x3, base: u16:0x8 },
            FseTableRecord { symbol: u8:0x7, num_of_bits: u8:0x4, base: u16:0x10 },
            FseTableRecord { symbol: u8:0x12, num_of_bits: u8:0x3, base: u16:0x10 },
            FseTableRecord { symbol: u8:0x17, num_of_bits: u8:0x3, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x1, num_of_bits: u8:0x3, base: u16:0x10 },
            FseTableRecord { symbol: u8:0x5, num_of_bits: u8:0x3, base: u16:0x10 },
            FseTableRecord { symbol: u8:0x10, num_of_bits: u8:0x4, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x14, num_of_bits: u8:0x3, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x3, base: u16:0x8 },
            FseTableRecord { symbol: u8:0x2, num_of_bits: u8:0x4, base: u16:0x10 },
            FseTableRecord { symbol: u8:0x6, num_of_bits: u8:0x3, base: u16:0x8 },
            FseTableRecord { symbol: u8:0x12, num_of_bits: u8:0x3, base: u16:0x18 },
            FseTableRecord { symbol: u8:0x17, num_of_bits: u8:0x3, base: u16:0x8 },
            FseTableRecord { symbol: u8:0x1, num_of_bits: u8:0x3, base: u16:0x18 },
            FseTableRecord { symbol: u8:0x5, num_of_bits: u8:0x3, base: u16:0x18 },
            FseTableRecord { symbol: u8:0x10, num_of_bits: u8:0x4, base: u16:0x10 },
            FseTableRecord { symbol: u8:0x14, num_of_bits: u8:0x3, base: u16:0x8 },
            zero!<FseTableRecord>(), ...
        ],
        FseLookupDecoderResp { status: FseLookupDecoderStatus::OK, accuracy_log: AccuracyLog:5 }
    ),
    (
        u64[64]:[u64:0x38100EEC60998840, u64:0, ...],
        FseTableRecord[TEST_FSE_RAM_SIZE]:[
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x4, base: u16:0x10 },
            FseTableRecord { symbol: u8:0x2, num_of_bits: u8:0x3, base: u16:0x8 },
            FseTableRecord { symbol: u8:0x5, num_of_bits: u8:0x4, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x7, num_of_bits: u8:0x3, base: u16:0x10 },
            FseTableRecord { symbol: u8:0x8, num_of_bits: u8:0x3, base: u16:0x10 },
            FseTableRecord { symbol: u8:0x1, num_of_bits: u8:0x4, base: u16:0x10 },
            FseTableRecord { symbol: u8:0x2, num_of_bits: u8:0x3, base: u16:0x10 },
            FseTableRecord { symbol: u8:0x6, num_of_bits: u8:0x3, base: u16:0x8 },
            FseTableRecord { symbol: u8:0x7, num_of_bits: u8:0x3, base: u16:0x18 },
            FseTableRecord { symbol: u8:0x8, num_of_bits: u8:0x3, base: u16:0x18 },
            FseTableRecord { symbol: u8:0x2, num_of_bits: u8:0x3, base: u16:0x18 },
            FseTableRecord { symbol: u8:0x5, num_of_bits: u8:0x4, base: u16:0x10 },
            FseTableRecord { symbol: u8:0x7, num_of_bits: u8:0x2, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x8, num_of_bits: u8:0x2, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x3, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x2, num_of_bits: u8:0x2, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x6, num_of_bits: u8:0x3, base: u16:0x10 },
            FseTableRecord { symbol: u8:0x7, num_of_bits: u8:0x2, base: u16:0x4 },
            FseTableRecord { symbol: u8:0x8, num_of_bits: u8:0x2, base: u16:0x4 },
            FseTableRecord { symbol: u8:0x1, num_of_bits: u8:0x3, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x3, num_of_bits: u8:0x4, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x6, num_of_bits: u8:0x3, base: u16:0x18 },
            FseTableRecord { symbol: u8:0x8, num_of_bits: u8:0x2, base: u16:0x8 },
            FseTableRecord { symbol: u8:0x0, num_of_bits: u8:0x3, base: u16:0x8 },
            FseTableRecord { symbol: u8:0x2, num_of_bits: u8:0x2, base: u16:0x4 },
            FseTableRecord { symbol: u8:0x6, num_of_bits: u8:0x2, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x7, num_of_bits: u8:0x2, base: u16:0x8 },
            FseTableRecord { symbol: u8:0x8, num_of_bits: u8:0x2, base: u16:0xc },
            FseTableRecord { symbol: u8:0x1, num_of_bits: u8:0x3, base: u16:0x8 },
            FseTableRecord { symbol: u8:0x3, num_of_bits: u8:0x4, base: u16:0x10 },
            FseTableRecord { symbol: u8:0x6, num_of_bits: u8:0x2, base: u16:0x4 },
            FseTableRecord { symbol: u8:0x7, num_of_bits: u8:0x2, base: u16:0xc },
            zero!<FseTableRecord>(), ...
        ],
        FseLookupDecoderResp { status: FseLookupDecoderStatus::OK, accuracy_log: AccuracyLog:5 }
    ),
    (
        u64[64]:[u64:0x6B1CA24D0CE43810, u64:0x6651065104A4DFFD, u64:0, ...],
        FseTableRecord[TEST_FSE_RAM_SIZE]:[
            FseTableRecord { symbol: u8:0x1, num_of_bits: u8:0x3, base: u16:0x10 },
            FseTableRecord { symbol: u8:0x2, num_of_bits: u8:0x4, base: u16:0x10 },
            FseTableRecord { symbol: u8:0x3, num_of_bits: u8:0x3, base: u16:0x10 },
            FseTableRecord { symbol: u8:0x6, num_of_bits: u8:0x4, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x12, num_of_bits: u8:0x4, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x1, num_of_bits: u8:0x3, base: u16:0x18 },
            FseTableRecord { symbol: u8:0x3, num_of_bits: u8:0x3, base: u16:0x18 },
            FseTableRecord { symbol: u8:0x5, num_of_bits: u8:0x4, base: u16:0x10 },
            FseTableRecord { symbol: u8:0x9, num_of_bits: u8:0x4, base: u16:0x10 },
            FseTableRecord { symbol: u8:0x24, num_of_bits: u8:0x4, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x2, num_of_bits: u8:0x3, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x3, num_of_bits: u8:0x2, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x6, num_of_bits: u8:0x4, base: u16:0x10 },
            FseTableRecord { symbol: u8:0xf, num_of_bits: u8:0x4, base: u16:0x10 },
            FseTableRecord { symbol: u8:0x1, num_of_bits: u8:0x2, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x3, num_of_bits: u8:0x2, base: u16:0x4 },
            FseTableRecord { symbol: u8:0x4, num_of_bits: u8:0x4, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x9, num_of_bits: u8:0x3, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x24, num_of_bits: u8:0x4, base: u16:0x10 },
            FseTableRecord { symbol: u8:0x1, num_of_bits: u8:0x2, base: u16:0x4 },
            FseTableRecord { symbol: u8:0x3, num_of_bits: u8:0x2, base: u16:0x8 },
            FseTableRecord { symbol: u8:0x5, num_of_bits: u8:0x3, base: u16:0x0 },
            FseTableRecord { symbol: u8:0xf, num_of_bits: u8:0x3, base: u16:0x0 },
            FseTableRecord { symbol: u8:0x1, num_of_bits: u8:0x2, base: u16:0x8 },
            FseTableRecord { symbol: u8:0x2, num_of_bits: u8:0x3, base: u16:0x8 },
            FseTableRecord { symbol: u8:0x4, num_of_bits: u8:0x4, base: u16:0x10 },
            FseTableRecord { symbol: u8:0x9, num_of_bits: u8:0x3, base: u16:0x8 },
            FseTableRecord { symbol: u8:0x12, num_of_bits: u8:0x4, base: u16:0x10 },
            FseTableRecord { symbol: u8:0x1, num_of_bits: u8:0x2, base: u16:0xc },
            FseTableRecord { symbol: u8:0x3, num_of_bits: u8:0x2, base: u16:0xc },
            FseTableRecord { symbol: u8:0x5, num_of_bits: u8:0x3, base: u16:0x8 },
            FseTableRecord { symbol: u8:0xf, num_of_bits: u8:0x3, base: u16:0x8 },
            zero!<FseTableRecord>(), ...
        ],
        FseLookupDecoderResp { status: FseLookupDecoderStatus::OK, accuracy_log: AccuracyLog:5 }
    ),
];

#[test_proc]
proc FseLookupDecoderTest {
    type Req = FseLookupDecoderReq<TEST_AXI_ADDR_WIDTH>;
    type Resp = FseLookupDecoderResp;
    type Status = FseLookupDecoderStatus;

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

    type TestcaseRamRdReq = ram::ReadReq<TEST_CASE_RAM_ADDR_WIDTH, TEST_CASE_RAM_NUM_PARTITIONS>;
    type TestcaseRamRdResp = ram::ReadResp<TEST_CASE_RAM_DATA_WIDTH>;
    type TestcaseRamWrReq = ram::WriteReq<TEST_CASE_RAM_ADDR_WIDTH, TEST_CASE_RAM_DATA_WIDTH, TEST_CASE_RAM_NUM_PARTITIONS>;
    type TestcaseRamWrResp = ram::WriteResp;

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

        let (fse_rd_req_s, fse_rd_req_r) = chan<FseRamRdReq>("fse_rd_req");
        let (fse_rd_resp_s, fse_rd_resp_r) = chan<FseRamRdResp>("fse_rd_resp");
        let (fse_wr_req_s, fse_wr_req_r) = chan<FseRamWrReq>("fse_wr_req");
        let (fse_wr_resp_s, fse_wr_resp_r) = chan<FseRamWrResp>("fse_wr_resp");

        let (testcase_rd_req_s, testcase_rd_req_r) = chan<TestcaseRamRdReq>("testcase_rd_req");
        let (testcase_rd_resp_s, testcase_rd_resp_r) = chan<TestcaseRamRdResp>("testcase_rd_resp");
        let (testcase_wr_req_s, testcase_wr_req_r) = chan<TestcaseRamWrReq>("testcase_wr_req");
        let (testcase_wr_resp_s, testcase_wr_resp_r) = chan<TestcaseRamWrResp>("testcase_wr_resp");

        spawn FseLookupDecoder<
            TEST_AXI_DATA_WIDTH, TEST_AXI_ADDR_WIDTH,
            TEST_DPD_RAM_DATA_WIDTH, TEST_DPD_RAM_ADDR_WIDTH, TEST_DPD_RAM_NUM_PARTITIONS,
            TEST_TMP_RAM_DATA_WIDTH, TEST_TMP_RAM_ADDR_WIDTH, TEST_TMP_RAM_NUM_PARTITIONS,
            TEST_FSE_RAM_DATA_WIDTH, TEST_FSE_RAM_ADDR_WIDTH, TEST_FSE_RAM_NUM_PARTITIONS,
        >(
            req_r,
            resp_s,
            mem_rd_req_s,
            mem_rd_resp_r,
            dpd_rd_req_s,
            dpd_rd_resp_r,
            dpd_wr_req_s,
            dpd_wr_resp_r,
            tmp_rd_req_s,
            tmp_rd_resp_r,
            tmp_wr_req_s,
            tmp_wr_resp_r,
            fse_rd_req_s,
            fse_rd_resp_r,
            fse_wr_req_s,
            fse_wr_resp_r,
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

        (
            terminator, req_s, resp_r, fse_rd_req_s, fse_rd_resp_r,
            fse_wr_req_s, fse_wr_resp_r, testcase_wr_req_s, testcase_wr_resp_r,
        )
    }

    init {}

    next(_: ()) {
        let tok = join();
        // This has to be outside of unroll_for!, otherwise typechecker reports type mismatch on identical types
        let req_start = Req { addr: uN[TEST_AXI_ADDR_WIDTH]:0x0 };

        let tok = unroll_for!(test_i, tok): (u32, token) in range(u32:0, array_size(FSE_LOOKUP_DECODER_TESTCASES)) {
            let (input, output, resp_ok) = FSE_LOOKUP_DECODER_TESTCASES[test_i];

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
            let tok = send(tok, req_s, req_start);
            let (tok, resp) = recv(tok, resp_r);
            assert_eq(resp, resp_ok);

            let tok = for ((i, output_data), tok): ((u32, FseTableRecord), token) in enumerate(output) {
                let req = FseRamRdReq {
                    addr: i as uN[TEST_FSE_RAM_ADDR_WIDTH],
                    mask: uN[TEST_FSE_RAM_NUM_PARTITIONS]:0x7,
                };
                let tok = send(tok, fse_rd_req_s, req);
                let (tok, resp) = recv(tok, fse_rd_resp_r);
                assert_eq(fse_table_creator::bits_to_fse_record(resp.data), output_data);

                // erase output for next test to start with clean memory
                let clear_req = FseRamWrReq {
                    addr: i as uN[TEST_FSE_RAM_ADDR_WIDTH],
                    mask: uN[TEST_FSE_RAM_NUM_PARTITIONS]:0x7,
                    data: uN[TEST_FSE_RAM_DATA_WIDTH]:0x0,
                };
                let tok = send(tok, fse_wr_req_s, clear_req);
                let (tok, _) = recv(tok, fse_wr_resp_r);
                tok
            }(tok);
            tok
        }(tok);

        send(tok, terminator, true);
    }
}
