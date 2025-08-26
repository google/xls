// Copyright 2025 The XLS Authors
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

// This file contains CompressBlockEncoder proc implementation
// The proc:
//  1. runs MatchFinder on input data which populates literals and sequences buffers
//  2. runs LiteralsEncoder on the literals from the literals buffer (RAW for now)
//  3. runs SequenceEncoder on the sequences (only PREDEFINED for now)
import std;

import xls.examples.ram;
import xls.modules.zstd.common;
import xls.modules.zstd.mem_copy;
import xls.modules.zstd.memory.mem_reader;
import xls.modules.zstd.memory.mem_writer;
import xls.modules.zstd.history_buffer;
import xls.modules.zstd.hash_table;
import xls.modules.zstd.mem_reader_simple_arbiter;
import xls.modules.zstd.mem_writer_simple_arbiter;
import xls.modules.zstd.match_finder;
import xls.modules.zstd.literal_encoder;
import xls.modules.zstd.sequence_encoder;

pub struct CompressBlockEncoderReq<ADDR_W: u32, DATA_W: u32> {
    addr: uN[ADDR_W],
    size: uN[ADDR_W],
    out_addr: uN[ADDR_W],
}

pub enum CompressBlockEncoderStatus : u1 {
    OK = 0,
    ERROR = 1,
}

pub struct CompressBlockEncoderResp<ADDR_W: u32> {
    status: CompressBlockEncoderStatus,
    length: uN[ADDR_W],
}

pub proc CompressBlockEncoder<ADDR_W: u32, DATA_W: u32, HB_SIZE: u32, HB_DATA_W: u32, HB_OFFSET_W:
u32, HB_RAM_ADDR_W: u32, HB_RAM_DATA_W: u32, HB_RAM_NUM: u32, HB_RAM_NUM_PARTITIONS: u32, HT_SIZE:
u32, HT_KEY_W: u32, HT_VALUE_W: u32, HT_SIZE_W: u32, HT_HASH_W: u32, HT_RAM_DATA_W: u32, HT_RAM_NUM_PARTITIONS:
u32, MIN_SEQ_LEN: u32, LITERALS_BUFFER_AXI_ADDR: u32, SEQUENCE_BUFFER_AXI_ADDR: u32, FSE_TABLE_RAM_ADDR_W:
u32, FSE_CTABLE_RAM_DATA_W: u32, FSE_TTABLE_RAM_DATA_W: u32, FSE_CTABLE_RAM_NUM_PARTITIONS: u32, FSE_TTABLE_RAM_NUM_PARTITIONS:
u32, BITSTREAM_BUFFER_W: u32, DATA_W_LOG2: u32 = {std::clog2(DATA_W + u32:1)}>
{
    type Req = CompressBlockEncoderReq<ADDR_W, DATA_W>;
    type Resp = CompressBlockEncoderResp;
    type Status = CompressBlockEncoderStatus;
    // input/output
    type MemReaderReq = mem_reader::MemReaderReq<ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<DATA_W, ADDR_W>;
    type MemReaderStatus = mem_reader::MemReaderStatus;
    type MemWriterReq = mem_writer::MemWriterReq<ADDR_W>;
    type MemWriterData = mem_writer::MemWriterDataPacket<DATA_W, ADDR_W>;
    type MemWriterResp = mem_writer::MemWriterResp;
    type MemWriterStatus = mem_writer::MemWriterRespStatus;
    // match finder
    type MfResp = match_finder::MatchFinderResp;
    type MfParams = match_finder::ZstdParams<HT_SIZE_W>;
    type MfReq = match_finder::MatchFinderReq<HT_SIZE_W, ADDR_W>;
    type MfRespStatus = match_finder::MatchFinderRespStatus;
    type HistoryBufferRdReq = history_buffer::HistoryBufferReadReq<HB_OFFSET_W>;
    type HistoryBufferRdResp = history_buffer::HistoryBufferReadResp<HB_DATA_W>;
    type HistoryBufferWrReq = history_buffer::HistoryBufferWriteReq<HB_DATA_W>;
    type HistoryBufferWrResp = history_buffer::HistoryBufferWriteResp;
    type HistoryBufferRamRdReq = ram::ReadReq<HB_RAM_ADDR_W, HB_RAM_NUM_PARTITIONS>;
    type HistoryBufferRamRdResp = ram::ReadResp<HB_RAM_DATA_W>;
    type HistoryBufferRamWrReq = ram::WriteReq<HB_RAM_ADDR_W, HB_RAM_DATA_W, HB_RAM_NUM_PARTITIONS>;
    type HistoryBufferRamWrResp = ram::WriteResp;
    type HashTableRdReq = hash_table::HashTableReadReq<HT_KEY_W, HT_SIZE, HT_SIZE_W>;
    type HashTableRdResp = hash_table::HashTableReadResp<HT_VALUE_W>;
    type HashTableWrReq = hash_table::HashTableWriteReq<HT_KEY_W, HT_VALUE_W, HT_SIZE, HT_SIZE_W>;
    type HashTableWrResp = hash_table::HashTableWriteResp;
    type HashTableRamRdReq = ram::ReadReq<HT_HASH_W, HT_RAM_NUM_PARTITIONS>;
    type HashTableRamRdResp = ram::ReadResp<HT_RAM_DATA_W>;
    type HashTableRamWrReq = ram::WriteReq<HT_HASH_W, HT_RAM_DATA_W, HT_RAM_NUM_PARTITIONS>;
    type HashTableRamWrResp = ram::WriteResp;
    // literals encoder
    type RawMemcopyBlockType = mem_copy::RawMemcopyBlockType;
    type RawMemcopyStatus = mem_copy::RawMemcopyStatus;
    type RawMemcopyReq = mem_copy::RawMemcopyReq;
    type RawMemcopyResp = mem_copy::RawMemcopyResp;
    // sequence encoder
    type SeReq = sequence_encoder::SequenceEncoderReq<ADDR_W>;
    type SeResp = sequence_encoder::SequenceEncoderResp<ADDR_W>;
    type SeRespStatus = sequence_encoder::SequenceEncoderStatus;
    type CTableRamRdReq = ram::ReadReq<FSE_TABLE_RAM_ADDR_W, FSE_CTABLE_RAM_NUM_PARTITIONS>;
    type CTableRamRdResp = ram::ReadResp<FSE_CTABLE_RAM_DATA_W>;
    type TTableRamRdReq = ram::ReadReq<FSE_TABLE_RAM_ADDR_W, FSE_TTABLE_RAM_NUM_PARTITIONS>;
    type TTableRamRdResp = ram::ReadResp<FSE_TTABLE_RAM_DATA_W>;
    req_r: chan<Req> in;
    resp_s: chan<Resp> out;
    mf_req_s: chan<MfReq> out;
    mf_resp_r: chan<MfResp> in;
    le_req_s: chan<RawMemcopyReq> out;
    le_resp_r: chan<RawMemcopyResp> in;
    se_req_s: chan<SeReq> out;
    se_resp_r: chan<SeResp> in;

    // input/output

    // match finder

    // literals encoder

    // sequence encoder
    config(req_r: chan<Req> in, resp_s: chan<Resp> out, input_mem_rd_req_s: chan<MemReaderReq> out,
           input_mem_rd_resp_r: chan<MemReaderResp> in, mf_buf_mem_rd_req_s: chan<MemReaderReq> out,
           mf_buf_mem_rd_resp_r: chan<MemReaderResp> in,
           mf_buf_mem_wr_req_s: chan<MemWriterReq> out,
           mf_buf_mem_wr_data_s: chan<MemWriterData> out,
           mf_buf_mem_wr_resp_r: chan<MemWriterResp> in,
           hb_ram_rd_req_s: chan<HistoryBufferRamRdReq>[HB_RAM_NUM] out,
           hb_ram_rd_resp_r: chan<HistoryBufferRamRdResp>[HB_RAM_NUM] in,
           hb_ram_wr_req_s: chan<HistoryBufferRamWrReq>[HB_RAM_NUM] out,
           hb_ram_wr_resp_r: chan<HistoryBufferRamWrResp>[HB_RAM_NUM] in,
           ht_ram_rd_req_s: chan<HashTableRamRdReq> out,
           ht_ram_rd_resp_r: chan<HashTableRamRdResp> in,
           ht_ram_wr_req_s: chan<HashTableRamWrReq> out,
           ht_ram_wr_resp_r: chan<HashTableRamWrResp> in,
           lhw_output_mem_wr_req_s: chan<MemWriterReq> out,
           lhw_output_mem_wr_data_s: chan<MemWriterData> out,
           lhw_output_mem_wr_resp_r: chan<MemWriterResp> in,
           le_output_mem_wr_req_s: chan<MemWriterReq> out,
           le_output_mem_wr_data_s: chan<MemWriterData> out,
           le_output_mem_wr_resp_r: chan<MemWriterResp> in,
           se_output_mem_wr_req_s: chan<MemWriterReq> out,
           se_output_mem_wr_data_s: chan<MemWriterData> out,
           se_output_mem_wr_resp_r: chan<MemWriterResp> in,
           ml_ctable_ram_rd_req_s: chan<CTableRamRdReq> out,
           ml_ctable_ram_rd_resp_r: chan<CTableRamRdResp> in,
           ll_ctable_ram_rd_req_s: chan<CTableRamRdReq> out,
           ll_ctable_ram_rd_resp_r: chan<CTableRamRdResp> in,
           of_ctable_ram_rd_req_s: chan<CTableRamRdReq> out,
           of_ctable_ram_rd_resp_r: chan<CTableRamRdResp> in,
           ml_ttable_ram_rd_req_s: chan<TTableRamRdReq> out,
           ml_ttable_ram_rd_resp_r: chan<TTableRamRdResp> in,
           ll_ttable_ram_rd_req_s: chan<TTableRamRdReq> out,
           ll_ttable_ram_rd_resp_r: chan<TTableRamRdResp> in,
           of_ttable_ram_rd_req_s: chan<TTableRamRdReq> out,
           of_ttable_ram_rd_resp_r: chan<TTableRamRdResp> in) {
        const CHANNEL_DEPTH = u32:1;

        // match finder
        let (mf_req_s, mf_req_r) = chan<MfReq, CHANNEL_DEPTH>("mf_req");
        let (mf_resp_s, mf_resp_r) = chan<MfResp, CHANNEL_DEPTH>("mf_resp");
        let (ht_rd_req_s, ht_rd_req_r) = chan<HashTableRdReq, CHANNEL_DEPTH>("ht_rd_req");
        let (ht_rd_resp_s, ht_rd_resp_r) = chan<HashTableRdResp, CHANNEL_DEPTH>("ht_rd_resp");
        let (ht_wr_req_s, ht_wr_req_r) = chan<HashTableWrReq, CHANNEL_DEPTH>("ht_wr_req");
        let (ht_wr_resp_s, ht_wr_resp_r) = chan<HashTableWrResp, CHANNEL_DEPTH>("ht_wr_resp");
        let (hb_rd_req_s, hb_rd_req_r) = chan<HistoryBufferRdReq, CHANNEL_DEPTH>("hb_rd_req");
        let (hb_rd_resp_s, hb_rd_resp_r) = chan<HistoryBufferRdResp, CHANNEL_DEPTH>("hb_rd_resp");
        let (hb_wr_req_s, hb_wr_req_r) = chan<HistoryBufferWrReq, CHANNEL_DEPTH>("hb_wr_req");
        let (hb_wr_resp_s, hb_wr_resp_r) = chan<HistoryBufferWrResp, CHANNEL_DEPTH>("hb_wr_resp");
        let (n_mf_buf_mem_rd_req_s, n_mf_buf_mem_rd_req_r) =
            chan<MemReaderReq, CHANNEL_DEPTH>[2]("n_mf_buf_rd_req");
        let (n_mf_buf_mem_rd_resp_s, n_mf_buf_mem_rd_resp_r) =
            chan<MemReaderResp, CHANNEL_DEPTH>[2]("n_mf_buf_rd_resp");
        // literals encoder
        let (le_req_s, le_req_r) = chan<RawMemcopyReq, CHANNEL_DEPTH>("le_req");
        let (le_resp_s, le_resp_r) = chan<RawMemcopyResp, CHANNEL_DEPTH>("le_resp");
        // sequence encoder
        let (se_req_s, se_req_r) = chan<SeReq, CHANNEL_DEPTH>("se_req");
        let (se_resp_s, se_resp_r) = chan<SeResp, CHANNEL_DEPTH>("se_resp");

        // match finder
        spawn hash_table::HashTable<HT_KEY_W, HT_VALUE_W, HT_SIZE, HT_SIZE_W>(
            ht_rd_req_r, ht_rd_resp_s, ht_wr_req_r, ht_wr_resp_s, ht_ram_rd_req_s, ht_ram_rd_resp_r,
            ht_ram_wr_req_s, ht_ram_wr_resp_r);
        spawn history_buffer::HistoryBuffer<HB_SIZE, HB_DATA_W>(
            hb_rd_req_r, hb_rd_resp_s, hb_wr_req_r, hb_wr_resp_s, hb_ram_rd_req_s, hb_ram_rd_resp_r,
            hb_ram_wr_req_s, hb_ram_wr_resp_r);
        spawn match_finder::MatchFinder<
            ADDR_W, DATA_W, HT_SIZE, HB_SIZE, MIN_SEQ_LEN, DATA_W_LOG2, HT_KEY_W, HT_VALUE_W, HT_SIZE_W, HB_DATA_W, HB_OFFSET_W>(
            mf_req_r, mf_resp_s, input_mem_rd_req_s, input_mem_rd_resp_r, mf_buf_mem_wr_req_s,
            mf_buf_mem_wr_data_s, mf_buf_mem_wr_resp_r, ht_rd_req_s, ht_rd_resp_r, ht_wr_req_s,
            ht_wr_resp_r, hb_rd_req_s, hb_rd_resp_r, hb_wr_req_s, hb_wr_resp_r);
        spawn mem_reader_simple_arbiter::MemReaderSimpleArbiter<ADDR_W, DATA_W, u32:2>(
            n_mf_buf_mem_rd_req_r, n_mf_buf_mem_rd_resp_s, mf_buf_mem_rd_req_s, mf_buf_mem_rd_resp_r);

        // literals encoder
        spawn literal_encoder::LiteralsEncoder<ADDR_W, DATA_W>(
            le_req_r, le_resp_s, n_mf_buf_mem_rd_req_s[0], n_mf_buf_mem_rd_resp_r[0],
            lhw_output_mem_wr_req_s, lhw_output_mem_wr_data_s, lhw_output_mem_wr_resp_r,
            le_output_mem_wr_req_s, le_output_mem_wr_data_s, le_output_mem_wr_resp_r);

        // sequence encoder
        spawn sequence_encoder::SequenceEncoder<
            ADDR_W, DATA_W, FSE_TABLE_RAM_ADDR_W, FSE_CTABLE_RAM_DATA_W, FSE_CTABLE_RAM_NUM_PARTITIONS, FSE_TTABLE_RAM_DATA_W, FSE_TTABLE_RAM_NUM_PARTITIONS, BITSTREAM_BUFFER_W>(
            se_req_r, se_resp_s, n_mf_buf_mem_rd_req_s[1], n_mf_buf_mem_rd_resp_r[1],
            se_output_mem_wr_req_s, se_output_mem_wr_data_s, se_output_mem_wr_resp_r,
            ml_ctable_ram_rd_req_s, ml_ctable_ram_rd_resp_r, ll_ctable_ram_rd_req_s,
            ll_ctable_ram_rd_resp_r, of_ctable_ram_rd_req_s, of_ctable_ram_rd_resp_r,
            ml_ttable_ram_rd_req_s, ml_ttable_ram_rd_resp_r, ll_ttable_ram_rd_req_s,
            ll_ttable_ram_rd_resp_r, of_ttable_ram_rd_req_s, of_ttable_ram_rd_resp_r);

        (req_r, resp_s, mf_req_s, mf_resp_r, le_req_s, le_resp_r, se_req_s, se_resp_r)
    }

    init {  }

    next(state: ()) {
        type Addr = uN[ADDR_W];

        let tok = join();
        let (tok, req) = recv(tok, req_r);
        trace_fmt!("[CompressBlockEncoder] received compression request {:#x}", req);

        let tok = send(
            tok, mf_req_s,
            MfReq {
                input_addr: req.addr,
                input_size: req.size,
                output_lit_addr: LITERALS_BUFFER_AXI_ADDR as Addr,
                output_seq_addr: SEQUENCE_BUFFER_AXI_ADDR as Addr,
                zstd_params: MfParams { num_entries_log2: std::clog2(HT_SIZE) as uN[HT_SIZE_W] },
            });
        let (tok, mf_resp) = recv(tok, mf_resp_r);
        trace_fmt!("[CompressBlockEncoder] received Match Finder response {:#x}", mf_resp);

        let tok = send(
            tok, le_req_s,
            RawMemcopyReq {
                lit_addr: LITERALS_BUFFER_AXI_ADDR,
                lit_cnt: mf_resp.lit_cnt,
                out_addr: req.out_addr,
            });
        let (tok, le_resp) = recv(tok, le_resp_r);
        trace_fmt!("[CompressBlockEncoder] received Literals encoder response {:#x}", le_resp);

        let tok = send(
            tok, se_req_s,
            SeReq {
                addr: req.out_addr + le_resp.length,
                seq_addr: SEQUENCE_BUFFER_AXI_ADDR,
                seq_cnt: mf_resp.seq_cnt as u17,
            });
        let (tok, se_resp) = recv(tok, se_resp_r);

        trace_fmt!("[CompressBlockEncoder] received Sequence encoder response {:#x}", se_resp);

        let resp = Resp { length: le_resp.length + se_resp.length, status: Status::OK };
        trace_fmt!("[CompressBlockEncoder] sent back response {:#x}", resp);
        let tok = send(tok, resp_s, resp);
    }
}
