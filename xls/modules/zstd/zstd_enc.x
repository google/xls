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

// This file contains SIMPLIFIED implementation of ZstdEncoder
// for now assuming the following things
//    1. single frame
//    2. write N blocks of the same size (last is of different size)
//    3. the block data is RAW => so the produced file is just data + headers (1 + N headers)

import std;

import xls.examples.ram;
import xls.modules.zstd.match_finder;
import xls.modules.zstd.mem_copy;
import xls.modules.zstd.rle_block_encoder;
import xls.modules.zstd.frame_header_dec;
import xls.modules.zstd.frame_header_enc;
import xls.modules.zstd.block_header;
import xls.modules.zstd.memory.mem_writer;
import xls.modules.zstd.memory.mem_reader;
import xls.modules.zstd.mem_writer_simple_arbiter;
import xls.modules.zstd.mem_reader_simple_arbiter;
import xls.modules.zstd.common;
import xls.modules.zstd.memory.axi;
import xls.modules.zstd.block_size;
import xls.modules.zstd.comp_block_enc;

pub enum ZstdEncodeRespStatus : u1 {
    ERROR=0,
    OK=1
}

pub struct ZstdEncodeParams {
    enable_rle: bool,
    enable_compressed: bool
}

pub struct ZstdEncodeReq<ADDR_W: u32, DATA_W: u32> {
    input_offset: uN[ADDR_W], // bytes
    data_size: uN[DATA_W],
    output_offset: uN[ADDR_W], // bytes
    max_block_size: uN[DATA_W], //
    params: ZstdEncodeParams
}

pub struct ZstdEncodeResp<ADDR_W: u32> {
    status: ZstdEncodeRespStatus,
    written_bytes: uN[ADDR_W],
}

struct ZstdEncoderBlockWriterConf<ADDR_W: u32, DATA_W: u32> {
    bytes_left: u32,
    input_offset: uN[ADDR_W],
    output_offset: uN[ADDR_W],
    max_block_size: uN[DATA_W],
    params: ZstdEncodeParams
}

struct ZstdEncoderBlockWriterState<ADDR_W: u32, DATA_W: u32> {
    active: bool,
    written_bytes: uN[ADDR_W],
    conf: ZstdEncoderBlockWriterConf<ADDR_W, DATA_W>,
}

struct ZstdEncoderBlockWriterResp<ADDR_W: u32> {
    status: ZstdEncodeRespStatus,
    length: uN[ADDR_W],
}

const BLOCK_HEADER_LENGTH_BYTES = u32:3;
const ZSTD_WINDOW_ABSOLUTEMIN = u64:1024;

pub proc ZstdEncoderBlockWriter<ADDR_W: u32, DATA_W: u32> {
    type MemReaderReq = mem_reader::MemReaderReq<ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<DATA_W, ADDR_W>;
    type MemWriterReq = mem_writer::MemWriterReq<ADDR_W>;
    type MemWriterData = mem_writer::MemWriterDataPacket<DATA_W, ADDR_W>;
    type MemWriterResp = mem_writer::MemWriterResp;
    type MemWriterStatus = mem_writer::MemWriterRespStatus;

    type Resp = ZstdEncoderBlockWriterResp<ADDR_W>;
    type State = ZstdEncoderBlockWriterState<ADDR_W, DATA_W>;
    type Conf = ZstdEncoderBlockWriterConf<ADDR_W, DATA_W>;
    type Status = ZstdEncodeRespStatus;

    type BlockHeader = block_header::BlockHeader;
    type BlockHeaderWriterReq = block_header::BlockHeaderWriterReq<ADDR_W>;
    type BlockHeaderWriterResp = block_header::BlockHeaderWriterResp;
    type BlockHeaderWriterStatus = block_header::BlockHeaderWriterStatus;

    type RawMemcopyReq = mem_copy::RawMemcopyReq<ADDR_W>;
    type RawMemcopyResp = mem_copy::RawMemcopyResp<ADDR_W>;
    type RawMemcopyRespStatus = mem_copy::RawMemcopyStatus;

    type RleBlockEncoderReq = rle_block_encoder::RleBlockEncoderReq<ADDR_W>;
    type RleBlockEncoderResp = rle_block_encoder::RleBlockEncoderResp<ADDR_W>;
    type RleBlockEncoderStatus = rle_block_encoder::RleBlockEncoderStatus;

    type CompressBlockEncoderReq = comp_block_enc::CompressBlockEncoderReq<ADDR_W, DATA_W>;
    type CompressBlockEncoderResp = comp_block_enc::CompressBlockEncoderResp<ADDR_W>;
    type CompressBlockEncoderStatus = comp_block_enc::CompressBlockEncoderStatus;

    type Addr = uN[ADDR_W];
    type Data = uN[DATA_W];
    type BlockType = common::BlockType;
    type BlockSize = common::BlockSize;

    // parent I/O
    conf_r: chan<Conf> in;
    resp_s: chan<Resp> out;

    // block header
    bhw_req_s: chan<BlockHeaderWriterReq> out;
    bhw_resp_r: chan<BlockHeaderWriterResp> in;

    // block types
    raw_req_s: chan<RawMemcopyReq> out;
    raw_resp_r: chan<RawMemcopyResp> in;
    rle_req_s: chan<RleBlockEncoderReq> out;
    rle_resp_r: chan<RleBlockEncoderResp> in;
    cbe_req_s: chan<CompressBlockEncoderReq> out;
    cbe_resp_r: chan<CompressBlockEncoderResp> in;

    // general memory writing
    rle_mem_wr_req_s: chan<MemWriterReq> out;
    rle_mem_wr_data_s: chan<MemWriterData> out;
    rle_mem_wr_resp_r: chan<MemWriterResp> in;

    init { zero!<State>() }
    config(
        conf_r: chan<Conf> in,
        resp_s: chan<Resp> out,
        bhw_req_s: chan<BlockHeaderWriterReq> out,
        bhw_resp_r: chan<BlockHeaderWriterResp> in,
        raw_req_s: chan<RawMemcopyReq> out,
        raw_resp_r: chan<RawMemcopyResp> in,
        rle_req_s: chan<RleBlockEncoderReq> out,
        rle_resp_r: chan<RleBlockEncoderResp> in,
        cbe_req_s: chan<CompressBlockEncoderReq> out,
        cbe_resp_r: chan<CompressBlockEncoderResp> in,
        rle_mem_wr_req_s: chan<MemWriterReq> out,
        rle_mem_wr_data_s: chan<MemWriterData> out,
        rle_mem_wr_resp_r: chan<MemWriterResp> in
    ) {
        (
            conf_r, resp_s,
            bhw_req_s, bhw_resp_r,
            raw_req_s, raw_resp_r,
            rle_req_s, rle_resp_r,
            cbe_req_s, cbe_resp_r,
            rle_mem_wr_req_s, rle_mem_wr_data_s, rle_mem_wr_resp_r
        )
    }
    next(state: State) {
        if !state.active {
            let (tok, conf) = recv(join(), conf_r);
            State {
                active: true,
                conf: conf,
                written_bytes: Addr:0,
            }
        } else {
            let tok = join();
            let conf = state.conf;
            let size = block_size::get_block_size(conf.bytes_left, conf.max_block_size as BlockSize);
            // BlockSize calculation
            // 1. Limit BlockSize value to rfc-defined and parameter-based max value,
            // and reduce it further in case there's not much data left to compress
            //    * done by get_block_size
            // 2. Try to split the input data by half or by fourth (TODO)
            //    * use samples of data from the beginning, middle and the end of the input
            //    * zstd C implementation uses 512 bytes from each of these parts
            //    * fingerprint comparison - calculate differences between the regions and split in halves
            //      in case the beginning and end differ significantly (with a threshold)
            //    * repeat the operation to split in fourths
            //    * the result is BlockSize, one of 32/64/96/128 KB depending on the differences in fingerprints
            //    * see `ZSTD_splitBlock_fromBorders` and `compareFingerprints` in zstd for reference
            //    * https://github.com/facebook/zstd/blob/d654fca78690fa15cceb8058ac47454d914a0e63/lib/compress/zstd_preSplit.c#L198
            //    * https://github.com/facebook/zstd/blob/d654fca78690fa15cceb8058ac47454d914a0e63/lib/compress/zstd_preSplit.c#L110
            let last_block: bool = state.conf.bytes_left <= size as u32;

            // step 1: choose block type
            let (
                tok,
                btype,
                rle_symbol
            ) =

            if conf.params.enable_compressed {
                (tok, BlockType::COMPRESSED, u8: 0)
            } else if conf.params.enable_rle {
                let tok = send(tok, rle_req_s, RleBlockEncoderReq {
                    addr: conf.input_offset,
                    length: size as uN[ADDR_W]
                });

                let (tok, rle_resp) = recv(tok, rle_resp_r);
                let btype = if rle_resp.status == RleBlockEncoderStatus::OK {
                    BlockType::RLE
                } else {
                    BlockType::RAW
                };

                (tok, btype, rle_resp.symbol)
            } else {
                (tok, BlockType::RAW, u8: 0)
            };

            // step 2: write block content
            let block_content_offset = conf.output_offset + BLOCK_HEADER_LENGTH_BYTES;
            let (tok2, blwr_status, block_content_size, block_header_size_field) = match btype {
                BlockType::RLE => {
                    let tok2 = send(tok, rle_mem_wr_req_s, MemWriterReq {
                        addr: block_content_offset, length: Addr:1
                    });
                    let tok2 = send(tok2, rle_mem_wr_data_s, MemWriterData {
                        data: rle_symbol as Data,
                        length: Addr:1,
                        last: true
                    });
                    trace_fmt!("writing rle pair: {:#x} -> {:#x} (symbol: {:#x} size: {})", conf.input_offset, block_content_offset, rle_symbol, size);
                    let (tok2, rle_resp) = recv(tok2, rle_mem_wr_resp_r);
                    let out_status =  if rle_resp.status == MemWriterStatus::OKAY { Status::OK } else { Status::ERROR };
                    (tok2, out_status, Addr:1, size as Addr)
                },
                BlockType::RAW => {
                    let tok2 = send(tok, raw_req_s, RawMemcopyReq {
                        lit_addr: conf.input_offset,
                        lit_cnt: size as u32,
                        out_addr: block_content_offset
                    });
                    trace_fmt!("raw copying: {:#x} -> {:#x} (size: {})", conf.input_offset, block_content_offset, size);
                    let (tok2, memcpy_resp) = recv(tok2, raw_resp_r);
                    trace_fmt!("received raw copy resp");
                    let out_status = if memcpy_resp.status == RawMemcopyRespStatus::OK { Status::OK } else { Status::ERROR };
                    (tok2, out_status, size as Addr, size as Addr)
                },
                BlockType::COMPRESSED => {
                    let tok2 = send(tok, cbe_req_s, CompressBlockEncoderReq {
                        addr: conf.input_offset,
                        out_addr: block_content_offset,
                        size: size as Addr
                    });
                    let (tok2, cbe_resp) = recv(tok2, cbe_resp_r);
                    let out_status = if cbe_resp.status == CompressBlockEncoderStatus::OK { Status::OK } else { Status::ERROR };
                    (tok2, out_status, cbe_resp.length, cbe_resp.length)
                },
                _ => {
                    trace_fmt!("Unsupported block type");
                    (tok, Status::ERROR, Addr:0, Addr:0)
                }
            };

            // step 3: write block header
            let block_header_req = BlockHeaderWriterReq{
                addr: conf.output_offset,
                header: BlockHeader {
                    last: last_block,
                    size: block_header_size_field as u21,
                    btype: btype
                }
            };
            let tok1 = send_if(tok, bhw_req_s, blwr_status == Status::OK, block_header_req);
            trace_fmt!("writing block header to {:#x} == {}", conf.output_offset, block_header_req);

            let (tok1, bhw_resp) = recv_if(tok1, bhw_resp_r,  blwr_status == Status::OK, zero!<BlockHeaderWriterResp>());

            let status = if blwr_status == Status::OK && bhw_resp.status == BlockHeaderWriterStatus::OKAY {
                Status::OK
            } else {
                trace_fmt!("failed writing block: {} {}", bhw_resp, blwr_status);
                Status::ERROR
            };

            let written_bytes = block_content_size + BLOCK_HEADER_LENGTH_BYTES;
            let total_written_bytes = state.written_bytes + written_bytes as Addr;
            if last_block || status != ZstdEncodeRespStatus::OK {
                let tok = send(join(tok1, tok2), resp_s, Resp {status, length: total_written_bytes});
                zero!<State>()
            } else {
                State {
                    active: true,
                    written_bytes: total_written_bytes,
                    conf: Conf {
                        bytes_left: conf.bytes_left - size as u32,
                        input_offset: conf.input_offset as Addr + size as Addr,
                        output_offset: conf.output_offset as Addr + written_bytes as Addr,
                        ..conf
                    }
                }
            }
        }
    }
}

pub proc ZstdEncoder<
    ADDR_W: u32, DATA_W: u32,
    // rle
    RLE_HEURISTIC_SAMPLE_COUNT: u32,
    // compressed blocks params
    HB_SIZE: u32, HB_DATA_W: u32, HB_OFFSET_W: u32, HB_RAM_ADDR_W: u32, HB_RAM_DATA_W: u32, HB_RAM_NUM: u32, HB_RAM_NUM_PARTITIONS: u32,
    HT_SIZE: u32, HT_KEY_W: u32, HT_VALUE_W: u32, HT_SIZE_W: u32, HT_HASH_W: u32, HT_RAM_DATA_W: u32, HT_RAM_NUM_PARTITIONS: u32,
    MIN_SEQ_LEN: u32, LITERALS_BUFFER_AXI_ADDR: u32, SEQUENCE_BUFFER_AXI_ADDR: u32,
    FSE_TABLE_RAM_ADDR_W: u32, FSE_CTABLE_RAM_DATA_W: u32, FSE_TTABLE_RAM_DATA_W: u32,
    FSE_CTABLE_RAM_NUM_PARTITIONS: u32, FSE_TTABLE_RAM_NUM_PARTITIONS: u32,
    FSE_BITSTREAM_BUFFER_W: u32
> {
    type MemReaderReq = mem_reader::MemReaderReq<ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<DATA_W, ADDR_W>;
    type MemWriterReq = mem_writer::MemWriterReq<ADDR_W>;
    type MemWriterData = mem_writer::MemWriterDataPacket<DATA_W, ADDR_W>;
    type MemWriterResp = mem_writer::MemWriterResp;
    type MemWriterStatus = mem_writer::MemWriterRespStatus;

    type Req = ZstdEncodeReq<ADDR_W, DATA_W>;
    type Resp = ZstdEncodeResp<ADDR_W>;
    type ZstdEncoderBlockWriterConf = ZstdEncoderBlockWriterConf<ADDR_W, DATA_W>;
    type ZstdEncoderBlockWriterResp = ZstdEncoderBlockWriterResp<ADDR_W>;
    type Status = ZstdEncodeRespStatus;

    type FrameHeader = frame_header_dec::FrameHeader;
    type FrameHeaderEncoderReq = frame_header_enc::FrameHeaderEncoderReq<ADDR_W>;
    type FrameHeaderEncoderResp = frame_header_enc::FrameHeaderEncoderResp;
    type FrameHeaderEncoderStatus = frame_header_enc::FrameHeaderEncoderStatus;

    type BlockHeader = block_header::BlockHeader;
    type BlockHeaderWriterReq = block_header::BlockHeaderWriterReq<ADDR_W>;
    type BlockHeaderWriterResp = block_header::BlockHeaderWriterResp;
    type BlockHeaderWriterStatus = block_header::BlockHeaderWriterStatus;

    type RawMemcopyReq = mem_copy::RawMemcopyReq<ADDR_W>;
    type RawMemcopyResp = mem_copy::RawMemcopyResp<ADDR_W>;
    type RawMemcopyRespStatus = mem_copy::RawMemcopyStatus;

    type RleBlockEncoderReq = rle_block_encoder::RleBlockEncoderReq<ADDR_W>;
    type RleBlockEncoderResp = rle_block_encoder::RleBlockEncoderResp<ADDR_W>;
    type RleBlockEncoderStatus = rle_block_encoder::RleBlockEncoderStatus;

    type CompressBlockEncoderReq = comp_block_enc::CompressBlockEncoderReq<ADDR_W, DATA_W>;
    type CompressBlockEncoderResp = comp_block_enc::CompressBlockEncoderResp<ADDR_W>;
    type CompressBlockEncoderStatus = comp_block_enc::CompressBlockEncoderStatus;

    type MfResp = match_finder::MatchFinderResp;
    type MfReq = match_finder::MatchFinderReq<HT_SIZE_W, ADDR_W>;
    type MfRespStatus = match_finder::MatchFinderRespStatus;
    type HistoryBufferRamRdReq = ram::ReadReq<HB_RAM_ADDR_W, HB_RAM_NUM_PARTITIONS>;
    type HistoryBufferRamRdResp = ram::ReadResp<HB_RAM_DATA_W>;
    type HistoryBufferRamWrReq = ram::WriteReq<HB_RAM_ADDR_W, HB_RAM_DATA_W, HB_RAM_NUM_PARTITIONS>;
    type HistoryBufferRamWrResp = ram::WriteResp;
    type HashTableRamRdReq = ram::ReadReq<HT_HASH_W, HT_RAM_NUM_PARTITIONS>;
    type HashTableRamRdResp = ram::ReadResp<HT_RAM_DATA_W>;
    type HashTableRamWrReq = ram::WriteReq<HT_HASH_W, HT_RAM_DATA_W, HT_RAM_NUM_PARTITIONS>;
    type HashTableRamWrResp = ram::WriteResp;

    type CTableRamRdReq = ram::ReadReq<FSE_TABLE_RAM_ADDR_W, FSE_CTABLE_RAM_NUM_PARTITIONS>;
    type CTableRamRdResp = ram::ReadResp<FSE_CTABLE_RAM_DATA_W>;
    type TTableRamRdReq = ram::ReadReq<FSE_TABLE_RAM_ADDR_W, FSE_TTABLE_RAM_NUM_PARTITIONS>;
    type TTableRamRdResp = ram::ReadResp<FSE_TTABLE_RAM_DATA_W>;

    // from
    enc_req: chan<Req> in;
    enc_resp_s: chan<Resp> out;

    // communication
    conf_s: chan<ZstdEncoderBlockWriterConf> out;
    bw_resp_r: chan<ZstdEncoderBlockWriterResp> in;
    fhw_req_s: chan<FrameHeaderEncoderReq> out;
    fhw_resp_r: chan<FrameHeaderEncoderResp> in;

    init { }

    config(
        enc_req_r: chan<Req> in,
        enc_resp_s: chan<Resp> out,

        // writers
        fhw_mem_wr_req_s: chan<MemWriterReq> out,
        fhw_mem_wr_data_s: chan<MemWriterData> out,
        fhw_mem_wr_resp_r: chan<MemWriterResp> in,
        bhw_mem_wr_req_s: chan<MemWriterReq> out,
        bhw_mem_wr_data_s: chan<MemWriterData> out,
        bhw_mem_wr_resp_r: chan<MemWriterResp> in,
        braw_mem_wr_req_s: chan<MemWriterReq> out,
        braw_mem_wr_data_s: chan<MemWriterData> out,
        braw_mem_wr_resp_r: chan<MemWriterResp> in,
        brle_mem_wr_req_s: chan<MemWriterReq> out,
        brle_mem_wr_data_s: chan<MemWriterData> out,
        brle_mem_wr_resp_r: chan<MemWriterResp> in,
        bcomp_lhw_mem_wr_req_s: chan<MemWriterReq> out,
        bcomp_lhw_mem_wr_data_s: chan<MemWriterData> out,
        bcomp_lhw_mem_wr_resp_r: chan<MemWriterResp> in,
        bcomp_le_mem_wr_req_s: chan<MemWriterReq> out,
        bcomp_le_mem_wr_data_s: chan<MemWriterData> out,
        bcomp_le_mem_wr_resp_r: chan<MemWriterResp> in,
        bcomp_se_mem_wr_req_s: chan<MemWriterReq> out,
        bcomp_se_mem_wr_data_s: chan<MemWriterData> out,
        bcomp_se_mem_wr_resp_r: chan<MemWriterResp> in,
        bcomp_mf_buf_mem_wr_req_s: chan<MemWriterReq> out,
        bcomp_mf_buf_mem_wr_data_s: chan<MemWriterData> out,
        bcomp_mf_buf_mem_wr_resp_r: chan<MemWriterResp> in,

        // readers
        braw_mem_rd_req_s: chan<MemReaderReq> out,
        braw_mem_rd_resp_r: chan<MemReaderResp> in,
        brle_mem_rd_req_s: chan<MemReaderReq> out,
        brle_mem_rd_resp_r: chan<MemReaderResp> in,
        // should be a separate buffer
        bcomp_mf_mem_rd_req_s: chan<MemReaderReq> out,
        bcomp_mf_mem_rd_resp_r: chan<MemReaderResp> in,
        bcomp_mf_buf_mem_rd_req_s: chan<MemReaderReq> out,
        bcomp_mf_buf_mem_rd_resp_r: chan<MemReaderResp> in,

        // buffers (match finder, history buffer, hash table)
        hb_ram_rd_req_s: chan<HistoryBufferRamRdReq>[HB_RAM_NUM] out,
        hb_ram_rd_resp_r: chan<HistoryBufferRamRdResp>[HB_RAM_NUM] in,
        hb_ram_wr_req_s: chan<HistoryBufferRamWrReq>[HB_RAM_NUM] out,
        hb_ram_wr_resp_r: chan<HistoryBufferRamWrResp>[HB_RAM_NUM] in,
        ht_ram_rd_req_s: chan<HashTableRamRdReq> out,
        ht_ram_rd_resp_r: chan<HashTableRamRdResp> in,
        ht_ram_wr_req_s: chan<HashTableRamWrReq> out,
        ht_ram_wr_resp_r: chan<HashTableRamWrResp>in,
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
        of_ttable_ram_rd_resp_r: chan<TTableRamRdResp> in
    ) {
        // internal
        let (conf_s, conf_r) = chan<ZstdEncoderBlockWriterConf, u32:1>("conf");
        let (bw_resp_s, bw_resp_r) = chan<ZstdEncoderBlockWriterResp, u32:1>("bw_resp");

        // headers
        let (fhw_req_s, fhw_req_r) = chan<FrameHeaderEncoderReq, u32:1>("bhw_req");
        let (fhw_resp_s, fhw_resp_r) = chan<FrameHeaderEncoderResp, u32:1>("bhw_resp");
        let (bhw_req_s, bhw_req_r) = chan<BlockHeaderWriterReq, u32:1>("bhw_req");
        let (bhw_resp_s, bhw_resp_r) = chan<BlockHeaderWriterResp, u32:1>("bhw_resp");

        // block types
        let (raw_req_s, raw_req_r) = chan<RawMemcopyReq, u32:1>("braw_req");
        let (raw_resp_s, raw_resp_r) = chan<RawMemcopyResp, u32:1>("braw_resp");
        let (rle_req_s, rle_req_r) = chan<RleBlockEncoderReq, u32:1>("brle_req");
        let (rle_resp_s, rle_resp_r) = chan<RleBlockEncoderResp, u32:1>("brle_resp");
        let (cbe_req_s, cbe_req_r) = chan<CompressBlockEncoderReq, u32:1>("cbe_req");
        let (cbe_resp_s, cbe_resp_r) = chan<CompressBlockEncoderResp, u32:1>("cbe_resp");

        spawn ZstdEncoderBlockWriter<ADDR_W, DATA_W>
        (
            conf_r, bw_resp_s,
            bhw_req_s, bhw_resp_r,
            raw_req_s, raw_resp_r,
            rle_req_s, rle_resp_r,
            cbe_req_s, cbe_resp_r,
            brle_mem_wr_req_s, brle_mem_wr_data_s, brle_mem_wr_resp_r
        );

        spawn frame_header_enc::FrameHeaderEncoder<DATA_W, ADDR_W>
        (
            fhw_req_r, fhw_resp_s,
            fhw_mem_wr_req_s, fhw_mem_wr_data_s, fhw_mem_wr_resp_r
        );

        spawn block_header::BlockHeaderWriter<DATA_W, ADDR_W>
        (
            bhw_req_r, bhw_resp_s,
            bhw_mem_wr_req_s, bhw_mem_wr_data_s, bhw_mem_wr_resp_r
        );

        spawn mem_copy::RawMemcopy<ADDR_W, DATA_W>
        (
            raw_req_r, raw_resp_s,
            braw_mem_rd_req_s, braw_mem_rd_resp_r,
            braw_mem_wr_req_s, braw_mem_wr_data_s, braw_mem_wr_resp_r
        );

        spawn rle_block_encoder::RleBlockEncoder<ADDR_W, DATA_W, ADDR_W, RLE_HEURISTIC_SAMPLE_COUNT>
        (
            rle_req_r, rle_resp_s,
            brle_mem_rd_req_s, brle_mem_rd_resp_r
        );

        spawn comp_block_enc::CompressBlockEncoder<
            ADDR_W, DATA_W,
            HB_SIZE, HB_DATA_W, HB_OFFSET_W, HB_RAM_ADDR_W, HB_RAM_DATA_W, HB_RAM_NUM, HB_RAM_NUM_PARTITIONS,
            HT_SIZE, HT_KEY_W, HT_VALUE_W, HT_SIZE_W, HT_HASH_W, HT_RAM_DATA_W, HT_RAM_NUM_PARTITIONS,
            MIN_SEQ_LEN, LITERALS_BUFFER_AXI_ADDR, SEQUENCE_BUFFER_AXI_ADDR,
            FSE_TABLE_RAM_ADDR_W, FSE_CTABLE_RAM_DATA_W, FSE_TTABLE_RAM_DATA_W,
            FSE_CTABLE_RAM_NUM_PARTITIONS, FSE_TTABLE_RAM_NUM_PARTITIONS,
            FSE_BITSTREAM_BUFFER_W
        > (
            cbe_req_r, cbe_resp_s,
            bcomp_mf_mem_rd_req_s, bcomp_mf_mem_rd_resp_r,
            bcomp_mf_buf_mem_rd_req_s, bcomp_mf_buf_mem_rd_resp_r,
            bcomp_mf_buf_mem_wr_req_s, bcomp_mf_buf_mem_wr_data_s, bcomp_mf_buf_mem_wr_resp_r,
            hb_ram_rd_req_s, hb_ram_rd_resp_r,
            hb_ram_wr_req_s, hb_ram_wr_resp_r,
            ht_ram_rd_req_s, ht_ram_rd_resp_r,
            ht_ram_wr_req_s, ht_ram_wr_resp_r,
            bcomp_lhw_mem_wr_req_s, bcomp_lhw_mem_wr_data_s, bcomp_lhw_mem_wr_resp_r,
            bcomp_le_mem_wr_req_s, bcomp_le_mem_wr_data_s, bcomp_le_mem_wr_resp_r,
            bcomp_se_mem_wr_req_s, bcomp_se_mem_wr_data_s, bcomp_se_mem_wr_resp_r,
            ml_ctable_ram_rd_req_s, ml_ctable_ram_rd_resp_r,
            ll_ctable_ram_rd_req_s, ll_ctable_ram_rd_resp_r,
            of_ctable_ram_rd_req_s, of_ctable_ram_rd_resp_r,
            ml_ttable_ram_rd_req_s, ml_ttable_ram_rd_resp_r,
            ll_ttable_ram_rd_req_s, ll_ttable_ram_rd_resp_r,
            of_ttable_ram_rd_req_s, of_ttable_ram_rd_resp_r
        );

        (
            enc_req_r, enc_resp_s,
            conf_s, bw_resp_r,
            fhw_req_s, fhw_resp_r
        )
    }

    next(state: ()) {
        let (tok, request) = recv(join(), enc_req);
        let window_size = ZSTD_WINDOW_ABSOLUTEMIN; // TODO: Calculate the window size based on the frame content
        trace_fmt!("writing frame header to {:#x}", request.output_offset);

        let tok = send(tok, fhw_req_s, FrameHeaderEncoderReq {
            addr: request.output_offset,
            window_log: u5:22, // TODO: Calculate window log based on window size
            src_size: request.data_size as u64,
            dict_id: u32:0,
            max_block_size: request.max_block_size as u64,
            provide_dict_id: false,
            provide_checksum: false,
            provide_content_size: true,
            provide_window_size: true,
        });
        let (tok, fhw_resp) = recv(tok, fhw_resp_r);
        trace_fmt!("written frame header {:#x}", fhw_resp);

        if (fhw_resp.status != FrameHeaderEncoderStatus::OKAY) {
            trace_fmt!("failed writing frame header");
            send(tok, enc_resp_s, ZstdEncodeResp{
                status: ZstdEncodeRespStatus::ERROR,
                written_bytes: uN[ADDR_W]:0
            });
        } else {
            let tok = send(tok, conf_s, ZstdEncoderBlockWriterConf {
                bytes_left: request.data_size as u32,
                input_offset: request.input_offset,
                output_offset: request.output_offset + fhw_resp.length as uN[ADDR_W],
                max_block_size: request.max_block_size,
                params: request.params
            });
            let (tok, bw_resp) = recv(tok, bw_resp_r);
            let resp = ZstdEncodeResp{
                status: bw_resp.status,
                written_bytes: fhw_resp.length as uN[ADDR_W] + bw_resp.length,
            };
            send(tok, enc_resp_s, resp);
        };
    }
}


const INST_ADDR_W = u32:32;
const INST_DATA_W = u32:64;
const INST_RLE_HEURISTIC_SAMPLE_COUNT = u32:8;
const INST_DATA_W_LOG2 = std::clog2(INST_DATA_W + u32:1);
const INST_DEST_W = u32:8;
const INST_ID_W = u32:8;

// sequence encoding
const INST_MIN_SEQ_LEN = u32:3;
const INST_HT_SIZE = u32:512;
const INST_HT_SIZE_W = std::clog2(INST_HT_SIZE + u32:1);
const INST_HT_KEY_W = u32:32;
const INST_HT_VALUE_W = INST_HT_KEY_W + INST_ADDR_W;
const INST_HT_HASH_W = std::clog2(INST_HT_SIZE);
const INST_HT_RAM_DATA_W = INST_HT_VALUE_W + u32:1;
const INST_HT_RAM_WORD_PARTITION_SIZE = u32:1;
const INST_HT_RAM_NUM_PARTITIONS = ram::num_partitions(INST_HT_RAM_WORD_PARTITION_SIZE, INST_HT_RAM_DATA_W);
const INST_HT_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR = ram::SimultaneousReadWriteBehavior::READ_BEFORE_WRITE;
const INST_HT_RAM_INITIALIZED = true;

const INST_LITERALS_BUFFER_AXI_ADDR = u32:0x10000;
const INST_SEQUENCE_BUFFER_AXI_ADDR = u32:0x20000;
const INST_HB_DATA_W = u32:64;
const INST_HB_SIZE = u32:1024;
const INST_HB_OFFSET_W = std::clog2(INST_HB_SIZE);
const INST_HB_RAM_NUM = u32:8;
const INST_HB_RAM_SIZE = INST_HB_SIZE / INST_HB_RAM_NUM;
const INST_HB_RAM_DATA_W = INST_HB_DATA_W / INST_HB_RAM_NUM;
const INST_HB_RAM_ADDR_W = std::clog2(INST_HB_RAM_SIZE);
const INST_HB_RAM_PARTITION_SIZE = INST_HB_RAM_DATA_W;
const INST_HB_RAM_NUM_PARTITIONS = ram::num_partitions(INST_HB_RAM_PARTITION_SIZE, INST_HB_RAM_DATA_W);
const INST_FSE_TABLE_RAM_ADDR_W = u32:32;
const INST_FSE_CTABLE_RAM_DATA_W = u32:16;
const INST_FSE_TTABLE_RAM_DATA_W = u32:64;
const INST_FSE_TTABLE_RAM_PARTITION_SIZE = u32:8;
const INST_FSE_TABLE_PARTITION_SIZE = u32:8;
const INST_FSE_CTABLE_RAM_NUM_PARTITIONS = ram::num_partitions(INST_FSE_TABLE_PARTITION_SIZE, INST_FSE_CTABLE_RAM_DATA_W);
const INST_FSE_TTABLE_RAM_NUM_PARTITIONS = ram::num_partitions(INST_FSE_TABLE_PARTITION_SIZE, INST_FSE_TTABLE_RAM_DATA_W);
const INST_FSE_BITSTREAM_BUFFER_W = u32:1024;

proc ZstdEncoderInst {
    type MemReaderReq = mem_reader::MemReaderReq<INST_ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<INST_DATA_W, INST_ADDR_W>;
    type MemWriterReq = mem_writer::MemWriterReq<INST_ADDR_W>;
    type MemWriterData = mem_writer::MemWriterDataPacket<INST_DATA_W, INST_ADDR_W>;
    type MemWriterResp = mem_writer::MemWriterResp;
    type MemWriterStatus = mem_writer::MemWriterRespStatus;
    type HistoryBufferRamRdReq = ram::ReadReq<INST_HB_RAM_ADDR_W, INST_HB_RAM_NUM_PARTITIONS>;
    type HistoryBufferRamRdResp = ram::ReadResp<INST_HB_RAM_DATA_W>;
    type HistoryBufferRamWrReq = ram::WriteReq<INST_HB_RAM_ADDR_W, INST_HB_RAM_DATA_W, INST_HB_RAM_NUM_PARTITIONS>;
    type HistoryBufferRamWrResp = ram::WriteResp;
    type HashTableRamRdReq = ram::ReadReq<INST_HT_HASH_W, INST_HT_RAM_NUM_PARTITIONS>;
    type HashTableRamRdResp = ram::ReadResp<INST_HT_RAM_DATA_W>;
    type HashTableRamWrReq = ram::WriteReq<INST_HT_HASH_W, INST_HT_RAM_DATA_W, INST_HT_RAM_NUM_PARTITIONS>;
    type HashTableRamWrResp = ram::WriteResp;
    type CTableRamRdReq = ram::ReadReq<INST_FSE_TABLE_RAM_ADDR_W, INST_FSE_CTABLE_RAM_NUM_PARTITIONS>;
    type CTableRamRdResp = ram::ReadResp<INST_FSE_CTABLE_RAM_DATA_W>;
    type TTableRamRdReq = ram::ReadReq<INST_FSE_TABLE_RAM_ADDR_W, INST_FSE_TTABLE_RAM_NUM_PARTITIONS>;
    type TTableRamRdResp = ram::ReadResp<INST_FSE_TTABLE_RAM_DATA_W>;
    type Req = ZstdEncodeReq<INST_ADDR_W, INST_DATA_W>;
    type Resp = ZstdEncodeResp<INST_ADDR_W>;

    init {}

    config(
        enc_req_r: chan<Req> in,
        enc_resp_s: chan<Resp> out,
        fhw_mem_wr_req_s: chan<MemWriterReq> out,
        fhw_mem_wr_data_s: chan<MemWriterData> out,
        fhw_mem_wr_resp_r: chan<MemWriterResp> in,
        bhw_mem_wr_req_s: chan<MemWriterReq> out,
        bhw_mem_wr_data_s: chan<MemWriterData> out,
        bhw_mem_wr_resp_r: chan<MemWriterResp> in,
        braw_mem_wr_req_s: chan<MemWriterReq> out,
        braw_mem_wr_data_s: chan<MemWriterData> out,
        braw_mem_wr_resp_r: chan<MemWriterResp> in,
        brle_mem_wr_req_s: chan<MemWriterReq> out,
        brle_mem_wr_data_s: chan<MemWriterData> out,
        brle_mem_wr_resp_r: chan<MemWriterResp> in,
        bcomp_lhw_mem_wr_req_s: chan<MemWriterReq> out,
        bcomp_lhw_mem_wr_data_s: chan<MemWriterData> out,
        bcomp_lhw_mem_wr_resp_r: chan<MemWriterResp> in,
        bcomp_le_output_mem_wr_req_s: chan<MemWriterReq> out,
        bcomp_le_output_mem_wr_data_s: chan<MemWriterData> out,
        bcomp_le_output_mem_wr_resp_r: chan<MemWriterResp> in,
        bcomp_se_output_mem_wr_req_s: chan<MemWriterReq> out,
        bcomp_se_output_mem_wr_data_s: chan<MemWriterData> out,
        bcomp_se_output_mem_wr_resp_r: chan<MemWriterResp> in,
        bcomp_mf_buf_mem_wr_req_s: chan<MemWriterReq> out,
        bcomp_mf_buf_mem_wr_data_s: chan<MemWriterData> out,
        bcomp_mf_buf_mem_wr_resp_r: chan<MemWriterResp> in,
        braw_mem_rd_req_s: chan<MemReaderReq> out,
        braw_mem_rd_resp_r: chan<MemReaderResp> in,
        brle_mem_rd_req_s: chan<MemReaderReq> out,
        brle_mem_rd_resp_r: chan<MemReaderResp> in,
        bcomp_mf_mem_rd_req_s: chan<MemReaderReq> out,
        bcomp_mf_mem_rd_resp_r: chan<MemReaderResp> in,
        bcomp_mf_buf_mem_rd_req_s: chan<MemReaderReq> out,
        bcomp_mf_buf_mem_rd_resp_r: chan<MemReaderResp> in,
        hb_ram_rd_req_s: chan<HistoryBufferRamRdReq>[INST_HB_RAM_NUM] out,
        hb_ram_rd_resp_r: chan<HistoryBufferRamRdResp>[INST_HB_RAM_NUM] in,
        hb_ram_wr_req_s: chan<HistoryBufferRamWrReq>[INST_HB_RAM_NUM] out,
        hb_ram_wr_resp_r: chan<HistoryBufferRamWrResp>[INST_HB_RAM_NUM] in,
        ht_ram_rd_req_s: chan<HashTableRamRdReq> out,
        ht_ram_rd_resp_r: chan<HashTableRamRdResp> in,
        ht_ram_wr_req_s: chan<HashTableRamWrReq> out,
        ht_ram_wr_resp_r: chan<HashTableRamWrResp>in,
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
        of_ttable_ram_rd_resp_r: chan<TTableRamRdResp> in
    ) {
        spawn  ZstdEncoder<INST_ADDR_W, INST_DATA_W,
            INST_RLE_HEURISTIC_SAMPLE_COUNT,
            INST_HB_SIZE, INST_HB_DATA_W, INST_HB_OFFSET_W, INST_HB_RAM_ADDR_W, INST_HB_RAM_DATA_W, INST_HB_RAM_NUM, INST_HB_RAM_NUM_PARTITIONS,
            INST_HT_SIZE, INST_HT_KEY_W, INST_HT_VALUE_W, INST_HT_SIZE_W, INST_HT_HASH_W, INST_HT_RAM_DATA_W, INST_HT_RAM_NUM_PARTITIONS,
            INST_MIN_SEQ_LEN, INST_LITERALS_BUFFER_AXI_ADDR, INST_SEQUENCE_BUFFER_AXI_ADDR,
            INST_FSE_TABLE_RAM_ADDR_W, INST_FSE_CTABLE_RAM_DATA_W, INST_FSE_TTABLE_RAM_DATA_W, INST_FSE_CTABLE_RAM_NUM_PARTITIONS, INST_FSE_TTABLE_RAM_NUM_PARTITIONS,
            INST_FSE_BITSTREAM_BUFFER_W
        > (
            enc_req_r, enc_resp_s,
            fhw_mem_wr_req_s, fhw_mem_wr_data_s, fhw_mem_wr_resp_r,
            bhw_mem_wr_req_s, bhw_mem_wr_data_s, bhw_mem_wr_resp_r,
            braw_mem_wr_req_s,braw_mem_wr_data_s,braw_mem_wr_resp_r,
            brle_mem_wr_req_s,brle_mem_wr_data_s,brle_mem_wr_resp_r,
            bcomp_lhw_mem_wr_req_s, bcomp_lhw_mem_wr_data_s, bcomp_lhw_mem_wr_resp_r,
            bcomp_le_output_mem_wr_req_s, bcomp_le_output_mem_wr_data_s, bcomp_le_output_mem_wr_resp_r,
            bcomp_se_output_mem_wr_req_s, bcomp_se_output_mem_wr_data_s, bcomp_se_output_mem_wr_resp_r,
            bcomp_mf_buf_mem_wr_req_s, bcomp_mf_buf_mem_wr_data_s, bcomp_mf_buf_mem_wr_resp_r,
            braw_mem_rd_req_s, braw_mem_rd_resp_r,
            brle_mem_rd_req_s, brle_mem_rd_resp_r,
            bcomp_mf_mem_rd_req_s, bcomp_mf_mem_rd_resp_r,
            bcomp_mf_buf_mem_rd_req_s, bcomp_mf_buf_mem_rd_resp_r,
            hb_ram_rd_req_s, hb_ram_rd_resp_r, hb_ram_wr_req_s, hb_ram_wr_resp_r,
            ht_ram_rd_req_s, ht_ram_rd_resp_r, ht_ram_wr_req_s, ht_ram_wr_resp_r,
            ml_ctable_ram_rd_req_s, ml_ctable_ram_rd_resp_r,
            ll_ctable_ram_rd_req_s, ll_ctable_ram_rd_resp_r,
            of_ctable_ram_rd_req_s, of_ctable_ram_rd_resp_r,
            ml_ttable_ram_rd_req_s, ml_ttable_ram_rd_resp_r,
            ll_ttable_ram_rd_req_s, ll_ttable_ram_rd_resp_r,
            of_ttable_ram_rd_req_s, of_ttable_ram_rd_resp_r
        );
    }

    next(state: ()) { }
}


const COCOTB_ADDR_W = u32:32;
const COCOTB_DATA_W = u32:32;
const COCOTB_RLE_HEURISTIC_SAMPLE_COUNT = u32:8;
const COCOTB_DATA_W_LOG2 = std::clog2(COCOTB_DATA_W + u32:1);
const COCOTB_DEST_W = u32:8;
const COCOTB_ID_W = u32:8;
const COCOTB_RAM_PARTITION_SIZE = COCOTB_DATA_W / u32:8;
const COCOTB_MEM_WRITER_ID = u32:0;

// sequence encoding
const COCOTB_MIN_SEQ_LEN = u32:3;
const COCOTB_HT_SIZE = u32:512;
const COCOTB_HT_SIZE_W = std::clog2(COCOTB_HT_SIZE + u32:1);
const COCOTB_HT_KEY_W = u32:32;
const COCOTB_HT_VALUE_W = COCOTB_HT_KEY_W + COCOTB_ADDR_W;
const COCOTB_HT_HASH_W = std::clog2(COCOTB_HT_SIZE);
const COCOTB_HT_RAM_DATA_W = COCOTB_HT_VALUE_W + u32:1;
const COCOTB_HT_RAM_WORD_PARTITION_SIZE = u32:1;
const COCOTB_HT_RAM_NUM_PARTITIONS = ram::num_partitions(COCOTB_HT_RAM_WORD_PARTITION_SIZE, COCOTB_HT_RAM_DATA_W);
const COCOTB_HT_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR = ram::SimultaneousReadWriteBehavior::READ_BEFORE_WRITE;
const COCOTB_HT_RAM_INITIALIZED = true;

const COCOTB_LITERALS_BUFFER_AXI_ADDR = u32:0x10000;
const COCOTB_SEQUENCE_BUFFER_AXI_ADDR = u32:0x20000;
const COCOTB_HB_DATA_W = u32:64;
const COCOTB_HB_SIZE = u32:1024;
const COCOTB_HB_OFFSET_W = std::clog2(COCOTB_HB_SIZE);
const COCOTB_HB_RAM_NUM = u32:8;
const COCOTB_HB_RAM_SIZE = COCOTB_HB_SIZE / COCOTB_HB_RAM_NUM;
const COCOTB_HB_RAM_DATA_W = COCOTB_HB_DATA_W / COCOTB_HB_RAM_NUM;
const COCOTB_HB_RAM_ADDR_W = std::clog2(COCOTB_HB_RAM_SIZE);
const COCOTB_HB_RAM_PARTITION_SIZE = COCOTB_HB_RAM_DATA_W;
const COCOTB_HB_RAM_NUM_PARTITIONS = ram::num_partitions(COCOTB_HB_RAM_PARTITION_SIZE, COCOTB_HB_RAM_DATA_W);
const COCOTB_FSE_TABLE_RAM_ADDR_W = u32:32;
const COCOTB_FSE_CTABLE_RAM_DATA_W = u32:16;
const COCOTB_FSE_TTABLE_RAM_DATA_W = u32:64;
const COCOTB_FSE_TTABLE_RAM_PARTITION_SIZE = u32:8;
const COCOTB_FSE_TABLE_PARTITION_SIZE = u32:8;
const COCOTB_FSE_CTABLE_RAM_NUM_PARTITIONS = ram::num_partitions(COCOTB_FSE_TABLE_PARTITION_SIZE, COCOTB_FSE_CTABLE_RAM_DATA_W);
const COCOTB_FSE_TTABLE_RAM_NUM_PARTITIONS = ram::num_partitions(COCOTB_FSE_TABLE_PARTITION_SIZE, COCOTB_FSE_TTABLE_RAM_DATA_W);
const COCOTB_FSE_BITSTREAM_BUFFER_W = u32:1024;

proc ZstdEncoderCocotbInst {
    type MemReaderReq = mem_reader::MemReaderReq<COCOTB_ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<COCOTB_DATA_W, COCOTB_ADDR_W>;
    type MemWriterReq = mem_writer::MemWriterReq<COCOTB_ADDR_W>;
    type MemWriterData = mem_writer::MemWriterDataPacket<COCOTB_DATA_W, COCOTB_ADDR_W>;
    type MemWriterResp = mem_writer::MemWriterResp;
    type MemWriterStatus = mem_writer::MemWriterRespStatus;
    type HistoryBufferRamRdReq = ram::ReadReq<COCOTB_HB_RAM_ADDR_W, COCOTB_HB_RAM_NUM_PARTITIONS>;
    type HistoryBufferRamRdResp = ram::ReadResp<COCOTB_HB_RAM_DATA_W>;
    type HistoryBufferRamWrReq = ram::WriteReq<COCOTB_HB_RAM_ADDR_W, COCOTB_HB_RAM_DATA_W, COCOTB_HB_RAM_NUM_PARTITIONS>;
    type HistoryBufferRamWrResp = ram::WriteResp;
    type HashTableRamRdReq = ram::ReadReq<COCOTB_HT_HASH_W, COCOTB_HT_RAM_NUM_PARTITIONS>;
    type HashTableRamRdResp = ram::ReadResp<COCOTB_HT_RAM_DATA_W>;
    type HashTableRamWrReq = ram::WriteReq<COCOTB_HT_HASH_W, COCOTB_HT_RAM_DATA_W, COCOTB_HT_RAM_NUM_PARTITIONS>;
    type HashTableRamWrResp = ram::WriteResp;
    type CTableRamRdReq = ram::ReadReq<COCOTB_FSE_TABLE_RAM_ADDR_W, COCOTB_FSE_CTABLE_RAM_NUM_PARTITIONS>;
    type CTableRamRdResp = ram::ReadResp<COCOTB_FSE_CTABLE_RAM_DATA_W>;
    type TTableRamRdReq = ram::ReadReq<COCOTB_FSE_TABLE_RAM_ADDR_W, COCOTB_FSE_TTABLE_RAM_NUM_PARTITIONS>;
    type TTableRamRdResp = ram::ReadResp<COCOTB_FSE_TTABLE_RAM_DATA_W>;
    type CTableRamWrReq = ram::WriteReq<COCOTB_FSE_TABLE_RAM_ADDR_W, COCOTB_FSE_CTABLE_RAM_DATA_W, COCOTB_FSE_CTABLE_RAM_NUM_PARTITIONS>;
    type CTableRamWrResp = ram::WriteResp;
    type TTableRamWrReq = ram::WriteReq<COCOTB_FSE_TABLE_RAM_ADDR_W, COCOTB_FSE_TTABLE_RAM_DATA_W, COCOTB_FSE_TTABLE_RAM_NUM_PARTITIONS>;
    type TTableRamWrResp = ram::WriteResp;
    type Req = ZstdEncodeReq<COCOTB_ADDR_W, COCOTB_DATA_W>;
    type Resp = ZstdEncodeResp<COCOTB_ADDR_W>;
    type AxiAr = axi::AxiAr<COCOTB_ADDR_W, COCOTB_ID_W>;
    type AxiR = axi::AxiR<COCOTB_DATA_W, COCOTB_ID_W>;
    type AxiAw = axi::AxiAw<COCOTB_ADDR_W, COCOTB_ID_W>;
    type AxiW = axi::AxiW<COCOTB_DATA_W, COCOTB_RAM_PARTITION_SIZE>;
    type AxiB = axi::AxiB<COCOTB_ID_W>;

    init {}

    ml_ctable_ram_wr_req_s: chan<CTableRamWrReq> out;
    ml_ctable_ram_wr_resp_r: chan<CTableRamWrResp> in;
    ll_ctable_ram_wr_req_s: chan<CTableRamWrReq> out;
    ll_ctable_ram_wr_resp_r: chan<CTableRamWrResp> in;
    of_ctable_ram_wr_req_s: chan<CTableRamWrReq> out;
    of_ctable_ram_wr_resp_r: chan<CTableRamWrResp> in;
    ml_ttable_ram_wr_req_s: chan<TTableRamWrReq> out;
    ml_ttable_ram_wr_resp_r: chan<TTableRamWrResp> in;
    ll_ttable_ram_wr_req_s: chan<TTableRamWrReq> out;
    ll_ttable_ram_wr_resp_r: chan<TTableRamWrResp> in;
    of_ttable_ram_wr_req_s: chan<TTableRamWrReq> out;
    of_ttable_ram_wr_resp_r: chan<TTableRamWrResp> in;

    config(
        enc_req_r: chan<Req> in,
        enc_resp_s: chan<Resp> out,
        axi_aw_s: chan<AxiAw> out,
        axi_w_s: chan<AxiW> out,
        axi_b_r: chan<AxiB> in,
        axi_ar_s: chan<AxiAr> out,
        axi_r_r: chan<AxiR> in,
        hb_ram_rd_req_s: chan<HistoryBufferRamRdReq>[COCOTB_HB_RAM_NUM] out,
        hb_ram_rd_resp_r: chan<HistoryBufferRamRdResp>[COCOTB_HB_RAM_NUM] in,
        hb_ram_wr_req_s: chan<HistoryBufferRamWrReq>[COCOTB_HB_RAM_NUM] out,
        hb_ram_wr_resp_r: chan<HistoryBufferRamWrResp>[COCOTB_HB_RAM_NUM] in,
        ht_ram_rd_req_s: chan<HashTableRamRdReq> out,
        ht_ram_rd_resp_r: chan<HashTableRamRdResp> in,
        ht_ram_wr_req_s: chan<HashTableRamWrReq> out,
        ht_ram_wr_resp_r: chan<HashTableRamWrResp>in,
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
        of_ttable_ram_rd_resp_r: chan<TTableRamRdResp> in,
        ml_ctable_ram_wr_req_s: chan<CTableRamWrReq> out,
        ml_ctable_ram_wr_resp_r: chan<CTableRamWrResp> in,
        ll_ctable_ram_wr_req_s: chan<CTableRamWrReq> out,
        ll_ctable_ram_wr_resp_r: chan<CTableRamWrResp> in,
        of_ctable_ram_wr_req_s: chan<CTableRamWrReq> out,
        of_ctable_ram_wr_resp_r: chan<CTableRamWrResp> in,
        ml_ttable_ram_wr_req_s: chan<TTableRamWrReq> out,
        ml_ttable_ram_wr_resp_r: chan<TTableRamWrResp> in,
        ll_ttable_ram_wr_req_s: chan<TTableRamWrReq> out,
        ll_ttable_ram_wr_resp_r: chan<TTableRamWrResp> in,
        of_ttable_ram_wr_req_s: chan<TTableRamWrReq> out,
        of_ttable_ram_wr_resp_r: chan<TTableRamWrResp> in
    ){
        let (mem_wr_req_s, mem_wr_req_r) = chan<MemWriterReq, u32:1>("mem_wr_req");
        let (mem_wr_data_s, mem_wr_data_r) = chan<MemWriterData, u32:1>("mem_wr_data");
        let (mem_wr_resp_s, mem_wr_resp_r) = chan<MemWriterResp, u32:1>("mem_wr_resp");
        let (mem_rd_req_s, mem_rd_req_r) = chan<MemReaderReq, u32:1>("mem_rd_req");
        let (mem_rd_resp_s, mem_rd_resp_r) = chan<MemReaderResp, u32:1>("mem_rd_resp");
        let (n_mem_wr_req_s, n_mem_wr_req_r) = chan<MemWriterReq, u32:1>[8]("n_req");
        let (n_mem_wr_data_s, n_mem_wr_data_r) = chan<MemWriterData, u32:1>[8]("n_data");
        let (n_mem_wr_resp_s, n_mem_wr_resp_r) = chan<MemWriterResp, u32:1>[8]("n_resp");
        let (n_mem_rd_req_s, n_mem_rd_req_r) = chan<MemReaderReq, u32:1>[4]("n_mem_rd_req");
        let (n_mem_rd_resp_s, n_mem_rd_resp_r) = chan<MemReaderResp, u32:1>[4]("n_mem_rd_resp");

        spawn mem_reader::MemReader<
        COCOTB_DATA_W, COCOTB_ADDR_W, COCOTB_DEST_W, COCOTB_ID_W,
        >(
            mem_rd_req_r, mem_rd_resp_s,
            axi_ar_s, axi_r_r,
        );
        spawn mem_writer::MemWriter<
        COCOTB_ADDR_W, COCOTB_DATA_W, COCOTB_DEST_W, COCOTB_ID_W, COCOTB_MEM_WRITER_ID
        >(
            mem_wr_req_r, mem_wr_data_r,
            axi_aw_s, axi_w_s, axi_b_r,
            mem_wr_resp_s
        );
        spawn mem_writer_simple_arbiter::MemWriterSimpleArbiter<COCOTB_ADDR_W, COCOTB_DATA_W, u32:8>
        (
            n_mem_wr_req_r, n_mem_wr_data_r, n_mem_wr_resp_s,
            mem_wr_req_s, mem_wr_data_s, mem_wr_resp_r,
        );
        spawn mem_reader_simple_arbiter::MemReaderSimpleArbiter<COCOTB_ADDR_W, COCOTB_DATA_W, u32:4> (
            n_mem_rd_req_r, n_mem_rd_resp_s,
            mem_rd_req_s, mem_rd_resp_r,
        );

        spawn ZstdEncoder<COCOTB_ADDR_W, COCOTB_DATA_W,
            COCOTB_RLE_HEURISTIC_SAMPLE_COUNT,
            COCOTB_HB_SIZE, COCOTB_HB_DATA_W, COCOTB_HB_OFFSET_W, COCOTB_HB_RAM_ADDR_W, COCOTB_HB_RAM_DATA_W, COCOTB_HB_RAM_NUM, COCOTB_HB_RAM_NUM_PARTITIONS,
            COCOTB_HT_SIZE, COCOTB_HT_KEY_W, COCOTB_HT_VALUE_W, COCOTB_HT_SIZE_W, COCOTB_HT_HASH_W, COCOTB_HT_RAM_DATA_W, COCOTB_HT_RAM_NUM_PARTITIONS,
            COCOTB_MIN_SEQ_LEN, COCOTB_LITERALS_BUFFER_AXI_ADDR, COCOTB_SEQUENCE_BUFFER_AXI_ADDR,
            COCOTB_FSE_TABLE_RAM_ADDR_W, COCOTB_FSE_CTABLE_RAM_DATA_W, COCOTB_FSE_TTABLE_RAM_DATA_W, COCOTB_FSE_CTABLE_RAM_NUM_PARTITIONS, COCOTB_FSE_TTABLE_RAM_NUM_PARTITIONS,
            COCOTB_FSE_BITSTREAM_BUFFER_W
        > (
            enc_req_r, enc_resp_s,
            n_mem_wr_req_s[0], n_mem_wr_data_s[0], n_mem_wr_resp_r[0],
            n_mem_wr_req_s[1], n_mem_wr_data_s[1], n_mem_wr_resp_r[1],
            n_mem_wr_req_s[2], n_mem_wr_data_s[2], n_mem_wr_resp_r[2],
            n_mem_wr_req_s[3], n_mem_wr_data_s[3], n_mem_wr_resp_r[3],
            n_mem_wr_req_s[4], n_mem_wr_data_s[4], n_mem_wr_resp_r[4],
            n_mem_wr_req_s[5], n_mem_wr_data_s[5], n_mem_wr_resp_r[5],
            n_mem_wr_req_s[6], n_mem_wr_data_s[6], n_mem_wr_resp_r[6],
            n_mem_wr_req_s[7], n_mem_wr_data_s[7], n_mem_wr_resp_r[7],
            n_mem_rd_req_s[0], n_mem_rd_resp_r[0],
            n_mem_rd_req_s[1], n_mem_rd_resp_r[1],
            n_mem_rd_req_s[2], n_mem_rd_resp_r[2],
            n_mem_rd_req_s[3], n_mem_rd_resp_r[3],
            hb_ram_rd_req_s, hb_ram_rd_resp_r, hb_ram_wr_req_s, hb_ram_wr_resp_r,
            ht_ram_rd_req_s, ht_ram_rd_resp_r, ht_ram_wr_req_s, ht_ram_wr_resp_r,
            ml_ctable_ram_rd_req_s, ml_ctable_ram_rd_resp_r,
            ll_ctable_ram_rd_req_s, ll_ctable_ram_rd_resp_r,
            of_ctable_ram_rd_req_s, of_ctable_ram_rd_resp_r,
            ml_ttable_ram_rd_req_s, ml_ttable_ram_rd_resp_r,
            ll_ttable_ram_rd_req_s, ll_ttable_ram_rd_resp_r,
            of_ttable_ram_rd_req_s, of_ttable_ram_rd_resp_r
        );
        (
            ml_ctable_ram_wr_req_s, ml_ctable_ram_wr_resp_r,
            ll_ctable_ram_wr_req_s, ll_ctable_ram_wr_resp_r,
            of_ctable_ram_wr_req_s, of_ctable_ram_wr_resp_r,
            ml_ttable_ram_wr_req_s, ml_ttable_ram_wr_resp_r,
            ll_ttable_ram_wr_req_s, ll_ttable_ram_wr_resp_r,
            of_ttable_ram_wr_req_s, of_ttable_ram_wr_resp_r
        )
    }

    next(state: ()) {
        let tok = join();
        send_if(tok, ml_ctable_ram_wr_req_s, false, zero!<CTableRamWrReq>());
        send_if(tok, ll_ctable_ram_wr_req_s, false, zero!<CTableRamWrReq>());
        send_if(tok, of_ctable_ram_wr_req_s, false, zero!<CTableRamWrReq>());
        send_if(tok, ml_ttable_ram_wr_req_s, false, zero!<TTableRamWrReq>());
        send_if(tok, ll_ttable_ram_wr_req_s, false, zero!<TTableRamWrReq>());
        send_if(tok, of_ttable_ram_wr_req_s, false, zero!<TTableRamWrReq>());
        recv_if(tok, ml_ctable_ram_wr_resp_r, false, zero!<CTableRamWrResp>());
        recv_if(tok, ll_ctable_ram_wr_resp_r, false, zero!<CTableRamWrResp>());
        recv_if(tok, of_ctable_ram_wr_resp_r, false, zero!<CTableRamWrResp>());
        recv_if(tok, ml_ttable_ram_wr_resp_r, false, zero!<CTableRamWrResp>());
        recv_if(tok, ll_ttable_ram_wr_resp_r, false, zero!<CTableRamWrResp>());
        recv_if(tok, of_ttable_ram_wr_resp_r, false, zero!<CTableRamWrResp>());
    }
}
