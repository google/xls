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
import xls.modules.zstd.mem_copy;
import xls.modules.zstd.frame_header_dec;
import xls.modules.zstd.frame_header_enc;
import xls.modules.zstd.block_header;
import xls.modules.zstd.memory.mem_writer;
import xls.modules.zstd.memory.mem_reader;
import xls.modules.zstd.mem_writer_simple_arbiter;
import xls.modules.zstd.common;
import xls.modules.zstd.memory.axi;
import xls.modules.zstd.compression.block_size;

pub enum ZstdEncodeRespStatus : u1 {
    ERROR=0,
    OK=1
}

pub struct ZstdEncodeReq<ADDR_W: u32, DATA_W: u32> {
    input_offset: uN[ADDR_W], // bytes
    data_size: uN[DATA_W],
    output_offset: uN[ADDR_W], // bytes
    max_block_size: uN[DATA_W], //
}

pub struct ZstdEncodeResp {
    status: ZstdEncodeRespStatus
}

struct ZstdEncoderBlockWriterConf<ADDR_W: u32, DATA_W: u32> {
    bytes_left: uN[DATA_W],
    input_offset: uN[ADDR_W],
    output_offset: uN[ADDR_W],
    max_block_size: uN[DATA_W],
}

struct ZstdEncoderBlockWriterState<ADDR_W: u32, DATA_W: u32> {
    active: bool,
    conf: ZstdEncoderBlockWriterConf<ADDR_W, DATA_W>,
}

const ARBITER_SUBBLOCKS = u32:2;
const BLOCK_HEADER_LENGTH_BYTES = u32:3;
const XFERS_FOR_HEADER = u32:5;
const ZSTD_WINDOW_ABSOLUTEMIN = u64:1024;

pub proc ZstdEncoderBlockWriter<ADDR_W: u32, DATA_W: u32> {
    type MemReaderReq = mem_reader::MemReaderReq<ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<DATA_W, ADDR_W>;
    type MemWriterReq = mem_writer::MemWriterReq<ADDR_W>;
    type MemWriterData = mem_writer::MemWriterDataPacket<DATA_W, ADDR_W>;
    type MemWriterResp = mem_writer::MemWriterResp;

    type Resp = ZstdEncodeResp;
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
    type RawMemcopyReq = RawMemcopyReq<ADDR_W>;

    type Addr = uN[ADDR_W];
    type Data = uN[DATA_W];
    type BlockType = common::BlockType;
    type BlockSize = common::BlockSize;

    // parent I/O
    conf_r: chan<Conf> in;
    resp_s: chan<Status> out;

    // communication
    bhw_req_s: chan<BlockHeaderWriterReq> out;
    bhw_resp_r: chan<BlockHeaderWriterResp> in;
    memcpy_req_s: chan<RawMemcopyReq> out;
    memcpy_resp_r: chan<RawMemcopyResp> in;

    init { zero!<State>() }
    config(
        mem_rd_req_s: chan<MemReaderReq> out,
        mem_rd_resp_r: chan<MemReaderResp> in,
        mem_wr_req_s: chan<MemWriterReq> out,
        mem_wr_data_s: chan<MemWriterData> out,
        mem_wr_resp_r: chan<MemWriterResp> in,
        conf_r: chan<Conf> in,
        resp_s: chan<Status> out
    ) {
        let (bhw_req_s, bhw_req_r) = chan<BlockHeaderWriterReq, u32:1>("bhw_req");
        let (bhw_resp_s, bhw_resp_r) = chan<BlockHeaderWriterResp, u32:1>("bhw_resp");
        let (memcpy_req_s, memcpy_req_r) = chan<RawMemcopyReq, u32:1>("memcpy_req");
        let (memcpy_resp_s, memcpy_resp_r) = chan<RawMemcopyResp, u32:1>("memcpy_resp");

        let (n_mem_wr_req_s, n_mem_wr_req_r) = chan<MemWriterReq, u32:1>[2]("n_req");
        let (n_mem_wr_data_s, n_mem_wr_data_r) = chan<MemWriterData, u32:1>[2]("n_data");
        let (n_mem_wr_resp_s, n_mem_wr_resp_r) = chan<MemWriterResp, u32:1>[2]("n_resp");

        spawn mem_writer_simple_arbiter::MemWriterSimpleArbiter<ADDR_W, DATA_W, u32:2>
        (
            n_mem_wr_req_r, n_mem_wr_data_r, n_mem_wr_resp_s,
            mem_wr_req_s, mem_wr_data_s, mem_wr_resp_r,
        );

        spawn block_header::BlockHeaderWriter<DATA_W, ADDR_W>
        (
            bhw_req_r, bhw_resp_s,
            n_mem_wr_req_s[0], n_mem_wr_data_s[0], n_mem_wr_resp_r[0]
        );

        spawn mem_copy::RawMemcopy<ADDR_W, DATA_W>
        (
            memcpy_req_r, memcpy_resp_s,
            mem_rd_req_s, mem_rd_resp_r,
            n_mem_wr_req_s[1], n_mem_wr_data_s[1], n_mem_wr_resp_r[1]
        );

        (
            conf_r, resp_s,
            bhw_req_s, bhw_resp_r,
            memcpy_req_s, memcpy_resp_r
        )
    }
    next(state: State) {
        if !state.active {
            let (tok, conf) = recv(join(), conf_r);
            State {
                active: true,
                conf: conf
            }
        } else {
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

            let tok1 = send(join(), bhw_req_s, BlockHeaderWriterReq{
                    addr: conf.output_offset,
                    header: BlockHeader {
                        last: last_block,
                        size: size,
                        btype: BlockType::RAW //TODO: Receive block type from the parent
                    }
                }
            );
            trace_fmt!("writing block header to {:#x}", conf.output_offset);
            let (tok1, bhw_resp) = recv(tok1, bhw_resp_r);

            let tok2 = send(join(), memcpy_req_s, RawMemcopyReq {
                lit_addr: conf.input_offset,
                lit_cnt: size as Data,
                out_addr: conf.output_offset + BLOCK_HEADER_LENGTH_BYTES
            });
            trace_fmt!("raw copying: {:#x} -> {:#x} (size: {})", conf.input_offset, conf.output_offset + BLOCK_HEADER_LENGTH_BYTES, size);
            let (tok2, memcpy_resp) = recv(tok2, memcpy_resp_r);

            let status = if memcpy_resp.status == RawMemcopyRespStatus::OK && bhw_resp.status == BlockHeaderWriterStatus::OKAY {
                Status::OK
            } else {
                trace_fmt!("failed writing block: {} {}", bhw_resp, memcpy_resp);
                Status::ERROR
            };

            if last_block || status != ZstdEncodeRespStatus::OK {
                let tok = send(join(tok1, tok2), resp_s, status);
                zero!<State>()
            } else {
                let bytes_written = size as Data + BLOCK_HEADER_LENGTH_BYTES;
                State {
                    active: true,
                    conf: Conf {
                        bytes_left: conf.bytes_left as Data - size as Data,
                        input_offset: conf.input_offset as Addr + size as Addr,
                        output_offset: conf.output_offset as Addr + bytes_written as Addr,
                        ..conf
                    }
                }
            }
        }
    }
}

pub proc ZstdEncoder<ADDR_W: u32, DATA_W: u32> {
    type MemReaderReq = mem_reader::MemReaderReq<ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<DATA_W, ADDR_W>;
    type MemWriterReq = mem_writer::MemWriterReq<ADDR_W>;
    type MemWriterData = mem_writer::MemWriterDataPacket<DATA_W, ADDR_W>;
    type MemWriterResp = mem_writer::MemWriterResp;
    type MemWriterStatus = mem_writer::MemWriterRespStatus;

    type Req = ZstdEncodeReq<ADDR_W, DATA_W>;
    type Resp = ZstdEncodeResp;
    type ZstdEncoderBlockWriterConf = ZstdEncoderBlockWriterConf<ADDR_W, DATA_W>;
    type Status = ZstdEncodeRespStatus;

    type FrameHeader = frame_header_dec::FrameHeader;
    type FrameHeaderEncoderReq = frame_header_enc::FrameHeaderEncoderReq<ADDR_W>;
    type FrameHeaderEncoderResp = frame_header_enc::FrameHeaderEncoderResp;
    type FrameHeaderEncoderStatus = frame_header_enc::FrameHeaderEncoderStatus;

    // from
    enc_req: chan<Req> in;
    enc_resp_s: chan<Resp> out;

    bw_resp_r: chan<Status> in;

    mem_rd_req_s: chan<MemReaderReq> out;
    mem_rd_resp_r: chan<MemReaderResp> in;

    // to
    mem_wr_req_s: chan<MemWriterReq> out;
    mem_wr_data_s: chan<MemWriterData> out;
    mem_wr_resp_r: chan<MemWriterResp> in;

    // communication
    conf_s: chan<ZstdEncoderBlockWriterConf> out;
    fhw_req_s: chan<FrameHeaderEncoderReq> out;
    fhw_resp_r: chan<FrameHeaderEncoderResp> in;

    init { }

    config(
        enc_req_r: chan<Req> in,
        enc_resp_s: chan<Resp> out,
        mem_rd_req_s: chan<MemReaderReq> out,
        mem_rd_resp_r: chan<MemReaderResp> in,
        mem_wr_req_s: chan<MemWriterReq> out,
        mem_wr_data_s: chan<MemWriterData> out,
        mem_wr_resp_r: chan<MemWriterResp> in,
    ) {
        let (conf_s, conf_r) = chan<ZstdEncoderBlockWriterConf, u32:1>("conf");

        let (fhw_req_s, fhw_req_r) = chan<FrameHeaderEncoderReq, u32:1>("bhw_req");
        let (fhw_resp_s, fhw_resp_r) = chan<FrameHeaderEncoderResp, u32:1>("bhw_resp");

        let (n_mem_wr_req_s, n_mem_wr_req_r) = chan<MemWriterReq, u32:1>[ARBITER_SUBBLOCKS]("n_req");
        let (n_mem_wr_data_s, n_mem_wr_data_r) = chan<MemWriterData, u32:1>[ARBITER_SUBBLOCKS]("n_data");
        let (n_mem_wr_resp_s, n_mem_wr_resp_r) = chan<MemWriterResp, u32:1>[ARBITER_SUBBLOCKS]("n_resp");

        let (bw_resp_s, bw_resp_r) = chan<Status, u32:1>("bw_resp");

        spawn mem_writer_simple_arbiter::MemWriterSimpleArbiter<ADDR_W, DATA_W, ARBITER_SUBBLOCKS>
        (
            n_mem_wr_req_r, n_mem_wr_data_r, n_mem_wr_resp_s,
            mem_wr_req_s, mem_wr_data_s, mem_wr_resp_r,
        );

        spawn frame_header_enc::FrameHeaderEncoder<XFERS_FOR_HEADER, DATA_W, ADDR_W>
        (
            fhw_req_r, fhw_resp_s,
            n_mem_wr_req_s[0], n_mem_wr_data_s[0], n_mem_wr_resp_r[0]
        );

        spawn ZstdEncoderBlockWriter<ADDR_W, DATA_W>
        (
            mem_rd_req_s, mem_rd_resp_r,
            n_mem_wr_req_s[1], n_mem_wr_data_s[1], n_mem_wr_resp_r[1],
            conf_r, bw_resp_s
        );

        (
            enc_req_r, enc_resp_s,
            bw_resp_r,
            mem_rd_req_s, mem_rd_resp_r,
            mem_wr_req_s, mem_wr_data_s, mem_wr_resp_r,
            conf_s, fhw_req_s, fhw_resp_r,
        )
    }

    next(state: ()) {
        let (tok, request) = recv(join(), enc_req);

        let window_size = ZSTD_WINDOW_ABSOLUTEMIN; // TODO: Calculate the window size based on the frame content
        trace_fmt!("writing frame header to {:#x}", request.output_offset);
        let tok = send(tok, fhw_req_s, FrameHeaderEncoderReq{
            addr: request.output_offset,
            header: FrameHeader {
                window_size: window_size,
                frame_content_size: request.data_size as u64,
                dictionary_id: u32: 0,
                content_checksum_flag: u1: 0
            },
            fixed_size: u1:0,
            single_segment_flag: u1:1
        });
        let (tok, resp) = recv(tok, fhw_resp_r);
        trace_fmt!("written frame header {:#x}", resp);

        if (resp.status != FrameHeaderEncoderStatus::OKAY) {
            trace_fmt!("failed writing frame header");
            send(tok, enc_resp_s, ZstdEncodeResp{status: ZstdEncodeRespStatus::ERROR});
        } else {
            let tok = send(tok, conf_s, ZstdEncoderBlockWriterConf {
                bytes_left: request.data_size,
                input_offset: request.input_offset,
                output_offset: request.output_offset + resp.length as uN[ADDR_W],
                max_block_size: request.max_block_size
            });
            let (tok, status) = recv(tok, bw_resp_r);
            send(tok, enc_resp_s, ZstdEncodeResp{status: status});
        };
    }
}

const INST_ADDR_W = u32:32;
const INST_RAM_DATA_W = u32:32;

proc ZstdEncoderInst {
    type MemReaderReq = mem_reader::MemReaderReq<INST_ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<INST_RAM_DATA_W, INST_ADDR_W>;
    type MemWriterReq = mem_writer::MemWriterReq<INST_ADDR_W>;
    type MemWriterData = mem_writer::MemWriterDataPacket<INST_RAM_DATA_W, INST_ADDR_W>;
    type MemWriterResp = mem_writer::MemWriterResp;
    type MemWriterStatus = mem_writer::MemWriterRespStatus;
    type Req = ZstdEncodeReq<INST_ADDR_W, INST_RAM_DATA_W>;
    type Resp = ZstdEncodeResp;

    init {}

    config(
        enc_req_r: chan<Req> in,
        enc_resp_s: chan<Resp> out,
        mem_rd_req_s: chan<MemReaderReq> out,
        mem_rd_resp_r: chan<MemReaderResp> in,
        mem_wr_req_s: chan<MemWriterReq> out,
        mem_wr_data_s: chan<MemWriterData> out,
        mem_wr_resp_r: chan<MemWriterResp> in,
    ) {
        spawn  ZstdEncoder<INST_ADDR_W, INST_RAM_DATA_W>
        (
            enc_req_r, enc_resp_s,
            mem_rd_req_s, mem_rd_resp_r,
            mem_wr_req_s, mem_wr_data_s, mem_wr_resp_r
        );
    }

    next(state: ()) { }
}

const COCOTB_INST_MEM_WRITER_ID = u32:0;
const COCOTB_INST_ADDR_W = u32:32;
const COCOTB_INST_DATA_W = u32:32;
const COCOTB_INST_ID_W = u32:8;
const COCOTB_INST_DEST_W = u32:8;

const COCOTB_INST_RAM_SIZE = u32:2048;
const COCOTB_INST_RAM_ADDR_W = COCOTB_INST_ADDR_W;
const COCOTB_INST_RAM_PARTITION_SIZE = COCOTB_INST_DATA_W / u32:8;

proc ZstdEncoderCocotbInst {
    type MemReaderReq = mem_reader::MemReaderReq<COCOTB_INST_ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<COCOTB_INST_DATA_W, COCOTB_INST_ADDR_W>;
    type MemWriterReq = mem_writer::MemWriterReq<COCOTB_INST_ADDR_W>;
    type MemWriterData = mem_writer::MemWriterDataPacket<COCOTB_INST_DATA_W, COCOTB_INST_ADDR_W>;
    type MemWriterResp = mem_writer::MemWriterResp;
    type MemWriterStatus = mem_writer::MemWriterRespStatus;

    type AxiAr = axi::AxiAr<COCOTB_INST_ADDR_W, COCOTB_INST_ID_W>;
    type AxiR = axi::AxiR<COCOTB_INST_DATA_W, COCOTB_INST_ID_W>;
    type AxiAw = axi::AxiAw<COCOTB_INST_ADDR_W, COCOTB_INST_ID_W>;
    type AxiW = axi::AxiW<COCOTB_INST_DATA_W, COCOTB_INST_RAM_PARTITION_SIZE>;
    type AxiB = axi::AxiB<COCOTB_INST_ID_W>;

    type Req = ZstdEncodeReq<COCOTB_INST_ADDR_W, COCOTB_INST_DATA_W>;
    type Resp = ZstdEncodeResp;

    init {}

    config(
        enc_req_r: chan<Req> in,
        enc_resp_s: chan<Resp> out,
        axi_aw_s: chan<AxiAw> out,
        axi_w_s: chan<AxiW> out,
        axi_b_r: chan<AxiB> in,
        axi_ar_s: chan<AxiAr> out,
        axi_r_r: chan<AxiR> in,
    ) {
        let (mem_wr_req_s, mem_wr_req_r) = chan<MemWriterReq, u32:1>("mem_wr_req");
        let (mem_wr_data_s, mem_wr_data_r) = chan<MemWriterData, u32:1>("mem_wr_data");
        let (mem_wr_resp_s, mem_wr_resp_r) = chan<MemWriterResp, u32:1>("mem_wr_resp");
        let (mem_rd_req_s, mem_rd_req_r) = chan<MemReaderReq, u32:1>("mem_rd_req");
        let (mem_rd_resp_s, mem_rd_resp_r) = chan<MemReaderResp, u32:1>("mem_rd_resp");

        spawn mem_reader::MemReader<
        COCOTB_INST_DATA_W, COCOTB_INST_ADDR_W, COCOTB_INST_DEST_W, COCOTB_INST_ID_W,
        >(
            mem_rd_req_r, mem_rd_resp_s,
            axi_ar_s, axi_r_r,
        );

        spawn mem_writer::MemWriter<
        COCOTB_INST_ADDR_W, COCOTB_INST_DATA_W, COCOTB_INST_DEST_W, COCOTB_INST_ID_W, COCOTB_INST_MEM_WRITER_ID
        >(
            mem_wr_req_r, mem_wr_data_r,
            axi_aw_s, axi_w_s, axi_b_r,
            mem_wr_resp_s
        );

        spawn  ZstdEncoder<COCOTB_INST_ADDR_W, COCOTB_INST_DATA_W>
        (
            enc_req_r, enc_resp_s,
            mem_rd_req_s, mem_rd_resp_r,
            mem_wr_req_s, mem_wr_data_s, mem_wr_resp_r
        );
    }

    next(state: ()) { }
}

