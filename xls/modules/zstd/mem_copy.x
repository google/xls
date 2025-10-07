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

// This file contains implementation of RawMemcopy

import std;

import xls.modules.zstd.memory.mem_reader;
import xls.modules.zstd.memory.mem_writer;

pub enum RawMemcopyStatus: u1 {
    OK = 0,
    ERROR = 1,
}

pub enum RawMemcopyBlockType: u3 {
    RAW        = 0,
    RLE        = 1,
    COMP       = 2,
    COMP_4     = 3,
    TREELESS   = 4,
    TREELESS_4 = 5,
}

pub struct RawMemcopyReq<ADDR_W: u32> {
    lit_addr: uN[ADDR_W],
    lit_cnt: u32,
    out_addr: uN[ADDR_W]
}

pub struct RawMemcopyResp<ADDR_W: u32> {
    status: RawMemcopyStatus,
    btype:  RawMemcopyBlockType,
    length: uN[ADDR_W],
}

proc RawMemcopyInternal<ADDR_W: u32, DATA_W: u32> {
    type MemReaderReq = mem_reader::MemReaderReq<ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<DATA_W, ADDR_W>;
    type MemWriterData = mem_writer::MemWriterDataPacket<DATA_W, ADDR_W>;
    type MemReaderStatus = mem_reader::MemReaderStatus;

    init {}

    mem_reader_resp_r: chan<MemReaderResp> in;
    mem_writer_data_s: chan<MemWriterData> out;
    done_s: chan<MemReaderStatus> out;

    config(
        mem_reader_resp_r: chan<MemReaderResp> in,
        mem_writer_data_s: chan<MemWriterData> out,
        done_s: chan<MemReaderStatus> out,
    ) {
        (mem_reader_resp_r, mem_writer_data_s, done_s)
    }

    next(state: ()) {
        let (resp_tok, resp) = recv(join(), mem_reader_resp_r);

        let is_ok = (resp.status == MemReaderStatus::OKAY);
        let mem_writer_data = MemWriterData {
            data: resp.data,
            length: resp.length,
            last: resp.last,
        };
        let data_tok = send_if(resp_tok, mem_writer_data_s, is_ok, mem_writer_data);

        let do_send_done = !is_ok || resp.last;
        send_if(data_tok, done_s, do_send_done, resp.status);
    }
}

pub proc RawMemcopy<ADDR_W: u32, DATA_W: u32> {
    type Req = RawMemcopyReq<ADDR_W>;
    type Resp = RawMemcopyResp<ADDR_W>;
    type Status = RawMemcopyStatus;

    type MemReaderReq = mem_reader::MemReaderReq<ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<DATA_W, ADDR_W>;
    type MemReaderStatus = mem_reader::MemReaderStatus;

    type MemWriterReq = mem_writer::MemWriterReq<ADDR_W>;
    type MemWriterData = mem_writer::MemWriterDataPacket<DATA_W, ADDR_W>;
    type MemWriterResp = mem_writer::MemWriterResp;
    type MemWriterStatus = mem_writer::MemWriterRespStatus;

    init {}

    req_r: chan<Req> in;
    resp_s: chan<Resp> out;

    mem_rd_req_s: chan<MemReaderReq> out;
    mem_copy_done_r: chan<MemReaderStatus> in;

    mem_wr_req_s: chan<MemWriterReq> out;
    mem_wr_resp_r: chan<MemWriterResp> in;

    config(
        req_r: chan<Req> in,
        resp_s: chan<Resp> out,

        mem_rd_req_s: chan<MemReaderReq> out,
        mem_rd_resp_r: chan<MemReaderResp> in,

        mem_wr_req_s: chan<MemWriterReq> out,
        mem_wr_data_s: chan<MemWriterData> out,
        mem_wr_resp_r: chan<MemWriterResp> in,
    ) {

        let (mem_copy_done_s, mem_copy_done_r) = chan<MemReaderStatus, u32:1>("mem_copy_done");

        spawn RawMemcopyInternal<ADDR_W, DATA_W>(
            mem_rd_resp_r,
            mem_wr_data_s,
            mem_copy_done_s,
        );

        (
            req_r, resp_s,
            mem_rd_req_s, mem_copy_done_r,
            mem_wr_req_s, mem_wr_resp_r,
        )
    }

    next(state: ()) {
        let (req_tok, req) = recv(join(), req_r);

        let mem_rd_req = MemReaderReq { addr: req.lit_addr, length: req.lit_cnt };
        let mem_rd_req_tok = send(req_tok, mem_rd_req_s, mem_rd_req);

        let mem_wr_req = MemWriterReq { addr: req.out_addr, length: req.lit_cnt };
        let mem_wr_req_tok = send(req_tok, mem_wr_req_s, mem_wr_req);

        let mem_req_tok = join(mem_rd_req_tok, mem_wr_req_tok);
        let (mem_copy_done_tok, mem_rd_status) = recv(mem_req_tok, mem_copy_done_r);
        let (mem_wr_resp_tok, mem_wr_resp) = recv_if(
            mem_req_tok, mem_wr_resp_r,
            mem_rd_status == MemReaderStatus::OKAY,
            MemWriterResp { status: mem_writer::MemWriterRespStatus::ERROR }
        );

        let mem_resp_tok = join(mem_copy_done_tok, mem_wr_resp_tok);

        let mem_rd_ok = (mem_rd_status == MemReaderStatus::OKAY);
        let mem_wr_ok = (mem_wr_resp.status == MemWriterStatus::OKAY);
        let status = if (mem_rd_ok && mem_wr_ok) { Status::OK } else { Status::ERROR };

        let resp = RawMemcopyResp {
            btype: RawMemcopyBlockType::RAW,
            length: req.lit_cnt,
            status,
        };

        send(mem_resp_tok, resp_s, resp);
    }
}

const INST_ADDR_W = u32:32;
const INST_DATA_W = u32:64;

proc RawMemcopyInst {
    type Req = RawMemcopyReq<INST_ADDR_W>;
    type Resp = RawMemcopyResp<INST_ADDR_W>;

    type MemReaderReq = mem_reader::MemReaderReq<INST_ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<INST_DATA_W, INST_ADDR_W>;
    type MemWriterReq = mem_writer::MemWriterReq<INST_ADDR_W>;
    type MemWriterData = mem_writer::MemWriterDataPacket<INST_DATA_W, INST_ADDR_W>;
    type MemWriterResp = mem_writer::MemWriterResp;

    init {}

    config(
        req_r: chan<Req> in,
        resp_s: chan<Resp> out,
        mem_rd_req_s: chan<MemReaderReq> out,
        mem_rd_resp_r: chan<MemReaderResp> in,
        mem_wr_req_s: chan<MemWriterReq> out,
        mem_wr_data_s: chan<MemWriterData> out,
        mem_wr_resp_r: chan<MemWriterResp> in,
    ) {
        spawn RawMemcopy<INST_ADDR_W, INST_DATA_W>(req_r, resp_s, mem_rd_req_s, mem_rd_resp_r, mem_wr_req_s, mem_wr_data_s, mem_wr_resp_r);
    }

    next(state: ()) { }
}
