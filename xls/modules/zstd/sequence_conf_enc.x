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

// This file contains implementation of SequenceHeaderWriter

import std;

import xls.modules.zstd.common;
import xls.modules.zstd.memory.mem_writer;

pub struct SequenceSectionHeaderWriterReq<ADDR_W: u32> { addr: uN[ADDR_W], conf: common::SequenceConf }

pub enum SequenceSectionHeaderWriterStatus : u1 {
    OK = 0,
    ERROR = 1,
}

pub struct SequenceSectionHeaderWriterResp<ADDR_W: u32> {
    status: SequenceSectionHeaderWriterStatus,
    length: uN[ADDR_W]
}

const LITERALS_MODE_BITS_OFFSET = u32:6;
const OFFSET_MODE_BITS_OFFSET = u32:4;
const MATCH_MODE_BITS_OFFSET = u32:2;
const LONGSEQ = u17:0x7F00;

pub proc SequenceHeaderWriter<ADDR_W: u32, DATA_W: u32> {
    type Req = SequenceSectionHeaderWriterReq<ADDR_W>;
    type Resp = SequenceSectionHeaderWriterResp<ADDR_W>;
    type MemWriterReq = mem_writer::MemWriterReq<ADDR_W>;
    type MemWriterData = mem_writer::MemWriterDataPacket<DATA_W, ADDR_W>;
    type MemWriterResp = mem_writer::MemWriterResp;
    type MemWriterStatus = mem_writer::MemWriterRespStatus;
    type Status = SequenceSectionHeaderWriterStatus;
    type Data = uN[DATA_W];
    type Length = uN[ADDR_W];
    req_r: chan<Req> in;
    resp_s: chan<Resp> out;
    mem_wr_req_s: chan<MemWriterReq> out;
    mem_wr_data_s: chan<MemWriterData> out;
    mem_wr_resp_r: chan<MemWriterResp> in;

    config(req_r: chan<Req> in, resp_s: chan<Resp> out, mem_wr_req_s: chan<MemWriterReq> out,
           mem_wr_data_s: chan<MemWriterData> out, mem_wr_resp_r: chan<MemWriterResp> in) {
        (req_r, resp_s, mem_wr_req_s, mem_wr_data_s, mem_wr_resp_r)
    }

    init {  }

    next(state: ()) {
        let (req_tok, req) = recv(join(), req_r);
        let sequence_count = req.conf.sequence_count;

        let compression_modes_byte = (req.conf.literals_mode as u8 << LITERALS_MODE_BITS_OFFSET) |
                                     (req.conf.offset_mode as u8 << OFFSET_MODE_BITS_OFFSET) |
                                     (req.conf.match_mode as u8 << MATCH_MODE_BITS_OFFSET);

        let (data, length) = if sequence_count < u17:0x80 {
            let b0 = checked_cast<u8>(sequence_count);
            let data = (compression_modes_byte ++ b0) as Data;
            (data, Length:2)
        } else if sequence_count < LONGSEQ {
            let b0 = checked_cast<u8>((sequence_count >> 8) + u17:128);
            let b1 = checked_cast<u8>(sequence_count & u17:0xFF);
            let data = (compression_modes_byte ++ b1 ++ b0) as Data;
            (data, Length:3)
        } else {
            let b0 = u8:0xFF;
            let sequence_count_decreased = sequence_count - LONGSEQ;
            let b1 = checked_cast<u8>(sequence_count_decreased >> 8);
            let b2 = checked_cast<u8>(sequence_count_decreased & u17:0xFF);
            let data = (compression_modes_byte ++ b2 ++ b1 ++ b0) as Data;
            (data, Length:4)
        };

        let mem_wr_req = MemWriterReq { addr: req.addr, length };
        let mem_wr_tok = send(req_tok, mem_wr_req_s, mem_wr_req);

        let mem_wr_data = MemWriterData { data, length, last: true };
        let mem_wr_data_tok = send(mem_wr_tok, mem_wr_data_s, mem_wr_data);

        let (req_tok, resp) = recv(mem_wr_data_tok, mem_wr_resp_r);
        let status = if resp.status == MemWriterStatus::OKAY { Status::OK } else { Status::ERROR };
        let resp = Resp { status: status, length: length };
        send(req_tok, resp_s, resp);
    }
}

const INST_ADDR_W = u32:32;
const INST_DATA_W = u32:64;

proc SequenceHeaderWriterInst {
    type Req = SequenceSectionHeaderWriterReq<INST_ADDR_W>;
    type Resp = SequenceSectionHeaderWriterResp<INST_ADDR_W>;
    type MemWriterReq = mem_writer::MemWriterReq<INST_ADDR_W>;
    type MemWriterData = mem_writer::MemWriterDataPacket<INST_DATA_W, INST_ADDR_W>;
    type MemWriterResp = mem_writer::MemWriterResp;
    type MemWriterStatus = mem_writer::MemWriterRespStatus;
    type Status = SequenceSectionHeaderWriterStatus;
    type Data = uN[INST_DATA_W];
    type Length = uN[INST_ADDR_W];

    config(req_r: chan<Req> in, resp_s: chan<Resp> out, mem_wr_req_s: chan<MemWriterReq> out,
           mem_wr_data_s: chan<MemWriterData> out, mem_wr_resp_r: chan<MemWriterResp> in) {
        spawn SequenceHeaderWriter<INST_ADDR_W, INST_DATA_W>(
            req_r, resp_s, mem_wr_req_s, mem_wr_data_s, mem_wr_resp_r);
    }

    init {  }

    next(state: ()) {  }
}

const TEST_ADDR_W = u32:32;
const TEST_DATA_W = u32:64;

#[test_proc]
proc SequenceHeaderWriterTest {
    type Req = SequenceSectionHeaderWriterReq<TEST_ADDR_W>;
    type Resp = SequenceSectionHeaderWriterResp<TEST_ADDR_W>;
    type Addr = uN[TEST_ADDR_W];
    type Length = uN[TEST_ADDR_W];
    type Status = SequenceSectionHeaderWriterStatus;
    type SequenceConf = common::SequenceConf;
    type CompressionMode = common::CompressionMode;
    type MemWriterReq = mem_writer::MemWriterReq<TEST_ADDR_W>;
    type MemWriterData = mem_writer::MemWriterDataPacket<TEST_DATA_W, TEST_ADDR_W>;
    type MemWriterResp = mem_writer::MemWriterResp;
    type MemWriterRespStatus = mem_writer::MemWriterRespStatus;
    terminator: chan<bool> out;
    req_s: chan<Req> out;
    resp_r: chan<Resp> in;
    mem_wr_req_r: chan<MemWriterReq> in;
    mem_wr_data_r: chan<MemWriterData> in;
    mem_wr_resp_s: chan<MemWriterResp> out;

    config(terminator: chan<bool> out) {
        let (mem_wr_req_s, mem_wr_req_r) = chan<MemWriterReq>("mem_wr_req");
        let (mem_wr_data_s, mem_wr_data_r) = chan<MemWriterData>("mem_wr_data");
        let (mem_wr_resp_s, mem_wr_resp_r) = chan<MemWriterResp>("mem_wr_resp");

        let (req_s, req_r) = chan<Req>("req");
        let (resp_s, resp_r) = chan<Resp>("resp");

        spawn SequenceHeaderWriter<TEST_ADDR_W, TEST_DATA_W>(
            req_r, resp_s, mem_wr_req_s, mem_wr_data_s, mem_wr_resp_r);

        (terminator, req_s, resp_r, mem_wr_req_r, mem_wr_data_r, mem_wr_resp_s)
    }

    init {  }

    next(state: ()) {
        let tok = join();

        // modes b2 b1 b0, length, config
        let tests: (u32, u3, SequenceConf)[11] = [
            (
                u32:0x00_00, u3:2,
                SequenceConf {
                    sequence_count: u17:0,
                    literals_mode: CompressionMode::PREDEFINED,
                    offset_mode: CompressionMode::PREDEFINED,
                    match_mode: CompressionMode::PREDEFINED,
                },
            ),
            (
                u32:0x6C_00, u3:2,
                SequenceConf {
                    sequence_count: u17:0,
                    literals_mode: CompressionMode::RLE,
                    offset_mode: CompressionMode::COMPRESSED,
                    match_mode: CompressionMode::REPEAT,
                },
            ),
            (
                u32:0xE4_01, u3:2,
                SequenceConf {
                    sequence_count: u17:0x01,
                    literals_mode: CompressionMode::REPEAT,
                    offset_mode: CompressionMode::COMPRESSED,
                    match_mode: CompressionMode::RLE,
                },
            ),
            (
                u32:0xAC_7F, u3:2,
                SequenceConf {
                    sequence_count: u17:0x7F,
                    literals_mode: CompressionMode::COMPRESSED,
                    offset_mode: CompressionMode::COMPRESSED,
                    match_mode: CompressionMode::REPEAT,
                },
            ),
            (
                u32:0x84_80_80,
                u3:3,
                SequenceConf {
                    // corner-case
                    sequence_count: u17:0x80,
                    literals_mode: CompressionMode::COMPRESSED,
                    offset_mode: CompressionMode::PREDEFINED,
                    match_mode: CompressionMode::RLE,
                }
            ),
            (
                u32:0x84_81_80, u3:3,
                SequenceConf {
                    sequence_count: u17:0x0081,
                    literals_mode: CompressionMode::COMPRESSED,
                    offset_mode: CompressionMode::PREDEFINED,
                    match_mode: CompressionMode::RLE,
                },
            ),
            (
                u32:0x18_FFFE, u3:3,
                SequenceConf {
                    sequence_count: u17:0x7EFF,
                    literals_mode: CompressionMode::PREDEFINED,
                    offset_mode: CompressionMode::RLE,
                    match_mode: CompressionMode::COMPRESSED,
                },
            ),
            (
                u32:0x18_0000FF, u3:4,
                SequenceConf {
                    sequence_count: u17:0x7F00,
                    literals_mode: CompressionMode::PREDEFINED,
                    offset_mode: CompressionMode::RLE,
                    match_mode: CompressionMode::COMPRESSED,
                },
            ),
            (
                u32:0x18_0100FF, u3:4,
                SequenceConf {
                    sequence_count: u17:0x7F01,
                    literals_mode: CompressionMode::PREDEFINED,
                    offset_mode: CompressionMode::RLE,
                    match_mode: CompressionMode::COMPRESSED,
                },
            ),
            (
                u32:0x18_FF80FF, u3:4,
                SequenceConf {
                    sequence_count: u17:0xFFFF,
                    literals_mode: CompressionMode::PREDEFINED,
                    offset_mode: CompressionMode::RLE,
                    match_mode: CompressionMode::COMPRESSED,
                },
            ),
            (
                u32:0x68_FFFFFF, u3:4,
                SequenceConf {
                    sequence_count: u17:0x17EFF,
                    literals_mode: CompressionMode::RLE,
                    offset_mode: CompressionMode::COMPRESSED,
                    match_mode: CompressionMode::COMPRESSED,
                },
            ),
        ];
        const ADDR = uN[TEST_ADDR_W]:0xBEEF;

        let tok =
            for ((_, (header, length, config)), tok): ((u32, (u32, u3, SequenceConf)), token) in
                enumerate(tests) {
                let tok = send(tok, req_s, Req { addr: ADDR, conf: config });

                // first communicate, simulate memory writer actions
                let (tok, recv_request) = recv(tok, mem_wr_req_r);
                let (tok, recv_data) = recv(tok, mem_wr_data_r);
                let tok =
                    send(tok, mem_wr_resp_s, MemWriterResp { status: MemWriterRespStatus::OKAY });
                let (tok, recv_status) = recv(tok, resp_r);

                // then assert
                assert_eq(recv_request, MemWriterReq { addr: ADDR, length: length as u32 });
                assert_eq(
                    recv_data,
                    MemWriterData { data: header as u64, last: true, length: length as u32 });
                assert_eq(recv_status.status, Status::OK);

                tok
            }(tok);

        // negative case: memory writer sends error
        let tok = send(
            tok, req_s,
            Req {
                addr: ADDR,
                conf: SequenceConf {
                    sequence_count: u17:0,
                    literals_mode: CompressionMode::PREDEFINED,
                    offset_mode: CompressionMode::PREDEFINED,
                    match_mode: CompressionMode::PREDEFINED,
                },
            });

        let (tok, recv_request) = recv(tok, mem_wr_req_r);
        let (tok, recv_data) = recv(tok, mem_wr_data_r);
        let tok = send(tok, mem_wr_resp_s, MemWriterResp { status: MemWriterRespStatus::ERROR });
        let (tok, recv_status) = recv(tok, resp_r);

        assert_eq(recv_request, MemWriterReq { addr: ADDR, length: u32:2 });
        assert_eq(recv_data, MemWriterData { data: u64:0x00_00, last: true, length: u32:2 });
        assert_eq(recv_status.status, Status::ERROR);

        send(tok, terminator, true);
    }
}
