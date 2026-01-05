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

import std;
import xls.modules.zstd.frame_header_dec;
import xls.modules.zstd.memory.mem_writer;
import xls.modules.zstd.memory.mem_writer_data_downscaler;


pub type FrameContentSize = u64;
pub type DictionaryId = u32;
pub type HeaderSize = u5;

type FrameHeader = frame_header_dec::FrameHeader;
type FrameHeaderDescriptor = frame_header_dec::FrameHeaderDescriptor;

const MAX_WINDOW_LOG = u32:22;
const WINDOW_LOG_W = std::clog2(MAX_WINDOW_LOG + u32:1);
type WindowLog = uN[WINDOW_LOG_W];

pub struct FrameHeaderEncoderReq<ADDR_W: u32> {
    addr: uN[ADDR_W],
    window_log: WindowLog,
    src_size: u64,
    dict_id: u32,
    max_block_size: u64,
    provide_dict_id: bool,
    provide_checksum: bool,
    provide_content_size: bool,
    provide_window_size: bool,
}

pub enum FrameHeaderEncoderStatus : u1 {
    OKAY = 0,
    ERROR = 1,
}

pub struct FrameHeaderEncoderResp {
    status: FrameHeaderEncoderStatus,
    length: HeaderSize
}

const MAGIC_NUMBER = u32:0xFD2FB528;
const MAGIC_NUMBER_LEN = u32:4;
const ZSTD_WINDOW_LOG_MIN = WindowLog:10;

// Implemented as in https://github.com/facebook/zstd/blob/12ea5f6e30c5b0411dd39d29089b5ee4e0044403/lib/compress/zstd_compress.c#L4703
fn create_frame_header_data(
    window_log: WindowLog, src_size: u64, dict_id: u32,
    provide_dict_id: bool, provide_checksum: bool, provide_content_size: bool, provide_window_size: bool,
    max_block_size: u64
) -> (uN[144], HeaderSize) {
    type Size = HeaderSize;
    type WindowSize = u64;

    const UNUSED_BIT = u1:0;
    const RESERVED_BIT = u1:0;

    let dict_id_size_code_length = match (dict_id) {
        u32:0   .. u32:1     => u2:0,
        u32:1   .. u32:256   => u2:1,
        u32:256 .. u32:65536 => u2:2,
        _                    => u2:3,
    };
    let dict_id_size_code = if provide_dict_id { dict_id_size_code_length } else { u2:0 };

    let window_size = WindowSize:1 << window_log;
    let single_segment = provide_content_size && (window_size >= src_size) && max_block_size <= src_size;
    let window_log_byte = ((window_log as u8 - ZSTD_WINDOW_LOG_MIN as u8) << 3) as u8;

    let fcs_code_length = match (src_size) {
        u64:0     .. u64:256        => u2:0,
        u64:256   .. u64:65792      => u2:1,
        u64:65792 .. u64:0xFFFFFFFF => u2:2,
        _                           => u2:3
    };
    let fcs_code = if provide_content_size { fcs_code_length } else { u2:0 };
    let frame_header_desc_byte = fcs_code_length ++ single_segment ++ UNUSED_BIT
                               ++ RESERVED_BIT ++ provide_checksum ++ dict_id_size_code;
    let raw_header = (frame_header_desc_byte as uN[112]) ++ MAGIC_NUMBER;

    // magic number
    let (raw_header, size) = if !single_segment && provide_window_size {
        (bit_slice_update(raw_header, u32:40, window_log_byte), Size:6) // size = magic number (4B) + desc (1B) + window (1B)
    } else {
        (raw_header, Size:5) // size = magic number + desc
    };

    let size_in_bits = size as u32 << 3;
    let (raw_header, size) = match (provide_dict_id, dict_id_size_code) {
      (false, _) => (raw_header, size),
      (true, u2:0) => (raw_header, size),
      (true, u2:1) => (bit_slice_update(raw_header, size_in_bits, dict_id as u8), size + Size:1),
      (true, u2:2) => (bit_slice_update(raw_header, size_in_bits, dict_id as u16), size + Size:2),
      (true, u2:3) => (bit_slice_update(raw_header, size_in_bits, dict_id as u32), size + Size:4),
      _  => fail!("dictionary_id_unreachable", (raw_header, size)),
    };

    let size_in_bits = size as u32 << 3;
    let (raw_header, size) = match (provide_content_size, fcs_code) {
        (false, _) => {
            (raw_header, size)
        },
        (true, u2:0) => {
            if single_segment {
                (bit_slice_update(raw_header, size_in_bits, src_size as u8), size + Size:1)
            } else {
                (raw_header, size)
            }
        },
        (true, u2:1) => (bit_slice_update(raw_header, size_in_bits, (src_size - u64:256) as u16), size + Size:2),
        (true, u2:2) => (bit_slice_update(raw_header, size_in_bits, src_size as u32), size + Size:4),
        (true, u2:3) => (bit_slice_update(raw_header, size_in_bits, src_size as u64), size + Size:8),
        _ => fail!("frame_content_size_unreachable", (raw_header, size)),
    };

    (raw_header, size)
}

pub const TEST_DATA_W = u32:32;
pub const TEST_ADDR_W = u32:16;

pub proc FrameHeaderEncoder<
    DATA_W: u32, ADDR_W: u32, WINDOW_LOG_MAX: u32 = { MAX_WINDOW_LOG as u32 },
    MAX_HEADER_SIZE: u32 = {u32:144},
> {
    type Req = FrameHeaderEncoderReq<ADDR_W>;
    type Resp = FrameHeaderEncoderResp;
    type Status = FrameHeaderEncoderStatus;
    type Addr = uN[ADDR_W];

    type MemWriterReq = mem_writer::MemWriterReq<ADDR_W>;
    type MemWriterData = mem_writer::MemWriterDataPacket<DATA_W, ADDR_W>;
    type MemWriterResp = mem_writer::MemWriterResp;
    type MemWriterDataWide = mem_writer::MemWriterDataPacket<MAX_HEADER_SIZE, ADDR_W>;

    req_r: chan<Req> in;
    resp_s: chan<Resp> out;
    mem_wr_req_s: chan<MemWriterReq> out;
    mem_wr_data_wide_s: chan<MemWriterDataWide> out;
    mem_wr_resp_r: chan<MemWriterResp> in;

    config(
        req_r: chan<Req> in,
        resp_s: chan<Resp> out,
        mem_wr_req_s: chan<MemWriterReq> out,
        mem_wr_data_s: chan<MemWriterData> out,
        mem_wr_resp_r: chan<MemWriterResp> in
    ) {

        let (mem_wr_data_wide_s, mem_wr_data_wide_r) = chan<MemWriterDataWide, u32:1>("mem_wr_data_wide");
        spawn mem_writer_data_downscaler::MemWriterDataDownscaler<ADDR_W, MAX_HEADER_SIZE, DATA_W>(
            mem_wr_data_wide_r, mem_wr_data_s,
        );

        (
            req_r, resp_s,
            mem_wr_req_s, mem_wr_data_wide_s, mem_wr_resp_r
        )
    }

    init { () }

    next(state: ()) {
        let tok0 = join();
        let (tok1, req) = recv(tok0, req_r);
        let (data, data_len) = create_frame_header_data(
            req.window_log, req.src_size, req.dict_id,
            req.provide_dict_id, req.provide_checksum, req.provide_content_size, req.provide_window_size,
            req.max_block_size
        );

        let length = checked_cast<Addr>(data_len);

        let mem_wr_req = MemWriterReq { addr: req.addr, length };
        let tok2_0 = send(tok1, mem_wr_req_s, mem_wr_req);
        let tok2_1 = send(tok1, mem_wr_data_wide_s, MemWriterDataWide { data, length, last: true });
        let tok2 = join(tok2_0, tok2_1);
        let (tok3, mem_wr_resp) = recv(tok2, mem_wr_resp_r);
        let status = if mem_wr_resp.status == mem_writer::MemWriterRespStatus::OKAY { Status::OKAY } else {Status::ERROR};
        let tok = send(tok3, resp_s, Resp { status, length: data_len });
    }
}

proc FrameHeaderEncoderInst {
    type Req = FrameHeaderEncoderReq<TEST_ADDR_W>;
    type Resp = FrameHeaderEncoderResp;
    type MemWriterReq = mem_writer::MemWriterReq<TEST_ADDR_W>;
    type MemWriterData = mem_writer::MemWriterDataPacket<TEST_DATA_W, TEST_ADDR_W>;
    type MemWriterResp = mem_writer::MemWriterResp;

    config(
        req_r: chan<Req> in,
        resp_s: chan<Resp> out,
        mem_wr_req_s: chan<MemWriterReq> out,
        mem_wr_data_s: chan<MemWriterData> out,
        mem_wr_resp_r: chan<MemWriterResp> in,
    ) {
        spawn FrameHeaderEncoder< TEST_DATA_W, TEST_ADDR_W>(
            req_r, resp_s,
            mem_wr_req_s, mem_wr_data_s, mem_wr_resp_r
        );

        ()
    }

    init {  }
    next(state: ()) {  }
}

struct TestCase<DATA_W: u32, ADDR_W: u32> {
    req: FrameHeaderEncoderReq<ADDR_W>,
    resp: FrameHeaderEncoderResp,
    mem_wr_req: mem_writer::MemWriterReq<ADDR_W>,
    mem_wr_resp: mem_writer::MemWriterResp,
    mem_wr_data: mem_writer::MemWriterDataPacket<DATA_W, ADDR_W>[5],
    mem_wr_data_length: u32,
}

#[test_proc]
proc FrameHeaderEncoderTest {
    type Req = FrameHeaderEncoderReq<TEST_ADDR_W>;
    type Resp = FrameHeaderEncoderResp;
    type MemWriterReq = mem_writer::MemWriterReq<TEST_ADDR_W>;
    type MemWriterDataPacket = mem_writer::MemWriterDataPacket<TEST_DATA_W, TEST_ADDR_W>;
    type MemWriterResp = mem_writer::MemWriterResp;
    type MemWriterRespStatus = mem_writer::MemWriterRespStatus;
    type TestCase = TestCase<TEST_DATA_W, TEST_ADDR_W>;

    type Data = uN[TEST_DATA_W];
    type Addr = uN[TEST_ADDR_W];

    terminator: chan<bool> out;
    req_s: chan<Req> out;
    resp_r: chan<Resp> in;

    mem_wr_req_r: chan<MemWriterReq> in;
    mem_wr_data_r: chan<MemWriterDataPacket> in;
    mem_wr_resp_s: chan<MemWriterResp> out;

    config(terminator: chan<bool> out) {
        let (req_s, req_r) = chan<Req>("req");
        let (resp_s, resp_r) = chan<Resp>("resp");
        let (mem_wr_req_s, mem_wr_req_r) = chan<MemWriterReq>("mem_wr_req");
        let (mem_wr_data_s, mem_wr_data_r) = chan<MemWriterDataPacket>("mem_wr_data");
        let (mem_wr_resp_s, mem_wr_resp_r) = chan<MemWriterResp>("mem_wr_resp");

        spawn FrameHeaderEncoder<TEST_DATA_W, TEST_ADDR_W>(
            req_r, resp_s,
            mem_wr_req_s, mem_wr_data_s, mem_wr_resp_r
        );

        (
            terminator,
            req_s, resp_r,
            mem_wr_req_r, mem_wr_data_r, mem_wr_resp_s
        )
    }

    init {  }

    next(state: ()) {
        const TEST_CASES = [
            TestCase { // 0
                // * content size encoding
                // * window size encoding
                // * no single segment
                req: FrameHeaderEncoderReq {
                    addr: Addr:1234,
                    window_log: WindowLog:22,
                    src_size: u64:0x3a16b33f3da53a79,
                    max_block_size: u64:0x1000,
                    dict_id: u32:0,
                    provide_dict_id: false,
                    provide_checksum: true,
                    provide_content_size: true,
                    provide_window_size: true,
                },
                resp: FrameHeaderEncoderResp { status: FrameHeaderEncoderStatus::OKAY, length: HeaderSize:14 },
                mem_wr_req: MemWriterReq { addr: Addr:1234, length: Addr:14 },
                mem_wr_resp: MemWriterResp { status: MemWriterRespStatus::OKAY },
                mem_wr_data: MemWriterDataPacket[5]:[
                    MemWriterDataPacket { data: Data:0xFD2FB528, length: Addr:4, last: u1:0 },
                    MemWriterDataPacket { data: Data:0x3A7960C4, length: Addr:4, last: u1:0 },
                    MemWriterDataPacket { data: Data:0xB33F3DA5, length: Addr:4, last: u1:0 },
                    MemWriterDataPacket { data: Data:0x00003A16, length: Addr:2, last: u1:1 },
                    zero!<MemWriterDataPacket>(), ...
                ],
                mem_wr_data_length: u32:4,
            },
            TestCase { // 1
                // * dict id encoding
                req: FrameHeaderEncoderReq {
                    addr: Addr:1234,
                    window_log: WindowLog:22,
                    src_size: u64:0x3a16b33f3da53a79,
                    max_block_size: u64:0x1000,
                    dict_id: u32:0xCAFE,
                    provide_dict_id: true,
                    provide_checksum: true,
                    provide_content_size: true,
                    provide_window_size: true,
                },
                resp: FrameHeaderEncoderResp { status: FrameHeaderEncoderStatus::OKAY, length: HeaderSize:16 },
                mem_wr_req: MemWriterReq { addr: Addr:1234, length: Addr:16 },
                mem_wr_resp: MemWriterResp { status: MemWriterRespStatus::OKAY },
                mem_wr_data: MemWriterDataPacket[5]:[
                    MemWriterDataPacket { data: Data:0xFD2FB528, length: Addr:4, last: u1:0 },
                    MemWriterDataPacket { data: Data:0xCAFE60C6, length: Addr:4, last: u1:0 },
                    MemWriterDataPacket { data: Data:0x3DA53A79, length: Addr:4, last: u1:0 },
                    MemWriterDataPacket { data: Data:0x3A16B33F, length: Addr:4, last: u1:1 },
                    zero!<MemWriterDataPacket>(), ...
                ],
                mem_wr_data_length: u32:4,
            },
            TestCase { // 2
                // * checksum flag disabled
                req: FrameHeaderEncoderReq {
                    addr: Addr:1234,
                    window_log: WindowLog:22,
                    src_size: u64:0x3a16b33f3da53a79,
                    max_block_size: u64:0x1000,
                    dict_id: u32:0xCAFE,
                    provide_dict_id: true,
                    provide_checksum: false,
                    provide_content_size: true,
                    provide_window_size: true,
                },
                resp: FrameHeaderEncoderResp { status: FrameHeaderEncoderStatus::OKAY, length: HeaderSize:16 },
                mem_wr_req: MemWriterReq { addr: Addr:1234, length: Addr:16 },
                mem_wr_resp: MemWriterResp { status: MemWriterRespStatus::OKAY },
                mem_wr_data: MemWriterDataPacket[5]:[
                    MemWriterDataPacket { data: Data:0xFD2FB528, length: Addr:4, last: u1:0 },
                    MemWriterDataPacket { data: Data:0xCAFE60C2, length: Addr:4, last: u1:0 },
                    MemWriterDataPacket { data: Data:0x3DA53A79, length: Addr:4, last: u1:0 },
                    MemWriterDataPacket { data: Data:0x3A16B33F, length: Addr:4, last: u1:1 },
                    zero!<MemWriterDataPacket>(), ...
                ],
                mem_wr_data_length: u32:4,
            },
            TestCase { // 3
                // * data in a single segment
                // * different response length
                req: FrameHeaderEncoderReq {
                    addr: Addr:1999,
                    window_log: WindowLog:22,
                    src_size: u64:0x200,
                    dict_id: u32:0xCAFE,
                    max_block_size: u64:0x200,
                    provide_dict_id: true,
                    provide_checksum: false,
                    provide_content_size: true,
                    provide_window_size: true,
                },
                resp: FrameHeaderEncoderResp { status: FrameHeaderEncoderStatus::OKAY, length: HeaderSize:9 },
                mem_wr_req: MemWriterReq { addr: Addr:1999, length: Addr:9 },
                mem_wr_resp: MemWriterResp { status: MemWriterRespStatus::OKAY },
                mem_wr_data: MemWriterDataPacket[5]:[
                    MemWriterDataPacket { data: Data:0xFD2FB528, length: Addr:4, last: u1:0 },
                    MemWriterDataPacket { data: Data:0x00CAFE62, length: Addr:4, last: u1:0 },
                    MemWriterDataPacket { data: Data:0x00000001, length: Addr:1, last: u1:1 },
                    zero!<MemWriterDataPacket>(), ...
                ],
                mem_wr_data_length: u32:3,
            },
            TestCase { // 4
                // * data of size 0
                req: FrameHeaderEncoderReq {
                    addr: Addr:1999,
                    window_log: WindowLog:22,
                    src_size: u64:0x0,
                    dict_id: u32:0xCAFE,
                    max_block_size: u64:0x0,
                    provide_dict_id: true,
                    provide_checksum: false,
                    provide_content_size: true,
                    provide_window_size: true,
                },
                resp: FrameHeaderEncoderResp { status: FrameHeaderEncoderStatus::OKAY, length: HeaderSize:8 },
                mem_wr_req: MemWriterReq { addr: Addr:1999, length: Addr:8 },
                mem_wr_resp: MemWriterResp { status: MemWriterRespStatus::OKAY },
                mem_wr_data: MemWriterDataPacket[5]:[
                    MemWriterDataPacket { data: Data:0xFD2FB528, length: Addr:4, last: u1:0 },
                    MemWriterDataPacket { data: Data:0x00CAFE22, length: Addr:4, last: u1:1 },
                    zero!<MemWriterDataPacket>(), ...
                ],
                mem_wr_data_length: u32:2,
            },
            TestCase { // 5
                req: FrameHeaderEncoderReq {
                    addr: Addr:1999,
                    window_log: WindowLog:22,
                    src_size: u64:0x0,
                    dict_id: u32:0xCAFE,
                    max_block_size: u64:0x0,
                    provide_dict_id: false,
                    provide_checksum: true,
                    provide_content_size: true,
                    provide_window_size: false,
                },
                resp: FrameHeaderEncoderResp { status: FrameHeaderEncoderStatus::OKAY, length: HeaderSize:6 },
                mem_wr_req: MemWriterReq { addr: Addr:1999, length: Addr:6 },
                mem_wr_resp: MemWriterResp { status: MemWriterRespStatus::OKAY },
                mem_wr_data: MemWriterDataPacket[5]:[
                    MemWriterDataPacket { data: Data:0xFD2FB528, length: Addr:4, last: u1:0 },
                    MemWriterDataPacket { data: Data:0x00000024, length: Addr:2, last: u1:1 },
                    zero!<MemWriterDataPacket>(), ...
                ],
                mem_wr_data_length: u32:2,
            },
            TestCase { // 6
                // test
                // * write error
                req: FrameHeaderEncoderReq {
                    addr: Addr:1999,
                    window_log: WindowLog:22,
                    src_size: u64:0x0,
                    dict_id: u32:0xCAFE,
                    max_block_size: u64:0x0,
                    provide_dict_id: false,
                    provide_checksum: true,
                    provide_content_size: true,
                    provide_window_size: false,
                },
                resp: FrameHeaderEncoderResp { status: FrameHeaderEncoderStatus::ERROR, length: HeaderSize:6 },
                mem_wr_req: MemWriterReq { addr: Addr:1999, length: Addr:6 },
                mem_wr_resp: MemWriterResp { status: MemWriterRespStatus::ERROR },
                mem_wr_data: MemWriterDataPacket[5]:[
                    MemWriterDataPacket { data: Data:0xFD2FB528, length: Addr:4, last: u1:0 },
                    MemWriterDataPacket { data: Data:0x00000024, length: Addr:2, last: u1:1 },
                    zero!<MemWriterDataPacket>(), ...
                ],
                mem_wr_data_length: u32:2,
            },
            TestCase { // 7
                // * skip content size
                // * shortest possible header
                req: FrameHeaderEncoderReq {
                    addr: Addr:1999,
                    window_log: WindowLog:22,
                    src_size: u64:0x0,
                    dict_id: u32:0xCAFE,
                    max_block_size: u64:0x0,
                    provide_dict_id: false,
                    provide_checksum: false,
                    provide_content_size: false,
                    provide_window_size: false,
                },
                resp: FrameHeaderEncoderResp { status: FrameHeaderEncoderStatus::ERROR, length: HeaderSize:5 },
                mem_wr_req: MemWriterReq { addr: Addr:1999, length: Addr:5 },
                mem_wr_resp: MemWriterResp { status: MemWriterRespStatus::ERROR },
                mem_wr_data: MemWriterDataPacket[5]:[
                    MemWriterDataPacket { data: Data:0xFD2FB528, length: Addr:4, last: u1:0 },
                    MemWriterDataPacket { data: Data:0x00000000, length: Addr:1, last: u1:1 },
                    zero!<MemWriterDataPacket>(), ...
                ],
                mem_wr_data_length: u32:2,
            },

        ];

        let tok = for (test_case, tok): (TestCase, token) in TEST_CASES {
            let tok = send(tok, req_s, test_case.req);
            let tok = send(tok, mem_wr_resp_s, test_case.mem_wr_resp);

            let (tok, mem_wr_req) = recv(tok, mem_wr_req_r);
            assert_eq(mem_wr_req, test_case.mem_wr_req);

            let tok = for (i, tok): (u32, token) in u32:0..array_size(test_case.mem_wr_data) {
                if i < test_case.mem_wr_data_length {
                    let (tok, mem_wr_data) = recv(tok, mem_wr_data_r);
                    assert_eq(mem_wr_data, test_case.mem_wr_data[i]);
                    tok
                } else { tok }
            }(tok);

            let (tok, resp) = recv(tok, resp_r);
            assert_eq(resp, test_case.resp);

            tok
        }(join());

        send(tok, terminator, true);
    }
}
