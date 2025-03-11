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

pub type WindowSize = u64;
pub type FrameContentSize = u64;
pub type DictionaryId = u32;
pub type HeaderSize = u5;

type FrameHeader = frame_header_dec::FrameHeader;
type FrameHeaderDescriptor = frame_header_dec::FrameHeaderDescriptor;

pub struct FrameHeaderEncoderReq<ADDR_W: u32> {
    addr: uN[ADDR_W],
    header: FrameHeader,
    // If true, forces encoder to always output 13 bytes. Requires dict_id=0, and single_segment_flag=1
    fixed_size: u1,
    // If true:
    // - window header descriptor is not written
    // - frame_content_size=0 is interpreted as "unknown"
    single_segment_flag: u1,
}

pub enum FrameHeaderEncoderStatus : u1 {
    OKAY = 0,
    ERROR = 1,
}

pub struct FrameHeaderEncoderResp { status: FrameHeaderEncoderStatus, length: HeaderSize }

enum FrameHeaderEncoderFsm : u2 {
    IDLE = 0,
    WRITE = 1,
    WAIT_FOR_END = 2,
}

struct FrameHeaderEncoderState<XFER_SIZE: u32, XFER_COUNT: u32> {
    fsm: FrameHeaderEncoderFsm,
    length: HeaderSize,
    write_idx: HeaderSize,
    end_idx: HeaderSize,
    last_word_len: HeaderSize,
    raw_data: uN[XFER_SIZE][XFER_COUNT],
}

// copy pasted from decoder, share it maybe
const MAGIC_NUMBER = u32:0xFD2FB528;
const MAGIC_NUMBER_LEN = u32:4;
const MAX_HEADER_LEN = u32:14;
const MAX_MAGIC_PLUS_HEADER_LEN = MAGIC_NUMBER_LEN + MAX_HEADER_LEN;

const MEM_WRITER_OK_RESP = mem_writer::MemWriterResp {
    status: mem_writer::MemWriterRespStatus::OKAY,
};
const MEM_WRITER_ERROR_RESP = mem_writer::MemWriterResp {
    status: mem_writer::MemWriterRespStatus::ERROR,
};

fn calc_frame_content_size_bytes(fcs: FrameContentSize, single_segment_flag: u1, fixed_size:u1) -> u4 {
    if (fixed_size) {
        assert!(
            single_segment_flag == u1:1,
            "single_segment_flag=0 is unsupported in fixed_size enc mode)");
        u4:8
    } else if (single_segment_flag) {
        if fcs <= u64:0xFF {
            u4:1
        } else if fcs <= u64:0xFF_FF + u64:256 {
            // 2 byte fcs uses biased representation with offset of 256
            u4:2
        } else if fcs <= u64:0xFFFF_FFFF {
            u4:4
        } else {
            u4:8
        }
    } else {
        if fcs == FrameContentSize:0 { // "unknown size"
            u4:0
        }  else if fcs <= u64:0xFF {
            // with single_segment_flag=0, values 1..256 need at least 4 bytes
            u4:4
        } else if fcs <= u64:0xFF_FF + u64:256 {
            // 2 byte fcs uses biased representation with offset of 256
            u4:2
        }  else if fcs <= u64:0xFFFF_FFFF {
            u4:4
        }  else {
            u4:8
        }
    }
}


#[test]
fn test_calc_frame_content_size_bytes() {
    const SINGLE_SEGMENT_FLAG_ON = u1:1;
    assert_eq(calc_frame_content_size_bytes(u64:0, SINGLE_SEGMENT_FLAG_ON, u1:0), u4:1);
    assert_eq(calc_frame_content_size_bytes(u64:255, SINGLE_SEGMENT_FLAG_ON, u1:0), u4:1);
    assert_eq(calc_frame_content_size_bytes(u64:256, SINGLE_SEGMENT_FLAG_ON, u1:0), u4:2);
    assert_eq(calc_frame_content_size_bytes(u64:0xFF_FF + u64:256, SINGLE_SEGMENT_FLAG_ON, u1:0), u4:2);
    assert_eq(calc_frame_content_size_bytes(u64:0xFF_FF + u64:257, SINGLE_SEGMENT_FLAG_ON, u1:0), u4:4);
    assert_eq(calc_frame_content_size_bytes(u64:0xFFFF_FFFF, SINGLE_SEGMENT_FLAG_ON, u1:0), u4:4);
    assert_eq(calc_frame_content_size_bytes(u64:0xFFFF_FFFF + u64:1, SINGLE_SEGMENT_FLAG_ON, u1:0), u4:8);
    assert_eq(calc_frame_content_size_bytes(all_ones!<u64>(), SINGLE_SEGMENT_FLAG_ON, u1:0), u4:8);

    const SINGLE_SEGMENT_FLAG_OFF = u1:0;
    assert_eq(calc_frame_content_size_bytes(u64:0, SINGLE_SEGMENT_FLAG_OFF, u1:0), u4:0);
    assert_eq(calc_frame_content_size_bytes(u64:1, SINGLE_SEGMENT_FLAG_OFF, u1:0), u4:4);
    assert_eq(calc_frame_content_size_bytes(u64:255, SINGLE_SEGMENT_FLAG_OFF, u1:0), u4:4);
    assert_eq(calc_frame_content_size_bytes(u64:256, SINGLE_SEGMENT_FLAG_OFF, u1:0), u4:2);
    assert_eq(calc_frame_content_size_bytes(u64:0xFF_FF + u64:256, SINGLE_SEGMENT_FLAG_OFF, u1:0), u4:2);
    assert_eq(calc_frame_content_size_bytes(u64:0xFF_FF + u64:257, SINGLE_SEGMENT_FLAG_OFF, u1:0), u4:4);
    assert_eq(calc_frame_content_size_bytes(u64:0xFFFF_FFFF, SINGLE_SEGMENT_FLAG_OFF, u1:0), u4:4);
    assert_eq(calc_frame_content_size_bytes(u64:0xFFFF_FFFF + u64:1, SINGLE_SEGMENT_FLAG_OFF, u1:0), u4:8);
    assert_eq(calc_frame_content_size_bytes(all_ones!<u64>(), SINGLE_SEGMENT_FLAG_OFF, u1:0), u4:8);

    // Fixed size:
    assert_eq(calc_frame_content_size_bytes(u64:0, SINGLE_SEGMENT_FLAG_ON, u1:1), u4:8);
    assert_eq(calc_frame_content_size_bytes(u64:255, SINGLE_SEGMENT_FLAG_ON, u1:1), u4:8);
    assert_eq(calc_frame_content_size_bytes(all_ones!<u64>(), SINGLE_SEGMENT_FLAG_ON, u1:1), u4:8);
}

fn calc_dictionary_id_bytes(dictionary_id: DictionaryId, fixed_size:u1) -> u3 {
    let id_bits = u6:32 - clz(dictionary_id) as u6;
    let id_bytes = (id_bits >> 3) as u3;
    let id_bytes = match id_bytes {
        // RFC8878 Section 3.1.1.1.2
        // when dictionary id is not present "it's up to the decoder to know which dictionary
        // to use". Our decoder interprets missing dictionary id field as id=0
        u3:0 => u3:0, // NOTE: if we want broad compat, we should consider returning 1 here
        u3:1 => u3:1,
        u3:2 => u3:2,
        u3:3 => u3:4,
        u3:4 => u3:4,
        _ => fail!("dictionary_id_bytes_unreachable", u3:0),
    };
    if(fixed_size) {
        assert!(id_bytes == u3:0, "only default (0) dictionary id is supported in fixed_size enc mode)");
    } else {};
    id_bytes as u3
}

#[test]
fn test_calc_dictionary_id_bytes() {
    assert_eq(calc_dictionary_id_bytes(u32:0, u1:1), u3:0);
    assert_eq(calc_dictionary_id_bytes(u32:0, u1:0), u3:0);
    assert_eq(calc_dictionary_id_bytes(u32:0xFF, u1:0), u3:1);
    assert_eq(calc_dictionary_id_bytes(u32:0xFFFF, u1:0), u3:2);
    assert_eq(calc_dictionary_id_bytes(u32:0xFF_FFFF, u1:0), u3:4);
    assert_eq(calc_dictionary_id_bytes(u32:0xFFFF_FFFF, u1:0), u3:4);
}

fn make_frame_header_descriptor
    (header: frame_header_dec::FrameHeader, fixed_size: u1, single_segment_flag: u1) -> FrameHeaderDescriptor {

    let frame_content_size_bytes = calc_frame_content_size_bytes(header.frame_content_size, single_segment_flag, fixed_size);
    let frame_content_size_flag = match frame_content_size_bytes {
        u4:0 => u2:0, // unknown size
        u4:1 => u2:0,
        u4:2 => u2:1,
        u4:4 => u2:2,
        u4:8 => u2:3,
        _ => fail!("frame_content_size_flag_unreachable", u2:0),
    };

    let dictionary_id_bytes = calc_dictionary_id_bytes(header.dictionary_id, fixed_size);
    let dictionary_id_flag = match dictionary_id_bytes {
        u3:0 => u2:0,
        u3:1 => u2:1,
        u3:2 => u2:2,
        u3:4 => u2:3,
        _ => fail!("dictionary_id_flag_unreachable", u2:0),
    };

    FrameHeaderDescriptor {
        frame_content_size_flag,
        single_segment_flag,
        unused: u1:0,
        reserved: u1:0,
        content_checksum_flag: header.content_checksum_flag,
        dictionary_id_flag,
    }
}

#[test]
fn test_make_frame_header_descriptor() {
    assert_eq(
        make_frame_header_descriptor(
            FrameHeader {
                window_size: u64:0x3a16b33f3da53a79,
                frame_content_size: u64:0x3a16b33f3da53a79,
                dictionary_id: u32:0,
                content_checksum_flag: u1:1,
            }, u1:0, u1:1),
        FrameHeaderDescriptor {
            frame_content_size_flag: u2:0x3,
            single_segment_flag: u1:0x1,
            unused: u1:0x0,
            reserved: u1:0x0,
            content_checksum_flag: u1:0x1,
            dictionary_id_flag: u2:0x0,
        });

    assert_eq(
        make_frame_header_descriptor(
            FrameHeader {
                window_size: u64:0xaa,
                frame_content_size: u64:255,
                dictionary_id: u32:0xCAFE,
                content_checksum_flag: u1:0,
            }, u1:0, u1:1),
        FrameHeaderDescriptor {
            frame_content_size_flag: u2:0x0,
            single_segment_flag: u1:0x1,
            unused: u1:0x0,
            reserved: u1:0x0,
            content_checksum_flag: u1:0x0,
            dictionary_id_flag: u2:2,
        });
    assert_eq(
        make_frame_header_descriptor(
            FrameHeader {
                window_size: u64:0xaa,
                frame_content_size: u64:0xFF_FF + u64:256,
                dictionary_id: u32:0xCAFE,
                content_checksum_flag: u1:0,
            }, u1:0, u1:1),
        FrameHeaderDescriptor {
            frame_content_size_flag: u2:0x1,
            single_segment_flag: u1:0x1,
            unused: u1:0x0,
            reserved: u1:0x0,
            content_checksum_flag: u1:0x0,
            dictionary_id_flag: u2:2,
        });
    assert_eq(
        make_frame_header_descriptor(
            FrameHeader {
                window_size: u64:0xaa,
                frame_content_size: u64:0xFF_FF + u64:257,
                dictionary_id: u32:0xCAFE,
                content_checksum_flag: u1:0,
            }, u1:0, u1:1),
        FrameHeaderDescriptor {
            frame_content_size_flag: u2:0x2,
            single_segment_flag: u1:0x1,
            unused: u1:0x0,
            reserved: u1:0x0,
            content_checksum_flag: u1:0x0,
            dictionary_id_flag: u2:2,
        });
}

fn make_raw_frame_header_descriptor(desc: FrameHeaderDescriptor) -> u8 {
    desc.frame_content_size_flag ++ desc.single_segment_flag ++ desc.unused ++ desc.reserved ++
    desc.content_checksum_flag ++ desc.dictionary_id_flag
}

#[test]
fn test_make_raw_frame_header_descriptor() {
    assert_eq(
        make_raw_frame_header_descriptor(
            FrameHeaderDescriptor {
                frame_content_size_flag: u2:0x3,
                single_segment_flag: u1:0x1,
                unused: u1:0x0,
                reserved: u1:0x0,
                content_checksum_flag: u1:0x1,
                dictionary_id_flag: u2:0x0,
            }), u8:0xE4);
}

fn make_window_descriptor(window_size: WindowSize) -> (u5, u3) {
    // Based on window_size calculation from: RFC 8878 Section 3.1.1.1.2:
    // windowLog = 10 + Exponent;
    // windowBase = 1 << windowLog;
    // windowAdd = (windowBase / 8) * Mantissa;
    // Window_Size = windowBase + windowAdd;

    assert!(window_size >= (WindowSize:1 << 10), "too small window size");
    assert!(
        window_size <= (WindowSize:1 << 41) + WindowSize:7 * (WindowSize:1 << 38),
        "too big window size");
    let window_size_len = u7:64 - clz(window_size) as u7;
    let window_log = window_size_len - u7:1;
    let exponent = (window_log - u7:10) as u5;
    let window_base = u64:1 << window_log;
    let window_add = window_size - window_base;
    let mantissa = (window_add << 3 >> window_log) as u3;  // equivalent of (window_add /
                                                           // (windowBase / 8))
    (exponent, mantissa)
}

#[test]
fn test_make_window_descriptor() {
    assert_eq(make_window_descriptor(u64:0x400), (HeaderSize:0, u3:0));
    assert_eq(make_window_descriptor(u64:0x900), (HeaderSize:1, u3:1));
    assert_eq(make_window_descriptor(u64:0x3c000000000), (HeaderSize:31, u3:7));
}

fn make_raw_magic_and_header(header: FrameHeader, fixed_size: u1, single_segment_flag: u1) -> (uN[144], HeaderSize) {
    let fhd = make_frame_header_descriptor(header, fixed_size, single_segment_flag);
    let raw_fhd = make_raw_frame_header_descriptor(fhd);
    let raw_header = raw_fhd as uN[112];

    let window_descriptor_start = u8:8;

    let (raw_header, window_descriptor_end) = if fhd.single_segment_flag {
        (raw_header, window_descriptor_start)
    } else {
        let (exponent, mantissa) = make_window_descriptor(header.window_size);
        let raw_window_descriptor = exponent ++ mantissa;
        (
            bit_slice_update(raw_header, window_descriptor_start, raw_window_descriptor),
            window_descriptor_start + u8:8,
        )
    };

    let dictionary_id_start = window_descriptor_end;
    let dictionary_id_bytes = calc_dictionary_id_bytes(header.dictionary_id, fixed_size);
    let dictionary_id_bits = (dictionary_id_bytes as u8 << 3);
    let dictionary_id_end = dictionary_id_start + dictionary_id_bits;

    let raw_header = match dictionary_id_bytes {
        u3:0 => raw_header,
        u3:1 => bit_slice_update(raw_header, dictionary_id_start, header.dictionary_id as u8),
        u3:2 => bit_slice_update(raw_header, dictionary_id_start, header.dictionary_id as u16),
        u3:4 => bit_slice_update(raw_header, dictionary_id_start, header.dictionary_id as u32),
        _ => fail!("dictionary_id_unreachable", raw_header),
    };

    let frame_content_size_start = dictionary_id_end;
    let frame_content_size_bytes = calc_frame_content_size_bytes(header.frame_content_size, single_segment_flag, fixed_size);
    let frame_content_size_bits = (frame_content_size_bytes as u8 << 3);
    let frame_content_size_end = frame_content_size_start + frame_content_size_bits;

    let frame_content_size_encoded = if frame_content_size_bytes == u4:2 {
        // 2 byte fcs uses offset of 256
        header.frame_content_size - FrameContentSize:256
    } else {
        header.frame_content_size
    };
    let raw_header = match frame_content_size_bytes {
        u4:0 => raw_header,
        u4:1 => bit_slice_update(raw_header, frame_content_size_start, frame_content_size_encoded as u8),
        u4:2 => bit_slice_update(raw_header, frame_content_size_start, frame_content_size_encoded as u16),
        u4:4 => bit_slice_update(raw_header, frame_content_size_start, frame_content_size_encoded as u32),
        u4:8 => bit_slice_update(raw_header, frame_content_size_start, frame_content_size_encoded as u64),
        _ => fail!("frame_content_size_unreachable", raw_header),
    };

    let header_len = (frame_content_size_end >> 3) as HeaderSize;
    (raw_header ++ MAGIC_NUMBER, header_len + MAGIC_NUMBER_LEN as HeaderSize)
}

fn partition_for_memwrite<DATA_W: u32, XFERS: u32, IN_SIZE: u32>
    (data: uN[IN_SIZE]) -> uN[DATA_W][XFERS] {
    let data_extended = data as uN[DATA_W * XFERS];
    data_extended as uN[DATA_W][XFERS]
}

fn calc_words<DATA_W: u32>(bytes: HeaderSize) -> (HeaderSize, HeaderSize) {
    const BYTES_PER_WORD = DATA_W / u32:8;
    let words = bytes as u32 / BYTES_PER_WORD;
    let leftover_bytes = bytes as u32 % BYTES_PER_WORD;
    let (words, last_word_len) = if leftover_bytes == u32:0 {
        (words, BYTES_PER_WORD)
    } else {
        (words + u32:1, leftover_bytes)
    };
    (words as HeaderSize, last_word_len as HeaderSize)
}

#[test]
fn test_calc_words() {
    assert_eq(calc_words<u32:32>(HeaderSize:1), (HeaderSize:1, HeaderSize:1));
    assert_eq(calc_words<u32:32>(HeaderSize:2), (HeaderSize:1, HeaderSize:2));
    assert_eq(calc_words<u32:32>(HeaderSize:3), (HeaderSize:1, HeaderSize:3));
    assert_eq(calc_words<u32:32>(HeaderSize:4), (HeaderSize:1, HeaderSize:4));
    assert_eq(calc_words<u32:32>(HeaderSize:5), (HeaderSize:2, HeaderSize:1));

    assert_eq(calc_words<u32:16>(HeaderSize:6), (HeaderSize:3, HeaderSize:2));
    assert_eq(calc_words<u32:8>(HeaderSize:21), (HeaderSize:21, HeaderSize:1));
}

pub proc FrameHeaderEncoder<WINDOW_LOG_MAX: u32, DATA_W: u32, ADDR_W: u32, MAX_XFERS_FOR_HEADER:
u32 = {
    ((MAX_MAGIC_PLUS_HEADER_LEN * u32:8) / DATA_W) + u32:1}>
{
    // type State = FrameHeaderEncoderState<DATA_W, MAX_XFERS_FOR_HEADER>;
    type State = FrameHeaderEncoderState<u32:32, u32:5>;
    // FIXME: hardcoded params. For some reason xls fails with internal error if we forward DATA_W
    // and XFERS_FOR_HEADER here
    type Fsm = FrameHeaderEncoderFsm;
    type Req = FrameHeaderEncoderReq<ADDR_W>;
    type Resp = FrameHeaderEncoderResp;
    type Status = FrameHeaderEncoderStatus;
    type WriterReq = mem_writer::MemWriterReq<ADDR_W>;
    type WriterData = mem_writer::MemWriterDataPacket<DATA_W, ADDR_W>;
    type WriterResp = mem_writer::MemWriterResp;
    // request, should contain a structure with all the values required to write the header to
    // memory
    req_r: chan<Req> in;
    // response, should return a status of the write operation
    resp_s: chan<Resp> out;
    // request to the MemoryWriter, with the target writer and length of the write transactions
    mem_wr_req_s: chan<WriterReq> out;
    // packets with the actual data that should be send to memory
    mem_wr_data_s: chan<WriterData> out;
    // response from the MemoryWriter with the status
    mem_wr_resp_r: chan<WriterResp> in;

    config(req_r: chan<Req> in, resp_s: chan<Resp> out, mem_wr_req_s: chan<WriterReq> out,
           mem_wr_data_s: chan<WriterData> out, mem_wr_resp_r: chan<WriterResp> in) {
        (req_r, resp_s, mem_wr_req_s, mem_wr_data_s, mem_wr_resp_r)
    }

    init { zero!<State>() }

    next(state: State) {
        let tok = join();

        let new_state = match state.fsm {
            Fsm::IDLE => {
                let (tok, req) = recv(tok, req_r);
                let (raw_data, raw_data_len) =
                    make_raw_magic_and_header(req.header, req.fixed_size, req.single_segment_flag);
                let raw_data = partition_for_memwrite<DATA_W, MAX_XFERS_FOR_HEADER>(raw_data);
                let (raw_data_words, last_word_len) = calc_words<DATA_W>(raw_data_len);
                let start_idx = MAX_XFERS_FOR_HEADER as HeaderSize - HeaderSize:1;
                let end_idx = MAX_XFERS_FOR_HEADER as HeaderSize - raw_data_words;
                let tok = send(
                    tok, mem_wr_req_s,
                    WriterReq { addr: req.addr, length: raw_data_len as uN[ADDR_W] });
                State {
                    fsm: Fsm::WRITE,
                    write_idx: start_idx,
                    length: raw_data_len,
                    end_idx,
                    last_word_len,
                    raw_data,
                }
            },
            Fsm::WRITE => {
                // raw_data is written from the highest to lowest index because
                // highest index in an array contains least significant bytes when
                // casting from a bit vector
                let last = state.write_idx == state.end_idx;
                let length = if last {
                    state.last_word_len as uN[ADDR_W]
                } else {
                    (DATA_W / u32:8) as uN[ADDR_W]
                };
                let tok = send(
                    tok, mem_wr_data_s,
                    WriterData { data: state.raw_data[state.write_idx], length, last });
                if last {
                    State { fsm: Fsm::WAIT_FOR_END, ..state }
                } else {
                    State { write_idx: state.write_idx - HeaderSize:1, ..state }
                }
            },
            Fsm::WAIT_FOR_END => {
                let (tok, mem_wr_resp) = recv(tok, mem_wr_resp_r);
                let status =
                    if mem_wr_resp == MEM_WRITER_OK_RESP { Status::OKAY } else { Status::ERROR };
                send(tok, resp_s, Resp { status, length: state.length as HeaderSize });

                State { fsm: Fsm::IDLE, ..state }
            },
            _ => fail!("FrameHeaderEncoder_fsm_unreachable", state),
        };
        new_state
    }
}

// constants copypasted from decoder
// The largest allowed WindowLog for DSLX tests
pub const TEST_WINDOW_LOG_MAX = u32:22;
pub const TEST_DATA_W = u32:32;
pub const TEST_ADDR_W = u32:16;
pub const TEST_XFERS_FOR_HEADER = ((MAX_MAGIC_PLUS_HEADER_LEN * u32:8) / TEST_DATA_W) + u32:1;

struct FrameHeaderEncoderTestCase<DATA_W: u32, XFERS_FOR_HEADER: u32> {
    header: FrameHeader,
    encoded: uN[DATA_W][XFERS_FOR_HEADER],
    fixed_size: u1,  // forces encoder to always output 13 bytes. Requires dict_id=0
    single_segment_flag: u1,
    length: HeaderSize,
}

#[test_proc]
proc FrameHeaderEncoderTest {
    type Req = FrameHeaderEncoderReq<TEST_ADDR_W>;
    type Resp = FrameHeaderEncoderResp;
    type WriterReq = mem_writer::MemWriterReq<TEST_ADDR_W>;
    type WriterData = mem_writer::MemWriterDataPacket<TEST_DATA_W, TEST_ADDR_W>;
    type WriterResp = mem_writer::MemWriterResp;
    type TestCase = FrameHeaderEncoderTestCase<TEST_DATA_W, TEST_XFERS_FOR_HEADER>;
    terminator: chan<bool> out;
    encode_req_s: chan<Req> out;
    encode_resp_r: chan<Resp> in;
    writer_req_r: chan<WriterReq> in;
    writer_data_r: chan<WriterData> in;
    writer_resp_s: chan<WriterResp> out;

    config(terminator: chan<bool> out) {
        let (encode_req_s, encode_req_r) = chan<Req>("encode_req");
        let (encode_resp_s, encode_resp_r) = chan<Resp>("encode_resp");
        let (writer_req_s, writer_req_r) = chan<WriterReq>("writer_req");
        let (writer_data_s, writer_data_r) = chan<WriterData>("writer_data");
        let (writer_resp_s, writer_resp_r) = chan<WriterResp>("writer_resp");
        spawn FrameHeaderEncoder<
            TEST_WINDOW_LOG_MAX, TEST_DATA_W, TEST_ADDR_W, TEST_XFERS_FOR_HEADER>(
            encode_req_r, encode_resp_s, writer_req_s, writer_data_s, writer_resp_r);
        (terminator, encode_req_s, encode_resp_r, writer_req_r, writer_data_r, writer_resp_s)
    }

    init {  }

    next(state: ()) {
        let tests: TestCase[11] = [
            TestCase {
                header: FrameHeader {
                    window_size: u64:0x3a16b33f3da53a79,
                    frame_content_size: u64:0x3a16b33f3da53a79,
                    dictionary_id: u32:0,
                    content_checksum_flag: u1:1,
                },
                encoded: [u32:0xFD2FB528, u32:0xA53A79_E4, u32:0x16B33F3D, u32:0x0000003A, u32:0x0],
                fixed_size: u1:0,
                single_segment_flag: u1:1,
                length: HeaderSize:13,
            },
            TestCase {
                header: FrameHeader {
                    window_size: u64:0x3a16b33f3da53a79,
                    frame_content_size: u64:0x3a16b33f3da53a79,
                    dictionary_id: u32:0xCAFE,
                    content_checksum_flag: u1:1,
                },
                encoded:
                    [u32:0xFD2FB528, u32:0x79_CAFE_E6, u32:0x3F3DA53A, u32:0x003a16b3, u32:0x0],
                fixed_size: u1:0,
                single_segment_flag: u1:1,
                length: HeaderSize:15,
            },
            TestCase {
                header: FrameHeader {
                    window_size: u64:0x3a16b33f3da53a79,
                    frame_content_size: u64:0x3a16b33f3da53a79,
                    dictionary_id: u32:0,
                    content_checksum_flag: u1:1,
                },
                encoded: [u32:0xFD2FB528, u32:0xA53A79_E4, u32:0x16B33F3D, u32:0x0000003A, u32:0x0],
                fixed_size: u1:1,
                // case in fixed_size flag doesn't change anything
                single_segment_flag: u1:1,
                length: HeaderSize:13,
            },
            TestCase {
                header: FrameHeader {
                    window_size: u64:0x3a16b33f3da53a79,
                    frame_content_size: u64:0x3a,
                    dictionary_id: u32:0,
                    content_checksum_flag: u1:1,
                },
                encoded: [u32:0xFD2FB528, u32:0x3A_24, u32:0x0, u32:0x0, u32:0x0],
                fixed_size: u1:0,
                single_segment_flag: u1:1,
                length: HeaderSize:6,
            },
            TestCase {
                header: FrameHeader {
                    window_size: u64:0x3a16b33f3da53a79,
                    frame_content_size: u64:0x3a,
                    dictionary_id: u32:0,
                    content_checksum_flag: u1:1,
                },
                encoded: [u32:0xFD2FB528, u32:0x3A_E4, u32:0x0, u32:0x0, u32:0x0],
                fixed_size: u1:1,
                single_segment_flag: u1:1,
                length: HeaderSize:13,
            },
            TestCase {
                header: FrameHeader {
                    window_size: u64:0x3a16b33f3da53a79,
                    frame_content_size: u64:0x100,
                    // encoded as u16:0 due to 256 offset
                    dictionary_id: u32:0,
                    content_checksum_flag: u1:1,
                },
                encoded: [u32:0xFD2FB528, u32:0x0000_64, u32:0x0, u32:0x0, u32:0x0],
                fixed_size: u1:0,
                single_segment_flag: u1:1,
                length: HeaderSize:7,
            },
            TestCase {
                header: FrameHeader {
                    window_size: u64:0x3a16b33f3da53a79,
                    frame_content_size: u64:0x101,
                    // encoded as u16:1 due to 256 offset
                    dictionary_id: u32:0,
                    content_checksum_flag: u1:1,
                },
                encoded: [u32:0xFD2FB528, u32:0x0001_64, u32:0x0, u32:0x0, u32:0x0],
                fixed_size: u1:0,
                single_segment_flag: u1:1,
                length: HeaderSize:7,
            },
            TestCase {
                header: FrameHeader {
                    window_size: u64:0x3a16b33f3da53a79,
                    frame_content_size: u64:0x100ff,
                    // encoded as u16:0xFFFF due to 256 offset
                    dictionary_id: u32:0,
                    content_checksum_flag: u1:1,
                },
                encoded: [u32:0xFD2FB528, u32:0xFFFF_64, u32:0x0, u32:0x0, u32:0x0],
                fixed_size: u1:0,
                single_segment_flag: u1:1,
                length: HeaderSize:7,
            },
            TestCase {
                header: FrameHeader {
                    window_size: u64:0x400,
                    frame_content_size: u64:0x0,
                    dictionary_id: u32:0,
                    content_checksum_flag: u1:1,
                },
                encoded: [u32:0xFD2FB528, u32:0x00_04, u32:0x0, u32:0x0, u32:0x0],
                fixed_size: u1:0,
                single_segment_flag: u1:0,
                length: HeaderSize:6,
            },
            TestCase {
                header: FrameHeader {
                    window_size: u64:0x400,
                    frame_content_size: u64:0xFF,
                    dictionary_id: u32:0,
                    content_checksum_flag: u1:1,
                },
                encoded: [u32:0xFD2FB528, u32:0x00ff_00_84, u32:0x0, u32:0x0, u32:0x0],
                fixed_size: u1:0,
                single_segment_flag: u1:0,
                length: HeaderSize:10,
            },
            TestCase {
                header: FrameHeader {
                    window_size: u64:0x900,
                    frame_content_size: u64:0x100ff,
                    // encoded as u16:0xFFFF due to 256 offset
                    dictionary_id: u32:0,
                    content_checksum_flag: u1:1,
                },
                encoded: [u32:0xFD2FB528, u32:0xFFFF_09_44, u32:0x0, u32:0x0, u32:0x0],
                fixed_size: u1:0,
                single_segment_flag: u1:0,
                length: HeaderSize:8,
            },
        ];

        let tok = join();
        const ADDR = u16:0x1234;
        let tok = for ((_, test_case), tok): ((u32, TestCase), token) in enumerate(tests) {
            let tok = send(
                tok, encode_req_s,
                Req { addr: ADDR, header: test_case.header, fixed_size: test_case.fixed_size , single_segment_flag: test_case.single_segment_flag});
            let (tok, writer_req) = recv(tok, writer_req_r);

            assert_eq(
                writer_req, WriterReq { addr: ADDR, length: test_case.length as uN[TEST_ADDR_W] });

            let (words, last_word_len) = calc_words<TEST_DATA_W>(test_case.length);
            let (tok, len_sum) =
                for ((j, word), (tok, len_sum)): ((u32, u32), (token, HeaderSize)) in
                    enumerate(test_case.encoded) {
                    let j = j as HeaderSize;
                    if j < words {
                        let last = j + HeaderSize:1 == words;
                        let length = if last {
                            last_word_len as uN[TEST_ADDR_W]
                        } else {
                            (TEST_DATA_W / u32:8) as uN[TEST_ADDR_W]
                        };
                        let (tok, writer_data) = recv(tok, writer_data_r);
                        assert_eq(writer_data, WriterData { data: word, length, last });
                        (tok, len_sum + length as HeaderSize)
                    } else {
                        (tok, len_sum)
                    }
                }((tok, HeaderSize:0));

            assert_eq(len_sum, test_case.length);

            let tok = send(tok, writer_resp_s, MEM_WRITER_OK_RESP);
            let (tok, encode_resp) = recv(tok, encode_resp_r);
            assert_eq(
                encode_resp,
                Resp { status: FrameHeaderEncoderStatus::OKAY, length: test_case.length });

            tok
        }(tok);

        send(tok, terminator, true);
    }
}

// Largest allowed WindowLog accepted by libzstd decompression function
// https://github.com/facebook/zstd/blob/v1.4.7/lib/decompress/zstd_decompress.c#L296
// Use only in C++ tests when comparing DSLX ZSTD Decoder with libzstd
pub const TEST_WINDOW_LOG_MAX_LIBZSTD = u32:30;

proc FrameHeaderEncoderInst {
    type Req = FrameHeaderEncoderReq<TEST_ADDR_W>;
    type Resp = FrameHeaderEncoderResp;
    type WriterReq = mem_writer::MemWriterReq<TEST_ADDR_W>;
    type WriterData = mem_writer::MemWriterDataPacket<TEST_DATA_W, TEST_ADDR_W>;
    type WriterResp = mem_writer::MemWriterResp;
    type TestCase = FrameHeaderEncoderTestCase<TEST_DATA_W, TEST_XFERS_FOR_HEADER>;
    encode_req_r: chan<Req> in;
    encode_resp_s: chan<Resp> out;
    writer_req_s: chan<WriterReq> out;
    writer_data_s: chan<WriterData> out;
    writer_resp_r: chan<WriterResp> in;

    config(encode_req_r: chan<Req> in, encode_resp_s: chan<Resp> out,
           writer_req_s: chan<WriterReq> out, writer_data_s: chan<WriterData> out,
           writer_resp_r: chan<WriterResp> in) {
        spawn FrameHeaderEncoder<
            TEST_WINDOW_LOG_MAX, TEST_DATA_W, TEST_ADDR_W, TEST_XFERS_FOR_HEADER>(
            encode_req_r, encode_resp_s, writer_req_s, writer_data_s, writer_resp_r);
        (encode_req_r, encode_resp_s, writer_req_s, writer_data_s, writer_resp_r)
    }

    init {  }

    next(state: ()) {  }
}
