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

// This file contains utilities related to ZSTD Frame Header parsing.
// More information about the ZSTD Frame Header can be found in:
// https://datatracker.ietf.org/doc/html/rfc8878#section-3.1.1.1

import std;
import xls.modules.zstd.memory.mem_reader;

pub type WindowSize = u64;
pub type FrameContentSize = u64;
pub type DictionaryId = u32;

// Structure for data obtained from decoding the Frame_Header_Descriptor
pub struct FrameHeader {
    window_size: WindowSize,
    frame_content_size: FrameContentSize,
    dictionary_id: DictionaryId,
    content_checksum_flag: u1,
}

// Status values reported by the frame header parsing function
pub enum FrameHeaderDecoderStatus: u2 {
    OKAY = 0,
    CORRUPTED = 1,
    UNSUPPORTED_WINDOW_SIZE = 2,
}

pub struct FrameHeaderDecoderReq<ADDR_W: u32> {
    addr: uN[ADDR_W],
}

pub struct FrameHeaderDecoderResp {
    status: FrameHeaderDecoderStatus,
    header: FrameHeader,
    length: u5,
}

// Maximal mantissa value for calculating maximal accepted window_size
// as per https://datatracker.ietf.org/doc/html/rfc8878#name-window-descriptor
const MAX_MANTISSA = WindowSize:0b111;

// Structure for holding ZSTD Frame_Header_Descriptor data, as in:
// https://datatracker.ietf.org/doc/html/rfc8878#section-3.1.1.1.1
pub struct FrameHeaderDescriptor {
    frame_content_size_flag: u2,
    single_segment_flag: u1,
    unused: u1,
    reserved: u1,
    content_checksum_flag: u1,
    dictionary_id_flag: u2,
}

// Auxiliary constant that can be used to initialize Proc's state
// with empty FrameHeader, because `zero!` cannot be used in that context
pub const ZERO_FRAME_HEADER = zero!<FrameHeader>();
pub const FRAME_CONTENT_SIZE_NOT_PROVIDED_VALUE = FrameContentSize::MAX;

// Extracts Frame_Header_Descriptor fields from 8-bit chunk of data
// that is assumed to be a valid Frame_Header_Descriptor
fn extract_frame_header_descriptor(data:u8) -> FrameHeaderDescriptor {
    FrameHeaderDescriptor {
        frame_content_size_flag: data[6:8],
        single_segment_flag: data[5:6],
        unused: data[4:5],
        reserved: data[3:4],
        content_checksum_flag: data[2:3],
        dictionary_id_flag: data[0:2],
    }
}

#[test]
fn test_extract_frame_header_descriptor() {
    assert_eq(
        extract_frame_header_descriptor(u8:0xA4),
        FrameHeaderDescriptor {
            frame_content_size_flag: u2:0x2,
            single_segment_flag: u1:0x1,
            unused: u1:0x0,
            reserved: u1:0x0,
            content_checksum_flag: u1:0x1,
            dictionary_id_flag: u2:0x0
        }
    );

    assert_eq(
        extract_frame_header_descriptor(u8:0x0),
        FrameHeaderDescriptor {
            frame_content_size_flag: u2:0x0,
            single_segment_flag: u1:0x0,
            unused: u1:0x0,
            reserved: u1:0x0,
            content_checksum_flag: u1:0x0,
            dictionary_id_flag: u2:0x0
        }
    );
}

// Returns a boolean showing if the Window_Descriptor section exists
// for the frame with the given FrameHeaderDescriptor
fn window_descriptor_exists(desc: FrameHeaderDescriptor) -> bool {
    desc.single_segment_flag == u1:0
}

#[test]
fn test_window_descriptor_exists() {
    let zero_desc = zero!<FrameHeaderDescriptor>();

    let desc_with_ss = FrameHeaderDescriptor {single_segment_flag: u1:1, ..zero_desc};
    assert_eq(window_descriptor_exists(desc_with_ss), false);

    let desc_without_ss = FrameHeaderDescriptor {single_segment_flag: u1:0, ..zero_desc};
    assert_eq(window_descriptor_exists(desc_without_ss), true);
}

// Extracts window size from 8-bit chunk of data
// that is assumed to be a valid Window_Descriptor
fn extract_window_size_from_window_descriptor(data: u8) -> u64 {
    let exponent = data[3:8];
    let mantissa = data[0:3];

    let window_base = (u42:1 << (u6:10 + exponent as u6));
    let window_base_add = (window_base >> u2:3) as u42;
    // optimization: perform multiplication by a 3-bit value with adds and shifts
    // because XLS only allows multiplying operands of the same width
    let window_add = match mantissa {
        u3:0 => u42:0,
        u3:1 => window_base_add,                                        // u39
        u3:2 => window_base_add + window_base_add,                      // u39 + u39 = u40
        u3:3 => (window_base_add << u1:1) + window_base_add,            // u40 + u39 = u41
        u3:4 => (window_base_add << u1:1) + (window_base_add << u1:1),  // u40 + u40 = u41
        u3:5 => (window_base_add << u2:2) + window_base_add,            // u41 + u39 = u42
        u3:6 => (window_base_add << u2:2) + (window_base_add << u2:1),  // u41 + u40 = u42
        u3:7 => (window_base_add << u2:3) - window_base_add,            // u42 - u39 = u42
        _ => fail!("extract_window_size_from_window_descriptor_unreachable", u42:0),
    };

    window_base as u64 + window_add as u64
}

#[test]
fn test_extract_window_size_from_window_descriptor() {
    assert_eq(extract_window_size_from_window_descriptor(u8:0x0), u64:0x400);
    assert_eq(extract_window_size_from_window_descriptor(u8:0x9), u64:0x900);
    assert_eq(extract_window_size_from_window_descriptor(u8:0xFF), u64:0x3c000000000);
}

// Returns boolean showing if the Frame_Content_Size section exists for
// the frame with the given FrameHeaderDescriptor.
fn frame_content_size_exists(desc: FrameHeaderDescriptor) -> bool {
    desc.single_segment_flag != u1:0 || desc.frame_content_size_flag != u2:0
}

#[test]
fn test_frame_content_size_exists() {
    let zero_desc = zero!<FrameHeaderDescriptor>();

    let desc = FrameHeaderDescriptor {single_segment_flag: u1:0, frame_content_size_flag: u2:0, ..zero_desc};
    assert_eq(frame_content_size_exists(desc), false);

    let desc = FrameHeaderDescriptor {single_segment_flag: u1:0, frame_content_size_flag: u2:2, ..zero_desc};
    assert_eq(frame_content_size_exists(desc), true);

    let desc = FrameHeaderDescriptor {single_segment_flag: u1:1, frame_content_size_flag: u2:0, ..zero_desc};
    assert_eq(frame_content_size_exists(desc), true);

    let desc = FrameHeaderDescriptor {single_segment_flag: u1:1, frame_content_size_flag: u2:3, ..zero_desc};
    assert_eq(frame_content_size_exists(desc), true);
}


// Calculate maximal accepted window_size for given WINDOW_LOG_MAX and return whether given
// window_size should be accepted or discarded.
// Based on window_size calculation from: RFC 8878
// https://datatracker.ietf.org/doc/html/rfc8878#name-window-descriptor
fn window_size_valid<WINDOW_LOG_MAX: u32>(window_size: WindowSize) -> bool {
    let max_window_size = (WindowSize:1 << WINDOW_LOG_MAX) + (((WindowSize:1 << WINDOW_LOG_MAX) >> WindowSize:3) * MAX_MANTISSA);

    window_size <= max_window_size
}


pub fn parse_frame_header(header_raw: uN[112]) -> (FrameHeader, u4, u1) {
    let fhd_raw = header_raw[0:8];
    let fhd = extract_frame_header_descriptor(fhd_raw);
    // RFC8878 Section 3.1.1.1.1.4
    // "This [reserved] bit is reserved for some future feature. Its value
    //  must be zero. A decoder compliant with this specification version must
    //  ensure it is not set."
    let header_ok = !fhd.reserved;

    let window_descriptor_start = u32:1;
    // RFC8878 Section 3.1.1.1.2
    // "When Single_Segment_Flag is set, Window_Descriptor is not present."
    let window_descriptor_len = match fhd.single_segment_flag {
        u1:0 => u1:1,
        u1:1 => u1:0,
        _ => fail!("window_descriptor_len_unreachable", u1:0),
    };
    let window_descriptor_raw = header_raw[u32:8*window_descriptor_start+:u8];
    let window_size = extract_window_size_from_window_descriptor(window_descriptor_raw);

    let dictionary_id_start = window_descriptor_start + window_descriptor_len as u32;
    let dictionary_id_len = match fhd.dictionary_id_flag {
        u2:0 => u32:0,
        u2:1 => u32:1,
        u2:2 => u32:2,
        u2:3 => u32:4,
        _ => fail!("dictionary_id_len_unreachable", u32:0),
    };
    let dictionary_id_raw = header_raw[u32:8*dictionary_id_start+:u32];
    let dictionary_id = dictionary_id_raw & match fhd.dictionary_id_flag {
        u2:0 => u32:0x0000_0000,
        u2:1 => u32:0x0000_00ff,
        u2:2 => u32:0x0000_ffff,
        u2:3 => u32:0xffff_ffff,
        _ => fail!("dictionary_id_unreachable", u32:0),
    };

    let frame_content_size_start = dictionary_id_start + dictionary_id_len;
    // RFC8878 Section 3.1.1.1.1.1
    // "When Frame_Content_Size_Flag is 0, FCS_Field_Size depends on
    //  Single_Segment_Flag: If Single_Segment_Flag is set, FCS_Field_Siz
    //  is 1. Otherwise, FCS_Field_Size is 0;"
    let frame_content_size_len = match (fhd.frame_content_size_flag, fhd.single_segment_flag) {
        (u2:0, u1:0) => u32:0,
        (u2:0, u1:1) => u32:1,
        (u2:1, _)    => u32:2,
        (u2:2, _)    => u32:4,
        (u2:3, _)    => u32:8,
        _ => fail!("frame_content_size_len_unreachable", u32:0),
    };

    let frame_content_size_raw = header_raw[u32:8*frame_content_size_start+:u64];
    let frame_content_size_masked = frame_content_size_raw & match frame_content_size_len {
        u32:0 => u64:0x0000_0000_0000_0000,
        u32:1 => u64:0x0000_0000_0000_00ff,
        u32:2 => u64:0x0000_0000_0000_ffff,
        u32:4 => u64:0x0000_0000_ffff_ffff,
        u32:8 => u64:0xffff_ffff_ffff_ffff,
        _ => fail!("frame_content_size_masked_unreachable", u64:0),
    };

    // RFC8878 Section 3.1.1.1.4
    // "When FCS_Field_Size is 2, the offset of 256 is added."
    let frame_content_size = frame_content_size_masked + match frame_content_size_len {
        u32:2 => u64:256,
        _     => u64:0,
    };

    // RFC8878 Section 3.1.1.1.2
    // "When Single_Segment_Flag is set, Window_Descriptor is not present.
    //  In this case, Window_Size is Frame_Content_Size [...]"
    let window_size = if (window_descriptor_exists(fhd)) {
        window_size
    } else if (frame_content_size_exists(fhd)) {
        frame_content_size
    } else {
        WindowSize:0
    };

    let total_header_len = (frame_content_size_start + frame_content_size_len) as u4;

    (FrameHeader {
        window_size: window_size,
        frame_content_size: if frame_content_size_len != u32:0 { frame_content_size } else { FrameContentSize:0 },
        dictionary_id: if dictionary_id_len != u32:0 { dictionary_id } else { DictionaryId:0 },
        content_checksum_flag: fhd.content_checksum_flag,
    }, total_header_len, header_ok)
}


#[test]
fn test_parse_frame_header() {
    // normal case
    let test_vec = uN[112]:0x1234567890ABCDEF_CAFE_09_C2;
    let (frame_header_result, len, ok) = parse_frame_header(test_vec);
    assert_eq(frame_header_result, FrameHeader {
        window_size: u64:0x900,
        frame_content_size: u64:0x1234567890ABCDEF,
        dictionary_id: u32:0xCAFE,
        content_checksum_flag: u1:0,
    });
    assert_eq(len, u4:12);
    assert_eq(ok, u1:1);

    // SingleSegmentFlag is set
    let test_vec = uN[112]:0xaa20;
    let (frame_header_result, len, ok) = parse_frame_header(test_vec);
    assert_eq(frame_header_result, FrameHeader {
                window_size: u64:0xaa,
                frame_content_size: u64:0xaa,
                dictionary_id: u32:0x0,
                content_checksum_flag: u1:0,
            });
    assert_eq(len, u4:2);
    assert_eq(ok, u1:1);

    // SingleSegmentFlag is set and FrameContentSize is bigger than accepted window_size
    let test_vec = uN[112]:0x1234567890ABCDEF_CAFE_E2;
    let (frame_header_result, len, ok) = parse_frame_header(test_vec);
    assert_eq(frame_header_result, FrameHeader {
        window_size: u64:0x1234567890ABCDEF,
        frame_content_size: u64:0x1234567890ABCDEF,
        dictionary_id: u32:0xCAFE,
        content_checksum_flag: u1:0,
    });
    assert_eq(len, u4:11);
    assert_eq(ok, u1:1);

    // Frame header descriptor is corrupted (we don't check frame header and length)
    let test_vec = uN[112]:0x1234567890ABCDEF_1234_09_CA;
    let (_, _, ok) = parse_frame_header(test_vec);
    assert_eq(ok, u1:0);

    // Large window size
    let test_vec = uN[112]:0xd310;
    let (frame_header_result, len, ok) = parse_frame_header(test_vec);
    assert_eq(frame_header_result, FrameHeader {
        window_size: u64:0x1600000000,
        ..zero!<FrameHeader>()
    });
    assert_eq(len, u4:2);
    assert_eq(ok, u1:1);

    // Large window size
    let test_vec = uN[112]:0xf45b5b5b0db1;
    let (frame_header_result, len, ok) = parse_frame_header(test_vec);
    assert_eq(frame_header_result, FrameHeader {
        window_size: u64:0xf45b5b5b,
        frame_content_size: u64:0xf45b5b5b,
        dictionary_id: u32:0xD,
        content_checksum_flag: u1:0,
    });
    assert_eq(len, u4:6);
    assert_eq(ok, u1:1);

    // Large window size
    let test_vec = uN[112]:0xc0659db6813a16b33f3da53a79e4;
    let (frame_header_result, len, ok) = parse_frame_header(test_vec);
    assert_eq(frame_header_result, FrameHeader {
        window_size: u64:0x3a16b33f3da53a79,
        frame_content_size: u64:0x3a16b33f3da53a79,
        dictionary_id: u32:0,
        content_checksum_flag: u1:1,
    });
    assert_eq(len, u4:9);
    assert_eq(ok, u1:1);
}


enum FrameHeaderDecoderFsm: u1 {
    RECV = 0,
    RESP = 1
}

// Magic number value, as in:
// https://datatracker.ietf.org/doc/html/rfc8878#section-3.1.1
const MAGIC_NUMBER = u32:0xFD2FB528;
const MAGIC_NUMBER_LEN = u32:4;

const MAX_HEADER_LEN = u32:14;
const MAX_MAGIC_PLUS_HEADER_LEN = MAGIC_NUMBER_LEN + MAX_HEADER_LEN;

struct FrameHeaderDecoderState<XFER_SIZE: u32, XFER_COUNT: u32> {
    fsm: FrameHeaderDecoderFsm,
    xfers: u32,
    raw_header: uN[XFER_SIZE][XFER_COUNT],
}

pub proc FrameHeaderDecoder<
    WINDOW_LOG_MAX: u32,
    DATA_W: u32,
    ADDR_W: u32,
    XFERS_FOR_HEADER: u32 = {((MAX_MAGIC_PLUS_HEADER_LEN * u32:8) / DATA_W) + u32:1},
> {
    type State = FrameHeaderDecoderState<DATA_W, XFERS_FOR_HEADER>;
    type Fsm = FrameHeaderDecoderFsm;
    type Req = FrameHeaderDecoderReq<ADDR_W>;
    type Resp = FrameHeaderDecoderResp;
    type ReaderReq = mem_reader::MemReaderReq<ADDR_W>;
    type ReaderResp = mem_reader::MemReaderResp<DATA_W, ADDR_W>;

    reader_req_s: chan<ReaderReq> out;
    reader_resp_r: chan<ReaderResp> in;

    decode_req_r: chan<Req> in;
    decode_resp_s: chan<Resp> out;

    config(
        reader_req_s: chan<ReaderReq> out,
        reader_resp_r: chan<ReaderResp> in,
        decode_req_r: chan<Req> in,
        decode_resp_s: chan<Resp> out
    ) {
        (reader_req_s, reader_resp_r, decode_req_r, decode_resp_s)
    }

    init { zero!<State>() }

    next(state: State) {
        type ReaderReq = mem_reader::MemReaderReq<ADDR_W>;
        type State = FrameHeaderDecoderState<DATA_W, XFERS_FOR_HEADER>;

        let tok0 = join();
        let (tok_req, req, do_req) = recv_non_blocking(tok0, decode_req_r, zero!<Req>());
        send_if(tok_req, reader_req_s, do_req, ReaderReq { addr: req.addr, length: MAX_MAGIC_PLUS_HEADER_LEN as uN[ADDR_W] });

        let do_recv = (state.fsm == Fsm::RECV);
        let (tok, resp, recvd) = recv_if_non_blocking(tok0, reader_resp_r, do_recv, zero!<ReaderResp>());

        let do_resp = (state.fsm == Fsm::RESP);
        let raw_header_bits = state.raw_header as uN[DATA_W * XFERS_FOR_HEADER];
        let raw_magic_number = raw_header_bits[:s32:8 * MAGIC_NUMBER_LEN as s32];
        let raw_header = raw_header_bits[s32:8 * MAGIC_NUMBER_LEN as s32 : s32:8 * MAX_MAGIC_PLUS_HEADER_LEN as s32];
        let magic_number_ok = raw_magic_number == MAGIC_NUMBER;
        let (decoded_header, header_len, header_ok) = parse_frame_header(raw_header);

        let status = if (!header_ok || !magic_number_ok) {
            FrameHeaderDecoderStatus::CORRUPTED
        } else if (!window_size_valid<WINDOW_LOG_MAX>(decoded_header.window_size)) {
            FrameHeaderDecoderStatus::UNSUPPORTED_WINDOW_SIZE
        } else {
            FrameHeaderDecoderStatus::OKAY
        };

        let header_result = FrameHeaderDecoderResp {
            status: status,
            header: decoded_header,
            length: header_len as u5 + MAGIC_NUMBER_LEN as u5,
        };

        send_if(tok0, decode_resp_s, do_resp, header_result);

        let next_state = match (state.fsm) {
            Fsm::RECV => {
                if (recvd) {
                    // raw_header is updated from the highest to lowest index because
                    // highest index in an array contains least significant bytes when
                    // casting to a bit vector
                    let update_idx = XFERS_FOR_HEADER - state.xfers - u32:1;
                    let next_raw_header = update(state.raw_header, update_idx, resp.data);
                    if (resp.last) {
                        State { raw_header: next_raw_header, fsm: Fsm::RESP, ..state }
                    } else {
                        State { raw_header: next_raw_header, xfers: state.xfers + u32:1, ..state }
                    }
                } else {
                    state
                }
            },
            Fsm::RESP => {
                State { fsm: Fsm::RECV, xfers: u32:0, ..state }
            },
            _ => fail!("FrameHeaderDecoder_fsm_unreachable", zero!<State>())
        };

        next_state
    }
}

// The largest allowed WindowLog for DSLX tests
pub const TEST_WINDOW_LOG_MAX = u32:22;
pub const TEST_DATA_W = u32:32;
pub const TEST_ADDR_W = u32:16;
pub const TEST_XFERS_FOR_HEADER = ((MAX_MAGIC_PLUS_HEADER_LEN * u32:8) / TEST_DATA_W) + u32:1;

#[test_proc]
proc FrameHeaderDecoderTest {
    type Req = FrameHeaderDecoderReq<u32:16>;
    type Resp = FrameHeaderDecoderResp;
    type ReaderReq = mem_reader::MemReaderReq<u32:16>;
    type ReaderResp = mem_reader::MemReaderResp<u32:32, u32:16>;

    terminator: chan<bool> out;

    reader_req_r: chan<ReaderReq> in;
    reader_resp_s: chan<ReaderResp> out;

    decode_req_s: chan<Req> out;
    decode_resp_r: chan<Resp> in;

    config(terminator: chan<bool> out) {
        let (reader_req_s, reader_req_r) = chan<ReaderReq>("reader_req");
        let (reader_resp_s, reader_resp_r) = chan<ReaderResp>("reader_resp");
        let (decode_req_s, decode_req_r) = chan<Req>("decode_req");
        let (decode_resp_s, decode_resp_r) = chan<Resp>("decode_resp");
        spawn FrameHeaderDecoder<TEST_WINDOW_LOG_MAX, TEST_DATA_W, TEST_ADDR_W, TEST_XFERS_FOR_HEADER>(
            reader_req_s,
            reader_resp_r,
            decode_req_r,
            decode_resp_s
        );
        (terminator, reader_req_r, reader_resp_s, decode_req_s, decode_resp_r)
    }

    init {}

    next(state: ()) {
        let tok = join();
        let tests: (u32[TEST_XFERS_FOR_HEADER], FrameHeaderDecoderResp)[7] = [
            (
                // normal case
                [u32:0xFD2FB528, u32:0xCAFE_09_C2, u32:0x90ABCDEF, u32:0x12345678, u32:0x0],
                FrameHeaderDecoderResp {
                    header: FrameHeader {
                        window_size: u64:0x900,
                        frame_content_size: u64:0x1234567890ABCDEF,
                        dictionary_id: u32:0xCAFE,
                        content_checksum_flag: u1:0,
                    },
                    status: FrameHeaderDecoderStatus::OKAY,
                    length: u5:16
                },
            ), (
                // SingleSegmentFlag is set
                [u32:0xFD2FB528, u32:0xAA20, u32:0x0, u32:0x0, u32:0x0],
                FrameHeaderDecoderResp {
                    header: FrameHeader {
                        window_size: u64:0xaa,
                        frame_content_size: u64:0xaa,
                        dictionary_id: u32:0x0,
                        content_checksum_flag: u1:0,
                    },
                    status: FrameHeaderDecoderStatus::OKAY,
                    length: u5:6
                },
            ), (
                // SingleSegmentFlag is set and FrameContentSize is bigger than accepted window_size
                [u32:0xFD2FB528, u32:0xEF_CAFE_E2, u32:0x7890ABCD, u32:0x123456, u32:0x0],
                FrameHeaderDecoderResp {
                    header: FrameHeader {
                        window_size: u64:0x1234567890ABCDEF,
                        frame_content_size: u64:0x1234567890ABCDEF,
                        dictionary_id: u32:0xCAFE,
                        content_checksum_flag: u1:0,
                    },
                    status: FrameHeaderDecoderStatus::UNSUPPORTED_WINDOW_SIZE,
                    length: u5:15
                },
            ), (
                // Frame header descriptor is corrupted (we don't check 'header' and 'length' fields)
                [u32:0xFD2FB528, u32:0x1234_09_CA, u32:0x90ABCDEF, u32:0x12345678, u32:0x0],
                FrameHeaderDecoderResp {
                    header: zero!<FrameHeader>(),
                    status: FrameHeaderDecoderStatus::CORRUPTED,
                    length: u5:0
                },
            ), (
                // Window size required by frame is too big for given decoder configuration
                [u32:0xFD2FB528, u32:0xD310, u32:0x0, u32:0x0, u32:0x0],
                FrameHeaderDecoderResp {
                    header: FrameHeader {
                        window_size: u64:0x1600000000,
                        ..zero!<FrameHeader>()
                    },
                    status: FrameHeaderDecoderStatus::UNSUPPORTED_WINDOW_SIZE,
                    length: u5:6
                },
            ), (
                // Window size required by frame is too big for given decoder configuration
                [u32:0xFD2FB528, u32:0x5B5B0DB1, u32:0xF45B, u32:0x0, u32:0x0],
                FrameHeaderDecoderResp {
                    header: FrameHeader {
                        window_size: u64:0xf45b5b5b,
                        frame_content_size: u64:0xf45b5b5b,
                        dictionary_id: u32:0xD,
                        content_checksum_flag: u1:0,
                    },
                    status: FrameHeaderDecoderStatus::UNSUPPORTED_WINDOW_SIZE,
                    length: u5:10
                },
            ), (
                // Window size required by frame is too big for given decoder configuration
                [u32:0xFD2FB528, u32:0xA53A79E4, u32:0x16B33F3D, u32:0x9DB6813A, u32:0xC065],
                FrameHeaderDecoderResp {
                    header: FrameHeader {
                        window_size: u64:0x3a16b33f3da53a79,
                        frame_content_size: u64:0x3a16b33f3da53a79,
                        dictionary_id: u32:0,
                        content_checksum_flag: u1:1,
                    },
                    status: FrameHeaderDecoderStatus::UNSUPPORTED_WINDOW_SIZE,
                    length: u5:13
                }
            )
        ];

        const ADDR = u16:0x1234;
        let tok = for ((_, (test_vec, expected)), tok): ((u32, (u32[TEST_XFERS_FOR_HEADER], FrameHeaderDecoderResp)), token) in enumerate(tests) {
            let tok = send(tok, decode_req_s, FrameHeaderDecoderReq { addr: ADDR });
            let (tok, recv_data) = recv(tok, reader_req_r);

            assert_eq(recv_data, ReaderReq { addr: ADDR, length: MAX_MAGIC_PLUS_HEADER_LEN as u16 });

            let tok = for ((j, word), tok): ((u32, u32), token) in enumerate(test_vec) {
                let last = j + u32:1 == array_size(test_vec);
                send(tok, reader_resp_s, ReaderResp {
                    status: mem_reader::MemReaderStatus::OKAY,
                    data: word,
                    length: if !last { (TEST_DATA_W / u32:8) as u16 } else { (MAX_MAGIC_PLUS_HEADER_LEN % TEST_XFERS_FOR_HEADER) as u16 },
                    last: last,
                })
            }(tok);

            let (tok, recv_data) = recv(tok, decode_resp_r);
            if (recv_data.status == FrameHeaderDecoderStatus::OKAY || recv_data.status == FrameHeaderDecoderStatus::UNSUPPORTED_WINDOW_SIZE) {
                assert_eq(recv_data, expected);
            } else {
                // if the header is corrupted we don't offer any guarantees
                // about its contents so we just check that the status matches
                assert_eq(recv_data.status, expected.status);
            };

            tok
        }(tok);

        send(tok, terminator, true);
    }
}


// Largest allowed WindowLog accepted by libzstd decompression function
// https://github.com/facebook/zstd/blob/v1.4.7/lib/decompress/zstd_decompress.c#L296
// Use only in C++ tests when comparing DSLX ZSTD Decoder with libzstd
pub const TEST_WINDOW_LOG_MAX_LIBZSTD = u32:30;

proc FrameHeaderDecoderInst {
    type Req = FrameHeaderDecoderReq<u32:16>;
    type Resp = FrameHeaderDecoderResp;
    type ReaderReq = mem_reader::MemReaderReq<u32:16>;
    type ReaderResp = mem_reader::MemReaderResp<u32:32, u32:16>;

    reader_req_s: chan<ReaderReq> out;
    reader_resp_r: chan<ReaderResp> in;

    decode_req_r: chan<Req> in;
    decode_resp_s: chan<Resp> out;

    config(
        reader_req_s: chan<ReaderReq> out,
        reader_resp_r: chan<ReaderResp> in,
        decode_req_r: chan<Req> in,
        decode_resp_s: chan<Resp> out,
    ) {
        spawn FrameHeaderDecoder<TEST_WINDOW_LOG_MAX_LIBZSTD, TEST_DATA_W, TEST_ADDR_W, TEST_XFERS_FOR_HEADER>(
            reader_req_s,
            reader_resp_r,
            decode_req_r,
            decode_resp_s
        );
        (reader_req_s, reader_resp_r, decode_req_r, decode_resp_s)
    }

    init {}

    next(state: ()) {}
}
