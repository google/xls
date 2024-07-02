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
import xls.modules.zstd.buffer as buff;

type Buffer = buff::Buffer;
type BufferStatus = buff::BufferStatus;
type BufferResult = buff::BufferResult;

pub type WindowSize = u64;
type FrameContentSize = u64;
type DictionaryId = u32;

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

// Structure for data obtained from decoding the Frame_Header_Descriptor
pub struct FrameHeader {
    window_size: WindowSize,
    frame_content_size: FrameContentSize,
    dictionary_id: DictionaryId,
    content_checksum_flag: u1,
}

// Status values reported by the frame header parsing function
pub enum FrameHeaderStatus: u2 {
    OK = 0,
    CORRUPTED = 1,
    NO_ENOUGH_DATA = 2,
    UNSUPPORTED_WINDOW_SIZE = 3,
}

// structure for returning results of parsing a frame header
pub struct FrameHeaderResult<CAPACITY: u32> {
    status: FrameHeaderStatus,
    header: FrameHeader,
    buffer: Buffer<CAPACITY>,
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

// Parses a Buffer and extracts information from the Frame_Header_Descriptor.
// The Buffer is assumed to contain a valid Frame_Header_Descriptor. The function
// returns BufferResult with the outcome of the operations on the buffer and
// information extracted from the Frame_Header_Descriptor
fn parse_frame_header_descriptor<CAPACITY: u32>(buffer: Buffer<CAPACITY>) -> (BufferResult<CAPACITY>, FrameHeaderDescriptor) {
    let (result, data) = buff::buffer_fixed_pop_checked<u32:8>(buffer);
    match result.status {
        BufferStatus::OK => {
            let frame_header_desc = extract_frame_header_descriptor(data);
            (result, frame_header_desc)
        },
        _ => (result, zero!<FrameHeaderDescriptor>())
    }
}

#[test]
fn test_parse_frame_header_descriptor() {
    let buffer = Buffer { content: u32:0xA4, length: u32:8 };
    let (result, header) = parse_frame_header_descriptor(buffer);
    assert_eq(result, BufferResult {
        status: BufferStatus::OK,
        buffer: Buffer { content: u32:0, length: u32:0 },
    });
    assert_eq(header, FrameHeaderDescriptor {
        frame_content_size_flag: u2:0x2,
        single_segment_flag: u1:0x1,
        unused: u1:0x0,
        reserved: u1:0x0,
        content_checksum_flag: u1:0x1,
        dictionary_id_flag: u2:0x0
    });

    let buffer = Buffer { content: u32:0x0, length: u32:8 };
    let (result, header) = parse_frame_header_descriptor(buffer);
    assert_eq(result, BufferResult {
        status: BufferStatus::OK,
        buffer: Buffer { content: u32:0, length: u32:0 },
    });
    assert_eq(header, FrameHeaderDescriptor {
        frame_content_size_flag: u2:0x0,
        single_segment_flag: u1:0x0,
        unused: u1:0x0,
        reserved: u1:0x0,
        content_checksum_flag: u1:0x0,
        dictionary_id_flag: u2:0x0
    });

    let buffer = Buffer { content: u32:0x0, length: u32:0 };
    let (result, header) = parse_frame_header_descriptor(buffer);
    assert_eq(result, BufferResult {
        status: BufferStatus::NO_ENOUGH_DATA,
        buffer: Buffer { content: u32:0, length: u32:0 },
    });
    assert_eq(header, zero!<FrameHeaderDescriptor>());
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
    let exponent = data >> u8:3;
    let mantissa = data & u8:7;

    let window_base = u64:1 << (u64:10 + exponent as u64);
    let window_add = (window_base >> u64:3) * (mantissa as u64);

    window_base + window_add
}

#[test]
fn test_extract_window_size_from_window_descriptor() {
    assert_eq(extract_window_size_from_window_descriptor(u8:0x0), u64:0x400);
    assert_eq(extract_window_size_from_window_descriptor(u8:0x9), u64:0x900);
    assert_eq(extract_window_size_from_window_descriptor(u8:0xFF), u64:0x3c000000000);
}

// Parses a Buffer with data and extracts information from the Window_Descriptor
// The buffer is assumed to contain a valid Window_Descriptor that is related to
// the same frame as the provided FrameHeaderDescriptor. The function returns
// BufferResult with the outcome of the operations on the buffer and window size.
fn parse_window_descriptor<CAPACITY: u32>(buffer: Buffer<CAPACITY>, desc: FrameHeaderDescriptor) -> (BufferResult<CAPACITY>, WindowSize) {
    assert!(window_descriptor_exists(desc), "window_descriptor_does_not_exist");

    let (result, data) = buff::buffer_fixed_pop_checked<u32:8>(buffer);
    match result.status {
        BufferStatus::OK => {
            let window_size = extract_window_size_from_window_descriptor(data);
            (result, window_size)
        },
        _ => (result, u64:0)
    }
}

#[test]
fn test_parse_window_descriptor() {
    let zero_desc = zero!<FrameHeaderDescriptor>();
    let desc_without_ss = FrameHeaderDescriptor {single_segment_flag: u1:0, ..zero_desc};

    let buffer = Buffer { content: u32:0xF, length: u32:0x4 };
    let (result, window_size) = parse_window_descriptor(buffer, desc_without_ss);
    assert_eq(result, BufferResult {
        status: BufferStatus::NO_ENOUGH_DATA,
        buffer: Buffer { content: u32:0xF, length: u32:0x4 },
    });
    assert_eq(window_size, u64:0);

    let buffer = Buffer { content: u32:0x0, length: u32:0x8 };
    let (result, window_size) = parse_window_descriptor(buffer, desc_without_ss);
    assert_eq(result, BufferResult {
        status: BufferStatus::OK,
        buffer: Buffer { content: u32:0x0, length: u32:0 },
    });
    assert_eq(window_size, u64:0x400);

    let buffer = Buffer { content: u32:0x9, length: u32:0x8 };
    let (result, window_size) = parse_window_descriptor(buffer, desc_without_ss);
    assert_eq(result, BufferResult {
        status: BufferStatus::OK,
        buffer: Buffer { content: u32:0x0, length: u32:0 },
    });
    assert_eq(window_size, u64:0x900);

    let buffer = Buffer { content: u32:0xFF, length: u32:0x8 };
    let (result, window_size) = parse_window_descriptor(buffer, desc_without_ss);
    assert_eq(result, BufferResult {
        status: BufferStatus::OK,
        buffer: Buffer { content: u32:0x0, length: u32:0 },
    });
    assert_eq(window_size, u64:0x3c000000000);
}

// Parses a Buffer with data and extracts information from the Dictionary_ID
// The buffer is assumed to contain a valid Dictionary_ID that is related to
// the same frame as the provided FrameHeaderDescriptor. The function returns
// BufferResult with the outcome of the operations on the buffer and dictionary ID
fn parse_dictionary_id<CAPACITY: u32>(buffer: Buffer<CAPACITY>, desc: FrameHeaderDescriptor) -> (BufferResult<CAPACITY>, DictionaryId) {
    let bytes = match desc.dictionary_id_flag {
      u2:0 => u32:0,
      u2:1 => u32:1,
      u2:2 => u32:2,
      u2:3 => u32:4,
      _    => fail!("not_possible", u32:0)
    };

    let (result, data) = buff::buffer_pop_checked(buffer, bytes * u32:8);
    match result.status {
        BufferStatus::OK => (result, data as u32),
        _ => (result, u32:0)
    }
}

#[test]
fn test_parse_dictionary_id() {
    let zero_desc = zero!<FrameHeaderDescriptor>();

    let buffer = Buffer { content: u32:0x0, length: u32:0x0 };
    let frame_header_desc = FrameHeaderDescriptor { dictionary_id_flag: u2:0, ..zero_desc};
    let (result, dictionary_id) = parse_dictionary_id(buffer, frame_header_desc);
    assert_eq(result, BufferResult {
        status: BufferStatus::OK,
        buffer: Buffer { content: u32:0x0, length: u32:0x0 },
    });
    assert_eq(dictionary_id, u32:0);

    let buffer = Buffer { content: u32:0x12, length: u32:0x8 };
    let frame_header_desc = FrameHeaderDescriptor { dictionary_id_flag: u2:0x1, ..zero_desc};
    let (result, dictionary_id) = parse_dictionary_id(buffer, frame_header_desc);
    assert_eq(result, BufferResult {
        status: BufferStatus::OK,
        buffer: Buffer { content: u32:0x0, length: u32:0 },
    });
    assert_eq(dictionary_id, u32:0x12);

    let buffer = Buffer { content: u32:0x1234, length: u32:0x10 };
    let frame_header_desc = FrameHeaderDescriptor { dictionary_id_flag: u2:0x2, ..zero_desc};
    let (result, dictionary_id) = parse_dictionary_id(buffer, frame_header_desc);
    assert_eq(result, BufferResult {
        status: BufferStatus::OK,
        buffer: Buffer { content: u32:0x0, length: u32:0 },
    });
    assert_eq(dictionary_id, u32:0x1234);

    let buffer = Buffer { content: u32:0x12345678, length: u32:0x20 };
    let frame_header_desc = FrameHeaderDescriptor { dictionary_id_flag: u2:0x3, ..zero_desc};
    let (result, dictionary_id) = parse_dictionary_id(buffer, frame_header_desc);
    assert_eq(result, BufferResult {
        status: BufferStatus::OK,
        buffer: Buffer { content: u32:0x0, length: u32:0 },
    });
    assert_eq(dictionary_id, u32:0x12345678);

    let buffer = Buffer { content: u32:0x1234, length: u32:0x10 };
    let frame_header_desc = FrameHeaderDescriptor { dictionary_id_flag: u2:0x3, ..zero_desc};
    let (result, dictionary_id) = parse_dictionary_id(buffer, frame_header_desc);
    assert_eq(result, BufferResult {
        status: BufferStatus::NO_ENOUGH_DATA,
        buffer: Buffer { content: u32:0x1234, length: u32:0x10 },
    });
    assert_eq(dictionary_id, u32:0x0);
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

// Parses a Buffer with data and extracts information from the Frame_Content_Size
// The buffer is assumed to contain a valid Frame_Content_Size that is related to
// the same frame as the provided FrameHeaderDescriptor. The function returns
// BufferResult with the outcome of the operations on the buffer and frame content size.
fn parse_frame_content_size<CAPACITY: u32>(buffer: Buffer<CAPACITY>, desc: FrameHeaderDescriptor) -> (BufferResult<CAPACITY>, FrameContentSize) {
    assert!(frame_content_size_exists(desc), "frame_content_size_does_not_exist");

    let bytes = match desc.frame_content_size_flag {
      u2:0 => u32:1,
      u2:1 => u32:2,
      u2:2 => u32:4,
      u2:3 => u32:8,
      _    => fail!("not_possible", u32:0)
    };

    let (result, data) = buff::buffer_pop_checked(buffer, bytes * u32:8);
    match (result.status, bytes) {
        (BufferStatus::OK, u32:2) => (result, data as u64 + u64:256),
        (BufferStatus::OK, _) => (result, data as u64),
        (_, _) => (result, u64:0)
    }
}

#[test]
fn test_parse_frame_content_size() {
    let zero_desc = zero!<FrameHeaderDescriptor>();

    let buffer = Buffer { content: u64:0x12, length: u32:8 };
    let frame_header_desc = FrameHeaderDescriptor {
            frame_content_size_flag: u2:0,
            single_segment_flag: u1:1,
            ..zero_desc
        };
    let (result, frame_content_size) = parse_frame_content_size(buffer, frame_header_desc);
    assert_eq(result, BufferResult {
        status: BufferStatus::OK,
        buffer: Buffer { content: u64:0x0, length: u32:0x0 },
    });
    assert_eq(frame_content_size, u64:0x12);

    let buffer = Buffer { content: u64:0x1234, length: u32:0x10 };
    let frame_header_desc = FrameHeaderDescriptor { frame_content_size_flag: u2:1, ..zero_desc};
    let (result, frame_content_size) = parse_frame_content_size(buffer, frame_header_desc);
    assert_eq(result, BufferResult {
        status: BufferStatus::OK,
        buffer: Buffer { content: u64:0x0, length: u32:0x0 },
    });
    assert_eq(frame_content_size, u64:0x1234 + u64:256);

    let buffer = Buffer { content: u64:0x12345678, length: u32:0x20 };
    let frame_header_desc = FrameHeaderDescriptor { frame_content_size_flag: u2:2, ..zero_desc};
    let (result, frame_content_size) = parse_frame_content_size(buffer, frame_header_desc);
    assert_eq(result, BufferResult {
        status: BufferStatus::OK,
        buffer: Buffer { content: u64:0x0, length: u32:0x0 },
    });
    assert_eq(frame_content_size, u64:0x12345678);

    let buffer = Buffer { content: u64:0x1234567890ABCDEF, length: u32:0x40 };
    let frame_header_desc = FrameHeaderDescriptor { frame_content_size_flag: u2:3, ..zero_desc};
    let (result, frame_content_size) = parse_frame_content_size(buffer, frame_header_desc);
    assert_eq(result, BufferResult {
        status: BufferStatus::OK,
        buffer: Buffer { content: u64:0x0, length: u32:0x0 },
    });
    assert_eq(frame_content_size, u64:0x1234567890ABCDEF);

    let buffer = Buffer { content: u32:0x12345678, length: u32:0x20 };
    let frame_header_desc = FrameHeaderDescriptor { frame_content_size_flag: u2:0x3, ..zero_desc};
    let (result, frame_content_size) = parse_frame_content_size(buffer, frame_header_desc);
    assert_eq(result, BufferResult {
        status: BufferStatus::NO_ENOUGH_DATA,
        buffer: Buffer { content: u32:0x12345678, length: u32:0x20 },
    });
    assert_eq(frame_content_size, u64:0x0);
}

// Calculate maximal accepted window_size for given WINDOW_LOG_MAX and return whether given
// window_size should be accepted or discarded.
// Based on window_size calculation from: RFC 8878
// https://datatracker.ietf.org/doc/html/rfc8878#name-window-descriptor
fn window_size_valid<WINDOW_LOG_MAX: WindowSize>(window_size: WindowSize) -> bool {
    let max_window_size = (WindowSize:1 << WINDOW_LOG_MAX) + (((WindowSize:1 << WINDOW_LOG_MAX) >> WindowSize:3) * MAX_MANTISSA);

    window_size <= max_window_size
}

// Parses a Buffer with data and extracts Frame_Header information. The buffer
// is assumed to contain a valid Frame_Header The function returns FrameHeaderResult
// with BufferResult that contains outcome of the operations on the Buffer,
// FrameHeader with the extracted frame header if the parsing was successful,
// and the status of the operation in FrameHeaderStatus. On failure, the returned
// buffer is the same as the input buffer.
// WINDOW_LOG_MAX is the base 2 logarithm used for calculating the maximal allowed
//                window_size. Frame header parsing function must discard all frames that
//                have window_size above the maximal allowed window_size.
// CAPACITY is the buffer capacity
pub fn parse_frame_header<WINDOW_LOG_MAX: WindowSize, CAPACITY: u32>(buffer: Buffer<CAPACITY>) -> FrameHeaderResult {
    trace_fmt!("parse_frame_header: ==== Parsing ==== \n");
    trace_fmt!("parse_frame_header: initial buffer: {:#x}", buffer);

    let (result, desc) = parse_frame_header_descriptor(buffer);
    trace_fmt!("parse_frame_header: buffer after parsing header descriptor: {:#x}", result.buffer);

    let (result, header) = match result.status {
        BufferStatus::OK => {
            let (result, window_size) = if window_descriptor_exists(desc) {
                trace_fmt!("parse_frame_header: window_descriptor exists, parse it");
                parse_window_descriptor(result.buffer, desc)
            } else {
                trace_fmt!("parse_frame_header: window_descriptor does not exist, skip parsing it");
                (result, u64:0)
            };
            trace_fmt!("parse_frame_header: buffer after parsing window_descriptor: {:#x}", result.buffer);

            match result.status {
                BufferStatus::OK => {
                    trace_fmt!("parse_frame_header: parse dictionary_id");
                    let (result, dictionary_id) = parse_dictionary_id(result.buffer, desc);
                    trace_fmt!("parse_frame_header: buffer after parsing dictionary_id: {:#x}", result.buffer);

                    match result.status {
                        BufferStatus::OK => {
                            let (result, frame_content_size) = if frame_content_size_exists(desc) {
                                trace_fmt!("parse_frame_header: frame_content_size exists, parse it");
                                parse_frame_content_size(result.buffer, desc)
                            } else {
                                trace_fmt!("parse_frame_header: frame_content_size does not exist, skip parsing it");
                                (result, FRAME_CONTENT_SIZE_NOT_PROVIDED_VALUE)
                            };
                            trace_fmt!("parse_frame_header: buffer after parsing frame_content_size: {:#x}", result.buffer);

                            match result.status {
                                BufferStatus::OK => {
                                    trace_fmt!("parse_frame_header: calculate frame header!");
                                    let window_size = match window_descriptor_exists(desc) {
                                        true => window_size,
                                        _ => frame_content_size,
                                    };

                                    (
                                        result,
                                        FrameHeader {
                                            window_size: window_size,
                                            frame_content_size: frame_content_size,
                                            dictionary_id: dictionary_id,
                                            content_checksum_flag: desc.content_checksum_flag,
                                        }
                                    )
                                },
                                _ => {
                                    trace_fmt!("parse_frame_header: Not enough data to parse frame_content_size!");
                                    (result, zero!<FrameHeader>())
                                }
                             }
                         },
                         _ => {
                            trace_fmt!("parse_frame_header: Not enough data to parse dictionary_id!");
                            (result, zero!<FrameHeader>())
                         }
                     }
                 },
                 _ => {
                    trace_fmt!("parse_frame_header: Not enough data to parse window_descriptor!");
                    (result, zero!<FrameHeader>())
                 }
            }
        },
        _ => {
            trace_fmt!("parse_frame_header: Not enough data to parse frame_header_descriptor!");
            (result, zero!<FrameHeader>())
        }
    };

    let (status, buffer) = match result.status {
        BufferStatus::OK => (FrameHeaderStatus::OK, result.buffer),
        _ => (FrameHeaderStatus::NO_ENOUGH_DATA, buffer)
    };

    let frame_header_result = FrameHeaderResult { status: status, header: header, buffer: buffer };

    // libzstd always reports NO_ENOUGH_DATA errors before CORRUPTED caused by
    // reserved bit being set
    if (desc.reserved == u1:1 && frame_header_result.status != FrameHeaderStatus::NO_ENOUGH_DATA) {
        trace_fmt!("parse_frame_header: frame descriptor corrupted!");
        // Critical failure - requires resetting the whole decoder
        FrameHeaderResult {
            status: FrameHeaderStatus::CORRUPTED,
            buffer: zero!<Buffer>(),
            header: zero!<FrameHeader>(),
        }
    } else if (!window_size_valid<WINDOW_LOG_MAX>(header.window_size)) {
        trace_fmt!("parse_frame_header: frame discarded: window_size to big: {}", header.window_size);
        FrameHeaderResult {
            status: FrameHeaderStatus::UNSUPPORTED_WINDOW_SIZE,
            buffer: zero!<Buffer>(),
            header: zero!<FrameHeader>(),
        }
    } else {
        frame_header_result
    }
}

// The largest allowed WindowLog for DSLX tests
pub const TEST_WINDOW_LOG_MAX = WindowSize:22;

#[test]
fn test_parse_frame_header() {
    // normal cases
    let buffer = Buffer { content: bits[128]:0x1234567890ABCDEF_CAFE_09_C2, length: u32:96 };
    let frame_header_result = parse_frame_header<TEST_WINDOW_LOG_MAX>(buffer);
    assert_eq(frame_header_result, FrameHeaderResult {
        status: FrameHeaderStatus::OK,
        buffer: Buffer {
            content: bits[128]:0x0,
            length: u32:0,
        },
        header: FrameHeader {
            window_size: u64:0x900,
            frame_content_size: u64:0x1234567890ABCDEF,
            dictionary_id: u32:0xCAFE,
            content_checksum_flag: u1:0,
        }
    });

    // SingleSegmentFlag is set and FrameContentSize is bigger than accepted window_size
    let buffer = Buffer { content: bits[128]:0x1234567890ABCDEF_CAFE_E2, length: u32:88 };
    let frame_header_result = parse_frame_header<TEST_WINDOW_LOG_MAX>(buffer);
    assert_eq(frame_header_result, FrameHeaderResult {
        status: FrameHeaderStatus::UNSUPPORTED_WINDOW_SIZE,
        buffer: Buffer { content: bits[128]:0x0, length: u32:0 },
        header: zero!<FrameHeader>()
    });

    let buffer = Buffer { content: bits[128]:0xaa20, length: u32:16 };
    let frame_header_result = parse_frame_header<TEST_WINDOW_LOG_MAX>(buffer);
    assert_eq(frame_header_result, FrameHeaderResult {
        status: FrameHeaderStatus::OK,
        buffer: Buffer {
            content: bits[128]:0x0,
            length: u32:0,
        },
        header: FrameHeader {
            window_size: u64:0xaa,
            frame_content_size: u64:0xaa,
            dictionary_id: u32:0x0,
            content_checksum_flag: u1:0,
        },
    });

    // when buffer is too short
    let buffer = Buffer { content: bits[128]:0x0, length: u32:0 };
    let frame_header_result = parse_frame_header<TEST_WINDOW_LOG_MAX>(buffer);
    assert_eq(frame_header_result, FrameHeaderResult {
        status: FrameHeaderStatus::NO_ENOUGH_DATA,
        buffer: buffer,
        header: zero!<FrameHeader>()
    });

    let buffer = Buffer { content: bits[128]:0xC2, length: u32:8 };
    let frame_header_result = parse_frame_header<TEST_WINDOW_LOG_MAX>(buffer);
    assert_eq(frame_header_result, FrameHeaderResult {
        status: FrameHeaderStatus::NO_ENOUGH_DATA,
        buffer: buffer,
        header: zero!<FrameHeader>()
    });

    let buffer = Buffer { content: bits[128]:0x09_C2, length: u32:16 };
    let frame_header_result = parse_frame_header<TEST_WINDOW_LOG_MAX>(buffer);
    assert_eq(frame_header_result, FrameHeaderResult {
        status: FrameHeaderStatus::NO_ENOUGH_DATA,
        buffer: buffer,
        header: zero!<FrameHeader>()
    });

    let buffer = Buffer { content: bits[128]:0x1234_09_C2, length: u32:32 };
    let frame_header_result = parse_frame_header<TEST_WINDOW_LOG_MAX>(buffer);
    assert_eq(frame_header_result, FrameHeaderResult {
        status: FrameHeaderStatus::NO_ENOUGH_DATA,
        buffer: buffer,
        header: zero!<FrameHeader>()
    });

    let buffer = Buffer { content: bits[128]:0x1234_09_C2, length: u32:32 };
    let frame_header_result = parse_frame_header<TEST_WINDOW_LOG_MAX>(buffer);
    assert_eq(frame_header_result, FrameHeaderResult {
        status: FrameHeaderStatus::NO_ENOUGH_DATA,
        buffer: buffer,
        header: zero!<FrameHeader>()
    });

    // when frame header descriptor is corrupted
    let buffer = Buffer { content: bits[128]:0x1234567890ABCDEF_1234_09_CA, length: u32:96 };
    let frame_header_result = parse_frame_header<TEST_WINDOW_LOG_MAX>(buffer);
    assert_eq(frame_header_result, FrameHeaderResult {
        status: FrameHeaderStatus::CORRUPTED,
        buffer: Buffer { content: bits[128]:0x0, length: u32:0 },
        header: zero!<FrameHeader>()
    });

    // Frame Header is discarded because Window size required by frame is too big for given decoder
    // configuration
    let buffer = Buffer { content: bits[128]:0xd310, length: u32:16 };
    let frame_header_result = parse_frame_header<TEST_WINDOW_LOG_MAX>(buffer);
    assert_eq(frame_header_result, FrameHeaderResult {
        status: FrameHeaderStatus::UNSUPPORTED_WINDOW_SIZE,
        buffer: Buffer { content: bits[128]:0x0, length: u32:0 },
        header: zero!<FrameHeader>()
    });
}
