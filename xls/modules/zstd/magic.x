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

// This file contains utilities related to ZSTD magic number parsing
// More information about the ZSTD Magic Number can be found in:
// https://datatracker.ietf.org/doc/html/rfc8878#section-3.1.1

import std;
import xls.modules.zstd.buffer as buff;

type Buffer = buff::Buffer;
type BufferStatus = buff::BufferStatus;

// Magic number value, as in:
// https://datatracker.ietf.org/doc/html/rfc8878#section-3.1.1
const MAGIC_NUMBER = u32:0xFD2FB528;

// Status values reported by the magic number parsing function
pub enum MagicStatus: u2 {
    OK = 0,
    CORRUPTED = 1,
    NO_ENOUGH_DATA = 2,
}

// structure for returning results of magic number parsing
pub struct MagicResult<CAPACITY: u32> {
    buffer: Buffer<CAPACITY>,
    status: MagicStatus,
}

// Parses a Buffer and checks if it contains the magic number.
// The buffer is assumed to contain a valid beginning of the ZSTD file.
// The function returns MagicResult structure with the buffer after parsing
// the magic number and the status of the operation. On failure, the returned
// buffer is the same as the input buffer.
pub fn parse_magic_number<CAPACITY: u32>(buffer: Buffer<CAPACITY>) -> MagicResult<CAPACITY> {
    let (result, data) = buff::buffer_fixed_pop_checked<u32:32>(buffer);

    match result.status {
        BufferStatus::OK => {
            if data == MAGIC_NUMBER {
                trace_fmt!("parse_magic_number: Magic number found!");
                MagicResult {status: MagicStatus::OK, buffer: result.buffer}
            } else {
                trace_fmt!("parse_magic_number: Magic number not found!");
                MagicResult {status: MagicStatus::CORRUPTED, buffer: buffer}
            }
        },
        _ => {
            trace_fmt!("parse_frame_header: Not enough data to parse magic number!");
            MagicResult {status: MagicStatus::NO_ENOUGH_DATA, buffer: buffer}
        }
    }
}

#[test]
fn test_parse_magic_number() {
    let buffer = Buffer { content: MAGIC_NUMBER, length: u32:32};
    let result = parse_magic_number(buffer);
    assert_eq(result, MagicResult {
        status: MagicStatus::OK,
        buffer: Buffer {content: u32:0, length: u32:0},
    });

    let buffer = Buffer { content: u32:0x12345678, length: u32:32};
    let result = parse_magic_number(buffer);
    assert_eq(result, MagicResult {
        status: MagicStatus::CORRUPTED,
        buffer: buffer
    });

    let buffer = Buffer { content: u32:0x1234, length: u32:16};
    let result = parse_magic_number(buffer);
    assert_eq(result, MagicResult {
        status: MagicStatus::NO_ENOUGH_DATA,
        buffer: buffer,
    });
}
