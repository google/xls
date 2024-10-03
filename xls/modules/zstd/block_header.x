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

// This file contains utilities related to ZSTD Block Header parsing.
// More information about the ZSTD Block Header can be found in:
// https://datatracker.ietf.org/doc/html/rfc8878#section-3.1.1.2

import std;
import xls.modules.zstd.buffer as buff;
import xls.modules.zstd.common as common;

type Buffer = buff::Buffer;
type BufferStatus = buff::BufferStatus;
type BlockType = common::BlockType;
type BlockSize = common::BlockSize;

// Status values reported by the block header parsing function
pub enum BlockHeaderStatus: u2 {
    OK = 0,
    CORRUPTED = 1,
    NO_ENOUGH_DATA = 2,
}

// Structure for data obtained from decoding Block_Header
pub struct BlockHeader {
    last: bool,
    btype: BlockType,
    size: BlockSize,
}

// Structure for returning results of block header parsing
pub struct BlockHeaderResult<CAPACITY: u32> {
    buffer: Buffer<CAPACITY>,
    status: BlockHeaderStatus,
    header: BlockHeader,
}

// Auxiliary constant that can be used to initialize Proc's state
// with empty FrameHeader, because `zero!` cannot be used in that context
pub const ZERO_BLOCK_HEADER = zero!<BlockHeader>();

// Extracts Block_Header fields from 24-bit chunk of data
// that is assumed to be a valid Block_Header
pub fn extract_block_header(data:u24) -> BlockHeader {
    BlockHeader {
        size: data[3:24],
        btype: data[1:3] as BlockType,
        last: data[0:1],
    }
}

// Parses a Buffer and extracts information from a Block_Header. Returns BufferResult
// with outcome of operations on buffer and information extracted from the Block_Header.
pub fn parse_block_header<CAPACITY: u32>(buffer: Buffer<CAPACITY>) -> BlockHeaderResult<CAPACITY> {
    let (result, data) = buff::buffer_fixed_pop_checked<u32:24>(buffer);

    match result.status {
        BufferStatus::OK => {
            let block_header = extract_block_header(data);
            if (block_header.btype != BlockType::RESERVED) {
                BlockHeaderResult {status: BlockHeaderStatus::OK, header: block_header, buffer: result.buffer}
            } else {
                BlockHeaderResult {status: BlockHeaderStatus::CORRUPTED, header: zero!<BlockHeader>(), buffer: buffer}
            }
        },
        _ => {
            trace_fmt!("parse_block_header: Not enough data to parse block header! {}", buffer.length);
            BlockHeaderResult {status: BlockHeaderStatus::NO_ENOUGH_DATA, header: zero!<BlockHeader>(), buffer: buffer}
        }
    }
}

#[test]
fn test_parse_block_header() {
  let buffer = Buffer { content: u32:0x8001 , length: u32:24};
  let result = parse_block_header(buffer);
  assert_eq(result, BlockHeaderResult {
      status: BlockHeaderStatus::OK,
      header: BlockHeader { last: u1:1, btype: BlockType::RAW, size: BlockSize:0x1000 },
      buffer: Buffer { content: u32:0, length: u32:0 }
  });

  let buffer = Buffer { content: u32:0x91A2, length: u32:24};
  let result = parse_block_header(buffer);
  assert_eq(result, BlockHeaderResult {
      status: BlockHeaderStatus::OK,
      header: BlockHeader { last: u1:0, btype: BlockType::RLE, size: BlockSize:0x1234 },
      buffer: Buffer { content: u32:0, length: u32:0 }
  });

  let buffer = Buffer { content: u32:0x001, length: u32:16};
  let result = parse_block_header(buffer);
  assert_eq(result, BlockHeaderResult {
      status: BlockHeaderStatus::NO_ENOUGH_DATA,
      header: zero!<BlockHeader>(),
      buffer: Buffer { content: u32:0x001, length: u32:16 }
  });
}
