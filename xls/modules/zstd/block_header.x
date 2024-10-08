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
import xls.modules.zstd.common as common;

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
