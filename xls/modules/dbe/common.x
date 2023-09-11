// Copyright 2023 The XLS Authors
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

import std

pub enum Mark : u4 {
    // Default initialization value used when marker field is not needed
    NONE = 0,
    // Signals end of sequence/block
    END = 1,
    // Requests reset of the processing chain
    RESET = 2,

    _ERROR_FIRST = 8,
    // Only error marks have values >= __ERROR_FIRST
    ERROR_BAD_MARK = 8,
    ERROR_INVAL_CP = 9,
}

pub fn is_error(mark: Mark) -> bool {
    (mark as u32) >= (Mark::_ERROR_FIRST as u32)
}

pub enum TokenKind : u2 {
    UNMATCHED_SYMBOL = 0,
    MATCHED_SYMBOL = 1,
    MATCH = 2,
    MARKER = 3,
}

pub struct Token<
    SYMBOL_WIDTH: u32, MATCH_OFFSET_WIDTH: u32, MATCH_LENGTH_WIDTH: u32
>{
    kind: TokenKind,
    symbol: uN[SYMBOL_WIDTH],
    match_offset: uN[MATCH_OFFSET_WIDTH],
    match_length: uN[MATCH_LENGTH_WIDTH],
    mark: Mark
}

pub struct PlainData<DATA_WIDTH: u32> {
    is_marker: bool,
    data: uN[DATA_WIDTH],
    mark: Mark,
}

/// Parameters of a classic LZ4 algorithm
/// NOTE: CPU LZ4 implementations do not have match length limited to 16 bits,
/// but 16 bits are more than enough for most real-life use cases, since
/// matches of more than 64KB are very uncommon.
pub const LZ4_SYMBOL_WIDTH = u32:8;
pub const LZ4_MATCH_OFFSET_WIDTH = u32:16;
pub const LZ4_MATCH_LENGTH_WIDTH = u32:16;
pub const LZ4_HASH_SYMBOLS = u32:4;
/// Different hash table sizes
pub const LZ4_HASH_WIDTH_4K = u32:12;
pub const LZ4_HASH_WIDTH_8K = u32:13;
pub const LZ4_HASH_WIDTH_16K = u32:14;

pub type Lz4Token = Token<
        LZ4_SYMBOL_WIDTH,
        LZ4_MATCH_OFFSET_WIDTH,
        LZ4_MATCH_LENGTH_WIDTH
    >;

pub type Lz4Data = PlainData<LZ4_SYMBOL_WIDTH>;
