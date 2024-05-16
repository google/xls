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

import std;

pub const DATA_WIDTH = u32:64;
pub const MAX_ID = u32::MAX;
pub const SYMBOL_WIDTH = u32:8;
pub const BLOCK_SIZE_WIDTH = u32:21;
pub const OFFSET_WIDTH = u32:22;
pub const HISTORY_BUFFER_SIZE_KB = u32:64;
pub const BUFFER_WIDTH = u32:128;

pub const BLOCK_PACKET_WIDTH = u32:32;

pub type BlockData = bits[DATA_WIDTH];
pub type BlockPacketLength = bits[BLOCK_PACKET_WIDTH];
pub type BlockSize = bits[BLOCK_SIZE_WIDTH];
pub type CopyOrMatchContent = BlockData;
pub type CopyOrMatchLength = u64;
pub type Offset = bits[OFFSET_WIDTH];

pub enum BlockType : u2 {
    RAW = 0,
    RLE = 1,
    COMPRESSED = 2,
    RESERVED = 3,
}

pub struct BlockDataPacket {
    last: bool,
    last_block: bool,
    id: u32,
    data: BlockData,
    length: BlockPacketLength,
}

pub enum SequenceExecutorMessageType : u1 {
    LITERAL = 0,
    SEQUENCE = 1,
}

pub struct ExtendedBlockDataPacket {
    msg_type: SequenceExecutorMessageType,
    packet: BlockDataPacket,
}

pub struct SequenceExecutorPacket {
    msg_type: SequenceExecutorMessageType,
    length: CopyOrMatchLength,  // Literal length or match length
    content: CopyOrMatchContent,  // Literal data or match offset
    last: bool,  // Last packet in frame
}

// Defines output format of the ZSTD Decoder
pub struct ZstdDecodedPacket {
    data: BlockData,
    length: BlockPacketLength, // valid bits in data
    last: bool, // Last decoded packet in frame
}

pub enum CompressionMode : u2 {
    PREDEFINED = 0,
    RLE = 1,
    COMPRESSED = 2,
    REPEAT = 3,
}

pub struct SequenceConf {
    sequence_count: u17,
    literals_mode: CompressionMode,
    offset_mode: CompressionMode,
    match_mode: CompressionMode,
}

pub struct SequencePathCtrl {
    literals_count: u20,
    last_block: bool,
    id: u32,
    sequence_conf: SequenceConf,
}

pub struct SequenceData { bytes: bits[64], length: u32, last: bool }

// FSE

pub const FSE_MAX_ACCURACY_LOG = u32:9;
pub const FSE_MAX_SYMBOLS = u32:256;

pub const FSE_ACCURACY_LOG_WIDTH = std::clog2(FSE_MAX_ACCURACY_LOG + u32:1);
pub const FSE_SYMBOL_COUNT_WIDTH = std::clog2(FSE_MAX_SYMBOLS + u32:1);
pub const FSE_REMAINING_PROBA_WIDTH = std::clog2((u32:1 << FSE_MAX_ACCURACY_LOG) + u32:1);

pub type FseRemainingProba = uN[FSE_REMAINING_PROBA_WIDTH];
pub type FseAccuracyLog = uN[FSE_ACCURACY_LOG_WIDTH];
pub type FseSymbolCount = uN[FSE_SYMBOL_COUNT_WIDTH];

// defined in https://datatracker.ietf.org/doc/html/rfc8878#section-3.1.1.3.2.2.1
pub const FSE_LITERAL_LENGTH_DEFAULT_DIST = s16[36]:[
    s16:4, s16:3, s16:2, s16:2, s16:2, s16:2, s16:2, s16:2, s16:2, s16:2, s16:2, s16:2, s16:2,
    s16:1, s16:1, s16:1, s16:2, s16:2, s16:2, s16:2, s16:2, s16:2, s16:2, s16:2, s16:2, s16:3,
    s16:2, s16:1, s16:1, s16:1, s16:1, s16:1, s16:-1, s16:-1, s16:-1, s16:-1,
];

// defined in https://datatracker.ietf.org/doc/html/rfc8878#section-3.1.1.3.2.2.2
pub const FSE_OFFSET_DEFAULT_DIST = s16[29]:[
    s16:1, s16:1, s16:1, s16:1, s16:1, s16:1, s16:2, s16:2, s16:2, s16:1, s16:1, s16:1, s16:1,
    s16:1, s16:1, s16:1, s16:1, s16:1, s16:1, s16:1, s16:1, s16:1, s16:1, s16:1, s16:-1, s16:-1,
    s16:-1, s16:-1, s16:-1,
];

// defined in https://datatracker.ietf.org/doc/html/rfc8878#section-3.1.1.3.2.2.3
pub const FSE_MATCH_LENGTH_DEFAULT_DIST = s16[53]:[
    s16:1, s16:4, s16:3, s16:2, s16:2, s16:2, s16:2, s16:2, s16:2, s16:1, s16:1, s16:1, s16:1,
    s16:1, s16:1, s16:1, s16:1, s16:1, s16:1, s16:1, s16:1, s16:1, s16:1, s16:1, s16:1, s16:1,
    s16:1, s16:1, s16:1, s16:1, s16:1, s16:1, s16:1, s16:1, s16:1, s16:1, s16:1, s16:1, s16:1,
    s16:1, s16:1, s16:1, s16:1, s16:1, s16:1, s16:1, s16:-1, s16:-1, s16:-1, s16:-1, s16:-1, s16:-1,
    s16:-1,
];

pub enum FSETableInitMode : u1 {
    DEFAULT = 0,
    EXTERNAL = 1,
}

pub enum FSETableType : u2 {
    LITERAL = 0,
    OFFSET = 1,
    MATCH = 2,
}

pub struct FseRemainder { value: u1, valid: bool }
pub struct FseProbaFreqDecoderCtrl { remainder: FseRemainder, finished: bool }
