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

pub type BlockData = bits[DATA_WIDTH];
pub type BlockPacketLength = u32;
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

pub struct SequenceExecutorPacket<DATA_WIDTH: u32> {
    msg_type: SequenceExecutorMessageType,
    length: CopyOrMatchLength, // Literal length or match length
    content: uN[DATA_WIDTH * u32:8], // Literal data or match offset
    last: bool, // Last packet in frame
}

// Defines output format of the ZSTD Decoder
pub struct ZstdDecodedPacket {
    data: BlockData,
    length: BlockPacketLength, // valid bits in data
    last: bool, // Last decoded packet in frame
}

// Literals decoding

pub const RLE_LITERALS_DATA_WIDTH = u32:8;
pub const RLE_LITERALS_REPEAT_WIDTH = u32:20;
pub const LITERALS_DATA_WIDTH = u32:64;
pub const LITERALS_LENGTH_WIDTH = std::clog2(
    std::ceil_div(LITERALS_DATA_WIDTH, RLE_LITERALS_DATA_WIDTH) + u32:1
);

pub type RleLitData = uN[RLE_LITERALS_DATA_WIDTH];
pub type RleLitRepeat = uN[RLE_LITERALS_REPEAT_WIDTH];
pub type LitData = uN[LITERALS_DATA_WIDTH];
pub type LitLength = uN[LITERALS_LENGTH_WIDTH];
pub type LitID = u32;

pub type DecompressedSize = u20;

pub enum LiteralType: u3 {
    RAW        = 0,
    RLE        = 1,
    COMP       = 2,
    COMP_4     = 3,
    TREELESS   = 4,
    TREELESS_4 = 5,
}

pub struct Streams {
    count: bits[2],
    stream_lengths: bits[20][4],
}

pub struct LiteralsPathCtrl {
    data_conf: Streams,
    decompressed_size: DecompressedSize,
    literals_type: LiteralType,
}

pub struct RleLiteralsData {
    data: RleLitData,
    repeat: RleLitRepeat,
    last: bool,
    id: LitID,
}

pub struct LiteralsData {
    data: LitData,
    length: LitLength,
    last: bool,
}

pub struct LiteralsDataWithSync {
    data: LitData,
    length: LitLength,
    last: bool,
    id: LitID,
    literals_last: bool,
}

pub struct LiteralsBufferCtrl {
    length: u32,
    last: bool,
}

