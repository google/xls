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

pub struct SequenceExecutorPacket {
    msg_type: SequenceExecutorMessageType,
    length: CopyOrMatchLength, // Literal length or match length
    content: CopyOrMatchContent, // Literal data or match offset
    last: bool, // Last packet in frame
}

// Defines output format of the ZSTD Decoder
pub struct ZstdDecodedPacket {
    data: BlockData,
    length: BlockPacketLength, // valid bits in data
    last: bool, // Last decoded packet in frame
}
