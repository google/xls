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

pub const DATA_WIDTH = u32:64;
pub const MAX_ID = u32::MAX;
pub const SYMBOL_WIDTH = u32:8;
pub const BLOCK_SIZE_WIDTH = u32:21;
pub const BLOCK_PACKET_WIDTH = u32:32;
pub const FSE_MAX_SYMBOLS = u32:256;
pub const FSE_MAX_ACCURACY_LOG = u32:9;

pub type BlockData = bits[DATA_WIDTH];
pub type BlockPacketLength = bits[BLOCK_PACKET_WIDTH];
pub type BlockSize = bits[BLOCK_SIZE_WIDTH];
pub type CopyOrMatchContent = BlockData;
pub type CopyOrMatchLength = u64;

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

pub enum CompressionMode : u2 {
    PredefinedMode    = 0,
    RLEMode           = 1,
    FSECompressedMode = 2,
    RepeatMode        = 3,
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

pub struct SequenceData {
    bytes: bits[64],
    length: u32,
    last: bool,
}
