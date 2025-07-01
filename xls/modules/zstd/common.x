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
import xls.examples.ram;
import xls.modules.zstd.shift_buffer;

pub const DATA_WIDTH = u32:64;
pub const MAX_ID = u32::MAX;
pub const SYMBOL_WIDTH = u32:8;
pub const BLOCK_SIZE_WIDTH = u32:21;
pub const OFFSET_WIDTH = u32:22;
pub const HISTORY_BUFFER_SIZE_KB = u32:64;
pub const BUFFER_WIDTH = u32:128;
pub const MAX_BLOCK_SIZE_KB = u32:64;

pub const BLOCK_PACKET_WIDTH = u32:32;
pub const SYMBOLS_IN_PACKET = DATA_WIDTH/SYMBOL_WIDTH;

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

pub struct SequenceExecutorPacket<DATA_W: u32> {
    msg_type: SequenceExecutorMessageType,
    // TODO: this should be max(8, clog2(maximum match value))
    length: CopyOrMatchLength, // Literal length or match length
    content: uN[DATA_W * u32:8], // Literal data or match offset
    last: bool, // Last packet in frame
}

pub struct BlockSyncData {
    id: u32,
    last_block: bool,
}

pub struct CommandConstructorData {
    sync: BlockSyncData,
    data: SequenceExecutorPacket<SYMBOLS_IN_PACKET>,
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

pub const FSE_MAX_ACCURACY_LOG = u32:15;
pub const FSE_MAX_SYMBOLS = u32:256;

pub const FSE_ACCURACY_LOG_WIDTH = std::clog2(FSE_MAX_ACCURACY_LOG + u32:1);
pub const FSE_SYMBOL_COUNT_WIDTH = std::clog2(FSE_MAX_SYMBOLS + u32:1);
pub const FSE_REMAINING_PROBA_WIDTH = std::clog2((u32:1 << FSE_MAX_ACCURACY_LOG) + u32:1);
pub const FSE_TABLE_INDEX_WIDTH = std::clog2(u32:1 << FSE_MAX_ACCURACY_LOG);

pub const FSE_PROB_DIST_WIDTH = u32:16;
pub const FSE_MAX_PROB_DIST = u32:256;
pub const FSE_SYMBOL_WIDTH = u32:16;

// FIXME: Tests in DSLX interpreter require smaller RAMs due to the problem
// with ram consumtopn descibed in https://github.com/google/xls/issues/1042
pub const TEST_FSE_MAX_ACCURACY_LOG = u32:9;

pub type FseRemainingProba = uN[FSE_REMAINING_PROBA_WIDTH];
pub type FseAccuracyLog = uN[FSE_ACCURACY_LOG_WIDTH];
pub type FseSymbolCount = uN[FSE_SYMBOL_COUNT_WIDTH];
pub type FseTableIndex = uN[FSE_TABLE_INDEX_WIDTH];


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

pub struct FseTableRecord {
    symbol: u8,
    num_of_bits: u8,
    base: u16
}

pub struct FseRemainder { value: u1, valid: bool }
pub struct FseProbaFreqDecoderCtrl { remainder: FseRemainder, finished: bool }

pub struct FseTableCreatorCtrl {
    accuracy_log: FseAccuracyLog,
    negative_proba_count: FseSymbolCount
}

pub fn highest_set_bit<N: u32>(num: uN[N]) -> uN[N] { std::flog2<N>(num) }

// SequenceDecoder

pub const SEQDEC_DPD_RAM_DATA_WIDTH = FSE_PROB_DIST_WIDTH;
pub const SEQDEC_DPD_RAM_SIZE = FSE_MAX_PROB_DIST;
pub const SEQDEC_DPD_RAM_WORD_PARTITION_SIZE = SEQDEC_DPD_RAM_DATA_WIDTH;
pub const SEQDEC_DPD_RAM_ADDR_WIDTH = std::clog2(SEQDEC_DPD_RAM_SIZE);
pub const SEQDEC_DPD_RAM_NUM_PARTITIONS = ram::num_partitions(SEQDEC_DPD_RAM_WORD_PARTITION_SIZE, SEQDEC_DPD_RAM_DATA_WIDTH);

pub const SEQDEC_TMP_RAM_DATA_WIDTH = FSE_PROB_DIST_WIDTH;
pub const SEQDEC_TMP_RAM_SIZE = FSE_MAX_PROB_DIST;
pub const SEQDEC_TMP_RAM_WORD_PARTITION_SIZE = SEQDEC_TMP_RAM_DATA_WIDTH;
pub const SEQDEC_TMP_RAM_ADDR_WIDTH = std::clog2(SEQDEC_TMP_RAM_SIZE);
pub const SEQDEC_TMP_RAM_NUM_PARTITIONS = ram::num_partitions(SEQDEC_TMP_RAM_WORD_PARTITION_SIZE, SEQDEC_TMP_RAM_DATA_WIDTH);

pub const SEQDEC_FSE_RAM_DATA_WIDTH = u32:32;
pub const SEQDEC_FSE_RAM_SIZE = FSE_MAX_SYMBOLS;
pub const SEQDEC_FSE_RAM_WORD_PARTITION_SIZE = SEQDEC_FSE_RAM_DATA_WIDTH / u32:3;
pub const SEQDEC_FSE_RAM_ADDR_WIDTH: u32 = std::clog2(SEQDEC_FSE_RAM_SIZE);
pub const SEQDEC_FSE_RAM_NUM_PARTITIONS: u32 = ram::num_partitions(SEQDEC_FSE_RAM_WORD_PARTITION_SIZE, SEQDEC_FSE_RAM_DATA_WIDTH);

pub const SEQDEC_BLOCK_RAM_DATA_WIDTH = DATA_WIDTH;
pub const SEQDEC_BLOCK_RAM_SIZE = (MAX_BLOCK_SIZE_KB * u32:1024 * u32:8) / SEQDEC_BLOCK_RAM_DATA_WIDTH;
pub const SEQDEC_BLOCK_RAM_WORD_PARTITION_SIZE = SEQDEC_BLOCK_RAM_DATA_WIDTH;
pub const SEQDEC_BLOCK_RAM_ADDR_WIDTH = std::clog2(SEQDEC_BLOCK_RAM_SIZE);
pub const SEQDEC_BLOCK_RAM_NUM_PARTITIONS: u32 = ram::num_partitions(SEQDEC_BLOCK_RAM_WORD_PARTITION_SIZE, SEQDEC_BLOCK_RAM_DATA_WIDTH);

pub const SEQDEC_SHIFT_BUFFER_DATA_WIDTH = DATA_WIDTH;
pub const SEQDEC_SHIFT_BUFFER_LENGTH_WIDTH = std::clog2(SEQDEC_SHIFT_BUFFER_DATA_WIDTH + u32:1);

pub type SeqDecDpdRamReadReq = ram::ReadReq<SEQDEC_DPD_RAM_ADDR_WIDTH, SEQDEC_DPD_RAM_NUM_PARTITIONS>;
pub type SeqDecDpdRamReadResp = ram::ReadResp<SEQDEC_DPD_RAM_DATA_WIDTH>;
pub type SeqDecDpdRamWriteReq = ram::WriteReq<SEQDEC_DPD_RAM_ADDR_WIDTH, SEQDEC_DPD_RAM_DATA_WIDTH, SEQDEC_DPD_RAM_NUM_PARTITIONS>;
pub type SeqDecDpdRamWriteResp = ram::WriteResp;
pub type SeqDecDpdRamAddr = bits[SEQDEC_DPD_RAM_ADDR_WIDTH];
pub type SeqDecDpdRamData = bits[SEQDEC_DPD_RAM_DATA_WIDTH];

pub type SeqDecTmpRamReadReq = ram::ReadReq<SEQDEC_TMP_RAM_ADDR_WIDTH, SEQDEC_TMP_RAM_NUM_PARTITIONS>;
pub type SeqDecTmpRamReadResp = ram::ReadResp<SEQDEC_TMP_RAM_DATA_WIDTH>;
pub type SeqDecTmpRamWriteReq = ram::WriteReq<SEQDEC_TMP_RAM_ADDR_WIDTH, SEQDEC_TMP_RAM_DATA_WIDTH, SEQDEC_TMP_RAM_NUM_PARTITIONS>;
pub type SeqDecTmpRamWriteResp = ram::WriteResp;
pub type SeqDecTmpRamAddr = bits[SEQDEC_TMP_RAM_ADDR_WIDTH];
pub type SeqDecTmpRamData = bits[SEQDEC_TMP_RAM_DATA_WIDTH];

pub type SeqDecFseRamReadReq = ram::ReadReq<SEQDEC_FSE_RAM_ADDR_WIDTH, SEQDEC_FSE_RAM_NUM_PARTITIONS>;
pub type SeqDecFseRamReadResp = ram::ReadResp<SEQDEC_FSE_RAM_DATA_WIDTH>;
pub type SeqDecFseRamWriteReq = ram::WriteReq<SEQDEC_FSE_RAM_ADDR_WIDTH, SEQDEC_FSE_RAM_DATA_WIDTH, SEQDEC_FSE_RAM_NUM_PARTITIONS>;
pub type SeqDecFseRamWriteResp = ram::WriteResp;
pub type SeqDecFseRamAddr = bits[SEQDEC_FSE_RAM_ADDR_WIDTH];
pub type SeqDecFseRamData = bits[SEQDEC_FSE_RAM_DATA_WIDTH];

pub type SeqDecBlockRamReadReq = ram::ReadReq<SEQDEC_BLOCK_RAM_ADDR_WIDTH, SEQDEC_BLOCK_RAM_NUM_PARTITIONS>;
pub type SeqDecBlockRamReadResp = ram::ReadResp<SEQDEC_BLOCK_RAM_DATA_WIDTH>;
pub type SeqDecBlockRamWriteReq = ram::WriteReq<SEQDEC_BLOCK_RAM_ADDR_WIDTH, SEQDEC_BLOCK_RAM_DATA_WIDTH, SEQDEC_BLOCK_RAM_NUM_PARTITIONS>;
pub type SeqDecBlockRamWriteResp = ram::WriteResp;
pub type SeqDecBlockRamAddr = bits[SEQDEC_BLOCK_RAM_ADDR_WIDTH];
pub type SeqDecBlockRamData = bits[SEQDEC_BLOCK_RAM_DATA_WIDTH];

pub type SeqDecShiftBufferCtrl = shift_buffer::ShiftBufferCtrl<SEQDEC_SHIFT_BUFFER_LENGTH_WIDTH>;
pub type SeqDecShiftBufferInput = shift_buffer::ShiftBufferPacket<SEQDEC_SHIFT_BUFFER_DATA_WIDTH, SEQDEC_SHIFT_BUFFER_LENGTH_WIDTH>;
pub type SeqDecShiftBufferOutput = shift_buffer::ShiftBufferOutput<SEQDEC_SHIFT_BUFFER_DATA_WIDTH, SEQDEC_SHIFT_BUFFER_LENGTH_WIDTH>;
pub type SeqDecShiftBufferPacket = shift_buffer::ShiftBufferPacket<SEQDEC_SHIFT_BUFFER_DATA_WIDTH, SEQDEC_SHIFT_BUFFER_LENGTH_WIDTH>;
pub type SeqDecShiftBufferStatus = shift_buffer::ShiftBufferStatus;

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

pub struct LiteralsData {
    data: LitData,
    length: LitLength,
    last: bool,
}

pub struct LiteralsDataWithSync {
    data: LitData,
    length: LitLength,
    last: bool,          // last packet in single literals section decoding
    id: LitID,
    literals_last: bool, // last literals section in ZSTD frame
}

pub struct LiteralsBufferCtrl {
    length: u32,
    last: bool,
}

pub enum LookupDecoderStatus: u1 {
    OK = u1:0,
    ERROR = u1:1,
}

pub struct LookupDecoderReq {}

pub struct LookupDecoderResp {
    status: LookupDecoderStatus,
    accuracy_log: FseAccuracyLog,
}

pub struct DataArray<BITS_PER_WORD: u32, LENGTH: u32>{
  data: uN[BITS_PER_WORD][LENGTH],
  length: u32,
  array_length: u32
}
