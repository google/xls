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
import xls.modules.zstd.common as common;
import xls.modules.zstd.memory.mem_writer as mem_writer;
import xls.modules.zstd.ram_printer as ram_printer;
import xls.examples.ram;

type BlockData = common::BlockData;
type SequenceExecutorMessageType = common::SequenceExecutorMessageType;
type SequenceExecutorPacket = common::SequenceExecutorPacket;
type CopyOrMatchContent = common::CopyOrMatchContent;
type CopyOrMatchLength = common::CopyOrMatchLength;
type ZstdDecodedPacket = common::ZstdDecodedPacket;
type BlockPacketLength = common::BlockPacketLength;
type Offset = common::Offset;

fn calculate_ram_addr_width(hb_size_kb: u32, ram_data_width: u32, ram_num: u32) -> u32 {
    ((hb_size_kb * u32:1024 * u32:8) / ram_data_width) / ram_num
}

// Configurable RAM parameters
pub const RAM_DATA_WIDTH = common::SYMBOL_WIDTH;
const RAM_NUM = u32:8;

type RamData = bits[RAM_DATA_WIDTH];

// Constants calculated from RAM parameters
const RAM_NUM_WIDTH = std::clog2(RAM_NUM);
pub const RAM_WORD_PARTITION_SIZE = RAM_DATA_WIDTH;
const RAM_ORDER_WIDTH = std::clog2(RAM_DATA_WIDTH);
pub const RAM_NUM_PARTITIONS = ram::num_partitions(RAM_WORD_PARTITION_SIZE, RAM_DATA_WIDTH);
const RAM_REQ_MASK_ALL = std::unsigned_max_value<RAM_NUM_PARTITIONS>();
const RAM_REQ_MASK_NONE = bits[RAM_NUM_PARTITIONS]:0;

type RamNumber = bits[RAM_NUM_WIDTH];
type RamOrder = bits[RAM_ORDER_WIDTH];

pub fn ram_size(hb_size_kb: u32) -> u32 { (hb_size_kb * u32:1024 * u32:8) / RAM_DATA_WIDTH / RAM_NUM }

fn ram_addr_width(hb_size_kb: u32) -> u32 { std::clog2(ram_size(hb_size_kb)) }

// RAM related constants common for tests
const TEST_HISTORY_BUFFER_SIZE_KB = u32:1;
const TEST_DATA_W = u32:64;
const TEST_ADDR_W = u32:16;
const TEST_RAM_SIZE = ram_size(TEST_HISTORY_BUFFER_SIZE_KB);
const TEST_RAM_ADDR_WIDTH = ram_addr_width(TEST_HISTORY_BUFFER_SIZE_KB);
pub const TEST_RAM_INITIALIZED = true;
pub const TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR = ram::SimultaneousReadWriteBehavior::READ_BEFORE_WRITE;

type TestRamAddr = bits[TEST_RAM_ADDR_WIDTH];
type TestWriteReq = ram::WriteReq<TEST_RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>;
type TestWriteResp = ram::WriteResp<TEST_RAM_ADDR_WIDTH>;
type TestReadReq = ram::ReadReq<TEST_RAM_ADDR_WIDTH, RAM_NUM_PARTITIONS>;
type TestReadResp = ram::ReadResp<RAM_DATA_WIDTH>;

struct HistoryBufferPtr<RAM_ADDR_WIDTH: u32> { number: RamNumber, addr: bits[RAM_ADDR_WIDTH] }

type HistoryBufferLength = u32;

enum SequenceExecutorStatus : u2 {
    IDLE = 0,
    LITERAL_WRITE = 1,
    SEQUENCE_READ = 2,
    SEQUENCE_WRITE = 3,
}

struct SequenceExecutorState<RAM_ADDR_WIDTH: u32> {
    status: SequenceExecutorStatus,
    // Packet handling
    packet: SequenceExecutorPacket,
    packet_valid: bool,
    // History Buffer handling
    hyp_ptr: HistoryBufferPtr<RAM_ADDR_WIDTH>,
    real_ptr: HistoryBufferPtr<RAM_ADDR_WIDTH>,
    hb_len: HistoryBufferLength,
    // Repeat Offset handling
    repeat_offsets: Offset[3],
    repeat_req: bool,
    seq_cnt: bool,
}

fn decode_literal_packet(packet: SequenceExecutorPacket) -> ZstdDecodedPacket {
    ZstdDecodedPacket {
        data: packet.content, length: packet.length as BlockPacketLength, last: packet.last
    }
}

#[test]
fn test_decode_literal_packet() {
    let content = CopyOrMatchContent:0xAA00BB11CC22DD33;
    let length = CopyOrMatchLength:64;
    let last = false;

    assert_eq(
        decode_literal_packet(
            SequenceExecutorPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                length, content, last
            }),
        ZstdDecodedPacket {
            length: length as BlockPacketLength,
            data: content, last
        })
}

fn convert_output_packet<ADDR_W: u32, DATA_W: u32>(packet: ZstdDecodedPacket) -> mem_writer::MemWriterDataPacket<DATA_W, ADDR_W> {
    type MemWriterDataPacket  = mem_writer::MemWriterDataPacket<DATA_W, ADDR_W>;
    MemWriterDataPacket {
        data: packet.data as uN[DATA_W],
        length: std::div_pow2(packet.length, u32:8) as uN[ADDR_W],
        last: packet.last
    }
}

#[test]
fn test_convert_output_packet() {
    const DATA_W = u32:64;
    const ADDR_W = u32:16;

    type MemWriterDataPacket  = mem_writer::MemWriterDataPacket<DATA_W, ADDR_W>;

    let packet = ZstdDecodedPacket {
        data: CopyOrMatchContent:0xAA00BB11CC22DD33,
        length: BlockPacketLength:64,
        last: false
    };
    let expected = MemWriterDataPacket {
        data: uN[DATA_W]:0xAA00BB11CC22DD33,
        length: uN[ADDR_W]:8,
        last: false
    };

    assert_eq(convert_output_packet<ADDR_W, DATA_W>(packet), expected)
}

fn round_up_to_pow2<Y: u32, N: u32, Y_CLOG2: u32 = {std::clog2(Y)}>(x: uN[N]) -> uN[N] {
    let base = x[Y_CLOG2 as s32:];
    let reminder = x[0:Y_CLOG2 as s32] != bits[Y_CLOG2]:0;
    (base as uN[N] + reminder as uN[N]) << Y_CLOG2
}

#[test]
fn test_round_up_to_pow2() {
    assert_eq(round_up_to_pow2<u32:8>(u16:0), u16:0);
    assert_eq(round_up_to_pow2<u32:8>(u16:1), u16:8);
    assert_eq(round_up_to_pow2<u32:8>(u16:7), u16:8);
    assert_eq(round_up_to_pow2<u32:8>(u16:8), u16:8);
    assert_eq(round_up_to_pow2<u32:8>(u16:9), u16:16);
    assert_eq(round_up_to_pow2<u32:16>(u16:9), u16:16);
}

fn hb_ptr_from_offset_back
    <HISTORY_BUFFER_SIZE_KB: u32, RAM_SIZE: u32 = {ram_size(HISTORY_BUFFER_SIZE_KB)},
     RAM_ADDR_WIDTH: u32 = {ram_addr_width(HISTORY_BUFFER_SIZE_KB)}>
    (ptr: HistoryBufferPtr<RAM_ADDR_WIDTH>, offset: Offset) -> HistoryBufferPtr<RAM_ADDR_WIDTH> {

    const_assert!(common::OFFSET_WIDTH < u32:32);
    type RamAddr = bits[RAM_ADDR_WIDTH];

    let buff_change = std::mod_pow2(offset as u32, RAM_NUM) as RamNumber;
    let rounded_offset = round_up_to_pow2<RAM_NUM>(offset as u32 + u32:1);
    let max_row_span = std::div_pow2(rounded_offset, RAM_NUM) as RamAddr;
    let (number, addr_change) = if ptr.number >= buff_change {
        (ptr.number - buff_change, max_row_span - RamAddr:1)
    } else {
        ((RAM_NUM + ptr.number as u32 - buff_change as u32) as RamNumber, max_row_span)
    };
    let addr = if ptr.addr > addr_change {
        ptr.addr - addr_change
    } else {
        (RAM_SIZE + ptr.addr as u32 - addr_change as u32) as RamAddr
    };
    HistoryBufferPtr { number, addr }
}

#[test]
fn test_hb_ptr_from_offset_back() {
    assert_eq(
        hb_ptr_from_offset_back<TEST_HISTORY_BUFFER_SIZE_KB>(
            HistoryBufferPtr { number: RamNumber:4, addr: TestRamAddr:2 }, Offset:0),
        HistoryBufferPtr { number: RamNumber:4, addr: TestRamAddr:2 });
    assert_eq(
        hb_ptr_from_offset_back<TEST_HISTORY_BUFFER_SIZE_KB>(
            HistoryBufferPtr { number: RamNumber:4, addr: TestRamAddr:2 }, Offset:1),
        HistoryBufferPtr { number: RamNumber:3, addr: TestRamAddr:2 });
    assert_eq(
        hb_ptr_from_offset_back<TEST_HISTORY_BUFFER_SIZE_KB>(
            HistoryBufferPtr { number: RamNumber:4, addr: TestRamAddr:2 }, Offset:2),
        HistoryBufferPtr { number: RamNumber:2, addr: TestRamAddr:2 });
    assert_eq(
        hb_ptr_from_offset_back<TEST_HISTORY_BUFFER_SIZE_KB>(
            HistoryBufferPtr { number: RamNumber:4, addr: TestRamAddr:2 }, Offset:3),
        HistoryBufferPtr { number: RamNumber:1, addr: TestRamAddr:2 });
    assert_eq(
        hb_ptr_from_offset_back<TEST_HISTORY_BUFFER_SIZE_KB>(
            HistoryBufferPtr { number: RamNumber:4, addr: TestRamAddr:2 }, Offset:4),
        HistoryBufferPtr { number: RamNumber:0, addr: TestRamAddr:2 });
    assert_eq(
        hb_ptr_from_offset_back<TEST_HISTORY_BUFFER_SIZE_KB>(
            HistoryBufferPtr { number: RamNumber:4, addr: TestRamAddr:2 }, Offset:5),
        HistoryBufferPtr { number: RamNumber:7, addr: TestRamAddr:1 });
    assert_eq(
        hb_ptr_from_offset_back<TEST_HISTORY_BUFFER_SIZE_KB>(
            HistoryBufferPtr { number: RamNumber:4, addr: TestRamAddr:2 }, Offset:6),
        HistoryBufferPtr { number: RamNumber:6, addr: TestRamAddr:1 });
    assert_eq(
        hb_ptr_from_offset_back<TEST_HISTORY_BUFFER_SIZE_KB>(
            HistoryBufferPtr { number: RamNumber:4, addr: TestRamAddr:2 }, Offset:7),
        HistoryBufferPtr { number: RamNumber:5, addr: TestRamAddr:1 });
    assert_eq(
        hb_ptr_from_offset_back<TEST_HISTORY_BUFFER_SIZE_KB>(
            HistoryBufferPtr { number: RamNumber:4, addr: TestRamAddr:2 }, Offset:8),
        HistoryBufferPtr { number: RamNumber:4, addr: TestRamAddr:1 });
    assert_eq(
        hb_ptr_from_offset_back<TEST_HISTORY_BUFFER_SIZE_KB>(
            HistoryBufferPtr { number: RamNumber:4, addr: TestRamAddr:2 }, Offset:15),
        HistoryBufferPtr { number: RamNumber:5, addr: TestRamAddr:0 });
    assert_eq(
        hb_ptr_from_offset_back<TEST_HISTORY_BUFFER_SIZE_KB>(
            HistoryBufferPtr { number: RamNumber:0, addr: TestRamAddr:0 }, Offset:1),
        HistoryBufferPtr { number: RamNumber:7, addr: (TEST_RAM_SIZE - u32:1) as TestRamAddr });
}

fn hb_ptr_from_offset_forw
    <HISTORY_BUFFER_SIZE_KB: u32, RAM_SIZE: u32 = {ram_size(HISTORY_BUFFER_SIZE_KB)},
     RAM_ADDR_WIDTH: u32 = {ram_addr_width(HISTORY_BUFFER_SIZE_KB)}>
    (ptr: HistoryBufferPtr<RAM_ADDR_WIDTH>, offset: Offset) -> HistoryBufferPtr<RAM_ADDR_WIDTH> {

    type RamAddr = bits[RAM_ADDR_WIDTH];
    const MAX_ADDR = (RAM_SIZE - u32:1) as RamAddr;

    let buff_change = std::mod_pow2(offset as u32, RAM_NUM) as RamNumber;
    let rounded_offset = round_up_to_pow2<RAM_NUM>(offset as u32 + u32:1);
    let max_row_span = std::div_pow2(rounded_offset, RAM_NUM) as RamAddr;
    let (number, addr_change) = if ptr.number as u32 + buff_change as u32 < RAM_NUM {
        (ptr.number + buff_change, max_row_span - RamAddr:1)
    } else {
        ((buff_change as u32 - (RAM_NUM - ptr.number as u32)) as RamNumber, max_row_span)
    };

    let addr = if ptr.addr + addr_change <= MAX_ADDR {
        ptr.addr + addr_change
    } else {
        (addr_change - (MAX_ADDR - ptr.addr))
    };

    HistoryBufferPtr { number, addr }
}

#[test]
fn test_hb_ptr_from_offset_forw() {
    assert_eq(
        hb_ptr_from_offset_forw<TEST_HISTORY_BUFFER_SIZE_KB>(
            HistoryBufferPtr { number: RamNumber:4, addr: TestRamAddr:2 }, Offset:0),
        HistoryBufferPtr { number: RamNumber:4, addr: TestRamAddr:2 });
    assert_eq(
        hb_ptr_from_offset_forw<TEST_HISTORY_BUFFER_SIZE_KB>(
            HistoryBufferPtr { number: RamNumber:4, addr: TestRamAddr:2 }, Offset:1),
        HistoryBufferPtr { number: RamNumber:5, addr: TestRamAddr:2 });
    assert_eq(
        hb_ptr_from_offset_forw<TEST_HISTORY_BUFFER_SIZE_KB>(
            HistoryBufferPtr { number: RamNumber:4, addr: TestRamAddr:2 }, Offset:2),
        HistoryBufferPtr { number: RamNumber:6, addr: TestRamAddr:2 });
    assert_eq(
        hb_ptr_from_offset_forw<TEST_HISTORY_BUFFER_SIZE_KB>(
            HistoryBufferPtr { number: RamNumber:4, addr: TestRamAddr:2 }, Offset:3),
        HistoryBufferPtr { number: RamNumber:7, addr: TestRamAddr:2 });
    assert_eq(
        hb_ptr_from_offset_forw<TEST_HISTORY_BUFFER_SIZE_KB>(
            HistoryBufferPtr { number: RamNumber:4, addr: TestRamAddr:2 }, Offset:4),
        HistoryBufferPtr { number: RamNumber:0, addr: TestRamAddr:3 });
    assert_eq(
        hb_ptr_from_offset_forw<TEST_HISTORY_BUFFER_SIZE_KB>(
            HistoryBufferPtr { number: RamNumber:4, addr: TestRamAddr:2 }, Offset:5),
        HistoryBufferPtr { number: RamNumber:1, addr: TestRamAddr:3 });
    assert_eq(
        hb_ptr_from_offset_forw<TEST_HISTORY_BUFFER_SIZE_KB>(
            HistoryBufferPtr { number: RamNumber:4, addr: TestRamAddr:2 }, Offset:6),
        HistoryBufferPtr { number: RamNumber:2, addr: TestRamAddr:3 });
    assert_eq(
        hb_ptr_from_offset_forw<TEST_HISTORY_BUFFER_SIZE_KB>(
            HistoryBufferPtr { number: RamNumber:4, addr: TestRamAddr:2 }, Offset:7),
        HistoryBufferPtr { number: RamNumber:3, addr: TestRamAddr:3 });
    assert_eq(
        hb_ptr_from_offset_forw<TEST_HISTORY_BUFFER_SIZE_KB>(
            HistoryBufferPtr { number: RamNumber:4, addr: TestRamAddr:2 }, Offset:8),
        HistoryBufferPtr { number: RamNumber:4, addr: TestRamAddr:3 });
    assert_eq(
        hb_ptr_from_offset_forw<TEST_HISTORY_BUFFER_SIZE_KB>(
            HistoryBufferPtr { number: RamNumber:4, addr: TestRamAddr:2 }, Offset:15),
        HistoryBufferPtr { number: RamNumber:3, addr: TestRamAddr:4 });
    assert_eq(
        hb_ptr_from_offset_forw<TEST_HISTORY_BUFFER_SIZE_KB>(
            HistoryBufferPtr { number: RamNumber:7, addr: (TEST_RAM_SIZE - u32:1) as TestRamAddr },
            Offset:1), HistoryBufferPtr { number: RamNumber:0, addr: TestRamAddr:0 });
}

fn literal_packet_to_single_write_req
    <HISTORY_BUFFER_SIZE_KB: u32, RAM_ADDR_WIDTH: u32 = {ram_addr_width(HISTORY_BUFFER_SIZE_KB)}>
    (ptr: HistoryBufferPtr<RAM_ADDR_WIDTH>, literal: SequenceExecutorPacket, number: RamNumber)
    -> ram::WriteReq<RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS> {

    let offset = std::mod_pow2(RAM_NUM - ptr.number as u32 + number as u32, RAM_NUM) as Offset;
    let we = literal.length >= (offset as CopyOrMatchLength + CopyOrMatchLength:1) << CopyOrMatchLength:3;
    let hb = hb_ptr_from_offset_forw<HISTORY_BUFFER_SIZE_KB>(ptr, offset);

    if we {
        ram::WriteReq {
            data: literal.content[offset as u32 << u32:3+:RamData] as RamData,
            addr: hb.addr,
            mask: std::unsigned_max_value<RAM_NUM_PARTITIONS>()
        }
    } else {
        ram::WriteReq {
            addr: bits[RAM_ADDR_WIDTH]:0,
            data: bits[RAM_DATA_WIDTH]:0,
            mask: bits[RAM_NUM_PARTITIONS]:0
        }
    }
}

#[test]
fn test_literal_packet_to_single_write_req() {
    // BEFORE:                         AFTER:
    //     7  6  5  4  3  2  1  0          7  6  5  4  3  2  1  0
    // 1 |  |  |  |  |  |  |  |  |     1 |  |  |  |  |  |  |  |  |
    // 2 | o|  |  |  |  |  |  |  |     2 |11|  |  |  |  |  |  |  |
    // 3 |  |  |  |  |  |  |  |  |     3 |  | o|77|66|55|44|33|22|
    // 4 |  |  |  |  |  |  |  |  |     4 |  |  |  |  |  |  |  |  |

    let ptr = HistoryBufferPtr { number: RamNumber:7, addr: TestRamAddr:2 };
    let literals = SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::LITERAL,
        content: CopyOrMatchContent:0x77_6655_4433_2211,
        length: CopyOrMatchLength:56,
        last: false
    };
    assert_eq(
        literal_packet_to_single_write_req<TEST_HISTORY_BUFFER_SIZE_KB>(ptr, literals, RamNumber:0),
        TestWriteReq { data: RamData:0x22, addr: TestRamAddr:0x3, mask: RAM_REQ_MASK_ALL });
    assert_eq(
        literal_packet_to_single_write_req<TEST_HISTORY_BUFFER_SIZE_KB>(ptr, literals, RamNumber:3),
        TestWriteReq { data: RamData:0x55, addr: TestRamAddr:0x3, mask: RAM_REQ_MASK_ALL });
    assert_eq(
        literal_packet_to_single_write_req<TEST_HISTORY_BUFFER_SIZE_KB>(ptr, literals, RamNumber:6),
        zero!<TestWriteReq>());
}

fn literal_packet_to_write_reqs
    <HISTORY_BUFFER_SIZE_KB: u32, RAM_ADDR_WIDTH: u32 = {ram_addr_width(HISTORY_BUFFER_SIZE_KB)}>
    (ptr: HistoryBufferPtr<RAM_ADDR_WIDTH>, literal: SequenceExecutorPacket)
    -> (ram::WriteReq<RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>[RAM_NUM], HistoryBufferPtr) {
    type WriteReq = ram::WriteReq<RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>;
    let result = WriteReq[RAM_NUM]:[
        literal_packet_to_single_write_req<HISTORY_BUFFER_SIZE_KB>(ptr, literal, RamNumber:0),
        literal_packet_to_single_write_req<HISTORY_BUFFER_SIZE_KB>(ptr, literal, RamNumber:1),
        literal_packet_to_single_write_req<HISTORY_BUFFER_SIZE_KB>(ptr, literal, RamNumber:2),
        literal_packet_to_single_write_req<HISTORY_BUFFER_SIZE_KB>(ptr, literal, RamNumber:3),
        literal_packet_to_single_write_req<HISTORY_BUFFER_SIZE_KB>(ptr, literal, RamNumber:4),
        literal_packet_to_single_write_req<HISTORY_BUFFER_SIZE_KB>(ptr, literal, RamNumber:5),
        literal_packet_to_single_write_req<HISTORY_BUFFER_SIZE_KB>(ptr, literal, RamNumber:6),
        literal_packet_to_single_write_req<HISTORY_BUFFER_SIZE_KB>(ptr, literal, RamNumber:7),
    ];

    let ptr_offset = literal.length >> CopyOrMatchLength:3;
    (result, hb_ptr_from_offset_forw<HISTORY_BUFFER_SIZE_KB>(ptr, ptr_offset as Offset))
}

#[test]
fn test_literal_packet_to_write_reqs() {
    // BEFORE:                         AFTER:
    //     7  6  5  4  3  2  1  0          7  6  5  4  3  2  1  0
    // 1 |  |  |  |  |  |  |  |  |     1 |  |  |  |  |  |  |  |  |
    // 2 | o|  |  |  |  |  |  |  |     2 |11|  |  |  |  |  |  |  |
    // 3 |  |  |  |  |  |  |  |  |     3 |  |  |  |  |  |  |  | o|
    // 4 |  |  |  |  |  |  |  |  |     4 |  |  |  |  |  |  |  |  |

    let ptr = HistoryBufferPtr { number: RamNumber:7, addr: TestRamAddr:0x2 };
    let literals = SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::LITERAL,
        content: CopyOrMatchContent:0x11,
        length: CopyOrMatchLength:8,
        last: false
    };
    assert_eq(
        literal_packet_to_write_reqs<TEST_HISTORY_BUFFER_SIZE_KB>(ptr, literals),
        (
            TestWriteReq[RAM_NUM]:[
                zero!<TestWriteReq>(), zero!<TestWriteReq>(), zero!<TestWriteReq>(),
                zero!<TestWriteReq>(), zero!<TestWriteReq>(), zero!<TestWriteReq>(),
                zero!<TestWriteReq>(),
                TestWriteReq { data: RamData:0x11, addr: TestRamAddr:0x2, mask: RAM_REQ_MASK_ALL },
            ], HistoryBufferPtr { number: RamNumber:0, addr: TestRamAddr:0x3 },
        ));

    // BEFORE:                         AFTER:
    //     7  6  5  4  3  2  1  0          7  6  5  4  3  2  1  0
    // 1 |  |  |  |  |  |  |  |  |     1 |  |  |  |  |  |  |  |  |
    // 2 | o|  |  |  |  |  |  |  |     2 |11|  |  |  |  |  |  |  |
    // 3 |  |  |  |  |  |  |  |  |     3 | o|88|77|66|55|44|33|22|
    // 4 |  |  |  |  |  |  |  |  |     4 |  |  |  |  |  |  |  |  |

    let ptr = HistoryBufferPtr { number: RamNumber:7, addr: TestRamAddr:2 };
    let literals = SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::LITERAL,
        content: CopyOrMatchContent:0x8877_6655_4433_2211,
        length: CopyOrMatchLength:64,
        last: false
    };
    assert_eq(
        literal_packet_to_write_reqs<TEST_HISTORY_BUFFER_SIZE_KB>(ptr, literals),
        (
            TestWriteReq[RAM_NUM]:[
                TestWriteReq { data: RamData:0x22, addr: TestRamAddr:0x3, mask: RAM_REQ_MASK_ALL },
                TestWriteReq { data: RamData:0x33, addr: TestRamAddr:0x3, mask: RAM_REQ_MASK_ALL },
                TestWriteReq { data: RamData:0x44, addr: TestRamAddr:0x3, mask: RAM_REQ_MASK_ALL },
                TestWriteReq { data: RamData:0x55, addr: TestRamAddr:0x3, mask: RAM_REQ_MASK_ALL },
                TestWriteReq { data: RamData:0x66, addr: TestRamAddr:0x3, mask: RAM_REQ_MASK_ALL },
                TestWriteReq { data: RamData:0x77, addr: TestRamAddr:0x3, mask: RAM_REQ_MASK_ALL },
                TestWriteReq { data: RamData:0x88, addr: TestRamAddr:0x3, mask: RAM_REQ_MASK_ALL },
                TestWriteReq { data: RamData:0x11, addr: TestRamAddr:0x2, mask: RAM_REQ_MASK_ALL },
            ], HistoryBufferPtr { number: RamNumber:7, addr: TestRamAddr:3 },
        ));
}

fn max_hb_ptr_for_sequence_packet
    <HISTORY_BUFFER_SIZE_KB: u32, RAM_ADDR_WIDTH: u32 = {ram_addr_width(HISTORY_BUFFER_SIZE_KB)}>
    (ptr: HistoryBufferPtr<RAM_ADDR_WIDTH>, seq: SequenceExecutorPacket)
    -> HistoryBufferPtr<RAM_ADDR_WIDTH> {
    hb_ptr_from_offset_back<HISTORY_BUFFER_SIZE_KB>(ptr, seq.content as Offset)
}

fn sequence_packet_to_single_read_req
    <HISTORY_BUFFER_SIZE_KB: u32, RAM_ADDR_WIDTH: u32 = {ram_addr_width(HISTORY_BUFFER_SIZE_KB)}>
    (ptr: HistoryBufferPtr<RAM_ADDR_WIDTH>, max_ptr: HistoryBufferPtr<RAM_ADDR_WIDTH>,
     seq: SequenceExecutorPacket, number: RamNumber)
    -> (ram::ReadReq<RAM_ADDR_WIDTH, RAM_NUM_PARTITIONS>, RamOrder) {
    type ReadReq = ram::ReadReq<RAM_ADDR_WIDTH, RAM_NUM_PARTITIONS>;
    let offset_change = if max_ptr.number > number {
        RAM_NUM - max_ptr.number as u32 + number as u32
    } else {
        number as u32 - max_ptr.number as u32
    };
    let offset = (seq.content as u32 - offset_change) as Offset;
    let re = (offset_change as CopyOrMatchLength) < seq.length;
    let hb = hb_ptr_from_offset_back<HISTORY_BUFFER_SIZE_KB>(ptr, offset);

    if re {
        (ReadReq { addr: hb.addr, mask: RAM_REQ_MASK_ALL }, offset_change as RamOrder)
    } else {
        (zero!<ReadReq>(), RamOrder:0)
    }
}

#[test]
fn test_sequence_packet_to_single_read_req() {
    // BEFORE:                         AFTER:
    //     7  6  5  4  3  2  1  0          7  6  5  4  3  2  1  0
    // 1 | x| x|  |  |  |  |  |  |     1 |  |  |  |  |  |  |  |  |
    // 2 |  |  |  |  |  |  | x| x|     2 |  |  |  |  |  |  |  |  |
    // 3 |  |  |  |  |  |  | o|  |     3 |  |  | o| y| y| y| y|  |
    // 4 |  |  |  |  |  |  |  |  |     4 |  |  |  |  |  |  |  |  |

    let ptr = HistoryBufferPtr { number: RamNumber:1, addr: TestRamAddr:0x3 };
    let sequence = SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::SEQUENCE,
        content: CopyOrMatchContent:11,
        length: CopyOrMatchLength:4,
        last: false
    };
    let max_ptr = max_hb_ptr_for_sequence_packet<TEST_HISTORY_BUFFER_SIZE_KB>(ptr, sequence);

    assert_eq(
        sequence_packet_to_single_read_req<TEST_HISTORY_BUFFER_SIZE_KB>(
            ptr, max_ptr, sequence, RamNumber:0),
        (TestReadReq { addr: TestRamAddr:0x2, mask: RAM_REQ_MASK_ALL }, RamOrder:2));

    assert_eq(
        sequence_packet_to_single_read_req<TEST_HISTORY_BUFFER_SIZE_KB>(
            ptr, max_ptr, sequence, RamNumber:1),
        (TestReadReq { addr: TestRamAddr:0x2, mask: RAM_REQ_MASK_ALL }, RamOrder:3));

    assert_eq(
        sequence_packet_to_single_read_req<TEST_HISTORY_BUFFER_SIZE_KB>(
            ptr, max_ptr, sequence, RamNumber:2), (zero!<TestReadReq>(), RamOrder:0));

    assert_eq(
        sequence_packet_to_single_read_req<TEST_HISTORY_BUFFER_SIZE_KB>(
            ptr, max_ptr, sequence, RamNumber:7),
        (TestReadReq { addr: TestRamAddr:0x1, mask: RAM_REQ_MASK_ALL }, RamOrder:1));

    assert_eq(
        sequence_packet_to_single_read_req<TEST_HISTORY_BUFFER_SIZE_KB>(
            ptr, max_ptr, sequence, RamNumber:6),
        (TestReadReq { addr: TestRamAddr:0x1, mask: RAM_REQ_MASK_ALL }, RamOrder:0));
}

fn sequence_packet_to_read_reqs
    <HISTORY_BUFFER_SIZE_KB: u32, RAM_ADDR_WIDTH: u32 = {ram_addr_width(HISTORY_BUFFER_SIZE_KB)}>
    (ptr: HistoryBufferPtr<RAM_ADDR_WIDTH>, seq: SequenceExecutorPacket, hb_len: HistoryBufferLength)
    -> (ram::ReadReq<RAM_ADDR_WIDTH, RAM_NUM_PARTITIONS>[RAM_NUM], RamOrder[RAM_NUM], SequenceExecutorPacket, bool) {
    type ReadReq = ram::ReadReq<RAM_ADDR_WIDTH, RAM_NUM_PARTITIONS>;

    let max_len = std::min(seq.length as u32, std::min(RAM_NUM, hb_len));

    let (next_seq, next_seq_valid) = if seq.length > max_len as CopyOrMatchLength {
        (
            SequenceExecutorPacket {
                msg_type: SequenceExecutorMessageType::SEQUENCE,
                length: seq.length - max_len as CopyOrMatchLength,
                content: seq.content,
                last: seq.last
            }, true,
        )
    } else {
        (zero!<SequenceExecutorPacket>(), false)
    };

    let max_ptr = max_hb_ptr_for_sequence_packet<HISTORY_BUFFER_SIZE_KB>(ptr, seq);
    let (req0, order0) =
        sequence_packet_to_single_read_req<HISTORY_BUFFER_SIZE_KB>(ptr, max_ptr, seq, RamNumber:0);
    let (req1, order1) =
        sequence_packet_to_single_read_req<HISTORY_BUFFER_SIZE_KB>(ptr, max_ptr, seq, RamNumber:1);
    let (req2, order2) =
        sequence_packet_to_single_read_req<HISTORY_BUFFER_SIZE_KB>(ptr, max_ptr, seq, RamNumber:2);
    let (req3, order3) =
        sequence_packet_to_single_read_req<HISTORY_BUFFER_SIZE_KB>(ptr, max_ptr, seq, RamNumber:3);
    let (req4, order4) =
        sequence_packet_to_single_read_req<HISTORY_BUFFER_SIZE_KB>(ptr, max_ptr, seq, RamNumber:4);
    let (req5, order5) =
        sequence_packet_to_single_read_req<HISTORY_BUFFER_SIZE_KB>(ptr, max_ptr, seq, RamNumber:5);
    let (req6, order6) =
        sequence_packet_to_single_read_req<HISTORY_BUFFER_SIZE_KB>(ptr, max_ptr, seq, RamNumber:6);
    let (req7, order7) =
        sequence_packet_to_single_read_req<HISTORY_BUFFER_SIZE_KB>(ptr, max_ptr, seq, RamNumber:7);

    let reqs = ReadReq[RAM_NUM]:[req0, req1, req2, req3, req4, req5, req6, req7];
    let orders = RamOrder[RAM_NUM]:[order0, order1, order2, order3, order4, order5, order6, order7];
    (reqs, orders, next_seq, next_seq_valid)
}

#[test]
fn test_sequence_packet_to_read_reqs() {
    // BEFORE:                         AFTER:
    //     7  6  5  4  3  2  1  0          7  6  5  4  3  2  1  0
    // 1 | x| x|  |  |  |  |  |  |     1 |  |  |  |  |  |  |  |  |
    // 2 |  |  |  |  |  |  | x| x|     2 |  |  |  |  |  |  |  |  |
    // 3 |  |  |  |  |  |  | o|  |     3 |  |  |  |  |  |  | o|  |
    // 4 |  |  |  |  |  |  |  |  |     4 |  |  |  |  |  |  |  |  |

    let ptr = HistoryBufferPtr { number: RamNumber:1, addr: TestRamAddr:0x3 };
    let sequence = SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::SEQUENCE,
        content: CopyOrMatchContent:11,
        length: CopyOrMatchLength:4,
        last: false
    };
    let result = sequence_packet_to_read_reqs<TEST_HISTORY_BUFFER_SIZE_KB>(
        ptr, sequence, HistoryBufferLength:20);
    let expected = (
        TestReadReq[RAM_NUM]:[
            TestReadReq { addr: TestRamAddr:0x2, mask: RAM_REQ_MASK_ALL },
            TestReadReq { addr: TestRamAddr:0x2, mask: RAM_REQ_MASK_ALL }, zero!<TestReadReq>(),
            zero!<TestReadReq>(), zero!<TestReadReq>(), zero!<TestReadReq>(),
            TestReadReq { addr: TestRamAddr:0x1, mask: RAM_REQ_MASK_ALL },
            TestReadReq { addr: TestRamAddr:0x1, mask: RAM_REQ_MASK_ALL },
        ],
        RamOrder[RAM_NUM]:[
            RamOrder:2, RamOrder:3, zero!<RamOrder>(), zero!<RamOrder>(), zero!<RamOrder>(),
            zero!<RamOrder>(), RamOrder:0, RamOrder:1,
        ], zero!<SequenceExecutorPacket>(), false,
    );
    assert_eq(result, expected);

    // BEFORE:                         AFTER:
    //     7  6  5  4  3  2  1  0          7  6  5  4  3  2  1  0
    // 1 |  |  |  |  |  |  |  |  |     1 |  |  |  |  |  |  |  |  |
    // 2 | x| x|  |  |  |  |  |  |     2 |  |  |  |  |  |  |  |  |
    // 3 |  |  | x| x| x| x| x| x|     3 |  | x|  |  |  |  |  |  |
    // 4 |  |  |  |  |  |  |  | o|     4 |  |  |  |  |  |  |  | o|

    let ptr = HistoryBufferPtr { number: RamNumber:0, addr: TestRamAddr:0x4 };
    let sequence = SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::SEQUENCE,
        content: CopyOrMatchContent:10,
        length: CopyOrMatchLength:9,
        last: false
    };
    let result = sequence_packet_to_read_reqs<TEST_HISTORY_BUFFER_SIZE_KB>(
        ptr, sequence, HistoryBufferLength:20);
    let expected = (
        TestReadReq[RAM_NUM]:[
            TestReadReq { addr: TestRamAddr:0x3, mask: RAM_REQ_MASK_ALL },
            TestReadReq { addr: TestRamAddr:0x3, mask: RAM_REQ_MASK_ALL },
            TestReadReq { addr: TestRamAddr:0x3, mask: RAM_REQ_MASK_ALL },
            TestReadReq { addr: TestRamAddr:0x3, mask: RAM_REQ_MASK_ALL },
            TestReadReq { addr: TestRamAddr:0x3, mask: RAM_REQ_MASK_ALL },
            TestReadReq { addr: TestRamAddr:0x3, mask: RAM_REQ_MASK_ALL },
            TestReadReq { addr: TestRamAddr:0x2, mask: RAM_REQ_MASK_ALL },
            TestReadReq { addr: TestRamAddr:0x2, mask: RAM_REQ_MASK_ALL },
        ],
        RamOrder[RAM_NUM]:[
            RamOrder:2, RamOrder:3, RamOrder:4, RamOrder:5, RamOrder:6, RamOrder:7, RamOrder:0,
            RamOrder:1,
        ],
        SequenceExecutorPacket {
            msg_type: SequenceExecutorMessageType::SEQUENCE,
            content: CopyOrMatchContent:10,
            length: CopyOrMatchLength:1,
            last: false
        }, true,
    );
    assert_eq(result, expected);
}

struct RamWrRespHandlerData<RAM_ADDR_WIDTH: u32> {
    resp: bool[RAM_NUM],
    ptr: HistoryBufferPtr<RAM_ADDR_WIDTH>,
}

fn create_ram_wr_data<RAM_ADDR_WIDTH: u32, RAM_DATA_WIDTH: u32, RAM_NUM_PARTITIONS: u32>
    (reqs: ram::WriteReq<RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>[RAM_NUM], ptr: HistoryBufferPtr) -> (bool, RamWrRespHandlerData) {
    let do_write = for (i, do_write): (u32, bool) in u32:0..RAM_NUM {
        do_write || reqs[i].mask
    }(false);

    let resp = bool[RAM_NUM]:[
            ((reqs[0]).mask != RAM_REQ_MASK_NONE),
            ((reqs[1]).mask != RAM_REQ_MASK_NONE),
            ((reqs[2]).mask != RAM_REQ_MASK_NONE),
            ((reqs[3]).mask != RAM_REQ_MASK_NONE),
            ((reqs[4]).mask != RAM_REQ_MASK_NONE),
            ((reqs[5]).mask != RAM_REQ_MASK_NONE),
            ((reqs[6]).mask != RAM_REQ_MASK_NONE),
            ((reqs[7]).mask != RAM_REQ_MASK_NONE),
    ];

    (do_write, RamWrRespHandlerData { resp, ptr })
}

proc RamWrRespHandler<RAM_ADDR_WIDTH: u32> {
    input_r: chan<RamWrRespHandlerData> in;
    output_s: chan<HistoryBufferPtr> out;
    wr_resp_m0_r: chan<ram::WriteResp> in;
    wr_resp_m1_r: chan<ram::WriteResp> in;
    wr_resp_m2_r: chan<ram::WriteResp> in;
    wr_resp_m3_r: chan<ram::WriteResp> in;
    wr_resp_m4_r: chan<ram::WriteResp> in;
    wr_resp_m5_r: chan<ram::WriteResp> in;
    wr_resp_m6_r: chan<ram::WriteResp> in;
    wr_resp_m7_r: chan<ram::WriteResp> in;

    config(input_r: chan<RamWrRespHandlerData<RAM_ADDR_WIDTH>> in,
           output_s: chan<HistoryBufferPtr<RAM_ADDR_WIDTH>> out,
           wr_resp_m0_r: chan<ram::WriteResp> in, wr_resp_m1_r: chan<ram::WriteResp> in,
           wr_resp_m2_r: chan<ram::WriteResp> in, wr_resp_m3_r: chan<ram::WriteResp> in,
           wr_resp_m4_r: chan<ram::WriteResp> in, wr_resp_m5_r: chan<ram::WriteResp> in,
           wr_resp_m6_r: chan<ram::WriteResp> in, wr_resp_m7_r: chan<ram::WriteResp> in) {
        (
            input_r, output_s, wr_resp_m0_r, wr_resp_m1_r, wr_resp_m2_r, wr_resp_m3_r, wr_resp_m4_r,
            wr_resp_m5_r, wr_resp_m6_r, wr_resp_m7_r,
        )
    }

    init {  }

    next(state: ()) {
        let tok0 = join();
        let (tok1, input) = recv(tok0, input_r);

        let (tok2_0, _) = recv_if(tok1, wr_resp_m0_r, input.resp[0], zero!<ram::WriteResp>());
        let (tok2_1, _) = recv_if(tok1, wr_resp_m1_r, input.resp[1], zero!<ram::WriteResp>());
        let (tok2_2, _) = recv_if(tok1, wr_resp_m2_r, input.resp[2], zero!<ram::WriteResp>());
        let (tok2_3, _) = recv_if(tok1, wr_resp_m3_r, input.resp[3], zero!<ram::WriteResp>());
        let (tok2_4, _) = recv_if(tok1, wr_resp_m4_r, input.resp[4], zero!<ram::WriteResp>());
        let (tok2_5, _) = recv_if(tok1, wr_resp_m5_r, input.resp[5], zero!<ram::WriteResp>());
        let (tok2_6, _) = recv_if(tok1, wr_resp_m6_r, input.resp[6], zero!<ram::WriteResp>());
        let (tok2_7, _) = recv_if(tok1, wr_resp_m7_r, input.resp[7], zero!<ram::WriteResp>());
        let tok2 = join(tok2_0, tok2_1, tok2_2, tok2_3, tok2_4, tok2_5, tok2_6, tok2_7);

        let tok3 = send(tok2, output_s, input.ptr);
    }
}

struct RamRdRespHandlerData {
    resp: bool[RAM_NUM],
    order: RamOrder[RAM_NUM],
    last: bool
}

fn create_ram_rd_data<RAM_ADDR_WIDTH: u32, RAM_DATA_WIDTH: u32, RAM_NUM_PARTITIONS: u32>
    (reqs: ram::ReadReq<RAM_ADDR_WIDTH, RAM_NUM_PARTITIONS>[RAM_NUM], order: RamOrder[RAM_NUM], last: bool, next_packet_valid: bool) -> (bool, RamRdRespHandlerData) {
    let do_read = for (i, do_read): (u32, bool) in u32:0..RAM_NUM {
        do_read || reqs[i].mask
    }(false);

    let resp = bool[RAM_NUM]:[
        ((reqs[0]).mask != RAM_REQ_MASK_NONE),
        ((reqs[1]).mask != RAM_REQ_MASK_NONE),
        ((reqs[2]).mask != RAM_REQ_MASK_NONE),
        ((reqs[3]).mask != RAM_REQ_MASK_NONE),
        ((reqs[4]).mask != RAM_REQ_MASK_NONE),
        ((reqs[5]).mask != RAM_REQ_MASK_NONE),
        ((reqs[6]).mask != RAM_REQ_MASK_NONE),
        ((reqs[7]).mask != RAM_REQ_MASK_NONE),
    ];

    let last = if next_packet_valid { false } else { last };
    (do_read, RamRdRespHandlerData { resp, order, last })
}

proc RamRdRespHandler {
    input_r: chan<RamRdRespHandlerData> in;
    output_s: chan<SequenceExecutorPacket> out;
    rd_resp_m0_r: chan<ram::ReadResp<RAM_DATA_WIDTH>> in;
    rd_resp_m1_r: chan<ram::ReadResp<RAM_DATA_WIDTH>> in;
    rd_resp_m2_r: chan<ram::ReadResp<RAM_DATA_WIDTH>> in;
    rd_resp_m3_r: chan<ram::ReadResp<RAM_DATA_WIDTH>> in;
    rd_resp_m4_r: chan<ram::ReadResp<RAM_DATA_WIDTH>> in;
    rd_resp_m5_r: chan<ram::ReadResp<RAM_DATA_WIDTH>> in;
    rd_resp_m6_r: chan<ram::ReadResp<RAM_DATA_WIDTH>> in;
    rd_resp_m7_r: chan<ram::ReadResp<RAM_DATA_WIDTH>> in;

    config(input_r: chan<RamRdRespHandlerData> in, output_s: chan<SequenceExecutorPacket> out,
           rd_resp_m0_r: chan<ram::ReadResp<RAM_DATA_WIDTH>> in,
           rd_resp_m1_r: chan<ram::ReadResp<RAM_DATA_WIDTH>> in,
           rd_resp_m2_r: chan<ram::ReadResp<RAM_DATA_WIDTH>> in,
           rd_resp_m3_r: chan<ram::ReadResp<RAM_DATA_WIDTH>> in,
           rd_resp_m4_r: chan<ram::ReadResp<RAM_DATA_WIDTH>> in,
           rd_resp_m5_r: chan<ram::ReadResp<RAM_DATA_WIDTH>> in,
           rd_resp_m6_r: chan<ram::ReadResp<RAM_DATA_WIDTH>> in,
           rd_resp_m7_r: chan<ram::ReadResp<RAM_DATA_WIDTH>> in) {
        (
            input_r, output_s, rd_resp_m0_r, rd_resp_m1_r, rd_resp_m2_r, rd_resp_m3_r, rd_resp_m4_r,
            rd_resp_m5_r, rd_resp_m6_r, rd_resp_m7_r,
        )
    }

    init {  }

    next(state: ()) {
        let tok0 = join();
        type ReadResp = ram::ReadResp<RAM_DATA_WIDTH>;

        let (tok1, input) = recv(tok0, input_r);

        let (tok2_0, resp_0) = recv_if(tok1, rd_resp_m0_r, input.resp[0], zero!<ReadResp>());
        let (tok2_1, resp_1) = recv_if(tok1, rd_resp_m1_r, input.resp[1], zero!<ReadResp>());
        let (tok2_2, resp_2) = recv_if(tok1, rd_resp_m2_r, input.resp[2], zero!<ReadResp>());
        let (tok2_3, resp_3) = recv_if(tok1, rd_resp_m3_r, input.resp[3], zero!<ReadResp>());
        let (tok2_4, resp_4) = recv_if(tok1, rd_resp_m4_r, input.resp[4], zero!<ReadResp>());
        let (tok2_5, resp_5) = recv_if(tok1, rd_resp_m5_r, input.resp[5], zero!<ReadResp>());
        let (tok2_6, resp_6) = recv_if(tok1, rd_resp_m6_r, input.resp[6], zero!<ReadResp>());
        let (tok2_7, resp_7) = recv_if(tok1, rd_resp_m7_r, input.resp[7], zero!<ReadResp>());
        let tok2 = join(tok2_0, tok2_1, tok2_2, tok2_3, tok2_4, tok2_5, tok2_6, tok2_7);

        let content = (resp_0.data as CopyOrMatchContent) << (input.order[0] as CopyOrMatchContent << 3) |
                      (resp_1.data as CopyOrMatchContent) << (input.order[1] as CopyOrMatchContent << 3) |
                      (resp_2.data as CopyOrMatchContent) << (input.order[2] as CopyOrMatchContent << 3) |
                      (resp_3.data as CopyOrMatchContent) << (input.order[3] as CopyOrMatchContent << 3) |
                      (resp_4.data as CopyOrMatchContent) << (input.order[4] as CopyOrMatchContent << 3) |
                      (resp_5.data as CopyOrMatchContent) << (input.order[5] as CopyOrMatchContent << 3) |
                      (resp_6.data as CopyOrMatchContent) << (input.order[6] as CopyOrMatchContent << 3) |
                      (resp_7.data as CopyOrMatchContent) << (input.order[7] as CopyOrMatchContent << 3);

        let converted = std::convert_to_bits_msb0(input.resp);
        let length = std::popcount(converted) << 3;

        let output_data = SequenceExecutorPacket {
            msg_type: SequenceExecutorMessageType::LITERAL,
            length: length as CopyOrMatchLength,
            content: content as CopyOrMatchContent,
            last: input.last,
        };

        let tok3 = send(tok2, output_s, output_data);
    }
}

fn handle_reapeated_offset_for_sequences
    (seq: SequenceExecutorPacket, repeat_offsets: Offset[3], repeat_req: bool)
    -> (SequenceExecutorPacket, Offset[3]) {
    let modified_repeat_offsets = if repeat_req {
        Offset[3]:[repeat_offsets[1], repeat_offsets[2], repeat_offsets[0] - Offset:1]
    } else {
        repeat_offsets
    };

    let (seq, final_repeat_offsets) = if seq.content == CopyOrMatchContent:0 {
        fail!(
            "match_offset_zero_not_allowed",
            (zero!<SequenceExecutorPacket>(), Offset[3]:[Offset:0, ...]))
    } else if seq.content == CopyOrMatchContent:1 {
        let offset = modified_repeat_offsets[0];
        (
            SequenceExecutorPacket { content: offset as CopyOrMatchContent, ..seq },
            Offset[3]:[
                offset, repeat_offsets[1], repeat_offsets[2],
            ],
        )
    } else if seq.content == CopyOrMatchContent:2 {
        let offset = modified_repeat_offsets[1];
        (
            SequenceExecutorPacket { content: offset as CopyOrMatchContent, ..seq },
            Offset[3]:[
                offset, repeat_offsets[0], repeat_offsets[2],
            ],
        )
    } else if seq.content == CopyOrMatchContent:3 {
        let offset = modified_repeat_offsets[2];
        (
            SequenceExecutorPacket { content: offset as CopyOrMatchContent, ..seq },
            Offset[3]:[
                offset, repeat_offsets[0], repeat_offsets[1],
            ],
        )
    } else {
        let offset = seq.content as Offset - Offset:3;
        (
            SequenceExecutorPacket { content: offset as CopyOrMatchContent, ..seq },
            Offset[3]:[
                offset, repeat_offsets[0], repeat_offsets[1],
            ],
        )
    };
    (seq, final_repeat_offsets)
}

pub proc SequenceExecutor<HISTORY_BUFFER_SIZE_KB: u32,
     AXI_DATA_W: u32, AXI_ADDR_W: u32,
     RAM_SIZE: u32 = {ram_size(HISTORY_BUFFER_SIZE_KB)},
     RAM_ADDR_WIDTH: u32 = {ram_addr_width(HISTORY_BUFFER_SIZE_KB)},
     INIT_HB_PTR_ADDR: u32 = {u32:0}, INIT_HB_PTR_RAM: u32 = {u32:0},
     INIT_HB_LENGTH: HistoryBufferLength = {HistoryBufferLength:0},
     RAM_SIZE_TOTAL: u32 = {RAM_SIZE * RAM_NUM}>
{
    type MemWriterDataPacket  = mem_writer::MemWriterDataPacket<AXI_DATA_W, AXI_ADDR_W>;

    input_r: chan<SequenceExecutorPacket> in;
    output_s: chan<ZstdDecodedPacket> out;
    output_mem_wr_data_in_s: chan<MemWriterDataPacket> out;
    ram_comp_input_s: chan<RamWrRespHandlerData<RAM_ADDR_WIDTH>> out;
    ram_comp_output_r: chan<HistoryBufferPtr<RAM_ADDR_WIDTH>> in;
    ram_resp_input_s: chan<RamRdRespHandlerData> out;
    ram_resp_output_r: chan<SequenceExecutorPacket> in;
    rd_req_m0_s: chan<ram::ReadReq<RAM_ADDR_WIDTH, RAM_NUM_PARTITIONS>> out;
    rd_req_m1_s: chan<ram::ReadReq<RAM_ADDR_WIDTH, RAM_NUM_PARTITIONS>> out;
    rd_req_m2_s: chan<ram::ReadReq<RAM_ADDR_WIDTH, RAM_NUM_PARTITIONS>> out;
    rd_req_m3_s: chan<ram::ReadReq<RAM_ADDR_WIDTH, RAM_NUM_PARTITIONS>> out;
    rd_req_m4_s: chan<ram::ReadReq<RAM_ADDR_WIDTH, RAM_NUM_PARTITIONS>> out;
    rd_req_m5_s: chan<ram::ReadReq<RAM_ADDR_WIDTH, RAM_NUM_PARTITIONS>> out;
    rd_req_m6_s: chan<ram::ReadReq<RAM_ADDR_WIDTH, RAM_NUM_PARTITIONS>> out;
    rd_req_m7_s: chan<ram::ReadReq<RAM_ADDR_WIDTH, RAM_NUM_PARTITIONS>> out;
    wr_req_m0_s: chan<ram::WriteReq<RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>> out;
    wr_req_m1_s: chan<ram::WriteReq<RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>> out;
    wr_req_m2_s: chan<ram::WriteReq<RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>> out;
    wr_req_m3_s: chan<ram::WriteReq<RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>> out;
    wr_req_m4_s: chan<ram::WriteReq<RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>> out;
    wr_req_m5_s: chan<ram::WriteReq<RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>> out;
    wr_req_m6_s: chan<ram::WriteReq<RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>> out;
    wr_req_m7_s: chan<ram::WriteReq<RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>> out;

    config(
           input_r: chan<SequenceExecutorPacket> in,
           output_s: chan<ZstdDecodedPacket> out,
           output_mem_wr_data_in_s: chan<MemWriterDataPacket> out,
           ram_resp_output_r: chan<SequenceExecutorPacket> in,
           ram_resp_output_s: chan<SequenceExecutorPacket> out,
           rd_req_m0_s: chan<ram::ReadReq<RAM_ADDR_WIDTH, RAM_NUM_PARTITIONS>> out,
           rd_req_m1_s: chan<ram::ReadReq<RAM_ADDR_WIDTH, RAM_NUM_PARTITIONS>> out,
           rd_req_m2_s: chan<ram::ReadReq<RAM_ADDR_WIDTH, RAM_NUM_PARTITIONS>> out,
           rd_req_m3_s: chan<ram::ReadReq<RAM_ADDR_WIDTH, RAM_NUM_PARTITIONS>> out,
           rd_req_m4_s: chan<ram::ReadReq<RAM_ADDR_WIDTH, RAM_NUM_PARTITIONS>> out,
           rd_req_m5_s: chan<ram::ReadReq<RAM_ADDR_WIDTH, RAM_NUM_PARTITIONS>> out,
           rd_req_m6_s: chan<ram::ReadReq<RAM_ADDR_WIDTH, RAM_NUM_PARTITIONS>> out,
           rd_req_m7_s: chan<ram::ReadReq<RAM_ADDR_WIDTH, RAM_NUM_PARTITIONS>> out,
           rd_resp_m0_r: chan<ram::ReadResp<RAM_DATA_WIDTH>> in,
           rd_resp_m1_r: chan<ram::ReadResp<RAM_DATA_WIDTH>> in,
           rd_resp_m2_r: chan<ram::ReadResp<RAM_DATA_WIDTH>> in,
           rd_resp_m3_r: chan<ram::ReadResp<RAM_DATA_WIDTH>> in,
           rd_resp_m4_r: chan<ram::ReadResp<RAM_DATA_WIDTH>> in,
           rd_resp_m5_r: chan<ram::ReadResp<RAM_DATA_WIDTH>> in,
           rd_resp_m6_r: chan<ram::ReadResp<RAM_DATA_WIDTH>> in,
           rd_resp_m7_r: chan<ram::ReadResp<RAM_DATA_WIDTH>> in,
           wr_req_m0_s: chan<ram::WriteReq<RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>> out,
           wr_req_m1_s: chan<ram::WriteReq<RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>> out,
           wr_req_m2_s: chan<ram::WriteReq<RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>> out,
           wr_req_m3_s: chan<ram::WriteReq<RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>> out,
           wr_req_m4_s: chan<ram::WriteReq<RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>> out,
           wr_req_m5_s: chan<ram::WriteReq<RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>> out,
           wr_req_m6_s: chan<ram::WriteReq<RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>> out,
           wr_req_m7_s: chan<ram::WriteReq<RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>> out,
           wr_resp_m0_r: chan<ram::WriteResp> in,
           wr_resp_m1_r: chan<ram::WriteResp> in,
           wr_resp_m2_r: chan<ram::WriteResp> in,
           wr_resp_m3_r: chan<ram::WriteResp> in,
           wr_resp_m4_r: chan<ram::WriteResp> in,
           wr_resp_m5_r: chan<ram::WriteResp> in,
           wr_resp_m6_r: chan<ram::WriteResp> in,
           wr_resp_m7_r: chan<ram::WriteResp> in
    ) {
        let (ram_comp_input_s, ram_comp_input_r) = chan<RamWrRespHandlerData<RAM_ADDR_WIDTH>, u32:1>("ram_comp_input");
        let (ram_comp_output_s, ram_comp_output_r) = chan<HistoryBufferPtr<RAM_ADDR_WIDTH>, u32:1>("ram_comp_output");
        let (ram_resp_input_s, ram_resp_input_r) = chan<RamRdRespHandlerData, u32:1>("ram_resp_input");

        spawn RamWrRespHandler<RAM_ADDR_WIDTH>(
            ram_comp_input_r, ram_comp_output_s,
            wr_resp_m0_r, wr_resp_m1_r, wr_resp_m2_r, wr_resp_m3_r,
            wr_resp_m4_r, wr_resp_m5_r, wr_resp_m6_r, wr_resp_m7_r);

        spawn RamRdRespHandler(
            ram_resp_input_r, ram_resp_output_s,
            rd_resp_m0_r, rd_resp_m1_r, rd_resp_m2_r,
            rd_resp_m3_r, rd_resp_m4_r, rd_resp_m5_r, rd_resp_m6_r, rd_resp_m7_r);

        (
            input_r, output_s, output_mem_wr_data_in_s,
            ram_comp_input_s, ram_comp_output_r,
            ram_resp_input_s, ram_resp_output_r,
            rd_req_m0_s, rd_req_m1_s, rd_req_m2_s, rd_req_m3_s,
            rd_req_m4_s, rd_req_m5_s, rd_req_m6_s, rd_req_m7_s,
            wr_req_m0_s, wr_req_m1_s, wr_req_m2_s, wr_req_m3_s,
            wr_req_m4_s, wr_req_m5_s, wr_req_m6_s, wr_req_m7_s,
        )
    }

    init {
        const_assert!(INIT_HB_PTR_RAM < RAM_NUM);
        const_assert!(INIT_HB_PTR_ADDR <= (std::unsigned_max_value<RAM_ADDR_WIDTH>() as u32));

        type RamAddr = bits[RAM_ADDR_WIDTH];
        let INIT_HB_PTR = HistoryBufferPtr {
            number: INIT_HB_PTR_RAM as RamNumber, addr: INIT_HB_PTR_ADDR as RamAddr
        };
        SequenceExecutorState {
            status: SequenceExecutorStatus::IDLE,
            packet: zero!<SequenceExecutorPacket>(),
            packet_valid: false,
            hyp_ptr: INIT_HB_PTR,
            real_ptr: INIT_HB_PTR,
            hb_len: INIT_HB_LENGTH,
            repeat_offsets: Offset[3]:[Offset:1, Offset:4, Offset:8],
            repeat_req: false,
            seq_cnt: false
        }
    }

    next(state: SequenceExecutorState<RAM_ADDR_WIDTH>) {
        let tok0 = join();
        type Status = SequenceExecutorStatus;
        type State = SequenceExecutorState<RAM_ADDR_WIDTH>;
        type MsgType = SequenceExecutorMessageType;
        type Packet = SequenceExecutorPacket;
        type ReadReq = ram::ReadReq<RAM_ADDR_WIDTH, RAM_NUM_PARTITIONS>;
        type ReadResp = ram::ReadResp<RAM_DATA_WIDTH>;
        type WriteReq = ram::WriteReq<RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>;
        type WriteResp = ram::WriteResp;

        const ZERO_READ_REQS = ReadReq[RAM_NUM]:[zero!<ReadReq>(), ...];
        const ZERO_WRITE_REQS = WriteReq[RAM_NUM]:[zero!<WriteReq>(), ...];
        const ZERO_ORDER = RamOrder[RAM_NUM]:[RamOrder:0, ...];

        // Recieve literals and sequences from the input channel ...
        let do_recv_input = !state.packet_valid && state.status != Status::SEQUENCE_READ &&
                            state.status != Status::SEQUENCE_WRITE;
        let (tok1_0, input_packet, input_packet_valid) =
            recv_if_non_blocking(tok0, input_r, do_recv_input, zero!<Packet>());

        // ... or our own sequences from the looped channel
        let do_recv_ram =
            (state.status == Status::SEQUENCE_READ || state.status == Status::SEQUENCE_WRITE);
        let (tok1_1, ram_packet, ram_packet_valid) =
            recv_if_non_blocking(tok0, ram_resp_output_r, do_recv_ram, zero!<Packet>());

        // Read RAM write completion, used for monitoring the real state
        // of the RAM and eventually changing the state to IDLE.
        // Going through the IDLE state is required for changing between
        // Literals and Sequences (and the other way around) and between every
        // Sequence read from the input (original sequence from the ZSTD stream).
        let (tok1_2, real_ptr, real_ptr_valid) =
            recv_non_blocking(tok0, ram_comp_output_r, zero!<HistoryBufferPtr>());
        if real_ptr_valid {
            trace_fmt!("SequenceExecutor:: Received completion update");
        } else { };

        let real_ptr = if real_ptr_valid { real_ptr } else { state.real_ptr };
        let tok1 = join(tok1_0, tok1_1, tok1_2);

        // Since we either get data from input, from frame, or from state,
        // we are always working on a single packet. The current state
        // can be use to determine the source of the packet.
        let (packet, packet_valid) = if input_packet_valid {
            (input_packet, true)
        } else if ram_packet_valid {
            (ram_packet, true)
        } else {
            (state.packet, state.packet_valid)
        };

        // if we are in the IDLE state and have a valid packet stored in the state,
        // or we have a new packet from the input go to the corresponding
        // processing step immediately. (added to be able to process a single
        // literal in one next() evaluation)
        let status = match (state.status, packet_valid, packet.msg_type) {
            (Status::IDLE, true, MsgType::LITERAL) => Status::LITERAL_WRITE,
            (Status::IDLE, true, MsgType::SEQUENCE) => Status::SEQUENCE_READ,
            _ => state.status,
        };

        let NO_VALID_PACKET_STATE = State { packet, packet_valid, real_ptr, ..state };
        let (write_reqs, read_reqs, order, new_state) = match (
            status, packet_valid, packet.msg_type
        ) {
            // Handling LITERAL_WRITE
            (Status::LITERAL_WRITE, true, MsgType::LITERAL) => {
                trace_fmt!("SequenceExecutor:: Handling LITERAL packet in LITERAL_WRITE step");
                let (write_reqs, new_hyp_ptr) =
                    literal_packet_to_write_reqs<HISTORY_BUFFER_SIZE_KB>(state.hyp_ptr, packet);
                let new_repeat_req = packet.length == CopyOrMatchLength:0;
                let hb_add = (packet.length >> 3) as HistoryBufferLength;
                let new_hb_len = std::mod_pow2(state.hb_len + hb_add, RAM_SIZE_TOTAL);
                (
                    write_reqs, ZERO_READ_REQS, ZERO_ORDER,
                    State {
                        status: Status::LITERAL_WRITE,
                        packet: zero!<SequenceExecutorPacket>(),
                        packet_valid: false,
                        hyp_ptr: new_hyp_ptr,
                        real_ptr,
                        repeat_offsets: state.repeat_offsets,
                        repeat_req: new_repeat_req,
                        hb_len: new_hb_len,
                        seq_cnt: false
                    },
                )
            },
            (Status::LITERAL_WRITE, _, _) => {
                let status =
                    if real_ptr == state.hyp_ptr { Status::IDLE } else { Status::LITERAL_WRITE };
                (
                    ZERO_WRITE_REQS, ZERO_READ_REQS, ZERO_ORDER,
                    State { status, ..NO_VALID_PACKET_STATE },
                )
            },
            // Handling SEQUENCE_READ
            (Status::SEQUENCE_READ, true, MsgType::SEQUENCE) => {
                trace_fmt!("Handling SEQUENCE in SEQUENCE_READ state");
                let (packet, new_repeat_offsets) = if !state.seq_cnt {
                    handle_reapeated_offset_for_sequences(
                        packet, state.repeat_offsets, state.repeat_req)
                } else {
                    (packet, state.repeat_offsets)
                };
                let (read_reqs, order, packet, packet_valid) = sequence_packet_to_read_reqs<
                    HISTORY_BUFFER_SIZE_KB>(
                    state.hyp_ptr, packet, state.hb_len);

                (
                    ZERO_WRITE_REQS, read_reqs, order,
                    SequenceExecutorState {
                        status: Status::SEQUENCE_WRITE,
                        packet,
                        packet_valid,
                        hyp_ptr: state.hyp_ptr,
                        real_ptr,
                        repeat_offsets: new_repeat_offsets,
                        repeat_req: false,
                        hb_len: state.hb_len,
                        seq_cnt: packet_valid
                    },
                )
            },
            (Status::SEQUENCE_READ, _, _) => {
                let ZERO_RETURN = (ZERO_WRITE_REQS, ZERO_READ_REQS, ZERO_ORDER, zero!<State>());
                fail!("should_no_happen", (ZERO_RETURN))
            },
            // Handling SEQUENCE_WRITE
            (Status::SEQUENCE_WRITE, true, MsgType::LITERAL) => {
                trace_fmt!("Handling LITERAL in SEQUENCE_WRITE state: {}", status);
                let (write_reqs, new_hyp_ptr) =
                    literal_packet_to_write_reqs<HISTORY_BUFFER_SIZE_KB>(state.hyp_ptr, packet);
                let hb_add = packet.length as HistoryBufferLength;
                let new_hb_len = std::mod_pow2(state.hb_len + hb_add, RAM_SIZE_TOTAL);

                (
                    write_reqs, ZERO_READ_REQS, ZERO_ORDER,
                    SequenceExecutorState {
                        status: zero!<SequenceExecutorStatus>(),
                        packet: state.packet,
                        packet_valid: state.packet_valid,
                        hyp_ptr: new_hyp_ptr,
                        real_ptr,
                        repeat_offsets: state.repeat_offsets,
                        repeat_req: state.repeat_req,
                        hb_len: new_hb_len,
                        seq_cnt: state.seq_cnt
                    },
                )
            },
            (Status::SEQUENCE_WRITE, _, _) => {
                let status = if real_ptr == state.hyp_ptr {
                    Status::IDLE
                } else if state.seq_cnt {
                    Status::SEQUENCE_READ
                } else {
                    Status::SEQUENCE_WRITE
                };
                (
                    ZERO_WRITE_REQS, ZERO_READ_REQS, ZERO_ORDER,
                    State { status, ..NO_VALID_PACKET_STATE },
                )
            },
            // Handling IDLE
            _ => {
                let status = Status::IDLE;
                (
                    ZERO_WRITE_REQS, ZERO_READ_REQS, ZERO_ORDER,
                    State { status, ..NO_VALID_PACKET_STATE },
                )
            },
        };

        let tok2_1 = send_if(tok1, wr_req_m0_s, (write_reqs[0]).mask != RAM_REQ_MASK_NONE, write_reqs[0]);
        let tok2_2 = send_if(tok1, wr_req_m1_s, (write_reqs[1]).mask != RAM_REQ_MASK_NONE, write_reqs[1]);
        let tok2_3 = send_if(tok1, wr_req_m2_s, (write_reqs[2]).mask != RAM_REQ_MASK_NONE, write_reqs[2]);
        let tok2_4 = send_if(tok1, wr_req_m3_s, (write_reqs[3]).mask != RAM_REQ_MASK_NONE, write_reqs[3]);
        let tok2_5 = send_if(tok1, wr_req_m4_s, (write_reqs[4]).mask != RAM_REQ_MASK_NONE, write_reqs[4]);
        let tok2_6 = send_if(tok1, wr_req_m5_s, (write_reqs[5]).mask != RAM_REQ_MASK_NONE, write_reqs[5]);
        let tok2_7 = send_if(tok1, wr_req_m6_s, (write_reqs[6]).mask != RAM_REQ_MASK_NONE, write_reqs[6]);
        let tok2_8 = send_if(tok1, wr_req_m7_s, (write_reqs[7]).mask != RAM_REQ_MASK_NONE, write_reqs[7]);

        // Write to output ask for completion
        let (do_write, wr_resp_handler_data) = create_ram_wr_data(write_reqs, new_state.hyp_ptr);
        if do_write {
            trace_fmt!("Sending request to RamWrRespHandler: {:#x}", wr_resp_handler_data);
        } else { };
        let tok2_9 = send_if(tok1, ram_comp_input_s, do_write, wr_resp_handler_data);

        let output_data = decode_literal_packet(packet);
        let do_write_output = do_write || (packet.last && packet.msg_type == SequenceExecutorMessageType::LITERAL);
        if do_write_output { trace_fmt!("Sending output data: {:#x}", output_data); } else {  };
        let tok2_10_0 = send_if(tok1, output_s, do_write_output, output_data);
        let output_mem_wr_data_in = convert_output_packet<AXI_ADDR_W, AXI_DATA_W>(output_data);
        let tok2_10_1 = send_if(tok1, output_mem_wr_data_in_s, do_write_output, output_mem_wr_data_in);

        // Ask for response
        let tok2_11 = send_if(tok1, rd_req_m0_s, (read_reqs[0]).mask != RAM_REQ_MASK_NONE, read_reqs[0]);
        let tok2_12 = send_if(tok1, rd_req_m1_s, (read_reqs[1]).mask != RAM_REQ_MASK_NONE, read_reqs[1]);
        let tok2_13 = send_if(tok1, rd_req_m2_s, (read_reqs[2]).mask != RAM_REQ_MASK_NONE, read_reqs[2]);
        let tok2_14 = send_if(tok1, rd_req_m3_s, (read_reqs[3]).mask != RAM_REQ_MASK_NONE, read_reqs[3]);
        let tok2_15 = send_if(tok1, rd_req_m4_s, (read_reqs[4]).mask != RAM_REQ_MASK_NONE, read_reqs[4]);
        let tok2_16 = send_if(tok1, rd_req_m5_s, (read_reqs[5]).mask != RAM_REQ_MASK_NONE, read_reqs[5]);
        let tok2_17 = send_if(tok1, rd_req_m6_s, (read_reqs[6]).mask != RAM_REQ_MASK_NONE, read_reqs[6]);
        let tok2_18 = send_if(tok1, rd_req_m7_s, (read_reqs[7]).mask != RAM_REQ_MASK_NONE, read_reqs[7]);

        let (do_read, rd_resp_handler_data) =
            create_ram_rd_data<RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>
            (read_reqs, order, packet.last, new_state.packet_valid);
        if do_read {
            trace_fmt!("Sending request to RamRdRespHandler: {:#x}", rd_resp_handler_data);
        } else { };
        let tok2_19 = send_if(tok1, ram_resp_input_s, do_read, rd_resp_handler_data);

        new_state
    }
}

pub const ZSTD_HISTORY_BUFFER_SIZE_KB: u32 = u32:64;
pub const ZSTD_RAM_SIZE = ram_size(ZSTD_HISTORY_BUFFER_SIZE_KB);
pub const ZSTD_RAM_ADDR_WIDTH = ram_addr_width(ZSTD_HISTORY_BUFFER_SIZE_KB);
const ZSTD_AXI_DATA_W = u32:64;
const ZSTD_AXI_ADDR_W = u32:16;

pub proc SequenceExecutorZstd {
    type MemWriterDataPacket  = mem_writer::MemWriterDataPacket<ZSTD_AXI_DATA_W, ZSTD_AXI_ADDR_W>;

    init {  }

    config(
        input_r: chan<SequenceExecutorPacket> in,
        output_s: chan<ZstdDecodedPacket> out,
        output_mem_wr_data_in_s: chan<MemWriterDataPacket> out,
        looped_channel_r: chan<SequenceExecutorPacket> in,
        looped_channel_s: chan<SequenceExecutorPacket> out,
        rd_req_m0_s: chan<ram::ReadReq<ZSTD_RAM_ADDR_WIDTH, RAM_NUM_PARTITIONS>> out,
        rd_req_m1_s: chan<ram::ReadReq<ZSTD_RAM_ADDR_WIDTH, RAM_NUM_PARTITIONS>> out,
        rd_req_m2_s: chan<ram::ReadReq<ZSTD_RAM_ADDR_WIDTH, RAM_NUM_PARTITIONS>> out,
        rd_req_m3_s: chan<ram::ReadReq<ZSTD_RAM_ADDR_WIDTH, RAM_NUM_PARTITIONS>> out,
        rd_req_m4_s: chan<ram::ReadReq<ZSTD_RAM_ADDR_WIDTH, RAM_NUM_PARTITIONS>> out,
        rd_req_m5_s: chan<ram::ReadReq<ZSTD_RAM_ADDR_WIDTH, RAM_NUM_PARTITIONS>> out,
        rd_req_m6_s: chan<ram::ReadReq<ZSTD_RAM_ADDR_WIDTH, RAM_NUM_PARTITIONS>> out,
        rd_req_m7_s: chan<ram::ReadReq<ZSTD_RAM_ADDR_WIDTH, RAM_NUM_PARTITIONS>> out,
        rd_resp_m0_r: chan<ram::ReadResp<RAM_DATA_WIDTH>> in,
        rd_resp_m1_r: chan<ram::ReadResp<RAM_DATA_WIDTH>> in,
        rd_resp_m2_r: chan<ram::ReadResp<RAM_DATA_WIDTH>> in,
        rd_resp_m3_r: chan<ram::ReadResp<RAM_DATA_WIDTH>> in,
        rd_resp_m4_r: chan<ram::ReadResp<RAM_DATA_WIDTH>> in,
        rd_resp_m5_r: chan<ram::ReadResp<RAM_DATA_WIDTH>> in,
        rd_resp_m6_r: chan<ram::ReadResp<RAM_DATA_WIDTH>> in,
        rd_resp_m7_r: chan<ram::ReadResp<RAM_DATA_WIDTH>> in,
        wr_req_m0_s: chan<ram::WriteReq<ZSTD_RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>> out,
        wr_req_m1_s: chan<ram::WriteReq<ZSTD_RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>> out,
        wr_req_m2_s: chan<ram::WriteReq<ZSTD_RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>> out,
        wr_req_m3_s: chan<ram::WriteReq<ZSTD_RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>> out,
        wr_req_m4_s: chan<ram::WriteReq<ZSTD_RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>> out,
        wr_req_m5_s: chan<ram::WriteReq<ZSTD_RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>> out,
        wr_req_m6_s: chan<ram::WriteReq<ZSTD_RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>> out,
        wr_req_m7_s: chan<ram::WriteReq<ZSTD_RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>> out,
        wr_resp_m0_r: chan<ram::WriteResp> in,
        wr_resp_m1_r: chan<ram::WriteResp> in,
        wr_resp_m2_r: chan<ram::WriteResp> in,
        wr_resp_m3_r: chan<ram::WriteResp> in,
        wr_resp_m4_r: chan<ram::WriteResp> in,
        wr_resp_m5_r: chan<ram::WriteResp> in,
        wr_resp_m6_r: chan<ram::WriteResp> in,
        wr_resp_m7_r: chan<ram::WriteResp> in
    ) {
        spawn SequenceExecutor<ZSTD_HISTORY_BUFFER_SIZE_KB,
                               ZSTD_AXI_DATA_W, ZSTD_AXI_ADDR_W> (
            input_r, output_s, output_mem_wr_data_in_s,
            looped_channel_r, looped_channel_s,
            rd_req_m0_s, rd_req_m1_s, rd_req_m2_s, rd_req_m3_s,
            rd_req_m4_s, rd_req_m5_s, rd_req_m6_s, rd_req_m7_s,
            rd_resp_m0_r, rd_resp_m1_r, rd_resp_m2_r, rd_resp_m3_r,
            rd_resp_m4_r, rd_resp_m5_r, rd_resp_m6_r, rd_resp_m7_r,
            wr_req_m0_s, wr_req_m1_s, wr_req_m2_s, wr_req_m3_s,
            wr_req_m4_s, wr_req_m5_s, wr_req_m6_s, wr_req_m7_s,
            wr_resp_m0_r, wr_resp_m1_r, wr_resp_m2_r, wr_resp_m3_r,
            wr_resp_m4_r, wr_resp_m5_r, wr_resp_m6_r, wr_resp_m7_r
        );
    }

    next (state: ()) { }
}

const LITERAL_TEST_INPUT_DATA = SequenceExecutorPacket[8]:[
     SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::LITERAL,
        length: CopyOrMatchLength:64,
        content: CopyOrMatchContent:0xAA00BB11CC22DD33,
        last: false
    },
    SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::LITERAL,
        length: CopyOrMatchLength:64,
        content: CopyOrMatchContent:0x447733220088CCFF,
        last: false
    },
    SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::LITERAL,
        length: CopyOrMatchLength:32,
        content: CopyOrMatchContent:0x88AA0022,
        last: false
    },
    SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::LITERAL,
        length: CopyOrMatchLength:32,
        content: CopyOrMatchContent:0xFFEEDD11,
        last: false
    },
    SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::LITERAL,
        length: CopyOrMatchLength:64,
        content: CopyOrMatchContent:0x9DAF8B41C913EFDA,
        last: false
    },
    SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::LITERAL,
        length: CopyOrMatchLength:64,
        content: CopyOrMatchContent:0x157D8C7EB8B97CA3,
        last: false
    },
    SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::LITERAL,
        length: CopyOrMatchLength:0,
        content: CopyOrMatchContent:0x0,
        last: false
    },
    SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::LITERAL,
        length: CopyOrMatchLength:0,
        content: CopyOrMatchContent:0x0,
        last: true
    },
];

const LITERAL_TEST_MEMORY_CONTENT:(TestRamAddr, RamData)[3][RAM_NUM] = [
    [
        (TestRamAddr:127, RamData:0x33),
        (TestRamAddr:0, RamData:0xFF),
        (TestRamAddr:1, RamData:0x22)
    ],
    [
        (TestRamAddr:127, RamData:0xDD),
        (TestRamAddr:0, RamData:0xCC),
        (TestRamAddr:1, RamData:0x00)
    ],
    [
        (TestRamAddr:127, RamData:0x22),
        (TestRamAddr:0, RamData:0x88),
        (TestRamAddr:1, RamData:0xAA)
    ],
    [
        (TestRamAddr:127, RamData:0xCC),
        (TestRamAddr:0, RamData:0x00),
        (TestRamAddr:1, RamData:0x88)
    ],
    [
        (TestRamAddr:127, RamData:0x11),
        (TestRamAddr:0, RamData:0x22),
        (TestRamAddr:1, RamData:0x11)
    ],
    [
        (TestRamAddr:127, RamData:0xBB),
        (TestRamAddr:0, RamData:0x33),
        (TestRamAddr:1, RamData:0xDD)
    ],
    [
        (TestRamAddr:127, RamData:0x00),
        (TestRamAddr:0, RamData:0x77),
        (TestRamAddr:1, RamData:0xEE)
    ],
    [
        (TestRamAddr:127, RamData:0xAA),
        (TestRamAddr:0, RamData:0x44),
        (TestRamAddr:1, RamData:0xFF)
    ],
];

#[test_proc]
proc SequenceExecutorLiteralsTest {
    type MemWriterDataPacket  = mem_writer::MemWriterDataPacket<TEST_DATA_W, TEST_ADDR_W>;
    terminator: chan<bool> out;

    input_s: chan<SequenceExecutorPacket<TEST_RAM_ADDR_WIDTH>> out;
    output_r: chan<ZstdDecodedPacket> in;
    output_mem_wr_data_in_r: chan<MemWriterDataPacket> in;

    print_start_s: chan<()> out;
    print_finish_r: chan<()> in;

    ram_rd_req_s: chan<TestReadReq>[RAM_NUM] out;
    ram_rd_resp_r: chan<TestReadResp>[RAM_NUM] in;
    ram_wr_req_s: chan<TestWriteReq>[RAM_NUM] out;
    ram_wr_resp_r: chan<TestWriteResp>[RAM_NUM] in;

    config(terminator: chan<bool> out) {
        let (input_s,  input_r) = chan<SequenceExecutorPacket<TEST_RAM_ADDR_WIDTH>>("input");
        let (output_s, output_r) = chan<ZstdDecodedPacket>("output");
        let (output_mem_wr_data_in_s,  output_mem_wr_data_in_r) = chan<MemWriterDataPacket>("output_mem_wr_data_in");

        let (looped_channel_s, looped_channel_r) = chan<SequenceExecutorPacket>("looped_channels");

        let (print_start_s, print_start_r) = chan<()>("print_start");
        let (print_finish_s, print_finish_r) = chan<()>("print_finish");

        let (ram_rd_req_s,  ram_rd_req_r) = chan<TestReadReq>[RAM_NUM]("ram_rd_req");
        let (ram_rd_resp_s, ram_rd_resp_r) = chan<TestReadResp>[RAM_NUM]("ram_rd_resp");
        let (ram_wr_req_s,  ram_wr_req_r) = chan<TestWriteReq>[RAM_NUM]("ram_wr_req");
        let (ram_wr_resp_s, ram_wr_resp_r) = chan<TestWriteResp>[RAM_NUM]("ram_wr_resp");

        let INIT_HB_PTR_ADDR = u32:127;
        spawn SequenceExecutor<
            TEST_HISTORY_BUFFER_SIZE_KB,
            TEST_DATA_W, TEST_ADDR_W,
            TEST_RAM_SIZE,
            TEST_RAM_ADDR_WIDTH,
            INIT_HB_PTR_ADDR,
        > (
            input_r, output_s, output_mem_wr_data_in_s,
            looped_channel_r, looped_channel_s,
            ram_rd_req_s[0], ram_rd_req_s[1], ram_rd_req_s[2], ram_rd_req_s[3],
            ram_rd_req_s[4], ram_rd_req_s[5], ram_rd_req_s[6], ram_rd_req_s[7],
            ram_rd_resp_r[0], ram_rd_resp_r[1], ram_rd_resp_r[2], ram_rd_resp_r[3],
            ram_rd_resp_r[4], ram_rd_resp_r[5], ram_rd_resp_r[6], ram_rd_resp_r[7],
            ram_wr_req_s[0], ram_wr_req_s[1], ram_wr_req_s[2], ram_wr_req_s[3],
            ram_wr_req_s[4], ram_wr_req_s[5], ram_wr_req_s[6], ram_wr_req_s[7],
            ram_wr_resp_r[0], ram_wr_resp_r[1], ram_wr_resp_r[2], ram_wr_resp_r[3],
            ram_wr_resp_r[4], ram_wr_resp_r[5], ram_wr_resp_r[6], ram_wr_resp_r[7]
        );

        spawn ram_printer::RamPrinter<
            RAM_DATA_WIDTH, TEST_RAM_SIZE, RAM_NUM_PARTITIONS,
            TEST_RAM_ADDR_WIDTH, RAM_NUM>
            (print_start_r, print_finish_s, ram_rd_req_s, ram_rd_resp_r);

        spawn ram::RamModel<
            RAM_DATA_WIDTH, TEST_RAM_SIZE, RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED>
            (ram_rd_req_r[0], ram_rd_resp_s[0], ram_wr_req_r[0], ram_wr_resp_s[0]);
        spawn ram::RamModel<
            RAM_DATA_WIDTH, TEST_RAM_SIZE, RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED>
            (ram_rd_req_r[1], ram_rd_resp_s[1], ram_wr_req_r[1], ram_wr_resp_s[1]);
        spawn ram::RamModel<
            RAM_DATA_WIDTH, TEST_RAM_SIZE, RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED>
            (ram_rd_req_r[2], ram_rd_resp_s[2], ram_wr_req_r[2], ram_wr_resp_s[2]);
        spawn ram::RamModel<
            RAM_DATA_WIDTH, TEST_RAM_SIZE, RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED>
            (ram_rd_req_r[3], ram_rd_resp_s[3], ram_wr_req_r[3], ram_wr_resp_s[3]);
        spawn ram::RamModel<
            RAM_DATA_WIDTH, TEST_RAM_SIZE, RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED>
            (ram_rd_req_r[4], ram_rd_resp_s[4], ram_wr_req_r[4], ram_wr_resp_s[4]);
        spawn ram::RamModel<
            RAM_DATA_WIDTH, TEST_RAM_SIZE, RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED>
            (ram_rd_req_r[5], ram_rd_resp_s[5], ram_wr_req_r[5], ram_wr_resp_s[5]);
        spawn ram::RamModel<
            RAM_DATA_WIDTH, TEST_RAM_SIZE, RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED>
            (ram_rd_req_r[6], ram_rd_resp_s[6], ram_wr_req_r[6], ram_wr_resp_s[6]);
        spawn ram::RamModel<
            RAM_DATA_WIDTH, TEST_RAM_SIZE, RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED>
            (ram_rd_req_r[7], ram_rd_resp_s[7], ram_wr_req_r[7], ram_wr_resp_s[7]);

        (
            terminator,
            input_s, output_r, output_mem_wr_data_in_r,
            print_start_s, print_finish_r,
            ram_rd_req_s, ram_rd_resp_r,
            ram_wr_req_s, ram_wr_resp_r
        )
    }

    init {  }

    next(state: ()) {
        let tok = join();
        for (i, ()): (u32, ()) in u32:0..array_size(LITERAL_TEST_INPUT_DATA) {
            let tok = send(tok, input_s, LITERAL_TEST_INPUT_DATA[i]);
            // Don't receive when there's an empty literals packet which is not last
            if (LITERAL_TEST_INPUT_DATA[i].msg_type != SequenceExecutorMessageType::LITERAL ||
                LITERAL_TEST_INPUT_DATA[i].length != CopyOrMatchLength:0 ||
                LITERAL_TEST_INPUT_DATA[i].last) {
                let (tok, recv_data) = recv(tok, output_r);
                let expected = decode_literal_packet(LITERAL_TEST_INPUT_DATA[i]);
                assert_eq(expected, recv_data);
                let (tok, recv_mem_writer_data) = recv(tok, output_mem_wr_data_in_r);
                let expected_mem_writer_data = convert_output_packet<TEST_ADDR_W, TEST_DATA_W>(expected);
                assert_eq(expected_mem_writer_data, recv_mem_writer_data);
            } else {}
        }(());

        for (i, ()): (u32, ()) in u32:0..RAM_NUM {
            for (j, ()): (u32, ()) in u32:0..array_size(LITERAL_TEST_MEMORY_CONTENT[0]) {
                let addr = LITERAL_TEST_MEMORY_CONTENT[i][j].0;
                let tok = send(tok, ram_rd_req_s[i], TestReadReq { addr, mask: RAM_REQ_MASK_ALL });
                let (tok, resp) = recv(tok, ram_rd_resp_r[i]);
                let expected = LITERAL_TEST_MEMORY_CONTENT[i][j].1;
                assert_eq(expected, resp.data);
            }(());
        }(());

        // Print RAM content
        let tok = send(tok, print_start_s, ());
        let (tok, _) = recv(tok, print_finish_r);

        send(tok, terminator, true);
    }
}

const SEQUENCE_TEST_INPUT_SEQUENCES = SequenceExecutorPacket[11]: [
    SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::SEQUENCE,
        length: CopyOrMatchLength:9,
        content: CopyOrMatchContent:13,
        last: false
    },
    SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::SEQUENCE,
        length: CopyOrMatchLength:1,
        content: CopyOrMatchContent:7,
        last:false
    },
    SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::SEQUENCE,
        length: CopyOrMatchLength:1,
        content: CopyOrMatchContent:8,
        last:false
    },
    SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::SEQUENCE,
        length: CopyOrMatchLength:5,
        content: CopyOrMatchContent:13,
        last:false
    },
    SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::SEQUENCE,
        length: CopyOrMatchLength:3,
        content: CopyOrMatchContent:3,
        last:false
    },
    SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::SEQUENCE,
        length: CopyOrMatchLength:1,
        content: CopyOrMatchContent:1,
        last:false
    },
    SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::LITERAL,
        length: CopyOrMatchLength:0,
        content: CopyOrMatchContent:0,
        last:false
    },
    SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::SEQUENCE,
        length: CopyOrMatchLength:1,
        content: CopyOrMatchContent:3,
        last:false
    },
    SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::LITERAL,
        length: CopyOrMatchLength:0,
        content: CopyOrMatchContent:0,
        last:false
    },
    SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::SEQUENCE,
        length: CopyOrMatchLength:10,
        content: CopyOrMatchContent:2,
        last:true,
    },
    SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::SEQUENCE,
        length: CopyOrMatchLength:1,
        content: CopyOrMatchContent:3,
        last:false
    },
];

const SEQUENCE_TEST_EXPECTED_SEQUENCE_RESULTS:ZstdDecodedPacket[11] = [
    ZstdDecodedPacket {
        data: BlockData:0x8C_7E_B8_B9_7C_A3_9D_AF,
        length: BlockPacketLength:64,
        last: false
    },
    ZstdDecodedPacket {
        data: BlockData:0x7D,
        length: BlockPacketLength:8,
        last: false
    },
    ZstdDecodedPacket {
        data: BlockData:0xB8,
        length: BlockPacketLength:8,
        last: false
    },
    ZstdDecodedPacket {
        data: BlockData:0xB8,
        length: BlockPacketLength:8,
        last: false
    },
    ZstdDecodedPacket {
        data: BlockData:0xB8_B9_7C_A3_9D,
        length: BlockPacketLength:40,
        last: false
    },
    ZstdDecodedPacket {
        data: BlockData:0xB9_7C_A3,
        length: BlockPacketLength:24,
        last: false
    },
    ZstdDecodedPacket {
        data: BlockData:0xB8,
        length: BlockPacketLength:8,
        last: false
    },
    ZstdDecodedPacket {
        data: BlockData:0x7C,
        length: BlockPacketLength:8,
        last: false
    },
    ZstdDecodedPacket {
        data: BlockData:0xB9_7C_A3_B8_B9_7C_A3_9D,
        length: BlockPacketLength:64,
        last: false
    },
    ZstdDecodedPacket {
        data: BlockData:0x7C_B8,
        length: BlockPacketLength:16,
        last: true
    },
    ZstdDecodedPacket {
        data: BlockData:0x9D,
        length: BlockPacketLength:8,
        last: false
    }
];

#[test_proc]
proc SequenceExecutorSequenceTest {
    type MemWriterDataPacket  = mem_writer::MemWriterDataPacket<TEST_DATA_W, TEST_ADDR_W>;
    terminator: chan<bool> out;

    input_s: chan<SequenceExecutorPacket> out;
    output_r: chan<ZstdDecodedPacket> in;
    output_mem_wr_data_in_r: chan<MemWriterDataPacket> in;

    print_start_s: chan<()> out;
    print_finish_r: chan<()> in;

    ram_rd_req_s: chan<TestReadReq>[RAM_NUM] out;
    ram_rd_resp_r: chan<TestReadResp>[RAM_NUM] in;
    ram_wr_req_s: chan<TestWriteReq>[RAM_NUM] out;
    ram_wr_resp_r: chan<TestWriteResp>[RAM_NUM] in;

    config(terminator: chan<bool> out) {
        let (input_s, input_r) = chan<SequenceExecutorPacket>("input");
        let (output_s, output_r) = chan<ZstdDecodedPacket>("output");
        let (output_mem_wr_data_in_s,  output_mem_wr_data_in_r) = chan<MemWriterDataPacket>("output_mem_wr_data_in");

        let (looped_channel_s, looped_channel_r) = chan<SequenceExecutorPacket>("looped_channel");

        let (print_start_s, print_start_r) = chan<()>("print_start");
        let (print_finish_s, print_finish_r) = chan<()>("print_finish");

        let (ram_rd_req_s, ram_rd_req_r) = chan<TestReadReq>[RAM_NUM]("ram_rd_req");
        let (ram_rd_resp_s, ram_rd_resp_r) = chan<TestReadResp>[RAM_NUM]("ram_rd_resp");
        let (ram_wr_req_s, ram_wr_req_r) = chan<TestWriteReq>[RAM_NUM]("ram_wr_req");
        let (ram_wr_resp_s, ram_wr_resp_r) = chan<TestWriteResp>[RAM_NUM]("ram_wr_resp");

        let INIT_HB_PTR_ADDR = u32:127;
        spawn SequenceExecutor<
            TEST_HISTORY_BUFFER_SIZE_KB,
            TEST_DATA_W, TEST_ADDR_W,
            TEST_RAM_SIZE,
            TEST_RAM_ADDR_WIDTH,
            INIT_HB_PTR_ADDR,
        > (
            input_r, output_s, output_mem_wr_data_in_s,
            looped_channel_r, looped_channel_s,
            ram_rd_req_s[0], ram_rd_req_s[1], ram_rd_req_s[2], ram_rd_req_s[3],
            ram_rd_req_s[4], ram_rd_req_s[5], ram_rd_req_s[6], ram_rd_req_s[7],
            ram_rd_resp_r[0], ram_rd_resp_r[1], ram_rd_resp_r[2], ram_rd_resp_r[3],
            ram_rd_resp_r[4], ram_rd_resp_r[5], ram_rd_resp_r[6], ram_rd_resp_r[7],
            ram_wr_req_s[0], ram_wr_req_s[1], ram_wr_req_s[2], ram_wr_req_s[3],
            ram_wr_req_s[4], ram_wr_req_s[5], ram_wr_req_s[6], ram_wr_req_s[7],
            ram_wr_resp_r[0], ram_wr_resp_r[1], ram_wr_resp_r[2], ram_wr_resp_r[3],
            ram_wr_resp_r[4], ram_wr_resp_r[5], ram_wr_resp_r[6], ram_wr_resp_r[7]
        );

        spawn ram_printer::RamPrinter<
            RAM_DATA_WIDTH, TEST_RAM_SIZE, RAM_NUM_PARTITIONS,
            TEST_RAM_ADDR_WIDTH, RAM_NUM>
            (print_start_r, print_finish_s, ram_rd_req_s, ram_rd_resp_r);

        spawn ram::RamModel<
            RAM_DATA_WIDTH, TEST_RAM_SIZE, RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED>
            (ram_rd_req_r[0], ram_rd_resp_s[0], ram_wr_req_r[0], ram_wr_resp_s[0]);
        spawn ram::RamModel<
            RAM_DATA_WIDTH, TEST_RAM_SIZE, RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED>
            (ram_rd_req_r[1], ram_rd_resp_s[1], ram_wr_req_r[1], ram_wr_resp_s[1]);
        spawn ram::RamModel<
            RAM_DATA_WIDTH, TEST_RAM_SIZE, RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED>
            (ram_rd_req_r[2], ram_rd_resp_s[2], ram_wr_req_r[2], ram_wr_resp_s[2]);
        spawn ram::RamModel<
            RAM_DATA_WIDTH, TEST_RAM_SIZE, RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED>
            (ram_rd_req_r[3], ram_rd_resp_s[3], ram_wr_req_r[3], ram_wr_resp_s[3]);
        spawn ram::RamModel<
            RAM_DATA_WIDTH, TEST_RAM_SIZE, RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED>
            (ram_rd_req_r[4], ram_rd_resp_s[4], ram_wr_req_r[4], ram_wr_resp_s[4]);
        spawn ram::RamModel<
            RAM_DATA_WIDTH, TEST_RAM_SIZE, RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED>
            (ram_rd_req_r[5], ram_rd_resp_s[5], ram_wr_req_r[5], ram_wr_resp_s[5]);
        spawn ram::RamModel<
            RAM_DATA_WIDTH, TEST_RAM_SIZE, RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED>
            (ram_rd_req_r[6], ram_rd_resp_s[6], ram_wr_req_r[6], ram_wr_resp_s[6]);
        spawn ram::RamModel<
            RAM_DATA_WIDTH, TEST_RAM_SIZE, RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED>
            (ram_rd_req_r[7], ram_rd_resp_s[7], ram_wr_req_r[7], ram_wr_resp_s[7]);

        (
            terminator,
            input_s, output_r, output_mem_wr_data_in_r,
            print_start_s, print_finish_r,
            ram_rd_req_s, ram_rd_resp_r, ram_wr_req_s, ram_wr_resp_r
        )
    }

    init {  }

    next(state: ()) {
        let tok = join();
        for (i, ()): (u32, ()) in u32:0..array_size(LITERAL_TEST_INPUT_DATA) {
            let tok = send(tok, input_s, LITERAL_TEST_INPUT_DATA[i]);
            // Don't receive when there's an empty literal packet which is not last
            if (LITERAL_TEST_INPUT_DATA[i].msg_type != SequenceExecutorMessageType::LITERAL ||
                LITERAL_TEST_INPUT_DATA[i].length != CopyOrMatchLength:0 ||
                LITERAL_TEST_INPUT_DATA[i].last) {
                let (tok, recv_data) = recv(tok, output_r);
                let expected = decode_literal_packet(LITERAL_TEST_INPUT_DATA[i]);
                assert_eq(expected, recv_data);
                let (tok, recv_mem_writer_data) = recv(tok, output_mem_wr_data_in_r);
                let expected_mem_writer_data = convert_output_packet<TEST_ADDR_W, TEST_DATA_W>(expected);
                assert_eq(expected_mem_writer_data, recv_mem_writer_data);
            } else {}
        }(());

        // Print RAM content
        let tok = send(tok, print_start_s, ());
        let (tok, _) = recv(tok, print_finish_r);

        let tok = send(tok, input_s, SEQUENCE_TEST_INPUT_SEQUENCES[0]);
        let (tok, recv_data) = recv(tok, output_r);
        assert_eq(SEQUENCE_TEST_EXPECTED_SEQUENCE_RESULTS[0], recv_data);
        let (tok, recv_data) = recv(tok, output_r);
        assert_eq(SEQUENCE_TEST_EXPECTED_SEQUENCE_RESULTS[1], recv_data);

        let tok = send(tok, input_s, SEQUENCE_TEST_INPUT_SEQUENCES[1]);
        let (tok, recv_data) = recv(tok, output_r);
        assert_eq(SEQUENCE_TEST_EXPECTED_SEQUENCE_RESULTS[2], recv_data);

        let tok = send(tok, input_s, SEQUENCE_TEST_INPUT_SEQUENCES[2]);
        let (tok, recv_data) = recv(tok, output_r);
        assert_eq(SEQUENCE_TEST_EXPECTED_SEQUENCE_RESULTS[3], recv_data);

        let tok = send(tok, input_s, SEQUENCE_TEST_INPUT_SEQUENCES[3]);
        let (tok, recv_data) = recv(tok, output_r);
        assert_eq(SEQUENCE_TEST_EXPECTED_SEQUENCE_RESULTS[4], recv_data);

        let tok = send(tok, input_s, SEQUENCE_TEST_INPUT_SEQUENCES[4]);
        let (tok, recv_data) = recv(tok, output_r);
        assert_eq(SEQUENCE_TEST_EXPECTED_SEQUENCE_RESULTS[5], recv_data);

        let tok = send(tok, input_s, SEQUENCE_TEST_INPUT_SEQUENCES[5]);
        let (tok, recv_data) = recv(tok, output_r);
        assert_eq(SEQUENCE_TEST_EXPECTED_SEQUENCE_RESULTS[6], recv_data);

        let tok = send(tok, input_s, SEQUENCE_TEST_INPUT_SEQUENCES[6]);
        let tok = send(tok, input_s, SEQUENCE_TEST_INPUT_SEQUENCES[7]);
        let (tok, recv_data) = recv(tok, output_r);
        assert_eq(SEQUENCE_TEST_EXPECTED_SEQUENCE_RESULTS[7], recv_data);

        let tok = send(tok, input_s, SEQUENCE_TEST_INPUT_SEQUENCES[8]);
        let tok = send(tok, input_s, SEQUENCE_TEST_INPUT_SEQUENCES[9]);
        let (tok, recv_data) = recv(tok, output_r);
        assert_eq(SEQUENCE_TEST_EXPECTED_SEQUENCE_RESULTS[8], recv_data);
        let (tok, recv_data) = recv(tok, output_r);
        assert_eq(SEQUENCE_TEST_EXPECTED_SEQUENCE_RESULTS[9], recv_data);

        let tok = send(tok, input_s, SEQUENCE_TEST_INPUT_SEQUENCES[10]);
        let (tok, recv_data) = recv(tok, output_r);
        assert_eq(SEQUENCE_TEST_EXPECTED_SEQUENCE_RESULTS[10], recv_data);

        // Print RAM content
        let tok = send(tok, print_start_s, ());
        let (tok, _) = recv(tok, print_finish_r);
        send(tok, terminator, true);
    }
}
