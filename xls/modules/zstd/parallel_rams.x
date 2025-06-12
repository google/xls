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

// this file contains implementation of parallel RAMs handling

import std;
import xls.modules.zstd.common as common;
import xls.examples.ram;

type BlockData = common::BlockData;
type SequenceExecutorMessageType = common::SequenceExecutorMessageType;
type SequenceExecutorPacket = common::SequenceExecutorPacket;
type CopyOrMatchContent = common::CopyOrMatchContent;
type CopyOrMatchLength = common::CopyOrMatchLength;
type ZstdDecodedPacket = common::ZstdDecodedPacket;
type BlockPacketLength = common::BlockPacketLength;
pub type Offset = common::Offset;

// Configurable RAM parameters, RAM_NUM has to be a power of 2
pub const RAM_NUM = u32:8;

// Constants calculated from RAM parameters
pub const RAM_NUM_WIDTH = std::clog2(RAM_NUM);

pub type RamNumber = bits[RAM_NUM_WIDTH];
pub type RamReadStart = bits[RAM_NUM_WIDTH];
pub type RamReadLen = bits[std::clog2(RAM_NUM + u32:1)];

pub fn ram_size<RAM_DATA_WIDTH: u32 = {common::SYMBOL_WIDTH}>(hb_size_kb: u32) -> u32 {
    (hb_size_kb * u32:1024 * u32:8) / RAM_DATA_WIDTH / RAM_NUM
}

pub fn ram_addr_width<RAM_DATA_WIDTH: u32 = {common::SYMBOL_WIDTH}>(hb_size_kb: u32) -> u32 {
    std::clog2(ram_size<RAM_DATA_WIDTH>(hb_size_kb))
}

// RAM related constants common for tests
const TEST_HISTORY_BUFFER_SIZE_KB = u32:1;
const TEST_RAM_SIZE = ram_size(TEST_HISTORY_BUFFER_SIZE_KB);
const TEST_RAM_ADDR_WIDTH = ram_addr_width(TEST_HISTORY_BUFFER_SIZE_KB);
const TEST_RAM_DATA_WIDTH = common::SYMBOL_WIDTH;
const TEST_RAM_WORD_PARTITION_SIZE = TEST_RAM_DATA_WIDTH;
const TEST_RAM_NUM_PARTITIONS = ram::num_partitions(TEST_RAM_WORD_PARTITION_SIZE, TEST_RAM_DATA_WIDTH);
const TEST_RAM_INITIALIZED = true;
const TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR = ram::SimultaneousReadWriteBehavior::READ_BEFORE_WRITE;
const TEST_RAM_REQ_MASK_ALL = std::unsigned_max_value<TEST_RAM_NUM_PARTITIONS>();
const TEST_RAM_REQ_MASK_NONE = bits[TEST_RAM_NUM_PARTITIONS]:0;

type TestRamAddr = bits[TEST_RAM_ADDR_WIDTH];
type TestWriteReq = ram::WriteReq<TEST_RAM_ADDR_WIDTH, TEST_RAM_DATA_WIDTH, TEST_RAM_NUM_PARTITIONS>;
type TestWriteResp = ram::WriteResp<TEST_RAM_ADDR_WIDTH>;
type TestReadReq = ram::ReadReq<TEST_RAM_ADDR_WIDTH, TEST_RAM_NUM_PARTITIONS>;
type TestReadResp = ram::ReadResp<TEST_RAM_DATA_WIDTH>;

pub struct HistoryBufferPtr<RAM_ADDR_WIDTH: u32> { number: RamNumber, addr: bits[RAM_ADDR_WIDTH] }

pub fn hb_ptr_from_offset_back<
    HISTORY_BUFFER_SIZE_KB: u32,
    RAM_SIZE: u32 = {ram_size(HISTORY_BUFFER_SIZE_KB)},
    RAM_ADDR_WIDTH: u32 = {ram_addr_width(HISTORY_BUFFER_SIZE_KB)}
>(
    ptr: HistoryBufferPtr<RAM_ADDR_WIDTH>, offset: Offset) -> HistoryBufferPtr<RAM_ADDR_WIDTH> {

    const_assert!(common::OFFSET_WIDTH < u32:32);
    type RamAddr = bits[RAM_ADDR_WIDTH];

    let buff_change = offset as RamNumber;
    let max_row_span = (offset >> RAM_NUM_WIDTH) as RamAddr;
    let addr_change = if ptr.number >= buff_change {
        (max_row_span)
    } else {
        (max_row_span + RamAddr:1)
    };
    let number = ptr.number - buff_change;
    let addr = ptr.addr - addr_change;
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

pub fn hb_ptr_from_offset_forw<
    HISTORY_BUFFER_SIZE_KB: u32,
    RAM_SIZE: u32 = {ram_size(HISTORY_BUFFER_SIZE_KB)},
    RAM_ADDR_WIDTH: u32 = {ram_addr_width(HISTORY_BUFFER_SIZE_KB)}
>(ptr: HistoryBufferPtr<RAM_ADDR_WIDTH>, offset: Offset) -> HistoryBufferPtr<RAM_ADDR_WIDTH> {

    type RamAddr = bits[RAM_ADDR_WIDTH];
    const MAX_ADDR = (RAM_SIZE - u32:1) as RamAddr;

    let buff_change = std::mod_pow2(offset as u32, RAM_NUM) as RamNumber;
    let rounded_offset = std::round_up_to_nearest_pow2_unsigned(offset as u32 + u32:1, RAM_NUM as u32);
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

fn literal_packet_to_single_write_req<
    HISTORY_BUFFER_SIZE_KB: u32,
    RAM_ADDR_WIDTH: u32 = {ram_addr_width(HISTORY_BUFFER_SIZE_KB)},
    RAM_DATA_WIDTH: u32 = {common::SYMBOL_WIDTH},
    RAM_WORD_PARTITION_SIZE: u32 = {RAM_DATA_WIDTH},
    RAM_NUM_PARTITIONS: u32 = {ram::num_partitions(RAM_WORD_PARTITION_SIZE, RAM_DATA_WIDTH)}
>(ptr: HistoryBufferPtr<RAM_ADDR_WIDTH>, literal: SequenceExecutorPacket<RAM_DATA_WIDTH>, number: RamNumber)
    -> ram::WriteReq<RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS> {
    type RamData = uN[RAM_DATA_WIDTH];
    type WriteReq = ram::WriteReq<RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>;

    let offset = std::mod_pow2(RAM_NUM - ptr.number as u32 + number as u32, RAM_NUM) as Offset;
    let we = literal.length >= offset as CopyOrMatchLength + CopyOrMatchLength:1;
    let hb = hb_ptr_from_offset_forw<HISTORY_BUFFER_SIZE_KB>(ptr, offset);

    if (we) {
        WriteReq {
            data: literal.content[offset as u32 * RAM_DATA_WIDTH+:RamData] as RamData,
            addr: hb.addr,
            mask: std::unsigned_max_value<RAM_NUM_PARTITIONS>()
        }
    } else {
        zero!<WriteReq>()
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
    type RamData = uN[TEST_RAM_DATA_WIDTH];

    let ptr = HistoryBufferPtr { number: RamNumber:7, addr: TestRamAddr:2 };
    let literals = SequenceExecutorPacket<TEST_RAM_DATA_WIDTH> {
        msg_type: SequenceExecutorMessageType::LITERAL,
        length: CopyOrMatchLength:7,
        content: CopyOrMatchContent:0x77_6655_4433_2211,
        last: false
    };
    assert_eq(
        literal_packet_to_single_write_req<TEST_HISTORY_BUFFER_SIZE_KB>(ptr, literals, RamNumber:0),
        TestWriteReq { data: RamData:0x22, addr: TestRamAddr:0x3, mask: TEST_RAM_REQ_MASK_ALL });
    assert_eq(
        literal_packet_to_single_write_req<TEST_HISTORY_BUFFER_SIZE_KB>(ptr, literals, RamNumber:3),
        TestWriteReq { data: RamData:0x55, addr: TestRamAddr:0x3, mask: TEST_RAM_REQ_MASK_ALL });
    assert_eq(
        literal_packet_to_single_write_req<TEST_HISTORY_BUFFER_SIZE_KB>(ptr, literals, RamNumber:6),
        zero!<TestWriteReq>());
}

pub fn literal_packet_to_write_reqs<
    HISTORY_BUFFER_SIZE_KB: u32,
    RAM_ADDR_WIDTH: u32 = {ram_addr_width(HISTORY_BUFFER_SIZE_KB)},
    RAM_DATA_WIDTH: u32 = {common::SYMBOL_WIDTH},
    RAM_WORD_PARTITION_SIZE: u32 = {RAM_DATA_WIDTH},
    RAM_NUM_PARTITIONS: u32 = {ram::num_partitions(RAM_WORD_PARTITION_SIZE, RAM_DATA_WIDTH)}
>(
    ptr: HistoryBufferPtr<RAM_ADDR_WIDTH>, literal: SequenceExecutorPacket<RAM_DATA_WIDTH>
) -> (ram::WriteReq<RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>[RAM_NUM], HistoryBufferPtr) {
    type WriteReq = ram::WriteReq<RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>;
    let result = WriteReq[RAM_NUM]:[
        literal_packet_to_single_write_req<HISTORY_BUFFER_SIZE_KB, RAM_ADDR_WIDTH, RAM_DATA_WIDTH>(ptr, literal, RamNumber:0),
        literal_packet_to_single_write_req<HISTORY_BUFFER_SIZE_KB, RAM_ADDR_WIDTH, RAM_DATA_WIDTH>(ptr, literal, RamNumber:1),
        literal_packet_to_single_write_req<HISTORY_BUFFER_SIZE_KB, RAM_ADDR_WIDTH, RAM_DATA_WIDTH>(ptr, literal, RamNumber:2),
        literal_packet_to_single_write_req<HISTORY_BUFFER_SIZE_KB, RAM_ADDR_WIDTH, RAM_DATA_WIDTH>(ptr, literal, RamNumber:3),
        literal_packet_to_single_write_req<HISTORY_BUFFER_SIZE_KB, RAM_ADDR_WIDTH, RAM_DATA_WIDTH>(ptr, literal, RamNumber:4),
        literal_packet_to_single_write_req<HISTORY_BUFFER_SIZE_KB, RAM_ADDR_WIDTH, RAM_DATA_WIDTH>(ptr, literal, RamNumber:5),
        literal_packet_to_single_write_req<HISTORY_BUFFER_SIZE_KB, RAM_ADDR_WIDTH, RAM_DATA_WIDTH>(ptr, literal, RamNumber:6),
        literal_packet_to_single_write_req<HISTORY_BUFFER_SIZE_KB, RAM_ADDR_WIDTH, RAM_DATA_WIDTH>(ptr, literal, RamNumber:7),
    ];

    let ptr_offset = literal.length;
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
    type RamData = uN[TEST_RAM_DATA_WIDTH];

    let ptr = HistoryBufferPtr { number: RamNumber:7, addr: TestRamAddr:0x2 };
    let literals = SequenceExecutorPacket<TEST_RAM_DATA_WIDTH> {
        msg_type: SequenceExecutorMessageType::LITERAL,
        content: CopyOrMatchContent:0x11,
        length: CopyOrMatchLength:1,
        last: false
    };
    assert_eq(
        literal_packet_to_write_reqs<TEST_HISTORY_BUFFER_SIZE_KB>(ptr, literals),
        (
            TestWriteReq[RAM_NUM]:[
                zero!<TestWriteReq>(), zero!<TestWriteReq>(), zero!<TestWriteReq>(),
                zero!<TestWriteReq>(), zero!<TestWriteReq>(), zero!<TestWriteReq>(),
                zero!<TestWriteReq>(),
                TestWriteReq { data: RamData:0x11, addr: TestRamAddr:0x2, mask: TEST_RAM_REQ_MASK_ALL },
            ], HistoryBufferPtr { number: RamNumber:0, addr: TestRamAddr:0x3 },
        ));

    // BEFORE:                         AFTER:
    //     7  6  5  4  3  2  1  0          7  6  5  4  3  2  1  0
    // 1 |  |  |  |  |  |  |  |  |     1 |  |  |  |  |  |  |  |  |
    // 2 | o|  |  |  |  |  |  |  |     2 |11|  |  |  |  |  |  |  |
    // 3 |  |  |  |  |  |  |  |  |     3 | o|88|77|66|55|44|33|22|
    // 4 |  |  |  |  |  |  |  |  |     4 |  |  |  |  |  |  |  |  |

    let ptr = HistoryBufferPtr { number: RamNumber:7, addr: TestRamAddr:2 };
    let literals = SequenceExecutorPacket<TEST_RAM_DATA_WIDTH> {
        msg_type: SequenceExecutorMessageType::LITERAL,
        content: CopyOrMatchContent:0x8877_6655_4433_2211,
        length: CopyOrMatchLength:8,
        last: false
    };
    assert_eq(
        literal_packet_to_write_reqs<TEST_HISTORY_BUFFER_SIZE_KB>(ptr, literals),
        (
            TestWriteReq[RAM_NUM]:[
                TestWriteReq { data: RamData:0x22, addr: TestRamAddr:0x3, mask: TEST_RAM_REQ_MASK_ALL },
                TestWriteReq { data: RamData:0x33, addr: TestRamAddr:0x3, mask: TEST_RAM_REQ_MASK_ALL },
                TestWriteReq { data: RamData:0x44, addr: TestRamAddr:0x3, mask: TEST_RAM_REQ_MASK_ALL },
                TestWriteReq { data: RamData:0x55, addr: TestRamAddr:0x3, mask: TEST_RAM_REQ_MASK_ALL },
                TestWriteReq { data: RamData:0x66, addr: TestRamAddr:0x3, mask: TEST_RAM_REQ_MASK_ALL },
                TestWriteReq { data: RamData:0x77, addr: TestRamAddr:0x3, mask: TEST_RAM_REQ_MASK_ALL },
                TestWriteReq { data: RamData:0x88, addr: TestRamAddr:0x3, mask: TEST_RAM_REQ_MASK_ALL },
                TestWriteReq { data: RamData:0x11, addr: TestRamAddr:0x2, mask: TEST_RAM_REQ_MASK_ALL },
            ], HistoryBufferPtr { number: RamNumber:7, addr: TestRamAddr:3 },
        ));
}

fn max_hb_ptr_for_sequence_packet<
    HISTORY_BUFFER_SIZE_KB: u32,
    RAM_ADDR_WIDTH: u32 = {ram_addr_width(HISTORY_BUFFER_SIZE_KB)},
    RAM_DATA_WIDTH: u32 = {common::SYMBOL_WIDTH},
> (
    ptr: HistoryBufferPtr<RAM_ADDR_WIDTH>, seq: SequenceExecutorPacket<RAM_DATA_WIDTH>
) -> HistoryBufferPtr<RAM_ADDR_WIDTH> {
    hb_ptr_from_offset_back<HISTORY_BUFFER_SIZE_KB>(ptr, seq.content as Offset)
}

fn sequence_packet_to_single_read_req<
    HISTORY_BUFFER_SIZE_KB: u32,
    RAM_ADDR_WIDTH: u32 = {ram_addr_width(HISTORY_BUFFER_SIZE_KB)},
    RAM_DATA_WIDTH: u32 = {common::SYMBOL_WIDTH},
    RAM_WORD_PARTITION_SIZE: u32 = {RAM_DATA_WIDTH},
    RAM_NUM_PARTITIONS: u32 = {ram::num_partitions(RAM_WORD_PARTITION_SIZE, RAM_DATA_WIDTH)}
> (
    ptr: HistoryBufferPtr<RAM_ADDR_WIDTH>, max_ptr: HistoryBufferPtr<RAM_ADDR_WIDTH>,
    seq: SequenceExecutorPacket<RAM_DATA_WIDTH>, number: RamNumber
) -> ram::ReadReq<RAM_ADDR_WIDTH, RAM_NUM_PARTITIONS> {
    const RAM_REQ_MASK_ALL = bits[RAM_NUM_PARTITIONS]:1;
    type ReadReq = ram::ReadReq<RAM_ADDR_WIDTH, RAM_NUM_PARTITIONS>;

    let offset_change = if max_ptr.number > number {
        RAM_NUM as RamNumber - max_ptr.number + number
    } else {
        number - max_ptr.number
    };
    let offset = (seq.content as Offset - offset_change as Offset) as Offset;
    let re = (offset_change as CopyOrMatchLength) < seq.length;
    let hb = hb_ptr_from_offset_back<HISTORY_BUFFER_SIZE_KB>(ptr, offset);

    if (re) {
        ReadReq { addr: hb.addr, mask: RAM_REQ_MASK_ALL }
    } else {
        zero!<ReadReq>()
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
    let sequence = SequenceExecutorPacket<TEST_RAM_DATA_WIDTH> {
        msg_type: SequenceExecutorMessageType::SEQUENCE,
        content: CopyOrMatchContent:11,
        length: CopyOrMatchLength:4,
        last: false
    };
    let max_ptr = max_hb_ptr_for_sequence_packet<
        TEST_HISTORY_BUFFER_SIZE_KB, TEST_RAM_ADDR_WIDTH, TEST_RAM_DATA_WIDTH
    >(ptr, sequence);

    assert_eq(
        sequence_packet_to_single_read_req<TEST_HISTORY_BUFFER_SIZE_KB>(
            ptr, max_ptr, sequence, RamNumber:0),
        TestReadReq { addr: TestRamAddr:0x2, mask: TEST_RAM_REQ_MASK_ALL });

    assert_eq(
        sequence_packet_to_single_read_req<TEST_HISTORY_BUFFER_SIZE_KB>(
            ptr, max_ptr, sequence, RamNumber:1),
        TestReadReq { addr: TestRamAddr:0x2, mask: TEST_RAM_REQ_MASK_ALL });

    assert_eq(
        sequence_packet_to_single_read_req<TEST_HISTORY_BUFFER_SIZE_KB>(
            ptr, max_ptr, sequence, RamNumber:2), zero!<TestReadReq>());

    assert_eq(
        sequence_packet_to_single_read_req<TEST_HISTORY_BUFFER_SIZE_KB>(
            ptr, max_ptr, sequence, RamNumber:7),
        TestReadReq { addr: TestRamAddr:0x1, mask: TEST_RAM_REQ_MASK_ALL });

    assert_eq(
        sequence_packet_to_single_read_req<TEST_HISTORY_BUFFER_SIZE_KB>(
            ptr, max_ptr, sequence, RamNumber:6),
        TestReadReq { addr: TestRamAddr:0x1, mask: TEST_RAM_REQ_MASK_ALL });
}

pub fn sequence_packet_to_read_reqs<
    HISTORY_BUFFER_SIZE_KB: u32,
    RAM_ADDR_WIDTH: u32 = {ram_addr_width(HISTORY_BUFFER_SIZE_KB)},
    RAM_DATA_WIDTH: u32 = {common::SYMBOL_WIDTH},
    RAM_WORD_PARTITION_SIZE: u32 = {RAM_DATA_WIDTH},
    RAM_NUM_PARTITIONS: u32 = {ram::num_partitions(RAM_WORD_PARTITION_SIZE, RAM_DATA_WIDTH)}
> (
    ptr: HistoryBufferPtr<RAM_ADDR_WIDTH>, seq: SequenceExecutorPacket<RAM_DATA_WIDTH>, hb_len: uN[RAM_ADDR_WIDTH + RAM_NUM_WIDTH]
) -> (ram::ReadReq<RAM_ADDR_WIDTH, RAM_NUM_PARTITIONS>[RAM_NUM], RamReadStart, RamReadLen, SequenceExecutorPacket<RAM_DATA_WIDTH>, bool) {
    type ReadReq = ram::ReadReq<RAM_ADDR_WIDTH, RAM_NUM_PARTITIONS>;
    type Packet = SequenceExecutorPacket<RAM_DATA_WIDTH>;

    let max_len = std::min(seq.length as u32, std::min(RAM_NUM, std::min(hb_len as u32, seq.content as u32)));

    let (curr_seq, next_seq, next_seq_valid) = if seq.length > max_len as CopyOrMatchLength {
        (
            seq,
            Packet {
                msg_type: SequenceExecutorMessageType::SEQUENCE,
                length:  seq.length - max_len as CopyOrMatchLength,
                content: seq.content,
                last: false,
            },
            true,
        )
    } else if seq.length > seq.content as CopyOrMatchLength {
        (
            Packet {
                msg_type: SequenceExecutorMessageType::SEQUENCE,
                length: max_len as CopyOrMatchLength,
                content: seq.content,
                last: false,
            },
            Packet {
                msg_type: SequenceExecutorMessageType::SEQUENCE,
                length: seq.length - (max_len as CopyOrMatchLength),
                content: seq.content + (max_len as uN[RAM_DATA_WIDTH * u32:8]),
                last: seq.last
            },
            true,
        )
    } else {
        (seq, zero!<Packet>(), false)
    };

    let max_ptr = max_hb_ptr_for_sequence_packet<HISTORY_BUFFER_SIZE_KB, RAM_ADDR_WIDTH, RAM_DATA_WIDTH>(ptr, curr_seq);
    let req0 = sequence_packet_to_single_read_req<HISTORY_BUFFER_SIZE_KB, RAM_ADDR_WIDTH, RAM_DATA_WIDTH>(ptr, max_ptr, curr_seq, RamNumber:0);
    let req1 = sequence_packet_to_single_read_req<HISTORY_BUFFER_SIZE_KB, RAM_ADDR_WIDTH, RAM_DATA_WIDTH>(ptr, max_ptr, curr_seq, RamNumber:1);
    let req2 = sequence_packet_to_single_read_req<HISTORY_BUFFER_SIZE_KB, RAM_ADDR_WIDTH, RAM_DATA_WIDTH>(ptr, max_ptr, curr_seq, RamNumber:2);
    let req3 = sequence_packet_to_single_read_req<HISTORY_BUFFER_SIZE_KB, RAM_ADDR_WIDTH, RAM_DATA_WIDTH>(ptr, max_ptr, curr_seq, RamNumber:3);
    let req4 = sequence_packet_to_single_read_req<HISTORY_BUFFER_SIZE_KB, RAM_ADDR_WIDTH, RAM_DATA_WIDTH>(ptr, max_ptr, curr_seq, RamNumber:4);
    let req5 = sequence_packet_to_single_read_req<HISTORY_BUFFER_SIZE_KB, RAM_ADDR_WIDTH, RAM_DATA_WIDTH>(ptr, max_ptr, curr_seq, RamNumber:5);
    let req6 = sequence_packet_to_single_read_req<HISTORY_BUFFER_SIZE_KB, RAM_ADDR_WIDTH, RAM_DATA_WIDTH>(ptr, max_ptr, curr_seq, RamNumber:6);
    let req7 = sequence_packet_to_single_read_req<HISTORY_BUFFER_SIZE_KB, RAM_ADDR_WIDTH, RAM_DATA_WIDTH>(ptr, max_ptr, curr_seq, RamNumber:7);

    let reqs = ReadReq[RAM_NUM]:[req0, req1, req2, req3, req4, req5, req6, req7];

    (reqs, max_ptr.number, max_len as RamReadLen, next_seq, next_seq_valid)
}

#[test]
fn test_sequence_packet_to_read_reqs() {
    // BEFORE:                         AFTER:
    //     7  6  5  4  3  2  1  0          7  6  5  4  3  2  1  0
    // 1 | x| x|  |  |  |  |  |  |     1 |  |  |  |  |  |  |  |  |
    // 2 |  |  |  |  |  |  | x| x|     2 |  |  |  |  |  |  |  |  |
    // 3 |  |  |  |  |  |  | o|  |     3 |  |  |  |  |  |  | o|  |
    // 4 |  |  |  |  |  |  |  |  |     4 |  |  |  |  |  |  |  |  |
    type Packet = SequenceExecutorPacket<TEST_RAM_DATA_WIDTH>;
    type HistoryBufferLength = uN[TEST_RAM_ADDR_WIDTH + RAM_NUM_WIDTH];

    let ptr = HistoryBufferPtr { number: RamNumber:1, addr: TestRamAddr:0x3 };
    let sequence = SequenceExecutorPacket<TEST_RAM_DATA_WIDTH> {
        msg_type: SequenceExecutorMessageType::SEQUENCE,
        content: CopyOrMatchContent:11,
        length: CopyOrMatchLength:4,
        last: false
    };
    let result = sequence_packet_to_read_reqs<TEST_HISTORY_BUFFER_SIZE_KB, TEST_RAM_ADDR_WIDTH, TEST_RAM_DATA_WIDTH>(
        ptr, sequence, HistoryBufferLength:20);
    let expected = (
        TestReadReq[RAM_NUM]:[
            TestReadReq { addr: TestRamAddr:0x2, mask: TEST_RAM_REQ_MASK_ALL },
            TestReadReq { addr: TestRamAddr:0x2, mask: TEST_RAM_REQ_MASK_ALL }, zero!<TestReadReq>(),
            zero!<TestReadReq>(), zero!<TestReadReq>(), zero!<TestReadReq>(),
            TestReadReq { addr: TestRamAddr:0x1, mask: TEST_RAM_REQ_MASK_ALL },
            TestReadReq { addr: TestRamAddr:0x1, mask: TEST_RAM_REQ_MASK_ALL },
        ],
        RamReadStart:6,
        RamReadLen:4,
        zero!<Packet>(), false,
    );
    assert_eq(result, expected);

    // BEFORE:                         AFTER:
    //     7  6  5  4  3  2  1  0          7  6  5  4  3  2  1  0
    // 1 |  |  |  |  |  |  |  |  |     1 |  |  |  |  |  |  |  |  |
    // 2 | x| x|  |  |  |  |  |  |     2 |  |  |  |  |  |  |  |  |
    // 3 |  |  | x| x| x| x| x| x|     3 |  | x|  |  |  |  |  |  |
    // 4 |  |  |  |  |  |  |  | o|     4 |  |  |  |  |  |  |  | o|

    let ptr = HistoryBufferPtr { number: RamNumber:0, addr: TestRamAddr:0x4 };
    let sequence = Packet {
        msg_type: SequenceExecutorMessageType::SEQUENCE,
        content: CopyOrMatchContent:10,
        length: CopyOrMatchLength:9,
        last: false
    };
    let result = sequence_packet_to_read_reqs<TEST_HISTORY_BUFFER_SIZE_KB>(
        ptr, sequence, HistoryBufferLength:20);
    let expected = (
        TestReadReq[RAM_NUM]:[
            TestReadReq { addr: TestRamAddr:0x3, mask: TEST_RAM_REQ_MASK_ALL },
            TestReadReq { addr: TestRamAddr:0x3, mask: TEST_RAM_REQ_MASK_ALL },
            TestReadReq { addr: TestRamAddr:0x3, mask: TEST_RAM_REQ_MASK_ALL },
            TestReadReq { addr: TestRamAddr:0x3, mask: TEST_RAM_REQ_MASK_ALL },
            TestReadReq { addr: TestRamAddr:0x3, mask: TEST_RAM_REQ_MASK_ALL },
            TestReadReq { addr: TestRamAddr:0x3, mask: TEST_RAM_REQ_MASK_ALL },
            TestReadReq { addr: TestRamAddr:0x2, mask: TEST_RAM_REQ_MASK_ALL },
            TestReadReq { addr: TestRamAddr:0x2, mask: TEST_RAM_REQ_MASK_ALL },
        ],
        RamReadStart:6,
        RamReadLen:8,
        Packet {
            msg_type: SequenceExecutorMessageType::SEQUENCE,
            content: CopyOrMatchContent:10,
            length: CopyOrMatchLength:1,
            last: false
        }, true,
    );
    assert_eq(result, expected);
}

pub struct RamWrRespHandlerData<RAM_ADDR_WIDTH: u32> {
    resp: bool[RAM_NUM],
    ptr: HistoryBufferPtr<RAM_ADDR_WIDTH>,
}

pub struct RamWrRespHandlerResp<RAM_ADDR_WIDTH: u32> {
    length: uN[std::clog2(RAM_NUM + u32:1)],
    ptr: HistoryBufferPtr<RAM_ADDR_WIDTH>,
}

pub fn create_ram_wr_data<RAM_ADDR_WIDTH: u32, RAM_DATA_WIDTH: u32, RAM_NUM_PARTITIONS: u32>
    (reqs: ram::WriteReq<RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>[RAM_NUM], ptr: HistoryBufferPtr) -> (bool, RamWrRespHandlerData) {
    const RAM_REQ_MASK_NONE = bits[RAM_NUM_PARTITIONS]:0;

    let (do_write, resp) = for (i, (do_write, resp)): (u32, (bool, bool[RAM_NUM])) in range(u32:0, RAM_NUM) {
        (
            do_write || reqs[i].mask,
            update(resp, i, reqs[i].mask != RAM_REQ_MASK_NONE)
        )
    }((false, zero!<bool[RAM_NUM]>()));

    (do_write, RamWrRespHandlerData { resp, ptr })
}

pub proc RamWrRespHandler<RAM_ADDR_WIDTH: u32, RAM_DATA_WIDTH: u32 = {u32:8}> {
    input_r: chan<RamWrRespHandlerData> in;
    output_s: chan<RamWrRespHandlerResp> out;
    wr_resp_m0_r: chan<ram::WriteResp> in;
    wr_resp_m1_r: chan<ram::WriteResp> in;
    wr_resp_m2_r: chan<ram::WriteResp> in;
    wr_resp_m3_r: chan<ram::WriteResp> in;
    wr_resp_m4_r: chan<ram::WriteResp> in;
    wr_resp_m5_r: chan<ram::WriteResp> in;
    wr_resp_m6_r: chan<ram::WriteResp> in;
    wr_resp_m7_r: chan<ram::WriteResp> in;

    config(input_r: chan<RamWrRespHandlerData<RAM_ADDR_WIDTH>> in,
           output_s: chan<RamWrRespHandlerResp<RAM_ADDR_WIDTH>> out,
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

        let tok3 = send(tok2, output_s, RamWrRespHandlerResp {
            length: std::popcount(std::convert_to_bits_msb0(input.resp)) as uN[std::clog2(RAM_NUM + u32:1)],
            ptr: input.ptr
        });
    }
}

pub struct RamRdRespHandlerData {
    resp: bool[RAM_NUM],
    read_start: RamReadStart,
    read_len: RamReadLen,
    last: bool
}

pub fn create_ram_rd_data<RAM_ADDR_WIDTH: u32, RAM_DATA_WIDTH: u32, RAM_NUM_PARTITIONS: u32>
    (reqs: ram::ReadReq<RAM_ADDR_WIDTH, RAM_NUM_PARTITIONS>[RAM_NUM], read_start: RamReadStart, read_len: RamReadLen, last: bool, next_packet_valid: bool) -> (bool, RamRdRespHandlerData) {
    const RAM_REQ_MASK_NONE = bits[RAM_NUM_PARTITIONS]:0;

    let (do_read, resp) = for (i, (do_read, resp)): (u32, (bool, bool[RAM_NUM])) in range(u32:0, RAM_NUM) {
        (
            do_read || reqs[i].mask,
            update(resp, i, reqs[i].mask != RAM_REQ_MASK_NONE)
        )
    }((false, zero!<bool[RAM_NUM]>()));

    let last = (!next_packet_valid) && last;
    (do_read, RamRdRespHandlerData { resp, read_start, read_len, last })
}

pub proc RamRdRespHandler<RAM_DATA_WIDTH: u32 = {u32:8}> {
    input_r: chan<RamRdRespHandlerData> in;
    output_s: chan<SequenceExecutorPacket<RAM_DATA_WIDTH>> out;
    rd_resp_m0_r: chan<ram::ReadResp<RAM_DATA_WIDTH>> in;
    rd_resp_m1_r: chan<ram::ReadResp<RAM_DATA_WIDTH>> in;
    rd_resp_m2_r: chan<ram::ReadResp<RAM_DATA_WIDTH>> in;
    rd_resp_m3_r: chan<ram::ReadResp<RAM_DATA_WIDTH>> in;
    rd_resp_m4_r: chan<ram::ReadResp<RAM_DATA_WIDTH>> in;
    rd_resp_m5_r: chan<ram::ReadResp<RAM_DATA_WIDTH>> in;
    rd_resp_m6_r: chan<ram::ReadResp<RAM_DATA_WIDTH>> in;
    rd_resp_m7_r: chan<ram::ReadResp<RAM_DATA_WIDTH>> in;

    config(input_r: chan<RamRdRespHandlerData> in, output_s: chan<SequenceExecutorPacket<RAM_DATA_WIDTH>> out,
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
        type Content = uN[RAM_DATA_WIDTH * u32:8];

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

        let resp_data = [
            resp_0.data, resp_1.data, resp_2.data, resp_3.data,
            resp_4.data, resp_5.data, resp_6.data, resp_7.data
        ];

        let content = (
            resp_data[input.read_start + u3:7] ++
            resp_data[input.read_start + u3:6] ++
            resp_data[input.read_start + u3:5] ++
            resp_data[input.read_start + u3:4] ++
            resp_data[input.read_start + u3:3] ++
            resp_data[input.read_start + u3:2] ++
            resp_data[input.read_start + u3:1] ++
            resp_data[input.read_start + u3:0]
        );

        let output_data = SequenceExecutorPacket<RAM_DATA_WIDTH> {
            msg_type: SequenceExecutorMessageType::LITERAL,
            length: input.read_len as CopyOrMatchLength,
            content: content as Content,
            last: input.last,
        };

        let tok3 = send(tok2, output_s, output_data);
    }
}
