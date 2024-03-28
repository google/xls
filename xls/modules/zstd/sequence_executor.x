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

import std;
import xls.modules.zstd.common as common;
import xls.modules.zstd.ram_printer as ram_printer;
import xls.examples.ram;

type BlockData = common::BlockData;
type BlockType = common::BlockType;
type SequenceExecutorMessageType = common::SequenceExecutorMessageType;
type SequenceExecutorPacket = common::SequenceExecutorPacket;
type CopyOrMatchContent = common::CopyOrMatchContent;
type CopyOrMatchLength = common::CopyOrMatchLength;
type ZstdDecodedPacket = common::ZstdDecodedPacket;
type BlockPacketLength = common::BlockPacketLength;
type Offset = common::Offset;
type Length = common::Length;

const DATA_WIDTH = common::DATA_WIDTH;

// const HISTORY_BUFFER_SIZE_KB = common::HISTORY_BUFFER_SIZE_KB;
const HISTORY_BUFFER_SIZE_KB = u32:1;

// RAM parameters
const RAM_DATA_WIDTH = common::SYMBOL_WIDTH;
const RAM_NUM = u32:8;
const RAM_WORD_PARTITION_SIZE = u32:0;
const RAM_SIZE = (HISTORY_BUFFER_SIZE_KB * u32:1024 * u32:8) / RAM_DATA_WIDTH / RAM_NUM;
const RAM_NUM_PARTITIONS = ram::num_partitions(RAM_WORD_PARTITION_SIZE, RAM_DATA_WIDTH);
const RAM_ADDR_WIDTH = std::clog2(RAM_SIZE);
const RAM_NUM_WIDTH = std::clog2(RAM_NUM);

type RamNumber = bits[RAM_NUM_WIDTH];
type RamData = bits[RAM_DATA_WIDTH];
type RamAddr = bits[RAM_ADDR_WIDTH];
pub type RWRamReq = ram::RWRamReq<RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>;
pub type RWRamResp = ram::RWRamResp<RAM_DATA_WIDTH>;

const ZERO_RWRAM_REQ = zero!<RWRamReq>();
const ZERO_RWRAM_RESP = zero!<RWRamResp>();
const RWRAM_WRITE_REQ = ram::RWRamReq { we: true, ..ZERO_RWRAM_REQ };
const RWRAM_READ_REQ = ram::RWRamReq { re: true, ..ZERO_RWRAM_REQ };

type HistoryBufferNumber = bits[std::clog2(RAM_NUM)];
type HistoryBufferAddr = bits[std::clog2(RAM_ADDR_WIDTH)];

struct HistoryBufferPtr { number: RamNumber, addr: RamAddr }

enum SequenceExecutorStatus : u2 {
    IDLE = 0,
    LITERAL_EXEC = 1,
    SEQUENCE_EXEC = 2,
}

struct SequenceExecutorState {
    status: SequenceExecutorStatus,
    packet: SequenceExecutorPacket,
    packet_valid: bool,
    hyp_ptr: HistoryBufferPtr,
    real_ptr: HistoryBufferPtr,
}

const ZERO_HISTORY_BUFFER_STATE = zero!<SequenceExecutorState>();

pub struct LiteralPacket { length: CopyOrMatchLength, data: CopyOrMatchContent }

fn ConvertToLiteralPacket(packet: SequenceExecutorPacket) -> LiteralPacket {
    LiteralPacket { length: packet.length, data: packet.content }
}

pub struct SequencePacket { match_length: CopyOrMatchLength, offset_value: CopyOrMatchContent }

fn ConvertToSequencePacket(packet: SequenceExecutorPacket) -> SequencePacket {
    SequencePacket { match_length: packet.length, offset_value: packet.content }
}

fn create_output_for_literal(packet: SequenceExecutorPacket) -> ZstdDecodedPacket {
    ZstdDecodedPacket {
        data: packet.content, length: packet.length as BlockPacketLength, last: packet.last
    }
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

fn print_ram_settings() {
    trace_fmt!("==== Ram Settings ====");
    trace_fmt!(" number of RAMs: {}", RAM_NUM);
    trace_fmt!(" data width: {}", RAM_DATA_WIDTH);
    trace_fmt!(" address width: {}", RAM_ADDR_WIDTH);
    trace_fmt!(" size: {:#x} ({})", RAM_SIZE, RAM_SIZE);
    trace_fmt!(" word partition size: {:#x}", RAM_WORD_PARTITION_SIZE);
    trace_fmt!(" number of partitions: {}", RAM_NUM_PARTITIONS);
}

fn compute_buffer_ptr_from_offset(ptr: HistoryBufferPtr, offset: Offset) -> HistoryBufferPtr {
    const_assert!(common::OFFSET_WIDTH < u32:32);

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
fn test_compute_buffer_ptr_from_offset
    (current: HistoryBufferPtr, literal: LiteralPacket, offset: Offset) {
    assert_eq(
        compute_buffer_ptr_from_offset(
            HistoryBufferPtr { number: RamNumber:4, addr: RamAddr:2 }, Offset:0),
            HistoryBufferPtr { number: RamNumber:4, addr: RamAddr:2 });
    assert_eq(
        compute_buffer_ptr_from_offset(
            HistoryBufferPtr { number: RamNumber:4, addr: RamAddr:2 }, Offset:1),
            HistoryBufferPtr { number: RamNumber:3, addr: RamAddr:2 });
    assert_eq(
        compute_buffer_ptr_from_offset(
            HistoryBufferPtr { number: RamNumber:4, addr: RamAddr:2 }, Offset:2),
            HistoryBufferPtr { number: RamNumber:2, addr: RamAddr:2 });
    assert_eq(
        compute_buffer_ptr_from_offset(
            HistoryBufferPtr { number: RamNumber:4, addr: RamAddr:2 }, Offset:3),
            HistoryBufferPtr { number: RamNumber:1, addr: RamAddr:2 });
    assert_eq(
        compute_buffer_ptr_from_offset(
            HistoryBufferPtr { number: RamNumber:4, addr: RamAddr:2 }, Offset:4),
            HistoryBufferPtr { number: RamNumber:0, addr: RamAddr:2 });
    assert_eq(
        compute_buffer_ptr_from_offset(
            HistoryBufferPtr { number: RamNumber:4, addr: RamAddr:2 }, Offset:5),
            HistoryBufferPtr { number: RamNumber:7, addr: RamAddr:1 });
    assert_eq(
        compute_buffer_ptr_from_offset(
            HistoryBufferPtr { number: RamNumber:4, addr: RamAddr:2 }, Offset:6),
            HistoryBufferPtr { number: RamNumber:6, addr: RamAddr:1 });
    assert_eq(
        compute_buffer_ptr_from_offset(
            HistoryBufferPtr { number: RamNumber:4, addr: RamAddr:2 }, Offset:7),
            HistoryBufferPtr { number: RamNumber:5, addr: RamAddr:1 });
    assert_eq(
        compute_buffer_ptr_from_offset(
            HistoryBufferPtr { number: RamNumber:4, addr: RamAddr:2 }, Offset:8),
            HistoryBufferPtr { number: RamNumber:4, addr: RamAddr:1 });
    assert_eq(
        compute_buffer_ptr_from_offset(
            HistoryBufferPtr { number: RamNumber:4, addr: RamAddr:2 }, Offset:15),
            HistoryBufferPtr { number: RamNumber:5, addr: RamAddr:0 });
    assert_eq(
        compute_buffer_ptr_from_offset(
            HistoryBufferPtr { number: RamNumber:0, addr: RamAddr:0 }, Offset:1),
            HistoryBufferPtr { number: RamNumber:7, addr: (RAM_SIZE - u32:1) as RamAddr });
}

fn compute_buffer_ptr_from_offset_neg(ptr: HistoryBufferPtr, offset: Offset) -> HistoryBufferPtr {
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
fn test_compute_buffer_ptr_from_offset_neg
    (current: HistoryBufferPtr, literal: LiteralPacket, offset: Offset)
    {
    assert_eq(
        compute_buffer_ptr_from_offset_neg(
            HistoryBufferPtr { number: RamNumber:4, addr: RamAddr:2 }, Offset:0),
            HistoryBufferPtr { number: RamNumber:4, addr: RamAddr:2 });
    assert_eq(
        compute_buffer_ptr_from_offset_neg(
            HistoryBufferPtr { number: RamNumber:4, addr: RamAddr:2 }, Offset:1),
            HistoryBufferPtr { number: RamNumber:5, addr: RamAddr:2 });
    assert_eq(
        compute_buffer_ptr_from_offset_neg(
            HistoryBufferPtr { number: RamNumber:4, addr: RamAddr:2 }, Offset:2),
            HistoryBufferPtr { number: RamNumber:6, addr: RamAddr:2 });
    assert_eq(
        compute_buffer_ptr_from_offset_neg(
            HistoryBufferPtr { number: RamNumber:4, addr: RamAddr:2 }, Offset:3),
            HistoryBufferPtr { number: RamNumber:7, addr: RamAddr:2 });
    assert_eq(
        compute_buffer_ptr_from_offset_neg(
            HistoryBufferPtr { number: RamNumber:4, addr: RamAddr:2 }, Offset:4),
            HistoryBufferPtr { number: RamNumber:0, addr: RamAddr:3 });
    assert_eq(
        compute_buffer_ptr_from_offset_neg(
            HistoryBufferPtr { number: RamNumber:4, addr: RamAddr:2 }, Offset:5),
            HistoryBufferPtr { number: RamNumber:1, addr: RamAddr:3 });
    assert_eq(
        compute_buffer_ptr_from_offset_neg(
            HistoryBufferPtr { number: RamNumber:4, addr: RamAddr:2 }, Offset:6),
            HistoryBufferPtr { number: RamNumber:2, addr: RamAddr:3 });
    assert_eq(
        compute_buffer_ptr_from_offset_neg(
            HistoryBufferPtr { number: RamNumber:4, addr: RamAddr:2 }, Offset:7),
            HistoryBufferPtr { number: RamNumber:3, addr: RamAddr:3 });
    assert_eq(
        compute_buffer_ptr_from_offset_neg(
            HistoryBufferPtr { number: RamNumber:4, addr: RamAddr:2 }, Offset:8),
            HistoryBufferPtr { number: RamNumber:4, addr: RamAddr:3 });
    assert_eq(
        compute_buffer_ptr_from_offset_neg(
            HistoryBufferPtr { number: RamNumber:4, addr: RamAddr:2 }, Offset:15),
            HistoryBufferPtr { number: RamNumber:3, addr: RamAddr:4 });
    assert_eq(
        compute_buffer_ptr_from_offset_neg(
            HistoryBufferPtr { number: RamNumber:7, addr: (RAM_SIZE - u32:1) as RamAddr }, Offset:1),
            HistoryBufferPtr { number: RamNumber:0, addr: RamAddr:0 });
}

fn calculate_literal_req_for_ram
    (wptr: HistoryBufferPtr, literal: LiteralPacket, number: RamNumber) -> RWRamReq {
    let offset = std::mod_pow2(RAM_NUM - wptr.number as u32 + number as u32, RAM_NUM) as Offset;
    let we = literal.length >= (offset as u64 + u64:1) << u64:3;
    let data = if we { literal.data[offset as u32 << u32:3+:u8] } else { RamData:0 };
    let hb = compute_buffer_ptr_from_offset_neg(wptr, offset);
    if we { RWRamReq { data, addr: hb.addr, we, ..ZERO_RWRAM_REQ } } else { ZERO_RWRAM_REQ }
}

fn literals_to_ram_req
    (wptr: HistoryBufferPtr, literal: LiteralPacket) -> (RWRamReq[RAM_NUM], HistoryBufferPtr) {
    let result = RWRamReq[RAM_NUM]:[
        calculate_literal_req_for_ram(wptr, literal, RamNumber:0),
        calculate_literal_req_for_ram(wptr, literal, RamNumber:1),
        calculate_literal_req_for_ram(wptr, literal, RamNumber:2),
        calculate_literal_req_for_ram(wptr, literal, RamNumber:3),
        calculate_literal_req_for_ram(wptr, literal, RamNumber:4),
        calculate_literal_req_for_ram(wptr, literal, RamNumber:5),
        calculate_literal_req_for_ram(wptr, literal, RamNumber:6),
        calculate_literal_req_for_ram(wptr, literal, RamNumber:7),
    ];

    let wptr_offset = literal.length >> BlockData:3;
    (result, compute_buffer_ptr_from_offset_neg(wptr, wptr_offset as Offset))
}

#[test]
fn test_literals_to_ram_req() {
    // BEFORE:                         AFTER:
    //     7  6  5  4  3  2  1  0          7  6  5  4  3  2  1  0
    // 1 |  |  |  |  |  |  |  |  |     1 |  |  |  |  |  |  |  |  |
    // 2 | o|  |  |  |  |  |  |  |     2 |11|  |  |  |  |  |  |  |
    // 3 |  |  |  |  |  |  |  |  |     3 |  |  |  |  |  |  |  | o|
    // 4 |  |  |  |  |  |  |  |  |     4 |  |  |  |  |  |  |  |  |

    let ptr = HistoryBufferPtr { number: RamNumber:7, addr: RamAddr:0x2 };
    let literals = LiteralPacket { data: u64:0x11, length: u64:8 };
    assert_eq(
        literals_to_ram_req(ptr, literals),
        (
            RWRamReq[RAM_NUM]:[
                ZERO_RWRAM_REQ,
                ZERO_RWRAM_REQ,
                ZERO_RWRAM_REQ,
                ZERO_RWRAM_REQ,
                ZERO_RWRAM_REQ,
                ZERO_RWRAM_REQ,
                ZERO_RWRAM_REQ,
                RWRamReq { data: RamData:0x11, addr: RamAddr:0x2, we: true, ..ZERO_RWRAM_REQ },
            ], HistoryBufferPtr { number: RamNumber:0, addr: RamAddr:0x3 },
        ));

    // BEFORE:                         AFTER:
    //     7  6  5  4  3  2  1  0          7  6  5  4  3  2  1  0
    // 1 |  |  |  |  |  |  |  |  |     1 |  |  |  |  |  |  |  |  |
    // 2 | o|  |  |  |  |  |  |  |     2 |11|  |  |  |  |  |  |  |
    // 3 |  |  |  |  |  |  |  |  |     3 | o|88|77|66|55|44|33|22|
    // 4 |  |  |  |  |  |  |  |  |     4 |  |  |  |  |  |  |  |  |

    let ptr = HistoryBufferPtr { number: RamNumber:7, addr: RamAddr:2 };
    let literals = LiteralPacket { data: u64:0x8877_6655_4433_2211, length: u64:64 };
    let result = literals_to_ram_req(ptr, literals);
    trace_fmt!("Results: {:#x}", result);
    assert_eq(
        literals_to_ram_req(ptr, literals),
        (
            RWRamReq[RAM_NUM]:[
                RWRamReq { data: RamData:0x22, addr: RamAddr:0x3, ..RWRAM_WRITE_REQ },
                RWRamReq { data: RamData:0x33, addr: RamAddr:0x3, ..RWRAM_WRITE_REQ },
                RWRamReq { data: RamData:0x44, addr: RamAddr:0x3, ..RWRAM_WRITE_REQ },
                RWRamReq { data: RamData:0x55, addr: RamAddr:0x3, ..RWRAM_WRITE_REQ },
                RWRamReq { data: RamData:0x66, addr: RamAddr:0x3, ..RWRAM_WRITE_REQ },
                RWRamReq { data: RamData:0x77, addr: RamAddr:0x3, ..RWRAM_WRITE_REQ },
                RWRamReq { data: RamData:0x88, addr: RamAddr:0x3, ..RWRAM_WRITE_REQ },
                RWRamReq { data: RamData:0x11, addr: RamAddr:0x2, ..RWRAM_WRITE_REQ },
            ], HistoryBufferPtr { number: RamNumber:7, addr: RamAddr:3 },
        ));
}

fn sequences_from_ram_req
    (ptr: HistoryBufferPtr, seq: SequencePacket) -> (RWRamReq[RAM_NUM], SequencePacket) {
    let next_seq = if seq.match_length > DATA_WIDTH as u64 {
        SequencePacket {
            match_length: seq.match_length - DATA_WIDTH as u64, offset_value: seq.offset_value
        }
    } else {
        zero!<SequencePacket>()
    };

    let reqs = for (i, reqs): (u32, RWRamReq[RAM_NUM]) in range(u32:0, RAM_NUM) {
        let min_length_for_write = (i as u64 + u64:1) << u64:3;
        if seq.match_length >= min_length_for_write {
            let rptr =
                compute_buffer_ptr_from_offset(ptr, (seq.offset_value - i as u64) as Offset);
            update(reqs, rptr.number, RWRamReq { addr: rptr.addr, re: true, ..ZERO_RWRAM_REQ })
        } else {
            reqs
        }
    }((RWRamReq[RAM_NUM]:[ZERO_RWRAM_REQ, ...]));
    (reqs, next_seq)
}

#[test]
fn test_sequences_from_ram_req() {
    //     7  6  5  4  3  2  1  0
    // 1 |  |  |  |  |  |  |  |  |
    // 2 |  |  |  |  |  |  |  |  |
    // 3 |  |  | o|  |  |XX|  |  |
    // 4 |  |  |  |  |  |  |  |  |

    let ptr = HistoryBufferPtr { number: RamNumber:5, addr: RamAddr:3 };
    let sequence = SequencePacket { match_length: u64:1 << u64:3, offset_value: u64:3 };

    assert_eq(
        sequences_from_ram_req(ptr, sequence),
        (
            RWRamReq[RAM_NUM]:[
                ZERO_RWRAM_REQ,
                ZERO_RWRAM_REQ,
                RWRamReq { addr: RamAddr:0x3, ..RWRAM_READ_REQ },
                ZERO_RWRAM_REQ,
                ZERO_RWRAM_REQ,
                ZERO_RWRAM_REQ,
                ZERO_RWRAM_REQ,
                ZERO_RWRAM_REQ,
            ], zero!<SequencePacket>(),
        ));

    //     7  6  5  4  3  2  1  0
    // 1 |  |  |  |  |  |  |  |  |
    // 2 |XX|XX|XX|XX|XX|  |  |  |
    // 3 |  |  | o|  |  |XX|XX|XX|
    // 4 |  |  |  |  |  |  |  |  |

    let ptr = HistoryBufferPtr { number: RamNumber:5, addr: RamAddr:3 };
    let sequence = SequencePacket { match_length: u64:8 << u64:3, offset_value: u64:10 };

    assert_eq(
        sequences_from_ram_req(ptr, sequence),
        (
            RWRamReq[RAM_NUM]:[
                RWRamReq { addr: RamAddr:0x3, ..RWRAM_READ_REQ },
                RWRamReq { addr: RamAddr:0x3, ..RWRAM_READ_REQ },
                RWRamReq { addr: RamAddr:0x3, ..RWRAM_READ_REQ },
                RWRamReq { addr: RamAddr:0x2, ..RWRAM_READ_REQ },
                RWRamReq { addr: RamAddr:0x2, ..RWRAM_READ_REQ },
                RWRamReq { addr: RamAddr:0x2, ..RWRAM_READ_REQ },
                RWRamReq { addr: RamAddr:0x2, ..RWRAM_READ_REQ },
                RWRamReq { addr: RamAddr:0x2, ..RWRAM_READ_REQ },
            ], zero!<SequencePacket>(),
        ));

    // BEFORE:                         AFTER:
    //     7  6  5  4  3  2  1  0          7  6  5  4  3  2  1  0
    // 1 |  |  |  |  |  |  |  |  |     1 |  |  |  |  |  |  |  |  |
    // 2 |XX|XX|XX|XX|XX|XX|  |  |     2 |  |  |  |  |  |  |  |  |
    // 3 |  |  | o|  |XX|XX|XX|XX|     3 |  |  | o|  |XX|XX|  |  |
    // 4 |  |  |  |  |  |  |  |  |     4 |  |  |  |  |  |  |  |  |

    let ptr = HistoryBufferPtr { number: RamNumber:5, addr: RamAddr:3 };
    let sequence = SequencePacket { match_length: u64:10 << u64:3, offset_value: u64:11 };

    assert_eq(
        sequences_from_ram_req(ptr, sequence),
        (
            RWRamReq[RAM_NUM]:[
                RWRamReq { addr: RamAddr:0x3, ..RWRAM_READ_REQ },
                RWRamReq { addr: RamAddr:0x3, ..RWRAM_READ_REQ },
                RWRamReq { addr: RamAddr:0x2, ..RWRAM_READ_REQ },
                RWRamReq { addr: RamAddr:0x2, ..RWRAM_READ_REQ },
                RWRamReq { addr: RamAddr:0x2, ..RWRAM_READ_REQ },
                RWRamReq { addr: RamAddr:0x2, ..RWRAM_READ_REQ },
                RWRamReq { addr: RamAddr:0x2, ..RWRAM_READ_REQ },
                RWRamReq { addr: RamAddr:0x2, ..RWRAM_READ_REQ },
            ], SequencePacket { match_length: u64:2 << u64:3, offset_value: u64:11 },
        ));
}

struct LiteralCompleterData {
    comp: bool[RAM_NUM],
    ptr: HistoryBufferPtr
}

fn create_literal_completer_data
    (reqs: RWRamReq[RAM_NUM], ptr: HistoryBufferPtr) -> LiteralCompleterData {
    LiteralCompleterData {
        comp:
        bool[RAM_NUM]:[
            (reqs[0]).we, (reqs[1]).we, (reqs[2]).we, (reqs[3]).we,
            (reqs[4]).we, (reqs[5]).we, (reqs[6]).we, (reqs[7]).we,
        ],
        ptr
    }
}

proc LiteralCompleter {
    input_r: chan<LiteralCompleterData> in;
    output_s: chan<HistoryBufferPtr> out;
    wr_comp_m0_r: chan<()> in;
    wr_comp_m1_r: chan<()> in;
    wr_comp_m2_r: chan<()> in;
    wr_comp_m3_r: chan<()> in;
    wr_comp_m4_r: chan<()> in;
    wr_comp_m5_r: chan<()> in;
    wr_comp_m6_r: chan<()> in;
    wr_comp_m7_r: chan<()> in;

    config(input_r: chan<LiteralCompleterData> in, output_s: chan<HistoryBufferPtr> out,
           wr_comp_m0_r: chan<()> in,
           wr_comp_m1_r: chan<()> in,
           wr_comp_m2_r: chan<()> in,
           wr_comp_m3_r: chan<()> in,
           wr_comp_m4_r: chan<()> in,
           wr_comp_m5_r: chan<()> in,
           wr_comp_m6_r: chan<()> in,
           wr_comp_m7_r: chan<()> in) {
        (
            input_r, output_s,
            wr_comp_m0_r, wr_comp_m1_r, wr_comp_m2_r, wr_comp_m3_r,
            wr_comp_m4_r, wr_comp_m5_r, wr_comp_m6_r, wr_comp_m7_r,
        )
    }

    init {  }

    next(tok0: token, state: ()) {
        let (tok1, input) = recv(tok0, input_r);
        trace_fmt!("LiteralCompleter:: received completion request {:#x}", input);

        let (tok2_0, _) = recv_if(tok1, wr_comp_m0_r, input.comp[0], ());
        let (tok2_1, _) = recv_if(tok1, wr_comp_m1_r, input.comp[1], ());
        let (tok2_2, _) = recv_if(tok1, wr_comp_m2_r, input.comp[2], ());
        let (tok2_3, _) = recv_if(tok1, wr_comp_m3_r, input.comp[3], ());
        let (tok2_4, _) = recv_if(tok1, wr_comp_m4_r, input.comp[4], ());
        let (tok2_5, _) = recv_if(tok1, wr_comp_m5_r, input.comp[5], ());
        let (tok2_6, _) = recv_if(tok1, wr_comp_m6_r, input.comp[6], ());
        let (tok2_7, _) = recv_if(tok1, wr_comp_m7_r, input.comp[7], ());
        let tok2 = join(tok2_0, tok2_1, tok2_2, tok2_3, tok2_4, tok2_5, tok2_6, tok2_7);

        trace_fmt!("LiteralCompleter:: sending completion info {:#x}", input.ptr);
        let tok3 = send(tok2, output_s, input.ptr);
    }
}

proc SequenceExecutor {
    input_r: chan<SequenceExecutorPacket> in;
    output_s: chan<ZstdDecodedPacket> out;
    l_final_r: chan<HistoryBufferPtr> in;
    l_comp_s: chan<LiteralCompleterData> out;
    req_m0_s: chan<RWRamReq> out;
    req_m1_s: chan<RWRamReq> out;
    req_m2_s: chan<RWRamReq> out;
    req_m3_s: chan<RWRamReq> out;
    req_m4_s: chan<RWRamReq> out;
    req_m5_s: chan<RWRamReq> out;
    req_m6_s: chan<RWRamReq> out;
    req_m7_s: chan<RWRamReq> out;
    resp_m0_r: chan<RWRamResp> in;
    resp_m1_r: chan<RWRamResp> in;
    resp_m2_r: chan<RWRamResp> in;
    resp_m3_r: chan<RWRamResp> in;
    resp_m4_r: chan<RWRamResp> in;
    resp_m5_r: chan<RWRamResp> in;
    resp_m6_r: chan<RWRamResp> in;
    resp_m7_r: chan<RWRamResp> in;

    config(input_r: chan<SequenceExecutorPacket> in, output_s: chan<ZstdDecodedPacket> out,
           req_m0_s: chan<RWRamReq> out,
           req_m1_s: chan<RWRamReq> out,
           req_m2_s: chan<RWRamReq> out,
           req_m3_s: chan<RWRamReq> out,
           req_m4_s: chan<RWRamReq> out,
           req_m5_s: chan<RWRamReq> out,
           req_m6_s: chan<RWRamReq> out,
           req_m7_s: chan<RWRamReq> out,
           resp_m0_r: chan<RWRamResp> in,
           resp_m1_r: chan<RWRamResp> in,
           resp_m2_r: chan<RWRamResp> in,
           resp_m3_r: chan<RWRamResp> in,
           resp_m4_r: chan<RWRamResp> in,
           resp_m5_r: chan<RWRamResp> in,
           resp_m6_r: chan<RWRamResp> in,
           resp_m7_r: chan<RWRamResp> in, wr_comp_m0_r: chan<()> in,
           wr_comp_m1_r: chan<()> in, wr_comp_m2_r: chan<()> in, wr_comp_m3_r: chan<()> in,
           wr_comp_m4_r: chan<()> in, wr_comp_m5_r: chan<()> in, wr_comp_m6_r: chan<()> in,
           wr_comp_m7_r: chan<()> in) {
        let (l_comp_s, l_comp_r) = chan<LiteralCompleterData, u32:1>;
        let (l_final_s, l_final_r) = chan<HistoryBufferPtr, u32:1>;

        spawn LiteralCompleter(
            l_comp_r, l_final_s, wr_comp_m0_r, wr_comp_m1_r, wr_comp_m2_r, wr_comp_m3_r,
            wr_comp_m4_r, wr_comp_m5_r, wr_comp_m6_r, wr_comp_m7_r);

        (
            input_r, output_s, l_final_r, l_comp_s, req_m0_s, req_m1_s, req_m2_s, req_m3_s,
            req_m4_s, req_m5_s, req_m6_s, req_m7_s, resp_m0_r, resp_m1_r, resp_m2_r, resp_m3_r,
            resp_m4_r, resp_m5_r, resp_m6_r, resp_m7_r,
        )
    }

    init { (ZERO_HISTORY_BUFFER_STATE) }

    next(tok0: token, state: SequenceExecutorState) {
        let is_seq_exec = state.status == SequenceExecutorStatus::SEQUENCE_EXEC;
        let is_lit_exec = state.status == SequenceExecutorStatus::LITERAL_EXEC;
        let is_idle = state.status == SequenceExecutorStatus::IDLE;
        let do_recv = !state.packet_valid;

        // Receive incoming data
        let (tok1, packet, packet_recv_valid) =
            recv_if_non_blocking(tok0, input_r, do_recv, state.packet);
        let is_lit_packet = packet.msg_type == SequenceExecutorMessageType::LITERAL;
        let is_seq_packet = packet.msg_type == SequenceExecutorMessageType::SEQUENCE;
        if packet_recv_valid {
            if is_lit_packet {
                trace_fmt!("SequenceExecutor:: received packet with literals {:#x}", packet);
            } else {
                trace_fmt!("SequenceExecutor:: received packet with sequences {:#x}", packet);
            };
        } else {  };

        // Send request to save literals to RAM
        let (reqs, hyp_ptr) = if packet_recv_valid && is_lit_packet && (is_idle | is_lit_exec) {
            let literals = ConvertToLiteralPacket(packet);
            literals_to_ram_req(state.hyp_ptr, literals)
        } else {
            (RWRamReq[RAM_NUM]:[ZERO_RWRAM_REQ, ...], zero!<HistoryBufferPtr>())
        };

        let tok2_1 = send_if(tok1, req_m0_s, (reqs[0]).we, reqs[0]);
        let tok2_2 = send_if(tok1, req_m1_s, (reqs[1]).we, reqs[1]);
        let tok2_3 = send_if(tok1, req_m2_s, (reqs[2]).we, reqs[2]);
        let tok2_4 = send_if(tok1, req_m3_s, (reqs[3]).we, reqs[3]);
        let tok2_5 = send_if(tok1, req_m4_s, (reqs[4]).we, reqs[4]);
        let tok2_6 = send_if(tok1, req_m5_s, (reqs[5]).we, reqs[5]);
        let tok2_7 = send_if(tok1, req_m6_s, (reqs[6]).we, reqs[6]);
        let tok2_8 = send_if(tok1, req_m7_s, (reqs[7]).we, reqs[7]);

        let do_send_lit = (reqs[0]).we | (reqs[1]).we | (reqs[2]).we | (reqs[3]).we
                        | (reqs[4]).we | (reqs[5]).we | (reqs[6]).we | (reqs[7]).we;

        let tok2_9 = send_if(
            tok1, l_comp_s, do_send_lit,
            LiteralCompleterData {
                comp: bool[RAM_NUM]:[
                    (reqs[0]).we, (reqs[1]).we, (reqs[2]).we, (reqs[3]).we,
                    (reqs[4]).we, (reqs[5]).we, (reqs[6]).we, (reqs[7]).we,
                ],
                ptr: hyp_ptr
            });

        let tok2_10 = send_if(tok1, output_s, do_send_lit, create_output_for_literal(packet));
        let tok2 = join(
            tok2_1, tok2_2, tok2_3, tok2_4, tok2_5,
            tok2_6, tok2_7, tok2_8, tok2_9, tok2_10
        );

        // Read RAM completion for literal writing
        let (tok3, real_ptr, valid) = recv_non_blocking(tok2, l_final_r, state.real_ptr);
        if valid {
            trace_fmt!("SequenceExecutor:: Received completion update");
        } else {  };

        // Update internal state of the module
        let status = if valid && (real_ptr == state.real_ptr) {
            SequenceExecutorStatus::IDLE
        } else {
            state.status
        };
        let real_ptr = if valid { real_ptr } else { state.real_ptr };
        let do_send_lit = (reqs[0]).we | (reqs[1]).we | (reqs[2]).we | (reqs[3]).we
                        | (reqs[4]).we | (reqs[5]).we | (reqs[6]).we | (reqs[7]).we;
        let hyp_ptr = if do_send_lit { hyp_ptr } else { state.hyp_ptr };

        let is_wrong_packet = (is_lit_exec && is_seq_packet) || (is_seq_exec && is_lit_packet);
        let (packet, packet_valid) =
            if is_wrong_packet { (packet, true) } else { (zero!<SequenceExecutorPacket>(), false) };

        let (tok1, _) = recv_if(tok0, resp_m0_r, false, ZERO_RWRAM_RESP);
        let (tok2, _) = recv_if(tok0, resp_m1_r, false, ZERO_RWRAM_RESP);
        let (tok3, _) = recv_if(tok0, resp_m2_r, false, ZERO_RWRAM_RESP);
        let (tok4, _) = recv_if(tok0, resp_m3_r, false, ZERO_RWRAM_RESP);
        let (tok5, _) = recv_if(tok0, resp_m4_r, false, ZERO_RWRAM_RESP);
        let (tok6, _) = recv_if(tok0, resp_m5_r, false, ZERO_RWRAM_RESP);
        let (tok7, _) = recv_if(tok0, resp_m6_r, false, ZERO_RWRAM_RESP);
        let (tok8, _) = recv_if(tok0, resp_m7_r, false, ZERO_RWRAM_RESP);

        SequenceExecutorState { status, hyp_ptr, real_ptr, packet, packet_valid }
    }
}

#[test_proc]
proc SequenceExecutorLiteralsTest {
    terminator: chan<bool> out;
    input_s: chan<SequenceExecutorPacket> out;
    output_r: chan<ZstdDecodedPacket> in;
    req1_s: chan<RWRamReq>[RAM_NUM] out;
    resp1_r: chan<RWRamResp>[RAM_NUM] in;
    wr_comp1_r: chan<()>[RAM_NUM] in;
    p_start_s: chan<()> out;
    p_finish_r: chan<()> in;

    config(terminator: chan<bool> out) {
        let (input_s, input_r) = chan<SequenceExecutorPacket>;
        let (output_s, output_r) = chan<ZstdDecodedPacket>;
        let (p_start_s, p_start_r) = chan<()>;
        let (p_finish_s, p_finish_r) = chan<()>;

        let (req0_s, req0_r) = chan<RWRamReq>[RAM_NUM];
        let (resp0_s, resp0_r) = chan<RWRamResp>[RAM_NUM];
        let (wr_comp0_s, wr_comp0_r) = chan<()>[RAM_NUM];
        let (req1_s, req1_r) = chan<RWRamReq>[RAM_NUM];
        let (resp1_s, resp1_r) = chan<RWRamResp>[RAM_NUM];
        let (wr_comp1_s, wr_comp1_r) = chan<()>[RAM_NUM];

        spawn SequenceExecutor(
            input_r, output_s,
            req0_s[0], req0_s[1], req0_s[2], req0_s[3],
            req0_s[4], req0_s[5], req0_s[6], req0_s[7],
            resp0_r[0], resp0_r[1], resp0_r[2], resp0_r[3],
            resp0_r[4], resp0_r[5], resp0_r[6], resp0_r[7],
            wr_comp0_r[0], wr_comp0_r[1], wr_comp0_r[2], wr_comp0_r[3],
            wr_comp0_r[4], wr_comp0_r[5], wr_comp0_r[6], wr_comp0_r[7]);

        spawn ram_printer::RamPrinter<
            RAM_DATA_WIDTH, RAM_SIZE, RAM_NUM_PARTITIONS, RAM_ADDR_WIDTH, RAM_NUM>(
            p_start_r, p_finish_s, req1_s, resp1_r);

        spawn ram::RamModel2RW<RAM_DATA_WIDTH, RAM_SIZE, RAM_WORD_PARTITION_SIZE>(
            req0_r[0], resp0_s[0], wr_comp0_s[0], req1_r[0], resp1_s[0], wr_comp1_s[0]);
        spawn ram::RamModel2RW<RAM_DATA_WIDTH, RAM_SIZE, RAM_WORD_PARTITION_SIZE>(
            req0_r[1], resp0_s[1], wr_comp0_s[1], req1_r[1], resp1_s[1], wr_comp1_s[1]);
        spawn ram::RamModel2RW<RAM_DATA_WIDTH, RAM_SIZE, RAM_WORD_PARTITION_SIZE>(
            req0_r[2], resp0_s[2], wr_comp0_s[2], req1_r[2], resp1_s[2], wr_comp1_s[2]);
        spawn ram::RamModel2RW<RAM_DATA_WIDTH, RAM_SIZE, RAM_WORD_PARTITION_SIZE>(
            req0_r[3], resp0_s[3], wr_comp0_s[3], req1_r[3], resp1_s[3], wr_comp1_s[3]);
        spawn ram::RamModel2RW<RAM_DATA_WIDTH, RAM_SIZE, RAM_WORD_PARTITION_SIZE>(
            req0_r[4], resp0_s[4], wr_comp0_s[4], req1_r[4], resp1_s[4], wr_comp1_s[4]);
        spawn ram::RamModel2RW<RAM_DATA_WIDTH, RAM_SIZE, RAM_WORD_PARTITION_SIZE>(
            req0_r[5], resp0_s[5], wr_comp0_s[5], req1_r[5], resp1_s[5], wr_comp1_s[5]);
        spawn ram::RamModel2RW<RAM_DATA_WIDTH, RAM_SIZE, RAM_WORD_PARTITION_SIZE>(
            req0_r[6], resp0_s[6], wr_comp0_s[6], req1_r[6], resp1_s[6], wr_comp1_s[6]);
        spawn ram::RamModel2RW<RAM_DATA_WIDTH, RAM_SIZE, RAM_WORD_PARTITION_SIZE>(
            req0_r[7], resp0_s[7], wr_comp0_s[7], req1_r[7], resp1_s[7], wr_comp1_s[7]);

        (
            terminator, input_s, output_r,
            req1_s, resp1_r, wr_comp1_r,
            p_start_s, p_finish_r,
        )
    }

    init {  }

    next(tok: token, state: ()) {

        trace_fmt!("==== Ram Settings ====");
        trace_fmt!(" number of RAMs: {}", RAM_NUM);
        trace_fmt!(" data width: {}", RAM_DATA_WIDTH);
        trace_fmt!(" address width: {}", RAM_ADDR_WIDTH);
        trace_fmt!(" size: {:#x} ({})", RAM_SIZE, RAM_SIZE);
        trace_fmt!(" word partition size: {:#x}", RAM_WORD_PARTITION_SIZE);
        trace_fmt!(" number of partitions: {}", RAM_NUM_PARTITIONS);

        let data_to_send: SequenceExecutorPacket[4] = [
            SequenceExecutorPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                length: BlockData:64,
                content: BlockData:0xAA00BB11CC22DD33,
                last: false
            },
            SequenceExecutorPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                length: BlockData:64,
                content: BlockData:0x447733220088CCFF,
                last: false
            },
            SequenceExecutorPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                length: BlockData:32,
                content: BlockData:0x88AA0022,
                last: false
            },
            SequenceExecutorPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                length: BlockData:32,
                content: BlockData:0xFFEEDD11,
                last: true
            },
        ];

        for (i, ()): (u32, ()) in range(u32:0, u32:4) {
            let tok = send(tok, input_s, data_to_send[i]);
            let (tok, recv_data) = recv(tok, output_r);
            let expected = create_output_for_literal(data_to_send[i]);
            trace_fmt!("{:#x} vs. {:#x}", expected, recv_data);
            assert_eq(expected, recv_data);
        }(());

        let tab:RamData[3][RAM_NUM] = [
            [ u8:0x33, u8:0xFF, u8:0x22 ],
            [ u8:0xDD, u8:0xCC, u8:0x00 ],
            [ u8:0x22, u8:0x88, u8:0xAA ],
            [ u8:0xCC, u8:0x00, u8:0x88 ],
            [ u8:0x11, u8:0x22, u8:0x11 ],
            [ u8:0xBB, u8:0x33, u8:0xDD ],
            [ u8:0x00, u8:0x77, u8:0xEE ],
            [ u8:0xAA, u8:0x44, u8:0xFF ],
        ];

        for (i, ()): (u32, ()) in range(u32:0, RAM_NUM) {
            for (j, ()): (u32, ()) in range(u32:0, u32:3) {
                let addr = j as RamAddr;
                let tok = send(tok, req1_s[i], ram::RWRamReq {
                    addr,
                    data: RamData:0,
                    write_mask: (),
                    read_mask: (),
                    we: false,
                    re: true
                });
            let (tok, resp) = recv(tok, resp1_r[i]);
            let expected = tab[i][j];
            assert_eq(expected, resp.data);
            }(());
        }(());

        // Print RAM contents
        let tok = send(tok, p_start_s, ());
        let (tok, _) = recv(tok, p_finish_r);

        send(tok, terminator, true);
    }
}
