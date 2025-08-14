// Copyright 2025 The XLS Authors
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

// This file contains SequenceEncoder proc implementation
// See steps 1-11 below for the functionality description
//
// NOTES:
// * only predefined compression tables
// * as of now the proc doesn't support long offsets (https://github.com/facebook/zstd/blob/e128976193546dceb24249206a02ff8f444f7120/lib/compress/zstd_compress_sequences.c#L318-L329)
// * ZSTD library uses a single buffer for storing both the transforms and symbol encodings where each half of the buffer serves different purpose, we split that to two
// * This file contains an implementation of a SequenceEncoderBuffer for writing bitstreams, it's quite generic and can be used for other procs
// * The biggest offset was assumed to fit in 16-bits
// * this assumes DATA_W <= WIDER_BUS_DATA_W

import std;

import xls.examples.ram;
import xls.modules.zstd.common;
import xls.modules.zstd.memory.axi;
import xls.modules.zstd.memory.axi_ram_reader;
import xls.modules.zstd.memory.axi_ram_writer;
import xls.modules.zstd.memory.mem_reader;
import xls.modules.zstd.memory.mem_writer;
import xls.modules.zstd.mem_writer_simple_arbiter;
import xls.modules.zstd.mem_reader_simple_arbiter;
import xls.modules.zstd.sequence_conf_enc;
import xls.modules.zstd.memory.mem_reader_data_upscaler;

const WIDER_BUS_DATA_W = u32:64;


fn highbit(val: u32) -> u32 {
    assert!(val != u32:0, "val is zero and thus has no highbit");
    u32:31 - clz(val)
}

#[test]
fn test_highbit() {
    assert_eq(highbit(u32:0x00_00_00_01), u32:0);
    assert_eq(highbit(u32:0x00_0F_A5_CC), u32:19);
    assert_eq(highbit(u32:0xFF_0F_A5_CC), u32:31);
}

const LL_DELTA_CODE = u32:19;
const LL_TO_CODE = u6[64]:[
    u6:0,  u6:1,  u6:2,  u6:3,  u6:4,  u6:5,  u6:6,  u6:7,
    u6:8,  u6:9,  u6:10, u6:11, u6:12, u6:13, u6:14, u6:15,
    u6:16, u6:16, u6:17, u6:17, u6:18, u6:18, u6:19, u6:19,
    u6:20, u6:20, u6:20, u6:20, u6:21, u6:21, u6:21, u6:21,
    u6:22, u6:22, u6:22, u6:22, u6:22, u6:22, u6:22, u6:22,
    u6:23, u6:23, u6:23, u6:23, u6:23, u6:23, u6:23, u6:23,
    u6:24, u6:24, u6:24, u6:24, u6:24, u6:24, u6:24, u6:24,
    u6:24, u6:24, u6:24, u6:24, u6:24, u6:24, u6:24, u6:24
];

// https://github.com/facebook/zstd/blob/5e7d721235e924b349b0e4b8b56860cf0e416394/lib/compress/zstd_compress_internal.h#L584
fn ll_to_code(litLength: u32) -> u6 {
    if litLength > u32:63 { (highbit(litLength) + LL_DELTA_CODE) as u6 } else { LL_TO_CODE[litLength] }
}

#[test]
fn test_ll_to_code() {
    assert_eq(ll_to_code(u32: 18), u6:17);  // val from LUT
    assert_eq(ll_to_code(u32: 63), u6:24);  // val from LUT
    assert_eq(ll_to_code(u32: 64), u6:25);  // hbit=6, 6 + 19 = 25
    assert_eq(ll_to_code(u32: 255), u6:26); // hbit=7, 7 + 19 = 26
    assert_eq(ll_to_code(u32: 256), u6:27); // hbit=8, 8 + 19 = 27
    assert_eq(ll_to_code(u32: 0xFFFF_FFFF), u6:50); // hbit=31, 31 + 19 = 50
}

const ML_DELTA_CODE = u32:36;
const ML_TO_CODE = u7[128]:[
    u7:0,  u7:1,  u7:2,  u7:3,  u7:4,  u7:5,  u7:6,  u7:7,  u7:8,  u7:9, u7:10, u7:11, u7:12, u7:13, u7:14, u7:15,
    u7:16, u7:17, u7:18, u7:19, u7:20, u7:21, u7:22, u7:23, u7:24, u7:25, u7:26, u7:27, u7:28, u7:29, u7:30, u7:31,
    u7:32, u7:32, u7:33, u7:33, u7:34, u7:34, u7:35, u7:35, u7:36, u7:36, u7:36, u7:36, u7:37, u7:37, u7:37, u7:37,
    u7:38, u7:38, u7:38, u7:38, u7:38, u7:38, u7:38, u7:38, u7:39, u7:39, u7:39, u7:39, u7:39, u7:39, u7:39, u7:39,
    u7:40, u7:40, u7:40, u7:40, u7:40, u7:40, u7:40, u7:40, u7:40, u7:40, u7:40, u7:40, u7:40, u7:40, u7:40, u7:40,
    u7:41, u7:41, u7:41, u7:41, u7:41, u7:41, u7:41, u7:41, u7:41, u7:41, u7:41, u7:41, u7:41, u7:41, u7:41, u7:41,
    u7:42, u7:42, u7:42, u7:42, u7:42, u7:42, u7:42, u7:42, u7:42, u7:42, u7:42, u7:42, u7:42, u7:42, u7:42, u7:42,
    u7:42, u7:42, u7:42, u7:42, u7:42, u7:42, u7:42, u7:42, u7:42, u7:42, u7:42, u7:42, u7:42, u7:42, u7:42, u7:42
];

// https://github.com/facebook/zstd/blob/5e7d721235e924b349b0e4b8b56860cf0e416394/lib/compress/zstd_compress_internal.h#L584
fn ml_to_code(mlBase: u32) -> u7 {
    if mlBase > u32:127 { (highbit(mlBase) + ML_DELTA_CODE) as u7 } else { ML_TO_CODE[mlBase] }
}

#[test]
fn test_ml_to_code() {
    assert_eq(ml_to_code(u32: 18), u7:18);  // val from LUT
    assert_eq(ml_to_code(u32: 127), u7:42); // val from LUT
    assert_eq(ml_to_code(u32: 128), u7:43); // hbit=7, 7 + 36 = 43
    assert_eq(ml_to_code(u32: 255), u7:43); // hbit=7, 7 + 36 = 43
    assert_eq(ml_to_code(u32: 256), u7:44); // hbit=8, 8 + 36 = 44
    assert_eq(ml_to_code(u32: 0xFFFF_FFFF), u7:67); // hbit=31, 31 + 36 = 67
}

const LL_CODE_TO_LEN = u5[36]:[
    u5:0, u5:0, u5:0, u5:0, u5:0, u5:0, u5:0, u5:0,
    u5:0, u5:0, u5:0, u5:0, u5:0, u5:0, u5:0, u5:0,
    u5:1, u5:1, u5:1, u5:1, u5:2, u5:2, u5:3, u5:3,
    u5:4, u5:6, u5:7, u5:8, u5:9, u5:10, u5:11, u5:12,
    u5:13, u5:14, u5:15, u5:16
];

const ML_CODE_TO_LEN = u5[53]: [
    u5:0, u5:0, u5:0, u5:0, u5:0, u5:0, u5:0, u5:0,
    u5:0, u5:0, u5:0, u5:0, u5:0, u5:0, u5:0, u5:0,
    u5:0, u5:0, u5:0, u5:0, u5:0, u5:0, u5:0, u5:0,
    u5:0, u5:0, u5:0, u5:0, u5:0, u5:0, u5:0, u5:0,
    u5:1, u5:1, u5:1, u5:1, u5:2, u5:2, u5:3, u5:3,
    u5:4, u5:4, u5:5, u5:7, u5:8, u5:9, u5:10, u5:11,
    u5:12, u5:13, u5:14, u5:15, u5:16
];

pub const SEQUENCE_RECORD_W = u32:48;
const SEQUENCE_RECORD_B = SEQUENCE_RECORD_W / u32:8;

pub struct Sequence {
    literals_len: u16,
    offset: u16,
    match_len: u16
}

fn sequence_to_codes(seq: Sequence) -> (u7, u6, u6) {
    (
        ml_to_code(seq.match_len as u32),
        ll_to_code(seq.literals_len as u32),
        highbit(seq.offset as u32) as u6
    )
}

fn get_sequence_addr<ADDR_W: u32>(base: uN[ADDR_W], count: u17, iter: uN[ADDR_W]) -> uN[ADDR_W] {
    let incr = (count - u17:1) as uN[ADDR_W] - iter;
    base + incr * SEQUENCE_RECORD_B as uN[ADDR_W]
}

pub enum SequenceEncoderStatus: u1 {
    OK = 0,
    ERROR = 1,
}

pub struct SequenceEncoderReq<ADDR_W: u32> {
    addr: uN[ADDR_W],
    seq_addr: uN[ADDR_W],
    seq_cnt: u17,
}

pub struct SequenceEncoderResp<ADDR_W: u32> {
    status: SequenceEncoderStatus,
    length: uN[ADDR_W]
}

struct CStatePtr<ADDR_W: u32> {
    value: uN[ADDR_W],
    state_table: uN[ADDR_W],
    symbol_tt: uN[ADDR_W],
    acc_log: u16
}

struct SequenceEncoderState<ADDR_W: u32, RAM_ADDR_W: u32> {
    active: bool,
    req: SequenceEncoderReq<ADDR_W>,
    iter: uN[ADDR_W],
    ll_acc_log: u16,
    ml_acc_log: u16,
    of_acc_log: u16,
    ll_value: u16,
    ml_value: u16,
    of_value: u16,
    bytes_written: uN[ADDR_W]
}

fn deserialize_sequence(data: uN[SEQUENCE_RECORD_W]) -> Sequence {
    Sequence {
        literals_len: data[32:48],
        offset: data[16:32],
        match_len: data[0:16]
    }
}

pub fn serialize_sequence(seq: Sequence) -> uN[SEQUENCE_RECORD_W] {
    seq.literals_len ++ seq.offset ++ seq.match_len
}

struct Delta {
    find_state: u32,
    nb_bits: u32
}

pub fn serialize_tt<RAM_DATA_W: u32>(delta: Delta) -> uN[RAM_DATA_W] {
    delta.find_state ++ delta.nb_bits
}

fn deserialize_tt<RAM_DATA_W: u32>(data: uN[RAM_DATA_W]) -> Delta {
    Delta {
        find_state: data[32:64],
        nb_bits: data[0:32]
    }
}

fn initial_value_address<RAM_ADDR_W: u32>(tt: Delta) -> uN[RAM_ADDR_W] {
    let nb_bits_out = (tt.nb_bits + (u32:1<<u32:15)) >> u32:16;
    let value = (nb_bits_out << u32:16) - tt.nb_bits;
    (value >> nb_bits_out) + tt.find_state
}

fn next_value_address_nbits<RAM_ADDR_W: u32>(tt:Delta, value: u16) -> (uN[RAM_ADDR_W], u5) {
    let nb_bits_out = (tt.nb_bits + (value as u32)) >> u32:16;
    ((value as u32 >> nb_bits_out) + tt.find_state, nb_bits_out as u5)
}

fn add_bits<BUFF_W: u32, BUFF_SIZE_W: u32>(buff: uN[BUFF_W], value:  uN[BUFF_W], len: uN[BUFF_SIZE_W], buff_size: uN[BUFF_SIZE_W]) -> uN[BUFF_W] {
    if len == uN[BUFF_SIZE_W]:0 {
        buff
    } else {
        let mask = (uN[BUFF_W]:1 << len) - uN[BUFF_W]:1;
        let masked = value & mask;
        (masked as uN[BUFF_W] << buff_size) | buff
    }
}

// generated with ZSTD: FSE_buildCTable_wksp(CTable_OffsetBits, OF_defaultNorm, DefaultMaxOff, OF_defaultNormLog, scratchBuffer, sizeof(scratchBuffer));
pub const OF_DEFAULT_CTABLE=u16[16]:[0x20,0x37,0x2e,0x25,0x33,0x2a,0x21,0x38,0x26,0x2f,0x2b,0x34,0x22,0x39,0x30,0x27,];
pub const OF_DEFAULT_TTABLE=Delta[16]:[
	Delta { find_state: u32:0xffffffff, nb_bits: u32:0x4ffe0 },
	Delta { find_state: u32:0x0, nb_bits: u32:0x4ffe0 },
	Delta { find_state: u32:0x1, nb_bits: u32:0x4ffe0 },
	Delta { find_state: u32:0x2, nb_bits: u32:0x4ffe0 },
	Delta { find_state: u32:0x3, nb_bits: u32:0x4ffe0 },
	Delta { find_state: u32:0x4, nb_bits: u32:0x4ffe0 },
	Delta { find_state: u32:0x4, nb_bits: u32:0x4ffc0 },
	Delta { find_state: u32:0x6, nb_bits: u32:0x4ffc0 },
	Delta { find_state: u32:0x8, nb_bits: u32:0x4ffc0 },
	Delta { find_state: u32:0xb, nb_bits: u32:0x4ffe0 },
	Delta { find_state: u32:0xc, nb_bits: u32:0x4ffe0 },
	Delta { find_state: u32:0xd, nb_bits: u32:0x4ffe0 },
	Delta { find_state: u32:0xe, nb_bits: u32:0x4ffe0 },
	Delta { find_state: u32:0xf, nb_bits: u32:0x4ffe0 },
	Delta { find_state: u32:0x10, nb_bits: u32:0x4ffe0 },
	Delta { find_state: u32:0x11, nb_bits: u32:0x4ffe0 }
];

// generated with ZSTD: FSE_buildCTable_wksp(CTable_LitLength, LL_defaultNorm, MaxLL, LL_defaultNormLog, scratchBuffer, sizeof(scratchBuffer));
pub const LL_DEFAULT_CTABLE=u16[32]:[0x40,0x41,0x56,0x6b,0x42,0x57,0x6c,0x58,0x6d,0x43,0x6e,0x44,0x59,0x5a,0x6f,0x45,0x70,0x46,0x5b,0x5c,0x71,0x47,0x72,0x48,0x5d,0x5e,0x73,0x49,0x74,0x5f,0x4a,0x75,];
pub const LL_DEFAULT_TTABLE=Delta[32]:[
	Delta { find_state: u32:0xfffffffc, nb_bits: u32:0x4ff80 },
	Delta { find_state: u32:0x1, nb_bits: u32:0x4ffa0 },
	Delta { find_state: u32:0x5, nb_bits: u32:0x5ff80 },
	Delta { find_state: u32:0x7, nb_bits: u32:0x5ff80 },
	Delta { find_state: u32:0x9, nb_bits: u32:0x5ff80 },
	Delta { find_state: u32:0xb, nb_bits: u32:0x5ff80 },
	Delta { find_state: u32:0xd, nb_bits: u32:0x5ff80 },
	Delta { find_state: u32:0xf, nb_bits: u32:0x5ff80 },
	Delta { find_state: u32:0x11, nb_bits: u32:0x5ff80 },
	Delta { find_state: u32:0x13, nb_bits: u32:0x5ff80 },
	Delta { find_state: u32:0x15, nb_bits: u32:0x5ff80 },
	Delta { find_state: u32:0x17, nb_bits: u32:0x5ff80 },
	Delta { find_state: u32:0x19, nb_bits: u32:0x5ff80 },
	Delta { find_state: u32:0x1c, nb_bits: u32:0x5ffc0 },
	Delta { find_state: u32:0x1d, nb_bits: u32:0x5ffc0 },
	Delta { find_state: u32:0x1e, nb_bits: u32:0x5ffc0 },
	Delta { find_state: u32:0x1e, nb_bits: u32:0x5ff80 },
	Delta { find_state: u32:0x20, nb_bits: u32:0x5ff80 },
	Delta { find_state: u32:0x22, nb_bits: u32:0x5ff80 },
	Delta { find_state: u32:0x24, nb_bits: u32:0x5ff80 },
	Delta { find_state: u32:0x26, nb_bits: u32:0x5ff80 },
	Delta { find_state: u32:0x28, nb_bits: u32:0x5ff80 },
	Delta { find_state: u32:0x2a, nb_bits: u32:0x5ff80 },
	Delta { find_state: u32:0x2c, nb_bits: u32:0x5ff80 },
	Delta { find_state: u32:0x2e, nb_bits: u32:0x5ff80 },
	Delta { find_state: u32:0x2f, nb_bits: u32:0x4ffa0 },
	Delta { find_state: u32:0x33, nb_bits: u32:0x5ff80 },
	Delta { find_state: u32:0x36, nb_bits: u32:0x5ffc0 },
	Delta { find_state: u32:0x37, nb_bits: u32:0x5ffc0 },
	Delta { find_state: u32:0x38, nb_bits: u32:0x5ffc0 },
	Delta { find_state: u32:0x39, nb_bits: u32:0x5ffc0 },
	Delta { find_state: u32:0x3a, nb_bits: u32:0x5ffc0 }
];

// generated with ZSTD: FSE_buildCTable_wksp(CTable_MatchLength, ML_defaultNorm, MaxML, ML_defaultNormLog, scratchBuffer, sizeof(scratchBuffer));
pub const ML_DEFAULT_CTABLE=u16[32]:[0x40,0x41,0x56,0x6b,0x6c,0x42,0x57,0x6d,0x43,0x58,0x59,0x6e,0x44,0x6f,0x45,0x5a,0x5b,0x70,0x46,0x71,0x5c,0x47,0x72,0x5d,0x48,0x73,0x5e,0x49,0x74,0x5f,0x4a,0x75,];
pub const ML_DEFAULT_TTABLE=Delta[32]:[
	Delta { find_state: u32:0xffffffff, nb_bits: u32:0x5ffc0 },
	Delta { find_state: u32:0xfffffffd, nb_bits: u32:0x4ff80 },
	Delta { find_state: u32:0x2, nb_bits: u32:0x4ffa0 },
	Delta { find_state: u32:0x6, nb_bits: u32:0x5ff80 },
	Delta { find_state: u32:0x8, nb_bits: u32:0x5ff80 },
	Delta { find_state: u32:0xa, nb_bits: u32:0x5ff80 },
	Delta { find_state: u32:0xc, nb_bits: u32:0x5ff80 },
	Delta { find_state: u32:0xe, nb_bits: u32:0x5ff80 },
	Delta { find_state: u32:0x10, nb_bits: u32:0x5ff80 },
	Delta { find_state: u32:0x13, nb_bits: u32:0x5ffc0 },
	Delta { find_state: u32:0x14, nb_bits: u32:0x5ffc0 },
	Delta { find_state: u32:0x15, nb_bits: u32:0x5ffc0 },
	Delta { find_state: u32:0x16, nb_bits: u32:0x5ffc0 },
	Delta { find_state: u32:0x17, nb_bits: u32:0x5ffc0 },
	Delta { find_state: u32:0x18, nb_bits: u32:0x5ffc0 },
	Delta { find_state: u32:0x19, nb_bits: u32:0x5ffc0 },
	Delta { find_state: u32:0x1a, nb_bits: u32:0x5ffc0 },
	Delta { find_state: u32:0x1b, nb_bits: u32:0x5ffc0 },
	Delta { find_state: u32:0x1c, nb_bits: u32:0x5ffc0 },
	Delta { find_state: u32:0x1d, nb_bits: u32:0x5ffc0 },
	Delta { find_state: u32:0x1e, nb_bits: u32:0x5ffc0 },
	Delta { find_state: u32:0x1f, nb_bits: u32:0x5ffc0 },
	Delta { find_state: u32:0x20, nb_bits: u32:0x5ffc0 },
	Delta { find_state: u32:0x21, nb_bits: u32:0x5ffc0 },
	Delta { find_state: u32:0x22, nb_bits: u32:0x5ffc0 },
	Delta { find_state: u32:0x23, nb_bits: u32:0x5ffc0 },
	Delta { find_state: u32:0x24, nb_bits: u32:0x5ffc0 },
	Delta { find_state: u32:0x25, nb_bits: u32:0x5ffc0 },
	Delta { find_state: u32:0x26, nb_bits: u32:0x5ffc0 },
	Delta { find_state: u32:0x27, nb_bits: u32:0x5ffc0 },
	Delta { find_state: u32:0x28, nb_bits: u32:0x5ffc0 },
	Delta { find_state: u32:0x29, nb_bits: u32:0x5ffc0 }
];

struct SequenceEncoderBufferState<ADDR_W: u32, BUFF_W: u32, BUFF_SIZE_W: u32> {
    flushing: bool,
    pad: bool,
    addr: uN[ADDR_W],
    buffer: uN[BUFF_W],
    buffer_size: uN[BUFF_SIZE_W],
    bytes_written: uN[ADDR_W]
}

type SequenceEncoderBufferInstr = u4;

const SET_ADDR = SequenceEncoderBufferInstr:1;
const WRITE = SequenceEncoderBufferInstr:2;
const FLUSH = SequenceEncoderBufferInstr:4;
const PAD = SequenceEncoderBufferInstr:8;

fn is_command(instr: SequenceEncoderBufferInstr, bit: SequenceEncoderBufferInstr) -> bool {
    instr & (bit as SequenceEncoderBufferInstr) != SequenceEncoderBufferInstr:0
}

struct SequenceEncoderBufferReq<ADDR_W: u32, BUFF_W: u32, BUFF_SIZE_W: u32> {
    instr: SequenceEncoderBufferInstr,
    write_data: uN[BUFF_W],
    write_data_size: uN[BUFF_SIZE_W],
    addr: uN[ADDR_W],
}

struct SequenceEncoderBufferSync<ADDR_W: u32> {
    bytes_written: uN[ADDR_W]
}

proc SequenceEncoderBuffer<
    ADDR_W: u32, DATA_W: u32, BUFF_W: u32,
    BUFF_SIZE_W: u32 = { std::clog2(BUFF_W) }
> {
    type State = SequenceEncoderBufferState<ADDR_W, BUFF_W, BUFF_SIZE_W>;
    type Req = SequenceEncoderBufferReq<ADDR_W, BUFF_W, BUFF_SIZE_W>;
    type Sync = SequenceEncoderBufferSync<ADDR_W>;

    type MemWriterReq = mem_writer::MemWriterReq<ADDR_W>;
    type MemWriterData = mem_writer::MemWriterDataPacket<DATA_W, ADDR_W>;
    type MemWriterResp = mem_writer::MemWriterResp;
    type MemWriterStatus = mem_writer::MemWriterRespStatus;

    init {zero!<State>()}

    req_r: chan<Req> in;
    sync_s: chan<Sync> out;
    mem_wr_req_s: chan<MemWriterReq> out;
    mem_wr_data_s: chan<MemWriterData> out;
    mem_wr_resp_r: chan<MemWriterResp> in;

    config(
        req_r: chan<Req> in,
        sync_s: chan<Sync> out,
        mem_wr_req_s: chan<MemWriterReq> out,
        mem_wr_data_s: chan<MemWriterData> out,
        mem_wr_resp_r: chan<MemWriterResp> in
    ) {
        (
            req_r, sync_s,
            mem_wr_req_s, mem_wr_data_s, mem_wr_resp_r
        )
    }

    next(state: State) {
        type BuffSize = uN[BUFF_SIZE_W];
        type Addr = uN[ADDR_W];
        const DATA_B = DATA_W / u32:8;
        const MINIMAL_SLACK = BuffSize:16 * BuffSize:6; // it's the max packet width the buffer can get from the encoder
        let tok = join();

        if !state.flushing {
            let (tok, req) = recv(tok, req_r);

            // WRITE
            let next_size = req.write_data_size + state.buffer_size;
            let buffer = if is_command(req.instr, WRITE) {
                trace_fmt!("[SequenceEncoderBuffer] Writing data from request {:#x} (size={})", req.write_data, req.write_data_size);
                let buffer = add_bits(state.buffer, req.write_data, req.write_data_size, state.buffer_size);
                trace_fmt!("[SequenceEncoderBuffer] buffer state: {:#b} (size={})", buffer, next_size);
                buffer
            } else {
                state.buffer
            };

            // SET_ADDR
            let addr = if is_command(req.instr, SET_ADDR) {
                trace_fmt!("[SequenceEncoderBuffer] setting address {:#x}", req.addr);
                req.addr
            } else {
                state.addr
            };

            // FLUSH or flushing condition met
            let flush = is_command(req.instr, FLUSH) || BUFF_W as BuffSize - next_size < MINIMAL_SLACK;
            let addr = if flush {
                let length = if is_command(req.instr, FLUSH) { std::ceil_div(next_size as u32, u32:8) } else { (next_size as u32 / DATA_W) * DATA_B };
                let tok = send(tok, mem_wr_req_s, MemWriterReq{
                    addr: addr,
                    length: length
                });
                trace_fmt!("[SequenceEncoderBuffer] Flushing {} B to {:#x}", length, addr);
                addr + length
            } else {
                let tok = send(tok, sync_s, Sync { bytes_written: Addr:0 });
                addr
            };

            State {
                flushing: flush,
                pad: is_command(req.instr, PAD),
                addr: addr,
                buffer: buffer,
                buffer_size: next_size,
                bytes_written: Addr:0
            }
        } else if state.buffer_size < DATA_W as BuffSize && state.pad {
            let to_write = state.buffer[s32:0:DATA_W as s32];
            let length =std::ceil_div(state.buffer_size as u32, u32:8) as u32;

            trace_fmt!("[SequenceEncoderBuffer] writing {:#x}", to_write);


            let tok = send(tok, mem_wr_data_s, MemWriterData {
                data: to_write as uN[DATA_W],
                length: length,
                last: true
            });
            let (tok, _) = recv(tok, mem_wr_resp_r);
            let tok = send(tok, sync_s, Sync { bytes_written: length + state.bytes_written });

            State {
                flushing: false,
                buffer: zero!<uN[BUFF_W]>(),
                buffer_size: BuffSize:0,
                bytes_written: Addr:0,
                ..state
            }
        }  else if state.buffer_size < DATA_W as BuffSize {
            let (tok, _) = recv(tok, mem_wr_resp_r);

            let tok = send(tok, sync_s, Sync { bytes_written: state.bytes_written });

            State {
                bytes_written: Addr:0,
                flushing: false,
                ..state
            }
        } else {
            let to_write = state.buffer[s32:0:DATA_W as s32];
            trace_fmt!("[SequenceEncoderBuffer] writing {:#x}", to_write);
            let tok = send(tok, mem_wr_data_s, MemWriterData {
                data: to_write,
                length: DATA_B,
                last: false
            });

            State {
                buffer: state.buffer >> DATA_W,
                buffer_size: state.buffer_size - DATA_W as BuffSize,
                bytes_written: state.bytes_written + DATA_B,
                ..state
            }
        }
    }
}

const INST_ADDR_W = u32:32;
const INST_DATA_W = u32:64;
const INST_BUFF_W = u32: 1024;
const INST_BUFF_SIZE_W = std::clog2(INST_BUFF_W + u32:1);
proc SequenceEncoderBufferInst {
    type Req = SequenceEncoderBufferReq<INST_ADDR_W, INST_BUFF_W, INST_BUFF_SIZE_W>;
    type Sync = SequenceEncoderBufferSync<INST_ADDR_W>;
    type MemWriterReq = mem_writer::MemWriterReq<INST_ADDR_W>;
    type MemWriterData = mem_writer::MemWriterDataPacket<INST_DATA_W, INST_ADDR_W>;
    type MemWriterResp = mem_writer::MemWriterResp;

    config(
        req_r: chan<Req> in,
        sync_s: chan<Sync> out,
        mem_wr_req_s: chan<MemWriterReq> out,
        mem_wr_data_s: chan<MemWriterData> out,
        mem_wr_resp_r: chan<MemWriterResp> in
    ) {
        spawn SequenceEncoderBuffer<INST_ADDR_W, INST_DATA_W, INST_BUFF_W, INST_BUFF_SIZE_W>
        (
            req_r, sync_s,
            mem_wr_req_s, mem_wr_data_s, mem_wr_resp_r
        );
    }
    init {}
    next(state:()) {}
}

pub proc SequenceEncoder<
    ADDR_W: u32, DATA_W: u32, RAM_ADDR_W: u32,
    CTABLE_RAM_DATA_W: u32, CTABLE_RAM_NUM_PARTITIONS: u32,
    TTABLE_RAM_DATA_W: u32, TTABLE_RAM_NUM_PARTITIONS: u32,
    BUFF_W: u32, BUFF_SIZE_W: u32 = { std::clog2(BUFF_W) }
> {
    type Req = SequenceEncoderReq<ADDR_W>;
    type Resp = SequenceEncoderResp<ADDR_W>;
    type Status = SequenceEncoderStatus;
    type State = SequenceEncoderState<ADDR_W, RAM_ADDR_W>;

    type HeaderReq = sequence_conf_enc::SequenceSectionHeaderWriterReq<ADDR_W>;
    type HeaderResp = sequence_conf_enc::SequenceSectionHeaderWriterResp;
    type HeaderStatus = sequence_conf_enc::SequenceSectionHeaderWriterStatus;
    type BufferReq = SequenceEncoderBufferReq<ADDR_W, BUFF_W, BUFF_SIZE_W>;
    type BufferSync = SequenceEncoderBufferSync<ADDR_W>;
    type CompressionMode = common::CompressionMode;

    type CTableRamRdReq = ram::ReadReq<RAM_ADDR_W, CTABLE_RAM_NUM_PARTITIONS>;
    type CTableRamRdResp = ram::ReadResp<CTABLE_RAM_DATA_W>;
    type TTableRamRdReq = ram::ReadReq<RAM_ADDR_W, TTABLE_RAM_NUM_PARTITIONS>;
    type TTableRamRdResp = ram::ReadResp<TTABLE_RAM_DATA_W>;

    type MemReaderReq = mem_reader::MemReaderReq<ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<DATA_W, ADDR_W>;
    type WiderMemReaderResp = mem_reader::MemReaderResp<WIDER_BUS_DATA_W, ADDR_W>;

    type MemReaderStatus = mem_reader::MemReaderStatus;

    type MemWriterReq = mem_writer::MemWriterReq<ADDR_W>;
    type MemWriterData = mem_writer::MemWriterDataPacket<DATA_W, ADDR_W>;
    type MemWriterResp = mem_writer::MemWriterResp;
    type MemWriterStatus = mem_writer::MemWriterRespStatus;

    init { zero!<State>() }

    req_r: chan<Req> in;
    resp_s: chan<Resp> out;

    sh_req_s: chan<HeaderReq> out;
    sh_resp_r: chan<HeaderResp> in;

    sb_req_s: chan<BufferReq> out;
    sb_sync_r: chan<BufferSync> in;

    mem_rd_req_s: chan<MemReaderReq> out;
    mem_rd_resp_r: chan<WiderMemReaderResp> in;

    ml_ctable_buf_req_s: chan<CTableRamRdReq> out;
    ml_ctable_buf_resp_r: chan<CTableRamRdResp> in;
    ll_ctable_buf_req_s: chan<CTableRamRdReq> out;
    ll_ctable_buf_resp_r: chan<CTableRamRdResp> in;
    of_ctable_buf_req_s: chan<CTableRamRdReq> out;
    of_ctable_buf_resp_r: chan<CTableRamRdResp> in;
    ml_ttable_buf_req_s: chan<TTableRamRdReq> out;
    ml_ttable_buf_resp_r: chan<TTableRamRdResp> in;
    ll_ttable_buf_req_s: chan<TTableRamRdReq> out;
    ll_ttable_buf_resp_r: chan<TTableRamRdResp> in;
    of_ttable_buf_req_s: chan<TTableRamRdReq> out;
    of_ttable_buf_resp_r: chan<TTableRamRdResp> in;


    config(
        req_r: chan<Req> in,
        resp_s: chan<Resp> out,

        mem_rd_req_s: chan<MemReaderReq> out,
        mem_rd_resp_r: chan<MemReaderResp> in,
        mem_wr_req_s: chan<MemWriterReq> out,
        mem_wr_data_s: chan<MemWriterData> out,
        mem_wr_resp_r: chan<MemWriterResp> in,

        ml_ctable_buf_req_s: chan<CTableRamRdReq> out,
        ml_ctable_buf_resp_r: chan<CTableRamRdResp> in,
        ll_ctable_buf_req_s: chan<CTableRamRdReq> out,
        ll_ctable_buf_resp_r: chan<CTableRamRdResp> in,
        of_ctable_buf_req_s: chan<CTableRamRdReq> out,
        of_ctable_buf_resp_r: chan<CTableRamRdResp> in,

        ml_ttable_buf_req_s: chan<TTableRamRdReq> out,
        ml_ttable_buf_resp_r: chan<TTableRamRdResp> in,
        ll_ttable_buf_req_s: chan<TTableRamRdReq> out,
        ll_ttable_buf_resp_r: chan<TTableRamRdResp> in,
        of_ttable_buf_req_s: chan<TTableRamRdReq> out,
        of_ttable_buf_resp_r: chan<TTableRamRdResp> in
    ) {
        let (sh_req_s, sh_req_r) = chan<HeaderReq, u32:1>("sh_req");
        let (sh_resp_s, sh_resp_r) = chan<HeaderResp, u32:1>("sh_resp");
        let (sb_req_s, sb_req_r) = chan<BufferReq, u32:1>("sb_req");
        let (sb_sync_s, sb_sync_r) = chan<BufferSync, u32:1>("sb_sync");
        let (n_mem_wr_req_s, n_mem_wr_req_r) = chan<MemWriterReq, u32:1>[u32:2]("n_mem_wr_req");
        let (n_mem_wr_data_s, n_mem_wr_data_r) = chan<MemWriterData, u32:1>[u32:2]("n_mem_wr_data");
        let (n_mem_wr_resp_s, n_mem_wr_resp_r) = chan<MemWriterResp, u32:1>[u32:2]("n_mem_wr_resp");
        let (wider_mem_rd_resp_s, wider_mem_rd_resp_r) = chan<WiderMemReaderResp, u32:1>("wider_mem_rd_resp");


        spawn mem_reader_data_upscaler::MemReaderDataUpscaler<ADDR_W, DATA_W, WIDER_BUS_DATA_W> (
            mem_rd_resp_r, wider_mem_rd_resp_s
        );


        spawn mem_writer_simple_arbiter::MemWriterSimpleArbiter<ADDR_W, DATA_W, u32:2> (
            n_mem_wr_req_r, n_mem_wr_data_r, n_mem_wr_resp_s,
            mem_wr_req_s, mem_wr_data_s, mem_wr_resp_r,
        );

        spawn sequence_conf_enc::SequenceHeaderWriter<ADDR_W, DATA_W>
        (
            sh_req_r, sh_resp_s,
            n_mem_wr_req_s[0], n_mem_wr_data_s[0], n_mem_wr_resp_r[0]
        );

        spawn SequenceEncoderBuffer<ADDR_W, DATA_W, BUFF_W>
        (
            sb_req_r, sb_sync_s,
            n_mem_wr_req_s[1], n_mem_wr_data_s[1], n_mem_wr_resp_r[1]
        );

        (
            req_r, resp_s,
            sh_req_s, sh_resp_r,
            sb_req_s, sb_sync_r,
            mem_rd_req_s, wider_mem_rd_resp_r,
            ml_ctable_buf_req_s, ml_ctable_buf_resp_r, ll_ctable_buf_req_s, ll_ctable_buf_resp_r, of_ctable_buf_req_s, of_ctable_buf_resp_r,
            ml_ttable_buf_req_s, ml_ttable_buf_resp_r, ll_ttable_buf_req_s, ll_ttable_buf_resp_r, of_ttable_buf_req_s, of_ttable_buf_resp_r
        )
    }

    next(state: State) {
        const DEFAULT_LL_ACC_LOG = u32:6;
        const DEFAULT_ML_ACC_LOG = u32:6;
        const DEFAULT_OF_ACC_LOG = u32:5;
        const TT_FULL_MASK = all_ones!<uN[TTABLE_RAM_NUM_PARTITIONS]>();
        const CT_FULL_MASK = all_ones!<uN[CTABLE_RAM_NUM_PARTITIONS]>();
        type Addr = uN[ADDR_W];

        let tok = join();
        if (!state.active) {
            let (tok, req) = recv(tok, req_r);
            trace_fmt!("[SequenceEncoder] Received request: {:#x}", req);

            // step 1. write the header
            let tok = send(tok, sh_req_s, HeaderReq {
                addr: req.addr,
                conf: common::SequenceConf {
                    sequence_count: req.seq_cnt,
                    literals_mode: CompressionMode::PREDEFINED,
                    offset_mode: CompressionMode::PREDEFINED,
                    match_mode: CompressionMode::PREDEFINED
                }
            });
            let (tok, sh_resp) = recv(tok, sh_resp_r);
            assert_eq(sh_resp.status, HeaderStatus::OK);

            if req.seq_cnt == u17:0 {
                // corner case: no sequences
                trace_fmt!("[SequenceEncoder] Warning! Called with seq_cnt=0");
                let resp = Resp {
                    status: Status::OK,
                    length: sh_resp.length
                };
                let tok = send(tok, resp_s, resp);
                zero!<State>()
            } else {
                // TODO: get ll, ml, of acc logs from compression table
                // TODO: write the compression table
                // step 2. read the first sequence
                let addr = get_sequence_addr<ADDR_W>(req.seq_addr, req.seq_cnt, u32:0);
                let tok = send(tok, mem_rd_req_s, MemReaderReq {
                    addr: addr,
                    length: SEQUENCE_RECORD_B
                });
                let (tok, seq) = recv(tok, mem_rd_resp_r);
                let serialized = seq.data;
                let seq = deserialize_sequence(serialized as uN[SEQUENCE_RECORD_W]);
                trace_fmt!("[SequenceEncoder] Serialized sequence {:#x} (at {:#x}) Deserialized sequence: {:#x}", serialized, addr, seq);
                let (ml, ll, of) = sequence_to_codes(seq);
                trace_fmt!("[SequenceEncoder] Received sequence[{:#x}]={:#x} (serialized={:#x}), encoded(ml={:#x}, ll={:#x}, of={:#x})", addr, seq, serialized, ml, ll, of);

                // step 3. read first delta info
                let tok = send(tok, ll_ttable_buf_req_s, TTableRamRdReq { addr: ll as Addr, mask: TT_FULL_MASK });
                let tok = send(tok, ml_ttable_buf_req_s, TTableRamRdReq { addr: ml as Addr, mask: TT_FULL_MASK });
                let tok = send(tok, of_ttable_buf_req_s, TTableRamRdReq { addr: of as Addr, mask: TT_FULL_MASK });
                let (tok, ll_tt) = recv(tok, ll_ttable_buf_resp_r);
                let (tok, ml_tt) = recv(tok, ml_ttable_buf_resp_r);
                let (tok, of_tt) = recv(tok, of_ttable_buf_resp_r);
                let ll_tt = deserialize_tt<TTABLE_RAM_DATA_W>(ll_tt.data);
                let ml_tt = deserialize_tt<TTABLE_RAM_DATA_W>(ml_tt.data);
                let of_tt = deserialize_tt<TTABLE_RAM_DATA_W>(of_tt.data);

                let ll_value_addr = initial_value_address<RAM_ADDR_W>(ll_tt);
                let ml_value_addr = initial_value_address<RAM_ADDR_W>(ml_tt);
                let of_value_addr = initial_value_address<RAM_ADDR_W>(of_tt);

                // step 4. read the first symbol encoding
                let tok = send(tok, ll_ctable_buf_req_s, CTableRamRdReq { addr: ll_value_addr, mask: CT_FULL_MASK});
                let tok = send(tok, ml_ctable_buf_req_s, CTableRamRdReq { addr: ml_value_addr, mask: CT_FULL_MASK});
                let tok = send(tok, of_ctable_buf_req_s, CTableRamRdReq { addr: of_value_addr, mask: CT_FULL_MASK});
                let (tok, ll_enc) = recv(tok, ll_ctable_buf_resp_r);
                let (tok, ml_enc) = recv(tok, ml_ctable_buf_resp_r);
                let (tok, of_enc) = recv(tok, of_ctable_buf_resp_r);

                trace_fmt!(
                    "[SequenceEncoder] Addresses: {:#x} {:#x} {:#x} Deltas: {:#x} {:#x} {:#x} Encoded: {:#x} {:#x} {:#x}",
                    ll_value_addr, ml_value_addr, of_value_addr, ll_tt, ml_tt, of_tt, ll_enc, ml_enc, of_enc
                );

                // step 5. write raw literal, match_len & offset
                // order:
                // - ll from sequence
                // - ml from sequence
                // - of from sequence
                // https://github.com/facebook/zstd/blob/e128976193546dceb24249206a02ff8f444f7120/lib/compress/zstd_compress_sequences.c#L314-L317
                // https://github.com/facebook/zstd/blob/e128976193546dceb24249206a02ff8f444f7120/lib/compress/zstd_compress_sequences.c#L328

                let write_size = uN[BUFF_SIZE_W]:0;
                let to_write = add_bits(zero!<uN[BUFF_W]>(), seq.literals_len as uN[BUFF_W], LL_CODE_TO_LEN[ll] as uN[BUFF_SIZE_W], write_size);
                let write_size = LL_CODE_TO_LEN[ll] as uN[BUFF_SIZE_W];
                let to_write = add_bits(to_write, seq.match_len as uN[BUFF_W], ML_CODE_TO_LEN[ml] as uN[BUFF_SIZE_W], write_size);
                let write_size = write_size + ML_CODE_TO_LEN[ml] as uN[BUFF_SIZE_W];
                let to_write = add_bits(to_write, seq.offset as uN[BUFF_W], of as uN[BUFF_SIZE_W], write_size);
                let write_size = write_size + of as uN[BUFF_SIZE_W];

                let tok = send(tok, sb_req_s, BufferReq {
                    instr: WRITE | SET_ADDR,
                    write_data: to_write,
                    write_data_size: write_size,
                    addr: req.addr + sh_resp.length
                });
                let (tok, sync) = recv(tok, sb_sync_r);

                // step 6. initialize state based on the last sequence read
                // https://github.com/facebook/zstd/blob/e128976193546dceb24249206a02ff8f444f7120/lib/common/fse.h#L443-L452
                State {
                    active: true,
                    req: req,
                    iter: Addr:1,
                    ll_acc_log: DEFAULT_LL_ACC_LOG as u16,
                    ml_acc_log: DEFAULT_ML_ACC_LOG as u16,
                    of_acc_log: DEFAULT_OF_ACC_LOG as u16,
                    ll_value: ll_enc.data,
                    ml_value: ml_enc.data,
                    of_value: of_enc.data,
                    bytes_written: sync.bytes_written + sh_resp.length
                }
            }

        } else if state.req.seq_cnt as Addr == state.iter {
            // step 11. write the final encoded values
            // order:
            // - ml value
            // - of value
            // - ll value
            // - end mark
            // https://github.com/facebook/zstd/blob/e128976193546dceb24249206a02ff8f444f7120/lib/compress/zstd_compress_sequences.c#L372-L380

            let write_size = uN[BUFF_SIZE_W]:0;
            let to_write = add_bits(zero!<uN[BUFF_W]>(), state.ml_value as uN[BUFF_W], state.ml_acc_log as uN[BUFF_SIZE_W], write_size);
            let write_size = write_size + state.ml_acc_log as uN[BUFF_SIZE_W];
            let to_write = add_bits(to_write, state.of_value as uN[BUFF_W], state.of_acc_log as uN[BUFF_SIZE_W], write_size);
            let write_size = write_size + state.of_acc_log as uN[BUFF_SIZE_W];
            let to_write = add_bits(to_write, state.ll_value as uN[BUFF_W], state.ll_acc_log as uN[BUFF_SIZE_W], write_size);
            let write_size = write_size + state.ll_acc_log as uN[BUFF_SIZE_W];
            let to_write = add_bits(to_write, uN[BUFF_W]:1, uN[BUFF_SIZE_W]:1, write_size); // end mark
            let write_size = write_size + uN[BUFF_SIZE_W]:1;


            trace_fmt!("Sent (final) {:#x}", to_write);
            let tok = send(tok, sb_req_s, BufferReq {
                instr: FLUSH | PAD | WRITE,
                write_data: to_write,
                write_data_size: write_size,
                addr: uN[ADDR_W]:0
            });
            let (tok, sync) = recv(tok, sb_sync_r);

            let resp = Resp {
                status: Status::OK,
                length: state.bytes_written + sync.bytes_written
            };
            let tok = send(tok, resp_s, resp);
            zero!<State>()
        } else {

            // step 7. read next sequence
            let addr =  get_sequence_addr<ADDR_W>(state.req.seq_addr, state.req.seq_cnt, state.iter);
            let tok = send(tok, mem_rd_req_s, MemReaderReq {
                addr: addr,
                length: SEQUENCE_RECORD_B
            });
            let (tok, seq) = recv(tok, mem_rd_resp_r);
            let serialized = seq.data;
            let seq = deserialize_sequence(serialized as uN[SEQUENCE_RECORD_W]);
            let (ml, ll, of) = sequence_to_codes(seq);
            trace_fmt!("[SequenceEncoder] Received sequence[{:#x}]={:#x} (serialized={:#x}), encoded(ml={:#x}, ll={:#x}, of={:#x})", addr, seq, serialized, ml, ll, of);

            // step 8. read delta info
            let tok = send(tok, ll_ttable_buf_req_s, TTableRamRdReq { addr: ll as Addr, mask: TT_FULL_MASK });
            let tok = send(tok, ml_ttable_buf_req_s, TTableRamRdReq { addr: ml as Addr, mask: TT_FULL_MASK });
            let tok = send(tok, of_ttable_buf_req_s, TTableRamRdReq { addr: of as Addr, mask: TT_FULL_MASK });
            let (tok, ll_tt) = recv(tok, ll_ttable_buf_resp_r);
            let (tok, ml_tt) = recv(tok, ml_ttable_buf_resp_r);
            let (tok, of_tt) = recv(tok, of_ttable_buf_resp_r);
            let ll_tt = deserialize_tt<TTABLE_RAM_DATA_W>(ll_tt.data);
            let ml_tt = deserialize_tt<TTABLE_RAM_DATA_W>(ml_tt.data);
            let of_tt = deserialize_tt<TTABLE_RAM_DATA_W>(of_tt.data);
            let (ll_value_addr, ll_nbits) = next_value_address_nbits<RAM_ADDR_W>(ll_tt, state.ll_value);
            let (ml_value_addr, ml_nbits) = next_value_address_nbits<RAM_ADDR_W>(ml_tt, state.ml_value);
            let (of_value_addr, of_nbits) = next_value_address_nbits<RAM_ADDR_W>(of_tt, state.of_value);

            // step 9. read next address
            let tok = send(tok, ll_ctable_buf_req_s, CTableRamRdReq { addr: ll_value_addr, mask: CT_FULL_MASK});
            let tok = send(tok, ml_ctable_buf_req_s, CTableRamRdReq { addr: ml_value_addr, mask: CT_FULL_MASK});
            let tok = send(tok, of_ctable_buf_req_s, CTableRamRdReq { addr: of_value_addr, mask: CT_FULL_MASK});
            let (tok, ll_enc) = recv(tok, ll_ctable_buf_resp_r);
            let (tok, ml_enc) = recv(tok, ml_ctable_buf_resp_r);
            let (tok, of_enc) = recv(tok, of_ctable_buf_resp_r);

            trace_fmt!(
                "[SequenceEncoder] Addresses: {:#x} {:#x} {:#x} Deltas: {:#x} {:#x} {:#x} Encoded: {:#x} {:#x} {:#x} nbits: {} {} {}",
                ll_value_addr, ml_value_addr, of_value_addr, ll_tt, ml_tt, of_tt, ll_enc, ml_enc, of_enc, ll_nbits, ml_nbits, of_nbits
            );

            // step 10. write raw literal values and encoded symbols from previous iteration
            // order:
            // - of value
            // - ml value
            // - ll value
            // - ll from sequence
            // - ml from sequence
            // - of from sequence
            // https://github.com/facebook/zstd/blob/e128976193546dceb24249206a02ff8f444f7120/lib/compress/zstd_compress_sequences.c#L345-L368
            let write_size = uN[BUFF_SIZE_W]:0;
            let to_write = add_bits(zero!<uN[BUFF_W]>(), state.of_value as uN[BUFF_W], of_nbits as uN[BUFF_SIZE_W], write_size);
            let write_size = of_nbits as uN[BUFF_SIZE_W];
            let to_write = add_bits(to_write, state.ml_value as uN[BUFF_W], ml_nbits as uN[BUFF_SIZE_W], write_size);
            let write_size = write_size + ml_nbits as uN[BUFF_SIZE_W];
            let to_write = add_bits(to_write, state.ll_value as uN[BUFF_W], ll_nbits as uN[BUFF_SIZE_W], write_size);
            let write_size = write_size + ll_nbits as uN[BUFF_SIZE_W];
            let to_write = add_bits(to_write, seq.literals_len as uN[BUFF_W], LL_CODE_TO_LEN[ll] as uN[BUFF_SIZE_W], write_size);
            let write_size = write_size + LL_CODE_TO_LEN[ll] as uN[BUFF_SIZE_W];
            let to_write = add_bits(to_write, seq.match_len as uN[BUFF_W], ML_CODE_TO_LEN[ml] as uN[BUFF_SIZE_W], write_size);
            let write_size = write_size + ML_CODE_TO_LEN[ml] as uN[BUFF_SIZE_W];
            let to_write = add_bits(to_write, seq.offset as uN[BUFF_W], of as uN[BUFF_SIZE_W], write_size);
            let write_size = write_size + of as uN[BUFF_SIZE_W];

            let tok = send(tok, sb_req_s, BufferReq {
                instr: WRITE,
                write_data: to_write,
                write_data_size: write_size,
                addr: uN[ADDR_W]:0
            });
            let (tok, sync) = recv(tok, sb_sync_r);
            trace_fmt!("Sent {:#x}", to_write);

            State {
                active: true,
                iter: state.iter + Addr:1,
                ll_value: ll_enc.data,
                ml_value: ml_enc.data,
                of_value: of_enc.data,
                bytes_written: state.bytes_written + sync.bytes_written,
                ..state
            }
        }
    }
}

const TEST_ADDR_W = u32:32;
const TEST_DATA_W = u32:64;
const TEST_DATA_W_DIV8 = TEST_DATA_W / u32:8;
const TEST_DEST_W = u32:8;
const TEST_ID_W = u32:8;
const TEST_WRITER_ID = u32:1;

const TEST_RAM_DATA_W = TEST_DATA_W;
const TEST_RAM_SIZE = u32:4096;
const TEST_RAM_ADDR_W = TEST_ADDR_W;
const TEST_RAM_PARTITION_SIZE = u32:8;
const TEST_RAM_NUM_PARTITIONS = ram::num_partitions(TEST_RAM_PARTITION_SIZE, TEST_RAM_DATA_W);
const TEST_RAM_SIMULTANEOUS_RW_BEHAVIOR = ram::SimultaneousReadWriteBehavior::READ_BEFORE_WRITE;
const TEST_RAM_INITIALIZED = true;
const TEST_RAM_ASSERT_VALID_READ = true;

const TEST_CT_DATA_W = u32:16;
const TEST_TT_DATA_W = u32:64;
const TEST_T_PARTITION_SIZE = u32:8;
const TEST_CT_NUM_PARTITIONS = ram::num_partitions(TEST_T_PARTITION_SIZE, TEST_CT_DATA_W);
const TEST_TT_NUM_PARTITIONS = ram::num_partitions(TEST_T_PARTITION_SIZE, TEST_TT_DATA_W);

const TEST_BUFF_W = u32:256;

const TEST_OUTPUT_ADDR = uN[TEST_ADDR_W]:0x100;
const TEST_SEQUENCES_CNT = u32:8;
const TEST_SEQUENCES = Sequence[TEST_SEQUENCES_CNT]:[
    Sequence { literals_len: u16:1, offset: u16:4, match_len: u16:0 },
    Sequence { literals_len: u16:0, offset: u16:5, match_len: u16:2 },
    Sequence { literals_len: u16:1, offset: u16:13, match_len: u16:1 },
    Sequence { literals_len: u16:0, offset: u16:7, match_len: u16:1 },
    Sequence { literals_len: u16:0, offset: u16:1, match_len: u16:0 },
    Sequence { literals_len: u16:0, offset: u16:11, match_len: u16:0 },
    Sequence { literals_len: u16:0, offset: u16:13, match_len: u16:0 },
    Sequence { literals_len: u16:0, offset: u16:25, match_len: u16:0 }
];

const EXPECTED_DATA_LEN = u32:19;
const EXPECTED_DATA = [
    // sequence header
    u8:0x08,    // sequence count
    u8:0x00,    // compression modes
    // encoded sequences
	u8:0x39,
	u8:0x01,
	u8:0x68,
	u8:0x01,
	u8:0x60,
	u8:0x05,
	u8:0x00,
	u8:0x00,
	u8:0xb0,
	u8:0x0b,
	u8:0x68,
	u8:0x89,
	u8:0xcb,
	u8:0x5d,
	u8:0x01,
	u8:0xe0,
	u8:0xae,
];

#[test_proc]
proc SequenceEncoderPredefinedTest {
    type Req = SequenceEncoderReq<TEST_ADDR_W>;
    type Resp = SequenceEncoderResp<TEST_ADDR_W>;
    type Addr = uN[TEST_ADDR_W];
    type Length = uN[TEST_ADDR_W];

    type CTRamWrReq = ram::WriteReq<TEST_RAM_ADDR_W, TEST_CT_DATA_W, TEST_CT_NUM_PARTITIONS>;
    type CTRamRdReq =  ram::ReadReq<TEST_RAM_ADDR_W, TEST_CT_NUM_PARTITIONS>;
    type CTRamRdResp = ram::ReadResp<TEST_CT_DATA_W>;
    type TTRamWrReq = ram::WriteReq<TEST_RAM_ADDR_W, TEST_TT_DATA_W, TEST_TT_NUM_PARTITIONS>;
    type TTRamRdReq =  ram::ReadReq<TEST_RAM_ADDR_W, TEST_TT_NUM_PARTITIONS>;
    type TTRamRdResp = ram::ReadResp<TEST_TT_DATA_W>;
    type RamRdReq = ram::ReadReq<TEST_RAM_ADDR_W, TEST_RAM_NUM_PARTITIONS>;
    type RamRdResp = ram::ReadResp<TEST_RAM_DATA_W>;
    type RamWrReq = ram::WriteReq<TEST_RAM_ADDR_W, TEST_RAM_DATA_W, TEST_RAM_NUM_PARTITIONS>;
    type RamWrResp = ram::WriteResp;
    type MemReaderReq = mem_reader::MemReaderReq<TEST_ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<TEST_DATA_W, TEST_ADDR_W>;
    type MemWriterReq = mem_writer::MemWriterReq<TEST_ADDR_W>;
    type MemWriterData = mem_writer::MemWriterDataPacket<TEST_DATA_W, TEST_ADDR_W>;
    type MemWriterResp = mem_writer::MemWriterResp;

    type AxiAr = axi::AxiAr<TEST_ADDR_W, TEST_ID_W>;
    type AxiR = axi::AxiR<TEST_DATA_W, TEST_ID_W>;
    type AxiAw = axi::AxiAw<TEST_ADDR_W, TEST_ID_W>;
    type AxiW = axi::AxiW<TEST_DATA_W, TEST_DATA_W_DIV8>;
    type AxiB = axi::AxiB<TEST_ID_W>;
    type AxiAddr = uN[TEST_ADDR_W];
    type AxiData = uN[TEST_DATA_W];
    type AxiId = uN[TEST_ID_W];
    type AxiStrb = uN[TEST_DEST_W];
    type AxiLen = u8;
    type AxiSize = axi::AxiAxSize;
    type AxiBurst = axi::AxiAxBurst;
    type AxiWriteResp = axi::AxiWriteResp;

    terminator: chan<bool> out;
    req_s: chan<Req> out;
    resp_r: chan<Resp> in;

    mem_rd_req_s: chan<MemReaderReq> out;
    mem_rd_resp_r: chan<MemReaderResp>in;
    mem_wr_req_s: chan<MemWriterReq> out;
    mem_wr_data_s: chan<MemWriterData> out;
    mem_wr_resp_r: chan<MemWriterResp> in;

    ml_ctable_buf_wr_req_s: chan<CTRamWrReq> out;
    ml_ctable_buf_wr_resp_r: chan<RamWrResp> in;
    ll_ctable_buf_wr_req_s: chan<CTRamWrReq> out;
    ll_ctable_buf_wr_resp_r: chan<RamWrResp> in;
    of_ctable_buf_wr_req_s: chan<CTRamWrReq> out;
    of_ctable_buf_wr_resp_r: chan<RamWrResp> in;
    ml_ttable_buf_wr_req_s: chan<TTRamWrReq> out;
    ml_ttable_buf_wr_resp_r: chan<RamWrResp> in;
    ll_ttable_buf_wr_req_s: chan<TTRamWrReq> out;
    ll_ttable_buf_wr_resp_r: chan<RamWrResp> in;
    of_ttable_buf_wr_req_s: chan<TTRamWrReq> out;
    of_ttable_buf_wr_resp_r: chan<RamWrResp> in;

    init {}

    config(terminator: chan<bool> out) {
        let (input_ram_rd_req_s, input_ram_rd_req_r) = chan<RamRdReq>("input_ram_rd_req");
        let (input_ram_rd_resp_s, input_ram_rd_resp_r) = chan<RamRdResp>("input_ram_rd_resp");
        let (input_ram_wr_req_s, input_ram_wr_req_r) = chan<RamWrReq>("input_ram_wr_req");
        let (input_ram_wr_resp_s, input_ram_wr_resp_r) = chan<RamWrResp>("input_ram_wr_resp");
        let (ll_ctable_ram_rd_req_s, ll_ctable_ram_rd_req_r) = chan<CTRamRdReq>("ll_ctable_ram_rd_req");
        let (ll_ctable_ram_rd_resp_s, ll_ctable_ram_rd_resp_r) = chan<CTRamRdResp>("ll_ctable_ram_rd_resp");
        let (ll_ctable_ram_wr_req_s, ll_ctable_ram_wr_req_r) = chan<CTRamWrReq>("ll_ctable_ram_wr_req");
        let (ll_ctable_ram_wr_resp_s, ll_ctable_ram_wr_resp_r) = chan<RamWrResp>("ll_ctable_ram_wr_resp");
        let (of_ctable_ram_rd_req_s, of_ctable_ram_rd_req_r) = chan<CTRamRdReq>("of_ctable_ram_rd_req");
        let (of_ctable_ram_rd_resp_s, of_ctable_ram_rd_resp_r) = chan<CTRamRdResp>("of_ctable_ram_rd_resp");
        let (of_ctable_ram_wr_req_s, of_ctable_ram_wr_req_r) = chan<CTRamWrReq>("of_ctable_ram_wr_req");
        let (of_ctable_ram_wr_resp_s, of_ctable_ram_wr_resp_r) = chan<RamWrResp>("of_ctable_ram_wr_resp");
        let (ml_ctable_ram_rd_req_s, ml_ctable_ram_rd_req_r) = chan<CTRamRdReq>("ml_ctable_ram_rd_req");
        let (ml_ctable_ram_rd_resp_s, ml_ctable_ram_rd_resp_r) = chan<CTRamRdResp>("ml_ctable_ram_rd_resp");
        let (ml_ctable_ram_wr_req_s, ml_ctable_ram_wr_req_r) = chan<CTRamWrReq>("ml_ctable_ram_wr_req");
        let (ml_ctable_ram_wr_resp_s, ml_ctable_ram_wr_resp_r) = chan<RamWrResp>("ml_ctable_ram_wr_resp");

        let (ll_ttable_ram_rd_req_s, ll_ttable_ram_rd_req_r) = chan<TTRamRdReq>("ll_ttable_ram_rd_req");
        let (ll_ttable_ram_rd_resp_s, ll_ttable_ram_rd_resp_r) = chan<TTRamRdResp>("ll_ttable_ram_rd_resp");
        let (ll_ttable_ram_wr_req_s, ll_ttable_ram_wr_req_r) = chan<TTRamWrReq>("ll_ttable_ram_wr_req");
        let (ll_ttable_ram_wr_resp_s, ll_ttable_ram_wr_resp_r) = chan<RamWrResp>("ll_ttable_ram_wr_resp");
        let (of_ttable_ram_rd_req_s, of_ttable_ram_rd_req_r) = chan<TTRamRdReq>("of_ttable_ram_rd_req");
        let (of_ttable_ram_rd_resp_s, of_ttable_ram_rd_resp_r) = chan<TTRamRdResp>("of_ttable_ram_rd_resp");
        let (of_ttable_ram_wr_req_s, of_ttable_ram_wr_req_r) = chan<TTRamWrReq>("of_ttable_ram_wr_req");
        let (of_ttable_ram_wr_resp_s, of_ttable_ram_wr_resp_r) = chan<RamWrResp>("of_ttable_ram_wr_resp");
        let (ml_ttable_ram_rd_req_s, ml_ttable_ram_rd_req_r) = chan<TTRamRdReq>("ml_ttable_ram_rd_req");
        let (ml_ttable_ram_rd_resp_s, ml_ttable_ram_rd_resp_r) = chan<TTRamRdResp>("ml_ttable_ram_rd_resp");
        let (ml_ttable_ram_wr_req_s, ml_ttable_ram_wr_req_r) = chan<TTRamWrReq>("ml_ttable_ram_wr_req");
        let (ml_ttable_ram_wr_resp_s, ml_ttable_ram_wr_resp_r) = chan<RamWrResp>("ml_ttable_ram_wr_resp");

        let (mem_rd_req_s, mem_rd_req_r) = chan<MemReaderReq>("mem_rd_req");
        let (mem_rd_resp_s, mem_rd_resp_r) = chan<MemReaderResp>("mem_rd_resp");
        let (mem_wr_req_s, mem_wr_req_r) = chan<MemWriterReq>("mem_wr_req");
        let (mem_wr_data_s, mem_wr_data_r) = chan<MemWriterData>("mem_wr_data");
        let (mem_wr_resp_s, mem_wr_resp_r) = chan<MemWriterResp>("mem_wr_resp");
        let (mem_axi_ar_s, mem_axi_ar_r) = chan<AxiAr>("mem_axi_ar");
        let (mem_axi_r_s, mem_axi_r_r) = chan<AxiR>("mem_axi_r");
        let (mem_axi_aw_s, mem_axi_aw_r) = chan<AxiAw>("mem_axi_aw");
        let (mem_axi_w_s, mem_axi_w_r) = chan<AxiW>("mem_axi_w");
        let (mem_axi_b_s, mem_axi_b_r) = chan<AxiB>("mem_axi_b");
        let (req_s, req_r) = chan<Req>("req");
        let (resp_s, resp_r) = chan<Resp>("resp");

        spawn ram::RamModel<
            TEST_RAM_DATA_W, TEST_RAM_SIZE, TEST_RAM_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_RW_BEHAVIOR, TEST_RAM_INITIALIZED,
            TEST_RAM_ASSERT_VALID_READ, TEST_RAM_ADDR_W,
        >(
            input_ram_rd_req_r, input_ram_rd_resp_s,
            input_ram_wr_req_r, input_ram_wr_resp_s,
        );

        spawn ram::RamModel<
        TEST_CT_DATA_W, TEST_RAM_SIZE, TEST_T_PARTITION_SIZE,
        TEST_RAM_SIMULTANEOUS_RW_BEHAVIOR, TEST_RAM_INITIALIZED,
        TEST_RAM_ASSERT_VALID_READ, TEST_RAM_ADDR_W, TEST_CT_NUM_PARTITIONS
        >(
            ll_ctable_ram_rd_req_r, ll_ctable_ram_rd_resp_s,
            ll_ctable_ram_wr_req_r, ll_ctable_ram_wr_resp_s,
        );

        spawn ram::RamModel<
        TEST_CT_DATA_W, TEST_RAM_SIZE, TEST_T_PARTITION_SIZE,
        TEST_RAM_SIMULTANEOUS_RW_BEHAVIOR, TEST_RAM_INITIALIZED,
        TEST_RAM_ASSERT_VALID_READ, TEST_RAM_ADDR_W, TEST_CT_NUM_PARTITIONS
        >(
            ml_ctable_ram_rd_req_r, ml_ctable_ram_rd_resp_s,
            ml_ctable_ram_wr_req_r, ml_ctable_ram_wr_resp_s,
        );

        spawn ram::RamModel<
        TEST_CT_DATA_W, TEST_RAM_SIZE, TEST_T_PARTITION_SIZE,
        TEST_RAM_SIMULTANEOUS_RW_BEHAVIOR, TEST_RAM_INITIALIZED,
        TEST_RAM_ASSERT_VALID_READ, TEST_RAM_ADDR_W, TEST_CT_NUM_PARTITIONS
        >(
            of_ctable_ram_rd_req_r, of_ctable_ram_rd_resp_s,
            of_ctable_ram_wr_req_r, of_ctable_ram_wr_resp_s,
        );

        spawn ram::RamModel<
        TEST_TT_DATA_W, TEST_RAM_SIZE, TEST_T_PARTITION_SIZE,
        TEST_RAM_SIMULTANEOUS_RW_BEHAVIOR, TEST_RAM_INITIALIZED,
        TEST_RAM_ASSERT_VALID_READ, TEST_RAM_ADDR_W, TEST_TT_NUM_PARTITIONS
        >(
            ll_ttable_ram_rd_req_r, ll_ttable_ram_rd_resp_s,
            ll_ttable_ram_wr_req_r, ll_ttable_ram_wr_resp_s,
        );

        spawn ram::RamModel<
        TEST_TT_DATA_W, TEST_RAM_SIZE, TEST_T_PARTITION_SIZE,
        TEST_RAM_SIMULTANEOUS_RW_BEHAVIOR, TEST_RAM_INITIALIZED,
        TEST_RAM_ASSERT_VALID_READ, TEST_RAM_ADDR_W, TEST_TT_NUM_PARTITIONS
        >(
            ml_ttable_ram_rd_req_r, ml_ttable_ram_rd_resp_s,
            ml_ttable_ram_wr_req_r, ml_ttable_ram_wr_resp_s,
        );

        spawn ram::RamModel<
        TEST_TT_DATA_W, TEST_RAM_SIZE, TEST_T_PARTITION_SIZE,
        TEST_RAM_SIMULTANEOUS_RW_BEHAVIOR, TEST_RAM_INITIALIZED,
        TEST_RAM_ASSERT_VALID_READ, TEST_RAM_ADDR_W, TEST_TT_NUM_PARTITIONS
        >(
            of_ttable_ram_rd_req_r, of_ttable_ram_rd_resp_s,
            of_ttable_ram_wr_req_r, of_ttable_ram_wr_resp_s,
        );

        spawn axi_ram_reader::AxiRamReader<
            TEST_ADDR_W, TEST_DATA_W,
            TEST_DEST_W, TEST_ID_W,
            TEST_RAM_SIZE,
        >(
            mem_axi_ar_r, mem_axi_r_s,
            input_ram_rd_req_s, input_ram_rd_resp_r,
        );

        spawn axi_ram_writer::AxiRamWriter<
            TEST_ADDR_W, TEST_DATA_W, TEST_ID_W, TEST_RAM_SIZE,
            TEST_RAM_ADDR_W, TEST_RAM_NUM_PARTITIONS
        > (
            mem_axi_aw_r, mem_axi_w_r, mem_axi_b_s,
            input_ram_wr_req_s, input_ram_wr_resp_r
        );

        spawn mem_reader::MemReader<
            TEST_DATA_W, TEST_ADDR_W, TEST_DEST_W, TEST_ID_W,
        >(
            mem_rd_req_r, mem_rd_resp_s,
            mem_axi_ar_s, mem_axi_r_r,
        );

        spawn mem_writer::MemWriter<TEST_ADDR_W, TEST_DATA_W, TEST_DEST_W, TEST_ID_W, TEST_WRITER_ID>(
            mem_wr_req_r, mem_wr_data_r,
            mem_axi_aw_s, mem_axi_w_s, mem_axi_b_r,
            mem_wr_resp_s,
        );

        spawn SequenceEncoder<
            TEST_ADDR_W, TEST_DATA_W, TEST_RAM_ADDR_W,
            TEST_CT_DATA_W, TEST_CT_NUM_PARTITIONS,
            TEST_TT_DATA_W, TEST_TT_NUM_PARTITIONS,
            TEST_BUFF_W
        >(
            req_r, resp_s,
            mem_rd_req_s, mem_rd_resp_r,
            mem_wr_req_s, mem_wr_data_s, mem_wr_resp_r,
            ml_ctable_ram_rd_req_s, ml_ctable_ram_rd_resp_r, ll_ctable_ram_rd_req_s, ll_ctable_ram_rd_resp_r, of_ctable_ram_rd_req_s, of_ctable_ram_rd_resp_r,
            ml_ttable_ram_rd_req_s, ml_ttable_ram_rd_resp_r, ll_ttable_ram_rd_req_s, ll_ttable_ram_rd_resp_r, of_ttable_ram_rd_req_s, of_ttable_ram_rd_resp_r
        );

        (
            terminator,
            req_s, resp_r,
            mem_rd_req_s, mem_rd_resp_r,
            mem_wr_req_s, mem_wr_data_s, mem_wr_resp_r,
            ml_ctable_ram_wr_req_s, ml_ctable_ram_wr_resp_r, ll_ctable_ram_wr_req_s, ll_ctable_ram_wr_resp_r, of_ctable_ram_wr_req_s, of_ctable_ram_wr_resp_r,
            ml_ttable_ram_wr_req_s, ml_ttable_ram_wr_resp_r, ll_ttable_ram_wr_req_s, ll_ttable_ram_wr_resp_r, of_ttable_ram_wr_req_s, of_ttable_ram_wr_resp_r
        )
    }

    next(state: ()) {
        type Addr = uN[TEST_ADDR_W];
        const CTMASK = all_ones!<uN[TEST_CT_NUM_PARTITIONS]>();
        const TTMASK = all_ones!<uN[TEST_TT_NUM_PARTITIONS]>();
        let tok = join();

        // setup test
        // write default values to compression tables & transition tables
        trace!("[TEST] Setting up the compression tables");

        let tok = for ((i, v), tok) in enumerate(OF_DEFAULT_CTABLE) {
            let tok = send(tok, of_ctable_buf_wr_req_s, CTRamWrReq{addr: i, data: v, mask: CTMASK});
            let (tok, _) = recv(tok, of_ctable_buf_wr_resp_r);
            tok
        }(tok);

        let tok = for ((i, v), tok) in enumerate(LL_DEFAULT_CTABLE) {
            let tok = send(tok, ll_ctable_buf_wr_req_s, CTRamWrReq{addr: i, data: v, mask: CTMASK});
            let (tok, _) = recv(tok, ll_ctable_buf_wr_resp_r);
            tok
        }(tok);

        let tok = for ((i, v), tok) in enumerate(ML_DEFAULT_CTABLE) {
            let tok = send(tok, ml_ctable_buf_wr_req_s, CTRamWrReq{addr: i, data: v, mask: CTMASK});
            let (tok, _) = recv(tok, ml_ctable_buf_wr_resp_r);
            tok
        }(tok);

        trace!("[TEST] Setting up the transform tables");

        let tok = for ((i, tt), tok) in enumerate(OF_DEFAULT_TTABLE) {
            let tok = send(tok, of_ttable_buf_wr_req_s, TTRamWrReq{addr: i, data: serialize_tt<TEST_TT_DATA_W>(tt), mask: TTMASK});
            let (tok, _) = recv(tok, of_ttable_buf_wr_resp_r);
            tok
        }(tok);

        let tok = for ((i, tt), tok) in enumerate(LL_DEFAULT_TTABLE) {
            let tok = send(tok, ll_ttable_buf_wr_req_s, TTRamWrReq{addr: i, data: serialize_tt<TEST_TT_DATA_W>(tt), mask: TTMASK});
            let (tok, _) = recv(tok, ll_ttable_buf_wr_resp_r);
            tok
        }(tok);

        let tok = for ((i, tt), tok) in enumerate(ML_DEFAULT_TTABLE) {
            let tok = send(tok, ml_ttable_buf_wr_req_s, TTRamWrReq{addr: i, data: serialize_tt<TEST_TT_DATA_W>(tt), mask: TTMASK});
            let (tok, _) = recv(tok, ml_ttable_buf_wr_resp_r);
            tok
        }(tok);

        // write sequences to memory
        let tok = send(tok, mem_wr_req_s, MemWriterReq {
            addr: Addr:0,
            length: TEST_SEQUENCES_CNT * SEQUENCE_RECORD_B
        });

        let tok = for ((_, seq), tok) in enumerate(TEST_SEQUENCES) {
            let data =  u16:0 ++ serialize_sequence(seq);
            trace_fmt!("[TEST] {:#x} serialized={:#x}", seq, data);

            let tok = send(tok, mem_wr_data_s, MemWriterData {
                data: data,
                length: SEQUENCE_RECORD_B,
                last: false
            });

            tok
        }(tok);
        let (tok, _) = recv(tok, mem_wr_resp_r);

        // start encoding the sequences
        let tok = send(tok, req_s, Req {
                addr: TEST_OUTPUT_ADDR,
                seq_addr: Addr:0,
                seq_cnt: TEST_SEQUENCES_CNT as u17
            }
        );
        trace!("[TEST] Running the encoding...");
        let (tok, resp) = recv(tok, resp_r);
        assert_eq(resp, SequenceEncoderResp {
            status: SequenceEncoderStatus::OK,
            length: EXPECTED_DATA_LEN
        });
        trace!("[TEST] Encoding done, checking output...");

        let tok = for ((i, expected), tok) in enumerate(EXPECTED_DATA) {
            let tok = send(tok, mem_rd_req_s, MemReaderReq {
                addr: TEST_OUTPUT_ADDR + i,
                length: u32:1
            });
            let (tok, data) = recv(tok, mem_rd_resp_r);

            trace_fmt!("[TEST] Offset {:#x}, comparing {:#x} == {:#x}", TEST_OUTPUT_ADDR + i, expected, data.data as u8);
            assert_eq(expected, data.data as u8);
            tok
        }(tok);

        send(tok, terminator, true);
    }
}
