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
import xls.modules.zstd.parallel_rams as parallel_rams;
import xls.modules.zstd.ram_printer as ram_printer;
import xls.examples.ram;

// Configurable RAM parameters
pub const RAM_DATA_WIDTH = common::SYMBOL_WIDTH;
const RAM_NUM = u32:8;
const RAM_NUM_CLOG2 = std::clog2(RAM_NUM);

type BlockData = common::BlockData;
type SequenceExecutorMessageType = common::SequenceExecutorMessageType;
type SequenceExecutorPacket = common::SequenceExecutorPacket<RAM_DATA_WIDTH>;
type CopyOrMatchContent = common::CopyOrMatchContent;
type CopyOrMatchLength = common::CopyOrMatchLength;
type ZstdDecodedPacket = common::ZstdDecodedPacket;
type BlockPacketLength = common::BlockPacketLength;
type Offset = common::Offset;

// Constants calculated from RAM parameters
const RAM_NUM_WIDTH = std::clog2(RAM_NUM);
pub const RAM_WORD_PARTITION_SIZE = RAM_DATA_WIDTH;
const RAM_ORDER_WIDTH = std::clog2(RAM_DATA_WIDTH);
pub const RAM_NUM_PARTITIONS = ram::num_partitions(RAM_WORD_PARTITION_SIZE, RAM_DATA_WIDTH);
const RAM_REQ_MASK_ALL = std::unsigned_max_value<RAM_NUM_PARTITIONS>();
const RAM_REQ_MASK_NONE = bits[RAM_NUM_PARTITIONS]:0;

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

type HistoryBufferPtr = parallel_rams::HistoryBufferPtr;
type RamWrRespHandlerData = parallel_rams::RamWrRespHandlerData;
type RamWrRespHandlerResp = parallel_rams::RamWrRespHandlerResp;
type RamRdRespHandlerData = parallel_rams::RamRdRespHandlerData;
type RamData = uN[RAM_DATA_WIDTH];
type RamNumber = parallel_rams::RamNumber;
type RamReadStart = parallel_rams::RamReadStart;
type RamReadLen = parallel_rams::RamReadLen;

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
    hb_len: uN[RAM_ADDR_WIDTH + RAM_NUM_CLOG2],
    // Repeat Offset handling
    repeat_offsets: Offset[3],
    repeat_req: bool,
    seq_cnt: bool,
}

fn decode_literal_packet<ADDR_W: u32, DATA_W: u32>(packet: SequenceExecutorPacket) -> mem_writer::MemWriterDataPacket<DATA_W, ADDR_W> {
    type MemWriterDataPacket  = mem_writer::MemWriterDataPacket<DATA_W, ADDR_W>;
    MemWriterDataPacket {
        data: packet.content as uN[DATA_W],
        length: packet.length as uN[ADDR_W],
        last: packet.last
    }
}

#[test]
fn test_decode_literal_packet() {
    const DATA_W = u32:64;
    const ADDR_W = u32:16;

    type MemWriterDataPacket  = mem_writer::MemWriterDataPacket<DATA_W, ADDR_W>;

    let content = CopyOrMatchContent:0xAA00_BB11_CC22_DD33;
    let length = CopyOrMatchLength:8;
    let last = false;

    assert_eq(
        decode_literal_packet<ADDR_W, DATA_W>(
            SequenceExecutorPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                length,
                content,
                last
            }
        ),
        MemWriterDataPacket {
            data: uN[DATA_W]:0xAA00BB11CC22DD33,
            length: uN[ADDR_W]:8,
            last: false
        }
    )
}

pub fn handle_repeated_offset_for_sequences<RAM_DATA_WIDTH: u32 = {u32:8}>
    (seq: SequenceExecutorPacket, repeat_offsets: Offset[3], repeat_req: bool)
    -> (SequenceExecutorPacket, Offset[3]) {
    type Packet = SequenceExecutorPacket;
    type Content = uN[RAM_DATA_WIDTH * u32:8];
    let modified_repeat_offsets = if repeat_req {
        Offset[3]:[repeat_offsets[1], repeat_offsets[2], repeat_offsets[0] - Offset:1]
    } else {
        repeat_offsets
    };

    let (seq, final_repeat_offsets) = if seq.content == Content:0 {
        fail!(
            "match_offset_zero_not_allowed",
            (zero!<Packet>(), Offset[3]:[Offset:0, ...]))
    } else if seq.content == Content:1 {
        let offset = modified_repeat_offsets[0];
        (
            Packet { content: offset as Content, ..seq },
            Offset[3]:[
                offset, repeat_offsets[1], repeat_offsets[2],
            ],
        )
    } else if seq.content == CopyOrMatchContent:2 {
        let offset = modified_repeat_offsets[1];
        (
            Packet { content: offset as Content, ..seq },
            Offset[3]:[
                offset, repeat_offsets[0], repeat_offsets[2],
            ],
        )
    } else if seq.content == CopyOrMatchContent:3 {
        let offset = modified_repeat_offsets[2];
        (
            Packet { content: offset as Content, ..seq },
            Offset[3]:[
                offset, repeat_offsets[0], repeat_offsets[1],
            ],
        )
    } else {
        let offset = seq.content as Offset - Offset:3;
        (
            Packet { content: offset as Content, ..seq },
            Offset[3]:[
                offset, repeat_offsets[0], repeat_offsets[1],
            ],
        )
    };
    (seq, final_repeat_offsets)
}

pub proc SequenceExecutor<HISTORY_BUFFER_SIZE_KB: u32,
     AXI_DATA_W: u32, AXI_ADDR_W: u32,
     RAM_SIZE: u32 = {parallel_rams::ram_size(HISTORY_BUFFER_SIZE_KB)},
     RAM_ADDR_WIDTH: u32 = {parallel_rams::ram_addr_width(HISTORY_BUFFER_SIZE_KB)},
     INIT_HB_PTR_ADDR: u32 = {u32:0}, INIT_HB_PTR_RAM: u32 = {u32:0},
     INIT_HB_LENGTH: u32 = {u32:0},
     RAM_SIZE_TOTAL: u32 = {RAM_SIZE * RAM_NUM}>
{
    type MemWriterDataPacket  = mem_writer::MemWriterDataPacket<AXI_DATA_W, AXI_ADDR_W>;

    input_r: chan<SequenceExecutorPacket> in;
    output_mem_wr_data_in_s: chan<MemWriterDataPacket> out;
    ram_comp_input_s: chan<RamWrRespHandlerData<RAM_ADDR_WIDTH>> out;
    ram_comp_output_r: chan<RamWrRespHandlerResp<RAM_ADDR_WIDTH>> in;
    ram_resp_input_s: chan<RamRdRespHandlerData> out;
    looped_channel_r: chan<SequenceExecutorPacket> in;
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
           output_mem_wr_data_in_s: chan<MemWriterDataPacket> out,
           looped_channel_r: chan<SequenceExecutorPacket> in,
           looped_channel_s: chan<SequenceExecutorPacket> out,
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
        let (ram_comp_output_s, ram_comp_output_r) = chan<RamWrRespHandlerResp<RAM_ADDR_WIDTH>, u32:1>("ram_comp_output");
        let (ram_resp_input_s, ram_resp_input_r) = chan<RamRdRespHandlerData, u32:1>("ram_resp_input");

        spawn parallel_rams::RamWrRespHandler<RAM_ADDR_WIDTH>(
            ram_comp_input_r, ram_comp_output_s,
            wr_resp_m0_r, wr_resp_m1_r, wr_resp_m2_r, wr_resp_m3_r,
            wr_resp_m4_r, wr_resp_m5_r, wr_resp_m6_r, wr_resp_m7_r);

        spawn parallel_rams::RamRdRespHandler(
            ram_resp_input_r, looped_channel_s,
            rd_resp_m0_r, rd_resp_m1_r, rd_resp_m2_r,
            rd_resp_m3_r, rd_resp_m4_r, rd_resp_m5_r, rd_resp_m6_r, rd_resp_m7_r);

        (
            input_r, output_mem_wr_data_in_s,
            ram_comp_input_s, ram_comp_output_r,
            ram_resp_input_s, looped_channel_r,
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
            hb_len: INIT_HB_LENGTH as uN[RAM_ADDR_WIDTH + RAM_NUM_CLOG2],
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
        type HistoryBufferLength = uN[RAM_ADDR_WIDTH + RAM_NUM_CLOG2];

        const ZERO_READ_REQS = ReadReq[RAM_NUM]:[zero!<ReadReq>(), ...];
        const ZERO_WRITE_REQS = WriteReq[RAM_NUM]:[zero!<WriteReq>(), ...];

        // Recieve literals and sequences from the input channel ...
        let do_recv_input = !state.packet_valid && state.status != Status::SEQUENCE_READ &&
                            state.status != Status::SEQUENCE_WRITE;
        let (tok1_0, input_packet, input_packet_valid) =
            recv_if_non_blocking(tok0, input_r, do_recv_input, zero!<Packet>());

        // ... or our own sequences from the looped channel
        let do_recv_ram = (
            state.status == Status::SEQUENCE_READ ||
            state.status == Status::SEQUENCE_WRITE
        );

        let (tok1_1, ram_packet, ram_packet_valid) = recv_if_non_blocking(tok0, looped_channel_r, do_recv_ram, zero!<Packet>());

        // Read RAM write completion, used for monitoring the real state
        // of the RAM and eventually changing the state to IDLE.
        // Going through the IDLE state is required for changing between
        // Literals and Sequences (and the other way around) and between every
        // Sequence read from the input (original sequence from the ZSTD stream).
        let (tok1_2, wr_resp, wr_resp_valid) =
            recv_non_blocking(tok0, ram_comp_output_r, zero!<RamWrRespHandlerResp>());
        if wr_resp_valid {
            trace_fmt!("SequenceExecutor:: Received completion update");
        } else { };

        let real_ptr = if wr_resp_valid { wr_resp.ptr } else { state.real_ptr };
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
        let (write_reqs, read_reqs, read_start, read_len, new_state) = match (
            status, packet_valid, packet.msg_type
        ) {
            // Handling LITERAL_WRITE
            (Status::LITERAL_WRITE, true, MsgType::LITERAL) => {
                trace_fmt!("SequenceExecutor:: Handling LITERAL packet in LITERAL_WRITE step");
                let (write_reqs, new_hyp_ptr) =
                    parallel_rams::literal_packet_to_write_reqs<HISTORY_BUFFER_SIZE_KB>(state.hyp_ptr, packet);
                let new_repeat_req = packet.length == CopyOrMatchLength:0;
                let hb_add = packet.length as HistoryBufferLength;
                let new_hb_len = std::mod_pow2(state.hb_len + hb_add, RAM_SIZE_TOTAL as uN[RAM_ADDR_WIDTH + RAM_NUM_CLOG2]);

                (
                    write_reqs, ZERO_READ_REQS, RamReadStart:0, RamReadLen:0,
                    State {
                        status: Status::LITERAL_WRITE,
                        packet: zero!<Packet>(),
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
                    ZERO_WRITE_REQS, ZERO_READ_REQS, RamReadStart:0, RamReadLen:0,
                    State { status, ..NO_VALID_PACKET_STATE },
                )
            },
            // Handling SEQUENCE_READ
            (Status::SEQUENCE_READ, true, MsgType::SEQUENCE) => {
                trace_fmt!("Handling SEQUENCE in SEQUENCE_READ state");
                let (packet, new_repeat_offsets) = if !state.seq_cnt {
                    handle_repeated_offset_for_sequences(
                        packet, state.repeat_offsets, state.repeat_req)
                } else {
                    (packet, state.repeat_offsets)
                };
                let (read_reqs, read_start, read_len, packet, packet_valid) = parallel_rams::sequence_packet_to_read_reqs<
                    HISTORY_BUFFER_SIZE_KB>(
                    state.hyp_ptr, packet, state.hb_len);

                (
                    ZERO_WRITE_REQS, read_reqs, read_start, read_len,
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
                let ZERO_RETURN = (ZERO_WRITE_REQS, ZERO_READ_REQS, RamReadStart:0, RamReadLen:0, zero!<State>());
                fail!("should_not_happen", (ZERO_RETURN))
            },
            // Handling SEQUENCE_WRITE
            (Status::SEQUENCE_WRITE, true, MsgType::LITERAL) => {
                trace_fmt!("Handling LITERAL in SEQUENCE_WRITE state: {}", status);
                let (write_reqs, new_hyp_ptr) =
                    parallel_rams::literal_packet_to_write_reqs<HISTORY_BUFFER_SIZE_KB>(state.hyp_ptr, packet);
                let hb_add = packet.length as HistoryBufferLength;
                let new_hb_len = std::mod_pow2(state.hb_len + hb_add, RAM_SIZE_TOTAL as uN[RAM_ADDR_WIDTH + RAM_NUM_CLOG2]);

                (
                    write_reqs, ZERO_READ_REQS, RamReadStart:0, RamReadLen:0,
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
                    ZERO_WRITE_REQS, ZERO_READ_REQS, RamReadStart:0, RamReadLen:0,
                    State { status, ..NO_VALID_PACKET_STATE },
                )
            },
            // Handling IDLE
            _ => {
                let status = Status::IDLE;
                (
                    ZERO_WRITE_REQS, ZERO_READ_REQS, RamReadStart:0, RamReadLen:0,
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
        let (do_write, wr_resp_handler_data) = parallel_rams::create_ram_wr_data(write_reqs, new_state.hyp_ptr);
        if do_write {
            trace_fmt!("Sending request to RamWrRespHandler: {:#x}", wr_resp_handler_data);
        } else { };
        let tok2_9 = send_if(tok1, ram_comp_input_s, do_write, wr_resp_handler_data);

        let do_write_output = do_write || (packet.last && packet.msg_type == SequenceExecutorMessageType::LITERAL);
        let output_mem_wr_data_in = decode_literal_packet<AXI_ADDR_W, AXI_DATA_W>(packet);
        if do_write_output { trace_fmt!("Sending output MemWriter data: {:#x}", output_mem_wr_data_in); } else {  };
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
            parallel_rams::create_ram_rd_data<RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>
            (read_reqs, read_start, read_len, packet.last, new_state.packet_valid);
        if do_read {
            trace_fmt!("Sending request to RamRdRespHandler: {:#x}", rd_resp_handler_data);
        } else { };
        let tok2_19 = send_if(tok1, ram_resp_input_s, do_read, rd_resp_handler_data);

        new_state
    }
}

pub const ZSTD_HISTORY_BUFFER_SIZE_KB: u32 = u32:64;
pub const ZSTD_RAM_SIZE = parallel_rams::ram_size(ZSTD_HISTORY_BUFFER_SIZE_KB);
pub const ZSTD_RAM_ADDR_WIDTH = parallel_rams::ram_addr_width(ZSTD_HISTORY_BUFFER_SIZE_KB);
const ZSTD_AXI_DATA_W = u32:64;
const ZSTD_AXI_ADDR_W = u32:16;

pub proc SequenceExecutorZstd {
    type MemWriterDataPacket  = mem_writer::MemWriterDataPacket<ZSTD_AXI_DATA_W, ZSTD_AXI_ADDR_W>;

    init {  }

    config(
        input_r: chan<SequenceExecutorPacket> in,
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
            input_r, output_mem_wr_data_in_s,
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
        length: CopyOrMatchLength:8,
        content: CopyOrMatchContent:0xAA00_BB11_CC22_DD33,
        last: false
    },
    SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::LITERAL,
        length: CopyOrMatchLength:8,
        content: CopyOrMatchContent:0x4477_3322_0088_CCFF,
        last: false
    },
    SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::LITERAL,
        length: CopyOrMatchLength:4,
        content: CopyOrMatchContent:0x88AA_0022,
        last: false
    },
    SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::LITERAL,
        length: CopyOrMatchLength:4,
        content: CopyOrMatchContent:0xFFEE_DD11,
        last: false
    },
    SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::LITERAL,
        length: CopyOrMatchLength:8,
        content: CopyOrMatchContent:0x9DAF_8B41_C913_EFDA,
        last: false
    },
    SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::LITERAL,
        length: CopyOrMatchLength:8,
        content: CopyOrMatchContent:0x157D_8C7E_B8B9_7CA3,
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

// #[test_proc]
// proc SequenceExecutorLiteralsTest {
//     type MemWriterDataPacket  = mem_writer::MemWriterDataPacket<TEST_DATA_W, TEST_ADDR_W>;
//     terminator: chan<bool> out;

//     input_s: chan<SequenceExecutorPacket> out;
//     output_mem_wr_data_in_r: chan<MemWriterDataPacket> in;

//     print_start_s: chan<()> out;
//     print_finish_r: chan<()> in;

//     ram_rd_req_s: chan<TestReadReq>[RAM_NUM] out;
//     ram_rd_resp_r: chan<TestReadResp>[RAM_NUM] in;
//     ram_wr_req_s: chan<TestWriteReq>[RAM_NUM] out;
//     ram_wr_resp_r: chan<TestWriteResp>[RAM_NUM] in;

//     config(terminator: chan<bool> out) {
//         let (input_s,  input_r) = chan<SequenceExecutorPacket>("input");
//         let (output_mem_wr_data_in_s,  output_mem_wr_data_in_r) = chan<MemWriterDataPacket>("output_mem_wr_data_in");

//         let (looped_channel_s, looped_channel_r) = chan<SequenceExecutorPacket>("looped_channels");

//         let (print_start_s, print_start_r) = chan<()>("print_start");
//         let (print_finish_s, print_finish_r) = chan<()>("print_finish");

//         let (ram_rd_req_s,  ram_rd_req_r) = chan<TestReadReq>[RAM_NUM]("ram_rd_req");
//         let (ram_rd_resp_s, ram_rd_resp_r) = chan<TestReadResp>[RAM_NUM]("ram_rd_resp");
//         let (ram_wr_req_s,  ram_wr_req_r) = chan<TestWriteReq>[RAM_NUM]("ram_wr_req");
//         let (ram_wr_resp_s, ram_wr_resp_r) = chan<TestWriteResp>[RAM_NUM]("ram_wr_resp");

//         let INIT_HB_PTR_ADDR = u32:127;
//         spawn SequenceExecutor<
//             TEST_HISTORY_BUFFER_SIZE_KB,
//             TEST_DATA_W, TEST_ADDR_W,
//             TEST_RAM_SIZE,
//             TEST_RAM_ADDR_WIDTH,
//             INIT_HB_PTR_ADDR,
//         > (
//             input_r, output_mem_wr_data_in_s,
//             looped_channel_r, looped_channel_s,
//             ram_rd_req_s[0], ram_rd_req_s[1], ram_rd_req_s[2], ram_rd_req_s[3],
//             ram_rd_req_s[4], ram_rd_req_s[5], ram_rd_req_s[6], ram_rd_req_s[7],
//             ram_rd_resp_r[0], ram_rd_resp_r[1], ram_rd_resp_r[2], ram_rd_resp_r[3],
//             ram_rd_resp_r[4], ram_rd_resp_r[5], ram_rd_resp_r[6], ram_rd_resp_r[7],
//             ram_wr_req_s[0], ram_wr_req_s[1], ram_wr_req_s[2], ram_wr_req_s[3],
//             ram_wr_req_s[4], ram_wr_req_s[5], ram_wr_req_s[6], ram_wr_req_s[7],
//             ram_wr_resp_r[0], ram_wr_resp_r[1], ram_wr_resp_r[2], ram_wr_resp_r[3],
//             ram_wr_resp_r[4], ram_wr_resp_r[5], ram_wr_resp_r[6], ram_wr_resp_r[7]
//         );

//         spawn ram_printer::RamPrinter<
//             RAM_DATA_WIDTH, TEST_RAM_SIZE, RAM_NUM_PARTITIONS,
//             TEST_RAM_ADDR_WIDTH, RAM_NUM>
//             (print_start_r, print_finish_s, ram_rd_req_s, ram_rd_resp_r);

//         spawn ram::RamModel<
//             RAM_DATA_WIDTH, TEST_RAM_SIZE, RAM_WORD_PARTITION_SIZE,
//             TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED>
//             (ram_rd_req_r[0], ram_rd_resp_s[0], ram_wr_req_r[0], ram_wr_resp_s[0]);
//         spawn ram::RamModel<
//             RAM_DATA_WIDTH, TEST_RAM_SIZE, RAM_WORD_PARTITION_SIZE,
//             TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED>
//             (ram_rd_req_r[1], ram_rd_resp_s[1], ram_wr_req_r[1], ram_wr_resp_s[1]);
//         spawn ram::RamModel<
//             RAM_DATA_WIDTH, TEST_RAM_SIZE, RAM_WORD_PARTITION_SIZE,
//             TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED>
//             (ram_rd_req_r[2], ram_rd_resp_s[2], ram_wr_req_r[2], ram_wr_resp_s[2]);
//         spawn ram::RamModel<
//             RAM_DATA_WIDTH, TEST_RAM_SIZE, RAM_WORD_PARTITION_SIZE,
//             TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED>
//             (ram_rd_req_r[3], ram_rd_resp_s[3], ram_wr_req_r[3], ram_wr_resp_s[3]);
//         spawn ram::RamModel<
//             RAM_DATA_WIDTH, TEST_RAM_SIZE, RAM_WORD_PARTITION_SIZE,
//             TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED>
//             (ram_rd_req_r[4], ram_rd_resp_s[4], ram_wr_req_r[4], ram_wr_resp_s[4]);
//         spawn ram::RamModel<
//             RAM_DATA_WIDTH, TEST_RAM_SIZE, RAM_WORD_PARTITION_SIZE,
//             TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED>
//             (ram_rd_req_r[5], ram_rd_resp_s[5], ram_wr_req_r[5], ram_wr_resp_s[5]);
//         spawn ram::RamModel<
//             RAM_DATA_WIDTH, TEST_RAM_SIZE, RAM_WORD_PARTITION_SIZE,
//             TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED>
//             (ram_rd_req_r[6], ram_rd_resp_s[6], ram_wr_req_r[6], ram_wr_resp_s[6]);
//         spawn ram::RamModel<
//             RAM_DATA_WIDTH, TEST_RAM_SIZE, RAM_WORD_PARTITION_SIZE,
//             TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED>
//             (ram_rd_req_r[7], ram_rd_resp_s[7], ram_wr_req_r[7], ram_wr_resp_s[7]);

//         (
//             terminator,
//             input_s, output_mem_wr_data_in_r,
//             print_start_s, print_finish_r,
//             ram_rd_req_s, ram_rd_resp_r,
//             ram_wr_req_s, ram_wr_resp_r
//         )
//     }

//     init {  }

//     next(state: ()) {
//         let tok = join();
//         for (i, ()): (u32, ()) in range(u32:0, array_size(LITERAL_TEST_INPUT_DATA)) {
//             let tok = send(tok, input_s, LITERAL_TEST_INPUT_DATA[i]);
//             // Don't receive when there's an empty literals packet which is not last
//             if (LITERAL_TEST_INPUT_DATA[i].msg_type != SequenceExecutorMessageType::LITERAL ||
//                 LITERAL_TEST_INPUT_DATA[i].length != CopyOrMatchLength:0 ||
//                 LITERAL_TEST_INPUT_DATA[i].last) {
//                 let expected_mem_writer_data = decode_literal_packet<TEST_ADDR_W, TEST_DATA_W>(LITERAL_TEST_INPUT_DATA[i]);
//                 let (tok, recv_mem_writer_data) = recv(tok, output_mem_wr_data_in_r);
//                 assert_eq(expected_mem_writer_data, recv_mem_writer_data);
//             } else {}
//         }(());

//         for (i, ()): (u32, ()) in range(u32:0, RAM_NUM) {
//             for (j, ()): (u32, ()) in range(u32:0, array_size(LITERAL_TEST_MEMORY_CONTENT[0])) {
//                 let addr = LITERAL_TEST_MEMORY_CONTENT[i][j].0;
//                 let tok = send(tok, ram_rd_req_s[i], TestReadReq { addr, mask: RAM_REQ_MASK_ALL });
//                 let (tok, resp) = recv(tok, ram_rd_resp_r[i]);
//                 let expected = LITERAL_TEST_MEMORY_CONTENT[i][j].1;
//                 assert_eq(expected, resp.data);
//             }(());
//         }(());

//         // Print RAM content
//         let tok = send(tok, print_start_s, ());
//         let (tok, _) = recv(tok, print_finish_r);

//         send(tok, terminator, true);
//     }
// }

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

type TestMemWriterDataPacket = mem_writer::MemWriterDataPacket<TEST_DATA_W, TEST_ADDR_W>;
const SEQUENCE_TEST_EXPECTED_SEQUENCE_RESULTS:TestMemWriterDataPacket[11] = [
    TestMemWriterDataPacket {
        data: uN[TEST_DATA_W]:0x8C_7E_B8_B9_7C_A3_9D_AF,
        length: uN[TEST_ADDR_W]:8,
        last: false
    },
    TestMemWriterDataPacket {
        data: uN[TEST_DATA_W]:0x7D,
        length: uN[TEST_ADDR_W]:1,
        last: false
    },
    TestMemWriterDataPacket {
        data: uN[TEST_DATA_W]:0xB8,
        length: uN[TEST_ADDR_W]:1,
        last: false
    },
    TestMemWriterDataPacket {
        data: uN[TEST_DATA_W]:0xB8,
        length: uN[TEST_ADDR_W]:1,
        last: false
    },
    TestMemWriterDataPacket {
        data: uN[TEST_DATA_W]:0xB8_B9_7C_A3_9D,
        length: uN[TEST_ADDR_W]:5,
        last: false
    },
    TestMemWriterDataPacket {
        data: uN[TEST_DATA_W]:0xB9_7C_A3,
        length: uN[TEST_ADDR_W]:3,
        last: false
    },
    TestMemWriterDataPacket {
        data: uN[TEST_DATA_W]:0xB8,
        length: uN[TEST_ADDR_W]:1,
        last: false
    },
    TestMemWriterDataPacket {
        data: uN[TEST_DATA_W]:0x7C,
        length: uN[TEST_ADDR_W]:1,
        last: false
    },
    TestMemWriterDataPacket {
        data: uN[TEST_DATA_W]:0xB9_7C_A3_B8_B9_7C_A3_9D,
        length: uN[TEST_ADDR_W]:8,
        last: false
    },
    TestMemWriterDataPacket {
        data: uN[TEST_DATA_W]:0x7C_B8,
        length: uN[TEST_ADDR_W]:2,
        last: true
    },
    TestMemWriterDataPacket {
        data: uN[TEST_DATA_W]:0x9D,
        length: uN[TEST_ADDR_W]:1,
        last: false
    }
];

#[test_proc]
proc SequenceExecutorSequenceTest {
    type MemWriterDataPacket  = mem_writer::MemWriterDataPacket<TEST_DATA_W, TEST_ADDR_W>;
    terminator: chan<bool> out;

    input_s: chan<SequenceExecutorPacket> out;
    output_mem_wr_data_in_r: chan<MemWriterDataPacket> in;

    print_start_s: chan<()> out;
    print_finish_r: chan<()> in;

    ram_rd_req_s: chan<TestReadReq>[RAM_NUM] out;
    ram_rd_resp_r: chan<TestReadResp>[RAM_NUM] in;
    ram_wr_req_s: chan<TestWriteReq>[RAM_NUM] out;
    ram_wr_resp_r: chan<TestWriteResp>[RAM_NUM] in;

    config(terminator: chan<bool> out) {
        let (input_s, input_r) = chan<SequenceExecutorPacket>("input");
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
            input_r, output_mem_wr_data_in_s,
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
            input_s, output_mem_wr_data_in_r,
            print_start_s, print_finish_r,
            ram_rd_req_s, ram_rd_resp_r, ram_wr_req_s, ram_wr_resp_r
        )
    }

    init {  }

    next(state: ()) {
        let tok = join();

        // Print RAM content
        // let tok = send(tok, print_start_s, ());
        // let (tok, _) = recv(tok, print_finish_r);

        let tok = send(tok, input_s, SequenceExecutorPacket {
            msg_type: SequenceExecutorMessageType::LITERAL,
            length: CopyOrMatchLength:1,
            content: CopyOrMatchContent:0x31,
            last: false
        });
        let (tok, recv_data) = recv(tok, output_mem_wr_data_in_r);
        assert_eq(recv_data, TestMemWriterDataPacket {
            data: uN[TEST_DATA_W]:0x31,
            length: uN[TEST_ADDR_W]:1,
            last: false
        });

        let tok = send(tok, print_start_s, ());
        let (tok, _) = recv(tok, print_finish_r);

        let tok = send(tok, input_s, SequenceExecutorPacket {
            msg_type: SequenceExecutorMessageType::SEQUENCE,
            length: CopyOrMatchLength:3,
            content: CopyOrMatchContent:4,
            last: false
        });
        let (tok, recv_data) = recv(tok, output_mem_wr_data_in_r);
        assert_eq(recv_data, TestMemWriterDataPacket {
            data: uN[TEST_DATA_W]:0x31,
            length: uN[TEST_ADDR_W]:1,
            last: false
        });

        let tok = send(tok, print_start_s, ());
        let (tok, _) = recv(tok, print_finish_r);


        let (tok, recv_data) = recv(tok, output_mem_wr_data_in_r);
        assert_eq(recv_data, TestMemWriterDataPacket {
            data: uN[TEST_DATA_W]:0x3131,
            length: uN[TEST_ADDR_W]:2,
            last: false
        });

        let tok = send(tok, print_start_s, ());
        let (tok, _) = recv(tok, print_finish_r);

        let tok = send(tok, input_s, SequenceExecutorPacket {
            msg_type: SequenceExecutorMessageType::SEQUENCE,
            length: CopyOrMatchLength:3,
            content: CopyOrMatchContent:7,
            last: false
        });
        let (tok, recv_data) = recv(tok, output_mem_wr_data_in_r);
        assert_eq(recv_data, TestMemWriterDataPacket {
            data: uN[TEST_DATA_W]:0x313131,
            length: uN[TEST_ADDR_W]:3,
            last: false
        });

        let tok = send(tok, print_start_s, ());
        let (tok, _) = recv(tok, print_finish_r);

        send(tok, terminator, true);
    }
}
