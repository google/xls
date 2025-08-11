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

// This file contains the implementation of LiteralsBuffer responsible for
// storing data received either from RAW, RLE or Huffman literals decoder and
// sending it to CommandConstructor.

import std;

import xls.examples.ram;
import xls.modules.zstd.common as common;
import xls.modules.zstd.parallel_rams as parallel_rams;
import xls.modules.zstd.ram_printer as ram_printer;

type CopyOrMatchContent = common::CopyOrMatchContent;
type CopyOrMatchLength = common::CopyOrMatchLength;
type LitData = common::LitData;
type LitID = common::LitID;
type LitLength = common::LitLength;
type LiteralsBufferCtrl = common::LiteralsBufferCtrl;
type LiteralsData = common::LiteralsData;
type LiteralsDataWithSync = common::LiteralsDataWithSync;
type SequenceExecutorMessageType = common::SequenceExecutorMessageType;
type SequenceExecutorPacket = common::SequenceExecutorPacket;

type HistoryBufferPtr = parallel_rams::HistoryBufferPtr;
type RamNumber = parallel_rams::RamNumber;
type RamReadStart = parallel_rams::RamReadStart;
type RamRdRespHandlerData = parallel_rams::RamRdRespHandlerData;
type RamWrRespHandlerData = parallel_rams::RamWrRespHandlerData;
type RamWrRespHandlerResp = parallel_rams::RamWrRespHandlerResp;

// Constants calculated from RAM parameters
pub const RAM_NUM = parallel_rams::RAM_NUM;
const RAM_NUM_WIDTH = parallel_rams::RAM_NUM_WIDTH;
pub const RAM_DATA_WIDTH = common::SYMBOL_WIDTH + u32:1; // the +1 is used to store "last" flag
pub const RAM_WORD_PARTITION_SIZE = RAM_DATA_WIDTH;
pub const RAM_NUM_PARTITIONS = ram::num_partitions(RAM_WORD_PARTITION_SIZE, RAM_DATA_WIDTH);

// Literals data with last flag
type LiteralsWithLast = uN[RAM_DATA_WIDTH * RAM_NUM];

// RAM related constants common for tests
const TEST_HISTORY_BUFFER_SIZE_KB = u32:1;
const TEST_RAM_SIZE = parallel_rams::ram_size(TEST_HISTORY_BUFFER_SIZE_KB);
const TEST_RAM_ADDR_WIDTH = parallel_rams::ram_addr_width(TEST_HISTORY_BUFFER_SIZE_KB);
const TEST_INIT_HB_PTR_ADDR = u32:127;
const TEST_RAM_INITIALIZED = true;
const TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR = ram::SimultaneousReadWriteBehavior::READ_BEFORE_WRITE;

type TestRamAddr = bits[TEST_RAM_ADDR_WIDTH];
type TestWriteReq = ram::WriteReq<TEST_RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>;
type TestWriteResp = ram::WriteResp<TEST_RAM_ADDR_WIDTH>;
type TestReadReq = ram::ReadReq<TEST_RAM_ADDR_WIDTH, RAM_NUM_PARTITIONS>;
type TestReadResp = ram::ReadResp<RAM_DATA_WIDTH>;

struct LiteralsBufferMuxState {
    // Literals sync handling
    ctrl_last: bool,
    literals_id: LitID,
    // Received literals
    raw_literals_valid: bool,
    raw_literals_data: LiteralsDataWithSync,
    rle_literals_valid: bool,
    rle_literals_data: LiteralsDataWithSync,
    huff_literals_valid: bool,
    huff_literals_data: LiteralsDataWithSync,
}

struct LiteralsBufferWriterState<RAM_ADDR_WIDTH: u32> {
    // History Buffer handling
    hyp_ptr: HistoryBufferPtr<RAM_ADDR_WIDTH>,
    hb_len: uN[RAM_ADDR_WIDTH + RAM_NUM_WIDTH],
    literals_in_ram: uN[RAM_ADDR_WIDTH + RAM_NUM_WIDTH],
}

struct LiteralsBufferReaderState<RAM_ADDR_WIDTH: u32> {
    // History Buffer handling
    hyp_ptr: HistoryBufferPtr<RAM_ADDR_WIDTH>,
    hb_len: uN[RAM_ADDR_WIDTH + RAM_NUM_WIDTH],
    literals_in_ram: uN[RAM_ADDR_WIDTH + RAM_NUM_WIDTH],
    ctrl_last: bool,
    left_to_read: u32,
}

struct LiteralsBufferWriterToReaderSync {
    literals_written: LitLength,
}

struct LiteralsBufferReaderToWriterSync {
    literals_read: LitLength,
}

// PacketDecoder is responsible for receiving read bytes from RAMs response
// handler, removing the "literals_last" flag from each literal and adding this flag
// to the packet. It also validates the data.
proc PacketDecoder<RAM_ADDR_WIDTH: u32> {
    literals_in_r: chan<SequenceExecutorPacket<RAM_DATA_WIDTH>> in;
    literals_out_s: chan<SequenceExecutorPacket<common::SYMBOL_WIDTH>> out;
    buffer_sync_s: chan<LiteralsBufferReaderToWriterSync> out;

    config(
        literals_in_r: chan<SequenceExecutorPacket<RAM_DATA_WIDTH>> in,
        literals_out_s: chan<SequenceExecutorPacket<common::SYMBOL_WIDTH>> out,
        buffer_sync_s: chan<LiteralsBufferReaderToWriterSync> out,
    ) {
        (literals_in_r, literals_out_s, buffer_sync_s)
    }

    init { }

    next (state: ()) {
        let tok = join();
        let (tok, literals) = recv(tok, literals_in_r);

        // Strip flag last from literals
        let literals_data = for (i, data): (u32, CopyOrMatchContent) in u32:0..RAM_NUM {
            bit_slice_update(
                data,
                common::SYMBOL_WIDTH * i,
                (literals.content >> (RAM_DATA_WIDTH * i)) as uN[common::SYMBOL_WIDTH]
            )
        }(CopyOrMatchContent:0);

        let literals_lasts = for (i, lasts): (u32, bool[RAM_NUM]) in u32:0..RAM_NUM {
            let last = (literals.content >> (RAM_DATA_WIDTH * (i + u32:1) - u32:1)) as u1;
            update(lasts, i, last)
        }(bool[RAM_NUM]:[0, ...]);
        let literals_last = literals_lasts[literals.length - u64:1];

        // TODO: Restore this check after extending request to CommandConstructor
        // assert!(literals.last == literals_last, "Invalid packet");

        // Send literals data
        let literals_out = SequenceExecutorPacket<common::SYMBOL_WIDTH> {
            msg_type: SequenceExecutorMessageType::LITERAL,
            length: literals.length,
            content: literals_data,
            last: literals_last
        };
        let tok = send(tok, literals_out_s, literals_out);

        // Send sync data to buffer writer
        let tok = send(tok, buffer_sync_s, LiteralsBufferReaderToWriterSync {
            literals_read: literals.length as LitLength,
        });
    }
}

fn literals_content(literal: u8, last: u1, pos: u3) -> LiteralsWithLast {
    (
        literal as LiteralsWithLast |
        ((last as LiteralsWithLast) << common::SYMBOL_WIDTH)) << (RAM_DATA_WIDTH * (pos as u32)
    )
}


const TEST_LITERALS_IN: SequenceExecutorPacket<RAM_DATA_WIDTH>[4] = [
    SequenceExecutorPacket<RAM_DATA_WIDTH> {
        msg_type: SequenceExecutorMessageType::LITERAL,
        length: CopyOrMatchLength:1,
        content: literals_content(u8:0xAB, u1:0, u3:0),
        last: false,
    },
    SequenceExecutorPacket<RAM_DATA_WIDTH> {
        msg_type: SequenceExecutorMessageType::LITERAL,
        length: CopyOrMatchLength:2,
        content: (
            literals_content(u8:0x12, u1:0, u3:1) |
            literals_content(u8:0x34, u1:0, u3:0)
        ),
        last: false,
    },
    SequenceExecutorPacket<RAM_DATA_WIDTH> {
        msg_type: SequenceExecutorMessageType::LITERAL,
        length: CopyOrMatchLength:8,
        content: (
            literals_content(u8:0xFE, u1:0, u3:7) |
            literals_content(u8:0xDC, u1:0, u3:6) |
            literals_content(u8:0xBA, u1:0, u3:5) |
            literals_content(u8:0x98, u1:0, u3:4) |
            literals_content(u8:0x76, u1:0, u3:3) |
            literals_content(u8:0x54, u1:0, u3:2) |
            literals_content(u8:0x32, u1:0, u3:1) |
            literals_content(u8:0x10, u1:0, u3:0)
        ),
        last: false,
    },
    SequenceExecutorPacket<RAM_DATA_WIDTH> {
        msg_type: SequenceExecutorMessageType::LITERAL,
        length: CopyOrMatchLength:4,
        content: (
            literals_content(u8:0xAA, u1:1, u3:3) |
            literals_content(u8:0xBB, u1:1, u3:2) |
            literals_content(u8:0xCC, u1:0, u3:1) |
            literals_content(u8:0xDD, u1:0, u3:0)
        ),
        last: true,
    },
];

const TEST_LITERALS_OUT: SequenceExecutorPacket<common::SYMBOL_WIDTH>[4] = [
    SequenceExecutorPacket<common::SYMBOL_WIDTH> {
        msg_type: SequenceExecutorMessageType::LITERAL,
        length: CopyOrMatchLength:1,
        content: CopyOrMatchContent:0xAB,
        last: false,
    },
    SequenceExecutorPacket<common::SYMBOL_WIDTH> {
        msg_type: SequenceExecutorMessageType::LITERAL,
        length: CopyOrMatchLength:2,
        content: CopyOrMatchContent:0x1234,
        last: false,
    },
    SequenceExecutorPacket<common::SYMBOL_WIDTH> {
        msg_type: SequenceExecutorMessageType::LITERAL,
        length: CopyOrMatchLength:8,
        content: CopyOrMatchContent:0xFEDC_BA98_7654_3210,
        last: false,
    },
    SequenceExecutorPacket<common::SYMBOL_WIDTH> {
        msg_type: SequenceExecutorMessageType::LITERAL,
        length: CopyOrMatchLength:4,
        content: CopyOrMatchContent:0xAABB_CCDD,
        last: true,
    },
];

#[test_proc]
proc PacketDecoder_test {
    terminator: chan<bool> out;

    literals_in_s: chan<SequenceExecutorPacket<RAM_DATA_WIDTH>> out;
    literals_out_r: chan<SequenceExecutorPacket<common::SYMBOL_WIDTH>> in;
    buffer_sync_r: chan<LiteralsBufferReaderToWriterSync> in;

    config(terminator: chan<bool> out) {
        let (literals_in_s, literals_in_r) = chan<SequenceExecutorPacket<RAM_DATA_WIDTH>>("literals_in");
        let (literals_out_s, literals_out_r) = chan<SequenceExecutorPacket<common::SYMBOL_WIDTH>>("literals_out");
        let (buffer_sync_s, buffer_sync_r) = chan<LiteralsBufferReaderToWriterSync>("buffer_sync");

        spawn PacketDecoder<TEST_RAM_ADDR_WIDTH>(literals_in_r, literals_out_s, buffer_sync_s);

        (terminator, literals_in_s, literals_out_r, buffer_sync_r)
    }

    init { }

    next (state: ()) {
        let tok = join();
        let tok = for (i, tok): (u32, token) in u32:0..array_size(TEST_LITERALS_IN) {
            let tok = send(tok, literals_in_s, TEST_LITERALS_IN[i]);
            trace_fmt!("Sent #{} literals {:#x}", i, TEST_LITERALS_IN[i]);
            tok
        }(tok);

        let tok = for (i, tok): (u32, token) in u32:0..array_size(TEST_LITERALS_OUT) {
            let (tok, literals) = recv(tok, literals_out_r);
            trace_fmt!("Received #{} literals {:#x}", i, literals);
            assert_eq(TEST_LITERALS_OUT[i], literals);
            tok
        }(tok);

        send(tok, terminator, true);
    }
}

// Proc responsible for receiving literals from RAW, RLE and Huffman decoders
// and sending them to the writer in correct order.
proc LiteralsBufferMux {
    raw_literals_r: chan<LiteralsDataWithSync> in;
    rle_literals_r: chan<LiteralsDataWithSync> in;
    huff_literals_r: chan<LiteralsDataWithSync> in;

    out_literals_s: chan<LiteralsData> out;

    config(
        raw_literals_r: chan<LiteralsDataWithSync> in,
        rle_literals_r: chan<LiteralsDataWithSync> in,
        huff_literals_r: chan<LiteralsDataWithSync> in,
        out_literals_s: chan<LiteralsData> out,
    ) {
        (
            raw_literals_r, rle_literals_r, huff_literals_r,
            out_literals_s
        )
    }

    init { zero!<LiteralsBufferMuxState>() }

    next (state: LiteralsBufferMuxState) {
        let tok0 = join();
        // Receive literals

        let (tok1_0, raw_literals, raw_literals_valid) = recv_if_non_blocking(
            tok0, raw_literals_r, !state.raw_literals_valid, state.raw_literals_data
        );
        let (tok1_1, rle_literals, rle_literals_valid) = recv_if_non_blocking(
            tok0, rle_literals_r, !state.rle_literals_valid, state.rle_literals_data
        );
        let (tok1_2, huff_literals, huff_literals_valid) = recv_if_non_blocking(
            tok0, huff_literals_r, !state.huff_literals_valid, state.huff_literals_data
        );
        let state = LiteralsBufferMuxState {
            raw_literals_valid: state.raw_literals_valid || raw_literals_valid,
            raw_literals_data: raw_literals,
            rle_literals_valid: state.rle_literals_valid || rle_literals_valid,
            rle_literals_data: rle_literals,
            huff_literals_valid: state.huff_literals_valid || huff_literals_valid,
            huff_literals_data: huff_literals,
            ..state
        };

        let tok1 = join(tok1_0, tok1_1, tok1_2);

        // Select proper literals
        let sel_raw_literals = state.raw_literals_valid && state.raw_literals_data.id == state.literals_id;
        let sel_rle_literals = state.rle_literals_valid && state.rle_literals_data.id == state.literals_id;
        let sel_huff_literals = state.huff_literals_valid && state.huff_literals_data.id == state.literals_id;
        let literals_valid = sel_raw_literals || sel_rle_literals || sel_huff_literals;

        let (literals_data, state) = if (sel_raw_literals) {
            (
                state.raw_literals_data,
                LiteralsBufferMuxState { raw_literals_valid: false, ..state }
            )
        } else if (sel_rle_literals) {
            (
                state.rle_literals_data,
                LiteralsBufferMuxState { rle_literals_valid: false, ..state }
            )
        } else if (sel_huff_literals) {
            (
                state.huff_literals_data,
                LiteralsBufferMuxState { huff_literals_valid: false, ..state }
            )
        } else {
            (
                zero!<LiteralsDataWithSync>(),
                state
            )
        };

        let out_literals = LiteralsData {
            data: literals_data.data,
            length: literals_data.length,
            last: literals_data.last,
        };

        send_if(tok1, out_literals_s, literals_valid, out_literals);
        if literals_valid {
            trace_fmt!("[LiteralsBufferMux] literals: {:#x}", out_literals);
        } else {};

        let next_state = match (literals_data.last, literals_data.literals_last) {
            (true, false) => LiteralsBufferMuxState { literals_id: state.literals_id + LitID:1, ..state },
            (true, true) => zero!<LiteralsBufferMuxState>(),
            (_, _) => state,
        };

        next_state
    }
}

// Proc responsible for writing received literals to RAMs
proc LiteralsBufferWriter<
    HISTORY_BUFFER_SIZE_KB: u32,
    RAM_SIZE: u32 = {parallel_rams::ram_size(HISTORY_BUFFER_SIZE_KB)},
    RAM_ADDR_WIDTH: u32 = {parallel_rams::ram_addr_width(HISTORY_BUFFER_SIZE_KB)},
    INIT_HB_PTR_ADDR: u32 = {u32:0},
    INIT_HB_PTR_RAM: u32 = {u32:0},
    INIT_HB_LENGTH: u32 = {u32:0},
    RAM_SIZE_TOTAL: u32 = {RAM_SIZE * RAM_NUM}
> {
    type HistoryBufferLength = uN[RAM_ADDR_WIDTH + std::clog2(RAM_NUM)];
    type RamAddr = bits[RAM_ADDR_WIDTH];
    type State = LiteralsBufferWriterState<RAM_ADDR_WIDTH>;
    type WriteReq = ram::WriteReq<RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>;
    type WriteResp = ram::WriteResp;

    literals_r: chan<LiteralsData> in;

    ram_comp_input_s: chan<RamWrRespHandlerData<RAM_ADDR_WIDTH>> out;
    ram_comp_output_r: chan<RamWrRespHandlerResp<RAM_ADDR_WIDTH>> in;

    buffer_sync_r: chan<LiteralsBufferReaderToWriterSync> in;
    buffer_sync_s: chan<LiteralsBufferWriterToReaderSync> out;

    wr_req_m0_s: chan<WriteReq> out;
    wr_req_m1_s: chan<WriteReq> out;
    wr_req_m2_s: chan<WriteReq> out;
    wr_req_m3_s: chan<WriteReq> out;
    wr_req_m4_s: chan<WriteReq> out;
    wr_req_m5_s: chan<WriteReq> out;
    wr_req_m6_s: chan<WriteReq> out;
    wr_req_m7_s: chan<WriteReq> out;

    config (
        literals_r: chan<LiteralsData> in,
        buffer_sync_r: chan<LiteralsBufferReaderToWriterSync> in,
        buffer_sync_s: chan<LiteralsBufferWriterToReaderSync> out,
        wr_req_m0_s: chan<WriteReq> out,
        wr_req_m1_s: chan<WriteReq> out,
        wr_req_m2_s: chan<WriteReq> out,
        wr_req_m3_s: chan<WriteReq> out,
        wr_req_m4_s: chan<WriteReq> out,
        wr_req_m5_s: chan<WriteReq> out,
        wr_req_m6_s: chan<WriteReq> out,
        wr_req_m7_s: chan<WriteReq> out,
        wr_resp_m0_r: chan<WriteResp> in,
        wr_resp_m1_r: chan<WriteResp> in,
        wr_resp_m2_r: chan<WriteResp> in,
        wr_resp_m3_r: chan<WriteResp> in,
        wr_resp_m4_r: chan<WriteResp> in,
        wr_resp_m5_r: chan<WriteResp> in,
        wr_resp_m6_r: chan<WriteResp> in,
        wr_resp_m7_r: chan<WriteResp> in
    ) {
        let (ram_comp_input_s, ram_comp_input_r) = chan<RamWrRespHandlerData<RAM_ADDR_WIDTH>, u32:1>("ram_comp_input");
        let (ram_comp_output_s, ram_comp_output_r) = chan<RamWrRespHandlerResp<RAM_ADDR_WIDTH>, u32:1>("ram_comp_output");

        spawn parallel_rams::RamWrRespHandler<RAM_ADDR_WIDTH, RAM_DATA_WIDTH>(
            ram_comp_input_r, ram_comp_output_s,
            wr_resp_m0_r, wr_resp_m1_r, wr_resp_m2_r, wr_resp_m3_r,
            wr_resp_m4_r, wr_resp_m5_r, wr_resp_m6_r, wr_resp_m7_r,
        );

        (
            literals_r,
            ram_comp_input_s, ram_comp_output_r,
            buffer_sync_r, buffer_sync_s,
            wr_req_m0_s, wr_req_m1_s, wr_req_m2_s, wr_req_m3_s,
            wr_req_m4_s, wr_req_m5_s, wr_req_m6_s, wr_req_m7_s,
        )
    }

    init {
        type State = LiteralsBufferWriterState<RAM_ADDR_WIDTH>;
        let INIT_HB_PTR = HistoryBufferPtr {
            number: INIT_HB_PTR_RAM as RamNumber, addr: INIT_HB_PTR_ADDR as RamAddr
        };

        State {
            hyp_ptr: INIT_HB_PTR,
            hb_len: INIT_HB_LENGTH as uN[RAM_ADDR_WIDTH + RAM_NUM_WIDTH],
            ..zero!<State>()
        }
    }
    next (state: State) {
        let tok0 = join();
        // TODO: Remove this workaround when fixed: https://github.com/google/xls/issues/1368
        type State = LiteralsBufferWriterState<RAM_ADDR_WIDTH>;
        type WriteReq = ram::WriteReq<RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>;

        const ZERO_WRITE_REQS = WriteReq[RAM_NUM]:[zero!<WriteReq>(), ...];
        const RAM_REQ_MASK_NONE = bits[RAM_NUM_PARTITIONS]:0;


        // read from sync
        let (_, sync_data, sync_data_valid) = recv_non_blocking(tok0, buffer_sync_r, zero!<LiteralsBufferReaderToWriterSync>());

        if (sync_data_valid) {
            trace_fmt!("[LiteralsBufferWriter] Received buffer reader-to-writer sync data {:#x}", sync_data);
        } else {};

        // read literals
        let do_recv_literals = state.hb_len as u32 < HISTORY_BUFFER_SIZE_KB << u32:10;

        let (tok1, literals_data, literals_data_valid) = recv_if_non_blocking(tok0, literals_r, do_recv_literals, zero!<LiteralsData>());

        // write literals to RAM
        let packet_data = for (i, data): (u32, LiteralsWithLast) in u32:0..RAM_NUM {
            let last = if literals_data.length as u32 == i + u32:1 { (literals_data.last as LiteralsWithLast) << common::SYMBOL_WIDTH } else {LiteralsWithLast:0};
            let literal = (((literals_data.data >> (common::SYMBOL_WIDTH * i)) as uN[common::SYMBOL_WIDTH]) as LiteralsWithLast) | last;
            data | (literal << (RAM_DATA_WIDTH * i))
        }(LiteralsWithLast:0);

        let packet = SequenceExecutorPacket<RAM_DATA_WIDTH> {
            msg_type: SequenceExecutorMessageType::LITERAL,
            length: literals_data.length as CopyOrMatchLength,
            content: packet_data,
            last: literals_data.last,
        };
        let (write_reqs, new_hyp_ptr) = parallel_rams::literal_packet_to_write_reqs<
            HISTORY_BUFFER_SIZE_KB, RAM_ADDR_WIDTH, RAM_DATA_WIDTH
        >(
            state.hyp_ptr, packet
        );
        let hb_add = packet.length as HistoryBufferLength;
        let new_hb_len = std::mod_pow2(state.hb_len + hb_add, RAM_SIZE_TOTAL as HistoryBufferLength);

        let write_reqs = if (literals_data_valid) {
            write_reqs
        } else {
            ZERO_WRITE_REQS
        };

        // send write requests to RAMs
        let tok2_0 = send_if(tok1, wr_req_m0_s, write_reqs[0].mask != RAM_REQ_MASK_NONE, write_reqs[0]);
        let tok2_1 = send_if(tok1, wr_req_m1_s, write_reqs[1].mask != RAM_REQ_MASK_NONE, write_reqs[1]);
        let tok2_2 = send_if(tok1, wr_req_m2_s, write_reqs[2].mask != RAM_REQ_MASK_NONE, write_reqs[2]);
        let tok2_3 = send_if(tok1, wr_req_m3_s, write_reqs[3].mask != RAM_REQ_MASK_NONE, write_reqs[3]);
        let tok2_4 = send_if(tok1, wr_req_m4_s, write_reqs[4].mask != RAM_REQ_MASK_NONE, write_reqs[4]);
        let tok2_5 = send_if(tok1, wr_req_m5_s, write_reqs[5].mask != RAM_REQ_MASK_NONE, write_reqs[5]);
        let tok2_6 = send_if(tok1, wr_req_m6_s, write_reqs[6].mask != RAM_REQ_MASK_NONE, write_reqs[6]);
        let tok2_7 = send_if(tok1, wr_req_m7_s, write_reqs[7].mask != RAM_REQ_MASK_NONE, write_reqs[7]);

        let tok2 = join(tok2_0, tok2_1, tok2_2, tok2_3, tok2_4, tok2_5, tok2_6, tok2_7);

        // write completion
        let (do_write, wr_resp_handler_data) = parallel_rams::create_ram_wr_data(write_reqs, state.hyp_ptr);
        if do_write {trace_fmt!("[LiteralsBufferWriter] Sending request to RamWrRespHandler: {:#x}", wr_resp_handler_data);} else { };

        let tok3_0 = send_if(tok2, ram_comp_input_s, do_write, wr_resp_handler_data);

        let (tok3_1, comp_data, comp_data_valid) = recv_non_blocking(tok2, ram_comp_output_r, zero!<RamWrRespHandlerResp>());

        // update state
        let state = if (literals_data_valid) {
            State {
                hyp_ptr: new_hyp_ptr,
                hb_len: new_hb_len,
                ..state
            }
        } else {
            state
        };

        let state = if (comp_data_valid) {
            trace_fmt!("[LiteralsBufferWriter] COMP {:#x}", comp_data);
            State {
                literals_in_ram: state.literals_in_ram + comp_data.length as uN[RAM_ADDR_WIDTH + std::clog2(RAM_NUM)],
                ..state
            }
        } else {
            state
        };

        let state = if (sync_data_valid) {
            State {
                literals_in_ram: state.literals_in_ram - sync_data.literals_read as HistoryBufferLength,
                hb_len: state.hb_len - sync_data.literals_read as HistoryBufferLength,
                ..state
            }
        } else {
            state
        };

        // send sync
        let tok3 = join(tok3_0, tok3_1);

        let sync_data = LiteralsBufferWriterToReaderSync {
            literals_written: comp_data.length,
        };
        let tok4 = send_if(tok3, buffer_sync_s, comp_data_valid, sync_data);

        if (comp_data_valid) {
            trace_fmt!("[LiteralsBufferWriter] Sent buffer writer-to-reader sync data {:#x}", sync_data);
        } else {};

        state
    }
}

// Proc responsible for reading requestes literals from RAMs
proc LiteralsBufferReader<
    HISTORY_BUFFER_SIZE_KB: u32,
    RAM_SIZE: u32 = {parallel_rams::ram_size(HISTORY_BUFFER_SIZE_KB)},
    RAM_ADDR_WIDTH: u32 = {parallel_rams::ram_addr_width(HISTORY_BUFFER_SIZE_KB)},
    INIT_HB_PTR_ADDR: u32 = {u32:0},
    INIT_HB_PTR_RAM: u32 = {u32:0},
    INIT_HB_LENGTH: u32 = {u32:0},
    RAM_SIZE_TOTAL: u32 = {RAM_SIZE * RAM_NUM}
> {
    type HistoryBufferLength = uN[RAM_ADDR_WIDTH + std::clog2(RAM_NUM)];
    type RamAddr = bits[RAM_ADDR_WIDTH];
    type ReadReq = ram::ReadReq<RAM_ADDR_WIDTH, RAM_NUM_PARTITIONS>;
    type ReadResp = ram::ReadResp<RAM_DATA_WIDTH>;
    type State = LiteralsBufferReaderState<RAM_ADDR_WIDTH>;

    literals_buf_ctrl_r: chan<LiteralsBufferCtrl> in;
    literals_s: chan<SequenceExecutorPacket<common::SYMBOL_WIDTH>> out;

    ram_resp_input_s: chan<RamRdRespHandlerData> out;

    buffer_sync_r: chan<LiteralsBufferWriterToReaderSync> in;

    rd_req_m0_s: chan<ReadReq> out;
    rd_req_m1_s: chan<ReadReq> out;
    rd_req_m2_s: chan<ReadReq> out;
    rd_req_m3_s: chan<ReadReq> out;
    rd_req_m4_s: chan<ReadReq> out;
    rd_req_m5_s: chan<ReadReq> out;
    rd_req_m6_s: chan<ReadReq> out;
    rd_req_m7_s: chan<ReadReq> out;

    config (
        literals_buf_ctrl_r: chan<LiteralsBufferCtrl> in,
        literals_s: chan<SequenceExecutorPacket<common::SYMBOL_WIDTH>> out,
        buffer_sync_r: chan<LiteralsBufferWriterToReaderSync> in,
        buffer_sync_s: chan<LiteralsBufferReaderToWriterSync> out,
        rd_req_m0_s: chan<ReadReq> out,
        rd_req_m1_s: chan<ReadReq> out,
        rd_req_m2_s: chan<ReadReq> out,
        rd_req_m3_s: chan<ReadReq> out,
        rd_req_m4_s: chan<ReadReq> out,
        rd_req_m5_s: chan<ReadReq> out,
        rd_req_m6_s: chan<ReadReq> out,
        rd_req_m7_s: chan<ReadReq> out,
        rd_resp_m0_r: chan<ReadResp> in,
        rd_resp_m1_r: chan<ReadResp> in,
        rd_resp_m2_r: chan<ReadResp> in,
        rd_resp_m3_r: chan<ReadResp> in,
        rd_resp_m4_r: chan<ReadResp> in,
        rd_resp_m5_r: chan<ReadResp> in,
        rd_resp_m6_r: chan<ReadResp> in,
        rd_resp_m7_r: chan<ReadResp> in,
    ) {
        let (ram_resp_input_s, ram_resp_input_r) = chan<RamRdRespHandlerData, u32:1>("ram_resp_input");
        let (literals_enc_s, literals_enc_r) = chan<SequenceExecutorPacket<RAM_DATA_WIDTH>, u32:1>("literals_enc");

        spawn parallel_rams::RamRdRespHandler<RAM_DATA_WIDTH>(
            ram_resp_input_r, literals_enc_s,
            rd_resp_m0_r, rd_resp_m1_r, rd_resp_m2_r, rd_resp_m3_r,
            rd_resp_m4_r, rd_resp_m5_r, rd_resp_m6_r, rd_resp_m7_r,
        );

        spawn PacketDecoder<RAM_ADDR_WIDTH>(
            literals_enc_r, literals_s, buffer_sync_s
        );

        (
            literals_buf_ctrl_r,
            literals_s,
            ram_resp_input_s,
            buffer_sync_r,
            rd_req_m0_s, rd_req_m1_s, rd_req_m2_s, rd_req_m3_s,
            rd_req_m4_s, rd_req_m5_s, rd_req_m6_s, rd_req_m7_s,
        )
    }

    init {
        type State = LiteralsBufferReaderState<RAM_ADDR_WIDTH>;
        let INIT_HB_PTR = HistoryBufferPtr {
            number: INIT_HB_PTR_RAM as RamNumber, addr: INIT_HB_PTR_ADDR as RamAddr
        };

        State {
            hyp_ptr: INIT_HB_PTR,
            hb_len: INIT_HB_LENGTH as uN[RAM_ADDR_WIDTH + RAM_NUM_WIDTH],
            ..zero!<State>()
        }
    }

    next (state: State) {
        let tok0 = join();
        // TODO: Remove this workaround when fixed: https://github.com/google/xls/issues/1368
        type ReadReq = ram::ReadReq<RAM_ADDR_WIDTH, RAM_NUM_PARTITIONS>;
        type State = LiteralsBufferReaderState<RAM_ADDR_WIDTH>;

        const ZERO_READ_REQS = ReadReq[RAM_NUM]:[zero!<ReadReq>(), ...];
        const RAM_REQ_MASK_NONE = bits[RAM_NUM_PARTITIONS]:0;

        // read from ctrl
        let (tok1, literals_buf_ctrl, literals_buf_ctrl_valid) = recv_if_non_blocking(
            tok0, literals_buf_ctrl_r, state.left_to_read == u32:0, zero!<LiteralsBufferCtrl>()
        );
        let (left_to_read, ctrl_last) = if (literals_buf_ctrl_valid) {
            (
                literals_buf_ctrl.length,
                literals_buf_ctrl.last
            )
        } else {
            (
                state.left_to_read,
                state.ctrl_last
            )
        };

        // read literals from RAM
        // limit read to 8 literals
        let literals_to_read = if (left_to_read > (RAM_NUM as u32)) {
            RAM_NUM as u32
        } else {
            left_to_read
        };
        // if there is not enough literals in RAMs, don't read and wait for more literals
        let literals_to_read = if (literals_to_read > state.literals_in_ram as u32) {
            u32:0
        } else {
            literals_to_read
        };

        let packet = SequenceExecutorPacket<RAM_DATA_WIDTH> {
            msg_type: SequenceExecutorMessageType::LITERAL,
            length: literals_to_read as CopyOrMatchLength,
            content: state.hb_len as LiteralsWithLast,
            last: ctrl_last,
        };

        let (read_reqs, read_start, read_len, _, _) = parallel_rams::sequence_packet_to_read_reqs<
            HISTORY_BUFFER_SIZE_KB, RAM_ADDR_WIDTH, RAM_DATA_WIDTH
        >(
            state.hyp_ptr, packet, state.hb_len
        );

        let (read_reqs, read_start, state) = if (literals_to_read > u32:0) {
            (
                read_reqs,
                read_start,
                State {
                    hb_len: state.hb_len - literals_to_read as HistoryBufferLength,
                    literals_in_ram: state.literals_in_ram - literals_to_read as uN[RAM_ADDR_WIDTH + RAM_NUM_WIDTH],
                    left_to_read: left_to_read - literals_to_read,
                    ctrl_last: ctrl_last,
                    ..state
                },
            )
        } else {
            (
                ZERO_READ_REQS,
                RamReadStart:0,
                State {
                    left_to_read: left_to_read,
                    ctrl_last: ctrl_last,
                    ..state
                }
            )
        };

        // read requests
        let tok2_0 = send_if(tok1, rd_req_m0_s, read_reqs[0].mask != RAM_REQ_MASK_NONE, read_reqs[0]);
        let tok2_1 = send_if(tok1, rd_req_m1_s, read_reqs[1].mask != RAM_REQ_MASK_NONE, read_reqs[1]);
        let tok2_2 = send_if(tok1, rd_req_m2_s, read_reqs[2].mask != RAM_REQ_MASK_NONE, read_reqs[2]);
        let tok2_3 = send_if(tok1, rd_req_m3_s, read_reqs[3].mask != RAM_REQ_MASK_NONE, read_reqs[3]);
        let tok2_4 = send_if(tok1, rd_req_m4_s, read_reqs[4].mask != RAM_REQ_MASK_NONE, read_reqs[4]);
        let tok2_5 = send_if(tok1, rd_req_m5_s, read_reqs[5].mask != RAM_REQ_MASK_NONE, read_reqs[5]);
        let tok2_6 = send_if(tok1, rd_req_m6_s, read_reqs[6].mask != RAM_REQ_MASK_NONE, read_reqs[6]);
        let tok2_7 = send_if(tok1, rd_req_m7_s, read_reqs[7].mask != RAM_REQ_MASK_NONE, read_reqs[7]);

        let tok2 = join(tok2_0, tok2_1, tok2_2, tok2_3, tok2_4, tok2_5, tok2_6, tok2_7);
        let last_access = if (state.left_to_read > u32:0) {
          false
        } else {
          state.ctrl_last
        };

        let (do_read, rd_resp_handler_data) =
            parallel_rams::create_ram_rd_data<RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>(
                read_reqs, read_start, read_len, last_access, !last_access
            );
        if do_read {
            trace_fmt!("[LiteralsBufferReader] Sending request to RamRdRespHandler: {:#x}", rd_resp_handler_data);
        } else { };
        let tok3 = send_if(tok2, ram_resp_input_s, do_read, rd_resp_handler_data);

        // read from sync
        let (_, sync_data, sync_data_valid) = recv_non_blocking(tok0, buffer_sync_r, zero!<LiteralsBufferWriterToReaderSync>());

        if (sync_data_valid) {
            trace_fmt!("[LiteralsBufferReader] Received buffer writer-to-reader sync data {:#x}", sync_data);
        } else {};

        let state = if (sync_data_valid) {
            State {
                hyp_ptr: parallel_rams::hb_ptr_from_offset_forw<HISTORY_BUFFER_SIZE_KB>(
                    state.hyp_ptr, sync_data.literals_written as parallel_rams::Offset
                ),
                hb_len: state.hb_len + sync_data.literals_written as HistoryBufferLength,
                literals_in_ram: state.literals_in_ram + sync_data.literals_written as uN[RAM_ADDR_WIDTH + std::clog2(RAM_NUM)],
                ..state
            }
        } else {
            state
        };

        state
    }
}

pub proc LiteralsBuffer<
    HISTORY_BUFFER_SIZE_KB: u32,
    RAM_SIZE: u32 = {parallel_rams::ram_size(HISTORY_BUFFER_SIZE_KB)},
    RAM_ADDR_WIDTH: u32 = {parallel_rams::ram_addr_width(HISTORY_BUFFER_SIZE_KB)},
    INIT_HB_PTR_ADDR: u32 = {u32:0},
    INIT_HB_PTR_RAM: u32 = {u32:0},
    INIT_HB_LENGTH: u32 = {u32:0},
    RAM_SIZE_TOTAL: u32 = {RAM_SIZE * RAM_NUM}
> {
    type ReadReq = ram::ReadReq<RAM_ADDR_WIDTH, RAM_NUM_PARTITIONS>;
    type ReadResp = ram::ReadResp<RAM_DATA_WIDTH>;
    type WriteReq = ram::WriteReq<RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>;
    type WriteResp = ram::WriteResp;

    init { }

    config (
        raw_literals_r: chan<LiteralsDataWithSync> in,
        rle_literals_r: chan<LiteralsDataWithSync> in,
        huff_literals_r: chan<LiteralsDataWithSync> in,
        literals_buf_ctrl_r: chan<LiteralsBufferCtrl> in,
        literals_s: chan<SequenceExecutorPacket<common::SYMBOL_WIDTH>> out,
        rd_req_m0_s: chan<ReadReq> out,
        rd_req_m1_s: chan<ReadReq> out,
        rd_req_m2_s: chan<ReadReq> out,
        rd_req_m3_s: chan<ReadReq> out,
        rd_req_m4_s: chan<ReadReq> out,
        rd_req_m5_s: chan<ReadReq> out,
        rd_req_m6_s: chan<ReadReq> out,
        rd_req_m7_s: chan<ReadReq> out,
        rd_resp_m0_r: chan<ReadResp> in,
        rd_resp_m1_r: chan<ReadResp> in,
        rd_resp_m2_r: chan<ReadResp> in,
        rd_resp_m3_r: chan<ReadResp> in,
        rd_resp_m4_r: chan<ReadResp> in,
        rd_resp_m5_r: chan<ReadResp> in,
        rd_resp_m6_r: chan<ReadResp> in,
        rd_resp_m7_r: chan<ReadResp> in,
        wr_req_m0_s: chan<WriteReq> out,
        wr_req_m1_s: chan<WriteReq> out,
        wr_req_m2_s: chan<WriteReq> out,
        wr_req_m3_s: chan<WriteReq> out,
        wr_req_m4_s: chan<WriteReq> out,
        wr_req_m5_s: chan<WriteReq> out,
        wr_req_m6_s: chan<WriteReq> out,
        wr_req_m7_s: chan<WriteReq> out,
        wr_resp_m0_r: chan<WriteResp> in,
        wr_resp_m1_r: chan<WriteResp> in,
        wr_resp_m2_r: chan<WriteResp> in,
        wr_resp_m3_r: chan<WriteResp> in,
        wr_resp_m4_r: chan<WriteResp> in,
        wr_resp_m5_r: chan<WriteResp> in,
        wr_resp_m6_r: chan<WriteResp> in,
        wr_resp_m7_r: chan<WriteResp> in
    ) {
        type SyncWriterToReader = LiteralsBufferWriterToReaderSync;
        type SyncReaderToWriter = LiteralsBufferReaderToWriterSync;

        let (buffer_sync_writer_to_reader_s, buffer_sync_writer_to_reader_r) = chan<SyncWriterToReader, u32:1>("buffer_sync_writer_to_reader");
        let (buffer_sync_reader_to_writer_s, buffer_sync_reader_to_writer_r) = chan<SyncReaderToWriter, u32:1>("buffer_sync_reader_to_writer");
        let (sync_literals_s, sync_literals_r) = chan<LiteralsData, u32:1>("sync_literals");

        spawn LiteralsBufferMux (
            raw_literals_r, rle_literals_r, huff_literals_r,
            sync_literals_s
        );

        spawn LiteralsBufferWriter<
            HISTORY_BUFFER_SIZE_KB, RAM_SIZE, RAM_ADDR_WIDTH, INIT_HB_PTR_ADDR, INIT_HB_PTR_RAM, INIT_HB_LENGTH, RAM_SIZE_TOTAL
        > (
            sync_literals_r,
            buffer_sync_reader_to_writer_r, buffer_sync_writer_to_reader_s,
            wr_req_m0_s, wr_req_m1_s, wr_req_m2_s, wr_req_m3_s,
            wr_req_m4_s, wr_req_m5_s, wr_req_m6_s, wr_req_m7_s,
            wr_resp_m0_r, wr_resp_m1_r, wr_resp_m2_r, wr_resp_m3_r,
            wr_resp_m4_r, wr_resp_m5_r, wr_resp_m6_r, wr_resp_m7_r,
        );

        spawn LiteralsBufferReader<
            HISTORY_BUFFER_SIZE_KB, RAM_SIZE, RAM_ADDR_WIDTH, INIT_HB_PTR_ADDR, INIT_HB_PTR_RAM, INIT_HB_LENGTH, RAM_SIZE_TOTAL
        > (
            literals_buf_ctrl_r, literals_s,
            buffer_sync_writer_to_reader_r, buffer_sync_reader_to_writer_s,
            rd_req_m0_s, rd_req_m1_s, rd_req_m2_s, rd_req_m3_s,
            rd_req_m4_s, rd_req_m5_s, rd_req_m6_s, rd_req_m7_s,
            rd_resp_m0_r, rd_resp_m1_r, rd_resp_m2_r, rd_resp_m3_r,
            rd_resp_m4_r, rd_resp_m5_r, rd_resp_m6_r, rd_resp_m7_r,
        );
    }

    next (state: ()) { }
}

const INST_HISTORY_BUFFER_SIZE_KB = u32:64;
const INST_RAM_ADDR_WIDTH = parallel_rams::ram_addr_width(INST_HISTORY_BUFFER_SIZE_KB);
const INST_RAM_NUM_PARTITIONS = RAM_NUM_PARTITIONS;
const INST_RAM_DATA_WIDTH = RAM_DATA_WIDTH;
const INST_SYMBOL_WIDTH = common::SYMBOL_WIDTH;

pub proc LiteralsBufferInst {
    type ReadReq = ram::ReadReq<INST_RAM_ADDR_WIDTH, INST_RAM_NUM_PARTITIONS>;
    type ReadResp = ram::ReadResp<INST_RAM_DATA_WIDTH>;
    type WriteReq = ram::WriteReq<INST_RAM_ADDR_WIDTH, INST_RAM_DATA_WIDTH, INST_RAM_NUM_PARTITIONS>;
    type WriteResp = ram::WriteResp;

    init { }

    config (
        raw_literals_r: chan<LiteralsDataWithSync> in,
        rle_literals_r: chan<LiteralsDataWithSync> in,
        huff_literals_r: chan<LiteralsDataWithSync> in,
        literals_buf_ctrl_r: chan<LiteralsBufferCtrl> in,
        literals_s: chan<SequenceExecutorPacket<INST_SYMBOL_WIDTH>> out,
        rd_req_m0_s: chan<ReadReq> out,
        rd_req_m1_s: chan<ReadReq> out,
        rd_req_m2_s: chan<ReadReq> out,
        rd_req_m3_s: chan<ReadReq> out,
        rd_req_m4_s: chan<ReadReq> out,
        rd_req_m5_s: chan<ReadReq> out,
        rd_req_m6_s: chan<ReadReq> out,
        rd_req_m7_s: chan<ReadReq> out,
        rd_resp_m0_r: chan<ReadResp> in,
        rd_resp_m1_r: chan<ReadResp> in,
        rd_resp_m2_r: chan<ReadResp> in,
        rd_resp_m3_r: chan<ReadResp> in,
        rd_resp_m4_r: chan<ReadResp> in,
        rd_resp_m5_r: chan<ReadResp> in,
        rd_resp_m6_r: chan<ReadResp> in,
        rd_resp_m7_r: chan<ReadResp> in,
        wr_req_m0_s: chan<WriteReq> out,
        wr_req_m1_s: chan<WriteReq> out,
        wr_req_m2_s: chan<WriteReq> out,
        wr_req_m3_s: chan<WriteReq> out,
        wr_req_m4_s: chan<WriteReq> out,
        wr_req_m5_s: chan<WriteReq> out,
        wr_req_m6_s: chan<WriteReq> out,
        wr_req_m7_s: chan<WriteReq> out,
        wr_resp_m0_r: chan<WriteResp> in,
        wr_resp_m1_r: chan<WriteResp> in,
        wr_resp_m2_r: chan<WriteResp> in,
        wr_resp_m3_r: chan<WriteResp> in,
        wr_resp_m4_r: chan<WriteResp> in,
        wr_resp_m5_r: chan<WriteResp> in,
        wr_resp_m6_r: chan<WriteResp> in,
        wr_resp_m7_r: chan<WriteResp> in
    ) {
        spawn LiteralsBuffer<INST_HISTORY_BUFFER_SIZE_KB> (
            raw_literals_r, rle_literals_r, huff_literals_r,
            literals_buf_ctrl_r, literals_s,
            rd_req_m0_s, rd_req_m1_s, rd_req_m2_s, rd_req_m3_s,
            rd_req_m4_s, rd_req_m5_s, rd_req_m6_s, rd_req_m7_s,
            rd_resp_m0_r, rd_resp_m1_r, rd_resp_m2_r, rd_resp_m3_r,
            rd_resp_m4_r, rd_resp_m5_r, rd_resp_m6_r, rd_resp_m7_r,
            wr_req_m0_s, wr_req_m1_s, wr_req_m2_s, wr_req_m3_s,
            wr_req_m4_s, wr_req_m5_s, wr_req_m6_s, wr_req_m7_s,
            wr_resp_m0_r, wr_resp_m1_r, wr_resp_m2_r, wr_resp_m3_r,
            wr_resp_m4_r, wr_resp_m5_r, wr_resp_m6_r, wr_resp_m7_r,
        );
    }

    next (state: ()) { }
}

enum LiteralsChannel: u2 {
    RAW  = 0,
    RLE  = 1,
    HUFF = 2,
}

const TEST_LITERALS_DATA: (LiteralsChannel, LiteralsDataWithSync)[9] = [
    (LiteralsChannel::RAW, LiteralsDataWithSync {data: LitData:0x12_3456_789A, length: LitLength:5, last: true, id: LitID:0, literals_last: false}),
    (LiteralsChannel::RLE, LiteralsDataWithSync {data: LitData:0xBBBB_BBBB, length: LitLength:4, last: true, id: LitID:1, literals_last: false}),
    (LiteralsChannel::HUFF, LiteralsDataWithSync {data: LitData:0x64, length: LitLength:1, last: true, id: LitID:2, literals_last: false}),
    (LiteralsChannel::RLE, LiteralsDataWithSync {data: LitData:0xABCD_DCBA_1234_4321, length: LitLength:8, last: true, id: LitID:3, literals_last: false}),
    (LiteralsChannel::RAW, LiteralsDataWithSync {data: LitData:0x21_4365, length: LitLength:3, last: true, id: LitID:4, literals_last: false}),
    (LiteralsChannel::RLE, LiteralsDataWithSync {data: LitData:0xAA_BBBB_CCCC_DDDD, length: LitLength:7, last: true, id: LitID:5, literals_last: false}),
    (LiteralsChannel::RAW, LiteralsDataWithSync {data: LitData:0xDCBA_ABCD_1234_4321, length: LitLength:8, last: false, id: LitID:6, literals_last: false}),
    (LiteralsChannel::RAW, LiteralsDataWithSync {data: LitData:0x78, length: LitLength:1, last: true, id: LitID:6, literals_last: false}),
    (LiteralsChannel::HUFF, LiteralsDataWithSync {data: LitData:0x26, length: LitLength:1, last: true, id: LitID:7, literals_last: true}),
];

const TEST_BUFFER_CTRL: LiteralsBufferCtrl[11] = [
    // Literal #0
    LiteralsBufferCtrl {length: u32:2, last: false},
    LiteralsBufferCtrl {length: u32:1, last: false},
    LiteralsBufferCtrl {length: u32:2, last: true},
    // Literal #1
    LiteralsBufferCtrl {length: u32:4, last: true},
    // Literal #2
    LiteralsBufferCtrl {length: u32:1, last: true},
    // Literal #3
    LiteralsBufferCtrl {length: u32:8, last: true},
    // Literal #4
    LiteralsBufferCtrl {length: u32:3, last: true},
    // Literal #5
    LiteralsBufferCtrl {length: u32:7, last: true},
    // Literal #6
    LiteralsBufferCtrl {length: u32:8, last: false},
    LiteralsBufferCtrl {length: u32:1, last: true},
    // Literal #7
    LiteralsBufferCtrl {length: u32:1, last: true},
];

const TEST_EXPECTED_PACKETS: SequenceExecutorPacket<common::SYMBOL_WIDTH>[11] = [
    // Literal #0
    SequenceExecutorPacket<common::SYMBOL_WIDTH> {
        msg_type: SequenceExecutorMessageType::LITERAL,
        length: CopyOrMatchLength:2,
        content: CopyOrMatchContent:0x789A,
        last: false
    },
    SequenceExecutorPacket<common::SYMBOL_WIDTH> {
        msg_type: SequenceExecutorMessageType::LITERAL,
        length: CopyOrMatchLength:1,
        content: CopyOrMatchContent:0x56,
        last: false
    },
    SequenceExecutorPacket<common::SYMBOL_WIDTH> {
        msg_type: SequenceExecutorMessageType::LITERAL,
        length: CopyOrMatchLength:2,
        content: CopyOrMatchContent:0x1234,
        last: true
    },
    // Literal #1
    SequenceExecutorPacket<common::SYMBOL_WIDTH> {
        msg_type: SequenceExecutorMessageType::LITERAL,
        length: CopyOrMatchLength:4,
        content: CopyOrMatchContent:0xBBBB_BBBB,
        last: true
    },
    // Literal #2
    SequenceExecutorPacket<common::SYMBOL_WIDTH> {
        msg_type: SequenceExecutorMessageType::LITERAL,
        length: CopyOrMatchLength:1,
        content: CopyOrMatchContent:0x64,
        last: true
    },
    // Literal #3
    SequenceExecutorPacket<common::SYMBOL_WIDTH> {
        msg_type: SequenceExecutorMessageType::LITERAL,
        length: CopyOrMatchLength:8,
        content: CopyOrMatchContent:0xABCD_DCBA_1234_4321,
        last: true
    },
    // Literal #4
    SequenceExecutorPacket<common::SYMBOL_WIDTH> {
        msg_type: SequenceExecutorMessageType::LITERAL,
        length: CopyOrMatchLength:3,
        content: CopyOrMatchContent:0x21_4365,
        last: true
    },
    // Literal #5
    SequenceExecutorPacket<common::SYMBOL_WIDTH> {
        msg_type: SequenceExecutorMessageType::LITERAL,
        length: CopyOrMatchLength:7,
        content: CopyOrMatchContent:0xAA_BBBB_CCCC_DDDD,
        last: true
    },
    // Literal #6
    SequenceExecutorPacket<common::SYMBOL_WIDTH> {
        msg_type: SequenceExecutorMessageType::LITERAL,
        length: CopyOrMatchLength:8,
        content: CopyOrMatchContent:0xDCBA_ABCD_1234_4321,
        last: false
    },
    SequenceExecutorPacket<common::SYMBOL_WIDTH> {
        msg_type: SequenceExecutorMessageType::LITERAL,
        length: CopyOrMatchLength:1,
        content: CopyOrMatchContent:0x78,
        last: true
    },
    // Literal #7
    SequenceExecutorPacket<common::SYMBOL_WIDTH> {
        msg_type: SequenceExecutorMessageType::LITERAL,
        length: CopyOrMatchLength:1,
        content: CopyOrMatchContent:0x26,
        last: true
    }
];

#[test_proc]
proc LiteralsBuffer_test {
    terminator: chan<bool> out;

    raw_literals_s: chan<LiteralsDataWithSync> out;
    rle_literals_s: chan<LiteralsDataWithSync> out;
    huff_literals_s: chan<LiteralsDataWithSync> out;

    literals_buf_ctrl_s: chan<LiteralsBufferCtrl> out;
    literals_r: chan<SequenceExecutorPacket<common::SYMBOL_WIDTH>> in;

    print_start_s: chan<()> out;
    print_finish_r: chan<()> in;

    ram_rd_req_s: chan<TestReadReq>[RAM_NUM] out;
    ram_rd_resp_r: chan<TestReadResp>[RAM_NUM] in;
    ram_wr_req_s: chan<TestWriteReq>[RAM_NUM] out;
    ram_wr_resp_r: chan<TestWriteResp>[RAM_NUM] in;

    config(terminator: chan<bool> out) {
        let (raw_literals_s, raw_literals_r) = chan<LiteralsDataWithSync>("raw_literals");
        let (rle_literals_s, rle_literals_r) = chan<LiteralsDataWithSync>("rle_literals");
        let (huff_literals_s, huff_literals_r) = chan<LiteralsDataWithSync>("huff_literals");

        let (literals_buf_ctrl_s, literals_buf_ctrl_r) = chan<LiteralsBufferCtrl>("literals_buf_ctrl");
        let (literals_s, literals_r) = chan<SequenceExecutorPacket<common::SYMBOL_WIDTH>>("literals");

        let (print_start_s, print_start_r) = chan<()>("print_start");
        let (print_finish_s, print_finish_r) = chan<()>("print_finish");

        let (ram_rd_req_s,  ram_rd_req_r) = chan<TestReadReq>[RAM_NUM]("ram_rd_req");
        let (ram_rd_resp_s, ram_rd_resp_r) = chan<TestReadResp>[RAM_NUM]("ram_rd_resp");
        let (ram_wr_req_s,  ram_wr_req_r) = chan<TestWriteReq>[RAM_NUM]("ram_wr_req");
        let (ram_wr_resp_s, ram_wr_resp_r) = chan<TestWriteResp>[RAM_NUM]("ram_wr_resp");

        spawn LiteralsBuffer<
            TEST_HISTORY_BUFFER_SIZE_KB,
            TEST_RAM_SIZE,
            TEST_RAM_ADDR_WIDTH,
            TEST_INIT_HB_PTR_ADDR
        > (
            raw_literals_r, rle_literals_r, huff_literals_r,
            literals_buf_ctrl_r, literals_s,
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
            raw_literals_s, rle_literals_s, huff_literals_s,
            literals_buf_ctrl_s, literals_r,
            print_start_s, print_finish_r,
            ram_rd_req_s, ram_rd_resp_r,
            ram_wr_req_s, ram_wr_resp_r,
        )
    }

    init { }

    next (state: ()) {
        let tok = join();
        // send literals
        let tok = for ((i, test_literals_data), tok): ((u32, (LiteralsChannel, LiteralsDataWithSync)), token) in enumerate(TEST_LITERALS_DATA) {
            let literals_channel_s = match test_literals_data.0 {
                LiteralsChannel::RAW => raw_literals_s,
                LiteralsChannel::RLE => rle_literals_s,
                LiteralsChannel::HUFF => huff_literals_s,
            };
            let tok = send(tok, literals_channel_s, test_literals_data.1);
            trace_fmt!("Sent #{} literals {:#x} to channel {}", i + u32:1, test_literals_data.1, test_literals_data.0);
            tok
        }(tok);

        // send ctrl
        let tok = for ((i, test_buf_ctrl), tok): ((u32, LiteralsBufferCtrl), token) in enumerate(TEST_BUFFER_CTRL) {
            let tok = send(tok, literals_buf_ctrl_s, test_buf_ctrl);
            trace_fmt!("Send #{} ctrl {:#x}", i + u32:1, test_buf_ctrl);
            tok
        }(tok);

        // receive and check packets
        let tok = for ((i, test_exp_literals), tok): ((u32, SequenceExecutorPacket<common::SYMBOL_WIDTH>), token) in enumerate(TEST_EXPECTED_PACKETS) {
            let (tok, literals) = recv(tok, literals_r);
            trace_fmt!("Received #{} literals packet {:#x}", i + u32:1, literals);
            assert_eq(test_exp_literals, literals);
            tok
        }(tok);

        // print RAM content
        let tok = send(tok, print_start_s, ());
        let (tok, _) = recv(tok, print_finish_r);

        send(tok, terminator, true);
    }
}
