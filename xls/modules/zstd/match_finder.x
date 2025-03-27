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

// This file contains implementation of MatchFinder

import std;

import xls.examples.ram as ram;
import xls.modules.zstd.common as common;
import xls.modules.zstd.memory.mem_reader as mem_reader;
import xls.modules.zstd.memory.mem_writer as mem_writer;
import xls.modules.zstd.memory.axi as axi;
import xls.modules.zstd.memory.axi_ram as axi_ram;
import xls.modules.zstd.history_buffer as history_buffer;
import xls.modules.zstd.hash_table as hash_table;
import xls.modules.zstd.aligned_parallel_ram as aligned_parallel_ram;


const SYMBOL_WIDTH = common::SYMBOL_WIDTH;

struct ZstdParams<HT_SIZE_W: u32> {
    num_entries_log2: uN[HT_SIZE_W],
}

enum MatchFinderRespStatus : u1 {
    OK = 0,
    FAIL = 1,
}

struct MatchFinderSequence {
    literals_len: u16,
    match_offset: u16,
    match_len: u16,
}

pub struct MatchFinderReq<HT_SIZE_W: u32, ADDR_W: u32> {
    input_addr: uN[ADDR_W],
    input_size: uN[ADDR_W],
    output_lit_addr: uN[ADDR_W],
    output_seq_addr: uN[ADDR_W],
    zstd_params: ZstdParams<HT_SIZE_W>,
}

pub struct MatchFinderResp {
    status: MatchFinderRespStatus, // indicate the state of the operation
    lit_cnt: u32, // number of literals
    seq_cnt: u32, // number of sequences
}

struct MatchFinderInputBufferReq<ADDR_W: u32> {
    input_addr: uN[ADDR_W],
    input_size: uN[ADDR_W],
}

struct MatchFinderInputBufferResp {
    data: uN[SYMBOL_WIDTH],
    last: bool,
}

struct MatchFinderInputBufferState<
    ADDR_W: u32, DATA_W: u32,
    DATA_W_LOG2: u32 = {std::clog2(DATA_W + u32:1)},
> {
    addr: uN[ADDR_W],
    left_to_read: uN[ADDR_W],
    buffer: uN[DATA_W],
    buffer_len: uN[DATA_W_LOG2],
}

proc MatchFinderInputBuffer<
    ADDR_W: u32, DATA_W: u32,
    DATA_W_LOG2: u32 = {std::clog2(DATA_W + u32:1)},
> {
    type MemReaderReq = mem_reader::MemReaderReq<ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<DATA_W, ADDR_W>;
    type MemReaderStatus = mem_reader::MemReaderStatus;

    type Req = MatchFinderInputBufferReq<ADDR_W>;
    type Resp = MatchFinderInputBufferResp;
    type State = MatchFinderInputBufferState<ADDR_W, DATA_W>;

    req_r: chan<Req> in;
    next_r: chan<()> in;
    out_s: chan<Resp> out;

    // MemReader interface
    mem_rd_req_s: chan<MemReaderReq> out;
    mem_rd_resp_r: chan<MemReaderResp> in;

    config (
        req_r: chan<Req> in,
        next_r: chan<()> in,
        out_s: chan<Resp> out,
        mem_rd_req_s: chan<MemReaderReq> out,
        mem_rd_resp_r: chan<MemReaderResp> in,
    ) {
        (req_r, next_r, out_s, mem_rd_req_s, mem_rd_resp_r)
    }

    init { zero!<State>() }

    next (state: State) {
        // receive request
        let do_recv_req = state.left_to_read == uN[ADDR_W]:0;
        let (tok_req, req, req_valid) = recv_if_non_blocking(join(), req_r, do_recv_req, zero!<Req>());

        // send memory read request
        let mem_rd_req = MemReaderReq {
            addr: req.input_addr,
            length: req.input_size,
        };
        send_if(tok_req, mem_rd_req_s, req_valid, mem_rd_req);

        // receive memory read response
        let do_recv_mem_rd_resp = (state.left_to_read > uN[ADDR_W]:0) && (state.buffer_len == uN[DATA_W_LOG2]:0);
        let (_, mem_rd_resp, mem_rd_resp_valid) = recv_if_non_blocking(join(), mem_rd_resp_r, do_recv_mem_rd_resp, zero!<MemReaderResp>());

        // receive next and send data
        let do_recv_next = (state.left_to_read > uN[ADDR_W]:0 && (state.buffer_len > uN[DATA_W_LOG2]:0));
        let (tok_next, _, next_valid) = recv_if_non_blocking(join(), next_r, do_recv_next, ());

        let resp = Resp {
            data: state.buffer as uN[SYMBOL_WIDTH],
            last: state.left_to_read == uN[ADDR_W]:1,
        };
        send_if(tok_next, out_s, next_valid, resp);

        // update state
        if req_valid {
            State {
                addr: req.input_addr,
                left_to_read: req.input_size,
                ..zero!<State>()
            }
        } else if mem_rd_resp_valid {
            State {
                buffer: mem_rd_resp.data,
                buffer_len: DATA_W as uN[DATA_W_LOG2],
                ..state
            }
        } else if next_valid {
            State {
                left_to_read: state.left_to_read - uN[ADDR_W]:1,
                buffer: state.buffer >> SYMBOL_WIDTH,
                buffer_len: state.buffer_len - (SYMBOL_WIDTH as uN[DATA_W_LOG2]),
                ..state
            }
        } else {
            state
        }
    }
}

enum MatchFinderFSM: u5 {
    IDLE = 0,
    INPUT_NEXT = 1,
    INPUT_READ = 2,
    HASH_TABLE_REQ = 4,
    HASH_TABLE_RESP = 5,
    HISTORY_BUFFER_RESP = 6,
    OUTPUT_LITERAL_REQ_PACKET = 8,
    OUTPUT_LITERAL_RESP = 9,
    OUTPUT_SEQUENCE_REQ_PACKET = 10,
    OUTPUT_SEQUENCE_RESP = 11,
    INPUT_READ_NEXT_REQ = 15,
    INPUT_READ_NEXT_RESP = 16,
    SEND_RESP = 17,
    FAILURE = 18,
}

struct MatchFinderState<
    DATA_W: u32, ADDR_W: u32, HT_SIZE_W: u32, MIN_SEQ_LEN: u32,
    DATA_W_LOG2: u32 = {std::clog2(DATA_W + u32:1)}
> {
    fsm: MatchFinderFSM,
    req: MatchFinderReq<HT_SIZE_W, ADDR_W>,
    input_data: uN[SYMBOL_WIDTH],
    input_last: bool,
    input_addr_offset: uN[ADDR_W],
    output_lit_addr: uN[ADDR_W],
    output_seq_addr: uN[ADDR_W],
    lit_buffer: uN[MIN_SEQ_LEN][SYMBOL_WIDTH],
    lit_buffer_last: bool,
    literals_length: u16,
    match_offset: u16,
    match_length: u16,
    lit_cnt: u32,
    seq_cnt: u32,
}

proc MatchFinder<
    ADDR_W: u32, DATA_W: u32, HT_SIZE: u32, HB_SIZE: u32, MIN_SEQ_LEN: u32,
    DATA_W_LOG2: u32 = {std::clog2(DATA_W + u32:1)},

    HT_KEY_W: u32 = {SYMBOL_WIDTH},
    HT_VALUE_W: u32 = {SYMBOL_WIDTH + ADDR_W},
    HT_SIZE_W: u32 = {std::clog2(HT_SIZE + u32:1)},

    HB_DATA_W: u32 = {SYMBOL_WIDTH},
    HB_OFFSET_W: u32 = {std::clog2(HB_SIZE)},
> {
    type MemReaderReq = mem_reader::MemReaderReq<ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<DATA_W, ADDR_W>;
    type MemReaderStatus = mem_reader::MemReaderStatus;

    type InputBufferReq =  MatchFinderInputBufferReq<ADDR_W>;
    type InputBufferResp =  MatchFinderInputBufferResp;

    type MemWriterReq = mem_writer::MemWriterReq<ADDR_W>;
    type MemWriterResp = mem_writer::MemWriterResp;
    type MemWriterDataPacket = mem_writer::MemWriterDataPacket<DATA_W, ADDR_W>;
    type MemWriterRespStatus = mem_writer::MemWriterRespStatus;

    type Addr = uN[ADDR_W];

    type HashTableRdReq = hash_table::HashTableReadReq<HT_KEY_W, HT_SIZE, HT_SIZE_W>;
    type HashTableRdResp = hash_table::HashTableReadResp<HT_VALUE_W>;
    type HashTableWrReq = hash_table::HashTableWriteReq<HT_KEY_W, HT_VALUE_W, HT_SIZE, HT_SIZE_W>;
    type HashTableWrResp = hash_table::HashTableWriteResp;

    type HistoryBufferRdReq = history_buffer::HistoryBufferReadReq<HB_OFFSET_W>;
    type HistoryBufferRdResp = history_buffer::HistoryBufferReadResp<HB_DATA_W>;
    type HistoryBufferWrReq = history_buffer::HistoryBufferWriteReq<HB_DATA_W>;
    type HistoryBufferWrResp = history_buffer::HistoryBufferWriteResp;

    type Req = MatchFinderReq<HT_SIZE_W, ADDR_W>;
    type Resp = MatchFinderResp;
    type RespStatus = MatchFinderRespStatus;
    type State = MatchFinderState<DATA_W, ADDR_W, HT_SIZE_W, MIN_SEQ_LEN, DATA_W_LOG2>;

    type NumEntries = uN[HT_SIZE_W];
    type Key = uN[HT_KEY_W];
    type FSM = MatchFinderFSM;

    req_r: chan<Req> in;
    resp_s: chan<Resp> out;

    // InputBuffer interface
    inp_buf_req_s: chan<InputBufferReq> out;
    inp_buf_next_s: chan<()> out;
    inp_buf_out_r: chan<InputBufferResp> in;

    // MemWriter interface
    mem_wr_req_s: chan<MemWriterReq> out;
    mem_wr_packet_s: chan<MemWriterDataPacket> out;
    mem_wr_resp_r: chan<MemWriterResp> in;

    // HashTable interface
    ht_rd_req_s: chan<HashTableRdReq> out;
    ht_rd_resp_r: chan<HashTableRdResp> in;
    ht_wr_req_s: chan<HashTableWrReq> out;
    ht_wr_resp_r: chan<HashTableWrResp> in;

   // HistoryBuffer interface
    hb_rd_req_s: chan<HistoryBufferRdReq> out;
    hb_rd_resp_r: chan<HistoryBufferRdResp> in;
    hb_wr_req_s: chan<HistoryBufferWrReq> out;
    hb_wr_resp_r: chan<HistoryBufferWrResp> in;

    config (
        // Req & Resp
        req_r: chan<Req> in,
        resp_s: chan<Resp> out,

        // Access to input
        mem_rd_req_s: chan<MemReaderReq> out,
        mem_rd_resp_r: chan<MemReaderResp> in,

        // Output
        mem_wr_req_s: chan<MemWriterReq> out,
        mem_wr_packet_s: chan<MemWriterDataPacket> out,
        mem_wr_resp_r: chan<MemWriterResp> in,

        // HashTable RAM interface
        ht_rd_req_s: chan<HashTableRdReq> out,
        ht_rd_resp_r: chan<HashTableRdResp> in,
        ht_wr_req_s: chan<HashTableWrReq> out,
        ht_wr_resp_r: chan<HashTableWrResp> in,

        // HistoryBuffer RAM interface
        hb_rd_req_s: chan<HistoryBufferRdReq> out,
        hb_rd_resp_r: chan<HistoryBufferRdResp> in,
        hb_wr_req_s: chan<HistoryBufferWrReq> out,
        hb_wr_resp_r: chan<HistoryBufferWrResp> in,
    ) {
        let (inp_buf_req_s, inp_buf_req_r) = chan<InputBufferReq, u32:0>("inp_buf_req");
        let (inp_buf_next_s, inp_buf_next_r) = chan<(), u32:0>("inp_buf_next");
        let (inp_buf_out_s, inp_buf_out_r) = chan<InputBufferResp, u32:0>("inp_buf_out");

        spawn MatchFinderInputBuffer<ADDR_W, DATA_W> (
            inp_buf_req_r, inp_buf_next_r, inp_buf_out_s,
            mem_rd_req_s, mem_rd_resp_r,
        );

        (
            req_r, resp_s,
            inp_buf_req_s, inp_buf_next_s, inp_buf_out_r,
            mem_wr_req_s, mem_wr_packet_s, mem_wr_resp_r,
            ht_rd_req_s, ht_rd_resp_r, ht_wr_req_s, ht_wr_resp_r,
            hb_rd_req_s, hb_rd_resp_r, hb_wr_req_s, hb_wr_resp_r,
        )
    }

    init { zero!<State>() }

    next (state: State) {
        let tok0 = join();

        // [IDLE]
        let (tok1_0, req) = recv_if(tok0, req_r, state.fsm == FSM::IDLE, state.req);
        let inp_buf_req = InputBufferReq { input_addr: req.input_addr, input_size: req.input_size };
        let tok1_1 = send_if(tok0, inp_buf_req_s, state.fsm == FSM::IDLE, inp_buf_req);

        // [INPUT_NEXT]
        let do_send_inp_buf_next = state.fsm == FSM::INPUT_NEXT || state.fsm == FSM::INPUT_READ_NEXT_REQ;

        // [INPUT_READ]
        let do_recv_inp_buf_out = state.fsm == FSM::INPUT_READ || state.fsm == FSM::INPUT_READ_NEXT_RESP;
        let (tok1_2, inp_buf_out) = recv_if(tok0, inp_buf_out_r, do_recv_inp_buf_out, zero!<InputBufferResp>());

        // [HASH_TABLE_REQ]
        let ht_rd_req = HashTableRdReq {
            num_entries_log2: state.req.zstd_params.num_entries_log2,
            key: state.input_data,
        };
        let tok1_3 = send_if(tok0, ht_rd_req_s, state.fsm == FSM::HASH_TABLE_REQ, ht_rd_req);

        // [HASH_TABLE_RESP]
        let (tok1_4, ht_rd_resp) = recv_if(tok0, ht_rd_resp_r, state.fsm == FSM::HASH_TABLE_RESP, zero!<HashTableRdResp>());
        let ht_is_match = ht_rd_resp.is_match && (((ht_rd_resp.value >> ADDR_W) as uN[SYMBOL_WIDTH]) == state.input_data);
        let hb_rd_req = if ht_is_match {
            HistoryBufferRdReq {
                offset: (state.req.input_addr + state.input_addr_offset - (ht_rd_resp.value as uN[ADDR_W])) as uN[HB_OFFSET_W],
            }
        } else {
            zero!<HistoryBufferRdReq>()
        };
        let tok1_4 = send_if(tok1_4, hb_rd_req_s, ht_is_match, hb_rd_req);

        // [HISTORY_BUFFER_RESP] | [HISTORY_BUFFER_NEXT_RESP]
        let (tok1_5, hb_rd_resp) = recv_if(tok0, hb_rd_resp_r, state.fsm == FSM::HISTORY_BUFFER_RESP, zero!<HistoryBufferRdResp>());
        let (tok1_2, inp_buf_out) = recv_if(tok0, inp_buf_out_r, state.fsm == FSM::HISTORY_BUFFER_RESP, zero!<InputBufferResp>());

        let is_next_match = state.input_data == hb_rd_resp.data;
        let tok1_2 = send_if(tok0, inp_buf_next_s, do_send_inp_buf_next || ht_is_match || is_next_match, ());

        // write entry in hash table
        let do_save_entry = (
            (state.fsm == FSM::HASH_TABLE_RESP && !ht_is_match) ||
            (state.fsm == FSM::HISTORY_BUFFER_RESP && !is_next_match)
        );
        let ht_wr_req =  HashTableWrReq {
            num_entries_log2: state.req.zstd_params.num_entries_log2,
            key: state.input_data,
            value: (
                (state.input_data) ++
                (state.req.input_addr + state.input_addr_offset)
            ),
        };
        let tok1_2 = send_if(tok1_2, ht_wr_req_s, do_save_entry, ht_wr_req);
        // write entry in history buffer
        let hb_wr_req = HistoryBufferWrReq {
            data: state.input_data
        };
        let tok1_2 = send_if(tok1_2, hb_wr_req_s, do_save_entry, hb_wr_req);

        // [OUTPUT_LITERAL_REQ_PACKET]
        let lit_mem_wr_req = MemWriterReq {
            addr: state.output_lit_addr,
            length: uN[ADDR_W]:1,
        };
        let tok1_7 = send_if(tok0, mem_wr_req_s, state.fsm == FSM::OUTPUT_LITERAL_REQ_PACKET, lit_mem_wr_req);

        let lit_mem_wr_packet = MemWriterDataPacket {
            data: state.input_data as uN[DATA_W],
            length: uN[ADDR_W]:1,
            last: true,
        };
        let tok1_8 = send_if(tok0, mem_wr_packet_s, state.fsm == FSM::OUTPUT_LITERAL_REQ_PACKET, lit_mem_wr_packet);

        // [OUTPUT_LITERAL_RESP]
        let (tok1_9, lit_mem_wr_resp) = recv_if(tok0, mem_wr_resp_r, state.fsm == FSM::OUTPUT_LITERAL_RESP, zero!<MemWriterResp>());

        // [OUTPUT_SEQUENCE_REQ_PACKET]
        let seq_mem_wr_req = MemWriterReq {
            addr: state.output_seq_addr,
            length: uN[ADDR_W]:6,
        };
        let tok1_9 = send_if(tok0, mem_wr_req_s, state.fsm == FSM::OUTPUT_SEQUENCE_REQ_PACKET, seq_mem_wr_req);

        let seq_mem_wr_packet = MemWriterDataPacket {
            data: (state.literals_length ++ state.match_offset ++ state.match_length) as uN[DATA_W],
            length: uN[ADDR_W]:6,
            last: true,
        };

        let tok1_10 = send_if(tok0, mem_wr_packet_s, state.fsm == FSM::OUTPUT_SEQUENCE_REQ_PACKET, seq_mem_wr_packet);

        // [OUTPUT_SEQUENCE_RESP]
        let (tok1_11, seq_mem_wr_resp) = recv_if(tok0, mem_wr_resp_r, state.fsm == FSM::OUTPUT_SEQUENCE_RESP, zero!<MemWriterResp>());

        // [INPUT_READ_NEXT_REQ]
        let hb_rd_req = HistoryBufferRdReq {
            offset: (state.match_offset + state.match_length) as uN[HB_OFFSET_W],
        };
        let tok1_15 = send_if(tok0, hb_rd_req_s, state.fsm == FSM::INPUT_READ_NEXT_REQ && !ht_is_match, hb_rd_req);

        // [SEND_RESP]
        let resp = Resp {
            status: RespStatus::OK,
            lit_cnt: state.lit_cnt,
            seq_cnt: state.seq_cnt,
        };

        let tok1_19 = send_if(tok0, resp_s, state.fsm == FSM::SEND_RESP, resp);

        // [FAILURE]
        let fail_resp = MatchFinderResp {
            status: MatchFinderRespStatus::FAIL,
            lit_cnt: u32:0,
            seq_cnt: u32:0,
        };
        let tok1_4 = send_if(tok0, resp_s, state.fsm == FSM::FAILURE, fail_resp);

        let (tok, _, _) = recv_if_non_blocking(tok0, ht_wr_resp_r, false, zero!<HashTableWrResp>());
        let (tok, _, _) = recv_if_non_blocking(tok0, hb_wr_resp_r, false, zero!<HistoryBufferWrResp>());

        match state.fsm {
            FSM::IDLE => {
                trace_fmt!("[IDLE] Received match finder request {:#x}", req);
                State {
                    fsm: FSM::INPUT_NEXT,
                    req: req,
                    input_addr_offset: uN[ADDR_W]:0,
                    output_lit_addr: req.output_lit_addr,
                    output_seq_addr: req.output_seq_addr,
                    ..zero!<State>()
                }
            },

            FSM::INPUT_NEXT => {
                trace_fmt!("[INPUT_NEXT] Sent next to input buffer");
                State {
                    fsm: FSM::INPUT_READ,
                    input_addr_offset: state.input_addr_offset + uN[ADDR_W]:1,
                    ..state
                }
            },

            FSM::INPUT_READ => {
                trace_fmt!("[INPUT_READ] Received input {:#x}", inp_buf_out);
                State {
                    fsm: FSM::HASH_TABLE_REQ,
                    input_data: inp_buf_out.data,
                    input_last: inp_buf_out.last,
                    ..state
                }
            },

            FSM::HASH_TABLE_REQ => {
                trace_fmt!("[HASH_TABLE_REQ] Sent HT read request {:#x}", ht_rd_req);
                State {
                    fsm: FSM::HASH_TABLE_RESP,
                    ..state
                }
            },

            FSM::HASH_TABLE_RESP => {
                trace_fmt!("[HASH_TABLE_RESP] Received HT read respose {:#x}", ht_rd_resp);
                if ht_is_match {
                    trace_fmt!("[HASH_TABLE_RESP] Sent HB read request {:#x}", hb_rd_req);
                    State {
                        fsm: FSM::HISTORY_BUFFER_RESP,
                        match_offset: hb_rd_req.offset as u16,
                        ..state
                    }
                } else {
                    State {
                        fsm: FSM::OUTPUT_LITERAL_REQ_PACKET,
                        ..state
                    }
                }
            },

            FSM::HISTORY_BUFFER_RESP => {
                trace_fmt!("[HISTORY_BUFFER_RESP] Received HB read response {:#x}", hb_rd_resp);
                trace_fmt!("[HISTORY_BUFFER_RESP] Next symbol {:#x}", state.input_data);
                if is_next_match {
                    State {
                        fsm: FSM::INPUT_NEXT,
                        match_length: state.match_length + u16:1,
                        ..state
                    }
                } else if state.match_length as u32 < MIN_SEQ_LEN {
                    State {
                        fsm: FSM::OUTPUT_LITERAL_REQ_PACKET,
                        ..state
                    }
                } else {
                    State {
                        fsm: FSM::OUTPUT_SEQUENCE_REQ_PACKET,
                        ..state
                    }
                }
            },

            FSM::OUTPUT_LITERAL_REQ_PACKET => {
                trace_fmt!("[OUTPUT_LITERAL_REQ_PACKET] Sent mem write request {:#x}", lit_mem_wr_req);
                trace_fmt!("[OUTPUT_LITERAL_REQ_PACKET] Sent mem write data {:#x}", lit_mem_wr_packet);
                State {
                    fsm: FSM::OUTPUT_LITERAL_RESP,
                    ..state
                }
            },

            FSM::OUTPUT_LITERAL_RESP => {
                trace_fmt!("[OUTPUT_LITERAL_RESP] Received mem write response {:#x}", lit_mem_wr_resp);
                if state.input_last {
                    State {
                        fsm: FSM::OUTPUT_SEQUENCE_REQ_PACKET,
                        output_lit_addr: state.output_lit_addr + uN[ADDR_W]:1,
                        literals_length: state.literals_length + u16:1,
                        ..state
                    }
                } else {
                    State {
                        fsm: FSM::INPUT_NEXT,
                        output_lit_addr: state.output_lit_addr + uN[ADDR_W]:1,
                        literals_length: state.literals_length + u16:1,
                        ..state
                    }
                }
            },

            FSM::OUTPUT_SEQUENCE_REQ_PACKET => {
                trace_fmt!("[OUTPUT_SEQUENCE_REQ_PACKET] Sent mem write request {:#x}", seq_mem_wr_req);
                trace_fmt!("[OUTPUT_SEQUENCE_REQ_PACKET] Sent mem write data {:#x}", seq_mem_wr_packet);
                State {
                    fsm: FSM::OUTPUT_SEQUENCE_RESP,
                    lit_cnt: state.lit_cnt + (state.literals_length as u32),
                    seq_cnt: state.seq_cnt + (state.match_length as u32),
                    ..state
                }
            },

            FSM::OUTPUT_SEQUENCE_RESP => {
                trace_fmt!("[OUTPUT_SEQUENCE_RESP] Received mem write response {:#x}", seq_mem_wr_resp);
                if state.input_last {
                    State {
                        fsm: FSM::SEND_RESP,
                        output_seq_addr: state.output_seq_addr + uN[ADDR_W]:6,
                        ..state
                    }
                } else {
                    State {
                        fsm: FSM::HASH_TABLE_REQ, // here we reuse last response, which was not matched
                        output_seq_addr: state.output_seq_addr + uN[ADDR_W]:6,
                        literals_length: u16:0,
                        match_length: u16:0,
                        ..state
                    }
                }
            },

            FSM::INPUT_READ_NEXT_REQ => {
                trace_fmt!("[INPUT_NEXT] Sent next to input buffer");
                State {
                    fsm: FSM::INPUT_READ_NEXT_RESP,
                    input_addr_offset: state.input_addr_offset + uN[ADDR_W]:1,
                    ..state
                }
            },

            FSM::INPUT_READ_NEXT_RESP => {
                trace_fmt!("[INPUT_READ_NEXT_RESP ] Received input {:#x}", inp_buf_out);
                State {
                    fsm: FSM::HISTORY_BUFFER_RESP,
                    input_data: inp_buf_out.data,
                    input_last: inp_buf_out.last,
                    ..state
                }
            },

            FSM::SEND_RESP => {
                trace_fmt!("[SEND_RESP] Sent response {:#x}", resp);
                State {
                    fsm: FSM::IDLE,
                    ..zero!<State>()
                }
            },

            FSM::FAILURE => {
                trace_fmt!("[FAILURE] !!!");
                State {
                    fsm: FSM::IDLE,
                    ..zero!<State>()
                }
            },

            _ => state,
        }
    }
}

const INST_ADDR_W = u32:32;
const INST_DATA_W = u32:64;
const INST_MIN_SEQ_LEN = u32:3;
const INST_DATA_W_LOG2 = std::clog2(INST_DATA_W + u32:1);

const INST_HT_SIZE = u32:512;
const INST_HT_SIZE_W = std::clog2(INST_HT_SIZE + u32:1);
const INST_HT_KEY_W = SYMBOL_WIDTH;
const INST_HT_VALUE_W = SYMBOL_WIDTH + INST_ADDR_W; // original symbol + address
const INST_HB_DATA_W = SYMBOL_WIDTH;
const INST_HB_SIZE = u32:1024;
const INST_HB_OFFSET_W = std::clog2(INST_HB_SIZE);

proc MatchFinderInst {
    type MemReaderReq = mem_reader::MemReaderReq<INST_ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<INST_DATA_W, INST_ADDR_W>;

    type MemWriterReq = mem_writer::MemWriterReq<INST_ADDR_W>;
    type MemWriterResp = mem_writer::MemWriterResp;
    type MemWriterDataPacket = mem_writer::MemWriterDataPacket<INST_DATA_W, INST_ADDR_W>;

    type HashTableRdReq = hash_table::HashTableReadReq<INST_HT_KEY_W, INST_HT_SIZE, INST_HT_SIZE_W>;
    type HashTableRdResp = hash_table::HashTableReadResp<INST_HT_VALUE_W>;
    type HashTableWrReq = hash_table::HashTableWriteReq<INST_HT_KEY_W, INST_HT_VALUE_W, INST_HT_SIZE, INST_HT_SIZE_W>;
    type HashTableWrResp = hash_table::HashTableWriteResp;

    type HistoryBufferRdReq = history_buffer::HistoryBufferReadReq<INST_HB_OFFSET_W>;
    type HistoryBufferRdResp = history_buffer::HistoryBufferReadResp<INST_HB_DATA_W>;
    type HistoryBufferWrReq = history_buffer::HistoryBufferWriteReq<INST_HB_DATA_W>;
    type HistoryBufferWrResp = history_buffer::HistoryBufferWriteResp;

    type Req = MatchFinderReq<INST_HT_SIZE_W, INST_ADDR_W>;
    type Resp = MatchFinderResp;

    config (
        // Req & Resp
        req_r: chan<Req> in,
        resp_s: chan<Resp> out,

        // Access to input
        mem_rd_req_s: chan<MemReaderReq> out,
        mem_rd_resp_r: chan<MemReaderResp> in,

        // Output
        mem_wr_req_s: chan<MemWriterReq> out,
        mem_wr_packet_s: chan<MemWriterDataPacket> out,
        mem_wr_resp_r: chan<MemWriterResp> in,

        // HashTable RAM interface
        ht_rd_req_s: chan<HashTableRdReq> out,
        ht_rd_resp_r: chan<HashTableRdResp> in,
        ht_wr_req_s: chan<HashTableWrReq> out,
        ht_wr_resp_r: chan<HashTableWrResp> in,

        // HistoryBuffer RAM interface
        hb_rd_req_s: chan<HistoryBufferRdReq> out,
        hb_rd_resp_r: chan<HistoryBufferRdResp> in,
        hb_wr_req_s: chan<HistoryBufferWrReq> out,
        hb_wr_resp_r: chan<HistoryBufferWrResp> in,
    ) {
        spawn MatchFinder<
            INST_ADDR_W, INST_DATA_W, INST_HT_SIZE, INST_HB_SIZE, INST_MIN_SEQ_LEN,
            INST_DATA_W_LOG2,
            INST_HT_KEY_W, INST_HT_VALUE_W, INST_HT_SIZE_W,
            INST_HB_DATA_W, INST_HB_OFFSET_W,
        >(
            req_r, resp_s,
            mem_rd_req_s, mem_rd_resp_r,
            mem_wr_req_s, mem_wr_packet_s, mem_wr_resp_r,
            ht_rd_req_s, ht_rd_resp_r, ht_wr_req_s, ht_wr_resp_r,
            hb_rd_req_s, hb_rd_resp_r, hb_wr_req_s, hb_wr_resp_r,
        );
    }

    init {}

    next (state: ()) {}
}
const TEST_ADDR_W = u32:32;
const TEST_DATA_W = u32:64;
const TEST_MIN_SEQ_LEN = u32:3;
const TEST_HT_SIZE = u32:512;
const TEST_HB_SIZE = u32:1024;
const TEST_DATA_W_LOG2 = std::clog2(TEST_DATA_W + u32:1);
const TEST_DEST_W = u32:8;
const TEST_ID_W = u32:8;

const TEST_HT_KEY_W = SYMBOL_WIDTH;
const TEST_HT_VALUE_W = SYMBOL_WIDTH + TEST_ADDR_W; // original symbol + address
const TEST_HT_HASH_W = std::clog2(TEST_HT_SIZE);
const TEST_HT_RAM_DATA_W = TEST_HT_VALUE_W + u32:1; // value + valid
const TEST_HT_RAM_WORD_PARTITION_SIZE = u32:1;
const TEST_HT_RAM_NUM_PARTITIONS = ram::num_partitions(TEST_HT_RAM_WORD_PARTITION_SIZE, TEST_HT_RAM_DATA_W);
const TEST_HT_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR = ram::SimultaneousReadWriteBehavior::READ_BEFORE_WRITE;
const TEST_HT_RAM_INITIALIZED = true;
const TEST_HT_SIZE_W = std::clog2(TEST_HT_SIZE + u32:1);

const TEST_HB_RAM_NUM = u32:8;
const TEST_HB_DATA_W = SYMBOL_WIDTH;
const TEST_HB_OFFSET_W = std::clog2(TEST_HB_SIZE);
const TEST_HB_RAM_SIZE = TEST_HB_SIZE / TEST_HB_RAM_NUM;
const TEST_HB_RAM_DATA_W = SYMBOL_WIDTH / TEST_HB_RAM_NUM;
const TEST_HB_RAM_ADDR_W = std::clog2(TEST_HB_RAM_SIZE);
const TEST_HB_RAM_PARTITION_SIZE = TEST_HB_RAM_DATA_W;
const TEST_HB_RAM_NUM_PARTITIONS = ram::num_partitions(TEST_HB_RAM_PARTITION_SIZE, TEST_HB_RAM_DATA_W);
const TEST_HB_RAM_SIMULTANEOUS_RW_BEHAVIOR = ram::SimultaneousReadWriteBehavior::READ_BEFORE_WRITE;
const TEST_HB_RAM_INITIALIZED = true;

const TEST_RAM_DATA_W = TEST_DATA_W;
const TEST_RAM_SIZE = u32:1024;
const TEST_RAM_ADDR_W = TEST_ADDR_W;
const TEST_RAM_PARTITION_SIZE = TEST_RAM_DATA_W / u32:8;
const TEST_RAM_NUM_PARTITIONS = ram::num_partitions(TEST_RAM_PARTITION_SIZE, TEST_RAM_DATA_W);
const TEST_RAM_SIMULTANEOUS_RW_BEHAVIOR = ram::SimultaneousReadWriteBehavior::READ_BEFORE_WRITE;
const TEST_RAM_INITIALIZED = true;
const TEST_RAM_ASSERT_VALID_READ = true;

const TEST_OUTPUT_LIT_ADDR = uN[TEST_ADDR_W]:0x100;
const TEST_OUTPUT_SEQ_ADDR = uN[TEST_ADDR_W]:0x200;
const TEST_OUTPUT_ADDR_MASK = uN[TEST_ADDR_W]:0xF00;

// Test data
// 010A_0B0C_0102_0104_0A0B_0C05_0A0B_0C09
//    ^- ----           ^--- --   ^--- --
// Expected output
//   literals: 010A_0B0C_0102_0104_0509
//   sequences: (8, 7, 3), (1, 4, 3), (1, 0, 0)

const TEST_DATA = uN[TEST_RAM_DATA_W][2]:[
    u64:0x0401_0201_0C0B_0A01,
    u64:0x090C_0B0A_050C_0B0A,
];

const TEST_LITERALS = u8[10]:[
    u8:0x09, u8:0x05, u8:0x04, u8:0x01, u8:0x02, u8:0x01, u8:0x0C, u8:0x0B, u8:0x0A, u8:0x01,
];

const TEST_SEQUENCES = MatchFinderSequence[3]:[
    MatchFinderSequence {
        literals_len: u16:8,
        match_offset: u16:7,
        match_len: u16:3,
    },
    MatchFinderSequence {
        literals_len: u16:1,
        match_offset: u16:4,
        match_len: u16:3,
    },
    MatchFinderSequence {
        literals_len: u16:1,
        match_offset: u16:0,
        match_len: u16:0,
    },
];

struct TestState {
    iteration: u32,
    lit_buffer: uN[SYMBOL_WIDTH][256],
    seq_buffer: MatchFinderSequence[32],
    wr_addr: uN[TEST_ADDR_W],
    wr_offset: uN[TEST_ADDR_W],
    wr_len: uN[TEST_ADDR_W],
}

#[test_proc]
proc MatchFinderTest {

    // Memory Reader + Input

    type MemReaderReq = mem_reader::MemReaderReq<TEST_ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<TEST_DATA_W, TEST_ADDR_W>;

    type InputBufferRamRdReq = ram::ReadReq<TEST_RAM_ADDR_W, TEST_RAM_NUM_PARTITIONS>;
    type InputBufferRamRdResp = ram::ReadResp<TEST_RAM_DATA_W>;
    type InputBufferRamWrReq = ram::WriteReq<TEST_RAM_ADDR_W, TEST_RAM_DATA_W, TEST_RAM_NUM_PARTITIONS>;
    type InputBufferRamWrResp = ram::WriteResp;

    type AxiAr = axi::AxiAr<TEST_ADDR_W, TEST_ID_W>;
    type AxiR = axi::AxiR<TEST_DATA_W, TEST_ID_W>;

    // Memory Writer

    type MemWriterReq = mem_writer::MemWriterReq<TEST_ADDR_W>;
    type MemWriterResp = mem_writer::MemWriterResp;
    type MemWriterDataPacket = mem_writer::MemWriterDataPacket<TEST_DATA_W, TEST_ADDR_W>;

    // Hash Table

    type HashTableRdReq = hash_table::HashTableReadReq<TEST_HT_KEY_W, TEST_HT_SIZE, TEST_HT_SIZE_W>;
    type HashTableRdResp = hash_table::HashTableReadResp<TEST_HT_VALUE_W>;
    type HashTableWrReq = hash_table::HashTableWriteReq<TEST_HT_KEY_W, TEST_HT_VALUE_W, TEST_HT_SIZE, TEST_HT_SIZE_W>;
    type HashTableWrResp = hash_table::HashTableWriteResp;

    type HashTableRamRdReq = ram::ReadReq<TEST_HT_HASH_W, TEST_HT_RAM_NUM_PARTITIONS>;
    type HashTableRamRdResp = ram::ReadResp<TEST_HT_RAM_DATA_W>;
    type HashTableRamWrReq = ram::WriteReq<TEST_HT_HASH_W, TEST_HT_RAM_DATA_W, TEST_HT_RAM_NUM_PARTITIONS>;
    type HashTableRamWrResp = ram::WriteResp;

    // History Buffer

    type HistoryBufferRdReq = history_buffer::HistoryBufferReadReq<TEST_HB_OFFSET_W>;
    type HistoryBufferRdResp = history_buffer::HistoryBufferReadResp<TEST_HB_DATA_W>;
    type HistoryBufferWrReq = history_buffer::HistoryBufferWriteReq<TEST_HB_DATA_W>;
    type HistoryBufferWrResp = history_buffer::HistoryBufferWriteResp;

    type HistoryBufferRamRdReq = ram::ReadReq<TEST_HB_RAM_ADDR_W, TEST_HB_RAM_NUM_PARTITIONS>;
    type HistoryBufferRamRdResp = ram::ReadResp<TEST_HB_RAM_DATA_W>;
    type HistoryBufferRamWrReq = ram::WriteReq<TEST_HB_RAM_ADDR_W, TEST_HB_RAM_DATA_W, TEST_HB_RAM_NUM_PARTITIONS>;
    type HistoryBufferRamWrResp = ram::WriteResp;

    // Match Finder

    type Req = MatchFinderReq<TEST_HT_SIZE_W, TEST_ADDR_W>;
    type Resp = MatchFinderResp;

    // Other

    type NumEntriesLog2 = uN[TEST_HT_SIZE_W];
    type RamAddr = uN[TEST_RAM_ADDR_W];
    type RamData = uN[TEST_RAM_DATA_W];
    type RamMask = uN[TEST_RAM_NUM_PARTITIONS];

    terminator: chan<bool> out;

    req_s: chan<Req> out;
    resp_r: chan<Resp> in;

    mem_wr_req_r: chan<MemWriterReq> in;
    mem_wr_packet_r: chan<MemWriterDataPacket> in;
    mem_wr_resp_s: chan<MemWriterResp> out;

    input_ram_rd_req_s: chan<InputBufferRamRdReq> out;
    input_ram_rd_resp_r: chan<InputBufferRamRdResp> in;
    input_ram_wr_req_s: chan<InputBufferRamWrReq> out;
    input_ram_wr_resp_r: chan<InputBufferRamWrResp> in;

    config(terminator: chan<bool> out) {

        // Hash Table RAM

        let (ht_ram_rd_req_s, ht_ram_rd_req_r) = chan<HashTableRamRdReq>("ht_ram_rd_req");
        let (ht_ram_rd_resp_s, ht_ram_rd_resp_r) = chan<HashTableRamRdResp>("ht_ram_rd_resp");
        let (ht_ram_wr_req_s, ht_ram_wr_req_r) = chan<HashTableRamWrReq>("ht_ram_wr_req");
        let (ht_ram_wr_resp_s, ht_ram_wr_resp_r) = chan<HashTableRamWrResp>("ht_ram_wr_resp");

        spawn ram::RamModel<
            TEST_HT_RAM_DATA_W, TEST_HT_SIZE, TEST_HT_RAM_WORD_PARTITION_SIZE,
            TEST_HT_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_HT_RAM_INITIALIZED
        >(
            ht_ram_rd_req_r, ht_ram_rd_resp_s,
            ht_ram_wr_req_r, ht_ram_wr_resp_s
        );

        // Hash Table

        let (ht_rd_req_s, ht_rd_req_r) = chan<HashTableRdReq>("ht_rd_req");
        let (ht_rd_resp_s, ht_rd_resp_r) = chan<HashTableRdResp>("ht_rd_resp");
        let (ht_wr_req_s, ht_wr_req_r) = chan<HashTableWrReq>("ht_wr_req");
        let (ht_wr_resp_s, ht_wr_resp_r) = chan<HashTableWrResp>("ht_wr_resp");

        spawn hash_table::HashTable<TEST_HT_KEY_W, TEST_HT_VALUE_W, TEST_HT_SIZE, TEST_HT_SIZE_W>(
            ht_rd_req_r, ht_rd_resp_s,
            ht_wr_req_r, ht_wr_resp_s,
            ht_ram_rd_req_s, ht_ram_rd_resp_r,
            ht_ram_wr_req_s, ht_ram_wr_resp_r,
        );

        // History Buffer RAM

        let (hb_ram_rd_req_s, hb_ram_rd_req_r) = chan<HistoryBufferRamRdReq>[8]("hb_ram_rd_req");
        let (hb_ram_rd_resp_s, hb_ram_rd_resp_r) = chan<HistoryBufferRamRdResp>[8]("hb_ram_rd_resp");
        let (hb_ram_wr_req_s, hb_ram_wr_req_r) = chan<HistoryBufferRamWrReq>[8]("hb_ram_wr_req");
        let (hb_ram_wr_resp_s, hb_ram_wr_resp_r) = chan<HistoryBufferRamWrResp>[8]("hb_ram_wr_resp");

        spawn ram::RamModel<
            TEST_HB_RAM_DATA_W, TEST_HB_RAM_SIZE, TEST_HB_RAM_PARTITION_SIZE,
            TEST_HB_RAM_SIMULTANEOUS_RW_BEHAVIOR, TEST_HB_RAM_INITIALIZED
        >(
            hb_ram_rd_req_r[0], hb_ram_rd_resp_s[0],
            hb_ram_wr_req_r[0], hb_ram_wr_resp_s[0],
        );

        spawn ram::RamModel<
            TEST_HB_RAM_DATA_W, TEST_HB_RAM_SIZE, TEST_HB_RAM_PARTITION_SIZE,
            TEST_HB_RAM_SIMULTANEOUS_RW_BEHAVIOR, TEST_HB_RAM_INITIALIZED
        >(
            hb_ram_rd_req_r[1], hb_ram_rd_resp_s[1],
            hb_ram_wr_req_r[1], hb_ram_wr_resp_s[1],
        );

        spawn ram::RamModel<
            TEST_HB_RAM_DATA_W, TEST_HB_RAM_SIZE, TEST_HB_RAM_PARTITION_SIZE,
            TEST_HB_RAM_SIMULTANEOUS_RW_BEHAVIOR, TEST_HB_RAM_INITIALIZED
        >(
            hb_ram_rd_req_r[2], hb_ram_rd_resp_s[2],
            hb_ram_wr_req_r[2], hb_ram_wr_resp_s[2],
        );

        spawn ram::RamModel<
            TEST_HB_RAM_DATA_W, TEST_HB_RAM_SIZE, TEST_HB_RAM_PARTITION_SIZE,
            TEST_HB_RAM_SIMULTANEOUS_RW_BEHAVIOR, TEST_HB_RAM_INITIALIZED
        >(
            hb_ram_rd_req_r[3], hb_ram_rd_resp_s[3],
            hb_ram_wr_req_r[3], hb_ram_wr_resp_s[3],
        );

        spawn ram::RamModel<
            TEST_HB_RAM_DATA_W, TEST_HB_RAM_SIZE, TEST_HB_RAM_PARTITION_SIZE,
            TEST_HB_RAM_SIMULTANEOUS_RW_BEHAVIOR, TEST_HB_RAM_INITIALIZED
        >(
            hb_ram_rd_req_r[4], hb_ram_rd_resp_s[4],
            hb_ram_wr_req_r[4], hb_ram_wr_resp_s[4],
        );

        spawn ram::RamModel<
            TEST_HB_RAM_DATA_W, TEST_HB_RAM_SIZE, TEST_HB_RAM_PARTITION_SIZE,
            TEST_HB_RAM_SIMULTANEOUS_RW_BEHAVIOR, TEST_HB_RAM_INITIALIZED
        >(
            hb_ram_rd_req_r[5], hb_ram_rd_resp_s[5],
            hb_ram_wr_req_r[5], hb_ram_wr_resp_s[5],
        );

        spawn ram::RamModel<
            TEST_HB_RAM_DATA_W, TEST_HB_RAM_SIZE, TEST_HB_RAM_PARTITION_SIZE,
            TEST_HB_RAM_SIMULTANEOUS_RW_BEHAVIOR, TEST_HB_RAM_INITIALIZED
        >(
            hb_ram_rd_req_r[6], hb_ram_rd_resp_s[6],
            hb_ram_wr_req_r[6], hb_ram_wr_resp_s[6],
        );

        spawn ram::RamModel<
            TEST_HB_RAM_DATA_W, TEST_HB_RAM_SIZE, TEST_HB_RAM_PARTITION_SIZE,
            TEST_HB_RAM_SIMULTANEOUS_RW_BEHAVIOR, TEST_HB_RAM_INITIALIZED
        >(
            hb_ram_rd_req_r[7], hb_ram_rd_resp_s[7],
            hb_ram_wr_req_r[7], hb_ram_wr_resp_s[7],
        );

        // History Buffer

        let (hb_rd_req_s, hb_rd_req_r) = chan<HistoryBufferRdReq>("hb_rd_req");
        let (hb_rd_resp_s, hb_rd_resp_r) = chan<HistoryBufferRdResp>("hb_rd_resp");
        let (hb_wr_req_s, hb_wr_req_r) = chan<HistoryBufferWrReq>("hb_wr_req");
        let (hb_wr_resp_s, hb_wr_resp_r) = chan<HistoryBufferWrResp>("hb_wr_resp");

        spawn history_buffer::HistoryBuffer<TEST_HB_SIZE, TEST_HB_DATA_W>(
            hb_rd_req_r, hb_rd_resp_s,
            hb_wr_req_r, hb_wr_resp_s,
            hb_ram_rd_req_s, hb_ram_rd_resp_r,
            hb_ram_wr_req_s, hb_ram_wr_resp_r,
        );

        // Input Memory

        let (input_ram_rd_req_s, input_ram_rd_req_r) = chan<InputBufferRamRdReq>("input_ram_rd_req");
        let (input_ram_rd_resp_s, input_ram_rd_resp_r) = chan<InputBufferRamRdResp>("input_ram_rd_resp");
        let (input_ram_wr_req_s, input_ram_wr_req_r) = chan<InputBufferRamWrReq>("input_ram_wr_req");
        let (input_ram_wr_resp_s, input_ram_wr_resp_r) = chan<InputBufferRamWrResp>("input_ram_wr_resp");

        spawn ram::RamModel<
            TEST_RAM_DATA_W, TEST_RAM_SIZE, TEST_RAM_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_RW_BEHAVIOR, TEST_RAM_INITIALIZED,
            TEST_RAM_ASSERT_VALID_READ, TEST_RAM_ADDR_W,
        >(
            input_ram_rd_req_r, input_ram_rd_resp_s,
            input_ram_wr_req_r, input_ram_wr_resp_s,
        );

        // Input Memory Axi Reader

        let (axi_ar_s, axi_ar_r) = chan<AxiAr>("axi_ar");
        let (axi_r_s, axi_r_r) = chan<AxiR>("axi_r");

        spawn axi_ram::AxiRamReader<
            TEST_ADDR_W, TEST_DATA_W,
            TEST_DEST_W, TEST_ID_W,
            TEST_RAM_SIZE,
        >(
            axi_ar_r, axi_r_s,
            input_ram_rd_req_s, input_ram_rd_resp_r,
        );

        // Input Memory Reader

        let (mem_rd_req_s, mem_rd_req_r) = chan<MemReaderReq>("mem_rd_req");
        let (mem_rd_resp_s, mem_rd_resp_r) = chan<MemReaderResp>("mem_rd_resp");

        spawn mem_reader::MemReader<
            TEST_DATA_W, TEST_ADDR_W, TEST_DEST_W, TEST_ID_W,
        >(
            mem_rd_req_r, mem_rd_resp_s,
            axi_ar_s, axi_r_r,
        );

        // Output Memory Writer

        let (mem_wr_req_s, mem_wr_req_r) = chan<MemWriterReq>("mem_wr_req");
        let (mem_wr_packet_s, mem_wr_packet_r) = chan<MemWriterDataPacket>("mem_wr_packet");
        let (mem_wr_resp_s, mem_wr_resp_r) = chan<MemWriterResp>("mem_wr_resp");

        // Match Finder

        let (req_s, req_r) = chan<Req>("req");
        let (resp_s, resp_r) = chan<Resp>("resp");

        spawn MatchFinder<
            TEST_ADDR_W, TEST_DATA_W, TEST_HT_SIZE, TEST_HB_SIZE, TEST_MIN_SEQ_LEN,
            TEST_DATA_W_LOG2,
            TEST_HT_KEY_W, TEST_HT_VALUE_W, TEST_HT_SIZE_W,
            TEST_HB_DATA_W, TEST_HB_OFFSET_W,
        >(
            req_r, resp_s,
            mem_rd_req_s, mem_rd_resp_r,
            mem_wr_req_s, mem_wr_packet_s, mem_wr_resp_r,
            ht_rd_req_s, ht_rd_resp_r, ht_wr_req_s, ht_wr_resp_r,
            hb_rd_req_s, hb_rd_resp_r, hb_wr_req_s, hb_wr_resp_r,
        );

      (
          terminator,
          req_s, resp_r,
          mem_wr_req_r, mem_wr_packet_r, mem_wr_resp_s,
          input_ram_rd_req_s, input_ram_rd_resp_r, input_ram_wr_req_s, input_ram_wr_resp_r,
      )

    }

    init { zero!<TestState>() }

    next(state: TestState) {
        let tok = join();

        let tok = if state.iteration == u32:0 {
        // Fill the input RAM
            let tok = for ((i, test_data), tok) in enumerate(TEST_DATA) {
                let ram_wr_req = InputBufferRamWrReq {
                    addr: i as RamAddr,
                    data: test_data,
                    mask: !RamMask:0,
                };
                let tok = send(tok, input_ram_wr_req_s, ram_wr_req);
                trace_fmt!("[TEST] Sent #{} data to input RAM {:#x}", i + u32:1, ram_wr_req);
                let (tok, _) = recv(tok, input_ram_wr_resp_r);
                tok
            }(tok);

            // Start the request

            let req = Req {
                input_addr: uN[TEST_ADDR_W]:0x0,
                input_size: (array_size(TEST_DATA) * TEST_DATA_W / SYMBOL_WIDTH) as u32,
                output_lit_addr: TEST_OUTPUT_LIT_ADDR,
                output_seq_addr: TEST_OUTPUT_SEQ_ADDR,
                zstd_params: ZstdParams<TEST_HT_SIZE_W> {
                    num_entries_log2: NumEntriesLog2:8,
                },
            };

            let tok = send(tok, req_s, req);
            trace_fmt!("[TEST] Sent request to the MatchFinder: {:#x}", req);

            tok
        } else {
            tok
        };

        let (tok, mem_wr_req, mem_wr_req_valid) = recv_if_non_blocking(
            tok, mem_wr_req_r, state.wr_len == uN[TEST_ADDR_W]:0, zero!<MemWriterReq>()
        );
        let (tok, mem_wr_packet, mem_wr_packet_valid) = recv_if_non_blocking(
            tok, mem_wr_packet_r, state.wr_len > uN[TEST_ADDR_W]:0, zero!<MemWriterDataPacket>()
        );
        let (tok, resp, resp_valid) = recv_if_non_blocking(
            tok, resp_r, state.wr_len == uN[TEST_ADDR_W]:0, zero!<MatchFinderResp>()
        );

        let tok = send_if(tok, mem_wr_resp_s, mem_wr_packet_valid, MemWriterResp{status: mem_writer::MemWriterRespStatus::OKAY});

        let state = if mem_wr_req_valid {
            TestState {
                wr_addr: mem_wr_req.addr,
                wr_offset: uN[TEST_ADDR_W]:0,
                wr_len: mem_wr_req.length,
                ..state
            }
        } else {
            state
        };

        if mem_wr_req_valid {
            trace_fmt!("[TEST] Received {:#x}", mem_wr_req);
        } else {};
        if mem_wr_packet_valid {
            trace_fmt!("[TEST] Received {:#x}", mem_wr_packet);
        } else {};

        let state = if mem_wr_packet_valid {
            if mem_wr_packet.length > state.wr_len {
                trace_fmt!("[TEST] Invalid packet length");
                fail!("invalid_packet_length", mem_wr_packet.length);
            } else {};
            let state = match (state.wr_addr & TEST_OUTPUT_ADDR_MASK) {
                TEST_OUTPUT_LIT_ADDR => {
                    trace_fmt!("[TEST] Received literals");
                    let lit_buffer = for (i, lit_buffer) in range(u32:0, TEST_DATA_W / SYMBOL_WIDTH) {
                        if i < mem_wr_packet.length {
                            let literal = (mem_wr_packet.data >> (SYMBOL_WIDTH * i)) as uN[SYMBOL_WIDTH];
                            let idx = (TEST_ADDR_W / SYMBOL_WIDTH) * (state.wr_addr - TEST_OUTPUT_LIT_ADDR + state.wr_offset) + i;
                            update(lit_buffer, idx, literal)
                        } else {
                            lit_buffer
                        }
                    }(state.lit_buffer);
                    TestState {
                        lit_buffer: lit_buffer,
                        wr_offset: state.wr_offset + mem_wr_packet.length,
                        wr_len: state.wr_len - mem_wr_packet.length,
                        ..state
                    }
                },
                TEST_OUTPUT_SEQ_ADDR => {
                    trace_fmt!("[TEST] Received sequence");
                    assert_eq(uN[TEST_ADDR_W]:6, mem_wr_packet.length);
                    let sequence = MatchFinderSequence {
                        literals_len: (mem_wr_packet.data >> u32:32) as u16,
                        match_offset: (mem_wr_packet.data >> u32:16) as u16,
                        match_len: mem_wr_packet.data as u16,
                    };
                    let idx = ((TEST_ADDR_W / SYMBOL_WIDTH) * (state.wr_addr - TEST_OUTPUT_SEQ_ADDR + state.wr_offset)) / u32:3;
                    let seq_buffer = update(state.seq_buffer, idx, sequence);
                    TestState {
                        seq_buffer: seq_buffer,
                        wr_offset: state.wr_offset + mem_wr_packet.length,
                        wr_len: state.wr_len - mem_wr_packet.length,
                        ..state
                    }
                },
                _ => {
                    trace_fmt!("[TEST] Invalid write addres");
                    fail!("invalid_wr_addr", state.wr_addr);
                    state
                },
            };
            state
        } else {
            state
        };

        if resp_valid {
            // check buffers content
            trace_fmt!("[TEST] Received Match Finder response {:#x}", resp);
            for ((i, test_lit), ()) in enumerate(TEST_LITERALS) {
                assert_eq(test_lit, state.lit_buffer[i]);
            }(());
            for ((i, test_seq), ()) in enumerate(TEST_SEQUENCES) {
                assert_eq(test_seq, state.seq_buffer[i]);
            }(());
        } else { };

        send_if(tok, terminator, resp_valid || state.iteration > u32:1000, true);

        TestState {
            iteration: state.iteration + u32:1,
            ..state
        }
    }
}
