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

import std;
import xls.modules.zstd.common;
import xls.modules.zstd.sequence_executor.history_buffer;
import xls.modules.zstd.memory.mem_writer;


pub struct HistoryCopyExecutorReq<HB_ADDR_W: u32> {
    max_possible_read: u32,
    match_length: u16,
    source_addr: uN[HB_ADDR_W],
    dest_addr: uN[HB_ADDR_W],
}

pub enum HistoryCopyExecutorStatus : u1 {
    OK = 0,
    ERROR = 1,
}

pub struct HistoryCopyExecutorResp {
    status: HistoryCopyExecutorStatus,
}

struct HistoryCopyExecutorState<HB_ADDR_W: u32, HB_DATA_W: u32, HB_NUM_PARTITIONS: u32> {
    max_possible_read: u32,
    literals_left_to_send: u16,
    source_addr: uN[HB_ADDR_W],
    dest_addr: uN[HB_ADDR_W],
    write_req: history_buffer::HistoryBufferReq<HB_ADDR_W, HB_DATA_W, HB_NUM_PARTITIONS>,
    sending: bool,
}

pub proc HistoryCopyExecutor<
        AXI_DATA_W: u32, AXI_ADDR_W: u32,
        HB_DATA_W: u32, HB_ADDR_W: u32,
        MAX_BYTES_PER_REQ: u32 = {AXI_DATA_W / u32:8},
        HB_NUM_PARTITIONS: u32 = {HB_DATA_W / u32:8}> {
    type HistoryBufferReq = history_buffer::HistoryBufferReq<HB_ADDR_W, HB_DATA_W, HB_NUM_PARTITIONS>;
    type HistoryBufferResp = history_buffer::HistoryBufferResp<HB_DATA_W>;
    type HistoryBufferComp = history_buffer::HistoryBufferWrComp;
    type MemWriterDataPacket = mem_writer::MemWriterDataPacket<AXI_DATA_W, AXI_ADDR_W>;
    type HistoryCopyExecutorState = HistoryCopyExecutorState<HB_ADDR_W, HB_DATA_W, HB_NUM_PARTITIONS>;
    type HistoryCopyExecutorReq = HistoryCopyExecutorReq<HB_ADDR_W>;

    input_r: chan<HistoryCopyExecutorReq> in;
    resp_s: chan<HistoryCopyExecutorResp> out;
    hb_req_s: chan<HistoryBufferReq> out;
    hb_resp_r: chan<HistoryBufferResp> in;
    hb_comp_r: chan<HistoryBufferComp> in;
    output_mem_wr_data_in_s: chan<MemWriterDataPacket> out;

    config(
        input_r: chan<HistoryCopyExecutorReq> in,
        resp_s: chan<HistoryCopyExecutorResp> out,
        hb_req_s: chan<HistoryBufferReq> out,
        hb_resp_r: chan<HistoryBufferResp> in,
        hb_comp_r: chan<HistoryBufferComp> in,
        output_mem_wr_data_in_s: chan<MemWriterDataPacket> out
    ) {
        (
            input_r, resp_s,
            hb_req_s, hb_resp_r, hb_comp_r,
            output_mem_wr_data_in_s,
        )
    }

    init {
        zero!<HistoryCopyExecutorState>()
    }

    next(state: HistoryCopyExecutorState) {
        type HistoryBufferReq = history_buffer::HistoryBufferReq<HB_ADDR_W, HB_DATA_W, HB_NUM_PARTITIONS>;
        type HistoryBufferResp = history_buffer::HistoryBufferResp<HB_DATA_W>;
        type MemWriterDataPacket = mem_writer::MemWriterDataPacket<AXI_DATA_W, AXI_ADDR_W>;

        let tok = join();

        let recv_req = state.literals_left_to_send == u16:0;
        let (tok, req) = recv_if(tok, input_r, recv_req, zero!<HistoryCopyExecutorReq>());
        if recv_req {
            trace_fmt!("[HistoryReadExecutor] Received request: {:#x}", req);
        } else {};

        let state = if recv_req {
            HistoryCopyExecutorState {
                max_possible_read: req.max_possible_read,
                literals_left_to_send: req.match_length,
                source_addr: req.source_addr,
                dest_addr: req.dest_addr,
                ..zero!<HistoryCopyExecutorState>()
            }
        } else {
            state
        };

        let bytes_to_read = std::min(state.max_possible_read, state.literals_left_to_send as u32);
        let data_mask = std::unsigned_max_value<MAX_BYTES_PER_REQ>() >> (MAX_BYTES_PER_REQ - bytes_to_read);
        let hb_req = if state.sending {
            state.write_req
        } else {
            HistoryBufferReq {
                addr: state.source_addr,
                read_mask: data_mask,
                ..zero!<HistoryBufferReq>()
            }
        };
        let tok = send(tok, hb_req_s, hb_req);
        trace_fmt!("[HistoryCopyExecutor] Sending history buffer request: {:#x}", hb_req);
        let (tok, _comp_resp) = recv_if(tok, hb_comp_r, state.sending, ());

        let (tok, hb_resp) = recv_if(tok, hb_resp_r, !state.sending, zero!<HistoryBufferResp>());
        if !state.sending {
            trace_fmt!("[HistoryCopyExecutor] Received history buffer response: {:#x}", hb_resp);
        } else {};

        let write_req = HistoryBufferReq {
            addr: state.dest_addr,
            data: hb_resp.data,
            write_mask: data_mask,
            ..zero!<HistoryBufferReq>()
        };

        let output_mem_write = MemWriterDataPacket {
            data: hb_req.data,
            length: bytes_to_read as uN[AXI_ADDR_W],
            last: false,
        };
        let tok = send_if(tok, output_mem_wr_data_in_s, state.sending, output_mem_write);
        if state.sending {
            trace_fmt!("[HistoryCopyExecutor] Sending output mem write: {:#x}", output_mem_write);
        } else {};

        let send_response = state.sending && (state.literals_left_to_send as u32 - bytes_to_read) == u32:0;
        let response = HistoryCopyExecutorResp {
            status: HistoryCopyExecutorStatus::OK,
        };
        let tok = send_if(tok, resp_s, send_response, response);
        if send_response {
            trace_fmt!("[HistoryCopyExecutor] Sending response: {:#x}", response);
        } else {};

        let new_state = if state.sending {
            HistoryCopyExecutorState {
                max_possible_read: state.max_possible_read,
                literals_left_to_send: state.literals_left_to_send - bytes_to_read as u16,
                source_addr: state.source_addr,
                dest_addr: state.dest_addr + bytes_to_read as uN[HB_ADDR_W],
                write_req,
                sending: false,
            }
        } else {
            HistoryCopyExecutorState {
                max_possible_read: state.max_possible_read,
                literals_left_to_send: state.literals_left_to_send,
                source_addr: state.source_addr + bytes_to_read as uN[HB_ADDR_W],
                dest_addr: state.dest_addr,
                write_req,
                sending: true,
            }
        };

        new_state
    }
}

const TEST_AXI_ADDR_W = u32:32;
const TEST_AXI_DATA_W = u32:64;
const TEST_HB_ADDR_W = u32:32;
const TEST_HB_DATA_W = u32:64;
const TEST_HB_NUM_PARTITIONS = TEST_HB_DATA_W / u32:8;

#[test_proc]
proc HistoryCopyExecutorTest {
    type HistoryBufferReq = history_buffer::HistoryBufferReq<TEST_HB_ADDR_W, TEST_HB_DATA_W, TEST_HB_NUM_PARTITIONS>;
    type HistoryBufferResp = history_buffer::HistoryBufferResp<TEST_HB_DATA_W>;
    type HistoryBufferComp = history_buffer::HistoryBufferWrComp;
    type MemWriterDataPacket  = mem_writer::MemWriterDataPacket<TEST_AXI_DATA_W, TEST_AXI_ADDR_W>;
    type HistoryCopyExecutorReq = HistoryCopyExecutorReq<TEST_HB_ADDR_W>;

    terminator: chan<bool> out;

    input_s: chan<HistoryCopyExecutorReq> out;
    resp_r: chan<HistoryCopyExecutorResp> in;
    hb_req_r: chan<HistoryBufferReq> in;
    hb_resp_s: chan<HistoryBufferResp> out;
    hb_comp_s: chan<HistoryBufferComp> out;
    output_mem_write_r: chan<MemWriterDataPacket> in;

    config(terminator: chan<bool> out) {
        let (input_s, input_r) = chan<HistoryCopyExecutorReq>("input");
        let (resp_s, resp_r) = chan<HistoryCopyExecutorResp>("response");
        let (hb_req_s, hb_req_r) = chan<HistoryBufferReq>("hb_request");
        let (hb_resp_s, hb_resp_r) = chan<HistoryBufferResp>("hb_response");
        let (hb_comp_s, hb_comp_r) = chan<HistoryBufferComp>("hb_comp");
        let (output_mem_write_s, output_mem_write_r) = chan<MemWriterDataPacket>("output_mem_write");

        spawn HistoryCopyExecutor<TEST_AXI_DATA_W, TEST_AXI_ADDR_W, TEST_HB_DATA_W, TEST_HB_ADDR_W>(
            input_r, resp_s,
            hb_req_s, hb_resp_r, hb_comp_r,
            output_mem_write_s,
        );

        (
            terminator,
            input_s,
            resp_r,
            hb_req_r, hb_resp_s, hb_comp_s,
            output_mem_write_r,
        )
    }

    init { }

    next(state: ()) {
        type HistoryBufferReq = history_buffer::HistoryBufferReq<TEST_HB_ADDR_W, TEST_HB_DATA_W, TEST_HB_NUM_PARTITIONS>;
        type HistoryBufferResp = history_buffer::HistoryBufferResp<TEST_HB_DATA_W>;
        type MemWriterDataPacket = mem_writer::MemWriterDataPacket<TEST_AXI_DATA_W, TEST_AXI_ADDR_W>;

        let tok = join();

        let tok = send(tok, input_s, HistoryCopyExecutorReq {
            match_length: u16:23,
            source_addr: u32:0x10,
            dest_addr: u32:0x18,
            max_possible_read: u32:8,
        });

        let (tok, hb_req) = recv(tok, hb_req_r);
        trace_fmt!("[HistoryCopyExecutorTest] Received history buffer req: {:x}", hb_req);
        assert_eq(hb_req, HistoryBufferReq {
            addr: u32:0x10,
            data: u64:0x0,
            write_mask: u8:0x0,
            read_mask: u8:0xff,
        });

        let hb_read_resp = HistoryBufferResp{data: u64:0x0123456789abcdef};
        trace_fmt!("[HistoryCopyExecutorTest] Sending history buffer read response: {:x}", hb_read_resp);
        let tok = send(tok, hb_resp_s, hb_read_resp);

        let (tok, hb_req) = recv(tok, hb_req_r);
        trace_fmt!("[HistoryCopyExecutorTest] Received history buffer req: {:x}", hb_req);
        assert_eq(hb_req, HistoryBufferReq {
            addr: u32:0x18,
            data: u64:0x0123456789abcdef,
            write_mask: u8:0xff,
            read_mask: u8:0x0,
        });
        let tok = send(tok, hb_comp_s, ());

        let (tok, mem_write_req) = recv(tok, output_mem_write_r);
        trace_fmt!("[HistoryCopyExecutorTest] Received mem write packet: {:x}", mem_write_req);
        assert_eq(mem_write_req, MemWriterDataPacket {
            data: u64:0x0123456789abcdef,
            length: u32:0x8,
            last: false,
        });

        let (tok, hb_req) = recv(tok, hb_req_r);
        trace_fmt!("[HistoryCopyExecutorTest] Received history buffer req: {:x}", hb_req);
        assert_eq(hb_req, HistoryBufferReq {
            addr: u32:0x18,
            data: u64:0x0,
            write_mask: u8:0x0,
            read_mask: u8:0xff,
        });

        let hb_read_resp = HistoryBufferResp{data: u64:0xfedcba9876543210};
        trace_fmt!("[HistoryCopyExecutorTest] Sending history buffer read response: {:x}", hb_read_resp);
        let tok = send(tok, hb_resp_s, hb_read_resp);

        let (tok, hb_req) = recv(tok, hb_req_r);
        trace_fmt!("[HistoryCopyExecutorTest] Received history buffer req: {:x}", hb_req);
        assert_eq(hb_req, HistoryBufferReq {
            addr: u32:0x20,
            data: u64:0xfedcba9876543210,
            write_mask: u8:0xff,
            read_mask: u8:0x0,
        });
        let tok = send(tok, hb_comp_s, ());

        let (tok, mem_write_req) = recv(tok, output_mem_write_r);
        trace_fmt!("[HistoryCopyExecutorTest] Received mem write packet: {:x}", mem_write_req);
        assert_eq(mem_write_req, MemWriterDataPacket {
            data: u64:0xfedcba9876543210,
            length: u32:0x8,
            last: false,
        });

        let (tok, hb_req) = recv(tok, hb_req_r);
        trace_fmt!("[HistoryCopyExecutorTest] Received history buffer req: {:x}", hb_req);
        assert_eq(hb_req, HistoryBufferReq {
            addr: u32:0x20,
            data: u64:0x0,
            write_mask: u8:0x0,
            read_mask: u8:0x7f,
        });

        let hb_read_resp = HistoryBufferResp{data: u64:0x12341234123412};
        trace_fmt!("[HistoryCopyExecutorTest] Sending history buffer read response: {:x}", hb_read_resp);
        let tok = send(tok, hb_resp_s, hb_read_resp);

        let (tok, hb_req) = recv(tok, hb_req_r);
        trace_fmt!("[HistoryCopyExecutorTest] Received history buffer req: {:x}", hb_req);
        assert_eq(hb_req, HistoryBufferReq {
            addr: u32:0x28,
            data: u64:0x12341234123412,
            write_mask: u8:0x7f,
            read_mask: u8:0x0,
        });
        let tok = send(tok, hb_comp_s, ());

        let (tok, mem_write_req) = recv(tok, output_mem_write_r);
        trace_fmt!("[HistoryCopyExecutorTest] Received mem write packet: {:x}", mem_write_req);
        assert_eq(mem_write_req, MemWriterDataPacket {
            data: u64:0x12341234123412,
            length: u32:0x7,
            last: false,
        });

        send(tok, terminator, true);
    }
}
