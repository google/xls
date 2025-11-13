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
import xls.modules.zstd.common as common;
import xls.modules.zstd.memory.mem_writer as mem_writer;

pub struct JoinOutputReq {
    literals_to_receive: u32,
    copies_to_receive: u32,
}

enum JoinOutputStatus : u2 {
    IDLE = 0,
    RECEIVING_LITERALS = 1,
    RECEIVING_COPIES = 2,
}

struct JoinOutputState {
    status: JoinOutputStatus,
    literals_to_receive: u32,
    copies_to_receive: u32,
}

pub proc JoinOutput<AXI_DATA_W: u32, AXI_ADDR_W: u32> {
    type MemWriterDataPacket = mem_writer::MemWriterDataPacket<AXI_DATA_W, AXI_ADDR_W>;

    join_output_req_r: chan<JoinOutputReq> in;
    output_wr_data_0_r: chan<MemWriterDataPacket> in;
    output_wr_data_1_r: chan<MemWriterDataPacket> in;
    output_mem_wr_data_s: chan<MemWriterDataPacket> out;
    
    config(
        join_output_req_r: chan<JoinOutputReq> in,
        output_wr_data_0_r: chan<MemWriterDataPacket> in,
        output_wr_data_1_r: chan<MemWriterDataPacket> in,
        output_mem_wr_data_s: chan<MemWriterDataPacket> out 
    ) {
        (
            join_output_req_r,
            output_wr_data_0_r,
            output_wr_data_1_r,
            output_mem_wr_data_s
        )
    }

    init {
        zero!<JoinOutputState>()
    }

    next(state: JoinOutputState) {
        let tok = join();

        let recv_req = state.status == JoinOutputStatus::IDLE;
        let (tok, req) = recv_if(tok, join_output_req_r, recv_req, zero!<JoinOutputReq>());
        if recv_req {
            trace_fmt!("[JoinOutput] Received request: {:x}", req);
        } else {};

        let req_valid = req.literals_to_receive != u32:0 || req.copies_to_receive != u32:0;
        let state = if recv_req && req_valid {
            JoinOutputState {
                status: if req.literals_to_receive > u32:0 { JoinOutputStatus::RECEIVING_LITERALS } else { JoinOutputStatus::RECEIVING_COPIES },
                literals_to_receive: req.literals_to_receive,
                copies_to_receive: req.copies_to_receive,
            }
        } else {
            state
        };

        let receive_literals = state.status == JoinOutputStatus::RECEIVING_LITERALS;
        let (tok, data0) = recv_if(tok, output_wr_data_0_r, receive_literals, zero!<MemWriterDataPacket>());
        if receive_literals {
            trace_fmt!("[JoinOutput] Received literals: {:x}", data0);
        } else {};
        let literals_finished = receive_literals && state.literals_to_receive == data0.length as u32;

        let receive_copies = state.status == JoinOutputStatus::RECEIVING_COPIES;
        let (tok, data1) = recv_if(tok, output_wr_data_1_r, receive_copies, zero!<MemWriterDataPacket>());
        if receive_copies {
            trace_fmt!("[JoinOutput] Received copies: {:x}", data1);
        } else {};

        let data = if receive_literals {
            MemWriterDataPacket {
                data: data0.data,
                length: data0.length,
                last: data0.last,
            }
        } else {
            MemWriterDataPacket {
                data: data1.data,
                length: data1.length,
                last: data1.last,
            }
        };
        let tok = send_if(tok, output_mem_wr_data_s, receive_literals || receive_copies, data);

        let new_literals_to_receive = state.literals_to_receive - data0.length as u32;
        let new_copies_to_receive = state.copies_to_receive - data1.length as u32;
        let state = match state.status {
            JoinOutputStatus::IDLE => zero!<JoinOutputState>(),
            JoinOutputStatus::RECEIVING_LITERALS => {
                let status = if literals_finished && new_copies_to_receive > u32:0 {
                    JoinOutputStatus::RECEIVING_COPIES
                } else if literals_finished {
                    JoinOutputStatus::IDLE
                } else {
                    JoinOutputStatus::RECEIVING_LITERALS
                };
                JoinOutputState {
                    status,
                    literals_to_receive: new_literals_to_receive,
                    copies_to_receive: new_copies_to_receive,
                }
            },
            JoinOutputStatus::RECEIVING_COPIES => {
                JoinOutputState {
                    status: if new_copies_to_receive == u32:0 {
                        JoinOutputStatus::IDLE
                    } else {
                        JoinOutputStatus::RECEIVING_COPIES
                    },
                    literals_to_receive: new_literals_to_receive,
                    copies_to_receive: new_copies_to_receive,
                }
            },
            _ => zero!<JoinOutputState>(),
        };

        state
    }
}

const TEST_AXI_DATA_W = u32:64;
const TEST_AXI_ADDR_W = u32:16;

#[test_proc]
proc JoinOutputTest {
    type MemWriterDataPacket = mem_writer::MemWriterDataPacket<TEST_AXI_DATA_W, TEST_AXI_ADDR_W>;

    terminator: chan<bool> out;

    join_output_req_s: chan<JoinOutputReq> out;
    output_write_0_s: chan<MemWriterDataPacket> out;
    output_write_1_s: chan<MemWriterDataPacket> out;
    output_mem_wr_data_r: chan<MemWriterDataPacket> in;

    config(terminator: chan<bool> out) {
        let (join_output_req_s, join_output_req_r) = chan<JoinOutputReq>("join_output_req");
        let (output_write_0_s, output_write_0_r) = chan<MemWriterDataPacket>("output_write_0");
        let (output_write_1_s, output_write_1_r) = chan<MemWriterDataPacket>("output_write_1");
        let (output_mem_wr_data_s, output_mem_wr_data_r) = chan<MemWriterDataPacket>("output_mem_wr_data");

        spawn JoinOutput<TEST_AXI_DATA_W, TEST_AXI_ADDR_W> (
            join_output_req_r,
            output_write_0_r,
            output_write_1_r,
            output_mem_wr_data_s,
        );

        (
            terminator,
            join_output_req_s,
            output_write_0_s,
            output_write_1_s,
            output_mem_wr_data_r,
        )
    }

    init {  }

    next(state: ()) {
        type MemWriterDataPacket = mem_writer::MemWriterDataPacket<TEST_AXI_DATA_W, TEST_AXI_ADDR_W>;

        let tok = join();

        let join_output_req = JoinOutputReq {
            literals_to_receive: u32:12,
            copies_to_receive: u32:15,
        };
        let tok = send(tok, join_output_req_s, join_output_req);

        // Process packets with literals
        let literals_data = MemWriterDataPacket {
            data: uN[TEST_AXI_DATA_W]:0x0102030405060708,
            length: uN[TEST_AXI_ADDR_W]:8,
            last: false,
        };
        let tok = send(tok, output_write_0_s, literals_data);

        let (tok, output_data) = recv(tok, output_mem_wr_data_r);
        trace_fmt!("[JoinOutputTest] Received output data: {:x}", output_data);
        assert_eq(output_data, literals_data);

        let literals_data = MemWriterDataPacket {
            data: uN[TEST_AXI_DATA_W]:0x01020304,
            length: uN[TEST_AXI_ADDR_W]:4,
            last: false,
        };
        let tok = send(tok, output_write_0_s, literals_data);

        let (tok, output_data) = recv(tok, output_mem_wr_data_r);
        trace_fmt!("[JoinOutputTest] Received output data: {:x}", output_data);
        assert_eq(output_data, literals_data);

        // Process packets with history copies
        let history_data = MemWriterDataPacket {
            data: uN[TEST_AXI_DATA_W]:0x0102030405060708,
            length: uN[TEST_AXI_ADDR_W]:8,
            last: false,
        };
        let tok = send(tok, output_write_1_s, history_data);

        let (tok, output_data) = recv(tok, output_mem_wr_data_r);
        trace_fmt!("[JoinOutputTest] Received output data: {:x}", output_data);
        assert_eq(output_data, history_data);

        let history_data = MemWriterDataPacket {
            data: uN[TEST_AXI_DATA_W]:0x01020304050607,
            length: uN[TEST_AXI_ADDR_W]:7,
            last: false,
        };
        let tok = send(tok, output_write_1_s, history_data);

        let (tok, output_data) = recv(tok, output_mem_wr_data_r);
        trace_fmt!("[JoinOutputTest] Received output data: {:x}", output_data);
        assert_eq(output_data, history_data);

        send(tok, terminator, true);
    }
}
