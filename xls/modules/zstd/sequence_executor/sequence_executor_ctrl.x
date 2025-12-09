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
import xls.modules.zstd.sequence_executor.join_output as join_output;
import xls.modules.zstd.sequence_executor.literals_executor as literals_executor;
import xls.modules.zstd.sequence_executor.history_copy_executor as history_copy_executor;

type Offset = common::Offset;
type SequenceExecutorPacket = common::SequenceExecutorPacket<common::AXI_DATA_BYTES_W>;
type SequenceExecutorMessageType = common::SequenceExecutorMessageType;
type JoinOutputReq = join_output::JoinOutputReq;

struct SequenceExecutorPacketSimple {
    literals: uN[common::AXI_DATA_W],
    length: uN[64],
    last: bool, 
}

struct SequenceExecutorPacketComp {
    ll: uN[16],
    ml: uN[16],
    of: uN[16],
    last: bool,
}

fn seq_exec_packet_as_simple(packet: SequenceExecutorPacket) -> SequenceExecutorPacketSimple {
    if packet.msg_type == SequenceExecutorMessageType::LITERAL {
        SequenceExecutorPacketSimple {
            length: packet.length as u64,
            literals: packet.content as uN[common::AXI_DATA_W],
            last: packet.last,
        }
    } else {
        SequenceExecutorPacketSimple {
            length: u64:0,
            literals: uN[common::AXI_DATA_W]:0,
            last: packet.last,
        }
    }
}

fn seq_exec_packet_as_comp(packet: SequenceExecutorPacket) -> SequenceExecutorPacketComp {
    if packet.msg_type == SequenceExecutorMessageType::SEQUENCE {
        SequenceExecutorPacketComp {
            ll: packet.length as u16,
            ml: packet.content[0 +: u16],
            of: packet.content[16 +: u16],
            last: packet.last,
        }
    } else {
        SequenceExecutorPacketComp {
            ll: u16:0,
            ml: u16:0,
            of: u16:0,
            last: packet.last,
        }
    }
}

struct SequenceExecutorState<RAM_ADDR_WIDTH: u32> {
    repeat_offsets: Offset[3],
    hb_addr: uN[RAM_ADDR_WIDTH],
}

pub fn handle_repeated_offset_for_sequences<RAM_DATA_WIDTH: u32 = {u32:8}>
    (seq: SequenceExecutorPacketComp, repeat_offsets: Offset[3])
    -> (SequenceExecutorPacketComp, Offset[3]) {
    type Sequence = SequenceExecutorPacketComp;

    let (offset, repeat_offsets) = if (seq.of as Offset <= Offset:3) {
        let idx = (seq.of as Offset - Offset:1) as u32;
        let idx = if (seq.ll == u16:0) {
            idx + u32:1
        } else { idx };

        if (idx == u32:0) {
            (repeat_offsets[0], repeat_offsets)
        } else {
            let offset = if idx < u32:3 { repeat_offsets[idx] } else { repeat_offsets[0] - Offset:1 };

            let repeat_offsets = if idx > u32:1 {
                update(repeat_offsets, u32:2, repeat_offsets[1])
            } else {repeat_offsets};
            let repeat_offsets = update(repeat_offsets, u32:1, repeat_offsets[0]);
            let repeat_offsets = update(repeat_offsets, u32:0, offset);

            (offset, repeat_offsets)
        }
    } else {
        let offset = (seq.of as Offset - Offset:3) as Offset;

        let repeat_offsets = update(repeat_offsets, u32:2, repeat_offsets[1]);
        let repeat_offsets = update(repeat_offsets, u32:1, repeat_offsets[0]);
        let repeat_offsets = update(repeat_offsets, u32:0, offset);

        (offset, repeat_offsets)
    };

    (
        Sequence { of: offset as u16, ..seq },
        repeat_offsets,
    )
}

pub proc SequenceExecutorCtrl<
    HISTORY_BUFFER_SIZE_KB: u32,
    RAM_DATA_WIDTH: u32,
    RAM_ADDR_WIDTH: u32,
    RAM_SIZE_TOTAL: u64,
    RAM_NUM: u32,
    RAM_DATA_WIDTH_BYTES: u32 = {RAM_DATA_WIDTH / u32:8},
    INIT_HB_PTR_ADDR: u32 = {u32:0},
    INIT_HB_LENGTH: u64 = {u64:0}>
{
    type SequenceExecutorState = SequenceExecutorState<RAM_ADDR_WIDTH>;
    type LiteralsExecutorReq = literals_executor::LiteralsExecutorReq;
    type LiteralsExecutorResp = literals_executor::LiteralsExecutorResp;
    type HistoryCopyExecutorReq = history_copy_executor::HistoryCopyExecutorReq<RAM_ADDR_WIDTH>;
    type HistoryCopyExecutorResp = history_copy_executor::HistoryCopyExecutorResp;

    input_r: chan<SequenceExecutorPacket> in;
    join_output_req_s: chan<JoinOutputReq> out;
    lit_exec_req_s: chan<LiteralsExecutorReq> out;
    lit_exec_resp_r: chan<LiteralsExecutorResp> in;
    history_copy_exec_req_s: chan<HistoryCopyExecutorReq> out;
    history_copy_exec_resp_r: chan<HistoryCopyExecutorResp> in;

    config(
        input_r: chan<SequenceExecutorPacket> in,
        join_output_req_s: chan<JoinOutputReq> out,
        lit_exec_req_s: chan<LiteralsExecutorReq> out,
        lit_exec_resp_r: chan<LiteralsExecutorResp> in,
        history_copy_exec_req_s: chan<HistoryCopyExecutorReq> out,
        history_copy_exec_resp_r: chan<HistoryCopyExecutorResp> in,
    ) {
        (
            input_r,
            join_output_req_s,
            lit_exec_req_s, lit_exec_resp_r,
            history_copy_exec_req_s, history_copy_exec_resp_r,
        )
    }

    init {
        const_assert!(INIT_HB_LENGTH <= RAM_SIZE_TOTAL);

        SequenceExecutorState {
            repeat_offsets: Offset[3]:[Offset:1, Offset:4, Offset:8],
            hb_addr: INIT_HB_PTR_ADDR as uN[RAM_ADDR_WIDTH],
        }
    }

    next(state: SequenceExecutorState) {
        let tok = join();

        let (tok, packet) = recv(tok, input_r);
        trace_fmt!("[SequenceExecutorCtrl] Received packet: {:x}", packet);

        // Converts received sequence packet to both simple and compressed version.
        // In case of raw literals, the as_comp_packet is equal to zero!<SequenceExecutorPacketComp>()
        // and in case of a compressed sequence, as_simple_packet is zero!<SequenceExecutorPacketSimple>().
        let is_comp_packet = packet.msg_type == SequenceExecutorMessageType::SEQUENCE;
        let as_simple_packet = seq_exec_packet_as_simple(packet);
        let as_comp_packet = seq_exec_packet_as_comp(packet);

        // We can combine lengths from both packets here as the other one will be always 0
        let join_output_req = JoinOutputReq {
            literals_to_receive: as_comp_packet.ll as u32 + as_simple_packet.length as u32,
            copies_to_receive: as_comp_packet.ml as u32,
        };
        let tok = send(tok, join_output_req_s, join_output_req);

        // Ignore handling repeat offsets if the compressed sequence is not valid
        let handle_repeat_offsets = is_comp_packet && (as_comp_packet.of != u16:0 && as_comp_packet.ml != u16:0);
        let (as_comp_packet, new_repeat_offsets) = if handle_repeat_offsets {
            handle_repeated_offset_for_sequences(as_comp_packet, state.repeat_offsets)
        } else {
            (as_comp_packet, state.repeat_offsets)
        };

        // Send request to literals executor only if there are literals to handle
        let lit_req_valid = as_comp_packet.ll != u16:0 || as_simple_packet.length != u64:0;
        let literals_executor_req = LiteralsExecutorReq {
            literal_length: as_comp_packet.ll,
            start_addr: state.hb_addr as u32,
            raw_literals: as_simple_packet.literals as u64,
            raw_literals_length: as_simple_packet.length as u32,
        };
        let tok = send_if(tok, lit_exec_req_s, lit_req_valid, literals_executor_req);
        if lit_req_valid {
            trace_fmt!("[SequenceExecutorCtrl] Sending request to literals executor: {:x}", literals_executor_req);
        } else {};

        let (tok, literals_executor_resp) = recv_if(tok, lit_exec_resp_r, lit_req_valid, zero!<LiteralsExecutorResp>());
        if lit_req_valid {
            trace_fmt!("[SequenceExecutorCtrl] Received response from literals executor: {:x}", literals_executor_resp);
        } else {};

        let history_copy_dest_addr = std::mod_pow2(state.hb_addr as u32 + as_comp_packet.ll as u32 + as_simple_packet.length as u32, RAM_SIZE_TOTAL as u32);
        let history_copy_source_addr = if as_comp_packet.of as u32 > history_copy_dest_addr {
            (RAM_SIZE_TOTAL as u32) - (as_comp_packet.of as u32 - history_copy_dest_addr)
        } else {
            history_copy_dest_addr - as_comp_packet.of as u32
        };
        trace_fmt!("[SequenceExecutorCtrl] History buffer base address: {:x}", state.hb_addr);
        trace_fmt!("[SequenceExecutorCtrl] History copy source address: {:x}", history_copy_source_addr);
        trace_fmt!("[SequenceExecutorCtrl] History copy dest address: {:x}", history_copy_dest_addr);

        // Send request to history copy executor only if there are history copies to handle
        let history_copy_req_valid = as_comp_packet.ml != u16:0 && is_comp_packet;
        let history_copy_executor_req = HistoryCopyExecutorReq {
            max_possible_read: std::min(as_comp_packet.of as u32, RAM_DATA_WIDTH / u32:8),
            match_length: as_comp_packet.ml,
            source_addr: history_copy_source_addr as uN[RAM_ADDR_WIDTH],
            dest_addr: history_copy_dest_addr as uN[RAM_ADDR_WIDTH],
        };
        let tok = send_if(tok, history_copy_exec_req_s, history_copy_req_valid, history_copy_executor_req);
        if history_copy_req_valid {
            trace_fmt!("[SequenceExecutorCtrl] Sending request to history copy executor: {:x}", history_copy_executor_req);
        } else {};

        let (tok, history_copy_executor_resp) = recv_if(tok, history_copy_exec_resp_r, history_copy_req_valid, zero!<HistoryCopyExecutorResp>());
        if history_copy_req_valid {
            trace_fmt!("[SequenceExecutorCtrl] Received response from history copy executor: {:x}", history_copy_executor_resp);
        } else {};

        // Wrap current base address of the history buffer if it reaches the end of it
        let new_hb_addr = std::mod_pow2(history_copy_dest_addr + as_comp_packet.ml as u32, RAM_SIZE_TOTAL as u32) as uN[RAM_ADDR_WIDTH];

        SequenceExecutorState {
            repeat_offsets: new_repeat_offsets,
            hb_addr: new_hb_addr,
        }
    }
}
