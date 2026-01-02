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
import xls.modules.zstd.sequence_executor.history_buffer as history_buffer;
import xls.modules.zstd.sequence_executor.literals_prefetch as literals_prefetch;

pub struct LiteralsExecutorReq {
    literal_length: u16,
    start_addr: u32,
    raw_literals: u64,
    raw_literals_length: u32,
}

pub enum LiteralsExecutorStatus : u1 {
    OK = 0,
    ERROR = 1,
}

pub struct LiteralsExecutorResp {
    status: LiteralsExecutorStatus,
}

type LiteralsBufCtrl = common::LiteralsBufferCtrl;

type SequenceExecutorMessageType = common::SequenceExecutorMessageType;
type CopyOrMatchLength = common::CopyOrMatchLength;

fn last_data_mask<AXI_DATA_W: u32, AXI_DATA_BYTES_W: u32 = {AXI_DATA_W / u32:8}>(literals_len: u32) -> uN[AXI_DATA_BYTES_W] {
    let remain = (literals_len % AXI_DATA_BYTES_W) as uN[AXI_DATA_BYTES_W];
    match remain {
        uN[AXI_DATA_BYTES_W]:0 => std::unsigned_max_value<AXI_DATA_BYTES_W>(),
        _  => std::unsigned_max_value<AXI_DATA_BYTES_W>() >> (AXI_DATA_BYTES_W - (literals_len as u32 % AXI_DATA_BYTES_W)),
    }
}

const TEST_AXI_DATA_W = u32:64;
const TEST_AXI_DATA_BYTES_W = TEST_AXI_DATA_W / u32:8;

#[test]
fn test_last_data_mask() {
    assert_eq(last_data_mask<TEST_AXI_DATA_W>(u32:1), uN[TEST_AXI_DATA_BYTES_W]:0b00000001);
    assert_eq(last_data_mask<TEST_AXI_DATA_W>(u32:2), uN[TEST_AXI_DATA_BYTES_W]:0b00000011);
    assert_eq(last_data_mask<TEST_AXI_DATA_W>(u32:3), uN[TEST_AXI_DATA_BYTES_W]:0b00000111);
    assert_eq(last_data_mask<TEST_AXI_DATA_W>(u32:7), uN[TEST_AXI_DATA_BYTES_W]:0b01111111);
    assert_eq(last_data_mask<TEST_AXI_DATA_W>(u32:8), uN[TEST_AXI_DATA_BYTES_W]:0b11111111);
    assert_eq(last_data_mask<TEST_AXI_DATA_W>(u32:63), uN[TEST_AXI_DATA_BYTES_W]:0b01111111);
    assert_eq(last_data_mask<TEST_AXI_DATA_W>(u32:64), uN[TEST_AXI_DATA_BYTES_W]:0b11111111);
    assert_eq(last_data_mask<TEST_AXI_DATA_W>(u32:65), uN[TEST_AXI_DATA_BYTES_W]:0b00000001);
}

struct LiteralsExecutorState<DATA_BYTES_W: u32, ADDR_W: u32> {
    iter: uN[ADDR_W],
    end_addr: uN[ADDR_W],
    last_mask: uN[DATA_BYTES_W]
}

// Module responsible for fetching data from LiteralsBuffer and writing it to both HistoryBuffer and Output
pub proc LiteralsExecutor<AXI_DATA_W: u32, AXI_ADDR_W: u32, HB_DATA_W: u32, HB_ADDR_W: u32,
    HB_NUM_PARTITIONS: u32 = {HB_DATA_W / u32:8}, AXI_DATA_BYTES_W: u32 = {AXI_DATA_W / u32:8}> {
    type HistoryBufferReq = history_buffer::HistoryBufferReq<HB_ADDR_W, HB_DATA_W, HB_NUM_PARTITIONS>;
    type HistoryBufferResp = history_buffer::HistoryBufferResp<HB_DATA_W>;
    type HistoryBufferComp = history_buffer::HistoryBufferWrComp;
    type MemWriterDataPacket = mem_writer::MemWriterDataPacket<AXI_DATA_W, AXI_ADDR_W>;
    type SequenceExecutorPacket = common::SequenceExecutorPacket<AXI_DATA_BYTES_W>;
    type Iter = uN[HB_ADDR_W];
    type State = LiteralsExecutorState<AXI_DATA_BYTES_W, HB_ADDR_W>;

    input_r: chan<LiteralsExecutorReq> in;
    resp_s: chan<LiteralsExecutorResp> out;

    hb_req_s: chan<HistoryBufferReq> out;
    hb_resp_r: chan<HistoryBufferResp> in;
    hb_comp_r: chan<HistoryBufferComp> in;

    lit_buf_ctrl_prefetch_s: chan<LiteralsBufCtrl> out;
    lit_buf_prefetch_out_r: chan<SequenceExecutorPacket> in;

    mem_wr_s: chan<MemWriterDataPacket> out;

    config(
        input_r: chan<LiteralsExecutorReq> in,
        resp_s: chan<LiteralsExecutorResp> out,
        hb_req_s: chan<HistoryBufferReq> out,
        hb_resp_r: chan<HistoryBufferResp> in,
        hb_comp_r: chan<HistoryBufferComp> in,
        lit_buf_ctrl_s: chan<LiteralsBufCtrl> out,
        lit_buf_out_r: chan<SequenceExecutorPacket> in,
        mem_wr_s: chan<MemWriterDataPacket> out,
    ) {
        let (lit_buf_ctrl_prefetch_s, lit_buf_ctrl_prefetch_r) = chan<LiteralsBufCtrl, u32:32>("lit_buf_ctrl_prefetch");
        let (lit_buf_prefetch_out_s, lit_buf_prefetch_out_r) = chan<SequenceExecutorPacket, u32:32>("lit_buf_prefetch_out");

        spawn literals_prefetch::LiteralsPrefetch<AXI_DATA_W>(
            lit_buf_ctrl_prefetch_r, // LiteralsExecutor sends ctrl to LiteralsPrefetch
            lit_buf_ctrl_s,          // LiteralsPrefetch sends ctrl to LiteralsBuffer
            lit_buf_out_r,           // LiteralsPrefetch gets data from LiteralsBuffer
            lit_buf_prefetch_out_s   // LiteralsPrefetch sends data to LiteralsExecutor
        );

        (
            input_r,
            resp_s,
            hb_req_s, hb_resp_r, hb_comp_r,
            lit_buf_ctrl_prefetch_s, lit_buf_prefetch_out_r,
            mem_wr_s
        )
    }

    init { zero!<State>() }

    next(state: State) {
        type Iter = uN[HB_ADDR_W];
        type State = LiteralsExecutorState<AXI_DATA_BYTES_W, HB_ADDR_W>;

        let tok = join();

        let req_valid = state.iter == state.end_addr;

        let (tok, req) = recv_if(tok, input_r, req_valid, zero!<LiteralsExecutorReq>());

        let state = if req_valid {
            trace_fmt!("[LiteralsExecutor] Received request: {}", req);
            State {
                iter: req.start_addr as Iter,
                end_addr: req.start_addr as Iter + req.literal_length as Iter + req.raw_literals_length as Iter,
                last_mask: last_data_mask<AXI_DATA_W>(req.literal_length as u32),
            }
        } else {
            state
        };

        // request literals from LiteralsBuffer
        // done once per request on input
        // length on input is u16, and the length requested from LiteralsBuffer is u32,
        // so it's always possible to request the whole data.
        let raw_literals = req.raw_literals_length != u32:0;
        let tok = send_if(tok, lit_buf_ctrl_prefetch_s, req_valid && !raw_literals, LiteralsBufCtrl {
            length: req.literal_length as u32,
            // TODO we have no information whether this is the last request or not.
            // We can extend `LiteralsExecutorReq` with such information in case it's needed.
            last: true
        });

        let (tok, literals) = recv_if(tok, lit_buf_prefetch_out_r, !raw_literals, zero!<SequenceExecutorPacket>());
        let literals = if raw_literals {
            SequenceExecutorPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                length: req.raw_literals_length as CopyOrMatchLength,
                content: req.raw_literals as uN[AXI_DATA_W],
                last: true,
            }
        } else {
            literals
        };

        let last = state.iter + literals.length as Iter == state.end_addr;

        let wr_mask = if last {
            state.last_mask
        } else {
            std::unsigned_max_value<AXI_DATA_BYTES_W>()
        };

        // write data to history buffer
        let tok_hb = send(tok, hb_req_s, HistoryBufferReq {
            addr: state.iter as uN[HB_ADDR_W],
            data: literals.content as uN[HB_DATA_W],
            write_mask: wr_mask,
            read_mask: uN[AXI_DATA_BYTES_W]:0,
        });
        let (tok_hb, _resp, _valid) = recv_if_non_blocking(tok_hb, hb_resp_r, false, zero!<HistoryBufferResp>());
        let (tok_hb, _resp) = recv(tok_hb, hb_comp_r);

        // write data to output
        let mem_packet = MemWriterDataPacket {
            data: literals.content as uN[AXI_DATA_W],
            length: literals.length as uN[AXI_ADDR_W],
            last: last,
        };
        let tok_out = send(tok, mem_wr_s, mem_packet);
        trace_fmt!("[LiteralsExecutor] Sending output mem write: {:#x}", mem_packet);

        let tok = join(tok_out, tok_hb);

        let response = LiteralsExecutorResp {
            status: LiteralsExecutorStatus::OK,
        };
        send_if(tok, resp_s, last, response);

        // iterate further
        State {
            iter: state.iter + literals.length as Iter,
            ..state
        }
    }
}

const INST_HB_DATA_W = common::AXI_DATA_W;
const INST_HB_DATA_BYTES_W = common::AXI_DATA_W / u32:8;
const INST_HB_ADDR_W = u32:32;
const INST_AXI_DATA_W = common::AXI_DATA_W;
const INST_AXI_DATA_BYTES_W = common::AXI_DATA_W / u32:8;
const INST_AXI_ADDR_W = u32:32;
const INST_NUM_PARTITIONS = INST_HB_DATA_W / u32:8;

proc LiteralsExecutorInst {
    type HistoryBufferReq = history_buffer::HistoryBufferReq<INST_HB_ADDR_W, INST_HB_DATA_W, INST_NUM_PARTITIONS>;
    type HistoryBufferResp = history_buffer::HistoryBufferResp<INST_HB_DATA_W>;
    type HistoryBufferComp = history_buffer::HistoryBufferWrComp;
    type MemWriterDataPacket = mem_writer::MemWriterDataPacket<INST_AXI_DATA_W, INST_AXI_ADDR_W>;
    type SequenceExecutorPacket = common::SequenceExecutorPacket<INST_AXI_DATA_BYTES_W>;

    input_r: chan<LiteralsExecutorReq> in;
    resp_s: chan<LiteralsExecutorResp> out;

    hb_req_s: chan<HistoryBufferReq> out;
    hb_resp_r: chan<HistoryBufferResp> in;
    hb_comp_r: chan<HistoryBufferComp> in;

    lit_buf_ctrl_s: chan<LiteralsBufCtrl> out;
    lit_buf_out_r: chan<SequenceExecutorPacket> in;

    mem_wr_s: chan<MemWriterDataPacket> out;

    config(
        input_r: chan<LiteralsExecutorReq> in,
        resp_s: chan<LiteralsExecutorResp> out,
        hb_req_s: chan<HistoryBufferReq> out,
        hb_resp_r: chan<HistoryBufferResp> in,
        hb_comp_r: chan<HistoryBufferComp> in,
        lit_buf_ctrl_s: chan<LiteralsBufCtrl> out,
        lit_buf_out_r: chan<SequenceExecutorPacket> in,
        mem_wr_s: chan<MemWriterDataPacket> out,
    ) {

        spawn LiteralsExecutor<INST_AXI_DATA_W, INST_AXI_ADDR_W, INST_HB_DATA_W, INST_HB_ADDR_W, INST_NUM_PARTITIONS>(
            input_r, resp_s, hb_req_s, hb_resp_r, hb_comp_r, lit_buf_ctrl_s, lit_buf_out_r, mem_wr_s);
        (
            input_r, resp_s,
            hb_req_s, hb_resp_r, hb_comp_r,
            lit_buf_ctrl_s, lit_buf_out_r,
            mem_wr_s
        )
    }

    init { () }

    next(state: ()) {}
}

#[test_proc]
proc LiteralsExecutorTest {
    type HistoryBufferReq = history_buffer::HistoryBufferReq<INST_HB_ADDR_W, INST_HB_DATA_W, INST_NUM_PARTITIONS>;
    type HistoryBufferResp = history_buffer::HistoryBufferResp<INST_HB_DATA_W>;
    type HistoryBufferComp = history_buffer::HistoryBufferWrComp;
    type MemWriterDataPacket = mem_writer::MemWriterDataPacket<INST_AXI_DATA_W, INST_AXI_ADDR_W>;
    type SequenceExecutorPacket = common::SequenceExecutorPacket<INST_AXI_DATA_BYTES_W>;

    input_s: chan<LiteralsExecutorReq> out;
    resp_r: chan<LiteralsExecutorResp> in;
    hb_req_r: chan<HistoryBufferReq> in;
    hb_comp_s: chan<HistoryBufferComp> out;
    lit_buf_ctrl_r: chan<LiteralsBufCtrl> in;
    lit_buf_out_s: chan<SequenceExecutorPacket> out;
    mem_wr_r: chan<MemWriterDataPacket> in;
    terminator: chan<bool> out;

    config (terminator: chan<bool> out) {
        let (input_s, input_r) = chan<LiteralsExecutorReq>("input");
        let (resp_s, resp_r) = chan<LiteralsExecutorResp>("resp");
        let (hb_req_s, hb_req_r) = chan<HistoryBufferReq>("hb_req");
        let (_hb_resp_s, hb_resp_r) = chan<HistoryBufferResp>("hb_resp");
        let (hb_comp_s, hb_comp_r) = chan<HistoryBufferComp>("hb_comp");
        let (lit_buf_ctrl_s, lit_buf_ctrl_r) = chan<LiteralsBufCtrl>("lit_buf_ctrl");
        let (lit_buf_out_s, lit_buf_out_r) = chan<SequenceExecutorPacket>("lit_buf_out");
        let (mem_wr_s, mem_wr_r) = chan<MemWriterDataPacket>("mem_wr");

        spawn LiteralsExecutor<INST_AXI_DATA_W, INST_AXI_ADDR_W, INST_AXI_DATA_W, INST_HB_ADDR_W, INST_NUM_PARTITIONS>(
            input_r,
            resp_s,
            hb_req_s,
            hb_resp_r,
            hb_comp_r,
            lit_buf_ctrl_s,
            lit_buf_out_r,
            mem_wr_s
        );

        (input_s, resp_r, hb_req_r, hb_comp_s, lit_buf_ctrl_r, lit_buf_out_s, mem_wr_r, terminator)
    }

    init { }

    next (state: ()) {
        let tok = join();
        let LITERAL_LENGTH = u16:5;
        let HB_START_ADDR = u32:4;
        let LITERALS_CONTENT = uN[common::AXI_DATA_W]:0xfedc009876;

        let tok = send(tok, input_s, LiteralsExecutorReq {
            literal_length: LITERAL_LENGTH,
            start_addr: HB_START_ADDR,
            ..zero!<LiteralsExecutorReq>()
        });

        // LiteralsPrefetch must request the literals from LiteralsBuffer.
        let (tok, lit_buf_req) = recv(tok, lit_buf_ctrl_r);
        assert_eq(lit_buf_req.length, u32:5);
        assert_eq(lit_buf_req.last, true);

        // LiteralsBuffer sends 5 literals
        let tok = send(tok, lit_buf_out_s, SequenceExecutorPacket {
            msg_type: common::SequenceExecutorMessageType::LITERAL,
            length: common::CopyOrMatchLength:5,
            content: LITERALS_CONTENT,
            last: true
        });

        // It must write the literals to HistoryBuffer
        let (tok_hb, hb_req) = recv(tok, hb_req_r);
        assert_eq(hb_req.write_mask, uN[INST_HB_DATA_BYTES_W]:0b11111);
        assert_eq(hb_req.data, LITERALS_CONTENT as uN[INST_AXI_DATA_W]);
        assert_eq(hb_req.read_mask, uN[INST_HB_DATA_BYTES_W]:0);
        assert_eq(hb_req.addr, HB_START_ADDR as uN[INST_HB_ADDR_W]);

        let tok_hb = send(tok_hb, hb_comp_s, ());

        // it must write the literals to output
        let (tok_mem, mem_out) = recv(tok, mem_wr_r);
        assert_eq(mem_out.data, LITERALS_CONTENT as uN[INST_AXI_DATA_W]);
        assert_eq(mem_out.length, LITERAL_LENGTH as u32);

        let tok = join(tok_mem, tok_hb, tok);

        // it must send a response
        let (tok, resp) = recv(tok, resp_r);
        assert_eq(resp.status, LiteralsExecutorStatus::OK);

        // second test, with larger data
        let LITERAL_LENGTH = u16:3300;
        let HB_START_ADDR = u32:0x100;
        let tok = send(tok, input_s, LiteralsExecutorReq {
            literal_length: LITERAL_LENGTH,
            start_addr: HB_START_ADDR,
            ..zero!<LiteralsExecutorReq>()
        });

        // It must request literals from LiteralsBuffer
        let (tok, lit_buf_req) = recv(tok, lit_buf_ctrl_r);
        assert_eq(lit_buf_req.length, u32:3300);
        assert_eq(lit_buf_req.last, true);

        // LiteralsBuffer sends the data
        let tok_literals_send = for (i, tok_literals_send) in u16:0..(LITERAL_LENGTH / u16:8) {
            send(tok_literals_send, lit_buf_out_s, SequenceExecutorPacket {
                msg_type: common::SequenceExecutorMessageType::LITERAL,
                length: common::CopyOrMatchLength:8,
                content: i as uN[INST_HB_DATA_W],
                last: false
            })
        }(tok);

        // LiteralsBuffer sends the remaining bytes
        let tok_literals_send = send(tok_literals_send, lit_buf_out_s, SequenceExecutorPacket {
            msg_type: common::SequenceExecutorMessageType::LITERAL,
            length: (LITERAL_LENGTH % u16:8) as common::CopyOrMatchLength,
            content: uN[INST_HB_DATA_W]:0xdead,
            last: false
        });

        // It must write the literals to HistoryBuffer
        let tok_hb = for (i, tok_hb) in u16:0..(LITERAL_LENGTH / u16:8) {
            let (tok_hb, hb_req) = recv(tok_hb, hb_req_r);
            assert_eq(hb_req.write_mask, std::unsigned_max_value<INST_HB_DATA_BYTES_W>());
            assert_eq(hb_req.read_mask, uN[INST_HB_DATA_BYTES_W]:0);
            // increment by 8 as 8 bytes are written each time.
            assert_eq(hb_req.addr, (HB_START_ADDR + (i as u32) * u32:8) as uN[INST_HB_ADDR_W]);
            let tok_hb = send(tok_hb, hb_comp_s, ());
            tok_hb
        }(tok);

        let (tok_hb, hb_req) = recv(tok_hb, hb_req_r);
        assert_eq(hb_req.write_mask, uN[INST_HB_DATA_BYTES_W]:0b00001111);
        assert_eq(hb_req.read_mask, uN[INST_HB_DATA_BYTES_W]:0);
        assert_eq(hb_req.data, uN[INST_HB_DATA_W]:0xdead);
        let tok_hb = send(tok_hb, hb_comp_s, ());

        // it must write the literals to output
        let tok_out = for (_i, tok_out) in u16:0..(LITERAL_LENGTH / u16:8) {
            let (tok_out, data_packet) = recv(tok, mem_wr_r);
            assert_eq(data_packet.length, uN[INST_HB_ADDR_W]:8);
            assert_eq(data_packet.last, false);
            tok_out
        }(tok);
        let (tok_out, data_packet) = recv(tok, mem_wr_r);
        assert_eq(data_packet.length, (LITERAL_LENGTH % u16:8) as uN[INST_HB_ADDR_W]);
        assert_eq(data_packet.last, true);

        let tok = join(tok_literals_send, tok_hb, tok_out);

        send(tok, terminator, true);
    }
}
