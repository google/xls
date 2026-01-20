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
import xls.modules.zstd.parallel_rams as parallel_rams;
import xls.modules.zstd.ram_printer as ram_printer;

import xls.modules.zstd.sequence_executor.join_output as join_output;
import xls.modules.zstd.sequence_executor.literals_executor as literals_executor;
import xls.modules.zstd.sequence_executor.history_copy_executor as history_copy_executor;
import xls.modules.zstd.sequence_executor.history_buffer as history_buffer;
import xls.modules.zstd.sequence_executor.sequence_executor_ctrl as sequence_executor_ctrl;
import xls.examples.ram;

type BlockData = common::BlockData;
type CopyOrMatchContent = common::CopyOrMatchContent;
type CopyOrMatchLength = common::CopyOrMatchLength;
type ZstdDecodedPacket = common::ZstdDecodedPacket;
type BlockPacketLength = common::BlockPacketLength;
type Offset = common::Offset;
type SequenceExecutorMessageType = common::SequenceExecutorMessageType;
type JoinOutputReq = join_output::JoinOutputReq;
type LiteralsBufCtrl = common::LiteralsBufferCtrl;

pub proc SequenceExecutor<
    HISTORY_BUFFER_SIZE_KB: u32,
    AXI_DATA_W: u32,
    AXI_ADDR_W: u32,
    HB_RAM_SIZE_TOTAL: u64,
    HB_RAM_ADDR_W: u32,
    HB_RAM_NUM: u32,
    SINGLE_RAM_DATA_W: u32 = {u32:8},
    SINGLE_HB_RAM_ADDR_W: u32 = {HB_RAM_ADDR_W - std::clog2(HB_RAM_NUM)},
    AXI_DATA_BYTES_W: u32 = {AXI_DATA_W / u32:8}>
{
    type LiteralsExecutorReq = literals_executor::LiteralsExecutorReq;
    type LiteralsExecutorResp = literals_executor::LiteralsExecutorResp;
    type HistoryCopyExecutorReq = history_copy_executor::HistoryCopyExecutorReq<HB_RAM_ADDR_W>;
    type HistoryCopyExecutorResp = history_copy_executor::HistoryCopyExecutorResp;
    type MemWriterDataPacket  = mem_writer::MemWriterDataPacket<AXI_DATA_W, AXI_ADDR_W>;
    type HistoryBufferReq = history_buffer::HistoryBufferReq<HB_RAM_ADDR_W, AXI_DATA_W, HB_RAM_NUM>;
    type HistoryBufferResp = history_buffer::HistoryBufferResp<AXI_DATA_W>;
    type HistoryBufferComp = history_buffer::HistoryBufferWrComp;
    type RWRamReq = ram::RWRamReq<SINGLE_HB_RAM_ADDR_W, SINGLE_RAM_DATA_W>;
    type RWRamResp = ram::RWRamResp<SINGLE_RAM_DATA_W>;
    type SequenceExecutorPacket = common::SequenceExecutorPacket<AXI_DATA_BYTES_W>;
    type RWRamWrComp = ();

    config(
        input_r: chan<SequenceExecutorPacket> in,
        lit_buf_ctrl_s: chan<LiteralsBufCtrl> out,
        lit_buf_out_r: chan<SequenceExecutorPacket> in,
        rw_req_s_0: chan<RWRamReq>[HB_RAM_NUM] out,
        rw_resp_r_0: chan<RWRamResp>[HB_RAM_NUM] in,
        rw_wr_comp_r_0: chan<RWRamWrComp>[HB_RAM_NUM] in,
        rw_req_s_1: chan<RWRamReq>[HB_RAM_NUM] out,
        rw_resp_r_1: chan<RWRamResp>[HB_RAM_NUM] in,
        rw_wr_comp_r_1: chan<RWRamWrComp>[HB_RAM_NUM] in,
        output_mem_wr_data_s: chan<MemWriterDataPacket> out,
    ) {
        const CHANNEL_DEPTH = u32:32;

        let (lit_exec_req_s, lit_exec_req_r) = chan<LiteralsExecutorReq, CHANNEL_DEPTH>("literals_executor_req");
        let (lit_exec_resp_s, lit_exec_resp_r) = chan<LiteralsExecutorResp, CHANNEL_DEPTH>("literals_executor_resp");

        let (lit_exec_hb_req_s, lit_exec_hb_req_r) = chan<HistoryBufferReq, CHANNEL_DEPTH>("literals_executor_history_buffer_req");
        let (lit_exec_hb_resp_s, lit_exec_hb_resp_r) = chan<HistoryBufferResp, CHANNEL_DEPTH>("literals_executor_history_buffer_resp");
        let (lit_exec_hb_comp_s, lit_exec_hb_comp_r) = chan<HistoryBufferComp, CHANNEL_DEPTH>("literals_executor_history_buffer_comp");

        let (history_copy_exec_req_s, history_copy_exec_req_r) = chan<HistoryCopyExecutorReq, CHANNEL_DEPTH>("history_copy_executor_req");
        let (history_copy_exec_resp_s, history_copy_exec_resp_r) = chan<HistoryCopyExecutorResp, CHANNEL_DEPTH>("history_copy_executor_resp");

        let (join_output_req_s, join_output_req_r) = chan<JoinOutputReq, CHANNEL_DEPTH>("join_output_req");
        let (output_write_s, output_write_r) = chan<MemWriterDataPacket, CHANNEL_DEPTH>[2]("output_write");

        let (history_copy_hb_req_s, history_copy_hb_req_r) = chan<HistoryBufferReq, CHANNEL_DEPTH>("history_copy_history_buffer_req");
        let (history_copy_hb_resp_s, history_copy_hb_resp_r) = chan<HistoryBufferResp, CHANNEL_DEPTH>("history_copy_history_buffer_resp");
        let (history_copy_hb_comp_s, history_copy_hb_comp_r) = chan<HistoryBufferComp, CHANNEL_DEPTH>("history_copy_history_buffer_comp");

        spawn sequence_executor_ctrl::SequenceExecutorCtrl<
            HISTORY_BUFFER_SIZE_KB,
            AXI_DATA_W,
            HB_RAM_ADDR_W,
            HB_RAM_SIZE_TOTAL,
            HB_RAM_NUM,
        >(
            input_r,
            join_output_req_s,
            lit_exec_req_s, lit_exec_resp_r,
            history_copy_exec_req_s, history_copy_exec_resp_r
        );

        spawn literals_executor::LiteralsExecutor<
            AXI_DATA_W, AXI_ADDR_W, AXI_DATA_W, HB_RAM_ADDR_W
        >(
            lit_exec_req_r,
            lit_exec_resp_s,
            lit_exec_hb_req_s,
            lit_exec_hb_resp_r,
            lit_exec_hb_comp_r,
            lit_buf_ctrl_s,
            lit_buf_out_r,
            output_write_s[0]
        );

        spawn history_copy_executor::HistoryCopyExecutor<
            AXI_DATA_W, AXI_ADDR_W, AXI_DATA_W, HB_RAM_ADDR_W
        >(
            history_copy_exec_req_r,
            history_copy_exec_resp_s,
            history_copy_hb_req_s,
            history_copy_hb_resp_r,
            history_copy_hb_comp_r,
            output_write_s[1]
        );

        spawn history_buffer::HistoryBuffer<AXI_DATA_W, HB_RAM_ADDR_W, SINGLE_RAM_DATA_W>(
            lit_exec_hb_req_r, lit_exec_hb_resp_s, lit_exec_hb_comp_s,
            history_copy_hb_req_r, history_copy_hb_resp_s, history_copy_hb_comp_s,
            rw_req_s_0, rw_resp_r_0, rw_wr_comp_r_0,
            rw_req_s_1, rw_resp_r_1, rw_wr_comp_r_1,
        );

        spawn join_output::JoinOutput<AXI_DATA_W, AXI_ADDR_W>(
            join_output_req_r,
            output_write_r[0],
            output_write_r[1],
            output_mem_wr_data_s
        );

        ()
    }

    init { }

    next(state: ()) { }
}

pub const ZSTD_SINGLE_RAM_DATA_W = u32:8;
pub const ZSTD_HISTORY_BUFFER_SIZE_KB = common::HISTORY_BUFFER_SIZE_KB;
pub const ZSTD_AXI_DATA_W = common::AXI_DATA_W;
pub const ZSTD_AXI_DATA_BYTES_W = ZSTD_AXI_DATA_W / u32:8;
pub const ZSTD_AXI_ADDR_W = common::AXI_ADDR_W;
pub const ZSTD_HB_RAM_NUM = ZSTD_AXI_DATA_W / ZSTD_SINGLE_RAM_DATA_W;
pub const ZSTD_HB_RAM_SIZE_TOTAL = ZSTD_HISTORY_BUFFER_SIZE_KB as u64 * u64:1024;
pub const ZSTD_HB_RAM_ADDR_W = ZSTD_AXI_ADDR_W;
pub const ZSTD_SINGLE_HB_RAM_ADDR_W = ZSTD_HB_RAM_ADDR_W - std::clog2(ZSTD_HB_RAM_NUM);

pub proc SequenceExecutorZstd {
    type SequenceExecutorPacket = common::SequenceExecutorPacket<ZSTD_AXI_DATA_BYTES_W>;
    type RWRamReq = ram::RWRamReq<ZSTD_SINGLE_HB_RAM_ADDR_W, ZSTD_SINGLE_RAM_DATA_W>;
    type RWRamResp = ram::RWRamResp<ZSTD_SINGLE_RAM_DATA_W>;
    type RWRamWrComp = ();
    type MemWriterDataPacket = mem_writer::MemWriterDataPacket<ZSTD_AXI_DATA_W, ZSTD_AXI_ADDR_W>;
    init {  }

    config(
        input_r: chan<SequenceExecutorPacket> in,
        lit_buf_ctrl_s: chan<LiteralsBufCtrl> out,
        lit_buf_out_r: chan<SequenceExecutorPacket> in,
        rw_req_s_0: chan<RWRamReq>[ZSTD_HB_RAM_NUM] out,
        rw_resp_r_0: chan<RWRamResp>[ZSTD_HB_RAM_NUM] in,
        rw_wr_comp_r_0: chan<RWRamWrComp>[ZSTD_HB_RAM_NUM] in,
        rw_req_s_1: chan<RWRamReq>[ZSTD_HB_RAM_NUM] out,
        rw_resp_r_1: chan<RWRamResp>[ZSTD_HB_RAM_NUM] in,
        rw_wr_comp_r_1: chan<RWRamWrComp>[ZSTD_HB_RAM_NUM] in,
        output_mem_wr_data_s: chan<MemWriterDataPacket> out,
    ) {
        spawn SequenceExecutor<
            ZSTD_HISTORY_BUFFER_SIZE_KB,
            ZSTD_AXI_DATA_W,
            ZSTD_AXI_ADDR_W,
            ZSTD_HB_RAM_SIZE_TOTAL,
            ZSTD_HB_RAM_ADDR_W,
            ZSTD_HB_RAM_NUM,
            ZSTD_SINGLE_RAM_DATA_W,
        >(
            input_r,
            lit_buf_ctrl_s, lit_buf_out_r,
            rw_req_s_0,
            rw_resp_r_0,
            rw_wr_comp_r_0,
            rw_req_s_1,
            rw_resp_r_1,
            rw_wr_comp_r_1,
            output_mem_wr_data_s
        );
    }

    next (state: ()) { }
}

#[test_proc]
proc SequenceExecutorTest {
    type SequenceExecutorPacket = common::SequenceExecutorPacket<ZSTD_AXI_DATA_BYTES_W>;
    type RWRamReq = ram::RWRamReq<ZSTD_SINGLE_HB_RAM_ADDR_W, ZSTD_SINGLE_RAM_DATA_W>;
    type RWRamResp = ram::RWRamResp<ZSTD_SINGLE_RAM_DATA_W>;
    type RWRamWrComp = ();
    type MemWriterDataPacket = mem_writer::MemWriterDataPacket<ZSTD_AXI_DATA_W, ZSTD_AXI_ADDR_W>;

    terminator: chan<bool> out;

    input_s: chan<SequenceExecutorPacket> out;
    lit_buf_ctrl_r: chan<LiteralsBufCtrl> in;
    lit_buf_out_s: chan<SequenceExecutorPacket> out;
    rw_req_r_0: chan<RWRamReq>[ZSTD_HB_RAM_NUM] in;
    rw_resp_s_0: chan<RWRamResp>[ZSTD_HB_RAM_NUM] out;
    rw_wr_comp_s_0: chan<RWRamWrComp>[ZSTD_HB_RAM_NUM] out;
    rw_req_r_1: chan<RWRamReq>[ZSTD_HB_RAM_NUM] in;
    rw_resp_s_1: chan<RWRamResp>[ZSTD_HB_RAM_NUM] out;
    rw_wr_comp_s_1: chan<RWRamWrComp>[ZSTD_HB_RAM_NUM] out;
    output_mem_wr_data_r: chan<MemWriterDataPacket> in;

    config(terminator: chan<bool> out) {
        let (input_s, input_r) = chan<SequenceExecutorPacket>("input");

        let (output_mem_wr_data_s, output_mem_wr_data_r) = chan<MemWriterDataPacket>("output_mem_wr_data");

        let (rw_req_s_0, rw_req_r_0) = chan<RWRamReq>[ZSTD_HB_RAM_NUM]("rw_req_0");
        let (rw_resp_s_0, rw_resp_r_0) = chan<RWRamResp>[ZSTD_HB_RAM_NUM]("rw_resp_0");
        let (rw_wr_comp_s_0, rw_wr_comp_r_0) = chan<RWRamWrComp>[ZSTD_HB_RAM_NUM]("rw_wr_comp_0");

        let (rw_req_s_1, rw_req_r_1) = chan<RWRamReq>[ZSTD_HB_RAM_NUM]("rw_req_1");
        let (rw_resp_s_1, rw_resp_r_1) = chan<RWRamResp>[ZSTD_HB_RAM_NUM]("rw_resp_1");
        let (rw_wr_comp_s_1, rw_wr_comp_r_1) = chan<RWRamWrComp>[ZSTD_HB_RAM_NUM]("rw_wr_comp_1");

        let (lit_buf_ctrl_s, lit_buf_ctrl_r) = chan<LiteralsBufCtrl>("lit_buf_ctrl");
        let (lit_buf_out_s, lit_buf_out_r) = chan<SequenceExecutorPacket>("lit_buf_out");

        spawn SequenceExecutor<
            ZSTD_HISTORY_BUFFER_SIZE_KB,
            ZSTD_AXI_DATA_W,
            ZSTD_AXI_ADDR_W,
            ZSTD_HB_RAM_SIZE_TOTAL,
            ZSTD_HB_RAM_ADDR_W,
            ZSTD_HB_RAM_NUM,
            ZSTD_SINGLE_RAM_DATA_W,
        > (
            input_r,
            lit_buf_ctrl_s, lit_buf_out_r,
            rw_req_s_0,
            rw_resp_r_0,
            rw_wr_comp_r_0,
            rw_req_s_1,
            rw_resp_r_1,
            rw_wr_comp_r_1,
            output_mem_wr_data_s,
        );

        (
            terminator,
            input_s,
            lit_buf_ctrl_r, lit_buf_out_s,
            rw_req_r_0, rw_resp_s_0, rw_wr_comp_s_0,
            rw_req_r_1, rw_resp_s_1, rw_wr_comp_s_1,
            output_mem_wr_data_r,
        )
    }

    init {  }

    next(state: ()) {
        let tok = join();

        let literal_length = u64:0x8;
        let literals_content = uN[ZSTD_AXI_DATA_W]:0x7060504030201000;
        let packet = SequenceExecutorPacket{
            msg_type: SequenceExecutorMessageType::SEQUENCE,
            length: literal_length,
            content: uN[ZSTD_AXI_DATA_W]:0xb0010,
            last: u1:0};
        let tok = send(tok, input_s, packet);

        let (tok, lit_buf_req) = recv(tok, lit_buf_ctrl_r);
        trace_fmt!("[SequenceExecutorTest] Received literals buffer request: {:x}", lit_buf_req);
        let literals_packet = SequenceExecutorPacket {
            msg_type: common::SequenceExecutorMessageType::LITERAL,
            length: literal_length as common::CopyOrMatchLength,
            content: literals_content,
            last: false,
        };
        trace_fmt!("[SequenceExecutorTest] Sending literals packet: {:x}", literals_packet);
        let tok = send(tok, lit_buf_out_s, literals_packet);

        // Expect 8 bytes write to history buffer from literals
        unroll_for!(i, _) : (u32, ()) in u32:0..u32:8 {
            let (tok, _data) = recv(tok, rw_req_r_0[i]);
            send(tok, rw_wr_comp_s_0[i], ());
        }(());

        // Expect output mem write from literals
        let (tok, literal_packet) = recv(tok, output_mem_wr_data_r);
        trace_fmt!("[SequenceExecutorTest] Received mem write packet: {:x}", literal_packet);
        // assert_eq(literal_packet.data, literals_content);

        // Expect 8 bytes read from history buffer for history copy
        unroll_for!(i, _) : (u32, ()) in u32:0..u32:8 {
            let (tok, _req) = recv(tok, rw_req_r_1[i]);
            send(tok, rw_resp_s_1[i], RWRamResp {
                data: i as uN[ZSTD_SINGLE_RAM_DATA_W]
            });
        }(());

        // Expect 8 bytes write to history buffer from history copy
        unroll_for!(i, _) : (u32, ()) in u32:0..u32:8 {
            let (tok, req) = recv(tok, rw_req_r_1[i]);
            assert_eq(req.data, i as uN[ZSTD_SINGLE_RAM_DATA_W]);
            send(tok, rw_wr_comp_s_1[i], ());
        }(());

        // Expect 8 bytes read to history buffer from history copy
        unroll_for!(i, _) : (u32, ()) in u32:0..u32:8 {
            let (tok, _data) = recv(tok, rw_req_r_1[i]);
            send(tok, rw_resp_s_1[i], RWRamResp {
                data: i as uN[ZSTD_SINGLE_RAM_DATA_W] + uN[ZSTD_SINGLE_RAM_DATA_W]:0x10
            });
        }(());

        // Expect 8 bytes write from history buffer for history copy
        unroll_for!(i, _) : (u32, ()) in u32:0..u32:8 {
            let (tok, req) = recv(tok, rw_req_r_1[i]);
            trace_fmt!("Request: {:x}", req);
            assert_eq(req.data, i as uN[ZSTD_SINGLE_RAM_DATA_W] + uN[ZSTD_SINGLE_RAM_DATA_W]:0x10);
            send(tok, rw_wr_comp_s_1[i], ());
        }(());

        // Expect output mem write from history copy
        let (tok, history_copy_packet) = recv(tok, output_mem_wr_data_r);
        trace_fmt!("[SequenceExecutorTest] Received mem write packet: {:x}", history_copy_packet);
        assert_eq(history_copy_packet.data, uN[ZSTD_AXI_DATA_W]:0x0706050403020100);

        // Expect output mem write from history copy
        let (tok, history_copy_packet) = recv(tok, output_mem_wr_data_r);
        trace_fmt!("[SequenceExecutorTest] Received mem write packet: {:x}", history_copy_packet);
        assert_eq(history_copy_packet.data, uN[ZSTD_AXI_DATA_W]:0x1716151413121110);

        send(tok, terminator, true);
    }
}
