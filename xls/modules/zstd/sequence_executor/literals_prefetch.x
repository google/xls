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

type LiteralsBufCtrl = common::LiteralsBufferCtrl;
type SequenceExecutorPacket = common::SequenceExecutorPacket<common::SYMBOL_WIDTH>;

struct PrefetchState {
    sent: u32,
    to_send: u32,
}

pub proc LiteralsPrefetch<AXI_DATA_W: u32> {
    lit_buf_ctrl_r: chan<LiteralsBufCtrl> in;
    lit_buf_ctrl_s: chan<LiteralsBufCtrl> out;
    lit_buf_out_r: chan<SequenceExecutorPacket> in;
    lit_buf_out_s: chan<SequenceExecutorPacket> out;

    config(
        lit_buf_ctrl_r: chan<LiteralsBufCtrl> in,
        lit_buf_ctrl_s: chan<LiteralsBufCtrl> out,
        lit_buf_out_r: chan<SequenceExecutorPacket> in,
        lit_buf_out_s: chan<SequenceExecutorPacket> out,
    ) {
        (
            lit_buf_ctrl_r,
            lit_buf_ctrl_s,
            lit_buf_out_r,
            lit_buf_out_s
        )
    }

    init { zero!<PrefetchState>() }

    next(state: PrefetchState) {
        let tok = join();

        let ready_for_req = state.sent == state.to_send;

        let (tok, req) = recv_if(tok, lit_buf_ctrl_r, ready_for_req, zero!<LiteralsBufCtrl>());
        if ready_for_req {
            trace_fmt!("[LiteralsPrefetch] Received request: {:x}", req);
        } else {};

        let state = if ready_for_req {
            PrefetchState {
                sent: u32:0,
                to_send: req.length,
            }
        } else {
            state
        };

        let lit_buf_req = LiteralsBufCtrl {
            length: req.length,
            ..req
        };
        if ready_for_req {
            trace_fmt!("[LiteralsPrefetch] Sending literals buffer request: {:x}", lit_buf_req);
        } else {};
        let tok = send_if(tok, lit_buf_ctrl_s, ready_for_req, lit_buf_req);

        let (tok, literals) = recv(tok, lit_buf_out_r);
        trace_fmt!("[LiteralsPrefetch] Sending literals: {:x}", literals);
        let tok = send(tok, lit_buf_out_s, literals);

        let literals_sent = state.sent + literals.length as u32;

        PrefetchState {
            sent: literals_sent,
            ..state
        }
    }
}

const TEST_DATA_W = u32:64;

#[test_proc]
proc LiteralsPrefetchTest {
    lit_buf_ctrl0_s: chan<LiteralsBufCtrl> out;
    lit_buf_ctrl1_r: chan<LiteralsBufCtrl> in;
    lit_buf_out0_s: chan<SequenceExecutorPacket> out;
    lit_buf_out1_r: chan<SequenceExecutorPacket> in;
    terminator: chan<bool> out;

    config (terminator: chan<bool> out) {
        let (lit_buf_ctrl0_s, lit_buf_ctrl0_r) = chan<LiteralsBufCtrl>("ctrl_to_prefetch");
        let (lit_buf_ctrl1_s, lit_buf_ctrl1_r) = chan<LiteralsBufCtrl>("ctrl_to_lit_buf");
        let (lit_buf_out0_s, lit_buf_out0_r) = chan<SequenceExecutorPacket>("lit_buf_out0");
        let (lit_buf_out1_s, lit_buf_out1_r) = chan<SequenceExecutorPacket>("lit_buf_out1");

        spawn LiteralsPrefetch<TEST_DATA_W>(
            lit_buf_ctrl0_r,
            lit_buf_ctrl1_s,
            lit_buf_out0_r,
            lit_buf_out1_s
        );

        (lit_buf_ctrl0_s, lit_buf_ctrl1_r, lit_buf_out0_s, lit_buf_out1_r, terminator)
    }

    init { }

    next (state: ()) {
        let tok = join();

        // request a single literal
        let tok = send(tok, lit_buf_ctrl0_s, LiteralsBufCtrl {
            length: u32:1,
            last: false
        });

        let (tok, req) = recv(tok, lit_buf_ctrl1_r);
        assert_eq(req.length, u32:1);
        assert_eq(req.last, false);

        let tok = send(tok, lit_buf_out0_s, SequenceExecutorPacket {
            msg_type: common::SequenceExecutorMessageType::LITERAL,
            length: common::CopyOrMatchLength:1,
            content: u64:0x88,
            last: true
        });

        // It must send 1 literal back as requested
        let (tok, data) = recv(tok, lit_buf_out1_r);
        assert_eq(data.msg_type, common::SequenceExecutorMessageType::LITERAL);
        assert_eq(data.length, common::CopyOrMatchLength:1);
        assert_eq(data.content, u64:0x88);

        // Let's request more (17 literals).
        let tok = send(tok, lit_buf_ctrl0_s, LiteralsBufCtrl {
            length: u32:17,
            last: true
        });

        // It must request 17 literals:
        let (tok, req) = recv(tok, lit_buf_ctrl1_r);
        assert_eq(req.length, u32:17);
        assert_eq(req.last, true);

        // feed the prefetch with 17 literals as requested
        let tok_feed = send(tok, lit_buf_out0_s, SequenceExecutorPacket {
            msg_type: common::SequenceExecutorMessageType::LITERAL,
            length: common::CopyOrMatchLength:8,
            content: u64:0xffeeddccbbaafefd,
            last: false
        });
        let tok_feed = send(tok_feed, lit_buf_out0_s, SequenceExecutorPacket {
            msg_type: common::SequenceExecutorMessageType::LITERAL,
            length: common::CopyOrMatchLength:8,
            content: u64:0x0102030405060708,
            last: false
        });
        let tok_feed = send(tok_feed, lit_buf_out0_s, SequenceExecutorPacket {
            msg_type: common::SequenceExecutorMessageType::LITERAL,
            length: common::CopyOrMatchLength:1,
            content: u64:0x08,
            last: true
        });

        // It must send 8, 8, 1 (17 in total) literals back as requested
        let (tok_recv, data) = recv(tok, lit_buf_out1_r);
        assert_eq(data.msg_type, common::SequenceExecutorMessageType::LITERAL);
        assert_eq(data.length, common::CopyOrMatchLength:8);
        // it must first send the literals that were prefetched eariler
        assert_eq(data.content, u64:0xffeeddccbbaafefd);

        let (tok_recv, data) = recv(tok, lit_buf_out1_r);
        assert_eq(data.msg_type, common::SequenceExecutorMessageType::LITERAL);
        assert_eq(data.length, common::CopyOrMatchLength:8);
        assert_eq(data.content, u64:0x0102030405060708);

        // the last literal
        let (tok_recv, data) = recv(tok, lit_buf_out1_r);
        assert_eq(data.msg_type, common::SequenceExecutorMessageType::LITERAL);
        assert_eq(data.length, common::CopyOrMatchLength:1);
        assert_eq(data.content, u64:0x08);

        send(tok, terminator, true);
    }
}
