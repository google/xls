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

// This file contains implementation of CommandConstructor, which adjusts
// the data obtained from the SequenceDecoder for the SequenceExecutor block.
// It can receive two types of values: sequences that are copied directly
// to the output, and length-only literals packets, which before sending
// them to the output, should be redirected to the LiteralsBuffer to be filled
// with actual data.

import std;
import xls.modules.zstd.common;

type SequenceExecutorPacket = common::SequenceExecutorPacket;
type SequenceExecutorMessageType = common::SequenceExecutorMessageType;
type CopyOrMatchContent = common::CopyOrMatchContent;
type CopyOrMatchLength = common::CopyOrMatchLength;
type LiteralsBufferCtrl = common::LiteralsBufferCtrl;

enum Status : u1 {
    RECV_COMMAND = 0,
    RECV_LITERALS = 1,
}

struct State {
    status: Status,
    received_literals: CopyOrMatchLength,
    literals_to_receive: CopyOrMatchLength,
}

proc CommandConstructor {
    sequence_decoder_r: chan<SequenceExecutorPacket> in;
    command_aggregator_s: chan<SequenceExecutorPacket> out;
    literals_buffer_resp_r: chan<SequenceExecutorPacket> in;
    literals_buffer_req_s: chan<LiteralsBufferCtrl> out;

    config(sequence_decoder_r: chan<SequenceExecutorPacket> in,
           command_aggregator_s: chan<SequenceExecutorPacket> out,
           literals_buffer_resp_r: chan<SequenceExecutorPacket> in,
           literals_buffer_req_s: chan<LiteralsBufferCtrl> out) {
        (sequence_decoder_r, command_aggregator_s, literals_buffer_resp_r, literals_buffer_req_s)
    }

    init { zero!<State>() }

    next(state: State) {
        let tok0 = join();

        let recv_command = state.status == Status::RECV_COMMAND;
        let (tok1_0, command) =
            recv_if(tok0, sequence_decoder_r, recv_command, zero!<SequenceExecutorPacket>());

        let recv_literals = state.status == Status::RECV_LITERALS;
        let (tok1_1, literals) =
            recv_if(tok0, literals_buffer_resp_r, recv_literals, zero!<SequenceExecutorPacket>());

        let tok1 = join(tok1_0, tok1_1);

        let (new_state, do_send_command, do_send_literals_req) = match (state.status) {
            Status::RECV_COMMAND => {
                if command.msg_type == SequenceExecutorMessageType::LITERAL {
                    (
                        State {
                            status: Status::RECV_LITERALS,
                            received_literals: CopyOrMatchLength:0,
                            literals_to_receive: command.length
                        }, false, true,
                    )
                } else {
                    (zero!<State>(), true, false)
                }
                },
            Status::RECV_LITERALS => {
                let received_literals = state.received_literals + literals.length;
                if received_literals < state.literals_to_receive {
                    (State { received_literals, ..state }, true, false)
                } else {
                    assert!(
                        received_literals >= state.literals_to_receive,
                        "Too many literals received");
                    (zero!<State>(), true, false)
                }
            },
            _ => fail!("impossible_case", (zero!<State>(), false, false)),
        };

        let req = LiteralsBufferCtrl { length: command.length };
        send_if(tok1, literals_buffer_req_s, do_send_literals_req, req);

        let resp = if state.status == Status::RECV_COMMAND { command } else { literals };
        send_if(tok1, command_aggregator_s, do_send_command, resp);

        new_state
    }
}

// Tests

enum FakeLiteralsBufferStatus : u1 {
    RECV = 0,
    SEND = 1,
}

struct FakeLiteralsBufferState {
    status: FakeLiteralsBufferStatus,
    literals_left_to_send: CopyOrMatchLength,
}

pub fn get_dummy_content(length: CopyOrMatchLength) -> CopyOrMatchContent {
    let value = std::unsigned_max_value<u32:64>() >> (CopyOrMatchLength:64 - length);
    value as CopyOrMatchContent
}

proc FakeLiteralsBuffer {
    literals_buffer_resp_s: chan<SequenceExecutorPacket> out;
    literals_buffer_req_r: chan<LiteralsBufferCtrl> in;

    config(literals_buffer_resp_s: chan<SequenceExecutorPacket> out,
           literals_buffer_req_r: chan<LiteralsBufferCtrl> in) {
        (literals_buffer_resp_s, literals_buffer_req_r)
    }

    init { zero!<FakeLiteralsBufferState>() }

    next(state: FakeLiteralsBufferState) {
        let tok = join();
        let do_recv_req = state.status == FakeLiteralsBufferStatus::RECV;
        let (tok, resp) =
            recv_if(tok, literals_buffer_req_r, do_recv_req, zero!<LiteralsBufferCtrl>());

        let (new_state, do_send, resp) = match (state.status) {
            FakeLiteralsBufferStatus::RECV => {
                (
                    FakeLiteralsBufferState {
                        status: FakeLiteralsBufferStatus::SEND,
                        literals_left_to_send: resp.length
                    }, false, zero!<SequenceExecutorPacket>(),
                )
                },
            FakeLiteralsBufferStatus::SEND => {
                let length = std::umin(state.literals_left_to_send, CopyOrMatchLength:64);
                let next_left_to_send = state.literals_left_to_send - length;
                let last = next_left_to_send == CopyOrMatchLength:0;
                let resp = SequenceExecutorPacket {
                    msg_type: SequenceExecutorMessageType::LITERAL,
                    content: get_dummy_content(length),
                    length,
                    last
                };

                if last {
                    (
                        FakeLiteralsBufferState {
                            status: FakeLiteralsBufferStatus::RECV,
                            literals_left_to_send: CopyOrMatchLength:0
                        }, true, resp,
                    )
                } else {
                    (
                        FakeLiteralsBufferState {
                            status: FakeLiteralsBufferStatus::SEND,
                            literals_left_to_send: next_left_to_send
                        }, true, resp,
                    )
                }
            },
            _ => {
                fail!(
                    "impossible_case",
                    (zero!<FakeLiteralsBufferState>(), false, zero!<SequenceExecutorPacket>()))
                },
        };

        send_if(tok, literals_buffer_resp_s, do_send, resp);
        new_state
    }
}

#[test_proc]
proc CommandConstructorTest {
    terminator: chan<bool> out;
    sequence_decoder_s: chan<SequenceExecutorPacket> out;
    command_aggregator_r: chan<SequenceExecutorPacket> in;

    config(terminator: chan<bool> out) {
        let (sequence_decoder_s, sequence_decoder_r) = chan<SequenceExecutorPacket>("sequence_decoder");
        let (command_aggregator_s, command_aggregator_r) = chan<SequenceExecutorPacket>("command_aggregator");

        let (literals_buffer_resp_s, literals_buffer_resp_r) = chan<SequenceExecutorPacket>("literals_buffer_resp");
        let (literals_buffer_req_s, literals_buffer_req_r) = chan<LiteralsBufferCtrl>("literals_buffer_req");

        spawn CommandConstructor(
            sequence_decoder_r, command_aggregator_s, literals_buffer_resp_r, literals_buffer_req_s);

        spawn FakeLiteralsBuffer(literals_buffer_resp_s, literals_buffer_req_r);

        (terminator, sequence_decoder_s, command_aggregator_r)
    }

    init {  }

    next(state: ()) {
        const EMPTY_PACKET = zero!<SequenceExecutorPacket>();

        let tok = join();

        let sequence_packet1 = SequenceExecutorPacket {
            msg_type: SequenceExecutorMessageType::SEQUENCE,
            content: CopyOrMatchContent:11,
            length: CopyOrMatchLength:4,
            last: true
        };
        let tok = send(tok, sequence_decoder_s, sequence_packet1);
        let (tok, resp) = recv(tok, command_aggregator_r);
        assert_eq(sequence_packet1, resp);

        let literals_packet1 = SequenceExecutorPacket {
            msg_type: SequenceExecutorMessageType::LITERAL, length: CopyOrMatchLength:4,
        ..EMPTY_PACKET
        };
        let tok = send(tok, sequence_decoder_s, literals_packet1);
        let (tok, resp) = recv(tok, command_aggregator_r);
        assert_eq(get_dummy_content(CopyOrMatchLength:4), resp.content);

        let literals_packet2 = SequenceExecutorPacket {
            msg_type: SequenceExecutorMessageType::LITERAL, length: CopyOrMatchLength:65,
        ..EMPTY_PACKET
        };
        let tok = send(tok, sequence_decoder_s, literals_packet2);
        let (tok, resp) = recv(tok, command_aggregator_r);
        assert_eq(get_dummy_content(CopyOrMatchLength:64), resp.content);
        let (tok, resp) = recv(tok, command_aggregator_r);
        assert_eq(get_dummy_content(CopyOrMatchLength:1), resp.content);

        let literals_packet3 = SequenceExecutorPacket {
            msg_type: SequenceExecutorMessageType::LITERAL, length: CopyOrMatchLength:64,
        ..EMPTY_PACKET
        };
        let tok = send(tok, sequence_decoder_s, literals_packet3);
        let (tok, resp) = recv(tok, command_aggregator_r);
        assert_eq(get_dummy_content(CopyOrMatchLength:64), resp.content);

        let literals_packet4 = SequenceExecutorPacket {
            msg_type: SequenceExecutorMessageType::LITERAL, length: CopyOrMatchLength:128,
        ..EMPTY_PACKET
        };
        let tok = send(tok, sequence_decoder_s, literals_packet4);
        let (tok, resp) = recv(tok, command_aggregator_r);
        assert_eq(get_dummy_content(CopyOrMatchLength:64), resp.content);
        let (tok, resp) = recv(tok, command_aggregator_r);
        assert_eq(get_dummy_content(CopyOrMatchLength:64), resp.content);

        let tok = send(tok, terminator, true);
    }
}
