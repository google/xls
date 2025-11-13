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

type SequenceExecutorMessageType = common::SequenceExecutorMessageType;
type CopyOrMatchContent = common::CopyOrMatchContent;
type CopyOrMatchLength = common::CopyOrMatchLength;

type SequenceExecutorPacket = common::SequenceExecutorPacket<common::SYMBOL_WIDTH>;
type LiteralsBufferCtrl = common::LiteralsBufferCtrl;
type CommandConstructorData = common::CommandConstructorData;
type ExtendedBlockDataPacket = common::ExtendedBlockDataPacket;
type BlockSyncData = common::BlockSyncData;
type BlockDataPacket = common::BlockDataPacket;

pub proc CommandConstructor {
    sequence_decoder_r: chan<CommandConstructorData> in;
    command_aggregator_s: chan<ExtendedBlockDataPacket> out;

    config(sequence_decoder_r: chan<CommandConstructorData> in,
           command_aggregator_s: chan<ExtendedBlockDataPacket> out) {
        (sequence_decoder_r, command_aggregator_s)
    }

    init { }

    next(state: ()) {
        let tok0 = join();

        let (tok0, sequence_command) = recv(tok0, sequence_decoder_r);
        trace_fmt!("[CommandConstructor] Received sequence command: {:#x}", sequence_command);

        let resp = ExtendedBlockDataPacket {
            msg_type: SequenceExecutorMessageType::SEQUENCE,
            packet: BlockDataPacket {
                last: sequence_command.data.last,
                last_block: sequence_command.sync.last_block,
                id: sequence_command.sync.id,
                data: sequence_command.data.content,
                length: sequence_command.data.length as u32,
            },
        };

        let tok0 = send(tok0, command_aggregator_s, resp);
        trace_fmt!("[CommandConstructor] Sending command: {:#x}", resp);
    }
}

#[test_proc]
proc CommandConstructorTest {
    type SequenceExecutorPacket = common::SequenceExecutorPacket<common::SYMBOL_WIDTH>;

    terminator: chan<bool> out;
    sequence_decoder_s: chan<CommandConstructorData> out;
    command_aggregator_r: chan<ExtendedBlockDataPacket> in;

    config(terminator: chan<bool> out) {
        let (sequence_decoder_s, sequence_decoder_r) = chan<CommandConstructorData>("sequence_decoder");
        let (command_aggregator_s, command_aggregator_r) = chan<ExtendedBlockDataPacket>("command_aggregator");

        spawn CommandConstructor(sequence_decoder_r, command_aggregator_s);

        (terminator, sequence_decoder_s, command_aggregator_r)
    }

    init {  }

    next(state: ()) {
        let tok = join();

        let sequence_packet1 = CommandConstructorData {
            data: SequenceExecutorPacket {
                msg_type: SequenceExecutorMessageType::SEQUENCE,
                content: CopyOrMatchContent:0x1005b,
                length: CopyOrMatchLength:9,
                last: false,
            },
            sync: BlockSyncData {
                id: u32:1234,
                last_block: false,
            },
        };
        let tok = send(tok, sequence_decoder_s, sequence_packet1);
        let (tok, resp) = recv(tok, command_aggregator_r);

        trace_fmt!("[CommandConstructorTest] Received command: {:#x}", resp);
        assert_eq(resp, ExtendedBlockDataPacket {
            msg_type: SequenceExecutorMessageType::SEQUENCE,
            packet: BlockDataPacket {
                last: u1:0,
                last_block: u1:0,
                id: u32:1234,
                data: u64:0x1005b,
                length: u32:9,
            },
        });

        let tok = send(tok, terminator, true);
    }
}
