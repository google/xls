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

// This file contains the implementation of RawBlockDecoder responsible for decoding
// ZSTD Raw Blocks. More information about Raw Block's format can be found in:
// https://datatracker.ietf.org/doc/html/rfc8878#section-3.1.1.2.2

import xls.modules.zstd.common as common;

type BlockDataPacket = common::BlockDataPacket;
type BlockPacketLength = common::BlockPacketLength;
type BlockData = common::BlockData;
type ExtendedBlockDataPacket = common::ExtendedBlockDataPacket;
type CopyOrMatchContent = common::CopyOrMatchContent;
type CopyOrMatchLength = common::CopyOrMatchLength;
type SequenceExecutorMessageType = common::SequenceExecutorMessageType;

struct RawBlockDecoderState {
    prev_id: u32, // ID of the previous block
    prev_last: bool, // if the previous packet was the last one that makes up the whole block
    prev_valid: bool, // if prev_id and prev_last contain valid data
}

const ZERO_RAW_BLOCK_DECODER_STATE = zero!<RawBlockDecoderState>();

// RawBlockDecoder is responsible for decoding Raw Blocks,
// it should be a part of the ZSTD Decoder pipeline.
pub proc RawBlockDecoder {
    input_r: chan<BlockDataPacket> in;
    output_s: chan<ExtendedBlockDataPacket> out;

    init { (ZERO_RAW_BLOCK_DECODER_STATE) }

    config(
        input_r: chan<BlockDataPacket> in,
        output_s: chan<ExtendedBlockDataPacket> out
    ) {(input_r, output_s)}

    next(state: RawBlockDecoderState) {
        let tok = join();
        let (tok, data) = recv(tok, input_r);
        if state.prev_valid && (data.id != state.prev_id) && (state.prev_last == false) {
            trace_fmt!("ID changed but previous packet have no last!");
            fail!("no_last", ());
        } else {};

        let output_data = ExtendedBlockDataPacket {
            // Decoded RAW block is always a literal
            msg_type: SequenceExecutorMessageType::LITERAL,
            packet: BlockDataPacket {
                last: data.last,
                last_block: data.last_block,
                id: data.id,
                data: data.data as BlockData,
                length: data.length as BlockPacketLength,
            },
        };

        let tok = send(tok, output_s, output_data);

        RawBlockDecoderState {
            prev_valid: true,
            prev_id: output_data.packet.id,
            prev_last: output_data.packet.last
        }
    }
}

#[test_proc]
proc RawBlockDecoderTest {
    terminator: chan<bool> out;
    dec_input_s: chan<BlockDataPacket> out;
    dec_output_r: chan<ExtendedBlockDataPacket> in;

    config(terminator: chan<bool> out) {
        let (dec_input_s, dec_input_r) = chan<BlockDataPacket>("dec_input");
        let (dec_output_s, dec_output_r) = chan<ExtendedBlockDataPacket>("dec_output");
        spawn RawBlockDecoder(dec_input_r, dec_output_s);
        (terminator, dec_input_s, dec_output_r)
    }

    init {  }

    next(state: ()) {
        let tok = join();
        let data_to_send: BlockDataPacket[5] = [
            BlockDataPacket { id: u32:1, last: u1:false, last_block: u1:false, data: BlockData:1, length: BlockPacketLength:32 },
            BlockDataPacket { id: u32:1, last: u1:false, last_block: u1:false, data: BlockData:2, length: BlockPacketLength:32 },
            BlockDataPacket { id: u32:1, last: u1:true, last_block: u1:false, data: BlockData:3, length: BlockPacketLength:32 },
            BlockDataPacket { id: u32:2, last: u1:false, last_block: u1:false, data: BlockData:4, length: BlockPacketLength:32 },
            BlockDataPacket { id: u32:2, last: u1:true, last_block: u1:true, data: BlockData:5, length: BlockPacketLength:32 },
        ];

        let tok = for ((_, data), tok): ((u32, BlockDataPacket), token) in enumerate(data_to_send) {
            let tok = send(tok, dec_input_s, data);
            let (tok, received_data) = recv(tok, dec_output_r);
            let expected_data = ExtendedBlockDataPacket {
                msg_type: SequenceExecutorMessageType::LITERAL,
                packet: data,
            };
            assert_eq(expected_data, received_data);
            (tok)
        }(tok);

        send(tok, terminator, true);
    }
}
