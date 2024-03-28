// Copyright 2023 The XLS Authors
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

// This file contains the DecoderMux Proc, which collects data from
// specialized Raw, RLE, and Compressed Block decoders and re-sends them in
// the correct order.

import std;
import xls.modules.zstd.common as common;

type BlockDataPacket = common::BlockDataPacket;
type ExtendedBlockDataPacket = common::ExtendedBlockDataPacket;
type BlockData = common::BlockData;
type BlockPacketLength = common::BlockPacketLength;
type CopyOrMatchContent = common::CopyOrMatchContent;
type CopyOrMatchLength = common::CopyOrMatchLength;
type SequenceExecutorMessageType = common::SequenceExecutorMessageType;
type SequenceExecutorPacket = common::SequenceExecutorPacket;

const MAX_ID = common::DATA_WIDTH;
const DATA_WIDTH = common::DATA_WIDTH;

struct DecoderMuxState {
    prev_id: u32,
    prev_last: bool,
    prev_last_block: bool,
    prev_valid: bool,
    raw_data: ExtendedBlockDataPacket,
    raw_data_valid: bool,
    rle_data: ExtendedBlockDataPacket,
    rle_data_valid: bool,
    compressed_data: ExtendedBlockDataPacket,
    compressed_data_valid: bool,
}

const ZERO_DECODER_MUX_STATE = zero!<DecoderMuxState>();

pub proc DecoderMux {
    raw_r: chan<ExtendedBlockDataPacket> in;
    rle_r: chan<ExtendedBlockDataPacket> in;
    cmp_r: chan<ExtendedBlockDataPacket> in;
    output_s: chan<SequenceExecutorPacket> out;

    init {(ZERO_DECODER_MUX_STATE)}

    config (
        raw_r: chan<ExtendedBlockDataPacket> in,
        rle_r: chan<ExtendedBlockDataPacket> in,
        cmp_r: chan<ExtendedBlockDataPacket> in,
        output_s: chan<SequenceExecutorPacket> out,
    ) {(raw_r, rle_r, cmp_r, output_s)}

    next (tok: token, state: DecoderMuxState) {
        let (tok, raw_data, raw_data_valid) = recv_if_non_blocking(
            tok, raw_r, !state.raw_data_valid, zero!<ExtendedBlockDataPacket>());
        let state = if (raw_data_valid) {
            DecoderMuxState {raw_data, raw_data_valid, ..state}
        } else { state };

        let (tok, rle_data, rle_data_valid) = recv_if_non_blocking(
            tok, rle_r, !state.rle_data_valid, zero!<ExtendedBlockDataPacket>());
        let state = if (rle_data_valid) {
            DecoderMuxState { rle_data, rle_data_valid, ..state}
        } else { state };

        let (tok, compressed_data, compressed_data_valid) = recv_if_non_blocking(
            tok, cmp_r, !state.compressed_data_valid, zero!<ExtendedBlockDataPacket>());
        let state = if (compressed_data_valid) {
            DecoderMuxState { compressed_data, compressed_data_valid, ..state}
        } else { state };

        let raw_id = if state.raw_data_valid { state.raw_data.packet.id } else { MAX_ID };
        let rle_id = if state.rle_data_valid { state.rle_data.packet.id } else { MAX_ID };
        let compressed_id = if state.compressed_data_valid { state.compressed_data.packet.id } else { MAX_ID };

        if state.prev_last_block && state.prev_last {
            if (std::umin(std::umin(rle_id, raw_id), compressed_id) != u32:0) {
                fail!("wrong_id_expected_0", ())
            } else {()};
        } else {
            if (state.prev_id > (std::umin(std::umin(rle_id, raw_id), compressed_id))) && (state.prev_valid) {
                fail!("wrong_id", ())
            } else {()};
        };

        let (do_send, data_to_send, state) = if (state.raw_data_valid &&
          ((state.raw_data.packet.id < std::umin(rle_id, compressed_id)) ||
           (state.raw_data.packet.id == state.prev_id))) {
            (true,
             SequenceExecutorPacket {
                 msg_type: state.raw_data.msg_type,
                 length: state.raw_data.packet.length as CopyOrMatchLength,
                 content: state.raw_data.packet.data as CopyOrMatchContent,
                 last: state.raw_data.packet.last && state.raw_data.packet.last_block,
             },
             DecoderMuxState {
                 raw_data_valid: false,
                 prev_valid : true,
                 prev_id: state.raw_data.packet.id,
                 prev_last: state.raw_data.packet.last,
                 prev_last_block: state.raw_data.packet.last_block,
                 ..state})
        } else if (state.rle_data_valid &&
                 ((state.rle_data.packet.id < std::umin(raw_id, compressed_id)) ||
                  (state.rle_data.packet.id == state.prev_id))) {
            (true,
             SequenceExecutorPacket {
                 msg_type: state.rle_data.msg_type,
                 length: state.rle_data.packet.length as CopyOrMatchLength,
                 content: state.rle_data.packet.data as CopyOrMatchContent,
                 last: state.rle_data.packet.last && state.rle_data.packet.last_block,
             },
             DecoderMuxState {
                 rle_data_valid: false,
                 prev_valid : true,
                 prev_id: state.rle_data.packet.id,
                 prev_last: state.rle_data.packet.last,
                 prev_last_block: state.rle_data.packet.last_block,
                 ..state})
        } else if (state.compressed_data_valid &&
                 ((state.compressed_data.packet.id < std::umin(raw_id, rle_id)) ||
                  (state.compressed_data.packet.id == state.prev_id))) {
            (true,
             SequenceExecutorPacket {
                 msg_type: state.compressed_data.msg_type,
                 length: state.compressed_data.packet.length as CopyOrMatchLength,
                 content: state.compressed_data.packet.data as CopyOrMatchContent,
                 last: state.compressed_data.packet.last && state.compressed_data.packet.last_block,
             },
             DecoderMuxState {
                 compressed_data_valid: false,
                 prev_valid : true,
                 prev_id: state.compressed_data.packet.id,
                 prev_last: state.compressed_data.packet.last,
                 prev_last_block: state.compressed_data.packet.last_block,
                 ..state})
        } else {
            (false, zero!<SequenceExecutorPacket>(), state)
        };

        let tok = send_if(tok, output_s, do_send, data_to_send);
        if (do_send) {
            trace_fmt!("sent {:#x}", data_to_send);
        } else {()};
        state
    }
}

#[test_proc]
proc DecoderMuxTest {
  terminator: chan<bool> out;
  raw_s: chan<ExtendedBlockDataPacket> out;
  rle_s: chan<ExtendedBlockDataPacket> out;
  cmp_s: chan<ExtendedBlockDataPacket> out;
  output_r: chan<SequenceExecutorPacket> in;

  init {}

  config (terminator: chan<bool> out) {
    let (raw_s, raw_r) = chan<ExtendedBlockDataPacket>;
    let (rle_s, rle_r) = chan<ExtendedBlockDataPacket>;
    let (cmp_s, cmp_r) = chan<ExtendedBlockDataPacket>;
    let (output_s, output_r) = chan<SequenceExecutorPacket>;

    spawn DecoderMux(raw_r, rle_r, cmp_r, output_s);
    (terminator, raw_s, rle_s, cmp_s, output_r)
  }

  next(tok: token, state: ()) {
    let tok = send(tok, raw_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:1, last: bool: false, last_block: bool: false, data: BlockData:0x11111111, length: BlockPacketLength:32 }});
    let tok = send(tok, raw_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:1, last: bool: false, last_block: bool: false, data: BlockData:0x22222222, length: BlockPacketLength:32 }});
    let tok = send(tok, rle_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:2, last: bool: false, last_block: bool: false, data: BlockData:0xAAAAAAAA, length: BlockPacketLength:32 }});
    let tok = send(tok, raw_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:1, last: bool: true,  last_block: bool: false, data: BlockData:0x33333333, length: BlockPacketLength:32 }});
    let tok = send(tok, cmp_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:3, last: bool: true,  last_block: bool: true,  data: BlockData:0x00000000, length: BlockPacketLength:32 }});
    let tok = send(tok, rle_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:2, last: bool: true,  last_block: bool: false, data: BlockData:0xBBBBBBBB, length: BlockPacketLength:32 }});

    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0x11111111, length: CopyOrMatchLength:32 });
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0x22222222, length: CopyOrMatchLength:32 });
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0x33333333, length: CopyOrMatchLength:32 });
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0xAAAAAAAA, length: CopyOrMatchLength:32 });
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0xBBBBBBBB, length: CopyOrMatchLength:32 });
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: true, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0x00000000, length: CopyOrMatchLength:32 });

    send(tok, terminator, true);
  }
}
