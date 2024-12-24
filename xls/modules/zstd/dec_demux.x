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

// This file contains DecoderDemux Proc, which is responsible for
// parsing Block_Header and sending the obtained data to the Raw, RLE,
// or Compressed Block decoders.

import std;
import xls.modules.zstd.common as common;
import xls.modules.zstd.block_header as block_header;

type BlockDataPacket = common::BlockDataPacket;

const DATA_WIDTH = common::DATA_WIDTH;

enum DecoderDemuxStatus : u2 {
    IDLE = 0,
    PASS_RAW = 1,
    PASS_RLE = 2,
    PASS_COMPRESSED = 3,
}

struct DecoderDemuxState {
    status: DecoderDemuxStatus,
    byte_to_pass: u21,
    send_data: u21,
    id: u32,
    last_packet: BlockDataPacket,
}

// It's safe to assume that data contains full header and some extra data.
// Previous stage aligns block header and data, it also guarantees
// new block headers in new packets.
fn handle_idle_state(data: BlockDataPacket, state: DecoderDemuxState)
  -> DecoderDemuxState {
    let header = block_header::extract_block_header(data.data[0:24] as u24);
    let data = BlockDataPacket {
        data: data.data[24:] as bits[DATA_WIDTH],
        length: data.length - u32:24,
        id: state.id,
        ..data
    };
    match header.btype {
        common::BlockType::RAW => {
            DecoderDemuxState {
                status: DecoderDemuxStatus::PASS_RAW,
                byte_to_pass: header.size,
                send_data: u21:0,
                last_packet: data,
                ..state
            }
        },
        common::BlockType::RLE => {
            DecoderDemuxState {
                status: DecoderDemuxStatus::PASS_RLE,
                byte_to_pass: header.size,
                send_data: u21:0,
                last_packet: data,
                ..state
            }
        },
        common::BlockType::COMPRESSED => {
            DecoderDemuxState {
                status: DecoderDemuxStatus::PASS_COMPRESSED,
                byte_to_pass: header.size,
                send_data: u21:0,
                last_packet: data,
                ..state
            }
        },
        _ => {
            fail!("Should_never_happen", state)
        }
    }
}

const ZERO_DECODER_DEMUX_STATE = zero!<DecoderDemuxState>();
const ZERO_DATA = zero!<BlockDataPacket>();

pub proc DecoderDemux {
    input_r: chan<BlockDataPacket> in;
    raw_s: chan<BlockDataPacket> out;
    rle_s: chan<BlockDataPacket> out;
    cmp_s: chan<BlockDataPacket> out;

    init {(ZERO_DECODER_DEMUX_STATE)}

    config (
        input_r: chan<BlockDataPacket> in,
        raw_s: chan<BlockDataPacket> out,
        rle_s: chan<BlockDataPacket> out,
        cmp_s: chan<BlockDataPacket> out,
    ) {(
        input_r,
        raw_s,
        rle_s,
        cmp_s
    )}

    next (state: DecoderDemuxState) {
        let tok = join();
        let (tok, data) = recv_if(tok, input_r, !state.last_packet.last, ZERO_DATA);
        if (!state.last_packet.last) {
            trace_fmt!("DecoderDemux: recv: {:#x}", data);
        } else {};
        let (send_raw, send_rle, send_cmp, new_state) = match state.status {
            DecoderDemuxStatus::IDLE =>
                (false, false, false, handle_idle_state(data, state)),
            DecoderDemuxStatus::PASS_RAW => {
                let new_state = DecoderDemuxState {
                    send_data: state.send_data + (state.last_packet.length >> 3) as u21,
                    last_packet: data,
                    ..state
                };
                (true, false, false, new_state)
            },
            DecoderDemuxStatus::PASS_RLE => {
                let new_state = DecoderDemuxState {
                    send_data: state.send_data + state.byte_to_pass,
                    last_packet: data,
                    ..state
                };
                (false, true, false, new_state)
            },
            DecoderDemuxStatus::PASS_COMPRESSED => {
                let new_state = DecoderDemuxState {
                    send_data: state.send_data +(state.last_packet.length >> 3) as u21,
                    last_packet: data,
                    ..state
                };
                (false, false, true, new_state)
            },
        };

        let end_state = if (send_raw || send_rle || send_cmp) {
            let max_packet_width = DATA_WIDTH;
            let block_size_bits = u32:24 + (state.byte_to_pass as u32 << 3);
            if (!send_rle) && ((block_size_bits <= max_packet_width) &&
                ((block_size_bits) != state.last_packet.length) && !state.last_packet.last) {
                // Demuxer expect that blocks would be received in a separate packets,
                // even if 2 block would fit entirely or even partially in a single packet.
                // It is the job of top-level ZSTD decoder to split each block into at least one
                // BlockDataPacket.
                // For Raw and Compressed blocks it is illegal to have block of size smaller than
                // max size of packet and have packet length greater than this size.
                fail!("Should_never_happen", state)
            } else {
                state
            };
            let data_to_send = BlockDataPacket {id: state.id, ..state.last_packet};
            let tok = send_if(tok, raw_s, send_raw, data_to_send);
            if (send_raw) {
                trace_fmt!("DecoderDemux: send_raw: {:#x}", data_to_send);
            } else {};
            // RLE module expects single byte in data field
            // and block length in length field. This is different from
            // Raw and Compressed modules.
            let rle_data = BlockDataPacket{
                data: state.last_packet.data[0:8] as bits[DATA_WIDTH],
                length: state.byte_to_pass as u32,
                id: state.id,
                ..state.last_packet
            };
            let tok = send_if(tok, rle_s, send_rle, rle_data);
            if (send_rle) {
                trace_fmt!("DecoderDemux: send_rle: {:#x}", rle_data);
            } else {};
            let tok = send_if(tok, cmp_s, send_cmp, data_to_send);
            if (send_cmp) {
                trace_fmt!("DecoderDemux: send_cmp: {:#x}", data_to_send);
            } else {};
            let end_state = if (new_state.send_data == new_state.byte_to_pass) {
                let next_id = if (state.last_packet.last && state.last_packet.last_block) {
                    u32: 0
                } else {
                    state.id + u32:1
                };
                DecoderDemuxState {
                    status: DecoderDemuxStatus::IDLE,
                    byte_to_pass: u21:0,
                    send_data: u21:0,
                    id: next_id,
                    last_packet: ZERO_DATA,
                }
            } else {
                new_state
            };
            end_state
        } else {
            new_state
        };

        end_state
    }
}

#[test_proc]
proc DecoderDemuxTest {
  terminator: chan<bool> out;
  input_s: chan<BlockDataPacket> out;
  raw_r: chan<BlockDataPacket> in;
  rle_r: chan<BlockDataPacket> in;
  cmp_r: chan<BlockDataPacket> in;

  init {}

  config (terminator: chan<bool> out) {
    let (raw_s, raw_r) = chan<BlockDataPacket, u32:1>("raw");
    let (rle_s, rle_r) = chan<BlockDataPacket>("rle");
    let (cmp_s, cmp_r) = chan<BlockDataPacket>("cmp");
    let (input_s, input_r) = chan<BlockDataPacket>("input");

    spawn DecoderDemux(input_r, raw_s, rle_s, cmp_s);
    (terminator, input_s, raw_r, rle_r, cmp_r)
  }

  next(state: ()) {
    let tok = join();
    let tok = send(tok, input_s, BlockDataPacket { id: u32:0, last: bool: false, last_block: bool: false, data: bits[DATA_WIDTH]:0x11111111110000c0, length: u32:64 });
    let tok = send(tok, input_s, BlockDataPacket { id: u32:0, last: bool: false, last_block: bool: false, data: bits[DATA_WIDTH]:0x2222222222111111, length: u32:64 });
    let tok = send(tok, input_s, BlockDataPacket { id: u32:0, last: bool: false, last_block: bool: false, data: bits[DATA_WIDTH]:0x3333333333222222, length: u32:64 });
    let tok = send(tok, input_s, BlockDataPacket { id: u32:0, last: bool: true,  last_block: bool: false, data: bits[DATA_WIDTH]:0x0000000000333333, length: u32:24 });

    let tok = send(tok, input_s, BlockDataPacket { id: u32:0, last: bool: false, last_block: bool: false, data: bits[DATA_WIDTH]:0xAAAAAAAAAA000100, length: u32:64 });
    let tok = send(tok, input_s, BlockDataPacket { id: u32:0, last: bool: false, last_block: bool: false, data: bits[DATA_WIDTH]:0xBBBBBBBBBBAAAAAA, length: u32:64 });
    let tok = send(tok, input_s, BlockDataPacket { id: u32:0, last: bool: false, last_block: bool: false, data: bits[DATA_WIDTH]:0xCCCCCCCCCCBBBBBB, length: u32:64 });
    let tok = send(tok, input_s, BlockDataPacket { id: u32:0, last: bool: false, last_block: bool: false, data: bits[DATA_WIDTH]:0x0000000000CCCCCC, length: u32:24 });
    let tok = send(tok, input_s, BlockDataPacket { id: u32:0, last: bool: true,  last_block: bool: false, data: bits[DATA_WIDTH]:0xDDDDDDDDDDDDDDDD, length: u32:64 });

    let tok = send(tok, input_s, BlockDataPacket { id: u32:0, last: bool: true, last_block: bool: false, data: bits[DATA_WIDTH]:0x0000000FF000102, length: u32:32 });

    let tok = send(tok, input_s, BlockDataPacket { id: u32:0, last: bool: false, last_block: bool: true, data: bits[DATA_WIDTH]:0x4444444444000145, length: u32:64 });
    let tok = send(tok, input_s, BlockDataPacket { id: u32:0, last: bool: false, last_block: bool: true, data: bits[DATA_WIDTH]:0x5555555555444444, length: u32:64 });
    let tok = send(tok, input_s, BlockDataPacket { id: u32:0, last: bool: false, last_block: bool: true, data: bits[DATA_WIDTH]:0x6666666666555555, length: u32:64 });
    let tok = send(tok, input_s, BlockDataPacket { id: u32:0, last: bool: false, last_block: bool: true, data: bits[DATA_WIDTH]:0x7777777777666666, length: u32:64 });
    let tok = send(tok, input_s, BlockDataPacket { id: u32:0, last: bool: false, last_block: bool: true, data: bits[DATA_WIDTH]:0x8888888888777777, length: u32:64 });
    let tok = send(tok, input_s, BlockDataPacket { id: u32:0, last: bool: true,  last_block: bool: true, data: bits[DATA_WIDTH]:0x0000000000888888, length: u32:24 });

    let (tok, data) = recv(tok, raw_r); assert_eq(data, BlockDataPacket { id: u32:0, last: bool: false, last_block: bool: false, data: bits[DATA_WIDTH]:0x0000001111111111, length: u32:40 });
    let (tok, data) = recv(tok, raw_r); assert_eq(data, BlockDataPacket { id: u32:0, last: bool: false, last_block: bool: false, data: bits[DATA_WIDTH]:0x2222222222111111, length: u32:64 });
    let (tok, data) = recv(tok, raw_r); assert_eq(data, BlockDataPacket { id: u32:0, last: bool: false, last_block: bool: false, data: bits[DATA_WIDTH]:0x3333333333222222, length: u32:64 });
    let (tok, data) = recv(tok, raw_r); assert_eq(data, BlockDataPacket { id: u32:0, last: bool: true,  last_block: bool: false, data: bits[DATA_WIDTH]:0x0000000000333333, length: u32:24 });

    let (tok, data) = recv(tok, raw_r); assert_eq(data, BlockDataPacket { id: u32:1, last: bool: false, last_block: bool: false, data: bits[DATA_WIDTH]:0x000000AAAAAAAAAA, length: u32:40 });
    let (tok, data) = recv(tok, raw_r); assert_eq(data, BlockDataPacket { id: u32:1, last: bool: false, last_block: bool: false, data: bits[DATA_WIDTH]:0xBBBBBBBBBBAAAAAA, length: u32:64 });
    let (tok, data) = recv(tok, raw_r); assert_eq(data, BlockDataPacket { id: u32:1, last: bool: false, last_block: bool: false, data: bits[DATA_WIDTH]:0xCCCCCCCCCCBBBBBB, length: u32:64 });
    let (tok, data) = recv(tok, raw_r); assert_eq(data, BlockDataPacket { id: u32:1, last: bool: false, last_block: bool: false, data: bits[DATA_WIDTH]:0x0000000000CCCCCC, length: u32:24 });
    let (tok, data) = recv(tok, raw_r); assert_eq(data, BlockDataPacket { id: u32:1, last: bool: true,  last_block: bool: false, data: bits[DATA_WIDTH]:0xDDDDDDDDDDDDDDDD, length: u32:64 });

    let (tok, data) = recv(tok, rle_r); assert_eq(data, BlockDataPacket { id: u32:2, last: bool: true,  last_block: bool: false, data: bits[DATA_WIDTH]:0xFF, length: u32:32 });

    let (tok, data) = recv(tok, cmp_r); assert_eq(data, BlockDataPacket { id: u32:3, last: bool: false, last_block: bool: true, data: bits[DATA_WIDTH]:0x0000004444444444, length: u32:40 });
    let (tok, data) = recv(tok, cmp_r); assert_eq(data, BlockDataPacket { id: u32:3, last: bool: false, last_block: bool: true, data: bits[DATA_WIDTH]:0x5555555555444444, length: u32:64 });
    let (tok, data) = recv(tok, cmp_r); assert_eq(data, BlockDataPacket { id: u32:3, last: bool: false, last_block: bool: true, data: bits[DATA_WIDTH]:0x6666666666555555, length: u32:64 });
    let (tok, data) = recv(tok, cmp_r); assert_eq(data, BlockDataPacket { id: u32:3, last: bool: false, last_block: bool: true, data: bits[DATA_WIDTH]:0x7777777777666666, length: u32:64 });
    let (tok, data) = recv(tok, cmp_r); assert_eq(data, BlockDataPacket { id: u32:3, last: bool: false, last_block: bool: true, data: bits[DATA_WIDTH]:0x8888888888777777, length: u32:64 });
    let (tok, data) = recv(tok, cmp_r); assert_eq(data, BlockDataPacket { id: u32:3, last: bool: true,  last_block: bool: true, data: bits[DATA_WIDTH]:0x0000000000888888, length: u32:24 });

    send(tok, terminator, true);
  }
}
