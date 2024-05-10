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
type SequenceExecutorPacket = common::SequenceExecutorPacket<common::SYMBOL_WIDTH>;

const MAX_ID = common::DATA_WIDTH;
const DATA_WIDTH = common::DATA_WIDTH;

struct DecoderMuxState {
    prev_id: u32,
    prev_last: bool,
    prev_last_block: bool,
    prev_valid: bool,
    raw_data: ExtendedBlockDataPacket,
    raw_data_valid: bool,
    raw_data_valid_next_frame: bool,
    rle_data: ExtendedBlockDataPacket,
    rle_data_valid: bool,
    rle_data_valid_next_frame: bool,
    compressed_data: ExtendedBlockDataPacket,
    compressed_data_valid: bool,
    compressed_data_valid_next_frame: bool,
}

const ZERO_DECODER_MUX_STATE = zero!<DecoderMuxState>();

pub proc DecoderMux {
    raw_r: chan<ExtendedBlockDataPacket> in;
    rle_r: chan<ExtendedBlockDataPacket> in;
    cmp_r: chan<ExtendedBlockDataPacket> in;
    output_s: chan<SequenceExecutorPacket> out;

    init {( DecoderMuxState { prev_id: u32:0xFFFFFFFF, prev_last: true, prev_last_block: true, ..ZERO_DECODER_MUX_STATE } )}

    config (
        raw_r: chan<ExtendedBlockDataPacket> in,
        rle_r: chan<ExtendedBlockDataPacket> in,
        cmp_r: chan<ExtendedBlockDataPacket> in,
        output_s: chan<SequenceExecutorPacket> out,
    ) {(raw_r, rle_r, cmp_r, output_s)}

    next (state: DecoderMuxState) {
        let tok = join();
        let (tok, raw_data, raw_data_valid) = recv_if_non_blocking(
            tok, raw_r, !state.raw_data_valid && !state.raw_data_valid_next_frame, zero!<ExtendedBlockDataPacket>());
        let state = if (raw_data_valid) {
            let state = if (raw_data.packet.id <= state.prev_id && state.prev_last && state.prev_valid && !state.prev_last_block) {
                // received ID the same as previous, but `last` occurred
                // this might be a packet from the next frame
                let raw_data_valid_next_frame = raw_data_valid;
                DecoderMuxState {raw_data, raw_data_valid_next_frame, ..state}
            } else {
                DecoderMuxState {raw_data, raw_data_valid, ..state}
            };
            state
        } else { state };

        let (tok, rle_data, rle_data_valid) = recv_if_non_blocking(
            tok, rle_r, !state.rle_data_valid && !state.rle_data_valid_next_frame, zero!<ExtendedBlockDataPacket>());
        let state = if (rle_data_valid) {
            trace_fmt!("DecoderMux: received RLE data packet {:#x}", rle_data);
            let state = if (rle_data.packet.id <= state.prev_id && state.prev_last && state.prev_valid && !state.prev_last_block) {
                // received ID the same as previous, but `last` occurred
                // this might be a packet from the next frame
                let rle_data_valid_next_frame = rle_data_valid;
                DecoderMuxState {rle_data, rle_data_valid_next_frame, ..state}
            } else {
                DecoderMuxState {rle_data, rle_data_valid, ..state}
            };
            state
        } else { state };

        let (tok, compressed_data, compressed_data_valid) = recv_if_non_blocking(
            tok, cmp_r, !state.compressed_data_valid && !state.compressed_data_valid_next_frame, zero!<ExtendedBlockDataPacket>());
        let state = if (compressed_data_valid) {
            trace_fmt!("DecoderMux: received compressed data packet {:#x}", compressed_data);
            let state = if (compressed_data.packet.id <= state.prev_id && state.prev_last && state.prev_valid && !state.prev_last_block) {
                // received ID the same as previous, but `last` occurred
                // this might be a packet from the next frame
                let compressed_data_valid_next_frame = compressed_data_valid;
                DecoderMuxState {compressed_data, compressed_data_valid_next_frame, ..state}
            } else {
                DecoderMuxState {compressed_data, compressed_data_valid, ..state}
            };
            state
        } else { state };

        let raw_id = if state.raw_data_valid { state.raw_data.packet.id } else { MAX_ID };
        let rle_id = if state.rle_data_valid { state.rle_data.packet.id } else { MAX_ID };
        let compressed_id = if state.compressed_data_valid { state.compressed_data.packet.id } else { MAX_ID };
        let any_valid = state.raw_data_valid || state.rle_data_valid || state.compressed_data_valid;
        let all_valid = state.raw_data_valid && state.rle_data_valid && state.compressed_data_valid;

        let state = if (any_valid) {
            let min_id = std::min(std::min(rle_id, raw_id), compressed_id);
            trace_fmt!("DecoderMux: rle_id: {}, raw_id: {}, compressed_id: {}", rle_id, raw_id, compressed_id);
            trace_fmt!("DecoderMux: min_id: {}", min_id);

            assert!((state.prev_id <= min_id) || !state.prev_valid || state.prev_last_block, "wrong_id");
            assert!(!state.prev_last_block || !state.prev_last || min_id == u32:0, "wrong_id_expected_0");
            assert!(state.prev_last_block || !state.prev_last || !all_valid || (min_id == (state.prev_id + u32:1)) || (min_id == state.prev_id), "id_continuity_failure");

            let (do_send, data_to_send, state) =
                if (state.raw_data_valid &&
                 (((state.raw_data.packet.id == (state.prev_id + u32:1)) && state.prev_last) ||
                  ((state.raw_data.packet.id == state.prev_id) && !state.prev_last))) {
                    assert!(!state.raw_data_valid_next_frame, "raw_packet_valid_in_current_and_next_frame");
                    (true,
                     SequenceExecutorPacket {
                         msg_type: state.raw_data.msg_type,
                         length: state.raw_data.packet.length as CopyOrMatchLength,
                         content: state.raw_data.packet.data as CopyOrMatchContent,
                         last: state.raw_data.packet.last && state.raw_data.packet.last_block,
                     },
                     DecoderMuxState {
                         raw_data_valid: false,
                         raw_data_valid_next_frame: if (state.raw_data.packet.last_block) {false} else {state.raw_data_valid_next_frame},
                         rle_data_valid: if (state.raw_data.packet.last_block) {state.rle_data_valid_next_frame} else {state.rle_data_valid},
                         rle_data_valid_next_frame: if (state.raw_data.packet.last_block) {false} else {state.rle_data_valid_next_frame},
                         compressed_data_valid: if (state.raw_data.packet.last_block) {state.compressed_data_valid_next_frame} else {state.compressed_data_valid},
                         compressed_data_valid_next_frame: if (state.raw_data.packet.last_block) {false} else {state.compressed_data_valid_next_frame},
                         prev_valid : true,
                         prev_id: if (state.raw_data.packet.last_block && state.raw_data.packet.last) {u32:0xffffffff} else {state.raw_data.packet.id},
                         prev_last: state.raw_data.packet.last,
                         prev_last_block: state.raw_data.packet.last_block,
                         ..state})
                } else if (state.rle_data_valid &&
                        (((state.rle_data.packet.id == (state.prev_id + u32:1)) && state.prev_last) ||
                         ((state.rle_data.packet.id == state.prev_id) && !state.prev_last))) {
                    assert!(!state.rle_data_valid_next_frame, "rle_packet_valid_in_current_and_next_frame");
                    (true,
                     SequenceExecutorPacket {
                         msg_type: state.rle_data.msg_type,
                         length: state.rle_data.packet.length as CopyOrMatchLength,
                         content: state.rle_data.packet.data as CopyOrMatchContent,
                         last: state.rle_data.packet.last && state.rle_data.packet.last_block,
                     },
                     DecoderMuxState {
                         raw_data_valid: if (state.rle_data.packet.last_block) {state.raw_data_valid_next_frame} else {state.raw_data_valid},
                         raw_data_valid_next_frame: if (state.rle_data.packet.last_block) {false} else {state.raw_data_valid_next_frame},
                         rle_data_valid: false,
                         rle_data_valid_next_frame: if (state.rle_data.packet.last_block) {false} else {state.rle_data_valid_next_frame},
                         compressed_data_valid: if (state.rle_data.packet.last_block) {state.compressed_data_valid_next_frame} else {state.compressed_data_valid},
                         compressed_data_valid_next_frame: if (state.rle_data.packet.last_block) {false} else {state.compressed_data_valid_next_frame},
                         prev_valid : true,
                         prev_id: if (state.rle_data.packet.last_block && state.rle_data.packet.last) {u32:0xffffffff} else {state.rle_data.packet.id},
                         prev_last: state.rle_data.packet.last,
                         prev_last_block: state.rle_data.packet.last_block,
                         ..state})
                } else if (state.compressed_data_valid &&
                        (((state.compressed_data.packet.id == (state.prev_id + u32:1)) && state.prev_last) ||
                         ((state.compressed_data.packet.id == state.prev_id) && !state.prev_last))) {
                    assert!(!state.compressed_data_valid_next_frame, "compressed_packet_valid_in_current_and_next_frame");
                    (true,
                     SequenceExecutorPacket {
                         msg_type: state.compressed_data.msg_type,
                         length: state.compressed_data.packet.length as CopyOrMatchLength,
                         content: state.compressed_data.packet.data as CopyOrMatchContent,
                         last: state.compressed_data.packet.last && state.compressed_data.packet.last_block,
                     },
                     DecoderMuxState {
                         raw_data_valid: if (state.compressed_data.packet.last_block) {state.raw_data_valid_next_frame} else {state.raw_data_valid},
                         raw_data_valid_next_frame: if (state.compressed_data.packet.last_block) {false} else {state.raw_data_valid_next_frame},
                         rle_data_valid: if (state.compressed_data.packet.last_block) {state.rle_data_valid_next_frame} else {state.rle_data_valid},
                         rle_data_valid_next_frame: if (state.compressed_data.packet.last_block) {false} else {state.rle_data_valid_next_frame},
                         compressed_data_valid: false,
                         compressed_data_valid_next_frame: if (state.compressed_data.packet.last_block) {false} else {state.compressed_data_valid_next_frame},
                         prev_valid : true,
                         prev_id: if (state.compressed_data.packet.last_block && state.compressed_data.packet.last) {u32:0xffffffff} else {state.compressed_data.packet.id},
                         prev_last: state.compressed_data.packet.last,
                         prev_last_block: state.compressed_data.packet.last_block,
                         ..state})
                } else {
                    (false, zero!<SequenceExecutorPacket>(), state)
                };

            let tok = send_if(tok, output_s, do_send, data_to_send);
            if (do_send) {
                trace_fmt!("DecoderMux: sent {:#x}", data_to_send);
            } else {()};
            state
        } else {
            state
        };

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
    let (raw_s, raw_r) = chan<ExtendedBlockDataPacket>("raw");
    let (rle_s, rle_r) = chan<ExtendedBlockDataPacket>("rle");
    let (cmp_s, cmp_r) = chan<ExtendedBlockDataPacket>("cmp");
    let (output_s, output_r) = chan<SequenceExecutorPacket>("output");

    spawn DecoderMux(raw_r, rle_r, cmp_r, output_s);
    (terminator, raw_s, rle_s, cmp_s, output_r)
  }

  next(state: ()) {
    let tok = join();
    let tok = send(tok, raw_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:0, last: bool: false, last_block: bool: false, data: BlockData:0x11111111, length: BlockPacketLength:4 }});
    let tok = send(tok, raw_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:0, last: bool: false, last_block: bool: false, data: BlockData:0x22222222, length: BlockPacketLength:4 }});
    let tok = send(tok, rle_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:1, last: bool: false, last_block: bool: false, data: BlockData:0xAAAAAAAA, length: BlockPacketLength:4 }});
    let tok = send(tok, raw_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:0, last: bool: true,  last_block: bool: false, data: BlockData:0x33333333, length: BlockPacketLength:4 }});
    let tok = send(tok, cmp_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:2, last: bool: true,  last_block: bool: true,  data: BlockData:0x00000000, length: BlockPacketLength:4 }});
    let tok = send(tok, rle_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:1, last: bool: true,  last_block: bool: false, data: BlockData:0xBBBBBBBB, length: BlockPacketLength:4 }});

    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0x11111111, length: CopyOrMatchLength:4 });
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0x22222222, length: CopyOrMatchLength:4 });
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0x33333333, length: CopyOrMatchLength:4 });
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0xAAAAAAAA, length: CopyOrMatchLength:4 });
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0xBBBBBBBB, length: CopyOrMatchLength:4 });
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: true,  msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0x00000000, length: CopyOrMatchLength:4 });

    send(tok, terminator, true);
  }
}

#[test_proc]
proc DecoderMuxEmptyRawBlocksTest {
  terminator: chan<bool> out;
  raw_s: chan<ExtendedBlockDataPacket> out;
  rle_s: chan<ExtendedBlockDataPacket> out;
  cmp_s: chan<ExtendedBlockDataPacket> out;
  output_r: chan<SequenceExecutorPacket> in;

  init {}

  config (terminator: chan<bool> out) {
    let (raw_s, raw_r) = chan<ExtendedBlockDataPacket>("raw");
    let (rle_s, rle_r) = chan<ExtendedBlockDataPacket>("rle");
    let (cmp_s, cmp_r) = chan<ExtendedBlockDataPacket>("cmp");
    let (output_s, output_r) = chan<SequenceExecutorPacket>("output");

    spawn DecoderMux(raw_r, rle_r, cmp_r, output_s);
    (terminator, raw_s, rle_s, cmp_s, output_r)
  }

  next(state: ()) {
    let tok = join();
    let tok = send(tok, raw_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:0, last: bool: true,  last_block: bool: false, data: BlockData:0x11111111, length: BlockPacketLength:4 }});
    let tok = send(tok, raw_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:1, last: bool: true,  last_block: bool: false, data: BlockData:0x22222222, length: BlockPacketLength:4 }});
    let tok = send(tok, raw_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:2, last: bool: true,  last_block: bool: false, data: BlockData:0x0,        length: BlockPacketLength:0  }});
    let tok = send(tok, raw_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:3, last: bool: true,  last_block: bool: false, data: BlockData:0x33333333, length: BlockPacketLength:4 }});
    let tok = send(tok, raw_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:4, last: bool: true,  last_block: bool: true,  data: BlockData:0x0,        length: BlockPacketLength:0  }});

    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0x11111111, length: CopyOrMatchLength:4 });
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0x22222222, length: CopyOrMatchLength:4 });
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0x0,        length: CopyOrMatchLength:0  });
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0x33333333, length: CopyOrMatchLength:4 });
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: true,  msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0x0       , length: CopyOrMatchLength:0  });

    send(tok, terminator, true);
  }
}

#[test_proc]
proc DecoderMuxEmptyRleBlocksTest {
  terminator: chan<bool> out;
  raw_s: chan<ExtendedBlockDataPacket> out;
  rle_s: chan<ExtendedBlockDataPacket> out;
  cmp_s: chan<ExtendedBlockDataPacket> out;
  output_r: chan<SequenceExecutorPacket> in;

  init {}

  config (terminator: chan<bool> out) {
    let (raw_s, raw_r) = chan<ExtendedBlockDataPacket>("raw");
    let (rle_s, rle_r) = chan<ExtendedBlockDataPacket>("rle");
    let (cmp_s, cmp_r) = chan<ExtendedBlockDataPacket>("cmp");
    let (output_s, output_r) = chan<SequenceExecutorPacket>("output");

    spawn DecoderMux(raw_r, rle_r, cmp_r, output_s);
    (terminator, raw_s, rle_s, cmp_s, output_r)
  }

  next(state: ()) {
    let tok = join();
    let tok = send(tok, raw_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:0, last: bool: true,  last_block: bool: false, data: BlockData:0x11111111, length: BlockPacketLength:4 }});
    let tok = send(tok, raw_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:1, last: bool: true,  last_block: bool: false, data: BlockData:0x22222222, length: BlockPacketLength:4 }});
    let tok = send(tok, raw_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:2, last: bool: true,  last_block: bool: false, data: BlockData:0x0,        length: BlockPacketLength:0  }});
    let tok = send(tok, raw_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:3, last: bool: true,  last_block: bool: false, data: BlockData:0x33333333, length: BlockPacketLength:4 }});
    let tok = send(tok, rle_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:4, last: bool: true,  last_block: bool: true,  data: BlockData:0x0,        length: BlockPacketLength:0  }});

    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0x11111111, length: CopyOrMatchLength:4 });
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0x22222222, length: CopyOrMatchLength:4 });
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0x0,        length: CopyOrMatchLength:0  });
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0x33333333, length: CopyOrMatchLength:4 });
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: true,  msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0x0,        length: CopyOrMatchLength:0  });

    send(tok, terminator, true);
  }
}

#[test_proc]
proc DecoderMuxEmptyBlockBetweenRegularBlocksOnTheSameInputChannelTest {
  terminator: chan<bool> out;
  raw_s: chan<ExtendedBlockDataPacket> out;
  rle_s: chan<ExtendedBlockDataPacket> out;
  cmp_s: chan<ExtendedBlockDataPacket> out;
  output_r: chan<SequenceExecutorPacket> in;

  init {}

  config (terminator: chan<bool> out) {
    let (raw_s, raw_r) = chan<ExtendedBlockDataPacket>("raw");
    let (rle_s, rle_r) = chan<ExtendedBlockDataPacket>("rle");
    let (cmp_s, cmp_r) = chan<ExtendedBlockDataPacket>("cmp");
    let (output_s, output_r) = chan<SequenceExecutorPacket>("output");

    spawn DecoderMux(raw_r, rle_r, cmp_r, output_s);
    (terminator, raw_s, rle_s, cmp_s, output_r)
  }

  next(state: ()) {
    let tok = join();
    let tok = send(tok, raw_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:0, last: bool: false, last_block: bool: false, data: BlockData:0x11111111, length: BlockPacketLength:4 }});
    let tok = send(tok, raw_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:0, last: bool: false, last_block: bool: false, data: BlockData:0x22222222, length: BlockPacketLength:4 }});
    let tok = send(tok, raw_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:0, last: bool: true,  last_block: bool: false, data: BlockData:0x33333333, length: BlockPacketLength:4 }});
    let tok = send(tok, raw_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:1, last: bool: true,  last_block: bool: false, data: BlockData:0x0,        length: BlockPacketLength:0  }});
    let tok = send(tok, raw_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:2, last: bool: false, last_block: bool: false, data: BlockData:0xAAAAAAAA, length: BlockPacketLength:4 }});
    let tok = send(tok, raw_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:2, last: bool: true,  last_block: bool: false, data: BlockData:0xBBBBBBBB, length: BlockPacketLength:4 }});
    let tok = send(tok, raw_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:3, last: bool: true,  last_block: bool: true,  data: BlockData:0x00000000, length: BlockPacketLength:4 }});

    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0x11111111, length: CopyOrMatchLength:4 });
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0x22222222, length: CopyOrMatchLength:4 });
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0x33333333, length: CopyOrMatchLength:4 });
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0x0,        length: CopyOrMatchLength:0  });
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0xAAAAAAAA, length: CopyOrMatchLength:4 });
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0xBBBBBBBB, length: CopyOrMatchLength:4 });
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: true,  msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0x00000000, length: CopyOrMatchLength:4 });

    send(tok, terminator, true);
  }
}

#[test_proc]
proc DecoderMuxEmptyBlockBetweenRegularBlocksOnDifferentInputChannelsTest {
  terminator: chan<bool> out;
  raw_s: chan<ExtendedBlockDataPacket> out;
  rle_s: chan<ExtendedBlockDataPacket> out;
  cmp_s: chan<ExtendedBlockDataPacket> out;
  output_r: chan<SequenceExecutorPacket> in;

  init {}

  config (terminator: chan<bool> out) {
    let (raw_s, raw_r) = chan<ExtendedBlockDataPacket>("raw");
    let (rle_s, rle_r) = chan<ExtendedBlockDataPacket>("rle");
    let (cmp_s, cmp_r) = chan<ExtendedBlockDataPacket>("cmp");
    let (output_s, output_r) = chan<SequenceExecutorPacket>("output");

    spawn DecoderMux(raw_r, rle_r, cmp_r, output_s);
    (terminator, raw_s, rle_s, cmp_s, output_r)
  }

  next(state: ()) {
    let tok = join();
    let tok = send(tok, raw_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:0, last: bool: false, last_block: bool: false, data: BlockData:0x11111111, length: BlockPacketLength:4 }});
    let tok = send(tok, raw_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:0, last: bool: false, last_block: bool: false, data: BlockData:0x22222222, length: BlockPacketLength:4 }});
    let tok = send(tok, rle_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:1, last: bool: true,  last_block: bool: false, data: BlockData:0x0,        length: BlockPacketLength:0  }});
    let tok = send(tok, rle_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:2, last: bool: false, last_block: bool: false, data: BlockData:0xAAAAAAAA, length: BlockPacketLength:4 }});
    let tok = send(tok, raw_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:0, last: bool: true,  last_block: bool: false, data: BlockData:0x33333333, length: BlockPacketLength:4 }});
    let tok = send(tok, cmp_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:3, last: bool: true,  last_block: bool: true,  data: BlockData:0x00000000, length: BlockPacketLength:4 }});
    let tok = send(tok, rle_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:2, last: bool: true,  last_block: bool: false, data: BlockData:0xBBBBBBBB, length: BlockPacketLength:4 }});

    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0x11111111, length: CopyOrMatchLength:4 });
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0x22222222, length: CopyOrMatchLength:4 });
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0x33333333, length: CopyOrMatchLength:4 });
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0x0,        length: CopyOrMatchLength:0  });
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0xAAAAAAAA, length: CopyOrMatchLength:4 });
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0xBBBBBBBB, length: CopyOrMatchLength:4 });
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: true,  msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0x00000000, length: CopyOrMatchLength:4 });

    send(tok, terminator, true);
  }
}

#[test_proc]
proc DecoderMuxMultipleFramesTest {
  terminator: chan<bool> out;
  raw_s: chan<ExtendedBlockDataPacket> out;
  rle_s: chan<ExtendedBlockDataPacket> out;
  cmp_s: chan<ExtendedBlockDataPacket> out;
  output_r: chan<SequenceExecutorPacket> in;

  init {}

  config (terminator: chan<bool> out) {
    let (raw_s, raw_r) = chan<ExtendedBlockDataPacket>("raw");
    let (rle_s, rle_r) = chan<ExtendedBlockDataPacket>("rle");
    let (cmp_s, cmp_r) = chan<ExtendedBlockDataPacket>("cmp");
    let (output_s, output_r) = chan<SequenceExecutorPacket>("output");

    spawn DecoderMux(raw_r, rle_r, cmp_r, output_s);
    (terminator, raw_s, rle_s, cmp_s, output_r)
  }

  next(state: ()) {
    let tok = join();
    // Frame #1
    let tok = send(tok, raw_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:0, last: bool: false, last_block: bool: false, data: BlockData:0x11111111, length: BlockPacketLength:4 }});
    let tok = send(tok, raw_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:0, last: bool: false, last_block: bool: false, data: BlockData:0x22222222, length: BlockPacketLength:4 }});
    let tok = send(tok, raw_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:0, last: bool: true,  last_block: bool: false, data: BlockData:0x33333333, length: BlockPacketLength:4 }});
    let tok = send(tok, rle_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:1, last: bool: false, last_block: bool: false, data: BlockData:0xAAAAAAAA, length: BlockPacketLength:4 }});
    let tok = send(tok, rle_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:1, last: bool: true,  last_block: bool: false, data: BlockData:0xBBBBBBBB, length: BlockPacketLength:4 }});
    let tok = send(tok, cmp_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:2, last: bool: true,  last_block: bool: false, data: BlockData:0xCCCCCCCC, length: BlockPacketLength:4 }});
    let tok = send(tok, raw_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:3, last: bool: true,  last_block: bool: false, data: BlockData:0xDDDDDDDD, length: BlockPacketLength:4 }});
    let tok = send(tok, rle_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:4, last: bool: true,  last_block: bool: false, data: BlockData:0xEEEEEEEE, length: BlockPacketLength:4 }});
    let tok = send(tok, cmp_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:5, last: bool: true,  last_block: bool: true,  data: BlockData:0xFFFFFFFF, length: BlockPacketLength:4 }});
    // Frame #2
    let tok = send(tok, raw_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:0, last: bool: true,  last_block: bool: false, data: BlockData:0x44444444, length: BlockPacketLength:4 }});
    let tok = send(tok, cmp_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:1, last: bool: true,  last_block: bool: false, data: BlockData:0x0,        length: BlockPacketLength:0  }});
    let tok = send(tok, cmp_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:2, last: bool: true,  last_block: bool: false, data: BlockData:0x0,        length: BlockPacketLength:0  }});
    let tok = send(tok, rle_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:3, last: bool: true,  last_block: bool: true,  data: BlockData:0x0,        length: BlockPacketLength:0  }});
    // Frame #3
    let tok = send(tok, raw_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:0, last: bool: true,  last_block: bool: false, data: BlockData:0x55555555, length: BlockPacketLength:4 }});
    let tok = send(tok, raw_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:1, last: bool: true,  last_block: bool: false, data: BlockData:0x0,        length: BlockPacketLength:0  }});
    let tok = send(tok, raw_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:2, last: bool: true,  last_block: bool: false, data: BlockData:0x0,        length: BlockPacketLength:0  }});
    let tok = send(tok, raw_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:3, last: bool: true,  last_block: bool: false, data: BlockData:0x0,        length: BlockPacketLength:0  }});
    let tok = send(tok, raw_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:4, last: bool: true,  last_block: bool: false, data: BlockData:0x0,        length: BlockPacketLength:0  }});
    let tok = send(tok, raw_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:5, last: bool: true,  last_block: bool: false, data: BlockData:0x0,        length: BlockPacketLength:0  }});
    let tok = send(tok, raw_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:6, last: bool: true,  last_block: bool: false, data: BlockData:0x0,        length: BlockPacketLength:0  }});
    let tok = send(tok, raw_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:7, last: bool: true,  last_block: bool: false, data: BlockData:0x0,        length: BlockPacketLength:0  }});
    let tok = send(tok, raw_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:8, last: bool: true,  last_block: bool: false, data: BlockData:0x0,        length: BlockPacketLength:0  }});
    let tok = send(tok, raw_s, ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { id: u32:9, last: bool: true,  last_block: bool: true,  data: BlockData:0x0,        length: BlockPacketLength:0  }});

    // Frame #1
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0x11111111, length: CopyOrMatchLength:4 });
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0x22222222, length: CopyOrMatchLength:4 });
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0x33333333, length: CopyOrMatchLength:4 });
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0xAAAAAAAA, length: CopyOrMatchLength:4 });
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0xBBBBBBBB, length: CopyOrMatchLength:4 });
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0xCCCCCCCC, length: CopyOrMatchLength:4 });
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0xDDDDDDDD, length: CopyOrMatchLength:4 });
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0xEEEEEEEE, length: CopyOrMatchLength:4 });
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: true,  msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0xFFFFFFFF, length: CopyOrMatchLength:4 });
    // Frame #2
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0x44444444, length: CopyOrMatchLength:4 });
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0x0       , length: CopyOrMatchLength:0  });
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0x0       , length: CopyOrMatchLength:0  });
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: true,  msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0x0       , length: CopyOrMatchLength:0  });
    // Frame #3
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0x55555555, length: CopyOrMatchLength:4 });
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0x0       , length: CopyOrMatchLength:0  });
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0x0       , length: CopyOrMatchLength:0  });
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0x0       , length: CopyOrMatchLength:0  });
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0x0       , length: CopyOrMatchLength:0  });
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0x0       , length: CopyOrMatchLength:0  });
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0x0       , length: CopyOrMatchLength:0  });
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0x0       , length: CopyOrMatchLength:0  });
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0x0       , length: CopyOrMatchLength:0  });
    let (tok, data) = recv(tok, output_r); assert_eq(data, SequenceExecutorPacket {last: bool: true,  msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0x0       , length: CopyOrMatchLength:0  });

    send(tok, terminator, true);
  }
}
