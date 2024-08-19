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

// Repacketizer
//
// Remove invalid bytes from input packets,
// form new packets with all bits valid if possible.

import std;
import xls.modules.zstd.common as common;

type ZstdDecodedPacket = common::ZstdDecodedPacket;
type BlockData = common::BlockData;
type BlockPacketLength = common::BlockPacketLength;

const DATA_WIDTH = common::DATA_WIDTH;

struct RepacketizerState {
    repacked_data: BlockData,
    valid_length: BlockPacketLength,
    to_fill: BlockPacketLength,
    send_last_leftover: bool
}

const ZERO_ZSTD_DECODED_PACKET = zero!<ZstdDecodedPacket>();
const ZERO_REPACKETIZER_STATE = zero!<RepacketizerState>();
const INIT_REPACKETIZER_STATE = RepacketizerState {to_fill: DATA_WIDTH, ..ZERO_REPACKETIZER_STATE};

pub proc Repacketizer {
    input_r: chan<ZstdDecodedPacket> in;
    output_s: chan<ZstdDecodedPacket> out;

    init {(INIT_REPACKETIZER_STATE)}

    config (
        input_r: chan<ZstdDecodedPacket> in,
        output_s: chan<ZstdDecodedPacket> out,
    ) {
        (input_r, output_s)
    }

    next (state: RepacketizerState) {
        let tok = join();
        // Don't receive if we process leftovers
        let (tok, decoded_packet) = recv_if(tok, input_r, !state.send_last_leftover, ZERO_ZSTD_DECODED_PACKET);

        // Will be able to send repacketized packet in current next() evaluation
        let send_now = state.to_fill <= decoded_packet.length || decoded_packet.last || state.send_last_leftover;
        // Received last packet in frame which won't fit into currently processed repacketized packet.
        // Set flag indicating that Repacketizer will send another packet to finish the frame in
        // next evaluation.
        let next_send_last_leftover = decoded_packet.last && state.to_fill < decoded_packet.length;

        let combined_length = state.valid_length + decoded_packet.length;
        let leftover_length = (combined_length - DATA_WIDTH) as s32;
        let next_valid_length = if leftover_length >= s32:0 {leftover_length as BlockPacketLength} else {combined_length};
        let next_to_fill = DATA_WIDTH - next_valid_length;

        let current_valid_length = if leftover_length >= s32:0 {DATA_WIDTH} else {combined_length};
        let bits_to_take_length = if leftover_length >= s32:0 {state.to_fill} else {decoded_packet.length};

        // Append lest signifiant bits of received packet to most significant positions of repacked data buffer
        let masked_data = ((BlockData:1 << bits_to_take_length) - BlockData:1) & decoded_packet.data;
        let repacked_data = state.repacked_data | (masked_data << state.valid_length);

        // Prepare buffer state for the next evaluation - take leftover most significant bits of
        // received packet
        let leftover_mask = (BlockData:1 << (decoded_packet.length - bits_to_take_length)) - BlockData:1;
        let leftover_masked_data = (decoded_packet.data >> bits_to_take_length) & leftover_mask;
        let next_repacked_data = if (send_now) {leftover_masked_data} else {repacked_data};

        let packet_to_send = ZstdDecodedPacket {
            data: repacked_data,
            length: current_valid_length,
            last: state.send_last_leftover || (decoded_packet.last && !next_send_last_leftover),
        };
        let tok = send_if(tok, output_s, send_now, packet_to_send);

        let next_state = if (state.send_last_leftover || (decoded_packet.last && !next_send_last_leftover)) {
            INIT_REPACKETIZER_STATE
        } else {
            RepacketizerState {
                repacked_data: next_repacked_data,
                valid_length: next_valid_length,
                to_fill: next_to_fill,
                send_last_leftover: next_send_last_leftover,
            }
        };

        trace_fmt!("Repacketizer: state: {:#x}", state);
        if (!state.send_last_leftover) {
            trace_fmt!("Repacketizer: Received packet: {:#x}", decoded_packet);
        } else {};
        trace_fmt!("Repacketizer: send_now: {}", send_now);
        trace_fmt!("Repacketizer: next_send_last_leftover: {}", next_send_last_leftover);
        trace_fmt!("Repacketizer: combined_length: {}", combined_length);
        trace_fmt!("Repacketizer: leftover_length: {}", leftover_length);
        trace_fmt!("Repacketizer: next_valid_length: {}", next_valid_length);
        trace_fmt!("Repacketizer: next_to_fill: {}", next_to_fill);
        trace_fmt!("Repacketizer: current_valid_length: {}", current_valid_length);
        trace_fmt!("Repacketizer: bits_to_take_length: {}", bits_to_take_length);
        trace_fmt!("Repacketizer: masked_data: {:#x}", masked_data);
        trace_fmt!("Repacketizer: repacked_data: {:#x}", repacked_data);
        trace_fmt!("Repacketizer: leftover_mask: {:#x}", leftover_mask);
        trace_fmt!("Repacketizer: leftover_masked_data: {:#x}", leftover_masked_data);
        trace_fmt!("Repacketizer: next_repacked_data: {:#x}", next_repacked_data);
        if (send_now) {
            trace_fmt!("Repacketizer: Sent repacketized packet: {:#x}", packet_to_send);
        } else {};
        trace_fmt!("Repacketizer: next_state: {:#x}", next_state);

        next_state
    }
}

#[test_proc]
proc RepacketizerTest {
    terminator: chan<bool> out;
    input_s: chan<ZstdDecodedPacket> out;
    output_r: chan<ZstdDecodedPacket> in;

    init {}

    config (terminator: chan<bool> out) {
        let (input_s, input_r) = chan<ZstdDecodedPacket>("input");
        let (output_s, output_r) = chan<ZstdDecodedPacket>("output");

        spawn Repacketizer(input_r, output_s);
        (terminator, input_s, output_r)
    }

    next(state: ()) {
        let tok = join();
        let DecodedInputs: ZstdDecodedPacket[24] = [
            // Full packet - no need for removing alignment zeros
            ZstdDecodedPacket {data: BlockData:0xDEADBEEF12345678, length: BlockPacketLength:64, last:false},
            // Data in 4 packets - should be batched together into one full output packet
            ZstdDecodedPacket {data: BlockData:0x78, length: BlockPacketLength:8, last:false},
            ZstdDecodedPacket {data: BlockData:0x56, length: BlockPacketLength:8, last:false},
            ZstdDecodedPacket {data: BlockData:0x1234, length: BlockPacketLength:16, last:false},
            ZstdDecodedPacket {data: BlockData:0xDEADBEEF, length: BlockPacketLength:32, last:false},
            // Small last packet - should be send out separatelly
            ZstdDecodedPacket {data: BlockData:0x9A, length: BlockPacketLength:8, last:true},
            // One not-full packet and consecutive last packet packet in frame which completes previous packet and
            // starts new one which should be marked as last
            ZstdDecodedPacket {data: BlockData:0xADBEEF12345678, length: BlockPacketLength:56, last:false},
            ZstdDecodedPacket {data: BlockData:0x9ADE, length: BlockPacketLength:16, last:true},
            // 8 1-byte packets forming single output packet
            ZstdDecodedPacket {data: BlockData:0xEF, length: BlockPacketLength:8, last:false},
            ZstdDecodedPacket {data: BlockData:0xCD, length: BlockPacketLength:8, last:false},
            ZstdDecodedPacket {data: BlockData:0xAB, length: BlockPacketLength:8, last:false},
            ZstdDecodedPacket {data: BlockData:0x89, length: BlockPacketLength:8, last:false},
            ZstdDecodedPacket {data: BlockData:0x67, length: BlockPacketLength:8, last:false},
            ZstdDecodedPacket {data: BlockData:0x45, length: BlockPacketLength:8, last:false},
            ZstdDecodedPacket {data: BlockData:0x23, length: BlockPacketLength:8, last:false},
            ZstdDecodedPacket {data: BlockData:0x01, length: BlockPacketLength:8, last:false},
            // 7 1-byte packets and 1 8-byte packet forming 1 full and 1 7-byte output packet
            // marked as last
            ZstdDecodedPacket {data: BlockData:0xEF, length: BlockPacketLength:8, last:false},
            ZstdDecodedPacket {data: BlockData:0xCD, length: BlockPacketLength:8, last:false},
            ZstdDecodedPacket {data: BlockData:0xAB, length: BlockPacketLength:8, last:false},
            ZstdDecodedPacket {data: BlockData:0x89, length: BlockPacketLength:8, last:false},
            ZstdDecodedPacket {data: BlockData:0x67, length: BlockPacketLength:8, last:false},
            ZstdDecodedPacket {data: BlockData:0x45, length: BlockPacketLength:8, last:false},
            ZstdDecodedPacket {data: BlockData:0x23, length: BlockPacketLength:8, last:false},
            ZstdDecodedPacket {data: BlockData:0xFEDCBA9876543201, length: BlockPacketLength:64, last:true},
        ];

        let DecodedOutputs: ZstdDecodedPacket[8] = [
            // Full packet - no need for removing alignment zeros
            ZstdDecodedPacket {data: BlockData:0xDEADBEEF12345678, length: BlockPacketLength:64, last:false},
            // Data in 4 packets - should be batched together into one full output packet
            ZstdDecodedPacket {data: BlockData:0xDEADBEEF12345678, length: BlockPacketLength:64, last:false},
            // Small last packet - should be send out separatelly
            ZstdDecodedPacket {data: BlockData:0x9A, length: BlockPacketLength:8, last:true},
            // One not-full packet and consecutive last packet packet in frame which completes previous packet and
            // starts new one which should be marked as last
            ZstdDecodedPacket {data: BlockData:0xDEADBEEF12345678, length: BlockPacketLength:64, last:false},
            ZstdDecodedPacket {data: BlockData:0x9A, length: BlockPacketLength:8, last:true},
            // 8 1-byte packets forming single output packet
            ZstdDecodedPacket {data: BlockData:0x0123456789ABCDEF, length: BlockPacketLength:64, last:false},
            // 7 1-byte packets and 1 8-byte packet forming 1 full and 1 7-byte output packet
            // marked as last
            ZstdDecodedPacket {data: BlockData:0x0123456789ABCDEF, length: BlockPacketLength:64, last:false},
            ZstdDecodedPacket {data: BlockData:0xFEDCBA98765432, length: BlockPacketLength:56, last:true},
        ];

        let tok = for ((counter, decoded_input), tok): ((u32, ZstdDecodedPacket), token) in enumerate(DecodedInputs) {
            let tok = send(tok, input_s, decoded_input);
            trace_fmt!("Sent #{} decoded zero-filled packet, {:#x}", counter + u32:1, decoded_input);
            (tok)
        } (tok);

        let tok = for ((counter, expected_output), tok): ((u32, ZstdDecodedPacket), token) in enumerate(DecodedOutputs) {
            let (tok, decoded_output) = recv(tok, output_r);
            trace_fmt!("Received #{} decoded non-zero-filled packet, {:#x}", counter + u32:1, decoded_output);
            trace_fmt!("Expected #{} decoded non-zero-filled packet, {:#x}", counter + u32:1, expected_output);
            assert_eq(decoded_output, expected_output);
            (tok)
        } (tok);

        send(tok, terminator, true);
    }
}
