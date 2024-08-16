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

import xls.modules.zstd.common;
import xls.modules.zstd.dec_demux as demux;
import xls.modules.zstd.raw_block_dec as raw;
import xls.modules.zstd.rle_block_dec as rle;
import xls.modules.zstd.dec_mux as mux;

type BlockDataPacket = common::BlockDataPacket;
type BlockData = common::BlockData;
type BlockPacketLength = common::BlockPacketLength;
type ExtendedBlockDataPacket = common::ExtendedBlockDataPacket;
type CopyOrMatchContent = common::CopyOrMatchContent;
type CopyOrMatchLength = common::CopyOrMatchLength;
type SequenceExecutorPacket = common::SequenceExecutorPacket<common::SYMBOL_WIDTH>;
type SequenceExecutorMessageType = common::SequenceExecutorMessageType;

// Proc responsible for connecting internal procs used in Block data decoding.
// It handles incoming block data packets by redirecting those to demuxer which passes those to
// block decoder procs specific for given block type. Results are then gathered by mux which
// transfers decoded data further. The connections are visualised on the following diagram:
//
//                  Block Decoder
//   ┌───────────────────────────────────────┐
//   │           Raw Block Decoder           │
//   │         ┌───────────────────┐         │
//   │       ┌─►                   ├┐        │
//   │ Demux │ └───────────────────┘│   Mux  │
//   │┌─────┐│   Rle Block Decoder  │ ┌─────┐│
//   ││     ├┘ ┌───────────────────┐└─►     ││
// ──┼►     ├──►                   ├──►     ├┼─►
//   ││     ├┐ └───────────────────┘┌─►     ││
//   │└─────┘│   Cmp Block Decoder  │ └─────┘│
//   │       │ ┌───────────────────┐│        │
//   │       └─►                   ├┘        │
//   │         └───────────────────┘         │
//   └───────────────────────────────────────┘

proc BlockDecoder {
    input_r: chan<BlockDataPacket> in;
    output_s: chan<SequenceExecutorPacket> out;

    config (input_r: chan<BlockDataPacket> in, output_s: chan<SequenceExecutorPacket> out) {
        let (demux_raw_s, demux_raw_r) = chan<BlockDataPacket, u32:1>("demux_raw");
        let (demux_rle_s, demux_rle_r) = chan<BlockDataPacket, u32:1>("demux_rle");
        let (demux_cmp_s, demux_cmp_r) = chan<BlockDataPacket, u32:1>("demux_cmp");
        let (mux_raw_s, mux_raw_r) = chan<ExtendedBlockDataPacket, u32:1>("mux_raw");
        let (mux_rle_s, mux_rle_r) = chan<ExtendedBlockDataPacket, u32:1>("mux_rle");
        let (mux_cmp_s, mux_cmp_r) = chan<ExtendedBlockDataPacket, u32:1>("mux_cmp");

        spawn demux::DecoderDemux(input_r, demux_raw_s, demux_rle_s, demux_cmp_s);
        spawn raw::RawBlockDecoder(demux_raw_r, mux_raw_s);
        spawn rle::RleBlockDecoder(demux_rle_r, mux_rle_s);
        // TODO(lpawelcz): 2023-11-28 change to compressed block decoder proc
        spawn raw::RawBlockDecoder(demux_cmp_r, mux_cmp_s);
        spawn mux::DecoderMux(mux_raw_r, mux_rle_r, mux_cmp_r, output_s);

        (input_r, output_s)
    }

    init {  }

    next(state: ()) {  }
}

#[test_proc]
proc BlockDecoderTest {
    terminator: chan<bool> out;
    input_s: chan<BlockDataPacket> out;
    output_r: chan<SequenceExecutorPacket> in;

    init {}

    config (terminator: chan<bool> out) {
        let (input_s, input_r) = chan<BlockDataPacket>("input");
        let (output_s, output_r) = chan<SequenceExecutorPacket>("output");

        spawn BlockDecoder(input_r, output_s);

        (terminator, input_s, output_r)
    }

    next(state: ()) {
        let tok = join();
        let EncodedDataBlocksPackets: BlockDataPacket[13] = [
            // RAW Block 1 byte
            BlockDataPacket { id: u32:0, last: true, last_block: false, data: BlockData:0xDE000008, length: BlockPacketLength:32 },
            // RAW Block 2 bytes
            BlockDataPacket { id: u32:1, last: true, last_block: false, data: BlockData:0xDEAD000010, length: BlockPacketLength:40 },
            // RAW Block 4 bytes
            BlockDataPacket { id: u32:2, last: true, last_block: false, data: BlockData:0xDEADBEEF000020, length: BlockPacketLength:56 },
            // RAW Block 5 bytes (block header takes one full packet)
            BlockDataPacket { id: u32:3, last: true, last_block: false, data: BlockData:0xDEADBEEFEF000028, length: BlockPacketLength:64 },
            // RAW Block 24 bytes (multi-packet block header with unaligned data in the last packet)
            BlockDataPacket { id: u32:4, last: false, last_block: false, data: BlockData:0x12345678900000C0, length: BlockPacketLength:64 },
            BlockDataPacket { id: u32:4, last: false, last_block: false, data: BlockData:0x1234567890ABCDEF, length: BlockPacketLength:64 },
            BlockDataPacket { id: u32:4, last: false, last_block: false, data: BlockData:0xFEDCBA0987654321, length: BlockPacketLength:64 },
            BlockDataPacket { id: u32:4, last: true, last_block: false, data: BlockData:0xF0F0F0, length: BlockPacketLength:24 },

            // RLE Block 1 byte
            BlockDataPacket { id: u32:5, last: true, last_block: false, data: BlockData:0x6700000a, length: BlockPacketLength:32 },
            // RLE Block 2 bytes
            BlockDataPacket { id: u32:6, last: true, last_block: false, data: BlockData:0x45000012, length: BlockPacketLength:32 },
            // RLE Block 4 bytes
            BlockDataPacket { id: u32:7, last: true, last_block: false, data: BlockData:0x23000022, length: BlockPacketLength:32 },
            // RLE Block 8 bytes (block takes one full packet)
            BlockDataPacket { id: u32:8, last: true, last_block: false, data: BlockData:0x10000042, length: BlockPacketLength:32 },
            // RLE Block 26 bytes (multi-packet block header with unaligned data in the last packet)
            BlockDataPacket { id: u32:9, last: true, last_block: true, data: BlockData:0xDE0000d2, length: BlockPacketLength:32 },
        ];

        let tok = for ((counter, block_packet), tok): ((u32, BlockDataPacket), token) in enumerate(EncodedDataBlocksPackets) {
            let tok = send(tok, input_s, block_packet);
            trace_fmt!("Sent #{} encoded block packet, {:#x}", counter + u32:1, block_packet);
            (tok)
        }(tok);

        let DecodedDataBlocksPackets: SequenceExecutorPacket[16] = [
            // RAW Block 1 byte
            SequenceExecutorPacket { last: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0xDE, length: CopyOrMatchLength:8 },
            // RAW Block 2 bytes
            SequenceExecutorPacket { last: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0xDEAD, length: CopyOrMatchLength:16 },
            // RAW Block 4 bytes
            SequenceExecutorPacket { last: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0xDEADBEEF, length: CopyOrMatchLength:32 },
            // RAW Block 5 bytes (block header takes one full packet)
            SequenceExecutorPacket { last: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0xDEADBEEFEF, length: CopyOrMatchLength:40 },
            // RAW Block 24 bytes (multi-packet block header with unaligned data in the last packet)
            SequenceExecutorPacket { last: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0x1234567890, length: CopyOrMatchLength:40 },
            SequenceExecutorPacket { last: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0x1234567890ABCDEF, length: CopyOrMatchLength:64 },
            SequenceExecutorPacket { last: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0xFEDCBA0987654321, length: CopyOrMatchLength:64 },
            SequenceExecutorPacket { last: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0xF0F0F0, length: CopyOrMatchLength:24 },

            // RLE Block 1 byte
            SequenceExecutorPacket { last: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0x67, length: CopyOrMatchLength:8 },
            // RLE Block 2 bytes
            SequenceExecutorPacket { last: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0x4545, length: CopyOrMatchLength:16 },
            // RLE Block 4 bytes
            SequenceExecutorPacket { last: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0x23232323, length: CopyOrMatchLength:32 },
            // RLE Block 8 bytes (block takes one full packet)
            SequenceExecutorPacket { last: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0x1010101010101010, length: CopyOrMatchLength:64 },
            // RLE Block 26 bytes (multi-packet block header with unaligned data in the last packet)
            SequenceExecutorPacket { last: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0xDEDEDEDEDEDEDEDE, length: CopyOrMatchLength:64 },
            SequenceExecutorPacket { last: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0xDEDEDEDEDEDEDEDE, length: CopyOrMatchLength:64 },
            SequenceExecutorPacket { last: false, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0xDEDEDEDEDEDEDEDE, length: CopyOrMatchLength:64 },
            SequenceExecutorPacket { last: true, msg_type: SequenceExecutorMessageType::LITERAL, content: CopyOrMatchContent:0xDEDE, length: CopyOrMatchLength:16 },
        ];

        let tok = for ((counter, expected_block_packet), tok): ((u32, SequenceExecutorPacket), token) in enumerate(DecodedDataBlocksPackets) {
            let (tok, decoded_block_packet) = recv(tok, output_r);
            trace_fmt!("Received #{} decoded block packet, data: 0x{:x}", counter + u32:1, decoded_block_packet);
            trace_fmt!("Expected #{} decoded block packet, data: 0x{:x}", counter + u32:1, expected_block_packet);
            assert_eq(decoded_block_packet, expected_block_packet);
            (tok)
        }(tok);

        send(tok, terminator, true);
    }
}
