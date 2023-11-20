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

// This file contains the implementation of RleBlockDecoder responsible for decoding
// ZSTD RLE Blocks. More Information about Rle Block's format can be found in:
// https://datatracker.ietf.org/doc/html/rfc8878#section-3.1.1.2.2
//
// The implementation consist of 3 procs:
// * RleDataPacker
// * RunLengthDecoder
// * BatchPacker
// Connections between those is represented on the diagram below:
//
//                                RleBlockDecoder
//    ┌─────────────────────────────────────────────────────────────┐
//    │    RleDataPacker       RunLengthDecoder       BatchPacker   │
//    │  ┌───────────────┐   ┌──────────────────┐   ┌─────────────┐ │
// ───┼─►│               ├──►│                  ├──►│             ├─┼──►
//    │  └───────┬───────┘   └──────────────────┘   └─────────────┘ │
//    │          │                                         ▲        │
//    │          │            SynchronizationData          │        │
//    │          └─────────────────────────────────────────┘        │
//    └─────────────────────────────────────────────────────────────┘
//
// RleDataPacker is responsible for receiving the incoming packets of block data, converting
// those to format accepted by RunLengthDecoder and passing the data to the actual decoder block.
// It also extracts from the input packets the synchronization data like block_id and last_block
// and then passes those to BatchPacker proc.
// RunLengthDecoder decodes RLE blocks and outputs one symbol for each transaction on output
// channel.
// BatchPacker then gathers those symbols into packets, appends synchronization data received from
// RleDataPacker and passes such packets to the output of the RleBlockDecoder.

import xls.modules.zstd.common;
import xls.modules.rle.rle_dec;
import xls.modules.rle.rle_common;

const SYMBOL_WIDTH = common::SYMBOL_WIDTH;
const BLOCK_SIZE_WIDTH = common::BLOCK_SIZE_WIDTH;
const DATA_WIDTH = common::DATA_WIDTH;
const BATCH_SIZE = DATA_WIDTH / SYMBOL_WIDTH;

type BlockDataPacket = common::BlockDataPacket;
type BlockPacketLength = common::BlockPacketLength;
type BlockData = common::BlockData;
type BlockSize = common::BlockSize;

type ExtendedBlockDataPacket = common::ExtendedBlockDataPacket;
type CopyOrMatchContent = common::CopyOrMatchContent;
type CopyOrMatchLength = common::CopyOrMatchLength;
type SequenceExecutorMessageType = common::SequenceExecutorMessageType;

type RleInput = rle_common::CompressedData<SYMBOL_WIDTH, BLOCK_SIZE_WIDTH>;
type RleOutput = rle_common::PlainData<SYMBOL_WIDTH>;
type Symbol = bits[SYMBOL_WIDTH];
type SymbolCount = BlockSize;

struct BlockSyncData {
    last_block: bool,
    id: u32
}

proc RleDataPacker {
    block_data_r: chan<BlockDataPacket> in;
    rle_data_s: chan<RleInput> out;
    sync_s: chan<BlockSyncData> out;

    config(
        block_data_r: chan<BlockDataPacket> in,
        rle_data_s: chan<RleInput> out,
        sync_s: chan<BlockSyncData> out
    ) {
        (block_data_r, rle_data_s, sync_s)
    }

    init {  }

    next(tok: token, state: ()) {
        let (tok, input) = recv(tok, block_data_r);
        let rle_dec_data = RleInput {
            symbol: input.data as Symbol, count: input.length as SymbolCount, last: true
        };
        let data_tok = send(tok, rle_data_s, rle_dec_data);
        let sync_data = BlockSyncData { last_block: input.last_block, id: input.id };
        let sync_tok = send(data_tok, sync_s, sync_data);
    }
}

type RleTestVector = (Symbol, SymbolCount);

#[test_proc]
proc RleDataPacker_test {
    terminator: chan<bool> out;
    in_s: chan<BlockDataPacket> out;
    out_r: chan<RleInput> in;
    sync_r: chan<BlockSyncData> in;

    config(terminator: chan<bool> out) {
        let (in_s, in_r) = chan<BlockDataPacket>;
        let (out_s, out_r) = chan<RleInput>;
        let (sync_s, sync_r) = chan<BlockSyncData>;

        spawn RleDataPacker(in_r, out_s, sync_s);

        (terminator, in_s, out_r, sync_r)
    }

    init {  }

    next(tok: token, state: ()) {
        let EncodedRleBlocks: RleTestVector[6] = [
            (Symbol:0x1, SymbolCount:0x1),
            (Symbol:0x2, SymbolCount:0x2),
            (Symbol:0x3, SymbolCount:0x4),
            (Symbol:0x4, SymbolCount:0x8),
            (Symbol:0x5, SymbolCount:0x10),
            (Symbol:0x6, SymbolCount:0x1F),
        ];
        let tok = for ((counter, block), tok): ((u32, RleTestVector), token) in enumerate(EncodedRleBlocks) {
            let last_block = (counter == (array_size(EncodedRleBlocks) - u32:1));
            let data_in = BlockDataPacket {
                last: true,
                last_block,
                id: counter,
                data: block.0 as BlockData,
                length: block.1 as BlockPacketLength
            };
            let tok = send(tok, in_s, data_in);
            trace_fmt!("Sent #{} raw encoded block, {:#x}", counter + u32:1, data_in);

            let data_out = RleInput {
                last: true, symbol: block.0 as Symbol, count: block.1 as BlockSize
            };
            let (tok, dec_output) = recv(tok, out_r);
            trace_fmt!("Received #{} packed rle encoded block, {:#x}", counter + u32:1, dec_output);
            assert_eq(dec_output, data_out);

            let sync_out = BlockSyncData {
                id: counter,
                last_block: counter == (array_size(EncodedRleBlocks) - u32:1),
            };
            let (tok, sync_output) = recv(tok, sync_r);
            trace_fmt!("Received #{} synchronization data, {:#x}", counter + u32:1, sync_output);
            assert_eq(sync_output, sync_out);
            (tok)
        }(tok);
        send(tok, terminator, true);
    }
}

struct BatchState {
    batch: BlockData,
    symbols_in_batch: BlockPacketLength,
    prev_id: u32,
    prev_last: bool,
    prev_last_block: bool,
}

const ZERO_BATCH_STATE = zero!<BatchState>();

proc BatchPacker {
    rle_data_r: chan<RleOutput> in;
    sync_r: chan<BlockSyncData> in;
    block_data_s: chan<ExtendedBlockDataPacket> out;

    config(
        rle_data_r: chan<RleOutput> in,
        sync_r: chan<BlockSyncData> in,
        block_data_s: chan<ExtendedBlockDataPacket> out
    ) {
        (rle_data_r, sync_r, block_data_s)
    }

    // Init the state to signal new batch to process
    init { (BatchState { prev_last: true, ..ZERO_BATCH_STATE }) }

    next(tok: token, state: BatchState) {
        let (tok, decoded_data) = recv(tok, rle_data_r);

        let symbols_in_batch = state.symbols_in_batch;
        let shift = symbols_in_batch << u32:3;  // multiply by 8 bits
        let batch = state.batch | ((decoded_data.symbol as BlockData) << shift);
        let symbols_in_batch = symbols_in_batch + BlockPacketLength:1;
        let do_send_batch = (decoded_data.last | (symbols_in_batch >= BATCH_SIZE));

        let block_in_progress_sync =
            BlockSyncData { id: state.prev_id, last_block: state.prev_last_block };
        let (tok, sync_data) =
            recv_if(tok, sync_r, state.prev_last, block_in_progress_sync);

        let decoded_batch_data = ExtendedBlockDataPacket {
            // Decoded RLE block is always a literal
            msg_type: SequenceExecutorMessageType::LITERAL,
            packet: BlockDataPacket {
                last: decoded_data.last,
                last_block: sync_data.last_block,
                id: sync_data.id,
                data: batch as BlockData,
                // length in bits
                length: (symbols_in_batch << 3) as BlockPacketLength,
            }
        };

        let data_tok =
            send_if(tok, block_data_s, do_send_batch, decoded_batch_data);

        let new_symbols_in_batch =
            if do_send_batch { BlockPacketLength:0 } else { symbols_in_batch };
        let new_batch = if do_send_batch { BlockData:0 } else { batch };
        BatchState {
            batch: new_batch,
            symbols_in_batch: new_symbols_in_batch,
            prev_last: decoded_data.last,
            prev_last_block: sync_data.last_block,
            prev_id: sync_data.id
        }
    }
}

type BatchTestVector = (Symbol, bool);

#[test_proc]
proc BatchPacker_test {
    terminator: chan<bool> out;
    in_s: chan<RleOutput> out;
    sync_s: chan<BlockSyncData> out;
    out_r: chan<ExtendedBlockDataPacket> in;

    config(terminator: chan<bool> out) {
        let (in_s, in_r) = chan<RleOutput>;
        let (sync_s, sync_r) = chan<BlockSyncData>;
        let (out_s, out_r) = chan<ExtendedBlockDataPacket>;

        spawn BatchPacker(in_r, sync_r, out_s);

        (terminator, in_s, sync_s, out_r)
    }

    init {  }

    next(tok: token, state: ()) {
        let SyncData: BlockSyncData[6] = [
            BlockSyncData { last_block: false, id: u32:0 },
            BlockSyncData { last_block: false, id: u32:1 },
            BlockSyncData { last_block: false, id: u32:2 },
            BlockSyncData { last_block: false, id: u32:3 },
            BlockSyncData { last_block: false, id: u32:4 },
            BlockSyncData { last_block: true, id: u32:5 },
        ];
        let tok = for ((counter, sync_data), tok): ((u32, BlockSyncData), token) in enumerate(SyncData) {
            let tok = send(tok, sync_s, sync_data);
            trace_fmt!("Sent #{} synchronization data, {:#x}", counter + u32:1, sync_data);
            (tok)
        }(tok);

        let DecodedRleBlocks: BatchTestVector[62] = [
            // 1st block
            (Symbol:0x01, bool:true),
            // 2nd block
            (Symbol:0x02, bool:false), (Symbol:0x02, bool:true),
            // 3rd block
            (Symbol:0x03, bool:false), (Symbol:0x03, bool:false), (Symbol:0x03, bool:false),
            (Symbol:0x03, bool:true),
            // 4th block
            (Symbol:0x04, bool:false), (Symbol:0x04, bool:false), (Symbol:0x04, bool:false),
            (Symbol:0x04, bool:false), (Symbol:0x04, bool:false), (Symbol:0x04, bool:false),
            (Symbol:0x04, bool:false), (Symbol:0x04, bool:true),
            // 5th block
            (Symbol:0x05, bool:false), (Symbol:0x05, bool:false), (Symbol:0x05, bool:false),
            (Symbol:0x05, bool:false), (Symbol:0x05, bool:false), (Symbol:0x05, bool:false),
            (Symbol:0x05, bool:false), (Symbol:0x05, bool:false), (Symbol:0x05, bool:false),
            (Symbol:0x05, bool:false), (Symbol:0x05, bool:false), (Symbol:0x05, bool:false),
            (Symbol:0x05, bool:false), (Symbol:0x05, bool:false), (Symbol:0x05, bool:false),
            (Symbol:0x05, bool:true),
            // 5th block
            (Symbol:0x06, bool:false), (Symbol:0x06, bool:false), (Symbol:0x06, bool:false),
            (Symbol:0x06, bool:false), (Symbol:0x06, bool:false), (Symbol:0x06, bool:false),
            (Symbol:0x06, bool:false), (Symbol:0x06, bool:false), (Symbol:0x06, bool:false),
            (Symbol:0x06, bool:false), (Symbol:0x06, bool:false), (Symbol:0x06, bool:false),
            (Symbol:0x06, bool:false), (Symbol:0x06, bool:false), (Symbol:0x06, bool:false),
            (Symbol:0x06, bool:false), (Symbol:0x06, bool:false), (Symbol:0x06, bool:false),
            (Symbol:0x06, bool:false), (Symbol:0x06, bool:false), (Symbol:0x06, bool:false),
            (Symbol:0x06, bool:false), (Symbol:0x06, bool:false), (Symbol:0x06, bool:false),
            (Symbol:0x06, bool:false), (Symbol:0x06, bool:false), (Symbol:0x06, bool:false),
            (Symbol:0x06, bool:false), (Symbol:0x06, bool:false), (Symbol:0x06, bool:false),
            (Symbol:0x06, bool:true),
        ];
        let tok = for ((counter, test_data), tok): ((u32, BatchTestVector), token) in enumerate(DecodedRleBlocks) {
            let symbol = test_data.0 as Symbol;
            let last = test_data.1;
            let data_in = RleOutput { symbol, last };
            let tok = send(tok, in_s, data_in);
            trace_fmt!("Sent #{} decoded rle symbol, {:#x}", counter + u32:1, data_in);
            (tok)
        }(tok);

        let BatchedDecodedRleSymbols: ExtendedBlockDataPacket[10] = [
            ExtendedBlockDataPacket {msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket {last: bool:true, last_block: bool:false, id: u32:0, data: BlockData:0x01, length: BlockPacketLength:8}},
            ExtendedBlockDataPacket {msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket {last: bool:true, last_block: bool:false, id: u32:1, data: BlockData:0x0202, length: BlockPacketLength:16}},
            ExtendedBlockDataPacket {msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket {last: bool:true, last_block: bool:false, id: u32:2, data: BlockData:0x03030303, length: BlockPacketLength:32}},
            ExtendedBlockDataPacket {msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket {last: bool:true, last_block: bool:false, id: u32:3, data: BlockData:0x0404040404040404, length: BlockPacketLength:64}},
            ExtendedBlockDataPacket { msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket {last: bool:false, last_block: bool:false, id: u32:4, data: BlockData:0x0505050505050505, length: BlockPacketLength:64}},
            ExtendedBlockDataPacket {msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket {last: bool:true, last_block: bool:false, id: u32:4, data: BlockData:0x0505050505050505, length: BlockPacketLength:64}},
            ExtendedBlockDataPacket {msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket {last: bool:false, last_block: bool:true, id: u32:5, data: BlockData:0x0606060606060606, length: BlockPacketLength:64}},
            ExtendedBlockDataPacket {msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket {last: bool:false, last_block: bool:true, id: u32:5, data: BlockData:0x0606060606060606, length: BlockPacketLength:64}},
            ExtendedBlockDataPacket {msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket {last: bool:false, last_block: bool:true, id: u32:5, data: BlockData:0x0606060606060606, length: BlockPacketLength:64}},
            ExtendedBlockDataPacket {msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket {last: bool:true, last_block: bool:true, id: u32:5, data: BlockData:0x06060606060606, length: BlockPacketLength:56}},
        ];

        let tok = for ((counter, expected), tok): ((u32, ExtendedBlockDataPacket), token) in enumerate(BatchedDecodedRleSymbols) {
            let (tok, dec_output) = recv(tok, out_r);
            trace_fmt!("Received #{} batched decoded rle symbols, {:#x}", counter + u32:1, dec_output);
            assert_eq(dec_output, expected);
            (tok)
        }(tok);
        send(tok, terminator, true);
    }
}

pub proc RleBlockDecoder {
    input_r: chan<BlockDataPacket> in;
    output_s: chan<ExtendedBlockDataPacket> out;

    config(input_r: chan<BlockDataPacket> in, output_s: chan<ExtendedBlockDataPacket> out) {
        let (in_s, in_r) = chan<RleInput>;
        let (out_s, out_r) = chan<RleOutput>;
        let (sync_s, sync_r) = chan<BlockSyncData>;

        spawn RleDataPacker(input_r, in_s, sync_s);
        spawn rle_dec::RunLengthDecoder<SYMBOL_WIDTH, BLOCK_SIZE_WIDTH>(
            in_r, out_s);
        spawn BatchPacker(out_r, sync_r, output_s);

        (input_r, output_s)
    }

    init {  }

    next(tok: token, state: ()) {  }
}

#[test_proc]
proc RleBlockDecoder_test {
    terminator: chan<bool> out;
    in_s: chan<BlockDataPacket> out;
    out_r: chan<ExtendedBlockDataPacket> in;

    config(terminator: chan<bool> out) {
        let (in_s, in_r) = chan<BlockDataPacket>;
        let (out_s, out_r) = chan<ExtendedBlockDataPacket>;

        spawn RleBlockDecoder(in_r, out_s);

        (terminator, in_s, out_r)
    }

    init {  }

    next(tok: token, state: ()) {
        let EncodedRleBlocks: RleTestVector[6] = [
            (Symbol:0x1, SymbolCount:0x1),
            (Symbol:0x2, SymbolCount:0x2),
            (Symbol:0x3, SymbolCount:0x4),
            (Symbol:0x4, SymbolCount:0x8),
            (Symbol:0x5, SymbolCount:0x10),
            (Symbol:0x6, SymbolCount:0x1F),
        ];
        let tok = for ((counter, block), tok): ((u32, RleTestVector), token) in enumerate(EncodedRleBlocks) {
            let last_block = (counter == (array_size(EncodedRleBlocks) - u32:1));
            let data_in = BlockDataPacket {
                last: true, // RLE block fits into single packet, each will be last for given block
                last_block,
                id: counter,
                data: block.0 as BlockData,
                length: block.1 as BlockPacketLength
            };
            let tok = send(tok, in_s, data_in);
            trace_fmt!("Sent #{} raw encoded block, {:#x}", counter + u32:1, data_in);
            (tok)
        }(tok);

        let BatchedDecodedRleSymbols: ExtendedBlockDataPacket[10] = [
            ExtendedBlockDataPacket {msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { last: bool:true,  last_block: bool:false, id: u32:0, data: BlockData:0x01, length: BlockPacketLength:8}},
            ExtendedBlockDataPacket {msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { last: bool:true,  last_block: bool:false, id: u32:1, data: BlockData:0x0202, length: BlockPacketLength:16}},
            ExtendedBlockDataPacket {msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { last: bool:true,  last_block: bool:false, id: u32:2, data: BlockData:0x03030303, length: BlockPacketLength:32}},
            ExtendedBlockDataPacket {msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { last: bool:true,  last_block: bool:false, id: u32:3, data: BlockData:0x0404040404040404, length: BlockPacketLength:64}},
            ExtendedBlockDataPacket {msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { last: bool:false, last_block: bool:false, id: u32:4, data: BlockData:0x0505050505050505, length: BlockPacketLength:64}},
            ExtendedBlockDataPacket {msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { last: bool:true,  last_block: bool:false, id: u32:4, data: BlockData:0x0505050505050505, length: BlockPacketLength:64}},
            ExtendedBlockDataPacket {msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { last: bool:false, last_block: bool:true,  id: u32:5, data: BlockData:0x0606060606060606, length: BlockPacketLength:64}},
            ExtendedBlockDataPacket {msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { last: bool:false, last_block: bool:true,  id: u32:5, data: BlockData:0x0606060606060606, length: BlockPacketLength:64}},
            ExtendedBlockDataPacket {msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { last: bool:false, last_block: bool:true,  id: u32:5, data: BlockData:0x0606060606060606, length: BlockPacketLength:64}},
            ExtendedBlockDataPacket {msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket { last: bool:true,  last_block: bool:true,  id: u32:5, data: BlockData:0x06060606060606, length: BlockPacketLength:56}},
        ];

        let tok = for ((counter, expected), tok): ((u32, ExtendedBlockDataPacket), token) in enumerate(BatchedDecodedRleSymbols) {
            let (tok, dec_output) = recv(tok, out_r);
            trace_fmt!("Received #{} batched decoded rle symbols, {:#x}", counter + u32:1, dec_output);
            assert_eq(dec_output, expected);
            (tok)
        }(tok);
        send(tok, terminator, true);
    }
}
