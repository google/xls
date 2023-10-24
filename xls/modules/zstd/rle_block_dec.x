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
    count: SymbolCount,
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

    next(state: ()) {
        let tok = join();
        let (tok, input) = recv(tok, block_data_r);
        let rle_dec_data = RleInput {
            symbol: input.data as Symbol, count: input.length as SymbolCount, last: true
        };
        // send RLE packet for decoding unless it has symbol count == 0
        let send_always = rle_dec_data.count != SymbolCount:0;
        let data_tok = send_if(tok, rle_data_s, send_always, rle_dec_data);
        let sync_data = BlockSyncData { last_block: input.last_block, count: rle_dec_data.count, id: input.id };
        // send last block packet even if it has symbol count == 0
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
        let (in_s, in_r) = chan<BlockDataPacket>("in");
        let (out_s, out_r) = chan<RleInput>("out");
        let (sync_s, sync_r) = chan<BlockSyncData>("sync");

        spawn RleDataPacker(in_r, out_s, sync_s);

        (terminator, in_s, out_r, sync_r)
    }

    init {  }

    next(state: ()) {
        let tok = join();
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
                count: block.1,
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

#[test_proc]
proc RleDataPacker_empty_blocks_test {
    terminator: chan<bool> out;
    in_s: chan<BlockDataPacket> out;
    out_r: chan<RleInput> in;
    sync_r: chan<BlockSyncData> in;

    config(terminator: chan<bool> out) {
        let (in_s, in_r) = chan<BlockDataPacket>("in");
        let (out_s, out_r) = chan<RleInput>("out");
        let (sync_s, sync_r) = chan<BlockSyncData>("sync");

        spawn RleDataPacker(in_r, out_s, sync_s);

        (terminator, in_s, out_r, sync_r)
    }

    init {  }

    next(state: ()) {
        let tok = join();
        let EncodedRleBlocks: RleTestVector[8] = [
            (Symbol:0xFF, SymbolCount:0x0),
            (Symbol:0x1, SymbolCount:0x1),
            (Symbol:0xFF, SymbolCount:0x0),
            (Symbol:0x3, SymbolCount:0x4),
            (Symbol:0xFF, SymbolCount:0x0),
            (Symbol:0x5, SymbolCount:0x10),
            (Symbol:0xFF, SymbolCount:0x0),
            (Symbol:0xFF, SymbolCount:0x0),
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
            (tok)
        }(tok);

        let RleInputs: RleInput[3] = [
            RleInput {last: true, symbol: Symbol:0x1, count: BlockSize:0x1},
            RleInput {last: true, symbol: Symbol:0x3, count: BlockSize:0x4},
            RleInput {last: true, symbol: Symbol:0x5, count: BlockSize:0x10},
        ];
        let tok = for ((counter, rle_in), tok): ((u32, RleInput), token) in enumerate(RleInputs) {
            let (tok, dec_output) = recv(tok, out_r);
            trace_fmt!("Received #{} packed rle encoded block, {:#x}", counter + u32:1, dec_output);
            assert_eq(dec_output, rle_in);
            (tok)
        }(tok);

        let BlockSyncDataInputs: BlockSyncData[8] = [
            BlockSyncData { id: 0, count: BlockSize:0x0, last_block: false },
            BlockSyncData { id: 1, count: BlockSize:0x1, last_block: false },
            BlockSyncData { id: 2, count: BlockSize:0x0, last_block: false },
            BlockSyncData { id: 3, count: BlockSize:0x4, last_block: false },
            BlockSyncData { id: 4, count: BlockSize:0x0, last_block: false },
            BlockSyncData { id: 5, count: BlockSize:0x10, last_block: false },
            BlockSyncData { id: 6, count: BlockSize:0x0, last_block: false },
            BlockSyncData { id: 7, count: BlockSize:0x0, last_block: true },
        ];
        let tok = for ((counter, sync_data), tok): ((u32, BlockSyncData), token) in enumerate(BlockSyncDataInputs) {
            let (tok, sync_output) = recv(tok, sync_r);
            trace_fmt!("Received #{} synchronization data, {:#x}", counter + u32:1, sync_output);
            assert_eq(sync_output, sync_data);
            (tok)
        }(tok);
        send(tok, terminator, true);
    }
}

struct BatchPackerState {
    batch: BlockData,
    symbols_in_batch: BlockPacketLength,
    symbols_in_block: BlockPacketLength,
    prev_last: bool,
    prev_sync: BlockSyncData,
}

const ZERO_BATCH_STATE = zero!<BatchPackerState>();
const ZERO_BLOCK_SYNC_DATA = zero!<BlockSyncData>();
const ZERO_RLE_OUTPUT = zero!<RleOutput>();
const EMPTY_RLE_OUTPUT = RleOutput {last: true, ..ZERO_RLE_OUTPUT};

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
    init { (BatchPackerState { prev_last: true, ..ZERO_BATCH_STATE }) }

    next(state: BatchPackerState) {
        let tok = join();
        trace_fmt!("start state: {:#x}", state);
        let prev_expected_symbols_in_block = state.prev_sync.count as BlockPacketLength;
        let symbols_in_batch = state.symbols_in_batch;
        let symbols_in_block = state.symbols_in_block;
        let block_in_progress = (symbols_in_block != prev_expected_symbols_in_block);
        trace_fmt!("block_in_progress: {:#x}", block_in_progress);

        // Finished receiving RLE data of the previous block
        // Proceed with receiving sync data for the next block
        let start_new_block = !block_in_progress;
        let (tok, sync_data) = recv_if(tok, sync_r, start_new_block, state.prev_sync);
        if (start_new_block) {
            trace_fmt!("received sync_data: {:#x}", sync_data);
        } else {
            trace_fmt!("got sync_data from the state: {:#x}", sync_data);
        };

        let expected_symbols_in_block = if (start_new_block) { sync_data.count as BlockPacketLength } else { prev_expected_symbols_in_block };
        trace_fmt!("expected_symbols_in_block: {:#x}", expected_symbols_in_block);

        let batch = state.batch;
        let empty_block = (expected_symbols_in_block == BlockPacketLength:0);
        trace_fmt!("batch: {:#x}", batch);
        trace_fmt!("empty_block: {:#x}", empty_block);

        let do_recv_rle = !empty_block && block_in_progress;
        let default_rle_output = if (empty_block) { EMPTY_RLE_OUTPUT } else { ZERO_RLE_OUTPUT };
        let (tok, decoded_data) = recv_if(tok, rle_data_r, do_recv_rle, default_rle_output);
        if (do_recv_rle) {
            trace_fmt!("received rle_data: {:#x}", decoded_data);
        } else {
            trace_fmt!("got empty rle_data: {:#x}", decoded_data);
        };

        let (batch, symbols_in_batch, symbols_in_block) = if (do_recv_rle) {
            // TODO: Improve performance: remove variable shift
            let shift = symbols_in_batch << u32:3;  // multiply by 8 bits
            let updated_batch = batch | ((decoded_data.symbol as BlockData) << shift);
            let updated_symbols_in_batch = symbols_in_batch + BlockPacketLength:1;
            let updated_symbols_in_block = symbols_in_block + BlockPacketLength:1;
            (updated_batch, updated_symbols_in_batch, updated_symbols_in_block)
        } else {
            (batch, symbols_in_batch, symbols_in_block)
        };
        trace_fmt!("updated batch: {:#x}", batch);
        trace_fmt!("updated symbols_in_batch: {:#x}", symbols_in_batch);
        trace_fmt!("updated symbols_in_block: {:#x}", symbols_in_block);

        let block_in_progress = (symbols_in_block != expected_symbols_in_block);
        trace_fmt!("updated block_in_progress: {:#x}", block_in_progress);

        // Last should not occur when batch is still being processed
        assert!(!(!block_in_progress ^ decoded_data.last), "corrupted_decoding_flow");

        let batch_full = symbols_in_batch >= BATCH_SIZE;
        trace_fmt!("batch_full: {:#x}", batch_full);
        // Send decoded RLE packet when
        // - batch size reached the maximal size
        // - RLE block decoding is finished
        // - Decoded RLE block is empty and is the last block in ZSTD frame
        let last = decoded_data.last || (sync_data.last_block && empty_block);
        let do_send_batch = (batch_full || last);
        trace_fmt!("do_send_batch: {:#x}", do_send_batch);

        let decoded_batch_data = ExtendedBlockDataPacket {
            // Decoded RLE block is always a literal
            msg_type: SequenceExecutorMessageType::LITERAL,
            packet: BlockDataPacket {
                last: last,
                last_block: sync_data.last_block,
                id: sync_data.id,
                data: batch as BlockData,
                // length in bits
                length: (symbols_in_batch << 3) as BlockPacketLength,
            }
        };

        let data_tok =
            send_if(tok, block_data_s, do_send_batch, decoded_batch_data);
        if (do_send_batch) {
            trace_fmt!("sent decoded_batch_data: {:#x}", decoded_batch_data);
        } else {
            trace_fmt!("decoded_batch_data: {:#x}", decoded_batch_data);
        };

        let (new_batch, new_symbols_in_batch) = if (do_send_batch) {
            (BlockData:0, BlockPacketLength:0)
        } else {
            (batch, symbols_in_batch)
        };

        let (new_sync_data, new_symbols_in_block) = if (decoded_data.last || (sync_data.last_block && empty_block)) {
            (ZERO_BLOCK_SYNC_DATA, BlockPacketLength:0)
        } else {
            (sync_data, symbols_in_block)
        };

        let new_state = BatchPackerState {
            batch: new_batch,
            symbols_in_batch: new_symbols_in_batch,
            symbols_in_block: new_symbols_in_block,
            prev_last: decoded_data.last,
            prev_sync: new_sync_data
        };

        trace_fmt!("new_state: {:#x}", new_state);

        new_state
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
        let (in_s, in_r) = chan<RleOutput>("in");
        let (sync_s, sync_r) = chan<BlockSyncData>("sync");
        let (out_s, out_r) = chan<ExtendedBlockDataPacket>("out");

        spawn BatchPacker(in_r, sync_r, out_s);

        (terminator, in_s, sync_s, out_r)
    }

    init {  }

    next(state: ()) {
        let tok = join();
        let SyncData: BlockSyncData[6] = [
            BlockSyncData { last_block: false, count: SymbolCount:1, id: u32:0 },
            BlockSyncData { last_block: false, count: SymbolCount:2, id: u32:1 },
            BlockSyncData { last_block: false, count: SymbolCount:4, id: u32:2 },
            BlockSyncData { last_block: false, count: SymbolCount:8, id: u32:3 },
            BlockSyncData { last_block: false, count: SymbolCount:16, id: u32:4 },
            BlockSyncData { last_block: true, count: SymbolCount:31, id: u32:5 },
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
            // 6th block
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
            ExtendedBlockDataPacket {msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket {last: bool:false, last_block: bool:false, id: u32:4, data: BlockData:0x0505050505050505, length: BlockPacketLength:64}},
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

#[test_proc]
proc BatchPacker_empty_blocks_test {
    terminator: chan<bool> out;
    in_s: chan<RleOutput> out;
    sync_s: chan<BlockSyncData> out;
    out_r: chan<ExtendedBlockDataPacket> in;

    config(terminator: chan<bool> out) {
        let (in_s, in_r) = chan<RleOutput>("in");
        let (sync_s, sync_r) = chan<BlockSyncData>("sync");
        let (out_s, out_r) = chan<ExtendedBlockDataPacket>("out");

        spawn BatchPacker(in_r, sync_r, out_s);

        (terminator, in_s, sync_s, out_r)
    }

    init {  }

    next(state: ()) {
        let tok = join();
        let SyncData: BlockSyncData[8] = [
            BlockSyncData { last_block: false, count: SymbolCount:0, id: u32:0 },
            BlockSyncData { last_block: false, count: SymbolCount:1, id: u32:1 },
            BlockSyncData { last_block: false, count: SymbolCount:0, id: u32:2 },
            BlockSyncData { last_block: false, count: SymbolCount:4, id: u32:3 },
            BlockSyncData { last_block: false, count: SymbolCount:0, id: u32:4 },
            BlockSyncData { last_block: false, count: SymbolCount:16, id: u32:5 },
            BlockSyncData { last_block: false, count: SymbolCount:0, id: u32:6 },
            BlockSyncData { last_block: true, count: SymbolCount:0, id: u32:7 },
        ];
        let tok = for ((counter, sync_data), tok): ((u32, BlockSyncData), token) in enumerate(SyncData) {
            let tok = send(tok, sync_s, sync_data);
            trace_fmt!("Sent #{} synchronization data, {:#x}", counter + u32:1, sync_data);
            (tok)
        }(tok);

        let DecodedRleBlocks: BatchTestVector[21] = [
            // 0 block
            // EMPTY
            // 1st block
            (Symbol:0x01, bool:true),
            // 2nd block
            // EMPTY
            // 3rd block
            (Symbol:0x03, bool:false), (Symbol:0x03, bool:false), (Symbol:0x03, bool:false),
            (Symbol:0x03, bool:true),
            // 4th block
            // EMPTY
            // 5th block
            (Symbol:0x05, bool:false), (Symbol:0x05, bool:false), (Symbol:0x05, bool:false),
            (Symbol:0x05, bool:false), (Symbol:0x05, bool:false), (Symbol:0x05, bool:false),
            (Symbol:0x05, bool:false), (Symbol:0x05, bool:false), (Symbol:0x05, bool:false),
            (Symbol:0x05, bool:false), (Symbol:0x05, bool:false), (Symbol:0x05, bool:false),
            (Symbol:0x05, bool:false), (Symbol:0x05, bool:false), (Symbol:0x05, bool:false),
            (Symbol:0x05, bool:true),
            // 6th block
            // EMPTY
            // 7th block
            // EMPTY
        ];
        let tok = for ((counter, test_data), tok): ((u32, BatchTestVector), token) in enumerate(DecodedRleBlocks) {
            let symbol = test_data.0 as Symbol;
            let last = test_data.1;
            let data_in = RleOutput { symbol, last };
            let tok = send(tok, in_s, data_in);
            trace_fmt!("Sent #{} decoded rle symbol, {:#x}", counter + u32:1, data_in);
            (tok)
        }(tok);

        let BatchedDecodedRleSymbols: ExtendedBlockDataPacket[9] = [
            // 0 block
            // EMPTY
            ExtendedBlockDataPacket {msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket {last: bool:true, last_block: bool:false, id: u32:0, data: BlockData:0x0, length: BlockPacketLength:0}},
            // 1st block
            ExtendedBlockDataPacket {msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket {last: bool:true, last_block: bool:false, id: u32:1, data: BlockData:0x01, length: BlockPacketLength:8}},
            // 2nd block
            // EMPTY
            ExtendedBlockDataPacket {msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket {last: bool:true, last_block: bool:false, id: u32:2, data: BlockData:0x0, length: BlockPacketLength:0}},
            // 3rd block
            ExtendedBlockDataPacket {msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket {last: bool:true, last_block: bool:false, id: u32:3, data: BlockData:0x03030303, length: BlockPacketLength:32}},
            // 4th block
            // EMPTY
            ExtendedBlockDataPacket {msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket {last: bool:true, last_block: bool:false, id: u32:4, data: BlockData:0x0, length: BlockPacketLength:0}},
            // 5th block
            ExtendedBlockDataPacket {msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket {last: bool:false, last_block: bool:false, id: u32:5, data: BlockData:0x0505050505050505, length: BlockPacketLength:64}},
            ExtendedBlockDataPacket {msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket {last: bool:true, last_block: bool:false, id: u32:5, data: BlockData:0x0505050505050505, length: BlockPacketLength:64}},
            // 6th block
            // EMPTY
            ExtendedBlockDataPacket {msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket {last: bool:true, last_block: bool:false, id: u32:6, data: BlockData:0x0, length: BlockPacketLength:0}},
            // 7th block
            // EMPTY with LAST_BLOCK
            ExtendedBlockDataPacket {msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket {last: bool:true, last_block: bool:true, id: u32:7, data: BlockData:0x0, length: BlockPacketLength:0}},
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
        let (in_s, in_r) = chan<RleInput, u32:1>("in");
        let (out_s, out_r) = chan<RleOutput, u32:1>("out");
        let (sync_s, sync_r) = chan<BlockSyncData, u32:1>("sync");

        spawn RleDataPacker(input_r, in_s, sync_s);
        spawn rle_dec::RunLengthDecoder<SYMBOL_WIDTH, BLOCK_SIZE_WIDTH>(
            in_r, out_s);
        spawn BatchPacker(out_r, sync_r, output_s);

        (input_r, output_s)
    }

    init {  }

    next(state: ()) {  }
}

#[test_proc]
proc RleBlockDecoder_test {
    terminator: chan<bool> out;
    in_s: chan<BlockDataPacket> out;
    out_r: chan<ExtendedBlockDataPacket> in;

    config(terminator: chan<bool> out) {
        let (in_s, in_r) = chan<BlockDataPacket>("in");
        let (out_s, out_r) = chan<ExtendedBlockDataPacket>("out");

        spawn RleBlockDecoder(in_r, out_s);

        (terminator, in_s, out_r)
    }

    init {  }

    next(state: ()) {
        let tok = join();
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

#[test_proc]
proc RleBlockDecoder_empty_blocks_test {
    terminator: chan<bool> out;
    in_s: chan<BlockDataPacket> out;
    out_r: chan<ExtendedBlockDataPacket> in;

    config(terminator: chan<bool> out) {
        let (in_s, in_r) = chan<BlockDataPacket>("in");
        let (out_s, out_r) = chan<ExtendedBlockDataPacket>("out");

        spawn RleBlockDecoder(in_r, out_s);

        (terminator, in_s, out_r)
    }

    init {  }

    next(state: ()) {
        let tok = join();
        let EncodedRleBlocks: RleTestVector[8] = [
            (Symbol:0xFF, SymbolCount:0x0),
            (Symbol:0x1, SymbolCount:0x1),
            (Symbol:0xFF, SymbolCount:0x0),
            (Symbol:0x3, SymbolCount:0x4),
            (Symbol:0xFF, SymbolCount:0x0),
            (Symbol:0x5, SymbolCount:0x10),
            (Symbol:0xFF, SymbolCount:0x0),
            (Symbol:0xFF, SymbolCount:0x0),
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

        let BatchedDecodedRleSymbols: ExtendedBlockDataPacket[9] = [
            // 0 block
            // EMPTY
            ExtendedBlockDataPacket {msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket {last: bool:true, last_block: bool:false, id: u32:0, data: BlockData:0x0, length: BlockPacketLength:0}},
            // 1st block
            ExtendedBlockDataPacket {msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket {last: bool:true, last_block: bool:false, id: u32:1, data: BlockData:0x01, length: BlockPacketLength:8}},
            // 2nd block
            // EMPTY
            ExtendedBlockDataPacket {msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket {last: bool:true, last_block: bool:false, id: u32:2, data: BlockData:0x0, length: BlockPacketLength:0}},
            // 3rd block
            ExtendedBlockDataPacket {msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket {last: bool:true, last_block: bool:false, id: u32:3, data: BlockData:0x03030303, length: BlockPacketLength:32}},
            // 4th block
            // EMPTY
            ExtendedBlockDataPacket {msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket {last: bool:true, last_block: bool:false, id: u32:4, data: BlockData:0x0, length: BlockPacketLength:0}},
            // 5th block
            ExtendedBlockDataPacket {msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket {last: bool:false, last_block: bool:false, id: u32:5, data: BlockData:0x0505050505050505, length: BlockPacketLength:64}},
            ExtendedBlockDataPacket {msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket {last: bool:true, last_block: bool:false, id: u32:5, data: BlockData:0x0505050505050505, length: BlockPacketLength:64}},
            // 6th block
            // EMPTY
            ExtendedBlockDataPacket {msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket {last: bool:true, last_block: bool:false, id: u32:6, data: BlockData:0x0, length: BlockPacketLength:0}},
            // 7th block
            // EMPTY with LAST_BLOCK
            ExtendedBlockDataPacket {msg_type: SequenceExecutorMessageType::LITERAL, packet: BlockDataPacket {last: bool:true, last_block: bool:true, id: u32:7, data: BlockData:0x0, length: BlockPacketLength:0}},
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
