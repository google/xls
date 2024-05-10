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

// This file contains the implementation of RleLiteralsDecoder responsible for decoding
// ZSTD RLE Literals. More information about Rle Literals's format can be found in:
// https://datatracker.ietf.org/doc/html/rfc8878#section-3.1.1.3.1

import std;

import xls.modules.zstd.common;
import xls.modules.rle.rle_dec;
import xls.modules.rle.rle_common;

const RLE_LITERALS_DATA_WIDTH = common::RLE_LITERALS_DATA_WIDTH;
const RLE_LITERALS_REPEAT_WIDTH = common::RLE_LITERALS_REPEAT_WIDTH;
const LITERALS_DATA_WIDTH = common::LITERALS_DATA_WIDTH;
const LITERALS_LENGTH_WIDTH = common::LITERALS_LENGTH_WIDTH;

type RleInput = rle_common::CompressedData<RLE_LITERALS_DATA_WIDTH, RLE_LITERALS_REPEAT_WIDTH>;
type RleOutput = rle_common::PlainData<RLE_LITERALS_DATA_WIDTH>;
type RleLiteralsData = common::RleLiteralsData;
type LiteralsDataWithSync = common::LiteralsDataWithSync;

type RleLitData = common::RleLitData;
type RleLitRepeat = common::RleLitRepeat;
type LitData = common::LitData;
type LitID = common::LitID;
type LitLength = common::LitLength;

struct LiteralsSyncData {
    count: RleLitRepeat,
    id: LitID,
    last: bool,
}

proc RleDataPacker {
    literals_data_r: chan<RleLiteralsData> in;
    rle_data_s: chan<RleInput> out;
    sync_s: chan<LiteralsSyncData> out;

    config(
        literals_data_r: chan<RleLiteralsData> in,
        rle_data_s: chan<RleInput> out,
        sync_s: chan<LiteralsSyncData> out,
    ) {
        (literals_data_r, rle_data_s, sync_s)
    }

    init { }

    next(state: ()) {
        let tok = join();
        let (tok, input) = recv(tok, literals_data_r);
        let not_zero_repeat = (input.repeat != RleLitRepeat:0);
        let rle_dec_data = RleInput { symbol: input.data, count: input.repeat, last: true };
        let data_tok = send_if(tok, rle_data_s, not_zero_repeat, rle_dec_data);
        let sync_data = LiteralsSyncData { count: input.repeat, id: input.id, last: input.last };
        let sync_tok = send_if(data_tok, sync_s, not_zero_repeat || input.last, sync_data);
    }
}

#[test_proc]
proc RleDataPacker_test {
    terminator: chan<bool> out;
    in_s: chan<RleLiteralsData> out;
    out_r: chan<RleInput> in;
    sync_r: chan<LiteralsSyncData> in;

    config(terminator: chan<bool> out) {
        let (in_s, in_r) = chan<RleLiteralsData>("in");
        let (out_s, out_r) = chan<RleInput>("out");
        let (sync_s, sync_r) = chan<LiteralsSyncData>("sync");

        spawn RleDataPacker(in_r, out_s, sync_s);

        (terminator, in_s, out_r, sync_r)
    }

    init {  }

    next(state: ()) {
        let tok = join();
        let test_data: RleLiteralsData[5] = [
            RleLiteralsData {data: RleLitData:0xAB, repeat: RleLitRepeat:11, id: LitID:0, last: bool:0},
            RleLiteralsData {data: RleLitData:0xCD, repeat: RleLitRepeat:3, id: LitID:1, last: bool:0},
            RleLiteralsData {data: RleLitData:0x12, repeat: RleLitRepeat:16, id: LitID:2, last: bool:0},
            RleLiteralsData {data: RleLitData:0x34, repeat: RleLitRepeat:20, id: LitID:3, last: bool:0},
            RleLiteralsData {data: RleLitData:0x56, repeat: RleLitRepeat:2, id: LitID:4, last: bool:1},
        ];

        let tok = for ((counter, test_data), tok): ((u32, RleLiteralsData), token) in enumerate(test_data) {
            let expected_data_out = RleInput {
                symbol: test_data.data,
                count: test_data.repeat,
                last: true,
            };

            let expected_sync_out = LiteralsSyncData {
                count: test_data.repeat,
                id: test_data.id,
                last: test_data.last,
            };

            let tok = send(tok, in_s, test_data);
            trace_fmt!("Send #{} rle literals data, {:#x}", counter + u32:1, test_data);

            let (tok, data_out) = recv(tok, out_r);
            trace_fmt!("Received #{} rle input data, {:#x}", counter + u32:1, data_out);
            assert_eq(data_out, expected_data_out);

            let (tok, sync_out) = recv(tok, sync_r);
            trace_fmt!("Received #{} sync data, {:#x}", counter + u32:1, sync_out);
            assert_eq(sync_out, expected_sync_out);

            (tok)
        }(tok);

        send(tok, terminator, true);
    }
}

struct BatchPackerState {
    batch: LitData,
    data_in_batch: LitLength,
    count_left: RleLitRepeat,
    sync_id: LitID,
    sync_last: bool,
}

// auxiliary variable used to replace multiplication with shifts
const_assert!(std::is_pow2(RLE_LITERALS_DATA_WIDTH));
const RLE_LITERALS_DATA_WIDTH_SHIFT = std::clog2(RLE_LITERALS_DATA_WIDTH);

proc BatchPacker {
    rle_data_r: chan<RleOutput> in;
    sync_r: chan<LiteralsSyncData> in;
    literals_data_s: chan<LiteralsDataWithSync> out;

    config(
        rle_data_r: chan<RleOutput> in,
        sync_r: chan<LiteralsSyncData> in,
        literals_data_s: chan<LiteralsDataWithSync> out,
    ) {
        (rle_data_r, sync_r, literals_data_s)
    }

    init { zero!<BatchPackerState>() }

    next(state: BatchPackerState) {
        let tok = join();
        let no_count_left = (state.count_left == RleLitRepeat:0);
        let (tok, sync_data) = recv_if(tok, sync_r, no_count_left, zero!<LiteralsSyncData>());
        let (count_left, sync_id, sync_last) = if (no_count_left) {
            (sync_data.count, sync_data.id, sync_data.last)
        } else {
            (state.count_left, state.sync_id, state.sync_last)
        };

        let (literals_data, do_send_batch, state) = if (count_left != RleLitRepeat:0) {
            let (tok, decoded_data) = recv(tok, rle_data_r);

            let data_in_batch = state.data_in_batch;
            // shift batch and append new symbol
            let shift = (data_in_batch as u32) << RLE_LITERALS_DATA_WIDTH_SHIFT;

            let batch = state.batch | ((decoded_data.symbol as LitData) << shift);
            let data_in_batch = data_in_batch + LitLength:1;
            // send batch if it is the last batch or it is full
            let do_send_batch = (
                decoded_data.last
                | (((data_in_batch as u32 + u32:1) << RLE_LITERALS_DATA_WIDTH_SHIFT) > LITERALS_DATA_WIDTH)
            );
            let literals_data = LiteralsDataWithSync {
                data: batch,
                length: data_in_batch,
                last: sync_last && decoded_data.last,
                id: sync_id,
                literals_last: decoded_data.last,
            };

            let state = if do_send_batch {
                BatchPackerState {
                    count_left: count_left - RleLitRepeat:1,
                    sync_id: sync_id,
                    sync_last: sync_last,
                    ..zero!<BatchPackerState>()
                }
            } else {
                BatchPackerState {
                    batch: batch,
                    data_in_batch: data_in_batch,
                    count_left: count_left - RleLitRepeat:1,
                    sync_id: sync_id,
                    sync_last: sync_last,
                }
            };

            (literals_data, do_send_batch, state)
        } else if (sync_data.last) {
            // handle empty literal with last set
            (
                LiteralsDataWithSync {id: sync_id, last: true, literals_last: true, ..zero!<LiteralsDataWithSync>()},
                true,
                BatchPackerState {
                    count_left: count_left,
                    sync_id: sync_id,
                    sync_last: sync_last,
                    ..state
                },
            )
        } else {
            // handle empty literal with last not set
            (
                zero!<LiteralsDataWithSync>(),
                false,
                BatchPackerState {
                    count_left: count_left,
                    sync_id: sync_id,
                    sync_last: sync_last,
                    ..state
                },
            )
        };

        let data_tok = send_if(tok, literals_data_s, do_send_batch, literals_data);

        state
    }
}

#[test_proc]
proc BatchPacker_test {
    terminator: chan<bool> out;
    in_s: chan<RleOutput> out;
    sync_s: chan<LiteralsSyncData> out;
    out_r: chan<LiteralsDataWithSync> in;

    config(terminator: chan<bool> out) {
        let (in_s, in_r) = chan<RleOutput>("in");
        let (sync_s, sync_r) = chan<LiteralsSyncData>("sync");
        let (out_s, out_r) = chan<LiteralsDataWithSync>("out");

        spawn BatchPacker(in_r, sync_r, out_s);

        (terminator, in_s, sync_s, out_r)
    }

    init {  }

    next(state: ()) {
        let tok = join();
        let test_sync_data: LiteralsSyncData[4] = [
            LiteralsSyncData {count: RleLitRepeat:1, id: LitID:0, last: false},
            LiteralsSyncData {count: RleLitRepeat:8, id: LitID:1, last: false},
            LiteralsSyncData {count: RleLitRepeat:10, id: LitID:2, last: false},
            LiteralsSyncData {count: RleLitRepeat:13, id: LitID:3, last: true},
        ];
        let test_rle_data: RleOutput[32] = [
            // 1st literal
            RleOutput {symbol: RleLitData:0x11, last: true},
            // 2nd literal
            RleOutput {symbol: RleLitData:0x22, last: false}, RleOutput {symbol: RleLitData:0x22, last: false},
            RleOutput {symbol: RleLitData:0x22, last: false}, RleOutput {symbol: RleLitData:0x22, last: false},
            RleOutput {symbol: RleLitData:0x22, last: false}, RleOutput {symbol: RleLitData:0x22, last: false},
            RleOutput {symbol: RleLitData:0x22, last: false}, RleOutput {symbol: RleLitData:0x22, last: true},
            // 3rd literal
            RleOutput {symbol: RleLitData:0x33, last: false}, RleOutput {symbol: RleLitData:0x33, last: false},
            RleOutput {symbol: RleLitData:0x33, last: false}, RleOutput {symbol: RleLitData:0x33, last: false},
            RleOutput {symbol: RleLitData:0x33, last: false}, RleOutput {symbol: RleLitData:0x33, last: false},
            RleOutput {symbol: RleLitData:0x33, last: false}, RleOutput {symbol: RleLitData:0x33, last: false},
            RleOutput {symbol: RleLitData:0x33, last: false}, RleOutput {symbol: RleLitData:0x33, last: true},
            // 4th literal
            RleOutput {symbol: RleLitData:0x44, last: false}, RleOutput {symbol: RleLitData:0x44, last: false},
            RleOutput {symbol: RleLitData:0x44, last: false}, RleOutput {symbol: RleLitData:0x44, last: false},
            RleOutput {symbol: RleLitData:0x44, last: false}, RleOutput {symbol: RleLitData:0x44, last: false},
            RleOutput {symbol: RleLitData:0x44, last: false}, RleOutput {symbol: RleLitData:0x44, last: false},
            RleOutput {symbol: RleLitData:0x44, last: false}, RleOutput {symbol: RleLitData:0x44, last: false},
            RleOutput {symbol: RleLitData:0x44, last: false}, RleOutput {symbol: RleLitData:0x44, last: false},
            RleOutput {symbol: RleLitData:0x44, last: true},
        ];
        let test_out_data: LiteralsDataWithSync[6] = [
            LiteralsDataWithSync {data: LitData:0x0000_0000_0000_0011, length: LitLength:1, id: LitID:0, last: false, literals_last: true},
            LiteralsDataWithSync {data: LitData:0x2222_2222_2222_2222, length: LitLength:8, id: LitID:1, last: false, literals_last: true},
            LiteralsDataWithSync {data: LitData:0x3333_3333_3333_3333, length: LitLength:8, id: LitID:2, last: false, literals_last: false},
            LiteralsDataWithSync {data: LitData:0x0000_0000_0000_3333, length: LitLength:2, id: LitID:2, last: false, literals_last: true},
            LiteralsDataWithSync {data: LitData:0x4444_4444_4444_4444, length: LitLength:8, id: LitID:3, last: false, literals_last: false},
            LiteralsDataWithSync {data: LitData:0x0000_0044_4444_4444, length: LitLength:5, id: LitID:3, last: true, literals_last: true},
        ];

        let tok = for ((counter, sync_data), tok): ((u32, LiteralsSyncData), token) in enumerate(test_sync_data) {
            let tok = send(tok, sync_s, sync_data);
            trace_fmt!("Sent #{} synchronization data, {:#x}", counter + u32:1, sync_data);
            (tok)
        }(tok);

        let tok = for ((counter, rle_data), tok): ((u32, RleOutput), token) in enumerate(test_rle_data) {
            let tok = send(tok, in_s, rle_data);
            trace_fmt!("Sent #{} rle data, {:#x}", counter + u32:1, rle_data);
            (tok)
        }(tok);

        let tok = for ((counter, expected_out_data), tok): ((u32, LiteralsDataWithSync), token) in enumerate(test_out_data) {
            let (tok, out_data) = recv(tok, out_r);
            trace_fmt!("Received #{} batched data, {:#x}", counter + u32:1, out_data);
            assert_eq(out_data, expected_out_data);
            (tok)
        }(tok);

        send(tok, terminator, true);
    }
}

pub proc RleLiteralsDecoder {
    input_r: chan<RleLiteralsData> in;
    output_s: chan<LiteralsDataWithSync> out;

    config(input_r: chan<RleLiteralsData> in, output_s: chan<LiteralsDataWithSync> out) {
        let (in_s, in_r) = chan<RleInput, u32:1>("in");
        let (out_s, out_r) = chan<RleOutput, u32:1>("in");
        let (sync_s, sync_r) = chan<LiteralsSyncData, u32:1>("sync");

        spawn RleDataPacker(input_r, in_s, sync_s);
        spawn rle_dec::RunLengthDecoder<RLE_LITERALS_DATA_WIDTH, RLE_LITERALS_REPEAT_WIDTH>(in_r, out_s);
        spawn BatchPacker(out_r, sync_r, output_s);

        (input_r, output_s)
    }

    init { }

    next(state: ()) { }
}

#[test_proc]
proc RleLiteralsDecoder_test {
    terminator: chan<bool> out;
    in_s: chan<RleLiteralsData> out;
    out_r: chan<LiteralsDataWithSync> in;

    config (terminator: chan<bool> out) {
        let (in_s, in_r) = chan<RleLiteralsData>("in");
        let (out_s, out_r) = chan<LiteralsDataWithSync>("out");

        spawn RleLiteralsDecoder(in_r, out_s);

        (terminator, in_s, out_r)
    }

    init {  }

    next(state: ()) {
        let tok = join();
        let test_rle_data: RleLiteralsData[7] = [
            RleLiteralsData {data: RleLitData:0x11, repeat: RleLitRepeat:11, id: LitID:0, last: false},
            RleLiteralsData {data: RleLitData:0x22, repeat: RleLitRepeat:3, id: LitID:1, last: false},
            RleLiteralsData {data: RleLitData:0x33, repeat: RleLitRepeat:16, id: LitID:2, last: false},
            RleLiteralsData {data: RleLitData:0x44, repeat: RleLitRepeat:0, id: LitID:0, last: false},
            RleLiteralsData {data: RleLitData:0x55, repeat: RleLitRepeat:2, id: LitID:3, last: false},
            RleLiteralsData {data: RleLitData:0x66, repeat: RleLitRepeat:20, id: LitID:4, last: false},
            RleLiteralsData {data: RleLitData:0x00, repeat: RleLitRepeat:0, id: LitID:5, last: true},
        ];

        let test_out_data: LiteralsDataWithSync[10] = [
            // 1st literal
            LiteralsDataWithSync {data: LitData:0x1111_1111_1111_1111, length: LitLength:8, last: false, id: LitID:0, literals_last: false},
            LiteralsDataWithSync {data: LitData:0x0000_0000_0011_1111, length: LitLength:3, last: false, id: LitID:0, literals_last: true},
            // 2nd literal
            LiteralsDataWithSync {data: LitData:0x0000_0000_0022_2222, length: LitLength:3, last: false, id: LitID:1, literals_last: true},
            // 3rd literal
            LiteralsDataWithSync {data: LitData:0x3333_3333_3333_3333, length: LitLength:8, last: false, id: LitID:2, literals_last: false},
            LiteralsDataWithSync {data: LitData:0x3333_3333_3333_3333, length: LitLength:8, last: false, id: LitID:2, literals_last: true},
            // 4th literal (empty)
            // 5th literal
            LiteralsDataWithSync {data: LitData:0x0000_0000_0000_5555, length: LitLength:2, last: false, id: LitID:3, literals_last: true},
            // 6th literal
            LiteralsDataWithSync {data: LitData:0x6666_6666_6666_6666, length: LitLength:8, last: false, id: LitID:4, literals_last: false},
            LiteralsDataWithSync {data: LitData:0x6666_6666_6666_6666, length: LitLength:8, last: false, id: LitID:4, literals_last: false},
            LiteralsDataWithSync {data: LitData:0x0000_0000_6666_6666, length: LitLength:4, last: false, id: LitID:4, literals_last: true},
            // 7th literal
            LiteralsDataWithSync {data: LitData:0x0000_0000_0000_0000, length: LitLength:0, last: true, id: LitID:5, literals_last: true},
        ];

        let tok = for ((counter, rle_data), tok): ((u32, RleLiteralsData), token) in enumerate(test_rle_data) {
            let tok = send(tok, in_s, rle_data);
            trace_fmt!("Sent #{} rle data, {:#x}", counter + u32:1, rle_data);
            (tok)
        }(tok);

        let tok = for ((counter, expected_out_data), tok): ((u32, LiteralsDataWithSync), token) in enumerate(test_out_data) {
            let (tok, out_data) = recv(tok, out_r);
            trace_fmt!("Received #{} batched data, {:#x}", counter + u32:1, out_data);
            assert_eq(out_data, expected_out_data);
            (tok)
        }(tok);

        send(tok, terminator, true);
    }
}
