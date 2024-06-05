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

// This file contains the implementation of LiteralsDispatcher responsible for
// dispatching ZSTD Literals. More information about Literals' format can be found in:
// https://datatracker.ietf.org/doc/html/rfc8878#section-3.1.1.3.1

import xls.modules.zstd.common;

type LiteralsPathCtrl = common::LiteralsPathCtrl;
type LiteralsData = common::LiteralsData;
type LiteralsDataWithSync = common::LiteralsDataWithSync;
type RleLiteralsData = common::RleLiteralsData;
type LiteralType = common::LiteralType;
type Streams = common::Streams;
type DecompressedSize = common::DecompressedSize;

type RleLitData = common::RleLitData;
type RleLitRepeat = common::RleLitRepeat;
type LitData = common::LitData;
type LitLength = common::LitLength;
type LitID = common::LitID;

struct LiteralsDispatcherState {
    // literals type received from ctrl channel
    literals_type: LiteralType,
    // number of literals to be read. The initial value is received
    // from ctrl channel and decreased after each read from literals channel
    left_to_read: DecompressedSize,
    literals_id: LitID,
}

pub proc LiteralsDispatcher {
    literals_ctrl_r: chan<LiteralsPathCtrl> in;
    literals_data_r: chan<LiteralsData> in;
    raw_literals_s: chan<LiteralsDataWithSync> out;
    rle_literals_s: chan<RleLiteralsData> out;
    huff_literals_s: chan<LiteralsDataWithSync> out;

    config (
        literals_ctrl_r: chan<LiteralsPathCtrl> in,
        literals_data_r: chan<LiteralsData> in,
        raw_literals_s: chan<LiteralsDataWithSync> out,
        rle_literals_s: chan<RleLiteralsData> out,
        huff_literals_s: chan<LiteralsDataWithSync> out,
    ) {
        (
            literals_ctrl_r,
            literals_data_r,
            raw_literals_s,
            rle_literals_s,
            huff_literals_s,
        )
    }

    init { zero!<LiteralsDispatcherState>() }

    next(state: LiteralsDispatcherState ) {

        let tok = join();
        let do_recv_ctrl = (state.left_to_read == DecompressedSize:0);
        let (tok, literals_path_ctrl) = recv_if(tok, literals_ctrl_r, do_recv_ctrl, zero!<LiteralsPathCtrl>());

        let (literals_type, left_to_read) = if do_recv_ctrl {
            (literals_path_ctrl.literals_type, literals_path_ctrl.decompressed_size)
        } else {
            (state.literals_type, state.left_to_read)
        };

        // RLE literals consist of single byte
        let (tok, literals_data) = recv(tok, literals_data_r);

        let is_empty = left_to_read == DecompressedSize:0 && !literals_data.last;

        let left_to_read = if (literals_type == LiteralType::RLE) {
            DecompressedSize:0
        } else {
            left_to_read - (literals_data.length as DecompressedSize)
        };

        let literals_data = LiteralsDataWithSync {
            data: literals_data.data,
            length: literals_data.length,
            last: literals_data.last,
            id: state.literals_id,
            literals_last: true
        };

        let tok = send_if(tok, raw_literals_s, LiteralType::RAW == literals_type, literals_data);


        let rle_literals_data = RleLiteralsData {
            data: (literals_data.data as u8),
            repeat: literals_path_ctrl.decompressed_size,
            id: if (is_empty) { LitID:0 } else { literals_data.id },
            last: literals_data.last,
        };
        let tok = send_if(tok, rle_literals_s, LiteralType::RLE == literals_type, rle_literals_data);

        let do_send_huff = (
            LiteralType::COMP == literals_type || LiteralType::COMP_4 == literals_type ||
            LiteralType::TREELESS == literals_type || LiteralType::TREELESS_4 == literals_type
        );
        assert!(do_send_huff, "Huffmann coding not implemented yet");
        let tok = send_if(tok, huff_literals_s, false, zero!<LiteralsDataWithSync>());

        // empty RLE literals with last not set will not be sent by RLE decoder to buffer
        let literals_id = if (is_empty && state.literals_type == LiteralType::RLE) {
            state.literals_id
        } else {
            state.literals_id + LitID:1
        };
        LiteralsDispatcherState {
            literals_type: literals_type,
            left_to_read: left_to_read,
            literals_id: literals_id,
        }
    }
}

#[test_proc]
proc LiteralsDispatcher_test {
    terminator: chan<bool> out;
    literals_ctrl_s: chan<LiteralsPathCtrl> out;
    literals_data_s: chan<LiteralsData> out;
    raw_literals_r: chan<LiteralsDataWithSync> in;
    rle_literals_r: chan<RleLiteralsData> in;
    huff_literals_r: chan<LiteralsDataWithSync> in;

    config(terminator: chan<bool> out) {
        let (literals_ctrl_s, literals_ctrl_r) = chan<LiteralsPathCtrl>("literals_ctrl");
        let (literals_data_s, literals_data_r) = chan<LiteralsData>("literals_data");
        let (raw_literals_s, raw_literals_r) = chan<LiteralsDataWithSync>("raw_literals");
        let (rle_literals_s, rle_literals_r) = chan<RleLiteralsData>("rle_literals");
        let (huff_literals_s, huff_literals_r) = chan<LiteralsDataWithSync>("huff_literals");

        spawn LiteralsDispatcher(
            literals_ctrl_r,
            literals_data_r,
            raw_literals_s,
            rle_literals_s,
            huff_literals_s,
        );

        (
            terminator,
            literals_ctrl_s,
            literals_data_s,
            raw_literals_r,
            rle_literals_r,
            huff_literals_r,
        )
    }

    init { }

    next(state: ()) {
        let tok = join();
        let test_ctrl: LiteralsPathCtrl[6] = [
            LiteralsPathCtrl {data_conf: zero!<Streams>(), decompressed_size: DecompressedSize:8, literals_type: LiteralType::RAW},
            LiteralsPathCtrl {data_conf: zero!<Streams>(), decompressed_size: DecompressedSize:4, literals_type: LiteralType::RLE},
            LiteralsPathCtrl {data_conf: zero!<Streams>(), decompressed_size: DecompressedSize:13, literals_type: LiteralType::RLE},
            LiteralsPathCtrl {data_conf: zero!<Streams>(), decompressed_size: DecompressedSize:15, literals_type: LiteralType::RAW},
            LiteralsPathCtrl {data_conf: zero!<Streams>(), decompressed_size: DecompressedSize:123, literals_type: LiteralType::RLE},
            LiteralsPathCtrl {data_conf: zero!<Streams>(), decompressed_size: DecompressedSize:31, literals_type: LiteralType::RAW},
        ];
        let test_data: LiteralsData[10] = [
            // 0. RAW
            LiteralsData {data: LitData:0x1657_3465_A6DB_5DB0, length: LitLength:8, last: false},
            // 1. RLE
            LiteralsData {data: LitData:0x23, length: LitLength:1, last: false},
            // 2. RLE
            LiteralsData {data: LitData:0x35, length: LitLength:1, last: true},
            // 3. RAW
            LiteralsData {data: LitData:0x4CFB_41C6_7B60_5370, length: LitLength:8, last: false},
            LiteralsData {data: LitData:0x009B_0F9C_E1BA_A96D, length: LitLength:7, last: false},
            // 4. RLE
            LiteralsData {data: LitData:0x5A, length: LitLength:1, last: false},
            // 5. RAW
            LiteralsData {data: LitData:0x6094_3E96_1834_C247, length: LitLength:8, last: false},
            LiteralsData {data: LitData:0xBC02_D0E8_D728_9ABE, length: LitLength:8, last: false},
            LiteralsData {data: LitData:0xF864_C38B_E1FA_8D12, length: LitLength:8, last: false},
            LiteralsData {data: LitData:0xFC19_63F1_CE21_C294, length: LitLength:7, last: true},
        ];
        let expected_raw: LiteralsDataWithSync[7] = [
            // 0.
            LiteralsDataWithSync {data: LitData:0x1657_3465_A6DB_5DB0, length: LitLength:8, id: LitID:0, last: false, literals_last: true},
            // 3.
            LiteralsDataWithSync {data: LitData:0x4CFB_41C6_7B60_5370, length: LitLength:8, id: LitID:3, last: false, literals_last: true},
            LiteralsDataWithSync {data: LitData:0x009B_0F9C_E1BA_A96D, length: LitLength:7, id: LitID:4, last: false, literals_last: true},
            // 5.
            LiteralsDataWithSync {data: LitData:0x6094_3E96_1834_C247, length: LitLength:8, id: LitID:6, last: false, literals_last: true},
            LiteralsDataWithSync {data: LitData:0xBC02_D0E8_D728_9ABE, length: LitLength:8, id: LitID:7, last: false, literals_last: true},
            LiteralsDataWithSync {data: LitData:0xF864_C38B_E1FA_8D12, length: LitLength:8, id: LitID:8, last: false, literals_last: true},
            LiteralsDataWithSync {data: LitData:0xFC19_63F1_CE21_C294, length: LitLength:7, id: LitID:9, last: true, literals_last: true},
        ];
        let expected_rle: RleLiteralsData[3] = [
            // 1.
            RleLiteralsData {data: RleLitData:0x23, repeat: RleLitRepeat:4, id: LitID:1, last: false},
            // 2.
            RleLiteralsData {data: RleLitData:0x35, repeat: RleLitRepeat:13, id: LitID:2, last: true},
            // 4.
            RleLiteralsData {data: RleLitData:0x5A, repeat: RleLitRepeat:123, id: LitID:5, last: false},
        ];
        let tok = for ((counter, test_ctrl), tok): ((u32, LiteralsPathCtrl), token) in enumerate(test_ctrl) {
            let tok = send(tok, literals_ctrl_s, test_ctrl);
            trace_fmt!("Send #{} literals ctrl, {:#x}", counter + u32:1, test_ctrl);
            (tok)
        }(tok);

        let tok = for ((counter, test_data), tok): ((u32, LiteralsData), token) in enumerate(test_data) {
            let tok = send(tok, literals_data_s, test_data);
            trace_fmt!("Send #{} literals data, {:#x}", counter + u32:1, test_data);
            (tok)
        }(tok);

        let tok_1 = for ((counter, expected_raw), tok_1): ((u32, LiteralsDataWithSync), token) in enumerate(expected_raw) {
            let (tok_1, raw) = recv(tok_1, raw_literals_r);
            trace_fmt!("Recv #{} raw literals, {:#x}", counter + u32:1, raw);
            assert_eq(expected_raw, raw);
            (tok_1)
        }(tok);

        let tok_2 = for ((counter, expected_rle), tok_2): ((u32, RleLiteralsData), token) in enumerate(expected_rle) {
            let (tok_2, rle) = recv(tok_2, rle_literals_r);
            trace_fmt!("Recv #{} rle literals, {:#x}", counter + u32:1, rle);
            assert_eq(expected_rle, rle);
            (tok_2)
        }(tok);

        send(tok, terminator, true);
    }
}
