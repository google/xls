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

// This file contains the implementation of LiteralsDecoder.

import std;

import xls.examples.ram;
import xls.modules.zstd.common as common;
import xls.modules.zstd.literals_buffer as literals_buffer;
import xls.modules.zstd.literals_dispatcher as literals_dispatcher;
import xls.modules.zstd.parallel_rams as parallel_rams;
import xls.modules.zstd.ram_printer as ram_printer;
import xls.modules.zstd.raw_literals_dec as raw_literals_dec;
import xls.modules.zstd.rle_literals_dec as rle_literals_dec;

type CopyOrMatchContent = common::CopyOrMatchContent;
type CopyOrMatchLength = common::CopyOrMatchLength;
type LitData = common::LitData;
type LitLength = common::LitLength;
type LiteralType = common::LiteralType;
type LiteralsBufferCtrl = common::LiteralsBufferCtrl;
type LiteralsData = common::LiteralsData;
type LiteralsDataWithSync = common::LiteralsDataWithSync;
type LiteralsPathCtrl = common::LiteralsPathCtrl;
type RleLiteralsData = common::RleLiteralsData;
type SequenceExecutorMessageType = common::SequenceExecutorMessageType;
type SequenceExecutorPacket = common::SequenceExecutorPacket<common::SYMBOL_WIDTH>;
type Streams = common::Streams;

proc LiteralsDecoder<
    HISTORY_BUFFER_SIZE_KB: u32,
    RAM_SIZE: u32 = {parallel_rams::ram_size(HISTORY_BUFFER_SIZE_KB)},
    RAM_ADDR_WIDTH: u32 = {parallel_rams::ram_addr_width(HISTORY_BUFFER_SIZE_KB)},
> {
    type ReadReq = ram::ReadReq<RAM_ADDR_WIDTH, literals_buffer::RAM_NUM_PARTITIONS>;
    type ReadResp = ram::ReadResp<literals_buffer::RAM_DATA_WIDTH>;
    type WriteReq = ram::WriteReq<RAM_ADDR_WIDTH, literals_buffer::RAM_DATA_WIDTH, literals_buffer::RAM_NUM_PARTITIONS>;
    type WriteResp = ram::WriteResp;

    literals_ctrl_r: chan<LiteralsPathCtrl> in;
    literals_data_r: chan<LiteralsData> in;
    literals_buf_ctrl_r: chan<LiteralsBufferCtrl> in;
    literals_s: chan<SequenceExecutorPacket> out;

    config (
        literals_ctrl_r: chan<LiteralsPathCtrl> in,
        literals_data_r: chan<LiteralsData> in,
        literals_buf_ctrl_r: chan<LiteralsBufferCtrl> in,
        literals_s: chan<SequenceExecutorPacket> out,
        rd_req_m0_s: chan<ReadReq> out,
        rd_req_m1_s: chan<ReadReq> out,
        rd_req_m2_s: chan<ReadReq> out,
        rd_req_m3_s: chan<ReadReq> out,
        rd_req_m4_s: chan<ReadReq> out,
        rd_req_m5_s: chan<ReadReq> out,
        rd_req_m6_s: chan<ReadReq> out,
        rd_req_m7_s: chan<ReadReq> out,
        rd_resp_m0_r: chan<ReadResp> in,
        rd_resp_m1_r: chan<ReadResp> in,
        rd_resp_m2_r: chan<ReadResp> in,
        rd_resp_m3_r: chan<ReadResp> in,
        rd_resp_m4_r: chan<ReadResp> in,
        rd_resp_m5_r: chan<ReadResp> in,
        rd_resp_m6_r: chan<ReadResp> in,
        rd_resp_m7_r: chan<ReadResp> in,
        wr_req_m0_s: chan<WriteReq> out,
        wr_req_m1_s: chan<WriteReq> out,
        wr_req_m2_s: chan<WriteReq> out,
        wr_req_m3_s: chan<WriteReq> out,
        wr_req_m4_s: chan<WriteReq> out,
        wr_req_m5_s: chan<WriteReq> out,
        wr_req_m6_s: chan<WriteReq> out,
        wr_req_m7_s: chan<WriteReq> out,
        wr_resp_m0_r: chan<WriteResp> in,
        wr_resp_m1_r: chan<WriteResp> in,
        wr_resp_m2_r: chan<WriteResp> in,
        wr_resp_m3_r: chan<WriteResp> in,
        wr_resp_m4_r: chan<WriteResp> in,
        wr_resp_m5_r: chan<WriteResp> in,
        wr_resp_m6_r: chan<WriteResp> in,
        wr_resp_m7_r: chan<WriteResp> in
    ) {
        let (raw_literals_s, raw_literals_r) = chan<LiteralsDataWithSync, u32:1>("raw_literals");
        let (rle_literals_s, rle_literals_r) = chan<RleLiteralsData, u32:1>("rle_literals");
        let (huff_literals_s, huff_literals_r) = chan<LiteralsDataWithSync, u32:1>("huff_literals");

        let (decoded_raw_literals_s, decoded_raw_literals_r) = chan<LiteralsDataWithSync, u32:1>("decoded_raw_literals");
        let (decoded_rle_literals_s, decoded_rle_literals_r) = chan<LiteralsDataWithSync, u32:1>("decoded_rle_literals");

        spawn literals_dispatcher::LiteralsDispatcher(
            literals_ctrl_r, literals_data_r,
            raw_literals_s, rle_literals_s, huff_literals_s,
        );

        spawn raw_literals_dec::RawLiteralsDecoder(raw_literals_r, decoded_raw_literals_s);

        spawn rle_literals_dec::RleLiteralsDecoder(rle_literals_r, decoded_rle_literals_s);

        spawn literals_buffer::LiteralsBuffer<HISTORY_BUFFER_SIZE_KB> (
            decoded_raw_literals_r, decoded_rle_literals_r, huff_literals_r,
            literals_buf_ctrl_r, literals_s,
            rd_req_m0_s, rd_req_m1_s, rd_req_m2_s, rd_req_m3_s,
            rd_req_m4_s, rd_req_m5_s, rd_req_m6_s, rd_req_m7_s,
            rd_resp_m0_r, rd_resp_m1_r, rd_resp_m2_r, rd_resp_m3_r,
            rd_resp_m4_r, rd_resp_m5_r, rd_resp_m6_r, rd_resp_m7_r,
            wr_req_m0_s, wr_req_m1_s, wr_req_m2_s, wr_req_m3_s,
            wr_req_m4_s, wr_req_m5_s, wr_req_m6_s, wr_req_m7_s,
            wr_resp_m0_r, wr_resp_m1_r, wr_resp_m2_r, wr_resp_m3_r,
            wr_resp_m4_r, wr_resp_m5_r, wr_resp_m6_r, wr_resp_m7_r,
        );

        (
            literals_ctrl_r, literals_data_r,
            literals_buf_ctrl_r, literals_s,
        )
    }

    init { }

    next (state: ()) { }
}

const ZSTD_HISTORY_BUFFER_SIZE_KB: u32 = u32:64;
const ZSTD_RAM_ADDR_WIDTH: u32 = parallel_rams::ram_addr_width(ZSTD_HISTORY_BUFFER_SIZE_KB);

proc LiteralsDecoderInst {
    type ReadReq = ram::ReadReq<ZSTD_RAM_ADDR_WIDTH, literals_buffer::RAM_NUM_PARTITIONS>;
    type ReadResp = ram::ReadResp<literals_buffer::RAM_DATA_WIDTH>;
    type WriteReq = ram::WriteReq<ZSTD_RAM_ADDR_WIDTH, literals_buffer::RAM_DATA_WIDTH, literals_buffer::RAM_NUM_PARTITIONS>;
    type WriteResp = ram::WriteResp;

    config (
        literals_ctrl_r: chan<LiteralsPathCtrl> in,
        literals_data_r: chan<LiteralsData> in,
        literals_buf_ctrl_r: chan<LiteralsBufferCtrl> in,
        literals_s: chan<SequenceExecutorPacket> out,
        rd_req_m0_s: chan<ReadReq> out,
        rd_req_m1_s: chan<ReadReq> out,
        rd_req_m2_s: chan<ReadReq> out,
        rd_req_m3_s: chan<ReadReq> out,
        rd_req_m4_s: chan<ReadReq> out,
        rd_req_m5_s: chan<ReadReq> out,
        rd_req_m6_s: chan<ReadReq> out,
        rd_req_m7_s: chan<ReadReq> out,
        rd_resp_m0_r: chan<ReadResp> in,
        rd_resp_m1_r: chan<ReadResp> in,
        rd_resp_m2_r: chan<ReadResp> in,
        rd_resp_m3_r: chan<ReadResp> in,
        rd_resp_m4_r: chan<ReadResp> in,
        rd_resp_m5_r: chan<ReadResp> in,
        rd_resp_m6_r: chan<ReadResp> in,
        rd_resp_m7_r: chan<ReadResp> in,
        wr_req_m0_s: chan<WriteReq> out,
        wr_req_m1_s: chan<WriteReq> out,
        wr_req_m2_s: chan<WriteReq> out,
        wr_req_m3_s: chan<WriteReq> out,
        wr_req_m4_s: chan<WriteReq> out,
        wr_req_m5_s: chan<WriteReq> out,
        wr_req_m6_s: chan<WriteReq> out,
        wr_req_m7_s: chan<WriteReq> out,
        wr_resp_m0_r: chan<WriteResp> in,
        wr_resp_m1_r: chan<WriteResp> in,
        wr_resp_m2_r: chan<WriteResp> in,
        wr_resp_m3_r: chan<WriteResp> in,
        wr_resp_m4_r: chan<WriteResp> in,
        wr_resp_m5_r: chan<WriteResp> in,
        wr_resp_m6_r: chan<WriteResp> in,
        wr_resp_m7_r: chan<WriteResp> in
    ) {
        spawn LiteralsDecoder<ZSTD_HISTORY_BUFFER_SIZE_KB> (
            literals_ctrl_r, literals_data_r,
            literals_buf_ctrl_r, literals_s,
            rd_req_m0_s, rd_req_m1_s, rd_req_m2_s, rd_req_m3_s,
            rd_req_m4_s, rd_req_m5_s, rd_req_m6_s, rd_req_m7_s,
            rd_resp_m0_r, rd_resp_m1_r, rd_resp_m2_r, rd_resp_m3_r,
            rd_resp_m4_r, rd_resp_m5_r, rd_resp_m6_r, rd_resp_m7_r,
            wr_req_m0_s, wr_req_m1_s, wr_req_m2_s, wr_req_m3_s,
            wr_req_m4_s, wr_req_m5_s, wr_req_m6_s, wr_req_m7_s,
            wr_resp_m0_r, wr_resp_m1_r, wr_resp_m2_r, wr_resp_m3_r,
            wr_resp_m4_r, wr_resp_m5_r, wr_resp_m6_r, wr_resp_m7_r,
        );
    }

    init {}

    next (state: ()) {}
}

// RAM related constants common for tests
const TEST_HISTORY_BUFFER_SIZE_KB = u32:1;
const TEST_RAM_SIZE = parallel_rams::ram_size(TEST_HISTORY_BUFFER_SIZE_KB);
const TEST_RAM_ADDR_WIDTH = parallel_rams::ram_addr_width(TEST_HISTORY_BUFFER_SIZE_KB);
const TEST_RAM_INITIALIZED = true;
const TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR = ram::SimultaneousReadWriteBehavior::READ_BEFORE_WRITE;

type TestRamAddr = bits[TEST_RAM_ADDR_WIDTH];
type TestWriteReq = ram::WriteReq<TEST_RAM_ADDR_WIDTH, literals_buffer::RAM_DATA_WIDTH, literals_buffer::RAM_NUM_PARTITIONS>;
type TestWriteResp = ram::WriteResp<TEST_RAM_ADDR_WIDTH>;
type TestReadReq = ram::ReadReq<TEST_RAM_ADDR_WIDTH, literals_buffer::RAM_NUM_PARTITIONS>;
type TestReadResp = ram::ReadResp<literals_buffer::RAM_DATA_WIDTH>;

const TEST_CTRL: LiteralsPathCtrl[7] = [
    LiteralsPathCtrl {data_conf: zero!<Streams>(), decompressed_size: u20:8, literals_type: LiteralType::RAW},
    LiteralsPathCtrl {data_conf: zero!<Streams>(), decompressed_size: u20:4, literals_type: LiteralType::RLE},
    LiteralsPathCtrl {data_conf: zero!<Streams>(), decompressed_size: u20:2, literals_type: LiteralType::RLE},
    LiteralsPathCtrl {data_conf: zero!<Streams>(), decompressed_size: u20:15, literals_type: LiteralType::RAW},
    LiteralsPathCtrl {data_conf: zero!<Streams>(), decompressed_size: u20:12, literals_type: LiteralType::RLE},
    LiteralsPathCtrl {data_conf: zero!<Streams>(), decompressed_size: u20:0, literals_type: LiteralType::RLE},
    LiteralsPathCtrl {data_conf: zero!<Streams>(), decompressed_size: u20:31, literals_type: LiteralType::RAW},
];

const TEST_DATA: LiteralsData[11] = [
    // 0. RAW
    LiteralsData {data: LitData:0x1657_3465_A6DB_5DB0, length: LitLength:8, last: false},
    // 1. RLE
    LiteralsData {data: LitData:0x23, length: LitLength:1, last: false},
    // 2. RLE
    LiteralsData {data: LitData:0x35, length: LitLength:1, last: false},
    // 3. RAW
    LiteralsData {data: LitData:0x4CFB_41C6_7B60_5370, length: LitLength:8, last: false},
    LiteralsData {data: LitData:0x009B_0F9C_E1BA_A96D, length: LitLength:7, last: true},
    // 4. RLE
    LiteralsData {data: LitData:0x5A, length: LitLength:1, last: false},
    // 5. RLE
    LiteralsData {data: LitData:0xFF, length: LitLength:1, last: false},
    // 6. RAW
    LiteralsData {data: LitData:0x6094_3E96_1834_C247, length: LitLength:8, last: false},
    LiteralsData {data: LitData:0xBC02_D0E8_D728_9ABE, length: LitLength:8, last: false},
    LiteralsData {data: LitData:0xF864_C38B_E1FA_8D12, length: LitLength:8, last: false},
    LiteralsData {data: LitData:0x0019_63F1_CE21_C294, length: LitLength:7, last: true},
];

const TEST_BUF_CTRL: LiteralsBufferCtrl[5] = [
    LiteralsBufferCtrl {length: u32:11, last: false},
    LiteralsBufferCtrl {length: u32:2, last: false},
    LiteralsBufferCtrl {length: u32:16, last: false},
    LiteralsBufferCtrl {length: u32:11, last: false},
    LiteralsBufferCtrl {length: u32:32, last: true},
];

const TEST_EXPECTED_LITERALS: SequenceExecutorPacket[11] = [
    // ctrl 0
    SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::LITERAL,
        length: CopyOrMatchLength:8,
        content: CopyOrMatchContent:0x1657_3465_A6DB_5DB0,
        last: false
    },
    SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::LITERAL,
        length: CopyOrMatchLength:3,
        content: CopyOrMatchContent:0x23_2323,
        last: false
    },
    // ctrl 1
    SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::LITERAL,
        length: CopyOrMatchLength:2,
        content: CopyOrMatchContent:0x35_23,
        last: false
    },
    // ctrl 2
    SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::LITERAL,
        length: CopyOrMatchLength:8,
        content: CopyOrMatchContent:0xFB41_C67B_6053_7035,
        last: false
    },
    SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::LITERAL,
        length: CopyOrMatchLength:8,
        content: CopyOrMatchContent:0x9B0F_9CE1_BAA9_6D4C,
        last: true
    },
    // ctrl 3
    SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::LITERAL,
        length: CopyOrMatchLength:8,
        content: CopyOrMatchContent:0x5A5A_5A5A_5A5A_5A5A,
        last: false
    },
    SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::LITERAL,
        length: CopyOrMatchLength:3,
        content: CopyOrMatchContent:0x5A_5A5A,
        last: false
    },
    // ctrl 4
    SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::LITERAL,
        length: CopyOrMatchLength:8,
        content: CopyOrMatchContent:0x943E_9618_34C2_475A,
        last: false
    },
    SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::LITERAL,
        length: CopyOrMatchLength:8,
        content: CopyOrMatchContent:0x02D0_E8D7_289A_BE60,
        last: false
    },
    SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::LITERAL,
        length: CopyOrMatchLength:8,
        content: CopyOrMatchContent:0x64C3_8BE1_FA8D_12BC,
        last: false
    },
    SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::LITERAL,
        length: CopyOrMatchLength:8,
        content: CopyOrMatchContent:0x1963_F1CE_21C2_94F8,
        last: true
    },
];

#[test_proc]
proc LiteralsDecoder_test {
    terminator: chan<bool> out;

    literals_ctrl_s: chan<LiteralsPathCtrl> out;
    literals_data_s: chan<LiteralsData> out;
    literals_buf_ctrl_s: chan<LiteralsBufferCtrl> out;
    literals_r: chan<SequenceExecutorPacket> in;

    print_start_s: chan<()> out;
    print_finish_r: chan<()> in;

    config (terminator: chan<bool> out) {
        let (literals_ctrl_s, literals_ctrl_r) = chan<LiteralsPathCtrl, u32:1>("literals_ctrl");
        let (literals_data_s, literals_data_r) = chan<LiteralsData, u32:1>("literals_data");
        let (literals_buf_ctrl_s, literals_buf_ctrl_r) = chan<LiteralsBufferCtrl, u32:1>("literals_buf_ctrl");
        let (literals_s, literals_r) = chan<SequenceExecutorPacket, u32:1>("literals");

        let (print_start_s, print_start_r) = chan<()>("print_start");
        let (print_finish_s, print_finish_r) = chan<()>("print_finish");

        let (ram_rd_req_s,  ram_rd_req_r) = chan<TestReadReq>[literals_buffer::RAM_NUM]("ram_rd_req");
        let (ram_rd_resp_s, ram_rd_resp_r) = chan<TestReadResp>[literals_buffer::RAM_NUM]("ram_rd_resp");
        let (ram_wr_req_s,  ram_wr_req_r) = chan<TestWriteReq>[literals_buffer::RAM_NUM]("ram_wr_req");
        let (ram_wr_resp_s, ram_wr_resp_r) = chan<TestWriteResp>[literals_buffer::RAM_NUM]("ram_wr_resp");

        spawn LiteralsDecoder<TEST_HISTORY_BUFFER_SIZE_KB>(
            literals_ctrl_r, literals_data_r,
            literals_buf_ctrl_r, literals_s,
            ram_rd_req_s[0], ram_rd_req_s[1], ram_rd_req_s[2], ram_rd_req_s[3],
            ram_rd_req_s[4], ram_rd_req_s[5], ram_rd_req_s[6], ram_rd_req_s[7],
            ram_rd_resp_r[0], ram_rd_resp_r[1], ram_rd_resp_r[2], ram_rd_resp_r[3],
            ram_rd_resp_r[4], ram_rd_resp_r[5], ram_rd_resp_r[6], ram_rd_resp_r[7],
            ram_wr_req_s[0], ram_wr_req_s[1], ram_wr_req_s[2], ram_wr_req_s[3],
            ram_wr_req_s[4], ram_wr_req_s[5], ram_wr_req_s[6], ram_wr_req_s[7],
            ram_wr_resp_r[0], ram_wr_resp_r[1], ram_wr_resp_r[2], ram_wr_resp_r[3],
            ram_wr_resp_r[4], ram_wr_resp_r[5], ram_wr_resp_r[6], ram_wr_resp_r[7]
        );

        spawn ram_printer::RamPrinter<
            literals_buffer::RAM_DATA_WIDTH, TEST_RAM_SIZE, literals_buffer::RAM_NUM_PARTITIONS,
            TEST_RAM_ADDR_WIDTH, literals_buffer::RAM_NUM>
            (print_start_r, print_finish_s, ram_rd_req_s, ram_rd_resp_r);

        spawn ram::RamModel<
            literals_buffer::RAM_DATA_WIDTH, TEST_RAM_SIZE, literals_buffer::RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED>
            (ram_rd_req_r[0], ram_rd_resp_s[0], ram_wr_req_r[0], ram_wr_resp_s[0]);
        spawn ram::RamModel<
            literals_buffer::RAM_DATA_WIDTH, TEST_RAM_SIZE, literals_buffer::RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED>
            (ram_rd_req_r[1], ram_rd_resp_s[1], ram_wr_req_r[1], ram_wr_resp_s[1]);
        spawn ram::RamModel<
            literals_buffer::RAM_DATA_WIDTH, TEST_RAM_SIZE, literals_buffer::RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED>
            (ram_rd_req_r[2], ram_rd_resp_s[2], ram_wr_req_r[2], ram_wr_resp_s[2]);
        spawn ram::RamModel<
            literals_buffer::RAM_DATA_WIDTH, TEST_RAM_SIZE, literals_buffer::RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED>
            (ram_rd_req_r[3], ram_rd_resp_s[3], ram_wr_req_r[3], ram_wr_resp_s[3]);
        spawn ram::RamModel<
            literals_buffer::RAM_DATA_WIDTH, TEST_RAM_SIZE, literals_buffer::RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED>
            (ram_rd_req_r[4], ram_rd_resp_s[4], ram_wr_req_r[4], ram_wr_resp_s[4]);
        spawn ram::RamModel<
            literals_buffer::RAM_DATA_WIDTH, TEST_RAM_SIZE, literals_buffer::RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED>
            (ram_rd_req_r[5], ram_rd_resp_s[5], ram_wr_req_r[5], ram_wr_resp_s[5]);
        spawn ram::RamModel<
            literals_buffer::RAM_DATA_WIDTH, TEST_RAM_SIZE, literals_buffer::RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED>
            (ram_rd_req_r[6], ram_rd_resp_s[6], ram_wr_req_r[6], ram_wr_resp_s[6]);
        spawn ram::RamModel<
            literals_buffer::RAM_DATA_WIDTH, TEST_RAM_SIZE, literals_buffer::RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED>
            (ram_rd_req_r[7], ram_rd_resp_s[7], ram_wr_req_r[7], ram_wr_resp_s[7]);

        (
            terminator,
            literals_ctrl_s, literals_data_s,
            literals_buf_ctrl_s, literals_r,
            print_start_s, print_finish_r,
        )
    }

    init { }

    next (state: ()) {
        // send literals
        let tok = for ((i, test_data), tok): ((u32, LiteralsData), token) in enumerate(TEST_DATA) {
            let tok = send(tok, literals_data_s, test_data);
            trace_fmt!("Sent #{} literals data, {:#x}", i + u32:1, test_data);
            tok
        }(tok);

        // send ctrl
        let tok = for ((i, test_ctrl), tok): ((u32, LiteralsPathCtrl), token) in enumerate(TEST_CTRL) {
            let tok = send(tok, literals_ctrl_s, test_ctrl);
            trace_fmt!("Sent #{} literals ctrl, {:#x}", i + u32:1, test_ctrl);
            tok
        }(tok);

        // send buffer ctrl
        let tok = for ((i, test_buf_ctrl), tok): ((u32, LiteralsBufferCtrl), token) in enumerate(TEST_BUF_CTRL) {
            let tok = send(tok, literals_buf_ctrl_s, test_buf_ctrl);
            trace_fmt!("Sent #{} ctrl {:#x}", i + u32:1, test_buf_ctrl);
            tok
        }(tok);

        // receive and check packets
        let tok = for ((i, test_exp_literals), tok): ((u32, SequenceExecutorPacket), token) in enumerate(TEST_EXPECTED_LITERALS) {
            let (tok, literals) = recv(tok, literals_r);
            trace_fmt!("Received #{} literals packet {:#x}", i + u32:1, literals);
            assert_eq(test_exp_literals, literals);
            tok
        }(tok);

        // print RAM content
        let tok = send(tok, print_start_s, ());
        let (tok, _) = recv(tok, print_finish_r);

        send(tok, terminator, true);
    }
}

// TODO: Uncomment this test when fixed: https://github.com/google/xls/issues/1502
// type RamData = uN[literals_buffer::RAM_DATA_WIDTH];

// // Expected RAM content after each ctrl
// const TEST_EXPECTED_RAM_CONTENT = RamData[literals_buffer::RAM_NUM][10][7]:[
//     [
//         [RamData:0x016, RamData:0x057, RamData:0x034, RamData:0x065, RamData:0x0a6, RamData:0x0db, RamData:0x05d, RamData:0x0b0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//     ],
//     [
//         [RamData:0x016, RamData:0x057, RamData:0x034, RamData:0x065, RamData:0x0a6, RamData:0x0db, RamData:0x05d, RamData:0x0b0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:0x023, RamData:0x023, RamData:0x023, RamData:0x023],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//     ],
//     [
//         [RamData:0x016, RamData:0x057, RamData:0x034, RamData:0x065, RamData:0x0a6, RamData:0x0db, RamData:0x05d, RamData:0x0b0],
//         [RamData:  0x0, RamData:  0x0, RamData:0x035, RamData:0x035, RamData:0x023, RamData:0x023, RamData:0x023, RamData:0x023],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//     ],
//     [
//         [RamData:0x016, RamData:0x057, RamData:0x034, RamData:0x065, RamData:0x0a6, RamData:0x0db, RamData:0x05d, RamData:0x0b0],
//         [RamData:0x053, RamData:0x070, RamData:0x035, RamData:0x035, RamData:0x023, RamData:0x023, RamData:0x023, RamData:0x023],
//         [RamData:0x1a9, RamData:0x16d, RamData:0x04c, RamData:0x0fb, RamData:0x041, RamData:0x0c6, RamData:0x07b, RamData:0x060],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:0x19b, RamData:0x10f, RamData:0x19c, RamData:0x1e1, RamData:0x1ba],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//     ],
//     [
//         [RamData:0x016, RamData:0x057, RamData:0x034, RamData:0x065, RamData:0x0a6, RamData:0x0db, RamData:0x05d, RamData:0x0b0],
//         [RamData:0x053, RamData:0x070, RamData:0x035, RamData:0x035, RamData:0x023, RamData:0x023, RamData:0x023, RamData:0x023],
//         [RamData:0x1a9, RamData:0x16d, RamData:0x04c, RamData:0x0fb, RamData:0x041, RamData:0x0c6, RamData:0x07b, RamData:0x060],
//         [RamData:0x05a, RamData:0x05a, RamData:0x05a, RamData:0x19b, RamData:0x10f, RamData:0x19c, RamData:0x1e1, RamData:0x1ba],
//         [RamData:0x05a, RamData:0x05a, RamData:0x05a, RamData:0x05a, RamData:0x05a, RamData:0x05a, RamData:0x05a, RamData:0x05a],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:0x05a],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//     ],
//     [
//         [RamData:0x016, RamData:0x057, RamData:0x034, RamData:0x065, RamData:0x0a6, RamData:0x0db, RamData:0x05d, RamData:0x0b0],
//         [RamData:0x053, RamData:0x070, RamData:0x035, RamData:0x035, RamData:0x023, RamData:0x023, RamData:0x023, RamData:0x023],
//         [RamData:0x1a9, RamData:0x16d, RamData:0x04c, RamData:0x0fb, RamData:0x041, RamData:0x0c6, RamData:0x07b, RamData:0x060],
//         [RamData:0x05a, RamData:0x05a, RamData:0x05a, RamData:0x19b, RamData:0x10f, RamData:0x19c, RamData:0x1e1, RamData:0x1ba],
//         [RamData:0x05a, RamData:0x05a, RamData:0x05a, RamData:0x05a, RamData:0x05a, RamData:0x05a, RamData:0x05a, RamData:0x05a],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:0x05a],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//     ],
//     [
//         [RamData:0x016, RamData:0x057, RamData:0x034, RamData:0x065, RamData:0x0a6, RamData:0x0db, RamData:0x05d, RamData:0x0b0],
//         [RamData:0x053, RamData:0x070, RamData:0x035, RamData:0x035, RamData:0x023, RamData:0x023, RamData:0x023, RamData:0x023],
//         [RamData:0x1a9, RamData:0x16d, RamData:0x04c, RamData:0x0fb, RamData:0x041, RamData:0x0c6, RamData:0x07b, RamData:0x060],
//         [RamData:0x05a, RamData:0x05a, RamData:0x05a, RamData:0x19b, RamData:0x10f, RamData:0x19c, RamData:0x1e1, RamData:0x1ba],
//         [RamData:0x05a, RamData:0x05a, RamData:0x05a, RamData:0x05a, RamData:0x05a, RamData:0x05a, RamData:0x05a, RamData:0x05a],
//         [RamData:0x094, RamData:0x03e, RamData:0x096, RamData:0x018, RamData:0x034, RamData:0x0c2, RamData:0x047, RamData:0x05a],
//         [RamData:0x002, RamData:0x0d0, RamData:0x0e8, RamData:0x0d7, RamData:0x028, RamData:0x09a, RamData:0x0be, RamData:0x060],
//         [RamData:0x064, RamData:0x0c3, RamData:0x08b, RamData:0x0e1, RamData:0x0fa, RamData:0x08d, RamData:0x012, RamData:0x0bc],
//         [RamData:0x119, RamData:0x163, RamData:0x1f1, RamData:0x1ce, RamData:0x121, RamData:0x1c2, RamData:0x194, RamData:0x0f8],
//         [RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0, RamData:  0x0],
//     ],
// ];

// const CYCLES_PER_RAM_READ = u32:16;

// #[test_proc]
// proc LiteralsDecoderRamContent_test {
//     terminator: chan<bool> out;

//     literals_ctrl_s: chan<LiteralsPathCtrl> out;
//     literals_data_s: chan<LiteralsData> out;
//     literals_buf_ctrl_s: chan<LiteralsBufferCtrl> out;
//     literals_r: chan<SequenceExecutorPacket> in;

//     ram_rd_req_m0_s: chan<TestReadReq> out;
//     ram_rd_req_m1_s: chan<TestReadReq> out;
//     ram_rd_req_m2_s: chan<TestReadReq> out;
//     ram_rd_req_m3_s: chan<TestReadReq> out;
//     ram_rd_req_m4_s: chan<TestReadReq> out;
//     ram_rd_req_m5_s: chan<TestReadReq> out;
//     ram_rd_req_m6_s: chan<TestReadReq> out;
//     ram_rd_req_m7_s: chan<TestReadReq> out;

//     ram_rd_resp_m0_r: chan<TestReadResp> in;
//     ram_rd_resp_m1_r: chan<TestReadResp> in;
//     ram_rd_resp_m2_r: chan<TestReadResp> in;
//     ram_rd_resp_m3_r: chan<TestReadResp> in;
//     ram_rd_resp_m4_r: chan<TestReadResp> in;
//     ram_rd_resp_m5_r: chan<TestReadResp> in;
//     ram_rd_resp_m6_r: chan<TestReadResp> in;
//     ram_rd_resp_m7_r: chan<TestReadResp> in;

//     config (terminator: chan<bool> out) {
//         let (literals_ctrl_s, literals_ctrl_r) = chan<LiteralsPathCtrl>("literals_ctrl");
//         let (literals_data_s, literals_data_r) = chan<LiteralsData>("literals_data");
//         let (literals_buf_ctrl_s, literals_buf_ctrl_r) = chan<LiteralsBufferCtrl>("literals_buf_ctrl");
//         let (literals_s, literals_r) = chan<SequenceExecutorPacket>("literals");

//         let (ram_rd_req_s,  ram_rd_req_r) = chan<TestReadReq>[literals_buffer::RAM_NUM]("ram_rd_req");
//         let (ram_rd_resp_s, ram_rd_resp_r) = chan<TestReadResp>[literals_buffer::RAM_NUM]("ram_rd_resp");
//         let (ram_wr_req_s,  ram_wr_req_r) = chan<TestWriteReq>[literals_buffer::RAM_NUM]("ram_wr_req");
//         let (ram_wr_resp_s, ram_wr_resp_r) = chan<TestWriteResp>[literals_buffer::RAM_NUM]("ram_wr_resp");

//         spawn LiteralsDecoder<TEST_HISTORY_BUFFER_SIZE_KB>(
//             literals_ctrl_r, literals_data_r,
//             literals_buf_ctrl_r, literals_s,
//             ram_rd_req_s[0], ram_rd_req_s[1], ram_rd_req_s[2], ram_rd_req_s[3],
//             ram_rd_req_s[4], ram_rd_req_s[5], ram_rd_req_s[6], ram_rd_req_s[7],
//             ram_rd_resp_r[0], ram_rd_resp_r[1], ram_rd_resp_r[2], ram_rd_resp_r[3],
//             ram_rd_resp_r[4], ram_rd_resp_r[5], ram_rd_resp_r[6], ram_rd_resp_r[7],
//             ram_wr_req_s[0], ram_wr_req_s[1], ram_wr_req_s[2], ram_wr_req_s[3],
//             ram_wr_req_s[4], ram_wr_req_s[5], ram_wr_req_s[6], ram_wr_req_s[7],
//             ram_wr_resp_r[0], ram_wr_resp_r[1], ram_wr_resp_r[2], ram_wr_resp_r[3],
//             ram_wr_resp_r[4], ram_wr_resp_r[5], ram_wr_resp_r[6], ram_wr_resp_r[7]
//         );

//         spawn ram::RamModel<
//             literals_buffer::RAM_DATA_WIDTH, TEST_RAM_SIZE, literals_buffer::RAM_WORD_PARTITION_SIZE,
//             TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED>
//             (ram_rd_req_r[0], ram_rd_resp_s[0], ram_wr_req_r[0], ram_wr_resp_s[0]);
//         spawn ram::RamModel<
//             literals_buffer::RAM_DATA_WIDTH, TEST_RAM_SIZE, literals_buffer::RAM_WORD_PARTITION_SIZE,
//             TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED>
//             (ram_rd_req_r[1], ram_rd_resp_s[1], ram_wr_req_r[1], ram_wr_resp_s[1]);
//         spawn ram::RamModel<
//             literals_buffer::RAM_DATA_WIDTH, TEST_RAM_SIZE, literals_buffer::RAM_WORD_PARTITION_SIZE,
//             TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED>
//             (ram_rd_req_r[2], ram_rd_resp_s[2], ram_wr_req_r[2], ram_wr_resp_s[2]);
//         spawn ram::RamModel<
//             literals_buffer::RAM_DATA_WIDTH, TEST_RAM_SIZE, literals_buffer::RAM_WORD_PARTITION_SIZE,
//             TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED>
//             (ram_rd_req_r[3], ram_rd_resp_s[3], ram_wr_req_r[3], ram_wr_resp_s[3]);
//         spawn ram::RamModel<
//             literals_buffer::RAM_DATA_WIDTH, TEST_RAM_SIZE, literals_buffer::RAM_WORD_PARTITION_SIZE,
//             TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED>
//             (ram_rd_req_r[4], ram_rd_resp_s[4], ram_wr_req_r[4], ram_wr_resp_s[4]);
//         spawn ram::RamModel<
//             literals_buffer::RAM_DATA_WIDTH, TEST_RAM_SIZE, literals_buffer::RAM_WORD_PARTITION_SIZE,
//             TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED>
//             (ram_rd_req_r[5], ram_rd_resp_s[5], ram_wr_req_r[5], ram_wr_resp_s[5]);
//         spawn ram::RamModel<
//             literals_buffer::RAM_DATA_WIDTH, TEST_RAM_SIZE, literals_buffer::RAM_WORD_PARTITION_SIZE,
//             TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED>
//             (ram_rd_req_r[6], ram_rd_resp_s[6], ram_wr_req_r[6], ram_wr_resp_s[6]);
//         spawn ram::RamModel<
//             literals_buffer::RAM_DATA_WIDTH, TEST_RAM_SIZE, literals_buffer::RAM_WORD_PARTITION_SIZE,
//             TEST_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_RAM_INITIALIZED>
//             (ram_rd_req_r[7], ram_rd_resp_s[7], ram_wr_req_r[7], ram_wr_resp_s[7]);

//         (
//             terminator,
//             literals_ctrl_s, literals_data_s,
//             literals_buf_ctrl_s, literals_r,
//             ram_rd_req_s[0], ram_rd_req_s[1], ram_rd_req_s[2], ram_rd_req_s[3],
//             ram_rd_req_s[4], ram_rd_req_s[5], ram_rd_req_s[6], ram_rd_req_s[7],
//             ram_rd_resp_r[0], ram_rd_resp_r[1], ram_rd_resp_r[2], ram_rd_resp_r[3],
//             ram_rd_resp_r[4], ram_rd_resp_r[5], ram_rd_resp_r[6], ram_rd_resp_r[7],
//         )
//     }

//     init { u32:0 }

//     next (state: u32) {
//         // send literals
//         let ok = if (state == u32:0) {
//             for ((i, test_data), tok): ((u32, LiteralsData), token) in enumerate(TEST_DATA) {
//                 let tok = send(tok, literals_data_s, test_data);
//                 trace_fmt!("Sent #{} literals data, {:#x}", i + u32:1, test_data);
//                 tok
//             }(tok)
//         } else { tok };

//         // send ctrl and read RAM content
//         let tok = for ((i, test_ctrl), tok): ((u32, LiteralsPathCtrl), token) in enumerate(TEST_CTRL) {
//             if (state == i * CYCLES_PER_RAM_READ) {
//                 let tok = send(tok, literals_ctrl_s, test_ctrl);
//                 trace_fmt!("Sent #{} literals ctrl, {:#x}", i + u32:1, test_ctrl);
//                 tok
//             } else if (state == (i + u32:1) * CYCLES_PER_RAM_READ - u32:1) {
//                 for (addr, tok): (u32, token) in range(u32:0, u32:10) {
//                     let read_req = TestReadReq {
//                         addr: addr as uN[TEST_RAM_ADDR_WIDTH],
//                         mask: u1:1
//                     };

//                     let tok = send(tok, ram_rd_req_m0_s, read_req);
//                     let tok = send(tok, ram_rd_req_m1_s, read_req);
//                     let tok = send(tok, ram_rd_req_m2_s, read_req);
//                     let tok = send(tok, ram_rd_req_m3_s, read_req);
//                     let tok = send(tok, ram_rd_req_m4_s, read_req);
//                     let tok = send(tok, ram_rd_req_m5_s, read_req);
//                     let tok = send(tok, ram_rd_req_m6_s, read_req);
//                     let tok = send(tok, ram_rd_req_m7_s, read_req);

//                     let (tok, ram_rd_resp_m0) = recv(tok, ram_rd_resp_m0_r);
//                     let (tok, ram_rd_resp_m1) = recv(tok, ram_rd_resp_m1_r);
//                     let (tok, ram_rd_resp_m2) = recv(tok, ram_rd_resp_m2_r);
//                     let (tok, ram_rd_resp_m3) = recv(tok, ram_rd_resp_m3_r);
//                     let (tok, ram_rd_resp_m4) = recv(tok, ram_rd_resp_m4_r);
//                     let (tok, ram_rd_resp_m5) = recv(tok, ram_rd_resp_m5_r);
//                     let (tok, ram_rd_resp_m6) = recv(tok, ram_rd_resp_m6_r);
//                     let (tok, ram_rd_resp_m7) = recv(tok, ram_rd_resp_m7_r);
//                     trace_fmt!(
//                         "Received RAM read responses: [{:#x}, {:#x}, {:#x}, {:#x}, {:#x}, {:#x}, {:#x}, {:#x}]",
//                         ram_rd_resp_m7.data, ram_rd_resp_m6.data, ram_rd_resp_m5.data, ram_rd_resp_m4.data,
//                         ram_rd_resp_m3.data, ram_rd_resp_m2.data, ram_rd_resp_m1.data, ram_rd_resp_m0.data,
//                     );

//                     assert_eq(TEST_EXPECTED_RAM_CONTENT[i][addr][7], ram_rd_resp_m0.data);
//                     assert_eq(TEST_EXPECTED_RAM_CONTENT[i][addr][6], ram_rd_resp_m1.data);
//                     assert_eq(TEST_EXPECTED_RAM_CONTENT[i][addr][5], ram_rd_resp_m2.data);
//                     assert_eq(TEST_EXPECTED_RAM_CONTENT[i][addr][4], ram_rd_resp_m3.data);
//                     assert_eq(TEST_EXPECTED_RAM_CONTENT[i][addr][3], ram_rd_resp_m4.data);
//                     assert_eq(TEST_EXPECTED_RAM_CONTENT[i][addr][2], ram_rd_resp_m5.data);
//                     assert_eq(TEST_EXPECTED_RAM_CONTENT[i][addr][1], ram_rd_resp_m6.data);
//                     assert_eq(TEST_EXPECTED_RAM_CONTENT[i][addr][0], ram_rd_resp_m7.data);

//                     tok
//                 }(tok)
//             } else {
//                 tok
//             }
//         }(tok);

//         send_if(tok, terminator, state == array_size(TEST_CTRL) * CYCLES_PER_RAM_READ, true);

//         state + u32:1
//     }
// }
