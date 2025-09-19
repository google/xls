// Copyright 2025 The XLS Authors
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

import std;

import xls.examples.ram;
import xls.modules.zstd.zstd_enc;
import xls.modules.zstd.memory.axi;
import xls.modules.zstd.memory.axi_ram;
import xls.modules.zstd.memory.axi_ram_writer;
import xls.modules.zstd.mem_copy;
import xls.modules.zstd.memory.mem_writer;
import xls.modules.zstd.memory.mem_reader;
import xls.modules.zstd.common;

const TEST_ADDR_W = u32:32;
const TEST_AXI_ADDR_W = u32:32;
const TEST_AXI_DATA_W = u32:32;
const TEST_AXI_DEST_W = u32:8;
const TEST_AXI_ID_W = u32:4;

const TEST_WRITER_ID = u32:3;

const TEST_RAM_DATA_W = TEST_AXI_DATA_W;
const TEST_RAM_WORD_PARTITION_SIZE = u32:8;
const TEST_RAM_NUM_PARTITIONS = TEST_RAM_DATA_W / TEST_RAM_WORD_PARTITION_SIZE;
const TEST_RAM_SIMULTANEOUS_RW_BEHAVIOR = ram::SimultaneousReadWriteBehavior::READ_BEFORE_WRITE;
const TEST_RAM_INITIALIZED = true;
const TEST_RAM_ASSERT_VALID_READ = true;

const TEST_INPUT_SIZE = u32:100;
const TEST_INPUT_SIZE_BYTES = TEST_INPUT_SIZE * (TEST_RAM_DATA_W / u32:8);
const TEST_OUTPUT_SIZE = u32:103;

// The expected output data was generated with zstd as a single block.
// Let's test with a single block first, then TODO:
//   * generate new reference output with zstd, with more than one block
//   * change the block size here
const TEST_MAX_BLOCK_SIZE = u32:0x1000;

type ZstdEncodeRespStatus = zstd_enc::ZstdEncodeRespStatus;

type TestReadReq = ram::ReadReq<TEST_ADDR_W, TEST_RAM_NUM_PARTITIONS>;
type TestWriteReq = ram::WriteReq<TEST_ADDR_W, TEST_RAM_DATA_W, TEST_RAM_NUM_PARTITIONS>;
type TestReadResp = ram::ReadResp<TEST_RAM_DATA_W>;
type TestWriteResp = ram::WriteResp;
type TestMemReaderReq = mem_reader::MemReaderReq<TEST_ADDR_W>;
type TestMemReaderResp = mem_reader::MemReaderResp<TEST_RAM_DATA_W, TEST_ADDR_W>;
type TestMemWriterReq = mem_writer::MemWriterReq<TEST_ADDR_W>;
type TestMemWriterData = mem_writer::MemWriterDataPacket<TEST_RAM_DATA_W, TEST_ADDR_W>;
type TestMemWriterResp = mem_writer::MemWriterResp;
type TestMemWriterStatus = mem_writer::MemWriterRespStatus;

type TestZstdEncodeReq = zstd_enc::ZstdEncodeReq<TEST_ADDR_W, TEST_RAM_DATA_W>;
type TestZstdEncodeResp = zstd_enc::ZstdEncodeResp;

const TEST_INPUT_DATA = u32[TEST_INPUT_SIZE]:[
    u32:0xD945_50A5, u32:0xA20C_D8D3, u32:0xB0BE_D046, u32:0xF83C_6D26, u32:0xFAE4_B0C4,
    u32:0x9A78_91C4, u32:0xFDA0_9B1E, u32:0x5E66_D76D, u32:0xCB7D_76CB, u32:0x4033_5F2F,
    u32:0x2128_9B0B, u32:0xD263_365F, u32:0xD989_DD81, u32:0xE4CB_45C9, u32:0x0425_06B6,
    u32:0x5D31_107C, u32:0x2282_7A67, u32:0xCAC7_0C94, u32:0x23A9_5FD8, u32:0x6122_BBC3,
    u32:0x1F99_F3D0, u32:0xA70C_FB34, u32:0x3812_5EF2, u32:0x9157_61BC, u32:0x171A_C1B1,

    u32:0xDE6F_1B08, u32:0x420D_F1AF, u32:0xAEE9_F51B, u32:0xB31E_E3A3, u32:0x66AC_09D6,
    u32:0x18E9_9703, u32:0xEE87_1E7A, u32:0xB63D_47DE, u32:0x59BF_4F52, u32:0x94D8_5636,
    u32:0x2B81_34EE, u32:0x6711_9968, u32:0xFB2B_F8CB, u32:0x173F_CB1B, u32:0xFB94_3A67,
    u32:0xF40B_714F, u32:0x383B_82FE, u32:0xA692_055E, u32:0x58A6_2110, u32:0x0185_B5E0,
    u32:0x9DF0_9C22, u32:0x54CA_DB57, u32:0xC626_097F, u32:0xEA04_3110, u32:0xF11C_4D36,

    u32:0xB8CC_FAB0, u32:0x7801_3B20, u32:0x8189_BF9C, u32:0xE380_A505, u32:0x4672_AE34,
    u32:0x1CD5_1B3A, u32:0x5F95_EE9E, u32:0xBC5C_9931, u32:0xBCE6_50D2, u32:0xC10D_0544,
    u32:0x5AB4_DEA1, u32:0x5E20_3394, u32:0x7FDA_0CA1, u32:0x6FEC_112E, u32:0x107A_2F81,
    u32:0x86CA_4491, u32:0xEA68_0EB7, u32:0x50F1_AA22, u32:0x3F47_F2CA, u32:0xE407_92F7,
    u32:0xF35C_EEE0, u32:0x1D6B_E819, u32:0x3FA7_05FA, u32:0x08BB_A499, u32:0x7C0C_4812,

    u32:0xF5A5_3D5C, u32:0x079A_BE16, u32:0xACA1_F84B, u32:0x4D2B_9402, u32:0x45B1_28FD,
    u32:0x2C7C_CBA5, u32:0x6874_FC32, u32:0x95A0_8288, u32:0xFB13_E707, u32:0x61F9_2FEF,
    u32:0xF6E3_DAFC, u32:0xDBA0_0A80, u32:0xBB84_831B, u32:0xAD63_2520, u32:0xEFB3_D817,
    u32:0xD190_C435, u32:0x9064_1E4F, u32:0x0839_3D28, u32:0x1C07_874C, u32:0xBBEB_D633,
    u32:0xB0A9_C751, u32:0x83B9_A340, u32:0x028A_FF8A, u32:0xB4ED_EE5C, u32:0xD700_BD9C,
];

// zstd -0 --no-check input.bin -o output.zst
// xxd -e -c4 output.zst
const TEST_EXPECTED_SIZE = TEST_OUTPUT_SIZE;
const TEST_EXPECTED = u32[TEST_EXPECTED_SIZE]:[
    // FH - frame header,   BH - block header,  BB - block content
    // Magic_Number            BHFH FHFH            BBBB BHBH            BBBB BBBB            BBBB ...
    u32:0xFD2F_B528,     u32:0x8100_9060,     u32:0x50A5_000C,     u32:0xD8D3_D945,     u32:0xD046_A20C,
    u32:0x6D26_B0BE,     u32:0xB0C4_F83C,     u32:0x91C4_FAE4,     u32:0x9B1E_9A78,     u32:0xD76D_FDA0,
    u32:0x76CB_5E66,     u32:0x5F2F_CB7D,     u32:0x9B0B_4033,     u32:0x365F_2128,     u32:0xDD81_D263,
    u32:0x45C9_D989,     u32:0x06B6_E4CB,     u32:0x107C_0425,     u32:0x7A67_5D31,     u32:0x0C94_2282,
    u32:0x5FD8_CAC7,     u32:0xBBC3_23A9,     u32:0xF3D0_6122,     u32:0xFB34_1F99,     u32:0x5EF2_A70C,
    u32:0x61BC_3812,     u32:0xC1B1_9157,     u32:0x1B08_171A,     u32:0xF1AF_DE6F,     u32:0xF51B_420D,
    u32:0xE3A3_AEE9,     u32:0x09D6_B31E,     u32:0x9703_66AC,     u32:0x1E7A_18E9,     u32:0x47DE_EE87,
    u32:0x4F52_B63D,     u32:0x5636_59BF,     u32:0x34EE_94D8,     u32:0x9968_2B81,     u32:0xF8CB_6711,
    u32:0xCB1B_FB2B,     u32:0x3A67_173F,     u32:0x714F_FB94,     u32:0x82FE_F40B,     u32:0x055E_383B,
    u32:0x2110_A692,     u32:0xB5E0_58A6,     u32:0x9C22_0185,     u32:0xDB57_9DF0,     u32:0x097F_54CA,
    u32:0x3110_C626,     u32:0x4D36_EA04,     u32:0xFAB0_F11C,     u32:0x3B20_B8CC,     u32:0xBF9C_7801,
    u32:0xA505_8189,     u32:0xAE34_E380,     u32:0x1B3A_4672,     u32:0xEE9E_1CD5,     u32:0x9931_5F95,
    u32:0x50D2_BC5C,     u32:0x0544_BCE6,     u32:0xDEA1_C10D,     u32:0x3394_5AB4,     u32:0x0CA1_5E20,
    u32:0x112E_7FDA,     u32:0x2F81_6FEC,     u32:0x4491_107A,     u32:0x0EB7_86CA,     u32:0xAA22_EA68,
    u32:0xF2CA_50F1,     u32:0x92F7_3F47,     u32:0xEEE0_E407,     u32:0xE819_F35C,     u32:0x05FA_1D6B,
    u32:0xA499_3FA7,     u32:0x4812_08BB,     u32:0x3D5C_7C0C,     u32:0xBE16_F5A5,     u32:0xF84B_079A,
    u32:0x9402_ACA1,     u32:0x28FD_4D2B,     u32:0xCBA5_45B1,     u32:0xFC32_2C7C,     u32:0x8288_6874,
    u32:0xE707_95A0,     u32:0x2FEF_FB13,     u32:0xDAFC_61F9,     u32:0x0A80_F6E3,     u32:0x831B_DBA0,
    u32:0x2520_BB84,     u32:0xD817_AD63,     u32:0xC435_EFB3,     u32:0x1E4F_D190,     u32:0x3D28_9064,
    u32:0x874C_0839,     u32:0xD633_1C07,     u32:0xC751_BBEB,     u32:0xA340_B0A9,     u32:0xFF8A_83B9,
    u32:0xee5c_028a,     u32:0xbd9c_b4ed,     u32:0x0000_d700
];

// touch empty.txt
// zstd --no-check empty.txt -o empty.zst
// xxd -e empty.zst
const TEST_EMPTY_EXPECTED_SIZE = u32:3;
const TEST_EMPTY_EXPECTED = u32[TEST_EMPTY_EXPECTED_SIZE]:[
    // Magic_Number      //      FH FHFH
    u32:0xFD2F_B528,     u32:0x0001_0020,   u32:0x0
];

const TEST_RAM_SIZE = TEST_EXPECTED_SIZE * u32:2;


#[test_proc]
proc ZstdEncoderTestEmpty {
    type AxiAr = axi::AxiAr<TEST_AXI_ADDR_W, TEST_AXI_ID_W>;
    type AxiR = axi::AxiR<TEST_AXI_DATA_W, TEST_AXI_ID_W>;
    type AxiAw = axi::AxiAw<TEST_AXI_ADDR_W, TEST_AXI_ID_W>;
    type AxiW = axi::AxiW<TEST_AXI_DATA_W, TEST_RAM_NUM_PARTITIONS>;
    type AxiB = axi::AxiB<TEST_AXI_ID_W>;

    input_wr_req_s: chan<TestWriteReq> out;
    input_wr_resp_r: chan<TestWriteResp> in;
    output_rd_req_s: chan<TestReadReq> out;
    output_rd_resp_r: chan<TestReadResp> in;
    enc_req_s: chan<TestZstdEncodeReq> out;
    enc_resp_r: chan<TestZstdEncodeResp> in;
    terminator: chan<bool> out;

    init {  }

    config(terminator: chan<bool> out) {
        // IO for Encoder <-> test
        let (enc_req_s, enc_req_r) = chan<TestZstdEncodeReq>("enc_req");
        let (enc_resp_s, enc_resp_r) = chan<TestZstdEncodeResp>("enc_resp");

        // IO for input RAM
        let (input_rd_req_s, input_rd_req_r) = chan<TestReadReq>("input_rd_req");
        let (input_rd_resp_s, input_rd_resp_r) = chan<TestReadResp>("input_rd_resp");
        let (input_wr_req_s, input_wr_req_r) = chan<TestWriteReq>("input_wr_req");
        let (input_wr_resp_s, input_wr_resp_r) = chan<TestWriteResp>("input_wr_resp");

        spawn ram::RamModel<
            TEST_RAM_DATA_W, TEST_RAM_SIZE, TEST_RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_RW_BEHAVIOR, TEST_RAM_INITIALIZED, TEST_RAM_ASSERT_VALID_READ, TEST_AXI_ADDR_W
        >(
            input_rd_req_r, input_rd_resp_s,
            input_wr_req_r, input_wr_resp_s
        );

        // IO for AxiRamReader <-> MemReader
        let (input_axi_ar_s, input_axi_ar_r) = chan<AxiAr>("input_axi_ar");
        let (input_axi_r_s, input_axi_r_r) = chan<AxiR>("input_axi_r");

        spawn axi_ram::AxiRamReader<
            TEST_AXI_ADDR_W, TEST_AXI_DATA_W, TEST_AXI_DEST_W, TEST_AXI_ID_W, TEST_RAM_SIZE
        >(
            input_axi_ar_r, input_axi_r_s,
            input_rd_req_s, input_rd_resp_r
        );

        // IO for MemReader <-> Encoder
        let (mem_rd_req_s, mem_rd_req_r) = chan<TestMemReaderReq>("mem_rd_req");
        let (mem_rd_resp_s, mem_rd_resp_r) = chan<TestMemReaderResp>("mem_rd_resp");

        spawn mem_reader::MemReader<TEST_AXI_DATA_W, TEST_AXI_ADDR_W, TEST_AXI_DEST_W, TEST_AXI_ID_W>(
            mem_rd_req_r, mem_rd_resp_s,
            input_axi_ar_s, input_axi_r_r
        );

        // IO for output RAM
        let (output_rd_req_s, output_rd_req_r) = chan<TestReadReq>("output_rd_req");
        let (output_rd_resp_s, output_rd_resp_r) = chan<TestReadResp>("output_rd_resp");
        let (output_wr_req_s, output_wr_req_r) = chan<TestWriteReq>("output_wr_req");
        let (output_wr_resp_s, output_wr_resp_r) = chan<TestWriteResp>("output_wr_resp");

        spawn ram::RamModel<
            TEST_RAM_DATA_W, TEST_RAM_SIZE, TEST_RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_RW_BEHAVIOR, TEST_RAM_INITIALIZED,
            TEST_RAM_ASSERT_VALID_READ, TEST_ADDR_W
        >(
            output_rd_req_r, output_rd_resp_s, output_wr_req_r, output_wr_resp_s
        );

        // IO for AxiRamWriter <-> MemWriter
        let (output_axi_aw_s, output_axi_aw_r) = chan<AxiAw>("output_axi_aw");
        let (output_axi_w_s, output_axi_w_r) = chan<AxiW>("output_axi_w");
        let (output_axi_b_s, output_axi_b_r) = chan<AxiB>("output_axi_b");

        spawn axi_ram_writer::AxiRamWriter<
            TEST_AXI_ADDR_W, TEST_AXI_DATA_W, TEST_AXI_ID_W, TEST_RAM_SIZE, TEST_ADDR_W, TEST_RAM_NUM_PARTITIONS,
        >(
            output_axi_aw_r, output_axi_w_r, output_axi_b_s,
            output_wr_req_s, output_wr_resp_r
        );

        // IO for MemWriter <-> Encoder
        let (mem_wr_req_s, mem_wr_req_r) = chan<TestMemWriterReq>("mem_wr_req");
        let (mem_wr_data_s, mem_wr_data_r) = chan<TestMemWriterData>("mem_wr_data");
        let (mem_wr_resp_s, mem_wr_resp_r) = chan<TestMemWriterResp>("mem_wr_resp");

        spawn mem_writer::MemWriter<
            TEST_AXI_ADDR_W, TEST_AXI_DATA_W, TEST_AXI_DEST_W, TEST_AXI_ID_W, TEST_WRITER_ID
        >(
            mem_wr_req_r, mem_wr_data_r,
            output_axi_aw_s, output_axi_w_s, output_axi_b_r,
            mem_wr_resp_s
        );

        spawn zstd_enc::ZstdEncoder<TEST_ADDR_W, TEST_RAM_DATA_W>
        (enc_req_r, enc_resp_s, mem_rd_req_s, mem_rd_resp_r, mem_wr_req_s, mem_wr_data_s, mem_wr_resp_r);

        (input_wr_req_s, input_wr_resp_r, output_rd_req_s, output_rd_resp_r, enc_req_s, enc_resp_r, terminator)
    }

    next(state: ()) {
        type Addr = bits[TEST_ADDR_W];

        // write ZEROS to RAM
        let tok = for ((i, _data), tok): ((u32, u32), token) in enumerate(TEST_INPUT_DATA) {
            let tok = send(tok, input_wr_req_s, TestWriteReq {
                addr: i as Addr,
                data: u32:0,
                mask: !bits[TEST_RAM_NUM_PARTITIONS]:0,
            });
            let (tok, _) = recv(tok, input_wr_resp_r);
            tok
        }(join());

        trace_fmt!("Request: encode data of size 0");
        let tok = send(tok, enc_req_s, TestZstdEncodeReq{
            input_offset: uN[TEST_ADDR_W]:0,
            data_size: uN[TEST_RAM_DATA_W]:0,
            output_offset: uN[TEST_ADDR_W]:0,
            max_block_size: TEST_MAX_BLOCK_SIZE,
        });
        let (tok, resp) = recv(tok, enc_resp_r);
        assert_eq(resp.status, ZstdEncodeRespStatus::OK);

        // read state of output RAM
        let tok = for ((i, expected_val), tok): ((u32, u32), token) in enumerate(TEST_EMPTY_EXPECTED) {
            let tok = send(tok, output_rd_req_s, TestReadReq {
                addr: i as Addr,
                mask: !bits[TEST_RAM_NUM_PARTITIONS]:0,
            });
            let (tok, data) = recv(tok, output_rd_resp_r);

            if (data.data != expected_val) {
                trace_fmt!("at index {} the expected value is {:#x}, the outcome is {:#x}", i, expected_val, data.data);
            } else { };

            assert_eq([i, data.data], [i, expected_val]);
            tok
        }(join());
        send(join(), terminator, true);
    }
}

#[test_proc]
proc ZstdEncoderTest {
    type AxiAr = axi::AxiAr<TEST_AXI_ADDR_W, TEST_AXI_ID_W>;
    type AxiR = axi::AxiR<TEST_AXI_DATA_W, TEST_AXI_ID_W>;
    type AxiAw = axi::AxiAw<TEST_AXI_ADDR_W, TEST_AXI_ID_W>;
    type AxiW = axi::AxiW<TEST_AXI_DATA_W, TEST_RAM_NUM_PARTITIONS>;
    type AxiB = axi::AxiB<TEST_AXI_ID_W>;

    input_wr_req_s: chan<TestWriteReq> out;
    input_wr_resp_r: chan<TestWriteResp> in;
    output_rd_req_s: chan<TestReadReq> out;
    output_rd_resp_r: chan<TestReadResp> in;
    enc_req_s: chan<TestZstdEncodeReq> out;
    enc_resp_r: chan<TestZstdEncodeResp> in;
    terminator: chan<bool> out;

    init {  }

    config(terminator: chan<bool> out) {
        // IO for Encoder <-> test
        let (enc_req_s, enc_req_r) = chan<TestZstdEncodeReq>("enc_req");
        let (enc_resp_s, enc_resp_r) = chan<TestZstdEncodeResp>("enc_resp");

        // IO for input RAM
        let (input_rd_req_s, input_rd_req_r) = chan<TestReadReq>("input_rd_req");
        let (input_rd_resp_s, input_rd_resp_r) = chan<TestReadResp>("input_rd_resp");
        let (input_wr_req_s, input_wr_req_r) = chan<TestWriteReq>("input_wr_req");
        let (input_wr_resp_s, input_wr_resp_r) = chan<TestWriteResp>("input_wr_resp");

        spawn ram::RamModel<
            TEST_RAM_DATA_W, TEST_RAM_SIZE, TEST_RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_RW_BEHAVIOR, TEST_RAM_INITIALIZED, TEST_RAM_ASSERT_VALID_READ, TEST_AXI_ADDR_W
        >(
            input_rd_req_r, input_rd_resp_s,
            input_wr_req_r, input_wr_resp_s
        );

        // IO for AxiRamReader <-> MemReader
        let (input_axi_ar_s, input_axi_ar_r) = chan<AxiAr>("input_axi_ar");
        let (input_axi_r_s, input_axi_r_r) = chan<AxiR>("input_axi_r");

        spawn axi_ram::AxiRamReader<
            TEST_AXI_ADDR_W, TEST_AXI_DATA_W, TEST_AXI_DEST_W, TEST_AXI_ID_W, TEST_RAM_SIZE
        >(
            input_axi_ar_r, input_axi_r_s,
            input_rd_req_s, input_rd_resp_r
        );

        // IO for MemReader <-> Encoder
        let (mem_rd_req_s, mem_rd_req_r) = chan<TestMemReaderReq>("mem_rd_req");
        let (mem_rd_resp_s, mem_rd_resp_r) = chan<TestMemReaderResp>("mem_rd_resp");

        spawn mem_reader::MemReader<TEST_AXI_DATA_W, TEST_AXI_ADDR_W, TEST_AXI_DEST_W, TEST_AXI_ID_W>(
            mem_rd_req_r, mem_rd_resp_s,
            input_axi_ar_s, input_axi_r_r
        );

        // IO for output RAM
        let (output_rd_req_s, output_rd_req_r) = chan<TestReadReq>("output_rd_req");
        let (output_rd_resp_s, output_rd_resp_r) = chan<TestReadResp>("output_rd_resp");
        let (output_wr_req_s, output_wr_req_r) = chan<TestWriteReq>("output_wr_req");
        let (output_wr_resp_s, output_wr_resp_r) = chan<TestWriteResp>("output_wr_resp");

        spawn ram::RamModel<
            TEST_RAM_DATA_W, TEST_RAM_SIZE, TEST_RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_RW_BEHAVIOR, TEST_RAM_INITIALIZED,
            TEST_RAM_ASSERT_VALID_READ, TEST_ADDR_W
        >(
            output_rd_req_r, output_rd_resp_s,
            output_wr_req_r, output_wr_resp_s
        );

        // IO for AxiRamWriter <-> MemWriter
        let (output_axi_aw_s, output_axi_aw_r) = chan<AxiAw>("output_axi_aw");
        let (output_axi_w_s, output_axi_w_r) = chan<AxiW>("output_axi_w");
        let (output_axi_b_s, output_axi_b_r) = chan<AxiB>("output_axi_b");

        spawn axi_ram_writer::AxiRamWriter<
            TEST_AXI_ADDR_W, TEST_AXI_DATA_W, TEST_AXI_ID_W, TEST_RAM_SIZE, TEST_ADDR_W, TEST_RAM_NUM_PARTITIONS,
        >(
            output_axi_aw_r, output_axi_w_r, output_axi_b_s,
            output_wr_req_s, output_wr_resp_r
        );

        // IO for MemWriter <-> Encoder
        let (mem_wr_req_s, mem_wr_req_r) = chan<TestMemWriterReq>("mem_wr_req");
        let (mem_wr_data_s, mem_wr_data_r) = chan<TestMemWriterData>("mem_wr_data");
        let (mem_wr_resp_s, mem_wr_resp_r) = chan<TestMemWriterResp>("mem_wr_resp");

        spawn mem_writer::MemWriter<
            TEST_AXI_ADDR_W, TEST_AXI_DATA_W, TEST_AXI_DEST_W, TEST_AXI_ID_W, TEST_WRITER_ID
        >(
            mem_wr_req_r, mem_wr_data_r,
            output_axi_aw_s, output_axi_w_s, output_axi_b_r,
            mem_wr_resp_s
        );

        spawn zstd_enc::ZstdEncoder<TEST_ADDR_W, TEST_RAM_DATA_W>
        (
            enc_req_r, enc_resp_s,
            mem_rd_req_s, mem_rd_resp_r,
            mem_wr_req_s, mem_wr_data_s, mem_wr_resp_r
        );

        (
            input_wr_req_s, input_wr_resp_r,
            output_rd_req_s, output_rd_resp_r,
            enc_req_s, enc_resp_r,
            terminator
        )
    }

    next(state: ()) {
        type Addr = bits[TEST_ADDR_W];

        // write input data to RAM
        let tok = for ((i, data), tok): ((u32, u32), token) in enumerate(TEST_INPUT_DATA) {
            let tok = send(tok, input_wr_req_s, TestWriteReq {
                addr: i as Addr,
                data: data,
                mask: !bits[TEST_RAM_NUM_PARTITIONS]:0,
            });
            let (tok, _) = recv(tok, input_wr_resp_r);
            tok
        }(join());

        trace_fmt!("Request: encode data of size: {:#x} bytes", TEST_INPUT_SIZE_BYTES);
        // send request to encoder
        let tok = send(tok, enc_req_s, TestZstdEncodeReq{
            input_offset: uN[TEST_ADDR_W]:0,
            data_size: TEST_INPUT_SIZE_BYTES as uN[TEST_RAM_DATA_W],
            output_offset: uN[TEST_ADDR_W]:0,
            max_block_size: TEST_MAX_BLOCK_SIZE,
        });
        let (tok, resp) = recv(tok, enc_resp_r);
        assert_eq(resp.status, ZstdEncodeRespStatus::OK);

        // read state of output RAM
        let tok = for ((i, expected_val), tok): ((u32, u32), token) in enumerate(TEST_EXPECTED) {
            let tok = send(tok, output_rd_req_s, TestReadReq {
                addr: i as Addr,
                mask: !bits[TEST_RAM_NUM_PARTITIONS]:0,
            });
            let (tok, data) = recv(tok, output_rd_resp_r);

            if (data.data != expected_val) {
                trace_fmt!("at index {} the expected value is {:#x}, the outcome is {:#x}", i, expected_val, data.data);
            } else { };

            assert_eq(data.data, expected_val);
            tok
        }(join());

        send(join(), terminator, true);
    }
}

// Faulty MemReader used to test reponses of the Encoder in case memory read fails.
proc MemReaderFaultResponder {
    type Req = mem_reader::MemReaderReq<TEST_AXI_ADDR_W>;
    type Resp = mem_reader::MemReaderResp<TEST_AXI_DATA_W, TEST_AXI_ADDR_W>;
    type Status = mem_reader::MemReaderStatus;

    req_r: chan<Req> in;
    resp_s: chan<Resp> out;

    config(
        req_r: chan<Req> in,
        resp_s: chan<Resp> out,
    ) {
        (req_r, resp_s)
    }

    init {}
    next(state: ()) {
        let (tok, _req) = recv(join(), req_r);
        let tok = send(tok, resp_s, Resp {
            status: Status::ERROR,
            data: uN[TEST_AXI_DATA_W]:0,
            length: uN[TEST_AXI_ADDR_W]:0,
            last: true,
        });
    }
}

#[test_proc]
proc ZstdEncoderReadFaultTest {
    type AxiAw = axi::AxiAw<TEST_AXI_ADDR_W, TEST_AXI_ID_W>;
    type AxiW = axi::AxiW<TEST_AXI_DATA_W, TEST_RAM_NUM_PARTITIONS>;
    type AxiB = axi::AxiB<TEST_AXI_ID_W>;

    output_rd_req_s: chan<TestReadReq> out;
    output_rd_resp_r: chan<TestReadResp> in;
    enc_req_s: chan<TestZstdEncodeReq> out;
    enc_resp_r: chan<TestZstdEncodeResp> in;
    terminator: chan<bool> out;

    init {  }

    config(terminator: chan<bool> out) {

        // IO for Encoder <-> test
        let (enc_req_s, enc_req_r) = chan<TestZstdEncodeReq>("enc_req");
        let (enc_resp_s, enc_resp_r) = chan<TestZstdEncodeResp>("enc_resp");

        // IO for MemReader <-> Encoder
        let (mem_rd_req_s, mem_rd_req_r) = chan<TestMemReaderReq>("mem_rd_req");
        let (mem_rd_resp_s, mem_rd_resp_r) = chan<TestMemReaderResp>("mem_rd_resp");

        spawn MemReaderFaultResponder(
            mem_rd_req_r, mem_rd_resp_s,
        );

        // IO for output RAM
        let (output_rd_req_s, output_rd_req_r) = chan<TestReadReq>("output_rd_req");
        let (output_rd_resp_s, output_rd_resp_r) = chan<TestReadResp>("output_rd_resp");
        let (output_wr_req_s, output_wr_req_r) = chan<TestWriteReq>("output_wr_req");
        let (output_wr_resp_s, output_wr_resp_r) = chan<TestWriteResp>("output_wr_resp");

        spawn ram::RamModel<
            TEST_RAM_DATA_W, TEST_RAM_SIZE, TEST_RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_RW_BEHAVIOR, TEST_RAM_INITIALIZED,
            TEST_RAM_ASSERT_VALID_READ, TEST_ADDR_W
        >(
            output_rd_req_r, output_rd_resp_s,
            output_wr_req_r, output_wr_resp_s
        );

        // IO for AxiRamWriter <-> MemWriter
        let (output_axi_aw_s, output_axi_aw_r) = chan<AxiAw>("output_axi_aw");
        let (output_axi_w_s, output_axi_w_r) = chan<AxiW>("output_axi_w");
        let (output_axi_b_s, output_axi_b_r) = chan<AxiB>("output_axi_b");

        spawn axi_ram_writer::AxiRamWriter<
            TEST_AXI_ADDR_W, TEST_AXI_DATA_W, TEST_AXI_ID_W, TEST_RAM_SIZE, TEST_ADDR_W, TEST_RAM_NUM_PARTITIONS,
        >(
            output_axi_aw_r, output_axi_w_r, output_axi_b_s,
            output_wr_req_s, output_wr_resp_r
        );

        // IO for MemWriter <-> Encoder
        let (mem_wr_req_s, mem_wr_req_r) = chan<TestMemWriterReq>("mem_wr_req");
        let (mem_wr_data_s, mem_wr_data_r) = chan<TestMemWriterData>("mem_wr_data");
        let (mem_wr_resp_s, mem_wr_resp_r) = chan<TestMemWriterResp>("mem_wr_resp");

        spawn mem_writer::MemWriter<
            TEST_AXI_ADDR_W, TEST_AXI_DATA_W, TEST_AXI_DEST_W, TEST_AXI_ID_W, TEST_WRITER_ID
        >(
            mem_wr_req_r, mem_wr_data_r,
            output_axi_aw_s, output_axi_w_s, output_axi_b_r,
            mem_wr_resp_s
        );

        spawn zstd_enc::ZstdEncoder<TEST_ADDR_W, TEST_RAM_DATA_W>
        (enc_req_r, enc_resp_s, mem_rd_req_s, mem_rd_resp_r, mem_wr_req_s, mem_wr_data_s, mem_wr_resp_r);

        (
            output_rd_req_s, output_rd_resp_r,
            enc_req_s, enc_resp_r,
            terminator
        )
    }

    next(state: ()) {
        type Addr = bits[TEST_ADDR_W];

        // send request to encoder
        let tok = send(join(), enc_req_s, TestZstdEncodeReq{
            input_offset: uN[TEST_ADDR_W]:0,
            data_size: TEST_INPUT_SIZE as uN[TEST_RAM_DATA_W],
            output_offset: uN[TEST_ADDR_W]:0,
            max_block_size: TEST_MAX_BLOCK_SIZE,
        });
        let (tok, resp) = recv(tok, enc_resp_r);

        // expect error due to the faulty (on purpose) MemReader
        assert_eq(resp.status, ZstdEncodeRespStatus::ERROR);

        send(join(), terminator, true);
    }
}

// Faulty MemWriter used to test reponses of the Encoder in case memory write fails.
proc MemWriterFaultResponder {
    req_r: chan<TestMemWriterReq> in;
    data_r: chan<TestMemWriterData> in;
    resp_s: chan<TestMemWriterResp> out;

    config(
        req_r: chan<TestMemWriterReq> in,
        data_r: chan<TestMemWriterData> in,
        resp_s: chan<TestMemWriterResp> out,
    ) {
        (req_r, data_r, resp_s)
    }

    init {}
    next(state: ()) {
        let (tok, _req) = recv(join(), req_r);
        let (tok, _data) = recv(tok, data_r);
        let tok = send(tok, resp_s, TestMemWriterResp {
            status: TestMemWriterStatus::ERROR,
        });
    }
}

#[test_proc]
proc ZstdEncoderWriteFaultTest {
    type AxiAr = axi::AxiAr<TEST_AXI_ADDR_W, TEST_AXI_ID_W>;
    type AxiR = axi::AxiR<TEST_AXI_DATA_W, TEST_AXI_ID_W>;
    type AxiAw = axi::AxiAw<TEST_AXI_ADDR_W, TEST_AXI_ID_W>;
    type AxiW = axi::AxiW<TEST_AXI_DATA_W, TEST_RAM_NUM_PARTITIONS>;
    type AxiB = axi::AxiB<TEST_AXI_ID_W>;

    input_wr_req_s: chan<TestWriteReq> out;
    input_wr_resp_r: chan<TestWriteResp> in;
    enc_req_s: chan<TestZstdEncodeReq> out;
    enc_resp_r: chan<TestZstdEncodeResp> in;
    terminator: chan<bool> out;

    init {  }

    config(terminator: chan<bool> out) {
        // IO for Encoder <-> test
        let (enc_req_s, enc_req_r) = chan<TestZstdEncodeReq>("enc_req");
        let (enc_resp_s, enc_resp_r) = chan<TestZstdEncodeResp>("enc_resp");

        // IO for input RAM
        let (input_rd_req_s, input_rd_req_r) = chan<TestReadReq>("input_rd_req");
        let (input_rd_resp_s, input_rd_resp_r) = chan<TestReadResp>("input_rd_resp");
        let (input_wr_req_s, input_wr_req_r) = chan<TestWriteReq>("input_wr_req");
        let (input_wr_resp_s, input_wr_resp_r) = chan<TestWriteResp>("input_wr_resp");

        spawn ram::RamModel<
            TEST_RAM_DATA_W, TEST_RAM_SIZE, TEST_RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_RW_BEHAVIOR, TEST_RAM_INITIALIZED, TEST_RAM_ASSERT_VALID_READ, TEST_AXI_ADDR_W
        >(
            input_rd_req_r, input_rd_resp_s,
            input_wr_req_r, input_wr_resp_s
        );

        // IO for AxiRamReader <-> MemReader
        let (input_axi_ar_s, input_axi_ar_r) = chan<AxiAr>("input_axi_ar");
        let (input_axi_r_s, input_axi_r_r) = chan<AxiR>("input_axi_r");

        spawn axi_ram::AxiRamReader<
            TEST_AXI_ADDR_W, TEST_AXI_DATA_W, TEST_AXI_DEST_W, TEST_AXI_ID_W, TEST_RAM_SIZE
        >(
            input_axi_ar_r, input_axi_r_s,
            input_rd_req_s, input_rd_resp_r
        );

        // IO for MemReader <-> Encoder
        let (mem_rd_req_s, mem_rd_req_r) = chan<TestMemReaderReq>("mem_rd_req");
        let (mem_rd_resp_s, mem_rd_resp_r) = chan<TestMemReaderResp>("mem_rd_resp");

        spawn mem_reader::MemReader<TEST_AXI_DATA_W, TEST_AXI_ADDR_W, TEST_AXI_DEST_W, TEST_AXI_ID_W>(
            mem_rd_req_r, mem_rd_resp_s,
            input_axi_ar_s, input_axi_r_r
        );

        // IO for MemWriter <-> Encoder
        let (mem_wr_req_s, mem_wr_req_r) = chan<TestMemWriterReq>("mem_wr_req");
        let (mem_wr_data_s, mem_wr_data_r) = chan<TestMemWriterData>("mem_wr_data");
        let (mem_wr_resp_s, mem_wr_resp_r) = chan<TestMemWriterResp>("mem_wr_resp");

        // faulty MemWriter
        spawn MemWriterFaultResponder(
            mem_wr_req_r, mem_wr_data_r, mem_wr_resp_s
        );

        spawn zstd_enc::ZstdEncoder<TEST_ADDR_W, TEST_RAM_DATA_W>
        (
            enc_req_r, enc_resp_s,
            mem_rd_req_s, mem_rd_resp_r,
            mem_wr_req_s, mem_wr_data_s, mem_wr_resp_r
        );

        (
            input_wr_req_s, input_wr_resp_r,
            enc_req_s, enc_resp_r,
            terminator
        )
    }

    next(state: ()) {
        type Addr = bits[TEST_ADDR_W];

        // write input data to RAM
        let tok = for ((i, data), tok): ((u32, u32), token) in enumerate(TEST_INPUT_DATA) {
            let tok = send(tok, input_wr_req_s, TestWriteReq {
                addr: i as Addr,
                data: data,
                mask: !bits[TEST_RAM_NUM_PARTITIONS]:0,
            });
            let (tok, _) = recv(tok, input_wr_resp_r);
            tok
        }(join());

        // send request to encoder
        let tok = send(tok, enc_req_s, TestZstdEncodeReq{
            input_offset: uN[TEST_ADDR_W]:0,
            data_size: TEST_INPUT_SIZE as uN[TEST_RAM_DATA_W],
            output_offset: uN[TEST_ADDR_W]:0,
            max_block_size: TEST_MAX_BLOCK_SIZE,
        });
        let (tok, resp) = recv(tok, enc_resp_r);

        // expect error due to the faulty (on purpose) MemWriter
        assert_eq(resp.status, ZstdEncodeRespStatus::ERROR);

        send(join(), terminator, true);
    }
}
