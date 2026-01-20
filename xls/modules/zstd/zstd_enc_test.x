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
import xls.modules.zstd.memory.axi_ram_reader;
import xls.modules.zstd.memory.axi_ram_writer;
import xls.modules.zstd.mem_copy;
import xls.modules.zstd.memory.mem_writer;
import xls.modules.zstd.memory.mem_reader;
import xls.modules.zstd.common;
import xls.modules.zstd.mem_writer_simple_arbiter;
import xls.modules.zstd.mem_reader_simple_arbiter;
import xls.modules.zstd.sequence_encoder;

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
const TEST_RLE_HEURISTIC_SAMPLE_COUNT = u32:8;
// sequence encoding
const TEST_MIN_SEQ_LEN = u32:3;
const TEST_HT_SIZE = u32:512;
const TEST_HT_SIZE_W = std::clog2(TEST_HT_SIZE + u32:1);
const TEST_HT_KEY_W = u32:32;
const TEST_HT_VALUE_W = TEST_HT_KEY_W + TEST_ADDR_W;
const TEST_HT_HASH_W = std::clog2(TEST_HT_SIZE);
const TEST_HT_RAM_DATA_W = TEST_HT_VALUE_W + u32:1;
const TEST_HT_RAM_WORD_PARTITION_SIZE = u32:1;
const TEST_HT_RAM_NUM_PARTITIONS = ram::num_partitions(TEST_HT_RAM_WORD_PARTITION_SIZE, TEST_HT_RAM_DATA_W);
const TEST_HT_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR = ram::SimultaneousReadWriteBehavior::READ_BEFORE_WRITE;
const TEST_HT_RAM_INITIALIZED = true;

const TEST_HB_DATA_W = u32:64;
const TEST_HB_SIZE = u32:1024;
const TEST_HB_OFFSET_W = std::clog2(TEST_HB_SIZE);
const TEST_HB_RAM_NUM = u32:8;
const TEST_HB_RAM_SIZE = TEST_HB_SIZE / TEST_HB_RAM_NUM;
const TEST_HB_RAM_DATA_W = TEST_HB_DATA_W / TEST_HB_RAM_NUM;
const TEST_HB_RAM_ADDR_W = std::clog2(TEST_HB_RAM_SIZE);
const TEST_HB_RAM_PARTITION_SIZE = TEST_HB_RAM_DATA_W;
const TEST_HB_RAM_NUM_PARTITIONS = ram::num_partitions(TEST_HB_RAM_PARTITION_SIZE, TEST_HB_RAM_DATA_W);
const TEST_HB_RAM_SIMULTANEOUS_RW_BEHAVIOR = ram::SimultaneousReadWriteBehavior::READ_BEFORE_WRITE;
const TEST_HB_RAM_INITIALIZED = true;
const TEST_FSE_TABLE_RAM_ADDR_W = u32:32;
const TEST_FSE_CTABLE_RAM_DATA_W = u32:16;
const TEST_FSE_TTABLE_RAM_DATA_W = u32:64;
const TEST_FSE_TTABLE_RAM_PARTITION_SIZE = u32:8;
const TEST_FSE_TABLE_PARTITION_SIZE = u32:8;
const TEST_FSE_CTABLE_RAM_NUM_PARTITIONS = ram::num_partitions(TEST_FSE_TABLE_PARTITION_SIZE, TEST_FSE_CTABLE_RAM_DATA_W);
const TEST_FSE_TTABLE_RAM_NUM_PARTITIONS = ram::num_partitions(TEST_FSE_TABLE_PARTITION_SIZE, TEST_FSE_TTABLE_RAM_DATA_W);
const TEST_FSE_BITSTREAM_BUFFER_W = u32:1024;


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
type TestHistoryBufferRamRdReq = ram::ReadReq<TEST_HB_RAM_ADDR_W, TEST_HB_RAM_NUM_PARTITIONS>;
type TestHistoryBufferRamRdResp = ram::ReadResp<TEST_HB_RAM_DATA_W>;
type TestHistoryBufferRamWrReq = ram::WriteReq<TEST_HB_RAM_ADDR_W, TEST_HB_RAM_DATA_W, TEST_HB_RAM_NUM_PARTITIONS>;
type TestHistoryBufferRamWrResp = ram::WriteResp;
type TestHashTableRamRdReq = ram::ReadReq<TEST_HT_HASH_W, TEST_HT_RAM_NUM_PARTITIONS>;
type TestHashTableRamRdResp = ram::ReadResp<TEST_HT_RAM_DATA_W>;
type TestHashTableRamWrReq = ram::WriteReq<TEST_HT_HASH_W, TEST_HT_RAM_DATA_W, TEST_HT_RAM_NUM_PARTITIONS>;
type TestHashTableRamWrResp = ram::WriteResp;
type TestCTableRamRdReq = ram::ReadReq<TEST_FSE_TABLE_RAM_ADDR_W, TEST_FSE_CTABLE_RAM_NUM_PARTITIONS>;
type TestCTableRamRdResp = ram::ReadResp<TEST_FSE_CTABLE_RAM_DATA_W>;
type TestTTableRamRdReq = ram::ReadReq<TEST_FSE_TABLE_RAM_ADDR_W, TEST_FSE_TTABLE_RAM_NUM_PARTITIONS>;
type TestTTableRamRdResp = ram::ReadResp<TEST_FSE_TTABLE_RAM_DATA_W>;
type TestCTableRamWrReq = ram::WriteReq<TEST_FSE_TABLE_RAM_ADDR_W, TEST_FSE_CTABLE_RAM_DATA_W, TEST_FSE_CTABLE_RAM_NUM_PARTITIONS>;
type TestTTableRamWrReq = ram::WriteReq<TEST_FSE_TABLE_RAM_ADDR_W, TEST_FSE_TTABLE_RAM_DATA_W, TEST_FSE_TTABLE_RAM_NUM_PARTITIONS>;
type TestCTableRamWrResp = ram::WriteResp;
type TestTTableRamWrResp = ram::WriteResp;

type TestZstdEncodeReq = zstd_enc::ZstdEncodeReq<TEST_ADDR_W, TEST_RAM_DATA_W>;
type TestZstdEncodeResp = zstd_enc::ZstdEncodeResp<TEST_ADDR_W>;

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
const TEST_LITERALS_BUFFER_AXI_ADDR = TEST_EXPECTED_SIZE * u32:2;
const TEST_SEQUENCE_BUFFER_AXI_ADDR = TEST_EXPECTED_SIZE * u32:4;
const TEST_RAM_SIZE = TEST_EXPECTED_SIZE * u32:5;

const TEST_EXPECTED = u32[TEST_EXPECTED_SIZE]:[
    // FH - frame header,   BH - block header,  BB - block content
    // Magic_Number           BHFH FHFH           BBBB BHBH           BBBB BBBB           BBBB ...
    u32:0xFD2F_B528,    u32:0x0090_6040,    u32:0xA500_0C81,    u32:0xD3D9_4550,    u32:0x46A2_0CD8,
    u32:0x26B0_BED0,    u32:0xC4F8_3C6D,    u32:0xC4FA_E4B0,    u32:0x1E9A_7891,    u32:0x6DFD_A09B,
    u32:0xCB5E_66D7,    u32:0x2FCB_7D76,    u32:0x0B40_335F,    u32:0x5F21_289B,    u32:0x81D2_6336,
    u32:0xC9D9_89DD,    u32:0xB6E4_CB45,    u32:0x7C04_2506,    u32:0x675D_3110,    u32:0x9422_827A,
    u32:0xD8CA_C70C,    u32:0xC323_A95F,    u32:0xD061_22BB,    u32:0x341F_99F3,    u32:0xF2A7_0CFB,
    u32:0xBC38_125E,    u32:0xB191_5761,    u32:0x0817_1AC1,    u32:0xAFDE_6F1B,    u32:0x1B42_0DF1,
    u32:0xA3AE_E9F5,    u32:0xD6B3_1EE3,    u32:0x0366_AC09,    u32:0x7A18_E997,    u32:0xDEEE_871E,
    u32:0x52B6_3D47,    u32:0x3659_BF4F,    u32:0xEE94_D856,    u32:0x682B_8134,    u32:0xCB67_1199,
    u32:0x1BFB_2BF8,    u32:0x6717_3FCB,    u32:0x4FFB_943A,    u32:0xFEF4_0B71,    u32:0x5E38_3B82,
    u32:0x10A6_9205,    u32:0xE058_A621,    u32:0x2201_85B5,    u32:0x579D_F09C,    u32:0x7F54_CADB,
    u32:0x10C6_2609,    u32:0x36EA_0431,    u32:0xB0F1_1C4D,    u32:0x20B8_CCFA,    u32:0x9C78_013B,
    u32:0x0581_89BF,    u32:0x34E3_80A5,    u32:0x3A46_72AE,    u32:0x9E1C_D51B,    u32:0x315F_95EE,
    u32:0xD2BC_5C99,    u32:0x44BC_E650,    u32:0xA1C1_0D05,    u32:0x945A_B4DE,    u32:0xA15E_2033,
    u32:0x2E7F_DA0C,    u32:0x816F_EC11,    u32:0x9110_7A2F,    u32:0xB786_CA44,    u32:0x22EA_680E,
    u32:0xCA50_F1AA,    u32:0xF73F_47F2,    u32:0xE0E4_0792,    u32:0x19F3_5CEE,    u32:0xFA1D_6BE8,
    u32:0x993F_A705,    u32:0x1208_BBA4,    u32:0x5C7C_0C48,    u32:0x16F5_A53D,    u32:0x4B07_9ABE,
    u32:0x02AC_A1F8,    u32:0xFD4D_2B94,    u32:0xA545_B128,    u32:0x322C_7CCB,    u32:0x8868_74FC,
    u32:0x0795_A082,    u32:0xEFFB_13E7,    u32:0xFC61_F92F,    u32:0x80F6_E3DA,    u32:0x1BDB_A00A,
    u32:0x20BB_8483,    u32:0x17AD_6325,    u32:0x35EF_B3D8,    u32:0x4FD1_90C4,    u32:0x2890_641E,
    u32:0x4C08_393D,    u32:0x331C_0787,    u32:0x51BB_EBD6,    u32:0x40B0_A9C7,    u32:0x8A83_B9A3,
    u32:0x5C02_8AFF,    u32:0x9CB4_EDEE,    u32:0x00D7_00BD,
];

// touch empty.txt
// zstd --no-check empty.txt -o empty.zst
// xxd -e empty.zst
const TEST_EMPTY_EXPECTED_SIZE = u32:3;
const TEST_EMPTY_EXPECTED = u32[TEST_EMPTY_EXPECTED_SIZE]:[
    // Magic_Number      //      FH FHFH
    u32:0xFD2F_B528,     u32:0x0001_6000,   u32:0x0
];

// decodecorpus -p./ -o./ -n1 --block-type=1 --content-size -s35766 (after hard-coding certain values)
// xxd -e -c4 z000000
const TEST_RLE_INPUT_SIZE = u32:64;
const TEST_RLE_INPUT_SIZE_BYTES = TEST_RLE_INPUT_SIZE * (TEST_RAM_DATA_W / u32:8);
const TEST_RLE_INPUT_DATA = u32[TEST_RLE_INPUT_SIZE]:[
    u32: 0xa3a3a3a3, u32: 0xa3a3a3a3, u32: 0xa3a3a3a3, u32: 0xa3a3a3a3, u32: 0xa3a3a3a3, u32: 0xa3a3a3a3, u32: 0xa3a3a3a3, u32: 0xa3a3a3a3,
    u32: 0x4d4d4d4d, u32: 0x4d4d4d4d, u32: 0x4d4d4d4d, u32: 0x4d4d4d4d, u32: 0x4d4d4d4d, u32: 0x4d4d4d4d, u32: 0x4d4d4d4d, u32: 0x4d4d4d4d,
    u32: 0x4f4f4f4f, u32: 0x4f4f4f4f, u32: 0x4f4f4f4f, u32: 0x4f4f4f4f, u32: 0x4f4f4f4f, u32: 0x4f4f4f4f, u32: 0x4f4f4f4f, u32: 0x4f4f4f4f,
    u32: 0x14141414, u32: 0x14141414, u32: 0x14141414, u32: 0x14141414, u32: 0x14141414, u32: 0x14141414, u32: 0x14141414, u32: 0x14141414,
    u32: 0xf6f6f6f6, u32: 0xf6f6f6f6, u32: 0xf6f6f6f6, u32: 0xf6f6f6f6, u32: 0xf6f6f6f6, u32: 0xf6f6f6f6, u32: 0xf6f6f6f6, u32: 0xf6f6f6f6,
    u32: 0xe0e0e0e0, u32: 0xe0e0e0e0, u32: 0xe0e0e0e0, u32: 0xe0e0e0e0, u32: 0xe0e0e0e0, u32: 0xe0e0e0e0, u32: 0xe0e0e0e0, u32: 0xe0e0e0e0,
    u32: 0x5b5b5b5b, u32: 0x5b5b5b5b, u32: 0x5b5b5b5b, u32: 0x5b5b5b5b, u32: 0x5b5b5b5b, u32: 0x5b5b5b5b, u32: 0x5b5b5b5b, u32: 0x5b5b5b5b,
    u32: 0x61616161, u32: 0x61616161, u32: 0x61616161, u32: 0x61616161, u32: 0x61616161, u32: 0x61616161, u32: 0x61616161, u32: 0x61616161,
];
// xxd -e -c4 z000000.zst
const TEST_RLE_EXPECTED_SIZE = u32:10;
const TEST_RLE_EXPECTED = u32[TEST_RLE_EXPECTED_SIZE]:[
    // block indexes        1               2 1 1 1         3 2 2 2
    // Magic_Number        BHFHFHFH        BHBBBHBH        BHBBBHBH
    u32: 0xfd2fb528, u32:0x02000060, u32:0x02a30001, u32:0x024d0001,
    //      4 3 3 3         5 4 4 4         6 5 5 5        7  6 6 6
    //     BHBBBHBH        BHBBBHBH        BHBBBHBH        BHBBBHBH
    u32: 0x024f0001, u32:0x02140001, u32:0x02f60001, u32:0x02e00001,
    //      8 7 7 7           8 8 8
    //     BHBBBHBH        --BBBHBH
    u32: 0x035b0001, u32:0x00610001
];


// How the input data below came to be:
// 1. encode some random data
// 2. try to decode it with zstd
// 3. see if it produces correct output
// 00000000: fd2fb528 01142820 00b5d000 f00db16b  (./. (......k...
// 00000010: beefb105 beefbaad ffbeefdd cafebabe  ................
// 00000020: dabbad00 603a0002 5d757018 ce400000  ......:`.pu]..@.
// 00000030: eefeedfa 000ff1ce     0000           ..........
const TEST_COMPRESSED_PREDEFINED_INPUT_SIZE = u32:10;
const TEST_COMPRESSED_PREDEFINED_INPUT_SIZE_BYTES = TEST_COMPRESSED_PREDEFINED_INPUT_SIZE * (TEST_RAM_DATA_W / u32:8);
const TEST_COMPRESSED_PREDEFINED_INPUT = u32[TEST_COMPRESSED_PREDEFINED_INPUT_SIZE]:[
    u32:0xB16B00B5, u32:0xB105F00D, u32:0xBAADBEEF, u32:0xBADDBEEF, u32:0xBEEFBEEF,
    u32:0xFFBEEFFF, u32:0xCAFEBABE, u32:0xDABBAD00, u32:0xFEEDFACE, u32:0x0FF1CEEE
];
const TEST_COMPRESS_PREDEFINED_EXPECTED_SIZE = u32:16;
const TEST_COMPRESS_PREDEFINED_EXPECTED = [
    // S - sequences
    // L - literals
    // SH - sequence header
    // magic number       BHBHBHBH         LLLLLLLL        LLLLLLLL
    u32:0xfd2fb528, u32:0x01142820, u32: 0x00b5d000, u32:0xf00db16b,
    //    LLLLLLLL        LLLLLLLL        LLLLLLLL        LLLLLLLL
    u32:0xbeefb105, u32:0xbeefbaad, u32:0xffbeefdd, u32:0xcafebabe,
    //    LLLLLLLL        SSSSSHSH        BHSSSSSS        LLBHBHBH
    u32:0xdabbad00, u32:0x603a0002, u32:0x55757018, u32:0xce400000,
    //    LLLLLLLL        SHLLLLLL        SH
    u32:0xeefeedfa, u32:0x000ff1ce, u32:0x00  // no sequences in this block (only header)
];


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
        // IO for AxiRamReader <-> MemReader
        let (input_axi_ar_s, input_axi_ar_r) = chan<AxiAr>("input_axi_ar");
        let (input_axi_r_s, input_axi_r_r) = chan<AxiR>("input_axi_r");
        // IO for MemReader <-> Encoder
        let (mem_rd_req_s, mem_rd_req_r) = chan<TestMemReaderReq>("mem_rd_req");
        let (mem_rd_resp_s, mem_rd_resp_r) = chan<TestMemReaderResp>("mem_rd_resp");
        // IO for output RAM
        let (output_rd_req_s, output_rd_req_r) = chan<TestReadReq>("output_rd_req");
        let (output_rd_resp_s, output_rd_resp_r) = chan<TestReadResp>("output_rd_resp");
        let (output_wr_req_s, output_wr_req_r) = chan<TestWriteReq>("output_wr_req");
        let (output_wr_resp_s, output_wr_resp_r) = chan<TestWriteResp>("output_wr_resp");
        // IO for AxiRamWriter <-> MemWriter
        let (output_axi_aw_s, output_axi_aw_r) = chan<AxiAw>("output_axi_aw");
        let (output_axi_w_s, output_axi_w_r) = chan<AxiW>("output_axi_w");
        let (output_axi_b_s, output_axi_b_r) = chan<AxiB>("output_axi_b");
        // IO for MemWriter <-> Encoder
        let (mem_wr_req_s, mem_wr_req_r) = chan<TestMemWriterReq>("mem_wr_req");
        let (mem_wr_data_s, mem_wr_data_r) = chan<TestMemWriterData>("mem_wr_data");
        let (mem_wr_resp_s, mem_wr_resp_r) = chan<TestMemWriterResp>("mem_wr_resp");
        let (n_mem_wr_req_s, n_mem_wr_req_r) = chan<TestMemWriterReq, u32:1>[8]("n_req");
        let (n_mem_wr_data_s, n_mem_wr_data_r) = chan<TestMemWriterData, u32:1>[8]("n_data");
        let (n_mem_wr_resp_s, n_mem_wr_resp_r) = chan<TestMemWriterResp, u32:1>[8]("n_resp");
        let (n_mem_rd_req_s, n_mem_rd_req_r) = chan<TestMemReaderReq, u32:1>[4]("n_mem_rd_req");
        let (n_mem_rd_resp_s, n_mem_rd_resp_r) = chan<TestMemReaderResp, u32:1>[4]("n_mem_rd_resp");
        let (hb_ram_rd_req_s, _) = chan<TestHistoryBufferRamRdReq>[TEST_HB_RAM_NUM]("hb_ram_rd_req");
        let (_, hb_ram_rd_resp_r) = chan<TestHistoryBufferRamRdResp>[TEST_HB_RAM_NUM]("hb_ram_rd_resp");
        let (hb_ram_wr_req_s, _) = chan<TestHistoryBufferRamWrReq>[TEST_HB_RAM_NUM]("hb_ram_wr_req");
        let (_, hb_ram_wr_resp_r) = chan<TestHistoryBufferRamWrResp>[TEST_HB_RAM_NUM]("hb_ram_wr_resp");
        let (ht_ram_rd_req_s, _) = chan<TestHashTableRamRdReq>("ht_ram_rd_req");
        let (_, ht_ram_rd_resp_r) = chan<TestHashTableRamRdResp>("ht_ram_rd_resp");
        let (ht_ram_wr_req_s, _) = chan<TestHashTableRamWrReq>("ht_ram_wr_req");
        let (_, ht_ram_wr_resp_r) = chan<TestHashTableRamWrResp>("ht_ram_wr_resp");
        let (ml_ctable_ram_rd_req_s, _) = chan<TestCTableRamRdReq>("ml_ctable_ram_rd_req");
        let (_, ml_ctable_ram_rd_resp_r) = chan<TestCTableRamRdResp>("ml_ctable_ram_rd_resp");
        let (ll_ctable_ram_rd_req_s, _) = chan<TestCTableRamRdReq>("ll_ctable_ram_rd_req");
        let (_, ll_ctable_ram_rd_resp_r) = chan<TestCTableRamRdResp>("ll_ctable_ram_rd_resp");
        let (of_ctable_ram_rd_req_s, _) = chan<TestCTableRamRdReq>("of_ctable_ram_rd_req");
        let (_, of_ctable_ram_rd_resp_r) = chan<TestCTableRamRdResp>("of_ctable_ram_rd_resp");
        let (ml_ttable_ram_rd_req_s, _) = chan<TestTTableRamRdReq>("ml_ttable_ram_rd_req");
        let (_, ml_ttable_ram_rd_resp_r) = chan<TestTTableRamRdResp>("ml_ttable_ram_rd_resp");
        let (ll_ttable_ram_rd_req_s, _) = chan<TestTTableRamRdReq>("ll_ttable_ram_rd_req");
        let (_, ll_ttable_ram_rd_resp_r) = chan<TestTTableRamRdResp>("ll_ttable_ram_rd_resp");
        let (of_ttable_ram_rd_req_s, _) = chan<TestTTableRamRdReq>("of_ttable_ram_rd_req");
        let (_, of_ttable_ram_rd_resp_r) = chan<TestTTableRamRdResp>("of_ttable_ram_rd_resp");

        spawn ram::RamModel<
            TEST_RAM_DATA_W, TEST_RAM_SIZE, TEST_RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_RW_BEHAVIOR, TEST_RAM_INITIALIZED, TEST_RAM_ASSERT_VALID_READ, TEST_AXI_ADDR_W
        >(
            input_rd_req_r, input_rd_resp_s,
            input_wr_req_r, input_wr_resp_s
        );
        spawn ram::RamModel<
        TEST_RAM_DATA_W, TEST_RAM_SIZE, TEST_RAM_WORD_PARTITION_SIZE,
        TEST_RAM_SIMULTANEOUS_RW_BEHAVIOR, TEST_RAM_INITIALIZED,
        TEST_RAM_ASSERT_VALID_READ, TEST_ADDR_W
        >(
            output_rd_req_r, output_rd_resp_s, output_wr_req_r, output_wr_resp_s
        );
        spawn axi_ram_reader::AxiRamReader<
            TEST_AXI_ADDR_W, TEST_AXI_DATA_W, TEST_AXI_DEST_W, TEST_AXI_ID_W, TEST_RAM_SIZE
        >(
            input_axi_ar_r, input_axi_r_s,
            input_rd_req_s, input_rd_resp_r
        );
        spawn axi_ram_writer::AxiRamWriter<
        TEST_AXI_ADDR_W, TEST_AXI_DATA_W, TEST_AXI_ID_W, TEST_RAM_SIZE, TEST_ADDR_W, TEST_RAM_NUM_PARTITIONS,
        >(
            output_axi_aw_r, output_axi_w_r, output_axi_b_s,
            output_wr_req_s, output_wr_resp_r
        );

        spawn mem_reader::MemReader<TEST_AXI_DATA_W, TEST_AXI_ADDR_W, TEST_AXI_DEST_W, TEST_AXI_ID_W>(
            mem_rd_req_r, mem_rd_resp_s,
            input_axi_ar_s, input_axi_r_r
        );
        spawn mem_writer::MemWriter<
            TEST_AXI_ADDR_W, TEST_AXI_DATA_W, TEST_AXI_DEST_W, TEST_AXI_ID_W, TEST_WRITER_ID
        >(
            mem_wr_req_r, mem_wr_data_r,
            output_axi_aw_s, output_axi_w_s, output_axi_b_r,
            mem_wr_resp_s
        );

        spawn mem_writer_simple_arbiter::MemWriterSimpleArbiter<TEST_ADDR_W, TEST_AXI_DATA_W, u32:8>
        (
            n_mem_wr_req_r, n_mem_wr_data_r, n_mem_wr_resp_s,
            mem_wr_req_s, mem_wr_data_s, mem_wr_resp_r,
        );

        spawn mem_reader_simple_arbiter::MemReaderSimpleArbiter<TEST_ADDR_W, TEST_AXI_DATA_W, u32:4> (
            n_mem_rd_req_r, n_mem_rd_resp_s,
            mem_rd_req_s, mem_rd_resp_r,
        );

        spawn zstd_enc::ZstdEncoder<
            TEST_AXI_ADDR_W, TEST_AXI_DATA_W,
            TEST_RLE_HEURISTIC_SAMPLE_COUNT,
            TEST_HB_SIZE, TEST_HB_DATA_W, TEST_HB_OFFSET_W, TEST_HB_RAM_ADDR_W, TEST_HB_RAM_DATA_W, TEST_HB_RAM_NUM, TEST_HB_RAM_NUM_PARTITIONS,
            TEST_HT_SIZE, TEST_HT_KEY_W, TEST_HT_VALUE_W, TEST_HT_SIZE_W, TEST_HT_HASH_W, TEST_HT_RAM_DATA_W, TEST_HT_RAM_NUM_PARTITIONS,
            TEST_MIN_SEQ_LEN, TEST_LITERALS_BUFFER_AXI_ADDR, TEST_SEQUENCE_BUFFER_AXI_ADDR,
            TEST_FSE_TABLE_RAM_ADDR_W, TEST_FSE_CTABLE_RAM_DATA_W, TEST_FSE_TTABLE_RAM_DATA_W, TEST_FSE_CTABLE_RAM_NUM_PARTITIONS, TEST_FSE_TTABLE_RAM_NUM_PARTITIONS,
            TEST_FSE_BITSTREAM_BUFFER_W
        >(
            enc_req_r, enc_resp_s,
            n_mem_wr_req_s[0], n_mem_wr_data_s[0], n_mem_wr_resp_r[0],
            n_mem_wr_req_s[1], n_mem_wr_data_s[1], n_mem_wr_resp_r[1],
            n_mem_wr_req_s[2], n_mem_wr_data_s[2], n_mem_wr_resp_r[2],
            n_mem_wr_req_s[3], n_mem_wr_data_s[3], n_mem_wr_resp_r[3],
            n_mem_wr_req_s[4], n_mem_wr_data_s[4], n_mem_wr_resp_r[4],
            n_mem_wr_req_s[5], n_mem_wr_data_s[5], n_mem_wr_resp_r[5],
            n_mem_wr_req_s[6], n_mem_wr_data_s[6], n_mem_wr_resp_r[6],
            n_mem_wr_req_s[7], n_mem_wr_data_s[7], n_mem_wr_resp_r[7],
            n_mem_rd_req_s[0], n_mem_rd_resp_r[0],
            n_mem_rd_req_s[1], n_mem_rd_resp_r[1],
            n_mem_rd_req_s[2], n_mem_rd_resp_r[2],
            n_mem_rd_req_s[3], n_mem_rd_resp_r[3],
            hb_ram_rd_req_s, hb_ram_rd_resp_r, hb_ram_wr_req_s, hb_ram_wr_resp_r,
            ht_ram_rd_req_s, ht_ram_rd_resp_r, ht_ram_wr_req_s, ht_ram_wr_resp_r,
            ml_ctable_ram_rd_req_s, ml_ctable_ram_rd_resp_r,
            ll_ctable_ram_rd_req_s, ll_ctable_ram_rd_resp_r,
            of_ctable_ram_rd_req_s, of_ctable_ram_rd_resp_r,
            ml_ttable_ram_rd_req_s, ml_ttable_ram_rd_resp_r,
            ll_ttable_ram_rd_req_s, ll_ttable_ram_rd_resp_r,
            of_ttable_ram_rd_req_s, of_ttable_ram_rd_resp_r
        );

        (input_wr_req_s, input_wr_resp_r, output_rd_req_s, output_rd_resp_r, enc_req_s, enc_resp_r, terminator)
    }

    next(state: ()) {
        type Addr = bits[TEST_ADDR_W];

        // write ZEROS to RAM
        let tok = for ((i, _data), tok): ((u32, u32), token) in enumerate(TEST_INPUT_DATA) {
            let tok = send(tok, input_wr_req_s, TestWriteReq {
                addr: i as Addr,
                data: u32:0,
                mask: all_ones!<uN[TEST_RAM_NUM_PARTITIONS]>(),
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
            params: zero!<zstd_enc::ZstdEncodeParams>()
        });
        let (tok, resp) = recv(tok, enc_resp_r);
        assert_eq(resp.status, ZstdEncodeRespStatus::OK);

        // read state of output RAM
        let tok = for ((i, expected_val), tok): ((u32, u32), token) in enumerate(TEST_EMPTY_EXPECTED) {
            let tok = send(tok, output_rd_req_s, TestReadReq {
                addr: i as Addr,
                mask: all_ones!<uN[TEST_RAM_NUM_PARTITIONS]>(),
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

proc ZstdEncoderTestBase {
    type AxiAr = axi::AxiAr<TEST_AXI_ADDR_W, TEST_AXI_ID_W>;
    type AxiR = axi::AxiR<TEST_AXI_DATA_W, TEST_AXI_ID_W>;
    type AxiAw = axi::AxiAw<TEST_AXI_ADDR_W, TEST_AXI_ID_W>;
    type AxiW = axi::AxiW<TEST_AXI_DATA_W, TEST_RAM_NUM_PARTITIONS>;
    type AxiB = axi::AxiB<TEST_AXI_ID_W>;

    init { }

    write_fse_tables_req_r: chan<()> in;
    write_fse_tables_resp_s: chan<()> out;
    ll_ctable_ram_wr_req_s: chan<TestCTableRamWrReq> out;
    ll_ctable_ram_wr_resp_r: chan<TestCTableRamWrResp> in;
    of_ctable_ram_wr_req_s: chan<TestCTableRamWrReq> out;
    of_ctable_ram_wr_resp_r: chan<TestCTableRamWrResp> in;
    ml_ctable_ram_wr_req_s: chan<TestCTableRamWrReq> out;
    ml_ctable_ram_wr_resp_r: chan<TestCTableRamWrResp> in;
    ll_ttable_ram_wr_req_s: chan<TestTTableRamWrReq> out;
    ll_ttable_ram_wr_resp_r: chan<TestTTableRamWrResp> in;
    of_ttable_ram_wr_req_s: chan<TestTTableRamWrReq> out;
    of_ttable_ram_wr_resp_r: chan<TestTTableRamWrResp> in;
    ml_ttable_ram_wr_req_s: chan<TestTTableRamWrReq> out;
    ml_ttable_ram_wr_resp_r: chan<TestTTableRamWrResp> in;

    config(
        write_fse_tables_req_r: chan<()> in,
        write_fse_tables_resp_s: chan<()> out,
        input_wr_req_r: chan<TestWriteReq> in,
        input_wr_resp_s: chan<TestWriteResp> out,
        output_rd_req_r: chan<TestReadReq> in,
        output_rd_resp_s: chan<TestReadResp> out,
        enc_req_r: chan<TestZstdEncodeReq> in,
        enc_resp_s: chan<TestZstdEncodeResp> out
    ) {
        // IO for input RAM
        let (input_rd_req_s, input_rd_req_r) = chan<TestReadReq>("input_rd_req");
        let (input_rd_resp_s, input_rd_resp_r) = chan<TestReadResp>("input_rd_resp");
        // IO for AxiRamReader <-> MemReader
        let (input_axi_ar_s, input_axi_ar_r) = chan<AxiAr>("input_axi_ar");
        let (input_axi_r_s, input_axi_r_r) = chan<AxiR>("input_axi_r");
        // IO for MemReader <-> Encoder
        let (mem_rd_req_s, mem_rd_req_r) = chan<TestMemReaderReq>("mem_rd_req");
        let (mem_rd_resp_s, mem_rd_resp_r) = chan<TestMemReaderResp>("mem_rd_resp");
        // IO for output RAM
        let (output_wr_req_s, output_wr_req_r) = chan<TestWriteReq>("output_wr_req");
        let (output_wr_resp_s, output_wr_resp_r) = chan<TestWriteResp>("output_wr_resp");

        // IO for AxiRamWriter <-> MemWriter
        let (output_axi_aw_s, output_axi_aw_r) = chan<AxiAw>("output_axi_aw");
        let (output_axi_w_s, output_axi_w_r) = chan<AxiW>("output_axi_w");
        let (output_axi_b_s, output_axi_b_r) = chan<AxiB>("output_axi_b");

        // IO for MemWriter <-> Encoder
        let (mem_wr_req_s, mem_wr_req_r) = chan<TestMemWriterReq>("mem_wr_req");
        let (mem_wr_data_s, mem_wr_data_r) = chan<TestMemWriterData>("mem_wr_data");
        let (mem_wr_resp_s, mem_wr_resp_r) = chan<TestMemWriterResp>("mem_wr_resp");
        let (n_mem_wr_req_s, n_mem_wr_req_r) = chan<TestMemWriterReq, u32:1>[7]("n_req");
        let (n_mem_wr_data_s, n_mem_wr_data_r) = chan<TestMemWriterData, u32:1>[7]("n_data");
        let (n_mem_wr_resp_s, n_mem_wr_resp_r) = chan<TestMemWriterResp, u32:1>[7]("n_resp");
        let (n_mem_rd_req_s, n_mem_rd_req_r) = chan<TestMemReaderReq, u32:1>[3]("n_mem_rd_req");
        let (n_mem_rd_resp_s, n_mem_rd_resp_r) = chan<TestMemReaderResp, u32:1>[3]("n_mem_rd_resp");
        let (hb_ram_rd_req_s, hb_ram_rd_req_r) = chan<TestHistoryBufferRamRdReq>[TEST_HB_RAM_NUM]("hb_ram_rd_req");
        let (hb_ram_rd_resp_s, hb_ram_rd_resp_r) = chan<TestHistoryBufferRamRdResp>[TEST_HB_RAM_NUM]("hb_ram_rd_resp");
        let (hb_ram_wr_req_s, hb_ram_wr_req_r) = chan<TestHistoryBufferRamWrReq>[TEST_HB_RAM_NUM]("hb_ram_wr_req");
        let (hb_ram_wr_resp_s, hb_ram_wr_resp_r) = chan<TestHistoryBufferRamWrResp>[TEST_HB_RAM_NUM]("hb_ram_wr_resp");
        let (ht_ram_rd_req_s, ht_ram_rd_req_r) = chan<TestHashTableRamRdReq>("ht_ram_rd_req");
        let (ht_ram_rd_resp_s, ht_ram_rd_resp_r) = chan<TestHashTableRamRdResp>("ht_ram_rd_resp");
        let (ht_ram_wr_req_s, ht_ram_wr_req_r) = chan<TestHashTableRamWrReq>("ht_ram_wr_req");
        let (ht_ram_wr_resp_s, ht_ram_wr_resp_r) = chan<TestHashTableRamWrResp>("ht_ram_wr_resp");
        let (ml_ctable_ram_rd_req_s, ml_ctable_ram_rd_req_r) = chan<TestCTableRamRdReq>("ml_ctable_ram_rd_req");
        let (ml_ctable_ram_rd_resp_s, ml_ctable_ram_rd_resp_r) = chan<TestCTableRamRdResp>("ml_ctable_ram_rd_resp");
        let (ll_ctable_ram_rd_req_s, ll_ctable_ram_rd_req_r) = chan<TestCTableRamRdReq>("ll_ctable_ram_rd_req");
        let (ll_ctable_ram_rd_resp_s, ll_ctable_ram_rd_resp_r) = chan<TestCTableRamRdResp>("ll_ctable_ram_rd_resp");
        let (of_ctable_ram_rd_req_s, of_ctable_ram_rd_req_r) = chan<TestCTableRamRdReq>("of_ctable_ram_rd_req");
        let (of_ctable_ram_rd_resp_s, of_ctable_ram_rd_resp_r) = chan<TestCTableRamRdResp>("of_ctable_ram_rd_resp");
        let (ml_ttable_ram_rd_req_s, ml_ttable_ram_rd_req_r) = chan<TestTTableRamRdReq>("ml_ttable_ram_rd_req");
        let (ml_ttable_ram_rd_resp_s, ml_ttable_ram_rd_resp_r) = chan<TestTTableRamRdResp>("ml_ttable_ram_rd_resp");
        let (ll_ttable_ram_rd_req_s, ll_ttable_ram_rd_req_r) = chan<TestTTableRamRdReq>("ll_ttable_ram_rd_req");
        let (ll_ttable_ram_rd_resp_s, ll_ttable_ram_rd_resp_r) = chan<TestTTableRamRdResp>("ll_ttable_ram_rd_resp");
        let (of_ttable_ram_rd_req_s, of_ttable_ram_rd_req_r) = chan<TestTTableRamRdReq>("of_ttable_ram_rd_req");
        let (of_ttable_ram_rd_resp_s, of_ttable_ram_rd_resp_r) = chan<TestTTableRamRdResp>("of_ttable_ram_rd_resp");
        let (ll_ctable_ram_wr_req_s, ll_ctable_ram_wr_req_r) = chan<TestCTableRamWrReq>("ll_ctable_ram_wr_req");
        let (ll_ctable_ram_wr_resp_s, ll_ctable_ram_wr_resp_r) = chan<TestCTableRamWrResp>("ll_ctable_ram_wr_resp");
        let (of_ctable_ram_wr_req_s, of_ctable_ram_wr_req_r) = chan<TestCTableRamWrReq>("of_ctable_ram_wr_req");
        let (of_ctable_ram_wr_resp_s, of_ctable_ram_wr_resp_r) = chan<TestCTableRamWrResp>("of_ctable_ram_wr_resp");
        let (ml_ctable_ram_wr_req_s, ml_ctable_ram_wr_req_r) = chan<TestCTableRamWrReq>("ml_ctable_ram_wr_req");
        let (ml_ctable_ram_wr_resp_s, ml_ctable_ram_wr_resp_r) = chan<TestCTableRamWrResp>("ml_ctable_ram_wr_resp");
        let (ll_ttable_ram_wr_req_s, ll_ttable_ram_wr_req_r) = chan<TestTTableRamWrReq>("ll_ttable_ram_wr_req");
        let (ll_ttable_ram_wr_resp_s, ll_ttable_ram_wr_resp_r) = chan<TestTTableRamWrResp>("ll_ttable_ram_wr_resp");
        let (of_ttable_ram_wr_req_s, of_ttable_ram_wr_req_r) = chan<TestTTableRamWrReq>("of_ttable_ram_wr_req");
        let (of_ttable_ram_wr_resp_s, of_ttable_ram_wr_resp_r) = chan<TestTTableRamWrResp>("of_ttable_ram_wr_resp");
        let (ml_ttable_ram_wr_req_s, ml_ttable_ram_wr_req_r) = chan<TestTTableRamWrReq>("ml_ttable_ram_wr_req");
        let (ml_ttable_ram_wr_resp_s, ml_ttable_ram_wr_resp_r) = chan<TestTTableRamWrResp>("ml_ttable_ram_wr_resp");
        // MatchFinder buffer
        let (buf_rd_req_s, buf_rd_req_r) = chan<TestReadReq>("buf_rd_req");
        let (buf_rd_resp_s, buf_rd_resp_r) = chan<TestReadResp>("buf_rd_resp");
        let (buf_wr_req_s, buf_wr_req_r) = chan<TestWriteReq>("buf_wr_req");
        let (buf_wr_resp_s, buf_wr_resp_r) = chan<TestWriteResp>("buf_wr_resp");
        let (buf_axi_aw_s, buf_axi_aw_r) = chan<AxiAw>("buf_axi_aw");
        let (buf_axi_w_s, buf_axi_w_r) = chan<AxiW>("buf_axi_w");
        let (buf_axi_b_s, buf_axi_b_r) = chan<AxiB>("buf_axi_b");
        let (buf_axi_ar_s, buf_axi_ar_r) = chan<AxiAr>("buf_axi_ar");
        let (buf_axi_r_s, buf_axi_r_r) = chan<AxiR>("buf_axi_r");
        let (buf_mem_wr_req_s, buf_mem_wr_req_r) = chan<TestMemWriterReq>("buf_mem_wr_req");
        let (buf_mem_wr_data_s, buf_mem_wr_data_r) = chan<TestMemWriterData>("buf_mem_wr_data");
        let (buf_mem_wr_resp_s, buf_mem_wr_resp_r) = chan<TestMemWriterResp>("buf_mem_wr_resp");
        let (buf_mem_rd_req_s, buf_mem_rd_req_r) = chan<TestMemReaderReq>("buf_mem_rd_req");
        let (buf_mem_rd_resp_s, buf_mem_rd_resp_r) = chan<TestMemReaderResp>("buf_mem_rd_resp");

        spawn ram::RamModel<
            TEST_RAM_DATA_W, TEST_RAM_SIZE, TEST_RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_RW_BEHAVIOR, TEST_RAM_INITIALIZED, TEST_RAM_ASSERT_VALID_READ, TEST_AXI_ADDR_W
        >(
            input_rd_req_r, input_rd_resp_s,
            input_wr_req_r, input_wr_resp_s
        );
        spawn ram::RamModel<
            TEST_RAM_DATA_W, TEST_RAM_SIZE, TEST_RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_RW_BEHAVIOR, TEST_RAM_INITIALIZED,
            TEST_RAM_ASSERT_VALID_READ, TEST_ADDR_W
        >(
            output_rd_req_r, output_rd_resp_s,
            output_wr_req_r, output_wr_resp_s
        );
        spawn ram::RamModel<
            TEST_RAM_DATA_W, TEST_RAM_SIZE, TEST_RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_RW_BEHAVIOR, TEST_RAM_INITIALIZED,
            TEST_RAM_ASSERT_VALID_READ, TEST_ADDR_W
        >(
            buf_rd_req_r, buf_rd_resp_s,
            buf_wr_req_r, buf_wr_resp_s
        );


        spawn axi_ram_reader::AxiRamReader<
            TEST_AXI_ADDR_W, TEST_AXI_DATA_W, TEST_AXI_DEST_W, TEST_AXI_ID_W, TEST_RAM_SIZE
        >(
            buf_axi_ar_r, buf_axi_r_s,
            buf_rd_req_s, buf_rd_resp_r
        );

        spawn axi_ram_writer::AxiRamWriter<
        TEST_AXI_ADDR_W, TEST_AXI_DATA_W, TEST_AXI_ID_W, TEST_RAM_SIZE, TEST_ADDR_W, TEST_RAM_NUM_PARTITIONS,
        >(
            buf_axi_aw_r, buf_axi_w_r, buf_axi_b_s,
            buf_wr_req_s, buf_wr_resp_r
        );

        spawn mem_reader::MemReader<TEST_AXI_DATA_W, TEST_AXI_ADDR_W, TEST_AXI_DEST_W, TEST_AXI_ID_W>(
            buf_mem_rd_req_r, buf_mem_rd_resp_s,
            buf_axi_ar_s, buf_axi_r_r
        );
        spawn mem_writer::MemWriter<
            TEST_AXI_ADDR_W, TEST_AXI_DATA_W, TEST_AXI_DEST_W, TEST_AXI_ID_W, TEST_WRITER_ID
        >(
            buf_mem_wr_req_r, buf_mem_wr_data_r,
            buf_axi_aw_s, buf_axi_w_s, buf_axi_b_r,
            buf_mem_wr_resp_s
        );

        spawn axi_ram_reader::AxiRamReader<
            TEST_AXI_ADDR_W, TEST_AXI_DATA_W, TEST_AXI_DEST_W, TEST_AXI_ID_W, TEST_RAM_SIZE
        >(
            input_axi_ar_r, input_axi_r_s,
            input_rd_req_s, input_rd_resp_r
        );
        spawn axi_ram_writer::AxiRamWriter<
        TEST_AXI_ADDR_W, TEST_AXI_DATA_W, TEST_AXI_ID_W, TEST_RAM_SIZE, TEST_ADDR_W, TEST_RAM_NUM_PARTITIONS,
        >(
            output_axi_aw_r, output_axi_w_r, output_axi_b_s,
            output_wr_req_s, output_wr_resp_r
        );

        spawn mem_reader::MemReader<TEST_AXI_DATA_W, TEST_AXI_ADDR_W, TEST_AXI_DEST_W, TEST_AXI_ID_W>(
            mem_rd_req_r, mem_rd_resp_s,
            input_axi_ar_s, input_axi_r_r
        );
        spawn mem_writer::MemWriter<
            TEST_AXI_ADDR_W, TEST_AXI_DATA_W, TEST_AXI_DEST_W, TEST_AXI_ID_W, TEST_WRITER_ID
        >(
            mem_wr_req_r, mem_wr_data_r,
            output_axi_aw_s, output_axi_w_s, output_axi_b_r,
            mem_wr_resp_s
        );

        spawn mem_writer_simple_arbiter::MemWriterSimpleArbiter<TEST_ADDR_W, TEST_AXI_DATA_W, u32:7>
        (
            n_mem_wr_req_r, n_mem_wr_data_r, n_mem_wr_resp_s,
            mem_wr_req_s, mem_wr_data_s, mem_wr_resp_r,
        );

        spawn mem_reader_simple_arbiter::MemReaderSimpleArbiter<TEST_ADDR_W, TEST_AXI_DATA_W, u32:3> (
            n_mem_rd_req_r, n_mem_rd_resp_s,
            mem_rd_req_s, mem_rd_resp_r,
        );

        spawn ram::RamModel<
        TEST_HT_RAM_DATA_W, TEST_HT_SIZE, TEST_HT_RAM_WORD_PARTITION_SIZE,
        TEST_HT_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_HT_RAM_INITIALIZED
        >(
            ht_ram_rd_req_r, ht_ram_rd_resp_s,
            ht_ram_wr_req_r, ht_ram_wr_resp_s
        );

        unroll_for! (i, _) : (u32, ()) in u32:0..u32:8 {
            spawn ram::RamModel<
            TEST_HB_RAM_DATA_W, TEST_HB_RAM_SIZE, TEST_HB_RAM_PARTITION_SIZE,
            TEST_HB_RAM_SIMULTANEOUS_RW_BEHAVIOR, TEST_HB_RAM_INITIALIZED
            >(
                hb_ram_rd_req_r[i], hb_ram_rd_resp_s[i],
                hb_ram_wr_req_r[i], hb_ram_wr_resp_s[i],
            );
        }(());

        spawn ram::RamModel<
        TEST_FSE_CTABLE_RAM_DATA_W, TEST_RAM_SIZE, TEST_FSE_TABLE_PARTITION_SIZE,
        TEST_RAM_SIMULTANEOUS_RW_BEHAVIOR, TEST_RAM_INITIALIZED,
        TEST_RAM_ASSERT_VALID_READ, TEST_FSE_TABLE_RAM_ADDR_W, TEST_FSE_CTABLE_RAM_NUM_PARTITIONS
        >(
            ll_ctable_ram_rd_req_r, ll_ctable_ram_rd_resp_s,
            ll_ctable_ram_wr_req_r, ll_ctable_ram_wr_resp_s,
        );

        spawn ram::RamModel<
        TEST_FSE_CTABLE_RAM_DATA_W, TEST_RAM_SIZE, TEST_FSE_TABLE_PARTITION_SIZE,
        TEST_RAM_SIMULTANEOUS_RW_BEHAVIOR, TEST_RAM_INITIALIZED,
        TEST_RAM_ASSERT_VALID_READ, TEST_FSE_TABLE_RAM_ADDR_W, TEST_FSE_CTABLE_RAM_NUM_PARTITIONS
        >(
            ml_ctable_ram_rd_req_r, ml_ctable_ram_rd_resp_s,
            ml_ctable_ram_wr_req_r, ml_ctable_ram_wr_resp_s,
        );

        spawn ram::RamModel<
        TEST_FSE_CTABLE_RAM_DATA_W, TEST_RAM_SIZE, TEST_FSE_TABLE_PARTITION_SIZE,
        TEST_RAM_SIMULTANEOUS_RW_BEHAVIOR, TEST_RAM_INITIALIZED,
        TEST_RAM_ASSERT_VALID_READ, TEST_FSE_TABLE_RAM_ADDR_W, TEST_FSE_CTABLE_RAM_NUM_PARTITIONS
        >(
            of_ctable_ram_rd_req_r, of_ctable_ram_rd_resp_s,
            of_ctable_ram_wr_req_r, of_ctable_ram_wr_resp_s,
        );

        spawn ram::RamModel<
        TEST_FSE_TTABLE_RAM_DATA_W, TEST_RAM_SIZE, TEST_FSE_TABLE_PARTITION_SIZE,
        TEST_RAM_SIMULTANEOUS_RW_BEHAVIOR, TEST_RAM_INITIALIZED,
        TEST_RAM_ASSERT_VALID_READ, TEST_FSE_TABLE_RAM_ADDR_W, TEST_FSE_TTABLE_RAM_NUM_PARTITIONS
        >(
            ll_ttable_ram_rd_req_r, ll_ttable_ram_rd_resp_s,
            ll_ttable_ram_wr_req_r, ll_ttable_ram_wr_resp_s,
        );

        spawn ram::RamModel<
        TEST_FSE_TTABLE_RAM_DATA_W, TEST_RAM_SIZE, TEST_FSE_TABLE_PARTITION_SIZE,
        TEST_RAM_SIMULTANEOUS_RW_BEHAVIOR, TEST_RAM_INITIALIZED,
        TEST_RAM_ASSERT_VALID_READ, TEST_FSE_TABLE_RAM_ADDR_W, TEST_FSE_TTABLE_RAM_NUM_PARTITIONS
        >(
            ml_ttable_ram_rd_req_r, ml_ttable_ram_rd_resp_s,
            ml_ttable_ram_wr_req_r, ml_ttable_ram_wr_resp_s,
        );

        spawn ram::RamModel<
        TEST_FSE_TTABLE_RAM_DATA_W, TEST_RAM_SIZE, TEST_FSE_TABLE_PARTITION_SIZE,
        TEST_RAM_SIMULTANEOUS_RW_BEHAVIOR, TEST_RAM_INITIALIZED,
        TEST_RAM_ASSERT_VALID_READ, TEST_FSE_TABLE_RAM_ADDR_W, TEST_FSE_TTABLE_RAM_NUM_PARTITIONS
        >(
            of_ttable_ram_rd_req_r, of_ttable_ram_rd_resp_s,
            of_ttable_ram_wr_req_r, of_ttable_ram_wr_resp_s,
        );

        spawn zstd_enc::ZstdEncoder<
            TEST_AXI_ADDR_W, TEST_AXI_DATA_W,
            TEST_RLE_HEURISTIC_SAMPLE_COUNT,
            TEST_HB_SIZE, TEST_HB_DATA_W, TEST_HB_OFFSET_W, TEST_HB_RAM_ADDR_W, TEST_HB_RAM_DATA_W, TEST_HB_RAM_NUM, TEST_HB_RAM_NUM_PARTITIONS,
            TEST_HT_SIZE, TEST_HT_KEY_W, TEST_HT_VALUE_W, TEST_HT_SIZE_W, TEST_HT_HASH_W, TEST_HT_RAM_DATA_W, TEST_HT_RAM_NUM_PARTITIONS,
            TEST_MIN_SEQ_LEN, TEST_LITERALS_BUFFER_AXI_ADDR, TEST_SEQUENCE_BUFFER_AXI_ADDR,
            TEST_FSE_TABLE_RAM_ADDR_W, TEST_FSE_CTABLE_RAM_DATA_W, TEST_FSE_TTABLE_RAM_DATA_W, TEST_FSE_CTABLE_RAM_NUM_PARTITIONS, TEST_FSE_TTABLE_RAM_NUM_PARTITIONS,
            TEST_FSE_BITSTREAM_BUFFER_W
        >(
            enc_req_r, enc_resp_s,
            n_mem_wr_req_s[0], n_mem_wr_data_s[0], n_mem_wr_resp_r[0],
            n_mem_wr_req_s[1], n_mem_wr_data_s[1], n_mem_wr_resp_r[1],
            n_mem_wr_req_s[2], n_mem_wr_data_s[2], n_mem_wr_resp_r[2],
            n_mem_wr_req_s[3], n_mem_wr_data_s[3], n_mem_wr_resp_r[3],
            n_mem_wr_req_s[4], n_mem_wr_data_s[4], n_mem_wr_resp_r[4],
            n_mem_wr_req_s[5], n_mem_wr_data_s[5], n_mem_wr_resp_r[5],
            n_mem_wr_req_s[6], n_mem_wr_data_s[6], n_mem_wr_resp_r[6],
            buf_mem_wr_req_s, buf_mem_wr_data_s, buf_mem_wr_resp_r,
            n_mem_rd_req_s[0], n_mem_rd_resp_r[0],
            n_mem_rd_req_s[1], n_mem_rd_resp_r[1],
            n_mem_rd_req_s[2], n_mem_rd_resp_r[2],
            buf_mem_rd_req_s, buf_mem_rd_resp_r,
            hb_ram_rd_req_s, hb_ram_rd_resp_r, hb_ram_wr_req_s, hb_ram_wr_resp_r,
            ht_ram_rd_req_s, ht_ram_rd_resp_r, ht_ram_wr_req_s, ht_ram_wr_resp_r,
            ml_ctable_ram_rd_req_s, ml_ctable_ram_rd_resp_r,
            ll_ctable_ram_rd_req_s, ll_ctable_ram_rd_resp_r,
            of_ctable_ram_rd_req_s, of_ctable_ram_rd_resp_r,
            ml_ttable_ram_rd_req_s, ml_ttable_ram_rd_resp_r,
            ll_ttable_ram_rd_req_s, ll_ttable_ram_rd_resp_r,
            of_ttable_ram_rd_req_s, of_ttable_ram_rd_resp_r
        );


        (
            write_fse_tables_req_r, write_fse_tables_resp_s,
            ll_ctable_ram_wr_req_s, ll_ctable_ram_wr_resp_r,
            of_ctable_ram_wr_req_s, of_ctable_ram_wr_resp_r,
            ml_ctable_ram_wr_req_s, ml_ctable_ram_wr_resp_r,
            ll_ttable_ram_wr_req_s, ll_ttable_ram_wr_resp_r,
            of_ttable_ram_wr_req_s, of_ttable_ram_wr_resp_r,
            ml_ttable_ram_wr_req_s,  ml_ttable_ram_wr_resp_r
        )
    }

    next(state: ()) {
        const CTMASK = all_ones!<uN[TEST_FSE_CTABLE_RAM_NUM_PARTITIONS]>();
        const TTMASK = all_ones!<uN[TEST_FSE_TTABLE_RAM_NUM_PARTITIONS]>();
        let tok = join();
        let (tok, _) = recv(tok, write_fse_tables_req_r);

        trace!("[TEST] Setting up the compression tables");

        let tok = for ((i, v), tok) in enumerate(sequence_encoder::OF_DEFAULT_CTABLE) {
            let tok = send(tok, of_ctable_ram_wr_req_s, TestCTableRamWrReq{addr: i, data: v, mask: CTMASK});
            let (tok, _) = recv(tok, of_ctable_ram_wr_resp_r);
            tok
        }(tok);

        let tok = for ((i, v), tok) in enumerate(sequence_encoder::LL_DEFAULT_CTABLE) {
            let tok = send(tok, ll_ctable_ram_wr_req_s, TestCTableRamWrReq{addr: i, data: v, mask: CTMASK});
            let (tok, _) = recv(tok, ll_ctable_ram_wr_resp_r);
            tok
        }(tok);

        let tok = for ((i, v), tok) in enumerate(sequence_encoder::ML_DEFAULT_CTABLE) {
            let tok = send(tok, ml_ctable_ram_wr_req_s, TestCTableRamWrReq{addr: i, data: v, mask: CTMASK});
            let (tok, _) = recv(tok, ml_ctable_ram_wr_resp_r);
            tok
        }(tok);

        trace!("[TEST] Setting up the transform tables");

        let tok = for ((i, tt), tok) in enumerate(sequence_encoder::OF_DEFAULT_TTABLE) {
            let tok = send(tok, of_ttable_ram_wr_req_s, TestTTableRamWrReq{addr: i, data: sequence_encoder::serialize_tt<TEST_FSE_TTABLE_RAM_DATA_W>(tt), mask: TTMASK});
            let (tok, _) = recv(tok, of_ttable_ram_wr_resp_r);
            tok
        }(tok);

        let tok = for ((i, tt), tok) in enumerate(sequence_encoder::LL_DEFAULT_TTABLE) {
            let tok = send(tok, ll_ttable_ram_wr_req_s, TestTTableRamWrReq{addr: i, data: sequence_encoder::serialize_tt<TEST_FSE_TTABLE_RAM_DATA_W>(tt), mask: TTMASK});
            let (tok, _) = recv(tok, ll_ttable_ram_wr_resp_r);
            tok
        }(tok);

        let tok = for ((i, tt), tok) in enumerate(sequence_encoder::ML_DEFAULT_TTABLE) {
            let tok = send(tok, ml_ttable_ram_wr_req_s, TestTTableRamWrReq{addr: i, data: sequence_encoder::serialize_tt<TEST_FSE_TTABLE_RAM_DATA_W>(tt), mask: TTMASK});
            let (tok, _) = recv(tok, ml_ttable_ram_wr_resp_r);
            tok
        }(tok);
        let tok = send(tok, write_fse_tables_resp_s, ());
    }
}


#[test_proc]
proc ZstdEncoderRawTest {
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
        let (input_wr_req_s, input_wr_req_r) = chan<TestWriteReq>("input_wr_req");
        let (input_wr_resp_s, input_wr_resp_r) = chan<TestWriteResp>("input_wr_resp");
        let (output_rd_req_s, output_rd_req_r) = chan<TestReadReq>("output_rd_req");
        let (output_rd_resp_s, output_rd_resp_r) = chan<TestReadResp>("output_rd_resp");
        let (enc_req_s, enc_req_r) = chan<TestZstdEncodeReq>("enc_req");
        let (enc_resp_s, enc_resp_r) = chan<TestZstdEncodeResp>("enc_resp");

        let (_, stub0_r) = chan<()>("stub0");
        let (stub1_s, _) = chan<()>("stub1");

        spawn ZstdEncoderTestBase (
            stub0_r, stub1_s,
            input_wr_req_r, input_wr_resp_s,
            output_rd_req_r, output_rd_resp_s,
            enc_req_r, enc_resp_s
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
                mask: all_ones!<uN[TEST_RAM_NUM_PARTITIONS]>(),
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
            params: zero!<zstd_enc::ZstdEncodeParams>()
        });
        let (tok, resp) = recv(tok, enc_resp_r);
        assert_eq(resp.status, ZstdEncodeRespStatus::OK);

        // read state of output RAM
        let tok = for ((i, expected_val), tok): ((u32, u32), token) in enumerate(TEST_EXPECTED) {
            let tok = send(tok, output_rd_req_s, TestReadReq {
                addr: i as Addr,
                mask: all_ones!<uN[TEST_RAM_NUM_PARTITIONS]>(),
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
        // IO for output RAM
        let (output_rd_req_s, output_rd_req_r) = chan<TestReadReq>("output_rd_req");
        let (output_rd_resp_s, output_rd_resp_r) = chan<TestReadResp>("output_rd_resp");
        let (output_wr_req_s, output_wr_req_r) = chan<TestWriteReq>("output_wr_req");
        let (output_wr_resp_s, output_wr_resp_r) = chan<TestWriteResp>("output_wr_resp");
        // IO for AxiRamWriter <-> MemWriter
        let (output_axi_aw_s, output_axi_aw_r) = chan<AxiAw>("output_axi_aw");
        let (output_axi_w_s, output_axi_w_r) = chan<AxiW>("output_axi_w");
        let (output_axi_b_s, output_axi_b_r) = chan<AxiB>("output_axi_b");
        // IO for MemWriter <-> Encoder
        let (mem_wr_req_s, mem_wr_req_r) = chan<TestMemWriterReq>("mem_wr_req");
        let (mem_wr_data_s, mem_wr_data_r) = chan<TestMemWriterData>("mem_wr_data");
        let (mem_wr_resp_s, mem_wr_resp_r) = chan<TestMemWriterResp>("mem_wr_resp");
        let (n_mem_wr_req_s, n_mem_wr_req_r) = chan<TestMemWriterReq, u32:1>[8]("n_req");
        let (n_mem_wr_data_s, n_mem_wr_data_r) = chan<TestMemWriterData, u32:1>[8]("n_data");
        let (n_mem_wr_resp_s, n_mem_wr_resp_r) = chan<TestMemWriterResp, u32:1>[8]("n_resp");
        let (n_mem_rd_req_s, n_mem_rd_req_r) = chan<TestMemReaderReq, u32:1>[4]("n_mem_rd_req");
        let (n_mem_rd_resp_s, n_mem_rd_resp_r) = chan<TestMemReaderResp, u32:1>[4]("n_mem_rd_resp");
        let (hb_ram_rd_req_s, _) = chan<TestHistoryBufferRamRdReq>[TEST_HB_RAM_NUM]("hb_ram_rd_req");
        let (_, hb_ram_rd_resp_r) = chan<TestHistoryBufferRamRdResp>[TEST_HB_RAM_NUM]("hb_ram_rd_resp");
        let (hb_ram_wr_req_s, _) = chan<TestHistoryBufferRamWrReq>[TEST_HB_RAM_NUM]("hb_ram_wr_req");
        let (_, hb_ram_wr_resp_r) = chan<TestHistoryBufferRamWrResp>[TEST_HB_RAM_NUM]("hb_ram_wr_resp");
        let (ht_ram_rd_req_s, _) = chan<TestHashTableRamRdReq>("ht_ram_rd_req");
        let (_, ht_ram_rd_resp_r) = chan<TestHashTableRamRdResp>("ht_ram_rd_resp");
        let (ht_ram_wr_req_s, _) = chan<TestHashTableRamWrReq>("ht_ram_wr_req");
        let (_, ht_ram_wr_resp_r) = chan<TestHashTableRamWrResp>("ht_ram_wr_resp");
        let (ml_ctable_ram_rd_req_s, _) = chan<TestCTableRamRdReq>("ml_ctable_ram_rd_req");
        let (_, ml_ctable_ram_rd_resp_r) = chan<TestCTableRamRdResp>("ml_ctable_ram_rd_resp");
        let (ll_ctable_ram_rd_req_s, _) = chan<TestCTableRamRdReq>("ll_ctable_ram_rd_req");
        let (_, ll_ctable_ram_rd_resp_r) = chan<TestCTableRamRdResp>("ll_ctable_ram_rd_resp");
        let (of_ctable_ram_rd_req_s, _) = chan<TestCTableRamRdReq>("of_ctable_ram_rd_req");
        let (_, of_ctable_ram_rd_resp_r) = chan<TestCTableRamRdResp>("of_ctable_ram_rd_resp");
        let (ml_ttable_ram_rd_req_s, _) = chan<TestTTableRamRdReq>("ml_ttable_ram_rd_req");
        let (_, ml_ttable_ram_rd_resp_r) = chan<TestTTableRamRdResp>("ml_ttable_ram_rd_resp");
        let (ll_ttable_ram_rd_req_s, _) = chan<TestTTableRamRdReq>("ll_ttable_ram_rd_req");
        let (_, ll_ttable_ram_rd_resp_r) = chan<TestTTableRamRdResp>("ll_ttable_ram_rd_resp");
        let (of_ttable_ram_rd_req_s, _) = chan<TestTTableRamRdReq>("of_ttable_ram_rd_req");
        let (_, of_ttable_ram_rd_resp_r) = chan<TestTTableRamRdResp>("of_ttable_ram_rd_resp");

        spawn MemReaderFaultResponder(
            mem_rd_req_r, mem_rd_resp_s,
        );
        spawn ram::RamModel<
            TEST_RAM_DATA_W, TEST_RAM_SIZE, TEST_RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_RW_BEHAVIOR, TEST_RAM_INITIALIZED,
            TEST_RAM_ASSERT_VALID_READ, TEST_ADDR_W
        >(
            output_rd_req_r, output_rd_resp_s,
            output_wr_req_r, output_wr_resp_s
        );
        spawn axi_ram_writer::AxiRamWriter<
            TEST_AXI_ADDR_W, TEST_AXI_DATA_W, TEST_AXI_ID_W, TEST_RAM_SIZE, TEST_ADDR_W, TEST_RAM_NUM_PARTITIONS,
        >(
            output_axi_aw_r, output_axi_w_r, output_axi_b_s,
            output_wr_req_s, output_wr_resp_r
        );
        spawn mem_writer::MemWriter<
            TEST_AXI_ADDR_W, TEST_AXI_DATA_W, TEST_AXI_DEST_W, TEST_AXI_ID_W, TEST_WRITER_ID
        >(
            mem_wr_req_r, mem_wr_data_r,
            output_axi_aw_s, output_axi_w_s, output_axi_b_r,
            mem_wr_resp_s
        );

        spawn mem_writer_simple_arbiter::MemWriterSimpleArbiter<TEST_ADDR_W, TEST_AXI_DATA_W, u32:8>
        (
            n_mem_wr_req_r, n_mem_wr_data_r, n_mem_wr_resp_s,
            mem_wr_req_s, mem_wr_data_s, mem_wr_resp_r,
        );

        spawn mem_reader_simple_arbiter::MemReaderSimpleArbiter<TEST_ADDR_W, TEST_AXI_DATA_W, u32:4> (
            n_mem_rd_req_r, n_mem_rd_resp_s,
            mem_rd_req_s, mem_rd_resp_r,
        );

        spawn zstd_enc::ZstdEncoder<
            TEST_AXI_ADDR_W, TEST_AXI_DATA_W,
            TEST_RLE_HEURISTIC_SAMPLE_COUNT,
            TEST_HB_SIZE, TEST_HB_DATA_W, TEST_HB_OFFSET_W, TEST_HB_RAM_ADDR_W, TEST_HB_RAM_DATA_W, TEST_HB_RAM_NUM, TEST_HB_RAM_NUM_PARTITIONS,
            TEST_HT_SIZE, TEST_HT_KEY_W, TEST_HT_VALUE_W, TEST_HT_SIZE_W, TEST_HT_HASH_W, TEST_HT_RAM_DATA_W, TEST_HT_RAM_NUM_PARTITIONS,
            TEST_MIN_SEQ_LEN, TEST_LITERALS_BUFFER_AXI_ADDR, TEST_SEQUENCE_BUFFER_AXI_ADDR,
            TEST_FSE_TABLE_RAM_ADDR_W, TEST_FSE_CTABLE_RAM_DATA_W, TEST_FSE_TTABLE_RAM_DATA_W, TEST_FSE_CTABLE_RAM_NUM_PARTITIONS, TEST_FSE_TTABLE_RAM_NUM_PARTITIONS,
            TEST_FSE_BITSTREAM_BUFFER_W
        >(
            enc_req_r, enc_resp_s,
            n_mem_wr_req_s[0], n_mem_wr_data_s[0], n_mem_wr_resp_r[0],
            n_mem_wr_req_s[1], n_mem_wr_data_s[1], n_mem_wr_resp_r[1],
            n_mem_wr_req_s[2], n_mem_wr_data_s[2], n_mem_wr_resp_r[2],
            n_mem_wr_req_s[3], n_mem_wr_data_s[3], n_mem_wr_resp_r[3],
            n_mem_wr_req_s[4], n_mem_wr_data_s[4], n_mem_wr_resp_r[4],
            n_mem_wr_req_s[5], n_mem_wr_data_s[5], n_mem_wr_resp_r[5],
            n_mem_wr_req_s[6], n_mem_wr_data_s[6], n_mem_wr_resp_r[6],
            n_mem_wr_req_s[7], n_mem_wr_data_s[7], n_mem_wr_resp_r[7],
            n_mem_rd_req_s[0], n_mem_rd_resp_r[0],
            n_mem_rd_req_s[1], n_mem_rd_resp_r[1],
            n_mem_rd_req_s[2], n_mem_rd_resp_r[2],
            n_mem_rd_req_s[3], n_mem_rd_resp_r[3],
            hb_ram_rd_req_s, hb_ram_rd_resp_r, hb_ram_wr_req_s, hb_ram_wr_resp_r,
            ht_ram_rd_req_s, ht_ram_rd_resp_r, ht_ram_wr_req_s, ht_ram_wr_resp_r,
            ml_ctable_ram_rd_req_s, ml_ctable_ram_rd_resp_r,
            ll_ctable_ram_rd_req_s, ll_ctable_ram_rd_resp_r,
            of_ctable_ram_rd_req_s, of_ctable_ram_rd_resp_r,
            ml_ttable_ram_rd_req_s, ml_ttable_ram_rd_resp_r,
            ll_ttable_ram_rd_req_s, ll_ttable_ram_rd_resp_r,
            of_ttable_ram_rd_req_s, of_ttable_ram_rd_resp_r
        );

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
            params: zero!<zstd_enc::ZstdEncodeParams>()
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
        // IO for AxiRamReader <-> MemReader
        let (input_axi_ar_s, input_axi_ar_r) = chan<AxiAr>("input_axi_ar");
        let (input_axi_r_s, input_axi_r_r) = chan<AxiR>("input_axi_r");
        // IO for MemWriter <-> Encoder
        let (mem_wr_req_s, mem_wr_req_r) = chan<TestMemWriterReq>("mem_wr_req");
        let (mem_wr_data_s, mem_wr_data_r) = chan<TestMemWriterData>("mem_wr_data");
        let (mem_wr_resp_s, mem_wr_resp_r) = chan<TestMemWriterResp>("mem_wr_resp");
        // IO for MemReader <-> Encoder
        let (mem_rd_req_s, mem_rd_req_r) = chan<TestMemReaderReq>("mem_rd_req");
        let (mem_rd_resp_s, mem_rd_resp_r) = chan<TestMemReaderResp>("mem_rd_resp");

        spawn ram::RamModel<
            TEST_RAM_DATA_W, TEST_RAM_SIZE, TEST_RAM_WORD_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_RW_BEHAVIOR, TEST_RAM_INITIALIZED, TEST_RAM_ASSERT_VALID_READ, TEST_AXI_ADDR_W
        >(
            input_rd_req_r, input_rd_resp_s,
            input_wr_req_r, input_wr_resp_s
        );



        spawn axi_ram_reader::AxiRamReader<
            TEST_AXI_ADDR_W, TEST_AXI_DATA_W, TEST_AXI_DEST_W, TEST_AXI_ID_W, TEST_RAM_SIZE
        >(
            input_axi_ar_r, input_axi_r_s,
            input_rd_req_s, input_rd_resp_r
        );

        spawn mem_reader::MemReader<TEST_AXI_DATA_W, TEST_AXI_ADDR_W, TEST_AXI_DEST_W, TEST_AXI_ID_W>(
            mem_rd_req_r, mem_rd_resp_s,
            input_axi_ar_s, input_axi_r_r
        );

        // faulty MemWriter
        spawn MemWriterFaultResponder(
            mem_wr_req_r, mem_wr_data_r, mem_wr_resp_s
        );

        let (n_mem_wr_req_s, n_mem_wr_req_r) = chan<TestMemWriterReq, u32:1>[8]("n_req");
        let (n_mem_wr_data_s, n_mem_wr_data_r) = chan<TestMemWriterData, u32:1>[8]("n_data");
        let (n_mem_wr_resp_s, n_mem_wr_resp_r) = chan<TestMemWriterResp, u32:1>[8]("n_resp");
        let (n_mem_rd_req_s, n_mem_rd_req_r) = chan<TestMemReaderReq, u32:1>[4]("n_mem_rd_req");
        let (n_mem_rd_resp_s, n_mem_rd_resp_r) = chan<TestMemReaderResp, u32:1>[4]("n_mem_rd_resp");
        let (hb_ram_rd_req_s, _) = chan<TestHistoryBufferRamRdReq>[TEST_HB_RAM_NUM]("hb_ram_rd_req");
        let (_, hb_ram_rd_resp_r) = chan<TestHistoryBufferRamRdResp>[TEST_HB_RAM_NUM]("hb_ram_rd_resp");
        let (hb_ram_wr_req_s, _) = chan<TestHistoryBufferRamWrReq>[TEST_HB_RAM_NUM]("hb_ram_wr_req");
        let (_, hb_ram_wr_resp_r) = chan<TestHistoryBufferRamWrResp>[TEST_HB_RAM_NUM]("hb_ram_wr_resp");
        let (ht_ram_rd_req_s, _) = chan<TestHashTableRamRdReq>("ht_ram_rd_req");
        let (_, ht_ram_rd_resp_r) = chan<TestHashTableRamRdResp>("ht_ram_rd_resp");
        let (ht_ram_wr_req_s, _) = chan<TestHashTableRamWrReq>("ht_ram_wr_req");
        let (_, ht_ram_wr_resp_r) = chan<TestHashTableRamWrResp>("ht_ram_wr_resp");
        let (ml_ctable_ram_rd_req_s, _) = chan<TestCTableRamRdReq>("ml_ctable_ram_rd_req");
        let (_, ml_ctable_ram_rd_resp_r) = chan<TestCTableRamRdResp>("ml_ctable_ram_rd_resp");
        let (ll_ctable_ram_rd_req_s, _) = chan<TestCTableRamRdReq>("ll_ctable_ram_rd_req");
        let (_, ll_ctable_ram_rd_resp_r) = chan<TestCTableRamRdResp>("ll_ctable_ram_rd_resp");
        let (of_ctable_ram_rd_req_s, _) = chan<TestCTableRamRdReq>("of_ctable_ram_rd_req");
        let (_, of_ctable_ram_rd_resp_r) = chan<TestCTableRamRdResp>("of_ctable_ram_rd_resp");
        let (ml_ttable_ram_rd_req_s, _) = chan<TestTTableRamRdReq>("ml_ttable_ram_rd_req");
        let (_, ml_ttable_ram_rd_resp_r) = chan<TestTTableRamRdResp>("ml_ttable_ram_rd_resp");
        let (ll_ttable_ram_rd_req_s, _) = chan<TestTTableRamRdReq>("ll_ttable_ram_rd_req");
        let (_, ll_ttable_ram_rd_resp_r) = chan<TestTTableRamRdResp>("ll_ttable_ram_rd_resp");
        let (of_ttable_ram_rd_req_s, _) = chan<TestTTableRamRdReq>("of_ttable_ram_rd_req");
        let (_, of_ttable_ram_rd_resp_r) = chan<TestTTableRamRdResp>("of_ttable_ram_rd_resp");

        spawn mem_writer_simple_arbiter::MemWriterSimpleArbiter<TEST_ADDR_W, TEST_AXI_DATA_W, u32:8>
        (
            n_mem_wr_req_r, n_mem_wr_data_r, n_mem_wr_resp_s,
            mem_wr_req_s, mem_wr_data_s, mem_wr_resp_r,
        );

        spawn mem_reader_simple_arbiter::MemReaderSimpleArbiter<TEST_ADDR_W, TEST_AXI_DATA_W, u32:4> (
            n_mem_rd_req_r, n_mem_rd_resp_s,
            mem_rd_req_s, mem_rd_resp_r,
        );

        spawn zstd_enc::ZstdEncoder<
            TEST_AXI_ADDR_W, TEST_AXI_DATA_W,
            TEST_RLE_HEURISTIC_SAMPLE_COUNT,
            TEST_HB_SIZE, TEST_HB_DATA_W, TEST_HB_OFFSET_W, TEST_HB_RAM_ADDR_W, TEST_HB_RAM_DATA_W, TEST_HB_RAM_NUM, TEST_HB_RAM_NUM_PARTITIONS,
            TEST_HT_SIZE, TEST_HT_KEY_W, TEST_HT_VALUE_W, TEST_HT_SIZE_W, TEST_HT_HASH_W, TEST_HT_RAM_DATA_W, TEST_HT_RAM_NUM_PARTITIONS,
            TEST_MIN_SEQ_LEN, TEST_LITERALS_BUFFER_AXI_ADDR, TEST_SEQUENCE_BUFFER_AXI_ADDR,
            TEST_FSE_TABLE_RAM_ADDR_W, TEST_FSE_CTABLE_RAM_DATA_W, TEST_FSE_TTABLE_RAM_DATA_W, TEST_FSE_CTABLE_RAM_NUM_PARTITIONS, TEST_FSE_TTABLE_RAM_NUM_PARTITIONS,
            TEST_FSE_BITSTREAM_BUFFER_W
        >(
            enc_req_r, enc_resp_s,
            n_mem_wr_req_s[0], n_mem_wr_data_s[0], n_mem_wr_resp_r[0],
            n_mem_wr_req_s[1], n_mem_wr_data_s[1], n_mem_wr_resp_r[1],
            n_mem_wr_req_s[2], n_mem_wr_data_s[2], n_mem_wr_resp_r[2],
            n_mem_wr_req_s[3], n_mem_wr_data_s[3], n_mem_wr_resp_r[3],
            n_mem_wr_req_s[4], n_mem_wr_data_s[4], n_mem_wr_resp_r[4],
            n_mem_wr_req_s[5], n_mem_wr_data_s[5], n_mem_wr_resp_r[5],
            n_mem_wr_req_s[6], n_mem_wr_data_s[6], n_mem_wr_resp_r[6],
            n_mem_wr_req_s[7], n_mem_wr_data_s[7], n_mem_wr_resp_r[7],
            n_mem_rd_req_s[0], n_mem_rd_resp_r[0],
            n_mem_rd_req_s[1], n_mem_rd_resp_r[1],
            n_mem_rd_req_s[2], n_mem_rd_resp_r[2],
            n_mem_rd_req_s[3], n_mem_rd_resp_r[3],
            hb_ram_rd_req_s, hb_ram_rd_resp_r, hb_ram_wr_req_s, hb_ram_wr_resp_r,
            ht_ram_rd_req_s, ht_ram_rd_resp_r, ht_ram_wr_req_s, ht_ram_wr_resp_r,
            ml_ctable_ram_rd_req_s, ml_ctable_ram_rd_resp_r,
            ll_ctable_ram_rd_req_s, ll_ctable_ram_rd_resp_r,
            of_ctable_ram_rd_req_s, of_ctable_ram_rd_resp_r,
            ml_ttable_ram_rd_req_s, ml_ttable_ram_rd_resp_r,
            ll_ttable_ram_rd_req_s, ll_ttable_ram_rd_resp_r,
            of_ttable_ram_rd_req_s, of_ttable_ram_rd_resp_r
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
                mask: all_ones!<uN[TEST_RAM_NUM_PARTITIONS]>(),
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
            params: zero!<zstd_enc::ZstdEncodeParams>()
        });
        let (tok, resp) = recv(tok, enc_resp_r);

        // expect error due to the faulty (on purpose) MemWriter
        assert_eq(resp.status, ZstdEncodeRespStatus::ERROR);

        send(join(), terminator, true);
    }
}

#[test_proc]
proc ZstdEncoderRleBlockTest {
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
        let (input_wr_req_s, input_wr_req_r) = chan<TestWriteReq>("input_wr_req");
        let (input_wr_resp_s, input_wr_resp_r) = chan<TestWriteResp>("input_wr_resp");
        let (output_rd_req_s, output_rd_req_r) = chan<TestReadReq>("output_rd_req");
        let (output_rd_resp_s, output_rd_resp_r) = chan<TestReadResp>("output_rd_resp");
        let (enc_req_s, enc_req_r) = chan<TestZstdEncodeReq>("enc_req");
        let (enc_resp_s, enc_resp_r) = chan<TestZstdEncodeResp>("enc_resp");
        let (_, stub0_r) = chan<()>("stub0");
        let (stub1_s, _) = chan<()>("stub1");

        spawn ZstdEncoderTestBase (
            stub0_r, stub1_s,
            input_wr_req_r, input_wr_resp_s,
            output_rd_req_r, output_rd_resp_s,
            enc_req_r, enc_resp_s
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
        let tok = for ((i, data), tok): ((u32, u32), token) in enumerate(TEST_RLE_INPUT_DATA) {
            let tok = send(tok, input_wr_req_s, TestWriteReq {
                addr: i as Addr,
                data: data,
                mask: all_ones!<uN[TEST_RAM_NUM_PARTITIONS]>(),
            });
            let (tok, _) = recv(tok, input_wr_resp_r);
            tok
        }(join());

        trace_fmt!("Request: encode data of size: {:#x} bytes", TEST_RLE_INPUT_SIZE_BYTES);
        // send request to encoder
        let tok = send(tok, enc_req_s, TestZstdEncodeReq{
            input_offset: uN[TEST_ADDR_W]:0,
            data_size: TEST_RLE_INPUT_SIZE_BYTES as uN[TEST_RAM_DATA_W],
            output_offset: uN[TEST_ADDR_W]:0,
            max_block_size: u32:32,
            params: zstd_enc::ZstdEncodeParams {
                enable_rle: true,
                enable_compressed: false
            }
        });
        let (tok, resp) = recv(tok, enc_resp_r);
        assert_eq(resp.status, ZstdEncodeRespStatus::OK);

        // read state of output RAM
        let tok = for ((i, expected_val), tok): ((u32, u32), token) in enumerate(TEST_RLE_EXPECTED) {
            let tok = send(tok, output_rd_req_s, TestReadReq {
                addr: i as Addr,
                mask: all_ones!<uN[TEST_RAM_NUM_PARTITIONS]>(),
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

#[test_proc]
proc ZstdEncoderRawLiteralsPredefinedSequencesTest {
    type AxiAr = axi::AxiAr<TEST_AXI_ADDR_W, TEST_AXI_ID_W>;
    type AxiR = axi::AxiR<TEST_AXI_DATA_W, TEST_AXI_ID_W>;
    type AxiAw = axi::AxiAw<TEST_AXI_ADDR_W, TEST_AXI_ID_W>;
    type AxiW = axi::AxiW<TEST_AXI_DATA_W, TEST_RAM_NUM_PARTITIONS>;
    type AxiB = axi::AxiB<TEST_AXI_ID_W>;

    write_fse_tables_req_s: chan<()> out;
    write_fse_tables_resp_r: chan<()> in;
    input_wr_req_s: chan<TestWriteReq> out;
    input_wr_resp_r: chan<TestWriteResp> in;
    output_rd_req_s: chan<TestReadReq> out;
    output_rd_resp_r: chan<TestReadResp> in;
    enc_req_s: chan<TestZstdEncodeReq> out;
    enc_resp_r: chan<TestZstdEncodeResp> in;
    terminator: chan<bool> out;

    init {  }

    config(terminator: chan<bool> out) {
        let (input_wr_req_s, input_wr_req_r) = chan<TestWriteReq>("input_wr_req");
        let (input_wr_resp_s, input_wr_resp_r) = chan<TestWriteResp>("input_wr_resp");
        let (output_rd_req_s, output_rd_req_r) = chan<TestReadReq>("output_rd_req");
        let (output_rd_resp_s, output_rd_resp_r) = chan<TestReadResp>("output_rd_resp");
        let (enc_req_s, enc_req_r) = chan<TestZstdEncodeReq>("enc_req");
        let (enc_resp_s, enc_resp_r) = chan<TestZstdEncodeResp>("enc_resp");
        let (write_fse_tables_req_s, write_fse_tables_req_r) = chan<()>("write_fse_tables_req");
        let (write_fse_tables_resp_s, write_fse_tables_resp_r) = chan<()>("write_fse_tables_resp");

        spawn ZstdEncoderTestBase (
            write_fse_tables_req_r, write_fse_tables_resp_s,
            input_wr_req_r, input_wr_resp_s,
            output_rd_req_r, output_rd_resp_s,
            enc_req_r, enc_resp_s
        );

        (
            write_fse_tables_req_s, write_fse_tables_resp_r,
            input_wr_req_s, input_wr_resp_r,
            output_rd_req_s, output_rd_resp_r,
            enc_req_s, enc_resp_r,
            terminator
        )
    }

    next(state: ()) {
        type Addr = bits[TEST_ADDR_W];

        // write input data to RAM
        let tok = for ((i, data), tok): ((u32, u32), token) in enumerate(TEST_COMPRESSED_PREDEFINED_INPUT) {
            let tok = send(tok, input_wr_req_s, TestWriteReq {
                addr: i as Addr,
                data: data,
                mask: all_ones!<uN[TEST_RAM_NUM_PARTITIONS]>(),
            });
            let (tok, _) = recv(tok, input_wr_resp_r);
            tok
        }(join());

        // write fse tables
        let tok = send(tok, write_fse_tables_req_s, ());
        let (tok, _) = recv(tok, write_fse_tables_resp_r);

        trace_fmt!("Request: encode data of size: {:#x} bytes", TEST_COMPRESSED_PREDEFINED_INPUT_SIZE_BYTES);
        // send request to encoder
        let tok = send(tok, enc_req_s, TestZstdEncodeReq{
            input_offset: uN[TEST_ADDR_W]:0,
            data_size: TEST_COMPRESSED_PREDEFINED_INPUT_SIZE_BYTES as uN[TEST_RAM_DATA_W],
            output_offset: uN[TEST_ADDR_W]:0,
            max_block_size: u32:32,
            params: zstd_enc::ZstdEncodeParams {
                enable_rle: false,
                enable_compressed: true
            }
        });
        let (tok, resp) = recv(tok, enc_resp_r);
        assert_eq(resp.status, ZstdEncodeRespStatus::OK);

        // read state of output RAM
        let tok = for ((i, expected_val), tok): ((u32, u32), token) in enumerate(TEST_COMPRESS_PREDEFINED_EXPECTED) {
            let tok = send(tok, output_rd_req_s, TestReadReq {
                addr: i as Addr,
                mask: all_ones!<uN[TEST_RAM_NUM_PARTITIONS]>(),
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
