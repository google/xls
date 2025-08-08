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

// This file contains implementation of LiteralsEncoder

import std;

import xls.examples.ram;
import xls.modules.zstd.memory.axi;
import xls.modules.zstd.memory.axi_ram_reader;
import xls.modules.zstd.memory.axi_ram_writer;
import xls.modules.zstd.memory.mem_reader;
import xls.modules.zstd.memory.mem_writer;
import xls.modules.zstd.mem_writer_simple_arbiter;
import xls.modules.zstd.mem_reader_simple_arbiter;
import xls.modules.zstd.mem_copy;
import xls.modules.zstd.rle_block_encoder;

type RawMemcopyBlockType = mem_copy::RawMemcopyBlockType;
type RawMemcopyStatus = mem_copy::RawMemcopyStatus;
type RawMemcopyReq = mem_copy::RawMemcopyReq;
type RawMemcopyResp = mem_copy::RawMemcopyResp;

// This parameter controls the sampling in RleBlockEncoder.
// It's the number of samples read from memory for a quick check
// before full scan is done to verify if all bytes are of the same value.
// TODO verify if it should be moved to ZstdEncoder
const RLE_HEURISTIC_SAMPLE_COUNT = u32:8;

enum LiteralSectionHeaderWriterStatus: u1 {
    OK = 0,
    ERROR = 1
}

struct LiteralSectionHeaderWriterReq<ADDR_W: u32> {
    addr: uN[ADDR_W],
    btype: RawMemcopyBlockType,
    regenerated_size: u20,
    compressed_size: u18,
}

struct LiteralSectionHeaderWriterResp<ADDR_W: u32> {
    status: LiteralSectionHeaderWriterStatus,
    length: uN[ADDR_W],
}

proc LiteralSectionHeaderWriter<ADDR_W: u32, DATA_W: u32> {
    type Req = LiteralSectionHeaderWriterReq<ADDR_W>;
    type Resp = LiteralSectionHeaderWriterResp<ADDR_W>;
    type Status = LiteralSectionHeaderWriterStatus;

    type MemWriterData = mem_writer::MemWriterDataPacket<DATA_W, ADDR_W>;
    type MemWriterReq = mem_writer::MemWriterReq<ADDR_W>;
    type MemWriterResp = mem_writer::MemWriterResp;
    type MemWriterStatus = mem_writer::MemWriterRespStatus;

    type Data = uN[DATA_W];
    type Length = uN[ADDR_W];

    req_r: chan<Req> in;
    resp_s: chan<Resp> out;

    mem_wr_req_s: chan<MemWriterReq> out;
    mem_wr_data_s: chan<MemWriterData> out;
    mem_wr_resp_r: chan<MemWriterResp> in;

    init {}

    config(
        req_r: chan<Req> in,
        resp_s: chan<Resp> out,

        mem_wr_req_s: chan<MemWriterReq> out,
        mem_wr_data_s: chan<MemWriterData> out,
        mem_wr_resp_r: chan<MemWriterResp> in,
    ) {
        (
            req_r, resp_s,
            mem_wr_req_s, mem_wr_data_s, mem_wr_resp_r,
        )
    }

    next(state: ()) {
        let (req_tok, req) = recv(join(), req_r);

        // TODO: Implement for Compressed_Literals_Block and Treeless_Literals_Block as well.
        assert!(req.btype == RawMemcopyBlockType::RAW || req.btype == RawMemcopyBlockType::RLE, "unsupported_literal_type");
        let btype = match (req.btype) {
            RawMemcopyBlockType::RAW => u2:0,
            RawMemcopyBlockType::RLE => u2:1,
            _ => u2:0,
        };


        let (data, length) = if req.regenerated_size <= u20:0x1F { // regenerated_size <= 31
            let size_format = u1:0;
            let regenerated_size = checked_cast<u5>(req.regenerated_size);
            let data = (regenerated_size ++ size_format ++ btype) as Data;

            (data, Length:1)

        } else if req.regenerated_size <= u20:0xFFF { // regenerated size <= 4095
            let size_format = u2:0b01;
            let regenerated_size = checked_cast<u12>(req.regenerated_size);
            let data = (regenerated_size ++ size_format ++ btype) as Data;

            (data, Length:2)

        } else {
            // regenerated size <= 1048575
            // No need to check the value as the whole range of u20 is covered.
            let size_format = u2:0b11;
            let regenerated_size = checked_cast<u20>(req.regenerated_size);
            let data = (regenerated_size ++ size_format ++ btype) as Data;

            (data, Length:3)
        };

        let mem_wr_req = MemWriterReq { addr: req.addr, length };
        let mem_wr_tok = send(req_tok, mem_wr_req_s, mem_wr_req);

        let mem_wr_data = MemWriterData { data, length, last: true };
        let mem_wr_data_tok = send(mem_wr_tok, mem_wr_data_s, mem_wr_data);

        let (tok, resp) = recv(mem_wr_data_tok, mem_wr_resp_r);
        let status = if resp.status == MemWriterStatus::OKAY { Status::OK } else { Status::ERROR };

        let resp = Resp { length, status };
        send(mem_wr_data_tok, resp_s, resp);
    }
}

proc LiteralsEncoder<ADDR_W: u32, DATA_W: u32> {
    type MemReaderReq = mem_reader::MemReaderReq<ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<DATA_W, ADDR_W>;
    type MemReaderStatus = mem_reader::MemReaderStatus;

    type MemWriterReq = mem_writer::MemWriterReq<ADDR_W>;
    type MemWriterData = mem_writer::MemWriterDataPacket<DATA_W, ADDR_W>;
    type MemWriterResp = mem_writer::MemWriterResp;
    type MemWriterStatus = mem_writer::MemWriterRespStatus;

    type HeaderWriterReq = LiteralSectionHeaderWriterReq<ADDR_W>;
    type HeaderWriterResp = LiteralSectionHeaderWriterResp<ADDR_W>;

    type Status = RawMemcopyStatus;

    type RleBlockEncoderReq = rle_block_encoder::RleBlockEncoderReq<ADDR_W>;
    type RleBlockEncoderResp = rle_block_encoder::RleBlockEncoderResp<ADDR_W>;
    type RleBlockEncoderStatus = rle_block_encoder::RleBlockEncoderStatus;

    init {}

    req_r: chan<RawMemcopyReq> in;
    resp_s: chan<RawMemcopyResp> out;

    lshwr_req_s: chan<HeaderWriterReq> out;
    lshwr_resp_r: chan<HeaderWriterResp> in;

    raw_req_s: chan<RawMemcopyReq> out;
    raw_resp_r: chan<RawMemcopyResp> in;

    rle_req_s: chan<RleBlockEncoderReq> out;
    rle_resp_r: chan<RleBlockEncoderResp> in;

    rle_mem_wr_req_s: chan<MemWriterReq> out;
    rle_mem_wr_data_s: chan<MemWriterData> out;
    rle_mem_wr_resp_r: chan<MemWriterResp> in;

    config(
        req_r: chan<RawMemcopyReq> in,
        resp_s: chan<RawMemcopyResp> out,

        raw_mem_rd_req_s: chan<MemReaderReq> out,
        raw_mem_rd_resp_r: chan<MemReaderResp> in,

        lshwr_mem_wr_req_s: chan<MemWriterReq> out,
        lshwr_mem_wr_data_s: chan<MemWriterData> out,
        lshwr_mem_wr_resp_r: chan<MemWriterResp> in,

        raw_mem_wr_req_s: chan<MemWriterReq> out,
        raw_mem_wr_data_s: chan<MemWriterData> out,
        raw_mem_wr_resp_r: chan<MemWriterResp> in,

        rle_mem_rd_req_s: chan<MemReaderReq> out,
        rle_mem_rd_resp_r: chan<MemReaderResp> in,

        rle_mem_wr_req_s: chan<MemWriterReq> out,
        rle_mem_wr_data_s: chan<MemWriterData> out,
        rle_mem_wr_resp_r: chan<MemWriterResp> in,
    ) {

        let (lshwr_req_s, lshwr_req_r) = chan<HeaderWriterReq>("lshwr_req");
        let (lshwr_resp_s, lshwr_resp_r) = chan<HeaderWriterResp>("lshwr_resp");

        spawn LiteralSectionHeaderWriter<ADDR_W, DATA_W>(
            lshwr_req_r, lshwr_resp_s,
            lshwr_mem_wr_req_s, lshwr_mem_wr_data_s, lshwr_mem_wr_resp_r,
        );

        let (raw_req_s, raw_req_r) = chan<RawMemcopyReq>("raw_req");
        let (raw_resp_s, raw_resp_r) = chan<RawMemcopyResp>("raw_resp");

        spawn mem_copy::RawMemcopy<ADDR_W, DATA_W>(
            raw_req_r, raw_resp_s,
            raw_mem_rd_req_s, raw_mem_rd_resp_r,
            raw_mem_wr_req_s, raw_mem_wr_data_s, raw_mem_wr_resp_r,
        );

        let (rle_req_s, rle_req_r) = chan<RleBlockEncoderReq>("rle_req");
        let (rle_resp_s, rle_resp_r) = chan<RleBlockEncoderResp>("rle_resp");

        spawn rle_block_encoder::RleBlockEncoder<ADDR_W, DATA_W, ADDR_W, RLE_HEURISTIC_SAMPLE_COUNT>
        (
            rle_req_r, rle_resp_s,
            rle_mem_rd_req_s, rle_mem_rd_resp_r
        );

        (
            req_r, resp_s,
            lshwr_req_s, lshwr_resp_r,
            raw_req_s, raw_resp_r,
            rle_req_s, rle_resp_r,
            rle_mem_wr_req_s, rle_mem_wr_data_s, rle_mem_wr_resp_r,
        )
    }

    next(state: ()) {
        let (tok, req) = recv(join(), req_r);

        // step 1: choose block type
        let tok = send(tok, rle_req_s, RleBlockEncoderReq {
            addr: req.lit_addr,
            length: req.lit_cnt as uN[ADDR_W]
        });

        trace_fmt!("sent RleBlockEncoder req: {:#x}",RleBlockEncoderReq {
            addr: req.lit_addr,
            length: req.lit_cnt as uN[ADDR_W]
        } );
        let (tok, rle_resp) = recv(tok, rle_resp_r);
        trace_fmt!("received RleBlockEncoder resp: {:#x}", rle_resp);
        let (btype, literals_stream_size) = if rle_resp.status == RleBlockEncoderStatus::OK {
            (RawMemcopyBlockType::RLE, u32:1)
        } else {
            (RawMemcopyBlockType::RAW, req.lit_cnt)
        };


        // step2: Write Literals Section Header
        let lshw_req = match(btype) {
            RawMemcopyBlockType::RAW => HeaderWriterReq {
                addr: req.out_addr,
                btype,
                regenerated_size: checked_cast<u20>(req.lit_cnt),
                compressed_size: u18:0,
            },
            RawMemcopyBlockType::RLE => HeaderWriterReq {
                addr: req.out_addr,
                btype,
                regenerated_size: checked_cast<u20>(req.lit_cnt),
                compressed_size: u18:0
            },
            _ => fail!("impossible_case_0", zero!<HeaderWriterReq>())
        };

        trace_fmt!("sending header write req: {:#x}", lshw_req);
        let tok = send(tok, lshwr_req_s, lshw_req);
        let (tok, lshwr_resp) = recv(tok, lshwr_resp_r);

        let lshwr_error = (lshwr_resp.status != LiteralSectionHeaderWriterStatus::OK);
        let literals_out_addr = req.out_addr + lshwr_resp.length;

        // step3: Write Literals Stream
        let (tok, status, literals_stream_size) = match (btype, lshwr_error) {
            (RawMemcopyBlockType::RLE, false) => {
                let tok = send(tok, rle_mem_wr_req_s, MemWriterReq {
                    addr: literals_out_addr, length: uN[ADDR_W]:1
                });
                let tok = send(tok, rle_mem_wr_data_s, MemWriterData {
                    data: rle_resp.symbol as uN[DATA_W],
                    length: uN[ADDR_W]:1,
                    last: true
                });
                trace_fmt!("writing rle symbol: {:#x} size: {})", rle_resp.symbol, literals_stream_size);
                let (tok, rle_resp) = recv(tok, rle_mem_wr_resp_r);
                let status =  if rle_resp.status == MemWriterStatus::OKAY { Status::OK } else { Status::ERROR };
                (tok, status, uN[ADDR_W]:1)
            },
            (RawMemcopyBlockType::RAW, false) => {
                let tok = send(tok, raw_req_s, RawMemcopyReq {
                    lit_addr: req.lit_addr,
                    lit_cnt: req.lit_cnt,
                    out_addr: literals_out_addr
                });
                let (tok, memcpy_resp) = recv(tok, raw_resp_r);

                let status = if memcpy_resp.status == RawMemcopyStatus::OK { Status::OK } else { Status::ERROR };
                (tok, status, req.lit_cnt)
            },
            (_, false) => {
                trace_fmt!("Unsupported literal type");
                (tok, Status::ERROR, uN[ADDR_W]:0)
            },
            (_, true) => {
                trace_fmt!("Writing literals section header failed");
                (tok, Status::ERROR, uN[ADDR_W]:0)
            },
        };

        let length = lshwr_resp.length + literals_stream_size;
        let resp = RawMemcopyResp { btype, length, status };
        send(tok, resp_s, resp);
    }
}

const TEST_MEM_READER_N = u32:2;
const TEST_MEM_WRITER_N = u32:4; // LiteralsEncoder uses 3, the test uses 1.

const TEST_ADDR_W = u32:32;
const TEST_DATA_W = u32:64;
const TEST_DATA_W_DIV8 = TEST_DATA_W / u32:8;
const TEST_DEST_W = u32:8;
const TEST_ID_W = u32:8;
const TEST_WRITER_ID = u32:1;

const TEST_RAM_DATA_W = TEST_DATA_W;
const TEST_RAM_SIZE = u32:1024;
const TEST_RAM_ADDR_W = TEST_ADDR_W;
const TEST_RAM_PARTITION_SIZE = u32:8;
const TEST_RAM_NUM_PARTITIONS = ram::num_partitions(TEST_RAM_PARTITION_SIZE, TEST_RAM_DATA_W);
const TEST_RAM_SIMULTANEOUS_RW_BEHAVIOR = ram::SimultaneousReadWriteBehavior::READ_BEFORE_WRITE;
const TEST_RAM_INITIALIZED = true;
const TEST_RAM_ASSERT_VALID_READ = true;

const TEST_CASES = [
    (
        u32:25, // number of bytes in input
        [ // input data
            u64:0x8877_6655_4433_2211,
            u64:0xFFEE_DDCC_BBAA_0099,
            u64:0x0807_0605_0403_0201,
            u64:0x09,
            u64:0x0,
            u64:0x0,
            u64:0x0,
        ],
        RawMemcopyBlockType::RAW,
        u32:26, // number of bytes in output
        [
            // expected output
            // Literals_Section_Header (8 bits) == 0xc8
            //   - Literals_Block_Type (2 bits) == 0 (Raw_Literals_block)
            //   - Size_format (1 bit) == 0
            //   - Regenerated_Size (5 bits) == 0x19
            //   - [Compressed_Size] - not present
            u64:0x7766_5544_3322_11_c8,
            u64:0xeedd_ccbb_aa00_9988,
            u64:0x706_0504_0302_01ff,
            u64:0x908,
            u64:0x0,
            u64:0x0,
            u64:0x0,
        ]
    ),
    (
        u32:47,
        [
            u64:0x8877_6655_4433_2211,
            u64:0xFFEE_DDCC_BBAA_0099,
            u64:0x0807_0605_0403_0201,
            u64:0x100f_0e0d_0c0b_0a09,
            u64:0x1817_1615_1413_1211,
            u64:0x001f_1e1d_1c1b_1a19,
            u64:0x0,
        ],
        RawMemcopyBlockType::RAW,
        u32:49,
        [
            // Literals_Section_Header (16 bits) ==  0x02f4
            //   - Literals_Block_Type (2 bits) == 0 (Raw_Literals_block)
            //   - Size_format (2 bits) == 1
            //   - Regenerated_Size (12 bits) == 0x2f
            //   - [Compressed_Size] - not present
            u64:0x6655_4433_2211_02f4,
            u64:0xDDCC_BBAA_0099_8877,
            u64:0x0605_0403_0201_FFEE,
            u64:0x0e0d_0c0b_0a09_0807,
            u64:0x1615_1413_1211_100f,
            u64:0x1e1d_1c1b_1a19_1817,
            u64:0x1f,
        ]
    ),
    (
        u32:1,
        [
            u64:0x11,
            u64:0x0,
            u64:0x0,
            u64:0x0,
            u64:0x0,
            u64:0x0,
            u64:0x0,
        ],
        RawMemcopyBlockType::RLE,
        u32:2,
        [
            // Literals_Section_Header (8 bits) == 0x09
            //   - Literals_Block_Type (2 bits) == 1 (RLE_Literals_block)
            //   - Size_format (1 bit) == 0
            //   - Regenerated_Size (5 bits) == 0x01
            //   - [Compressed_Size] - not present
            u64:0x1109,
            u64:0x0,
            u64:0x0,
            u64:0x0,
            u64:0x0,
            u64:0x0,
            u64:0x0,
        ]
    ),
    (
        u32:25,
        [
            u64:0x1111_1111_1111_1111,
            u64:0x1111_1111_1111_1111,
            u64:0x1111_1111_1111_1111,
            u64:0x11,
            u64:0x0,
            u64:0x0,
            u64:0x0,
        ],
        RawMemcopyBlockType::RLE,
        u32:2,
        [
            // expected output
            // Literals_Section_Header (8 bits) == 0xc9
            //   - Literals_Block_Type (2 bits) == 1 (RLE_Literals_block)
            //   - Size_format (1 bit) == 0
            //   - Regenerated_Size (5 bits) == 0x19
            //   - [Compressed_Size] - not present
            u64:0x11c9,
            u64:0x0,
            u64:0x0,
            u64:0x0,
            u64:0x0,
            u64:0x0,
            u64:0x0,
        ]
    ),
    (
        u32:45,
        [
            u64:0x5555_5555_5555_5555,
            u64:0x5555_5555_5555_5555,
            u64:0x5555_5555_5555_5555,
            u64:0x5555_5555_5555_5555,
            u64:0x5555_5555_5555_5555,
            u64:0x0000_0055_5555_5555,
            u64:0x0,
        ],
        RawMemcopyBlockType::RLE,
        u32:3,
        [
            // Literals_Section_Header (16 bits) == 0x2d5
            //   - Literals_Block_Type (2 bits) == 1 (RLE_Literals_block)
            //   - Size_format (2 bits) == 1
            //   - Regenerated_Size (12 bits) == 0x2d
            //   - [Compressed_Size] - not present
            u64:0x55_02d5,
            u64:0x0,
            u64:0x0,
            u64:0x0,
            u64:0x0,
            u64:0x0,
            u64:0x0,
        ]
    ),
];

// Test the case with size > 4095
const LARGE_TEST_BYTES = u32:5000;
const LARGE_TEST_VALUES = [
    (
        // value to write
        u64:0xfa,
        // expected type
        RawMemcopyBlockType::RAW,
        // expected number of bytes
        LARGE_TEST_BYTES + u32:3,
        // expected first word value
        // Literals_Section_Header (24 bits) == 0x01_388c
        //   - Literals_Block_Type (2 bits) == 0 (Raw_Literals_block)
        //   - Size_format (2 bits) == 1
        //   - Regenerated_Size (20 bits) == 0x1388
        //   - [Compressed_Size] - not present
        u64:0xfa01_388c
    ),
    (
        u64:0x2222_2222_2222_2222,
        RawMemcopyBlockType::RLE,
        u32:4, // 3 bytes for the header, one byte for repeated 0x22
        // Literals_Section_Header (24 bits) == 0x01_388d
        //   - Literals_Block_Type (2 bits) == 1 (Raw_Literals_block)
        //   - Size_format (2 bits) == 1
        //   - Regenerated_Size (20 bits) == 0x1388
        //   - [Compressed_Size] - not present
        u64:0x2201_388d
    ),
];

#[test_proc]
proc LiteralsEncoderTest {
    type Req = RawMemcopyReq<TEST_ADDR_W>;
    type Resp = RawMemcopyResp<TEST_ADDR_W>;
    type Addr = uN[TEST_ADDR_W];
    type Length = uN[TEST_ADDR_W];
    type Data = uN[TEST_DATA_W];

    type InputRamRdReq = ram::ReadReq<TEST_RAM_ADDR_W, TEST_RAM_NUM_PARTITIONS>;
    type InputRamRdResp = ram::ReadResp<TEST_RAM_DATA_W>;
    type InputRamWrReq = ram::WriteReq<TEST_RAM_ADDR_W, TEST_RAM_DATA_W, TEST_RAM_NUM_PARTITIONS>;
    type InputRamWrResp = ram::WriteResp;

    type OutputRamRdReq = ram::ReadReq<TEST_RAM_ADDR_W, TEST_RAM_NUM_PARTITIONS>;
    type OutputRamRdResp = ram::ReadResp<TEST_RAM_DATA_W>;
    type OutputRamWrReq = ram::WriteReq<TEST_RAM_ADDR_W, TEST_RAM_DATA_W, TEST_RAM_NUM_PARTITIONS>;
    type OutputRamWrResp = ram::WriteResp;

    type MemReaderReq = mem_reader::MemReaderReq<TEST_ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<TEST_DATA_W, TEST_ADDR_W>;

    type MemWriterReq = mem_writer::MemWriterReq<TEST_ADDR_W>;
    type MemWriterData = mem_writer::MemWriterDataPacket<TEST_DATA_W, TEST_ADDR_W>;
    type MemWriterResp = mem_writer::MemWriterResp;

    type AxiAr = axi::AxiAr<TEST_ADDR_W, TEST_ID_W>;
    type AxiR = axi::AxiR<TEST_DATA_W, TEST_ID_W>;

    type AxiAw = axi::AxiAw<TEST_ADDR_W, TEST_ID_W>;
    type AxiW = axi::AxiW<TEST_DATA_W, TEST_DATA_W_DIV8>;
    type AxiB = axi::AxiB<TEST_ID_W>;

    terminator: chan<bool> out;

    req_s: chan<Req> out;
    resp_r: chan<Resp> in;

    input_ram_wr_req_s: chan<InputRamWrReq> out;
    input_ram_wr_resp_r: chan<InputRamWrResp> in;

    output_ram_rd_req_s: chan<OutputRamRdReq> out;
    output_ram_rd_resp_r: chan<OutputRamRdResp> in;

    output_mem_wr_req_s: chan<MemWriterReq> out;
    output_mem_wr_data_s: chan<MemWriterData> out;
    output_mem_wr_resp_r: chan<MemWriterResp> in;

    init {}

    config(terminator: chan<bool> out) {

        // RamModel <-> AxiRamReader channels
        let (input_ram_rd_req_s, input_ram_rd_req_r) = chan<InputRamRdReq>("input_ram_rd_req");
        let (input_ram_rd_resp_s, input_ram_rd_resp_r) = chan<InputRamRdResp>("input_ram_rd_resp");

        // Test writes the input to RamModel
        let (input_ram_wr_req_s, input_ram_wr_req_r) = chan<InputRamWrReq>("input_ram_wr_req");
        let (input_ram_wr_resp_s, input_ram_wr_resp_r) = chan<InputRamWrResp>("input_ram_wr_resp");

        spawn ram::RamModel<
            TEST_RAM_DATA_W, TEST_RAM_SIZE, TEST_RAM_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_RW_BEHAVIOR, TEST_RAM_INITIALIZED,
            TEST_RAM_ASSERT_VALID_READ, TEST_RAM_ADDR_W,
        >(
            input_ram_rd_req_r, input_ram_rd_resp_s,
            input_ram_wr_req_r, input_ram_wr_resp_s,
        );

        // MemReader <-> AxiRamReader channels
        let (mem_axi_ar_s, mem_axi_ar_r) = chan<AxiAr>("mem_axi_ar");
        let (mem_axi_r_s, mem_axi_r_r) = chan<AxiR>("mem_axi_r");

        spawn axi_ram_reader::AxiRamReader<
            TEST_ADDR_W, TEST_DATA_W,
            TEST_DEST_W, TEST_ID_W,
            TEST_RAM_SIZE,
        >(
            mem_axi_ar_r, mem_axi_r_s,
            input_ram_rd_req_s, input_ram_rd_resp_r,
        );

        // MemReader <-> MemReadSimpleArbiter channels
        let (mem_rd_req_s, mem_rd_req_r) = chan<MemReaderReq>("mem_rd_req");
        let (mem_rd_resp_s, mem_rd_resp_r) = chan<MemReaderResp>("mem_rd_resp");

        spawn mem_reader::MemReader<
            TEST_DATA_W, TEST_ADDR_W, TEST_DEST_W, TEST_ID_W,
        >(
            mem_rd_req_r, mem_rd_resp_s,
            mem_axi_ar_s, mem_axi_r_r,
        );

        let (n_mem_rd_req_s, n_mem_rd_req_r) = chan<MemReaderReq>[TEST_MEM_READER_N]("n_mem_rd_req");
        let (n_mem_rd_resp_s, n_mem_rd_resp_r) = chan<MemReaderResp>[TEST_MEM_READER_N]("n_mem_rd_resp");

        spawn mem_reader_simple_arbiter::MemReaderSimpleArbiter<TEST_ADDR_W, TEST_DATA_W, TEST_MEM_READER_N> (
            n_mem_rd_req_r, n_mem_rd_resp_s,
            mem_rd_req_s, mem_rd_resp_r,
        );

        // Output Access

        // RamModel <-> test output interface
        let (output_ram_rd_req_s, output_ram_rd_req_r) = chan<OutputRamRdReq>("output_ram_rd_req");
        let (output_ram_rd_resp_s, output_ram_rd_resp_r) = chan<OutputRamRdResp>("output_ram_rd_resp");

        // RamModel <-> AxiRamWriter
        let (output_ram_wr_req_s, output_ram_wr_req_r) = chan<OutputRamWrReq>("output_ram_wr_req");
        let (output_ram_wr_resp_s, output_ram_wr_resp_r) = chan<OutputRamWrResp>("output_ram_wr_resp");

        spawn ram::RamModel<
            TEST_RAM_DATA_W, TEST_RAM_SIZE, TEST_RAM_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_RW_BEHAVIOR, TEST_RAM_INITIALIZED,
            TEST_RAM_ASSERT_VALID_READ, TEST_RAM_ADDR_W,
        >(
            output_ram_rd_req_r, output_ram_rd_resp_s,
            output_ram_wr_req_r, output_ram_wr_resp_s,
        );

        // MemWriter <-> AxiRamWriter channels
        let (output_ram_axi_aw_s, output_ram_axi_aw_r) = chan<AxiAw>("output_ram_axi_aw");
        let (output_ram_axi_w_s, output_ram_axi_w_r) = chan<AxiW>("output_ram_axi_w");
        let (output_ram_axi_b_s, output_ram_axi_b_r) = chan<AxiB>("output_ram_axi_b");

        spawn axi_ram_writer::AxiRamWriter<
            TEST_ADDR_W, TEST_DATA_W, TEST_ID_W, TEST_RAM_SIZE, TEST_RAM_ADDR_W
        >(
            output_ram_axi_aw_r, output_ram_axi_w_r, output_ram_axi_b_s,
            output_ram_wr_req_s, output_ram_wr_resp_r
        );

        let (mem_wr_req_s, mem_wr_req_r) = chan<MemWriterReq>("mem_wr_req");
        let (mem_wr_data_s, mem_wr_data_r) = chan<MemWriterData>("mem_wr_data");
        let (mem_wr_resp_s, mem_wr_resp_r) = chan<MemWriterResp>("mem_wr_resp");

        spawn mem_writer::MemWriter<TEST_ADDR_W, TEST_DATA_W, TEST_DEST_W, TEST_ID_W, TEST_WRITER_ID>(
            mem_wr_req_r, mem_wr_data_r,
            output_ram_axi_aw_s, output_ram_axi_w_s, output_ram_axi_b_r,
            mem_wr_resp_s,
        );

        let (n_mem_wr_req_s, n_mem_wr_req_r) = chan<MemWriterReq>[TEST_MEM_WRITER_N]("n_mem_wr_req");
        let (n_mem_wr_data_s, n_mem_wr_data_r) = chan<MemWriterData>[TEST_MEM_WRITER_N]("n_mem_wr_data");
        let (n_mem_wr_resp_s, n_mem_wr_resp_r) = chan<MemWriterResp>[TEST_MEM_WRITER_N]("n_mem_wr_resp");

        spawn mem_writer_simple_arbiter::MemWriterSimpleArbiter<TEST_ADDR_W, TEST_DATA_W, TEST_MEM_WRITER_N> (
            n_mem_wr_req_r, n_mem_wr_data_r, n_mem_wr_resp_s,
            mem_wr_req_s, mem_wr_data_s, mem_wr_resp_r,
        );

        // Literals Encoder

        let (req_s, req_r) = chan<Req>("req");
        let (resp_s, resp_r) = chan<Resp>("resp");

        spawn LiteralsEncoder<TEST_ADDR_W, TEST_DATA_W>(
            req_r, resp_s,
            n_mem_rd_req_s[0], n_mem_rd_resp_r[0],
            n_mem_wr_req_s[0], n_mem_wr_data_s[0], n_mem_wr_resp_r[0],
            n_mem_wr_req_s[1], n_mem_wr_data_s[1], n_mem_wr_resp_r[1],
            n_mem_rd_req_s[1], n_mem_rd_resp_r[1],
            n_mem_wr_req_s[2], n_mem_wr_data_s[2], n_mem_wr_resp_r[2],
        );

        (
            terminator,
            req_s, resp_r,
            input_ram_wr_req_s, input_ram_wr_resp_r,
            output_ram_rd_req_s, output_ram_rd_resp_r,
            // MemWriter interface to write zeros to the output memory before each test case
            n_mem_wr_req_s[3], n_mem_wr_data_s[3], n_mem_wr_resp_r[3]
        )
    }

    next(state: ()) {
        let tok = join();

        let tok = for ((i, (input_bytes, input_data, btype, expected_bytes, expected_data)), tok) in enumerate(TEST_CASES) {
            trace_fmt!("test case: {}", i);

            // write input data
            let tok = for ((j, input_data_word), tok) in enumerate(input_data) {
                let ram_wr_req = InputRamWrReq {
                    addr: j as uN[TEST_ADDR_W],
                    data: input_data_word,
                    mask: !uN[TEST_RAM_NUM_PARTITIONS]:0,
                };

                let tok = send(tok, input_ram_wr_req_s, ram_wr_req);
                let (tok, _) = recv(tok, input_ram_wr_resp_r);
                tok
            }(tok);

            // Literals Encoder Request
            let req = Req {
                lit_addr: Addr:0,
                lit_cnt: input_bytes,
                out_addr: Addr:0,
            };
            let tok = send(tok, req_s, req);

            // Literals Encoder Response
            let (tok, resp) = recv(tok, resp_r);
            assert_eq(resp, RawMemcopyResp {
                status: RawMemcopyStatus::OK,
                btype: btype,
                length: expected_bytes as Length
            });

            // read output, compare with the expected values
            let tok = for ((j, expected_data_word), tok) in enumerate(expected_data) {
                let ram_rd_req = OutputRamRdReq {
                    addr: j as uN[TEST_ADDR_W],
                    mask: !uN[TEST_RAM_NUM_PARTITIONS]:0,
                };

                let tok = send(tok, output_ram_rd_req_s, ram_rd_req);
                let (tok, resp) = recv(tok, output_ram_rd_resp_r);
                trace_fmt!("read data: {:#x}", resp.data);
                trace_fmt!("expected data: {:#x}", expected_data_word);
                assert_eq(resp.data, expected_data_word);
                tok
            }(tok);

            // clear the output memory
            let mem_wr_req = MemWriterReq { addr: Addr:0, length: TEST_RAM_SIZE as uN[TEST_ADDR_W] };
            let tok = send(tok, output_mem_wr_req_s, mem_wr_req);
            for (_, tok) in range(u32:0, (TEST_RAM_SIZE/TEST_DATA_W_DIV8)-u32:1) {
                let mem_wr_data = MemWriterData {data: Data:0, length: TEST_DATA_W_DIV8, last: false };
                let tok = send(tok, output_mem_wr_data_s, mem_wr_data);
                tok
            }(tok);
            let mem_wr_data = MemWriterData {data: Data:0, length: TEST_DATA_W_DIV8, last: true };
            let tok = send(tok, output_mem_wr_data_s, mem_wr_data);
            let (tok, _) = recv(tok, output_mem_wr_resp_r);

            tok
        }(tok);

        let tok = for ((_, (value, btype, result_size, expected_data_word)), tok) in enumerate(LARGE_TEST_VALUES) {

            // write input data
            let tok = for (j, tok) in range(u32:0 , LARGE_TEST_BYTES/TEST_DATA_W_DIV8) {
                let ram_wr_req = InputRamWrReq {
                    addr: j as Addr,
                    data: value as Data,
                    mask: all_ones!<uN[TEST_RAM_NUM_PARTITIONS]>(),
                };

                let tok = send(tok, input_ram_wr_req_s, ram_wr_req);
                let (tok, _) = recv(tok, input_ram_wr_resp_r);
                tok
            }(tok);

            // Literals Encoder Request
            let req = Req {
                lit_addr: Addr:0,
                lit_cnt: LARGE_TEST_BYTES,
                out_addr: Addr:0,
            };
            let tok = send(tok, req_s, req);

            // Literals Encoder Response
            let (tok, resp) = recv(tok, resp_r);
            assert_eq(resp, RawMemcopyResp {
                status: RawMemcopyStatus::OK,
                btype: btype,
                length: result_size as Length
            });

            let ram_rd_req = OutputRamRdReq {
                addr: Addr:0,
                mask: !uN[TEST_RAM_NUM_PARTITIONS]:0,
            };

            let tok = send(tok, output_ram_rd_req_s, ram_rd_req);
            let (tok, resp) = recv(tok, output_ram_rd_resp_r);
            // The large test asserts the first value read only,
            // which contains the header and the beginning of data.
            assert_eq(resp.data, expected_data_word);
            tok
        }(tok);


        send(tok, terminator, true);
    }
}
