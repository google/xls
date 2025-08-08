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

// This file contains implementation of MatchFinder

import std;

import xls.examples.ram;
import xls.modules.zstd.memory.axi;
import xls.modules.zstd.memory.axi_ram_reader;
import xls.modules.zstd.memory.mem_reader;
import xls.modules.zstd.memory.mem_writer;
import xls.modules.zstd.mem_writer_simple_arbiter;
import xls.modules.zstd.mem_reader_simple_arbiter;
import xls.modules.zstd.mem_copy;

type RawMemcopyBlockType = mem_copy::RawMemcopyBlockType;
type RawMemcopyStatus = mem_copy::RawMemcopyStatus;
type RawMemcopyReq = mem_copy::RawMemcopyReq;
type RawMemcopyResp = mem_copy::RawMemcopyResp;

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

        // TODO: Generalize the proc
        assert!(req.btype == RawMemcopyBlockType::RAW, "unsupported_literal_type");
        let btype = u2:0;

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

    init {}

    req_r: chan<RawMemcopyReq> in;
    resp_s: chan<RawMemcopyResp> out;

    lshwr_req_s: chan<HeaderWriterReq> out;
    lshwr_resp_r: chan<HeaderWriterResp> in;

    raw_req_s: chan<RawMemcopyReq> out;
    raw_resp_r: chan<RawMemcopyResp> in;

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

        (
            req_r, resp_s,
            lshwr_req_s, lshwr_resp_r,
            raw_req_s, raw_resp_r,
        )
    }

    next(state: ()) {
        let (req_tok, req) = recv(join(), req_r);
        let selected_btype = RawMemcopyBlockType::RAW;

        // Write Literasl Secion Header
        let lshw_req = match(selected_btype) {
            RawMemcopyBlockType::RAW => HeaderWriterReq {
                addr: req.out_addr,
                btype: selected_btype,
                regenerated_size: checked_cast<u20>(req.lit_cnt),
                compressed_size: u18:0,
            },
            _ => fail!("impossible_case_0", zero!<HeaderWriterReq>())
        };

        let lshwr_req_tok = send(req_tok, lshwr_req_s, lshw_req);
        let (lshwr_resp_tok, lshwr_resp) = recv(lshwr_req_tok, lshwr_resp_r);

        let lshwr_error = (lshwr_resp.status != LiteralSectionHeaderWriterStatus::OK);

        // Write Literals
        let new_req = RawMemcopyReq {
            lit_addr: req.lit_addr,
            lit_cnt: req.lit_cnt,
            out_addr: req.out_addr + lshwr_resp.length
        };

        let do_send_raw_req = !lshwr_error && (selected_btype == RawMemcopyBlockType::RAW);
        let raw_req_tok = send_if(lshwr_req_tok, raw_req_s, do_send_raw_req, new_req);

        let (raw_resp_tok, raw_resp) = recv_if(lshwr_resp_tok, raw_resp_r, do_send_raw_req, zero!<RawMemcopyResp>());
        let raw_error = if do_send_raw_req { raw_resp.status != RawMemcopyStatus::OK } else { false };

        let error = match(selected_btype) {
            RawMemcopyBlockType::RAW => lshwr_error || raw_error,
            _ => fail!("impossible_case_1", true)
        };
        let status = if error { Status::ERROR } else { Status::OK };
        let resp_tok = join(raw_resp_tok);

        let resp = RawMemcopyResp {
            btype: selected_btype,
            length: lshwr_resp.length + req.lit_cnt,
            status,
        };

        send(resp_tok, resp_s, resp);
    }
}

const TEST_MEM_READER_N = u32:1;
const TEST_MEM_WRITER_N = u32:2;

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

const TEST_LITERALS_CNT = u32:25;
const TEST_DATA = uN[TEST_RAM_DATA_W][4]:[
    u64:0x8877_6655_4433_2211, // 0x0
    u64:0xFFEE_DDCC_BBAA_0099,
    u64:0x0807_0605_0403_0201,
    u64:0x09,
];

#[test_proc]
proc LiteralsEncoderTest {
    type Req = RawMemcopyReq<TEST_ADDR_W>;
    type Resp = RawMemcopyResp<TEST_ADDR_W>;
    type Addr = uN[TEST_ADDR_W];
    type Length = uN[TEST_ADDR_W];

    type InputRamRdReq = ram::ReadReq<TEST_RAM_ADDR_W, TEST_RAM_NUM_PARTITIONS>;
    type InputRamRdResp = ram::ReadResp<TEST_RAM_DATA_W>;
    type InputRamWrReq = ram::WriteReq<TEST_RAM_ADDR_W, TEST_RAM_DATA_W, TEST_RAM_NUM_PARTITIONS>;
    type InputRamWrResp = ram::WriteResp;

    type MemReaderReq = mem_reader::MemReaderReq<TEST_ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<TEST_DATA_W, TEST_ADDR_W>;

    type MemWriterReq = mem_writer::MemWriterReq<TEST_ADDR_W>;
    type MemWriterData = mem_writer::MemWriterDataPacket<TEST_DATA_W, TEST_ADDR_W>;
    type MemWriterResp = mem_writer::MemWriterResp;

    type HeaderWriterReq = LiteralSectionHeaderWriterReq<TEST_ADDR_W>;
    type HeaderWriterResp = LiteralSectionHeaderWriterResp<TEST_ADDR_W>;

    type AxiAr = axi::AxiAr<TEST_ADDR_W, TEST_ID_W>;
    type AxiR = axi::AxiR<TEST_DATA_W, TEST_ID_W>;

    type AxiAw = axi::AxiAw<TEST_ADDR_W, TEST_ID_W>;
    type AxiW = axi::AxiW<TEST_DATA_W, TEST_DATA_W_DIV8>;
    type AxiB = axi::AxiB<TEST_ID_W>;

    type AxiAddr = uN[TEST_ADDR_W];
    type AxiData = uN[TEST_DATA_W];
    type AxiId = uN[TEST_ID_W];
    type AxiStrb = uN[TEST_DEST_W];
    type AxiLen = u8;
    type AxiSize = axi::AxiAxSize;
    type AxiBurst = axi::AxiAxBurst;
    type AxiWriteResp = axi::AxiWriteResp;

    terminator: chan<bool> out;

    req_s: chan<Req> out;
    resp_r: chan<Resp> in;

    input_ram_wr_req_s: chan<InputRamWrReq> out;
    input_ram_wr_resp_r: chan<InputRamWrResp> in;

    mem_axi_aw_r: chan<AxiAw> in;
    mem_axi_w_r: chan<AxiW> in;
    mem_axi_b_s: chan<AxiB> out;

    init {}

    config(terminator: chan<bool> out) {

        // Input Access

        let (input_ram_rd_req_s, input_ram_rd_req_r) = chan<InputRamRdReq>("input_ram_rd_req");
        let (input_ram_rd_resp_s, input_ram_rd_resp_r) = chan<InputRamRdResp>("input_ram_rd_resp");
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

        let (mem_axi_aw_s, mem_axi_aw_r) = chan<AxiAw>("mem_axi_aw");
        let (mem_axi_w_s, mem_axi_w_r) = chan<AxiW>("mem_axi_w");
        let (mem_axi_b_s, mem_axi_b_r) = chan<AxiB>("mem_axi_b");

        let (mem_wr_req_s, mem_wr_req_r) = chan<MemWriterReq>("mem_wr_req");
        let (mem_wr_data_s, mem_wr_data_r) = chan<MemWriterData>("mem_wr_data");
        let (mem_wr_resp_s, mem_wr_resp_r) = chan<MemWriterResp>("mem_wr_resp");

        spawn mem_writer::MemWriter<TEST_ADDR_W, TEST_DATA_W, TEST_DEST_W, TEST_ID_W, TEST_WRITER_ID>(
            mem_wr_req_r, mem_wr_data_r,
            mem_axi_aw_s, mem_axi_w_s, mem_axi_b_r,
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
        );

        (
            terminator,
            req_s, resp_r,
            input_ram_wr_req_s, input_ram_wr_resp_r,
            mem_axi_aw_r, mem_axi_w_r, mem_axi_b_s,
        )
    }

    next(state: ()) {
        let tok = join();

        // Input Data
        let tok = for ((i, test_data), tok) in enumerate(TEST_DATA) {
            let ram_wr_req = InputRamWrReq {
                addr: i as uN[TEST_ADDR_W],
                data: test_data,
                mask: !uN[TEST_RAM_NUM_PARTITIONS]:0,
            };

            let tok = send(tok, input_ram_wr_req_s, ram_wr_req);
            let (tok, _) = recv(tok, input_ram_wr_resp_r);
            tok
        }(tok);

        // Literals Encoder Request
        let req = Req {
            lit_addr: Addr:0,
            lit_cnt: TEST_LITERALS_CNT,
            out_addr: Addr:0x100,
        };
        let tok = send(tok, req_s, req);

        // Header Request
        let (tok, aw) = recv(tok, mem_axi_aw_r);
        assert_eq(aw, AxiAw {
            id: AxiId:0x1,
            addr: AxiAddr:0x100,
            size: AxiSize::MAX_8B_TRANSFER,
            len: AxiLen:0x0,
            burst: AxiBurst::INCR
        });

        let (tok, w) = recv(tok, mem_axi_w_r);
        assert_eq(w, AxiW {
            data: AxiData:0xc8,
            strb: AxiStrb:0x1,
            last: true
        });
        let tok = send(tok, mem_axi_b_s, AxiB {
            resp: AxiWriteResp::OKAY,
            id: aw.id,
        });

        // Copying Literals
        let (tok, aw) = recv(tok, mem_axi_aw_r);
        assert_eq(aw, AxiAw {
            id: AxiId:2,
            addr: AxiAddr:0x100,
            size: AxiSize::MAX_8B_TRANSFER,
            len: AxiLen:3,
            burst: AxiBurst::INCR,
        });
        let (tok, w) = recv(tok, mem_axi_w_r);
        assert_eq(w, AxiW {
            data: AxiData:0x7766_5544_3322_1100,
            strb: AxiStrb:0xfe,
            last: false,
        });
        let (tok, w) = recv(tok, mem_axi_w_r);
        assert_eq(w, AxiW {
            data: AxiData:0xeedd_ccbb_aa00_9988,
            strb: AxiStrb:0xff,
            last: false,
        });
        let (tok, w) = recv(tok, mem_axi_w_r);
        assert_eq(w, AxiW {
            data: AxiData:0x706_0504_0302_01ff,
            strb: AxiStrb:0xff,
            last: false
        });
        let (tok, w) = recv(tok, mem_axi_w_r);
        assert_eq(w, AxiW {
            data: AxiData:0x908,
            strb: AxiStrb:0x3,
            last: true,
        });
        let tok = send(tok, mem_axi_b_s, AxiB {
            resp: AxiWriteResp::OKAY,
            id: AxiId:2,
        });

        // Literals Encoder Response
        let (tok, resp) = recv(tok, resp_r);
        assert_eq(resp, RawMemcopyResp {
            status: RawMemcopyStatus::OK,
            btype: RawMemcopyBlockType::RAW,
            length: Length:0x1a
        });

        send(tok, terminator, true);
    }
}
