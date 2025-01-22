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


import std;
import xls.examples.ram;
import xls.modules.zstd.common;
import xls.modules.zstd.refilling_shift_buffer;
import xls.modules.zstd.fse_table_creator;

pub proc RleLookupDecoder<
    AXI_DATA_W: u32,
    FSE_RAM_DATA_W: u32, FSE_RAM_ADDR_W: u32, FSE_RAM_NUM_PARTITIONS: u32,
    SB_LENGTH_W: u32 = {refilling_shift_buffer::length_width(AXI_DATA_W)},
> {
    type Req = common::LookupDecoderReq;
    type Resp = common::LookupDecoderResp;

    type Status = common::LookupDecoderStatus;

    type FseRamWrReq = ram::WriteReq<FSE_RAM_ADDR_W, FSE_RAM_DATA_W, FSE_RAM_NUM_PARTITIONS>;
    type FseRamWrResp = ram::WriteResp;

    type SBOutput = refilling_shift_buffer::RefillingShiftBufferOutput<AXI_DATA_W, SB_LENGTH_W>;
    type SBCtrl = refilling_shift_buffer::RefillingShiftBufferCtrl<SB_LENGTH_W>;

    req_r: chan<Req> in;
    resp_s: chan<Resp> out;

    fse_wr_req_s: chan<FseRamWrReq> out;
    fse_wr_resp_r: chan<FseRamWrResp> in;

    buffer_ctrl_s: chan<SBCtrl> out;
    buffer_data_r: chan<SBOutput> in;

    init {}

    config(
        req_r: chan<Req> in,
        resp_s: chan<Resp> out,

        fse_wr_req_s: chan<FseRamWrReq> out,
        fse_wr_resp_r: chan<FseRamWrResp> in,

        buffer_ctrl_s: chan<SBCtrl> out,
        buffer_data_r: chan<SBOutput> in,
    ) {
        (
            req_r, resp_s,
            fse_wr_req_s, fse_wr_resp_r,
            buffer_ctrl_s, buffer_data_r,
        )
    }

    next(state: ()) {
        let tok = join();
        // receive request
        let (tok, _) = recv(tok, req_r);
        // ask shift buffer for one byte
        let tok = send(tok, buffer_ctrl_s, SBCtrl {
            length: uN[SB_LENGTH_W]:8
        });
        // receive byte
        let (tok, byte) = recv(tok, buffer_data_r);
        // write byte to first location in memory

        let fse_wr_req = FseRamWrReq {
            addr: uN[FSE_RAM_ADDR_W]:0,
            data: fse_table_creator::fse_record_to_bits(common::FseTableRecord {
                symbol: byte.data as u8,
                num_of_bits: u8:0,
                base: u16:0,
            }),
            mask: all_ones!<uN[FSE_RAM_NUM_PARTITIONS]>(),
        };
        trace_fmt!("RLE RAM REQUEST: {:#x}", fse_wr_req);

        let tok = send(tok, fse_wr_req_s, fse_wr_req);
        // receive write response
        let (tok, _) = recv(tok, fse_wr_resp_r);
        // send response
        let tok = send(tok, resp_s, Resp {
            status: if byte.error { Status::ERROR } else { Status::OK },
            accuracy_log: common::FseAccuracyLog:0,
        });
    }
}


const TEST_AXI_DATA_W = u32:64;
const TEST_SB_LENGTH_W = refilling_shift_buffer::length_width(TEST_AXI_DATA_W);

const TEST_FSE_RAM_DATA_W = u32:32;
const TEST_FSE_RAM_SIZE = u32:1 << common::FSE_MAX_ACCURACY_LOG;
const TEST_FSE_RAM_ADDR_W = std::clog2(TEST_FSE_RAM_SIZE);
const TEST_FSE_RAM_WORD_PARTITION_SIZE = TEST_FSE_RAM_DATA_W;
const TEST_FSE_RAM_NUM_PARTITIONS = ram::num_partitions(TEST_FSE_RAM_WORD_PARTITION_SIZE, TEST_FSE_RAM_DATA_W);

#[test_proc]
proc RleLookupDecoderTest {
    type Req = common::LookupDecoderReq;
    type Resp = common::LookupDecoderResp;
    type Status = common::LookupDecoderStatus;

    type FseRamRdReq = ram::ReadReq<TEST_FSE_RAM_ADDR_W, TEST_FSE_RAM_NUM_PARTITIONS>;
    type FseRamRdResp = ram::ReadResp<TEST_FSE_RAM_DATA_W>;
    type FseRamWrReq = ram::WriteReq<TEST_FSE_RAM_ADDR_W, TEST_FSE_RAM_DATA_W, TEST_FSE_RAM_NUM_PARTITIONS>;
    type FseRamWrResp = ram::WriteResp;

    type SBOutput = refilling_shift_buffer::RefillingShiftBufferOutput<TEST_AXI_DATA_W, TEST_SB_LENGTH_W>;
    type SBCtrl = refilling_shift_buffer::RefillingShiftBufferCtrl<TEST_SB_LENGTH_W>;

    terminator: chan<bool> out;

    req_s: chan<Req> out;
    resp_r: chan<Resp> in;

    fse_wr_req_r: chan<FseRamWrReq> in;
    fse_wr_resp_s: chan<FseRamWrResp> out;

    buffer_ctrl_r: chan<SBCtrl> in;
    buffer_data_s: chan<SBOutput> out;

    init {}

    config(terminator: chan<bool> out) {

        let (req_s, req_r) = chan<Req>("req");
        let (resp_s, resp_r) = chan<Resp>("resp");
        let (fse_wr_req_s, fse_wr_req_r) = chan<FseRamWrReq>("fse_wr_req");
        let (fse_wr_resp_s, fse_wr_resp_r) = chan<FseRamWrResp>("fse_wr_resp");
        let (buffer_ctrl_s, buffer_ctrl_r) = chan<SBCtrl>("buffer_ctrl");
        let (buffer_data_s, buffer_data_r) = chan<SBOutput>("buffer_data");

        spawn RleLookupDecoder<
            TEST_AXI_DATA_W,
            TEST_FSE_RAM_DATA_W, TEST_FSE_RAM_ADDR_W, TEST_FSE_RAM_NUM_PARTITIONS,
        >(
            req_r, resp_s,
            fse_wr_req_s, fse_wr_resp_r,
            buffer_ctrl_s, buffer_data_r,
        );

        (
            terminator,
            req_s, resp_r,
            fse_wr_req_r, fse_wr_resp_s,
            buffer_ctrl_r, buffer_data_s,
        )
    }

    next(_: ()) {
        let tok = join();

        let tok = send(tok, req_s, Req {});
        let (tok, buf_req) = recv(tok, buffer_ctrl_r);
        assert_eq(buf_req, SBCtrl {
            length: uN[TEST_SB_LENGTH_W]:8
        });
        let tok = send(tok, buffer_data_s, SBOutput {
            length: uN[TEST_SB_LENGTH_W]:8,
            data: uN[TEST_AXI_DATA_W]:0xC5,
            error: false,
        });
        let (tok, ram_req) = recv(tok, fse_wr_req_r);
        assert_eq(ram_req, FseRamWrReq {
            addr: uN[TEST_FSE_RAM_ADDR_W]:0,
            data: u32:0xC5,
            mask: uN[TEST_FSE_RAM_NUM_PARTITIONS]:0x1,
        });
        let tok = send(tok, fse_wr_resp_s, FseRamWrResp {});
        let (tok, resp) = recv(tok, resp_r);
        assert_eq(resp, Resp {
            status: Status::OK,
            accuracy_log: common::FseAccuracyLog:0,
        });
        send(tok, terminator, true);
    }
}
