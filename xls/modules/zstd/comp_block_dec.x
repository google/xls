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
import xls.modules.zstd.memory.axi;
import xls.modules.zstd.memory.mem_reader;
import xls.modules.zstd.fse_proba_freq_dec;
import xls.modules.zstd.comp_block_header_dec;


struct CompressBlockDecoderReq { }
struct CompressBlockDecoderResp { }

struct CompressBlockDecoderControlState { }

proc CompressBlockDecoderControl {
    type Req = CompressBlockDecoderReq;
    type Resp = CompressBlockDecoderResp;
    type State = CompressBlockDecoderControlState;

    type HeaderDecoderReq = comp_block_header_dec::CompressBlockHeaderDecoderReq;
    type HeaderDecoderResp = comp_block_header_dec::CompressBlockHeaderDecoderResp;

    req_r: chan<Req> in;
    resp_s: chan<Resp> out;
    cmph_req_s: chan<HeaderDecoderReq> out;
    cmph_resp_r: chan<HeaderDecoderResp> in;

    init { zero!<State>() }

    config(
        req_r: chan<Req> in,
        resp_s: chan<Resp> out,
        cmph_req_s: chan<HeaderDecoderReq> out,
        cmph_resp_r: chan<HeaderDecoderResp> in,
    ) {
        (req_r, resp_s, cmph_req_s, cmph_resp_r)
    }

    next (state: State) {
        let tok = join();

        send_if(tok, resp_s, false, zero!<Resp>());
        send_if(tok, cmph_req_s, false, zero!<HeaderDecoderReq>());

        recv_if(tok, req_r, false, zero!<Req>());
        recv_if(tok, cmph_resp_r, false, zero!<HeaderDecoderResp>());

        state
    }
}

proc CompressBlockDecoder<
    // AXI parameters
    AXI_DATA_W: u32, AXI_ADDR_W: u32, AXI_ID_W: u32, AXI_DEST_W: u32,

    // FSE proba
    FSE_PROBA_DIST_W: u32 = {u32:16},
    FSE_PROBA_MAX_DISTS: u32 = {u32:256},

    // constants
    FSE_DEF_PROBA_DIST_RAM_DATA_W: u32 = {FSE_PROBA_DIST_W},
    FSE_DEF_PROBA_DIST_RAM_SIZE: u32 = {FSE_PROBA_MAX_DISTS},
    FSE_DEF_PROBA_DIST_ADDR_W: u32 = {std::clog2(FSE_DEF_PROBA_DIST_RAM_SIZE)},
    FSE_DEF_PROBA_DIST_RAM_WORD_PARTITION_SIZE: u32 = { FSE_DEF_PROBA_DIST_RAM_DATA_W },
    FSE_DEF_PROBA_DIST_RAM_NUM_PARTITIONS: u32 = { ram::num_partitions(FSE_DEF_PROBA_DIST_RAM_WORD_PARTITION_SIZE, FSE_DEF_PROBA_DIST_RAM_DATA_W) },
    AXI_DATA_W_DIV8: u32 = {AXI_DATA_W / u32:8},
> {
    type Req = CompressBlockDecoderReq;
    type Resp = CompressBlockDecoderResp;

    type MemReaderReq  = mem_reader::MemReaderReq<AXI_ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<AXI_DATA_W, AXI_ADDR_W>;

    type MemAxiAr = axi::AxiAr<AXI_ADDR_W, AXI_ID_W>;
    type MemAxiR = axi::AxiR<AXI_DATA_W, AXI_ID_W>;
    type MemAxiAw = axi::AxiAw<AXI_ADDR_W, AXI_ID_W>;
    type MemAxiW = axi::AxiW<AXI_DATA_W, AXI_DATA_W_DIV8>;
    type MemAxiB = axi::AxiB<AXI_ID_W>;

    type HeaderDecoderReq = comp_block_header_dec::CompressBlockHeaderDecoderReq;
    type HeaderDecoderResp = comp_block_header_dec::CompressBlockHeaderDecoderResp;

    //type ShiftBufferInput = shift_buffer::ShiftBufferInput<DATA_WIDTH, LENGTH_WIDTH>;
    //type ShiftBufferCtrl = shift_buffer::ShiftBufferCtrl<LENGTH_WIDTH>;
    //type ShiftBufferOutput = shift_buffer::ShiftBufferOutput<DATA_WIDTH, LENGTH_WIDTH>;

    type DpdRamRdReq = ram::ReadReq<FSE_DEF_PROBA_DIST_ADDR_W, FSE_DEF_PROBA_DIST_RAM_NUM_PARTITIONS>;
    type DpdRamRdResp = ram::ReadResp<FSE_DEF_PROBA_DIST_RAM_DATA_W>;
    type DpdRamWrReq = ram::WriteReq<FSE_DEF_PROBA_DIST_ADDR_W, FSE_DEF_PROBA_DIST_RAM_DATA_W, FSE_DEF_PROBA_DIST_RAM_NUM_PARTITIONS>;
    type DpdRamWrResp = ram::WriteResp;

    init {}
    next(state: ()) {}

    config(
        req_r: chan<Req> in,
        resp_s: chan<Resp> out,

        // AXI Compressed Block Header Decoder (manager)
        cmph_axi_ar_s: chan<MemAxiAr> out,
        cmph_axi_r_r: chan<MemAxiR> in,

        // AXI Fse Probability Frequence Decoder (manager)
        //pfd_axi_ar_s: chan<MemAxiAr> out,
        //pfd_axi_r_r: chan<MemAxiR> in,

    ) {

        const CHANNEL_DEPTH = u32:1;

        // Compress Block Header Memory Reader

        let (cmph_mem_rd_req_s,  cmph_mem_rd_req_r) = chan<MemReaderReq, CHANNEL_DEPTH>("cmph_mem_rd_req");
        let (cmph_mem_rd_resp_s, cmph_mem_rd_resp_r) = chan<MemReaderResp, CHANNEL_DEPTH>("cmph_mem_rd_resp");

        spawn mem_reader::MemReader<AXI_DATA_W, AXI_ADDR_W, AXI_DEST_W, AXI_ID_W, CHANNEL_DEPTH>(
           cmph_mem_rd_req_r, cmph_mem_rd_resp_s,
           cmph_axi_ar_s, cmph_axi_r_r,
        );

        let (cmph_req_s, cmph_req_r) = chan<HeaderDecoderReq>("cmph_dec_req");
        let (cmph_resp_s, cmph_resp_r) = chan<HeaderDecoderResp>("cmph_dec_resp");

        spawn comp_block_header_dec::CompressBlockHeaderDecoder<AXI_DATA_W, AXI_ADDR_W>(
            cmph_mem_rd_req_s, cmph_mem_rd_resp_r,
            cmph_req_r, cmph_resp_s
        );

        // FseProbaFreqDecoder channels
        //
        // This part of the decoder is used to prepare FSE lookups if they
        // are provided in the encoded ZSTD bitstream (FSE_Compressed_Mode).
        // This part of the design sould take the input from the input buffer
        // located on the system bus, and decode the probability frequencies
        // into probability distribution. The outpud data should be stored in
        // a local RAM. One chalange related to this part of the code is how
        // to us a single instance of the proc to prepare three different FSE tables
        // (for Offsets, Matches, and Literals Lenghts).

        // spawn mem_reader::MemReader<
        //     AXI_DATA_W, AXI_ADDR_W, AXI_DEST_W, AXI_ID_W, CHANNEL_DEPTH
        // >(
        //     pfd_mem_rd_req_r, pfd_mem_rd_resp_s,
        //     pfd_axi_ar_s, pfd_axi_r_r,
        // );
        //
        // spawn shift_buffer::ShiftBuffer<
        //     SHIFT_BUFFER_DATA_W, SHIFT_BUFFER_LENGTH_W
        // >(buff_data_r, ctrl_r, out_s);
        //
        //
        // spawn FseProbaFreqDecoder<
        //     FSE_DEF_PROBA_DIST_RAM_DATA_W,
        //     FSE_DEF_PROBA_DIST_RAM_SIZE,
        //     FSE_DEF_PROBA_DIST_RAM_WORD_PARTITION_SIZE
        // >(
        //     req_r, resp_s,
        //     buff_in_ctrl_s, buff_out_data_r,
        //     rd_req_s, rd_resp_r, wr_req_s, wr_resp_r
        // );
        //
        // let (dpd_rd_req_r, dpd_rd_req_s) = chan<DpdRamRdReq>("rd_req");
        // let (dpd_rd_req_r, dpd_rd_req_s) = chan<DpdRamRdResp>("rd_resp");
        // let (dpd_wr_req_r, dpd_wr_req_s) = chan<DpdRamWrReq>("wr_req");
        // let (dpd_wr_resp_r, dpd_wr_resp_s) = chan<DpdRamWrResp>("wr_resp");
        //
        // spawn ram::RamModel<
        //     FSE_PROBA_DATA_W,
        //     FSE_PROBA_RAM_SIZE,
        //     FSE_PROBA_RAM_WORD_PARTITION_SIZE
        // >(
        //     rd_req_r, rd_resp_s, wr_req_r, wr_resp_s
        // );

        //spawn CompressBlockDecoderControl<
        //>(
        //    req_r, resp_s
        //    cmph_req_s, cmph_resp_r,
        //);

        //spawn fse_proba_freq_dec::FseProbaFreqDecoder();
        ()
    }
}

const TEST_AXI_DATA_W = u32:64;
const TEST_AXI_ADDR_W = u32:32;
const TEST_AXI_ID_W = u32:4;
const TEST_AXI_DEST_W = u32:4;

#[test_proc]
proc CompressBlockDecoderTest {
    type Req = CompressBlockDecoderReq;
    type Resp = CompressBlockDecoderResp;

    type MemAxiAr = axi::AxiAr<TEST_AXI_ADDR_W, TEST_AXI_ID_W>;
    type MemAxiR = axi::AxiR<TEST_AXI_DATA_W, TEST_AXI_ID_W>;

    terminator: chan<bool> out;
    req_s: chan<Req> out;
    resp_r: chan<Resp> in;
    cmph_axi_ar_r: chan<MemAxiAr> in;
    cmph_axi_r_s: chan<MemAxiR> out;

    init {}
    config(terminator: chan<bool> out) {
        let (req_s, req_r) = chan<Req>("req");
        let (resp_s, resp_r) = chan<Resp>("resp");

        let (cmph_axi_ar_s, cmph_axi_ar_r) = chan<MemAxiAr>("cmph_axi_ar");
        let (cmph_axi_r_s, cmph_axi_r_r) = chan<MemAxiR>("cmph_axi_r");

        spawn CompressBlockDecoder<
            TEST_AXI_DATA_W, TEST_AXI_ADDR_W, TEST_AXI_ID_W, TEST_AXI_DEST_W
        >(
            req_r, resp_s,
            cmph_axi_ar_s, cmph_axi_r_r);

        (
            terminator,
            req_s, resp_r,
            cmph_axi_ar_r, cmph_axi_r_s
        )
    }

    next(state: ()) {
        let tok = join();
        send(tok, terminator, true);
    }
}

