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
import xls.modules.zstd.memory.axi;
import xls.modules.zstd.memory.axi_ram;
import xls.modules.zstd.memory.mem_reader;
import xls.modules.zstd.sequence_conf_dec;
import xls.modules.zstd.fse_lookup_dec;
import xls.modules.zstd.ram_demux3;
import xls.modules.zstd.ram_demux;
import xls.modules.zstd.ram_mux;
import xls.modules.zstd.refilling_shift_buffer;
import xls.modules.zstd.fse_dec;
import xls.modules.shift_buffer.shift_buffer;
import xls.modules.zstd.fse_table_creator;


type SequenceExecutorPacket = common::SequenceExecutorPacket<common::SYMBOL_WIDTH>;
type SequenceExecutorMessageType = common::SequenceExecutorMessageType;

type BlockSyncData = common::BlockSyncData;
type CommandConstructorData = common::CommandConstructorData;

enum SequenceDecoderStatus: u3 {
    OK = 0,
    ERROR = 1,
}

pub struct SequenceDecoderReq<ADDR_W: u32> {
    start_addr: uN[ADDR_W],
    end_addr: uN[ADDR_W],
    sync: BlockSyncData,
    literals_count: u20,
}

pub struct SequenceDecoderResp {
    status: SequenceDecoderStatus,
}

enum SequenceDecoderFSM: u3 {
    IDLE = 0,
    DECODE_SEQUENCE_HEADER = 1,
    PREPARE_LL_TABLE = 2,
    PREPARE_OF_TABLE = 3,
    PREPARE_ML_TABLE = 4,

    ERROR = 7,
}

struct SequenceDecoderState<ADDR_W: u32> {
    fsm: SequenceDecoderFSM,
    req: SequenceDecoderReq<ADDR_W>,
    conf_resp: sequence_conf_dec::SequenceConfDecoderResp,
}

struct FseLookupCtrlReq<AXI_ADDR_W: u32> {
    ll: bool,
    ml: bool,
    of: bool,
    addr: uN[AXI_ADDR_W],
}

type AccuracyLog = common::FseAccuracyLog;
struct FseLookupCtrlResp {
    ll_accuracy_log: AccuracyLog,
    ml_accuracy_log: AccuracyLog,
    of_accuracy_log: AccuracyLog,
}

struct FseLookupCtrlState<AXI_ADDR_W: u32> {
    decode: bool[3],
    decode_valid: bool,
    addr: uN[AXI_ADDR_W],
    resp: FseLookupCtrlResp,
    cnt: u2,
}

pub proc FseLookupCtrl<AXI_ADDR_W: u32> {
    type Req = FseLookupCtrlReq<AXI_ADDR_W>;
    type Resp = FseLookupCtrlResp;
    type State = FseLookupCtrlState<AXI_ADDR_W>;

    type FseLookupDecoderReq = fse_lookup_dec::FseLookupDecoderReq<AXI_ADDR_W>;
    type FseLookupDecoderResp = fse_lookup_dec::FseLookupDecoderResp;

    req_r: chan<Req> in;
    resp_s: chan<Resp> out;

    fld_req_s: chan<FseLookupDecoderReq> out;
    fld_resp_r: chan<FseLookupDecoderResp> in;

    fse_demux_req_s: chan<u2> out;
    fse_demux_resp_r: chan<()> in;

    init { zero!<State>() }

    config(
        req_r: chan<Req> in,
        resp_s: chan<Resp> out,

        fld_req_s: chan<FseLookupDecoderReq> out,
        fld_resp_r: chan<FseLookupDecoderResp> in,

        fse_demux_req_s: chan<u2> out,
        fse_demux_resp_r: chan<()> in,
    ) {
        (
            req_r, resp_s,
            fld_req_s, fld_resp_r,
            fse_demux_req_s, fse_demux_resp_r,
        )
    }

    next(state: State) {
        let tok0 = join();

        if !state.decode_valid {
            let (tok1_0, req) = recv(tok0, req_r);
            State {
                decode: bool[3]:[req.ll, req.of, req.ml],
                decode_valid: true,
                cnt: u2:0,
                addr: req.addr,
                ..zero!<State>()
            }
        } else {
            let do_set = state.decode[state.cnt];
            match(state.cnt) {
                u2:0 => trace_fmt!("Handling LL"),
                u2:1 => trace_fmt!("Handling OF"),
                u2:2 => trace_fmt!("Handling ML"),
                _    => trace_fmt!("Impossible case"),
            };

            // trace_fmt!("Sending request to demux {:#x}", state.cnt);
            let tok1 = send_if(tok0, fse_demux_req_s, do_set, state.cnt);
            if do_set {
                trace_fmt!("[SequenceDecoderCtrl/FseLookupCtrl]: Sent fse_demux req {:#x}", state.cnt);
            } else {};

            let (tok2, demux_resp) = recv_if(tok1, fse_demux_resp_r, do_set, ());
            if do_set {
                trace_fmt!("[SequenceDecoderCtrl/FseLookupCtrl]: Received demux resp {:#x}", demux_resp);
            } else {};
            // trace_fmt!("Received response from demux");

            let fld_req = FseLookupDecoderReq { addr: state.addr };
            let tok3 = send_if(tok2, fld_req_s, do_set, fld_req);
            if do_set {
                trace_fmt!("[SequenceDecoderCtrl/FseLookupCtrl]: Sent FseLookupDecoder req {:#x}", fld_req);
            } else {};

            // FIXME: Extract RefillingShiftBuffer from the FseLookupDecoder
            // to support decoding multiple FSE Tables
            let (tok4, fld_resp) = recv_if(tok3, fld_resp_r, do_set, zero!<FseLookupDecoderResp>());
            if do_set {
                trace_fmt!("[SequenceDecoderCtrl/FseLookupCtrl]: Received FseLookupDecoder resp {:#x}", fld_resp);
            } else {};

            let resp = match(state.cnt) {
                u2:0 => FseLookupCtrlResp { ll_accuracy_log: fld_resp.accuracy_log, ..state.resp},
                u2:1 => FseLookupCtrlResp { of_accuracy_log: fld_resp.accuracy_log, ..state.resp},
                u2:2 => FseLookupCtrlResp { ml_accuracy_log: fld_resp.accuracy_log, ..state.resp},
                _ => fail!("impossible_cnt", zero!<FseLookupCtrlResp>()),
            };

            // trace_fmt!("Received response from from FseLookupDecoder {:#x}", fld_resp);

            if state.cnt >= u2:2 {
                let tok5 = send(tok4, resp_s, resp);
                zero!<State>()
            } else {
                State { cnt: state.cnt + u2:1, resp, ..state}
            }
        }
    }
}

const INST_AXI_ADDR_W = u32:32;

pub proc FseLookupCtrlInst {
    type Req = FseLookupCtrlReq<INST_AXI_ADDR_W>;
    type Resp = FseLookupCtrlResp;

    type FseLookupDecoderReq = fse_lookup_dec::FseLookupDecoderReq<INST_AXI_ADDR_W>;
    type FseLookupDecoderResp = fse_lookup_dec::FseLookupDecoderResp;

    init { }

    config(
        req_r: chan<Req> in,
        resp_s: chan<Resp> out,

        fld_req_s: chan<FseLookupDecoderReq> out,
        fld_resp_r: chan<FseLookupDecoderResp> in,

        demux_req_s: chan<u2> out,
        demux_resp_r: chan<()> in,
    ) {
         spawn FseLookupCtrl<INST_AXI_ADDR_W>(
            req_r, resp_s,
            fld_req_s, fld_resp_r,
            demux_req_s, demux_resp_r,
        );
    }

    next(state: ()) {}
}

const TEST_FLC_AXI_ADDR_W = u32:32;

//#[test_proc]
//proc FseLookupCtrlTest {
//
//    type Req = FseLookupCtrlReq<TEST_FLC_AXI_ADDR_W>;
//    type Resp = FseLookupCtrlResp;
//
//    type Addr = uN[TEST_FLC_AXI_ADDR_W];
//
//    type FseLookupDecoderReq = fse_lookup_dec::FseLookupDecoderReq<TEST_FLC_AXI_ADDR_W>;
//    type FseLookupDecoderResp = fse_lookup_dec::FseLookupDecoderResp;
//    type FseLookupDecoderStatus = fse_lookup_dec::FseLookupDecoderStatus;
//
//    terminator: chan<bool> out;
//
//    req_s: chan<Req> out;
//    resp_r: chan<Resp> in;
//    fld_req_r: chan<FseLookupDecoderReq> in;
//    fld_resp_s: chan<FseLookupDecoderResp> out;
//    demux_req_r: chan<u2> in;
//    demux_resp_s: chan<()> out;
//
//    init {}
//
//    config(
//        terminator: chan<bool> out,
//    ) {
//        let (req_s, req_r) = chan<Req>("req");
//        let (resp_s, resp_r) = chan<Resp>("resp");
//        let (fld_req_s, fld_req_r) = chan<FseLookupDecoderReq>("fld_req");
//        let (fld_resp_s, fld_resp_r) = chan<FseLookupDecoderResp>("fld_resp");
//        let (demux_req_s, demux_req_r) = chan<u2>("demux_req");
//        let (demux_resp_s, demux_resp_r) = chan<()>("demux_resp");
//
//        spawn FseLookupCtrl<TEST_FLC_AXI_ADDR_W>(
//            req_r, resp_s,
//            fld_req_s, fld_resp_r,
//            demux_req_s, demux_resp_r,
//        );
//
//        (
//            terminator,
//            req_s, resp_r,
//            fld_req_r, fld_resp_s,
//            demux_req_r, demux_resp_s,
//        )
//    }
//
//    next(state: ()) {
//
//        // Decode all the tables
//        // ---------------------
//
//        // Start
//        let tok = join();
//        let tok = send(tok, req_s, Req { ll: true, of: true, ml: true, addr: Addr:0 });
//
//        // Select LL ( u2:0 )
//        let (tok, demux_req) = recv(tok, demux_req_r);
//        assert_eq(demux_req, u2:0);
//
//        let tok = send(tok, demux_resp_s, ());
//        let (tok, fld_req) = recv(tok, fld_req_r);
//
//        assert_eq(fld_req, zero!<FseLookupDecoderReq>());
//        let tok = send(tok, fld_resp_s, FseLookupDecoderResp {status: FseLookupDecoderStatus::OK});
//
//        // Select OF ( u2:1 )
//        let (tok, demux_req) = recv(tok, demux_req_r);
//        assert_eq(demux_req, u2:1);
//
//        let tok = send(tok, demux_resp_s, ());
//        let (tok, fld_req) = recv(tok, fld_req_r);
//
//        assert_eq(fld_req, zero!<FseLookupDecoderReq>());
//        let tok = send(tok, fld_resp_s, FseLookupDecoderResp {status: FseLookupDecoderStatus::OK});
//
//        // Select ML ( u2:2 )
//        let (tok, demux_req) = recv(tok, demux_req_r);
//        assert_eq(demux_req, u2:2);
//
//        let tok = send(tok, demux_resp_s, ());
//        let (tok, _fld_req) = recv(tok, fld_req_r);
//
//        assert_eq(fld_req, zero!<FseLookupDecoderReq>());
//        let tok = send(tok, fld_resp_s, FseLookupDecoderResp {status: FseLookupDecoderStatus::OK});
//
//        // Stop
//        let (tok, resp) = recv(tok, resp_r);
//        assert_eq(resp, FseLookupCtrlResp {});
//
//        // Decode only LL and ML
//        // ---------------------
//
//        // Start
//        let tok = join();
//        let tok = send(tok, req_s, Req { ll: true, of: false, ml: true, addr: Addr:0 });
//
//        // Select LL ( u2:0 )
//        let (tok, demux_req) = recv(tok, demux_req_r);
//        assert_eq(demux_req, u2:0);
//
//        let tok = send(tok, demux_resp_s, ());
//        let (tok, fld_req) = recv(tok, fld_req_r);
//
//        assert_eq(fld_req, zero!<FseLookupDecoderReq>());
//        let tok = send(tok, fld_resp_s, FseLookupDecoderResp {status: FseLookupDecoderStatus::OK});
//
//        // Select ML ( u2:2 )
//        let (tok, demux_req) = recv(tok, demux_req_r);
//        assert_eq(demux_req, u2:2);
//
//        let tok = send(tok, demux_resp_s, ());
//        let (tok, _fld_req) = recv(tok, fld_req_r);
//
//        assert_eq(fld_req, zero!<FseLookupDecoderReq>());
//        let tok = send(tok, fld_resp_s, FseLookupDecoderResp {status: FseLookupDecoderStatus::OK});
//
//        // Stop
//        let (tok, resp) = recv(tok, resp_r);
//        assert_eq(resp, FseLookupCtrlResp {});
//
//
//        // Decode only OF
//        // ---------------------
//
//        // Start
//        let tok = join();
//        let tok = send(tok, req_s, Req { ll: false, of: true, ml: false, addr: Addr:0 });
//
//        // Select OF ( u2:1 )
//        let (tok, demux_req) = recv(tok, demux_req_r);
//        assert_eq(demux_req, u2:1);
//
//        let tok = send(tok, demux_resp_s, ());
//        let (tok, fld_req) = recv(tok, fld_req_r);
//
//        assert_eq(fld_req, zero!<FseLookupDecoderReq>());
//        let tok = send(tok, fld_resp_s, FseLookupDecoderResp {status: FseLookupDecoderStatus::OK});
//
//        // Stop
//        let (tok, resp) = recv(tok, resp_r);
//        assert_eq(resp, FseLookupCtrlResp {});
//
//        let tok = send(tok, terminator, true);
//    }
//}

pub proc SequenceDecoderCtrl<
    AXI_ADDR_W: u32, AXI_DATA_W: u32,
    REFILLING_SB_DATA_W: u32 = {AXI_DATA_W},
    REFILLING_SB_LENGTH_W: u32 = {refilling_shift_buffer::length_width(AXI_DATA_W)},
> {
    type Req = SequenceDecoderReq<AXI_ADDR_W>;
    type Resp = SequenceDecoderResp;
    type State = SequenceDecoderState<AXI_ADDR_W>;
    type FSM = SequenceDecoderFSM;
    type Status = SequenceDecoderStatus;

    type Addr = uN[AXI_ADDR_W];

    type CompressionMode = common::CompressionMode;
    type SequenceConfDecoderStatus = sequence_conf_dec::SequenceConfDecoderStatus;

    type SequenceConfDecoderReq = sequence_conf_dec::SequenceConfDecoderReq<AXI_ADDR_W>;
    type SequenceConfDecoderResp = sequence_conf_dec::SequenceConfDecoderResp;

    type FseLookupDecoderReq = fse_lookup_dec::FseLookupDecoderReq<AXI_ADDR_W>;
    type FseLookupDecoderResp = fse_lookup_dec::FseLookupDecoderResp;

    type RefillingShiftBufferStart = refilling_shift_buffer::RefillStart<AXI_ADDR_W>;
    type RefillingShiftBufferError = refilling_shift_buffer::RefillingShiftBufferInput<REFILLING_SB_DATA_W, REFILLING_SB_LENGTH_W>;
    type RefillingShiftBufferOutput = refilling_shift_buffer::RefillingShiftBufferOutput<REFILLING_SB_DATA_W, REFILLING_SB_LENGTH_W>;
    type RefillingShiftBufferCtrl = refilling_shift_buffer::RefillingShiftBufferCtrl<REFILLING_SB_LENGTH_W>;

    type FseDecoderCtrl = fse_dec::FseDecoderCtrl;
    type FseDecoderFinish = fse_dec::FseDecoderFinish;

    sd_req_r: chan<Req> in;
    sd_resp_s: chan<Resp> out;

    scd_req_s: chan<SequenceConfDecoderReq> out;
    scd_resp_r: chan<SequenceConfDecoderResp> in;

    flc_req_s: chan<FseLookupCtrlReq> out;
    flc_resp_r: chan<FseLookupCtrlResp> in;

    ll_demux_req_s: chan<u1> out;
    ll_demux_resp_r: chan<()> in;

    of_demux_req_s: chan<u1> out;
    of_demux_resp_r: chan<()> in;

    ml_demux_req_s: chan<u1> out;
    ml_demux_resp_r: chan<()> in;

    fd_rsb_start_req_s: chan<RefillingShiftBufferStart> out;
    fd_rsb_stop_flush_req_s: chan<()> out;
    fd_rsb_flushing_done_r: chan<()> in;

    fd_ctrl_s: chan<FseDecoderCtrl> out;
    fd_finish_r: chan<FseDecoderFinish> in;

    init { }

    config(
        sd_req_r: chan<Req> in,
        sd_resp_s: chan<Resp> out,

        scd_req_s: chan<SequenceConfDecoderReq> out,
        scd_resp_r: chan<SequenceConfDecoderResp> in,

        fld_req_s: chan<FseLookupDecoderReq> out,
        fld_resp_r: chan<FseLookupDecoderResp> in,

        fld_demux_req_s: chan<u2> out,
        fld_demux_resp_r: chan<()> in,

        ll_demux_req_s: chan<u1> out,
        ll_demux_resp_r: chan<()> in,

        of_demux_req_s: chan<u1> out,
        of_demux_resp_r: chan<()> in,

        ml_demux_req_s: chan<u1> out,
        ml_demux_resp_r: chan<()> in,

        fd_rsb_start_req_s: chan<RefillingShiftBufferStart> out,
        fd_rsb_stop_flush_req_s: chan<()> out,
        fd_rsb_flushing_done_r: chan<()> in,

        fd_ctrl_s: chan<FseDecoderCtrl> out,
        fd_finish_r: chan<FseDecoderFinish> in,
    ) {
        const CHANNEL_DEPTH = u32:1;

        let (flc_req_s, flc_req_r) = chan<FseLookupCtrlReq, CHANNEL_DEPTH>("flc_req");
        let (flc_resp_s, flc_resp_r) = chan<FseLookupCtrlResp, CHANNEL_DEPTH>("flc_resp");

        spawn FseLookupCtrl<AXI_ADDR_W>(
            flc_req_r, flc_resp_s,
            fld_req_s, fld_resp_r,
            fld_demux_req_s, fld_demux_resp_r,
        );

        (
            sd_req_r, sd_resp_s,
            scd_req_s, scd_resp_r,
            flc_req_s, flc_resp_r,
            ll_demux_req_s, ll_demux_resp_r,
            of_demux_req_s, of_demux_resp_r,
            ml_demux_req_s, ml_demux_resp_r,
            fd_rsb_start_req_s, fd_rsb_stop_flush_req_s, fd_rsb_flushing_done_r,
            fd_ctrl_s, fd_finish_r,
        )
    }

    next(state: ()) {

        // Receive Sequence Decoder request
        let (tok_req_sd, req) = recv(join(), sd_req_r);
        trace_fmt!("[SequenceDecoderCtrl]: Received Sequence Decoder request: {:#x}", req);

        // Request decoding Sequence Header
        let scd_req = SequenceConfDecoderReq { addr: req.start_addr };
        let tok_send_scd = send(tok_req_sd, scd_req_s, scd_req);
        trace_fmt!("[SequenceDecoderCtrl]: Sent Sequence Decoder request: {:#x}", scd_req);

        // Receive decoded Seqence Header
        let (tok_recv_scd, conf_resp) = recv(tok_send_scd, scd_resp_r);
        trace_fmt!("[SequenceDecoderCtrl]: Received decoded Sequence header: {:#x}", conf_resp);

        // Request decoding lookups
        let flc_req = FseLookupCtrlReq {
            addr: req.start_addr + conf_resp.length as Addr,
            ll: (conf_resp.header.literals_mode == CompressionMode::COMPRESSED),
            ml: (conf_resp.header.match_mode == CompressionMode::COMPRESSED),
            of: (conf_resp.header.offset_mode == CompressionMode::COMPRESSED),
        };

        let zero_sequences = (conf_resp.header.sequence_count == u17:0);
        let tok_send_ctrl = send_if(tok_recv_scd, flc_req_s, !zero_sequences, flc_req);
        if !zero_sequences {
            trace_fmt!("[SequenceDecoderCtrl]: Sent FseLookupCtrl request: {:#x}", flc_req);
        } else {};

        // Receive response about deoded lookups
        let (tok_recv_ctrl, flc_resp) = recv_if(tok_send_ctrl, flc_resp_r, !zero_sequences, zero!<FseLookupCtrlResp>());

        // Set proper LL lookup through demux
        let ll_demux_sel = (conf_resp.header.literals_mode != CompressionMode::PREDEFINED);
        let tok_ll_demux = send_if(tok_recv_scd, ll_demux_req_s, !zero_sequences, ll_demux_sel);
        // Receive response from LL lookup demux
        let (tok_ll_demux, _) = recv_if(tok_ll_demux, ll_demux_resp_r, !zero_sequences, ());

        // Set proper ML lookup through demux
        let ml_demux_sel = (conf_resp.header.match_mode != CompressionMode::PREDEFINED);
        let tok_ml_demux = send_if(tok_recv_scd, ml_demux_req_s, !zero_sequences, ml_demux_sel);
        // Receive response from ML lookup demux
        let (tok_ml_demux, _) = recv_if(tok_ml_demux, ml_demux_resp_r, !zero_sequences, ());

        // Set proper OF lookup through demux
        let of_demux_sel = (conf_resp.header.match_mode != CompressionMode::PREDEFINED);
        let tok_of_demux = send_if(tok_recv_scd, of_demux_req_s, !zero_sequences, of_demux_sel);
        // Receive response from OF lookup demux
        let (tok_of_demux, _) = recv_if(tok_of_demux, of_demux_resp_r, !zero_sequences, ());

        let tok_demux = join(tok_ll_demux, tok_ml_demux, tok_of_demux);

        let fd_rsb_start_req = RefillingShiftBufferStart { start_addr: req.end_addr };
        let tok_rsb_start = send_if(tok_demux, fd_rsb_start_req_s, !zero_sequences, fd_rsb_start_req);
        if !zero_sequences {
            trace_fmt!("[SequenceDecoderCtrl]: Sent RefillingShiftBufferStart request: {:#x}", fd_rsb_start_req);
        } else {};

        let fd_ctrl = FseDecoderCtrl {
            sync: req.sync,
            sequences_count: conf_resp.header.sequence_count as u24,
            literals_count: req.literals_count,
            ll_acc_log: if (conf_resp.header.literals_mode == CompressionMode::PREDEFINED) { u7:6 } else { flc_resp.ll_accuracy_log as u7 },
            of_acc_log: if (conf_resp.header.offset_mode == CompressionMode::PREDEFINED) { u7:5 } else { flc_resp.of_accuracy_log as u7 },
            ml_acc_log: if (conf_resp.header.match_mode == CompressionMode::PREDEFINED) { u7:6 } else { flc_resp.ml_accuracy_log as u7 },
        };

        let tok_fse_dec = send(tok_demux, fd_ctrl_s, fd_ctrl);
        let (tok_fse_dec, _) = recv(tok_fse_dec, fd_finish_r);
        trace_fmt!("[SequenceDecoderCtrl]: Fse finished!");

        let tok_rsb_flush = send_if(tok_fse_dec, fd_rsb_stop_flush_req_s, !zero_sequences, ());
        trace_fmt!("[SequenceDecoderCtrl]: Send flush request");
        let (tok_rsb_flush_done, _) = recv_if(tok_rsb_flush, fd_rsb_flushing_done_r, !zero_sequences, ());
        trace_fmt!("[SequenceDecoderCtrl]: Flush done");

        let resp = SequenceDecoderResp { status: Status::OK };
        send(tok_rsb_flush_done, sd_resp_s, resp);
    }
}

const SDC_TEST_AXI_ADDR_W = u32:32;
const SDC_TEST_AXI_DATA_W = u32:64;
const SDC_TEST_REFILLING_SB_DATA_W = {SDC_TEST_AXI_DATA_W};
const SDC_TEST_REFILLING_SB_LENGTH_W = refilling_shift_buffer::length_width(SDC_TEST_AXI_DATA_W);

//#[test_proc]
//proc SequenceDecoderCtrlTest {
//
//    type Req = SequenceDecoderReq<SDC_TEST_AXI_ADDR_W>;
//    type Resp = SequenceDecoderResp;
//    type Status = SequenceDecoderStatus;
//
//    type CompressionMode = common::CompressionMode;
//    type Addr = uN[SDC_TEST_AXI_ADDR_W];
//
//    type SequenceConf = common::SequenceConf;
//    type SequenceConfDecoderReq = sequence_conf_dec::SequenceConfDecoderReq<SDC_TEST_AXI_ADDR_W>;
//    type SequenceConfDecoderResp = sequence_conf_dec::SequenceConfDecoderResp;
//    type SequenceConfDecoderStatus = sequence_conf_dec::SequenceConfDecoderStatus;
//
//    type FseLookupDecoderReq = fse_lookup_dec::FseLookupDecoderReq<SDC_TEST_AXI_ADDR_W>;
//    type FseLookupDecoderResp = fse_lookup_dec::FseLookupDecoderResp;
//    type FseLookupDecoderStatus = fse_lookup_dec::FseLookupDecoderStatus;
//
//    type RefillingShiftBufferStart = refilling_shift_buffer::RefillStart<SDC_TEST_AXI_ADDR_W>;
//    type RefillingShiftBufferError = refilling_shift_buffer::RefillingShiftBufferInput<SDC_TEST_REFILLING_SB_DATA_W, SDC_TEST_REFILLING_SB_LENGTH_W>;
//    type RefillingShiftBufferOutput = refilling_shift_buffer::RefillingShiftBufferOutput<SDC_TEST_REFILLING_SB_DATA_W, SDC_TEST_REFILLING_SB_LENGTH_W>;
//    type RefillingShiftBufferCtrl = refilling_shift_buffer::RefillingShiftBufferCtrl<SDC_TEST_REFILLING_SB_LENGTH_W>;
//
//    type FseDecoderCtrl = fse_dec::FseDecoderCtrl;
//    type FseDecoderFinish = fse_dec::FseDecoderFinish;
//
//    terminator: chan<bool> out;
//
//    sd_req_s: chan<Req> out;
//    sd_resp_r: chan<Resp> in;
//
//    scd_req_r: chan<SequenceConfDecoderReq> in;
//    scd_resp_s: chan<SequenceConfDecoderResp> out;
//
//    fld_req_r: chan<FseLookupDecoderReq> in;
//    fld_resp_s: chan<FseLookupDecoderResp> out;
//
//    fse_demux_req_r: chan<u2> in;
//    fse_demux_resp_s: chan<()> out;
//
//    ll_demux_req_r: chan<u1> in;
//    ll_demux_resp_s: chan<()> out;
//
//    of_demux_req_r: chan<u1> in;
//    of_demux_resp_s: chan<()> out;
//
//    ml_demux_req_r: chan<u1> in;
//    ml_demux_resp_s: chan<()> out;
//
//    fd_rsb_start_req_r: chan<RefillingShiftBufferStart> in;
//    fd_rsb_stop_flush_req_r: chan<()> in;
//    fd_rsb_flushing_done_s: chan<()> out;
//
//    fd_ctrl_r: chan<FseDecoderCtrl> in;
//    fd_finish_s: chan<FseDecoderFinish> out;
//
//    init { }
//
//    config(terminator: chan<bool> out) {
//        let (sd_req_s, sd_req_r) = chan<Req>("sd_req");
//        let (sd_resp_s, sd_resp_r) = chan<Resp>("sd_resp");
//
//        let (scd_req_s, scd_req_r) = chan<SequenceConfDecoderReq>("scd_req");
//        let (scd_resp_s, scd_resp_r) = chan<SequenceConfDecoderResp>("scd_resp");
//
//        let (fld_req_s, fld_req_r) = chan<FseLookupDecoderReq>("fld_req");
//        let (fld_resp_s, fld_resp_r) = chan<FseLookupDecoderResp>("fld_resp");
//
//        let (fse_demux_req_s, fse_demux_req_r) = chan<u2>("fse_demux_req");
//        let (fse_demux_resp_s, fse_demux_resp_r) = chan<()>("fse_demux_resp");
//
//        let (ll_demux_req_s, ll_demux_req_r) = chan<u1>("ll_demux_req");
//        let (ll_demux_resp_s, ll_demux_resp_r) = chan<()>("ll_demux_resp");
//
//        let (of_demux_req_s, of_demux_req_r) = chan<u1>("of_demux_req");
//        let (of_demux_resp_s, of_demux_resp_r) = chan<()>("of_demux_resp");
//
//        let (ml_demux_req_s, ml_demux_req_r) = chan<u1>("ml_demux_req");
//        let (ml_demux_resp_s, ml_demux_resp_r) = chan<()>("ml_demux_resp");
//
//        let (fd_rsb_start_req_s, fd_rsb_start_req_r) = chan<RefillingShiftBufferStart>("fd_rsb_start_req");
//        let (fd_rsb_stop_flush_req_s, fd_rsb_stop_flush_req_r) = chan<()>("fd_rsb_stop_flush_req");
//        let (fd_rsb_flushing_done_s, fd_rsb_flushing_done_r) = chan<()>("fd_rsb_flushing_done");
//
//        let (fd_ctrl_s, fd_ctrl_r) = chan<FseDecoderCtrl>("fd_ctrl");
//        let (fd_finish_s, fd_finish_r) = chan<FseDecoderFinish>("fd_finish");
//
//        spawn SequenceDecoderCtrl<
//            SDC_TEST_AXI_ADDR_W, SDC_TEST_AXI_DATA_W
//        >(
//            sd_req_r, sd_resp_s,
//            scd_req_s, scd_resp_r,
//            fld_req_s, fld_resp_r,
//            fse_demux_req_s, fse_demux_resp_r,
//            ll_demux_req_s, ll_demux_resp_r,
//            of_demux_req_s, of_demux_resp_r,
//            ml_demux_req_s, ml_demux_resp_r,
//            fd_rsb_start_req_s, fd_rsb_stop_flush_req_s, fd_rsb_flushing_done_r,
//            fd_ctrl_s, fd_finish_r,
//        );
//
//        (
//            terminator,
//            sd_req_s, sd_resp_r,
//            scd_req_r, scd_resp_s,
//            fld_req_r, fld_resp_s,
//            fse_demux_req_r, fse_demux_resp_s,
//            ll_demux_req_r, ll_demux_resp_s,
//            of_demux_req_r, of_demux_resp_s,
//            ml_demux_req_r, ml_demux_resp_s,
//            fd_rsb_start_req_r, fd_rsb_stop_flush_req_r, fd_rsb_flushing_done_s,
//            fd_ctrl_r, fd_finish_s,
//        )
//    }
//
//    next(state: ()) {
//        let tok = join();
//
//        let tok = send(tok, sd_req_s, Req {
//            start_addr: Addr:0x1000,
//            end_addr: Addr:0x1012,
//        });
//
//        let (tok, scd_req) = recv(tok, scd_req_r);
//        assert_eq(scd_req, SequenceConfDecoderReq { addr: Addr: 0x1000 });
//
//        let scd_resp = SequenceConfDecoderResp {
//             header: SequenceConf {
//                 sequence_count: u17:1,
//                 literals_mode: CompressionMode::PREDEFINED,
//                 offset_mode: CompressionMode::RLE,
//                 match_mode: CompressionMode::COMPRESSED,
//             },
//             length: u3:5,
//             status: SequenceConfDecoderStatus::OKAY
//        };
//        let tok = send(tok, scd_resp_s, scd_resp);
//
//        let (tok, demux_req) = recv(tok, fse_demux_req_r);
//        assert_eq(demux_req, u2:2);
//        let tok = send(tok, fse_demux_resp_s, ());
//
//        let (tok, fld_req) = recv(tok, fld_req_r);
//        assert_eq(fld_req, FseLookupDecoderReq {
//            addr: Addr:0x1005,
//        });
//
//        let tok = send(tok, fld_resp_s, FseLookupDecoderResp {status: FseLookupDecoderStatus::OK});
//
//        let (tok, ll_demux) = recv(tok, ll_demux_req_r);
//        assert_eq(ll_demux, u1:0);
//        let tok = send(tok, ll_demux_resp_s, ());
//
//        let (tok, ml_demux) = recv(tok, ml_demux_req_r);
//        assert_eq(ml_demux, u1:1);
//        let tok = send(tok, ml_demux_resp_s, ());
//
//        let (tok, of_demux) = recv(tok, of_demux_req_r);
//        assert_eq(of_demux, u1:1);
//        let tok = send(tok, of_demux_resp_s, ());
//
//        let (tok, fd_ctrl)  = recv(tok, fd_ctrl_r);
//        assert_eq(fd_ctrl, zero!<FseDecoderCtrl>());
//
//        send(tok, terminator, true);
//    }
//}

pub proc SequenceDecoder<
    AXI_ADDR_W: u32, AXI_DATA_W: u32, AXI_DEST_W: u32, AXI_ID_W: u32,
    DPD_RAM_ADDR_W: u32, DPD_RAM_DATA_W: u32, DPD_RAM_NUM_PARTITIONS: u32,
    TMP_RAM_ADDR_W: u32, TMP_RAM_DATA_W: u32, TMP_RAM_NUM_PARTITIONS: u32,
    FSE_RAM_ADDR_W: u32, FSE_RAM_DATA_W: u32, FSE_RAM_NUM_PARTITIONS: u32,

    AXI_DATA_W_DIV8: u32 = {AXI_DATA_W / u32:8},
    REFILLING_SB_DATA_W: u32 = {AXI_DATA_W},
    REFILLING_SB_LENGTH_W: u32 = {refilling_shift_buffer::length_width(AXI_DATA_W)},
> {
    type Req = SequenceDecoderReq<AXI_ADDR_W>;
    type Resp = SequenceDecoderResp;

    type MemAxiAr = axi::AxiAr<AXI_ADDR_W, AXI_ID_W>;
    type MemAxiR = axi::AxiR<AXI_DATA_W, AXI_ID_W>;
    type MemAxiAw = axi::AxiAw<AXI_ADDR_W, AXI_ID_W>;
    type MemAxiW = axi::AxiW<AXI_DATA_W, AXI_DATA_W_DIV8>;
    type MemAxiB = axi::AxiB<AXI_ID_W>;

    type MemReaderStatus = mem_reader::MemReaderStatus;
    type MemReaderReq  = mem_reader::MemReaderReq<AXI_ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<AXI_DATA_W, AXI_ADDR_W>;

    type SequenceConfDecoderReq = sequence_conf_dec::SequenceConfDecoderReq<AXI_ADDR_W>;
    type SequenceConfDecoderResp = sequence_conf_dec::SequenceConfDecoderResp;

    type FseLookupDecoderReq =  fse_lookup_dec::FseLookupDecoderReq<AXI_ADDR_W>;
    type FseLookupDecoderResp = fse_lookup_dec::FseLookupDecoderResp;

    type FseDecoderCtrl = fse_dec::FseDecoderCtrl;
    type FseDecoderFinish = fse_dec::FseDecoderFinish;

    type RefillingShiftBufferStart = refilling_shift_buffer::RefillStart<AXI_ADDR_W>;
    type RefillingShiftBufferError = refilling_shift_buffer::RefillingShiftBufferInput<REFILLING_SB_DATA_W, REFILLING_SB_LENGTH_W>;
    type RefillingShiftBufferOutput = refilling_shift_buffer::RefillingShiftBufferOutput<REFILLING_SB_DATA_W, REFILLING_SB_LENGTH_W>;
    type RefillingShiftBufferCtrl = refilling_shift_buffer::RefillingShiftBufferCtrl<REFILLING_SB_LENGTH_W>;

    type DpdRamRdReq = ram::ReadReq<DPD_RAM_ADDR_W, DPD_RAM_NUM_PARTITIONS>;
    type DpdRamRdResp = ram::ReadResp<DPD_RAM_DATA_W>;
    type DpdRamWrReq = ram::WriteReq<DPD_RAM_ADDR_W, DPD_RAM_DATA_W, DPD_RAM_NUM_PARTITIONS>;
    type DpdRamWrResp = ram::WriteResp;

    type TmpRamRdReq = ram::ReadReq<TMP_RAM_ADDR_W, TMP_RAM_NUM_PARTITIONS>;
    type TmpRamRdResp = ram::ReadResp<TMP_RAM_DATA_W>;
    type TmpRamWrReq = ram::WriteReq<TMP_RAM_ADDR_W, TMP_RAM_DATA_W, TMP_RAM_NUM_PARTITIONS>;
    type TmpRamWrResp = ram::WriteResp;

    type FseRamRdReq = ram::ReadReq<FSE_RAM_ADDR_W, FSE_RAM_NUM_PARTITIONS>;
    type FseRamRdResp = ram::ReadResp<FSE_RAM_DATA_W>;
    type FseRamWrReq = ram::WriteReq<FSE_RAM_ADDR_W, FSE_RAM_DATA_W, FSE_RAM_NUM_PARTITIONS>;
    type FseRamWrResp = ram::WriteResp;

    init { }

    fd_ctrl_s: chan<FseDecoderCtrl> out;
    fd_finish_r: chan<FseDecoderFinish> in;

    fd_rsb_ctrl_r: chan<RefillingShiftBufferCtrl> in;
    fd_rsb_data_s: chan<RefillingShiftBufferOutput> out;

    ll_wr_req_s: chan<FseRamWrReq> out;
    ll_wr_resp_s: chan<FseRamWrResp> in;
    ml_wr_req_s: chan<FseRamWrReq> out;
    ml_wr_resp_s: chan<FseRamWrResp> in;
    of_wr_req_s: chan<FseRamWrReq> out;
    of_wr_resp_s: chan<FseRamWrResp> in;

    dummy_ll_fse_rd_req_r: chan<FseRamRdReq> in;
    dummy_ll_fse_rd_resp_s: chan<FseRamRdResp> out;
    dummy_ll_fse_wr_req_r: chan<FseRamWrReq> in;
    dummy_ll_fse_wr_resp_s: chan<FseRamWrResp> out;
    dummy_ml_fse_rd_req_r: chan<FseRamRdReq> in;
    dummy_ml_fse_rd_resp_s: chan<FseRamRdResp> out;
    dummy_ml_fse_wr_req_r: chan<FseRamWrReq> in;
    dummy_ml_fse_wr_resp_s: chan<FseRamWrResp> out;
    dummy_of_fse_rd_req_r: chan<FseRamRdReq> in;
    dummy_of_fse_rd_resp_s: chan<FseRamRdResp> out;
    dummy_of_fse_wr_req_r: chan<FseRamWrReq> in;
    dummy_of_fse_wr_resp_s: chan<FseRamWrResp> out;

    config (
        // Sequence Conf Decoder (manager)
        scd_axi_ar_s: chan<MemAxiAr> out,
        scd_axi_r_r: chan<MemAxiR> in,

        // Fse Lookup Decoder (manager)
        fld_axi_ar_s: chan<MemAxiAr> out,
        fld_axi_r_r: chan<MemAxiR> in,

        // FSE decoder (manager)
        fd_axi_ar_s: chan<MemAxiAr> out,
        fd_axi_r_r: chan<MemAxiR> in,

        req_r: chan<Req> in,
        resp_s: chan<Resp> out,

        // Command constructor
        fd_command_s: chan<CommandConstructorData> out,

        // RAMs

        dpd_rd_req_s: chan<DpdRamRdReq> out,
        dpd_rd_resp_r: chan<DpdRamRdResp> in,
        dpd_wr_req_s: chan<DpdRamWrReq> out,
        dpd_wr_resp_r: chan<DpdRamWrResp> in,

        tmp_rd_req_s: chan<TmpRamRdReq> out,
        tmp_rd_resp_r: chan<TmpRamRdResp> in,
        tmp_wr_req_s: chan<TmpRamWrReq> out,
        tmp_wr_resp_r: chan<TmpRamWrResp> in,

        ll_def_fse_rd_req_s: chan<FseRamRdReq> out,
        ll_def_fse_rd_resp_r: chan<FseRamRdResp> in,
        ll_def_fse_wr_req_s: chan<FseRamWrReq> out,
        ll_def_fse_wr_resp_r: chan<FseRamWrResp> in,

        ll_fse_rd_req_s: chan<FseRamRdReq> out,
        ll_fse_rd_resp_r: chan<FseRamRdResp> in,
        ll_fse_wr_req_s: chan<FseRamWrReq> out,
        ll_fse_wr_resp_r: chan<FseRamWrResp> in,

        ml_def_fse_rd_req_s: chan<FseRamRdReq> out,
        ml_def_fse_rd_resp_r: chan<FseRamRdResp> in,
        ml_def_fse_wr_req_s: chan<FseRamWrReq> out,
        ml_def_fse_wr_resp_r: chan<FseRamWrResp> in,

        ml_fse_rd_req_s: chan<FseRamRdReq> out,
        ml_fse_rd_resp_r: chan<FseRamRdResp> in,
        ml_fse_wr_req_s: chan<FseRamWrReq> out,
        ml_fse_wr_resp_r: chan<FseRamWrResp> in,

        of_def_fse_rd_req_s: chan<FseRamRdReq> out,
        of_def_fse_rd_resp_r: chan<FseRamRdResp> in,
        of_def_fse_wr_req_s: chan<FseRamWrReq> out,
        of_def_fse_wr_resp_r: chan<FseRamWrResp> in,

        of_fse_rd_req_s: chan<FseRamRdReq> out,
        of_fse_rd_resp_r: chan<FseRamRdResp> in,
        of_fse_wr_req_s: chan<FseRamWrReq> out,
        of_fse_wr_resp_r: chan<FseRamWrResp> in,
    ) {
        const CHANNEL_DEPTH = u32:1;
        const READ_BACKWORD = true;

        // Sequence Section Decoder

        let (scd_mem_rd_req_s,  scd_mem_rd_req_r) = chan<MemReaderReq, CHANNEL_DEPTH>("scd_mem_rd_req");
        let (scd_mem_rd_resp_s, scd_mem_rd_resp_r) = chan<MemReaderResp, CHANNEL_DEPTH>("scd_mem_rd_resp");

        spawn mem_reader::MemReader<AXI_DATA_W, AXI_ADDR_W, AXI_DEST_W, AXI_ID_W, CHANNEL_DEPTH>(
           scd_mem_rd_req_r, scd_mem_rd_resp_s,
           scd_axi_ar_s, scd_axi_r_r,
        );

        let (scd_req_s, scd_req_r) = chan<SequenceConfDecoderReq, CHANNEL_DEPTH>("scd_req");
        let (scd_resp_s, scd_resp_r) = chan<SequenceConfDecoderResp, CHANNEL_DEPTH>("scd_resp");

        spawn sequence_conf_dec::SequenceConfDecoder<AXI_DATA_W, AXI_ADDR_W>(
            scd_mem_rd_req_s, scd_mem_rd_resp_r,
            scd_req_r, scd_resp_s,
        );

        // FseLookupDecoder

        let (fld_mem_rd_req_s,  fld_mem_rd_req_r) = chan<MemReaderReq, CHANNEL_DEPTH>("fld_mem_rd_req");
        let (fld_mem_rd_resp_s, fld_mem_rd_resp_r) = chan<MemReaderResp, CHANNEL_DEPTH>("fld_mem_rd_resp");

        spawn mem_reader::MemReader<AXI_DATA_W, AXI_ADDR_W, AXI_DEST_W, AXI_ID_W, CHANNEL_DEPTH>(
            fld_mem_rd_req_r, fld_mem_rd_resp_s,
            fld_axi_ar_s, fld_axi_r_r,
        );

        let (fld_req_s, fld_req_r) = chan<FseLookupDecoderReq, CHANNEL_DEPTH>("fse_req");
        let (fld_resp_s, fld_resp_r) = chan<FseLookupDecoderResp, CHANNEL_DEPTH>("fse_resp");

        let (fse_rd_req_s, fse_rd_req_r) = chan<FseRamRdReq, CHANNEL_DEPTH>("fse_rd_req");
        let (fse_rd_resp_s, fse_rd_resp_r) = chan<FseRamRdResp, CHANNEL_DEPTH>("fse_rd_resp");
        let (fse_wr_req_s, fse_wr_req_r) = chan<FseRamWrReq, CHANNEL_DEPTH>("fse_wr_req");
        let (fse_wr_resp_s, fse_wr_resp_r) = chan<FseRamWrResp, CHANNEL_DEPTH>("fse_wr_resp");

        spawn fse_lookup_dec::FseLookupDecoder<
            AXI_DATA_W, AXI_ADDR_W,
            DPD_RAM_DATA_W, DPD_RAM_ADDR_W, DPD_RAM_NUM_PARTITIONS,
            TMP_RAM_DATA_W, TMP_RAM_ADDR_W, TMP_RAM_NUM_PARTITIONS,
            FSE_RAM_DATA_W, FSE_RAM_ADDR_W, FSE_RAM_NUM_PARTITIONS,
        >(
            fld_req_r, fld_resp_s,
            fld_mem_rd_req_s, fld_mem_rd_resp_r,
            dpd_rd_req_s, dpd_rd_resp_r, dpd_wr_req_s, dpd_wr_resp_r,
            tmp_rd_req_s, tmp_rd_resp_r, tmp_wr_req_s, tmp_wr_resp_r,
            fse_rd_req_s, fse_rd_resp_r, fse_wr_req_s, fse_wr_resp_r,
        );

        // RamDemux3
        let (fse_demux_req_s, fse_demux_req_r) = chan<u2, CHANNEL_DEPTH>("fse_demux_req");
        let (fse_demux_resp_s, fse_demux_resp_r) = chan<(), CHANNEL_DEPTH>("fse_demux_resp");

        spawn ram_demux3::RamDemux3<FSE_RAM_ADDR_W, FSE_RAM_DATA_W, FSE_RAM_NUM_PARTITIONS>(
            fse_demux_req_r, fse_demux_resp_s,
            fse_rd_req_r, fse_rd_resp_s, fse_wr_req_r, fse_wr_resp_s,
            ll_fse_rd_req_s, ll_fse_rd_resp_r, ll_fse_wr_req_s, ll_fse_wr_resp_r,
            of_fse_rd_req_s, of_fse_rd_resp_r, of_fse_wr_req_s, of_fse_wr_resp_r,
            ml_fse_rd_req_s, ml_fse_rd_resp_r, ml_fse_wr_req_s, ml_fse_wr_resp_r,
        );

        let (ll_demux_req_s, ll_demux_req_r) = chan<u1, CHANNEL_DEPTH>("ll_demux_req");
        let (ll_demux_resp_s, ll_demux_resp_r) = chan<(), CHANNEL_DEPTH>("ll_demux_resp");

        let (ll_rd_req_s, ll_rd_req_r) = chan<FseRamRdReq, CHANNEL_DEPTH>("ll_rd_req");
        let (ll_rd_resp_s, ll_rd_resp_r) = chan<FseRamRdResp, CHANNEL_DEPTH>("ll_rd_resp");
        let (ll_wr_req_s, ll_wr_req_r) = chan<FseRamWrReq, CHANNEL_DEPTH>("ll_wr_req");
        let (ll_wr_resp_s, ll_wr_resp_r) = chan<FseRamWrResp, CHANNEL_DEPTH>("ll_wr_resp");

        let (dummy_ll_fse_rd_req_s, dummy_ll_fse_rd_req_r) = chan<FseRamRdReq, CHANNEL_DEPTH>("dummy_ll_fse_rd_req");
        let (dummy_ll_fse_rd_resp_s, dummy_ll_fse_rd_resp_r) = chan<FseRamRdResp, CHANNEL_DEPTH>("dummy_ll_fse_rd_resp");
        let (dummy_ll_fse_wr_req_s, dummy_ll_fse_wr_req_r) = chan<FseRamWrReq, CHANNEL_DEPTH>("dummy_ll_fse_wr_req");
        let (dummy_ll_fse_wr_resp_s, dummy_ll_fse_wr_resp_r) = chan<FseRamWrResp, CHANNEL_DEPTH>("dummy_ll_fse_wr_resp");

        spawn ram_demux::RamDemux<
            FSE_RAM_ADDR_W, FSE_RAM_DATA_W, FSE_RAM_NUM_PARTITIONS
        > (
            ll_demux_req_r, ll_demux_resp_s,
            ll_rd_req_r, ll_rd_resp_s, ll_wr_req_r, ll_wr_resp_s,
            ll_def_fse_rd_req_s, ll_def_fse_rd_resp_r, ll_def_fse_wr_req_s, ll_def_fse_wr_resp_r,
            dummy_ll_fse_rd_req_s, dummy_ll_fse_rd_resp_r, dummy_ll_fse_wr_req_s, dummy_ll_fse_wr_resp_r,
        );

        let (ml_demux_req_s, ml_demux_req_r) = chan<u1, CHANNEL_DEPTH>("ml_demux_req");
        let (ml_demux_resp_s, ml_demux_resp_r) = chan<(), CHANNEL_DEPTH>("ml_demux_resp");

        let (ml_rd_req_s, ml_rd_req_r) = chan<FseRamRdReq, CHANNEL_DEPTH>("ml_rd_req");
        let (ml_rd_resp_s, ml_rd_resp_r) = chan<FseRamRdResp, CHANNEL_DEPTH>("ml_rd_resp");
        let (ml_wr_req_s, ml_wr_req_r) = chan<FseRamWrReq, CHANNEL_DEPTH>("ml_wr_req");
        let (ml_wr_resp_s, ml_wr_resp_r) = chan<FseRamWrResp, CHANNEL_DEPTH>("ml_wr_resp");

        let (dummy_ml_fse_rd_req_s, dummy_ml_fse_rd_req_r) = chan<FseRamRdReq, CHANNEL_DEPTH>("dummy_ml_fse_rd_req");
        let (dummy_ml_fse_rd_resp_s, dummy_ml_fse_rd_resp_r) = chan<FseRamRdResp, CHANNEL_DEPTH>("dummy_ml_fse_rd_resp");
        let (dummy_ml_fse_wr_req_s, dummy_ml_fse_wr_req_r) = chan<FseRamWrReq, CHANNEL_DEPTH>("dummy_ml_fse_wr_req");
        let (dummy_ml_fse_wr_resp_s, dummy_ml_fse_wr_resp_r) = chan<FseRamWrResp, CHANNEL_DEPTH>("dummy_ml_fse_wr_resp");

        spawn ram_demux::RamDemux<
            FSE_RAM_ADDR_W, FSE_RAM_DATA_W, FSE_RAM_NUM_PARTITIONS
        > (
            ml_demux_req_r, ml_demux_resp_s,
            ml_rd_req_r, ml_rd_resp_s, ml_wr_req_r, ml_wr_resp_s,
            ml_def_fse_rd_req_s, ml_def_fse_rd_resp_r, ml_def_fse_wr_req_s, ml_def_fse_wr_resp_r,
            dummy_ml_fse_rd_req_s, dummy_ml_fse_rd_resp_r, dummy_ml_fse_wr_req_s, dummy_ml_fse_wr_resp_r,
        );

        let (of_demux_req_s, of_demux_req_r) = chan<u1, CHANNEL_DEPTH>("of_demux_req");
        let (of_demux_resp_s, of_demux_resp_r) = chan<(), CHANNEL_DEPTH>("of_demux_resp");

        let (of_rd_req_s, of_rd_req_r) = chan<FseRamRdReq, CHANNEL_DEPTH>("of_rd_req");
        let (of_rd_resp_s, of_rd_resp_r) = chan<FseRamRdResp, CHANNEL_DEPTH>("of_rd_resp");
        let (of_wr_req_s, of_wr_req_r) = chan<FseRamWrReq, CHANNEL_DEPTH>("of_wr_req");
        let (of_wr_resp_s, of_wr_resp_r) = chan<FseRamWrResp, CHANNEL_DEPTH>("of_wr_resp");

        let (dummy_of_fse_rd_req_s, dummy_of_fse_rd_req_r) = chan<FseRamRdReq, CHANNEL_DEPTH>("dummy_of_fse_rd_req");
        let (dummy_of_fse_rd_resp_s, dummy_of_fse_rd_resp_r) = chan<FseRamRdResp, CHANNEL_DEPTH>("dummy_of_fse_rd_resp");
        let (dummy_of_fse_wr_req_s, dummy_of_fse_wr_req_r) = chan<FseRamWrReq, CHANNEL_DEPTH>("dummy_of_fse_wr_req");
        let (dummy_of_fse_wr_resp_s, dummy_of_fse_wr_resp_r) = chan<FseRamWrResp, CHANNEL_DEPTH>("dummy_of_fse_wr_resp");

        spawn ram_demux::RamDemux<
            FSE_RAM_ADDR_W, FSE_RAM_DATA_W, FSE_RAM_NUM_PARTITIONS
        > (
            of_demux_req_r, of_demux_resp_s,
            of_rd_req_r, of_rd_resp_s, of_wr_req_r, of_wr_resp_s,
            of_def_fse_rd_req_s, of_def_fse_rd_resp_r, of_def_fse_wr_req_s, of_def_fse_wr_resp_r,
            dummy_of_fse_rd_req_s, dummy_of_fse_rd_resp_r, dummy_of_fse_wr_req_s, dummy_of_fse_wr_resp_r,
        );

        let (fd_mem_rd_req_s,  fd_mem_rd_req_r) = chan<MemReaderReq, CHANNEL_DEPTH>("fd_mem_rd_req");
        let (fd_mem_rd_resp_s, fd_mem_rd_resp_r) = chan<MemReaderResp, CHANNEL_DEPTH>("fd_mem_rd_resp");

        spawn mem_reader::MemReader<AXI_DATA_W, AXI_ADDR_W, AXI_DEST_W, AXI_ID_W, CHANNEL_DEPTH>(
           fd_mem_rd_req_r, fd_mem_rd_resp_s,
           fd_axi_ar_s, fd_axi_r_r,
        );

        let (fd_rsb_start_req_s, fd_rsb_start_req_r) = chan<RefillingShiftBufferStart, CHANNEL_DEPTH>("fd_rsb_start_req");
        let (fd_rsb_stop_flush_req_s, fd_rsb_stop_flush_req_r) = chan<(), CHANNEL_DEPTH>("fd_rsb_stop_flush_req");
        let (fd_rsb_ctrl_s, fd_rsb_ctrl_r) = chan<RefillingShiftBufferCtrl, CHANNEL_DEPTH>("fd_rsb_ctrl");
        let (fd_rsb_data_s, fd_rsb_data_r) = chan<RefillingShiftBufferOutput, CHANNEL_DEPTH>("fd_rsb_data");
        let (fd_rsb_flushing_done_s, fd_rsb_flushing_done_r) = chan<(), CHANNEL_DEPTH>("fd_rsb_flushing_done");

        spawn refilling_shift_buffer::RefillingShiftBuffer<AXI_DATA_W, AXI_ADDR_W, READ_BACKWORD, u32:0xFF> (
            fd_mem_rd_req_s, fd_mem_rd_resp_r,
            fd_rsb_start_req_r, fd_rsb_stop_flush_req_r,
            fd_rsb_ctrl_r, fd_rsb_data_s,
            fd_rsb_flushing_done_s,
        );

        let (fd_ctrl_s, fd_ctrl_r) = chan<FseDecoderCtrl, CHANNEL_DEPTH>("fd_ctrl");
        let (fd_finish_s, fd_finish_r) = chan<FseDecoderFinish, CHANNEL_DEPTH>("fd_finish");

        spawn fse_dec::FseDecoder<
            FSE_RAM_DATA_W, FSE_RAM_ADDR_W, FSE_RAM_NUM_PARTITIONS, AXI_DATA_W,
        >(
            fd_ctrl_r, fd_finish_s,
            fd_rsb_ctrl_s, fd_rsb_data_r,
            fd_command_s,
            ll_rd_req_s, ll_rd_resp_r,
            ml_rd_req_s, ml_rd_resp_r,
            of_rd_req_s, of_rd_resp_r,
        );

        spawn SequenceDecoderCtrl<AXI_ADDR_W, AXI_DATA_W>(
            req_r, resp_s,
            scd_req_s, scd_resp_r,
            fld_req_s, fld_resp_r,
            fse_demux_req_s, fse_demux_resp_r,
            ll_demux_req_s, ll_demux_resp_r,
            of_demux_req_s, of_demux_resp_r,
            ml_demux_req_s, ml_demux_resp_r,
            fd_rsb_start_req_s, fd_rsb_stop_flush_req_s, fd_rsb_flushing_done_r,
            fd_ctrl_s, fd_finish_r,
        );

        (
            fd_ctrl_s, fd_finish_r,
            fd_rsb_ctrl_r, fd_rsb_data_s,
            ll_wr_req_s, ll_wr_resp_r,
            ml_wr_req_s, ml_wr_resp_r,
            of_wr_req_s, of_wr_resp_r,
             dummy_ll_fse_rd_req_r,
            dummy_ll_fse_rd_resp_s,
            dummy_ll_fse_wr_req_r,
            dummy_ll_fse_wr_resp_s,
            dummy_ml_fse_rd_req_r,
            dummy_ml_fse_rd_resp_s,
            dummy_ml_fse_wr_req_r,
            dummy_ml_fse_wr_resp_s,
            dummy_of_fse_rd_req_r,
            dummy_of_fse_rd_resp_s,
            dummy_of_fse_wr_req_r,
            dummy_of_fse_wr_resp_s,

        )
    }

    next(state: ()) {
        send_if(join(), ll_wr_req_s, false, zero!<FseRamWrReq>());
        recv_if(join(), ll_wr_resp_s, false, zero!<FseRamWrResp>());
        send_if(join(), ml_wr_req_s, false, zero!<FseRamWrReq>());
        recv_if(join(), ml_wr_resp_s, false, zero!<FseRamWrResp>());
        send_if(join(), of_wr_req_s, false, zero!<FseRamWrReq>());
        recv_if(join(), of_wr_resp_s, false, zero!<FseRamWrResp>());

        recv_if(join(), dummy_ll_fse_rd_req_r, false, zero!<FseRamRdReq>());
        send_if(join(), dummy_ll_fse_rd_resp_s, false, zero!<FseRamRdResp>());
        recv_if(join(), dummy_ll_fse_wr_req_r, false, zero!<FseRamWrReq>());
        send_if(join(), dummy_ll_fse_wr_resp_s, false, zero!<FseRamWrResp>());
        recv_if(join(), dummy_ml_fse_rd_req_r, false, zero!<FseRamRdReq>());
        send_if(join(), dummy_ml_fse_rd_resp_s, false, zero!<FseRamRdResp>());
        recv_if(join(), dummy_ml_fse_wr_req_r, false, zero!<FseRamWrReq>());
        send_if(join(), dummy_ml_fse_wr_resp_s, false, zero!<FseRamWrResp>());
        recv_if(join(), dummy_of_fse_rd_req_r, false, zero!<FseRamRdReq>());
        send_if(join(), dummy_of_fse_rd_resp_s, false, zero!<FseRamRdResp>());
        recv_if(join(), dummy_of_fse_wr_req_r, false, zero!<FseRamWrReq>());
        send_if(join(), dummy_of_fse_wr_resp_s, false, zero!<FseRamWrResp>());
    }
}

const TEST_AXI_ADDR_W = u32:32;
const TEST_AXI_DATA_W = u32:64;
const TEST_AXI_DEST_W = u32:8;
const TEST_AXI_ID_W = u32:8;

const TEST_INPUT_RAM_DATA_W = TEST_AXI_DATA_W;
const TEST_INPUT_RAM_SIZE = u32:1024;
const TEST_INPUT_RAM_ADDR_W = TEST_AXI_ADDR_W;
const TEST_INPUT_RAM_WORD_PARTITION_SIZE = TEST_INPUT_RAM_DATA_W / u32:8;
const TEST_INPUT_RAM_NUM_PARTITIONS = ram::num_partitions(TEST_INPUT_RAM_WORD_PARTITION_SIZE, TEST_INPUT_RAM_DATA_W);
const TEST_INPUT_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR = ram::SimultaneousReadWriteBehavior::READ_BEFORE_WRITE;
const TEST_INPUT_RAM_INITIALIZED = true;
const TEST_INPUT_RAM_ASSERT_VALID_READ = true;

const TEST_DPD_RAM_DATA_W = u32:16;
const TEST_DPD_RAM_SIZE = u32:256;
const TEST_DPD_RAM_ADDR_W = std::clog2(TEST_DPD_RAM_SIZE);
const TEST_DPD_RAM_WORD_PARTITION_SIZE = TEST_DPD_RAM_DATA_W;
const TEST_DPD_RAM_NUM_PARTITIONS = ram::num_partitions(TEST_DPD_RAM_WORD_PARTITION_SIZE, TEST_DPD_RAM_DATA_W);

const TEST_FSE_RAM_DATA_W = u32:32;
const TEST_FSE_RAM_SIZE = u32:256;
const TEST_FSE_RAM_ADDR_W = std::clog2(TEST_FSE_RAM_SIZE);
const TEST_FSE_RAM_WORD_PARTITION_SIZE = TEST_FSE_RAM_DATA_W / u32:3;
const TEST_FSE_RAM_NUM_PARTITIONS = ram::num_partitions(TEST_FSE_RAM_WORD_PARTITION_SIZE, TEST_FSE_RAM_DATA_W);
const TEST_FSE_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR = ram::SimultaneousReadWriteBehavior::READ_BEFORE_WRITE;

const TEST_TMP_RAM_DATA_W = u32:16;
const TEST_TMP_RAM_SIZE = u32:256;
const TEST_TMP_RAM_ADDR_W = std::clog2(TEST_TMP_RAM_SIZE);
const TEST_TMP_RAM_WORD_PARTITION_SIZE = TEST_TMP_RAM_DATA_W;
const TEST_TMP_RAM_NUM_PARTITIONS = ram::num_partitions(TEST_TMP_RAM_WORD_PARTITION_SIZE, TEST_TMP_RAM_DATA_W);

const TEST_RAM_DATA = u64[40]:[
    u64:0x0, u64:0x0,        // 0x000
    u64:0x0, u64:0x0,        // 0x010
    u64:0x0, u64:0x0,        // 0x020
    u64:0x0, u64:0x0,        // 0x030
    u64:0x0, u64:0x0,        // 0x040
    u64:0x0, u64:0x0,        // 0x050
    u64:0x0, u64:0x0,        // 0x060
    u64:0x0, u64:0x0,        // 0x070
    u64:0x0, u64:0x0,        // 0x080
    u64:0x0, u64:0x0,        // 0x090
    u64:0x0, u64:0x0,        // 0x0A0
    u64:0x0, u64:0x0,        // 0x0B0
    u64:0x0, u64:0x0,        // 0x0C0
    u64:0x0, u64:0x0,        // 0x0D0
    u64:0x0, u64:0x0,        // 0x0E0
    u64:0x0, u64:0x0,        // 0x0F0
    u64:0x0016A400E6C3000A,  // 0x100
    u64:0x225100295B0012D6,  // 0x138
    u64:0x1E00123CB813DB80,  // 0x140
    u64:0x1026064002C4800A,  // 0x148
    u64:0x0, ...
];

const EXPECTED_OUTPUT = SequenceExecutorPacket[16]:[
    SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::SEQUENCE,
        length: u64:0x0003,
        content: u64:0x001c,
        last: false,
    },
    SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::LITERAL,
        length: u64:0x000a,
        content: u64:0x0,
        last: false,
    },
    SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::SEQUENCE,
        length: u64:0x0003,
        content: u64:0x0389,
        last: false,
    },
    SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::SEQUENCE,
        length: u64:0x0003,
        content: u64:0x013c,
        last: false,
    },
    SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::SEQUENCE,
        length: u64:0x0003,
        content: u64:0x0479,
        last: false,
    },
    SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::LITERAL,
        length: u64:0x0009,
        content: u64:0x0,
        last: false,
    },
    SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::SEQUENCE,
        length: u64:0x0003,
        content: u64:0x001d,
        last: false,
    },
    SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::LITERAL,
        length: u64:0x0001,
        content: u64:0x0,
        last: false,
    },
    SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::SEQUENCE,
        length: u64:0x0003,
        content: u64:0x024a,
        last: false,
    },
    SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::LITERAL,
        length: u64:0x0001,
        content: u64:0x0,
        last: false,
    },
    SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::SEQUENCE,
        length: u64:0x0003,
        content: u64:0x032b,
        last: false,
    },
    SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::LITERAL,
        length: u64:0x0001,
        content: u64:0x0,
        last: false,
    },
    SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::SEQUENCE,
        length: u64:0x0003,
        content: u64:0x02b5,
        last: false,
    },
    SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::LITERAL,
        length: u64:0x0002,
        content: u64:0x0,
        last: false,
    },
    SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::SEQUENCE,
        length: u64:0x0003,
        content: u64:0x03a9,
        last: false,
    },
    SequenceExecutorPacket {
        msg_type: SequenceExecutorMessageType::SEQUENCE,
        length: u64:0x0004,
        content: u64:0x06c3,
        last: true,
    },
];

type Base = u16;
type Symbol = u8;
type NumOfBits = u8;

type FseTableRecord = common::FseTableRecord;

pub const DEFAULT_LL_TABLE = FseTableRecord[64]: [
    FseTableRecord { symbol: Symbol:0,  num_of_bits: NumOfBits:4, base: Base:0  },
    FseTableRecord { symbol: Symbol:0,  num_of_bits: NumOfBits:4, base: Base:16 },
    FseTableRecord { symbol: Symbol:1,  num_of_bits: NumOfBits:5, base: Base:32 },
    FseTableRecord { symbol: Symbol:3,  num_of_bits: NumOfBits:5, base: Base:0  },
    FseTableRecord { symbol: Symbol:4,  num_of_bits: NumOfBits:5, base: Base:0  },
    FseTableRecord { symbol: Symbol:6,  num_of_bits: NumOfBits:5, base: Base:0  },
    FseTableRecord { symbol: Symbol:7,  num_of_bits: NumOfBits:5, base: Base:0  },
    FseTableRecord { symbol: Symbol:9,  num_of_bits: NumOfBits:5, base: Base:0  },
    FseTableRecord { symbol: Symbol:10, num_of_bits: NumOfBits:5, base: Base:0  },
    FseTableRecord { symbol: Symbol:12, num_of_bits: NumOfBits:5, base: Base:0  },
    FseTableRecord { symbol: Symbol:14, num_of_bits: NumOfBits:6, base: Base:0  },
    FseTableRecord { symbol: Symbol:16, num_of_bits: NumOfBits:5, base: Base:0  },
    FseTableRecord { symbol: Symbol:18, num_of_bits: NumOfBits:5, base: Base:0  },
    FseTableRecord { symbol: Symbol:19, num_of_bits: NumOfBits:5, base: Base:0  },
    FseTableRecord { symbol: Symbol:21, num_of_bits: NumOfBits:5, base: Base:0  },
    FseTableRecord { symbol: Symbol:22, num_of_bits: NumOfBits:5, base: Base:0  },
    FseTableRecord { symbol: Symbol:24, num_of_bits: NumOfBits:5, base: Base:0  },
    FseTableRecord { symbol: Symbol:25, num_of_bits: NumOfBits:5, base: Base:32 },
    FseTableRecord { symbol: Symbol:26, num_of_bits: NumOfBits:5, base: Base:0  },
    FseTableRecord { symbol: Symbol:27, num_of_bits: NumOfBits:6, base: Base:0  },
    FseTableRecord { symbol: Symbol:29, num_of_bits: NumOfBits:6, base: Base:0  },
    FseTableRecord { symbol: Symbol:31, num_of_bits: NumOfBits:6, base: Base:0  },
    FseTableRecord { symbol: Symbol:0,  num_of_bits: NumOfBits:4, base: Base:32 },
    FseTableRecord { symbol: Symbol:1,  num_of_bits: NumOfBits:4, base: Base:0  },
    FseTableRecord { symbol: Symbol:2,  num_of_bits: NumOfBits:5, base: Base:0  },
    FseTableRecord { symbol: Symbol:4,  num_of_bits: NumOfBits:5, base: Base:32 },
    FseTableRecord { symbol: Symbol:5,  num_of_bits: NumOfBits:5, base: Base:0  },
    FseTableRecord { symbol: Symbol:7,  num_of_bits: NumOfBits:5, base: Base:32 },
    FseTableRecord { symbol: Symbol:8,  num_of_bits: NumOfBits:5, base: Base:0  },
    FseTableRecord { symbol: Symbol:10, num_of_bits: NumOfBits:5, base: Base:32 },
    FseTableRecord { symbol: Symbol:11, num_of_bits: NumOfBits:5, base: Base:0  },
    FseTableRecord { symbol: Symbol:13, num_of_bits: NumOfBits:6, base: Base:0  },
    FseTableRecord { symbol: Symbol:16, num_of_bits: NumOfBits:5, base: Base:32 },
    FseTableRecord { symbol: Symbol:17, num_of_bits: NumOfBits:5, base: Base:0  },
    FseTableRecord { symbol: Symbol:19, num_of_bits: NumOfBits:5, base: Base:32 },
    FseTableRecord { symbol: Symbol:20, num_of_bits: NumOfBits:5, base: Base:0  },
    FseTableRecord { symbol: Symbol:22, num_of_bits: NumOfBits:5, base: Base:32 },
    FseTableRecord { symbol: Symbol:23, num_of_bits: NumOfBits:5, base: Base:0  },
    FseTableRecord { symbol: Symbol:25, num_of_bits: NumOfBits:4, base: Base:0  },
    FseTableRecord { symbol: Symbol:25, num_of_bits: NumOfBits:4, base: Base:16 },
    FseTableRecord { symbol: Symbol:26, num_of_bits: NumOfBits:5, base: Base:32 },
    FseTableRecord { symbol: Symbol:28, num_of_bits: NumOfBits:6, base: Base:0  },
    FseTableRecord { symbol: Symbol:30, num_of_bits: NumOfBits:6, base: Base:0  },
    FseTableRecord { symbol: Symbol:0,  num_of_bits: NumOfBits:4, base: Base:48 },
    FseTableRecord { symbol: Symbol:1,  num_of_bits: NumOfBits:4, base: Base:16 },
    FseTableRecord { symbol: Symbol:2,  num_of_bits: NumOfBits:5, base: Base:32 },
    FseTableRecord { symbol: Symbol:3,  num_of_bits: NumOfBits:5, base: Base:32 },
    FseTableRecord { symbol: Symbol:5,  num_of_bits: NumOfBits:5, base: Base:32 },
    FseTableRecord { symbol: Symbol:6,  num_of_bits: NumOfBits:5, base: Base:32 },
    FseTableRecord { symbol: Symbol:8,  num_of_bits: NumOfBits:5, base: Base:32 },
    FseTableRecord { symbol: Symbol:9,  num_of_bits: NumOfBits:5, base: Base:32 },
    FseTableRecord { symbol: Symbol:11, num_of_bits: NumOfBits:5, base: Base:32 },
    FseTableRecord { symbol: Symbol:12, num_of_bits: NumOfBits:5, base: Base:32 },
    FseTableRecord { symbol: Symbol:15, num_of_bits: NumOfBits:6, base: Base:0  },
    FseTableRecord { symbol: Symbol:17, num_of_bits: NumOfBits:5, base: Base:32 },
    FseTableRecord { symbol: Symbol:18, num_of_bits: NumOfBits:5, base: Base:32 },
    FseTableRecord { symbol: Symbol:20, num_of_bits: NumOfBits:5, base: Base:32 },
    FseTableRecord { symbol: Symbol:21, num_of_bits: NumOfBits:5, base: Base:32 },
    FseTableRecord { symbol: Symbol:23, num_of_bits: NumOfBits:5, base: Base:32 },
    FseTableRecord { symbol: Symbol:24, num_of_bits: NumOfBits:5, base: Base:32 },
    FseTableRecord { symbol: Symbol:35, num_of_bits: NumOfBits:6, base: Base:0  },
    FseTableRecord { symbol: Symbol:34, num_of_bits: NumOfBits:6, base: Base:0  },
    FseTableRecord { symbol: Symbol:33, num_of_bits: NumOfBits:6, base: Base:0  },
    FseTableRecord { symbol: Symbol:32, num_of_bits: NumOfBits:6, base: Base:0  },
];

pub const DEFAULT_ML_TABLE = FseTableRecord[64]: [
    FseTableRecord { symbol: Symbol:0,  num_of_bits: NumOfBits:6,  base: Base:0  },
    FseTableRecord { symbol: Symbol:1,  num_of_bits: NumOfBits:4,  base: Base:0  },
    FseTableRecord { symbol: Symbol:2,  num_of_bits: NumOfBits:5,  base: Base:32 },
    FseTableRecord { symbol: Symbol:3,  num_of_bits: NumOfBits:5,  base: Base:0  },
    FseTableRecord { symbol: Symbol:5,  num_of_bits: NumOfBits:5,  base: Base:0  },
    FseTableRecord { symbol: Symbol:6,  num_of_bits: NumOfBits:5,  base: Base:0  },
    FseTableRecord { symbol: Symbol:8,  num_of_bits: NumOfBits:5,  base: Base:0  },
    FseTableRecord { symbol: Symbol:10, num_of_bits: NumOfBits:6,  base: Base:0  },
    FseTableRecord { symbol: Symbol:13, num_of_bits: NumOfBits:6,  base: Base:0  },
    FseTableRecord { symbol: Symbol:16, num_of_bits: NumOfBits:6,  base: Base:0  },
    FseTableRecord { symbol: Symbol:19, num_of_bits: NumOfBits:6,  base: Base:0  },
    FseTableRecord { symbol: Symbol:22, num_of_bits: NumOfBits:6,  base: Base:0  },
    FseTableRecord { symbol: Symbol:25, num_of_bits: NumOfBits:6,  base: Base:0  },
    FseTableRecord { symbol: Symbol:28, num_of_bits: NumOfBits:6,  base: Base:0  },
    FseTableRecord { symbol: Symbol:31, num_of_bits: NumOfBits:6,  base: Base:0  },
    FseTableRecord { symbol: Symbol:33, num_of_bits: NumOfBits:6,  base: Base:0  },
    FseTableRecord { symbol: Symbol:35, num_of_bits: NumOfBits:6,  base: Base:0  },
    FseTableRecord { symbol: Symbol:37, num_of_bits: NumOfBits:6,  base: Base:0  },
    FseTableRecord { symbol: Symbol:39, num_of_bits: NumOfBits:6,  base: Base:0  },
    FseTableRecord { symbol: Symbol:41, num_of_bits: NumOfBits:6,  base: Base:0  },
    FseTableRecord { symbol: Symbol:43, num_of_bits: NumOfBits:6,  base: Base:0  },
    FseTableRecord { symbol: Symbol:45, num_of_bits: NumOfBits:6,  base: Base:0  },
    FseTableRecord { symbol: Symbol:1,  num_of_bits: NumOfBits:4,  base: Base:16 },
    FseTableRecord { symbol: Symbol:2,  num_of_bits: NumOfBits:4,  base: Base:0  },
    FseTableRecord { symbol: Symbol:3,  num_of_bits: NumOfBits:5,  base: Base:32 },
    FseTableRecord { symbol: Symbol:4,  num_of_bits: NumOfBits:5,  base: Base:0  },
    FseTableRecord { symbol: Symbol:6,  num_of_bits: NumOfBits:5,  base: Base:32 },
    FseTableRecord { symbol: Symbol:7,  num_of_bits: NumOfBits:5,  base: Base:0  },
    FseTableRecord { symbol: Symbol:9,  num_of_bits: NumOfBits:6,  base: Base:0  },
    FseTableRecord { symbol: Symbol:12, num_of_bits: NumOfBits:6,  base: Base:0  },
    FseTableRecord { symbol: Symbol:15, num_of_bits: NumOfBits:6,  base: Base:0  },
    FseTableRecord { symbol: Symbol:18, num_of_bits: NumOfBits:6,  base: Base:0  },
    FseTableRecord { symbol: Symbol:21, num_of_bits: NumOfBits:6,  base: Base:0  },
    FseTableRecord { symbol: Symbol:24, num_of_bits: NumOfBits:6,  base: Base:0  },
    FseTableRecord { symbol: Symbol:27, num_of_bits: NumOfBits:6,  base: Base:0  },
    FseTableRecord { symbol: Symbol:30, num_of_bits: NumOfBits:6,  base: Base:0  },
    FseTableRecord { symbol: Symbol:32, num_of_bits: NumOfBits:6,  base: Base:0  },
    FseTableRecord { symbol: Symbol:34, num_of_bits: NumOfBits:6,  base: Base:0  },
    FseTableRecord { symbol: Symbol:36, num_of_bits: NumOfBits:6,  base: Base:0  },
    FseTableRecord { symbol: Symbol:38, num_of_bits: NumOfBits:6,  base: Base:0  },
    FseTableRecord { symbol: Symbol:40, num_of_bits: NumOfBits:6,  base: Base:0  },
    FseTableRecord { symbol: Symbol:42, num_of_bits: NumOfBits:6,  base: Base:0  },
    FseTableRecord { symbol: Symbol:44, num_of_bits: NumOfBits:6,  base: Base:0  },
    FseTableRecord { symbol: Symbol:1,  num_of_bits: NumOfBits:4,  base: Base:32 },
    FseTableRecord { symbol: Symbol:1,  num_of_bits: NumOfBits:4,  base: Base:48 },
    FseTableRecord { symbol: Symbol:2,  num_of_bits: NumOfBits:4,  base: Base:16 },
    FseTableRecord { symbol: Symbol:4,  num_of_bits: NumOfBits:5,  base: Base:32 },
    FseTableRecord { symbol: Symbol:5,  num_of_bits: NumOfBits:5,  base: Base:32 },
    FseTableRecord { symbol: Symbol:7,  num_of_bits: NumOfBits:5,  base: Base:32 },
    FseTableRecord { symbol: Symbol:8,  num_of_bits: NumOfBits:5,  base: Base:32 },
    FseTableRecord { symbol: Symbol:11, num_of_bits: NumOfBits:6,  base: Base:0  },
    FseTableRecord { symbol: Symbol:14, num_of_bits: NumOfBits:6,  base: Base:0  },
    FseTableRecord { symbol: Symbol:17, num_of_bits: NumOfBits:6,  base: Base:0  },
    FseTableRecord { symbol: Symbol:20, num_of_bits: NumOfBits:6,  base: Base:0  },
    FseTableRecord { symbol: Symbol:23, num_of_bits: NumOfBits:6,  base: Base:0  },
    FseTableRecord { symbol: Symbol:26, num_of_bits: NumOfBits:6,  base: Base:0  },
    FseTableRecord { symbol: Symbol:29, num_of_bits: NumOfBits:6,  base: Base:0  },
    FseTableRecord { symbol: Symbol:52, num_of_bits: NumOfBits:6,  base: Base:0  },
    FseTableRecord { symbol: Symbol:51, num_of_bits: NumOfBits:6,  base: Base:0  },
    FseTableRecord { symbol: Symbol:50, num_of_bits: NumOfBits:6,  base: Base:0  },
    FseTableRecord { symbol: Symbol:49, num_of_bits: NumOfBits:6,  base: Base:0  },
    FseTableRecord { symbol: Symbol:48, num_of_bits: NumOfBits:6,  base: Base:0  },
    FseTableRecord { symbol: Symbol:47, num_of_bits: NumOfBits:6,  base: Base:0  },
    FseTableRecord { symbol: Symbol:46, num_of_bits: NumOfBits:6,  base: Base:0  },
];

pub const DEFAULT_OF_TABLE = FseTableRecord[32]:[
    FseTableRecord { symbol: Symbol:0,  num_of_bits: NumOfBits:5,  base: Base:0  },
    FseTableRecord { symbol: Symbol:6,  num_of_bits: NumOfBits:4,  base: Base:0  },
    FseTableRecord { symbol: Symbol:9,  num_of_bits: NumOfBits:5,  base: Base:0  },
    FseTableRecord { symbol: Symbol:15, num_of_bits: NumOfBits:5,  base: Base:0  },
    FseTableRecord { symbol: Symbol:21, num_of_bits: NumOfBits:5,  base: Base:0  },
    FseTableRecord { symbol: Symbol:3,  num_of_bits: NumOfBits:5,  base: Base:0  },
    FseTableRecord { symbol: Symbol:7,  num_of_bits: NumOfBits:4,  base: Base:0  },
    FseTableRecord { symbol: Symbol:12, num_of_bits: NumOfBits:5,  base: Base:0  },
    FseTableRecord { symbol: Symbol:18, num_of_bits: NumOfBits:5,  base: Base:0  },
    FseTableRecord { symbol: Symbol:23, num_of_bits: NumOfBits:5,  base: Base:0  },
    FseTableRecord { symbol: Symbol:5,  num_of_bits: NumOfBits:5,  base: Base:0  },
    FseTableRecord { symbol: Symbol:8,  num_of_bits: NumOfBits:4,  base: Base:0  },
    FseTableRecord { symbol: Symbol:14, num_of_bits: NumOfBits:5,  base: Base:0  },
    FseTableRecord { symbol: Symbol:20, num_of_bits: NumOfBits:5,  base: Base:0  },
    FseTableRecord { symbol: Symbol:2,  num_of_bits: NumOfBits:5,  base: Base:0  },
    FseTableRecord { symbol: Symbol:7,  num_of_bits: NumOfBits:4,  base: Base:16 },
    FseTableRecord { symbol: Symbol:11, num_of_bits: NumOfBits:5,  base: Base:0  },
    FseTableRecord { symbol: Symbol:17, num_of_bits: NumOfBits:5,  base: Base:0  },
    FseTableRecord { symbol: Symbol:22, num_of_bits: NumOfBits:5,  base: Base:0  },
    FseTableRecord { symbol: Symbol:4,  num_of_bits: NumOfBits:5,  base: Base:0  },
    FseTableRecord { symbol: Symbol:8,  num_of_bits: NumOfBits:4,  base: Base:16 },
    FseTableRecord { symbol: Symbol:13, num_of_bits: NumOfBits:5,  base: Base:0  },
    FseTableRecord { symbol: Symbol:19, num_of_bits: NumOfBits:5,  base: Base:0  },
    FseTableRecord { symbol: Symbol:1,  num_of_bits: NumOfBits:5,  base: Base:0  },
    FseTableRecord { symbol: Symbol:6,  num_of_bits: NumOfBits:4,  base: Base:16 },
    FseTableRecord { symbol: Symbol:10, num_of_bits: NumOfBits:5,  base: Base:0  },
    FseTableRecord { symbol: Symbol:16, num_of_bits: NumOfBits:5,  base: Base:0  },
    FseTableRecord { symbol: Symbol:28, num_of_bits: NumOfBits:5,  base: Base:0  },
    FseTableRecord { symbol: Symbol:27, num_of_bits: NumOfBits:5,  base: Base:0  },
    FseTableRecord { symbol: Symbol:26, num_of_bits: NumOfBits:5,  base: Base:0  },
    FseTableRecord { symbol: Symbol:25, num_of_bits: NumOfBits:5,  base: Base:0  },
    FseTableRecord { symbol: Symbol:24, num_of_bits: NumOfBits:5,  base: Base:0  },
];

#[test_proc]
proc SequenceDecoderTest {
    type Req = SequenceDecoderReq<TEST_AXI_ADDR_W>;
    type Resp = SequenceDecoderResp;

    type InputAddr = uN[TEST_INPUT_RAM_ADDR_W];
    type InputData = uN[TEST_INPUT_RAM_DATA_W];
    type InputMask = uN[TEST_INPUT_RAM_NUM_PARTITIONS];

    type InputRamRdReq = ram::ReadReq<TEST_INPUT_RAM_ADDR_W, TEST_INPUT_RAM_NUM_PARTITIONS>;
    type InputRamRdResp = ram::ReadResp<TEST_INPUT_RAM_DATA_W>;
    type InputRamWrReq = ram::WriteReq<TEST_INPUT_RAM_ADDR_W, TEST_INPUT_RAM_DATA_W, TEST_INPUT_RAM_NUM_PARTITIONS>;
    type InputRamWrResp = ram::WriteResp;

    type DpdRamRdReq = ram::ReadReq<TEST_DPD_RAM_ADDR_W, TEST_DPD_RAM_NUM_PARTITIONS>;
    type DpdRamRdResp = ram::ReadResp<TEST_DPD_RAM_DATA_W>;
    type DpdRamWrReq = ram::WriteReq<TEST_DPD_RAM_ADDR_W, TEST_DPD_RAM_DATA_W, TEST_DPD_RAM_NUM_PARTITIONS>;
    type DpdRamWrResp = ram::WriteResp;

    type TmpRamRdReq = ram::ReadReq<TEST_TMP_RAM_ADDR_W, TEST_TMP_RAM_NUM_PARTITIONS>;
    type TmpRamRdResp = ram::ReadResp<TEST_TMP_RAM_DATA_W>;
    type TmpRamWrReq = ram::WriteReq<TEST_TMP_RAM_ADDR_W, TEST_TMP_RAM_DATA_W, TEST_TMP_RAM_NUM_PARTITIONS>;
    type TmpRamWrResp = ram::WriteResp;

    type FseAddr = uN[TEST_FSE_RAM_ADDR_W];
    type FseData = uN[TEST_FSE_RAM_DATA_W];
    type FseMask = uN[TEST_FSE_RAM_NUM_PARTITIONS];

    type FseRamRdReq = ram::ReadReq<TEST_FSE_RAM_ADDR_W, TEST_FSE_RAM_NUM_PARTITIONS>;
    type FseRamRdResp = ram::ReadResp<TEST_FSE_RAM_DATA_W>;
    type FseRamWrReq = ram::WriteReq<TEST_FSE_RAM_ADDR_W, TEST_FSE_RAM_DATA_W, TEST_FSE_RAM_NUM_PARTITIONS>;
    type FseRamWrResp = ram::WriteResp;

    type MemAxiAr = axi::AxiAr<TEST_AXI_ADDR_W, TEST_AXI_ID_W>;
    type MemAxiR = axi::AxiR<TEST_AXI_DATA_W, TEST_AXI_ID_W>;

    terminator: chan<bool> out;

    req_s: chan<Req> out;
    resp_r: chan<Resp> in;

    fd_command_r: chan<CommandConstructorData> in;

    input0_rd_req_s: chan<InputRamRdReq> out;
    input0_rd_resp_r: chan<InputRamRdResp> in;
    input0_wr_req_s: chan<InputRamWrReq> out;
    input0_wr_resp_r: chan<InputRamWrResp> in;

    input1_rd_req_s: chan<InputRamRdReq> out;
    input1_rd_resp_r: chan<InputRamRdResp> in;
    input1_wr_req_s: chan<InputRamWrReq> out;
    input1_wr_resp_r: chan<InputRamWrResp> in;

    input2_rd_req_s: chan<InputRamRdReq> out;
    input2_rd_resp_r: chan<InputRamRdResp> in;
    input2_wr_req_s: chan<InputRamWrReq> out;
    input2_wr_resp_r: chan<InputRamWrResp> in;

    ll_sel_test_s: chan<u1> out;
    ll_def_test_rd_req_s: chan<FseRamRdReq> out;
    ll_def_test_rd_resp_r: chan<FseRamRdResp> in;
    ll_def_test_wr_req_s: chan<FseRamWrReq> out;
    ll_def_test_wr_resp_r: chan<FseRamWrResp> in;

    ml_sel_test_s: chan<u1> out;
    ml_def_test_rd_req_s: chan<FseRamRdReq> out;
    ml_def_test_rd_resp_r: chan<FseRamRdResp> in;
    ml_def_test_wr_req_s: chan<FseRamWrReq> out;
    ml_def_test_wr_resp_r: chan<FseRamWrResp> in;

    of_sel_test_s: chan<u1> out;
    of_def_test_rd_req_s: chan<FseRamRdReq> out;
    of_def_test_rd_resp_r: chan<FseRamRdResp> in;
    of_def_test_wr_req_s: chan<FseRamWrReq> out;
    of_def_test_wr_resp_r: chan<FseRamWrResp> in;

    init { }

    config(
        terminator: chan<bool> out
    ) {
        // RAM for probability distribution
        let (dpd_rd_req_s, dpd_rd_req_r) = chan<DpdRamRdReq>("dpd_rd_req");
        let (dpd_rd_resp_s, dpd_rd_resp_r) = chan<DpdRamRdResp>("dpd_rd_resp");
        let (dpd_wr_req_s, dpd_wr_req_r) = chan<DpdRamWrReq>("dpd_wr_req");
        let (dpd_wr_resp_s, dpd_wr_resp_r) = chan<DpdRamWrResp>("dpd_wr_resp");

        spawn ram::RamModel<
            TEST_DPD_RAM_DATA_W,
            TEST_DPD_RAM_SIZE,
            TEST_DPD_RAM_WORD_PARTITION_SIZE
        >(dpd_rd_req_r, dpd_rd_resp_s, dpd_wr_req_r, dpd_wr_resp_s);

        // RAMs for temporary values when decoding probability distribution
        let (tmp_rd_req_s, tmp_rd_req_r) = chan<TmpRamRdReq>("tmp_rd_req");
        let (tmp_rd_resp_s, tmp_rd_resp_r) = chan<TmpRamRdResp>("tmp_rd_resp");
        let (tmp_wr_req_s, tmp_wr_req_r) = chan<TmpRamWrReq>("tmp_wr_req");
        let (tmp_wr_resp_s, tmp_wr_resp_r) = chan<TmpRamWrResp>("tmp_wr_resp");

        spawn ram::RamModel<
            TEST_TMP_RAM_DATA_W,
            TEST_TMP_RAM_SIZE,
            TEST_TMP_RAM_WORD_PARTITION_SIZE
        >(tmp_rd_req_r, tmp_rd_resp_s, tmp_wr_req_r, tmp_wr_resp_s);

        // RAM with default FSE lookup for Literal Lengths

        let (ll_sel_test_s, ll_sel_test_r) = chan<u1>("ll_test_sel");

        let (ll_def_test_rd_req_s, ll_def_test_rd_req_r) = chan<FseRamRdReq>("ll_def_test_rd_req");
        let (ll_def_test_rd_resp_s, ll_def_test_rd_resp_r) = chan<FseRamRdResp>("ll_def_test_rd_resp");
        let (ll_def_test_wr_req_s, ll_def_test_wr_req_r) = chan<FseRamWrReq>("ll_def_test_wr_req");
        let (ll_def_test_wr_resp_s, ll_def_test_wr_resp_r) = chan<FseRamWrResp>("ll_def_test_wr_resp");

        let (ll_def_fse_rd_req_s, ll_def_fse_rd_req_r) = chan<FseRamRdReq>("ll_def_fse_rd_req");
        let (ll_def_fse_rd_resp_s, ll_def_fse_rd_resp_r) = chan<FseRamRdResp>("ll_def_fse_rd_resp");
        let (ll_def_fse_wr_req_s, ll_def_fse_wr_req_r) = chan<FseRamWrReq>("ll_def_fse_wr_req");
        let (ll_def_fse_wr_resp_s, ll_def_fse_wr_resp_r) = chan<FseRamWrResp>("ll_def_fse_wr_resp");

        let (ll_def_rd_req_s, ll_def_rd_req_r) = chan<FseRamRdReq>("ll_def_rd_req");
        let (ll_def_rd_resp_s, ll_def_rd_resp_r) = chan<FseRamRdResp>("ll_def_rd_resp");
        let (ll_def_wr_req_s, ll_def_wr_req_r) = chan<FseRamWrReq>("ll_def_wr_req");
        let (ll_def_wr_resp_s, ll_def_wr_resp_r) = chan<FseRamWrResp>("ll_def_wr_resp");

        spawn ram_mux::RamMux<
            TEST_FSE_RAM_ADDR_W,
            TEST_FSE_RAM_DATA_W,
            TEST_FSE_RAM_NUM_PARTITIONS,
        >(
            ll_sel_test_r,
            ll_def_test_rd_req_r, ll_def_test_rd_resp_s, ll_def_test_wr_req_r, ll_def_test_wr_resp_s,
            ll_def_fse_rd_req_r, ll_def_fse_rd_resp_s, ll_def_fse_wr_req_r, ll_def_fse_wr_resp_s,
            ll_def_rd_req_s, ll_def_rd_resp_r, ll_def_wr_req_s, ll_def_wr_resp_r,
        );

        spawn ram::RamModel<
            TEST_FSE_RAM_DATA_W,
            TEST_FSE_RAM_SIZE,
            TEST_FSE_RAM_WORD_PARTITION_SIZE
        >(ll_def_rd_req_r, ll_def_rd_resp_s, ll_def_wr_req_r, ll_def_wr_resp_s);

        // RAM for FSE lookup for Literal Lengths
        let (ll_fse_rd_req_s, ll_fse_rd_req_r) = chan<FseRamRdReq>("ll_fse_rd_req");
        let (ll_fse_rd_resp_s, ll_fse_rd_resp_r) = chan<FseRamRdResp>("ll_fse_rd_resp");
        let (ll_fse_wr_req_s, ll_fse_wr_req_r) = chan<FseRamWrReq>("ll_fse_wr_req");
        let (ll_fse_wr_resp_s, ll_fse_wr_resp_r) = chan<FseRamWrResp>("ll_fse_wr_resp");

        spawn ram::RamModel<
            TEST_FSE_RAM_DATA_W,
            TEST_FSE_RAM_SIZE,
            TEST_FSE_RAM_WORD_PARTITION_SIZE
        >(ll_fse_rd_req_r, ll_fse_rd_resp_s, ll_fse_wr_req_r, ll_fse_wr_resp_s);

        // RAM with default FSE lookup for Match Lengths

        let (ml_sel_test_s, ml_sel_test_r) = chan<u1>("ml_sel_test");

        let (ml_def_test_rd_req_s, ml_def_test_rd_req_r) = chan<FseRamRdReq>("ml_def_test_rd_req");
        let (ml_def_test_rd_resp_s, ml_def_test_rd_resp_r) = chan<FseRamRdResp>("ml_def_test_rd_resp");
        let (ml_def_test_wr_req_s, ml_def_test_wr_req_r) = chan<FseRamWrReq>("ml_def_test_wr_req");
        let (ml_def_test_wr_resp_s, ml_def_test_wr_resp_r) = chan<FseRamWrResp>("ml_def_test_wr_resp");

        let (ml_def_fse_rd_req_s, ml_def_fse_rd_req_r) = chan<FseRamRdReq>("ml_def_fse_rd_req");
        let (ml_def_fse_rd_resp_s, ml_def_fse_rd_resp_r) = chan<FseRamRdResp>("ml_def_fse_rd_resp");
        let (ml_def_fse_wr_req_s, ml_def_fse_wr_req_r) = chan<FseRamWrReq>("ml_def_fse_wr_req");
        let (ml_def_fse_wr_resp_s, ml_def_fse_wr_resp_r) = chan<FseRamWrResp>("ml_def_fse_wr_resp");

        let (ml_def_rd_req_s, ml_def_rd_req_r) = chan<FseRamRdReq>("ml_def_rd_req");
        let (ml_def_rd_resp_s, ml_def_rd_resp_r) = chan<FseRamRdResp>("ml_def_rd_resp");
        let (ml_def_wr_req_s, ml_def_wr_req_r) = chan<FseRamWrReq>("ml_def_wr_req");
        let (ml_def_wr_resp_s, ml_def_wr_resp_r) = chan<FseRamWrResp>("ml_def_wr_resp");

        spawn ram_mux::RamMux<
            TEST_FSE_RAM_ADDR_W,
            TEST_FSE_RAM_DATA_W,
            TEST_FSE_RAM_NUM_PARTITIONS,
        >(
            ml_sel_test_r,
            ml_def_test_rd_req_r, ml_def_test_rd_resp_s, ml_def_test_wr_req_r, ml_def_test_wr_resp_s,
            ml_def_fse_rd_req_r, ml_def_fse_rd_resp_s, ml_def_fse_wr_req_r, ml_def_fse_wr_resp_s,
            ml_def_rd_req_s, ml_def_rd_resp_r, ml_def_wr_req_s, ml_def_wr_resp_r,
        );

        spawn ram::RamModel<
            TEST_FSE_RAM_DATA_W,
            TEST_FSE_RAM_SIZE,
            TEST_FSE_RAM_WORD_PARTITION_SIZE
        >(ml_def_rd_req_r, ml_def_rd_resp_s, ml_def_wr_req_r, ml_def_wr_resp_s);

        // RAM for FSE lookup for Match Lengths
        let (ml_fse_rd_req_s, ml_fse_rd_req_r) = chan<FseRamRdReq>("ml_fse_rd_req");
        let (ml_fse_rd_resp_s, ml_fse_rd_resp_r) = chan<FseRamRdResp>("ml_fse_rd_resp");
        let (ml_fse_wr_req_s, ml_fse_wr_req_r) = chan<FseRamWrReq>("ml_fse_wr_req");
        let (ml_fse_wr_resp_s, ml_fse_wr_resp_r) = chan<FseRamWrResp>("ml_fse_wr_resp");

        spawn ram::RamModel<
            TEST_FSE_RAM_DATA_W,
            TEST_FSE_RAM_SIZE,
            TEST_FSE_RAM_WORD_PARTITION_SIZE
        >(ml_fse_rd_req_r, ml_fse_rd_resp_s, ml_fse_wr_req_r, ml_fse_wr_resp_s);

        // RAM with default FSE lookup for Offsets

        let (of_sel_test_s, of_sel_test_r) = chan<u1>("of_sel_test");

        let (of_def_test_rd_req_s, of_def_test_rd_req_r) = chan<FseRamRdReq>("of_def_test_rd_req");
        let (of_def_test_rd_resp_s, of_def_test_rd_resp_r) = chan<FseRamRdResp>("of_def_test_rd_resp");
        let (of_def_test_wr_req_s, of_def_test_wr_req_r) = chan<FseRamWrReq>("of_def_test_wr_req");
        let (of_def_test_wr_resp_s, of_def_test_wr_resp_r) = chan<FseRamWrResp>("of_def_test_wr_resp");

        let (of_def_fse_rd_req_s, of_def_fse_rd_req_r) = chan<FseRamRdReq>("of_def_fse_rd_req");
        let (of_def_fse_rd_resp_s, of_def_fse_rd_resp_r) = chan<FseRamRdResp>("of_def_fse_rd_resp");
        let (of_def_fse_wr_req_s, of_def_fse_wr_req_r) = chan<FseRamWrReq>("of_def_fse_wr_req");
        let (of_def_fse_wr_resp_s, of_def_fse_wr_resp_r) = chan<FseRamWrResp>("of_def_fse_wr_resp");

        let (of_def_rd_req_s, of_def_rd_req_r) = chan<FseRamRdReq>("of_def_rd_req");
        let (of_def_rd_resp_s, of_def_rd_resp_r) = chan<FseRamRdResp>("of_def_rd_resp");
        let (of_def_wr_req_s, of_def_wr_req_r) = chan<FseRamWrReq>("of_def_wr_req");
        let (of_def_wr_resp_s, of_def_wr_resp_r) = chan<FseRamWrResp>("of_def_wr_resp");

        spawn ram_mux::RamMux<
            TEST_FSE_RAM_ADDR_W,
            TEST_FSE_RAM_DATA_W,
            TEST_FSE_RAM_NUM_PARTITIONS,
        >(
            of_sel_test_r,
            of_def_test_rd_req_r, of_def_test_rd_resp_s, of_def_test_wr_req_r, of_def_test_wr_resp_s,
            of_def_fse_rd_req_r, of_def_fse_rd_resp_s, of_def_fse_wr_req_r, of_def_fse_wr_resp_s,
            of_def_rd_req_s, of_def_rd_resp_r, of_def_wr_req_s, of_def_wr_resp_r,
        );

        spawn ram::RamModel<
            TEST_FSE_RAM_DATA_W,
            TEST_FSE_RAM_SIZE,
            TEST_FSE_RAM_WORD_PARTITION_SIZE
        >(of_def_rd_req_r, of_def_rd_resp_s, of_def_wr_req_r, of_def_wr_resp_s);

        // RAM for FSE lookup for Offsets
        let (of_fse_rd_req_s, of_fse_rd_req_r) = chan<FseRamRdReq>("of_fse_rd_req");
        let (of_fse_rd_resp_s, of_fse_rd_resp_r) = chan<FseRamRdResp>("of_fse_rd_resp");
        let (of_fse_wr_req_s, of_fse_wr_req_r) = chan<FseRamWrReq>("of_fse_wr_req");
        let (of_fse_wr_resp_s, of_fse_wr_resp_r) = chan<FseRamWrResp>("of_fse_wr_resp");

        spawn ram::RamModel<
            TEST_FSE_RAM_DATA_W,
            TEST_FSE_RAM_SIZE,
            TEST_FSE_RAM_WORD_PARTITION_SIZE
        >(of_fse_rd_req_r, of_fse_rd_resp_s, of_fse_wr_req_r, of_fse_wr_resp_s);

        // Input Memory

        let (input0_rd_req_s,  input0_rd_req_r) = chan<InputRamRdReq>("input_rd_req");
        let (input0_rd_resp_s, input0_rd_resp_r) = chan<InputRamRdResp>("input_rd_resp");
        let (input0_wr_req_s,  input0_wr_req_r) = chan<InputRamWrReq>("input_wr_req");
        let (input0_wr_resp_s, input0_wr_resp_r) = chan<InputRamWrResp>("input_wr_resp");

         spawn ram::RamModel<
            TEST_INPUT_RAM_DATA_W,
            TEST_INPUT_RAM_SIZE,
            TEST_INPUT_RAM_WORD_PARTITION_SIZE,
            TEST_INPUT_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR,
            TEST_INPUT_RAM_INITIALIZED,
            TEST_INPUT_RAM_ASSERT_VALID_READ,
            TEST_INPUT_RAM_ADDR_W,
            TEST_INPUT_RAM_NUM_PARTITIONS,
        >(input0_rd_req_r, input0_rd_resp_s, input0_wr_req_r, input0_wr_resp_s);

        let (ss_axi_ar_s, ss_axi_ar_r) = chan<MemAxiAr>("ss_axi_ar");
        let (ss_axi_r_s, ss_axi_r_r) = chan<MemAxiR>("ss_axi_r");

        spawn axi_ram::AxiRamReader<
            TEST_AXI_ADDR_W, TEST_AXI_DATA_W, TEST_AXI_DEST_W, TEST_AXI_ID_W,
            TEST_INPUT_RAM_SIZE
        >(
            ss_axi_ar_r, ss_axi_r_s,
            input0_rd_req_s, input0_rd_resp_r,
        );

        let (input1_rd_req_s,  input1_rd_req_r) = chan<InputRamRdReq>("input_rd_req");
        let (input1_rd_resp_s, input1_rd_resp_r) = chan<InputRamRdResp>("input_rd_resp");
        let (input1_wr_req_s,  input1_wr_req_r) = chan<InputRamWrReq>("input_wr_req");
        let (input1_wr_resp_s, input1_wr_resp_r) = chan<InputRamWrResp>("input_wr_resp");

         spawn ram::RamModel<
            TEST_INPUT_RAM_DATA_W,
            TEST_INPUT_RAM_SIZE,
            TEST_INPUT_RAM_WORD_PARTITION_SIZE,
            TEST_INPUT_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR,
            TEST_INPUT_RAM_INITIALIZED,
            TEST_INPUT_RAM_ASSERT_VALID_READ,
            TEST_INPUT_RAM_ADDR_W,
            TEST_INPUT_RAM_NUM_PARTITIONS,
        >(input1_rd_req_r, input1_rd_resp_s, input1_wr_req_r, input1_wr_resp_s);

        let (fl_axi_ar_s, fl_axi_ar_r) = chan<MemAxiAr>("fl_axi_ar");
        let (fl_axi_r_s, fl_axi_r_r) = chan<MemAxiR>("fl_axi_r");

        spawn axi_ram::AxiRamReader<
            TEST_AXI_ADDR_W, TEST_AXI_DATA_W, TEST_AXI_DEST_W, TEST_AXI_ID_W,
            TEST_INPUT_RAM_SIZE
        >(
            fl_axi_ar_r, fl_axi_r_s,
            input1_rd_req_s, input1_rd_resp_r,
        );

        let (input2_rd_req_s,  input2_rd_req_r) = chan<InputRamRdReq>("input_rd_req");
        let (input2_rd_resp_s, input2_rd_resp_r) = chan<InputRamRdResp>("input_rd_resp");
        let (input2_wr_req_s,  input2_wr_req_r) = chan<InputRamWrReq>("input_wr_req");
        let (input2_wr_resp_s, input2_wr_resp_r) = chan<InputRamWrResp>("input_wr_resp");

         spawn ram::RamModel<
            TEST_INPUT_RAM_DATA_W,
            TEST_INPUT_RAM_SIZE,
            TEST_INPUT_RAM_WORD_PARTITION_SIZE,
            TEST_INPUT_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR,
            TEST_INPUT_RAM_INITIALIZED,
            TEST_INPUT_RAM_ASSERT_VALID_READ,
            TEST_INPUT_RAM_ADDR_W,
            TEST_INPUT_RAM_NUM_PARTITIONS,
        >(input2_rd_req_r, input2_rd_resp_s, input2_wr_req_r, input2_wr_resp_s);


        let (fd_axi_ar_s, fd_axi_ar_r) = chan<MemAxiAr>("fd_axi_ar");
        let (fd_axi_r_s, fd_axi_r_r) = chan<MemAxiR>("fd_axi_r");

        spawn axi_ram::AxiRamReader<
            TEST_AXI_ADDR_W, TEST_AXI_DATA_W, TEST_AXI_DEST_W, TEST_AXI_ID_W,
            TEST_INPUT_RAM_SIZE
        >(
            fd_axi_ar_r, fd_axi_r_s,
            input2_rd_req_s, input2_rd_resp_r,
        );

       // Sequence Decoder

       let (req_s, req_r) = chan<Req>("req");
       let (resp_s, resp_r) = chan<Resp>("resp");

       let (fd_command_s, fd_command_r) = chan<CommandConstructorData>("fd_command");

       spawn SequenceDecoder<
           TEST_AXI_ADDR_W, TEST_AXI_DATA_W, TEST_AXI_DEST_W, TEST_AXI_ID_W,
           TEST_DPD_RAM_ADDR_W, TEST_DPD_RAM_DATA_W, TEST_DPD_RAM_NUM_PARTITIONS,
           TEST_TMP_RAM_ADDR_W, TEST_TMP_RAM_DATA_W, TEST_TMP_RAM_NUM_PARTITIONS,
           TEST_FSE_RAM_ADDR_W, TEST_FSE_RAM_DATA_W, TEST_FSE_RAM_NUM_PARTITIONS,
       > (
           ss_axi_ar_s, ss_axi_r_r,
           fl_axi_ar_s, fl_axi_r_r,
           fd_axi_ar_s, fd_axi_r_r,

           req_r, resp_s,
           fd_command_s,

           dpd_rd_req_s, dpd_rd_resp_r, dpd_wr_req_s, dpd_wr_resp_r,
           tmp_rd_req_s, tmp_rd_resp_r, tmp_wr_req_s, tmp_wr_resp_r,

           ll_def_fse_rd_req_s, ll_def_fse_rd_resp_r, ll_def_fse_wr_req_s, ll_def_fse_wr_resp_r,
           ll_fse_rd_req_s, ll_fse_rd_resp_r, ll_fse_wr_req_s, ll_fse_wr_resp_r,

           ml_def_fse_rd_req_s, ml_def_fse_rd_resp_r, ml_def_fse_wr_req_s, ml_def_fse_wr_resp_r,
           ml_fse_rd_req_s, ml_fse_rd_resp_r, ml_fse_wr_req_s, ml_fse_wr_resp_r,

           of_def_fse_rd_req_s, of_def_fse_rd_resp_r, of_def_fse_wr_req_s, of_def_fse_wr_resp_r,
           of_fse_rd_req_s, of_fse_rd_resp_r, of_fse_wr_req_s, of_fse_wr_resp_r,
       );

       (
           terminator,
           req_s, resp_r,
           fd_command_r,

           input0_rd_req_s, input0_rd_resp_r, input0_wr_req_s, input0_wr_resp_r,
           input1_rd_req_s, input1_rd_resp_r, input1_wr_req_s, input1_wr_resp_r,
           input2_rd_req_s, input2_rd_resp_r, input2_wr_req_s, input2_wr_resp_r,

           ll_sel_test_s,
           ll_def_test_rd_req_s, ll_def_test_rd_resp_r, ll_def_test_wr_req_s, ll_def_test_wr_resp_r,

           ml_sel_test_s,
           ml_def_test_rd_req_s, ml_def_test_rd_resp_r, ml_def_test_wr_req_s, ml_def_test_wr_resp_r,

           of_sel_test_s,
           of_def_test_rd_req_s, of_def_test_rd_resp_r, of_def_test_wr_req_s, of_def_test_wr_resp_r,
       )
   }

    next(state: ()) {
        let tok = join();

        // FILL THE TEST DATA
        let tok = unroll_for! (i, tok): (u32, token) in range(u32:0, array_size(TEST_RAM_DATA)) {
            let req = InputRamWrReq {
                addr: i as InputAddr,
                data: TEST_RAM_DATA[i] as InputData,
                mask: !InputMask:0,
            };
            let tok = send(tok, input0_wr_req_s, req);
            let (tok, _) = recv(tok, input0_wr_resp_r);
            let tok = send(tok, input1_wr_req_s, req);
            let (tok, _) = recv(tok, input1_wr_resp_r);
            let tok = send(tok, input2_wr_req_s, req);
            let (tok, _) = recv(tok, input2_wr_resp_r);
            tok
        }(tok);

        // FILL THE LL DEFAULT RAM
        let tok = send(tok, ll_sel_test_s, u1:0);
        let tok = unroll_for! (i, tok): (u32, token) in range(u32:0, array_size(DEFAULT_LL_TABLE)) {
            let req = FseRamWrReq {
                addr: i as FseAddr,
                data: fse_table_creator::fse_record_to_bits(DEFAULT_LL_TABLE[i]),
                mask: !FseMask:0,
            };
            let tok = send(tok, ll_def_test_wr_req_s, req);
            let (tok, _) = recv(tok, ll_def_test_wr_resp_r);
            tok
        }(tok);
        let tok = send(tok, ll_sel_test_s, u1:1);

        // FILL THE OF DEFAULT RAM
        let tok = send(tok, of_sel_test_s, u1:0);
        let tok = unroll_for! (i, tok): (u32, token) in range(u32:0, array_size(DEFAULT_OF_TABLE)) {
            let req = FseRamWrReq {
                addr: i as FseAddr,
                data: fse_table_creator::fse_record_to_bits(DEFAULT_OF_TABLE[i]),
                mask: !FseMask:0,
            };
            let tok = send(tok, of_def_test_wr_req_s, req);
            let (tok, _) = recv(tok, of_def_test_wr_resp_r);
            tok
        }(tok);
        let tok = send(tok, of_sel_test_s, u1:1);

        // FILL THE ML DEFAULT RAM
        let tok = send(tok, ml_sel_test_s, u1:0);
        let tok = unroll_for! (i, tok): (u32, token) in range(u32:0, array_size(DEFAULT_ML_TABLE)) {
            let req = FseRamWrReq {
                addr: i as FseAddr,
                data: fse_table_creator::fse_record_to_bits(DEFAULT_ML_TABLE[i]),
                mask: !FseMask:0,
            };
            let tok = send(tok, ml_def_test_wr_req_s, req);
            let (tok, _) = recv(tok, ml_def_test_wr_resp_r);
            tok
        }(tok);
        let tok = send(tok, ml_sel_test_s, u1:1);

        // START DECODING
        let tok = send(tok, req_s, Req {
            sync: BlockSyncData {
                id: u32:0,
                last_block: false,
            },
            start_addr: uN[TEST_AXI_ADDR_W]:0x100,
            end_addr: uN[TEST_AXI_ADDR_W]:0x120,
            literals_count: u20:24,
        });

        let tok = for ((i, output), tok): ((u32, SequenceExecutorPacket), token) in enumerate(EXPECTED_OUTPUT) {
            let (tok, recv_output) = recv(tok, fd_command_r);
            trace_fmt!("[{}]: Expected: {:#x}\nGot: {:#x}\n", i, output, recv_output);
            assert_eq(output, recv_output.data);
            tok
        }(tok);

        let (tok, _) = recv(tok, resp_r);

        // START DECODING - ask for more literals - expecting additional empty output packet with
        // last set
        let tok = send(tok, req_s, Req {
            sync: BlockSyncData {
                id: u32:0,
                last_block: false,
            },
            start_addr: uN[TEST_AXI_ADDR_W]:0x100,
            end_addr: uN[TEST_AXI_ADDR_W]:0x120,
            literals_count: u20:26,
        });

        // Don't read the last output packet from the expected output array
        let tok = for (i, tok): (u32, token) in u32:0..(array_size(EXPECTED_OUTPUT) - u32:1) {
            let output = EXPECTED_OUTPUT[i];
            let (tok, recv_output) = recv(tok, fd_command_r);
            trace_fmt!("[{}]: Expected: {:#x}\nGot: {:#x}\n", i, output, recv_output);
            assert_eq(output, recv_output.data);
            tok
        }(tok);

        // The last packet from the expected output array is now expected to have last not set
        let expected = SequenceExecutorPacket {
            last: false,
            ..EXPECTED_OUTPUT[array_size(EXPECTED_OUTPUT) - u32:1]
        };
        let (tok, recv_output) = recv(tok, fd_command_r);
        trace_fmt!("[LAST-1]: Expected: {:#x}\nGot: {:#x}\n", expected, recv_output);
        assert_eq(expected, recv_output.data);

        // This is the actual last output packet
        let expected = SequenceExecutorPacket {
            msg_type: SequenceExecutorMessageType::LITERAL,
            length: u64:0x2,
            content: u64:0x0,
            last: true,
        };
        let (tok, recv_output) = recv(tok, fd_command_r);
        trace_fmt!("[LAST]: Expected: {:#x}\nGot: {:#x}\n", expected, recv_output);
        assert_eq(expected, recv_output.data);

        let (tok, _) = recv(tok, resp_r);

        send(tok, terminator, true);
   }

}
