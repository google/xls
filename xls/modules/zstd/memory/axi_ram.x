// Copyright 2023-2024 The XLS Authors
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

import xls.modules.zstd.math;
import xls.modules.zstd.memory.axi;
import xls.examples.ram;

type AxiAr = axi::AxiAr;
type AxiR = axi::AxiR;

type AxiReadResp = axi::AxiReadResp;
type AxiAxBurst = axi::AxiAxBurst;

const AXI_AXSIZE_ENCODING_TO_SIZE = axi::AXI_AXSIZE_ENCODING_TO_SIZE;

enum AxiRamReaderStatus: u1 {
    IDLE = 0,
    READ_BURST = 1,
}

// FIXME: add default value for RAM_DATA_W_PLUS1_LOG2 = {std::clog2(AXI_DATA_W + u32:1)} (https://github.com/google/xls/issues/992)
struct AxiRamReaderSync<AXI_ID_W: u32, RAM_DATA_W: u32, RAM_DATA_W_PLUS1_LOG2: u32> {
    do_recv_ram_resp: bool,
    read_data_size: uN[RAM_DATA_W_PLUS1_LOG2],
    read_data_offset: uN[RAM_DATA_W_PLUS1_LOG2],
    send_data: bool,
    resp: AxiReadResp,
    id: uN[AXI_ID_W],
    last: bool,
}

struct AxiRamReaderRequesterState<AXI_ADDR_W: u32, AXI_ID_W: u32> {
    status: AxiRamReaderStatus,
    ar_bundle: AxiAr<AXI_ADDR_W, AXI_ID_W>,
    read_data_size: u32,
    addr: uN[AXI_ADDR_W],
    ram_rd_req_idx: u8,
}

// FIXME: add default value for AXI_DATA_W_PLUS1_LOG2 = {std::clog2(AXI_DATA_W + u32:1)} (https://github.com/google/xls/issues/992)
struct AxiRamReaderResponderState<AXI_DATA_W: u32, AXI_DATA_W_PLUS1_LOG2:u32> {
    data: uN[AXI_DATA_W],
    data_size: uN[AXI_DATA_W_PLUS1_LOG2],
}

// Translates RAM requests to AXI read requests
proc AxiRamReaderRequester<
    // AXI parameters
    AXI_ADDR_W: u32, AXI_DATA_W: u32, AXI_DEST_W: u32, AXI_ID_W: u32,

    // RAM parameters
    RAM_SIZE: u32,
    BASE_ADDR: u32 = {u32:0},
    RAM_DATA_W: u32 = {AXI_DATA_W},
    RAM_ADDR_W: u32 = {AXI_ADDR_W},
    RAM_NUM_PARTITIONS: u32 = {AXI_DATA_W / u32:8 },

    AXI_DATA_W_DIV8: u32 = { AXI_DATA_W / u32:8 },
    RAM_DATA_W_LOG2: u32 = { std::clog2(RAM_DATA_W) },
    AXI_DATA_W_LOG2: u32 = { std::clog2(AXI_DATA_W) },
    AXI_DATA_W_PLUS1_LOG2: u32 = { std::clog2(AXI_DATA_W + u32:1) },
    RAM_DATA_W_PLUS1_LOG2: u32 = { std::clog2(RAM_DATA_W + u32:1) },
> {
    type AxiAr = axi::AxiAr<AXI_ADDR_W, AXI_ID_W>;
    type ReadReq = ram::ReadReq<RAM_ADDR_W, RAM_NUM_PARTITIONS>;

    type State = AxiRamReaderRequesterState<AXI_ADDR_W, AXI_ID_W>;
    type Status = AxiRamReaderStatus;
    type Sync = AxiRamReaderSync<AXI_ID_W, RAM_DATA_W, RAM_DATA_W_PLUS1_LOG2>;

    axi_ar_r: chan<AxiAr> in;
    rd_req_s: chan<ReadReq> out;

    sync_s: chan<Sync> out;

    init { zero!<State>() }

    config(
        // AXI interface
        axi_ar_r: chan<AxiAr> in,
        rd_req_s: chan<ReadReq> out,
        sync_s: chan<Sync> out,
    ) {
        (axi_ar_r, rd_req_s, sync_s)
    }

    next(state: State) {
        const RAM_DATA_W_DIV8 = RAM_DATA_W >> u32:3;

        // receive AXI read request
        let (tok, ar_bundle, ar_bundle_valid) = recv_if_non_blocking(join(), axi_ar_r, state.status == Status::IDLE, zero!<AxiAr>());

        // validate bundle
        let ar_bundle_ok = ar_bundle_valid && ((ar_bundle.size as u32 + u32:3) <= AXI_DATA_W_LOG2);
        //if ar_bundle_valid {
        //    trace_fmt!("{:#x}", ar_bundle);
        //} else {};
        let tok = send_if(tok, sync_s, ar_bundle_valid && !ar_bundle_ok, Sync {
            id: ar_bundle.id,
            resp: AxiReadResp::SLVERR,
            last: true,
            send_data: true,
            ..zero!<Sync>()
        });

        // send RAM read reqest
        let addr_valid = state.addr < ((RAM_SIZE * RAM_DATA_W_DIV8) as uN[AXI_ADDR_W]);
        let addr = (state.addr / RAM_DATA_W_DIV8 as uN[AXI_ADDR_W]) as uN[RAM_ADDR_W];

        let do_read_from_ram = (
            (state.status == Status::READ_BURST) &&
            addr_valid &&
            (state.ram_rd_req_idx <= state.ar_bundle.len)
        );
        let ram_read_req = ReadReq {
            addr: addr,
            mask: !uN[RAM_NUM_PARTITIONS]:0,
        };
        let tok = send_if(join(), rd_req_s, do_read_from_ram, ram_read_req);
        if do_read_from_ram {
            trace_fmt!("Sent RAM read request {:#x}", ram_read_req);
        } else {};

        // send sync
        let resp = if addr_valid {
            AxiReadResp::OKAY
        } else {
            AxiReadResp::DECERR
        };

        // calculate read size and offset
        let arsize_bits = AXI_AXSIZE_ENCODING_TO_SIZE[state.ar_bundle.size as u3] as uN[AXI_DATA_W_PLUS1_LOG2];

        let (read_data_size, read_data_offset) = if (arsize_bits > RAM_DATA_W as uN[AXI_DATA_W_PLUS1_LOG2]) {
            (
                RAM_DATA_W as uN[RAM_DATA_W_PLUS1_LOG2],
                uN[RAM_DATA_W_PLUS1_LOG2]:0,
            )
        } else {
            (
                arsize_bits,
                ((state.addr % RAM_DATA_W_DIV8 as uN[AXI_ADDR_W]) << u32:3) as uN[RAM_DATA_W_PLUS1_LOG2],
            )
        };

        let tok = send_if(tok, sync_s, state.status == Status::READ_BURST, Sync {
            do_recv_ram_resp: do_read_from_ram,
            read_data_size: read_data_size,
            read_data_offset: read_data_offset,
            send_data: read_data_size == arsize_bits,
            resp: resp,
            id: state.ar_bundle.id,
            last: state.ram_rd_req_idx == state.ar_bundle.len,
        });

        // update state
        match state.status {
            Status::IDLE => {
                if ar_bundle_ok {
                    State {
                        status: AxiRamReaderStatus::READ_BURST,
                        ar_bundle: ar_bundle,
                        addr: ar_bundle.addr,
                        ram_rd_req_idx: u8:0,
                        read_data_size: u32:0,
                    }
                } else { state }
            },
            Status::READ_BURST => {
                if (state.ram_rd_req_idx == state.ar_bundle.len) {
                    State {
                        status: Status::IDLE,
                        ..state
                    }
                } else {
                    let incr = math::logshiftl(uN[AXI_ADDR_W]:1, state.ar_bundle.size as uN[AXI_ADDR_W]);
                    let addr = match state.ar_bundle.burst {
                        AxiAxBurst::FIXED => state.addr,
                        AxiAxBurst::INCR => state.addr + incr,
                        AxiAxBurst::WRAP => if ((state.addr + incr) as u32 >= (RAM_SIZE * RAM_DATA_W_DIV8)) {
                            uN[AXI_ADDR_W]:0
                        } else {
                            state.addr + incr
                        },
                        _ => fail!("invalid_burst_mode", state.addr),
                    };
                    State {
                        ram_rd_req_idx: state.ram_rd_req_idx + u8:1,
                        addr: addr,
                        ..state
                    }
                }
            },
            _ => state,
        }
    }
}

// Should translate RAM responses to AXI read responses
proc AxiRamReaderResponder<
    // AXI parameters
    AXI_ADDR_W: u32, AXI_DATA_W: u32, AXI_DEST_W: u32, AXI_ID_W: u32,

    // RAM parameters
    RAM_SIZE: u32,
    BASE_ADDR: u32 = {u32:0},
    RAM_DATA_W: u32 = {AXI_DATA_W},
    RAM_ADDR_W: u32 = {AXI_ADDR_W},
    RAM_NUM_PARTITIONS: u32 = {AXI_DATA_W / u32:8 },

    AXI_DATA_W_DIV8: u32 = { AXI_DATA_W / u32:8 },
    AXI_DATA_W_LOG2: u32 = { std::clog2(AXI_DATA_W) },
    RAM_DATA_W_LOG2: u32 = { std::clog2(RAM_DATA_W) },
    AXI_DATA_W_PLUS1_LOG2: u32 = { std::clog2(AXI_DATA_W + u32:1) },
    RAM_DATA_W_PLUS1_LOG2: u32 = { std::clog2(RAM_DATA_W + u32:1) },
> {
    type AxiR = axi::AxiR<AXI_DATA_W, AXI_ID_W>;
    type ReadResp = ram::ReadResp<RAM_DATA_W>;

    type State = AxiRamReaderResponderState<AXI_DATA_W, AXI_DATA_W_PLUS1_LOG2>;
    type Sync = AxiRamReaderSync<AXI_ID_W, RAM_DATA_W, RAM_DATA_W_PLUS1_LOG2>;

    rd_resp_r: chan<ReadResp> in;
    axi_r_s: chan<AxiR> out;

    sync_r: chan<Sync> in;

    init { zero!<State>() }

    config(
        rd_resp_r: chan<ReadResp> in,
        axi_r_s: chan<AxiR> out,
        sync_r: chan<Sync> in,
    ) {
        (rd_resp_r, axi_r_s, sync_r)
    }

    next(state: State) {
        let tok = join();

        // receive sync
        let (tok, sync_data) = recv(tok, sync_r);
        trace_fmt!("Received sync {:#x}", sync_data);

        // receive RAM read respose
        let (tok, ram_read_resp) = recv_if(tok, rd_resp_r, sync_data.do_recv_ram_resp, zero!<ReadResp>());
        if sync_data.do_recv_ram_resp {
            trace_fmt!("Received RAM response {:#x}", ram_read_resp);
        } else {};

        let mask = math::logshiftl(uN[RAM_DATA_W]:1, sync_data.read_data_size as uN[RAM_DATA_W]) - uN[RAM_DATA_W]:1;
        let mask = math::logshiftl(mask, state.data_size);

        let ram_data_shifted = if (sync_data.read_data_offset > state.data_size) {
            math::logshiftr(ram_read_resp.data, sync_data.read_data_offset - state.data_size) as uN[AXI_DATA_W] & mask
        } else {
            math::logshiftl(ram_read_resp.data, state.data_size - sync_data.read_data_offset) as uN[AXI_DATA_W] & mask
        };

        // update state
        let state = State {
            data: ram_data_shifted,
            data_size: state.data_size + sync_data.read_data_size,
        };

        // send AXI read response
        let axi_r_bundle = AxiR {
            id: sync_data.id,
            data: state.data,
            resp: sync_data.resp,
            last: sync_data.last,
        };
        let tok = send_if(tok, axi_r_s, sync_data.send_data, axi_r_bundle);

        if sync_data.send_data {
            zero!<State>()
        } else {
            state
        }
    }
}

pub proc AxiRamReader<
    // AXI parameters
    AXI_ADDR_W: u32,
    AXI_DATA_W: u32,
    AXI_DEST_W: u32,
    AXI_ID_W: u32,

    // RAM parameters
    RAM_SIZE: u32,
    BASE_ADDR: u32 = {u32:0},
    RAM_DATA_W: u32 = {AXI_DATA_W},
    RAM_ADDR_W: u32 = {AXI_ADDR_W},
    RAM_NUM_PARTITIONS: u32 = { AXI_DATA_W / u32:8 },

    AXI_DATA_W_DIV8: u32 = { AXI_DATA_W / u32:8 },
    RAM_DATA_W_LOG2: u32 = { std::clog2(RAM_DATA_W) },
    RAM_DATA_W_PLUS1_LOG2: u32 = { std::clog2(RAM_DATA_W + u32:1) },
> {
    type AxiAr = axi::AxiAr<AXI_ADDR_W, AXI_ID_W>;
    type AxiR = axi::AxiR<AXI_DATA_W, AXI_ID_W>;

    type ReadReq = ram::ReadReq<RAM_ADDR_W, RAM_NUM_PARTITIONS>;
    type ReadResp = ram::ReadResp<RAM_DATA_W>;

    type Sync = AxiRamReaderSync<AXI_ID_W, RAM_DATA_W, RAM_DATA_W_PLUS1_LOG2>;

    init { }

    config(
        // AXI interface
        axi_ar_r: chan<AxiAr> in,
        axi_r_s: chan<AxiR> out,

        // RAM interface
        rd_req_s: chan<ReadReq> out,
        rd_resp_r: chan<ReadResp> in,
    ) {
        let (sync_s, sync_r) = chan<Sync, u32:1>("sync");

        spawn AxiRamReaderRequester<
            AXI_ADDR_W, AXI_DATA_W, AXI_DEST_W, AXI_ID_W,
            RAM_SIZE, BASE_ADDR, RAM_DATA_W, RAM_ADDR_W, RAM_NUM_PARTITIONS,
            AXI_DATA_W_DIV8,
        >(axi_ar_r, rd_req_s, sync_s);
        spawn AxiRamReaderResponder<
            AXI_ADDR_W, AXI_DATA_W, AXI_DEST_W, AXI_ID_W,
            RAM_SIZE, BASE_ADDR, RAM_DATA_W, RAM_ADDR_W, RAM_NUM_PARTITIONS,
            AXI_DATA_W_DIV8,
        >(rd_resp_r, axi_r_s, sync_r);
    }

    next(state: ()) { }
}

const INST_AXI_ADDR_W = u32:32;
const INST_AXI_DATA_W = u32:32;
const INST_AXI_DEST_W = u32:8;
const INST_AXI_ID_W = u32:8;
const INST_AXI_DATA_W_DIV8 = INST_AXI_DATA_W / u32:8;

const INST_RAM_SIZE = u32:100;
const INST_RAM_DATA_W = INST_AXI_DATA_W;
const INST_RAM_ADDR_W = std::clog2(INST_RAM_SIZE);
const INST_RAM_WORD_PARTITION_SIZE = u32:8;
const INST_RAM_NUM_PARTITIONS = INST_RAM_DATA_W / INST_RAM_WORD_PARTITION_SIZE;

const INST_BASE_ADDR = u32:0x8000;

proc AxiRamReaderInst<
    FAKE_PARAM: u32 = {u32:0} // FIXME: remove after https://github.com/google/xls/issues/1415 is fixed
> {
    type AxiAr = axi::AxiAr<INST_AXI_ADDR_W, INST_AXI_ID_W>;
    type AxiR = axi::AxiR<INST_AXI_DATA_W, INST_AXI_ID_W>;
    type ReadReq = ram::ReadReq<INST_RAM_ADDR_W, INST_RAM_NUM_PARTITIONS>;
    type ReadResp = ram::ReadResp<INST_RAM_DATA_W>;

    init { }

    config(
        // AXI interface
        axi_ar_r: chan<AxiAr> in,
        axi_r_s: chan<AxiR> out,
        // RAM interface
        rd_req_s: chan<ReadReq> out,
        rd_resp_r: chan<ReadResp> in,
    ) {
        spawn AxiRamReader<
            INST_AXI_ADDR_W, INST_AXI_DATA_W, INST_AXI_DEST_W, INST_AXI_ID_W,
            INST_RAM_SIZE, INST_BASE_ADDR, INST_RAM_DATA_W, INST_RAM_ADDR_W, INST_RAM_NUM_PARTITIONS,
            INST_AXI_DATA_W_DIV8
        > (axi_ar_r, axi_r_s, rd_req_s, rd_resp_r);
    }

    next(state: ()) { }
}

// only for RAM rewrite
proc AxiRamReaderInstWithEmptyWrites {
    type AxiAr = axi::AxiAr<INST_AXI_ADDR_W, INST_AXI_ID_W>;
    type AxiR = axi::AxiR<INST_AXI_DATA_W, INST_AXI_ID_W>;
    type ReadReq = ram::ReadReq<INST_RAM_ADDR_W, INST_RAM_NUM_PARTITIONS>;
    type ReadResp = ram::ReadResp<INST_RAM_DATA_W>;
    type WriteReq = ram::WriteReq<INST_RAM_ADDR_W, INST_RAM_DATA_W, INST_RAM_NUM_PARTITIONS>;
    type WriteResp = ram::WriteResp;

    wr_req_s: chan<WriteReq> out;
    wr_resp_r: chan<WriteResp> in;

    init { }

    config(
        // AXI interface
        axi_ar_r: chan<AxiAr> in,
        axi_r_s: chan<AxiR> out,
        // RAM interface
        rd_req_s: chan<ReadReq> out,
        rd_resp_r: chan<ReadResp> in,
        wr_req_s: chan<WriteReq> out,
        wr_resp_r: chan<WriteResp> in,
    ) {
        spawn AxiRamReader<
            INST_AXI_ADDR_W, INST_AXI_DATA_W, INST_AXI_DEST_W, INST_AXI_ID_W,
            INST_RAM_SIZE, INST_BASE_ADDR, INST_RAM_DATA_W, INST_RAM_ADDR_W, INST_RAM_NUM_PARTITIONS,
            INST_AXI_DATA_W_DIV8
        > (axi_ar_r, axi_r_s, rd_req_s, rd_resp_r);

        (
            wr_req_s, wr_resp_r
        )
    }

    next(state: ()) {
        send_if(join(), wr_req_s, false, zero!<WriteReq>());
        recv_if(join(), wr_resp_r, false, zero!<WriteResp>());
    }
}

const TEST_AXI_ADDR_W = u32:32;
const TEST_AXI_DATA_W = u32:32;
const TEST_AXI_DEST_W = u32:8;
const TEST_AXI_ID_W = u32:8;
const TEST_AXI_DATA_W_DIV8 = TEST_AXI_DATA_W / u32:8;

const TEST_RAM_SIZE = u32:100;
const TEST_RAM_DATA_W = TEST_AXI_DATA_W;
const TEST_RAM_ADDR_W = std::clog2(TEST_RAM_SIZE);
const TEST_RAM_WORD_PARTITION_SIZE = u32:8;
const TEST_RAM_NUM_PARTITIONS = TEST_RAM_DATA_W / TEST_RAM_WORD_PARTITION_SIZE;
const TEST_RAM_SIZE_BYTES = TEST_RAM_SIZE * (TEST_RAM_DATA_W / u32:8);

const TEST_BASE_ADDR = u32:0x8000;

type TestAxiAr = axi::AxiAr<TEST_AXI_ADDR_W, TEST_AXI_ID_W>;
type TestAxiR = axi::AxiR<TEST_AXI_DATA_W, TEST_AXI_ID_W>;

type TestReadReq = ram::ReadReq<TEST_RAM_ADDR_W, TEST_RAM_NUM_PARTITIONS>;
type TestReadResp = ram::ReadResp<TEST_RAM_DATA_W>;
type TestWriteReq = ram::WriteReq<TEST_RAM_ADDR_W, TEST_RAM_DATA_W, TEST_RAM_NUM_PARTITIONS>;
type TestWriteResp = ram::WriteResp;

const ZERO_AXI_AR_BUNDLE = zero!<TestAxiAr>();

type TestAxiId   = uN[TEST_AXI_ID_W];
type TestAxiAddr   = uN[TEST_AXI_ADDR_W];
type TestAxiRegion = uN[4];
type TestAxiLen    = uN[8];
type TestAxiSize   = axi::AxiAxSize;
type TestAxiBurst  = axi::AxiAxBurst;
type TestAxiCache  = axi::AxiArCache;
type TestAxiProt   = uN[3];
type TestAxiQos    = uN[4];

const TEST_RAM_DATA = u32[TEST_RAM_SIZE]:[
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

const TEST_AXI_AR_BUNDLES = TestAxiAr[16]:[
    AxiAr {
        id: TestAxiId:0,
        addr: TestAxiAddr:40,
        len: TestAxiLen:8,
        size: TestAxiSize::MAX_4B_TRANSFER,
        burst: TestAxiBurst::FIXED,
        ..ZERO_AXI_AR_BUNDLE
    },
    AxiAr {
        id: TestAxiId:0,
        addr: TestAxiAddr:440,
        len: TestAxiLen:8,
        size: TestAxiSize::MAX_4B_TRANSFER,
        burst: TestAxiBurst::FIXED,
        ..ZERO_AXI_AR_BUNDLE
    },
    AxiAr {
        id: TestAxiId:1,
        addr: TestAxiAddr:32,
        len: TestAxiLen:8,
        size: TestAxiSize::MAX_4B_TRANSFER,
        burst: TestAxiBurst::FIXED,
        ..ZERO_AXI_AR_BUNDLE
    },
    AxiAr {
        id: TestAxiId:2,
        addr: TestAxiAddr:16,
        len: TestAxiLen:8,
        size: TestAxiSize::MAX_4B_TRANSFER,
        burst: TestAxiBurst::INCR,
        ..ZERO_AXI_AR_BUNDLE
    },
    AxiAr {
        id: TestAxiId:3,
        addr: TestAxiAddr:92,
        len: TestAxiLen:4,
        size: TestAxiSize::MAX_4B_TRANSFER,
        burst: TestAxiBurst::INCR,
        ..ZERO_AXI_AR_BUNDLE
    },
    AxiAr {
        id: TestAxiId:4,
        addr: TestAxiAddr:0,
        len: TestAxiLen:2,
        size: TestAxiSize::MAX_4B_TRANSFER,
        burst: TestAxiBurst::INCR,
        ..ZERO_AXI_AR_BUNDLE
    },
    AxiAr {
        id: TestAxiId:5,
        addr: TestAxiAddr:52,
        len: TestAxiLen:20,
        size: TestAxiSize::MAX_4B_TRANSFER,
        burst: TestAxiBurst::INCR,
        ..ZERO_AXI_AR_BUNDLE
    },
    AxiAr {
        id: TestAxiId:6,
        addr: TestAxiAddr:96,
        len: TestAxiLen:10,
        size: TestAxiSize::MAX_4B_TRANSFER,
        burst: TestAxiBurst::INCR,
        ..ZERO_AXI_AR_BUNDLE
    },
    AxiAr {
        id: TestAxiId:7,
        addr: TestAxiAddr:128,
        len: TestAxiLen:16,
        size: TestAxiSize::MAX_4B_TRANSFER,
        burst: TestAxiBurst::WRAP,
        ..ZERO_AXI_AR_BUNDLE
    },
    AxiAr {
        id: TestAxiId:8,
        addr: TestAxiAddr:256,
        len: TestAxiLen:2,
        size: TestAxiSize::MAX_4B_TRANSFER,
        burst: TestAxiBurst::WRAP,
        ..ZERO_AXI_AR_BUNDLE
    },
    AxiAr {
        id: TestAxiId:9,
        addr: TestAxiAddr:32,
        len: TestAxiLen:4,
        size: TestAxiSize::MAX_2B_TRANSFER,
        burst: TestAxiBurst::FIXED,
        ..ZERO_AXI_AR_BUNDLE
    },
    AxiAr {
        id: TestAxiId:10,
        addr: TestAxiAddr:80,
        len: TestAxiLen:4,
        size: TestAxiSize::MAX_1B_TRANSFER,
        burst: TestAxiBurst::INCR,
        ..ZERO_AXI_AR_BUNDLE
    },
    AxiAr {
        id: TestAxiId:11,
        addr: TestAxiAddr:256,
        len: TestAxiLen:16,
        size: TestAxiSize::MAX_2B_TRANSFER,
        burst: TestAxiBurst::WRAP,
        ..ZERO_AXI_AR_BUNDLE
    },
    AxiAr {
        id: TestAxiId:12,
        addr: TestAxiAddr:64,
        len: TestAxiLen:2,
        size: TestAxiSize::MAX_8B_TRANSFER,
        burst: TestAxiBurst::FIXED,
        ..ZERO_AXI_AR_BUNDLE
    },
    AxiAr {
        id: TestAxiId:13,
        addr: TestAxiAddr:192,
        len: TestAxiLen:16,
        size: TestAxiSize::MAX_64B_TRANSFER,
        burst: TestAxiBurst::INCR,
        ..ZERO_AXI_AR_BUNDLE
    },
    AxiAr {
        id: TestAxiId:14,
        addr: TestAxiAddr:16,
        len: TestAxiLen:16,
        size: TestAxiSize::MAX_128B_TRANSFER,
        burst: TestAxiBurst::INCR,
        ..ZERO_AXI_AR_BUNDLE
    },
];

#[test_proc]
proc AxiRamReaderTest {
    terminator: chan<bool> out;

    axi_ar_s: chan<TestAxiAr> out;
    axi_r_r: chan<TestAxiR> in;

    wr_req_s: chan<TestWriteReq> out;
    wr_resp_r: chan<TestWriteResp> in;

    init {}

    config(
        terminator: chan<bool> out,
    ) {
        let (rd_req_s, rd_req_r) = chan<TestReadReq>("rd_req");
        let (rd_resp_s, rd_resp_r) = chan<TestReadResp>("rd_resp");
        let (wr_req_s, wr_req_r) = chan<TestWriteReq>("wr_req");
        let (wr_resp_s, wr_resp_r) = chan<TestWriteResp>("wr_resp");

        spawn ram::RamModel<TEST_RAM_DATA_W, TEST_RAM_SIZE, TEST_RAM_WORD_PARTITION_SIZE> (
            rd_req_r, rd_resp_s, wr_req_r, wr_resp_s
        );

        let (axi_ar_s, axi_ar_r) = chan<TestAxiAr>("axi_ar");
        let (axi_r_s, axi_r_r) = chan<TestAxiR>("axi_r");

        spawn AxiRamReader<
            TEST_AXI_ADDR_W, TEST_AXI_DATA_W, TEST_AXI_DEST_W, TEST_AXI_ID_W,
            TEST_RAM_SIZE, TEST_BASE_ADDR, TEST_RAM_DATA_W, TEST_RAM_ADDR_W, TEST_RAM_NUM_PARTITIONS,
            TEST_AXI_DATA_W_DIV8,
        >(axi_ar_r, axi_r_s, rd_req_s, rd_resp_r);

        (
            terminator,
            axi_ar_s, axi_r_r, wr_req_s, wr_resp_r,
        )
    }

    next(state: ()) {
        type RamAddr = bits[TEST_RAM_ADDR_W];
        type RamData = bits[TEST_RAM_DATA_W];
        type RamMask = bits[TEST_RAM_NUM_PARTITIONS];

        let tok = join();

        // write test RAM data
        let tok = for ((i, data), tok): ((u32, u32), token) in enumerate(TEST_RAM_DATA) {
            let tok = send(tok, wr_req_s, TestWriteReq {
                addr: i as RamAddr,
                data: data,
                mask: !bits[TEST_RAM_NUM_PARTITIONS]:0,
            });
            let (tok, _) = recv(tok, wr_resp_r);

            tok
        }(tok);

        let tok = for ((_i, axi_ar_bundle), tok): ((u32, TestAxiAr), token) in enumerate(TEST_AXI_AR_BUNDLES) {
            let tok = send(tok, axi_ar_s, axi_ar_bundle);
            // trace_fmt!("Sent bundle #{} {:#x}", i + u32:1, axi_ar_bundle);

            let size_valid = (u32:1 << (axi_ar_bundle.size as u32 + u32:3)) <= TEST_AXI_DATA_W;

            let data_len = if size_valid {
                axi_ar_bundle.len as u32
            } else {
                u32:0
            };

            for (j, tok): (u32, token) in range(u32:0, TEST_RAM_SIZE) {
                if (j <= data_len) {
                    let (tok, data) = recv(tok, axi_r_r);
                    trace_fmt!("Received data #{} {:#x}", j, data);
                    // compute address
                    let araddr = match axi_ar_bundle.burst {
                        AxiAxBurst::FIXED => {
                            axi_ar_bundle.addr
                        },
                        AxiAxBurst::INCR => {
                            axi_ar_bundle.addr + j * (u32:1 << (axi_ar_bundle.size as u32))
                        },
                        AxiAxBurst::WRAP => {
                            (axi_ar_bundle.addr + j * (u32:1 << (axi_ar_bundle.size as u32))) % (TEST_RAM_SIZE * (TEST_RAM_DATA_W / u32:8))
                        },
                    };
                    // create expected data using RAM data
                    let (expected_data, addr_valid) = for (k, (expected_data, addr_valid)): (u32, (uN[TEST_AXI_DATA_W], bool)) in range(u32:0, TEST_AXI_DATA_W / u32:8) {
                        if k < (u32:1 << (axi_ar_bundle.size as u32)) {
                            let ram_addr = (araddr + k) / (TEST_RAM_DATA_W / u32:8);
                            let ram_offset = ((araddr + k) % (TEST_RAM_DATA_W / u32:8)) * u32:8;
                            if ram_addr < TEST_RAM_SIZE {
                                (
                                    expected_data | (((TEST_RAM_DATA[ram_addr] >> ram_offset) & u32:0xFF) << (u32:8 * k)),
                                    addr_valid,
                                )
                            } else {
                                (
                                    uN[TEST_AXI_DATA_W]:0,
                                    false,
                                )
                            }
                        } else {
                            (
                                expected_data,
                                addr_valid
                            )
                        }
                    }((uN[TEST_AXI_DATA_W]:0, true));

                    let expected_rresp = if !size_valid {
                        AxiReadResp::SLVERR
                    } else if addr_valid {
                        AxiReadResp::OKAY
                    } else {
                        AxiReadResp::DECERR
                    };

                    assert_eq(expected_rresp, data.resp);
                    assert_eq(j == data_len, data.last);
                    assert_eq(axi_ar_bundle.id, data.id);
                    if expected_rresp == AxiReadResp::OKAY {
                        // valid read
                        assert_eq(expected_data, data.data);
                    } else { };
                    tok
                } else { tok }
            }(tok)
        }(tok);

        send(tok, terminator, true);
    }
}


