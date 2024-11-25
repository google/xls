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

// This file contains implementation of HistoryBuffer.


import std;

import xls.examples.ram as ram;
import xls.modules.zstd.aligned_parallel_ram as aligned_parallel_ram;

const RAM_NUM = aligned_parallel_ram::RAM_NUM;
const RAM_NUM_W = aligned_parallel_ram::RAM_NUM_W;


pub struct HistoryBufferReadReq<OFFSET_W: u32> {
    offset: uN[OFFSET_W],
}

pub struct HistoryBufferReadResp<DATA_W: u32> {
    data: uN[DATA_W],
}

pub struct HistoryBufferWriteReq<DATA_W: u32> {
    data: uN[DATA_W],
}

pub struct HistoryBufferWriteResp {}

struct HistoryBufferState<OFFSET_W: u32> {
    curr_offset: uN[OFFSET_W],
    length: uN[OFFSET_W],
}

proc HistoryBufferReadRespHandler<DATA_W: u32> {
    type ReadResp = HistoryBufferReadResp<DATA_W>;
    type ParallelRamReadResp = aligned_parallel_ram::AlignedParallelRamReadResp<DATA_W>;

    parallel_ram_read_resp_r: chan<ParallelRamReadResp> in;
    read_resp_s: chan<ReadResp> out;

    config(
        parallel_ram_read_resp_r: chan<ParallelRamReadResp> in,
        read_resp_s: chan<ReadResp> out,
    ) {
        (
            parallel_ram_read_resp_r,
            read_resp_s,
        )
    }

    init { }

    next (state: ()) {
        let (tok, parallel_ram_read_resp, parallel_ram_read_resp_valid) = recv_non_blocking(
            join(), parallel_ram_read_resp_r, zero!<ParallelRamReadResp>()
        );

        let read_resp = ReadResp {
            data: parallel_ram_read_resp.data,
        };

        send_if(tok, read_resp_s, parallel_ram_read_resp_valid, read_resp);
    }
}

proc HistoryBufferWriteRespHandler {
    type WriteResp = HistoryBufferWriteResp;
    type ParallelRamWriteResp = aligned_parallel_ram::AlignedParallelRamWriteResp;

    parallel_ram_write_resp_r: chan<ParallelRamWriteResp> in;
    write_resp_s: chan<WriteResp> out;

    config(
        parallel_ram_write_resp_r: chan<ParallelRamWriteResp> in,
        write_resp_s: chan<WriteResp> out,
    ) {
        (
            parallel_ram_write_resp_r,
            write_resp_s,
        )
    }

    init { }

    next (state: ()) {
        let (tok, _, parallel_ram_write_resp_valid) = recv_non_blocking(
            join(), parallel_ram_write_resp_r, zero!<ParallelRamWriteResp>()
        );

        let write_resp = WriteResp { };

        send_if(tok, write_resp_s, parallel_ram_write_resp_valid, write_resp);
    }
}

pub proc HistoryBuffer<
    SIZE: u32,
    DATA_W: u32,
    OFFSET_W: u32 = {std::clog2(SIZE)},
    RAM_SIZE: u32 = {SIZE / RAM_NUM},
    RAM_DATA_W: u32 = {DATA_W / RAM_NUM},
    RAM_ADDR_W: u32 = {std::clog2(RAM_SIZE)},
    RAM_PARTITION_SIZE: u32 = {RAM_DATA_W},
    RAM_NUM_PARTITIONS: u32 = {ram::num_partitions(RAM_PARTITION_SIZE, RAM_DATA_W)},
>{
    type ReadReq = HistoryBufferReadReq<OFFSET_W>;
    type ReadResp = HistoryBufferReadResp<DATA_W>;
    type WriteReq = HistoryBufferWriteReq<DATA_W>;
    type WriteResp = HistoryBufferWriteResp;

    type ParallelRamReadReq = aligned_parallel_ram::AlignedParallelRamReadReq<OFFSET_W>;
    type ParallelRamReadResp = aligned_parallel_ram::AlignedParallelRamReadResp<DATA_W>;
    type ParallelRamWriteReq = aligned_parallel_ram::AlignedParallelRamWriteReq<OFFSET_W, DATA_W>;
    type ParallelRamWriteResp = aligned_parallel_ram::AlignedParallelRamWriteResp;

    type RamReadReq = ram::ReadReq<RAM_ADDR_W, RAM_NUM_PARTITIONS>;
    type RamReadResp = ram::ReadResp<RAM_DATA_W>;
    type RamWriteReq = ram::WriteReq<RAM_ADDR_W, RAM_DATA_W, RAM_NUM_PARTITIONS>;
    type RamWriteResp = ram::WriteResp;

    type State = HistoryBufferState<OFFSET_W>;
    type Offset = uN[OFFSET_W];

    read_req_r: chan<ReadReq> in;
    write_req_r: chan<WriteReq> in;

    // RAM interface
    parallel_ram_read_req_s: chan<ParallelRamReadReq> out;
    parallel_ram_write_req_s: chan<ParallelRamWriteReq> out;

    config (
        read_req_r: chan<ReadReq> in,
        read_resp_s: chan<ReadResp> out,
        write_req_r: chan<WriteReq> in,
        write_resp_s: chan<WriteResp> out,
        ram_read_req_s: chan<RamReadReq>[RAM_NUM] out,
        ram_read_resp_r: chan<RamReadResp>[RAM_NUM] in,
        ram_write_req_s: chan<RamWriteReq>[RAM_NUM] out,
        ram_write_resp_r: chan<RamWriteResp>[RAM_NUM] in,
    ) {
        let (parallel_ram_read_req_s, parallel_ram_read_req_r) = chan<ParallelRamReadReq, u32:1>("parallel_ram_read_req");
        let (parallel_ram_read_resp_s, parallel_ram_read_resp_r) = chan<ParallelRamReadResp, u32:1>("parallel_ram_read_resp");
        let (parallel_ram_write_req_s, parallel_ram_write_req_r) = chan<ParallelRamWriteReq, u32:1>("parallel_ram_write_req");
        let (parallel_ram_write_resp_s, parallel_ram_write_resp_r) = chan<ParallelRamWriteResp, u32:1>("parallel_ram_write_resp");

        spawn HistoryBufferReadRespHandler<DATA_W> (
            parallel_ram_read_resp_r, read_resp_s,
        );

        spawn HistoryBufferWriteRespHandler (
            parallel_ram_write_resp_r, write_resp_s,
        );

        spawn aligned_parallel_ram::AlignedParallelRam<SIZE, DATA_W>(
            parallel_ram_read_req_r, parallel_ram_read_resp_s,
            parallel_ram_write_req_r, parallel_ram_write_resp_s,
            ram_read_req_s, ram_read_resp_r,
            ram_write_req_s, ram_write_resp_r,
        );

        (
            read_req_r,
            write_req_r,
            parallel_ram_read_req_s,
            parallel_ram_write_req_s,
        )
    }

    init { zero!<State>() }

    next (state: State) {
        const ONE_TRANSFER_WIDTH = ((DATA_W as uN[OFFSET_W]) >> u32:3);
        const MAX_OFFSET = zero!<Offset>();

        // handle read request
        let (tok, read_req, read_req_valid) = recv_non_blocking(join(), read_req_r, zero!<ReadReq>());
        let offset_invalid = (read_req.offset > state.length - ONE_TRANSFER_WIDTH);
        if read_req_valid & offset_invalid {
            trace_fmt!("WARNING: Asking for too high offset (req: {:#x}, max: {:#x})",
                       read_req.offset, state.length - ONE_TRANSFER_WIDTH);
        } else {};

        let parallel_ram_read_req = ParallelRamReadReq {
            addr: state.curr_offset - read_req.offset - ONE_TRANSFER_WIDTH,
        };

        send_if(tok, parallel_ram_read_req_s, read_req_valid, parallel_ram_read_req);

        // handle write request
        let (tok, write_req, write_req_valid) = recv_non_blocking(join(), write_req_r, zero!<WriteReq>());

        let parallel_ram_write_req = ParallelRamWriteReq {
            addr: state.curr_offset,
            data: write_req.data,
        };

        send_if(tok, parallel_ram_write_req_s, write_req_valid, parallel_ram_write_req);

        // update offset
        if write_req_valid {
            let next_length = if state.length > MAX_OFFSET - ONE_TRANSFER_WIDTH {
                MAX_OFFSET
            } else {
                state.length + ONE_TRANSFER_WIDTH
            };

            State {
                curr_offset: state.curr_offset + ONE_TRANSFER_WIDTH,
                length: next_length,
            }
        } else {
            state
        }
    }
}

const INST_SIZE = u32:1024;
const INST_DATA_W = u32:64;
const INST_OFFSET_W = {std::clog2(INST_SIZE)};
const INST_RAM_SIZE = INST_SIZE / RAM_NUM;
const INST_RAM_DATA_W = {INST_DATA_W / RAM_NUM};
const INST_RAM_ADDR_W = {std::clog2(INST_RAM_SIZE)};
const INST_RAM_PARTITION_SIZE = {INST_RAM_DATA_W};
const INST_RAM_NUM_PARTITIONS = {ram::num_partitions(INST_RAM_PARTITION_SIZE, INST_RAM_DATA_W)};

proc HistoryBufferInst {
    type InstReadReq = HistoryBufferReadReq<INST_OFFSET_W>;
    type InstReadResp = HistoryBufferReadResp<INST_DATA_W>;
    type InstWriteReq = HistoryBufferWriteReq<INST_DATA_W>;
    type InstWriteResp = HistoryBufferWriteResp;

    type InstParallelRamReadReq = aligned_parallel_ram::AlignedParallelRamReadReq<INST_OFFSET_W>;
    type InstParallelRamReadResp = aligned_parallel_ram::AlignedParallelRamReadResp<INST_DATA_W>;
    type InstParallelRamWriteReq = aligned_parallel_ram::AlignedParallelRamWriteReq<INST_OFFSET_W, INST_DATA_W>;
    type InstParallelRamWriteResp = aligned_parallel_ram::AlignedParallelRamWriteResp;

    type InstRamReadReq = ram::ReadReq<INST_RAM_ADDR_W, INST_RAM_NUM_PARTITIONS>;
    type InstRamReadResp = ram::ReadResp<INST_RAM_DATA_W>;
    type InstRamWriteReq = ram::WriteReq<INST_RAM_ADDR_W, INST_RAM_DATA_W, INST_RAM_NUM_PARTITIONS>;
    type InstRamWriteResp = ram::WriteResp;

    config (
        read_req_r: chan<InstReadReq> in,
        read_resp_s: chan<InstReadResp> out,
        write_req_r: chan<InstWriteReq> in,
        write_resp_s: chan<InstWriteResp> out,
        ram_read_req_s: chan<InstRamReadReq>[RAM_NUM] out,
        ram_read_resp_r: chan<InstRamReadResp>[RAM_NUM] in,
        ram_write_req_s: chan<InstRamWriteReq>[RAM_NUM] out,
        ram_write_resp_r: chan<InstRamWriteResp>[RAM_NUM] in,
    ) {
        spawn HistoryBuffer<INST_SIZE, INST_DATA_W> (
            read_req_r, read_resp_s,
            write_req_r, write_resp_s,
            ram_read_req_s, ram_read_resp_r,
            ram_write_req_s, ram_write_resp_r,
        );
    }

    init { }

    next (state: ()) { }
}

const TEST_SIZE = u32:128;
const TEST_DATA_W = u32:64;
const TEST_OFFSET_W = {std::clog2(TEST_SIZE)};
const TEST_RAM_SIZE = TEST_SIZE / aligned_parallel_ram::RAM_NUM;
const TEST_RAM_DATA_W = {TEST_DATA_W / RAM_NUM};
const TEST_RAM_ADDR_W = {std::clog2(TEST_RAM_SIZE)};
const TEST_RAM_PARTITION_SIZE = {TEST_RAM_DATA_W};
const TEST_RAM_NUM_PARTITIONS = {ram::num_partitions(TEST_RAM_PARTITION_SIZE, TEST_RAM_DATA_W)};

const TEST_RAM_SIMULTANEOUS_RW_BEHAVIOR = ram::SimultaneousReadWriteBehavior::READ_BEFORE_WRITE;
const TEST_RAM_INITIALIZED = true;

type TestReadReq = HistoryBufferReadReq<TEST_OFFSET_W>;
type TestReadResp = HistoryBufferReadResp<TEST_DATA_W>;
type TestWriteReq = HistoryBufferWriteReq<TEST_DATA_W>;
type TestWriteResp = HistoryBufferWriteResp;

type TestRamReadReq = ram::ReadReq<TEST_RAM_ADDR_W, TEST_RAM_NUM_PARTITIONS>;
type TestRamReadResp = ram::ReadResp<TEST_RAM_DATA_W>;
type TestRamWriteReq = ram::WriteReq<TEST_RAM_ADDR_W, TEST_RAM_DATA_W, TEST_RAM_NUM_PARTITIONS>;
type TestRamWriteResp = ram::WriteResp;

const TEST_DATA = uN[TEST_DATA_W][64]:[
    uN[TEST_DATA_W]:0x44f9_072b_5ef2_8a80, uN[TEST_DATA_W]:0x4c01_2eda_5f3d_f4e1,
    uN[TEST_DATA_W]:0x75ac_641d_fd42_0551, uN[TEST_DATA_W]:0x3bd0_3798_b29b_725f,
    uN[TEST_DATA_W]:0x840e_71e9_e0e6_4fe1, uN[TEST_DATA_W]:0x9436_c1e7_bf3c_14e2,
    uN[TEST_DATA_W]:0x1c64_9595_300d_4e0c, uN[TEST_DATA_W]:0x26ad_a821_926d_d9e5,
    uN[TEST_DATA_W]:0x27e4_6ce8_a2b4_3a71, uN[TEST_DATA_W]:0xf9d6_cf94_6a39_5c5d,
    uN[TEST_DATA_W]:0x7894_d415_88b5_dd0c, uN[TEST_DATA_W]:0x5c4e_4607_96bc_5d54,
    uN[TEST_DATA_W]:0x29a4_3388_8b44_d9eb, uN[TEST_DATA_W]:0xda83_ee49_d921_7fb6,
    uN[TEST_DATA_W]:0xf25f_c785_12e6_dfd0, uN[TEST_DATA_W]:0x1b8c_06fb_32ea_165a,
    uN[TEST_DATA_W]:0x5dda_92e0_faca_af84, uN[TEST_DATA_W]:0x1157_8f9a_6e5c_7e78,
    uN[TEST_DATA_W]:0xc908_a151_5b8c_b908, uN[TEST_DATA_W]:0xe978_4a80_f2e9_b11a,
    uN[TEST_DATA_W]:0xc34e_96c0_4ae1_dfa9, uN[TEST_DATA_W]:0x4b06_4c8c_df6d_cae5,
    uN[TEST_DATA_W]:0x9d51_4716_fd6f_afe9, uN[TEST_DATA_W]:0xfe42_4a9d_29ae_4bc4,
    uN[TEST_DATA_W]:0x77ec_b4dd_9238_38b9, uN[TEST_DATA_W]:0xdf45_a790_a3da_1768,
    uN[TEST_DATA_W]:0x45e9_5594_ffca_0604, uN[TEST_DATA_W]:0xe496_09f4_18ca_f955,
    uN[TEST_DATA_W]:0x57cb_3c3d_ed78_62fd, uN[TEST_DATA_W]:0x0254_24bc_24fa_99f8,
    uN[TEST_DATA_W]:0xc405_370f_a58a_1303, uN[TEST_DATA_W]:0xa451_310a_65b2_4785,
    uN[TEST_DATA_W]:0x4373_65ac_f3ce_97ec, uN[TEST_DATA_W]:0x2a85_abd3_afde_133c,
    uN[TEST_DATA_W]:0x836e_ce62_56cb_50ec, uN[TEST_DATA_W]:0x53ce_ab2f_d079_eb9a,
    uN[TEST_DATA_W]:0xae76_7db7_0e64_8b88, uN[TEST_DATA_W]:0x079a_c187_642d_cbac,
    uN[TEST_DATA_W]:0x2d07_5e3b_6150_d5c5, uN[TEST_DATA_W]:0x7865_5206_3c5a_98ed,
    uN[TEST_DATA_W]:0xe905_351c_edda_0682, uN[TEST_DATA_W]:0xf41d_f3f2_1106_3639,
    uN[TEST_DATA_W]:0xa44c_05c0_24b3_86ad, uN[TEST_DATA_W]:0xaa1f_c6b5_4c02_1f0c,
    uN[TEST_DATA_W]:0xad67_cc1a_8740_87ae, uN[TEST_DATA_W]:0xf382_3bbf_f4b8_2f81,
    uN[TEST_DATA_W]:0xe0cd_1eb3_b8c0_820b, uN[TEST_DATA_W]:0xb5d5_1c98_3415_1319,
    uN[TEST_DATA_W]:0x583e_9722_ed31_84e6, uN[TEST_DATA_W]:0x6063_ccb6_6228_286e,
    uN[TEST_DATA_W]:0xc642_cca8_e04f_769e, uN[TEST_DATA_W]:0x7cc7_ab72_7a9c_05d8,
    uN[TEST_DATA_W]:0x4a66_f7c1_7b5e_6d30, uN[TEST_DATA_W]:0xd3d2_5e04_0310_7689,
    uN[TEST_DATA_W]:0xe99d_a201_5dee_8e16, uN[TEST_DATA_W]:0xee15_ca30_c679_e1dd,
    uN[TEST_DATA_W]:0xe61c_4ac3_183e_9478, uN[TEST_DATA_W]:0x2528_e948_2349_f8fd,
    uN[TEST_DATA_W]:0xf15d_4275_a042_2135, uN[TEST_DATA_W]:0x05b5_3768_34e9_4bca,
    uN[TEST_DATA_W]:0x1e00_a1a9_cffd_7a84, uN[TEST_DATA_W]:0x3396_a42c_2433_76f2,
    uN[TEST_DATA_W]:0x80ba_e00e_9b93_7d76, uN[TEST_DATA_W]:0x85d4_10e6_404f_fa4d,
];

#[test_proc]
proc HistoryBuffer_test {
    terminator: chan<bool> out;

    read_req_s: chan<TestReadReq> out;
    read_resp_r: chan<TestReadResp> in;
    write_req_s: chan<TestWriteReq> out;
    write_resp_r: chan<TestWriteResp> in;

    config (terminator: chan<bool> out) {
        let (read_req_s, read_req_r) = chan<TestReadReq, u32:1>("read_req");
        let (read_resp_s, read_resp_r) = chan<TestReadResp, u32:1>("read_resp");
        let (write_req_s, write_req_r) = chan<TestWriteReq, u32:1>("write_req");
        let (write_resp_s, write_resp_r) = chan<TestWriteResp, u32:1>("write_resp");

        let (ram_read_req_s, ram_read_req_r) = chan<TestRamReadReq, u32:1>[RAM_NUM]("ram_read_req");
        let (ram_read_resp_s, ram_read_resp_r) = chan<TestRamReadResp, u32:1>[RAM_NUM]("ram_read_resp");
        let (ram_write_req_s, ram_write_req_r) = chan<TestRamWriteReq, u32:1>[RAM_NUM]("ram_write_req");
        let (ram_write_resp_s, ram_write_resp_r) = chan<TestRamWriteResp, u32:1>[RAM_NUM]("ram_write_resp");

        spawn ram::RamModel<
            TEST_RAM_DATA_W, TEST_RAM_SIZE, TEST_RAM_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_RW_BEHAVIOR, TEST_RAM_INITIALIZED
        >(
            ram_read_req_r[0], ram_read_resp_s[0], ram_write_req_r[0], ram_write_resp_s[0],
        );

        spawn ram::RamModel<
            TEST_RAM_DATA_W, TEST_RAM_SIZE, TEST_RAM_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_RW_BEHAVIOR, TEST_RAM_INITIALIZED
        >(
            ram_read_req_r[1], ram_read_resp_s[1], ram_write_req_r[1], ram_write_resp_s[1],
        );

        spawn ram::RamModel<
            TEST_RAM_DATA_W, TEST_RAM_SIZE, TEST_RAM_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_RW_BEHAVIOR, TEST_RAM_INITIALIZED
        >(
            ram_read_req_r[2], ram_read_resp_s[2], ram_write_req_r[2], ram_write_resp_s[2],
        );

        spawn ram::RamModel<
            TEST_RAM_DATA_W, TEST_RAM_SIZE, TEST_RAM_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_RW_BEHAVIOR, TEST_RAM_INITIALIZED
        >(
            ram_read_req_r[3], ram_read_resp_s[3], ram_write_req_r[3], ram_write_resp_s[3],
        );

        spawn ram::RamModel<
            TEST_RAM_DATA_W, TEST_RAM_SIZE, TEST_RAM_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_RW_BEHAVIOR, TEST_RAM_INITIALIZED
        >(
            ram_read_req_r[4], ram_read_resp_s[4], ram_write_req_r[4], ram_write_resp_s[4],
        );

        spawn ram::RamModel<
            TEST_RAM_DATA_W, TEST_RAM_SIZE, TEST_RAM_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_RW_BEHAVIOR, TEST_RAM_INITIALIZED
        >(
            ram_read_req_r[5], ram_read_resp_s[5], ram_write_req_r[5], ram_write_resp_s[5],
        );

        spawn ram::RamModel<
            TEST_RAM_DATA_W, TEST_RAM_SIZE, TEST_RAM_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_RW_BEHAVIOR, TEST_RAM_INITIALIZED
        >(
            ram_read_req_r[6], ram_read_resp_s[6], ram_write_req_r[6], ram_write_resp_s[6],
        );

        spawn ram::RamModel<
            TEST_RAM_DATA_W, TEST_RAM_SIZE, TEST_RAM_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_RW_BEHAVIOR, TEST_RAM_INITIALIZED
        >(
            ram_read_req_r[7], ram_read_resp_s[7], ram_write_req_r[7], ram_write_resp_s[7],
        );

        spawn HistoryBuffer<TEST_SIZE, TEST_DATA_W> (
            read_req_r, read_resp_s,
            write_req_r, write_resp_s,
            ram_read_req_s, ram_read_resp_r,
            ram_write_req_s, ram_write_resp_r,
        );

        (
            terminator,
            read_req_s, read_resp_r,
            write_req_s, write_resp_r,
        )
    }

    init { }

    next (state: ()) {
        let tok = join();

        let tok = for (i, tok) in range(u32:0, array_size(TEST_DATA)) {
            let test_data = TEST_DATA[i];

            // write current test data
            let write_req = TestWriteReq {
                data: test_data,
            };
            let tok = send(tok, write_req_s, write_req);
            trace_fmt!("Sent #{} write request {:#x}", i + u32:1, write_req);

            let (tok, _) = recv(tok, write_resp_r);
            trace_fmt!("Received #{} write response", i + u32:1);

            // check written data
            let read_req = TestReadReq {
                offset: uN[TEST_OFFSET_W]:0,
            };
            let tok = send(tok, read_req_s, read_req);
            trace_fmt!("Sent #{} read request {:#x}", i + u32:1, read_req);

            let (tok, read_resp) = recv(tok, read_resp_r);
            trace_fmt!("Received #{} read response {:#x}", i + u32:1, read_resp);
            assert_eq(test_data, read_resp.data);

            // check previously saved data
            let tok = for (offset, tok) in range(u32:0, TEST_SIZE - TEST_DATA_W) {
                // check only offsets where data was written also dont check for all offsets
                // to speedup the test
                if (offset < RAM_NUM * i) && ((offset + i) % u32:13 == u32:0) {
                    let data_idx_0 = ((i * RAM_NUM - offset) >> RAM_NUM_W) % array_size(TEST_DATA);
                    let data_idx_1 = (((i * RAM_NUM - offset) >> RAM_NUM_W) + u32:1) % array_size(TEST_DATA);
                    let ram_offset = (offset) as uN[RAM_NUM_W];
                    let prev_test_data = if ram_offset == uN[RAM_NUM_W]:0 {
                        TEST_DATA[data_idx_0]
                    } else {
                        (
                            TEST_DATA[data_idx_1] << (TEST_RAM_DATA_W * ram_offset as u32) |
                            TEST_DATA[data_idx_0] >> (TEST_RAM_DATA_W * (aligned_parallel_ram::RAM_NUM - ram_offset as u32))
                        )
                    };

                    let read_req = TestReadReq {
                        offset: offset as uN[TEST_OFFSET_W],
                    };
                    let tok = send(tok, read_req_s, read_req);
                    trace_fmt!("Sent #{}.{} read request {:#x}", i + u32:1, offset, read_req);

                    let (tok, read_resp) = recv(tok, read_resp_r);
                    trace_fmt!("Received #{}.{} read response {:#x}", i + u32:1, offset, read_resp);
                    assert_eq(prev_test_data, read_resp.data);

                    tok
                } else {
                    tok
                }
            }(tok);

            tok
        }(tok);

        send(tok, terminator, true);
    }
}
