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

// This file contains the implementation of Huffmann tree decoder.

import std;
import xls.dslx.stdlib.acm_random as random;

import xls.examples.ram;
import xls.modules.zstd.common as common;
import xls.modules.zstd.huffman_common as hcommon;

// TODO: Enable once parametrics work
//fn WeightPreScanMetaDataSize(PARALLEL_ACCESS_WIDTH: u32) -> u32 {
//    let COUNTER_WIDTH = {std::clog2(PARALLEL_ACCESS_WIDTH + u32:1)};
//    (COUNTER_WIDTH as u32) * (PARALLEL_ACCESS_WIDTH as u32) +
//    (MAX_WEIGHT as u32) + u32:1 +
//    (COUNTER_WIDTH as u32) * (MAX_WEIGHT as u32 + u32:1)
//}
//
//fn InternalStructToBits<
//    PARALLEL_ACCESS_WIDTH: u32,
//    BITS: u32 = {WeightPreScanMetaDataSize(PARALLEL_ACCESS_WIDTH)}
//> (internalStruct: WeightPreScanMetaData<PARALLEL_ACCESS_WIDTH>) -> bits[BITS] {
//    internalStruct as bits[BITS]
//}
//
//fn BitsToInternalStruct<
//    PARALLEL_ACCESS_WIDTH: u32,
//    BITS: u32 = {WeightPreScanMetaDataSize(PARALLEL_ACCESS_WIDTH)}
//> (rawBits: bits[BITS]) -> WeightPreScanMetaData<PARALLEL_ACCESS_WIDTH> {
//    rawBits as WeightPreScanMetaData<PARALLEL_ACCESS_WIDTH>
//}

const MAX_WEIGHT = hcommon::MAX_WEIGHT;
const WEIGHT_LOG = hcommon::WEIGHT_LOG;
const MAX_SYMBOL_COUNT = hcommon::MAX_SYMBOL_COUNT;

const PARALLEL_ACCESS_WIDTH = hcommon::PARALLEL_ACCESS_WIDTH;
const COUNTER_WIDTH = hcommon::COUNTER_WIDTH;

type WeightPreScanMetaData = hcommon::WeightPreScanMetaData;
type WeightPreScanOutput = hcommon::WeightPreScanOutput;

pub fn WeightPreScanMetaDataSize() -> u32 {
    (COUNTER_WIDTH as u32) * (PARALLEL_ACCESS_WIDTH as u32) +
    (MAX_WEIGHT as u32) + u32:1 +
    (COUNTER_WIDTH as u32) * (MAX_WEIGHT as u32 + u32:1)
}

fn InternalStructToBits<
    BITS: u32 = {WeightPreScanMetaDataSize()},
    OCCURANCE_WIDTH: u32 ={COUNTER_WIDTH * PARALLEL_ACCESS_WIDTH},
> (internalStruct: WeightPreScanMetaData) -> bits[BITS] {
    (internalStruct.weights_count as bits[COUNTER_WIDTH * (MAX_WEIGHT + u32:1)] ++
     internalStruct.valid_weights as bits[MAX_WEIGHT + u32:1] ++
     internalStruct.occurance_number as bits[OCCURANCE_WIDTH]) as bits[BITS]
}

fn BitsToInternalStruct<
    BITS: u32 = {WeightPreScanMetaDataSize()},
    OCCURANCE_WIDTH: u32 ={COUNTER_WIDTH * PARALLEL_ACCESS_WIDTH},
> (rawBits: bits[BITS]) -> WeightPreScanMetaData {
    WeightPreScanMetaData<PARALLEL_ACCESS_WIDTH> {
        occurance_number: rawBits[0:OCCURANCE_WIDTH as s32] as uN[COUNTER_WIDTH][PARALLEL_ACCESS_WIDTH],
        valid_weights:    rawBits[OCCURANCE_WIDTH as s32:(OCCURANCE_WIDTH + MAX_WEIGHT + u32:1) as s32] as u1[MAX_WEIGHT + u32:1],
        weights_count:    rawBits[(OCCURANCE_WIDTH + MAX_WEIGHT + u32:1) as s32:BITS as s32] as uN[COUNTER_WIDTH][MAX_WEIGHT + u32:1]
    }
}

#[quickcheck(test_count=50000)]
fn bits_to_struct_to_bits_qtest(x: bits[WeightPreScanMetaDataSize()]) -> bool {
    x == InternalStructToBits(BitsToInternalStruct(x))
}

#[quickcheck(test_count=50000)]
fn struct_to_bots_to_struct_qtest(x: WeightPreScanMetaData) -> bool {
  x == BitsToInternalStruct(InternalStructToBits(x))
}

pub const RAM_SIZE = MAX_SYMBOL_COUNT/PARALLEL_ACCESS_WIDTH * u32:8 / WEIGHT_LOG;
pub const RAM_ADDR_WIDTH = {std::clog2(RAM_SIZE)};
pub const RAM_ACCESS_WIDTH = PARALLEL_ACCESS_WIDTH * WEIGHT_LOG;
const MAX_RAM_ADDR = MAX_SYMBOL_COUNT/PARALLEL_ACCESS_WIDTH;

enum WeightPreScanFSM: u2 {
    IDLE        = u2:0,
    FIRST_RUN   = u2:1,
    SECOND_RUN  = u2:2,
}

struct WeightPreScanState {
    fsm: WeightPreScanFSM,
    addr: u9,
    internal_addr: u9,
}

pub proc WeightPreScan
// TODO: enable parametric expresion when they start working
//proc WeightPreScan<
//    PARALLEL_ACCESS_WIDTH: u32 = {u32:8},
//    RAM_SIZE: u32 = {MAX_SYMBOL_COUNT/PARALLEL_ACCESS_WIDTH},
//    RAM_ADDR_WIDTH: u32 = {std::clog2(RAM_SIZE)},
//    RAM_ACCESS_WIDTH: u32 = {PARALLEL_ACCESS_WIDTH * WEIGHT_LOG},
//    MAX_RAM_ADDR: u32 = {(u32:1<<RAM_ADDR_WIDTH - u32:1)},
//> {
{
    type State = WeightPreScanState;
    type FSM = WeightPreScanFSM;

    type ExternalRamAddr = uN[RAM_ADDR_WIDTH];
    type ExternalRamData = uN[RAM_ACCESS_WIDTH];

    type OutData = WeightPreScanOutput;

    type ReadReq  = ram::ReadReq<RAM_ADDR_WIDTH, u32:1>;
    type ReadResp = ram::ReadResp<RAM_ACCESS_WIDTH>;

    type InternalRamAddr = uN[RAM_ADDR_WIDTH];
    type InternalData     = WeightPreScanMetaData;
    type InternalRamData = bits[WeightPreScanMetaDataSize()];

    type InternalReadReq    = ram::ReadReq<RAM_ADDR_WIDTH, u32:1>;
    type InternalReadResp   = ram::ReadResp<{WeightPreScanMetaDataSize()}>;
    type InternalWriteReq   = ram::WriteReq<RAM_ADDR_WIDTH, {WeightPreScanMetaDataSize()}, u32:1>;
    type InternalWriteResp  = ram::WriteResp;

    start_r:    chan<bool> in;
    read_req_s: chan<ReadReq> out;
    read_rsp_r: chan<ReadResp> in;
    weight_s:   chan<OutData> out;

    internal_read_req_s:  chan<InternalReadReq> out;
    internal_read_rsp_r:  chan<InternalReadResp> in;
    internal_write_req_s: chan<InternalWriteReq> out;
    internal_write_rsp_r: chan<InternalWriteResp> in;

    internal_memory_written_s: chan<InternalRamAddr> out;
    internal_memory_written_r: chan<InternalRamAddr> in;

    config (
        start_r:    chan<bool> in,
        read_req_s: chan<ReadReq> out,
        read_rsp_r: chan<ReadResp> in,
        weight_s:   chan<OutData> out,
        internal_read_req_s:  chan<InternalReadReq> out,
        internal_read_rsp_r:  chan<InternalReadResp> in,
        internal_write_req_s: chan<InternalWriteReq> out,
        internal_write_rsp_r: chan<InternalWriteResp> in
    ) {
        let (internal_memory_written_s, internal_memory_written_r) =
            chan<InternalRamAddr, u32:1>("internal_loopback");
        (start_r, read_req_s, read_rsp_r, weight_s,
         internal_read_req_s, internal_read_rsp_r,
         internal_write_req_s, internal_write_rsp_r,
         internal_memory_written_s, internal_memory_written_r)
    }

    init {zero!<State>()}

    next(state: State) {
        let tok = join();
        trace_fmt!("State {}", state.fsm);
        let (recv_start, send_addr, write_internal, read_internal, addr) = match state.fsm {
            FSM::IDLE => (true, false, false, false, u32:0 as ExternalRamAddr),
            FSM::FIRST_RUN => (false, true, true, false, state.addr as ExternalRamAddr),
            FSM::SECOND_RUN => {
                let valid_data = state.addr < state.internal_addr || state.internal_addr as u32 == MAX_RAM_ADDR - u32:1;
                (false, valid_data, false, valid_data, state.addr as ExternalRamAddr)
            },
            _ => {
                assert!(false, "Invalid state");
                (false, false, false, false, u9:0 as ExternalRamAddr)
            }
        };
        let (tok1, start) = recv_if(tok, start_r, recv_start, false);
        if recv_start {
            trace_fmt!("Start received");
        } else {};

        let (tok2, internal_addr, internal_addr_valid) = recv_non_blocking(tok, internal_memory_written_r, state.internal_addr as InternalRamAddr);
        if internal_addr_valid {
            trace_fmt!("Received internal addr {:#x}", internal_addr);
        } else {};
        let next_state = match (state.fsm, start, send_addr, state.addr as u32 == MAX_RAM_ADDR - u32:1) {
            (FSM::IDLE, true, _, _) => State {
                fsm: FSM::FIRST_RUN,
                addr: u9:0,
                internal_addr: u9:0
            },
            (FSM::FIRST_RUN, _, false, _) => State {
                fsm: FSM::FIRST_RUN,
                addr: state.addr,
                internal_addr: internal_addr as u9
            },
            (FSM::FIRST_RUN, _, true, false) => State {
                fsm: FSM::FIRST_RUN,
                addr: state.addr + u9:1,
                internal_addr: internal_addr as u9
            },
            (FSM::FIRST_RUN, _, true, true) => State {
                fsm: FSM::SECOND_RUN,
                addr: u9:0,
                internal_addr: internal_addr as u9
            },
            (FSM::SECOND_RUN, _, false, _) => State {
                fsm: FSM::SECOND_RUN,
                addr: state.addr,
                internal_addr: internal_addr as u9
            },
            (FSM::SECOND_RUN, _, true, false) => State {
                fsm: FSM::SECOND_RUN,
                addr: state.addr + u9:1,
                internal_addr: internal_addr as u9
            },
            (FSM::SECOND_RUN, _, true, true) => State {
                fsm: FSM::IDLE,
                addr: u9:0,
                internal_addr: internal_addr as u9
            },
            _ => {
                assert!(false, "Invalid state");
                State {
                    fsm: FSM::IDLE,
                    addr: u9:0,
                    internal_addr: u9:0
                }
            }
        };

        let external_ram_req = ReadReq {
            addr: addr,
            mask: u1:1,
        };
        let tok3 = send_if(tok, read_req_s, send_addr, external_ram_req);
        if send_addr {
            trace_fmt!("Sent read request {:#x}", external_ram_req);
        } else {};
        let (tok3, ram_data) = recv_if(tok3, read_rsp_r, send_addr, zero!<ReadResp>());
        let ram_data = ram_data.data as uN[WEIGHT_LOG][PARALLEL_ACCESS_WIDTH];
        if send_addr {
            trace_fmt!("Received read response {:#x}", ram_data);
        } else {};

        let internal_ram_r_req = InternalReadReq {
            addr: addr,
            mask: u1:1,
        };
        let tok4 = send_if(tok, internal_read_req_s, read_internal, internal_ram_r_req);
        let (tok4, meta_data_flat) = recv_if(tok4, internal_read_rsp_r, read_internal, zero!<InternalReadResp>());
        let meta_data = BitsToInternalStruct(meta_data_flat.data);
        let tok34 = join(tok3, tok4);

        if read_internal {
            trace_fmt!("Reading internal memory data: {:#x}", meta_data);
        } else {};

        let prescan_output = OutData {
            weights: ram_data,
            meta_data: meta_data
        };
        let tok34 = send_if(tok34, weight_s, send_addr, prescan_output);
        if send_addr {
            trace_fmt!("Sent output {:#x}", prescan_output);
        } else {};

        let occurance_matrix = for (i, occurance_matrix) in range(u32:0, PARALLEL_ACCESS_WIDTH) {
            let row = for (j, row) in range(u32:0, MAX_WEIGHT + u32:1) {
                if (ram_data[i] == j as uN[COUNTER_WIDTH]) {
                    update(row, j, row[j] + uN[COUNTER_WIDTH]:1)
                } else { row }
            } (occurance_matrix[i]);
            update(occurance_matrix, i + u32:1, row)
        }(zero!<uN[COUNTER_WIDTH][MAX_WEIGHT + u32:1][PARALLEL_ACCESS_WIDTH + u32:1]>());

        let valid_weights = for(i, valid_weights) in range(u32:0, MAX_WEIGHT + u32:1) {
            if (occurance_matrix[PARALLEL_ACCESS_WIDTH][i] != uN[COUNTER_WIDTH]:0) {
                update(valid_weights, i, true)
            } else { valid_weights }
        }(zero!<u1[MAX_WEIGHT + u32:1]>());

        let occurance_number = for (i, occurance_number) in range(u32:0, PARALLEL_ACCESS_WIDTH) {
            let number = occurance_matrix[i][ram_data[i]];
            update(occurance_number, i, number)
        }(zero!<uN[COUNTER_WIDTH][PARALLEL_ACCESS_WIDTH]>());
        let _meta_data = WeightPreScanMetaData {
            occurance_number: occurance_number,
            valid_weights:    valid_weights,
            weights_count:    occurance_matrix[PARALLEL_ACCESS_WIDTH],
        };

        let internal_ram_w_req = InternalWriteReq {
            addr: addr,
            data: InternalStructToBits(_meta_data),
            mask: u1:1
        };
        let tok5 = send_if(tok, internal_write_req_s, write_internal, internal_ram_w_req);
        let (tok5, _) = recv_if(tok5, internal_write_rsp_r, write_internal, zero!<InternalWriteResp>());
        send_if(tok5, internal_memory_written_s, state.fsm == FSM::FIRST_RUN, addr as InternalRamAddr);
        if write_internal {
            trace_fmt!("Internal write {:#x}", _meta_data);
        } else {};

        next_state
    }
}

#[test_proc]
proc Prescan_test{
    type external_ram_addr = uN[RAM_ADDR_WIDTH];
    type external_ram_data = uN[RAM_ACCESS_WIDTH];

    type PrescanOut    = WeightPreScanOutput;

    type ReadReq  = ram::ReadReq<RAM_ADDR_WIDTH, u32:1>;
    type ReadResp = ram::ReadResp<RAM_ACCESS_WIDTH>;
    type WriteReq  = ram::WriteReq<RAM_ADDR_WIDTH, RAM_ACCESS_WIDTH, u32:1>;
    type WriteResp = ram::WriteResp<>;

    type InternalReadReq  = ram::ReadReq<RAM_ADDR_WIDTH, u32:1>;
    type InternalReadResp = ram::ReadResp<{WeightPreScanMetaDataSize()}>;
    type InternalWriteReq  = ram::WriteReq<RAM_ADDR_WIDTH, {WeightPreScanMetaDataSize()}, u32:1>;
    type InternalWriteResp = ram::WriteResp<>;

    terminator:         chan<bool> out;
    external_ram_req:   chan<WriteReq> out;
    external_ram_resp:  chan<WriteResp> in;
    start_prescan:      chan<bool> out;
    prescan_response:   chan<PrescanOut> in;
    init{()}
    config (terminator: chan<bool> out) {
        // Emulate external memory
        let (RAMExternalWriteReq_s, RAMExternalWriteReq_r) = chan<WriteReq>("Write_channel_req");
        let (RAMExternalWriteResp_s, RAMExternalWriteResp_r) = chan<WriteResp>("Write_channel_resp");
        let (RAMExternalReadReq_s, RAMExternalReadReq_r) = chan<ReadReq>("Read_channel_req");
        let (RAMExternalReadResp_s, RAMExternalReadResp_r) = chan<ReadResp>("Read_channel_resp");
        spawn ram::RamModel<RAM_ACCESS_WIDTH, RAM_SIZE, RAM_ACCESS_WIDTH>(
            RAMExternalReadReq_r, RAMExternalReadResp_s, RAMExternalWriteReq_r, RAMExternalWriteResp_s
        );

        // Emulate Internal prescan memory
        let (RAMInternalWriteReq_s, RAMInternalWriteReq_r) = chan<InternalWriteReq>("Internal_write_channel_req");
        let (RAMInternalWriteResp_s, RAMInternalWriteResp_r) = chan<InternalWriteResp>("Internal_write_channel_resp");
        let (RAMInternalReadReq_s, RAMInternalReadReq_r) = chan<InternalReadReq>("Internal_read_channel_req");
        let (RAMInternalReadResp_s, RAMInternalReadResp_r) = chan<InternalReadResp>("Internal_read_channel_resp");
        spawn ram::RamModel<{WeightPreScanMetaDataSize()}, RAM_SIZE, {WeightPreScanMetaDataSize()}>(
            RAMInternalReadReq_r, RAMInternalReadResp_s, RAMInternalWriteReq_r, RAMInternalWriteResp_s
        );

        let (PreScanStart_s, PreScanStart_r) = chan<bool>("Start_prescan");
        let (PreScanResponse_s, PreScanResponse_r) = chan<PrescanOut>("Start_prescan");
        spawn WeightPreScan(
                PreScanStart_r, RAMExternalReadReq_s,RAMExternalReadResp_r, PreScanResponse_s,
                RAMInternalReadReq_s, RAMInternalReadResp_r, RAMInternalWriteReq_s, RAMInternalWriteResp_r);
        (terminator, RAMExternalWriteReq_s, RAMExternalWriteResp_r, PreScanStart_s, PreScanResponse_r)
    }
    next(state: ()) {
        let tok = join();
        let rand_state = random::rng_new(random::rng_deterministic_seed());
        // Setup external memory with random values
        for (i, rand_state) in range(u32:0, MAX_SYMBOL_COUNT/PARALLEL_ACCESS_WIDTH) {
            let (new_rand_state, data_to_send) = for (j, (rand_state, data_to_send)) in range(u32:0, PARALLEL_ACCESS_WIDTH) {
                let (new_rand_state, data) = random::rng_next(rand_state);
                let weight = (data - (data/u32:12) * u32:12) as u4;
                let new_data_to_send = update(data_to_send as uN[WEIGHT_LOG][PARALLEL_ACCESS_WIDTH], j, weight) as external_ram_data;
                (new_rand_state, new_data_to_send)
            }((rand_state, zero!<external_ram_data>()));
            let external_w_req = WriteReq {
                addr: i as uN[RAM_ADDR_WIDTH],
                data: data_to_send,
                mask: u1:1
            };
            send(tok, external_ram_req, external_w_req);
            recv(tok, external_ram_resp);
            new_rand_state
        }(rand_state);
        send(tok, start_prescan, true);
        // First run
        for (_, rand_state) in range(u32:0, MAX_SYMBOL_COUNT/PARALLEL_ACCESS_WIDTH) {
            // Generate expected output
            let (new_rand_state, expected_data) = for (j, (rand_state, data_to_send)) in range(u32:0, PARALLEL_ACCESS_WIDTH) {
                let (new_rand_state, data) = random::rng_next(rand_state);
                let weight = (data - (data/u32:12) * u32:12) as u4;
                let new_data_to_send = update(data_to_send as uN[WEIGHT_LOG][PARALLEL_ACCESS_WIDTH], j, weight) as external_ram_data;
                (new_rand_state, new_data_to_send)
            }((rand_state, zero!<external_ram_data>()));
            let (_, prescan_resp) = recv(tok, prescan_response);
            let expected_data = PrescanOut {
                weights: expected_data as uN[WEIGHT_LOG][PARALLEL_ACCESS_WIDTH],
                meta_data: zero!<WeightPreScanMetaData>()
            };
            assert_eq(prescan_resp, expected_data);
            new_rand_state
        }(rand_state);

        // Second run
        for (_, rand_state) in range(u32:0, MAX_SYMBOL_COUNT/PARALLEL_ACCESS_WIDTH) {
            // Generate expected output
            let (new_rand_state, expected_data) = for (j, (rand_state, data_to_send)) in range(u32:0, PARALLEL_ACCESS_WIDTH) {
                let (new_rand_state, data) = random::rng_next(rand_state);
                let weight = (data - (data/u32:12) * u32:12) as u4;
                let new_data_to_send = update(data_to_send as uN[WEIGHT_LOG][PARALLEL_ACCESS_WIDTH], j, weight) as external_ram_data;
                (new_rand_state, new_data_to_send)
            }((rand_state, zero!<external_ram_data>()));
            let expected_data = expected_data as uN[WEIGHT_LOG][PARALLEL_ACCESS_WIDTH];
            let valid_weights = for (i, seen_weights) in range(u32:0, PARALLEL_ACCESS_WIDTH) {
                update(seen_weights, expected_data[i], true)
            }(zero!<u1[MAX_WEIGHT + u32:1]>());
            let occurance_number = for (i, occurance_number) in range(u32:0, PARALLEL_ACCESS_WIDTH) {
                let number = for (j, number) in range(u32:0, PARALLEL_ACCESS_WIDTH){
                    if (j < i && expected_data[j] == expected_data[i]) {
                        number + u4:1
                    } else {
                        number
                    }
                }(zero!<uN[COUNTER_WIDTH]>());
                update(occurance_number, i, number)
            }(zero!<uN[COUNTER_WIDTH][PARALLEL_ACCESS_WIDTH]>());
            let weights_count = for (i, weights_count) in range(u32:0, MAX_WEIGHT + u32:1) {
                let count = for (j, count) in range(u32:0, PARALLEL_ACCESS_WIDTH) {
                    if (expected_data[j] == i as uN[COUNTER_WIDTH]) {
                        count + uN[COUNTER_WIDTH]:1
                    } else {
                        count
                    }
                }(zero!<uN[COUNTER_WIDTH]>());
                update(weights_count, i, count)
            }(zero!<uN[COUNTER_WIDTH][MAX_WEIGHT + u32:1]>());
            let (_, prescan_resp) = recv(tok, prescan_response);
            let expected_data = PrescanOut {
                weights: expected_data,
                meta_data: WeightPreScanMetaData {
                    occurance_number: occurance_number,
                    valid_weights:    valid_weights,
                    weights_count:    weights_count,
                }
            };
            assert_eq(prescan_resp, expected_data);
            new_rand_state
        }(rand_state);

        send(tok, terminator, true);
    }
}
