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

// This file contains the implementation of Huffman tree decoder.

import std;
import xls.dslx.stdlib.acm_random as random;

import xls.examples.ram;
import xls.modules.zstd.common as common;
import xls.modules.zstd.huffman_common as hcommon;

const MAX_WEIGHT = hcommon::MAX_WEIGHT;
const WEIGHT_LOG = hcommon::WEIGHT_LOG;
const MAX_SYMBOL_COUNT = hcommon::MAX_SYMBOL_COUNT;

const PARALLEL_ACCESS_WIDTH = hcommon::PARALLEL_ACCESS_WIDTH;
const COUNTER_WIDTH = hcommon::COUNTER_WIDTH;

const RECV_COUNT = MAX_SYMBOL_COUNT/PARALLEL_ACCESS_WIDTH;
const RECV_COUNT_W = std::clog2(RECV_COUNT + u32:1);
const MAX_RECV   = RECV_COUNT as uN[RECV_COUNT_W];

type WeightPreScanMetaData = hcommon::WeightPreScanMetaData;
type WeightPreScanOutput = hcommon::WeightPreScanOutput;
type CodeBuilderToPreDecoderOutput = hcommon::CodeBuilderToPreDecoderOutput;
type CodeBuilderToDecoderOutput = hcommon::CodeBuilderToDecoderOutput;

enum WeightCodeBuilderFSM: u2 {
    IDLE               = u2:0,
    GATHER_WEIGHTS_RUN = u2:1,
    COMPUTE_MAX_LENGTH = u2:2,
    GENERATE_CODES_RUN = u2:3,
}

struct WeightCodeBuilderState {
    fsm:                    WeightCodeBuilderFSM,
    recv_counter:           uN[RECV_COUNT_W],
    loopback_counter:       uN[RECV_COUNT_W],
    sum_of_weights_powers:  uN[MAX_WEIGHT + u32:2],
    huffman_codes:          uN[MAX_WEIGHT][MAX_WEIGHT + u32:1],
    seen_weights:           u1[MAX_WEIGHT + u32:1],
    max_number_of_bits:     uN[WEIGHT_LOG],
}

pub proc WeightCodeBuilder
// TODO: enable parametric expresion when they start working
//proc WeightCodeBuilder<
//    PARALLEL_ACCESS_WIDTH: u32 = {u32:8},
//> {
{
    type State = WeightCodeBuilderState;
    type FSM = WeightCodeBuilderFSM;
    type PreScanData = WeightPreScanOutput;
    type DecoderOutput = CodeBuilderToDecoderOutput;
    type PreDecoderOutput = CodeBuilderToPreDecoderOutput;

    start_r:    chan<bool> in;
    weight_r:   chan<PreScanData> in;
    codes_s:    chan<DecoderOutput> out;
    lookahead_config_s: chan<PreDecoderOutput> out;

    weights_pow_sum_loopback_s: chan<uN[MAX_WEIGHT + u32:2]> out;
    weights_pow_sum_loopback_r: chan<uN[MAX_WEIGHT + u32:2]> in;

    config (
        start_r:    chan<bool> in,
        weight_r:   chan<PreScanData> in,
        codes_s:    chan<DecoderOutput> out,
        lookahead_config_s: chan<PreDecoderOutput> out,
        weights_pow_sum_loopback_s: chan<uN[MAX_WEIGHT + u32:2]> out,
        weights_pow_sum_loopback_r: chan<uN[MAX_WEIGHT + u32:2]> in,
    ) {
        (start_r, weight_r, codes_s, lookahead_config_s, weights_pow_sum_loopback_s, weights_pow_sum_loopback_r)
    }

    init {zero!<State>()}

    next(state: State) {
        let tok = join();

        let (recv_start, recv_prescan) = match state.fsm {
            FSM::IDLE => (true, false),
            FSM::GATHER_WEIGHTS_RUN => (false, true),
            FSM::COMPUTE_MAX_LENGTH => (false, false),
            FSM::GENERATE_CODES_RUN => (false, true),
            _ => {
                assert!(false, "Invalid state");
                (false, false)
            }
        };
        let (_, start, start_valid) = recv_if_non_blocking(tok, start_r, recv_start, false);
        let (_, prescan_data, prescan_data_valid) = recv_if_non_blocking(tok, weight_r, recv_prescan, zero!<PreScanData>());

        if start_valid {
            trace_fmt!("Received start {:#x}", start);
        } else {};

        if prescan_data_valid {
            trace_fmt!("Received prescan {:#x}", prescan_data);
        } else {};

        let (advance_state, send_lookahead, send_codes) = match state.fsm {
            FSM::IDLE => (start && start_valid, false, false),
            FSM::GATHER_WEIGHTS_RUN => (state.recv_counter == MAX_RECV, false, false),
            FSM::COMPUTE_MAX_LENGTH => (state.loopback_counter == MAX_RECV, false, false),
            FSM::GENERATE_CODES_RUN => {
                let advance_state = state.recv_counter == (MAX_RECV * uN[RECV_COUNT_W]:2);
                (advance_state, advance_state, prescan_data_valid)
            },
            _ => {
                assert!(false, "Invalid state");
                (false, false, false)
            }
        };

        let next_fsm_state = match(state.fsm, advance_state) {
            (FSM::IDLE, true) => {
                trace_fmt!("IDLE -> GATHER_WEIGHTS_RUN");
                FSM::GATHER_WEIGHTS_RUN
            },
            (FSM::GATHER_WEIGHTS_RUN, true) => {
                trace_fmt!("GATHER_WEIGHTS_RUN -> COMPUTE_MAX_LENGTH");
                FSM::COMPUTE_MAX_LENGTH
            },
            (FSM::COMPUTE_MAX_LENGTH, true) => {
                trace_fmt!("COMPUTE_MAX_LENGTH -> GENERATE_CODES_RUN");
                FSM::GENERATE_CODES_RUN
            },
            (FSM::GENERATE_CODES_RUN, true) => {
                trace_fmt!("GENERATE_CODES_RUN -> IDLE");
                FSM::IDLE
            },
            (_, false) => state.fsm,
            _ => {
                assert!(false, "Invalid state");
                FSM::IDLE
            }
        };

        let meta_data = prescan_data.meta_data;

        // update seen weights
        let seen_weights = for (i, weights) in range(u32:0, MAX_WEIGHT + u32:1) {
            update(weights, i, weights[i] | meta_data.valid_weights[i])
        }(state.seen_weights);

        // compute sum of weights powers and send it to loopback
        let do_send_loopback = (state.fsm == FSM::GATHER_WEIGHTS_RUN) && prescan_data_valid;

        let sum_of_weights_powers = if do_send_loopback {
            for (i, acc) in range(u32:0, PARALLEL_ACCESS_WIDTH) {
                if (prescan_data.weights[i] != uN[WEIGHT_LOG]:0) {
                    acc + (uN[MAX_WEIGHT + u32:2]:1 << prescan_data.weights[i] as uN[MAX_WEIGHT + u32:2])
                } else {
                    acc
                }
            }(uN[MAX_WEIGHT + u32:2]:0)
        } else {
            uN[MAX_WEIGHT + u32:2]:0
        };

        send_if(tok, weights_pow_sum_loopback_s, do_send_loopback, sum_of_weights_powers);

        // receive sum of weights powers from loopback
        let (_, sum_of_weights_powers, sum_of_weights_powers_valid) = recv_non_blocking(
            tok, weights_pow_sum_loopback_r, uN[MAX_WEIGHT + u32:2]:0
        );
        let sum_of_weights_powers = state.sum_of_weights_powers + sum_of_weights_powers;
        let loopback_counter = if sum_of_weights_powers_valid {
            trace_fmt!("Sum of weights powers: {}", sum_of_weights_powers);
            state.loopback_counter + uN[RECV_COUNT_W]:1
        } else {
            state.loopback_counter
        };

        // compute max number of bits
        let max_number_of_bits = encode(sum_of_weights_powers >> u32:1) as uN[WEIGHT_LOG];

        // intial value for huffman codes is 0 for weight 1 and 1 for the rest
        // then the value is computed based on number of occurances of given weight
        let huffman_codes = match(state.fsm, advance_state) {
            (FSM::IDLE, _) => {
                let huffman_codes = for (i, codes) in range(u32:0, MAX_WEIGHT + u32:1) {
                    update(codes, i, uN[MAX_WEIGHT]:1)
                }(zero!<uN[MAX_WEIGHT][MAX_WEIGHT + u32:1]>());
                update(huffman_codes, u32:1, uN[MAX_WEIGHT]:0)
            },
            (FSM::GENERATE_CODES_RUN, _) => {
                let weights_count = meta_data.weights_count;
                for(i, codes) in range(u32:0, MAX_WEIGHT + u32:1) {
                    update(codes, i, codes[i] + (weights_count[i] as uN[MAX_WEIGHT]))
                }(state.huffman_codes)
            },
            _ => state.huffman_codes,
        };

        let next_state = match(state.fsm,) {
            (FSM::IDLE) => {
                State {
                    fsm: next_fsm_state,
                    huffman_codes: huffman_codes,
                    ..zero!<State>()
                }
            },
            (FSM::GATHER_WEIGHTS_RUN) => {
                let recv_counter = if prescan_data_valid {
                    state.recv_counter + uN[RECV_COUNT_W]:1
                } else {
                    state.recv_counter
                };
                State {
                    fsm: next_fsm_state,
                    loopback_counter: loopback_counter,
                    sum_of_weights_powers: sum_of_weights_powers,
                    recv_counter: recv_counter,
                    ..state
                }
            },
            (FSM::COMPUTE_MAX_LENGTH) => {
                State {
                    fsm: next_fsm_state,
                    loopback_counter: loopback_counter,
                    sum_of_weights_powers: sum_of_weights_powers,
                    max_number_of_bits: max_number_of_bits,
                    ..state
                }
            },
            (FSM::GENERATE_CODES_RUN) => {
                let recv_counter = if prescan_data_valid {
                    state.recv_counter + uN[RECV_COUNT_W]:1
                } else {
                    state.recv_counter
                };
                State {
                    fsm: next_fsm_state,
                    recv_counter: recv_counter,
                    huffman_codes: huffman_codes,
                    seen_weights: seen_weights,
                    ..state
                }
            },
            _ => {
                assert!(false, "Invalid state");
                zero!<State>()
            }
        };

        let lookahead_packet = PreDecoderOutput {
            max_code_length: state.max_number_of_bits,
            valid_weights: seen_weights,
        };
        send_if(tok, lookahead_config_s, send_lookahead, lookahead_packet);

        // set symbol valid if weight is nonzero
        let symbols_valid = for (i, symbol_valid) in range(u32:0, PARALLEL_ACCESS_WIDTH) {
            update(symbol_valid, i, prescan_data.weights[i] != uN[WEIGHT_LOG]:0)
        }(zero!<u1[PARALLEL_ACCESS_WIDTH]>());

        // set symbol length as max_length - weight + 1
        let codes_length = for (i, code_length) in range(u32:0, PARALLEL_ACCESS_WIDTH) {
            update(code_length, i, state.max_number_of_bits - prescan_data.weights[i] + uN[WEIGHT_LOG]:1)
        }(zero!<uN[WEIGHT_LOG][PARALLEL_ACCESS_WIDTH]>());

        // set codes using weight, occurance number and Huffman codes per weight from previous iteration
        let codes = for (i, codes) in range(u32:0, PARALLEL_ACCESS_WIDTH) {
            let length = state.max_number_of_bits - prescan_data.weights[i] + uN[WEIGHT_LOG]:1;
            let base_code = for(j, base_code) in range(u32:0, MAX_WEIGHT + u32:1) {
                if (prescan_data.weights[i] == j as uN[WEIGHT_LOG]) {
                    state.huffman_codes[j]
                } else {
                    base_code
                }
            }(uN[MAX_WEIGHT]:0);
            let code = base_code + (meta_data.occurance_number[i] as uN[MAX_WEIGHT]);
            let code = rev(code) >> (MAX_WEIGHT - length as u32);
            update(codes, i, code)
        }(zero!<uN[MAX_WEIGHT][PARALLEL_ACCESS_WIDTH]>());

        if send_codes {
            trace_fmt!("{}\n{}\n{:#b}\n{:#b}", symbols_valid, codes_length, codes, state.huffman_codes);
        } else {};

        let code_packet = DecoderOutput {
            symbol_valid: symbols_valid,
            code_length: codes_length,
            code: codes
        };
        send_if(tok, codes_s, send_codes, code_packet);

        next_state
    }
}

//#[test_proc]
//proc WeightCodeBuilderSimpleTest{
//    type PrescanOut       = WeightPreScanOutput;
//    type DecoderOutput    = CodeBuilderToDecoderOutput;
//    type PreDecoderOutput = CodeBuilderToPreDecoderOutput;
//
//    terminator:         chan<bool> out;
////    external_ram_req:   chan<WriteReq> out;
////    external_ram_resp:  chan<WriteResp> in;
////    start_prescan:      chan<bool> out;
////    prescan_response:   chan<PrescanOut> in;
//    init{()}
////    config (terminator: chan<bool> out) {
////        // Emulate external memory
////        let (RAMExternalWriteReq_s, RAMExternalWriteReq_r) = chan<WriteReq>("Write_channel_req");
////        let (RAMExternalWriteResp_s, RAMExternalWriteResp_r) = chan<WriteResp>("Write_channel_resp");
////        let (RAMExternalReadReq_s, RAMExternalReadReq_r) = chan<ReadReq>("Read_channel_req");
////        let (RAMExternalReadResp_s, RAMExternalReadResp_r) = chan<ReadResp>("Read_channel_resp");
////        spawn ram::RamModel<RAM_ACCESS_WIDTH, RAM_SIZE, RAM_ACCESS_WIDTH>(
////            RAMExternalReadReq_r, RAMExternalReadResp_s, RAMExternalWriteReq_r, RAMExternalWriteResp_s
////        );
////
////        // Emulate Internal prescan memory
////        let (RAMInternalWriteReq_s, RAMInternalWriteReq_r) = chan<InternalWriteReq>("Internal_write_channel_req");
////        let (RAMInternalWriteResp_s, RAMInternalWriteResp_r) = chan<InternalWriteResp>("Internal_write_channel_resp");
////        let (RAMInternalReadReq_s, RAMInternalReadReq_r) = chan<InternalReadReq>("Internal_read_channel_req");
////        let (RAMInternalReadResp_s, RAMInternalReadResp_r) = chan<InternalReadResp>("Internal_read_channel_resp");
////        spawn ram::RamModel<{WeightPreScanMetaDataSize()}, RAM_SIZE, {WeightPreScanMetaDataSize()}>(
////            RAMInternalReadReq_r, RAMInternalReadResp_s, RAMInternalWriteReq_r, RAMInternalWriteResp_s
////        );
////
////        let (PreScanStart_s, PreScanStart_r) = chan<bool>("Start_prescan");
////        let (PreScanResponse_s, PreScanResponse_r) = chan<PrescanOut>("Start_prescan");
////        spawn WeightPreScan(
////                PreScanStart_r, RAMExternalReadReq_s,RAMExternalReadResp_r, PreScanResponse_s,
////                RAMInternalReadReq_s, RAMInternalReadResp_r, RAMInternalWriteReq_s, RAMInternalWriteResp_r);
////        (terminator, RAMExternalWriteReq_s, RAMExternalWriteResp_r, PreScanStart_s, PreScanResponse_r)
////    }
////    next(state: ()) {
////        let tok = join();
////        let rand_state = random::rng_new(random::rng_deterministic_seed());
////        // Setup external memory with random values
////        for (i, rand_state) in range(u32:0, MAX_SYMBOL_COUNT/PARALLEL_ACCESS_WIDTH) {
////            let (new_rand_state, data_to_send) = for (j, (rand_state, data_to_send)) in range(u32:0, PARALLEL_ACCESS_WIDTH) {
////                let (new_rand_state, data) = random::rng_next(rand_state);
////                let weight = (data - (data/u32:12) * u32:12) as u4;
////                let new_data_to_send = update(data_to_send as uN[WEIGHT_LOG][PARALLEL_ACCESS_WIDTH], j, weight) as external_ram_data;
////                (new_rand_state, new_data_to_send)
////            }((rand_state, zero!<external_ram_data>()));
////            let external_w_req = WriteReq {
////                addr: i as u5,
////                data: data_to_send,
////                mask: u1:1
////            };
////            send(tok, external_ram_req, external_w_req);
////            recv(tok, external_ram_resp);
////            new_rand_state
////        }(rand_state);
////        send(tok, start_prescan, true);
////        // First run
////        for (_, rand_state) in range(u32:0, MAX_SYMBOL_COUNT/PARALLEL_ACCESS_WIDTH) {
////            // Generate expected output
////            let (new_rand_state, expected_data) = for (j, (rand_state, data_to_send)) in range(u32:0, PARALLEL_ACCESS_WIDTH) {
////                let (new_rand_state, data) = random::rng_next(rand_state);
////                let weight = (data - (data/u32:12) * u32:12) as u4;
////                let new_data_to_send = update(data_to_send as uN[WEIGHT_LOG][PARALLEL_ACCESS_WIDTH], j, weight) as external_ram_data;
////                (new_rand_state, new_data_to_send)
////            }((rand_state, zero!<external_ram_data>()));
////            let (_, prescan_resp) = recv(tok, prescan_response);
////            let expected_data = PrescanOut {
////                weights: expected_data as uN[WEIGHT_LOG][PARALLEL_ACCESS_WIDTH],
////                meta_data: zero!<WeightPreScanMetaData>()
////            };
////            assert_eq(prescan_resp, expected_data);
////            new_rand_state
////        }(rand_state);
////
////        // Second run
////        for (_, rand_state) in range(u32:0, MAX_SYMBOL_COUNT/PARALLEL_ACCESS_WIDTH) {
////            // Generate expected output
////            let (new_rand_state, expected_data) = for (j, (rand_state, data_to_send)) in range(u32:0, PARALLEL_ACCESS_WIDTH) {
////                let (new_rand_state, data) = random::rng_next(rand_state);
////                let weight = (data - (data/u32:12) * u32:12) as u4;
////                let new_data_to_send = update(data_to_send as uN[WEIGHT_LOG][PARALLEL_ACCESS_WIDTH], j, weight) as external_ram_data;
////                (new_rand_state, new_data_to_send)
////            }((rand_state, zero!<external_ram_data>()));
////            let expected_data = expected_data as uN[WEIGHT_LOG][PARALLEL_ACCESS_WIDTH];
////            let valid_weights = for (i, seen_weights) in range(u32:0, PARALLEL_ACCESS_WIDTH) {
////                update(seen_weights, expected_data[i], true)
////            }(zero!<u1[MAX_WEIGHT + u32:1]>());
////            let occurance_number = for (i, occurance_number) in range(u32:0, PARALLEL_ACCESS_WIDTH) {
////                let number = for (j, number) in range(u32:0, PARALLEL_ACCESS_WIDTH){
////                    if (j < i && expected_data[j] == expected_data[i]) {
////                        number + u4:1
////                    } else {
////                        number
////                    }
////                }(zero!<uN[COUNTER_WIDTH]>());
////                update(occurance_number, i, number)
////            }(zero!<uN[COUNTER_WIDTH][PARALLEL_ACCESS_WIDTH]>());
////            let weights_count = for (i, weights_count) in range(u32:0, MAX_WEIGHT + u32:1) {
////                let count = for (j, count) in range(u32:0, PARALLEL_ACCESS_WIDTH) {
////                    if (expected_data[j] == i as uN[COUNTER_WIDTH]) {
////                        count + uN[COUNTER_WIDTH]:1
////                    } else {
////                        count
////                    }
////                }(zero!<uN[COUNTER_WIDTH]>());
////                update(weights_count, i, count)
////            }(zero!<uN[COUNTER_WIDTH][MAX_WEIGHT + u32:1]>());
////            let (_, prescan_resp) = recv(tok, prescan_response);
////            let expected_data = PrescanOut {
////                weights: expected_data,
////                meta_data: WeightPreScanMetaData {
////                    occurance_number: occurance_number,
////                    valid_weights:    valid_weights,
////                    weights_count:    weights_count,
////                }
////            };
////            assert_eq(prescan_resp, expected_data);
////            new_rand_state
////        }(rand_state);
////
////        send(tok, terminator, true);
////    }
//}
