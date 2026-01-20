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

// Address Generator

import std;
import xls.modules.dma.common;
import xls.modules.dma.config;

type TransferDescBundle = common::TransferDescBundle;
type MainCtrlBundle = common::MainCtrlBundle;

enum AddressGeneratorStatusEnum : u2 {
    IDLE = 0,
    WAIT = 1,
    BUSY = 2,
    DONE = 3,
}

struct AddressGeneratorState<ADDR_W: u32> {
    configuration: MainCtrlBundle<ADDR_W>,
    status: AddressGeneratorStatusEnum,
    transfer: TransferDescBundle<ADDR_W>,
    it: u32,
    rsp_counter: u32,
}

proc AddressGenerator<ADDR_W: u32, DATA_W_DIV8: u32> {
    configuration: chan<MainCtrlBundle<ADDR_W>> in;
    start_ch: chan<u1> in;
    busy_ch: chan<u1> out;
    done_ch: chan<u1> out;
    addr_gen_req: chan<TransferDescBundle<ADDR_W>> out;
    addr_gen_rsp: chan<()> in;

    config(configuration: chan<MainCtrlBundle<ADDR_W>> in, start_ch: chan<u1> in,
           busy_ch: chan<u1> out, done_ch: chan<u1> out,
           addr_gen_req: chan<TransferDescBundle<ADDR_W>> out, addr_gen_rsp: chan<()> in) {
        (configuration, start_ch, busy_ch, done_ch, addr_gen_req, addr_gen_rsp)
    }

    init {
        (AddressGeneratorState<ADDR_W> {
            configuration: common::zeroMainCtrlBundle<ADDR_W>(),
            status: AddressGeneratorStatusEnum::IDLE,
            transfer: common::zeroTransferDescBundle<ADDR_W>(),
            it: u32:0,
            rsp_counter: u32:0
        })
    }

    next(tok: token, state: AddressGeneratorState<ADDR_W>) {

        let (tok, start) = recv(tok, start_ch);
        let goto_wait = start && (state.status == AddressGeneratorStatusEnum::IDLE);

        let (tok, configuration) = recv(tok, configuration);
        let goto_busy = state.status == AddressGeneratorStatusEnum::WAIT;
        let configuration = if goto_busy {
            trace_fmt!("[AG] New configuration = {}", configuration);
            configuration
        } else {
            state.configuration
        };

        let send_transfer = state.status == AddressGeneratorStatusEnum::BUSY &&
                            (state.it != (state.configuration.line_count as u32));
        let tok = send_if(tok, addr_gen_req, send_transfer, state.transfer);

        let it = if send_transfer { state.it + u32:1 } else { state.it };

        let (tok, _, valid_addr_gen_rsp) = recv_if_non_blocking(
            tok, addr_gen_rsp, state.status == AddressGeneratorStatusEnum::BUSY, ());

        let rsp_counter =
            if valid_addr_gen_rsp { state.rsp_counter + u32:1 } else { state.rsp_counter };

        let goto_done = (state.status == AddressGeneratorStatusEnum::BUSY) &&
                        (rsp_counter == (state.configuration.line_count as u32));

        let tok = send_if(tok, done_ch, goto_done, u1:1);

        let goto_idle = state.status == AddressGeneratorStatusEnum::DONE;

        // Next state logic
        let nextState = if state.status == AddressGeneratorStatusEnum::IDLE {
            if goto_wait {
                AddressGeneratorStatusEnum::WAIT
            } else {
                AddressGeneratorStatusEnum::IDLE
            }
        } else if state.status == AddressGeneratorStatusEnum::WAIT {
            if goto_busy {
                AddressGeneratorStatusEnum::BUSY
            } else {
                AddressGeneratorStatusEnum::WAIT
            }
        } else if state.status == AddressGeneratorStatusEnum::BUSY {
            if goto_done {
                AddressGeneratorStatusEnum::DONE
            } else {
                AddressGeneratorStatusEnum::BUSY
            }
        } else if state.status == AddressGeneratorStatusEnum::DONE {
            if goto_idle {
                AddressGeneratorStatusEnum::IDLE
            } else {
                AddressGeneratorStatusEnum::DONE
            }
        } else {
            AddressGeneratorStatusEnum::IDLE
        };
        // trace_fmt!("State 		: {} Next state 	: {}", state.status, nextState);

        let nextBundle = if state.status == AddressGeneratorStatusEnum::BUSY {
            TransferDescBundle<ADDR_W> {
                address:
                state.transfer.address +
                (DATA_W_DIV8 as uN[ADDR_W]) *
                (state.configuration.line_length + state.configuration.line_stride),
                length: state.configuration.line_length
            }
        } else if state.status == AddressGeneratorStatusEnum::WAIT {
            TransferDescBundle<ADDR_W> {
                address: configuration.start_address, length: configuration.line_length
            }
        } else {
            state.transfer
        };

        let it = if state.status == AddressGeneratorStatusEnum::DONE { u32:0 } else { it };
        let rsp_counter =
            if state.status == AddressGeneratorStatusEnum::DONE { u32:0 } else { rsp_counter };

        let is_busy = (state.status != AddressGeneratorStatusEnum::IDLE) as u1;
        let tok = send(tok, busy_ch, is_busy);

        trace!(state);
        AddressGeneratorState {
            configuration, transfer: nextBundle, status: nextState, it, rsp_counter
        }
    }
}

pub fn AddressGeneratorReferenceFunction<C: u32, ADDR_W: u32, DATA_W_DIV8: u32>
    (config: MainCtrlBundle<ADDR_W>) -> TransferDescBundle<ADDR_W>[C] {

    let a = for (i, a): (u32, TransferDescBundle[C]) in range(u32:0, C) {
        if i == u32:0 {
            update(
                a, i,
                TransferDescBundle<ADDR_W> {
                    address: config.start_address, length: config.line_length
                })
        } else {
            update(
                a, i,
                TransferDescBundle<ADDR_W> {
                    address:
                    (a[i - u32:1]).address +
                    DATA_W_DIV8 * (config.line_length + config.line_stride),
                    length: config.line_length
                })
        }
    }(TransferDescBundle<ADDR_W>[C]:[common::zeroTransferDescBundle<ADDR_W>(), ...]);
    a
}

#[test]
fn TestAddressGeneratorReferenceFunction() {
    let ADDR_W = u32:32;
    let dataWidthDiv8 = u32:4;
    let testConfig = MainCtrlBundle<ADDR_W> {
        start_address: uN[ADDR_W]:1000,
        line_count: uN[ADDR_W]:4,
        line_length: uN[ADDR_W]:3,
        line_stride: uN[ADDR_W]:2
    };
    // TODO: Is using parametric from a struct field a good practice?
    let C = testConfig.line_count;
    let a = AddressGeneratorReferenceFunction<C, ADDR_W, dataWidthDiv8>(testConfig);
    assert_eq((a[0]).address, u32:1000);
    assert_eq((a[1]).address, u32:1020);
    assert_eq((a[2]).address, u32:1040);
    assert_eq((a[3]).address, u32:1060);

    assert_eq((a[0]).length, u32:3);
    assert_eq((a[1]).length, u32:3);
    assert_eq((a[2]).length, u32:3);
    assert_eq((a[3]).length, u32:3);
}

const TEST_DATA_W_DIV8 = u32:4;
const TEST_ADDR_W = u32:32;

#[test_proc]
proc TestAddressGenerator {
    configuration: chan<MainCtrlBundle<TEST_ADDR_W>> out;
    start_ch: chan<u1> out;
    busy_ch: chan<u1> in;
    done_ch: chan<u1> in;
    addr_gen_req: chan<TransferDescBundle<TEST_ADDR_W>> in;
    addr_gen_rsp: chan<()> out;
    terminator: chan<bool> out;

    config(terminator: chan<bool> out) {
        let (configuration_s, configuration_r) = chan<MainCtrlBundle<TEST_ADDR_W>>;
        let (start_ch_s, start_ch_r) = chan<u1>;
        let (busy_ch_s, busy_ch_r) = chan<u1>;
        let (done_ch_s, done_ch_r) = chan<u1>;
        let (addr_gen_req_s, addr_gen_req_r) = chan<TransferDescBundle<TEST_ADDR_W>>;
        let (addr_gen_rsp_s, addr_gen_rsp_r) = chan<()>;
        spawn AddressGenerator<TEST_ADDR_W, TEST_DATA_W_DIV8>(
            configuration_r, start_ch_r, busy_ch_s, done_ch_s, addr_gen_req_s, addr_gen_rsp_r);
        (
            configuration_s, start_ch_s, busy_ch_r, done_ch_r, addr_gen_req_r, addr_gen_rsp_s,
            terminator,
        )
    }

    init { (u32:0) }

    next(tok: token, state: u32) {
        let testConfig = MainCtrlBundle<TEST_ADDR_W> {
            start_address: uN[TEST_ADDR_W]:1000,
            line_count: uN[TEST_ADDR_W]:5,
            line_length: uN[TEST_ADDR_W]:3,
            line_stride: uN[TEST_ADDR_W]:0
        };

        let tok = send(tok, start_ch, u1:1);
        let tok = send(tok, configuration, testConfig);

        let (tok, r_data, r_data_valid) =
            recv_non_blocking(tok, addr_gen_req, common::zeroTransferDescBundle<TEST_ADDR_W>());

        let state = if r_data_valid {
            trace_fmt!("r_data = {}", r_data);
            let tok = send(tok, addr_gen_rsp, ());
            state + u32:1
        } else {
            state
        };

        let (tok, done, done_valid) = recv_non_blocking(tok, done_ch, u1:1);

        let do_terminate = done && done_valid;
        if do_terminate { assert_eq(state, testConfig.line_count); } else {  };
        let tok = send_if(tok, terminator, do_terminate, do_terminate);

        state
    }
}

// Verilog example
proc address_generator {
    config(configuration: chan<MainCtrlBundle<config::TOP_ADDR_W>> in, start_ch: chan<u1> in,
           busy_ch: chan<u1> out, done_ch: chan<u1> out,
           addr_gen_req: chan<TransferDescBundle<config::TOP_ADDR_W>> out, addr_gen_rsp: chan<()> in) {
        spawn AddressGenerator<config::TOP_ADDR_W, config::TOP_DATA_W_DIV8>(
            configuration, start_ch, busy_ch, done_ch, addr_gen_req, addr_gen_rsp);
        ()
    }

    init { () }

    next(tok: token, state: ()) {  }
}
