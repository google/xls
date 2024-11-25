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

// this file contains implementation of parallel RAMs with aligned access
// write requests' address should be a multiple of data width
// read requests` address dont have to be a multiple of data width


import std;
import xls.modules.zstd.common as common;
import xls.examples.ram;

// Configurable RAM parameters, RAM_NUM has to be a power of 2
pub const RAM_NUM = u32:8;
pub const RAM_NUM_W = std::clog2(RAM_NUM);

pub struct AlignedParallelRamReadReq<ADDR_W: u32> {
    addr: uN[ADDR_W],
}

pub struct AlignedParallelRamReadResp<DATA_W: u32> {
    data: uN[DATA_W],
}

pub struct AlignedParallelRamWriteReq<ADDR_W: u32, DATA_W: u32> {
    addr: uN[ADDR_W],
    data: uN[DATA_W],
}

pub struct AlignedParallelRamWriteResp {}

enum AlignedParallelRamReadRespHandlerFSM : u1 {
    IDLE = 0,
    READ_RESP = 1,
}

struct AlignedParallelRamReadRespHandlerState<RAM_DATA_W: u32> {
    fsm: AlignedParallelRamReadRespHandlerFSM,
    ram_offset: uN[RAM_NUM_W],
    resp_recv: bool[RAM_NUM],
    resp_data: uN[RAM_DATA_W][RAM_NUM],
}

struct AlignedParallelRamReadRespHandlerCtrl {
    ram_offset: uN[RAM_NUM_W],
}

proc AlignedParallelRamReadRespHandler<
    DATA_W: u32,
    RAM_DATA_W: u32 = {DATA_W / RAM_NUM},
> {
    type ReadResp = AlignedParallelRamReadResp<DATA_W>;
    type RamReadResp = ram::ReadResp<RAM_DATA_W>;

    type FSM = AlignedParallelRamReadRespHandlerFSM;
    type Ctrl = AlignedParallelRamReadRespHandlerCtrl;
    type State = AlignedParallelRamReadRespHandlerState<RAM_DATA_W>;

    ctrl_r: chan<Ctrl> in;

    read_resp_s: chan<ReadResp> out;

    ram_read_resp_0_r: chan<RamReadResp> in;
    ram_read_resp_1_r: chan<RamReadResp> in;
    ram_read_resp_2_r: chan<RamReadResp> in;
    ram_read_resp_3_r: chan<RamReadResp> in;
    ram_read_resp_4_r: chan<RamReadResp> in;
    ram_read_resp_5_r: chan<RamReadResp> in;
    ram_read_resp_6_r: chan<RamReadResp> in;
    ram_read_resp_7_r: chan<RamReadResp> in;

    config (
        ctrl_r: chan<Ctrl> in,
        read_resp_s: chan<ReadResp> out,
        ram_read_resp_0_r: chan<RamReadResp> in,
        ram_read_resp_1_r: chan<RamReadResp> in,
        ram_read_resp_2_r: chan<RamReadResp> in,
        ram_read_resp_3_r: chan<RamReadResp> in,
        ram_read_resp_4_r: chan<RamReadResp> in,
        ram_read_resp_5_r: chan<RamReadResp> in,
        ram_read_resp_6_r: chan<RamReadResp> in,
        ram_read_resp_7_r: chan<RamReadResp> in,
    ) {
        (
            ctrl_r,
            read_resp_s,
            ram_read_resp_0_r,
            ram_read_resp_1_r,
            ram_read_resp_2_r,
            ram_read_resp_3_r,
            ram_read_resp_4_r,
            ram_read_resp_5_r,
            ram_read_resp_6_r,
            ram_read_resp_7_r,
        )
    }

    init { zero!<State>() }

    next (state: State) {
        // receive ctrl
        let (_, ctrl, ctrl_valid) = recv_if_non_blocking(join(), ctrl_r, state.fsm == FSM::IDLE, zero!<Ctrl>());

        let state = if ctrl_valid {
            State {
                fsm: FSM::READ_RESP,
                ram_offset: ctrl.ram_offset,
                ..state
            }
        } else {
            state
        };

        // receive response from each RAM
        let (_, ram_read_resp_0, ram_read_resp_0_valid) = recv_if_non_blocking(
            join(), ram_read_resp_0_r, !state.resp_recv[u32:0] && state.fsm == FSM::READ_RESP, zero!<RamReadResp>()
        );
        let (_, ram_read_resp_1, ram_read_resp_1_valid) = recv_if_non_blocking(
            join(), ram_read_resp_1_r, !state.resp_recv[u32:1] && state.fsm == FSM::READ_RESP, zero!<RamReadResp>()
        );
        let (_, ram_read_resp_2, ram_read_resp_2_valid) = recv_if_non_blocking(
            join(), ram_read_resp_2_r, !state.resp_recv[u32:2] && state.fsm == FSM::READ_RESP, zero!<RamReadResp>()
        );
        let (_, ram_read_resp_3, ram_read_resp_3_valid) = recv_if_non_blocking(
            join(), ram_read_resp_3_r, !state.resp_recv[u32:3] && state.fsm == FSM::READ_RESP, zero!<RamReadResp>()
        );
        let (_, ram_read_resp_4, ram_read_resp_4_valid) = recv_if_non_blocking(
            join(), ram_read_resp_4_r, !state.resp_recv[u32:4] && state.fsm == FSM::READ_RESP, zero!<RamReadResp>()
        );
        let (_, ram_read_resp_5, ram_read_resp_5_valid) = recv_if_non_blocking(
            join(), ram_read_resp_5_r, !state.resp_recv[u32:5] && state.fsm == FSM::READ_RESP, zero!<RamReadResp>()
        );
        let (_, ram_read_resp_6, ram_read_resp_6_valid) = recv_if_non_blocking(
            join(), ram_read_resp_6_r, !state.resp_recv[u32:6] && state.fsm == FSM::READ_RESP, zero!<RamReadResp>()
        );
        let (_, ram_read_resp_7, ram_read_resp_7_valid) = recv_if_non_blocking(
            join(), ram_read_resp_7_r, !state.resp_recv[u32:7] && state.fsm == FSM::READ_RESP, zero!<RamReadResp>()
        );

        let ram_read_resp_valid = [
            ram_read_resp_0_valid,
            ram_read_resp_1_valid,
            ram_read_resp_2_valid,
            ram_read_resp_3_valid,
            ram_read_resp_4_valid,
            ram_read_resp_5_valid,
            ram_read_resp_6_valid,
            ram_read_resp_7_valid,
        ];

        let ram_read_resp = [
            ram_read_resp_0,
            ram_read_resp_1,
            ram_read_resp_2,
            ram_read_resp_3,
            ram_read_resp_4,
            ram_read_resp_5,
            ram_read_resp_6,
            ram_read_resp_7,
        ];

        let state = for (i, state) in range(u32:0, RAM_NUM) {
            if ram_read_resp_valid[i] {
                State {
                    resp_recv: update(state.resp_recv, i, true),
                    resp_data: update(state.resp_data, i, ram_read_resp[i].data),
                    ..state
                }
            } else {
                state
            }
        }(state);

        // check if all data is received
        let all_received = for (i, all_received) in range(u32:0, RAM_NUM) {
            all_received & state.resp_recv[i]
        }(true);

        // concatenate data
        let concat_data = (
            state.resp_data[state.ram_offset + uN[RAM_NUM_W]:7] ++
            state.resp_data[state.ram_offset + uN[RAM_NUM_W]:6] ++
            state.resp_data[state.ram_offset + uN[RAM_NUM_W]:5] ++
            state.resp_data[state.ram_offset + uN[RAM_NUM_W]:4] ++
            state.resp_data[state.ram_offset + uN[RAM_NUM_W]:3] ++
            state.resp_data[state.ram_offset + uN[RAM_NUM_W]:2] ++
            state.resp_data[state.ram_offset + uN[RAM_NUM_W]:1] ++
            state.resp_data[state.ram_offset + uN[RAM_NUM_W]:0]
        );

        // send response
        send_if(join(), read_resp_s, all_received, ReadResp {
            data: concat_data
        });

        // reset state
        let state = if all_received {
            zero!<State>()
        } else {
            state
        };

        state
    }
}

struct AlignedParallelRamWriteRespHandlerState {
    resp_recv: bool[RAM_NUM],
}

proc AlignedParallelRamWriteRespHandler {
    type WriteResp = AlignedParallelRamWriteResp;
    type RamWriteResp = ram::WriteResp;

    type State = AlignedParallelRamWriteRespHandlerState;

    write_resp_s: chan<WriteResp> out;

    ram_write_resp_0_r: chan<RamWriteResp> in;
    ram_write_resp_1_r: chan<RamWriteResp> in;
    ram_write_resp_2_r: chan<RamWriteResp> in;
    ram_write_resp_3_r: chan<RamWriteResp> in;
    ram_write_resp_4_r: chan<RamWriteResp> in;
    ram_write_resp_5_r: chan<RamWriteResp> in;
    ram_write_resp_6_r: chan<RamWriteResp> in;
    ram_write_resp_7_r: chan<RamWriteResp> in;

    config (
        write_resp_s: chan<WriteResp> out,
        ram_write_resp_0_r: chan<RamWriteResp> in,
        ram_write_resp_1_r: chan<RamWriteResp> in,
        ram_write_resp_2_r: chan<RamWriteResp> in,
        ram_write_resp_3_r: chan<RamWriteResp> in,
        ram_write_resp_4_r: chan<RamWriteResp> in,
        ram_write_resp_5_r: chan<RamWriteResp> in,
        ram_write_resp_6_r: chan<RamWriteResp> in,
        ram_write_resp_7_r: chan<RamWriteResp> in,
    ) {
        (
            write_resp_s,
            ram_write_resp_0_r,
            ram_write_resp_1_r,
            ram_write_resp_2_r,
            ram_write_resp_3_r,
            ram_write_resp_4_r,
            ram_write_resp_5_r,
            ram_write_resp_6_r,
            ram_write_resp_7_r,
        )
    }

    init { zero!<State>() }

    next (state: State) {
        // receive response from each RAM
        let (_, _, ram_read_resp_0_valid) = recv_if_non_blocking(
            join(), ram_write_resp_0_r, !state.resp_recv[u32:0], zero!<RamWriteResp>()
        );
        let (_, _, ram_read_resp_1_valid) = recv_if_non_blocking(
            join(), ram_write_resp_1_r, !state.resp_recv[u32:1], zero!<RamWriteResp>()
        );
        let (_, _, ram_read_resp_2_valid) = recv_if_non_blocking(
            join(), ram_write_resp_2_r, !state.resp_recv[u32:2], zero!<RamWriteResp>()
        );
        let (_, _, ram_read_resp_3_valid) = recv_if_non_blocking(
            join(), ram_write_resp_3_r, !state.resp_recv[u32:3], zero!<RamWriteResp>()
        );
        let (_, _, ram_read_resp_4_valid) = recv_if_non_blocking(
            join(), ram_write_resp_4_r, !state.resp_recv[u32:4], zero!<RamWriteResp>()
        );
        let (_, _, ram_read_resp_5_valid) = recv_if_non_blocking(
            join(), ram_write_resp_5_r, !state.resp_recv[u32:5], zero!<RamWriteResp>()
        );
        let (_, _, ram_read_resp_6_valid) = recv_if_non_blocking(
            join(), ram_write_resp_6_r, !state.resp_recv[u32:6], zero!<RamWriteResp>()
        );
        let (_, _, ram_read_resp_7_valid) = recv_if_non_blocking(
            join(), ram_write_resp_7_r, !state.resp_recv[u32:7], zero!<RamWriteResp>()
        );

        let ram_read_resp_valid = [
            ram_read_resp_0_valid,
            ram_read_resp_1_valid,
            ram_read_resp_2_valid,
            ram_read_resp_3_valid,
            ram_read_resp_4_valid,
            ram_read_resp_5_valid,
            ram_read_resp_6_valid,
            ram_read_resp_7_valid,
        ];

        let state = for (i, state) in range(u32:0, RAM_NUM) {
            if ram_read_resp_valid[i] {
                State {
                    resp_recv: update(state.resp_recv, i, true),
                }
            } else {
                state
            }
        }(state);

        // check if all data is received
        let all_received = for (i, all_received) in range(u32:0, RAM_NUM) {
            all_received & state.resp_recv[i]
        }(true);

        // send response
        send_if(join(), write_resp_s, all_received, WriteResp {});

        // reset state
        let state = if all_received {
            zero!<State>()
        } else {
            state
        };

        state
    }

}


pub proc AlignedParallelRam<
    SIZE: u32,
    DATA_W: u32,
    ADDR_W: u32 = {std::clog2(SIZE)},
    RAM_SIZE: u32 = {SIZE / RAM_NUM},
    RAM_DATA_W: u32 = {DATA_W / RAM_NUM},
    RAM_ADDR_W: u32 = {std::clog2(RAM_SIZE)},
    RAM_PARTITION_SIZE: u32 = {RAM_DATA_W},
    RAM_NUM_PARTITIONS: u32 = {ram::num_partitions(RAM_PARTITION_SIZE, RAM_DATA_W)},
> {
    type ReadReq = AlignedParallelRamReadReq<ADDR_W>;
    type ReadResp = AlignedParallelRamReadResp<DATA_W>;
    type WriteReq = AlignedParallelRamWriteReq<ADDR_W, DATA_W>;
    type WriteResp = AlignedParallelRamWriteResp;

    type RamReadReq = ram::ReadReq<RAM_ADDR_W, RAM_NUM_PARTITIONS>;
    type RamReadResp = ram::ReadResp<RAM_DATA_W>;
    type RamWriteReq = ram::WriteReq<RAM_ADDR_W, RAM_DATA_W, RAM_NUM_PARTITIONS>;
    type RamWriteResp = ram::WriteResp;

    read_req_r: chan<ReadReq> in;
    write_req_r: chan<WriteReq> in;

    read_resp_handler_ctrl_s: chan<AlignedParallelRamReadRespHandlerCtrl> out;

    // RAMs read interfaces
    ram_read_req_0_s: chan<RamReadReq> out;
    ram_read_req_1_s: chan<RamReadReq> out;
    ram_read_req_2_s: chan<RamReadReq> out;
    ram_read_req_3_s: chan<RamReadReq> out;
    ram_read_req_4_s: chan<RamReadReq> out;
    ram_read_req_5_s: chan<RamReadReq> out;
    ram_read_req_6_s: chan<RamReadReq> out;
    ram_read_req_7_s: chan<RamReadReq> out;

    // RAMs write interfaces
    ram_write_req_0_s: chan<RamWriteReq> out;
    ram_write_req_1_s: chan<RamWriteReq> out;
    ram_write_req_2_s: chan<RamWriteReq> out;
    ram_write_req_3_s: chan<RamWriteReq> out;
    ram_write_req_4_s: chan<RamWriteReq> out;
    ram_write_req_5_s: chan<RamWriteReq> out;
    ram_write_req_6_s: chan<RamWriteReq> out;
    ram_write_req_7_s: chan<RamWriteReq> out;

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
        let (read_resp_handler_ctrl_s, read_resp_handler_ctrl_r) =
            chan<AlignedParallelRamReadRespHandlerCtrl, u32:1>("read_resp_handler_ctrl");

        spawn AlignedParallelRamReadRespHandler<DATA_W, RAM_DATA_W>(
            read_resp_handler_ctrl_r,
            read_resp_s,
            ram_read_resp_r[0],
            ram_read_resp_r[1],
            ram_read_resp_r[2],
            ram_read_resp_r[3],
            ram_read_resp_r[4],
            ram_read_resp_r[5],
            ram_read_resp_r[6],
            ram_read_resp_r[7],
        );

        spawn AlignedParallelRamWriteRespHandler (
            write_resp_s,
            ram_write_resp_r[0],
            ram_write_resp_r[1],
            ram_write_resp_r[2],
            ram_write_resp_r[3],
            ram_write_resp_r[4],
            ram_write_resp_r[5],
            ram_write_resp_r[6],
            ram_write_resp_r[7],
        );

        (
            read_req_r,
            write_req_r,
            read_resp_handler_ctrl_s,
            ram_read_req_s[0],
            ram_read_req_s[1],
            ram_read_req_s[2],
            ram_read_req_s[3],
            ram_read_req_s[4],
            ram_read_req_s[5],
            ram_read_req_s[6],
            ram_read_req_s[7],
            ram_write_req_s[0],
            ram_write_req_s[1],
            ram_write_req_s[2],
            ram_write_req_s[3],
            ram_write_req_s[4],
            ram_write_req_s[5],
            ram_write_req_s[6],
            ram_write_req_s[7],
        )
    }

    init { }

    next (state: ()) {
        // handle read request
        let (tok_read, read_req, read_req_valid) = recv_non_blocking(join(), read_req_r, zero!<ReadReq>());

        // send ctrl to read resp hanlder
        let read_resp_handler_ctrl = AlignedParallelRamReadRespHandlerCtrl {
            ram_offset: read_req.addr as uN[RAM_NUM_W],
        };
        send_if(tok_read, read_resp_handler_ctrl_s, read_req_valid, read_resp_handler_ctrl);

        // send requests to each RAM
        let ram_read_req =  for (i, ram_read_req) in range(u32:0, RAM_NUM) {
            let offset = if read_req.addr as uN[RAM_NUM_W] > i as uN[RAM_NUM_W] {
                uN[RAM_ADDR_W]:1
            } else {
                uN[RAM_ADDR_W]:0
            };
            update(ram_read_req, i, RamReadReq {
                addr: (read_req.addr >> std::clog2(RAM_NUM)) as uN[RAM_ADDR_W] + offset,
                mask: !uN[RAM_NUM_PARTITIONS]:0,
            })
        }(zero!<RamReadReq[RAM_NUM]>());
        send_if(tok_read, ram_read_req_0_s, read_req_valid, ram_read_req[0]);
        send_if(tok_read, ram_read_req_1_s, read_req_valid, ram_read_req[1]);
        send_if(tok_read, ram_read_req_2_s, read_req_valid, ram_read_req[2]);
        send_if(tok_read, ram_read_req_3_s, read_req_valid, ram_read_req[3]);
        send_if(tok_read, ram_read_req_4_s, read_req_valid, ram_read_req[4]);
        send_if(tok_read, ram_read_req_5_s, read_req_valid, ram_read_req[5]);
        send_if(tok_read, ram_read_req_6_s, read_req_valid, ram_read_req[6]);
        send_if(tok_read, ram_read_req_7_s, read_req_valid, ram_read_req[7]);

        // handle write request
        let (tok_write, write_req, write_req_valid) = recv_non_blocking(join(), write_req_r, zero!<WriteReq>());

        // send requests to each RAM
        let ram_write_req = for (i, ram_write_req) in range(u32:0, RAM_NUM) {
            update(ram_write_req, i, RamWriteReq {
                addr: (write_req.addr >> std::clog2(RAM_NUM)) as uN[RAM_ADDR_W],
                data: (write_req.data >> (RAM_DATA_W * i)) as uN[RAM_DATA_W],
                mask: !uN[RAM_NUM_PARTITIONS]:0,
            })
        }(zero!<RamWriteReq[RAM_NUM]>());
        send_if(tok_read, ram_write_req_0_s, write_req_valid, ram_write_req[0]);
        send_if(tok_read, ram_write_req_1_s, write_req_valid, ram_write_req[1]);
        send_if(tok_read, ram_write_req_2_s, write_req_valid, ram_write_req[2]);
        send_if(tok_read, ram_write_req_3_s, write_req_valid, ram_write_req[3]);
        send_if(tok_read, ram_write_req_4_s, write_req_valid, ram_write_req[4]);
        send_if(tok_read, ram_write_req_5_s, write_req_valid, ram_write_req[5]);
        send_if(tok_read, ram_write_req_6_s, write_req_valid, ram_write_req[6]);
        send_if(tok_read, ram_write_req_7_s, write_req_valid, ram_write_req[7]);
    }
}


const INST_SIZE = u32:1024;
const INST_DATA_W = u32:64;
const INST_ADDR_W = std::clog2(INST_SIZE);
const INST_RAM_SIZE = INST_SIZE / RAM_NUM;
const INST_RAM_DATA_W = {INST_DATA_W / RAM_NUM};
const INST_RAM_ADDR_W = {std::clog2(INST_RAM_SIZE)};
const INST_RAM_PARTITION_SIZE = {INST_RAM_DATA_W};
const INST_RAM_NUM_PARTITIONS = {ram::num_partitions(INST_RAM_PARTITION_SIZE, INST_RAM_DATA_W)};

proc AlignedParallelRamInst {
    type InstReadReq = AlignedParallelRamReadReq<INST_ADDR_W>;
    type InstReadResp = AlignedParallelRamReadResp<INST_DATA_W>;
    type InstWriteReq = AlignedParallelRamWriteReq<INST_ADDR_W, INST_DATA_W>;
    type InstWriteResp = AlignedParallelRamWriteResp;

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
        spawn AlignedParallelRam<INST_SIZE, INST_DATA_W>(
            read_req_r, read_resp_s,
            write_req_r, write_resp_s,
            ram_read_req_s, ram_read_resp_r,
            ram_write_req_s, ram_write_resp_r,
        );
    }

    init { }

    next (state: ()) { }
}


const TEST_SIZE = u32:1024;
const TEST_DATA_W = u32:64;
const TEST_ADDR_W = std::clog2(TEST_SIZE);
const TEST_RAM_SIZE = TEST_SIZE / RAM_NUM;
const TEST_RAM_DATA_W = {TEST_DATA_W / RAM_NUM};
const TEST_RAM_ADDR_W = {std::clog2(TEST_RAM_SIZE)};
const TEST_RAM_PARTITION_SIZE = {TEST_RAM_DATA_W};
const TEST_RAM_NUM_PARTITIONS = {ram::num_partitions(TEST_RAM_PARTITION_SIZE, TEST_RAM_DATA_W)};

const TEST_RAM_SIMULTANEOUS_RW_BEHAVIOR = ram::SimultaneousReadWriteBehavior::READ_BEFORE_WRITE;
const TEST_RAM_INITIALIZED = true;

type TestReadReq = AlignedParallelRamReadReq<TEST_ADDR_W>;
type TestReadResp = AlignedParallelRamReadResp<TEST_DATA_W>;
type TestWriteReq = AlignedParallelRamWriteReq<TEST_ADDR_W, TEST_DATA_W>;
type TestWriteResp = AlignedParallelRamWriteResp;

type TestRamReadReq = ram::ReadReq<TEST_RAM_ADDR_W, TEST_RAM_NUM_PARTITIONS>;
type TestRamReadResp = ram::ReadResp<TEST_RAM_DATA_W>;
type TestRamWriteReq = ram::WriteReq<TEST_RAM_ADDR_W, TEST_RAM_DATA_W, TEST_RAM_NUM_PARTITIONS>;
type TestRamWriteResp = ram::WriteResp;

struct TestData {
    addr: uN[TEST_ADDR_W],
    data: uN[TEST_DATA_W],
}

const TEST_DATA = TestData[64]:[
    TestData {addr: uN[TEST_ADDR_W]:0x0c8, data: uN[TEST_DATA_W]:0x698dbd57f739d8ce},
    TestData {addr: uN[TEST_ADDR_W]:0x248, data: uN[TEST_DATA_W]:0x4cf6fc9b695676ad},
    TestData {addr: uN[TEST_ADDR_W]:0x3a0, data: uN[TEST_DATA_W]:0x5da52c3bd7b39603},
    TestData {addr: uN[TEST_ADDR_W]:0x208, data: uN[TEST_DATA_W]:0x5afa80c1c45a5bd2},
    TestData {addr: uN[TEST_ADDR_W]:0x068, data: uN[TEST_DATA_W]:0x27befcb367237e3f},
    TestData {addr: uN[TEST_ADDR_W]:0x358, data: uN[TEST_DATA_W]:0xa477d4887cec7fc2},
    TestData {addr: uN[TEST_ADDR_W]:0x328, data: uN[TEST_DATA_W]:0x38ecf19cf314ba5c},
    TestData {addr: uN[TEST_ADDR_W]:0x258, data: uN[TEST_DATA_W]:0x97a504cfa39e6750},
    TestData {addr: uN[TEST_ADDR_W]:0x1b8, data: uN[TEST_DATA_W]:0x2fa75c1effecf687},
    TestData {addr: uN[TEST_ADDR_W]:0x2e8, data: uN[TEST_DATA_W]:0xb1315d70b63629d8},
    TestData {addr: uN[TEST_ADDR_W]:0x2f0, data: uN[TEST_DATA_W]:0x44c025ebee513c44},
    TestData {addr: uN[TEST_ADDR_W]:0x250, data: uN[TEST_DATA_W]:0x295250fa0d795902},
    TestData {addr: uN[TEST_ADDR_W]:0x2a0, data: uN[TEST_DATA_W]:0x1f76bb3cf745235e},
    TestData {addr: uN[TEST_ADDR_W]:0x168, data: uN[TEST_DATA_W]:0x0d06b1d161037460},
    TestData {addr: uN[TEST_ADDR_W]:0x010, data: uN[TEST_DATA_W]:0x0c7b320db86382df},
    TestData {addr: uN[TEST_ADDR_W]:0x178, data: uN[TEST_DATA_W]:0x547e5874fdae8c09},
    TestData {addr: uN[TEST_ADDR_W]:0x0f8, data: uN[TEST_DATA_W]:0xc75ca52d83d65bba},
    TestData {addr: uN[TEST_ADDR_W]:0x0d0, data: uN[TEST_DATA_W]:0x3c10031e89ac070a},
    TestData {addr: uN[TEST_ADDR_W]:0x3f8, data: uN[TEST_DATA_W]:0xe881ce7c3e4515b4},
    TestData {addr: uN[TEST_ADDR_W]:0x378, data: uN[TEST_DATA_W]:0xa10c92b84419eb3d},
    TestData {addr: uN[TEST_ADDR_W]:0x018, data: uN[TEST_DATA_W]:0x7b9537f92c4958e0},
    TestData {addr: uN[TEST_ADDR_W]:0x350, data: uN[TEST_DATA_W]:0x38a1a5e8a7206e81},
    TestData {addr: uN[TEST_ADDR_W]:0x030, data: uN[TEST_DATA_W]:0xda2cf6b0b380862c},
    TestData {addr: uN[TEST_ADDR_W]:0x248, data: uN[TEST_DATA_W]:0xa56492b3fb19c8b8},
    TestData {addr: uN[TEST_ADDR_W]:0x258, data: uN[TEST_DATA_W]:0x9cbfccbf72c7948b},
    TestData {addr: uN[TEST_ADDR_W]:0x008, data: uN[TEST_DATA_W]:0x7fb6d361a608db56},
    TestData {addr: uN[TEST_ADDR_W]:0x108, data: uN[TEST_DATA_W]:0xba2aef614c7c5c1e},
    TestData {addr: uN[TEST_ADDR_W]:0x090, data: uN[TEST_DATA_W]:0xe7a5ab55633078fa},
    TestData {addr: uN[TEST_ADDR_W]:0x0c0, data: uN[TEST_DATA_W]:0xb5132e7e378f3f5b},
    TestData {addr: uN[TEST_ADDR_W]:0x198, data: uN[TEST_DATA_W]:0xeac9fe191bfd8b31},
    TestData {addr: uN[TEST_ADDR_W]:0x218, data: uN[TEST_DATA_W]:0x82ad45d959f8dbec},
    TestData {addr: uN[TEST_ADDR_W]:0x070, data: uN[TEST_DATA_W]:0x4d4e255058d00ccb},
    TestData {addr: uN[TEST_ADDR_W]:0x3a0, data: uN[TEST_DATA_W]:0x2a69306cf695b2f5},
    TestData {addr: uN[TEST_ADDR_W]:0x1e0, data: uN[TEST_DATA_W]:0x571a30f8cd940e39},
    TestData {addr: uN[TEST_ADDR_W]:0x300, data: uN[TEST_DATA_W]:0x7069a4c406076fd9},
    TestData {addr: uN[TEST_ADDR_W]:0x2a8, data: uN[TEST_DATA_W]:0x9af366c878230764},
    TestData {addr: uN[TEST_ADDR_W]:0x328, data: uN[TEST_DATA_W]:0x1e6bc1e2df3c8a7b},
    TestData {addr: uN[TEST_ADDR_W]:0x298, data: uN[TEST_DATA_W]:0x1ff9be4f810cd87a},
    TestData {addr: uN[TEST_ADDR_W]:0x250, data: uN[TEST_DATA_W]:0x9ad30cee350aebfa},
    TestData {addr: uN[TEST_ADDR_W]:0x090, data: uN[TEST_DATA_W]:0x31fca7f91dfcafb5},
    TestData {addr: uN[TEST_ADDR_W]:0x3b8, data: uN[TEST_DATA_W]:0xe434deef583c3cd1},
    TestData {addr: uN[TEST_ADDR_W]:0x3c0, data: uN[TEST_DATA_W]:0x4170c371a2025f27},
    TestData {addr: uN[TEST_ADDR_W]:0x0e8, data: uN[TEST_DATA_W]:0x616754a100d9decc},
    TestData {addr: uN[TEST_ADDR_W]:0x1f0, data: uN[TEST_DATA_W]:0x8d93fa35edab37b7},
    TestData {addr: uN[TEST_ADDR_W]:0x208, data: uN[TEST_DATA_W]:0x6582012a83ffcec3},
    TestData {addr: uN[TEST_ADDR_W]:0x3d0, data: uN[TEST_DATA_W]:0x6c66a69e87eac130},
    TestData {addr: uN[TEST_ADDR_W]:0x248, data: uN[TEST_DATA_W]:0xbfd5e4e261bbd7e3},
    TestData {addr: uN[TEST_ADDR_W]:0x058, data: uN[TEST_DATA_W]:0x2f8ba1fd6a8b6ee9},
    TestData {addr: uN[TEST_ADDR_W]:0x1a0, data: uN[TEST_DATA_W]:0xef9ab2936ef6833e},
    TestData {addr: uN[TEST_ADDR_W]:0x380, data: uN[TEST_DATA_W]:0x279130ba7b5ced6f},
    TestData {addr: uN[TEST_ADDR_W]:0x170, data: uN[TEST_DATA_W]:0xc1977f6a2153db09},
    TestData {addr: uN[TEST_ADDR_W]:0x3d8, data: uN[TEST_DATA_W]:0xd4ea85571e440cef},
    TestData {addr: uN[TEST_ADDR_W]:0x360, data: uN[TEST_DATA_W]:0x9bc5756ab3328603},
    TestData {addr: uN[TEST_ADDR_W]:0x2f8, data: uN[TEST_DATA_W]:0x14217d1804170f39},
    TestData {addr: uN[TEST_ADDR_W]:0x268, data: uN[TEST_DATA_W]:0x0098755165e9ae81},
    TestData {addr: uN[TEST_ADDR_W]:0x050, data: uN[TEST_DATA_W]:0x3ee0b48789cc94e0},
    TestData {addr: uN[TEST_ADDR_W]:0x398, data: uN[TEST_DATA_W]:0x9ff7fbc9906d3d63},
    TestData {addr: uN[TEST_ADDR_W]:0x068, data: uN[TEST_DATA_W]:0x507bc61f805b0e68},
    TestData {addr: uN[TEST_ADDR_W]:0x350, data: uN[TEST_DATA_W]:0x77802819dc14663a},
    TestData {addr: uN[TEST_ADDR_W]:0x168, data: uN[TEST_DATA_W]:0xd8ca0711ca37bfa9},
    TestData {addr: uN[TEST_ADDR_W]:0x068, data: uN[TEST_DATA_W]:0x30464e3d2630b6de},
    TestData {addr: uN[TEST_ADDR_W]:0x360, data: uN[TEST_DATA_W]:0xdbac58596c50f62f},
    TestData {addr: uN[TEST_ADDR_W]:0x2e0, data: uN[TEST_DATA_W]:0x9992cfd966824669},
    TestData {addr: uN[TEST_ADDR_W]:0x2e0, data: uN[TEST_DATA_W]:0x1a4a65b0257c223b},
];

#[test_proc]
proc AlignedParallelRam_test_aligned_read {
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

        spawn AlignedParallelRam<TEST_SIZE, TEST_DATA_W>(
            read_req_r, read_resp_s,
            write_req_r, write_resp_s,
            ram_read_req_s, ram_read_resp_r,
            ram_write_req_s, ram_write_resp_r,
        );

        (
            terminator,
            read_req_s,
            read_resp_r,
            write_req_s,
            write_resp_r,
        )
    }

    init { }

    next (state: ()) {
        let tok = join();

        let tok = for (i, tok):(u32, token) in range(u32:0, array_size(TEST_DATA)) {
            let test_data = TEST_DATA[i];

            let write_req = TestWriteReq {
                addr: test_data.addr,
                data: test_data.data,
            };
            let tok = send(tok, write_req_s, write_req);
            trace_fmt!("Sent #{} write request {:#x}", i + u32:1, write_req);

            let (tok, write_resp) = recv(tok, write_resp_r);
            trace_fmt!("Received #{} write response {:#x}", i + u32:1, write_resp);

            let read_req = TestReadReq {
                addr: test_data.addr,
            };
            let tok = send(tok, read_req_s, read_req);
            trace_fmt!("Sent #{} read request {:#x}", i + u32:1, read_req);

            let (tok,  read_resp) = recv(tok, read_resp_r);
            trace_fmt!("Received #{} read response {:#x}", i + u32:1, read_resp);

            assert_eq(test_data.data, read_resp.data);

            tok
        }(tok);

        send(tok, terminator, true);
    }
}

const TEST_RAM_CONTENT = uN[TEST_DATA_W][64]:[
    uN[TEST_DATA_W]:0x2122337ed367496b, uN[TEST_DATA_W]:0x33de22e4291ecb66,
    uN[TEST_DATA_W]:0x62052eccbde0009d, uN[TEST_DATA_W]:0xfa179c402e7b5f47,
    uN[TEST_DATA_W]:0x118fa2c81d1230e9, uN[TEST_DATA_W]:0xe48ee076b41120c0,
    uN[TEST_DATA_W]:0xa33d467d80575e5b, uN[TEST_DATA_W]:0x61213ebe00890570,
    uN[TEST_DATA_W]:0xe9210eae2507442f, uN[TEST_DATA_W]:0x4b8c721627c19c44,
    uN[TEST_DATA_W]:0x55e768d2e4586bba, uN[TEST_DATA_W]:0x2fc234017ac1deb5,
    uN[TEST_DATA_W]:0xdc6afd5db30446aa, uN[TEST_DATA_W]:0xe91512402a2f68ab,
    uN[TEST_DATA_W]:0x13fd96b93aef2c85, uN[TEST_DATA_W]:0x980b4054b9f66fc2,
    uN[TEST_DATA_W]:0xa8bf09e77757ca28, uN[TEST_DATA_W]:0x94a67ff04004de7b,
    uN[TEST_DATA_W]:0x6d1f3071a446b0c3, uN[TEST_DATA_W]:0x01605527a50fdecf,
    uN[TEST_DATA_W]:0xd839258508c3efd1, uN[TEST_DATA_W]:0x1207a4d0de5c9724,
    uN[TEST_DATA_W]:0xef39682f0810f43c, uN[TEST_DATA_W]:0x4781977bc26ce834,
    uN[TEST_DATA_W]:0x0805a350ed25812f, uN[TEST_DATA_W]:0x8ad82cb67bf49cef,
    uN[TEST_DATA_W]:0x2fd11ff78f85f169, uN[TEST_DATA_W]:0xd624be58457eab2a,
    uN[TEST_DATA_W]:0x873e4be71afa1355, uN[TEST_DATA_W]:0x6e0e9f264151b531,
    uN[TEST_DATA_W]:0xa69015c537b4da78, uN[TEST_DATA_W]:0x0879638aa8045ad9,
    uN[TEST_DATA_W]:0x30dd170b7bf89cbf, uN[TEST_DATA_W]:0xcbfeb8219960a267,
    uN[TEST_DATA_W]:0x9f6fcd2d4a4ba9f2, uN[TEST_DATA_W]:0xdf0ead33b3e55ac3,
    uN[TEST_DATA_W]:0x64a24c19037f850b, uN[TEST_DATA_W]:0x4dcbb4de2d3aba5a,
    uN[TEST_DATA_W]:0xa40749b2be1450b6, uN[TEST_DATA_W]:0x99bed65d2d28d1f6,
    uN[TEST_DATA_W]:0xbe8d35f27bb892b4, uN[TEST_DATA_W]:0x23315a3a70110048,
    uN[TEST_DATA_W]:0x68b0e22cb8885787, uN[TEST_DATA_W]:0xcac1a152d43dae98,
    uN[TEST_DATA_W]:0x1fb5cec8c64ad46a, uN[TEST_DATA_W]:0xcbe25f0d2b21e9a1,
    uN[TEST_DATA_W]:0x46e161fd5f490ae7, uN[TEST_DATA_W]:0xff2dd0e7a120d222,
    uN[TEST_DATA_W]:0xa764165b6d09fb90, uN[TEST_DATA_W]:0xee4d9484b63f6a66,
    uN[TEST_DATA_W]:0x204d6d789e9fe377, uN[TEST_DATA_W]:0x9ad53311a1a95bcf,
    uN[TEST_DATA_W]:0x63d497f105d8661f, uN[TEST_DATA_W]:0x40e7a242cc26483c,
    uN[TEST_DATA_W]:0x5a82a7265627cab1, uN[TEST_DATA_W]:0x42de42323222a24b,
    uN[TEST_DATA_W]:0xdede8c218f3ef36a, uN[TEST_DATA_W]:0xec86b8e8734da0c7,
    uN[TEST_DATA_W]:0x9b209d6959c36b79, uN[TEST_DATA_W]:0x829c158fd6678675,
    uN[TEST_DATA_W]:0x5c59a4845b68a509, uN[TEST_DATA_W]:0xcc9e851a38b01d04,
    uN[TEST_DATA_W]:0x5e15f41bd09acd33, uN[TEST_DATA_W]:0x953425686ce51623,
];

const TEST_READ_ADDR = uN[TEST_ADDR_W][128]:[
    uN[TEST_ADDR_W]:0x0d0, uN[TEST_ADDR_W]:0x01c, uN[TEST_ADDR_W]:0x094, uN[TEST_ADDR_W]:0x153,
    uN[TEST_ADDR_W]:0x1cb, uN[TEST_ADDR_W]:0x14f, uN[TEST_ADDR_W]:0x021, uN[TEST_ADDR_W]:0x1f5,
    uN[TEST_ADDR_W]:0x155, uN[TEST_ADDR_W]:0x0db, uN[TEST_ADDR_W]:0x070, uN[TEST_ADDR_W]:0x13a,
    uN[TEST_ADDR_W]:0x0bf, uN[TEST_ADDR_W]:0x16b, uN[TEST_ADDR_W]:0x143, uN[TEST_ADDR_W]:0x0b4,
    uN[TEST_ADDR_W]:0x1f4, uN[TEST_ADDR_W]:0x17f, uN[TEST_ADDR_W]:0x096, uN[TEST_ADDR_W]:0x03a,
    uN[TEST_ADDR_W]:0x0ec, uN[TEST_ADDR_W]:0x030, uN[TEST_ADDR_W]:0x0e1, uN[TEST_ADDR_W]:0x1e7,
    uN[TEST_ADDR_W]:0x006, uN[TEST_ADDR_W]:0x088, uN[TEST_ADDR_W]:0x1e9, uN[TEST_ADDR_W]:0x16f,
    uN[TEST_ADDR_W]:0x152, uN[TEST_ADDR_W]:0x1a2, uN[TEST_ADDR_W]:0x0ac, uN[TEST_ADDR_W]:0x0d3,
    uN[TEST_ADDR_W]:0x0d5, uN[TEST_ADDR_W]:0x107, uN[TEST_ADDR_W]:0x121, uN[TEST_ADDR_W]:0x01a,
    uN[TEST_ADDR_W]:0x1c2, uN[TEST_ADDR_W]:0x117, uN[TEST_ADDR_W]:0x0e9, uN[TEST_ADDR_W]:0x0ac,
    uN[TEST_ADDR_W]:0x16e, uN[TEST_ADDR_W]:0x105, uN[TEST_ADDR_W]:0x01e, uN[TEST_ADDR_W]:0x186,
    uN[TEST_ADDR_W]:0x1bb, uN[TEST_ADDR_W]:0x05b, uN[TEST_ADDR_W]:0x07a, uN[TEST_ADDR_W]:0x1d3,
    uN[TEST_ADDR_W]:0x120, uN[TEST_ADDR_W]:0x142, uN[TEST_ADDR_W]:0x0ee, uN[TEST_ADDR_W]:0x083,
    uN[TEST_ADDR_W]:0x1ce, uN[TEST_ADDR_W]:0x016, uN[TEST_ADDR_W]:0x041, uN[TEST_ADDR_W]:0x040,
    uN[TEST_ADDR_W]:0x073, uN[TEST_ADDR_W]:0x197, uN[TEST_ADDR_W]:0x1d1, uN[TEST_ADDR_W]:0x074,
    uN[TEST_ADDR_W]:0x087, uN[TEST_ADDR_W]:0x168, uN[TEST_ADDR_W]:0x1f7, uN[TEST_ADDR_W]:0x19e,
    uN[TEST_ADDR_W]:0x06f, uN[TEST_ADDR_W]:0x0c9, uN[TEST_ADDR_W]:0x102, uN[TEST_ADDR_W]:0x077,
    uN[TEST_ADDR_W]:0x0ff, uN[TEST_ADDR_W]:0x1ac, uN[TEST_ADDR_W]:0x02c, uN[TEST_ADDR_W]:0x116,
    uN[TEST_ADDR_W]:0x04d, uN[TEST_ADDR_W]:0x16b, uN[TEST_ADDR_W]:0x14c, uN[TEST_ADDR_W]:0x173,
    uN[TEST_ADDR_W]:0x055, uN[TEST_ADDR_W]:0x1e1, uN[TEST_ADDR_W]:0x028, uN[TEST_ADDR_W]:0x103,
    uN[TEST_ADDR_W]:0x01c, uN[TEST_ADDR_W]:0x168, uN[TEST_ADDR_W]:0x096, uN[TEST_ADDR_W]:0x15b,
    uN[TEST_ADDR_W]:0x1aa, uN[TEST_ADDR_W]:0x010, uN[TEST_ADDR_W]:0x08c, uN[TEST_ADDR_W]:0x083,
    uN[TEST_ADDR_W]:0x014, uN[TEST_ADDR_W]:0x013, uN[TEST_ADDR_W]:0x00d, uN[TEST_ADDR_W]:0x1eb,
    uN[TEST_ADDR_W]:0x09d, uN[TEST_ADDR_W]:0x079, uN[TEST_ADDR_W]:0x146, uN[TEST_ADDR_W]:0x191,
    uN[TEST_ADDR_W]:0x070, uN[TEST_ADDR_W]:0x1bc, uN[TEST_ADDR_W]:0x037, uN[TEST_ADDR_W]:0x130,
    uN[TEST_ADDR_W]:0x0d8, uN[TEST_ADDR_W]:0x0d1, uN[TEST_ADDR_W]:0x136, uN[TEST_ADDR_W]:0x05b,
    uN[TEST_ADDR_W]:0x1f3, uN[TEST_ADDR_W]:0x036, uN[TEST_ADDR_W]:0x0db, uN[TEST_ADDR_W]:0x149,
    uN[TEST_ADDR_W]:0x11e, uN[TEST_ADDR_W]:0x1c2, uN[TEST_ADDR_W]:0x0a3, uN[TEST_ADDR_W]:0x061,
    uN[TEST_ADDR_W]:0x0eb, uN[TEST_ADDR_W]:0x131, uN[TEST_ADDR_W]:0x04a, uN[TEST_ADDR_W]:0x0ab,
    uN[TEST_ADDR_W]:0x0d5, uN[TEST_ADDR_W]:0x083, uN[TEST_ADDR_W]:0x1cb, uN[TEST_ADDR_W]:0x03f,
    uN[TEST_ADDR_W]:0x02d, uN[TEST_ADDR_W]:0x14d, uN[TEST_ADDR_W]:0x120, uN[TEST_ADDR_W]:0x194,
    uN[TEST_ADDR_W]:0x062, uN[TEST_ADDR_W]:0x182, uN[TEST_ADDR_W]:0x124, uN[TEST_ADDR_W]:0x06d,
];

#[test_proc]
proc AlignedParallelRam_test_non_aligned_read {
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

        spawn AlignedParallelRam<TEST_SIZE, TEST_DATA_W>(
            read_req_r, read_resp_s,
            write_req_r, write_resp_s,
            ram_read_req_s, ram_read_resp_r,
            ram_write_req_s, ram_write_resp_r,
        );

        (
            terminator,
            read_req_s,
            read_resp_r,
            write_req_s,
            write_resp_r,
        )
    }

    init { }

    next (state: ()) {
        let tok = join();

        // write RAM content
        let tok = for (i, tok):(u32, token) in range(u32:0, array_size(TEST_RAM_CONTENT)) {
            let test_data = TEST_RAM_CONTENT[i];

            let write_req = TestWriteReq {
                addr: (i * TEST_RAM_DATA_W) as uN[TEST_ADDR_W],
                data: test_data,
            };
            let tok = send(tok, write_req_s, write_req);
            trace_fmt!("Sent #{} write request {:#x}", i + u32:1, write_req);

            let (tok, write_resp) = recv(tok, write_resp_r);
            trace_fmt!("Received #{} write response {:#x}", i + u32:1, write_resp);

            let read_req = TestReadReq {
                addr: (i * TEST_RAM_DATA_W) as uN[TEST_ADDR_W],
            };
            let tok = send(tok, read_req_s, read_req);
            trace_fmt!("Sent #{} read request {:#x}", i + u32:1, read_req);

            let (tok,  read_resp) = recv(tok, read_resp_r);
            trace_fmt!("Received #{} read response {:#x}", i + u32:1, read_resp);

            assert_eq(test_data, read_resp.data);

            tok
        }(tok);

        // read unaligned data
        let tok = for (i, tok):(u32, token) in range(u32:0, array_size(TEST_READ_ADDR)) {
            let test_read_addr = TEST_READ_ADDR[i];

            let read_req = TestReadReq {
                addr: test_read_addr,
            };

            let tok = send(tok, read_req_s, read_req);
            trace_fmt!("Sent #{} read request {:#x}", i + u32:1, read_req);

            let (tok,  read_resp) = recv(tok, read_resp_r);
            trace_fmt!("Received #{} read response {:#x}", i + u32:1, read_resp);

            let ram_offset = test_read_addr as uN[RAM_NUM_W];
            let expected_data = if ram_offset == uN[RAM_NUM_W]:0 {
                TEST_RAM_CONTENT[test_read_addr >> RAM_NUM_W]
            } else {
                let data_0 = TEST_RAM_CONTENT[test_read_addr >> RAM_NUM_W];
                let data_1 = TEST_RAM_CONTENT[(test_read_addr >> RAM_NUM_W) + uN[TEST_ADDR_W]:1];
                (
                    (data_0 >> (TEST_RAM_DATA_W * ram_offset as u32)) |
                    (data_1 << (TEST_RAM_DATA_W * (RAM_NUM - ram_offset as u32)))
                )
            };

            assert_eq(expected_data, read_resp.data);

            tok
        }(tok);

        send(tok, terminator, true);
    }
}
