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

// Generic Register File

import std;
import xls.modules.dma.config;
import xls.modules.dma.common;

// FIXME: casting imported types is a workaround
// https://github.com/google/xls/issues/1030
type MainCtrlBundle = common::MainCtrlBundle;

pub struct ReadReq<ADDR_W: u32> { addr: uN[ADDR_W] }

pub struct ReadResp<DATA_W: u32> { data: uN[DATA_W] }

pub struct WriteReq<ADDR_W: u32, DATA_W: u32> { addr: uN[ADDR_W], data: uN[DATA_W] }

pub struct WriteResp {}

pub fn WriteWordReq<ADDR_W: u32, DATA_W: u32>(addr: uN[ADDR_W], data: uN[DATA_W]) -> WriteReq {
    WriteReq<ADDR_W, DATA_W> { addr, data }
}

pub fn ReadWordReq<ADDR_W: u32>(addr: uN[ADDR_W]) -> ReadReq { ReadReq<ADDR_W> { addr } }

#[test]
fn TestWrite() {
    assert_eq(WriteWordReq(u8:0x34, u16:0x1234), WriteReq { addr: u8:0x34, data: u16:0x1234 })
}

#[test]
fn TestRead() { assert_eq(ReadWordReq(u8:0x34), ReadReq { addr: u8:0x34 }) }

pub fn ptrace<DATA_W: u32, REGS_N: u32>(registers: uN[DATA_W][REGS_N]) -> () {
    trace_fmt!("------------------------------------");
    trace_fmt!("CONTROL_REGISTER            0x00  {:b}", registers[0]);
    trace_fmt!("STATUS_REGISTER             0x01  {:b}", registers[1]);
    trace_fmt!("INTERRUPT_MASK_REGISTER     0x02  {:b}", registers[2]);
    trace_fmt!("INTERRUPT_STATUS_REGISTER   0x03  {:b}", registers[3]);
    trace_fmt!("READER_START_ADDRESS        0x04  {:x}", registers[4]);
    trace_fmt!("READER_LINE_LENGTH          0x05  {:x}", registers[5]);
    trace_fmt!("READER_LINE_COUNT           0x06  {:x}", registers[6]);
    trace_fmt!("READER_STRIDE_BETWEEN_LINES 0x07  {:x}", registers[7]);
    trace_fmt!("WRITER_START_ADDRESS        0x08  {:x}", registers[8]);
    trace_fmt!("WRITER_LINE_LENGTH          0x09  {:x}", registers[9]);
    trace_fmt!("WRITER_LINE_COUNT           0x0A  {:x}", registers[10]);
    trace_fmt!("WRITER_STRIDE_BETWEEN_LINES 0x0B  {:x}", registers[11]);
    trace_fmt!("VERSION_REGISTER            0x0C  {:x}", registers[12]);
    trace_fmt!("CONFIGURATION_REGISTER      0x0D  {:x}", registers[13]);
    trace_fmt!("------------------------------------");
}

struct CsrState<DATA_W: u32, REGS_N: u32> {
    register_file: uN[DATA_W][REGS_N],
    writer_sync: u1,
    writer_wait: u1,
    reader_sync: u1,
    reader_wait: u1,
}

// TODO: If software writes '1' to interrupt register bit, the bit should be cleared
// TODO: Feature to synchronize to external clock signal is missing
proc Csr<ADDR_W: u32, DATA_W: u32, REGS_N: u32> {
    read_req: chan<ReadReq<ADDR_W>> in;
    read_resp: chan<ReadResp<DATA_W>> out;
    write_req: chan<WriteReq<ADDR_W, DATA_W>> in;
    write_resp: chan<WriteResp> out;
    ch_writer_start: chan<u1> out;
    ch_writer_configuration: chan<MainCtrlBundle<ADDR_W>> out;
    ch_writer_busy: chan<u1> in;
    ch_writer_done: chan<u1> in;
    ch_reader_start: chan<u1> out;
    ch_reader_configuration: chan<MainCtrlBundle<ADDR_W>> out;
    ch_reader_busy: chan<u1> in;
    ch_reader_done: chan<u1> in;
    reader_sync_req: chan<()> in;
    reader_sync_rsp: chan<()> out;
    writer_sync_req: chan<()> in;
    writer_sync_rsp: chan<()> out;

    config(read_req: chan<ReadReq<ADDR_W>> in, read_resp: chan<ReadResp<DATA_W>> out,
           write_req: chan<WriteReq<ADDR_W, DATA_W>> in, write_resp: chan<WriteResp> out,
           ch_writer_start: chan<u1> out, ch_writer_configuration: chan<MainCtrlBundle<ADDR_W>> out,
           ch_writer_busy: chan<u1> in, ch_writer_done: chan<u1> in, ch_reader_start: chan<u1> out,
           ch_reader_configuration: chan<MainCtrlBundle<ADDR_W>> out, ch_reader_busy: chan<u1> in,
           ch_reader_done: chan<u1> in, reader_sync_req: chan<()> in, reader_sync_rsp: chan<()> out,
           writer_sync_req: chan<()> in, writer_sync_rsp: chan<()> out) {
        (
            read_req, read_resp, write_req, write_resp, ch_writer_start, ch_writer_configuration,
            ch_writer_busy, ch_writer_done, ch_reader_start, ch_reader_configuration,
            ch_reader_busy, ch_reader_done, reader_sync_req, reader_sync_rsp, writer_sync_req,
            writer_sync_rsp,
        )
    }

    init {
        (CsrState {
            register_file: uN[DATA_W][REGS_N]:[uN[DATA_W]:0, ...],
            writer_sync: u1:0,
            reader_sync: u1:0,
            writer_wait: u1:0,
            reader_wait: u1:0
        })
    }

    next(tok: token, state: CsrState<DATA_W, REGS_N>) {
        trace!(state.register_file);
        trace!(state.writer_sync);
        trace!(state.writer_wait);
        trace!(state.reader_sync);
        trace!(state.reader_wait);
        // Handle read request
        let zero_read_req = ReadReq { addr: uN[ADDR_W]:0 };
        let (tok, read_req, read_req_valid) = recv_non_blocking(tok, read_req, zero_read_req);

        // Handle write request
        let zero_write_req = WriteReq { addr: uN[ADDR_W]:0, data: uN[DATA_W]:0 };
        let (tok, write_req, write_req_valid) = recv_non_blocking(tok, write_req, zero_write_req);

        // Handle read response
        let read_value = state.register_file[read_req.addr];
        let read_resp_value = ReadResp { data: read_value };
        let tok = send_if(tok, read_resp, read_req_valid, read_resp_value);

        // Handle write response
        let next_register_file = if write_req_valid {
            update(state.register_file, write_req.addr, write_req.data)
        } else {
            state.register_file
        };
        let tok = send_if(tok, write_resp, write_req_valid, WriteResp {});

        // Interface to Address Generator

        // Writer: External synchronization
        let (tok, _, writer_sync_valid) = recv_non_blocking(tok, writer_sync_req, ());
        let tok = send_if(tok, writer_sync_rsp, writer_sync_valid, ());

        let next_writer_sync = if writer_sync_valid { u1:1 } else { state.writer_sync };

        let next_writer_wait = if state.register_file[config::CONTROL_REGISTER][0:1] {
            u1:1
        } else {
            state.writer_wait
        };

        // Writer: Transfer
        let start_write_transfer = if state.register_file[config::CONTROL_REGISTER][2:3] {
            state.register_file[config::CONTROL_REGISTER][0:1]
        } else {
            state.writer_wait & state.writer_sync
        };

        let (next_writer_sync, next_writer_wait) = if state.writer_wait && state.writer_sync {
            (u1:0, u1:0)
        } else {
            (next_writer_sync, next_writer_wait)
        };

        // Writer: Transfer configuration
        let tok = send(tok, ch_writer_start, start_write_transfer);
        let writer_ctrl_bundle = MainCtrlBundle<ADDR_W> {
            start_address: state.register_file[config::WRITER_START_ADDRESS] as uN[ADDR_W],
            line_count: state.register_file[config::WRITER_LINE_LENGTH] as uN[ADDR_W],
            line_length: state.register_file[config::WRITER_LINE_COUNT] as uN[ADDR_W],
            line_stride: state.register_file[config::WRITER_STRIDE_BETWEEN_LINES] as uN[ADDR_W]
        };
        let tok = send(tok, ch_writer_configuration, writer_ctrl_bundle);

        // Writer: Status busy
        let (tok, writer_busy, writer_busy_valid) = recv_non_blocking(tok, ch_writer_busy, u1:0);
        let next_register_file = if writer_busy_valid {
            let status_register = next_register_file[config::STATUS_REGISTER];
            // Bit 0 is "Writer busy"
            let status_register = bit_slice_update(status_register, uN[ADDR_W]:0, writer_busy);
            update(next_register_file, config::STATUS_REGISTER, status_register)
        } else {
            next_register_file
        };

        // Writer: Interrupt
        let (tok, writer_done, writer_done_valid) = recv_non_blocking(tok, ch_writer_done, u1:0);
        let next_register_file = if writer_done_valid {
            let irq_status_register = next_register_file[config::INTERRUPT_STATUS_REGISTER];
            // Bit 0 is "writer done"
            let irq_status_register =
                bit_slice_update(irq_status_register, uN[ADDR_W]:0, writer_done);
            update(next_register_file, config::INTERRUPT_STATUS_REGISTER, irq_status_register)
        } else {
            next_register_file
        };

        // reader: External synchronization
        let (tok, _, reader_sync_valid) = recv_non_blocking(tok, reader_sync_req, ());
        let tok = send_if(tok, reader_sync_rsp, reader_sync_valid, ());

        let next_reader_sync = if reader_sync_valid { u1:1 } else { state.reader_sync };

        let next_reader_wait = if state.register_file[config::CONTROL_REGISTER][1:2] {
            u1:1
        } else {
            state.reader_wait
        };

        // reader: Transfer
        let start_read_transfer = if state.register_file[config::CONTROL_REGISTER][3:4] {
            state.register_file[config::CONTROL_REGISTER][1:2]
        } else {
            state.reader_wait & state.reader_sync
        };

        let (next_reader_sync, next_reader_wait) = if state.reader_wait && state.reader_sync {
            (u1:0, u1:0)
        } else {
            (next_reader_sync, next_reader_wait)
        };

        // Reader: Transfer configuration
        let tok = send(tok, ch_reader_start, start_read_transfer);
        let reader_ctrl_bundle = MainCtrlBundle<ADDR_W> {
            start_address: state.register_file[config::READER_START_ADDRESS] as uN[ADDR_W],
            line_count: state.register_file[config::READER_LINE_LENGTH] as uN[ADDR_W],
            line_length: state.register_file[config::READER_LINE_COUNT] as uN[ADDR_W],
            line_stride: state.register_file[config::READER_STRIDE_BETWEEN_LINES] as uN[ADDR_W]
        };
        let tok = send(tok, ch_reader_configuration, reader_ctrl_bundle);

        // Reader: Status busy
        let (tok, reader_busy, reader_busy_valid) = recv_non_blocking(tok, ch_reader_busy, u1:0);
        let next_register_file = if reader_busy_valid {
            let status_register = next_register_file[config::STATUS_REGISTER];
            // Bit 1 is "reader busy"
            let status_register = bit_slice_update(status_register, uN[ADDR_W]:1, reader_busy);
            update(next_register_file, config::STATUS_REGISTER, status_register)
        } else {
            next_register_file
        };

        // Reader: Interrupt
        let (tok, reader_done, reader_done_valid) = recv_non_blocking(tok, ch_reader_done, u1:0);
        let next_register_file = if reader_done_valid {
            let irq_status_register = next_register_file[config::INTERRUPT_STATUS_REGISTER];
            // Bit 1 is "reader done"
            let irq_status_register =
                bit_slice_update(irq_status_register, uN[ADDR_W]:1, reader_done);
            update(next_register_file, config::INTERRUPT_STATUS_REGISTER, irq_status_register)
        } else {
            next_register_file
        };

        // Common: Clear start bit unless in loop-mode
        let control_register = state.register_file[config::CONTROL_REGISTER];
        let is_writer_loop_mode = control_register[4:5];
        let is_reader_loop_mode = control_register[5:6];
        //
        let next_register_file = if !is_writer_loop_mode & control_register[0:1] {
            let control_register = next_register_file[config::CONTROL_REGISTER];
            let control_register = bit_slice_update(control_register, uN[ADDR_W]:0, u1:0);
            update(next_register_file, config::CONTROL_REGISTER, control_register)
        } else {
            next_register_file
        };
        //
        let next_register_file = if !is_reader_loop_mode & control_register[1:2] {
            let control_register = next_register_file[config::CONTROL_REGISTER];
            let control_register = bit_slice_update(control_register, uN[ADDR_W]:1, u1:0);
            update(next_register_file, config::CONTROL_REGISTER, control_register)
        } else {
            next_register_file
        };

        // State
        CsrState {
            register_file: next_register_file,
            writer_sync: next_writer_sync,
            reader_sync: next_reader_sync,
            writer_wait: next_writer_wait,
            reader_wait: next_reader_wait
        }
    }
}

// Test writing and reading from CSRs

// Important notes for test!
// 1.
// Processing is selected with a CTRL_WORD:
// 000011 =  3 - in single mode
// 001111 = 15 - in single mode, disable sync
// 110011 = 51 - in loop mode
// 111111 = 63 - in loop mode, disable sync
// 2.
// TODO: Improve signalling in SingleMode tests
// The state variable in this test:
// - increments after initial config
// - increments with each transfer
// - increases by 10 with each `next` cycle without a transfer
// In Single Mode, there is only 1 valid transfer, so we expect the state to be 2,
// however, there will be some cycles without transfers.
// The value of 1200 is set to let the test execute 120 `next` cycles,
// which is sufficient for current implementation.

const TEST_ADDR_W = config::CSR_ADDR_W;
const TEST_DATA_W = config::CSR_DATA_W;
const TEST_REGS_W = config::CSR_DATA_W;
const TEST_REGS_N = config::CSR_REGS_N;

type AdrDatPair = (uN[TEST_ADDR_W], uN[TEST_DATA_W]);

#[test_proc]
proc TestReadWrite {
    read_req: chan<ReadReq<TEST_ADDR_W>> out;
    read_resp: chan<ReadResp<TEST_DATA_W>> in;
    write_req: chan<WriteReq<TEST_ADDR_W, TEST_DATA_W>> out;
    write_resp: chan<WriteResp> in;
    ch_writer_start: chan<u1> in;
    ch_writer_configuration: chan<MainCtrlBundle<TEST_ADDR_W>> in;
    ch_writer_busy: chan<u1> out;
    ch_writer_done: chan<u1> out;
    ch_reader_start: chan<u1> in;
    ch_reader_configuration: chan<MainCtrlBundle<TEST_ADDR_W>> in;
    ch_reader_busy: chan<u1> out;
    ch_reader_done: chan<u1> out;
    reader_sync_req: chan<()> out;
    reader_sync_rsp: chan<()> in;
    writer_sync_req: chan<()> out;
    writer_sync_rsp: chan<()> in;
    terminator: chan<bool> out;

    config(terminator: chan<bool> out) {
        let (read_req_s, read_req_r) = chan<ReadReq<TEST_ADDR_W>>;
        let (read_resp_s, read_resp_r) = chan<ReadResp<TEST_DATA_W>>;
        let (write_req_s, write_req_r) = chan<WriteReq<TEST_ADDR_W, TEST_DATA_W>>;
        let (write_resp_s, write_resp_r) = chan<WriteResp>;

        let (ch_writer_start_s, ch_writer_start_r) = chan<u1>;
        let (ch_writer_configuration_s, ch_writer_configuration_r) =
            chan<MainCtrlBundle<TEST_ADDR_W>>;
        let (ch_writer_busy_s, ch_writer_busy_r) = chan<u1>;
        let (ch_writer_done_s, ch_writer_done_r) = chan<u1>;
        let (ch_reader_start_s, ch_reader_start_r) = chan<u1>;
        let (ch_reader_configuration_s, ch_reader_configuration_r) =
            chan<MainCtrlBundle<TEST_ADDR_W>>;
        let (ch_reader_busy_s, ch_reader_busy_r) = chan<u1>;
        let (ch_reader_done_s, ch_reader_done_r) = chan<u1>;

        let (reader_sync_req_s, reader_sync_req_r) = chan<()>;
        let (reader_sync_rsp_s, reader_sync_rsp_r) = chan<()>;
        let (writer_sync_req_s, writer_sync_req_r) = chan<()>;
        let (writer_sync_rsp_s, writer_sync_rsp_r) = chan<()>;

        spawn Csr<TEST_ADDR_W, TEST_DATA_W, TEST_REGS_N>(
            read_req_r, read_resp_s, write_req_r, write_resp_s, ch_writer_start_s,
            ch_writer_configuration_s, ch_writer_busy_r, ch_writer_done_r, ch_reader_start_s,
            ch_reader_configuration_s, ch_reader_busy_r, ch_reader_done_r, reader_sync_req_r,
            reader_sync_rsp_s, writer_sync_req_r, writer_sync_rsp_s);
        (
            read_req_s, read_resp_r, write_req_s, write_resp_r, ch_writer_start_r,
            ch_writer_configuration_r, ch_writer_busy_s, ch_writer_done_s, ch_reader_start_r,
            ch_reader_configuration_r, ch_reader_busy_s, ch_reader_done_s, reader_sync_req_s,
            reader_sync_rsp_r, writer_sync_req_s, writer_sync_rsp_r, terminator,
        )
    }

    init { () }

    next(tok: token, state: ()) {
        // Write to memory
        // Add 4 to data so that 2 LSB are not used (they reset themselves)
        let tok = for (i, tok): (u32, token) in range(u32:0, TEST_REGS_N) {
            let address = i as uN[TEST_ADDR_W];
            let data = (address + uN[TEST_ADDR_W]:4) as uN[TEST_DATA_W];
            let tok = send(tok, write_req, WriteWordReq(address, data));
            let (tok, _) = recv(tok, write_resp);
            tok
        }(tok);

        // Read from memory
        let tok = for (i, tok): (u32, token) in range(u32:0, TEST_REGS_N) {
            let address = i as uN[TEST_ADDR_W];
            let data = (address + uN[TEST_ADDR_W]:4) as uN[TEST_DATA_W];
            let tok = send(tok, read_req, ReadWordReq(address));
            let (tok, read_data) = recv(tok, read_resp);
            assert_eq(data, read_data.data as uN[TEST_DATA_W]);
            tok
        }(tok);

        let tok = send(tok, terminator, true);
    }
}

#[test_proc]
proc TestLoopMode {
    read_req: chan<ReadReq<TEST_ADDR_W>> out;
    read_resp: chan<ReadResp<TEST_DATA_W>> in;
    write_req: chan<WriteReq<TEST_ADDR_W, TEST_DATA_W>> out;
    write_resp: chan<WriteResp> in;
    ch_writer_start: chan<u1> in;
    ch_writer_configuration: chan<MainCtrlBundle<TEST_ADDR_W>> in;
    ch_writer_busy: chan<u1> out;
    ch_writer_done: chan<u1> out;
    ch_reader_start: chan<u1> in;
    ch_reader_configuration: chan<MainCtrlBundle<TEST_ADDR_W>> in;
    ch_reader_busy: chan<u1> out;
    ch_reader_done: chan<u1> out;
    reader_sync_req: chan<()> out;
    reader_sync_rsp: chan<()> in;
    writer_sync_req: chan<()> out;
    writer_sync_rsp: chan<()> in;
    terminator: chan<bool> out;

    config(terminator: chan<bool> out) {
        let (read_req_s, read_req_r) = chan<ReadReq<TEST_ADDR_W>>;
        let (read_resp_s, read_resp_r) = chan<ReadResp<TEST_DATA_W>>;
        let (write_req_s, write_req_r) = chan<WriteReq<TEST_ADDR_W, TEST_DATA_W>>;
        let (write_resp_s, write_resp_r) = chan<WriteResp>;

        let (ch_writer_start_s, ch_writer_start_r) = chan<u1>;
        let (ch_writer_configuration_s, ch_writer_configuration_r) =
            chan<MainCtrlBundle<TEST_ADDR_W>>;
        let (ch_writer_busy_s, ch_writer_busy_r) = chan<u1>;
        let (ch_writer_done_s, ch_writer_done_r) = chan<u1>;
        let (ch_reader_start_s, ch_reader_start_r) = chan<u1>;
        let (ch_reader_configuration_s, ch_reader_configuration_r) =
            chan<MainCtrlBundle<TEST_ADDR_W>>;
        let (ch_reader_busy_s, ch_reader_busy_r) = chan<u1>;
        let (ch_reader_done_s, ch_reader_done_r) = chan<u1>;

        let (reader_sync_req_s, reader_sync_req_r) = chan<()>;
        let (reader_sync_rsp_s, reader_sync_rsp_r) = chan<()>;
        let (writer_sync_req_s, writer_sync_req_r) = chan<()>;
        let (writer_sync_rsp_s, writer_sync_rsp_r) = chan<()>;

        spawn Csr<TEST_ADDR_W, TEST_DATA_W, TEST_REGS_N>(
            read_req_r, read_resp_s, write_req_r, write_resp_s, ch_writer_start_s,
            ch_writer_configuration_s, ch_writer_busy_r, ch_writer_done_r, ch_reader_start_s,
            ch_reader_configuration_s, ch_reader_busy_r, ch_reader_done_r, reader_sync_req_r,
            reader_sync_rsp_s, writer_sync_req_r, writer_sync_rsp_s);
        (
            read_req_s, read_resp_r, write_req_s, write_resp_r, ch_writer_start_r,
            ch_writer_configuration_r, ch_writer_busy_s, ch_writer_done_s, ch_reader_start_r,
            ch_reader_configuration_r, ch_reader_busy_s, ch_reader_done_s, reader_sync_req_s,
            reader_sync_rsp_r, writer_sync_req_s, writer_sync_rsp_r, terminator,
        )
    }

    init { (u32:0) }

    next(tok: token, state: u32) {
        let CTRL_WORD = uN[TEST_DATA_W]:51;
        // let CTRL_WORD = uN[TEST_DATA_W]:63;

        let state = if state == u32:0 {
            // Prepare configs
            let rw_config = MainCtrlBundle<TEST_ADDR_W> {
                start_address: u32:0x1000,
                line_count: u32:0x64,
                line_length: u32:0x01,
                line_stride: u32:0x00
            };

            let init_csr_values = AdrDatPair[u32:8]:[
                (config::READER_START_ADDRESS, rw_config.start_address),
                (config::READER_LINE_LENGTH, rw_config.line_length),
                (config::READER_LINE_COUNT, rw_config.line_count),
                (config::READER_STRIDE_BETWEEN_LINES, rw_config.line_stride),
                (config::WRITER_START_ADDRESS, rw_config.start_address),
                (config::WRITER_LINE_LENGTH, rw_config.line_length),
                (config::WRITER_LINE_COUNT, rw_config.line_count),
                (config::WRITER_STRIDE_BETWEEN_LINES, rw_config.line_stride),
            ];

            for (i, tok): (u32, token) in u32:0..u32:8 {
                let tok = send(
                    tok, write_req, WriteWordReq((init_csr_values[i]).0, (init_csr_values[i]).1));
                let (tok, _) = recv(tok, write_resp);
                (tok)
            }(tok);

            // Status register should be clear
            let tok = send(tok, read_req, ReadWordReq(config::STATUS_REGISTER));
            let (tok, read_data) = recv(tok, read_resp);
            assert_eq(uN[TEST_ADDR_W]:0, read_data.data as uN[TEST_ADDR_W]);

            // Enable interrupts
            let tok = send(
                tok, write_req, WriteWordReq(config::INTERRUPT_MASK_REGISTER, uN[TEST_DATA_W]:3));
            let (tok, _) = recv(tok, write_resp);

            // Interrupt status register should be clear
            let tok = send(tok, read_req, ReadWordReq(config::INTERRUPT_STATUS_REGISTER));
            let (tok, read_data) = recv(tok, read_resp);
            assert_eq(uN[TEST_ADDR_W]:0, read_data.data as uN[TEST_ADDR_W]);

            let tok = send(tok, write_req, WriteWordReq(config::CONTROL_REGISTER, CTRL_WORD));
            let (tok, _) = recv(tok, write_resp);
            u32:1
        } else {
            state
        };

        if CTRL_WORD[2:4] == u2:0 {
            // Synchronize to external
            let tok = send(tok, reader_sync_req, ());
            let (tok, _) = recv(tok, reader_sync_rsp);

            let tok = send(tok, writer_sync_req, ());
            let (tok, _) = recv(tok, writer_sync_rsp);
        } else {


        };

        // AG receives the config
        let (tok, do_writer_start, do_writer_start_valid) =
            recv_non_blocking(tok, ch_writer_start, u1:0);
        let (tok, do_reader_start, do_reader_start_valid) =
            recv_non_blocking(tok, ch_reader_start, u1:0);

        let state = if do_writer_start_valid && do_writer_start {
            trace_fmt!("-----");
            let (tok, _) = recv(tok, ch_writer_configuration);
            let tok = send(tok, ch_writer_busy, u1:1);
            let tok = send(tok, ch_writer_done, u1:1);

            // Status register should be 1
            let tok = send(tok, read_req, ReadWordReq(config::STATUS_REGISTER));
            let (tok, _) = recv(tok, read_resp);

            let tok = send(tok, read_req, ReadWordReq(config::STATUS_REGISTER));
            let (tok, status) = recv(tok, read_resp);
            assert_eq(uN[TEST_ADDR_W]:1, status.data as uN[TEST_ADDR_W]);

            let tok = send(tok, ch_writer_busy, u1:0);

            // Interrupt status register should be 1
            let tok = send(tok, read_req, ReadWordReq(config::INTERRUPT_STATUS_REGISTER));
            let (tok, irq) = recv(tok, read_resp);
            assert_eq(uN[TEST_ADDR_W]:1, irq.data as uN[TEST_ADDR_W]);

            // Host clears interrupt
            let tok = send(
                tok, write_req, WriteWordReq(config::INTERRUPT_STATUS_REGISTER, uN[TEST_DATA_W]:0));
            let (tok, _) = recv(tok, write_resp);

            state + u32:1
        } else {
            state + u32:10
        };

        let state = if do_reader_start_valid && do_reader_start {
            trace_fmt!("-----");
            let (tok, _) = recv(tok, ch_reader_configuration);

            // AG sets busy
            let tok = send(tok, ch_reader_busy, u1:1);

            // AG sets done
            let tok = send(tok, ch_reader_done, u1:1);

            // Status register should be 2
            let tok = send(tok, read_req, ReadWordReq(config::STATUS_REGISTER));
            let (tok, _) = recv(tok, read_resp);

            let tok = send(tok, read_req, ReadWordReq(config::STATUS_REGISTER));
            let (tok, status) = recv(tok, read_resp);
            assert_eq(uN[TEST_ADDR_W]:2, status.data as uN[TEST_ADDR_W]);

            // AG clears busy status
            let tok = send(tok, ch_reader_busy, u1:0);

            // Interrupt status register should be 2
            let tok = send(tok, read_req, ReadWordReq(config::INTERRUPT_STATUS_REGISTER));
            let (tok, irq) = recv(tok, read_resp);
            assert_eq(uN[TEST_ADDR_W]:2, irq.data as uN[TEST_ADDR_W]);

            // Host clears interrupt
            let tok = send(
                tok, write_req, WriteWordReq(config::INTERRUPT_STATUS_REGISTER, uN[TEST_DATA_W]:0));
            let (tok, _) = recv(tok, write_resp);
            state + u32:1
        } else {
            state + u32:10
        };

        let do_terminate = state > u32:3;
        let tok = send_if(tok, terminator, do_terminate, do_terminate);
        state
    }
}

#[test_proc]
proc TestLoopModeDisableSync {
    read_req: chan<ReadReq<TEST_ADDR_W>> out;
    read_resp: chan<ReadResp<TEST_DATA_W>> in;
    write_req: chan<WriteReq<TEST_ADDR_W, TEST_DATA_W>> out;
    write_resp: chan<WriteResp> in;
    ch_writer_start: chan<u1> in;
    ch_writer_configuration: chan<MainCtrlBundle<TEST_ADDR_W>> in;
    ch_writer_busy: chan<u1> out;
    ch_writer_done: chan<u1> out;
    ch_reader_start: chan<u1> in;
    ch_reader_configuration: chan<MainCtrlBundle<TEST_ADDR_W>> in;
    ch_reader_busy: chan<u1> out;
    ch_reader_done: chan<u1> out;
    reader_sync_req: chan<()> out;
    reader_sync_rsp: chan<()> in;
    writer_sync_req: chan<()> out;
    writer_sync_rsp: chan<()> in;
    terminator: chan<bool> out;

    config(terminator: chan<bool> out) {
        let (read_req_s, read_req_r) = chan<ReadReq<TEST_ADDR_W>>;
        let (read_resp_s, read_resp_r) = chan<ReadResp<TEST_DATA_W>>;
        let (write_req_s, write_req_r) = chan<WriteReq<TEST_ADDR_W, TEST_DATA_W>>;
        let (write_resp_s, write_resp_r) = chan<WriteResp>;

        let (ch_writer_start_s, ch_writer_start_r) = chan<u1>;
        let (ch_writer_configuration_s, ch_writer_configuration_r) =
            chan<MainCtrlBundle<TEST_ADDR_W>>;
        let (ch_writer_busy_s, ch_writer_busy_r) = chan<u1>;
        let (ch_writer_done_s, ch_writer_done_r) = chan<u1>;
        let (ch_reader_start_s, ch_reader_start_r) = chan<u1>;
        let (ch_reader_configuration_s, ch_reader_configuration_r) =
            chan<MainCtrlBundle<TEST_ADDR_W>>;
        let (ch_reader_busy_s, ch_reader_busy_r) = chan<u1>;
        let (ch_reader_done_s, ch_reader_done_r) = chan<u1>;

        let (reader_sync_req_s, reader_sync_req_r) = chan<()>;
        let (reader_sync_rsp_s, reader_sync_rsp_r) = chan<()>;
        let (writer_sync_req_s, writer_sync_req_r) = chan<()>;
        let (writer_sync_rsp_s, writer_sync_rsp_r) = chan<()>;

        spawn Csr<TEST_ADDR_W, TEST_DATA_W, TEST_REGS_N>(
            read_req_r, read_resp_s, write_req_r, write_resp_s, ch_writer_start_s,
            ch_writer_configuration_s, ch_writer_busy_r, ch_writer_done_r, ch_reader_start_s,
            ch_reader_configuration_s, ch_reader_busy_r, ch_reader_done_r, reader_sync_req_r,
            reader_sync_rsp_s, writer_sync_req_r, writer_sync_rsp_s);
        (
            read_req_s, read_resp_r, write_req_s, write_resp_r, ch_writer_start_r,
            ch_writer_configuration_r, ch_writer_busy_s, ch_writer_done_s, ch_reader_start_r,
            ch_reader_configuration_r, ch_reader_busy_s, ch_reader_done_s, reader_sync_req_s,
            reader_sync_rsp_r, writer_sync_req_s, writer_sync_rsp_r, terminator,
        )
    }

    init { (u32:0) }

    next(tok: token, state: u32) {
        // let CTRL_WORD = uN[TEST_DATA_W]:51;
        let CTRL_WORD = uN[TEST_DATA_W]:63;

        let state = if state == u32:0 {
            // Prepare configs
            let rw_config = MainCtrlBundle<TEST_ADDR_W> {
                start_address: u32:0x1000,
                line_count: u32:0x64,
                line_length: u32:0x01,
                line_stride: u32:0x00
            };

            let init_csr_values = AdrDatPair[u32:8]:[
                (config::READER_START_ADDRESS, rw_config.start_address),
                (config::READER_LINE_LENGTH, rw_config.line_length),
                (config::READER_LINE_COUNT, rw_config.line_count),
                (config::READER_STRIDE_BETWEEN_LINES, rw_config.line_stride),
                (config::WRITER_START_ADDRESS, rw_config.start_address),
                (config::WRITER_LINE_LENGTH, rw_config.line_length),
                (config::WRITER_LINE_COUNT, rw_config.line_count),
                (config::WRITER_STRIDE_BETWEEN_LINES, rw_config.line_stride),
            ];

            for (i, tok): (u32, token) in u32:0..u32:8 {
                let tok = send(
                    tok, write_req, WriteWordReq((init_csr_values[i]).0, (init_csr_values[i]).1));
                let (tok, _) = recv(tok, write_resp);
                (tok)
            }(tok);

            // Status register should be clear
            let tok = send(tok, read_req, ReadWordReq(config::STATUS_REGISTER));
            let (tok, read_data) = recv(tok, read_resp);
            assert_eq(uN[TEST_ADDR_W]:0, read_data.data as uN[TEST_ADDR_W]);

            // Enable interrupts
            let tok = send(
                tok, write_req, WriteWordReq(config::INTERRUPT_MASK_REGISTER, uN[TEST_DATA_W]:3));
            let (tok, _) = recv(tok, write_resp);

            // Interrupt status register should be clear
            let tok = send(tok, read_req, ReadWordReq(config::INTERRUPT_STATUS_REGISTER));
            let (tok, read_data) = recv(tok, read_resp);
            assert_eq(uN[TEST_ADDR_W]:0, read_data.data as uN[TEST_ADDR_W]);

            let tok = send(tok, write_req, WriteWordReq(config::CONTROL_REGISTER, CTRL_WORD));
            let (tok, _) = recv(tok, write_resp);
            u32:1
        } else {
            state
        };

        if CTRL_WORD[2:4] == u2:0 {
            // Synchronize to external
            let tok = send(tok, reader_sync_req, ());
            let (tok, _) = recv(tok, reader_sync_rsp);

            let tok = send(tok, writer_sync_req, ());
            let (tok, _) = recv(tok, writer_sync_rsp);
        } else {


        };

        // AG receives the config
        let (tok, do_writer_start, do_writer_start_valid) =
            recv_non_blocking(tok, ch_writer_start, u1:0);
        let (tok, do_reader_start, do_reader_start_valid) =
            recv_non_blocking(tok, ch_reader_start, u1:0);

        let state = if do_writer_start_valid && do_writer_start {
            trace_fmt!("-----");
            let (tok, _) = recv(tok, ch_writer_configuration);
            let tok = send(tok, ch_writer_busy, u1:1);
            let tok = send(tok, ch_writer_done, u1:1);

            // Status register should be 1
            let tok = send(tok, read_req, ReadWordReq(config::STATUS_REGISTER));
            let (tok, _) = recv(tok, read_resp);

            let tok = send(tok, read_req, ReadWordReq(config::STATUS_REGISTER));
            let (tok, status) = recv(tok, read_resp);
            assert_eq(uN[TEST_ADDR_W]:1, status.data as uN[TEST_ADDR_W]);

            let tok = send(tok, ch_writer_busy, u1:0);

            // Interrupt status register should be 1
            let tok = send(tok, read_req, ReadWordReq(config::INTERRUPT_STATUS_REGISTER));
            let (tok, irq) = recv(tok, read_resp);
            assert_eq(uN[TEST_ADDR_W]:1, irq.data as uN[TEST_ADDR_W]);

            // Host clears interrupt
            let tok = send(
                tok, write_req, WriteWordReq(config::INTERRUPT_STATUS_REGISTER, uN[TEST_DATA_W]:0));
            let (tok, _) = recv(tok, write_resp);

            state + u32:1
        } else {
            state + u32:10
        };

        let state = if do_reader_start_valid && do_reader_start {
            trace_fmt!("-----");
            let (tok, _) = recv(tok, ch_reader_configuration);

            // AG sets busy
            let tok = send(tok, ch_reader_busy, u1:1);

            // AG sets done
            let tok = send(tok, ch_reader_done, u1:1);

            // Status register should be 2
            let tok = send(tok, read_req, ReadWordReq(config::STATUS_REGISTER));
            let (tok, _) = recv(tok, read_resp);

            let tok = send(tok, read_req, ReadWordReq(config::STATUS_REGISTER));
            let (tok, status) = recv(tok, read_resp);
            assert_eq(uN[TEST_ADDR_W]:2, status.data as uN[TEST_ADDR_W]);

            // AG clears busy status
            let tok = send(tok, ch_reader_busy, u1:0);

            // Interrupt status register should be 2
            let tok = send(tok, read_req, ReadWordReq(config::INTERRUPT_STATUS_REGISTER));
            let (tok, irq) = recv(tok, read_resp);
            assert_eq(uN[TEST_ADDR_W]:2, irq.data as uN[TEST_ADDR_W]);

            // Host clears interrupt
            let tok = send(
                tok, write_req, WriteWordReq(config::INTERRUPT_STATUS_REGISTER, uN[TEST_DATA_W]:0));
            let (tok, _) = recv(tok, write_resp);
            state + u32:1
        } else {
            state + u32:10
        };

        let do_terminate = state > u32:3;
        let tok = send_if(tok, terminator, do_terminate, do_terminate);
        state
    }
}

#[test_proc]
proc TestSingleMode {
    read_req: chan<ReadReq<TEST_ADDR_W>> out;
    read_resp: chan<ReadResp<TEST_DATA_W>> in;
    write_req: chan<WriteReq<TEST_ADDR_W, TEST_DATA_W>> out;
    write_resp: chan<WriteResp> in;
    ch_writer_start: chan<u1> in;
    ch_writer_configuration: chan<MainCtrlBundle<TEST_ADDR_W>> in;
    ch_writer_busy: chan<u1> out;
    ch_writer_done: chan<u1> out;
    ch_reader_start: chan<u1> in;
    ch_reader_configuration: chan<MainCtrlBundle<TEST_ADDR_W>> in;
    ch_reader_busy: chan<u1> out;
    ch_reader_done: chan<u1> out;
    reader_sync_req: chan<()> out;
    reader_sync_rsp: chan<()> in;
    writer_sync_req: chan<()> out;
    writer_sync_rsp: chan<()> in;
    terminator: chan<bool> out;

    config(terminator: chan<bool> out) {
        let (read_req_s, read_req_r) = chan<ReadReq<TEST_ADDR_W>>;
        let (read_resp_s, read_resp_r) = chan<ReadResp<TEST_DATA_W>>;
        let (write_req_s, write_req_r) = chan<WriteReq<TEST_ADDR_W, TEST_DATA_W>>;
        let (write_resp_s, write_resp_r) = chan<WriteResp>;

        let (ch_writer_start_s, ch_writer_start_r) = chan<u1>;
        let (ch_writer_configuration_s, ch_writer_configuration_r) =
            chan<MainCtrlBundle<TEST_ADDR_W>>;
        let (ch_writer_busy_s, ch_writer_busy_r) = chan<u1>;
        let (ch_writer_done_s, ch_writer_done_r) = chan<u1>;
        let (ch_reader_start_s, ch_reader_start_r) = chan<u1>;
        let (ch_reader_configuration_s, ch_reader_configuration_r) =
            chan<MainCtrlBundle<TEST_ADDR_W>>;
        let (ch_reader_busy_s, ch_reader_busy_r) = chan<u1>;
        let (ch_reader_done_s, ch_reader_done_r) = chan<u1>;

        let (reader_sync_req_s, reader_sync_req_r) = chan<()>;
        let (reader_sync_rsp_s, reader_sync_rsp_r) = chan<()>;
        let (writer_sync_req_s, writer_sync_req_r) = chan<()>;
        let (writer_sync_rsp_s, writer_sync_rsp_r) = chan<()>;

        spawn Csr<TEST_ADDR_W, TEST_DATA_W, TEST_REGS_N>(
            read_req_r, read_resp_s, write_req_r, write_resp_s, ch_writer_start_s,
            ch_writer_configuration_s, ch_writer_busy_r, ch_writer_done_r, ch_reader_start_s,
            ch_reader_configuration_s, ch_reader_busy_r, ch_reader_done_r, reader_sync_req_r,
            reader_sync_rsp_s, writer_sync_req_r, writer_sync_rsp_s);
        (
            read_req_s, read_resp_r, write_req_s, write_resp_r, ch_writer_start_r,
            ch_writer_configuration_r, ch_writer_busy_s, ch_writer_done_s, ch_reader_start_r,
            ch_reader_configuration_r, ch_reader_busy_s, ch_reader_done_s, reader_sync_req_s,
            reader_sync_rsp_r, writer_sync_req_s, writer_sync_rsp_r, terminator,
        )
    }

    init { (u32:0) }

    next(tok: token, state: u32) {
        let CTRL_WORD = uN[TEST_DATA_W]:3;
        // let CTRL_WORD = uN[TEST_DATA_W]:15;

        let state = if state == u32:0 {
            // Prepare configs
            let rw_config = MainCtrlBundle<TEST_ADDR_W> {
                start_address: u32:0x1000,
                line_count: u32:0x64,
                line_length: u32:0x01,
                line_stride: u32:0x00
            };

            let init_csr_values = AdrDatPair[u32:8]:[
                (config::READER_START_ADDRESS, rw_config.start_address),
                (config::READER_LINE_LENGTH, rw_config.line_length),
                (config::READER_LINE_COUNT, rw_config.line_count),
                (config::READER_STRIDE_BETWEEN_LINES, rw_config.line_stride),
                (config::WRITER_START_ADDRESS, rw_config.start_address),
                (config::WRITER_LINE_LENGTH, rw_config.line_length),
                (config::WRITER_LINE_COUNT, rw_config.line_count),
                (config::WRITER_STRIDE_BETWEEN_LINES, rw_config.line_stride),
            ];

            for (i, tok): (u32, token) in u32:0..u32:8 {
                let tok = send(
                    tok, write_req, WriteWordReq((init_csr_values[i]).0, (init_csr_values[i]).1));
                let (tok, _) = recv(tok, write_resp);
                (tok)
            }(tok);

            // Status register should be clear
            let tok = send(tok, read_req, ReadWordReq(config::STATUS_REGISTER));
            let (tok, read_data) = recv(tok, read_resp);
            assert_eq(uN[TEST_ADDR_W]:0, read_data.data as uN[TEST_ADDR_W]);

            // Enable interrupts
            let tok = send(
                tok, write_req, WriteWordReq(config::INTERRUPT_MASK_REGISTER, uN[TEST_DATA_W]:3));
            let (tok, _) = recv(tok, write_resp);

            // Interrupt status register should be clear
            let tok = send(tok, read_req, ReadWordReq(config::INTERRUPT_STATUS_REGISTER));
            let (tok, read_data) = recv(tok, read_resp);
            assert_eq(uN[TEST_ADDR_W]:0, read_data.data as uN[TEST_ADDR_W]);

            let tok = send(tok, write_req, WriteWordReq(config::CONTROL_REGISTER, CTRL_WORD));
            let (tok, _) = recv(tok, write_resp);

            if CTRL_WORD[2:4] == u2:0 {
                trace_fmt!("SYNC TO EXTERNAL");
                // Synchronize to external
                let tok = send(tok, reader_sync_req, ());
                let (tok, _) = recv(tok, reader_sync_rsp);

                let tok = send(tok, writer_sync_req, ());
                let (tok, _) = recv(tok, writer_sync_rsp);
            } else {


            };

            u32:1
        } else {
            state
        };

        // AG receives the config
        let (tok, do_writer_start, do_writer_start_valid) =
            recv_non_blocking(tok, ch_writer_start, u1:0);
        let (tok, do_reader_start, do_reader_start_valid) =
            recv_non_blocking(tok, ch_reader_start, u1:0);

        let state = if do_writer_start_valid && do_writer_start {
            trace_fmt!("-----");
            let (tok, _) = recv(tok, ch_writer_configuration);
            let tok = send(tok, ch_writer_busy, u1:1);
            let tok = send(tok, ch_writer_done, u1:1);

            // Status register should be 1
            let tok = send(tok, read_req, ReadWordReq(config::STATUS_REGISTER));
            let (tok, _) = recv(tok, read_resp);

            let tok = send(tok, read_req, ReadWordReq(config::STATUS_REGISTER));
            let (tok, status) = recv(tok, read_resp);
            assert_eq(uN[TEST_ADDR_W]:1, status.data as uN[TEST_ADDR_W]);

            let tok = send(tok, ch_writer_busy, u1:0);

            // Interrupt status register should be 1
            let tok = send(tok, read_req, ReadWordReq(config::INTERRUPT_STATUS_REGISTER));
            let (tok, irq) = recv(tok, read_resp);
            assert_eq(uN[TEST_ADDR_W]:1, irq.data as uN[TEST_ADDR_W]);

            // Host clears interrupt
            let tok = send(
                tok, write_req, WriteWordReq(config::INTERRUPT_STATUS_REGISTER, uN[TEST_DATA_W]:0));
            let (tok, _) = recv(tok, write_resp);

            state + u32:1
        } else {
            state + u32:10
        };

        let state = if do_reader_start_valid && do_reader_start {
            trace_fmt!("-----");
            let (tok, _) = recv(tok, ch_reader_configuration);

            // AG sets busy
            let tok = send(tok, ch_reader_busy, u1:1);

            // AG sets done
            let tok = send(tok, ch_reader_done, u1:1);

            // Status register should be 2
            let tok = send(tok, read_req, ReadWordReq(config::STATUS_REGISTER));
            let (tok, _) = recv(tok, read_resp);

            let tok = send(tok, read_req, ReadWordReq(config::STATUS_REGISTER));
            let (tok, status) = recv(tok, read_resp);
            assert_eq(uN[TEST_ADDR_W]:2, status.data as uN[TEST_ADDR_W]);

            // AG clears busy status
            let tok = send(tok, ch_reader_busy, u1:0);

            // Interrupt status register should be 2
            let tok = send(tok, read_req, ReadWordReq(config::INTERRUPT_STATUS_REGISTER));
            let (tok, irq) = recv(tok, read_resp);
            assert_eq(uN[TEST_ADDR_W]:2, irq.data as uN[TEST_ADDR_W]);

            // Host clears interrupt
            let tok = send(
                tok, write_req, WriteWordReq(config::INTERRUPT_STATUS_REGISTER, uN[TEST_DATA_W]:0));
            let (tok, _) = recv(tok, write_resp);
            state + u32:1
        } else {
            state + u32:10
        };

        let do_terminate = state == u32:1203;
        trace_fmt!("state = {}", state);
        let tok = send_if(tok, terminator, do_terminate, do_terminate);
        state
    }
}

#[test_proc]
proc TestSingleModeDisableSync {
    read_req: chan<ReadReq<TEST_ADDR_W>> out;
    read_resp: chan<ReadResp<TEST_DATA_W>> in;
    write_req: chan<WriteReq<TEST_ADDR_W, TEST_DATA_W>> out;
    write_resp: chan<WriteResp> in;
    ch_writer_start: chan<u1> in;
    ch_writer_configuration: chan<MainCtrlBundle<TEST_ADDR_W>> in;
    ch_writer_busy: chan<u1> out;
    ch_writer_done: chan<u1> out;
    ch_reader_start: chan<u1> in;
    ch_reader_configuration: chan<MainCtrlBundle<TEST_ADDR_W>> in;
    ch_reader_busy: chan<u1> out;
    ch_reader_done: chan<u1> out;
    reader_sync_req: chan<()> out;
    reader_sync_rsp: chan<()> in;
    writer_sync_req: chan<()> out;
    writer_sync_rsp: chan<()> in;
    terminator: chan<bool> out;

    config(terminator: chan<bool> out) {
        let (read_req_s, read_req_r) = chan<ReadReq<TEST_ADDR_W>>;
        let (read_resp_s, read_resp_r) = chan<ReadResp<TEST_DATA_W>>;
        let (write_req_s, write_req_r) = chan<WriteReq<TEST_ADDR_W, TEST_DATA_W>>;
        let (write_resp_s, write_resp_r) = chan<WriteResp>;

        let (ch_writer_start_s, ch_writer_start_r) = chan<u1>;
        let (ch_writer_configuration_s, ch_writer_configuration_r) =
            chan<MainCtrlBundle<TEST_ADDR_W>>;
        let (ch_writer_busy_s, ch_writer_busy_r) = chan<u1>;
        let (ch_writer_done_s, ch_writer_done_r) = chan<u1>;
        let (ch_reader_start_s, ch_reader_start_r) = chan<u1>;
        let (ch_reader_configuration_s, ch_reader_configuration_r) =
            chan<MainCtrlBundle<TEST_ADDR_W>>;
        let (ch_reader_busy_s, ch_reader_busy_r) = chan<u1>;
        let (ch_reader_done_s, ch_reader_done_r) = chan<u1>;

        let (reader_sync_req_s, reader_sync_req_r) = chan<()>;
        let (reader_sync_rsp_s, reader_sync_rsp_r) = chan<()>;
        let (writer_sync_req_s, writer_sync_req_r) = chan<()>;
        let (writer_sync_rsp_s, writer_sync_rsp_r) = chan<()>;

        spawn Csr<TEST_ADDR_W, TEST_DATA_W, TEST_REGS_N>(
            read_req_r, read_resp_s, write_req_r, write_resp_s, ch_writer_start_s,
            ch_writer_configuration_s, ch_writer_busy_r, ch_writer_done_r, ch_reader_start_s,
            ch_reader_configuration_s, ch_reader_busy_r, ch_reader_done_r, reader_sync_req_r,
            reader_sync_rsp_s, writer_sync_req_r, writer_sync_rsp_s);
        (
            read_req_s, read_resp_r, write_req_s, write_resp_r, ch_writer_start_r,
            ch_writer_configuration_r, ch_writer_busy_s, ch_writer_done_s, ch_reader_start_r,
            ch_reader_configuration_r, ch_reader_busy_s, ch_reader_done_s, reader_sync_req_s,
            reader_sync_rsp_r, writer_sync_req_s, writer_sync_rsp_r, terminator,
        )
    }

    init { (u32:0) }

    next(tok: token, state: u32) {
        // let CTRL_WORD = uN[TEST_DATA_W]:3;
        let CTRL_WORD = uN[TEST_DATA_W]:15;

        let state = if state == u32:0 {
            // Prepare configs
            let rw_config = MainCtrlBundle<TEST_ADDR_W> {
                start_address: u32:0x1000,
                line_count: u32:0x64,
                line_length: u32:0x01,
                line_stride: u32:0x00
            };

            let init_csr_values = AdrDatPair[u32:8]:[
                (config::READER_START_ADDRESS, rw_config.start_address),
                (config::READER_LINE_LENGTH, rw_config.line_length),
                (config::READER_LINE_COUNT, rw_config.line_count),
                (config::READER_STRIDE_BETWEEN_LINES, rw_config.line_stride),
                (config::WRITER_START_ADDRESS, rw_config.start_address),
                (config::WRITER_LINE_LENGTH, rw_config.line_length),
                (config::WRITER_LINE_COUNT, rw_config.line_count),
                (config::WRITER_STRIDE_BETWEEN_LINES, rw_config.line_stride),
            ];

            for (i, tok): (u32, token) in u32:0..u32:8 {
                let tok = send(
                    tok, write_req, WriteWordReq((init_csr_values[i]).0, (init_csr_values[i]).1));
                let (tok, _) = recv(tok, write_resp);
                (tok)
            }(tok);

            // Status register should be clear
            let tok = send(tok, read_req, ReadWordReq(config::STATUS_REGISTER));
            let (tok, read_data) = recv(tok, read_resp);
            assert_eq(uN[TEST_ADDR_W]:0, read_data.data as uN[TEST_ADDR_W]);

            // Enable interrupts
            let tok = send(
                tok, write_req, WriteWordReq(config::INTERRUPT_MASK_REGISTER, uN[TEST_DATA_W]:3));
            let (tok, _) = recv(tok, write_resp);

            // Interrupt status register should be clear
            let tok = send(tok, read_req, ReadWordReq(config::INTERRUPT_STATUS_REGISTER));
            let (tok, read_data) = recv(tok, read_resp);
            assert_eq(uN[TEST_ADDR_W]:0, read_data.data as uN[TEST_ADDR_W]);

            let tok = send(tok, write_req, WriteWordReq(config::CONTROL_REGISTER, CTRL_WORD));
            let (tok, _) = recv(tok, write_resp);

            if CTRL_WORD[2:4] == u2:0 {
                trace_fmt!("SYNC TO EXTERNAL");
                // Synchronize to external
                let tok = send(tok, reader_sync_req, ());
                let (tok, _) = recv(tok, reader_sync_rsp);

                let tok = send(tok, writer_sync_req, ());
                let (tok, _) = recv(tok, writer_sync_rsp);
            } else {


            };

            u32:1
        } else {
            state
        };

        // AG receives the config
        let (tok, do_writer_start, do_writer_start_valid) =
            recv_non_blocking(tok, ch_writer_start, u1:0);
        let (tok, do_reader_start, do_reader_start_valid) =
            recv_non_blocking(tok, ch_reader_start, u1:0);

        let state = if do_writer_start_valid && do_writer_start {
            trace_fmt!("-----");
            let (tok, _) = recv(tok, ch_writer_configuration);
            let tok = send(tok, ch_writer_busy, u1:1);
            let tok = send(tok, ch_writer_done, u1:1);

            // Status register should be 1
            let tok = send(tok, read_req, ReadWordReq(config::STATUS_REGISTER));
            let (tok, _) = recv(tok, read_resp);

            let tok = send(tok, read_req, ReadWordReq(config::STATUS_REGISTER));
            let (tok, status) = recv(tok, read_resp);
            assert_eq(uN[TEST_ADDR_W]:1, status.data as uN[TEST_ADDR_W]);

            let tok = send(tok, ch_writer_busy, u1:0);

            // Interrupt status register should be 1
            let tok = send(tok, read_req, ReadWordReq(config::INTERRUPT_STATUS_REGISTER));
            let (tok, irq) = recv(tok, read_resp);
            assert_eq(uN[TEST_ADDR_W]:1, irq.data as uN[TEST_ADDR_W]);

            // Host clears interrupt
            let tok = send(
                tok, write_req, WriteWordReq(config::INTERRUPT_STATUS_REGISTER, uN[TEST_DATA_W]:0));
            let (tok, _) = recv(tok, write_resp);

            state + u32:1
        } else {
            state + u32:10
        };

        let state = if do_reader_start_valid && do_reader_start {
            trace_fmt!("-----");
            let (tok, _) = recv(tok, ch_reader_configuration);

            // AG sets busy
            let tok = send(tok, ch_reader_busy, u1:1);

            // AG sets done
            let tok = send(tok, ch_reader_done, u1:1);

            // Status register should be 2
            let tok = send(tok, read_req, ReadWordReq(config::STATUS_REGISTER));
            let (tok, _) = recv(tok, read_resp);

            let tok = send(tok, read_req, ReadWordReq(config::STATUS_REGISTER));
            let (tok, status) = recv(tok, read_resp);
            assert_eq(uN[TEST_ADDR_W]:2, status.data as uN[TEST_ADDR_W]);

            // AG clears busy status
            let tok = send(tok, ch_reader_busy, u1:0);

            // Interrupt status register should be 2
            let tok = send(tok, read_req, ReadWordReq(config::INTERRUPT_STATUS_REGISTER));
            let (tok, irq) = recv(tok, read_resp);
            assert_eq(uN[TEST_ADDR_W]:2, irq.data as uN[TEST_ADDR_W]);

            // Host clears interrupt
            let tok = send(
                tok, write_req, WriteWordReq(config::INTERRUPT_STATUS_REGISTER, uN[TEST_DATA_W]:0));
            let (tok, _) = recv(tok, write_resp);
            state + u32:1
        } else {
            state + u32:10
        };

        let do_terminate = state == u32:1203;
        trace_fmt!("state = {}", state);
        let tok = send_if(tok, terminator, do_terminate, do_terminate);
        state
    }
}

// FIXME: Verilog generation
// Verilog example
proc csr {
    config(read_req: chan<ReadReq<u32:8>> in, read_resp: chan<ReadResp<u32:32>> out,
           write_req: chan<WriteReq<u32:8, u32:32>> in, write_resp: chan<WriteResp> out,
           ch_writer_start: chan<u1> out, ch_writer_configuration: chan<MainCtrlBundle<u32:8>> out,
           ch_writer_busy: chan<u1> in, ch_writer_done: chan<u1> in, ch_reader_start: chan<u1> out,
           ch_reader_configuration: chan<MainCtrlBundle<u32:8>> out, ch_reader_busy: chan<u1> in,
           ch_reader_done: chan<u1> in, reader_sync_req: chan<()> in, reader_sync_rsp: chan<()> out,
           writer_sync_req: chan<()> in, writer_sync_rsp: chan<()> out) {

        spawn Csr<u32:8, u32:32, u32:14>(
            read_req, read_resp, write_req, write_resp, ch_writer_start, ch_writer_configuration,
            ch_writer_busy, ch_writer_done, ch_reader_start, ch_reader_configuration,
            ch_reader_busy, ch_reader_done, reader_sync_req, reader_sync_rsp, writer_sync_req,
            writer_sync_rsp);
        ()
    }

    init { () }

    next(tok: token, state: ()) {  }
}
