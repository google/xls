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

// FIFO

import std;
import xls.examples.ram;
import xls.modules.dma.config;
import xls.modules.dma.gpf;
import xls.modules.dma.bus.axi_st_pkg;

type AxiStreamBundle = axi_st_pkg::AxiStreamBundle;

struct ReaderState<ADDR_W: u32> { ptr_r: uN[ADDR_W], is_empty: u1, ptr_w: uN[ADDR_W] }

struct WriterState<ADDR_W: u32> { ptr_w: uN[ADDR_W], is_full: u1, ptr_r: uN[ADDR_W] }

proc FifoRAM<DATA_W: u32, ADDR_W: u32, FIFO_LENGTH: u32> {
    req_chan0: chan<ram::RWRamReq<ADDR_W, DATA_W, u32:0>> in;
    resp_chan0: chan<ram::RWRamResp<DATA_W>> out;
    req_chan1: chan<ram::RWRamReq<ADDR_W, DATA_W, u32:0>> in;
    wr_comp_chan1: chan<()> out;
    wr_comp_chan0: chan<()> in;
    resp_chan1: chan<ram::RWRamResp<DATA_W>> in;

    config(req_chan0: chan<ram::RWRamReq<ADDR_W, DATA_W, u32:0>> in,
           resp_chan0: chan<ram::RWRamResp<DATA_W>> out,
           req_chan1: chan<ram::RWRamReq<ADDR_W, DATA_W, u32:0>> in, wr_comp_chan1: chan<()> out) {

        let (wr_comp_chan0_s, wr_comp_chan0_r) = chan<()>;
        let (resp_chan1_s, resp_chan1_r) = chan<ram::RWRamResp<DATA_W>>;

        spawn ram::RamModel2RW<DATA_W, FIFO_LENGTH>(
            req_chan0, resp_chan0, wr_comp_chan0_s, req_chan1, resp_chan1_s, wr_comp_chan1);

        (req_chan0, resp_chan0, req_chan1, wr_comp_chan1, wr_comp_chan0_r, resp_chan1_r)
    }

    init { () }

    next(tok: token, state: ()) {
        let (tok, _, _) = recv_non_blocking(tok, wr_comp_chan0, ());
        let zero_rw_ram_resp = ram::RWRamResp { data: uN[DATA_W]:0 };
        let (tok, _, _) = recv_non_blocking(tok, resp_chan1, zero_rw_ram_resp);
    }
}

proc Reader<DATA_W: u32, DATA_W_DIV8: u32, ID_W: u32, DEST_W: u32, ADDR_W: u32, FIFO_LENGTH: u32> {
    ch_read: chan<AxiStreamBundle<DATA_W, DATA_W_DIV8, ID_W, DEST_W>> out;
    ch_mem_read_req: chan<ram::RWRamReq<ADDR_W, DATA_W, u32:0>> out;
    ch_mem_read_rsp: chan<ram::RWRamResp<DATA_W>> in;
    ch_ptr_r: chan<ReaderState<ADDR_W>> out;
    ch_ptr_w: chan<WriterState<ADDR_W>> in;

    config(ch_read: chan<AxiStreamBundle<DATA_W, DATA_W_DIV8, ID_W, DEST_W>> out,
           ch_mem_read_req: chan<ram::RWRamReq<ADDR_W, DATA_W, u32:0>> out,
           ch_mem_read_rsp: chan<ram::RWRamResp<DATA_W>> in,
           ch_ptr_r: chan<ReaderState<ADDR_W>> out, ch_ptr_w: chan<WriterState<ADDR_W>> in) {
        (ch_read, ch_mem_read_req, ch_mem_read_rsp, ch_ptr_r, ch_ptr_w)
    }

    init { (ReaderState<ADDR_W> { ptr_r: uN[ADDR_W]:0, is_empty: u1:1, ptr_w: uN[ADDR_W]:0 }) }

    next(tok: token, state: ReaderState<ADDR_W>) {
        trace_fmt!("Reader State = {}", state);

        // Obtain writer state
        let tok = send(tok, ch_ptr_r, state);
        let (tok, writer_state, update_ptr_w) =
            recv_non_blocking(tok, ch_ptr_w, zero!<WriterState>());

        let ptr_w = if update_ptr_w { writer_state.ptr_w } else { state.ptr_w };

        // Check state
        // let is_almost_empty = (state.ptr_r + uN[ADDR_W]:1) == ptr_w;
        let is_empty = (state.ptr_r == ptr_w);  //|| is_almost_empty;

        // Fetch data from RAM
        let tok = send_if(
            tok, ch_mem_read_req, !state.is_empty,
            ram::RWRamReq {
                addr: state.ptr_r,
                data: uN[DATA_W]:0,
                write_mask: (),
                read_mask: (),
                we: false,
                re: true
            });

        let zero_rw_ram_resp = ram::RWRamResp { data: uN[DATA_W]:0 };
        let (tok, read_data) = recv_if(tok, ch_mem_read_rsp, !state.is_empty, zero_rw_ram_resp);

        // Push data into stream interface
        let tok = send_if(
            tok, ch_read, !state.is_empty,
            AxiStreamBundle {
                tdata: read_data.data,
                tstr: uN[DATA_W_DIV8]:0,
                tkeep: uN[DATA_W_DIV8]:0,
                tlast: u1:1,
                tid: uN[ID_W]:0,
                tdest: uN[DEST_W]:0
            });

        let ptr_r = if !state.is_empty { state.ptr_r + uN[ADDR_W]:1 } else { state.ptr_r };

        ReaderState { ptr_r, is_empty, ptr_w }
    }
}

proc Writer<DATA_W: u32, DATA_W_DIV8: u32, ID_W: u32, DEST_W: u32, ADDR_W: u32, FIFO_LENGTH: u32> {
    ch_write: chan<AxiStreamBundle<DATA_W, DATA_W_DIV8, ID_W, DEST_W>> in;
    ch_mem_write_req: chan<ram::RWRamReq<ADDR_W, DATA_W, u32:0>> out;
    ch_mem_write_rsp: chan<()> in;
    ch_ptr_r: chan<ReaderState<ADDR_W>> in;
    ch_ptr_w: chan<WriterState<ADDR_W>> out;

    config(ch_write: chan<AxiStreamBundle<DATA_W, DATA_W_DIV8, ID_W, DEST_W>> in,
           ch_mem_write_req: chan<ram::RWRamReq<ADDR_W, DATA_W, u32:0>> out,
           ch_mem_write_rsp: chan<()> in, ch_ptr_r: chan<ReaderState<ADDR_W>> in,
           ch_ptr_w: chan<WriterState<ADDR_W>> out) {
        (ch_write, ch_mem_write_req, ch_mem_write_rsp, ch_ptr_r, ch_ptr_w)
    }

    init { (WriterState<ADDR_W> { ptr_w: uN[ADDR_W]:0, is_full: u1:0, ptr_r: uN[ADDR_W]:0 }) }

    next(tok: token, state: WriterState<ADDR_W>) {
        trace_fmt!("Writer state = {}", state);

        // Obtain reader state
        let tok = send(tok, ch_ptr_w, state);
        let (tok, reader_state, update_ptr_r) =
            recv_non_blocking(tok, ch_ptr_r, zero!<ReaderState>());
        let ptr_r = if update_ptr_r { reader_state.ptr_r } else { state.ptr_r };

        // Check state
        let is_full = (state.ptr_w + uN[ADDR_W]:1) == ptr_r;

        // Check for write requests
        let zero_write = AxiStreamBundle {
            tdata: uN[DATA_W]:0,
            tstr: uN[DATA_W_DIV8]:0,
            tkeep: uN[DATA_W_DIV8]:0,
            tlast: u1:0,
            tid: uN[ID_W]:0,
            tdest: uN[DEST_W]:0
        };
        let (tok, write_data) = recv_if(tok, ch_write, !is_full, zero_write);

        // Send write req to RAM
        let ram_req = ram::RWRamReq {
            addr: state.ptr_w,
            data: write_data.tdata,
            write_mask: (),
            read_mask: (),
            we: true,
            re: false
        };
        let tok = send(tok, ch_mem_write_req, ram_req);

        // Complete write requests
        trace_fmt!("Waiting for mem, req = {}", ram_req);
        let (tok, _) = recv(tok, ch_mem_write_rsp);
        trace_fmt!("Never got response!");
        let ptr_w = state.ptr_w + uN[ADDR_W]:1;

        WriterState { ptr_w, is_full, ptr_r }
    }
}

proc FIFO<DATA_W: u32, DATA_W_DIV8: u32, ID_W: u32, DEST_W: u32, ADDR_W: u32, FIFO_LENGTH: u32> {
    ch_read: chan<AxiStreamBundle<DATA_W, DATA_W_DIV8, ID_W, DEST_W>> out;
    ch_write: chan<AxiStreamBundle<DATA_W, DATA_W_DIV8, ID_W, DEST_W>> in;

    config(ch_read: chan<AxiStreamBundle<DATA_W, DATA_W_DIV8, ID_W, DEST_W>> out,
           ch_write: chan<AxiStreamBundle<DATA_W, DATA_W_DIV8, ID_W, DEST_W>> in) {
        let (ch_mem_read_req_s, ch_mem_read_req_r) = chan<ram::RWRamReq<ADDR_W, DATA_W, u32:0>>;
        let (ch_mem_read_rsp_s, ch_mem_read_rsp_r) = chan<ram::RWRamResp<DATA_W>>;
        let (ch_mem_write_req_s, ch_mem_write_req_r) = chan<ram::RWRamReq<ADDR_W, DATA_W, u32:0>>;
        let (ch_mem_write_rsp_s, ch_mem_write_rsp_r) = chan<()>;
        let (ch_ptr_r_s, ch_ptr_r_r) = chan<ReaderState<ADDR_W>>;
        let (ch_ptr_w_s, ch_ptr_w_r) = chan<WriterState<ADDR_W>>;

        spawn FifoRAM<DATA_W, ADDR_W, FIFO_LENGTH>(
            ch_mem_read_req_r, ch_mem_read_rsp_s, ch_mem_write_req_r, ch_mem_write_rsp_s);

        spawn Reader<DATA_W, DATA_W_DIV8, ID_W, DEST_W, ADDR_W, FIFO_LENGTH>(
            ch_read, ch_mem_read_req_s, ch_mem_read_rsp_r, ch_ptr_r_s, ch_ptr_w_r);

        spawn Writer<DATA_W, DATA_W_DIV8, ID_W, DEST_W, ADDR_W, FIFO_LENGTH>(
            ch_write, ch_mem_write_req_s, ch_mem_write_rsp_r, ch_ptr_r_r, ch_ptr_w_s);

        (ch_read, ch_write)
    }

    init { () }

    next(tok: token, state: ()) {  }
}

const TEST_0_DATA_W = u32:8;
const TEST_0_DATA_W_DIV8 = u32:1;
const TEST_0_ID_W = u32:1;
const TEST_0_DEST_W = u32:1;
const TEST_0_ADDR_W = u32:4;
const TEST_0_FIFO_L = u32:16;

const TEST_0_MAX_ITER = u32:73;
const TEST_0_BEGIN = uN[TEST_0_DATA_W]:10;

pub struct TestState<DATA_W: u32> { data: uN[DATA_W], iter: u32, read_counter: u32 }

#[test_proc]
proc test_fifo {
    ch_fifo_read:
    chan<AxiStreamBundle<TEST_0_DATA_W, TEST_0_DATA_W_DIV8, TEST_0_ID_W, TEST_0_DEST_W>> in;
    ch_fifo_write:
    chan<AxiStreamBundle<TEST_0_DATA_W, TEST_0_DATA_W_DIV8, TEST_0_ID_W, TEST_0_DEST_W>> out;
    terminator: chan<bool> out;

    config(terminator: chan<bool> out) {
        let (ch_fifo_read_s, ch_fifo_read_r) =
            chan<AxiStreamBundle<TEST_0_DATA_W, TEST_0_DATA_W_DIV8, TEST_0_ID_W, TEST_0_DEST_W>>;
        let (ch_fifo_write_s, ch_fifo_write_r) =
            chan<AxiStreamBundle<TEST_0_DATA_W, TEST_0_DATA_W_DIV8, TEST_0_ID_W, TEST_0_DEST_W>>;
        spawn FIFO<
            TEST_0_DATA_W, TEST_0_DATA_W_DIV8, TEST_0_ID_W, TEST_0_DEST_W, TEST_0_ADDR_W, TEST_0_FIFO_L>(
            ch_fifo_read_s, ch_fifo_write_r);
        (ch_fifo_read_r, ch_fifo_write_s, terminator)
    }

    init { (TestState<TEST_0_DATA_W> { data: TEST_0_BEGIN, iter: u32:0, read_counter: u32:0 }) }

    next(tok: token, state: TestState<TEST_0_DATA_W>) {
        // Write to FIFO
        let write_data = AxiStreamBundle<TEST_0_DATA_W, TEST_0_DATA_W_DIV8, TEST_0_ID_W, TEST_0_DEST_W>
        {
            tdata: state.data,
            tstr: uN[TEST_0_DATA_W_DIV8]:0,
            tkeep: uN[TEST_0_DATA_W_DIV8]:0,
            tlast: u1:1,
            tid: uN[TEST_0_ID_W]:0,
            tdest: uN[TEST_0_DEST_W]:0
        };
        let do_send = state.iter < TEST_0_MAX_ITER;
        if do_send { trace_fmt!("Write DATA={}", write_data.tdata); } else {  };
        let tok = send_if(tok, ch_fifo_write, do_send, write_data);

        // Read from FIFO
        let zero_read_data = AxiStreamBundle<TEST_0_DATA_W, TEST_0_DATA_W_DIV8, TEST_0_ID_W, TEST_0_DEST_W>
        {
            tdata: uN[TEST_0_DATA_W]:0,
            tstr: uN[TEST_0_DATA_W_DIV8]:0,
            tkeep: uN[TEST_0_DATA_W_DIV8]:0,
            tlast: u1:1,
            tid: uN[TEST_0_ID_W]:0,
            tdest: uN[TEST_0_DEST_W]:0
        };
        let (tok, read_data, is_r_valid) = recv_non_blocking(tok, ch_fifo_read, zero_read_data);

        let read_counter = if is_r_valid {
            trace_fmt!("Read DATA={}", read_data.tdata);
            state.read_counter + u32:1
        } else {
            state.read_counter
        };

        if is_r_valid {
            assert_eq(read_data.tdata as u32, (TEST_0_BEGIN as u32) + (read_counter - u32:1));
        } else {


        };

        // Terminate test?
        let terminate = read_counter == TEST_0_MAX_ITER;
        if terminate { trace_fmt!("Terminate at: {}", read_counter); } else {  };
        let tok = send_if(tok, terminator, terminate, true);

        TestState<TEST_0_DATA_W> {
            data: state.data + uN[TEST_0_DATA_W]:1, iter: state.iter + u32:1, read_counter
        }
    }
}

const TEST_1_DATA_W = u32:8;
const TEST_1_DATA_W_DIV8 = u32:1;
const TEST_1_ID_W = u32:1;
const TEST_1_DEST_W = u32:1;
const TEST_1_ADDR_W = u32:4;
const TEST_1_FIFO_L = u32:16;

const TEST_1_MAX_ITER = u32:10;
const TEST_1_BEGIN = uN[TEST_1_DATA_W]:10;

// This test proc affects exeuction of the previous test proc!
// If test_double_fifo_gpf is commented, then test_fifo proc ends succesfully.
// Otherwise:
// [ RUN UNITTEST  ] test_fifo
// E1219 13:17:55.614045  108138 run_routines.cc:282] Internal error: DEADLINE_EXCEEDED: Exceeded
// limit of 100000 proc ticks before terminating
// [        FAILED ] test_fifo: internal error: DEADLINE_EXCEEDED: Exceeded limit of 100000 proc
// ticks before terminating
// [ RUN UNITTEST  ] test_double_fifo_gpf
// [            OK ]
// [===============] 2 test(s) ran; 1 failed; 0 skipped.

// FIFO_0 --> GPF --> FIFO_1

#[test_proc]
proc test_double_fifo_gpf {
    ch_fifo1_read:
    chan<AxiStreamBundle<TEST_1_DATA_W, TEST_1_DATA_W_DIV8, TEST_1_ID_W, TEST_1_DEST_W>> in;
    ch_fifo0_write:
    chan<AxiStreamBundle<TEST_1_DATA_W, TEST_1_DATA_W_DIV8, TEST_1_ID_W, TEST_1_DEST_W>> out;
    terminator: chan<bool> out;

    config(terminator: chan<bool> out) {
        let (ch_fifo0_read_s, ch_fifo0_read_r) =
            chan<AxiStreamBundle<TEST_1_DATA_W, TEST_1_DATA_W_DIV8, TEST_1_ID_W, TEST_1_DEST_W>>;
        let (ch_fifo0_write_s, ch_fifo0_write_r) =
            chan<AxiStreamBundle<TEST_1_DATA_W, TEST_1_DATA_W_DIV8, TEST_1_ID_W, TEST_1_DEST_W>>;

        let (ch_fifo1_read_s, ch_fifo1_read_r) =
            chan<AxiStreamBundle<TEST_1_DATA_W, TEST_1_DATA_W_DIV8, TEST_1_ID_W, TEST_1_DEST_W>>;
        let (ch_fifo1_write_s, ch_fifo1_write_r) =
            chan<AxiStreamBundle<TEST_1_DATA_W, TEST_1_DATA_W_DIV8, TEST_1_ID_W, TEST_1_DEST_W>>;

        // Order of `spawn` expressions here matters!

        // Using option 1
        // I trace the state of 2 RAMs in 2 FIFOs
        // 1st FIFO: RAM State = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 0, 0, 0, 0, 0]
        // 2nd FIFO: RAM State = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        // This is almost what I expected whilst developing. There is a bunch of writes to first FIFO
        // and I expect that later data will go through the GPF and into the 2nd FIFO.

        // Using option 2 (reverse order of `spawns`)
        // I trace the state of 2 RAMs in 2 FIFOs
        // 1st FIFO: RAM State = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        // 2nd FIFO: RAM State = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        // Both are always empty, not even the first write occurs.

        // I find this behaviour confusing and am not sure how to continue with testing/debugging the
        // FIFO

        // Option 1.
        // spawn FIFO<TEST_1_DATA_W, TEST_1_DATA_W_DIV8, TEST_1_ID_W, TEST_1_DEST_W, TEST_1_ADDR_W,
        // TEST_1_FIFO_L>
        // (ch_fifo1_read_s, ch_fifo1_write_r);

        // spawn FIFO<TEST_1_DATA_W, TEST_1_DATA_W_DIV8, TEST_1_ID_W, TEST_1_DEST_W, TEST_1_ADDR_W,
        // TEST_1_FIFO_L>
        // (ch_fifo0_read_s, ch_fifo0_write_r);

        // Option 2.
        spawn FIFO<
            TEST_1_DATA_W, TEST_1_DATA_W_DIV8, TEST_1_ID_W, TEST_1_DEST_W, TEST_1_ADDR_W, TEST_1_FIFO_L>(
            ch_fifo0_read_s, ch_fifo0_write_r);

        spawn FIFO<
            TEST_1_DATA_W, TEST_1_DATA_W_DIV8, TEST_1_ID_W, TEST_1_DEST_W, TEST_1_ADDR_W, TEST_1_FIFO_L>(
            ch_fifo1_read_s, ch_fifo1_write_r);

        spawn gpf::gpf<TEST_1_DATA_W, TEST_1_DATA_W_DIV8, TEST_1_ID_W, TEST_1_DEST_W>(
            ch_fifo0_read_r, ch_fifo1_write_s);

        (ch_fifo1_read_r, ch_fifo0_write_s, terminator)
    }

    init { (TestState<TEST_1_DATA_W> { data: TEST_1_BEGIN, iter: u32:0, read_counter: u32:0 }) }

    next(tok: token, state: TestState<TEST_1_DATA_W>) {
        // Write to FIFO
        let write_data = AxiStreamBundle<TEST_1_DATA_W, TEST_1_DATA_W_DIV8, TEST_1_ID_W, TEST_1_DEST_W>
        {
            tdata: state.data,
            tstr: uN[TEST_1_DATA_W_DIV8]:0,
            tkeep: uN[TEST_1_DATA_W_DIV8]:0,
            tlast: u1:1,
            tid: uN[TEST_1_ID_W]:0,
            tdest: uN[TEST_1_DEST_W]:0
        };
        let do_send = state.iter < TEST_1_MAX_ITER;
        if do_send { trace_fmt!("Write DATA={}", write_data.tdata); } else {  };
        let tok = send_if(tok, ch_fifo0_write, do_send, write_data);

        // Read from FIFO
        let zero_read_data = AxiStreamBundle<TEST_1_DATA_W, TEST_1_DATA_W_DIV8, TEST_1_ID_W, TEST_1_DEST_W>
        {
            tdata: uN[TEST_1_DATA_W]:0,
            tstr: uN[TEST_1_DATA_W_DIV8]:0,
            tkeep: uN[TEST_1_DATA_W_DIV8]:0,
            tlast: u1:1,
            tid: uN[TEST_1_ID_W]:0,
            tdest: uN[TEST_1_DEST_W]:0
        };
        let (tok, read_data, is_r_valid) = recv_non_blocking(tok, ch_fifo1_read, zero_read_data);

        let read_counter = if is_r_valid {
            trace_fmt!("Read DATA={}", read_data.tdata);
            state.read_counter + u32:1
        } else {
            state.read_counter
        };

        if is_r_valid {
            assert_eq(read_data.tdata as u32, (TEST_1_BEGIN as u32) + (read_counter - u32:1));
        } else {


        };

        // Terminate test?
        // let terminate = read_counter == TEST_1_MAX_ITER;
        let terminate = state.iter >= u32:50;  // Even with 1000 I don't see data in RAM (option 2)
        if terminate { trace_fmt!("Terminate at: {}", read_counter); } else {  };
        let tok = send_if(tok, terminator, terminate, true);

        TestState<TEST_1_DATA_W> {
            data: state.data + uN[TEST_1_DATA_W]:1, iter: state.iter + u32:1, read_counter
        }
    }
}

// This proc affects exeuction of the previous test proc!
// If commented, `test_fifo` proc ends succesfully.
// If uncommented, this proc causes the 'test_fifo' to end with:
// E1219 13:15:11.058317  106946 run_routines.cc:282] Internal error: DEADLINE_EXCEEDED: Exceeded
// limit of 100000 proc ticks before terminating
// [        FAILED ] test_fifo: internal error: DEADLINE_EXCEEDED: Exceeded limit of 100000 proc
// ticks before terminating

// Verilog generation
// const SYNTH_0_DATA_W = u32:8;
// const SYNTH_0_DATA_W_DIV8 = u32:1;
// const SYNTH_0_ID_W = u32:1;
// const SYNTH_0_DEST_W = u32:1;
// const SYNTH_0_ADDR_W = u32:4;
// const SYNTH_0_FIFO_L = u32:16;

// proc fifo_synth {
//     config( ch_fifo_read:chan<AxiStreamBundle<SYNTH_0_DATA_W, SYNTH_0_DATA_W_DIV8, SYNTH_0_ID_W,
//     SYNTH_0_DEST_W>> out,
//             ch_fifo_write:chan<AxiStreamBundle<SYNTH_0_DATA_W, SYNTH_0_DATA_W_DIV8, SYNTH_0_ID_W,
//             SYNTH_0_DEST_W>> in
//             ) {
//         spawn FIFO<
//             SYNTH_0_DATA_W, SYNTH_0_DATA_W_DIV8, SYNTH_0_ID_W, SYNTH_0_DEST_W, SYNTH_0_ADDR_W,
//             SYNTH_0_FIFO_L>(
//             ch_fifo_read, ch_fifo_write);
//         ()
//     }

//     init { () }

//     next(tok: token, state: ()) { () }
// }

// This does not affect behavior of previous test
proc fifo_synth_2 {
    config() { () }

    init { () }

    next(tok: token, state: ()) { () }
}
