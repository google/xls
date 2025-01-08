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

// This file contains proc responsible for automatically refilling ShiftBuffer
// with data from memory from some starting address, consecutive data from
// increasing addresses


import std;
import xls.modules.shift_buffer.shift_buffer;
import xls.modules.zstd.memory.mem_reader;


pub type RefillingShiftBufferInput = shift_buffer::ShiftBufferPacket;
pub type RefillingShiftBufferCtrl = shift_buffer::ShiftBufferCtrl;

pub struct RefillStart<ADDR_W: u32> {
    start_addr: uN[ADDR_W]
}

pub struct RefillingShiftBufferOutput<DATA_WIDTH: u32, LENGTH_WIDTH: u32> {
    data: uN[DATA_WIDTH],
    length: uN[LENGTH_WIDTH],
    error: bool,
}

enum RefillerFsm: u2 {
    IDLE = 0,
    REFILLING = 1,
    FLUSHING = 2,
}

struct RefillerState<ADDR_W: u32, LENGTH_W: u32, BUFFER_W_CLOG2: u32> {
    curr_addr: uN[ADDR_W],                      // next memory address to request data from
    fsm: RefillerFsm,                           // FSM state
    future_buf_occupancy: uN[BUFFER_W_CLOG2],   // amount of bits that are currently in the ShiftBuffer +
                                                // amount of bits that will enter ShiftBuffer once all
                                                // pending memory requests are served
    axi_error: bool,                            // whether or not at least one memory read resulted in AXI error -
                                                // - this bit is sticky and can be cleared only by flushing
    bits_to_axi_error: uN[BUFFER_W_CLOG2],      // amount of bits that we need to consume from the
                                                // ShiftBuffer to trigger an AXI error
    bits_to_flush: uN[BUFFER_W_CLOG2],          // amount of bits left to flush during flushing state
}

pub fn length_width(data_width: u32) -> u32 {
    shift_buffer::length_width(data_width)
}

// works only on values with bit length divisible by 8
fn reverse_byte_order<N_BITS: u32, N_BYTES: u32 = {N_BITS / u32:8}>(data: uN[N_BITS]) -> uN[N_BITS] {
    const_assert!(std::ceil_div(N_BITS, u32:8) == N_BITS / u32:8);
    unroll_for! (i, acc): (u32, uN[N_BITS]) in range(u32:0, N_BYTES) {
        let offset = i * u32:8;
        let offset_rev = (N_BYTES - i - u32:1) * u32:8;
        acc | (rev(data[offset +: u8]) as uN[N_BITS] << offset_rev)
    }(uN[N_BITS]:0)
}

#[test]
fn test_reverse_byte_order() {
    assert_eq(reverse_byte_order(u64:0b00000001_00100011_01000101_01100111_10001001_10101011_11001101_11101111), u64:0b11110111_10110011_11010101_10010001_11100110_10100010_11000100_10000000);
    assert_eq(reverse_byte_order(u32:0b10001001_10101011_11001101_11101111), u32:0b11110111_10110011_11010101_10010001);
    assert_eq(reverse_byte_order(u16:0b11001101_11101111), u16:0b11110111_10110011);
}

proc RefillingShiftBufferInternal<
    DATA_W: u32, ADDR_W: u32, BACKWARDS: bool = {false}, INSTANCE: u32 = {u32:0},
    LENGTH_W: u32 = {length_width(DATA_W)},
    DATA_W_DIV8: u32 = {DATA_W / u32:8},
    BUFFER_W: u32 = {DATA_W * u32:2},             // TODO: fix implementation detail of ShiftBuffer leaking here
    BUFFER_W_CLOG2: u32 = {std::clog2(BUFFER_W) + u32:1},
>{
    type MemReaderReq = mem_reader::MemReaderReq<ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<DATA_W, ADDR_W>;
    type MemReaderStatus = mem_reader::MemReaderStatus;
    type StartReq = RefillStart<ADDR_W>;
    type RSBInput = RefillingShiftBufferInput<DATA_W, LENGTH_W>;
    type RSBOutput = RefillingShiftBufferOutput<DATA_W, LENGTH_W>;
    type RSBCtrl = RefillingShiftBufferCtrl<LENGTH_W>;
    type SBOutput = shift_buffer::ShiftBufferOutput<DATA_W, LENGTH_W>;
    type State = RefillerState<ADDR_W, LENGTH_W, BUFFER_W_CLOG2>;
    type Fsm = RefillerFsm;
    type BufferSize = uN[BUFFER_W_CLOG2];

    reader_req_s: chan<MemReaderReq> out;
    reader_resp_r: chan<MemReaderResp> in;
    start_req_r: chan<StartReq> in;
    stop_flush_req_r: chan<()> in;
    buffer_data_in_s: chan<RSBInput> out;
    buffer_data_out_s: chan<RSBOutput> out;
    buffer_ctrl_r: chan<RSBCtrl> in;
    snoop_data_out_r: chan<SBOutput> in;
    snoop_ctrl_s: chan<RSBCtrl> out;
    flushing_done_s: chan<()> out;

    config(
        reader_req_s: chan<MemReaderReq> out,
        reader_resp_r: chan<MemReaderResp> in,
        start_req_r: chan<StartReq> in,
        stop_flush_req_r: chan<()> in,
        buffer_ctrl_r: chan<RSBCtrl> in,
        buffer_data_out_s: chan<RSBOutput> out,
        snoop_ctrl_s: chan<RSBCtrl> out,
        buffer_data_in_s: chan<RSBInput> out,
        snoop_data_out_r: chan<SBOutput> in,
        flushing_done_s: chan<()> out,
    ) {
        (reader_req_s, reader_resp_r, start_req_r, stop_flush_req_r,
        buffer_data_in_s, buffer_data_out_s, buffer_ctrl_r, snoop_data_out_r,
        snoop_ctrl_s, flushing_done_s)
    }

    init {
        zero!<State>()
    }

    next(state: State) {
        let tok = join();

        // trace_fmt!("Current refiller state: {:#x}", state);

        // receive start and stop&flush requests
        let (_, start_req, start_valid) = recv_if_non_blocking(tok, start_req_r, state.fsm == Fsm::IDLE, zero!<StartReq>());
        let (_, (), stop_flush_valid) = recv_if_non_blocking(tok, stop_flush_req_r, state.fsm == Fsm::REFILLING, ());

        // flush logic
        let flushing_end = state.future_buf_occupancy == BufferSize:0;
        let flushing = state.fsm == Fsm::FLUSHING;
        // flush at most DATA_W bits in a given next() evaluation
        let flush_amount_bits = std::min(DATA_W as BufferSize, state.bits_to_flush);
        // send "flushing done" notification once we complete it
        send_if(tok, flushing_done_s, flushing && flushing_end, ());
        // if (flushing && flushing_end) {
        //     trace_fmt!("Sent done on the flushing done channel");
        // } else {};

        // snooping logic for the ShiftBuffer control channel
        // recv and immediately send out control packets heading for ShiftBuffer,
        // unless we're flushing, if so we block receiving any new control packets
        let (_, snoop_ctrl, snoop_ctrl_valid) = recv_if_non_blocking(tok, buffer_ctrl_r, !flushing, zero!<RSBCtrl>());
        // If we're flushing send our packets for taking out data from the shiftbuffer
        // (that data will then be discarded)
        let ctrl_packet = if (flushing) {
            RSBCtrl {length: flush_amount_bits as uN[LENGTH_W]}
        } else if (snoop_ctrl_valid) {
            snoop_ctrl
        } else {
            zero!<RSBCtrl>()
        };
        let do_send_ctrl = (flushing && flush_amount_bits > BufferSize:0) || snoop_ctrl_valid;
        send_if(tok, snoop_ctrl_s, do_send_ctrl, ctrl_packet);
        // if do_send_ctrl {
        //     trace_fmt!("Sent snooped/injected control packet: {:#x}", ctrl_packet);
        // } else {};

        // snoop data output packet (for keeping track how many bits in ShiftBuffer are occupied)
        let (_, snoop_data, snoop_data_valid) = recv_non_blocking(tok, snoop_data_out_r, zero!<SBOutput>());

        // refilling logic
        const REFILL_SIZE = DATA_W_DIV8 as uN[ADDR_W];
        // we eagerly request data based on the *future* capacity of the buffer,
        // this might stall us (and in turn MemReader and potentially the whole bus)
        // on send to buffer_data_in_s if the proc sending control requests isn't
        // receiving the data on the output channel fast enough, but this is true
        // of any proc that uses MemReader and we don't consider this an issue
        let buf_will_have_enough_space = state.future_buf_occupancy <= DATA_W as BufferSize;    // TODO: fix implementation detail of ShiftBuffer leaking here
        let do_refill_cycle = state.fsm == Fsm::REFILLING && buf_will_have_enough_space;
        // send request to memory for more data under the assumption
        // that there's enough space in the ShiftBuffer to fit it
        let mem_req = MemReaderReq {
            addr: state.curr_addr,
            length: REFILL_SIZE,
        };
        send_if(tok, reader_req_s, do_refill_cycle, mem_req);
        // if (do_refill_cycle) {
        //     trace_fmt!("[{:#x}] Sent request for data to memory: {:#x}", INSTANCE, mem_req);
        // } else {};
        // receive data from memory
        let (_, reader_resp, reader_resp_valid) = recv_non_blocking(tok, reader_resp_r, zero!<MemReaderResp>());
        // if reader_resp_valid {
        //     trace_fmt!("[{:#x}] Received data from memory: {:#x}", INSTANCE, reader_resp);
        // } else {};
        // always send some data regardless of the reader_resp.status to allow for all requests
        // to complete (possibly with invalid data) since the response channel queue must be empty for
        // flushing to work correctly
        let do_buffer_refill = reader_resp_valid;
        let reader_resp_len_bits = DATA_W as uN[LENGTH_W];
        let data_packet = RSBInput {
            data: if BACKWARDS { reverse_byte_order(reader_resp.data) } else { reader_resp.data },
            length: reader_resp_len_bits,
        };
        // this send might stall only if proc that receives responses isn't reading from the
        // ShiftBuffer fast enough, apart from that since part of the condition `do_buffer_refill`
        // is `buf_will_have_enough_space` it should not block
        send_if(tok, buffer_data_in_s, do_buffer_refill, data_packet);
        // if (do_buffer_refill) {
        //     trace_fmt!("Sent data to the ShiftBuffer: {:#x}", data_packet);
        // } else {};

        // length of additional data that will be inserted into the ShiftBuffer *in the future*
        // once all pending memory requests are served
        let future_input_bits = if (do_refill_cycle) {
            DATA_W as BufferSize
        } else {
            BufferSize:0
        };
        // actual amount of bits inserted into the ShiftBuffer in this next() evaluation
        let input_bits = if (do_buffer_refill) {
            DATA_W as BufferSize
        } else {
            BufferSize:0
        };
        // length of data that was snooped on the ShiftBuffer output
        // note: default value of snoop_ctrl.length from its recv_if_non_blocking is 0
        let output_bits = snoop_data.length as BufferSize;
        // calculate the difference in the amount of bits inserted/taken out
        // this will never underflow as it's always true that output_bits <= state.future_buf_occupancy
        // (because output_bits is based on the number of outgoing bits from the buffer which cannot be
        // larger than its current occupancy)
        let next_future_buf_occupancy = state.future_buf_occupancy + future_input_bits - output_bits;

        // keep track of the amount of remaining bits to flush
        let next_bits_to_flush = if (flushing) {
            state.bits_to_flush - flush_amount_bits
        } else {
            next_future_buf_occupancy
        };

        // error handling
        // we've encountered an error, either previously or in this next() evaluation
        let axi_error = state.axi_error || (reader_resp_valid && reader_resp.status == MemReaderStatus::ERROR);
        let next_bits_to_axi_error = if (axi_error) {
            if (state.bits_to_axi_error < snoop_data.length as BufferSize) {
                // prevent underflow
                BufferSize:0
            } else {
                // keep track of amount of bits to reach offending data (from ERROR memory response)
                state.bits_to_axi_error - (snoop_data.length as BufferSize)
            }
        } else if (flushing_end) {
            // reset the counter after a flush since its state will be invalid after that
            BufferSize:0
        } else {
            // keep track of current amount of bits in the buffer
            state.bits_to_axi_error + input_bits - output_bits
        };
        // check if we will consume at least one bit from the data that returned AXI error
        let reads_error_bits = snoop_data_valid && state.bits_to_axi_error < snoop_data.length as BufferSize;

        // data snoop forwarding logic
        // forward data heading for the ShiftBuffer output, attaching an error bit
        // if we've encountered an AXI error, unless we're flushing - in that case discard snoop_data
        let forward_snooped_data = snoop_data_valid && !flushing;
        send_if(tok, buffer_data_out_s, forward_snooped_data, RSBOutput {
            data: if BACKWARDS {
                rev(snoop_data.data) >> (u32:64 - snoop_data.length as u32)
            } else {
                snoop_data.data
            },
            length: snoop_data.length,
            error: axi_error && reads_error_bits,
        });
        // if forward_snooped_data {
        //     trace_fmt!("[{:#x}] Forwarded snooped data output packet: {:#x}", INSTANCE, snoop_data);
        // } else {};

        // FSM
        let next_state = match (state.fsm) {
            Fsm::IDLE => {
                if (start_valid) {
                    State {
                        fsm: Fsm::REFILLING,
                        curr_addr: if BACKWARDS {
                            start_req.start_addr - DATA_W_DIV8 as uN[ADDR_W]
                        } else {
                            start_req.start_addr
                        },
                        ..state
                    }
                } else {
                    state
                }
            },
            Fsm::REFILLING => {
                // stop and AXI error might happen on the same cycle,
                // in that case stop&flush takes precedence over error
                if (stop_flush_valid) {
                    State {
                        fsm: Fsm::FLUSHING,
                        ..state
                    }
                } else if (do_refill_cycle) {
                    State {
                        curr_addr: if BACKWARDS {
                            state.curr_addr - REFILL_SIZE
                        } else {
                            state.curr_addr + REFILL_SIZE
                        },
                        ..state
                    }
                } else {
                    state
                }
            },
            Fsm::FLUSHING => {
                if (flushing_end) {
                    State {
                        fsm: Fsm::IDLE,
                        ..state
                    }
                } else {
                    state
                }
            },
            _ => fail!("refilling_shift_buffer_fsm_unreachable", zero!<State>())
        };

        let next_axi_error = axi_error && next_state.fsm == Fsm::REFILLING;

        // combine next FSM state with buffer occupancy data
        let next_state = State {
            future_buf_occupancy: next_future_buf_occupancy,
            bits_to_axi_error: next_bits_to_axi_error,
            bits_to_flush: next_bits_to_flush,
            axi_error: next_axi_error,
            ..next_state
        };

        // check some invariants
        // asserts are equivalent to implications in a preceding comment
        // state.fsm == Fsm::IDLE -> next_future_buf_occupancy == 0
        assert!(!(state.fsm == Fsm::IDLE) || state.future_buf_occupancy == BufferSize:0, "future_buf_occupancy was not 0 in IDLE state");
        // state.fsm == Fsm::IDLE -> state.bits_to_axi_error == BufferSize:0
        assert!(!(state.fsm == Fsm::IDLE) || state.bits_to_axi_error == BufferSize:0, "bits_to_axi_error was not 0 in IDLE state");
        // state.fsm == Fsm::IDLE -> state.bits_to_flush == BufferSize:0
        assert!(!(state.fsm == Fsm::IDLE) || state.bits_to_flush == BufferSize:0, "bits_to_flush was not 0 in IDLE state");

        // state.fsm == Fsm::REFILLING -> state.future_buf_occupancy >= state.bits_to_axi_error
        assert!(!(state.fsm == Fsm::REFILLING) || state.future_buf_occupancy >= state.bits_to_axi_error, "future_buf_occupancy >= bits_to_axi_error in REFILLING state");
        // state.fsm == Fsm::REFILLING -> state.future_buf_occupancy >= state.bits_to_flush
        assert!(!(state.fsm == Fsm::REFILLING) || state.future_buf_occupancy >= state.bits_to_flush, "future_buf_occupancy >= bits_to_flush in REFILLING state");
        // state.fsm == Fsm::REFILLING -> state.bits_to_flush >= state.bits_to_axi_error
        assert!(!(state.fsm == Fsm::REFILLING) || state.bits_to_flush >= state.bits_to_axi_error, "bits_to_flush >= bits_to_axi_error in REFILLING state");

        // state.fsm != Fsm::REFILLING -> state.axi_error == false
        assert!(!(state.fsm != Fsm::REFILLING) || state.axi_error == false, "axi_error was true in a state other than REFILLING");
        // axi_error -> state.bits_to_axi_error >= next_bits_to_axi_error
        assert!(!axi_error || state.bits_to_axi_error >= next_bits_to_axi_error, "state.bits_to_axi_error increased during axi_error");
        // flushing -> state.bits_to_flush >= next_bits_to_flush
        assert!(!flushing || state.bits_to_flush >= next_bits_to_flush, "state.bits_to_flush increased during flushing");

        next_state
    }
}

// Main proc for RefillingShiftBuffer
//
// Typical usage pattern is as follows:
// 1. Send start request with starting address where the refilling is supposed
//    to start from on start_req channel
// 2. Send requests for up to DATA_W bits on buffer_ctrl channel
// 3. Receive responses on buffer_data_out channel
// 4. Once you're done, send a request on stop_flush_req channel
//    and wait for confirmation on flushing_done channel
//
// In case of an AXI error on the bus an error bit is set in response
// on buffer_data_out channel. You may still send requests on buffer_ctrl
// and receive responses on buffer_data_out but the data is not guaranteed
// to be correct and said error bit will always be set from that point
// onwards until you trigger a flush
//
// To send a request on stop_flush_req channel, you must first receive all
// responses from the buffer_data_out channel that you sent requests for on
// buffer_ctrl channel

pub proc RefillingShiftBuffer<
    DATA_W: u32,
    ADDR_W: u32,
    BACKWARDS: bool = {false},
    INSTANCE: u32 = {u32:0},
    LENGTH_W: u32 = {length_width(DATA_W)},
> {
    type MemReaderReq = mem_reader::MemReaderReq<ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<DATA_W, ADDR_W>;
    type StartReq = RefillStart<ADDR_W>;
    type RSBInput = RefillingShiftBufferInput<DATA_W, LENGTH_W>;
    type RSBOutput = RefillingShiftBufferOutput<DATA_W, LENGTH_W>;
    type RSBCtrl = RefillingShiftBufferCtrl<LENGTH_W>;
    type SBOutput = shift_buffer::ShiftBufferOutput<DATA_W, LENGTH_W>;

    config(
        reader_req_s: chan<MemReaderReq> out,
        reader_resp_r: chan<MemReaderResp> in,
        start_req_r: chan<StartReq> in,
        stop_flush_req_r: chan<()> in,
        buffer_ctrl_r: chan<RSBCtrl> in,
        buffer_data_out_s: chan<RSBOutput> out,
        flushing_done_s: chan<()> out,
    ) {
        const CHANNEL_DEPTH = u32:1;

        let (buffer_data_in_s, buffer_data_in_r) = chan<RSBInput, CHANNEL_DEPTH>("buffer_data_in");
        let (snoop_data_out_s, snoop_data_out_r) = chan<SBOutput, CHANNEL_DEPTH>("snoop_data_out_s");
        let (snoop_ctrl_s, snoop_ctrl_r) = chan<RSBCtrl, CHANNEL_DEPTH>("snoop_ctrl");

        spawn shift_buffer::ShiftBuffer<DATA_W, LENGTH_W>(
            snoop_ctrl_r, buffer_data_in_r, snoop_data_out_s
        );
        spawn RefillingShiftBufferInternal<DATA_W, ADDR_W, BACKWARDS, INSTANCE>(
            reader_req_s,
            reader_resp_r,
            start_req_r,
            stop_flush_req_r,
            buffer_ctrl_r,
            buffer_data_out_s,
            snoop_ctrl_s,
            buffer_data_in_s,
            snoop_data_out_r,
            flushing_done_s,
        );
    }

    init {}

    next(_: ()) {}
}


const TEST_DATA_W = u32:64;
const TEST_ADDR_W = u32:32;
const TEST_LENGTH_W = length_width(TEST_DATA_W);
const TEST_DATA_W_DIV8 = TEST_DATA_W / u32:8;
const TEST_BUFFER_W = TEST_DATA_W * u32:2;             // TODO: fix implementation detail of ShiftBuffer leaking here
const TEST_BUFFER_W_CLOG2 = std::clog2(TEST_BUFFER_W);

proc RefillingShiftBufferTest<BACKWARDS: bool> {
    type MemReaderReq = mem_reader::MemReaderReq<TEST_ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<TEST_DATA_W, TEST_ADDR_W>;
    type MemReaderStatus = mem_reader::MemReaderStatus;
    type StartReq = RefillStart<TEST_ADDR_W>;
    type RSBInput = RefillingShiftBufferInput<TEST_DATA_W, TEST_LENGTH_W>;
    type RSBOutput = RefillingShiftBufferOutput<TEST_DATA_W, TEST_LENGTH_W>;
    type RSBCtrl = RefillingShiftBufferCtrl<TEST_LENGTH_W>;
    type State = RefillerState<TEST_ADDR_W, TEST_BUFFER_W_CLOG2>;

    terminator: chan<bool> out;
    reader_req_r: chan<MemReaderReq> in;
    reader_resp_s: chan<MemReaderResp> out;
    start_req_s: chan<StartReq> out;
    stop_flush_req_s: chan<()> out;
    buffer_ctrl_s: chan<RSBCtrl> out;
    buffer_data_out_r: chan<RSBOutput> in;
    flushing_done_r: chan<()> in;

    config(terminator: chan<bool> out) {
        let (reader_req_s, reader_req_r) = chan<MemReaderReq>("reader_req");
        let (reader_resp_s, reader_resp_r) = chan<MemReaderResp>("reader_resp");
        let (start_req_s, start_req_r) = chan<StartReq>("start_req");
        let (stop_flush_req_s, stop_flush_req_r) = chan<()>("stop_flush_req");
        let (buffer_ctrl_s, buffer_ctrl_r) = chan<RSBCtrl>("buffer_ctrl");
        let (buffer_data_out_s, buffer_data_out_r) = chan<RSBOutput>("buffer_data_out");
        let (flushing_done_s, flushing_done_r) = chan<()>("flushing_done");

        spawn RefillingShiftBuffer<TEST_DATA_W, TEST_ADDR_W, BACKWARDS>(
            reader_req_s, reader_resp_r, start_req_r, stop_flush_req_r,
            buffer_ctrl_r, buffer_data_out_s, flushing_done_s,
        );

        (
            terminator, reader_req_r, reader_resp_s, start_req_s,
            stop_flush_req_s, buffer_ctrl_s, buffer_data_out_r,
            flushing_done_r,
        )
    }

    init { }

    next(state: ()) {
        type Addr = uN[TEST_ADDR_W];
        type Data = uN[TEST_DATA_W];
        type Length = uN[TEST_LENGTH_W];

        let tok = join();

        const REFILL_SIZE = TEST_DATA_W_DIV8 as Addr;
        let tok = send(tok, start_req_s, StartReq { start_addr: Addr:0xDEAD_0008 });

        // proc should ask for data 2 times (2/3 of the size of the internal ShiftBuffer)
        let (tok, req) = recv(tok, reader_req_r);
        assert_eq(req, MemReaderReq {
            addr: if BACKWARDS { Addr: 0xDEAD_0000 } else { Addr:0xDEAD_0008 },
            length: REFILL_SIZE,
        });
        let tok = send(tok, reader_resp_s, MemReaderResp {
            status: MemReaderStatus::OKAY,
            data: Data:0x01234567_89ABCDEF,
            length: REFILL_SIZE,
            last: true,
        });
        let (tok, req) = recv(tok, reader_req_r);
        assert_eq(req, MemReaderReq {
            addr: if BACKWARDS { Addr: 0xDEAC_FFF8 } else { Addr:0xDEAD_0010 },
            length: REFILL_SIZE,
        });
        let tok = send(tok, reader_resp_s, MemReaderResp {
            status: MemReaderStatus::OKAY,
            data: Data:0xFEDCBA98_76543210,
            length: REFILL_SIZE,
            last: true,
        });

        // read single byte
        let tok = send(tok, buffer_ctrl_s, RSBCtrl {
            length: Length:8
        });
        let (tok, resp) = recv(tok, buffer_data_out_r);
        assert_eq(resp, RSBOutput {
            data: if BACKWARDS { Data:0x01 } else { Data:0xEF },
            length: Length:8,
            error: false,
        });

        // proc shouldn't be asking for any more data at this point
        let tok = for (_, tok): (u32, token) in u32:1..u32:100 {
            let (tok, _, data_valid) = recv_non_blocking(tok, reader_req_r, zero!<MemReaderReq>());
            assert_eq(data_valid, false);
            tok
        }(tok);

        // read enough data from the buffer to trigger a refill
        let tok = send(tok, buffer_ctrl_s, RSBCtrl {
            length: Length:56
        });
        let (tok, resp) = recv(tok, buffer_data_out_r);
        assert_eq(resp, RSBOutput {
            data: if BACKWARDS { Data:0x23456789ABCDEF } else { Data:0x01234567_89ABCD },
            length: Length:56,
            error: false,
        });
        let (tok, req) = recv(tok, reader_req_r);
        assert_eq(req, MemReaderReq {
            addr: if BACKWARDS { Addr: 0xDEAC_FFF0 } else { Addr:0xDEAD_0018 } ,
            length: REFILL_SIZE,
        });
        // don't respond to the request yet

        // we have 64 bits in the buffer at this point - almost empty it manually
        let tok = send(tok, buffer_ctrl_s, RSBCtrl {
            length: Length:60
        });
        let (tok, resp) = recv(tok, buffer_data_out_r);
        assert_eq(resp, RSBOutput {
            data: if BACKWARDS { Data:0x0FEDCBA9_87654321 } else { Data:0xEDCBA98_76543210 },
            length: Length:60,
            error: false,
        });

        // ask for more data from the buffer (but not enough data is available)
        let tok = send(tok, buffer_ctrl_s, RSBCtrl {
            length: Length:12
        });
        // make sure that reading from output is stuck
        let tok = for (_, tok): (u32, token) in u32:1..u32:100 {
            let (tok, _, data_valid) = recv_non_blocking(tok, buffer_data_out_r, zero!<RSBOutput>());
            assert_eq(data_valid, false);
            tok
        }(tok);

        // serve earlier memory request
        let tok = send(tok, reader_resp_s, MemReaderResp {
            status: MemReaderStatus::OKAY,
            data: Data:0x02481357_8ACE9BD0,
            length: REFILL_SIZE,
            last: true,
        });
        // should be able to receive from the buffer now
        let (tok, resp) = recv(tok, buffer_data_out_r);
        assert_eq(resp, RSBOutput {
            data: if BACKWARDS { Data:0x02 } else { Data:0xD0F },
            length: Length:12,
            error: false,
        });

        // buffer now contains 56 bits - proc should have sent 1 more
        // memory requests by this point - serve it
        let (tok, req) = recv(tok, reader_req_r);
        assert_eq(req, MemReaderReq {
            addr: if BACKWARDS { Addr:0xDEAC_FFE8 } else { Addr:0xDEAD_0020 },
            length: REFILL_SIZE,
        });
        let tok = send(tok, reader_resp_s, MemReaderResp {
            status: MemReaderStatus::OKAY,
            data: Data:0x86868686_42424242,
            length: REFILL_SIZE,
            last: true,
        });

        // make sure proc is not requesting more data that we can insert into the buffer
        let tok = for (_, tok): (u32, token) in u32:1..u32:100 {
            let (tok, _, req_valid) = recv_non_blocking(tok, reader_req_r, zero!<MemReaderReq>());
            assert_eq(req_valid, false);
            tok
        }(tok);

        // try flushing
        let tok = send(tok, stop_flush_req_s, ());
        let (tok, ()) = recv(tok, flushing_done_r);

        // start from a new address and refill buffer with more data
        let tok = send(tok, start_req_s, StartReq { start_addr: u32: 0x1000_11F0 });
        let (tok, req) = recv(tok, reader_req_r);
        assert_eq(req, MemReaderReq {
            addr: if BACKWARDS { Addr: 0x1000_11E8 } else { Addr: 0x1000_11F0 } ,
            length: REFILL_SIZE,
        });
        let tok = send(tok, reader_resp_s, MemReaderResp {
            status: MemReaderStatus::OKAY,
            data: Data:0xFEFDFCFB_FAF9F8F7,
            length: REFILL_SIZE,
            last: true,
        });

        // try reading data from the buffer after the flush
        let tok = send(tok, buffer_ctrl_s, RSBCtrl {
            length: Length:4
        });
        let (tok, resp) = recv(tok, buffer_data_out_r);
        assert_eq(resp, RSBOutput {
            data: if BACKWARDS { Data: 0xF } else { Data:0x7 },
            length: Length:4,
            error: false,
        });

        // refill with even more data
        let (tok, req) = recv(tok, reader_req_r);
        assert_eq(req, MemReaderReq {
            addr: if BACKWARDS { Addr: 0x1000_11E0 } else { Addr:0x1000_11F8 },
            length: REFILL_SIZE,
        });
        let tok = send(tok, reader_resp_s, MemReaderResp {
            status: MemReaderStatus::OKAY,
            data: Data:0xABBA_BAAB_AABB_BBAA,
            length: REFILL_SIZE,
            last: true,
        });

        // test receiving more than DATA_W bits
        let tok = send(tok, buffer_ctrl_s, RSBCtrl {
            length: Length:64
        });
        let tok = send(tok, buffer_ctrl_s, RSBCtrl {
            length: Length:60
        });

        // receive all of the new data and verify that no old data
        // remained in the buffer
        let (tok, resp) = recv(tok, buffer_data_out_r);
        assert_eq(resp, RSBOutput {
            data: if BACKWARDS { Data:0xEFDFCFBFAF9F8F7A } else { Data:0xAFEFDFCF_BFAF9F8F },
            length: Length:64,
            error: false,
        });
        let (tok, resp) = recv(tok, buffer_data_out_r);
        assert_eq(resp, RSBOutput {
            data: if BACKWARDS { Data:0xBBABAABAABBBBAA } else { Data:0xABBA_BAAB_AABB_BBA },
            length: Length:60,
            error: false,
        });

        // proc should've requested more data by now
        // respond with AXI error from MemReader
        let (tok, req) = recv(tok, reader_req_r);
        assert_eq(req, MemReaderReq {
            addr: if BACKWARDS { Addr:0x1000_11D8 } else { Addr:0x1000_1200 },
            length: REFILL_SIZE,
        });
        let tok = send(tok, reader_resp_s, MemReaderResp {
            status: MemReaderStatus::ERROR,
            data: Data:0x0,
            length: Addr:0x0,
            last: true,
        });

        // try reading from the buffer that's tainted by
        // AXI error - should induce a packet on the error channel
        let tok = send(tok, buffer_ctrl_s, RSBCtrl {
            length: Length:1
        });

        // to comply with the usage protocol of refiller we need to recv response
        let (tok, resp) = recv(tok, buffer_data_out_r);
        // don't assume anything about the response except that the lenght must be 1 and error true
        assert_eq(resp.length, Length:1);
        assert_eq(resp.error, true);

        // send some more data, can be OK status this time
        let (tok, req) = recv(tok, reader_req_r);
        assert_eq(req, MemReaderReq {
            addr: if BACKWARDS { Addr:0x1000_11D0 } else { Addr:0x1000_1208 },
            length: REFILL_SIZE,
        });
        let tok = send(tok, reader_resp_s, MemReaderResp {
            status: MemReaderStatus::OKAY,
            data: Data:0xDEADBEEF_FEEBDAED,
            length: Addr:0x40,
            last: true,
        });

        // check that we get another error after trying to read from the buffer once more
        let tok = send(tok, buffer_ctrl_s, RSBCtrl {
            length: Length:64
        });
        let (tok, resp) = recv(tok, buffer_data_out_r);
        // again don't assume anything about the response other data length and error
        assert_eq(resp.length, Length:64);
        assert_eq(resp.error, true);

        // to comply with the usage protocol of refiller we must flush it after
        // receiving the error to permit further operation in non-error state
        let tok = send(tok, stop_flush_req_s, ());

        // test that flushing works even if response from memory arrives after
        // flushing is requested
        let (tok, req) = recv(tok, reader_req_r);
        assert_eq(req, MemReaderReq {
            addr: if BACKWARDS { Addr:0x1000_11C8 } else { Addr:0x1000_1210 },
            length: REFILL_SIZE,
        });
        let tok = send(tok, reader_resp_s, MemReaderResp {
            status: MemReaderStatus::OKAY,
            data: Data:0xFFFF_EEEE_DDDD_CCCC,
            length: Addr:0x40,
            last: true,
        });

        let (tok, ()) = recv(tok, flushing_done_r);

        // test that we can restart refilling after flushing from an error state
        let tok = send(tok, start_req_s, StartReq {
            start_addr: Addr:0xABCD_0000
        });

        // respond to memory request
        let (tok, req) = recv(tok, reader_req_r);
        assert_eq(req, MemReaderReq {
            addr: if BACKWARDS { Addr:0xABCC_FFF8 } else { Addr:0xABCD_0000 },
            length: REFILL_SIZE,
        });
        let tok = send(tok, reader_resp_s, MemReaderResp {
            status: MemReaderStatus::OKAY,
            data: Data:0x0123_4567_89AB_CDEF,
            length: Addr:0x40,
            last: true,
        });

        // ask for some data
        let tok = send(tok, buffer_ctrl_s, RSBCtrl {
            length: Length:8
        });
        let (tok, resp) = recv(tok, buffer_data_out_r);
        assert_eq(resp, RSBOutput {
            data: if BACKWARDS { Data:0x01 } else { Data:0xEF },
            length: Length:8,
            error: false,
        });

        // respond to second memory request
        let (tok, req) = recv(tok, reader_req_r);
        assert_eq(req, MemReaderReq {
            addr: if BACKWARDS { Addr: 0xABCC_FFF0 } else { Addr:0xABCD_0008 },
            length: REFILL_SIZE,
        });
        // taint this response
        let tok = send(tok, reader_resp_s, MemReaderResp {
            status: MemReaderStatus::ERROR,
            data: Data:0x8888_7777_6666_5555,
            length: Addr:0x40,
            last: true,
        });

        // ask for data that won't trigger an error
        let tok = send(tok, buffer_ctrl_s, RSBCtrl {
            length: Length:48
        });
        let (tok, resp) = recv(tok, buffer_data_out_r);
        assert_eq(resp, RSBOutput {
            data: Data:0x23456789ABCD,
            length: Length:48,
            error: false,
        });

        // now ask for data that *will* trigger an error
        // we have 72 bits in the buffer, 8 untainted and 64 tainted
        let tok = send(tok, buffer_ctrl_s, RSBCtrl {
            length: Length: 9
        });
        let (tok, resp) = recv(tok, buffer_data_out_r);
        assert_eq(resp.length, Length:9);
        assert_eq(resp.error, true);

        send(tok, terminator, true);
    }
}

#[test_proc]
proc RefillingShiftBufferTestForward {
    terminator_r: chan<bool> in;
    terminator: chan<bool> out;

    config(terminator: chan<bool> out) {
        // we need to instantiate an intermediate channel since terminator channel
        // cannot be passed directly to the proc
        let (terminator_s, terminator_r) = chan<bool>("terminator");
        spawn RefillingShiftBufferTest<false>(terminator_s);
        (terminator_r, terminator)
    }
    init {}
    next(_: ()) {
        let tok = join();
        let (tok, value) = recv(tok, terminator_r);
        send(tok, terminator, value);
    }
}

#[test_proc]
proc RefillingShiftBufferTestBackward {
    terminator_r: chan<bool> in;
    terminator: chan<bool> out;

    config(terminator: chan<bool> out) {
        let (terminator_s, terminator_r) = chan<bool>("terminator");
        spawn RefillingShiftBufferTest<true>(terminator_s);
        (terminator_r, terminator)
    }
    init {}
    next(_: ()) {
        let tok = join();
        let (tok, value) = recv(tok, terminator_r);
        send(tok, terminator, value);
    }
}

proc RefillingShiftBufferInternalInst {
    type MemReaderReq = mem_reader::MemReaderReq<TEST_ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<TEST_DATA_W, TEST_ADDR_W>;
    type MemReaderStatus = mem_reader::MemReaderStatus;
    type StartReq = RefillStart<TEST_ADDR_W>;
    type RSBInput = RefillingShiftBufferInput<TEST_DATA_W, TEST_LENGTH_W>;
    type RSBOutput = RefillingShiftBufferOutput<TEST_DATA_W, TEST_LENGTH_W>;
    type RSBCtrl = RefillingShiftBufferCtrl<TEST_LENGTH_W>;
    type SBOutput = shift_buffer::ShiftBufferOutput<TEST_DATA_W, TEST_LENGTH_W>;
    type State = RefillerState<TEST_ADDR_W, TEST_LENGTH_W, TEST_BUFFER_W_CLOG2>;

    config(
        reader_req_s: chan<MemReaderReq> out,
        reader_resp_r: chan<MemReaderResp> in,
        start_req_r: chan<StartReq> in,
        stop_flush_req_r: chan<()> in,
        buffer_ctrl_r: chan<RSBCtrl> in,
        buffer_data_out_s: chan<RSBOutput> out,
        snoop_ctrl_s: chan<RSBCtrl> out,
        buffer_data_in_s: chan<RSBInput> out,
        snoop_data_out_r: chan<SBOutput> in,
        flushing_done_s: chan<()> out,
    ) {
        // instantiate with BACKWARDS = true to test worst-case results
        spawn RefillingShiftBufferInternal<TEST_DATA_W, TEST_ADDR_W, true>(
            reader_req_s, reader_resp_r, start_req_r, stop_flush_req_r,
            buffer_ctrl_r, buffer_data_out_s, snoop_ctrl_s,
            buffer_data_in_s, snoop_data_out_r, flushing_done_s,
        );
    }

    init { }

    next(state: ()) { }
}
