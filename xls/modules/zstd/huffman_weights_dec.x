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
import xls.modules.zstd.common ;
import xls.modules.zstd.huffman_prescan;
import xls.modules.zstd.memory.axi;
import xls.modules.zstd.memory.axi_ram;
import xls.modules.zstd.memory.mem_reader;
import xls.modules.zstd.ram_mux;
import xls.modules.zstd.refilling_shift_buffer;
import xls.modules.zstd.comp_lookup_dec;
import xls.modules.zstd.fse_table_creator;
import xls.modules.zstd.math;

const HUFFMAN_FSE_MAX_ACCURACY_LOG = u32:9;
const HUFFMAN_FSE_ACCURACY_W = std::clog2(HUFFMAN_FSE_MAX_ACCURACY_LOG + u32:1);

struct HuffmanRawWeightsDecoderReq<AXI_ADDR_W: u32> {
    addr: uN[AXI_ADDR_W],
    n_symbols: u8,
}

enum HuffmanRawWeightsDecoderStatus: u1 {
    OKAY = 0,
    ERROR = 1,
}

struct HuffmanRawWeightsDecoderResp {
    status: HuffmanRawWeightsDecoderStatus,
}

enum HuffmanRawWeightsDecoderFSM : u2 {
    IDLE = 0,
    DECODING = 1,
    FILL_ZERO = 2,
    RESP = 3,
}

struct HuffmanRawWeightsDecoderState<
    AXI_ADDR_W: u32, AXI_DATA_W: u32,
    WEIGHTS_RAM_ADDR_W: u32, WEIGHTS_RAM_DATA_W: u32,
    BUFF_LEN: u32 = {AXI_DATA_W + WEIGHTS_RAM_DATA_W},
    BUFF_LEN_LOG2: u32 = {std::clog2(BUFF_LEN + u32:1)},
> {
    fsm: HuffmanRawWeightsDecoderFSM,
    req: HuffmanRawWeightsDecoderReq<AXI_ADDR_W>,
    data_decoded: uN[AXI_ADDR_W],
    ram_addr: uN[WEIGHTS_RAM_ADDR_W],
    buffer: uN[BUFF_LEN],
    buffer_len: uN[BUFF_LEN_LOG2],
    ram_wr_resp_to_handle: u4,
    sum: u32, // The sum of 2^(weight-1) from HTD
}

proc HuffmanRawWeightsDecoder<
    AXI_ADDR_W: u32, AXI_DATA_W: u32,
    WEIGHTS_RAM_ADDR_W: u32, WEIGHTS_RAM_DATA_W: u32,
    WEIGHTS_RAM_NUM_PARTITIONS: u32,
    BUFF_LEN: u32 = {AXI_DATA_W + WEIGHTS_RAM_DATA_W},
    BUFF_LEN_LOG2: u32 = {std::clog2(BUFF_LEN + u32:1)},
> {
    type Req = HuffmanRawWeightsDecoderReq<AXI_ADDR_W>;
    type Resp = HuffmanRawWeightsDecoderResp;
    type Status = HuffmanRawWeightsDecoderStatus;
    type FSM = HuffmanRawWeightsDecoderFSM;
    type State = HuffmanRawWeightsDecoderState<
        AXI_ADDR_W, AXI_DATA_W, WEIGHTS_RAM_ADDR_W, WEIGHTS_RAM_DATA_W,
        BUFF_LEN, BUFF_LEN_LOG2,
    >;

    type MemReaderReq  = mem_reader::MemReaderReq<AXI_ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<AXI_DATA_W, AXI_ADDR_W>;

    type WeightsRamWrReq = ram::WriteReq<WEIGHTS_RAM_ADDR_W, WEIGHTS_RAM_DATA_W, WEIGHTS_RAM_NUM_PARTITIONS>;
    type WeightsRamWrResp = ram::WriteResp;

    // Control
    req_r: chan<Req> in;
    resp_s: chan<Resp> out;

    // MemReader interface for fetching Huffman Tree Description
    mem_rd_req_s: chan<MemReaderReq> out;
    mem_rd_resp_r: chan<MemReaderResp> in;

    weights_ram_wr_req_s: chan<WeightsRamWrReq> out;
    weights_ram_wr_resp_r: chan<WeightsRamWrResp> in;

    init {
        zero!<State>()
    }

    config(
        // Control
        req_r: chan<Req> in,
        resp_s: chan<Resp> out,

        // MemReader interface for fetching Huffman Tree Description
        mem_rd_req_s: chan<MemReaderReq> out,
        mem_rd_resp_r: chan<MemReaderResp> in,

        // RAM Write interface (goes to Huffman Weights Memory)
        weights_ram_wr_req_s: chan<WeightsRamWrReq> out,
        weights_ram_wr_resp_r: chan<WeightsRamWrResp> in,
    ) {

        (
            req_r, resp_s,
            mem_rd_req_s, mem_rd_resp_r,
            weights_ram_wr_req_s, weights_ram_wr_resp_r
        )
    }

    next (state: State) {
        let tok = join();

        // [IDLE]
        let (tok, req, req_valid) = recv_if_non_blocking(join(), req_r, state.fsm == FSM::IDLE, zero!<Req>());

        // Fetch Data
        let mem_rd_req = MemReaderReq {
            addr: req.addr + uN[AXI_ADDR_W]:1, // skip header
            length: ((req.n_symbols + u8:1) >> u32:1) as uN[AXI_ADDR_W], // ceil(number_of_symbols/2)
        };
        let tok = send_if(tok, mem_rd_req_s, req_valid, mem_rd_req);

        // [DECODING]
        let buffer = state.buffer;
        let buffer_len = state.buffer_len;

        // Buffer AXI data
        let do_recv_data = state.fsm == FSM::DECODING && buffer_len < (WEIGHTS_RAM_DATA_W as uN[BUFF_LEN_LOG2]);
        let (tok, mem_rd_resp, mem_rd_resp_valid) = recv_if_non_blocking(tok, mem_rd_resp_r, do_recv_data, zero!<MemReaderResp>());
        if do_recv_data && mem_rd_resp_valid {
            trace_fmt!("[RAW] Received MemReader response {:#x}", mem_rd_resp);
            trace_fmt!("[RAW] Data {:#x}", mem_rd_resp.data);
        } else {};

        const MAX_WEIGHTS_IN_PACKET = AXI_DATA_W >> u32:2;
        let weights = mem_rd_resp.data as u4[MAX_WEIGHTS_IN_PACKET];
        let sum = for (i, sum): (u32, u32) in u32:0..MAX_WEIGHTS_IN_PACKET {
            if (weights[i] != u4:0) {
                sum + (u32:1 << (weights[i] - u4:1))
            } else {
                sum
            }
        } (state.sum);

        let state = if (mem_rd_resp_valid) {
            State {
                sum: sum,
                ..state
            }
        } else {
            state
        };

        let next_power = u32:1 << std::clog2(state.sum);
        let last_weight = (next_power - state.sum) as u4;

        // It is required to change the ordering of the weights.
        // Huffman literals decoder expects the weight of the first symbol
        // as the most significant nibble at the most significant byte
        // in the first cell of the WeightsMemory.

        // Inject the last weight, take into the acount the reverse
        let weights = if (state.req.n_symbols > u8:0 && (mem_rd_resp_valid && mem_rd_resp.last)) {
            trace_fmt!("[RAW] The sum of weight's powers of 2's: {}", state.sum);
            trace_fmt!("[RAW] The last weight: {}", last_weight);
            trace_fmt!("[RAW] Injected {:#x} into weights[{}]", last_weight, MAX_WEIGHTS_IN_PACKET as u8 - state.req.n_symbols);
            update(weights, (MAX_WEIGHTS_IN_PACKET as u8 - (state.req.n_symbols % MAX_WEIGHTS_IN_PACKET as u8)), last_weight)
        } else {
            weights
        };

        let reversed_weights = match(AXI_DATA_W) {
            u32:32 => (
                weights[7] ++ weights[6] ++ weights[5] ++ weights[4] ++
                weights[3] ++ weights[2] ++ weights[1] ++ weights[0]
            ) as uN[AXI_DATA_W],
            u32:64 => (
                weights[15] ++ weights[14] ++ weights[13] ++ weights[12] ++
                weights[11] ++ weights[10] ++ weights[9]  ++ weights[8] ++
                weights[7]  ++ weights[6]  ++ weights[5]  ++ weights[4] ++
                weights[3]  ++ weights[2]  ++ weights[1]  ++ weights[0]
            ) as uN[AXI_DATA_W],
            _ => fail!("unsupported_axi_data_width", uN[AXI_DATA_W]:0),
        };

        if do_recv_data && mem_rd_resp_valid {
            trace_fmt!("[RAW] Weights: {:#x}", weights);
        } else {};

        if do_recv_data && mem_rd_resp_valid {
            trace_fmt!("[RAW] Weights: {:#x}", weights);
        } else {};

        let (buffer, buffer_len) = if do_recv_data && mem_rd_resp_valid {
            (
                buffer | ((reversed_weights as uN[BUFF_LEN] << (BUFF_LEN - AXI_DATA_W - buffer_len as u32))),
                buffer_len + (AXI_DATA_W as uN[BUFF_LEN_LOG2]),
            )
        } else {
            (
                buffer,
                buffer_len,
            )
        };
        // Send to RAM
        let do_send_data = state.fsm == FSM::DECODING && buffer_len >= (WEIGHTS_RAM_DATA_W as uN[BUFF_LEN_LOG2]);
        let weights_ram_wr_req = WeightsRamWrReq {
            addr: state.ram_addr,
            data: buffer[-(WEIGHTS_RAM_DATA_W as s32):] as uN[WEIGHTS_RAM_DATA_W],
            mask: !uN[WEIGHTS_RAM_NUM_PARTITIONS]:0,
        };
        let tok = send_if(tok, weights_ram_wr_req_s, do_send_data, weights_ram_wr_req);
        if do_send_data {
            trace_fmt!("[RAW] Buffer length: {}", buffer_len);
            trace_fmt!("[RAW] Sent RAM write request {:#x}", weights_ram_wr_req);
        } else  {};

        let (buffer, buffer_len, data_decoded) = if do_send_data {
            (
                buffer << WEIGHTS_RAM_DATA_W,
                buffer_len - (WEIGHTS_RAM_DATA_W as uN[BUFF_LEN_LOG2]),
                WEIGHTS_RAM_DATA_W as uN[AXI_ADDR_W],
            )
        } else {
            (
                buffer,
                buffer_len,
                uN[AXI_ADDR_W]:0,
            )
        };

        // [FILL_ZERO]
        let weights_ram_wr_req = WeightsRamWrReq {
            data: uN[WEIGHTS_RAM_DATA_W]:0,
            addr: state.ram_addr,
            mask: !uN[WEIGHTS_RAM_NUM_PARTITIONS]:0,
        };
        let rok = send_if(tok, weights_ram_wr_req_s, state.fsm == FSM::FILL_ZERO, weights_ram_wr_req);

        // [RESP]
        let tok = send_if(tok, resp_s, state.fsm == FSM::RESP, zero!<Resp>());

        // Update state
        let ram_wr_resp_to_handle = state.ram_wr_resp_to_handle + do_send_data as u4 + (state.fsm == FSM::FILL_ZERO) as u4;
        let (tok, _, weights_ram_wr_resp_valid) = recv_if_non_blocking(tok, weights_ram_wr_resp_r, ram_wr_resp_to_handle > u4:0, zero!<WeightsRamWrResp>());
        let state = if weights_ram_wr_resp_valid {
            State {
                ram_wr_resp_to_handle: ram_wr_resp_to_handle - u4:1,
                ..state
            }
        } else {
            State {
                ram_wr_resp_to_handle: ram_wr_resp_to_handle,
                ..state
            }
        };

        match state.fsm {
            FSM::IDLE => {
                if req_valid {
                    trace_fmt!("[RAW] Received decoding request {:#x}", req);
                    trace_fmt!("[RAW] Sent MemReader request {:#x}", mem_rd_req);
                    State {
                        fsm: FSM::DECODING,
                        req: req,
                        ..zero!<State>()
                    }
                } else {
                    state
                }
            },
            FSM::DECODING => {
                let data_to_be_decoded = uN[AXI_ADDR_W]:8 * (((state.req.n_symbols as uN[AXI_ADDR_W] + uN[AXI_ADDR_W]:1) >> u32:1));
                trace_fmt!("[RAW] Decoded {} / {}", state.data_decoded + data_decoded, data_to_be_decoded);
                trace_fmt!("[RAW] Buffer {:#x}", state.buffer);
                if state.data_decoded + data_decoded < data_to_be_decoded {
                    State {
                        data_decoded: state.data_decoded + data_decoded,
                        ram_addr: state.ram_addr + (data_decoded / WEIGHTS_RAM_DATA_W as uN[AXI_ADDR_W]) as uN[WEIGHTS_RAM_ADDR_W],
                        buffer: buffer,
                        buffer_len: buffer_len,
                        ..state
                    }
                } else {
                    State {
                        fsm: FSM::FILL_ZERO,
                        ram_addr: state.ram_addr + (data_decoded / WEIGHTS_RAM_DATA_W as uN[AXI_ADDR_W]) as uN[WEIGHTS_RAM_ADDR_W],
                        ..state
                    }
                }
            },
            FSM::FILL_ZERO => {
                if state.ram_addr < !uN[WEIGHTS_RAM_ADDR_W]:0 {
                    trace_fmt!("[RAW] Filling with zeros {} / {}", state.ram_addr + uN[WEIGHTS_RAM_ADDR_W]:1, !uN[WEIGHTS_RAM_ADDR_W]:0);
                    State {
                        ram_addr: state.ram_addr + uN[WEIGHTS_RAM_ADDR_W]:1,
                        ..state
                    }
                } else {
                    State {
                        fsm: FSM::RESP,
                        ..state
                    }
                }
            },
            FSM::RESP => {
                State {
                    fsm: FSM::IDLE,
                    ..state
                }
            },
            _ => fail!("impossible_state", state)
        }
    }
}


enum HuffmanFseDecoderStatus: u1 {
    OK = 0,
    ERROR = 1,
}

struct HuffmanFseDecoderCtrl {
    acc_log: uN[HUFFMAN_FSE_ACCURACY_W ],
    length: u8,
}

struct HuffmanFseDecoderFinish {
    status: HuffmanFseDecoderStatus,
}

type HuffmanFseTableRecord = common::FseTableRecord;

struct CommandConstructorData {}

enum HuffmanFseDecoderFSM : u4 {
    RECV_CTRL = 0,
    PADDING = 1,
    INIT_EVEN_STATE = 2,
    INIT_ODD_STATE = 3,
    SEND_RAM_EVEN_RD_REQ = 4,
    RECV_RAM_EVEN_RD_RESP = 5,
    SEND_RAM_ODD_RD_REQ = 6,
    RECV_RAM_ODD_RD_RESP = 7,
    UPDATE_EVEN_STATE = 8,
    UPDATE_ODD_STATE = 9,
    DECODE_LAST_WEIGHT = 10,
    SEND_WEIGHT = 11,
    SEND_WEIGHT_DONE = 12,
    FILL_ZEROS = 13,
    SEND_FINISH = 14,
}
struct HuffmanFseDecoderState<WEIGHTS_RAM_DATA_W: u32> {
    fsm: HuffmanFseDecoderFSM,
    ctrl: HuffmanFseDecoderCtrl,    // decode request
    even: u8,
    odd: u8,
    even_table_record: HuffmanFseTableRecord,   // FSE lookup record for even_state
    odd_table_record: HuffmanFseTableRecord,    // FSE lookup record for odd_state
    even_table_record_valid: bool,
    odd_table_record_valid: bool,
    even_state: u16,             // analogous to state1 in educational ZSTD decoder
    odd_state: u16,              // analogous to state1 in educational ZSTD decoder
                                 // https://github.com/facebook/zstd/blob/fe34776c207f3f879f386ed4158a38d927ff6d10/doc/educational_decoder/zstd_decompress.c#L2069
    read_bits_needed: u7,        // how many bits to request from the ShiftBuffer next
    sent_buf_ctrl: bool,         // have we sent request to ShiftBuffer in this FSM state already?
    shift_buffer_error: bool,    // sticky flag, asserted if ShiftBuffer returns error in data
                                 // payload, cleared when going to initial state
    padding: u4,                 // how much padding have we consumed (used for checking stream validity)
    current_iteration: u8,       // which iteration of the FSE-encoded-weights decoding loop are we in:
                                 // https://github.com/facebook/zstd/blob/fe34776c207f3f879f386ed4158a38d927ff6d10/doc/educational_decoder/zstd_decompress.c#L2081
    stream_len: u8,              // how long is the FSE-encoded-weights stream, in bytes
    stream_empty: bool,          // did we ask for more bits than available in the stream (i.e. caused stream underflow)?
                                 // analogous to 'offset < 0' check from educational ZSTD decoder:
                                 // https://github.com/facebook/zstd/blob/fe34776c207f3f879f386ed4158a38d927ff6d10/doc/educational_decoder/zstd_decompress.c#L2089
    last_weight: u1,             // whether the last weight is odd (last_weight == 1) or even (last_weight == 0) -
                                 // - analogue to whether we should end up here:
                                 // https://github.com/facebook/zstd/blob/fe34776c207f3f879f386ed4158a38d927ff6d10/doc/educational_decoder/zstd_decompress.c#L2091
                                 // ...or here:
                                 // https://github.com/facebook/zstd/blob/fe34776c207f3f879f386ed4158a38d927ff6d10/doc/educational_decoder/zstd_decompress.c#L2100
    weights_pow_of_two_sum: u32, // sum of 2^weight for all weights, needed to calculate last weight
    last_weight_decoded: bool,   // have we decoded last weight?
}

pub proc HuffmanFseDecoder<
    RAM_DATA_W: u32, RAM_ADDR_W: u32, RAM_NUM_PARTITIONS:u32,
    WEIGHTS_RAM_DATA_W: u32, WEIGHTS_RAM_ADDR_W: u32, WEIGHTS_RAM_NUM_PARTITIONS: u32,
    AXI_DATA_W: u32,
    REFILLING_SB_DATA_W: u32 = {AXI_DATA_W},
    REFILLING_SB_LENGTH_W: u32 = {refilling_shift_buffer::length_width(REFILLING_SB_DATA_W)},
> {
    type Ctrl = HuffmanFseDecoderCtrl;
    type Finish = HuffmanFseDecoderFinish;
    type Status = HuffmanFseDecoderStatus;
    type State = HuffmanFseDecoderState<WEIGHTS_RAM_DATA_W>;
    type FSM = HuffmanFseDecoderFSM;

    type FseRamRdReq = ram::ReadReq<RAM_ADDR_W, RAM_NUM_PARTITIONS>;
    type FseRamRdResp = ram::ReadResp<RAM_DATA_W>;

    type RefillingSBCtrl = refilling_shift_buffer::RefillingShiftBufferCtrl<REFILLING_SB_LENGTH_W>;
    type RefillingSBOutput = refilling_shift_buffer::RefillingShiftBufferOutput<REFILLING_SB_DATA_W, REFILLING_SB_LENGTH_W>;

    type WeightsRamWrReq = ram::WriteReq<WEIGHTS_RAM_ADDR_W, WEIGHTS_RAM_DATA_W, WEIGHTS_RAM_NUM_PARTITIONS>;
    type WeightsRamWrResp = ram::WriteResp;

    // Control
    ctrl_r: chan<Ctrl> in;
    finish_s: chan<Finish> out;

    // Shift buffer
    rsb_ctrl_s: chan<RefillingSBCtrl> out;
    rsb_data_r: chan<RefillingSBOutput> in;

    // FSE table RAMs
    table_rd_req_s: chan<FseRamRdReq> out;
    table_rd_resp_r: chan<FseRamRdResp> in;

    // Weights RAMs
    weights_wr_req_s: chan<WeightsRamWrReq> out;
    weights_wr_resp_r: chan<WeightsRamWrResp> in;

    config (
        ctrl_r: chan<Ctrl> in,
        finish_s: chan<Finish> out,
        rsb_ctrl_s: chan<RefillingSBCtrl> out,
        rsb_data_r: chan<RefillingSBOutput> in,
        table_rd_req_s: chan<FseRamRdReq> out,
        table_rd_resp_r: chan<FseRamRdResp> in,
        weights_wr_req_s: chan<WeightsRamWrReq> out,
        weights_wr_resp_r: chan<WeightsRamWrResp> in,
    ) {
        (
            ctrl_r, finish_s,
            rsb_ctrl_s, rsb_data_r,
            table_rd_req_s, table_rd_resp_r,
            weights_wr_req_s, weights_wr_resp_r,
        )
    }

    init { zero!<HuffmanFseDecoderState>() }

    next (state: HuffmanFseDecoderState) {
        type RamAddr = uN[RAM_ADDR_W];
        const RAM_MASK_ALL = std::unsigned_max_value<RAM_NUM_PARTITIONS>();

        let tok = join();

        // receive ctrl
        let (_, ctrl, ctrl_valid) = recv_if_non_blocking(tok, ctrl_r, state.fsm == FSM::RECV_CTRL, zero!<Ctrl>());
        if ctrl_valid {
            trace_fmt!("ctrl: {:#x}", ctrl);
        } else {};
        let state = if ctrl_valid {
            HuffmanFseDecoderState {
                ctrl: ctrl,
                stream_len: ctrl.length * u8:8,
                ..state
            }
        } else { state };

        // receive ram read response
        let do_recv_table_rd_resp = state.fsm == FSM::RECV_RAM_EVEN_RD_RESP || state.fsm == FSM::RECV_RAM_ODD_RD_RESP;
        let (_, table_rd_resp, table_rd_resp_valid) = recv_if_non_blocking(tok, table_rd_resp_r, do_recv_table_rd_resp, zero!<FseRamRdResp>());

        let table_record = fse_table_creator::bits_to_fse_record(table_rd_resp.data);

        if table_rd_resp_valid {
            trace_fmt!("table_record: {:#x}", table_record);
        } else {};

        // request records
        let do_send_ram_rd_req = state.fsm == FSM::SEND_RAM_EVEN_RD_REQ || state.fsm == FSM::SEND_RAM_ODD_RD_REQ;
        let ram_rd_req_addr = match (state.fsm) {
            FSM::SEND_RAM_EVEN_RD_REQ => state.even_state as RamAddr,
            FSM::SEND_RAM_ODD_RD_REQ => state.odd_state as RamAddr,
            _ => RamAddr:0,
        };

        let table_req = FseRamRdReq { addr: ram_rd_req_addr, mask: RAM_MASK_ALL };

        send_if(tok, table_rd_req_s, do_send_ram_rd_req, table_req);

        if do_send_ram_rd_req {
            trace_fmt!("table_req: {:#x}", table_req);
        } else {};

        // read bits
        let do_read_bits = (
            state.fsm == FSM::PADDING ||
            state.fsm == FSM::INIT_EVEN_STATE ||
            state.fsm == FSM::INIT_ODD_STATE ||
            state.fsm == FSM::UPDATE_EVEN_STATE ||
            state.fsm == FSM::UPDATE_ODD_STATE
        );
        let do_send_buf_ctrl = do_read_bits && !state.sent_buf_ctrl && state.stream_len > u8:0;

        let read_length = if state.read_bits_needed as u8 > state.stream_len {
            state.stream_len as u7
        } else {
            state.read_bits_needed
        };

        let state = if state.read_bits_needed > u7:0 {
            HuffmanFseDecoderState {
                stream_empty: state.read_bits_needed as u8 > state.stream_len,
                ..state
            }
        } else { state };

        if do_send_buf_ctrl {
            trace_fmt!("[FseDecoder] Asking for {:#x} data", read_length);
        } else {};

        send_if(tok, rsb_ctrl_s, do_send_buf_ctrl, RefillingSBCtrl {
            length: read_length,
        });

        let state = if do_send_buf_ctrl {
            HuffmanFseDecoderState { sent_buf_ctrl: do_send_buf_ctrl, ..state }
        } else { state };

        let recv_sb_output = (do_read_bits && state.sent_buf_ctrl);
        let (_, buf_data, buf_data_valid) = recv_if_non_blocking(tok, rsb_data_r, recv_sb_output, zero!<RefillingSBOutput>());
        if buf_data_valid && buf_data.length as u32 > u32:0{
            trace_fmt!("[FseDecoder] Received data {:#x} in state {}", buf_data, state.fsm);
        } else { };

        let state = if do_read_bits & buf_data_valid {
            HuffmanFseDecoderState {
                sent_buf_ctrl: false,
                shift_buffer_error: state.shift_buffer_error | buf_data.error,
                stream_len: state.stream_len - buf_data.length as u8,
                ..state
            }
        } else { state };

        // decode last weight
        let max_bits = common::highest_set_bit(state.weights_pow_of_two_sum) + u32:1;
        let next_power = u32:1 << max_bits;
        let left_over = (next_power - state.weights_pow_of_two_sum);
        let last_weight = (common::highest_set_bit(left_over) + u32:1) as u8;

        // write weight
        const WEIGHTS_RAM_BYTES = WEIGHTS_RAM_DATA_W as u8 / u8:8;
        let iter_mod4_inv = u8:3 - (state.current_iteration & u8:0x3);
        let weight = (((state.even as u8) << u32:4) & u8:0xF0) | ((state.odd as u8) & u8:0x0F);
        let weights_wr_req = WeightsRamWrReq {
            addr: (state.current_iteration / WEIGHTS_RAM_BYTES) as uN[WEIGHTS_RAM_ADDR_W],
            data: (weight as uN[WEIGHTS_RAM_DATA_W] << (u8:8 * iter_mod4_inv)),
            // mask appropriate byte in 32-bit word with 4-bit slices
            mask: uN[WEIGHTS_RAM_NUM_PARTITIONS]:0x3 << (u8:2 * iter_mod4_inv),
        };
        let tok = send_if(tok, weights_wr_req_s, state.fsm == FSM::SEND_WEIGHT, weights_wr_req);
        if (state.fsm == FSM::SEND_WEIGHT) {
            trace_fmt!("Sent weight to RAM: {:#x}", weights_wr_req);
        } else {};

        let (tok, _, weights_wr_resp_valid) = recv_if_non_blocking(
            tok, weights_wr_resp_r, state.fsm == FSM::SEND_WEIGHT_DONE, zero!<WeightsRamWrResp>()
        );

        // send finish
        send_if(tok, finish_s, state.fsm == FSM::SEND_FINISH, Finish {
            status: if state.shift_buffer_error { Status::ERROR } else { Status::OK }
        });

        // update state
        match (state.fsm) {
            FSM::RECV_CTRL => {
                if (ctrl_valid) {
                    trace_fmt!("[FseDecoder] Moving to PADDING");
                    State {
                        fsm: FSM::PADDING,
                        ctrl: ctrl,
                        read_bits_needed: u7:1,
                        ..state
                    }
                } else { state }
            },
            FSM::PADDING => {
                if (buf_data_valid) {
                    let padding = state.padding + u4:1;
                    assert!(padding <= u4:8, "invalid_padding");

                    let padding_available = (buf_data.data as u1 == u1:0);
                    if padding_available {
                        State {
                            fsm: FSM::PADDING,
                            read_bits_needed: u7:1,
                            padding, ..state
                        }
                    } else {
                        trace_fmt!("[FseDecoder] Moving to INIT_LOOKUP_STATE");
                        trace_fmt!("padding is: {:#x}", padding);
                        State {
                            fsm: FSM::INIT_EVEN_STATE,
                            read_bits_needed: state.ctrl.acc_log as u7,
                            ..state
                        }
                    }
                } else { state }
            },
            FSM::INIT_EVEN_STATE => {
                if (buf_data_valid) {
                    trace_fmt!("[FseDecoder] Moving to INIT_ODD_STATE");
                    State {
                        fsm: FSM::INIT_ODD_STATE,
                        even_state: buf_data.data as u16,
                        read_bits_needed: state.ctrl.acc_log as u7,
                        ..state
                    }
                } else { state }
            },
            FSM::INIT_ODD_STATE => {
                if (buf_data_valid) {
                    trace_fmt!("[FseDecoder] Moving to SEND_RAM_EVEN_RD_REQ");
                    State {
                        fsm: FSM::SEND_RAM_EVEN_RD_REQ,
                        odd_state: buf_data.data as u16,
                        read_bits_needed: u7:0,
                        ..state
                    }
                } else { state }
            },
            FSM::SEND_RAM_EVEN_RD_REQ => {
                trace_fmt!("[FseDecoder] Moving to RECV_RAM_EVEN_RD_RESP");
                trace_fmt!("State even: {:#x}", state.even_state);
                State {
                    fsm: FSM::RECV_RAM_EVEN_RD_RESP,
                    even_table_record_valid: false,
                    ..state
                }
            },
            FSM::RECV_RAM_EVEN_RD_RESP => {
                // save fse records in state
                let state = if table_rd_resp_valid {
                    State { even_table_record: table_record, even_table_record_valid: true, ..state }
                } else { state };

                if state.even_table_record_valid {
                    let symbol = state.even_table_record.symbol;
                    let pow = if symbol != u8:0 {
                        u32:1 << (symbol - u8:1)
                    } else {
                        u32:0
                    };
                    if state.stream_empty {
                        trace_fmt!("[FseDecoder] Moving to DECODE_LAST_WEIGHT");
                        State {
                            fsm: FSM::DECODE_LAST_WEIGHT,
                            even: symbol,
                            weights_pow_of_two_sum: state.weights_pow_of_two_sum + pow,
                            ..state
                        }
                    } else {
                        trace_fmt!("[FseDecoder] Moving to SEND_RAM_ODD_RD_REQ");
                        State {
                            fsm: FSM::SEND_RAM_ODD_RD_REQ,
                            even: symbol,
                            weights_pow_of_two_sum: state.weights_pow_of_two_sum + pow,
                            ..state
                        }
                    }
                } else { state }
            },
            FSM::SEND_RAM_ODD_RD_REQ => {
                trace_fmt!("[FseDecoder] Moving to RECV_RAM_ODD_RD_RESP");
                trace_fmt!("State odd: {:#x}", state.odd_state);
                State {
                    fsm: FSM::RECV_RAM_ODD_RD_RESP,
                    odd_table_record_valid: false,
                    ..state
                }
            },
            FSM::RECV_RAM_ODD_RD_RESP => {
                // save fse records in state
                let state = if table_rd_resp_valid {
                    State { odd_table_record: table_record, odd_table_record_valid: true, ..state }
                } else { state };

                if state.odd_table_record_valid {
                    let symbol = state.odd_table_record.symbol;
                    let pow = if symbol != u8:0 {
                        u32:1 << (symbol - u8:1)
                    } else {
                        u32:0
                    };
                    if state.stream_empty {
                        trace_fmt!("[FseDecoder] Moving to SEND_WEIGHT");
                        State {
                            fsm: FSM::DECODE_LAST_WEIGHT,
                            odd: symbol,
                            weights_pow_of_two_sum: state.weights_pow_of_two_sum + pow,
                            ..state
                        }
                    } else {
                        trace_fmt!("[FseDecoder] Moving to UPDATE_EVEN_STATE");
                        State {
                            fsm: FSM::UPDATE_EVEN_STATE,
                            odd: state.odd_table_record.symbol,
                            weights_pow_of_two_sum: state.weights_pow_of_two_sum + pow,
                            read_bits_needed: state.even_table_record.num_of_bits as u7,
                            ..state
                        }
                    }
                } else { state }
            },
            FSM::UPDATE_EVEN_STATE => {
                if state.stream_empty {
                    trace_fmt!("[FseDecoder] Moving to SEND_WEIGHT");
                    State {
                        fsm: FSM::SEND_WEIGHT,
                        even_state: state.even_table_record.base + buf_data.data as u16,
                        read_bits_needed: state.odd_table_record.num_of_bits as u7,
                        last_weight: u1:0,
                        ..state
                    }
                } else if buf_data_valid || state.stream_len == u8:0 {
                    trace_fmt!("[FseDecoder] Moving to UPDATE_ODD_STATE");
                    State {
                        fsm: FSM::UPDATE_ODD_STATE,
                        even_state: state.even_table_record.base + buf_data.data as u16,
                        read_bits_needed: state.odd_table_record.num_of_bits as u7,
                        ..state
                    }
                } else { state }
            },
            FSM::UPDATE_ODD_STATE => {
                if state.stream_empty {
                    trace_fmt!("[FseDecoder] Moving to SEND_WEIGHT");
                    State {
                        fsm: FSM::SEND_WEIGHT,
                        odd_state: state.odd_table_record.base + buf_data.data as u16,
                        read_bits_needed: u7:0,
                        last_weight: u1:1,
                        ..state
                    }
                } else if buf_data_valid || state.stream_len == u8:0 {
                    trace_fmt!("[FseDecoder] Moving to SEND_WEIGHT");
                    State {
                        fsm: FSM::SEND_WEIGHT,
                        odd_state: state.odd_table_record.base + buf_data.data as u16,
                        read_bits_needed: u7:0,
                        ..state
                    }
                } else { state }
            },
            FSM::DECODE_LAST_WEIGHT => {
                trace_fmt!("[FseDecoder] Moving to SEND_WEIGHT");
                trace_fmt!("[FseDecoder] Last weight {:#x}, weights^2: {}, max_bits: {}, left_over: {}, iteration {}", last_weight, state.weights_pow_of_two_sum, max_bits, left_over, state.current_iteration);
                if state.last_weight == u1:0 {
                    State {
                        fsm: FSM::SEND_WEIGHT,
                        even: last_weight,
                        odd: u8:0,
                        last_weight_decoded: true,
                        ..state
                    }
                } else {
                    State {
                        fsm: FSM::SEND_WEIGHT,
                        // even weight should be kept unchanged
                        odd: last_weight,
                        last_weight_decoded: true,
                        ..state
                    }
                }
            },
            FSM::SEND_WEIGHT => {
                trace_fmt!("[FseDecoder] Current iteration: {}, weights: {} {} {}", state.current_iteration, state.even_state, state.odd_state, weight);
                State {
                    fsm: FSM::SEND_WEIGHT_DONE,
                    current_iteration: state.current_iteration + u8:1,
                    ..state
                }
            },
            FSM::SEND_WEIGHT_DONE => {
                if weights_wr_resp_valid {
                    trace_fmt!("Weights write done");
                    let fsm = if state.stream_empty {
                        if state.last_weight_decoded {
                            FSM::FILL_ZEROS
                        } else {
                            // get second-to-last weight
                            if state.last_weight == u1:1 {
                                FSM::SEND_RAM_EVEN_RD_REQ
                            } else {
                                FSM::SEND_RAM_ODD_RD_REQ
                            }
                        }
                    } else {
                        FSM::SEND_RAM_EVEN_RD_REQ
                    };
                    State {
                        fsm: fsm,
                        ..state
                    }
                } else { state }
            },
            FSM::FILL_ZEROS => {
                if state.current_iteration == u8:0x7F {
                    State {
                        fsm: FSM::SEND_FINISH,
                        ..state
                    }
                } else {
                    State {
                        fsm: FSM::SEND_WEIGHT,
                        even: u8:0,
                        odd: u8:0,
                        ..state
                    }
                }
            },
            FSM::SEND_FINISH => {
                trace_fmt!("[FseDecoder] Moving to RECV_CTRL");
                State {
                    fsm:FSM::RECV_CTRL,
                    ..zero!<State>()
                }
            },
            _ => {
                fail!("impossible_case", state)
            },
        }
    }
}

struct HuffmanFseWeightsDecoderReq<AXI_ADDR_W: u32> {
    addr: uN[AXI_ADDR_W],
    length: u8,
}

enum HuffmanFseWeightsDecoderStatus: u1 {
    OKAY = 0,
    ERROR = 1,
}

struct HuffmanFseWeightsDecoderResp {
    status: HuffmanFseWeightsDecoderStatus,
}

struct HuffmanFseWeightsDecoderState { }

proc HuffmanFseWeightsDecoder<
    AXI_ADDR_W: u32, AXI_DATA_W: u32, AXI_ID_W: u32,
    WEIGHTS_RAM_ADDR_W: u32, WEIGHTS_RAM_DATA_W: u32, WEIGHTS_RAM_NUM_PARTITIONS: u32,
    DPD_RAM_ADDR_W: u32, DPD_RAM_DATA_W: u32, DPD_RAM_NUM_PARTITIONS: u32,
    TMP_RAM_ADDR_W: u32, TMP_RAM_DATA_W: u32, TMP_RAM_NUM_PARTITIONS: u32,
    TMP2_RAM_ADDR_W: u32, TMP2_RAM_DATA_W: u32, TMP2_RAM_NUM_PARTITIONS: u32,
    FSE_RAM_ADDR_W: u32, FSE_RAM_DATA_W: u32, FSE_RAM_NUM_PARTITIONS: u32,
    REFILLING_SB_DATA_W: u32 = {AXI_DATA_W},
    REFILLING_SB_LENGTH_W: u32 = {refilling_shift_buffer::length_width(AXI_DATA_W)},
> {
    type Req = HuffmanFseWeightsDecoderReq<AXI_ADDR_W>;
    type Resp = HuffmanFseWeightsDecoderResp;
    type Status = HuffmanFseWeightsDecoderStatus;
    type State = HuffmanFseWeightsDecoderState;

    type MemReaderReq  = mem_reader::MemReaderReq<AXI_ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<AXI_DATA_W, AXI_ADDR_W>;

    type WeightsRamWrReq = ram::WriteReq<WEIGHTS_RAM_ADDR_W, WEIGHTS_RAM_DATA_W, WEIGHTS_RAM_NUM_PARTITIONS>;
    type WeightsRamWrResp = ram::WriteResp;

    type CompLookupDecoderReq =  comp_lookup_dec::CompLookupDecoderReq<AXI_ADDR_W>;
    type CompLookupDecoderResp = comp_lookup_dec::CompLookupDecoderResp;

    type DpdRamRdReq = ram::ReadReq<DPD_RAM_ADDR_W, DPD_RAM_NUM_PARTITIONS>;
    type DpdRamRdResp = ram::ReadResp<DPD_RAM_DATA_W>;
    type DpdRamWrReq = ram::WriteReq<DPD_RAM_ADDR_W, DPD_RAM_DATA_W, DPD_RAM_NUM_PARTITIONS>;
    type DpdRamWrResp = ram::WriteResp;

    type TmpRamRdReq = ram::ReadReq<TMP_RAM_ADDR_W, TMP_RAM_NUM_PARTITIONS>;
    type TmpRamRdResp = ram::ReadResp<TMP_RAM_DATA_W>;
    type TmpRamWrReq = ram::WriteReq<TMP_RAM_ADDR_W, TMP_RAM_DATA_W, TMP_RAM_NUM_PARTITIONS>;
    type TmpRamWrResp = ram::WriteResp;

    type Tmp2RamRdReq = ram::ReadReq<TMP2_RAM_ADDR_W, TMP2_RAM_NUM_PARTITIONS>;
    type Tmp2RamRdResp = ram::ReadResp<TMP2_RAM_DATA_W>;
    type Tmp2RamWrReq = ram::WriteReq<TMP2_RAM_ADDR_W, TMP2_RAM_DATA_W, TMP2_RAM_NUM_PARTITIONS>;
    type Tmp2RamWrResp = ram::WriteResp;

    type FseRamRdReq = ram::ReadReq<FSE_RAM_ADDR_W, FSE_RAM_NUM_PARTITIONS>;
    type FseRamRdResp = ram::ReadResp<FSE_RAM_DATA_W>;
    type FseRamWrReq = ram::WriteReq<FSE_RAM_ADDR_W, FSE_RAM_DATA_W, FSE_RAM_NUM_PARTITIONS>;
    type FseRamWrResp = ram::WriteResp;

    type RefillingShiftBufferStart = refilling_shift_buffer::RefillStart<AXI_ADDR_W>;
    type RefillingShiftBufferError = refilling_shift_buffer::RefillingShiftBufferInput<REFILLING_SB_DATA_W, REFILLING_SB_LENGTH_W>;
    type RefillingShiftBufferOutput = refilling_shift_buffer::RefillingShiftBufferOutput<REFILLING_SB_DATA_W, REFILLING_SB_LENGTH_W>;
    type RefillingShiftBufferCtrl = refilling_shift_buffer::RefillingShiftBufferCtrl<REFILLING_SB_LENGTH_W>;

    type MemAxiAr = axi::AxiAr<AXI_ADDR_W, AXI_ID_W>;
    type MemAxiR = axi::AxiR<AXI_DATA_W, AXI_ID_W>;

    // Control
    req_r: chan<Req> in;
    resp_s: chan<Resp> out;

    // Refilling shift buffer for lookup decoder
    fld_rsb_start_req_s: chan<RefillingShiftBufferStart> out;
    fld_rsb_stop_flush_req_s: chan<()> out;
    fld_rsb_flushing_done_r: chan<()> in;

    // Refilling shift buffer for FSE decoder
    fd_rsb_start_req_s: chan<RefillingShiftBufferStart> out;
    fd_rsb_stop_flush_req_s: chan<()> out;
    fd_rsb_flushing_done_r: chan<()> in;

    // FSE Lookup Decoder
    fld_req_s: chan<CompLookupDecoderReq> out;
    fld_resp_r: chan<CompLookupDecoderResp> in;

    // Huffman FSE Decoder
    fd_ctrl_s: chan<HuffmanFseDecoderCtrl> out;
    fd_finish_r: chan<HuffmanFseDecoderFinish> in;

    init {
        zero!<State>()
    }

    config(
        // Control
        req_r: chan<Req> in,
        resp_s: chan<Resp> out,

        // MemReader interface for fetching Huffman Tree Description
        lookup_mem_rd_req_s: chan<MemReaderReq> out,
        lookup_mem_rd_resp_r: chan<MemReaderResp> in,
        decoder_mem_rd_req_s: chan<MemReaderReq> out,
        decoder_mem_rd_resp_r: chan<MemReaderResp> in,

        // RAM Write interface (goes to Huffman Weights Memory)
        weights_ram_wr_req_s: chan<WeightsRamWrReq> out,
        weights_ram_wr_resp_r: chan<WeightsRamWrResp> in,

        // FSE RAMs
        dpd_rd_req_s: chan<DpdRamRdReq> out,
        dpd_rd_resp_r: chan<DpdRamRdResp> in,
        dpd_wr_req_s: chan<DpdRamWrReq> out,
        dpd_wr_resp_r: chan<DpdRamWrResp> in,

        tmp_rd_req_s: chan<TmpRamRdReq> out,
        tmp_rd_resp_r: chan<TmpRamRdResp> in,
        tmp_wr_req_s: chan<TmpRamWrReq> out,
        tmp_wr_resp_r: chan<TmpRamWrResp> in,

        tmp2_rd_req_s: chan<Tmp2RamRdReq> out,
        tmp2_rd_resp_r: chan<Tmp2RamRdResp> in,
        tmp2_wr_req_s: chan<Tmp2RamWrReq> out,
        tmp2_wr_resp_r: chan<Tmp2RamWrResp> in,

        fse_rd_req_s: chan<FseRamRdReq> out,
        fse_rd_resp_r: chan<FseRamRdResp> in,
        fse_wr_req_s: chan<FseRamWrReq> out,
        fse_wr_resp_r: chan<FseRamWrResp> in,
    ) {
        const CHANNEL_DEPTH = u32:1;

        // CompLookupDecoder
        let (fld_rsb_start_req_s, fld_rsb_start_req_r) = chan<RefillingShiftBufferStart, CHANNEL_DEPTH>("fd_rsb_start_req");
        let (fld_rsb_stop_flush_req_s, fld_rsb_stop_flush_req_r) = chan<(), CHANNEL_DEPTH>("fd_rsb_stop_flush_req");
        let (fld_rsb_ctrl_s, fld_rsb_ctrl_r) = chan<RefillingShiftBufferCtrl, CHANNEL_DEPTH>("fd_rsb_ctrl");
        let (fld_rsb_data_s, fld_rsb_data_r) = chan<RefillingShiftBufferOutput, CHANNEL_DEPTH>("fd_rsb_data");
        let (fld_rsb_flushing_done_s, fld_rsb_flushing_done_r) = chan<(), CHANNEL_DEPTH>("fd_rsb_flushing_done");

        spawn refilling_shift_buffer::RefillingShiftBuffer<AXI_DATA_W, AXI_ADDR_W, false, u32:0xFF> (
            lookup_mem_rd_req_s, lookup_mem_rd_resp_r,
            fld_rsb_start_req_r, fld_rsb_stop_flush_req_r,
            fld_rsb_ctrl_r, fld_rsb_data_s,
            fld_rsb_flushing_done_s,
        );

        let (fld_req_s, fld_req_r) = chan<CompLookupDecoderReq, CHANNEL_DEPTH>("fse_req");
        let (fld_resp_s, fld_resp_r) = chan<CompLookupDecoderResp, CHANNEL_DEPTH>("fse_resp");

        spawn comp_lookup_dec::CompLookupDecoder<
            AXI_DATA_W,
            DPD_RAM_DATA_W, DPD_RAM_ADDR_W, DPD_RAM_NUM_PARTITIONS,
            TMP_RAM_DATA_W, TMP_RAM_ADDR_W, TMP_RAM_NUM_PARTITIONS,
            TMP2_RAM_DATA_W, TMP2_RAM_ADDR_W, TMP2_RAM_NUM_PARTITIONS,
            FSE_RAM_DATA_W, FSE_RAM_ADDR_W, FSE_RAM_NUM_PARTITIONS,
        >(
            fld_req_r, fld_resp_s,
            dpd_rd_req_s, dpd_rd_resp_r, dpd_wr_req_s, dpd_wr_resp_r,
            tmp_rd_req_s, tmp_rd_resp_r, tmp_wr_req_s, tmp_wr_resp_r,
            tmp2_rd_req_s, tmp2_rd_resp_r, tmp2_wr_req_s, tmp2_wr_resp_r,
            fse_wr_req_s, fse_wr_resp_r,
            fld_rsb_ctrl_s, fld_rsb_data_r,
        );

        // Huffman FSE Decoder
        let (fd_rsb_start_req_s, fd_rsb_start_req_r) = chan<RefillingShiftBufferStart, CHANNEL_DEPTH>("fd_rsb_start_req");
        let (fd_rsb_stop_flush_req_s, fd_rsb_stop_flush_req_r) = chan<(), CHANNEL_DEPTH>("fd_rsb_stop_flush_req");
        let (fd_rsb_ctrl_s, fd_rsb_ctrl_r) = chan<RefillingShiftBufferCtrl, CHANNEL_DEPTH>("fd_rsb_ctrl");
        let (fd_rsb_data_s, fd_rsb_data_r) = chan<RefillingShiftBufferOutput, CHANNEL_DEPTH>("fd_rsb_data");
        let (fd_rsb_flushing_done_s, fd_rsb_flushing_done_r) = chan<(), CHANNEL_DEPTH>("fd_rsb_flushing_done");

        spawn refilling_shift_buffer::RefillingShiftBuffer<AXI_DATA_W, AXI_ADDR_W, true, u32:0xFE> (
            decoder_mem_rd_req_s, decoder_mem_rd_resp_r,
            fd_rsb_start_req_r, fd_rsb_stop_flush_req_r,
            fd_rsb_ctrl_r, fd_rsb_data_s,
            fd_rsb_flushing_done_s,
        );

        let (fd_ctrl_s, fd_ctrl_r) = chan<HuffmanFseDecoderCtrl, CHANNEL_DEPTH>("fd_ctrl");
        let (fd_finish_s, fd_finish_r) = chan<HuffmanFseDecoderFinish, CHANNEL_DEPTH>("fd_finish");

        spawn HuffmanFseDecoder<
            FSE_RAM_DATA_W, FSE_RAM_ADDR_W, FSE_RAM_NUM_PARTITIONS,
            WEIGHTS_RAM_DATA_W, WEIGHTS_RAM_ADDR_W, WEIGHTS_RAM_NUM_PARTITIONS,
            AXI_DATA_W,
        >(
            fd_ctrl_r, fd_finish_s,
            fd_rsb_ctrl_s, fd_rsb_data_r,
            fse_rd_req_s, fse_rd_resp_r,
            weights_ram_wr_req_s, weights_ram_wr_resp_r,
        );

        (
            req_r, resp_s,
            fld_rsb_start_req_s, fld_rsb_stop_flush_req_s, fld_rsb_flushing_done_r,
            fd_rsb_start_req_s, fd_rsb_stop_flush_req_s, fd_rsb_flushing_done_r,
            fld_req_s, fld_resp_r,
            fd_ctrl_s, fd_finish_r,
        )
    }

    next (state: State) {
        let tok = join();

        // Receive decoding request
        let (tok, req) = recv(tok, req_r);
        trace_fmt!("[FSE] Received decoding request {:#x}", req);

        // Decode lookup
        let fld_rsb_start_req = RefillingShiftBufferStart {
            start_addr: req.addr + uN[AXI_ADDR_W]:1, // skip header byte
        };
        let tok = send(tok, fld_rsb_start_req_s, fld_rsb_start_req);
        trace_fmt!("[FSE] Sent refilling shift buffer start request {:#x}", fld_rsb_start_req);

        let fld_req = CompLookupDecoderReq {};
        let tok = send(tok, fld_req_s, fld_req);
        trace_fmt!("[FSE] Sent FSE lookup decoding request {:#x}", fld_req);

        let (tok, fld_resp) = recv(tok, fld_resp_r);
        trace_fmt!("[FSE] Received FSE lookup decoding response {:#x}", fld_resp);

        let tok = send(tok, fld_rsb_stop_flush_req_s, ());
        trace_fmt!("[FSE] Sent refilling shift buffer stop flush request");

        let (tok, _) = recv(tok, fld_rsb_flushing_done_r);
        trace_fmt!("[FSE] Received refilling shift buffer flushing done");

        // Decode weights
        let fd_rsb_start_req = RefillingShiftBufferStart {
            start_addr: req.addr + uN[AXI_ADDR_W]:1 + req.length as uN[AXI_ADDR_W]
        };
        let tok = send(tok, fd_rsb_start_req_s, fd_rsb_start_req);
        trace_fmt!("[FSE] Sent refilling shift buffer start request {:#x}", fd_rsb_start_req);

        let fd_ctrl = HuffmanFseDecoderCtrl {
            length: req.length - checked_cast<u8>(fld_resp.consumed_bytes),
            acc_log: fld_resp.accuracy_log,
        };
        let tok = send(tok, fd_ctrl_s, fd_ctrl);
        trace_fmt!("[FSE] Sent FSE decoding request {:#x}", fd_ctrl);

        let (tok, fd_finish) = recv(tok, fd_finish_r);
        trace_fmt!("[FSE] Received FSE decoding finish {:#x}", fd_finish);

        let tok = send(tok, fd_rsb_stop_flush_req_s, ());
        trace_fmt!("[FSE] Sent refilling shift buffer stop flush request");

        let (tok, _) = recv(tok, fd_rsb_flushing_done_r);
        trace_fmt!("[FSE] Received refilling shift buffer flushing done");

        // Send decoding response
        let tok = send(tok, resp_s, zero!<Resp>());

        zero!<State>()
    }
}

pub struct HuffmanWeightsDecoderReq<AXI_ADDR_W: u32> {
    addr: uN[AXI_ADDR_W],
}

pub enum HuffmanWeightsDecoderStatus: u1 {
    OKAY = 0,
    ERROR = 1,
}

enum WeightsType: u1 {
    RAW = 0,
    FSE = 1,
}

pub struct HuffmanWeightsDecoderResp<AXI_ADDR_W: u32> {
    status: HuffmanWeightsDecoderStatus,
    tree_description_size: uN[AXI_ADDR_W],
}

pub struct HuffmanWeightsDecoderState<AXI_ADDR_W: u32> {
    req: HuffmanWeightsDecoderReq<AXI_ADDR_W>,
}

pub proc HuffmanWeightsDecoder<
    AXI_ADDR_W: u32, AXI_DATA_W: u32, AXI_ID_W: u32,
    WEIGHTS_RAM_ADDR_W: u32, WEIGHTS_RAM_DATA_W: u32, WEIGHTS_RAM_NUM_PARTITIONS: u32,
    DPD_RAM_ADDR_W: u32, DPD_RAM_DATA_W: u32, DPD_RAM_NUM_PARTITIONS: u32,
    TMP_RAM_ADDR_W: u32, TMP_RAM_DATA_W: u32, TMP_RAM_NUM_PARTITIONS: u32,
    TMP2_RAM_ADDR_W: u32, TMP2_RAM_DATA_W: u32, TMP2_RAM_NUM_PARTITIONS: u32,
    FSE_RAM_ADDR_W: u32, FSE_RAM_DATA_W: u32, FSE_RAM_NUM_PARTITIONS: u32,
> {
    type Req = HuffmanWeightsDecoderReq<AXI_ADDR_W>;
    type Resp = HuffmanWeightsDecoderResp<AXI_ADDR_W>;
    type Status = HuffmanWeightsDecoderStatus;
    type State = HuffmanWeightsDecoderState<AXI_ADDR_W>;

    type MemReaderReq  = mem_reader::MemReaderReq<AXI_ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<AXI_DATA_W, AXI_ADDR_W>;
    type WeightsRamRdReq = ram::ReadReq<WEIGHTS_RAM_ADDR_W, WEIGHTS_RAM_NUM_PARTITIONS>;
    type WeightsRamRdResp = ram::ReadResp<WEIGHTS_RAM_DATA_W>;
    type WeightsRamWrReq = ram::WriteReq<WEIGHTS_RAM_ADDR_W, WEIGHTS_RAM_DATA_W, WEIGHTS_RAM_NUM_PARTITIONS>;
    type WeightsRamWrResp = ram::WriteResp;

    // Types used internally
    type RawWeightsReq = HuffmanRawWeightsDecoderReq<AXI_ADDR_W>;
    type RawWeightsResp = HuffmanRawWeightsDecoderResp;
    type FseWeightsReq = HuffmanFseWeightsDecoderReq<AXI_ADDR_W>;
    type FseWeightsResp = HuffmanFseWeightsDecoderResp;

    // FSE RAMs
    type DpdRamRdReq = ram::ReadReq<DPD_RAM_ADDR_W, DPD_RAM_NUM_PARTITIONS>;
    type DpdRamRdResp = ram::ReadResp<DPD_RAM_DATA_W>;
    type DpdRamWrReq = ram::WriteReq<DPD_RAM_ADDR_W, DPD_RAM_DATA_W, DPD_RAM_NUM_PARTITIONS>;
    type DpdRamWrResp = ram::WriteResp;

    type TmpRamRdReq = ram::ReadReq<TMP_RAM_ADDR_W, TMP_RAM_NUM_PARTITIONS>;
    type TmpRamRdResp = ram::ReadResp<TMP_RAM_DATA_W>;
    type TmpRamWrReq = ram::WriteReq<TMP_RAM_ADDR_W, TMP_RAM_DATA_W, TMP_RAM_NUM_PARTITIONS>;
    type TmpRamWrResp = ram::WriteResp;

    type Tmp2RamRdReq = ram::ReadReq<TMP2_RAM_ADDR_W, TMP2_RAM_NUM_PARTITIONS>;
    type Tmp2RamRdResp = ram::ReadResp<TMP2_RAM_DATA_W>;
    type Tmp2RamWrReq = ram::WriteReq<TMP2_RAM_ADDR_W, TMP2_RAM_DATA_W, TMP2_RAM_NUM_PARTITIONS>;
    type Tmp2RamWrResp = ram::WriteResp;

    type FseRamRdReq = ram::ReadReq<FSE_RAM_ADDR_W, FSE_RAM_NUM_PARTITIONS>;
    type FseRamRdResp = ram::ReadResp<FSE_RAM_DATA_W>;
    type FseRamWrReq = ram::WriteReq<FSE_RAM_ADDR_W, FSE_RAM_DATA_W, FSE_RAM_NUM_PARTITIONS>;
    type FseRamWrResp = ram::WriteResp;

    type MemAxiAr = axi::AxiAr<AXI_ADDR_W, AXI_ID_W>;
    type MemAxiR = axi::AxiR<AXI_DATA_W, AXI_ID_W>;

    // Control
    req_r: chan<Req> in;
    resp_s: chan<Resp> out;

    // MemReader interface for fetching Huffman Tree Description
    header_mem_rd_req_s: chan<MemReaderReq> out;
    header_mem_rd_resp_r: chan<MemReaderResp> in;

    // Select for RamMux
    decoded_weights_sel_s: chan<u1> out;

    // Raw Huffman Tree Description Decoder control
    raw_weights_req_s: chan<RawWeightsReq> out;
    raw_weights_resp_r: chan<RawWeightsResp> in;

    // Fse Huffman Tree Description Decoder control
    fse_weights_req_s: chan<FseWeightsReq> out;
    fse_weights_resp_r: chan<FseWeightsResp> in;

    // Fake ram read request channels (required by RamMux)
    raw_weights_ram_rd_req_s: chan<WeightsRamRdReq> out;
    raw_weights_ram_rd_resp_r: chan<WeightsRamRdResp> in;
    fse_weights_ram_rd_req_s: chan<WeightsRamRdReq> out;
    fse_weights_ram_rd_resp_r: chan<WeightsRamRdResp> in;
    weights_ram_rd_req_r: chan<WeightsRamRdReq> in;
    weights_ram_rd_resp_s: chan<WeightsRamRdResp> out;

    init {
        zero!<State>()
    }

    config(
        // Control
        req_r: chan<Req> in,
        resp_s: chan<Resp> out,

        // MemReader interface for fetching Huffman Tree Description
        header_mem_rd_req_s: chan<MemReaderReq> out,
        header_mem_rd_resp_r: chan<MemReaderResp> in,

        // MemReader interface for Raw Huffman Tree Description Decoder
        raw_weights_mem_rd_req_s: chan<MemReaderReq> out,
        raw_weights_mem_rd_resp_r: chan<MemReaderResp> in,

        // MemReader interface for Fse Huffman Tree Description Decoder
        fse_lookup_weights_mem_rd_req_s: chan<MemReaderReq> out,
        fse_lookup_weights_mem_rd_resp_r: chan<MemReaderResp> in,
        fse_decoder_weights_mem_rd_req_s: chan<MemReaderReq> out,
        fse_decoder_weights_mem_rd_resp_r: chan<MemReaderResp> in,

        // Muxed internal RAM Write interface (goes to Huffman Weights Memory)
        weights_ram_wr_req_s: chan<WeightsRamWrReq> out,
        weights_ram_wr_resp_r: chan<WeightsRamWrResp> in,

        // FSE RAMs
        dpd_rd_req_s: chan<DpdRamRdReq> out,
        dpd_rd_resp_r: chan<DpdRamRdResp> in,
        dpd_wr_req_s: chan<DpdRamWrReq> out,
        dpd_wr_resp_r: chan<DpdRamWrResp> in,

        tmp_rd_req_s: chan<TmpRamRdReq> out,
        tmp_rd_resp_r: chan<TmpRamRdResp> in,
        tmp_wr_req_s: chan<TmpRamWrReq> out,
        tmp_wr_resp_r: chan<TmpRamWrResp> in,

        tmp2_rd_req_s: chan<Tmp2RamRdReq> out,
        tmp2_rd_resp_r: chan<Tmp2RamRdResp> in,
        tmp2_wr_req_s: chan<Tmp2RamWrReq> out,
        tmp2_wr_resp_r: chan<Tmp2RamWrResp> in,

        fse_rd_req_s: chan<FseRamRdReq> out,
        fse_rd_resp_r: chan<FseRamRdResp> in,
        fse_wr_req_s: chan<FseRamWrReq> out,
        fse_wr_resp_r: chan<FseRamWrResp> in,
    ) {
        // Decoded Weights select for RamMux
        let (decoded_weights_sel_s, decoded_weights_sel_r) = chan<u1, u32:1>("decoded_weights_sel");

        // Raw Huffman Tree Description control
        let (raw_weights_req_s, raw_weights_req_r) = chan<RawWeightsReq, u32:1>("raw_weights_req");
        let (raw_weights_resp_s, raw_weights_resp_r) = chan<RawWeightsResp, u32:1>("raw_weights_resp");

        // Fse Huffman Tree Description control
        let (fse_weights_req_s, fse_weights_req_r) = chan<FseWeightsReq, u32:1>("fse_weights_req");
        let (fse_weights_resp_s, fse_weights_resp_r) = chan<FseWeightsResp, u32:1>("fse_weights_resp");

        // Internal RAM Write interface with decoded RAW Huffman Tree Description
        let (raw_weights_ram_wr_req_s, raw_weights_ram_wr_req_r) = chan<WeightsRamWrReq, u32:1>("raw_weights_ram_wr_req");
        let (raw_weights_ram_wr_resp_s, raw_weights_ram_wr_resp_r) = chan<WeightsRamWrResp, u32:1>("raw_weights_ram_wr_resp");

        // Internal RAM Write interface with decoded Fse Huffman Tree Description
        let (fse_weights_ram_wr_req_s, fse_weights_ram_wr_req_r) = chan<WeightsRamWrReq, u32:1>("fse_weights_ram_wr_req");
        let (fse_weights_ram_wr_resp_s, fse_weights_ram_wr_resp_r) = chan<WeightsRamWrResp, u32:1>("fse_weights_ram_wr_resp_s");

        let (raw_weights_ram_rd_req_s, raw_weights_ram_rd_req_r) = chan<WeightsRamRdReq, u32:1>("raw_weights_ram_rd_req");
        let (raw_weights_ram_rd_resp_s, raw_weights_ram_rd_resp_r) = chan<WeightsRamRdResp, u32:1>("raw_weights_ram_rd_resp");

        let (fse_weights_ram_rd_req_s, fse_weights_ram_rd_req_r) = chan<WeightsRamRdReq, u32:1>("fse_weights_ram_rd_req");
        let (fse_weights_ram_rd_resp_s, fse_weights_ram_rd_resp_r) = chan<WeightsRamRdResp, u32:1>("fse_weights_ram_rd_resp");

        let (weights_ram_rd_req_s, weights_ram_rd_req_r) = chan<WeightsRamRdReq, u32:1>("weights_ram_rd_req");
        let (weights_ram_rd_resp_s, weights_ram_rd_resp_r) = chan<WeightsRamRdResp, u32:1>("weights_ram_rd_resp");

        spawn HuffmanRawWeightsDecoder<
            AXI_ADDR_W, AXI_DATA_W,
            WEIGHTS_RAM_ADDR_W, WEIGHTS_RAM_DATA_W,
            WEIGHTS_RAM_NUM_PARTITIONS
        >(
            raw_weights_req_r, raw_weights_resp_s,
            raw_weights_mem_rd_req_s, raw_weights_mem_rd_resp_r,
            raw_weights_ram_wr_req_s, raw_weights_ram_wr_resp_r
        );

        spawn HuffmanFseWeightsDecoder<
            AXI_ADDR_W, AXI_DATA_W, AXI_ID_W,
            WEIGHTS_RAM_ADDR_W, WEIGHTS_RAM_DATA_W, WEIGHTS_RAM_NUM_PARTITIONS,
            DPD_RAM_ADDR_W, DPD_RAM_DATA_W, DPD_RAM_NUM_PARTITIONS,
            TMP_RAM_ADDR_W, TMP_RAM_DATA_W, TMP_RAM_NUM_PARTITIONS,
            TMP2_RAM_ADDR_W, TMP2_RAM_DATA_W, TMP2_RAM_NUM_PARTITIONS,
            FSE_RAM_ADDR_W, FSE_RAM_DATA_W, FSE_RAM_NUM_PARTITIONS,
        >(
            fse_weights_req_r, fse_weights_resp_s,
            fse_lookup_weights_mem_rd_req_s, fse_lookup_weights_mem_rd_resp_r,
            fse_decoder_weights_mem_rd_req_s, fse_decoder_weights_mem_rd_resp_r,
            fse_weights_ram_wr_req_s, fse_weights_ram_wr_resp_r,
            dpd_rd_req_s, dpd_rd_resp_r, dpd_wr_req_s, dpd_wr_resp_r,
            tmp_rd_req_s, tmp_rd_resp_r, tmp_wr_req_s, tmp_wr_resp_r,
            tmp2_rd_req_s, tmp2_rd_resp_r, tmp2_wr_req_s, tmp2_wr_resp_r,
            fse_rd_req_s, fse_rd_resp_r, fse_wr_req_s, fse_wr_resp_r,
        );

        spawn ram_mux::RamMux<WEIGHTS_RAM_ADDR_W, WEIGHTS_RAM_DATA_W, WEIGHTS_RAM_NUM_PARTITIONS>(
            decoded_weights_sel_r,
            raw_weights_ram_rd_req_r, raw_weights_ram_rd_resp_s, // We don't care about read side
            raw_weights_ram_wr_req_r, raw_weights_ram_wr_resp_s,
            fse_weights_ram_rd_req_r, fse_weights_ram_rd_resp_s, // We don't care about read side
            fse_weights_ram_wr_req_r, fse_weights_ram_wr_resp_s,
            weights_ram_rd_req_s, weights_ram_rd_resp_r,         // We don't care about read side
            weights_ram_wr_req_s, weights_ram_wr_resp_r
        );

        (
            req_r, resp_s,
            header_mem_rd_req_s, header_mem_rd_resp_r,
            decoded_weights_sel_s,
            raw_weights_req_s, raw_weights_resp_r,
            fse_weights_req_s, fse_weights_resp_r,
            raw_weights_ram_rd_req_s, raw_weights_ram_rd_resp_r, // We don't care about read side
            fse_weights_ram_rd_req_s, fse_weights_ram_rd_resp_r, // We don't care about read side
            weights_ram_rd_req_r, weights_ram_rd_resp_s,         // We don't care about read side
        )
    }

    next (state: State) {
        let tok = join();

        let (tok, req) = recv(tok, req_r);
        trace_fmt!("Received Huffman weights decoding request {:#x}", req);
        // Fetch Huffman Tree Header
        let header_mem_rd_req = MemReaderReq {
            addr: req.addr,
            length: uN[AXI_ADDR_W]:1,
        };
        let tok = send(tok, header_mem_rd_req_s, header_mem_rd_req);
        let (tok, header_mem_rd_resp) = recv(tok, header_mem_rd_resp_r);

        // Decode Huffman Tree Header
        // Now we know Huffman Tree Description size and the type of the description (RAW or FSE)
        // Send proper select signal for the RamMux
        // Send decoding request to HuffmanRawWeightsDecoder or HuffmanFseWeightsDecoder
        // Receive response from HuffmanRawWeightsDecoder or HuffmanFseWeightsDecoder

        let header_byte = header_mem_rd_resp.data as u8;
        trace_fmt!("Huffman weights header: {:#x}", header_byte);

        let weights_type = if header_byte < u8:128 {
            WeightsType::FSE
        } else {
            WeightsType::RAW
        };

        let tok = send(tok, decoded_weights_sel_s, weights_type == WeightsType::FSE);

        // FSE
        if weights_type == WeightsType::FSE {
            trace_fmt!("Decoding FSE Huffman weights");
        } else {};
        let fse_weights_req = FseWeightsReq {
            addr: req.addr,
            length: header_byte,
        };
        let tok = send_if(tok, fse_weights_req_s, weights_type == WeightsType::FSE, fse_weights_req);
        let (tok, fse_weights_resp) = recv_if(tok, fse_weights_resp_r, weights_type == WeightsType::FSE, zero!<FseWeightsResp>());

        let fse_status = match fse_weights_resp.status {
            HuffmanFseWeightsDecoderStatus::OKAY => HuffmanWeightsDecoderStatus::OKAY,
            HuffmanFseWeightsDecoderStatus::ERROR => HuffmanWeightsDecoderStatus::ERROR,
            _ => fail!("impossible_status_fse", HuffmanWeightsDecoderStatus::ERROR)
        };

        // RAW
        if weights_type == WeightsType::RAW {
            trace_fmt!("Decoding RAW Huffman weights");
        } else {};
        let raw_weights_req = RawWeightsReq {
            addr: req.addr,
            n_symbols: header_byte - u8:127,
        };
        let tok = send_if(tok, raw_weights_req_s, weights_type == WeightsType::RAW, raw_weights_req);
        let (tok, raw_weights_resp) = recv_if(tok, raw_weights_resp_r, weights_type == WeightsType::RAW, zero!<RawWeightsResp>());

        let raw_status = match raw_weights_resp.status {
            HuffmanRawWeightsDecoderStatus::OKAY => HuffmanWeightsDecoderStatus::OKAY,
            HuffmanRawWeightsDecoderStatus::ERROR => HuffmanWeightsDecoderStatus::ERROR,
            _ => fail!("impossible_status_raw", HuffmanWeightsDecoderStatus::ERROR)
        };

        let resp = match weights_type {
            WeightsType::RAW => {
                Resp {
                    status: raw_status,
                    tree_description_size: (((header_byte - u8:127) >> u8:1) + u8:1) as uN[AXI_ADDR_W] + uN[AXI_ADDR_W]:1, // include header size
                }
            },
            WeightsType::FSE => {
                Resp {
                    status: fse_status,
                    tree_description_size: header_byte as uN[AXI_ADDR_W] + uN[AXI_ADDR_W]:1, // include header size
                }
            },
            _ => fail!("impossible_weights_type", zero!<Resp>()),
        };

        let tok = send(tok, resp_s, resp);

        // Handle fake ram read request channels
        let tok = send_if(tok, raw_weights_ram_rd_req_s, false, zero!<WeightsRamRdReq>());
        let tok = send_if(tok, fse_weights_ram_rd_req_s, false, zero!<WeightsRamRdReq>());
        let (tok, _) = recv_if(tok, weights_ram_rd_req_r, false, zero!<WeightsRamRdReq>());
        let tok = send_if(tok, weights_ram_rd_resp_s, false, zero!<WeightsRamRdResp>());
        let (tok, _) = recv_if(tok, raw_weights_ram_rd_resp_r, false, zero!<WeightsRamRdResp>());
        let (tok, _) = recv_if(tok, fse_weights_ram_rd_resp_r, false, zero!<WeightsRamRdResp>());

        zero!<State>()
    }
}

const INST_AXI_ADDR_W = u32:32;
const INST_AXI_DATA_W = u32:64;
const INST_AXI_DEST_W = u32:8;
const INST_AXI_ID_W = u32:8;

const INST_RAM_DATA_W = INST_AXI_DATA_W;
const INST_RAM_SIZE = u32:1024;
const INST_RAM_ADDR_W = INST_AXI_ADDR_W;
const INST_RAM_PARTITION_SIZE = INST_RAM_DATA_W / u32:8;
const INST_RAM_NUM_PARTITIONS = ram::num_partitions(INST_RAM_PARTITION_SIZE, INST_RAM_DATA_W);
const INST_RAM_SIMULTANEOUS_RW_BEHAVIOR = ram::SimultaneousReadWriteBehavior::READ_BEFORE_WRITE;
const INST_RAM_INITIALIZED = true;
const INST_RAM_ASSERT_VALID_READ = true;

const INST_WEIGHTS_RAM_ADDR_W = huffman_prescan::RAM_ADDR_WIDTH;
const INST_WEIGHTS_RAM_SIZE = huffman_prescan::RAM_SIZE;
const INST_WEIGHTS_RAM_DATA_W = huffman_prescan::RAM_ACCESS_WIDTH;
const INST_WEIGHTS_RAM_PARTITION_SIZE = INST_WEIGHTS_RAM_DATA_W / u32:8;
const INST_WEIGHTS_RAM_NUM_PARTITIONS = ram::num_partitions(INST_WEIGHTS_RAM_PARTITION_SIZE, INST_WEIGHTS_RAM_DATA_W);
const INST_WEIGHTS_RAM_SIMULTANEOUS_RW_BEHAVIOR = ram::SimultaneousReadWriteBehavior::READ_BEFORE_WRITE;
const INST_WEIGHTS_RAM_INITIALIZED = true;
const INST_WEIGHTS_RAM_ASSERT_VALID_READ = true;

const INST_DPD_RAM_DATA_W = u32:16;
const INST_DPD_RAM_SIZE = u32:256;
const INST_DPD_RAM_ADDR_W = std::clog2(INST_DPD_RAM_SIZE);
const INST_DPD_RAM_WORD_PARTITION_SIZE = INST_DPD_RAM_DATA_W;
const INST_DPD_RAM_NUM_PARTITIONS = ram::num_partitions(INST_DPD_RAM_WORD_PARTITION_SIZE, INST_DPD_RAM_DATA_W);

const INST_FSE_RAM_DATA_W = u32:32;
const INST_FSE_RAM_SIZE = u32:256;
const INST_FSE_RAM_ADDR_W = std::clog2(INST_FSE_RAM_SIZE);
const INST_FSE_RAM_WORD_PARTITION_SIZE = INST_FSE_RAM_DATA_W / u32:3;
const INST_FSE_RAM_NUM_PARTITIONS = ram::num_partitions(INST_FSE_RAM_WORD_PARTITION_SIZE, INST_FSE_RAM_DATA_W);

const INST_TMP_RAM_DATA_W = u32:16;
const INST_TMP_RAM_SIZE = u32:256;
const INST_TMP_RAM_ADDR_W = std::clog2(INST_TMP_RAM_SIZE);
const INST_TMP_RAM_WORD_PARTITION_SIZE = INST_TMP_RAM_DATA_W;
const INST_TMP_RAM_NUM_PARTITIONS = ram::num_partitions(INST_TMP_RAM_WORD_PARTITION_SIZE, INST_TMP_RAM_DATA_W);

const INST_TMP2_RAM_DATA_W = u32:8;
const INST_TMP2_RAM_SIZE = u32:512;
const INST_TMP2_RAM_ADDR_W = std::clog2(INST_TMP2_RAM_SIZE);
const INST_TMP2_RAM_WORD_PARTITION_SIZE = INST_TMP2_RAM_DATA_W;
const INST_TMP2_RAM_NUM_PARTITIONS = ram::num_partitions(INST_TMP2_RAM_WORD_PARTITION_SIZE, INST_TMP2_RAM_DATA_W);

proc HuffmanWeightsDecoderInst {
    // Memory Reader + Input

    type MemReaderReq = mem_reader::MemReaderReq<INST_AXI_ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<INST_AXI_DATA_W, INST_AXI_ADDR_W>;

    type InputBufferRamRdReq = ram::ReadReq<INST_RAM_ADDR_W, INST_RAM_NUM_PARTITIONS>;
    type InputBufferRamRdResp = ram::ReadResp<INST_RAM_DATA_W>;
    type InputBufferRamWrReq = ram::WriteReq<INST_RAM_ADDR_W, INST_RAM_DATA_W, INST_RAM_NUM_PARTITIONS>;
    type InputBufferRamWrResp = ram::WriteResp;

    type AxiAr = axi::AxiAr<INST_AXI_ADDR_W, INST_AXI_ID_W>;
    type AxiR = axi::AxiR<INST_AXI_DATA_W, INST_AXI_ID_W>;

    // Weights RAM

    type WeightsRamRdReq = ram::ReadReq<INST_WEIGHTS_RAM_ADDR_W, INST_WEIGHTS_RAM_NUM_PARTITIONS>;
    type WeightsRamRdResp = ram::ReadResp<INST_WEIGHTS_RAM_DATA_W>;
    type WeightsRamWrReq = ram::WriteReq<INST_WEIGHTS_RAM_ADDR_W, INST_WEIGHTS_RAM_DATA_W, INST_WEIGHTS_RAM_NUM_PARTITIONS>;
    type WeightsRamWrResp = ram::WriteResp;

    // Huffman Weights Decoder
    type Req = HuffmanWeightsDecoderReq<INST_AXI_ADDR_W>;
    type Resp = HuffmanWeightsDecoderResp<INST_AXI_ADDR_W>;
    type Status = HuffmanWeightsDecoderStatus;
    type State = HuffmanWeightsDecoderState<INST_AXI_ADDR_W>;

    // FSE RAMs
    type DpdRamRdReq = ram::ReadReq<INST_DPD_RAM_ADDR_W, INST_DPD_RAM_NUM_PARTITIONS>;
    type DpdRamRdResp = ram::ReadResp<INST_DPD_RAM_DATA_W>;
    type DpdRamWrReq = ram::WriteReq<INST_DPD_RAM_ADDR_W, INST_DPD_RAM_DATA_W, INST_DPD_RAM_NUM_PARTITIONS>;
    type DpdRamWrResp = ram::WriteResp;

    type TmpRamRdReq = ram::ReadReq<INST_TMP_RAM_ADDR_W, INST_TMP_RAM_NUM_PARTITIONS>;
    type TmpRamRdResp = ram::ReadResp<INST_TMP_RAM_DATA_W>;
    type TmpRamWrReq = ram::WriteReq<INST_TMP_RAM_ADDR_W, INST_TMP_RAM_DATA_W, INST_TMP_RAM_NUM_PARTITIONS>;
    type TmpRamWrResp = ram::WriteResp;

    type Tmp2RamRdReq = ram::ReadReq<INST_TMP2_RAM_ADDR_W, INST_TMP2_RAM_NUM_PARTITIONS>;
    type Tmp2RamRdResp = ram::ReadResp<INST_TMP2_RAM_DATA_W>;
    type Tmp2RamWrReq = ram::WriteReq<INST_TMP2_RAM_ADDR_W, INST_TMP2_RAM_DATA_W, INST_TMP2_RAM_NUM_PARTITIONS>;
    type Tmp2RamWrResp = ram::WriteResp;

    type FseRamRdReq = ram::ReadReq<INST_FSE_RAM_ADDR_W, INST_FSE_RAM_NUM_PARTITIONS>;
    type FseRamRdResp = ram::ReadResp<INST_FSE_RAM_DATA_W>;
    type FseRamWrReq = ram::WriteReq<INST_FSE_RAM_ADDR_W, INST_FSE_RAM_DATA_W, INST_FSE_RAM_NUM_PARTITIONS>;
    type FseRamWrResp = ram::WriteResp;

    type MemAxiAr = axi::AxiAr<INST_AXI_ADDR_W, INST_AXI_ID_W>;
    type MemAxiR = axi::AxiR<INST_AXI_DATA_W, INST_AXI_ID_W>;


    config (
        req_r: chan<Req> in,
        resp_s: chan<Resp> out,
        header_mem_rd_req_s: chan<MemReaderReq> out,
        header_mem_rd_resp_r: chan<MemReaderResp> in,
        raw_weights_mem_rd_req_s: chan<MemReaderReq> out,
        raw_weights_mem_rd_resp_r: chan<MemReaderResp> in,
        fse_lookup_weights_mem_rd_req_s: chan<MemReaderReq> out,
        fse_lookup_weights_mem_rd_resp_r: chan<MemReaderResp> in,
        fse_decoder_weights_mem_rd_req_s: chan<MemReaderReq> out,
        fse_decoder_weights_mem_rd_resp_r: chan<MemReaderResp> in,
        weights_ram_wr_req_s: chan<WeightsRamWrReq> out,
        weights_ram_wr_resp_r: chan<WeightsRamWrResp> in,
        dpd_rd_req_s: chan<DpdRamRdReq> out,
        dpd_rd_resp_r: chan<DpdRamRdResp> in,
        dpd_wr_req_s: chan<DpdRamWrReq> out,
        dpd_wr_resp_r: chan<DpdRamWrResp> in,
        tmp_rd_req_s: chan<TmpRamRdReq> out,
        tmp_rd_resp_r: chan<TmpRamRdResp> in,
        tmp_wr_req_s: chan<TmpRamWrReq> out,
        tmp_wr_resp_r: chan<TmpRamWrResp> in,
        tmp2_rd_req_s: chan<Tmp2RamRdReq> out,
        tmp2_rd_resp_r: chan<Tmp2RamRdResp> in,
        tmp2_wr_req_s: chan<Tmp2RamWrReq> out,
        tmp2_wr_resp_r: chan<Tmp2RamWrResp> in,
        fse_rd_req_s: chan<FseRamRdReq> out,
        fse_rd_resp_r: chan<FseRamRdResp> in,
        fse_wr_req_s: chan<FseRamWrReq> out,
        fse_wr_resp_r: chan<FseRamWrResp> in,
    ) {
        spawn HuffmanWeightsDecoder<
            INST_AXI_ADDR_W, INST_AXI_DATA_W, INST_AXI_ID_W,
            INST_WEIGHTS_RAM_ADDR_W, INST_WEIGHTS_RAM_DATA_W, INST_WEIGHTS_RAM_NUM_PARTITIONS,
            INST_DPD_RAM_ADDR_W, INST_DPD_RAM_DATA_W, INST_DPD_RAM_NUM_PARTITIONS,
            INST_TMP_RAM_ADDR_W, INST_TMP_RAM_DATA_W, INST_TMP_RAM_NUM_PARTITIONS,
            INST_TMP2_RAM_ADDR_W, INST_TMP2_RAM_DATA_W, INST_TMP2_RAM_NUM_PARTITIONS,
            INST_FSE_RAM_ADDR_W, INST_FSE_RAM_DATA_W, INST_FSE_RAM_NUM_PARTITIONS,
        > (
            req_r, resp_s,
            header_mem_rd_req_s, header_mem_rd_resp_r,
            raw_weights_mem_rd_req_s, raw_weights_mem_rd_resp_r,
            fse_lookup_weights_mem_rd_req_s, fse_lookup_weights_mem_rd_resp_r,
            fse_decoder_weights_mem_rd_req_s, fse_decoder_weights_mem_rd_resp_r,
            weights_ram_wr_req_s, weights_ram_wr_resp_r,
            dpd_rd_req_s, dpd_rd_resp_r, dpd_wr_req_s, dpd_wr_resp_r,
            tmp_rd_req_s, tmp_rd_resp_r, tmp_wr_req_s, tmp_wr_resp_r,
            tmp2_rd_req_s, tmp2_rd_resp_r, tmp2_wr_req_s, tmp2_wr_resp_r,
            fse_rd_req_s, fse_rd_resp_r, fse_wr_req_s, fse_wr_resp_r,
        );
    }

    init { }

    next (state: ()) { }
}

const TEST_RAM_N = u32:4;

const TEST_AXI_ADDR_W = u32:32;
const TEST_AXI_DATA_W = u32:64;
const TEST_AXI_DEST_W = u32:8;
const TEST_AXI_ID_W = u32:8;

const TEST_RAM_DATA_W = TEST_AXI_DATA_W;
const TEST_RAM_SIZE = u32:1024;
const TEST_RAM_ADDR_W = TEST_AXI_ADDR_W;
const TEST_RAM_PARTITION_SIZE = TEST_RAM_DATA_W / u32:8;
const TEST_RAM_NUM_PARTITIONS = ram::num_partitions(TEST_RAM_PARTITION_SIZE, TEST_RAM_DATA_W);
const TEST_RAM_SIMULTANEOUS_RW_BEHAVIOR = ram::SimultaneousReadWriteBehavior::READ_BEFORE_WRITE;
const TEST_RAM_INITIALIZED = true;
const TEST_RAM_ASSERT_VALID_READ = true;

const TEST_WEIGHTS_RAM_ADDR_W = huffman_prescan::RAM_ADDR_WIDTH;
const TEST_WEIGHTS_RAM_SIZE = huffman_prescan::RAM_SIZE;
const TEST_WEIGHTS_RAM_DATA_W = huffman_prescan::RAM_ACCESS_WIDTH;
const TEST_WEIGHTS_RAM_PARTITION_SIZE = TEST_WEIGHTS_RAM_DATA_W / u32:8;
const TEST_WEIGHTS_RAM_NUM_PARTITIONS = ram::num_partitions(TEST_WEIGHTS_RAM_PARTITION_SIZE, TEST_WEIGHTS_RAM_DATA_W);
const TEST_WEIGHTS_RAM_SIMULTANEOUS_RW_BEHAVIOR = ram::SimultaneousReadWriteBehavior::READ_BEFORE_WRITE;
const TEST_WEIGHTS_RAM_INITIALIZED = true;
const TEST_WEIGHTS_RAM_ASSERT_VALID_READ = true;

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

const TEST_TMP2_RAM_DATA_W = u32:8;
const TEST_TMP2_RAM_SIZE = u32:512;
const TEST_TMP2_RAM_ADDR_W = std::clog2(TEST_TMP2_RAM_SIZE);
const TEST_TMP2_RAM_WORD_PARTITION_SIZE = TEST_TMP2_RAM_DATA_W;
const TEST_TMP2_RAM_NUM_PARTITIONS = ram::num_partitions(TEST_TMP2_RAM_WORD_PARTITION_SIZE, TEST_TMP2_RAM_DATA_W);

// RAW weights
const TEST_RAW_INPUT_ADDR = uN[TEST_AXI_ADDR_W]:0x40;

// Weights sum is 1010, so the last one will be 14
const TEST_RAW_DATA = u8[65]:[
    // len      x0 x1      x2 x3      x4 x5      x6 x7      x8 x9      xA xB      xC xD      xE xF
    u8:248, u8:0xB__6, u8:0x8__5, u8:0x6__A, u8:0x9__C, u8:0x0__C, u8:0xA__9, u8:0x0__0, u8:0xD__0, // 0x0x
            u8:0x6__E, u8:0x3__9, u8:0x8__4, u8:0x7__C, u8:0xC__2, u8:0x4__2, u8:0xB__A, u8:0x4__E, // 0x1x
            u8:0xF__6, u8:0x2__7, u8:0x9__4, u8:0xD__1, u8:0xD__8, u8:0x2__B, u8:0xE__2, u8:0xD__1, // 0x2x
            u8:0x8__F, u8:0x2__4, u8:0xD__3, u8:0x0__E, u8:0xF__E, u8:0x1__B, u8:0xF__9, u8:0x8__2, // 0x3x
            u8:0xC__A, u8:0x6__1, u8:0x0__3, u8:0xD__C, u8:0xF__5, u8:0x1__D, u8:0x7__0, u8:0x1__6, // 0x4x
            u8:0xA__A, u8:0x3__2, u8:0x8__8, u8:0x0__6, u8:0xE__7, u8:0x6__7, u8:0x8__E, u8:0x6__2, // 0x5x
            u8:0x1__F, u8:0x3__E, u8:0xF__0, u8:0xC__7, u8:0x4__1, u8:0x7__E, u8:0x8__C, u8:0x8__4, // 0x6x
            u8:0x3__3, u8:0xA__8, u8:0xE__E, u8:0x4__B, u8:0x0__0, u8:0x0__0, u8:0x0__0, u8:0x0__0, // 0x7x
];

const TEST_RAW_DATA_LAST_WEIGHT = u8:0xA;

// FSE weights
const TEST_FSE_INPUT_ADDR = uN[TEST_AXI_ADDR_W]:0x200;

// Testcase format - tuple of:
// - array of: header (first byte) + FSE bitstream
// - array of: expected weights RAM contents
const TESTCASES_FSE: (u8[32], u8[128])[7] = [
    (
        u8[32]:[
            u8:10,
            u8:0xC0, u8:0x25, u8:0x1D, u8:0x49, u8:0x6E, u8:0xC2, u8:0xFF, u8:0xFF,
            u8:0xEE, u8:0x06, u8:0, ...
        ],
        u8[128]:[
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x01, u8:0x12, u8:0x34, u8:0x56, u8:0, ...
        ],
    ),
    (
        u8[32]:[
            u8:15,
            u8:0xC0, u8:0x25, u8:0x1D, u8:0x9B, u8:0x1E, u8:0xAD, u8:0xFE, u8:0xFF,
            u8:0x7F, u8:0x67, u8:0xFE, u8:0xD3, u8:0xFF, u8:0xCE, u8:0x05, u8:0, ...
        ],
        u8[128]:[
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x10, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x30,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x50, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x01, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x02,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x04, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x06, u8:0, ...
        ],
    ),
    (
        u8[32]:[
            u8:23,
            u8:0x90, u8:0x25, u8:0x49, u8:0x3A, u8:0xEB, u8:0x3B, u8:0xBD, u8:0x7E,
            u8:0xD6, u8:0x5D, u8:0x3C, u8:0xB3, u8:0x66, u8:0x77, u8:0xA8, u8:0xBB,
            u8:0x25, u8:0x76, u8:0xBA, u8:0xFF, u8:0x20, u8:0xA8, u8:0x01, u8:0, ...
        ],
        u8[128]:[
            u8:0x00, u8:0x10, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x02,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x30, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x04, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x50,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x06, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x70, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x08, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x01, u8:0x00,
            u8:0x00, u8:0x90, u8:0, ...
        ],
    ),
    (
        u8[32]:[
            u8:8,
            u8:0xF0, u8:0x39, u8:0xFF, u8:0x23, u8:0x45, u8:0x55, u8:0xCF, u8:0x99, u8:0, ...
        ],
        u8[128]:[
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x20, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x01, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x10, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x03, u8:0, ...
        ],
    ),
    (
        u8[32]:[
            u8:24,
            u8:0xB0, u8:0xA5, u8:0x92, u8:0x0E, u8:0x14, u8:0x3B, u8:0x7B, u8:0x58,
            u8:0xED, u8:0xB0, u8:0x1D, u8:0x9C, u8:0x43, u8:0x82, u8:0xC5, u8:0x8E,
            u8:0xD3, u8:0x38, u8:0x36, u8:0x87, u8:0x73, u8:0x08, u8:0x58, u8:0x02, u8:0, ...
        ],
        u8[128]:[
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x20, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x10, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x10,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x07, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x10, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x05, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x10, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x02, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x01,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x01, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x80, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x01, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x60,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x01, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00, u8:0x30, u8:0, ...
        ],
    ),
    (
        u8[32]:[
            u8:9,
            u8:0xE0, u8:0xE9, u8:0x40, u8:0x0D, u8:0x80, u8:0x0A, u8:0x10, u8:0x59,
            u8:0x04, u8:0, ...
        ],
        u8[128]:[
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x03,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x20, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x01, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00, u8:0x10, u8:0, ...
        ],
    ),
    (
        u8[32]:[
            u8:9,
            u8:0xF0, u8:0x19, u8:0x03, u8:0x23, u8:0x7D, u8:0x9F, u8:0xD7, u8:0xB5,
            u8:0x06, u8:0, ...
        ],
        u8[128]:[
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x10, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x30, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x01, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x00, u8:0x00, u8:0x00,
            u8:0x00, u8:0x02, u8:0, ...
        ],
    ),
];

#[test_proc]
proc HuffmanWeightsDecoder_test {
    // Memory Reader + Input

    type MemReaderReq = mem_reader::MemReaderReq<TEST_AXI_ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<TEST_AXI_DATA_W, TEST_AXI_ADDR_W>;

    type InputBufferRamRdReq = ram::ReadReq<TEST_RAM_ADDR_W, TEST_RAM_NUM_PARTITIONS>;
    type InputBufferRamRdResp = ram::ReadResp<TEST_RAM_DATA_W>;
    type InputBufferRamWrReq = ram::WriteReq<TEST_RAM_ADDR_W, TEST_RAM_DATA_W, TEST_RAM_NUM_PARTITIONS>;
    type InputBufferRamWrResp = ram::WriteResp;

    type AxiAr = axi::AxiAr<TEST_AXI_ADDR_W, TEST_AXI_ID_W>;
    type AxiR = axi::AxiR<TEST_AXI_DATA_W, TEST_AXI_ID_W>;

    // Weights RAM

    type WeightsRamRdReq = ram::ReadReq<TEST_WEIGHTS_RAM_ADDR_W, TEST_WEIGHTS_RAM_NUM_PARTITIONS>;
    type WeightsRamRdResp = ram::ReadResp<TEST_WEIGHTS_RAM_DATA_W>;
    type WeightsRamWrReq = ram::WriteReq<TEST_WEIGHTS_RAM_ADDR_W, TEST_WEIGHTS_RAM_DATA_W, TEST_WEIGHTS_RAM_NUM_PARTITIONS>;
    type WeightsRamWrResp = ram::WriteResp;

    // Huffman Weights Decoder
    type Req = HuffmanWeightsDecoderReq<TEST_AXI_ADDR_W>;
    type Resp = HuffmanWeightsDecoderResp<TEST_AXI_ADDR_W>;
    type Status = HuffmanWeightsDecoderStatus;
    type State = HuffmanWeightsDecoderState<TEST_AXI_ADDR_W>;

    // FSE RAMs
    type DpdRamRdReq = ram::ReadReq<TEST_DPD_RAM_ADDR_W, TEST_DPD_RAM_NUM_PARTITIONS>;
    type DpdRamRdResp = ram::ReadResp<TEST_DPD_RAM_DATA_W>;
    type DpdRamWrReq = ram::WriteReq<TEST_DPD_RAM_ADDR_W, TEST_DPD_RAM_DATA_W, TEST_DPD_RAM_NUM_PARTITIONS>;
    type DpdRamWrResp = ram::WriteResp;

    type TmpRamRdReq = ram::ReadReq<TEST_TMP_RAM_ADDR_W, TEST_TMP_RAM_NUM_PARTITIONS>;
    type TmpRamRdResp = ram::ReadResp<TEST_TMP_RAM_DATA_W>;
    type TmpRamWrReq = ram::WriteReq<TEST_TMP_RAM_ADDR_W, TEST_TMP_RAM_DATA_W, TEST_TMP_RAM_NUM_PARTITIONS>;
    type TmpRamWrResp = ram::WriteResp;

    type Tmp2RamRdReq = ram::ReadReq<TEST_TMP2_RAM_ADDR_W, TEST_TMP2_RAM_NUM_PARTITIONS>;
    type Tmp2RamRdResp = ram::ReadResp<TEST_TMP2_RAM_DATA_W>;
    type Tmp2RamWrReq = ram::WriteReq<TEST_TMP2_RAM_ADDR_W, TEST_TMP2_RAM_DATA_W, TEST_TMP2_RAM_NUM_PARTITIONS>;
    type Tmp2RamWrResp = ram::WriteResp;

    type FseRamRdReq = ram::ReadReq<TEST_FSE_RAM_ADDR_W, TEST_FSE_RAM_NUM_PARTITIONS>;
    type FseRamRdResp = ram::ReadResp<TEST_FSE_RAM_DATA_W>;
    type FseRamWrReq = ram::WriteReq<TEST_FSE_RAM_ADDR_W, TEST_FSE_RAM_DATA_W, TEST_FSE_RAM_NUM_PARTITIONS>;
    type FseRamWrResp = ram::WriteResp;

    type MemAxiAr = axi::AxiAr<TEST_AXI_ADDR_W, TEST_AXI_ID_W>;
    type MemAxiR = axi::AxiR<TEST_AXI_DATA_W, TEST_AXI_ID_W>;

    terminator: chan<bool> out;

    req_s: chan<Req> out;
    resp_r: chan<Resp> in;

    input_ram_wr_req_s: chan<InputBufferRamWrReq>[TEST_RAM_N] out;
    input_ram_wr_resp_r: chan<InputBufferRamWrResp>[TEST_RAM_N] in;

    weights_ram_rd_req_s: chan<WeightsRamRdReq> out;
    weights_ram_rd_resp_r: chan<WeightsRamRdResp> in;

    config (terminator: chan<bool> out) {

        // Input Memory

        let (input_ram_rd_req_s, input_ram_rd_req_r) = chan<InputBufferRamRdReq>[TEST_RAM_N]("input_ram_rd_req");
        let (input_ram_rd_resp_s, input_ram_rd_resp_r) = chan<InputBufferRamRdResp>[TEST_RAM_N]("input_ram_rd_resp");
        let (input_ram_wr_req_s, input_ram_wr_req_r) = chan<InputBufferRamWrReq>[TEST_RAM_N]("input_ram_wr_req");
        let (input_ram_wr_resp_s, input_ram_wr_resp_r) = chan<InputBufferRamWrResp>[TEST_RAM_N]("input_ram_wr_resp");

        unroll_for! (i, _) in range(u32:0, TEST_RAM_N) {
            spawn ram::RamModel<
                TEST_RAM_DATA_W, TEST_RAM_SIZE, TEST_RAM_PARTITION_SIZE,
                TEST_RAM_SIMULTANEOUS_RW_BEHAVIOR, TEST_RAM_INITIALIZED,
                TEST_RAM_ASSERT_VALID_READ, TEST_RAM_ADDR_W,
            >(
                input_ram_rd_req_r[i], input_ram_rd_resp_s[i],
                input_ram_wr_req_r[i], input_ram_wr_resp_s[i],
            );
        }(());

        // Input Memory Axi Reader

        let (axi_ar_s, axi_ar_r) = chan<AxiAr>[TEST_RAM_N]("axi_ar");
        let (axi_r_s, axi_r_r) = chan<AxiR>[TEST_RAM_N]("axi_r");

        unroll_for! (i, _) in range(u32:0, TEST_RAM_N) {
            spawn axi_ram::AxiRamReader<
                TEST_AXI_ADDR_W, TEST_AXI_DATA_W,
                TEST_AXI_DEST_W, TEST_AXI_ID_W,
                TEST_RAM_SIZE,
            >(
                axi_ar_r[i], axi_r_s[i],
                input_ram_rd_req_s[i], input_ram_rd_resp_r[i],
            );
        }(());

        // Input Memory Reader

        let (mem_rd_req_s, mem_rd_req_r) = chan<MemReaderReq>[TEST_RAM_N]("mem_rd_req");
        let (mem_rd_resp_s, mem_rd_resp_r) = chan<MemReaderResp>[TEST_RAM_N]("mem_rd_resp");

        unroll_for! (i, _) in range(u32:0, TEST_RAM_N) {
            spawn mem_reader::MemReader<
                TEST_AXI_DATA_W, TEST_AXI_ADDR_W, TEST_AXI_DEST_W, TEST_AXI_ID_W,
            >(
                mem_rd_req_r[i], mem_rd_resp_s[i],
                axi_ar_s[i], axi_r_r[i],
            );
        }(());

        // Weights RAM

        let (weights_ram_rd_req_s, weights_ram_rd_req_r) = chan<WeightsRamRdReq>("weights_ram_rd_req");
        let (weights_ram_rd_resp_s, weights_ram_rd_resp_r) = chan<WeightsRamRdResp>("weights_ram_rd_resp");
        let (weights_ram_wr_req_s, weights_ram_wr_req_r) = chan<WeightsRamWrReq>("weights_ram_wr_req");
        let (weights_ram_wr_resp_s, weights_ram_wr_resp_r) = chan<WeightsRamWrResp>("weights_ram_wr_resp");

        spawn ram::RamModel<
            TEST_WEIGHTS_RAM_DATA_W, TEST_WEIGHTS_RAM_SIZE, TEST_WEIGHTS_RAM_PARTITION_SIZE,
            TEST_WEIGHTS_RAM_SIMULTANEOUS_RW_BEHAVIOR, TEST_WEIGHTS_RAM_INITIALIZED,
            TEST_WEIGHTS_RAM_ASSERT_VALID_READ, TEST_WEIGHTS_RAM_ADDR_W,
        >(
            weights_ram_rd_req_r, weights_ram_rd_resp_s,
            weights_ram_wr_req_r, weights_ram_wr_resp_s,
        );

        // Huffman Weights Decoder

        let (req_s, req_r) = chan<Req>("req");
        let (resp_s, resp_r) = chan<Resp>("resp");

        // FSE RAMs
        let (dpd_rd_req_s, dpd_rd_req_r) = chan<DpdRamRdReq>("dpd_rd_req");
        let (dpd_rd_resp_s, dpd_rd_resp_r) = chan<DpdRamRdResp>("dpd_rd_resp");
        let (dpd_wr_req_s, dpd_wr_req_r) = chan<DpdRamWrReq>("dpd_wr_req");
        let (dpd_wr_resp_s, dpd_wr_resp_r) = chan<DpdRamWrResp>("dpd_wr_resp");

        spawn ram::RamModel<
            TEST_DPD_RAM_DATA_W,
            TEST_DPD_RAM_SIZE,
            TEST_DPD_RAM_WORD_PARTITION_SIZE
        >(dpd_rd_req_r, dpd_rd_resp_s, dpd_wr_req_r, dpd_wr_resp_s);

        let (tmp_rd_req_s, tmp_rd_req_r) = chan<TmpRamRdReq>("tmp_rd_req");
        let (tmp_rd_resp_s, tmp_rd_resp_r) = chan<TmpRamRdResp>("tmp_rd_resp");
        let (tmp_wr_req_s, tmp_wr_req_r) = chan<TmpRamWrReq>("tmp_wr_req");
        let (tmp_wr_resp_s, tmp_wr_resp_r) = chan<TmpRamWrResp>("tmp_wr_resp");

        spawn ram::RamModel<
            TEST_TMP_RAM_DATA_W,
            TEST_TMP_RAM_SIZE,
            TEST_TMP_RAM_WORD_PARTITION_SIZE
        >(tmp_rd_req_r, tmp_rd_resp_s, tmp_wr_req_r, tmp_wr_resp_s);

        let (tmp2_rd_req_s, tmp2_rd_req_r) = chan<Tmp2RamRdReq>("tmp2_rd_req");
        let (tmp2_rd_resp_s, tmp2_rd_resp_r) = chan<Tmp2RamRdResp>("tmp2_rd_resp");
        let (tmp2_wr_req_s, tmp2_wr_req_r) = chan<Tmp2RamWrReq>("tmp2_wr_req");
        let (tmp2_wr_resp_s, tmp2_wr_resp_r) = chan<Tmp2RamWrResp>("tmp2_wr_resp");

        spawn ram::RamModel<
            TEST_TMP2_RAM_DATA_W,
            TEST_TMP2_RAM_SIZE,
            TEST_TMP2_RAM_WORD_PARTITION_SIZE
        >(tmp2_rd_req_r, tmp2_rd_resp_s, tmp2_wr_req_r, tmp2_wr_resp_s);

        let (fse_rd_req_s, fse_rd_req_r) = chan<FseRamRdReq>("fse_rd_req");
        let (fse_rd_resp_s, fse_rd_resp_r) = chan<FseRamRdResp>("fse_rd_resp");
        let (fse_wr_req_s, fse_wr_req_r) = chan<FseRamWrReq>("fse_wr_req");
        let (fse_wr_resp_s, fse_wr_resp_r) = chan<FseRamWrResp>("fse_wr_resp");

        spawn ram::RamModel<
            TEST_FSE_RAM_DATA_W,
            TEST_FSE_RAM_SIZE,
            TEST_FSE_RAM_WORD_PARTITION_SIZE
        >(fse_rd_req_r, fse_rd_resp_s, fse_wr_req_r, fse_wr_resp_s);

        spawn HuffmanWeightsDecoder<
            TEST_AXI_ADDR_W, TEST_AXI_DATA_W, TEST_AXI_ID_W,
            TEST_WEIGHTS_RAM_ADDR_W, TEST_WEIGHTS_RAM_DATA_W, TEST_WEIGHTS_RAM_NUM_PARTITIONS,
            TEST_DPD_RAM_ADDR_W, TEST_DPD_RAM_DATA_W, TEST_DPD_RAM_NUM_PARTITIONS,
            TEST_TMP_RAM_ADDR_W, TEST_TMP_RAM_DATA_W, TEST_TMP_RAM_NUM_PARTITIONS,
            TEST_TMP2_RAM_ADDR_W, TEST_TMP2_RAM_DATA_W, TEST_TMP2_RAM_NUM_PARTITIONS,
            TEST_FSE_RAM_ADDR_W, TEST_FSE_RAM_DATA_W, TEST_FSE_RAM_NUM_PARTITIONS,
        > (
            req_r, resp_s,
            mem_rd_req_s[0], mem_rd_resp_r[0],
            mem_rd_req_s[1], mem_rd_resp_r[1],
            mem_rd_req_s[2], mem_rd_resp_r[2],
            mem_rd_req_s[3], mem_rd_resp_r[3],
            weights_ram_wr_req_s, weights_ram_wr_resp_r,
            dpd_rd_req_s, dpd_rd_resp_r, dpd_wr_req_s, dpd_wr_resp_r,
            tmp_rd_req_s, tmp_rd_resp_r, tmp_wr_req_s, tmp_wr_resp_r,
            tmp2_rd_req_s, tmp2_rd_resp_r, tmp2_wr_req_s, tmp2_wr_resp_r,
            fse_rd_req_s, fse_rd_resp_r, fse_wr_req_s, fse_wr_resp_r,
        );

        (
            terminator,
            req_s, resp_r,
            input_ram_wr_req_s, input_ram_wr_resp_r,
            weights_ram_rd_req_s, weights_ram_rd_resp_r,
        )
    }

    init { }

    next (state: ()) {
        const TEST_DATA_PER_RAM_WRITE = TEST_RAM_DATA_W / u32:8;

        let tok = join();

        // RAW weights

        // Fill input RAM
        for (i, tok) in range(u32:0, (array_size(TEST_RAW_DATA) + TEST_DATA_PER_RAM_WRITE - u32:1) / TEST_DATA_PER_RAM_WRITE) {
            let ram_data = for (j, ram_data) in range(u32:0, TEST_DATA_PER_RAM_WRITE) {
                let data_idx = i * TEST_DATA_PER_RAM_WRITE + j;
                if (data_idx < array_size(TEST_RAW_DATA)) {
                    ram_data | ((TEST_RAW_DATA[data_idx] as uN[TEST_RAM_DATA_W]) << (u32:8 * j))
                } else {
                    ram_data
                }
            }(uN[TEST_RAM_DATA_W]:0);

            let input_ram_wr_req = InputBufferRamWrReq {
                addr: (TEST_RAW_INPUT_ADDR / u32:8) + i as uN[TEST_RAM_ADDR_W],
                data: ram_data,
                mask: !uN[TEST_RAM_NUM_PARTITIONS]:0,
            };

            let tok = unroll_for! (i, tok) in range(u32:0, TEST_RAM_N) {
                let tok = send(tok, input_ram_wr_req_s[i], input_ram_wr_req);
                let (tok, _) = recv(tok, input_ram_wr_resp_r[i]);
                tok
            }(tok);

            trace_fmt!("[TEST] Sent RAM write request to input RAMs {:#x}", input_ram_wr_req);

            tok
        }(tok);

        // Send decoding request
        let req = Req {
            addr: TEST_RAW_INPUT_ADDR,
        };
        let tok = send(tok, req_s, req);
        trace_fmt!("[TEST] Sent request {:#x}", req);

        // Receive response
        let (tok, resp) = recv(tok, resp_r);
        trace_fmt!("[TEST] Received respose {:#x}", resp);
        assert_eq(HuffmanWeightsDecoderStatus::OKAY, resp.status);
        assert_eq((((TEST_RAW_DATA[0] - u8:127) >> u32:1) + u8:2) as uN[TEST_AXI_ADDR_W], resp.tree_description_size);

        // Insert last weight in test data
        let last_weight_idx = ((TEST_RAW_DATA[0] as u32 - u32:127) / u32:2) + u32:1;
        let last_weight_entry = (
            TEST_RAW_DATA[last_weight_idx] |
            (TEST_RAW_DATA_LAST_WEIGHT << (u32:4 * (u32:1 - ((TEST_RAW_DATA[0] - u8:127) as u1 as u32))))
        );
        let test_data = update(TEST_RAW_DATA, last_weight_idx, last_weight_entry);

        // Check output RAM
        let tok = for (i, tok) in range(u32:0, u32:32) {
            let expected_value = if i < u32:16 {
                (
                    (test_data[4*i + u32:1] as u4) ++ ((test_data[4*i + u32:1] >> u32:4) as u4) ++
                    (test_data[4*i + u32:2] as u4) ++ ((test_data[4*i + u32:2] >> u32:4) as u4) ++
                    (test_data[4*i + u32:3] as u4) ++ ((test_data[4*i + u32:3] >> u32:4) as u4) ++
                    (test_data[4*i + u32:4] as u4) ++ ((test_data[4*i + u32:4] >> u32:4) as u4)
                )
            } else {
                u32:0
            };

            let weights_ram_rd_req = WeightsRamRdReq {
                addr: i as uN[TEST_WEIGHTS_RAM_ADDR_W],
                mask: !uN[TEST_WEIGHTS_RAM_NUM_PARTITIONS]:0,
            };
            let tok = send(tok, weights_ram_rd_req_s, weights_ram_rd_req);
            let (tok, weights_ram_rd_resp) = recv(tok, weights_ram_rd_resp_r);
            trace_fmt!("[TEST] Weights RAM content - addr: {:#x} data: expected {:#x}, got {:#x}", i, expected_value, weights_ram_rd_resp.data);

            assert_eq(expected_value, weights_ram_rd_resp.data);

            tok
        }(tok);


        // FSE-encoded weights
        unroll_for! (i, tok) in range(u32:0, array_size(TESTCASES_FSE)) {
            let (TEST_FSE_DATA, TEST_FSE_WEIGHTS) = TESTCASES_FSE[i];
            // Fill input RAM
            for (i, tok) in range(u32:0, (array_size(TEST_FSE_DATA) + TEST_DATA_PER_RAM_WRITE - u32:1) / TEST_DATA_PER_RAM_WRITE) {
                let ram_data = for (j, ram_data) in range(u32:0, TEST_DATA_PER_RAM_WRITE) {
                    let data_idx = i * TEST_DATA_PER_RAM_WRITE + j;
                    if (data_idx < array_size(TEST_FSE_DATA)) {
                        ram_data | ((TEST_FSE_DATA[data_idx] as uN[TEST_RAM_DATA_W]) << (u32:8 * j))
                    } else {
                        ram_data
                    }
                }(uN[TEST_RAM_DATA_W]:0);

                let input_ram_wr_req = InputBufferRamWrReq {
                    addr: (TEST_FSE_INPUT_ADDR / u32:8) + i as uN[TEST_RAM_ADDR_W],
                    data: ram_data,
                    mask: !uN[TEST_RAM_NUM_PARTITIONS]:0,
                };

                let tok = unroll_for! (i, tok) in range(u32:0, TEST_RAM_N) {
                    let tok = send(tok, input_ram_wr_req_s[i], input_ram_wr_req);
                    let (tok, _) = recv(tok, input_ram_wr_resp_r[i]);
                    tok
                }(tok);

                trace_fmt!("[TEST] Sent RAM write request to input RAMs {:#x}", input_ram_wr_req);

                tok
            }(tok);

            // Send decoding request
            let req = Req {
                addr: TEST_FSE_INPUT_ADDR,
            };
            let tok = send(tok, req_s, req);
            trace_fmt!("[TEST] Sent request {:#x}", req);

            // Receive response
            let (tok, resp) = recv(tok, resp_r);
            trace_fmt!("[TEST] Received respose {:#x}", resp);
            assert_eq(HuffmanWeightsDecoderStatus::OKAY, resp.status);
            assert_eq((TEST_FSE_DATA[0] + u8:1) as uN[TEST_AXI_ADDR_W], resp.tree_description_size);

            // Check output RAM
            let tok = for (i, tok) in range(u32:0, u32:32) {
                let expected_value = (
                    TEST_FSE_WEIGHTS[4*i + u32:0] ++
                    TEST_FSE_WEIGHTS[4*i + u32:1] ++
                    TEST_FSE_WEIGHTS[4*i + u32:2] ++
                    TEST_FSE_WEIGHTS[4*i + u32:3]
                );

                let weights_ram_rd_req = WeightsRamRdReq {
                    addr: i as uN[TEST_WEIGHTS_RAM_ADDR_W],
                    mask: !uN[TEST_WEIGHTS_RAM_NUM_PARTITIONS]:0,
                };
                let tok = send(tok, weights_ram_rd_req_s, weights_ram_rd_req);
                let (tok, weights_ram_rd_resp) = recv(tok, weights_ram_rd_resp_r);
                trace_fmt!("[TEST] Weights RAM content - addr: {:#x} data: expected {:#x}, got {:#x}", i, expected_value, weights_ram_rd_resp.data);

                assert_eq(expected_value, weights_ram_rd_resp.data);

                tok
            }(tok);
            tok
        }(tok);

        send(tok, terminator, true);
    }
}
