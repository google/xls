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

import xls.examples.ram as ram;
import xls.modules.zstd.huffman_prescan as huffman_prescan;
import xls.modules.zstd.memory.axi as axi;
import xls.modules.zstd.memory.axi_ram as axi_ram;
import xls.modules.zstd.memory.mem_reader as mem_reader;
import xls.modules.zstd.ram_mux as ram_mux;

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
        } else {};

        // FIXME: support injecting the last implicit weight for Huffman Tree Descriptions larger
        // than a single bus width
        //
        // ^ This is partially done with saving the sum calculated below in the state.
        // When the last mem_reader packet is received (received the last chunk of HTD)
        // We should calculate the final sum and use it to calculate the `last_weight` of the HTD
        // and put it in a correct spot in the `weights` array

        // Calculate the last weight by summing 2^(weight - 1)
        // for each weight read from the HTD (excluding 0's)
        // The resulting sum must be subtracted from the next power of 2 after the resulting sum.
        // The result is the weight of the last literal.
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
        //trace_fmt!("[RAW] Reversed weights: {:#x}", reversed_weights);
        //trace_fmt!("[RAW] BUFF_LEN: {:#x}", BUFF_LEN);
        //trace_fmt!("[RAW] WEIGHTS_RAM_DATA_W: {:#x}", WEIGHTS_RAM_DATA_W);
        //trace_fmt!("[RAW] AXI_DATA_W: {:#x}", AXI_DATA_W);
        //trace_fmt!("[RAW] buffer_len: {:#x}", buffer_len);

        if do_recv_data && mem_rd_resp_valid {
            trace_fmt!("[RAW] Weights: {:#x}", weights);
        } else {};

        let (buffer, buffer_len) = if do_recv_data && mem_rd_resp_valid {
            //trace_fmt!("[RAW] shift: {:#x}", BUFF_LEN - AXI_DATA_W - buffer_len as u32);
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

struct HuffmanFseWeightsDecoderReq<AXI_ADDR_W: u32> {
    addr: uN[AXI_ADDR_W]
}

enum HuffmanFseWeightsDecoderStatus: u1 {
    OKAY = 0,
    ERROR = 1,
}

struct HuffmanFseWeightsDecoderResp {
    status: HuffmanFseWeightsDecoderStatus,
}

struct HuffmanFseWeightsDecoderState {
    placeholder: u1
}

proc HuffmanFseWeightsDecoder<
    AXI_ADDR_W: u32, AXI_DATA_W: u32,
    WEIGHTS_RAM_ADDR_W: u32, WEIGHTS_RAM_DATA_W: u32,
    WEIGHTS_RAM_NUM_PARTITIONS: u32
> {
    type Req = HuffmanFseWeightsDecoderReq<AXI_ADDR_W>;
    type Resp = HuffmanFseWeightsDecoderResp;
    type Status = HuffmanFseWeightsDecoderStatus;
    type State = HuffmanFseWeightsDecoderState;

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

        let (tok, req) = recv_if(tok, req_r, false, zero!<Req>());

        // Fetch Data
        let tok = send_if(tok, mem_rd_req_s, false, zero!<MemReaderReq>());
        let (tok, mem_rd_resp) = recv_if(tok, mem_rd_resp_r, false, zero!<MemReaderResp>());

        // Send to RAM
        let tok = send_if(tok, weights_ram_wr_req_s, false, zero!<WeightsRamWrReq>());
        let (tok, _) = recv_if(tok, weights_ram_wr_resp_r, false, zero!<WeightsRamWrResp>());

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
    AXI_ADDR_W: u32, AXI_DATA_W: u32,
    WEIGHTS_RAM_ADDR_W: u32, WEIGHTS_RAM_DATA_W: u32,
    WEIGHTS_RAM_NUM_PARTITIONS: u32
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
        fse_weights_mem_rd_req_s: chan<MemReaderReq> out,
        fse_weights_mem_rd_resp_r: chan<MemReaderResp> in,

        // Muxed internal RAM Write interface (goes to Huffman Weights Memory)
        weights_ram_wr_req_s: chan<WeightsRamWrReq> out,
        weights_ram_wr_resp_r: chan<WeightsRamWrResp> in,
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
        let (fse_weights_ram_wr_resp_s, fse_weights_ram_wr_resp_r) = chan<WeightsRamWrResp, u32:1>("fse_weights_ram_wr_resp");

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
            AXI_ADDR_W, AXI_DATA_W,
            WEIGHTS_RAM_ADDR_W, WEIGHTS_RAM_DATA_W,
            WEIGHTS_RAM_NUM_PARTITIONS
        >(
            fse_weights_req_r, fse_weights_resp_s,
            fse_weights_mem_rd_req_s, fse_weights_mem_rd_resp_r,
            fse_weights_ram_wr_req_s, fse_weights_ram_wr_resp_r
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
        trace_fmt!("Decoding FSE Huffman weights");
        let fse_weights_req = zero!<FseWeightsReq>();
        let tok = send_if(tok, fse_weights_req_s, weights_type == WeightsType::FSE, fse_weights_req);
        let (tok, fse_weights_resp) = recv_if(tok, fse_weights_resp_r, weights_type == WeightsType::FSE, zero!<FseWeightsResp>());

        let fse_status = match fse_weights_resp.status {
            HuffmanFseWeightsDecoderStatus::OKAY => HuffmanWeightsDecoderStatus::OKAY,
            HuffmanFseWeightsDecoderStatus::ERROR => HuffmanWeightsDecoderStatus::ERROR,
            _ => fail!("impossible_status_fse", HuffmanWeightsDecoderStatus::ERROR)
        };

        // RAW
        trace_fmt!("Decoding RAW Huffman weights");
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
                    tree_description_size: (((header_byte - u8:127) >> u8:1) + u8:1) as uN[AXI_ADDR_W],
                }
            },
            WeightsType::FSE => {
                Resp {
                    status: fse_status,
                    tree_description_size: header_byte as uN[AXI_ADDR_W],
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

    config (
        req_r: chan<Req> in,
        resp_s: chan<Resp> out,
        header_mem_rd_req_s: chan<MemReaderReq> out,
        header_mem_rd_resp_r: chan<MemReaderResp> in,
        raw_weights_mem_rd_req_s: chan<MemReaderReq> out,
        raw_weights_mem_rd_resp_r: chan<MemReaderResp> in,
        fse_weights_mem_rd_req_s: chan<MemReaderReq> out,
        fse_weights_mem_rd_resp_r: chan<MemReaderResp> in,
        weights_ram_wr_req_s: chan<WeightsRamWrReq> out,
        weights_ram_wr_resp_r: chan<WeightsRamWrResp> in,
    ) {
        spawn HuffmanWeightsDecoder<
            INST_AXI_ADDR_W, INST_AXI_DATA_W,
            INST_WEIGHTS_RAM_ADDR_W, INST_WEIGHTS_RAM_DATA_W,
            INST_WEIGHTS_RAM_NUM_PARTITIONS
        > (
            req_r, resp_s,
            header_mem_rd_req_s, header_mem_rd_resp_r,
            raw_weights_mem_rd_req_s, raw_weights_mem_rd_resp_r,
            fse_weights_mem_rd_req_s, fse_weights_mem_rd_resp_r,
            weights_ram_wr_req_s, weights_ram_wr_resp_r,
        );
    }

    init { }

    next (state: ()) { }
}


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

const TEST_INPUT_ADDR = uN[TEST_AXI_ADDR_W]:0x40;

// Weights sum is 1010, so the last one will be 14
const TEST_DATA = u8[65]:[
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

const TEST_DATA_LAST_WEIGHT = u8:0xA;

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

    terminator: chan<bool> out;

    req_s: chan<Req> out;
    resp_r: chan<Resp> in;

    input_ram_wr_req_s: chan<InputBufferRamWrReq>[3] out;
    input_ram_wr_resp_r: chan<InputBufferRamWrResp>[3] in;

    weights_ram_rd_req_s: chan<WeightsRamRdReq> out;
    weights_ram_rd_resp_r: chan<WeightsRamRdResp> in;

    config (terminator: chan<bool> out) {

        // Input Memory

        let (input_ram_rd_req_s, input_ram_rd_req_r) = chan<InputBufferRamRdReq>[3]("input_ram_rd_req");
        let (input_ram_rd_resp_s, input_ram_rd_resp_r) = chan<InputBufferRamRdResp>[3]("input_ram_rd_resp");
        let (input_ram_wr_req_s, input_ram_wr_req_r) = chan<InputBufferRamWrReq>[3]("input_ram_wr_req");
        let (input_ram_wr_resp_s, input_ram_wr_resp_r) = chan<InputBufferRamWrResp>[3]("input_ram_wr_resp");

        unroll_for! (i, _) in range(u32:0, u32:3) {
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

        let (axi_ar_s, axi_ar_r) = chan<AxiAr>[3]("axi_ar");
        let (axi_r_s, axi_r_r) = chan<AxiR>[3]("axi_r");

        unroll_for! (i, _) in range(u32:0, u32:3) {
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

        let (mem_rd_req_s, mem_rd_req_r) = chan<MemReaderReq>[3]("mem_rd_req");
        let (mem_rd_resp_s, mem_rd_resp_r) = chan<MemReaderResp>[3]("mem_rd_resp");

        unroll_for! (i, _) in range(u32:0, u32:3) {
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

        spawn HuffmanWeightsDecoder<
            TEST_AXI_ADDR_W, TEST_AXI_DATA_W,
            TEST_WEIGHTS_RAM_ADDR_W, TEST_WEIGHTS_RAM_DATA_W,
            TEST_WEIGHTS_RAM_NUM_PARTITIONS
        > (
            req_r, resp_s,
            mem_rd_req_s[0], mem_rd_resp_r[0],
            mem_rd_req_s[1], mem_rd_resp_r[1],
            mem_rd_req_s[2], mem_rd_resp_r[2],
            weights_ram_wr_req_s, weights_ram_wr_resp_r,
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

        // Fill input RAM
        for (i, tok) in range(u32:0, (array_size(TEST_DATA) + TEST_DATA_PER_RAM_WRITE - u32:1) / TEST_DATA_PER_RAM_WRITE) {
            let ram_data = for (j, ram_data) in range(u32:0, TEST_DATA_PER_RAM_WRITE) {
                let data_idx = i * TEST_DATA_PER_RAM_WRITE + j;
                if (data_idx < array_size(TEST_DATA)) {
                    ram_data | ((TEST_DATA[data_idx] as uN[TEST_RAM_DATA_W]) << (u32:8 * j))
                } else {
                    ram_data
                }
            }(uN[TEST_RAM_DATA_W]:0);

            let input_ram_wr_req = InputBufferRamWrReq {
                addr: (TEST_INPUT_ADDR / u32:8) + i as uN[TEST_RAM_ADDR_W],
                data: ram_data,
                mask: !uN[TEST_RAM_NUM_PARTITIONS]:0,
            };

            let tok = send(tok, input_ram_wr_req_s[0], input_ram_wr_req);
            let (tok, _) = recv(tok, input_ram_wr_resp_r[0]);
            let tok = send(tok, input_ram_wr_req_s[1], input_ram_wr_req);
            let (tok, _) = recv(tok, input_ram_wr_resp_r[1]);
            let tok = send(tok, input_ram_wr_req_s[2], input_ram_wr_req);
            let (tok, _) = recv(tok, input_ram_wr_resp_r[2]);

            trace_fmt!("[TEST] Sent RAM write request to input RAMs {:#x}", input_ram_wr_req);

            tok
        }(tok);

        // Send decoding request
        let req = Req {
            addr: TEST_INPUT_ADDR,
        };
        let tok = send(tok, req_s, req);
        trace_fmt!("[TEST] Sent request {:#x}", req);

        // Receive response
        let (tok, resp) = recv(tok, resp_r);
        trace_fmt!("[TEST] Received respose {:#x}", resp);
        assert_eq(HuffmanWeightsDecoderStatus::OKAY, resp.status);
        assert_eq(((TEST_DATA[0] - u8:127 + u8:1) >> u32:1) as uN[TEST_AXI_ADDR_W], resp.tree_description_size);

        // Insert last weight in test data
        let last_weight_idx = ((TEST_DATA[0] as u32 - u32:127) / u32:2) + u32:1;
        let last_weight_entry = (
            TEST_DATA[last_weight_idx] |
            (TEST_DATA_LAST_WEIGHT << (u32:4 * (u32:1 - ((TEST_DATA[0] - u8:127) as u1 as u32))))
        );
        let test_data = update(TEST_DATA, last_weight_idx, last_weight_entry);

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
            trace_fmt!("[TEST] Weights RAM content - addr: {:#x} data: {:#x}", i, weights_ram_rd_resp.data);

            assert_eq(expected_value, weights_ram_rd_resp.data);

            tok
        }(tok);

        send(tok, terminator, true);
    }
}
