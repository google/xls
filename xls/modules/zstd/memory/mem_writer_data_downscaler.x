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
import xls.modules.zstd.memory.mem_writer;

struct MemWriterDataDownscalerState<
    ADDR_W: u32, DATA_IN_W: u32, DATA_OUT_W: u32,
    RATIO: u32, RATIO_W: u32,
> {
    packet: mem_writer::MemWriterDataPacket<DATA_IN_W, ADDR_W>,
    i: uN[RATIO_W],
}

pub proc MemWriterDataDownscaler<
    ADDR_W: u32, DATA_IN_W: u32, DATA_OUT_W: u32,
    RATIO: u32 = {std::ceil_div(DATA_IN_W, DATA_OUT_W)},
    RATIO_W: u32 = {u32:3}, //{std::clog2(RATIO + u32:1)}
> {
    type InData = mem_writer::MemWriterDataPacket<DATA_IN_W, ADDR_W>;
    type OutData = mem_writer::MemWriterDataPacket<DATA_OUT_W, ADDR_W>;
    type State = MemWriterDataDownscalerState<ADDR_W, DATA_IN_W, DATA_OUT_W, RATIO, RATIO_W>;

    const_assert!(DATA_IN_W >= DATA_OUT_W); // input should be wider than output

    in_r: chan<InData> in;
    out_s: chan<OutData> out;

    config(
        in_r: chan<InData> in,
        out_s: chan<OutData> out
    ) { (in_r, out_s) }

    init { zero!<State>() }

    next(state: State) {
        const FULL_LENGTH = (DATA_OUT_W / u32:8) as uN[ADDR_W];

        let do_recv = (state.i == uN[RATIO_W]:0);
        let (tok, packet) = recv_if(join(), in_r, do_recv, state.packet);

        trace_fmt!("packet: {}", packet);

        let data = packet.data[DATA_OUT_W * state.i as u32 +: uN[DATA_OUT_W]];
        let (length, last) = if packet.length > FULL_LENGTH {
            (FULL_LENGTH, u1:0)
        } else {
            (packet.length, packet.last)
        };

        let out_data = OutData { data, length, last };
        let tok = send(tok, out_s, out_data);

        if last {
            zero!<State>()
        } else {
            State {
                i:  state.i + uN[RATIO_W]:1,
                packet: InData {
                    data: packet.data,
                    last: packet.last,
                    length: packet.length - FULL_LENGTH
                }
            }
        }
    }
}

const INST_ADDR_W = u32:32;
const INST_DATA_IN_W = u32:144;
const INST_DATA_OUT_W = u32:32;

proc MemWriterDataDownscalerInst {
    type InDataPacket = mem_writer::MemWriterDataPacket<INST_DATA_IN_W, INST_ADDR_W>;
    type OutDataPacket = mem_writer::MemWriterDataPacket<INST_DATA_OUT_W, INST_ADDR_W>;

    config(
        in_r: chan<InDataPacket> in,
        out_s: chan<OutDataPacket> out
    ) {
        spawn MemWriterDataDownscaler<
            INST_ADDR_W, INST_DATA_IN_W, INST_DATA_OUT_W,
        >(in_r, out_s);
    }

    init {  }

    next(state: ()) {  }
}

const TEST_ADDR_W = u32:32;
const TEST_DATA_IN_W = u32:144;
const TEST_DATA_OUT_W = u32:32;

#[test_proc]
proc MemWriterDownscalerTest {
    type Addr = uN[TEST_ADDR_W];
    type InData = uN[TEST_DATA_IN_W];
    type InDataPacket = mem_writer::MemWriterDataPacket<TEST_DATA_IN_W, TEST_ADDR_W>;
    type OutData = uN[TEST_DATA_OUT_W];
    type OutDataPacket = mem_writer::MemWriterDataPacket<TEST_DATA_OUT_W, TEST_ADDR_W>;


    terminator: chan<bool> out;
    in_s: chan<InDataPacket> out;
    out_r: chan<OutDataPacket> in;

    config(terminator: chan<bool> out) {
        let (in_s, in_r) = chan<InDataPacket>("in");
        let (out_s, out_r) = chan<OutDataPacket>("out");

        spawn MemWriterDataDownscaler<
            TEST_ADDR_W, TEST_DATA_IN_W, TEST_DATA_OUT_W,
        > (in_r, out_s);

        (terminator, in_s, out_r)
    }

    init { }

    next(state: ()) {
        let tok = join();

        let tok = send(tok, in_s, InDataPacket {
            data: InData:0x2211_FFEE_DDCC_BBAA_0099_8877_6655_4433_2211,
            length: Addr:18,
            last: u1:1,
        });

        let (tok, data) = recv(tok, out_r);
        assert_eq(data, OutDataPacket {
            data: OutData:0x4433_2211,
            length: Addr:4,
            last: u1:0,
        });

        let (tok, data) = recv(tok, out_r);
        assert_eq(data, OutDataPacket {
            data: OutData:0x8877_6655,
            length: Addr:4,
            last: u1:0,
        });

        let (tok, data) = recv(tok, out_r);
        assert_eq(data, OutDataPacket {
            data: OutData:0xBBAA_0099,
            length: Addr:4,
            last: u1:0,
        });


        let (tok, data) = recv(tok, out_r);
        assert_eq(data, OutDataPacket {
            data: OutData:0xFFEE_DDCC,
            length: Addr:4,
            last: u1:0,
        });

        let (tok, data) = recv(tok, out_r);
        assert_eq(data, OutDataPacket {
            data: OutData:0x2211,
            length: Addr:2,
            last: u1:1,
        });

        send(tok, terminator, true);
    }
}
