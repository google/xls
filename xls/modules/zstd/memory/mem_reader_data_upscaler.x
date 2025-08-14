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
import xls.modules.zstd.memory.mem_reader;

struct MemReaderDataUpscalerState<
    ADDR_W: u32, DATA_IN_W: u32, DATA_OUT_W: u32,
> {
    response: mem_reader::MemReaderResp<DATA_OUT_W, ADDR_W>,
}

pub proc MemReaderDataUpscaler<
    ADDR_W: u32, DATA_IN_W: u32, DATA_OUT_W: u32,
> {
    type InData = mem_reader::MemReaderResp<DATA_IN_W, ADDR_W>;
    type OutData = mem_reader::MemReaderResp<DATA_OUT_W, ADDR_W>;
    type State = MemReaderDataUpscalerState<ADDR_W, DATA_IN_W, DATA_OUT_W>;
    type Status = mem_reader::MemReaderStatus;

    const_assert!(DATA_IN_W <= DATA_OUT_W); // input should be narrower than output

    in_r: chan<InData> in;
    out_s: chan<OutData> out;

    config(
        in_r: chan<InData> in,
        out_s: chan<OutData> out
    ) { (in_r, out_s) }

    init { zero!<State>() }

    next(state: State) {
        const IN_FULL_LENGTH = (DATA_IN_W / u32:8) as uN[ADDR_W];
        const OUT_FULL_LENGTH = (DATA_OUT_W / u32:8) as uN[ADDR_W];
        let (tok, data) = recv(join(), in_r);
        trace_fmt!("[MemReaderDataUpscaler] data: {}", data);
        let last = data.length < IN_FULL_LENGTH;
        let shift_bits = state.response.length * uN[ADDR_W]:8;
        let response = OutData{
            status: Status::OKAY,
            data: (data.data as uN[DATA_OUT_W] << shift_bits) | state.response.data ,
            length: state.response.length + data.length,
            last: last
        };
        let tok = send_if(tok, out_s, response.length == OUT_FULL_LENGTH || last, response);

        State {
            response: if last { zero!<OutData>() } else { response }
        }
    }
}
