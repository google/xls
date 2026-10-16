// Copyright 2026 The XLS Authors
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

// Proc used to test eval_proc_main's 1R1W memory read/write collision check.
// Performs 9 non-colliding simultaneous read+write requests, then collides on
// the 10th request by reading and writing the same address in the same cycle.

proc test_proc {
    in_ch: chan<u32> in;
    out_ch: chan<u32> out;
    mem_read_request: chan<(u2, ())> out;
    mem_read_response: chan<(u32)> in;
    mem_write_request: chan<(u2, u32, ())> out;
    mem_write_response: chan<()> in;

    config(in_ch: chan<u32> in, out_ch: chan<u32> out, mem_read_request: chan<(u2, ())> out,
           mem_read_response: chan<(u32)> in, mem_write_request: chan<(u2, u32, ())> out,
           mem_write_response: chan<()> in) {
        (in_ch, out_ch, mem_read_request, mem_read_response, mem_write_request, mem_write_response)
    }

    init { u4:0 }

    next(cnt: u4) {
        let wr_addr = cnt as u2;
        let rd_addr = if cnt == u4:9 { wr_addr } else { wr_addr - u2:1 };
        let (tkn, v) = recv(join(), in_ch);
        let tkn_rd = send(tkn, mem_read_request, (rd_addr, ()));
        let tkn_wr = send(tkn, mem_write_request, (wr_addr, v, ()));
        let (tkn_rd, (rd_val)) = recv(tkn_rd, mem_read_response);
        let (tkn_wr, ()) = recv(tkn_wr, mem_write_response);
        let tkn_out = join(tkn_rd, tkn_wr);
        send(tkn_out, out_ch, rd_val);
        cnt + u4:1
    }
}
