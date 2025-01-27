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

// Proc used to test eval_proc_main's memory support.

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

    init { (u2:0, false) }

    next(st: (u2, bool)) {
        let (idx, read) = st;
        if read {
            let tkn = send(join(), mem_read_request, (idx, ()));
            let (tkn, (v)) = recv(tkn, mem_read_response);
            send(tkn, out_ch, v * u32:3);
        } else {
            let (tkn, v) = recv(join(), in_ch);
            let tkn = send(tkn, mem_write_request, (idx, v, ()));
            recv(tkn, mem_write_response);
        };
        let idx = idx + u2:1;
        let read = if idx != u2:0 { read } else { !read };
        (idx, read)
    }
}
