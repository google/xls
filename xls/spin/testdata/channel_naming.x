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

#![feature(type_inference_v2)]

proc MultiplyBy2 {
    req_r: chan<u32> in;
    resp_s: chan<u32> out;

    config(req_r: chan<u32> in, resp_s: chan<u32> out) { (req_r, resp_s) }

    init { () }

    next(state: ()) {
        let (tok, data) = recv(join(), req_r);
        let tok = send(tok, resp_s, data * 2);
    }
}

proc MultiplyBy4 {
    config(req_r: chan<u32> in, resp_s: chan<u32> out) {
        let (tmp_s, tmp_r) = chan<u32>("tmp");
        spawn MultiplyBy2(req_r, tmp_s);
        spawn MultiplyBy2(tmp_r, resp_s);
    }

    init { () }
    next(state: ()) { }
}

#[test_proc]
proc MultiplyBy4Test {
    terminator: chan<bool> out;
    req_s: chan<u32> out;
    resp_r: chan<u32> in;

    config(terminator: chan<bool> out) {
        let (req_s, req_r) = chan<u32>("req");
        let (resp_s, resp_r) = chan<u32>("resp");
        spawn MultiplyBy4(req_r, resp_s);
        (terminator, req_s, resp_r)
    }

    init {  }

    next(state: ()) {
        let tok = send(join(), req_s, u32:1);
        let (tok, received_data) = recv(tok, resp_r);

        assert_eq(received_data, u32:4);
        send(tok, terminator, true);
    }
}
