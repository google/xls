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

proc Sender {
    ready_s: chan<bool> out;
    data_s: chan<u32> out;

    config(ready_s: chan<bool> out, data_s: chan<u32> out) { (ready_s, data_s) }

    init { () }

    next(state: ()) {
        let tok = send(join(), ready_s, true);
        send(tok, data_s, u32:42);
    }
}

#[test_proc]
proc SendAtomicityTest {
    terminator: chan<bool> out;
    ready_r: chan<bool> in;
    data_r: chan<u32> in;

    config(terminator: chan<bool> out) {
        let (ready_s, ready_r) = chan<bool, u32:1>("ready");
        let (data_s, data_r) = chan<u32, u32:1>("data");
        spawn Sender(ready_s, data_s);
        (terminator, ready_r, data_r)
    }

    init { u32:0 }

    next(tick: u32) {
        let (tok, _, ready)      = recv_non_blocking(join(), ready_r, false);
        let (tok, _, data_valid) = recv_non_blocking(tok, data_r, u32:0);
        assert_eq(!ready || data_valid, true);
        send_if(tok, terminator, tick == u32:9, true);
        tick + u32:1
    }
}
