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

proc ValidDriver {
    valid_s: chan<bool> out;

    config(valid_s: chan<bool> out) { (valid_s,) }

    init { () }

    next(state: ()) {
        send(join(), valid_s, true);
    }
}

proc DataDriver {
    data_s: chan<u32> out;

    config(data_s: chan<u32> out) { (data_s,) }

    init { u32:0 }

    next(count: u32) {
        send(join(), data_s, count + u32:1);
        count + u32:1
    }
}

#[test_proc]
proc ValidDataRaceTest {
    terminator: chan<bool> out;
    valid_r: chan<bool> in;
    data_r: chan<u32> in;

    config(terminator: chan<bool> out) {
        let (valid_s, valid_r) = chan<bool, u32:1>("valid");
        let (data_s, data_r) = chan<u32, u32:1>("data");
        spawn ValidDriver(valid_s);
        spawn DataDriver(data_s);
        (terminator, valid_r, data_r)
    }

    init { u32:0 }

    next(tick: u32) {
        let (tok, _, valid_ok) = recv_non_blocking(join(), valid_r, false);
        let (tok, _, data_ok) = recv_non_blocking(tok, data_r, u32:0);
        assert_eq(valid_ok, data_ok);
        send_if(tok, terminator, tick == u32:9, true);
        tick + u32:1
    }
}
