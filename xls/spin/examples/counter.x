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

// Counter sends values 0..9 then becomes a no-op; this bounds SPIN iterations.
proc Counter {
    data_s: chan<u32> out;
    config(data_s: chan<u32> out) { (data_s,) }
    init { u32:0 }
    next(state: u32) {
        send_if(join(), data_s, state < u32:10, state);
        state + u32:1
    }
}

#[test_proc]
proc CounterTest {
    terminator: chan<bool> out;
    data_r: chan<u32> in;
    config(terminator: chan<bool> out) {
        let (data_s, data_r) = chan<u32>("data");
        spawn Counter(data_s);
        (terminator, data_r)
    }

    init { u32:10 }

    next(count: u32) {
        let tok = join();
        let (tok, _value) = recv_if(tok, data_r, count > u32:0, u32:0);
        send_if(tok, terminator, count == u32:1, true);

        if count > u32:0 { count - u32:1 } else { u32:0 }
    }
}
