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

// A simple proc that forwards the received information from
// an input channel to an output channel.

import std;

proc Passthrough {
    data_r: chan<u32> in;
    data_s: chan<u32> out;

    config(data_r: chan<u32> in, data_s: chan<u32> out) { (data_r, data_s) }

    init { () }

    next(state: ()) {
        let (tok, data) = recv(join(), data_r);
        let tok = send(tok, data_s, data);
    }
}

#[test_proc]
proc PassthroughTest {
    terminator: chan<bool> out;
    data_s: chan<u32> out;
    data_r: chan<u32> in;

    config(terminator: chan<bool> out) {
        let (data_s, data_r) = chan<u32>("data");
        spawn Passthrough(data_r, data_s);
        (terminator, data_s, data_r)
    }

    init { u32:10 }

    next(count: u32) {
        let tok: token = join();
        let data_to_send = count * count;
        let tok = send(tok, data_s, data_to_send);
        let (tok, received_data) = recv(tok, data_r);

        assert_eq(data_to_send, received_data);
        send_if(tok, terminator, count == u32:0, true);

        count - u32:1
    }
}
