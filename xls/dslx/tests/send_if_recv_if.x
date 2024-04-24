// Copyright 2020 The XLS Authors
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
proc producer {
    s: chan<u32> out;

    init { true }

    config(s: chan<u32> out) {
        (s,)
    }

    next(tok: token, do_send: bool) {
        let tok = send_if(tok, s, do_send, ((do_send) as u32));
        !do_send
    }
}

proc consumer {
    r: chan<u32> in;

    init { true }

    config(r: chan<u32> in) {
        (r,)
    }

    next(tok: token, do_recv: bool) {
        let (tok, _) = recv_if(tok, r, do_recv, u32:42);
        !do_recv
    }
}

proc main {
    init { () }
    config() {
        let (s, r) = chan<u32>("my_chan");
        spawn producer(s);
        spawn consumer(r);
    }
    next(tok: token, state: ()) { () }
}

#[test_proc]
proc test_main {
    terminator: chan<bool> out;
    data0: chan<u32> in;
    data1: chan<u32> out;

    init { () }

    config(terminator: chan<bool> out) {
        let (data0_s, data0_r) = chan<u32>("data0");
        let (data1_s, data1_r) = chan<u32>("data1");

        spawn producer(data0_s);
        spawn consumer(data1_r);

        (terminator, data0_r, data1_s)
    }

    next(tok: token, state: ()) {
        // Sending consumer data.
        let tok = send(tok, data1, u32:10);
        let tok = send(tok, data1, u32:20);

        // Receiving producer data.
        let (tok, v) = recv(tok, data0);
        assert_eq(v, u32:1);

        let (tok, v) = recv(tok, data0);
        assert_eq(v, u32:1);

        let tok = send(tok, terminator, true);
    }
}
