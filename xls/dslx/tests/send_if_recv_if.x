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
    p: chan<u32> out;

    init { true }

    config(p: chan<u32> out) {
        (p,)
    }

    next(tok: token, do_send: bool) {
        let tok = send_if(tok, p, do_send, ((do_send) as u32));
        !do_send
    }
}

proc consumer {
    c: chan<u32> in;

    init { true }

    config(c: chan<u32> in) {
        (c,)
    }

    next(tok: token, do_recv: bool) {
        let (tok, foo) = recv_if(tok, c, do_recv);
        !do_recv
    }
}

proc main {
    init { () }
    config() {
        let (p, c) = chan<u32>;
        spawn producer(p);
        spawn consumer(c);
        ()
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
        let (data0_p, data0_c) = chan<u32>;
        let (data1_p, data1_c) = chan<u32>;

        spawn producer(data0_p);
        spawn consumer(data1_c);

        (terminator, data0_c, data1_p)
    }

    next(tok: token, state: ()) {
        // Sending consumer data.
        let tok = send(tok, data1, u32:10);
        let tok = send(tok, data1, u32:20);

        // Receiving producer data.
        let (tok, v) = recv(tok, data0);
        let _ = assert_eq(v, u32:1);

        let (tok, v) = recv(tok, data0);
        let _ = assert_eq(v, u32:1);

        let tok = send(tok, terminator, true);
        ()
    }
}
