// Copyright 2022 The XLS Authors
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

// Simple test for the nonblocking recv_if op.

proc Main {
    do_recv_in: chan<bool> in;
    data_in: chan<u32> in;
    result_out: chan<(u32, bool)> out;

    init { () }

    config(do_recv_in: chan<bool> in,
           data_in: chan<u32> in,
           result_out: chan<(u32, bool)> out) {
        (do_recv_in, data_in, result_out)
    }

    next(tok: token, state: ()) {
        let (tok, do_recv) = recv(tok, do_recv_in);
        let (tok, foo, foo_valid) = recv_if_non_blocking(tok, data_in, do_recv);
        let _ = send(tok, result_out, (foo, foo_valid));
        ()
    }
}

#[test_proc]
proc Tester {
    terminator: chan<bool> out;
    do_recv_out: chan<bool> out;
    data_out: chan<u32> out;
    result_in: chan<(u32, bool)> in;

    init { () }

    config(terminator: chan<bool> out) {
        let (do_recv_out, do_recv_in) = chan<bool>;
        let (data_out, data_in) = chan<u32>;
        let (result_out, result_in) = chan<(u32, bool)>;
        spawn Main(do_recv_in, data_in, result_out);
        (terminator, do_recv_out, data_out, result_in)
    }

    next(tok: token, state: ()) {
        // First, tell the proc to receive without data present. Expect a
        // result of 0.
        let tok = send(tok, do_recv_out, true);
        let (tok, result) = recv(tok, result_in);
        let _ = assert_eq(result, (u32:0, false));

        // Next, tell the proc to NOT receive without data present.
        let tok = send(tok, do_recv_out, false);
        let (tok, result) = recv(tok, result_in);
        let _ = assert_eq(result, (u32:0, false));

        // Next, tell the proc to not receive with data present.
        let tok = send(tok, data_out, u32:2);
        let tok = send(tok, do_recv_out, false);
        let (tok, result) = recv(tok, result_in);
        let _ = assert_eq(result, (u32:0, false));

        // Finally, tell the proc to receive with data present. Expect the
        // previously-sent value.
        let tok = send(tok, do_recv_out, true);
        let (tok, result) = recv(tok, result_in);
        let _ = assert_eq(result, (u32:2, true));

        let tok = send(tok, terminator, true);
        ()
    }
}
