// Copyright 2023 The XLS Authors
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

// Simple test for procs with only nonblocking recv/recv_if ops.

proc Main {
    do_recv_in: chan<bool> in;
    data_in: chan<u32> in;
    result_out: chan<(u32, bool)> out;

    config(do_recv_in: chan<bool> in, data_in: chan<u32> in, result_out: chan<(u32, bool)> out) {
        (do_recv_in, data_in, result_out)
    }

    init { () }

    next(tok: token, state: ()) {
        let (tok, do_recv, _) = recv_non_blocking(tok, do_recv_in, false);
        let (tok, foo, foo_valid) = recv_if_non_blocking(tok, data_in, do_recv, u32:42);
        send(tok, result_out, (foo, foo_valid));
    }
}

#[test_proc]
proc Tester {
    terminator: chan<bool> out;
    do_recv_out: chan<bool> out;
    data_out: chan<u32> out;
    result_in: chan<(u32, bool)> in;

    config(terminator: chan<bool> out) {
        let (do_recv_out, do_recv_in) = chan<bool>("do_recv");
        let (data_out, data_in) = chan<u32>("data");
        let (result_out, result_in) = chan<(u32, bool)>("result");
        spawn Main(do_recv_in, data_in, result_out);
        (terminator, do_recv_out, data_out, result_in)
    }

    init { () }

    next(tok: token, state: ()) {
        // First, tell the proc to receive without data present. Expect a
        // result of 42 (the supplied default value).
        let tok = send(tok, do_recv_out, true);
        let (tok, result) = recv(tok, result_in);
        assert_eq(result, (u32:42, false));

        // Next, tell the proc to NOT receive without data present.
        let tok = send(tok, do_recv_out, false);
        let (tok, result) = recv(tok, result_in);
        assert_eq(result, (u32:42, false));

        // Next, tell the proc to not receive with data present.
        let tok = send(tok, data_out, u32:2);
        let tok = send(tok, do_recv_out, false);
        let (tok, result) = recv(tok, result_in);
        assert_eq(result, (u32:42, false));

        // Finally, tell the proc to receive with data present. Expect the
        // previously-sent value.
        //
        // Multiple recvs are performed as one valid ordering is that the
        // Main proc executes multiple times before this proc's send.
        let tok = send(tok, do_recv_out, true);
        let (tok, result0) = recv(tok, result_in);
        let (tok, result1) = recv(tok, result_in);
        let (tok, result2) = recv(tok, result_in);
        assert_eq(result0.0 + result1.0 + result2.0, u32:86);

        let tok = send(tok, terminator, true);
    }
}
