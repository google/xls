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

// Test case for https://github.com/google/xls/issues/1789

// Receives an input once and sends it twice.
proc bottom {
    i: chan<u32> in;
    o: chan<u32> out;

    config(i: chan<u32> in, o: chan<u32> out) { (i, o) }

    init { (u1:0, u32:0) }

    next(s: (u1, u32)) {
        let tok = join();
        let (tok, d) = recv_if(tok, i, s.0 == u1:0, u32:0);
        let next_d = if s.0 == u1:0 { d } else { s.1 };
        let tok = send(tok, o, next_d);
        (s.0 + u1:1, next_d)
    }
}

// Receives an input once and sends it four times by spawning bottom twice in series.
proc middle {
    config(i: chan<u32> in, o: chan<u32> out) {
        let (tmp0_s, tmp0_r) = chan<u32>("passthru");
        spawn bottom(i, tmp0_s);
        spawn bottom(tmp0_r, o);
    }

    init { () }

    next(state: ()) { () }
}

// Receives an input once and sends it four times by trivially spawning `middle`.
proc my_top {
    config(i: chan<u32> in, o: chan<u32> out) { spawn middle(i, o); }

    init { () }

    next(state: ()) { () }
}

#[test_proc]
proc test1 {
    terminator: chan<bool> out;
    i: chan<u32> out;
    o: chan<u32> in;

    config(terminator: chan<bool> out) {
        let (i_s, i_r) = chan<u32>("i_test1");
        let (o_s, o_r) = chan<u32>("o_test1");
        spawn my_top(i_r, o_s);
        (terminator, i_s, o_r)
    }

    init { () }

    next(state: ()) {
        let tok = join();
        let tok = send(tok, i, u32:100);

        let (tok, d) = recv(tok, o);
        assert_eq(d, u32:100);
        let (tok, d) = recv(tok, o);
        assert_eq(d, u32:100);
        let tok = send(tok, terminator, true);
    }
}

#[test_proc]
proc test2 {
    terminator: chan<bool> out;
    i: chan<u32> out;
    o: chan<u32> in;

    config(terminator: chan<bool> out) {
        let (i_s, i_r) = chan<u32>("i_test2");
        let (o_s, o_r) = chan<u32>("o_test2");
        spawn my_top(i_r, o_s);
        (terminator, i_s, o_r)
    }

    init { () }

    next(state: ()) {
        let tok = join();
        let tok = send(tok, i, u32:42);
        let (tok, d) = recv(tok, o);
        assert_eq(d, u32:42);
        let tok = send(tok, terminator, true);
    }
}
