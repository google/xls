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

// Demonstrates a scheduling-dependent bug that is hidden under the default
// fixed proc order and exposed by the randomized scheduler.
//
// PriorityMux tries WriterA first (non-blocking). If A is not ready it falls
// back to WriterB. Under fixed scheduling A always runs before PriorityMux,
// so A is always ready and the test passes. Under randomized scheduling
// PriorityMux may run before A, fall back to B, and fire the assertion.

#![feature(type_inference_v2)]

// Sends increasing values to its output
pub proc Writer<BASE_COUNT: u32> {
    out_s: chan<u32> out;

    config(out_s: chan<u32> out) { (out_s,) }

    init { u32:0 }

    next(count: u32) {
        send(join(), out_s, BASE_COUNT + count);
        count + u32:1
    }
}

// Priority mux: forwards A if available (non-blocking check), else B.
// Bug: relies on WriterA always running before this proc so that A's
// channel always has data when the non-blocking check is performed.
pub proc PriorityMux {
    a_r: chan<u32> in;
    b_r: chan<u32> in;
    out_s: chan<u32> out;

    config(a_r: chan<u32> in, b_r: chan<u32> in, out_s: chan<u32> out) {
        (a_r, b_r, out_s)
    }

    init { () }

    next(state: ()) {
        let (tok, val_a, got_a) = recv_non_blocking(join(), a_r, u32:0);
        let tok = if got_a {
            send(tok, out_s, val_a)
        } else {
            let (tok, val_b) = recv(tok, b_r);
            send(tok, out_s, val_b)
        };
    }
}

// Asserts that every value from PriorityMux is small (came from WriterA).
// Without scheduling randomization this always passes: WriterA runs before
// PriorityMux each tick, so A is always available and always wins.
// With randomized scheduling PriorityMux may run before WriterA, fall back
// to WriterB, and return a value >= 100, firing the assertion.
#[test_proc]
proc PriorityMuxTest {
    terminator: chan<bool> out;
    out_r: chan<u32> in;

    config(terminator: chan<bool> out) {
        const WRITER_A_BASE_COUNT = u32:0;
        const WRITER_B_BASE_COUNT = u32:100;

        let (a_s, a_r) = chan<u32>("writer_a");
        let (b_s, b_r) = chan<u32>("writer_b");
        let (out_s, out_r) = chan<u32>("mux_out");

        spawn Writer<WRITER_A_BASE_COUNT>(a_s);
        spawn Writer<WRITER_B_BASE_COUNT>(b_s);
        spawn PriorityMux(a_r, b_r, out_s);
        (terminator, out_r)
    }

    init { u32:0 }

    next(count: u32) {
        let (tok, val) = recv_if(join(), out_r, count < u32:3, u32:0);
        assert_eq(val < u32:10, true);
        let done = count == u32:2;
        send_if(tok, terminator, done, true);
        if done { u32:0 } else { count + u32:1 }
    }
}
