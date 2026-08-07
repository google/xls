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

// Shows busperf catching a backpressure bottleneck: Passthrough feeds
// SlowConsumer through a depth-4 FIFO, and SlowConsumer<4> stalls it while
// SlowConsumer<0> stays healthy.

#![feature(type_inference_v2)]

proc Passthrough {
    data_r: chan<u32> in;
    data_s: chan<u32> out;

    config(
        data_r: chan<u32> in,
        data_s: chan<u32> out
    ) { (data_r, data_s) }

    init { }

    next(state: ()) {
        let (tok, data) = recv(join(), data_r);
        let tok = send(tok, data_s, data);
    }
}

proc SlowConsumer<STALL_CYCLES: u32> {
    data_r: chan<u32> in;
    data_s: chan<u32> out;

    config(
        data_r: chan<u32> in,
        data_s: chan<u32> out
    ) { (data_r, data_s) }

    init { u32:0 }

    next(state: u32) {
        let do_recv = const if STALL_CYCLES == u32:0 {
            true
        } else {
            state % STALL_CYCLES == u32:0
        };
        let (tok, data) = recv_if(join(), data_r, do_recv, u32:0);
        let tok = send_if(tok, data_s, do_recv, data);
        const if STALL_CYCLES == u32:0 {
            u32:0
        } else {
            (state + u32:1) % STALL_CYCLES
        }
    }
}

proc BottleneckNoStall {
    config(data_r: chan<u32> in, data_s: chan<u32> out) {
        let (fifo_s, fifo_r) = chan<u32, u32:4>("fifo");
        spawn Passthrough(data_r, fifo_s);
        spawn SlowConsumer<u32:0>(fifo_r, data_s);
    }
    init { }
    next(state: ()) { }
}

#[test_proc]
proc BottleneckNoStallTest {
    terminator: chan<bool> out;
    stim_s: chan<u32> out;
    resp_r: chan<u32> in;

    config(terminator: chan<bool> out) {
        let (stim_s, stim_r) = chan<u32>("stim");
        let (resp_s, resp_r) = chan<u32>("resp");
        spawn BottleneckNoStall(stim_r, resp_s);
        (terminator, stim_s, resp_r)
    }

    init { }

    next(state: ()) {
        const TEST_VALUE = u32:42;
        let tok = join();
        let tok = send(tok, stim_s, TEST_VALUE);
        let (tok, value) = recv(tok, resp_r);
        assert_eq(value, TEST_VALUE);
        send(tok, terminator, true);
    }
}

proc BottleneckStall {
    config(data_r: chan<u32> in, data_s: chan<u32> out) {
        let (fifo_s, fifo_r) = chan<u32, u32:4>("fifo");
        spawn Passthrough(data_r, fifo_s);
        spawn SlowConsumer<u32:4>(fifo_r, data_s);
    }
    init { }
    next(state: ()) { }
}

#[test_proc]
proc BottleneckStallTest {
    terminator: chan<bool> out;
    stim_s: chan<u32> out;
    resp_r: chan<u32> in;

    config(terminator: chan<bool> out) {
        let (stim_s, stim_r) = chan<u32>("stim");
        let (resp_s, resp_r) = chan<u32>("resp");
        spawn BottleneckStall(stim_r, resp_s);
        (terminator, stim_s, resp_r)
    }

    init { }

    next(state: ()) {
        const TEST_VALUE = u32:42;
        let tok = join();
        let tok = send(tok, stim_s, TEST_VALUE);
        let (tok, value) = recv(tok, resp_r);
        assert_eq(value, TEST_VALUE);
        send(tok, terminator, true);
    }
}
