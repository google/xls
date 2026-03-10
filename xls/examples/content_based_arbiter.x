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

import std;

type Priority = u8;

struct Packet {
    priority: Priority,  // 0 is the highest prio
    data: u32,
}

struct ContentBasedArbiterState<N: u32> {
    enabled: bool,
    storage: Packet[N],
    storage_valid: bool[N],
}

fn highest_priority_packet<N: u32>(storage: Packet[N], storage_valid: bool[N]) -> (Packet, u32) {
    let (init_packet, init_idx) = for (i, (packet, idx)): (u32, (Packet, u32)) in 0..N {
        let i = N - i - 1;
        if storage_valid[i] { (storage[i], i) } else { (packet, idx) }
    }((zero!<Packet>(), u32:0));

    for (i, (packet, idx)): (u32, (Packet, u32)) in u32:0..N {
        let has_priority = storage_valid[i] && (storage[i].priority < packet.priority);
        if has_priority { (storage[i], i) } else { (packet, idx) }
    }((init_packet, init_idx))
}

pub proc ContentBasedArbiter<N: u32> {
    type State = ContentBasedArbiterState<N>;
    enable_r: chan<bool> in;
    enable_comp_s: chan<()> out;
    inputs_r: chan<Packet>[N] in;
    output_s: chan<Packet> out;

    config(
        enable_r: chan<bool> in,
        enable_comp_s: chan<()> out,
        inputs_r: chan<Packet>[N] in,
        output_s: chan<Packet> out
    ) {
        (enable_r, enable_comp_s, inputs_r, output_s)
    }

    init { zero!<State>() }

    next(state: State) {
        let (recv_en_tok, enabled, enabled_valid) =
            recv_non_blocking(join(), enable_r, state.enabled);
        let send_en_tok = send_if(recv_en_tok, enable_comp_s, enabled_valid, ());

        if state.enabled {
            let (storage, storage_valid, recv_in_tok) =
                unroll_for! (i, (storage, storage_valid, prev_tok)) in u32:0..N {
                    let (tok, data, data_valid) = recv_if_non_blocking(
                        join(), inputs_r[i], !state.storage_valid[i], state.storage[i]);
                    if data_valid {
                        (
                            update(storage, i, data), update(storage_valid, i, data_valid),
                            join(prev_tok, tok),
                        )
                    } else {
                        (storage, storage_valid, join(prev_tok, tok))
                    }
                }((state.storage, state.storage_valid, join()));

            let (packet, idx) = highest_priority_packet(storage, storage_valid);

            let has_value = or_reduce(std::convert_to_bits_msb0(storage_valid));
            let sent_out_tok = send_if(recv_in_tok, output_s, has_value, packet);
            let storage_valid = update(storage_valid, idx, false);

            State { enabled, storage, storage_valid }
        } else {
            State { enabled, ..state }
        }
    }
}

proc ContentBasedArbiterInst {
    config(
        enable_r: chan<bool> in,
        enable_comp_s: chan<()> out,
        inputs_r: chan<Packet>[3] in,
        output_s: chan<Packet> out
    ) {
        spawn ContentBasedArbiter<3>(enable_r, enable_comp_s, inputs_r, output_s);
    }

    init {  }
    next(state: ()) {  }
}

#[test_proc]
proc ContentBasedArbiterTest {
    terminator: chan<bool> out;
    enable_s: chan<bool> out;
    enable_comp_r: chan<()> in;
    inputs_s: chan<Packet>[3] out;
    output_r: chan<Packet> in;

    config(terminator: chan<bool> out) {
        let (enable_s, enable_r) = chan<bool>("enable");
        let (enable_comp_s, enable_comp_r) = chan<()>("enable_comp");
        let (inputs_s, inputs_r) = chan<Packet>[3]("inputs");
        let (output_s, output_r) = chan<Packet>("output");

        spawn ContentBasedArbiter<3>(enable_r, enable_comp_s, inputs_r, output_s);
        (terminator, enable_s, enable_comp_r, inputs_s, output_r)
    }

    init {  }

    next(state: ()) {
        let tok = join();

        // Send input data

        let tok = send(tok, inputs_s[0], Packet { priority: 3, data: 2 });
        let tok = send(tok, inputs_s[0], Packet { priority: 5, data: 5 });
        let tok = send(tok, inputs_s[0], Packet { priority: 7, data: 6 });

        let tok = send(tok, inputs_s[1], Packet { priority: 2, data: 0 });
        let tok = send(tok, inputs_s[1], Packet { priority: 0, data: 1 });
        let tok = send(tok, inputs_s[1], Packet { priority: 3, data: 3 });

        let tok = send(tok, inputs_s[2], Packet { priority: 4, data: 4 });
        let tok = send(tok, inputs_s[2], Packet { priority: 7, data: 7 });
        let tok = send(tok, inputs_s[2], Packet { priority: 0, data: 8 });

        // Enable arbiter

        let tok = send(tok, enable_s, true);
        let (tok, _) = recv(tok, enable_comp_r);

        // Collect output

        // I0: (p: 3), (p: 5), (p: 7)
        // I1: (p: 2), (p: 0), (p: 3)
        // I2: (p: 4), (p: 7), (p: 0)

        let (tok, data) = recv(tok, output_r);
        trace_fmt!("Received: {}", data);
        assert_eq(data, Packet { priority: 2, data: 0 });

        // I0: (p: 3), (p: 5), (p: 7)
        // I1: (p: 0), (p: 3)
        // I2: (p: 4), (p: 7), (p: 0)

        let (tok, data) = recv(tok, output_r);
        trace_fmt!("Received: {}", data);
        assert_eq(data, Packet { priority: 0, data: 1 });

        // I0: (p: 3), (p: 5), (p: 7)
        // I1: (p: 3)
        // I2: (p: 4), (p: 7), (p: 0)

        let (tok, data) = recv(tok, output_r);
        trace_fmt!("Received: {}", data);
        assert_eq(data, Packet { priority: 3, data: 2 });

        // I0: (p: 5), (p: 7)
        // I1: (p: 3)
        // I2: (p: 4), (p: 7), (p: 0)

        let (tok, data) = recv(tok, output_r);
        trace_fmt!("Received: {}", data);
        assert_eq(data, Packet { priority: 3, data: 3 });

        // I0: (p: 5), (p: 7)
        // I1:
        // I2: (p: 4), (p: 7), (p: 0)

        let (tok, data) = recv(tok, output_r);
        trace_fmt!("Received: {}", data);
        assert_eq(data, Packet { priority: 4, data: 4 });

        // I0: (p: 5), (p: 7)
        // I1:
        // I2: (p: 7), (p: 0)

        let (tok, data) = recv(tok, output_r);
        trace_fmt!("Received: {}", data);
        assert_eq(data, Packet { priority: 5, data: 5 });

        // I0: (p: 7)
        // I1:
        // I2: (p: 7), (p: 0)

        let (tok, data) = recv(tok, output_r);
        trace_fmt!("Received: {}", data);
        assert_eq(data, Packet { priority: 7, data: 6 });

        // I0:
        // I1:
        // I2: (p: 7), (p: 0)

        let (tok, data) = recv(tok, output_r);
        trace_fmt!("Received: {}", data);
        assert_eq(data, Packet { priority: 7, data: 7 });

        // I0:
        // I1:
        // I2: (p: 0)

        let (tok, data) = recv(tok, output_r);
        trace_fmt!("Received: {}", data);
        assert_eq(data, Packet { priority: 0, data: 8 });

        send(tok, terminator, true);
    }
}
