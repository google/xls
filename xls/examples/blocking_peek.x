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

struct Packet {
    id: u8,
    data: uN[1024]
}

// This proc presents a possible use case for the blocking variant of peek(),
// which may be useful when the decision about which packet to send must be
// made only after data becomes available on the input channel.
// When the input is available, the proc may decide either to forward/use
// that packet or to produce a different output.

// The proc below implements a packet filler. We assume that packet IDs
// are provided in monotonically increasing order, although some values
// may be missing. When this happens, the filler proc generates artificial
// packets so that all IDs in the sequence are produced.
//
// Implementation that uses recv is provided after the commented-out
// code, so it is possible to verify the expected behaviour of this proc.

struct PacketFillerState {
    packet: Packet, // this is large
    packet_valid: bool,
    current_id: u8
}

struct PeekPacketFillerState {
    current_id: u8
}

proc PeekPacketFiller {
    type State = PeekPacketFillerState;

    req_r: chan<Packet> in;
    resp_s: chan<Packet> out;

    init { zero!<State>() }

    config(
        req_r: chan<Packet> in,
        resp_s: chan<Packet> out
    ) {
        (req_r, resp_s)
    }

    next(state: State) {
        let (tok, packet) = peek(join(), req_r);
        let (packet, next_state) = if state.current_id < packet.id {
            // we need to generate an artificial packet
            (
                Packet { id: state.current_id, ..zero!<Packet>() },
                State { current_id: state.current_id + u8:1}
            )
        } else {
            // we can use a packet from the input
            let (tok, packet) = recv(tok, req_r);
            (
                packet,
                State { current_id: packet.id + u8:1 }
            )
        };

        let tok = send(tok, resp_s, packet);
        next_state
    }
}

#[test_proc]
proc PacketFillerTest {
    terminator: chan<bool> out;

    req_s: chan<Packet> out;
    resp_r: chan<Packet> in;

    init { }

    config(terminator: chan<bool> out) {
        let (req_s, req_r) = chan<Packet>("req");
        let (resp_s, resp_r) = chan<Packet>("resp");

        spawn PacketFiller(req_r, resp_s);
        (terminator, req_s, resp_r)
    }

    next(state: ()) {
        let tok = join();
        let tok = send(tok, req_s, Packet { id: 3, data: uN[1024]:0xA });
        let tok = send(tok, req_s, Packet { id: 5, data: uN[1024]:0xB });
        let tok = send(tok, req_s, Packet { id: 7, data: uN[1024]:0xC });

        let (tok, data) = recv(tok, resp_r);
        assert_eq(data, Packet { id: 0 , data: uN[1024]:0 });
        let (tok, data) = recv(tok, resp_r);
        assert_eq(data, Packet { id: 1 , data: uN[1024]:0 });
        let (tok, data) = recv(tok, resp_r);
        assert_eq(data, Packet { id: 2 , data: uN[1024]:0 });
        let (tok, data) = recv(tok, resp_r);
        assert_eq(data, Packet { id: 3 , data: uN[1024]:0xA });
        let (tok, data) = recv(tok, resp_r);
        assert_eq(data, Packet { id: 4 , data: uN[1024]:0 });
        let (tok, data) = recv(tok, resp_r);
        assert_eq(data, Packet { id: 5 , data: uN[1024]:0xB });
        let (tok, data) = recv(tok, resp_r);
        assert_eq(data, Packet { id: 6 , data: uN[1024]:0 });
        let (tok, data) = recv(tok, resp_r);
        assert_eq(data, Packet { id: 7 , data: uN[1024]:0xC });

        send(tok, terminator, true);
    }
}
