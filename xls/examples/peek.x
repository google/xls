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

struct Packet {
    id: u8,
    data: uN[1024]
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
proc Test {
    req_s: chan<Packet> out;
    resp_r: chan<Packet> in;
    terminator: chan<bool> out;

    config(terminator: chan<bool> out) {
        let (req_s, req_r) = chan<Packet>("req");
        let (resp_s, resp_r) = chan<Packet>("resp");
        spawn PeekPacketFiller(req_r, resp_s);

        (req_s, resp_r, terminator)
    }

    init {  }

    next(_: ()) {
        const PACKET_ID = 5;
        let tok = send(join(), req_s, Packet{id: PACKET_ID, zero!<Packet>()});
        const for (_, _): (u32, ()) in u32:0..PACKET_ID {
            let (tok, packet) = recv(tok, resp_r);
            trace_fmt!("Packet: {:#x}", packet);
        }(());
        let (tok, packet) = recv(tok, resp_r);
        trace_fmt!("Last packet: {:#x}", packet);
    }
}
