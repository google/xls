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
    id: u32,
    data: u32,
}

proc Peek {
    req_r: chan<Packet> in;
    resp_s: chan<Packet> out;

    init {  }

    config(
        req_r: chan<Packet> in,
        resp_s: chan<Packet> out
    ) {
        (req_r, resp_s)
    }

    next(state: ()) {
        let (tok, packet) = peek(join(), req_r);
        let handle_packet = packet.id > u32:4;
        let (tok, packet) = recv_if(tok, req_r, handle_packet, zero!<Packet>());
        send(tok, resp_s, packet);
    }
}

#[test_proc]
proc PeekTest {
    req_s: chan<Packet> out;
    resp_r: chan<Packet> in;
    terminator: chan<bool> out;

    config(terminator: chan<bool> out) {
        let (req_s, req_r) = chan<Packet>("req");
        let (resp_s, resp_r) = chan<Packet>("resp");
        spawn Peek(req_r, resp_s);

        (req_s, resp_r, terminator)
    }

    init {  }

    next(_: ()) {
        // First packet
        const FIRST_PACKET_ID = u32:5;
        const FIRST_PACKET_DATA = u32:4;
        let tok = send(join(), req_s, Packet{
            id: FIRST_PACKET_ID,
            data: FIRST_PACKET_DATA
        });
        let (tok, packet) = recv(tok, resp_r);
        trace_fmt!("Received packet: {}", packet);

        // Second packet
        const SECOND_PACKET_ID = u32:3;
        const SECOND_PACKET_DATA = u32:16;
        let tok = send(tok, req_s, Packet{
            id: SECOND_PACKET_ID,
            data: SECOND_PACKET_DATA
        });
        let (tok, packet) = recv(tok, resp_r);
        trace_fmt!("Received packet: {}", packet);

        send(tok, terminator, true);
    }
}

proc PeekIf {
    req_r: chan<Packet> in;
    resp_s: chan<Packet> out;
    enable_r: chan<bool> in;

    init { false }

    config(
        req_r: chan<Packet> in,
        resp_s: chan<Packet> out,
        enable_r: chan<bool> in,
    ) {
        (req_r, resp_s, enable_r)
    }

    next(state: bool) {
        let (tok, enabled, valid) = recv_non_blocking(join(), enable_r, state);
        let state = if valid { enabled } else { state };
        let (tok, packet) = peek_if(join(), req_r, state, zero!<Packet>());
        let handle_packet = packet.id > u32:4;
        let (tok, packet) = recv_if(tok, req_r, state && handle_packet, zero!<Packet>());
        send_if(tok, resp_s, state && handle_packet, packet);
        state
    }
}

#[test_proc]
proc PeekIfTest {
    req_s: chan<Packet> out;
    resp_r: chan<Packet> in;
    enable_s: chan<bool> out;
    terminator: chan<bool> out;

    config(terminator: chan<bool> out) {
        let (req_s, req_r) = chan<Packet>("req");
        let (resp_s, resp_r) = chan<Packet>("resp");
        let (enable_s, enable_r) = chan<bool>("enable");
        spawn PeekIf(req_r, resp_s, enable_r);

        (req_s, resp_r, enable_s, terminator)
    }

    init {  }

    next(_: ()) {
        // First packet
        const FIRST_PACKET_ID = u32:5;
        const FIRST_PACKET_DATA = u32:4;
        let tok = send(join(), req_s, Packet{
            id: FIRST_PACKET_ID,
            data: FIRST_PACKET_DATA
        });
        let tok = send(tok, enable_s, true);
        let (tok, packet) = recv(tok, resp_r);
        trace_fmt!("Received packet: {}", packet);

        send(tok, terminator, true);
    }
}

proc PeekNonBlocking {
    req_r: chan<Packet> in;
    resp_s: chan<Packet> out;

    init {  }

    config(
        req_r: chan<Packet> in,
        resp_s: chan<Packet> out
    ) {
        (req_r, resp_s)
    }

    next(state: ()) {
        let (tok, packet, valid) = peek_non_blocking(join(), req_r, zero!<Packet>());
        let (tok, _) = recv_if(tok, req_r, valid, zero!<Packet>());
        send_if(tok, resp_s, valid, packet);
    }
}

#[test_proc]
proc PeekNonBlockingTest {
    req_s: chan<Packet> out;
    resp_r: chan<Packet> in;
    terminator: chan<bool> out;

    config(terminator: chan<bool> out) {
        let (req_s, req_r) = chan<Packet>("req");
        let (resp_s, resp_r) = chan<Packet>("resp");
        spawn PeekNonBlocking(req_r, resp_s);

        (req_s, resp_r, terminator)
    }

    init {  }

    next(_: ()) {
        // First packet
        const FIRST_PACKET_ID = u32:5;
        const FIRST_PACKET_DATA = u32:4;
        let tok = send(join(), req_s, Packet{
            id: FIRST_PACKET_ID,
            data: FIRST_PACKET_DATA
        });
        let (tok, packet) = recv(tok, resp_r);
        trace_fmt!("Received packet: {}", packet);

        // Second packet
        const SECOND_PACKET_ID = u32:3;
        const SECOND_PACKET_DATA = u32:16;
        let tok = send(tok, req_s, Packet{
            id: SECOND_PACKET_ID,
            data: SECOND_PACKET_DATA
        });
        let (tok, packet) = recv(tok, resp_r);
        trace_fmt!("Received packet: {}", packet);

        send(tok, terminator, true);
    }
}

proc PeekIfNonBlocking {
    req_r: chan<Packet> in;
    resp_s: chan<Packet> out;
    enable_r: chan<bool> in;

    init { false }

    config(
        req_r: chan<Packet> in,
        resp_s: chan<Packet> out,
        enable_r: chan<bool> in,
    ) {
        (req_r, resp_s, enable_r)
    }

    next(state: bool) {
        let (tok, enabled, valid) = recv_non_blocking(join(), enable_r, state);
        let state = if valid { enabled } else { state };
        let (tok, packet, packet_valid) = peek_if_non_blocking(join(), req_r, state, zero!<Packet>());
        let (tok, packet) = recv_if(tok, req_r, state && packet_valid, zero!<Packet>());
        send_if(tok, resp_s, state && packet_valid, packet);
        state
    }
}

#[test_proc]
proc PeekIfNonBlockingTest {
    req_s: chan<Packet> out;
    resp_r: chan<Packet> in;
    enable_s: chan<bool> out;
    terminator: chan<bool> out;

    config(terminator: chan<bool> out) {
        let (req_s, req_r) = chan<Packet>("req");
        let (resp_s, resp_r) = chan<Packet>("resp");
        let (enable_s, enable_r) = chan<bool>("enable");
        spawn PeekIfNonBlocking(req_r, resp_s, enable_r);

        (req_s, resp_r, enable_s, terminator)
    }

    init {  }

    next(_: ()) {
        // First packet
        const FIRST_PACKET_ID = u32:5;
        const FIRST_PACKET_DATA = u32:4;
        let tok = send(join(), req_s, Packet{
            id: FIRST_PACKET_ID,
            data: FIRST_PACKET_DATA
        });
        let tok = send(tok, enable_s, true);
        let (tok, packet) = recv(tok, resp_r);
        trace_fmt!("Received packet: {}", packet);

        send(tok, terminator, true);
    }
}
