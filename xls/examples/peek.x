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

// Minimal peek and wait examples for development and testing purposes.

#![feature(type_inference_v2)]

proc Peek {
    req_r: chan<u32> in;
    resp_s: chan<u32> out;

    init {  }

    config(
        req_r: chan<u32> in,
        resp_s: chan<u32> out
    ) {
        (req_r, resp_s)
    }

    next(state: ()) {
        let (tok, packet, valid) = peek(join(), req_r, u32:0);
        let handle_packet = packet > u32:4;
        let (tok, packet) = recv_if(tok, req_r, valid && handle_packet, u32:0);
        send_if(tok, resp_s, valid, packet);
    }
}

#[test_proc]
proc PeekTest {
    req_s: chan<u32> out;
    resp_r: chan<u32> in;
    terminator: chan<bool> out;

    config(terminator: chan<bool> out) {
        let (req_s, req_r) = chan<u32>("req");
        let (resp_s, resp_r) = chan<u32>("resp");
        spawn Peek(req_r, resp_s);

        (req_s, resp_r, terminator)
    }

    init {  }

    next(_: ()) {
        // First packet which should be passed through.
        const FIRST_PACKET_DATA = u32:5;
        let tok = send(join(), req_s, FIRST_PACKET_DATA);
        let (tok, packet) = recv(tok, resp_r);
        assert_eq(packet, FIRST_PACKET_DATA);

        // Second packet which should be ignored.
        const SECOND_PACKET_DATA = u32:3;
        let tok = send(tok, req_s, SECOND_PACKET_DATA);
        let (tok, packet) = recv(tok, resp_r);
        assert_eq(packet, u32:0);

        send(tok, terminator, true);
    }
}

proc PeekIf {
    req_r: chan<u32> in;
    resp_s: chan<u32> out;
    enable_r: chan<bool> in;

    init { false }

    config(
        req_r: chan<u32> in,
        resp_s: chan<u32> out,
        enable_r: chan<bool> in,
    ) {
        (req_r, resp_s, enable_r)
    }

    next(state: bool) {
        let (tok, enabled, valid) = recv_non_blocking(join(), enable_r, state);
        let (tok, packet, packet_valid) = peek_if(join(), req_r, enabled, u32:0);
        let handle_packet = (packet > u32:4) && packet_valid;
        let packet_cond = !enabled || handle_packet;
        let (tok, packet) = recv_if(tok, req_r, packet_cond, u32:0);
        send(tok, resp_s, packet);
        enabled
    }
}

#[test_proc]
proc PeekIfTest {
    req_s: chan<u32> out;
    resp_r: chan<u32> in;
    enable_s: chan<bool> out;
    terminator: chan<bool> out;

    config(terminator: chan<bool> out) {
        let (req_s, req_r) = chan<u32>("req");
        let (resp_s, resp_r) = chan<u32>("resp");
        let (enable_s, enable_r) = chan<bool>("enable");
        spawn PeekIf(req_r, resp_s, enable_r);

        (req_s, resp_r, enable_s, terminator)
    }

    init {  }

    next(_: ()) {
        // First packet which should be passed through
        // as peek guard isn't enabled.
        const FIRST_PACKET_DATA = u32:3;
        let tok = send(join(), req_s, FIRST_PACKET_DATA);
        let (tok, packet) = recv(tok, resp_r);
        assert_eq(packet, FIRST_PACKET_DATA);

        let tok = send(tok, enable_s, true);

        // Second packet which should be ignored after enabling peek guard.
        const SECOND_PACKET_DATA = u32:2;
        let tok = send(tok, req_s, SECOND_PACKET_DATA);
        let (tok, packet) = recv(tok, resp_r);
        assert_eq(packet, u32:0);

        send(tok, terminator, true);
    }
}
