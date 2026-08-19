#![feature(type_inference_v2)]
#![feature(explicit_state_access)]
#![feature(generics)]

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

import xls.examples.dslx_module.some_caps;

pub proc some_caps_streaming<N: u32> {
    bytes_src: chan<u8[N]> in,
    bytes_result: chan<u8[N]> out,
    state: some_caps::Choice,
}

impl some_caps_streaming<N> {
    fn new(bytes_src: chan<u8[N]> in, bytes_result: chan<u8[N]> out) -> Self {
        some_caps_streaming { bytes_src, bytes_result, state: some_caps::Choice::CAPITALIZE }
    }

    fn next(self) {
        let state = read(self.state);
        trace_fmt!("state={}", state);
        let (tok, val) = recv(join(), self.bytes_src);
        let ns = match state {
            some_caps::Choice::CAPITALIZE => some_caps::Choice::NOTHING,
            some_caps::Choice::NOTHING => some_caps::Choice::SPONGE,
            some_caps::Choice::SPONGE => some_caps::Choice::CAPITALIZE,
        };
        let tok = send(tok, self.bytes_result, some_caps::maybe_capitalize(val, state));
        write(self.state, ns);
    }
}

const TEST_SIZE = u32:6;

#[test]
proc test_streaming_somecaps {
    bytes_to_proc: chan<u8[TEST_SIZE]> out,
    bytes_from_proc: chan<u8[TEST_SIZE]> in,
    terminator: chan<bool> out,
}

impl test_streaming_somecaps {
    fn new(terminator: chan<bool> out) -> Self {
        let (send_to_proc_in, send_to_proc_out) = chan<u8[TEST_SIZE]>("send_to_proc");
        let (recv_from_proc_in, recv_from_proc_out) = chan<u8[TEST_SIZE]>("recv_from_proc");
        some_caps_streaming<TEST_SIZE>::new(send_to_proc_out, recv_from_proc_in).spawn();
        test_streaming_somecaps {
            bytes_to_proc: send_to_proc_in,
            bytes_from_proc: recv_from_proc_out,
            terminator,
        }
    }

    fn next(self) {
        // cap
        let tok = send(join(), self.bytes_to_proc, "foobar");
        let (tok, v) = recv(tok, self.bytes_from_proc);
        assert_eq(v, "FOOBAR");

        // nothing
        let tok = send(tok, self.bytes_to_proc, "foobar");
        let (tok, v) = recv(tok, self.bytes_from_proc);
        assert_eq(v, "foobar");

        // sponge
        let tok = send(tok, self.bytes_to_proc, "foobar");
        let (tok, v) = recv(tok, self.bytes_from_proc);
        assert_eq(v, "FoObAr");

        // cap
        let tok = send(tok, self.bytes_to_proc, "123456");
        let (tok, v) = recv(tok, self.bytes_from_proc);
        assert_eq(v, "123456");

        // nothing
        let tok = send(tok, self.bytes_to_proc, "123456");
        let (tok, v) = recv(tok, self.bytes_from_proc);
        assert_eq(v, "123456");

        // sponge
        let tok = send(tok, self.bytes_to_proc, "123456");
        let (tok, v) = recv(tok, self.bytes_from_proc);
        assert_eq(v, "123456");

        // cap
        let tok = send(tok, self.bytes_to_proc, "FOOBAR");
        let (tok, v) = recv(tok, self.bytes_from_proc);
        assert_eq(v, "FOOBAR");

        // nothing
        let tok = send(tok, self.bytes_to_proc, "FOOBAR");
        let (tok, v) = recv(tok, self.bytes_from_proc);
        assert_eq(v, "FOOBAR");

        // sponge
        let tok = send(tok, self.bytes_to_proc, "FOOBAR");
        let (tok, v) = recv(tok, self.bytes_from_proc);
        assert_eq(v, "FoObAr");

        // cap
        let tok = send(tok, self.bytes_to_proc, "fOoBaR");
        let (tok, v) = recv(tok, self.bytes_from_proc);
        assert_eq(v, "FOOBAR");

        // nothing
        let tok = send(tok, self.bytes_to_proc, "fOoBaR");
        let (tok, v) = recv(tok, self.bytes_from_proc);
        assert_eq(v, "fOoBaR");

        // sponge
        let tok = send(tok, self.bytes_to_proc, "fOoBaR");
        let (tok, v) = recv(tok, self.bytes_from_proc);
        assert_eq(v, "FoObAr");

        // cap
        let tok = send(tok, self.bytes_to_proc, "FoObAr");
        let (tok, v) = recv(tok, self.bytes_from_proc);
        assert_eq(v, "FOOBAR");

        // nothing
        let tok = send(tok, self.bytes_to_proc, "FoObAr");
        let (tok, v) = recv(tok, self.bytes_from_proc);
        assert_eq(v, "FoObAr");

        // sponge
        let tok = send(tok, self.bytes_to_proc, "FoObAr");
        let (tok, v) = recv(tok, self.bytes_from_proc);
        assert_eq(v, "FoObAr");

        send(tok, self.terminator, true);
    }
}
