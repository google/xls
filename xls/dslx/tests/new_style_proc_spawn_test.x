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

proc Loopback {
    c_in: chan<u32> in,
    c_out: chan<u32> out,
}

impl Loopback {
    fn new(c_in: chan<u32> in, c_out: chan<u32> out) -> Self {
        Loopback { c_in, c_out }
    }

    fn next(self) {
        let (t, val) = recv(join(), self.c_in);
        send(t, self.c_out, val);
    }
}

#[test]
proc Main {
    test_terminator: chan<bool> out,
    c_in_from_loopback: chan<u32> in,
    c_out_to_loopback: chan<u32> out,
}

impl Main {
    fn new(test_terminator: chan<bool> out) -> Self {
        let (out_to_loopback, loopback_in) = chan<u32>("main_to_loopback");
        let (loopback_out, in_from_loopback) = chan<u32>("loopback_to_main");
        Loopback::new(loopback_in, loopback_out).spawn();

        Main {
            test_terminator,
            c_in_from_loopback: in_from_loopback,
            c_out_to_loopback: out_to_loopback,
        }
    }

    fn next(self) {
        let tok = send(join(), self.c_out_to_loopback, u32:42);
        let (tok, loopback_val) = recv(tok, self.c_in_from_loopback);
        assert_eq(loopback_val, u32:42);
        send(tok, self.test_terminator, true);
    }
}
