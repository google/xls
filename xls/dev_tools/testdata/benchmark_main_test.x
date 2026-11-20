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

proc FailOnTick {
    in_ch: chan<u8> in;

    init { false }

    config(in_ch: chan<u8> in) { (in_ch,) }

    next(st: bool) {
        let (_tok, _val) = recv(join(), in_ch);
        assert!(false, "random_input_guard");
        st
    }
}

proc DeadlockAfterFirstTick {
    internal_in: chan<u8> in;
    internal_out: chan<u8> out;

    init { false }

    config() {
        let (s, r) = chan<u8>("internal_ch");
        (r, s)
    }

    next(armed: bool) {
        let tok = join();
        let (_tok_r, _val) = recv_if(tok, internal_in, armed, u8:0);
        let _tok_s = send_if(tok, internal_out, false, u8:0);
        true
    }
}

pub fn inv(x: bool) -> bool { !x }
