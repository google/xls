// Copyright 2024 The XLS Authors
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

proc p {
    c0: chan<u32> in;
    c1: chan<u32> in;

    config(c0: chan<u32> in, c1: chan<u32> in) { (c0, c1) }

    init { u32:0 }

    next(state: u32) {
        let (tok', state'): (token, u32) = match state {
            u32:0 => recv(join(), c0),
            _ => recv(join(), c1),
        };
        state'
    }
}

#[test_proc]
proc main {
    c0: chan<u32> out;
    c1: chan<u32> out;
    terminator: chan<bool> out;

    config(terminator: chan<bool> out) {
        let (c0_s, c0_r) = chan<u32>("c0");
        let (c1_s, c1_r) = chan<u32>("c1");
        spawn p(c0_r, c1_r);
        (c0_s, c1_s, terminator)
    }

    init { () }

    next(state: ()) {
        let tok = send(join(), c0, u32:0);
        let tok = send(tok, c0, u32:0);
        let tok = send(tok, c0, u32:1);
        let tok = send(tok, c1, u32:0);
        let tok = send(tok, c0, u32:0);
        let tok = send(tok, c0, u32:0);
        let tok = send(tok, c0, u32:1);
        let tok = send(tok, c1, u32:0);
        let tok = send(tok, c0, u32:1);
        let tok = send(tok, c1, u32:1);
        let tok = send(tok, c1, u32:1);
        let tok = send(tok, c1, u32:1);

        let tok = send(tok, terminator, true);
    }
}
