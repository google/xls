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

proc P {
    type MyU32 = u32;
    s: chan<MyU32> out;

    config(s: chan<MyU32> out) { (s,) }

    init { MyU32:42 }

    next(tok: token, state: MyU32) {
        send(tok, s, state);
        let new_state = state + MyU32:1;
        new_state
    }
}

#[test_proc]
proc TestProc {
    terminator: chan<bool> out;
    r: chan<u32> in;

    config(terminator: chan<bool> out) {
        let (s, r) = chan<u32>("my_chan");
        spawn P(s);
        (terminator, r)
    }

    init { () }

    next(tok: token, state: ()) {
        let (tok, value) = recv(tok, r);
        trace!(value);
        assert_eq(value, u32:42);

        let tok = send(tok, terminator, true);
    }
}
