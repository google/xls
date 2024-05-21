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

proc P<N: u32> {
    type MyUN = uN[N];
    s: chan<MyUN> out;

    config(s: chan<MyUN> out) { (s,) }

    init { MyUN:42 }

    next(state: MyUN) {
        send(join(), s, state);
        let new_state = state + MyUN:1;
        new_state
    }
}

#[test_proc]
proc TestProc {
    terminator: chan<bool> out;
    r: chan<u32> in;

    config(terminator: chan<bool> out) {
        let (s, r) = chan<u32>("my_chan");
        spawn P<u32:32>(s);
        (terminator, r)
    }

    init { () }

    next(state: ()) {
        let (tok, value) = recv(join(), r);
        trace!(value);
        assert_eq(value, u32:42);

        let tok = send(tok, terminator, true);
    }
}
