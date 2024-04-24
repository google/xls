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

proc counter {
    output: chan<u8> out;

    config(output: chan<u8> out) { (output,) }

    init { u8:0 }

    next(tok: token, state: u8) {
        let tok = send(tok, output, state);
        state + u8:1
    }
}

#[test_proc]
proc counter_test {
    terminator: chan<bool> out;
    output_s: chan<u8> out;
    output_r: chan<u8> in;

    config(t: chan<bool> out) {
        let (output_s, output_r) = chan<u8>("output");
        spawn counter(output_s);
        (t, output_s, output_r)
    }

    init { () }

    next(tok: token, state: ()) {
        let increment = u8:1;
        for (_, expected) in u32:0..u32:32 {
            assert_eq(increment, u8:1);
            //assert_eq(state, ());
            let (tok, value) = recv(tok, output_r);
            assert_eq(value, expected);
            value + increment
        }(u8:0);
        let tok = send(tok, terminator, true);
    }
}
