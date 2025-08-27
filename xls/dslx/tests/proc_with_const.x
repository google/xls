// Copyright 2021 The XLS Authors
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

// TODO(williamjhuang): This is a workaround for TIv2 since ConstantDefs in Proc are not handled
// correctly. Since we are going to switch to the new impl-based proc, this bug won't be fixed.
proc counter {
    output: chan<u8> out;

    config(output: chan<u8> out) { (output,) }

    init { u8:0 }

    next(state: u8) {
        const MAX_VAL = u8:15;
        let tok = send(join(), output, state);
        if state < MAX_VAL { state + u8:1 } else { MAX_VAL }
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

    next(state: ()) {
        let increment = u8:1;
        // First 15 should increment.
        for (_, expected) in u32:0..u32:15 {
            let (tok, value) = recv(join(), output_r);
            assert_eq(value, expected);
            expected + increment
        }(u8:0);
        // Afterwards, all values are capped at 15.
        for (_, count) in u32:0..u32:15 {
            let (tok, value) = recv(join(), output_r);
            assert_eq(value, u8:15);
            count + increment
        }(u8:0);
        let tok = send(join(), terminator, true);
    }
}
