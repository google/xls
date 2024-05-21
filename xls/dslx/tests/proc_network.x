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

proc second_level_proc {
    input_r: chan<u32> in;
    output_s: chan<u32> out;

    config(input_r: chan<u32> in, output_s: chan<u32> out) { (input_r, output_s) }

    init { () }

    next(state: ()) { () }
}

proc first_level_proc {
    input_s0: chan<u32> out;
    input_s1: chan<u32> out;
    output_r0: chan<u32> in;
    output_r1: chan<u32> in;

    config() {
        let (input_s0, input_r0) = chan<u32>("input0");
        let (output_s0, output_r0) = chan<u32>("output0");
        spawn second_level_proc(input_r0, output_s0);

        let (input_s1, input_r1) = chan<u32>("input1");
        let (output_s1, output_r1) = chan<u32>("output1");
        spawn second_level_proc(input_r1, output_s1);

        (input_s0, input_s1, output_r0, output_r1)
    }

    init { () }

    next(state: ()) { () }
}

#[test_proc]
proc main {
    terminator: chan<bool> out;

    config(terminator: chan<bool> out) {
        spawn first_level_proc();
        (terminator,)
    }

    init { () }

    next(state: ()) { let tok = send(join(), terminator, true); }
}
