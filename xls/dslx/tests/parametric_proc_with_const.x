// Copyright 2020 The XLS Authors
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
proc parametric<N: u32, M: u32> {
    c: chan<uN[u32:37]> in;
    s: chan<uN[M]> out;

    config(c: chan<uN[M]> in, s: chan<uN[M]> out) { (c, s) }

    init { () }

    next(state: ()) {
        const DOUBLE_M = u32:37;
        let (tok, input) = recv(join(), c);
        let output = ((input as uN[DOUBLE_M]) * uN[DOUBLE_M]:2) as uN[M];
        let tok = send(tok, s, output);
    }
}

#[test_proc]
proc test_proc {
    terminator: chan<bool> out;
    output_c: chan<u37> in;
    input_p: chan<u37> out;

    config(terminator: chan<bool> out) {
        let (input_p, input_c) = chan<u37>("input");
        let (output_p, output_c) = chan<u37>("output");
        spawn parametric<u32:32, u32:37>(input_c, output_p);
        (terminator, output_c, input_p)
    }

    init { () }

    next(state: ()) {
        let tok = send(join(), input_p, u37:1);
        let (tok, result) = recv(tok, output_c);
        assert_eq(result, u37:2);

        let tok = send(tok, input_p, u37:8);
        let (tok, result) = recv(tok, output_c);
        assert_eq(result, u37:16);

        let tok = send(tok, terminator, true);
    }
}
