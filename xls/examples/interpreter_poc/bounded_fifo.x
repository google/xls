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

#![feature(type_inference_v2)]

pub proc Passthrough {
    data_r: chan<u32> in;
    result_s: chan<u32> out;

    config(data_r: chan<u32> in, result_s: chan<u32> out) { (data_r, result_s) }

    init { () }

    next(state: ()) {
        let (tok, data) = recv(join(), data_r);
        let tok = send(tok, result_s, data);
    }
}

#[test_proc]
proc BoundedFifoTest {
    terminator: chan<bool> out;
    data_s: chan<u32> out;
    result_r: chan<u32> in;

    config(terminator: chan<bool> out) {
        let (data_s, data_r) = chan<u32, u32:1>("data");
        let (result_s, result_r) = chan<u32, u32:1>("result");

        spawn Passthrough(data_r, result_s);
        (terminator, data_s, result_r)
    }

    init { }

    next(_: ()) {
        let tok = join();
        let tok = send(tok, data_s, u32:1);
        let tok = send(tok, data_s, u32:2);
        let tok = send(tok, data_s, u32:3);

        let (tok, result) = recv(tok, result_r);
        assert_eq(result, u32:1);
        let (tok, result) = recv(tok, result_r);
        assert_eq(result, u32:2);
        let (tok, result) = recv(tok, result_r);
        assert_eq(result, u32:3);

        send(tok, terminator, true);
    }
}
