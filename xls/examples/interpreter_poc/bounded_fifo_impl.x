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
    data_r: chan<u32> in,
    result_s: chan<u32> out,
}
impl Passthrough {
    fn new(data_r: chan<u32> in, result_s: chan<u32> out) -> Self {
        Passthrough { data_r: data_r, result_s: result_s }
    }
    fn next(self) {
        let (tok, data) = recv(join(), self.data_r);
        let tok = send(tok, self.result_s, data);
    }
}

#[test]
proc BoundedFifoTest {
    terminator: chan<bool> out,
    data_s: chan<u32> out,
    result_r: chan<u32> in,
}
impl BoundedFifoTest {
    fn new(terminator: chan<bool> out) -> Self {
        let (data_s, data_r) = chan<u32, u32:1>("data");
        let (result_s, result_r) = chan<u32, u32:1>("result");
        Passthrough::new(data_r, result_s).spawn();
        BoundedFifoTest { terminator: terminator, data_s: data_s, result_r: result_r }
    }
    fn next(self) {
        let tok = join();
        let tok = send(tok, self.data_s, u32:1);
        let tok = send(tok, self.data_s, u32:2);
        let tok = send(tok, self.data_s, u32:3);

        let (tok, result) = recv(tok, self.result_r);
        assert_eq(result, u32:1);
        let (tok, result) = recv(tok, self.result_r);
        assert_eq(result, u32:2);
        let (tok, result) = recv(tok, self.result_r);
        assert_eq(result, u32:3);

        send(tok, self.terminator, true);
    }
}
