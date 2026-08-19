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

#![feature(type_inference_v2)]
#![feature(explicit_state_access)]
#![feature(generics)]

// A simple proc that forwards the received information from
// an input channel to an output channel.

proc Passthrough {
    data_r: chan<u32> in,
    data_s: chan<u32> out,
}

impl Passthrough {
    fn new(data_r: chan<u32> in, data_s: chan<u32> out) -> Self {
        Passthrough { data_r, data_s }
    }

    fn next(self) {
        let (tok, data) = recv(join(), self.data_r);
        let tok = send(tok, self.data_s, data);
    }
}

#[test]
proc PassthroughTest {
    terminator: chan<bool> out,
    data_s: chan<u32> out,
    data_r: chan<u32> in,
    count: u32,
}

impl PassthroughTest {
    fn new(terminator: chan<bool> out) -> Self {
        let (data_s, data_r) = chan<u32>("data");
        Passthrough::new(data_r, data_s).spawn();
        PassthroughTest { terminator, data_s, data_r, count: u32:10 }
    }

    fn next(self) {
        let count = read(self.count);
        let tok: token = join();
        let data_to_send = count * count;
        let tok = send(tok, self.data_s, data_to_send);
        let (tok, received_data) = recv(tok, self.data_r);

        assert_eq(data_to_send, received_data);
        send_if(tok, self.terminator, count == u32:0, true);
        write(self.count, count - u32:1);
    }
}
