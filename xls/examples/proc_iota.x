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

#![feature(type_inference_v2)]
#![feature(explicit_state_access)]
#![feature(generics)]

// Basic example showing how a proc network can be created and connected.

proc producer {
    s: chan<u32> out,
    i: u32,
}

impl producer {
    fn new(input_s: chan<u32> out) -> Self {
        producer { s: input_s, i: u32:0 }
    }

    fn next(self) {
        let i = read(self.i);
        let foo = i + u32:1;
        let tok = send(join(), self.s, foo);
        write(self.i, foo);
    }
}

proc consumer<N: u32> {
    r: chan<u32> in,
    i: u32,
}

impl consumer<N> {
    fn new(input_r: chan<u32> in) -> Self {
        consumer { r: input_r, i: u32:0 }
    }

    fn next(self) {
        let i = read(self.i);
        let (tok, e) = recv(join(), self.r);
        write(self.i, i + e + N);
    }
}

proc main {}

impl main {
    fn new() -> Self {
        let (s, r) = chan<u32, u32:1>("my_chan");
        producer::new(s).spawn();
        consumer<u32:2>::new(r).spawn();
        main {}
    }
}
