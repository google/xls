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

#![feature(explicit_state_access)]
#![feature(generics)]

fn const_for_mask<N: u32>() -> u8 {
    const for (idx, mask): (u32, u8) in u32:0..N {
        mask | (u8:1 << idx)
    }(u8:0)
}

#[test]
fn const_for_test() {
    assert_eq(const_for_mask<u32:5>(), u8:0b11111);
    assert_eq(const_for_mask<u32:8>(), u8::MAX);
}

fn main() -> u8 { const_for_mask<u32:3>() }

proc ConstForInst {
    req_r: chan<()> in,
    resp_s: chan<u32> out,
}

impl ConstForInst {
    fn new(req_r: chan<()> in, resp_s: chan<u32> out) -> Self {
        ConstForInst { req_r, resp_s }
    }

    fn next(self) {
        let (tok, _d) = recv(join(), self.req_r);
        let result = const for (idx, mask): (u32, u32) in u32:0..u32:5 {
            mask | (u32:1 << idx)
        }(u32:0);
        let tok = send(tok, self.resp_s, result);
    }
}

const NUM_OF_CHANNELS = u32:2;

#[test]
proc ConstForProcSpawnTest {
    req_s: chan<()>[NUM_OF_CHANNELS] out,
    resp_r: chan<u32>[NUM_OF_CHANNELS] in,
    terminator: chan<bool> out,
}

impl ConstForProcSpawnTest {
    fn new(terminator: chan<bool> out) -> Self {
        let (req_s, req_r) = chan<()>[NUM_OF_CHANNELS]("req");
        let (resp_s, resp_r) = chan<u32>[NUM_OF_CHANNELS]("resp");
        const for (idx, _): (u32, ()) in u32:0..NUM_OF_CHANNELS {
            ConstForInst::new(req_r[idx], resp_s[idx]).spawn();
        }(());
        ConstForProcSpawnTest { req_s, resp_r, terminator }
    }

    fn next(self) {
        let tok = send(join(), self.req_s[0], ());
        let (tok, resp) = recv(tok, self.resp_r[0]);
        assert_eq(resp, u32:0b11111);

        let tok = send(join(), self.req_s[1], ());
        let (tok, resp) = recv(tok, self.resp_r[1]);
        assert_eq(resp, u32:0b11111);

        let tok = send(tok, self.terminator, true);
    }
}
