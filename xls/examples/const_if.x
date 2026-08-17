// Copyright 2025 The XLS Authors
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

proc Falsy {
    req_r: chan<()> in,
    resp_s: chan<bool> out,
}

impl Falsy {
    fn new(req_r: chan<()> in, resp_s: chan<bool> out) -> Self {
        Falsy { req_r, resp_s }
    }

    fn next(self) {
        let (tok, _d) = recv(join(), self.req_r);
        let tok = send(tok, self.resp_s, false);
    }
}

proc Truthy {
    req_r: chan<()> in,
    resp_s: chan<bool> out,
}

impl Truthy {
    fn new(req_r: chan<()> in, resp_s: chan<bool> out) -> Self {
        Truthy { req_r, resp_s }
    }

    fn next(self) {
        let (tok, _d) = recv(join(), self.req_r);
        let tok = send(tok, self.resp_s, true);
    }
}

proc Foo<CONFIG: bool> {}

impl Foo<CONFIG> {
    fn new(req_r: chan<()> in, resp_s: chan<bool> out) -> Self {
        const if CONFIG {
            Truthy::new(req_r, resp_s).spawn();
        } else {
            Falsy::new(req_r, resp_s).spawn();
        };
        Foo {}
    }
}

proc Main {}

impl Main {
    fn new(req_r: chan<()>[2] in, resp_s: chan<bool>[2] out) -> Self {
        Foo<true>::new(req_r[0], resp_s[0]).spawn();
        Foo<false>::new(req_r[1], resp_s[1]).spawn();
        Main {}
    }
}

#[test]
proc TestMain {
    req_s: chan<()>[2] out,
    resp_r: chan<bool>[2] in,
    terminator: chan<bool> out,
}

impl TestMain {
    fn new(terminator: chan<bool> out) -> Self {
        let (req_s, req_r) = chan<()>[2]("req");
        let (resp_s, resp_r) = chan<bool>[2]("resp");
        Main::new(req_r, resp_s).spawn();
        TestMain { req_s, resp_r, terminator }
    }

    fn next(self) {
        let tok = send(join(), self.req_s[0], ());
        let (tok, resp) = recv(tok, self.resp_r[0]);
        assert_eq(resp, true);
        let tok = send(join(), self.req_s[1], ());
        let (tok, resp) = recv(tok, self.resp_r[1]);
        assert_eq(resp, false);
        let tok = send(tok, self.terminator, true);
    }
}
