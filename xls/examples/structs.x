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

type Num = u32;

struct Request {
    id: u32,
    a: Num,
    b: Num,
}

struct Data {
    id: u32,
    packet: Num,
    last: u1,
}

proc Sender {
    req_r: chan<Request> in;
    resp_s: chan<Data> out;

    config(req_r: chan<Request> in, resp_s: chan<Data> out) { (req_r, resp_s) }

    init { u32:0 }

    next(count: u32) {
        let (tok, req) = recv(join(), req_r);
        let tok = send(tok, resp_s, Data{
            id: req.id, packet: req.a + req.b, last: count >= u32:10});
        count + u32:1
    }
}

#[test_proc]
proc SenderTest {
    req_s: chan<Request> out;
    resp_r: chan<Data> in;
    terminator: chan<bool> out;

    config(terminator: chan<bool> out) {
        let (req_s, req_r) = chan<Request>("req");
        let (resp_s, resp_r) = chan<Data>("resp");
        spawn Sender(req_r, resp_s);

        (req_s, resp_r, terminator)
    }

    init {  }

    next(_: ()) {
        let tok = send(join(), req_s, Request{id: u32:5, a: u32:0xBE, b: u32:0xEF});
        let (tok, resp) = recv(tok, resp_r);
        assert_eq(resp.last, u1:0);
        send(tok, terminator, true);
    }
}
