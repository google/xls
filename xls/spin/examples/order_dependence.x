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

proc Worker<VALUE: u32> {
    req_r: chan<u32> in;
    resp_s: chan<u32> out;

    config(req_r: chan<u32> in, resp_s: chan<u32> out) {
        (req_r, resp_s)
    }

    init {  }

    next(state: ()) {
        let (tok, req) = recv(join(), req_r);
        let tok = send(tok, resp_s, req + VALUE);
    }
}

proc Receiver {
    req_r: chan<u32>[2] in;
    resp_s: chan<u32> out;

    config(req_r: chan<u32>[2] in, resp_s: chan<u32> out) {
        (req_r, resp_s)
    }

    init {  }

    next(state: ()) {
        let (tok0, req0, valid0) = recv_non_blocking(join(), req_r[0], u32:0);
        let (tok1, req1, valid1) = recv_non_blocking(join(), req_r[1], u32:0);
        let tok = send_if(join(tok0, tok1), resp_s, valid0 || valid1, req0 + req1);
    }
}

proc Arbiter {
    req_r: chan<u32> in;
    resp_s: chan<u32> out;
    worker_req_s: chan<u32>[2] out;
    receiver_resp_r: chan<u32> in;

    config(req_r: chan<u32> in, resp_s: chan<u32> out) {
        let (worker_req_s, worker_req_r) = chan<u32, u32:1>[2]("worker_req");
        let (receiver_req_s, receiver_req_r) = chan<u32, u32:1>[2]("receiver_req");
        let (receiver_resp_s, receiver_resp_r) = chan<u32, u32:1>("receiver_resp");

        spawn Worker<u32:5>(worker_req_r[0], receiver_req_s[0]);
        spawn Receiver(receiver_req_r, receiver_resp_s);
        spawn Worker<u32:16>(worker_req_r[1], receiver_req_s[1]);

        (req_r, resp_s, worker_req_s, receiver_resp_r)
    }

    init {  }

    next(state: ()) {
        let (tok, req, req_valid) = recv_non_blocking(join(), req_r, u32:0);
        let tok0 = send_if(tok, worker_req_s[0], req_valid, req);
        let tok1 = send_if(tok, worker_req_s[1], req_valid, req);

        let (tok, resp, valid) = recv_non_blocking(join(tok0, tok1), receiver_resp_r, u32:0);
        let tok = send_if(tok, resp_s, valid, resp);
    }
}

#[test_proc]
proc Tester {
    req_s: chan<u32> out;
    resp_r: chan<u32> in;
    terminator: chan<bool> out;

    config(terminator: chan<bool> out) {
        let (req_s, req_r) = chan<u32>("req");
        let (resp_s, resp_r) = chan<u32>("resp");
        spawn Arbiter(req_r, resp_s);

        (req_s, resp_r, terminator)
    }

    init {  }

    next(_: ()) {
        let tok = send(join(), req_s, u32:16);

        let (tok, resp) = recv(tok, resp_r);
        assert_eq(resp, u32:21);
        let tok = send(tok, terminator, true);
    }
}
