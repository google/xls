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

proc Falsy {
    req_r: chan<()> in;
    resp_s: chan<bool> out;

    config(req_r: chan<()> in, resp_s: chan<bool> out) { (req_r, resp_s) }

    init {  }

    next(_: ()) {
        let (tok, _d) = recv(join(), req_r);
        let tok = send(tok, resp_s, false);
    }
}

proc Truthy {
    req_r: chan<()> in;
    resp_s: chan<bool> out;

    config(req_r: chan<()> in, resp_s: chan<bool> out) { (req_r, resp_s) }

    init {  }

    next(_: ()) {
        let (tok, _d) = recv(join(), req_r);
        let tok = send(tok, resp_s, true);
    }
}

proc Foo<CONFIG: bool> {
    config(req_r: chan<()> in, resp_s: chan<bool> out) {
        const if CONFIG { spawn Truthy(req_r, resp_s); } else { spawn Falsy(req_r, resp_s); };
        ()
    }

    init {  }

    next(_: ()) {  }
}

proc Main {
    config(req_r: chan<()>[2] in, resp_s: chan<bool>[2] out) {
        spawn Foo<true>(req_r[0], resp_s[0]);
        spawn Foo<false>(req_r[1], resp_s[1]);
        ()
    }

    init {  }

    next(_: ()) {  }
}

#[test_proc]
proc TestMain {
    req_s: chan<()>[2] out;
    resp_r: chan<bool>[2] in;
    terminator: chan<bool> out;

    config(terminator: chan<bool> out) {
        let (req_s, req_r) = chan<()>[2]("req");
        let (resp_s, resp_r) = chan<bool>[2]("resp");
        spawn Main(req_r, resp_s);

        (req_s, resp_r, terminator)
    }

    init {  }

    next(_: ()) {
        let tok = send(join(), req_s[0], ());
        let (tok, resp) = recv(tok, resp_r[0]);
        assert_eq(resp, true);
        let tok = send(join(), req_s[1], ());
        let (tok, resp) = recv(tok, resp_r[1]);
        assert_eq(resp, false);
        let tok = send(tok, terminator, true);
    }
}
