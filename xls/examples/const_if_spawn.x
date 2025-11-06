// Copyright 2022 The XLS Authors
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

proc Add3<WIDTH: u32> {
    in_r: chan<uN[WIDTH]> in;
    out_s: chan<uN[WIDTH]> out;
    init {}
    config(
        in_r: chan<uN[WIDTH]> in,
        out_s: chan<uN[WIDTH]> out
    ) { (in_r, out_s) }
    next(_: ()) {
        let (tok, data) = recv(join(), in_r);
        send(tok, out_s, data + uN[WIDTH]:3);
    }
}

proc Adapter<IN_W: u32, OUT_W: u32> {
    in_r: chan<uN[IN_W]> in;
    out_s: chan<uN[OUT_W]> out;
    init {}
    config(
        in_r: chan<uN[IN_W]> in,
        out_s: chan<uN[OUT_W]> out
    ) { (in_r, out_s) }
    next(_: ()) {
        let (tok, data) = recv(join(), in_r);
        send(tok, out_s, data as uN[OUT_W]);
    }
}


proc Add3Adv<IN_W: u32, OUT_W: u32 = {IN_W}> {
    config(
        in_r: chan<uN[IN_W]> in,
        out_s: chan<uN[OUT_W]> out
    ) {
        // TODO: Uncommented version should work too
        // const if IN_W != OUT_W {
            let (conn_s, conn_r) = chan<uN[OUT_W]>("conn");
            spawn Adapter<IN_W, OUT_W>(in_r, conn_s);
            spawn Add3<OUT_W>(conn_r, out_s);
        // } else {
           // spawn Add3<IN_W>(in_r, out_s);
        // };
    }

    init { () }
    next(state: ()) { }
}

proc Add3AdvInst {
    config(
        in_r: chan<u32> in,
        out_s: chan<u32> out
    ) {
        spawn Add3Adv<u32:32, u32:32>(in_r, out_s);
    }
    init { () }
    next(state: ()) { }
}

#[test_proc]
proc Add3AdvTest {
    terminator: chan<bool> out;
    in_s: chan<u32> out;
    out_r: chan<u64> in;

    init {}

    config (terminator: chan<bool> out) {
        let (in_s, in_r) = chan<u32>("in");
        let (out_s, out_r) = chan<u64>("out");
        spawn Add3Adv<u32:32, u32:64>(in_r, out_s);

        (terminator, in_s, out_r)
    }

    next(_: ()) {
        let tok = send(join(), in_s, u32:3);
        let (tok, result) = recv(tok, out_r);
        assert_eq(result, u64:6);
        send(tok, terminator, true);
    }
}
