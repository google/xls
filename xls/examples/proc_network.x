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

// Simple example proc network showing spawning and channels.

proc A_proc {
    inp: chan<s32> in;
    output: chan<s32> out;
    ext: chan<s32> in;

    config(inp: chan<s32> in, output: chan<s32> out, ext: chan<s32> in) { (inp, output, ext) }

    init { () }

    next(st: ()) {
        let (tok, data_1) = recv(join(), ext);
        let (tok, data_2, _) = recv_non_blocking(join(), inp, s32:0);
        send(tok, output, data_1 + data_2 + s32:1);
    }
}

// Spawned 3 times.
proc B_proc {
    inp: chan<s32> in;
    output: chan<s32> out;

    config(inp: chan<s32> in, output: chan<s32> out) { (inp, output) }

    init { () }

    next(st: ()) {
        let (tok, data) = recv(join(), inp);
        send(tok, output, data + s32:100);
    }
}

proc C_proc {
    inp: chan<s32> in;
    output: chan<s32> out;
    ext: chan<s32> out;

    config(inp: chan<s32> in, output: chan<s32> out, ext: chan<s32> out) { (inp, output, ext) }

    init { () }

    next(st: ()) {
        let (tok, data) = recv(join(), inp);
        send(tok, output, data + s32:10000);
        send(tok, ext, data + s32:10000);
    }
}

// Generate a channel graph like
//
// A -> B1 -> B2 -> B3 -> C -> A
// EXT -> A
// C -> EXT
pub proc Initiator {
    ext_in: chan<s32> in;
    ext_out: chan<s32> out;
    init_snd: chan<s32> out;
    init_recv: chan<s32> in;

    config(ext_in: chan<s32> in, ext_out: chan<s32> out) {
        let (c_ext, init_rcv) = chan<s32>("init_in_chans");
        let (init_snd, a_ext) = chan<s32>("init_out_chans");
        let (a_to_b1_out, a_to_b1_in) = chan<s32>("a_to_b1");
        let (b1_to_b2_out, b1_to_b2_in) = chan<s32>("b1_to_b2");
        let (b2_to_b3_out, b2_to_b3_in) = chan<s32>("b2_to_b3");
        let (b3_to_c_out, b3_to_c_in) = chan<s32>("b3_to_c");
        let (c_to_a_out, c_to_a_in) = chan<s32>("c_to_a");
        spawn A_proc(c_to_a_in, a_to_b1_out, a_ext);
        spawn B_proc(a_to_b1_in, b1_to_b2_out);
        spawn B_proc(b1_to_b2_in, b2_to_b3_out);
        spawn B_proc(b2_to_b3_in, b3_to_c_out);
        spawn C_proc(b3_to_c_in, c_to_a_out, c_ext);
        (ext_in, ext_out, init_snd, init_rcv)
    }

    init { () }

    next(st: ()) {
        let (tok, data) = recv(join(), ext_in);
        let tok = send(tok, init_snd, data);
        let (tok, res) = recv(tok, init_recv);
        send(tok, ext_out, res);
    }
}

#[test_proc]
proc Testing {
    terminator: chan<bool> out;
    ext_send: chan<s32> out;
    ext_recv: chan<s32> in;

    config(terminator: chan<bool> out) {
        let (c_ext, init_snd) = chan<s32>("init_in_chans");
        let (init_recv, a_ext) = chan<s32>("init_out_chans");
        spawn Initiator(a_ext, c_ext);
        (terminator, init_recv, init_snd)
    }

    init { () }

    next(st: ()) {
        let tok = send(join(), ext_send, s32:0);
        let (tok, data) = recv(tok, ext_recv);
        assert_eq(data, s32:10301);
        send(tok, terminator, true);
    }
}
