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

// Test file containing multiple procs which all hit the same functions.

// multiple procs use this.
fn double_it(n: u32) -> u32 { n + n }

pub proc proc_double {
    bytes_src: chan<u32> in;
    bytes_result: chan<u32> out;

    config(bytes_src: chan<u32> in, bytes_result: chan<u32> out) { (bytes_src, bytes_result) }

    init { () }

    next(_: ()) {
        let tok = join();
        let (tok, v) = recv(tok, bytes_src);
        send(tok, bytes_result, double_it(v));
    }
}

pub proc proc_quad {
    bytes_src: chan<u32> in;
    bytes_result: chan<u32> out;

    config(bytes_src: chan<u32> in, bytes_result: chan<u32> out) { (bytes_src, bytes_result) }

    init { () }

    next(_: ()) {
        let tok = join();
        let (tok, v) = recv(tok, bytes_src);
        send(tok, bytes_result, double_it(double_it(v)));
    }
}

pub proc proc_ten {
    bytes_src: chan<u32> in;
    bytes_result: chan<u32> out;
    send_to_double: chan<u32> out;
    recv_double: chan<u32> in;
    send_to_quad: chan<u32> out;
    recv_quad: chan<u32> in;

    config(bytes_src: chan<u32> in, bytes_result: chan<u32> out) {
        let (send_double_out_side, send_double_in_side) = chan<u32>("send_double_pipe");
        let (send_quad_out_side, send_quad_in_side) = chan<u32>("send_quad_pipe");
        let (recv_double_out_side, recv_double_in_side) = chan<u32>("recv_double_pipe");
        let (recv_quad_out_side, recv_quad_in_side) = chan<u32>("recv_quad_pipe");
        spawn proc_double(send_double_in_side, recv_double_out_side);
        spawn proc_quad(send_quad_in_side, recv_quad_out_side);
        (
            bytes_src, bytes_result, send_double_out_side, recv_double_in_side, send_quad_out_side,
            recv_quad_in_side,
        )
    }

    init { () }

    next(_: ()) {
        let tok = join();
        let (tok, v) = recv(tok, bytes_src);
        let tok = send(tok, send_to_quad, v);
        let (tok, qv) = recv(tok, recv_quad);
        let ev = double_it(qv);
        let tok = send(tok, send_to_double, v);
        let (tok, dv) = recv(tok, recv_double);
        // 4x * 2 + 2x = 10x
        send(tok, bytes_result, ev + dv);
    }
}
