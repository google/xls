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

import std;

proc test_impl {
  in0: chan<u32> in;
  in1: chan<u32> in;
  out0 : chan<u32> out;

  init { u32:0 }

  config(in0: chan<u32> in,
         in1: chan<u32> in,
         out0: chan<u32> out) {
    (in0, in1, out0)
  }

  next(state: u32) {
    let (tok0, i0, valid0) = recv_non_blocking(join(), in0, u32:123);
    let (tok1, i1, valid1) = recv_non_blocking(join(), in1, u32:42);

    let o0 = u32:0;
    let o0 = if(valid0) { o0 + i0 } else { o0 };
    let o0 = if(valid1) { o0 + i1 } else { o0 };

    let tok_recv = join(tok0, tok1);
    let tok_send = send(tok_recv, out0, o0);

    let state = o0;

    state
  }
}

pub proc proc_main {
  init { () }

  config(in0: chan<u32> in,
         in1: chan<u32> in,
         out0: chan<u32> out) {
    spawn test_impl(in0, in1, out0);
  }

  next(state: ()) { () }
}

#[test_proc]
proc test_main {
  terminator: chan<bool> out;
  in0: chan<u32> out;
  in1: chan<u32> out;
  out0: chan<u32> in;

  init { () }

  config(terminator: chan<bool> out) {
    let (in0_s, in0_r) = chan<u32>("in0");
    let (in1_s, in1_r) = chan<u32>("in1");
    let (out0_s, out0_r) = chan<u32>("out0");

    spawn proc_main(in0_r, in1_r, out0_s);

    (terminator, in0_s, in1_s, out0_r)
  }

  next(state: ()) {
    // Not sending on either channel means output is 0.
    let (tok, v) = recv(join(), out0);
    assert_eq(v, u32:0);
    let (tok, v) = recv(tok, out0);
    assert_eq(v, u32:0);

    // Sending on on channel means output is the input.
    let tok = send(tok, in0, u32:3);
    let (tok, v) = recv(tok, out0);
    assert_eq(v, u32:3);

    let (tok, v) = recv(tok, out0);
    assert_eq(v, u32:0);

    let tok = send(tok, in1, u32:5);
    let (tok, v) = recv(tok, out0);
    assert_eq(v, u32:5);

    let (tok, v) = recv(tok, out0);
    assert_eq(v, u32:0);

    // Sending on both channels means output is the sum of inputs.
    let tok = send(tok, in0, u32:10);
    let tok = send(tok, in1, u32:20);
    let (tok, v) = recv(tok, out0);
    assert_eq(v, u32:30);

    let (tok, v) = recv(tok, out0);
    assert_eq(v, u32:0);

    let tok = send(tok, terminator, true);
  }
}
