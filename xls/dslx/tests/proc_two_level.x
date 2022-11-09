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

// Another proc "smoke test": this one just spawns two procs.

proc doubler {
  c: chan<u32> in;
  p: chan<u32> out;

  init { () }

  config(c: chan<u32> in, p: chan<u32> out) {
    (c, p)
  }

  next(tok: token, state: ()) {
    let (tok, input) = recv(tok, c);
    let tok = send(tok, p, input * u32:2);
    ()
  }
}

proc strange_mather {
  c: chan<u32> in;
  p: chan<u32> out;
  doubler_input_p: chan<u32> out;
  doubler_output_c: chan<u32> in;
  factor: u32;

  init { u32:0 }

  config(c: chan<u32> in, p: chan<u32> out, factor: u32) {
    let (doubler_input_c, doubler_input_p) = chan<u32>;
    let (doubler_output_c, doubler_output_p) = chan<u32>;
    spawn doubler(doubler_input_c, doubler_output_p);
    (c, p, doubler_input_p, doubler_output_c, factor)
  }

  next(tok: token, acc: u32) {
    let (tok, input) = recv(tok, c);

    let tok = send(tok, doubler_input_p, input);
    let (tok, double_input) = recv(tok, doubler_output_c);

    let tok = send(tok, p, acc);
    acc * factor + double_input
  }
}

#[test_proc]
proc test_proc {
  terminator: chan<bool> out;
  p: chan<u32> out;
  c: chan<u32> in;

  init { () }

  config(terminator: chan<bool> out) {
    let (input_p, input_c) = chan<u32>;
    let (output_p, output_c) = chan<u32>;
    spawn strange_mather(input_c, output_p, u32:2);
    (terminator, input_p, output_c)
  }

  next(tok: token, state: ()) {
    let tok = send(tok, p, u32:1);
    let (tok, res) = recv(tok, c);
    let _ = assert_eq(res, u32:0);
    let _ = trace!(res);

    let tok = send(tok, p, u32:1);
    let (tok, res) = recv(tok, c);
    let _ = assert_eq(res, u32:2);
    let _ = trace!(res);

    let tok = send(tok, p, u32:1);
    let (tok, res) = recv(tok, c);
    let _ = assert_eq(res, u32:6);
    let _ = trace!(res);

    let tok = send(tok, terminator, true);
    ()
  }
}
