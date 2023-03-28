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
  r: chan<u32> in;
  s: chan<u32> out;

  init { () }

  config(r: chan<u32> in, s: chan<u32> out) {
    (r, s)
  }

  next(tok: token, state: ()) {
    let (tok, input) = recv(tok, r);
    let tok = send(tok, s, input * u32:2);
    ()
  }
}

proc strange_mather {
  r: chan<u32> in;
  s: chan<u32> out;
  doubler_input_s: chan<u32> out;
  doubler_output_r: chan<u32> in;
  factor: u32;

  init { u32:0 }

  config(r: chan<u32> in, s: chan<u32> out, factor: u32) {
    let (doubler_input_s, doubler_input_r) = chan<u32>;
    let (doubler_output_s, doubler_output_r) = chan<u32>;
    spawn doubler(doubler_input_r, doubler_output_s);
    (r, s, doubler_input_s, doubler_output_r, factor)
  }

  next(tok: token, acc: u32) {
    let (tok, input) = recv(tok, r);

    let tok = send(tok, doubler_input_s, input);
    let (tok, double_input) = recv(tok, doubler_output_r);

    let tok = send(tok, s, acc);
    acc * factor + double_input
  }
}

#[test_proc]
proc test_proc {
  terminator: chan<bool> out;
  s: chan<u32> out;
  r: chan<u32> in;

  init { () }

  config(terminator: chan<bool> out) {
    let (input_s, input_r) = chan<u32>;
    let (output_s, output_r) = chan<u32>;
    spawn strange_mather(input_r, output_s, u32:2);
    (terminator, input_s, output_r)
  }

  next(tok: token, state: ()) {
    let tok = send(tok, s, u32:1);
    let (tok, res) = recv(tok, r);
    let _ = assert_eq(res, u32:0);
    let _ = trace!(res);

    let tok = send(tok, s, u32:1);
    let (tok, res) = recv(tok, r);
    let _ = assert_eq(res, u32:2);
    let _ = trace!(res);

    let tok = send(tok, s, u32:1);
    let (tok, res) = recv(tok, r);
    let _ = assert_eq(res, u32:6);
    let _ = trace!(res);

    let tok = send(tok, terminator, true);
    ()
  }
}
