// Copyright 2021 The XLS Authors
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
proc second_level_proc {
  input_c: chan<u32> in;
  output_p: chan<u32> out;
  init { () }

  config(input_c: chan<u32> in, output_p: chan<u32> out) {
    (input_c, output_p)
  }

  next(tok: token, state: ()) { () }
}

proc first_level_proc {
  input_p0: chan<u32> out;
  input_p1: chan<u32> out;
  output_c0: chan<u32> in;
  output_c1: chan<u32> in;

  init { () }

  config() {
    let (input_p0, input_c0) = chan<u32>;
    let (output_p0, output_c0) = chan<u32>;
    spawn second_level_proc(input_c0, output_p0);

    let (input_p1, input_c1) = chan<u32>;
    let (output_p1, output_c1) = chan<u32>;
    spawn second_level_proc(input_c1, output_p1);

    (input_p0, input_p1, output_p0, output_p1)
  }

  next(tok: token, state: ()) { () }
}

#[test_proc]
proc main {
  terminator: chan<bool> out;
  init { () }
  config(terminator: chan<bool> out) {
    spawn first_level_proc();
    (terminator,)
  }

  next(tok: token, state: ()) {
    let tok = send(tok, terminator, true);
    ()
  }
}
