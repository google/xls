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

proc proc_under_test {
  a: u32;
  input_r: chan<u32> in;

  init { () }

  config(r: chan<u32> in) {
    (u32:0, r)
  }

  next(tok: token, state: ()) {
    let (tok, val) = recv(tok, input_r);
    ()
  }
}

#[test_proc]
proc test_main {
  input_s: chan<u32> out;
  terminator_s: chan<bool> out;

  init { u32:0 }

  config(terminator_s: chan<bool> out) {
    let (s, r) = chan<u32>;
    spawn proc_under_test(r);
    (s, terminator_s)
  }

  // Run for two iterations then exit.
  next(tok: token, iter: u32) {
    let tok = send(tok, input_s, u32:0);
    let tok = send_if(tok, terminator_s, iter == u32:2, true);
    iter + u32:1
  }
}
