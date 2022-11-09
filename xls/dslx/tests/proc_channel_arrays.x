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

// Test demonstrating use (and correct implementation) of arrays of channels.
proc consumer {
  cs: chan<u16>[128][64] in;
  ps: chan<u16>[128] out;
  init { () }
  config(cs: chan<u16>[128][64] out, ps: chan<u16>[128] in) {
    (cs, ps)
  }
  next(tok: token, state: ()) {
    let (tok, i) = recv(tok, cs[0][0]);
    let tok = send(tok, ps[1], i + i);
    ()
  }
}

#[test_proc]
proc producer {
  ps: chan<u16>[128][64][32] out;
  cs: chan<u16>[128][64][32] in;
  terminator: chan<bool> out;

  init { () }

  config(terminator: chan<bool> out) {
    let (ps, cs) = chan<u16>[128][64][32];
    spawn consumer(cs[0], ps[0][1]);
    (ps, cs, terminator)
  }

  next(tok: token, state: ()) {
    let tok = send(tok, ps[0][0][0], u16:1);
    let (tok, result) = recv(tok, cs[0][1][1]);
    let _ = assert_eq(result, u16:2);

    let tok = send(tok, terminator, true);
    ()
  }
}


