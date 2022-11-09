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

// Basic example showing how a proc network can be created and connected.

proc producer {
  c: chan<u32> out;

  init { u32:0 }

  config(input_c: chan<u32> out) {
    (input_c,)
  }

  next(tok: token, i: u32) {
    let foo = i + u32:1;
    let tok = send(tok, c, foo);
    foo
  }
}

proc consumer<N:u32> {
  c: chan<u32> in;

  init { u32: 0 }

  config(input_c: chan<u32> in) {
    (input_c,)
  }

  next(tok: token, i: u32) {
    let (tok, e) = recv(tok, c);
    i + e + N
  }
}

proc main {
  init { () }

  config() {
    let (p, c) = chan<u32>;
    spawn producer(p);
    spawn consumer<u32:2>(c);
    ()
  }

  next(tok: token, state: ()) {
    ()
  }
}
