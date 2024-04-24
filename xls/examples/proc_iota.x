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
  s: chan<u32> out;

  init { u32:0 }

  config(input_s: chan<u32> out) {
    (input_s,)
  }

  next(tok: token, i: u32) {
    let foo = i + u32:1;
    let tok = send(tok, s, foo);
    foo
  }
}

proc consumer<N:u32> {
  r: chan<u32> in;

  init { u32: 0 }

  config(input_r: chan<u32> in) {
    (input_r,)
  }

  next(tok: token, i: u32) {
    let (tok, e) = recv(tok, r);
    i + e + N
  }
}

proc main {
  init { () }

  config() {
    let (s, r) = chan<u32>("my_chan");
    spawn producer(s);
    spawn consumer<u32:2>(r);
    ()
  }

  next(tok: token, state: ()) {
    ()
  }
}
