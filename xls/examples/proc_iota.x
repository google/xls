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
  config(input_c: chan out u32) {
    let c = input_c;
    ()
  }
  next(i: u32) {
    let foo = i + u32:1;
    send(c, foo);
    foo
  }

  c: chan out u32;
}

proc consumer<N:u32> {
  config(input_c: chan in u32) {
    let c = input_c;
    ()
  }
  next(i: u32) {
    let e = recv(c);
    i + e + N
  }
  c: chan in u32;
}

proc main {
  config() {
    let (p, c) = chan u32;
    spawn producer(p)(u32:0);
    spawn consumer<u32:2>(c)(u32:0)
  }
  next() {
    ()
  }
}
