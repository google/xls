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

// Basic example of a proc with a scheduling constraint; the send and receive
// are scheduled to be exactly 2 cycles apart in the BUILD file.

proc main {
  req: chan<u32> out;
  resp: chan<u32> in;

  init { u32: 0 }

  config(req: chan<u32> out, resp: chan<u32> in) {
    (req, resp)
  }

  next(state: u32) {
    let request = state * state;
    let tok = send(join(), req, request);
    let (tok, _response) = recv(tok, resp);
    state + u32:1
  }
}
