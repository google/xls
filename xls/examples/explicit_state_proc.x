// Copyright 2026 The XLS Authors
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

#![feature(explicit_state_access)]

#[test_proc]
proc explicit_state_proc_test {
  terminator: chan<bool> out;

  config(terminator: chan<bool> out) {
      (terminator,)
  }

  init {
    u32:0
  }

  next (state: u32) {
    let current = read(state);
    let tok = trace_fmt!("current: {}", current);
    send_if(join(tok), terminator, current == u32:10, true);
    write(state, current + u32:1);
  }
}