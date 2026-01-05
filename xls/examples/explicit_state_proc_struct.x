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

struct dummy_state {
  x: u32,
  y: u32,
}

#[test_proc]
proc explicit_state_proc_test {
  terminator: chan<bool> out;

  config(terminator: chan<bool> out) {
      (terminator,)
  }

  init { dummy_state { x: u32:0, y: u32:0 } }

  next(state: dummy_state) {
      let current = read(state.x);
      let y = labeled_read(state.y, "y_value");
      send_if(join(), terminator, current == u32:10 && y == u32:10, true);
      write(state.x, current + u32:1);
      labeled_write(state.y, y + u32:1, "y_value");
  }
}
