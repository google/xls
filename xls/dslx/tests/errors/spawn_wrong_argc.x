// Copyright 2023 The XLS Authors
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

import std;

pub proc foo {
  init { () }
  config () { () }

  next(state: ()) {
    std::min(u32:1, u32:2);
    ()
  }
}

#[test_proc]
proc test_case {
  terminator: chan<bool> out;

  init { () }

  config(terminator: chan<bool> out) {
    let (ch_in, ch_out) = chan<bool>("ch");
    spawn foo(ch_in, ch_out);
    (terminator, )
  }

  next(state: ()) {
    std::min(u32:1, u32:2);
    let tok = send(join(), terminator, true);
    ()
  }
}
