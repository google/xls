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
  consumer_input: chan in u32;

  config(c: chan in u32) {
    (u32:0, c)
  }

  next(tok: token) { () }
}

#![test_proc(u32:0)]
proc test_main {
  input_p: chan out u32;
  terminator_p: chan out bool;
  config(terminator_p: chan out bool) {
    let (p, c) = chan u32;
    spawn proc_under_test(c)();
    (p, terminator_p)
  }

  next(tok: token, iter: u32) {
    let tok = send_if(tok, terminator_p, iter == u32:2, true);
    (iter + u32:1,)
  }
}
