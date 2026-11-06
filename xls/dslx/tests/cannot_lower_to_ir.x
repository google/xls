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

// From `ProcWithUnconvertibleConfigGivesUsefulError` IR converter test
// as an example of proc that cannot be converted to IR.
proc Generator {
  out_ch: chan<u32> out;
  val: u32;

  init {()}
  config(out_ch: chan<u32> out, val: u32, val2: u32) {
    (out_ch, val + val2)
  }
  next(state: ()) {
    send(join(), out_ch, val);
  }
}

#[test_proc]
proc Testing {
  terminator: chan<bool> out;
  response: chan<u32> in;

  init {  }

  config(terminator: chan<bool> out){
    let (s, r) = chan<u32, u32:1>("test_chan");
    spawn Generator(s, u32:66, u32:99);
    (terminator, r)
  }

  next(state: ()) {
    let (tok, data) = recv(join(), response);
    send(tok, terminator, true);
  }
}
