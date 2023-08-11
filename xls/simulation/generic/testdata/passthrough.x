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

import std

proc Passthrough {
  data_r: chan<u32> in;
  data_s: chan<u32> out;

  init {()}

  config(data_r: chan<u32> in, data_s: chan<u32> out) {
    (data_r, data_s)
  }

  next(tok: token, state: ()) {
    let (tok, data) = recv(tok, data_r);
    let tok = send(tok, data_s, data);
  }
}

#[test_proc]
proc PassthroughTester {
  terminator: chan<bool> out;
  data_s: chan<u32> out;
  data_r: chan<u32> in;

  init { u32:0 }

  config (terminator: chan<bool> out) {
    let (data_s, data_r) = chan<u32>;

    spawn Passthrough(data_r, data_s);
    (terminator, data_s, data_r)
  }

  next(tok: token, recv_cnt: u32) {
    let next_state = if (recv_cnt < u32:10) {
      let data_send = recv_cnt;
      let tok = send(tok, data_s, data_send);
      let (tok, data_recv) = recv(tok, data_r);

      assert_eq(data_send, data_recv);
      trace_fmt!("Received {} cnt", recv_cnt);
      recv_cnt + u32:1
    } else {
      send(tok, terminator, true);
      u32:0
    };

    (next_state)
  }
}
