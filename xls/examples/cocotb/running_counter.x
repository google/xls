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

proc RunningCounter {
  base_r: chan<u32> in;
  cnt_s: chan<u32> out;

  init { u32:0 }

  config(base_r: chan<u32> in, cnt_s: chan<u32> out) {
    (base_r, cnt_s)
  }

  next(tok: token, cnt: u32) {
    let (tok, base, base_valid) = recv_non_blocking(tok, base_r, u32:0);
    let tok = send_if(tok, cnt_s, base_valid,  cnt + base);
    let cnt_next = cnt + u32:1;
    cnt_next
  }
}

#[test_proc]
proc RunningCounterTester {
  terminator: chan<bool> out;
  base_s: chan<u32> out;
  cnt_r: chan<u32> in;

  init { u32:0 }

  config (terminator: chan<bool> out) {
    let (base_s, base_r) = chan<u32>;
    let (cnt_s, cnt_r) = chan<u32>;

    spawn RunningCounter(base_r, cnt_s);
    (terminator, base_s, cnt_r)
  }

  next(tok: token, recv_cnt: u32) {
    let next_state = if (recv_cnt < u32:10) {
      let tok = send(tok, base_s, u32:1);
      let (tok, cnt) = recv(tok, cnt_r);

      trace_fmt!("Received {} cnt", recv_cnt);
      assert_lt(u32:1, cnt);

      recv_cnt + u32:1
    } else {
      send(tok, terminator, true);
      u32:0
    };

    (next_state)
  }
}
