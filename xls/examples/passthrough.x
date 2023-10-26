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

// A simple proc that sends back the information received on
// an input channel over an output channel.

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

const NUMBER_OF_TESTED_TRANSACTIONS = u32:10;

#[test_proc]
proc PassthroughTest {
  terminator: chan<bool> out;
  data_s: chan<u32> out;
  data_r: chan<u32> in;

  init { u32:0 }

  config (terminator: chan<bool> out) {
    let (data_s, data_r) = chan<u32>;
    spawn Passthrough(data_r, data_s);
    (terminator, data_s, data_r)
  }

  next(tok: token, count: u32) {
    let data_to_send = count;
    let tok = send(tok, data_s, data_to_send);
    let (tok, received_data) = recv(tok, data_r);

    trace_fmt!("send: {}, received: {}, in transaction {}",
      data_to_send, received_data, count + u32:1);

    assert_eq(data_to_send, received_data);

    let do_send = (count == NUMBER_OF_TESTED_TRANSACTIONS - u32:1);
    send_if(tok, terminator, do_send, true);

    count + u32:1
  }
}
