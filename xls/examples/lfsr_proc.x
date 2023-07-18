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
// See the License for the specific language governing permissions and
// limitations under the License.

////////////////////////////////////////////////////////////////////////////////
// 8-bit LFSR proc
// 
// Two operations are supported:
// 1. setting the seed
// 2. getting the next value from the LFSR counter
////////////////////////////////////////////////////////////////////////////////

import xls.examples.lfsr

proc user_module {
  output_s: chan<u8> out;
  seed_r: chan<u8> in;

  init {
    (u8:1)
  }

  config(output_s: chan<u8> out, seed_r: chan<u8> in) {
    (output_s, seed_r)
  }

  next(tok: token, state: u8) {
    let (tok, seed, seed_valid) = recv_non_blocking(tok, seed_r, state);
    let state = if(seed_valid) {
        (seed)
    } else {
        lfsr::lfsr(state, u8:0b10111000)
    };
    send(tok, output_s, state);
    (state)
  }
}

#[test_proc]
proc test {
  value_r: chan<u8> in;
  seed_s: chan<u8> out;
  terminator: chan<bool> out;

  init { () }

  config(terminator: chan<bool> out) {
    let (value_s, value_r) = chan<u8>;
    let (seed_s, seed_r) = chan<u8>;
    spawn user_module(value_s, seed_r);
    (value_r, seed_s, terminator)
  }

  next(tok: token, state: ()) {
      let (tok, value) = recv(tok, value_r);
      assert_eq(value, u8:2);
      let (tok, value) = recv(tok, value_r);
      assert_eq(value, u8:4);
      let (tok, value) = recv(tok, value_r);
      assert_eq(value, u8:8);
      let (tok, value) = recv(tok, value_r);
      assert_eq(value, u8:17);

      let tok = send(tok, seed_s, u8:237);
      let (tok, value) = recv(tok, value_r);
      assert_eq(value, u8:237);
      let (tok, value) = recv(tok, value_r);
      assert_eq(value, u8:219);

      let tok = send(tok, terminator, true);
      ()
  }
}

