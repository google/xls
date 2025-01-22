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
// LFSR proc
// A parametric proc to leverage a customizable LFSR.
// It takes a single parameter: the bit width of the LFSR counter.
// It supports two operations:
// 1. setting the seed and tap mask (both must have the same bit width as the
//    LFSR)
// 2. getting the next value from the LFSR counter
////////////////////////////////////////////////////////////////////////////////

import xls.examples.lfsr;

proc user_module<BIT_WIDTH: u32> {
  output_s: chan<uN[BIT_WIDTH]> out;
  seed_and_mask_r: chan<(uN[BIT_WIDTH], uN[BIT_WIDTH])> in;

  init {
    // state = (seed, tap_mask)
    (uN[BIT_WIDTH]:1, uN[BIT_WIDTH]:1)
  }

  config(output_s: chan<uN[BIT_WIDTH]> out, seed_and_mask_r: chan<(uN[BIT_WIDTH], uN[BIT_WIDTH])> in) {
    (output_s, seed_and_mask_r)
  }

  next(state: (uN[BIT_WIDTH], uN[BIT_WIDTH])) {
    let (tok, new_state, _) = recv_non_blocking(join(), seed_and_mask_r, state);
    send(tok, output_s, new_state.0);
    (lfsr::lfsr(new_state.0, new_state.1), new_state.1)
  }
}

#[test_proc]
proc test {
  value_r: chan<u8> in;
  seed_s: chan<(u8, u8)> out;
  terminator: chan<bool> out;

  init { () }

  config(terminator: chan<bool> out) {
    let (value_s, value_r) = chan<u8>("value");
    let (seed_s, seed_r) = chan<(u8, u8)>("seed");
    spawn user_module<u32:8>(value_s, seed_r);
    (value_r, seed_s, terminator)
  }

  next(state: ()) {
      let (tok, value) = recv(join(), value_r);
      assert_eq(value, u8:1);

      let tok = send(tok, seed_s, (u8:1, u8:0b10111000));
      let (tok, value) = recv(tok, value_r);
      assert_eq(value, u8:1);
      let (tok, value) = recv(tok, value_r);
      assert_eq(value, u8:2);
      let (tok, value) = recv(tok, value_r);
      assert_eq(value, u8:4);
      let (tok, value) = recv(tok, value_r);
      assert_eq(value, u8:8);
      let (tok, value) = recv(tok, value_r);
      assert_eq(value, u8:17);

      let tok = send(tok, seed_s, (u8:237, u8:0b10111000));
      let (tok, value) = recv(tok, value_r);
      assert_eq(value, u8:237);
      let (tok, value) = recv(tok, value_r);
      assert_eq(value, u8:219);

      let tok = send(tok, terminator, true);
  }
}
