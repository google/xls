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

#![feature(type_inference_v2)]
#![feature(explicit_state_access)]
#![feature(generics)]

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
    output_s: chan<uN[BIT_WIDTH]> out,
    seed_and_mask_r: chan<(uN[BIT_WIDTH], uN[BIT_WIDTH])> in,
    state: (uN[BIT_WIDTH], uN[BIT_WIDTH]),
}

impl user_module<BIT_WIDTH> {
    // state = (seed, tap_mask)
    fn new
        (output_s: chan<uN[BIT_WIDTH]> out, seed_and_mask_r: chan<(uN[BIT_WIDTH], uN[BIT_WIDTH])> in)
        -> Self {
        user_module { output_s, seed_and_mask_r, state: (uN[BIT_WIDTH]:1, uN[BIT_WIDTH]:1) }
    }

    fn next(self) {
        let state = read(self.state);
        let (tok, new_state, _) = recv_non_blocking(join(), self.seed_and_mask_r, state);
        send(tok, self.output_s, new_state.0);
        write(self.state, (lfsr::lfsr(new_state.0, new_state.1), new_state.1));
    }
}

#[test]
proc test {
    value_r: chan<u8> in,
    seed_s: chan<(u8, u8)> out,
    terminator: chan<bool> out,
}

impl test {
    fn new(terminator: chan<bool> out) -> Self {
        let (value_s, value_r) = chan<u8>("value");
        let (seed_s, seed_r) = chan<(u8, u8)>("seed");
        user_module<u32:8>::new(value_s, seed_r).spawn();
        test { value_r, seed_s, terminator }
    }

    fn next(self) {
        let (tok, value) = recv(join(), self.value_r);
        assert_eq(value, u8:1);

        let tok = send(tok, self.seed_s, (u8:1, u8:0b10111000));
        let (tok, value) = recv(tok, self.value_r);
        assert_eq(value, u8:1);
        let (tok, value) = recv(tok, self.value_r);
        assert_eq(value, u8:2);
        let (tok, value) = recv(tok, self.value_r);
        assert_eq(value, u8:4);
        let (tok, value) = recv(tok, self.value_r);
        assert_eq(value, u8:8);
        let (tok, value) = recv(tok, self.value_r);
        assert_eq(value, u8:17);

        let tok = send(tok, self.seed_s, (u8:237, u8:0b10111000));
        let (tok, value) = recv(tok, self.value_r);
        assert_eq(value, u8:237);
        let (tok, value) = recv(tok, self.value_r);
        assert_eq(value, u8:219);

        let tok = send(tok, self.terminator, true);
    }
}
