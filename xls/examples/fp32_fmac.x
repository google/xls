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

#![feature(type_inference_v2)]
#![feature(explicit_state_access)]
#![feature(generics)]

import float32;
import xls.examples.apfloat_fmac;

type F32 = float32::F32;

const F32_ZERO = float32::zero(false);
const F32_ONE = float32::one(false);

proc fp32_fmac {}

impl fp32_fmac {
    fn new
        (input_a: chan<F32> in, input_b: chan<F32> in, reset: chan<bool> in, output: chan<F32> out)
        -> Self {
        apfloat_fmac::fmac<u32:8, u32:23>::new(input_a, input_b, reset, output).spawn();
        fp32_fmac {}
    }

    // Nothing to do here - the spawned fmac does all the lifting.
    fn next(self) {}
}

#[test]
proc smoke_test {
    input_a_s: chan<F32> out,
    input_b_s: chan<F32> out,
    reset_s: chan<bool> out,
    output_r: chan<F32> in,
    terminator: chan<bool> out,
}

impl smoke_test {
    fn new(terminator: chan<bool> out) -> Self {
        let (input_a_s, input_a_r) = chan<F32>("input_a");
        let (input_b_s, input_b_r) = chan<F32>("input_b");
        let (reset_s, reset_r) = chan<bool>("reset");
        let (output_s, output_r) = chan<F32>("output");
        fp32_fmac::new(input_a_r, input_b_r, reset_r, output_s).spawn();
        smoke_test { input_a_s, input_b_s, reset_s, output_r, terminator }
    }

    fn next(self) {
        let tok = send(join(), self.input_a_s, F32_ZERO);
        let tok = send(tok, self.input_b_s, F32_ZERO);
        let tok = send(tok, self.reset_s, false);
        let (tok, result) = recv(tok, self.output_r);
        assert_eq(result, F32_ZERO);

        let tok = send(tok, self.input_a_s, F32_ONE);
        let tok = send(tok, self.input_b_s, F32_ZERO);
        let tok = send(tok, self.reset_s, false);
        let (tok, result) = recv(tok, self.output_r);
        assert_eq(result, F32_ZERO);

        let tok = send(tok, self.terminator, true);
    }
}
