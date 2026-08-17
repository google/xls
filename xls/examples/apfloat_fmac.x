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

// DSLX implementation of a fused multiply-accumulate module, calculating
// `acc = a * b + acc`
// This is, effectively, an FMA unit that stores its result between ticks.
import apfloat;

type APFloat = apfloat::APFloat;

pub proc fmac<EXP_SZ: u32, SFD_SZ: u32> {
    input_a: chan<APFloat<EXP_SZ, SFD_SZ>> in,
    input_b: chan<APFloat<EXP_SZ, SFD_SZ>> in,
    reset: chan<bool> in,
    output: chan<APFloat<EXP_SZ, SFD_SZ>> out,
    acc: APFloat<EXP_SZ, SFD_SZ>,
}

impl fmac<EXP_SZ, SFD_SZ> {
    fn new
        (input_a: chan<APFloat<EXP_SZ, SFD_SZ>> in, input_b: chan<APFloat<EXP_SZ, SFD_SZ>> in,
         reset: chan<bool> in, output: chan<APFloat<EXP_SZ, SFD_SZ>> out) -> Self {
        fmac { input_a, input_b, reset, output, acc: apfloat::zero<EXP_SZ, SFD_SZ>(false) }
    }

    fn next(self) {
        let acc = read(self.acc);
        let (tok0, a) = recv(join(), self.input_a);
        let (tok1, b) = recv(join(), self.input_b);
        let (tok2, do_reset) = recv(join(), self.reset);

        let acc = apfloat::fma<EXP_SZ, SFD_SZ>(a, b, acc);
        let zero = apfloat::zero<EXP_SZ, SFD_SZ>(false);
        let acc = if do_reset { zero } else { acc };

        let tok3 = join(tok0, tok1, tok2);
        send(tok3, self.output, acc);
        write(self.acc, acc);
    }
}
