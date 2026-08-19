// Copyright 2024 The XLS Authors
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

// This file implements a (not particularly useful) serializer that decomposes a
// 128-bit input into four 32-bit outputs, with guarantees that the next
// decomposition will not start until after the previous one has produced all
// outputs.
//
// TODO(https://github.com/google/xls/issues/1023): Use a single output channel once we can set
// channel strictness from DSLX; `total_order` seems appropriate in this case.
//
// This serves as a demonstration of how we can use tokens as state to coordinate
// across activations.

#![feature(type_inference_v2)]
#![feature(explicit_state_access)]
#![feature(generics)]

pub proc serialized_decomposer {
    ch_in: chan<uN[128]> in,
    result1: chan<u32> out,
    result2: chan<u32> out,
    result3: chan<u32> out,
    result4: chan<u32> out,
    tok: token,
}

impl serialized_decomposer {
    fn new
        (ch_in: chan<uN[128]> in, result1: chan<u32> out, result2: chan<u32> out,
         result3: chan<u32> out, result4: chan<u32> out) -> Self {
        serialized_decomposer { ch_in, result1, result2, result3, result4, tok: token() }
    }

    fn next(self) {
        let tok = read(self.tok);
        let (input_tok, val) = recv(token(), self.ch_in);
        let tok = join(tok, input_tok);
        let tok = send(tok, self.result1, val[0:32]);
        let tok = send(tok, self.result2, val[32:64]);
        let tok = send(tok, self.result3, val[64:96]);
        write(self.tok, send(tok, self.result4, val[96:128]));
    }
}

#[test]
proc serialized_decomposer_smoke_test {
    data_in_s: chan<uN[128]> out,
    data_out_1_r: chan<u32> in,
    data_out_2_r: chan<u32> in,
    data_out_3_r: chan<u32> in,
    data_out_4_r: chan<u32> in,
    terminator: chan<bool> out,
}

impl serialized_decomposer_smoke_test {
    fn new(terminator: chan<bool> out) -> Self {
        let (data_in_s, data_in_r) = chan<uN[128]>("data_in");
        let (data_out_1_s, data_out_1_r) = chan<u32>("data_out_1");
        let (data_out_2_s, data_out_2_r) = chan<u32>("data_out_2");
        let (data_out_3_s, data_out_3_r) = chan<u32>("data_out_3");
        let (data_out_4_s, data_out_4_r) = chan<u32>("data_out_4");
        serialized_decomposer::new(
            data_in_r, data_out_1_s, data_out_2_s, data_out_3_s, data_out_4_s).spawn(
            );
        serialized_decomposer_smoke_test {
            data_in_s,
            data_out_1_r,
            data_out_2_r,
            data_out_3_r,
            data_out_4_r,
            terminator,
        }
    }

    fn next(self) {
        let tok = send(token(), self.data_in_s, uN[128]:18446744073709551616);
        let (tok, v1) = recv(tok, self.data_out_1_r);
        assert_eq(v1, u32:0);
        let (tok, v2) = recv(tok, self.data_out_2_r);
        assert_eq(v2, u32:0);
        let (tok, v3) = recv(tok, self.data_out_3_r);
        assert_eq(v3, u32:1);
        let (tok, v4) = recv(tok, self.data_out_4_r);
        assert_eq(v4, u32:0);

        send(tok, self.terminator, true);
    }
}
