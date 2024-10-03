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

// Proc used to test eval_proc_main.

pub proc test_proc {
    in_ch: chan<u64> in;
    in_ch_2: chan<()> in;
    out_ch: chan<u64> out;
    out_ch_2: chan<()> out;

    config(in_ch: chan<u64> in, in_ch_2: chan<()> in, out_ch: chan<u64> out, out_ch_2: chan<()> out) {
        (in_ch, in_ch_2, out_ch, out_ch_2)
    }

    init { u64:1 }

    next(st: u64) {
        let (tkn, r1) = recv(join(), in_ch);
        let (tkn, _) = recv(tkn, in_ch_2);
        let sum = r1 + st;
        let tkn = send(tkn, out_ch, sum);
        let tkn = send(tkn, out_ch_2, ());
        st + u64:10
    }
}
