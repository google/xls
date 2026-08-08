// Copyright 2026 The XLS Authors
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

proc test_proc {
    out_ch: chan<u32> out;

    config(out_ch: chan<u32> out) { (out_ch,) }

    init { u32:0 }

    next(st: u32) {
        let const_zero = st & u32:0;
        let trailing_zero = st << u32:2;
        let add1 = st + u32:1 + const_zero + (trailing_zero & u32:0);
        assert!(st == st, "test assert");
        let tok = send(join(), out_ch, add1);
        add1
    }
}
