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

const VERBOSITY_LEV_0 = u32:0;
const VERBOSITY_LEV_1 = u32:2;
const VERBOSITY_LEV_2 = u32:4;
const VERBOSITY_LEV_3 = u32:8;
const VERBOSITY_LEV_4 = u32:16;
const VERBOSITY_LEV_5 = u32:32;
const VERBOSITY_LEV_6 = u32:64;

fn vtrace_fmt_example(a: u32, b: u32) -> u32 {
    vtrace_fmt!(VERBOSITY_LEV_0, "Verbosity level {:d}", VERBOSITY_LEV_0);
    vtrace_fmt!(VERBOSITY_LEV_1, "Verbosity level {:d}", VERBOSITY_LEV_1);
    vtrace_fmt!(VERBOSITY_LEV_2, "Verbosity level {:d}", VERBOSITY_LEV_2);
    vtrace_fmt!(VERBOSITY_LEV_3, "Verbosity level {:d}", VERBOSITY_LEV_3);
    vtrace_fmt!(VERBOSITY_LEV_4, "Verbosity level {:d}", VERBOSITY_LEV_4);
    vtrace_fmt!(VERBOSITY_LEV_5, "Verbosity level {:d}", VERBOSITY_LEV_5);
    vtrace_fmt!(VERBOSITY_LEV_6, "Verbosity level {:d}", VERBOSITY_LEV_6);
    trace_fmt!("Trace verification.");
    a + b
}

#[test]
fn example_test() {
    assert_eq(vtrace_fmt_example(u32:1, u32:2), u32:3);
}
