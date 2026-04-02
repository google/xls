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

// Simple example proc that receives a trigger signal to start printing
// messages according to the set maximum verbosity value.
const VERBOSITY_LEV_0 = u32:0;
const VERBOSITY_LEV_1 = u32:8;
const VERBOSITY_LEV_2 = u32:16;

proc Vprinter {
    trigger: chan<bool> in;

    config(trigger: chan<bool> in) { (trigger,) }

    init { () }

    next(st: ()) {
        vtrace_fmt!(VERBOSITY_LEV_0, "Verbosity level {:d}", VERBOSITY_LEV_0);
        vtrace_fmt!(VERBOSITY_LEV_1, "Verbosity level {:d}", VERBOSITY_LEV_1);
        vtrace_fmt!(VERBOSITY_LEV_2, "Verbosity level {:d}", VERBOSITY_LEV_2);
        trace_fmt!("Trace verification.");
        let (tok, _) = recv(join(), trigger);
    }
}

#[test_proc]
proc TestVtrace {
    start: chan<bool> out;
    terminator: chan<bool> out;

    config(terminator: chan<bool> out) {
        let (s, r) = chan<bool>("init");
        spawn Vprinter(r);
        (s, terminator)
    }

    init { () }

    next(st: ()) {
        let tok = send(join(), start, true);
        let tok = send(tok, terminator, true);
    }
}
