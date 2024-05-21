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

// A specialization of some_caps
import xls.examples.dslx_module.some_caps_streaming;

const REAL_LENGTH: u32 = u32:8;

// Note: Until the advent of proc-scoped channels this example will only work on
// unopt-ir JIT because opt_main will optimize-away the spawned proc since it
// does not look connected. There are ways to work around this using mangled
// symbols as the top but for now we will just use unopt and make a manually
// wired version for opt.
pub proc some_caps_specialized {
    blocker: chan<()> in;

    config(blocker: chan<()> in, string_input: chan<u8[REAL_LENGTH]> in,
           string_output: chan<u8[REAL_LENGTH]> out) {
        spawn some_caps_streaming::some_caps_streaming<REAL_LENGTH>(string_input, string_output);
        // Phony channel to block this proc while the spawned one we actually care about ticks over.
        (blocker,)
    }

    init { () }

    next(state: ()) { recv(join(), blocker); }
}

// A caps proc with manual wiring to ensure that the spawned proc stays alive.
pub proc manual_chan_caps_specialized {
    external_input_wire: chan<u8[REAL_LENGTH]> in;
    real_proc_input_wire: chan<u8[REAL_LENGTH]> out;
    real_proc_output_wire: chan<u8[REAL_LENGTH]> in;
    external_output_wire: chan<u8[REAL_LENGTH]> out;

    config(external_input_wire: chan<u8[REAL_LENGTH]> in,
           external_output_wire: chan<u8[REAL_LENGTH]> out) {
        let (real_input_wire_out, real_input_wire_in) = chan<u8[REAL_LENGTH]>("real_input");
        let (real_output_wire_out, real_output_wire_in) = chan<u8[REAL_LENGTH]>("real_output");
        spawn some_caps_streaming::some_caps_streaming<REAL_LENGTH>(
            real_input_wire_in, real_output_wire_out);
        (external_input_wire, real_input_wire_out, real_output_wire_in, external_output_wire)
    }

    init { () }

    next(state: ()) {
        let (tok, send_to_inner) = recv(join(), external_input_wire);
        let tok = send(tok, real_proc_input_wire, send_to_inner);
        let (tok, recv_from_inner) = recv(tok, real_proc_output_wire);
        send(tok, external_output_wire, recv_from_inner);
    }
}
