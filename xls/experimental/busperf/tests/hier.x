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

// Minimal hierarchical example exercising busperf setup generation for
// internal, per-instance ready/valid channels between a parent and its
// spawned children.

#![feature(type_inference_v2)]

proc Doubler {
    in_ch: chan<u32> in;
    out_ch: chan<u32> out;

    config(in_ch: chan<u32> in, out_ch: chan<u32> out) { (in_ch, out_ch) }

    init { () }

    next(state: ()) {
        let (tok, x) = recv(join(), in_ch);
        let tok = send(tok, out_ch, x + x);
    }
}

proc Parent {
    config(in0: chan<u32> in, out0: chan<u32> out,
           in1: chan<u32> in, out1: chan<u32> out) {
        spawn Doubler(in0, out0);
        spawn Doubler(in1, out1);
        ()
    }

    init { () }

    next(state: ()) { () }
}
