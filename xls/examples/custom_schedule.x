#![feature(type_inference_v2)]
#![feature(explicit_state_access)]
#![feature(generics)]

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

// This is just an example proc to display how io_constraints can force particular schedules if
// needed.
pub proc Accumulator {
    data_in: chan<uN[8]> in,
    activate: chan<uN[1]> in,
    data_out: chan<uN[256]> out,
    old_state: chan<uN[256]> out,
    acc: uN[256],
}

impl Accumulator {
    fn new
        (data_in: chan<uN[8]> in, activate: chan<uN[1]> in, data_out: chan<uN[256]> out,
         old_state: chan<uN[256]> out) -> Self {
        Accumulator { data_in, activate, data_out, old_state, acc: uN[256]:0 }
    }

    fn next(self) {
        let acc = read(self.acc);
        let (tok, next_recv) = recv(join(), self.data_in);
        let tok = send(tok, self.old_state, acc);
        let to_send = acc + (next_recv as uN[256]);
        let tok = send(tok, self.data_out, to_send);
        let (tok, act) = recv(tok, self.activate);
        write(self.acc, if act { to_send } else { acc });
    }
}
