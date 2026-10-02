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

#![feature(explicit_state_access)]
#![feature(generics)]

// This demonstrates how to emulate a pipeline in DSLX as well as
// how stage can be modeled.
//
//   next_x =  (x + input_r) * input_r
//
// Pictorally the pipeline looks like the following:
//
//         STAGE 0                 STAGE 1            STAGE 2
//
//                           []            []
//   [ input_r >------+------[]---\        []
//                    |      []    [MUL]---[]--+--[ output_w >
//       +---------[ ADD ]---[]---/        []  |
//       |                   []            []  |
//   [ state x ]             []            []  |
//       ^                   []            []  |
//       |                                     |
//       +-------------------------------------+
//                 ii=3 next state
//

proc state_proc {
    // State read channel
    state_value_w: chan<u32> out,
    next_state_r: chan<u32> in,
    state: (u32, bool),
}

impl state_proc {
    fn new(state_value_w: chan<u32> out, next_state_r: chan<u32> in) -> Self {
        Self { state_value_w, next_state_r, state: (u32:0, bool:true) }
    }

    fn next(self) {
        let state = read(self.state);
        let x = state.0;
        let full = state.1;

        send_if(join(), self.state_value_w, full, x);

        // Note that this doesn't fully encode the desired behavior down-to
        // BlockIR as this channel needs to be marked as not having backpressure
        // to prevent a combinational loop.
        let (_, next_state_data, next_state_data_valid) =
            recv_non_blocking(join(), self.next_state_r, u32:0);

        let next_x = if next_state_data_valid { next_state_data } else { x };
        let next_full = next_state_data_valid;
        write(self.state, (next_x, next_full));
    }
}

proc stage_0 {
    // External input channel.
    input_r: chan<u32> in,
    // Datapath channels.
    data_0_w: chan<u32> out,
    data_1_w: chan<u32> out,
    // State channels.
    state_x_r: chan<u32> in,
}

impl stage_0 {
    fn new
        (input_r: chan<u32> in, data_0_w: chan<u32> out, data_1_w: chan<u32> out,
         state_x_r: chan<u32> in) -> Self {
        Self { input_r, data_0_w, data_1_w, state_x_r }
    }

    fn next(self) {
        let (tok, input_r_data) = recv(join(), self.input_r);
        let (tok, state_x_data) = recv(tok, self.state_x_r);

        let data_1 = input_r_data + state_x_data;

        send(tok, self.data_0_w, input_r_data);
        send(tok, self.data_1_w, data_1);
    }
}

proc stage_1 {
    // Datapath channels.
    data_0_r: chan<u32> in,
    data_1_r: chan<u32> in,
    data_2_w: chan<u32> out,
}

impl stage_1 {
    fn new(data_0_r: chan<u32> in, data_1_r: chan<u32> in, data_2_w: chan<u32> out) -> Self {
        Self { data_0_r, data_1_r, data_2_w }
    }

    fn next(self) {
        let (tok, data_0_data) = recv(join(), self.data_0_r);
        let (tok, data_1_data) = recv(tok, self.data_1_r);

        let data_2_data = data_0_data * data_1_data;

        send(tok, self.data_2_w, data_2_data);
    }
}

proc stage_2 {
    // External output channel.
    output_w: chan<u32> out,
    // Datapath channels.
    data_2_r: chan<u32> in,
    // Next state channel.
    next_state_x_w: chan<u32> out,
}

impl stage_2 {
    fn new(output_w: chan<u32> out, data_2_r: chan<u32> in, next_state_x_w: chan<u32> out) -> Self {
        Self { output_w, data_2_r, next_state_x_w }
    }

    fn next(self) {
        let (tok, data_2_data) = recv(join(), self.data_2_r);

        // This send should be marked as no-backpressure, always ready.
        send(tok, self.next_state_x_w, data_2_data);
        send(tok, self.output_w, data_2_data);
    }
}

pub proc xls_pipeline {}

impl xls_pipeline {
    fn new(input_r: chan<u32> in, output_w: chan<u32> out) -> Self {
        let (data_0_out, data_0_in) = chan<u32>("data_0");
        let (data_1_out, data_1_in) = chan<u32>("data_1");
        let (data_2_out, data_2_in) = chan<u32>("data_2");

        let (state_x_out, state_x_in) = chan<u32>("state_x");
        let (next_state_x_out, next_state_x_in) = chan<u32>("next_state_x");

        stage_0::new(input_r, data_0_out, data_1_out, state_x_in).spawn();
        stage_1::new(data_0_in, data_1_in, data_2_out).spawn();
        stage_2::new(output_w, data_2_in, next_state_x_out).spawn();

        state_proc::new(state_x_out, next_state_x_in).spawn();
        Self {}
    }
}

#[test]
proc xls_pipeline_test {
    input_w: chan<u32> out,
    output_r: chan<u32> in,
    terminator: chan<bool> out,
}

impl xls_pipeline_test {
    fn new(terminator: chan<bool> out) -> Self {
        let (input_w, input_r) = chan<u32>("input_ch");
        let (output_w, output_r) = chan<u32>("output_ch");

        xls_pipeline::new(input_r, output_w).spawn();
        Self { input_w, output_r, terminator }
    }

    fn next(self) {
        let tok = join();

        let tok = send(tok, self.input_w, u32:1);
        let tok = send(tok, self.input_w, u32:2);
        let tok = send(tok, self.input_w, u32:3);

        let (tok, val_0) = recv(tok, self.output_r);
        let (tok, val_1) = recv(tok, self.output_r);
        let (tok, val_2) = recv(tok, self.output_r);

        assert_eq(val_0, u32:1);
        assert_eq(val_1, u32:6);
        assert_eq(val_2, u32:27);

        send(tok, self.terminator, true);
    }
}
