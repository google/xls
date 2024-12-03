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

// DSLX implementation of a 4x4 systolic array, appropriate for part of a
// matrix multiplier.
import float32;

type F32 = float32::F32;

// "node" performs the actual work of this systolic array, multiplying an input
// activation by the baked-in weight.
proc node {
    from_west: chan<F32> in;
    from_north: chan<F32> in;
    to_east: chan<F32> out;
    to_south: chan<F32> out;
    weight: F32;

    config(from_west: chan<F32> in, from_north: chan<F32> in, to_east: chan<F32> out,
           to_south: chan<F32> out, weight: F32) {
        (from_west, from_north, to_east, to_south, weight)
    }

    init { () }

    next(state: ()) {
        let (tok, activation) = recv(join(), from_west);
        let (tok, partial_sum) = recv(tok, from_north);

        // Compute our partial product.
        let product = float32::mul(activation, weight);

        // Send the activation east and the partial product south.
        let tok = send(tok, to_east, activation);
        let tok = send(tok, to_south, product);
    }
}

proc matmul<ROWS: u32, COLS: u32> {
    activations_in: chan<F32>[ROWS] in;
    results_out: chan<F32>[COLS] out;
    west_inputs: chan<F32>[COLS + u32:1][ROWS] in;
    east_outputs: chan<F32>[COLS + u32:1][ROWS] out;
    north_inputs: chan<F32>[COLS][ROWS + u32:1] in;
    south_outputs: chan<F32>[COLS][ROWS + u32:1] out;

    config(activations_in: chan<F32>[ROWS] in, results_out: chan<F32>[COLS] out) {
        // Declare the east-to-west channels.
        let (east_outputs, west_inputs) = chan<F32>[COLS + u32:1][ROWS]("east_west");
        // Declare the north-to-south channels.
        let (south_outputs, north_inputs) = chan<F32>[COLS][ROWS + u32:1]("north_south");
        unroll_for! (row, _): (u32, ()) in u32:0..ROWS {
            unroll_for! (col, _): (u32, ()) in u32:0..COLS {
                let weight = F32 {
                    sign: false,
                    bexp: if col == row { u8:128 } else { u8:0 },
                    fraction: u23:0,
                };
                spawn node(
                    west_inputs[row][col], north_inputs[row][col], east_outputs[row][col + u32:1],
                    south_outputs[row + u32:1][col], weight);
            }(());
        }(());
        (activations_in, results_out, west_inputs, east_outputs, north_inputs, south_outputs)
    }

    init { () }

    next(state: ()) {
        // Send activation to the "left"-end of the array.
        let activations_col = u32:0;
        unroll_for! (row, _): (u32, ()) in u32:0..ROWS {
            let (tok, activation) = recv(join(), activations_in[row]);
            send(tok, east_outputs[row][activations_col], activation);
        }(());

        // Send zero values to the "top"-end of the array.
        let zeroes_row = u32:0;
        unroll_for! (col, _): (u32, ()) in u32:0..COLS {
            send(join(), south_outputs[zeroes_row][col], float32::zero(false));
        }(());

        // Consume and drop values on the "right"-end of the array.
        // TODO - google/xls#1750: remove unroll_for! workaround.
        unroll_for! (drops_col, _): (u32, ()) in COLS..COLS + u32:1 {
            unroll_for! (row, _): (u32, ()) in u32:0..ROWS {
                recv(join(), west_inputs[row][drops_col]);
            }(());
        }(());

        // Forward result from the "bottom"-end of the array.
        // TODO - google/xls#1750: remove unroll_for! workaround.
        unroll_for! (results_row, _): (u32, ()) in ROWS..ROWS + u32:1 {
            unroll_for! (col, _): (u32, ()) in u32:0..COLS {
                let (tok, result) = recv(join(), north_inputs[results_row][col]);
                send(tok, results_out[col], result);
            }(());
        }(());
    }
}

proc matmul_4x4 {
    config(activations_in: chan<F32>[4] in, results_out: chan<F32>[4] out) {
        spawn matmul<u32:4, u32:4>(activations_in, results_out);
    }

    init { () }

    next(state: ()) {  }
}

#[test_proc]
proc test_proc {
    activations_out: chan<F32>[4] out;
    results_in: chan<F32>[4] in;
    terminator: chan<bool> out;

    config(terminator: chan<bool> out) {
        let (activations_out, activations_in) = chan<F32>[4]("activations");
        let (results_out, results_in) = chan<F32>[4]("results");
        spawn matmul_4x4(activations_in, results_out);
        (activations_out, results_in, terminator)
    }

    init { () }

    next(state: ()) {
        let f32_0 = float32::zero(false);
        let f32_2 = F32 { sign: false, bexp: u8:128, fraction: u23:0 };
        let f32_4 = F32 { sign: false, bexp: u8:129, fraction: u23:0 };

        // Send the desired inputs.
        let activations_tok = unroll_for! (i, tok): (u32, token) in u32:0..u32:4 {
            let activation_tok = send(token(), activations_out[i], f32_2);
            join(activation_tok, tok)
        }(token());

        let (results_0_tok, value) = recv(activations_tok, results_in[0]);
        assert_eq(value, f32_0);
        let (results_1_tok, value) = recv(activations_tok, results_in[1]);
        assert_eq(value, f32_0);
        let (results_2_tok, value) = recv(activations_tok, results_in[2]);
        assert_eq(value, f32_0);
        let (results_3_tok, value) = recv(activations_tok, results_in[3]);
        assert_eq(value, f32_4);

        let results_tok = join(results_0_tok, results_1_tok, results_2_tok, results_3_tok);

        send(results_tok, terminator, true);
    }
}
