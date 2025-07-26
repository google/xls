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

#![feature(type_inference_v2)]

import float32;

type F32 = float32::F32;

const F32_2_BITS = float32::flatten(F32 { sign: u1:0, bexp: u8:128, fraction: u23:0 });

fn f32_scalar_matrix_get_element_const<ROW: u32, COL: u32, F32_BITS: u32>() -> F32 {
    if ROW == COL { float32::unflatten(F32_BITS) } else { float32::zero(u1:0) }
}

#[test]
fn f32_scalar_matrix_get_element_const_test() {
    const F32_1_BITS = float32::flatten(float32::one(u1:0));
    assert_eq(f32_scalar_matrix_get_element_const<u32:0, u32:0, F32_1_BITS>(), float32::one(u1:0));
    assert_eq(f32_scalar_matrix_get_element_const<u32:1, u32:0, F32_1_BITS>(), float32::zero(u1:0));
}

// "node" performs the actual work of this systolic array, multiplying an input
// activation by the baked-in weight.
proc node<ROW: u32, COL: u32> {
    from_west: chan<F32> in;
    from_north: chan<F32> in;
    to_east: chan<F32> out;
    to_south: chan<F32> out;

    config(from_west: chan<F32> in, from_north: chan<F32> in, to_east: chan<F32> out,
           to_south: chan<F32> out) {
        (from_west, from_north, to_east, to_south)
    }

    init { () }

    next(state: ()) {
        // TODO: google/xls#2678 - move weights to top level proc.
        let weight = f32_scalar_matrix_get_element_const<ROW, COL, F32_2_BITS>();
        let (activation_tok, activation) = recv(token(), from_west);
        let (partial_sum_tok, partial_sum) = recv(token(), from_north);
        let input_tok = join(activation_tok, partial_sum_tok);

        // Compute our partial product.
        let result = float32::fma(activation, weight, partial_sum);
        trace_fmt!("node[{}, {}]: fma(activation={}, weight={}, partial_sum={}) = {}",
        ROW, COL, activation, weight, partial_sum, result);
        // Send the activation east and the partial product south.
        send(input_tok, to_east, activation);
        send(input_tok, to_south, result);
    }
}

// "matmul" spawns a network of ROWS x COLS nodes.
proc matmul<ROWS: u32, COLS: u32> {
    activations: chan<F32>[ROWS] in;
    results: chan<F32>[COLS] out;
    from_wests: chan<F32>[COLS + u32:1][ROWS] in;
    to_easts: chan<F32>[COLS + u32:1][ROWS] out;
    from_norths: chan<F32>[COLS][ROWS + u32:1] in;
    to_souths: chan<F32>[COLS][ROWS + u32:1] out;

    config(activations: chan<F32>[ROWS] in, results: chan<F32>[COLS] out) {
        // Declare the east-to-west channels.
        let (to_easts, from_wests) = chan<F32>[COLS + u32:1][ROWS]("east_west");
        // Declare the north-to-south channels.
        let (to_souths, from_norths) = chan<F32>[COLS][ROWS + u32:1]("north_south");
        unroll_for! (row, _): (u32, ()) in u32:0..ROWS {
            unroll_for! (col, _): (u32, ()) in u32:0..COLS {
                spawn node<row, col>(
                    from_wests[row][col], from_norths[row][col], to_easts[row][col + u32:1],
                    to_souths[row + u32:1][col]);
            }(());
        }(());
        (activations, results, from_wests, to_easts, from_norths, to_souths)
    }

    init { () }

    next(state: ()) {
        // Send activation to the "left"-end of the array.
        const ACTIVATIONS_COL = u32:0;
        unroll_for! (row, _): (u32, ()) in u32:0..ROWS {
            let (tok, activation) = recv(token(), activations[row]);
            send(tok, to_easts[row][ACTIVATIONS_COL], activation);
        }(());

        // Send zero values to the "top"-end of the array.
        const ZEROES_ROW = u32:0;
        unroll_for! (col, _): (u32, ()) in u32:0..COLS {
            send(token(), to_souths[ZEROES_ROW][col], float32::zero(false));
        }(());

        // Consume and drop values on the "right"-end of the array.
        const DROPS_COL = COLS;
        unroll_for! (row, _): (u32, ()) in u32:0..ROWS {
            recv(token(), from_wests[row][DROPS_COL]);
        }(());

        // Forward result from the "bottom"-end of the array.
        const RESULTS_ROW = ROWS;
        unroll_for! (col, _): (u32, ()) in u32:0..COLS {
            let (tok, result) = recv(token(), from_norths[RESULTS_ROW][col]);
            send(tok, results[col], result);
        }(());
    }
}

// "matmul_4x4" is a top-level proc that instantiates a
proc matmul_4x4 {
    config(activations: chan<F32>[4] in, results: chan<F32>[4] out) {
        spawn matmul<u32:4, u32:4>(activations, results);
    }

    init { () }

    next(state: ()) {  }
}

#[test_proc]
proc matmul_4x4_test {
    activations: chan<F32>[4] out;
    results: chan<F32>[4] in;
    terminator: chan<bool> out;

    config(terminator: chan<bool> out) {
        let (activations_out, activations_in) = chan<F32>[4]("activations");
        let (results_out, results_in) = chan<F32>[4]("results");
        spawn matmul_4x4(activations_in, results_out);
        (activations_out, results_in, terminator)
    }

    init { () }

    next(state: ()) {
        const F32_1 = F32 { sign: u1:0, bexp: u8:127, fraction: u23:0 };
        const F32_2 = F32 { sign: u1:0, bexp: u8:128, fraction: u23:0 };
        const F32_3 = F32 { sign: u1:0, bexp: u8:128, fraction: u1:1 ++ u22:0 };
        const F32_4 = F32 { sign: u1:0, bexp: u8:129, fraction: u23:0 };
        const F32_6 = F32 { sign: u1:0, bexp: u8:129, fraction: u1:1 ++ u22:0 };
        const F32_8 = F32 { sign: u1:0, bexp: u8:130, fraction: u23:0 };

        let tok = send(token(), activations[0], F32_1);
        let tok = send(tok, activations[1], F32_2);
        let tok = send(tok, activations[2], F32_3);
        let tok = send(tok, activations[3], F32_4);

        let (tok, value) = recv(tok, results[0]);
        assert_eq(value, F32_2);
        let (tok, value) = recv(tok, results[1]);
        assert_eq(value, F32_4);
        let (tok, value) = recv(tok, results[2]);
        assert_eq(value, F32_6);
        let (tok, value) = recv(tok, results[3]);
        assert_eq(value, F32_8);

        send(tok, terminator, true);
    }
}
