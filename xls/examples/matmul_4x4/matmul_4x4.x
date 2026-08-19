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

// DSLX implementation of a 4x4 systolic array.

#![feature(type_inference_v2)]
#![feature(explicit_state_access)]
#![feature(generics)]

import float32;

type F32 = float32::F32;

struct NodeConfig { is_configured: bool, weight: F32 }

// "node" implements the  multiplier-accumulator PE (processing element) of the systolic array.
proc node<ROW: u32, COL: u32> {
    load_weight: chan<F32> in,
    from_west: chan<F32> in,
    from_north: chan<F32> in,
    to_east: chan<F32> out,
    to_south: chan<F32> out,
    state: NodeConfig,
}

impl node<ROW, COL> {
    fn new
        (load_weight: chan<F32> in, from_west: chan<F32> in, from_north: chan<F32> in,
         to_east: chan<F32> out, to_south: chan<F32> out) -> Self {
        node {
            load_weight,
            from_west,
            from_north,
            to_east,
            to_south,
            state: NodeConfig { is_configured: false, weight: float32::qnan() },
        }
    }

    fn next(self) {
        let state = read(self.state);
        // If weight is undefined (NaN), receive it from the load weight channel.
        let (weight_tok, weight) =
            recv_if(token(), self.load_weight, !state.is_configured, state.weight);
        // Receive the activation from the west channel.
        let (activation_tok, activation) = recv(weight_tok, self.from_west);
        // Receive the partial sum from the north channel.
        let (partial_sum_tok, partial_sum) = recv(weight_tok, self.from_north);
        let input_tok = join(activation_tok, partial_sum_tok);

        // Multiply and accumulate.
        let result = float32::fma(activation, weight, partial_sum);
        trace_fmt!("node[{}, {}]: fma(activation={}, weight={}, partial_sum={}) = {}",
        ROW, COL, activation, weight, partial_sum, result);
        // Forward the activation to the east channel.
        send(input_tok, self.to_east, activation);
        // Send the result to the south channel.
        send(input_tok, self.to_south, result);
        write(self.state, NodeConfig { is_configured: true, weight });
    }
}

// "matmul" implements the systolic array as a network of ROWS x COLS PE nodes.
proc matmul<ROWS: u32, COLS: u32> {
    weights: chan<F32[COLS][ROWS]> in,
    node_weights: chan<F32>[COLS][ROWS] out,
    activations: chan<F32>[ROWS] in,
    results: chan<F32>[COLS] out,
    from_wests: chan<F32>[COLS + u32:1][ROWS] in,
    to_easts: chan<F32>[COLS + u32:1][ROWS] out,
    from_norths: chan<F32>[COLS][ROWS + u32:1] in,
    to_souths: chan<F32>[COLS][ROWS + u32:1] out,
    weight_loaded: bool,
}

impl matmul<ROWS, COLS> {
    fn new
        (weights: chan<F32[COLS][ROWS]> in, activations: chan<F32>[ROWS] in,
         results: chan<F32>[COLS] out) -> Self {
        let (store_node_weights, load_node_weights) = chan<F32>[COLS][ROWS]("node_weights");
        // Declare the east-to-west channels.
        let (to_easts, from_wests) = chan<F32>[COLS + u32:1][ROWS]("east_west");
        // Declare the north-to-south channels.
        let (to_souths, from_norths) = chan<F32>[COLS][ROWS + u32:1]("north_south");
        unroll_for! (row, _): (u32, ()) in u32:0..ROWS {
            unroll_for! (col, _): (u32, ()) in u32:0..COLS {
                node<row, col>::new(
                    load_node_weights[row][col], from_wests[row][col], from_norths[row][col],
                    to_easts[row][col + u32:1], to_souths[row + u32:1][col]).spawn(
                    );
            }(());
        }(());
        matmul {
            weights,
            node_weights: store_node_weights,
            activations,
            results,
            from_wests,
            to_easts,
            from_norths,
            to_souths,
            weight_loaded: false,
        }
    }

    fn next(self) {
        let weight_loaded = read(self.weight_loaded);

        // Receive the weights from the load weight channel.
        // Note: this is a one-off operation that happens on the first activation.
        let tok = if !weight_loaded {
            let (recv_weights_tok, weights_matrix) = recv(token(), self.weights);
            unroll_for! (row, tok): (u32, token) in u32:0..ROWS {
                unroll_for! (col, tok): (u32, token) in u32:0..COLS {
                    let weight_tok = send(
                        recv_weights_tok, self.node_weights[row][col], weights_matrix[row][col]);
                    join(weight_tok, tok)
                }(tok)
            }(token())
        } else {
            token()
        };

        // Send activation to the "left"-end of the array.
        const ACTIVATIONS_COL = u32:0;
        unroll_for! (row, _): (u32, ()) in u32:0..ROWS {
            let (tok, activation) = recv(tok, self.activations[row]);
            send(tok, self.to_easts[row][ACTIVATIONS_COL], activation);
        }(());

        // Send zero values to the "top"-end of the array.
        const ZEROES_ROW = u32:0;
        unroll_for! (col, _): (u32, ()) in u32:0..COLS {
            send(token(), self.to_souths[ZEROES_ROW][col], float32::zero(false));
        }(());

        // Consume and drop values on the "right"-end of the array.
        const DROPS_COL = COLS;
        unroll_for! (row, _): (u32, ()) in u32:0..ROWS {
            recv(token(), self.from_wests[row][DROPS_COL]);
        }(());

        // Forward result from the "bottom"-end of the array.
        const RESULTS_ROW = ROWS;
        unroll_for! (col, _): (u32, ()) in u32:0..COLS {
            let (tok, result) = recv(token(), self.from_norths[RESULTS_ROW][col]);
            send(tok, self.results[col], result);
        }(());
        write(self.weight_loaded, true);
    }
}

// "matmul_4x4" is top-level proc that spawns the 4x4 matmul proc.
proc matmul_4x4 {}

impl matmul_4x4 {
    fn new
        (weights: chan<F32[4][4]> in, activations: chan<F32>[4] in, results: chan<F32>[4] out)
        -> Self {
        matmul<u32:4, u32:4>::new(weights, activations, results).spawn();
        matmul_4x4 {}
    }
}

#[test]
proc matmul_4x4_test {
    weights: chan<F32[4][4]> out,
    activations: chan<F32>[4] out,
    results: chan<F32>[4] in,
    terminator: chan<bool> out,
}

impl matmul_4x4_test {
    fn new(terminator: chan<bool> out) -> Self {
        let (weights_out, weights_in) = chan<F32[4][4]>("weights");
        let (activations_out, activations_in) = chan<F32>[4]("activations");
        let (results_out, results_in) = chan<F32>[4]("results");
        matmul_4x4::new(weights_in, activations_in, results_out).spawn();
        matmul_4x4_test {
            weights: weights_out,
            activations: activations_out,
            results: results_in,
            terminator,
        }
    }

    fn next(self) {
        const F32_0 = float32::zero(u1:0);
        const F32_1 = float32::one(u1:0);
        const F32_2 = F32 { sign: u1:0, bexp: u8:128, fraction: u23:0 };
        const F32_3 = F32 { sign: u1:0, bexp: u8:128, fraction: u1:1 ++ u22:0 };
        const F32_4 = F32 { sign: u1:0, bexp: u8:129, fraction: u23:0 };
        const F32_6 = F32 { sign: u1:0, bexp: u8:129, fraction: u1:1 ++ u22:0 };
        const F32_8 = F32 { sign: u1:0, bexp: u8:130, fraction: u23:0 };
        const SCALAR_MATRIX_4X4_F32_2 = F32[4][4]:[
            [F32_2, F32_0, F32_0, F32_0], [F32_0, F32_2, F32_0, F32_0],
            [F32_0, F32_0, F32_2, F32_0], [F32_0, F32_0, F32_0, F32_2],
        ];

        let tok = send(token(), self.weights, SCALAR_MATRIX_4X4_F32_2);

        let tok = send(tok, self.activations[0], F32_1);
        let tok = send(tok, self.activations[1], F32_2);
        let tok = send(tok, self.activations[2], F32_3);
        let tok = send(tok, self.activations[3], F32_4);

        let (tok, value) = recv(tok, self.results[0]);
        assert_eq(value, F32_2);
        let (tok, value) = recv(tok, self.results[1]);
        assert_eq(value, F32_4);
        let (tok, value) = recv(tok, self.results[2]);
        assert_eq(value, F32_6);
        let (tok, value) = recv(tok, self.results[3]);
        assert_eq(value, F32_8);

        send(tok, self.terminator, true);
    }
}
