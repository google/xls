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

import float32;

type F32 = float32::F32;

enum Op : u1 {
    LOAD_WEIGHT = 0,
    MULTIPLE_ACCUMULATE = 1,
}

struct NodeCommand { op: Op, weight: F32 }

// "node" implements the  multiplier-accumulator PE (processing element) of the systolic array.
proc node {
    node_commands: chan<NodeCommand> in;
    from_west: chan<F32> in;
    from_north: chan<F32> in;
    to_east: chan<F32> out;
    to_south: chan<F32> out;

    config(node_commands: chan<NodeCommand> in, from_west: chan<F32> in, from_north: chan<F32> in,
           to_east: chan<F32> out, to_south: chan<F32> out) {
        (node_commands, from_west, from_north, to_east, to_south)
    }

    init { float32::qnan() }

    next(weight: F32) {
        // Receive the command from the west channel.
        let (command_tok, command) = recv(token(), node_commands);

        match command.op {
            Op::LOAD_WEIGHT => {
                // Update the weight state.
                command.weight
            },
            Op::MULTIPLE_ACCUMULATE => {
                let (activation_tok, activation) = recv(command_tok, from_west);
                // Receive the partial sum from the north channel.
                let (partial_sum_tok, partial_sum) = recv(command_tok, from_north);
                let input_tok = join(activation_tok, partial_sum_tok);
                // Multiply and accumulate.
                let result = float32::fma(activation, weight, partial_sum);
                // Forward the activation to the east channel.
                send(input_tok, to_east, activation);
                // Send the result to the south channel.
                send(input_tok, to_south, result);
                // Weight stays the same.
                weight
            },
        }
    }
}

// TODO: google/xls#2706 - Replace w/ a parametrics proc.
const ROWS = u32:4;
const COLS = u32:4;

// TODO: google/xls#1306 - Replace w/ union.
struct MatMulCommand { op: Op, weights: F32[COLS][ROWS], activations: F32[ROWS] }

// "matmul" implements the systolic array as a network of ROWS x COLS PE nodes.
proc matmul_4x4 {
    matmul_commands: chan<MatMulCommand> in;
    results: chan<F32[COLS]> out;
    node_commands: chan<NodeCommand>[COLS][ROWS] out;
    from_wests: chan<F32>[COLS + u32:1][ROWS] in;
    to_easts: chan<F32>[COLS + u32:1][ROWS] out;
    from_norths: chan<F32>[COLS][ROWS + u32:1] in;
    to_souths: chan<F32>[COLS][ROWS + u32:1] out;

    config(matmul_commands: chan<MatMulCommand> in, results: chan<F32[COLS]> out) {
        // Define node command channels.
        let (node_commands_out, node_commands_in) = chan<NodeCommand>[COLS][ROWS]("node_commands");
        // Define east-to-west channels.
        let (to_easts, from_wests) = chan<F32>[COLS + u32:1][ROWS]("east_west");
        // Define the north-to-south channels.
        let (to_souths, from_norths) = chan<F32>[COLS][ROWS + u32:1]("north_south");

        unroll_for! (row, _): (u32, ()) in u32:0..ROWS {
            unroll_for! (col, _): (u32, ()) in u32:0..COLS {

                spawn node(
                    node_commands_in[row][col], from_wests[row][col], from_norths[row][col],
                    to_easts[row][col + u32:1], to_souths[row + u32:1][col]);
            }(());
        }(());
        (matmul_commands, results, node_commands_out, from_wests, to_easts, from_norths, to_souths)
    }

    init { () }

    next(state: ()) {
        let next_tok = token();
        let (command_tok, command) = recv(next_tok, matmul_commands);

        match command.op {
            Op::LOAD_WEIGHT => {
                let weights = command.weights;
                unroll_for! (row, _): (u32, ()) in u32:0..ROWS {
                    unroll_for! (col, _): (u32, ()) in u32:0..COLS {
                        // Send a load-weight command to each node.
                        send(
                            command_tok, node_commands[row][col],
                            NodeCommand { op: Op::LOAD_WEIGHT, weight: weights[row][col] });
                    }(())
                }(());
            },
            Op::MULTIPLE_ACCUMULATE => {
                let node_commands_tok = unroll_for! (row, tok): (u32, token) in u32:0..ROWS {
                    unroll_for! (col, tok): (u32, token) in u32:0..COLS {
                        // Send a load-weight command to each node.
                        let node_command_tok = send(
                            command_tok, node_commands[row][col],
                            NodeCommand { op: Op::MULTIPLE_ACCUMULATE, weight: zero!<F32>() });
                        join(node_command_tok, tok)
                    }(tok)
                }(command_tok);

                let activations = command.activations;
                // Send a MAC command to the "left"-end of the array.
                const ACTIVATIONS_COL = u32:0;
                unroll_for! (row, _): (u32, ()) in u32:0..ROWS {
                    send(node_commands_tok, to_easts[row][ACTIVATIONS_COL], activations[row]);
                }(());

                // Send zero values to the "top"-end of the array.
                const ZEROES_ROW = u32:0;
                unroll_for! (col, _): (u32, ()) in u32:0..COLS {
                    send(next_tok, to_souths[ZEROES_ROW][col], float32::zero(false));
                }(());

                // Consume and drop values on the "right"-end of the array.
                const DROPS_COL = COLS;
                unroll_for! (row, _): (u32, ()) in u32:0..ROWS {
                    recv(next_tok, from_wests[row][DROPS_COL]);
                }(());

                // Forward result from the "bottom"-end of the array.
                const RESULTS_ROW = ROWS;
                let (results_tok, results_f32s) =
                    unroll_for! (col, (results_tok, results_f32s)): (u32, (token, F32[COLS])) in
                        u32:0..COLS {
                        let (tok, result) = recv(next_tok, from_norths[RESULTS_ROW][col]);
                        (join(tok, results_tok), update(results_f32s, col, result))
                    }((next_tok, zero!<F32[COLS]>()));
                send(results_tok, results, results_f32s);
            },
        }
    }
}

#[test_proc]
proc matmul_4x4_test {
    commands: chan<MatMulCommand> out;
    results: chan<F32[COLS]> in;
    terminator: chan<bool> out;

    config(terminator: chan<bool> out) {
        let (commands_out, commands_in) = chan<MatMulCommand>("commands");
        let (results_out, results_in) = chan<F32[COLS]>("results");

        spawn matmul_4x4(commands_in, results_out);
        (commands_out, results_in, terminator)
    }

    init { () }

    next(state: ()) {
        const F32_0 = float32::zero(u1:0);
        const F32_1 = float32::one(u1:0);
        const F32_2 = F32 { sign: u1:0, bexp: u8:128, fraction: u23:0 };
        const F32_3 = F32 { sign: u1:0, bexp: u8:128, fraction: u1:1 ++ u22:0 };
        const F32_4 = F32 { sign: u1:0, bexp: u8:129, fraction: u23:0 };
        const F32_6 = F32 { sign: u1:0, bexp: u8:129, fraction: u1:1 ++ u22:0 };
        const F32_8 = F32 { sign: u1:0, bexp: u8:130, fraction: u23:0 };
        const SCALAR_MATRIX_4X4_F32_2 = F32[COLS][ROWS]:[
            [F32_2, F32_0, F32_0, F32_0], [F32_0, F32_2, F32_0, F32_0],
            [F32_0, F32_0, F32_2, F32_0], [F32_0, F32_0, F32_0, F32_2],
        ];

        // Load the weights matrix.
        let tok = send(
            token(), commands,
            MatMulCommand {
                op: Op::LOAD_WEIGHT,
                weights: SCALAR_MATRIX_4X4_F32_2,
                activations: zero!<F32[ROWS]>(),
            });

        // Send the activations.
        let tok = send(
            tok, commands,
            MatMulCommand {
                op: Op::MULTIPLE_ACCUMULATE,
                weights: zero!<F32[COLS][ROWS]>(),
                activations: F32[ROWS]:[F32_1, F32_2, F32_3, F32_4],
            });

        // Receive and check the results.
        let (tok, results_f32s) = recv(tok, results);
        assert_eq(results_f32s, [F32_2, F32_4, F32_6, F32_8]);

        send(tok, terminator, true);
    }
}
