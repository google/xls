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

// os_matmul.x -- output-stationary systolic matmul array for ML (float32).
//
// 2D output-stationary systolic array: C = A * B (matrix-matrix, f32).
// A is M x K  (a BATCH of M activation vectors, each length K)
// B is K x N  (the weight matrix: B[k][j] weights input feature k -> output j)
// C is M x N  (C[i][j] = sum_k A[i][k] * B[k][j])
// and the M*N outputs stay put until the contraction finishes, then drain in parallel.
//
// Parametric over M (batch), K (contraction), N (outputs).  Grid = M*N PEs, each a
// single multiplier; the counter time-multiplexes the K contraction steps.
//
// Everything adheres to the IEEE-754 float32 (`float32::F32`) standard.

import float32;

type F32 = float32::F32;

// bit-pattern to F32 helpers, for scalars, vectors and matrices
fn f32c(raw: u32) -> F32 { float32::unflatten(raw) }

fn fv<L: u32>(raw: u32[L]) -> F32[L] {
    for (i, v): (u32, F32[L]) in u32:0..L {
        update(v, i, f32c(raw[i]))
    }(zero!<F32[L]>())
}

fn fv2<C: u32, R: u32>(raw: u32[C][R]) -> F32[C][R] {
    for (i, m): (u32, F32[C][R]) in u32:0..R {
        update(m, i, fv(raw[i]))
    }(zero!<F32[C][R]>())
}

// ================ OUTPUT-STATIONARY SYSTOLIC MATMUL (FP32) =============
// output-stationary PE: multiply-accumulates the K activation/weight pairs
// that stream past it, emits the finished value, and sends both operands onward
proc os_pe<K: u32> {
    // activation from the west
    a_in: chan<F32> in;
    // weight from the north
    b_in: chan<F32> in;
    // activation onward to the east
    a_out: chan<F32> out;
    // weight onward to the south
    b_out: chan<F32> out;
    // finished C[i][j] leaves here on the last step
    y_out: chan<F32> out;

    config(a_in: chan<F32> in, b_in: chan<F32> in, a_out: chan<F32> out, b_out: chan<F32> out,
           y_out: chan<F32> out) {
        (a_in, b_in, a_out, b_out, y_out)
    }

    init { (u32:0, float32::zero(false)) }

    next(st: (u32, F32)) {
        let tok = token();

        // continue this PE's running accumulation (k counts the contraction steps)
        let (k, acc) = st;

        // receive an activation from the west and a weight from the north, then
        // multiply accumulate into the running sum
        let (tok, a) = recv(tok, a_in);
        let (tok, b) = recv(tok, b_in);
        let acc = float32::fma(a, b, acc);

        // march both operands on to the neighbouring PEs
        let tok = send(tok, a_out, a);
        let tok = send(tok, b_out, b);

        // emit the finished C[i][j] on the final contraction step
        let last = k == K - u32:1;
        let tok = send_if(tok, y_out, last, acc);

        // reset the counter and accumulator once the output has drained
        (if last { u32:0 } else { k + u32:1 }, if last { float32::zero(false) } else { acc })
    }
}

// Drains a spent operand off an edge so it never back-pressures.
proc os_drain {
    d_in: chan<F32> in;

    config(d_in: chan<F32> in) { (d_in,) }

    init { () }

    next(st: ()) { let (tok, _) = recv(join(), d_in); }
}

// One-element forwarder: recv then send.
proc os_fwd {
    f_in: chan<F32> in;
    f_out: chan<F32> out;

    config(f_in: chan<F32> in, f_out: chan<F32> out) { (f_in, f_out) }

    init { () }

    next(st: ()) {
        let (tok, v) = recv(join(), f_in);
        let tok = send(tok, f_out, v);
    }
}

// The grid: M*N PEs.  a_in[i] feeds row i; b_in[j] feeds col j
pub proc os_matmul<M: u32, K: u32, N: u32, DEPTH: u32 = {u32:1}> {
    // row i activations:  A[i][0], A[i][1], ...
    a_in: chan<F32>[M] in;
    // col j weights:      B[0][j], B[1][j], ...
    b_in: chan<F32>[N] in;
    // C[i][j] on y_out[i][j]
    y_out: chan<F32>[N][M] out;

    config(a_in: chan<F32>[M] in, b_in: chan<F32>[N] in, y_out: chan<F32>[N][M] out) {
        // Activation links, one extra COLUMN per row: arow[i][j] flows INTO column j of
        // row i (j in 0..=N).  PE(i,j) reads arow[i][j], writes arow[i][j+1].
        let (arp, arc) = chan<F32, DEPTH>[N + u32:1][M]("arow");

        // Weight links, one extra ROW per column: bcol[i][j] flows INTO row i of column j
        // (i in 0..=M).  PE(i,j) reads bcol[i][j], writes bcol[i+1][j].
        let (bcp, bcc) = chan<F32, DEPTH>[N][M + u32:1]("bcol");

        // west edge: stage each external row input into the grid
        unroll_for! (i, ()): (u32, ()) in u32:0..M {
            spawn os_fwd(a_in[i], arp[i][u32:0]);
        }(());
        // east edge: drain each row's spent activation
        unroll_for! (i, ()): (u32, ()) in u32:0..M {
            spawn os_drain(arc[i][N]);
        }(());
        // north edge: stage each external column input into the grid
        unroll_for! (j, ()): (u32, ()) in u32:0..N {
            spawn os_fwd(b_in[j], bcp[u32:0][j]);
        }(());
        // south edge: drain each column's spent weight
        unroll_for! (j, ()): (u32, ()) in u32:0..N {
            spawn os_drain(bcc[M][j]);
        }(());

        // the M x N grid -- every neighbour is a plain array index, no conditionals
        // os_pe args: west in, north in, east out, south out, C[i][j] out
        unroll_for! (i, ()): (u32, ()) in u32:0..M {
            unroll_for! (j, ()): (u32, ()) in u32:0..N {
                spawn os_pe<K>(
                    arc[i][j], bcc[i][j], arp[i][j + u32:1], bcp[i + u32:1][j], y_out[i][j]);
            }(());
        }(());
        (a_in, b_in, y_out)
    }

    init { () }

    next(st: ()) { () }
}

// ================================= OUTPUT-STATIONARY TESTS ==================================
// Wrappers for use in testing
// M=2 batch, K=2 contraction, N=2 outputs -> 2x2 grid.
proc os_2x2x2 {
    config(a_in: chan<F32>[2] in, b_in: chan<F32>[2] in, y_out: chan<F32>[2][2] out) {
        spawn os_matmul<u32:2, u32:2, u32:2>(a_in, b_in, y_out);
        ()
    }

    init { () }

    next(st: ()) { () }
}

// Rectangular instance with a genuine INTERIOR PE(1,1): M=2, K=3, N=3 -> 2x3 grid.
//   A=[[1,2,3],[4,5,6]] (2x3), B=[[1,2,3],[4,5,6],[7,8,9]] (3x3)
//   C = [[30,36,42],[66,81,96]]  (2x3)
proc os_2x3x3 {
    config(a_in: chan<F32>[2] in, b_in: chan<F32>[3] in, y_out: chan<F32>[3][2] out) {
        spawn os_matmul<u32:2, u32:3, u32:3>(a_in, b_in, y_out);
        ()
    }

    init { () }

    next(st: ()) { () }
}

// Square 3x3 grid with interior PE(1,1): M=3, K=3, N=3.
//   A=B=[[1,2,3],[4,5,6],[7,8,9]]  ->  C=[[30,36,42],[66,81,96],[102,126,150]]
proc os_3x3x3 {
    config(a_in: chan<F32>[3] in, b_in: chan<F32>[3] in, y_out: chan<F32>[3][3] out) {
        spawn os_matmul<u32:3, u32:3, u32:3>(a_in, b_in, y_out);
        ()
    }

    init { () }

    next(st: ()) { () }
}

// --------------------- 2x2 * 2x2 -> 2x2 ---------------------
#[test_proc]
proc test_os_2x2x2 {
    terminator: chan<bool> out;
    // row-0, row-1 activation feeds
    a0: chan<F32> out;
    a1: chan<F32> out;
    // col-0, col-1 weight feeds
    b0: chan<F32> out;
    b1: chan<F32> out;
    y0: chan<F32> in;
    y1: chan<F32> in;
    y2: chan<F32> in;
    y3: chan<F32> in;

    config(terminator: chan<bool> out) {
        let (a_p, a_c) = chan<F32>[2]("a");
        let (b_p, b_c) = chan<F32>[2]("b");
        let (y_p, y_c) = chan<F32>[2][2]("y");
        spawn os_2x2x2(a_c, b_c, y_p);
        (
            terminator, a_p[u32:0], a_p[u32:1], b_p[u32:0], b_p[u32:1], y_c[u32:0][u32:0],
            y_c[u32:0][u32:1], y_c[u32:1][u32:0], y_c[u32:1][u32:1],
        )
    }

    init { u32:0 }

    next(c: u32) {
        let tok = join();
        // row i gets A[i][*]:  a0 = 1,2 ; a1 = 3,4
        let tok = send_if(tok, a0, c == u32:0, f32c(u32:0x3f800000));  // A[0][0]=1
        let tok = send_if(tok, a0, c == u32:1, f32c(u32:0x40000000));  // A[0][1]=2
        let tok = send_if(tok, a1, c == u32:0, f32c(u32:0x40400000));  // A[1][0]=3
        let tok = send_if(tok, a1, c == u32:1, f32c(u32:0x40800000));  // A[1][1]=4
        // col j gets B[*][j]:  b0 = 1,3 ; b1 = 2,4
        let tok = send_if(tok, b0, c == u32:0, f32c(u32:0x3f800000));  // B[0][0]=1
        let tok = send_if(tok, b0, c == u32:1, f32c(u32:0x40400000));  // B[1][0]=3
        let tok = send_if(tok, b1, c == u32:0, f32c(u32:0x40000000));  // B[0][1]=2
        let tok = send_if(tok, b1, c == u32:1, f32c(u32:0x40800000));  // B[1][1]=4
        // outputs stay put until k=K-1, then all four drain together
        let chk = c == u32:3;
        let (tok, r0) = recv_if(tok, y0, chk, float32::zero(false));  // C[0][0]=7
        let (tok, r1) = recv_if(tok, y1, chk, float32::zero(false));  // C[0][1]=10
        let (tok, r2) = recv_if(tok, y2, chk, float32::zero(false));  // C[1][0]=15
        let (tok, r3) = recv_if(tok, y3, chk, float32::zero(false));  // C[1][1]=22
        let tok = if chk {
            assert_eq(r0, f32c(u32:0x40e00000));  // 7
            assert_eq(r1, f32c(u32:0x41200000));  // 10
            assert_eq(r2, f32c(u32:0x41700000));  // 15
            assert_eq(r3, f32c(u32:0x41b00000));  // 22
            send(tok, terminator, true)
        } else {
            tok
        };
        c + u32:1
    }
}

// A=[[0.5,1.5],[2.5,0.25]], B=[[1.5,0.5],[0.25,2.0]]  ->  C=A*B
// C[0][0]=0.5*1.5+1.5*0.25 = 1.125 ; C[0][1]=0.5*0.5+1.5*2.0 = 3.25
// C[1][0]=2.5*1.5+0.25*0.25 = 3.8125 ; C[1][1]=2.5*0.5+0.25*2.0 = 1.75
#[test_proc]
proc test_os_2x2x2_fp {
    terminator: chan<bool> out;
    a0: chan<F32> out;
    a1: chan<F32> out;
    b0: chan<F32> out;
    b1: chan<F32> out;
    y0: chan<F32> in;
    y1: chan<F32> in;
    y2: chan<F32> in;
    y3: chan<F32> in;

    config(terminator: chan<bool> out) {
        let (a_p, a_c) = chan<F32>[2]("a");
        let (b_p, b_c) = chan<F32>[2]("b");
        let (y_p, y_c) = chan<F32>[2][2]("y");
        spawn os_2x2x2(a_c, b_c, y_p);
        (
            terminator, a_p[u32:0], a_p[u32:1], b_p[u32:0], b_p[u32:1], y_c[u32:0][u32:0],
            y_c[u32:0][u32:1], y_c[u32:1][u32:0], y_c[u32:1][u32:1],
        )
    }

    init { u32:0 }

    next(c: u32) {
        let tok = join();
        // row i gets A[i][*]:  a0 = 0.5,1.5 ; a1 = 2.5,0.25
        let tok = send_if(tok, a0, c == u32:0, f32c(u32:0x3f000000));  // A[0][0]=0.5
        let tok = send_if(tok, a0, c == u32:1, f32c(u32:0x3fc00000));  // A[0][1]=1.5
        let tok = send_if(tok, a1, c == u32:0, f32c(u32:0x40200000));  // A[1][0]=2.5
        let tok = send_if(tok, a1, c == u32:1, f32c(u32:0x3e800000));  // A[1][1]=0.25
        // col j gets B[*][j]:  b0 = 1.5,0.25 ; b1 = 0.5,2.0
        let tok = send_if(tok, b0, c == u32:0, f32c(u32:0x3fc00000));  // B[0][0]=1.5
        let tok = send_if(tok, b0, c == u32:1, f32c(u32:0x3e800000));  // B[1][0]=0.25
        let tok = send_if(tok, b1, c == u32:0, f32c(u32:0x3f000000));  // B[0][1]=0.5
        let tok = send_if(tok, b1, c == u32:1, f32c(u32:0x40000000));  // B[1][1]=2.0
        let chk = c == u32:3;
        let (tok, r0) = recv_if(tok, y0, chk, float32::zero(false));
        let (tok, r1) = recv_if(tok, y1, chk, float32::zero(false));
        let (tok, r2) = recv_if(tok, y2, chk, float32::zero(false));
        let (tok, r3) = recv_if(tok, y3, chk, float32::zero(false));
        let tok = if chk {
            assert_eq(r0, f32c(u32:0x3f900000));  // 1.125
            assert_eq(r1, f32c(u32:0x40500000));  // 3.25
            assert_eq(r2, f32c(u32:0x40740000));  // 3.8125
            assert_eq(r3, f32c(u32:0x3fe00000));  // 1.75
            send(tok, terminator, true)
        } else {
            tok
        };
        c + u32:1
    }
}

// --------------------- 2x3 * 3x3 -> 2x3 ---------------------
// A=[[1,2,3],[4,5,6]] (2x3), B=[[1,2,3],[4,5,6],[7,8,9]] (3x3)
// C = [[30,36,42],[66,81,96]]  (2x3).  Each PE emits its C[i][j]
// exactly once after the K=3 contraction
#[test_proc]
proc test_os_2x3x3 {
    terminator: chan<bool> out;
    // row feeds:  a[i] streams A[i][*]
    a: chan<F32>[2] out;
    // col feeds:  b[j] streams B[*][j]
    b: chan<F32>[3] out;
    // yy[i][j] = C[i][j]
    yy: chan<F32>[3][2] in;

    config(terminator: chan<bool> out) {
        let (a_p, a_c) = chan<F32>[2]("a");
        let (b_p, b_c) = chan<F32>[3]("b");
        let (y_p, y_c) = chan<F32>[3][2]("y");
        spawn os_2x3x3(a_c, b_c, y_p);
        (terminator, a_p, b_p, y_c)
    }

    init { u32:0 }

    next(c: u32) {
        let tok = join();
        let z = float32::zero(false);
        // A[i][k] on a[i] at cycle k:  row 0 = 1,2,3 ; row 1 = 4,5,6
        let arows = fv2(
            u32[3][2]:[
                [0x3f800000, 0x40000000, 0x40400000], // 1, 2, 3
                [0x40800000, 0x40a00000, 0x40c00000], // 4, 5, 6
            ]);
        let tok = unroll_for! (i, tok): (u32, token) in u32:0..u32:2 {
            unroll_for! (k, tok): (u32, token) in u32:0..u32:3 {
                send_if(tok, a[i], c == k, arows[i][k])
            }(tok)
        }(tok);
        // B[k][j] on b[j] at cycle k:  col 0 = 1,4,7 ; col 1 = 2,5,8 ; col 2 = 3,6,9
        let bcols = fv2(
            u32[3][3]:[
                [0x3f800000, 0x40800000, 0x40e00000], // 1, 4, 7
                [0x40000000, 0x40a00000, 0x41000000], // 2, 5, 8
                [0x40400000, 0x40c00000, 0x41100000], // 3, 6, 9
            ]);
        let tok = unroll_for! (j, tok): (u32, token) in u32:0..u32:3 {
            unroll_for! (k, tok): (u32, token) in u32:0..u32:3 {
                send_if(tok, b[j], c == k, bcols[j][k])
            }(tok)
        }(tok);
        // collect all six C[i][j] together once the contraction has drained
        let exp = fv2(
            u32[3][2]:[
                [0x41f00000, 0x42100000, 0x42280000], // 30, 36, 42
                [0x42840000, 0x42a20000, 0x42c00000], // 66, 81, 96
            ]);
        let chk = c == u32:5;
        let tok = unroll_for! (i, tok): (u32, token) in u32:0..u32:2 {
            unroll_for! (j, tok): (u32, token) in u32:0..u32:3 {
                let (tok, r) = recv_if(tok, yy[i][j], chk, z);
                if chk { assert_eq(r, exp[i][j]); } else { () };
                tok
            }(tok)
        }(tok);
        let tok = if chk { send(tok, terminator, true) } else { tok };
        c + u32:1
    }
}

// --------------------- 3x3 * 3x3 -> 3x3 ---------------------
// A=B=[[1,2,3],[4,5,6],[7,8,9]]  ->  C=[[30,36,42],[66,81,96],[102,126,150]].
#[test_proc]
proc test_os_3x3x3 {
    terminator: chan<bool> out;
    // row feeds:  a[i] streams A[i][*]
    a: chan<F32>[3] out;
    // col feeds:  b[j] streams B[*][j]
    b: chan<F32>[3] out;
    // yy[i][j] = C[i][j]
    yy: chan<F32>[3][3] in;

    config(terminator: chan<bool> out) {
        let (a_p, a_c) = chan<F32>[3]("a");
        let (b_p, b_c) = chan<F32>[3]("b");
        let (y_p, y_c) = chan<F32>[3][3]("y");
        spawn os_3x3x3(a_c, b_c, y_p);
        (terminator, a_p, b_p, y_c)
    }

    init { u32:0 }

    next(c: u32) {
        let tok = join();
        let z = float32::zero(false);
        // A[i][k] on a[i] at cycle k:  rows 1,2,3 / 4,5,6 / 7,8,9
        let arows = fv2(
            u32[3][3]:[
                [0x3f800000, 0x40000000, 0x40400000], // 1, 2, 3
                [0x40800000, 0x40a00000, 0x40c00000], // 4, 5, 6
                [0x40e00000, 0x41000000, 0x41100000], // 7, 8, 9
            ]);
        let tok = unroll_for! (i, tok): (u32, token) in u32:0..u32:3 {
            unroll_for! (k, tok): (u32, token) in u32:0..u32:3 {
                send_if(tok, a[i], c == k, arows[i][k])
            }(tok)
        }(tok);
        // B[k][j] on b[j] at cycle k:  cols 1,4,7 / 2,5,8 / 3,6,9
        let bcols = fv2(
            u32[3][3]:[
                [0x3f800000, 0x40800000, 0x40e00000], // 1, 4, 7
                [0x40000000, 0x40a00000, 0x41000000], // 2, 5, 8
                [0x40400000, 0x40c00000, 0x41100000], // 3, 6, 9
            ]);
        let tok = unroll_for! (j, tok): (u32, token) in u32:0..u32:3 {
            unroll_for! (k, tok): (u32, token) in u32:0..u32:3 {
                send_if(tok, b[j], c == k, bcols[j][k])
            }(tok)
        }(tok);
        // collect all nine C[i][j] together once the contraction has drained
        let exp = fv2(
            u32[3][3]:[
                [0x41f00000, 0x42100000, 0x42280000], // 30, 36, 42
                [0x42840000, 0x42a20000, 0x42c00000], // 66, 81, 96
                [0x42cc0000, 0x42fc0000, 0x43160000], // 102, 126, 150
            ]);
        let chk = c == u32:6;
        let tok = unroll_for! (i, tok): (u32, token) in u32:0..u32:3 {
            unroll_for! (j, tok): (u32, token) in u32:0..u32:3 {
                let (tok, r) = recv_if(tok, yy[i][j], chk, z);
                if chk { assert_eq(r, exp[i][j]); } else { () };
                tok
            }(tok)
        }(tok);
        let tok = if chk { send(tok, terminator, true) } else { tok };
        c + u32:1
    }
}
