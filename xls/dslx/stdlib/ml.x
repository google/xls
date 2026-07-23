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

// ml.x -- machine-learning primitives for the DSLX standard library (float32).
//
// A self-contained candidate stdlib module bundling the FP32 building blocks we
// use to assemble small MLP / matmul accelerators. Current blocks include:
// * relu_f32_scalar          -- scalar ReLU activation
// * relu_f32                 -- elementwise ReLU activation
// * ws_matmul                -- parametric weight-stationary systolic matmul array
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

// ============================= ReLU (FP32) =============================
// y = max(0, x). A float is negative exactly when its sign bit is set (covers
// -0 too, which maps to +0 -- fine). One comparator + mux per element, fully
// parallel and essentially zero logic depth.

pub fn relu_f32_scalar(x: F32) -> F32 { if x.sign { float32::zero(false) } else { x } }

pub fn relu_f32<L: u32>(x: F32[L]) -> F32[L] {
    for (i, y): (u32, F32[L]) in u32:0..L {
        update(y, i, relu_f32_scalar(x[i]))
    }(F32[L]:[float32::zero(false), ...])
}

// ============================= RELU TESTS =============================
#[test]
fn test_relu_f32_scalar() {
    let pz = f32c(u32:0x00000000);  // +0.0
    assert_eq(relu_f32_scalar(f32c(u32:0x40e00000)), f32c(u32:0x40e00000));  // 7.0 -> 7.0
    assert_eq(relu_f32_scalar(f32c(u32:0xbf000000)), pz);  // -0.5 -> +0
    assert_eq(relu_f32_scalar(pz), pz);  // +0.0 -> +0
    assert_eq(relu_f32_scalar(f32c(u32:0x80000000)), pz);  // -0.0 -> +0
    assert_eq(relu_f32_scalar(f32c(u32:0x7f800000)), f32c(u32:0x7f800000));  // +inf -> +inf
    assert_eq(relu_f32_scalar(f32c(u32:0xff800000)), pz);  // -inf -> +0
}

#[test]
fn test_relu_f32() {
    let x = fv(u32[4]:[0xbf800000, 0x00000000, 0x40e00000, 0xbf000000]);  // -1, 0, 7, -0.5
    let want = fv(u32[4]:[0x00000000, 0x00000000, 0x40e00000, 0x00000000]);  //  0, 0, 7,  0
    assert_eq(relu_f32<u32:4>(x), want)
}

#[test]
fn test_relu_f32_all_negative() {
    let x = fv(u32[3]:[0xbf800000, 0xc0000000, 0xc0a00000]);  // -1, -2, -5
    let want = fv(u32[3]:[0x00000000, 0x00000000, 0x00000000]);  //  0,  0,  0
    assert_eq(relu_f32<u32:3>(x), want)
}

#[test]
fn test_relu_f32_all_positive() {
    let x = fv(u32[3]:[0x3f800000, 0x40000000, 0x40a00000]);  // 1, 2, 5 -- unchanged
    assert_eq(relu_f32<u32:3>(x), x)
}

// ================ WEIGHT-STATIONARY SYSTOLIC MATMUL (FP32) =============
// To make the systolic array proc stateless, we need four kinds of processing
// elements to account for the various edge / corner behaviors. For instance, a
// PE on the top row should not receive a partial sum from the north. Each PE's
// unique behavior is detailed below.

// the full interior PE that receives partials from the north and activations from the west,
// forwards accumulated partial south and activations east
proc ws_pe_fwd {
    weight_in: chan<F32> in;
    // from the west
    activation_in: chan<F32> in;
    // psum from the north
    partial_in: chan<F32> in;
    // to the east
    activation_out: chan<F32> out;
    // psum to the south
    partial_out: chan<F32> out;

    config(w_in: chan<F32> in, a_in: chan<F32> in, p_in: chan<F32> in, a_out: chan<F32> out,
           p_out: chan<F32> out) {
        (w_in, a_in, p_in, a_out, p_out)
    }

    init { (float32::zero(false), false) }

    next(st: (F32, bool)) {
        let tok = token();

        // latch weight if not yet loaded
        let (weight, loaded) = st;
        let (tok, weight) = recv_if(tok, weight_in, !loaded, weight);

        // receive activation and partial, then multiply accumulate
        let (tok, activation) = recv(tok, activation_in);
        let (tok, partial) = recv(tok, partial_in);
        let partial_sum = float32::fma(weight, activation, partial);

        // send activation and accumulated partial
        let tok = send(tok, activation_out, activation);
        let tok = send(tok, partial_out, partial_sum);

        // set weight to latched for next state
        (weight, true)
    }
}

// PEs for the top row, that do not receive any input from the north
proc ws_pe_top {
    weight_in: chan<F32> in;
    activation_in: chan<F32> in;
    activation_out: chan<F32> out;
    partial_out: chan<F32> out;

    config(w_in: chan<F32> in, a_in: chan<F32> in, a_out: chan<F32> out, p_out: chan<F32> out) {
        (w_in, a_in, a_out, p_out)
    }

    init { (float32::zero(false), false) }

    next(st: (F32, bool)) {
        let tok = token();

        // latch weight it not yet loaded
        let (weight, loaded) = st;
        let (tok, weight) = recv_if(tok, weight_in, !loaded, weight);

        // receive activation, no partial to be received; multiply with weight
        let (tok, activation) = recv(tok, activation_in);
        let partial_sum = float32::fma(weight, activation, float32::zero(false));

        // forward activation and partials
        let tok = send(tok, activation_out, activation);
        let tok = send(tok, partial_out, partial_sum);

        // set weight to latched for next state
        (weight, true)
    }
}

// PEs for the rightmost column, that drain activations rather than forwarding them eastward
proc ws_pe_drain {
    weight_in: chan<F32> in;
    activation_in: chan<F32> in;
    partial_in: chan<F32> in;
    partial_out: chan<F32> out;

    config(w_in: chan<F32> in, a_in: chan<F32> in, p_in: chan<F32> in, p_out: chan<F32> out) {
        (w_in, a_in, p_in, p_out)
    }

    init { (float32::zero(false), false) }

    next(st: (F32, bool)) {
        let tok = token();

        // latch weight if not loaded
        let (weight, loaded) = st;
        let (tok, weight) = recv_if(tok, weight_in, !loaded, weight);

        // receive activation and partial, then multiply accumulate
        let (tok, activation) = recv(tok, activation_in);
        let (tok, partial) = recv(tok, partial_in);
        let partial_sum = float32::fma(weight, activation, partial);

        // forward partial, activations are drained on right edge
        let tok = send(tok, partial_out, partial_sum);

        // set weight to latched for next state
        (weight, true)
    }
}

// PE for the top-right corner, which neither takes from the north nor forwards east
proc ws_pe_topdrain {
    weight_in: chan<F32> in;
    activation_in: chan<F32> in;
    partial_out: chan<F32> out;

    config(w_in: chan<F32> in, a_in: chan<F32> in, p_out: chan<F32> out) { (w_in, a_in, p_out) }

    init { (float32::zero(false), false) }

    next(st: (F32, bool)) {
        let tok = token();

        // latch weight if not loaded
        let (weight, loaded) = st;
        let (tok, weight) = recv_if(tok, weight_in, !loaded, weight);

        // receive activation, no partial to be received; multiply with weight
        let (tok, activation) = recv(tok, activation_in);
        let partial_sum = float32::fma(weight, activation, float32::zero(false));

        // forward partial, activations are drained on right edge
        let tok = send(tok, partial_out, partial_sum);

        // set weight to latched for next state
        (weight, true)
    }
}

// weight-stationary systolic array with K*N PEs (where K and N are parameterized)
pub proc ws_matmul<K: u32, N: u32, DEPTH: u32 = {u32:1}> {
    w_in: chan<F32>[N][K] in;
    // PE(k,j) weight = w_in[k][j]
    a_in: chan<F32>[K] in;
    // row k activations: A[0][k], A[1][k], ...
    c_out: chan<F32>[N] out;

    // col j outputs: C[0][j], C[1][j], ...
    config(w_in: chan<F32>[N][K] in, a_in: chan<F32>[K] in, c_out: chan<F32>[N] out) {
        // design assumes N >= 2 and K >= 2 so we assert this condition here
        const_assert!(K >= u32:2);
        const_assert!(N >= u32:2);

        // activation links: to_east[k][j] carries PE(k,j)'s activation to PE(k,j+1), for j = 0..N-2
        let (to_east, from_west) = chan<F32, DEPTH>[N - u32:1][K]("arow");
        // partial-sum links: to_south[k][j] carries PE(k,j)'s psum to PE(k+1,j), for k = 0..K-2.
        let (to_south, from_north) = chan<F32, DEPTH>[N][K - u32:1]("pcol");

        // --- four corners ---
        // (0, 0) top-left: inject 0.0, forward east, south to link
        spawn ws_pe_top(
            w_in[u32:0][u32:0], a_in[u32:0], to_east[u32:0][u32:0], to_south[u32:0][u32:0]);

        // (0, N-1) top-right: inject 0.0, drop east, south to link
        spawn ws_pe_topdrain(
            w_in[u32:0][N - u32:1], from_west[u32:0][N - u32:2], to_south[u32:0][N - u32:1]);

        // (K-1, 0) bottom-left: recv north, forward east, south to c_out[0]
        spawn ws_pe_fwd(
            w_in[K - u32:1][u32:0], a_in[K - u32:1], from_north[K - u32:2][u32:0],
            to_east[K - u32:1][u32:0], c_out[u32:0]);

        // (K-1, N-1) bottom-right: recv north, drop east, south to c_out[N-1]
        spawn ws_pe_drain(
            w_in[K - u32:1][N - u32:1], from_west[K - u32:1][N - u32:2],
            from_north[K - u32:2][N - u32:1], c_out[N - u32:1]);

        // --- four edges (each varies one index between the corners) ---
        // left edge: col 0, rows 1..K-2  (recv north, forward east, west = a_in)
        unroll_for! (k, ()): (u32, ()) in u32:1..(K - u32:1) {
            spawn ws_pe_fwd(
                w_in[k][u32:0], a_in[k], from_north[k - u32:1][u32:0], to_east[k][u32:0],
                to_south[k][u32:0]);
        }(());

        // right edge: col N-1, rows 1..K-2  (recv north, drop east)
        unroll_for! (k, ()): (u32, ()) in u32:1..(K - u32:1) {
            spawn ws_pe_drain(
                w_in[k][N - u32:1], from_west[k][N - u32:2], from_north[k - u32:1][N - u32:1],
                to_south[k][N - u32:1]);
        }(());

        // top edge: row 0, cols 1..N-2  (inject 0.0, forward east)
        unroll_for! (j, ()): (u32, ()) in u32:1..(N - u32:1) {
            spawn ws_pe_top(
                w_in[u32:0][j], from_west[u32:0][j - u32:1], to_east[u32:0][j], to_south[u32:0][j]);
        }(());

        // bottom edge: row K-1, cols 1..N-2  (recv north, forward east, south = c_out[j])
        unroll_for! (j, ()): (u32, ()) in u32:1..(N - u32:1) {
            spawn ws_pe_fwd(
                w_in[K - u32:1][j], from_west[K - u32:1][j - u32:1], from_north[K - u32:2][j],
                to_east[K - u32:1][j], c_out[j]);
        }(());

        // --- interior: rows 1..K-2, cols 1..N-2  (recv north, forward east, all links) ---
        unroll_for! (k, ()): (u32, ()) in u32:1..(K - u32:1) {
            unroll_for! (j, ()): (u32, ()) in u32:1..(N - u32:1) {
                spawn ws_pe_fwd(
                    w_in[k][j], from_west[k][j - u32:1], from_north[k - u32:1][j], to_east[k][j],
                    to_south[k][j]);
            }(());
        }(());

        // maintain channels for next state
        (w_in, a_in, c_out)
    }

    init { () }

    next(st: ()) { () }
}

// ================================= WEIGHT-STATIONARY TESTS ==================================
// Concrete grid wrappers -- XLS requires a non-parametric top to spawn from a
// #[test_proc], so each tested shape gets a thin wrapper around it.
proc ws_2x2 {
    config(w: chan<F32>[2][2] in, a: chan<F32>[2] in, c: chan<F32>[2] out) {
        spawn ws_matmul<u32:2, u32:2>(w, a, c);
        ()
    }

    init { () }

    next(st: ()) { () }
}

proc ws_3x3 {
    config(w: chan<F32>[3][3] in, a: chan<F32>[3] in, c: chan<F32>[3] out) {
        spawn ws_matmul<u32:3, u32:3>(w, a, c);
        ()
    }

    init { () }

    next(st: ()) { () }
}

proc ws_4x4 {
    config(w: chan<F32>[4][4] in, a: chan<F32>[4] in, c: chan<F32>[4] out) {
        spawn ws_matmul<u32:4, u32:4>(w, a, c);
        ()
    }

    init { () }

    next(st: ()) { () }
}

proc ws_2x3 {
    // non-square grid
    config(w: chan<F32>[3][2] in, a: chan<F32>[2] in, c: chan<F32>[3] out) {
        spawn ws_matmul<u32:2, u32:3>(w, a, c);
        ()
    }

    init { () }

    next(st: ()) { () }
}

// --------------------- 2x2 * 2x2 -> 2x2 ---------------------
#[test_proc]
proc test_ws_2x2 {
    terminator: chan<bool> out;
    // weights B[k][j] on w[k][j], sent once
    w: chan<F32>[2][2] out;
    // row feeds: a[k] streams A[*][k]
    a: chan<F32>[2] out;
    // col outputs: cc[j] streams C[*][j]
    cc: chan<F32>[2] in;

    config(terminator: chan<bool> out) {
        let (w_p, w_c) = chan<F32>[2][2]("w");
        let (a_p, a_c) = chan<F32>[2]("a");
        let (c_p, c_c) = chan<F32>[2]("c");
        spawn ws_2x2(w_c, a_c, c_p);
        (terminator, w_p, a_p, c_c)
    }

    init { u32:0 }

    next(c: u32) {
        let z = float32::zero(false);

        // wv = [[1, 2],
        //       [3, 4]]
        let wv = fv2(u32[2][2]:[[0x3f800000, 0x40000000], [0x40400000, 0x40800000]]);
        let tok = unroll_for! (k, tok): (u32, token) in u32:0..u32:2 {
            unroll_for! (j, tok): (u32, token) in u32:0..u32:2 {
                send_if(tok, w[k][j], c == u32:0, wv[k][j])
            }(tok)
        }(join());

        // img = [[1, 2],
        //        [3, 4]]
        let img = fv2(u32[2][2]:[[0x3f800000, 0x40000000], [0x40400000, 0x40800000]]);
        let tok = unroll_for! (i, tok): (u32, token) in u32:0..u32:2 {
            unroll_for! (k, tok): (u32, token) in u32:0..u32:2 {
                send_if(tok, a[k], c == i, img[i][k])
            }(tok)
        }(tok);

        // exp = [[7, 10],
        //        [15, 22]]
        let exp = fv2(u32[2][2]:[[0x40e00000, 0x41200000], [0x41700000, 0x41b00000]]);
        let chk0 = c == u32:3;
        let chk1 = c == u32:4;
        let tok = unroll_for! (j, tok): (u32, token) in u32:0..u32:2 {
            let (tok, r0) = recv_if(tok, cc[j], chk0, z);
            if chk0 { assert_eq(r0, exp[u32:0][j]); } else { () };
            let (tok, r1) = recv_if(tok, cc[j], chk1, z);
            if chk1 { assert_eq(r1, exp[u32:1][j]); } else { () };
            tok
        }(tok);
        let tok = if chk1 { send(tok, terminator, true) } else { tok };
        c + u32:1
    }
}

// --------------------- 2x3 * 3x3 -> 2x3 ---------------------
#[test_proc]
proc test_ws_3x3_batch2 {
    terminator: chan<bool> out;
    // weights B[k][j] on w[k][j], sent once
    w: chan<F32>[3][3] out;
    // row feeds: a[k] streams A[*][k]
    a: chan<F32>[3] out;
    // col outputs: cc[j] streams C[*][j]
    cc: chan<F32>[3] in;

    config(terminator: chan<bool> out) {
        let (w_p, w_c) = chan<F32>[3][3]("w");
        let (a_p, a_c) = chan<F32>[3]("a");
        let (c_p, c_c) = chan<F32>[3]("c");
        spawn ws_3x3(w_c, a_c, c_p);
        (terminator, w_p, a_p, c_c)
    }

    init { u32:0 }

    next(c: u32) {
        let z = float32::zero(false);

        // wv = [[1, 2, 3],
        //       [4, 5, 6],
        //       [7, 8, 9]]
        let wv = fv2(
            u32[3][3]:[
                [0x3f800000, 0x40000000, 0x40400000], [0x40800000, 0x40a00000, 0x40c00000],
                [0x40e00000, 0x41000000, 0x41100000],
            ]);
        let tok = unroll_for! (k, tok): (u32, token) in u32:0..u32:3 {
            unroll_for! (j, tok): (u32, token) in u32:0..u32:3 {
                send_if(tok, w[k][j], c == u32:0, wv[k][j])
            }(tok)
        }(join());

        // img = [[1, 2, 3],
        //        [4, 5, 6]]
        let img = fv2(
            u32[3][2]:[[0x3f800000, 0x40000000, 0x40400000], [0x40800000, 0x40a00000, 0x40c00000]]);
        let tok = unroll_for! (k, tok): (u32, token) in u32:0..u32:3 {
            let tok = send_if(tok, a[k], c == u32:0, img[u32:0][k]);
            send_if(tok, a[k], c == u32:1, img[u32:1][k])
        }(tok);

        // exp = [[30, 36, 42],
        //        [66, 81, 96]]
        let exp = fv2(
            u32[3][2]:[[0x41f00000, 0x42100000, 0x42280000], [0x42840000, 0x42a20000, 0x42c00000]]);
        let chk0 = c == u32:7;
        let chk1 = c == u32:8;
        let tok = unroll_for! (j, tok): (u32, token) in u32:0..u32:3 {
            let (tok, r0) = recv_if(tok, cc[j], chk0, z);
            if chk0 { assert_eq(r0, exp[u32:0][j]); } else { () };
            let (tok, r1) = recv_if(tok, cc[j], chk1, z);
            if chk1 { assert_eq(r1, exp[u32:1][j]); } else { () };
            tok
        }(tok);
        let tok = if chk1 { send(tok, terminator, true) } else { tok };
        c + u32:1
    }
}

// --------------------- 2x3 * 3x3 -> 2x3 ---------------------
#[test_proc]
proc test_ws_3x3_batch2_fp {
    terminator: chan<bool> out;
    // weights B[k][j] on w[k][j], sent once
    w: chan<F32>[3][3] out;
    // row feeds: a[k] streams A[*][k]
    a: chan<F32>[3] out;
    // col outputs: cc[j] streams C[*][j]
    cc: chan<F32>[3] in;

    config(terminator: chan<bool> out) {
        let (w_p, w_c) = chan<F32>[3][3]("w");
        let (a_p, a_c) = chan<F32>[3]("a");
        let (c_p, c_c) = chan<F32>[3]("c");
        spawn ws_3x3(w_c, a_c, c_p);
        (terminator, w_p, a_p, c_c)
    }

    init { u32:0 }

    next(c: u32) {
        let z = float32::zero(false);

        // wv = [[ 1.5, -2.5,  0.9],
        //       [-2.5,  2.25, 0.0625],
        //       [ 3.3,  1.3,  4.4]]
        let wv = fv2(
            u32[3][3]:[
                [0x3fc00000, 0xc0200000, 0x3f666666], [0xc0200000, 0x40100000, 0x3d800000],
                [0x40533333, 0x3fa66666, 0x408ccccd],
            ]);
        let tok = unroll_for! (k, tok): (u32, token) in u32:0..u32:3 {
            unroll_for! (j, tok): (u32, token) in u32:0..u32:3 {
                send_if(tok, w[k][j], c == u32:0, wv[k][j])
            }(tok)
        }(join());

        // img = [[-3.7,   2.2,   3.75],
        //        [-0.35, -1.25, -1.1]]
        let img = fv2(
            u32[3][2]:[[0xc06ccccd, 0x400ccccd, 0x40700000], [0xbeb33333, 0xbfa00000, 0xbf8ccccd]]);
        let tok = unroll_for! (k, tok): (u32, token) in u32:0..u32:3 {
            let tok = send_if(tok, a[k], c == u32:0, img[u32:0][k]);
            send_if(tok, a[k], c == u32:1, img[u32:1][k])
        }(tok);

        // exp = [[ 1.3249996, 19.074999,  13.307501],
        //        [-1.0300001, -3.3675001, -5.2331252]]
        let exp = fv2(
            u32[3][2]:[[0x3fa99996, 0x41989999, 0x4154eb86], [0xbf83d70b, 0xc057851f, 0xc0a775c3]]);
        let chk0 = c == u32:7;
        let chk1 = c == u32:8;
        let tok = unroll_for! (j, tok): (u32, token) in u32:0..u32:3 {
            let (tok, r0) = recv_if(tok, cc[j], chk0, z);
            if chk0 { assert_eq(r0, exp[u32:0][j]); } else { () };
            let (tok, r1) = recv_if(tok, cc[j], chk1, z);
            if chk1 { assert_eq(r1, exp[u32:1][j]); } else { () };
            tok
        }(tok);
        let tok = if chk1 { send(tok, terminator, true) } else { tok };
        c + u32:1
    }
}

// --------------------- 4x4 * 4x4 -> 4x4 ---------------------
#[test_proc]
proc test_ws_4x4_fp {
    terminator: chan<bool> out;
    // weights B[k][j] on w[k][j], sent once
    w: chan<F32>[4][4] out;
    // row feeds: a[k] streams A[*][k]
    a: chan<F32>[4] out;
    // col outputs: cc[j] streams C[*][j]
    cc: chan<F32>[4] in;

    config(terminator: chan<bool> out) {
        let (w_p, w_c) = chan<F32>[4][4]("w");
        let (a_p, a_c) = chan<F32>[4]("a");
        let (c_p, c_c) = chan<F32>[4]("c");
        spawn ws_4x4(w_c, a_c, c_p);
        (terminator, w_p, a_p, c_c)
    }

    init { u32:0 }

    next(c: u32) {
        let z = float32::zero(false);

        // wv = [[ 0.2, -0.7,  -0.35,  2.2],
        //       [ 1.5, -6.1,  -0.35,  0.0625],
        //       [ 4.4,  3.75,  3.75, -0.35],
        //       [-3.7,  1.3,   5.6,  -2.5]]
        let wv = fv2(
            u32[4][4]:[
                [0x3e4ccccd, 0xbf333333, 0xbeb33333, 0x400ccccd],
                [0x3fc00000, 0xc0c33333, 0xbeb33333, 0x3d800000],
                [0x408ccccd, 0x40700000, 0x40700000, 0xbeb33333],
                [0xc06ccccd, 0x3fa66666, 0x40b33333, 0xc0200000],
            ]);
        let tok = unroll_for! (k, tok): (u32, token) in u32:0..u32:4 {
            unroll_for! (j, tok): (u32, token) in u32:0..u32:4 {
                send_if(tok, w[k][j], c == u32:0, wv[k][j])
            }(tok)
        }(join());

        // img = [[3.3,  0.3,  -0.7,  4.4],
        //        [3.75, 2.25,  0.125, 1.1],
        //        [0.1,  2.2,   0.9,   2.25],
        //        [0.7,  2.25,  0.1,   0.9]]
        let img = fv2(
            u32[4][4]:[
                [0x40533333, 0x3e99999a, 0xbf333333, 0x408ccccd],
                [0x40700000, 0x40100000, 0x3e000000, 0x3f8ccccd],
                [0x3dcccccd, 0x400ccccd, 0x3f666666, 0x40100000],
                [0x3f333333, 0x40100000, 0x3dcccccd, 0x3f666666],
            ]);
        let tok = unroll_for! (i, tok): (u32, token) in u32:0..u32:4 {
            unroll_for! (k, tok): (u32, token) in u32:0..u32:4 {
                send_if(tok, a[k], c == i, img[i][k])
            }(tok)
        }(tok);

        // exp = [[-18.25,  -1.045,    20.755001, -3.47625],
        //        [  0.605, -14.45125,  4.52875,   5.596875],
        //        [ -1.045,  -7.19,    15.17,     -5.5825],
        //        [  0.625, -12.67,     4.3825,   -0.604375]]
        let exp = fv2(
            u32[4][4]:[
                [0xc1920000, 0xbf85c28f, 0x41a60a3e, 0xc05e7ae1],
                [0x3f1ae149, 0xc1673852, 0x4090eb85, 0x40b3199a],
                [0xbf85c28e, 0xc0e6147b, 0x4172b852, 0xc0b2a3d7],
                [0x3f200003, 0xc14ab852, 0x408c3d70, 0xbf1ab851],
            ]);
        let reading = c >= u32:4 && c <= u32:7;
        let i = if reading { c - u32:4 } else { u32:0 };
        let tok = unroll_for! (j, tok): (u32, token) in u32:0..u32:4 {
            let (tok, r) = recv_if(tok, cc[j], reading, z);
            if reading { assert_eq(r, exp[i][j]); } else { () };
            tok
        }(tok);
        let tok = if reading && i == u32:3 { send(tok, terminator, true) } else { tok };
        c + u32:1
    }
}

// --------------------- 2x2 * 2x3 -> 2x3 ---------------------
#[test_proc]
proc test_ws_2x3 {
    terminator: chan<bool> out;
    w: chan<F32>[3][2] out;
    a: chan<F32>[2] out;
    cc: chan<F32>[3] in;

    config(terminator: chan<bool> out) {
        let (w_p, w_c) = chan<F32>[3][2]("w");
        let (a_p, a_c) = chan<F32>[2]("a");
        let (c_p, c_c) = chan<F32>[3]("c");
        spawn ws_2x3(w_c, a_c, c_p);
        (terminator, w_p, a_p, c_c)
    }

    init { u32:0 }

    next(c: u32) {
        let z = float32::zero(false);

        // wv = [[1, 2, 3],
        //       [4, 5, 6]]
        let wv = fv2(
            u32[3][2]:[[0x3f800000, 0x40000000, 0x40400000], [0x40800000, 0x40a00000, 0x40c00000]]);
        let tok = unroll_for! (k, tok): (u32, token) in u32:0..u32:2 {
            unroll_for! (j, tok): (u32, token) in u32:0..u32:3 {
                send_if(tok, w[k][j], c == u32:0, wv[k][j])
            }(tok)
        }(join());

        // img = [[1, 2],
        //        [3, 4]]
        let img = fv2(u32[2][2]:[[0x3f800000, 0x40000000], [0x40400000, 0x40800000]]);
        let tok = unroll_for! (i, tok): (u32, token) in u32:0..u32:2 {
            unroll_for! (k, tok): (u32, token) in u32:0..u32:2 {
                send_if(tok, a[k], c == i, img[i][k])
            }(tok)
        }(tok);

        // exp = [[ 9, 12, 15],
        //        [19, 26, 33]]
        let exp = fv2(
            u32[3][2]:[[0x41100000, 0x41400000, 0x41700000], [0x41980000, 0x41d00000, 0x42040000]]);
        let reading = c >= u32:2 && c <= u32:3;
        let i = if reading { c - u32:2 } else { u32:0 };
        let tok = unroll_for! (j, tok): (u32, token) in u32:0..u32:3 {
            let (tok, r) = recv_if(tok, cc[j], reading, z);
            if reading { assert_eq(r, exp[i][j]); } else { () };
            tok
        }(tok);
        let tok = if reading && i == u32:1 { send(tok, terminator, true) } else { tok };
        c + u32:1
    }
}
