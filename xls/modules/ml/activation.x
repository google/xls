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

// activation.x -- floating-point activation functions for ML (float32).
//
// FP32 activation primitives used to assemble small MLP / matmul accelerators:
// * relu_f32_scalar        / relu_f32        -- ReLU:        max(0, x)
// * leaky_relu_f32_scalar  / leaky_relu_f32  -- Leaky ReLU:  x if x>=0 else alpha*x
//                                               (alpha parametric; default 1/16)
// * sigmoid_f32_scalar     / sigmoid_f32     -- Sigmoid:     1/(1+e^-x)   (poly approx)
// * softmax_f32                              -- Softmax:     e^x_i / sum_j e^x_j (approx)
//
// Everything adheres to the IEEE-754 float32 (`float32::F32`) standard.

import float32;

type F32 = float32::F32;

// bit-pattern to F32 helpers, for scalars and vectors
fn f32c(raw: u32) -> F32 { float32::unflatten(raw) }

fn fv<L: u32>(raw: u32[L]) -> F32[L] {
    for (i, v): (u32, F32[L]) in u32:0..L {
        update(v, i, f32c(raw[i]))
    }(zero!<F32[L]>())
}

// ============================= ReLU (FP32) =============================
// y = max(0, x). A float is negative exactly when its sign bit is set.
// One comparator + mux per element, fully parallel and essentially zero logic depth.

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

// ============================= Leaky ReLU (FP32) =============================
// y = x for x >= 0, else alpha*x, with a small positive slope alpha on the
// negative side (so gradients do not vanish for negative inputs).

const LEAKY_ALPHA_DEFAULT = u32:0x3d800000;  // 0.0625 = 1/16

pub fn leaky_relu_f32_scalar<ALPHA_BITS: u32 = {LEAKY_ALPHA_DEFAULT}>(x: F32) -> F32 {
    if x.sign { float32::mul(x, float32::unflatten(ALPHA_BITS)) } else { x }
}

pub fn leaky_relu_f32<L: u32, ALPHA_BITS: u32 = {LEAKY_ALPHA_DEFAULT}>(x: F32[L]) -> F32[L] {
    for (i, y): (u32, F32[L]) in u32:0..L {
        update(y, i, leaky_relu_f32_scalar<ALPHA_BITS>(x[i]))
    }(F32[L]:[float32::zero(false), ...])
}

// ============================= LEAKY RELU TESTS =============================
// alpha = 1/16 is exact and a multiply by a power of two is exact, so the
// negative-side results below are exact float32 values (no approximation).
#[test]
fn test_leaky_relu_f32_scalar() {
    let pz = f32c(u32:0x00000000);  // +0.0
    assert_eq(leaky_relu_f32_scalar(f32c(u32:0x40e00000)), f32c(u32:0x40e00000));  // 7 -> 7
    assert_eq(leaky_relu_f32_scalar(f32c(u32:0xc0800000)), f32c(u32:0xbe800000));  // -4 -> -0.25
    assert_eq(leaky_relu_f32_scalar(f32c(u32:0xbf800000)), f32c(u32:0xbd800000));  // -1 -> -0.0625
    assert_eq(leaky_relu_f32_scalar(pz), pz);  // +0 -> +0
    assert_eq(leaky_relu_f32_scalar(f32c(u32:0x80000000)), f32c(u32:0x80000000));  // -0 -> -0
}

#[test]
fn test_leaky_relu_f32() {
    let x = fv(u32[4]:[0xc0800000, 0x40000000, 0xbf800000, 0x00000000]);  // -4, 2, -1, 0
    let want = fv(u32[4]:[0xbe800000, 0x40000000, 0xbd800000, 0x00000000]);  // -0.25, 2, -0.0625, 0
    assert_eq(leaky_relu_f32<u32:4>(x), want)
}

// Non-default slope: alpha = 0.5 (0x3f000000) halves the negative side exactly.
#[test]
fn test_leaky_relu_f32_custom_alpha() {
    assert_eq(leaky_relu_f32_scalar<u32:0x3f000000>(f32c(u32:0xc0800000)), f32c(u32:0xc0000000));  // -4// ->
                                                                                                   // -2
    let x = fv(u32[2]:[0xc0800000, 0x40000000]);  // -4, 2
    let want = fv(u32[2]:[0xc0000000, 0x40000000]);  // -2, 2  (positive side unchanged)
    assert_eq(leaky_relu_f32<u32:2, u32:0x3f000000>(x), want)
}

// ============================= Sigmoid (FP32) =============================
// True sigmoid s(x) = 1/(1+e^-x) is transcendental; with no exp/div in the float
// stdlib we approximate the curve itself.  Sigmoid is odd-symmetric about
// (0, 0.5), so we fit an ODD polynomial around 0.5 and clamp to [0, 1]:
// s(x) ~= clamp( 0.5 + c1*x + c3*x^3 + c5*x^5 , 0, 1 )

fn sig_half() -> F32 { f32c(u32:0x3f000000) }  //  0.5

fn sig_one() -> F32 { f32c(u32:0x3f800000) }  //  1.0

fn sig_k1() -> F32 { f32c(u32:0x3e5e50f2) }  //  0.21710566

fn sig_k3() -> F32 { f32c(u32:0xbc002d51) }  // -0.00782330

fn sig_k5() -> F32 { f32c(u32:0x38f80b98) }  //  0.00011828

pub fn sigmoid_f32_scalar(x: F32) -> F32 {
    // we implement 0.5 + x*(k1 + x2*(k3 + x2*k5)), where x2 = x*x.
    let x2 = float32::mul(x, x);
    let p = float32::fma(x2, sig_k5(), sig_k3());  // k5*x2 + k3
    let p = float32::fma(x2, p, sig_k1());  // (k5*x2+k3)*x2 + k1
    let p = float32::fma(x, p, sig_half());  // x*p + 0.5
    // clamp to [0, 1]
    let p = if float32::gt_2(p, sig_one()) { sig_one() } else { p };
    if float32::lt_2(p, float32::zero(false)) { float32::zero(false) } else { p }
}

pub fn sigmoid_f32<L: u32>(x: F32[L]) -> F32[L] {
    for (i, o): (u32, F32[L]) in u32:0..L {
        update(o, i, sigmoid_f32_scalar(x[i]))
    }(F32[L]:[float32::zero(false), ...])
}

// ============================= SIGMOID TESTS =============================
// Exact, hand-verifiable point: x=0 -> every odd term vanishes -> exactly 0.5.
#[test]
fn test_sigmoid_f32_zero() { assert_eq(sigmoid_f32_scalar(float32::zero(false)), sig_half()) }

#[test]
fn test_sigmoid_f32_vec_zero() {
    let z = float32::zero(false);
    let x = F32[3]:[z, z, z];
    let want = F32[3]:[sig_half(), sig_half(), sig_half()];
    assert_eq(sigmoid_f32<u32:3>(x), want)
}

// |a - b| <= tol, avoiding abs (not exported): both signed gaps are within tol.
fn within(a: F32, b: F32, tol: F32) -> bool {
    float32::lte_2(float32::sub(a, b), tol) && float32::lte_2(float32::sub(b, a), tol)
}

// Non-trivial points.
#[test]
fn test_sigmoid_f32_accuracy() {
    let tol = f32c(u32:0x3d4ccccd);  // 0.05
    assert_eq(within(sigmoid_f32_scalar(f32c(u32:0x3f000000)), f32c(u32:0x3f1f597f), tol), true);  // s(0.5)=0.6224593
    assert_eq(within(sigmoid_f32_scalar(f32c(u32:0x3f800000)), f32c(u32:0x3f3b26a8), tol), true);  // s(1)=0.7310586
    assert_eq(within(sigmoid_f32_scalar(f32c(u32:0x40000000)), f32c(u32:0x3f617beb), tol), true);  // s(2)=0.8807971
    assert_eq(within(sigmoid_f32_scalar(f32c(u32:0x40400000)), f32c(u32:0x3f73dbe6), tol), true);  // s(3)=0.9525741
    assert_eq(within(sigmoid_f32_scalar(f32c(u32:0xbf800000)), f32c(u32:0x3e89b2b1), tol), true);  // s(-1)=0.2689414
    assert_eq(within(sigmoid_f32_scalar(f32c(u32:0xc0000000)), f32c(u32:0x3df420a9), tol), true);  // s(-2)=0.1192029
}

// Shape: the tails saturate to EXACTLY 0/1 (the clamp fires), and the curve is
// monotonically increasing across the transition.  Both are exact, so assert_eq.
#[test]
fn test_sigmoid_f32_saturates() {
    assert_eq(sigmoid_f32_scalar(f32c(u32:0x41000000)), sig_one());  // s(8)  -> 1.0 (clamped)
    assert_eq(sigmoid_f32_scalar(f32c(u32:0xc1000000)), float32::zero(false));  // s(-8) -> 0.0
                                                                                // (clamped)
    // increasing: s(-2) < s(0) < s(2)
    let sm2 = sigmoid_f32_scalar(f32c(u32:0xc0000000));
    let s0 = sigmoid_f32_scalar(float32::zero(false));
    let sp2 = sigmoid_f32_scalar(f32c(u32:0x40000000));
    assert_eq(float32::lt_2(sm2, s0) && float32::lt_2(s0, sp2), true);
}

// ============================= Softmax (FP32) =============================
// y_i = e^{x_i} / sum_j e^{x_j}.  With no exp/div in the float stdlib:
//   - exp is a range-reduced cubic:  exp(d) = 2^-k * 2^-f, with t = -d/ln2 >= 0,
//     k = floor(t), f = t - k in [0,1); 2^-f via a cubic fit, 2^-k via ldexp.
//   - the divide by the sum is a reciprocal:  1/S = fast_rsqrt(S)^2.

fn inv_ln2() -> F32 { f32c(u32:0x3fb8aa3b) }  //  1.44269504

fn cub_f3() -> F32 { f32c(u32:0xbd21d551) }  // -0.03951008

fn cub_f2() -> F32 { f32c(u32:0x3e6c20cd) }  //  0.23059388

fn cub_f1() -> F32 { f32c(u32:0xbf30ea60) }  // -0.69107628

fn cub_f0() -> F32 { f32c(u32:0x3f7ff95b) }  //  0.99989861

fn vmax<L: u32>(x: F32[L]) -> F32 {
    for (i, m): (u32, F32) in u32:1..L {
        if float32::gt_2(x[i], m) { x[i] } else { m }
    }(x[0])
}

fn vsum<L: u32>(e: F32[L]) -> F32 {
    for (i, s): (u32, F32) in u32:1..L {
        float32::add(s, e[i])
    }(e[0])
}

// 1/S = fast_rsqrt(S)^2  (S > 0, a sum of positive exps)
fn recip(s: F32) -> F32 {
    let r = float32::fast_rsqrt(s);
    float32::mul(r, r)
}

// exp(d) for d <= 0 via range-reduced cubic (see header)
fn exp_poly(d: F32) -> F32 {
    let t = float32::mul(float32::negate(d), inv_ln2());
    let kf = float32::floor_daz(t);
    let k = float32::to_int32(kf);
    let f = float32::sub(t, kf);
    let g = float32::fma(f, cub_f3(), cub_f2());
    let g = float32::fma(f, g, cub_f1());
    let g = float32::fma(f, g, cub_f0());
    float32::ldexp(g, -k)
}

pub fn softmax_f32<L: u32>(x: F32[L]) -> F32[L] {
    let m = vmax<L>(x);
    let e = for (i, e): (u32, F32[L]) in u32:0..L {
        update(e, i, exp_poly(float32::sub(x[i], m)))
    }(x);
    let r = recip(vsum<L>(e));
    for (i, o): (u32, F32[L]) in u32:0..L {
        update(o, i, float32::mul(e[i], r))
    }(e)
}

// ============================= SOFTMAX TESTS =============================
// We assert two invariants that hold by construction:
// 1. equal inputs -> equal outputs (permutation symmetry)
// 2. shift invariance: softmax(x) == softmax(x + c).

#[test]
fn test_softmax_f32_equal_inputs() {
    let c = f32c(u32:0x40400000);  // 3.0
    let x = F32[4]:[c, c, c, c];
    let y = softmax_f32<u32:4>(x);
    // all four outputs are bit-identical to each other
    assert_eq(y[0], y[1]);
    assert_eq(y[1], y[2]);
    assert_eq(y[2], y[3]);
}

#[test]
fn test_softmax_f32_shift_invariant() {
    let x = fv(u32[4]:[0x3f800000, 0x40000000, 0x40400000, 0x40800000]);  // 1, 2, 3, 4
    let x_shift = fv(u32[4]:[0x41300000, 0x41400000, 0x41500000, 0x41600000]);  // 11,12,13,14
    assert_eq(softmax_f32<u32:4>(x), softmax_f32<u32:4>(x_shift))
}

// More general accuracy test.
#[test]
fn test_softmax_f32_accuracy() {
    let tol = f32c(u32:0x3c23d70a);  // 0.01
    let x = fv(u32[4]:[0x3f800000, 0x40000000, 0x40400000, 0x40800000]);  // 1, 2, 3, 4
    let y = softmax_f32<u32:4>(x);
    assert_eq(within(y[0], f32c(u32:0x3d034fe2), tol), true);  // 0.0320586
    assert_eq(within(y[1], f32c(u32:0x3db278b8), tol), true);  // 0.0871443
    assert_eq(within(y[2], f32c(u32:0x3e729169), tol), true);  // 0.2368828
    assert_eq(within(y[3], f32c(u32:0x3f24d791), tol), true);  // 0.6439143
}
