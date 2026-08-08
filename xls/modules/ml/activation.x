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
// * relu_f32_scalar -- scalar ReLU activation
// * relu_f32        -- elementwise ReLU activation
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
