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

// mac.x -- combinational multiply-accumulate via a balanced adder tree (float32).
//
// Computes a length-N dot product  y = sum_i a[i]*b[i]  as a purely
// combinational block: N parallel multipliers feeding a balanced binary
// adder tree.  The tree has ceil(log2 N) add-depth rather than the N-1 depth
// of a linear accumulation chain.
//
// Everything adheres to IEEE-754 float32 (`float32::F32`).

import float32;
import std;

type F32 = float32::F32;

// bit-pattern to F32 helper (constants have no float literal syntax in DSLX).
fn f32c(raw: u32) -> F32 { float32::unflatten(raw) }

// Balanced float adder tree: reduce N summands to one in ceil(log2 N) levels.
// Each level pairs neighbours and compacts the survivors to the front of the
// working array.  An odd survivor at a level is carried by adding +0.0 (exact
// for any finite/inf operand), so N need not be a power of two.
pub fn tree_sum<N: u32>(x: F32[N]) -> F32 {
    let (reduced, _) = for (_, st): (u32, (F32[N], u32)) in u32:0..std::clog2(N) {
        let (cur, active) = st;
        let next = (active + u32:1) >> u32:1;  // survivors after this level
        let folded = for (i, c): (u32, F32[N]) in u32:0..((N + u32:1) >> u32:1) {
            if i < next {
                let lo = cur[i * u32:2];
                // pull the partner, or +0.0 when this is an odd leftover
                let hi = if (i * u32:2 + u32:1) < active {
                    cur[i * u32:2 + u32:1]
                } else {
                    float32::zero(false)
                };
                update(c, i, float32::add(lo, hi))
            } else {
                c
            }
        }(cur);
        (folded, next)
    }((x, N));
    reduced[0]
}

// Dot product y = sum_i a[i]*b[i] via N parallel multipliers + the adder tree.
pub fn mac_f32<N: u32>(a: F32[N], b: F32[N]) -> F32 {
    let prod = for (i, p): (u32, F32[N]) in u32:0..N {
        update(p, i, float32::mul(a[i], b[i]))
    }(zero!<F32[N]>());
    tree_sum<N>(prod)
}

// ---- tests -------------------------------------------------------------

fn fv<L: u32>(raw: u32[L]) -> F32[L] {
    for (i, v): (u32, F32[L]) in u32:0..L {
        update(v, i, f32c(raw[i]))
    }(zero!<F32[L]>())
}

#[test]
fn test_tree_sum_zero() {
    let x = F32[4]:[float32::zero(false), ...];
    assert_eq(tree_sum<u32:4>(x), float32::zero(false));
}

#[test]
fn test_tree_sum_pow2() {
    // 1 + 2 + 3 + 4 = 10
    let x = fv(u32[4]:[0x3f800000, 0x40000000, 0x40400000, 0x40800000]);
    assert_eq(tree_sum<u32:4>(x), f32c(u32:0x41200000));  // 10.0
}

#[test]
fn test_tree_sum_odd_len() {
    // non-power-of-two length exercises the +0.0 leftover carry: 1 + 2 + 3 = 6
    let x = fv(u32[3]:[0x3f800000, 0x40000000, 0x40400000]);
    assert_eq(tree_sum<u32:3>(x), f32c(u32:0x40c00000));  // 6.0
}

#[test]
fn test_tree_sum_len1() {
    // clog2(1) = 0 levels -> passthrough of the single element
    let x = fv(u32[1]:[0x40490fdb]);  // ~3.14159
    assert_eq(tree_sum<u32:1>(x), f32c(u32:0x40490fdb));
}

#[test]
fn test_mac_f32_ones() {
    // dot([1,2,3,4], [1,1,1,1]) = 10
    let a = fv(u32[4]:[0x3f800000, 0x40000000, 0x40400000, 0x40800000]);
    let b = F32[4]:[f32c(u32:0x3f800000), ...];
    assert_eq(mac_f32<u32:4>(a, b), f32c(u32:0x41200000));  // 10.0
}

#[test]
fn test_mac_f32_dot3() {
    // dot([1,2,3], [4,5,6]) = 4 + 10 + 18 = 32
    let a = fv(u32[3]:[0x3f800000, 0x40000000, 0x40400000]);
    let b = fv(u32[3]:[0x40800000, 0x40a00000, 0x40c00000]);
    assert_eq(mac_f32<u32:3>(a, b), f32c(u32:0x42000000));  // 32.0
}

#[test]
fn test_mac_f32_8() {
    // dot(1..8, all-ones) = 36
    let a = fv(
        u32[8]:[
            0x3f800000, 0x40000000, 0x40400000, 0x40800000, 0x40a00000, 0x40c00000, 0x40e00000,
            0x41000000,
        ]);
    let b = F32[8]:[f32c(u32:0x3f800000), ...];
    assert_eq(mac_f32<u32:8>(a, b), f32c(u32:0x42100000));  // 36.0
}

#[test]
fn test_mac_f32_fractional() {
    let a = fv(u32[4]:[0x3dcccccd, 0x3e4ccccd, 0x3e99999a, 0x3ecccccd]);  // .1 .2 .3 .4
    let b = fv(u32[4]:[0x3f000000, 0x3e800000, 0x3e000000, 0x3d800000]);  // .5 .25 .125 .0625
    assert_eq(mac_f32<u32:4>(a, b), f32c(u32:0x3e266666));  // 0.1625
}
