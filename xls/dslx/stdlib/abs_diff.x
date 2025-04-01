// Copyright 2025 The XLS Authors
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

//! Functionality for computing an absolute difference between two numbers with a
//! "residual" correction term. By keeping a correction term we avoid eagerly
//! incorporating a second carry chain in computing the final result.
//!
//! Sample usage:
//!
//! ```dslx
//! let x: u4 = u4:0;
//! let y: u4 = u4:1;
//! let adr: AbsDiffResult<4> = abs_diff(x, y);
//! assert_eq(to_corrected(adr), u4:1);
//! assert_eq(is_x_larger(adr), false);
//! assert_eq(is_zero(adr), false);
//! ```

import std;

/// Describes the result of an absolute difference operation.
///
/// By keeping the result in two terms with one being a correction term, we
/// can avoid a second carry chain being applied in the operation.
///
/// Additionally, this information can be used to tell us which of the two operands
/// was larger -- use `is_x_larger` for this.
pub struct AbsDiffResult<N: u32> {
    correction: bool,
    /// Note: this is the *uncorrected* absolute difference -- the `correction` term
    /// must be added to get the true absolute difference -- use `to_corrected` to
    /// flatten this structure to the absolute difference value.
    uncorrected: uN[N],
}

/// Returns `true` if the first operand was larger, `false` if the second operand was larger.
pub fn is_x_larger<N: u32>(adr: AbsDiffResult<N>) -> bool { adr.correction }

/// Collapses the `AbsDiffResult` structure into a single absolute difference value.
///
/// (Note that this incurs the second carry chain that we avoided by keeping the
/// result in two terms.)
pub fn to_corrected<N: u32>(adr: AbsDiffResult<N>) -> uN[N] {
    adr.uncorrected + adr.correction as uN[N]
}

/// Returns `true` if the absolute difference is zero, `false` otherwise.
///
/// This does not require a carry chain, so it is faster than `to_corrected`.
///
/// Note: the absolute diff result `| x - y |` is only ever zero when `x == y`.
pub fn is_zero<N: u32>(adr: AbsDiffResult<N>) -> bool {
    adr.correction == false && adr.uncorrected == uN[N]:0
}

/// Computes the absolute difference between two values.
///
/// This is a "residual" absolute difference operation -- the result is a structure
/// with a `correction` term and an `uncorrected` term.
///
/// Use `to_corrected` to collapse the result into a single absolute difference value, but
/// note that this incurs a second carry chain.
///
/// Implementation note: this is a technique also used in literature that optimizes
/// absolute difference calculations, e.g. for motion estimation, such that you can accumulate
/// correction terms over many calculations without needing carries at every step. See e.g.
/// Jehng et al. 10.1109/78.193224 fig 11(a) for a digram exhibiting the boolean formula.
///
/// Proof derivation steps given here thanks to ericastor@:
/// ```
/// result = z ^ signex(!carry_out)
///        = if carry_out { z } else { !z }  // XOR with signex == controlled-not
///        = if y < x { z } else { !z }  // because carry_out = (y < x)
///        = if y < x { x + !y } else { !(x + !y) }  // definition of z
///        = if y < x { x - y - 1 } else { !(x - y - 1) }  // x - y = x + !y + 1
///        = if y < x { x - y - 1 } else { -(x - y - 1) - 1 }  // -q = !q + 1
///        = if y < x { x - y - 1 } else { y - x }  // algebraic simplification
/// ```
pub fn abs_diff<N: u32>(x: uN[N], y: uN[N]) -> AbsDiffResult<N> {
    let ynot = !y;
    let (carry_out, z): (bool, uN[N]) = std::uadd_with_overflow<N>(x, ynot);
    let nc: bool = !carry_out;
    // Turn the `nc` bit into a mask via the `signex` builtin which replicates the msb.
    let xor_mask: uN[N] = signex(nc, zero!<uN[N]>());
    let result = (z ^ xor_mask);
    AbsDiffResult { correction: carry_out, uncorrected: result }
}

// -- tests and test helpers

/// "Golden reference model" for us to compare to for absolute difference calculation's
/// corrected result.
fn grm<N: u32>(x: uN[N], y: uN[N]) -> uN[N] { if x > y { x - y } else { y - x } }

fn do_test_diff_result<N: u32>
    (x: uN[N], y: uN[N], expected_correction: bool, expected_uncorrected: uN[N])
    {
    let result: AbsDiffResult<N> = abs_diff(x, y);
    assert_eq(
        result, AbsDiffResult { correction: expected_correction, uncorrected: expected_uncorrected });
    assert_eq(to_corrected(result), grm(x, y));

    // Also check the "x is larger" predicate
    assert_eq(is_x_larger(result), (x > y));
}

#[test]
fn test_abs_diff_0_vs_1() {
    do_test_diff_result(u8:1, u8:0, true, u8:0);
    do_test_diff_result(u8:0, u8:1, false, u8:1);
}

#[test]
fn test_abs_diff_0_vs_2() {
    do_test_diff_result(u8:2, u8:0, true, u8:1);
    do_test_diff_result(u8:0, u8:2, false, u8:2);
}

#[test]
fn test_abs_diff_1_vs_2() {
    do_test_diff_result(u8:2, u8:1, true, u8:0);
    do_test_diff_result(u8:1, u8:2, false, u8:1);
}

#[test]
fn test_abs_diff_1_vs_3() {
    do_test_diff_result(u8:3, u8:1, true, u8:1);
    do_test_diff_result(u8:1, u8:3, false, u8:2);
}

#[test]
fn test_abs_diff_corners() {
    do_test_diff_result(u8:0, u8:0, false, u8:0);
    do_test_diff_result(u8:255, u8:255, false, u8:0);
}

#[quickcheck(exhaustive)]
fn prop_x_larger_means_correction(x: u4, y: u4) -> bool {
    if x < y {
        !is_x_larger(abs_diff(x, y))
    } else if y > x {
        is_x_larger(abs_diff(x, y))
    } else {
        true
    }
}

#[quickcheck(exhaustive)]
fn prop_is_zero_predicate_matches_after_correction(x: u4, y: u4) -> bool {
    let adr = abs_diff(x, y);
    is_zero(adr) == (to_corrected(adr) == u4:0)
}

#[quickcheck(exhaustive)]
fn prop_zero_on_lhs_is_rhs_when_corrected(x: u4) -> bool {
    let adr = abs_diff(u4:0, x);
    to_corrected(adr) == x
}

#[quickcheck(exhaustive)]
fn prop_zero_on_lhs_means_y_larger(x: u4) -> bool {
    let adr = abs_diff(u4:0, x);

    // Zero is never *larger* than the right hand side.
    // (It is equal in the case of zero but that is not *larger*.)
    is_x_larger(adr) == false
}

#[quickcheck(exhaustive)]
fn prop_zero_on_rhs_is_lhs_when_corrected(x: u4) -> bool {
    let adr = abs_diff(x, u4:0);
    to_corrected(adr) == x
}

// Note the asymmetry here vs `prop_zero_on_lhs_means_y_larger` -- when they are both zero
// we get `false` for `is_x_larger`.
#[quickcheck(exhaustive)]
fn prop_zero_on_rhs_means_x_larger(x: u4) -> bool {
    let adr = abs_diff(x, u4:0);
    // x is larger when it is non-zero. When it is zero it is equal, therefore /not/ larger.
    let expected_x_larger = x != u4:0;
    is_x_larger(adr) == expected_x_larger
}
