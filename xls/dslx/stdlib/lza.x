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

// Leading Zero Anticipator (LZA)
// Based on Bruguera and Lang
// Leading-One Prediction Scheme for Latency Improvement in Single Datapath Floating-Point Adders
// https://ieeexplore.ieee.org/document/727065

import abs_diff;
import std;

struct BMEY<N: u32> { begin: uN[N], mid: uN[N], end: uN[N], yes: uN[N] }

// Merges (B, M, E, Y) labels as defined in the paper.
// The Y, or yes, indicates a correction pattern has been seen, which amounts to the following:
//   Case 1: 0s ++ 1 ++ 0s ++ -1 ++ ...
//   Case 2: 0s ++ 1 ++ -1s ++ 0s ++ -1 ++ ...
//
// Case 1 requires correction because 1 ++ 0s ++ -1 annihilates to 0 ++ 1s which shifts the
// leading one to the right by one index.
// Case 2 requires correction because 1 ++ -1s ++ 0s ++ -1 repeatedly annihilates until you're left
// with case 1, i.e. 1 ++ -1 becomes 0 ++ 1 over and over until you reduce to case 1.
fn merge_branch_labels<N: u32>(left: BMEY<N>, right: BMEY<N>) -> BMEY<N> {
    const_assert!(N > u32:0);
    // begin is unimpacted by mid (zeroes around it)
    let begin = (left.begin & right.mid) | (left.mid & right.begin);
    // mid is a sequence of 0s
    let mid = left.mid & right.mid;
    // end is the sequence of 0s ++ -1
    let end = left.end | (left.mid & right.end);

    // yes ++ any -> yes , mid ++ yes -> yes
    // begin ++ end -> yes, the full correction pattern has been seen.
    let yes = left.yes | (left.mid & right.yes) | (left.begin & right.end);

    BMEY<N> { begin, mid, end, yes }
}

fn tree_reduce<N: u32, STAGES: u32 = {std::clog2(N)}>(labels: BMEY<u32:1>[N]) -> BMEY<u32:1> {
    const_assert!(N > u32:0);
    const MID_LABEL = BMEY<u32:1> { begin: u1:0, mid: u1:1, end: u1:0, yes: u1:0 };

    let final_labels =
        for (stage, current_labels): (uN[N], BMEY<u32:1>[N]) in uN[N]:0..STAGES as uN[N] {
            // Used to merge a left and right node in the tree; the stride is the offset we add to
            // the left node's index in the array to get the right node's index in the array.
            //
            // For the first stage, stride == 1 means we merge labels[0] with labels[1], labels[2]
            // with labels[3], etc. For the second stage, stride == 2 means we merge
            // labels[0] with labels[2], labels[4] with labels[6], etc...
            // With each step, half the number of merges occur. For stride == 1, every other node
            // performs a merge into itself. For stride == 2, every 4th node does so, etc...
            let stride = uN[N]:1 << stage;
            for (i, next_labels): (uN[N], BMEY<u32:1>[N]) in uN[N]:0..N as uN[N] {
                let right_idx = i + stride;
                let right_label =
                    if right_idx < N as uN[N] { current_labels[right_idx] } else { MID_LABEL };

                // Double the stride is the index periodicity with which we merge labels. So when
                // stride == 1, we perform the merge operation every 2nd element; when stride == 2,
                // we merge every 4th element, etc...
                let merge_period = stride << uN[N]:1;
                // When unrolled, computing `is_active` should be optimized to a constant.
                let is_active = (i % merge_period) == uN[N]:0;
                let merged = merge_branch_labels(current_labels[i], right_label);
                if is_active { update(next_labels, i, merged) } else { next_labels }
            }(current_labels)
        }(labels);

    final_labels[0]
}

fn make_label<N: u32>(begin: uN[N], end: uN[N]) -> BMEY<N> {
    const_assert!(N > u32:0);
    BMEY<N> { begin, mid: !(begin | end), end, yes: uN[N]:0 }
}

// Predicts leading zero count for |a - b| and whether shift needs correction.
pub fn lza<N: u32, RESULT_BITS: u32 = {std::clog2(N + u32:1)}>
    (a: uN[N], b: uN[N]) -> (uN[RESULT_BITS], u1) {
    let MID_LABEL = BMEY<u32:1> { begin: u1:0, mid: u1:1, end: u1:0, yes: u1:0 };
    let init_pos_labels = BMEY<u32:1>[N]:[MID_LABEL, ...];
    let init_neg_labels = BMEY<u32:1>[N]:[MID_LABEL, ...];

    // Shifted 'a' and 'b' are used to examine bits at index bit_idx-1 and bit_idx+1 below
    let a_early = a >> uN[N]:1;
    let b_early = b >> uN[N]:1;
    let a_late = a << uN[N]:1;
    let b_late = b << uN[N]:1;

    let (indicator_vector, pos_labels, neg_labels) =
        for (k, (vec, p_labels, n_labels)): (uN[N], (uN[N], BMEY<u32:1>[N], BMEY<u32:1>[N])) in
            uN[N]:0..N as uN[N] {
            let bit_idx = N as uN[N] - uN[N]:1 - k;
            let a_left = a_early[bit_idx+:u1];
            let b_left = b_early[bit_idx+:u1];
            let a_center = a[bit_idx+:u1];
            let b_center = b[bit_idx+:u1];
            let a_right = a_late[bit_idx+:u1];
            let b_right = b_late[bit_idx+:u1];

            // a_i-1 == b_i-1 becomes label 0
            let left_e = !(a_left ^ b_left);
            // a_i > b_i becomes label 1
            let center_g = a_center & !b_center;
            // a_i < b_i becomes label -1
            let center_s = !a_center & b_center;
            // a_i+1 > b_i+1
            let right_g = a_right & !b_right;
            // a_i+1 < b_i+1
            let right_s = !a_right & b_right;

            // Computing F from the paper:
            // For example, ne_s_ns in terms of labels is [1|-1] ++ -1 ++ [0|1]
            let ne_s_ns = !left_e & center_s & !right_s;
            let e_g_ns = left_e & center_g & !right_s;
            let e_s_ng = left_e & center_s & !right_g;
            let ne_g_ng = !left_e & center_g & !right_g;
            let indicator_bit = (e_g_ns | ne_s_ns) | (e_s_ng | ne_g_ng);
            let next_vec = vec | ((indicator_bit as uN[N]) << bit_idx);

            // Computing detection trees in W > 0:
            let pos_begin = (center_g & !right_s) | ne_s_ns;
            let pos_end = left_e & center_s;
            let pos_label = make_label(pos_begin, pos_end);

            // Computing detection trees in W < 0:
            let neg_begin = (center_s & !right_g) | ne_g_ng;
            let neg_end = left_e & center_g;
            let neg_label = make_label(neg_begin, neg_end);

            (next_vec, update(p_labels, k, pos_label), update(n_labels, k, neg_label))
        }((uN[N]:0, init_pos_labels, init_neg_labels));

    let pred_clz = std::clzt(indicator_vector);
    let has_error = tree_reduce(pos_labels).yes | tree_reduce(neg_labels).yes;
    (pred_clz, has_error)
}

#[test]
fn test_lza() {
    let (shift, err) = lza(u8:1, u8:0);
    assert_eq(shift, u4:7);
    assert_eq(err, u1:0);

    let (shift, err) = lza(u8:4, u8:1);
    assert_eq(shift, u4:5);
    assert_eq(err, u1:1);
}

#[quickcheck(exhaustive)]
fn lza_u8_nonnegative(a: u8, b: u8) -> bool {
    let (pred_clz, has_error) = lza(a, b);
    let ab_absdiff = abs_diff::to_corrected(abs_diff::abs_diff(a, b));
    let actual_clz = std::clzt(ab_absdiff);

    let eq_or_one_less = (pred_clz == actual_clz) || (pred_clz + u4:1 == actual_clz);
    (has_error || (pred_clz == actual_clz)) && eq_or_one_less
}

#[quickcheck]
fn lza_u16_nonnegative(a: u16, b: u16) -> bool {
    let (pred_clz, has_error) = lza(a, b);
    let ab_absdiff = abs_diff::to_corrected(abs_diff::abs_diff(a, b));
    let actual_clz = std::clzt(ab_absdiff);

    let eq_or_one_less = (pred_clz == actual_clz) || (pred_clz + u5:1 == actual_clz);
    (has_error || (pred_clz == actual_clz)) && eq_or_one_less
}

fn tree_reduce_vectorized<N: u32, STAGES: u32 = {std::clog2(N)}>(labels: BMEY<N>) -> BMEY<u32:1> {
    const_assert!(N > u32:0);
    let final_labels = for (stage, current_labels): (uN[N], BMEY<N>) in uN[N]:0..STAGES as uN[N] {
        let stride = uN[N]:1 << stage;
        let right_labels = BMEY<N> {
            begin: current_labels.begin << stride,
            mid: current_labels.mid << stride,
            end: current_labels.end << stride,
            yes: current_labels.yes << stride,
        };
        merge_branch_labels(current_labels, right_labels)
    }(labels);

    BMEY<u32:1> {
        begin: final_labels.begin[-1:],
        mid: final_labels.mid[-1:],
        end: final_labels.end[-1:],
        yes: final_labels.yes[-1:],
    }
}

// The same implementation as above, but vectorized; this comes with an area footprint and so is
// not used by default; however, it is more readable and has been left here in case future
// optimizations make it viable.
pub fn lza_vectorized<N: u32, RESULT_BITS: u32 = {std::clog2(N + u32:1)}>
    (a: uN[N], b: uN[N]) -> (uN[RESULT_BITS], u1) {

    let a_early = a >> uN[N]:1;
    let b_early = b >> uN[N]:1;
    let a_late = a << uN[N]:1;
    let b_late = b << uN[N]:1;

    let left_e = !(a_early ^ b_early);
    let center_g = a & !b;
    let center_s = !a & b;
    let right_g = a_late & !b_late;
    let right_s = !a_late & b_late;

    let ne_s_ns = !left_e & center_s & !right_s;
    let e_g_ns = left_e & center_g & !right_s;
    let e_s_ng = left_e & center_s & !right_g;
    let ne_g_ng = !left_e & center_g & !right_g;
    let indicator_vector = (e_g_ns | ne_s_ns) | (e_s_ng | ne_g_ng);
    let pos_begin = (center_g & !right_s) | ne_s_ns;
    let pos_end = left_e & center_s;
    let neg_begin = (center_s & !right_g) | ne_g_ng;
    let neg_end = left_e & center_g;

    let pos_labels = make_label(pos_begin, pos_end);
    let neg_labels = make_label(neg_begin, neg_end);

    let pred_clz = std::clzt(indicator_vector);
    let has_error = tree_reduce_vectorized(pos_labels).yes | tree_reduce_vectorized(neg_labels).yes;
    (pred_clz, has_error)
}

#[quickcheck(exhaustive)]
fn lza_vectorized_u8_nonneg_equivalent_to_nonvectorized(a: u8, b: u8) -> bool {
    let (pred_clz, has_error) = lza_vectorized(a, b);
    let ab_absdiff = abs_diff::to_corrected(abs_diff::abs_diff(a, b));
    let actual_clz = std::clzt(ab_absdiff);

    let eq_or_one_less = (pred_clz == actual_clz) || (pred_clz + u4:1 == actual_clz);
    let (pred_clz_nonvec, has_error_nonvec) = lza(a, b);
    let both_impls_match = pred_clz == pred_clz_nonvec && has_error == has_error_nonvec;
    (has_error || (pred_clz == actual_clz)) && eq_or_one_less && both_impls_match
}

#[quickcheck]
fn lza_vectorized_u16_nonneg_equivalent_to_nonvectorized(a: u16, b: u16) -> bool {
    let (pred_clz, has_error) = lza_vectorized(a, b);
    let ab_absdiff = abs_diff::to_corrected(abs_diff::abs_diff(a, b));
    let actual_clz = std::clzt(ab_absdiff);

    let eq_or_one_less = (pred_clz == actual_clz) || (pred_clz + u5:1 == actual_clz);
    let (pred_clz_nonvec, has_error_nonvec) = lza(a, b);
    let both_impls_match = pred_clz == pred_clz_nonvec && has_error == has_error_nonvec;
    (has_error || (pred_clz == actual_clz)) && eq_or_one_less && both_impls_match
}
