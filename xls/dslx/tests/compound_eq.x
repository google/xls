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

// The leaf values have a tiny range so that 1000 tests can find a
// counterexample if the equality check is broken.
type TestBlob = (s2[2], (u2, u1), bool);

fn main() -> bool {
    let x = ([s2:0, s2:3], (u2:1, u1:0), true);
    x == x
}

fn slice_off_array(x: TestBlob) -> ((u2, u1), bool) { (x.1, x.2) }

// Manually expand a test blob into its leaf components to check equality.
fn blob_eq(x: TestBlob, y: TestBlob) -> bool {
    match (slice_off_array(x), slice_off_array(y)) {
        (((x_tup1, x_tup2), x_bool), ((y_tup1, y_tup2), y_bool)) =>
            x_tup1 == y_tup1 && x_tup2 == y_tup2 && x_bool == y_bool,
    }
}

// Check equality of TestBlob arrays element-by-element two different ways:
// 1. Relying on built-in support for tuple equality.
// 2. Manually expanding the TestBlob and checking the leaf values directly.
fn eq_by_element(x: TestBlob[3], y: TestBlob[3]) -> bool {
    for (i, eq): (u32, bool) in u32:0..u32:3 {
        eq && x[i] == y[i] && blob_eq(y[i], x[i])
    }(true)
}

// Manually expand a test blob into its leaf components to check if any are
// not equal.
fn blob_neq(x: TestBlob, y: TestBlob) -> bool {
    match (slice_off_array(x), slice_off_array(y)) {
        (((x_tup1, x_tup2), x_bool), ((y_tup1, y_tup2), y_bool)) =>
            x_tup1 != y_tup1 || x_tup2 != y_tup2 || x_bool != y_bool,
    }
}

// Check inequality of TestBlob arrays element-by-element two different ways:
// 1. Relying on built-in support for tuple inequality.
// 2. Manually expanding the TestBlob and checking the leaf values directly.
fn neq_by_element(x: TestBlob[3], y: TestBlob[3]) -> bool {
    for (i, neq): (u32, bool) in u32:0..u32:3 {
        neq || x[i] != y[i] || blob_neq(y[i], x[i])
    }(false)
}

// The default 1000 tests aren't enough to generate a counterexample
// when eq_by_element has a bug (like not checking every element).
#[quickcheck(test_count=100000)]
fn prop_consistent_eq(x: TestBlob[3], y: TestBlob[3]) -> bool {
    x == x && eq_by_element(x, x) && y == y && eq_by_element(y, y) &&
    (x == y) == eq_by_element(x, y)
}

// Use the same test count as prop_consistent_eq because while the edge cases
// may be different, the difficulty in finding them should be similar.
#[quickcheck(test_count=100000)]
fn prop_consistent_neq(x: TestBlob[3], y: TestBlob[3]) -> bool {
    !(x != x) && !(neq_by_element(x, x)) && !(y != y) && !(neq_by_element(y, y)) &&
    (x != y) == neq_by_element(x, y)
}

#[test]
fn empty_eq_test() {
    assert_eq(() == (), true);
    assert_eq(() != (), false);

    // The following, more natural, definitions require unifying type inference.
    // TODO(amfv): 2021-05-26 Switch to them once have it.
    //  let a: u32[0] = [];
    //  let b: u32[0] = [];
    let a = u32[0]:[];
    let b = u32[0]:[];
    assert_eq(a == b, true);
    assert_eq(a != b, false);
}
