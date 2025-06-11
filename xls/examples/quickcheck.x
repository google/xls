#![feature(type_inference_v2)]

// Copyright 2022 The XLS Authors
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

// Examples demonstrating quickcheck.

// Reversing a value twice gets you the original value.
#[quickcheck(test_count=50000)]
fn prop_double_reverse(x: u32) -> bool {
  x == rev(rev(x))
}

// From https://github.com/google/xls/issues/84
// Array concatenation is not commutative.
#[quickcheck]
fn prop_array_comparison_commutative(xs: u32[2], ys: u32[2]) -> bool {
  (xs == ys) || ((xs ++ ys) != (ys ++ xs))
}

// x + y != x unless y is zero.
#[quickcheck]
fn prop_addition_not_equal_unless_identity(x: u32, y: u32) -> bool {
  y == u32:0 || x + y != x
}
