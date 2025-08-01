#![feature(type_inference_v1)]
#![feature(use_syntax)]

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

use std::{clog2, is_pow2};

fn p<N: u32, OUT: u32 = {N + clog2(N)}>(x: uN[N]) -> uN[OUT] {
    const_assert!(is_pow2(N));
    x as uN[OUT]
}

fn main() -> u37 { p(u32:42) }

#[test]
fn test_main() { assert_eq(main(), u37:42); }
