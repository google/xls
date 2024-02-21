// Copyright 2024 The XLS Authors
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

// Regression test for https://github.com/google/xls/issues/1197
//
// Note that it has no tests, but should not fail in typechecking.
//
// The issue demonstrated that adding a test could perturb the issue that was
// seen.

import std;

fn foo<WIDTH: u32, LOG_WIDTH: u32 = {std::clog2(WIDTH)}>(a: uN[WIDTH]) -> uN[LOG_WIDTH] {
    uN[LOG_WIDTH]:0
}

fn bar<A_WIDTH: u32>(a: uN[A_WIDTH]) -> u32 { foo(a) as u32 }

fn main() -> u32 { bar(u32:0) }
