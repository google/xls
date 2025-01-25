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

// This sample shows use of a global constant definition in a local type alias.

const N = u32:7;

fn main() -> (uN[N], uN[N], uN[N]) {
    type T = uN[N];
    (T::MIN, T::ZERO, T::MAX)
}

#[test]
fn test_f() {
    let (min, zero, max) = main();
    assert_eq(min, u7:0);
    assert_eq(zero, u7:0);
    assert_eq(max, u7:127);
}
