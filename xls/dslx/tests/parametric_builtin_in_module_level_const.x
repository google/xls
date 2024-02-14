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

const U8_0 = u8:0;
const U8_1 = u8:1;
const U8_128 = u8:128;

// Here we apply the (parametric builtin) `clz()` to module-level constant
// values to produce more module-level constant values.
const U8_0_CLZ = clz(U8_0);
const U8_1_CLZ = clz(U8_1);
const U8_128_CLZ = clz(U8_128);

// Sample entry point that uses the constants that we can convert to IR.
fn main() -> u8 { U8_0_CLZ + U8_1_CLZ + U8_128_CLZ }

#[test]
fn test_consts() {
    assert_eq(U8_0_CLZ, u8:8);
    assert_eq(U8_1_CLZ, u8:7);
    assert_eq(U8_128_CLZ, u8:0);
}
