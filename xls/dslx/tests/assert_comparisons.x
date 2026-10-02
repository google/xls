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

// Tests comparison assertions through the DSLX interpreter, IR conversion, and
// the JIT.
//
// Put assertions in `main` so they reach IR. Use `x` to prevent constant
// folding.

// `x` is a `u8`, so `unsigned_x` is in [0, 255] and `signed_x` is in
// [-128, 127]. Every assertion below holds for all inputs.
fn main(x: u8) -> u32 {
    let unsigned_x = x as u32;
    let signed_x = x as s32 - s32:128;

    // Unsigned operands compare with the unsigned IR ops (`ult`, `ule`, ...).
    assert_lt(unsigned_x, u32:256);
    assert_le(unsigned_x, u32:255);
    assert_gt(u32:256, unsigned_x);
    assert_ge(u32:255, unsigned_x);

    // These fail if signed values are compared as unsigned.
    assert_lt(signed_x, s32:128);
    assert_le(signed_x, s32:127);
    assert_gt(s32:128, signed_x);
    assert_ge(s32:127, signed_x);

    // `assert_ne` accepts any type `T`, not just bits.
    assert_ne(unsigned_x, u32:256);
    assert_ne((unsigned_x, u8:2), (unsigned_x, u8:3));

    unsigned_x
}

#[test]
fn test_main() {
    assert_eq(main(u8:0), u32:0);
    assert_eq(main(u8:255), u32:255);
}
