// Copyright 2020 The XLS Authors
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

enum MyEnum : u2 {
    A = 0,
    B = 1,
    C = 2,
    D = 3,
}

#[test]
fn enum_values() {
    let a: MyEnum = MyEnum::A;
    // Cast some to unsigned.
    let b_u2: u2 = MyEnum::B as u2;
    let c_u2: u2 = MyEnum::C as u2;
    // Cast one to signed and sign extend it.
    let d_s2: s2 = MyEnum::D as s2;
    let d_signext: s3 = d_s2 as s3;
    assert_eq(d_signext, s3:0b111);

    // Extend values to u3 and sum them up.
    let sum = (a as u2 as u3) + (b_u2 as u3) + (c_u2 as u3) + (d_s2 as u2 as u3);
    assert_eq(sum, u3:6);

    // A bunch of equality/comparison checks.
    assert_eq(a, MyEnum::A);
    assert_eq(true, a == MyEnum::A);
    assert_eq(false, a != MyEnum::A);
    assert_eq(a, a as u2 as MyEnum);
    assert_eq(a, a as u2 as MyEnum);
}

#[test]
fn enum_values_widen_from_unsigned() {
    let d_s4: s4 = MyEnum::D as s4;
    assert_eq(s4:0b0011, d_s4)
}

#[test]
fn enum_values_narrow_from_unsigned() {
    let d_s1: s1 = MyEnum::D as s1;
    assert_eq(s1:0b1, d_s1)
}

enum MyEnumSigned : s2 {
    A = 0,
    B = 1,
    C = -2,
    D = -1,
}

#[test]
fn enum_values_widen_from_signed() {
    let d_s4: s4 = MyEnumSigned::D as s4;
    assert_eq(s4:0b1111, d_s4)
}

#[test]
fn enum_values_narrow_from_signed() {
    let d_s1: s1 = MyEnumSigned::D as s1;
    assert_eq(s1:0b1, d_s1)
}
