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

// Duck typing examples using impl providing methods with the same name.
// Allows to implement generic operations even without using traits.

#![feature(generics)]

import apfloat;
import float32;
import float64;

// A number wrapping a generic apfloat.
struct FloatNum<EXP_SIZE: u32, FRACTION_SIZE: u32> { n: apfloat::APFloat<EXP_SIZE, FRACTION_SIZE> }

// Two useful types for floating point numbers
type Float32Num = FloatNum<float32::F32_EXP_SZ, float32::F32_FRACTION_SZ>;
type Float64Num = FloatNum<float64::F64_EXP_SZ, float64::F64_FRACTION_SZ>;

impl FloatNum<EXP_SIZE, FRACTION_SIZE> {
    fn default() -> Self { FloatNum { n: apfloat::zero<EXP_SIZE, FRACTION_SIZE>(0) } }

    fn from(value: apfloat::APFloat<EXP_SIZE, FRACTION_SIZE>) -> Self { FloatNum { n: value } }

    fn add(self, b: FloatNum<EXP_SIZE, FRACTION_SIZE>) -> Self {
        FloatNum { n: apfloat::add<EXP_SIZE, FRACTION_SIZE>(self.n, b.n) }
    }

    fn is_zero(self) -> u1 { apfloat::is_zero_or_subnormal<EXP_SIZE, FRACTION_SIZE>(self.n) }
}

struct IntNum<T: type> { n: T }

type sInt32 = IntNum<s32>;
type sInt64 = IntNum<s64>;

impl IntNum<T> {
    fn default() -> Self { IntNum { n: 0 } }

    fn from(value: T) -> Self { IntNum { n: value } }

    fn add(self, b: IntNum<T>) -> Self { IntNum { n: self.n + b.n } }

    fn is_zero(self) -> u1 { self.n == 0 }
}

// A generic function adding two objects together that happen to have an
// implementation that provides an add() method. If it quacks like a duck...
fn generic_add<T: type>(a: T, b: T) -> T { a.add(b) }

#[test]
fn float32_add_test() {
    let one = Float32Num::from(float32::one(0));
    let zero = Float32Num::default();
    let result = generic_add(one, zero);
    assert_eq(result, one);
}

#[test]
fn float64_add_test() {
    let one = Float64Num::from(float64::one(0));
    let minus_one = Float64Num::from(float64::one(1));
    let result = generic_add(one, minus_one);
    assert_eq(result.is_zero(), 1);
}

#[test]
fn int_add_test() {
    let a = sInt32::from(42);
    let b = sInt32::default();
    let result = generic_add(a, b);
    assert_eq(result, sInt32::from(42));
}
