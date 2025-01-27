// Copyright 2023 The XLS Authors
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

// -- scalar via type alias

type MyU32 = u32;

fn scalar_bits_type() -> u32 {
    let x: u32 = zero!<u32>();
    let y: MyU32 = zero!<MyU32>();
    x + y
}

#[test]
fn test_scalar_bits_type() { assert_eq(scalar_bits_type(), u32:0) }

// -- enum with zero

enum EnumWithZero : u2 {
    ZERO = 0,
    ONE = 1,
}

fn make_enum_with_zero() -> EnumWithZero { zero!<EnumWithZero>() }

#[test]
fn test_enum_with_zero() { assert_eq(make_enum_with_zero(), EnumWithZero::ZERO) }

// -- array of bits type

fn array_of_bits_type() -> u32[4] { zero!<u32[4]>() }

#[test]
fn test_array_of_bits_type() { assert_eq(array_of_bits_type(), u32[4]:[0, ...]) }

// -- tuple of bits types

fn tuple_of_bits_types() -> (u1, u2, u3) { zero!<(u1, u2, u3)>() }

#[test]
fn test_tuple_of_bits_types() { assert_eq(tuple_of_bits_types(), (u1:0, u2:0, u3:0)) }

// -- struct of bits types

struct MyPoint { x: u32, y: u32 }

fn make_struct() -> MyPoint { zero!<MyPoint>() }

#[test]
fn test_struct_of_bits_types() { assert_eq(make_struct(), MyPoint { x: u32:0, y: u32:0 }) }

// -- array of bits constructor based types

fn p<S: bool, N: u32, COUNT: u32>() -> xN[S][N][COUNT] { zero!<xN[S][N][COUNT]>() }

#[test]
fn test_array_of_bits_constructor_based_types() {
    assert_eq(p<true, u32:3, u32:4>(), s3[4]:[0, ...])
}

// -- parametric struct containing another struct

struct ParametricStruct<N: u32> { p: MyPoint, u: bits[N] }

// TODO(leary): 2023-03-01 Make it so we can use this parametric instantiation
// syntax directly within the `zero!<>` macro (without a typedef). This
// requires extending the grammar.
type PS8 = ParametricStruct<8>;

fn ps8() -> ParametricStruct<8> { zero!<PS8>() }

#[test]
fn test_parametric_struct() {
    assert_eq(ps8(), ParametricStruct { p: MyPoint { x: u32:0, y: u32:0 }, u: u8:0 })
}

// -- main function for IR emission

fn main() -> (PS8, s3[4]) { (ps8(), p<true, u32:3, u32:4>()) }

#[test]
fn test_main() {
    assert_eq(main(), (PS8 { p: MyPoint { x: u32:0, y: u32:0 }, u: u8:0 }, s3[4]:[0, ...]))
}
