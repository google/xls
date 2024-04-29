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
    let x: u32 = all_ones!<u32>();
    let y: MyU32 = all_ones!<MyU32>();
    x + y
}

#[test]
fn test_scalar_bits_type() { assert_eq(scalar_bits_type(), u32:0xFFFFFFFE) }

// -- enum with zero

enum EnumWithZero : u2 {
    ZERO = 0,
    ONE = 1,
    THREE = 3,
}

fn make_enum_with_zero() -> EnumWithZero { all_ones!<EnumWithZero>() }

#[test]
fn test_enum_with_zero() { assert_eq(make_enum_with_zero(), EnumWithZero::THREE) }

// -- array of bits type

fn array_of_bits_type() -> u32[4] { all_ones!<u32[4]>() }

#[test]
fn test_array_of_bits_type() { assert_eq(array_of_bits_type(), u32[4]:[0xFFFFFFFF, ...]) }

// -- tuple of bits types

fn tuple_of_bits_types() -> (u1, u2, u3) { all_ones!<(u1, u2, u3)>() }

#[test]
fn test_tuple_of_bits_types() { assert_eq(tuple_of_bits_types(), (u1:1, u2:3, u3:7)) }

// -- struct of bits types

struct MyPoint { x: u32, y: u32 }

fn make_struct() -> MyPoint { all_ones!<MyPoint>() }

#[test]
fn test_struct_of_bits_types() {
    assert_eq(make_struct(), MyPoint { x: u32:0xFFFFFFFF, y: u32:0xFFFFFFFF })
}

// -- parametric struct containing another struct

struct ParametricStruct<N: u32> { p: MyPoint, u: bits[N] }

// TODO: google/xls#984 - Make it so we can use this parametric instantiation
// syntax directly within the `all_ones!<>` macro (without a typedef). This
// requires extending the grammar.
type PS8 = ParametricStruct<8>;

fn main() -> ParametricStruct<8> { all_ones!<PS8>() }

#[test]
fn test_parametric_struct() {
    assert_eq(
        main(), ParametricStruct { p: MyPoint { x: u32:0xFFFFFFFF, y: u32:0xFFFFFFFF }, u: u8:255 })
}
