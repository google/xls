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

// Identity function helper.
fn id<N: u32>(x: bits[N]) -> bits[N] { x }

// Slice wrapper to test parametric widths.
fn get_middle_bits<N: u32, R: u32 = {N - u32:2}>(x: bits[N]) -> bits[R] {
    // Slice middle bits out of parametric width.
    x[1:-1]
}

fn foo() -> u6 { get_middle_bits(u8:5) }

#[test]
fn bit_slice_syntax() {
    assert_eq(u2:0b11, get_middle_bits(u4:0b0110));
    assert_eq(u3:0b101, get_middle_bits(u5:0b01010));

    let x = u6:0b100111;
    // Slice out two bits.
    assert_eq(u2:0b11, x[0:2]);
    assert_eq(u2:0b11, x[1:3]);
    assert_eq(u2:0b01, x[2:4]);
    assert_eq(u2:0b00, x[3:5]);

    // Slice out three bits.
    assert_eq(u3:0b111, x[0:3]);
    assert_eq(u3:0b011, x[1:4]);
    assert_eq(u3:0b001, x[2:5]);
    assert_eq(u3:0b100, x[3:6]);

    // Slice out from the end.
    assert_eq(u1:0b1, x[-1:]);
    assert_eq(u1:0b1, x[-1:6]);
    assert_eq(u2:0b10, x[-2:]);
    assert_eq(u2:0b10, x[-2:6]);
    assert_eq(u3:0b100, x[-3:]);
    assert_eq(u3:0b100, x[-3:6]);
    assert_eq(u4:0b1001, x[-4:]);
    assert_eq(u4:0b1001, x[-4:6]);

    // Slice both relative to the end (MSb).
    assert_eq(u2:0b01, x[-4:-2]);
    assert_eq(u2:0b11, x[-6:-4]);

    // Slice out from the beginning (LSb).
    assert_eq(u5:0b00111, x[:-1]);
    assert_eq(u4:0b0111, x[:-2]);
    assert_eq(u3:0b111, x[:-3]);
    assert_eq(u2:0b11, x[:-4]);
    assert_eq(u1:0b1, x[:-5]);

    // Slicing past the end just means we hit the end (as in Python).
    assert_eq(u1:0b1, x[5:7]);
    assert_eq(u1:0b1, x[-7:1]);
    assert_eq(bits[0]:0, x[-7:-6]);
    assert_eq(bits[0]:0, x[-6:-6]);
    assert_eq(bits[0]:0, x[6:6]);
    assert_eq(bits[0]:0, x[6:7]);
    assert_eq(u1:1, x[-6:-5]);

    // Slice of a slice.
    assert_eq(u2:0b11, x[:4][1:3]);

    // Slice of an invocation.
    assert_eq(u2:0b01, id(x)[2:4]);

    // Explicit-width slices.
    assert_eq(u2:0b01, x[2+:u2]);
    assert_eq(s3:0b100, x[3+:s3]);
}
