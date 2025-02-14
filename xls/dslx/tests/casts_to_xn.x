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

#[test]
fn test_casts() {
    assert_eq(s1:-1, u1:1 as xN[true][1]);
    assert_eq(u1:1, s1:-1 as xN[false][1]);
}

#[test]
fn test_widening_casts() {
    assert_eq(s2:0b11, s1:-1 as xN[true][2]);
    assert_eq(u2:0b01, u1:1 as xN[false][2]);

    assert_eq(s2:0b11, xN[true][1]:-1 as s2);
    assert_eq(u2:0b01, xN[false][1]:1 as u2);
}

fn p<S: bool, N: u32>(x: bits[N]) -> xN[S][N] { x as xN[S][N] }

fn q<S: bool, O: u32, N: u32>(x: sN[N]) -> xN[S][O] { x as xN[S][O] }

// Create a main entry point with a bunch of conversions so we can test IR conversions as well.
fn main() -> (u1, u2, u3, s1, s2, s3, s4, s4, s4) {
    let a = p<false>(u1::MAX);
    let b = p<false>(u2::MAX);
    let c = p<false>(u3::MAX);
    let d = p<true>(u1::MAX);
    let e = p<true>(u2::MAX);
    let f = p<true>(u3::MAX);
    let g = q<true, u32:4>(s1:0b1);
    let h = q<true, u32:4>(s2:0b11);
    let i = q<true, u32:4>(s3:0b111);
    (a, b, c, d, e, f, g, h, i)
}

#[test]
fn test_parametric_conversion() {
    let (a, b, c, d, e, f, g, h, i) = main();
    assert_eq(a, u1:0b1);
    assert_eq(b, u2:0b11);
    assert_eq(c, u3:0b111);
    assert_eq(d, s1:0b1);
    assert_eq(e, s2:0b11);
    assert_eq(f, s3:0b111);
    assert_eq(g, s4:0b1111);
    assert_eq(h, s4:0b1111);
    assert_eq(i, s4:0b1111);
}
