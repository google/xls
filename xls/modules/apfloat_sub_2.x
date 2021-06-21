// Copyright 2021 The XLS Authors
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

// This file implements most of IEEE
// floating-point subtraction (and comparisons that are
// implemented using subtraction), with the following exceptions:
//  - Both input and output denormals are treated as/flushed to 0.
//  - Only round-to-nearest mode is supported.
//  - No exception flags are raised/reported.
// In all other cases, results should be identical to other
// conforming implementations (modulo exact significand
// values in the NaN case.
import apfloat
import xls.modules.apfloat_add_2

type APFloat = apfloat::APFloat;

// Usage:
//  - EXP_SZ: The number of bits in the exponent.
//  - SFD_SZ: The number of bits in the significand (see
//        https://en.wikipedia.org/wiki/Significand for "significand"
//        vs "mantissa" naming).
//  - x, y: The two floating-point numbers to subract.
pub fn apfloat_sub_2<EXP_SZ: u32, SFD_SZ: u32>(
    x: APFloat<EXP_SZ, SFD_SZ>,
    y: APFloat<EXP_SZ, SFD_SZ>) -> APFloat<EXP_SZ, SFD_SZ> {
  let y = APFloat<EXP_SZ, SFD_SZ>{sign: !y.sign, bexp: y.bexp, sfd: y.sfd};
  apfloat_add_2::add(x, y)
}

// apfloat_add_2 is thoroughly tested elsewhere
// and fpsub_2x64 is a trivial modification of
// apfloat_add_2, so a few simple tests is sufficient.
#![test]
fn test_apfloat_sub_2() {
  let one = apfloat::one<u32:8, u32:23>(u1:0);
  let two = apfloat_add_2::add<u32:8, u32:23>(one, one);
  let neg_two = APFloat<u32:8, u32:23>{sign: u1:1, ..two};
  let three = apfloat_add_2::add<u32:8, u32:23>(one, two);
  let four = apfloat_add_2::add<u32:8, u32:23>(two, two);

  let _ = assert_eq(apfloat_sub_2(four, one), three);
  let _ = assert_eq(apfloat_sub_2(four, two), two);
  let _ = assert_eq(apfloat_sub_2(four, three), one);
  let _ = assert_eq(apfloat_sub_2(three, two), one);
  let _ = assert_eq(apfloat_sub_2(two, four), neg_two);
  ()
}

