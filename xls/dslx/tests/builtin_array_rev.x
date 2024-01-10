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

import std;

fn do_rev_rev<N: u32>(x: uN[N]) -> bool {
    let a: bool[N] = std::convert_to_bools_lsb0(x);
    array_rev(array_rev(a)) == a
}

#[quickcheck]
fn quickcheck_rev_rev_is_orig2(x: u2) -> bool { do_rev_rev(x) }

#[quickcheck]
fn quickcheck_rev_rev_is_orig3(x: u3) -> bool { do_rev_rev(x) }

#[quickcheck]
fn quickcheck_rev_rev_is_orig4(x: u4) -> bool { do_rev_rev(x) }

#[quickcheck]
fn quickcheck_rev_rev_is_orig5(x: u5) -> bool { do_rev_rev(x) }

fn main(x: bool[3]) -> bool[3] { array_rev(x) }

#[test]
fn test_main() {
    assert_eq(main(bool[3]:[true, false, false]), bool[3]:[false, false, true]);
    assert_eq(main(bool[3]:[false, false, true]), bool[3]:[true, false, false]);
}
