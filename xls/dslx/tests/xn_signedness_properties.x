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

fn to_xn<S: bool, N: u32>(x: uN[N]) -> xN[S][N] { x as xN[S][N] }

fn increment_signed_xn_works(x: u4) -> bool {
    let sn = to_xn<true>(x);
    let one = to_xn<true>(u4:1);
    let incremented = sn + one;
    incremented > sn || sn == s4::MAX
}

fn main() -> bool[3] {
    [
        increment_signed_xn_works(u4:0xf), increment_signed_xn_works(u4:0),
        increment_signed_xn_works(u4:7),
    ]
}

#[test]
fn test_increment_signed_xn_works() { assert_eq(main(), [true, true, true]); }
