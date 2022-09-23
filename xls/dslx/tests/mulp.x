// Copyright 2022 The XLS Authors
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

fn max(x: u32, y: u32) -> u32 {
    if (x > y) {x} else {y}
}

fn smulp_with_add<M: u32, N: u32, O: u32 = max(M, N)>(x: sN[M], y: sN[N]) -> sN[O] {
    let partial_product = smulp(x as sN[O], y as sN[O]);
    partial_product.0 + partial_product.1
}

fn umulp_with_add<M: u32, N: u32, O: u32 = max(M, N)>(x: uN[M], y: uN[N]) -> uN[O] {
    let partial_product = umulp(x as uN[O], y as uN[O]);
    partial_product.0 + partial_product.1
}

#[test]
fn smulp_examples() {
    let _ = assert_eq(s10: 15, smulp_with_add<u32:10, u32:10>(s10: 5, s10: 3));
    let _ = assert_eq(s12: 45, smulp_with_add(s12: 15, s12: 3));
    let _ = assert_eq(s12: -45, smulp_with_add(s12: 15, s10: -3));
}

#[test]
fn umulp_examples() {
    let _ = assert_eq(u10: 15, umulp_with_add<u32:10, u32:10>(u10: 5, u10: 3));
    let _ = assert_eq(u12: 45, umulp_with_add(u12: 15, u12: 3));
    let _ = assert_eq(u12: 75, umulp_with_add(u12: 15, u10: 5));
}
