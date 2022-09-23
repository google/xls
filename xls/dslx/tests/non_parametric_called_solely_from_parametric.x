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

// Note: double is only called from the body of a parametric.
fn double(x: u8) -> u8 { x * u8:2 }

fn p<N: u32>(x: bits[N]) -> bits[N] {
  double(x)
}

fn main(x: u8) -> u8 {
  p(x)
}

#[test]
fn main_test() {
  assert_eq(main(u8:0x10), u8:0x20)
}
