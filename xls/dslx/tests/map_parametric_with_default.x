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

fn p<N: u32>(x: bits[N]) -> bits[N] { x << 0 }

fn main(xs: u4[1]) -> u4[1] { map(xs, p) }

#[test]
fn test_main() {
  let xs = u4[1]:[0x7];
  assert_eq(main(xs), xs)
}
