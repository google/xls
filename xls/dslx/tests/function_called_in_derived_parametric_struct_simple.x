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

fn double(x: u32) -> u32 { x * u32:2 }

struct MyParametric<A: u32, B: u32 = double(A)> {
  x: bits[A],
  y: bits[B]
}

// TODO(leary): 2020-12-19 This doesn't work, we have to annotate B as well.
// We should be able to infer it.
// fn f() -> MyParametric<u32:8> {
fn main() -> MyParametric<u32:8, u32:16> {
  MyParametric { x: u8:1, y: u16:2 }
}

#[test]
fn test_main() {
  assert_eq(MyParametric { x: u8:1, y: u16:2 }, main())
}
