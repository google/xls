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

fn main(x: s8, y: s8) -> s8 {
  x - y
}

#[test]
fn main_test() {
  let x: s8 = s8:2;
  let y: s8 = s8:3;
  let z: s8 = main(x, y);
  let _ = assert_lt(z, s8:0);
  let _ = assert_eq(false, z >= s8:0);
  ()
}
