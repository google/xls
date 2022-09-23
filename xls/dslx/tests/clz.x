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


fn main() -> u32 {
  let x0 = clz(u32:0x0005a000);
  let x1 = clz(x0);
  clz(x1)
}

#[test]
fn clz_test() {
  let _ = assert_eq(u3:0, clz(u3:0b111));
  let _ = assert_eq(u3:1, clz(u3:0b011));
  let _ = assert_eq(u3:2, clz(u3:0b001));
  let _ = assert_eq(u3:3, clz(u3:0b000));
  ()
}

