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

fn double(value: u32) -> u32 {
  u32: 2 * value
}

fn triple(value: u32) -> u32 {
  u32: 3 * value
}

fn main(value: u32) -> u32 {
  double(value) + triple(value)
}

#![test]
fn test_main_value() {
  let _ = assert_eq(double(u32: 2), u32:4);
  let _ = assert_eq(triple(u32: 2), u32:6);
  let _ = assert_eq(main(u32: 2), u32:10);
  ()
}